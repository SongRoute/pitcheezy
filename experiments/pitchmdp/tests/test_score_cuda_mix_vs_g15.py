"""Synthetic tests: scripts/score_cuda_mix_vs_g15.py (EXP-P48-001 Stage B, Mac CPU).

Small synthetic frozen world (3 games per split, 5 seeds, 10 classes): frequency baselines, G0-global Cpanel and
whole-MLB members, flatten direct heads, G0 bundle / G15 config / recorded EXP-P16-001 results with sha pins, and a
fake 15-task CUDA family.  Covers key alignment and permutation, the MIX_FIT grid weight (known optimum, blend split
only), N / Holm plumbing, the reproduction gate (raises before any candidate is read), the written outputs and the
refusal to write into a repository results/ directory.  No real data, no SSD, no GPU.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).resolve().parent)]
import run_cuda_direct as rc  # noqa: E402
import score_cuda_mix_vs_g15 as s  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  fake CUDA task writer
from pitchmdp import direct_adoption_confirm_registry as reg  # noqa: E402
from pitchmdp.matrix_metrics import pitch_losses, prediction_metrics  # noqa: E402

RUN = REPO / 'configs' / 'EXP-P48-001.json'
ROOT_NAME = reg.STAGE_B_RULES['inputs']['runs_root_name']
SIZES = {'blend': 30, 'dev': 36, 'mlb_dev': 45}
GAMES = {'blend': 101, 'dev': 201, 'mlb_dev': 301}
MIX = reg.REFERENCE_ARM


def meta_rows(split):
    rng = np.random.default_rng(zlib.crc32(split.encode()))
    n = SIZES[split]
    i = np.arange(n)
    game = (GAMES[split] + i * 3 // n).astype(np.int64)  # three games per split
    return {'keys': np.column_stack([game, i // 4 + 1, i % 4 + 1]).astype(np.int64),
            'y': rng.integers(0, 10, n).astype(np.int64), 'game_pk': game,
            'pitcher': rng.integers(1, 6, n).astype(np.int64)}


META = {split: meta_rows(split) for split in SIZES}


def dirichlet(tag, split):
    return np.random.default_rng(zlib.crc32(tag.encode())).dirichlet(np.ones(10), SIZES[split])


def sharper(p, split):
    return (p + np.eye(10)[META[split]['y']]) / 2


def flat(seed, split):
    return dirichlet(f'flatten/{seed}/{split}', split)


def head(arm, seed, split):
    """CUDA heads: control == the G15 flatten head (zero sanity delta), periodic random, tabm sharper."""
    if arm == 'control':
        return flat(seed, split)
    if arm == 'tabm_residual_counts':
        return sharper(flat(seed, split), split)
    return dirichlet(f'{arm}/{seed}/{split}', split)


def part(split, p, level=None, prefix=None, meta=None):
    prefix, meta = prefix or split, meta or META[split]
    out = {prefix: p, prefix + '_raw': p, prefix + '_keys': meta['keys'], prefix + '_y': meta['y'],
           prefix + '_game_pk': meta['game_pk'], prefix + '_pitcher': meta['pitcher']}
    if level is not None:
        out[prefix + '_delivery_level'] = np.full(len(p), level, dtype=np.int64)
    return out


def save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return rc.sha256_file(path)


def state(dest, name, identity, files, dependencies=None):
    value = {'identity': identity, 'artifact_hashes': {f: rc.sha256_file(dest / f) for f in files}}
    if dependencies:
        value['dependencies'] = dependencies
    return dump(dest / name, value)


def model_sha(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def frozen_world(tmp_path, gate_shift=0.):
    """Frozen inputs under tmp/<runs root name>, G0 bundle, G15 config and recorded results; returns paths + rules."""
    root, spec = tmp_path / ROOT_NAME, reg.STAGE_B_RULES['inputs']
    save(root / spec['cpanel_baseline'], {**part('blend', dirichlet('freq', 'blend')),
                                          **part('dev', dirichlet('freq', 'dev'))})
    save(root / spec['mlb_baseline'], part('mlb_dev', dirichlet('freq', 'mlb_dev')))
    g0_files, g15_files = {}, {}
    for seed in reg.SEEDS:
        k = str(seed)
        dest = root / spec['g0_cpanel'][k]
        save(dest / 'predictions.npz', {**part('blend', dirichlet(f'g0/{k}', 'blend'), 1),
                                        **part('dev', dirichlet(f'g0/{k}', 'dev'), 3)})
        for name in ('calibration.json', 'prediction_runtime.json'):
            dump(dest / name, {'seed': seed})
        g0_files[f'seed{k}_prediction_state'] = {'path': str(dest / 'prediction_state.json'), 'sha256': state(
            dest, 'prediction_state.json', {'cell': 'G0-global', 'seed': seed, 'preparation_sha256': 'p'},
            ['predictions.npz', 'calibration.json', 'prediction_runtime.json'],
            {str(dest / 'model.pt'): model_sha(f'g0/{k}')})}
        dest = root / spec['g0_mlb'][k]
        save(dest / 'predictions.npz', part('mlb_dev', dirichlet(f'g0/{k}', 'mlb_dev'), 2, prefix='dev'))
        dump(dest / 'prediction_runtime.json', {'seed': seed})
        state(dest, 'prediction_state.json', {'cell': 'G0-global', 'seed': seed, 'model_sha256': model_sha(f'g0/{k}')},
              ['predictions.npz', 'prediction_runtime.json'])
        dest = root / spec['flatten'][k]
        save(dest / 'predictions.npz', {**part('blend', flat(seed, 'blend'), -2), **part('dev', flat(seed, 'dev'), -2)})
        save(dest / 'mlb_predictions.npz', part('mlb_dev', flat(seed, 'mlb_dev'), -2))
        for name in ('calibration.json', 'prediction_runtime.json', 'mlb_prediction_runtime.json'):
            dump(dest / name, {'seed': seed})
        identity = {'preparation_sha256': 'p', 'seed': seed, 'smoke': False, 'model_sha256': model_sha(f'flat/{k}')}
        for name, files in (('prediction_state.json',
                             ['predictions.npz', 'prediction_runtime.json', 'calibration.json']),
                            ('mlb_prediction_state.json', ['mlb_predictions.npz', 'mlb_prediction_runtime.json'])):
            g15_files[f'seed{k}_{name[:-5]}'] = {'path': str(dest / name), 'sha256': state(dest, name, identity, files)}
    for key, rel in (('p4_frequency', spec['cpanel_baseline']), ('p11_frequency', spec['mlb_baseline'])):
        g0_files[key] = {'path': str(root / rel), 'sha256': rc.sha256_file(root / rel)}
    g0_bundle = {'members': {str(x): {'model_sha256': model_sha(f'g0/{x}')} for x in reg.SEEDS}, 'files': g0_files}
    g0_path = tmp_path / 'configs' / 'G0.json'
    g0_sha = dump(g0_path, g0_bundle)
    p16_path = tmp_path / 'results-frozen' / 'EXP-P16-001.json'
    g15 = {'definition': {'arm': MIX, 'direct_weight': .5}, 'g0_component': {'bundle_sha256': g0_sha},
           'members': {str(x): {'model_sha256': model_sha(f'flat/{x}')} for x in reg.SEEDS},
           'files': {**g15_files, 'results_p16': {'path': str(p16_path), 'sha256': None}}}
    rules = copy.deepcopy(reg.STAGE_B_RULES)
    cpanel, mlb, g0, g0_mlb, flatten, flatten_mlb = s.load_frozen(rules, root, g15, g0_bundle, {})
    ref = s.g15_reference(flatten, g0, cpanel, flatten_mlb, g0_mlb, mlb['mlb_dev'], .5, s.helpers())
    y = mlb['mlb_dev_y']
    g15_nll, g0_nll = ref['report']['primary']['log_loss'] + gate_shift, ref['g0_report']['primary']['log_loss']
    recorded = {'cpanel_dev': {'reports': {MIX: {'primary': {'log_loss': g15_nll}},
                                           'G0-global': {'primary': {'log_loss': g0_nll}}}},
                'mlb_dev_blended': {'comparison': {MIX: {
                    'candidate': {'log_loss': prediction_metrics(y, ref['mlb_blended'])['log_loss']},
                    'control': {'log_loss': prediction_metrics(y, ref['g0_mlb_blended'])['log_loss']}}}}}
    g15['files']['results_p16']['sha256'] = dump(p16_path, recorded)
    g15_path = tmp_path / 'configs' / 'G15.json'
    rules['reference'].update(config_sha256=dump(g15_path, g15), g0_bundle_sha256=g0_sha,
                              recorded_results_sha256=g15['files']['results_p16']['sha256'])
    return {'root': root, 'g15': g15_path, 'g0': g0_path, 'p16': p16_path, 'rules': rules}


def cuda_family(tmp_path, cfg, permute=(), broken=None, name='fam'):
    """Fake 15-task CUDA family; `permute` arms write reversed rows, `broken` (arm) loses one dev key."""
    fam, digest, sources = tmp_path / name, cfg['export']['manifest_sha256'], rc.own_hashes()
    for arm in reg.ARM_ORDER:
        for seed in cfg['seeds']:
            meta, probs = {}, {}
            for split in SIZES:
                order = np.arange(SIZES[split])[::-1] if arm in permute else np.arange(SIZES[split])
                meta[split] = {k: v[order].copy() for k, v in META[split].items()}
                probs[split] = head(arm, seed, split)[order]
            if arm == broken:
                meta['dev']['keys'][0] = [999, 9, 9]
            dest = tf.fake_task(fam, cfg, digest, arm, seed, meta, sources, probs=probs)
            fit = json.loads((dest / 'fit.json').read_text())
            fit['report'] = {'epochs_run': 20, 'best_epoch': 15}
            fit['epoch_cap'] = rc.epoch_record(fit['report'], 60, 5)
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest)
    return fam


@pytest.fixture
def world(tmp_path, monkeypatch):
    built = frozen_world(tmp_path)
    monkeypatch.setattr(reg, 'STAGE_B_RULES', built['rules'])
    cfg = json.loads(RUN.read_text())
    cfg.update(status='REGISTERED', execution={'enabled': True})
    cfg['comparison'][reg.STAGE_B_KEY]['rules'] = built['rules']
    built['config'] = tmp_path / 'configs' / 'EXP-P48-001.json'
    dump(built['config'], cfg)
    built['cfg'] = rc.config_check(json.loads(built['config'].read_text()))
    return built


def run(world, family, dest, **kw):
    return s.score(world['config'], family, world['g15'], world['root'], dest, g0_bundle=world['g0'],
                   recorded_results=world['p16'], **kw)


def results_snapshot():
    root = REPO / 'results'
    return sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in root.iterdir()) if root.is_dir() else []


# ------------------------------------------------------------------ pure helpers

def test_helpers_are_the_p16_scoring_functions_and_constants_match():
    import run_ml_direct_head as base
    import score_ml_matrix
    assert s.helpers() == (score_ml_matrix.assert_aligned, score_ml_matrix.summarize_cell, base.mixed, base.paired)
    assert reg.REFERENCE_ARM == base.MIX and reg.G0_CELL == base.CELL
    rules = s.stage_b_rules(json.loads(RUN.read_text()))
    assert rules == reg.STAGE_B_RULES and rules['bootstrap'] == {'draws': 10000, 'seed': 20260924, 'unit': 'whole_game'}


def test_key_permutation_aligns_by_key_and_fails_closed():
    target = META['dev']['keys']
    assert s.key_permutation(target.copy(), target) is None
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(target))
    order = s.key_permutation(target[shuffled], target)
    assert np.array_equal(target[shuffled][order], target)
    missing = target[shuffled].copy()
    missing[3] = [999, 1, 1]
    with pytest.raises(ValueError, match='missing from the CUDA archive'):
        s.key_permutation(missing, target)
    with pytest.raises(ValueError, match='unique'):
        s.key_permutation(np.vstack([target[:-1], target[:1]]), target)
    with pytest.raises(ValueError, match='rows'):
        s.key_permutation(target[:-1], target)
    with pytest.raises(ValueError, match='unique'):
        s.key_permutation(target[:, :2], target)


def two_group_rows(n1=31, n2=24):
    """True-class probabilities a (head) / b (G0): group 1 .8/.2, group 2 .2/.6; the NLL optimum is exactly w = .7."""
    def rows(p_true, n):
        p = np.full((n, 10), (1 - p_true) / 9)
        p[:, 0] = p_true
        return p
    y = np.zeros(n1 + n2, dtype=np.int64)
    return y, np.vstack([rows(.8, n1), rows(.2, n2)]), np.vstack([rows(.2, n1), rows(.6, n2)])


def test_grid_weight_recovers_the_known_optimum_and_ties_go_low():
    y, h, g = two_group_rows()
    fit = s.fit_mix_weight(y, h, g, reg.FIT_GRID)
    assert fit['weight'] == .7 and fit['grid'] == reg.FIT_GRID and fit['rows'] == 55
    fine = np.linspace(0, 1, 1001)
    nll = [pitch_losses(y, w * h + (1 - w) * g)[:, 0].mean() for w in fine]
    assert abs(fine[int(np.argmin(nll))] - .7) < 1e-9
    assert fit['blend_nll_by_weight'][reg.FIT_GRID.index(.7)] == min(fit['blend_nll_by_weight'])
    assert s.fit_mix_weight(y, g, g, reg.FIT_GRID)['weight'] == .5  # all tied -> smallest grid weight
    assert s.fit_mix_weight(y, np.eye(10)[y] * .9 + .01, g, reg.FIT_GRID)['weight'] == .95  # boundary
    for grid in ([], [.6, .5], [.5, .5], [.5, 1.2]):
        with pytest.raises(ValueError, match='grid'):
            s.fit_mix_weight(y, h, g, grid)


def members(arrays_by_seed):
    return [{**{f'{p}{x}': META[p][x[1:]] for p in ('blend', 'dev') for x in ('_keys', '_y', '_game_pk', '_pitcher')},
             **{f'{p}{x}': arrays_by_seed(seed, p) for p in ('blend', 'dev') for x in ('', '_raw')}}
            for seed in reg.SEEDS]


def test_mix_fit_weights_use_the_blend_split_only(tmp_path):
    cpanel = {**part('blend', dirichlet('freq', 'blend')), **part('dev', dirichlet('freq', 'dev'))}
    g0 = members(lambda seed, p: dirichlet(f'g0/{seed}', p))
    tools, rules = s.helpers(), reg.STAGE_B_RULES
    head_mlb = [head('tabm_residual_counts', x, 'mlb_dev') for x in reg.SEEDS]
    g0_mlb = [dirichlet(f'g0/{x}', 'mlb_dev') for x in reg.SEEDS]
    freq = dirichlet('freq', 'mlb_dev')
    base = members(lambda seed, p: head('tabm_residual_counts', seed, p))
    got = s.candidate(base, g0, cpanel, head_mlb, g0_mlb, freq, reg.MIX_FIT, rules, tools)['weights']
    new_dev = members(lambda seed, p: head('tabm_residual_counts', seed, p) if p == 'blend'
                      else dirichlet(f'x/{seed}', p))
    same = s.candidate(new_dev, g0, cpanel, head_mlb, g0_mlb, freq, reg.MIX_FIT, rules, tools)['weights']
    assert same['seeds'] == got['seeds'] and same['ensemble'] == got['ensemble']
    new_blend = members(lambda seed, p: dirichlet(f'x/{seed}', p) if p == 'blend'
                        else head('tabm_residual_counts', seed, p))
    moved = s.candidate(new_blend, g0, cpanel, head_mlb, g0_mlb, freq, reg.MIX_FIT, rules, tools)['weights']
    assert moved['seeds'] != got['seeds']
    for x, fit in zip(reg.SEEDS, got['selection']['seeds']):
        expected = s.fit_mix_weight(cpanel['blend_y'], base[x]['blend'], g0[x]['blend'], reg.FIT_GRID)
        assert fit['weight'] == expected['weight']
    fixed = s.candidate(base, g0, cpanel, head_mlb, g0_mlb, freq, reg.MIX_FIXED, rules, tools)['weights']
    assert fixed == {'seeds': [.5] * 5, 'ensemble': .5, 'selection': None}


def test_n_decision_plumbing_holm_across_the_candidate_family_only():
    def row(delta, p, seeds):
        return {'paired': {'status': 'measured', 'nll': {'delta': delta, 'ci95': [delta - .001, delta + .001],
                                                         'p_less': p},
                           'brier': {'delta': -.001, 'ci95': [-.002, 0.], 'p_less': .01}}, 'seed_deltas': seeds}
    good = [-.01] * 5
    rows = {'periodic_residual_counts/MIX_FIXED': row(-.01, .02, good),
            'periodic_residual_counts/MIX_FIT': row(-.01, .02, good),
            'tabm_residual_counts/MIX_FIXED': row(-.01, .001, good),
            'tabm_residual_counts/MIX_FIT': row(-.01, .001, [-.01, -.01, -.01, .01, .01]),
            'control/MIX_FIXED': row(0., .5, [0.] * 5)}
    family = list(rows)[:4]
    out = s.decide(rows, family, reg.STAGE_B_RULES['N'])
    assert out['control/MIX_FIXED']['N'] is None
    assert out['periodic_residual_counts/MIX_FIXED']['N']['p_holm'] == pytest.approx(.04)  # 2 x .02 (Holm rank 3)
    assert out['periodic_residual_counts/MIX_FIXED']['N']['status'] == 'predictive_improvement'
    assert out['tabm_residual_counts/MIX_FIXED']['N']['p_holm'] == pytest.approx(.004)
    assert out['tabm_residual_counts/MIX_FIT']['N']['status'] == 'inconclusive'  # 3/5 seeds < final-stage 4/5
    assert not out['tabm_residual_counts/MIX_FIT']['N']['criteria']['seed_direction_stable']


def test_reproduction_gate_raises_on_mismatch_and_before_any_candidate():
    gate = reg.STAGE_B_RULES['reproduction_gate']
    y = META['mlb_dev']['y']
    p = dirichlet('a', 'mlb_dev')
    reference = {'report': {'primary': {'log_loss': 1.25}}, 'g0_report': {'primary': {'log_loss': 1.5}},
                 'mlb_blended': p, 'g0_mlb_blended': p}
    nll = prediction_metrics(y, p)['log_loss']
    recorded = {'cpanel_dev': {'reports': {MIX: {'primary': {'log_loss': 1.25}},
                                           'G0-global': {'primary': {'log_loss': 1.5}}}},
                'mlb_dev_blended': {'comparison': {MIX: {'candidate': {'log_loss': nll},
                                                         'control': {'log_loss': nll}}}}}
    result = s.reproduction_gate(reference, y, recorded, gate)
    assert result['passed'] and all(r['abs_diff'] == 0 for r in result['fields'].values())
    near = copy.deepcopy(recorded)
    near['cpanel_dev']['reports'][MIX]['primary']['log_loss'] += 5e-10
    assert s.reproduction_gate(reference, y, near, gate)['passed']
    far = copy.deepcopy(recorded)
    far['mlb_dev_blended']['comparison'][MIX]['candidate']['log_loss'] += 2e-9
    with pytest.raises(ValueError, match='Reproduction gate failed.*g15_whole_mlb_dev_blended_nll'):
        s.reproduction_gate(reference, y, far, gate)


def test_destination_refuses_repository_results_and_existing_dirs(tmp_path):
    before = results_snapshot()
    with pytest.raises(ValueError, match='results/ directory'):
        s.check_destination(REPO / 'results' / 'EXP-P48-001-stage-b')
    main_results = next(d for d in s.results_dirs() if d != (REPO / 'results').resolve()) \
        if len(s.results_dirs()) > 1 else REPO / 'results'
    with pytest.raises(ValueError, match='results/ directory'):
        s.check_destination(Path(main_results) / 'x' / 'y')
    (tmp_path / 'exists').mkdir()
    with pytest.raises(FileExistsError):
        s.check_destination(tmp_path / 'exists')
    assert s.check_destination(tmp_path / 'new') == (tmp_path / 'new').resolve()
    assert results_snapshot() == before and not (REPO / 'results' / 'EXP-P48-001-stage-b').exists()


# ------------------------------------------------------------------ end to end (synthetic frozen world)

def test_score_writes_outputs_with_gate_rows_confirmation_and_provenance(world, tmp_path):
    before = results_snapshot()
    family = cuda_family(tmp_path, world['cfg'])
    result = run(world, family, tmp_path / 'out')
    out = tmp_path / 'out'
    assert sorted(p.name for p in out.iterdir()) == ['SHA256SUMS', 'SUMMARY.md', 'results.json']
    assert all(not os.access(out / n, os.W_OK) for n in ('results.json', 'SUMMARY.md', 'SHA256SUMS'))
    assert json.loads((out / 'results.json').read_text())['confirmation'] == result['confirmation']
    sums = dict(line.split('  ')[::-1] for line in (out / 'SHA256SUMS').read_text().splitlines())
    assert sums == {n: rc.sha256_file(out / n) for n in ('results.json', 'SUMMARY.md')}
    assert results_snapshot() == before  # nothing written into the repository results/
    assert result['reproduction_gate']['passed'] and set(result['reproduction_gate']['fields']) == \
        set(reg.STAGE_B_RULES['reproduction_gate']['fields'])
    names = [f'{a}/{r}' for a in reg.ARM_ORDER for r in (reg.MIX_FIXED, reg.MIX_FIT)]
    family_names = [f'{a}/{r}' for a in ('periodic_residual_counts', 'tabm_residual_counts')
                    for r in (reg.MIX_FIXED, reg.MIX_FIT)]
    assert result['multiplicity']['family'] == family_names
    for split in (s.CPANEL, s.MLB):
        rows = result[split]['rows']
        assert list(rows) == names
        assert result[split]['games'] == 3
        for name in names:
            row = rows[name]
            assert len(row['seed_deltas']) == 5
            assert (row['paired']['draws'], row['paired']['seed']) == (10000, 20260924)
            if name in family_names:
                assert row['role'] == 'candidate' and row['N']['p_holm'] >= row['paired']['nll']['p_less']
            else:
                assert row['N'] is None and row['role'].endswith('descriptive')
        sanity = rows['control/MIX_FIXED']['paired']  # control head == G15 flatten head -> identical mix
        assert sanity['nll']['delta'] == 0 and sanity['nll']['ci95'] == [0, 0] and sanity['brier']['delta'] == 0
        assert rows['tabm_residual_counts/MIX_FIT']['N']['status'] == s.PASS
        assert rows['tabm_residual_counts/MIX_FIXED']['N']['status'] == s.PASS
        assert rows['periodic_residual_counts/MIX_FIXED']['N']['status'] != s.PASS
    conf = result['confirmation']
    assert conf['primary'] == 'tabm_residual_counts/MIX_FIT' and conf['primary_status'] == 'confirmed'
    assert conf['rows']['periodic_residual_counts/MIX_FIXED']['status'] == 'not_confirmed'
    assert conf['rows']['tabm_residual_counts/MIX_FIXED']['role'] == 'secondary'
    assert result['cross_device_reproduction'][s.CPANEL]['nll']['delta'] == 0
    weights = result['mix_weights']
    assert weights['tabm_residual_counts'][reg.MIX_FIXED]['ensemble'] == .5
    assert all(w in reg.FIT_GRID for w in weights['tabm_residual_counts'][reg.MIX_FIT]['seeds'])
    heads = result['descriptive_unblended_head_vs_g15_head']
    assert heads['cpanel_dev']['control']['paired']['nll']['delta'] == 0
    assert heads['whole_mlb_dev']['tabm_residual_counts']['paired']['nll']['delta'] < 0
    assert result['robustness_R']['status'] == 'not_computed' and 'score_ml_matrix' in result['robustness_R']['reason']
    assert result['alignment'] == {**result['alignment'], 'permuted': [], 'checked': 45}
    prov = result['provenance']
    assert prov['config']['sha256'] == rc.sha256_file(world['config'])
    for path in (world['g15'], world['g0'], world['p16'], world['root'] / 'EXP-P4-001/baseline_predictions.npz',
                 world['root'] / 'EXP-P16-001/4/mlb_predictions.npz',
                 rc.task_dir(family, 'tabm_residual_counts', 4, False) / 'mlb_predictions.npz'):
        assert prov['inputs_sha256'][str(Path(path).resolve())] == rc.sha256_file(path), path
    assert prov['git']['commit'] and prov['seconds'] > 0 and prov['peak_rss']['ru_maxrss'] > 0
    assert set(prov['family']['tasks']) == set(reg.ARM_ORDER)
    summary = (out / 'SUMMARY.md').read_text()
    assert 'tabm_residual_counts/MIX_FIT' in summary and 'confirmed' in summary and prov['config']['sha256'] in summary
    with pytest.raises(FileExistsError):
        run(world, family, tmp_path / 'out')


def test_permuted_cuda_rows_are_aligned_by_key_and_recorded(world, tmp_path):
    plain = run(world, cuda_family(tmp_path, world['cfg']), tmp_path / 'plain')
    family = cuda_family(tmp_path, world['cfg'], permute=('tabm_residual_counts',), name='perm')
    shuffled = run(world, family, tmp_path / 'perm-out')
    assert len(shuffled['alignment']['permuted']) == 15
    assert all(k.startswith('tabm_residual_counts/') for k in shuffled['alignment']['permuted'])
    for split in (s.CPANEL, s.MLB):
        for name, row in plain[split]['rows'].items():
            assert shuffled[split]['rows'][name]['paired']['nll'] == row['paired']['nll'], name
            assert shuffled[split]['rows'][name]['seed_deltas'] == row['seed_deltas'], name


def test_missing_cuda_key_fails_before_scoring(world, tmp_path):
    family = cuda_family(tmp_path, world['cfg'], broken='periodic_residual_counts')
    with pytest.raises(ValueError, match='missing from the CUDA archive'):
        run(world, family, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_gate_mismatch_aborts_before_any_candidate_is_read(tmp_path, monkeypatch):
    built = frozen_world(tmp_path, gate_shift=1e-6)
    monkeypatch.setattr(reg, 'STAGE_B_RULES', built['rules'])
    cfg = json.loads(RUN.read_text())
    cfg.update(status='REGISTERED', execution={'enabled': True})
    cfg['comparison'][reg.STAGE_B_KEY]['rules'] = built['rules']
    built['config'] = tmp_path / 'configs' / 'EXP-P48-001.json'
    dump(built['config'], cfg)
    family = cuda_family(tmp_path, rc.config_check(json.loads(built['config'].read_text())))

    def boom(*args, **kwargs):
        raise AssertionError('a candidate was read before the reproduction gate')
    monkeypatch.setattr(s, 'cuda_members', boom)
    monkeypatch.setattr(s, 'candidate', boom)
    with pytest.raises(ValueError, match='Reproduction gate failed.*g15_cpanel_dev_after_blend_nll'):
        run(built, family, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_frozen_input_tampering_and_draft_config_are_refused(world, tmp_path):
    family = cuda_family(tmp_path, world['cfg'])
    member = world['root'] / reg.STAGE_B_RULES['inputs']['g0_cpanel']['3'] / 'predictions.npz'
    original = member.read_bytes()
    member.write_bytes(original + b'x')
    with pytest.raises(ValueError, match='artifact identity changed'):
        run(world, family, tmp_path / 'a')
    member.write_bytes(original)
    draft = json.loads(world['config'].read_text())
    draft.update(status='DRAFT', execution={'enabled': False})
    path = tmp_path / 'configs' / 'draft.json'
    dump(path, draft)
    with pytest.raises(ValueError, match='REGISTERED'):
        s.score(path, family, world['g15'], world['root'], tmp_path / 'b', g0_bundle=world['g0'],
                recorded_results=world['p16'])
    with pytest.raises(ValueError, match='G15 config differs from its pin'):
        s.score(world['config'], family, world['g0'], world['root'], tmp_path / 'c', g0_bundle=world['g0'],
                recorded_results=world['p16'])
    with pytest.raises(ValueError, match='Incomplete family'):
        run(world, tmp_path / 'no-family', tmp_path / 'd')
    assert not any((tmp_path / n).exists() for n in 'abcd')


def test_cli_main_scores_the_synthetic_world(world, tmp_path, capsys):
    family = cuda_family(tmp_path, world['cfg'])
    s.main(['--config', str(world['config']), '--family', str(family), '--g15', str(world['g15']),
            '--runs-root', str(world['root']), '--destination', str(tmp_path / 'cli'), '--g0-bundle', str(world['g0']),
            '--recorded-results', str(world['p16'])])
    assert 'CUDA_MIX_VS_G15_SCORED' in capsys.readouterr().out
    assert json.loads((tmp_path / 'cli' / 'results.json').read_text())['experiment_id'] == 'EXP-P48-001'


def test_held_heavy_lock_fails_fast_before_reading_frozen_inputs(world, tmp_path, monkeypatch):
    import fcntl
    family = cuda_family(tmp_path, world['cfg'])

    def boom(*args, **kwargs):
        raise AssertionError('frozen inputs were read while another heavy job held the lock')
    monkeypatch.setattr(s, 'load_frozen', boom)
    with open(world['root'] / '.heavy.lock', 'a+b') as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another heavy job'):
            run(world, family, tmp_path / 'out')
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
    assert not (tmp_path / 'out').exists()
