"""Synthetic tests: scripts/score_cuda_mix_vs_g15.py (EXP-P48-001 Stage B, Mac CPU).

Small synthetic frozen world (3 games per split, 5 seeds, 10 classes): frequency baselines, G0-global Cpanel and
whole-MLB members, flatten direct heads, an EXP-P11-001-style analysis manifest, G0 bundle / G15 config / recorded
EXP-P16-001 results with sha pins, and a fake 15-task CUDA family.  Covers key alignment and permutation, the MIX_FIT
grid weight (known optimum, blend split only), the single-primary / exploratory-secondary N plumbing, the pinned
anchor chains of every frozen input (resealed or seed-swapped whole-MLB G0 members are refused before any candidate is
read), the reproduction gate incl. per-seed G15 - G0 deltas, the written outputs and the refusal to write into a
repository results/ directory.  No real data, no SSD, no GPU.  Review round 1: REVIEW-EXP-P48-001-ASTRA-001.
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
from pitchmdp.matrix_metrics import pitch_losses  # noqa: E402

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


def set_path(value, path, leaf):
    for key in path[:-1]:
        value = value.setdefault(key, {})
    value[path[-1]] = leaf


def reseal(dest, name):
    """Rewrite a member state's artifact_hashes from the files now on disk (identity kept): a consistent replacement."""
    value = json.loads((dest / name).read_text())
    value['artifact_hashes'] = {f: rc.sha256_file(dest / f) for f in value['artifact_hashes']}
    return dump(dest / name, value)


def manifest_inputs(root, rels):
    """{absolute path: sha256} of every file under the given runs-root paths, like the P11 manifest `inputs`."""
    out = {}
    for rel in rels:
        path = root / rel
        for one in ([path] if path.is_file() else sorted(q for q in path.iterdir() if q.is_file())):
            out[str(one)] = rc.sha256_file(one)
    return out


def write_manifest(root, inputs):
    path = root / reg.STAGE_B_RULES['inputs']['p11_manifest']['path']
    return path, dump(path, {'inputs': inputs, 'results_sha256': 'r', 'predictions_sha256': 'p', 'scoring_sources': {}})


def frozen_world(tmp_path, gate_shift=0.):
    """Frozen inputs under tmp/<runs root name>, the P11 analysis manifest, G0 bundle, G15 config and recorded results;
    returns paths, the bundle dicts and the rules pinned to them."""
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
    # EXP-P11-001 analysis manifest (as scripts/score_ml_g0_whole.py records it): baselines, the G0 Cpanel members and
    # every whole-MLB G0 member's state and artifacts; the G0 bundle pins its sha256 (`p11_manifest`).
    manifest_path, manifest_sha = write_manifest(root, manifest_inputs(root, [
        spec['cpanel_baseline'], spec['mlb_baseline'], *spec['g0_cpanel'].values(), *spec['g0_mlb'].values()]))
    g0_files['p11_manifest'] = {'path': str(manifest_path), 'sha256': manifest_sha}
    g0_bundle = {'members': {str(x): {'model_sha256': model_sha(f'g0/{x}')} for x in reg.SEEDS}, 'files': g0_files}
    g0_path = tmp_path / 'configs' / 'G0.json'
    g0_sha = dump(g0_path, g0_bundle)
    p16_path = tmp_path / 'results-frozen' / 'EXP-P16-001.json'
    g15 = {'definition': {'arm': MIX, 'direct_weight': .5}, 'g0_component': {'bundle_sha256': g0_sha},
           'members': {str(x): {'model_sha256': model_sha(f'flat/{x}')} for x in reg.SEEDS},
           'files': {**g15_files, 'results_p16': {'path': str(p16_path), 'sha256': None}}}
    rules = copy.deepcopy(reg.STAGE_B_RULES)
    cpanel, mlb, g0, g0_mlb, flatten, flatten_mlb = s.load_frozen(rules, root, g15, g0_bundle, {})
    ref = s.g15_reference(flatten, g0, cpanel, flatten_mlb, g0_mlb, mlb['mlb_dev'], mlb['mlb_dev_y'], .5, s.helpers())
    gate, recorded = rules['reproduction_gate'], {}
    for name, path in {**gate['fields'], **gate['seed_delta_fields']}.items():
        set_path(recorded, path, ref['recomputed'][name])
    recorded['cpanel_dev']['reports'][MIX]['primary']['log_loss'] += gate_shift
    g15['files']['results_p16']['sha256'] = dump(p16_path, recorded)
    g15_path = tmp_path / 'configs' / 'G15.json'
    rules['reference'].update(config_sha256=dump(g15_path, g15), g0_bundle_sha256=g0_sha,
                              recorded_results_sha256=g15['files']['results_p16']['sha256'])
    return {'root': root, 'g15': g15_path, 'g0': g0_path, 'p16': p16_path, 'rules': rules, 'g15_dict': g15,
            'g0_dict': g0_bundle, 'recorded': recorded}


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


def test_n_decision_single_primary_unadjusted_and_holm_across_three_secondary_rows():
    def row(delta, p, seeds):
        return {'paired': {'status': 'measured', 'nll': {'delta': delta, 'ci95': [delta - .001, delta + .001],
                                                         'p_less': p},
                           'brier': {'delta': -.001, 'ci95': [-.002, 0.], 'p_less': .01}}, 'seed_deltas': seeds}
    good = [-.01] * 5
    primary, secondary = s.test_rows(reg.STAGE_B_RULES)
    assert (primary, secondary) == (reg.PRIMARY_ROW, reg.SECONDARY_ROWS) == (
        'tabm_residual_counts/MIX_FIT',
        ['periodic_residual_counts/MIX_FIXED', 'periodic_residual_counts/MIX_FIT', 'tabm_residual_counts/MIX_FIXED'])
    rows = {primary: row(-.01, .04, good),
            'periodic_residual_counts/MIX_FIXED': row(-.01, .02, good),
            'periodic_residual_counts/MIX_FIT': row(-.01, .02, good),
            'tabm_residual_counts/MIX_FIXED': row(-.01, .001, [-.01, -.01, -.01, .01, .01]),
            'control/MIX_FIXED': row(0., .5, [0.] * 5)}
    out = s.decide(rows, primary, secondary, reg.STAGE_B_RULES['N'])
    # one confirmatory test: its own unadjusted p (.04 passes at .05; a four-test Holm on this fixture would have made it .06)
    assert out[primary]['p_decision'] == .04 and out[primary]['N']['status'] == 'predictive_improvement'
    assert out[primary]['p_adjustment'].startswith('none')
    # secondary: Holm across the three -> .001 x 3 = .003, .02 x 2 = .04, max(.04, .02) = .04
    assert out['tabm_residual_counts/MIX_FIXED']['p_decision'] == pytest.approx(.003)
    assert out['periodic_residual_counts/MIX_FIXED']['p_decision'] == pytest.approx(.04)
    assert out['periodic_residual_counts/MIX_FIT']['p_decision'] == pytest.approx(.04)
    assert all(out[n]['p_adjustment'].startswith('holm across the 3 secondary') for n in secondary)
    assert out['tabm_residual_counts/MIX_FIXED']['N']['status'] == 'inconclusive'  # 3/5 seeds < final-stage 4/5
    assert out['control/MIX_FIXED']['N'] is None and out['control/MIX_FIXED']['p_decision'] is None


def recomputed_values(shift=0., seed_shift=0., short=False):
    seeds = [-.001, -.002, -.003, -.004, -.005]
    return {'g15_cpanel_dev_after_blend_nll': 1.25 + shift, 'g15_whole_mlb_dev_blended_nll': 1.4,
            'g0_cpanel_dev_after_blend_nll': 1.5, 'g0_whole_mlb_dev_blended_nll': 1.45,
            'g15_minus_g0_cpanel_dev_after_blend_seed_deltas': seeds[:4] if short else seeds,
            'g15_minus_g0_whole_mlb_dev_blended_seed_deltas': [x + seed_shift * (i == 2) for i, x in enumerate(seeds)]}


def test_reproduction_gate_compares_scalars_and_per_seed_deltas():
    gate, recorded = reg.STAGE_B_RULES['reproduction_gate'], {}
    for name, path in {**gate['fields'], **gate['seed_delta_fields']}.items():
        set_path(recorded, path, recomputed_values()[name])
    result = s.reproduction_gate({'recomputed': recomputed_values()}, recorded, gate)
    assert result['passed'] and all(r['abs_diff'] == 0 for r in result['fields'].values())
    assert s.reproduction_gate({'recomputed': recomputed_values(shift=5e-10)}, recorded, gate)['passed']
    for bad, match in ((recomputed_values(shift=2e-9), 'g15_cpanel_dev_after_blend_nll'),
                       (recomputed_values(seed_shift=2e-9), 'g15_minus_g0_whole_mlb_dev_blended_seed_deltas'),
                       (recomputed_values(short=True), 'g15_minus_g0_cpanel_dev_after_blend_seed_deltas')):
        with pytest.raises(ValueError, match='Reproduction gate failed.*' + match):
            s.reproduction_gate({'recomputed': bad}, recorded, gate)
    with pytest.raises(ValueError, match='fields differ'):
        s.reproduction_gate({'recomputed': {k: v for k, v in recomputed_values().items() if 'seed' not in k}},
                            recorded, gate)


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
    gate = reg.STAGE_B_RULES['reproduction_gate']
    assert result['reproduction_gate']['passed'] and set(result['reproduction_gate']['fields']) == \
        set(gate['fields']) | set(gate['seed_delta_fields'])
    names = [f'{a}/{r}' for a in reg.ARM_ORDER for r in (reg.MIX_FIXED, reg.MIX_FIT)]
    primary, secondary = reg.PRIMARY_ROW, reg.SECONDARY_ROWS
    assert result['multiplicity']['primary'] == primary and result['multiplicity']['secondary'] == secondary
    for split in (s.CPANEL, s.MLB):
        rows = result[split]['rows']
        assert list(rows) == names and result[split]['games'] == 3
        for name in names:
            row = rows[name]
            assert len(row['seed_deltas']) == 5
            assert (row['paired']['draws'], row['paired']['seed']) == (10000, 20260924)
            if name == primary:
                assert row['role'] == 'primary' and row['p_decision'] == row['paired']['nll']['p_less']
            elif name in secondary:
                assert row['role'] == 'secondary_exploratory' and row['p_decision'] >= row['paired']['nll']['p_less']
            else:
                assert row['N'] is None and row['p_decision'] is None and row['role'].endswith('descriptive')
        sanity = rows['control/MIX_FIXED']['paired']  # control head == G15 flatten head -> identical mix
        assert sanity['nll']['delta'] == 0 and sanity['nll']['ci95'] == [0, 0] and sanity['brier']['delta'] == 0
        assert rows[primary]['N']['status'] == s.PASS
        assert rows['tabm_residual_counts/MIX_FIXED']['N']['status'] == s.PASS
        assert rows['periodic_residual_counts/MIX_FIXED']['N']['status'] != s.PASS
    conf = result['confirmation']
    assert conf['primary'] == primary and conf['primary_status'] == 'confirmed'
    assert conf['splits'] == {s.CPANEL: s.PASS, s.MLB: s.PASS} and 'intersection-union' in conf['test']
    explored = result['secondary_exploratory']['rows']
    assert list(explored) == secondary and all(r['status'] == 'exploratory' for r in explored.values())
    assert explored['tabm_residual_counts/MIX_FIXED']['both_splits_predictive_improvement'] is True  # still exploratory
    assert explored['periodic_residual_counts/MIX_FIXED']['both_splits_predictive_improvement'] is False
    assert 'confirmed' not in json.dumps(result['secondary_exploratory'])
    assert result['pipeline_difference']['row'] == 'control/MIX_FIXED'
    assert result['pipeline_difference'][s.CPANEL]['nll']['delta'] == 0
    assert 'cap 30' in result['pipeline_difference']['note']
    weights = result['mix_weights']
    assert weights['tabm_residual_counts'][reg.MIX_FIXED]['ensemble'] == .5
    assert all(w in reg.FIT_GRID for w in weights['tabm_residual_counts'][reg.MIX_FIT]['seeds'])
    heads = result['descriptive_unblended_head_vs_g15_head']
    assert heads['cpanel_dev']['control']['paired']['nll']['delta'] == 0
    assert heads['whole_mlb_dev']['tabm_residual_counts']['paired']['nll']['delta'] < 0
    assert result['robustness_R']['status'] == 'not_computed' and 'score_ml_matrix' in result['robustness_R']['reason']
    assert result['alignment'] == {**result['alignment'], 'permuted': [], 'checked': 45}
    assert any('ceiling of this direct-head pipeline on the exposed 2025 DEV' in x for x in result['limits'])
    assert any('cap 30 on MPS' in x for x in result['limits'])
    prov = result['provenance']
    assert prov['config']['sha256'] == rc.sha256_file(world['config'])
    for path in (world['g15'], world['g0'], world['p16'], world['root'] / 'EXP-P4-001/baseline_predictions.npz',
                 world['root'] / 'EXP-P16-001/4/mlb_predictions.npz',
                 world['root'] / 'EXP-P11-001/members/G0-global/seed3/predictions.npz',
                 world['root'] / reg.STAGE_B_RULES['inputs']['p11_manifest']['path'],
                 rc.task_dir(family, 'tabm_residual_counts', 4, False) / 'mlb_predictions.npz'):
        assert prov['inputs_sha256'][str(Path(path).resolve())] == rc.sha256_file(path), path
    anchors = prov['frozen_anchors']
    assert anchors['chains'] == {'cpanel_baseline': 'g0_bundle', 'mlb_baseline': 'g0_bundle',
                                 'g0_cpanel': 'g0_bundle', 'g0_mlb': 'p11_manifest', 'flatten': 'g15'}
    files = anchors['files']
    for seed in reg.SEEDS:
        for name in ('prediction_state.json', 'predictions.npz'):
            one = files[f'EXP-P11-001/members/G0-global/seed{seed}/{name}']
            assert one['anchor'] == 'p11_manifest' and one['listed_in'] == ['p11_manifest']
    assert files['EXP-P4-001/members/G0-global/seed0/prediction_state.json']['listed_in'] == ['g0_bundle',
                                                                                              'p11_manifest']
    assert files['EXP-P16-001/3/mlb_prediction_state.json']['anchor'] == 'g15'
    assert files['EXP-P16-001/3/mlb_predictions.npz']['anchor'] == 'g15 -> state artifact_hashes'
    assert prov['git']['commit'] and prov['seconds'] > 0 and prov['peak_rss']['ru_maxrss'] > 0
    assert set(prov['family']['tasks']) == set(reg.ARM_ORDER)
    summary = (out / 'SUMMARY.md').read_text()
    assert primary in summary and 'confirmed' in summary and prov['config']['sha256'] in summary
    assert '보조 행 (탐색, confirmed 아님)' in summary and 'exploratory' in summary
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


# ------------------------------------------------------------------ review round 1: pinned chain of the P11 members

def forbid_candidates(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('a candidate was read or scored before the frozen chain / gate passed')
    for name in ('cuda_members', 'candidate', 'score_arrays', 'cuda_alignment'):
        monkeypatch.setattr(s, name, boom)


def p11_member(world, seed):
    return world['root'] / reg.STAGE_B_RULES['inputs']['g0_mlb'][str(seed)]


def test_resealed_p11_member_is_refused_before_any_candidate(world, tmp_path, monkeypatch):
    """Archive replaced AND adjacent state resealed consistently: the state no longer matches the pinned manifest."""
    family = cuda_family(tmp_path, world['cfg'])
    dest = p11_member(world, 2)
    save(dest / 'predictions.npz', part('mlb_dev', dirichlet('replacement', 'mlb_dev'), 2, prefix='dev'))
    reseal(dest, 'prediction_state.json')
    forbid_candidates(monkeypatch)
    with pytest.raises(ValueError, match='seed2/prediction_state.json: differs from the pinned p11_manifest record'):
        run(world, family, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_seed_array_swap_between_p11_members_is_refused_before_any_candidate(world, tmp_path, monkeypatch):
    """Seed-1 and seed-3 members exchanged two ways: (a) probability archives swapped with both states resealed
    (identities kept), (b) whole members (archive + original state) moved across the seed directories."""
    family = cuda_family(tmp_path, world['cfg'])
    one, three = p11_member(world, 1), p11_member(world, 3)
    names = ('predictions.npz', 'prediction_state.json')
    original = {(d, n): (d / n).read_bytes() for d in (one, three) for n in names}
    forbid_candidates(monkeypatch)
    (one / names[0]).write_bytes(original[three, names[0]])
    (three / names[0]).write_bytes(original[one, names[0]])
    for dest in (one, three):
        reseal(dest, names[1])
    with pytest.raises(ValueError, match='seed1/prediction_state.json: differs from the pinned p11_manifest record'):
        run(world, family, tmp_path / 'a')
    for name in names:
        (one / name).write_bytes(original[three, name])
        (three / name).write_bytes(original[one, name])
    with pytest.raises(ValueError, match='seed1/prediction_state.json: differs from the pinned p11_manifest record'):
        run(world, family, tmp_path / 'b')
    assert not (tmp_path / 'a').exists() and not (tmp_path / 'b').exists()


def test_in_memory_seed_swap_keeps_the_aggregate_but_fails_the_per_seed_gate(world):
    """The reviewer's probe: swapping two whole-MLB G0 members leaves every aggregate NLL unchanged, so only the
    registered per-seed G15 - G0 deltas can catch it; the gate raises before any head is requested."""
    rules = world['rules']
    cpanel, mlb, g0, g0_mlb, flatten, flatten_mlb = s.load_frozen(rules, world['root'], world['g15_dict'],
                                                                  world['g0_dict'], {})
    swapped = [g0_mlb[i] for i in (0, 3, 2, 1, 4)]
    tools = s.helpers()
    plain = s.g15_reference(flatten, g0, cpanel, flatten_mlb, g0_mlb, mlb['mlb_dev'], mlb['mlb_dev_y'], .5, tools)
    moved = s.g15_reference(flatten, g0, cpanel, flatten_mlb, swapped, mlb['mlb_dev'], mlb['mlb_dev_y'], .5, tools)
    for name in rules['reproduction_gate']['fields']:
        assert moved['recomputed'][name] == pytest.approx(plain['recomputed'][name], abs=1e-12), name
    seeds = 'g15_minus_g0_whole_mlb_dev_blended_seed_deltas'
    assert moved['recomputed'][seeds] != plain['recomputed'][seeds]

    def boom(arm):
        raise AssertionError('a head was requested before the reproduction gate')
    with pytest.raises(ValueError, match='Reproduction gate failed.*' + seeds):
        s.score_arrays(boom, flatten, g0, cpanel, boom, flatten_mlb, swapped, mlb, rules, world['recorded'], tools)


def rewrite_manifest(world, edit):
    """Rewrite the P11 manifest with `edit` applied to its inputs; re-pin it in an in-memory copy of the G0 bundle."""
    root, path = world['root'], world['root'] / reg.STAGE_B_RULES['inputs']['p11_manifest']['path']
    inputs = json.loads(path.read_text())['inputs']
    edit(inputs, root)
    _, digest = write_manifest(root, inputs)
    bundle = copy.deepcopy(world['g0_dict'])
    bundle['files']['p11_manifest']['sha256'] = digest
    return bundle


def test_every_link_of_the_p11_manifest_chain_fails_closed(world):
    load = lambda bundle: s.load_frozen(world['rules'], world['root'], world['g15_dict'], bundle, {})  # noqa: E731
    load(world['g0_dict'])  # the built chain is intact
    no_pin = copy.deepcopy(world['g0_dict'])
    del no_pin['files']['p11_manifest']
    with pytest.raises(ValueError, match='does not pin the EXP-P11-001 analysis manifest'):
        load(no_pin)
    moved = copy.deepcopy(world['g0_dict'])
    moved['files']['p11_manifest']['path'] = moved['files']['p11_manifest']['path'].replace('analysis', 'other')
    with pytest.raises(ValueError, match='does not pin the EXP-P11-001 analysis manifest'):
        load(moved)
    member = lambda seed, name: str(world['root'] / reg.STAGE_B_RULES['inputs']['g0_mlb'][str(seed)] / name)  # noqa
    cases = {'not reachable from the pinned p11_manifest chain': [
                 lambda i, r: i.pop(member(4, 'prediction_state.json')),
                 lambda i, r: i.pop(member(2, 'predictions.npz'))],
             'differs from the pinned p11_manifest record': [
                 lambda i, r: i.update({member(0, 'predictions.npz'): '0' * 64}),
                 lambda i, r: i.update({str(r / 'EXP-P4-001/members/G0-global/seed1/predictions.npz'): '0' * 64}),
                 lambda i, r: i.update({str(r / reg.STAGE_B_RULES['inputs']['mlb_baseline']): '0' * 64})]}
    for match, edits in cases.items():
        for edit in edits:
            with pytest.raises(ValueError, match=match):
                load(rewrite_manifest(world, edit))
    # the manifest bytes on disk now differ from the sha256 the original bundle pinned (and from any other pin)
    wrong_pin = copy.deepcopy(world['g0_dict'])
    wrong_pin['files']['p11_manifest']['sha256'] = 'f' * 64
    for bundle in (world['g0_dict'], wrong_pin):
        with pytest.raises(ValueError, match='analysis manifest differs from the G0 bundle pin'):
            load(bundle)


def test_anchor_rules_must_cover_every_frozen_input(world):
    rules = copy.deepcopy(world['rules'])
    del rules['inputs']['anchors']['g0_mlb']
    with pytest.raises(ValueError, match='anchors must name a pinned chain'):
        s.load_frozen(rules, world['root'], world['g15_dict'], world['g0_dict'], {})
    rules = copy.deepcopy(world['rules'])
    rules['inputs']['anchors']['g0_mlb'] = {'source': 'nowhere', 'files': ['state']}
    with pytest.raises(ValueError, match='anchors must name a pinned chain'):
        s.load_frozen(rules, world['root'], world['g15_dict'], world['g0_dict'], {})
