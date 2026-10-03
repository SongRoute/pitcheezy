"""Synthetic tests: data-quantity / pitcher-information / combination arms in the CUDA runner and scorer (CPU only)."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export / fake-task helpers
from pitchmdp import direct_combo_registry as cr  # noqa: E402

COMBO = REPO / 'configs' / 'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json'
FRONTIER_RUN = REPO / 'configs' / 'DRAFT-CUDA-FRONTIER-RUN-20261004.json'
DATA = ('data25', 'data50', 'data75')
PITCHER = ('no_pitcher_profile', 'no_pitcher_support', 'no_pitcher_all')
WORKLOAD_ARMS = ('residual_counts', 'residual_block', 'periodic_counts', 'periodic_block')


def load(registered=False, unpin=True):
    return tf.load(COMBO, registered, unpin)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


# ------------------------------------------------------------------ config

BASE = ('counts', 'block', 'residual', 'periodic')
BASE_CONFIG = REPO / 'configs' / 'DRAFT-CUDA-DIRECT-20261004.json'


def test_config_is_draft_complete_factorial_plus_data_and_pitcher_arms():
    raw, frontier = json.loads(COMBO.read_text()), json.loads(FRONTIER_RUN.read_text())
    cfg = load(unpin=False)
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and cfg['fits'] == 45
    assert raw['experiment_id'].startswith('DRAFT-') and 'P27' not in raw['experiment_id']
    assert [a['name'] for a in cfg['arms']] == ['control', *BASE, *cr.ARM_NAMES] and cfg['arms'][0] == frontier['arms'][0]
    assert len(cfg['arms']) * len(cfg['seeds']) == cfg['fits'] and {a['priority'] for a in cfg['arms']} == {1}
    base = {a['name']: a for a in json.loads(BASE_CONFIG.read_text())['arms']}
    for name in BASE:  # exact DRAFT-CUDA-DIRECT specs (priority aside)
        assert {**rc.arm_spec(cfg, name), 'priority': 0} == {**base[name], 'priority': 0}
    cells = {(a['model'], tuple(a['workload'])) for a in cfg['arms'] if 'train_fraction' not in a and 'drop_context' not in a}
    assert cells == {(m, w) for m in ('flatten_mlp', 'residual_mlp', 'periodic_flatten_mlp')
                     for w in ((), tuple(cr.COUNTS), tuple(cr.BLOCK))}  # complete 3x3
    contrasts = sc.check_contrasts(cfg)
    assert [(c['candidate'], c['reference']) for c in contrasts] == [
        (j, ref) for j in WORKLOAD_ARMS for ref in (j.split('_')[0], j.split('_')[1])]
    for key in ('protocol', 'training', 'calibration', 'device', 'seeds', 'data_rules', 'export'):
        assert cfg[key] == frontier[key]
    assert len(cfg['device']['allowed_gpu_uuids']) == 2 and cfg['device']['max_parallel'] == 2
    for source in cfg['source_configs'].values():
        assert rc.sha256_file(REPO / source['path']) == source['sha256']
    assert cfg['train_subset']['fractions'] == {'data25': .25, 'data50': .5, 'data75': .75}
    assert cfg['train_subset']['salt'] and cfg['train_subset']['chosen_without_outcomes'] is True
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.require_executable(cfg, smoke=False)


def test_masks_and_periodic_indices_are_exact():
    cfg = load()
    names = cfg['export']['context_names']
    index = {a['name']: [names.index(c) for c in a['drop_context']] for a in cfg['arms'] if 'drop_context' in a}
    assert index == {'no_pitcher_profile': list(range(28, 51)), 'no_pitcher_support': [51],
                     'no_pitcher_all': list(range(28, 52))}
    base_periodic = json.loads((REPO / 'configs' / 'DRAFT-CUDA-DIRECT-20261004.json').read_text())['arms'][4]
    assert list(cr.BASE_PERIODIC) == base_periodic['context_continuous']
    assert rc.arm_spec(cfg, 'periodic_counts')['context_continuous'] == [*base_periodic['context_continuous'], 52]
    assert rc.arm_spec(cfg, 'periodic_block')['context_continuous'] == [*base_periodic['context_continuous'], 52, 53, 54, 55]
    assert rc.arm_spec(cfg, 'residual_counts')['workload'] == ['prior_records_per100']
    assert rc.arm_spec(cfg, 'residual_block')['workload'] == cfg['export']['workload_columns']


def test_combo_arms_strict_and_earlier_configs_unchanged():
    for path in (tf.BASE, tf.RUN):
        rc.config_check(json.loads(path.read_text()))
    cfg = load()

    def broken(edit):
        bad = copy.deepcopy(cfg)
        edit(bad)
        with pytest.raises((ValueError, KeyError)):
            rc.config_check(bad)

    arm = lambda c, n: next(a for a in c['arms'] if a['name'] == n)  # noqa: E731
    broken(lambda c: arm(c, 'no_pitcher_support')['drop_context'].append('batter_left'))
    broken(lambda c: arm(c, 'no_pitcher_all').update(drop_history=True))
    broken(lambda c: arm(c, 'data25').update(train_fraction=.3))
    broken(lambda c: c['train_subset']['fractions'].update(data50=.4))
    broken(lambda c: c['train_subset'].update(salt=''))
    broken(lambda c: c['train_subset'].update(method='random'))
    broken(lambda c: c.pop('train_subset'))
    broken(lambda c: arm(c, 'residual_counts').update(train_fraction=.5))
    broken(lambda c: arm(c, 'residual_counts').update(workload=['log_rest_days']))
    broken(lambda c: arm(c, 'periodic_block')['context_continuous'].pop())
    broken(lambda c: arm(c, 'periodic_counts')['context_continuous'].append(32))
    broken(lambda c: arm(c, 'data75').update(combo=False))
    broken(lambda c: arm(c, 'data75').update(priority=3))
    broken(lambda c: c['arms'].append(copy.deepcopy(arm(c, 'data25'))))
    broken(lambda c: c['arms'].append({**copy.deepcopy(arm(c, 'data25')), 'name': 'data10'}))
    broken(lambda c: c['export']['context_names'].reverse())
    broken(lambda c: c['device'].update(max_parallel=3))
    # a base (non-combo) arm can never subsample TRAIN
    base = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match='Only registered'):
        rc.train_subset(base, {**base['arms'][0], 'train_fraction': .5}, {'keys': np.zeros((4, 3), np.int64)})


def test_combo_modules_are_pinned_sources_and_config_check_needs_no_torch():
    assert {'pitchmdp/direct_combo_model.py', 'pitchmdp/direct_combo_registry.py'} <= set(rc.MODEL_SOURCES)
    assert rc.MODEL_SOURCES[0] == 'pitchmdp/matrix_models.py' and set(rc.own_hashes()) == {rc.SELF, *rc.MODEL_SOURCES}
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'rc.config_check(json.load(open({str(COMBO)!r}))); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------ nested TRAIN subset

def test_subset_is_nested_order_preserving_and_depends_on_keys_and_salt_only():
    rng = np.random.default_rng(0)
    keys = np.column_stack([np.repeat(np.arange(500), 4) + 700000, np.tile([1, 1, 2, 2], 500),
                            np.tile([1, 2, 1, 2], 500)]).astype(np.int64)
    arrays = {'keys': keys, 'y': rng.integers(0, 10, len(keys)), 'context': rng.normal(size=(len(keys), 3))}
    masks = {f: cr.subset_mask(keys, 'salt-a', f) for f in (.25, .5, .75)}
    assert (masks[.25] <= masks[.5]).all() and (masks[.5] <= masks[.75]).all()  # nested
    for f, m in masks.items():
        assert abs(m.mean() - f) < .05
    kept, record = cr.subset_split(arrays, 'salt-a', .5)
    index = np.flatnonzero(masks[.5])
    assert np.array_equal(kept['keys'], keys[index]) and np.array_equal(kept['y'], arrays['y'][index])
    assert (np.diff(index) > 0).all()  # original order
    assert record['n_source'] == len(keys) and record['n_selected'] == masks[.5].sum()
    assert record['selected_rows_sha256'] == cr.rows_sha256(keys[index]) and record['salt'] == 'salt-a'
    # labels / other arrays never matter; the salt does; deterministic
    shuffled_y = {**arrays, 'y': rng.permutation(arrays['y']), 'context': -arrays['context']}
    assert cr.subset_split(shuffled_y, 'salt-a', .5)[1]['selected_mask_sha256'] == record['selected_mask_sha256']
    assert not np.array_equal(cr.subset_mask(keys, 'salt-b', .5), masks[.5])
    assert np.array_equal(cr.subset_mask(keys.copy(), 'salt-a', .5), masks[.5])
    # a row's membership does not depend on the other rows
    assert np.array_equal(cr.subset_mask(keys[::3], 'salt-a', .5), masks[.5][::3])
    with pytest.raises(ValueError, match='unique'):
        cr.subset_mask(np.vstack([keys, keys[:1]]), 'salt-a', .5)
    with pytest.raises(ValueError):
        cr.subset_mask(keys, 'salt-a', 1.)


@pytest.mark.parametrize('arm', ['control', *DATA])
def test_data_arm_task_subsets_train_only_and_is_seed_independent(export, tmp_path, arm):
    path, digest, splits = export
    cfg = load()
    fits = {}
    for seed in (0, 1):
        dest = rc.run_task(cfg, path, digest, tmp_path / 'd-smoke', arm, seed, 'cpu', {'test': True}, smoke=True)
        rc.verify_task(dest, rc.expected_identity(cfg, digest, arm, seed, True, 'cpu', rc.own_hashes()))
        fits[seed] = json.loads((dest / 'fit.json').read_text())
        with np.load(dest / 'predictions.npz') as data:
            assert np.array_equal(data['dev_keys'], splits['dev']['keys'])  # non-TRAIN rows unchanged
        with np.load(dest / 'mlb_predictions.npz') as data:
            assert np.array_equal(data['mlb_dev_keys'], splits['mlb_dev']['keys'])
    full = cr.rows_sha256(splits['train']['keys'])
    early = cr.rows_sha256(splits['earlystop']['keys'])
    assert fits[0]['earlystop_rows_sha256'] == fits[1]['earlystop_rows_sha256'] == early
    assert fits[0]['train_rows_sha256'] == fits[1]['train_rows_sha256']  # independent of the model seed
    if arm == 'control':
        assert fits[0]['train_subset'] is None and fits[0]['train_rows_sha256'] == full
        return
    sub = fits[0]['train_subset']
    assert sub == fits[1]['train_subset'] and sub['fraction'] == cr.FRACTIONS[arm]
    mask = cr.subset_mask(splits['train']['keys'], cfg['train_subset']['salt'], cr.FRACTIONS[arm])
    assert sub['n_source'] == 400 and sub['n_selected'] == mask.sum() == fits[0]['report']['training_rows']
    assert sub['source_rows_sha256'] == full and fits[0]['train_rows_sha256'] == sub['selected_rows_sha256'] \
        == cr.rows_sha256(splits['train']['keys'][mask])


# ------------------------------------------------------------------ pitcher masks and workload combinations

@pytest.mark.parametrize('arm', PITCHER)
def test_pitcher_mask_is_exact_and_roundtrips(export, tmp_path, arm):
    path, digest, splits = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'm-smoke', arm, 0, 'cpu', {}, smoke=True)
    fit = json.loads((dest / 'fit.json').read_text())
    assert fit['context_names_verified'] is True and fit['context_width'] == 52 and fit['train_subset'] is None
    spec, names = rc.arm_spec(cfg, arm), cfg['export']['context_names']
    model = rc.load_model(spec, dest / 'model.pt', 'cpu', names)
    dropped = [names.index(c) for c in spec['drop_context']]
    assert model.drop_index == dropped
    inputs = rc.arm_inputs(splits['dev'], spec, tf.COLUMNS)
    noisy = inputs[2].copy()
    noisy[:, dropped] += 5.
    assert np.allclose(model.logits(inputs), model.logits((inputs[0], inputs[1], noisy)), atol=1e-6)
    kept = inputs[2].copy()
    kept[:, 27] += 5.  # last batter membership: never masked by a pitcher arm
    assert not np.allclose(model.logits(inputs), model.logits((inputs[0], inputs[1], kept)), atol=1e-4)
    other = rc.arm_spec(cfg, 'no_pitcher_all' if arm != 'no_pitcher_all' else 'no_pitcher_support')
    with pytest.raises(ValueError, match='pitcher-information'):
        rc.load_model(other, dest / 'model.pt', 'cpu', names)
    with pytest.raises(ValueError, match='context names'):
        rc.load_model(spec, dest / 'model.pt', 'cpu', list(reversed(names)))
    with pytest.raises(ValueError, match='context names'):
        rc.build_model(spec, 0, 'cpu', None)


@pytest.mark.parametrize('arm', WORKLOAD_ARMS)
def test_workload_combination_input_and_checkpoint_roundtrip(export, tmp_path, arm):
    path, digest, splits = export
    cfg = load()
    spec = rc.arm_spec(cfg, arm)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'w-smoke', arm, 2, 'cpu', {}, smoke=True)
    fit = json.loads((dest / 'fit.json').read_text())
    assert fit['context_width'] == 52 + len(spec['workload']) and fit['train_subset'] is None
    model = rc.load_model(spec, dest / 'model.pt', 'cpu')
    assert model.net.config['n_context'] == fit['context_width'] and model.kind == spec['model']
    if spec['model'] == 'periodic_flatten_mlp':
        assert model.net.config['context_continuous'] == spec['context_continuous']
    inputs = rc.arm_inputs(splits['dev'], spec, tf.COLUMNS)
    keep = [tf.COLUMNS.index(c) for c in spec['workload']]
    assert np.array_equal(inputs[2][:, 52:], splits['dev']['workload'][:, keep])  # appended after the 52 columns
    assert np.array_equal(inputs[2][:, :52], splits['dev']['context'])
    temperature = json.loads((dest / 'calibration.json').read_text())['delivery_temperature']
    with np.load(dest / 'predictions.npz') as data:
        again, _ = rc.probabilities(rc.g0_log_probs(model.logits(inputs)), temperature)
        assert np.allclose(again, data['dev'], atol=1e-6)
    moved = inputs[2].copy()
    moved[:, 52:] += 3.
    assert not np.allclose(model.logits(inputs), model.logits((inputs[0], inputs[1], moved)), atol=1e-4)
    twin = {'residual_counts': 'residual_block', 'residual_block': 'residual_counts',
            'periodic_counts': 'periodic_block', 'periodic_block': 'periodic_counts'}[arm]
    with pytest.raises(ValueError):
        rc.load_model(rc.arm_spec(cfg, twin), dest / 'model.pt', 'cpu')


# ------------------------------------------------------------------ portable scorer

def fake_combo_family(tmp_path, cfg, digest, splits, tamper=None):
    root = tmp_path / 'fam'
    full = cr.rows_sha256(splits['train']['keys'])
    for arm in [a['name'] for a in cfg['arms']]:
        for seed in cfg['seeds']:
            dest = tf.fake_task(root, cfg, digest, arm, seed, splits, rc.own_hashes())
            fit = json.loads((dest / 'fit.json').read_text())
            spec = rc.arm_spec(cfg, arm)
            if 'train_fraction' in spec:
                _, record = cr.subset_split(splits['train'], cfg['train_subset']['salt'], spec['train_fraction'])
                fit.update(train_subset=record, train_rows_sha256=record['selected_rows_sha256'])
            else:
                fit.update(train_subset=None, train_rows_sha256=full)
            if tamper and tamper[0] == (arm, seed):
                tamper[1](fit)
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest)
    return root


def test_scorer_scores_combo_family_with_subset_provenance(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    result = tf.run_score(tmp_path, cfg, export, fake_combo_family(tmp_path, cfg, digest, splits))
    assert result['decision'] is None and result['ranking'] is None
    assert list(result['splits']['dev']) == [a['name'] for a in cfg['arms']]
    tasks = result['provenance']['tasks']
    assert tasks['control']['0']['train_subset'] is None
    assert [tasks[a]['0']['train_subset']['fraction'] for a in DATA] == [.25, .5, .75]
    n = [tasks[a]['1']['train_subset']['n_selected'] for a in DATA]
    assert n[0] < n[1] < n[2] < 400


@pytest.mark.parametrize('tamper', [
    (('data50', 1), lambda f: f['train_subset'].update(salt='other')),
    (('data50', 0), lambda f: f['train_subset'].update(fraction=.25)),
    (('data25', 2), lambda f: f.update(train_rows_sha256='0' * 64)),
    (('data75', 0), lambda f: f.update(train_subset=None)),
    (('control', 1), lambda f: f.update(train_rows_sha256='0' * 64)),
    (('residual_block', 0), lambda f: f.update(train_subset={'fraction': .5})),
])
def test_scorer_refuses_subset_provenance_mismatch(export, tmp_path, tamper):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = fake_combo_family(tmp_path, cfg, digest, splits, tamper)
    with pytest.raises(ValueError, match='TRAIN'):
        tf.run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()


def test_scorer_old_source_map_still_scores_base_family(tmp_path):
    """The earlier pinned (pre-combo) task source map remains a valid --source-manifest."""
    first = {k: '1' * 64 for k in (rc.SELF, 'pitchmdp/matrix_models.py', 'pitchmdp/direct_residual_model.py',
                                     'pitchmdp/direct_periodic_model.py')}
    frontier = {**first, 'pitchmdp/direct_frontier_model.py': '2' * 64, 'pitchmdp/direct_frontier_registry.py': '3' * 64}
    for i, old in enumerate((first, frontier)):
        manifest = tmp_path / f'sources-{i}.json'
        manifest.write_text(json.dumps({'format': sc.SOURCE_FORMAT, 'label': 'earlier batch', 'source_sha256': old}))
        assert sc.task_sources(manifest)[0] == old


# ------------------------------------------------------------------ factorial contrasts

def with_contrasts(cfg, edit):
    cfg = copy.deepcopy(cfg)
    edit(cfg['comparison']['factorial_contrasts'])
    return cfg


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0])),                                         # duplicate name / pair
    lambda cs: cs[0].update(reference='residual_counts'),                      # candidate == reference
    lambda cs: cs[0].update(reference='nonexistent'),                          # unregistered arm
    lambda cs: cs[0].update(isolates='architecture'),                          # wrong declared factor
    lambda cs: cs[0].update(reference='control'),                              # two changes at once
    lambda cs: cs[0].update(reference='data50'),                               # data arm never enters
    lambda cs: cs[0].update(winner=True),                                      # no extra keys (no verdict)
    lambda cs: cs.clear(),
])
def test_contrast_config_is_strict(edit):
    with pytest.raises(ValueError):
        sc.check_contrasts(with_contrasts(load(registered=True), edit))


def test_scorer_refuses_bad_contrast_before_reading_predictions(export, tmp_path):
    cfg = with_contrasts(load(registered=True), lambda cs: cs[1].update(reference='periodic'))
    with pytest.raises(ValueError, match='factor'):
        tf.run_score(tmp_path, cfg, export, tmp_path / 'no-family-needed')
    assert not (tmp_path / 'score').exists()


def test_scorer_reports_factorial_contrasts_with_primary_helper(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    result = tf.run_score(tmp_path, cfg, export, fake_combo_family(tmp_path, cfg, digest, splits))
    assert result['decision'] is None and result['ranking'] is None
    for split in ('dev', 'mlb_dev'):
        block = result['factorial_contrasts'][split]
        assert list(block) == [c['name'] for c in cfg['comparison']['factorial_contrasts']] and len(block) == 8
        one = block['residual_counts_vs_residual']
        assert one['isolates'] == 'workload' and set(one['seeds']) == {'0', '1', '2'}
        assert set(one['ensemble']) == set(result['splits'][split]['counts']['paired_vs_control']['ensemble'])
        arms = result['splits'][split]
        for c in cfg['comparison']['factorial_contrasts']:  # candidate - reference, same helper as vs-control
            vs = lambda a: 0. if a == 'control' else arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
            assert block[c['name']]['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)


def test_family_without_contrasts_has_no_contrast_block(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    del cfg['comparison']['factorial_contrasts']
    assert sc.check_contrasts(cfg) is None
    result = tf.run_score(tmp_path, cfg, export, fake_combo_family(tmp_path, cfg, digest, splits))
    assert 'factorial_contrasts' not in result


def test_incomplete_factorial_family_is_refused(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = fake_combo_family(tmp_path, cfg, digest, splits)
    import shutil
    shutil.rmtree(rc.task_dir(fam, 'residual', 1, False))
    with pytest.raises((ValueError, FileNotFoundError)):
        tf.run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()
