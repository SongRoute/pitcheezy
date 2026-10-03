"""Synthetic CPU tests: TabM-mini runner/scorer integration (DRAFT-CUDA-TABM-MINI-20261004)."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export / fake-task helpers
from pitchmdp import direct_tabm_registry as tr  # noqa: E402
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402
from pitchmdp.direct_dropout_model import PeriodicHeadDropoutModel  # noqa: E402
from pitchmdp.direct_periodic_model import DirectPeriodicModel  # noqa: E402
from pitchmdp.direct_tabm_model import ENSEMBLE, DirectTabMModel  # noqa: E402

torch.set_num_threads(1)
RUN = REPO / 'configs' / 'DRAFT-CUDA-TABM-MINI-20261004.json'
P30 = REPO / 'configs' / 'EXP-P30-001.json'
P30_DRAFT = REPO / 'configs' / 'DRAFT-CUDA-WORKLOAD-DROPOUT-20261004.json'
COMBO = REPO / 'configs' / 'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json'
BASE = REPO / 'configs' / 'DRAFT-CUDA-DIRECT-20261004.json'
COLUMNS = ['prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing']


def load(registered=False, unpin=True):
    return tf.load(RUN, registered, unpin)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


# ------------------------------------------------------------------ config

def test_config_is_draft_exact_5_arm_family_on_seeds_13_15():
    cfg = load(unpin=False)
    p30 = json.loads(P30.read_text())
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'repeat_tag' not in cfg
    assert 'registration' not in cfg and cfg['experiment_id'] != p30['experiment_id']
    assert [a['name'] for a in cfg['arms']] == list(tr.ARM_ORDER) and len(tr.ARM_ORDER) == 5
    assert cfg['seeds'] == [13, 14, 15] and cfg['fits'] == 15 == 5 * len(cfg['seeds'])
    old = {a['name']: a for a in p30['arms']}
    arms = {a['name']: a for a in cfg['arms']}
    for name in ('control', 'periodic_counts', 'periodic_counts_d10'):  # exact registered P30 entries
        assert arms[name] == old[name]
    for name in tr.ARMS:
        assert arms[name]['workload'] == old['periodic_counts']['workload'] == ['prior_records_per100']
        assert arms[name]['context_continuous'] == old['periodic_counts']['context_continuous'] == wr.CONTINUOUS['counts']
        assert (arms[name]['k'], arms[name]['scaling_init']) == (ENSEMBLE['k'], ENSEMBLE['scaling_init'])
    assert cfg['training'] == rc.TRAINING == p30['training'] and cfg['calibration'] == p30['calibration']
    assert cfg['export'] == p30['export'] and cfg['device'] == p30['device']
    q = cfg['queue']
    assert (q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']) == (3600, 36000, 2)
    assert (q['date'], q['no_new_fit_after'], q['kill_own_after'], q['aggregate_by']) == ('2026-10-04', '19:00', '19:30', '20:00')
    assert cfg['comparison'][tr.CONTRAST_KEY] == tr.CONTRASTS and len(tr.CONTRASTS) == 3
    assert {k for k in cfg['comparison'] if k.endswith('_contrasts')} == {tr.CONTRAST_KEY}
    assert cfg['source_configs']['workload_dropout_run']['sha256'] == rc.sha256_file(P30)


def test_seed_blocks_are_disjoint_from_every_earlier_block():
    assert not set(tr.SEEDS) & {0, 1, 2, *wr.FIRST_SEEDS, *wr.SEED_BLOCKS['repeat2_seeds_8_12']}


@pytest.mark.parametrize('edit', [
    lambda c: c['arms'][3].update(k=16),
    lambda c: c['arms'][3].update(scaling_init='random-signs'),
    lambda c: c['arms'][4].update(head_dropout=.2),
    lambda c: c['arms'][3].update(workload=['prior_records_per100', 'rest_missing']),
    lambda c: c['arms'][4].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][3].update(lr=1e-3),                                          # unchecked extra parameter
    lambda c: c['arms'][3].update(priority=2),
    lambda c: c['arms'][3].pop('tabm'),
    lambda c: c['arms'][3].update(tabm=False),
    lambda c: c['arms'][1].update(tabm=True),                                        # old arm flagged
    lambda c: c['arms'][2].update(head_dropout=.2),                                  # old arm altered
    lambda c: c['arms'][1].update(workload=[]),
    lambda c: c['arms'].pop(),
    lambda c: c['arms'].insert(1, c['arms'].pop(3)),
    lambda c: c['arms'].append({**c['arms'][3], 'name': 'tabm_counts_d20', 'head_dropout': .2}),
    lambda c: c.update(seeds=[0, 1, 2]),
    lambda c: c.update(seeds=[3, 4, 5]),
    lambda c: c.update(seeds=[13, 14, 15, 16, 17]),
    lambda c: c.update(seeds=[15, 14, 13]),
    lambda c: c.update(repeat_tag='repeat2_seeds_8_12'),
    lambda c: c.update(fits=14),
    lambda c: c.update(training={**c['training'], 'epochs': 40}),
    lambda c: c['device'].update(max_parallel=3),
])
def test_config_check_refuses_any_family_or_seed_drift(edit):
    cfg = copy.deepcopy(load())
    edit(cfg)
    with pytest.raises(ValueError):
        rc.config_check(cfg)


@pytest.mark.parametrize('path', [BASE, COMBO, P30_DRAFT, P30])
def test_other_families_unchanged_and_refuse_tabm_seeds_or_arms(path):
    cfg = json.loads(path.read_text())
    seeds = rc.config_check(copy.deepcopy(cfg))['seeds']
    assert seeds == ([3, 4, 5, 6, 7] if 'workload_dropout_contrasts' in cfg.get('comparison', {}) else [0, 1, 2])
    assert tr.check_contrasts(cfg) is None and not rc.uses_tabm(cfg)
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'seeds': [13, 14, 15]})
    tabm_arm = {'name': 'tabm_counts', 'priority': 1, 'tabm': True, 'workload': tr.WORKLOAD, **tr.ARMS['tabm_counts']}
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'arms': [*cfg['arms'], tabm_arm]})


def test_contrasts_change_exactly_one_factor_with_candidate_minus_reference():
    cfg = load()
    arms = {a['name']: a for a in cfg['arms']}
    assert tr.check_contrasts(cfg) == tr.CONTRASTS
    seen = set()
    for c in tr.CONTRASTS:
        (ch, cd, *_), (rh, rd, *_) = tr.factors(arms[c['candidate']]), tr.factors(arms[c['reference']])
        if c['isolates'] == 'ensemble_head':
            assert (ch, rh) == ('tabm_mini', 'single') and cd == rd == c['held']['head_dropout']
        else:
            assert ch == rh == 'single' and (cd, rd) == (.1, 0.)
        assert 'control' not in (c['candidate'], c['reference'])
        assert c['name'] == f"{c['candidate']}_vs_{c['reference']}"
        seen.add((c['candidate'], c['reference']))
    assert len(seen) == 3


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0], name='again')),
    lambda cs: cs[0].update(candidate='periodic_counts', reference='tabm_counts'),   # reversed sign
    lambda cs: cs[0].update(reference='periodic_counts_d10'),                         # head AND dropout change
    lambda cs: cs[1].update(reference='control'),
    lambda cs: cs[2].update(held={'head': 'tabm_mini'}),
    lambda cs: cs[0].update(winner=True),
    lambda cs: cs.pop(),
])
def test_contrast_list_is_fixed(edit):
    cfg = copy.deepcopy(load(registered=True))
    edit(cfg['comparison'][tr.CONTRAST_KEY])
    with pytest.raises(ValueError):
        tr.check_contrasts(cfg)


def test_modules_pinned_and_config_check_needs_no_torch():
    assert {'pitchmdp/direct_tabm_model.py', 'pitchmdp/direct_tabm_factory.py', 'pitchmdp/direct_tabm_registry.py'} \
        <= set(rc.MODEL_SOURCES)
    assert 'pitchmdp/direct_tabm_registry.py' in sc.SCORER_SOURCES
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'rc.config_check(json.load(open({str(RUN)!r}))); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------ networks / training

def inputs(arm, n, seed):
    a = tf.split_arrays(n, seed, 0)
    return rc.arm_inputs(a, arm, COLUMNS), a['y']


def small_fit(model, arm):
    train, y = inputs(arm, 300, 11)
    early, ey = inputs(arm, 90, 12)
    return model.fit(train, y, early, ey, epochs=3, patience=2, batch_size=64)


def test_single_arms_keep_old_code_paths_and_tabm_arms_build_tabm():
    cfg = load()
    assert type(rc.build_model(rc.arm_spec(cfg, 'periodic_counts'), 5, 'cpu')) is DirectPeriodicModel
    assert type(rc.build_model(rc.arm_spec(cfg, 'periodic_counts_d10'), 5, 'cpu')) is PeriodicHeadDropoutModel
    for d, name in tr.TABM.items():
        spec = rc.arm_spec(cfg, name)
        model = small_fit(rc.build_model(spec, 5, 'cpu'), spec)
        assert type(model) is DirectTabMModel
        assert model.architecture['head_dropout'] == d and model.net.config['k'] == 8
        assert model.net.config['n_context'] == 53 and model.context_continuous == tr.CONTINUOUS
        assert [m.p for m in model.net.modules() if isinstance(m, torch.nn.Dropout)] == ([d, d] if d else [0., 0.])
        x, _ = inputs(spec, 40, 13)
        assert model.member_logits(x).shape == (40, 8, 10) and model.logits(x).shape == (40, 10)
    with pytest.raises(ValueError):
        rc.build_model(rc.arm_spec(cfg, 'tabm_counts'), 3, 'cpu', ['x'] * 52)


@pytest.mark.parametrize('name', ['tabm_counts', 'tabm_counts_d10'])
def test_checkpoint_roundtrip_and_strict_loader(tmp_path, name):
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    model = small_fit(rc.build_model(spec, 3, 'cpu'), spec)
    model.save(tmp_path / 'm.pt')
    again = rc.load_model(spec, tmp_path / 'm.pt', 'cpu')
    x, _ = inputs(spec, 30, 15)
    assert np.array_equal(model.logits(x), again.logits(x))
    for other in ('tabm_counts', 'tabm_counts_d10', 'periodic_counts', 'periodic_counts_d10'):
        if other != name:
            with pytest.raises(ValueError):
                rc.load_model(rc.arm_spec(cfg, other), tmp_path / 'm.pt', 'cpu')
    wide = {**spec, 'workload': ['prior_records_per100', 'rest_missing']}  # context width guard
    with pytest.raises(ValueError, match='context width'):
        rc.load_model(wide, tmp_path / 'm.pt', 'cpu')


def test_runner_smoke_task_cpu_and_seed_refusal(export, tmp_path):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'tabm_counts_d10', 13, 'cpu', {}, smoke=True)
    fit = json.loads((dest / 'fit.json').read_text())
    assert fit['context_width'] == 53
    assert fit['parameter_count'] > DirectTabMModel(0, tr.CONTINUOUS).architecture['width']  # recorded
    identity = json.loads((dest / 'identity.json').read_text())
    assert identity['seed'] == 13 and identity['arm']['k'] == 8 and identity['arm']['head_dropout'] == .1
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    with np.load(dest / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(axis=1), 1.) and p['dev'].shape[1] == 10
    for seed in (0, 3, 8, 16):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'tabm_counts_d10', seed, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(cfg, path, digest, tmp_path / 'real', 'tabm_counts_d10', 13, 'cpu', {})


# ------------------------------------------------------------------ scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # synthetic speed only; the scorer default stays 10,000


def test_scorer_reports_all_contrasts_with_sign(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    result = tf.run_score(tmp_path, cfg, export, tf.family(tmp_path, cfg, digest, splits, same_as_control=()))
    assert result['decision'] is None and result['ranking'] is None
    assert not {'factorial_contrasts', 'regularization_contrasts', 'workload_dropout_contrasts'} & set(result)
    assert result['seed_block']['seeds'] == [13, 14, 15] and result['seed_block']['repeat_tag'] is None
    assert 'compute-matched' in result['seed_block']['note']
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(tr.ARM_ORDER)
        block = result['tabm_contrasts'][split]
        assert list(block) == [c['name'] for c in tr.CONTRASTS]
        vs = lambda a: arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in tr.CONTRASTS:
            one = block[c['name']]
            assert (one['isolates'], one['held']) == (c['isolates'], c['held'])
            assert set(one['seeds']) == {'13', '14', '15'}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert 'tabm_counts_d10_vs_periodic_counts_d10' in table and '연산량은 같지 않습니다' in table


def test_scorer_refuses_incomplete_or_missing_contrasts(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=(('tabm_counts', 15),))
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    bare = copy.deepcopy(cfg)
    del bare['comparison'][tr.CONTRAST_KEY]
    with pytest.raises(ValueError, match='fixed contrasts'):
        tf.run_score(tmp_path, bare, export, tmp_path / 'no-family-needed')
    both = copy.deepcopy(cfg)
    both['comparison'][wr.CONTRAST_KEY] = wr.CONTRASTS
    with pytest.raises(ValueError):
        tf.run_score(tmp_path, both, export, tmp_path / 'no-family-needed')
    assert not (tmp_path / 'score').exists()
