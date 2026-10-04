"""Synthetic CPU tests: embedded-input residual runner/scorer integration (DRAFT-CUDA-EMBED-RESIDUAL-20261004)."""
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
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_tabm_registry as tr  # noqa: E402
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402
from pitchmdp.direct_dropout_model import PeriodicHeadDropoutModel  # noqa: E402
from pitchmdp.direct_periodic_model import DirectPeriodicModel  # noqa: E402
from pitchmdp.direct_periodic_residual_model import (DirectLinearResidualModel,  # noqa: E402
                                                     DirectPeriodicResidualModel)
from pitchmdp.direct_residual_model import DirectResidualModel  # noqa: E402
from pitchmdp.matrix_models import MatrixModel  # noqa: E402

torch.set_num_threads(1)
RUN = REPO / 'configs' / 'DRAFT-CUDA-EMBED-RESIDUAL-20261004.json'
TABM = REPO / 'configs' / 'DRAFT-CUDA-TABM-MINI-20261004.json'
P31 = REPO / 'configs' / 'EXP-P31-001.json'
P30 = REPO / 'configs' / 'EXP-P30-001.json'
P30_DRAFT = REPO / 'configs' / 'DRAFT-CUDA-WORKLOAD-DROPOUT-20261004.json'
COMBO = REPO / 'configs' / 'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json'
BASE = REPO / 'configs' / 'DRAFT-CUDA-DIRECT-20261004.json'
COLUMNS = ['prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing']
NEW = {'periodic_residual_counts': DirectPeriodicResidualModel, 'linear_residual_counts': DirectLinearResidualModel}


def load(registered=False, unpin=True):
    return tf.load(RUN, registered, unpin)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


# ------------------------------------------------------------------ config

def test_config_is_draft_exact_6_arm_family_on_seeds_16_18():
    cfg = load(unpin=False)
    tabm = json.loads(TABM.read_text())
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'repeat_tag' not in cfg
    assert 'registration' not in cfg and cfg['experiment_id'] == 'DRAFT-CUDA-EMBED-RESIDUAL-20261004'
    assert [a['name'] for a in cfg['arms']] == list(er.ARM_ORDER) and len(er.ARM_ORDER) == 6
    assert cfg['seeds'] == [16, 17, 18] and cfg['fits'] == 18 == 6 * len(cfg['seeds'])
    old = {**{a['name']: a for a in json.loads(COMBO.read_text())['arms'] if a['name'] == 'residual_counts'},
           **{a['name']: a for a in json.loads(P30.read_text())['arms']}}
    arms = {a['name']: a for a in cfg['arms']}
    for name in er.BASELINES:  # exact earlier registered / run entries, JSON-exact
        assert er.same(arms[name], old[name])
    for name in er.ARMS:
        assert arms[name]['workload'] == old['periodic_counts']['workload'] == ['prior_records_per100']
        assert arms[name]['context_continuous'] == old['periodic_counts']['context_continuous'] == wr.CONTINUOUS['counts']
    # everything outside the family-specific fields is the TabM DRAFT (itself the P30 export/training/device)
    for key in ('protocol', 'export', 'training', 'calibration', 'device', 'data_rules', 'decision'):
        assert cfg[key] == tabm[key]
    q, tq = cfg['queue'], tabm['queue']
    assert {k: v for k, v in q.items() if k not in ('batch', 'order', 'start_after')} == \
        {k: v for k, v in tq.items() if k not in ('batch', 'order', 'start_after')}
    assert (q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']) == (3600, 36000, 2)
    assert (q['date'], q['no_new_fit_after'], q['kill_own_after'], q['aggregate_by']) == ('2026-10-04', '19:00', '19:30', '20:00')
    assert '/home/sds733/pitcheezy-autonomy/gpu-locks' in q['start_after']
    assert cfg['comparison'][er.CONTRAST_KEY] == er.CONTRASTS and len(er.CONTRASTS) == 6
    assert {k for k in cfg['comparison'] if k.endswith('_contrasts')} == {er.CONTRAST_KEY}
    for key, path in (('tabm_run', TABM), ('data_combo_run', COMBO), ('workload_dropout_run', P30)):
        assert cfg['source_configs'][key]['sha256'] == rc.sha256_file(path)
    note = cfg['source_configs']['design_note']
    assert rc.sha256_file(REPO / note['path']) == note['sha256']


def test_seed_block_is_disjoint_from_every_earlier_block():
    assert not set(er.SEEDS) & {0, 1, 2, *wr.FIRST_SEEDS, *wr.SEED_BLOCKS['repeat2_seeds_8_12'], *tr.SEEDS}


@pytest.mark.parametrize('edit', [
    lambda c: c['arms'][4].update(embedding='linear'),
    lambda c: c['arms'][5].update(model='periodic_residual_mlp'),
    lambda c: c['arms'][4].update(width=256),
    lambda c: c['arms'][4].update(workload=['prior_records_per100', 'rest_missing']),
    lambda c: c['arms'][5].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][4].update(dropout=.2),                                       # unchecked extra parameter
    lambda c: c['arms'][4].update(priority=2),
    lambda c: c['arms'][4].pop('embed_residual'),
    lambda c: c['arms'][4].update(embed_residual=1),
    lambda c: c['arms'][1].update(embed_residual=True),                              # old arm flagged
    lambda c: c['arms'][1].update(model='periodic_flatten_mlp'),                     # old arm altered
    lambda c: c['arms'][3].update(head_dropout=.2),
    lambda c: c['arms'][2].update(workload=[]),
    lambda c: c['arms'].pop(),
    lambda c: c['arms'].insert(1, c['arms'].pop(4)),
    lambda c: c['arms'].append({**c['arms'][4], 'name': 'periodic_residual_block'}),
    lambda c: c.update(seeds=[0, 1, 2]),
    lambda c: c.update(seeds=[13, 14, 15]),
    lambda c: c.update(seeds=[16, 17, 18, 19]),
    lambda c: c.update(seeds=[18, 17, 16]),
    lambda c: c.update(repeat_tag='repeat2_seeds_8_12'),
    lambda c: c.update(fits=15),
    lambda c: c.update(training={**c['training'], 'epochs': 40}),
    lambda c: c['device'].update(max_parallel=3),
    lambda c: c['export'].update(last_year=2026),
])
def test_config_check_refuses_any_family_or_seed_drift(edit):
    cfg = copy.deepcopy(load())
    edit(cfg)
    with pytest.raises(ValueError):
        rc.config_check(cfg)


DROPOUT_RUN = REPO / 'configs' / 'DRAFT-CUDA-DROPOUT-RUN-20261004.json'
FRONTIER_RUN = REPO / 'configs' / 'DRAFT-CUDA-FRONTIER-RUN-20261004.json'


@pytest.mark.parametrize('path', [BASE, COMBO, DROPOUT_RUN, FRONTIER_RUN, P30_DRAFT, P30, TABM, P31])
def test_other_families_unchanged_and_refuse_embed_seeds_or_arms(path):
    cfg = json.loads(path.read_text())
    seeds = rc.config_check(copy.deepcopy(cfg))['seeds']
    expected = {'workload_dropout_contrasts': [3, 4, 5, 6, 7], 'tabm_contrasts': [13, 14, 15]}
    assert seeds == next((v for k, v in expected.items() if k in cfg.get('comparison', {})), [0, 1, 2])
    assert er.check_contrasts(cfg) is None and not rc.uses_embed_residual(cfg)
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'seeds': [16, 17, 18]})
    arm = {'name': 'periodic_residual_counts', 'priority': 1, er.FLAG: True, 'workload': er.WORKLOAD,
           **er.ARMS['periodic_residual_counts']}
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'arms': [*cfg['arms'], arm]})
    foreign = copy.deepcopy(cfg)  # review 040c8b4: the embed contrast key never attaches to another family
    foreign.setdefault('comparison', {})[er.CONTRAST_KEY] = copy.deepcopy(er.CONTRASTS)
    with pytest.raises(ValueError):
        er.check_contrasts(foreign)


def test_scorer_refuses_embed_contrasts_on_a_registered_other_family_before_any_read(tmp_path):
    """Review 040c8b4: P31 (REGISTERED) + embed_residual_contrasts fails before the export / family is opened."""
    cfg = json.loads(P31.read_text())
    cfg['comparison'][er.CONTRAST_KEY] = copy.deepcopy(er.CONTRASTS)
    cpath, csha = tf.write_cfg(tmp_path, cfg)
    with pytest.raises(ValueError, match='Embedded-residual family must be exactly'):
        sc.score(cpath, csha, tmp_path / 'no-export', '0' * 64, tmp_path / 'no-family', tmp_path / 'score')
    assert not (tmp_path / 'score').exists()


def test_contrast_list_matches_registry_and_changes_declared_factors():
    cfg = load()
    assert er.check_contrasts(cfg) == er.CONTRASTS
    assert all('control' not in (c['candidate'], c['reference']) for c in er.CONTRASTS)
    assert all(c['name'] == f"{c['candidate']}_vs_{c['reference']}" for c in er.CONTRASTS)


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0], name='again')),
    lambda cs: cs[0].update(candidate='periodic_counts', reference='periodic_residual_counts'),   # reversed sign
    lambda cs: cs[3].update(reference='periodic_counts'),
    lambda cs: cs[4].update(held={'backbone': 'flatten'}),
    lambda cs: cs[0].update(winner=True),
    lambda cs: cs.pop(),
])
def test_contrast_list_is_fixed(edit):
    cfg = copy.deepcopy(load(registered=True))
    edit(cfg['comparison'][er.CONTRAST_KEY])
    with pytest.raises(ValueError):
        er.check_contrasts(cfg)


def test_modules_pinned_and_config_check_needs_no_torch():
    assert {'pitchmdp/direct_periodic_residual_model.py', 'pitchmdp/direct_embed_residual_factory.py',
            'pitchmdp/direct_embed_residual_registry.py'} <= set(rc.MODEL_SOURCES)
    assert 'pitchmdp/direct_embed_residual_registry.py' in sc.SCORER_SOURCES
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


def test_baseline_arms_keep_old_code_paths_and_new_arms_build_embed_residual():
    cfg = load()
    assert type(rc.build_model(rc.arm_spec(cfg, 'control'), 5, 'cpu')) is MatrixModel
    assert type(rc.build_model(rc.arm_spec(cfg, 'residual_counts'), 5, 'cpu')) is DirectResidualModel
    assert type(rc.build_model(rc.arm_spec(cfg, 'periodic_counts'), 5, 'cpu')) is DirectPeriodicModel
    assert type(rc.build_model(rc.arm_spec(cfg, 'periodic_counts_d10'), 5, 'cpu')) is PeriodicHeadDropoutModel
    for name, cls in NEW.items():
        spec = rc.arm_spec(cfg, name)
        model = small_fit(rc.build_model(spec, 5, 'cpu'), spec)
        assert type(model) is cls and model.kind == spec['model']
        assert model.net.config['n_context'] == 53 and model.context_continuous == er.CONTINUOUS
        assert [m.p for m in model.net.modules() if isinstance(m, torch.nn.Dropout)] == [.1, .1]
        x, _ = inputs(spec, 40, 13)
        assert model.logits(x).shape == (40, 10)
    with pytest.raises(ValueError):
        rc.build_model(rc.arm_spec(cfg, 'periodic_residual_counts'), 3, 'cpu', ['x'] * 52)


@pytest.mark.parametrize('name', list(NEW))
def test_checkpoint_roundtrip_and_strict_loader(tmp_path, name):
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    model = small_fit(rc.build_model(spec, 3, 'cpu'), spec)
    model.save(tmp_path / 'm.pt')
    again = rc.load_model(spec, tmp_path / 'm.pt', 'cpu')
    x, _ = inputs(spec, 30, 15)
    assert np.array_equal(model.logits(x), again.logits(x))
    for other in (*NEW, 'residual_counts', 'periodic_counts', 'periodic_counts_d10'):
        if other != name:
            with pytest.raises(ValueError):
                rc.load_model(rc.arm_spec(cfg, other), tmp_path / 'm.pt', 'cpu')
    wide = {**spec, 'workload': ['prior_records_per100', 'rest_missing']}  # context width guard
    with pytest.raises(ValueError, match='context width'):
        rc.load_model(wide, tmp_path / 'm.pt', 'cpu')


def test_embed_arms_refuse_baseline_and_tabm_archives(tmp_path):
    """Review 040c8b4 (vice versa of the strict-loader test): no frozen-baseline or TabM archive loads as an embed arm."""
    cfg, tabm = load(), tf.load(TABM)
    sources = {name: rc.arm_spec(cfg, name) for name in ('control', 'residual_counts', 'periodic_counts',
                                                          'periodic_counts_d10')}
    sources['tabm_counts'] = rc.arm_spec(tabm, 'tabm_counts')
    for name, spec in sources.items():
        small_fit(rc.build_model(spec, 3, 'cpu'), spec).save(tmp_path / f'{name}.pt')
        for embed in NEW:
            with pytest.raises(ValueError):
                rc.load_model(rc.arm_spec(cfg, embed), tmp_path / f'{name}.pt', 'cpu')


@pytest.mark.parametrize('name', list(NEW))
def test_loader_refuses_same_length_other_continuous_map(tmp_path, name):
    """Review F4 of 1b1ff30: an archive with a valid, same-length but different map is not this arm."""
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    other = [i for i in range(53) if i not in er.CONTINUOUS][:1] + er.CONTINUOUS[1:]  # swap one index
    assert len(other) == len(er.CONTINUOUS) and other != er.CONTINUOUS
    tampered = NEW[name](3, other, device='cpu')
    small_fit(tampered, spec)
    tampered.save(tmp_path / 't.pt')
    assert NEW[name].load(tmp_path / 't.pt', 'cpu').context_continuous == other  # the class loader accepts it
    with pytest.raises(ValueError, match='continuous map'):
        rc.load_model(spec, tmp_path / 't.pt', 'cpu')


def test_runner_smoke_task_cpu_and_seed_refusal(export, tmp_path):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'periodic_residual_counts', 16, 'cpu', {}, smoke=True)
    fit = json.loads((dest / 'fit.json').read_text())
    assert fit['context_width'] == 53 and fit['parameter_count'] > 0
    identity = json.loads((dest / 'identity.json').read_text())
    assert identity['seed'] == 16 and identity['arm']['embedding'] == 'periodic'
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    with np.load(dest / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(axis=1), 1.) and p['dev'].shape[1] == 10
    for seed in (0, 3, 8, 13, 19):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'linear_residual_counts', seed, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(cfg, path, digest, tmp_path / 'real', 'linear_residual_counts', 16, 'cpu', {})


# ------------------------------------------------------------------ scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # synthetic speed only; the scorer default stays 10,000


def test_scorer_reports_all_contrasts_with_sign(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    result = tf.run_score(tmp_path, cfg, export, tf.family(tmp_path, cfg, digest, splits, same_as_control=()))
    assert result['decision'] is None and result['ranking'] is None
    assert not {'factorial_contrasts', 'regularization_contrasts', 'workload_dropout_contrasts', 'tabm_contrasts'} \
        & set(result)
    assert result['seed_block']['seeds'] == [16, 17, 18] and result['seed_block']['repeat_tag'] is None
    assert 'post-selection' in result['seed_block']['note']
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(er.ARM_ORDER)
        block = result['embed_residual_contrasts'][split]
        assert list(block) == [c['name'] for c in er.CONTRASTS]
        vs = lambda a: arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in er.CONTRASTS:
            one = block[c['name']]
            assert (one['isolates'], one['held']) == (c['isolates'], c['held'])
            assert set(one['seeds']) == {'16', '17', '18'}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
        for metric in ('nll', 'brier'):
            d = {c['isolates']: block[c['name']]['ensemble'][metric]['delta'] for c in er.CONTRASTS}
            assert d['input'] == pytest.approx(d['embedding_type'] + d['embedding'], abs=1e-9)
            # review 040c8b4: design note identity 1 = 2 + 3 (bundle = structure + head dropout, same seeds)
            assert d['backbone_bundle'] == pytest.approx(d['backbone_structure'] + d['head_dropout'], abs=1e-9)
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert 'periodic_residual_counts_vs_linear_residual_counts' in table and '독립은 넷' in table and 'backbone_bundle = backbone_structure + head_dropout' in table


def test_scorer_refuses_incomplete_or_missing_or_mixed_contrasts(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=(('linear_residual_counts', 18),))
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    bare = copy.deepcopy(cfg)
    del bare['comparison'][er.CONTRAST_KEY]
    with pytest.raises(ValueError, match='fixed contrasts'):
        tf.run_score(tmp_path, bare, export, tmp_path / 'no-family-needed')
    for key, value in ((wr.CONTRAST_KEY, wr.CONTRASTS), (tr.CONTRAST_KEY, tr.CONTRASTS)):
        both = copy.deepcopy(cfg)
        both['comparison'][key] = value
        with pytest.raises(ValueError):
            tf.run_score(tmp_path, both, export, tmp_path / 'no-family-needed')
    assert not (tmp_path / 'score').exists()
