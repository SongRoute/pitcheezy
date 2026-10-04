"""Synthetic CPU tests: TabM-mini x periodic-residual runner/scorer integration (DRAFT-CUDA-TABM-RESIDUAL-20261004)."""
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
from pitchmdp import direct_ema_registry as emr  # noqa: E402
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_tabm_registry as tr  # noqa: E402
from pitchmdp import direct_tabm_residual_model as trm  # noqa: E402
from pitchmdp import direct_tabm_residual_registry as reg  # noqa: E402
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402
from pitchmdp.direct_ema_model import FlattenEMAModel  # noqa: E402
from pitchmdp.direct_frontier_model import FlattenRegularizedModel  # noqa: E402
from pitchmdp.direct_periodic_residual_model import DirectPeriodicResidualModel  # noqa: E402
from pitchmdp.direct_tabm_model import DirectTabMModel  # noqa: E402
from pitchmdp.direct_tabm_residual_model import DirectTabMResidualModel  # noqa: E402
from pitchmdp.matrix_models import MatrixModel  # noqa: E402

RUN = REPO / 'configs' / 'DRAFT-CUDA-TABM-RESIDUAL-20261004.json'
P31, P32, P33 = (REPO / 'configs' / f'EXP-P3{i}-001.json' for i in (1, 2, 3))
DRAFTS = [REPO / 'configs' / n for n in ('DRAFT-CUDA-TABM-MINI-20261004.json', 'DRAFT-CUDA-EMBED-RESIDUAL-20261004.json',
                                          'DRAFT-CUDA-EMA-20261004.json')]
EARLIER = [REPO / 'configs' / n for n in ('DRAFT-CUDA-DIRECT-20261004.json', 'DRAFT-CUDA-FRONTIER-RUN-20261004.json',
                                           'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json', 'DRAFT-CUDA-DROPOUT-RUN-20261004.json',
                                           'DRAFT-CUDA-WORKLOAD-DROPOUT-20261004.json', 'EXP-P30-001.json')]
COLUMNS = ['prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing']
NEW = 'tabm_residual_counts'


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    """Mac shares the machine with live recording: keep focused tests on one native thread."""
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


def load(registered=False, unpin=True):
    return tf.load(RUN, registered, unpin)


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


def keys_deep(value):
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_deep(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_deep(v)}
    return set()


# ------------------------------------------------------------------ config / registry

def test_config_is_draft_exact_5_arm_family_on_seeds_24_28_with_cap_60():
    cfg, ema = read(RUN), read(DRAFTS[2])
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'repeat_tag' not in cfg
    assert 'registration' not in cfg and cfg['experiment_id'] == 'DRAFT-CUDA-TABM-RESIDUAL-20261004'
    assert [a['name'] for a in cfg['arms']] == list(reg.ARM_ORDER) == ['control', 'periodic_residual_counts',
                                                                        'tabm_counts', NEW, 'tabm_counts_d10']
    assert cfg['seeds'] == [24, 25, 26, 27, 28] and cfg['fits'] == 25 == 5 * len(cfg['seeds'])
    assert cfg['training'] == {**rc.TRAINING, 'epochs': 60} == reg.TRAINING and reg.EPOCH_CAP == 60
    assert cfg['export'] == ema['export'] and cfg['export']['manifest_sha256'].startswith('49097109')
    assert cfg['export']['workload_columns'] == COLUMNS and cfg['export']['last_year'] == 2025
    assert cfg['calibration'] == ema['calibration'] and cfg['device'] == read(P33)['device']
    assert cfg['device']['allowed_gpu_uuids'] == ['GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9',
                                                  'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99']
    q = cfg['queue']
    assert (q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']) == (3600, 36000, 2)
    assert 'PLACEHOLDER' in q['date_note'] and 're-dated' in q['date_note']
    assert '/home/sds733/pitcheezy-autonomy/gpu-locks' in q['start_after'] and 'EXP-P33-001' in q['start_after']
    assert cfg['comparison'][reg.CONTRAST_KEY] == reg.CONTRASTS and len(reg.CONTRASTS) == 7
    assert {k for k in cfg['comparison'] if k.endswith('_contrasts')} == {reg.CONTRAST_KEY}
    for phrase in ('repeatedly exposed', '<=2025', 'post-selection', 'Descriptive only', 'No 2026'):
        assert phrase in cfg['exposure'], phrase
    assert 'No 2026 rows' in cfg['data_rules']
    assert not {'winner', 'promotion', 'promote', 'verdict', 'ranking', 'selected'} & keys_deep(cfg)
    for source in cfg['source_configs'].values():
        assert source['sha256'] == rc.sha256_file(REPO / source['path']), source['path']
    assert cfg['source_configs']['design_note']['path'] == \
        'experiments/pitchmdp/docs/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md'


def test_earlier_arms_equal_the_committed_registered_entries():
    arms = {a['name']: a for a in read(RUN)['arms']}
    p31, p32 = ({a['name']: a for a in read(p)['arms']} for p in (P31, P32))
    assert reg.same(arms['periodic_residual_counts'], p32['periodic_residual_counts'])
    assert reg.same(arms['tabm_counts'], p31['tabm_counts'])
    assert reg.same(arms['tabm_counts_d10'], p31['tabm_counts_d10'])  # Amendment 1 arm 5
    assert reg.same(arms['control'], p31['control']) and reg.same(arms['control'], p32['control'])
    for name, spec in reg.BASELINES.items():
        assert reg.same(spec, arms[name])
    new = arms[NEW]
    assert new['tabm_residual'] is True and new['workload'] == ['prior_records_per100'] == reg.WORKLOAD
    assert new['context_continuous'] == p32['periodic_residual_counts']['context_continuous'] == wr.CONTINUOUS['counts']
    assert (new['k'], new['scaling_init']) == (p31['tabm_counts']['k'], p31['tabm_counts']['scaling_init'])


def test_registry_constants_match_the_torch_module():
    assert (reg.K, reg.SCALING_INIT) == (trm.ENSEMBLE['k'], trm.ENSEMBLE['scaling_init'])
    assert reg.ARMS[NEW]['model'] == trm.KIND == DirectTabMResidualModel.KIND
    assert reg.ARMS[NEW]['width'] == trm.ARCHITECTURE['backbone']['width']
    assert reg.FACTORS[NEW][3] == trm.ARCHITECTURE['backbone']['dropout']


def test_seed_block_is_disjoint_from_every_earlier_block():
    earlier = {0, 1, 2, *wr.FIRST_SEEDS, *wr.SEED_BLOCKS['repeat2_seeds_8_12'], *tr.SEEDS, *er.SEEDS, *emr.SEEDS}
    assert earlier == set(range(24)) and not set(reg.SEEDS) & earlier
    for path in (*DRAFTS, *EARLIER, P31, P32, P33):
        assert not set(read(path)['seeds']) & set(reg.SEEDS)


@pytest.mark.parametrize('seeds', [[24, 25, 26], [24, 25, 26, 27], [24, 25, 26, 27, 28, 29], [28, 27, 26, 25, 24],
                                   [19, 20, 21, 22, 23], [0, 1, 2], [24.0, 25, 26, 27, 28]])
def test_own_check_seeds_accepts_only_24_28(seeds):
    cfg = load()
    assert reg.check_seeds(cfg) == [24, 25, 26, 27, 28]
    with pytest.raises(ValueError):
        reg.check_seeds({**cfg, 'seeds': seeds})
    for other in (tr, er, emr):  # no earlier registry accepts this block
        with pytest.raises(ValueError):
            other.check_seeds(cfg)


@pytest.mark.parametrize('edit', [
    lambda c: c['training'].update(epochs=30),
    lambda c: c['training'].update(epochs=61),
    lambda c: c['training'].update(patience=10),
    lambda c: c['training'].update(learning_rate=.001),
    lambda c: c['arms'][3].update(k=4),
    lambda c: c['arms'][3].update(k=8.0),
    lambda c: c['arms'][3].update(scaling_init='random-signs'),
    lambda c: c['arms'][3].update(head_dropout=0.),                                   # unchecked extra parameter
    lambda c: c['arms'][3].update(width=256),
    lambda c: c['arms'][3].update(workload=['prior_records_per100', 'rest_missing']),
    lambda c: c['arms'][3].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][3].update(priority=2),
    lambda c: c['arms'][3].pop('tabm_residual'),
    lambda c: c['arms'][3].update(tabm_residual=1),
    lambda c: c['arms'][1].update(tabm_residual=True),                                # earlier arm flagged
    lambda c: c['arms'][1].update(embedding='linear'),                                # earlier arm altered
    lambda c: c['arms'][2].update(head_dropout=.1),
    lambda c: c['arms'][2].update(workload=[]),
    lambda c: c['arms'][4].update(head_dropout=0.),                                   # arm 5 altered
    lambda c: c['arms'][4].update(tabm_residual=True),
    lambda c: c['arms'].insert(2, c['arms'].pop(4)),                                  # arm order
    lambda c: c.update(fits=20),
    lambda c: c['arms'][0].update(workload=['prior_records_per100']),
    lambda c: c['arms'].pop(),
    lambda c: c['arms'].pop(1),
    lambda c: c['arms'].insert(1, c['arms'].pop(3)),
    lambda c: c['arms'].append({**c['arms'][3], 'name': 'tabm_residual_block'}),
    lambda c: c.update(seeds=[24, 25, 26]),
    lambda c: c.update(seeds=[19, 20, 21, 22, 23]),
    lambda c: c.update(repeat_tag='x'),
    lambda c: c.update(fits=15),
    lambda c: c['device'].update(max_parallel=3),
    lambda c: c['export'].update(last_year=2026),
])
def test_config_check_refuses_any_family_seed_or_cap_drift(edit):
    cfg = copy.deepcopy(load())
    edit(cfg)
    with pytest.raises(ValueError):
        rc.config_check(cfg)


@pytest.mark.parametrize('path', [*EARLIER, *DRAFTS, P31, P32, P33], ids=lambda p: p.name)
def test_other_families_unchanged_and_refuse_tabm_residual_seeds_arms_cap_or_key(path):
    cfg = read(path)
    before = copy.deepcopy(cfg)
    checked = rc.config_check(copy.deepcopy(cfg))
    assert checked == before and cfg['training'] == rc.TRAINING == rc.training_budget(cfg)
    assert rc.training_budget(cfg) is rc.TRAINING and cfg['training']['epochs'] == 30
    assert not rc.uses_tabm_residual(cfg) and reg.check_contrasts(cfg) is None
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'seeds': list(reg.SEEDS)})
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'training': dict(reg.TRAINING)})
    arm = {'name': NEW, 'priority': 1, reg.FLAG: True, 'workload': reg.WORKLOAD, **reg.ARMS[NEW]}
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'arms': [*cfg['arms'], arm]})
    foreign = copy.deepcopy(cfg)
    foreign.setdefault('comparison', {})[reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
    with pytest.raises(ValueError):
        reg.check_contrasts(foreign)


def test_registered_p31_p32_p33_keep_their_family_check_seeds_and_dispatch():
    expected = {P31: (tr, [13, 14, 15], {'tabm_counts': DirectTabMModel}),
                P32: (er, [16, 17, 18], {'periodic_residual_counts': DirectPeriodicResidualModel}),
                P33: (emr, [19, 20, 21, 22, 23], {'ema': FlattenEMAModel, 'dropout': FlattenRegularizedModel})}
    for path, (registry, seeds, types) in expected.items():
        cfg = read(path)
        assert cfg['status'] == 'REGISTERED' and rc.config_check(copy.deepcopy(cfg))['seeds'] == seeds
        assert registry.check_contrasts(cfg) == registry.CONTRASTS
        assert type(rc.build_model(rc.arm_spec(cfg, 'control'), seeds[0], 'cpu')) is MatrixModel
        for name, cls in types.items():
            assert type(rc.build_model(rc.arm_spec(cfg, name), seeds[0], 'cpu')) is cls


def test_scorer_refuses_tabm_residual_contrasts_on_a_registered_other_family_before_any_read(tmp_path):
    for path in (P31, P32, P33):
        cfg = read(path)
        cfg['comparison'][reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
        cpath, csha = tf.write_cfg(tmp_path, cfg, name=path.name)
        with pytest.raises(ValueError, match='TabM-residual family must be exactly'):
            sc.score(cpath, csha, tmp_path / 'no-export', '0' * 64, tmp_path / 'no-family', tmp_path / 'score')
    assert not (tmp_path / 'score').exists()


def test_contrast_list_matches_preregistration_and_changes_declared_factors():
    cfg = load()
    assert reg.check_contrasts(cfg) == reg.CONTRASTS
    assert [(c['candidate'], c['reference']) for c in reg.CONTRASTS] == [
        (NEW, 'periodic_residual_counts'), (NEW, 'tabm_counts_d10'), (NEW, 'tabm_counts'), ('tabm_counts', 'control'),
        ('tabm_counts_d10', 'control'), ('periodic_residual_counts', 'control'), (NEW, 'control')]
    assert all(c['name'] == f"{c['candidate']}_vs_{c['reference']}" for c in reg.CONTRASTS)
    assert reg.CONTRASTS[0]['isolates'] == 'ensemble_head' and reg.ISOLATES['ensemble_head'] == {'head'}
    assert reg.CONTRASTS[1]['isolates'] == 'backbone_bundle_d10'  # dropout .1 both sides; backbone + placement (disclosed)
    assert reg.ISOLATES['backbone_bundle_d10'] == {'backbone', 'dropout_position'}
    assert reg.CONTRASTS[1]['held']['dropout_rate'] == .1
    assert reg.CONTRASTS[2]['isolates'] == 'backbone_bundle'  # backbone + dropout together (disclosed)


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0], name='again')),
    lambda cs: cs[0].update(candidate='periodic_residual_counts', reference=NEW),   # reversed sign
    lambda cs: cs[1].update(isolates='ensemble_head'),
    lambda cs: cs[1].update(isolates='backbone_bundle'),                            # d10 pair claims a dropout-rate change
    lambda cs: cs[1].update(held={**cs[1]['held'], 'dropout_rate': 0.}),
    lambda cs: cs.pop(1),                                                           # Amendment 1 pair dropped
    lambda cs: cs[0].update(held={'backbone': 'flatten'}),
    lambda cs: cs[0].update(winner=True),
    lambda cs: cs.pop(),
    lambda cs: cs.reverse(),
])
def test_contrast_list_is_fixed(edit):
    cfg = copy.deepcopy(load(registered=True))
    edit(cfg['comparison'][reg.CONTRAST_KEY])
    with pytest.raises(ValueError):
        reg.check_contrasts(cfg)


def test_modules_pinned_and_config_check_needs_no_torch():
    assert {'pitchmdp/direct_tabm_residual_model.py', 'pitchmdp/direct_tabm_residual_factory.py',
            'pitchmdp/direct_tabm_residual_registry.py'} <= set(rc.MODEL_SOURCES)
    assert rc.MODEL_SOURCES[-2:] == ['pitchmdp/direct_frontier_model.py', 'pitchmdp/direct_frontier_registry.py']
    assert 'pitchmdp/direct_tabm_residual_registry.py' in sc.SCORER_SOURCES
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'rc.config_check(json.load(open({str(RUN)!r}))); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------ factory

def inputs(arm, n, seed):
    a = tf.split_arrays(n, seed, 0)
    return rc.arm_inputs(a, arm, COLUMNS), a['y']


def small_fit(model, arm, epochs=3):
    train, y = inputs(arm, 300, 11)
    early, ey = inputs(arm, 90, 12)
    return model.fit(train, y, early, ey, epochs=epochs, patience=2, batch_size=64)


def test_earlier_arms_keep_old_code_paths_and_new_arm_builds_tabm_residual():
    cfg = load()
    assert type(rc.build_model(rc.arm_spec(cfg, 'control'), 24, 'cpu')) is MatrixModel
    assert type(rc.build_model(rc.arm_spec(cfg, 'periodic_residual_counts'), 24, 'cpu')) is DirectPeriodicResidualModel
    assert type(rc.build_model(rc.arm_spec(cfg, 'tabm_counts'), 24, 'cpu')) is DirectTabMModel
    spec = rc.arm_spec(cfg, NEW)
    model = small_fit(rc.build_model(spec, 24, 'cpu'), spec)
    assert type(model) is DirectTabMResidualModel and model.kind == spec['model'] and model.seed == 24
    assert model.architecture == {**trm.ARCHITECTURE, 'k': 8, 'scaling_init': 'normal'}
    assert model.net.config['n_context'] == 53 and model.context_continuous == reg.CONTINUOUS
    assert model.net.scaling.shape[0] == 8 and model.report['epochs_run'] <= 3
    x, _ = inputs(spec, 40, 13)
    assert model.logits(x).shape == (40, 10)
    with pytest.raises(ValueError):
        rc.build_model(spec, 24, 'cpu', ['x'] * 52)


def test_checkpoint_roundtrip_and_strict_loader(tmp_path):
    cfg = load()
    spec = rc.arm_spec(cfg, NEW)
    model = small_fit(rc.build_model(spec, 25, 'cpu'), spec)
    model.save(tmp_path / 'm.pt')
    again = rc.load_model(spec, tmp_path / 'm.pt', 'cpu')
    x, _ = inputs(spec, 30, 15)
    assert type(again) is DirectTabMResidualModel and np.array_equal(model.logits(x), again.logits(x))
    assert again.report == model.report
    for other in ('control', 'periodic_residual_counts', 'tabm_counts'):
        with pytest.raises(Exception):
            rc.load_model(rc.arm_spec(cfg, other), tmp_path / 'm.pt', 'cpu')
    wide = {**spec, 'workload': ['prior_records_per100', 'rest_missing']}  # context width guard
    with pytest.raises(ValueError, match='context width'):
        rc.load_model(wide, tmp_path / 'm.pt', 'cpu')


def test_new_arm_refuses_parent_and_control_archives(tmp_path):
    cfg = load()
    for name in ('control', 'periodic_residual_counts', 'tabm_counts'):
        spec = rc.arm_spec(cfg, name)
        small_fit(rc.build_model(spec, 26, 'cpu'), spec).save(tmp_path / f'{name}.pt')
        with pytest.raises(ValueError):
            rc.load_model(rc.arm_spec(cfg, NEW), tmp_path / f'{name}.pt', 'cpu')


def test_loader_refuses_same_length_other_continuous_map_and_other_k(tmp_path):
    cfg = load()
    spec = rc.arm_spec(cfg, NEW)
    other = [i for i in range(53) if i not in reg.CONTINUOUS][:1] + reg.CONTINUOUS[1:]  # swap one index
    assert len(other) == len(reg.CONTINUOUS) and other != reg.CONTINUOUS
    tampered = small_fit(DirectTabMResidualModel(3, other, device='cpu'), spec)
    tampered.save(tmp_path / 't.pt')
    assert DirectTabMResidualModel.load(tmp_path / 't.pt', 'cpu').context_continuous == other  # class loader accepts
    with pytest.raises(ValueError, match='continuous map'):
        rc.load_model(spec, tmp_path / 't.pt', 'cpu')
    for kwargs in ({'k': 4}, {'scaling_init': 'random-signs'}):
        small_fit(DirectTabMResidualModel(3, reg.CONTINUOUS, device='cpu', **kwargs), spec).save(tmp_path / 'k.pt')
        with pytest.raises(ValueError, match='not this TabM-residual arm'):
            rc.load_model(spec, tmp_path / 'k.pt', 'cpu')


# ------------------------------------------------------------------ runner / epoch cap

def test_epoch_record_marks_cap_and_patience_stops():
    assert rc.epoch_record({'epochs_run': 60, 'best_epoch': 58}, 60, 5)['stopped_by'] == 'cap'
    assert rc.epoch_record({'epochs_run': 60, 'best_epoch': 60}, 60, 5)['stopped_by'] == 'cap'
    assert rc.epoch_record({'epochs_run': 60, 'best_epoch': 55}, 60, 5)['stopped_by'] == 'patience'  # exactly at cap
    assert rc.epoch_record({'epochs_run': 17, 'best_epoch': 12}, 60, 5) == {
        'cap': 60, 'patience': 5, 'epochs_run': 17, 'best_epoch': 12, 'stopped_by': 'patience'}
    for report in ({'epochs_run': 61, 'best_epoch': 60}, {'epochs_run': 17, 'best_epoch': 14},
                   {'epochs_run': 10, 'best_epoch': 11}, {'epochs_run': 5, 'best_epoch': 0}):
        with pytest.raises(ValueError):
            rc.epoch_record(report, 60, 5)


def test_runner_smoke_task_cpu_records_cap_and_refuses_other_seeds(export, tmp_path):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', NEW, 24, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    assert fit['context_width'] == 53 and fit['parameter_count'] == trm.parameter_count(53, 38, 6, reg.CONTINUOUS)
    assert fit['epoch_cap'] == {'cap': rc.SMOKE['epochs'], 'patience': 5, 'epochs_run': rc.SMOKE['epochs'],
                                'best_epoch': fit['report']['best_epoch'], 'stopped_by': 'cap'}
    identity = read(dest / 'identity.json')
    assert identity['seed'] == 24 and identity['arm']['tabm_residual'] is True and identity['arm']['k'] == 8
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    with np.load(dest / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(axis=1), 1.) and p['dev'].shape[1] == 10
    assert type(rc.load_model(rc.arm_spec(cfg, NEW), dest / 'model.pt', 'cpu')) is DirectTabMResidualModel
    for seed in (0, 3, 8, 13, 16, 19, 23, 29):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', NEW, seed, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(cfg, path, digest, tmp_path / 'real', NEW, 24, 'cpu', {})


@pytest.mark.parametrize('arm', ['control', 'periodic_residual_counts', 'tabm_counts', 'tabm_counts_d10'])
def test_every_earlier_arm_in_this_family_records_the_cap_too(export, tmp_path, arm):
    path, digest, _ = export
    dest = rc.run_task(load(), path, digest, tmp_path / 'smoke', arm, 28, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    assert fit['epoch_cap'] == rc.epoch_record(fit['report'], rc.SMOKE['epochs'], 5)
    assert fit['epoch_cap']['epochs_run'] == rc.SMOKE['epochs'] and fit['epoch_cap']['stopped_by'] == 'cap'


def test_queue_block_follows_the_ema_pattern_and_parses():
    import cuda_direct_queue as cq
    q, ema = read(RUN)['queue'], read(P33)['queue']
    assert set(q) == set(ema) | {'date_note', 'stop_rule'}
    assert {k: q[k] for k in ('timezone', 'no_new_fit_after', 'kill_own_after', 'aggregate_by', 'order')} == \
        {**{k: ema[k] for k in ('timezone', 'no_new_fit_after', 'kill_own_after', 'aggregate_by')},
         'order': 'priority then seed then arm (all 5 arms priority 1; config order)'}
    deadlines = cq.Deadlines(q)
    assert deadlines.no_new < deadlines.kill < deadlines.aggregate
    assert '1800 s' in q['stop_rule'] and 'twice' in q['stop_rule'] and '1,224 updates/epoch x 60' in q['stop_rule']


def test_other_family_fit_json_has_no_epoch_cap_record(export, tmp_path):
    """Old families write fit.json exactly as before (no new key)."""
    path, digest, _ = export
    cfg = tf.load(DRAFTS[2])
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'control', 19, 'cpu', {}, smoke=True)
    assert 'epoch_cap' not in read(dest / 'fit.json')


@pytest.mark.parametrize('arm', ['control', NEW])
def test_registered_config_epoch_cap_60_reaches_fit(export, tmp_path, monkeypatch, arm):
    """Non-smoke run on the tiny synthetic export: fit receives epochs=60 from config.training; fit.json records it."""
    path, digest, _ = export
    cfg = load(registered=True)
    cls = DirectTabMResidualModel if arm == NEW else MatrixModel
    seen, original = [], cls.fit

    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, 'fit', spy)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'real', arm, 27, 'cpu', {})
    assert [k['epochs'] for k in seen] == [60] and seen[0]['patience'] == 5 and seen[0]['batch_size'] == 1024
    fit = read(dest / 'fit.json')
    record = fit['epoch_cap']
    assert record['cap'] == 60 and record['patience'] == 5
    assert (record['epochs_run'], record['best_epoch']) == (fit['report']['epochs_run'], fit['report']['best_epoch'])
    assert record == rc.epoch_record(fit['report'], 60, 5)


# ------------------------------------------------------------------ scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # synthetic speed only; the scorer default stays 10,000


def with_epoch_records(fam, cfg, cap_hit=((NEW, 25),)):
    """Add the runner's report / epoch_cap fields to every fake task (cap hit where listed) and reseal."""
    for arm in reg.ARM_ORDER:
        for seed in cfg['seeds']:
            dest = rc.task_dir(fam, arm, seed, False)
            if not dest.exists():
                continue
            fit = read(dest / 'fit.json')
            fit['report'] = {'epochs_run': 60, 'best_epoch': 59} if (arm, seed) in cap_hit else \
                {'epochs_run': 20 + seed - 24, 'best_epoch': 15 + seed - 24}
            fit['epoch_cap'] = rc.epoch_record(fit['report'], 60, 5)
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest)
    return fam


def test_scorer_reports_all_contrasts_and_per_fit_cap_hits(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_epoch_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg)
    result = tf.run_score(tmp_path, cfg, export, fam)
    assert result['decision'] is None and result['ranking'] is None
    assert not {'factorial_contrasts', 'regularization_contrasts', 'workload_dropout_contrasts', 'tabm_contrasts',
                'embed_residual_contrasts', 'ema_contrasts'} & set(result)
    block = result['seed_block']
    assert block['seeds'] == [24, 25, 26, 27, 28] and block['repeat_tag'] is None and block['epoch_cap'] == 60
    assert 'post-selection' in block['note'] and 'not compared with' in block['note']
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(reg.ARM_ORDER)
        contrasts = result['tabm_residual_contrasts'][split]
        assert list(contrasts) == [c['name'] for c in reg.CONTRASTS]
        vs = lambda a: 0. if a == 'control' else arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in reg.CONTRASTS:
            one = contrasts[c['name']]
            assert (one['isolates'], one['held']) == (c['isolates'], c['held'])
            assert set(one['seeds']) == {str(s) for s in cfg['seeds']}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    tasks = result['provenance']['tasks']
    assert tasks[NEW]['25']['epoch_cap']['stopped_by'] == 'cap'
    assert sum(t['epoch_cap']['stopped_by'] == 'cap' for a in tasks.values() for t in a.values()) == 1
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert 'TabM-mini × 잔차 구조 대비' in table and 'epoch 상한(60) 도달 여부' in table and '25: 60/59/cap' in table


def test_scorer_refuses_missing_or_wrong_epoch_cap_records(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=())
    with pytest.raises(ValueError, match='epoch cap record'):
        tf.run_score(tmp_path, cfg, export, fam)  # fake tasks without an epoch_cap record
    fam = with_epoch_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=(), name='fam2'), cfg)
    dest = rc.task_dir(fam, 'tabm_counts', 26, False)
    fit = read(dest / 'fit.json')
    fit['epoch_cap'] = {**fit['epoch_cap'], 'cap': 30}  # e.g. a fit given the old 30 cap
    rc.write_json_atomic(dest / 'fit.json', fit)
    tf.reseal(dest)
    with pytest.raises(ValueError, match='tabm_counts/26: epoch cap record'):
        tf.run_score(tmp_path, cfg, export, fam, dest='score2')
    assert not (tmp_path / 'score').exists() and not (tmp_path / 'score2').exists()


def test_scorer_refuses_incomplete_missing_or_mixed_contrasts(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=((NEW, 28),))
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    bare = copy.deepcopy(cfg)
    del bare['comparison'][reg.CONTRAST_KEY]
    with pytest.raises(ValueError, match='fixed contrasts'):
        tf.run_score(tmp_path, bare, export, tmp_path / 'no-family-needed')
    for key, value in ((tr.CONTRAST_KEY, tr.CONTRASTS), (er.CONTRAST_KEY, er.CONTRASTS),
                       (emr.CONTRAST_KEY, emr.CONTRASTS)):
        both = copy.deepcopy(cfg)
        both['comparison'][key] = value
        with pytest.raises(ValueError):
            tf.run_score(tmp_path, both, export, tmp_path / 'no-family-needed')
    with pytest.raises(ValueError, match='REGISTERED'):
        tf.run_score(tmp_path, load(), export, tmp_path / 'no-family-needed')
    assert not (tmp_path / 'score').exists()


def test_module_hash_is_the_post_disposition_pin():
    """Wiring requirement (Amendment 1): the module pinned AFTER review disposition 0d8a65b (R1 + N1/N2)."""
    assert rc.sha256_file(PROJECT / 'pitchmdp/direct_tabm_residual_model.py') == \
        '6a9daff6c3de13a753357d8615bd9f0d5a7bfe796b350f7f7bd0a5e60c9287cf'
    assert '6a9daff6c3de13a753357d8615bd9f0d5a7bfe796b350f7f7bd0a5e60c9287cf' in read(RUN)['arm_notes']['module_pin']


def test_arm_5_dispatches_to_the_old_tabm_code_path_with_head_dropout_10():
    cfg = load()
    model = rc.build_model(rc.arm_spec(cfg, 'tabm_counts_d10'), 24, 'cpu')
    assert type(model) is DirectTabMModel and type(model) is not DirectTabMResidualModel
    assert model.architecture['head_dropout'] == .1 and model.architecture['k'] == reg.K
