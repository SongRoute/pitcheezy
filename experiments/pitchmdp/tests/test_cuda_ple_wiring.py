"""Synthetic CPU tests: PLE-embedding family wiring (DRAFT-CUDA-PLE-RESIDUAL-20261004, EXP-P35-001 DRAFT).

Registry seed/arm/contrast checks with mutants, the P35-predicate-first dispatch rule (review N1) in config_check,
training_budget, run_task and the scorer, factory round trip and refusals, CPU runner smoke tasks, the smoke-subset
(first SMOKE['train'] TRAIN rows) bins path and the scorer's one-bins-hash rule (N12).  No real data, no GPU.
"""
import copy
import hashlib
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
import export_cuda_direct as ex  # noqa: E402
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export / fake-task helpers
from pitchmdp import direct_ema_registry as emr  # noqa: E402
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_ple_model as plm  # noqa: E402
from pitchmdp import direct_ple_registry as reg  # noqa: E402
from pitchmdp import direct_tabm_registry as tr  # noqa: E402
from pitchmdp import direct_tabm_residual_registry as trr  # noqa: E402
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402
from pitchmdp.direct_periodic_residual_model import DirectPeriodicResidualModel  # noqa: E402
from pitchmdp.direct_ple_model import DirectPLEResidualModel, DirectPLETabMResidualModel  # noqa: E402
from pitchmdp.direct_tabm_residual_model import DirectTabMResidualModel  # noqa: E402
from pitchmdp.matrix_models import MatrixModel  # noqa: E402

CONFIGS = REPO / 'configs'
RUN = CONFIGS / 'DRAFT-CUDA-PLE-RESIDUAL-20261004.json'
DRAFT_AT_CONFIG = '82dd5b154997dec6c776ec8c350a2d35e494a7970efbc0b3495d4dd7a95c4786'  # DRAFT as pinned by RUN (5e8b1d0)
DRAFT_AMENDED = '9bdf9054f4e0b834a296aab7b6ea6c3cc36033940dc74d6d2e0768f728e9af8c'  # + wiring-review fixes (b1a2852)
P34, P34_DRAFT = CONFIGS / 'EXP-P34-001.json', CONFIGS / 'DRAFT-CUDA-TABM-RESIDUAL-20261004.json'
REGISTERED = [CONFIGS / f'EXP-P3{i}-001.json' for i in range(5)]  # P30-001 .. P34-001
DRAFTS = [CONFIGS / n for n in ('DRAFT-CUDA-DIRECT-20261004.json', 'DRAFT-CUDA-FRONTIER-RUN-20261004.json',
                                'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json', 'DRAFT-CUDA-DROPOUT-RUN-20261004.json',
                                'DRAFT-CUDA-WORKLOAD-DROPOUT-20261004.json', 'DRAFT-CUDA-TABM-MINI-20261004.json',
                                'DRAFT-CUDA-EMBED-RESIDUAL-20261004.json', 'DRAFT-CUDA-EMA-20261004.json',
                                'DRAFT-CUDA-TABM-RESIDUAL-20261004.json')]
SINGLE, TABM = 'ple_residual_counts', 'ple_tabm_residual_counts'
NEW = (SINGLE, TABM)
# Frozen modules: never edited by this wiring (same values as tests/test_direct_ple_model.py).
FROZEN = {'direct_periodic_model.py': '18f0b96a9b8a0d9eb39af2de4b0cc1b7a10a0dd42ca92977f726042e1aec80d3',
          'direct_periodic_residual_model.py': 'ee1b3d3f84b19f3564c853acf2bd12c640b1e5e597b8c62d485f839aeb8c992b',
          'direct_residual_model.py': '7feb9cfdcad85f4cb02c50a75e2ba2b528dd5546e600526b0334dcf0cf8e2d59',
          'direct_tabm_model.py': 'c968aec94fd7278d30dbbc53cb81fe4fcfc23c0b33b912a48ee09122e216b319',
          'direct_tabm_residual_model.py': '6a9daff6c3de13a753357d8615bd9f0d5a7bfe796b350f7f7bd0a5e60c9287cf'}


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


def p30_002():
    """EXP-P30-002 is registered on another branch (96784b5): same arms as P30-001, seeds 8-12 + its repeat_tag."""
    return {**read(REGISTERED[0]), 'experiment_id': 'EXP-P30-002', 'seeds': [8, 9, 10, 11, 12],
            'repeat_tag': 'repeat2_seeds_8_12'}


# ------------------------------------------------------------------ config / registry

def test_config_is_draft_exact_5_arm_family_on_seeds_29_33_with_cap_60():
    raw = RUN.read_text()
    cfg, p34 = json.loads(raw), read(P34)
    assert raw.isascii() and raw == json.dumps(cfg, indent=2) + '\n'
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'repeat_tag' not in cfg
    assert 'registration' not in cfg and cfg['experiment_id'] == 'DRAFT-CUDA-PLE-RESIDUAL-20261004'
    assert not (CONFIGS / 'EXP-P35-001.json').exists() and not list(CONFIGS.glob('*P35*'))
    assert [a['name'] for a in cfg['arms']] == list(reg.ARM_ORDER) == [
        'control', 'periodic_residual_counts', SINGLE, 'tabm_residual_counts', TABM]
    assert cfg['seeds'] == [29, 30, 31, 32, 33] and cfg['fits'] == 25 == 5 * len(cfg['seeds'])
    assert cfg['training'] == {**rc.TRAINING, 'epochs': 60} == reg.TRAINING == p34['training'] == trr.TRAINING
    assert cfg['export'] == p34['export'] and cfg['export']['manifest_sha256'].startswith('49097109')
    assert cfg['calibration'] == p34['calibration'] and cfg['device'] == p34['device']
    q = cfg['queue']
    assert (q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']) == (3600, 36000, 2)
    assert q['date'] == '2026-10-05' and 'PLACEHOLDER' in q['date_note'] and 're-dated' in q['date_note']
    assert 'EXP-P30-002' in q['start_after'] and 'EXP-P34-001' in q['start_after'] and 'gpu-locks' in q['start_after']
    for phrase in ('Amendment-2 method only', '<= 1800 s', '<= 36000', 'exit 0', 'twice', 'gates not raised'):
        assert phrase in q['stop_rule'], phrase
    assert '1,224 updates/epoch' not in q['stop_rule']  # the superseded Amendment-1 smoke formula (review B2)
    assert cfg['comparison'][reg.CONTRAST_KEY] == reg.CONTRASTS and len(reg.CONTRASTS) == 7
    assert {k for k in cfg['comparison'] if k.endswith('_contrasts')} == {reg.CONTRAST_KEY}
    for phrase in ('repeatedly exposed', '<=2025', 'post-selection', 'Descriptive only', 'No 2026'):
        assert phrase in cfg['exposure'], phrase
    assert 'No 2026 rows' in cfg['data_rules'] and 'PLE binning' in cfg['data_rules']
    assert not {'winner', 'promotion', 'promote', 'verdict', 'ranking', 'selected'} & keys_deep(cfg)
    for key, source in cfg['source_configs'].items():
        if key == 'design_note':
            continue
        assert source['sha256'] == rc.sha256_file(REPO / source['path']), source['path']
    note = cfg['source_configs']['design_note']
    assert note['path'] == 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P35-001-ple-embedding-2026-10-04.md'
    # The DRAFT was amended before registration (wiring review: §10.1, §5, §2 N-a, §10.7); the frozen DRAFT config pins
    # the DRAFT as committed with it (5e8b1d0). Any further DRAFT edit must update DRAFT_AMENDED in its own reviewed
    # commit, and the registered config must pin the amended DRAFT (probe review F5: no git-object dependency).
    assert note['sha256'] == DRAFT_AT_CONFIG
    assert rc.sha256_file(REPO / note['path']) == DRAFT_AMENDED
    assert reg.MODULE_SHA256 not in raw and 'MODULE_SHA256' in cfg['arm_notes']['module_pin']  # one pin only


def test_earlier_arms_equal_the_committed_p34_entries():
    arms, p34 = ({a['name']: a for a in read(p)['arms']} for p in (RUN, P34))
    for name in ('control', 'periodic_residual_counts', 'tabm_residual_counts'):
        assert reg.same(arms[name], p34[name]) and reg.same(reg.BASELINES[name], p34[name]), name
    assert arms['tabm_residual_counts'][trr.FLAG] is True  # keeps the P34 flag (why P35 dispatch goes first)
    for name in NEW:
        arm = arms[name]
        assert arm[reg.FLAGS[name]] is True and not set(reg.FLAG_KEYS) - {reg.FLAGS[name]} & set(arm)
        assert arm['workload'] == ['prior_records_per100'] == reg.WORKLOAD and arm['priority'] == 1
        assert arm['context_continuous'] == wr.CONTINUOUS['counts'] == p34['periodic_residual_counts']['context_continuous']
        assert 'tabm_residual' not in arm and 'tabm' not in arm and 'embed_residual' not in arm
    assert (arms[TABM]['k'], arms[TABM]['scaling_init']) == (p34['tabm_residual_counts']['k'],
                                                            p34['tabm_residual_counts']['scaling_init'])


def test_registry_constants_match_the_torch_module_and_sources():
    from pitchmdp import sequence_data
    from pitchmdp import direct_tabm_residual_model as trm
    assert (reg.N_BINS, reg.D_EMBEDDING) == (plm.N_BINS, plm.D_EMBEDDING) == (48, 8)
    assert reg.KINDS == {SINGLE: plm.KINDS['single'], TABM: plm.KINDS['tabm']}
    assert (reg.K, reg.SCALING_INIT) == (trm.ENSEMBLE['k'], trm.ENSEMBLE['scaling_init'])
    assert reg.ARMS[TABM]['base_model'] == trm.KIND and reg.ARMS[SINGLE]['base_model'] == 'periodic_residual_mlp'
    assert reg.PHYSICAL_CHANNELS == sequence_data.PHYSICAL_CHANNELS
    assert reg.feature_labels(6, reg.CONTINUOUS) == plm.feature_labels(6, reg.CONTINUOUS) and \
        len(reg.feature_labels(6, reg.CONTINUOUS)) == 81
    assert reg.FLAG_KEYS == rc.PLE_FLAGS == ('ple_residual', 'ple_tabm_residual')
    assert reg.FACTORS['ple_residual_counts'][3] == plm.ARCHITECTURE['backbone']['dropout']


def test_module_pin_and_frozen_modules_unchanged():
    """The PLE module is under separate review: its sha is pinned ONLY in direct_ple_registry.MODULE_SHA256."""
    assert rc.sha256_file(PROJECT / 'pitchmdp' / 'direct_ple_model.py') == reg.MODULE_SHA256
    for name, digest in FROZEN.items():
        assert rc.sha256_file(PROJECT / 'pitchmdp' / name) == digest, name
    pins = [p for p in (PROJECT / 'pitchmdp').glob('*.py') if reg.MODULE_SHA256 in p.read_text()]
    assert pins == [PROJECT / 'pitchmdp' / 'direct_ple_registry.py']


def test_seed_block_is_disjoint_from_every_earlier_block():
    earlier = {0, 1, 2, *wr.FIRST_SEEDS, *wr.SEED_BLOCKS['repeat2_seeds_8_12'], *tr.SEEDS, *er.SEEDS, *emr.SEEDS,
               *trr.SEEDS}
    assert earlier == set(range(29)) == {s for block in reg.EARLIER_SEEDS.values() for s in block}
    assert not set(reg.SEEDS) & earlier
    for path in (*DRAFTS, *REGISTERED):
        assert not set(read(path)['seeds']) & set(reg.SEEDS), path.name
    for block in reg.EARLIER_SEEDS:
        assert block in read(RUN)['seed_rule'] or block.split('-')[1] in read(RUN)['seed_rule']


@pytest.mark.parametrize('seeds', [[29, 30, 31], [29, 30, 31, 32], [29, 30, 31, 32, 33, 34], [33, 32, 31, 30, 29],
                                   [24, 25, 26, 27, 28], [0, 1, 2], [29.0, 30, 31, 32, 33], [30, 31, 32, 33, 34]])
def test_own_check_seeds_accepts_only_29_33(seeds):
    cfg = load()
    assert reg.check_seeds(cfg) == [29, 30, 31, 32, 33]
    with pytest.raises(ValueError):
        reg.check_seeds({**cfg, 'seeds': seeds})
    for other in (tr, er, emr, trr):  # no earlier registry accepts this block
        with pytest.raises(ValueError):
            other.check_seeds(cfg)


@pytest.mark.parametrize('edit', [
    lambda c: c['training'].update(epochs=30),
    lambda c: c['training'].update(epochs=61),
    lambda c: c['training'].update(patience=10),
    lambda c: c['training'].update(learning_rate=.001),
    lambda c: c['arms'][2].update(n_bins=32),
    lambda c: c['arms'][2].update(n_bins=48.0),
    lambda c: c['arms'][2].update(d_embedding=12),
    lambda c: c['arms'][2].update(embedding='periodic'),
    lambda c: c['arms'][2].update(width=256),
    lambda c: c['arms'][2].update(workload=['prior_records_per100', 'rest_missing']),
    lambda c: c['arms'][2].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][2].update(priority=2),
    lambda c: c['arms'][2].pop('ple_residual'),
    lambda c: c['arms'][2].update(ple_residual=1),
    lambda c: c['arms'][2].update(ple_tabm_residual=True),                          # both flags
    lambda c: c['arms'][2].update(ple_tabm_residual=True) or c['arms'][2].pop('ple_residual'),  # swapped flag
    lambda c: c['arms'][2].update(tabm_residual=True),                              # P34 flag on a PLE arm
    lambda c: c['arms'][4].update(k=4),
    lambda c: c['arms'][4].update(scaling_init='random-signs'),
    lambda c: c['arms'][4].pop('ple_tabm_residual'),
    lambda c: c['arms'][4].update(ple_residual=True),
    lambda c: c['arms'][1].update(ple_residual=True),                               # earlier arm flagged
    lambda c: c['arms'][1].update(embedding='linear'),                              # earlier arm altered
    lambda c: c['arms'][3].update(k=4),
    lambda c: c['arms'][3].pop('tabm_residual'),
    lambda c: c['arms'][3].update(ple_tabm_residual=True),
    lambda c: c['arms'][0].update(workload=['prior_records_per100']),
    lambda c: c['arms'].insert(2, c['arms'].pop(3)),                                # arm order
    lambda c: c['arms'].pop(),
    lambda c: c['arms'].pop(2),
    lambda c: c['arms'].append({**c['arms'][2], 'name': 'ple_residual_block'}),
    lambda c: c.update(fits=20),
    lambda c: c.update(seeds=[29, 30, 31]),
    lambda c: c.update(seeds=[24, 25, 26, 27, 28]),
    lambda c: c.update(repeat_tag='x'),
    lambda c: c['device'].update(max_parallel=3),
    lambda c: c['export'].update(last_year=2026),
])
def test_config_check_refuses_any_family_seed_flag_or_cap_drift(edit):
    cfg = copy.deepcopy(load())
    edit(cfg)
    with pytest.raises(ValueError):
        rc.config_check(cfg)


@pytest.mark.parametrize('change', [{'epochs': 30}, {'epochs': 60.0}, {'patience': 6}, {'batch_size': 512},
                                    {'hpo_trials': 1}, {'loss_weighting': 'weighted'}])
def test_registry_family_check_alone_refuses_budget_drift(change):
    """The registry's own training check (not only the runner's training_budget comparison) refuses drift."""
    cfg = copy.deepcopy(load())
    cfg['training'].update(change)
    with pytest.raises(ValueError, match='cuda_direct_v1 budget with epochs 60'):
        reg.check_family(cfg)


def test_contrast_list_matches_the_draft_order_and_changes_declared_factors():
    cfg = load()
    assert reg.check_contrasts(cfg) == reg.CONTRASTS
    assert [(c['candidate'], c['reference'], c['isolates']) for c in reg.CONTRASTS] == [
        (SINGLE, 'periodic_residual_counts', 'embedding_type_single'),
        (TABM, 'tabm_residual_counts', 'embedding_type_tabm'), (TABM, SINGLE, 'ensemble_head_ple'),
        ('periodic_residual_counts', 'control', 'bundle'), (SINGLE, 'control', 'bundle'),
        ('tabm_residual_counts', 'control', 'bundle'), (TABM, 'control', 'bundle')]
    assert all(c['name'] == f"{c['candidate']}_vs_{c['reference']}" for c in reg.CONTRASTS)
    assert list(reg.HEADLINE) == [c['name'] for c in reg.CONTRASTS[:2]]
    assert reg.CHANGED[reg.CONTRASTS[0]['name']] == reg.CHANGED[reg.CONTRASTS[1]['name']] == {'input'}
    assert reg.CONTRASTS[0]['held']['head'] == 'single' and reg.CONTRASTS[1]['held']['head'] == 'tabm_mini'
    assert reg.CONTRASTS[2]['held']['input'] == 'ple'


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0], name='again')),
    lambda cs: cs[0].update(candidate='periodic_residual_counts', reference=SINGLE),  # reversed sign
    lambda cs: cs[0].update(isolates='bundle'),
    lambda cs: cs[3].update(isolates='embedding_type_single'),
    lambda cs: cs[0].update(held={**cs[0]['held'], 'head': 'tabm_mini'}),
    lambda cs: cs[2].update(held={**cs[2]['held'], 'input': 'periodic'}),
    lambda cs: cs[1].update(reference='periodic_residual_counts'),
    lambda cs: cs.pop(2),
    lambda cs: cs[0].update(winner=True),
    lambda cs: cs.pop(),
    lambda cs: cs.insert(0, cs.pop(2)),                                              # headline order
    lambda cs: cs.reverse(),
])
def test_contrast_list_is_fixed(edit):
    cfg = copy.deepcopy(load(registered=True))
    edit(cfg['comparison'][reg.CONTRAST_KEY])
    with pytest.raises(ValueError):
        reg.check_contrasts(cfg)


def test_contrast_factor_check_catches_a_registry_level_mutation(monkeypatch):
    """Mutant: the expected changed set of an embedding_type row widened -> the factor check refuses."""
    monkeypatch.setitem(reg.CHANGED, reg.HEADLINE[0], {'input', 'head'})
    with pytest.raises(ValueError, match='declared factors'):
        reg.check_contrasts(load())


# ------------------------------------------------------------------ dispatch: P35 predicate first (review N1)

def forbid_other_family_checks(monkeypatch):
    """Any route of a P35 config into an earlier family-level check or the TabM-residual predicate fails loudly."""
    def boom(*args, **kwargs):
        raise AssertionError('P35 config reached an earlier family route')
    for module in (trr, emr, tr, er, wr):
        monkeypatch.setattr(module, 'check_family', boom)
    monkeypatch.setattr(rc, 'uses_tabm_residual', boom)
    monkeypatch.setattr(rc, 'uses_ema', boom)
    monkeypatch.setattr(rc, 'uses_tabm', boom)


def test_p35_config_never_reaches_the_p34_check_family_or_budget(monkeypatch):
    cfg = load()
    forbid_other_family_checks(monkeypatch)
    assert rc.config_check(copy.deepcopy(cfg))['seeds'] == reg.SEEDS
    assert rc.training_budget(cfg) is reg.TRAINING and rc.uses_epoch_cap(cfg)


def test_p35_family_is_refused_by_the_p34_check_it_must_never_reach():
    """Why order matters: the P34 family check would reject this config (wrong arms / seeds)."""
    cfg = load()
    assert rc.uses_tabm_residual(cfg) and rc.uses_ple(cfg)
    with pytest.raises(ValueError, match='TabM-residual family must be exactly'):
        trr.check_family(cfg)


@pytest.mark.parametrize('path', [*DRAFTS, *REGISTERED, 'p30_002'], ids=lambda p: getattr(p, 'name', p))
def test_earlier_families_route_exactly_as_before(path, monkeypatch):
    cfg = p30_002() if path == 'p30_002' else read(path)
    before = copy.deepcopy(cfg)
    budget = trr.TRAINING if rc.uses_tabm_residual(cfg) else rc.TRAINING

    def boom(*args, **kwargs):
        raise AssertionError('earlier family reached the P35 route')
    monkeypatch.setattr(reg, 'check_family', boom)
    monkeypatch.setattr(reg, 'check_arm', boom)
    assert rc.config_check(copy.deepcopy(cfg)) == before
    assert not rc.uses_ple(cfg) and rc.training_budget(cfg) is budget and reg.check_contrasts(cfg) is None
    assert rc.uses_epoch_cap(cfg) == rc.uses_tabm_residual(cfg) == (cfg['seeds'] == trr.SEEDS)


@pytest.mark.parametrize('path', [*DRAFTS, *REGISTERED], ids=lambda p: p.name)
def test_earlier_families_refuse_ple_seeds_arms_cap_or_key(path):
    cfg = read(path)
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'seeds': list(reg.SEEDS)})
    arm = {'name': SINGLE, 'priority': 1, 'ple_residual': True, 'workload': reg.WORKLOAD, **reg.ARMS[SINGLE]}
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'arms': [*cfg['arms'], arm]})
    foreign = copy.deepcopy(cfg)
    foreign.setdefault('comparison', {})[reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
    with pytest.raises(ValueError):
        reg.check_contrasts(foreign)


def test_registered_p34_keeps_its_route_and_dispatch():
    cfg = read(P34)
    assert rc.config_check(copy.deepcopy(cfg))['seeds'] == trr.SEEDS and rc.training_budget(cfg) is trr.TRAINING
    assert trr.check_contrasts(cfg) == trr.CONTRASTS and reg.check_contrasts(cfg) is None
    assert type(rc.build_model(rc.arm_spec(cfg, 'tabm_residual_counts'), 24, 'cpu')) is DirectTabMResidualModel
    assert type(rc.build_model(rc.arm_spec(cfg, 'periodic_residual_counts'), 24, 'cpu')) is DirectPeriodicResidualModel


def test_build_dispatch_of_every_p35_arm():
    cfg = load()
    expected = {'control': MatrixModel, 'periodic_residual_counts': DirectPeriodicResidualModel,
                SINGLE: DirectPLEResidualModel, 'tabm_residual_counts': DirectTabMResidualModel,
                TABM: DirectPLETabMResidualModel}
    for name, cls in expected.items():
        model = rc.build_model(rc.arm_spec(cfg, name), 29, 'cpu')
        assert type(model) is cls, name
    assert rc.build_model(rc.arm_spec(cfg, TABM), 29, 'cpu').architecture == {
        **plm.TABM_ARCHITECTURE, 'k': 8, 'scaling_init': 'normal'}
    with pytest.raises(ValueError, match='no context names'):
        rc.build_model(rc.arm_spec(cfg, SINGLE), 29, 'cpu', ['x'] * 52)


def test_scorer_refuses_ple_contrasts_on_registered_other_families_before_any_read(tmp_path):
    for path in REGISTERED[1:]:
        cfg = read(path)
        cfg['comparison'][reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
        cpath, csha = tf.write_cfg(tmp_path, cfg, name=path.name)
        with pytest.raises(ValueError, match='PLE family must be exactly'):
            sc.score(cpath, csha, tmp_path / 'no-export', '0' * 64, tmp_path / 'no-family', tmp_path / 'score')
    assert not (tmp_path / 'score').exists()


def test_config_check_and_scorer_import_need_no_torch():
    assert {'pitchmdp/direct_ple_model.py', 'pitchmdp/direct_ple_factory.py',
            'pitchmdp/direct_ple_registry.py'} <= set(rc.MODEL_SOURCES)
    assert rc.MODEL_SOURCES[-2:] == ['pitchmdp/direct_frontier_model.py', 'pitchmdp/direct_frontier_registry.py']
    assert 'pitchmdp/direct_ple_registry.py' in sc.SCORER_SOURCES
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'c = json.load(open({str(RUN)!r})); rc.config_check(c); rc.training_budget(c); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------ fit.json PLE record

def test_column_name_maps_labels_to_export_columns():
    names = read(RUN)['export']['context_names']
    assert reg.column_name('past0.phys0', 6, names, reg.WORKLOAD) == 'past0.effective_speed'
    assert reg.column_name('past4.phys7', 6, names, reg.WORKLOAD) == 'past4.plate_z'
    assert reg.column_name('context40', 6, names, reg.WORKLOAD) == names[40]
    assert reg.column_name('context52', 6, names, reg.WORKLOAD) == 'prior_records_per100'
    for bad in ('past5.phys0', 'past0.phys8', 'context53', 'context-1', 'ctx1', 'past0.phys0 ', 'context1x'):
        with pytest.raises(ValueError):
            reg.column_name(bad, 6, names, reg.WORKLOAD)


ALLOCATED = {SINGLE: 275_738, TABM: 291_600}  # module tests / review: allocated counts, independent of T_j


def network_record(counts, digest='ab' * 32, allocated=ALLOCATED[SINGLE]):
    labels = reg.feature_labels(6, reg.CONTINUOUS)
    return {'context_continuous': reg.CONTINUOUS, 'n_bins': 48, 'ple_bins_sha256': digest, 'ple_n_bins': list(counts),
            'ple_degenerate_columns': [label for label, t in zip(labels, counts) if t == 0],
            'ple_effective_parameter_count': allocated - 8 * (len(counts) * 48 - sum(counts))}


def test_fit_record_rederives_and_refuses_inconsistent_records():
    names = read(RUN)['export']['context_names']
    counts = [48] * 81
    counts[2], counts[40 + 21] = 0, 0  # past0.phys2 and the context column at continuous position 21
    record = reg.fit_record(network_record(counts), 6, names, reg.WORKLOAD)
    assert record['ple_degenerate_columns'] == ['past0.spin_axis_sin', names[reg.CONTINUOUS[21]]]
    assert record['ple_degenerate_count'] == 2 and record['ple_n_bins'] == counts
    for edit in (lambda n: n.update(ple_bins_sha256='AB' * 32), lambda n: n.update(ple_bins_sha256='ab' * 31),
                 lambda n: n.update(ple_n_bins=counts[:80]), lambda n: n.update(ple_n_bins=[*counts[:80], 49]),
                 lambda n: n.update(ple_n_bins=[*counts[:80], 48.0]), lambda n: n.update(ple_degenerate_columns=[]),
                 lambda n: n.update(ple_degenerate_columns=n['ple_degenerate_columns'][::-1]),
                 lambda n: n.update(n_bins=32), lambda n: n.pop('ple_bins_sha256')):
        network = network_record(counts)
        edit(network)
        with pytest.raises(ValueError):
            reg.fit_record(network, 6, names, reg.WORKLOAD)


# ------------------------------------------------------------------ factory

def inputs(arm, n, seed):
    a = tf.split_arrays(n, seed, 0)
    return rc.arm_inputs(a, arm, tf.COLUMNS), a['y']


def small_fit(model, arm, epochs=2):
    train, y = inputs(arm, 300, 11)
    early, ey = inputs(arm, 90, 12)
    return model.fit(train, y, early, ey, epochs=epochs, patience=2, batch_size=64)


def saved(tmp_path, name, model):
    """model.pt plus the run's fit.json record, as run_task leaves them."""
    dest = tmp_path / name
    dest.mkdir()
    model.save(dest / 'model.pt')
    rc.write_json_atomic(dest / 'fit.json', {'ple_bins_sha256': model.net.config['ple_bins_sha256']})
    return dest / 'model.pt'


@pytest.mark.parametrize('name', NEW)
def test_factory_round_trip_and_recorded_hash(tmp_path, name):
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    model = small_fit(rc.build_model(spec, 30, 'cpu'), spec)
    path = saved(tmp_path, 'task', model)
    again = rc.load_model(spec, path, 'cpu')
    x, _ = inputs(spec, 30, 15)
    assert type(again) is type(model) and np.array_equal(model.logits(x), again.logits(x))
    assert again.report == model.report and again.context_continuous == reg.CONTINUOUS
    assert again.net.config['ple_bins_sha256'] == plm.bins_sha256(plm.compute_bins(
        tuple(np.asarray(a) for a in inputs(spec, 300, 11)[0]), reg.CONTINUOUS))
    explicit = rc.load_model(spec, path, 'cpu', ple_bins_sha256=model.net.config['ple_bins_sha256'])
    assert np.array_equal(explicit.logits(x), model.logits(x))
    with pytest.raises(ValueError, match='recorded by the run'):
        rc.load_model(spec, path, 'cpu', ple_bins_sha256='0' * 64)
    rc.write_json_atomic(path.parent / 'fit.json', {'ple_bins_sha256': '1' * 64})
    with pytest.raises(ValueError, match='recorded by the run'):
        rc.load_model(spec, path, 'cpu')
    rc.write_json_atomic(path.parent / 'fit.json', {})
    with pytest.raises(ValueError, match='recorded by the run'):
        rc.load_model(spec, path, 'cpu')
    (path.parent / 'fit.json').unlink()
    with pytest.raises(ValueError, match='fit.json'):
        rc.load_model(spec, path, 'cpu')
    wide = {**spec, 'workload': ['prior_records_per100', 'rest_missing']}
    with pytest.raises(ValueError, match='context width'):
        rc.load_model(wide, path, 'cpu', ple_bins_sha256=model.net.config['ple_bins_sha256'])


def test_factory_refuses_other_arm_archives_both_ways(tmp_path):
    cfg = load()
    paths = {}
    for name in ('control', 'periodic_residual_counts', 'tabm_residual_counts', *NEW):
        spec = rc.arm_spec(cfg, name)
        paths[name] = saved(tmp_path, name, small_fit(rc.build_model(spec, 31, 'cpu'), spec)) if name in NEW else \
            tmp_path / f'{name}.pt'
        if name not in NEW:
            small_fit(rc.build_model(spec, 31, 'cpu'), spec).save(paths[name])
            rc.write_json_atomic(tmp_path / 'fit.json', {'ple_bins_sha256': '0' * 64})
    for name in NEW:
        for other, path in paths.items():
            if other != name:
                with pytest.raises(Exception):
                    rc.load_model(rc.arm_spec(cfg, name), path, 'cpu', ple_bins_sha256='0' * 64)
    for name in ('control', 'periodic_residual_counts', 'tabm_residual_counts'):
        for new in NEW:
            with pytest.raises(Exception):
                rc.load_model(rc.arm_spec(cfg, name), paths[new], 'cpu')


@pytest.mark.parametrize('name', NEW)
def test_factory_refuses_same_length_other_continuous_map(tmp_path, name):
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    other = [i for i in range(53) if i not in reg.CONTINUOUS][:1] + reg.CONTINUOUS[1:]
    assert len(other) == len(reg.CONTINUOUS) and other != reg.CONTINUOUS
    cls = DirectPLETabMResidualModel if name == TABM else DirectPLEResidualModel
    model = small_fit(cls(3, other, device='cpu'), spec)
    path = saved(tmp_path, 'm', model)
    assert cls.load(path, 'cpu').context_continuous == other  # the class loader alone accepts it
    with pytest.raises(ValueError, match='continuous map'):
        rc.load_model(spec, path, 'cpu')


def test_factory_refuses_registry_spec_drift(tmp_path, monkeypatch):
    """Mutant: the registry asks for another n_bins -> an honest 48-bin archive is refused."""
    cfg = load()
    spec = rc.arm_spec(cfg, SINGLE)
    path = saved(tmp_path, 'm', small_fit(rc.build_model(spec, 32, 'cpu'), spec))
    monkeypatch.setitem(reg.ARMS, SINGLE, {**reg.ARMS[SINGLE], 'n_bins': 32})
    with pytest.raises(ValueError, match='not this PLE arm'):
        rc.load_model(spec, path, 'cpu')


# ------------------------------------------------------------------ runner tasks (CPU, synthetic)

def train_inputs(splits, arm, n=None):
    a = {k: v[:n] if n else v for k, v in splits['train'].items()}
    return tuple(np.asarray(v) for v in rc.arm_inputs(a, arm, tf.COLUMNS))


@pytest.mark.parametrize('name', NEW)
def test_runner_smoke_task_records_bins_cap_and_identity(export, tmp_path, name):
    path, digest, splits = export
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', name, 29, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    network = fit['report']['network']
    assert fit['context_width'] == 53 and fit['parameter_count'] == plm.parameter_count(
        53, 38, 6, reg.CONTINUOUS, plm.compute_bins(train_inputs(splits, spec), reg.CONTINUOUS), tabm=name == TABM)[0]
    expected = plm.bins_sha256(plm.compute_bins(train_inputs(splits, spec, rc.SMOKE['train']), reg.CONTINUOUS))
    assert fit['ple_bins_sha256'] == network['ple_bins_sha256'] == expected
    assert fit['ple_n_bins'] == network['ple_n_bins'] and len(fit['ple_n_bins']) == 81
    assert fit['ple_degenerate_columns'] == [] and fit['ple_degenerate_count'] == 0
    assert fit['epoch_cap'] == {'cap': rc.SMOKE['epochs'], 'patience': 5, 'epochs_run': rc.SMOKE['epochs'],
                                'best_epoch': fit['report']['best_epoch'], 'stopped_by': 'cap'}
    identity = read(dest / 'identity.json')
    assert identity['seed'] == 29 and identity['arm'][reg.FLAGS[name]] is True and identity['smoke'] is True
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    assert identity['source_sha256']['pitchmdp/direct_ple_model.py'] == reg.MODULE_SHA256
    with np.load(dest / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(axis=1), 1.) and p['dev'].shape[1] == 10
    model = rc.load_model(spec, dest / 'model.pt', 'cpu')  # reads the run-recorded hash next to the archive
    assert model.net.config['ple_bins_sha256'] == expected
    for seed in (0, 3, 8, 13, 16, 19, 24, 28, 34):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', name, seed, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(cfg, path, digest, tmp_path / 'real', name, 29, 'cpu', {})


@pytest.mark.parametrize('arm', ['control', 'periodic_residual_counts', 'tabm_residual_counts'])
def test_earlier_arms_in_this_family_record_the_cap_and_no_bins(export, tmp_path, arm):
    path, digest, _ = export
    dest = rc.run_task(load(), path, digest, tmp_path / 'smoke', arm, 33, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    assert fit['epoch_cap'] == rc.epoch_record(fit['report'], rc.SMOKE['epochs'], 5)
    assert not {k for k in fit if k.startswith('ple_')}


def test_ple_task_maps_degenerate_names_only_through_the_verified_export_layout(tmp_path):
    """Names come from the export manifest's recorded layout (== config); another layout fails closed."""
    cfg = load()
    profile = list(reversed(cfg['export']['context_layout']['pitcher_profile_columns']))
    path, digest, _ = tf.make_export(tmp_path / 'export', tf.lineage(cfg, pitcher_profile_columns=profile))
    with pytest.raises(ValueError, match='context layout'):
        rc.run_task(cfg, path, digest, tmp_path / 'smoke', SINGLE, 29, 'cpu', {}, smoke=True)
    assert not (tmp_path / 'smoke' / f'{SINGLE}-smoke' / '29').exists() and list((tmp_path / 'smoke' / 'failed').iterdir())


def test_runner_p35_task_never_uses_the_tabm_residual_predicate(export, tmp_path, monkeypatch):
    path, digest, _ = export
    cfg = load()
    forbid_other_family_checks(monkeypatch)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'periodic_residual_counts', 30, 'cpu', {}, smoke=True)
    assert 'epoch_cap' in read(dest / 'fit.json')


def slice_constant_export(root, cfg, n_slice):
    """Synthetic export whose context column 40 and past0 physics channel 2 are constant in the first n_slice TRAIN
    rows only (B1 case: the smoke slice has a one-valued column, full TRAIN does not)."""
    sizes = {'train': 400, 'earlystop': 120, 'temperature': 80, 'blend': 40, 'dev': 60, 'mlb_dev': 50}
    splits = {s: tf.split_arrays(n, i, 1000 * (i + 1)) for i, (s, n) in enumerate(sizes.items())}
    splits['train']['context'][:n_slice, 40] = .25
    splits['train']['tokens'][:n_slice, 0, 2] = -.5
    return root, ex.write_export(root, splits, tf.lineage(cfg), tf.SHAPES, tf.COLUMNS), splits


def test_smoke_subset_bins_come_from_the_first_smoke_rows_and_b1_rule_applies(tmp_path, monkeypatch):
    cfg = load()
    n_slice = 128  # synthetic stand-in for SMOKE['train'] = 8,192 (same code path, smaller slice)
    path, digest, splits = slice_constant_export(tmp_path / 'export', cfg, n_slice)
    monkeypatch.setitem(rc.SMOKE, 'train', n_slice)
    names = cfg['export']['context_names']
    hashes = set()
    for name in NEW:
        spec = rc.arm_spec(cfg, name)
        fit = read(rc.run_task(cfg, path, digest, tmp_path / 'fam-smoke', name, 31, 'cpu', {}, smoke=True) / 'fit.json')
        expected = plm.compute_bins(train_inputs(splits, spec, n_slice), reg.CONTINUOUS)
        assert fit['ple_bins_sha256'] == plm.bins_sha256(expected) and fit['report']['training_rows'] == n_slice
        assert fit['ple_degenerate_columns'] == ['past0.spin_axis_sin', names[40]] and fit['ple_degenerate_count'] == 2
        assert fit['report']['network']['ple_degenerate_columns'] == ['past0.phys2', 'context40']
        hashes.add(fit['ple_bins_sha256'])
    assert len(hashes) == 1  # both PLE arms, one TRAIN slice -> one bins hash
    full = plm.compute_bins(train_inputs(splits, rc.arm_spec(cfg, SINGLE)), reg.CONTINUOUS)
    assert plm.bins_sha256(full) not in hashes and 0 not in plm.check_bins(full)  # full TRAIN: no degenerate column


@pytest.mark.parametrize('name', NEW)
def test_registered_config_epoch_cap_60_reaches_fit(export, tmp_path, monkeypatch, name):
    """Non-smoke run on the tiny synthetic export: fit receives epochs=60; bins from the whole TRAIN split."""
    path, digest, splits = export
    cfg = load(registered=True)
    cls = DirectPLETabMResidualModel if name == TABM else DirectPLEResidualModel
    seen, original = [], cls.fit

    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, 'fit', spy)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'real', name, 32, 'cpu', {})
    assert [k['epochs'] for k in seen] == [60] and seen[0]['patience'] == 5 and seen[0]['batch_size'] == 1024
    fit = read(dest / 'fit.json')
    assert fit['epoch_cap'] == rc.epoch_record(fit['report'], 60, 5) and fit['epoch_cap']['cap'] == 60
    spec = rc.arm_spec(cfg, name)
    assert fit['ple_bins_sha256'] == plm.bins_sha256(plm.compute_bins(train_inputs(splits, spec), reg.CONTINUOUS))


@pytest.mark.parametrize('name', NEW)
def test_epoch_record_agrees_with_the_real_loop_of_ple_arms(name):
    spec = rc.arm_spec(load(), name)
    train, y = inputs(spec, 200, 31)
    early, ey = inputs(spec, 60, 32)
    for (epochs, patience), (run, best, stopped_by) in {(10, 2): (3, 1, 'patience'), (2, 2): (2, 1, 'cap')}.items():
        model = rc.build_model(spec, 29, 'cpu').fit(train, y, early, ey, epochs=epochs, patience=patience,
                                                    batch_size=64, learning_rate=0.)
        record = rc.epoch_record(model.report, epochs, patience)
        assert (record['epochs_run'], record['best_epoch'], record['stopped_by']) == (run, best, stopped_by)


# ------------------------------------------------------------------ scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # synthetic speed only; the scorer default stays 10,000


def with_records(fam, cfg, cap_hit=((SINGLE, 30),), counts=None, digests=None, gpus=None):
    """Add the runner's report / epoch_cap / PLE fields to every fake task and reseal."""
    counts = counts or [48] * 80 + [0]
    names = cfg['export']['context_names']
    for arm in reg.ARM_ORDER:
        for seed in cfg['seeds']:
            dest = rc.task_dir(fam, arm, seed, False)
            if not dest.exists():
                continue
            fit = read(dest / 'fit.json')
            fit['report'] = {'epochs_run': 60, 'best_epoch': 59} if (arm, seed) in cap_hit else \
                {'epochs_run': 20 + seed - 29, 'best_epoch': 15 + seed - 29}
            fit['epoch_cap'] = rc.epoch_record(fit['report'], 60, 5)
            if gpus:
                fit['runtime']['device_uuid'] = gpus(arm, seed)
            if arm in NEW:
                network = network_record(counts, (digests or {}).get((arm, seed), 'cd' * 32), ALLOCATED[arm])
                fit['report']['network'] = network
                fit.update(reg.fit_record(network, 6, names, reg.WORKLOAD), parameter_count=ALLOCATED[arm])
                fit.update(reg.effective_record(network, ALLOCATED[arm]))
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest)
    return fam


def test_scorer_reports_contrasts_headline_bins_and_gpu_pairs(export, tmp_path, few_draws, monkeypatch):
    path, digest, splits = export
    cfg = load(registered=True)
    gpu = lambda arm, seed: 'GPU-a' if (arm in ('control', SINGLE, 'tabm_residual_counts') or seed == 29) else 'GPU-b'  # noqa: E731
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg, gpus=gpu)
    forbid_other_family_checks(monkeypatch)  # the whole scorer path for P35 never consults the P34 predicate
    result = tf.run_score(tmp_path, cfg, export, fam)
    assert result['decision'] is None and result['ranking'] is None
    assert not {'factorial_contrasts', 'regularization_contrasts', 'workload_dropout_contrasts', 'tabm_contrasts',
                'embed_residual_contrasts', 'ema_contrasts', 'tabm_residual_contrasts'} & set(result)
    block = result['seed_block']
    assert block['seeds'] == reg.SEEDS and block['repeat_tag'] is None and block['epoch_cap'] == 60
    assert block['headline'] == list(reg.HEADLINE) and 'post-selection' in block['note']
    pairs = block['same_gpu_pairs']
    assert list(pairs) == [c['name'] for c in reg.CONTRASTS]
    assert pairs[reg.HEADLINE[0]] == 1 and pairs['ple_residual_counts_vs_control'] == 5  # observed, not scheduled
    assert result['ple_bins'] == {'ple_bins_sha256': 'cd' * 32, 'n_fits': 10, 'n_bins_total': 48 * 80,
                                  'degenerate_count': 1, 'degenerate_columns': ['prior_records_per100']}
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(reg.ARM_ORDER)
        contrasts = result['ple_contrasts'][split]
        assert list(contrasts) == [c['name'] for c in reg.CONTRASTS]
        vs = lambda a: 0. if a == 'control' else arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in reg.CONTRASTS:
            one = contrasts[c['name']]
            assert (one['isolates'], one['held']) == (c['isolates'], c['held'])
            assert set(one['seeds']) == {str(s) for s in cfg['seeds']}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    tasks = result['provenance']['tasks']
    assert tasks[SINGLE]['30']['epoch_cap']['stopped_by'] == 'cap' and 'ple' in tasks[TABM]['33']
    assert 'ple' not in tasks['tabm_residual_counts']['29']
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    head = table.index('PLE 임베딩 대비 — 주요 행')
    assert head < table.index('## dev PLE 임베딩 대비') < table.index('epoch 상한(60) 도달 여부')
    assert all(name in table[head:table.index('## dev PLE')] for name in reg.HEADLINE)
    assert '30: 60/59/cap' in table and '값이 하나뿐인 열(T_j = 0) 1개: prior_records_per100' in table


def test_scorer_refuses_two_bins_hashes_across_ple_fits(export, tmp_path, few_draws):
    """N12: bins are TRAIN-only, so the 10 PLE fits (both arms) must carry one hash."""
    path, digest, splits = export
    cfg = load(registered=True)
    for i, other in enumerate(((SINGLE, 31), (TABM, 33))):
        fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=(), name=f'fam{i}'), cfg,
                           digests={other: 'ef' * 32})
        with pytest.raises(ValueError, match='one bins hash'):
            tf.run_score(tmp_path, cfg, export, fam, dest=f'score{i}')
        assert not (tmp_path / f'score{i}').exists()


def test_scorer_refuses_missing_or_inconsistent_ple_records(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg)
    for i, (arm, edit) in enumerate(((SINGLE, lambda f: f.pop('ple_bins_sha256')),
                                     (TABM, lambda f: f.update(ple_degenerate_columns=[])),
                                     (SINGLE, lambda f: f.update(ple_bins_sha256='ab' * 32)),
                                     (TABM, lambda f: f['report'].pop('network')))):
        dest = rc.task_dir(fam, arm, 32, False)
        fit = read(dest / 'fit.json')
        good = copy.deepcopy(fit)
        edit(fit)
        rc.write_json_atomic(dest / 'fit.json', fit)
        tf.reseal(dest)
        with pytest.raises(ValueError, match=f'{arm}/32: PLE bins record'):
            tf.run_score(tmp_path, cfg, export, fam, dest=f'score{i}')
        rc.write_json_atomic(dest / 'fit.json', good)
        tf.reseal(dest)
    fam2 = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), name='fam2')
    with pytest.raises(ValueError, match='epoch cap record'):
        tf.run_score(tmp_path, cfg, export, fam2, dest='score-nocap')
    assert not list(tmp_path.glob('score*'))


def test_scorer_refuses_incomplete_missing_or_mixed_contrasts(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=((TABM, 33),))
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    bare = copy.deepcopy(cfg)
    del bare['comparison'][reg.CONTRAST_KEY]
    with pytest.raises(ValueError, match='PLE family is scored only'):
        tf.run_score(tmp_path, bare, export, tmp_path / 'no-family-needed')
    for key, value in ((trr.CONTRAST_KEY, trr.CONTRASTS), (tr.CONTRAST_KEY, tr.CONTRASTS),
                       (er.CONTRAST_KEY, er.CONTRASTS), (emr.CONTRAST_KEY, emr.CONTRASTS)):
        both = copy.deepcopy(cfg)
        both['comparison'][key] = value
        with pytest.raises(ValueError):
            tf.run_score(tmp_path, both, export, tmp_path / 'no-family-needed')
    with pytest.raises(ValueError, match='REGISTERED'):
        tf.run_score(tmp_path, load(), export, tmp_path / 'no-family-needed')
    assert not (tmp_path / 'score').exists()


def test_draft_refuses_real_fits_and_hash_is_reported():
    with pytest.raises(ValueError):
        rc.require_executable(read(RUN), smoke=False)
    assert hashlib.sha256(RUN.read_bytes()).hexdigest() == rc.sha256_file(RUN)
