"""Synthetic CPU tests: robust-clip family wiring (DRAFT-CUDA-ROBUST-CLIP-20261005, tentative EXP-P36-001 DRAFT).

Registry seed/arm/contrast/module-pin checks with mutants, the P36-predicate-first dispatch rule in config_check,
training_budget, run_task and the scorer, reference arms through their old build paths with identical initial
parameters, factory round trip and refusals, CPU runner smoke tasks (statistics from the TRAIN argument only, never
recomputed at predict time) and the scorer's one-statistics-hash rule.  No real data, no GPU.
author_only; same_model_review_pending; Codex_review_pending.
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
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export / fake-task helpers
from pitchmdp import direct_combo_model  # noqa: E402
from pitchmdp import direct_ema_registry as emr  # noqa: E402
from pitchmdp import direct_embed_residual_factory as erf  # noqa: E402
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_ple_registry as ple  # noqa: E402
from pitchmdp import direct_residual_model as rm  # noqa: E402
from pitchmdp import direct_robust_clip_factory as factory  # noqa: E402
from pitchmdp import direct_robust_clip_model as rcm  # noqa: E402
from pitchmdp import direct_robust_clip_registry as reg  # noqa: E402
from pitchmdp import direct_tabm_registry as tr  # noqa: E402
from pitchmdp import direct_tabm_residual_registry as trr  # noqa: E402
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402
from pitchmdp.direct_periodic_residual_model import DirectPeriodicResidualModel  # noqa: E402
from pitchmdp.direct_residual_model import DirectResidualModel  # noqa: E402
from pitchmdp.direct_robust_clip_model import DirectRobustClipResidualModel  # noqa: E402
from pitchmdp.matrix_models import MatrixModel  # noqa: E402

CONFIGS = REPO / 'configs'
RUN = CONFIGS / 'DRAFT-CUDA-ROBUST-CLIP-20261005.json'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P36-001-robust-clip-2026-10-04.md'
P32, P35_DRAFT = CONFIGS / 'EXP-P32-001.json', CONFIGS / 'DRAFT-CUDA-PLE-RESIDUAL-20261004.json'
REGISTERED = [CONFIGS / f'EXP-P3{i}-001.json' for i in range(5)]  # P30-001 .. P34-001
DRAFTS = [CONFIGS / n for n in ('DRAFT-CUDA-DIRECT-20261004.json', 'DRAFT-CUDA-FRONTIER-RUN-20261004.json',
                                'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json', 'DRAFT-CUDA-DROPOUT-RUN-20261004.json',
                                'DRAFT-CUDA-WORKLOAD-DROPOUT-20261004.json', 'DRAFT-CUDA-TABM-MINI-20261004.json',
                                'DRAFT-CUDA-EMBED-RESIDUAL-20261004.json', 'DRAFT-CUDA-EMA-20261004.json',
                                'DRAFT-CUDA-TABM-RESIDUAL-20261004.json', 'DRAFT-CUDA-PLE-RESIDUAL-20261004.json')]
NEW = 'robust_clip_residual_counts'
REFS = ('residual_counts', 'periodic_residual_counts')
# Frozen modules: never edited by this wiring (same values as tests/test_direct_robust_clip_model.py / PLE tests).
FROZEN = {'direct_periodic_model.py': '18f0b96a9b8a0d9eb39af2de4b0cc1b7a10a0dd42ca92977f726042e1aec80d3',
          'direct_periodic_residual_model.py': 'ee1b3d3f84b19f3564c853acf2bd12c640b1e5e597b8c62d485f839aeb8c992b',
          'direct_residual_model.py': '7feb9cfdcad85f4cb02c50a75e2ba2b528dd5546e600526b0334dcf0cf8e2d59',
          'direct_ple_model.py': 'd3575019f971b06ba26060a1fa31e3894a070fd3b4b88191393743cb16ea175e',
          'matrix_models.py': 'b6cc77fe3e337cf96f8c7cb31f02e8fd8a4bf71bb20713d01c462cdbd33d2fcd',
          'direct_tabm_residual_model.py': '6a9daff6c3de13a753357d8615bd9f0d5a7bfe796b350f7f7bd0a5e60c9287cf'}
FORBIDDEN_KEYS = {'winner', 'promotion', 'promote', 'verdict', 'ranking', 'selected', 'selection', 'decision_rule'}


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

def test_config_is_draft_exact_4_arm_family_on_seeds_34_38_with_cap_60():
    raw = RUN.read_text()
    cfg, p35 = json.loads(raw), read(P35_DRAFT)
    assert raw.isascii() and raw == json.dumps(cfg, indent=2) + '\n'
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'repeat_tag' not in cfg
    assert 'registration' not in cfg and cfg['experiment_id'] == 'DRAFT-CUDA-ROBUST-CLIP-20261005'
    assert not (CONFIGS / 'EXP-P36-001.json').exists() and not list(CONFIGS.glob('*P36*'))
    assert [a['name'] for a in cfg['arms']] == list(reg.ARM_ORDER) == ['control', *REFS, NEW]
    assert cfg['seeds'] == [34, 35, 36, 37, 38] and cfg['fits'] == 20 == 4 * len(cfg['seeds'])
    assert cfg['training'] == {**rc.TRAINING, 'epochs': 60} == reg.TRAINING == p35['training']
    for block in ('export', 'calibration', 'device'):
        assert cfg[block] == p35[block], block
    assert cfg['export']['manifest_sha256'].startswith('49097109')
    q = cfg['queue']
    assert (q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']) == (3600, 36000, 2)
    assert q['date'] == '2026-10-05' and 'PLACEHOLDER' in q['date_note'] and 're-dated' in q['date_note']
    assert q['no_new_fit_after'] < q['kill_own_after'] < q['aggregate_by'] <= '23:00'
    assert 'EXP-P35-001 terminal' in q['start_after'] and 'gpu-locks' in q['start_after']
    for phrase in ('all 20 tasks', '<= 1800 s', '<= 36000', 'exit 0', 'within 10%', 'at most 2 smoke attempts',
                   'failing twice', 'gates not raised'):
        assert phrase in q['stop_rule'], phrase
    assert cfg['comparison'][reg.CONTRAST_KEY] == reg.CONTRASTS and len(reg.CONTRASTS) == 2
    assert {k for k in cfg['comparison'] if k.endswith('_contrasts')} == {reg.CONTRAST_KEY}
    for phrase in ('repeatedly exposed', '<=2025', 'post-selection', 'Descriptive only', 'No 2026', 'P24-P35'):
        assert phrase in cfg['exposure'], phrase
    assert 'No 2026 rows' in cfg['data_rules'] and 'robust-scale statistics' in cfg['data_rules']
    assert 'P24-P35' in cfg['data_rules'] and PREREG in cfg['purpose']
    assert not FORBIDDEN_KEYS & keys_deep(cfg)
    for key, source in cfg['source_configs'].items():
        assert source['sha256'] == rc.sha256_file(REPO / source['path']), source['path']
    assert cfg['source_configs']['design_note']['path'] == PREREG
    assert cfg['source_configs']['ple_draft']['path'] == 'configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json'
    assert reg.MODULE_SHA256 not in raw and 'MODULE_SHA256' in cfg['arm_notes']['module_pin']  # one pin only


def test_earlier_arms_equal_the_committed_entries():
    arms, p32, p35 = ({a['name']: a for a in read(p)['arms']} for p in (RUN, P32, P35_DRAFT))
    assert reg.same(arms['residual_counts'], p32['residual_counts']) and arms['residual_counts']['combo'] is True
    assert reg.same(arms['control'], p35['control']) and reg.same(arms['control'], p32['control'])
    assert reg.same(arms['periodic_residual_counts'], p35['periodic_residual_counts'])
    assert reg.same(arms['periodic_residual_counts'], p32['periodic_residual_counts'])
    assert arms['periodic_residual_counts'][er.FLAG] is True  # keeps the P32 flag (why P36 dispatch goes first)
    for name in ('control', *REFS):
        assert reg.same(reg.BASELINES[name], arms[name]), name
    arm = arms[NEW]
    assert arm[reg.FLAG] is True and arm['workload'] == ['prior_records_per100'] == reg.WORKLOAD
    assert arm['priority'] == 1 and arm['clip'] == 3.0
    assert arm['context_continuous'] == wr.CONTINUOUS['counts'] == p32['periodic_residual_counts']['context_continuous']
    assert len(arm['context_continuous']) == 41
    assert not {'combo', 'embed_residual', 'tabm_residual', 'tabm', 'ple_residual', 'ple_tabm_residual'} & set(arm)


def test_registry_constants_match_the_torch_module():
    assert reg.KIND == rcm.KIND and reg.CLIP == rcm.CLIP == rcm.PREPROCESSING['clip_constant'] == 3.
    assert reg.ARMS[NEW]['preprocessing'] == rcm.PREPROCESSING['type']
    assert reg.ARMS[NEW]['base_model'] == rm.KIND and reg.ARMS[NEW]['width'] == rm.ARCHITECTURE['width']
    assert reg.FLAG == rc.ROBUST_CLIP_FLAG == 'robust_clip_residual'
    assert reg.FACTORS[NEW][3] == rm.ARCHITECTURE['dropout'] == reg.FACTORS['residual_counts'][3]
    assert reg.n_scalars(6) == 81 == 5 * 8 + len(reg.CONTINUOUS)
    assert reg.MODULE == Path(rcm.__file__).name


def test_module_pin_and_frozen_modules_unchanged():
    """The module is under separate review (T100b): its sha is pinned ONLY in direct_robust_clip_registry."""
    assert rc.sha256_file(PROJECT / 'pitchmdp' / 'direct_robust_clip_model.py') == reg.MODULE_SHA256 \
        == reg.check_module() == reg.module_sha256()
    for name, digest in FROZEN.items():
        assert rc.sha256_file(PROJECT / 'pitchmdp' / name) == digest, name
    pins = [p for p in (PROJECT / 'pitchmdp').glob('*.py') if reg.MODULE_SHA256 in p.read_text()]
    assert pins == [PROJECT / 'pitchmdp' / 'direct_robust_clip_registry.py']


def test_wrong_module_sha_fails_closed_everywhere(tmp_path, monkeypatch):
    cfg = load()
    spec = rc.arm_spec(cfg, NEW)
    model = small_fit(rc.build_model(spec, 34, 'cpu'), spec)
    path = saved(tmp_path, 'm', model)
    monkeypatch.setattr(reg, 'MODULE_SHA256', '0' * 64)
    for call in (lambda: rc.config_check(copy.deepcopy(cfg)), lambda: reg.check_contrasts(cfg),
                 lambda: rc.build_model(spec, 34, 'cpu'), lambda: rc.load_model(spec, path, 'cpu')):
        with pytest.raises(ValueError, match='differs from the registry pin'):
            call()


def test_seed_block_is_disjoint_from_every_earlier_block():
    earlier = {0, 1, 2, *wr.FIRST_SEEDS, *wr.SEED_BLOCKS['repeat2_seeds_8_12'], *tr.SEEDS, *er.SEEDS, *emr.SEEDS,
               *trr.SEEDS, *ple.SEEDS}
    assert earlier == set(range(34)) == {s for block in reg.EARLIER_SEEDS.values() for s in block}
    assert not set(reg.SEEDS) & earlier
    for path in (*DRAFTS, *REGISTERED):
        assert not set(read(path)['seeds']) & set(reg.SEEDS), path.name
    for block in reg.EARLIER_SEEDS:
        assert block in read(RUN)['seed_rule'] or block.split('-')[1] in read(RUN)['seed_rule'], block


def test_seed_overlap_with_an_earlier_block_fails_closed(monkeypatch):
    monkeypatch.setitem(reg.EARLIER_SEEDS, 'EXP-P35-001', [29, 30, 31, 32, 33, 34])
    with pytest.raises(ValueError, match='overlap'):
        rc.config_check(copy.deepcopy(load()))


@pytest.mark.parametrize('seeds', [[34, 35, 36], [34, 35, 36, 37], [34, 35, 36, 37, 38, 39], [38, 37, 36, 35, 34],
                                   [29, 30, 31, 32, 33], [0, 1, 2], [34.0, 35, 36, 37, 38], [35, 36, 37, 38, 39]])
def test_own_check_seeds_accepts_only_34_38(seeds):
    cfg = load()
    assert reg.check_seeds(cfg) == [34, 35, 36, 37, 38]
    with pytest.raises(ValueError):
        reg.check_seeds({**cfg, 'seeds': seeds})
    for other in (tr, er, emr, trr, ple):  # no earlier registry accepts this block
        with pytest.raises(ValueError):
            other.check_seeds(cfg)


@pytest.mark.parametrize('edit', [
    lambda c: c['training'].update(epochs=30),
    lambda c: c['training'].update(epochs=61),
    lambda c: c['training'].update(patience=10),
    lambda c: c['training'].update(learning_rate=.001),
    lambda c: c['arms'][3].update(clip=2.5),
    lambda c: c['arms'][3].update(clip=3),                                           # int, not the float constant
    lambda c: c['arms'][3].update(preprocessing='standardize'),
    lambda c: c['arms'][3].update(width=256),
    lambda c: c['arms'][3].update(workload=['prior_records_per100', 'rest_missing']),
    lambda c: c['arms'][3].update(workload=[]),
    lambda c: c['arms'][3].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][3].update(context_continuous=c['arms'][3]['context_continuous'][:-1]),
    lambda c: c['arms'][3].update(priority=2),
    lambda c: c['arms'][3].pop('robust_clip_residual'),
    lambda c: c['arms'][3].update(robust_clip_residual=1),
    lambda c: c['arms'][3].update(embed_residual=True),                              # P32 flag on the new arm
    lambda c: c['arms'][3].update(name='robust_clip_residual_block'),                # unknown arm
    lambda c: c['arms'][2].update(robust_clip_residual=True),                        # earlier arm flagged
    lambda c: c['arms'][2].update(embedding='linear'),                               # earlier arm altered
    lambda c: c['arms'][1].update(model='periodic_flatten_mlp'),
    lambda c: c['arms'][1].pop('combo'),
    lambda c: c['arms'][1].update(workload=[]),
    lambda c: c['arms'][0].update(workload=['prior_records_per100']),
    lambda c: c['arms'].insert(1, c['arms'].pop(2)),                                 # arm order
    lambda c: c['arms'].pop(),
    lambda c: c['arms'].pop(1),
    lambda c: c['arms'].append({**c['arms'][3], 'name': 'robust_clip_residual_block'}),
    lambda c: c.update(fits=25),
    lambda c: c.update(seeds=[34, 35, 36]),
    lambda c: c.update(seeds=[29, 30, 31, 32, 33]),
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
    cfg = copy.deepcopy(load())
    cfg['training'].update(change)
    with pytest.raises(ValueError, match='cuda_direct_v1 budget with epochs 60'):
        reg.check_family(cfg)


def test_contrast_list_matches_the_prereg_and_changes_only_the_numeric_input():
    cfg = load()
    assert reg.check_contrasts(cfg) == reg.CONTRASTS
    assert [(c['candidate'], c['reference'], c['isolates'], c['role'], c['compute_matched']) for c in reg.CONTRASTS] == [
        (NEW, 'residual_counts', 'input_preprocessing', 'primary', True),
        (NEW, 'periodic_residual_counts', 'numeric_input_treatment', 'secondary', False)]
    assert all(c['name'] == f"{c['candidate']}_vs_{c['reference']}" for c in reg.CONTRASTS)
    assert (reg.PRIMARY, reg.SECONDARY) == ('robust_clip_residual_counts_vs_residual_counts',
                                            'robust_clip_residual_counts_vs_periodic_residual_counts')
    assert all(c['held'] == {'workload': 'counts', 'backbone': 'residual', 'dropout_rate': .1, 'head': 'single'}
               for c in reg.CONTRASTS)


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0], name='again')),
    lambda cs: cs[0].update(candidate='residual_counts', reference=NEW),                 # reversed sign
    lambda cs: cs[0].update(isolates='numeric_input_treatment'),
    lambda cs: cs[1].update(compute_matched=True),                                        # secondary is not matched
    lambda cs: cs[0].update(role='secondary'),
    lambda cs: cs[0].update(held={**cs[0]['held'], 'backbone': 'flatten'}),
    lambda cs: cs[1].update(reference='control'),
    lambda cs: cs.pop(),
    lambda cs: cs.pop(0),
    lambda cs: cs[0].update(winner=True),
    lambda cs: cs.reverse(),                                                              # primary first
    lambda cs: cs.append({'name': 'robust_clip_residual_counts_vs_control', 'candidate': NEW, 'reference': 'control',
                          'isolates': 'bundle', 'role': 'secondary', 'compute_matched': False, 'held': {}}),
])
def test_contrast_list_is_fixed(edit):
    cfg = copy.deepcopy(load(registered=True))
    edit(cfg['comparison'][reg.CONTRAST_KEY])
    with pytest.raises(ValueError):
        reg.check_contrasts(cfg)


def test_contrast_factor_check_catches_a_registry_level_mutation(monkeypatch):
    """Mutant: the robust-clip arm's registered backbone factor drifts -> the factor check refuses."""
    monkeypatch.setitem(reg.FACTORS, NEW, ('robust_clip', 'counts', 'flatten', .1, 'blocks', 'single'))
    with pytest.raises(ValueError, match='declared factors'):
        reg.check_contrasts(load())


# ------------------------------------------------------------------ dispatch: P36 predicate first

def forbid_other_family_checks(monkeypatch):
    """Any route of a P36 config into an earlier family-level check or predicate fails loudly."""
    def boom(*args, **kwargs):
        raise AssertionError('P36 config reached an earlier family route')
    for module in (ple, trr, emr, tr, er, wr):
        monkeypatch.setattr(module, 'check_family', boom)
    for name in ('uses_ple', 'uses_tabm_residual', 'uses_ema', 'uses_tabm', 'uses_embed_residual',
                 'uses_workload_dropout'):
        monkeypatch.setattr(rc, name, boom)


def test_p36_config_never_reaches_an_earlier_family_check_or_budget(monkeypatch):
    cfg = load()
    forbid_other_family_checks(monkeypatch)
    assert rc.config_check(copy.deepcopy(cfg))['seeds'] == reg.SEEDS
    assert rc.training_budget(cfg) is reg.TRAINING and rc.uses_epoch_cap(cfg)


def test_p36_family_is_refused_by_the_p32_check_it_must_never_reach():
    """Why order matters: the embedded-residual family check would reject this config (wrong arms / seeds)."""
    cfg = load()
    assert rc.uses_embed_residual(cfg) and rc.uses_robust_clip(cfg) and not rc.uses_ple(cfg)
    with pytest.raises(ValueError, match='Embedded-residual family must be exactly'):
        er.check_family(cfg)


@pytest.mark.parametrize('path', [*DRAFTS, *REGISTERED, 'p30_002'], ids=lambda p: getattr(p, 'name', p))
def test_earlier_families_route_exactly_as_before(path, monkeypatch):
    cfg = p30_002() if path == 'p30_002' else read(path)
    before = copy.deepcopy(cfg)
    budget = ple.TRAINING if rc.uses_ple(cfg) else trr.TRAINING if rc.uses_tabm_residual(cfg) else rc.TRAINING
    capped = rc.uses_ple(cfg) or rc.uses_tabm_residual(cfg)

    def boom(*args, **kwargs):
        raise AssertionError('earlier family reached the P36 route')
    monkeypatch.setattr(reg, 'check_family', boom)
    monkeypatch.setattr(reg, 'check_arm', boom)
    monkeypatch.setattr(factory, 'build_robust_clip', boom)
    assert rc.config_check(copy.deepcopy(cfg)) == before
    assert not rc.uses_robust_clip(cfg) and rc.training_budget(cfg) == budget and reg.check_contrasts(cfg) is None
    assert rc.uses_epoch_cap(cfg) == capped


@pytest.mark.parametrize('path', [*DRAFTS, *REGISTERED], ids=lambda p: p.name)
def test_earlier_families_refuse_robust_clip_seeds_arm_or_key(path):
    cfg = read(path)
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'seeds': list(reg.SEEDS)})
    arm = {'name': NEW, 'priority': 1, reg.FLAG: True, 'workload': reg.WORKLOAD, **reg.ARMS[NEW]}
    with pytest.raises(ValueError):
        rc.config_check({**copy.deepcopy(cfg), 'arms': [*cfg['arms'], arm]})
    foreign = copy.deepcopy(cfg)
    foreign.setdefault('comparison', {})[reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
    with pytest.raises(ValueError):
        reg.check_contrasts(foreign)


def test_build_dispatch_of_every_p36_arm_reference_arms_use_their_old_paths(monkeypatch):
    cfg = load()
    calls = []
    for module, name in ((direct_combo_model, 'build_combo'), (erf, 'build_embed_residual'),
                         (factory, 'build_robust_clip')):
        original = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *a, _o=original, _n=name, **k: calls.append(_n) or _o(*a, **k))
    expected = {'control': (MatrixModel, None), 'residual_counts': (DirectResidualModel, 'build_combo'),
                'periodic_residual_counts': (DirectPeriodicResidualModel, 'build_embed_residual'),
                NEW: (DirectRobustClipResidualModel, 'build_robust_clip')}
    for name, (cls, path) in expected.items():
        calls.clear()
        model = rc.build_model(rc.arm_spec(cfg, name), 34, 'cpu')
        assert type(model) is cls and calls == ([path] if path else []), name
    assert rc.build_model(rc.arm_spec(cfg, NEW), 34, 'cpu').context_continuous == reg.CONTINUOUS
    with pytest.raises(ValueError, match='no context names'):
        rc.build_model(rc.arm_spec(cfg, NEW), 34, 'cpu', ['x'] * 52)
    with pytest.raises(ValueError, match='Unregistered arm'):
        rc.arm_spec(cfg, 'robust_clip_residual_block')


def initial_parameters(spec, seed):
    """Parameters after a zero-learning-rate fit: the frozen loop's seeded initial values (AdamW step is 0 * ...)."""
    train, y = inputs(spec, 120, 41)
    early, ey = inputs(spec, 40, 42)
    model = rc.build_model(spec, seed, 'cpu').fit(train, y, early, ey, epochs=1, patience=1, batch_size=64,
                                                  learning_rate=0.)
    return {k: v.detach().clone() for k, v in model.net.named_parameters()}


def test_same_seed_gives_identical_initial_parameters_for_the_primary_contrast():
    cfg = load()
    base, new = (initial_parameters(rc.arm_spec(cfg, n), 35) for n in ('residual_counts', NEW))
    assert list(base) == list(new) and all(torch.equal(base[k], new[k]) for k in base)
    other = initial_parameters(rc.arm_spec(cfg, NEW), 36)
    assert not all(torch.equal(other[k], new[k]) for k in new)


@pytest.mark.parametrize('name, source', [('residual_counts', P32), ('periodic_residual_counts', P32),
                                          ('periodic_residual_counts', P35_DRAFT)])
def test_reference_arms_build_exactly_as_in_their_earlier_family(name, source):
    """Same arm entry, same old code path => same class and identical initial parameters for one seed."""
    ours, theirs = rc.arm_spec(load(), name), rc.arm_spec(read(source), name)
    assert reg.same(ours, theirs)
    a, b = initial_parameters(ours, 37), initial_parameters(theirs, 37)
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_config_check_and_scorer_import_need_no_torch():
    new = ['pitchmdp/direct_robust_clip_model.py', 'pitchmdp/direct_robust_clip_factory.py',
           'pitchmdp/direct_robust_clip_registry.py']
    assert set(new) <= set(rc.MODEL_SOURCES) and len(set(rc.MODEL_SOURCES)) == len(rc.MODEL_SOURCES)
    assert rc.MODEL_SOURCES[-2:] == ['pitchmdp/direct_frontier_model.py', 'pitchmdp/direct_frontier_registry.py']
    assert 'pitchmdp/direct_robust_clip_registry.py' in sc.SCORER_SOURCES
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'c = json.load(open({str(RUN)!r})); rc.config_check(c); rc.training_budget(c); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


def test_scorer_refuses_robust_clip_contrasts_on_other_families_before_any_read(tmp_path):
    for path in (*REGISTERED[1:], P35_DRAFT):
        cfg = read(path)
        cfg['status'], cfg['execution'] = 'REGISTERED', {'enabled': True}
        cfg['comparison'][reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
        cpath, csha = tf.write_cfg(tmp_path, cfg, name=path.name)
        with pytest.raises(ValueError, match='Robust-clip family must be exactly'):
            sc.score(cpath, csha, tmp_path / 'no-export', '0' * 64, tmp_path / 'no-family', tmp_path / 'score')
    assert not (tmp_path / 'score').exists()


# ------------------------------------------------------------------ fit.json record

def report(digest='ab' * 32, counts=None):
    return {'rc_stats_sha256': digest, 'rc_rule_counts': counts or {'iqr': 78, 'range': 2, 'constant': 1},
            'network': {'kind': reg.KIND, 'context_continuous': list(reg.CONTINUOUS), 'rc_stats_sha256': digest}}


@pytest.mark.parametrize('edit', [
    lambda r: r.pop('rc_stats_sha256'),
    lambda r: r.update(rc_stats_sha256='AB' * 32),
    lambda r: r['network'].update(rc_stats_sha256='cd' * 32),
    lambda r: r['network'].update(kind=rm.KIND),
    lambda r: r['network'].update(context_continuous=r['network']['context_continuous'][:-1]),
    lambda r: r['rc_rule_counts'].update(iqr=77),
    lambda r: r['rc_rule_counts'].update(iqr=78.0),
    lambda r: r['rc_rule_counts'].pop('constant'),
    lambda r: r.pop('network'),
])
def test_fit_record_rederives_and_refuses_inconsistent_records(edit):
    good = report()
    assert reg.fit_record(good, 6) == {'rc_stats_sha256': 'ab' * 32,
                                       'rc_rule_counts': {'iqr': 78, 'range': 2, 'constant': 1}}
    bad = copy.deepcopy(good)
    edit(bad)
    with pytest.raises(ValueError):
        reg.fit_record(bad, 6)


@pytest.mark.parametrize('record', [None, {}, {'train_rows': 0, 'past_valid_rows': [0] * 5},
                                    {'train_rows': 10, 'past_valid_rows': [5] * 4},
                                    {'train_rows': 10, 'past_valid_rows': [11, 10, 10, 10, 10]},
                                    {'train_rows': 10.0, 'past_valid_rows': [5] * 5},
                                    {'train_rows': 10, 'past_valid_rows': [5] * 5, 'extra': 1}])
def test_stats_rows_record_is_checked(record):
    assert reg.check_stats_rows({'train_rows': 10, 'past_valid_rows': [5, 10, 10, 10, 10]}, 6)
    with pytest.raises(ValueError):
        reg.check_stats_rows(record, 6)


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
    rc.write_json_atomic(dest / 'fit.json', {'rc_stats_sha256': model.net.config['rc_stats_sha256']})
    return dest / 'model.pt'


def test_factory_round_trip_statistics_from_train_only_and_recorded_hash(tmp_path):
    spec = rc.arm_spec(load(), NEW)
    model = small_fit(rc.build_model(spec, 35, 'cpu'), spec)
    train = tuple(np.asarray(a) for a in inputs(spec, 300, 11)[0])
    early = tuple(np.asarray(a) for a in inputs(spec, 90, 12)[0])
    digest = rcm.stats_sha256(*rcm.compute_stats(train, reg.CONTINUOUS)[:2])
    assert model.net.config['rc_stats_sha256'] == model.report['rc_stats_sha256'] == digest
    pooled = tuple(np.concatenate(p) for p in zip(train, early))
    assert digest not in {rcm.stats_sha256(*rcm.compute_stats(x, reg.CONTINUOUS)[:2]) for x in (early, pooled)}
    path = saved(tmp_path, 'task', model)
    again = rc.load_model(spec, path, 'cpu')
    x, _ = inputs(spec, 30, 15)
    assert type(again) is DirectRobustClipResidualModel and np.array_equal(model.logits(x), again.logits(x))
    assert again.report == model.report and again.context_continuous == reg.CONTINUOUS
    explicit = rc.load_model(spec, path, 'cpu', rc_stats_sha256=digest)
    assert np.array_equal(explicit.logits(x), model.logits(x))
    with pytest.raises(ValueError, match='recorded by the run'):
        rc.load_model(spec, path, 'cpu', rc_stats_sha256='0' * 64)
    rc.write_json_atomic(path.parent / 'fit.json', {'rc_stats_sha256': '1' * 64})
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
        rc.load_model(wide, path, 'cpu', rc_stats_sha256=digest)


def test_predict_and_load_never_recompute_statistics(tmp_path, monkeypatch):
    spec = rc.arm_spec(load(), NEW)
    model = small_fit(rc.build_model(spec, 36, 'cpu'), spec)
    path = saved(tmp_path, 'task', model)

    def boom(*args, **kwargs):
        raise AssertionError('statistics recomputed after fit')
    monkeypatch.setattr(rcm, 'compute_stats', boom)
    x, _ = inputs(spec, 30, 16)
    again = rc.load_model(spec, path, 'cpu')
    assert np.array_equal(model.logits(x), again.logits(x))


def test_factory_refuses_other_arm_archives_both_ways(tmp_path):
    cfg = load()
    paths = {}
    for name in ('control', *REFS, NEW):
        spec = rc.arm_spec(cfg, name)
        model = small_fit(rc.build_model(spec, 37, 'cpu'), spec)
        if name == NEW:
            paths[name] = saved(tmp_path, name, model)
        else:
            paths[name] = tmp_path / f'{name}.pt'
            model.save(paths[name])
    for other, path in paths.items():
        if other != NEW:
            with pytest.raises(Exception):
                rc.load_model(rc.arm_spec(cfg, NEW), path, 'cpu', rc_stats_sha256='0' * 64)
    for name in ('control', *REFS):
        with pytest.raises(Exception):
            rc.load_model(rc.arm_spec(cfg, name), paths[NEW], 'cpu')


def test_factory_refuses_same_length_other_continuous_map(tmp_path):
    spec = rc.arm_spec(load(), NEW)
    other = [i for i in range(53) if i not in reg.CONTINUOUS][:1] + reg.CONTINUOUS[1:]
    assert len(other) == len(reg.CONTINUOUS) and other != reg.CONTINUOUS
    model = small_fit(DirectRobustClipResidualModel(3, other, device='cpu'), spec)
    path = saved(tmp_path, 'm', model)
    assert DirectRobustClipResidualModel.load(path, 'cpu').context_continuous == other  # class loader alone accepts
    with pytest.raises(ValueError, match='continuous map'):
        rc.load_model(spec, path, 'cpu')


def test_factory_refuses_registry_spec_drift(tmp_path, monkeypatch):
    """Mutant: the registry asks for another model kind -> an honest archive is refused."""
    spec = rc.arm_spec(load(), NEW)
    path = saved(tmp_path, 'm', small_fit(rc.build_model(spec, 38, 'cpu'), spec))
    monkeypatch.setitem(reg.ARMS, NEW, {**reg.ARMS[NEW], 'model': rm.KIND})
    with pytest.raises(ValueError, match='not this robust-clip arm'):
        rc.load_model(spec, path, 'cpu')


# ------------------------------------------------------------------ runner tasks (CPU, synthetic)

def train_inputs(splits, arm, n=None):
    a = {k: v[:n] if n else v for k, v in splits['train'].items()}
    return tuple(np.asarray(v) for v in rc.arm_inputs(a, arm, tf.COLUMNS))


def test_runner_smoke_task_records_statistics_cap_module_and_identity(export, tmp_path, monkeypatch):
    path, digest, splits = export
    cfg = load()
    spec = rc.arm_spec(cfg, NEW)
    seen, original = [], rcm.compute_stats

    def spy(arrays, context_continuous):
        seen.append(len(arrays[0]))
        return original(arrays, context_continuous)
    monkeypatch.setattr(rcm, 'compute_stats', spy)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', NEW, 34, 'cpu', {}, smoke=True)
    n_train = min(rc.SMOKE['train'], len(splits['train']['y']))
    assert seen == [n_train]  # once, in fit, on the TRAIN argument; never at calibration / predict time
    fit = read(dest / 'fit.json')
    assert fit['context_width'] == 53 and fit['parameter_count'] == rm.parameter_count(53, 38, 6)  # == residual_counts
    train = train_inputs(splits, spec, rc.SMOKE['train'])
    expected = rcm.stats_sha256(*original(train, reg.CONTINUOUS)[:2])
    assert fit['rc_stats_sha256'] == fit['report']['rc_stats_sha256'] == fit['report']['network']['rc_stats_sha256'] \
        == expected
    early = tuple(np.asarray(v) for v in rc.arm_inputs(splits['earlystop'], spec, tf.COLUMNS))
    assert expected != rcm.stats_sha256(*original(tuple(np.concatenate(p) for p in zip(train, early)),
                                                  reg.CONTINUOUS)[:2])
    assert fit['rc_rule_counts'] == fit['report']['rc_rule_counts'] and sum(fit['rc_rule_counts'].values()) == 81
    assert fit['rc_stats_rows'] == {'train_rows': n_train,
                                    'past_valid_rows': [int(v) for v in train[1][:, :-1].sum(axis=0)]}
    assert fit['rc_module_sha256'] == reg.MODULE_SHA256
    assert fit['epoch_cap'] == {'cap': rc.SMOKE['epochs'], 'patience': 5, 'epochs_run': rc.SMOKE['epochs'],
                                'best_epoch': fit['report']['best_epoch'], 'stopped_by': 'cap'}
    identity = read(dest / 'identity.json')
    assert identity['seed'] == 34 and identity['arm'][reg.FLAG] is True and identity['smoke'] is True
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    assert identity['source_sha256']['pitchmdp/direct_robust_clip_model.py'] == reg.MODULE_SHA256
    with np.load(dest / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(axis=1), 1.) and p['dev'].shape[1] == 10
    model = rc.load_model(spec, dest / 'model.pt', 'cpu')  # reads the run-recorded hash next to the archive
    assert model.net.config['rc_stats_sha256'] == expected and seen == [n_train]
    for seed in (0, 3, 8, 13, 16, 19, 24, 29, 33, 39):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', NEW, seed, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='Unregistered arm'):
        rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'robust_clip_residual_block', 34, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(cfg, path, digest, tmp_path / 'real', NEW, 34, 'cpu', {})


@pytest.mark.parametrize('arm', ['control', *REFS])
def test_earlier_arms_in_this_family_record_the_cap_and_no_statistics(export, tmp_path, arm):
    path, digest, _ = export
    dest = rc.run_task(load(), path, digest, tmp_path / 'smoke', arm, 38, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    assert fit['epoch_cap'] == rc.epoch_record(fit['report'], rc.SMOKE['epochs'], 5)
    assert not {k for k in fit if k.startswith(('rc_', 'ple_'))}


def test_runner_p36_task_never_uses_an_earlier_family_predicate(export, tmp_path, monkeypatch):
    path, digest, _ = export
    cfg = load()
    forbid_other_family_checks(monkeypatch)
    for arm in ('periodic_residual_counts', NEW):
        dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', arm, 35, 'cpu', {}, smoke=True)
        assert 'epoch_cap' in read(dest / 'fit.json')


def test_registered_config_epoch_cap_60_reaches_fit_and_statistics_use_whole_train(export, tmp_path, monkeypatch):
    path, digest, splits = export
    cfg = load(registered=True)
    seen, original = [], DirectRobustClipResidualModel.fit

    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(DirectRobustClipResidualModel, 'fit', spy)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'real', NEW, 36, 'cpu', {})
    assert [k['epochs'] for k in seen] == [60] and seen[0]['patience'] == 5 and seen[0]['batch_size'] == 1024
    fit = read(dest / 'fit.json')
    assert fit['epoch_cap'] == rc.epoch_record(fit['report'], 60, 5) and fit['epoch_cap']['cap'] == 60
    spec = rc.arm_spec(cfg, NEW)
    assert fit['rc_stats_sha256'] == rcm.stats_sha256(*rcm.compute_stats(train_inputs(splits, spec), reg.CONTINUOUS)[:2])
    assert fit['rc_stats_rows']['train_rows'] == len(splits['train']['y'])


def test_runner_refuses_a_module_differing_from_the_identity_source_map(export, tmp_path, monkeypatch):
    path, digest, _ = export
    original = rc.own_hashes
    monkeypatch.setattr(rc, 'own_hashes', lambda: {**original(), 'pitchmdp/direct_robust_clip_model.py': '0' * 64})
    with pytest.raises(RuntimeError):
        rc.run_task(load(), path, digest, tmp_path / 'smoke', NEW, 37, 'cpu', {}, smoke=True)
    assert not (tmp_path / 'smoke' / f'{NEW}-smoke' / '37').exists() and list((tmp_path / 'smoke' / 'failed').iterdir())


# ------------------------------------------------------------------ scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # synthetic speed only; the scorer default stays 10,000


def with_records(fam, cfg, digests=None, module=None, train_rows=None, gpus=None):
    """Add the runner's report / epoch_cap / robust-clip fields to every fake task and reseal."""
    for arm in reg.ARM_ORDER:
        for seed in cfg['seeds']:
            dest = rc.task_dir(fam, arm, seed, False)
            if not dest.exists():
                continue
            fit = read(dest / 'fit.json')
            fit['report'] = {'epochs_run': 20 + seed - 34, 'best_epoch': 15 + seed - 34}
            fit['epoch_cap'] = rc.epoch_record(fit['report'], 60, 5)
            fit['train_rows_sha256'] = (train_rows or {}).get((arm, seed), 'aa' * 32)
            if gpus:
                fit['runtime']['device_uuid'] = gpus(arm, seed)
            if arm == NEW:
                fit['report'].update(report((digests or {}).get(seed, 'cd' * 32)))
                fit.update(reg.fit_record(fit['report'], 6), rc_module_sha256=module or reg.MODULE_SHA256,
                           rc_stats_rows={'train_rows': 400, 'past_valid_rows': [200, 400, 400, 400, 400]})
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest)
    return fam


def test_scorer_reports_both_contrasts_statistics_and_gpu_pairs_descriptively(export, tmp_path, few_draws,
                                                                              monkeypatch):
    path, digest, splits = export
    cfg = load(registered=True)
    gpu = lambda arm, seed: 'GPU-a' if arm in ('control', 'residual_counts', NEW) or seed == 34 else 'GPU-b'  # noqa: E731
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg, gpus=gpu)
    def boom(*args, **kwargs):
        raise AssertionError('P36 config reached an earlier family check')
    for module in (ple, trr, emr, tr, er, wr):  # the scorer path for P36 never runs an earlier family check
        monkeypatch.setattr(module, 'check_family', boom)
    result = tf.run_score(tmp_path, cfg, export, fam)
    assert result['decision'] is None and result['ranking'] is None
    assert not FORBIDDEN_KEYS & keys_deep({k: v for k, v in result.items() if k not in ('decision', 'ranking')})
    assert not {'factorial_contrasts', 'regularization_contrasts', 'workload_dropout_contrasts', 'tabm_contrasts',
                'embed_residual_contrasts', 'ema_contrasts', 'tabm_residual_contrasts', 'ple_contrasts'} & set(result)
    block = result['seed_block']
    assert block['seeds'] == reg.SEEDS and block['repeat_tag'] is None and block['epoch_cap'] == 60
    assert (block['primary'], block['secondary']) == (reg.PRIMARY, reg.SECONDARY)
    assert 'post-selection' in block['note'] and 'not compute-matched' in block['note']
    assert block['same_gpu_pairs'] == {reg.PRIMARY: 5, reg.SECONDARY: 1}  # observed, not scheduled
    stats = result['robust_clip_stats']
    assert stats['rc_stats_sha256'] == 'cd' * 32 and stats['n_fits'] == 5
    assert stats['rc_module_sha256'] == reg.MODULE_SHA256 and stats['rc_stats_rows']['train_rows'] == 400
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(reg.ARM_ORDER)
        contrasts = result['robust_clip_contrasts'][split]
        assert list(contrasts) == [reg.PRIMARY, reg.SECONDARY]
        vs = lambda a: arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in reg.CONTRASTS:
            one = contrasts[c['name']]
            assert (one['isolates'], one['role'], one['compute_matched'], one['held']) == \
                (c['isolates'], c['role'], c['compute_matched'], c['held'])
            assert set(one['seeds']) == {str(s) for s in cfg['seeds']}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    tasks = result['provenance']['tasks']
    assert 'robust_clip' in tasks[NEW]['36'] and 'robust_clip' not in tasks['residual_counts']['36']
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert table.index('## dev 강건 스케일') < table.index('## mlb_dev 강건 스케일') < table.index('epoch 상한(60)')
    assert reg.PRIMARY in table and '| primary |' in table and '| secondary |' in table and 'False' in table


@pytest.mark.parametrize('field', ['digest', 'train_rows'])
def test_scorer_refuses_two_statistics_or_train_hashes_across_seeds(export, tmp_path, few_draws, field):
    """Same TRAIN => same statistics for every seed and GPU: one differing seed fails closed."""
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=())
    fam = with_records(fam, cfg, digests={37: 'ef' * 32}) if field == 'digest' else \
        with_records(fam, cfg, train_rows={(NEW, 37): 'bb' * 32})
    with pytest.raises(ValueError, match='one statistics hash'):
        tf.run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()


def test_scorer_refuses_missing_or_inconsistent_robust_clip_records(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg)
    cases = ((lambda f: f.pop('rc_stats_sha256'), 'differs from the fit report'),
             (lambda f: f.update(rc_stats_sha256='ab' * 32), 'differs from the fit report'),
             (lambda f: f.update(rc_rule_counts={'iqr': 81, 'range': 0, 'constant': 0}), 'differs from the fit report'),
             (lambda f: f['report']['network'].update(rc_stats_sha256='ab' * 32), 'malformed'),
             (lambda f: f.pop('rc_stats_rows'), 'malformed'),
             (lambda f: f.update(rc_module_sha256='0' * 64), 'registry pin'),
             (lambda f: f.pop('rc_module_sha256'), 'registry pin'))
    for i, (edit, message) in enumerate(cases):
        dest = rc.task_dir(fam, NEW, 35, False)
        fit = read(dest / 'fit.json')
        good = copy.deepcopy(fit)
        edit(fit)
        rc.write_json_atomic(dest / 'fit.json', fit)
        tf.reseal(dest)
        with pytest.raises(ValueError, match=f'{NEW}/35: robust-clip .*{message}'):
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
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=((NEW, 38),)), cfg)
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    bare = copy.deepcopy(cfg)
    del bare['comparison'][reg.CONTRAST_KEY]
    with pytest.raises(ValueError, match='robust-clip family is scored only'):
        tf.run_score(tmp_path, bare, export, tmp_path / 'no-family-needed')
    for key, value in ((er.CONTRAST_KEY, er.CONTRASTS), (ple.CONTRAST_KEY, ple.CONTRASTS),
                       (trr.CONTRAST_KEY, trr.CONTRASTS), (tr.CONTRAST_KEY, tr.CONTRASTS),
                       (emr.CONTRAST_KEY, emr.CONTRASTS)):
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
