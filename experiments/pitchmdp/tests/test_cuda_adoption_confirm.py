"""Synthetic CPU tests: adoption-seed confirmation family (EXP-P48-001 DRAFT, marker adoption_confirm).

Registry exactness against the earlier registries, the DRAFT config (export / calibration / arms copied from
EXP-P35-001, cap 60, five-GPU device, Stage-B rules), config_check mutants, the marker-first dispatch rule in
config_check / training_budget / uses_epoch_cap and the Stage-A scorer, earlier registered configs routing exactly as
before, CPU runner smoke tasks through the unchanged arm paths (epoch cap recorded, registry in the task identity), the
Stage-A scorer's contrast validation and report, and the five-GPU queue.  No real data, no GPU, no DEV metric.
"""
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
from pitchmdp import direct_adoption_confirm_registry as reg  # noqa: E402
from pitchmdp import direct_ema_registry as emr  # noqa: E402
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_embed_type_registry as et  # noqa: E402
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402
from pitchmdp import direct_ple_registry as ple  # noqa: E402
from pitchmdp import direct_robust_clip_registry as rcr  # noqa: E402
from pitchmdp import direct_tabm_registry as tr  # noqa: E402
from pitchmdp import direct_tabm_residual_registry as trr  # noqa: E402
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402

CONFIGS = REPO / 'configs'
RUN = CONFIGS / 'EXP-P48-001.json'
P35 = CONFIGS / 'EXP-P35-001.json'
PREREG = PROJECT / 'docs' / 'PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md'
REVIEW = PROJECT / 'docs' / 'reviews' / 'REVIEW-EXP-P48-001-ASTRA-001.md'
REVIEW_SHA256 = '2e1d9c420da522101ad33fffa335460abc676fdbee5157925fd43f433014df44'  # verbatim copy (evidence)
EARLIER = sorted(p for p in CONFIGS.glob('EXP-P[34]*-001.json') if p != RUN)  # registered CUDA families P30..P43
FIVE = ['GPU-edb6bc12-c9aa-d79b-d3e7-9bc28118e4d4', 'GPU-0cc63040-5522-7fea-20d8-93b9196ce019',
        'GPU-18e648d0-b68e-9321-9c77-69da241aad85', 'GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9',
        'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99']


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


def read(path):
    return json.loads(Path(path).read_text())


def load(registered=False):
    return tf.load(RUN, registered, True)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


def mutate(cfg, edit):
    cfg = copy.deepcopy(cfg)
    edit(cfg)
    return cfg


# ------------------------------------------------------------------ registry exactness

def test_registry_arms_contrasts_budget_are_the_earlier_registered_entries():
    assert reg.ARM_ORDER == ('control', 'periodic_residual_counts', 'tabm_residual_counts')
    assert reg.same(reg.BASELINES['control'], trr.BASELINES['control'])
    assert reg.same(reg.BASELINES['periodic_residual_counts'], trr.BASELINES['periodic_residual_counts'])
    assert reg.same(reg.BASELINES['tabm_residual_counts'], ple.BASELINES['tabm_residual_counts'])
    p34 = {a['name']: a for a in read(CONFIGS / 'EXP-P34-001.json')['arms']}
    p35 = {a['name']: a for a in read(P35)['arms']}
    for name in reg.ARM_ORDER:
        assert reg.same(reg.BASELINES[name], p34[name]) and reg.same(reg.BASELINES[name], p35[name])
    assert reg.BASELINES['periodic_residual_counts'][er.FLAG] is True
    assert reg.BASELINES['tabm_residual_counts'][trr.FLAG] is True
    assert reg.TRAINING is trr.TRAINING and reg.TRAINING['epochs'] == reg.EPOCH_CAP == 60
    assert {k: v for k, v in reg.TRAINING.items() if k != 'epochs'} == {k: v for k, v in rc.TRAINING.items()
                                                                       if k != 'epochs'}
    assert reg.SEEDS == [0, 1, 2, 3, 4] and reg.FITS == 15
    assert reg.CONTRASTS == [next(c for c in trr.CONTRASTS if c['name'] == n) for n in reg.CONTRAST_NAMES]
    assert [c['name'] for c in reg.CONTRASTS] == [
        'tabm_residual_counts_vs_control', 'periodic_residual_counts_vs_control',
        'tabm_residual_counts_vs_periodic_residual_counts']
    assert reg.CONTRASTS[0] is not next(c for c in trr.CONTRASTS if c['name'] == reg.CONTRAST_NAMES[0])  # a copy


def test_marker_constants_and_disclosed_seed_reuse():
    assert rc.ADOPTION_CONFIRM_MARKER == reg.MARKER == 'adoption_confirm'
    assert reg.OTHER_MARKERS == (et.MARKER, pid.MARKER)
    earlier = ple.EARLIER_SEEDS
    assert set(earlier['P24/P25/P27/P29']) == {0, 1, 2} and {3, 4} <= set(earlier['EXP-P30-001'])
    assert reg.EARLIER_USES['P24/P25/P27/P29 CUDA arms (MODEL RNG)'] == [0, 1, 2]
    assert reg.EARLIER_USES['EXP-P30-001 CUDA arms (MODEL RNG)'] == [3, 4]
    doc = reg.__doc__
    for needle in ('P24/P25/P27/P29', 'EXP-P30-001', 'post-selection', 'PRIMARY', 'Stage B', 'FIRST', '2027',
                   'exactly ONE', 'intersection-union', 'exploratory, never confirmed', 'Pipeline comparison',
                   'cap 30 on MPS', 'EXP-P11-001 analysis manifest'):
        assert needle in doc, needle
    assert tuple(FIVE) == reg.GPU_UUIDS and len(set(FIVE)) == 5


def test_stage_b_rules_are_fixed_and_point_at_the_frozen_reference():
    rules = reg.STAGE_B_RULES
    ref = rules['reference']
    for key, path in (('config_sha256', ref['config']), ('g0_bundle_sha256', ref['g0_bundle']),
                      ('recorded_results_sha256', ref['recorded_results'])):
        assert rc.sha256_file(REPO / path) == ref[key], path
    g15 = read(REPO / ref['config'])
    assert g15['definition']['arm'] == ref['arm'] == reg.REFERENCE_ARM and g15['definition']['direct_weight'] == .5
    assert g15['g0_component']['bundle_sha256'] == ref['g0_bundle_sha256']
    assert g15['files']['results_p16']['sha256'] == ref['recorded_results_sha256']
    assert rules['mix_rules'][reg.MIX_FIXED] == {'weight': .5}
    assert rules['mix_rules'][reg.MIX_FIT]['grid'] == [round(.5 + .05 * i, 2) for i in range(10)] == reg.FIT_GRID
    assert rules['primary'] == {'arm': 'tabm_residual_counts', 'mix_rule': 'MIX_FIT'}
    assert rules['candidate_arms'] == ['periodic_residual_counts', 'tabm_residual_counts']
    assert rules['descriptive_arms'] == ['control']
    p16 = read(REPO / 'configs' / 'EXP-P16-001.yaml')
    assert {k: v for k, v in rules['N'].items() if k != 'stage'} == p16['metrics']['N']
    assert rules['N']['stage'] == 'final'
    assert (rules['bootstrap']['draws'], rules['bootstrap']['seed']) == (p16['metrics']['bootstrap']['draws'],
                                                                         p16['metrics']['bootstrap']['seed'])
    recorded = read(REPO / ref['recorded_results'])
    gate = rules['reproduction_gate']
    for path in gate['fields'].values():
        value = recorded
        for key in path:
            value = value[key]
        assert isinstance(value, float)
    for path in gate['seed_delta_fields'].values():  # per-seed G15 - G0 deltas exist for seeds 0-4 (values not read)
        value = recorded
        for key in path:
            value = value[key]
        assert isinstance(value, list) and len(value) == 5 and all(isinstance(x, float) for x in value)
    assert gate['tolerance'] == 1e-9 and not set(gate['fields']) & set(gate['seed_delta_fields'])
    assert rules['alpha'] == .05 and rules['confirmation'] == 'primary_only_predictive_improvement_on_both_splits'
    assert set(rules['multiplicity']) == {'primary', 'secondary', 'combined_secondary_outcome', 'descriptive'}
    assert 'intersection_union' in rules['multiplicity']['primary']
    assert 'no_adjustment' in rules['multiplicity']['primary']
    assert 'holm' in rules['multiplicity']['secondary'] and 'three_secondary' in rules['multiplicity']['secondary']
    assert rules['multiplicity']['combined_secondary_outcome'] == 'exploratory_never_confirmed'
    assert reg.PRIMARY_ROW == 'tabm_residual_counts/MIX_FIT'
    assert reg.SECONDARY_ROWS == ['periodic_residual_counts/MIX_FIXED', 'periodic_residual_counts/MIX_FIT',
                                  'tabm_residual_counts/MIX_FIXED']
    inputs = rules['inputs']
    assert inputs['runs_root_name'] == 'ML-MATRIX-20260924' and sorted(inputs['g0_mlb']) == ['0', '1', '2', '3', '4']
    listed = [entry['path'] for entry in g15['files'].values()]
    for s, rel in inputs['flatten'].items():
        assert any(p.endswith(f'ML-MATRIX-20260924/{rel}/prediction_state.json') for p in listed), s
    g0 = read(REPO / ref['g0_bundle'])
    for s, rel in inputs['g0_cpanel'].items():
        path = g0['files'][f'seed{s}_prediction_state']['path']
        assert path.endswith(f'ML-MATRIX-20260924/{rel}/prediction_state.json')
    assert g0['files']['p4_frequency']['path'].endswith(inputs['cpanel_baseline'])
    assert g0['files']['p11_frequency']['path'].endswith(inputs['mlb_baseline'])
    # whole-MLB G0 members: not listed in the bundle itself, reached through the P11 analysis manifest it pins
    assert not any('EXP-P11-001/members' in entry['path'] for entry in g0['files'].values())
    chain = inputs['p11_manifest']
    assert g0['files'][chain['g0_bundle_key']]['path'].endswith('ML-MATRIX-20260924/' + chain['path'])
    assert g0['files'][chain['g0_bundle_key']]['sha256'] == \
        'aaf5db0d0a3d134b4d92393efa640dc67dba8efb4a4af330d21f2ab66d9ebda9'
    assert inputs['anchors'] == {'cpanel_baseline': {'source': 'g0_bundle', 'files': ['archive']},
                                 'mlb_baseline': {'source': 'g0_bundle', 'files': ['archive']},
                                 'g0_cpanel': {'source': 'g0_bundle', 'files': ['state']},
                                 'g0_mlb': {'source': 'p11_manifest', 'files': ['state', 'archive']},
                                 'flatten': {'source': 'g15', 'files': ['state']}}


# ------------------------------------------------------------------ config

def test_config_is_the_draft_family_with_p35_export_and_five_gpus():
    raw, cfg, p35 = RUN.read_text(), read(RUN), read(P35)
    assert raw.isascii() and raw == json.dumps(cfg, indent=2) + '\n'
    assert cfg['protocol'] == 'cuda_direct_v1' and cfg['experiment_id'] == 'EXP-P48-001'
    # DRAFT before registration; REGISTERED (execution enabled) after the 2026-10-27 registration edit
    assert cfg['status'] in ('DRAFT', 'REGISTERED') and cfg['execution'] == {'enabled': cfg['status'] == 'REGISTERED'}
    assert cfg[reg.MARKER] is True
    assert 'repeat_tag' not in cfg and cfg['seeds'] == [0, 1, 2, 3, 4] and cfg['fits'] == 15
    for key in ('export', 'calibration'):
        assert cfg[key] == p35[key], key  # copied verbatim (manifest sha, splits, shapes, workload, context layout)
    assert cfg['export']['manifest_sha256'] == '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
    assert cfg['training'] == {'epochs': 60, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
                               'loss_weighting': 'uniform', 'hpo_trials': 0}
    assert [a['name'] for a in cfg['arms']] == list(reg.ARM_ORDER)
    assert cfg['device'] == {'kind': 'cuda', 'allowed_gpu_uuids': FIVE, 'max_parallel': 5, 'cpu_fallback': False}
    assert cfg['comparison'][reg.CONTRAST_KEY] == reg.CONTRASTS and cfg['comparison']['control_arm'] == 'control'
    assert cfg['comparison'][reg.STAGE_B_KEY]['rules'] == reg.STAGE_B_RULES
    assert 'PRIMARY' in cfg['comparison'][reg.STAGE_B_KEY]['description']['role']
    for source in cfg['source_configs'].values():
        assert rc.sha256_file(REPO / source['path']) == source['sha256'], source['path']
    q = cfg['queue']
    assert (q['timezone'], q['date'], q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']) == \
        ('Asia/Seoul', '2026-10-27', 3600, 36000, 2)
    assert q['no_new_fit_after'] < q['kill_own_after'] < q['aggregate_by'] and 'set at registration' in q['date_note']
    assert set(q['clock_fields_set_at_registration']) == {'date', 'no_new_fit_after', 'kill_own_after', 'aggregate_by'}
    assert 'failing twice aborts the family' in q['stop_rule'] and '1800 s' in q['stop_rule']
    for needle in ('No 2026 rows', 'post-selection', 'not new data', 'prospective'):
        assert needle in cfg['exposure'], needle
    registration = cfg['registration']
    assert registration['preregistration']['path'] == str(PREREG.relative_to(REPO))
    if cfg['status'] == 'DRAFT':  # before registration: the registration block is empty and real fits refuse
        assert {k for k, v in registration.items() if v is None} == {'source_commit', 'draft_sha256', 'smoke', 'timing',
                                                                      'independent_review', 'go', 'registered_kst'}
        assert registration['preregistration']['sha256'] is None
        with pytest.raises(ValueError, match='REGISTERED'):
            rc.require_executable(cfg, smoke=False)
    else:  # after the 2026-10-27 registration edit: every field filled, the pre-registration pinned, real fits allowed
        assert not {k for k, v in registration.items() if v is None}
        assert registration['preregistration']['sha256'] == rc.sha256_file(PREREG)
        assert registration['independent_review']['verdict'] in ('PASS', 'PASS_WITH_NOTES')
        rc.require_executable(cfg, smoke=False)
    rc.require_executable(cfg, smoke=True)


def test_config_check_accepts_the_draft_and_budget_is_cap_60():
    cfg = read(RUN)
    assert rc.config_check(copy.deepcopy(cfg)) == cfg
    assert rc.uses_adoption_confirm(cfg) and rc.uses_epoch_cap(cfg)
    assert rc.training_budget(cfg) is reg.TRAINING and rc.training_budget(cfg)['epochs'] == 60
    assert reg.check_contrasts(cfg) == reg.CONTRASTS and reg.check_stage_b(cfg) == reg.STAGE_B_RULES


EDITS = {
    'seeds_0_2': lambda c: c.update(seeds=[0, 1, 2], fits=9),
    'seed_5': lambda c: c.update(seeds=[0, 1, 2, 3, 5]),
    'seed_order': lambda c: c.update(seeds=[4, 3, 2, 1, 0]),
    'fits_25': lambda c: c.update(fits=25),
    'repeat_tag': lambda c: c.update(repeat_tag='r2'),
    'arm_order': lambda c: c.update(arms=[c['arms'][0], c['arms'][2], c['arms'][1]]),
    'drop_arm': lambda c: c.update(arms=c['arms'][:2]),
    'extra_arm': lambda c: c['arms'].append(copy.deepcopy(trr.BASELINES['tabm_counts'])),
    'width': lambda c: c['arms'][1].update(width=144),
    'k_4': lambda c: c['arms'][2].update(k=4),
    'scaling': lambda c: c['arms'][2].update(scaling_init='uniform'),
    'columns': lambda c: c['arms'][2].update(context_continuous=c['arms'][2]['context_continuous'][:-1]),
    'workload': lambda c: c['arms'][1].update(workload=[]),
    'priority': lambda c: c['arms'][2].update(priority=2),
    'flag_int': lambda c: c['arms'][2].update(tabm_residual=1),
    'control_workload': lambda c: c['arms'][0].update(workload=['prior_records_per100']),
    'cap_30': lambda c: c['training'].update(epochs=30),
    'lr': lambda c: c['training'].update(learning_rate=.001),
    'patience': lambda c: c['training'].update(patience=6),
    'six_gpus': lambda c: c['device']['allowed_gpu_uuids'].append('GPU-00000000-0000-0000-0000-000000000006'),
    'unknown_gpu': lambda c: c['device'].update(allowed_gpu_uuids=['GPU-00000000-0000-0000-0000-000000000006']),
    'repeated_gpu': lambda c: c['device'].update(allowed_gpu_uuids=[FIVE[0], FIVE[0]]),
    'no_gpu': lambda c: c['device'].update(allowed_gpu_uuids=[]),
    'parallel_6': lambda c: c['device'].update(max_parallel=6),
    'parallel_0': lambda c: c['device'].update(max_parallel=0),
    'parallel_float': lambda c: c['device'].update(max_parallel=5.0),
    'cpu_fallback': lambda c: c['device'].update(cpu_fallback=True),
    'device_extra_key': lambda c: c['device'].update(note='x'),
    'marker_false': lambda c: c.update({reg.MARKER: False}),
    'marker_one': lambda c: c.update({reg.MARKER: 1}),
    'embed_type_marker': lambda c: c.update({et.MARKER: True}),
    'pitcher_id_marker': lambda c: c.update({pid.MARKER: True}),
    'no_contrasts': lambda c: c['comparison'].pop(reg.CONTRAST_KEY),
    'contrast_swap': lambda c: c['comparison'][reg.CONTRAST_KEY][0].update(candidate='control',
                                                                          reference='tabm_residual_counts'),
    'contrast_extra': lambda c: c['comparison'][reg.CONTRAST_KEY].append(copy.deepcopy(trr.CONTRASTS[1])),
    'no_stage_b': lambda c: c['comparison'].pop(reg.STAGE_B_KEY),
    'grid': lambda c: c['comparison'][reg.STAGE_B_KEY]['rules']['mix_rules']['MIX_FIT']['grid'].append(1.0),
    'primary': lambda c: c['comparison'][reg.STAGE_B_KEY]['rules']['primary'].update(mix_rule='MIX_FIXED'),
    'n_rule': lambda c: c['comparison'][reg.STAGE_B_KEY]['rules']['N'].update(required_negative_seeds=3),
    'reference_sha': lambda c: c['comparison'][reg.STAGE_B_KEY]['rules']['reference'].update(config_sha256='0' * 64),
    'last_year': lambda c: c['export'].update(last_year=2026),
    'bounds': lambda c: c['calibration'].update(bounds=[.5, 3.]),
    'protocol': lambda c: c.update(protocol='cuda_direct_v2'),
}


@pytest.mark.parametrize('edit', EDITS, ids=list(EDITS))
def test_config_check_refuses_family_drift(edit):
    with pytest.raises(ValueError):
        rc.config_check(mutate(read(RUN), EDITS[edit]))


def test_any_allowed_gpu_subset_passes():
    for gpus in ([FIVE[3]], FIVE[:2], list(reversed(FIVE))):
        for parallel in (1, len(gpus), 5):
            rc.config_check(mutate(read(RUN), lambda c: c['device'].update(allowed_gpu_uuids=gpus,
                                                                          max_parallel=parallel)))


# ------------------------------------------------------------------ dispatch: marker first

def forbid_other_family_routes(monkeypatch, predicates=True):
    """Any route of a P48 config into an earlier family-level check (or, for config_check / budget, an earlier family
    predicate) fails loudly.  The scorer's per-task record checks may still ask uses_ple / uses_robust_clip."""
    def boom(*args, **kwargs):
        raise AssertionError('P48 config reached an earlier family route')
    for module in (trr, ple, rcr, et, pid, emr, tr, er, wr):
        monkeypatch.setattr(module, 'check_family', boom)
    for name in ('uses_tabm_residual', 'uses_ple', 'uses_robust_clip', 'uses_embed_type', 'uses_ema', 'uses_tabm',
                 'uses_embed_residual', 'uses_workload_dropout') if predicates else ():
        monkeypatch.setattr(rc, name, boom)


def test_p48_never_reaches_an_earlier_family_route(monkeypatch):
    cfg = read(RUN)
    forbid_other_family_routes(monkeypatch)
    assert rc.config_check(copy.deepcopy(cfg))['seeds'] == reg.SEEDS
    assert rc.training_budget(cfg) is reg.TRAINING and rc.uses_epoch_cap(cfg)


def test_marker_with_p34_arms_is_judged_by_the_new_registry_not_the_p34_one(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('reached the P34 family check')
    monkeypatch.setattr(trr, 'check_family', boom)
    p34 = read(CONFIGS / 'EXP-P34-001.json')
    with pytest.raises(ValueError, match='Adoption-confirm family must be exactly'):
        rc.config_check({**copy.deepcopy(p34), reg.MARKER: True})
    assert rc.uses_tabm_residual(p34) and rc.training_budget({**p34, reg.MARKER: True}) is reg.TRAINING


def test_without_the_marker_the_p34_route_claims_and_refuses_this_family():
    cfg = read(RUN)
    del cfg[reg.MARKER]
    assert rc.uses_tabm_residual(cfg) and not rc.uses_adoption_confirm(cfg)
    with pytest.raises(ValueError, match='TabM-residual family must be exactly'):
        rc.config_check(cfg)


@pytest.mark.parametrize('path', EARLIER, ids=[p.stem for p in EARLIER])
def test_earlier_registered_configs_route_and_budget_exactly_as_before(path, monkeypatch):
    cfg = read(path)
    before = copy.deepcopy(cfg)
    budget = rc.training_budget(cfg)

    def boom(*args, **kwargs):
        raise AssertionError('earlier family reached the P48 route')
    monkeypatch.setattr(reg, 'check_family', boom)
    assert not rc.uses_adoption_confirm(cfg) and reg.check_contrasts(cfg) is None
    assert rc.config_check(copy.deepcopy(cfg)) == before and rc.training_budget(cfg) == budget
    monkeypatch.undo()
    with pytest.raises(ValueError):  # the marker can never be bolted onto an earlier family
        rc.config_check({**copy.deepcopy(cfg), reg.MARKER: True})
    foreign = copy.deepcopy(cfg)
    foreign.setdefault('comparison', {})[reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
    with pytest.raises(ValueError):
        reg.check_contrasts(foreign)


def test_sources_are_wired_and_checks_need_no_torch():
    assert 'pitchmdp/direct_adoption_confirm_registry.py' in rc.MODEL_SOURCES
    assert rc.MODEL_SOURCES[-2:] == ['pitchmdp/direct_frontier_model.py', 'pitchmdp/direct_frontier_registry.py']
    assert len(set(rc.MODEL_SOURCES)) == len(rc.MODEL_SOURCES) and set(rc.own_hashes()) == {rc.SELF, *rc.MODEL_SOURCES}
    assert 'pitchmdp/direct_adoption_confirm_registry.py' in sc.SCORER_SOURCES
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'c = json.load(open({str(RUN)!r})); rc.config_check(c); rc.training_budget(c); import score_cuda_direct; '
            'from pitchmdp import direct_adoption_confirm_registry as r; r.check_contrasts(c); '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


def test_prereg_states_the_fixed_design_and_the_draft_config_sha():
    text = PREREG.read_text()
    cfg = read(RUN)
    draft = rc.sha256_file(RUN) if cfg['status'] == 'DRAFT' else cfg['registration']['draft_sha256']
    for needle in ('EXP-P48-001', draft, '0, 1, 2, 3, 4', 'control', 'periodic_residual_counts', 'tabm_residual_counts',
                   'MIX_FIXED', 'MIX_FIT', '0.50, 0.55', 'Holm', '4/5', '-0.003', 'G15-MIX-RESEARCH-FROZEN-v1.json',
                   'results/EXP-P16-001.json', '1e-9', 'P24/P25/P27/P29', 'EXP-P30-001', 'post-selection', '2026',
                   'c13cc989', 'RS-02', '2027', 'not recorded here', '627', '135', 'Review round 1',
                   'REVIEW-EXP-P48-001-ASTRA-001.md', 'ceiling of this direct-head pipeline on the exposed 2025 DEV',
                   'cap 30', 'pipeline comparison', 'intersection-union', 'exploratory', 'never "confirmed"',
                   'EXP-P11-001 analysis manifest', 'per-seed'):
        assert needle in text, needle
    assert 'ceiling of the direct-head line' not in text


def test_review_round_1_is_kept_verbatim_with_its_required_fixes():
    import hashlib
    assert hashlib.sha256(REVIEW.read_bytes()).hexdigest() == REVIEW_SHA256
    text = REVIEW.read_text()
    assert 'VERDICT: FAIL' in text and 'Required fixes before execution' in text
    for needle in ('P11 analysis manifest', 'seed-array swaps', 'multiplicity claim'):
        assert needle in text, needle


# ------------------------------------------------------------------ runner (CPU smoke, unchanged arm paths)

def test_draft_refuses_real_fits(export, tmp_path):
    path, digest, _ = export
    draft = load()
    draft['status'], draft['execution'] = 'DRAFT', {'enabled': False}  # the DRAFT precondition, independent of the shipped status
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(draft, path, digest, tmp_path / 'real', 'tabm_residual_counts', 0, 'cpu', {})


@pytest.mark.parametrize('arm', reg.ARM_ORDER)
def test_runner_smoke_task_records_cap_and_registry_identity(export, tmp_path, arm):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', arm, 0, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    assert fit['epoch_cap']['cap'] == rc.SMOKE['epochs'] and fit['epoch_cap']['stopped_by'] == 'cap'
    identity = read(dest / 'identity.json')
    assert identity['seed'] == 0 and identity['smoke'] is True and identity['arm'] == reg.BASELINES[arm]
    assert identity['source_sha256']['pitchmdp/direct_adoption_confirm_registry.py'] == \
        rc.sha256_file(PROJECT / 'pitchmdp/direct_adoption_confirm_registry.py')
    with np.load(dest / 'mlb_predictions.npz') as data:
        assert {'mlb_dev', 'mlb_dev_raw', 'mlb_dev_delivery_level', 'mlb_dev_keys', 'mlb_dev_y', 'mlb_dev_game_pk',
                'mlb_dev_pitcher'} == set(data.files)
    for seed in (5, 24, 29):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', arm, seed, 'cpu', {}, smoke=True)


def test_build_dispatch_keeps_the_earlier_model_classes():
    from pitchmdp.direct_periodic_residual_model import DirectPeriodicResidualModel
    from pitchmdp.direct_tabm_residual_model import DirectTabMResidualModel
    from pitchmdp.matrix_models import MatrixModel
    cfg = load()
    for name, cls in (('control', MatrixModel), ('periodic_residual_counts', DirectPeriodicResidualModel),
                      ('tabm_residual_counts', DirectTabMResidualModel)):
        assert type(rc.build_model(rc.arm_spec(cfg, name), 0, 'cpu')) is cls, name


# ------------------------------------------------------------------ Stage-A scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)


def with_records(fam, cfg, gpus=None):
    for arm in reg.ARM_ORDER:
        for seed in cfg['seeds']:
            dest = rc.task_dir(fam, arm, seed, False)
            fit = read(dest / 'fit.json')
            fit['report'] = {'epochs_run': 20 + seed, 'best_epoch': 15 + seed}
            fit['epoch_cap'] = rc.epoch_record(fit['report'], 60, 5)
            fit['runtime']['device_uuid'] = (gpus or (lambda a, s: 'GPU-a' if s % 2 else 'GPU-b'))(arm, seed)
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest)
    return fam


def test_stage_a_scorer_reports_the_three_contrasts_without_verdict(export, tmp_path, few_draws, monkeypatch):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg)
    forbid_other_family_routes(monkeypatch, predicates=False)
    result = tf.run_score(tmp_path, cfg, export, fam)
    assert result['decision'] is None and result['ranking'] is None
    others = {'factorial_contrasts', 'regularization_contrasts', 'workload_dropout_contrasts', 'tabm_contrasts',
              'embed_residual_contrasts', 'ema_contrasts', 'tabm_residual_contrasts', 'ple_contrasts',
              'robust_clip_contrasts', 'embed_type_contrasts', 'pitcher_id_contrasts'}
    assert not others & set(result)
    block = result['seed_block']
    assert block['seeds'] == reg.SEEDS and block['repeat_tag'] is None and block['epoch_cap'] == 60
    assert 'post-selection' in block['note'] and 'score_cuda_mix_vs_g15.py' in block['stage_b']
    assert list(block['same_gpu_pairs']) == list(reg.CONTRAST_NAMES)
    assert block['same_gpu_pairs']['tabm_residual_counts_vs_control'] == 5  # same GPU per seed in with_records
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(reg.ARM_ORDER)
        contrasts = result['adoption_confirm_contrasts'][split]
        assert list(contrasts) == list(reg.CONTRAST_NAMES)
        vs = lambda a: 0. if a == 'control' else arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in reg.CONTRASTS:
            one = contrasts[c['name']]
            assert (one['isolates'], one['held']) == (c['isolates'], c['held'])
            assert set(one['seeds']) == {str(s) for s in reg.SEEDS}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    assert result['provenance']['tasks']['tabm_residual_counts']['4']['epoch_cap']['cap'] == 60
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert table.index('## dev 채택 시드 확인 가족 대비') < table.index('## mlb_dev 채택 시드 확인 가족 대비') \
        < table.index('epoch 상한(60) 도달 여부')
    assert 'Stage B' in table and '0: 20/15/patience' in table


def test_stage_a_scorer_refuses_incomplete_family_or_contrast_drift(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, skip=(('tabm_residual_counts', 3),))
    with pytest.raises(ValueError, match='Incomplete family'):
        tf.run_score(tmp_path, cfg, export, fam, dest='s1')
    drift = copy.deepcopy(cfg)
    drift['comparison'][reg.CONTRAST_KEY] = drift['comparison'][reg.CONTRAST_KEY][:2]
    cpath, csha = tf.write_cfg(tmp_path, drift, name='drift.json')
    with pytest.raises(ValueError, match='differ from the fixed pre-registered pairs'):
        sc.score(cpath, csha, path, digest, fam, tmp_path / 's2')
    assert not (tmp_path / 's1').exists() and not (tmp_path / 's2').exists()


def test_stage_a_scorer_refuses_the_contrast_key_on_other_registered_families(tmp_path):
    for path in EARLIER:
        cfg = read(path)
        cfg.setdefault('comparison', {})[reg.CONTRAST_KEY] = copy.deepcopy(reg.CONTRASTS)
        cpath, csha = tf.write_cfg(tmp_path, cfg, name=path.name)
        with pytest.raises(ValueError):
            sc.score(cpath, csha, tmp_path / 'no-export', '0' * 64, tmp_path / 'no-family', tmp_path / 'score')
    assert not (tmp_path / 'score').exists()


CONTRAST_EDITS = {
    'isolates': lambda c: c['comparison'][reg.CONTRAST_KEY][2].update(isolates='backbone_bundle'),
    'held': lambda c: c['comparison'][reg.CONTRAST_KEY][1]['held'].update(head='tabm_mini'),
    'order': lambda c: c['comparison'][reg.CONTRAST_KEY].reverse(),
    'empty': lambda c: c['comparison'].update({reg.CONTRAST_KEY: []}),
}


@pytest.mark.parametrize('edit', CONTRAST_EDITS, ids=list(CONTRAST_EDITS))
def test_contrast_list_is_fixed(edit):
    with pytest.raises(ValueError):
        reg.check_contrasts(mutate(read(RUN), CONTRAST_EDITS[edit]))


def test_contrast_factor_check_catches_a_registry_level_mutation(monkeypatch):
    assert reg.check_contrasts(read(RUN)) == reg.CONTRASTS
    monkeypatch.setitem(reg.FACTORS, 'periodic_residual_counts',
                        ('periodic', 'counts', 'residual', .1, 'head', 'single'))
    with pytest.raises(ValueError, match='declared factors'):
        reg.check_contrasts(read(RUN))


# ------------------------------------------------------------------ queue (no change needed)

def test_queue_runs_the_15_tasks_on_up_to_five_allowed_gpus(tmp_path):
    import cuda_direct_queue as cq
    cfg = read(RUN)
    make = lambda gpus, d: cq.Queue(cfg, tmp_path / d, gpus, lambda *a: None, lambda *a: False,  # noqa: E731
                                    lock_dir=tmp_path / 'locks')
    queue = make(FIVE, 'ok')
    assert queue.gpus == FIVE and sorted(queue.locks) == sorted(FIVE) and len(queue.tasks) == 15
    assert [t['id'] for t in queue.tasks[:3]] == ['control-0', 'periodic_residual_counts-0', 'tabm_residual_counts-0']
    for name, gpus in {'six': [*FIVE, 'GPU-00000000-0000-0000-0000-000000000006'], 'twice': [FIVE[0], FIVE[0]],
                       'unknown': ['GPU-00000000-0000-0000-0000-000000000006'], 'none': []}.items():
        with pytest.raises(ValueError):
            make(gpus, name)
    q = cfg['queue']  # clock fields are placeholders before registration and the real deadlines after it
    assert cq.Deadlines(q).no_new.isoformat().startswith(f"{q['date']}T{q['no_new_fit_after']}")
