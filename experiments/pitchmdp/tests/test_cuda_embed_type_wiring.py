"""Synthetic CPU tests: embedding-map control family (DRAFT-CUDA-EMBED-TYPE-20261005, tentative EXP-P37-001 DRAFT).

Registry arm/seed/budget/contrast checks with mutants, the P37-marker-first dispatch rule in config_check,
training_budget and the scorer, earlier registered configs routing exactly as before, parameter counts re-derived
from the frozen networks, deterministic config generation, CPU runner smoke tasks through the unchanged embedded-
residual path (epoch cap recorded) and the scorer's contrast + cost report.  No real data, no GPU, no DEV metric.
author_only; same_model_distinct_review_pending.
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
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_embed_type_registry as et  # noqa: E402
from pitchmdp import direct_ple_registry as ple  # noqa: E402
from pitchmdp import direct_robust_clip_registry as rcr  # noqa: E402
from pitchmdp import direct_tabm_residual_registry as trr  # noqa: E402
from pitchmdp.direct_periodic_residual_model import parameter_count as embed_parameters  # noqa: E402

CONFIGS = REPO / 'configs'
RUN = CONFIGS / 'DRAFT-CUDA-EMBED-TYPE-20261005.json'
RUN_SHA256 = '75265fc715f0fec998f609f0316ffe1de486c609677588d0fbd7241fc427ace2'  # single-GPU regeneration (was 1e3a529c)
GPU3 = 'GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9'  # retained (Song 2026-10-05)
GPU4 = 'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99'  # released to another project
P36_SHA256 = 'b6479afa43a59df91774e0b714da2397692bdb06959fabb7716a129a5826b87e'
PREREG = REPO / 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P37-001-embed-type-2026-10-05.md'
MAKE = REPO / 'experiments/pitchmdp/docs/launch/make_p37_config.py'
EARLIER = [CONFIGS / f'EXP-P3{i}-001.json' for i in range(7)]  # P30-001 .. P36-001 (registered)
FORBIDDEN_KEYS = {'winner', 'promotion', 'promote', 'verdict', 'ranking', 'selected', 'selection', 'decision_rule'}


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


def keys_deep(value):
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_deep(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_deep(v)}
    return set()


# ------------------------------------------------------------------ config / registry

def test_config_file_is_the_generated_draft_and_p36_source_is_the_registered_file():
    assert hashlib.sha256(RUN.read_bytes()).hexdigest() == RUN_SHA256
    assert hashlib.sha256((CONFIGS / 'EXP-P36-001.json').read_bytes()).hexdigest() == P36_SHA256
    out = subprocess.run([sys.executable, '-c', f'import runpy, json; ns = runpy.run_path({str(MAKE)!r}); '
                          'print(json.dumps(ns["build"](), indent=2, ensure_ascii=False))'],
                         capture_output=True, text=True, check=True).stdout
    assert out == RUN.read_text()  # deterministic generator, nothing hand-edited


def test_config_is_draft_exact_3_arm_family_on_seeds_39_43_with_cap_60():
    cfg = read(RUN)
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'registration' not in cfg
    assert cfg[et.MARKER] is True and cfg['experiment_id'] == 'EXP-P37-001'
    assert [a['name'] for a in cfg['arms']] == list(et.ARM_ORDER) and cfg['fits'] == 15 == et.FITS
    assert cfg['seeds'] == [39, 40, 41, 42, 43] and 'repeat_tag' not in cfg
    assert cfg['training'] == {'epochs': 60, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
                               'loss_weighting': 'uniform', 'hpo_trials': 0}
    p36 = read(CONFIGS / 'EXP-P36-001.json')
    for key in ('protocol', 'export', 'training', 'calibration', 'data_rules'):
        assert cfg[key] == p36[key], key
    # Song 2026-10-05 single-GPU rule: device replaced (P36 two-GPU history stays as registered)
    assert p36['device']['max_parallel'] == 2 and sorted(p36['device']['allowed_gpu_uuids']) == sorted([GPU3, GPU4])
    assert cfg['device'] == {'kind': 'cuda', 'allowed_gpu_uuids': [GPU3], 'max_parallel': 1, 'cpu_fallback': False}
    assert cfg['device'] == et.DEVICE and et.RETAINED_GPU == GPU3 and et.RELEASED_GPUS == (GPU4,)
    q = cfg['queue']
    assert q['launch_by'] < q['no_new_fit_after'] < q['kill_own_after'] < q['aggregate_by'] < '23:00'
    assert GPU3 in q['resource_rule'] and GPU4 in q['resource_rule'] and 'one fit at a time' in q['start_after']
    assert GPU4 not in json.dumps({k: v for k, v in cfg.items() if k != 'queue'})
    assert cfg['comparison'][et.CONTRAST_KEY] == et.CONTRASTS
    assert not FORBIDDEN_KEYS & keys_deep(cfg)


def test_arms_are_the_exact_earlier_registered_entries():
    p32 = {a['name']: a for a in read(CONFIGS / 'EXP-P32-001.json')['arms']}
    p36 = {a['name']: a for a in read(CONFIGS / 'EXP-P36-001.json')['arms']}
    for arm in read(RUN)['arms']:
        assert et.same(arm, p32[arm['name']])
        if arm['name'] in p36:
            assert et.same(arm, p36[arm['name']])


def test_parameter_counts_rederived_from_the_frozen_networks():
    shapes = read(RUN)['export']['shapes']
    n_context = shapes['n_context'] + 1  # + the counts workload scalar
    assert embed_parameters('periodic', n_context, shapes['n_token'], shapes['length'], et.CONTINUOUS) == \
        et.PARAMETERS['periodic_residual_counts']
    assert embed_parameters('linear', n_context, shapes['n_token'], shapes['length'], et.CONTINUOUS) == \
        et.PARAMETERS['linear_residual_counts']
    # control (flatten_mlp 128) is re-derived by the CPU runner smoke test through rc.run_task at these shapes
    diff = et.PARAMETERS['periodic_residual_counts'] - et.PARAMETERS['linear_residual_counts']
    assert diff == 264 and diff / et.PARAMETERS['linear_residual_counts'] < .0011  # near-matched, not equal


def test_seed_block_is_disjoint_from_every_earlier_block():
    earlier = {s for block in et.EARLIER_SEEDS.values() for s in block}
    assert earlier == set(range(39)) and not set(et.SEEDS) & earlier


def mutate(cfg, edit):
    cfg = copy.deepcopy(cfg)
    edit(cfg)
    return cfg


EDITS = {
    'marker_false': lambda c: c.update({et.MARKER: False}),
    'seed_38': lambda c: c.update(seeds=[38, 40, 41, 42, 43]),
    'four_seeds': lambda c: c.update(seeds=[39, 40, 41, 42], fits=12),
    'repeat_tag': lambda c: c.update(repeat_tag='r2'),
    'fits_20': lambda c: c.update(fits=20),
    'cap_30': lambda c: c['training'].update(epochs=30),
    'lr': lambda c: c['training'].update(learning_rate=.001),
    'arm_order': lambda c: c.update(arms=[c['arms'][0], c['arms'][2], c['arms'][1]]),
    'extra_arm': lambda c: c['arms'].append(copy.deepcopy(er.BASELINES['residual_counts'])),
    'width': lambda c: c['arms'][2].update(width=144),
    'columns': lambda c: c['arms'][2].update(context_continuous=c['arms'][2]['context_continuous'][:-1]),
    'workload': lambda c: c['arms'][1].update(workload=[]),
    'gpu4_added': lambda c: c['device'].update(allowed_gpu_uuids=[GPU3, GPU4]),
    'gpu4_only': lambda c: c['device'].update(allowed_gpu_uuids=[GPU4]),
    'gpu4_added_parallel_2': lambda c: c['device'].update(allowed_gpu_uuids=[GPU3, GPU4], max_parallel=2),
    'parallel_2': lambda c: c['device'].update(max_parallel=2),
    'other_gpu': lambda c: c['device'].update(allowed_gpu_uuids=['GPU-edb6bc12-c9aa-d79b-d3e7-9bc28118e4d4']),
    'cpu_fallback': lambda c: c['device'].update(cpu_fallback=True),
    'p36_device': lambda c: c.update(device=read(CONFIGS / 'EXP-P36-001.json')['device']),
}


@pytest.mark.parametrize('edit', EDITS, ids=list(EDITS))
def test_config_check_refuses_family_drift(edit):
    with pytest.raises(ValueError):
        rc.config_check(mutate(read(RUN), EDITS[edit]))


CONTRAST_EDITS = {
    'swap': lambda c: c['comparison'][et.CONTRAST_KEY][0].update(candidate='linear_residual_counts',
                                                                 reference='periodic_residual_counts'),
    'compute_matched': lambda c: c['comparison'][et.CONTRAST_KEY][0].update(compute_matched=True),
    'parameter_matched': lambda c: c['comparison'][et.CONTRAST_KEY][0].update(parameter_matched=True),
    'vs_control': lambda c: c['comparison'][et.CONTRAST_KEY].append(
        {**et.CONTRASTS[0], 'name': 'x', 'reference': 'control'}),
    'empty': lambda c: c['comparison'].update({et.CONTRAST_KEY: []}),
}


@pytest.mark.parametrize('edit', CONTRAST_EDITS, ids=list(CONTRAST_EDITS))
def test_contrast_list_is_fixed(edit):
    with pytest.raises(ValueError):
        et.check_contrasts(mutate(read(RUN), CONTRAST_EDITS[edit]))


def test_contrast_changes_exactly_the_embedding_map(monkeypatch):
    assert et.check_contrasts(read(RUN)) == et.CONTRASTS
    monkeypatch.setitem(et.FACTORS, 'linear_residual_counts',
                        ('linear', 'context_continuous+past_physics', 16, 'counts', 'residual', .1, 'blocks', 'single'))
    with pytest.raises(ValueError, match='exactly the embedding map'):
        et.check_contrasts(read(RUN))


def test_p37_never_reaches_an_earlier_family_check(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('P37 config reached an earlier family check')
    for module in (rcr, ple, trr, er):
        monkeypatch.setattr(module, 'check_family', boom)
    cfg = rc.config_check(read(RUN))
    assert rc.training_budget(cfg) == et.TRAINING and rc.uses_epoch_cap(cfg)


def test_without_the_marker_the_p32_check_refuses_this_family():
    cfg = read(RUN)
    del cfg[et.MARKER]
    with pytest.raises(ValueError):
        rc.config_check(cfg)


@pytest.mark.parametrize('path', EARLIER, ids=[p.stem for p in EARLIER])
def test_earlier_registered_configs_route_and_budget_exactly_as_before(path):
    cfg = read(path)
    assert not rc.uses_embed_type(cfg) and et.check_contrasts(cfg) is None
    rc.config_check(copy.deepcopy(cfg))
    with pytest.raises(ValueError):  # the marker can never be bolted onto an earlier family
        rc.config_check({**copy.deepcopy(cfg), et.MARKER: True})


def test_marker_constant_and_sources_are_wired():
    assert rc.EMBED_TYPE_MARKER == et.MARKER
    assert 'pitchmdp/direct_embed_type_registry.py' in rc.MODEL_SOURCES
    assert 'pitchmdp/direct_embed_type_registry.py' in sc.SCORER_SOURCES


def test_draft_refuses_real_fits(export, tmp_path):
    path, digest, _ = export
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(load(), path, digest, tmp_path / 'real', 'linear_residual_counts', 39, 'cpu', {})


def test_prereg_exists_and_states_the_fixed_family():
    text = PREREG.read_text()
    for needle in ('EXP-P37-001', '39, 40, 41, 42, 43', 'periodic_residual_counts', 'linear_residual_counts',
                   'parameter-near-matched', 'NOT compute-matched', 'EXP-P32-001', '2026', RUN_SHA256[:12],
                   GPU3, GPU4, 'exactly one at a time', 'max_parallel` 1', 'aggregate by 22:30'):
        assert needle in text, needle


# ------------------------------------------------------------------ runner (CPU smoke, unchanged embedded path)

@pytest.mark.parametrize('arm', et.ARM_ORDER)
def test_runner_smoke_task_records_cap_and_registered_parameters(export, tmp_path, arm):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', arm, 39, 'cpu', {}, smoke=True)
    fit = read(dest / 'fit.json')
    assert fit['epoch_cap']['cap'] == rc.SMOKE['epochs'] and fit['epoch_cap']['stopped_by'] == 'cap'
    record = et.check_parameters(arm, et.cost_record(fit))  # synthetic export has the registered shapes
    assert record['optimizer_updates'] == fit['report']['optimizer_updates'] > 0
    identity = read(dest / 'identity.json')
    assert identity['seed'] == 39 and identity['smoke'] is True
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    for seed in (16, 34, 38, 44):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', arm, seed, 'cpu', {}, smoke=True)


def test_cost_record_refuses_malformed_fits():
    good = {'parameter_count': 244634, 'fit_seconds': 10., 'seconds_total': 12., 'peak_device_bytes': None,
            'report': {'optimizer_updates': 100, 'epochs_run': 7, 'best_epoch': 2, 'seconds': 7.}}
    record = et.cost_record(good)
    assert record['seconds_per_update'] == pytest.approx(.1)
    assert (record['train_seconds'], record['train_seconds_per_epoch'], record['train_seconds_per_update']) == \
        pytest.approx((7., 1., .07))
    for bad in ({**good, 'fit_seconds': 0.}, {**good, 'report': {**good['report'], 'optimizer_updates': 0}},
                {**good, 'report': {**good['report'], 'epochs_run': 0}},
                {**good, 'report': {**good['report'], 'seconds': float('nan')}},
                {**good, 'report': {**good['report'], 'seconds': 0.}},
                {**good, 'report': {k: v for k, v in good['report'].items() if k != 'seconds'}},
                {k: v for k, v in good.items() if k != 'parameter_count'}, {**good, 'report': {}}):
        with pytest.raises(ValueError):
            et.cost_record(bad)
    with pytest.raises(ValueError, match='registered'):
        et.check_parameters('periodic_residual_counts', et.cost_record(good))


# ------------------------------------------------------------------ scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)


def with_records(fam, cfg, skip=(), wrong=()):
    for arm in et.ARM_ORDER:
        for seed in cfg['seeds']:
            dest = rc.task_dir(fam, arm, seed, False)
            if not dest.exists():
                continue
            fit = read(dest / 'fit.json')
            fit['report'] = {'epochs_run': 20 + seed - 39, 'best_epoch': 15 + seed - 39,
                             'optimizer_updates': 1000 + seed, 'seconds': 50. + (arm != 'control') * 25}
            fit['epoch_cap'] = rc.epoch_record(fit['report'], 60, 5)
            if (arm, seed) not in skip:
                fit.update(parameter_count=et.PARAMETERS[arm] + ((arm, seed) in wrong),
                           fit_seconds=100. + (arm != 'control') * 50 + seed, seconds_total=200.,
                           peak_device_bytes=1 << 30)
            fit['runtime']['device_uuid'] = 'GPU-a' if seed % 2 else 'GPU-b'
            rc.write_json_atomic(dest / 'fit.json', fit)
            tf.reseal(dest)
    return fam


def test_scorer_reports_the_contrast_cost_and_no_verdict(export, tmp_path, few_draws, monkeypatch):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg)

    def boom(*args, **kwargs):
        raise AssertionError('P37 config reached an earlier family check')
    for module in (rcr, ple, trr, er):
        monkeypatch.setattr(module, 'check_family', boom)
    result = tf.run_score(tmp_path, cfg, export, fam)
    assert result['decision'] is None and result['ranking'] is None
    assert not FORBIDDEN_KEYS & keys_deep({k: v for k, v in result.items() if k not in ('decision', 'ranking')})
    assert not {'embed_residual_contrasts', 'robust_clip_contrasts', 'ple_contrasts'} & set(result)
    block = result['seed_block']
    assert block['seeds'] == et.SEEDS and block['epoch_cap'] == 60 and block['primary'] == et.PRIMARY
    assert 'NOT compute-matched' in block['note'] and 'EXP-P32-001' in block['note']
    assert block['same_gpu_pairs'] == {et.PRIMARY: 5}
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(et.ARM_ORDER)
        one = result['embed_type_contrasts'][split][et.PRIMARY]
        assert (one['isolates'], one['compute_matched'], one['parameter_near_matched']) == ('embedding_map', False, True)
        vs = lambda a: arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        assert one['ensemble']['nll']['delta'] == pytest.approx(
            vs('periodic_residual_counts') - vs('linear_residual_counts'), abs=1e-9)
        assert set(one['seeds']) == {str(s) for s in cfg['seeds']}
        assert 'top_label_ece10' in arms['linear_residual_counts']['ensemble']
    cost = result['cost']
    assert cost['arms']['linear_residual_counts']['parameter_count'] == 244634
    assert cost['arms']['control']['fit_seconds_median'] == 141.
    ratio = cost['candidate_over_reference'][et.PRIMARY]['39']
    assert ratio['fit_seconds'] == pytest.approx(1.) and ratio['optimizer_updates'] == pytest.approx(1.)
    assert cost['arms']['control']['train_seconds_median'] == 50. and ratio['train_seconds'] == pytest.approx(1.)
    assert cost['arms']['control']['train_seconds_per_epoch_median'] == pytest.approx(50. / 22)
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert table.index('## dev 임베딩 함수') < table.index('## mlb_dev 임베딩 함수') < table.index('## 비용') \
        < table.index('epoch 상한(60)')


def test_scorer_refuses_missing_cost(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg,
                       skip=(('linear_residual_counts', 41),))
    with pytest.raises(ValueError, match='cost record'):
        tf.run_score(tmp_path, cfg, export, fam)


def test_scorer_refuses_a_parameter_count_off_the_registered_value(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg,
                       wrong=(('periodic_residual_counts', 42),))
    with pytest.raises(ValueError, match='registered'):
        tf.run_score(tmp_path, cfg, export, fam)


def test_scorer_refuses_missing_fit_or_contrast_key(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = with_records(tf.family(tmp_path, cfg, digest, splits, skip=(('control', 43),)), cfg)
    with pytest.raises(ValueError, match='Incomplete family'):
        tf.run_score(tmp_path, cfg, export, fam, dest='s1')
    no_key = copy.deepcopy(cfg)
    del no_key['comparison'][et.CONTRAST_KEY]
    with pytest.raises(ValueError, match='fixed contrast'):
        tf.run_score(tmp_path, no_key, export, fam, dest='s2')


# ---- single-GPU resource rule (Song 2026-10-05): GPU3 only, max 1 fit, GPU4 refused everywhere it could be chosen

def test_check_device_refuses_the_released_gpu_with_its_own_message():
    with pytest.raises(ValueError, match='released GPU'):
        et.check_device(mutate(read(RUN), EDITS['gpu4_added']))
    assert et.check_device(read(RUN)) == et.DEVICE


def test_queue_accepts_only_gpu3_alone(tmp_path):
    import cuda_direct_queue as cq
    cfg = read(RUN)
    make = lambda gpus, d: cq.Queue(cfg, tmp_path / d, gpus, lambda *a: None, lambda *a: False,  # noqa: E731
                                    lock_dir=tmp_path / 'locks')
    queue = make([GPU3], 'ok')
    assert queue.gpus == [GPU3] and list(queue.locks) == [GPU3] and len(queue.tasks) == 15
    assert queue.locks[GPU3].path.name == GPU3 + '.lock'
    for name, gpus in {'gpu4': [GPU4], 'both': [GPU3, GPU4], 'twice': [GPU3, GPU3], 'none': []}.items():
        with pytest.raises(ValueError):
            make(gpus, name)


def test_fit_binding_refuses_gpu4_before_any_cuda_call(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='not in the allowed list'):
        rc.bind_gpu(GPU4, read(RUN)['device']['allowed_gpu_uuids'])
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES') is None
