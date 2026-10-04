"""Synthetic CPU tests: P36 robust-clip timing-only TRAIN probe (AMENDMENT-P36-TIMING-PROBE-20261004).

Mirrors tests/test_timing_probe_ple_residual.py (P35 probe) for the DRAFT robust-clip family, plus the P36 specifics:
the robust-clip ``compute_stats`` inside prepare (timed separately), the exact DRAFT and module sha256 pins checked
before anything else, probe seed 39 only, two probed arms, the fixed P35 constants 464 s / 1,046 s for control and
periodic_residual_counts, peak host RSS, and the three gate booleans.  No real data, no GPU.
"""
import copy
import json
import os
from pathlib import Path
import signal
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
import timing_probe_robust_clip as tp  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402
from cuda_direct_queue import GpuLock  # noqa: E402
from pitchmdp import direct_robust_clip_model as rcm  # noqa: E402
from pitchmdp import direct_robust_clip_registry as reg  # noqa: E402

RUN = REPO / 'configs' / 'DRAFT-CUDA-ROBUST-CLIP-20261005.json'
DRAFT_SHA = '644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a'
MODULE_SHA = 'a3d6cbcef8350586a23d1a6d35213e36ba8e0ccf21288288464f8ce7dcbf52fa'
GPUS = ['GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9', 'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99']
ARMS = ['residual_counts', 'robust_clip_residual_counts']
RC = 'robust_clip_residual_counts'
FAKE = Path(__file__).resolve().parent / 'fake_robust_clip_probe_worker.py'
WINDOW_BASE = '08da77f'  # the base of this probe's branch (robust-clip review disposition); frozen files fixed after it
FROZEN = ['experiments/pitchmdp/pitchmdp', 'experiments/pitchmdp/scripts/run_cuda_direct.py',
          'experiments/pitchmdp/scripts/cuda_direct_queue.py', 'experiments/pitchmdp/scripts/score_cuda_direct.py',
          'experiments/pitchmdp/scripts/timing_probe_ple_residual.py',
          'experiments/pitchmdp/scripts/timing_probe_tabm_residual.py',
          'experiments/pitchmdp/tests/test_timing_probe_ple_residual.py',
          'experiments/pitchmdp/tests/test_timing_probe_tabm_residual.py',
          'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P36-001-robust-clip-2026-10-04.md',
          'experiments/pitchmdp/docs/launch', 'configs']


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """Mac shares the machine with live recording: one native thread; the fixed N8 env for the probe's own check."""
    torch.set_num_threads(1)
    for key, value in tp.LAUNCH_ENV.items():
        monkeypatch.setenv(key, value)


def load():
    return tf.load(RUN, False, True)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


def train_of(export):
    root, digest, _ = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    return cfg, manifest, rc.load_split(root, manifest, 'train')


# ------------------------------------------------------------------ config, seed, DRAFT and module pins

def test_config_check_accepts_only_the_draft_robust_clip_family():
    assert tp.check_config(load()) == ARMS
    assert tp.CONFIG_ARMS == list(reg.ARM_ORDER)
    bad = load()
    bad['experiment_id'] = 'EXP-P36-001'
    with pytest.raises(ValueError, match='only for the DRAFT robust-clip'):
        tp.check_config(bad)
    bad = load()
    bad['arms'] = bad['arms'][:3]
    with pytest.raises(ValueError):
        tp.check_config(bad)
    bad = load()
    bad['training']['epochs'] = 30
    with pytest.raises(ValueError):
        tp.check_config(bad)
    for other in ('DRAFT-CUDA-PLE-RESIDUAL-20261004.json', 'DRAFT-CUDA-TABM-RESIDUAL-20261004.json',
                  'EXP-P34-001.json'):
        with pytest.raises(ValueError):
            tp.check_config(tf.load(REPO / 'configs' / other, False, True))


@pytest.mark.parametrize('seed', [34, 35, 36, 37, 38, 0, 29, 33, 40, 1000])
def test_any_seed_other_than_39_is_refused(seed, monkeypatch, export):
    with pytest.raises(ValueError, match='must be 39'):
        tp.check_seed(seed, load())
    cfg, _, train = train_of(export)
    with pytest.raises(ValueError, match='must be 39'):
        tp.probe_arm(cfg, RC, train, cfg['export']['workload_columns'], 10, 10, 'cpu', seed=seed)
    if seed in (34, 35, 36, 37, 38, 0, 29, 33):  # a used seed is refused even as the module constant
        monkeypatch.setattr(tp, 'PROBE_SEED', seed)
        with pytest.raises(ValueError, match='must be 39'):
            tp.check_config(load())


def test_probe_seed_is_39_and_fixed():
    assert tp.PROBE_SEED == 39 and tp.check_seed(39, load()) == 39
    assert tp.FAMILY_SEEDS == (34, 35, 36, 37, 38) == tuple(load()['seeds']) == tuple(reg.SEEDS)
    assert 39 not in {s for block in reg.EARLIER_SEEDS.values() for s in block}


def test_config_file_pin_is_the_exact_reviewed_draft(tmp_path):
    assert tp.CONFIG_SHA256 == DRAFT_SHA == rc.sha256_file(RUN)
    tp.check_config_file(RUN, tp.CONFIG_SHA256)
    with pytest.raises(ValueError, match='not the reviewed'):
        tp.check_config_file(RUN, '0' * 64)
    copy_ = tmp_path / RUN.name
    copy_.write_bytes(RUN.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='not the reviewed'):
        tp.check_config_file(copy_, rc.sha256_file(copy_))  # self-consistent pin of an edited file
    with pytest.raises(ValueError, match='differs from the pinned'):
        tp.check_config_file(copy_, tp.CONFIG_SHA256)


def test_module_pin_is_the_reviewed_module(monkeypatch):
    assert tp.MODULE_SHA256 == MODULE_SHA == reg.MODULE_SHA256 == rc.sha256_file(PROJECT / tp.MODULE)
    assert tp.check_module_pin() == MODULE_SHA
    monkeypatch.setattr(tp, 'MODULE_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='Registry module pin'):
        tp.check_module_pin()
    monkeypatch.setattr(tp, 'MODULE_SHA256', MODULE_SHA)
    monkeypatch.setattr(reg, 'MODULE_SHA256', '1' * 64)
    with pytest.raises(ValueError, match='Registry module pin'):
        tp.check_module_pin()
    monkeypatch.setattr(reg, 'MODULE_SHA256', MODULE_SHA)
    real = rc.sha256_file
    monkeypatch.setattr(rc, 'sha256_file', lambda p: '2' * 64 if Path(p).name == 'direct_robust_clip_model.py'
                        else real(p))
    with pytest.raises(ValueError, match='module differs'):
        tp.check_module_pin()


def test_launch_env_is_required():
    assert tp.check_env(dict(tp.LAUNCH_ENV)) == tp.LAUNCH_ENV
    for key in tp.LAUNCH_ENV:
        with pytest.raises(ValueError, match='Launch environment'):
            tp.check_env({**tp.LAUNCH_ENV, key: '1'})
    with pytest.raises(ValueError):
        tp.check_env({})


# ------------------------------------------------------------------ measurement (CPU, synthetic export)

@pytest.mark.parametrize('arm', ARMS)
def test_probe_arm_reads_train_only_and_records_timing_only(export, monkeypatch, arm, tmp_path):
    root, digest, splits = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    opened = []
    real = rc.load_split
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]) or real(*a, **k))
    n_train, n_early, n_predict = tp.row_counts(manifest)
    train = rc.load_split(root, manifest, 'train')
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    calls = []
    record = tp.probe_arm(cfg, arm, train, cfg['export']['workload_columns'], n_early, n_predict, 'cpu',
                          sync=lambda: calls.append('sync'))
    assert opened == ['train']
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before  # nothing written (no runs/ task)
    assert record['seed'] == 39
    assert record['warmup']['epochs'] == 1 and record['measured']['epochs'] == 2
    per_epoch = -(-n_train // cfg['training']['batch_size'])
    assert record['warmup']['optimizer_updates'] == per_epoch
    assert record['measured']['optimizer_updates'] == 2 * per_epoch
    assert record['validation_standin_rows'] == n_early == len(splits['earlystop']['y'])
    assert record['predict_standin_rows'] == n_predict == sum(len(splits[s]['y']) for s in tp.PREDICT_SPLITS)
    m = record['measured']
    assert record['measured_seconds_per_epoch'] == max(m['wall_seconds'], m['fit_report_seconds']) / 2
    assert len(calls) == 6
    for rss in (record['host_peak_rss_after_prepare'], m['host_peak_rss'], record['host_peak_rss']):
        assert set(rss) == {'ru_maxrss', 'unit', 'bytes'} and rss['ru_maxrss'] > 0
    assert record['host_peak_rss']['ru_maxrss'] >= record['host_peak_rss_after_prepare']['ru_maxrss']  # monotone peak
    if arm == RC:
        inputs = rc.arm_inputs(train, rc.arm_spec(cfg, arm), cfg['export']['workload_columns'])
        center, scale, rules = tp.rc_stats(inputs)
        assert record['rc']['stats_sha256'] == rcm.stats_sha256(center, scale)
        assert 0 < record['rc']['compute_stats_seconds'] <= record['prepare_seconds']
        assert sum(record['rc']['rule_counts'].values()) == reg.n_scalars(inputs[0].shape[1]) == 81
        assert record['rc']['stats_rows'] == rc.ple_bin_rows(inputs)
        assert record['rc']['module_sha256'] == MODULE_SHA
        assert set(record['rc']['rule_columns']) == {'range', 'constant'}
    else:
        assert 'rc' not in record
    tp.no_forbidden_keys(record)
    text = json.dumps(record).lower()
    assert 'nll' not in text and 'history' not in text and 'best_epoch' not in text and 'logit' not in text


def test_probe_refuses_an_arm_it_does_not_probe(export):
    cfg, _, train = train_of(export)
    for name in ('control', 'periodic_residual_counts'):
        with pytest.raises(ValueError, match='not probed'):
            tp.probe_arm(cfg, name, train, cfg['export']['workload_columns'], 10, 10, 'cpu')


def test_rc_stats_of_the_probe_are_the_stats_of_the_frozen_fit(export):
    cfg, _, train = train_of(export)
    inputs = rc.arm_inputs(train, rc.arm_spec(cfg, RC), cfg['export']['workload_columns'])
    model = rc.build_model(rc.arm_spec(cfg, RC), tp.PROBE_SEED, 'cpu')
    labels = np.array(train['y'])
    model.fit(inputs, labels, *tp.take(inputs, labels, 50), epochs=1, patience=5, batch_size=1024, learning_rate=5e-4)
    center, scale, rules = tp.rc_stats(inputs)
    assert model.report['rc_stats_sha256'] == rcm.stats_sha256(center, scale)
    assert model.report['rc_rule_counts'] == {r: rules.count(r) for r in ('iqr', 'range', 'constant')}


def test_probe_arm_refuses_stats_that_differ_from_the_fit(export, monkeypatch):
    cfg, _, train = train_of(export)
    real = tp.rc_stats
    monkeypatch.setattr(tp, 'rc_stats', lambda inputs: real(tuple(a[:50] for a in inputs)))
    with pytest.raises(RuntimeError, match='other robust-scale statistics'):
        tp.probe_arm(cfg, RC, train, cfg['export']['workload_columns'], 50, 50, 'cpu')


def test_validation_and_predict_standins_are_train_rows_of_manifest_counts(export, monkeypatch):
    cfg, manifest, train = train_of(export)
    n_train, n_early, n_predict = tp.row_counts(manifest)
    seen = {}
    Model = rcm.DirectRobustClipResidualModel
    real_fit, real_logits = Model.fit, Model.logits

    def fit(self, inputs, labels, early, early_y, **kw):
        seen.setdefault('early', []).append((early[0].shape[0], np.array_equal(early[0], inputs[0][:n_early])))
        return real_fit(self, inputs, labels, early, early_y, **kw)

    def logits(self, arrays, batch_size=4096):
        seen.setdefault('logits', []).append(arrays[0].shape[0])
        return real_logits(self, arrays, batch_size)

    monkeypatch.setattr(Model, 'fit', fit)
    monkeypatch.setattr(Model, 'logits', logits)
    tp.probe_arm(cfg, RC, train, cfg['export']['workload_columns'], n_early, n_predict, 'cpu')
    assert seen['early'] == [(n_early, True), (n_early, True)]
    assert seen['logits'][-1] == n_predict  # earlier entries: per-epoch validation inside fit (n_early each)
    assert seen['logits'][:-1] == [n_early] * 3


class FakeModel:
    """Stands in for build_model: fixed report seconds, a measurable wall time, one parameter."""

    def __init__(self, report_seconds, sleep):
        self.report_seconds, self.sleep, self.net, self.device = report_seconds, sleep, torch.nn.Linear(1, 1), 'cpu'

    def fit(self, inputs, labels, early, early_y, epochs, **kw):
        import time
        time.sleep(self.sleep)
        self.report = {'epochs_run': epochs, 'optimizer_updates': 10 * epochs, 'seconds': self.report_seconds}

    def logits(self, arrays):
        return None


@pytest.mark.parametrize('report_seconds,sleep', [(1000., 0.), (0., .05)])
def test_per_epoch_is_max_of_wall_and_report_over_two(export, monkeypatch, report_seconds, sleep):
    cfg, _, train = train_of(export)
    monkeypatch.setattr(rc, 'build_model', lambda arm, seed, device: FakeModel(report_seconds, sleep))
    record = tp.probe_arm(cfg, 'residual_counts', train, cfg['export']['workload_columns'], 50, 50, 'cpu')
    m = record['measured']
    if report_seconds:
        assert m['wall_seconds'] < 1. and record['measured_seconds_per_epoch'] == 500.
    else:
        assert m['fit_report_seconds'] == 0. and record['measured_seconds_per_epoch'] == m['wall_seconds'] / 2 >= .025
    assert record['measured_seconds_per_update'] == report_seconds / 20


def test_row_counts_refuse_train_smaller_than_standins():
    manifest = {'splits': {'train': {'n': 100}, 'earlystop': {'n': 120}, 'temperature': {'n': 1}, 'blend': {'n': 1},
                           'dev': {'n': 1}, 'mlb_dev': {'n': 1}}}
    with pytest.raises(ValueError):
        tp.row_counts(manifest)
    manifest['splits']['earlystop']['n'] = 50
    assert tp.row_counts(manifest) == (100, 50, 4)
    manifest['splits']['dev']['n'] = 200
    with pytest.raises(ValueError):
        tp.row_counts(manifest)


def test_host_peak_rss_units():
    rss = tp.host_peak_rss()
    linux = sys.platform.startswith('linux')
    assert rss['unit'] == ('KiB' if linux else 'bytes')
    assert rss['bytes'] == rss['ru_maxrss'] * (1024 if linux else 1) > 0


# ------------------------------------------------------------------ no model artifact, no DEV / MLB DEV row

def test_probe_never_saves_a_model(export, monkeypatch, tmp_path):
    cfg, manifest, train = train_of(export)
    from pitchmdp import direct_residual_model as drm
    from pitchmdp import matrix_models as mm

    def refuse(*a, **k):
        raise AssertionError('probe tried to save a model')

    for cls in (rcm.DirectRobustClipResidualModel, drm.DirectResidualModel, mm.MatrixModel):
        monkeypatch.setattr(cls, 'save', refuse)
    monkeypatch.setattr(torch, 'save', refuse)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    _, n_early, n_predict = tp.row_counts(manifest)
    for arm in ARMS:
        tp.probe_arm(cfg, arm, train, cfg['export']['workload_columns'], n_early, n_predict, 'cpu')
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before
    source = (PROJECT / tp.SELF).read_text()
    assert '.save(' not in source and 'checkpoint' not in source and 'savez' not in source


def test_worker_end_to_end_on_cpu_never_loads_dev_rows(export, tmp_path, monkeypatch):
    """The real probe_arm on CPU through the worker: only TRAIN arrays are np.load-ed; no file but the worker JSON."""
    root, digest, _ = export
    args, cfg = worker_args(tmp_path, root, digest, monkeypatch)
    loaded, real_load, real_probe = [], np.load, tp.probe_arm
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: {'bound_uuid': uuid})
    monkeypatch.setattr(rc.np, 'load', lambda path, *a, **k: loaded.append(Path(path)) or real_load(path, *a, **k))
    real_split = rc.load_split

    def train_only(export_, manifest_, split, *a, **k):
        if split != 'train':
            raise AssertionError('probe opened split ' + split)
        return real_split(export_, manifest_, split, *a, **k)

    monkeypatch.setattr(rc, 'load_split', train_only)
    monkeypatch.setattr(tp, 'probe_arm', lambda config, name, train, columns, n_early, n_predict, device, **kw:
                        real_probe(config, name, train, columns, n_early, n_predict, 'cpu'))
    tp.worker(args, cfg)
    manifest = json.loads((root / 'manifest.json').read_text())
    train_files = {(root / meta['path']).resolve() for meta in manifest['splits']['train']['files'].values()}
    assert loaded and {p.resolve() for p in loaded} <= train_files
    out = json.loads((args.output / f'worker-{GPUS[0]}.json').read_text())
    assert [a['arm'] for a in out['arms']] == ARMS and out['host_peak_rss_after_startup']['ru_maxrss'] > 0
    assert sorted(p.name for p in args.output.iterdir()) == ['preflight.json', f'worker-{GPUS[0]}.json']


# ------------------------------------------------------------------ projection and gates

def fake_record(arm, per_epoch, prepare=2., predict=3., stats=1.):
    record = {'arm': arm, 'prepare_seconds': prepare, 'predict_seconds': predict, 'measured_seconds_per_epoch': per_epoch}
    if arm == RC:
        record['rc'] = {'compute_stats_seconds': stats}
    return record


def workers_with(per_epoch, startup=(10., 12.)):
    """Two GPUs; per_epoch maps (gpu index, arm) -> seconds, default 5."""
    return [{'gpu': g, 'startup_seconds': startup[i],
             'arms': [fake_record(a, per_epoch.get((i, a), 5.)) for a in ARMS]} for i, g in enumerate(GPUS)]


OK = {g: 0 for g in GPUS}


def test_fixed_reference_constants_and_their_citation():
    assert tp.FIXED_TASK_SECONDS == {'control': 464., 'periodic_residual_counts': 1046.}
    assert tp.FIXED_SOURCE == 'docs/RESULT-P35-TIMING-PROBE-3b317f2-20261004.md:30-31'
    path, lines = tp.FIXED_SOURCE.split(':')
    first, last = (int(x) for x in lines.split('-'))
    cited = (PROJECT / path).read_text().splitlines()[first - 1:last]
    assert cited[0].startswith('| control |') and '| 464 s |' in cited[0]
    assert cited[1].startswith('| periodic_residual_counts |') and '| 1,046 s |' in cited[1]
    out = tp.summarize(workers_with({}), 5, OK)
    assert out['fixed_task_seconds'] == {'arms': {'control': 464., 'periodic_residual_counts': 1046.},
                                         'source': tp.FIXED_SOURCE, 'measured_here': False}
    assert out['slowest_task_seconds_projected']['control'] == 464.
    assert out['slowest_task_seconds_projected']['periodic_residual_counts'] == 1046.
    assert list(out['slowest_task_seconds_projected']) == tp.CONFIG_ARMS
    assert 'control' not in out['per_arm'] and 'periodic_residual_counts' not in out['per_arm']


def test_projection_and_gate_arithmetic():
    workers = workers_with({(1, RC): 6., (1, 'residual_counts'): 4.})
    out = tp.summarize(workers, 5, OK)
    # residual: GPU0 10 + 2 + 300 + 3 + 30 = 345 (GPU1 12 + 2 + 240 + 33 = 287); robust-clip GPU1 12 + 2 + 360 + 33 = 407
    assert out['probed_task_seconds_projected'] == {'residual_counts': pytest.approx(345.), RC: pytest.approx(407.)}
    assert out['family_worker_seconds_projected'] == pytest.approx(5 * (464. + 345. + 1046. + 407.))
    assert out['gate'] == {'task_seconds_max': 1800, 'family_seconds_max': 36000, 'every_arm_within': True,
                           'family_within': True, 'probe_exit_zero': True}
    assert out['marginal_arms'] == []
    row = out['per_arm'][RC][0]
    assert row['compute_stats_seconds_measured'] == 1.
    assert row['compute_stats_seconds_inside_projection_not_credited'] == pytest.approx(30.)
    assert row['task_seconds_projected'] == pytest.approx(345.)  # not credited: the projection keeps it
    assert 'compute_stats_seconds_measured' not in out['per_arm']['residual_counts'][0]


def test_failing_gates():
    # robust-clip on GPU0: 10 + 2 + 60 x 29.5 + 3 + 30 = 1815 > 1800 -> every_arm_within false, marginal
    out = tp.summarize(workers_with({(0, RC): 29.5}), 5, OK)
    assert out['gate']['every_arm_within'] is False and out['gate']['family_within'] is True
    assert out['marginal_arms'] == [RC]
    # the reference arm alone over the gate also fails it
    out = tp.summarize(workers_with({(1, 'residual_counts'): 29.5}), 5, OK)
    assert out['gate']['every_arm_within'] is False
    # exactly 1800 (45 + 1755) is within (<=)
    assert tp.summarize(workers_with({(0, RC): 29.25}), 5, OK)['gate']['every_arm_within'] is True
    # family: with control 464 and periodic 1046 fixed, two arms <= 1800 give at most 5 x 5,110 = 25,550, so the
    # family gate can only fail together with an arm gate.  residual_counts slowest = GPU1 12 + 2 + 300 + 33 = 347;
    # robust-clip GPU0 10 + 2 + 6000 + 33 = 6045 -> 5 x (464 + 347 + 1046 + 6045) = 39,510
    out = tp.summarize(workers_with({(0, RC): 100.}), 5, OK)
    assert out['family_worker_seconds_projected'] == pytest.approx(5 * (464. + 347. + 1046. + 6045.))
    assert out['gate']['family_within'] is False and out['gate']['every_arm_within'] is False
    # family boundary: 5 x (1510 + 347 + x) <= 36000 <=> x <= 5343 = 45 + 60 x per_epoch (5298)
    edge = tp.summarize(workers_with({(0, RC): 5298. / 60}), 5, OK)
    assert edge['family_worker_seconds_projected'] == pytest.approx(36000.) and edge['gate']['family_within'] is True
    over = tp.summarize(workers_with({(0, RC): 5299. / 60}), 5, OK)  # 36,005 > 36,000
    assert over['gate']['family_within'] is False
    # probe exit: any non-zero code is false
    for codes in ({GPUS[0]: 0, GPUS[1]: 1}, {GPUS[0]: 'timeout', GPUS[1]: 0}, {}):
        assert tp.summarize(workers_with({}), 5, codes)['gate']['probe_exit_zero'] is False


def test_summary_refuses_missing_or_extra_arms():
    workers = workers_with({})
    with pytest.raises(ValueError):
        tp.summarize([workers[0], {**workers[1], 'arms': workers[1]['arms'][:1]}], 5, OK)
    with pytest.raises(ValueError):
        tp.summarize([{**w, 'arms': w['arms'][::-1]} for w in workers], 5, OK)
    with pytest.raises(ValueError):
        tp.summarize([{**w, 'arms': [fake_record('control', 1.), *w['arms']]} for w in workers], 5, OK)


def test_marginal_band_is_two_sided():
    # 45 s fixed + 60 x per_epoch: 1650 (150 below), 1950 (150 above)
    out = tp.summarize(workers_with({(0, 'residual_counts'): 26.75, (0, RC): 31.75}), 5, OK)
    assert out['marginal_arms'] == ARMS
    out = tp.summarize(workers_with({(0, 'residual_counts'): 25.5, (0, RC): 32.5}), 5, OK)  # 1575, 1995
    assert out['marginal_arms'] == []


def test_forbidden_output_keys_refused():
    for bad in ({'arms': [{'measured': {'history': []}}]}, {'earlystop_conditional_nll': 1.}, {'dev_logits': []},
                {'probabilities': []}, {'Brier': 1.}, {'ece10': 1.}, {'score': 1.}):
        with pytest.raises(ValueError):
            tp.no_forbidden_keys(bad)
    assert tp.no_forbidden_keys({'measured': {'fit_report_seconds': 1.}, 'peak_device_bytes': 1, 'rc': {},
                                 'host_peak_rss': {'ru_maxrss': 1}})


# ------------------------------------------------------------------ parent: preflight, locks, timeouts

def args_for(tmp_path, root, digest, monkeypatch, output='probe-timing', gpus=GPUS):
    """A synthetic config copy (export pin removed) whose sha is pinned in place of the DRAFT's for the run path."""
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(load()))
    monkeypatch.setattr(tp, 'CONFIG_SHA256', rc.sha256_file(cfg_path))
    ns = type('A', (), {})()
    ns.config, ns.config_sha256, ns.export, ns.export_manifest_sha256 = cfg_path, rc.sha256_file(cfg_path), root, digest
    ns.output, ns.lock_dir, ns.gpu, ns.python = tmp_path / output, tmp_path / 'locks', list(gpus), sys.executable
    return ns, json.loads(cfg_path.read_text())


def test_preflight_refusals(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    ident = tp.preflight(args, cfg)
    assert ident['row_counts'] == {'train': 400, 'earlystop': 120, 'predict': 230}
    assert ident['module_sha256'] == MODULE_SHA
    assert ident['fixed']['warmup_epochs'] == 1 and ident['fixed']['measure_epochs'] == 2
    assert ident['fixed']['probe_seed'] == 39 and ident['fixed']['launch_env'] == tp.LAUNCH_ENV
    assert ident['fixed']['probed_arms'] == ARMS
    assert ident['fixed']['fixed_task_seconds'] == {'control': 464., 'periodic_residual_counts': 1046.}
    assert set(ident['source_sha256']) >= {tp.SELF, tp.QUEUE, rc.SELF, tp.MODULE,
                                           'pitchmdp/direct_robust_clip_registry.py',
                                           'pitchmdp/direct_robust_clip_factory.py'}
    assert 'scripts/timing_probe_ple_residual.py' not in ident['source_sha256']
    for change in ({'output': tmp_path / 'probe'}, {'output': tmp_path / 'runs-timing'},
                   {'gpu': [GPUS[0], GPUS[0]]}, {'gpu': [GPUS[0]]}, {'gpu': [GPUS[0], 'GPU-other']},
                   {'config_sha256': '0' * 64}, {'export_manifest_sha256': '0' * 64}):
        bad, _ = args_for(tmp_path, root, digest, monkeypatch)
        for k, v in change.items():
            setattr(bad, k, v)
        with pytest.raises(ValueError):
            tp.preflight(bad, cfg)
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    with pytest.raises(ValueError, match='Launch environment'):
        tp.preflight(args, cfg)


def test_preflight_checks_draft_and_module_pins_before_anything_else(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    touched = []
    monkeypatch.setattr(rc, 'verify_export', lambda *a, **k: touched.append('export'))
    monkeypatch.setattr(rc, 'config_check', lambda c: touched.append('config_check') or c)
    monkeypatch.setattr(tp, 'check_env', lambda: touched.append('env'))
    monkeypatch.setattr(tp, 'MODULE_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='module pin'):
        tp.preflight(args, cfg)
    assert touched == []
    monkeypatch.setattr(tp, 'MODULE_SHA256', MODULE_SHA)
    monkeypatch.setattr(tp, 'CONFIG_SHA256', rc.sha256_file(RUN))  # the shipped pin: the synthetic copy is refused
    with pytest.raises(ValueError, match='not the reviewed'):
        tp.preflight(args, cfg)
    assert touched == []


def test_host_snapshot_fields():
    snap = tp.host_snapshot()
    assert snap['nproc'] == os.cpu_count() and len(snap['loadavg']) == 3 and snap['env'] == tp.LAUNCH_ENV
    assert {'nvidia_smi_gpus', 'nvidia_smi_compute_apps'} <= set(snap)  # an error record on the Mac (no nvidia-smi)
    tp.no_forbidden_keys(snap)


def test_run_refuses_busy_gpu_lock_and_existing_output(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    holder = GpuLock(args.lock_dir, GPUS[1])
    assert holder.acquire()
    try:
        with pytest.raises(RuntimeError, match='lock busy'):
            tp.run(args, cfg)
    finally:
        holder.release()
    assert not list(args.output.glob('worker-*'))  # nothing launched
    assert GpuLock(args.lock_dir, GPUS[0]).acquire()  # the first lock was released
    with pytest.raises(FileExistsError):
        tp.run(args, cfg)  # output already exists (never reused)


def fake_worker_python(tmp_path, body='exec sleep 60'):
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/sh\n' + body + '\n')
    fake.chmod(0o755)
    return str(fake)


def group_gone(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_run_failed_workers_release_locks_and_write_incomplete_summary(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_worker_python(tmp_path, 'exit 3')
    summary = tp.run(args, cfg)
    assert summary['complete'] is False and summary['exit_codes'] == {g: 3 for g in GPUS}
    assert 'gate' not in summary and 'host_post' in summary and 'host_pre' in summary['preflight']
    assert all(GpuLock(args.lock_dir, g).acquire() for g in GPUS)
    with pytest.raises(SystemExit):
        cmd = ['--config', str(args.config), '--config-sha256', args.config_sha256, '--export', str(root),
               '--export-manifest-sha256', digest, '--output', str(tmp_path / 'second-probe-timing'),
               '--lock-dir', str(tmp_path / 'locks2'), '--gpu', GPUS[0], '--gpu', GPUS[1],
               '--python', fake_worker_python(tmp_path, 'exit 3'), 'run']
        tp.main(cmd)  # run exits non-zero when incomplete


def test_run_interrupt_stops_worker_groups_before_releasing_locks(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_worker_python(tmp_path)
    pids, alive_at_release, handlers = [], [], []
    before = [signal.getsignal(s) for s in tp.SIGNALS]
    real_popen, real_release = tp.subprocess.Popen, GpuLock.release

    class Popen(real_popen):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            pids.append(self.pid)

        def wait(self, timeout=None):
            if timeout is not None and timeout > 100:  # the parent's worker wait (not a stop grace): interrupted
                handlers.extend(signal.getsignal(s) for s in tp.SIGNALS)
                raise KeyboardInterrupt
            return super().wait(timeout)

    def release(lock):
        alive_at_release.extend(p for p in pids if not group_gone(p))
        real_release(lock)

    monkeypatch.setattr(tp.subprocess, 'Popen', Popen)
    monkeypatch.setattr(GpuLock, 'release', release)
    try:
        with pytest.raises(KeyboardInterrupt):
            tp.run(args, cfg)
        assert len(pids) == 2 and alive_at_release == []  # no lock released while an own worker group lived
        assert all(group_gone(p) for p in pids)
        assert all(GpuLock(args.lock_dir, g).acquire() for g in GPUS)
        assert handlers == [tp._interrupt] * 3  # SIGTERM/SIGHUP/SIGINT reach the cleanup while workers run
        assert [signal.getsignal(s) for s in tp.SIGNALS] == before  # and are restored afterwards
    finally:
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_run_worker_timeout_kills_groups_and_writes_incomplete_summary(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_worker_python(tmp_path)
    monkeypatch.setattr(tp, 'WORKER_TIMEOUT_SECONDS', 1)
    pids = []
    real_popen = tp.subprocess.Popen
    monkeypatch.setattr(tp.subprocess, 'Popen', lambda *a, **k: pids.append(real_popen(*a, **k)) or pids[-1])
    summary = tp.run(args, cfg)
    assert summary['complete'] is False and summary['exit_codes'] == {g: 'timeout' for g in GPUS}
    assert 'gate' not in summary and json.loads((args.output / 'summary.json').read_text())['complete'] is False
    assert all(group_gone(p.pid) for p in pids)
    assert all(GpuLock(args.lock_dir, g).acquire() for g in GPUS)
    assert sorted(p.name for p in args.output.iterdir()) == sorted(
        ['preflight.json', 'summary.json'] + [f'worker-{g}.log' for g in GPUS])  # no task / model output


def test_parent_preflight_and_lock_never_import_torch(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, _ = args_for(tmp_path, root, digest, monkeypatch)
    code = ('import json, sys, types; from pathlib import Path; sys.path.insert(0, sys.argv[1]);'
            'import timing_probe_robust_clip as tp; from cuda_direct_queue import GpuLock;'
            'a = json.loads(sys.argv[2]); tp.CONFIG_SHA256 = a["config_sha256"];'
            'ns = types.SimpleNamespace(**{k: Path(v) if k in ("config", "export", "output", "lock_dir")'
            ' else v for k, v in a.items()}); tp.preflight(ns, json.loads(ns.config.read_text())); tp.host_snapshot();'
            'print("torch" in sys.modules)')
    payload = {'config': str(args.config), 'config_sha256': args.config_sha256, 'export': str(root),
               'export_manifest_sha256': digest, 'output': str(args.output), 'lock_dir': str(args.lock_dir),
               'gpu': GPUS}
    out = subprocess.run([sys.executable, '-c', code, str(PROJECT / 'scripts'), json.dumps(payload)],
                         capture_output=True, text=True, timeout=120, check=True, env={**os.environ}).stdout.split()
    assert out == ['False']


# ------------------------------------------------------------------ worker (bind_gpu stubbed: no CUDA on the Mac)

def worker_args(tmp_path, root, digest, monkeypatch):
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.output.mkdir()
    args.gpu_uuid = GPUS[0]
    rc.write_json_atomic(args.output / 'preflight.json', tp.preflight(args, cfg))
    return args, cfg


def test_worker_opens_train_only_and_passes_manifest_standins(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = worker_args(tmp_path, root, digest, monkeypatch)
    opened, probed = [], []
    real = rc.load_split
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: {'bound_uuid': uuid})
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]) or real(*a, **k))
    monkeypatch.setattr(tp, 'probe_arm', lambda config, name, train, columns, n_early, n_predict, device, **kw:
                        probed.append((name, device, len(train['y']), n_early, n_predict)) or fake_record(name, 1.))
    tp.worker(args, cfg)
    assert opened == ['train']
    assert [p[0] for p in probed] == ARMS and {p[1:] for p in probed} == {('cuda', 400, 120, 230)}
    out = json.loads((args.output / f'worker-{GPUS[0]}.json').read_text())
    assert [a['arm'] for a in out['arms']] == ARMS and out['gpu'] == GPUS[0]


def test_worker_refuses_changed_sources_before_loading_data(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = worker_args(tmp_path, root, digest, monkeypatch)
    preflight = json.loads((args.output / 'preflight.json').read_text())
    preflight['source_sha256'][tp.SELF] = '0' * 64
    rc.write_json_atomic(args.output / 'preflight.json', preflight)
    opened = []
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: {'bound_uuid': uuid})
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]))
    with pytest.raises(RuntimeError, match='Sources differ'):
        tp.worker(args, cfg)
    assert opened == [] and not (args.output / f'worker-{GPUS[0]}.json').exists()


def test_worker_checks_draft_and_module_pins_first_then_env(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = worker_args(tmp_path, root, digest, monkeypatch)
    opened, bound = [], []
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: bound.append(uuid) or {'bound_uuid': uuid})
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]))
    monkeypatch.setattr(tp, 'MODULE_SHA256', '0' * 64)
    monkeypatch.setenv('MKL_NUM_THREADS', '8')  # a wrong env too: the module pin is refused first
    with pytest.raises(ValueError, match='module pin'):
        tp.worker(args, cfg)
    monkeypatch.setattr(tp, 'MODULE_SHA256', MODULE_SHA)
    with pytest.raises(ValueError, match='Launch environment'):
        tp.worker(args, cfg)
    assert bound == []  # refused before the GPU is bound
    monkeypatch.setenv('MKL_NUM_THREADS', '2')
    args.config.write_text(args.config.read_text() + '\n')
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    with pytest.raises(ValueError, match='Config differs'):  # the DRAFT sha is refused before the env
        tp.worker(args, cfg)
    assert opened == [] and bound == []


# ------------------------------------------------------------------ frozen files

def test_frozen_runner_queue_models_configs_and_prereg_unchanged_in_the_probe_window():
    # Window 08da77f (this branch's base, the robust-clip review disposition) .. working tree.  Once the probe has
    # run, re-scope the end of the window to the archived probe commit (as T70c did for the P34 probe test).
    changed = subprocess.run(['git', 'diff', '--name-only', WINDOW_BASE, '--', *FROZEN], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split()
    assert changed == []
    assert rc.sha256_file(PROJECT / tp.MODULE) == MODULE_SHA


# ------------------------------------------------------------------ full parent run with a fake worker

def fake_python(tmp_path):
    path = tmp_path / 'fake-python'
    path.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE} "$@"\n')
    path.chmod(0o755)
    return str(path)


def test_complete_run_writes_gates_and_passes_the_forbidden_key_check(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_python(tmp_path)
    summary = tp.run(args, cfg)
    assert summary['complete'] is True and summary['exit_codes'] == {g: 0 for g in GPUS}
    assert set(summary['gate']) == {'task_seconds_max', 'family_seconds_max', 'every_arm_within', 'family_within',
                                    'probe_exit_zero'}
    assert all(summary['gate'][k] is True for k in ('every_arm_within', 'family_within', 'probe_exit_zero'))
    task = 10 + 2 + 300 + 3 + 30
    assert summary['slowest_task_seconds_projected'] == {'control': 464., 'residual_counts': pytest.approx(task),
                                                         'periodic_residual_counts': 1046., RC: pytest.approx(task)}
    assert summary['family_worker_seconds_projected'] == pytest.approx(5 * (464 + 1046 + 2 * task))
    on_disk = json.loads((args.output / 'summary.json').read_text())
    assert on_disk['gate'] == summary['gate'] and tp.no_forbidden_keys(on_disk)


def test_partial_run_never_writes_gates(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_python(tmp_path)
    monkeypatch.setenv('PZ_FAKE_EXIT_' + GPUS[1].replace('-', '_'), '4')  # JSON written, then a non-zero exit
    summary = tp.run(args, cfg)
    assert summary['complete'] is False and summary['exit_codes'] == {GPUS[0]: 0, GPUS[1]: 4}
    assert 'gate' not in summary and 'per_arm' not in summary
    assert 'gate' not in json.loads((args.output / 'summary.json').read_text())


def test_both_workers_share_one_deadline(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    fake = tmp_path / 'sleep-python'
    fake.write_text('#!/bin/sh\nexec sleep 60\n')
    fake.chmod(0o755)
    args.python = str(fake)
    monkeypatch.setattr(tp, 'WORKER_TIMEOUT_SECONDS', 2)
    waits, real_popen = [], tp.subprocess.Popen

    class Popen(real_popen):
        def wait(self, timeout=None):
            if timeout is not None and timeout < 10:  # the parent's worker waits (not the 20 s stop grace)
                waits.append(timeout)
            return super().wait(timeout)

    monkeypatch.setattr(tp.subprocess, 'Popen', Popen)
    summary = tp.run(args, cfg)
    assert summary['exit_codes'] == {g: 'timeout' for g in GPUS}
    assert len(waits) == 2 and sum(waits) <= 2 + 1 + .5  # one deadline: the second wait gets the 1 s floor


def test_worker_sets_its_own_alarm(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, _ = args_for(tmp_path, root, digest, monkeypatch)
    alarms = []
    monkeypatch.setattr(signal, 'alarm', lambda s: alarms.append(s))
    monkeypatch.setattr(tp, 'worker', lambda a, c: None)
    tp.main(['--config', str(args.config), '--config-sha256', args.config_sha256, '--export', str(root),
             '--export-manifest-sha256', digest, '--output', str(args.output), '--lock-dir', str(args.lock_dir),
             '--gpu', GPUS[0], '--gpu', GPUS[1], 'worker', '--gpu-uuid', GPUS[0]])
    assert alarms == [tp.WORKER_TIMEOUT_SECONDS + 60] == [1560]


def test_summary_is_checked_for_forbidden_keys(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_python(tmp_path)
    monkeypatch.setattr(tp, 'host_snapshot', lambda: {'loss': 1.})
    with pytest.raises(ValueError, match='loss'):
        tp.run(args, cfg)
    assert not (args.output / 'summary.json').exists()
