"""Synthetic CPU tests: P35 PLE-embedding timing-only TRAIN probe (AMENDMENT-P35-TIMING-PROBE-20261004).

Mirrors tests/test_timing_probe_tabm_residual.py (P34 Amendment 2 at 9de203a) for the P35 DRAFT family, plus the PLE
specifics: TRAIN binning inside prepare, the exact DRAFT config name and sha256 pin, probe seed outside 29-33, the
fixed N8 launch env and the host snapshots.  No real data, no GPU.
"""
import copy
import json
import os
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
import timing_probe_ple_residual as tp  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402
from cuda_direct_queue import GpuLock  # noqa: E402
from pitchmdp import direct_ple_model as plm  # noqa: E402

RUN = REPO / 'configs' / 'DRAFT-CUDA-PLE-RESIDUAL-20261004.json'
GPUS = ['GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9', 'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99']
ARMS = ['control', 'periodic_residual_counts', 'ple_residual_counts', 'tabm_residual_counts',
        'ple_tabm_residual_counts']
PLE = ('ple_residual_counts', 'ple_tabm_residual_counts')
WINDOW_BASE = '0978558'  # the base of this probe's branch (P35 smoke record); frozen files must not move after it
FROZEN = ['experiments/pitchmdp/pitchmdp', 'experiments/pitchmdp/scripts/run_cuda_direct.py',
          'experiments/pitchmdp/scripts/cuda_direct_queue.py', 'experiments/pitchmdp/scripts/score_cuda_direct.py',
          'experiments/pitchmdp/scripts/timing_probe_tabm_residual.py',
          'experiments/pitchmdp/tests/test_timing_probe_tabm_residual.py', 'configs']


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


# ------------------------------------------------------------------ config, seed and sha pin

def test_config_check_accepts_only_the_draft_ple_family():
    assert tp.check_config(load()) == ARMS
    bad = load()
    bad['experiment_id'] = 'EXP-P35-001'
    with pytest.raises(ValueError, match='only for the DRAFT PLE'):
        tp.check_config(bad)
    bad = load()
    bad['arms'] = bad['arms'][:4]
    with pytest.raises(ValueError):
        tp.check_config(bad)
    bad = load()
    bad['training']['epochs'] = 30
    with pytest.raises(ValueError):
        tp.check_config(bad)
    for other in ('DRAFT-CUDA-TABM-RESIDUAL-20261004.json', 'EXP-P34-001.json', 'EXP-P31-001.json'):
        with pytest.raises(ValueError):
            tp.check_config(tf.load(REPO / 'configs' / other, False, True))


@pytest.mark.parametrize('seed', [29, 30, 31, 32, 33])
def test_probe_seed_inside_the_family_block_is_refused(seed, monkeypatch, export):
    with pytest.raises(ValueError, match='outside the family seeds'):
        tp.check_seed(seed, load())
    monkeypatch.setattr(tp, 'PROBE_SEED', seed)
    with pytest.raises(ValueError, match='outside the family seeds'):
        tp.check_config(load())
    root, digest, _ = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    with pytest.raises(ValueError, match='outside the family seeds'):
        tp.probe_arm(cfg, 'control', rc.load_split(root, manifest, 'train'), cfg['export']['workload_columns'],
                     10, 10, 'cpu', seed=seed)


def test_probe_seed_is_34_and_fixed():
    assert tp.PROBE_SEED == 34 and tp.check_seed(34, load()) == 34
    assert tp.FAMILY_SEEDS == (29, 30, 31, 32, 33) == tuple(load()['seeds'])


def test_config_file_pin_is_the_exact_reviewed_draft(tmp_path):
    assert rc.sha256_file(RUN) == tp.CONFIG_SHA256
    tp.check_config_file(RUN, tp.CONFIG_SHA256)
    with pytest.raises(ValueError, match='not the reviewed'):
        tp.check_config_file(RUN, '0' * 64)
    copy_ = tmp_path / RUN.name
    copy_.write_bytes(RUN.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='not the reviewed'):
        tp.check_config_file(copy_, rc.sha256_file(copy_))  # self-consistent pin of an edited file
    with pytest.raises(ValueError, match='differs from the pinned'):
        tp.check_config_file(copy_, tp.CONFIG_SHA256)


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
    assert record['seed'] == tp.PROBE_SEED
    assert record['warmup']['epochs'] == 1 and record['measured']['epochs'] == 2
    per_epoch = -(-n_train // cfg['training']['batch_size'])
    assert record['warmup']['optimizer_updates'] == per_epoch
    assert record['measured']['optimizer_updates'] == 2 * per_epoch
    assert record['validation_standin_rows'] == n_early == len(splits['earlystop']['y'])
    assert record['predict_standin_rows'] == n_predict == sum(len(splits[s]['y']) for s in tp.PREDICT_SPLITS)
    m = record['measured']
    assert record['measured_seconds_per_epoch'] == max(m['wall_seconds'], m['fit_report_seconds']) / 2
    assert len(calls) == 6
    if arm in PLE:
        inputs = rc.arm_inputs(train, rc.arm_spec(cfg, arm), cfg['export']['workload_columns'])
        assert record['ple']['bins_sha256'] == plm.bins_sha256(tp.ple_bins(arm, inputs))
        assert 0 < record['ple']['bins_seconds'] <= record['prepare_seconds']
        assert record['ple']['past_valid_rows'] == rc.ple_bin_rows(inputs)['past_valid_rows']
        assert record['ple']['effective_parameter_count'] <= record['parameter_count']
    else:
        assert 'ple' not in record
    tp.no_forbidden_keys(record)
    text = json.dumps(record).lower()
    assert 'nll' not in text and 'history' not in text and 'best_epoch' not in text and 'logit' not in text


def test_ple_bins_of_the_probe_are_the_bins_of_the_frozen_fit(export):
    root, digest, _ = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    train = rc.load_split(root, manifest, 'train')
    columns = cfg['export']['workload_columns']
    for arm in PLE:
        inputs = rc.arm_inputs(train, rc.arm_spec(cfg, arm), columns)
        model = rc.build_model(rc.arm_spec(cfg, arm), tp.PROBE_SEED, 'cpu')
        labels = np.array(train['y'])
        model.fit(inputs, labels, *tp.take(inputs, labels, 50), epochs=1, patience=5, batch_size=1024,
                  learning_rate=5e-4)
        assert model.net.config['ple_bins_sha256'] == plm.bins_sha256(tp.ple_bins(arm, inputs))


def test_probe_arm_refuses_bins_that_differ_from_the_fit(export, monkeypatch):
    root, digest, _ = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    train = rc.load_split(root, manifest, 'train')
    real = tp.ple_bins
    monkeypatch.setattr(tp, 'ple_bins', lambda arm, inputs: real(arm, tuple(a[:50] for a in inputs)))
    with pytest.raises(RuntimeError, match='other bins'):
        tp.probe_arm(cfg, 'ple_residual_counts', train, cfg['export']['workload_columns'], 50, 50, 'cpu')


def test_validation_and_predict_standins_are_train_rows_of_manifest_counts(export, monkeypatch):
    root, digest, _ = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    n_train, n_early, n_predict = tp.row_counts(manifest)
    train = rc.load_split(root, manifest, 'train')
    seen = {}
    from pitchmdp.direct_ple_model import DirectPLEResidualModel
    real_fit, real_logits = DirectPLEResidualModel.fit, DirectPLEResidualModel.logits

    def fit(self, inputs, labels, early, early_y, **kw):
        seen.setdefault('early', []).append((early[0].shape[0], np.array_equal(early[0], inputs[0][:n_early])))
        return real_fit(self, inputs, labels, early, early_y, **kw)

    def logits(self, arrays, batch_size=4096):
        seen.setdefault('logits', []).append(arrays[0].shape[0])
        return real_logits(self, arrays, batch_size)

    monkeypatch.setattr(DirectPLEResidualModel, 'fit', fit)
    monkeypatch.setattr(DirectPLEResidualModel, 'logits', logits)
    tp.probe_arm(cfg, 'ple_residual_counts', train, cfg['export']['workload_columns'], n_early, n_predict, 'cpu')
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
    root, digest, _ = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    train = rc.load_split(root, manifest, 'train')
    monkeypatch.setattr(rc, 'build_model', lambda arm, seed, device: FakeModel(report_seconds, sleep))
    record = tp.probe_arm(cfg, 'control', train, cfg['export']['workload_columns'], 50, 50, 'cpu')
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


# ------------------------------------------------------------------ projection and gates

def fake_record(arm, per_epoch, prepare=2., predict=3., bins=1.):
    record = {'arm': arm, 'prepare_seconds': prepare, 'predict_seconds': predict, 'measured_seconds_per_epoch': per_epoch}
    if arm in PLE:
        record['ple'] = {'bins_seconds': bins}
    return record


def test_projection_and_gate_arithmetic():
    workers = [{'gpu': GPUS[0], 'startup_seconds': 10., 'arms': [fake_record(a, 5.) for a in ARMS]},
               {'gpu': GPUS[1], 'startup_seconds': 12., 'arms': [fake_record(a, 6. if a == PLE[1] else 4.)
                                                                  for a in ARMS]}]
    out = tp.summarize(workers, 5)
    # slowest GPU: 12 + 2 + 60*6 + 3 + 30 = 407 for ple_tabm; others 10 + 2 + 300 + 3 + 30 = 345
    assert out['ple_task_seconds_projected'] == {PLE[0]: pytest.approx(345.), PLE[1]: pytest.approx(407.)}
    assert out['slowest_task_seconds_projected']['control'] == pytest.approx(345.)
    assert out['family_worker_seconds_projected'] == pytest.approx(5 * (4 * 345. + 407.))
    assert out['gate'] == {'task_seconds_max': 1800, 'family_seconds_max': 36000, 'ple_arms_within': True,
                           'every_arm_within': True, 'family_within': True}
    assert out['marginal_arms'] == []
    row = out['per_arm'][PLE[0]][0]
    assert row['ple_binning_seconds_inside_projection_not_credited'] == pytest.approx(30.)
    assert row['task_seconds_projected'] == pytest.approx(345.)  # not credited: the projection keeps it
    slow = copy.deepcopy(workers)
    slow[0]['arms'][2]['measured_seconds_per_epoch'] = 29.5  # ple_residual: 10 + 2 + 1770 + 3 + 30 = 1815 > 1800
    out = tp.summarize(slow, 5)
    assert out['gate']['ple_arms_within'] is False and out['gate']['every_arm_within'] is False
    assert out['marginal_arms'] == [PLE[0]]
    ref = copy.deepcopy(workers)
    ref[1]['arms'][3]['measured_seconds_per_epoch'] = 29.5  # tabm_residual (reference) only: 12 + 2 + 1770 + 33
    out = tp.summarize(ref, 5)
    assert out['gate']['ple_arms_within'] is True and out['gate']['every_arm_within'] is False
    edge = copy.deepcopy(workers)
    edge[0]['arms'][2]['measured_seconds_per_epoch'] = 29.25  # 45 + 1755 = exactly 1800 -> within (<=)
    assert tp.summarize(edge, 5)['gate']['ple_arms_within'] is True
    family = copy.deepcopy(workers)
    for w in family:
        for r in w['arms']:
            r['measured_seconds_per_epoch'] = 23.  # 12 + 2 + 1380 + 33 = 1427 per arm -> 5 x 5 x 1427 = 35675
    assert tp.summarize(family, 5)['gate']['family_within'] is True
    for r in family[1]['arms']:
        r['measured_seconds_per_epoch'] = 23.3  # 1445 per arm -> 36125 > 36000
    out = tp.summarize(family, 5)
    assert out['gate']['family_within'] is False and out['gate']['every_arm_within'] is True
    with pytest.raises(ValueError):
        tp.summarize([workers[0], {**workers[1], 'arms': workers[1]['arms'][:4]}], 5)
    with pytest.raises(ValueError):
        tp.summarize([{**w, 'arms': w['arms'][::-1]} for w in workers], 5)


def test_forbidden_output_keys_refused():
    for bad in ({'arms': [{'measured': {'history': []}}]}, {'earlystop_conditional_nll': 1.}, {'dev_logits': []},
                {'probabilities': []}, {'Brier': 1.}, {'ece10': 1.}, {'score': 1.}):
        with pytest.raises(ValueError):
            tp.no_forbidden_keys(bad)
    assert tp.no_forbidden_keys({'measured': {'fit_report_seconds': 1.}, 'peak_device_bytes': 1, 'ple': {}})


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
    assert ident['fixed']['warmup_epochs'] == 1 and ident['fixed']['measure_epochs'] == 2
    assert ident['fixed']['probe_seed'] == 34 and ident['fixed']['launch_env'] == tp.LAUNCH_ENV
    assert set(ident['source_sha256']) >= {tp.SELF, tp.QUEUE, rc.SELF, 'pitchmdp/direct_ple_model.py'}
    assert 'scripts/timing_probe_tabm_residual.py' not in ident['source_sha256']
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


def test_preflight_refuses_the_real_draft_unless_its_exact_bytes(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    monkeypatch.setattr(tp, 'CONFIG_SHA256', rc.sha256_file(RUN))  # the shipped pin: the synthetic copy is refused
    with pytest.raises(ValueError, match='not the reviewed'):
        tp.preflight(args, cfg)


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
    import signal
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
            'import timing_probe_ple_residual as tp; from cuda_direct_queue import GpuLock;'
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


def test_worker_refuses_changed_config_or_env_before_loading_data(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = worker_args(tmp_path, root, digest, monkeypatch)
    opened, bound = [], []
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: bound.append(uuid) or {'bound_uuid': uuid})
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]))
    monkeypatch.setenv('MKL_NUM_THREADS', '8')
    with pytest.raises(ValueError, match='Launch environment'):
        tp.worker(args, cfg)
    assert bound == []  # refused before the GPU is bound
    monkeypatch.setenv('MKL_NUM_THREADS', '2')
    args.config.write_text(args.config.read_text() + '\n')
    with pytest.raises(RuntimeError, match='Config differs'):
        tp.worker(args, cfg)
    assert opened == []


# ------------------------------------------------------------------ frozen files

def test_frozen_runner_queue_models_and_configs_unchanged_in_the_probe_window():
    # Window 0978558 (this branch's base, the P35 smoke record) .. working tree.  Once the probe has run, re-scope the
    # end of the window to the archived probe commit (as T70c did for the P34 probe test).
    changed = subprocess.run(['git', 'diff', '--name-only', WINDOW_BASE, '--', *FROZEN], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split()
    assert changed == []
    assert rc.sha256_file(PROJECT / 'pitchmdp' / 'direct_ple_model.py') == \
        'd3575019f971b06ba26060a1fa31e3894a070fd3b4b88191393743cb16ea175e'
