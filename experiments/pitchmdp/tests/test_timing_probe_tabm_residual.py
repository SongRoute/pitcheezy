"""Synthetic CPU tests: TabM-residual timing-only TRAIN probe (Amendment 2)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuda_direct as rc  # noqa: E402
import timing_probe_tabm_residual as tp  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402
from cuda_direct_queue import GpuLock  # noqa: E402

RUN = REPO / 'configs' / 'DRAFT-CUDA-TABM-RESIDUAL-20261004.json'
GPUS = ['GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9', 'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99']
ARMS = ['control', 'periodic_residual_counts', 'tabm_counts', 'tabm_residual_counts', 'tabm_counts_d10']


def load():
    return tf.load(RUN, False, True)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


def test_config_check_accepts_only_the_draft_family():
    assert tp.check_config(load()) == ARMS
    bad = load()
    bad['arms'] = bad['arms'][:4]
    with pytest.raises(ValueError):
        tp.check_config(bad)
    bad = load()
    bad['training']['epochs'] = 30
    with pytest.raises(ValueError):
        tp.check_config(bad)
    bad = load()
    bad['seeds'] = bad['seeds'] + [tp.PROBE_SEED]
    with pytest.raises(ValueError):
        tp.check_config(bad)
    other = tf.load(REPO / 'configs' / 'EXP-P31-001.json', False, True)
    with pytest.raises(ValueError):
        tp.check_config(other)


@pytest.mark.parametrize('arm', ARMS)
def test_probe_arm_reads_train_only_and_records_timing_only(export, monkeypatch, arm):
    root, digest, splits = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    opened = []
    real = rc.load_split
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]) or real(*a, **k))
    n_train, n_early, n_predict = tp.row_counts(manifest)
    train = rc.load_split(root, manifest, 'train')
    calls = []
    record = tp.probe_arm(cfg, arm, train, cfg['export']['workload_columns'], n_early, n_predict, 'cpu',
                          sync=lambda: calls.append('sync'))
    assert opened == ['train']
    assert record['warmup']['epochs'] == 1 and record['measured']['epochs'] == 2
    per_epoch = -(-n_train // cfg['training']['batch_size'])
    assert record['warmup']['optimizer_updates'] == per_epoch
    assert record['measured']['optimizer_updates'] == 2 * per_epoch
    assert record['validation_standin_rows'] == n_early == len(splits['earlystop']['y'])
    assert record['predict_standin_rows'] == n_predict == sum(len(splits[s]['y']) for s in tp.PREDICT_SPLITS)
    assert record['measured_seconds_per_epoch'] * 2 >= record['measured']['wall_seconds']
    assert record['measured_seconds_per_epoch'] * 2 >= record['measured']['fit_report_seconds']
    assert len(calls) == 6
    tp.no_forbidden_keys(record)
    text = json.dumps(record)
    assert 'nll' not in text and 'history' not in text and 'best_epoch' not in text


def test_validation_and_predict_standins_are_train_rows_of_manifest_counts(export, monkeypatch):
    root, digest, _ = export
    cfg = load()
    manifest = rc.verify_export(root, cfg, digest)
    n_train, n_early, n_predict = tp.row_counts(manifest)
    train = rc.load_split(root, manifest, 'train')
    seen = {}
    from pitchmdp.matrix_models import MatrixModel
    real_fit, real_logits = MatrixModel.fit, MatrixModel.logits

    def fit(self, inputs, labels, early, early_y, **kw):
        seen.setdefault('early', []).append((early[0].shape[0], np.array_equal(early[0], inputs[0][:n_early])))
        return real_fit(self, inputs, labels, early, early_y, **kw)

    def logits(self, arrays, batch_size=4096):
        seen.setdefault('logits', []).append(arrays[0].shape[0])
        return real_logits(self, arrays, batch_size)

    monkeypatch.setattr(MatrixModel, 'fit', fit)
    monkeypatch.setattr(MatrixModel, 'logits', logits)
    tp.probe_arm(cfg, 'control', train, cfg['export']['workload_columns'], n_early, n_predict, 'cpu')
    assert seen['early'] == [(n_early, True), (n_early, True)]
    assert seen['logits'][-1] == n_predict  # earlier entries: per-epoch validation inside fit (n_early each)
    assert seen['logits'][:-1] == [n_early] * 3


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


def fake_record(arm, per_epoch, prepare=2., predict=3.):
    return {'arm': arm, 'prepare_seconds': prepare, 'predict_seconds': predict, 'measured_seconds_per_epoch': per_epoch}


def test_projection_and_gate_arithmetic():
    workers = [{'gpu': GPUS[0], 'startup_seconds': 10., 'arms': [fake_record(a, 5.) for a in ARMS]},
               {'gpu': GPUS[1], 'startup_seconds': 12., 'arms': [fake_record(a, 6. if a == 'tabm_residual_counts' else 4.)
                                                                  for a in ARMS]}]
    out = tp.summarize(workers, 5)
    # slowest GPU: 12 + 2 + 60*6 + 3 + 30 = 407 for the new arm; others 10 + 2 + 300 + 3 + 30 = 345
    assert out['tabm_residual_task_seconds_projected'] == pytest.approx(407.)
    assert out['slowest_task_seconds_projected']['control'] == pytest.approx(345.)
    assert out['family_worker_seconds_projected'] == pytest.approx(5 * (4 * 345. + 407.))
    assert out['gate'] == {'task_seconds_max': 1800, 'family_seconds_max': 36000, 'tabm_residual_within': True,
                           'every_arm_within': True, 'family_within': True}
    slow = copy.deepcopy(workers)
    slow[0]['arms'][3]['measured_seconds_per_epoch'] = 29.5  # 10 + 2 + 1770 + 3 + 30 = 1815 > 1800
    assert tp.summarize(slow, 5)['gate']['tabm_residual_within'] is False
    with pytest.raises(ValueError):
        tp.summarize([workers[0], {**workers[1], 'arms': workers[1]['arms'][:4]}], 5)


def test_forbidden_output_keys_refused():
    with pytest.raises(ValueError):
        tp.no_forbidden_keys({'arms': [{'measured': {'history': []}}]})
    with pytest.raises(ValueError):
        tp.no_forbidden_keys({'earlystop_conditional_nll': 1.})
    assert tp.no_forbidden_keys({'measured': {'fit_report_seconds': 1.}})


def args_for(tmp_path, root, digest, output='probe-timing', gpus=GPUS):
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(load()))
    ns = type('A', (), {})()
    ns.config, ns.config_sha256, ns.export, ns.export_manifest_sha256 = cfg_path, rc.sha256_file(cfg_path), root, digest
    ns.output, ns.lock_dir, ns.gpu, ns.python = tmp_path / output, tmp_path / 'locks', list(gpus), sys.executable
    return ns, json.loads(cfg_path.read_text())


def test_preflight_refusals(export, tmp_path):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest)
    ident = tp.preflight(args, cfg)
    assert ident['row_counts'] == {'train': 400, 'earlystop': 120, 'predict': 230}
    assert ident['fixed']['warmup_epochs'] == 1 and ident['fixed']['measure_epochs'] == 2
    assert set(ident['source_sha256']) >= {tp.SELF, tp.QUEUE, rc.SELF}
    for change in ({'output': tmp_path / 'probe'}, {'gpu': [GPUS[0], GPUS[0]]}, {'gpu': ['GPU-other']},
                   {'config_sha256': '0' * 64}, {'export_manifest_sha256': '0' * 64}):
        bad, _ = args_for(tmp_path, root, digest)
        for k, v in change.items():
            setattr(bad, k, v)
        with pytest.raises(ValueError):
            tp.preflight(bad, cfg)


def test_run_refuses_busy_gpu_lock_and_existing_output(export, tmp_path):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest)
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


def test_frozen_runner_and_models_unchanged_by_probe():
    import subprocess
    changed = subprocess.run(['git', 'diff', '--name-only', 'ee954a5', '--', 'experiments/pitchmdp/pitchmdp',
                              'experiments/pitchmdp/scripts/run_cuda_direct.py',
                              'experiments/pitchmdp/scripts/cuda_direct_queue.py', 'configs'],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    assert changed == []


# ------------------------------------------------------------------ review additions (same_model_review, 2026-10-04)

def worker_args(tmp_path, root, digest):
    args, cfg = args_for(tmp_path, root, digest)
    args.output.mkdir()
    args.gpu_uuid = GPUS[0]
    rc.write_json_atomic(args.output / 'preflight.json', tp.preflight(args, cfg))
    return args, cfg


def test_worker_opens_train_only_and_passes_manifest_standins(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = worker_args(tmp_path, root, digest)
    opened, probed = [], []
    real = rc.load_split
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: {'bound_uuid': uuid})  # no CUDA on the Mac
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
    args, cfg = worker_args(tmp_path, root, digest)
    preflight = json.loads((args.output / 'preflight.json').read_text())
    preflight['source_sha256'][tp.SELF] = '0' * 64
    rc.write_json_atomic(args.output / 'preflight.json', preflight)
    opened = []
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: {'bound_uuid': uuid})
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]))
    with pytest.raises(RuntimeError, match='Sources differ'):
        tp.worker(args, cfg)
    assert opened == [] and not (args.output / f'worker-{GPUS[0]}.json').exists()


def test_parent_preflight_and_lock_never_import_torch(export, tmp_path):
    import subprocess
    root, digest, _ = export
    args, _ = args_for(tmp_path, root, digest)
    code = ('import json, sys, types; from pathlib import Path; sys.path.insert(0, sys.argv[1]);'
            'import timing_probe_tabm_residual as tp; from cuda_direct_queue import GpuLock;'
            'a = json.loads(sys.argv[2]); ns = types.SimpleNamespace(**{k: Path(v) if k in ("config", "export", "output")'
            ' else v for k, v in a.items()}); tp.preflight(ns, json.loads(ns.config.read_text()));'
            'print("torch" in sys.modules)')
    payload = {'config': str(args.config), 'config_sha256': args.config_sha256, 'export': str(root),
               'export_manifest_sha256': digest, 'output': str(args.output), 'gpu': GPUS}
    out = subprocess.run([sys.executable, '-c', code, str(PROJECT / 'scripts'), json.dumps(payload)],
                         capture_output=True, text=True, timeout=120, check=True).stdout.split()
    assert out == ['False']


def fake_worker_python(tmp_path):
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/sh\nexec sleep 60\n')  # a worker that never finishes on its own
    fake.chmod(0o755)
    return str(fake)


def group_gone(pid):
    import os
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_run_interrupt_stops_worker_groups_before_releasing_locks(export, tmp_path, monkeypatch):
    import os
    import signal
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest)
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
    args, cfg = args_for(tmp_path, root, digest)
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


def test_worker_refuses_changed_config_before_loading_data(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = worker_args(tmp_path, root, digest)
    args.config.write_text(args.config.read_text() + '\n')
    opened = []
    monkeypatch.setattr(rc, 'bind_gpu', lambda uuid, allowed: {'bound_uuid': uuid})
    monkeypatch.setattr(rc, 'load_split', lambda *a, **k: opened.append(a[2]))
    with pytest.raises(RuntimeError, match='Config differs'):
        tp.worker(args, cfg)
    assert opened == []
