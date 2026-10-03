"""Synthetic tests for the DRAFT CUDA direct-head export, runner and queue (no real data, CPU only)."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import cuda_direct_queue as cq  # noqa: E402
import export_cuda_direct as ex  # noqa: E402
import run_cuda_direct as rc  # noqa: E402

CONFIG = REPO / 'configs' / 'DRAFT-CUDA-DIRECT-20261004.json'
SHAPES = {'length': 6, 'n_token': 38, 'n_context': 52, 'n_workload': 4}
COLUMNS = ['prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing']
COUNTS = {'train': 1252824, 'earlystop': 16000, 'temperature': 2603, 'blend': 4821, 'dev': 12334, 'mlb_dev': 311721}
GPU3, GPU4 = 'GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9', 'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99'


def config():
    return rc.config_check(json.loads(CONFIG.read_text()))


def split_arrays(n, seed, game0, date=20240601):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 6, 38)).astype(np.float32)
    tokens[:, -1, :8] = 0.        # current physics zero (direct head)
    tokens[:, -1, -11:] = 0.      # current outcome zero (enriched contract)
    valid = np.ones((n, 6), dtype=bool)
    valid[:, 0] = rng.random(n) > .5
    game = game0 + np.arange(n) // 20
    keys = np.column_stack([game, np.arange(n) % 20 // 4 + 1, np.arange(n) % 4 + 1]).astype(np.int64)
    return {'tokens': tokens, 'valid': valid, 'context': rng.normal(size=(n, 52)).astype(np.float32),
            'workload': rng.random((n, 4)).astype(np.float32), 'keys': keys,
            'y': rng.integers(0, 10, n).astype(np.int64), 'game_pk': game.astype(np.int64),
            'pitcher': rng.integers(1, 5, n).astype(np.int64), 'date': np.full(n, date, dtype=np.int32)}


@pytest.fixture
def export(tmp_path):
    sizes = {'train': 600, 'earlystop': 200, 'temperature': 120, 'blend': 80, 'dev': 90, 'mlb_dev': 70}
    splits = {s: split_arrays(n, i, 1000 * (i + 1)) for i, (s, n) in enumerate(sizes.items())}
    digest = ex.write_export(tmp_path / 'export', splits, {'synthetic': True}, SHAPES, COLUMNS)
    return tmp_path / 'export', digest, splits


# ------------------------------------------------------------------ export

def test_estimate_matches_metadata_and_shapes():
    est = ex.estimate_bytes(COUNTS, SHAPES)
    assert est['row_bytes'] == 912 + 6 + 208 + 16 + 24 + 24 + 4 == 1194
    assert est['total'] == sum(COUNTS.values()) * 1194 == 1910761782


def test_export_writes_manifest_and_refuses_existing(export):
    path, digest, splits = export
    manifest = rc.verify_export(path, config(), digest)
    assert manifest['splits']['train']['n'] == 600 and manifest['estimate_bytes']['row_bytes'] == 1194
    assert not list(path.parent.glob('*.tmp-*'))
    with pytest.raises(FileExistsError):
        ex.write_export(path, splits, {}, SHAPES, COLUMNS)


def test_export_rejects_2026_and_physics(tmp_path):
    bad = {'train': split_arrays(10, 0, 1, date=20260401)}
    with pytest.raises(ValueError, match='2026'):
        ex.write_export(tmp_path / 'a', bad, {}, SHAPES, COLUMNS)
    leak = {'train': split_arrays(10, 0, 1)}
    leak['train']['tokens'][:, -1, 0] = 1.
    with pytest.raises(ValueError, match='physics'):
        ex.write_export(tmp_path / 'b', leak, {}, SHAPES, COLUMNS)
    assert not (tmp_path / 'a').exists() and not (tmp_path / 'b').exists()


def test_export_size_guard(tmp_path):
    with pytest.raises(ValueError, match='differs'):
        ex.write_export(tmp_path / 'c', {'train': split_arrays(10, 0, 1)}, {}, SHAPES, COLUMNS, expect_bytes=1)


def test_tampered_export_fails_closed(export):
    path, digest, _ = export
    target = path / 'dev' / 'y.npy'
    os.chmod(target, 0o644)
    np.save(target, np.zeros(90, dtype=np.int64))
    with pytest.raises(ValueError, match='hash'):
        rc.verify_export(path, config(), digest)
    with pytest.raises(ValueError, match='pinned'):
        rc.verify_export(path, config(), '0' * 64)


# ------------------------------------------------------------------ semantics equal to the frozen Mac path

def test_workload_placement_equals_workload_context():
    from pitchmdp.data import KEY
    from pitchmdp.matrix_sharing import training_arrays
    from run_ml_direct_workload import WorkloadContext
    arrays = split_arrays(30, 1, 7)
    full = np.concatenate([arrays['context'], np.ones((30, 7), np.float32)], axis=1)  # + 7 sharing columns
    frame = pd.DataFrame(arrays['keys'], columns=KEY)

    class Base:
        def transform(self, f):
            return full[f.index.to_numpy()]
    for arm in config()['arms']:
        keep = [COLUMNS.index(c) for c in arm['workload']]
        wrapped = WorkloadContext(Base(), frame, arrays['workload'][:, keep])
        mac = training_arrays((arrays['tokens'], arrays['valid'], wrapped.transform(frame.iloc[np.arange(30)])))[2]
        assert np.array_equal(rc.arm_inputs(arrays, arm, COLUMNS)[2], mac)


def test_calibration_and_probabilities_equal_frozen_chain():
    from pitchmdp.matrix_benchmark import integrated_probabilities
    from pitchmdp.matrix_sharing import SharingPredictor
    from pitchmdp.sequence_delivery import JointDelivery
    rng = np.random.default_rng(3)
    logits, y = rng.normal(size=(200, 10)) * 2, rng.integers(0, 10, 200)

    class Global:
        def logits(self, arrays):
            return logits
    ctx = np.zeros((200, 9))
    lp_mac = SharingPredictor('G0-global', Global(), {}).logits((np.zeros((200, 6, 38)), np.ones((200, 6), bool), ctx))
    lp = rc.g0_log_probs(logits)
    assert np.array_equal(lp, lp_mac)

    class Fixed(JointDelivery):
        def logits(self, *args):
            return lp[:, None, :], np.full(200, -2)

    class Model:
        report = {}
    model = Model()
    Fixed().calibrate(model, None, None, np.arange(200), y)
    temperature, loss = rc.fit_temperature(lp, y)
    assert temperature == model.delivery_temperature and loss == model.report['calibration_integrated_log_loss']
    for a, b in zip(rc.probabilities(lp, temperature), integrated_probabilities(lp[:, None, :], temperature)):
        assert np.allclose(a, b, atol=1e-12, rtol=0)


# ------------------------------------------------------------------ runner (CPU smoke of train/predict/save/load)

@pytest.mark.parametrize('arm', ['control', 'counts', 'block', 'residual', 'periodic'])
def test_task_train_predict_save_load_cpu(export, tmp_path, arm):
    path, digest, splits = export
    cfg = config()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'out-smoke', arm, 0, 'cpu', {'test': True}, smoke=True)
    state = rc.verify_task(dest)
    assert state['identity']['arm']['name'] == arm and state['identity']['export_manifest_sha256'] == digest
    fit = json.loads((dest / 'fit.json').read_text())
    assert fit['context_width'] == 52 + len(rc.arm_spec(cfg, arm)['workload'])
    assert fit['train_rows_sha256'] == rc.read_json(path / 'manifest.json')['splits']['train']['rows_sha256']
    with np.load(dest / 'predictions.npz') as data:
        for split in ('blend', 'dev'):
            assert np.array_equal(data[split + '_keys'], splits[split]['keys'])
            assert np.array_equal(data[split + '_y'], splits[split]['y'])
            assert np.allclose(data[split].sum(1), 1.) and (data[split + '_delivery_level'] == -2).all()
        dev = data['dev']
    with np.load(dest / 'mlb_predictions.npz') as data:
        assert len(data['mlb_dev']) == 70
    temperature = json.loads((dest / 'calibration.json').read_text())['delivery_temperature']
    model = rc.load_model(rc.arm_spec(cfg, arm), dest / 'model.pt', 'cpu')
    manifest = rc.read_json(path / 'manifest.json')
    part = rc.load_split(path, manifest, 'dev')
    again, _ = rc.probabilities(rc.g0_log_probs(model.logits(rc.arm_inputs(part, rc.arm_spec(cfg, arm), COLUMNS))), temperature)
    assert np.allclose(again, dev, atol=1e-6)
    with pytest.raises(FileExistsError):
        rc.run_task(cfg, path, digest, tmp_path / 'out-smoke', arm, 0, 'cpu', {}, smoke=True)


def test_failed_task_is_preserved(export, tmp_path):
    path, _, _ = export
    with pytest.raises(ValueError, match='pinned'):
        rc.run_task(config(), path, 'f' * 64, tmp_path / 'o-smoke', 'control', 1, 'cpu', {}, smoke=True)
    failed = list((tmp_path / 'o-smoke' / 'failed').iterdir())
    assert len(failed) == 1 and (failed[0] / 'error.json').is_file() and (failed[0] / 'identity.json').is_file()
    assert not (tmp_path / 'o-smoke' / 'control-smoke' / '1').exists()


def test_gpu_binding_refuses(monkeypatch):
    allowed = config()['device']['allowed_gpu_uuids']
    with pytest.raises(ValueError, match='allowed'):
        rc.bind_gpu('GPU-00000000-0000-0000-0000-000000000000', allowed)
    import torch  # noqa: F401  (already imported in this process)
    with pytest.raises(RuntimeError, match='before the GPU was bound'):
        rc.bind_gpu(GPU3, allowed)


def test_config_rejects_changed_budget():
    cfg = json.loads(CONFIG.read_text())
    cfg['training']['epochs'] = 40
    with pytest.raises(ValueError, match='budget'):
        rc.config_check(cfg)
    cfg = json.loads(CONFIG.read_text())
    cfg['device']['allowed_gpu_uuids'].append('GPU-x')
    with pytest.raises(ValueError, match='CUDA'):
        rc.config_check(cfg)


# ------------------------------------------------------------------ queue

class Clock:
    def __init__(self, hhmmss, step):
        h, m, s = hhmmss
        self.now, self.step = datetime(2026, 10, 4, h, m, s, tzinfo=cq.KST), step

    def __call__(self):
        return self.now

    def sleep(self, _):
        import time
        time.sleep(.05)
        from datetime import timedelta
        self.now += timedelta(seconds=self.step)


def make_queue(tmp_path, command, complete, clock, gpus=(GPU3,), priorities=(1,), **kw):
    cfg = config()
    return Queue(cfg, tmp_path / 'q', list(gpus), command, complete, clock=clock, sleep=clock.sleep, poll=0,
                 grace=2, priorities=priorities, **kw)


Queue = cq.Queue


def test_plan_is_predeclared_priority_first():
    tasks = cq.plan(config())
    assert [t['id'] for t in tasks[:3]] == ['control-0', 'counts-0', 'block-0']
    assert len(tasks) == 15 and all(t['priority'] == 1 for t in tasks[:9]) and all(t['priority'] == 2 for t in tasks[9:])


def test_queue_rejects_disallowed_or_three_gpus(tmp_path):
    clock = Clock((10, 0, 0), 1)
    with pytest.raises(ValueError):
        make_queue(tmp_path, None, None, clock, gpus=('GPU-other',))
    with pytest.raises(ValueError):
        make_queue(tmp_path, None, None, clock, gpus=(GPU3, GPU4, GPU3))


def test_failing_task_runs_exactly_twice_then_final(tmp_path):
    calls = []

    def command(task, gpu):
        calls.append(task['id'])
        return [sys.executable, '-c', 'import sys; sys.exit(3)']
    queue = make_queue(tmp_path, command, lambda t: False, Clock((10, 0, 0), 1))
    queue.tasks = queue.tasks[:1]
    summary = queue.run()
    assert calls == ['control-0', 'control-0']
    record = queue.status(queue.tasks[0])
    assert record['state'] == 'failed_final' and record['attempts'] == 2
    assert summary['comparison_allowed'] is False
    # restart: attempts persist, no third launch
    queue2 = make_queue(tmp_path, command, lambda t: False, Clock((10, 0, 0), 1))
    queue2.tasks = queue2.tasks[:1]
    queue2.run()
    assert calls == ['control-0', 'control-0']


def test_success_marks_done_and_allows_comparison(tmp_path):
    done = set()

    def command(task, gpu):
        done.add(task['id'])
        assert gpu == GPU3
        return [sys.executable, '-c', 'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == "%s"' % GPU3]
    queue = make_queue(tmp_path, command, lambda t: t['id'] in done, Clock((10, 0, 0), 1))
    summary = queue.run()
    assert summary['priority1_complete'] and summary['comparison_allowed'] and summary['counts'] == {'done': 9}
    assert len(list((tmp_path / 'q' / 'logs').iterdir())) == 9


def test_no_new_fit_after_1500(tmp_path):
    calls = []
    queue = make_queue(tmp_path, lambda t, g: calls.append(t) or [sys.executable, '-c', 'pass'],
                       lambda t: False, Clock((15, 0, 0), 1))
    summary = queue.run()
    assert calls == [] and summary['counts'] == {'not_run': 9}
    assert queue.status(queue.tasks[0])['stages'][-1]['stage'] == 'deadline_no_new_fit'


def test_kill_own_group_at_1530(tmp_path):
    marker = tmp_path / 'child.pid'
    code = f'import os, time; open({str(marker)!r}, "w").write(str(os.getpid())); time.sleep(60)'
    queue = make_queue(tmp_path, lambda t, g: [sys.executable, '-c', code], lambda t: False, Clock((14, 59, 50), 60))
    queue.tasks = queue.tasks[:1]
    summary = queue.run()
    assert queue.status(queue.tasks[0])['state'] == 'killed_deadline'
    assert summary['counts'] == {'killed_deadline': 1}
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_task_timeout_counts_as_failure(tmp_path):
    cfg_queue = make_queue(tmp_path, lambda t, g: [sys.executable, '-c', 'import time; time.sleep(60)'],
                           lambda t: False, Clock((10, 0, 0), 1))
    cfg_queue.timeout = .3
    cfg_queue.tasks = cfg_queue.tasks[:1]
    cfg_queue.run()
    record = cfg_queue.status(cfg_queue.tasks[0])
    assert record['state'] == 'failed_final' and record['attempts'] == 2
    assert [s['stage'] for s in record['stages']].count('timeout_killed') == 2


def test_gpu_lock_is_exclusive(tmp_path):
    lock = cq.GpuLock(tmp_path / 'locks', GPU3)
    assert lock.acquire()
    other = cq.GpuLock(tmp_path / 'locks', GPU3)
    assert not other.acquire()
    lock.release()
    assert other.acquire()
    other.release()


def test_deadlines_must_be_ordered():
    with pytest.raises(ValueError):
        cq.Deadlines({'date': '2026-10-04', 'no_new_fit_after': '15:30', 'kill_own_after': '15:00', 'aggregate_by': '16:00'})
