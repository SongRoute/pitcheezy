"""Queue-only CPU children; no torch, experiment configs, data, or GPU probes."""
from datetime import datetime
import os
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cuda_direct_queue as cq

GPUS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(6)]
NOW = datetime(2026, 10, 6, 12, tzinfo=cq.KST)


def queue_at(path, **kwargs):
    config = {
        'experiment_id': 'synthetic-queue-only',
        'device': {'allowed_gpu_uuids': GPUS, 'max_parallel': kwargs.pop('max_parallel', 5)},
        'arms': [{'name': 'fake', 'priority': 1}], 'seeds': list(range(5)),
        'queue': {'date': '2026-10-06', 'no_new_fit_after': '19:00',
                  'kill_own_after': '19:30', 'aggregate_by': '20:00',
                  'task_timeout_seconds': 10, 'family_budget_seconds': 100,
                  'max_attempts': 2},
    }
    return cq.Queue(config, path, kwargs.pop('gpus', GPUS[:5]),
                    kwargs.pop('command_for', lambda t, g: [sys.executable, '-c', 'pass']),
                    kwargs.pop('is_complete', lambda t: False),
                    lock_dir=path.parent / 'shared-locks', clock=lambda: NOW,
                    poll=.01, grace=.2, **kwargs)


@pytest.mark.parametrize('gpus', [[], GPUS, [GPUS[0]] * 2, ['GPU-not-allowed']])
def test_rejects_invalid_device_lists(tmp_path, gpus):
    with pytest.raises(ValueError):
        queue_at(tmp_path / 'q', gpus=gpus, max_parallel=6)


def test_registered_parallel_limit_still_applies(tmp_path):
    queue = queue_at(tmp_path / 'q')
    queue.config['device']['max_parallel'] = 1
    with pytest.raises(ValueError):
        cq.Queue(queue.config, tmp_path / 'other', GPUS[:2], None, None)


def test_five_concurrent_children_locks_binding_restart_and_budget(tmp_path):
    markers = tmp_path / 'markers'
    markers.mkdir()
    launches, waves = [], []

    def command(task, gpu):
        launches.append(task['id'])
        code = ('import os, pathlib, time; '
                f'p=pathlib.Path({str(markers)!r})/{task["id"]!r}; '
                'p.write_text(os.environ["CUDA_VISIBLE_DEVICES"]); '
                f'release=pathlib.Path({str(tmp_path / "release")!r}); '
                '\nwhile not release.exists(): time.sleep(.01)')
        return [sys.executable, '-c', code]

    def sleep(_):
        if not waves:
            assert len(queue.running) == 5
            assert len({t['id'] for t, *_ in queue.running.values()}) == 5
            for gpu, (_, process, _, _) in queue.running.items():
                assert process.poll() is None
                assert not cq.GpuLock(tmp_path / 'shared-locks', gpu).acquire()
            waves.append(5)
            (tmp_path / 'release').touch()
        time.sleep(.01)

    queue = queue_at(tmp_path / 'q', command_for=command,
                     is_complete=lambda t: (markers / t['id']).exists(), sleep=sleep)
    assert queue.run()['counts'] == {'done': 5}
    assert waves == [5] and len(set(launches)) == len(launches) == 5
    assert {p.read_text() for p in markers.iterdir()} == set(GPUS[:5])
    spent = queue.spent()
    again = queue_at(tmp_path / 'q', command_for=lambda t, g: pytest.fail('duplicate launch'))
    assert again.run()['family_seconds'] == spent > 0
    again.tasks = [{'id': 'fresh-99', 'arm': 'fresh', 'seed': 99, 'priority': 1}]
    again.budget = spent
    assert again.run()['counts'] == {'not_run': 1}
    assert again.status(again.tasks[0])['stages'][-1]['stage'] == 'family_budget'


@pytest.mark.parametrize('reason', ['exception', 'deadline'])
def test_all_five_children_stop_before_shared_locks_release(tmp_path, reason):
    children = []
    queue = queue_at(tmp_path / 'q', command_for=lambda t, g:
                     [sys.executable, '-c', 'import time; time.sleep(60)'])
    for gpu, lock in queue.locks.items():
        release = lock.release

        def checked_release(gpu=gpu, release=release):
            child = next(p for g, p in children if g == gpu)
            assert child.poll() is not None
            with pytest.raises(ProcessLookupError):
                os.killpg(child.pid, 0)
            release()
        lock.release = checked_release

    def sleep(_):
        children.extend((g, p) for g, (_, p, _, _) in queue.running.items())
        assert len(children) == 5
        if reason == 'exception':
            raise RuntimeError('synthetic interruption')
        queue.clock = lambda: NOW.replace(hour=19, minute=30)
    queue.sleep = sleep
    if reason == 'exception':
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            queue.run()
    else:
        assert queue.run()['counts'] == {'killed_deadline': 5}
    assert not queue.running and queue.spent() > 0
    for gpu in GPUS[:5]:
        lock = cq.GpuLock(tmp_path / 'shared-locks', gpu)
        assert lock.acquire()
        lock.release()
    lock = cq.GpuLock(tmp_path / 'q', 'queue')
    assert lock.acquire()
    lock.release()
