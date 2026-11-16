"""A transport child that answers and exits between the parent's poll and its liveness check must not be read as
'upload_unknown' (2026-11-15: at publisher start-up, with the main thread busy, the game file's PUT got 200 but the
parent dropped the child's answer, so the path stayed stuck)."""
import json
import multiprocessing

from observer_app import upload_worker


class LatePipe:
    """The answer is already in the pipe, but the first (timed) poll misses it, as when the worker thread waits for
    the GIL after a poll timeout while the child writes and exits."""
    def __init__(self, real):
        self.real, self.polls = real, 0

    def poll(self, timeout=0):
        self.polls += 1
        return False if self.polls == 1 else self.real.poll(timeout)

    def __getattr__(self, name):
        return getattr(self.real, name)


class FinishedChild:
    pid, exitcode = 4242, 0

    def __init__(self, target, args, daemon):
        self.send = args[0]

    def start(self):
        self.send.send_bytes(json.dumps({'ok': True, 'result': {'etag': '"1"'}}).encode())

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass

    def terminate(self):
        pass

    def kill(self):
        pass

    def close(self):
        pass


class Context:
    def __init__(self):
        self.pipes = 0

    def Pipe(self, duplex=True):
        receive, send = multiprocessing.Pipe(duplex=False)
        self.pipes += 1
        return (LatePipe(receive), send) if self.pipes == 1 else (receive, send)

    def Process(self, target, args, daemon):
        return FinishedChild(target, args, daemon)


def test_answer_from_an_already_exited_child_is_read(monkeypatch):
    monkeypatch.setattr(upload_worker.multiprocessing, 'get_context', lambda method: Context())
    worker = upload_worker.UploadWorker(lambda path: None, {'operation_deadline_s': 5, 'shutdown_grace_s': 0})
    try:
        assert worker.request('HEAD', 'v1/game/1.json') == {'etag': '"1"'}
        assert worker.process is None
    finally:
        worker.close()
