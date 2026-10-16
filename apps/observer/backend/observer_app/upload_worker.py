"""Bounded upload coordinator. Only owned transport children touch the network; no publisher state in a child."""
import json
import multiprocessing
import os
import threading
import time


MAX_MAILBOX = 128
MAX_IPC = 4096


def _transport(connection, lifetime, config, job):
    # Spawn receives only these pipe handles, not the publisher's non-inheritable output flock descriptor.
    # If the owner disappears, stop its transport too. CAS still fences a request already accepted by the server.
    def owner_exit():
        try:
            lifetime.recv_bytes(1)
        except (EOFError, OSError):
            os._exit(0)
    threading.Thread(target=owner_exit, daemon=True).start()
    from .publisher import HttpSink, UPLOAD_MAX_BYTES, UploadError
    try:
        remote = HttpSink(config)
        if job['method'] == 'HEAD':
            result = remote.head(job['path'])
        elif job['method'] == 'DELETE':
            result = remote.delete(job['path'], expected=job['expected'])
        else:
            import hashlib
            with open(job['snapshot'], 'rb') as handle:
                data = handle.read(UPLOAD_MAX_BYTES+1)
            if len(data) > UPLOAD_MAX_BYTES or hashlib.sha256(data).hexdigest() != job['sha256']:
                raise UploadError('upload_local_mismatch', outcome='blocked')
            result = remote.write(job['path'], data, expected=job['expected'], final=job['final'])
        message = {'ok': True, 'result': result}
    except UploadError as error:
        message = {'ok': False, 'code': str(error), 'outcome': error.outcome, 'retry_after': error.retry_after}
    except BaseException:
        message = {'ok': False, 'code': 'upload_unknown', 'outcome': 'blocked', 'retry_after': 0}
    raw = json.dumps(message).encode()
    if len(raw) < MAX_IPC-4:  # one small pipe frame; parent also limits allocation before receiving it
        connection.send_bytes(raw)
    connection.close()


class UploadWorker:
    """One coordinator, one bounded metadata mailbox, at most one short-lived transport child."""

    def __init__(self, transfer, config):
        self.transfer, self.config = transfer, config
        self.deadline = config.get('operation_deadline_s', 15)
        self.grace = config.get('shutdown_grace_s', 5)
        if (type(self.deadline) not in (int, float) or not 1 <= self.deadline <= 60
                or type(self.grace) not in (int, float) or not 0 <= self.grace <= 30
                or len(json.dumps(config).encode()) > 8192):
            from .publisher import UploadError
            raise UploadError('upload_config_invalid')
        self.condition = threading.Condition()
        self.mailbox, self.active, self.closing = {}, False, False
        self.stop = threading.Event()
        self.process = None
        self.peak_entries = 0
        self.thread = threading.Thread(target=self._run, name='publisher-upload', daemon=False)
        self.thread.start()

    def submit(self, path):
        with self.condition:
            if self.closing or self.stop.is_set():
                return False
            if path not in self.mailbox and len(self.mailbox) >= MAX_MAILBOX:
                return False  # pending intent remains durable; a later cycle can enqueue it
            self.mailbox[path] = None
            self.peak_entries = max(self.peak_entries, len(self.mailbox))
            self.condition.notify_all()
            return True

    def _run(self):
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.stop.is_set() or self.mailbox)
                    if self.stop.is_set():
                        break
                    path = next(iter(self.mailbox))
                    del self.mailbox[path]
                    self.active = True
                try:
                    self.transfer(path)
                except Exception:
                    # Intent/attempt remains durable. Never log exception text from transport/config boundaries.
                    import logging
                    logging.getLogger(__name__).warning('upload_failed')
                    if self.process is not None:
                        self.stop.set()  # never replace the handle of an unreaped owned child
                finally:
                    with self.condition:
                        self.active = False
                        self.condition.notify_all()
        finally:
            self._reap()

    def _reap(self):
        process = self.process
        if process is None:
            return
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            if process.is_alive():
                from .publisher import UploadError
                raise UploadError('upload_worker_not_stopped')
        process.close()
        self.process = None

    def request(self, method, path, **options):
        from .publisher import UploadError
        if self.stop.is_set():
            raise UploadError('upload_unknown', outcome='blocked')
        context = multiprocessing.get_context('spawn')
        receive, send = context.Pipe(duplex=False)
        lifetime, owner = context.Pipe(duplex=False)
        job = {'method': method, 'path': path} | options
        if len(json.dumps(job).encode()) > 8192:
            for pipe in (receive, send, lifetime, owner): pipe.close()
            raise UploadError('upload_request_invalid')
        process = self.process = context.Process(target=_transport, args=(send, lifetime, self.config, job), daemon=False)
        started = time.monotonic()
        try:
            process.start()  # small descriptor only, never a multi-megabyte body in the spawn/IPC buffer
            send.close()
            lifetime.close()
            while not self.stop.is_set() and time.monotonic()-started < self.deadline:
                if receive.poll(min(.05, max(0, self.deadline-(time.monotonic()-started)))):
                    # recv_bytes checks advertised frame length before allocating the body.
                    message = json.loads(receive.recv_bytes(MAX_IPC))
                    if message['ok']:
                        return message['result']
                    raise UploadError(message['code'], outcome=message['outcome'], retry_after=message['retry_after'])
                if not process.is_alive():
                    break
            raise UploadError('upload_unknown', outcome='blocked')
        except UploadError:
            raise
        except (OSError, EOFError, ValueError, RuntimeError, KeyError):
            raise UploadError('upload_unknown', outcome='blocked') from None
        finally:
            try:
                self._reap()
            finally:
                for pipe in (receive, send, lifetime, owner): pipe.close()

    def close(self):
        with self.condition:
            self.closing = True
            until = time.monotonic()+self.grace
            while (self.mailbox or self.active) and self.thread.is_alive() and time.monotonic() < until:
                self.condition.wait(max(0, until-time.monotonic()))
            self.stop.set()
            self.condition.notify_all()
        self.thread.join(3)
        if self.thread.is_alive() or self.process is not None:
            from .publisher import UploadError
            raise UploadError('upload_worker_not_stopped')  # caller MUST retain its output lock
