"""2026-11-16: connection drops on a real loopback socket, through the background worker's spawn child.
An unknown PUT outcome is reconciled by HEAD (same SHA -> done; else resend under the HEAD's ETag) instead of blocking
the path for the rest of the run; past UNKNOWN_RETRIES consecutive unknowns it blocks as before. Synthetic data only."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest
from observer_app import publisher
from test_publisher import make, world  # noqa: F401  (pytest fixtures)
from test_publisher_upload import Endpoint, HEALTH
from test_publisher_background import attach_background, until, synthetic_boundary  # noqa: F401

PATH = 'v1/game/990001.json'


@pytest.fixture
def server():
    """CAS model on 127.0.0.1. drops[(method, path)] = deque of 'before' (close without storing or answering) /
    'after' (store, then close without answering: arrived but the client cannot know)."""
    endpoint = Endpoint()
    endpoint.drops, endpoint.arrivals = {}, []
    guard = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self): self.handle_request()
        def do_PUT(self): self.handle_request()
        def do_DELETE(self): self.handle_request()
        def handle_request(self):
            path = self.path.removeprefix('/channel/')
            data = self.rfile.read(int(self.headers.get('Content-Length', 0))) if self.command == 'PUT' else None
            with guard:
                queue = endpoint.drops.get((self.command, path))
                drop = queue.popleft() if queue else None
                endpoint.arrivals.append((self.command, path, drop))
                if drop != 'before':
                    status = endpoint.handle(self.command, self.path, data, dict(self.headers))
                    headers, body = dict(endpoint.response_headers), endpoint.response_body
            if drop:
                self.close_connection = True  # no status line at all: the client sees the connection drop
                return
            try:
                self.send_response(status)
                for key, value in headers.items(): self.send_header(key, value)
                self.send_header('Content-Length', str(len(body))); self.end_headers()
                if self.command != 'HEAD': self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args): pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    endpoint.url = f'http://127.0.0.1:{http.server_port}/channel'
    endpoint.release = threading.Event()  # attach_background's server contract
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try: yield endpoint
    finally:
        http.shutdown(); http.server_close(); thread.join(2)


@pytest.fixture
def sink(world, server, monkeypatch):
    monkeypatch.setattr(publisher, 'UNKNOWN_RETRY_BASE_S', .05)
    return attach_background(world, server, monkeypatch)


def settle(sink, server, done, seconds=10):
    """Run retry passes (what each publisher cycle does) until done() or the bound expires."""
    def step():
        if done(): return True
        with sink.lock:
            sink.attempted.clear(); sink.retry(HEALTH)
        return False
    until(step, seconds)


def puts(server, path=PATH):
    return [a for a in server.arrivals if a[:2] == ('PUT', path)]


@pytest.mark.parametrize('existing', [True, False])
def test_put_dropped_before_arrival_is_resent_after_head(sink, server, existing):
    if existing:
        sink.write(PATH, b'{"v":1}'); until(lambda: server.objects.get(PATH) == b'{"v":1}' and PATH not in sink.pending)
    server.drops[('PUT', PATH)] = deque(['before'])
    sink.write(PATH, b'{"v":2}')
    until(lambda: sink.state['upload_attempts'].get(PATH, {}).get('phase') == 'retry')
    attempt = sink.state['upload_attempts'][PATH]
    assert (attempt['reason'], attempt['unknown_count']) == ('upload_unknown', 1)
    assert attempt['detail'] in ('RemoteDisconnected', 'ConnectionResetError', 'BadStatusLine')
    assert attempt['expected'] == (server.tags[PATH] if existing else None)
    settle(sink, server, lambda: PATH not in sink.pending)
    assert server.objects[PATH] == b'{"v":2}' and PATH not in sink.state['upload_attempts']
    last = [r for r in server.requests if r[:2] == ('PUT', PATH)][-1][3]
    assert ('If-Match' in last) is existing and (last.get('If-None-Match') == '*') is (not existing)
    assert [d for *_, d in puts(server)] == ([None] if existing else [])+['before', None]


@pytest.mark.parametrize('existing', [True, False])
def test_put_stored_but_answer_lost_is_confirmed_by_head_without_resending(sink, server, existing):
    if existing:
        sink.write(PATH, b'{"v":1}'); until(lambda: server.objects.get(PATH) == b'{"v":1}' and PATH not in sink.pending)
    server.drops[('PUT', PATH)] = deque(['after'])
    sink.write(PATH, b'{"v":2}')
    until(lambda: sink.state['upload_attempts'].get(PATH, {}).get('phase') == 'retry')
    assert server.objects[PATH] == b'{"v":2}'  # it did arrive
    settle(sink, server, lambda: PATH not in sink.pending)
    assert PATH not in sink.state['upload_attempts']
    assert [d for *_, d in puts(server)] == ([None] if existing else [])+['after']  # no second PUT of the same bytes
    assert server.arrivals[-1][:2] == ('HEAD', PATH)


def test_head_drop_changes_nothing_and_the_next_pass_uploads(sink, server):
    server.drops[('HEAD', PATH)] = deque(['before'])
    sink.write(PATH, b'{"v":1}')
    until(lambda: ('HEAD', PATH, 'before') in server.arrivals)
    assert PATH not in sink.state['upload_attempts'] and PATH in sink.pending
    settle(sink, server, lambda: PATH not in sink.pending)
    assert server.objects[PATH] == b'{"v":1}' and len(puts(server)) == 1


def test_consecutive_drops_past_the_limit_block_the_path(sink, server):
    server.drops[('PUT', PATH)] = deque(['before']*(publisher.UNKNOWN_RETRIES+1))
    sink.write(PATH, b'{"v":1}')
    settle(sink, server, lambda: sink.state['upload_attempts'].get(PATH, {}).get('phase') == 'blocked')
    attempt = sink.state['upload_attempts'][PATH]
    assert (attempt['reason'], attempt['unknown_count']) == ('upload_unknown', publisher.UNKNOWN_RETRIES+1)
    assert len(puts(server)) == publisher.UNKNOWN_RETRIES+1
    with sink.lock:
        sink.attempted.clear(); sink.retry(HEALTH)
    assert len(puts(server)) == publisher.UNKNOWN_RETRIES+1 and PATH in sink.pending  # blocked for this run


def test_drops_below_the_limit_recover_and_reset_the_count(sink, server):
    server.drops[('PUT', PATH)] = deque(['before']*publisher.UNKNOWN_RETRIES)
    sink.write(PATH, b'{"v":1}')
    settle(sink, server, lambda: PATH not in sink.pending)
    assert server.objects[PATH] == b'{"v":1}' and len(puts(server)) == publisher.UNKNOWN_RETRIES+1
    server.drops[('PUT', PATH)] = deque(['before'])
    sink.write(PATH, b'{"v":2}')
    until(lambda: sink.state['upload_attempts'].get(PATH, {}).get('phase') == 'retry')
    assert sink.state['upload_attempts'][PATH]['unknown_count'] == 1  # a confirmed success started a new count
    settle(sink, server, lambda: PATH not in sink.pending)
    assert server.objects[PATH] == b'{"v":2}'
