"""HTTP upload contract, synthetic publisher games only; loopback or socket-free fallback."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import logging
from pathlib import Path
import threading
import traceback

import pytest

from observer_app import publisher
from publish_fixture import ARCHIVE_GAME, ARCHIVE_DATE, LIVE_GAME, snap
from test_publisher import make, world, start_demo, codes, candidates, unconfirmed

TOKEN = 'upload-test-secret-never-persisted'
GAME = f'v1/game/{ARCHIVE_GAME}.json'
SETUP = f'v1/setup/{ARCHIVE_GAME}.json'
HEALTH = 'v1/health.json'


class Endpoint:
    def __init__(self):
        self.requests, self.objects, self.statuses = [], {}, {}
        self.default = 204
        self.after = lambda method, path: None

    def handle(self, method, path, data, headers):
        assert path.startswith('/channel/')
        path = path.removeprefix('/channel/')
        self.requests.append((method, path, data, dict(headers)))
        status = self.statuses.get((method, path), self.default)
        if isinstance(status, deque):
            status = status.popleft()
        if 200 <= status < 300:
            if method == 'PUT':
                self.objects[path] = data
            else:
                self.objects.pop(path, None)
        self.after(method, path)
        return status


@pytest.fixture
def endpoint(monkeypatch):
    endpoint = Endpoint()

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            data = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            self.respond('PUT', data)

        def do_DELETE(self):
            self.respond('DELETE', None)

        def respond(self, method, data):
            status = endpoint.handle(method, self.path, data, self.headers)
            self.send_response(status)
            self.end_headers()
            # Deliberately hostile error response; the transport must never read/log it.
            if status >= 400:
                try:
                    self.wfile.write(TOKEN.encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *_):
            pass

    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    except PermissionError:
        class Connection:
            def __init__(self, host, port=None, *, timeout):
                assert host == '127.0.0.1' and 0 < timeout <= 60

            def request(self, method, path, body=None, headers=None):
                self.status = endpoint.handle(method, path, body, headers)

            def getresponse(self):
                return type('Response', (), {'status': self.status})()

            def close(self):
                pass
        monkeypatch.setattr(http.client, 'HTTPConnection', Connection)
        endpoint.url, endpoint.mode = 'http://127.0.0.1/channel', 'in-process (sandbox blocked bind)'
        yield endpoint
    else:
        endpoint.url, endpoint.mode = f'http://127.0.0.1:{server.server_port}/channel', 'loopback HTTP server'
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield endpoint
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def config(endpoint, **overrides):
    return publisher.load_config(env={})['upload'] | {'enabled': True, 'base_url': endpoint.url, 'retries': 0} | overrides


def attach(world, endpoint, monkeypatch, **overrides):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    world.config['upload'] = config(endpoint, **overrides)
    world.start()
    return world.publisher.sink


def test_http_happy_path_health_last_and_headers(world, endpoint, monkeypatch, record_property):
    record_property('http_test_transport', endpoint.mode)
    sink = attach(world, endpoint, monkeypatch)
    world.cycle(settle=False)
    assert endpoint.requests[-1][0:2] == ('PUT', HEALTH)
    assert set(endpoint.objects) == set(world.files())
    for method, path, data, headers in endpoint.requests:
        assert method in ('PUT', 'DELETE')
        assert headers['Authorization'] == f'Bearer {TOKEN}'
        assert headers['Content-Type'] == 'application/json'
        if method == 'PUT':
            assert data == world.path(path).read_bytes()
            kind = 'final' if json.loads(data)['publish'].get('final') else 'live'
            assert headers['Cache-Control'] == world.config['upload']['headers'][kind]['Cache-Control']
    assert sink.read(GAME) == world.path(GAME).read_bytes()
    assert sink.read_tail(GAME, 50) == sink.read(GAME)[-50:]
    assert sink.remote.read(GAME) is None
    assert world.publisher.state['upload_pending'] == {}


def test_retry_with_backoff_then_success(world, endpoint, monkeypatch):
    waits = []
    attach(world, endpoint, monkeypatch, retries=2, backoff_s=0.1)
    monkeypatch.setattr(publisher.time, 'sleep', waits.append)
    endpoint.statuses['PUT', GAME] = deque([503, 429, 201])
    world.cycle(settle=False)
    assert waits == [0.1, 0.2]
    assert len([r for r in endpoint.requests if r[:2] == ('PUT', GAME)]) == 3
    assert world.publisher.state['upload_pending'] == {}


def test_permanent_failure_isolated_pending_survives_restart_and_unchanged_file_retried(world, endpoint, monkeypatch):
    attach(world, endpoint, monkeypatch)
    endpoint.statuses['PUT', GAME] = 403
    health = world.cycle(settle=False)
    assert codes(health) == [(ARCHIVE_GAME, 'upload_failed')]
    assert HEALTH in endpoint.objects and f'v1/game/{LIVE_GAME}.json' in endpoint.objects
    saved = json.loads(world.publisher.state_path.read_text())
    assert saved['upload_pending'] == {GAME: 'put'}
    before = world.path(GAME).read_bytes()
    world.start()
    endpoint.requests.clear()
    endpoint.statuses['PUT', GAME] = 204
    health = world.cycle(settle=False)
    assert codes(health) == [] and health['upload']['pending'] == 0
    assert endpoint.objects[GAME] == before == world.path(GAME).read_bytes()
    assert endpoint.requests[-1][1] == HEALTH and any(r[:2] == ('PUT', GAME) for r in endpoint.requests)
    assert json.loads(world.publisher.state_path.read_text())['upload_pending'] == {}


def test_health_failure_retries_latest_health_next_cycle(world, endpoint, monkeypatch):
    attach(world, endpoint, monkeypatch)
    endpoint.statuses['PUT', HEALTH] = 500
    world.cycle(settle=False)
    assert world.publisher.state['upload_pending'] == {HEALTH: 'put'}
    assert world.read(HEALTH)['sequence'] == 1
    endpoint.statuses['PUT', HEALTH] = 200
    endpoint.requests.clear()
    health = world.cycle(settle=False)
    assert health['upload']['code'] == 'upload_failed'  # previous failed health, sampled before this PUT
    assert json.loads(endpoint.objects[HEALTH])['sequence'] == 2
    assert [r[1] for r in endpoint.requests].count(HEALTH) == 1 and endpoint.requests[-1][1] == HEALTH
    assert world.publisher.state['upload_pending'] == {}


@pytest.mark.parametrize('status', [204, 404, 503])
def test_withdrawal_replaces_pending_put_and_retries_delete(world, endpoint, monkeypatch, status):
    attach(world, endpoint, monkeypatch)
    endpoint.statuses['PUT', GAME] = 503
    world.cycle(settle=False)
    endpoint.statuses['DELETE', GAME] = status
    endpoint.requests.clear()
    sid, head = start_demo(world)
    health = world.cycle(settle=False)
    assert not world.path(GAME).exists() and not world.path(SETUP).exists()
    assert ('PUT', GAME) not in [r[:2] for r in endpoint.requests]
    assert ('DELETE', GAME) in [r[:2] for r in endpoint.requests]
    assert ('DELETE', SETUP) in [r[:2] for r in endpoint.requests]
    assert world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['embargoed']
    if status == 503:
        assert codes(health) == [(ARCHIVE_GAME, 'upload_failed')]
        assert world.publisher.state['upload_pending'] == {GAME: 'delete'}
        world.start()
        endpoint.statuses['DELETE', GAME] = 404
        assert codes(world.cycle(settle=False)) == []
    assert world.publisher.state['upload_pending'] == {}
    assert world.client.post(f'/api/service/operator/sessions/{sid}/control', json={'action': 'end'}, headers=head).status_code == 200
    endpoint.statuses['PUT', GAME] = 204
    world.cycle(settle=False)
    assert GAME in endpoint.objects and world.path(GAME).exists()


def test_pending_setup_delete_retried_when_game_not_due(world, endpoint, monkeypatch):
    attach(world, endpoint, monkeypatch)
    world.cycle(settle=False)
    world.write_setup(sha256='0'*64)
    endpoint.statuses['DELETE', SETUP] = 503
    world.cycle(after=world.config['settled_recheck_s'], settle=False)
    assert world.publisher.state['upload_pending'] == {SETUP: 'delete'}
    endpoint.statuses['DELETE', SETUP] = 204
    world.start()
    world.cycle(settle=False)
    assert SETUP not in endpoint.objects and world.publisher.state['upload_pending'] == {}


def test_late_embargo_during_pending_retry_is_withdrawn_before_health(world, endpoint, monkeypatch):
    attach(world, endpoint, monkeypatch)
    endpoint.statuses['PUT', GAME] = 503
    world.cycle(settle=False)
    endpoint.statuses['PUT', GAME] = 204
    started = []
    def after(method, path):
        if (method, path) == ('PUT', GAME) and not started:
            started.append(start_demo(world))
    endpoint.after = after
    world.cycle(settle=False)
    assert started and GAME not in endpoint.objects and SETUP not in endpoint.objects
    assert endpoint.requests[-1][1] == HEALTH


def test_local_invariants_during_whole_run_outage(world, endpoint, monkeypatch):
    sink = attach(world, endpoint, monkeypatch)
    endpoint.default = 503
    world.cycle()
    assert candidates(world.game()) == ['SL']
    assert world.feed.store.served_recommendations(LIVE_GAME)
    world.cycle(unconfirmed())
    body = world.game()
    assert body['receipts']['unconfirmed'] == 1 and candidates(body) == []
    assert body['pitches'][-1]['actual']['pitch_type'] is None
    world.cycle(snap('20261005_100100', half='final'), settle=False)
    final, sequence = world.path(f'v1/game/{LIVE_GAME}.json').read_bytes(), world.publisher.state['sequence']
    world.start()
    world.cycle(settle=False)
    assert world.path(f'v1/game/{LIVE_GAME}.json').read_bytes() == final
    assert world.publisher.state['sequence'] == sequence+1
    assert endpoint.objects == {} and all(publisher.public_path(p) for p in world.files())
    assert list(sink.local.staging.iterdir()) == []
    start_demo(world)
    world.cycle(settle=False)
    assert not world.path(GAME).exists() and not world.path(SETUP).exists()
    # Lost state still recovers sequence from local tails, without remote GETs.
    old = world.publisher.state['sequence']
    world.publisher.close()
    world.publisher.state_path.unlink()
    world.start()
    assert world.publisher.state['sequence'] == old
    world.cycle(settle=False)
    assert world.publisher.state['sequence'] == old+1
    with pytest.raises(publisher.AlreadyRunning):
        publisher.Publisher(None, publisher.MirrorSink(publisher.LocalDirSink(world.out), sink.remote), config=world.config)


@pytest.mark.parametrize('path', ['v1/.secret.json', '.env', 'v1/.tmp/game/1.json', 'v1/game/1.json.tmp',
                                  'v1/game/.1.json', 'v1/notes.json', '../v1/health.json', '/v1/health.json',
                                  'v1/game/../health.json', 'v1//health.json', 'v1/game/1.json?x'])
def test_dot_temp_and_non_public_names_never_uploaded(endpoint, monkeypatch, path):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    remote = publisher.HttpSink(config(endpoint))
    for action in (lambda: remote.write(path, b'{}'), lambda: remote.delete(path)):
        with pytest.raises(publisher.UploadError, match='^upload_path_invalid$'):
            action()
    assert endpoint.requests == []


def test_existing_local_files_seed_only_public_names(world, endpoint, monkeypatch):
    world.cycle(settle=False)
    (world.out/'v1/.private.json').write_text(TOKEN)
    (world.out/'v1/notes.json').write_text(TOKEN)
    (world.out/'v1/game/1.json.tmp').write_text(TOKEN)
    attach(world, endpoint, monkeypatch)
    world.cycle(settle=False)
    assert GAME in endpoint.objects and SETUP in endpoint.objects
    assert all(publisher.public_path(r[1]) and TOKEN.encode() not in (r[2] or b'') for r in endpoint.requests)


@pytest.mark.parametrize('url', ['http://example.com/api', 'http://127.0.0.1.example.com', 'ftp://localhost',
                                 'https://user:secret@example.com', 'https://example.com/?token=secret',
                                 'https://example.com/#secret', 'https://', 'http://192.168.1.1'])
def test_https_enforcement_and_no_url_credentials(endpoint, url):
    with pytest.raises(publisher.UploadError, match='^upload_config_invalid$'):
        publisher.HttpSink(config(endpoint, base_url=url), dry_run=True)


@pytest.mark.parametrize('url', ['https://example.invalid/api', 'http://localhost:123', 'http://127.0.0.2', 'http://[::1]:123'])
def test_https_or_loopback_allowed_without_connecting(endpoint, url):
    publisher.HttpSink(config(endpoint, base_url=url), dry_run=True)


@pytest.mark.parametrize('option', [{'timeout_s': 0}, {'timeout_s': float('nan')}, {'timeout_s': 61}, {'retries': 6},
                                     {'retries': -1}, {'retries': 1.5}, {'backoff_s': float('inf')}, {'backoff_s': -1},
                                     {'headers': {'live': {'Content-Type': 'application/json', 'Cache-Control': 'x\r\nAuthorization: evil'}}}])
def test_upload_limits_validated(endpoint, option):
    with pytest.raises(publisher.UploadError, match='^upload_config_invalid$'):
        publisher.HttpSink(config(endpoint, **option), dry_run=True)


def test_token_not_in_logs_exception_health_or_state(world, endpoint, monkeypatch, caplog, capsys):
    attach(world, endpoint, monkeypatch)
    endpoint.default = 403
    caplog.set_level(logging.INFO)
    world.cycle(settle=False)
    errors = []
    with pytest.raises(publisher.UploadError) as caught:
        world.publisher.sink.remote.write(GAME, b'{}')
    errors.append(''.join(traceback.format_exception(caught.value)))
    def broken(*args, **kwargs):
        raise OSError(TOKEN)
    monkeypatch.setattr(http.client, 'HTTPConnection', broken)
    with pytest.raises(publisher.UploadError) as caught:
        world.publisher.sink.remote.write(GAME, b'{}')
    errors.append(''.join(traceback.format_exception(caught.value)))
    world.cycle(settle=False)
    captured = capsys.readouterr()
    assert TOKEN not in caplog.text+captured.out+captured.err+''.join(errors)
    for path in [*world.out.rglob('*'), world.publisher.state_path]:
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes()


def test_token_file_outside_checkout_and_invalid_sources(tmp_path, endpoint, monkeypatch):
    token = tmp_path/'token'
    token.write_text(TOKEN+'\n')
    remote = publisher.HttpSink(config(endpoint, token_env=None, token_file=str(token)))
    remote.write(GAME, b'{}')
    assert endpoint.requests[-1][3]['Authorization'] == f'Bearer {TOKEN}'
    monkeypatch.delenv('PITCHEEZY_PUBLISH_TOKEN', raising=False)
    for options in ({}, {'token_file': str(token)}, {'token_env': None, 'token_file': 'relative'},
                    {'token_env': None, 'token_file': str(publisher.CONFIG_PATH)},
                    {'token_env': None, 'token_file': str(tmp_path/'absent')}):
        with pytest.raises(publisher.UploadError, match='^upload_config_invalid$'):
            publisher.HttpSink(config(endpoint, **options))
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN+'\r\nX: injected')
    with pytest.raises(publisher.UploadError) as caught:
        publisher.HttpSink(config(endpoint))
    assert TOKEN not in ''.join(traceback.format_exception(caught.value))


@pytest.mark.parametrize('method,status,success', [('PUT', 202, True), ('PUT', 299, True), ('PUT', 404, False),
                                                 ('PUT', 302, False), ('DELETE', 404, True), ('DELETE', 401, False)])
def test_status_contract(endpoint, monkeypatch, method, status, success):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    remote = publisher.HttpSink(config(endpoint))
    endpoint.statuses[method, GAME] = status
    action = (lambda: remote.write(GAME, b'{}')) if method == 'PUT' else (lambda: remote.delete(GAME))
    if success:
        action()
    else:
        with pytest.raises(publisher.UploadError, match='^upload_failed$'):
            action()
    assert len(endpoint.requests) == 1


def cli_config(tmp_path, endpoint, **overrides):
    path = tmp_path/'config.json'
    path.write_text(json.dumps(publisher.load_config(env={}) | {'upload': config(endpoint, **overrides)}))
    return str(path)


def test_check_remote_cli_does_not_build_app(tmp_path, endpoint, monkeypatch, capsys):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    monkeypatch.setattr(publisher, 'build', lambda *a, **kw: pytest.fail('probe must not build an app'))
    assert publisher.main(['--check-remote', '--config', cli_config(tmp_path, endpoint)]) == 0
    assert 'PUT and DELETE accepted' in capsys.readouterr().out
    assert [r[0] for r in endpoint.requests] == ['PUT', 'DELETE']
    assert endpoint.requests[0][1] == endpoint.requests[1][1]
    assert endpoint.requests[0][1].startswith('_publisher_probe/') and endpoint.objects == {}


@pytest.mark.parametrize('failed_method', ['PUT', 'DELETE'])
def test_check_remote_cli_failure_still_cleans_up(tmp_path, endpoint, monkeypatch, failed_method):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    original = endpoint.handle
    def handle(method, path, data, headers):
        endpoint.default = 403 if method == failed_method else 204
        return original(method, path, data, headers)
    endpoint.handle = handle
    with pytest.raises(SystemExit, match='remote check failed: upload_failed'):
        publisher.main(['--check-remote', '--config', cli_config(tmp_path, endpoint)])
    assert [r[0] for r in endpoint.requests] == ['PUT', 'DELETE']


def test_dry_run_cli_sends_nothing_and_real_run_retries(world, endpoint, monkeypatch, caplog):
    from live_fixture import listed_dir
    tmp = world.tmp/'cli'
    tmp.mkdir()
    empty = tmp/'empty'
    empty.mkdir()
    monkeypatch.delenv('PITCHEEZY_PUBLISH_TOKEN', raising=False)
    monkeypatch.delenv('PITCHEEZY_PUBLISH_STATE', raising=False)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(listed_dir(tmp/'replay')))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S', '0')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp/'store.sqlite3'))
    path = cli_config(tmp, endpoint, enabled=False)
    args = ['--once', '--config', path, '--out', str(tmp/'out'), '--date', '2026-09-30',
            '--feed-dir', str(empty), '--watch-dir', str(empty), '--intent-dir', str(empty), '--demo-dir', str(tmp/'demo')]
    caplog.set_level(logging.INFO)
    assert publisher.main(args+['--dry-run-upload']) == 0
    assert endpoint.requests == [] and 'dry-run-upload PUT v1/health.json' in caplog.text
    state_path = tmp/'out.state.json'
    assert json.loads(state_path.read_text())['upload_pending']
    # Real upload subsequently sends all dry-run files.
    cfg = json.loads(Path(path).read_text())
    cfg['upload']['enabled'] = True
    Path(path).write_text(json.dumps(cfg))
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    assert publisher.main(args) == 0
    assert endpoint.requests[-1][1] == HEALTH and json.loads(state_path.read_text())['upload_pending'] == {}
    assert TOKEN not in caplog.text


def test_transport_timeout_has_bounded_attempts_and_no_exception_text(endpoint, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    timeouts, waits = [], []
    def timeout(host, port, *, timeout):
        timeouts.append(timeout)
        raise TimeoutError(TOKEN)
    monkeypatch.setattr(http.client, 'HTTPConnection', timeout)
    monkeypatch.setattr(publisher.time, 'sleep', waits.append)
    with pytest.raises(publisher.UploadError) as caught:
        publisher.HttpSink(config(endpoint, timeout_s=1.5, retries=2, backoff_s=0.2)).write(GAME, b'{}')
    assert timeouts == [1.5]*3 and waits == [0.2, 0.4]
    assert TOKEN not in ''.join(traceback.format_exception(caught.value))


def test_pending_is_durable_before_remote_attempt_even_if_process_crashes(world, endpoint, monkeypatch):
    sink = attach(world, endpoint, monkeypatch)
    real = sink.remote.write
    def crash(path, data, *, final=False):
        if path == GAME:
            assert json.loads(world.publisher.state_path.read_text())['upload_pending'][path] == 'put'
            raise KeyboardInterrupt
        return real(path, data, final=final)
    monkeypatch.setattr(sink.remote, 'write', crash)
    with pytest.raises(KeyboardInterrupt):
        world.cycle(settle=False)
    before = world.path(GAME).read_bytes()
    world.start()
    world.cycle(settle=False)
    assert endpoint.objects[GAME] == before == world.path(GAME).read_bytes()
    assert world.publisher.state['sequence'] > json.loads(before)['publish']['sequence']
    assert world.publisher.state['upload_pending'] == {}


def test_local_write_failure_never_sends_uncommitted_bytes(world, endpoint, monkeypatch):
    sink = attach(world, endpoint, monkeypatch)
    original = sink.local.write
    def broken(path, data, *, final=False):
        if path == GAME:
            raise OSError('disk full')
        return original(path, data, final=final)
    monkeypatch.setattr(sink.local, 'write', broken)
    health = world.cycle(settle=False)
    assert GAME not in endpoint.objects and not world.path(GAME).exists()
    assert (ARCHIVE_GAME, 'write_failed') in codes(health)
    assert endpoint.requests[-1][1] == HEALTH


def test_dry_run_withdrawal_logs_delete_without_loading_secret(world, endpoint, monkeypatch, caplog):
    world.config['upload'] = config(endpoint)
    world.config['dry_run_upload'] = True
    monkeypatch.delenv('PITCHEEZY_PUBLISH_TOKEN', raising=False)
    world.start()
    world.cycle(settle=False)
    start_demo(world)
    caplog.set_level(logging.INFO)
    world.cycle(settle=False)
    assert f'dry-run-upload DELETE {GAME}' in caplog.text
    assert endpoint.requests == [] and world.publisher.state['upload_pending'][GAME] == 'delete'


def test_mirror_preserves_local_symlink_guards(world, endpoint, monkeypatch):
    attach(world, endpoint, monkeypatch)
    world.cycle(settle=False)
    world.publisher.close()
    world.path(GAME).unlink()
    world.path(GAME).symlink_to(world.tmp/'outside.json')
    with pytest.raises(publisher.UnsafeOutput):
        world.start()


def test_reactivation_uploads_file_finalized_while_disabled(world, endpoint, monkeypatch):
    attach(world, endpoint, monkeypatch)
    world.cycle(settle=False)
    path = f'v1/game/{LIVE_GAME}.json'
    world.config['upload']['enabled'] = False
    world.start()
    world.cycle(snap('20261005_102100', half='final'), settle=False)
    new_path = 'v1/game/990099.json'
    world.publisher._put(new_path, {'synthetic': 'new settled file'}, always=False, final=True)
    world.publisher._save_state()
    new_file = world.path(new_path).read_bytes()
    final = world.path(path).read_bytes()
    assert json.loads(final)['publish']['final'] is True
    assert endpoint.objects[path] != final
    assert json.loads(world.publisher.state_path.read_text())['upload_pending'][path] == 'put'
    # Even an older local-only writer that failed to track the PUT is recovered by catalog reconciliation.
    world.publisher.state['upload_pending'].pop(path)
    world.publisher._save_state()
    world.config['upload']['enabled'] = True
    world.start()
    health = world.cycle(settle=False)
    assert endpoint.objects[path] == final == world.path(path).read_bytes()
    assert endpoint.objects[new_path] == new_file == world.path(new_path).read_bytes()
    assert world.publisher.state['upload_pending'] == {} and codes(health) == []


def test_reactivation_preserves_deletion_made_while_disabled(world, endpoint, monkeypatch):
    attach(world, endpoint, monkeypatch)
    world.cycle(settle=False)
    world.config['upload']['enabled'] = False
    world.start()
    assert world.publisher._drop(GAME)
    assert GAME in endpoint.objects and not world.path(GAME).exists()
    assert json.loads(world.publisher.state_path.read_text())['upload_pending'][GAME] == 'delete'
    world.config['upload']['enabled'] = True
    world.start()
    world.publisher.sink.retry(HEALTH)
    assert GAME not in endpoint.objects
    assert ('DELETE', GAME) in [r[:2] for r in endpoint.requests]


def test_reactivation_keeps_pending_delete_over_residual_local_file(world, endpoint, monkeypatch):
    sink = attach(world, endpoint, monkeypatch)
    world.cycle(settle=False)
    def failed_delete(path):
        raise OSError('local delete failed')
    monkeypatch.setattr(sink.local, 'delete', failed_delete)
    assert not world.publisher._drop(GAME)
    assert world.path(GAME).exists() and GAME in endpoint.objects
    world.config['upload']['enabled'] = False
    world.start()
    world.config['upload']['enabled'] = True
    world.start()
    assert world.publisher.sink.pending[GAME] == 'delete'
    world.publisher.sink.retry(HEALTH)
    assert not world.path(GAME).exists() and GAME not in endpoint.objects
