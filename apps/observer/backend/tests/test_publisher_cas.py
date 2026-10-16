"""CAS state machine checks: synthetic JSON and socket-free gateway model only."""
import hashlib
import json
from pathlib import Path

import pytest
from observer_app import publisher
from test_publisher_upload import endpoint, config, TOKEN, GAME


@pytest.fixture
def mirror(tmp_path, endpoint, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    state, saved = {}, tmp_path/'state.json'
    sink = publisher.MirrorSink(publisher.LocalDirSink(tmp_path/'out'), publisher.HttpSink(config(endpoint)))
    sink.bind(state, lambda: saved.write_text(json.dumps(state)))
    return sink, state, saved


def restart(sink, saved):
    state = json.loads(saved.read_text())
    new = publisher.MirrorSink(sink.local, sink.remote)
    new.bind(state, lambda: saved.write_text(json.dumps(state)))
    return new, state


def test_create_replace_delete_identical_restore_has_distinct_conditions(mirror, endpoint):
    sink, state, _ = mirror
    body = b'{"synthetic":1}'
    sink.write(GAME, body)
    first = endpoint.tags[GAME]
    assert endpoint.requests[-1][3]['If-None-Match'] == '*'
    sink.write(GAME, body)
    second = endpoint.tags[GAME]
    assert second != first and endpoint.requests[-1][3]['If-Match'] == first
    sink.delete(GAME)
    deleted = endpoint.tags[GAME]
    assert deleted != second and GAME not in endpoint.objects
    assert sink.remote.head(GAME)['deleted'] is True
    sink.write(GAME, body)
    assert endpoint.tags[GAME] not in (first, second, deleted)
    assert endpoint.requests[-1][3]['If-Match'] == deleted
    assert state['upload_pending'] == {} and state['upload_attempts'] == {}


@pytest.mark.parametrize('method', ['put', 'delete'])
def test_conflict_and_newer_local_intent_never_rebase_even_after_restart(mirror, endpoint, method):
    sink, state, saved = mirror
    sink.write(GAME, b'{"initial":1}')
    endpoint.statuses[method.upper(), GAME] = 412
    getattr(sink, 'write' if method == 'put' else 'delete')(GAME, *([b'{"old":1}'] if method == 'put' else []))
    frozen = dict(state['upload_attempts'][GAME])
    assert frozen['phase'] == 'blocked' and frozen['reason'] == 'upload_conflict'
    endpoint.statuses[method.upper(), GAME] = 200
    sink, state = restart(sink, saved)
    endpoint.requests.clear()
    sink.write(GAME, b'{"latest":2}')
    sink.retry('v1/health.json')
    assert endpoint.requests == [] and state['upload_attempts'][GAME] == frozen
    assert state['upload_pending'][GAME] == 'put' and sink.read(GAME) == b'{"latest":2}'


@pytest.mark.parametrize('stored', [False, True])
def test_lost_response_is_blocked_with_or_without_remote_commit(mirror, endpoint, monkeypatch, stored):
    sink, state, saved = mirror
    original = endpoint.handle
    def uncertain(method, path, data, headers):
        if method == 'PUT':
            if stored:
                original(method, path, data, headers)
            raise TimeoutError(TOKEN)
        return original(method, path, data, headers)
    monkeypatch.setattr(endpoint, 'handle', uncertain)
    sink.write(GAME, b'{"attempt":1}')
    assert (GAME in endpoint.objects) is stored
    assert state['upload_attempts'][GAME]['reason'] == 'upload_unknown'
    monkeypatch.setattr(endpoint, 'handle', original)
    sink, state = restart(sink, saved)
    endpoint.requests.clear()
    sink.write(GAME, b'{"newer":2}')
    assert endpoint.requests == []  # matching remote SHA is not an ownership receipt
    assert state['upload_attempts'][GAME]['sha256'] == hashlib.sha256(b'{"attempt":1}').hexdigest()


@pytest.mark.parametrize('damage', ['weak', 'missing', 'wrong_sha', 'stored_false', 'deleted', 'schema', 'too_large', 'not_json'])
def test_invalid_success_receipts_are_unknown_and_never_retried(mirror, endpoint, monkeypatch, damage):
    sink, state, saved = mirror
    original = endpoint.handle
    def broken(method, path, data, headers):
        status = original(method, path, data, headers)
        if method == 'PUT' and status == 200:
            if damage == 'weak': endpoint.response_headers['ETag'] = 'W/'+endpoint.response_headers['ETag']
            elif damage == 'missing': endpoint.response_headers.pop('ETag')
            elif damage == 'too_large': endpoint.response_body = b'x'*5000
            elif damage == 'not_json': endpoint.response_body = b'bad'
            else:
                body = json.loads(endpoint.response_body)
                body[{'wrong_sha':'sha256', 'stored_false':'stored', 'deleted':'deleted', 'schema':'schema_validated'}[damage]] = '0'*64 if damage == 'wrong_sha' else damage in ('deleted', 'schema')
                endpoint.response_body = json.dumps(body).encode()
        return status
    monkeypatch.setattr(endpoint, 'handle', broken)
    sink.write(GAME, b'{}')
    assert state['upload_attempts'][GAME]['phase'] == 'blocked'
    assert state['upload_pending'][GAME] == 'put'
    sink, state = restart(sink, saved)
    endpoint.requests.clear()
    sink.write(GAME, b'{"new":1}')
    assert endpoint.requests == []


def test_attempt_is_fsynced_before_write_and_survives_interruption(mirror, endpoint, monkeypatch):
    sink, state, saved = mirror
    original = sink.remote.write
    def interrupt(path, data, *, expected, final=False):
        persisted = json.loads(saved.read_text())['upload_attempts'][path]
        assert persisted == state['upload_intents'][path] | {'expected': expected, 'phase': 'inflight'}
        assert persisted['sha256'] == hashlib.sha256(data).hexdigest()
        raise KeyboardInterrupt
    monkeypatch.setattr(sink.remote, 'write', interrupt)
    with pytest.raises(KeyboardInterrupt): sink.write(GAME, b'{}')
    monkeypatch.setattr(sink.remote, 'write', original)
    sink, state = restart(sink, saved)
    assert state['upload_attempts'][GAME]['reason'] == 'interrupted_unknown'
    endpoint.requests.clear()
    sink.write(GAME, b'{"new":1}')
    assert endpoint.requests == []


def test_legacy_outbox_is_blocked_and_destination_is_bound(tmp_path, endpoint, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN', TOKEN)
    local = publisher.LocalDirSink(tmp_path/'out')
    local.write(GAME, b'{}')
    state = {'upload_pending': {GAME:'put'}}
    sink = publisher.MirrorSink(local, publisher.HttpSink(config(endpoint)))
    sink.bind(state, lambda: None)
    sink.write(GAME, b'{"new":1}')
    assert endpoint.requests == [] and state['upload_attempts'][GAME]['reason'] == 'legacy_unknown'
    changed = publisher.MirrorSink(local, publisher.HttpSink(config(endpoint, base_url=endpoint.url+'/changed')))
    with pytest.raises(publisher.UploadError, match='upload_destination_changed'): changed.bind(state, lambda: None)


@pytest.mark.parametrize('path', ['v2/health.json', 'v1/game/0.json', 'v1/game/01.json', 'v1/game/1234567890123.json', 'v1/games/2026-02-30.json'])
def test_partner_whitelist_is_stricter_than_local_versions(mirror, endpoint, path):
    sink, _, _ = mirror
    with pytest.raises(publisher.UploadError, match='upload_path_invalid'): sink.remote.write(path, b'{}')
    assert endpoint.requests == []


@pytest.mark.parametrize('body', [b'[]', b'null', b'"s"', b'\xff', b'{"v":'+b'0'*publisher.UPLOAD_MAX_BYTES+b'}'])
def test_invalid_or_oversized_bytes_never_reach_transport(mirror, endpoint, body):
    sink, _, _ = mirror
    with pytest.raises(publisher.UploadError, match='upload_body_invalid'): sink.remote.write(GAME, body)
    assert endpoint.requests == []


def test_sha_mismatch_is_rejected_by_cas_endpoint(mirror, endpoint):
    sink, _, _ = mirror
    assert endpoint.handle('PUT', '/channel/'+GAME, b'{}', {'If-None-Match':'*', 'X-Content-SHA256':'0'*64}) == 422
    assert endpoint.objects == {}


def test_delete_absent_is_metadata_noop_and_tombstone_is_not_absent(mirror, endpoint):
    sink, state, _ = mirror
    sink.delete(GAME)
    assert [request[0] for request in endpoint.requests] == ['HEAD']
    assert state['upload_pending'] == {}
    sink.write(GAME, b'{}')
    sink.delete(GAME)
    first = endpoint.tags[GAME]
    sink.delete(GAME)
    assert endpoint.tags[GAME] != first and endpoint.requests[-1][3]['If-Match'] == first


def test_old_acknowledgement_does_not_clear_newer_local_intent(mirror, endpoint):
    sink, state, _ = mirror
    latest = b'{"latest":2}'
    def new_intent(method, path):
        if method == 'PUT' and path == GAME:
            sink._queue(path, 'put', latest)
            sink.local.write(path, latest)
    endpoint.after = new_intent
    sink.write(GAME, b'{"old":1}')
    assert state['upload_pending'][GAME] == 'put'
    assert state['upload_intents'][GAME]['sha256'] == hashlib.sha256(latest).hexdigest()
    assert endpoint.objects[GAME] != sink.read(GAME)
    endpoint.after = lambda *args: None
    sink.attempted.clear()
    sink.retry('v1/health.json')
    assert endpoint.objects[GAME] == latest and state['upload_pending'] == {}


def test_first_create_race_is_412_without_refreshed_condition(mirror, endpoint, monkeypatch):
    sink, state, _ = mirror
    original = sink.remote.head
    def raced(path):
        result = original(path)
        assert result['etag'] is None
        endpoint.handle('PUT', '/channel/'+path, b'{"other":1}', {'If-None-Match':'*',
            'X-Content-SHA256':hashlib.sha256(b'{"other":1}').hexdigest()})
        return result
    monkeypatch.setattr(sink.remote, 'head', raced)
    sink.write(GAME, b'{"ours":1}')
    assert state['upload_attempts'][GAME]['reason'] == 'upload_conflict'
    assert endpoint.objects[GAME] == b'{"other":1}'
    requests = len(endpoint.requests)
    sink.write(GAME, b'{"ours":2}')
    assert len(endpoint.requests) == requests


@pytest.mark.parametrize('damage', [{'phase':'garbage'}, {'phase':'retry'}, {'phase':'rejected','expected':'W/"a"'}, {'phase':'inflight','revision':-1}])
def test_malformed_attempt_metadata_is_not_silently_reset(mirror, damage):
    sink, state, saved = mirror
    state['upload_attempts'][GAME] = damage
    saved.write_text(json.dumps(state))
    with pytest.raises(publisher.UploadError, match='upload_state_invalid'): restart(sink, saved)
