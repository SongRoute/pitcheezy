"""Focused tests for the offline demo fallback (synthetic tiny timeline; no network)."""
import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from capture_snapshot import needed_paths, snapshot_file
from serve_offline import load_snapshot, make_handler

GAME = 1


def decision(i, ab, pn, pa):
    return {'index': i, 'at_bat_number': ab, 'pitch_number': pn, 'pa_id': pa}


TIMELINE = {'game': {'game_pk': GAME}, 'decisions': [
    decision(0, 1, 1, 'a'), decision(1, 1, 2, 'a'), decision(2, 1, 3, 'a'),
    decision(3, 2, 1, 'b'), decision(4, 2, 2, 'b')]}


def test_needed_paths_cover_plate_appearance_prefix_and_summary():
    scenes = [{'pitchId': 'x', 'atBat': 1, 'pitchNumber': 3}, {'pitchId': 'y', 'atBat': 2, 'pitchNumber': 1}]
    paths, missing = needed_paths(TIMELINE, scenes)
    assert missing == []
    assert paths == [f'/api/watch/{GAME}/reveal/{j}' for j in (0, 1, 2, 3)] + \
        [f'/api/watch/{GAME}/summary?through=2&top=5', f'/api/watch/{GAME}/summary?through=3&top=5']


def test_needed_paths_report_unmatched_scene():
    _, missing = needed_paths(TIMELINE, [{'pitchId': 'z', 'atBat': 9, 'pitchNumber': 1}])
    assert missing == ['z']


def test_snapshot_file_names_are_flat_and_distinct():
    a = snapshot_file('/api/watch/1/summary?through=2&top=5')
    b = snapshot_file('/api/watch/1/summary?through=12&top=5')
    assert a != b and '/' not in a.removeprefix('api/') and '?' not in a


def write_bundle(tmp_path):
    dist = tmp_path / 'dist'
    (dist / 'assets').mkdir(parents=True)
    (dist / 'index.html').write_bytes(b'<html><body><div id="root"></div></body></html>')
    (dist / 'assets' / 'app.js').write_bytes(b'console.log(1)')
    (tmp_path / 'secret.txt').write_text('outside')
    snap = tmp_path / 'snap'
    answers = {f'/api/watch/{GAME}': json.dumps(TIMELINE).encode(),
               f'/api/watch/{GAME}/reveal/2': b'{"actual": "FF"}',
               f'/api/watch/{GAME}/summary?through=2&top=5': b'{"line": []}'}
    files = []
    for path, body in answers.items():
        target = snap / snapshot_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        files.append({'path': path, 'file': snapshot_file(path), 'bytes': len(body),
                      'sha256': hashlib.sha256(body).hexdigest()})
    (snap / 'manifest.json').write_text(json.dumps(
        {'game_pk': GAME, 'captured_kst': '2026-10-04 14:40 KST', 'files': files}))
    return dist, snap, answers


@pytest.fixture
def served(tmp_path):
    dist, snap, answers = write_bundle(tmp_path)
    manifest, loaded = load_snapshot(snap)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(dist, manifest, loaded))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_address[1]}', answers
    server.shutdown()
    server.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_saved_answers_are_byte_identical(served):
    base, answers = served
    for path, body in answers.items():
        status, headers, got = get(base + path)
        assert (status, got) == (200, body)
        assert headers['Content-Type'] == 'application/json'


def test_summary_query_order_does_not_matter(served):
    base, answers = served
    assert get(base + f'/api/watch/{GAME}/summary?top=5&through=2')[2] == answers[f'/api/watch/{GAME}/summary?through=2&top=5']


def test_unsaved_answers_are_explicit_errors_not_made_up(served):
    base, _ = served
    for path, code in [(f'/api/watch/{GAME}/reveal/3', 404), (f'/api/watch/{GAME}/summary?through=3&top=5', 404),
                       ('/api/watch/999', 404), (f'/api/watch/{GAME}?x=1', 404), ('/api/live/games?date=2026-10-06', 503),
                       ('/api/live/1/state', 503)]:
        status, _, body = get(base + path)
        assert status == code, path
        assert '오프라인' in json.loads(body)['detail']


def test_pages_carry_saved_copy_line_except_embedded_frame(served):
    base, _ = served
    status, _, page = get(base + '/showcase?scene=3')
    assert status == 200 and b'pz-offline-note' in page and '2026-10-04 14:40 KST'.encode() in page
    _, _, frame = get(base + f'/watch?game={GAME}&embed=1')
    assert b'pz-offline-note' not in frame and b'id="root"' in frame


def test_static_assets_and_no_path_escape(served):
    base, _ = served
    status, headers, body = get(base + '/assets/app.js')
    assert (status, body) == (200, b'console.log(1)') and 'immutable' in headers['Cache-Control']
    assert get(base + '/../secret.txt')[0] == 404
    assert get(base + '/assets/%2e%2e/%2e%2e/secret.txt')[0] == 404
    assert get(base + '/missing.js')[0] == 404


def test_changed_snapshot_file_stops_the_server(tmp_path):
    _, snap, _ = write_bundle(tmp_path)
    (snap / snapshot_file(f'/api/watch/{GAME}/reveal/2')).write_bytes(b'{"actual": "SL"}')
    with pytest.raises(SystemExit, match='changed'):
        load_snapshot(snap)
