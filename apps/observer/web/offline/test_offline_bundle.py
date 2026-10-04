"""Focused tests for the offline demo fallback (synthetic tiny timeline; no network)."""
import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from capture_snapshot import all_decision_paths, check_setup, needed_paths, setup_path, snapshot_file
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
        {'game_pk': GAME, 'scenes': 2, 'captured_kst': '2026-10-04 14:40 KST', 'files': files}))
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
    assert '경기 1 시연 장면만'.encode() in page and '전체'.encode() not in page  # v1/v2 manifest: no coverage field
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


def test_repeated_summary_params_have_no_saved_answer(served):
    base, _ = served
    for query in ('through=2&through=3&top=5', 'through=2&top=5&top=6'):
        status, _, body = get(base + f'/api/watch/{GAME}/summary?{query}')
        assert status == 404 and '시연 장면 2개' in json.loads(body)['detail'], query


def test_manifest_cannot_point_outside_api_or_lose_a_file(tmp_path):
    _, snap, _ = write_bundle(tmp_path)
    manifest = json.loads((snap / 'manifest.json').read_text())
    entry = manifest['files'][1]
    (snap / entry['file']).unlink()
    with pytest.raises(SystemExit, match='missing'):
        load_snapshot(snap)
    entry['file'] = '../secret.txt'
    (snap / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match='unexpected file name'):
        load_snapshot(snap)


def test_all_decision_paths_cover_every_reveal_and_summary():
    paths = all_decision_paths(TIMELINE)
    assert paths == [f'/api/watch/{GAME}/reveal/{i}' for i in range(5)] + \
        [f'/api/watch/{GAME}/summary?through={i}&top=5' for i in range(5)]
    scenes = [{'pitchId': 'x', 'atBat': 1, 'pitchNumber': 3}, {'pitchId': 'y', 'atBat': 2, 'pitchNumber': 2}]
    assert set(needed_paths(TIMELINE, scenes)[0]) <= set(paths)


def test_all_decision_paths_refuse_gapped_indices():
    gapped = {'game': {'game_pk': GAME}, 'decisions': [decision(0, 1, 1, 'a'), decision(2, 1, 2, 'a')]}
    with pytest.raises(SystemExit, match='0..N-1'):
        all_decision_paths(gapped)


def write_all_bundle(tmp_path, drop=None, **manifest_overrides):
    dist = tmp_path / 'dist'
    dist.mkdir()
    (dist / 'index.html').write_bytes(b'<html><body><div id="root"></div></body></html>')
    snap = tmp_path / 'snap'
    answers = {f'/api/watch/{GAME}': json.dumps(TIMELINE).encode()}
    for i, path in enumerate(all_decision_paths(TIMELINE)):
        answers[path] = json.dumps({'n': i}).encode()
    answers.pop(drop, None)
    files = []
    for path, body in answers.items():
        target = snap / snapshot_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        files.append({'path': path, 'file': snapshot_file(path), 'bytes': len(body),
                      'sha256': hashlib.sha256(body).hexdigest()})
    (snap / 'manifest.json').write_text(json.dumps(
        {'game_pk': GAME, 'scenes': 2, 'top': 5, 'coverage': 'all_decisions', 'decisions': 5,
         'captured_kst': '2026-10-04 15:00 KST', 'files': files, **manifest_overrides}))
    return dist, snap, answers


def test_all_decisions_bundle_serves_every_pitch_and_says_so(tmp_path):
    dist, snap, answers = write_all_bundle(tmp_path)
    manifest, loaded = load_snapshot(snap)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(dist, manifest, loaded))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        for path, body in answers.items():
            assert get(base + path)[::2] == (200, body), path
        page = get(base + '/showcase')[2].decode()
        assert '공 5개 전체' in page and '시연 장면만' not in page
        status, _, body = get(base + f'/api/watch/{GAME}/reveal/5')
        assert status == 404 and json.loads(body)['detail'].endswith(f'경기 {GAME}의 공 5개만 저장돼 있어요.')
    finally:
        server.shutdown()
        server.server_close()


def test_all_decisions_bundle_missing_an_answer_does_not_start(tmp_path):
    _, snap, _ = write_all_bundle(tmp_path, drop=f'/api/watch/{GAME}/summary?through=4&top=5')
    with pytest.raises(SystemExit, match='incomplete: 1 of 10'):
        load_snapshot(snap)


@pytest.mark.parametrize('decisions', ['5', 5.0, None, True, 0, -1])
def test_all_decisions_bundle_with_bad_count_does_not_start(tmp_path, decisions):
    _, snap, _ = write_all_bundle(tmp_path, decisions=decisions)
    with pytest.raises(SystemExit, match='no valid decision count'):
        load_snapshot(snap)


@pytest.mark.parametrize('decisions', [4, 6])
def test_all_decisions_count_must_match_saved_timeline(tmp_path, decisions):
    _, snap, _ = write_all_bundle(tmp_path, decisions=decisions)  # 4 would pass the file check yet claim the wrong total
    with pytest.raises(SystemExit, match='saved timeline does not'):
        load_snapshot(snap)


def test_all_decisions_top_mismatch_does_not_start(tmp_path):
    _, snap, _ = write_all_bundle(tmp_path, top=4)
    with pytest.raises(SystemExit, match='incomplete: 5 of 10'):
        load_snapshot(snap)


def test_setup_answer_file_is_distinct_and_checked():
    names = {snapshot_file(p) for p in [f'/api/watch/{GAME}', *all_decision_paths(TIMELINE)]}
    assert snapshot_file(setup_path(GAME)) not in names
    assert check_setup(json.dumps({'game_pk': GAME, 'pitches': []}).encode(), GAME)['pitches'] == []
    for bad in [{'game_pk': GAME + 1, 'pitches': []}, {'game_pk': GAME}, {'game_pk': GAME, 'pitches': {}}]:
        with pytest.raises(SystemExit):
            check_setup(json.dumps(bad).encode(), GAME)


def test_all_decisions_bundle_serves_saved_setup_answer(tmp_path):
    body = json.dumps({'game_pk': GAME, 'pitches': [{'at_bat_number': 1, 'pitch_number': 1, 'status': 'estimated',
                                                      'plate_x_feet': -0.25}]}).encode()
    dist, snap, answers = write_all_bundle(tmp_path)
    (snap / snapshot_file(setup_path(GAME))).write_bytes(body)
    manifest = json.loads((snap / 'manifest.json').read_text())
    manifest['files'].append({'path': setup_path(GAME), 'file': snapshot_file(setup_path(GAME)), 'bytes': len(body),
                              'sha256': hashlib.sha256(body).hexdigest()})
    (snap / 'manifest.json').write_text(json.dumps(manifest))
    manifest, loaded = load_snapshot(snap)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(dist, manifest, loaded))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        assert get(base + setup_path(GAME))[::2] == (200, body)
        assert get(base + setup_path(GAME + 1))[0] == 404  # another game's setup is not made up
    finally:
        server.shutdown()


# capture_snapshot.main() against a fake 8771 (no network). The setup path is written out literally: it must be the
# exact request of WatchAlong.tsx (`get(`/watch/${gamePk}/setup`)`, get() prefixes /api), not whatever setup_path says.
SETUP_BODY = json.dumps({'game_pk': GAME, 'badge': 'b', 'pitches': [
    {'at_bat_number': 1, 'pitch_number': 1, 'status': 'estimated', 'plate_x_feet': -0.25}]}).encode()


def run_capture(tmp_path, monkeypatch, *flags, setup=(200, SETUP_BODY)):
    import capture_snapshot
    site = {f'/api/watch/{GAME}': (200, json.dumps(TIMELINE).encode()),
            '/api/watch/games': (200, json.dumps({'games': [{'game_pk': GAME}, {'game_pk': GAME + 1}]}).encode()),
            f'/api/watch/{GAME}/setup': setup}
    for i, path in enumerate(all_decision_paths(TIMELINE)):
        site[path] = (200, json.dumps({'n': i}).encode())
    asked = []

    def fake_fetch(url, timeout=15):
        path = url.removeprefix('http://fake')
        asked.append(path)
        return site.get(path, (404, b'{"detail": "none"}'))

    monkeypatch.setattr(capture_snapshot, 'fetch', fake_fetch)
    monkeypatch.setattr(capture_snapshot.time, 'sleep', lambda s: None)
    tmp_path.mkdir(exist_ok=True)
    scenes = tmp_path / 'scenes.json'
    scenes.write_text(json.dumps({'gamePk': GAME, 'scenes': [{'pitchId': 'x', 'atBat': 1, 'pitchNumber': 3}]}))
    out = tmp_path / 'snap'
    monkeypatch.setattr('sys.argv', ['capture_snapshot.py', '--base', 'http://fake', '--scenes', str(scenes),
                                     '--out', str(out), *flags])
    capture_snapshot.main()
    return out, asked


def test_capture_with_setup_saves_the_page_request_byte_identically(tmp_path, monkeypatch):
    out, asked = run_capture(tmp_path, monkeypatch, '--all-decisions', '--with-setup')
    assert asked.count(f'/api/watch/{GAME}/setup') == 1
    manifest, loaded = load_snapshot(out)
    assert manifest['setup_rows'] == 1
    assert loaded[f'/api/watch/{GAME}/setup'] == SETUP_BODY


def test_capture_without_setup_flag_never_asks_for_it(tmp_path, monkeypatch):
    out, asked = run_capture(tmp_path, monkeypatch, '--all-decisions')
    manifest, loaded = load_snapshot(out)
    assert manifest['setup_rows'] is None
    assert f'/api/watch/{GAME}/setup' not in asked and f'/api/watch/{GAME}/setup' not in loaded


@pytest.mark.parametrize('setup', [(404, b'{"detail": "none"}'), (200, json.dumps({'game_pk': GAME + 1, 'pitches': []}).encode())])
def test_capture_with_setup_stops_without_a_manifest(tmp_path, monkeypatch, setup):
    with pytest.raises(SystemExit):
        run_capture(tmp_path, monkeypatch, '--all-decisions', '--with-setup', setup=setup)
    assert not (tmp_path / 'snap' / 'manifest.json').exists()


def test_capture_cap_counts_the_setup_answer_before_any_reveal(tmp_path, monkeypatch):
    import capture_snapshot
    monkeypatch.setattr(capture_snapshot, 'MAX_REQUESTS_ALL', 1 + len(all_decision_paths(TIMELINE)))
    run_capture(tmp_path / 'a', monkeypatch, '--all-decisions')  # timeline + every pitch: exactly at the cap
    with pytest.raises(SystemExit, match='exceed cap'):
        run_capture(tmp_path / 'b', monkeypatch, '--all-decisions', '--with-setup')
