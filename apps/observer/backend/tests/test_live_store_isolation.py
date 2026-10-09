"""Synthetic-only recovery checks; no game recordings or serving DBs."""
import gzip
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed


def feed(event):
    return {'metaData': {'eventWall': event, 'timeStamp': f'20260101_000{event}'}, 'liveData': {}}


def test_recovery_flushes_memory_before_accepting_late_feed(tmp_path):
    store = LiveSnapshotStore(tmp_path/'synthetic.sqlite3')
    live = PersistentLiveFeed(None, store=store, wall=lambda: 130)
    buffer = live._buffer(1)
    live._accept(1, feed(100), 100, buffer)
    ingest = store.ingest
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('private /path')
    store.ingest = fail
    live._accept(1, feed(120), 120, buffer)
    assert not live.sync_extra(1, None)['store_ok']
    live._accept(1, feed(110), 125, buffer)
    assert buffer[-1]['feed'] == feed(120)
    store.ingest = ingest
    live._accept(1, feed(110), 130, buffer)
    assert live.sync_extra(1, None)['store_ok']
    restored = store.restore(1, 130, 120)
    assert [(x['feed'], x['fetched']) for x in restored] == [(feed(100), 100), (feed(120), 120)]
    assert store.read_after(1, until_event=110)['cursor'] == 1
    backup = store.backup(tmp_path/'backup.sqlite3')
    assert LiveSnapshotStore(backup).restore(1, 130, 120) == restored


def pitch(details='ball'):
    return {'isPitch': True, 'playId': 'p1', 'pitchNumber': 1, 'startTime': '2026-01-01T00:00:00Z', 'details': details}


def test_malformed_plays_beside_valid_pitch_reach_ledger(tmp_path):
    store = LiveSnapshotStore(tmp_path/'synthetic.sqlite3')
    live = PersistentLiveFeed(None, store=store, wall=lambda: 1767225700)
    buffer = live._buffer(1)
    events = [pitch(), 'junk', {'isPitch': True, 'startTime': 'not-a-time'}]
    malformed = {'metaData': {'timeStamp': '20260101_000010'},
                 'liveData': {'plays': {'currentPlay': 'bad', 'allPlays': [{'about': {'atBatIndex': 0}, 'playEvents': events}, None]}}}
    live._accept(1, malformed, 1767225700, buffer)
    assert buffer[-1]['feed'] == malformed and live.store_error is None
    rows = store.pitch_events(1)['events']
    assert [(r['provider_event_id'], r['outcome'], r['reason']) for r in rows] == [
        ('p1', 'new', None), (None, 'quarantine', 'malformed_event'), (None, 'quarantine', 'incomplete_identity')]


def test_older_provider_stamp_with_newer_event_wall_is_never_a_correction(tmp_path):
    store = LiveSnapshotStore(tmp_path/'synthetic.sqlite3')
    live = PersistentLiveFeed(None, store=store, wall=lambda: 300)
    buffer = live._buffer(1)
    def snap(event, stamp, details):
        play = {'about': {'atBatIndex': 0}, 'playEvents': [pitch(details)]}
        return {'metaData': {'eventWall': event, 'timeStamp': stamp}, 'liveData': {'plays': {'allPlays': [play]}}}
    first, late = snap(100, '20260101_000100', 'ball'), snap(200, '20260101_000050', 'strike')
    live._accept(1, first, 100, buffer)
    live._accept(1, late, 200, buffer)
    stale_late = snap(150, '20260101_000040', 'foul')
    live._accept(1, stale_late, 210, buffer)
    assert buffer[-1]['feed'] == late and live.stale[1] == 1
    rows = store.pitch_events(1)['events']
    # the late slot is selected despite its older stamp (newer event); its changed pitch is still refused, and the
    # stale arrival's pitch gets a receipt whose store_seq names the selected version (2), not the arrival
    assert [(r['outcome'], r['reason'], r['revision'], r['first_received'], r['received'], r['store_seq']) for r in rows] == [
        ('new', None, 1, 100, 100, 1), ('quarantine', 'stale_revision', None, None, 200, 2),
        ('quarantine', 'stale_revision', None, None, 210, 2)]
    assert [x['feed'] for x in store.restore(1, 300, 300)] == [first, late]


@pytest.mark.parametrize('body', [b'broken gzip', gzip.compress(b'broken json'), b'\x1f\x8b'])
def test_corrupt_restore_degrades_once_and_keeps_error(tmp_path, body):
    store = LiveSnapshotStore(tmp_path/'synthetic.sqlite3')
    with sqlite3.connect(store.path) as db:
        db.execute('INSERT INTO snapshots(game_pk,event_wall,version,sha,first_fetched,fetched,source,body) '
                   'VALUES(1,100,1,?,100,100,?,?)', ('fake', 'synthetic', body))
    live = PersistentLiveFeed(None, store=store, wall=lambda: 130)
    buffer = live._buffer(1)
    assert buffer == [] and live.store_error.startswith('restore:')
    live._accept(1, feed(120), 130, buffer)
    assert live._buffer(1) is buffer and buffer[-1]['feed'] == feed(120)
    assert not live.sync_extra(1, None)['store_ok']


def test_stop_refuses_foreign_live_pid(tmp_path):
    script = Path(__file__).resolve().parents[2]/'deploy/staging/live_store_staging.sh'
    with subprocess.Popen(['sleep', '20']) as foreign:
        try:
            (tmp_path/'server.pid').write_text(str(foreign.pid))
            result = subprocess.run(['sh', str(script), 'stop'], env=dict(os.environ, PZ_STAGING_DIR=str(tmp_path)), capture_output=True, text=True)
            assert result.returncode == 1 and 'unowned' in result.stderr
            assert foreign.poll() is None
        finally:
            foreign.terminate()


def test_synthetic_api_delay_restore_and_private_health(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from observer_app.live_armb import LiveArmB
    from observer_app.main import create_app
    def game(event):
        value = feed(event)
        value['gameData'] = {'game': {'pk': 1, 'type': 'R'}, 'datetime': {'officialDate': '2025-01-01'}, 'status': {'abstractGameState': 'Live'}, 'teams': {'away': {'name': 'Synthetic A'}, 'home': {'name': 'Synthetic B'}}}
        value['liveData'] = {'linescore': {'outs': 3}, 'plays': {'allPlays': [], 'currentPlay': {}}}
        return value
    now = [100]
    store = LiveSnapshotStore(tmp_path/'private.sqlite3')
    live = PersistentLiveFeed(lambda *args, **kwargs: game(now[0]), store=store, clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None)
    live.snapshot(1, 0)
    now[0] = 120
    live.snapshot(1, 0)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(store.path))
    def down(*args, **kwargs):
        raise OSError('offline')
    restored = PersistentLiveFeed(down, store=LiveSnapshotStore(store.path), clock=lambda: 130, wall=lambda: 130, sleep=lambda _: None)
    with TestClient(create_app(object(), live_feed=restored, live_policy=LiveArmB(lambda: None))) as client:
        response = client.get('/api/live/1/state?delay=30')
        assert response.status_code == 200
        assert response.json()['sync']['restored'] is True
        assert response.json()['sync']['feed_ok'] is False
        assert 'private.sqlite3' not in client.get('/api/health').text
        assert [x['event_wall'] for x in client.get('/api/live/1/log?delay=30').json()['items']] == [100]


def test_uncommitted_process_crash_is_rolled_back(tmp_path):
    store = LiveSnapshotStore(tmp_path/'crash.sqlite3')
    store.ingest(1, feed(100), fetched=100, source='synthetic')
    code = '''import os, sqlite3, sys
c = sqlite3.connect(sys.argv[1], isolation_level=None)
c.execute('BEGIN IMMEDIATE')
c.execute("INSERT INTO snapshots(game_pk,event_wall,version,sha,first_fetched,fetched,source,body) VALUES(1,120,1,'fake',120,120,'synthetic',x'00')")
os._exit(9)
'''
    assert subprocess.run([sys.executable, '-c', code, str(store.path)]).returncode == 9
    reopened = LiveSnapshotStore(store.path)
    assert reopened.status()['quick_check'] == 'ok'
    assert reopened.read_after(1, until_event=200)['cursor'] == 1


def test_unopenable_store_health_redacts_private_path(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from observer_app.live_armb import LiveArmB
    from observer_app.main import create_app
    (tmp_path/'private-file').write_text('not a directory')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp_path/'private-file'/'secret.sqlite3'))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(tmp_path/'empty-replay'))
    with TestClient(create_app(object(), live_policy=LiveArmB(lambda: None))) as client:
        # Empty synthetic upstream fails independently; open failure must still initialize memory fallback.
        client.get('/api/live/1/state?delay=0')
        health = client.get('/api/health')
        assert health.json()['live_store']['error'] == 'open: FileExistsError'
        assert health.json()['live_store']['started'] is False
        assert str(tmp_path) not in health.text


@pytest.mark.parametrize('first_persisted', [True, False])
def test_outage_correction_keeps_first_and_version_arrival(tmp_path, first_persisted):
    store = LiveSnapshotStore(tmp_path/'arrival.sqlite3')
    live = PersistentLiveFeed(None, store=store, wall=lambda: 130)
    buffer = live._buffer(1)
    ingest = store.ingest
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('outage')
    if not first_persisted:
        store.ingest = fail
    live._accept(1, feed(100), 100, buffer)
    store.ingest = fail
    corrected = feed(100)
    corrected['metaData']['timeStamp'] = '20260101_000101'
    live._accept(1, corrected, 120, buffer)
    live._accept(1, corrected, 125, buffer)
    store.ingest = ingest
    live._accept(1, corrected, 130, buffer)
    row = store.read_after(1, until_event=200)['items'][-1]
    assert (row['event_wall'], row['first_fetched'], row['fetched']) == (100, 100, 120)
    assert row['source'] == 'memory_recovery'
    assert live.sync_extra(1, None)['store_ok']


@pytest.mark.parametrize('healthy', [True, False])
def test_staging_owns_process_before_health_and_cleans_up(tmp_path, healthy):
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    assert not 8760 <= port <= 8779
    fake = tmp_path/'synthetic-python'
    fake.write_text(f'''#!{sys.executable}
import http.server, os, sys, time
from pathlib import Path
Path(os.environ['SYNTHETIC_PID']).write_text(str(os.getpid()))
if os.environ['SYNTHETIC_HEALTH'] == 'yes':
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{{}}')
        def log_message(self, *args): pass
    http.server.HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])), Handler).serve_forever()
else:
    time.sleep(60)
''')
    fake.chmod(0o755)
    directory = tmp_path/'stage'
    directory.mkdir()
    (directory/'server.identity').write_text('stale prior launch identity')
    env = dict(os.environ, PZ_STAGING_DIR=str(directory), PITCHEEZY_STAGING_PORT=str(port),
               PITCHEEZY_OBSERVER_PYTHON=str(fake), SYNTHETIC_PID=str(tmp_path/'child.pid'),
               SYNTHETIC_HEALTH='yes' if healthy else 'no')
    script = Path(__file__).resolve().parents[2]/'deploy/staging/live_store_staging.sh'
    try:
        launched = subprocess.run(['sh', str(script), 'start'], env=env, capture_output=True, text=True, timeout=25)
        assert launched.returncode == (0 if healthy else 1), launched.stderr
        if healthy:
            assert 'stale' not in (directory/'server.identity').read_text()
            stopped = subprocess.run(['sh', str(script), 'stop'], env=env, capture_output=True, text=True, timeout=15)
            assert stopped.returncode == 0, stopped.stderr
        else:
            assert 'no health' in launched.stderr and 'unowned' not in launched.stderr
        assert not (directory/'server.pid').exists()
        assert not (directory/'server.identity').exists()
        child = int((tmp_path/'child.pid').read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
    finally:
        pidfile = tmp_path/'child.pid'
        if pidfile.exists():
            child = int(pidfile.read_text())
            command = subprocess.run(['ps', '-p', str(child), '-o', 'command='], capture_output=True, text=True).stdout
            if str(fake) in command:
                os.kill(child, 15)
