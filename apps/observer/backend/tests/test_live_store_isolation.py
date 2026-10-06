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
    return {'metaData': {'eventWall': event, 'timeStamp': str(event)}, 'liveData': {}}


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
