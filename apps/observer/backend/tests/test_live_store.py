"""Opt-in SQLite live store: restart resumes the delay buffer, corrections/duplicates/late feeds, atomic commits."""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_armb import LiveArmB
from observer_app.live_feed import LiveFeed
from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed
from observer_app.main import create_app
from live_fixture import listed

FIXTURES = Path(__file__).parent/'fixtures/live'
GAME = 849845


def at(name, wall):
    snapshot = listed(json.loads((FIXTURES/name).read_text()))  # as a real feed lists its current play
    snapshot['metaData']['eventWall'] = wall
    return snapshot


BEFORE, PITCH = at('849845_20260929_195940.json', 990.0), at('849845_20260929_203124.json', 1012.0)


class Upstream:
    """Serves ``feeds[i]`` for poll i (the last one after that); ``down`` makes every poll fail."""
    def __init__(self, *feeds):
        self.feeds, self.calls, self.down = list(feeds), 0, False

    def __call__(self, url, timeout):
        if self.down:
            raise OSError('feed down')
        self.calls += 1
        return self.feeds[min(self.calls, len(self.feeds))-1]


def app_for(live):
    return create_app(object(), live_feed=live, live_policy=LiveArmB(lambda: None))


def visible(body):
    return {key: body.get(key) for key in ('status', 'game', 'situation', 'pitcher', 'batter', 'pitch_sequence')}


def test_ingest_dedupes_versions_corrections_and_drops_late_older_feeds(tmp_path):
    store = LiveSnapshotStore(tmp_path/'live.sqlite3')
    assert store.ingest(GAME, BEFORE, fetched=1000.0, source='t')[0] == 'new'
    assert store.ingest(GAME, copy.deepcopy(BEFORE), fetched=1005.0, source='t')[0] == 'duplicate'
    fixed = copy.deepcopy(BEFORE)
    fixed['liveData']['linescore']['outs'] = 2  # same game event, content corrected upstream
    outcome, entry = store.ingest(GAME, fixed, fetched=1008.0, source='t')
    assert outcome == 'correction' and (entry['event'], entry['fetched']) == (990.0, 1000.0)  # first-seen times kept
    assert store.ingest(GAME, PITCH, fetched=1020.0, source='t')[0] == 'new'
    assert store.ingest(GAME, BEFORE, fetched=1030.0, source='t') == ('stale', None)  # older feed answering late
    log = store.read_after(GAME, 0, until_event=2000.0)
    assert [(i['event_wall'], i['version'], i['fetched']) for i in log['items']] == [(990.0, 1, 1000.0), (990.0, 2, 1008.0), (1012.0, 1, 1020.0)]
    restored = store.restore(GAME, 1030.0, 120)
    assert [r['feed'] for r in restored] == [fixed, PITCH]  # the current version of each slot
    page = store.read_after(GAME, 0, until_event=1000.0, limit=1)
    assert [i['version'] for i in page['items']] == [1] and store.read_after(GAME, page['cursor'], until_event=1000.0)['cursor'] == 2
    assert store.read_after(GAME, 2, until_event=1000.0) == {'items': [], 'cursor': 2}  # the pitch is not past this delay yet


def test_restart_resumes_each_viewers_delay_without_showing_the_future(tmp_path):
    now = [1000.0]
    clock = dict(clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None)
    db = tmp_path/'live.sqlite3'
    reference = LiveFeed(Upstream(BEFORE, PITCH), delay_s=30, **clock)  # the old read path, never restarted
    first = PersistentLiveFeed(Upstream(BEFORE, PITCH), store=LiveSnapshotStore(db), delay_s=30, **clock)
    with TestClient(app_for(first)) as client:
        for t in (1000.0, 1020.0):
            now[0] = t
            client.get(f'/api/live/{GAME}/state?delay=5')
            reference.snapshot(GAME, 5)
        before_restart = client.get(f'/api/live/{GAME}/state').json()
    assert before_restart['situation'] is not None and before_restart['sync']['restored'] is False

    # Process restart while MLB is unreachable: the new process only has the database.
    upstream = Upstream(BEFORE)
    upstream.down = True
    second = PersistentLiveFeed(upstream, store=LiveSnapshotStore(db), delay_s=30, **clock)
    now[0] = 1025.0
    with TestClient(app_for(second)) as client:
        body = client.get(f'/api/live/{GAME}/state').json()
        assert visible(body) == visible(before_restart)  # same state a 30 s viewer saw before the restart
        assert visible(body)['pitch_sequence'] != visible(client.get(f'/api/live/{GAME}/state?delay=0').json())['pitch_sequence']
        assert {key: body['sync'][key] for key in ('restored', 'feed_ok', 'store_ok')} == {'restored': True, 'feed_ok': False, 'store_ok': True}  # honest: old data, feed down
        assert client.get(f'/api/live/{GAME}/log?after=0').json()['items'] == [
            item for item in client.get(f'/api/live/{GAME}/log?after=0&delay=0').json()['items'] if item['event_wall'] <= 995.0]
        now[0], upstream.down = 1042.0, False  # MLB is back, but answers with an old cached snapshot
        served = client.get(f'/api/live/{GAME}/state').json()
        assert served['sync']['stale_dropped'] == 1 and served['sync']['feed_ok'] is True
        assert served['pitch_sequence'] == parse(reference.snapshot(GAME, 30)[0]['feed'])['pitch_sequence']  # 1012 + 30: no rewind
    rewinding = LiveFeed(Upstream(PITCH, BEFORE), delay_s=0, **clock)  # the in-memory path has no such guard
    now[0] = 1100.0
    rewinding.snapshot(GAME)
    now[0] = 1111.0
    assert rewinding.snapshot(GAME)[0]['feed'] is BEFORE


def parse(feed):
    from observer_app.live_feed import parse_state
    return parse_state(feed, None)


def test_a_write_killed_mid_transaction_leaves_nothing_and_the_cursor_waits_for_commits(tmp_path):
    db = tmp_path/'live.sqlite3'
    store = LiveSnapshotStore(db)
    store.ingest(GAME, BEFORE, fetched=1000.0, source='t')
    crash = ('import os, sqlite3, sys\n'
             'db = sqlite3.connect(sys.argv[1], isolation_level=None); db.execute("BEGIN IMMEDIATE")\n'
             'db.execute("INSERT INTO snapshots(game_pk,event_wall,version,feed_ts,sha,first_fetched,fetched,source,body) '
             'VALUES (849845,1012,1,\'x\',\'x\',1020,1020,\'crash\',x\'00\')")\n'
             'os._exit(9)\n')
    assert subprocess.run([sys.executable, '-c', crash, str(db)]).returncode == 9
    reopened = LiveSnapshotStore(db)
    assert reopened.status()['quick_check'] == 'ok' and reopened.read_after(GAME, 0, until_event=2000.0)['cursor'] == 1
    other = sqlite3.connect(db, isolation_level=None)  # a writer that has not committed yet
    other.execute('BEGIN IMMEDIATE')
    other.execute("INSERT INTO snapshots(game_pk,event_wall,version,feed_ts,sha,first_fetched,fetched,source,body) "
                  "VALUES (849845,1012,1,'y','y',1020,1020,'t',x'00')")
    assert reopened.read_after(GAME, 1, until_event=2000.0) == {'items': [], 'cursor': 1}
    other.execute('COMMIT')
    assert reopened.read_after(GAME, 1, until_event=2000.0)['cursor'] == 2
    with sqlite3.connect(db) as raw:
        try:
            raw.execute('UPDATE snapshots SET source="edited"')
            raise AssertionError('append-only trigger missing')
        except sqlite3.DatabaseError as exc:
            assert 'append-only' in str(exc)


def test_a_failing_store_keeps_serving_from_memory_and_says_so(tmp_path):
    class Broken(LiveSnapshotStore):
        def ingest(self, *args, **kwargs):
            raise sqlite3.OperationalError('disk I/O error')

    now = [1000.0]
    live = PersistentLiveFeed(Upstream(BEFORE), store=Broken(tmp_path/'live.sqlite3'), delay_s=0,
                              clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None)
    with TestClient(app_for(live)) as client:
        body = client.get(f'/api/live/{GAME}/state').json()
    assert body['status'] == 'ready' and body['sync']['store_ok'] is False and 'OperationalError' in live.store_error


def test_an_unopenable_store_path_still_serves_live_from_memory(tmp_path, monkeypatch):
    (tmp_path/'file').write_text('')  # a file where the db's directory should be (like a missing/readonly volume)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(FIXTURES))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp_path/'file'/'live.sqlite3'))
    with TestClient(create_app(object(), live_policy=LiveArmB(lambda: None))) as client:
        assert client.get(f'/api/live/{GAME}/state?delay=0').json()['status'] == 'ready'
        store = client.get('/api/health').json()['live_store']
    assert store['started'] is False and store['error'].startswith('open: ')
