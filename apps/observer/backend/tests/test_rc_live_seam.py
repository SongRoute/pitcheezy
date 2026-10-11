"""RC seam: receipt = response completely received (request start kept beside it); polled snapshots reach the
append-only store; a pre-pitch recommendation is in the append-only ledger before it is displayed.
Uses the registered 849845 SERVICE_REGRESSION_EXPOSED_DEV fixture and a fake policy; no network, no model."""
import copy
import io
import json
from pathlib import Path
import sqlite3
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import live_feed as live_feed_module, live_ledger
from observer_app.live_armb import LiveArmB
from observer_app.live_feed import LIVE_CONFIG, LiveFeed, ReplayTransport, http_json, replay_config
from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed
from observer_app.main import create_app
from test_live_feed import FakePolicy, wait_computed

FIXTURES = Path(__file__).parent/'fixtures/live'
GAME = 849845


def at(name, wall):
    snapshot = json.loads((FIXTURES/name).read_text())
    snapshot['metaData']['eventWall'] = wall
    return snapshot


class Slow:
    """An upstream whose response takes ``took`` s to arrive completely (it moves the injected wall clock)."""
    def __init__(self, now, feed, took, parse=0.0):
        self.now, self.feed, self.took, self.parse = now, feed, took, parse


class SlowPlain(Slow):
    def __call__(self, url, timeout):
        self.now[0] += self.took
        return self.feed


class SlowReporting(Slow):
    def __call__(self, url, timeout, received=None):
        self.now[0] += self.took
        received()
        self.now[0] += self.parse  # parsing after the last byte is not part of the receipt
        return self.feed


def thrown(wall, speed=90.0, ts='20261009_000100'):
    """A minimal synthetic feed with one identified pitch (the 849845 fixture is trimmed: no ``allPlays``)."""
    return {'metaData': {'eventWall': wall, 'timeStamp': ts}, 'gameData': {'game': {'pk': GAME}}, 'liveData': {'plays': {'allPlays': [
        {'about': {'atBatIndex': 0, 'inning': 1}, 'playEvents': [
            {'isPitch': True, 'playId': 'p1', 'pitchNumber': 1, 'startTime': '2026-10-09T00:00:01Z',
             'details': {'call': {'code': 'B'}}, 'pitchData': {'startSpeed': speed}}]}]}}}


def live(transport, now, **options):
    return LiveFeed(transport, delay_s=0, clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None, **options)


def test_receipt_is_stamped_after_the_response_arrived_and_the_request_start_is_kept():
    now = [1000.0]
    entry, _ = live(SlowPlain(now, at('849845_20260929_195940.json', 990.0), 2.5), now).snapshot(GAME)
    assert (entry['requested'], entry['fetched']) == (1000.0, 1002.5)  # not the wall time before the request
    now = [1000.0]
    entry, _ = live(SlowReporting(now, at('849845_20260929_195940.json', 990.0), 2.5, parse=0.4), now).snapshot(GAME)
    assert (entry['requested'], entry['fetched']) == (1000.0, 1002.5) and now[0] == 1002.9


def test_an_event_claimed_after_the_request_but_before_the_receipt_is_not_clamped_to_the_request():
    now = [1000.0]  # the pitch happened at 1001, while the request was in flight
    entry, _ = live(SlowPlain(now, at('849845_20260929_195940.json', 1001.0), 2.0), now).snapshot(GAME)
    assert (entry['event'], entry['fetched']) == (1001.0, 1002.0)


def test_http_json_reports_receipt_after_the_body_is_read_and_before_parsing(monkeypatch):
    order = []

    class Response(io.BytesIO):
        def read(self, *args):
            order.append('read')
            return super().read(*args)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(live_feed_module.urllib.request, 'urlopen', lambda request, timeout: Response(b'{"a": 1}'))
    monkeypatch.setattr(live_feed_module.json, 'loads', lambda body: order.append('parse') or {'a': 1})
    assert http_json('http://example.invalid/x', 1, received=lambda: order.append('received')) == {'a': 1}
    assert order == ['read', 'received', 'parse']
    assert http_json('http://example.invalid/x', 1) == {'a': 1}  # the callback is optional


def test_a_failed_poll_records_the_outage_at_its_failure_time():
    now, snapshot = [1000.0], at('849845_20260929_195940.json', 990.0)

    class Flaky(SlowPlain):
        down = False

        def __call__(self, url, timeout):
            if self.down:
                self.now[0] += 3.0
                raise OSError('down')
            return super().__call__(url, timeout)

    transport = Flaky(now, snapshot, 1.0)
    feed = live(transport, now)
    feed.snapshot(GAME)
    transport.down, now[0] = True, 1020.0
    entry, _ = feed.snapshot(GAME)
    assert entry['fetched'] == 1001.0 and feed.failing[GAME] == 1023.0


def persistent(tmp_path, transport, now, **options):
    return PersistentLiveFeed(transport, store=LiveSnapshotStore(tmp_path/'live.sqlite3'), delay_s=0,
                              clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None, **options)


def rows(tmp_path, sql):
    with sqlite3.connect(tmp_path/'live.sqlite3') as db:
        return db.execute(sql).fetchall()


def test_store_keeps_both_times_per_version_and_pitch_receipts_use_the_response_time(tmp_path):
    now, before, fixed = [1000.0], thrown(990.0), thrown(990.0, 91.5, '20261009_000200')
    transport = SlowReporting(now, before, 2.0)
    feed = persistent(tmp_path, transport, now)
    entry, _ = feed.snapshot(GAME)
    assert (entry['requested'], entry['fetched'], entry['version_requested']) == (1000.0, 1002.0, 1000.0)
    transport.feed, now[0] = fixed, 1020.0
    entry, _ = feed.snapshot(GAME)  # a correction keeps the slot's first receipt and adds its own pair
    assert (entry['fetched'], entry['version_fetched'], entry['version_requested']) == (1002.0, 1022.0, 1020.0)
    assert rows(tmp_path, 'SELECT version, requested, fetched, source FROM snapshots ORDER BY seq') == [
        (1, 1000.0, 1002.0, 'live_poll'), (2, 1020.0, 1022.0, 'live_poll')]
    assert rows(tmp_path, 'SELECT outcome, revision, first_received, received, receipt_clock FROM pitch_events ORDER BY id') == [
        ('new', 1, 1002.0, 1002.0, 'observer_local'), ('correction', 2, 1002.0, 1022.0, 'observer_local')]
    restored = persistent(tmp_path, transport, now)._buffer(GAME)[-1]
    assert (restored['version_requested'], restored['version_fetched'], restored['requested']) == (1020.0, 1022.0, None)


def test_a_store_made_before_the_receipt_split_is_migrated_in_place(tmp_path):
    path = tmp_path/'live.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE snapshots (seq INTEGER PRIMARY KEY AUTOINCREMENT, game_pk INTEGER NOT NULL, '
                   'event_wall REAL NOT NULL, version INTEGER NOT NULL, feed_ts TEXT, sha TEXT NOT NULL, '
                   'first_fetched REAL NOT NULL, fetched REAL NOT NULL, source TEXT NOT NULL, body BLOB NOT NULL, '
                   'UNIQUE(game_pk, event_wall, version))')
    store = LiveSnapshotStore(path)
    outcome, entry = store.ingest(GAME, at('849845_20260929_195940.json', 990.0), fetched=1000.0, source='t')
    assert outcome == 'new' and entry['version_requested'] is None
    assert LiveSnapshotStore(path).status()['games'][0]['versions'] == 1  # reopening does not migrate twice


def test_a_replayed_recording_is_never_recorded_on_the_observer_clock(tmp_path):
    now = [1000.0]
    feed = persistent(tmp_path, lambda url, timeout: thrown(990.0), now, source='replay_poll')
    feed.snapshot(GAME)
    assert set(rows(tmp_path, 'SELECT source FROM snapshots')) == {('replay_poll',)}
    assert set(rows(tmp_path, 'SELECT receipt_clock FROM pitch_events')) == {('unverified',)}


DISPLAYED = {'status': 'ready', 'reason': None, 'key': f'{GAME}:1:1', 'policy_identity': 'abc',
             'recommendation': {'candidates': [{'pitch_type': 'SL'}]}}


def committed(tmp_path):
    store = LiveSnapshotStore(tmp_path/'live.sqlite3')
    _, entry = store.ingest(GAME, at('849845_20260929_195940.json', 990.0), fetched=1000.0, source='live_poll')
    return store, entry


def record(store, entry, displayed=DISPLAYED, **over):
    return store.append_recommendation(**({'game_pk': GAME, 'state_sha': 's1', 'policy_identity': 'abc',
                                           'pitch_key': displayed['key'], 'snapshot': entry, 'displayed': displayed,
                                           'decided': 1001.0, 'feed_ok': True} | over))


def test_recommendation_ledger_is_append_only_and_never_flips_a_stored_decision(tmp_path):
    store, entry = committed(tmp_path)
    outcome, first = record(store, entry)
    assert outcome == 'first' and first['displayed'] == DISPLAYED and first['decision_clock'] == 'observer_local'
    assert first['snapshot'] == {'store_seq': entry['seq'], 'slot_version': 1, 'feed_sha256': entry['sha'],
                                 'source': 'live_poll', 'received': 1000.0, 'provider_order': 'ordered'}
    assert first['target'] == {'status': 'UNKNOWN', 'reason': 'next_pitch_not_observed_at_decision'}
    assert record(store, entry, decided=1005.0) == ('duplicate', first)  # another viewer: nothing written
    flipped = DISPLAYED | {'recommendation': {'candidates': [{'pitch_type': 'FF'}]}}
    assert record(store, entry, flipped, decided=1006.0) == ('conflict', first)  # the stored one stays the shown one
    assert record(store, entry, flipped, decided=1007.0) == ('conflict', first)
    ledger = store.served_recommendations(GAME)
    assert [(r['outcome'], r['displayed']['recommendation']['candidates'][0]['pitch_type'], r['decided'])
            for r in ledger] == [('first', 'SL', 1001.0), ('conflict', 'FF', 1006.0)]
    # a provider correction of the inputs is another decision for the same pending pitch, not a rewrite
    assert record(store, entry, flipped, state_sha='s2', decided=1008.0)[0] == 'first'
    assert [r['pitch_key'] for r in store.served_recommendations(GAME)] == [f'{GAME}:1:1']*3
    with sqlite3.connect(tmp_path/'live.sqlite3') as db:
        for statement in ("UPDATE served_recommendations SET outcome='first'", 'DELETE FROM served_recommendations'):
            with pytest.raises(sqlite3.DatabaseError, match='append-only served recommendation'):
                db.execute(statement)


def test_recommendation_ledger_refuses_a_snapshot_that_is_not_the_committed_one(tmp_path):
    store, entry = committed(tmp_path)
    for bad, decided in ((entry | {'seq': 99}, 1001.0), (entry | {'sha': '0'*64}, 1001.0), (entry | {'version': 2}, 1001.0),
                         (entry, 999.0)):  # the last: "decided" before the snapshot was even received
        with pytest.raises(ValueError):
            record(store, bad, decided=decided)
    assert store.served_recommendations(GAME) == []
    _, replayed = store.ingest(GAME+1, at('849845_20260929_195940.json', 990.0), fetched=1000.0, source='replay_poll')
    assert record(store, replayed, game_pk=GAME+1)[1]['decision_clock'] == 'unverified'


def serve(tmp_path, now, armb, **options):
    feed = persistent(tmp_path, ReplayTransport(FIXTURES), now, **options)
    return feed, create_app(object(), live_feed=feed, live_policy=armb, watch_dir=tmp_path, feed_dir=tmp_path,
                            demo_dir=tmp_path/'demo', intent_dir=tmp_path)


def ready(client, armb, path=f'/api/live/{GAME}/state'):
    client.get(path)
    wait_computed(armb, 1)
    return client.get(path).json()


def test_live_route_writes_the_recommendation_to_the_ledger_before_returning_it(tmp_path):
    now, armb = [1000.0], LiveArmB(lambda: FakePolicy())
    feed, app = serve(tmp_path, now, armb)
    with TestClient(app) as client:
        now[0] = 1003.0
        shown = ready(client, armb)['recommendation']
        stored = feed.store.served_recommendations(GAME)
        assert len(stored) == 1 and stored[0]['outcome'] == 'first' and stored[0]['decided'] == 1003.0
        outcome = shown['ledger'].pop('outcome')  # 'first' if this very request wrote it, else an earlier one did
        assert outcome in ('first', 'duplicate') and shown['ledger'] == {
            'status': 'recorded', 'decision_id': stored[0]['decision_id'], 'decided': 1003.0, 'decision_clock': 'observer_local'}
        assert {key: shown[key] for key in live_ledger.DISPLAYED} == stored[0]['displayed']
        assert stored[0]['snapshot']['received'] <= stored[0]['decided'] and stored[0]['feed_ok'] is True
        for _ in range(3):  # more viewers and polls of the same state: still one row
            assert client.get(f'/api/live/{GAME}/state').json()['recommendation']['ledger']['outcome'] == 'duplicate'
        assert len(feed.store.served_recommendations(GAME)) == 1
        health = client.get('/api/health').json()
        assert health['demo']['live_policy']['state'] == 'ready'


def test_fan_service_route_shows_only_a_ledgered_recommendation(tmp_path):
    now, armb = [1000.0], LiveArmB(lambda: FakePolicy())
    feed, app = serve(tmp_path, now, armb)
    with TestClient(app) as client:
        rec = ready(client, armb, f'/api/service/games/{GAME}')['next']['rec']
        stored = feed.store.served_recommendations(GAME)
        assert rec['provenance'] == 'captured_live' and rec['ledger']['decision_id'] == stored[0]['decision_id']
        assert rec['recorded_at'] == '1970-01-01T00:16:40Z'  # the ledger's decision time (wall 1000), not "now"
        assert [c['pitch_type'] for c in rec['candidates']] == \
            [c['pitch_type'] for c in stored[0]['displayed']['recommendation']['candidates']]


def test_an_unrecorded_recommendation_is_withheld_not_displayed(tmp_path, monkeypatch):
    now, armb = [1000.0], LiveArmB(lambda: FakePolicy())
    feed, app = serve(tmp_path, now, armb)
    with TestClient(app) as client:
        def broken(**_):
            raise sqlite3.OperationalError('disk I/O error')
        monkeypatch.setattr(feed.store, 'append_recommendation', broken)
        shown = ready(client, armb)
        assert shown['status'] == 'ready' and shown['situation'] is not None  # the game state is still served
        assert shown['recommendation'] == {'status': 'unavailable', 'reason': live_ledger.WITHHELD, 'recommendation': None,
                                           'ledger': {'status': 'withheld', 'why': 'OperationalError'}}
        rec = client.get(f'/api/service/games/{GAME}').json()['next']['rec']
        assert rec['provenance'] == 'absent' and rec['candidates'] == []
        monkeypatch.undo()
        entry = feed.buffers[GAME][-1]
        memory_only = {key: entry[key] for key in ('feed', 'event', 'fetched')}  # what a store outage leaves
        view = {'recommendation': {'candidates': [1]}, 'status': 'ready'}
        assert live_ledger.admit(feed, memory_only, view, game_pk=GAME)['ledger'] == {
            'status': 'withheld', 'why': 'snapshot_not_committed'}
        tampered = entry | {'feed': copy.deepcopy(entry['feed']) | {'extra': 1}}
        assert live_ledger.admit(feed, tampered, view, game_pk=GAME)['ledger']['why'] == 'snapshot_differs_from_committed_version'
        other = copy.deepcopy(entry['feed'])  # a view computed from another body than the entry's is never recorded
        assert live_ledger.admit(feed, entry, view, game_pk=GAME, snapshot=other)['ledger']['why'] == \
            'view_not_computed_from_this_snapshot'
        assert feed.store.served_recommendations(GAME) == []


def test_without_a_store_or_with_the_ledger_off_the_view_is_untouched(tmp_path, monkeypatch):
    view = {'recommendation': {'candidates': [1]}, 'status': 'ready'}
    assert live_ledger.admit(LiveFeed(ReplayTransport(FIXTURES), delay_s=0), None, view, game_pk=GAME) is view
    now, armb = [1000.0], LiveArmB(lambda: FakePolicy())
    feed, app = serve(tmp_path, now, armb)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_REC_LEDGER', '0')
    with TestClient(app) as client:
        shown = ready(client, armb)['recommendation']
    assert shown['status'] == 'ready' and 'ledger' not in shown and feed.store.served_recommendations(GAME) == []


def test_a_game_with_no_play_yet_is_served_not_an_error(tmp_path):
    from test_service_demo_api import FakeLive, feed as final_feed
    snap = final_feed(900200, '2026-10-05T10:00:00Z')
    snap['liveData']['plays']['allPlays'] = []
    snap['gameData']['status'] = {'abstractGameState': 'Preview', 'detailedState': 'Warmup'}
    app = create_app(object(), start_worker=False, watch_dir=tmp_path, feed_dir=tmp_path, demo_dir=tmp_path/'demo',
                     intent_dir=tmp_path, live_feed=FakeLive(snap))
    with TestClient(app) as client:
        response = client.get('/api/service/games/900200')
        assert response.status_code == 200 and response.json()['pitches'] == []
        assert response.json()['current']['phase'] == 'pregame'


def test_env_configured_replay_run_goes_through_the_store_and_the_ledger(tmp_path, monkeypatch):
    """The RC rehearsal configuration (``serve.sh`` env): recorded snapshots -> store -> ledger -> display."""
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(FIXTURES))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp_path/'rc.sqlite3'))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_DELAY_S', '0')
    armb = LiveArmB(lambda: FakePolicy())
    app = create_app(object(), live_policy=armb, watch_dir=tmp_path, feed_dir=tmp_path, demo_dir=tmp_path/'demo',
                     intent_dir=tmp_path)
    with TestClient(app) as client:
        shown = ready(client, armb)
        assert shown['recommendation']['ledger']['decision_clock'] == 'unverified'  # a replay is not the game's clock
        assert shown['sync']['store_ok'] is True and shown['sync']['fetch_s'] >= 0
        store = client.get('/api/health').json()['live_store']
        assert store['source'] == 'replay_poll' and store['recommendation_ledger'] is True and store['error'] is None
        assert client.get(f'/api/live/{GAME}/log', params={'delay': 0}).json()['items'][0]['source'] == 'replay_poll'
    stored = LiveSnapshotStore(tmp_path/'rc.sqlite3').served_recommendations(GAME)
    assert [(row['outcome'], row['snapshot']['source']) for row in stored] == [('first', 'replay_poll')]


def test_a_correction_arriving_mid_request_never_relabels_the_snapshot_a_view_was_computed_from(tmp_path):
    now, before, fixed = [1000.0], thrown(990.0), thrown(990.0, 91.5, '20261009_000200')
    transport = SlowPlain(now, before, 0.0)
    feed = persistent(tmp_path, transport, now)
    held, _ = feed.snapshot(GAME)  # request A holds version 1 and is still computing
    transport.feed, now[0] = fixed, 1020.0
    newer, _ = feed.snapshot(GAME)  # another request's poll corrects the same slot
    assert (held['feed'], held['version']) == (before, 1) and (newer['feed'], newer['version']) == (fixed, 2)
    view = {'recommendation': {'candidates': [1]}, 'status': 'ready', 'key': 'k', 'policy_identity': 'abc'}
    for entry in (held, newer):
        assert live_ledger.admit(feed, entry, view, game_pk=GAME, snapshot=entry['feed'])['ledger']['status'] == 'recorded'
    # the revised pitch is another decision state: each row cites the version its view was computed from
    assert [row['snapshot']['slot_version'] for row in feed.store.served_recommendations(GAME)] == [1, 2]


def test_after_a_store_outage_the_buffer_gets_its_committed_identity_back(tmp_path, monkeypatch):
    now, first, fixed, later = [1000.0], thrown(990.0), thrown(990.0, 91.5, '20261009_000200'), thrown(1030.0, 92.0, '20261009_000300')
    transport = SlowPlain(now, first, 0.0)
    feed = persistent(tmp_path, transport, now)
    feed.snapshot(GAME)
    real = feed.store.ingest

    def down(*args, **kwargs):
        raise sqlite3.OperationalError('database is locked')
    view = {'recommendation': {'candidates': [1]}, 'status': 'ready', 'key': 'k', 'policy_identity': 'abc'}
    admit = lambda entry: live_ledger.admit(feed, entry, view, game_pk=GAME, snapshot=entry['feed'])['ledger']
    for step, (body, wall) in enumerate(((fixed, 1020.0), (later, 1040.0))):  # a correction, then a new slot, both unsaved
        monkeypatch.setattr(feed.store, 'ingest', down)
        transport.feed, now[0] = body, wall
        entry, _ = feed.snapshot(GAME)
        assert 'seq' not in entry and admit(entry) == {'status': 'withheld', 'why': 'snapshot_not_committed'}
        monkeypatch.setattr(feed.store, 'ingest', real)  # the store is back; the same content is polled again
        now[0] = wall+10
        entry, _ = feed.snapshot(GAME)
        assert feed.store_error is None and entry['feed'] == body and admit(entry)['status'] == 'recorded', step
    assert [(e['event'], e.get('seq') is not None) for e in feed.buffers[GAME]] == [(990.0, True), (1030.0, True)]


def test_replay_pace_is_configurable_for_a_rehearsal_and_never_changes_the_live_pace(tmp_path, monkeypatch):
    """``PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S``: a replay hands out one recorded body per poll, so the poll
    interval is the replay speed. Only a replay run reads it; a real upstream keeps the feed's pace."""
    assert replay_config(None) is LIVE_CONFIG and replay_config('') is LIVE_CONFIG
    fast = replay_config('0.5')
    assert (fast['poll_interval_s'], fast['fast_poll_s']) == (0.5, 0.5) and LIVE_CONFIG['poll_interval_s'] == 10
    assert replay_config('60')['fast_poll_s'] == LIVE_CONFIG['fast_poll_s']  # the fast poll never exceeds the pace
    for bad in ('-1', 'nan', '99999', 'soon'):
        with pytest.raises(ValueError):
            replay_config(bad)

    def versions(name, pace):
        monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(FIXTURES))
        monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp_path/name))
        monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
        if pace is None:
            monkeypatch.delenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S', raising=False)
        else:
            monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S', pace)
        app = create_app(object(), watch_dir=tmp_path, feed_dir=tmp_path, demo_dir=tmp_path/'demo', intent_dir=tmp_path)
        with TestClient(app) as client:
            for _ in range(3):
                assert client.get(f'/api/service/games/{GAME}').status_code == 200
            health = client.get('/api/health').json()['live_store']
            return health['poll_interval_s'], len(client.get(f'/api/live/{GAME}/log', params={'delay': 0}).json()['items'])

    assert versions('paced.sqlite3', None) == (10, 1)  # three requests inside one live poll interval: one body
    assert versions('fast.sqlite3', '0') == (0.0, 2)  # one body per request: both recorded bodies, then the last is held

    monkeypatch.delenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR')  # a real upstream: the rehearsal pace is not applied
    monkeypatch.delenv('PITCHEEZY_OBSERVER_LIVE_STORE')
    seen, init = {}, LiveFeed.__init__
    monkeypatch.setattr(LiveFeed, '__init__', lambda self, transport, **options: (seen.update(options), init(self, transport, **options))[1])
    app = create_app(object(), watch_dir=tmp_path, feed_dir=tmp_path, demo_dir=tmp_path/'demo', intent_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get(f'/api/live/{GAME}/log').status_code == 404  # builds the feed without fetching anything
    assert 'config' not in seen and 'record_dir' in seen


def test_a_quiet_game_is_not_reported_stale_while_the_feed_keeps_answering(tmp_path):
    """``feed.stale_s`` (the fan screen warns above 30 s) is the age of the last upstream answer. It used to be the
    age of the shown state's first receipt, so every inning break looked like a stalled feed."""
    now, body = [1000.0], at('849845_20260929_203124.json', 990.0)
    feed = persistent(tmp_path, lambda url, timeout: copy.deepcopy(body), now)
    app = create_app(object(), live_feed=feed, live_policy=LiveArmB(lambda: FakePolicy()), watch_dir=tmp_path,
                     feed_dir=tmp_path, demo_dir=tmp_path/'demo', intent_dir=tmp_path)
    with TestClient(app) as client:
        stale = lambda: client.get(f'/api/service/games/{GAME}').json()['feed']
        first = stale()
        assert first == {'as_of': '1970-01-01T00:16:40Z', 'ok': True, 'stale_s': 0.0}
        for now[0] in (1040.0, 1100.0):  # the same body again: nothing new in the game, the feed still answers
            assert stale() == first  # ``as_of`` stays the state's first receipt; it is not stale
        now[0] = 1103.0
        assert stale()['stale_s'] == 3.0  # between polls it is the time since the last answer

        def down(url, timeout):
            raise OSError('upstream down')
        feed.transport = down
        now[0] = 1150.0
        assert stale() == first | {'ok': False, 'stale_s': 50.0}  # a failing feed ages from its last answer
