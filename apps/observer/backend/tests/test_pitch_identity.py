"""Synthetic one-game pitch identity ledger checks; no real feed, DB, or recorded snapshot."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed, _stale_stamp
from observer_app.service_research import capture_input


def pitch(pid, number, call='B', speed=90.):
    return {'isPitch': True, 'playId': pid, 'pitchNumber': number, 'startTime': f'2026-10-09T00:00:{number:02d}Z',
            'details': {'call': {'code': call}}, 'pitchData': {'startSpeed': speed}}


def feed(plays, ts, wall):
    """plays: list of (inning, [events]); atBatIndex is the list position."""
    return {'metaData': {'eventWall': wall, 'timeStamp': ts},
            'liveData': {'plays': {'allPlays': [{'about': {'atBatIndex': i, 'inning': inning}, 'playEvents': events}
                                                for i, (inning, events) in enumerate(plays)]}}}


def accepted(ledger):
    return [e for e in ledger['events'] if e['outcome'] != 'quarantine']


def test_first_to_last_synthetic_game_with_substitution_disconnect_restart(tmp_path):
    path = tmp_path/'ledger.sqlite3'
    store = LiveSnapshotStore(path)
    sub = {'isPitch': False, 'type': 'action', 'details': {'eventType': 'pitching_substitution'}}
    pa1 = (1, [pitch('a1', 1), pitch('a2', 2, 'X')])
    pa2 = (1, [pitch('b1', 1), sub, pitch('b2', 2), pitch('b3', 3, 'X')])
    pa3 = (9, [pitch('c1', 1, 'X')])
    store.ingest(5, feed([(1, pa1[1][:1])], '20261009_000001', 100.), fetched=101., source='live_poll')
    # disconnect: next poll already holds two later at-bats; nothing was polled in between
    store.ingest(5, feed([pa1, pa2], '20261009_000005', 140.), fetched=150., source='live_poll')
    # restart: a new store object on the same file, catch-up replays everything already seen
    store = LiveSnapshotStore(path)
    for f, at in ((feed([pa1, pa2], '20261009_000005', 140.), 151.), (feed([pa1, pa2, pa3], '20261009_000009', 200.), 201.)):
        store.ingest(5, f, fetched=at, source='live_poll')
    store.ingest(5, feed([pa1, pa2, pa3], '20261009_000009', 200.), fetched=202., source='live_poll')
    ledger = store.pitch_events(5)
    assert [(e['at_bat_number'], e['pitch_number']) for e in accepted(ledger)] == [
        (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1)]
    assert all(e['outcome'] == 'new' for e in ledger['events'])
    assert ledger['coverage'] == {1: [1, 2], 2: [1, 2, 3], 3: [1]} and not ledger['gaps'] and not ledger['missing_at_bats']
    first = {e['provider_event_id']: e['first_received'] for e in ledger['events']}
    assert first['a1'] == 101. and first['a2'] == first['b3'] == 150. and first['c1'] == 201.
    assert {e['receipt_clock'] for e in ledger['events']} == {'observer_local'}


def test_correction_appends_revision_and_keeps_first_receipt_and_captured_input(tmp_path):
    store = LiveSnapshotStore(tmp_path/'ledger.sqlite3')
    _, entry = store.ingest(6, feed([(1, [pitch('p1', 1, 'B')])], '20261009_000001', 100.), fetched=101., source='live_poll')
    receipt = capture_input(entry, {'balls': 0, 'strikes': 0}, game_pk=6,
                            decision_at=datetime.fromtimestamp(102., timezone.utc).isoformat(), trusted_clock=True)
    frozen = copy.deepcopy(receipt)
    store.ingest(6, feed([(1, [pitch('p1', 1, 'C')])], '20261009_000002', 100.), fetched=110., source='live_poll')
    store.ingest(6, feed([(1, [pitch('p1', 1, 'S')])], '20261009_000003', 100.), fetched=120., source='live_poll')
    rows = store.pitch_events(6)['events']
    assert [(r['outcome'], r['revision'], r['first_received'], r['received']) for r in rows] == [
        ('new', 1, 101., 101.), ('correction', 2, 101., 110.), ('correction', 3, 101., 120.)]
    assert len({r['sha'] for r in rows}) == 3 and [r['provider_ts'] for r in rows][-1] == '20261009_000003'
    assert receipt == frozen and receipt['first_received_at'] == datetime.fromtimestamp(101., timezone.utc).isoformat()
    with pytest.raises(sqlite3.DatabaseError):
        with sqlite3.connect(store.path) as db:
            db.execute('UPDATE pitch_events SET received=0')


def test_same_revision_hash_conflicting_key_and_ambiguous_remap_are_quarantined(tmp_path):
    store = LiveSnapshotStore(tmp_path/'ledger.sqlite3')
    store.ingest(7, feed([(1, [pitch('p1', 1), pitch('p2', 2)])], '20261009_000001', 100.), fetched=101., source='live_poll')
    # same provider timeStamp, changed payload -> inconsistent revision
    store.ingest(7, feed([(1, [pitch('p1', 1, speed=91.), pitch('p2', 2)])], '20261009_000001', 100.), fetched=102., source='live_poll')
    # p2 moved to another key; a new id claims p1's key
    store.ingest(7, feed([(1, [pitch('zz', 1), pitch('p2', 3)])], '20261009_000002', 100.), fetched=103., source='live_poll')
    # repeated bad input does not grow the quarantine
    store.ingest(7, feed([(1, [pitch('zz', 1), pitch('p2', 3)]), (1, [])], '20261009_000003', 100.), fetched=104., source='live_poll')
    rows = store.pitch_events(7)['events']
    assert [(r['provider_event_id'], r['outcome'], r['reason']) for r in rows] == [
        ('p1', 'new', None), ('p2', 'new', None), ('p1', 'quarantine', 'same_revision_hash_mismatch'),
        ('zz', 'quarantine', 'ambiguous_remap'), ('p2', 'quarantine', 'conflicting_key')]
    assert [(r['at_bat_number'], r['pitch_number'], r['revision']) for r in accepted(store.pitch_events(7))] == [(1, 1, 1), (1, 2, 1)]


def test_recorded_or_recovered_receipt_is_unverified(tmp_path):
    store = LiveSnapshotStore(tmp_path/'ledger.sqlite3')
    store.ingest(8, feed([(1, [pitch('r1', 1)])], '20261009_000001', 100.), fetched=105., source='recorded:x.json.gz')
    assert store.pitch_events(8)['events'][0]['receipt_clock'] == 'unverified'


def test_changed_payload_needs_strictly_later_valid_provider_timestamp(tmp_path):
    store = LiveSnapshotStore(tmp_path/'ledger.sqlite3')
    store.ingest(10, feed([(1, [pitch('p1', 1, 'B')])], '20261009_000005', 100.), fetched=101., source='live_poll')
    # newer game event (event_wall 200 > 100) but OLDER provider timeStamp: never a correction
    store.ingest(10, feed([(1, [pitch('p1', 1, 'C')])], '20261009_000004', 200.), fetched=201., source='live_poll')
    # malformed timeStamp cannot be ordered; '2026109_0000099' would sort after lexically
    store.ingest(10, feed([(1, [pitch('p1', 1, 'S')])], '2026109_0000099', 300.), fetched=301., source='live_poll')
    store.ingest(10, feed([(1, [pitch('p1', 1, 'X')])], '20261009_000006', 400.), fetched=401., source='live_poll')
    rows = store.pitch_events(10)['events']
    assert [(r['outcome'], r['reason'], r['revision']) for r in rows] == [
        ('new', None, 1), ('quarantine', 'stale_revision', None), ('quarantine', 'unordered_revision', None),
        ('correction', None, 2)]


def test_malformed_items_quarantine_alone_and_keep_valid_pitches(tmp_path):
    store = LiveSnapshotStore(tmp_path/'ledger.sqlite3')
    bad = [dict(pitch('n1', 1), pitchNumber=True), dict(pitch('n2', 0)), dict(pitch('n3', 2), pitchNumber='2'),
           dict(pitch('', 3)), dict(pitch(None, 4)), dict(pitch(True, 5)), dict(pitch(['l'], 6)), 'junk']
    plays = [{'about': {'atBatIndex': 0}, 'playEvents': [pitch('ok1', 1), *bad]},
             {'about': {'atBatIndex': True}, 'playEvents': [pitch('b1', 1)]},
             {'about': {'atBatIndex': -1}, 'playEvents': [pitch('b2', 1)]},
             {'about': {'atBatIndex': 'x'}, 'playEvents': [pitch('b3', 1)]},
             {'about': {'atBatIndex': 4}, 'playEvents': [pitch('ok2', 1)]}]
    outcome, _ = store.ingest(11, {'metaData': {'eventWall': 100., 'timeStamp': '20261009_000001'},
                                   'liveData': {'plays': {'allPlays': plays}}}, fetched=101., source='live_poll')
    ledger = store.pitch_events(11)
    assert outcome == 'new'
    assert [(e['provider_event_id'], e['at_bat_number'], e['pitch_number']) for e in accepted(ledger)] == [
        ('ok1', 1, 1), ('ok2', 5, 1)]
    quarantined = [e for e in ledger['events'] if e['outcome'] == 'quarantine']
    assert len(quarantined) == 11 and all(e['at_bat_number'] is None and e['pitch_number'] is None for e in quarantined)
    assert {e['reason'] for e in quarantined} == {'incomplete_identity', 'malformed_event'}


@pytest.mark.parametrize('bad', ['junk', None, '2026109_0000099'])
def test_unorderable_stamp_never_displaces_valid_selection_and_still_gets_receipt(tmp_path, bad):
    store = LiveSnapshotStore(tmp_path/'ledger.sqlite3')
    live = PersistentLiveFeed(None, store=store, wall=lambda: 1000.)
    buffer = live._buffer(13)

    def poll(call, ts, at):
        f = feed([(1, [pitch('p1', 1, call)])], ts, 100.)
        if ts is None:
            del f['metaData']['timeStamp']
        live._accept(13, f, at, buffer)
        return f

    def selected():
        stored = store.read_after(13, until_event=200.)['items'][-1]
        assert store.restore(13, 1000., 1000.)[-1]['seq'] == stored['seq'] == buffer[-1]['seq']
        return buffer[-1]['feed'], stored['version']

    def ledger():
        return [(r['outcome'], r['reason'], r['revision'], r['first_received'], r['received'], r['store_seq'])
                for r in store.pitch_events(13)['events']]

    b = poll('B', '20261009_000100', 100.)
    poll('C', bad, 110.)  # same slot, unorderable stamp: refused, but its changed pitch is received
    assert selected() == (b, 1) and live.stale[13] == 1
    assert ledger() == [('new', None, 1, 100., 100., 1), ('quarantine', 'unordered_revision', None, None, 110., 1)]
    s = poll('S', '20261009_000200', 120.)  # later valid correction recovers
    assert selected() == (s, 2)
    assert ledger()[2:] == [('correction', None, 2, 100., 120., 2)]
    poll('C', bad, 130.)  # repeat: refused again, quarantine deduplicated
    assert selected() == (s, 2) and live.stale[13] == 2 and len(ledger()) == 3
    poll('F', '20261009_000050', 140.)  # older valid stamp: refused, received as stale_revision
    assert selected() == (s, 2) and live.stale[13] == 3
    assert ledger()[3:] == [('quarantine', 'stale_revision', None, None, 140., 2)]
    assert ledger()[0][3] == 100. and live.store_error is None


def test_stale_stamp_ordering_rule():
    valid, later = '20261009_000100', '20261009_000200'
    assert _stale_stamp(valid, later, False) and not _stale_stamp(later, valid, False)
    for bad in ('junk', None, '', '2026109_0000099'):
        assert _stale_stamp(bad, valid, False) and not _stale_stamp(valid, bad, False)
        assert not _stale_stamp(bad, valid, True)  # a newer event is its own slot whatever its stamp
    # neither orderable: legacy raw compare, so a lexically smaller string is refused, otherwise the later arrival wins
    assert _stale_stamp('110', '120', False) and not _stale_stamp('junk', None, False) and not _stale_stamp(None, 'x', False)


@pytest.mark.parametrize('live_data', ['bad', {'plays': 'bad'}, {'plays': {'allPlays': 'bad'}}])
def test_malformed_live_data_or_plays_is_stored_without_raising(tmp_path, live_data):
    store = LiveSnapshotStore(tmp_path/'ledger.sqlite3')
    live = PersistentLiveFeed(None, store=store, wall=lambda: 1000.)
    buffer = live._buffer(14)
    live._accept(14, feed([(1, [pitch('p1', 1)])], '20261009_000100', 100.), 100., buffer)
    bad = {'metaData': {'eventWall': 200., 'timeStamp': '20261009_000200'}, 'liveData': live_data}
    live._accept(14, bad, 200., buffer)
    assert buffer[-1]['feed'] == bad and buffer[-1]['seq'] == 2 and live.store_error is None
    assert [(r['provider_event_id'], r['outcome']) for r in store.pitch_events(14)['events']] == [('p1', 'new')]


def test_sqlite_outage_and_network_gap_lose_different_things(tmp_path):
    """Observed ceiling of the SVC-003 memory recovery: per slot only the last version received during the outage
    is written back (as unverified ``memory_recovery``); a correction the poller never fetched leaves no trace."""

    class Flaky(LiveSnapshotStore):
        down = False

        def ingest(self, *args, **kwargs):
            if self.down:
                raise sqlite3.OperationalError('disk I/O error')
            return super().ingest(*args, **kwargs)

    serving, now = [None], [1000.]
    store = Flaky(tmp_path/'ledger.sqlite3')
    live = PersistentLiveFeed(lambda url, timeout: serving[0], store=store, delay_s=0,
                              clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None)

    def poll(f, at):
        serving[0], now[0] = f, at
        live.fetched_at.clear()
        live.snapshot(12)

    poll(feed([(1, [pitch('p1', 1, 'B')])], '20261009_000001', 1000.), 1000.)
    store.down = True  # received but unpersisted
    poll(feed([(1, [pitch('p1', 1, 'C')])], '20261009_000002', 1000.), 1010.)
    poll(feed([(1, [pitch('p1', 1, 'S')])], '20261009_000003', 1000.), 1020.)
    assert live.store_error == 'ingest: OperationalError'
    store.down = False
    # network gap: the provider's '20261009_000004' version (call 'F') is never fetched
    poll(feed([(1, [pitch('p1', 1, 'X')])], '20261009_000005', 1000.), 1050.)
    rows = store.pitch_events(12)['events']
    assert [(r['outcome'], r['revision'], r['provider_ts'], r['receipt_clock'], r['received']) for r in rows] == [
        ('new', 1, '20261009_000001', 'observer_local', 1000.),
        ('correction', 2, '20261009_000003', 'unverified', 1020.),
        ('correction', 3, '20261009_000005', 'observer_local', 1050.)]
    assert live.store_error is None
    # 'C' (received, unpersisted) and 'F' (never received) are both absent; revisions stay contiguous, so
    # neither loss is detectable from the ledger alone.
