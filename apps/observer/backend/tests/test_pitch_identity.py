"""Synthetic one-game pitch identity ledger checks; no real feed, DB, or recorded snapshot."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_store import LiveSnapshotStore
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
