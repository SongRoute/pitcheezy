"""Synthetic nine-inning first->last game through the real poll path (``LiveFeed.snapshot`` ->
``PersistentLiveFeed._accept`` -> ``LiveSnapshotStore.ingest``); no real feed, fixture, DB, model or policy.

The expected pitch identities come from ``plan()`` alone, never from the store or its parser. Feeds use the
synthetic ``metaData.eventWall`` the replay harness uses, so real GUMBO event-time derivation is not exercised.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import urllib.error

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed
from observer_app.service_research import capture_input, conditional_candidates

GAME, T0 = 31, 1_000_000.


def plan():
    """Truth for the known final feed: (at_bat_number, pitch_number, play_id) per pitch in game order.
    Nine innings, three batters a half, no bottom ninth (home team ahead), 1-4 pitches per at-bat."""
    keys, ab = [], 0
    for inning in range(1, 10):
        for half in ('top', 'bottom')[:1 if inning == 9 else 2]:
            for _ in range(3):
                ab += 1
                keys += [(ab, n, f'g{ab:02d}p{n}') for n in range(1, 2+ab % 4)]
    return keys


TRUTH = plan()
STEP = {key[2]: i for i, key in enumerate(TRUTH)}
LAST = len(TRUTH)-1
SWAP = {(11, 2, 'g11p2'), (11, 3, 'g11p3')}  # provider first sent g11p3 as pitch 2, then renumbered
GAP, OUTAGE, CRASH = range(60, 65), range(80, 85), range(100, 105)
STALE_STEP, UNORDERED_STEP, DUP_STEPS = 40, 45, {5, 15, 35, 50, 70, 90, 110, 120}
# (step, sub-poll, play_id, call): provider revisions of an earlier pitch's payload
REVISIONS = [(20, 3, TRUTH[18][2], 'C'),                      # legitimate later correction, received
             (61, 1, TRUTH[59][2], 'C'), (63, 1, TRUTH[59][2], 'S'),  # during disconnect: 'C' never received
             (81, 2, TRUTH[79][2], 'C'), (81, 3, TRUTH[79][2], 'S'),  # store down: 'C' received, superseded
             (101, 1, TRUTH[99][2], 'C'), (103, 1, TRUTH[99][2], 'S')]  # store down then crash


def stamp(step, sub=0):
    return (datetime(2026, 10, 9, tzinfo=timezone.utc)+timedelta(seconds=10*step+sub)).strftime('%Y%m%d_%H%M%S')


def provider(step, sub=0, *, ts=None, conflict=False):
    """The provider's cumulative feed after pitch ``step``. Substitutions are non-pitch action events."""
    calls = {pid: call for at, s, pid, call in REVISIONS if (at, s) <= (step, sub)}
    plays = {}
    for i, (ab, n, pid) in enumerate(TRUTH[:step+1]):
        if (ab, n) == (11, 2) and step < STEP['g11p3']:
            pid = 'g11p3'
        if (ab, n) == (5, 1) and step == i:
            pid = None  # first sighting before the provider assigned a playId
        events = plays.setdefault(ab, [])
        if ab == 20 and n == 1:
            events.append({'isPitch': False, 'type': 'action', 'details': {'eventType': 'pitching_substitution'}})
        if ab == 30 and n == 2:
            events.append({'isPitch': False, 'type': 'action', 'details': {'eventType': 'offensive_substitution'}})
        call = 'X' if conflict and i == step else calls.get(pid, 'B')
        events.append({'isPitch': True, 'playId': pid, 'pitchNumber': n, 'details': {'call': {'code': call}},
                       'startTime': stamp(i)})
    meta = {'eventWall': T0+10*step, 'timeStamp': stamp(step, sub) if ts is None else ts}
    if ts is False:
        del meta['timeStamp']
    return {'metaData': meta, 'liveData': {'plays': {'allPlays': [
        {'about': {'atBatIndex': ab-1, 'inning': (ab-1)//6+1},
         'matchup': {'batter': 600+ab if ab == 30 and len(events) > 2 else 100+ab, 'pitcher': 1 if ab < 20 else 2},
         'playEvents': events} for ab, events in plays.items()]}}}


class Flaky(LiveSnapshotStore):
    down, outcomes = False, []

    def ingest(self, game_pk, feed, **kwargs):
        if Flaky.down:
            raise sqlite3.OperationalError('disk I/O error')
        outcome, entry = super().ingest(game_pk, feed, **kwargs)
        Flaky.outcomes.append((kwargs['source'], outcome))
        return outcome, entry


def partition(store, upto):
    """Independent denominator TRUTH[:upto+1] split into mutually exclusive accepted / quarantined-only /
    missing identity keys; ``extra`` are accepted keys absent from that truth."""
    events = store.pitch_events(GAME)['events']
    key = lambda e: (e['at_bat_number'], e['pitch_number'], e['provider_event_id'])
    accepted = {key(e) for e in events if e['outcome'] != 'quarantine'}
    quarantined = {key(e) for e in events if e['outcome'] == 'quarantine'}
    truth = set(TRUTH[:upto+1])
    return {'accepted': truth & accepted, 'quarantined_only': (truth & quarantined)-accepted,
            'missing': truth-accepted-quarantined, 'extra': accepted-truth}


def expected(upto, unpersisted=()):
    truth = set(TRUTH[:upto+1])
    missing = {TRUTH[i] for i in unpersisted}
    quarantined = SWAP & truth
    return {'accepted': truth-missing-quarantined, 'quarantined_only': quarantined, 'missing': missing,
            'extra': {(11, 2, 'g11p3')} if upto >= STEP['g11p2'] else set()}


def test_nine_inning_first_to_last_accounting_through_outage_gap_and_restart(tmp_path):
    path, now, serving, received = tmp_path/'game.sqlite3', [T0], [None], []
    Flaky.down, Flaky.outcomes = False, []

    def transport(url, timeout):
        if isinstance(serving[0], Exception):
            raise serving[0]
        received.append(serving[0])
        return serving[0]

    def start():
        return PersistentLiveFeed(transport, store=Flaky(path), delay_s=0, clock=lambda: now[0],
                                  wall=lambda: now[0], sleep=lambda _: None)

    def poll(feed, at):
        serving[0], now[0] = feed, at
        live.fetched_at.clear()
        live.snapshot(GAME)

    live, checks = start(), {}
    for step in range(LAST+1):
        at = T0+10*step+1
        Flaky.down = step in OUTAGE or step in CRASH
        if step in GAP:
            poll(urllib.error.URLError('disconnected'), at)
            if step in (GAP.start+2, GAP.stop-1):
                checks[step] = partition(live.store, step)
            continue
        if step == CRASH.stop:  # process crashed during the outage: memory is gone, a new process restores
            checks['before_restart'] = partition(live.store, step-1)
            live = start()
        ts = stamp(step-3) if step == STALE_STEP else False if step == UNORDERED_STEP else None
        poll(provider(step, ts=ts), at)
        if step in DUP_STEPS:
            poll(provider(step), at+.5)
        for s in sorted({s for st, s, _, _ in REVISIONS if st == step}):
            poll(provider(step, s), at+s/10)
        if step == 30:
            poll(provider(step, conflict=True), at+.2)  # same provider stamp, different payload
    store = live.store
    final = partition(store, LAST)

    assert len(TRUTH) == 129 and TRUTH[0] == (1, 1, 'g01p1') and TRUTH[-1] == (51, 4, 'g51p4')
    assert len({(ab, n) for ab, n, _ in TRUTH}) == len({pid for *_, pid in TRUTH}) == 129
    assert checks[GAP.start+2] == expected(GAP.start+2, range(GAP.start, GAP.start+3))
    assert checks[GAP.stop-1] == expected(GAP.stop-1, GAP)
    assert checks['before_restart'] == expected(104, CRASH)
    received_ids = {e.get('playId') for f in received for p in f['liveData']['plays']['allPlays'] for e in p['playEvents']}
    assert {TRUTH[i][2] for i in CRASH} <= received_ids  # received but never persisted before the crash
    assert final == expected(LAST)
    assert (len(final['accepted']), len(final['quarantined_only']), len(final['missing'])) == (127, 2, 0)
    assert TRUTH[0] in final['accepted'] and TRUTH[-1] in final['accepted']

    rows = store.pitch_events(GAME)['events']
    quarantine = sorted(((r['provider_event_id'], r['reason']) for r in rows if r['outcome'] == 'quarantine'), key=str)
    assert quarantine == sorted([(None, 'incomplete_identity'), ('g11p2', 'ambiguous_remap'), ('g11p3', 'conflicting_key'),
                                 (TRUTH[30][2], 'same_revision_hash_mismatch')], key=str)
    corrections = [(r['provider_event_id'], r['revision'], r['provider_ts'], r['receipt_clock'])
                   for r in rows if r['outcome'] == 'correction']
    assert corrections == [(TRUTH[18][2], 2, stamp(20, 3), 'observer_local'),
                           (TRUTH[59][2], 2, stamp(65), 'observer_local'),   # 'C' never received
                           (TRUTH[79][2], 2, stamp(81, 3), 'unverified'),    # memory_recovery; 'C' lost
                           (TRUTH[99][2], 2, stamp(105), 'observer_local')]  # 'C' lost in the crash
    live_dups = [o for s, o in Flaky.outcomes if s == 'live_poll' and o == 'duplicate']
    # 'new' rows: 129 identities minus g11p2 (never accepted); g11p3's one 'new' row is its misnumbered key
    assert len(live_dups) == len(DUP_STEPS) and len(rows) == 128+4+4

    first = {}
    for r in rows:
        if r['outcome'] != 'quarantine':
            assert first.setdefault(r['provider_event_id'], r['first_received']) == r['first_received']
    # recorded first receipt: local poll time, not the provider's first publication
    assert first['g01p1'] == T0+1 and first[TRUTH[-1][2]] == T0+10*LAST+1
    assert all(first[TRUTH[i][2]] == T0+10*GAP.stop+1 for i in GAP)
    assert all(first[TRUTH[i][2]] == T0+10*CRASH.stop+1 for i in CRASH)
    # memory_recovery writes each slot once with its LAST in-memory receipt time (unverified clock), so a pitch
    # whose slot was re-polled during the outage gets a recorded first receipt later than its real local one
    assert [first[TRUTH[i][2]] for i in OUTAGE] == [T0+801, T0+811.3, T0+821, T0+831, T0+841]
    assert live.store_error is None
    history = store.pitch_events(GAME)['history']
    assert history['status'] == 'UNKNOWN' and history['reasons'] == ['intermediate_revisions_unlogged',
                                                                      'store_outage_recovered']
    assert LiveSnapshotStore(path).pitch_events(GAME)['history'] == history  # restart: same status/as-of


def test_newer_event_with_stale_or_missing_provider_stamp_is_not_fresh_input(tmp_path):
    path = tmp_path/'fresh.sqlite3'
    store = LiveSnapshotStore(path)
    entries = {}
    for step, ts in ((0, None), (1, None), (2, stamp(0)), (3, False), (4, None)):
        outcome, entries[step] = store.ingest(GAME, provider(step, ts=ts), fetched=T0+10*step+1, source='live_poll')
        assert outcome == 'new'  # every step is a newer game event, so each is selected as its own slot
    assert [entries[s]['provider_order'] for s in range(5)] == ['ordered', 'ordered', 'stale', 'unorderable', 'ordered']
    restored = LiveSnapshotStore(path).restore(GAME, T0+100, 1000)
    assert [e['provider_order'] for e in restored] == ['ordered', 'ordered', 'stale', 'unorderable', 'ordered']
    calls, vectors = [], np.zeros((1, 199))

    def predict(x):  # stub, never a model
        calls.append(len(x))
        return np.full((len(x), 10), .1)

    for entry, reason in zip(restored, (None, None, 'stale_provider_stamp', 'unorderable_provider_stamp', None)):
        receipt = capture_input(entry, {'step': entry['seq']}, game_pk=GAME, trusted_clock=True,
                                decision_at=datetime.fromtimestamp(T0+100, timezone.utc).isoformat())
        assert receipt['freshness_reason'] == reason and receipt['feed_receipt_status'] == 'verified_local_receipt'
        if reason:
            evidence = {'kind': 'synthetic_fixture', 'input_sha256': receipt['input_sha256'],
                        'vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(),
                        'actions_sha256': hashlib.sha256(json.dumps(['FF']).encode()).hexdigest(),
                        'model_sha256': 'a'*64, 'encoder_sha256': 'b'*64}
            result = conditional_candidates(receipt, vectors, ['FF'], predict, model_sha256='a'*64,
                                            encoder_sha256='b'*64, support_counts=[40], input_evidence=evidence)
            assert result == {'status': 'unsupported', 'reason': reason, 'candidates': []}
    assert not calls
    memory_only = {k: v for k, v in restored[-1].items() if k != 'provider_order'}
    assert capture_input(memory_only, {}, game_pk=GAME, decision_at=datetime.fromtimestamp(
        T0+100, timezone.utc).isoformat())['freshness_reason'] == 'provider_order_unknown'
