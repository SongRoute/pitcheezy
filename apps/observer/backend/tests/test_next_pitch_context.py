"""SYNTHETIC NextPitchContext adapter + append-only decision record; no real feed, fixture, model or policy.

The ``syntheticPrePitch`` object is an explicit test-declared pre-pitch state, not a real provider field."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_store import SYNTHETIC_SCHEMA, SYNTHETIC_SOURCE, LiveSnapshotStore
from observer_app.service_research import CONTEXT_FIELDS, next_pitch_context

GAME = 7
FIELDS = ('balls', 'strikes', 'outs_when_up', 'bases', 'stand')  # 'stand' is opted in but absent from the body
PATHS = {'balls': 'balls', 'strikes': 'strikes', 'outs_when_up': 'outs', 'bases': 'bases'}


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def feed(pitches, *, ts, wall, call='B', state=None):
    """Latest play holds ``pitches`` (key ab=1); ``state`` is the explicit synthetic pre-pitch state."""
    events = [{'isPitch': True, 'playId': f'p{n}', 'pitchNumber': n, 'details': {'call': {'code': call if n == pitches else 'B'}}}
              for n in range(1, pitches+1)]
    body = {'metaData': {'eventWall': wall, 'timeStamp': ts},
            'liveData': {'plays': {'allPlays': [{'about': {'atBatIndex': 0}, 'matchup': {'batter': 101, 'pitcher': 1},
                                                 'playEvents': events}]}}}
    if state is not None:
        body['syntheticPrePitch'] = {'schema': SYNTHETIC_SCHEMA} | state
    return body


def build(entry, at, **kwargs):
    options = dict(game_pk=GAME, decision_at=iso(at), decision_clock='synthetic_test_clock', disconnected=False,
                   trusted_clock=True, known_fields=FIELDS)
    return next_pitch_context(entry, **(options | kwargs))


def test_known_context_abstains_and_persists_immutably(tmp_path):
    path = tmp_path/'ctx.sqlite3'
    store = LiveSnapshotStore(path)
    _, entry = store.ingest(GAME, feed(1, ts='20261008_000100', wall=100., state={'balls': 1, 'strikes': 0, 'outs': 2,
                                                                                    'bases': [1]}),
                            fetched=101., source=SYNTHETIC_SOURCE)
    predicted = []

    def gate(record):  # a caller may reach a predictor only from 'eligible'
        if record['opportunity']['category'] == 'eligible':
            predicted.append(record['decision_id'])
        return record

    # (1) explicit as-of state: a few KNOWN fields with bounded provenance, the rest and the target UNKNOWN
    record = gate(build(entry, 105.))
    known = {f: v for f, v in record['fields'].items() if v['status'] == 'KNOWN'}
    assert {f: v['value'] for f, v in known.items()} == {'balls': 1, 'strikes': 0, 'outs_when_up': 2, 'bases': [1]}
    for f, v in known.items():
        assert v['provenance'] == {'source': 'selected_snapshot', 'path': ['syntheticPrePitch', PATHS[f]],
                                   'store_seq': entry['seq'], 'slot_version': 1, 'feed_sha256': entry['sha'],
                                   'version_received_at': iso(101.), 'receipt_clock': 'synthetic_test_clock',
                                   'evidence': 'synthetic_test_state_not_provider'}
    assert record['fields']['stand'] == {'value': None, 'status': 'UNKNOWN', 'provenance': None,
                                         'reason': 'synthetic_field_absent'}
    assert set(record['fields']) == set(CONTEXT_FIELDS)
    assert all(record['fields'][f]['status'] == 'UNKNOWN' and record['fields'][f]['value'] is None
               for f in ('batter', 'pitcher', 'inning', 'batter_style_priors', 'prior_pitch_tokens'))
    assert record['target'] == {'game_pk': GAME, 'at_bat_number': None, 'pitch_number': None, 'status': 'UNKNOWN',
                                'reason': 'next_key_not_assignable_before_pitch'}
    assert record['last_received_pitch'] == {'at_bat_number': 1, 'pitch_number': 1}
    assert record['opportunity'] == {'category': 'feature_unavailable', 'reason': 'required_f1_field_unknown'}
    assert record['model'] is None and record['decision_at'] == iso(105.)
    assert record['snapshot']['receipt_clock'] == 'synthetic_test_clock' and record['snapshot']['disconnected'] is False
    assert 'p1' not in str(record) and "'call'" not in str(record)  # no playId/call copied in
    with pytest.raises(ValueError, match='No snapshot source'):
        build(entry, 105., known_fields=('prior_pitch_tokens',))

    # (2) future, disconnected, unknown connection, unknown receipt clock, stale, memory-only: all abstain
    store.ingest(GAME, feed(2, ts='20261008_000200', wall=110., state={'balls': 2}), fetched=111.,
                            source='live_poll')
    store.ingest(GAME, feed(2, ts='20261008_000200', wall=110., call='S', state={'balls': 2}), fetched=112.,
                 source='live_poll')  # same-stamp conflicting body
    conflict = store.restore(GAME, 115., 100)[-1]
    memory_only = {k: v for k, v in entry.items() if k != 'seq'}
    cases = {'future': (build(entry, 100.5), 'unsupported', 'selected_snapshot_unverifiable'),
             'disconnected': (build(entry, 105., disconnected=True), 'ambiguous', 'feed_disconnected_last_snapshot_only'),
             'connection_unknown': (build(entry, 105., disconnected=None), 'ambiguous',
                                    'connection_status_not_captured_with_snapshot'),
             'receipt_clock': (build(entry, 105., trusted_clock=False), 'ambiguous', 'receipt_clock_unverified'),
             'decision_clock': (build(entry, 105., decision_clock='UNKNOWN'), 'ambiguous', 'decision_clock_unverified'),
             'same_stamp': (build(conflict, 115.), 'ambiguous', 'same_stamp_conflict_provider_stamp'),
             'memory_only': (build(memory_only, 105.), 'unsupported', 'selected_snapshot_unverifiable'),
             'no_snapshot': (build(None, 105.), 'unsupported', 'no_snapshot')}
    for name, (abstained, category, reason) in cases.items():
        gate(abstained)
        assert abstained['opportunity'] == {'category': category, 'reason': reason}, name
        assert all(v['status'] == 'UNKNOWN' and v['value'] is None for v in abstained['fields'].values()), name
        assert abstained['target']['at_bat_number'] is None, name
    assert cases['receipt_clock'][0]['snapshot']['receipt_clock'] == 'UNKNOWN'
    assert cases['same_stamp'][0]['snapshot']['restored'] is True
    assert not predicted

    # (3) append once, reload, reconcile later, correct later, reload: the original never changes
    assert store.append_decision(record) == 'appended'
    original = LiveSnapshotStore(path).decision_history(record['decision_id'])['decision']
    assert store.append_decision(copy.deepcopy(record)) == 'duplicate'
    with pytest.raises(ValueError, match='conflicting'):
        store.append_decision(record | {'decision_at': iso(104.)})
    with pytest.raises(ValueError, match='not a committed'):
        store.append_decision(build(entry, 105.) | {'snapshot': record['snapshot'] | {'store_seq': 99}})
    for name, (abstained, _, _) in cases.items():
        assert store.append_decision(abstained) == 'appended', name
    counts = store.decision_counts(GAME)
    assert counts == {'unit': 'local_decision_opportunity', 'decisions': 9,
                      'categories': {'feature_unavailable': 1, 'ambiguous': 5, 'unsupported': 3}}

    ledger = store.pitch_events(GAME)['events']
    before, arrived = ledger[0], [r for r in ledger if (r['pitch_number'], r['outcome']) == (2, 'new')][0]
    with pytest.raises(ValueError, match='receipt clock'):  # the synthetic-source row is not on the observer clock
        store.append_reconciliation(record['decision_id'], reconciled_at=iso(120.), evidence=[before['id']])
    with pytest.raises(ValueError, match='after decision_at'):
        store.append_reconciliation(record['decision_id'], reconciled_at=iso(104.), evidence=[arrived['id']])
    first = store.append_reconciliation(record['decision_id'], reconciled_at=iso(120.), evidence=[arrived['id']])
    assert first['revision'] == 1 and first['previous'] is None and 'target' not in first
    assert first['candidate_target'] == {'game_pk': GAME, 'at_bat_number': 1, 'pitch_number': 2,
                                         'status': 'UNVERIFIED_NEXT', 'intervening': [],
                                         'basis': 'accepted_row_first_received_after_decision',
                                         'reason': 'later_observation_not_proof_of_immediate_next_pitch'}
    _, fixed = store.ingest(GAME, feed(2, ts='20261008_000300', wall=110., call='C', state={'balls': 2}), fetched=130.,
                            source='live_poll')
    correction = [r for r in store.pitch_events(GAME)['events'] if r['outcome'] == 'correction'][-1]
    second = store.append_reconciliation(record['decision_id'], reconciled_at=iso(131.), evidence=[correction['id']])
    assert second['revision'] == 2 and second['evidence'][0]['outcome'] == 'correction'
    assert second['candidate_target']['status'] == 'UNVERIFIED_NEXT'
    assert second['previous'] == {'revision': 1, 'at_bat_number': 1, 'pitch_number': 2,
                                  'relation': 'same_key_later_observation'}

    history = LiveSnapshotStore(path).decision_history(record['decision_id'])
    assert history['decision'] == original
    assert [r['revision'] for r in history['reconciliations']] == [1, 2]
    assert history['reconciliations'][0] == first and history['reconciliations'][1] == second
    restored = history['record']
    assert restored['decision_at'] == iso(105.) and restored['snapshot'] == record['snapshot']
    assert restored['target']['at_bat_number'] is None
    with sqlite3.connect(path) as db, pytest.raises(sqlite3.IntegrityError, match='append-only'):
        db.execute("UPDATE context_decisions SET body=x'00'")
    with sqlite3.connect(path) as db, pytest.raises(sqlite3.IntegrityError, match='append-only'):
        db.execute('DELETE FROM context_reconciliations')
    assert LiveSnapshotStore(path).decision_history(record['decision_id']) == history

    # an unverified receipt clock cannot place evidence after the decision
    recorded = LiveSnapshotStore(tmp_path/'recorded.sqlite3')
    _, old = recorded.ingest(GAME, feed(1, ts='20261008_000100', wall=100.), fetched=101., source='live_poll')
    late = build(old, 105.)
    recorded.append_decision(late)
    recorded.ingest(GAME, feed(2, ts='20261008_000200', wall=110.), fetched=111., source='recorded:x.json')
    row = recorded.pitch_events(GAME)['events'][-1]
    assert row['receipt_clock'] == 'unverified'
    with pytest.raises(ValueError, match='receipt clock'):
        recorded.append_reconciliation(late['decision_id'], reconciled_at=iso(120.), evidence=[row['id']])


def test_reviewer_counterexamples_stay_unknown_and_unverified(tmp_path):
    store = LiveSnapshotStore(tmp_path/'ce.sqlite3')
    everything = tuple(f for f in CONTEXT_FIELDS if f not in ('batter_style_priors', 'prior_pitch_tokens'))

    def known(record):
        return {f for f, v in record['fields'].items() if v['status'] == 'KNOWN'}

    # CE1a: a Final-like live_poll body with a retrospective linescore (and a look-alike envelope): nothing KNOWN
    final = feed(3, ts='20261008_000100', wall=100., state={'balls': 3, 'strikes': 2, 'outs': 2})
    final['gameData'] = {'status': {'abstractGameState': 'Final'}}
    final['liveData']['linescore'] = {'balls': 3, 'strikes': 2, 'outs': 3, 'currentInning': 9, 'inningHalf': 'Bottom'}
    _, live = store.ingest(GAME, final, fetched=101., source='live_poll')
    for clock in ('observer_local', 'synthetic_test_clock'):
        rec = build(live, 105., decision_clock=clock, known_fields=everything)
        assert not known(rec) and rec['fields']['balls']['value'] is None
    assert build(live, 105., decision_clock='observer_local',
                 known_fields=everything)['fields']['balls']['reason'] == 'not_synthetic_test_source'
    with pytest.raises(ValueError, match='No snapshot source'):
        build(live, 105., known_fields=('liveData',))  # no path can be named any more

    # CE1b: spoofed values on the synthetic source stay UNKNOWN, as do bool-as-int and malformed bases
    bad = {'balls': 'not-a-count', 'strikes': -7, 'outs': 99, 'bases': [3, 1], 'stand': 'Final', 'pitcher': True,
           'inning': 0, 'home_score': 1.0}
    _, spoof = store.ingest(GAME, feed(4, ts='20261008_000200', wall=110., state=bad), fetched=111., source=SYNTHETIC_SOURCE)
    rec = build(spoof, 115., known_fields=everything)
    assert not known(rec)
    assert {f: rec['fields'][f]['reason'] for f in ('balls', 'strikes', 'outs_when_up', 'bases', 'stand', 'pitcher',
                                                     'inning', 'home_score')} == dict.fromkeys(
        ('balls', 'strikes', 'outs_when_up', 'bases', 'stand', 'pitcher', 'inning', 'home_score'), 'invalid_synthetic_value')
    nolabel = feed(4, ts='20261008_000300', wall=120.)
    nolabel['syntheticPrePitch'] = {'balls': 1}  # no schema tag
    _, untagged = store.ingest(GAME, nolabel, fetched=121., source=SYNTHETIC_SOURCE)
    assert build(untagged, 125., known_fields=('balls',))['fields']['balls']['reason'] == 'synthetic_envelope_absent'

    # CE1c: forged KNOWN records citing the right seq/sha are refused by append_decision
    forged = build(live, 105., decision_clock='observer_local')
    forged['fields']['balls'] = {'value': 3, 'status': 'KNOWN', 'reason': None, 'provenance': {
        'source': 'selected_snapshot', 'path': ['syntheticPrePitch', 'balls'], 'store_seq': live['seq'],
        'slot_version': 1, 'feed_sha256': live['sha'], 'version_received_at': iso(101.),
        'receipt_clock': 'observer_local', 'evidence': 'synthetic_test_state_not_provider'}}
    with pytest.raises(ValueError, match='not_synthetic_test_source'):
        store.append_decision(forged)
    linescore = copy.deepcopy(forged)
    linescore['fields']['balls']['provenance']['path'] = ['liveData', 'linescore', 'balls']
    with pytest.raises(ValueError, match='fixed synthetic'):
        store.append_decision(linescore)
    _, good = store.ingest(GAME, feed(5, ts='20261008_000400', wall=130., state={'balls': 1, 'outs': 0}), fetched=131.,
                           source=SYNTHETIC_SOURCE)
    honest = build(good, 135., known_fields=('balls', 'outs_when_up'))
    assert known(honest) == {'balls', 'outs_when_up'}
    for value in (2, True, 99):
        lie = copy.deepcopy(honest)
        lie['fields']['balls']['value'] = value
        with pytest.raises(ValueError, match='stored synthetic value'):
            store.append_decision(lie)
    assert store.decision_history(honest['decision_id']) is None
    assert store.append_decision(honest) == 'appended'

    # CE2: decision after (1,1); accepted (1,2) then (1,3) arrive; citing (1,3) is NOT_NEXT, nothing is verified next
    s2 = LiveSnapshotStore(tmp_path/'ce2.sqlite3')
    _, d0 = s2.ingest(GAME, feed(1, ts='20261008_000100', wall=100.), fetched=101., source='live_poll')
    dec = build(d0, 105., decision_clock='observer_local')
    s2.append_decision(dec)
    original = s2.decision_history(dec['decision_id'])['decision']
    with pytest.raises(ValueError, match='after the decision'):
        s2.append_reconciliation(dec['decision_id'], reconciled_at=iso(106.),
                                 evidence=[s2.pitch_events(GAME)['events'][0]['id']])
    s2.ingest(GAME, feed(2, ts='20261008_000200', wall=110.), fetched=111., source='live_poll')
    s2.ingest(GAME, feed(3, ts='20261008_000300', wall=120.), fetched=121., source='live_poll')
    rows = {(r['pitch_number'], r['outcome']): r for r in s2.pitch_events(GAME)['events']}
    nxt, later = rows[(2, 'new')], rows[(3, 'new')]
    skip = s2.append_reconciliation(dec['decision_id'], reconciled_at=iso(130.), evidence=[later['id']])
    assert 'target' not in skip and skip['candidate_target']['status'] == 'NOT_NEXT'
    assert skip['candidate_target']['reason'] == 'intervening_accepted_new_key'
    assert skip['candidate_target']['intervening'] == [{'pitch_event_id': nxt['id'], 'at_bat_number': 1,
                                                        'pitch_number': 2, 'first_received': 111.}]
    assert skip['candidate_observation'] == {'pitch_event_ids': [later['id']], 'first_received': 121.,
                                             'receipt_clock': 'observer_local'}
    other = s2.append_reconciliation(dec['decision_id'], reconciled_at=iso(131.), evidence=[nxt['id']])
    assert other['candidate_target']['status'] == 'UNVERIFIED_NEXT'
    assert other['previous'] == {'revision': 1, 'at_bat_number': 1, 'pitch_number': 3,
                                 'relation': 'supersedes_different_key'}
    history = LiveSnapshotStore(tmp_path/'ce2.sqlite3').decision_history(dec['decision_id'])
    assert history['decision'] == original and history['record']['target']['status'] == 'UNKNOWN'
    assert [(r['revision'], r['candidate_target']['pitch_number'], r['candidate_target']['status'])
            for r in history['reconciliations']] == [(1, 3, 'NOT_NEXT'), (2, 2, 'UNVERIFIED_NEXT')]
    assert all(r['candidate_target']['status'] != 'VERIFIED' and 'target' not in r for r in history['reconciliations'])
