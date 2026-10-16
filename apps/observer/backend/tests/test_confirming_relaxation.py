"""확인 중 relaxation (Song, 2026-10-12) and provider pitch-identity drops (849831), on synthetic bodies only.

Rules under test:
1. A pitch key the pitch ledger accepted at least once keeps showing its LAST ACCEPTED body when the served body is
   refused (``held_last_accepted``); only a key never accepted is 확인 중 (blank).
2. The game-level same-stamp gate holds only when the conflicting bodies differ in the pending decision's inputs
   (``live_armb.state_key``); extras (weather, attendance, boxscore) do not gate; a game with 0 pitches never gates.
3. A re-accepted pitch whose shown content did not change is not ``corrected`` (849831: 257/257 were, from
   ``endTime``/``typeConfidence``/break-term rewrites).
4. The two 849831 identity-drop patterns: receipt seq 239 (every pitch lost playId/type/location for one version)
   and seq 310 (earlier pitches lost ids, six were renamed). Held pitches are shown from accepted bodies only; a
   held or unconfirmed pitch of the CURRENT at-bat withholds the recommendation, earlier at-bats do not.
5. Immutable first receipt and append-only ledgers are unchanged.
"""
import copy
import sqlite3

import pytest

from observer_app import display_state, publish_audit
from observer_app.live_store import LiveSnapshotStore, pitch_view, view_changed
from test_display_state import GAME, LIVE, Policy, Run, shown, snap


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)

T0, T1, T2, T3 = '20261005_100000', '20261005_100100', '20261005_100200', '20261005_100300'


def pitches_of(body):
    """Every pitch event of ``body`` by official key (``currentPlay`` is the last ``allPlays`` entry in ``snap``)."""
    return {(play['about']['atBatIndex']+1, event['pitchNumber']): event
            for play in body['liveData']['plays']['allPlays'] for event in play['playEvents'] if event.get('isPitch')}


def strip(event):
    """What the provider sent on 849831 receipt seq 239 for each pitch: no playId, no type, no plate location or
    speed, a rewritten startTime."""
    event.pop('playId', None)
    event['details'] = {k: v for k, v in event['details'].items() if k != 'type'}
    event['pitchData'] = {'coordinates': {'x': 118.64, 'y': 184.14}, 'breaks': {}}
    event['startTime'] = '2026-10-05T10:00:59.000Z'


def bookkeeping(event):
    """A provider rewrite that changes nothing a viewer is shown: what 849831's final body did to every pitch."""
    event['endTime'] = '2026-10-05T10:05:00.000Z'
    data = event.setdefault('pitchData', {})
    data['typeConfidence'] = .91
    data.setdefault('breaks', {}).update(breakAngle=12.0, breakLength=6.0, breakY=24.0)


def accepted_views(store):
    views = {}
    for row in store.pitch_events(GAME)['events']:
        if row['outcome'] != 'quarantine':
            views.setdefault((row['at_bat_number'], row['pitch_number']), []).append(row['view'])
    return views


def never_accepted_exposed(view, store):
    """Pitches shown as received whose type/speed/location no accepted ledger body of their key has (must be 0)."""
    audit = publish_audit.PublishAudit(accepted_views=lambda pk: accepted_views(store))
    audit._accepted('v1/game/x.json', view)
    return len(audit.violations)


def ledger_rows(store):
    with sqlite3.connect(store.path) as db:
        return db.execute('SELECT id, outcome, reason, sha, first_received FROM pitch_events WHERE game_pk=? ORDER BY id',
                          (GAME,)).fetchall()


# ---------- 1 + 4a: receipt seq 239 pattern ----------
def test_every_pitch_losing_its_identity_for_one_version_stays_shown_with_its_accepted_body(run):
    before = run.poll()
    assert before['display_state']['state'] == 'recommended'
    rows_before, recs_before, calls = ledger_rows(run.feed.store), len(run.feed.store.served_recommendations(GAME)), list(run.policy.calls)
    dropped = snap(995.0, T1)
    for event in pitches_of(dropped).values():
        strip(event)
    view = run.poll(dropped)
    receipts = [p['receipt'] for p in view['pitches']]
    assert len(receipts) == 9 and receipts == [{'status': 'received', 'reason': display_state.HELD, 'corrected': False}]*9
    assert [p['actual'] for p in view['pitches']] == [p['actual'] for p in before['pitches']]  # the accepted bodies
    assert view['receipts']['unconfirmed'] == 0 and never_accepted_exposed(view, run.feed.store) == 0
    # the open at-bat's only pitch is held: the policy would read the stripped body, so nothing is recommended
    state = view['display_state']
    assert (state['state'], state['reason'], state['detail']) == ('unconfirmed', 'prior_pitch_unconfirmed', '5:1/incomplete_identity')
    assert shown(view) == [] and run.policy.calls == calls
    assert len(run.feed.store.served_recommendations(GAME)) == recs_before
    # the ledger only grew: every earlier row is unchanged, the refused bodies are quarantined
    rows = ledger_rows(run.feed.store)
    assert rows[:len(rows_before)] == rows_before
    added = {(o, r) for _, o, r, *_ in rows[len(rows_before):]}  # identical refused bodies are kept once (ledger dedup)
    assert added == {('quarantine', 'incomplete_identity')}
    # the legacy route lists served bodies: a held pitch's refused body is blanked there
    legacy = run.poll(path=LIVE)
    assert {item['receipt'] for item in legacy['pitch_sequence']} == {'unconfirmed'} and legacy['recommendation'] is None

    restored = snap(995.0, T2)  # ~11 s later the provider sends the pitches again, ids back, end times rewritten
    for event in pitches_of(restored).values():
        bookkeeping(event)
    again = run.poll(restored)
    assert {(p['receipt']['status'], p['receipt']['reason'], p['receipt']['corrected']) for p in again['pitches']} == {('received', None, False)}
    assert again['display_state']['state'] == 'recommended' and shown(again) == ['SL'] and never_accepted_exposed(again, run.feed.store) == 0
    assert again['receipts']['corrected'] == 0  # re-accepted with the same shown content: not a correction
    assert sum(o == 'correction' for _, o, *_ in ledger_rows(run.feed.store)) == 9  # ...although each was re-accepted


def test_a_held_pitch_in_an_earlier_at_bat_does_not_withhold_the_current_recommendation(run):
    run.poll()
    body = snap(995.0, T1)
    for (ab, _), event in pitches_of(body).items():
        if ab < 5:
            strip(event)
    view = run.poll(body)
    assert [p['receipt']['reason'] for p in view['pitches']] == [display_state.HELD]*8 + [None]
    assert view['display_state']['state'] == 'recommended' and shown(view) == ['SL']


# ---------- 1 + 4b: receipt seq 310 pattern ----------
def test_ids_dropped_and_renamed_in_earlier_at_bats_with_a_new_unidentified_pitch(run):
    before = run.poll()
    body = snap(995.0, T1, second=True)  # pitch 5:2 arrives for the first time, without a provider id
    events = pitches_of(body)
    for key in ((1, 1), (2, 1), (2, 2), (4, 2)):
        strip(events[key])
    events[(3, 1)]['playId'] = 'renamed-3-1'  # same key, new provider id: ambiguous_remap
    del events[(5, 2)]['playId']
    view = run.poll(body)
    by_key = {(p['at_bat_number'], p['pitch_number']): p for p in view['pitches']}
    held = {key for key, p in by_key.items() if p['receipt']['reason'] == display_state.HELD}
    assert held == {(1, 1), (2, 1), (2, 2), (3, 1), (4, 2)}
    old = {(p['at_bat_number'], p['pitch_number']): p['actual'] for p in before['pitches']}
    assert all(by_key[key]['actual'] == old[key] for key in held)
    fresh = by_key[(5, 2)]  # never accepted: 확인 중, nothing of its body on screen
    assert fresh['receipt'] == {'status': 'unconfirmed', 'reason': 'incomplete_identity', 'corrected': False}
    assert fresh['actual']['pitch_type'] is None and fresh['actual']['x'] is None and fresh['actual']['result_label'] == '확인 중'
    assert view['receipts']['unconfirmed'] == 1 and never_accepted_exposed(view, run.feed.store) == 0
    state = view['display_state']  # the current at-bat's newest pitch is not admitted: no recommendation
    assert (state['state'], state['detail']) == ('unconfirmed', '5:2/incomplete_identity') and shown(view) == []
    # the renamed id never displaces the accepted one: the key stays held by its first provider id
    assert all(r['provider_event_id'] != 'renamed-3-1' for r in run.feed.store.pitch_events(GAME)['events'] if r['outcome'] != 'quarantine')


def test_a_key_never_accepted_stays_hidden_while_an_accepted_one_is_held():
    rows = [{'id': 1, 'outcome': 'new', 'at_bat_number': 1, 'pitch_number': 1, 'sha': display_state.body_sha({'a': 1}),
             'reason': None, 'store_seq': 1, 'received': 10.0, 'view': pitch_view({'a': 1})}]
    feed = {'liveData': {'plays': {'allPlays': [{'about': {'atBatIndex': 0}, 'playEvents': [
        {'isPitch': True, 'pitchNumber': 1, 'b': 'refused'}, {'isPitch': True, 'pitchNumber': 2, 'b': 'never'}]}]}}}
    receipts = display_state.pitch_receipts(feed, rows)
    assert (receipts[(1, 1)]['status'], receipts[(1, 1)]['reason'], receipts[(1, 1)]['held']['id']) == ('received', display_state.HELD, 1)
    assert receipts[(1, 1)]['held']['reason'] == 'not_accepted' and receipts[(1, 1)]['first_seq'] == 1
    assert (receipts[(1, 2)]['status'], receipts[(1, 2)]['held']) == ('unconfirmed', None)
    display_state.attach_held(receipts, lambda ids: {1: {'a': 1}})
    shown_feed = display_state.held_feed(feed, receipts)
    assert shown_feed['liveData']['plays']['allPlays'][0]['playEvents'] == [{'a': 1}, {'isPitch': True, 'pitchNumber': 2, 'b': 'never'}]
    assert feed['liveData']['plays']['allPlays'][0]['playEvents'][0]['b'] == 'refused'  # the served body is not modified
    assert display_state.held_feed(feed, display_state.pitch_receipts(feed, [])) is feed  # nothing held: same object


@pytest.mark.parametrize('load', [lambda ids: (_ for _ in ()).throw(OSError('gone')), lambda ids: {}, lambda ids: {1: {'a': 2}}])
def test_a_held_body_that_cannot_be_read_back_exactly_falls_back_to_unconfirmed(load):
    rows = [{'id': 1, 'outcome': 'new', 'at_bat_number': 1, 'pitch_number': 1, 'sha': display_state.body_sha({'a': 1}),
             'reason': None, 'store_seq': 1, 'received': 10.0}]
    feed = {'liveData': {'plays': {'allPlays': [{'about': {'atBatIndex': 0}, 'playEvents': [{'isPitch': True, 'pitchNumber': 1}]}]}}}
    receipts = display_state.attach_held(display_state.pitch_receipts(feed, rows), load)
    assert (receipts[(1, 1)]['status'], receipts[(1, 1)]['reason'], receipts[(1, 1)]['held']) == ('unconfirmed', 'not_accepted', None)
    assert display_state.held_feed(feed, receipts) is feed


def test_a_pitch_without_its_official_number_still_fails_its_whole_at_bat_closed():
    rows = [{'id': 1, 'outcome': 'new', 'at_bat_number': 1, 'pitch_number': 1, 'sha': display_state.body_sha({'a': 1}),
             'reason': None, 'store_seq': 1, 'received': 10.0}]
    feed = {'liveData': {'plays': {'allPlays': [{'about': {'atBatIndex': 0}, 'playEvents': [
        {'isPitch': True, 'pitchNumber': 1, 'b': 1}, {'isPitch': True}]}]}}}
    receipts = display_state.pitch_receipts(feed, rows)
    assert (receipts[(1, 1)]['status'], receipts[(1, 1)]['held']) == ('unconfirmed', None)


# ---------- 2: same-stamp gate only on decision inputs; never at 0 pitches ----------
def test_a_same_stamp_conflict_only_in_extras_does_not_gate(run):
    assert run.poll()['display_state']['state'] == 'recommended'
    extras = snap(995.0, T0)
    extras['gameData']['weather'] = {'condition': 'Cloudy', 'temp': '61'}
    extras['liveData']['boxscore'] = {'info': [{'label': 'Att', 'value': '41,000.'}]}
    view = run.poll(extras)
    state = view['display_state']
    assert state['feed']['provider_order'] == 'same_stamp_conflict'
    assert (state['state'], state['reason']) == ('recommended', 'ledger_recorded') and shown(view) == ['SL']


@pytest.mark.parametrize('change', ['runner', 'count', 'pitch_body'])
def test_a_same_stamp_conflict_in_a_decision_input_still_gates(run, change):
    run.poll()
    body = snap(995.0, T0)
    if change == 'runner':
        body['liveData']['linescore']['offense']['first'] = {'id': 77}
    elif change == 'count':
        body['liveData']['plays']['currentPlay']['count']['strikes'] = 1
    else:
        body['liveData']['plays']['currentPlay']['playEvents'][0]['pitchData']['startSpeed'] = 88.0
    state = run.poll(body)['display_state']
    assert (state['state'], state['reason']) == ('unconfirmed', 'same_stamp_conflict_provider_stamp')


def zero_pitch(wall, ts, *, batter=32):
    """A live game whose first at-bat has begun and no pitch has been thrown."""
    body = snap(wall, ts)
    plays = body['liveData']['plays']
    first = copy.deepcopy(plays['allPlays'][0])
    first['about'] |= {'atBatIndex': 0, 'isComplete': False}
    first['playEvents'], first['result'], first['count'] = [], {}, {'balls': 0, 'strikes': 0, 'outs': 0}
    first['matchup']['batter'] = {'id': batter}
    plays['allPlays'], plays['currentPlay'] = [first], first
    body['liveData']['linescore'] |= {'currentInning': 1, 'isTopInning': True, 'inningState': 'Top', 'outs': 0,
                                      'offense': {'batter': {'id': batter}}}
    return body


class ZeroPolicy(Policy):
    def __call__(self, snapshot):
        key = display_state.pending_key(snapshot)
        self.calls.append(key)
        pre = {'status': 'ready', 'reason': None, 'recommendation': {'candidates': [{'rank': 1, 'pitch_type': 'SL', 'pitch_label': '슬라이더',
                                                                                    'zone_id': 'low_away', 'zone_label': '바깥쪽 낮게'}]}}
        return {'key': key, 'pitch': {'key': key, 'pre': pre, 'status': 'SUPPORTED'}, 'pa_pitches': [], 'previous': None,
                'home_we_now': .5, 'policy_identity': 'a6dffaea'+'0'*56, 'seconds': {'bind': 0.0}}


def test_a_game_with_no_pitch_yet_is_never_held_back_by_a_same_stamp_conflict(tmp_path):
    run = Run(tmp_path, first=zero_pitch(995.0, T0))
    run.policy = ZeroPolicy()
    first = run.poll()
    assert display_state.pitch_count(run.feed.newest(GAME)['feed']) == 0 and first['pitches'] == []
    assert first['display_state']['state'] == 'recommended'
    view = run.poll(zero_pitch(995.0, T0, batter=33))  # same stamp, a decision input differs: no pitch yet, no gate
    state = view['display_state']
    assert state['feed']['provider_order'] == 'same_stamp_conflict' and state['state'] == 'recommended'


def test_stale_and_unorderable_stamps_still_gate_a_game_with_no_pitch(tmp_path):
    run = Run(tmp_path, first=zero_pitch(995.0, T1))
    run.policy = ZeroPolicy()
    run.poll()
    state = run.poll(zero_pitch(1010.0, T0))['display_state']  # newer event under an older stamp
    assert (state['state'], state['reason']) == ('unconfirmed', 'stale_provider_stamp')


def test_same_stamp_scope_reads_only_committed_versions_and_fails_closed(tmp_path):
    store = LiveSnapshotStore(tmp_path/'live.sqlite3')
    _, a = store.ingest(GAME, snap(995.0, T0), fetched=1000.0, source='live_poll')
    weather = snap(995.0, T0)
    weather['gameData']['weather'] = {'condition': 'Rain'}
    _, b = store.ingest(GAME, weather, fetched=1010.0, source='live_poll')
    count = copy.deepcopy(weather)
    count['liveData']['plays']['currentPlay']['count']['balls'] = 3
    _, c = store.ingest(GAME, count, fetched=1020.0, source='live_poll')
    assert (a['provider_order'], b['provider_order'], c['provider_order']) == ('ordered', 'same_stamp_conflict', 'same_stamp_conflict')
    assert [store.same_stamp_scope(GAME, e['seq']) for e in (a, b, c)] == [None, 'extras_only', 'inputs']
    assert store.same_stamp_scope(GAME, 999) is None

    def broken(game_pk, seq):
        raise sqlite3.OperationalError('disk I/O error')
    feed = type('F', (), {'store': type('S', (), {'same_stamp_scope': staticmethod(broken)})()})()
    assert display_state._same_stamp_scope(feed, GAME, b) is None  # unreadable: the gate stays


# ---------- 3: corrected only when shown content changed ----------
def test_re_acceptance_with_unchanged_shown_content_is_not_a_correction(tmp_path):
    store = LiveSnapshotStore(tmp_path/'live.sqlite3')
    store.ingest(GAME, snap(995.0, T0), fetched=1000.0, source='live_poll')
    rewritten = snap(995.0, T1)
    for event in pitches_of(rewritten).values():
        bookkeeping(event)
    store.ingest(GAME, rewritten, fetched=1010.0, source='live_poll')
    rows = store.pitch_events(GAME)['events']
    assert sum(r['outcome'] == 'correction' for r in rows) == 9
    receipts = display_state.pitch_receipts(rewritten, rows)
    assert {(v['status'], v['corrected']) for v in receipts.values()} == {('received', False)}

    filled = copy.deepcopy(rewritten)  # tracking that appears later fills a blank: not a change of what was shown
    for event in pitches_of(filled).values():
        event['pitchData']['extension'] = 6.4
    filled['metaData']['timeStamp'] = T2
    store.ingest(GAME, filled, fetched=1020.0, source='live_poll')
    receipts = display_state.pitch_receipts(filled, store.pitch_events(GAME)['events'])
    assert not any(v['corrected'] for v in receipts.values())

    called = copy.deepcopy(filled)  # a challenged call: what was shown changed
    called['metaData']['timeStamp'] = T3
    event = pitches_of(called)[(1, 1)]
    event['details']['call'] = {'code': 'C'}
    store.ingest(GAME, called, fetched=1030.0, source='live_poll')
    receipts = display_state.pitch_receipts(called, store.pitch_events(GAME)['events'])
    assert [k for k, v in receipts.items() if v['corrected']] == [(1, 1)]


def test_pitch_view_and_view_changed():
    body = {'details': {'call': {'code': 'B'}, 'type': {'code': 'FF'}}, 'pitchData': {'startSpeed': 95.0, 'coordinates': {'pX': .1, 'pZ': 2.0}},
            'endTime': 'a', 'playId': 'x'}
    rewritten = copy.deepcopy(body) | {'endTime': 'b', 'playId': 'y'}
    rewritten['pitchData'] |= {'typeConfidence': .9, 'breaks': {'breakAngle': 3.0}, 'extension': 6.5}
    assert not view_changed(pitch_view(body), pitch_view(rewritten))
    assert view_changed(pitch_view(rewritten), pitch_view(body))  # a value the viewer saw (extension) went away
    assert view_changed(None, pitch_view(body)) and pitch_view('x') is None
    legacy = [{'outcome': 'new'}, {'outcome': 'correction'}]  # rows without views (older callers): as before
    assert display_state._corrected(legacy) is True


# ---------- audit + immutability ----------
def test_the_audit_flags_a_received_pitch_whose_shown_body_no_ledger_row_accepted():
    views = {(1, 1): [{'type': 'FF', 'speed': 95.0, 'px': .1, 'pz': 2.0}]}
    pitch = lambda **actual: {'at_bat_number': 1, 'pitch_number': 1, 'receipt': {'status': 'received', 'reason': None, 'corrected': False},
                              'actual': {'pitch_type': 'FF', 'speed_mph': 95.0, 'x': .1, 'z': 2.0} | actual}
    audit = publish_audit.PublishAudit(accepted_views=lambda pk: views)
    audit._accepted('v1/game/1.json', {'game': {'game_pk': 1}, 'pitches': [pitch(), pitch(pitch_type=None, speed_mph=None, x=None, z=None)]})
    assert [(v['invariant'], v['pointer']) for v in audit.violations] == [('received_pitch_accepted', '/pitches/1/actual')]
    assert 'received_pitch_accepted' in publish_audit.INVARIANTS
    assert publish_audit.game_facts({'pitches': [{'receipt': {'status': 'received', 'reason': display_state.HELD}}]})['held'] == 1


def test_the_stores_stay_append_only(run):
    run.poll()
    dropped = snap(995.0, T1)
    for event in pitches_of(dropped).values():
        strip(event)
    run.poll(dropped)
    with sqlite3.connect(run.feed.store.path) as db:
        for statement in ('UPDATE pitch_events SET outcome=outcome', 'DELETE FROM pitch_events', 'UPDATE snapshots SET seq=seq'):
            with pytest.raises(sqlite3.DatabaseError):
                db.execute(statement)
        firsts = db.execute("SELECT first_received FROM pitch_events WHERE outcome='new' ORDER BY id").fetchall()
    assert len(firsts) == 9 and all(value is not None for value, in firsts)


# ---------- REVIEW-001 C1: the policy's context inputs are decision inputs (gate and result cache) ----------
def _score_runner(body):
    """An earlier play's runner now scores: only the replayed score (home_score/away_score) changes."""
    body['liveData']['plays']['allPlays'][0]['runners'][0]['movement'] |= {'start': None, 'end': 'score'}


def _score_line(body):
    body['liveData']['linescore']['teams']['home']['runs'] = 2


def _stand(body):
    body['liveData']['plays']['currentPlay']['matchup']['batSide'] = {'code': 'L'}


def _p_throws(body):
    body['liveData']['plays']['currentPlay']['matchup']['pitchHand'] = {'code': 'L'}


def _event_count(body):  # an earlier at-bat's per-event count (statcast_rows replays the count from every event)
    body['liveData']['plays']['allPlays'][1]['playEvents'][0]['count']['balls'] = 0


CONTEXT_CHANGES = {'score_from_runner': _score_runner, 'score_on_linescore': _score_line, 'stand': _stand,
                   'p_throws': _p_throws, 'per_event_count': _event_count}


@pytest.mark.parametrize('change', sorted(CONTEXT_CHANGES))
def test_a_same_stamp_conflict_in_a_policy_context_input_still_gates(run, change):
    assert run.poll()['display_state']['state'] == 'recommended'
    body = snap(995.0, T0)
    CONTEXT_CHANGES[change](body)
    state = run.poll(body)['display_state']
    assert state['feed']['provider_order'] == 'same_stamp_conflict'
    assert (state['state'], state['reason']) == ('unconfirmed', 'same_stamp_conflict_provider_stamp')


@pytest.mark.parametrize('change', sorted(CONTEXT_CHANGES))
def test_the_result_cache_misses_when_a_policy_context_input_changes(change):
    from observer_app.live_armb import state_key
    body = snap(995.0, T0)
    changed = copy.deepcopy(body)
    CONTEXT_CHANGES[change](changed)
    assert state_key(changed) != state_key(body)
    extras = copy.deepcopy(body)
    extras['gameData']['weather'] = {'condition': 'Rain'}
    extras['liveData']['boxscore'] = {'info': [{'label': 'Att', 'value': '1.'}]}
    assert state_key(extras) == state_key(body)  # extras are still not decision inputs


def test_a_score_change_on_a_new_stamp_is_computed_again(run):
    """The LiveArmB cache is keyed by ``state_key``: a later (ordered) body that changes only the score asks the
    policy again instead of serving the result computed for the old score."""
    run.poll()
    calls = len(run.policy.calls)
    body = snap(995.0, T1)
    _score_runner(body)
    assert run.poll(body)['display_state']['state'] == 'recommended' and len(run.policy.calls) == calls+1
