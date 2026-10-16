"""Past pitches of a live game carry the recommendation the ledger recorded BEFORE them, or nothing: an ambiguous
join shows nothing rather than a wrong pairing (``rec_pairing``). Also: the fan copy of the 확인 중 family, a
``currentPlay`` pitch the pitch ledger never admitted, and the between-innings payload.
Synthetic feeds and a fake policy only; no network, no model."""
import copy
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import display_state, rec_pairing, service_api
from test_display_state import GAME, LIVE, SERVICE, Run, snap

KEY = f'{GAME}:5:2'  # the pitch that is pending in ``snap()`` and thrown in ``snap(second=True)``


def thrown(ts='20261005_100100', **options):
    """The next body: the pending pitch 5:2 has been thrown."""
    return snap(1015.0, ts, second=True, **options)


def past(view, key=KEY):
    return next(p for p in view['pitches'] if p['key'] == key)['rec']


def rule(view, key=KEY):
    return past(view, key)['pairing']['rule']


def tops(rec):
    return [c['pitch_type'] for c in rec['candidates']]


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


# ---------- the join ----------
def test_a_past_pitch_carries_the_one_recommendation_recorded_before_it(run):
    before = run.poll()
    assert tops(before['next']['rec']) == ['SL'] and before['receipts']['paired'] == 0
    view = run.poll(thrown())
    rec, row = past(view), run.feed.store.served_recommendations(GAME)[0]
    assert rec == {  # the same shape ``service_rec`` gives the pending pitch
        'provenance': 'captured_live', 'status': 'ready', 'reason': None, 'recorded_at': '1970-01-01T00:16:52Z',
        'policy': {'name': 'ARM-B', 'identity': 'a6dffaea'},
        'ledger': {'status': 'recorded', 'outcome': 'first', 'decision_id': row['decision_id'], 'decided': 1012.0,
                   'decision_clock': 'observer_local'},
        'candidates': [{'rank': 1, 'pitch_type': 'SL', 'pitch_label': '슬라이더', 'zone_id': 'low_away', 'zone_label': '바깥쪽 낮게',
                        'target': None, 'detail': {'probability': None}}],
        'pairing': {'status': 'paired', 'rule': 'recorded_before_pitch_same_inputs'}}
    assert set(rec) - {'pairing'} == set(before['next']['rec'])  # nothing the pending pitch's rec does not have
    assert row['pitch_key'] == KEY and row['inputs'] == {  # what the decision was computed for is on the row
        'pitcher_id': 41, 'batter_id': 32, 'inning': 1, 'half': 'Bot', 'outs': 0, 'balls': 1, 'strikes': 0,
        'bases': 0, 'home_score': 1, 'away_score': 0}
    others = [p['rec'] for p in view['pitches'] if p['key'] != KEY]
    assert len(others) == 9 and {r['provenance'] for r in others} == {'absent'} and view['receipts']['paired'] == 1
    assert {(r['reason'], r['pairing']['rule'], r['status'], tuple(r['candidates'])) for r in others} == {
        (rec_pairing.ABSENT, 'no_recorded_recommendation', 'missing', ())}


def test_of_several_decisions_the_last_one_before_the_pitch_is_shown(run):
    assert tops(run.poll()['next']['rec']) == ['SL']
    assert tops(run.poll(snap(995.0, '20261005_100100', call='C'))['next']['rec']) == ['FF']  # 5:1 corrected: a new decision
    assert [r['pitch_key'] for r in run.feed.store.served_recommendations(GAME)] == [KEY, KEY]
    view = run.poll(thrown('20261005_100200', call='C'))
    assert tops(past(view)) == ['FF'] and past(view)['ledger']['decided'] == 1024.0


def test_the_last_decision_is_not_replaced_by_an_older_one_when_its_inputs_were_not_the_real_ones(run):
    run.poll()  # decided for a 1-0 count (5:1 a ball)...
    run.poll(snap(995.0, '20261005_100100', call='C'))  # ...then for 0-1 (5:1 corrected to a called strike)
    view = run.poll(thrown('20261005_100200'))  # the pitch arrives in a body that says 5:1 was a ball after all
    assert rule(view) == 'different_situation' and past(view)['provenance'] == 'absent'


def test_a_decision_recorded_after_the_pitch_appeared_is_ignored(run):
    run.policy.status = 'UNSUPPORTED_MID_PA'  # nothing is recorded before the pitch
    run.poll()
    view = run.poll(thrown())
    assert rule(view) == 'no_recorded_recommendation'
    first, second = run.feed.store.read_after(GAME, 0, until_event=2e9)['items']
    late = dict(game_pk=GAME, policy_identity='p', pitch_key=KEY, feed_ok=True, inputs=None,
                displayed={'status': 'ready', 'recommendation': {'candidates': [{'pitch_type': 'FF'}]}, 'policy_identity': 'p'})
    # a viewer on a long delay writes a row from the OLD snapshot after the pitch arrived, and one from the new one
    run.feed.store.append_recommendation(state_sha='late', decided=run.now[0], snapshot=first | {'seq': first['seq']}, **late)
    run.feed.store.append_recommendation(state_sha='after', decided=run.now[0], snapshot=second | {'seq': second['seq']}, **late)
    view = run.poll()
    assert rule(view) == 'recorded_after_pitch_appeared' and past(view)['candidates'] == [] and view['receipts']['paired'] == 0


@pytest.mark.parametrize('role, expected', [('pitcher', 'different_pitcher'), ('batter', 'different_batter')])
def test_a_recommendation_computed_for_another_pitcher_or_batter_is_not_paired(run, role, expected):
    assert tops(run.poll()['next']['rec']) == ['SL']  # recorded for pitcher 41 against batter 32
    body = thrown()
    body['liveData']['plays']['currentPlay']['matchup'][role] = {'id': 77, 'fullName': 'Somebody Else'}  # then replaced
    view = run.poll(body)
    assert past(view) == rec_pairing.absent(expected) and past(view)['reason'] == '이 공은 기록된 추천이 없어요.'


def test_a_pairing_that_a_later_correction_undoes_changes_the_version_an_open_screen_watches(run):
    run.poll()
    paired = run.poll(thrown())
    assert tops(past(paired)) == ['SL'] and run.poll()['receipts']['version'] == paired['receipts']['version']
    # 5:1 is corrected (ball -> called strike) after 5:2 was thrown: the count before 5:2 is not the one decided for
    undone = run.poll(thrown('20261005_100200', call='C'))
    assert rule(undone) == 'different_situation' and undone['receipts']['paired'] == 0
    again = run.poll(thrown('20261005_100300'))  # corrected back: 5:1 keeps its 정정됨 mark, so only the pairing differs
    assert tops(past(again)) == ['SL'] and again['receipts']['corrected'] == undone['receipts']['corrected']
    assert len({paired['receipts']['version'], undone['receipts']['version'], again['receipts']['version']}) == 3


def test_a_conflict_row_never_replaces_the_first_kept_content(run):
    assert tops(run.poll()['next']['rec']) == ['SL']
    run.armb.cache.clear()  # recomputed with a different result for the very same decision
    run.policy.top = 'FF'
    run.poll()
    assert [r['outcome'] for r in run.feed.store.served_recommendations(GAME)] == ['first', 'conflict']
    run.policy.top = None
    view = run.poll(thrown())
    assert tops(past(view)) == ['SL'] and past(view)['ledger']['outcome'] == 'first'


def test_an_unconfirmed_pitch_gets_no_pairing(run):
    run.poll()
    body = thrown()
    del body['liveData']['plays']['currentPlay']['playEvents'][1]['playId']  # the new pitch cannot be identified
    view = run.poll(body)
    pitch = next(p for p in view['pitches'] if p['key'] == KEY)
    assert pitch['receipt']['status'] == 'unconfirmed' and rule(view) == 'pitch_unconfirmed' and view['receipts']['paired'] == 0
    view = run.poll(thrown('20261005_100200'))  # the provider completes it: the receipt changes, so does the version
    assert tops(past(view)) == ['SL']


def test_an_unreadable_ledger_shows_nothing_and_never_an_error(run, monkeypatch):
    run.poll()
    good = run.poll(thrown())
    assert tops(past(good)) == ['SL']

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(run.feed.store, 'served_recommendations', broken)
    view = run.poll()
    assert {p['rec']['pairing']['rule'] for p in view['pitches']} == {'ledger_unreadable'} and view['receipts']['paired'] == 0
    assert {p['rec']['reason'] for p in view['pitches']} == {rec_pairing.ABSENT}
    # an open screen compares ``receipts.version``: it changes while pairings are withheld and comes back after
    assert view['receipts']['version'] != good['receipts']['version']
    monkeypatch.undo()
    back = run.poll()
    assert tops(past(back)) == ['SL'] and back['receipts']['version'] == good['receipts']['version']


def test_an_unreadable_pitch_ledger_pairs_nothing_and_marks_the_version(run, monkeypatch):
    run.poll()
    good = run.poll(thrown())

    def broken(game_pk):
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(run.feed.store, 'pitch_events', broken)
    view = run.poll()  # the bodies admitted before stay confirmed, but when each first appeared is not known
    assert {p['receipt']['status'] for p in view['pitches']} == {'received'}
    assert rule(view) == 'pitch_first_seen_unknown' and view['receipts']['version'] != good['receipts']['version']


def test_without_a_store_nothing_is_paired(tmp_path):
    run = Run(tmp_path, store=False)
    assert tops(run.poll()['next']['rec']) == ['SL']
    view = run.poll(thrown())
    assert view['receipts'] is None and {(p['rec']['provenance'], p['rec']['reason'], p['rec']['pairing']['rule'])
                                         for p in view['pitches']} == {('absent', rec_pairing.ABSENT, 'no_store')}


def test_with_the_ledger_off_nothing_is_paired(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_REC_LEDGER', '0')
    run = Run(tmp_path)
    run.poll()
    assert rule(run.poll(thrown())) == 'no_store'


def test_a_row_written_before_inputs_were_recorded_is_checked_against_its_stored_snapshot(run, monkeypatch):
    monkeypatch.setattr(rec_pairing, 'decision_inputs', lambda parsed: None)  # as an older build wrote its rows
    run.poll()
    assert run.feed.store.served_recommendations(GAME)[0]['inputs'] is None
    monkeypatch.undo()
    assert tops(past(run.poll(thrown()))) == ['SL']  # pitcher, batter and count read back from the committed body
    replaced = thrown('20261005_100200')
    replaced['liveData']['plays']['currentPlay']['matchup']['pitcher'] = {'id': 77}
    run.start()  # a new process: nothing remembered
    assert rule(run.poll(replaced)) == 'different_pitcher'


def test_the_cursor_sends_the_new_pitch_with_its_pairing_and_nothing_older(run):
    run.poll()
    run.poll(thrown())
    tail = run.client.get(SERVICE, params={'after': 8}).json()
    assert [(p['index'], p['rec']['provenance']) for p in tail['pitches']] == [(9, 'captured_live')] and tail['receipts']['paired'] == 1


# ---------- the rules, one by one ----------
def pitch(at_bat, number, *, pitcher=41, batter=32, balls=0, strikes=0):
    return {'key': f'{GAME}:{at_bat}:{number}', 'at_bat_number': at_bat, 'pitch_number': number,
            'pitcher': {'id': pitcher}, 'batter': {'id': batter},
            'situation_before': {'inning': 1, 'half': 'Top', 'outs': 0, 'balls': balls, 'strikes': strikes,
                                 'bases': 0, 'home_score': 0, 'away_score': 0}}


def receipt(seq, *, status='received', received=None):
    return {'status': status, 'reason': None, 'sha': 'x', 'corrected': False, 'first_seq': seq,
            'first_received': seq*10.0 if received is None else received}


def ledger(row_id, at_bat, number, seq, *, top='SL', outcome='first', decided=None, balls=0, strikes=0, pitcher=41, batter=32):
    return {'id': row_id, 'outcome': outcome, 'pitch_key': f'{GAME}:{at_bat}:{number}', 'decision_id': f'd{row_id}',
            'decided': seq*10.0+1 if decided is None else decided, 'decision_clock': 'observer_local', 'store_seq': seq,
            'feed_sha256': 's', 'policy_identity': 'a6dffaea', 'candidates': [{'pitch_type': top}], 'derived': False,
            'inputs': {'pitcher_id': pitcher, 'batter_id': batter, 'inning': 1, 'half': 'Top', 'outs': 0, 'balls': balls,
                       'strikes': strikes, 'bases': 0, 'home_score': 0, 'away_score': 0}}


def verdict(pitches, receipts, rows, served):
    out = rec_pairing.pair(pitches, receipts, rows, served)
    return {key.split(':', 1)[1]: (rec['pairing']['rule'] if rec['provenance'] == 'absent' else tops(rec)[0]) for key, rec in out.items()}


def test_a_viewer_on_a_longer_delay_gets_no_later_decision():
    pitches, receipts = [pitch(1, 1)], {(1, 1): receipt(5)}
    rows = [ledger(1, 1, 1, 3, top='SL'), ledger(2, 1, 1, 4, top='CU', decided=55.0),  # 2: written after the pitch arrived
            ledger(3, 1, 1, 7, top='FF')]  # 3: from a body newer than the pitch itself
    assert verdict(pitches, receipts, rows, 5) == {'1:1': 'SL'}
    assert verdict(pitches, receipts, rows, 9) == {'1:1': 'SL'}  # and whoever is further ahead sees the same pairing
    assert verdict(pitches, receipts, rows[1:], 9) == {'1:1': 'recorded_after_pitch_appeared'}
    # served from a version older than the pitch's first body (cannot happen with as-of receipts): nothing
    assert verdict(pitches, receipts, rows, 4) == {'1:1': 'pitch_first_seen_unknown'}
    assert verdict(pitches, receipts, rows, None) == {'1:1': 'snapshot_not_committed'}
    newer = [ledger(1, 1, 2, 6, top='SL', balls=1)]  # the next pitch's decision is newer than a viewer served at 6
    two = [pitch(1, 1), pitch(1, 2, balls=1)]
    assert verdict(two, {(1, 1): receipt(5), (1, 2): receipt(8)}, newer, 8) == {'1:1': 'no_recorded_recommendation', '1:2': 'SL'}


def test_the_newest_state_before_the_pitch_wins_whatever_the_row_order():
    rows = [ledger(1, 1, 1, 4, top='FF'), ledger(2, 1, 1, 2, top='SL')]  # row 2 was written later from an older body
    assert verdict([pitch(1, 1)], {(1, 1): receipt(5)}, rows, 5) == {'1:1': 'FF'}
    rows.append(ledger(3, 1, 1, 4, top='CU', outcome='conflict'))  # never displayable, never paired
    assert verdict([pitch(1, 1)], {(1, 1): receipt(5)}, rows, 5) == {'1:1': 'FF'}


def test_a_key_that_is_one_pitch_off_is_refused():
    """The policy numbers automatic balls/strikes; if the official number does not, its key ``:2`` names what the
    feed calls pitch 1 of a 1-0 count, and its ``:3`` what the feed calls pitch 2."""
    pitches = [pitch(1, 1, balls=1), pitch(1, 2, balls=1, strikes=1)]
    receipts = {(1, 1): receipt(5), (1, 2): receipt(7)}
    rows = [ledger(1, 1, 1, 2), ledger(2, 1, 2, 3, balls=1)]  # ':1' before the automatic ball, ':2' after it
    assert verdict(pitches, receipts, rows, 9) == {'1:1': 'different_situation', '1:2': 'previous_pitch_not_in_decision'}
    rows = [ledger(1, 1, 2, 6, balls=1, strikes=1)]  # decided when pitch 1 was already listed: this one is pitch 2's
    assert verdict(pitches, receipts, rows, 9) == {'1:1': 'no_recorded_recommendation', '1:2': 'SL'}


@pytest.mark.parametrize('field, value', [('bases', 2), ('home_score', 1), ('away_score', 1)])
def test_a_decision_made_before_a_runner_moved_or_a_run_scored_is_refused(field, value):
    thrown_ = pitch(1, 1)
    thrown_['situation_before'] |= {field: value}  # the pitch was thrown with a runner on second, or a run in
    assert verdict([thrown_], {(1, 1): receipt(5)}, [ledger(1, 1, 1, 2)], 5) == {'1:1': 'different_situation'}
    row = ledger(1, 1, 1, 2)
    del row['inputs'][field]  # a row that does not say cannot be checked
    assert verdict([pitch(1, 1)], {(1, 1): receipt(5)}, [row], 5) == {'1:1': 'decision_inputs_unknown'}


def on_first(body):
    play = body['liveData']['plays']['currentPlay']
    play['runners'] = [{'movement': {'start': None, 'end': '1B'}, 'details': {'playIndex': 0, 'runner': {'id': 99}}}]
    body['liveData']['linescore']['offense'] = {'batter': {'id': 32}, 'first': {'id': 99}}
    return body


def test_a_balk_that_arrives_with_the_pitch_leaves_it_without_a_recommendation(run):
    assert tops(run.poll(on_first(snap(995.0, '20261005_100000')))['next']['rec']) == ['SL']  # decided with a runner on first
    assert run.feed.store.served_recommendations(GAME)[-1]['inputs']['bases'] == 1
    body = on_first(thrown())
    play = body['liveData']['plays']['currentPlay']
    play['playEvents'].insert(1, {'isPitch': False, 'type': 'action', 'index': 1, 'count': dict(play['playEvents'][0]['count']),
                                  'details': {'event': 'Balk', 'eventType': 'balk', 'description': 'Balk'}})
    play['runners'].append({'movement': {'start': '1B', 'end': '2B'}, 'details': {'playIndex': 1, 'runner': {'id': 99}}})
    body['liveData']['linescore']['offense'] = {'batter': {'id': 32}, 'second': {'id': 99}}
    view = run.poll(body)  # the balk (first -> second) and the pitch arrive in ONE body
    thrown_ = next(p for p in view['pitches'] if p['key'] == KEY)
    assert thrown_['situation_before']['bases'] == 2 and thrown_['receipt']['status'] == 'received'
    assert rule(view) == 'different_situation' and view['receipts']['paired'] == 0


def test_unknown_inputs_or_an_unknown_first_appearance_refuse():
    row = ledger(1, 1, 1, 2) | {'inputs': None}
    assert verdict([pitch(1, 1)], {(1, 1): receipt(5)}, [row], 5) == {'1:1': 'decision_inputs_unknown'}
    assert verdict([pitch(1, 1)], {(1, 1): receipt(None, received=1.0)}, [ledger(1, 1, 1, 2)], 5) == {'1:1': 'pitch_first_seen_unknown'}
    assert verdict([pitch(1, 1)], {(1, 1): receipt(5, status='unconfirmed')}, [ledger(1, 1, 1, 2)], 5) == {'1:1': 'pitch_unconfirmed'}
    assert verdict([pitch(1, 1)], {}, [ledger(1, 1, 1, 2)], 5) == {'1:1': 'pitch_unconfirmed'}
    tainted = {(1, 1): receipt(5), display_state.RECEIPT_ALL: receipt(None, status='unconfirmed', received=1.0)}
    assert verdict([pitch(1, 1)], tainted, [ledger(1, 1, 1, 2)], 5) == {'1:1': 'pitch_unconfirmed'}
    assert set(rec_pairing.TRANSIENT) <= set(rec_pairing.RULES)


def test_the_ledger_is_read_once_per_request_and_only_for_new_rows(run, monkeypatch):
    run.poll()
    run.poll(thrown())
    calls, real = [], run.feed.store.served_recommendations
    monkeypatch.setattr(run.feed.store, 'served_recommendations', lambda game_pk, after=0: calls.append(after) or real(game_pk, after))
    monkeypatch.setattr(run.feed.store, 'snapshot_feed', lambda *a: pytest.fail('a row that carries its inputs needs no snapshot read'))
    assert tops(past(run.client.get(SERVICE).json())) == ['SL']
    assert len(calls) == 1 and calls[0] == real(GAME)[-1]['id']  # one query, asking only for rows after the last held


# ---------- (d) the 확인 중 family in a first-time fan's words ----------
def test_the_unconfirmed_copy_is_exactly_the_agreed_two_sentences():
    game, ball = '경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.', '방금 던진 공을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.'
    expected = {'stale_provider_stamp': game, 'unorderable_provider_stamp': game, 'same_stamp_conflict_provider_stamp': game,
                'provider_order_unknown': game, 'receipts_unreadable': game, 'prior_pitch_unconfirmed': ball}
    assert {reason: copy_ for (state, reason), copy_ in display_state.COPY.items() if state == 'unconfirmed'} == {
        reason: ('확인 중', sentence) for reason, sentence in expected.items()}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in ('detail', 'reason_code'):  # machine fields are never rendered
                yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def fan_text(view):
    """Every server-authored sentence the fan screen can render for the pending slot and the pitch list."""
    state = view['display_state']
    parts = [state['label'], state['message'], (state.get('correction') or {}).get('message'),
             ((view.get('next') or {}).get('rec') or {}).get('reason')]
    for item in view.get('pitches', []):
        parts += [item['rec']['reason'], item['actual']['result_label'], item['actual']['call_label']]
    parts += [pa.get('result_text') for pa in view.get('pas', [])]
    return [part for part in parts if isinstance(part, str)]


def test_no_unconfirmed_family_message_uses_a_pipeline_word(tmp_path):
    assert display_state.PIPELINE_WORDS == ('자료', '시각', '순서', '격리', '스냅샷', '원장')
    fixed = [message for (state, _), (_, message) in display_state.COPY.items() if state == 'unconfirmed']
    fixed += [*display_state.CORRECTION_COPY.values(), service_api.UNCONFIRMED_LABEL, rec_pairing.ABSENT, '확인 중']
    served = []
    bodies = {'stale_provider_stamp': snap(1010.0, '20261005_095900', second=True), 'unorderable_provider_stamp': snap(1010.0, None, second=True),
              'same_stamp_conflict_provider_stamp': snap(1010.0, '20261005_100000', second=True),
              'prior_pitch_unconfirmed': snap(995.0, '20261005_100100', pid='renamed')}
    for index, (reason, body) in enumerate(bodies.items()):
        run = Run(tmp_path/str(index))
        assert tops(run.poll()['next']['rec']) == ['SL']
        view = run.poll(body)
        state = view['display_state']
        assert (state['state'], state['reason'], state['label']) == ('unconfirmed', reason, '확인 중'), reason
        assert view['next']['rec']['reason'] == state['message']
        if reason == 'prior_pitch_unconfirmed':  # same pending pitch as before: the withdrawal note is in the payload too
            # the renamed pitch was accepted before: held with its accepted body (Song 2026-10-12), not 확인 중
            assert state['correction']['kind'] == 'withdrawn' and view['pitches'][-1]['receipt']['reason'] == display_state.HELD
        served += fan_text(view)
        legacy = run.poll(path=LIVE)
        served += [legacy['display_state']['message'], *(strings(legacy['recommendation']) if legacy['recommendation'] else [])]
    assert len(served) > 40
    for sentence in [*fixed, *served]:
        assert not [word for word in display_state.PIPELINE_WORDS if word in sentence], sentence


# ---------- (f) a ``currentPlay`` pitch the pitch ledger never admitted ----------
def unlisted(body):
    """A trimmed feed: the open at-bat is ``currentPlay`` only, ``allPlays`` stops at the previous one."""
    plays = body['liveData']['plays']
    plays['currentPlay'] = copy.deepcopy(plays['allPlays'][-1])
    plays['allPlays'] = plays['allPlays'][:-1]
    return body


def test_a_current_play_pitch_without_an_admission_row_is_not_shown_as_observed(tmp_path):
    run = Run(tmp_path, first=unlisted(snap(995.0, '20261005_100000')))
    live = run.poll(path=LIVE)
    state = live['display_state']
    assert (state['state'], state['reason'], state['detail']) == ('unconfirmed', 'prior_pitch_unconfirmed', '5:1/not_in_pitch_ledger')
    assert state['message'] == display_state.RECHECK_PITCH and state['recommendation_shown'] is False
    assert live['pitch_sequence'] == [{key: None for key in live['pitch_sequence'][0]} | {'pitch_number': 1, 'receipt': 'unconfirmed'}]
    assert live['recommendation'] is None and run.feed.store.served_recommendations(GAME) == []  # nothing recorded on that input
    fan = run.poll()  # the fan route lists ``allPlays`` only, and withholds the recommendation for the same reason
    assert (fan['display_state']['state'], fan['display_state']['reason']) == ('unconfirmed', 'prior_pitch_unconfirmed')
    assert fan['next']['rec']['candidates'] == [] and [p['key'] for p in fan['pitches']][-1] == f'{GAME}:4:2'
    assert run.feed.store.served_recommendations(GAME) == []


def test_a_current_play_pitch_that_differs_from_its_listed_body_is_not_shown_as_observed(tmp_path):
    body = snap(995.0, '20261005_100000')
    plays = body['liveData']['plays']
    plays['currentPlay'] = copy.deepcopy(plays['allPlays'][-1])
    plays['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'CU'}  # not the body the ledger admitted
    run = Run(tmp_path, first=body)
    live = run.poll(path=LIVE)
    assert live['display_state']['detail'] == '5:1/body_differs_from_listed' and live['recommendation'] is None
    assert live['pitch_sequence'][0]['pitch_type'] is None and live['pitch_sequence'][0]['receipt'] == 'unconfirmed'


def test_a_complete_current_play_that_is_not_listed_blocks_the_next_at_bat_too(tmp_path):
    body = unlisted(snap(995.0, '20261005_100000'))
    body['liveData']['plays']['currentPlay']['about']['isComplete'] = True  # between batters: the newest pitch is unadmitted
    state = Run(tmp_path, first=body).poll()['display_state']
    assert (state['state'], state['reason'], state['detail']) == ('unconfirmed', 'prior_pitch_unconfirmed', '5:1/not_in_pitch_ledger')


def test_a_current_play_that_lags_the_listed_plays_gets_no_recommendation(tmp_path):
    body = snap(995.0, '20261005_100000', second=True)  # allPlays: 5:1 and 5:2 were thrown
    plays = body['liveData']['plays']
    plays['currentPlay'] = copy.deepcopy(plays['allPlays'][-1])
    plays['currentPlay']['playEvents'] = plays['currentPlay']['playEvents'][:1]  # currentPlay still says only 5:1
    plays['currentPlay']['count'] = {'balls': 1, 'strikes': 0, 'outs': 0}
    run = Run(tmp_path, first=body)
    for path in (SERVICE, LIVE):
        state = run.poll(path=path)['display_state']
        assert (state['state'], state['reason'], state['detail']) == (
            'unconfirmed', 'prior_pitch_unconfirmed', '5:2/current_play_lags_listed'), path
        assert state['recommendation_shown'] is False and state['message'] == display_state.RECHECK_PITCH
    assert run.feed.store.served_recommendations(GAME) == []  # nothing recorded on a state that is already over


def test_a_current_play_that_is_an_older_at_bat_gets_no_recommendation(tmp_path):
    body = snap(995.0, '20261005_100000')
    plays = body['liveData']['plays']
    plays['currentPlay'] = copy.deepcopy(plays['allPlays'][-3])  # a finished, older at-bat
    run = Run(tmp_path, first=body)
    for path in (SERVICE, LIVE):
        state = run.poll(path=path)['display_state']
        assert (state['state'], state['reason'], state['detail']) == (
            'unconfirmed', 'prior_pitch_unconfirmed', '5:1/current_play_lags_listed'), path
    assert run.feed.store.served_recommendations(GAME) == []
    empty = snap(995.0, '20261005_100000')  # same pitches, but another at-bat: named by the at-bat itself
    empty['liveData']['plays']['allPlays'][-1]['playEvents'] = []
    empty['liveData']['plays']['currentPlay'] = copy.deepcopy(empty['liveData']['plays']['allPlays'][-2])
    empty['liveData']['plays']['currentPlay']['playEvents'] = []
    assert display_state.unlisted_current(empty, {}) == {(4, None): 'current_play_not_last_listed'}


def test_the_real_feed_shape_and_a_run_without_a_store_are_unchanged(tmp_path):
    listed = Run(tmp_path/'a').poll(path=LIVE)  # ``currentPlay`` is the last ``allPlays`` entry, as MLB sends it
    assert listed['display_state']['state'] == 'recommended' and listed['pitch_sequence'][0]['pitch_type'] == 'FF'
    assert 'receipt' not in listed['pitch_sequence'][0]
    plain = Run(tmp_path/'b', store=False, first=unlisted(snap(995.0, '20261005_100000'))).poll(path=LIVE)
    assert plain['display_state']['state'] == 'recommended' and plain['pitch_sequence'][0]['pitch_type'] == 'FF'
    assert display_state.unlisted_current(unlisted(snap(995.0, '20261005_100000')), None) == {}


# ---------- (b) between innings ----------
def test_between_innings_names_nobody_and_keeps_the_score(tmp_path):
    view = Run(tmp_path, first=snap(995.0, '20261005_100000', half='between')).poll(settle=False)
    current = dict(view['current'])
    context = current.pop('context')
    assert current == {'phase': 'between_innings', 'situation': None, 'batter': None, 'pitcher': None, 'pa_key': None,
                       'finished_half': {'inning': 1, 'half': 'Bot'}}
    assert context['next_batters'] == [] and set(context['batter'].values()) == {None} and set(context['pitcher'].values()) == {None}
    assert view['next'] is None and (view['display_state']['state'], view['display_state']['reason']) == ('waiting', 'between_innings')
    card = view['game']
    assert (card['away_score'], card['home_score'], card['inning'], card['half'], card['status']) == (0, 1, None, None, 'live')
    middle = snap(995.0, '20261005_100000', half='between')
    middle['liveData']['linescore'] |= {'inningState': 'Middle', 'currentInning': 3}
    assert service_api._finished_half(middle) == {'inning': 3, 'half': 'Top'}
    assert service_api._finished_half({'liveData': {'linescore': {'inningState': 'Middle'}}}) is None


# ---------- review round 1 ----------
def test_what_the_fan_reads_under_a_withheld_recommendation_is_the_states_sentence(run):
    run.policy.status = run.policy.reason = 'UNSUPPORTED_BRAND_NEW_CODE'  # a policy status this build does not know
    view = run.poll()
    state = view['display_state']
    assert (state['state'], state['reason'], state['recommendation_shown']) == ('unsupported', 'policy_refused', False)
    assert view['next']['rec']['reason'] == state['message'] == '이 상황은 추천 대상이 아니에요.'
    assert run.poll(path=LIVE)['recommendation']['reason'] == state['message']  # never the bare code


def test_a_disconnect_before_the_first_ledger_write_reads_as_disconnected(run):
    run.client.get(LIVE, params={'delay': 45})  # a delayed viewer is buffering; the head start computes the newest state
    deadline = time.time()+5
    while time.time() < deadline and not run.armb.cache:
        time.sleep(.01)
    assert run.armb.cache and run.feed.store.served_recommendations(GAME) == []  # computed, never recorded
    view = run.poll(urllib.error.URLError('down'))
    state = view['display_state']
    assert (state['state'], state['recommendation_shown']) == ('disconnected', False)
    assert view['next']['rec']['reason'] == state['message'] and view['next']['rec']['candidates'] == []
    assert run.poll(path=LIVE)['recommendation']['reason'] == state['message']  # not "추천 기록을 저장하지 못해…"


def test_a_first_poll_that_fails_is_a_calm_503_not_a_generic_error(tmp_path):
    run = Run(tmp_path, store=False)
    run.upstream.body = urllib.error.URLError('down')
    answer = run.client.get(SERVICE)
    assert answer.status_code == 503 and answer.headers['cache-control'] == 'no-store'
    assert answer.json() == {'detail': 'MLB 경기 정보를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.'}


@pytest.mark.parametrize('value', ['fast', '-1', '5'])
def test_a_real_upstream_ignores_the_replay_pace_variable(tmp_path, monkeypatch, value):
    from observer_app import live_feed
    from observer_app.main import create_app
    asked = []
    monkeypatch.setattr(live_feed, 'http_json', lambda url, timeout, **_: asked.append(url) or snap(time.time()-5, '20261005_100000'))
    for name in ('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', 'PITCHEEZY_OBSERVER_LIVE_STORE', 'PITCHEEZY_OBSERVER_LIVE_DELAY_S'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S', value)  # stray, even invalid: this is not a replay
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    monkeypatch.setattr(live_feed.LiveFeed, '_record', lambda *a, **k: None, raising=False)  # no snapshot files from a test
    app = create_app(object(), start_worker=False, watch_dir=tmp_path, feed_dir=tmp_path, demo_dir=tmp_path/'demo', intent_dir=tmp_path)
    answer = TestClient(app).get(SERVICE)
    assert answer.status_code == 200 and len(asked) == 1 and answer.json()['display_state']['reason'] == 'recommender_off'
