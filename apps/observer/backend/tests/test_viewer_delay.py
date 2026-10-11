"""C2 viewer delay on the fan route: ``/api/service/games/{pk}?delay=`` serves the game as it stood for a viewer
whose broadcast runs that far behind, and nothing newer; buffering; the owner switches; the list; D1 per delay;
record at receipt; the tap sync. Synthetic feeds and a fake policy only; no network, no model."""
import copy
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import features, live_feed as live_feed_module
from observer_app.live_feed import TAP, pitch_arrivals, tap_sync
from test_display_state import GAME, LIVE, SERVICE, EventPolicy, Run, brief, shown, snap, with_pickoff
from test_rec_pairing import KEY, past, rule, thrown, tops

TAP_URL = f'{SERVICE}/tap'


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def settle(run):
    deadline = time.time()+5
    while time.time() < deadline and (run.armb.pending or run.armb.running is not None or run.armb.loading):
        time.sleep(.01)


def ask(run, delay=None, *, after=0.0, body=None, path=SERVICE, again=True, cursor=None, **params):
    """Move the clock, optionally change upstream, and ask the fan route as a viewer on ``delay`` (twice: the fake
    policy worker gets its instant turn in between, as two polls of a real screen would give it)."""
    if body is not None:
        run.upstream.body = body
    run.now[0] += after
    params = params | ({} if delay is None else {'delay': delay}) | ({} if cursor is None else {'after': cursor})
    answer = run.client.get(path, params=params)
    if again:
        settle(run)
        answer = run.client.get(path, params=params)
    assert answer.status_code == 200, answer.text
    return answer.json()


def scored(body, runs=3):
    """``body`` with ``runs`` more home runs on the line: a later state must not lend its score to an earlier one."""
    body = copy.deepcopy(body)
    body['liveData']['linescore']['teams']['home']['runs'] += runs
    return body


GAME_CONTENT = ('game', 'cutoff', 'pitches', 'pas', 'current', 'linescore', 'pregame', 'zone_bounds')


# ---------- everything a delayed viewer gets is their own snapshot ----------
def test_a_delayed_viewer_gets_the_game_exactly_as_a_live_viewer_had_it_then(run):
    then = ask(run, after=12.0)  # wall 1012: state A (event 995), one pitch of the open at-bat, 5:2 pending
    assert then['viewer'] == {'enabled': True, 'default_s': 30, 'provisional': True, 'delay_s': 0.0, 'max_delay_s': 120,
                              'buffering': False, 'wait_s': 0.0}
    now = ask(run, after=12.0, body=scored(thrown()))  # wall 1024: state B (event 1015): 5:2 thrown, three more runs
    assert len(now['pitches']) == len(then['pitches'])+1 and now['game']['home_score'] == then['game']['home_score']+3
    late = ask(run, 30, after=12.0)  # wall 1036: a viewer 30 s behind is still at A (B's event is only 21 s old)
    for part in GAME_CONTENT:
        assert late[part] == then[part], part
    assert late['next']['pitch_key'] == then['next']['pitch_key'] == KEY and shown(late) == shown(then) == ['SL']
    assert late['next']['rec']['candidates'] == then['next']['rec']['candidates']
    assert late['receipts']['version'] == then['receipts']['version'] and late['receipts']['paired'] == 0
    assert late['feed']['as_of'] == then['feed']['as_of']  # the receipt of the served state, not of the newest one
    state = late['display_state']
    assert (state['state'], state['pitch_key'], state['correction']) == ('recommended', KEY, None)
    assert state['feed']['received_at'] == then['display_state']['feed']['received_at']
    assert late['viewer']['delay_s'] == 30.0 and late['viewer']['buffering'] is False
    assert json.dumps(late['game']).count('"home_score": %d' % then['game']['home_score']) == 1
    caught_up = ask(run, 30, after=10.0)  # wall 1046: B's event is now 31 s old
    for part in GAME_CONTENT:
        assert caught_up[part] == now[part], part
    assert rule(caught_up) == 'recorded_before_pitch_same_inputs' and tops(past(caught_up)) == ['SL']


def test_two_delayed_viewers_each_keep_their_own_moment(run):
    ask(run, after=12.0)
    ask(run, after=12.0, body=thrown())  # B at wall 1024 (event 1015)
    run.now[0] += 12.0  # wall 1036
    near, far = ask(run, 20), ask(run, 30)  # 20 s behind has reached B (event 21 s old); 30 s behind has not
    assert len(near['pitches']) == len(far['pitches'])+1 and near['next']['pitch_key'] != far['next']['pitch_key']
    assert far['next']['pitch_key'] == KEY and ask(run, 30)['pitches'] == far['pitches']  # the other viewer changed nothing
    assert near['viewer']['delay_s'] == 20.0 and far['viewer']['delay_s'] == 30.0


def test_the_cursor_works_within_a_delayed_view(run):
    ask(run, after=12.0)
    ask(run, after=12.0, body=thrown())
    late = ask(run, 30, after=12.0)
    last = late['pitches'][-1]['index']
    assert ask(run, 30, cursor=last)['pitches'] == []  # nothing newer than this viewer's own last pitch
    assert [p['index'] for p in ask(run, 30, after=10.0, cursor=last)['pitches']] == [last+1]


# ---------- buffering ----------
def test_a_viewer_whose_moment_is_not_buffered_yet_waits_calmly_and_gets_nothing_newer(run):
    first = ask(run, 30, after=2.0, again=False)  # wall 1002: the only state's event (995) is 7 s old
    assert first['viewer']['buffering'] is True and first['viewer']['wait_s'] == 23.0 and first['viewer']['delay_s'] == 30.0
    assert first['pitches'] == [] and first['pas'] == [] and first['next'] is None and first['linescore'] is None
    card = first['game']
    assert (card['away_score'], card['home_score'], card['inning'], card['half']) == (None, None, None, None)
    assert card['score_hidden'] is True and card['status'] == 'live' and card['away_team'] and card['home_team']
    assert first['current'] == {'phase': None, 'situation': None, 'batter': None, 'pitcher': None, 'pa_key': None, 'context': None}
    state = first['display_state']
    assert (state['state'], state['reason'], state['recommendation_shown']) == ('waiting', 'buffering', False)
    assert state['message'] == '내 중계 시점까지 기다리는 중이에요 (약 23초).' and state['label'] == '대기 중'
    assert first['source'] == {'kind': 'live', 'session_id': None} and first['features']['viewer_delay']['enabled'] is True
    assert 'Batter' not in json.dumps(first) and 'pitch_type' not in json.dumps(first)
    assert ask(run, 30, after=22.0)['viewer']['buffering'] is True  # wall 1024: one second short
    there = ask(run, 30, after=2.0)  # wall 1026
    assert there['viewer']['buffering'] is False and there['next']['pitch_key'] == KEY and len(there['pitches']) == 9


def test_a_game_that_ended_meanwhile_is_not_announced_to_a_buffering_viewer(run):
    view = ask(run, 30, after=2.0, body=snap(995.0, '20261005_100000', half='final'), again=False)
    assert view['viewer']['buffering'] is True and view['game']['status'] == 'live' and view['game']['status_text'] is None
    assert view['display_state']['state'] == 'waiting'


def test_the_very_first_poll_failing_is_still_the_calm_503(run):
    run.upstream.body = OSError('down')
    run.now[0] += 2.0
    assert run.client.get(SERVICE, params={'delay': 30}).status_code == 503


# ---------- the owner switches ----------
def test_with_the_switch_off_the_parameter_is_ignored(run, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', '0')
    view = ask(run, 30, after=2.0)
    assert view['viewer'] | {'max_delay_s': None} == {'enabled': False, 'default_s': 30, 'provisional': True, 'delay_s': 0.0,
                                                      'max_delay_s': None, 'buffering': False, 'wait_s': 0.0}
    assert view['next']['pitch_key'] == KEY and len(view['pitches']) == 9  # served at delay 0, as before the feature
    assert view['features']['viewer_delay']['enabled'] is False
    assert set(run.feed.watching[GAME]) == {0.0}
    assert run.client.get(TAP_URL, params={'ago': '1'}).status_code == 404
    assert run.client.get('/api/service/features').json()['features']['viewer_delay']['enabled'] is False


def test_no_parameter_is_delay_zero_and_a_huge_one_is_clamped_to_the_buffer(run):
    assert ask(run, after=2.0)['viewer']['delay_s'] == 0.0
    view = ask(run, 99999)
    assert view['viewer']['delay_s'] == 120.0 and view['viewer']['max_delay_s'] == 120 and view['viewer']['buffering'] is True
    assert run.client.get(SERVICE, params={'delay': -1}).status_code == 400
    assert run.client.get(SERVICE, params={'delay': 'nan'}).status_code == 400
    assert run.client.get(SERVICE, params={'delay': 'soon'}).status_code == 400


def test_features_come_from_the_config_with_environment_overrides_and_survive_bad_values(monkeypatch, caplog):
    assert features.features() == features.DEFAULTS | {'provisional': True}
    assert features.features({'features': {'viewer_delay': {'default_s': 20, 'sync_prompt': False, 'enabled': 'yes'},
                                           'step5_expectation': {'ledger': '/x/ledger.sqlite3', 'enabled': True}}}) == {
        'viewer_delay': features.DEFAULTS['viewer_delay'] | {'default_s': 20, 'sync_prompt': False},
        'profile_reference': features.DEFAULTS['profile_reference'],
        'step5_expectation': {'enabled': True, 'ledger': '/x/ledger.sqlite3', 'allow_replayed': False}, 'provisional': True}
    for name, value in [('VIEWER_DELAY', '0'), ('VIEWER_DELAY_DEFAULT_S', '45'), ('VIEWER_SYNC_PROMPT', '0'),
                        ('PROFILE_REFERENCE', '0'), ('PROFILE_SOURCES', f'/a/one.json{features.os.pathsep}/b/two.json'),
                        ('STEP5_EXPECTATION', '1'), ('STEP5_LEDGER', '/c/ledger.sqlite3'), ('STEP5_ALLOW_REPLAYED', '1')]:
        monkeypatch.setenv(f'PITCHEEZY_OBSERVER_{name}', value)
    now = features.features()
    assert now['viewer_delay'] == features.DEFAULTS['viewer_delay'] | {'enabled': False, 'default_s': 45, 'sync_prompt': False}
    assert now['profile_reference'] == {'enabled': False, 'profile_sources': ['/a/one.json', '/b/two.json']}
    assert now['step5_expectation'] == {'enabled': True, 'ledger': '/c/ledger.sqlite3', 'allow_replayed': True}
    shown_to_fans = features.public(now)
    assert shown_to_fans == {'provisional': True, 'profile_reference': {'enabled': False}, 'step5_expectation': {'enabled': True},
                             'viewer_delay': {'enabled': False, 'default_s': 45, 'sync_prompt': False, 'fold_actual_until_synced': True}}
    assert '/' not in json.dumps(shown_to_fans)  # never a path
    for name, value in [('VIEWER_DELAY', 'maybe'), ('VIEWER_DELAY_DEFAULT_S', 'soon'), ('VIEWER_SYNC_PROMPT', '2'),
                        ('STEP5_EXPECTATION', 'TRUE')]:
        monkeypatch.setenv(f'PITCHEEZY_OBSERVER_{name}', value)
    with caplog.at_level('WARNING'):
        bad = features.features()
        features.features()
    assert bad['viewer_delay'] == features.DEFAULTS['viewer_delay'] and bad['step5_expectation']['enabled'] is False
    assert sum('PITCHEEZY_OBSERVER_VIEWER_DELAY=' in record.getMessage() for record in caplog.records) == 1  # warned once
    monkeypatch.setenv('PITCHEEZY_OBSERVER_VIEWER_DELAY_DEFAULT_S', '-5')
    assert features.features()['viewer_delay']['default_s'] == 30


# ---------- the list ----------
def test_a_live_game_card_carries_no_score_for_a_delayed_viewer(run, monkeypatch):
    row = lambda pk, abstract, status: {
        'game_pk': pk, 'game_type': 'W', 'start': '2026-10-05T10:00:00Z', 'status': status, 'live': abstract == 'Live',
        'away_team': 'A', 'home_team': 'B', 'abstract': abstract, 'official_date': '2026-10-05', 'away_score': 2, 'home_score': 5,
        'game_number': 1, 'double_header': 'N', 'inning': 7, 'half': 'Top'}
    run.feed.schedule = lambda day: ([row(1, 'Live', 'In Progress'), row(2, 'Final', 'Final'), row(3, 'Preview', 'Scheduled')]
                                     if day == '2026-10-05' else [])
    cards = lambda **params: {c['game_pk']: c for c in run.client.get('/api/service/games', params={'date': '2026-10-05'} | params).json()['games']}
    plain = cards()
    assert (plain[1]['away_score'], plain[1]['home_score'], plain[1]['inning'], plain[1]['half']) == (2, 5, 7, 'Top')
    assert 'score_hidden' not in plain[1]
    late = cards(delay=30)
    assert (late[1]['away_score'], late[1]['home_score'], late[1]['inning'], late[1]['half']) == (None, None, None, None)
    assert late[1]['score_hidden'] is True and late[1]['status'] == 'live' and late[1]['live'] is True
    assert late[2] == plain[2] and late[3] == plain[3]  # final and scheduled games are unchanged
    assert cards(delay=0) == plain
    monkeypatch.setenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', '0')
    assert cards(delay=30) == plain
    assert run.client.get('/api/service/games', params={'date': '2026-10-05', 'delay': 30}).json()['features']['viewer_delay']['enabled'] is False


# ---------- D1 per viewer delay ----------
def test_interleaved_viewers_each_see_the_change_when_it_reaches_their_own_screen(tmp_path):
    run = Run(tmp_path)
    run.policy = EventPolicy()
    run.start()
    assert brief(ask(run, after=12.0)) == ('recommended', True, None)  # wall 1012, live viewer: SL
    newer = ask(run, after=12.0, body=with_pickoff(1015.0, '20261005_100100'))  # wall 1024: same pending pitch, FF now
    assert brief(newer) == ('recommended', True, 'replaced') and shown(newer) == ['FF']
    late = ask(run, 30, after=6.0)  # wall 1030: this viewer is at the first state and has only ever seen SL
    assert brief(late) == ('recommended', True, None) and shown(late) == ['SL']
    assert brief(ask(run)) == ('recommended', True, 'replaced')  # the live viewer in between: unchanged
    assert brief(ask(run, 30, after=10.0)) == ('recommended', True, None)  # wall 1040: still SL, still no note
    turned = ask(run, 30, after=6.0)  # wall 1046: the change reaches the delayed screen, as a visible correction
    assert brief(turned) == ('recommended', True, 'replaced') and shown(turned) == ['FF']
    assert turned['display_state']['correction']['previous']['pitch_type'] == 'SL'
    assert [r['outcome'] for r in run.feed.store.served_recommendations(GAME)] == ['first', 'first']  # one row per decision


def test_a_correction_that_arrived_before_a_viewers_moment_is_not_announced_to_them(run):
    assert shown(ask(run, after=12.0)) == ['SL']  # wall 1012, live viewer (state A, event 995)
    fixed = snap(995.0, '20261005_100100', call='C')  # the provider corrects A in place (ball -> called strike)
    live_now = ask(run, after=6.0, body=fixed)  # wall 1018
    assert shown(live_now) == ['FF'] and live_now['display_state']['correction']['kind'] == 'replaced'
    # 20 s behind: A reached that screen at 1015, the correction came at 1018 -> they saw SL, and are told
    near = ask(run, 20, after=2.0)  # wall 1020
    assert shown(near) == ['FF'] and near['display_state']['correction']['previous']['pitch_type'] == 'SL'
    # 30 s behind: A reaches that screen at 1025, already corrected. SL was never on it: no false "it changed"
    far = ask(run, 30, after=6.0)  # wall 1026
    assert shown(far) == ['FF'] and far['display_state']['correction'] is None and far['display_state']['state'] == 'recommended'
    assert far['pitches'][-1]['receipt'] == {'status': 'received', 'reason': None, 'corrected': True}


def test_a_delayed_viewer_is_not_told_the_data_is_late_because_of_their_own_delay(tmp_path, monkeypatch):
    run = Run(tmp_path, first=snap(900.0, '20261005_100000'))
    ask(run, after=12.0)
    live_now = ask(run, after=12.0, body=snap(950.0, '20261005_100100', second=True))  # proven 62 s late (threshold 45)
    assert live_now['display_state']['state'] == 'feed_delayed' and '실제 경기보다' in live_now['display_state']['message']
    behind = ask(run, 30)['display_state']  # 62 s is later than a 30 s broadcast too: it is late for them, in their words
    assert behind['state'] == 'feed_delayed' and behind['message'] == '경기 자료가 62초 넘게 늦게 도착했어요. 추천이 내 중계보다 늦을 수 있어요.'
    far = ask(run, 70)['display_state']  # a 70 s broadcast is still behind this data: nothing is late for them
    assert (far['state'], far['feed']['delayed']) == ('recommended', False)


# ---------- record at receipt ----------
def only_delayed_viewer(run, until, step=2.0):
    """A single screen on a 30 s delay polling every ``step`` seconds up to wall ``until``; the last answer."""
    view = None
    while run.now[0] < until:
        view = ask(run, 30, after=step, again=False)
        settle(run)
    return view


def test_with_only_a_delayed_viewer_decisions_are_still_recorded_before_the_pitch(run):
    first = ask(run, 30, after=2.0, again=False)  # wall 1002: buffering, but the newest state's decision is queued
    assert first['viewer']['buffering'] is True and 'SL' not in json.dumps(first)
    settle(run)
    only_delayed_viewer(run, 1006.0)
    row, = run.feed.store.served_recommendations(GAME)
    assert row['pitch_key'] == KEY and row['outcome'] == 'first' and row['decided'] <= 1006.0  # long before this viewer's turn
    run.upstream.body = thrown()  # the pitch is thrown at 1015 and fetched by the next due poll
    at_a = only_delayed_viewer(run, 1030.0)
    assert at_a['next']['pitch_key'] == KEY and shown(at_a) == ['SL']
    assert at_a['next']['rec']['ledger']['decided'] == row['decided'] and at_a['next']['rec']['ledger']['outcome'] == 'duplicate'
    at_b = only_delayed_viewer(run, 1048.0)
    assert rule(at_b) == 'recorded_before_pitch_same_inputs' and tops(past(at_b)) == ['SL'] and at_b['receipts']['paired'] == 1
    body_received = next(e for e in run.feed.store.pitch_events(GAME)['events']
                         if (e['at_bat_number'], e['pitch_number'], e['outcome']) == (5, 2, 'new'))['received']
    assert row['decided'] < body_received  # in the ledger before the body that first holds the pitch arrived
    assert set(run.feed.watching[GAME]) == {30.0}  # no pretended live viewer: polling stayed at the slow pace
    assert len({r['decision_id'] for r in run.feed.store.served_recommendations(GAME)}) == len(run.feed.store.served_recommendations(GAME))


def test_without_record_at_receipt_a_lone_delayed_viewer_records_too_late_to_pair(run, monkeypatch):
    monkeypatch.setitem(live_feed_module.LIVE_CONFIG, 'features', {'viewer_delay': {'record_at_receipt': False}})
    only_delayed_viewer(run, 1006.0)
    assert run.feed.store.served_recommendations(GAME) == []
    run.upstream.body = thrown()
    at_b = only_delayed_viewer(run, 1048.0)
    assert rule(at_b) == 'recorded_after_pitch_appeared' and at_b['receipts']['paired'] == 0


def test_record_at_receipt_never_records_an_unconfirmed_input_and_settles_each_version_once(run):
    only_delayed_viewer(run, 1006.0)
    assert len(run.feed.store.served_recommendations(GAME)) == 1 and len(run.policy.calls) == 1
    only_delayed_viewer(run, 1010.0)
    assert len(run.policy.calls) == 1  # the settled version is not looked at again
    run.upstream.body = snap(1010.0, '20261005_095900', second=True)  # a newer event under an OLDER provider stamp
    only_delayed_viewer(run, 1024.0)
    assert len(run.feed.store.served_recommendations(GAME)) == 1 and len(run.policy.calls) == 1  # gated before the policy
    run.upstream.body = OSError('down')  # a failed poll: nothing new is recorded either
    only_delayed_viewer(run, 1040.0)
    assert len(run.feed.store.served_recommendations(GAME)) == 1


def test_a_failure_while_recording_at_receipt_never_fails_the_viewers_request(run, monkeypatch):
    def broken(game_pk):
        raise RuntimeError('boom')
    monkeypatch.setattr(run.feed, 'newest', broken)
    assert ask(run, 30, after=30.0)['next']['pitch_key'] == KEY


# ---------- tap sync: the pure function ----------
ARRIVALS = [900.0, 921.0, 938.0, 962.0, 980.0]  # irregular gaps: 21, 17, 24, 18


def tap(taps, **options):
    options = {'newest_event': 2000.0, 'now': 1100.0, 'current': 30.0} | options
    return tap_sync(ARRIVALS, taps, **options)


def test_two_taps_on_different_pitches_agree_on_one_delay():
    found = tap([962.0+33.0, 980.0+33.4])
    assert found == {'status': 'ok', 'hint': None, 'delay_s': 33.2}
    assert tap([980.0+33.4, 962.0+33.0]) == found  # order does not matter
    assert tap([938.0+7.2, 962.0+7.0, 980.0+6.8])['delay_s'] == pytest.approx(7.0, abs=0.2)  # three taps, a short delay


def test_one_tap_is_ambiguous_unless_only_one_pitch_is_in_range_or_the_device_was_synced():
    assert tap([980.0+33.0]) == {'status': 'unclear', 'hint': 'tap_again', 'delay_s': None}  # 33, 51, 75, 92 or 113 s
    assert tap([980.0+33.0], sure=True) == {'status': 'ok', 'hint': None, 'delay_s': 33.0}  # fine adjustment near 30
    assert tap([980.0+52.0], sure=True)['status'] == 'unclear'  # 52 and 34 are both more than 8 s from... 34 is not
    assert tap([980.0+52.0], sure=True, current=50.0) == {'status': 'ok', 'hint': None, 'delay_s': 52.0}
    assert tap_sync([980.0], [980.0+41.0], newest_event=2000.0, now=1100.0, current=30.0) == {
        'status': 'ok', 'hint': None, 'delay_s': 41.0}  # the first pitch after a break: nothing else it could be


def test_a_tap_before_the_feed_has_the_pitch_waits_instead_of_guessing():
    # true delay 5 s: the pitch arrived at 1000, the tap is at 1005, the feed (18 s behind) only knows up to 980
    early = dict(newest_event=985.0, now=1006.0)
    assert tap([1005.0], **early) == {'status': 'wait', 'hint': 'tap_again', 'delay_s': None}
    assert tap([980.0+5.2, 1005.0], **early) == {'status': 'wait', 'hint': None, 'delay_s': None}  # not 25 s by mistake
    assert tap([1005.0], sure=True, current=5.0, **early)['status'] == 'wait'  # nothing near 5 s yet
    known = dict(newest_event=1006.0, now=1024.0)  # the feed now holds a later event, so every earlier pitch is known
    assert tap_sync([*ARRIVALS, 1000.0], [980.0+5.2, 1005.0], current=30.0, **known) == {'status': 'ok', 'hint': None, 'delay_s': 5.1}
    # a quiet feed (inning break): after the provisional settle time the answer no longer waits
    assert tap([1005.0], newest_event=985.0, now=1005.0+TAP['settle_s'])['status'] == 'unclear'


def test_taps_that_contradict_each_other_start_over_and_evenly_spaced_pitches_ask_for_one_more():
    assert tap([962.0+33.0, 980.0+41.0]) == {'status': 'unclear', 'hint': 'reset', 'delay_s': None}
    # 33 s fits both taps within 0.1 s, 75 s only within 0.9 s: the clearly tighter one is the answer
    assert tap([962.0+33.0, 980.0+33.1])['delay_s'] == pytest.approx(33.05, abs=0.06)
    even = [900.0, 920.0, 940.0, 960.0]
    two = tap_sync(even, [940.0+30.0, 960.0+30.0], newest_event=2000.0, now=1100.0, current=30.0)
    assert two == {'status': 'unclear', 'hint': 'tap_again', 'delay_s': None}  # 30, 50 and 70 s all fit
    assert tap_sync(even, [940.0+30.0, 960.0+30.0], newest_event=2000.0, now=1100.0, current=30.0, sure=True)['delay_s'] == 30.0
    four = tap_sync([880.0, *even], [900.0+30.0, 920.0+30.0, 940.0+30.0, 960.0+30.0], newest_event=2000.0, now=1100.0, current=30.0)
    assert four['status'] == 'unclear' and four['hint'] == 'reset'  # four taps did not settle it: start over


def test_no_pitch_in_range_a_double_press_and_the_limits():
    assert tap([]) == {'status': 'unclear', 'hint': 'no_pitch', 'delay_s': None}
    assert tap_sync([], [1000.0], newest_event=2000.0, now=1100.0, current=30.0)['hint'] == 'no_pitch'
    assert tap([980.0+130.0], now=1200.0)['hint'] == 'no_pitch'  # an inning break: the last pitch is beyond the buffer
    assert tap([962.0+33.0, 980.0+33.0, 980.0+33.9]) == tap([962.0+33.0, 980.0+33.9])  # a double press is one tap
    assert tap_sync([980.0], [979.5], newest_event=2000.0, now=1100.0, current=0.0)['delay_s'] == 0.0  # never below zero
    assert tap_sync([980.0], [980.0+119.0], newest_event=2000.0, now=1100.0, current=0.0, max_delay_s=100.0)['hint'] == 'no_pitch'
    assert tap([700.0, 980.0+33.0])['hint'] == 'tap_again'  # an old tap with no pitch left in range is dropped, not a reset


def test_pitch_arrivals_follow_a_replay_that_moved_the_recording_onto_todays_clock():
    feed = {'metaData': {'timeStamp': '20261009_000030'}, 'liveData': {'plays': {'allPlays': [{'playEvents': [
        {'isPitch': True, 'startTime': '2026-10-09T00:00:01Z', 'endTime': '2026-10-09T00:00:05Z'},
        {'isPitch': False, 'startTime': '2026-10-09T00:00:10Z'},
        {'isPitch': True, 'startTime': '2026-10-09T00:00:20Z', 'endTime': '2026-10-09T00:00:30Z'}]}]}}}
    base = live_feed_module._wall('2026-10-09T00:00:00Z')
    assert pitch_arrivals(feed) == [base+1.5, base+20.5]
    moved = feed | {'metaData': feed['metaData'] | {'eventWall': 5000.0}}  # its newest event (00:00:30) is now at 5000
    assert pitch_arrivals(moved) == [5000.0-30+1.5, 5000.0-30+20.5]
    assert pitch_arrivals({'metaData': {'eventWall': 5000.0}, 'liveData': {'plays': {}}}) == []


# ---------- tap sync: the endpoint ----------
def timed(wall, ts, starts, shift):
    """State A whose open at-bat's pitch (and the four finished at-bats' last pitches) carry start times, moved so
    that the newest lands at ``wall`` on the run's clock."""
    body = snap(wall, ts)
    plays = body['liveData']['plays']['allPlays']
    for play, start in zip(plays, starts):
        for event in play['playEvents']:
            event.pop('startTime', None), event.pop('endTime', None)
        stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start+shift))
        play['playEvents'][-1] |= {'startTime': stamp, 'endTime': stamp}
    body['metaData']['timeStamp'] = time.strftime('%Y%m%d_%H%M%S', time.gmtime(starts[-1]+shift))
    return body


def test_the_tap_endpoint_answers_a_number_and_nothing_about_the_game(run):
    starts = [899.5, 920.5, 937.5, 961.5, 979.5]  # arrivals 900, 921, 938, 962, 980 on the run's clock
    run.upstream.body = timed(979.5, '20261005_100000', starts, shift=86400.0*20700)  # recorded on another day
    ask(run, 30, after=30.0)  # wall 1030: the screen is open, so the feed holds the state (newest event 979.5)
    one = run.client.get(TAP_URL, params={'ago': '17', 'current': 30})  # one tap at 1013, after every known event
    assert one.status_code == 200 and one.headers['cache-control'] == 'no-store'
    assert one.json() == {'status': 'wait', 'hint': 'tap_again', 'delay_s': None, 'message': '좋아요. 다음 공에서 한 번 더 눌러 주세요.',
                          'retry_s': 2.0, 'max_delay_s': 120.0}
    both = run.client.get(TAP_URL, params={'ago': '17,35', 'current': 30}).json()
    assert (both['status'], both['message']) == ('wait', '맞추는 중이에요. 잠시만 기다려 주세요.')
    # a device that was synced before only fine-tunes: the pitch next to its setting answers at once
    assert run.client.get(TAP_URL, params={'ago': '17', 'current': 30, 'sure': 1}).json()['delay_s'] == 33.0
    ask(run, 30, after=30.0)  # wall 1060: 45 s after the newest tap, every pitch before it has reached the feed
    answer = run.client.get(TAP_URL, params={'ago': f'{1060-995.0},{1060-1013.2}', 'current': 30})  # taps at 995 and 1013.2
    assert answer.json() == {'status': 'ok', 'hint': None, 'delay_s': 33.1, 'message': None, 'retry_s': None, 'max_delay_s': 120.0}
    assert set(answer.json()) == {'status', 'hint', 'delay_s', 'message', 'retry_s', 'max_delay_s'}
    assert set(run.feed.watching[GAME]) == {30.0}  # the tap is not a viewer and polled nothing
    for bad in ('', 'x', '-1', '1,nan', '700', ','.join(['1']*9)):
        assert run.client.get(TAP_URL, params={'ago': bad}).status_code == 400, bad
    assert run.client.get(f'/api/service/games/{GAME+1}/tap', params={'ago': '1'}).json()['hint'] == 'no_pitch'  # unknown game


def test_the_legacy_route_is_unchanged_and_serves_the_same_snapshot_as_the_fan_route(run):
    ask(run, after=12.0)
    ask(run, after=12.0, body=thrown())
    run.now[0] += 12.0
    fan, old = ask(run, 30), ask(run, 30, path=LIVE)
    assert old['delay_s'] == 30.0 and fan['display_state']['pitch_key'] == old['display_state']['pitch_key'] == KEY
    assert shown(fan) == [c['pitch_type'] for c in old['recommendation']['recommendation']['candidates']] == ['SL']
    assert fan['current']['situation']['balls'] == old['situation']['balls'] and len(old['pitch_sequence']) == 1
