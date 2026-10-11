"""A2 reference card and B2 "던지기 전 예상" on the fan route (``service_extras``): attached only in the listed states,
absent with the switch off, computed from the snapshot the viewer is served, and never an error.
Synthetic feeds, a fake policy and stand-in modules; one test runs the real ``profile_reference`` on a synthetic
committed profile."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import profile_reference as PR, service_extras
from test_display_state import GAME, Run, snap
from test_rec_pairing import KEY, thrown
from test_viewer_delay import ask

CARD = {'kind': 'profile_reference', 'title': PR.TITLE, 'chip': PR.CHIP, 'items': [{'pitch_type': 'FF', 'pitch_label': '포심', 'share_pct': 60}]}
EXPECT = {'kind': 'step5_expectation', 'text': '던지기 전 예상: 헛스윙 12% · 땅볼 9%', 'note': '2024 모델 · 참고', 'recorded': 'live_before_pitch'}
# operator facts the modules also return: never on the fan payload (``service_extras.*_PUBLIC``)
PRIVATE_CARD = {'provenance': {'profile_sha256_prefix': 'abcdef012345', 'thresholds': {'min_total': 50}}}
PRIVATE_EXPECT = {'provenance': {'attempt_key': 'k', 'model': {'parent': 'pppp'}}, 'omitted': [{'leaf': 'ground_ball_in_play'}]}


class Reference:
    def __init__(self):
        self.asked, self.fail = [], False

    def card(self, pitcher_id, batter_side, balls, strikes, official_date, *, pitcher_hand=None):
        self.asked.append((pitcher_id, batter_side, balls, strikes, official_date, pitcher_hand))
        if self.fail:
            raise RuntimeError('boom')
        return CARD | PRIVATE_CARD | {'note': f'{balls}-{strikes}'}


class Shadow:
    def __init__(self):
        self.asked, self.answer = [], EXPECT | PRIVATE_EXPECT

    def for_pitch(self, pitch, *, pitch_first_received=None):
        self.asked.append((pitch['key'], pitch['actual']['event'], pitch_first_received))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Both switches on, with stand-in modules behind them."""
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_REFERENCE', '1')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_SOURCES', '/nowhere/profile')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_STEP5_EXPECTATION', '1')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_STEP5_LEDGER', '/nowhere/ledger.sqlite3')
    run = Run(tmp_path)
    extras = run.app.state.service_extras
    run.reference, run.shadow = Reference(), Shadow()
    extras.reference, extras.shadow = (lambda block: run.reference), (lambda block: run.shadow)
    return run


def struck_out(wall=1030.0, ts='20261005_100200'):
    """The open at-bat ended on its second pitch with a strikeout; the next batter is due."""
    body = snap(wall, ts, second=True)
    play = body['liveData']['plays']['currentPlay']
    play['about']['isComplete'] = True
    play['result'] = {'type': 'atBat', 'event': 'Strikeout', 'eventType': 'strikeout', 'description': 'struck out'}
    play['playEvents'][1]['details'] = {'call': {'code': 'S'}, 'description': 'Swinging Strike', 'type': {'code': 'SL'}}
    play['playEvents'][1]['count'] = {'balls': 1, 'strikes': 1, 'outs': 1}
    play['count'] = {'balls': 1, 'strikes': 1, 'outs': 1}
    body['liveData']['linescore']['outs'] = 1
    return body


# ---------- A2: reference card ----------
@pytest.mark.parametrize('status, reason, attached', [
    ('UNSUPPORTED_UNKNOWN_PITCHER', 'unknown_pitcher', True), ('UNSUPPORTED_PITCHER_HAND', 'unknown_pitcher', True),
    ('UNSUPPORTED_EMPTY_SUPPORT', 'no_supported_pitch_types', True),
    ('UNSUPPORTED_MID_PA', 'at_bat_rejected', False), ('SOMETHING_ELSE', 'policy_refused', False)])
def test_the_card_is_attached_only_for_a_pitcher_level_unsupported_state(run, status, reason, attached):
    run.policy.status = status
    view = ask(run, after=12.0)
    state = view['display_state']
    assert (state['state'], state['reason'], state['label'], state['recommendation_shown']) == ('unsupported', reason, '추천 없음', False)
    assert view['next']['rec']['candidates'] == [] and 'reference' not in state  # the honest state is untouched
    assert ('reference' in view['next']) is attached
    if attached:
        current = view['current']
        assert view['next']['reference'] == CARD | {'note': '1-0'}  # no provenance, no thresholds, no hash
        assert 'provenance' not in view['next']['reference'] and 'abcdef' not in json.dumps(view) and 'min_total' not in json.dumps(view)
        assert run.reference.asked[-1] == (current['pitcher']['id'], current['batter']['side'], 1, 0, '2026-10-05', current['pitcher']['hand'])
    else:
        assert run.reference.asked == []


def test_no_card_while_a_recommendation_is_shown_or_no_pitch_is_pending(run):
    view = ask(run, after=12.0)
    assert view['display_state']['state'] == 'recommended' and 'reference' not in view['next']
    run.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    for body in (snap(1030.0, '20261005_100100', half='between'), snap(1045.0, '20261005_100200', half='final')):
        assert ask(run, after=12.0, body=body)['next'] is None
    unsure = ask(Run(run.tmp/'other'), after=12.0)  # another process: same switches, real module, unusable source
    assert 'reference' not in unsure['next']
    assert run.reference.asked == []


def test_with_the_switch_off_the_field_is_absent_and_nothing_is_asked(run, monkeypatch):
    run.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    on = ask(run, after=12.0)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_REFERENCE', '0')
    asked = len(run.reference.asked)
    off = ask(run)
    assert 'reference' in on['next'] and 'reference' not in off['next'] and len(run.reference.asked) == asked
    on['next'].pop('reference')
    strip = lambda view: {k: v for k, v in view.items() if k not in ('features', 'feed')}
    assert strip(off) == strip(on)  # nothing else differs
    assert off['features']['profile_reference'] == {'enabled': False}


def test_a_module_failure_or_no_card_leaves_the_view_as_it_is(run):
    run.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    run.reference.fail = True
    view = ask(run, after=12.0)
    assert view['display_state']['reason'] == 'unknown_pitcher' and 'reference' not in view['next']
    run.reference.fail, run.reference.card = False, lambda *a, **k: None
    assert 'reference' not in ask(run)['next']


def test_a_delayed_viewer_gets_the_card_of_the_count_they_are_shown(run, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', '1')
    run.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    ask(run, after=12.0)
    now = ask(run, after=12.0, body=snap(1015.0, '20261005_100100', second=True))  # wall 1024: count 2-0, pitch 5:3 pending
    late = ask(run, 30, after=12.0)  # 30 s behind: still 1-0, pitch 5:2 pending
    assert now['next']['reference']['note'] == '2-0' and now['next']['pitch_key'] == f'{GAME}:5:3'
    assert late['next']['reference']['note'] == '1-0' and late['next']['pitch_key'] == KEY


@pytest.mark.skipif(not all(Path(PR.SOURCE[k]).is_file() for k in PR.SOURCE['pins']), reason='pinned Step 5 source is not on this machine')
def test_the_real_module_serves_a_card_from_a_committed_strictly_prior_profile(tmp_path, monkeypatch):
    from test_profile_reference import cell, write_profile
    first = Run(tmp_path/'probe')
    current = ask(first, after=12.0)['current']
    pitcher, hand, side = current['pitcher']['id'], current['pitcher']['hand'], current['batter']['side']
    profile = write_profile(tmp_path/'profile', pitchers={pitcher: {'FF': 600, 'SL': 300}},
                            stats=[cell('FF', 30, pitcher=pitcher, hand=hand, stand=side, balls=1, strikes=0),
                                   cell('SL', 10, pitcher=pitcher, hand=hand, stand=side, balls=1, strikes=0)])
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_REFERENCE', '1')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_SOURCES', str(profile))
    run = Run(tmp_path/'run')
    run.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    view = ask(run, after=12.0)
    card = view['next']['reference']
    assert (card['chip'], card['title'], card['as_of'], card['sample']['pitches']) == (PR.CHIP, PR.TITLE, '2026-09-27', 40)
    assert [(i['pitch_type'], i['share_pct']) for i in card['items']] == [('FF', 75), ('SL', 25)]
    assert '추천이나 다음 공 예측이 아닙니다' in card['note'] and view['display_state']['label'] == '추천 없음'
    health = run.client.get('/api/health').json()['service_extras']
    assert health['profile_reference']['enabled'] is True and health['profile_reference']['module']['outcomes'] == {'card': 2}
    assert str(profile) not in json.dumps(view)  # no path on the fan payload


# ---------- B2: 던지기 전 예상 ----------
def marked(view):
    return {p['key'] for p in view['pitches'] if 'expectation' in p}


def key_moments(view):
    """Pitches that ended a plate appearance in a key moment (the fixture's finished at-bats hold some already)."""
    from observer_app.shadow_expectation import is_key_moment
    return {p['key'] for p in view['pitches'] if is_key_moment(p['actual']['event'])}


def test_only_the_last_pitch_of_a_key_moment_plate_appearance_gets_the_expectation(run):
    before = ask(run, after=12.0)
    earlier = key_moments(before)
    assert marked(before) == earlier and KEY not in earlier and 0 < len(earlier) < len(before['pitches'])
    assert 'expectation' not in json.dumps(before['next'])
    view = ask(run, after=12.0, body=struck_out())
    assert marked(view) == earlier | {KEY} == key_moments(view)  # the strikeout pitch joins; no other pitch does
    pitch = next(p for p in view['pitches'] if p['key'] == KEY)
    assert pitch['actual']['event'] == 'strikeout' and pitch['expectation'] == EXPECT  # no provenance, no omitted leaves
    assert 'attempt_key' not in json.dumps(view) and 'omitted' not in json.dumps(view)
    received = next(e for e in run.feed.store.pitch_events(GAME)['events']
                    if (e['at_bat_number'], e['pitch_number'], e['outcome']) == (5, 2, 'new'))['received']
    assert [a for a in run.shadow.asked if a[0] == KEY] == [(KEY, 'strikeout', received)]  # with its first receipt, once
    assert len(run.shadow.asked) == len(earlier)+1  # every pitch is asked once and then kept
    assert 'expectation' not in json.dumps(view['next']) and 'expectation' not in json.dumps(view['current'])


def test_a_plate_appearance_still_in_progress_gets_nothing(run):
    earlier = marked(ask(run, after=12.0))
    view = ask(run, after=12.0, body=thrown())  # 5:2 thrown, the at-bat goes on
    assert marked(view) == earlier and not any(a[0].startswith(f'{GAME}:5:') for a in run.shadow.asked)


def test_switch_off_abstain_and_failure_leave_the_pitch_without_the_field(run, monkeypatch):
    run.shadow.answer = None  # the module abstains (unsupported cell, replayed ledger not allowed, no forecast...)
    ask(run, after=12.0)
    assert marked(ask(run, after=12.0, body=struck_out())) == set()
    run.shadow.answer = RuntimeError('boom')
    run.app.state.service_extras._missed.clear()
    view = ask(run)
    assert view['display_state']['state'] in ('recommended', 'waiting') and marked(view) == set()
    run.shadow.answer = EXPECT | PRIVATE_EXPECT
    run.app.state.service_extras._missed.clear()
    on = ask(run)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_STEP5_EXPECTATION', '0')
    asked = len(run.shadow.asked)
    off = ask(run)
    assert marked(on) == key_moments(on) and KEY in marked(on) and marked(off) == set() and len(run.shadow.asked) == asked
    for pitch in on['pitches']:
        pitch.pop('expectation', None)
    assert off['pitches'] == on['pitches']


def test_an_abstention_is_asked_again_only_after_a_while(run):
    run.shadow.answer = None
    ask(run, after=12.0)
    asked = len(run.shadow.asked)
    ask(run), ask(run)
    assert len(run.shadow.asked) == asked  # not on every poll
    run.app.state.service_extras._missed.clear()  # (the retry time has passed)
    run.shadow.answer = EXPECT | PRIVATE_EXPECT
    assert marked(ask(run)) == key_moments(ask(run))


def test_an_unconfirmed_key_moment_pitch_gets_nothing(run):
    ask(run, after=12.0)
    body = struck_out()
    body['liveData']['plays']['currentPlay']['playEvents'][1].pop('pitchNumber')  # incomplete identity: never ``received``
    view = ask(run, after=12.0, body=body)
    assert not any(key.startswith(f'{GAME}:5:') for key in marked(view))
    assert not any(a[0].startswith(f'{GAME}:5:') for a in run.shadow.asked)


def test_a_delayed_viewer_gets_no_expectation_for_a_pitch_they_have_not_been_served(run, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', '1')
    ask(run, after=12.0)
    now = ask(run, after=12.0, body=struck_out(wall=1015.0, ts='20261005_100100'))  # wall 1024
    late = ask(run, 30, after=12.0)  # wall 1036: the strikeout is 21 s old
    assert KEY in marked(now)
    assert KEY not in marked(late) and KEY not in [p['key'] for p in late['pitches']]
    assert KEY in marked(ask(run, 30, after=10.0))  # wall 1046: served, so shown


def test_without_a_store_nothing_is_attached(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_STEP5_EXPECTATION', '1')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_STEP5_LEDGER', '/nowhere/ledger.sqlite3')
    run = Run(tmp_path, store=False)
    run.app.state.service_extras.shadow = lambda block: pytest.fail('asked without receipts')
    ask(run, after=12.0)
    assert all('expectation' not in p for p in ask(run, after=12.0, body=struck_out())['pitches'])


def test_the_real_modules_refuse_quietly_when_their_sources_are_unusable(tmp_path, monkeypatch):
    for name, value in [('PROFILE_REFERENCE', '1'), ('PROFILE_SOURCES', str(tmp_path/'missing')), ('STEP5_EXPECTATION', '1'),
                        ('STEP5_LEDGER', str(tmp_path/'missing-{game_pk}.sqlite3'))]:
        monkeypatch.setenv(f'PITCHEEZY_OBSERVER_{name}', value)
    run = Run(tmp_path)
    run.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    assert 'reference' not in ask(run, after=12.0)['next']
    view = ask(run, after=12.0, body=struck_out())
    assert all('expectation' not in p for p in view['pitches']) and not list(tmp_path.glob('missing*'))  # nothing created
    health = run.client.get('/api/health').json()['service_extras']
    assert health['step5_expectation'] | {'module': None} == {'enabled': True, 'ledger_configured': True, 'allow_replayed': False, 'module': None}
    assert service_extras.PITCHER_REASONS == ('unknown_pitcher', 'no_supported_pitch_types')
