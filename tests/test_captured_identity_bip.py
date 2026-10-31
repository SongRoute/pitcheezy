"""Generated-only matchup/BIP hypotheses; no real receipt or I3 read."""
import copy

from pitcheezy.seq_shadow.captured_identity_bip import inspect_hypotheses
from test_captured_bridge import feed


def hypotheses(body):
    return inspect_hypotheses(body, game_pk=99, game_date='2026-10-11')


def bip_body():
    body = feed()
    play = body['liveData']['plays']['currentPlay']
    play['about']['isComplete'] = True
    play['result']['eventType'] = 'single'
    pitch = play['playEvents'][0]
    pitch['details'].update(call={'code': 'X'}, isInPlay=True)
    pitch['hitData'] = {'trajectory': 'ground_ball'}
    return body


def test_pitch_only_pa_matchup_and_final_bip_are_unaccepted_hypotheses():
    row = hypotheses(bip_body())[0]
    assert row['key'] == (99, 1, 1)
    assert row['identity_hypothesis'] == {'pitcher': 10, 'batter': 20, 'stand': 'L', 'throws': 'R'}
    assert row['description_bip_hypothesis'] == 'hit_into_play'
    assert row['events_bip_hypothesis'] == 'single'
    assert row['outcome10_bip_hypothesis'] == 4
    assert row['bb_type_hypothesis'] == 'ground_ball'
    assert row['accepted'] is False
    assert 'paired_pitch_level_identity' in row['proof_missing']


def test_unregistered_bip_result_or_trajectory_remains_unavailable():
    body = bip_body()
    play = body['liveData']['plays']['currentPlay']
    play['result']['eventType'] = 'field_error'
    play['playEvents'][0]['hitData']['trajectory'] = 'GB'
    row = hypotheses(body)[0]
    assert row['events_bip_hypothesis'] == 'field_error'
    assert row['outcome10_bip_hypothesis'] is None
    assert row['bb_type_hypothesis'] is None
    assert row['reasons']['bip'] == 'bip_result_or_trajectory_unavailable'
    play['result']['eventType'] = 'double_play'
    play['playEvents'][0]['hitData']['trajectory'] = 'popup'
    row = hypotheses(body)[0]
    assert row['outcome10_bip_hypothesis'] == 9
    assert row['bb_type_hypothesis'] == 'popup'


def test_substitution_or_provider_conflict_refuses_identity():
    body = bip_body()
    play = body['liveData']['plays']['currentPlay']
    play['playEvents'].insert(0, {'isPitch': False, 'details': {'eventType': 'pitching_substitution'}})
    assert hypotheses(body)[0]['identity_hypothesis'] is None
    play['playEvents'].pop(0)
    duplicate = copy.deepcopy(play['playEvents'][0])
    duplicate['pitchNumber'] = 2
    play['playEvents'].append(duplicate)
    assert all(row['identity_hypothesis'] is None for row in hypotheses(body))


def test_nonfinal_or_incomplete_inplay_pitch_does_not_borrow_result():
    body = bip_body()
    play = body['liveData']['plays']['currentPlay']
    play['about']['isComplete'] = False
    assert hypotheses(body)[0]['outcome10_bip_hypothesis'] is None
    play['about']['isComplete'] = True
    later = copy.deepcopy(play['playEvents'][0])
    later.update(playId='p2', pitchNumber=2)
    later['details'].update(isInPlay=False, call={'code': 'B'})
    play['playEvents'].append(later)
    assert hypotheses(body)[0]['outcome10_bip_hypothesis'] is None


def test_malformed_call_and_trajectory_are_unavailable_not_errors():
    body = bip_body()
    pitch = body['liveData']['plays']['currentPlay']['playEvents'][0]
    pitch['details']['call']['code'] = ['X']
    pitch['hitData']['trajectory'] = {'code': 'ground_ball'}
    row = hypotheses(body)[0]
    assert row['description_bip_hypothesis'] is None
    assert row['events_bip_hypothesis'] is None
    assert row['bb_type_hypothesis'] is None
