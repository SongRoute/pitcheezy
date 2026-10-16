"""Generated receipt-local identity transitions, never provider truth."""
import copy

from pitcheezy.seq_shadow.captured_identity_bip import inspect_hypotheses
from pitcheezy.seq_shadow.captured_native import inspect_tokens
from pitcheezy.seq_shadow.captured_native_identity import inspect_receipt_identity
from test_captured_bridge import feed

SHA = 'a' * 64


def identity(body):
    old = inspect_hypotheses(body, game_pk=99, game_date='2026-10-11')
    return inspect_receipt_identity(body, game_pk=99, game_date='2026-10-11', all_pitch_rows=old)


def mixed(action):
    body = feed()
    play = body['liveData']['plays']['currentPlay']
    first = play['playEvents'][0]
    first.update(index=0)
    first['details']['type'] = {'code': 'FF'}
    second = copy.deepcopy(first)
    second.update(index=2, pitchNumber=2, playId='p2')
    second['count'] = {'balls': 2, 'strikes': 0, 'outs': 0}
    action = dict(action, index=1, isPitch=False, count={'balls': 1, 'strikes': 0, 'outs': 0})
    play['playEvents'] = [first, action, second]
    play['count'] = {'balls': 2, 'strikes': 0, 'outs': 0}
    return body


def test_pitch_only_legacy_hypothesis_remains_source_native():
    body = feed()
    body['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'FF'}
    row = inspect_tokens(body, game_pk=99, game_date='2026-10-11', receipt_seq=1, source_sha256=SHA)[0]
    assert row['status'] == 'native_candidate'
    assert row['identity_provenance']['rule'] == 'pitch_only_play_matchup'


def test_structured_pitching_change_splits_same_receipt_pitch_identity():
    body = mixed({'isSubstitution': True, 'details': {'eventType': 'pitching_substitution'},
                  'player': {'id': 11}, 'replacedPlayer': {'id': 10}})
    body['liveData']['plays']['currentPlay']['matchup']['pitcher']['id'] = 11
    mapping = identity(body)
    assert mapping[(99, 1, 1)]['identity_hypothesis']['pitcher'] == 10
    assert mapping[(99, 1, 2)]['identity_hypothesis']['pitcher'] == 11
    assert mapping[(99, 1, 1)]['identity_provenance']['later_action_chain'] == [
        {'index': 1, 'assumption': 'structured_pitching_substitution'}]
    assert mapping[(99, 1, 1)]['identity_hypothesis']['throws'] is None
    rows = inspect_tokens(body, game_pk=99, game_date='2026-10-11', receipt_seq=1, source_sha256=SHA)
    assert [row['native_token']['pitcher'] for row in rows] == [10, 11]


def test_explicit_benign_event_is_assumption_unknown_action_stops_earlier_segment():
    body = mixed({'isSubstitution': False, 'details': {'eventType': 'mound_visit'}})
    mapping = identity(body)
    assert set(mapping) == {(99, 1, 1), (99, 1, 2)}
    assert mapping[(99, 1, 1)]['identity_provenance']['later_action_chain'][0]['assumption'] == \
        'provider_flag_benign_mound_visit_assumption'
    body['liveData']['plays']['currentPlay']['playEvents'][1]['details']['eventType'] = 'unknown_action'
    assert set(identity(body)) == {(99, 1, 2)}
    body['liveData']['plays']['currentPlay']['playEvents'][1].pop('isSubstitution')
    assert set(identity(body)) == {(99, 1, 2)}


def test_pinch_hitter_stays_unknown_without_active_batter_slot_proof():
    action = {'isSubstitution': True, 'details': {'eventType': 'offensive_substitution'},
              'player': {'id': 21}, 'replacedPlayer': {'id': 20},
              'position': {'abbreviation': 'PH'}, 'battingOrder': '300'}
    body = mixed(action)
    body['liveData']['plays']['currentPlay']['matchup']['batter']['id'] = 21
    assert (99, 1, 1) not in identity(body)
    assert identity(body)[(99, 1, 2)]['identity_hypothesis']['batter'] == 21
    for field, value in [('position', {'abbreviation': 'PR'}), ('battingOrder', None),
                         ('player', {'id': 22})]:
        altered = copy.deepcopy(body)
        altered['liveData']['plays']['currentPlay']['playEvents'][1][field] = value
        assert set(identity(altered)) == {(99, 1, 2)}


def test_conflicting_views_or_event_order_or_defensive_change_do_not_guess():
    body = mixed({'isSubstitution': True, 'details': {'eventType': 'defensive_substitution'},
                  'player': {'id': 11}, 'replacedPlayer': {'id': 10}})
    assert set(identity(body)) == {(99, 1, 2)}
    out_of_order = copy.deepcopy(body)
    out_of_order['liveData']['plays']['currentPlay']['playEvents'][1]['index'] = 0
    assert identity(out_of_order) == {}
    conflicting = copy.deepcopy(body)
    conflicting['liveData']['plays']['allPlays'] = [copy.deepcopy(body['liveData']['plays']['currentPlay'])]
    conflicting['liveData']['plays']['allPlays'][0]['matchup']['pitcher']['id'] = 99
    assert identity(conflicting) == {}
    reused_provider = copy.deepcopy(body)
    reused_provider['liveData']['plays']['currentPlay']['playEvents'][2]['playId'] = 'p1'
    assert identity(reused_provider) == {}


def test_malformed_action_and_broken_pitcher_chain_keep_earlier_pitch_unknown():
    body = mixed({'isSubstitution': False, 'details': {'eventType': ['mound_visit']}})
    assert set(identity(body)) == {(99, 1, 2)}
    body['liveData']['plays']['currentPlay']['playEvents'][1] = {
        'index': 1, 'isPitch': False, 'isSubstitution': True,
        'details': {'eventType': 'pitching_substitution'},
        'player': {'id': 99}, 'replacedPlayer': {'id': 10},
    }
    body['liveData']['plays']['currentPlay']['matchup']['pitcher']['id'] = 11
    assert set(identity(body)) == {(99, 1, 2)}
