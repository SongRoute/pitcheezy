"""Generated only: guard explanations must match the unchanged identity helper."""
import copy

import pytest

from pitcheezy.seq_shadow.captured_identity_bip import inspect_hypotheses
from pitcheezy.seq_shadow.captured_native_identity_diagnostic import (
    inspect_identity_guards, summarize_guard_census,
)
from test_captured_native_identity import mixed


def guards(body):
    rows = inspect_hypotheses(body, game_pk=99, game_date='2026-10-11')
    return inspect_identity_guards(body, game_pk=99, game_date='2026-10-11',
                                   all_pitch_rows=rows)


def test_reverse_action_guard_and_metadata_shape_without_player_id_export():
    body = mixed({'isSubstitution': True, 'details': {'eventType': 'offensive_substitution'},
                  'player': {'id': 21}, 'replacedPlayer': {'id': 20},
                  'position': {'abbreviation': 'PH'}})
    result = guards(body)
    first, second = result[(99, 1, 1)], result[(99, 1, 2)]
    assert first['guard'] == 'reverse_action_chain_broken'
    assert first['blocking_action']['event_type'] == 'offensive_substitution'
    assert first['blocking_action']['issue'] == 'action_kind_not_reversible'
    assert first['blocking_action']['player_id'] == 'valid'
    assert second['guard'] == 'identity_hypothesis_present'
    assert '21' not in str(first)
    summary = summarize_guard_census([(1, result), (2, result)])
    assert summary['receipt_views'] == 4
    assert summary['distinct_keys'] == 2
    assert summary['event_type'] == {'offensive_substitution': 2}


def test_distinct_first_failing_guards_and_malformed_action():
    body = mixed({'isSubstitution': False, 'details': {'eventType': 'mound_visit'}})
    assert all(row['guard'] == 'identity_hypothesis_present' for row in guards(body).values())
    body['liveData']['plays']['currentPlay']['playEvents'][1]['details']['extra'] = 'x'
    first = guards(body)[(99, 1, 1)]
    assert first['guard'] == 'reverse_action_chain_broken'
    assert first['blocking_action']['issue'] == 'details_extra_keys'
    assert first['blocking_action']['extra_details_keys'] == ('extra',)
    body['liveData']['plays']['currentPlay']['playEvents'][1]['details'].pop('eventType')
    assert guards(body)[(99, 1, 1)]['blocking_action']['event_type'] == 'absent'
    body['liveData']['plays']['currentPlay']['playEvents'][1]['details']['eventType'] = ['bad']
    assert guards(body)[(99, 1, 1)]['blocking_action']['event_type'] == 'non_string'

    malformed = copy.deepcopy(body)
    action = malformed['liveData']['plays']['currentPlay']['playEvents'][1]
    action['details']['eventType'] = 'pitching_substitution'
    action['isSubstitution'] = True
    action['player'] = {'id': True}
    assert guards(malformed)[(99, 1, 1)]['blocking_action']['player_id'] == 'id_malformed'
    action['player'] = {}
    assert guards(malformed)[(99, 1, 1)]['blocking_action']['player_id'] == 'id_absent'

    disordered = copy.deepcopy(body)
    disordered['liveData']['plays']['currentPlay']['playEvents'][1]['index'] = 0
    assert {row['guard'] for row in guards(disordered).values()} == {'mixed_timeline_order_invalid'}
    duplicate = copy.deepcopy(body)
    duplicate['liveData']['plays']['currentPlay']['playEvents'][2]['playId'] = 'p1'
    assert guards(duplicate)[(99, 1, 2)]['guard'] == 'provider_id_missing_or_reused'


def test_game_identity_hard_fail():
    body = mixed({'isSubstitution': False, 'details': {'eventType': 'mound_visit'}})
    rows = inspect_hypotheses(body, game_pk=99, game_date='2026-10-11')
    with pytest.raises(ValueError, match='game/date mismatch'):
        inspect_identity_guards(body, game_pk=100, game_date='2026-10-11',
                                all_pitch_rows=rows)


def test_conflict_anchor_and_pitch_only_baseline_are_separate_failures():
    body = mixed({'isSubstitution': False, 'details': {'eventType': 'mound_visit'}})
    conflict = copy.deepcopy(body)
    listed = copy.deepcopy(conflict['liveData']['plays']['currentPlay'])
    listed['matchup']['pitcher']['id'] = 88
    conflict['liveData']['plays']['allPlays'] = [listed]
    assert {row['guard'] for row in guards(conflict).values()} == {'view_conflict'}

    anchor = copy.deepcopy(body)
    anchor['liveData']['plays']['currentPlay']['matchup']['pitcher']['id'] = True
    assert {row['guard'] for row in guards(anchor).values()} == {'end_matchup_anchor_invalid'}

    pitch_only = copy.deepcopy(body)
    play = pitch_only['liveData']['plays']['currentPlay']
    play['playEvents'].pop(1)
    play['matchup']['batter'].pop('id')
    assert {row['guard'] for row in guards(pitch_only).values()} == {
        'pitch_only_baseline_unavailable'}
