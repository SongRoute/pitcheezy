"""Generated transition checks; no captured or corrected game rows read."""
import copy

from pitcheezy.seq_shadow.captured_beforestate import first_pitch_candidates
from test_captured_bridge import feed


def candidates(body):
    return first_pitch_candidates(body, game_pk=99, game_date='2026-10-11')


def body_with_prior():
    body = feed()
    prior = copy.deepcopy(body['liveData']['plays']['currentPlay'])
    prior['about']['isComplete'] = True
    prior['count']['outs'] = 1
    prior['result']['homeScore'] = 2
    current = copy.deepcopy(prior)
    current['about'].update(atBatIndex=1, isComplete=False)
    current['playEvents'][0].update(playId='p2')
    current['playEvents'][0]['count']['outs'] = 1
    body['liveData']['plays']['allPlays'] = [prior, current]
    body['liveData']['plays']['currentPlay'] = current
    return body, prior, current


def test_same_half_uses_only_immediate_prior_final_state():
    body, prior, current = body_with_prior()
    current['result']['homeScore'] = 3  # current PA final state must not leak backward.
    found = candidates(body)
    assert found[-1]['key'] == (99, 2, 1)
    assert found[-1]['before_outs_score_hypothesis'] == (1, 2, 0)
    assert found[-1]['source'] == 'immediately_prior_completed_pa_final'
    assert not found[-1]['accepted']
    assert found[0]['before_outs_score_hypothesis'] is None


def test_half_change_resets_outs_and_invalid_transitions_abstain():
    body, prior, current = body_with_prior()
    prior['count']['outs'] = 3
    current['about']['isTopInning'] = False
    assert candidates(body)[-1]['before_outs_score_hypothesis'] == (0, 2, 0)
    current['about']['inning'] = 2
    assert candidates(body)[-1]['reason'] == 'half_transition_unverified'
    current['about']['inning'] = 1
    prior['count']['outs'] = 2
    assert candidates(body)[-1]['reason'] == 'half_transition_unverified'


def test_incomplete_prior_or_intervening_action_refuses_candidate():
    body, prior, current = body_with_prior()
    prior['about']['isComplete'] = False
    assert candidates(body)[-1]['reason'] == 'prior_pa_completion_unverified'
    prior['about']['isComplete'] = True
    current['playEvents'].insert(0, {'isPitch': False, 'count': {'balls': 0, 'strikes': 0}})
    assert candidates(body)[-1]['reason'] == 'intervening_action_unverified'
