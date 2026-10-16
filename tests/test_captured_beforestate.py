"""Generated transition checks; no captured or corrected game rows read."""
import copy

from pitcheezy.seq_shadow.captured_beforestate import prefix_candidates
from test_captured_bridge import feed


def candidates(body):
    return prefix_candidates(body, game_pk=99, game_date='2026-10-11')


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
    assert found[-1]['source'] == 'immediately_prior_completed_pa_final_then_direct_pitch_counts'
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
    assert candidates(body)[-1]['reason'] == 'intervening_action_or_terminal_pitch'


def test_prefix_carries_only_past_verified_nonterminal_pitch():
    body, _, current = body_with_prior()
    second = copy.deepcopy(current['playEvents'][0])
    second.update(pitchNumber=2, playId='p3')
    second['details']['call']['code'] = 'C'
    second['count'].update(balls=1, strikes=1)
    current['playEvents'].append(second)
    assert [r['before_outs_score_hypothesis'] for r in candidates(body) if r['key'][1] == 2] == [
        (1, 2, 0), (1, 2, 0)]
    third = copy.deepcopy(second)
    third.update(pitchNumber=3, playId='p4')
    third['details'].update(isInPlay=True, call={'code': 'X'})
    fourth = copy.deepcopy(second)
    fourth.update(pitchNumber=4, playId='p5')
    current['playEvents'].extend((third, fourth))
    later = [r for r in candidates(body) if r['key'][1] == 2]
    assert later[2]['before_outs_score_hypothesis'] == (1, 2, 0)
    assert later[3]['before_outs_score_hypothesis'] is None
    assert all(not r['accepted'] for r in later)


def test_duplicate_or_missing_provider_and_conflicting_listed_pa_abstain():
    body, prior, current = body_with_prior()
    current['playEvents'].append(copy.deepcopy(current['playEvents'][0]))
    assert all(r['before_outs_score_hypothesis'] is None for r in candidates(body) if r['key'][1] == 2)
    current['playEvents'].pop()
    del current['playEvents'][0]['playId']
    assert candidates(body)[-1]['reason'] == 'first_pitch_identity_unverified'
    current['playEvents'][0]['playId'] = 'p2'
    conflicting = copy.deepcopy(prior)
    conflicting['result']['homeScore'] = 99
    body['liveData']['plays']['allPlays'].insert(1, conflicting)
    assert candidates(body)[-1]['reason'] == 'conflicting_listed_pa_views'
