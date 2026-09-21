"""Synthetic checks for disjoint fitting, repair gating inputs and policy evaluation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from pitchmdp.archetypes import STYLE_COLUMNS
from pitchmdp.planner import TERMINALS
from policy_evaluation_models import JulyEvaluator, OutcomeCalibration
from run_service_validation import partition, select_policy_cases, compare_policies, summarize


def sample():
    return pd.DataFrame({'game_date': ['2025-07-02']*120, 'game_pk': [1]*120,
        'balls': [0]*120, 'strikes': [0]*120, 'stand': ['R']*120, 'p_throws': ['R']*120,
        'pitch_type': ['FF']*120, 'outs_when_up': [1]*120, 'bases': [1]*120, 'pitcher': [10]*120,
        STYLE_COLUMNS[0]: [.8]*120, STYLE_COLUMNS[4]: [.15]*120,
        'description': ['ball']*80+['called_strike']*40, 'events': ['']*120})


def test_evaluator_does_not_accept_other_training_periods():
    with pytest.raises(ValueError, match='July'):
        JulyEvaluator().fit(sample().assign(game_date='2025-06-30'))
    with pytest.raises(ValueError, match='July'):
        JulyEvaluator().fit(sample().assign(game_date='2025-08-01'))


def test_empirical_hierarchy_unseen_players_and_context():
    model = JulyEvaluator().fit(sample())
    query = sample().iloc[:2].copy()
    query.loc[query.index[1], ['pitcher', 'bases']] = [999, 7]
    p, support, origin = model.predict_with_support(query)
    np.testing.assert_allclose(p.sum(1), 1)
    assert (p > 0).all() and (p[:, 0] > p[:, 1]).all()
    assert origin[0] == 4 and origin[1] == 1
    query['stand'] = 'L'
    unseen, _, levels = model.predict_with_support(query)
    np.testing.assert_allclose(unseen, np.tile(model.global_p, (2, 1)))
    assert (levels == -1).all()


def test_calibration_preserves_impossible_events_and_checks_date():
    p = np.tile(np.array([.6, .4]+[0.]*8), (120, 1))
    y = np.array([0]*80+[1]*40)
    with pytest.raises(ValueError, match='August'):
        OutcomeCalibration().fit(p, y, ['2025-08-08']*120)
    model = OutcomeCalibration().fit(p, y, ['2025-08-01']*120)
    fixed = model.apply(p)
    np.testing.assert_allclose(fixed.sum(1), 1)
    assert (fixed[:, 2:] == 0).all()
    assert fixed[0, 0] > p[0, 0]


def test_split_rejects_same_game_in_two_periods():
    frame = pd.DataFrame({'game_date': ['2025-07-01', '2025-08-01', '2025-08-08', '2025-08-16'],
                          'game_pk': [1, 2, 3, 4]})
    assert len(partition(frame)) == 4
    frame.loc[1, 'game_pk'] = 1
    with pytest.raises(ValueError, match='overlapping'):
        partition(frame)


def test_evaluator_does_not_reselect_actions_for_selected_policy():
    p = np.zeros((4, 3, 1, 2, 10))
    p[..., 0, 3], p[..., 1, 7] = 1, 1
    judge = p[..., ::-1, :].copy()
    value = {event: float(event == 'out') for event in TERMINALS}
    scores = compare_policies(p, judge, value, value, value, np.array([.5, .5]))
    assert scores['frequency'] == pytest.approx(.5)
    assert scores['we_full'] == pytest.approx(0)
    assert scores['we_one'] == pytest.approx(0)


def test_case_selection_is_first_five_per_game():
    frame = pd.DataFrame({'game_pk': [1]*8, 'game_date': ['2025-09-01']*8,
                          'pitcher': [10]*8, 'at_bat_number': range(1, 9), 'pitch_number': [1]*8,
                          'balls': [0]*8, 'strikes': [0]*8})
    assert select_policy_cases(frame.sample(frac=1, random_state=42)).at_bat_number.tolist() == [1, 2, 3, 4, 5]


def test_primary_intervals_account_for_two_comparisons():
    cases = [{'game_pk': i, 'original': {'we_full': .6, 'we_one': .55, 'frequency': .5, 're_full': .59},
              'selected': {'we_full': .6, 'we_one': .55, 'frequency': .5, 're_full': .59}} for i in range(3)]
    result = summarize(cases)
    assert result['comparisons']['original']['we_full_minus_frequency']['coverage'] == .975
    assert result['comparisons']['original']['we_full_minus_re_full']['coverage'] == .95
