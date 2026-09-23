from dataclasses import dataclass

import pytest

from scripts.b_choice_metrics import action_accounting, concentration, stability


@dataclass(frozen=True)
class Action:
    pitch_type: str
    zone_id: str


def test_actual_type_uses_best_zone_among_all_supported_actions():
    actions = [Action('FF', 'low'), Action('FF', 'high'), Action('CH', 'low'),
               Action('SI', 'low'), Action('SI', 'high')]
    got = action_accounting(actions, [.8, .7, .6, .59, .58], 'SI')
    assert got['actual_best_rank'] == 4
    assert got['actual_best_zone'] == 'low'
    assert got['gap_pp'] == pytest.approx(21)
    assert got['supported_actions'] == 5 and got['supported_types'] == 3


def test_unsupported_actual_type_has_null_rank_and_gap():
    got = action_accounting([Action('FF', 'low')], [.8], 'SI')
    assert got['top_type'] == 'FF'
    assert got['actual_type_supported'] is False
    assert got['actual_best_rank'] is None and got['gap_pp'] is None


def test_rank_tie_is_deterministic_and_zero_gap():
    got = action_accounting([Action('FF', 'low'), Action('SI', 'low')], [.8, .8], 'SI')
    assert got['top_type'] == 'FF' and got['actual_best_rank'] == 2
    assert got['gap_pp'] == 0


def test_concentration_compares_same_rows_and_counts_types_once():
    base = {'pitcher': 1, 'balls': 0, 'strikes': 0, 'game_pk': 2,
            'at_bat_number': 3}
    rows = [base | {'actual_type': 'FF', 'top_type': 'SI'},
            base | {'actual_type': 'FF', 'top_type': 'SI'},
            base | {'actual_type': 'SI', 'top_type': 'SI'},
            base | {'actual_type': 'CH', 'top_type': None}]
    got = concentration(rows)[0]
    assert got['pitches'] == 3
    assert got['actual_share'] == {'FF': 2/3, 'SI': 1/3}
    assert got['top_type_share'] == {'FF': 0, 'SI': 1}
    assert got['total_variation'] == pytest.approx(2/3)


def test_stability_requires_aligned_keys_and_distinguishes_action_type():
    a = {'top_action': {'pitch_type': 'FF', 'zone_id': 'low'}, 'top_type': 'FF',
         'q_by_action': {'FF|low': .8, 'FF|high': .7, 'SI|low': .6},
         'actual_type_supported': True, 'actual_best_rank': 2}
    b = a | {'top_action': {'pitch_type': 'FF', 'zone_id': 'high'}, 'actual_best_rank': 3,
             'q_by_action': {'FF|low': .7, 'FF|high': .8, 'SI|low': .6}}
    got = stability({1: {(2, 3, 1): a}, 2: {(2, 3, 1): b}}, {(2, 3, 1): a})
    assert got['top_action_agreement_rate'] == 0
    assert got['top_type_agreement_rate'] == 1
    assert got['mean_pairwise_top_action_agreement'] == 0
    assert got['mean_top_action_agreement_with_ensemble'] == .5
    assert got['mean_pairwise_action_rank_spearman'] == pytest.approx(.5)
    assert got['per_pitch'][0]['actual_rank_min'] == 2
    assert got['per_pitch'][0]['actual_rank_max'] == 3
    with pytest.raises(ValueError, match='different pitch keys'):
        stability({1: {(2, 3, 1): a}, 2: {(2, 3, 2): b}})
