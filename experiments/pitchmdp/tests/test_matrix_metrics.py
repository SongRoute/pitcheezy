import numpy as np
import pytest

from pitchmdp.matrix_metrics import (holm_adjust, paired_game_comparison, pitch_losses,
                                    prediction_decision, prediction_metrics, validate_probabilities)


def probabilities(n, probability):
    result = np.full((n, 10), (1-probability)/9)
    result[:, 0] = probability
    return result


def test_proper_scoring_rules_and_unobserved_class():
    y = np.array([0, 1])
    p = np.eye(10)[y]
    result = prediction_metrics(y, p)
    assert result['log_loss'] == 0
    assert result['brier_multiclass'] == 0
    assert result['accuracy'] == 1
    assert result['classes'][2]['f1'] is None
    assert result['classes'][2]['recall'] is None
    assert result['macro_f1_observed_classes'] == 1
    assert result['macro_f1_class_count'] == 2


def test_brier_is_class_sum_and_probability_tolerance_is_absolute():
    y = np.array([0])
    p = np.eye(10)[[1]]
    assert pitch_losses(y, p)[0, 1] == 2
    bad = probabilities(1, .5)
    bad[0, 0] += 2e-6
    with pytest.raises(ValueError, match='mass'):
        validate_probabilities(y, bad)


def test_paired_bootstrap_uses_pitch_weighted_not_game_weighted_mean():
    y = np.zeros(10, dtype=int)
    base = probabilities(10, .5)
    candidate = probabilities(10, .6)
    candidate[0] = probabilities(1, .1)[0]
    games = np.array([1] + [2]*9)
    result = paired_game_comparison(y, candidate, base, games, draws=1000, seed=9)
    losses = pitch_losses(y, candidate) - pitch_losses(y, base)
    assert result['nll']['delta'] == pytest.approx(losses[:, 0].mean())
    assert result['nll']['delta'] != pytest.approx((losses[0, 0] + losses[1:, 0].mean())/2)
    assert result == paired_game_comparison(y, candidate, base, games, draws=1000, seed=9)


def test_null_and_improvement_bootstraps_and_single_game():
    y = np.zeros(12, dtype=int)
    base, better = probabilities(12, .4), probabilities(12, .6)
    games = np.repeat(np.arange(4), 3)
    null = paired_game_comparison(y, base, base, games, draws=200)
    assert null['nll']['p_less'] == 1
    improved = paired_game_comparison(y, better, base, games, draws=200)
    assert improved['nll']['ci95'][1] < 0
    assert improved['nll']['p_less'] == pytest.approx(1/201)
    single = paired_game_comparison(y, better, base, np.ones(12), draws=200)
    assert single['nll']['ci95'] is None


def test_holm_preserves_family_and_order():
    assert holm_adjust([.04, .001, .03]) == pytest.approx([.06, .003, .06])
    assert holm_adjust([.01, None]) == [.02, None]


def test_decision_requires_seed_stability_and_never_claims_policy_effect():
    y = np.zeros(12, dtype=int)
    comparison = paired_game_comparison(y, probabilities(12, .6), probabilities(12, .4),
                                       np.repeat(np.arange(4), 3), draws=200)
    result = prediction_decision(comparison, .01, [-.1, -.1, .01])
    assert result['status'] == 'predictive_improvement'
    assert result['policy_effect'] is None
    unstable = prediction_decision(comparison, .01, [-.1, .1, .1])
    assert unstable['status'] == 'inconclusive'


def test_shape_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        prediction_metrics(np.array([0]), np.array([[1.]]))
    p = probabilities(1, .5)
    p[0, 0] = np.nan
    with pytest.raises(ValueError):
        prediction_metrics(np.array([0]), p)


def test_decisions_reject_incomplete_or_nonfinite_seeds_and_final_needs_four():
    y = np.zeros(12, dtype=int)
    comp = paired_game_comparison(y, probabilities(12, .6), probabilities(12, .4),
                                 np.repeat(np.arange(4), 3), draws=200)
    for deltas in ([-.1, -.1], [-.1, -.1, np.nan]):
        with pytest.raises(ValueError, match='Complete'):
            prediction_decision(comp, .01, deltas)
    assert prediction_decision(comp, .01, [-.1]*3 + [.1]*2, stage='final')['status'] == 'inconclusive'
    assert prediction_decision(comp, .01, [-.1]*4 + [.1], stage='final')['status'] == 'predictive_improvement'
