import numpy as np
import pytest
from pitchmdp.matrix_interaction_metrics import factorial_contrasts, interaction_decision


def test_factorial_sign_and_pitch_weighting_use_losses_not_probability_mixtures():
    # Unequal game sizes: pitch-weighted interaction differs from game mean.
    losses = np.ones((4, 4, 2))
    losses[:, 3, 0] = [1.02, .98, .98, .98]
    result = factorial_contrasts(losses, [1, 2, 2, 2], draws=1000)
    assert result['contrasts']['interaction']['nll']['delta'] == pytest.approx(-.01)
    assert result['contrasts']['data_main']['nll']['delta'] == pytest.approx(-.005)
    assert result['contrasts']['architecture_main']['nll']['delta'] == pytest.approx(-.005)
    assert result['contrasts']['mlp_data']['nll']['delta'] == 0
    assert result['contrasts']['interaction']['brier']['delta'] == 0
    again = factorial_contrasts(losses, [1, 2, 2, 2], draws=1000)
    assert result == again


def test_constant_model_effect_has_zero_architecture_data_interaction():
    losses = np.ones((10, 4, 2))
    losses[:, 1] -= .01
    losses[:, 2] -= .02
    losses[:, 3] -= .03
    result = factorial_contrasts(losses, np.arange(10), draws=100)
    assert result['contrasts']['interaction']['nll']['delta'] == pytest.approx(0, abs=1e-14)
    assert result['contrasts']['data_main']['nll']['delta'] == pytest.approx(-.01)
    assert result['contrasts']['architecture_main']['nll']['delta'] == pytest.approx(-.02)


def test_preregistered_interaction_requires_magnitude_uncertainty_and_seed_direction():
    c = {'delta': -.004, 'ci95': [-.006, -.002], 'p_two_sided_centered': .01}
    assert interaction_decision(c, [-.005, -.004, .001])['status'] == 'interaction_detected'
    assert interaction_decision(c, [-.005, .001, .001])['status'] == 'inconclusive'
    assert interaction_decision({**c, 'delta': -.002}, [-.002]*3)['status'] == 'inconclusive'
    assert interaction_decision({**c, 'ci95': [-.006, .001]}, [-.004]*3)['status'] == 'inconclusive'
