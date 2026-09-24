import numpy as np
import pandas as pd
from pitchmdp.matrix_group_metrics import bootstrap_difference, masks, guardrails


def test_bootstrap_preserves_pitch_weighting_and_constant_upper():
    delta = np.column_stack([np.arange(10) / 100, np.ones(10) * -.01])
    r = bootstrap_difference(delta, np.array([1] + [2] * 9), draws=200, upper_alpha=.005)
    np.testing.assert_allclose(r['delta'], [.045, -.01], rtol=0, atol=1e-15)
    assert abs(r['simultaneous_upper'][1] + .01) < 1e-12


def test_missing_panel_role_and_zero_volume_cannot_pass_robustness():
    n = 600
    metadata = pd.DataFrame({'game_role': ['starter'] * n, 'throwing_hand': ['R'] * n,
        'train_volume': ['high'] * n, 'two_strikes': [True] * n, 'runners_on': [False] * n})
    p = np.full((n, 10), .1)
    result = guardrails(np.zeros(n, dtype=int), p, p, np.repeat(np.arange(30), 20), metadata)
    assert result['status'] == 'unconfirmed'
    assert result['groups']['volume_zero']['passed'] is None
    assert result['family_size'] == 96
    assert masks(metadata)['role_starter'].all()
