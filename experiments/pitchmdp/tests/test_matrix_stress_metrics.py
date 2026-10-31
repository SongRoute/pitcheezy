from pathlib import Path
import sys
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from pitchmdp.matrix_stress_metrics import frozen_predictions, combined_status
from score_ml_benchmark import comparison_family
from run_ml_stress import exposure_report
from test_matrix_stress import fixture


def test_archived_blends_keep_seed_order_without_new_calibration():
    y = np.array([0, 1])
    base = np.full((2, 10), .1)
    members = []
    for j in range(3):
        p = base.copy()
        p[:, j] += .2
        p[:, 9] -= .05
        p /= p.sum(1, keepdims=True)
        members.append({'dev': p, 'dev_raw': p})
    weights = [0., .3, 1.]
    archived = {'selection': {'model_weight': .7}, 'seeds': [
        {'blend_selection': {'model_weight': w}} for w in weights]}
    values = frozen_predictions(members, {'dev_y': y, 'dev': base}, archived)
    np.testing.assert_allclose(values['primary'], .7*np.mean([m['dev'] for m in members], axis=0)+.3*base)
    for j, w in enumerate(weights):
        np.testing.assert_allclose(values['seed_primary'][j], w*members[j]['dev']+(1-w)*base)
    with pytest.raises(ValueError):
        frozen_predictions(members[:2], {'dev_y': y, 'dev': base}, archived)


def test_missing_robustness_never_masks_measured_failure():
    assert combined_status([{'passed': None}, {'passed': False}]) == 'failed'
    assert combined_status([{'passed': None}, {'passed': True}]) == 'unconfirmed'
    assert combined_status([{'passed': True}]) == 'passed'
    with pytest.raises(ValueError):
        combined_status([])


def test_architecture_family_rejects_duplicate_omitted_or_reordered_tests():
    names = ['A1-linear', 'A2-lightgbm', 'A3-lstm', 'A4-gru', 'A5-melville', 'A6-transformer']
    def config(items):
        return {'registration': {'control': 'A0-MLP', 'primary_candidates': items}}
    assert comparison_family(config(names))[1] == names
    for items in (names+names[:1], names[:-1], names[::-1]):
        with pytest.raises(ValueError):
            comparison_family(config(items))


def test_exposure_denominator_excludes_delivery_repetition_and_padding():
    store, _ = fixture()
    rows = np.arange(len(store.frame))
    clean = exposure_report(store, rows, 'clean')
    masked = exposure_report(store, rows, 'H0')
    assert clean['present_occurrences'] == 6*sum(range(6))
    assert clean['changed_occurrences'] == 0
    assert masked['dropped_occurrences'] == clean['present_occurrences']
    assert masked['dropped_fraction'] == 1.
    mild, strong = [exposure_report(store, rows, s) for s in ('mask20', 'mask50')]
    assert 0 <= mild['dropped_fraction'] <= strong['dropped_fraction'] <= 1
