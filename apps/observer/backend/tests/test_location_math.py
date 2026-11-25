"""Scientific adapter invariants without loading models or private data."""
import numpy as np
import pytest

from observer_app.recommender import location_weights, supported_actions
from observer_app.domain import actual_zone, spatial_comparison, target_point, ZONES


def test_identical_deliveries_have_full_support_and_normalized_weights():
    weights, support, mass = location_weights(np.zeros((2, 400, 2)), np.array([[0., 0.], [1., 1.]]), .45)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    np.testing.assert_allclose(support, 400)
    assert (mass[:, 0] > mass[:, 1]).all()


def test_location_weights_preserve_local_joint_sample_and_handle_extreme_distances():
    xz = np.array([[[-1., 2.], [1., 4.]]])
    weights, support, _ = location_weights(xz, np.array([[-1., 2.], [1000., 1000.]]), .1)
    assert weights[0, 0, 0] > .999
    assert weights[0, 1, 1] > .999
    np.testing.assert_allclose(support, 1)
    assert np.isfinite(weights).all()


def test_target_coordinate_round_trip_and_missing_actual():
    bounds = {'bottom': 1.5, 'top': 3.5}
    for zone in ZONES:
        point = target_point(zone['id'], bounds)
        assert actual_zone(point['x'], point['z'], bounds) == zone['label']
    assert actual_zone(None, 2, bounds) == '위치 정보 없음'
    assert actual_zone(2, 2, bounds) == '존 바깥'
    comparison = spatial_comparison({'x': None, 'z': None, 'zone_bounds': bounds}, 'low_left')
    assert comparison['target_error_zone_units'] is None
    assert comparison['actual_zone_label'] == '위치 정보 없음'


def test_high_effective_sample_size_cannot_validate_distant_target():
    _, support, mass = location_weights(np.zeros((12, 400, 2)), np.array([[0., 0.], [8., 8.]]), .45)
    np.testing.assert_allclose(support, 400)
    assert supported_actions(support.reshape(4, 3, 2), mass.reshape(4, 3, 2)).tolist() == [True, False]
