"""Synthetic ARM-B type ranking + realized-delivery location proxy; no model or data."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.recommendation_adapter import (ARMB_MODE, LOCATION_NOTE, CandidateAction, OUTCOMES,
    PrePitchEvaluation, armb_recommendation)
from observer_app.recommender import Recommender

BOUNDS = {'bottom': 1.5, 'top': 3.5}


def evaluation():
    # Supported actions: FF at two zones, SL at one; CH has no supported zone.
    actions = (CandidateAction(0, 'FF', 'low_left', {'x': -.55, 'z': 1.83}),
               CandidateAction(1, 'FF', 'high_right', {'x': .55, 'z': 3.17}),
               CandidateAction(2, 'SL', 'low_right', {'x': .55, 'z': 1.83}))
    mass = np.zeros((4, 3, 3))
    mass[1, 2] = [.02, .05, .03]   # at 1-2, FF concentrates high_right
    mass[0, 0] = [.05, .02, .03]   # at 0-0, FF concentrates low_left
    ess = np.full((4, 3, 3), 25.)
    return PrePitchEvaluation('ready', None, 'id', 'sha', 'test-v1', 'defense-we-pa-v1',
        'observer-repertoire-kernel-v1', OUTCOMES, actions, None, ess, mass, None, None, None, None, {})


def test_type_ranking_is_primary_and_location_is_unevaluated_proxy():
    result = armb_recommendation({'FF': .2, 'SL': .5, 'CH': .3, 'CU': 0.}, evaluation(), 1, 2, BOUNDS,
                                 policy_identity='armb-test')
    assert result['mode'] == ARMB_MODE and result['status'] == 'ready'
    assert result['location_basis'] == 'realized_delivery_proxy' and result['location_evaluated'] is False
    assert result['location_note'] == LOCATION_NOTE and '평가되지 않았습니다' in LOCATION_NOTE
    assert result['baseline_value'] is None and result['policy_identity'] == 'armb-test'
    assert [c['pitch_type'] for c in result['candidates']] == ['SL', 'CH', 'FF']  # CU (p=0) excluded
    sl, ch, ff = result['candidates']
    assert list(sl)[:6] == ['rank', 'pitch_type', 'pitch_label', 'zone_id', 'zone_label', 'target']
    assert sl['zone_id'] == 'low_right' and sl['detail']['probability'] == .5
    assert ch['zone_id'] is None and ch['target'] is None and ch['detail']['kernel_mass'] is None
    assert ff['zone_id'] == 'high_right' and ff['detail']['kernel_ess'] == 25.
    assert all(c['location_basis'] == 'realized_delivery_proxy' and c['location_evaluated'] is False
               for c in result['candidates'])
    # The proxy zone follows the count-specific realized distribution.
    assert armb_recommendation({'FF': 1.}, evaluation(), 0, 0, BOUNDS)['candidates'][0]['zone_id'] == 'low_left'
    json.dumps(result, allow_nan=False)


def test_unavailable_location_evaluation_keeps_type_ranking():
    empty = PrePitchEvaluation('unavailable', 'x', 'id', 'sha', 'v', 'v', 'b', OUTCOMES, (),
                               None, None, None, None, None, None, None, {})
    result = armb_recommendation({'FF': .6, 'SL': .4}, empty, 0, 0, BOUNDS)
    assert [c['pitch_type'] for c in result['candidates']] == ['FF', 'SL']
    assert all(c['zone_id'] is None for c in result['candidates'])


@pytest.mark.parametrize('probs', [{}, {'FF': .5}, {'FF': 1.2, 'SL': -.2}, {'FF': float('nan')}])
def test_rejects_invalid_type_distribution(probs):
    with pytest.raises(ValueError, match='distribution'):
        armb_recommendation(probs, evaluation(), 0, 0, BOUNDS)


def test_recommender_dispatches_to_armb_only_when_type_policy_bound():
    def policy(inputs):
        assert 'actual' not in inputs.request  # pre-pitch allowlist only
        return {'SL': .7, 'FF': .3}
    policy.identity = 'armb-stub'
    recommender = object.__new__(Recommender)
    recommender.type_policy, recommender.identity = policy, 'observer-id'
    recommender.evaluate_pre_pitch = lambda pitch, pa: evaluation()
    pitch = {'request': {'balls': 1, 'strikes': 2}, 'actual': {'pitch_type': 'FF'}}
    pa = {'zone_bounds': BOUNDS, 'repertoire_counts': {'FF': 100, 'SL': 80}}
    result = recommender.recommend(pitch, pa)
    assert result['mode'] == ARMB_MODE and result['policy_identity'] == 'armb-stub'
    assert [(c['pitch_type'], c['zone_id']) for c in result['candidates']] == [('SL', 'low_right'), ('FF', 'high_right')]
    assert result['id'] == recommender.recommend(pitch | {'actual': {'pitch_type': 'SL'}}, pa)['id']
