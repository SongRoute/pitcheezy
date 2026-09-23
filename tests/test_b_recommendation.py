"""Frozen model boundary and strict within-PA history checks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'apps/observer/backend'), str(ROOT/'scripts')]
from scripts.b_pa_history_eval import add_prior
from observer_app.recommender import Recommender


@pytest.fixture(scope='module')
def model():
    os.environ['PITCHEEZY_OBSERVER_RUNTIME'] = 'research'
    return Recommender()


def example():
    return json.loads((ROOT/'docs/contracts/examples/model-v1.json').read_text())


def test_full_boundary_matches_existing_recommendation_and_fixture(model):
    case = example()
    inputs = case['input']
    current = model.recommend(**inputs)
    result = model.evaluate_pre_pitch(**inputs)
    assert result.recommendation == current
    assert result.model_identity == model.identity
    assert result.model_sha256 == model.model_sha256
    assert result.value_spec_version == 'defense-we-pa-v1'
    assert result.baseline_policy_id == 'observer-repertoire-kernel-v1'
    assert len(result.actions) == len({(a.pitch_type, a.zone_id) for a in result.actions})
    assert result.probabilities.shape == (4, 3, 1, len(result.actions), 10)
    assert result.candidate_values.shape == (4, 3, 1, len(result.actions))
    np.testing.assert_allclose(result.probabilities.sum(-1), 1, atol=1e-12)
    np.testing.assert_allclose(result.baseline_policy.sum(), 1, atol=1e-12)
    assert np.isfinite(result.values).all() and np.isfinite(result.baseline_values).all()
    assert all(float(result.support_ess[:, :, i].min()) >= 20 for i in range(len(result.actions)))
    assert all(float(result.kernel_mass[:, :, i].min()) >= .01 for i in range(len(result.actions)))
    for got, frozen in zip(current['candidates'], case['output']['candidates']):
        assert (got['pitch_type'], got['zone_id']) == (frozen['pitch_type'], frozen['zone_id'])
        assert got['value'] == pytest.approx(frozen['value'], abs=1e-12)
        assert got['delta_pp'] == pytest.approx(frozen['delta_pp'], abs=1e-10)


def test_future_actual_fields_do_not_reach_boundary(model):
    inputs = example()['input']
    changed = deepcopy(inputs)
    changed['pitch']['actual'] = {'pitch_type': 'CH', 'x': 99, 'result': 'future'}
    changed['pitch']['future_pitches'] = [{'result': 'future'}]
    changed['pa']['terminal_state'] = {'result': 'future'}
    changed['pa']['pitches'] = [{'result': 'future'}]
    a, b = model.evaluate_pre_pitch(**inputs), model.evaluate_pre_pitch(**changed)
    assert a.recommendation == b.recommendation
    np.testing.assert_array_equal(a.probabilities, b.probabilities)
    assert a.actions == b.actions


def test_unsupported_has_empty_mapping_and_null_values(model):
    inputs = example()['input']
    inputs['pa']['repertoire_counts'] = {}
    result = model.evaluate_pre_pitch(**inputs)
    assert result.status == 'unavailable' and result.reason
    assert result.actions == () and result.probabilities is None
    assert result.values is None and result.baseline_values is None
    assert result.recommendation['candidates'] == []


def test_previous_family_never_crosses_pa_boundary():
    cfg = json.loads((ROOT/'configs/EXP-B-PAHISTORY-001.json').read_text())
    frame = pd.DataFrame({'game_pk': [2, 1, 1, 2], 'at_bat_number': [1, 1, 2, 1],
                          'pitch_number': [2, 1, 1, 1], 'pitch_type': ['CH', 'FF', 'SL', 'SI']})
    out = add_prior(frame, cfg)
    assert out.previous_family.tolist() == ['NONE', 'NONE', 'NONE', 'fastball']
