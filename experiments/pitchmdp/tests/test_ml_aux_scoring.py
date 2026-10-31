import importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('score_aux', Path(__file__).parents[1] / 'scripts/score_ml_aux_data.py')
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def reference_fixture():
    baseline = {'dev_keys': np.array([[1, 1, 1], [2, 1, 1]]), 'dev_y': np.array([0, 1]),
                'dev_game_pk': np.array([1, 2]), 'dev_pitcher': np.array([7, 8])}
    predictions = {cell: {kind: np.full((2, 10), .1) for kind in ('primary', 'raw', 'calibrated', 'seed_primary')}
                   for cell in ('D1-25', 'D1-100')}
    archived = {name: baseline['dev_' + name].copy() for name in ('keys', 'y', 'game_pk', 'pitcher')}
    archived.update({cell + '_' + kind: p.copy() for cell, values in predictions.items() for kind, p in values.items()})
    return archived, baseline, predictions


def test_reused_reference_requires_exact_metadata_and_blends():
    archived, baseline, predictions = reference_fixture()
    SCORER.verify_archived_reference(archived, baseline, predictions)
    archived['pitcher'][0] = 99
    with pytest.raises(ValueError, match='metadata'):
        SCORER.verify_archived_reference(archived, baseline, predictions)
    archived, baseline, predictions = reference_fixture()
    predictions['D1-25']['primary'][0, 0] += 1e-12
    with pytest.raises(ValueError, match='byte-identical'):
        SCORER.verify_archived_reference(archived, baseline, predictions)


def test_arm_specific_frequency_is_part_of_intervention():
    # Identical neural probabilities must retain different TRAIN-fitted
    # frequency baselines; replacing D1's baseline would alter its result.
    y = np.tile(np.arange(10), 2)
    neural = np.full((20, 10), .1)
    useful = np.full((20, 10), .5 / 9)
    useful[np.arange(20), y] = .5
    neutral = {'blend_y': y, 'dev_y': y, 'blend': neural, 'dev': neural}
    informed = {**neutral, 'blend': useful, 'dev': useful}
    member = {'blend': neural, 'dev': neural, 'dev_raw': neural}
    first, _ = SCORER.summarize_cell([member] * 3, neutral)
    second, _ = SCORER.summarize_cell([member] * 3, informed)
    assert second['primary']['log_loss'] < first['primary']['log_loss']
