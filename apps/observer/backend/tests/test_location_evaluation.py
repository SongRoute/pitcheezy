import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'research'))
from location_evaluator import LocationEvaluator, location_features, location_cells, paired_games, legal


def test_location_basis_does_not_use_outcome_or_player_id():
    frame = pd.DataFrame({'plate_x': [0., -.8], 'plate_z': [2.5, 1.6], 'events': ['single', 'home_run'], 'batter': [1, 2]})
    expected = location_features(frame)
    assert expected.shape == (2, 18)
    frame['events'], frame['batter'] = 'strikeout', 999
    np.testing.assert_array_equal(location_features(frame), expected)
    frame.loc[0, 'plate_x'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        location_features(frame)


def test_july_evaluator_rejects_later_dates():
    with pytest.raises(ValueError, match='July'):
        LocationEvaluator().fit(pd.DataFrame({'game_date': ['2025-08-01'], 'game_pk': [1]}))


def test_cluster_bootstrap_and_location_cells():
    summary = paired_games(np.full(6, -.1), [1, 1, 2, 2, 3, 3], replicates=100)
    np.testing.assert_allclose(summary['ci95'], [-.1, -.1])
    assert summary['games'] == 3
    frame = pd.DataFrame({'plate_x': [-2., 2., 0., 0., 0.], 'plate_z': [2.5, 2.5, 0., 5., 2.5]})
    assert location_cells(frame).tolist() == [9, 10, 11, 12, 4]


def test_legality_removes_impossible_double_play_without_reassigning_to_out():
    p = np.full((2, 10), .1)
    result = legal(p, pd.DataFrame({'outs_when_up': [2, 0], 'bases': [1, 1]}))
    assert result[0, 9] == 0 and result[1, 9] == .1
    np.testing.assert_allclose(result.sum(1), 1)
    assert result[0, 3] == pytest.approx(1/9)
