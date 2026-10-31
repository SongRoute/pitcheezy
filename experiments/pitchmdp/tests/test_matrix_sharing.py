import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from pitchmdp.matrix_sharing import SharingPredictor, fit_pitcher_clusters, training_arrays


class Constant:
    def __init__(self, label, width):
        self.label, self.width = label, width
    def logits(self, arrays):
        assert arrays[2].shape[1] == self.width  # Routing identities never enter model.
        result = np.zeros((len(arrays[0]), 10))
        result[:, self.label] = 3
        return result


def test_personal_and_partial_routing_preserve_unknown_fallback():
    context = np.zeros((3, 11), dtype=np.float32)
    context[:, -2:] = [[0, 101], [0, 102], [-1, 999]]
    arrays = np.zeros((3, 6, 30), dtype=np.float32), np.ones((3, 6), bool), context
    clusters = {'pitcher_counts': {'101': 1000, '102': 100}, 'cluster_counts': {'0': 10000}}
    global_model = Constant(0, 4)
    model = SharingPredictor('G4-partial', global_model, clusters,
        personal_models={101: Constant(2, 4)}, cluster_models={0: Constant(1, 9)})
    result = softmax(model.logits(arrays), axis=1)
    g, c, i = [softmax(np.eye(10)[label] * 3) for label in (0, 1, 2)]
    np.testing.assert_allclose(result[0], .5 * i + .5 * (.5 * c + .5 * g))
    np.testing.assert_allclose(result[1], .5 * c + .5 * g)
    np.testing.assert_allclose(result[2], g)
    assert training_arrays(arrays)[2].shape[1] == 4


def test_clusters_deterministic_and_refuse_dev():
    frame = pd.DataFrame({'pitcher': np.repeat(np.arange(8), 4), 'split': 'train',
        'effective_speed': np.repeat(np.arange(8) + 85, 4), 'release_spin_rate': 2200.,
        'pfx_x': np.tile([0., 1., 2., 3.], 8), 'pfx_z': 1.,
        'pitch_type': ['FF', 'SL', 'FF', 'CH'] * 8, 'p_throws': ['R'] * 16 + ['L'] * 16})
    first = fit_pitcher_clusters(frame)
    assert first == fit_pitcher_clusters(frame.sample(frac=1, random_state=9))
    assert sum(first['cluster_counts'].values()) == len(frame)
    frame.loc[0, 'split'] = 'dev'
    with pytest.raises(ValueError, match='TRAIN'):
        fit_pitcher_clusters(frame)
