import numpy as np
import pandas as pd

from scripts.b_history_diagnosis import contribution, loss


def test_loss_and_partition_contributions_reconcile():
    y = np.array([0, 1, 0])
    p0 = np.array([[.8, .2], [.4, .6], [.6, .4]])
    p1 = np.array([[.7, .3], [.3, .7], [.5, .5]])
    l0, l1 = loss(y, p0), loss(y, p1)
    frame = pd.DataFrame({'game_pk': [1, 1, 2], 'loss0': l0, 'loss1': l1, 'delta': l1-l0})
    parts = contribution(frame, ['game_pk'])
    assert sum(part['pitches'] for part in parts) == len(y)
    np.testing.assert_allclose(sum(part['global_contribution'] for part in parts), (l1-l0).mean())
