"""Physical sequence chronology, keyed joins, and frozen preprocessing contracts."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.sequence_data import (HistoryStore, PHYSICAL_COLUMNS, PhysicalNormalizer,
                                   build_history_indices, join_physics)


class SequenceDataContracts(unittest.TestCase):
    @staticmethod
    def frame():
        frame = pd.DataFrame({"game_pk": [1]*8 + [2]*2,
                              "at_bat_number": [1]*7 + [2] + [1]*2,
                              "pitch_number": list(range(1, 8)) + [1, 1, 2],
                              "game_date": pd.to_datetime(["2024-01-01"]*8 + ["2025-07-01"]*2),
                              "split": ["train"]*8 + ["dev"]*2})
        for j, column in enumerate(PHYSICAL_COLUMNS):
            frame[column] = np.arange(len(frame), dtype=float) + j*10
        return frame

    def test_history_order_padding_and_resets(self):
        frame = self.frame()
        history = build_history_indices(frame)
        np.testing.assert_array_equal(history[0], [-1]*5)
        np.testing.assert_array_equal(history[2], [-1, -1, -1, 0, 1])
        np.testing.assert_array_equal(history[6], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(history[7:9], [[-1]*5]*2)
        np.testing.assert_array_equal(history[9], [-1, -1, -1, -1, 8])
        self.assertTrue(np.all((history < np.arange(len(frame))[:, None])))

    def test_current_future_mutations_never_change_past_tokens(self):
        frame = self.frame()
        normalizer = PhysicalNormalizer().fit(frame.iloc[:8])
        before = HistoryStore.from_frame(frame, normalizer)
        altered = frame.copy()
        altered.loc[4:, list(PHYSICAL_COLUMNS)] = 900.
        after = HistoryStore.from_frame(altered, normalizer)
        left, mask = before.gather([4])
        right, _ = after.gather([4])
        np.testing.assert_array_equal(left[:, :-1], right[:, :-1])
        self.assertFalse(np.array_equal(left[:, -1], right[:, -1]))
        np.testing.assert_array_equal(mask, [[False, True, True, True, True, True]])
        self.assertTrue(np.all(left[~mask] == 0))
        candidate, _ = after.gather([4], current=before.physical[[4]])
        np.testing.assert_array_equal(left, candidate)

    def test_normalization_is_train_only_and_frozen(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "train"):
            PhysicalNormalizer().fit(frame)
        frame.loc[0, "effective_speed"] = np.nan
        frame.loc[:, "pfx_x"] = np.nan
        norm = PhysicalNormalizer().fit(frame.iloc[:8])
        mean = norm.mean.copy()
        output = norm.transform(frame)
        self.assertEqual(output.shape, (10, 8))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())
        frame.loc[8:, list(PHYSICAL_COLUMNS)] = 1e8
        np.testing.assert_array_equal(norm.transform(frame)[:8], output[:8])
        np.testing.assert_array_equal(norm.mean, mean)
        # Angular wrap is represented continuously, not 0 versus 360 degrees.
        frame.loc[0, "spin_axis"], frame.loc[1, "spin_axis"] = 0., 360.
        np.testing.assert_allclose(norm.transform(frame)[0, 2:4], norm.transform(frame)[1, 2:4])

    def test_join_uses_keys_and_preserves_old_columns(self):
        frame = self.frame()
        sidecar = frame[["game_pk", "at_bat_number", "pitch_number", "effective_speed", "spin_axis"]]
        old = frame.drop(columns=["effective_speed", "spin_axis"])
        joined = join_physics(old, sidecar.sample(frac=1, random_state=42))
        pd.testing.assert_frame_equal(joined[frame.columns], frame)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            join_physics(old, pd.concat([sidecar, sidecar.iloc[[0]]]))
        wrong = sidecar.copy()
        wrong.loc[0, "game_pk"] = 999
        with self.assertRaisesRegex(ValueError, "missing"):
            join_physics(old, wrong)

    def test_misordered_and_noncontiguous_pas_are_rejected(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "increasing"):
            build_history_indices(frame.iloc[[1, 0, *range(2, 10)]])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_history_indices(frame.iloc[[0, 7, *range(1, 7), 8, 9]])


if __name__ == "__main__":
    unittest.main()
