"""Synthetic contracts for E4 workload rebuilds: strictly prior rows, no future columns."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.workload_features import audit, rebuild


def frame():
    # Pitcher 1: game 10 on 05-20 (PA1 two pitches, PA2 one pitch), game 11 on 05-25.
    # Pitcher 2 relieves mid-PA 2 of game 10. Rows are shuffled on purpose.
    rows = [
        ("2023-05-20", 10, 1, 1, 1), ("2023-05-20", 10, 1, 2, 1), ("2023-05-20", 10, 2, 1, 1),
        ("2023-05-20", 10, 2, 2, 2), ("2023-05-25", 11, 1, 1, 1), ("2023-05-01", 9, 1, 1, 1),
    ]
    f = pd.DataFrame(rows, columns=["game_date", "game_pk", "at_bat_number", "pitch_number", "pitcher"])
    f["game_date"] = pd.to_datetime(f.game_date)
    f["n_thruorder_pitcher"] = [1, 1, 1, 1, 1, 1]
    f["pitcher_days_since_prev_game"] = [19, 19, 19, np.nan, 5, np.nan]
    return f.iloc[[4, 2, 0, 5, 3, 1]]


class WorkloadTest(unittest.TestCase):
    def test_prior_count_excludes_current_and_follows_key_order(self):
        b = rebuild(frame())
        self.assertEqual(b.loc[[0, 1, 2, 3, 4, 5], "prior_pitches_game"].tolist(), [0, 1, 2, 0, 0, 0])
        self.assertTrue(b.index.equals(frame().index))

    def test_changing_later_rows_never_changes_value(self):
        f = frame()
        before = rebuild(f).loc[2]
        later = f.copy()
        later.loc[later.game_date.eq("2023-05-25"), "pitcher"] = 99
        later = later.drop(index=3)
        pd.testing.assert_series_equal(rebuild(later).loc[2], before)

    def test_days_since_uses_earlier_dates_only(self):
        b = rebuild(frame())
        self.assertTrue(np.isnan(b.loc[5, "rebuilt_days_since"]))
        self.assertEqual(b.loc[0, "rebuilt_days_since"], 19)
        self.assertEqual(b.loc[4, "rebuilt_days_since"], 5)
        self.assertTrue(np.isnan(b.loc[3, "rebuilt_days_since"]))

    def test_thruorder_counts_batters_faced_before_pa(self):
        rows = [("2024-04-01", 1, ab, 1, 7) for ab in range(1, 12)]
        f = pd.DataFrame(rows, columns=["game_date", "game_pk", "at_bat_number", "pitch_number", "pitcher"])
        self.assertEqual(rebuild(f).rebuilt_thruorder.tolist(), [1] * 9 + [2, 2])

    def test_future_columns_and_post_train_rows_refused(self):
        f = frame()
        with self.assertRaises(ValueError):
            rebuild(f.assign(pitcher_days_until_next_game=1))
        with self.assertRaises(ValueError):
            audit(f.assign(game_date=pd.Timestamp("2025-05-01")))

    def test_audit_reports_train_rows_only(self):
        r = audit(frame())
        self.assertEqual(r["train_rows"], 5)
        days = r["columns"]["pitcher_days_since_prev_game"]
        self.assertEqual(days["missing_rate"], 0.2)
        self.assertEqual(days["agreement_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
