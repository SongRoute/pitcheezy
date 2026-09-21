"""Targeted data-contract checks for boundary outs and chronology leakage."""
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.data import add_history, add_splits, add_transitions_and_support, reconstruct_outs


class DataContracts(unittest.TestCase):
    def test_outs_include_pickoff_and_third_out_and_substitution_bounds(self):
        df = pd.DataFrame({
            "game_pk": [1] * 5, "inning": [1, 1, 1, 1, 2],
            "inning_topbot": ["Top", "Top", "Top", "Bot", "Top"],
            "outs_when_up": [0, 1, 2, 0, 0], "pitcher": [10, 10, 11, 20, 11],
            "events": [None, None, "strikeout", "field_out", "field_out"],
        })
        got = reconstruct_outs(df)
        self.assertEqual(got.outs_recorded.tolist(), [1, 1, 1, 3, 1])
        self.assertEqual(got.outs_ambiguous_extra.tolist(), [0, 1, 0, 0, 0])

    def test_previous_pitch_resets_and_daily_history_excludes_doubleheader(self):
        df = pd.DataFrame({
            "game_pk": [1, 1, 2, 3], "at_bat_number": [1, 1, 1, 1],
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-01", "2024-04-01", "2024-04-02"]),
            "pitch_type": ["FF", "SL", "CH", "FF"], "description": ["ball", "hit_into_play", "hit_into_play", "called_strike"],
            "plate_x": [0., .1, .2, .3], "plate_z": [2., 2., 2., 2.],
            "batter": [9] * 4, "is_pa_terminal": [False, True, True, True],
            "events": [None, "single", "home_run", "strikeout"],
        })
        add_history(df)
        self.assertEqual(df.prev_pitch_type.tolist(), ["START", "FF", "START", "START"])
        self.assertEqual(df.batter_pa_prior.tolist(), [0, 0, 0, 2])
        self.assertAlmostEqual(df.batter_obp_prior.iloc[-1], (2 + 30 * .32) / 32)

    def test_split_boundary_is_date_not_row_or_game_order(self):
        df = pd.DataFrame({"game_date": pd.to_datetime(["2023-05-14", "2023-05-15", "2025-04-30", "2025-05-01", "2025-06-30", "2025-07-01"])})
        add_splits(df)
        self.assertEqual(df.split.tolist(), ["history", "train", "train", "calibration", "calibration", "dev"])

    @staticmethod
    def walk_fixture(next_runner_base=1, final_inning=9):
        """Minimal complete-game log retaining the exact first PA transition."""
        rows = []
        for ball in range(4):
            rows.append({"at_bat_number": 1, "pitch_number": ball + 1, "balls": ball,
                         "strikes": 0, "description": "ball", "events": "walk" if ball == 3 else None})
        rows.append({"at_bat_number": 2, "pitch_number": 1, "balls": 0, "strikes": 0,
                     "description": "hit_into_play", "events": "field_out",
                     f"on_{next_runner_base}b": 99})
        rows.append({"at_bat_number": 3, "pitch_number": 1, "balls": 0, "strikes": 2,
                     "description": "called_strike", "events": "strikeout",
                     "inning": final_inning, "inning_topbot": "Bot", "outs_when_up": 2})
        defaults = {"game_pk": 1, "inning": 1, "inning_topbot": "Top", "outs_when_up": 0,
                    "home_score": 0, "away_score": 2, "post_home_score": 0, "post_away_score": 2,
                    "on_1b": None, "on_2b": None, "on_3b": None,
                    "game_type": "R", "pitcher": 10, "batter": 20,
                    "pitch_type": "FF", "plate_x": 0., "plate_z": 2.5, "sz_bot": 1.5, "sz_top": 3.5}
        df = pd.DataFrame([{**defaults, **row} for row in rows])
        for col, values in reconstruct_outs(df).items():
            df[col] = values
        add_transitions_and_support(df)
        return df

    def test_deterministic_award_excludes_unmodeled_extra_runner_advancement(self):
        valid = self.walk_fixture(next_runner_base=1)
        unsupported = self.walk_fixture(next_runner_base=2)
        self.assertTrue(valid.loc[valid.at_bat_number.eq(1), "supported_pa"].all())
        self.assertFalse(unsupported.loc[unsupported.at_bat_number.eq(1), "supported_pa"].any())
        self.assertEqual(valid.loc[3, "next_bases"], 1)

    def test_complete_game_labels_reject_short_or_unobserved_game_end(self):
        complete = self.walk_fixture()
        shortened = self.walk_fixture(final_inning=8)
        self.assertTrue(complete.complete_game.all())
        self.assertTrue(complete.final_home_win.eq(0).all())
        self.assertFalse(shortened.complete_game.any())
        self.assertTrue(shortened.final_home_win.isna().all())
        self.assertFalse(complete.iloc[-1].supported_pa)


if __name__ == "__main__":
    unittest.main()
