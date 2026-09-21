"""CPU contracts for frozen frequency restoration and CAL-only convex blends."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline
from run_sequence_frequency_blend import restore_frequency_model, fit_selected_blends, apply_selected_blends


class FrequencyBlendContracts(unittest.TestCase):
    def test_portable_tables_restore_exact_probabilities_without_refitting(self):
        train = pd.DataFrame({"pitcher": [1, 1, 2, 2], "pitch_type": ["FF", "FF", "SL", "SL"],
            "balls": [0]*4, "strikes": [0]*4, "stand": ["R"]*4, "p_throws": ["R"]*4,
            "game_date": pd.to_datetime(["2024-01-01"]*4), "split": ["train"]*4,
            "description": ["ball", "called_strike", "ball", "foul"], "events": [None]*4})
        for use_pitcher in (False, True):
            original = HierarchicalFrequencyBaseline(use_pitcher).fit(train)
            restored = restore_frequency_model({"report": original.report, "global_probability": original.parent.global_p,
                "count_hand_table": original.parent.table, "type_table": original.type_table,
                "pitcher_table": original.pitcher_table if use_pitcher else None})
            query = train.copy()
            query["description"], query["events"] = "hit_into_play", "home_run"
            query["split"] = "dev"
            query.loc[0, "pitcher"], query.loc[1, "pitch_type"] = 999, "UNSEEN"
            np.testing.assert_array_equal(restored.predict(query), original.predict(query))

    def test_brier_weight_matches_closed_form_interior_solution(self):
        labels = np.array([0, 1, 0, 1])
        ensemble = np.array([[.9,.1], [.2,.8], [.2,.8], [.8,.2]])
        baseline = np.array([[.5,.5], [.5,.5], [.6,.4], [.3,.7]])
        difference = ensemble-baseline
        expected = np.clip(np.sum(difference*(np.eye(2)[labels]-baseline))/np.sum(difference**2), 0, 1)
        fitted = fit_selected_blends(labels, ensemble, baseline)
        self.assertAlmostEqual(fitted["brier_multiclass"]["model_weight"], expected, places=5)
        self.assertGreater(expected, 0)
        self.assertLess(expected, 1)

    def test_exact_endpoints_and_dev_inputs_cannot_change_frozen_weights(self):
        labels = np.array([0, 0])
        ensemble, baseline = np.array([[.1,.9]]*2), np.array([[.9,.1]]*2)
        fitted = fit_selected_blends(labels, ensemble, baseline)
        before = {key: dict(value) for key, value in fitted.items()}
        for objective in fitted:
            self.assertEqual(fitted[objective]["model_weight"], 0.)
        predicted = apply_selected_blends(fitted, ensemble[::-1], baseline[::-1])
        for p in predicted.values():
            np.testing.assert_array_equal(p, baseline[::-1])
        apply_selected_blends(fitted, baseline, ensemble)
        self.assertEqual(fitted, before)

    def test_probability_and_weight_guards(self):
        p = np.array([[.5,.5], [.5,.5]])
        fit = fit_selected_blends(np.array([0, 1]), p, p)
        bad = p.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            apply_selected_blends(fit, bad, p)
        with self.assertRaises(ValueError):
            apply_selected_blends(fit, p[:1], p)
        fit["log_loss"]["model_weight"] = -1
        with self.assertRaises(ValueError):
            apply_selected_blends(fit, p, p)


if __name__ == "__main__":
    unittest.main()
