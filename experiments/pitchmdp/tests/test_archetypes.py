"""Leakage, low-evidence, and frozen-transform contracts for batter styles."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.archetypes import (Archetypes, HISTORY_COLUMNS, INITIAL_RATES,
                                RELIABILITY_COLUMNS, STYLE_COLUMNS,
                                add_batter_style_history)


class BatterStyleContracts(unittest.TestCase):
    @staticmethod
    def pitches():
        return pd.DataFrame({
            'batter': [1, 1, 2, 1, 1, 2, 1],
            'game_date': pd.to_datetime(['2024-04-01']*3 + ['2024-04-02']*3 + ['2024-04-03']),
            'events': ['home_run', 'strikeout', 'walk', 'single', 'field_out', 'double', None],
            'description': ['hit_into_play', 'swinging_strike', 'ball', 'hit_into_play',
                            'hit_into_play', 'hit_into_play', 'called_strike'],
            'is_pa_terminal': [True]*6 + [False],
            'launch_angle': [30., np.nan, np.nan, 0., np.nan, 20., np.nan],
            'split': ['train']*7,
        })

    def test_same_date_exclusion_and_nonterminal_date_included(self):
        df = add_batter_style_history(self.pitches())
        np.testing.assert_allclose(df.loc[:2, list(STYLE_COLUMNS)], np.tile(INITIAL_RATES, (3, 1)))
        np.testing.assert_allclose(df.loc[3, list(HISTORY_COLUMNS)].to_numpy(float),
                                   df.loc[4, list(HISTORY_COLUMNS)].to_numpy(float))
        # Two first-day swings, one contact; no same-day single enters April 2.
        league_contact = (1 + 500*.76)/(2 + 500)
        self.assertAlmostEqual(df.loc[3, STYLE_COLUMNS[0]], (1 + 100*league_contact)/102, places=6)
        self.assertAlmostEqual(df.loc[6, RELIABILITY_COLUMNS[0]], 4/104, places=6)
        # Missing launch angle on the field out adds no groundball denominator.
        self.assertAlmostEqual(df.loc[6, RELIABILITY_COLUMNS[5]], 2/62, places=6)

    def test_future_and_current_outcomes_cannot_change_previous_profiles(self):
        base = self.pitches()
        changed = base.copy()
        changed.loc[changed.game_date.ge('2024-04-02'), ['events', 'description', 'launch_angle']] = ['home_run', 'hit_into_play', 45.]
        left, right = add_batter_style_history(base), add_batter_style_history(changed)
        np.testing.assert_array_equal(left.loc[:5, list(HISTORY_COLUMNS)], right.loc[:5, list(HISTORY_COLUMNS)])
        # Batting history must not depend on input sorting or frame index.
        shuffled = self.pitches().sample(frac=1, random_state=12)
        add_batter_style_history(shuffled)
        np.testing.assert_array_equal(left[list(HISTORY_COLUMNS)], shuffled.sort_index()[list(HISTORY_COLUMNS)])

    def test_unknown_player_uses_prior_league_and_zero_evidence(self):
        df = self.pitches()
        df.loc[6, 'batter'] = 999
        add_batter_style_history(df)
        self.assertTrue(df.loc[6, list(RELIABILITY_COLUMNS)].eq(0).all())
        self.assertAlmostEqual(df.loc[6, STYLE_COLUMNS[4]], (4 + 500*.165)/(5 + 500), places=6)

    def test_fit_is_train_only_transform_has_no_id_and_is_frozen(self):
        df = add_batter_style_history(self.pitches())
        archetypes = Archetypes(n_clusters=2).fit(df)
        centers = archetypes.centers.copy()
        x = df[list(HISTORY_COLUMNS)].copy()  # Inference does not even contain ID.
        p = archetypes.transform(x)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(p[:3], .5)
        self.assertTrue(np.isfinite(archetypes.numeric_features(x)).all())
        self.assertEqual(archetypes.numeric_features(x).shape, (7, 14))
        altered = x.copy()
        altered.loc[6, STYLE_COLUMNS[4]] = 3.
        changed_p = archetypes.transform(altered)
        np.testing.assert_array_equal(changed_p[:6], p[:6])
        np.testing.assert_array_equal(archetypes.centers, centers)
        df.loc[6, 'split'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'train'):
            Archetypes(n_clusters=2).fit(df)

    def test_pitch_duplication_does_not_reweight_centers(self):
        df = add_batter_style_history(self.pitches())
        first = Archetypes(n_clusters=2).fit(df)
        repeated = pd.concat([df, df.iloc[[3]*100]], ignore_index=True)
        second = Archetypes(n_clusters=2).fit(repeated)
        np.testing.assert_array_equal(first.centers, second.centers)
        self.assertEqual(first.n_snapshots, 5)


if __name__ == '__main__':
    unittest.main()
