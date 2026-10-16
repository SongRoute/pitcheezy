"""Final-only adapter on generated frames; no manifest/parquet/model access."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.ws_fullseason import data, final_source as final
from test_ws_fullseason import rows


class FinalProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = data.closure()
        cls.base = json.loads((Path(__file__).resolve().parents[1] / 'configs/EXP-WS-FULLSEASON-T-001.json').read_text())

    def inputs(self):
        recipe = copy.deepcopy(self.base)
        recipe.update(execution_stage='FINAL_REFIT', production_cutoff=final.CUTOFF)
        recipe['row_policy'].update(end=final.CUTOFF, game_types=list(final.TYPES))
        source = rows()[self.M.cp.PROJECT]
        authority = {'mode': 'PRODUCTION_FIT', 'root_approved': True, 'sealed_assets_read': False,
                     'sources': copy.deepcopy(final.SOURCES)}
        return {'S2': source, 'I3': source.iloc[:0].copy()}, recipe, authority

    def test_regular_only_matches_global_legacy_projection(self):
        frames, recipe, authority = self.inputs()
        before = frames['S2'].copy(deep=True)
        expected, census = self.M.cp.project(before, recipe, self.M.safe.outcome_labels, self.M.safe.head_labels)
        actual, report, fit, oof, y7 = final.project(frames, recipe, self.M, authority)
        pd.testing.assert_frame_equal(expected, actual)
        pd.testing.assert_frame_equal(before, frames['S2'])
        for name, value in census.items(): self.assertEqual(report[name], value)
        self.assertTrue(fit.all()); self.assertFalse(oof.any()); self.assertEqual(len(y7), len(actual))

    def test_original_postseason_types_and_lineage_restored(self):
        frames, recipe, authority = self.inputs(); post = rows()[self.M.cp.PROJECT]
        post['game_pk'] += 100
        for i, kind in enumerate(('F', 'D', 'L', 'W')):
            post.loc[i*3:i*3+2, 'game_type'] = kind
            post.loc[i*3:i*3+2, 'game_date'] = f'2026-10-{i+1:02}'
        frames['I3'] = post
        g, c, _, _, _ = final.project(frames, recipe, self.M, authority)
        self.assertEqual(c['accepted_game_types'], {'R': 12, 'F': 3, 'D': 3, 'L': 3, 'W': 3})
        self.assertEqual(c['source_lineage']['S2']['accepted_unique_keys'], 12)
        self.assertEqual(c['source_lineage']['I3']['accepted_unique_keys'], 12)
        self.assertEqual(set(g.loc[g.game_pk >= 900100, 'game_type']), {'F', 'D', 'L', 'W'})
        self.assertEqual(set(frames['I3'].game_type), {'F', 'D', 'L', 'W'})

    def test_nonallowed_type_excluded_with_global_denominator(self):
        frames, recipe, authority = self.inputs(); extra = frames['S2'].iloc[:3].copy()
        extra['game_pk'] += 100; extra['game_type'] = 'S'; frames['I3'] = extra
        g, c, _, _, _ = final.project(frames, recipe, self.M, authority)
        self.assertEqual((len(g), c['source_rows'], c['excluded']['excluded_non_registered_game_type'], c['stop_share_denominator']), (12, 15, 3, 12))
        self.assertEqual(c['source_lineage']['I3']['nonaccepted_rows'], 3)

    def test_original_type_and_value_conflicts_stop_before_mapping(self):
        for field, value in [('game_type', 'F'), ('balls', 1)]:
            frames, recipe, authority = self.inputs(); frames['I3'] = frames['S2'].iloc[[0]].copy()
            frames['I3'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'conflicting original'):
                final.project(frames, recipe, self.M, authority)
        frames, recipe, authority = self.inputs(); frames['I3'] = frames['S2'].iloc[[0]].copy()
        frames['I3']['pitch_number'] = 99; frames['I3']['game_type'] = 'F'
        with self.assertRaisesRegex(ValueError, 'mixed original game types'): final.project(frames, recipe, self.M, authority)

    def test_exact_duplicate_and_multidate_keep_legacy_fates(self):
        frames, recipe, authority = self.inputs()
        frames['I3'] = frames['S2'].iloc[:3].copy()
        recipe['row_policy']['stop_share']['excluded_duplicate_key'] = 1
        g, c, _, _, _ = final.project(frames, recipe, self.M, authority)
        self.assertEqual((len(g), c['excluded']['excluded_duplicate_key']), (12, 3))
        self.assertEqual(c['source_lineage']['I3']['accepted_unique_keys'], 0)
        frames, recipe, authority = self.inputs(); frames['S2'].loc[0, 'game_date'] = '2026-03-26'
        recipe['row_policy']['stop_share']['quarantined_multi_date_game'] = 1
        _, c, _, _, _ = final.project(frames, recipe, self.M, authority)
        self.assertEqual((c['multi_date_games'], c['excluded']['quarantined_multi_date_game']), (1, 3))

    def test_global_stop_share_not_partitioned_by_source_or_type(self):
        frames, recipe, authority = self.inputs(); post = frames['S2'].iloc[:3].copy()
        post['game_pk'] += 100; post['game_type'] = 'F'; post.loc[0, 'bb_type'] = 'ground_ball'
        frames['I3'] = post; recipe['row_policy']['stop_share']['excluded_nonbip_with_bb_type'] = .07
        # One bad row /15 global passes; one bad row /3 in an isolated type would fail.
        _, c, _, _, _ = final.project(frames, recipe, self.M, authority)
        self.assertEqual((c['stop_share_denominator'], c['excluded']['excluded_nonbip_with_bb_type']), (15, 1))

    def test_historical_segregation_is_only_lifted_for_fixed_final_authority(self):
        frames, recipe, authority = self.inputs(); post = frames['S2'].iloc[:3].copy()
        post['game_pk'] = final.HISTORICAL_SEGREGATION[0]; post['game_type'] = 'F'; frames['I3'] = post
        _, c, _, _, _ = final.project(frames, recipe, self.M, authority)
        self.assertEqual(c['historical_segregation_rows_from_official_source'], 3)
        self.assertEqual(recipe['row_policy']['segregation_game_pks'], list(final.HISTORICAL_SEGREGATION))
        with self.assertRaisesRegex(ValueError, 'segregated postseason'): data.project(post, self.base, self.M)
        for field, value in [('mode', 'OOF'), ('root_approved', False), ('sealed_assets_read', True)]:
            bad = copy.deepcopy(authority); bad[field] = value
            with patch.object(final.pd, 'concat') as concat, self.assertRaises(ValueError): final.project(frames, recipe, self.M, bad)
            concat.assert_not_called()
        authority['sources']['I3']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): final.project(frames, recipe, self.M, authority)


if __name__ == '__main__': unittest.main()
