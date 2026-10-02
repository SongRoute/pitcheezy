"""Synthetic T2 exclusion/replay checks; no real data or neural fitting."""
from pathlib import Path
import hashlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.archetypes import HISTORY_COLUMNS, add_batter_style_history
from pitchmdp.data import KEY
from pitchmdp.matrix_generalization import (GeneralizationFold, LEGACY_PRIORS,
    select_heldout_pitchers, select_heldout_batters, audit_new_matchups)
from pitchmdp.matrix_panel import SELECTOR_VERSION
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_data import PhysicalNormalizer


def observations():
    games = [
        ('2023-03-01', 'history', 1, 10, 201), ('2023-03-02', 'unused', 2, 20, 100),
        ('2024-04-01', 'train', 3, 10, 200), ('2024-04-02', 'train', 4, 20, 100),
        ('2024-04-03', 'train', 5, 20, 201), ('2024-04-04', 'train', 6, 30, 202),
        ('2024-04-05', 'train', 7, 40, 203), ('2024-04-06', 'train', 8, 50, 204),
        ('2024-04-07', 'train', 9, 99, 205), ('2024-04-08', 'train', 10, 10, 206),
        ('2025-05-02', 'earlystop', 11, 20, 201), ('2025-05-03', 'earlystop', 12, 10, 100),
        ('2025-05-20', 'temperature', 13, 20, 201), ('2025-05-21', 'temperature', 14, 10, 100),
        ('2025-06-05', 'blend', 15, 20, 201), ('2025-06-06', 'blend', 16, 10, 100),
        ('2025-07-01', 'dev', 21, 10, 100), ('2025-07-01', 'dev', 22, 10, 100),
        ('2025-07-01', 'dev', 23, 10, 100), ('2025-07-02', 'dev', 24, 10, 100),
        ('2025-07-03', 'dev', 25, 10, 100), ('2025-07-04', 'dev', 26, 20, 201)]
    rows = []
    for date, split, game, pitcher, batter in games:
        for number in (1, 2):
            rows.append({'game_date': pd.Timestamp(date), 'split': split, 'game_pk': game,
                'at_bat_number': 1, 'pitch_number': number, 'pitcher': pitcher, 'batter': batter,
                'pitch_type': 'FF' if number == 1 else 'SL', 'description': 'ball' if number == 1 else 'hit_into_play',
                'events': None if number == 1 else 'single', 'is_pa_terminal': number == 2,
                'launch_angle': np.nan if number == 1 else 15., 'supported_pa': True,
                'plate_x': .1*number, 'plate_z': 2.+.1*number, 'balls': number-1, 'strikes': 0,
                'effective_speed': 90.+game/100, 'release_spin_rate': 2000.+game,
                'spin_axis': game*5., 'pfx_x': game/100, 'pfx_z': game/200,
                'p_throws': 'R', 'stand': 'L', 'outs_when_up': 0, 'inning': 1,
                'inning_topbot': 'Top', 'home_score': 0, 'away_score': 0, 'bases': 0})
    frame = pd.DataFrame(rows)
    frame.loc[frame.game_pk.eq(10) & frame.pitch_number.eq(1), 'pitcher'] = 20
    # Deliberately stale/infected derived columns must be overwritten or removed.
    for name in [*LEGACY_PRIORS, *HISTORY_COLUMNS]:
        frame[name] = 9876.
    frame['prev_description'] = 'stale_future_label'
    return frame


def row(frame, game, pitch=2):
    return int(np.flatnonzero(frame.game_pk.eq(game) & frame.pitch_number.eq(pitch))[0])


class GeneralizationTests(unittest.TestCase):
    def test_selection_uses_only_frozen_train_panel_and_hashes(self):
        panel = {'version': SELECTOR_VERSION, 'pitcher_ids': [1, 2, 3, 4], 'strata': [
            {'role': 'starter', 'hand': 'R', 'volume': 'high', 'pitcher_ids': [1, 2, 3]},
            {'role': 'relief', 'hand': 'L', 'volume': 'low', 'pitcher_ids': [4]},
            {'role': 'relief', 'hand': 'R', 'volume': 'middle', 'pitcher_ids': []}]}
        selected = select_heldout_pitchers(panel)
        expected = min([1, 2, 3], key=lambda pid: hashlib.sha256(
            f'ml_t2_pitcher_v1|20260924|{pid}'.encode()).hexdigest())
        self.assertEqual(selected['ids'], sorted([expected, 4]))
        self.assertEqual(selected['strata'][-1]['status'], 'empty')
        frame = observations()
        train = frame.loc[frame.split.eq('train')]
        a = select_heldout_batters(train)
        b = select_heldout_batters(train.iloc[::-1].assign(description='hit_by_pitch'))
        self.assertEqual(a['ids'], b['ids'])
        self.assertEqual(a['train_counts'], b['train_counts'])
        with self.assertRaises(ValueError):
            select_heldout_batters(frame)

    def test_whole_pa_fitting_purge_and_pretrain_exclusion(self):
        frame = observations()
        fold = GeneralizationFold(frame, 'pitcher', [10])
        self.assertTrue((~fold.allowed_fit['train'][frame.game_pk.eq(10)]).all())
        for split, mask in fold.allowed_fit.items():
            self.assertTrue(frame.loc[mask, 'split'].eq(split).all())
            self.assertFalse(frame.loc[mask, 'pitcher'].eq(10).any())
        self.assertFalse(fold.observation_mask('Z')[row(frame, 1)])
        self.assertFalse(fold.observation_mask('O')[row(frame, 1)])
        self.assertTrue(fold.observation_mask('Z')[row(frame, 2)])

    def test_prefix_order_and_post_date_evaluation_are_paired(self):
        frame = observations()
        fold = GeneralizationFold(frame, 'pitcher', [10, 99])
        self.assertEqual(fold.prefix_records[0]['games'], [21, 22])
        self.assertEqual(set(frame.loc[fold.target_eval, 'game_pk']), {24, 25})
        self.assertEqual(fold.prefix_records[1]['status'], 'unmeasured_no_post_prefix_rows')
        self.assertFalse(fold.observation_mask('O')[row(frame, 23)])
        self.assertFalse(fold.observation_mask('O')[row(frame, 24)])
        for regime in ('Z', 'W', 'O'):
            np.testing.assert_array_equal(fold.evaluation_mask(regime), fold.target_eval)
        self.assertEqual(set(frame.loc[fold.prefix, 'game_pk']), {21, 22})

    def test_excluded_outcomes_and_physics_cannot_change_allowed_train_or_z(self):
        frame = observations()
        for axis, entity in [('pitcher', 10), ('batter', 100)]:
            fold = GeneralizationFold(frame, axis, [entity])
            changed = frame.copy()
            changed.loc[fold.heldout, 'description'] = 'swinging_strike'
            changed.loc[fold.heldout, 'events'] = 'home_run'
            changed.loc[fold.heldout, 'launch_angle'] = -30.
            changed.loc[fold.heldout, 'effective_speed'] = 999.
            other = GeneralizationFold(changed, axis, [entity])
            left, right = fold.reconstruct('Z'), other.reconstruct('Z')
            columns = [*LEGACY_PRIORS, *HISTORY_COLUMNS]
            # Includes profiles of OTHER batters whose league fallback could leak.
            np.testing.assert_array_equal(left.feature_frame[columns], right.feature_frame[columns])
            lp, rp = fold.auxiliary_fit_plan(), other.auxiliary_fit_plan()
            pd.testing.assert_frame_equal(lp['normalizer_train'], rp['normalizer_train'])
            pd.testing.assert_frame_equal(lp['supervised_train'], rp['supervised_train'])
            normalizer = PhysicalNormalizer().fit(lp['normalizer_train'])
            types = sorted(lp['type_vocabulary_train'].pitch_type.unique())
            lstore = left.history_store(normalizer=normalizer, type_vocabulary=types)
            rstore = right.history_store(normalizer=normalizer, type_vocabulary=types)
            rows = np.flatnonzero(fold.target_eval)
            for a, b in zip(lstore.gather(rows, np.zeros((len(rows), 8))),
                            rstore.gather(rows, np.zeros((len(rows), 8)))):
                np.testing.assert_array_equal(a, b)

    def test_two_views_preserve_truth_and_restore_legal_w_tokens(self):
        frame = observations()
        before = frame.copy(deep=True)
        fold = GeneralizationFold(frame, 'pitcher', [10])
        rec = fold.reconstruct('W')
        rows = np.flatnonzero(fold.target_eval)
        self.assertTrue((outcome_labels(rec.statistics_frame)[rows] == -1).all())
        np.testing.assert_array_equal(outcome_labels(rec.feature_frame)[rows], fold.truth_labels[rows])
        self.assertTrue(fold.truth_eligible[rows].all())
        self.assertNotIn('prev_description', rec.feature_frame)
        plan = fold.auxiliary_fit_plan()
        normalizer = PhysicalNormalizer().fit(plan['normalizer_train'])
        store = rec.history_store(normalizer=normalizer, type_vocabulary=['FF', 'SL'])
        tokens, valid = store.gather([row(frame, 24)], np.zeros((1, 8)))
        self.assertEqual(valid.sum(), 2)
        self.assertEqual(tokens[0, -2, -11:].argmax(), 0)  # actual prior ball, not sanitized unknown
        self.assertTrue((tokens[0, -1, -11:] == 0.).all())
        with self.assertRaises(ValueError):
            store.gather([row(frame, 24)])
        with self.assertRaises(ValueError):
            store.gather([row(frame, 3)])  # forbidden training PA
        pd.testing.assert_frame_equal(frame, before)

    def test_w_changes_only_observed_prior_tokens_and_o_only_prefix_statistics(self):
        frame = observations()
        fold = GeneralizationFold(frame, 'pitcher', [10])
        changed = frame.copy()
        changed.loc[row(frame, 24, 1), 'description'] = 'called_strike'
        other = GeneralizationFold(changed, 'pitcher', [10])
        plan = fold.auxiliary_fit_plan()
        normalizer = PhysicalNormalizer().fit(plan['normalizer_train'])
        stores = [f.reconstruct('W').history_store(normalizer=normalizer, type_vocabulary=['FF', 'SL'])
                  for f in (fold, other)]
        first = [s.gather([row(frame, 24, 1)], np.zeros((1, 8)))[0] for s in stores]
        np.testing.assert_array_equal(*first)  # current label never visible
        following = [s.gather([row(frame, 24, 2)], np.zeros((1, 8)))[0] for s in stores]
        self.assertFalse(np.array_equal(*following))
        for regime in ('Z', 'W', 'O'):
            np.testing.assert_array_equal(fold.reconstruct(regime).feature_frame[list(HISTORY_COLUMNS)],
                                           other.reconstruct(regime).feature_frame[list(HISTORY_COLUMNS)])
        prefix_changed = frame.copy()
        prefix_changed.loc[fold.prefix, 'description'] = 'swinging_strike'
        prefix_changed.loc[fold.prefix, 'events'] = 'strikeout'
        prefix_other = GeneralizationFold(prefix_changed, 'pitcher', [10])
        left, right = fold.reconstruct('O'), prefix_other.reconstruct('O')
        train_mask = fold.allowed_fit['train']
        pd.testing.assert_frame_equal(left.feature_frame.loc[train_mask], right.feature_frame.loc[train_mask])
        self.assertFalse(np.array_equal(left.feature_frame.loc[fold.target_eval, list(HISTORY_COLUMNS)],
                                       right.feature_frame.loc[fold.target_eval, list(HISTORY_COLUMNS)]))

    def test_pretrain_league_contribution_is_actually_removed(self):
        frame = observations()
        unsanitized = add_batter_style_history(frame.copy())
        sanitized = GeneralizationFold(frame, 'pitcher', [10]).reconstruct('Z').feature_frame
        # Pitcher 10's March game faced batter 201; excluding it changes both
        # that batter and the league fallback for another April batter.
        columns = list(HISTORY_COLUMNS)
        self.assertFalse(np.array_equal(unsanitized.loc[row(frame, 5), columns], sanitized.loc[row(frame, 5), columns]))
        self.assertFalse(np.array_equal(unsanitized.loc[row(frame, 6), columns], sanitized.loc[row(frame, 6), columns]))

    def test_visibility_rejects_cross_pa_and_future_references(self):
        frame = observations()
        fold = GeneralizationFold(frame, 'pitcher', [10])
        target = row(frame, 24)
        with self.assertRaises(ValueError):
            fold.h5_visibility([target], [[-1, -1, -1, -1, row(frame, 23)]], 'W')
        with self.assertRaises(ValueError):
            fold.h5_visibility([target], [[-1, -1, -1, -1, target]], 'O')

    def test_new_matchup_audits_external_pretrain_cal_and_first_dev_meeting(self):
        frame = observations()
        train = frame.loc[frame.split.eq('train')]
        # Construct new 30/201 pair: both identities known individually.
        query = frame.loc[frame.game_pk.isin([24, 25])].copy()
        query['pitcher'], query['batter'] = 30, 201
        result = audit_new_matchups(query, history_sources={'complete_log': frame}, fitting_train=train)
        self.assertEqual(query.loc[result['mask'], 'game_pk'].tolist(), [24, 24])
        for old_split, old_date in [('history', '2023-03-01'), ('temperature', '2025-05-20'), ('blend', '2025-06-05')]:
            hidden = query.iloc[:1].copy()
            hidden['game_pk'], hidden['game_date'], hidden['split'] = 999, pd.Timestamp(old_date), old_split
            blocked = audit_new_matchups(query, history_sources={'complete_log': frame, 'another_history': hidden}, fitting_train=train)
            self.assertFalse(blocked['mask'].any())
        earlier = query.iloc[:1].copy()
        earlier['game_pk'], earlier['game_date'] = 20, pd.Timestamp('2025-07-01')
        earlier_result = audit_new_matchups(query, history_sources={'complete_log': frame, 'earlier_dev': earlier}, fitting_train=train)
        self.assertFalse(earlier_result['mask'].any())

    def test_temporal_and_source_contracts_fail_closed(self):
        frame = observations()
        fold = GeneralizationFold(frame, 'pitcher', [10])
        inherited = PhysicalNormalizer().fit(frame.loc[frame.split.eq('train')])
        with self.assertRaises(ValueError):
            fold.reconstruct('Z').history_store(normalizer=inherited, type_vocabulary=['FF', 'SL'])
        changed = frame.copy()
        changed.loc[changed.split.eq('dev'), 'split'] = 'train'
        with self.assertRaises(ValueError):
            GeneralizationFold(changed, 'pitcher', [10])
        changed = frame.copy()
        changed.loc[0, 'game_date'] = pd.Timestamp('2026-01-01')
        with self.assertRaises(ValueError):
            GeneralizationFold(changed, 'pitcher', [10])
        with self.assertRaises(ValueError):
            GeneralizationFold(frame, 'pitcher', [123456789])
        with self.assertRaises(ValueError):
            audit_new_matchups(frame, history_sources={'actual_train': frame},
                               fitting_train=frame.loc[frame.split.eq('train')])


if __name__ == '__main__':
    unittest.main()
