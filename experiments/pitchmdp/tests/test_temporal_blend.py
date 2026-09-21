"""Synthetic CPU contracts for prospective membership and disjoint temporal CAL."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from run_temporal_blend import assign_fold, select_cohort, samples_for
from pitchmdp.data import KEY
from pitchmdp.sequence_data import build_history_indices
from audit_temporal_blend import compare_scores, digest, verify_hashes


def observations():
    rows = []
    for pitcher in range(1, 8):
        rows.append(dict(game_date='2023-06-01', pitcher=pitcher,
                         outs_recorded=30-pitcher, split='dev'))
    for date in ('2024-05-01', '2024-05-15', '2024-05-16', '2024-05-31',
                 '2024-06-01', '2024-06-30', '2024-07-01', '2024-09-30'):
        # Six TRAIN-selected pitchers remain selected even when only one returns.
        for pitcher in (1, 7):
            rows.append(dict(game_date=date, pitcher=pitcher, outs_recorded=3, split='train'))
    frame = pd.DataFrame(rows)
    frame['game_date'] = pd.to_datetime(frame.game_date)
    frame['game_pk'] = np.arange(len(frame)) + 100
    frame['at_bat_number'] = 1
    frame['pitch_number'] = 1
    frame['starter_pitcher'] = frame.pitcher
    frame['outs_ambiguous_extra'] = 0
    frame['player_name'] = frame.pitcher.map(lambda p: f'Pitcher {p}')
    frame['description'] = 'ball'
    frame['events'] = None
    frame['supported_pa'] = True
    frame['pitch_type'] = 'FF'
    frame['plate_x'] = 0.
    frame['plate_z'] = 2.5
    frame['balls'] = 0
    frame['strikes'] = 0
    return frame


class TemporalBlendContracts(unittest.TestCase):
    def test_dates_override_old_splits_with_exact_boundaries_and_no_mutation(self):
        dates = ['2023-05-14', '2023-05-15', '2024-04-30', '2024-05-01',
                 '2024-05-15', '2024-05-16', '2024-05-31', '2024-06-01',
                 '2024-06-30', '2024-07-01', '2024-09-30', '2024-10-01',
                 '2025-04-30']
        raw = pd.DataFrame({'game_date': pd.to_datetime(dates), 'split': 'train'})
        before = raw.copy(deep=True)
        result = assign_fold(raw, 2024)
        self.assertEqual(result.split.tolist(), ['unused', 'train', 'train',
            'earlystop', 'earlystop', 'temperature', 'temperature', 'blend',
            'blend', 'dev', 'dev'])
        self.assertTrue(result.index.equals(pd.RangeIndex(len(result))))
        pd.testing.assert_frame_equal(raw, before)
        with self.assertRaises(ValueError):
            assign_fold(raw, 2026)

    def test_second_fold_is_expanding_training_with_fresh_calibration(self):
        frame = pd.DataFrame({'game_date': pd.to_datetime(['2023-05-15',
            '2024-08-01', '2025-04-30', '2025-05-16', '2025-06-30', '2025-07-01']),
            'split': 'unused'})
        self.assertEqual(assign_fold(frame, 2025).split.tolist(),
                         ['train', 'train', 'train', 'temperature', 'blend', 'dev'])

    def test_cohort_is_invariant_to_future_workload_and_eligibility(self):
        frame = assign_fold(observations(), 2024)
        # Ranking must include observed innings even for unsupported training PAs.
        frame.loc[frame.pitcher.eq(6), 'supported_pa'] = False
        expected, _ = select_cohort(frame)
        self.assertEqual(expected, [1, 2, 3, 4, 5, 6])
        changed = frame.copy()
        future = ~changed.split.eq('train')
        changed.loc[future, 'outs_recorded'] = 1000000
        changed.loc[future, 'pitcher'] = 999
        changed.loc[future, 'starter_pitcher'] = 999
        changed.loc[future, 'player_name'] = 'Future superstar'
        self.assertEqual(select_cohort(changed)[0], expected)
        self.assertEqual(select_cohort(frame.loc[frame.split.eq('train')])[0], expected)

    def test_cohort_ties_use_train_starts_then_id_and_exclude_relief(self):
        frame = assign_fold(observations(), 2024)
        train = frame.loc[frame.split.eq('train')].copy()
        train['outs_recorded'] = 10
        second_start = train.loc[train.pitcher.eq(2)].copy()
        second_start['game_pk'] = 900
        second_start['outs_recorded'] = 0
        relief = train.iloc[[0]].copy()
        relief['pitcher'], relief['outs_recorded'] = 888, 99999
        combined = pd.concat([train, second_start, relief], ignore_index=True)
        self.assertEqual(select_cohort(combined)[0], [2, 1, 3, 4, 5, 6])

    def test_calibration_samples_are_cohort_targeted_and_game_disjoint(self):
        frame = assign_fold(observations(), 2024)
        ids, _ = select_cohort(frame)
        full, parts = samples_for(frame, ids)
        self.assertEqual(len(full), 7)
        self.assertEqual(set(parts['earlystop'].pitcher), {1, 7})
        for name in ('temperature', 'blend', 'dev'):
            self.assertEqual(set(parts[name].pitcher), {1})
            self.assertTrue(parts[name].pitcher.eq(parts[name].starter_pitcher).all())
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6])
        for name, part in parts.items():
            self.assertTrue(part.split.eq(name).all())
            self.assertFalse(part.duplicated(KEY).any())
            for other, otherpart in parts.items():
                if name != other:
                    self.assertTrue(set(part.game_pk).isdisjoint(otherpart.game_pk))
        _, again = samples_for(frame, ids)
        for name in parts:
            pd.testing.assert_frame_equal(parts[name], again[name])

    def test_absent_aggregate_calibration_fails_without_replacement(self):
        frame = assign_fold(observations(), 2024)
        ids, _ = select_cohort(frame)
        frame.loc[frame.split.eq('temperature') & frame.pitcher.isin(ids), 'supported_pa'] = False
        with self.assertRaisesRegex(ValueError, 'Empty aggregate'):
            samples_for(frame, ids)

    def test_duplicate_game_across_periods_is_rejected(self):
        frame = assign_fold(observations(), 2024)
        ids, _ = select_cohort(frame)
        train_game = frame.loc[frame.split.eq('train'), 'game_pk'].iloc[0]
        frame.loc[frame.split.eq('blend'), 'game_pk'] = train_game
        with self.assertRaisesRegex(ValueError, 'games overlap'):
            samples_for(frame, ids)

    def test_history_keeps_prior_ineligible_pitch_without_future_rows(self):
        frame = observations().iloc[[0, 0, 0]].copy().reset_index(drop=True)
        frame['pitch_number'] = [1, 2, 3]
        frame['supported_pa'] = [False, True, True]
        folded = assign_fold(frame, 2024)
        history = build_history_indices(folded)
        np.testing.assert_array_equal(history[1], [-1, -1, -1, -1, 0])
        np.testing.assert_array_equal(history[2], [-1, -1, -1, 0, 1])
        self.assertTrue((history < np.arange(len(folded))[:, None]).all())

    def test_saved_artifact_audit_rejects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'result.json'
            path.write_text('{"n": 2}')
            hashes = {'result.json': digest(path)}
            verify_hashes(root, hashes)
            path.write_text('{"n": 3}')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                verify_hashes(root, hashes)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                verify_hashes(root, {'../outside.json': 'forbidden'})

    def test_independent_metrics_detect_changed_labels(self):
        labels = np.array([0, 1])
        probability = np.zeros((2, 10), dtype=np.float32)
        probability[:, :2] = [[.75, .25], [.25, .75]]
        recorded = {'n': 2, 'log_loss': -np.log(.75), 'brier_multiclass': .125, 'accuracy': 1.}
        compare_scores(labels, probability, recorded)
        with self.assertRaises(AssertionError):
            compare_scores(labels[::-1], probability, recorded)


if __name__ == '__main__':
    unittest.main()
