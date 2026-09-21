"""CPU-only contracts for deterministic actual-history planning examples."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_sequence_midpa import evaluate_case, observed_history, replay_row, select_case_positions
from recommend_sequence import recommend
from pitchmdp.archetypes import HISTORY_COLUMNS
from pitchmdp.sequence_data import build_history_indices


class MidpaTests(unittest.TestCase):
    def test_history_uses_last_five_same_pa_rows_and_ignores_current_future_mutations(self):
        frame = pd.DataFrame([dict(game_pk=1, game_date='2025-06-01', at_bat_number=pa,
                                   pitch_number=pitch, physical=10*pa+pitch)
                              for pa in (1, 2) for pitch in range(1, 8)])
        indices = build_history_indices(frame)
        normalizer = SimpleNamespace(transform=lambda f: np.repeat(f.physical.to_numpy()[:, None], 8, axis=1).astype(np.float32))
        tokens, keys, mask = observed_history(frame, indices, 6, normalizer)
        np.testing.assert_array_equal(tokens[:, 0], [12, 13, 14, 15, 16])
        self.assertEqual(keys, [[1, 1, p] for p in range(2, 7)])
        self.assertEqual(mask, [True] * 5)
        frame.loc[6:, 'physical'] = 999
        changed, _, _ = observed_history(frame, indices, 6, normalizer)
        np.testing.assert_array_equal(tokens, changed)
        empty, _, mask = observed_history(frame, indices, 7, normalizer)
        self.assertEqual(empty.shape, (0, 8))
        self.assertEqual(mask, [False] * 5)
        broken = indices.copy()
        broken[6, -1] = 6
        with self.assertRaisesRegex(ValueError, 'current/future'):
            observed_history(frame, broken, 6, normalizer)

    def test_selection_is_chronological_state_only_after_existing_eligibility(self):
        frame = pd.DataFrame([
            dict(game_pk=1, game_date='2025-06-01', at_bat_number=1, pitch_number=4, strikes=2, bases=0, outs_when_up=0),
            dict(game_pk=1, game_date='2025-06-01', at_bat_number=2, pitch_number=1, strikes=0, bases=1, outs_when_up=1),
            dict(game_pk=1, game_date='2025-06-01', at_bat_number=1, pitch_number=3, strikes=2, bases=0, outs_when_up=0),
            dict(game_pk=1, game_date='2025-06-01', at_bat_number=3, pitch_number=1, strikes=0, bases=2, outs_when_up=2),
        ]).assign(pitcher=1, starter_pitcher=1, split='dev', events='home_run', final_home_win=0)
        selected, unavailable = select_case_positions(frame, np.ones(len(frame), bool), [1, 2])
        self.assertEqual(selected, [('two_strikes', 2), ('runners_and_outs_pa_start', 1)])
        self.assertEqual(len(unavailable), 2)
        frame['events'], frame['final_home_win'] = 'strikeout', 1
        again, _ = select_case_positions(frame, np.ones(len(frame), bool), [1, 2])
        self.assertEqual(again, selected)

    def test_replay_context_removes_current_physics_outcomes_and_actual_type(self):
        row = self.row().copy()
        row['plate_x'], row['effective_speed'], row['events'], row['description'], row['final_home_win'] = 999, 999, 'home_run', 'hit_into_play', 1
        clean = replay_row(row)
        self.assertFalse({'plate_x', 'effective_speed', 'events', 'description', 'final_home_win'} & set(clean.index))
        self.assertEqual(clean.pitch_type, 'UNOBSERVED_CURRENT_TYPE')

    @staticmethod
    def row():
        return pd.Series(dict(game_pk=1, game_date='2025-06-01', pitcher=10, batter=20, at_bat_number=1,
                              pitch_number=1, balls=0, strikes=0, stand='R', p_throws='R', inning=1,
                              inning_topbot='Top', outs_when_up=0, bases=0, home_score=0, away_score=0,
                              supported_pa=True, pitch_type='FF', **{c: .5 for c in HISTORY_COLUMNS}))

    def test_generalized_evaluator_preserves_original_first_pitch_reference(self):
        class Delivery:
            TIERS = [('pitcher', 'pitch_type')]

            def sample(self, frame):
                return np.zeros((len(frame), 3, 8)), np.zeros(len(frame), dtype=int)

        class Model:
            temperature = 1.

            def predict(self, arrays):
                p = np.zeros((len(arrays[0]), 10))
                p[:, :3] = [.2, .2, .1]
                p[:, 3] = .25 + .1 * np.tanh(arrays[0][:, -1, 6])
                p[:, 7] = .5 - p[:, 3]
                return p

        class Baseline:
            def predict(self, frame):
                p = np.zeros((len(frame), 10))
                p[:, :2] = .25
                p[:, 3] = .5
                return p

        class We:
            def predict_defense(self, state, defender_is_home):
                return .5 + state.outs * .03 - state.away_score * .02

        row = self.row()
        actions = [dict(pitch_type='FF', target_x_ft=x, target_z_ft=2.4, training_local_n=n,
                        training_pitch_type_n=300) for x, n in [(0., 100), (.7, 80)]]
        context = SimpleNamespace(transform=lambda f: np.column_stack([f.balls.to_numpy()/3, f.strikes.to_numpy()/2]).astype(np.float32))
        encoders = {'delivery': Delivery(), 'normalizer': SimpleNamespace(mean=np.zeros(8), scale=np.ones(8)), 'context': context}
        planning = {'we': We(), 'advancement': None, 'baseline': Baseline(), 'actions': {(10, 'R'): actions}}
        reference = recommend(Model(), encoders, planning, row)
        clean = replay_row(row)
        case = {'case_id': 'test', 'kind': 'reference', 'row': clean, 'history': np.empty((0, 8)),
                'context': context.transform(pd.DataFrame([clean]))[0], 'actions': actions,
                'history_keys': [], 'history_valid_mask': [False] * 5}
        actual = evaluate_case(Model(), encoders, planning, case)
        self.assertAlmostEqual(actual['bounded_model_internal_defense_we'], reference['bounded_defense_we'])
        self.assertEqual(actual['bounded_best_action_index'], reference['bounded_best_action_index'])
        np.testing.assert_allclose([a['bounded_model_internal_defense_we'] for a in actual['ranked_actions']],
                                   [a['bounded_defense_we'] for a in reference['ranked_actions']])
        # Actual two-strike replay now exposes nonzero immediate strikeout mass.
        clean['pitch_number'], clean['strikes'] = 3, 2
        case.update(history=np.zeros((2, 8)), context=context.transform(pd.DataFrame([clean]))[0],
                    history_keys=[[1, 1, 1], [1, 1, 2]], history_valid_mask=[False] * 3 + [True] * 2)
        midpa = evaluate_case(Model(), encoders, planning, case)
        self.assertEqual(midpa['actual_history_length'], 2)
        self.assertTrue(all(a['root_strikeout_probability'] > 0 for a in midpa['ranked_actions']))


if __name__ == '__main__':
    unittest.main()
