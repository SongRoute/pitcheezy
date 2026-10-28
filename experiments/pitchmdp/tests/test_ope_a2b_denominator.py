"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 B2 denominator: the B2 decision stream (logging := pi_b_hat_B2 and
rho recomputed only in the evaluated prefix of E0 PAs; positivity, non-E0 and post-refusal rows untouched), the B2
validity rule and its counts, decision vectors through the frozen context rows (agreement checks), the A3 weight
diagnostics with the ESS ratio, and G-INT-R3 at relative 1e-9. The frozen-B2 loading and the end-to-end path with
a small LightGBM B2 run in a fresh interpreter (one OpenMP runtime, the A3 launch rule):
isolated_ope_a2b_denominator_lightgbm.py. No real data, ledger or model is read.
"""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ope_a2b_fixtures import b2_stream, b2_vectors, e0_ids_of, evaluator_inputs  # noqa: E402
from pitchmdp import ope_denominator_a2b as dn  # noqa: E402
from pitchmdp import policy_estimator as est  # noqa: E402
from pitchmdp.behavior_model import BehaviorTable  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decisions, cls.facts, _ = evaluator_inputs(seed=9)
        cls.e0 = dn.e0_map(cls.facts)
        cls.e0_ids = e0_ids_of(cls.facts)
        cls.vectors = b2_vectors(cls.decisions)

    def test_the_stream_equals_the_reference_construction(self):
        mine = list(dn.b2_decision_stream(self.decisions, self.vectors, self.e0_ids)())
        reference = list(b2_stream(self.decisions, self.vectors, self.e0_ids)())
        self.assertEqual(mine, reference)
        self.assertEqual(len(mine), len(self.decisions))

    def test_only_the_evaluated_prefix_of_e0_pas_changes(self):
        stream = list(dn.b2_decision_stream(self.decisions, self.vectors, self.e0_ids)())
        changed = 0
        for old, new in zip(self.decisions, stream):
            if old['status'] == est.POSITIVITY or old['pa_id'] not in self.e0_ids or not isinstance(old.get('result'), dict):
                self.assertEqual(old, new)
                continue
            if new['result']['logging'] != old['result']['logging']:
                changed += 1
                a = new['result']['logged_index']
                v = np.asarray(new['result']['logging'])
                for policy in ('candidate', 'reference'):
                    self.assertEqual(new['result'][f'rho_{policy}'], float(np.asarray(old['result'][policy])[a] / v[a]))
                self.assertEqual(new['result']['q_reference'], old['result']['q_reference'])
                self.assertEqual(new['result']['candidate'], old['result']['candidate'])
        self.assertGreater(changed, 100)
        for row in stream:  # the protected estimator accepts every recomputed row
            if row['status'] in est.EVALUATED and row['pa_id'] in self.e0_ids:
                est._step(row, len(row['result']['logging']))

    def test_validity_rule_and_counts(self):
        ok = dn.b2_validity(self.decisions, self.vectors, self.e0_ids)
        self.assertTrue(ok['pass'])
        self.assertGreater(ok['decisions_checked'], 0)
        target = next(d for d in self.decisions if d['status'] == 'SUPPORTED' and d['pa_id'] in self.e0_ids)
        key = (target['pa_id'], target['decision_index'])
        zero = b2_vectors(self.decisions, zero_at={key})
        bad = dn.b2_validity(self.decisions, zero, self.e0_ids)
        self.assertFalse(bad['pass'])
        self.assertEqual(bad['zero_mass_with_positive_pi'], 1)
        self.assertEqual(bad['nonfinite_rho'], 1)
        self.assertEqual(bad['support_differs_from_b0'], 1)
        missing = dict(self.vectors)
        missing.pop(key)
        self.assertEqual(dn.b2_validity(self.decisions, missing, self.e0_ids)['missing_vectors'], 1)
        malformed = dict(self.vectors)
        malformed[key] = [v * 1.01 for v in malformed[key]]
        self.assertEqual(dn.b2_validity(self.decisions, malformed, self.e0_ids)['malformed_vectors'], 1)
        with self.assertRaisesRegex(IntegrityError, 'B2 denominator invalid'):
            dn.require_valid_vectors(bad)

    def test_weight_diagnostics_and_the_identity_denominator(self):
        common = dict(large=10.0, mei=0.001, draws=100, seed=3287529750)
        weights, b0 = dn.b2_weights(self.decisions, self.e0, self.vectors, **common)
        self.assertEqual(set(weights['E5']), {'thresholds', 'checks', 'pass'})
        self.assertEqual(weights['E5']['thresholds'], {'pa': 100, 'game': 30})
        self.assertIsNotNone(weights['kl_new_vs_b0']['mean'])
        self.assertGreater(weights['kl_new_vs_b0']['mean'], 0)
        self.assertIn('paired', weights['weight_shift_D'])
        self.assertEqual(weights['large_weight_threshold'], 10.0)
        identity = {(d['pa_id'], d['decision_index']): list(d['result']['logging']) for d in self.decisions
                    if d['status'] in est.EVALUATED and isinstance(d.get('result'), dict)}
        same, _ = dn.b2_weights(self.decisions, self.e0, identity, **common)
        self.assertEqual(same['kl_new_vs_b0']['mean'], 0.0)
        self.assertEqual(same['weight_shift_D']['paired'], 0.0)
        self.assertEqual(same['ess_ratio_vs_b0'], {p: {'pa': 1.0, 'game': 1.0} for p in ('candidate', 'reference')})
        self.assertEqual(same['policies'], b0['policies'])

    def test_gint_r3_relative_tolerance(self):
        common = dict(large=10.0, mei=0.001, draws=100, seed=3287529750)
        weights, _ = dn.b2_weights(self.decisions, self.e0, self.vectors, **common)
        record = dn.gint_r3(weights, copy.deepcopy(weights))
        self.assertEqual(record['result'], 'PASS')
        near = copy.deepcopy(weights)
        near['policies']['candidate']['ess']['pa'] *= 1 + 1e-10
        self.assertEqual(dn.gint_r3(weights, near)['result'], 'PASS')
        far = copy.deepcopy(weights)
        far['policies']['candidate']['ess']['pa'] *= 1 + 2e-9
        bad = dn.gint_r3(weights, far)
        self.assertEqual((bad['result'], bad['mismatched_keys']), ('FAIL', ['policies']))
        flag = copy.deepcopy(weights)
        flag['E5']['pass'] = not flag['E5']['pass']
        self.assertEqual(dn.gint_r3(weights, flag)['result'], 'FAIL')


class VectorTests(unittest.TestCase):
    def table(self):
        frame = pd.DataFrame({'game_pk': [700001] * 3, 'at_bat_number': [1, 1, 1], 'pitch_number': [1, 2, 3],
                              'pitcher': [501] * 3, 'batter_side': ['R'] * 3, 'balls': [0, 1, 1], 'strikes': [0, 0, 1],
                              'previous': ['<START>', 'FF', 'SL']})
        return BehaviorTable(frame)

    def rows(self):
        out = []
        for i, (b, s, hist) in enumerate(((0, 0, []), (1, 0, ['FF']), (1, 1, ['FF', 'SL']))):
            out.append({'pa_id': '700001:1', 'decision_index': i, 'context_key': f'700001:1:{i + 1}', 'pitcher': '501',
                        'batter_side': 'R', 'balls': b, 'strikes': s, 'history_actions': hist, 'status': 'SUPPORTED'})
        return out

    class Model:
        def batch(self, table, rows):
            rows = np.asarray(rows)
            out = np.zeros((len(rows), 18))
            out[:, 0], out[:, 1] = 0.25 + 0.1 * rows, 0.75 - 0.1 * rows
            return out

    def test_vectors_follow_the_frozen_context_rows(self):
        vectors = dn.decision_vectors(self.Model(), self.table(), self.rows())
        self.assertEqual(sorted(vectors), [('700001:1', 0), ('700001:1', 1), ('700001:1', 2)])
        self.assertAlmostEqual(vectors[('700001:1', 2)][0], 0.45)
        rows = self.rows()
        rows[1]['balls'] = 0  # the ledger state disagrees with its frozen pre-pitch row
        with self.assertRaisesRegex(IntegrityError, 'disagree with their frozen context rows'):
            dn.decision_vectors(self.Model(), self.table(), rows)
        rows = self.rows()
        rows[2]['context_key'] = '700001:1:9'
        with self.assertRaisesRegex(IntegrityError, 'no frozen context row'):
            dn.decision_vectors(self.Model(), self.table(), rows)
        self.assertEqual(dn.decision_vectors(self.Model(), self.table(), []), {})


class IsolatedLightGBMTests(unittest.TestCase):
    def test_frozen_b2_paths_run_with_one_openmp_runtime(self):
        """The frozen-B2 loading (pins, manifest) and the end-to-end B2 rows with a small LightGBM B2 run in a
        fresh interpreter whose torch and LightGBM share one OpenMP runtime (A3 launch rule)."""
        from behavior_calibration_synthetic import run_isolated
        done = run_isolated('isolated_ope_a2b_denominator_lightgbm.py', timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-3000:])
        self.assertIn(' passed', done.stdout)
        self.assertNotIn('failed', done.stdout)


if __name__ == '__main__':
    unittest.main()
