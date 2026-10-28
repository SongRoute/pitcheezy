"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 sealed evaluator: the 12 B0 rows of the 25-row evaluator are
bit-identical to the EXP-OPE-RES-01 ``rehearsal_evaluate`` on a synthetic ledger at the same seeds; the S6 mode
computes and records only the 12 A2 pair rows and refuses the 13 others; the S6 mode equals the 12-row part of the
25-row mode; the lambda-fixed row never reaches the public file (selected lambda 100 or not) and its public-form items
are sealed until A2b-UB (G-CODE MAJOR-1); lambda = inf identity; the B2 rows (identity denominator = B0 rows, invalid denominator = invalid rows,
classification checked); the strict public file; the G-INT-EVAL integrity record (absolute 1e-12, counts and
lambda / M by equality). No real data, ledger or model is read.
"""
import copy
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ope_a2b_fixtures import b2_stream, b2_vectors, e0_ids_of, evaluator_inputs  # noqa: E402
from pitchmdp import ope_diagnostics as od  # noqa: E402
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp import ope_sealed as osd  # noqa: E402
from pitchmdp import ope_sealed_a2b as sa  # noqa: E402
from pitchmdp import ope_variants as ov  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

A2_SEEDS = dict(rb.EVALUATOR_SEEDS[rb.MODE_S6])


def bits(value):
    """Exact float bytes of every float in a nested structure (bit identity, not closeness)."""
    if isinstance(value, dict):
        return [(str(k), bits(value[k])) for k in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [bits(v) for v in value]
    if isinstance(value, np.ndarray):
        return bits(value.tolist())
    if isinstance(value, (float, np.floating)):
        return struct.pack('<d', float(value))
    return value


def weights_record(valid=True):
    return {'validity': {'pass': valid, 'decisions_checked': 3, 'zero_mass_with_positive_pi': 0 if valid else 1,
                         'nonfinite_rho': 0, 'support_differs_from_b0': 0}, 'weights': None}


def a2_results(public, sealed):
    """The EXP-OPE-RES-01 results-file layout built from the A2 evaluator's own outputs (synthetic record)."""
    return {'S3_public': json.loads(osd.seal(public, sealed)[0]),
            'UB_unblinded': {'selection': osd._jsonable(sealed['selection']),
                             'pairs': osd._jsonable({p: {k: sealed['pairs'][p][k] for k in ('bootstrap', 'center',
                                                                                            'shift_center')
                                                         if k in sealed['pairs'][p]} for p in reg.PAIRS}),
                             'decision_record_sha256': 'f' * 64}}


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decisions, cls.facts, cls.common = evaluator_inputs()
        cls.stream = staticmethod(lambda: iter(cls.decisions))
        result0, rows0, gint = osd.gint_gate(cls.stream, cls.facts, references={}, draws=200, minimum_games=5,
                                             minimum_pa_starts=10)
        a2_kw = {k: v for k, v in cls.common.items()}
        cls.a2_public, cls.a2_sealed, _ = osd.rehearsal_evaluate(
            cls.stream, cls.facts, baseline=(result0, rows0), gint=gint, r3_status='NOT_RUN:R3', **a2_kw)
        cls.gint = {'G-INT-A2': 'PASS (synthetic)'}
        cls.public6, cls.sealed6, _ = sa.evaluate(cls.stream, cls.facts, mode=rb.MODE_S6, gint=cls.gint, **cls.common)
        cls.e0_ids = e0_ids_of(cls.facts)
        cls.vectors = b2_vectors(cls.decisions)
        cls.public25, cls.sealed25, _ = sa.evaluate(
            cls.stream, cls.facts, mode=rb.MODE_REHEARSAL, seeds=A2_SEEDS, gint=cls.gint,
            decisions_b2=b2_stream(cls.decisions, cls.vectors, cls.e0_ids), b2=weights_record(), **cls.common)

    def test_s6_mode_is_bit_identical_to_the_a2_evaluator(self):
        for row in rb.S6_MODE_ROWS:
            pair = rb.a2_pair(row)
            self.assertEqual(bits(self.public6['pairs'][row]), bits(self.a2_public['pairs'][pair]), row)
            self.assertEqual(self.public6['S5'][row], self.a2_public['S5'][pair])
            for section in ('e', 'f'):
                self.assertEqual(bits(self.public6['R5'][section][row]), bits(self.a2_public['R5'][section][pair]))
            self.assertEqual(bits(self.sealed6['pairs'][row]), bits(self.a2_sealed['pairs'][pair]), row)
        for section in ('a', 'b', 'c', 'd', 'g', 'h', 'i'):
            self.assertEqual(bits(self.public6['R5'][section]), bits(self.a2_public['R5'][section]), section)
        for key in ('r1_variance', 'integrity', 'validity'):
            self.assertEqual(bits(self.public6[key]), bits(self.a2_public[key]), key)
        rename = lambda name: name.replace('xB0', '')
        self.assertEqual(bits({rename(k): v for k, v in self.sealed6['selection'].items()}), bits(self.a2_sealed['selection']))
        self.assertEqual(bits({rename(k): v for k, v in self.sealed6['grid_points'].items()}),
                         bits(self.a2_sealed['grid_points']))
        for key in ('placebo', 'corr_r_r_tilde', 'p_counts', 'counts'):
            self.assertEqual(bits(self.sealed6[key]), bits(self.a2_sealed[key]), key)
        for key in ('pigeonhole', 'p_nonpoint'):  # keyed by row: compare under the A2 pair names
            self.assertEqual(bits({rename(k): v for k, v in self.sealed6[key].items()}), bits(self.a2_sealed[key]), key)
        self.assertEqual(bits({rename(k): v for k, v in self.sealed6['grid_bootstrap']['game'].items()}),
                         bits(self.a2_sealed['grid_bootstrap']['game']))

    def test_the_25_row_evaluator_reproduces_a2_on_its_12_pair_part(self):
        for row in rb.S6_MODE_ROWS:
            pair = rb.a2_pair(row)
            self.assertEqual(bits(self.public25['pairs'][row]), bits(self.a2_public['pairs'][pair]), row)
            self.assertEqual(self.public25['S5'][row], self.a2_public['S5'][pair])
            self.assertEqual(bits(self.sealed25['pairs'][row]), bits(self.a2_sealed['pairs'][pair]), row)
        for section in ('a', 'b', 'c', 'd', 'g', 'h', 'i'):
            self.assertEqual(bits(self.public25['R5'][section]), bits(self.a2_public['R5'][section]), section)
        self.assertEqual(set(self.public25['pairs']), set(rb.PUBLIC_ROWS))  # 24 public rows
        self.assertEqual(set(self.sealed25['pairs']) & set(rb.ROWS), set(rb.ROWS))  # 25 computed rows, sealed records
        self.assertEqual(self.public25['counts'], {'rows': 24})

    def test_s6_mode_equals_the_12_row_part_of_the_25_row_mode(self):
        for row in rb.S6_MODE_ROWS:
            self.assertEqual(bits(self.public6['pairs'][row]), bits(self.public25['pairs'][row]), row)
            self.assertEqual(self.public6['S5'][row], self.public25['S5'][row])
            self.assertEqual(bits(self.sealed6['pairs'][row]), bits(self.sealed25['pairs'][row]), row)
            for section in ('e', 'f'):
                self.assertEqual(bits(self.public6['R5'][section][row]), bits(self.public25['R5'][section][row]))
        b0 = lambda d: {k: v for k, v in d.items() if k.endswith('xB0')}
        self.assertEqual(bits(b0(self.sealed6['selection'])), bits(b0(self.sealed25['selection'])))
        self.assertEqual(bits(self.sealed6['placebo']), bits(self.sealed25['placebo']))

    def test_s6_mode_refuses_the_13_rows_and_every_b2_input(self):
        with self.assertRaisesRegex(IntegrityError, 'refuses rows'):
            sa.evaluate(self.stream, self.facts, mode=rb.MODE_S6, rows=rb.ROWS, gint=self.gint, **self.common)
        for row in (rb.LAMBDA_FIXED_ROW, 'R1_0xR2_0xB2'):
            with self.subTest(row), self.assertRaisesRegex(IntegrityError, 'refuses rows'):
                sa.evaluate(self.stream, self.facts, mode=rb.MODE_S6, rows=(*rb.S6_MODE_ROWS, row), gint=self.gint,
                            **self.common)
        with self.assertRaisesRegex(IntegrityError, 'refuses every B2 input'):
            sa.evaluate(self.stream, self.facts, mode=rb.MODE_S6, gint=self.gint,
                        decisions_b2=b2_stream(self.decisions, self.vectors, self.e0_ids), **self.common)
        with self.assertRaisesRegex(IntegrityError, 'EXP-OPE-RES-01 seeds'):
            sa.evaluate(self.stream, self.facts, mode=rb.MODE_S6, gint=self.gint,
                        seeds=dict(rb.EVALUATOR_SEEDS[rb.MODE_REHEARSAL]), **self.common)
        self.assertEqual(set(self.public6['pairs']), set(rb.S6_MODE_ROWS))
        self.assertFalse(any(k.endswith('xB2') or k == rb.LAMBDA_FIXED_ROW for k in self.sealed6['pairs']))
        self.assertNotIn('lambda_fixed_row_public', self.sealed6)
        self.assertNotIn('B2', self.public6)
        with self.assertRaisesRegex(IntegrityError, 'integrity record is required'):
            sa.evaluate(self.stream, self.facts, mode=rb.MODE_S6, gint=None, **self.common)

    def forced(self, value):
        """The 25-row evaluation with the selected lambda forced to ``value`` (every component and denominator)."""
        real = ov.select_parameter

        def force(kind, by_value, clusters):
            chosen_value, objectives = real(kind, by_value, clusters)
            return (value if kind == 'shrink' else chosen_value), objectives
        with mock.patch.object(sa.ov, 'select_parameter', force):
            return sa.evaluate(
                self.stream, self.facts, mode=rb.MODE_REHEARSAL, seeds=A2_SEEDS, gint=self.gint,
                decisions_b2=b2_stream(self.decisions, self.vectors, self.e0_ids), b2=weights_record(), **self.common)

    def test_the_lambda_fixed_row_is_sealed_until_unblind(self):
        """G-CODE MAJOR-1: whether lambda = 100 is selected or not, the public file carries no key and no value of the
        lambda = 100 row (next to R1_b x R2_1 it would reveal the selection); its public-form items (pairs, S5, R5 e /
        f) are in the sealed file only, with the descriptive equality flag."""
        held = {}
        for value in (100.0, 30.0):
            with self.subTest(selected=value):
                public, sealed, _ = self.forced(value)
                self.assertEqual(sealed['selection']['R1_bxR2_1xB0']['selected'], str(value))
                public_bytes, sealed_bytes = sa.seal(public, sealed)
                shown = json.loads(public_bytes)
                self.assertNotIn(rb.LAMBDA_FIXED_ROW, public_bytes.decode())
                self.assertNotIn('lambda100', public_bytes.decode())
                for section in (shown['pairs'], shown['S5'], shown['R5']['e'], shown['R5']['f']):
                    self.assertEqual(set(section), set(rb.PUBLIC_ROWS))
                self.assertEqual(shown['counts'], {'rows': 24})
                fixed = held[value] = sealed['lambda_fixed_row_public']
                self.assertEqual((fixed['row'], fixed['lambda'], fixed['decision_input']), (rb.LAMBDA_FIXED_ROW, 100.0, False))
                self.assertEqual(set(fixed), {'row', 'lambda', 'pairs', 'S5', 'R5', 'equals_R1_bxR2_1xB0', 'decision_input',
                                              'disclosure'})
                self.assertIn('lambda_fixed_row_public', sealed_bytes.decode())
                chosen = public['pairs']['R1_bxR2_1xB0']
                if value == 100.0:
                    self.assertTrue(fixed['equals_R1_bxR2_1xB0'])
                    self.assertEqual(bits(fixed['pairs']), bits(chosen))  # shown once only, as the selected row
                else:
                    self.assertFalse(fixed['equals_R1_bxR2_1xB0'])
                    grid = sealed['grid_bootstrap']['game']['grid:R1_bxR2_1xB0:100.0']['mean_p']
                    self.assertEqual(fixed['pairs']['H_point']['game'], grid['half_width'])
                    secret = {v for _, v in rb.leaves([fixed['pairs'], fixed['R5']]) if isinstance(v, float) and v != 0.0}
                    floats = {v for _, v in rb.leaves(shown) if isinstance(v, float)}
                    self.assertTrue(secret)
                    self.assertFalse(secret & floats)  # no value of the lambda = 100 row reaches the public file
        self.assertEqual(bits(held[100.0]['pairs']), bits(held[30.0]['pairs']))  # the fixed row ignores the selection
        self.assertEqual(bits(held[100.0]['R5']), bits(held[30.0]['R5']))
        self.assertNotIn(rb.LAMBDA_FIXED_ROW, self.public25['pairs'])
        self.assertIn('center', self.sealed25['pairs'][rb.LAMBDA_FIXED_ROW])  # its sealed record as every row's

    def test_lambda_inf_identity(self):
        self.assertEqual(self.public25['integrity']['inf_variants_reproduce_baseline'], 'PASS')
        real = ov.transform

        def broken(kind, param=None):
            if param == rb.INF and kind == 'shrink':
                return lambda r: r * (1 + 1e-9)
            return real(kind, param)
        with mock.patch.object(sa.ov, 'transform', broken):
            with self.assertRaisesRegex(IntegrityError, 'lambda = inf'):
                sa.evaluate(self.stream, self.facts, mode=rb.MODE_S6, gint=self.gint, **self.common)

    def test_identity_b2_denominator_reproduces_the_b0_rows(self):
        identity = {(d['pa_id'], d['decision_index']): list(d['result']['logging']) for d in self.decisions
                    if d['status'] in ('SUPPORTED', 'OUTSIDE_POLICY_SUPPORT')}
        public, sealed, _ = sa.evaluate(
            self.stream, self.facts, mode=rb.MODE_REHEARSAL, seeds=A2_SEEDS, gint=self.gint,
            decisions_b2=b2_stream(self.decisions, identity, self.e0_ids), b2=weights_record(), **self.common)
        for pair in reg.PAIRS:
            self.assertEqual(bits(public['pairs'][f'{pair}xB2']), bits(public['pairs'][f'{pair}xB0']), pair)
            self.assertEqual(bits(sealed['pairs'][f'{pair}xB2']['center']), bits(sealed['pairs'][f'{pair}xB0']['center']))
        self.assertEqual(public['S5']['R1_0xR2_0xB2'], {'game': 'INCONCLUSIVE', 'pitcher': 'INCONCLUSIVE'})

    def test_b2_rows_differ_and_an_invalid_denominator_invalidates_them(self):
        self.assertNotEqual(bits(self.public25['pairs']['R1_0xR2_0xB2']), bits(self.public25['pairs']['R1_0xR2_0xB0']))
        public, sealed, _ = sa.evaluate(self.stream, self.facts, mode=rb.MODE_REHEARSAL, seeds=A2_SEEDS, gint=self.gint,
                                        decisions_b2=None, b2=weights_record(valid=False), **self.common)
        for row in rb.B2_ROWS:
            self.assertEqual(public['pairs'][row]['valid'], {'game': False, 'pitcher': False}, row)
            self.assertEqual(public['pairs'][row]['H_point'], {'game': None, 'pitcher': None})
            self.assertEqual(public['S5'][row], {'game': 'INVALID', 'pitcher': 'INVALID'})
        self.assertFalse(public['B2']['validity']['pass'])
        self.assertTrue(public['pairs'][rb.BASELINE]['valid']['game'])
        with self.assertRaisesRegex(IntegrityError, 'B2 decision stream exactly when B2 is valid'):
            sa.evaluate(self.stream, self.facts, mode=rb.MODE_REHEARSAL, gint=self.gint,
                        decisions_b2=b2_stream(self.decisions, self.vectors, self.e0_ids), b2=weights_record(valid=False),
                        **self.common)

    def test_a_b2_stream_must_keep_the_b0_classification(self):
        broken = copy.deepcopy(self.decisions)
        e0 = sorted(self.e0_ids)
        for row in broken:
            if row['pa_id'] == e0[0] and isinstance(row.get('result'), dict):
                row['result']['q_reference'] = [None if v is None else v + 0.01 for v in row['result']['q_reference']]
        stream = b2_stream(broken, self.vectors, self.e0_ids)
        with self.assertRaisesRegex(IntegrityError, 'B2 values or q-hat differ'):
            sa.evaluate(self.stream, self.facts, mode=rb.MODE_REHEARSAL, gint=self.gint, decisions_b2=stream,
                        b2=weights_record(), **self.common)

    def test_public_file_is_the_typed_allowlist_and_centers_stay_sealed(self):
        public_bytes, sealed_bytes = sa.seal(self.public25, self.sealed25)
        public = json.loads(public_bytes)
        self.assertTrue(rb.validate_public(public))
        self.assertEqual(public['sealed_sha256'], sa.sha256_bytes(sealed_bytes))
        text = json.dumps(public)
        for word in ('selected', 'center', '"delta"', 'shift', 'objective', 'placebo_delta', 'mean_p_ci95'):
            self.assertNotIn(word, text)
        self.assertEqual(public['experiment'], 'EXP-OPE-RES-02')
        self.assertEqual(len(public['pairs']), 24)
        self.assertIn('selected', sealed_bytes.decode())
        leak = copy.deepcopy(self.public25)
        leak['pairs']['R1_bxR2_1xB0']['center'] = 1e-4
        with self.assertRaises(IntegrityError):
            sa.seal(leak, self.sealed25)
        fixed = self.sealed25['lambda_fixed_row_public']
        for section, value in (('pairs', fixed['pairs']), ('S5', fixed['S5'])):  # the fixed row is no public key
            leak = copy.deepcopy(self.public25)
            leak[section][rb.LAMBDA_FIXED_ROW] = copy.deepcopy(value)
            with self.subTest(section), self.assertRaises(IntegrityError):
                sa.seal(leak, self.sealed25)
        leak = copy.deepcopy(self.public25)
        leak['R5']['e'][rb.LAMBDA_FIXED_ROW] = copy.deepcopy(fixed['R5']['e'])
        with self.assertRaises(IntegrityError):
            sa.seal(leak, self.sealed25)

    def test_the_rehearsal_runs_the_pigeonhole_with_role_9_sealed_only(self):
        """PM decision (G-CODE (c)3): the rehearsal's descriptive two-way bootstrap runs with the A2b role 9 seed and
        stays in the sealed file (as in EXP-OPE-RES-01); the S6 mode keeps the A2 seed."""
        public, sealed, _ = sa.evaluate(self.stream, self.facts, mode=rb.MODE_REHEARSAL, gint=self.gint,
                                        decisions_b2=b2_stream(self.decisions, self.vectors, self.e0_ids),
                                        b2=weights_record(), **self.common)
        self.assertEqual(sealed['seeds']['pigeonhole'], 3206983116)
        self.assertNotEqual(sealed['pigeonhole'].get('status'), 'NOT_RUN')
        self.assertEqual(set(sealed['pigeonhole']), set(self.sealed6['pigeonhole']))
        self.assertNotEqual(bits(sealed['pigeonhole']), bits(self.sealed6['pigeonhole']))  # another seed and rows
        self.assertNotIn('pigeonhole', sa.seal(public, sealed)[0].decode())
        self.assertEqual(self.sealed6['seeds']['pigeonhole'], reg.ROLE_VALUES['pigeonhole'])

    def test_placebo_copy_matches_the_a2_function(self):
        table = od.DecisionTable.from_ledger(iter(self.decisions), self.e0_ids)
        steps = [[i] for i in range(min(len(table), 40))]
        args = (steps, table, {'game': (None, np.arange(len(steps)))})
        ends = [0.5] * len(steps)
        a2 = od.placebo_check(*args, rewards=ends, placebos=3)
        mine = sa.placebo_check(*args, rewards=ends, placebos=3, stream=rb.placebo_stream(('EXP-OPE-RES-01', 'placebo')))
        self.assertEqual(bits(a2), bits(mine))
        other = sa.placebo_check(*args, rewards=ends, placebos=3, stream=rb.placebo_stream(('EXP-OPE-RES-02', 'placebo')))
        self.assertNotEqual(bits(a2[1]['placebo_delta_hat']), bits(other[1]['placebo_delta_hat']))

    def test_missing_descriptive_output_is_a_stage_integrity_failure(self):
        for path in (('R5', 'f'), ('validity',), ('B2',)):
            public = copy.deepcopy(self.public25)
            node = public
            for part in path[:-1]:
                node = node[part]
            del node[path[-1]]
            with self.subTest(path), self.assertRaisesRegex(IntegrityError, 'descriptive output missing'):
                sa.require_descriptive(public, self.sealed25, rb.ROWS, rb.MODE_REHEARSAL)
        sealed = copy.deepcopy(self.sealed25)
        del sealed['pairs'][rb.LAMBDA_FIXED_ROW]['shift_center']
        with self.assertRaisesRegex(IntegrityError, 'paired shift missing'):
            sa.require_descriptive(self.public25, sealed, rb.ROWS, rb.MODE_REHEARSAL)
        sealed = copy.deepcopy(self.sealed25)
        del sealed['lambda_fixed_row_public']
        with self.assertRaisesRegex(IntegrityError, 'lambda-fixed row items missing'):
            sa.require_descriptive(self.public25, sealed, rb.ROWS, rb.MODE_REHEARSAL)
        public = copy.deepcopy(self.public25)
        public['pairs'][rb.LAMBDA_FIXED_ROW] = self.sealed25['lambda_fixed_row_public']['pairs']
        with self.assertRaisesRegex(IntegrityError, 'never public before A2b-UB'):
            sa.require_descriptive(public, self.sealed25, rb.ROWS, rb.MODE_REHEARSAL)


class GintEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        decisions, facts, common = evaluator_inputs(seed=7)
        stream = lambda: iter(decisions)
        result0, rows0, gint = osd.gint_gate(stream, facts, references={}, draws=200, minimum_games=5,
                                             minimum_pa_starts=10)
        public, sealed, _ = osd.rehearsal_evaluate(stream, facts, baseline=(result0, rows0), gint=gint,
                                                   r3_status='NOT_RUN:R3', **common)
        cls.results = a2_results(public, sealed)
        cls.public6, cls.sealed6, _ = sa.evaluate(stream, facts, mode=rb.MODE_S6, gint={'G-INT-A2': 'PASS'}, **common)
        cls.reference = {'path': 'results/EXP-OPE-RES-01.json', 'sha256': 'a' * 64}

    def record(self, results):
        return sa.gint_eval_record(self.public6, self.sealed6, results, reference=self.reference)

    def test_the_s6_mode_reproduces_the_a2_record_exactly(self):
        record = self.record(self.results)
        self.assertEqual(record['result'], 'PASS')
        self.assertEqual(record['max_abs_delta'], 0.0)
        self.assertEqual(record['mismatched_keys'], [])
        self.assertEqual(record['not_comparable_keys'], [])
        self.assertGreater(len(record['compared_keys']), 500)
        self.assertTrue(any(k.startswith('unblinded.selection.R1_bxR2_1.selected') for k in record['compared_keys']))
        self.assertTrue(rb.validate_integrity_record(record, 'gint_eval'))
        floats = [v for _, v in rb.leaves(record) if isinstance(v, float)]
        self.assertEqual(floats, [0.0, 1e-12])  # max |delta| and the tolerance only: no center or other value

    def test_absolute_tolerance_counts_and_selected_values(self):
        def mutated(path, fn):
            results = copy.deepcopy(self.results)
            node = results
            for part in path[:-1]:
                node = node[part]
            node[path[-1]] = fn(node[path[-1]])
            return self.record(results)
        center = ('UB_unblinded', 'pairs', 'R1_axR2_0', 'center', 'mean_p')
        self.assertEqual(mutated(center, lambda v: v + 5e-13)['result'], 'PASS')
        bad = mutated(center, lambda v: v + 2e-12)
        self.assertEqual(bad['result'], 'FAIL')
        self.assertEqual(bad['mismatched_keys'], ['unblinded.pairs.R1_axR2_0.center.mean_p'])
        self.assertGreater(bad['max_abs_delta'], 1e-12)
        selected = ('UB_unblinded', 'selection', 'R1_0xR2_1', 'selected')
        flipped = mutated(selected, lambda v: '100.0' if v == 'inf' else 'inf')
        self.assertEqual(flipped['result'], 'FAIL')
        count = ('S3_public', 'pairs', 'R1_bxR2_0', 'invalid_replicates', 'game')
        self.assertEqual(mutated(count, lambda v: v + 1)['result'], 'FAIL')
        width = ('S3_public', 'pairs', 'R1_bxR2_0', 'H_point', 'pitcher')
        self.assertEqual(mutated(width, lambda v: v * (1 + 1e-6))['result'], 'FAIL')
        with self.assertRaises(IntegrityError):
            sa.require_gint_pass(bad, 'G-INT-EVAL')

    def test_the_record_takes_the_12_row_s6_output_only(self):
        public = copy.deepcopy(self.public6)
        public['pairs']['R1_0xR2_0xB2'] = public['pairs'][rb.BASELINE]
        with self.assertRaisesRegex(IntegrityError, '12-row S6 mode output'):
            sa.gint_eval_record(public, self.sealed6, self.results, reference=self.reference)


if __name__ == '__main__':
    unittest.main()
