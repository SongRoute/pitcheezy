"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 gate record A2b-G and decision: the gate is deterministic from the
public file alone and refuses any center key (and the sealed file); (ii) per bootstrap and S-5 decide the phase-2
work list; label precedence (label 4 unreachable under DIAGNOSTIC_ONLY:DECIDED, reachable by its rule otherwise),
the selection order and tie window, failing families, invalid / not-run cells, flags. No real data is read.
"""
import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ope_a2b_fixtures import a2_nets, a2_tab, pooled, public_doc  # noqa: E402
from pitchmdp import ope_decision_a2b as dc  # noqa: E402
from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

GOOD = (0.6e-3, 0.65e-3)  # ratios 0.60 / 0.59 of the baseline (1.0e-3, 1.1e-3)


class GateTests(unittest.TestCase):
    def test_the_gate_is_deterministic_from_the_public_file(self):
        public = public_doc({'R1_bxR2_1xB0': GOOD, 'R1_0xR2_1xB0': GOOD})
        first = dc.gate_record(public, e0_count=10131)
        again = dc.gate_record(copy.deepcopy(public), e0_count=10131)
        self.assertEqual(dc.gate_bytes(first), dc.gate_bytes(again))
        self.assertEqual(dc.gate_sha256(first), dc.gate_sha256(again))
        self.assertEqual(first['gated'], ['R1_0xR2_1xB0', 'R1_bxR2_1xB0'])  # total order
        self.assertEqual(first['work_list']['S2_prime_cells'], ['2*zeta_p', '2*kappa_p', 'descriptive_kappa_prime'])
        self.assertFalse(first['work_list']['tab']['rerun'])
        self.assertFalse(first['empty'])

    def test_both_bootstraps_and_s5_are_required(self):
        public = public_doc({'R1_axR2_0xB0': (0.6e-3, 0.78e-3),  # pitcher ratio 0.709 > 0.70
                             'R1_axR2_1xB0': GOOD, 'R1_bxR2_0xB0': GOOD},
                            {'R1_axR2_1xB0': ('PASS', 'FAIL')})
        record = dc.gate_record(public, e0_count=10131)
        self.assertEqual(record['gated'], ['R1_bxR2_0xB0'])
        self.assertEqual(record['candidates']['R1_axR2_0xB0']['ii']['per_bootstrap'], {'game': True, 'pitcher': False})
        self.assertIn('S-5', record['candidates']['R1_axR2_1xB0']['reason'])
        none = dc.gate_record(public_doc(), e0_count=10131)
        self.assertTrue(none['empty'])
        self.assertEqual(none['work_list']['S2_prime_cells'], [])

    def test_work_lists_and_the_tab_reuse_rule(self):
        only_r1a = dc.gate_record(public_doc({'R1_axR2_1xB0': GOOD}), e0_count=12500)
        self.assertEqual(only_r1a['work_list']['S2_prime_cells'], ['2*zeta_p'])
        self.assertTrue(only_r1a['work_list']['tab']['rerun'])  # +23% outside +-20%
        self.assertEqual(only_r1a['work_list']['tab']['rerun_dataset_pas'], 12500)
        r10 = dc.gate_record(public_doc({'R1_0xR2_1xB0': GOOD}), e0_count=9000)
        self.assertEqual(r10['work_list']['S2_prime_cells'], [])
        self.assertFalse(r10['work_list']['tab']['rerun'])

    def test_b2_rows_are_never_gated_under_the_decided_status(self):
        public = public_doc({row: GOOD for row in rb.B2_ROWS}, b2_pass=True)
        record = dc.gate_record(public, e0_count=10131)
        self.assertEqual(record['gated'], [])
        self.assertTrue(record['b2_precision_pass_diagnostic'])
        self.assertEqual(record['work_list']['b2_phase_2'], [])

    def test_center_keys_and_the_sealed_file_are_refused(self):
        public = public_doc({'R1_bxR2_1xB0': GOOD})
        leak = copy.deepcopy(public)
        leak['pairs']['R1_bxR2_1xB0']['center'] = 1e-4
        with self.assertRaises(IntegrityError):
            dc.gate_record(leak, e0_count=10131)
        sealed_like = {'selection': {'R1_bxR2_1xB0': {'selected': '100.0'}}, 'pairs': {}}
        with self.assertRaises(IntegrityError):
            dc.gate_record(sealed_like, e0_count=10131)
        view = dc.PublicView(public)
        with self.assertRaisesRegex(IntegrityError, 'public key missing'):
            view.get('pairs', 'R1_bxR2_1xB0', 'center')
        with self.assertRaises(IntegrityError):
            view.get('selection')


class DecisionTests(unittest.TestCase):
    def decide(self, public, *, tab=None, nets=None, s2=None, e0=10131, **kw):
        gate = dc.gate_record(public, e0_count=e0)
        return dc.require_label(dc.decide(public, gate, tab=tab or a2_tab(), nets=nets or a2_nets(),
                                          s2_prime=s2 or {}, **kw))

    def full_s2(self, **kw):
        return {c: pooled(c, **kw) for c in rb.S2P_CELLS}

    def test_label_precedence(self):
        public = public_doc({'R1_bxR2_1xB0': GOOD})
        self.assertEqual(self.decide(public, integrity_failures=['A2b-S3'])['label'], 'FAILED_INTEGRITY')
        self.assertEqual(self.decide(public_doc(baseline=(None, None)))['label'], 'NOT_EVALUABLE')
        self.assertEqual(self.decide(public, s2=self.full_s2())['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_bxR2_1xB0')
        bias = self.decide(public, s2=self.full_s2(outcome='INCONCLUSIVE'))
        self.assertEqual(bias['label'], 'BIAS_NOT_CERTIFIED:R1_bxR2_1xB0')
        self.assertEqual(bias['failing_families'], {'R1_bxR2_1xB0': ['R1_SURROGATE']})
        self.assertEqual(self.decide(public_doc())['label'], 'NO_PRECISION_GAIN')

    def test_r1_0_r2_1_needs_no_s2_prime_and_s5_failures_are_real_shift(self):
        adopt = self.decide(public_doc({'R1_0xR2_1xB0': GOOD}))
        self.assertEqual(adopt['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_0xR2_1xB0')
        shifted = self.decide(public_doc({'R1_0xR2_1xB0': GOOD}, {'R1_0xR2_1xB0': ('PASS', 'FAIL')}))
        self.assertEqual(shifted['label'], 'BIAS_NOT_CERTIFIED:R1_0xR2_1xB0')
        self.assertEqual(shifted['failing_families'], {'R1_0xR2_1xB0': ['REAL_SHIFT']})
        tab = self.decide(public_doc({'R1_0xR2_1xB0': GOOD}), tab=a2_tab(override={('T-A2', 'R1_0xR2_1'): 'FAIL'}))
        self.assertEqual(tab['failing_families'], {'R1_0xR2_1xB0': ['TAB']})
        exact = self.decide(public_doc({'R1_0xR2_1xB0': GOOD}), tab=a2_tab(override={('T-A1', 'R1_0xR2_1'): 'EXACT_PASS'}))
        self.assertEqual(exact['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_0xR2_1xB0')  # the exact-value rule passes
        net = self.decide(public_doc({'R1_0xR2_1xB0': GOOD}), nets=a2_nets('FAIL'))
        self.assertEqual(net['failing_families'], {'R1_0xR2_1xB0': ['COARSE_NET']})

    def test_selection_order_and_tie_window(self):
        widths = {'R1_bxR2_1xB0': (0.50e-3, 0.55e-3), 'R1_axR2_0xB0': (0.51e-3, 0.56e-3), 'R1_0xR2_1xB0': (0.60e-3, 0.61e-3)}
        record = self.decide(public_doc(widths), s2=self.full_s2())
        # H* = 0.55e-3 (R1_b x R2_1) and 0.56e-3 (R1_a x R2_0) lie inside 0.02 x 1.1e-3 = 0.022e-3: first in order
        self.assertEqual(record['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_axR2_0xB0')
        widths['R1_axR2_0xB0'] = (0.51e-3, 0.58e-3)  # outside the window now
        self.assertEqual(self.decide(public_doc(widths), s2=self.full_s2())['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_bxR2_1xB0')
        self.assertEqual(dc.select({'R1_bxR2_1xB0': 1.0}, ['R1_bxR2_1xB0'], 10.0), 'R1_bxR2_1xB0')
        self.assertIsNone(dc.select({}, [], 1.0))

    def test_invalid_or_not_run_cells_never_adopt(self):
        public = public_doc({'R1_bxR2_0xB0': GOOD})
        missing = self.decide(public, s2={'2*zeta_p': pooled('2*zeta_p')})  # 2 kappa_p not pooled
        self.assertTrue(missing['label'].startswith('BIAS_NOT_CERTIFIED'))
        self.assertIn('S2:2*kappa_p', missing['tuples']['R1_bxR2_0xB0']['i']['invalid_cells'])
        invalid = self.decide(public, s2={c: pooled(c, invalid_tuple='R1_bxR2_0xB0') for c in rb.S2P_CELLS})
        self.assertFalse(invalid['tuples']['R1_bxR2_0xB0']['mandatory_valid'])
        no_inv = self.decide(public, s2={**self.full_s2(), '2*kappa_p': pooled('2*kappa_p', invariant=None)})
        self.assertTrue(no_inv['label'].startswith('BIAS_NOT_CERTIFIED'))
        with self.assertRaisesRegex(IntegrityError, 'without a pooled record'):
            dc.check_inputs(a2_tab(), a2_nets(), {}, dc.gate_record(public, e0_count=10131))

    def test_label_4_is_unreachable_under_the_decided_status_and_implemented_otherwise(self):
        b2 = public_doc({row: GOOD for row in rb.B2_ROWS}, b2_pass=True)
        decided = self.decide(b2)
        self.assertEqual(decided['label'], 'NO_PRECISION_GAIN')  # B2 rows are descriptive (D192 2 3)
        self.assertIn('B2_PRECISION_PASS_DIAGNOSTIC', decided['flags'])
        b2['B2'] = {'validity': {'pass': True}, 'E5': {'pass': True}}  # options 1/2 gate the B2 tuples on these too
        gate = dc.gate_record(b2, e0_count=10131, r3_status='ADOPTION_ELIGIBLE')
        self.assertEqual(gate['work_list']['b2_phase_2'][:2], ['R1_0xR2_0xB2', 'R1_axR2_0xB2'])
        failing = {t: {'TAB-R3:C1': 'EXACT_FAIL'} for t in gate['work_list']['b2_phase_2']}
        hypothetical = dc.decide(b2, gate, tab=a2_tab(), nets=a2_nets(), s2_prime=self.full_s2(),
                                 r3_status='ADOPTION_ELIGIBLE', denominator_cells=failing)
        self.assertEqual(hypothetical['label'], 'DENOMINATOR_NOT_CERTIFIED:' + ','.join(gate['work_list']['b2_phase_2']))
        with self.assertRaisesRegex(IntegrityError, 'unreachable'):
            dc.require_label({**hypothetical, 'r3_status': rb.R3_STATUS_DECIDED})
        surrogate = dc.decide(b2, gate, tab=a2_tab(), nets=a2_nets(), s2_prime=self.full_s2(outcome='INCONCLUSIVE'),
                              r3_status='ADOPTION_ELIGIBLE', denominator_cells=failing)
        self.assertTrue(surrogate['label'].startswith('DENOMINATOR_NOT_CERTIFIED:R1_0xR2_0xB2'))  # R1_0 needs no S-2'
        self.assertEqual(surrogate['tuples']['R1_axR2_0xB2']['i']['failing_families'], ['DENOMINATOR', 'R1_SURROGATE'])
        passing = {t: {'TAB-R3:C1': 'EXACT_PASS'} for t in gate['work_list']['b2_phase_2']}
        adopt = dc.decide(b2, gate, tab=a2_tab(), nets=a2_nets(), s2_prime=self.full_s2(), r3_status='ADOPTION_ELIGIBLE',
                          denominator_cells=passing)
        self.assertEqual(adopt['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_0xR2_0xB2')  # B0 < B2, then R2_0, then R1_0
        b2['B2']['E5']['pass'] = False
        stopped = dc.gate_record(b2, e0_count=10131, r3_status='ADOPTION_ELIGIBLE')
        self.assertEqual(stopped['work_list']['b2_phase_2'], [])
        self.assertEqual(stopped['candidates']['R1_0xR2_0xB2']['reason'], 'E5 or B2 validity')
        self.assertEqual(rb.LABELS.index('DENOMINATOR_NOT_CERTIFIED:<B2 tuples>'), 3)

    def test_flags(self):
        record = self.decide(public_doc(flag='SE_CALIBRATION_WARNING'), r3_stage_failed_integrity=True)
        self.assertEqual(record['flags'], ['SE_CALIBRATION_WARNING', 'r3_stage_failed_integrity'])
        self.assertEqual(record['rehearsal_exposure'], 'OPE_FIRST_USE_DEV')
        self.assertEqual(record['r3_status'], 'DIAGNOSTIC_ONLY:DECIDED')


if __name__ == '__main__':
    unittest.main()
