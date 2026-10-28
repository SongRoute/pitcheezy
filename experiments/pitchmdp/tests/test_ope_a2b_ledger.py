"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 selection and ledger stage pieces: the S6 list recomputed by the
registered ML-POLICY rule must equal the S6 ledger's games, the new list excludes S6 and straddling games and
follows the month quotas over DEV minus S6 minus straddlers, the two salts, the S5 games inside S6, the G-REG pins;
date guards; the reward-free submission (no PA end bound), the M-10 paired identity run with pass/fail and counts
only, the identity checks, the driver-equivalence comparator and the no-DR / no-center output check. The runtime
is the existing synthetic candidate runtime (fake G0 + WE, test_policy_validation); no real data is read.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))

import run_policy_validation as rpv  # noqa: E402
from pitchmdp import ledger_a2b as lg  # noqa: E402
from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp import policy_requests as preq  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402


def selection_frame(seed=0, games_per_month=(14, 12, 10), straddle=3):
    """Synthetic game_pk / game_date / split rows: DEV games over July-September 2025, TRAIN rows, and a few games
    with rows in both June (another split) and July (straddling)."""
    rng = np.random.default_rng(seed)
    rows, g = [], 800000
    for month, n in zip(('2025-07', '2025-08', '2025-09'), games_per_month):
        for _ in range(n):
            day = f'{month}-{int(rng.integers(1, 28)):02d}'
            rows += [{'game_pk': g, 'game_date': day, 'split': 'dev'}] * int(rng.integers(2, 5))
            g += 1
    for _ in range(straddle):
        rows += [{'game_pk': g, 'game_date': '2025-06-30', 'split': 'june'}, {'game_pk': g, 'game_date': '2025-07-01',
                                                                             'split': 'dev'}]
        g += 1
    rows += [{'game_pk': 700000 + i, 'game_date': '2024-06-01', 'split': 'train'} for i in range(5)]
    return pd.DataFrame(rows)


class SelectionTests(unittest.TestCase):
    def test_the_selection_procedure_and_its_salts(self):
        frame = selection_frame()
        n = 6
        s6 = rpv.select_games(frame, 'dev', n, rb.S6_SELECTION_SALT_RECOMPUTED)
        record = lg.selection(frame, rpv.select_games, rpv.straddling_games, s6_ledger_games=s6, n_games=n)
        self.assertEqual(record['s6_games'], sorted(s6))
        self.assertEqual(len(record['new_games']), n)
        self.assertFalse(set(record['new_games']) & set(s6))
        straddle = set(rpv.straddling_games(frame))
        self.assertEqual(len(straddle), 3)
        self.assertFalse(set(record['new_games']) & straddle)
        expected_new = rpv.select_games(frame.loc[~frame.game_pk.isin(s6)], 'dev', n, 759396736)
        self.assertEqual(record['new_games'], sorted(expected_new))
        self.assertNotEqual(sorted(expected_new),
                            sorted(rpv.select_games(frame.loc[~frame.game_pk.isin(s6)], 'dev', n, 2021242497)))
        table = record['month_quota_table']
        self.assertEqual(sum(v['quota'] for v in table.values()), n)
        self.assertEqual(sum(v['eligible_games'] for v in table.values()), 36 - n)  # DEV minus S6 minus straddlers
        self.assertTrue(all(v['quota'] == v['chosen'] for v in table.values()))
        self.assertEqual(record['selection_s6_game_list_sha256'], lg.game_list_sha256(s6))
        self.assertEqual(record['reads'], ['game_pk', 'game_date', 'split'])
        s5 = rpv.select_games(frame, 'dev', 2, rb.S6_SELECTION_SALT_RECOMPUTED)  # top-hash games of their months
        self.assertTrue(set(s5) <= set(s6))

    def test_the_s6_ledger_games_and_the_pins_must_match(self):
        frame = selection_frame()
        s6 = rpv.select_games(frame, 'dev', 6, rb.S6_SELECTION_SALT_RECOMPUTED)
        with self.assertRaisesRegex(IntegrityError, 'S6-dr-a5 ledger games'):
            lg.selection(frame, rpv.select_games, rpv.straddling_games, s6_ledger_games=s6[:-1], n_games=6)
        record = lg.selection(frame, rpv.select_games, rpv.straddling_games, s6_ledger_games=s6, n_games=6)
        pins = {k: record[k] for k in ('selection_s6_game_list_sha256', 'a2b_new_game_list_sha256', 'month_quota_table')}
        self.assertEqual(lg.selection(frame, rpv.select_games, rpv.straddling_games, s6_ledger_games=s6, n_games=6,
                                      expected=pins)['new_games'], record['new_games'])
        for key, value in (('a2b_new_game_list_sha256', 'f' * 64), ('month_quota_table', {'2025-07': 6})):
            with self.subTest(key), self.assertRaisesRegex(IntegrityError, 'G-REG pin'):
                lg.selection(frame, rpv.select_games, rpv.straddling_games, s6_ledger_games=s6, n_games=6,
                             expected={**pins, key: value})
        self.assertEqual(lg.ledger_games([{'pa_id': '800003:4'}, {'pa_id': '800001:1'}, {'pa_id': '800003:5'}]),
                         [800001, 800003])

    def test_date_guards(self):
        frame = pd.DataFrame({'game_date': ['2025-07-02', '2025-08-01', '2026-04-01', '2025-06-20'],
                              'split': ['dev', 'dev', 'dev', 'june']})
        self.assertEqual(lg.guard_rehearsal_dates(frame, [('a', np.array([0, 1]))])['rows'], 2)
        with self.assertRaisesRegex(IntegrityError, '2026'):
            lg.guard_rehearsal_dates(frame, [('a', np.array([0, 2]))])
        with self.assertRaisesRegex(IntegrityError, 'non-DEV'):
            lg.guard_rehearsal_dates(frame, [('a', np.array([0, 3]))])


class LedgerStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_policy_validation as tpv
        cls.tpv = tpv

    def setUp(self):
        from pitchmdp import policy_artifacts as pa
        from pitchmdp import policy_runtime as prt
        from pitchmdp.rollout_policy import RowBudget
        fx = self.tpv.SemiSyntheticTests('test_v2_matches_world_truth_and_is_not_vacuous')
        fx.setUp()
        self.addCleanup(shutil.rmtree, fx.root, True)
        out, report = fx.materialize()
        self.pins = fx.pins(out, report)
        self.fx = fx
        self.blocks = self.tpv.blocks_of(fx.frame, 'dev')
        contexts, _ = rpv.pa_contexts(fx.store, self.blocks)
        self.components = fx.syn.bind(pa.load_train_bc(*self.pins['bc']), contexts, loader=self.tpv.typed_loader)
        self.out = fx.root / 'A2b-L'
        self.out.mkdir()

        def runtime(name):
            return prt.build_runtime(*self.pins['bc'], *self.pins['support'], self.out / name, components=self.components,
                                     budget=RowBudget(10 ** 8, seed_count=5), tau=.003, samples=4, pitch_cap=4, seed=701,
                                     hand_registry=self.pins['hands'], positivity_record=True)
        self.make_runtime = runtime
        self.pair = lambda name: prt.build_reference_pair_runtime(*self.pins['bc'], *self.pins['support'], self.out / name,
                                                                  hand_registry=self.pins['hands'], positivity_record=True)

    def test_reward_free_submission_m10_and_identity_checks(self):
        from pitchmdp.behavior_model import VectorBC
        from pitchmdp.policy_artifacts import load_train_bc
        runtime = self.make_runtime('ledger-dev-0.jsonl')
        facts, sealed = lg.submit(rpv, runtime, self.fx.store, self.blocks, rpv.Deadline(), no_pitch=self.tpv.NO_PITCH,
                                  expected_pas=len(self.blocks))
        self.assertEqual(sealed['pas'], len(self.blocks))
        self.assertTrue(all('reward' not in info for info in facts.values()))
        decisions = list(runtime.ledger.iter_decisions())
        games = preq.game_table(self.fx.store.frame, {int(self.fx.store.frame.game_pk.iloc[p[0]]) for _, p in self.blocks})
        outcome = rpv.outcome_function(self.fx.store, self.components, games, 'structural-end-v1')
        kwargs = {'draws': 50, 'seed': 1, 'invalid_share_max': 0.05, 'minimum': {'games': 1, 'pa_starts': 1},
                  'censoring': 'l1r'}
        m10 = lg.m10_check(rpv, self.pair('ledger-dev-paired-identity.jsonl'), self.fx.store, self.blocks, facts,
                           decisions, no_pitch=self.tpv.NO_PITCH, kwargs=kwargs, outcome=outcome)
        self.assertTrue(m10['pass'] and m10['complete_sets_equal'] and m10['max_abs_delta_is_zero'])
        self.assertGreater(m10['complete_pas'], 0)
        self.assertEqual(set(m10), {'check', 'pass', 'complete_pas', 'complete_sets_equal', 'max_abs_delta_is_zero',
                                    'runtime_sha256'})  # pass/fail and counts only (no delta value)
        header = lg.ledger_header(self.out / 'ledger-dev-0.jsonl')
        candidate_pin = header['pins']['candidate']['sha256']
        artifact = load_train_bc(*self.pins['bc'])
        vector = VectorBC.from_artifact(artifact, vocabulary=tuple(artifact.bc.actions))  # fixture: 3 codes
        pre = {'components_sha256': self.components.sha256}
        record = lg.identity_checks(header, runtime, self.components, pre, decisions, vector,
                                    candidate_sha256=candidate_pin)
        self.assertTrue(all(record['checks'].values()))
        self.assertLessEqual(record['vectorized_b0_max_abs_diff'], 1e-12)
        with self.assertRaisesRegex(IntegrityError, 'header_candidate_identity'):
            lg.identity_checks(header, runtime, self.components, pre, decisions, vector, candidate_sha256='f' * 64)
        with self.assertRaisesRegex(IntegrityError, 'components_equal_bind_identity'):
            lg.identity_checks(header, runtime, self.components, {'components_sha256': 'e' * 64}, decisions, vector,
                               candidate_sha256=candidate_pin)
        (self.out / 'facts.reward_free.json').write_text(json.dumps(lg.reward_free_facts(facts)))
        (self.out / 'ledger_record.json').write_text(json.dumps({'m10': m10, 'identity': record, 'e0': lg.e0_count(facts)}))
        self.assertTrue(lg.check_no_reward_outputs(self.out))
        self.assertGreater(lg.e0_count(facts), 0)

    def test_driver_equivalence_comparator(self):
        first = self.make_runtime('ledger-a.jsonl')
        lg.submit(rpv, first, self.fx.store, self.blocks, rpv.Deadline(), no_pitch=self.tpv.NO_PITCH,
                  expected_pas=len(self.blocks))
        second = self.make_runtime('ledger-b.jsonl')
        part = self.blocks[:3]
        lg.submit(rpv, second, self.fx.store, part, rpv.Deadline(), no_pitch=self.tpv.NO_PITCH, expected_pas=len(part))
        reference, rerun = list(first.ledger.iter_decisions()), list(second.ledger.iter_decisions())
        record = lg.compare_allowed_fields(reference, rerun)
        self.assertTrue(record['pass'], record)  # the same requests through a fresh runtime: identical allowed fields
        self.assertEqual(record['compared'], len(rerun))
        changed = [dict(r) for r in rerun]
        changed[0] = {**changed[0], 'result': {**changed[0]['result'], 'candidate': list(np.roll(changed[0]['result']['candidate'], 1))}}
        self.assertFalse(lg.compare_allowed_fields(reference, changed)['pass'])

    def test_forbidden_outputs_are_refused(self):
        root = Path(tempfile.mkdtemp(prefix='a2b-l-')).resolve()
        self.addCleanup(shutil.rmtree, root, True)
        (root / 'ledger.json').write_text(json.dumps({'pas': 3}))
        self.assertTrue(lg.check_no_reward_outputs(root))
        (root / 'dr.json').write_text('{}')
        with self.assertRaisesRegex(IntegrityError, 'forbidden file'):
            lg.check_no_reward_outputs(root)
        (root / 'dr.json').unlink()
        (root / 'facts.json').write_text(json.dumps({'1:1': {'game': 1, 'reward': 0.5}}))
        with self.assertRaisesRegex(IntegrityError, 'reward / center key'):
            lg.check_no_reward_outputs(root)
        with self.assertRaisesRegex(IntegrityError, 'a PA end reached'):
            lg.reward_free_facts({'1:1': {'game': 1, 'reward': 0.5}})


if __name__ == '__main__':
    unittest.main()


class TrainGuardTests(unittest.TestCase):
    def test_the_reused_train_guard_refuses_2026_and_non_train_rows(self):
        from pitchmdp import ope_surrogate as osg
        osg.guard_train(pd.Series(['2023-05-15', '2025-04-30']))
        for dates in (['2026-04-01'], ['2025-05-01'], ['2023-05-14']):
            with self.subTest(dates), self.assertRaises(IntegrityError):
                osg.guard_train(pd.Series(dates))
