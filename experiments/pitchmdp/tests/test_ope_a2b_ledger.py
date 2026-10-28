"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 selection and ledger stage pieces: the S6 list recomputed by the
registered ML-POLICY rule must equal the S6 ledger's games, the new list excludes S6 and straddling games and
follows the month quotas over DEV minus S6 minus straddlers, the two salts, the S5 games inside S6, the G-REG pins;
date guards; the reward-free submission (no PA end bound), the M-10 paired identity run with pass/fail and counts
only, the identity checks, the driver-equivalence comparator and the no-DR / no-center output check. The runtime
is the existing synthetic candidate runtime (fake G0 + WE, test_policy_validation); no real data is read.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))

import run_policy_validation as rpv  # noqa: E402
from ope_a2b_fixtures import ml_policy_world, read_s6  # noqa: E402
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
        replayed = {r['pa_id'] for r in rerun}
        record = lg.compare_allowed_fields([r for r in reference if r['pa_id'] in replayed], rerun)
        self.assertTrue(record['pass'], record)  # the same requests through a fresh runtime: identical allowed fields
        self.assertEqual(record['compared_decisions'], len(rerun))
        self.assertEqual(record['max_abs_delta'], 0.0)
        partial = lg.compare_allowed_fields(reference, rerun)  # every reference decision must be re-run
        self.assertFalse(partial['pass'])
        self.assertEqual(partial['missing_decisions'], len(reference) - len(rerun))
        changed = [dict(r) for r in rerun]
        changed[0] = {**changed[0], 'result': {**changed[0]['result'], 'candidate': list(np.roll(changed[0]['result']['candidate'], 1))}}
        self.assertFalse(lg.compare_allowed_fields([r for r in reference if r['pa_id'] in replayed], changed)['pass'])
        self.assertEqual(lg.ALLOWED_FIELDS, __import__('pitchmdp.behavior_weights', fromlist=['x']).ALLOWED_DECISION)
        self.assertEqual(lg.ALLOWED_RESULT, __import__('pitchmdp.behavior_weights', fromlist=['x']).ALLOWED_RESULT)

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

    def test_appledouble_sidecars_are_skipped(self):
        """G-CODE MAJOR-2: macOS writes ``._*`` AppleDouble sidecars next to every file on an exFAT output root (T7);
        the last guard of A2b-L must skip them (they are binary, not JSON), while an ordinary ``._`` file and the
        forbidden names stay checked."""
        from pitchmdp.matrix_policy_artifacts import APPLEDOUBLE_MAGIC
        root = Path(tempfile.mkdtemp(prefix='a2b-l-ad-')).resolve()
        self.addCleanup(shutil.rmtree, root, True)
        sidecar = APPLEDOUBLE_MAGIC + bytes.fromhex('00020000') + b'Mac OS X        ' + bytes([0x80, 0xff, 0x00]) * 40
        for name in ('selection.json', 'facts.reward_free.json', 'ledger_record.json'):
            (root / name).write_text(json.dumps({'pas': 3, 'rewards_written': False}))
            (root / f'._{name}').write_bytes(sidecar)
        (root / 'sub').mkdir()
        (root / 'sub' / '._ledger-dev-0.jsonl').write_bytes(sidecar)
        with self.assertRaises(UnicodeDecodeError):  # what the unguarded JSON read did
            (root / '._selection.json').read_text()
        self.assertTrue(lg.check_no_reward_outputs(root))
        (root / '._notes.json').write_text(json.dumps({'center': 1e-4}))  # no AppleDouble magic: an artifact
        with self.assertRaisesRegex(IntegrityError, 'reward / center key'):
            lg.check_no_reward_outputs(root)


class DriverEquivalenceTests(unittest.TestCase):
    """G-CODE MAJOR-4 (design 9.2, PM decision) on a SYNTHETIC ML-POLICY registration built through the registered
    dispatch chain (its S6 = a runtime-v3 dr-evaluate stage of 2 of 4 synthetic DEV games): the path rule, PASS and
    FAIL of path A ``s6_two_game_replay`` and path B ``same_request_twice``, the absolute 1e-12 tolerance; the S6
    rows come through the allowlisted reader only. No real data, ledger or model is read."""
    @classmethod
    def setUpClass(cls):
        cls.world = ml_policy_world()
        cls.addClassCleanup(shutil.rmtree, cls.world.root, True)
        cls.header, cls.decisions, cls.pins = read_s6(cls.world)
        w = cls.world
        cls.pre = w.rpv.prerequisites('dr-evaluate', w.policy)
        cls.s6_games = lg.ledger_games(cls.decisions)
        cls.selection = lg.selection(w.inputs['frame'], w.rpv.select_games, w.rpv.straddling_games,
                                     s6_ledger_games=cls.s6_games, n_games=2)
        cls.replay = lg.replay_games(cls.decisions)
        cls.rows = [r for r in cls.decisions if lg.game_of(r) in cls.replay]

    def run_check(self, header=None, rows=None):
        w = self.world
        out = Path(tempfile.mkdtemp(prefix='a2b-eq-', dir=w.root)).resolve()
        return lg.driver_equivalence(w.rpv, w.policy, w.inputs, pre=self.pre, s6_header=header or self.header,
                                     s6_rows=self.rows if rows is None else rows, s6_games=self.s6_games,
                                     new_games=self.selection['new_games'], out=out, deadline=rpv.Deadline(),
                                     no_pitch=w.no_pitch)

    def without_style(self, header):
        pins = dict(header['pins'])
        pins['provenance'] = {k: v for k, v in pins['provenance'].items() if k != 'style_report'}
        return {**header, 'pins': pins}

    def test_path_a_replays_the_first_and_last_s6_games_on_the_reproduced_runtime(self):
        self.assertEqual(self.selection['new_games'], sorted(set(range(200, 204)) - set(self.s6_games)))
        self.assertIsNotNone(lg.s6_style_report(self.header))
        record = self.run_check()
        self.assertEqual((record['path'], record['pass'], record['style_source'], record['reason']),
                         ('s6_two_game_replay', True, 'header', None))
        self.assertTrue(record['runtime_sha256_reproduced'])
        self.assertEqual(record['runtime_sha256'], self.pins['runtime_sha256'])
        self.assertEqual(record['games'], [lg.game_of(self.decisions[0]), lg.game_of(self.decisions[-1])])
        self.assertEqual(record['compared_decisions'], len(self.rows))
        self.assertGreater(record['compared_fields'], 10 * len(self.rows))
        self.assertEqual(record['max_abs_delta'], 0.0)
        self.assertFalse(record['rewards_read'])
        text = json.dumps(record)
        for word in ('reward', 'delta_bounds', 'center', '"logging"'):  # pass / fail, counts and key names only
            self.assertNotIn(word, text.replace('rewards_read', ''))
        self.assertTrue(all(set(r) <= {*lg.ALLOWED_FIELDS, 'result'} for r in self.rows))  # the allowlisted reader
        recomputed = self.run_check(self.without_style(self.header))  # the style report from the S6 game list
        self.assertEqual((recomputed['path'], recomputed['pass'], recomputed['style_source']),
                         ('s6_two_game_replay', True, 'recomputed_from_the_s6_game_list'))
        self.assertTrue(lg.require_equivalence(record))

    def test_path_a_compares_floats_absolutely_and_the_rest_exactly(self):
        def tampered(delta=None, index=None, drop=False):
            rows = copy.deepcopy(self.rows)
            target = next(r for r in rows if isinstance(r.get('result'), dict))
            if delta is not None:
                target['result']['logging'] = [v + delta if v > 0 else v for v in target['result']['logging']]
            if index is not None:
                target['result']['logged_index'] = target['result']['logged_index'] + index
            return [r for r in rows if r is not target] if drop else rows
        self.assertTrue(self.run_check(rows=tampered(5e-13))['pass'])  # within the absolute 1e-12
        far = self.run_check(rows=tampered(2e-12))
        self.assertFalse(far['pass'])
        self.assertGreater(far['mismatched_fields'], 0)
        self.assertGreater(far['max_abs_delta'], 1e-12)
        self.assertTrue(far['mismatched_keys_shown'][0].endswith('result.logging.[0]') or far['mismatched_keys_shown'])
        with self.assertRaisesRegex(IntegrityError, 'driver equivalence failed on path s6_two_game_replay'):
            lg.require_equivalence(far)
        self.assertFalse(self.run_check(rows=tampered(index=1))['pass'])  # an integer field: equality
        dropped = self.run_check(rows=tampered(drop=True))
        self.assertEqual((dropped['pass'], dropped['extra_decisions']), (False, 1))  # the re-run has one more

    def test_path_b_when_the_s6_runtime_is_not_reproduced(self):
        w = self.world
        other = {**self.without_style(self.header), 'runtime_sha256': 'e' * 64}
        record = self.run_check(other)
        self.assertEqual((record['path'], record['pass'], record['reason'], record['style_source']),
                         ('same_request_twice', True, 'runtime_sha_not_reproduced', 'recomputed_from_the_s6_game_list'))
        self.assertFalse(record['runtime_sha256_reproduced'])
        first_new = int(w.frame.game_pk.iloc[w.rpv.game_blocks(w.frame, self.selection['new_games'], 'dev')[0][1][0]])
        self.assertEqual(record['games'], [first_new])
        self.assertGreater(record['compared_decisions'], 0)
        out = Path(tempfile.mkdtemp(prefix='a2b-main-', dir=w.root))  # path B ran the new DEV ledger's identity
        main, _, _ = lg.candidate(w.rpv, w.policy, w.inputs, w.rpv.game_blocks(w.frame, self.selection['new_games'], 'dev'),
                                  out, pre=self.pre)
        self.assertEqual(record['runtime_sha256'], main.sha256)
        styled = self.run_check({**self.header, 'runtime_sha256': 'e' * 64})  # header style, sha not reproduced
        self.assertEqual((styled['path'], styled['style_source']), ('same_request_twice', 'header'))

    def test_path_b_detects_two_runtimes_that_disagree(self):
        real = lg.candidate

        def drifting(*args, **kwargs):
            runtime, components, provenance = real(*args, **kwargs)
            if kwargs.get('ledger_name') == lg.EQUIVALENCE_LEDGERS['twice'][1]:
                original = runtime.ledger.decisions

                def decisions():
                    rows = original()
                    target = next(r for r in rows if isinstance(r.get('result'), dict))
                    target['result']['rho_candidate'] = target['result']['rho_candidate'] + 1e-9
                    return rows
                runtime.ledger.decisions = decisions
            return runtime, components, provenance
        with mock.patch.object(lg, 'candidate', drifting):
            record = self.run_check({**self.header, 'runtime_sha256': 'e' * 64})
        self.assertEqual((record['path'], record['pass'], record['mismatched_fields']), ('same_request_twice', False, 1))
        with self.assertRaisesRegex(IntegrityError, 'same_request_twice'):
            lg.require_equivalence(record)

    def test_the_replay_pair(self):
        self.assertEqual(lg.replay_games([{'pa_id': '9:1'}, {'pa_id': '7:2'}, {'pa_id': '8:1'}]), (9, 8))
        with self.assertRaisesRegex(IntegrityError, 'two distinct games'):
            lg.replay_games([{'pa_id': '9:1'}, {'pa_id': '9:2'}])
        with self.assertRaises(IntegrityError):
            lg.replay_games([])

    def test_the_registered_runtime_pins_equal_the_assembly(self):
        """G-CODE MINOR-3: runtime_setting and dev_style_snapshot are compared with what candidate() uses."""
        w = self.world
        used = lg.runtime_setting_used(w.policy, self.pre)
        path, sha = w.rpv.registered_path(w.policy, 'style_dev')
        pins = {'runtime_setting': dict(used),
                'dev_style_snapshot': {'path': str(path), 'sha256': sha, 'as_of_exclusive': '2025-07-01'}}
        record = lg.require_runtime_pins(pins, w.rpv, w.policy, self.pre)
        self.assertEqual(record['runtime_setting'], used)
        self.assertEqual(used['evaluation_role'], 'evaluation')  # the frozen dr_q source is the evaluation seed
        self.assertEqual(used['planning_seed'], 0)
        for key, value in (('selected_tau', used['selected_tau'] * 2), ('samples', used['samples'] + 1),
                           ('pitch_cap', used['pitch_cap'] + 1), ('row_budget', 9000000), ('planning_seed', 1),
                           ('evaluation_role', None), ('samples', float(used['samples']))):
            bad = copy.deepcopy(pins)
            bad['runtime_setting'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(IntegrityError, 'runtime_setting'):
                lg.require_runtime_pins(bad, w.rpv, w.policy, self.pre)
        for key, value in (('sha256', 'f' * 64), ('path', str(w.root / 'style_dev.json')), ('as_of_exclusive', '2025-06-01')):
            bad = copy.deepcopy(pins)
            bad['dev_style_snapshot'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(IntegrityError, 'dev_style_snapshot'):
                lg.require_runtime_pins(bad, w.rpv, w.policy, self.pre)


if __name__ == '__main__':
    unittest.main()


class TrainGuardTests(unittest.TestCase):
    def test_the_reused_train_guard_refuses_2026_and_non_train_rows(self):
        from pitchmdp import ope_surrogate as osg
        osg.guard_train(pd.Series(['2023-05-15', '2025-04-30']))
        for dates in (['2026-04-01'], ['2025-05-01'], ['2023-05-14']):
            with self.subTest(dates), self.assertRaises(IntegrityError):
                osg.guard_train(pd.Series(dates))
