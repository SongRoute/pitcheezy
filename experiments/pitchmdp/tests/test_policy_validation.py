"""SYNTHETIC-ONLY tests for the COOP-018/D93 request generator, runtime v2 refusals, DR estimator
v2, tau selection, V2/V3 world and the <=2025 validation runner. Tiny synthetic games; fake pinned
G0/WE files from test_policy_identity; the D89 enumerated toy PA as the exact DR oracle. No real
data, model weights or 2026 access.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts'))
from test_policy_identity import PROV, TYPES, WE, Network, SyntheticPolicy, load_member
from pitchmdp.matrix_sharing import SharingPredictor
from pitchmdp.archetypes import HISTORY_COLUMNS, add_batter_style_history
from pitchmdp.matrix_data import ordered_key_hash
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_policy import safe_rows
from pitchmdp.rollout_policy import BCRecord, CategoricalBC, PAState, PastPitch, RowBudget, kl_policy
from pitchmdp.sequence_data import PHYSICAL_COLUMNS, PhysicalNormalizer
from pitchmdp import policy_artifacts as pa
from pitchmdp import policy_estimator as est
from pitchmdp import policy_identity as pi
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as pr
from pitchmdp import policy_semisynthetic as pss
from pitchmdp import policy_tau as ptau
import run_policy_validation as rpv

spec = importlib.util.spec_from_file_location('policy_contract', REPO / 'scripts/check_2026_policy_contract.py')
TOY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TOY)

SEQUENCES = {  # (type, description); the last row carries the event
    'short': [('FF', 'ball'), ('SL', 'called_strike'), ('FF', 'foul'), ('CH', 'hit_into_play')],
    'long': [('FF', 'ball'), ('SL', 'called_strike'), ('FF', 'foul'), ('FF', 'foul'), ('SL', 'foul'), ('FF', 'foul'),
             ('SL', 'hit_into_play')],
    'automatic': [('FF', 'ball'), (None, 'automatic_ball'), ('SL', 'called_strike'), ('FF', 'hit_into_play')],
    'missing': [('FF', 'ball'), (None, 'ball'), ('SL', 'called_strike'), ('FF', 'hit_into_play')],
}
STEP = {'ball': (1, 0), 'called_strike': (0, 1), 'automatic_ball': (1, 0), 'foul': (0, 1)}
NO_PITCH = frozenset({'automatic_ball', 'automatic_strike'})
DEV_SCRIPT = [('short', {}), ('long', {}), ('automatic', {}), ('short', {'pitcher': 77}), ('short', {'drop_first': True}),
              ('short', {'truncated': True}), ('missing', {}), ('long', {'event': 'field_out'})]
MIN = {'games': 1, 'pa_starts': 1}
FULL = np.ones(3, bool)


def pa_rows(game, ab, date, split, kind, *, pitcher=9, event='single', drop_first=False, truncated=False):
    rows, balls, strikes = [], 0, 0
    for index, (pitch_type, description) in enumerate(SEQUENCES[kind]):
        last = index == len(SEQUENCES[kind]) - 1
        rows.append(dict(game_pk=game, at_bat_number=ab, pitch_number=index + 1, game_date=date, pitcher=pitcher,
                         batter=2 + ab % 3, stand='L', p_throws='R', balls=balls, strikes=strikes,
                         inning=5 + ab // 6, inning_topbot='Top' if (ab // 3) % 2 == 0 else 'Bot', outs_when_up=ab % 3,
                         bases=0, home_score=0, away_score=0, post_home_score=0, post_away_score=0,
                         pitch_type=pitch_type, description=description,
                         events=('truncated_pa' if truncated else event) if last else None,
                         launch_angle=12. if description == 'hit_into_play' else np.nan, supported_pa=True,
                         split=split, complete_game=True))
        db, ds = STEP.get(description, (0, 0))
        balls, strikes = balls + db, min(2, strikes + ds) if description == 'foul' else strikes + ds
    return rows[1:] if drop_first else rows


def game_frame():
    rng, rows = np.random.default_rng(3), []
    for g in range(8):  # TRAIN
        date = (pd.Timestamp('2024-06-01') + pd.Timedelta(days=g)).strftime('%Y-%m-%d')
        for ab in range(1, 7):
            rows += pa_rows(100 + g, ab, date, 'train', ('short', 'long')[ab % 2], event=('single', 'field_out')[ab % 2])
    for ab in range(1, 4):  # one May game (profile starts)
        rows += pa_rows(150, ab, '2025-05-20', 'temperature', 'short')
    for g in range(2):  # two June games (tau selection)
        for ab in range(1, 4):
            rows += pa_rows(160 + g, ab, f'2025-06-1{g}', 'blend', ('short', 'long')[ab % 2])
    for g in range(2):  # DEV
        date = (pd.Timestamp('2025-07-01') + pd.Timedelta(days=g)).strftime('%Y-%m-%d')
        for ab, (kind, extra) in enumerate(DEV_SCRIPT, start=1):
            rows += pa_rows(200 + g, ab, date, 'dev', kind, **extra)
    frame = pd.DataFrame(rows)
    for c in PHYSICAL_COLUMNS:
        frame[c] = rng.normal(size=len(frame))
    last = ~frame.duplicated(['game_pk', 'at_bat_number'], keep='last')
    frame['is_pa_terminal'] = last & frame.events.notna() & frame.events.ne('truncated_pa')
    for game, part in frame.groupby('game_pk'):  # home wins every game: final result is determined
        frame.loc[part.index[-1], ['post_home_score', 'post_away_score']] = (1, 0)
    return add_batter_style_history(frame)


class TypedNetwork(Network):
    """Fake G0 whose outcome law depends strongly on the pitch type (SL -> outs, FF -> home runs), so
    the V2 world separates the candidate from the reference by far more than the Monte Carlo error."""
    def logits(self, arrays):
        z = super().logits(arrays)
        z[:, 3] += 8 * arrays[0][:, -1, 8 + 1 + TYPES.index('SL')]
        z[:, 7] += 8 * arrays[0][:, -1, 8 + 1 + TYPES.index('FF')]
        return z


def typed_loader(config, record, clusters):
    predictor = SharingPredictor('G0-global', TypedNetwork(record['seed']), clusters)
    predictor.delivery_temperature = record['delivery_temperature']
    return predictor, 'cpu'


def build_store(frame):
    normalizer = PhysicalNormalizer().fit(frame.loc[frame.split.eq('train')])
    return MatrixHistoryStore.from_frame(frame, normalizer, 5, type_vocabulary=TYPES)


def sha(path):
    return pa.hash_file(Path(path))


def blocks_of(frame, split):
    return preq.pa_blocks(frame, np.flatnonzero(frame.split.eq(split).to_numpy()))


# ---------------------------------------------------------------- runtime v2 refusals

class RuntimeRefusalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        records = [BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'FF', 'SL')]
        records += [BCRecord(PAState(0, 0, '8', 'L'), a, 'train') for a in ('CH', 'FF')]
        records += [BCRecord(PAState(0, 0, '7', 'L'), 'FF', 'train')]
        self.art = pa.save_train_bc(CategoricalBC().fit(records), self.root / 'bc.json', {**PROV, 'train_rows': 6})
        _, self.support = pa.save_support_table(self.art, [('9', 'L', np.array([False, True, True])),
                                                           ('8', 'L', np.array([True, True, False]))], self.root / 's.json')
        _, self.hands = pa.save_hand_registry(self.art, {'9': 'R', '8': 'L', '7': 'AMBIGUOUS'}, 6, self.root / 'h.json')
        self.rt = pr.build_runtime(self.root / 'bc.json', self.art.file_sha256, self.root / 's.json', self.support,
                                   self.root / 'l.jsonl', hand_registry=(self.root / 'h.json', self.hands))

    def submit(self, rid, pa_id, index, state, logged, hand='R'):
        return self.rt.submit(pr.DecisionRequest(rid, pa_id, index, state, logged, self.rt.sha256, hand))

    def past(self, action, outcome, balls=0, strikes=0):
        return PastPitch(action, (0.,) * 8, outcome, balls, strikes)

    def test_split_no_action_start_and_count_path_refusals(self):
        self.assertEqual(self.submit('a0', 'a', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        no_pitch = self.submit('a1', 'a', 1, PAState(1, 0, '9', 'L', (self.past('FF', 'ball'),)), pr.NO_PITCH)
        self.assertEqual(no_pitch['status'], pr.NO_LOGGED_ACTION)
        later = PAState(2, 0, '9', 'L', (self.past('FF', 'ball'), self.past(pr.NO_PITCH, 'unknown', 1, 0)))
        self.assertEqual(self.submit('a2', 'a', 2, later, 'SL')['status'], pr.MID_PA)  # sticky, chain still checked
        missing = self.submit('m0', 'm', 0, PAState(0, 0, '9', 'L'), pr.MISSING)
        self.assertEqual(missing['status'], pr.MISSING_LABEL)  # a real pitch without a type: its own cause
        self.assertEqual(self.submit('b0', 'b', 0, PAState(1, 0, '9', 'L'), 'FF')['status'], pr.INCOMPLETE_START)
        self.assertEqual(self.submit('c0', 'c', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        jumped = PAState(0, 1, '9', 'L', (self.past('FF', 'ball'),))  # a ball cannot lead to 0-1
        self.assertEqual(self.submit('c1', 'c', 1, jumped, 'SL')['status'], pr.INCONSISTENT_HISTORY)
        self.assertEqual(self.submit('d0', 'd', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        ended = PAState(0, 0, '9', 'L', (self.past('FF', 'single'),))  # a pitch after the PA ended
        self.assertEqual(self.submit('d1', 'd', 1, ended, 'SL')['status'], pr.INCONSISTENT_HISTORY)
        self.assertEqual(self.submit('u0', 'u', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        unknown = PAState(1, 0, '9', 'L', (self.past('FF', 'unknown'),))  # uncheckable outcome of a real pitch
        refused = self.submit('u1', 'u', 1, unknown, 'SL')
        self.assertEqual((refused['status'], 'uncheckable' in refused['detail']), (pr.INCONSISTENT_HISTORY, True))
        summary = self.rt.summary()  # refusals after decision 0 roll up to MID_PA; the first cause is kept
        self.assertEqual(summary['pa_status'], {pr.MID_PA: 4, pr.MISSING_LABEL: 1, pr.INCOMPLETE_START: 1})
        self.assertEqual(summary['pa_first_refusal'], {pr.NO_LOGGED_ACTION: 1, pr.MISSING_LABEL: 1,
                                                       pr.INCOMPLETE_START: 1, pr.INCONSISTENT_HISTORY: 3})
        with self.assertRaisesRegex(pa.IntegrityError, 'unknown logged action'):  # a real unknown code still halts
            self.submit('e0', 'e', 0, PAState(0, 0, '9', 'L'), 'ZZ')

    def test_hand_registry_refusals_and_start_population(self):
        self.assertEqual(self.submit('h0', 'h', 0, PAState(0, 0, '9', 'L'), 'FF', hand='L')['detail'],
                         f'{pr.PITCHER_HAND}: mismatch')
        self.assertEqual(self.submit('h1', 'h', 1, PAState(1, 0, '9', 'L', (self.past('FF', 'ball'),)), 'FF')['status'],
                         pr.MID_PA)  # sticky
        self.assertEqual(self.submit('i0', 'i', 0, PAState(0, 0, '9', 'L'), 'FF', hand=None)['detail'],
                         f'{pr.PITCHER_HAND}: missing_or_invalid')
        self.assertEqual(self.submit('j0', 'j', 0, PAState(0, 0, '9', 'L'), 'FF', hand='S')['detail'],
                         f'{pr.PITCHER_HAND}: missing_or_invalid')
        self.assertEqual(self.submit('k0', 'k', 0, PAState(0, 0, '7', 'L'), 'FF')['detail'],
                         f'{pr.PITCHER_HAND}: ambiguous_train')
        # A known pitcher change inside a PA with a different registered hand: both decisions evaluated (D-4).
        self.assertEqual(self.submit('n0', 'n', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        changed = self.submit('n1', 'n', 1, PAState(1, 0, '8', 'L', (self.past('FF', 'ball'),)), 'FF', hand='L')
        self.assertEqual(changed['status'], pr.SUPPORTED)
        start = self.rt.start_population
        self.assertEqual(start(PAState(0, 0, '9', 'L'), 'R'), (True, None))
        self.assertEqual(start(PAState(0, 0, '9', 'L'), 'L'), (False, f'{pr.PITCHER_HAND}:{pr.PITCHER_HAND}: mismatch'))
        self.assertEqual(start(PAState(1, 0, '9', 'L'), 'R'), (False, pr.INCOMPLETE_START))
        self.assertEqual(start(PAState(0, 0, '5', 'L'), 'R'), (False, pr.UNKNOWN_PITCHER))
        self.assertEqual(start(PAState(0, 0, '9', 'R'), 'R'), (False, pr.EMPTY_SUPPORT))
        _, bad = pa.save_hand_registry(self.art, {'9': 'R', '8': 'AMBIGUOUS', '7': 'R'}, 6, self.root / 'h2.json')
        with self.assertRaisesRegex(pa.IntegrityError, 'single-hand registry'):  # an ambiguous pitcher in the table
            pr.build_runtime(self.root / 'bc.json', self.art.file_sha256, self.root / 's.json', self.support,
                             self.root / 'x.jsonl', hand_registry=(self.root / 'h2.json', bad))
        _, partial = pa.save_support_table(self.art, [('9', 'L', np.array([False, True, True]))], self.root / 's3.json')
        with self.assertRaisesRegex(pa.IntegrityError, 'must equal the single-hand'):  # pitcher 8 has no rows (D-8 c4)
            pr.build_runtime(self.root / 'bc.json', self.art.file_sha256, self.root / 's3.json', partial,
                             self.root / 'y.jsonl', hand_registry=(self.root / 'h.json', self.hands))

    def test_positivity_refusal_precedes_the_candidate_and_abort_halts(self):
        calls = []

        def candidate(state):
            calls.append(state)
            return self.rt.reference.probabilities(state)
        artifact = pa.load_train_bc(self.root / 'bc.json', self.art.file_sha256)
        table, support_sha = pa.load_support_table(self.root / 's.json', self.support, artifact)
        rt = pr.PolicyRuntime(artifact, table, support_sha, self.root / 'c.jsonl', candidate=candidate,
                              candidate_identity={'name': 'test'})
        row = rt.submit(pr.DecisionRequest('p0', 'p', 0, PAState(0, 0, '9', 'L'), 'CH', rt.sha256, 'R'))
        self.assertEqual((row['status'], calls), (pr.LOGGING_POSITIVITY, []))  # no search for a refused request
        rt.abort('wall guard')
        self.assertEqual(rt.summary()['run_status'], 'HALTED')
        with self.assertRaisesRegex(pa.IntegrityError, 'halted'):
            rt.submit(pr.DecisionRequest('p1', 'q', 0, PAState(0, 0, '9', 'L'), 'FF', rt.sha256, 'R'))

    def test_runtime_v3_records_the_positivity_row_with_rho_zero(self):
        """COOP-021/022: opt-in v3 evaluates the candidate first and records pi, Q and rho = 0 on the
        refused row; status and stickiness stay; the default runtime keeps the v2 pins."""
        calls = []

        def candidate(state):
            calls.append(state)
            mask = self.rt.reference.support(state)
            q = np.where(mask, .5, np.nan)
            return self.rt.reference.probabilities(state), {'q': q, 'mc_se': q * 0, 'q_planning': q,
                                                            'planning_diff_se': q * 0, 'source': 'test'}
        artifact = pa.load_train_bc(self.root / 'bc.json', self.art.file_sha256)
        table, support_sha = pa.load_support_table(self.root / 's.json', self.support, artifact)
        v2 = pr.PolicyRuntime(artifact, table, support_sha, self.root / 'v2.jsonl', candidate=candidate,
                              candidate_identity={'name': 'test'})
        rt = pr.PolicyRuntime(artifact, table, support_sha, self.root / 'v3.jsonl', candidate=candidate,
                              candidate_identity={'name': 'test'}, positivity_record=True)
        self.assertEqual((v2.pins['contract'], 'positivity_record' in v2.pins), (pr.CONTRACT, False))
        self.assertEqual((rt.pins['contract'], rt.pins['positivity_record']), (pr.CONTRACT_V3, True))
        self.assertNotEqual(rt.sha256, v2.sha256)
        row = rt.submit(pr.DecisionRequest('p0', 'p', 0, PAState(0, 0, '9', 'L'), 'CH', rt.sha256, 'R'))
        self.assertEqual((row['status'], len(calls)), (pr.LOGGING_POSITIVITY, 1))
        self.assertEqual(rt.summary()['positivity_candidate_searches'], 1)  # S-M1: the v3 search cost is visible
        self.assertNotIn('positivity_candidate_searches', v2.summary())  # v2 summary keys unchanged
        result = row['result']
        self.assertEqual((result['rho_candidate'], result['rho_reference'], result['logged_index']), (0., 0., 0))
        self.assertEqual((result['logging'][0], result['mask'][0], result['candidate'][0]), (0., False, 0.))
        after = rt.submit(pr.DecisionRequest('p1', 'p', 1, PAState(0, 1, '9', 'L', (self.past('CH', 'strike'),)),
                                             'FF', rt.sha256, 'R'))
        self.assertEqual((after['status'], after['result']), (pr.MID_PA, None))  # still sticky
        value = est.pa_value(rt.ledger.decisions(), {'game': 1, 'in_population': True, 'reward': .4}, 'l1r')
        self.assertEqual((value['resolution'], value['delta']), ('rho_zero', 0.))  # cand = ref here
        self.assertAlmostEqual(value['candidate'], .5, places=12)  # V_0 = v_pi(H_0) = sum pi q

    def test_runtime_v3_search_failure_on_a_positivity_row_fails_closed(self):
        """S-M1: a BudgetExceeded (or any candidate exception) during the v3 search on a positivity row
        is a recorded FAILED_RUNTIME that names the positivity row, halts the run and propagates."""
        from pitchmdp.rollout_policy import BudgetExceeded

        def candidate(state):
            raise BudgetExceeded('row budget exhausted')
        artifact = pa.load_train_bc(self.root / 'bc.json', self.art.file_sha256)
        table, support_sha = pa.load_support_table(self.root / 's.json', self.support, artifact)
        rt = pr.PolicyRuntime(artifact, table, support_sha, self.root / 'v3b.jsonl', candidate=candidate,
                              candidate_identity={'name': 'test'}, positivity_record=True)
        with self.assertRaisesRegex(BudgetExceeded, 'row budget') as caught:
            rt.submit(pr.DecisionRequest('p0', 'p', 0, PAState(0, 0, '9', 'L'), 'CH', rt.sha256, 'R'))
        row = rt.ledger.decisions()[-1]
        self.assertEqual(row['status'], pr.FAILED_RUNTIME)
        self.assertIn(f'{pr.LOGGING_POSITIVITY} row', row['detail'])
        self.assertIn(pr.LOGGING_POSITIVITY, ' '.join(caught.exception.__notes__))
        self.assertEqual(rt.summary()['run_status'], 'HALTED')
        v2 = pr.PolicyRuntime(artifact, table, support_sha, self.root / 'v2b.jsonl', candidate=candidate,
                              candidate_identity={'name': 'test'})  # v2 refuses before the search: no failure
        self.assertEqual(v2.submit(pr.DecisionRequest('p0', 'p', 0, PAState(0, 0, '9', 'L'), 'CH', v2.sha256,
                                                      'R'))['status'], pr.LOGGING_POSITIVITY)
        with self.assertRaisesRegex(BudgetExceeded, 'row budget'):  # a supported row: detail has no positivity note
            v2.submit(pr.DecisionRequest('q0', 'q', 0, PAState(0, 0, '9', 'L'), 'FF', v2.sha256, 'R'))
        self.assertEqual(v2.ledger.decisions()[-1]['detail'], 'BudgetExceeded: row budget exhausted')

    def test_no_bc_or_candidate_query_after_a_sentinel(self):
        """D-3 guard: the BC fit keys use '<UNKNOWN>' where requests use sentinels; harmless only because
        nothing queries the BC or the candidate after a sentinel (sticky refusal)."""
        calls = []

        def candidate(state):
            calls.append(state)
            return self.rt.reference.probabilities(state)
        artifact = pa.load_train_bc(self.root / 'bc.json', self.art.file_sha256)
        table, support_sha = pa.load_support_table(self.root / 's.json', self.support, artifact)
        rt = pr.PolicyRuntime(artifact, table, support_sha, self.root / 'spy.jsonl', candidate=candidate,
                              candidate_identity={'name': 'spy'})
        with mock.patch.object(rt.bc, 'probabilities', wraps=rt.bc.probabilities) as spy:
            rt.submit(pr.DecisionRequest('s0', 's', 0, PAState(0, 0, '9', 'L'), 'FF', rt.sha256, 'R'))
            before = (spy.call_count, len(calls))
            for index, sentinel in enumerate((pr.NO_PITCH, pr.MISSING)):
                history = (self.past('FF', 'ball'),) + tuple(self.past(pr.NO_PITCH, 'unknown', 1, 0) for _ in range(index))
                rt.submit(pr.DecisionRequest(f's{index + 1}', 's', index + 1, PAState(1, 0, '9', 'L', history), sentinel,
                                             rt.sha256, 'R'))
            self.assertEqual((spy.call_count, len(calls)), before)

    def test_next_count(self):
        self.assertEqual([pr.next_count(3, 1, 'ball'), pr.next_count(0, 2, 'strike'), pr.next_count(1, 2, 'foul'),
                          pr.next_count(1, 1, 'foul'), pr.next_count(0, 0, 'out'), pr.next_count(2, 1, 'unknown')],
                         [None, None, (1, 2), (1, 2), None, 'unknown'])
        with self.assertRaises(pa.IntegrityError):
            pr.next_count(0, 0, 'catch')


# ---------------------------------------------------------------- request generator and outcomes

class RequestTests(unittest.TestCase):
    def setUp(self):
        self.frame = game_frame()
        self.store = build_store(self.frame)
        self.games = preq.game_table(self.frame)
        self.we = WE()

    def test_full_history_labels_and_structure(self):
        blocks = dict(blocks_of(self.frame, 'dev'))
        long_requests, problem, index = preq.pa_requests(self.store, blocks['200:2'], 'sha')
        self.assertEqual((problem, index), (None, None))
        self.assertEqual([len(r.state.history) for r in long_requests], list(range(7)))  # beyond the 5-row cap
        self.assertEqual({r.pitcher_hand for r in long_requests}, {'R'})
        automatic, _, _ = preq.pa_requests(self.store, blocks['200:3'], 'sha')
        self.assertEqual([r.logged_action for r in automatic], ['FF', pr.NO_PITCH, 'SL', 'FF'])
        self.assertEqual((automatic[2].state.history[-1].action, automatic[2].state.history[-1].outcome),
                         (pr.NO_PITCH, 'unknown'))
        missing, _, _ = preq.pa_requests(self.store, blocks['200:7'], 'sha')
        self.assertEqual([r.logged_action for r in missing], ['FF', pr.MISSING, 'SL', 'FF'])
        self.assertEqual(missing[2].state.history[-1].outcome, 'ball')  # a real pitch keeps its outcome
        self.assertEqual((preq.logged_label(np.nan, 'ball'), preq.logged_label('', 'foul'),
                          preq.logged_label('FF', 'automatic_strike')), (pr.MISSING, pr.MISSING, pr.NO_PITCH))
        self.assertEqual(preq.pa_requests(self.store, blocks['200:2'][::-1], 'sha'), ([], 'ordering', 0))
        self.assertEqual(preq.pa_requests(self.store, blocks['200:5'], 'sha'), ([], 'missing_row', 0))  # pitch 1 absent
        bad = self.frame.copy(); bad.loc[blocks['200:1'][2], 'balls'] = 4
        prefix, problem, index = preq.pa_requests(build_store(bad), blocks['200:1'], 'sha')
        self.assertEqual((len(prefix), problem, index), (2, 'illegal_count', 2))  # earlier decisions kept (R1/R4c)
        gap = self.frame.drop(index=blocks['200:2'][3]).reset_index(drop=True)
        prefix, problem, index = preq.pa_requests(build_store(gap), dict(blocks_of(gap, 'dev'))['200:2'], 'sha')
        self.assertEqual((len(prefix), problem, index), (3, 'missing_row', 3))
        nameless = self.frame.copy(); nameless.loc[blocks['200:1'][1], 'stand'] = None
        self.assertEqual(preq.pa_requests(build_store(nameless), blocks['200:1'], 'sha')[1:], ('missing_identity', 1))

    def test_requests_ignore_post_decision_fields(self):
        """R4c: changing the current row's physics/outcome and every later row leaves earlier states unchanged."""
        position = dict(blocks_of(self.frame, 'dev'))['200:2']
        base, _, _ = preq.pa_requests(self.store, position, 'sha')
        changed = self.frame.copy()
        changed.loc[position[3:], PHYSICAL_COLUMNS] += 5.
        changed.loc[position[3:], ['description', 'events', 'launch_angle']] = ['hit_into_play', 'home_run', 40.]
        changed.loc[position[4:], 'pitch_type'] = 'CH'
        other, _, _ = preq.pa_requests(build_store(changed), position, 'sha')
        for before, after in zip(base[:4], other[:4]):  # decisions up to and including row k
            self.assertEqual(before.state, after.state)
        self.assertEqual(base[3].logged_action, other[3].logged_action)
        self.assertNotEqual(base[4].state, other[4].state)  # row k's outcome is history for row k+1

    def test_bc_population_mask_ignores_post_decision_fields(self):
        mask = preq.bc_population_mask(self.frame)
        train = self.frame.split.eq('train').to_numpy()
        self.assertEqual(mask.sum(), train.sum())  # no automatic/missing rows in TRAIN here
        changed = self.frame.copy()
        changed['events'] = 'home_run'
        changed['supported_pa'] = False
        changed.loc[changed.description.eq('ball'), 'description'] = 'called_strike'
        np.testing.assert_array_equal(preq.bc_population_mask(changed), mask)
        altered = self.frame.copy()
        rows = np.flatnonzero(train)[:3]
        altered.loc[rows[0], 'description'] = 'automatic_ball'  # the description wins over a present type
        altered.loc[rows[1], 'pitch_type'] = ''
        altered.loc[rows[2], 'balls'] = 4
        self.assertEqual(preq.bc_population_mask(altered).sum(), mask.sum() - 3)

    def test_structural_pa_end_rewards(self):
        blocks = dict(blocks_of(self.frame, 'dev'))
        we = lambda s, d: self.we.predict_defense(s, d)
        normal = preq.pa_outcome(self.frame, blocks['200:1'], we, self.games)
        nxt, first = self.frame.iloc[blocks['200:2'][0]], self.frame.iloc[blocks['200:1'][0]]
        expected = .5 + .05 * nxt.outs_when_up + (.03 if first.inning_topbot == 'Top' else -.03)  # the initial defender
        self.assertEqual((normal['end'], normal['end_kind']), ('next_row_state', 'batter_event'))
        self.assertAlmostEqual(normal['reward'], expected, places=12)
        truncated = preq.pa_outcome(self.frame, blocks['200:6'], we, self.games)  # next PA observed: the PA ended
        self.assertEqual((truncated['end_kind'], truncated['reward'] is not None), ('non_batter_end', True))
        legacy = preq.pa_outcome(self.frame, blocks['200:6'], we, self.games, rule='r5-events-v1')
        self.assertEqual((legacy['reason'], legacy['kind']), ('truncated_pa_r5', preq.NO_TERMINAL))
        gap = preq.pa_outcome(self.frame, blocks['200:4'], we, self.games)  # next PA's first pitch is missing
        self.assertEqual((gap['reason'], gap['kind']), ('end_state_gap', preq.TERMINAL_VALUE_MISSING))
        final = preq.pa_outcome(self.frame, blocks['200:8'], we, self.games)
        top = self.frame.iloc[blocks['200:8'][0]].inning_topbot == 'Top'
        self.assertEqual((final['end'], final['reward'], final['end_kind']), ('final_result', 1. if top else 0., 'game_final'))
        heuristic = self.frame.copy(); heuristic.loc[heuristic.game_pk.eq(200), 'complete_game'] = False
        self.assertEqual(preq.pa_outcome(heuristic, blocks['200:8'], we, preq.game_table(heuristic))['end'],
                         'final_result')  # data.py complete_game is not the OPE rule (D-6 critic 3)
        tied = self.frame.copy(); tied.loc[blocks['200:8'][-1], ['post_home_score', 'post_away_score']] = (1, 1)
        self.assertEqual(preq.pa_outcome(tied, blocks['200:8'], we, preq.game_table(tied))['reason'], 'game_not_final_v1')
        short = self.frame.copy(); short.loc[blocks['200:8'], 'inning'] = 4  # shorter than a regulation game
        self.assertEqual(preq.pa_outcome(short, blocks['200:8'], we, preq.game_table(short))['kind'],
                         preq.TERMINAL_VALUE_MISSING)
        no_event = self.frame.copy(); no_event.loc[blocks['200:1'][-1], 'events'] = None
        self.assertEqual(preq.pa_outcome(no_event, blocks['200:1'], we, self.games)['kind'], preq.NO_TERMINAL)
        illegal = self.frame.copy(); illegal.loc[blocks['200:2'], 'outs_when_up'] = 0  # outs went down in a half
        self.assertEqual(preq.pa_outcome(illegal, blocks['200:1'], we, self.games)['reason'], 'end_state_gap')
        flagged = self.frame.copy(); flagged.loc[blocks['200:1'][-1], 'post_home_score'] = 1
        self.assertEqual(preq.pa_outcome(flagged, blocks['200:1'], we, self.games)['flags'],
                         ['post_pitch_score_disagrees_next_row'])
        post = preq.pa_outcome(flagged, blocks['200:1'], we, self.games, score_source='post_pitch')
        self.assertAlmostEqual(post['reward'], expected, places=12)  # the fake WE ignores the score
        with self.assertRaisesRegex(pa.IntegrityError, 'post-pitch score columns'):
            preq.game_table(self.frame.drop(columns='post_home_score'))
        unscored = self.frame.copy(); unscored.loc[blocks['200:8'][-1], 'post_home_score'] = np.nan
        with self.assertRaisesRegex(pa.IntegrityError, 'post-pitch score missing on the last row of game 200'):
            preq.game_table(unscored, {200})  # R5: FAILED_INTEGRITY, never a silent censoring
        self.assertEqual(set(preq.game_table(unscored, {201})), {201})  # only the selected games are read
        missing_post = flagged.copy(); missing_post.loc[blocks['200:1'][-1], 'post_home_score'] = np.nan
        self.assertEqual(preq.pa_outcome(missing_post, blocks['200:1'], we, self.games, score_source='post_pitch')['reason'],
                         'post_pitch_score_missing')
        split = pd.concat([self.frame.iloc[blocks['200:1']], self.frame.iloc[blocks['201:1']],
                           self.frame.iloc[blocks['200:2']]])
        with self.assertRaisesRegex(pa.IntegrityError, 'contiguous'):
            preq.game_table(split.reset_index(drop=True))

    def test_terminal_placeholder_rows_are_not_decisions(self):
        blocks = dict(blocks_of(self.frame, 'dev'))
        last = blocks['200:1'][-1]
        walk = self.frame.copy(); walk.loc[last, ['pitch_type', 'description', 'events']] = [None, 'automatic_ball', 'walk']
        requests, problem, _ = preq.pa_requests(build_store(walk), blocks['200:1'], 'sha')
        self.assertEqual((len(requests), problem), (3, None))  # ball four by the clock ends the PA: not a decision
        runner = self.frame.copy(); runner.loc[last, ['pitch_type', 'description', 'events']] = [None, None, 'caught_stealing_2b']
        self.assertEqual(len(preq.pa_requests(build_store(runner), blocks['200:1'], 'sha')[0]), 3)
        real = self.frame.copy(); real.loc[last, 'pitch_type'] = None  # a real pitch without a type stays a decision
        self.assertEqual(preq.pa_requests(build_store(real), blocks['200:1'], 'sha')[0][-1].logged_action, pr.MISSING)
        lone = self.frame.copy(); lone.loc[blocks['200:1'][0], ['pitch_type', 'description', 'events']] = [None, None, 'truncated_pa']
        self.assertEqual(preq.pa_requests(build_store(lone), blocks['200:1'][:1], 'sha'), ([], 'no_decision', 0))
        # A real pitch lost just before the closing row is a defect, not a clean end (D-6 critic 1).
        gap = walk.drop(index=blocks['200:1'][-2]).reset_index(drop=True)
        requests, problem, index = preq.pa_requests(build_store(gap), dict(blocks_of(gap, 'dev'))['200:1'], 'sha')
        self.assertEqual((len(requests), problem, index), (2, 'missing_row', 2))
        late = lone.drop(index=blocks['200:1'][1:]).reset_index(drop=True)  # the PA is only its closing row ...
        first = dict(blocks_of(late, 'dev'))['200:1']
        late.loc[first[0], 'pitch_number'] = 2  # ... numbered after pitches that are missing
        self.assertEqual(preq.pa_requests(build_store(late), first, 'sha'), ([], 'missing_row', 0))
        row = est.pa_value([], {'game': 1, 'in_population': False, 'problem': 'no_decision', 'problem_index': 0,
                                'reward': .6})
        self.assertEqual((row['status'], row['delta_bounds'], row['candidate_bounds']), (est.NO_DECISION, [0., 0.], [.6, .6]))

    def test_style_snapshot_is_frozen_and_roster_blind(self):
        as_of = '2025-07-01'
        snapshot = preq.style_snapshot(self.frame, as_of)
        self.assertEqual(snapshot.index[-1], preq.LEAGUE)
        self.assertEqual(snapshot.loc[preq.LEAGUE, [c for c in HISTORY_COLUMNS if 'reliability' in c]].tolist(), [0.] * 6)
        # On the as-of date the rolling prior uses only earlier dates: bit-identical to the snapshot.
        dev = self.frame.loc[self.frame.split.eq('dev')]
        check = preq.snapshot_rolling_mismatches(dev, snapshot, as_of)
        self.assertEqual((check['mismatches'], check['rows_on_as_of'] > 0), (0, True))
        # In-window events and the window roster cannot move it.
        noisy = self.frame.copy()
        late = pd.to_datetime(noisy.game_date) >= as_of
        noisy.loc[late, 'events'] = 'home_run'
        noisy.loc[late, 'batter'] = noisy.loc[late, 'batter'] + 1000
        pd.testing.assert_frame_equal(preq.style_snapshot(noisy, as_of), snapshot)
        applied, report = preq.apply_style_snapshot(safe_rows(noisy.loc[late]), snapshot, as_of)
        self.assertEqual(report['unknown_batter_rows'], int(late.sum()))  # all unseen -> league row
        np.testing.assert_array_equal(applied[list(HISTORY_COLUMNS)].to_numpy(np.float32)[0],
                                      snapshot.loc[preq.LEAGUE].to_numpy(np.float32))
        with self.assertRaisesRegex(pa.IntegrityError, 'before the style as-of'):
            preq.apply_style_snapshot(safe_rows(self.frame), snapshot, as_of)
        path = Path(tempfile.mkdtemp()) / 'style.json'
        _, file_sha = pa.save_style_snapshot(snapshot, as_of, {'test': 1}, path)
        loaded, loaded_as_of, _ = pa.load_style_snapshot(path, file_sha)
        pd.testing.assert_frame_equal(loaded, snapshot)
        self.assertEqual(loaded_as_of, as_of)

    def test_style_snapshot_le2025_output_unchanged(self):
        # Golden digest from the pre-F2 base (6bd76fd): the <=2025 path must stay bit-identical.
        s = preq.style_snapshot(self.frame, '2025-07-01')
        digest = hashlib.sha256(s.to_numpy('float32').tobytes() + '|'.join(s.index).encode()).hexdigest()
        self.assertEqual(digest, 'b1478c46e72b50aa4d8f5b74b7ee057b383dfbc9473e6db0d53923f2cd0513b8')

    def test_style_snapshot_2026_pinned_source_guard_and_end_of_history(self):
        # <=2025 history plus synthetic 2026 regular-season rows (on/after the pinned opening day).
        history = self.frame.drop(columns=list(HISTORY_COLUMNS))
        season = history.copy()
        season['game_date'] = '2026-04-02'
        season['events'] = 'home_run'
        frame = pd.concat([history, season], ignore_index=True)
        snapshot = preq.style_snapshot_2026(frame)
        pd.testing.assert_frame_equal(snapshot, preq.style_snapshot(history, preq.PROFILE_AS_OF_2026))
        self.assertEqual(preq.snapshot_end_of_history_mismatches(history, snapshot),
                         {'batters': len(snapshot) - 1, 'same_batters': True, 'mismatches': 0})
        # A later evaluation-window start would pull 2026 rows into the source: refused.
        with self.assertRaisesRegex(pa.IntegrityError, 'pinned opening day'):
            preq.style_snapshot_2026(frame, '2026-04-10')
        early = frame.copy()
        early.loc[len(history), 'game_date'] = '2026-03-20'  # a 2026 row dated before as-of
        with self.assertRaisesRegex(pa.IntegrityError, r'max\(game_date\) <= 2025-12-31'):
            preq.style_snapshot_2026(early)
        # The replacement check catches a snapshot that absorbed evidence outside the <=2025 history
        # (proxy: extra rows dated late 2025, since a 2026-dated source is refused upstream).
        extra = season.assign(game_date='2025-11-01')
        leaked = preq.style_snapshot(pd.concat([history, extra], ignore_index=True), preq.PROFILE_AS_OF_2026)
        self.assertGreater(preq.snapshot_end_of_history_mismatches(history, leaked)['mismatches'], 0)
        with self.assertRaisesRegex(pa.IntegrityError, '<=2025 rows only'):
            preq.snapshot_end_of_history_mismatches(early, snapshot)

    def test_census_manifest_and_roles(self):
        census = preq.census(self.frame, TYPES)
        dev = census['dev']
        self.assertEqual((dev['pas'], dev['games'], dev['no_pitch_rows'], dev['missing_label_rows']), (16, 2, 2, 2))
        self.assertEqual((dev['pa_first_pitch_number_not_1'], dev['pas_by_first_no_action_position']), (2, {'middle': 4}))
        self.assertEqual((dev['pitchers_unseen_in_train'], dev['count_path_breaks'], dev['codes_outside_vocabulary']),
                         (1, 0, []))
        self.assertEqual(dev['missing_type_by_description'], {'automatic_ball': 2, 'ball': 2})
        self.assertNotIn('outcome_adjacent', dev)  # DEV stays outcome-blind at S0
        self.assertEqual((dev['pas_by_game'], dev['games_with_several_dates'], census['games_spanning_splits']),
                         ({'200': 8, '201': 8}, 0, []))
        self.assertEqual(census['train']['outcome_adjacent']['games_final_v1'], 8)
        self.assertEqual(census['train']['outcome_adjacent']['truncated_pa'], 0)
        self.assertEqual(census['train_pitchers_ambiguous_hand'], 0)
        self.assertEqual(census['bc_p_train']['rows'], int(self.frame.split.eq('train').sum()))
        changed = self.frame.copy()
        block = dict(blocks_of(changed, 'dev'))['200:2']
        changed.loc[block[3:], 'pitcher'] = 8
        changed.loc[block[5:], 'batter'] = 99
        manifest = preq.pa_manifest(changed, block)
        self.assertEqual((manifest['first_pitcher_change_index'], manifest['batter_change_index']), (3, 5))
        self.assertEqual(preq.census(changed, TYPES)['dev']['pa_with_pitcher_change'], 1)
        roles = preq.pitcher_roles(changed)
        self.assertEqual(roles.iloc[block[:3]].tolist(), ['SP'] * 3)
        self.assertEqual(roles.iloc[block[3:]].tolist(), ['RP'] * 4)


# ---------------------------------------------------------------- DR estimator

def toy_rows(steps, candidate, reference, logging, q_hat, mask, pa_id='pa'):
    rows = []
    for t, (history, action) in enumerate(steps):
        p_c, p_r, p_b = candidate(history), reference(history), logging(history)
        a = TOY.VOCAB.index(action)
        q = [q_hat(history, b) if mask[i] else None for i, b in enumerate(TOY.VOCAB)]
        rows.append({'pa_id': pa_id, 'decision_index': t, 'pitcher': '9',
                     'status': 'SUPPORTED' if mask[a] else 'OUTSIDE_POLICY_SUPPORT',
                     'result': {'mask': mask.tolist(), 'logging': p_b.tolist(), 'reference': p_r.tolist(),
                                'candidate': p_c.tolist(), 'logged_index': a, 'q_reference': q,
                                'rho_reference': float(p_r[a] / p_b[a]), 'rho_candidate': float(p_c[a] / p_b[a])}})
    return rows


def refuse_from(rows, k, status='UNSUPPORTED_NO_LOGGED_ACTION'):
    return rows[:k] + [{**rows[k], 'status': status, 'result': None}] + [
        {**r, 'status': 'UNSUPPORTED_MID_PA', 'result': None} for r in rows[k + 1:]]


class EstimatorTests(unittest.TestCase):
    def expected(self, candidate, reference, logging, q_hat, mask):
        total = {'candidate': 0., 'reference': 0.}
        for steps, reward, weight in TOY.trajectories(TOY.PI_B):
            values = est.pa_dr(toy_rows(steps, candidate, reference, logging, q_hat, mask), float(reward))
            for name in total:
                total[name] += weight * values[name]
        return total

    def test_exact_oracle_on_enumerated_toy(self):
        got = self.expected(TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL)  # correct logging law, wrong q
        self.assertAlmostEqual(got['candidate'], TOY.value(TOY.PI_CAND), places=12)
        self.assertAlmostEqual(got['reference'], TOY.value(TOY.PI_REF), places=12)
        masked = self.expected(TOY.PI_REF_RESTRICTED, TOY.PI_REF_RESTRICTED, TOY.PI_B, TOY.wrong_q, TOY.POLICY_MASK)
        self.assertAlmostEqual(masked['reference'], TOY.value(TOY.PI_REF_RESTRICTED), places=12)
        for steps, reward, _ in TOY.trajectories(TOY.PI_B):  # same law in both slots: paired delta exactly 0
            self.assertEqual(est.pa_dr(toy_rows(steps, TOY.PI_REF, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL),
                                       float(reward))['delta'], 0.)

    def test_censored_bounds_contain_the_truth_under_action_dependent_censoring(self):
        """C0 regression (D-5 C): censoring that depends on logged actions after decision 0. The
        flat [0,1] slot of the WIP estimator excluded the truth; the in-recursion bound cannot."""
        truth = TOY.value(TOY.PI_CAND) - TOY.value(TOY.PI_REF)
        scenarios = {'refused_after_SL': lambda steps: 1 if len(steps) > 1 and steps[0][1] == 'SL' else None,
                     'no_end_after_FF': lambda steps: 'end' if steps[-1][1] == 'FF' else None,
                     'end_value_missing_after_SL': lambda steps: 'value' if steps[-1][1] == 'SL' else None}
        for name, censor in scenarios.items():
            lo = hi = 0.
            for steps, reward, weight in TOY.trajectories(TOY.PI_B):
                rows = toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL)
                node = censor(steps)
                info = {'game': 1, 'in_population': True, 'reward': float(reward)}
                if node == 'end':
                    info.update(reward=None, kind=est.NO_TERMINAL)
                elif node == 'value':
                    info.update(reward=None, kind=est.TERMINAL_VALUE_MISSING)
                elif node is not None:
                    rows = refuse_from(rows, node)
                row = est.pa_value(rows, info)
                lo, hi = lo + weight * row['delta_bounds'][0], hi + weight * row['delta_bounds'][1]
            with self.subTest(name):
                self.assertLessEqual(lo, truth + 1e-12)
                self.assertGreaterEqual(hi, truth - 1e-12)
        # The reviewers' one-decision case: 19 FF-logged PAs complete, 1 SL-logged PA without its end.
        def rows(a, pa_id):
            return [{'pa_id': pa_id, 'decision_index': 0, 'pitcher': '9', 'status': 'SUPPORTED', 'result': {
                'mask': [False, True, True], 'logging': [0., .95, .05], 'reference': [0., .95, .05],
                'candidate': [0., .1, .9], 'logged_index': a, 'q_reference': [None, .5, .5],
                'rho_reference': 1., 'rho_candidate': [0., .1 / .95, .9 / .05][a]}}]
        decisions = [r for i in range(19) for r in rows(1, f'a{i}')] + rows(2, 'b')
        pas = {**{f'a{i}': {'game': i, 'in_population': True, 'reward': .5} for i in range(19)},
               'b': {'game': 19, 'in_population': True, 'reward': None, 'kind': est.NO_TERMINAL}}
        result, _ = est.estimate(decisions, pas, draws=50, seed=1, invalid_share_max=1., minimum=MIN)
        bounds = result['layers']['L1_start_population']['delta_bounds']
        self.assertTrue(bounds[0] <= .085 <= bounds[1], bounds)  # the WIP flat bound was [-.05, .05]

    def test_affine_slopes_by_censor_kind(self):
        steps, reward, _ = next(p for p in TOY.trajectories(TOY.PI_B) if len(p[0]) >= 2)
        rows = toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL)
        rho = {n: [r['result'][f'rho_{n}'] for r in rows] for n in est.POLICIES}
        refused = est.pa_value(refuse_from(rows, 1), {'game': 1, 'in_population': True, 'reward': .5})
        self.assertEqual((refused['status'], refused['node'], refused['kind']), (est.CENSORED, 1, est.REFUSED))
        self.assertAlmostEqual(refused['weight_candidate'], rho['candidate'][0], places=15)  # prod_{t<k} rho
        prod = {n: float(np.prod(rho[n])) for n in est.POLICIES}
        missing = est.pa_value(rows, {'game': 1, 'in_population': True, 'reward': None, 'kind': est.TERMINAL_VALUE_MISSING})
        self.assertAlmostEqual(missing['delta_bounds'][1] - missing['delta_bounds'][0],
                               abs(prod['candidate'] - prod['reference']), places=12)  # one shared unknown
        no_end = est.pa_value(rows, {'game': 1, 'in_population': True, 'reward': None, 'kind': est.NO_TERMINAL})
        self.assertAlmostEqual(no_end['delta_bounds'][1] - no_end['delta_bounds'][0],
                               prod['candidate'] + prod['reference'], places=12)  # independent unknowns
        complete = est.pa_dr(rows, float(reward))
        self.assertTrue(missing['delta_bounds'][0] - 1e-12 <= complete['delta'] <= missing['delta_bounds'][1] + 1e-12)
        with self.assertRaisesRegex(pa.IntegrityError, 'E0 start population disagrees'):
            est.pa_value(refuse_from(rows, 0, pr.UNKNOWN_PITCHER), {'game': 1, 'in_population': True, 'reward': .5})

    def test_natural_course_continuation_is_exact_for_the_regime(self):
        """COOP-021: under L1-R an H_k-fixed refusal (after a first-pitch ball) hands both policies to
        the logged behaviour; the expected estimate equals the exact regime values, and the default
        is unchanged."""
        refused = lambda steps: len(steps) > 1 and steps[1][0][0][1] == 'ball'

        def regime(pi):
            return lambda h: TOY.PI_B(h) if len(h) >= 1 and h[0][1] == 'ball' else pi(h)
        got = {'candidate': 0., 'reference': 0.}
        for steps, reward, weight in TOY.trajectories(TOY.PI_B):
            rows = toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL)
            info = {'game': 1, 'in_population': True, 'reward': float(reward)}
            if refused(steps):
                rows = refuse_from(rows, 1)
                worst = est.pa_value(rows, info)
                self.assertEqual(worst, est.pa_value(rows, info, 'worst_case'))  # default = registered D-5
                self.assertNotIn('resolution', worst)
            row = est.pa_value(rows, info, 'l1r')
            self.assertEqual(row['candidate_bounds'][0], row['candidate_bounds'][1])  # point, not a bound
            for name in ('candidate', 'reference'):
                got[name] += weight * row[f'{name}_bounds'][0]
        self.assertAlmostEqual(got['candidate'], TOY.value(regime(TOY.PI_CAND)), places=12)
        self.assertAlmostEqual(got['reference'], TOY.value(regime(TOY.PI_REF)), places=12)
        # Unobserved end after an H_k-fixed refusal: one shared unknown (width |prod rho_c - prod rho_r|).
        steps, _, _ = next(p for p in TOY.trajectories(TOY.PI_B) if refused(p[0]))
        rows = refuse_from(toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL), 1)
        info = {'game': 1, 'in_population': True, 'reward': None, 'kind': est.NO_TERMINAL}
        shared = est.pa_value(rows, info, 'l1r')
        rho = {n: rows[0]['result'][f'rho_{n}'] for n in est.POLICIES}
        self.assertAlmostEqual(shared['delta_bounds'][1] - shared['delta_bounds'][0],
                               abs(rho['candidate'] - rho['reference']), places=12)
        # S-NP: the same refusal outside the natural-course list keeps independent unknowns (G-R residual).
        worst = est.pa_value(rows, {**info, 'reward': .5}, 'l1r', natural=())
        self.assertEqual(worst['resolution'], 'worst_case_residual')
        self.assertAlmostEqual(worst['delta_bounds'][1] - worst['delta_bounds'][0], sum(rho.values()), places=12)
        gap = est.pa_value(rows, {**info, 'reward': .5}, 'l1r', natural=(), gap_delta=.01)  # S-C
        self.assertAlmostEqual(gap['delta_bounds'][1] - gap['delta_bounds'][0],
                               abs(rho['candidate'] - rho['reference']) + 2 * .01 * rho['candidate'], places=12)
        # A v2 positivity row (no recorded result) cannot enter L1-R; unknown rule names are refused.
        positivity = refuse_from(rows, 1, pr.LOGGING_POSITIVITY)
        with self.assertRaisesRegex(pa.IntegrityError, 'runtime v3 required'):
            est.pa_value(positivity, {**info, 'reward': .5}, 'l1r')
        with self.assertRaisesRegex(pa.IntegrityError, 'unknown censoring rule'):
            est.pa_value(rows, info, 'shared')
        no_end = {'game': 2, 'in_population': True, 'reward': None, 'kind': est.NO_TERMINAL, 'date': '2026-04-01'}
        pas = {'1:1': {**info, 'reward': .5, 'date': '2026-04-01'}, '2:1': no_end}
        full = toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL)
        decisions = [{**r, 'pa_id': '1:1'} for r in rows] + [{**r, 'pa_id': '2:1'} for r in full]
        default, _ = est.estimate(decisions, pas, draws=20, seed=1, invalid_share_max=1., minimum=MIN)
        chosen, _ = est.estimate(decisions, pas, draws=20, seed=1, invalid_share_max=1., minimum=MIN, censoring='l1r')
        self.assertNotIn('censoring_rule', default)
        rule = chosen['censoring_rule']
        self.assertEqual((rule['resolutions'], rule['worst_case_residual_share']),
                         ({'natural_course': 1, 'worst_case_residual': 1}, .5))
        self.assertIn('ess_all_e0', chosen)
        width = lambda r: np.subtract(*r['layers']['L1_start_population']['delta_bounds'][::-1])
        self.assertLess(width(chosen), width(default))

    def one_decision(self, f):
        """One pitch: true logging law b = (f, (1-f) .6, (1-f) .4) over (CH, FF, SL); pi_b_hat misses CH
        (TRAIN never saw it) and both policies live on {FF, SL}; the logged CH is a runtime-v3
        positivity row with rho = 0 recorded."""
        b_hat, b = np.array([0., .6, .4]), np.array([f, (1 - f) * .6, (1 - f) * .4])
        pi = {'candidate': np.array([0., .7, .3]), 'reference': b_hat.copy()}
        reward, q = np.array([.3, .5, .8]), [None, .45, .7]

        def row(a, logging):
            rho = {n: (float(pi[n][a] / logging[a]) if a else 0.) for n in est.POLICIES}
            return {'pa_id': 'x', 'decision_index': 0, 'pitcher': '9',
                    'status': 'SUPPORTED' if a else pr.LOGGING_POSITIVITY, 'result': {
                        'mask': [False, True, True], 'logging': logging.tolist(), 'reference': pi['reference'].tolist(),
                        'candidate': pi['candidate'].tolist(), 'logged_index': a, 'q_reference': list(q),
                        **{f'rho_{n}': v for n, v in rho.items()}}}
        truth = {n: float(pi[n] @ reward) for n in est.POLICIES}
        v_hat = {n: float(pi[n][1:] @ np.array(q[1:])) for n in est.POLICIES}
        return row, b, b_hat, reward, truth, v_hat

    def test_positivity_rho_zero_is_exact_and_its_bias_is_minus_f(self):
        """COOP-022: with the true logging law on the supported actions, the rho = 0 positivity node
        makes the expected estimate exact; with pi_b_hat (which misses the new pitch of mass f) the
        error is exactly -f (V - v_hat) per policy: the (1 - f) shrinkage of every other ratio."""
        for f in (0., .1, .3):
            row, b, b_hat, reward, truth, v_hat = self.one_decision(f)
            for logging, exact in ((b, True), (b_hat, False)):
                expected = {n: 0. for n in est.POLICIES}
                for a in range(3):
                    if b[a] == 0:
                        continue
                    value = est.pa_value([row(a, b_hat if a == 0 else logging)],
                                         {'game': 1, 'in_population': True, 'reward': float(reward[a])}, 'l1r')
                    self.assertEqual(value['status'], est.CENSORED if a == 0 else est.COMPLETE)
                    self.assertTrue(value['l2'])  # COOP-019 F4: the rho = 0 point value is in L2
                    for n in est.POLICIES:
                        expected[n] += b[a] * value[f'{n}_bounds'][0]
                for n in est.POLICIES:
                    bias = 0. if exact else -f * (truth[n] - v_hat[n])
                    with self.subTest(f=f, exact=exact, policy=n):
                        self.assertAlmostEqual(expected[n] - truth[n], bias, places=12)
        row, *_ = self.one_decision(.2)
        info = {'game': 1, 'in_population': True, 'reward': .3}
        bad = row(0, np.array([0., .6, .4]))
        bad['result']['rho_candidate'] = float('nan')
        with self.assertRaisesRegex(pa.IntegrityError, 'recorded rho must be 0'):
            est.pa_value([bad], info, 'l1r')
        bad = row(0, np.array([0., .6, .4]))
        bad['result'].update(candidate=[.1, .6, .3], reference=[.1, .5, .4], mask=[True, True, True],
                             q_reference=[.1, .45, .7])
        with self.assertRaisesRegex(pa.IntegrityError, 'off every support'):
            est.pa_value([bad], info, 'l1r')

    def test_l1r_v2_positivity_first_row_and_undefined_end_mean(self):
        """S-m7: a v2 positivity row (no recorded result) at k = 0 is an IntegrityError, not a TypeError.
        S-m5: an L2 made only of rho = 0 point values without an observed end has no end mean (None)."""
        row, *_ = self.one_decision(.2)
        v2 = {**row(0, np.array([0., .6, .4])), 'result': None}
        with self.assertRaisesRegex(pa.IntegrityError, 'runtime v3 required'):
            est.pa_value([v2], {'game': 1, 'in_population': True, 'reward': .3}, 'l1r')
        pas = {'x': {'game': 1, 'in_population': True, 'reward': None, 'kind': est.NO_TERMINAL}}
        result, rows = est.estimate([row(0, np.array([0., .6, .4]))], pas, draws=20, seed=1, invalid_share_max=1.,
                                    minimum=MIN, censoring='l1r')
        self.assertTrue(rows[0]['l2'])
        self.assertIsNone(result['layers']['L2_complete_conditional']['observed_mean_end_value'])
        json.dumps(result, allow_nan=False)  # the stage dump never meets a NaN

    def test_revealed_new_pitch_and_decision_labels(self):
        rows = [{'pa_id': pa_id, 'decision_index': 0, 'pitcher': p, 'status': s} for pa_id, p, s in (
            ('10:1', '9', pr.LOGGING_POSITIVITY), ('10:2', '9', 'SUPPORTED'), ('11:1', '9', 'SUPPORTED'),
            ('12:1', '9', 'SUPPORTED'), ('10:3', '8', 'SUPPORTED'), ('9:7', '9', 'SUPPORTED'))]
        pas = {'10:1': {'date': '2026-04-02'}, '10:2': {'date': '2026-04-02'}, '11:1': {'date': '2026-04-02'},
               '12:1': {'date': '2026-04-03'}, '10:3': {'date': '2026-04-02'}, '9:7': {'date': '2026-04-02'}}
        # F-m1 strict (date, game_pk, at_bat): a later PA of the same game, a larger game_pk that day and a later
        # date are excluded; a smaller game_pk that day (even a larger at-bat) and another pitcher are kept
        self.assertEqual(est.revealed_new_pitch(rows, pas), {'10:2', '11:1', '12:1'})
        result = {'bootstrap': {'draws': 100, 'L1_lower_endpoint_ci95': [.005, .01],
                                'L1_upper_endpoint_ci95': [.006, .012], 'L2_invalid_replicates': 0},
                  'censoring_rule': {'worst_case_residual_share': .001}, 'ess': {'label': None}}
        kw = dict(mei=.001, b_v=.0027, gr_cap=.005, invalid_share_max=.05, pair_pass=True)
        label = lambda r=result, **k: est.decide(r, **{**kw, **k})['label']
        self.assertEqual(label(), 'IMPROVEMENT_SUPPORTED')
        self.assertIn('2 s = 0.002000', est.decide(result, **kw)['passable_effect_approx'])
        self.assertEqual(label(b_v=.0045), 'IMPROVEMENT_SUPPORTED_STATISTICAL')
        self.assertEqual(label(pair_pass=False), 'FAILED_INTEGRITY')
        self.assertEqual(label(gr_cap=.0005), 'NOT_DECIDABLE_CENSORING')
        self.assertEqual(label({**result, 'ess': {'label': 'UNCONFIRMED_WEAK_OVERLAP'}}), 'UNCONFIRMED_WEAK_OVERLAP')
        harm = {**result, 'bootstrap': {**result['bootstrap'], 'L1_lower_endpoint_ci95': [-.01, -.005],
                                        'L1_upper_endpoint_ci95': [-.004, -.001]}}
        self.assertEqual(label(harm), 'HARM_SUPPORTED')
        null = {**result, 'bootstrap': {**result['bootstrap'], 'L1_lower_endpoint_ci95': [-.002, 0.],
                                        'L1_upper_endpoint_ci95': [0., .002]}}
        self.assertEqual(label(null), 'NO_EVIDENCE_OF_IMPROVEMENT')

    def test_ledger_values_are_revalidated(self):
        steps, reward, _ = next(path for path in TOY.trajectories(TOY.PI_B)
                                if all(TOY.PI_CAND(h)[TOY.VOCAB.index(a)] > 0 for h, a in path[0]))
        rows = toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL)

        def refused(message, mutate, reward=float(reward)):
            bad = copy.deepcopy(rows); mutate(bad)
            with self.assertRaisesRegex(pa.IntegrityError, message):
                est.pa_dr(bad, reward)
        refused('recorded ratio differs', lambda r: r[0]['result'].update(rho_candidate=r[0]['result']['rho_candidate'] * 1.01))
        refused('exactly on the mask', lambda r: r[0]['result']['q_reference'].__setitem__(0, None))
        refused('terminal WE', lambda r: None, reward=1.2)
        refused('ordered PA prefix', lambda r: r.pop(0))
        refused('only evaluated', lambda r: r[0].update(status=pr.MID_PA))
        refused('mass outside the policy mask', lambda r: r[0]['result'].update(mask=[False, True, True]))
        refused('status differs', lambda r: r[0].update(status='OUTSIDE_POLICY_SUPPORT'))

    def test_layers_statuses_bootstrap_and_secondary(self):
        paths = [p for p in TOY.trajectories(TOY.PI_B) if len(p[0]) >= 2][:4]
        decisions, pas = [], {}
        for i, (steps, reward, _) in enumerate(paths):
            decisions += toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL, pa_id=f'g{i % 2}:{i}')
            pas[f'g{i % 2}:{i}'] = {'game': i % 2, 'in_population': True, 'reward': float(reward)}
        # A pitcher change at 1 refused as an unknown pitcher: primary censored, secondary evaluable.
        steps, reward, _ = paths[0]
        change = refuse_from(toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL, pa_id='g1:c'), 1,
                             pr.UNKNOWN_PITCHER)
        change[1:] = [{**r, 'pitcher': '77'} for r in change[1:]]
        decisions += change
        pas['g1:c'] = {'game': 1, 'in_population': True, 'reward': float(reward), 'first_pitcher_change_index': 1}
        pas['g0:x'] = {'game': 0, 'in_population': False, 'start_reason': pr.UNKNOWN_PITCHER}
        decisions.append({'pa_id': 'g0:x', 'decision_index': 0, 'pitcher': '5', 'status': pr.UNKNOWN_PITCHER, 'result': None})
        pas['g1:u'] = {'game': 1, 'in_population': False, 'problem': 'ordering', 'problem_index': 0}
        kwargs = dict(draws=300, seed=7, invalid_share_max=.5, minimum=MIN)
        result, rows = est.estimate(decisions, pas, **kwargs)
        self.assertEqual(result['status'], {est.COMPLETE: 4, est.CENSORED: 1, est.EXCLUDED_PRE_START: 1,
                                            est.UNSUBMITTABLE: 1})
        layers = result['layers']
        self.assertEqual((layers['L0_all_pas']['pas'], layers['L1_start_population']['pas'],
                          layers['L2_complete_conditional']['pas']), (7, 5, 4))
        complete = [r['delta'] for r in rows if r['status'] == est.COMPLETE]
        self.assertAlmostEqual(layers['L2_complete_conditional']['delta_mean'], np.mean(complete), places=15)
        l1 = layers['L1_start_population']['delta_bounds']
        self.assertAlmostEqual(layers['L0_all_pas']['delta_bounds'][0], (l1[0] * 5 - 2) / 7, places=12)
        self.assertEqual(est.estimate(decisions, pas, **kwargs)[0]['bootstrap'], result['bootstrap'])  # seeded
        self.assertIsNone(est.estimate(decisions, pas, **{**kwargs, 'minimum': {'games': 30, 'pa_starts': 50}})[0][
            'bootstrap']['inference'])
        secondary = result['secondary_natural_course_after_pitcher_change']
        self.assertEqual(secondary['changed_pas'], 1)
        point = secondary['all_secondary_evaluable']  # the changed PA is point-valued there
        self.assertLess(point['delta_bounds'][1] - point['delta_bounds'][0], l1[1] - l1[0])
        failed = decisions + [{'pa_id': 'g0:f', 'decision_index': 0, 'pitcher': '9', 'status': 'FAILED_INTEGRITY',
                               'result': None}]
        with self.assertRaisesRegex(pa.IntegrityError, 'halted ledger'):
            est.estimate(failed, {**pas, 'g0:f': {'game': 0, 'in_population': True, 'reward': .5}}, **kwargs)
        with self.assertRaisesRegex(pa.IntegrityError, 'manifest'):
            est.estimate(decisions, {**pas, 'g1:c': {**pas['g1:c'], 'first_pitcher_change_index': 2}}, **kwargs)
        self.assertIsNone(est.effective_sample_size([0., 0.]))
        self.assertEqual(est.effective_sample_size([1., 1., 2.], games=[1, 1, 2]), 2.)

    def test_invalid_replicates_and_missing_overlap_fail_closed(self):
        steps, reward, _ = next(p for p in TOY.trajectories(TOY.PI_B) if len(p[0]) >= 2)
        rows = toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL, pa_id='a')
        censored = refuse_from(toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL, pa_id='b'), 1)
        pas = {'a': {'game': 'A', 'in_population': True, 'reward': float(reward)},
               'b': {'game': 'B', 'in_population': True, 'reward': float(reward)}}
        strict, _ = est.estimate(rows + censored, pas, draws=400, seed=3, invalid_share_max=0, minimum=MIN)
        self.assertGreater(strict['bootstrap']['L2_invalid_replicates'], 0)  # resamples of game B alone
        self.assertIsNone(strict['bootstrap']['L2_conditional_ci95'])
        loose, _ = est.estimate(rows + censored, pas, draws=400, seed=3, invalid_share_max=1., minimum=MIN)
        self.assertIsNotNone(loose['bootstrap']['L2_conditional_ci95'])
        none, _ = est.estimate(censored, {'b': pas['b']}, draws=10, seed=3, invalid_share_max=1., minimum=MIN,
                               ess_gate={'pa': 1., 'game': 1.})
        self.assertEqual(none['ess']['label'], 'UNCONFIRMED_WEAK_OVERLAP')  # no complete PA: overlap unestablished
        zero = copy.deepcopy(rows)
        for r in zero:  # a candidate with zero weight on every logged action: ESS undefined -> gate fails
            r['result']['candidate'] = [1., 0., 0.] if r['result']['logged_index'] else [0., 1., 0.]
            r['result']['rho_candidate'] = 0.
        gated, _ = est.estimate(zero, {'a': pas['a']}, draws=10, seed=3, invalid_share_max=1., minimum=MIN,
                                ess_gate={'pa': 1., 'game': 1.})
        self.assertEqual((gated['ess']['gate_min_candidate_reference']['pa'], gated['ess']['label']),
                         (None, 'UNCONFIRMED_WEAK_OVERLAP'))


# ---------------------------------------------------------------- tau selection (D-9 S3b)

class TauTests(unittest.TestCase):
    def ledger(self):
        rows, facts = [], {}
        for i, (steps, _, _) in enumerate(TOY.trajectories(TOY.PI_B)[:40]):
            pa_id = f'{i % 8}:{i}'
            for row in toy_rows(steps, TOY.PI_REF, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, FULL, pa_id=pa_id):
                history = steps[row['decision_index']][0]
                row['result']['q_planning'] = [TOY.wrong_q(history, b) for b in TOY.VOCAB]
                row['result']['q_planning_diff_se'] = [.001, 0., .002]
                rows.append(row)
            facts[pa_id] = {'game': i % 8, 'in_population': True, 'terminal_marker': True}
        return rows, facts

    def test_tau_rule_selects_smallest_passing_tau_without_outcomes(self):
        rows, facts = self.ledger()
        loose = {'pa_ess_ratio_min': .01, 'game_ess_min': 1., 'ess_ratio_candidate_reference_min': .01,
                 'noise_ratio_q90_max': 100., 'safety_multiplier': 1.}
        taus = [.001, .01, .1, 1.]
        result = ptau.tau_table(rows, facts, taus, loose)
        self.assertEqual((result['status'], result['selected_tau']), ('SELECTED', .001))
        self.assertEqual(len({json.dumps(e['reference']) for e in result['table']}), 1)  # reference ignores tau
        strict = {**loose, 'noise_ratio_q90_max': 1.5}  # max diff s.e. .002 / tau <= 1.5 needs tau >= .00133
        self.assertEqual(ptau.tau_table(rows, facts, taus, strict)['selected_tau'], .01)
        self.assertEqual(ptau.tau_table(rows, facts, taus, {**loose, 'ess_ratio_candidate_reference_min': 10.})['status'],
                         'P3_NOT_EVALUABLE')
        self.assertEqual(ptau.tau_table(rows, facts, taus, {**loose, 'game_ess_min': 1e6})['status'], 'ARM_B_NOT_EVALUABLE')
        with self.assertRaisesRegex(pa.IntegrityError, 'thresholds'):
            ptau.tau_table(rows, facts, taus, {**loose, 'noise_ratio_q90_max': None})
        defect = {**facts, next(iter(facts)): {**next(iter(facts.values())), 'problem': 'missing_row'}}
        self.assertEqual(ptau.tau_table(rows, defect, taus, loose)['counted_not_weighted_pas'], 1)  # prefix not weighted
        noisy = copy.deepcopy(rows)
        for r in noisy:
            r['result']['q_planning_diff_se'] = [None, None, None]  # samples = 1: no paired s.e.
        with self.assertRaisesRegex(pa.IntegrityError, 'samples >= 3'):
            ptau.tau_table(noisy, facts, taus, loose)

    def test_search_settings_use_measured_rows(self):
        profiles = [{'samples': 2, 'pitch_cap': 8, 'decisions': 10, 'conditional_rows': 300},
                    {'samples': 4, 'pitch_cap': 8, 'decisions': 10, 'conditional_rows': 600},
                    {'samples': 16, 'pitch_cap': 4, 'decisions': 10, 'conditional_rows': 2000}]
        self.assertEqual(ptau.select_search_settings(profiles, {'rows': 60000, 'decisions': 1000}),
                         {'samples': 4, 'pitch_cap': 8})  # (16, 4) needs 200 rows per decision
        self.assertIsNone(ptau.select_search_settings(profiles, {'rows': 10, 'decisions': 1000}))
        self.assertEqual(ptau.select_search_settings(profiles, {'rows': 10 ** 6, 'decisions': 1000}, min_samples=3),
                         {'samples': 16, 'pitch_cap': 4})  # below the noise-rule minimum is never selected
        self.assertIsNone(ptau.select_search_settings(profiles[:2], {'rows': 10 ** 6, 'decisions': 1000}, min_samples=5))


# ---------------------------------------------------------------- runner stages

class RunnerFixture(unittest.TestCase):
    """Synthetic games, store and pinned synthetic G0/WE files (no tests of its own)."""
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.frame = game_frame()
        self.store = build_store(self.frame)
        self.syn = SyntheticPolicy(self.root / 'g0')
        train = self.frame.loc[self.frame.split.eq('train')]
        self.keys = train.loc[~train.game_pk.eq(107), ['game_pk', 'at_bat_number', 'pitch_number']].reset_index(drop=True)
        self.gates = {'bc_e_rows': len(self.keys), 'bc_e_actions': ['CH', 'FF', 'SL'], 'token_vocabulary': list(TYPES),
                      'train_rows': int(self.frame.split.eq('train').sum())}

    def materialize(self, gates=None):
        out = self.root / f'S1-{len(list(self.root.glob("S1-*")))}'
        out.mkdir()
        bundle_path, bundle_sha = self.syn.bundle_file()
        provenance = {'source_ids': {'synthetic': 'a' * 64}, 'config_sha256': 'b' * 64, 'code_commit': 'c' * 40,
                      'data_version': 'synthetic'}
        report = rpv.run_materialize(
            self.frame, self.store, out, provenance=provenance,
            bind_inputs=lambda art, rows: pi.bind_policy_inputs(bundle_path, bundle_sha, self.syn.paths, bc_artifact=art,
                                                                context_rows=rows, aux_classes=self.syn.classes['aux']),
            train_keys=self.keys, keys_record={'n': len(self.keys), 'rows_sha256': ordered_key_hash(self.keys)},
            gates=gates or self.gates, bc_parameters={'prior_strength': 20., 'minimum_action_count': 1},
            no_pitch=NO_PITCH, volume_quantiles=[1 / 3, 2 / 3], prep_vocabulary=TYPES,
            bc_p_only_rule='evaluate; report as a stratum')
        return out, report

    def pins(self, out, report):
        return {'bc': (out / 'bc_BC_P.json', report['bc']['BC_P']['file_sha256']),
                'support': (out / 'support_primary.json', report['support_table']['file_sha256']),
                'hands': (out / 'hands.json', report['hands']['file_sha256'])}

    def candidate(self, pins, blocks, name, *, snapshot=None, as_of=None, evaluation_seed=None, loader=load_member,
                  tau=.01, samples=2):
        contexts, _ = rpv.pa_contexts(self.store, blocks, snapshot, as_of)
        components = self.syn.bind(pa.load_train_bc(*pins['bc']), contexts, loader=loader)
        runtime = pr.build_runtime(*pins['bc'], *pins['support'], self.root / name, components=components,
                                   budget=RowBudget(10 ** 8, seed_count=5), tau=tau, samples=samples, pitch_cap=4, seed=701,
                                   evaluation_seed=evaluation_seed, hand_registry=pins['hands'])
        return runtime, components


class RunnerStageTests(RunnerFixture):
    def test_materialize_gates_hands_and_support(self):
        out, report = self.materialize()
        train = self.frame.loc[self.frame.split.eq('train')]
        self.assertEqual(report['bc']['BC_P']['train_rows'], len(train))  # no outcome/support filter
        self.assertEqual(report['bc']['BC_E']['train_rows'], len(self.keys))
        self.assertTrue(all(report['s1_gates']['checks'].values()))
        self.assertEqual(report['change_decisions']['decisions_after_change'], 0)
        self.assertEqual((report['bc_p_only_pitchers']['ids'],
                          report['bc_p_only_pitchers']['support_actions_from_league_tiers_only']), ([], 0))
        game107 = train.loc[train.game_pk.eq(107)]  # the rows BC_E leaves out
        self.assertEqual(report['bc_p_minus_bc_e_codes'],
                         {k: int(v) for k, v in game107.pitch_type.value_counts().sort_index().items()})
        self.assertEqual((report['hands']['single'], report['hands']['ambiguous']), (1, 0))
        table, _ = pa.load_support_table(out / 'support_primary.json', report['support_table']['file_sha256'],
                                         pa.load_train_bc(out / 'bc_BC_P.json', report['bc']['BC_P']['file_sha256']))
        self.assertEqual((table[('9', 'L')].tolist(), table[('9', 'R')].tolist()), ([False, True, True], [False] * 3))
        self.assertEqual(report['logging_mass_on_mask']['decision_share_mass_equal_one'], 0.)  # CH off the mask
        with self.assertRaisesRegex(pa.IntegrityError, 'S1 gate failed'):
            self.materialize(gates={**self.gates, 'bc_e_rows': len(self.keys) + 1})
        with self.assertRaisesRegex(pa.IntegrityError, 'train_rows_equal_registered'):
            self.materialize(gates={**self.gates, 'train_rows': 1})
        ambiguous = train.copy()
        ambiguous.loc[ambiguous.index[0], 'p_throws'] = 'L'
        self.assertEqual(pi.hand_registry(ambiguous, ('9',)), {'9': 'AMBIGUOUS'})
        self.assertEqual(len(pi.support_templates(safe_rows(ambiguous), {'9': 'AMBIGUOUS'})), 0)
        self.assertEqual(pi.hand_registry(train.assign(p_throws=None), ('9',)), {'9': 'AMBIGUOUS'})

    def test_v5_and_dr_passes_end_to_end(self):
        out, report = self.materialize()
        pins = self.pins(out, report)
        runtime = pr.build_runtime(*pins['bc'], *pins['support'], self.root / 'v5.jsonl', hand_registry=pins['hands'])
        dev = blocks_of(self.frame, 'dev')
        v5 = rpv.run_v5(runtime, self.store, dev, rpv.Deadline(), no_pitch=NO_PITCH)
        requests = v5['summary']['request_status']
        self.assertEqual((v5['pas_in_split'], v5['structural_problems']), (16, {'missing_row': 2}))
        for status in (pr.NO_LOGGED_ACTION, pr.MISSING_LABEL, pr.UNKNOWN_PITCHER):
            self.assertEqual(requests[status], 2)
        self.assertGreater(requests[pr.OUTSIDE_POLICY_SUPPORT], 0)  # logged CH: valid rho=0
        self.assertEqual((v5['summary']['run_status'], v5['sealed_counts']['pas']), ('OK', 16))
        self.assertEqual(v5['start_population']['inside'], 12)
        snapshot = preq.style_snapshot(self.frame, '2025-07-01')
        runtime, components = self.candidate(pins, dev, 'dr.jsonl', snapshot=snapshot, as_of='2025-07-01',
                                             evaluation_seed=1701)
        boot = {'draws': 200, 'seed': 5, 'invalid_share_max': .5, 'minimum': MIN}
        sensitivities = ['r5-events-v1', 'flags_to_bounds', 'post_pitch_scores']
        pair = lambda: pr.build_reference_pair_runtime(*pins['bc'], *pins['support'], self.root / 'pair.jsonl',
                                                        hand_registry=pins['hands'])
        result, rows = rpv.run_dr(runtime, self.store, components, dev, rpv.Deadline(), no_pitch=NO_PITCH, bootstrap=boot,
                                  ess_gate={'pa': 100., 'game': 30.}, sensitivities=sensitivities,
                                  strata=rpv.strata_function(self.frame, [1., 2.]), expected_pas=16, pair=pair)
        self.assertEqual(result['status'], {est.COMPLETE: 8, est.CENSORED: 4, est.EXCLUDED_PRE_START: 2,
                                            est.UNSUBMITTABLE: 2})
        self.assertEqual((result['censor_kinds'], result['censor_nodes']), ({est.REFUSED: 4}, {'1': 4}))
        self.assertEqual((result['layers']['L0_all_pas']['pas'], result['layers']['L1_start_population']['pas']), (16, 12))
        self.assertEqual(result['ess']['label'], 'UNCONFIRMED_WEAK_OVERLAP')
        self.assertIsNone(result['causal_effect'])
        self.assertEqual(result['sensitivity']['r5-events-v1']['status'][est.COMPLETE], 6)  # truncated PAs censored
        self.assertEqual({k: result['paired_identity_run'][k] for k in ('pas', 'max_abs_delta', 'pass')}, {
            'pas': 8, 'max_abs_delta': 0., 'pass': True})  # an independent cand=ref callable through the runtime
        self.assertEqual(set(result['strata']), {'end_kind', 'role', 'month', 'volume_bin', 'extra_innings',
                                                 'bc_p_only_pitcher'})
        rows_supported = [r for r in runtime.ledger.decisions() if r['status'] == pr.SUPPORTED]
        for row in rows_supported:  # D-9: the tau table recomputes exactly the runtime's candidate law
            self.assertAlmostEqual(ptau._decision(row, .01)[0], row['result']['rho_candidate'], places=12)
            r, a = row['result'], row['result']['logged_index']
            q = np.array([np.nan if v is None else v for v in r['q_planning']])
            for tau in (.003, .1, 1.):  # ... and at other taus the law kl_policy gives from the recorded values
                expected = kl_policy(np.where(r['mask'], q, -np.inf), np.array(r['reference']), np.array(r['mask']), tau)
                self.assertAlmostEqual(ptau._decision(row, tau)[0], expected[a] / r['logging'][a], places=12)
        self.assertGreater(len({round(ptau._decision(row, t)[0], 9) for row in rows_supported[:1] for t in (.003, 1.)}), 1)
        q = next(r for r in runtime.ledger.decisions() if r['status'] == pr.SUPPORTED)['result']
        self.assertEqual(q['q_source'], 'evaluation_seed')  # M-7: the DR q-hat is an independent search
        self.assertNotEqual(q['q_reference'], q['q_planning'])
        header = json.loads(runtime.ledger.path.read_text().splitlines()[0])
        self.assertEqual(header['pins']['hand_registry_sha256'], runtime.pins['hand_registry_sha256'])
        again, components = self.candidate(pins, dev, 'dr.jsonl', snapshot=snapshot, as_of='2025-07-01',
                                           evaluation_seed=1701)
        replay, _ = rpv.run_dr(again, self.store, components, dev, rpv.Deadline(), no_pitch=NO_PITCH, bootstrap=boot,
                               ess_gate=None, sensitivities=[])
        self.assertEqual(replay['layers'], result['layers'])  # the ledger replays every request

    def test_stage_records_hang_guard_and_deadline(self):
        with rpv.stage(self.root / 'ok', 'census', {'config_sha256': 'x'}) as out:
            (out / 'census.json').write_text('{}')
        manifest = json.loads((self.root / 'ok/manifest.json').read_text())
        self.assertEqual(set(manifest['artifact_sha256']), {'census.json', 'started.json'})
        self.assertIn('wall_seconds', manifest['cost'])
        with self.assertRaises(rpv.HangGuardExceeded):
            with rpv.stage(self.root / 'hang', 'census', {}, hang_guard_seconds=.05):
                time.sleep(1)
        failure = json.loads(next((self.root / 'hang').glob('failure-*.json')).read_text())
        self.assertEqual((failure['error_type'], failure['citable']), ('HangGuardExceeded', False))
        self.assertFalse((self.root / 'hang/manifest.json').exists())
        with self.assertRaisesRegex(pa.IntegrityError, 'stage output exists'):
            with rpv.stage(self.root / 'ok', 'census', {}):
                pass
        with self.assertRaises(rpv.BudgetExceeded):
            rpv.Deadline(1e-9).check()
        with self.assertRaisesRegex(pa.IntegrityError, 'after 2025'):
            rpv.guard_dates(self.frame.assign(game_date='2026-04-01'))

    def test_registered_sensitivities_change_what_they_should(self):
        out, report = self.materialize()
        pins = self.pins(out, report)
        frame = self.frame.copy()
        dev = dict(blocks_of(frame, 'dev'))
        frame.loc[dev['200:2'][-1], 'post_home_score'] = 1  # the post-pitch score disagrees with the next row
        # (200:2 never throws the off-mask CH, so its ratio products stay positive and the end value matters)
        self.frame, self.store = frame, build_store(frame)
        blocks = [(k, dev[k]) for k in ('200:1', '200:2')]
        runtime, components = self.candidate(pins, blocks, 'sens.jsonl')
        boot = {'draws': 20, 'seed': 5, 'invalid_share_max': 1., 'minimum': MIN}
        result, rows = rpv.run_dr(runtime, self.store, components, blocks, rpv.Deadline(), no_pitch=NO_PITCH,
                                  bootstrap=boot, ess_gate=None, sensitivities=['flags_to_bounds', 'post_pitch_scores'])
        self.assertEqual(result['reward_flags'], 1)
        flagged = result['sensitivity']['flags_to_bounds']
        self.assertEqual((result['status'][est.COMPLETE], flagged['status'][est.COMPLETE]), (2, 1))
        self.assertNotEqual(flagged['layers']['L1_start_population']['delta_bounds'],
                            result['layers']['L1_start_population']['delta_bounds'])
        games = preq.game_table(self.store.frame, {200})
        facts = {'200:2': {'positions': dev['200:2'], 'game': 200}}
        scored = SimpleNamespace(defense_we=lambda state, home: .5 + .1 * (state.home_score - state.away_score) * (1 if home else -1))
        primary = rpv.variant_facts(facts, 'r5-events-v1', self.store, scored, games)['200:2']['reward']
        post = rpv.variant_facts(facts, 'post_pitch_scores', self.store, scored, games)['200:2']['reward']
        self.assertNotEqual(post, primary)  # the post-pitch 1-0 score enters the end state only in this sensitivity

    def test_bind_probe_and_straddling_games(self):
        out, report = self.materialize()
        pins = self.pins(out, report)
        dev = blocks_of(self.frame, 'dev')
        clean = np.r_[dict(dev)['200:1'], dict(dev)['200:2']]
        rows = clean[self.frame.pitch_type.iloc[clean].isin(['FF', 'SL']).to_numpy()][:6]  # rows with action pools
        components = self.syn.bind(pa.load_train_bc(*pins['bc']), rpv.pa_contexts(self.store, rpv.blocks_containing(
            self.frame, rows))[0])
        requests = {}
        for _, block in rpv.blocks_containing(self.frame, rows):
            requests.update(zip(block.tolist(), preq.pa_requests(self.store, block, 'probe')[0]))
        states = [requests[int(r)].state for r in rows]
        actions = [requests[int(r)].logged_action for r in rows]
        primary = pi.integrated_predictions(components.g0, states, actions)
        raw = components.g0.baseline.baseline.predict(pd.DataFrame([components.inputs.query(s, a)
                                                                     for s, a in zip(states, actions)]))
        sealed = {'keys': self.frame.iloc[rows][['game_pk', 'at_bat_number', 'pitch_number']].to_numpy(),
                  'levels': [np.zeros(len(rows), int)] * 5, 'primary': primary, 'frequency_raw': raw}
        probe = rpv.run_bind_probe(self.store, components, sealed, count=len(rows), atol_primary=1e-6,
                                   atol_frequency=1e-12, no_pitch=NO_PITCH)
        self.assertEqual((probe['pass'], probe['probe_dates']), (True, {'2025-07-01': len(rows)}))
        self.assertFalse(rpv.run_bind_probe(self.store, components, {**sealed, 'primary': primary + .01}, count=len(rows),
                                            atol_primary=1e-6, atol_frequency=1e-12, no_pitch=NO_PITCH)['pass'])
        moved = self.frame.copy(); moved.loc[dict(dev)['200:1'], 'split'] = 'blend'  # game 200 starts in June
        self.assertEqual(rpv.straddling_games(moved), [200])
        self.assertEqual(rpv.select_games(moved, 'dev', 1, 3), [201])
        self.assertEqual(set(moved.iloc[np.concatenate([p for _, p in rpv.game_blocks(moved, [200, 201], 'dev')])].split),
                         {'dev'})

    def test_ledger_append_survives_the_hang_guard(self):
        """C10: the alarm may reach any thread (torch workers); the append must still land whole."""
        out, report = self.materialize()
        pins = self.pins(out, report)
        runtime = pr.build_runtime(*pins['bc'], *pins['support'], self.root / 'hang.jsonl', hand_registry=pins['hands'])
        request = preq.pa_requests(self.store, dict(blocks_of(self.frame, 'dev'))['200:1'], runtime.sha256)[0][0]
        stop = threading.Event()
        worker = threading.Thread(target=stop.wait, daemon=True)  # a thread that does not block SIGALRM
        worker.start()
        real_fsync = pr.os.fsync
        previous = rpv.signal.signal(rpv.signal.SIGALRM, rpv._raise_hang)
        try:
            def slow_fsync(fd):
                real_fsync(fd)
                time.sleep(.3)  # the alarm expires here, inside the append
            rpv.signal.setitimer(rpv.signal.ITIMER_REAL, .05)
            with mock.patch.object(pr.os, 'fsync', slow_fsync), self.assertRaises(rpv.HangGuardExceeded):
                runtime.submit(request)
        finally:
            rpv.signal.setitimer(rpv.signal.ITIMER_REAL, 0)
            rpv.signal.signal(rpv.signal.SIGALRM, previous)
            stop.set()
        runtime.abort('HangGuardExceeded')
        replay = pr.build_runtime(*pins['bc'], *pins['support'], self.root / 'hang.jsonl', hand_registry=pins['hands'])
        self.assertEqual((replay.summary()['requests'], replay.summary()['run_status']), (1, 'HALTED'))

    def test_hang_guard_inside_the_search_is_not_a_decision_row(self):
        out, report = self.materialize()
        pins = self.pins(out, report)
        dev = [b for b in blocks_of(self.frame, 'dev') if b[0] == '200:1']
        runtime, _ = self.candidate(pins, dev, 'search.jsonl')
        request = preq.pa_requests(self.store, dev[0][1], runtime.sha256)[0][0]
        with mock.patch.object(runtime.improvement, 'q_values', side_effect=rpv.HangGuardExceeded('guard')):
            with self.assertRaises(rpv.HangGuardExceeded):
                rpv.submit_pas(runtime, self.store, dev, rpv.Deadline(), no_pitch=NO_PITCH)
        self.assertEqual([r['kind'] for r in runtime.ledger.rows], ['header', 'aborted'])  # no FAILED_* or malformed row

    def test_candidate_context_hand_mismatch_halts(self):
        out, report = self.materialize()
        pins = self.pins(out, report)
        dev = [b for b in blocks_of(self.frame, 'dev') if b[0] == '200:1']
        runtime, _ = self.candidate(pins, dev, 'hand.jsonl')
        request = preq.pa_requests(self.store, dev[0][1], runtime.sha256)[0][0]
        with self.assertRaisesRegex(pa.IntegrityError, 'hand differs from the bound context row'):
            runtime.submit(pr.DecisionRequest(request.request_id, request.pa_id, 0, request.state, request.logged_action,
                                              runtime.sha256, 'L'))
        self.assertEqual(runtime.summary()['run_status'], 'HALTED')

    def test_game_selection_is_month_stratified_and_salted(self):
        many = pd.DataFrame({'game_pk': range(30), 'game_date': ['2025-07-05'] * 10 + ['2025-08-05'] * 20,
                             'split': 'dev'})
        chosen = rpv.select_games(many, 'dev', 6, 11)
        self.assertEqual((sum(g < 10 for g in chosen), len(chosen)), (2, 6))
        self.assertEqual(chosen, rpv.select_games(many, 'dev', 6, 11))
        self.assertNotEqual(chosen, rpv.select_games(many, 'dev', 6, 12))


class LoadInputsTests(unittest.TestCase):
    def test_frame_follows_the_g0_date_order(self):
        """D98: game_pk order can disagree with game dates (a postponed game); the loaded frame keeps
        the G0 order (game date, then pitch key) the history store requires."""
        frame = game_frame()
        frame.loc[frame.game_pk.eq(100), 'game_pk'] = 999  # the earliest TRAIN game gets the largest game_pk
        normalizer = PhysicalNormalizer().fit(frame.loc[frame.split.eq('train')])
        docs = {'bundle': {'files': {role: {'path': f'/{role}', 'sha256': role}
                                     for role in ('p4_parent_preparation', 'p4_preparation', 'p4_auxiliary')}},
                'p4_parent_preparation': {}, 'p4_preparation': {'features': {'tokens': {'type_vocabulary': TYPES}}}}
        config = {'identity_registration': {'g0_bundle': {'path': 'bundle.json', 'file_sha256': 'bundle'}, 'we_paths': {}}}
        with mock.patch('run_ml_benchmark.regular_frame', return_value=frame.sample(frac=1, random_state=0)), \
                mock.patch.object(rpv.pid, 'pinned_json', lambda path, sha: docs[sha]), \
                mock.patch.object(rpv.pid, 'pinned_pickle', return_value={'normalizer': normalizer}):
            out = rpv.load_inputs(config, {})
        self.assertTrue(pd.to_datetime(out['frame'].game_date).is_monotonic_increasing)
        expected = frame.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
        pd.testing.assert_frame_equal(out['frame'][['game_pk', 'at_bat_number', 'pitch_number']],
                                      expected[['game_pk', 'at_bat_number', 'pitch_number']].reset_index(drop=True))
        self.assertEqual(len(out['store'].frame), len(frame))


    def test_holdout_frame_season_order_and_suspended_games(self):
        """ope-2026 frame (SYNTHETIC rows only): regular season, G0 order, games on several dates
        dropped by schedule, style placeholders, and the season guard."""
        frame = game_frame()
        raw = frame.loc[frame.split.eq('dev')].assign(game_type='R', on_1b=1., on_2b=np.nan, on_3b=np.nan)
        raw['game_date'] = raw.game_date.str.replace('2025-07', '2026-05')
        raw.loc[raw.game_pk.eq(201) & raw.at_bat_number.ge(5), 'game_date'] = '2026-05-09'  # suspended, resumed
        raw = raw[list(rpv.HOLDOUT_COLUMNS)].sample(frac=1, random_state=1)
        out, dropped = rpv.holdout_frame(raw)
        self.assertEqual((dropped, sorted(out.game_pk.unique())), ([201], [200]))
        self.assertTrue(out[['at_bat_number', 'pitch_number']].apply(tuple, axis=1).is_monotonic_increasing)
        self.assertEqual((set(out.bases), set(out.split)), ({1}, {'eval2026'}))
        self.assertTrue(out[list(HISTORY_COLUMNS)].isna().all().all())
        with self.assertRaisesRegex(pa.IntegrityError, 'outside the registered 2026 season'):
            rpv.holdout_frame(raw.assign(game_date='2026-10-02'))
        self.assertEqual(len(rpv.holdout_frame(pd.concat([raw, raw.assign(game_type='S', game_pk=5)]))[0]), len(out))

    def test_failure_record_hashes_partial_ledgers_unread(self):
        """F-m2: a failed stage records the sha256 of every partial ledger file (bytes only, never parsed)."""
        out = Path(tempfile.mkdtemp()) / 'stage'
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            with rpv.stage(out, rpv.OPE, {}) as directory:
                (directory / 'ledger-eval2026-0.jsonl').write_bytes(b'{"partial": ')  # not even valid JSON
                raise RuntimeError('boom')
        record = json.loads(next(out.glob('failure-*.json')).read_text())
        self.assertEqual(record['partial_ledger_sha256'],
                         {'ledger-eval2026-0.jsonl': hashlib.sha256(b'{"partial": ').hexdigest()})
        self.assertFalse((out / 'manifest.json').exists())

    def test_one_shot_guard_counts_every_attempt_directory(self):
        """S-m4: a directory without started.json is an attempt; a corrupt started.json is a clear
        integrity failure; a sealed attempt ends the registration."""
        root = Path(tempfile.mkdtemp())
        self.assertEqual(rpv.one_shot_guard(root / 'absent', 2), [])
        (root / 'a1').mkdir()
        (root / 'a1' / 'started.json').write_text(json.dumps({'command': rpv.OPE}))
        (root / 'a2').mkdir()  # crashed before started.json was written
        (root / 'note.txt').write_text('not an attempt')
        self.assertEqual(len(rpv.one_shot_guard(root, 3)), 2)
        with self.assertRaisesRegex(pa.IntegrityError, 'FAILED_INFRA: 2 failed'):
            rpv.one_shot_guard(root, 2)
        (root / 'a3').mkdir()
        (root / 'a3' / 'started.json').write_text('{"command": "ope-')
        with self.assertRaisesRegex(pa.IntegrityError, 'corrupt or partial started.json'):
            rpv.one_shot_guard(root, 5)
        (root / 'a3' / 'started.json').write_text(json.dumps({'command': rpv.OPE}))
        (root / 'a3' / 'manifest.json').write_text('{}')
        with self.assertRaisesRegex(pa.IntegrityError, 'already sealed'):
            rpv.one_shot_guard(root, 5)

    def test_vocabulary_mapping_to_missing_and_cap(self):
        """F-M1: codes outside the TRAIN vocabulary become missing labels (-> MISSING_ACTION_LABEL
        requests), counted and listed; above the registered cap the run is REFUSED_VOCABULARY."""
        frame = pd.DataFrame({'pitch_type': ['FF', 'ZZ', None, '', 'SL', 'ZZ', 'KN'],
                              'description': ['ball', 'ball', 'ball', 'ball', 'ball', 'automatic_ball', 'ball']})
        mapped, record = rpv.map_vocabulary(frame, TYPES, .6)
        self.assertEqual((record['codes'], record['rows'], record['labelled_rows']), ({'KN': 1, 'ZZ': 2}, 3, 5))
        self.assertAlmostEqual(record['share'], .6)
        labels = [preq.logged_label(t, d, NO_PITCH) for t, d in zip(mapped.pitch_type, mapped.description)]
        self.assertEqual(labels, ['FF', pr.MISSING, pr.MISSING, pr.MISSING, 'SL', pr.NO_PITCH, pr.MISSING])
        self.assertEqual(frame.pitch_type.tolist()[1], 'ZZ')  # the input frame is not modified
        with self.assertRaisesRegex(pa.IntegrityError, 'REFUSED_VOCABULARY'):
            rpv.map_vocabulary(frame, TYPES, .5)

    def test_manifest_skips_appledouble_files(self):
        out = Path(tempfile.mkdtemp()) / 'stage'
        with rpv.stage(out, 'census', {}) as directory:
            (directory / 'census.json').write_text('{}')
            (directory / '._census.json').write_bytes(b'\x00\x05\x16\x07' + b'\x00' * 28)  # AppleDouble magic
        self.assertEqual(sorted(json.loads((out / 'manifest.json').read_text())['artifact_sha256']),
                         ['census.json', 'started.json'])

class DispatchTests(RunnerFixture):
    """The registered chain through ``dispatch``: gates, addenda, sealed pins and every stage the
    synthetic fixture can exercise (bind-probe reads sealed G0 archives; its record is a labelled
    stub sealed in its own directory)."""
    def config(self):
        runs = self.root / 'artifact' / 'runs' / 'ML-MATRIX-20260924' / 'VAL'
        guard = {'hang_guard_seconds': 600}
        stages = {
            'S0_census': {**guard, 'outcome_adjacent_splits': ['train']}, 'S1_materialize': {**guard, 'gates': self.gates},
            'S1b_style_snapshot': {**guard, 'as_of_exclusive': {'dev': '2025-07-01', 'temperature': '2025-05-16',
                                                                'blend': '2025-06-01'}},
            'S2_bind_probe': {**guard, 'rows': 4, 'atol_primary': 1e-6, 'atol_frequency_raw': 1e-12},
            'S3_profile': {'starts': 2, 'candidates': [{'samples': 3, 'pitch_cap': 4}, {'samples': 2, 'pitch_cap': 4}],
                           'row_budget': 10 ** 8, 'selection_row_budget': {'rows': 10 ** 8, 'decisions': 1000}},
            'S3b_tau_select': {'n_games': 2, 'tau_grid': [.01, .1], 'row_budget': 10 ** 8, 'determinism_check_pas': 1,
                               'samples_minimum_for_noise_rule': 3,
                               'thresholds': {'pa_ess_ratio_min': 1e-6, 'game_ess_min': 1., 'safety_multiplier': 1.,
                                              'ess_ratio_candidate_reference_min': 1e-6, 'noise_ratio_q90_max': 1e9}},
            'S4_V5_denominators': guard,
            'S5_V2_V3': {'n_games': 1, 'logs_per_start': 12, 'truth_rollouts': 12, 'cap': 4, 'row_budget_per_run': 10 ** 8,
                         'tempered_alpha_grid': [.5], 'tolerance': 1.},
            'S6_V4': {'n_games': 2, 'row_budget': 10 ** 8, 'dr_q_source': rpv.DR_Q_RULE, 'planned_decisions': 1000,
                      'd7_diagnostic_starts': 2}}
        return runs, {
            'protocol': rpv.PROTOCOL, 'registered': True, 'status': 'REGISTERED',
            'execution': {'enabled': True, 'real_data_enabled': True}, 'decisions': dict(rpv.DECISION_VALUES),
            'review_gates': {'code_review': {'status': 'PASS'}, 'independent_review': {'status': 'PASS'}},
            'seeds': {'base': 20260929, 'planning_main': 0, 'planning_v2': [0, 1]},
            'pa_time_rules': {'R3_codes': {'no_pitch_descriptions': sorted(NO_PITCH)},
                              'R7_mid_pa_change': {'thresholds_before_S0': {
                                  'switch_to_secondary_primary_if_unknown_change_share_above': .5,
                                  'light_version_if_below': .01}}},
            'train_bc_plan': {'bc_parameters': {'prior_strength': 20., 'minimum_action_count': 1},
                              'bc_p_only_pitcher_rule': 'evaluate; report as a stratum'},
            'subgroups': {'volume_quantiles': [1 / 3, 2 / 3]}, 'ess_gate': {'thresholds': {'pa': 100., 'game': 1.}},
            'sensitivity': {'same_ledger': ['r5-events-v1']},
            'identity_registration': {'classes': self.syn.classes, 'we_contract_sha256': self.syn.we_sha,
                                      'expected_identity_sha256': None,
                                      'g0_bundle': {'file_sha256': self.syn.bundle_file()[1]},
                                      'member_loader': {'file': 'x', 'sha256': 'y'}},
            'le2025_validation_plan': {'stages': stages, 'output_root': str(runs), 'local_config': {'sha256': 'z' * 64},
                                       'bootstrap': {'draws': 100, 'invalid_share_max': .5,
                                                     'minimum': {'games': 1, 'pa_starts': 1}}}}

    def test_registered_chain_through_dispatch(self):
        runs, config = self.config()
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(config))
        bundle_path, bundle_sha = self.syn.bundle_file()
        keys_path = self.root / 'train_keys.parquet'
        self.keys.to_parquet(keys_path, index=False)
        files = {**self.syn.bundle['files'], 'p4_train_keys': {'path': str(keys_path), 'sha256': sha(keys_path)}}
        inputs = {'bundle_path': bundle_path, 'bundle_sha': bundle_sha, 'files': files,
                  'paths': {**self.syn.paths, 'p4_train_keys': keys_path},
                  'prep': {'features': {'tokens': {'type_vocabulary': TYPES}},
                           'samples': {'train': {'n': len(self.keys), 'rows_sha256': ordered_key_hash(self.keys)}}},
                  'parent': {'dataset_identity': {'sources': [{'file': 'synthetic', 'sha256': 'a' * 64}],
                                                  'processed_sha256': 'f' * 64}},
                  'frame': self.frame, 'member_loader': load_member}
        loads, chain = [], [config_path]

        def load(store):
            loads.append(store)
            return {**inputs, 'store': self.store if store else None}

        def run(command, name):
            rpv.dispatch(command, rpv.load_registration(chain[0], chain[1:]), {}, runs / name, load)
            return runs / name

        def register(stage, entries):
            path = self.root / f'addendum-{len(chain)}.json'
            path.write_text(json.dumps({'parent_sha256': rpv.hash_file_bytes(chain[-1].read_bytes()), 'stage': stage,
                                        'registered_inputs': {n: {'path': str(p), 'file_sha256': sha(p)}
                                                              for n, p in entries.items()}}))
            chain.append(path)
        runs.mkdir(parents=True)
        with mock.patch('run_ml_matrix.check_location', lambda local, output: self.root / 'artifact'):
            census = run('census', 'S0')
            self.assertEqual(loads, [False])  # census never builds the history store
            with self.assertRaisesRegex(pa.IntegrityError, 'registered input required: census'):
                run('materialize-bc', 'S1-too-early')
            self.assertEqual(loads, [False])  # a missing prerequisite refuses before any data load
            register('S0', {'census': census / 'census.json'})
            s1 = run('materialize-bc', 'S1')
            s1b = run('style-snapshot', 'S1b')
            stub = runs / 'S2-stub'  # bind-probe needs sealed G0 archives; its sealed record is a labelled stub
            stub.mkdir()
            bound = self.syn.bind(pa.load_train_bc(s1 / 'bc_BC_P.json', sha(s1 / 'bc_BC_P.json')),
                                  rpv.pa_contexts(self.store, blocks_of(self.frame, 'dev')[:1])[0])
            (stub / 'probe.json').write_text(json.dumps({'pass': True, 'synthetic_stub': True}))
            (stub / 'identity.json').write_text(json.dumps({'sha256': bound.sha256}))
            (stub / 'manifest.json').write_text(json.dumps({'command': 'bind-probe', 'artifact_sha256': {
                'probe.json': sha(stub / 'probe.json'), 'identity.json': sha(stub / 'identity.json')}}))
            register('S1-S2', {'bc': s1 / 'bc_BC_P.json', 'hands': s1 / 'hands.json', 'support': s1 / 'support_primary.json',
                               'materialize': s1 / 'materialize.json',
                               **{f'style_{s}': s1b / f'style_{s}.json' for s in ('dev', 'temperature', 'blend')},
                               'bind_probe': stub / 'probe.json', 'bind_identity': stub / 'identity.json'})
            v5 = run('v5-denominators', 'S4')
            self.assertEqual(json.loads((v5 / 'v5.json').read_text())['sealed_counts']['pas'], 16)
            profile = run('profile', 'S3')
            measured = json.loads((profile / 'profile.json').read_text())
            self.assertEqual((len(measured['profiles']), measured['selection']), (2, {'samples': 3, 'pitch_cap': 4}))
            self.assertTrue(all(p['evaluation_rows'] > 0 for p in measured['profiles']))  # M-7 cost is measured
            register('S3-S4', {'profile': profile / 'profile.json', 'v5': v5 / 'v5.json'})
            freeze = run('tau-select', 'S3b')
            frozen = json.loads((freeze / 'tau_freeze.json').read_text())
            self.assertEqual((frozen['status'], frozen['outcomes_read'], frozen['determinism_check']['max_abs_q_difference'],
                              frozen['dr_q']['source']), ('SELECTED', False, 0., 'evaluation_seed'))
            register('S3b', {'tau_freeze': freeze / 'tau_freeze.json'})
            v2 = run('v2-world', 'S5')
            report = json.loads((v2 / 'v2.json').read_text())
            self.assertEqual(len(report['runs']), 2 * 3)  # 2 planning seeds x (V2, frequency, tempered)
            self.assertTrue(report['accept'])  # synthetic tolerance 1.0: the chain continues to S6
            register('S5', {'v2': v2 / 'v2.json'})
            dr = run('dr-evaluate', 'S6')
            result = json.loads((dr / 'dr.json').read_text())
            self.assertEqual((result['layers']['L0_all_pas']['pas'], result['sealed_counts']['pas']), (16, 16))
            self.assertEqual((result['identity_sha256'], result['paired_identity_run']['pass']),
                             (frozen['final_identity_sha256'], True))
            self.assertEqual(result['provenance']['as_of_exclusive'], '2025-07-01')
            self.assertEqual((result['d7_label_blind_diagnostic']['starts'], result['dr_q']['source']), (2, 'evaluation_seed'))
            self.assertIn('bc_p_only_pitcher', result['strata'])
            started = json.loads((dr / 'started.json').read_text())
            self.assertEqual(len(started['registration_chain']), len(chain))
            self.assertIn('experiments/pitchmdp/pitchmdp/policy_tau.py', started['runner_sources'])
            self.assertIn('stratum_role', pd.read_parquet(dr / 'pa_values.parquet').columns)
            with self.assertRaisesRegex(pa.IntegrityError, 'stage output exists'):  # attempts are never overwritten
                run('census', 'S0')
            with self.assertRaisesRegex(pa.IntegrityError, 'directly under'):
                rpv.dispatch('census', rpv.load_registration(chain[0], chain[1:]), {}, runs / 'S0' / 'nested', load)
            failed = runs / 'S1-failed'
            failed.mkdir()
            (failed / 'bc.json').write_text('{}')  # written before a failure: no sealed manifest
            register('bad', {'bad_bc': failed / 'bc.json'})
            with self.assertRaisesRegex(pa.IntegrityError, 'not inside a sealed stage'):
                rpv.registered_path(rpv.load_registration(chain[0], chain[1:]), 'bad_bc')
            wrong = {**rpv.load_registration(chain[0], chain[1:]), 'inputs': {'v5': {'path': str(census / 'census.json'),
                                                                                    'file_sha256': sha(census / 'census.json')}}}
            with self.assertRaisesRegex(pa.IntegrityError, 'does not come from a sealed v5-denominators stage'):
                rpv.registered_path(wrong, 'v5')
            # COOP-021/022: a new config (runtime v3 + L1-R estimator block, the 2026 OPE block) re-pins the
            # same addenda; the <=2025 S6 rehearsal runs as a new attempt, then the single 2026 OPE on a
            # SYNTHETIC holdout snapshot (dates moved to 2026; no real 2026 row is ever read in tests).
            estimator = {'runtime': pr.CONTRACT_V3, 'censoring': 'l1r', 'natural_course_refusals': list(est.H_K_REFUSALS),
                         'gr_cap': .5, 'c_deltas': [.008, .0278], 'mei': .001, 'v3_law': 'tempered_alpha_0.5'}
            raw = self.frame.loc[self.frame.split.eq('dev')].copy()
            raw['game_date'] = raw.game_date.str.replace('2025-07', '2026-04')
            raw = raw.assign(game_type='R', on_1b=np.nan, on_2b=np.nan, on_3b=np.nan)[list(rpv.HOLDOUT_COLUMNS)]
            extra = raw.loc[raw.game_pk.eq(201)].assign(game_type='S')  # spring training rows never enter
            new_code = raw.index[raw.game_pk.eq(200) & raw.at_bat_number.eq(1) & raw.pitch_number.eq(2)]
            raw.loc[new_code, 'pitch_type'] = 'ZZ'  # F-M1: a 2026 code outside the TRAIN vocabulary
            holdout_path = self.root / 'statcast_2026.parquet'
            pd.concat([raw, extra.assign(game_pk=299)]).to_parquet(holdout_path, index=False)
            b_v = est.v2_bias_record(report, .001)['b_v']
            ope_root = self.root / 'artifact' / 'runs' / 'OPE-2026'
            config2 = copy.deepcopy(config)
            config2['le2025_validation_plan']['stages']['S6_V4']['estimator'] = estimator
            config2['mlb2026_ope'] = {
                'registered': True, 'status': 'REGISTERED', 'review_gate': {'status': 'PASS'}, 'output_root': str(ope_root),
                'snapshot': {'version': rpv.HOLDOUT_VERSION, 'sha256': sha(holdout_path), 'path': str(holdout_path)},
                'season': list(rpv.HOLDOUT_SEASON), 'row_budget': 10 ** 8, 'b_v': b_v, 'max_attempts': 2,
                'hang_guard_multiplier': 1000, 'candidate_identity_sha256': frozen['final_identity_sha256'],
                'estimator': estimator, 'vocab_unmapped_share_max': .2}
            config2_path = self.root / 'config2.json'
            config2_path.write_text(json.dumps(config2))
            old, chain[:] = list(chain), [config2_path]
            for path in old[1:]:  # same pins, new parent hashes
                addendum = json.loads(path.read_text())
                addendum['parent_sha256'] = rpv.hash_file_bytes(chain[-1].read_bytes())
                new = path.with_name('re-' + path.name)
                new.write_text(json.dumps(addendum))
                chain.append(new)
            ope_root.mkdir()

            def ope(name):
                with mock.patch.object(rpv, 'HOLDOUT_SHA256', sha(holdout_path)):
                    rpv.dispatch(rpv.OPE, rpv.load_registration(chain[0], chain[1:]), {}, ope_root / name, load)
                return ope_root / name
            with self.assertRaisesRegex(pa.IntegrityError, 'registered input required: s6_dr'):
                ope('OPE-early')
            self.assertFalse((ope_root / 'OPE-early').exists())  # a missing prerequisite never costs an attempt
            s6v3 = run('dr-evaluate', 'S6-v3')
            rehearsal = json.loads((s6v3 / 'dr.json').read_text())
            self.assertEqual(rehearsal['censoring_rule']['rule'], 'L1-R')
            self.assertEqual(set(rehearsal['censoring_sensitivity']), {'S-v1', 'S-NP', 'S-B', 'S-C:0.008', 'S-C:0.0278'})
            self.assertEqual(rehearsal['censoring_sensitivity']['S-v1']['layers'], result['layers'])  # D-5 reproduced
            self.assertIn(rehearsal['rehearsal_labels']['primary']['label'], est.LABELS)
            self.assertEqual(rehearsal['ledger']['runtime_sha256'], json.loads(
                (s6v3 / 'ledger-dev-0.jsonl').read_text().splitlines()[0])['runtime_sha256'])
            register('S6-v3', {'s6_dr': s6v3 / 'dr.json'})
            registered = rpv.load_registration(chain[0], chain[1:])
            self.assertEqual(rpv.s6_estimator_block(rehearsal), estimator)  # S-m8: read back from the sealed S6
            for key, value in (('gr_cap', .4), ('c_deltas', [.008]), ('mei', .002), ('v3_law', 'frequency')):
                other = copy.deepcopy(registered)
                other['config']['mlb2026_ope']['estimator'][key] = value
                with self.subTest(key), self.assertRaisesRegex(pa.IntegrityError, 'S6 rehearsal estimator block differs'):
                    rpv.prerequisites(rpv.OPE, other)
            with mock.patch.object(rpv, 'HOLDOUT_SHA256', 'f' * 64), \
                    self.assertRaisesRegex(pa.IntegrityError, 'only the frozen holdout snapshot'):
                rpv.dispatch(rpv.OPE, registered, {}, ope_root / 'x', load)
            # S-M2: every outcome-free fatal check of the snapshot runs before the attempt directory exists.
            flooded = lambda snapshot: raw.assign(pitch_type='ZZ')
            duplicated = lambda snapshot: pd.concat([raw, raw.iloc[:1]])
            late = lambda snapshot: raw.assign(game_date='2026-10-02')
            for name, reader, message in (('v', flooded, 'REFUSED_VOCABULARY'), ('d', duplicated, 'duplicate pitch keys'),
                                          ('s', late, 'outside the registered 2026 season')):
                with self.subTest(name), mock.patch.object(rpv, 'HOLDOUT_SHA256', sha(holdout_path)), \
                        self.assertRaisesRegex(pa.IntegrityError, message):
                    rpv.dispatch(rpv.OPE, registered, {}, ope_root / f'refused-{name}', load, load_holdout=reader)
                self.assertFalse((ope_root / f'refused-{name}').exists())
            # S-m3: the one-shot guard is checked again under the heavy lock, right before the attempt directory.
            with mock.patch.object(rpv, 'one_shot_guard', side_effect=[[], pa.IntegrityError('FAILED_INFRA: raced')]), \
                    mock.patch.object(rpv, 'HOLDOUT_SHA256', sha(holdout_path)), \
                    self.assertRaisesRegex(pa.IntegrityError, 'raced'):
                rpv.dispatch(rpv.OPE, registered, {}, ope_root / 'raced', load)
            self.assertFalse((ope_root / 'raced').exists())
            # S-m4: a real failed dispatch (data load fails inside the attempt) costs one attempt.
            def broken(store):
                raise OSError('volume went away')
            with mock.patch.object(rpv, 'HOLDOUT_SHA256', sha(holdout_path)), self.assertRaisesRegex(OSError, 'volume'):
                rpv.dispatch(rpv.OPE, registered, {}, ope_root / 'OPE-a0', broken)
            self.assertTrue(list((ope_root / 'OPE-a0').glob('failure-*.json')))
            self.assertEqual(rpv.one_shot_guard(ope_root, 2), [str(ope_root / 'OPE-a0')])
            done = ope('OPE-a1')
            record = json.loads((done / 'ope2026.json').read_text())
            self.assertEqual(record['prior_attempts'], [str(ope_root / 'OPE-a0')])
            self.assertEqual((record['vocabulary_mapping']['codes'], record['vocabulary_mapping']['rows']), ({'ZZ': 1}, 1))
            # the mapped pitch is a MISSING_ACTION_LABEL refusal (worst-case residual, in G-R) beside the two
            # synthetic 'missing' PAs of the fixture
            self.assertEqual(record['ledger']['request_status'].get(pr.MISSING_LABEL), 3)
            self.assertIn(f'REFUSED:{pr.MISSING_LABEL}', record['censoring_rule']['worst_case_residual_reasons'])
            self.assertIn('positivity_candidate_searches', record['ledger'])  # S-M1 cost is reported
            s6_seconds = json.loads((s6v3 / 'manifest.json').read_text())['cost']['wall_seconds']
            self.assertAlmostEqual(record['hang_guard_seconds'], s6_seconds * 2 / len(rehearsal['games']) * 1000)
            self.assertEqual((record['identity_sha256'], record['games'], record['paired_identity_run']['pass']),
                             (frozen['final_identity_sha256'], 2, True))
            self.assertEqual(record['provenance']['as_of_exclusive'], '2026-03-25')
            self.assertEqual(record['provenance']['end_of_history_check']['mismatches'], 0)
            self.assertIn(record['label'], est.LABELS)
            self.assertIn('passable_effect_approx', record['labels']['primary'])
            self.assertIsNone(record['labels']['v3']['v3_flag'] if record['labels']['v3']['v3_max_abs_gap'] <= .001 else None)
            self.assertEqual(record['layers']['L0_all_pas']['pas'], 16)  # spring-training game 299 never entered
            with self.assertRaisesRegex(pa.IntegrityError, 'already sealed'):
                ope('OPE-a2')

    def test_registration_refusals(self):
        repo_config = json.loads((REPO / 'configs/ML-POLICY-MATERIALIZATION-v1.json').read_text())
        with self.assertRaisesRegex(pa.IntegrityError, 'not registered'):
            rpv.registration({**repo_config, 'registered': False})
        rpv.registration(repo_config, 'census')  # the committed config is registered (D96)
        closed = copy.deepcopy(repo_config)  # independent of the committed gate state (D102)
        closed['review_gates']['independent_review']['status'] = None
        with self.assertRaisesRegex(pa.IntegrityError, 'independent review'):
            rpv.registration(closed, 'profile')
        _, config = self.config()
        rpv.registration(config, 'census')
        with self.assertRaisesRegex(pa.IntegrityError, 'registered decision required: D-11'):
            rpv.registration({**config, 'decisions': {**config['decisions'], 'D-11': 'C'}})
        closed = {**config, 'review_gates': {'code_review': {'status': 'PASS'}, 'independent_review': {'status': None}}}
        rpv.registration(closed, 'bind-probe')
        with self.assertRaisesRegex(pa.IntegrityError, 'independent review'):
            rpv.registration(closed, 'v5-denominators')
        for dotted, command in (('le2025_validation_plan.stages.S6_V4.n_games', 'dr-evaluate'),
                                ('le2025_validation_plan.bootstrap.invalid_share_max', 'dr-evaluate'),
                                ('le2025_validation_plan.stages.S5_V2_V3.tolerance', 'v2-world'),
                                ('le2025_validation_plan.stages.S0_census.hang_guard_seconds', 'census'),
                                ('pa_time_rules.R7_mid_pa_change.thresholds_before_S0.light_version_if_below', 'census'),
                                ('le2025_validation_plan.stages.S3_profile.candidates', 'profile')):
            unset = copy.deepcopy(config)
            *path, last = dotted.split('.')
            rpv._field(unset, '.'.join(path))[last] = None
            with self.subTest(dotted), self.assertRaisesRegex(pa.IntegrityError, dotted.replace('.', r'\.')):
                rpv.registration(unset, command)
        bad = copy.deepcopy(config); bad['le2025_validation_plan']['stages']['S6_V4']['dr_q_source'] = 'other'
        with self.assertRaisesRegex(pa.IntegrityError, 'dr_q_source'):
            rpv.registration(bad, 'dr-evaluate')
        bad = copy.deepcopy(config); bad['ess_gate']['thresholds']['game'] = 30.
        with self.assertRaisesRegex(pa.IntegrityError, 'same game gate|ESS gate'):
            rpv.registration(bad, 'tau-select')
        bad = copy.deepcopy(config); bad['le2025_validation_plan']['stages']['S3_profile']['selection_row_budget']['rows'] = 1
        with self.assertRaisesRegex(pa.IntegrityError, 'S3b row budget'):
            rpv.registration(bad, 'profile')
        with self.assertRaisesRegex(pa.IntegrityError, 'unknown command'):
            rpv.registration(config, 'fit-everything')
        with self.assertRaisesRegex(pa.IntegrityError, 'mlb2026_ope'):  # the 2026 OPE stays closed until registered
            rpv.registration(config, rpv.OPE)
        with self.assertRaisesRegex(pa.IntegrityError, 'mlb2026_ope'):
            rpv.registration(repo_config, rpv.OPE)
        estimator = {'runtime': pr.CONTRACT_V3, 'censoring': 'l1r', 'natural_course_refusals': list(est.H_K_REFUSALS),
                     'gr_cap': .005, 'c_deltas': [.008], 'mei': .001, 'v3_law': 'tempered_alpha_0.5'}
        ope = {'registered': True, 'status': 'REGISTERED', 'review_gate': {'status': 'PASS'}, 'output_root': '/elsewhere',
               'snapshot': {'version': rpv.HOLDOUT_VERSION, 'sha256': rpv.HOLDOUT_SHA256, 'path': '/x.parquet'},
               'season': list(rpv.HOLDOUT_SEASON), 'row_budget': 1, 'b_v': .0027, 'max_attempts': 2,
               'hang_guard_multiplier': 2, 'candidate_identity_sha256': 'a' * 64, 'estimator': estimator,
               'vocab_unmapped_share_max': .001}
        rpv.registration({**config, 'mlb2026_ope': ope}, rpv.OPE)
        for change, message in (({'snapshot': {**ope['snapshot'], 'sha256': 'b' * 64}}, 'only the frozen holdout'),
                                ({'snapshot': {**ope['snapshot'], 'version': 'd20260911-s2325'}}, 'only the frozen holdout'),
                                ({'registered': False}, 'not registered'), ({'review_gate': {'status': None}}, 'review gate'),
                                ({'max_attempts': 3}, 'max_attempts'), ({'season': ['2026-03-25', '2026-10-31']}, 'season'),
                                ({'output_root': config['le2025_validation_plan']['output_root']}, 'must differ'),
                                ({'estimator': {**estimator, 'runtime': pr.CONTRACT}}, 'runtime v3'),
                                ({'estimator': {**estimator, 'natural_course_refusals': []}}, 'H_K_REFUSALS'),
                                ({'vocab_unmapped_share_max': None}, 'vocab_unmapped_share_max'),
                                ({'vocab_unmapped_share_max': 1.5}, 'vocab_unmapped_share_max'),
                                ({'hang_guard_seconds': 600}, 'must not register hang_guard_seconds')):
            with self.subTest(change), self.assertRaisesRegex(pa.IntegrityError, message):
                rpv.registration({**config, 'mlb2026_ope': {**ope, **change}}, rpv.OPE)
        with self.assertRaisesRegex(pa.IntegrityError, 'estimator block'):
            bad = copy.deepcopy(config); bad['le2025_validation_plan']['stages']['S6_V4']['estimator'] = {'runtime': 'v3'}
            rpv.registration(bad, 'dr-evaluate')
        path = self.root / 'c.json'
        path.write_text(json.dumps(config))
        broken = self.root / 'a.json'
        broken.write_text(json.dumps({'parent_sha256': '0' * 64, 'registered_inputs': {}}))
        with self.assertRaisesRegex(pa.IntegrityError, 'next link'):
            rpv.load_registration(path, [broken])
        first = self.root / 'a1.json'
        first.write_text(json.dumps({'parent_sha256': sha(path), 'registered_inputs': {'bc': {'path': 'x', 'file_sha256': 'y'}}}))
        second = self.root / 'a2.json'
        second.write_text(json.dumps({'parent_sha256': sha(first), 'registered_inputs': {'bc': {'path': 'z', 'file_sha256': 'w'}}}))
        with self.assertRaisesRegex(pa.IntegrityError, 'only add new'):
            rpv.load_registration(path, [first, second])
        extra = self.root / 'a3.json'
        extra.write_text(json.dumps({'parent_sha256': sha(path), 'tau': .5}))  # thresholds/values are never addenda
        with self.assertRaisesRegex(pa.IntegrityError, 'next link'):
            rpv.load_registration(path, [extra])
        loader = 'experiments/pitchmdp/scripts/run_ml_g0_whole.py'
        config['le2025_validation_plan']['source_commit'] = 'abc'
        config['identity_registration']['member_loader'] = {'file': loader, 'sha256': sha(REPO / loader)}
        clean = {'commit': 'def', 'code_dirty': False, 'source_is_ancestor': True, 'code_changed_since_source': [],
                 'registration_uncommitted': []}
        rpv.enforce_source(config, clean)  # registration commits on top of the source commit are fine
        with self.assertRaisesRegex(pa.IntegrityError, 'uncommitted'):
            rpv.enforce_source(config, {**clean, 'code_dirty': True})
        with self.assertRaisesRegex(pa.IntegrityError, 'code at HEAD differs'):
            rpv.enforce_source(config, {**clean, 'code_changed_since_source': ['experiments/pitchmdp/x.py']})
        with self.assertRaisesRegex(pa.IntegrityError, 'code at HEAD differs'):
            rpv.enforce_source(config, {**clean, 'source_is_ancestor': False})
        with self.assertRaisesRegex(pa.IntegrityError, 'not committed at HEAD'):
            rpv.enforce_source(config, {**clean, 'registration_uncommitted': ['configs/x.json']})
        self.assertTrue(rpv.committed(REPO / 'configs/G0-RESEARCH-FROZEN-v1.json'))
        self.assertFalse(rpv.committed(path))  # a config outside the repository is never a registration


# ---------------------------------------------------------------- V2/V3 world (D-10)

class SemiSyntheticTests(RunnerFixture):
    """V2/V3 in the declared world W (fake G0 + WE, off-mask CH absorbed with rho = 0)."""
    def world(self, name, blocks=('200:1', '201:1')):
        out, report = self.materialize()
        dev = [b for b in blocks_of(self.frame, 'dev') if b[0] in blocks]
        runtime, components = self.candidate(self.pins(out, report), dev, f'{name}.jsonl', loader=typed_loader, tau=.003,
                                             samples=16)
        starts = []
        for _, positions in dev:
            request = preq.pa_requests(self.store, positions, runtime.sha256)[0][0]
            starts.append((request.state, request.pitcher_hand))
        return runtime, components, starts

    def run_v2(self, name, **extra):
        runtime, components, starts = self.world(name)
        law = lambda state: (runtime.bc.actions, runtime.bc.probabilities(state))
        settings = dict(law=law, law_identity='pi_b_hat', logs_per_start=150, cap=4, truth_rollouts=300, seed=21,
                        draws=100, budget=RowBudget(10 ** 8, seed_count=5), tolerance=.05)
        return pss.run_world(runtime, components, starts, **{**settings, **extra})

    def test_v2_matches_world_truth_and_is_not_vacuous(self):
        report = self.run_v2('v2')
        self.assertTrue(report['interval_contains_zero'] and report['accept'], report['delta_gap_ci95'])
        self.assertGreater(report['ends']['absorbed'], 0)  # logged CH outside the mask ended its PA with rho = 0
        for row in report['per_start']:
            for name in ('candidate', 'reference'):
                se = np.hypot(row['dr_se'][name], row['truth'][f'{name}_mc_se'])
                self.assertLess(abs(row['dr_mean'][name] - row['truth'][name]), 4 * se, (name, row))
        truth_delta = np.mean([row['truth']['delta'] for row in report['per_start']])
        # Non-vacuous: the policies differ by far more than the gap's s.e., so an estimator that swaps
        # candidate and reference or ignores the policies (delta = 0) falls outside the interval.
        self.assertGreater(abs(truth_delta), 4 * report['delta_gap_se'])
        self.assertIsNone(report['estimate']['causal_effect'])

    def test_absorbing_value_is_irrelevant_and_v3_laws(self):
        zero = self.run_v2('abs0', absorbing_value=0., logs_per_start=40, truth_rollouts=20)
        one = self.run_v2('abs1', absorbing_value=1., logs_per_start=40, truth_rollouts=20)
        self.assertGreater(zero['ends']['absorbed'], 0)
        for layer in ('L0_all_pas', 'L1_start_population'):  # bit-identical DR values
            self.assertEqual(zero['estimate']['layers'][layer], one['estimate']['layers'][layer])
        pick = lambda r: {k: v for k, v in r['estimate']['layers']['L2_complete_conditional'].items()
                          if k != 'observed_mean_end_value'}  # the raw end value naturally moves
        self.assertEqual(pick(zero), pick(one))
        self.assertEqual([r['delta_dr'] for r in zero['per_start']], [r['delta_dr'] for r in one['per_start']])
        for law_name, make in (('frequency', pss.frequency_law), ('tempered', lambda bc: pss.tempered_law(bc, .5))):
            runtime, components, starts = self.world(f'v3-{law_name}')  # one ledger per generating law
            report = pss.run_world(runtime, components, starts, law=make(runtime.bc), law_identity=law_name,
                                   logs_per_start=20, cap=4,
                                   truth_rollouts=10, seed=9 + len(law_name), draws=20,
                                   budget=RowBudget(10 ** 8, seed_count=5))
            self.assertIsNone(report['accept'])  # V3 is a transport report, not a pass/fail
        runtime, components, starts = self.world('v3bad')
        broken = lambda s: (runtime.bc.actions, np.array([.5, .5, .5]))  # not a probability vector
        with self.assertRaisesRegex(pa.IntegrityError, 'probability vector'):
            pss.run_world(runtime, components, starts, law=broken, law_identity='broken', logs_per_start=2, cap=4,
                          truth_rollouts=2, seed=1, draws=5, budget=RowBudget(10 ** 8, seed_count=5))

    def test_acceptance_guards_and_off_mask_continuation(self):
        strict = self.run_v2('tight', logs_per_start=40, truth_rollouts=40, tolerance=1e-9)
        self.assertFalse(strict['accept'])  # |gap| above a registered tolerance is a rejection, not a warning
        runtime, components, starts = self.world('guard')
        stub = SimpleNamespace(bc=SimpleNamespace(actions=runtime.bc.actions, probabilities=lambda s: np.array([0., .5, .5])))
        with self.assertRaisesRegex(pa.IntegrityError, 'pi_b_hat > 0'):
            pss._law(lambda s: (runtime.bc.actions, np.array([.2, .4, .4])), stub, starts[0][0])
        # D-10: rows logged after an off-mask action (rho = 0) change nothing, so absorbing there is exact.
        start, hand = starts[0]
        ch = runtime.submit(pr.DecisionRequest('c:0', 'c', 0, start, 'CH', runtime.sha256, hand))
        self.assertEqual((ch['status'], ch['result']['rho_candidate'], ch['result']['rho_reference']),
                         (pr.OUTSIDE_POLICY_SUPPORT, 0., 0.))
        after = PAState(start.balls + 1, start.strikes, start.pitcher, start.batter_side,
                        (PastPitch('CH', (0.,) * 8, 'ball', start.balls, start.strikes),), start.context_key)
        later = runtime.submit(pr.DecisionRequest('c:1', 'c', 1, after, 'FF', runtime.sha256, hand))
        self.assertIn(later['status'], pr.EVALUATED)
        for reward in (0., .37, 1.):
            full, absorbed = est.pa_dr([ch, later], reward), est.pa_dr([ch], .5)
            self.assertEqual((full['candidate'], full['reference']), (absorbed['candidate'], absorbed['reference']))

    def test_declared_censoring_hazard_bounds_contain_the_truth(self):
        runtime, components, starts = self.world('hazard')
        law = lambda state: (runtime.bc.actions, runtime.bc.probabilities(state))
        hazard = lambda state, action: .6 if action == 'SL' else 0.  # depends on the logged action (D-5 C)
        report = pss.run_world(runtime, components, starts, law=law, law_identity='pi_b_hat', logs_per_start=150, cap=4,
                               truth_rollouts=300, seed=21, draws=50, censor_hazard=hazard)
        self.assertGreater(len(report['censored_pas']), 0)
        lo, hi = report['l1_delta_bounds']
        self.assertTrue(lo <= report['truth_delta_mean'] <= hi and report['truth_inside_l1_bounds'])
        by_pa = {}
        for row in runtime.ledger.decisions():
            by_pa.setdefault(row['pa_id'], []).append(row)
        for pa_id in report['censored_pas']:  # the exact NO_TERMINAL width on real runtime rows, last decision included
            rows = sorted(by_pa[pa_id], key=lambda r: r['decision_index'])
            value = est.pa_value(rows, {'game': 1, 'in_population': True, 'reward': None, 'kind': est.NO_TERMINAL})
            width = value['delta_bounds'][1] - value['delta_bounds'][0]
            expected = sum(float(np.prod([r['result'][f'rho_{n}'] for r in rows])) for n in est.POLICIES)
            self.assertAlmostEqual(width, expected, places=9)  # not 0 (c = .5 imputation) nor the flat slot's 2

    def test_start_outside_e0_is_reported_not_generated(self):
        runtime, components, starts = self.world('e0', blocks=('200:1', '200:4'))  # 200:4 is an unknown pitcher
        report = pss.run_world(runtime, components, starts, law=lambda s: (runtime.bc.actions, runtime.bc.probabilities(s)),
                               law_identity='pi_b_hat', logs_per_start=10, cap=4, truth_rollouts=10, seed=3, draws=10,
                               budget=RowBudget(10 ** 8, seed_count=5))
        self.assertEqual(report['excluded_starts'], [{'start': 1, 'reason': pr.UNKNOWN_PITCHER}])
        self.assertEqual(len(report['per_start']), 1)


if __name__ == '__main__':
    unittest.main()
