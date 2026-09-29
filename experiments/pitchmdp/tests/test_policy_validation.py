"""SYNTHETIC-ONLY tests for the COOP-018 request generator, runtime refusals, DR estimator and
<=2025 validation runner stages. Tiny synthetic games; fake pinned G0/WE files from
test_policy_identity; the D89 enumerated toy PA as the exact DR oracle. No real data, model
weights or 2026 access.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'scripts'))
from test_policy_identity import CLUSTERS, PROV, TYPES, WE, SyntheticPolicy, load_member
from pitchmdp.archetypes import HISTORY_COLUMNS, add_batter_style_history
from pitchmdp.matrix_data import ordered_key_hash
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_policy import safe_rows
from pitchmdp.rollout_policy import BCRecord, CategoricalBC, PAState, PastPitch, RowBudget
from pitchmdp.sequence_data import PHYSICAL_COLUMNS, PhysicalNormalizer
from pitchmdp import policy_artifacts as pa
from pitchmdp import policy_estimator as est
from pitchmdp import policy_identity as pi
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as pr
from pitchmdp import policy_semisynthetic as pss
import run_policy_validation as rpv

spec = importlib.util.spec_from_file_location('policy_contract', REPO / 'scripts/check_2026_policy_contract.py')
TOY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TOY)

SEQUENCES = {  # (type, description, events on the last row)
    'short': [('FF', 'ball'), ('SL', 'called_strike'), ('FF', 'foul'), ('CH', 'hit_into_play')],
    'long': [('FF', 'ball'), ('SL', 'called_strike'), ('FF', 'foul'), ('FF', 'foul'), ('SL', 'foul'), ('FF', 'foul'),
             ('SL', 'hit_into_play')],
    'automatic': [('FF', 'ball'), (None, 'automatic_ball'), ('SL', 'called_strike'), ('FF', 'hit_into_play')],
}
STEP = {'ball': (1, 0), 'called_strike': (0, 1), 'automatic_ball': (1, 0), 'foul': (0, 1)}


def pa_rows(game, ab, date, split, kind, *, pitcher=9, event='single', drop_first=False, truncated=False, outs=0):
    rows, balls, strikes = [], 0, 0
    for index, (pitch_type, description) in enumerate(SEQUENCES[kind]):
        last = index == len(SEQUENCES[kind]) - 1
        rows.append(dict(game_pk=game, at_bat_number=ab, pitch_number=index + 1, game_date=date, pitcher=pitcher,
                         batter=2 + ab % 3, stand='L', p_throws='R', balls=balls, strikes=strikes,
                         inning=1 + ab // 6, inning_topbot='Top' if (ab // 3) % 2 == 0 else 'Bot', outs_when_up=outs,
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
            rows += pa_rows(100 + g, ab, date, 'train', ('short', 'long')[ab % 2], outs=ab % 3,
                            event=('single', 'field_out')[ab % 2])
    dev_script = [('short', {}), ('long', {}), ('automatic', {}), ('short', {'pitcher': 77}),
                  ('short', {'drop_first': True}), ('short', {'truncated': True}), ('long', {'event': 'field_out'})]
    for ab in range(1, 4):  # one May game (profile starts)
        rows += pa_rows(150, ab, '2025-05-20', 'temperature', 'short', outs=ab % 3)
    for g in range(2):  # DEV
        date = (pd.Timestamp('2025-07-02') + pd.Timedelta(days=g)).strftime('%Y-%m-%d')
        for ab, (kind, extra) in enumerate(dev_script, start=1):
            rows += pa_rows(200 + g, ab, date, 'dev', kind, outs=ab % 3, **extra)
    frame = pd.DataFrame(rows)
    for c in PHYSICAL_COLUMNS:
        frame[c] = rng.normal(size=len(frame))
    last = ~frame.duplicated(['game_pk', 'at_bat_number'], keep='last')
    frame['is_pa_terminal'] = last & frame.events.notna() & frame.events.ne('truncated_pa')
    for game, part in frame.groupby('game_pk'):  # home wins every game: final result is determined
        frame.loc[part.index[-1], ['post_home_score', 'post_away_score']] = (1, 0)
    return add_batter_style_history(frame)


def build_store(frame):
    normalizer = PhysicalNormalizer().fit(frame.loc[frame.split.eq('train')])
    return MatrixHistoryStore.from_frame(frame, normalizer, 5, type_vocabulary=TYPES)


def sha(path):
    return pa.hash_file(Path(path))


# ---------------------------------------------------------------- runtime refusals

class RuntimeRefusalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        bc = CategoricalBC().fit([BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'FF', 'SL', 'CH')])
        art = pa.save_train_bc(bc, self.root / 'bc.json', PROV)
        _, support = pa.save_support_table(art, [('9', 'L', np.array([False, True, True]))], self.root / 's.json')
        self.rt = pr.build_runtime(self.root / 'bc.json', art.file_sha256, self.root / 's.json', support, self.root / 'l.jsonl')

    def submit(self, rid, pa_id, index, state, logged):
        return self.rt.submit(pr.DecisionRequest(rid, pa_id, index, state, logged, self.rt.sha256))

    def past(self, action, outcome, balls=0, strikes=0):
        return PastPitch(action, (0.,) * 8, outcome, balls, strikes)

    def test_no_pitch_start_and_count_path_refusals(self):
        self.assertEqual(self.submit('a0', 'a', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        no_pitch = self.submit('a1', 'a', 1, PAState(1, 0, '9', 'L', (self.past('FF', 'ball'),)), pr.NO_PITCH)
        self.assertEqual(no_pitch['status'], pr.NO_LOGGED_ACTION)
        later = PAState(2, 0, '9', 'L', (self.past('FF', 'ball'), self.past(pr.NO_PITCH, 'unknown', 1, 0)))
        self.assertEqual(self.submit('a2', 'a', 2, later, 'SL')['status'], pr.MID_PA)  # sticky, chain still checked
        self.assertEqual(self.submit('b0', 'b', 0, PAState(1, 0, '9', 'L'), 'FF')['status'], pr.INCOMPLETE_START)
        self.assertEqual(self.submit('c0', 'c', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        jumped = PAState(0, 1, '9', 'L', (self.past('FF', 'ball'),))  # a ball cannot lead to 0-1
        self.assertEqual(self.submit('c1', 'c', 1, jumped, 'SL')['status'], pr.INCONSISTENT_HISTORY)
        self.assertEqual(self.submit('d0', 'd', 0, PAState(0, 0, '9', 'L'), 'FF')['status'], pr.SUPPORTED)
        ended = PAState(0, 0, '9', 'L', (self.past('FF', 'single'),))  # a pitch after the PA ended
        self.assertEqual(self.submit('d1', 'd', 1, ended, 'SL')['status'], pr.INCONSISTENT_HISTORY)
        summary = self.rt.summary()  # refusals after decision 0 roll up to MID_PA at the PA level
        self.assertEqual(summary['pa_status'], {pr.MID_PA: 3, pr.INCOMPLETE_START: 1})
        self.assertEqual({k: summary['request_status'][k] for k in (pr.NO_LOGGED_ACTION, pr.INCONSISTENT_HISTORY)},
                         {pr.NO_LOGGED_ACTION: 1, pr.INCONSISTENT_HISTORY: 2})
        with self.assertRaisesRegex(pa.IntegrityError, 'unknown logged action'):  # a real unknown code still halts
            self.submit('e0', 'e', 0, PAState(0, 0, '9', 'L'), 'ZZ')

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

    def blocks(self, split='dev'):
        return preq.pa_blocks(self.frame, np.flatnonzero(self.frame.split.eq(split).to_numpy()))

    def test_full_history_labels_and_structure(self):
        blocks = dict(self.blocks())
        long_requests, problem = preq.pa_requests(self.store, blocks['200:2'], 'sha')
        self.assertIsNone(problem)
        self.assertEqual([len(r.state.history) for r in long_requests], list(range(7)))  # beyond the 5-row cap
        self.assertEqual([r.decision_index for r in long_requests], list(range(7)))
        automatic, _ = preq.pa_requests(self.store, blocks['200:3'], 'sha')
        self.assertEqual([r.logged_action for r in automatic], ['FF', pr.NO_PITCH, 'SL', 'FF'])
        self.assertEqual(automatic[2].state.history[-1].action, pr.NO_PITCH)  # same label in later history
        self.assertEqual(automatic[2].state.history[-1].outcome, 'unknown')
        self.assertEqual(preq.logged_label(np.nan, 'ball'), pr.NO_PITCH)
        self.assertEqual(preq.pa_requests(self.store, blocks['200:2'][::-1], 'sha'), ([], 'ordering'))
        bad = self.frame.copy(); bad.loc[blocks['200:1'][1], 'balls'] = 4
        self.assertEqual(preq.pa_requests(build_store(bad), blocks['200:1'], 'sha'), ([], 'illegal_count'))

    def test_requests_ignore_post_decision_fields(self):
        """R4c: changing the current row's physics/outcome and every later row leaves earlier states unchanged."""
        position = dict(self.blocks())['200:2']
        base, _ = preq.pa_requests(self.store, position, 'sha')
        changed = self.frame.copy()
        k = position[3]
        changed.loc[position[3:], PHYSICAL_COLUMNS] += 5.
        changed.loc[position[3:], ['description', 'events', 'launch_angle']] = ['hit_into_play', 'home_run', 40.]
        changed.loc[position[4:], 'pitch_type'] = 'CH'
        other, _ = preq.pa_requests(build_store(changed), position, 'sha')
        for before, after in zip(base[:4], other[:4]):  # decisions up to and including row k
            self.assertEqual(before.state, after.state)
        self.assertEqual(base[3].logged_action, other[3].logged_action)  # row k's type unchanged here
        self.assertNotEqual(base[4].state, other[4].state)  # row k's outcome is history for row k+1
        del k

    def test_pa_outcome_rewards(self):
        blocks, we = dict(self.blocks()), WE()
        normal = preq.pa_outcome(self.frame, blocks['200:1'], lambda s, d: we.predict_defense(s, d))
        nxt = self.frame.iloc[blocks['200:2'][0]]
        self.assertEqual((normal['end'], normal['reward']), ('next_row_state', .5 + .05 * nxt.outs_when_up))
        self.assertEqual(preq.pa_outcome(self.frame, blocks['200:6'], we.predict_defense)['reason'], 'no_terminal_event')
        final = preq.pa_outcome(self.frame, blocks['200:7'], we.predict_defense)  # game's last PA, home won
        top = self.frame.iloc[blocks['200:7'][0]].inning_topbot == 'Top'
        self.assertEqual((final['end'], final['reward']), ('final_result', 1. if top else 0.))
        unfinished = self.frame.copy(); unfinished.loc[unfinished.game_pk.eq(200), 'complete_game'] = False
        self.assertEqual(preq.pa_outcome(unfinished, blocks['200:7'], we.predict_defense)['reason'], 'incomplete_game')
        flagged = self.frame.copy(); flagged.loc[blocks['200:1'][-1], 'post_home_score'] = 1
        self.assertEqual(preq.pa_outcome(flagged, blocks['200:1'], we.predict_defense)['flags'],
                         ['post_pitch_score_disagrees_next_row'])

    def test_style_snapshot_freezes_window(self):
        as_of = '2025-07-03'
        snapshot = preq.style_snapshot(self.frame, as_of)
        expected = add_batter_style_history(self.frame.loc[pd.to_datetime(self.frame.game_date) < as_of].copy())
        batter = int(self.frame.loc[pd.to_datetime(self.frame.game_date) >= as_of, 'batter'].iloc[0])
        # A batter's frozen value equals the rolling prior of a row on the as-of date, i.e. only earlier data.
        probe = self.frame.loc[pd.to_datetime(self.frame.game_date).eq(pd.Timestamp(as_of)) & self.frame.batter.eq(batter)]
        np.testing.assert_allclose(snapshot.loc[batter].to_numpy(float), probe[list(HISTORY_COLUMNS)].iloc[0].to_numpy(float))
        del expected
        # In-window events after as_of cannot move the snapshot.
        noisy = self.frame.copy()
        late = pd.to_datetime(noisy.game_date) >= as_of
        noisy.loc[late, 'events'] = 'home_run'
        pd.testing.assert_frame_equal(preq.style_snapshot(noisy, as_of), snapshot)
        applied = preq.apply_style_snapshot(safe_rows(self.frame), snapshot, as_of)
        late_rows = applied.loc[pd.to_datetime(applied.game_date) >= as_of]
        np.testing.assert_allclose(late_rows[list(HISTORY_COLUMNS)].to_numpy(float),
                                   snapshot.loc[late_rows.batter].to_numpy(float))

    def test_census_is_label_blind_structure(self):
        census = preq.census(self.frame, TYPES)
        dev = census['dev']
        self.assertEqual((dev['pas'], dev['games'], dev['missing_pitch_type_rows'], dev['automatic_call_rows']), (14, 2, 2, 2))
        self.assertEqual((dev['pa_first_row_not_0_0'], dev['pa_without_terminal_event'], dev['pitchers_unseen_in_train']),
                         (2, 2, 1))
        self.assertEqual(census['train_pitchers_with_two_hands'], 0)
        self.assertEqual(dev['codes_outside_vocabulary'], [])


# ---------------------------------------------------------------- DR estimator

def toy_rows(steps, candidate, reference, logging, q_hat, mask):
    rows = []
    for t, (history, action) in enumerate(steps):
        p_c, p_r, p_b = candidate(history), reference(history), logging(history)
        a = TOY.VOCAB.index(action)
        q = [q_hat(history, b) if mask[i] else None for i, b in enumerate(TOY.VOCAB)]
        rows.append({'pa_id': 'pa', 'decision_index': t, 'status': 'SUPPORTED' if mask[a] else 'OUTSIDE_POLICY_SUPPORT',
                     'result': {'mask': mask.tolist(), 'logging': p_b.tolist(), 'reference': p_r.tolist(),
                                'candidate': p_c.tolist(), 'logged_index': a, 'q_reference': q,
                                'rho_reference': float(p_r[a] / p_b[a]), 'rho_candidate': float(p_c[a] / p_b[a])}})
    return rows


class EstimatorTests(unittest.TestCase):
    def expected(self, candidate, reference, logging, q_hat, mask):
        total = {'candidate': 0., 'reference': 0.}
        for steps, reward, weight in TOY.trajectories(TOY.PI_B):
            values = est.pa_dr(toy_rows(steps, candidate, reference, logging, q_hat, mask), float(reward))
            for name in total:
                total[name] += weight * values[name]
        return total

    def test_exact_oracle_on_enumerated_toy(self):
        full = np.ones(3, bool)
        got = self.expected(TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, full)  # correct logging law, wrong q
        self.assertAlmostEqual(got['candidate'], TOY.value(TOY.PI_CAND), places=12)
        self.assertAlmostEqual(got['reference'], TOY.value(TOY.PI_REF), places=12)
        # Masked reference (CH unsupported): logged CH rows are OUTSIDE_POLICY_SUPPORT (rho=0), still exact.
        masked = self.expected(TOY.PI_REF_RESTRICTED, TOY.PI_REF_RESTRICTED, TOY.PI_B, TOY.wrong_q, TOY.POLICY_MASK)
        self.assertAlmostEqual(masked['reference'], TOY.value(TOY.PI_REF_RESTRICTED), places=12)
        # Same law in both slots: paired delta exactly zero on every path.
        for steps, reward, _ in TOY.trajectories(TOY.PI_B):
            self.assertEqual(est.pa_dr(toy_rows(steps, TOY.PI_REF, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, full), float(reward))['delta'], 0.)

    def test_ledger_values_are_revalidated(self):
        steps, reward, _ = next(path for path in TOY.trajectories(TOY.PI_B)  # positive candidate ratios
                                if all(TOY.PI_CAND(h)[TOY.VOCAB.index(a)] > 0 for h, a in path[0]))
        rows = toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, np.ones(3, bool))

        def refused(message, mutate, reward=float(reward)):
            bad = copy.deepcopy(rows); mutate(bad)
            with self.assertRaisesRegex(pa.IntegrityError, message):
                est.pa_dr(bad, reward)
        refused('recorded ratio differs', lambda r: r[0]['result'].update(rho_candidate=r[0]['result']['rho_candidate'] * 1.01))
        refused('exactly on the mask', lambda r: r[0]['result']['q_reference'].__setitem__(0, None))
        refused('terminal WE', lambda r: None, reward=1.2)
        refused('complete ordered PA', lambda r: r.pop(0))
        refused('only evaluated', lambda r: r[0].update(status=pr.MID_PA))
        refused('mass outside the policy mask', lambda r: r[0]['result'].update(mask=[False, True, True]))
        refused('status differs', lambda r: r[0].update(status='OUTSIDE_POLICY_SUPPORT'))

    def test_estimate_denominators_bounds_bootstrap_ess(self):
        positive = [path for path in TOY.trajectories(TOY.PI_B)
                    if all(TOY.PI_CAND(h)[TOY.VOCAB.index(a)] > 0 for h, a in path[0])]
        path_a, path_b = positive[:2]
        full = np.ones(3, bool)
        rows = []
        for pa_id, (steps, _, _) in (('g1:1', path_a), ('g2:1', path_b)):
            for row in toy_rows(steps, TOY.PI_CAND, TOY.PI_REF, TOY.PI_B, TOY.wrong_q, full):
                rows.append({**row, 'pa_id': pa_id})
        rows.append({'pa_id': 'g2:2', 'decision_index': 0, 'status': pr.UNKNOWN_PITCHER, 'result': None})
        rows.append({'pa_id': 'g1:3', 'decision_index': 0, **{k: v for k, v in rows[0].items() if k not in ('pa_id', 'decision_index')}})
        outcomes = {'g1:1': {'game': 1, 'reward': float(path_a[1])}, 'g2:1': {'game': 2, 'reward': float(path_b[1])},
                    'g2:2': {'game': 2, 'reward': .5}, 'g1:3': {'game': 1, 'reward': None, 'reason': 'no_terminal_event'}}
        result, per_pa = est.estimate(rows, outcomes, draws=500, seed=11)
        self.assertEqual(result['status'], {est.COMPLETE: 2, est.UNSUPPORTED: 1, est.INCOMPLETE_NO_TERMINAL: 1})
        deltas = [r['delta'] for r in per_pa if r['status'] == est.COMPLETE]
        self.assertAlmostEqual(result['conditional']['delta_mean'], np.mean(deltas), places=15)
        self.assertEqual(result['population_delta_bounds'], [(sum(deltas) - 2) / 4, (sum(deltas) + 2) / 4])
        again, _ = est.estimate(rows, outcomes, draws=500, seed=11)
        self.assertEqual(again['conditional']['bootstrap'], result['conditional']['bootstrap'])  # seeded
        self.assertIsNone(result['population_value'])
        weights = [r['weight_candidate'] for r in per_pa if r['status'] == est.COMPLETE]
        self.assertAlmostEqual(result['conditional']['ess']['candidate']['pa'], sum(weights) ** 2 / sum(w * w for w in weights))
        self.assertIsNone(est.effective_sample_size([0., 0.]))  # all-zero weights: ESS undefined, not 0/0
        self.assertEqual(est.effective_sample_size([1., 1., 2.], games=[1, 1, 2]), 2.)
        with self.assertRaisesRegex(pa.IntegrityError, 'exactly one outcome'):
            est.estimate(rows, {k: v for k, v in outcomes.items() if k != 'g1:1'}, draws=10, seed=1)
        self.assertIsNone(est.game_bootstrap([.1], [1], draws=10, seed=1)['ci95'])


# ---------------------------------------------------------------- runner stages

class RunnerFixture(unittest.TestCase):
    """Synthetic games, store and pinned synthetic G0/WE files (no tests of its own)."""
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.frame = game_frame()
        self.store = build_store(self.frame)
        self.syn = SyntheticPolicy(self.root / 'g0')

    def materialize(self, rule='BC_P', compare=None, two_hand_rule='refuse'):
        out = self.root / f'S1-{rule}-{compare}-{len(list(self.root.glob("S1-*")))}'
        out.mkdir()
        bundle_path, bundle_sha = self.syn.bundle_file()
        train = self.frame.loc[self.frame.split.eq('train')]
        keys = train.loc[train.events.ne('truncated_pa'), ['game_pk', 'at_bat_number', 'pitch_number']].reset_index(drop=True)
        provenance = {'source_ids': {'synthetic': 'a' * 64}, 'config_sha256': 'b' * 64, 'code_commit': 'c' * 40,
                      'data_version': 'synthetic'}
        report = rpv.run_materialize(self.frame, self.store, out, rule=rule, provenance=provenance,
                                     two_hand_rule=two_hand_rule, compare_rule=compare, train_keys=keys,
                                     keys_record={'n': len(keys), 'rows_sha256': ordered_key_hash(keys)},
                                     bind_inputs=lambda art, rows: pi.bind_policy_inputs(
                                         bundle_path, bundle_sha, self.syn.paths, bc_artifact=art, context_rows=rows,
                                         aux_classes=self.syn.classes['aux']))
        return out, report


class RunnerStageTests(RunnerFixture):
    def test_materialize_bc_support_and_mass_report(self):
        out, report = self.materialize(compare='BC_E')
        primary = report['bc']['BC_P']
        train = self.frame.loc[self.frame.split.eq('train')]
        self.assertEqual(primary['train_rows'], int(train.pitch_type.notna().sum()))  # no outcome/support filter
        self.assertEqual(primary['train_dates'], 'observed_by_export_train_bc')
        self.assertEqual(report['bc']['BC_E']['train_rows'], len(train))
        support = report['support_table']
        self.assertEqual((support['rows'], support['hand_conflict_pitchers']), (2, []))  # pitcher 9 x {L, R}
        table, _ = pa.load_support_table(out / 'support_primary.json', support['file_sha256'],
                                         pa.load_train_bc(out / 'bc_BC_P.json', primary['file_sha256']))
        self.assertEqual((table[('9', 'L')].tolist(), table[('9', 'R')].tolist()), ([False, True, True], [False] * 3))
        mass = report['logging_mass_on_mask']  # CH carries logging mass off the mask in every TRAIN cell
        self.assertEqual(mass['decision_share_mass_equal_one'], 0.)
        self.assertLess(mass['decision_weighted_quantiles']['0.95'], 1.)
        with self.assertRaisesRegex(pa.IntegrityError, 'ordered key identity'):
            rpv.bc_rows(self.frame, 'BC_E', self.frame.loc[:3, ['game_pk', 'at_bat_number', 'pitch_number']],
                        {'n': 4, 'rows_sha256': '0' * 64})

    def test_v5_and_dr_passes_end_to_end(self):
        out, report = self.materialize()
        bc_sha, support_sha = report['bc']['BC_P']['file_sha256'], report['support_table']['file_sha256']
        runtime = pr.build_runtime(out / 'bc_BC_P.json', bc_sha, out / 'support_primary.json', support_sha, self.root / 'v5.jsonl')
        v5 = rpv.run_v5(runtime, self.store, 'dev', rpv.Deadline(60))
        requests = v5['summary']['request_status']
        self.assertEqual((v5['pas_in_split'], v5['unsubmittable_pas']), (14, 0))
        for status in (pr.NO_LOGGED_ACTION, pr.UNKNOWN_PITCHER, pr.INCOMPLETE_START):
            self.assertEqual(requests[status], 2)
        self.assertGreater(requests[pr.OUTSIDE_POLICY_SUPPORT], 0)  # logged CH: valid rho=0
        self.assertEqual(v5['summary']['run_status'], 'OK')
        # Candidate runtime over the same DEV PAs, bound components, rewards and the DR estimator.
        bc = pa.load_train_bc(out / 'bc_BC_P.json', bc_sha)
        blocks = preq.pa_blocks(self.frame, np.flatnonzero(self.frame.split.eq('dev').to_numpy()))
        components = self.syn.bind(bc, rpv.pa_contexts(self.store, blocks))
        runtime = pr.build_runtime(out / 'bc_BC_P.json', bc_sha, out / 'support_primary.json', support_sha,
                                   self.root / 'dr.jsonl', components=components, budget=RowBudget(10 ** 7, seed_count=5),
                                   tau=.01, samples=2, pitch_cap=4, seed=701)
        result, rows = rpv.run_dr(runtime, self.store, components, 'dev', rpv.Deadline(120), draws=200, seed=5)
        self.assertEqual(result['status'], {est.COMPLETE: 6, est.UNSUPPORTED: 6, est.INCOMPLETE_NO_TERMINAL: 2})
        self.assertEqual(result['conditional']['bootstrap']['games'], 2)
        total = sum(r['delta'] for r in rows if r['status'] == est.COMPLETE)
        self.assertEqual(result['population_delta_bounds'], [(total - 8) / 14, (total + 8) / 14])
        self.assertIsNone(result['causal_effect'])
        self.assertTrue(all(np.isfinite(r['delta']) for r in rows if r['status'] == est.COMPLETE))
        # Rebuilding from the same ledger replays every request and reproduces the estimate exactly.
        again = pr.build_runtime(out / 'bc_BC_P.json', bc_sha, out / 'support_primary.json', support_sha,
                                 self.root / 'dr.jsonl', components=components, budget=RowBudget(10 ** 7, seed_count=5),
                                 tau=.01, samples=2, pitch_cap=4, seed=701)
        replay, _ = rpv.run_dr(again, self.store, components, 'dev', rpv.Deadline(120), draws=200, seed=5)
        self.assertEqual(replay['conditional'], result['conditional'])

    def test_stage_records_and_deadline(self):
        with rpv.stage(self.root / 'ok', 'census', {'config_sha256': 'x'}) as out:
            (out / 'census.json').write_text('{}')
        manifest = json.loads((self.root / 'ok/manifest.json').read_text())
        self.assertEqual(set(manifest['artifact_sha256']), {'census.json', 'started.json'})
        with self.assertRaises(RuntimeError):
            with rpv.stage(self.root / 'bad', 'census', {}):
                raise RuntimeError('boom')
        self.assertEqual(len(list((self.root / 'bad').glob('failure-*.json'))), 1)
        self.assertFalse((self.root / 'bad/manifest.json').exists())
        with self.assertRaisesRegex(pa.IntegrityError, 'stage output exists'):
            with rpv.stage(self.root / 'ok', 'census', {}):
                pass
        deadline = rpv.Deadline(1e-9)
        with self.assertRaises(rpv.BudgetExceeded):
            deadline.check()
        with self.assertRaisesRegex(pa.IntegrityError, 'after 2025'):
            rpv.guard_dates(self.frame.assign(game_date='2026-04-01'))

    def test_registration_gate_and_dispatch(self):
        repo_config = json.loads((REPO / 'configs/ML-POLICY-MATERIALIZATION-v1.json').read_text())
        with self.assertRaisesRegex(pa.IntegrityError, 'not registered'):
            rpv.registration(repo_config)  # the committed proposal cannot run
        out, report = self.materialize()
        bundle_path, bundle_sha = self.syn.bundle_file()
        config = {'protocol': rpv.PROTOCOL, 'registered': True, 'status': 'REGISTERED',
                  'execution': {'enabled': True, 'real_data_enabled': True},
                  'decisions': {'D-1': 'BC_P', 'D-2': 'all_regular_season_pas', 'D-3': 'refuse_no_logged_action_sticky',
                                'D-4': 'continue_known_pitcher', 'D-5': 'conditional_plus_worst_case_bounds',
                                'D-6': 'observed_next_row_state', 'D-7': 'window_start_snapshot', 'D-8': 'refuse'},
                  'identity_registration': {'classes': self.syn.classes, 'we_contract_sha256': self.syn.we_sha,
                                            'expected_identity_sha256': None,
                                            'search': {'tau': .01, 'samples': 2, 'pitch_cap': 4, 'planning_seed': 701,
                                                       'budget': 10 ** 7}},
                  'le2025_validation_plan': {'stages': {'S3_profile': {'starts': 2, 'cap_seconds': 60},
                                                        'S4_V5_denominators': {'cap_seconds': 60},
                                                        'V4_dr_evaluate': {'cap_seconds': 120}},
                                             'bootstrap': {'draws': 100, 'seed': 3}, 'profile_as_of': {'dev': '2025-07-02'}},
                  'registered_inputs': {'bc': {'path': str(out / 'bc_BC_P.json'), 'file_sha256': report['bc']['BC_P']['file_sha256']},
                                        'support': {'path': str(out / 'support_primary.json'),
                                                    'file_sha256': report['support_table']['file_sha256']}}}
        config_path = self.root / 'config.json'; config_path.write_text(json.dumps(config))
        inputs = {'bundle_path': bundle_path, 'bundle_sha': bundle_sha, 'files': self.syn.bundle['files'],
                  'paths': self.syn.paths, 'prep': {'features': {'tokens': {'type_vocabulary': TYPES}}},
                  'frame': self.frame, 'store': self.store, 'member_loader': load_member}
        runs = self.root / 'artifact' / 'runs' / 'ML-MATRIX-20260924'
        runs.mkdir(parents=True)
        with mock.patch('run_ml_matrix.check_location', lambda local, output: self.root / 'artifact'):
            for command in ('census', 'v5-denominators', 'profile', 'dr-evaluate'):
                with self.subTest(command):
                    rpv.dispatch(command, config, config_path, {}, runs / 'VAL' / command, inputs)
                    self.assertTrue((runs / 'VAL' / command / 'manifest.json').exists())
            dr = json.loads((runs / 'VAL/dr-evaluate/dr.json').read_text())
            self.assertEqual(dr['evaluable_pas'], 6)
            started = json.loads((runs / 'VAL/dr-evaluate/started.json').read_text())
            self.assertEqual(started['decisions']['D-7'], 'window_start_snapshot')
            with self.assertRaisesRegex(pa.IntegrityError, 'stage output exists'):  # attempts are never overwritten
                rpv.dispatch('census', config, config_path, {}, runs / 'VAL' / 'census', inputs)
            broken = {**config, 'decisions': {**config['decisions'], 'D-1': 'eligible_only'}}
            with self.assertRaisesRegex(pa.IntegrityError, 'registered decision required: D-1'):
                rpv.dispatch('census', broken, config_path, {}, runs / 'VAL' / 'x', inputs)


class SemiSyntheticTests(RunnerFixture):
    """V2/V3 in the declared world W (fake G0 + WE, CH delivered by the league fallback)."""
    def world(self, name):
        out, report = self.materialize()
        bc_sha, support_sha = report['bc']['BC_P']['file_sha256'], report['support_table']['file_sha256']
        bc = pa.load_train_bc(out / 'bc_BC_P.json', bc_sha)
        blocks = preq.pa_blocks(self.frame, np.flatnonzero(self.frame.split.eq('dev').to_numpy()))
        normal = [b for b in blocks if b[0] in ('200:1', '201:1')]
        components = self.syn.bind(bc, rpv.pa_contexts(self.store, normal))
        runtime = pr.build_runtime(out / 'bc_BC_P.json', bc_sha, out / 'support_primary.json', support_sha,
                                   self.root / f'{name}.jsonl', components=components,
                                   budget=RowBudget(10 ** 8, seed_count=5), tau=.01, samples=2, pitch_cap=4, seed=701)
        starts = [preq.pa_requests(self.store, positions, runtime.sha256)[0][0].state for _, positions in normal]
        return runtime, components, starts, bc

    def test_v2_known_logging_law_matches_world_truth(self):
        runtime, components, starts, bc = self.world('v2')
        law = lambda state: (runtime.bc.actions, runtime.bc.probabilities(state))  # generating law = pi_b_hat
        report = pss.run_world(runtime, components, starts, law=law, rule='fallback', logs_per_start=150, cap=4,
                               truth_rollouts=150, seed=21, draws=200, budget=RowBudget(10 ** 8, seed_count=5))
        self.assertEqual((report['world_refused_logs'], report['estimate']['evaluable_pas']), (0, 300))
        for row in report['per_start']:
            for name in ('candidate', 'reference'):
                se = np.hypot(row['dr_se'][name], row['truth'][f'{name}_mc_se'])
                self.assertLess(abs(row['dr_mean'][name] - row['truth'][name]), 4 * se, (name, row))
        outside = sum(r['status'] == pr.OUTSIDE_POLICY_SUPPORT for r in runtime.ledger.decisions())
        self.assertGreater(outside, 0)  # logged CH outside the mask was delivered by the world, rho=0

    def test_world_refuse_rule_and_v3_transport(self):
        runtime, components, starts, bc = self.world('v3')
        refused = pss.run_world(runtime, components, starts, law=lambda s: (runtime.bc.actions, runtime.bc.probabilities(s)),
                                rule='refuse', logs_per_start=20, cap=4, truth_rollouts=10, seed=5, draws=50,
                                budget=RowBudget(10 ** 8, seed_count=5))
        self.assertGreater(refused['world_refused_logs'], 0)  # CH has no pool: V2 cannot run as defined for it
        runtime, components, starts, bc = self.world('v3b')
        tilted = lambda s: (runtime.bc.actions, np.array([.05, .9, .05]))  # generated under another law than pi_b_hat
        report = pss.run_world(runtime, components, starts, law=tilted, rule='fallback', logs_per_start=40, cap=4,
                               truth_rollouts=40, seed=9, draws=50, budget=RowBudget(10 ** 8, seed_count=5))
        self.assertIsNotNone(report['delta_gap'])
        self.assertIsNone(report['estimate']['causal_effect'])


if __name__ == '__main__':
    unittest.main()
