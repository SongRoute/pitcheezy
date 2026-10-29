"""SYNTHETIC-ONLY tests for EXP-OPE-1STEP-01 spec v2 (pitchmdp/ope_onestep.py, scripts/run_ope_onestep.py).

The sealed-candidate identity (passes and fails deterministically), tau = inf, the enumerated-toy exactness of the
one-step DR and IPS terms (correct and wrong q-hat), the outcome-shuffle null, the model-term identity M = tau (KL(pi||
ref) + KL(ref||pi)) when q_pi == q_dr, off-mask / positivity rows, the shrinkage family, D margins and ties, the perm
RNG and KL matching, the row-wise kl_policy mirror, the shared cluster bootstrap (determinism, invalid replicates,
paired identical columns exactly 0), the runner's refusals, and --check / --run on a synthetic sealed stage built
through the registered runtime (v3), request generator, PA-end rule and estimator. No real data, no 2026 rows.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))
from test_policy_identity import PROV, WE
from test_policy_validation import NO_PITCH, build_store, game_frame
from pitchmdp import ope_onestep as ope
from pitchmdp import policy_artifacts as pa
from pitchmdp import policy_estimator as est
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as pr
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.rollout_policy import BCRecord, CategoricalBC, PAState, kl_policy
import run_policy_validation as rpv
import run_ope_onestep as runner

TAU = .1
GAMES = [200, 201]
GRID = ['inf', .5, .2, .1, .09, .07, .05, .02]
TAUS = [ope.parse_tau(t) for t in GRID]
LEDGER = 'ledger-dev-0.jsonl'
STAGE_FILES = (LEDGER, 'pa_values.parquet', 'dr.json', 'manifest.json', 'started.json')
MARGINS = [0, .005, '1se']
OPEN = {'ess_ratio_min': 0., 'noise_q90_max': 1e9, 'game_ess_min': 0., 'pa_ess_min': 0.}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_q(state, mask, salt):
    """A deterministic stand-in for a Monte Carlo Q: finite exactly on the mask, -inf elsewhere (as q_values)."""
    rng = np.random.default_rng([salt, int(state.pitcher), state.balls, state.strikes, len(state.history)])
    return np.where(mask, .3 + .4 * rng.random(len(mask)), -np.inf)


class SyntheticStage:
    """A sealed S6-like stage directory: TRAIN BC (pitcher 9: CH/FF/SL with mask FF/SL; pitcher 8: CH/FF with mask
    CH/FF), a runtime-v3 candidate ledger over two DEV games (P3 at tau 0.1 from a planning Q; q_reference an
    independent evaluation Q; paired-difference SE 0 at the planning argmax), PA facts from
    run_policy_validation.submit_pas with the registered PA-end rule, and pa_values / dr.json from
    policy_estimator.estimate (L1-R), sealed by run_policy_validation.stage. Game 201 PA 2 is thrown by pitcher 8 (its
    logged SL has no logging mass: a positivity row); game 200 PA 4 by pitcher 9 (E0, PA end unobservable)."""
    def __init__(self, root, tau=TAU, name='S6-synthetic'):
        frame = game_frame()
        frame.loc[frame.index[(frame.game_pk == 201) & (frame.at_bat_number == 2)], 'pitcher'] = 8
        frame.loc[frame.index[(frame.game_pk == 200) & (frame.at_bat_number == 4)], 'pitcher'] = 9
        self.frame = frame
        store = build_store(frame)
        records = [BCRecord(PAState(0, 0, '9', 'L'), a, 'train') for a in ('FF', 'FF', 'SL', 'CH')]
        records += [BCRecord(PAState(0, 0, '8', 'L'), a, 'train') for a in ('CH', 'FF')]
        root.mkdir(parents=True, exist_ok=True)
        art = pa.save_train_bc(CategoricalBC().fit(records), root / f'{name}-bc.json', {**PROV, 'train_rows': 6})
        _, support_sha = pa.save_support_table(art, [('9', 'L', np.array([False, True, True])),
                                                     ('8', 'L', np.array([True, True, False]))], root / f'{name}-s.json')
        _, hands_sha = pa.save_hand_registry(art, {'9': 'R', '8': 'R'}, 6, root / f'{name}-h.json')
        artifact = pa.load_train_bc(root / f'{name}-bc.json', art.file_sha256)
        support, support_identity = pa.load_support_table(root / f'{name}-s.json', support_sha, artifact)
        hands, hands_identity = pa.load_hand_registry(root / f'{name}-h.json', hands_sha, artifact)
        reference = pr.MaskedReference(artifact.bc, support)

        def candidate(state):  # build_runtime's candidate with the planner replaced by synthetic_q
            mask = reference.support(state)
            q = synthetic_q(state, mask, 1)
            p = kl_policy(q, reference.probabilities(state), mask, tau)
            off = np.where(mask, 0., np.nan)
            se = np.where(mask, .1, np.nan)
            se[int(np.argmax(q))] = 0.
            return p, {'q': synthetic_q(state, mask, 2) + off, 'mc_se': np.full(len(mask), .01) + off,
                       'q_planning': q + off, 'planning_diff_se': se, 'source': 'evaluation_seed'}
        identity = {'contract': 'ML-POLICY-IDENTITY-v1', 'search': {'policy': 'P3', 'tau': TAU, 'samples': 4,
                                                                    'pitch_cap': 4, 'seed': 701},
                    'dr_q': {'source': 'evaluation_seed', 'evaluation_seed': 1701}}
        identity['sha256'] = canonical_hash(identity)
        self.identity = identity
        self.dir = root / name
        we = WE()
        with rpv.stage(self.dir, 'dr-evaluate', {'git': {'commit': 'c' * 40}, 'registration_chain': []}) as out:
            runtime = pr.PolicyRuntime(artifact, support, support_identity, out / LEDGER, candidate=candidate,
                                       candidate_identity=identity, hand_registry=hands,
                                       hand_registry_sha256=hands_identity, positivity_record=True)
            blocks = rpv.game_blocks(frame, GAMES, 'dev')
            table = preq.game_table(frame, set(GAMES))
            self.facts = rpv.submit_pas(runtime, store, blocks, rpv.Deadline(), no_pitch=NO_PITCH,
                                        outcome=lambda positions: preq.pa_outcome(frame, positions, we.predict_defense,
                                                                                  table))
            result, self.rows = est.estimate(runtime.ledger.iter_decisions(), self.facts, draws=20, seed=5,
                                             invalid_share_max=1., minimum={'games': 1, 'pa_starts': 1},
                                             censoring='l1r')
            rpv.dump(out / 'dr.json', {**result, 'games': GAMES, 'identity_sha256': identity['sha256'],
                                       'dr_q': identity['dr_q']})
            pd.DataFrame([{**{k: v for k, v in r.items() if k != 'strata'},
                           **{f'stratum_{k}': v for k, v in r['strata'].items()}} for r in self.rows]
                         ).to_parquet(out / 'pa_values.parquet', index=False)
        self.ledger = runtime.ledger
        self.decisions = runtime.ledger.decisions()

    def pas(self):
        e0 = {r['pa_id']: r['in_population'] for r in self.rows}
        return {p: {**f, 'in_population': e0[p]} for p, f in self.facts.items()}

    def scan(self, families=('temperature', 'shrinkage', 'cross_replicate', 'neg'), tau=TAU):
        return ope.Scan(taus=TAUS, sealed_tau=tau, identity_tolerance=1e-12, statuses=list(ope.RESULT_STATUSES),
                        families=list(families), margins=MARGINS, expected_q_source='evaluation_seed').run(
            self.ledger.iter_decisions(), self.pas())


class SyntheticEnvironment(runner.Environment):
    """The synthetic frame and fake WE in place of the registered loaders; a fixed git state."""
    def __init__(self, frame):
        self.frame = frame

    def git_state(self, source_commit):
        return {'commit': 'f' * 40, 'code_paths': list(runner.CODE_PATHS), 'code_dirty': False,
                'source_commit': source_commit, 'source_is_ancestor': True, 'code_changed_since_source': []}

    def load_frame_and_we(self, spec, checkpoint):
        we = WE()
        config = {'seeds': {'base': 7}, 'le2025_validation_plan': {'stages': {'S6_V4': {'n_games': len(GAMES)}}}}
        return self.frame, (lambda state, home: float(we.predict_defense(state, home))), config, {'synthetic': True}


def make_config(stage_dir, output_dir, **overrides):
    config = {
        'experiment_id': 'EXP-OPE-1STEP-01-SYNTHETIC', 'exposure_label': 'SYNTHETIC', 'stage_dir': str(stage_dir),
        'ledger': LEDGER, 'pins': {name: sha(Path(stage_dir) / name) for name in STAGE_FILES},
        'output_dir': str(output_dir), 'tau_grid': GRID, 'sealed_tau': TAU, 'identity_tolerance': 1e-12,
        'expected_dr_q_source': 'evaluation_seed', 'families': ['temperature', 'shrinkage'],
        'primary': {'family': 'temperature', 'tau': TAU, 'subset': 'all', 'variant': 'dr'},
        'disagreement': {'q': 'q_planning', 'ties': 'not_in_D', 'margins': MARGINS, 'confident': '1se'},
        'eligibility': {'statuses': list(ope.RESULT_STATUSES), 'population': 'E0', 'require_reward': True},
        'evaluability': dict(OPEN),
        'placebos': {'perm_draws': 3, 'perm_seed': 4255355903, 'kl_tolerance': .01, 'outcome_shuffle_draws': 4,
                     'outcome_shuffle_seed': 709946931, 'neg': True},
        'sensitivities': {'switch_caps': [5, 10], 's_pos': True, 'cross_replicate': True},
        'bootstrap': {'draws': 200, 'seeds': {'game': 1322940374, 'pitcher': 1542332542, 'trend': 2753817544},
                      'invalid_share_max': .5, 'minimum': {'games': 1, 'pitchers': 1, 'pas': 1}},
        'pa_facts': {'source': 'frame', 'split': 'dev',
                     'policy_registration': {'config': {'path': '/nonexistent/ML-POLICY.json', 'sha256': 'a' * 64}},
                     'local_config': {'path': '/nonexistent/local.json', 'sha256': 'b' * 64}}}
    config.update(overrides)
    return config


def toy_decision(mask, logging, reference, q_hat, a, q_pi=None):
    """decision_arrays-shaped dict for a hand-built decision with logged action ``a``."""
    mask = np.asarray(mask, dtype=bool)
    logging, reference, q_hat = (np.asarray(v, dtype=float) for v in (logging, reference, q_hat))
    q_pi = q_hat if q_pi is None else np.asarray(q_pi, dtype=float)
    on = bool(mask[a])
    return {'mask': mask, 'logging': logging, 'reference': reference, 'q_dr': np.where(mask, q_hat, np.nan),
            'q_pi': np.where(mask, q_pi, np.nan), 'se': np.where(mask, .01, np.nan), 'a': a, 'on_mask': on,
            'b_a': float(logging[a]), 'rho_ref': float(reference[a] / logging[a]) if on else 0.}


class StageCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.stage = SyntheticStage(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def result_rows(self):
        return [r for r in self.stage.decisions if r['status'] in ope.RESULT_STATUSES]


class FamilyAndTermTests(StageCase):
    def test_identity_check_passes_and_fails_deterministically(self):
        rows = self.result_rows()
        self.assertEqual({r['status'] for r in rows}, set(ope.RESULT_STATUSES))
        for row in rows:
            candidate = np.array(row['result']['candidate'])
            self.assertTrue(np.array_equal(ope.policy_tau(row, TAU), candidate))  # bit for bit
            d = ope.decision_arrays(row, 3)
            self.assertTrue(np.array_equal(ope.policy_tau(row, ope.INF), d['reference']))
            pi = ope.policy_tau(row, TAU)
            self.assertEqual(pi[d['a']] / d['b_a'] if d['on_mask'] else 0., row['result']['rho_candidate'])
        scan = self.stage.scan()
        self.assertEqual((scan.identity['rows_checked'], scan.identity['max_abs_diff_candidate'],
                          scan.identity['max_abs_diff_rho']), (len(rows), 0., 0.))
        other = SyntheticStage(self.root / 'other', tau=.11, name='S6-tau011')  # candidate built at another tau
        for _ in range(2):  # deterministic refusal
            with self.assertRaisesRegex(pa.IntegrityError, 'identity check failed'):
                other.scan()
        swapped = copy.deepcopy(rows[0])  # the candidate is NOT reproduced from q_reference
        swapped['result']['q_planning'] = swapped['result']['q_reference']
        self.assertFalse(np.array_equal(ope.policy_tau(swapped, TAU), np.array(rows[0]['result']['candidate'])))

    def test_enumerated_toy_unbiasedness_selection_bias_and_outcome_shuffle_null(self):
        # actions: 0 off the mask with logging mass (OUTSIDE_POLICY_SUPPORT), 1-3 on the mask, 4 off without mass
        mask, logging = [False, True, True, True, False], [.15, .4, .3, .15, 0.]
        reference = [0., .5, .3, .2, 0.]
        q_true = np.array([.42, .55, .48, .61, .5])
        outcomes = {a: ([q_true[a] - .2, q_true[a] + .3], [.6, .4]) for a in range(5)}  # E[G | a] = q_true[a]
        marginal = [(g, logging[a] * p) for a in range(5) for g, p in zip(*outcomes[a])]  # G independent of a
        q_pi = np.array([0., .2, .9, .4, 0.])
        for q_hat in (q_true, np.array([0., .1, .9, .3, 0.])):  # correct and deliberately wrong q-hat
            for tau in (.5, .1, .02):
                d0 = toy_decision(mask, logging, reference, q_hat, 1, q_pi)
                pi = ope.tilt(d0['q_pi'], d0['reference'], d0['mask'], tau)
                truth = float(((pi - d0['reference']) * np.where(d0['mask'], q_true, 0.)).sum())
                dr = ips = conditional = kept = shuffle_dr = shuffle_ips = 0.
                for a in range(5):
                    d = toy_decision(mask, logging, reference, q_hat, a, q_pi)
                    for g, p in zip(*outcomes[a]):
                        model, correction, ips_term, _ = ope.onestep_terms(pi, d, g)
                        dr += logging[a] * p * (model + correction)
                        ips += logging[a] * p * ips_term
                        if mask[a]:  # what dropping the OUTSIDE_POLICY_SUPPORT action would estimate
                            conditional += logging[a] * p * (model + correction)
                            kept += logging[a] * p
                    for g, p in marginal:
                        model, correction, ips_term, _ = ope.onestep_terms(pi, d, g)
                        shuffle_dr += logging[a] * p * (model + correction)
                        shuffle_ips += logging[a] * p * ips_term
                self.assertAlmostEqual(dr, truth, places=14)
                self.assertAlmostEqual(ips, truth, places=14)
                self.assertAlmostEqual(shuffle_dr, 0., places=14)  # the outcome-shuffle null is 0 in expectation
                self.assertAlmostEqual(shuffle_ips, 0., places=14)
                if q_hat is not q_true:
                    self.assertGreater(abs(conditional / kept - truth), 1e-4)  # selection on a_t biases DR

    def test_model_term_identity_when_the_two_q_arrays_agree(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            k = 5
            mask = np.ones(k, dtype=bool)
            mask[rng.integers(k)] = False
            ref = np.where(mask, rng.dirichlet(np.ones(k)), 0.)
            ref /= ref.sum()
            q = rng.random(k)
            d = toy_decision(mask, ref, ref, q, int(np.flatnonzero(mask)[0]))
            for tau in (.5, .1, .02):
                pi = ope.tilt(d['q_pi'], ref, mask, tau)
                model = ope.onestep_terms(pi, d, .5)[0]
                sym = ope.kl(pi, ref, mask) + ope.kl(ref, pi, mask)
                self.assertAlmostEqual(model, tau * sym, places=14)  # M = tau (KL(pi||ref) + KL(ref||pi))

    def test_off_mask_and_positivity_rows_keep_only_the_model_term(self):
        rows = self.result_rows()
        positivity = next(r for r in rows if r['status'] == pr.LOGGING_POSITIVITY)
        outside = next(r for r in rows if r['status'] == pr.OUTSIDE_POLICY_SUPPORT)
        for row in (positivity, outside):
            d = ope.decision_arrays(row, 3)
            self.assertFalse(d['on_mask'])
            self.assertTrue(np.isnan(d['q_dr'][d['a']]))  # never read
            pi = ope.policy_tau(row, .02)
            model, correction, ips, rho = ope.onestep_terms(pi, d, .9)
            self.assertEqual((correction, ips, rho), (0., 0., 0.))
            self.assertTrue(np.isfinite(model) and model != 0.)
        self.assertEqual(ope.decision_arrays(positivity, 3)['b_a'], 0.)
        bad = copy.deepcopy(positivity)
        bad['result']['rho_reference'] = .5
        with self.assertRaisesRegex(pa.IntegrityError, 'recorded rho must be 0'):
            ope.decision_arrays(bad, 3)
        bad = copy.deepcopy(outside)
        bad['result']['q_planning'][bad['result']['logged_index']] = .4
        with self.assertRaisesRegex(pa.IntegrityError, 'off the mask'):
            ope.decision_arrays(bad, 3)
        with self.assertRaisesRegex(pa.IntegrityError, 'no recorded result'):
            ope.decision_arrays(next(r for r in self.stage.decisions if r['status'] == pr.MID_PA), 3)

    def test_shrinkage_factor_and_zero_shrinkage_is_the_reference(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            k = int(rng.integers(1, 6))
            mask = np.ones(k, dtype=bool)
            q, se = rng.random(k), rng.random(k) * rng.choice([.01, .3, 3.])
            c = ope.shrink_factor(q, se, mask)
            self.assertTrue(0. <= c <= 1.)
            if k < 2:
                self.assertEqual(c, 0.)
        mask = np.array([True, True, True])
        q, se = np.array([.4, .5, .6]), np.array([1., 1., 1.])  # noise far above the spread: c = 0
        self.assertEqual(ope.shrink_factor(q, se, mask), 0.)
        d = toy_decision(mask, [.3, .3, .4], [.3, .3, .4], q, 0, q)
        d['se'] = se
        for tau in (.5, .02):
            pi, _, c = ope.family_policy('shrinkage', d, tau)
            self.assertEqual(c, 0.)
            self.assertTrue(np.array_equal(pi, d['reference']))  # exactly
        d['se'] = np.zeros(3)  # no noise: c = 1, the temperature law
        pi, _, c = ope.family_policy('shrinkage', d, .1)
        self.assertEqual(c, 1.)
        self.assertTrue(np.allclose(pi, ope.tilt(q, d['reference'], mask, .1), rtol=0, atol=1e-15))

    def test_disagreement_margins_and_ties(self):
        mask = np.array([True, True, True, False])
        ref = np.array([.5, .3, .2, 0.])
        se = np.array([.004, .0, .01, np.nan])  # se[ref_best = 0] = 0.004
        q = np.array([.1, .9, .2, np.nan])
        self.assertEqual(ope.disagreement(q, ref, mask, se, MARGINS), ([True, True, True], False, False))
        close = np.array([.5, .503, .2, np.nan])  # gap .003: in D_0 only
        self.assertEqual(ope.disagreement(close, ref, mask, se, MARGINS), ([True, False, False], False, False))
        mid = np.array([.5, .5045, .2, np.nan])  # gap .0045 > se .004 but < .005
        self.assertEqual(ope.disagreement(mid, ref, mask, se, MARGINS), ([True, False, True], False, False))
        self.assertEqual(ope.disagreement(np.array([.9, .1, .2, np.nan]), ref, mask, se, MARGINS),
                         ([False] * 3, False, False))
        self.assertEqual(ope.disagreement(np.array([.1, .9, .9, np.nan]), ref, mask, se, MARGINS),
                         ([False] * 3, True, False))
        self.assertEqual(ope.disagreement(q, np.array([.4, .4, .2, 0.]), mask, se, MARGINS), ([False] * 3, False, True))

    def test_perm_rng_kl_matching_and_the_row_mirror(self):
        row = self.result_rows()[0]
        sha8 = int(row['sha256'][:8], 16)
        q, mask = np.array([.1, .2, .3, .4, .5, np.nan]), np.array([True] * 5 + [False])
        a = ope.permuted(q, mask, 4255355903, 0, sha8)
        self.assertEqual(sorted(a[mask]), sorted(q[mask]))
        self.assertTrue(np.array_equal(a, ope.permuted(q, mask, 4255355903, 0, sha8), equal_nan=True))
        draws = {tuple(ope.permuted(q, mask, 4255355903, k, sha8)[mask]) for k in range(20)}
        self.assertGreater(len(draws), 1)  # the draw index drives it
        expected = np.random.default_rng(np.random.SeedSequence([4255355903, 3, sha8])).permutation(5)
        self.assertTrue(np.array_equal(ope.permuted(q, mask, 4255355903, 3, sha8)[mask], q[mask][expected]))
        d = toy_decision([True, True], [.5, .5], [.5, .5], [.3, .7], 0)
        for tau in (.5, .1, .01):
            real = ope.tilt(d['q_pi'], d['reference'], d['mask'], tau)
            neg = ope.tilt(-d['q_pi'], d['reference'], d['mask'], tau)
            self.assertAlmostEqual(ope.onestep_terms(neg, d, .5)[0], -ope.onestep_terms(real, d, .5)[0], places=15)
        rng = np.random.default_rng(2)
        n, v = 300, 5
        mask = rng.random((n, v)) < .7
        mask[:, 0] = True
        ref = np.where(mask, rng.random((n, v)) + .05, 0.)
        ref /= ref.sum(axis=1, keepdims=True)
        q = np.where(mask, rng.random((n, v)), np.nan)
        for tau in (.5, .07, .02):  # the row-wise mirror is kl_policy
            rows = ope.tilt_rows(q, ref, mask, tau)
            worst = max(float(np.abs(rows[i] - ope.tilt(q[i], ref[i], mask[i], tau)).max()) for i in range(n))
            self.assertLessEqual(worst, 1e-15)
        target = float(ope._kl_rows(ope.tilt_rows(q, ref, mask, .05), ref, mask).mean())
        shuffled = q.copy()
        for i in range(n):
            idx = np.flatnonzero(mask[i])
            shuffled[i, idx] = q[i, idx[rng.permutation(len(idx))]]
        tau_p, value, matched = ope.match_tau(shuffled, ref, mask, target, .01)
        self.assertTrue(matched)
        self.assertLessEqual(abs(value - target), .01 * target)  # converged within the tolerance
        self.assertAlmostEqual(float(ope._kl_rows(ope.tilt_rows(shuffled, ref, mask, tau_p), ref, mask).mean()), value)
        self.assertFalse(ope.match_tau(shuffled, ref, mask, 50., .01)[2])  # an unreachable boldness is flagged
        self.assertEqual(ope.match_tau(shuffled, ref, mask, 0., .01)[:2], (ope.INF, 0.))


class BootstrapTests(StageCase):
    def engine(self, minimum=None, draws=300):
        table = self.stage.scan().table()
        return table, ope.Engine(table, draws=draws, seeds={'game': 5, 'pitcher': 6},
                                 minimum=minimum or {'games': 1, 'pitchers': 1, 'pas': 1}, invalid_share_max=.5)

    def test_cluster_bootstrap_is_deterministic_and_counts_invalid_replicates(self):
        keys = ['b', 'a', 'c', 'a', 'b', 'c', 'c']
        one, two = (ope.ClusterBootstrap(keys, unit='game', draws=500, seed=9) for _ in range(2))
        self.assertTrue(np.array_equal(one.m, two.m))
        self.assertEqual(one.order, ['a', 'b', 'c'])
        sample = np.random.default_rng(9).integers(3, size=(500, 3))  # the A2 draw
        self.assertTrue(np.array_equal(one.m, np.stack([np.bincount(s, minlength=3) for s in sample]).astype(float)))
        self.assertTrue(np.allclose(one.replicate(np.arange(7.)), one.m @ np.array([1 + 3, 0 + 4, 2 + 5 + 6])))
        table, eng = self.engine()
        positivity_rows = eng.t['status'] == pr.LOGGING_POSITIVITY
        self.assertEqual(int(positivity_rows.sum()), 1)
        cell = eng.family_cell('temperature', 'dr', 'all')
        again = self.engine()[1].family_cell('temperature', 'dr', 'all')
        for unit in ope.UNITS:
            self.assertTrue(np.array_equal(cell.reps[unit]['delta'], again.reps[unit]['delta'], equal_nan=True))
        game_201 = eng.boots['game'].order.index(201)  # S-pos subset: PAs with a positivity node removed
        validity = eng.validity('two_strike')
        count = eng.boots['game'].replicate((eng.t['strikes'] == 2).astype(float))
        self.assertTrue(np.array_equal(validity['units']['game'], count > 0))
        self.assertGreater(game_201, -1)
        strict = ope.Engine(table, draws=300, seeds={'game': 5, 'pitcher': 6},
                            minimum={'games': 2, 'pitchers': 1, 'pas': 1}, invalid_share_max=.5)
        expected_invalid = int((strict.boots['game'].drawn.sum(axis=1) < 2).sum())  # both games must be drawn
        record = strict.record(strict.family_cell('temperature', 'dr', 'all'))[ope.tau_key(.05)]
        self.assertEqual(record['delta']['game']['invalid_replicates'], expected_invalid)
        self.assertGreater(expected_invalid, 0)
        below = ope.Engine(table, draws=300, seeds={'game': 5, 'pitcher': 6},
                           minimum={'games': 3, 'pitchers': 1, 'pas': 1}, invalid_share_max=.5)
        self.assertIn('below the registered minimum',
                      below.record(below.family_cell('temperature', 'dr', 'all'))['0.1']['delta']['game']['reason'])

    def test_paired_difference_of_identical_columns_is_exactly_zero(self):
        _, eng = self.engine()
        cell = eng.family_cell('temperature', 'dr', 'all')
        same = ope.paired(cell, eng.family_cell('temperature', 'dr', 'all'))
        for unit in ope.UNITS:
            self.assertTrue((same.reps[unit]['delta'][same.valid[unit]['delta']] == 0).all())
        record = eng.record(same)
        for key in ('0.1', '0.02'):
            self.assertEqual(record[key]['delta']['mean'], 0.)
            self.assertEqual(record[key]['delta']['game']['ci95'], [0., 0.])
        r = eng.record(cell)
        self.assertIsNone(r['inf']['R']['mean'])  # M = 0 at tau = inf: the realized/promised ratio is undefined
        self.assertEqual(r['inf']['delta']['mean'], 0.)
        m, c, dlt = r['0.05']['M']['mean'], r['0.05']['C']['mean'], r['0.05']['delta']['mean']
        self.assertAlmostEqual(m + c, dlt, places=15)
        self.assertAlmostEqual(r['0.05']['R']['mean'], dlt / m, places=12)

    def test_trend_max_t_and_bounds(self):
        table, eng = self.engine()
        cell = eng.family_cell('temperature', 'dr', 'all')
        ev = {ope.tau_key(t): True for t in TAUS}
        result = ope.trend(cell, TAUS, ev, eng.draws, .5)
        x = np.log(1 / np.array(TAUS[1:]))
        y = cell.point['delta'][1:]
        self.assertAlmostEqual(result['slope'], float(np.polyfit(x, y, 1)[0]), places=12)
        self.assertEqual(ope.trend(cell, TAUS, {'0.1': True}, eng.draws, .5)['slope'], None)  # one evaluable tau
        mt = ope.max_t(cell, TAUS, ev, eng.draws, .5)
        self.assertTrue(0. <= mt['game']['p_value'] <= 1.)
        b = ope.bounds(table, 'temperature')
        self.assertEqual((b['unobserved_decisions'], b['unobserved_pas']), (4, 1))
        observed = int(table['observed'].sum())
        for key, (lo, hi) in b['bounds'].items():
            point = eng.record(cell)[key]['delta']['mean'] * observed / len(table['observed'])
            self.assertLessEqual(lo, hi)
            self.assertLessEqual(lo, point + 1e-15)  # the unobserved PAs' model terms enter both ends
        self.assertEqual(b['bounds']['inf'], [0., 0.])


class RunnerTests(StageCase):
    def write_config(self, name, **overrides):
        out = Path(overrides.pop('output_dir', None) or Path(tempfile.mkdtemp(dir=self.root)) / 'out')
        path = self.root / f'{name}.json'
        path.write_text(json.dumps(make_config(self.stage.dir, out, **overrides)))
        return path, out

    def env(self):
        return SyntheticEnvironment(self.stage.frame)

    def test_frame_facts_follow_the_registered_submit_path(self):
        frame = self.stage.frame
        blocks = rpv.game_blocks(frame, GAMES, 'dev')
        we = WE()
        facts = ope.frame_facts(frame, blocks, preq.game_table(frame, set(GAMES)),
                                lambda s, h: float(we.predict_defense(s, h)))
        self.assertEqual(list(facts), list(self.stage.facts))
        for pa_id, info in self.stage.facts.items():
            for key in ('game', 'reward', 'kind', 'reason', 'end', 'end_kind', 'flags', 'start_pitcher', 'date'):
                self.assertEqual(facts[pa_id][key], info[key], (pa_id, key))
        sealed = ope.sealed_rows(pd.read_parquet(self.stage.dir / 'pa_values.parquet').to_dict('records'))
        check = ope.FactsCheck('l1r', 1e-12)
        pas = {p: {**facts[p], 'in_population': sealed[p]['in_population']} for p in sealed}
        by = {p: rows for p, _, rows in est.by_pa(self.stage.ledger.iter_decisions(), pas)}
        for p in sealed:
            check.check(p, by[p], ope.estimator_info(sealed[p], pas[p]), sealed[p], pas[p])
        self.assertEqual(check.record()['pas'], len(sealed))
        self.assertGreater(check.record()['reward_verified_pas'], 0)
        verified = next(iter(check.reward_verified))
        g = pas[verified]['reward']
        wrong = {**pas[verified], 'reward': g - 1e-6 if g > .5 else g + 1e-6}  # stays inside [0, 1]
        with self.assertRaisesRegex(pa.IntegrityError, 'pa_values cross-check'):
            ope.FactsCheck('l1r', 1e-12).check(verified, by[verified], ope.estimator_info(sealed[verified], wrong),
                                              sealed[verified], wrong)

    def test_runner_refuses_bad_pins_and_existing_output_before_reading(self):
        path, out = self.write_config('pins', pins={**make_config(self.stage.dir, '/x')['pins'], 'dr.json': '0' * 64})
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
        self.assertFalse(out.exists())  # refused before any output directory exists
        copy_dir = self.root / 'broken-stage'
        copy_dir.mkdir()
        for name in STAGE_FILES:
            (copy_dir / name).write_bytes((self.stage.dir / name).read_bytes())
        config = make_config(self.stage.dir, self.root / 'unused')
        (copy_dir / 'dr.json').write_bytes(b'not json at all')  # unparseable, but the pin is checked first
        path = self.root / 'broken.json'
        path.write_text(json.dumps({**config, 'stage_dir': str(copy_dir), 'output_dir': str(self.root / 'b-out')}))
        stderr, old = io.StringIO(), sys.stderr
        sys.stderr = stderr
        try:
            self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
        finally:
            sys.stderr = old
        self.assertIn('differs from its pin: dr.json', stderr.getvalue())
        path, out = self.write_config('exists')
        (out / 'check').mkdir(parents=True)
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
        self.assertEqual(list((out / 'check').iterdir()), [])
        for name, override in (('inside', {'output_dir': str(self.stage.dir / 'x')}), ('unknown', {'surprise': 1}),
                               ('grid', {'tau_grid': [.5, 'inf', .1]}), ('primary', {'primary': {
                                   'family': 'temperature', 'tau': .05, 'subset': 'all', 'variant': 'dr'}}),
                               ('margins', {'disagreement': {'q': 'q_reference', 'ties': 'not_in_D', 'margins': MARGINS,
                                                             'confident': '1se'}})):
            path, _ = self.write_config(name, **override)
            self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2, name)
        manifest = json.loads((self.stage.dir / 'manifest.json').read_text())
        unlisted = self.root / 'unlisted-stage'
        unlisted.mkdir()
        for name in STAGE_FILES:
            (unlisted / name).write_bytes((self.stage.dir / name).read_bytes())
        manifest['artifact_sha256'].pop('pa_values.parquet')
        (unlisted / 'manifest.json').write_text(json.dumps(manifest))
        path = self.root / 'unlisted.json'
        path.write_text(json.dumps(make_config(unlisted, self.root / 'u-out')))
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
        path, out = self.write_config('drq', expected_dr_q_source='planning_reuse')
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
        self.assertIn('DR q-hat source', json.loads((out / 'check' / 'failed.json').read_text())['error'])

    def test_check_on_the_synthetic_stage_writes_the_expected_counts_and_prediag(self):
        path, out = self.write_config('check')
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 0)
        doc = json.loads((out / 'check' / 'check.json').read_text())
        statuses = dict(pd.Series([r['status'] for r in self.stage.decisions]).value_counts())
        self.assertEqual(doc['counts']['by_status'], {k: int(v) for k, v in statuses.items()})
        self.assertEqual(doc['counts']['by_status'], {pr.SUPPORTED: 41, pr.OUTSIDE_POLICY_SUPPORT: 5,
                                                      pr.LOGGING_POSITIVITY: 1, pr.NO_LOGGED_ACTION: 2,
                                                      pr.UNKNOWN_PITCHER: 1, pr.MISSING_LABEL: 2, pr.MID_PA: 16})
        counts = doc['counts']
        self.assertEqual(counts['eligible_decisions'], 43)
        self.assertEqual(counts['eligible_by_status'], {pr.SUPPORTED: 38, pr.OUTSIDE_POLICY_SUPPORT: 4,
                                                        pr.LOGGING_POSITIVITY: 1})
        self.assertEqual(counts['excluded_decisions'], {
            'no_reward_pa:TERMINAL_VALUE_MISSING:end_state_gap': 4, f'status:{pr.MID_PA}': 16,
            f'status:{pr.MISSING_LABEL}': 2, f'status:{pr.NO_LOGGED_ACTION}': 2, f'status:{pr.UNKNOWN_PITCHER}': 1})
        self.assertEqual(counts['e0_pas_without_reward'], {'TERMINAL_VALUE_MISSING:end_state_gap': 1})
        self.assertEqual((counts['pas']['pas'], counts['pas']['e0'], counts['pas']['eligible_pas']), (16, 13, 12))
        self.assertEqual(counts['p_pos']['decision'], 1 / 43)
        self.assertEqual(counts['p_out']['pa'], 4 / 12)
        identity = doc['provenance']['identity_check']
        self.assertEqual((identity['rows_checked'], identity['max_abs_diff_candidate']), (47, 0.))
        self.assertEqual(doc['provenance']['ledger']['header']['dr_q']['source'], 'evaluation_seed')
        self.assertEqual(doc['provenance']['ledger']['header']['candidate_sha256'], self.stage.identity['sha256'])
        self.assertEqual(doc['provenance']['pa_facts']['crosscheck']['pas'], 16)
        expected = [0, 0, 0]
        e0 = {r['pa_id'] for r in self.stage.rows if r['in_population']}
        rewards = {p: f['reward'] for p, f in self.stage.facts.items()}
        for row in self.stage.decisions:
            if row['status'] in ope.RESULT_STATUSES and row['pa_id'] in e0 and rewards[row['pa_id']] is not None:
                d = ope.decision_arrays(row, 3)
                flags = ope.disagreement(d['q_pi'], d['reference'], d['mask'], d['se'], MARGINS)[0]
                expected = [e + f for e, f in zip(expected, flags)]
        self.assertEqual(list(counts['disagreement']['eligible'].values()), expected)
        self.assertGreaterEqual(expected[0], max(expected[1], expected[2]))  # D_0.005, D_1se are slices of D_0
        self.assertNotIn('dose_response', doc)  # --check computes no G-dependent number
        prediag = json.loads((out / 'check' / 'prediag.json').read_text())
        row = prediag['prediag']['families']['temperature']['0.1']
        for key in ('decision_ess', 'pa_ess', 'pa_ess_ratio_candidate_reference', 'game_ess', 'noise_ratio_q90',
                    'mean_kl', 'mean_M', 'max_rho_share', 'evaluable'):
            self.assertIn(key, row)
        self.assertAlmostEqual(prediag['prediag']['families']['temperature']['inf']['pa_ess_ratio_candidate_reference'], 1.)
        text = json.dumps(prediag['prediag'])
        for forbidden in ('"delta"', '"C"', '"R"', 'reward', 'ci95'):  # no G-dependent quantity in prediag
            self.assertNotIn(forbidden, text)
        names = sorted(line.split('  ')[1] for line in (out / 'check' / 'SHA256SUMS').read_text().splitlines())
        self.assertEqual(names, ['check.json', 'prediag.json', 'started.json'])
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)  # never overwritten

    def test_run_writes_every_section_and_matches_an_independent_recomputation(self):
        path, out = self.write_config('run', outputs={'decisions_parquet': True})
        self.assertEqual(runner.main(['--config', str(path), '--run'], env=self.env()), 0)
        doc = json.loads((out / 'run' / 'results.json').read_text())
        for section in ('provenance', 'counts', 'prediag', 'primary', 'dose_response', 'subsets', 'placebos',
                        'sensitivities', 'bounds', 'trend', 'max_t', 'pa_level'):
            self.assertIn(section, doc)
        self.assertEqual(doc['tau_inf_identity'], {'max_abs_model': 0., 'max_abs_rho_minus_rho_ref': 0.})
        e0 = {r['pa_id'] for r in self.stage.rows if r['in_population']}
        rewards = {p: f['reward'] for p, f in self.stage.facts.items()}
        for tau in (.05, .02):
            values, models, ips = [], [], []
            for row in self.stage.decisions:
                if row['status'] not in ope.RESULT_STATUSES or row['pa_id'] not in e0 or rewards[row['pa_id']] is None:
                    continue
                r, a, g = row['result'], row['result']['logged_index'], rewards[row['pa_id']]
                mask = np.array(r['mask'])
                qp = np.array([-np.inf if v is None else v for v in r['q_planning']])
                qh = np.array([0. if v is None else v for v in r['q_reference']])
                ref, logging = np.array(r['reference']), np.array(r['logging'])
                pi = kl_policy(qp, ref, mask, tau)
                model = float(((pi - ref) * qh)[mask].sum())
                weight = (pi[a] - ref[a]) / logging[a] if mask[a] else 0.
                values.append(model + weight * (g - qh[a]))
                models.append(model)
                ips.append(weight * g)
            cell = doc['dose_response']['temperature']['dr'][repr(tau)]
            self.assertEqual(cell['n'], 43)
            self.assertAlmostEqual(cell['delta']['mean'], float(np.mean(values)), places=13)
            self.assertAlmostEqual(cell['M']['mean'], float(np.mean(models)), places=13)
            self.assertAlmostEqual(doc['dose_response']['temperature']['ips'][repr(tau)]['delta']['mean'],
                                   float(np.mean(ips)), places=13)
            self.assertIsNotNone(cell['delta']['game']['ci95'])
        self.assertEqual(doc['primary']['tau'], '0.1')
        self.assertEqual(doc['primary']['cell'], doc['dose_response']['temperature']['dr']['0.1'])
        self.assertEqual(set(doc['dose_response']['temperature']), {'dr', 'ips', 'sndr', 'switch:5', 'switch:10'})
        self.assertEqual(set(doc['subsets']['by_subset']), {'all', 'D:0', 'D:0.005', 'D:1se', 'not_D:0', 'two_strike'})
        placebos = doc['placebos']
        self.assertEqual(placebos['perm']['draws'], 3)
        for key, entry in placebos['perm']['tau_p'].items():
            if key != 'inf':
                self.assertLessEqual(entry['max_relative_kl_mismatch'], .01 + 1e-12)
        self.assertLessEqual(placebos['batch_mirror_max_abs_diff'], 1e-12)
        self.assertEqual(set(placebos['outcome_shuffle']) >= {'dr', 'ips'}, True)
        self.assertEqual(placebos['donor']['status'], 'NOT_IMPLEMENTED')
        self.assertIn('paired_real_minus_mean', placebos['estimator_stress'])
        self.assertIsNotNone(doc['trend']['temperature']['slope'])
        self.assertIn('p_value', doc['max_t']['temperature']['game'])
        self.assertEqual(doc['bounds']['temperature']['unobserved_pas'], 1)
        self.assertIn('cross_replicate', doc['sensitivities'])
        rows = pd.read_csv(out / 'run' / 'per_tau.csv')
        self.assertEqual(len(rows) % len(GRID), 0)
        self.assertIn('placebos', set(rows['section']))
        decisions = pd.read_parquet(out / 'run' / 'decisions.parquet')
        self.assertEqual(len(decisions), 43)
        self.assertTrue((decisions['delta_inf'] == 0).all())
        names = sorted(line.split('  ')[1] for line in (out / 'run' / 'SHA256SUMS').read_text().splitlines())
        self.assertEqual(names, ['decisions.parquet', 'per_tau.csv', 'prediag.json', 'results.json', 'started.json'])
        self.assertEqual(json.loads((out / 'run' / 'prediag.json').read_text())['prediag'], doc['prediag'])
        self.assertIsNotNone(doc['provenance']['run']['peak_rss_bytes'])

    def test_registered_thresholds_mark_small_samples_descriptive(self):
        path, out = self.write_config('strict', evaluability={'ess_ratio_min': .5, 'noise_q90_max': .5,
                                                              'game_ess_min': 30, 'pa_ess_min': 100})
        self.assertEqual(runner.main(['--config', str(path), '--run'], env=self.env()), 0)
        doc = json.loads((out / 'run' / 'results.json').read_text())
        self.assertFalse(any(row['evaluable'] for row in doc['prediag']['families']['temperature'].values()))
        self.assertTrue(doc['dose_response']['temperature']['dr']['0.1']['descriptive'])
        self.assertIsNone(doc['trend']['temperature']['slope'])
        self.assertIn('reason', doc['max_t']['temperature'])

    def test_hang_guard_aborts_with_a_failure_record(self):
        path, out = self.write_config('guard')
        self.assertEqual(runner.main(['--config', str(path), '--run', '--max-seconds', '1e-9'], env=self.env()), 3)
        failed = json.loads((out / 'run' / 'failed.json').read_text())
        self.assertEqual(failed['error_type'], 'HangGuardExceeded')
        self.assertFalse((out / 'run' / 'results.json').exists())
        path, out = self.write_config('tolerance', sealed_tau=.05, primary={'family': 'temperature', 'tau': .05,
                                                                              'subset': 'all', 'variant': 'dr'})
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
        self.assertIn('sealed tau', json.loads((out / 'check' / 'failed.json').read_text())['error'])


if __name__ == '__main__':
    unittest.main()
