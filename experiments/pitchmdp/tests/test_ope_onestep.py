"""SYNTHETIC-ONLY tests for EXP-OPE-1STEP-01 spec v3 (pitchmdp/ope_onestep.py, scripts/run_ope_onestep.py).

G-INT (identity, ratios, positivity rows; deterministic pass/fail), tau = inf, the enumerated-toy exactness of the
one-step DR and IPS terms (correct and wrong q-hat), the model-term identity M = tau (KL(pi||ref) + KL(ref||pi)), off-mask
rows, the shrinkage family via tau/c (c = 1 is T bit for bit), D margins and ties, the perm RNG, the flat K-draw mirror
and the common placebo temperature (NOT_MATCHABLE), the outcome-shuffle null (and its power against an injected logging
misspecification), the shared cluster bootstrap, H_P bounds, every branch of the registered decide step, the runner's
refusals, and --check / --run on a synthetic sealed stage built through the registered runtime (v3), request generator,
PA-end rule and estimator. No real data, no 2026 rows.
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
OPEN = {'ess_ratio_min': 0., 'max_weight_share': 1., 'games_min': 1, 'pitchers_min': 1, 'noise_q90_max': 1e9,
        'noise_exempt': [{'family': 'temperature', 'tau': TAU}]}
STRICT = {'ess_ratio_min': .5, 'max_weight_share': .01, 'games_min': 30, 'pitchers_min': 30, 'noise_q90_max': .5,
          'noise_exempt': [{'family': 'temperature', 'tau': TAU}]}


def pytest_approx(values, tol=1e-15):
    """A list of floats equal to ``values`` within tol (list comparison helper)."""
    class Approx(float):
        def __eq__(self, other):
            return abs(float(self) - float(other)) <= tol
        __hash__ = float.__hash__
    return [Approx(v) for v in values]


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
        return ope.Scan(taus=TAUS, sealed_tau=tau, identity_tolerance=1e-12, ratio_rtol=1e-12,
                        statuses=list(ope.RESULT_STATUSES), families=list(families), margins=MARGINS,
                        expected_q_source='evaluation_seed').run(self.ledger.iter_decisions(), self.pas())


class SyntheticEnvironment(runner.Environment):
    """The synthetic frame and fake WE in place of the registered loaders; a fixed git state and free memory."""
    def __init__(self, frame, available=64 * 2 ** 30):
        self.frame, self.available = frame, available

    def git_state(self, source_commit):
        return {'commit': 'f' * 40, 'branch': 'research/side-onestep-1029', 'worktree': '/synthetic',
                'code_paths': list(runner.CODE_PATHS), 'code_dirty': False, 'source_commit': source_commit,
                'source_is_ancestor': True, 'code_changed_since_source': []}

    def available_memory_bytes(self):
        return self.available

    def load_frame_and_we(self, spec, checkpoint):
        we = WE()
        config = {'seeds': {'base': 7}, 'le2025_validation_plan': {'stages': {'S6_V4': {'n_games': len(GAMES)}}}}
        return self.frame, (lambda state, home: float(we.predict_defense(state, home))), config, {'synthetic': True}


def make_config(stage_dir, output_dir, **overrides):
    candidate = ope.read_header(Path(stage_dir) / LEDGER)['pins']['candidate']['sha256']
    config = {
        'experiment_id': 'EXP-OPE-1STEP-01-SYNTHETIC', 'stage_label': 'S6_EXPLORATORY', 'exposure_label': 'SYNTHETIC',
        'stage_dir': str(stage_dir), 'source_commit': 'f' * 40,
        'ledger': LEDGER, 'pins': {name: sha(Path(stage_dir) / name) for name in STAGE_FILES},
        'output_dir': str(output_dir), 'tau_grid': GRID, 'sealed_tau': TAU, 'identity_tolerance': 1e-12,
        'ratio_rtol': 1e-12, 'expected_dr_q_source': 'evaluation_seed', 'candidate_identity_sha256': candidate,
        'families': ['temperature', 'shrinkage'],
        'primary': {'family': 'temperature', 'tau': TAU, 'subset': 'all', 'variant': 'dr'},
        'confirmatory': {'P': 'all', 'P_D': 'D:1se'},
        'disagreement': {'q': 'q_planning', 'ties': 'not_in_D', 'margins': MARGINS, 'confident': '1se'},
        'eligibility': {'statuses': list(ope.RESULT_STATUSES), 'population': 'E0', 'require_reward': True},
        'gate': copy.deepcopy(OPEN), 'sample_minima': {'games': 1, 'pas': 1, 'pitchers': 1},
        'placebos': {'perm_draws': 3, 'perm_seed': 4255355903, 'kl_tolerance': .01, 'tau_bracket_factor': 100,
                     'bisection_iterations': 60, 'outcome_shuffle_null_draws': 4, 'outcome_shuffle_null_seed': 3477356613,
                     'neg': True},
        'sensitivities': {'switch_caps': [5, 10], 's_pos': True, 'cross_replicate': True},
        'bootstrap': {'draws': 200, 'seeds': {'game': 1322940374, 'pitcher': 1542332542, 'trend': 2753817544},
                      'invalid_share_max': .5},
        'decision': {'alpha_one_sided': .025, 'mde_multiplier': 2.8016, 'mde_multiplier_holm_first': 3.083},
        'guards': {'max_seconds': 600, 'max_rss_gb': 8, 'min_available_gb': 10},
        'pa_facts': {'source': 'frame', 'split': 'dev',
                     'policy_registration': {'config': {'path': '/nonexistent/ML-POLICY.json', 'sha256': 'a' * 64},
                                             'addenda': []},
                     'local_config': {'path': '/nonexistent/local.json', 'sha256': 'b' * 64},
                     'processed_frame': {'path': '/nonexistent/pitches.parquet', 'sha256': 'c' * 64},
                     'we_contract_sha256': 'd' * 64}}
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
    def test_gint_passes_and_fails_deterministically(self):
        rows = self.result_rows()
        self.assertEqual({r['status'] for r in rows}, set(ope.RESULT_STATUSES))
        for row in rows:
            candidate = np.array(row['result']['candidate'])
            self.assertTrue(np.array_equal(ope.policy_tau(row, TAU), candidate))  # bit for bit
            d = ope.decision_arrays(row, 3)
            self.assertTrue(np.array_equal(ope.policy_tau(row, ope.INF), d['reference']))
            self.assertEqual(ope.gint(row, d, TAU, 1e-12, 1e-12), 0.)
        scan = self.stage.scan()
        self.assertEqual((scan.identity['rows_checked'], scan.identity['max_abs_diff_candidate'],
                          scan.identity['positivity_rows_checked']), (len(rows), 0., 1))
        other = SyntheticStage(self.root / 'other', tau=.11, name='S6-tau011')  # candidate built at another tau
        for _ in range(2):  # deterministic refusal
            with self.assertRaisesRegex(pa.IntegrityError, 'G-INT .*kl_policy'):
                other.scan()
        swapped = copy.deepcopy(rows[0])  # the candidate is NOT reproduced from q_reference
        swapped['result']['q_planning'] = swapped['result']['q_reference']
        self.assertFalse(np.array_equal(ope.policy_tau(swapped, TAU), np.array(rows[0]['result']['candidate'])))

    def test_positivity_row_gint_path(self):
        row = next(r for r in self.result_rows() if r['status'] == pr.LOGGING_POSITIVITY)
        d = ope.decision_arrays(row, 3)
        self.assertEqual((d['b_a'], d['candidate'][d['a']], d['rho_candidate'], d['rho_reference']), (0., 0., 0., 0.))
        self.assertEqual(ope.gint(row, d, TAU, 1e-12, 1e-12), 0.)  # no division on a positivity row
        bad = copy.deepcopy(d)
        bad['rho_reference'] = 1e-300
        with self.assertRaisesRegex(pa.IntegrityError, 'positivity row needs'):
            ope.gint(row, bad, TAU, 1e-12, 1e-12)
        supported = next(r for r in self.result_rows() if r['status'] == pr.SUPPORTED)
        d = ope.decision_arrays(supported, 3)
        drifted = {**d, 'rho_candidate': d['rho_candidate'] * (1 + 1e-10)}  # relative 1e-10 > rtol 1e-12
        with self.assertRaisesRegex(pa.IntegrityError, 'rho_candidate'):
            ope.gint(supported, drifted, TAU, 1e-12, 1e-12)
        tied = copy.deepcopy(supported)  # further exact zeros are CRN ties: allowed and counted
        mask = tied['result']['mask']
        tied['result']['q_planning_diff_se'] = [0. if m else None for m in mask]
        self.assertEqual(ope.decision_arrays(tied, 3)['se_ties'], sum(mask) - 1)
        no_zero = copy.deepcopy(supported)  # no zero at the argmax of q_planning: INVALID
        no_zero['result']['q_planning_diff_se'] = [.1 if m else None for m in mask]
        with self.assertRaisesRegex(pa.IntegrityError, 'must be 0 at the argmax'):
            ope.decision_arrays(no_zero, 3)
        negative = copy.deepcopy(supported)
        negative['result']['q_planning_diff_se'] = [(-.1 if i != ope.decision_arrays(supported, 3)['best'] else 0.)
                                                    if m else None for i, m in enumerate(mask)]
        with self.assertRaisesRegex(pa.IntegrityError, 'nonnegative'):
            ope.decision_arrays(negative, 3)
        with self.assertRaisesRegex(pa.IntegrityError, 'q_source'):
            ope.Scan(taus=TAUS, sealed_tau=TAU, identity_tolerance=1e-12, ratio_rtol=1e-12,
                     statuses=list(ope.RESULT_STATUSES), families=['temperature'], margins=MARGINS,
                     expected_q_source='planning_reuse').run(self.stage.ledger.iter_decisions(), self.stage.pas())

    def test_enumerated_toy_unbiasedness_and_selection_bias(self):
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
                dr = ips = conditional = kept = null_dr = null_ips = 0.
                for a in range(5):
                    d = toy_decision(mask, logging, reference, q_hat, a, q_pi)
                    for g, p in zip(*outcomes[a]):
                        model, correction, ips_term, _ = ope.onestep_terms(pi, d, g)
                        dr += logging[a] * p * (model + correction)
                        ips += logging[a] * p * ips_term
                        if mask[a]:  # what dropping the OUTSIDE_POLICY_SUPPORT action would estimate
                            conditional += logging[a] * p * (model + correction)
                            kept += logging[a] * p
                    for g, p in marginal:  # an outcome independent of the action: true advantage 0
                        model, correction, ips_term, _ = ope.onestep_terms(pi, d, g)
                        null_dr += logging[a] * p * (model + correction)
                        null_ips += logging[a] * p * ips_term
                self.assertAlmostEqual(dr, truth, places=14)
                self.assertAlmostEqual(ips, truth, places=14)
                self.assertAlmostEqual(null_dr, 0., places=14)
                self.assertAlmostEqual(null_ips, 0., places=14)
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
                self.assertAlmostEqual(model, tau * (ope.kl(pi, ref, mask) + ope.kl(ref, pi, mask)), places=14)

    def test_off_mask_rows_keep_only_the_model_term(self):
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
        bad = copy.deepcopy(outside)
        bad['result']['q_planning'][bad['result']['logged_index']] = .4
        with self.assertRaisesRegex(pa.IntegrityError, 'exactly on the mask'):
            ope.decision_arrays(bad, 3)
        with self.assertRaisesRegex(pa.IntegrityError, 'no recorded result'):
            ope.decision_arrays(next(r for r in self.stage.decisions if r['status'] == pr.MID_PA), 3)

    def test_shrinkage_is_kl_policy_at_tau_over_c(self):
        mask = np.array([True, True, True])
        q, se = np.array([.4, .5, .7]), np.array([.05, .02, 0.])  # argmax 2 carries the zero SE
        sigma2 = (.05 ** 2 / 2 + .02 ** 2 / 2) / 2
        expected = 1 - sigma2 / np.var(q, ddof=1)
        self.assertAlmostEqual(ope.shrink_factor(q, se, mask), expected, places=15)
        d = toy_decision(mask, [.3, .3, .4], [.3, .3, .4], q, 0, q)
        d['se'] = se
        c = ope.shrink_factor(q, se, mask)
        for tau in (.5, .1, .02):
            self.assertTrue(np.array_equal(ope.family_policy('shrinkage', d, tau), ope.tilt(q, d['reference'], mask,
                                                                                            tau / c)))
            fq = (d['q_pi'], d['q_dr'], 1., False)  # c = 1 reproduces T bit for bit
            self.assertTrue(np.array_equal(ope.family_policy('shrinkage', d, tau, fq),
                                           ope.family_policy('temperature', d, tau)))
        noisy = np.array([1., 1., 0.])  # sigma2 far above the spread: c = 0, the reference exactly
        self.assertEqual(ope.shrink_factor(q, noisy, mask), 0.)
        d['se'] = noisy
        self.assertTrue(np.array_equal(ope.family_policy('shrinkage', d, .02), d['reference']))
        self.assertEqual(ope.shrink_factor(np.array([.5]), np.array([0.]), np.array([True])), 0.)  # mask < 2
        self.assertEqual(ope.shrink_factor(np.array([.5, .5]), np.array([0., .1]), mask[:2]), 0.)  # var 0
        rng = np.random.default_rng(1)
        for _ in range(50):
            k = int(rng.integers(2, 6))
            self.assertTrue(0. <= ope.shrink_factor(rng.random(k), rng.random(k), np.ones(k, dtype=bool)) <= 1.)

    def test_disagreement_margins_and_ties(self):
        mask = np.array([True, True, True, False])
        ref = np.array([.5, .3, .2, 0.])
        se = np.array([.004, .0, .01, np.nan])  # se[ref_best = 0] = 0.004, 0 at the q argmax 1
        q = np.array([.1, .9, .2, np.nan])
        self.assertEqual(ope.disagreement(q, ref, mask, se, MARGINS), ([True, True, True], False, False, False))
        close = np.array([.5, .503, .2, np.nan])  # gap .003: in D_0 only
        self.assertEqual(ope.disagreement(close, ref, mask, se, MARGINS), ([True, False, False], False, False, False))
        mid = np.array([.5, .5045, .2, np.nan])  # gap .0045 > se .004 but < .005
        self.assertEqual(ope.disagreement(mid, ref, mask, se, MARGINS), ([True, False, True], False, False, False))
        self.assertEqual(ope.disagreement(np.array([.9, .1, .2, np.nan]), ref, mask, se, MARGINS),
                         ([False] * 3, False, False, False))
        self.assertEqual(ope.disagreement(np.array([.1, .9, .9, np.nan]), ref, mask, se, MARGINS),
                         ([False] * 3, True, False, False))
        self.assertEqual(ope.disagreement(q, np.array([.4, .4, .2, 0.]), mask, se, MARGINS),
                         ([False] * 3, False, True, False))
        crn = ope.disagreement(q, ref, mask, np.array([0., 0., .01, np.nan]), MARGINS)  # an exact CRN tie: se_pair 0
        self.assertEqual(crn, ([True, True, True], False, False, True))  # D_1se = D_0 for this row, counted


def flat_table(n=300, v=5, seed=2):
    """A small observed table with only what FlatRows / common_tau read."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, v)) < .7
    mask[:, 0] = True
    ref = np.where(mask, rng.random((n, v)) + .05, 0.)
    ref /= ref.sum(axis=1, keepdims=True)
    a = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
    logging = .8 * ref + .2 / v  # some logging mass off the mask
    logging /= logging.sum(axis=1, keepdims=True)
    return {'observed': np.ones(n, dtype=bool), 'mask': mask, 'ref': ref, 'q_pi': np.where(mask, rng.random((n, v)), np.nan),
            'q_dr': np.where(mask, rng.random((n, v)), np.nan), 'a': a, 'on_mask': np.ones(n, dtype=bool),
            'b_a': logging[np.arange(n), a], 'sha8': rng.integers(2 ** 32, size=n), 'logging': logging}


class PlaceboTests(StageCase):
    def test_perm_rng_flat_mirror_and_common_tau(self):
        row = self.result_rows()[0]
        sha8 = int(row['sha256'][:8], 16)
        q, mask = np.array([.1, .2, .3, .4, .5, np.nan]), np.array([True] * 5 + [False])
        a = ope.permuted(q, mask, 4255355903, 0, sha8)
        self.assertEqual(sorted(a[mask]), sorted(q[mask]))
        self.assertTrue(np.array_equal(a, ope.permuted(q, mask, 4255355903, 0, sha8), equal_nan=True))
        expected = np.random.default_rng(np.random.SeedSequence([4255355903, 3, sha8])).permutation(5)
        self.assertTrue(np.array_equal(ope.permuted(q, mask, 4255355903, 3, sha8)[mask], q[mask][expected]))
        table = flat_table()
        flat = ope.FlatRows(table)
        for tau in (.5, .07, .02):  # the flat mirror is kl_policy
            pi = flat.tilt(flat.q_pi, tau)
            worst = 0.
            for i in range(flat.n):
                ref_pi = ope.tilt(table['q_pi'][i], table['ref'][i], table['mask'][i], tau)
                s, m = flat.starts[i], flat.lengths[i]
                worst = max(worst, float(np.abs(pi[s:s + m] - ref_pi[table['mask'][i]]).max()))
            self.assertLessEqual(worst, 1e-15)
        k0 = flat.permutation(4255355903, 0)  # the flat permutation is the per-row registered RNG
        for i in range(5):
            s, m = flat.starts[i], flat.lengths[i]
            expected = ope.permuted(table['q_pi'][i], table['mask'][i], 4255355903, 0, table['sha8'][i])
            self.assertTrue(np.array_equal(flat.q_pi[k0][s:s + m], expected[table['mask'][i]]))
        perms = np.stack([flat.permutation(4255355903, k) for k in range(8)])
        target = flat.mean_kl(flat.tilt(flat.q_pi, .05))
        tau_p, value, status = ope.common_tau(flat, perms, target, .05, factor=100, iterations=60, tol=.01)
        self.assertEqual(status, 'MATCHED')
        self.assertLessEqual(abs(value - target), .01 * target)  # one temperature for every draw, within 1%
        self.assertTrue(.05 / 100 <= tau_p <= .05 * 100)
        tau_p, _, status = ope.common_tau(flat, perms, 50., .05, factor=100, iterations=60, tol=.01)
        self.assertEqual((status, tau_p), ('NOT_MATCHABLE', .05))  # outside the attainable range: tau_p = tau
        self.assertEqual(ope.common_tau(flat, perms, 0., .05, factor=100, iterations=60, tol=.01)[2], 'MATCHED')

    def test_outcome_shuffle_null_has_power_against_a_logging_misspecification(self):
        def suspect(shift):
            table = power_table(shift)
            eng = ope.Engine(table, draws=400, seeds={'game': 5, 'pitcher': 6}, invalid_share_max=.05,
                             minima={'games': 1, 'pas': 1, 'pitchers': 1})
            accs, info = ope.outcome_shuffle_null(eng, draws_k=20, seed=3477356613, subsets=['all', 'D:1se'])
            flags = []
            for (variant, sub), acc in accs.items():
                cell = acc.cell().at(1, eng.draws, eng.share, '0.1')
                flags.append(all(cell['delta'][u]['ci95'] is not None and (cell['delta'][u]['ci95'][0] > 0 or
                                                                           cell['delta'][u]['ci95'][1] < 0)
                                 for u in ope.UNITS))
            return any(flags), info
        excludes, info = suspect(0.)
        self.assertFalse(excludes)  # the correct logging law: the descriptive null CI covers 0
        self.assertEqual((info['pas_keeping_own_dG'], info['label']), (0, 'descriptive (sets no label)'))
        excludes, _ = suspect(.25)
        self.assertTrue(excludes)  # logging mass moved between two actions: the descriptive null moves off 0

    def test_shuffle_donors_come_from_other_games_of_the_same_pitcher(self):
        pitcher = np.array(['a', 'a', 'a', 'b', 'b', 'c'], dtype=object)
        game = np.array([1, 1, 2, 1, 1, 3])
        usable = np.array([True, True, True, True, True, True])
        cands = ope.other_game_candidates(pitcher, game, usable)
        self.assertEqual({k: v.tolist() for k, v in cands.items()}, {0: [2], 1: [2], 2: [0, 1]})  # b, c keep own dG
        usable[2] = False  # without WE_start: neither donor nor receiver
        self.assertEqual(ope.other_game_candidates(pitcher, game, usable), {})


def power_table(shift, seed=11, n_pa=1500, games=60, pitchers=40):
    """True one-step advantage 0 (G independent of the action), actions drawn from b = [.5, .3, .2]; the recorded
    logging law (and habit) moves ``shift`` of mass from action 0 to action 1."""
    rng = np.random.default_rng(seed)
    v, per_pa = 3, 2
    n = n_pa * per_pa
    pa_index = np.repeat(np.arange(n_pa), per_pa)
    b_true = np.array([.5, .3, .2])
    logging = np.array([.5 - shift, .3 + shift, .2])
    a = rng.choice(v, size=n, p=b_true)
    mask = np.ones((n, v), dtype=bool)
    ref = np.tile(logging, (n, 1))
    q = np.array([.6, .4, .5]) + .02 * rng.standard_normal((n, v))
    taus = [ope.INF, .1]
    model, rho = np.zeros((n, 1, 2)), np.zeros((n, 1, 2))
    for j, tau in enumerate(taus):
        pi = ope.tilt_rows(q, ref, mask, tau)
        model[:, 0, j] = ((pi - ref) * q).sum(axis=1)
        rho[:, 0, j] = pi[np.arange(n), a] / logging[a]
    we_start = (.4 + .2 * rng.random(n_pa))[pa_index]
    reward = (we_start[::per_pa] + .3 + .1 * rng.standard_normal(n_pa))[pa_index]
    d = np.zeros((n, 3), dtype=bool)
    d[:, 0] = rng.random(n) < .5
    d[:, 2] = d[:, 0] & (rng.random(n) < .6)
    first = np.zeros(n, dtype=bool)
    first[::per_pa] = True
    return {'n': n, 'families': ['temperature'], 'taus': taus, 'margins': MARGINS, 'dr_uses_pi': [False],
            'observed': np.ones(n, dtype=bool), 'game': (pa_index % games) + 900000,
            'pitcher': (pa_index % pitchers).astype(str).astype(object), 'pa_index': pa_index, 'G': reward,
            'we_start': we_start, 'rho_ref': np.ones(n), 'on_mask': np.ones(n, dtype=bool), 'model_dr': model,
            'model_pi': model.copy(), 'rho': rho, 'qa': q[np.arange(n), a][:, None], 'D': d,
            'strikes': np.zeros(n, dtype=np.int64), 'pa_pos': np.zeros(n, dtype=bool), 'first_D0': d[:, 0] & first,
            'first_eligible': first, 'status': np.array([pr.SUPPORTED] * n, dtype=object)}


class BootstrapTests(StageCase):
    def engine(self, draws=300):
        table = ope.attach_rewards(self.stage.scan().table(), self.stage.pas())
        return table, ope.Engine(table, draws=draws, seeds={'game': 5, 'pitcher': 6}, invalid_share_max=.5,
                                 minima={'games': 1, 'pas': 1, 'pitchers': 1})

    def test_cluster_bootstrap_is_deterministic_and_counts_invalid_replicates(self):
        keys = ['b', 'a', 'c', 'a', 'b', 'c', 'c']
        one, two = (ope.ClusterBootstrap(keys, unit='game', draws=500, seed=9) for _ in range(2))
        self.assertTrue(np.array_equal(one.m, two.m))
        self.assertEqual(one.order, ['a', 'b', 'c'])
        sample = np.random.default_rng(9).integers(3, size=(500, 3))  # the A2 draw
        self.assertTrue(np.array_equal(one.m, np.stack([np.bincount(s, minlength=3) for s in sample]).astype(float)))
        self.assertTrue(np.allclose(one.replicate(np.arange(7.)), one.m @ np.array([1 + 3, 0 + 4, 2 + 5 + 6])))
        table, eng = self.engine()
        positivity = eng.t['status'] == pr.LOGGING_POSITIVITY  # the one positivity row lies in game 201
        self.assertEqual(int(positivity.sum()), 1)
        validity = eng.validity('two_strike')
        self.assertTrue(np.array_equal(validity['units']['game'],
                                       eng.boots['game'].replicate((eng.t['strikes'] == 2).astype(float)) > 0))
        one_game = dict(table)  # D_0 kept only inside game 201: a replicate without game 201 is invalid
        one_game['D'] = table['D'].copy()
        one_game['D'][:, 0] &= table['game'] == 201
        self.assertGreater(int(one_game['D'][table['observed'], 0].sum()), 0)
        local = ope.Engine(one_game, draws=300, seeds={'game': 5, 'pitcher': 6}, invalid_share_max=1.,
                           minima={'games': 1, 'pas': 1, 'pitchers': 1})
        expected = int((local.boots['game'].m[:, local.boots['game'].order.index(201)] == 0).sum())
        self.assertGreater(expected, 0)
        self.assertEqual(local.record(local.family_cell('temperature', 'dr', 'D:0'))['0.1']['delta']['game'][
            'invalid_replicates'], expected)
        self.assertIsNone(ope.Engine(one_game, draws=300, seeds={'game': 5, 'pitcher': 6}, invalid_share_max=.05,
                                     minima={'games': 1, 'pas': 1, 'pitchers': 1}).record(
            local.family_cell('temperature', 'dr', 'D:0'))['0.1']['delta']['game']['ci95'])  # share > 0.05: INVALID
        cell = eng.family_cell('temperature', 'dr', 'all')
        again = self.engine()[1].family_cell('temperature', 'dr', 'all')
        for unit in ope.UNITS:
            self.assertTrue(np.array_equal(cell.reps[unit]['delta'], again.reps[unit]['delta'], equal_nan=True))
        strict = ope.Engine(table, draws=300, seeds={'game': 5, 'pitcher': 6}, invalid_share_max=.5,
                            minima={'games': 30, 'pas': 50, 'pitchers': 30})
        self.assertTrue(strict.validity('all')['below_minima'])  # a contrast-level fact, not an interval rule
        self.assertIsNotNone(strict.record(strict.family_cell('temperature', 'dr', 'all'))['0.1']['delta']['game']['ci95'])

    def test_paired_difference_of_identical_columns_and_cell_identities(self):
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
        self.assertEqual(r['inf']['R'], ope.R_UNDETERMINED)  # M = 0 at tau = inf: the denominator is not separated
        self.assertEqual(r['inf']['delta']['mean'], 0.)
        m, c, dlt = r['0.05']['M_dr']['mean'], r['0.05']['C']['mean'], r['0.05']['delta']['mean']
        self.assertAlmostEqual(m + c, dlt, places=15)
        self.assertGreater(r['0.05']['M_pi']['mean'], 0.)  # the KL argmax property of the temperature law
        for ratio, denominator in (('R', 'M_dr'), ('R_pi', 'M_pi')):  # a CI only with a separated denominator
            if ope._separated(r['0.05'][denominator]):
                self.assertAlmostEqual(r['0.05'][ratio]['mean'], dlt / r['0.05'][denominator]['mean'], places=12)
            else:
                self.assertEqual(r['0.05'][ratio], ope.R_UNDETERMINED)
        p_plus, p_minus = r['0.05']['delta']['game']['p_plus'], r['0.05']['delta']['game']['p_minus']
        self.assertAlmostEqual(p_plus + p_minus, 1., places=12)

    def test_r_is_undetermined_without_a_separated_denominator(self):
        _, eng = self.engine()
        cell = eng.family_cell('temperature', 'dr', 'all')
        j = TAUS.index(.05)
        straddle = ope.Cell({k: v.copy() for k, v in cell.point.items()},
                            {u: {k: v.copy() for k, v in reps.items()} for u, reps in cell.reps.items()},
                            cell.valid, dict(cell.info))
        for unit in ope.UNITS:  # M_dr replicates on both sides of 0 in the game bootstrap only
            straddle.reps[unit]['M_dr'][:, j] = np.linspace(-1, 1, eng.draws) if unit == 'game' else 1.
        at = straddle.at(j, eng.draws, eng.share, '0.05')
        self.assertEqual(at['R'], ope.R_UNDETERMINED)
        for unit in ope.UNITS:  # separated in both bootstraps: the R interval is reported
            straddle.reps[unit]['M_dr'][:, j] = np.linspace(.5, 1., eng.draws)
        self.assertIsInstance(straddle.at(j, eng.draws, eng.share, '0.05')['R'], dict)

    def test_max_t_uses_one_se_per_tau_on_a_hand_computable_toy(self):
        taus = [ope.INF, .1, .05]
        point = np.array([0., .2, .3])
        game = np.array([[0., .1, .3], [0., .3, .3], [0., .2, .1], [0., .2, .5]])
        pitcher = np.array([[0., 0., .25], [0., .4, .35], [0., .2, .3], [0., .2, .3]])
        valid = np.ones((4, 3), dtype=bool)
        cell = ope.Cell({'delta': point}, {'game': {'delta': game}, 'pitcher': {'delta': pitcher}},
                        {'game': {'delta': valid}, 'pitcher': {'delta': valid}}, {})
        out = ope.max_t(cell, taus, ['0.1', '0.05'], 4, 1., .025)
        se_game = np.array([np.sqrt(.02 / 3), np.sqrt(.08 / 3)])  # SD (ddof 1) of each bootstrap at tau .1, .05
        se_pitcher = np.array([np.sqrt(.08 / 3), np.sqrt(.005 / 3)])
        se = np.maximum(se_game, se_pitcher)  # = [sqrt(.08/3), sqrt(.08/3)]: SE_game != SE_pitcher at both taus
        self.assertEqual(list(out['se_tau'].values()), pytest_approx(se))
        t_obs = np.array([.2, .3]) / se  # [1.2247, 1.8371]
        self.assertAlmostEqual(out['t']['0.1'], 1.224744871391589, places=12)
        self.assertAlmostEqual(out['max_t'], 1.8371173070873836, places=12)
        null_game = ((game[:, 1:] - point[1:]) / se).max(axis=1)  # [0, .6124, 0, 1.2247]
        null_pitcher = ((pitcher[:, 1:] - point[1:]) / se).max(axis=1)  # [-.3062, 1.2247, 0, 0]
        self.assertTrue(np.allclose(null_game, [0., .6123724356957945, 0., 1.224744871391589]))
        self.assertTrue(np.allclose(null_pitcher, [-.30618621784789724, 1.224744871391589, 0., 0.]))
        self.assertEqual((out['game']['p_value'], out['pitcher']['p_value'], out['p']), (0., 0., 0.))
        self.assertTrue(out['significant'])
        lower = ope.Cell({'delta': np.array([0., .05, .06])}, cell.reps, cell.valid, {})  # small deltas: p > 0
        weak = ope.max_t(lower, taus, ['0.1', '0.05'], 4, 1., .025)
        t_weak = np.array([.05, .06]) / se
        expected = [float(np.mean(((r[:, 1:] - [.05, .06]) / se).max(axis=1) >= t_weak.max())) for r in (game, pitcher)]
        self.assertEqual((weak['game']['p_value'], weak['pitcher']['p_value']), tuple(expected))
        self.assertEqual(weak['p'], max(expected))

    def test_trend_max_t(self):
        _, eng = self.engine()
        cell = eng.family_cell('temperature', 'dr', 'all')
        used = [ope.tau_key(t) for t in TAUS[1:]]
        result = ope.trend(cell, TAUS, used, eng.draws, .5)
        x = np.log(1 / np.array(TAUS[1:]))
        self.assertAlmostEqual(result['slope'], float(np.polyfit(x, cell.point['delta'][1:], 1)[0]), places=12)
        self.assertIsNone(ope.trend(cell, TAUS, ['0.1'], eng.draws, .5)['slope'])  # one evaluable tau
        mt = ope.max_t(cell, TAUS, used, eng.draws, .5, .025)
        self.assertEqual(mt['p'], max(mt['game']['p_value'], mt['pitcher']['p_value']))
        self.assertEqual(mt['significant'], mt['p'] <= .025)

    def test_h_p_bounds_arithmetic_on_a_toy(self):
        # PA 0 observed (2 rows), PA 1 unobserved (2 SUPPORTED rows), PA 2 unobserved (1 off-mask row)
        taus = [ope.INF, .1]
        model = np.zeros((5, 1, 2))
        model[:, 0, 1] = [.01, .02, .03, -.01, .05]
        rho = np.ones((5, 1, 2))
        rho[:, 0, 1] = [1.5, .5, 2., .25, 0.]
        rho[4, 0, 0] = 0.
        table = {'families': ['temperature'], 'taus': taus, 'dr_uses_pi': [False], 'model_dr': model,
                 'model_pi': model.copy(), 'rho': rho, 'rho_ref': np.array([1., 1., 1., 1., 0.]),
                 'on_mask': np.array([True, True, True, True, False]), 'qa': np.array([[.4], [.5], [.3], [.6], [0.]]),
                 'G': np.array([.7, .7, np.nan, np.nan, np.nan]), 'observed': np.array([True, True, False, False, False]),
                 'pa_index': np.array([0, 0, 1, 1, 2])}
        b = ope.bounds(table, 'temperature')
        observed = (.01 + .5 * (.7 - .4)) + (.02 - .5 * (.7 - .5))
        c0 = 1. * (0 - .3) + (-.75) * (0 - .6)  # PA 1: sum (rho' - rho_r)(g - q_dr[a]) at g = 0 and g = 1
        c1 = 1. * (1 - .3) + (-.75) * (1 - .6)
        lo = (observed + (.03 - .01) + min(c0, c1) + .05) / 5
        hi = (observed + (.03 - .01) + max(c0, c1) + .05) / 5
        self.assertAlmostEqual(b['bounds']['0.1'][0], lo, places=15)
        self.assertAlmostEqual(b['bounds']['0.1'][1], hi, places=15)
        self.assertEqual((b['unobserved_decisions'], b['unobserved_pas']), (3, 2))


def decision_inputs(**contrast_overrides):
    base = {'subset': 'all', 'evaluable': True, 'not_evaluable_reasons': [], 'delta': .002,
            'se': {'game': .0007, 'pitcher': .0008}, 'ci95': {'game': [.0006, .0034], 'pitcher': [.0004, .0036]},
            'p_plus': {'game': .001, 'pitcher': .002}, 'p_minus': {'game': .999, 'pitcher': .998},
            'perm_paired_ci': {'game': [.0001, .003], 'pitcher': [.0002, .003]},
            's_pos': {'delta': .0019, 'ci': {'game': [.0003, .0035], 'pitcher': [.0002, .0036]}}}
    contrasts = {'P': copy.deepcopy(base), 'P_D': {**copy.deepcopy(base), 'subset': 'D:1se'}}
    for name, override in contrast_overrides.items():
        contrasts[name].update(override)
    return {'study': {'estimator_suspect': False, 'reasons': []}, 'contrasts': contrasts,
            'max_t': {'E': ['0.5', '0.2', '0.1'], 'p': .5, 'significant': False},
            'path_ii': {k: {'studentized': v, 'perm_paired_ci': {'game': [.0001, .003], 'pitcher': [.0002, .003]},
                            's_pos': {'delta': .002, 'ci': {'game': [.0003, .004], 'pitcher': [.0002, .004]}}}
                        for k, v in (('0.5', 1.), ('0.2', 2.5), ('0.1', 2.5))},
            'alpha': .025, 'mde_multipliers': {'one_sided_alpha': 2.8016, 'holm_first_step': 3.083},
            'primary_tau': '0.1', 'stage_label': 'S6_EXPLORATORY', 'any_main_family_tau_evaluable': True}


GAIN, HARM = 'ONESTEP_GAIN_S6_EXPLORATORY', 'ONESTEP_HARM_S6_EXPLORATORY'


class DecideTests(unittest.TestCase):
    def test_every_label_branch(self):
        out = ope.decide(decision_inputs())
        self.assertEqual([out['contrasts'][n]['label'] for n in ('P', 'P_D')], [GAIN, GAIN])
        self.assertEqual((out['stage2']['enter'], [c['contrast'] for c in out['stage2']['contrasts']]),
                         (True, ['P', 'P_D']))
        self.assertEqual(out['holm']['plus']['steps'][0]['threshold'], .0125)  # alpha / 2 first, then alpha
        holm = ope.decide(decision_inputs(P={'p_plus': {'game': .02, 'pitcher': .02}}))
        self.assertEqual(holm['holm']['plus']['rejected'], ['P', 'P_D'])  # .002 <= .0125, then .02 <= .025
        stop = ope.decide(decision_inputs(P={'p_plus': {'game': .02, 'pitcher': .02}},
                                          P_D={'p_plus': {'game': .02, 'pitcher': .03}}))
        self.assertEqual(stop['holm']['plus']['rejected'], [])  # the smallest .02 > .0125: Holm stops
        self.assertEqual(stop['contrasts']['P']['label'], 'NO_DIFFERENCE_ESTABLISHED')
        self.assertAlmostEqual(stop['contrasts']['P']['mde']['x2.8016'], 2.8016 * .0008)
        self.assertAlmostEqual(stop['contrasts']['P']['mde']['x3.083'], 3.083 * .0008)
        self.assertEqual([step['rejected'] for step in stop['holm']['plus']['steps']], [False])  # step-down stops
        sensitive = ope.decide(decision_inputs(P={'perm_paired_ci': {'game': [-.001, .003], 'pitcher': [.0002, .003]}}))
        self.assertEqual(sensitive['contrasts']['P']['label'], 'NO_DIFFERENCE_ESTABLISHED:SENSITIVE')
        spos = ope.decide(decision_inputs(P_D={'s_pos': {'delta': -.001, 'ci': {'game': [-.002, .001],
                                                                                  'pitcher': [-.002, .001]}}}))
        self.assertEqual(spos['contrasts']['P_D']['label'], 'NO_DIFFERENCE_ESTABLISHED:SENSITIVE')
        harm_side = {'delta': -.002, 'p_plus': {'game': .999, 'pitcher': .998}, 'p_minus': {'game': .001, 'pitcher': .002},
                     's_pos': {'delta': -.0019, 'ci': {'game': [-.0035, -.0003], 'pitcher': [-.0036, -.0002]}}}
        harm = ope.decide(decision_inputs(P=harm_side))
        self.assertEqual(harm['contrasts']['P']['label'], HARM)
        self.assertEqual(harm['study_summary'], ['HARM_ESCALATE'])
        self.assertEqual((harm['flags'], harm['stage2']['enter']), (['HARM_ESCALATE'], False))
        weak_harm = ope.decide(decision_inputs(P={**harm_side, 's_pos': {'delta': .001, 'ci': {
            'game': [-.001, .002], 'pitcher': [-.001, .002]}}}))
        self.assertEqual(weak_harm['contrasts']['P']['label'], 'NO_DIFFERENCE_ESTABLISHED:SENSITIVE')
        gated = ope.decide(decision_inputs(P={'evaluable': False, 'not_evaluable_reasons': ['eligibility gate failed']}))
        self.assertEqual(gated['contrasts']['P']['label'], 'NOT_EVALUABLE:P')
        self.assertEqual(gated['holm']['plus']['steps'][-1]['contrast'], 'P')  # a non-evaluable contrast has p = 1
        suspect = ope.decide({**decision_inputs(), 'study': {'estimator_suspect': True, 'reasons': ['shuffle']}})
        self.assertEqual((suspect['study_label'], suspect['stage2']['enter']), ('ESTIMATOR_SUSPECT', False))
        invalid = ope.decide({**decision_inputs(), 'study': {'invalid': True, 'estimator_suspect': True}})
        self.assertEqual(invalid['study_label'], 'INVALID')
        null = {'p_plus': {'game': .4, 'pitcher': .45}, 'p_minus': {'game': .6, 'pitcher': .55}}
        quiet = ope.decide(decision_inputs(P=null, P_D=null))
        self.assertEqual((quiet['stage2']['enter'], quiet['stage2']['reason']), (False, 'no GAIN and path (ii) not applied'))
        self.assertEqual(quiet['path_ii']['reason'], 'max-t not significant')
        maxt = decision_inputs(P=null, P_D=null)
        maxt['max_t'] = {'E': ['0.5', '0.2', '0.1'], 'p': .01, 'significant': True}
        chosen = ope.decide(maxt)
        self.assertTrue(chosen['stage2']['enter'])
        self.assertEqual(chosen['stage2']['contrasts'][0]['tau'], '0.2')  # tie 2.5 = 2.5 -> the larger tau
        no_perm = copy.deepcopy(maxt)
        no_perm['path_ii']['0.2']['perm_paired_ci'] = {'game': [-.001, .003], 'pitcher': [.0002, .003]}
        blocked = ope.decide(no_perm)
        self.assertEqual((blocked['stage2']['enter'], blocked['path_ii']['reason']),
                         (False, 'perm or S-pos condition failed at the chosen tau'))
        no_spos = copy.deepcopy(maxt)
        no_spos['path_ii']['0.2']['s_pos'] = {'delta': .002, 'ci': {'game': [-.001, .004], 'pitcher': [.0002, .004]}}
        self.assertFalse(ope.decide(no_spos)['stage2']['enter'])
        single = copy.deepcopy(maxt)
        single['max_t']['E'] = ['0.1']
        self.assertEqual(ope.decide(single)['path_ii']['reason'], '|E| < 2: path (ii) not applied')

    def test_holm_step_down_and_gain_with_harm(self):
        self.assertEqual(ope.holm({'P': .01, 'P_D': .02}, .025)['rejected'], ['P', 'P_D'])
        self.assertEqual(ope.holm({'P': .013, 'P_D': .02}, .025)['rejected'], [])  # the smaller p > 0.0125: stop
        self.assertEqual(ope.holm({'P': .001, 'P_D': .03}, .025)['rejected'], ['P'])
        self.assertEqual(ope.holm({'P': 1., 'P_D': .012}, .025)['rejected'], ['P_D'])  # p = 1: never rejected
        steps = ope.holm({'P': 1., 'P_D': .012}, .025)['steps']
        self.assertEqual([(st['contrast'], st['threshold']) for st in steps], [('P_D', .0125), ('P', .025)])
        harm_side = {'delta': -.002, 'p_plus': {'game': .999, 'pitcher': .998}, 'p_minus': {'game': .001, 'pitcher': .002},
                     's_pos': {'delta': -.0019, 'ci': {'game': [-.0035, -.0003], 'pitcher': [-.0036, -.0002]}}}
        mixed = ope.decide(decision_inputs(P_D=harm_side))  # GAIN on P and HARM on P_D at the same time
        self.assertEqual([mixed['contrasts'][n]['label'] for n in ('P', 'P_D')], [GAIN, HARM])
        self.assertEqual((mixed['flags'], mixed['stage2']['enter']), (['HARM_ESCALATE'], False))
        gated = ope.decide(decision_inputs(P_D={'evaluable': False, 'not_evaluable_reasons': ['gate']}))
        self.assertEqual((gated['contrasts']['P_D']['label'], gated['contrasts']['P']['label']),
                         ('NOT_EVALUABLE:P_D', GAIN))  # P alone at alpha/2 = 0.0125


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
        defense = lambda s, h: float(we.predict_defense(s, h))
        facts = ope.frame_facts(frame, blocks, preq.game_table(frame, set(GAMES)), defense)
        self.assertEqual(list(facts), list(self.stage.facts))
        from pitchmdp.game import GameState
        for pa_id, info in self.stage.facts.items():
            for key in ('game', 'reward', 'kind', 'reason', 'end', 'end_kind', 'flags', 'start_pitcher', 'date'):
                self.assertEqual(facts[pa_id][key], info[key], (pa_id, key))
            first = frame.iloc[info['positions'][0]]
            self.assertEqual(facts[pa_id]['we_start'], defense(GameState.from_row(first), first.inning_topbot == 'Top'))
        sealed = ope.sealed_rows(pd.read_parquet(self.stage.dir / 'pa_values.parquet',
                                                 columns=list(ope.PHASE_A_COLUMNS)).to_dict('records'))
        values = ope.sealed_rows(pd.read_parquet(self.stage.dir / 'pa_values.parquet',
                                                 columns=list(ope.PHASE_B_COLUMNS)).to_dict('records'), ope.PHASE_B_COLUMNS)
        with self.assertRaisesRegex(pa.IntegrityError, 'value columns'):
            ope.FactsCheck('l1r', 1e-12).phase_b(next(iter(sealed)), [], {}, sealed[next(iter(sealed))])
        ope.add_value_columns(sealed, values)
        pas = {p: {**facts[p], 'in_population': sealed[p]['in_population']} for p in sealed}
        by = {p: rows for p, _, rows in est.by_pa(self.stage.ledger.iter_decisions(), pas)}

        def run(check, phase, facts_by_pa):
            for p in sealed:
                getattr(check, phase)(p, by[p], ope.estimator_info(sealed[p], facts_by_pa[p], len(by[p])), sealed[p])
            return check
        dr = json.loads((self.stage.dir / 'dr.json').read_text())
        record = run(run(ope.FactsCheck('l1r', 1e-12), 'phase_a', pas), 'phase_b', pas).record(dr)
        self.assertEqual((record['phase_a_pas'], record['phase_b_pas']), (len(sealed), len(sealed)))
        self.assertEqual(record['phase_b_max_abs_diff'], {'candidate': 0., 'reference': 0., 'delta': 0.})
        self.assertEqual(record['aggregate_check']['abs_diff'], 0.)  # the frame rewards reproduce the sealed L2 mean
        self.assertEqual(run(ope.FactsCheck('l1r', 1e-12), 'phase_a', pas).record()['aggregate_check'],
                         'not run (phase b only)')
        nudged = {p: {**f, 'reward': None if f['reward'] is None else (f['reward'] - 1e-6 if f['reward'] > .5
                                                                        else f['reward'] + 1e-6)} for p, f in pas.items()}
        run(ope.FactsCheck('l1r', 1e-12), 'phase_a', nudged)  # phase (a) never reads a G value
        with self.assertRaisesRegex(pa.IntegrityError, r'pa_values reproduction \(b\)'):
            run(ope.FactsCheck('l1r', 1e-12), 'phase_b', nudged)
        missing = dict(pas)
        p0 = next(p for p in sealed if sealed[p]['status'] == est.COMPLETE)
        missing[p0] = {**pas[p0], 'reward': None, 'kind': preq.NO_TERMINAL, 'reason': 'no_terminal_event'}
        with self.assertRaisesRegex(pa.IntegrityError, r'pa_values reproduction \(a\)'):
            run(ope.FactsCheck('l1r', 1e-12), 'phase_a', missing)
        self.assertNotIn('terminal_residual', ope.PA_VALUE_COLUMNS)
        self.assertFalse(any(c.endswith('_bounds') for c in ope.PA_VALUE_COLUMNS))

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
                               ('confirm', {'confirmatory': {'P': 'all', 'P_D': 'D:0'}}),
                               ('guards', {'guards': {'max_seconds': 600}})):
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
        failed = json.loads((out / 'check' / 'failed.json').read_text())
        self.assertIn('DR q-hat source', failed['error'])
        self.assertEqual(failed['study_label'], 'INVALID')

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
        self.assertEqual((counts['prediag_decisions'], counts['estimation_decisions'],
                          counts['prediag_minus_estimation']), (47, 43, 4))
        self.assertEqual(counts['eligible_by_status'], {pr.SUPPORTED: 38, pr.OUTSIDE_POLICY_SUPPORT: 4,
                                                        pr.LOGGING_POSITIVITY: 1})
        self.assertEqual(counts['excluded_decisions'], {
            'no_reward_pa:TERMINAL_VALUE_MISSING:end_state_gap': 4, f'status:{pr.MID_PA}': 16,
            f'status:{pr.MISSING_LABEL}': 2, f'status:{pr.NO_LOGGED_ACTION}': 2, f'status:{pr.UNKNOWN_PITCHER}': 1})
        self.assertEqual(counts['e0_pas_without_reward'], {'TERMINAL_VALUE_MISSING:end_state_gap': 1})
        self.assertEqual((counts['pas']['pas'], counts['pas']['e0'], counts['pas']['eligible_pas']), (16, 13, 12))
        self.assertEqual(counts['p_pos']['decision'], 1 / 43)
        identity = doc['provenance']['identity_check']
        self.assertEqual((identity['rows_checked'], identity['max_abs_diff_candidate']), (47, 0.))
        header = doc['provenance']['ledger']['header']
        self.assertEqual((header['dr_q']['source'], header['candidate_sha256']), ('evaluation_seed',
                                                                                  self.stage.identity['sha256']))
        crosscheck = doc['provenance']['pa_facts']['crosscheck']
        self.assertEqual((crosscheck['phase_a_pas'], crosscheck['phase_b_pas'], crosscheck['aggregate_check']),
                         (16, None, 'not run (phase b only)'))  # --check reads no G value
        self.assertEqual(doc['provenance']['interpretations'], ope.INTERPRETATIONS)
        self.assertNotIn('dose_response', doc)  # --check computes no G-dependent number
        prediag = json.loads((out / 'check' / 'prediag.json').read_text())['prediag']
        self.assertEqual((prediag['common']['estimation_eligible_decisions'], prediag['common']['all_kept_decisions'],
                          prediag['common']['all_minus_eligible']), (43, 47, 4))  # R4-4: eligible first, all rows second
        self.assertEqual(prediag['families']['temperature']['0.1']['decisions'], 43)
        self.assertLess(prediag['common']['p_D']['D:1se'], prediag['common']['p_D']['D:0'])
        gate = prediag['gates']['temperature']['0.1']['all']
        for key in ('ess_ratio', 'max_weight_share', 'games', 'pitchers', 'noise_ratio_q90', 'checks', 'pass'):
            self.assertIn(key, gate)
        self.assertTrue(gate['noise_exempt'])
        self.assertFalse(prediag['gates']['temperature']['0.05']['all']['noise_exempt'])
        row = prediag['families']['temperature']['0.1']
        self.assertGreater(row['mean_M_pi'], 0.)
        self.assertIn('pa_product_ess_descriptive', row)
        text = json.dumps(prediag)
        for forbidden in ('"delta"', '"C"', '"R"', 'reward', 'ci95'):  # no G-dependent quantity in prediag
            self.assertNotIn(forbidden, text)
        names = sorted(line.split('  ')[1] for line in (out / 'check' / 'SHA256SUMS').read_text().splitlines())
        self.assertEqual(names, ['check.json', 'prediag.json', 'started.json'])
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)  # never overwritten

    def test_run_writes_every_section_the_decision_and_matches_an_independent_recomputation(self):
        path, out = self.write_config('run', outputs={'decisions_parquet': True})
        self.assertEqual(runner.main(['--config', str(path), '--run'], env=self.env()), 0)
        doc = json.loads((out / 'run' / 'results.json').read_text())
        for section in ('provenance', 'counts', 'prediag', 'dose_response', 'subsets', 'placebos', 'sensitivities',
                        'bounds', 'trend', 'max_t', 'pa_level', 'decision'):
            self.assertIn(section, doc)
        self.assertTrue(doc['tau_inf_identity']['zero'])
        e0 = {r['pa_id'] for r in self.stage.rows if r['in_population']}
        rewards = {p: f['reward'] for p, f in self.stage.facts.items()}
        for tau in (.05, .02):
            values, models, models_pi, ips = [], [], [], []
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
                models_pi.append(float(((pi - ref) * np.where(mask, qp, 0.))[mask].sum()))
                ips.append(weight * g)
            cell = doc['dose_response']['temperature']['dr'][repr(tau)]
            self.assertEqual(cell['n'], 43)
            self.assertAlmostEqual(cell['delta']['mean'], float(np.mean(values)), places=13)
            self.assertAlmostEqual(cell['M_dr']['mean'], float(np.mean(models)), places=13)
            self.assertAlmostEqual(cell['M_pi']['mean'], float(np.mean(models_pi)), places=13)
            self.assertAlmostEqual(doc['dose_response']['temperature']['ips'][repr(tau)]['delta']['mean'],
                                   float(np.mean(ips)), places=13)
        crosscheck = doc['provenance']['pa_facts']['crosscheck']
        self.assertEqual((crosscheck['phase_a_pas'], crosscheck['phase_b_pas']), (16, 16))
        self.assertEqual(crosscheck['phase_b_max_abs_diff'], {'candidate': 0., 'reference': 0., 'delta': 0.})
        self.assertEqual(crosscheck['aggregate_check']['abs_diff'], 0.)
        decision = doc['decision']
        self.assertEqual(decision['interpretations'], ope.INTERPRETATIONS)
        self.assertIn(decision['contrasts']['P']['label'], ('ONESTEP_GAIN_S6_EXPLORATORY', 'ONESTEP_HARM_S6_EXPLORATORY',
                                                            'NO_DIFFERENCE_ESTABLISHED',
                                                            'NO_DIFFERENCE_ESTABLISHED:SENSITIVE'))
        self.assertEqual(decision['stage_label'], 'S6_EXPLORATORY')
        self.assertEqual(set(doc['provenance']['recorded_not_read']), set())
        self.assertEqual(set(decision['inputs']['contrasts']), {'P', 'P_D'})
        self.assertEqual(decision['inputs']['contrasts']['P_D']['subset'], 'D:1se')
        placebos = doc['placebos']
        self.assertEqual(placebos['perm']['tau_p']['0.1']['status'], 'MATCHED')
        self.assertIn('band', placebos['perm']['by_contrast']['P'])
        self.assertNotIn('band', placebos['neg']['P'])  # neg / inf: paired CI only
        shuffle = placebos['outcome_shuffle_null']
        self.assertEqual((shuffle['seed'], set(shuffle['by_contrast']['P'])), (3477356613, {'dr', 'ips'}))
        self.assertEqual(doc['bounds']['H_P']['tau'], '0.1')
        self.assertEqual(doc['bounds']['temperature']['unobserved_pas'], 1)
        self.assertIn('s_pos', doc['sensitivities'])
        rows = pd.read_csv(out / 'run' / 'per_tau.csv')
        self.assertEqual(len(rows) % len(GRID), 0)
        decisions = pd.read_parquet(out / 'run' / 'decisions.parquet')
        self.assertEqual(len(decisions), 43)
        self.assertTrue((decisions['delta_inf'] == 0).all())
        names = sorted(line.split('  ')[1] for line in (out / 'run' / 'SHA256SUMS').read_text().splitlines())
        self.assertEqual(names, ['decisions.parquet', 'per_tau.csv', 'prediag.json', 'results.json', 'started.json'])
        self.assertEqual(json.loads((out / 'run' / 'prediag.json').read_text())['prediag'], doc['prediag'])

    def test_registered_gate_marks_the_small_stage_not_evaluable(self):
        path, out = self.write_config('strict', gate=copy.deepcopy(STRICT),
                                      sample_minima={'games': 30, 'pas': 50, 'pitchers': 30})
        self.assertEqual(runner.main(['--config', str(path), '--run'], env=self.env()), 0)
        doc = json.loads((out / 'run' / 'results.json').read_text())
        self.assertFalse(doc['prediag']['gates']['temperature']['0.1']['all']['pass'])
        self.assertEqual([doc['decision']['contrasts'][n]['label'] for n in ('P', 'P_D')],
                         ['NOT_EVALUABLE:P', 'NOT_EVALUABLE:P_D'])
        self.assertFalse(doc['decision']['stage2']['enter'])
        self.assertIn('NOT_EVALUABLE_ALL', doc['decision']['study_summary'])
        self.assertTrue(doc['dose_response']['temperature']['dr']['0.1']['descriptive'])
        self.assertIsNone(doc['trend']['temperature']['slope'])

    def test_a_wrong_frame_reward_stops_invalid_after_prediag_before_estimates(self):
        class Nudged(SyntheticEnvironment):  # the frozen WE off by 1e-6: every next-row reward moves
            def load_frame_and_we(self, spec, checkpoint):
                frame, defense, config, provenance = super().load_frame_and_we(spec, checkpoint)
                return frame, (lambda state, home: defense(state, home) + 1e-6), config, provenance
        path, out = self.write_config('nudged')
        self.assertEqual(runner.main(['--config', str(path), '--run'], env=Nudged(self.stage.frame)), 2)
        failed = json.loads((out / 'run' / 'failed.json').read_text())
        self.assertEqual(failed['study_label'], 'INVALID')
        self.assertIn('pa_values reproduction (b)', failed['error'])
        self.assertTrue((out / 'run' / 'prediag.json').exists())  # written before phase (b)
        self.assertFalse((out / 'run' / 'results.json').exists())  # no estimate after a failed reproduction
        path, out = self.write_config('nudged-check')
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=Nudged(self.stage.frame)), 0)  # G-free

    def test_row_identity_passes_and_catches_a_broken_off_mask_rule(self):
        scan = self.stage.scan()
        rid = scan.row_identity
        rows = sum(r['status'] in ope.RESULT_STATUSES for r in self.stage.decisions)
        self.assertEqual((rid['rows'], rid['evaluations']), (rows, rows * 4 * len(TAUS) * len(ope.IDENTITY_G)))
        self.assertLessEqual(rid['max_abs'], 1e-12)
        setup = ope.perm_setup(scan.table(), draws_k=3, seed=4255355903, kl_tol=.01, factor=100, iterations=60)
        self.assertLessEqual(setup.identity['max_abs'], 1e-12)  # every perm / estimator_stress policy too
        self.assertIn('kl_ratio_D1se_placebo_over_real', setup.record['0.1'])
        original = ope.row_term
        try:  # off-mask rows lose their model term: the identity must fail
            ope.row_term = lambda model, rho, rho_ref, on, reward, qa: (np.where(on, model, 0.)
                                                                         + ope.row_correction(rho, rho_ref, on, reward, qa))
            with self.assertRaisesRegex(ope.EstimatorSuspect, 'row identity failed'):
                self.stage.scan()
            path, out = self.write_config('broken-identity')
            self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
            failed = json.loads((out / 'check' / 'failed.json').read_text())
            self.assertEqual(failed['study_label'], 'ESTIMATOR_SUSPECT')
            self.assertFalse((out / 'check' / 'prediag.json').exists())  # stops before prediag
        finally:
            ope.row_term = original

    def test_subset_gate_marks_p_d_not_evaluable(self):
        path, out = self.write_config('gate-pd', gate={**copy.deepcopy(OPEN), 'max_weight_share': .2})
        self.assertEqual(runner.main(['--run', '--config', str(path)], env=self.env()), 0)
        doc = json.loads((out / 'run' / 'results.json').read_text())
        gates = doc['prediag']['gates']['temperature']['0.1']
        self.assertTrue(gates['all']['pass'])  # max share over 43 decisions is small
        self.assertFalse(gates['D:1se']['pass'])  # a few D_1se decisions: one weight carries > 20%
        self.assertEqual(gates['D:1se']['population'], 'estimation-eligible')
        self.assertEqual(doc['decision']['contrasts']['P_D']['label'], 'NOT_EVALUABLE:P_D')
        self.assertNotEqual(doc['decision']['contrasts']['P']['label'], 'NOT_EVALUABLE:P')

    def test_registration_essentials_are_refused_when_null_or_partial(self):
        for name, override in (('null-source', {'source_commit': None}),
                               ('null-candidate', {'candidate_identity_sha256': None}),
                               ('no-selection', {'pa_facts': {**make_config(self.stage.dir, '/x')['pa_facts'],
                                                              'check_selection': False}}),
                               ('partial-statuses', {'eligibility': {'statuses': [pr.SUPPORTED], 'population': 'E0',
                                                                     'require_reward': True}}),
                               ('stage-label', {'stage_label': 'S6'})):
            path, out = self.write_config(name, **override)
            self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2, name)
            self.assertFalse(out.exists(), name)

    def test_phase_a_never_loads_the_value_columns(self):
        reads = []
        original = runner.pd.read_parquet

        def spy(source, columns=None, **kwargs):
            prediag_written = any(p.name == 'prediag.json' for p in out.rglob('prediag.json'))
            reads.append((tuple(columns or ()), prediag_written))
            return original(source, columns=columns, **kwargs)
        runner.pd.read_parquet = spy
        try:
            path, out = self.write_config('columns-check')
            self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 0)
            self.assertEqual(reads, [(ope.PHASE_A_COLUMNS, False)])  # --check: the 7 G-free columns only
            reads.clear()
            path, out = self.write_config('columns-run')
            self.assertEqual(runner.main(['--config', str(path), '--run'], env=self.env()), 0)
            self.assertEqual(reads, [(ope.PHASE_A_COLUMNS, False), (ope.PHASE_B_COLUMNS, True)])  # values after prediag
        finally:
            runner.pd.read_parquet = original
        for column in ('resolution', 'candidate', 'reference', 'delta', 'l2'):
            self.assertNotIn(column, ope.PHASE_A_COLUMNS)

    def test_a_broken_estimator_correction_fails_the_row_identity(self):
        original = ope.row_correction
        try:  # the estimator's own correction loses the reference ratio
            ope.row_correction = lambda rho, rho_ref, on, reward, qa: np.where(on, rho * np.where(on, reward - qa, 0.), 0.)
            with self.assertRaisesRegex(ope.EstimatorSuspect, 'row identity failed'):
                self.stage.scan()
        finally:
            ope.row_correction = original
        original = ope.ratio
        try:  # the estimator's own ratio reads the wrong action
            ope.ratio = lambda p, logging, a, on: original(p, logging, np.roll(np.atleast_1d(a), 1)
                                                           if np.ndim(a) else a, on)
            with self.assertRaisesRegex(ope.EstimatorSuspect, 'row identity failed'):
                self.stage.scan()
        finally:
            ope.ratio = original
        self.assertLessEqual(self.stage.scan().row_identity['max_abs'], 1e-12)  # restored

    def test_final_write_failure_leaves_failed_json_and_exit_2(self):
        original = runner.encode

        def failing(value):
            if 'decision' in value:
                raise pa.IntegrityError('not a JSON value: synthetic')
            return original(value)
        runner.encode = failing
        try:
            path, out = self.write_config('write-failure')
            self.assertEqual(runner.main(['--config', str(path), '--run'], env=self.env()), 2)
        finally:
            runner.encode = original
        failed = json.loads((out / 'run' / 'failed.json').read_text())
        self.assertEqual(failed['study_label'], 'INVALID')
        self.assertFalse((out / 'run' / 'results.json').exists())
        self.assertFalse((out / 'run' / 'SHA256SUMS').exists())

    def test_placebo_vectors_are_checked_against_kl_policy_at_run_time(self):
        path, out = self.write_config('kl-check')
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 0)
        check = json.loads((out / 'check' / 'prediag.json').read_text())['prediag']['perm_placebo_setup']
        record = check['row_identity']['kl_policy_check']
        self.assertEqual(record['perm_rows_checked'], 43)  # all rows when fewer than 500
        self.assertLessEqual(max(record['real_family_every_row_max_abs'], record['perm_vectors_max_abs']), 1e-12)
        original = ope.FlatRows.tilt
        try:  # a drifting vectorised placebo law is caught and stops the run
            ope.FlatRows.tilt = lambda self, q, tau: original(self, q, tau) * (1 + 1e-9)
            path, out = self.write_config('kl-broken')
            self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
            self.assertEqual(json.loads((out / 'check' / 'failed.json').read_text())['study_label'], 'ESTIMATOR_SUSPECT')
        finally:
            ope.FlatRows.tilt = original
        rows = flat_table(n=40)  # tilt_rows (the power-test helper) is kl_policy too
        for tau in (.5, .05):
            mirror = ope.tilt_rows(rows['q_pi'], rows['ref'], rows['mask'], tau)
            for i in range(40):
                self.assertLessEqual(float(np.abs(mirror[i] - ope.tilt(rows['q_pi'][i], rows['ref'][i], rows['mask'][i],
                                                                       tau)).max()), 1e-15)

    def test_vm_stat_parser_counts_free_inactive_and_speculative(self):
        text = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               '
                '1000.\nPages active:                            5000.\nPages inactive:                          '
                '2000.\nPages speculative:                        300.\nPages wired down:                         '
                '400.\n')
        self.assertEqual(runner.parse_vm_stat(text), (1000 + 2000 + 300) * 16384)
        self.assertIn('speculative', runner.Environment.MEMORY_FORMULA)

    def test_memory_start_condition(self):
        path, out = self.write_config('low-memory')
        stderr, old = io.StringIO(), sys.stderr
        sys.stderr = stderr
        try:
            code = runner.main(['--config', str(path), '--check'], env=SyntheticEnvironment(self.stage.frame, 2 ** 30))
        finally:
            sys.stderr = old
        self.assertEqual(code, 2)
        self.assertIn('below the registered start condition', stderr.getvalue())
        self.assertFalse(out.exists())  # refused before anything is written
        path, out = self.write_config('enough-memory')
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 0)
        started = json.loads((out / 'check' / 'started.json').read_text())
        self.assertEqual(started['guards']['available_memory_bytes_at_start'], 64 * 2 ** 30)
        self.assertEqual(started['guards']['available_memory_formula'], runner.Environment.MEMORY_FORMULA)
        self.assertEqual((started['git']['branch'], started['git']['worktree']), ('research/side-onestep-1029',
                                                                                  '/synthetic'))

    def test_the_registration_template_matches_the_schema(self):
        template = json.loads((HERE.parent / 'scripts' / 'run_ope_onestep.config-template.json').read_text())
        self.assertIsNone(template['output_dir'])
        self.assertIsNone(template['source_commit'])
        with self.assertRaises(pa.IntegrityError):  # output_dir stays null until the PM registers it
            runner.validate_config(template)
        with self.assertRaisesRegex(pa.IntegrityError, 'source_commit'):  # null source_commit is refused too
            runner.validate_config({**template, 'output_dir': str(self.root / 'template-out')})
        filled = {**template, 'output_dir': str(self.root / 'template-out'), 'source_commit': 'b' * 40}
        spec = runner.validate_config(filled)
        self.assertEqual((spec['stage_label'], spec['guards']['min_available_gb']), ('S6_EXPLORATORY', 6))
        self.assertEqual(set(spec['recorded_not_read']), {'ledger-dev-paired-identity.jsonl', 'processed_frame_aligned',
                                                          'processed_frame_aligned_identity_manifest'})
        self.assertEqual(template['interpretations'], ope.INTERPRETATIONS)
        with self.assertRaisesRegex(pa.IntegrityError, 'interpretations differ'):
            runner.validate_config({**filled, 'interpretations': {**ope.INTERPRETATIONS, 'holm_not_evaluable': 'drop'}})
        self.assertEqual(template['pins'], {
            'ledger-dev-0.jsonl': '4555989946a6f72fb593b3b8cdb905205b65338b007788b170206e04e208a4a0',
            'pa_values.parquet': '57284a76e26db6534d65f425f30ede8ef789841e97d4654d5038f781bb625db7',
            'dr.json': '5e56119ddff97215bea0e459af03257e72bdb55c6bd4d602e3bdba5568f27d38',
            'manifest.json': '0bf8981b5bcbefe33c30de614ad36c6a441e9be15f9bac16c0e4cf61fcadfd17',
            'started.json': '1fa9a703c82e59f045762ac094178ee7a47db69c59454a9f66d17274510a76b0'})
        self.assertEqual(len(spec['pa_facts']['policy_registration']['addenda']), 8)
        self.assertEqual(spec['guards'], {'max_seconds': 7200, 'max_rss_gb': 8, 'min_available_gb': 6})
        self.assertEqual(spec['placebos']['outcome_shuffle_null_seed'], 3477356613)
        self.assertEqual([ope.tau_key(t) for t in spec['taus']], ['inf', '0.5', '0.2', '0.1', '0.09', '0.07', '0.05',
                                                                   '0.02'])
        for link in [filled['pa_facts']['policy_registration']['config'],
                     *filled['pa_facts']['policy_registration']['addenda']]:  # the committed chain bytes
            self.assertEqual(sha(runner._resolve(link['path'])), link['sha256'], link['path'])

    def test_guards_abort_with_a_failure_record(self):
        path, out = self.write_config('guard')
        self.assertEqual(runner.main(['--config', str(path), '--run', '--max-seconds', '1e-9'], env=self.env()), 3)
        failed = json.loads((out / 'run' / 'failed.json').read_text())
        self.assertEqual((failed['error_type'], failed['study_label']), ('HangGuardExceeded', None))
        self.assertFalse((out / 'run' / 'results.json').exists())
        path, out = self.write_config('loose', guards={'max_seconds': 1e-9, 'max_rss_gb': 8, 'min_available_gb': 10})
        self.assertEqual(runner.main(['--config', str(path), '--run', '--max-seconds', '600'], env=self.env()), 3)
        self.assertEqual(json.loads((out / 'run' / 'started.json').read_text())['guards']['max_seconds'], 1e-9)
        path, out = self.write_config('tolerance', sealed_tau=.05, primary={'family': 'temperature', 'tau': .05,
                                                                              'subset': 'all', 'variant': 'dr'},
                                      gate={**copy.deepcopy(OPEN), 'noise_exempt': [{'family': 'temperature',
                                                                                     'tau': .05}]})
        self.assertEqual(runner.main(['--config', str(path), '--check'], env=self.env()), 2)
        self.assertIn('sealed tau', json.loads((out / 'check' / 'failed.json').read_text())['error'])


if __name__ == '__main__':
    unittest.main()
