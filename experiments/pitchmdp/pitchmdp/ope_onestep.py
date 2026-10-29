"""EXP-OPE-1STEP-01: one-step dose-response OPE on a sealed decision ledger (side experiment, spec v3, 2026-10-29).

Design: docs/pm/research/side-onestep-ope-proposal-2026-10-29.md, SPEC DELTA v2 (REVIEW-DESIGN-SONNET-001) and SPEC
DELTA 2 (REVIEW-DESIGN-SONNET-002). Pure functions over loaded inputs; scripts/run_ope_onestep.py owns the pins, the
frame/WE load, the files and the ``decide`` step. Sealed-path modules are imported, never modified.

Estimand. For a decision t (pre-decision history H_t), a target law pi and the recorded habit pi_ref (the masked TRAIN
BC ``reference``): A_t(pi) = sum_a [pi(a|H_t) - pi_ref(a|H_t)] Q_b(H_t, a), Q_b(H_t, a) = E[G | H_t, a_t = a, the rest
of the PA as logged]. G is the registered PA-end reward (policy_requests.pa_outcome, structural-end-v1, next-row state:
frozen C0 initial-defender WE, or the final W/L). The confirmatory contrasts are P = (temperature, tau 0.1, all, DR)
and P_D = (temperature, tau 0.1, D_1se, DR), WE per decision. Not a policy value, not the full-PA L1-R deviation, not a
causal effect; <=2025 DEV, exploratory.

Two Q arrays. q_pi = result['q_planning'] builds the policy families, D and the policy-side placebos: the recorded
candidate is P3 from the PLANNING Q (policy_runtime.build_runtime -> ``candidate``: ``q, diagnostics =
improvement.q_values(state)``; ``p = kl_policy(q, reference.probabilities(state), mask, tau)``; ``q`` is recorded as
``q_planning``, ``q_eval`` as ``q_reference``). q_dr = result['q_reference'] (the registered DR q-hat, evaluation seed,
M-7) is the DR model term and residual. M_dr = sum_mask (pi - reference) q_dr and M_pi = sum_mask (pi - reference) q_pi
(G-free) are both reported; R = mean D / mean M_dr (CI), R_pi = mean D / mean M_pi (descriptive).

G-INT (any violation is study-level INVALID and stops): kl_policy(q_pi, reference, mask, sealed tau) equals the recorded
candidate within the identity tolerance; where logging[a_t] > 0 the ratios pi(a_t)/b(a_t) and reference(a_t)/b(a_t)
equal the recorded rho_candidate / rho_reference with rtol 1e-12, atol 0; on a positivity row candidate[a_t] ==
reference[a_t] == 0 and both recorded rho are exactly 0.0 (no division); q_planning, q_reference and
q_planning_diff_se are finite exactly on the mask; q_planning_diff_se is 0 at the argmax of q_planning and >= 0
elsewhere (further exact zeros are CRN ties: allowed and counted); q_source is the registered DR q-hat source on every
row; the ledger header pins the candidate sha256. tau = inf is the reference itself (kl_policy rejects inf).

Row identity (deterministic estimator check; any violation is study-level ESTIMATOR_SUSPECT and stops, as is a non-zero
D at tau = inf): for every row with a result, every family x tau and every placebo policy, S_t = sum over actions a with
logging(a) > 0 of logging(a) D_t(a; G') with the estimator's own per-decision term (``row_term``) for a hypothetical
logged action a and a constant G' in {0.5, 0}; S_t = M_t + sum_mask (pi - ref)(G' - q_dr) = 0 exactly, so |S_t| <= 1e-12
is required (max reported). G-free: it runs in --check and before prediag.json.

Populations and row rules. An ESTIMATION-ELIGIBLE decision is a ledger row (i) of an E0 PA (sealed pa_values
``in_population``), (ii) whose PA-end reward G is observed, (iii) with a recorded result (SUPPORTED,
OUTSIDE_POLICY_SUPPORT, UNSUPPORTED_LOGGING_POSITIVITY): ``estimation_eligible(table)``. The prediag population drops
(ii) (independent of G); whole PAs without an observed reward enter only the worst-case H_P bounds with g in {0, 1}.
SUPPORTED: D_t = M_t + C_t, C_t = (rho' - rho_r)(G - q_dr[a_t]); OUTSIDE_POLICY_SUPPORT and LOGGING_POSITIVITY rows:
D_t = M_t (q_dr[a_t] is never read, no ratio is formed). All three stay in (dropping one would select on a_t;
enumerated toy test). Other statuses have no result at H_t: excluded with counts.

Families. temperature T: pi = kl_policy(q_pi, reference, mask, tau). shrinkage S (descriptive only: never in the
confirmatory family, max-t or the stage-2 selection): the per-decision temperature tau/c passed to kl_policy, c =
max(0, 1 - sigma2/var_mask(q_pi)), var the sample variance (ddof 1) over the mask actions, sigma2 = mean over the mask
actions other than the q_pi argmax of q_planning_diff_se^2 / 2; mask < 2 or var == 0 -> c = 0 -> the reference; c = 1
reproduces T bit for bit. cross_replicate (sensitivity): policy from q_dr, DR with q_pi. neg (placebo): -q_pi.

D_k (tau-free): argmax_mask q_pi != argmax_mask reference and q_pi[best] - q_pi[ref_best] > k, k in {0, 0.005 WE,
1se}; 1se = q_planning_diff_se[argmax reference on the mask] (> 0 there by the G-INT invariant); exact ties -> outside
every D. Smoke assertion: p(D_1se) < p(D_0) on the prediag population (else INVALID).

Eligibility gate (G-free, per family x tau x contrast subset, on the estimation-eligible decisions of the subset: P on
all, P_D within D_1se): decision Kish ESS ratio of rho' to rho_r >= 0.5, max normalised weight share max(rho')/sum(rho')
<= 0.01, games >= 30 and pitchers >= 30 (game from the pa_id 'game:at_bat'), noise ratio q90 of max_mask(c se)/tau <= 0.5
(the registered policy (T, 0.1) is exempt from the noise criterion; reported). Family x tau diagnostics and the
PA-product ESS (policy_tau.tau_table arithmetic, DESCRIPTIVE) are reported on the prediag population.

Inference: cluster bootstraps by game and by the PA's first-row pitcher (a PA lies in one game and one cluster; A2
ope_bootstrap.Resampler conventions: clusters sorted by str, default_rng(seed).integers(C, size=(draws, C))). One
multiplicity matrix per unit serves every statistic (paired across tau, families, variants and placebos). Denominators
are recounted per replicate; a replicate is invalid when the subset has no decision or the statistic is not finite; a
statistic whose invalid share exceeds the registered maximum is INVALID. Percentile 95% CI, SE = SD (ddof 1) of the
valid replicates; one-sided p+ = share(D* <= 0), p- = share(D* >= 0) of the valid replicates.

Placebos (policy side; q_dr fixed; D membership always from the real q_pi): perm (K draws, SeedSequence([perm_seed, k,
int(row sha256[:8], 16)]), ONE placebo temperature per tau bisected on [tau/100, 100 tau] so that the mean KL over the
draws and the estimation-eligible decisions matches T's on 'all'; outside the attainable range NOT_MATCHABLE with tau_p =
tau; the placebo/real KL ratio within D_1se is reported), estimator_stress (q_dr permuted alike), neg, tau = inf. The
outcome_shuffle_null is DESCRIPTIVE (it sets no label): dG = G - WE_start (frozen defensive WE of the PA's first-row
pre-pitch state) taken from a donor PA of the same first-row pitcher in ANOTHER game (drawn with replacement; a PA whose
pitcher has no other-game eligible PA keeps its own dG, counted), G' = WE_start(own) + dG(donor), K draws. Bands
(2.5-97.5% of the K draw means) only for perm and estimator_stress; neg/inf report the paired CI only. The
other-pitcher donor placebo is NOT_IMPLEMENTED. R = mean D / mean M_dr is reported with a CI only when the M_dr CI
excludes 0 in both bootstraps, else 'undetermined (denominator not separated)'.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np

from . import policy_estimator as est
from .policy_artifacts import IntegrityError, _require
from .policy_runtime import FATAL, LOGGING_POSITIVITY, OUTSIDE_POLICY_SUPPORT, SUPPORTED, Ledger
from .rollout_policy import kl_policy

CONTRACT = 'EXP-OPE-1STEP-01/onestep-v4'
INF = math.inf
Q_PI, Q_DR, SE_KEY = 'q_planning', 'q_reference', 'q_planning_diff_se'
RESULT_STATUSES = (SUPPORTED, OUTSIDE_POLICY_SUPPORT, LOGGING_POSITIVITY)
HALTING_KINDS = ('conflict', 'malformed', 'aborted')
MAIN_FAMILIES = ('temperature', 'shrinkage')
FAMILIES = ('temperature', 'shrinkage', 'cross_replicate', 'neg')
UNITS = ('game', 'pitcher')
CI_LEVELS = (0.025, 0.975)
STATS = ('delta', 'M_dr', 'M_pi', 'C', 'R', 'R_pi')
CONFIRMATORY = {'P': 'all', 'P_D': 'D:1se'}
STAGE_LABELS = ('S6_EXPLORATORY', 'DEV_STATISTICAL')  # stage 1 (this S6 reanalysis) and stage 2
INTERPRETATIONS = {  # the readings this implementation registers where the review text left a choice
    'noise_exemption': 'the noise-criterion exemption applies to the policy (temperature, tau 0.1), hence to both '
                       'confirmatory contrasts P and P_D (the ESS and cluster criteria still apply to each)',
    'stage2_tau_ranking': 'with no GAIN, path (ii) takes the tau in E with the largest max-t statistic t_tau = delta_hat '
                          '/ SE_tau, SE_tau = max(SE_game, SE_pitcher) (the same SE as max-t); ties -> the larger tau',
    'holm_not_evaluable': 'a NOT_EVALUABLE contrast enters Holm with p = 1; m stays 2',
    's_pos_rule': 'S-pos keeps the sign and its 95% CI excludes 0 in the same direction in both bootstraps',
    'reproduction_phase_a': 'before prediag only the 7 G-free pa_values columns are read and phase (a) classifies with '
                            'the reward value masked; the value columns and the PA-end values enter only after '
                            'prediag.json (phase b, attach_rewards)',
    'gate_population': 'every prediag metric and the gate are computed on the estimation-eligible decisions (P: all, '
                       'P_D: D_1se); the all-kept-rows count is reported second',
    'shuffle_donors': 'outcome_shuffle_null donors are drawn with replacement among the same pitcher\'s eligible PAs '
                      'in other games (never the PA itself or its game)',
    'path_ii': 'path (ii) needs |E| >= 2, a significant max-t, and at the chosen tau the perm paired CI > 0 and S-pos '
               'retention (subset all)'}
MASKED_REWARD = .5
IDENTITY_G = (.5, 0.)  # the constant outcomes of the row identity
IDENTITY_TOL = 1e-12
R_UNDETERMINED = 'undetermined (denominator not separated)'


class EstimatorSuspect(IntegrityError):
    """Study-level ESTIMATOR_SUSPECT: the deterministic row identity (or the tau = inf identity) failed; the run stops."""
ESTIMAND = {
    'confirmatory': 'P = (temperature, tau 0.1, all, DR), P_D = (temperature, tau 0.1, D_1se, DR): mean over the '
                    'estimation population of A_t = sum_a [pi(a) - reference(a)] Q_b(H_t, a); WE per decision',
    'pa_first_D:0': 'mean over PAs of the deviation at the first D_0 decision (0 without one); WE per PA',
    'populations': 'prediag: E0 rows with a recorded result (independent of G); estimation: those of PAs with an '
                   'observed PA-end reward',
    'row_rules': "SUPPORTED: D = M_dr + (rho' - rho_r)(G - q_dr[a]); off-mask rows: D = M_dr",
    'q_pi': Q_PI, 'q_dr': Q_DR, 'units': 'frozen C0 initial-defender WE',
    'not': 'not a policy value, not the full-PA L1-R deviation, not a causal effect; <=2025 DEV exploratory'}


# ---------------------------------------------------------------- grid, ledger, rows

def parse_tau(value):
    if value == 'inf':
        return INF
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0,
             f'tau must be "inf" or a positive finite number: {value!r}')
    return float(value)


def tau_key(tau):
    return 'inf' if tau == INF else repr(float(tau))


def check_grid(taus, sealed_tau):
    """The grid order is the boldness order: strictly decreasing from inf; the sealed tau is on it."""
    _require(len(taus) >= 2 and taus[0] == INF and all(a > b for a, b in zip(taus, taus[1:])),
             'tau grid must start at "inf" and decrease strictly (boldness order)')
    _require(sealed_tau != INF and sealed_tau in taus, 'the sealed candidate tau must be a finite grid value')
    return list(taus)


def margin_key(margin):
    """'1se' or the margin in WE written with %g (0 -> '0', 0.005 -> '0.005'): the D subset names are D:<key>."""
    return '1se' if margin == '1se' else f'{float(margin):g}'


def read_header(path):
    """The first ledger row only, verified as policy_runtime.Ledger._parse verifies every row; it must be the header."""
    with Path(path).open('rb') as stream:
        line = stream.readline()
    _require(line.endswith(b'\n'), 'ledger header missing or truncated')
    row = Ledger._parse(line, 0)
    _require(row.get('kind') == 'header' and row.get('seq') == 0 and row.get('prev') is None, 'ledger header missing')
    return row


class LedgerStream:
    """Read-only verified stream (the checks of policy_runtime.Ledger._stream: canonical bytes, own hash, seq, prev,
    newline-terminated rows) without constructing a Ledger, which would create a missing file. ``sha256`` is the hash of
    the bytes streamed, set when the stream is exhausted (compared with the pin by the caller)."""
    def __init__(self, path):
        self.path = Path(path)
        self.header, self.rows, self.head, self.sha256 = None, 0, None, None
        self.kinds = Counter()

    def decisions(self):
        digest, seq, prev = hashlib.sha256(), 0, None
        with self.path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                _require(line.endswith(b'\n'), 'ledger truncated mid-row; refuse to read')
                row = Ledger._parse(line, seq)
                _require(row.get('seq') == seq and row.get('prev') == prev,
                         f'ledger row {seq} tampered, reordered or not canonical')
                if seq == 0:
                    _require(row.get('kind') == 'header', 'ledger header missing')
                    self.header = row
                self.kinds[row['kind']] += 1
                seq, prev = seq + 1, row['sha256']
                if row['kind'] == 'decision':
                    yield row
        _require(seq > 0, 'empty ledger')
        self.rows, self.head, self.sha256 = seq, prev, digest.hexdigest()


def _exactly_on_mask(values, mask, name):
    _require(isinstance(values, list) and len(values) == len(mask), f'recorded {name} required (candidate runtime)')
    _require(all((v is None) != bool(m) for v, m in zip(values, mask)), f'{name} must be recorded exactly on the mask')
    out = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    _require(np.isfinite(out[mask]).all(), f'{name} must be finite on the mask')
    return out


def decision_arrays(row, n):
    """Validated arrays of a row with a recorded result: policy_estimator._step (positivity rows included) plus q_pi and
    the paired-difference SE, recorded exactly on the mask; the SE is 0 at the q_pi argmax (further zeros counted)."""
    status = row.get('status')
    _require(status in RESULT_STATUSES, f'{status}: no recorded result is expected for this status')
    policies, q_dr, mask, a, rho = est._step(row, n, positivity=True)
    result = row['result']
    q_pi = _exactly_on_mask(result.get(Q_PI), mask, Q_PI)
    se = _exactly_on_mask(result.get(SE_KEY), mask, SE_KEY)
    _require((se[mask] >= 0).all(), f'{SE_KEY} must be nonnegative')
    idx = np.flatnonzero(mask)
    best = int(idx[np.argmax(q_pi[idx])])
    _require(se[best] == 0, f'{SE_KEY} must be 0 at the argmax of {Q_PI} (index {best})')
    se_ties = int((se[idx] == 0).sum()) - 1  # further exact zeros: CRN ties, allowed and counted
    logging = np.asarray(result['logging'], dtype=np.float64)
    reference = policies['reference']
    on = bool(mask[a])
    rho_ref = ratio(reference, logging, a, on)
    return {'status': status, 'mask': mask, 'logging': logging, 'reference': reference, 'candidate': policies['candidate'],
            'q_dr': q_dr, 'q_pi': q_pi, 'se': se, 'best': best, 'se_ties': se_ties, 'a': int(a), 'on_mask': on,
            'b_a': float(logging[a]),
            'rho_ref': rho_ref, 'rho_candidate': result.get('rho_candidate'),
            'rho_reference': result.get('rho_reference'), 'q_source': result.get('q_source')}


# ---------------------------------------------------------------- policies, terms, D, placebo inputs

def tilt(q, reference, mask, tau):
    """pi_tau = rollout_policy.kl_policy(q, reference, mask, tau) with q = -inf off the mask (policy_tau._decision);
    tau = inf is the reference exactly (kl_policy rejects inf)."""
    if tau == INF:
        return np.array(reference, dtype=np.float64, copy=True)
    return kl_policy(np.where(mask, q, -np.inf), reference, mask, tau)


def tilt_rows(q, reference, mask, tau):
    """Row-wise mirror of kl_policy for already validated rows ([n, V]): the same arithmetic (probabilities
    renormalisation, log(bc) + (q - max q)/tau, exp(lw - max lw), normalise); checked against kl_policy in tests."""
    if tau == INF:
        return np.array(reference, dtype=np.float64, copy=True)
    bc = reference / reference.sum(axis=1, keepdims=True)
    qmax = np.where(mask, q, -np.inf).max(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        lw = np.where(mask, np.log(bc) + (q - qmax) / tau, -np.inf)
    w = np.where(mask, np.exp(lw - lw.max(axis=1, keepdims=True)), 0.)
    return w / w.sum(axis=1, keepdims=True)


def policy_tau(row, tau, n=None):
    """pi_tau of the temperature family for one ledger row (q_pi, recorded reference and mask)."""
    d = decision_arrays(row, len(row['result']['logging']) if n is None else n)
    return tilt(d['q_pi'], d['reference'], d['mask'], tau)


def shrink_factor(q, se, mask):
    """c = max(0, 1 - sigma2 / var_mask(q)); var = sample variance (ddof 1) over the mask actions, sigma2 = mean over the
    mask actions other than the q argmax of se^2 / 2 (the SE is 0 at the argmax); mask < 2 actions or var 0 -> 0."""
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return 0.
    qm = np.asarray(q, dtype=np.float64)[idx]
    var = float(np.var(qm, ddof=1))
    if not var > 0:
        return 0.
    others = np.delete(idx, int(np.argmax(qm)))
    sigma2 = float(np.mean(np.asarray(se, dtype=np.float64)[others] ** 2 / 2))
    return max(0., 1. - sigma2 / var)


def family_q(name, d):
    """(policy Q, DR q-hat, c, DR q-hat is q_pi) of one decision: T (q_pi, q_dr), S (q_pi at tau/c, q_dr),
    cross_replicate (q_dr, q_pi), neg (-q_pi, q_dr)."""
    _require(name in FAMILIES, f'unknown family {name}')
    if name == 'temperature':
        return d['q_pi'], d['q_dr'], 1., False
    if name == 'shrinkage':
        return d['q_pi'], d['q_dr'], shrink_factor(d['q_pi'], d['se'], d['mask']), False
    if name == 'cross_replicate':
        return d['q_dr'], d['q_pi'], 1., True
    return -d['q_pi'], d['q_dr'], 1., False


def family_policy(name, d, tau, fq=None):
    """pi of a family at tau: the reference when tau = inf or c = 0, else kl_policy at the per-decision temperature
    tau / c (c = 1 except shrinkage, so T is kl_policy at tau itself)."""
    q, q_hat, c, _ = family_q(name, d) if fq is None else fq
    if tau == INF or c == 0.:
        return np.array(d['reference'], dtype=np.float64, copy=True)
    return tilt(q, d['reference'], d['mask'], tau / c)


def ratio(p, logging, a, on):
    """The estimator's importance ratio p(a)/logging(a) of logged action(s) ``a`` on the mask, exactly 0 off it (no
    division). Scalar or vectorised over an array of actions."""
    a, on = np.asarray(a), np.asarray(on, dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(on, np.asarray(p)[a] / np.where(on, np.asarray(logging)[a], 1.), 0.)
    return float(out) if out.ndim == 0 else out


def model_term(pi, reference, mask, q_hat):
    """The estimator's model term sum_mask (pi(a) - reference(a)) q_hat(a)."""
    return float((pi[mask] - reference[mask]) @ q_hat[mask])


def onestep_terms(pi, d, reward, q_hat=None):
    """(M, C, IPS, rho') of one decision for target law ``pi`` (q_hat defaults to q_dr): ``model_term``, ``ratio`` and
    ``row_correction`` (the estimator's row functions). Off the mask C = IPS = rho' = 0 and q_hat[a_t] is never read."""
    q_hat = d['q_dr'] if q_hat is None else q_hat
    model = model_term(pi, d['reference'], d['mask'], q_hat)
    on = d['on_mask']
    rho_pi = ratio(pi, d['logging'], d['a'], on)
    qa = float(q_hat[d['a']]) if on else 0.
    corr = float(row_correction(rho_pi, d['rho_ref'], on, float(reward), qa))
    ips = float(row_correction(rho_pi, d['rho_ref'], on, float(reward), 0.))
    return model, corr, ips, rho_pi


def kl(pi, reference, mask):
    p, r = pi[mask], reference[mask]
    pos = p > 0
    return float(np.sum(p[pos] * np.log(p[pos] / r[pos])))


def row_correction(rho, rho_ref, on, reward, qa):
    """The correction of the row rules: (rho' - rho_r)(G - q_dr[a]) for a logged action on the mask, exactly 0 off it
    (q_dr[a] is never read there). Engine.parts and the row identity both use it."""
    return np.where(on, (rho - rho_ref) * np.where(on, reward - qa, 0.), 0.)


def row_term(model, rho, rho_ref, on, reward, qa):
    """The per-decision one-step DR term of the row rules: the model term for every row plus ``row_correction``."""
    return model + row_correction(rho, rho_ref, on, reward, qa)


def row_identity(pi, reference, logging, mask, q_hat, model, reward):
    """S = sum over actions a with logging(a) > 0 of logging(a) row_term(a; G') for a hypothetical logged action a, built
    from the estimator's own row functions (``ratio`` for rho' and rho_r, ``row_term`` for the per-decision term; the
    model term ``model`` comes from ``model_term``). Exactly 0 in exact arithmetic for any G'."""
    actions = np.flatnonzero(logging > 0)
    on = mask[actions]
    terms = row_term(model, ratio(pi, logging, actions, on), ratio(reference, logging, actions, on), on, reward,
                     np.where(on, q_hat[actions], 0.))
    return float(np.sum(logging[actions] * terms))


def disagreement(q, reference, mask, se, margins):
    """(flags per margin, tie_q, tie_ref, se_pair_zero): argmax_mask q != argmax_mask reference and q[best] -
    q[ref_best] > k; k = '1se' is se[ref_best] (an exact CRN tie can make it 0: then D_1se = D_0 for that row, counted).
    A tie at either maximum keeps the decision outside every D."""
    idx = np.flatnonzero(mask)
    qm, rm = np.asarray(q)[idx], np.asarray(reference)[idx]
    tie_q, tie_r = int((qm == qm.max()).sum()) > 1, int((rm == rm.max()).sum()) > 1
    if tie_q or tie_r:
        return [False] * len(margins), tie_q, tie_r, False
    best, ref_best = int(np.argmax(qm)), int(np.argmax(rm))
    if best == ref_best:
        return [False] * len(margins), False, False, False
    se_pair = float(np.asarray(se)[idx[ref_best]])
    gap = float(qm[best] - qm[ref_best])
    return [gap > (se_pair if m == '1se' else float(m)) for m in margins], False, False, se_pair == 0


def perm_rng(seed, k, sha8):
    """The registered perm RNG of one row and draw: SeedSequence([placebo_perm_seed, k, int(row sha256[:8], 16)])."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(k), int(sha8)]))


def permuted(q, mask, seed, k, sha8):
    """q with its mask values permuted by perm_rng (the multiset over the mask is kept)."""
    idx = np.flatnonzero(mask)
    out = np.array(q, dtype=np.float64, copy=True)
    out[idx] = np.asarray(q)[idx[perm_rng(seed, k, sha8).permutation(len(idx))]]
    return out


def gint(row, d, sealed_tau, tol, rtol):
    """The candidate identity and ratio checks of one row (IntegrityError = study-level INVALID). Returns the vector
    difference."""
    pi = tilt(d['q_pi'], d['reference'], d['mask'], sealed_tau)
    diff = float(np.max(np.abs(pi - d['candidate'])))
    _require(diff <= tol, f'G-INT {row["request_id"]}: |kl_policy(q_planning) - candidate| = {diff!r}')
    a = d['a']
    if d['b_a'] > 0:
        for name, value, recorded in (('rho_candidate', pi[a] / d['b_a'], d['rho_candidate']),
                                      ('rho_reference', d['reference'][a] / d['b_a'], d['rho_reference'])):
            _require(isinstance(recorded, float) and bool(np.isclose(value, recorded, rtol=rtol, atol=0)),
                     f'G-INT {row["request_id"]}: {name} {recorded!r} != recomputed {float(value)!r}')
    else:
        _require(d['candidate'][a] == 0 and pi[a] == 0 and d['reference'][a] == 0 and d['rho_candidate'] == 0.
                 and d['rho_reference'] == 0. and type(d['rho_candidate']) is float
                 and type(d['rho_reference']) is float,
                 f'G-INT {row["request_id"]}: positivity row needs candidate[a] = reference[a] = 0 and rho = 0.0')
    return diff


# ---------------------------------------------------------------- sealed pa_values, frame facts

PHASE_A_COLUMNS = ('pa_id', 'game', 'in_population', 'status', 'kind', 'reason', 'end_kind')  # G-free
PHASE_B_COLUMNS = ('pa_id', 'resolution', 'candidate', 'reference', 'delta', 'l2')  # read only after prediag.json
PA_VALUE_COLUMNS = PHASE_A_COLUMNS + PHASE_B_COLUMNS[1:]  # never terminal_residual or the *_bounds columns


def plain(value):
    """A parquet/pandas cell as plain python: NaN -> None, arrays -> lists, numpy scalars -> python."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) else float(value)
    import pandas as pd
    return None if value is pd.NA or value is pd.NaT else value


def sealed_rows(records, columns=PHASE_A_COLUMNS):
    """{pa_id: row} of the sealed pa_values records (file order kept) with exactly ``columns``."""
    out = {}
    for record in records:
        _require(set(record) == set(columns), f'pa_values read with columns {sorted(record)} instead of {list(columns)}')
        row = {k: plain(record.get(k)) for k in columns}
        _require(isinstance(row['pa_id'], str) and row['pa_id'] not in out, 'pa_values: PA ids must be unique strings')
        if 'status' in row:
            _require(row['in_population'] == (row['status'] in (est.COMPLETE, est.CENSORED)),
                     f'pa_values {row["pa_id"]}: in_population differs from its status')
        out[row['pa_id']] = row
    return out


def add_value_columns(sealed, values):
    """Phase (b): the value columns (PHASE_B_COLUMNS) joined to the phase (a) rows by pa_id, same PAs in the same order."""
    _require(list(values) == list(sealed), 'pa_values value columns cover other PAs (ids or order)')
    for pa_id, row in values.items():
        sealed[pa_id].update({k: v for k, v in row.items() if k != 'pa_id'})
    return sealed


def estimator_info(pa_row, facts, n_decisions):
    """policy_estimator facts of one PA: E0, start reason and structural problem from the sealed row (the problem index
    of a structural censoring is the submitted prefix length), the PA end from the frame."""
    status, reason = pa_row['status'], pa_row['reason']
    problem = index = None
    if status == est.NO_DECISION:
        problem, index = 'no_decision', 0
    elif status == est.UNSUBMITTABLE:
        problem, index = reason, 0
    elif (status == est.CENSORED and pa_row['kind'] == est.REFUSED and isinstance(reason, str)
          and reason.startswith('structural:')):
        problem, index = reason.split(':', 1)[1], n_decisions
    return {'game': facts['game'], 'in_population': pa_row['in_population'],
            'start_reason': reason if status == est.EXCLUDED_PRE_START else None, 'problem': problem,
            'problem_index': index, **{k: facts[k] for k in ('reward', 'kind', 'reason', 'end', 'end_kind', 'flags')}}


def _same(a, b, tol):
    if a is None or b is None:
        return a is None and b is None, 0.
    diff = abs(float(a) - float(b))
    return diff <= tol, diff


class FactsCheck:
    """The per-PA reproduction of the sealed pa_values rows by policy_estimator.pa_value (censoring of the sealed run)
    on the ledger rows and the frame facts, in two phases. Phase (a), G-free (in --check and before prediag): the reward
    value is masked (presence kept), status / kind / reason / in_population must equal the sealed row and game /
    end_kind the frame facts. Phase (b), after prediag (--run only): the real reward; status / kind / reason /
    resolution / l2 exact and candidate / reference / delta within ``tol`` (None and NaN alike). Phase (b) also feeds
    the aggregate check against dr.json (L2 PAs exact, their mean reward within ``tol``). Any mismatch raises
    IntegrityError (study-level INVALID)."""
    def __init__(self, censoring, tol):
        _require(censoring in est.CENSORING and isinstance(tol, float) and 0 <= tol < 1e-6, 'cross-check settings')
        self.censoring, self.tol = censoring, tol
        self.phase = {'a': 0, 'b': 0}
        self.max_abs = {k: 0. for k in ('candidate', 'reference', 'delta')}
        self.l2_pas, self.l2_rewards, self.missing_reward = 0, [], Counter()

    def _exact(self, pa_id, phase, pa_row, got, keys):
        for key in keys:
            _require(pa_row[key] == plain(got.get(key)), f'pa_values reproduction ({phase}) {pa_id}: {key} '
                                                         f'{pa_row[key]!r} != recomputed {plain(got.get(key))!r}')

    def phase_a(self, pa_id, decisions, info, pa_row):
        masked = {**info, 'reward': None if info['reward'] is None else MASKED_REWARD}
        got = est.pa_value(decisions, masked, self.censoring)
        self._exact(pa_id, 'a', pa_row, got, ('status', 'kind', 'reason', 'in_population'))
        _require(pa_row['game'] == info['game'], f'pa_values reproduction (a) {pa_id}: game differs')
        _require(pa_row['end_kind'] == info['end_kind'], f'pa_values reproduction (a) {pa_id}: end_kind differs')
        self.phase['a'] += 1
        if info['reward'] is None:
            self.missing_reward[f'{info["kind"]}:{info["reason"]}'] += 1

    def phase_b(self, pa_id, decisions, info, pa_row):
        _require(set(PHASE_B_COLUMNS) <= set(pa_row), 'phase (b) needs the pa_values value columns (add_value_columns)')
        got = est.pa_value(decisions, info, self.censoring)
        self._exact(pa_id, 'b', pa_row, got, ('status', 'kind', 'reason', 'resolution', 'l2', 'in_population'))
        for key in self.max_abs:
            same, diff = _same(pa_row[key], plain(got.get(key)), self.tol)
            _require(same, f'pa_values reproduction (b) {pa_id}: {key} {pa_row[key]!r} != recomputed {got.get(key)!r}')
            self.max_abs[key] = max(self.max_abs[key], diff)
        self.phase['b'] += 1
        if got['in_population'] and got.get('l2', got['status'] == est.COMPLETE):
            self.l2_pas += 1
            if info['reward'] is not None:
                self.l2_rewards.append(float(info['reward']))

    def record(self, dr=None):
        """The reproduction record; after phase (b) and with the sealed dr.json, the aggregate reward check."""
        out = {'rule': 'policy_estimator.pa_value on the ledger rows + frame facts reproduces the sealed pa_values row',
               'censoring': self.censoring, 'tolerance': self.tol,
               'columns_read': {'phase_a': list(PHASE_A_COLUMNS), 'phase_b': list(PHASE_B_COLUMNS[1:])},
               'phase_a_pas': self.phase['a'], 'phase_b_pas': self.phase['b'] or None,
               'phase_b_max_abs_diff': dict(self.max_abs) if self.phase['b'] else None,
               'missing_reward_pas': dict(self.missing_reward), 'aggregate_check': 'not run (phase b only)'}
        if not self.phase['b']:
            return out
        layer = ((dr or {}).get('layers') or {}).get('L2_complete_conditional') or {}
        mean = float(np.mean(self.l2_rewards)) if self.l2_rewards else None
        out.update(l2_pas=self.l2_pas, l2_mean_reward=mean)
        if layer.get('observed_mean_end_value') is None:
            out['aggregate_check'] = 'skipped: dr.json has no L2 observed_mean_end_value'
            return out
        _require(layer.get('pas') == self.l2_pas, f'L2 PAs {self.l2_pas} differ from the sealed {layer.get("pas")}')
        diff = abs(mean - float(layer['observed_mean_end_value']))
        _require(diff <= self.tol, f'mean frame reward over the L2 PAs differs from the sealed value by {diff!r}')
        out['aggregate_check'] = {'sealed_pas': layer['pas'], 'sealed_mean': layer['observed_mean_end_value'],
                                  'abs_diff': diff}
        return out


def reproduce(decisions, pas, sealed, facts_check, checkpoint=None):
    """Phase (b): a second streamed pass, every PA's sealed row reproduced with the real reward."""
    for k, (pa_id, info, rows) in enumerate(est.by_pa(decisions, pas)):
        if checkpoint is not None and k % 200 == 0:
            checkpoint()
        rows = sorted(rows, key=lambda d: d['decision_index'])
        facts_check.phase_b(pa_id, rows, estimator_info(sealed[pa_id], info, len(rows)), sealed[pa_id])
    return facts_check


def frame_facts(frame, blocks, games_table, defense):
    """{pa_id: facts} in block order: the PA end exactly as the registered runner computes it
    (run_policy_validation.outcome_function -> policy_requests.pa_outcome(frame, positions, components.defense_we,
    games, rule='structural-end-v1', score_source='next_row'); the store's frame is the loaded frame object),
    pa_manifest's start pitcher and date, and WE_start = the same frozen defensive WE of the PA's first-row pre-pitch
    state for the initial defender (None when that state is invalid)."""
    from . import policy_requests as preq
    from .game import GameState
    out = {}
    for pa_id, positions in blocks:
        _require(pa_id not in out, f'PA {pa_id} appears in more than one block')
        outcome = preq.pa_outcome(frame, positions, defense, games_table, rule='structural-end-v1',
                                  score_source='next_row')
        manifest = preq.pa_manifest(frame, positions)
        first = frame.iloc[positions[0]]
        try:
            we_start = float(defense(GameState.from_row(first), str(first.inning_topbot) == 'Top'))
        except (ValueError, TypeError, KeyError):
            we_start = None
        out[pa_id] = {'game': int(frame.game_pk.iloc[positions[0]]), 'start_pitcher': manifest['start_pitcher'],
                      'date': manifest['date'], 'reward': outcome['reward'], 'kind': outcome['kind'],
                      'reason': outcome['reason'], 'end': outcome['end'], 'end_kind': outcome['end_kind'],
                      'flags': list(outcome['flags']), 'we_start': we_start}
    return out


# ---------------------------------------------------------------- the streamed pass

class Scan:
    """One streamed pass in PA order (policy_estimator.by_pa): G-INT, D, the population accounting, the sealed facts
    cross-check and, for every kept row (eligible status in an E0 PA, observed reward or not), the per-family M_dr,
    M_pi, rho', KL and DR q-hat at a_t per tau (kl_policy), plus the padded vectors the vectorised placebos need. G is
    stored, never aggregated here."""
    def __init__(self, *, taus, sealed_tau, identity_tolerance, ratio_rtol, statuses, families, margins,
                 expected_q_source, runtime_sha256=None):
        self.taus = check_grid(taus, sealed_tau)
        _require(isinstance(identity_tolerance, float) and 0 <= identity_tolerance < 1e-6
                 and isinstance(ratio_rtol, float) and 0 <= ratio_rtol < 1e-6, 'identity tolerance / ratio rtol')
        _require(statuses and set(statuses) <= set(RESULT_STATUSES) and len(set(statuses)) == len(statuses),
                 f'eligible statuses must be distinct values of {RESULT_STATUSES}')
        _require(families and families[0] == 'temperature' and set(families) <= set(FAMILIES)
                 and len(set(families)) == len(families), 'families: temperature first, known names')
        _require(margins and margins[0] == 0 and '1se' in margins and all(
            m == '1se' or (isinstance(m, (int, float)) and m >= 0) for m in margins), 'D margins: 0 first, "1se" in')
        _require(isinstance(expected_q_source, str) and expected_q_source, 'the registered DR q-hat source is required')
        self.sealed_tau, self.tol, self.rtol, self.statuses = sealed_tau, identity_tolerance, ratio_rtol, tuple(statuses)
        self.families, self.margins, self.expected_q_source = list(families), list(margins), expected_q_source
        self.runtime_sha256 = runtime_sha256
        self.n_vocab = None
        self.by_status, self.excluded, self.kept_status = Counter(), Counter(), Counter()
        self.q_source, self.mask_sizes, self.pa, self.pa_no_reward = Counter(), Counter(), Counter(), Counter()
        self.identity = {'tau': sealed_tau, 'rows_checked': 0, 'max_abs_diff_candidate': 0., 'tolerance': identity_tolerance,
                         'ratio_rtol': ratio_rtol, 'positivity_rows_checked': 0}
        self.d_rows = {'rows': 0, 'tie_q': 0, 'tie_ref': 0, 'se_pair_zero_in_D0': 0,
                       **{f'D:{margin_key(m)}': 0 for m in self.margins}}
        self.se_ties = {'rows': 0, 'entries': 0}
        self.row_identity = {'rule': 'S = sum_{a: logging(a) > 0} logging(a) row_term(a; G\') == 0 for every row with a '
                                     'result, every family x tau', 'G': list(IDENTITY_G), 'tolerance': IDENTITY_TOL,
                             'rows': 0, 'evaluations': 0, 'max_abs': 0.}
        self.cols = {k: [] for k in ('pa_index', 'pa_id', 'game', 'pitcher', 'decision_index', 'balls', 'strikes',
                                     'status', 'on_mask', 'observed', 'G', 'we_start', 'rho_ref', 'b_a', 'a', 'sha8', 'D',
                                     'tie', 'c', 'noise', 'model_dr', 'model_pi', 'rho', 'kl', 'qa', 'mask', 'ref', 'q_pi',
                                     'q_dr', 'logging', 'pa_pos', 'pa_out', 'first_eligible', 'first_D0')}

    def run(self, decisions, pas, *, sealed=None, facts_check=None, checkpoint=None):
        """``pas``: {pa_id: facts} in the ledger's PA order (game, in_population, start_pitcher, reward, kind, reason,
        end_kind, flags, we_start); ``sealed`` {pa_id: pa_values row} with ``facts_check``."""
        for k, (pa_id, info, rows) in enumerate(est.by_pa(decisions, pas)):
            if checkpoint is not None and k % 200 == 0:
                checkpoint()
            self._pa(k, pa_id, info, sorted(rows, key=lambda d: d['decision_index']), sealed, facts_check)
        return self

    def _pa(self, k, pa_id, info, rows, sealed, facts_check):
        self.pa['pas'] += 1
        if facts_check is not None:
            facts_check.phase_a(pa_id, rows, estimator_info(sealed[pa_id], info, len(rows)), sealed[pa_id])
        if rows:
            _require(str(rows[0]['pitcher']) == str(info['start_pitcher']),
                     f'PA {pa_id}: first ledger row pitcher differs from the frame start pitcher')
        e0, reward = bool(info['in_population']), info['reward']
        if e0:
            self.pa['e0'] += 1
            if reward is None:
                self.pa_no_reward[f'{info["kind"]}:{info["reason"]}'] += 1
        start = len(self.cols['pa_index'])
        for row in rows:
            status = row['status']
            self.by_status[status] += 1
            _require(status not in FATAL, f'FAILED decision in PA {pa_id}: a halted ledger is not estimated')
            _require(self.runtime_sha256 is None or row.get('runtime_sha256') == self.runtime_sha256,
                     'decision row pinned to another runtime than the ledger header')
            result = row.get('result')
            if status not in RESULT_STATUSES:
                _require(result is None, f'{status} row with a recorded result')
                self.excluded[f'status:{status}'] += 1
                continue
            _require(isinstance(result, dict), f'{status} row without a recorded result: runtime v3 required')
            self.n_vocab = len(result['logging']) if self.n_vocab is None else self.n_vocab
            d = decision_arrays(row, self.n_vocab)
            self.q_source[str(d['q_source'])] += 1
            _require(d['q_source'] == self.expected_q_source,
                     f'G-INT {row["request_id"]}: q_source {d["q_source"]!r} is not {self.expected_q_source!r}')
            diff = gint(row, d, self.sealed_tau, self.tol, self.rtol)
            self.identity['rows_checked'] += 1
            self.identity['positivity_rows_checked'] += d['b_a'] == 0
            self.identity['max_abs_diff_candidate'] = max(self.identity['max_abs_diff_candidate'], diff)
            flags, tie_q, tie_r, se_zero = disagreement(d['q_pi'], d['reference'], d['mask'], d['se'], self.margins)
            self.d_rows['rows'] += 1
            self.d_rows['tie_q'] += tie_q
            self.d_rows['tie_ref'] += tie_r
            self.d_rows['se_pair_zero_in_D0'] += bool(se_zero and flags[0])
            for m, flag in zip(self.margins, flags):
                self.d_rows[f'D:{margin_key(m)}'] += flag
            self.se_ties['rows'] += d['se_ties'] > 0
            self.se_ties['entries'] += d['se_ties']
            fam = self._families(row, d)
            if status not in self.statuses:
                self.excluded[f'status:{status}'] += 1
                continue
            if not e0:
                self.excluded['pa_not_in_E0'] += 1
                continue
            if reward is None:
                self.excluded[f'no_reward_pa:{info["kind"]}:{info["reason"]}'] += 1
            self._keep(k, pa_id, info, row, d, flags, tie_q or tie_r, reward, fam)
        end = len(self.cols['pa_index'])
        if end > start:
            kept_status = self.cols['status'][start:end]
            pos, out = LOGGING_POSITIVITY in kept_status, OUTSIDE_POLICY_SUPPORT in kept_status
            d0 = [flags[0] for flags in self.cols['D'][start:end]]
            first_d = d0.index(True) if True in d0 else None
            for i in range(end - start):
                self.cols['pa_pos'].append(pos)
                self.cols['pa_out'].append(out)
                self.cols['first_eligible'].append(i == 0)
                self.cols['first_D0'].append(first_d == i)
            if reward is not None:
                self.pa['eligible_pas'] += 1
                self.pa['eligible_pas_with_positivity'] += pos
                self.pa['eligible_pas_with_outside'] += out

    def _families(self, row, d):
        """Per family x tau (kl_policy): M_dr, M_pi, rho', KL; per family the DR q-hat at a_t and the noise numerator;
        the row identity for both constant outcomes (EstimatorSuspect on a violation)."""
        shape = (len(self.families), len(self.taus))
        model_dr, model_pi, rho, kls = (np.zeros(shape) for _ in range(4))
        qa, noise = np.zeros(len(self.families)), np.zeros(len(self.families))
        se_max, shrink = float(d['se'][d['mask']].max()), np.nan
        rid = self.row_identity
        rid['rows'] += 1
        for f, name in enumerate(self.families):
            fq = family_q(name, d)
            q_hat = fq[1]
            for j, tau in enumerate(self.taus):
                pi = family_policy(name, d, tau, fq)
                model_dr[f, j], _, _, rho[f, j] = onestep_terms(pi, d, 0.)
                model_pi[f, j] = onestep_terms(pi, d, 0., d['q_pi'])[0]
                kls[f, j] = kl(pi, d['reference'], d['mask'])
                model = model_pi[f, j] if fq[3] else model_dr[f, j]
                for g in IDENTITY_G:
                    value = row_identity(pi, d['reference'], d['logging'], d['mask'], q_hat, model, g)
                    rid['evaluations'] += 1
                    if not abs(value) <= IDENTITY_TOL:
                        raise EstimatorSuspect(f'row identity failed at {row["request_id"]} ({name}, tau {tau_key(tau)}, '
                                               f"G' {g}): S = {value!r}")
                    rid['max_abs'] = max(rid['max_abs'], abs(value))
            qa[f] = float(q_hat[d['a']]) if d['on_mask'] else 0.
            noise[f] = fq[2] * se_max
            if name == 'shrinkage':
                shrink = fq[2]
        return {'model_dr': model_dr, 'model_pi': model_pi, 'rho': rho, 'kl': kls, 'qa': qa, 'noise': noise, 'c': shrink}

    def _keep(self, k, pa_id, info, row, d, flags, tie, reward, fam):
        c = self.cols
        game = int(str(pa_id).split(':', 1)[0])
        _require(game == int(info['game']), f'PA {pa_id}: game in the pa_id differs from the frame game')
        self.kept_status[d['status']] += 1
        self.mask_sizes[int(d['mask'].sum())] += 1
        we_start = info.get('we_start')
        for key, value in (('pa_index', k), ('pa_id', pa_id), ('game', game), ('pitcher', str(info['start_pitcher'])),
                           ('decision_index', int(row['decision_index'])), ('balls', int(row['balls'])),
                           ('strikes', int(row['strikes'])), ('status', d['status']), ('on_mask', d['on_mask']),
                           ('observed', reward is not None), ('G', np.nan),  # values: attach_rewards after prediag
                           ('we_start', np.nan if we_start is None else float(we_start)), ('rho_ref', d['rho_ref']),
                           ('b_a', d['b_a']), ('a', d['a']), ('sha8', int(row['sha256'][:8], 16)), ('D', list(flags)),
                           ('tie', tie), ('mask', d['mask']), ('ref', d['reference']), ('q_pi', d['q_pi']),
                           ('q_dr', d['q_dr']), ('logging', d['logging'])):
            c[key].append(value)
        for key in ('model_dr', 'model_pi', 'rho', 'kl', 'qa', 'noise', 'c'):
            c[key].append(fam[key])

    def table(self):
        """Per kept decision (prediag population), compact arrays: ids, cluster keys, count, status, flags, G (NaN when
        unobserved), WE_start, rho_r, and [n, families, taus] M_dr / M_pi / rho' / KL; [n, families] DR q-hat at a_t and
        noise numerators; [n, V] vectors for the placebos."""
        c, n = self.cols, len(self.cols['pa_index'])
        v, f, t = self.n_vocab or 0, len(self.families), len(self.taus)
        out = {'n': n, 'families': list(self.families), 'taus': list(self.taus), 'margins': list(self.margins),
               'dr_uses_pi': [name == 'cross_replicate' for name in self.families],
               'pa_id': np.array(c['pa_id'], dtype=object), 'pitcher': np.array(c['pitcher'], dtype=object),
               'status': np.array(c['status'], dtype=object),
               'D': np.array(c['D'], dtype=bool).reshape(n, len(self.margins))}
        for key, dtype in (('pa_index', np.int64), ('game', np.int64), ('decision_index', np.int64), ('balls', np.int64),
                           ('strikes', np.int64), ('on_mask', bool), ('observed', bool), ('G', np.float64),
                           ('we_start', np.float64), ('rho_ref', np.float64), ('b_a', np.float64), ('a', np.int64),
                           ('sha8', np.int64), ('tie', bool), ('c', np.float64), ('pa_pos', bool), ('pa_out', bool),
                           ('first_eligible', bool), ('first_D0', bool)):
            out[key] = np.array(c[key], dtype=dtype)
        for key, shape in (('model_dr', (f, t)), ('model_pi', (f, t)), ('rho', (f, t)), ('kl', (f, t)), ('qa', (f,)),
                           ('noise', (f,)), ('mask', (v,)), ('ref', (v,)), ('q_pi', (v,)), ('q_dr', (v,)),
                           ('logging', (v,))):
            out[key] = np.stack(c[key]) if n else np.zeros((0, *shape))
        return out

    def records(self, table=None):
        table = self.table() if table is None else table
        obs = table['observed']
        n = int(obs.sum())
        status = table['status'][obs]
        return {
            'decision_rows': sum(self.by_status.values()), 'by_status': dict(self.by_status),
            'q_source': dict(self.q_source), 'vocabulary_size': self.n_vocab,
            'row_identity': dict(self.row_identity), 'se_ties': dict(self.se_ties),
            'identity_check': {**self.identity, 'q_pi': Q_PI, 'pass': True,
                               'rule': 'kl_policy(q_planning, reference, mask, sealed tau) == recorded candidate; ratios '
                                       'rtol 1e-12 where logging[a] > 0; positivity rows exact zeros'},
            'counts': {'prediag_decisions': int(table['n']), 'estimation_decisions': n,
                       'prediag_minus_estimation': int(table['n']) - n,
                       'kept_decisions_by_status': dict(self.kept_status),
                       'eligible_by_status': dict(Counter(status.tolist())),
                       'excluded_decisions': dict(sorted(self.excluded.items())), 'eligible_statuses': list(self.statuses),
                       'pas': dict(self.pa), 'e0_pas_without_reward': dict(sorted(self.pa_no_reward.items())),
                       'mask_sizes': {str(k): v for k, v in sorted(self.mask_sizes.items())},
                       'two_strike_decisions': int((table['strikes'][obs] == 2).sum()),
                       'p_out': {'decision': float(np.mean(status == OUTSIDE_POLICY_SUPPORT)) if n else None,
                                 'pa': (self.pa['eligible_pas_with_outside'] / self.pa['eligible_pas']
                                        if self.pa['eligible_pas'] else None)},
                       'p_pos': {'decision': float(np.mean(status == LOGGING_POSITIVITY)) if n else None,
                                 'pa': (self.pa['eligible_pas_with_positivity'] / self.pa['eligible_pas']
                                        if self.pa['eligible_pas'] else None)}},
            'disagreement': {'definition': 'argmax_mask q_planning != argmax_mask reference and q[best] - q[ref_best] > k; '
                                           "k = '1se' is q_planning_diff_se[argmax reference]; ties outside every D",
                             'margins': [margin_key(m) for m in self.margins], 'result_rows': dict(self.d_rows),
                             'estimation': {f'D:{margin_key(m)}': int(table['D'][obs, i].sum()) for i, m in
                                            enumerate(self.margins)} if n else {},
                             'ties_estimation': int(table['tie'][obs].sum())}}


# ---------------------------------------------------------------- prediag and the eligibility gate (no G)

def kish(weights):
    w = np.asarray(weights, dtype=np.float64)
    s2 = float((w ** 2).sum())
    return None if s2 == 0 else float(w.sum() ** 2 / s2)


def _pa_products(values, pa_index):
    """Per-PA products of per-decision values (rows of a PA are contiguous, in ledger order)."""
    starts = np.flatnonzero(np.r_[True, pa_index[1:] != pa_index[:-1]])
    return np.multiply.reduceat(values, starts), starts


def _table_subset(table, name, rows=None):
    """Rows of a named subset of the table (or of ``rows``): all, D:<k>, not_D:<k>, two_strike, no_positivity_pa,
    combined with '&'."""
    n = len(table['observed'])
    keep = np.ones(n, dtype=bool) if rows is None else rows.copy()
    for part in name.split('&'):
        if part == 'all':
            continue
        if part.startswith('D:') or part.startswith('not_D:'):
            index = [margin_key(m) for m in table['margins']].index(part.split(':', 1)[1])
            d = table['D'][:, index]
            keep &= d if part.startswith('D:') else ~d
        elif part == 'two_strike':
            keep &= table['strikes'] == 2
        elif part == 'no_positivity_pa':
            keep &= ~table['pa_pos']
        else:
            raise IntegrityError(f'unknown subset {part}')
    return keep


def attach_rewards(table, pas):
    """After prediag.json: the PA-end rewards enter the table (until then only their presence, ``observed``, is in it)."""
    rewards = np.array([np.nan if pas[p]['reward'] is None else float(pas[p]['reward']) for p in table['pa_id']])
    _require(np.array_equal(np.isfinite(rewards), table['observed']), 'reward presence differs from the scanned table')
    table['G'] = rewards
    return table


def estimation_eligible(table):
    """The ESTIMATION-ELIGIBLE decisions of a table of kept rows (every kept row is of an E0 PA and has a recorded
    result): those whose PA-end reward is observed. The single definition used by the gate, the sample minima and the
    estimates."""
    return np.asarray(table['observed'], dtype=bool).copy()


def gate_record(table, rows, f, j, gate, exempt):
    """Eligibility criteria of one family x tau on ``rows`` (prediag population)."""
    tau = table['taus'][j]
    rho, ref = table['rho'][rows, f, j], table['rho_ref'][rows]
    out = {'decisions': int(rows.sum()), 'games': int(len(set(table['game'][rows].tolist()))),
           'pitchers': int(len(set(table['pitcher'][rows].tolist())))}
    if not out['decisions']:
        return {**out, 'pass': False, 'reason': 'no decision'}
    e, e_ref = kish(rho), kish(ref)
    out['ess'], out['ess_reference'] = e, e_ref
    out['ess_ratio'] = None if e is None or not e_ref else e / e_ref
    out['max_weight_share'] = None if not rho.sum() > 0 else float(rho.max() / rho.sum())
    noise = np.zeros(int(rows.sum())) if tau == INF else table['noise'][rows, f] / tau
    out['noise_ratio_q50'], out['noise_ratio_q90'] = (float(np.quantile(noise, q)) for q in (.5, .9))
    out['noise_exempt'] = bool(exempt)
    checks = {'ess_ratio': out['ess_ratio'] is not None and out['ess_ratio'] >= gate['ess_ratio_min'],
              'max_weight_share': out['max_weight_share'] is not None
              and out['max_weight_share'] <= gate['max_weight_share'],
              'games': out['games'] >= gate['games_min'], 'pitchers': out['pitchers'] >= gate['pitchers_min'],
              'noise': bool(exempt) or out['noise_ratio_q90'] <= gate['noise_q90_max']}
    out['checks'] = checks
    out['pass'] = all(checks.values())
    return out


def _restrict(table, rows):
    """The per-row arrays of ``table`` at ``rows`` (lists and scalars kept)."""
    n = len(table['observed'])
    return {k: (v[rows] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in table.items()}


def prediag(table, gate, *, gate_subsets=tuple(CONFIRMATORY.values())):
    """G-free diagnostics (v4.1 R4-4): every metric on the ESTIMATION-ELIGIBLE decisions (the all-kept-rows count is
    reported second): per family x tau ESS / weight share / clusters / noise / KL / M, the eligibility gate per family x
    tau x contrast subset, the PA-product ESS (DESCRIPTIVE), the D smoke assertion p(D_1se) < p(D_0) (else INVALID) and
    the tau = inf identity on every kept row (else ESTIMATOR_SUSPECT)."""
    _require(int(table['n']) > 0, 'no E0 decision with a recorded result')
    identity = inf_identity(table)
    if not identity['zero']:
        raise EstimatorSuspect(f'tau = inf identity failed (pi = reference must give D = 0 exactly): {identity}')
    eligible = estimation_eligible(table)
    sub = _restrict(table, eligible)
    n = int(eligible.sum())
    _require(n > 0, 'no estimation-eligible decision')
    keys = [margin_key(m) for m in table['margins']]
    p_d = {f'D:{k}': float(sub['D'][:, i].mean()) for i, k in enumerate(keys)}
    _require(p_d['D:1se'] < p_d['D:0'], f'smoke assertion failed: p(D_1se) {p_d["D:1se"]} >= p(D_0) {p_d["D:0"]}')
    status, pa = sub['status'], sub['pa_index']
    ref_w, starts = _pa_products(sub['rho_ref'], pa)
    pa_games = sub['game'][starts]
    ref_pa_ess = est.effective_sample_size(ref_w)
    exempt = {(e['family'], float(e['tau'])) for e in gate['noise_exempt']}
    common = {'population': 'estimation-eligible decisions (E0 PA, observed G, result recorded)',
              'estimation_eligible_decisions': n, 'all_kept_decisions': int(table['n']),
              'all_minus_eligible': int(table['n']) - n, 'estimation_eligible_pas': int(len(starts)),
              'games': int(len(set(sub['game'].tolist()))), 'pitchers': int(len(set(sub['pitcher'].tolist()))),
              'p_D': p_d,
              'p_out': {'decision': float(np.mean(status == OUTSIDE_POLICY_SUPPORT)),
                        'pa': float(np.mean(sub['pa_out'][starts]))},
              'p_pos': {'decision': float(np.mean(status == LOGGING_POSITIVITY)),
                        'pa': float(np.mean(sub['pa_pos'][starts]))},
              'reference': {'decision_ess': kish(sub['rho_ref']), 'pa_ess_descriptive': ref_pa_ess},
              'gate': {k: v for k, v in gate.items()}}
    everything = np.ones(n, dtype=bool)
    families, gates = {}, {}
    for f, fam in enumerate(table['families']):
        rows, by_gate = {}, {}
        for j, tau in enumerate(table['taus']):
            rho = sub['rho'][:, f, j]
            w, _ = _pa_products(rho, pa)
            pa_ess = est.effective_sample_size(w)
            m_dr, m_pi = sub['model_dr'][:, f, j], sub['model_pi'][:, f, j]
            metrics = gate_record(sub, everything, f, j, gate, (fam, tau) in exempt)
            rows[tau_key(tau)] = {
                **{k: v for k, v in metrics.items() if k not in ('pass', 'checks')},
                'population': 'estimation-eligible',
                'mean_kl': float(sub['kl'][:, f, j].mean()), 'mean_M_dr': float(m_dr.mean()),
                'mean_M_pi': float(m_pi.mean()), 'max_rho': float(rho.max()),
                'mean_M_dr_by_class': {s: (float(m_dr[status == s].mean()) if (status == s).any() else None)
                                       for s in RESULT_STATUSES},
                'pa_product_ess_descriptive': pa_ess,
                'pa_product_ess_ratio_descriptive': None if pa_ess is None or not ref_pa_ess else pa_ess / ref_pa_ess,
                'pa_product_game_ess_descriptive': est.effective_sample_size(w, pa_games)}
            by_gate[tau_key(tau)] = {name: {**gate_record(sub, _table_subset(sub, name), f, j, gate, (fam, tau) in exempt),
                                            'population': 'estimation-eligible'} for name in gate_subsets}
        families[fam], gates[fam] = rows, by_gate
    shrink = sub['c']
    finite_taus = [tau_key(t) for t in table['taus'] if t != INF]
    any_evaluable = any(gates[fam][k]['all']['pass'] for fam in gates if fam in MAIN_FAMILIES for k in finite_taus)
    return {'common': common, 'families': families, 'gates': gates, 'tau_inf_identity': identity,
            'any_main_family_tau_evaluable': any_evaluable,
            'noise_numerator': 'c x max_mask(q_planning_diff_se) (c = 1 except shrinkage; cross_replicate uses '
                               'q_planning_diff_se as a proxy)',
            'shrinkage_c': ({str(q): float(np.quantile(shrink, q)) for q in (0., .1, .5, .9, 1.)}
                            if np.isfinite(shrink).all() and len(shrink) else None)}


def evaluable_taus(prediag_record, family, subset='all'):
    """{tau key: gate pass} of a family on a gate subset."""
    return {key: entry[subset]['pass'] for key, entry in prediag_record['gates'][family].items()}


# ---------------------------------------------------------------- shared cluster bootstrap

class ClusterBootstrap:
    """Cluster multiplicities of one unit (A2 ope_bootstrap.Resampler conventions): clusters sorted by str,
    default_rng(seed).integers(C, size=(draws, C)); m[r, c] = how often cluster c is drawn in replicate r."""
    def __init__(self, keys, *, unit, draws, seed):
        _require(type(draws) is int and draws >= 2 and type(seed) is int and seed >= 0, 'registered draws/seed required')
        self.unit, self.draws, self.seed = unit, draws, seed
        self.order = sorted(set(keys), key=str)
        index = {key: i for i, key in enumerate(self.order)}
        self.row_cluster = np.array([index[key] for key in keys], dtype=np.int64)
        self.n_clusters = len(self.order)
        _require(self.n_clusters > 0, 'no cluster to resample')
        self._sort = np.argsort(self.row_cluster, kind='stable')
        self._starts = np.flatnonzero(np.r_[True, np.diff(self.row_cluster[self._sort]) != 0])
        sample = np.random.default_rng(seed).integers(self.n_clusters, size=(draws, self.n_clusters))
        flat = (np.arange(draws)[:, None] * self.n_clusters + sample).ravel()
        self.m = np.bincount(flat, minlength=draws * self.n_clusters).reshape(draws, self.n_clusters).astype(np.float64)

    def sums(self, values):
        values = np.asarray(values, dtype=np.float64)
        return np.add.reduceat(values[self._sort], self._starts, axis=0)

    def replicate(self, values):
        """sum over rows of m[r, cluster(row)] * values[row] for every replicate r: [draws, ...]."""
        s = self.sums(values)
        return (self.m @ s.reshape(self.n_clusters, -1)).reshape(self.draws, *s.shape[1:])


def interval(replicates, valid, draws, invalid_share_max):
    """Percentile 95% CI and SE (SD of the valid replicates, ddof 1) of one statistic (A2 ``interval``)."""
    replicates = np.asarray(replicates, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(replicates)
    invalid = int(draws - ok.sum())
    out = {'invalid_replicates': invalid, 'ci95': None, 'se': None, 'valid': False}
    if invalid / draws > invalid_share_max or ok.sum() < 2:
        return {**out, 'reason': 'INVALID: invalid replicate share above the registered maximum'}
    lo, hi = (float(v) for v in np.quantile(replicates[ok], CI_LEVELS))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        return {**out, 'reason': 'INVALID: nonfinite or reversed CI'}
    return {**out, 'ci95': [lo, hi], 'se': float(np.std(replicates[ok], ddof=1)), 'valid': True, 'reason': None}


def one_sided(replicates, valid):
    """(p+, p-) = shares of the valid finite replicates with D* <= 0 and with D* >= 0."""
    replicates = np.asarray(replicates, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(replicates)
    if not ok.any():
        return None, None
    return float(np.mean(replicates[ok] <= 0)), float(np.mean(replicates[ok] >= 0))


def _separated(entry):
    """The 95% CI of a statistic excludes 0 in the same direction in both bootstraps."""
    if not isinstance(entry, dict):
        return False
    cis = [(entry.get(u) or {}).get('ci95') for u in UNITS]
    return all(ci is not None for ci in cis) and (all(ci[0] > 0 for ci in cis) or all(ci[1] < 0 for ci in cis))


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


class Cell:
    """Statistics of one (family, variant, subset) over the tau grid: point values [T], per-unit replicates and validity
    [draws, T], subset counts and sample minima."""
    def __init__(self, point, reps, valid, info):
        self.point, self.reps, self.valid, self.info = point, reps, valid, info

    def at(self, j, draws, invalid_share_max, key=None, gate=None):
        cell = {**self.info, 'tau': key}
        if gate is not None:
            cell['gate'] = gate
            cell['descriptive'] = not gate.get('pass', False)
        for stat, values in self.point.items():
            entry = {'mean': _finite(values[j])}
            for unit in self.reps:
                entry[unit] = interval(self.reps[unit][stat][:, j], self.valid[unit][stat][:, j], draws, invalid_share_max)
                if stat == 'delta':
                    entry[unit]['p_plus'], entry[unit]['p_minus'] = one_sided(self.reps[unit][stat][:, j],
                                                                              self.valid[unit][stat][:, j])
            cell[stat] = entry
        for ratio, denominator in (('R', 'M_dr'), ('R_pi', 'M_pi')):  # a CI only with a separated denominator
            if ratio in cell and not _separated(cell.get(denominator)):
                cell[ratio] = R_UNDETERMINED
        return cell

    def record(self, taus, draws, invalid_share_max, gates=None):
        return {tau_key(t): self.at(j, draws, invalid_share_max, tau_key(t),
                                    None if gates is None else gates.get(tau_key(t))) for j, t in enumerate(taus)}


class Engine:
    """Shared-resampling statistics over the estimation population (one multiplicity matrix per unit)."""
    def __init__(self, table, *, draws, seeds, invalid_share_max, minima):
        _require(set(seeds) >= set(UNITS), 'one registered bootstrap seed per unit (game, pitcher)')
        _require(set(minima) == {'games', 'pitchers', 'pas'} and all(type(v) is int and v >= 1 for v in minima.values()),
                 'sample minima: games, pitchers, pas')
        obs = estimation_eligible(table)
        self.n_all = len(obs)
        self.t = {k: (v[obs] if isinstance(v, np.ndarray) and v.shape[:1] == (self.n_all,) else v)
                  for k, v in table.items()}
        self.n = int(obs.sum())
        _require(self.n > 0, 'no estimation-eligible decision')
        _require(np.isfinite(self.t['G']).all(), 'rewards are not attached (attach_rewards after prediag)')
        self.taus, self.families = table['taus'], table['families']
        self.draws, self.minima, self.share = draws, dict(minima), invalid_share_max
        keys = {'game': self.t['game'].tolist(), 'pitcher': self.t['pitcher'].tolist()}
        self.boots = {u: ClusterBootstrap(keys[u], unit=u, draws=draws, seed=seeds[u]) for u in UNITS}
        self._validity = {}

    def subset(self, name):
        """(numerator rows, denominator rows) of a decision subset (also 'A&B'), or the PA auxiliary pa_first_D:0."""
        if name == 'pa_first_D:0':
            return self.t['first_D0'], self.t['first_eligible']
        rows = _table_subset(self.t, name)
        return rows, rows

    def validity(self, name):
        """Replicate validity per unit (the subset has a decision in the replicate) and the subset's sample counts
        (games, PAs, pitchers among its decisions) against the registered minima."""
        if name in self._validity:
            return self._validity[name]
        num, den = self.subset(name)
        rows = np.flatnonzero(den)
        pa = self.t['pa_index'][rows]
        out = {'n': int(num.sum()), 'n_denominator': int(den.sum()), 'pas': int(len(set(pa.tolist()))),
               'games': int(len(set(self.t['game'][rows].tolist()))),
               'pitchers': int(len(set(self.t['pitcher'][rows].tolist()))), 'units': {}}
        out['below_minima'] = bool(out['games'] < self.minima['games'] or out['pas'] < self.minima['pas']
                                   or out['pitchers'] < self.minima['pitchers'])
        for unit, boot in self.boots.items():
            out['units'][unit] = boot.replicate(den.astype(np.float64)) > 0
        self._validity[name] = out
        return out

    def parts(self, f, variant, *, model_dr=None, model_pi=None, rho=None, qa=None, reward=None, dr_uses_pi=None):
        """Per-row parts [n, T] of a variant for family index f (or explicit placebo arrays)."""
        t = self.t
        model_dr = t['model_dr'][:, f, :] if model_dr is None else model_dr
        model_pi = (t['model_pi'][:, f, :] if model_pi is None else model_pi) if model_pi is not False else None
        rho = t['rho'][:, f, :] if rho is None else rho
        qa = t['qa'][:, f] if qa is None else qa
        reward = t['G'] if reward is None else reward
        uses_pi = t['dr_uses_pi'][f] if dr_uses_pi is None else dr_uses_pi
        on = t['on_mask'][:, None]
        w = (rho - t['rho_ref'][:, None]) * on
        resid = np.where(on, reward[:, None] - qa[:, None], 0.)
        corr = row_correction(rho, t['rho_ref'][:, None], on, reward[:, None], qa[:, None])
        base = {'Mfam': model_pi if uses_pi else model_dr, 'Mdr': model_dr}
        if model_pi is not None:
            base['Mpi'] = model_pi
        if variant == 'dr':
            return {**base, 'C': corr}
        if variant.startswith('switch:'):
            return {**base, 'C': corr * (rho <= float(variant.split(':', 1)[1]))}
        if variant == 'ips':
            return {'I': w * np.where(on, reward[:, None], 0.)}
        if variant == 'sndr':
            ref = np.broadcast_to(t['rho_ref'][:, None] * on, rho.shape)
            return {**base, 'pr': rho * resid, 'p': rho, 'rr': ref * resid, 'r': ref}
        raise IntegrityError(f'unknown variant {variant}')

    @staticmethod
    def stats(variant, s, count):
        count = np.asarray(count, dtype=np.float64)[..., None]
        with np.errstate(divide='ignore', invalid='ignore'):
            if variant == 'ips':
                delta = s['I'] / count
                return {'delta': delta}, np.ones(delta.shape, dtype=bool)
            model = s['Mfam'] / count
            if variant == 'sndr':
                corr = s['pr'] / s['p'] - s['rr'] / s['r']
                ok = (s['p'] > 0) & (s['r'] > 0)
            else:
                corr = s['C'] / count
                ok = np.ones(model.shape, dtype=bool)
            delta = model + corr
            out = {'delta': delta, 'M_dr': s['Mdr'] / count, 'C': corr}
            out['R'] = delta / out['M_dr']
            if 'Mpi' in s:
                out['M_pi'] = s['Mpi'] / count
                out['R_pi'] = delta / out['M_pi']
            return out, ok

    def cell(self, parts, variant, sub):
        num, den = self.subset(sub)
        val = self.validity(sub)
        w = num.astype(np.float64)[:, None]
        point, _ = self.stats(variant, {k: (v * w).sum(axis=0) for k, v in parts.items()}, den.sum())
        reps, valid = {}, {}
        for unit, boot in self.boots.items():
            stats, ok_u = self.stats(variant, {k: boot.replicate(v * w) for k, v in parts.items()},
                                     boot.replicate(den.astype(np.float64)))
            reps[unit] = stats
            valid[unit] = {s: val['units'][unit][:, None] & ok_u & np.isfinite(v) for s, v in stats.items()}
        info = {k: val[k] for k in ('n', 'n_denominator', 'pas', 'games', 'pitchers', 'below_minima')}
        info.update(subset=sub, variant=variant, unit='WE/PA' if sub.startswith('pa_') else 'WE/decision')
        return Cell(point, reps, valid, info)

    def family_cell(self, family, variant, sub):
        return self.cell(self.parts(self.families.index(family), variant), variant, sub)

    def record(self, cell, gates=None):
        return cell.record(self.taus, self.draws, self.share, gates)


def paired(a, b, stat='delta'):
    """a - b on the shared replicates (both valid)."""
    return Cell({stat: a.point[stat] - b.point[stat]}, {u: {stat: a.reps[u][stat] - b.reps[u][stat]} for u in a.reps},
                {u: {stat: a.valid[u][stat] & b.valid[u][stat]} for u in a.reps}, dict(a.info))


class DrawMean:
    """Streaming mean over placebo draws of one statistic, replicate by replicate (paired with every other cell on the
    shared resampling), plus each draw's full-sample points (for the band). No draw's replicates are kept."""
    def __init__(self, stat='delta'):
        self.stat, self.k, self.point, self.reps, self.valid, self.info = stat, 0, None, None, None, None
        self.points = {}

    def add(self, cell):
        for name, values in cell.point.items():
            self.points.setdefault(name, []).append(np.array(values, dtype=np.float64))
        s = self.stat
        if self.k == 0:
            self.point = np.array(cell.point[s], dtype=np.float64)
            self.reps = {u: np.array(cell.reps[u][s], dtype=np.float64) for u in cell.reps}
            self.valid = {u: np.array(cell.valid[u][s], dtype=bool) for u in cell.reps}
            self.info = dict(cell.info)
        else:
            self.point = self.point + cell.point[s]
            for u in self.reps:
                self.reps[u] = self.reps[u] + cell.reps[u][s]
                self.valid[u] = self.valid[u] & cell.valid[u][s]
        self.k += 1
        return self

    def cell(self):
        _require(self.k > 0, 'no placebo draw')
        s = self.stat
        return Cell({s: self.point / self.k}, {u: {s: r / self.k} for u, r in self.reps.items()},
                    {u: {s: v} for u, v in self.valid.items()}, dict(self.info))

    def band(self, taus):
        out = {}
        for j, t in enumerate(taus):
            column = np.array([v[j] for v in self.points[self.stat]], dtype=np.float64)
            column = column[np.isfinite(column)]
            out[tau_key(t)] = ({'mean': float(column.mean()), 'band_2.5_97.5': [float(np.quantile(column, .025)),
                                                                                 float(np.quantile(column, .975))]}
                               if len(column) else None)
        return {'draws': self.k, 'per_tau': out}


# ---------------------------------------------------------------- trend, max-t, bounds

def trend(cell, taus, used_keys, draws, invalid_share_max, stat='delta'):
    """OLS slope of the per-tau point estimate on log(1/tau) over ``used_keys`` (evaluable finite taus), with the
    shared-bootstrap percentile CI per unit (a replicate is valid when every used tau is valid). Descriptive."""
    used = [j for j, t in enumerate(taus) if t != INF and tau_key(t) in used_keys]
    out = {'x': 'log(1/tau)', 'taus': [tau_key(taus[j]) for j in used], 'slope': None, 'descriptive': True}
    if len(used) < 2:
        return {**out, 'reason': 'fewer than two evaluable finite taus'}
    x = np.log(1. / np.array([taus[j] for j in used]))
    xc = x - x.mean()

    def slope(y):
        return ((y - y.mean(axis=-1, keepdims=True)) * xc).sum(axis=-1) / (xc @ xc)
    out['slope'] = _finite(slope(cell.point[stat][used]))
    for unit in cell.reps:
        reps = cell.reps[unit][stat][:, used]
        ok = cell.valid[unit][stat][:, used].all(axis=1)
        out[unit] = interval(slope(np.where(ok[:, None], reps, 0.)), ok, draws, invalid_share_max)
    return out


def max_t(cell, taus, used_keys, draws, invalid_share_max, alpha, stat='delta'):
    """One-sided joint max-t over ``used_keys`` (the evaluable finite T taus) with ONE standard error per tau, SE_tau =
    max(SE_game, SE_pitcher) (each the SD, ddof 1, of that bootstrap's replicates valid at every used tau): t_tau =
    delta_tau / SE_tau; per bootstrap the null T* = max_tau (rep_tau - delta_tau) / SE_tau on its centered shared
    replicates and p = share(T* >= max t); the contrast p is the larger of the two; significant when <= alpha."""
    used = [j for j, t in enumerate(taus) if t != INF and tau_key(t) in used_keys]
    out = {'taus': [tau_key(taus[j]) for j in used], 'one_sided': 'H0: every delta <= 0', 'alpha': alpha, 'p': None,
           'significant': False, 'se_rule': 'SE_tau = max(SE_game, SE_pitcher), shared by t_obs and both nulls'}
    if not used:
        return {**out, 'reason': 'no evaluable finite tau'}
    point = cell.point[stat][used]
    reps, ok, ses = {}, {}, {}
    for unit in cell.reps:
        reps[unit] = cell.reps[unit][stat][:, used]
        ok[unit] = cell.valid[unit][stat][:, used].all(axis=1)
        if (draws - ok[unit].sum()) / draws > invalid_share_max or ok[unit].sum() < 2:
            out[unit] = {'valid': False, 'reason': 'INVALID: invalid replicate share above the registered maximum'}
            return out
        ses[unit] = reps[unit][ok[unit]].std(axis=0, ddof=1)
    se = np.maximum.reduce([ses[u] for u in cell.reps])
    if not (se > 0).all():
        return {**out, 'reason': 'zero bootstrap SE at a used tau'}
    t_obs = point / se
    out.update(se_tau={tau_key(taus[j]): float(v) for j, v in zip(used, se)},
               t={tau_key(taus[j]): float(v) for j, v in zip(used, t_obs)}, max_t=float(t_obs.max()))
    ps = []
    for unit in cell.reps:
        null = ((reps[unit][ok[unit]] - point) / se).max(axis=1)
        p = float(np.mean(null >= t_obs.max()))
        out[unit] = {'valid': True, 'se': {tau_key(taus[j]): float(v) for j, v in zip(used, ses[unit])}, 'p_value': p,
                     'critical_95': float(np.quantile(null, .95)), 'replicates': int(ok[unit].sum())}
        ps.append(p)
    out['p'] = max(ps)
    out['significant'] = out['p'] <= alpha
    return out


def bounds(table, family):
    """Worst-case bounds of the subset-all DR mean with the excluded no-reward PAs added back: per excluded PA e, C_e(g) =
    sum over its SUPPORTED rows of (rho' - rho_r)(g - q_dr[a]) for g in {0, 1} (lo/hi = min/max per PA), plus their
    rows' model terms in the numerator and their rows in the denominator."""
    f = table['families'].index(family)
    obs, on = table['observed'], table['on_mask']
    w = (table['rho'][:, f, :] - table['rho_ref'][:, None]) * on[:, None]
    qa = np.where(on, table['qa'][:, f], 0.)
    model = table['model_pi' if table['dr_uses_pi'][f] else 'model_dr'][:, f, :]
    observed_sum = (model[obs] + w[obs] * (table['G'][obs][:, None] - qa[obs][:, None])).sum(axis=0)
    un = ~obs
    out = {'family': family, 'unobserved_decisions': int(un.sum()),
           'unobserved_pas': int(len(set(table['pa_index'][un].tolist()))), 'rule': 'g in {0, 1} per excluded PA'}
    lo, hi = observed_sum.copy(), observed_sum.copy()
    if un.any():
        pa = table['pa_index'][un]
        starts = np.flatnonzero(np.r_[True, pa[1:] != pa[:-1]])
        model_sum = np.add.reduceat(model[un], starts, axis=0)
        c0 = np.add.reduceat(-w[un] * qa[un][:, None], starts, axis=0)
        c1 = np.add.reduceat(w[un] * (1. - qa[un][:, None]), starts, axis=0)
        lo += (model_sum + np.minimum(c0, c1)).sum(axis=0)
        hi += (model_sum + np.maximum(c0, c1)).sum(axis=0)
    total = len(obs)
    out['bounds'] = {tau_key(t): [_finite(lo[j] / total), _finite(hi[j] / total)] for j, t in enumerate(table['taus'])}
    return out


# ---------------------------------------------------------------- placebos

class FlatRows:
    """The estimation-eligible decisions' mask entries as one flat array (row-major), for the vectorised K-draw perm
    placebo: the row-wise kl_policy arithmetic on segments (np.*.reduceat), checked against kl_policy in tests."""
    def __init__(self, table):
        obs = estimation_eligible(table)
        mask, ref = table['mask'][obs], table['ref'][obs]
        r, c = np.nonzero(mask)
        self.row, self.col, self.n = r, c, mask.shape[0]
        self.mask_full, self.ref_full = mask, ref
        self.starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        _require(len(self.starts) == self.n, 'every decision needs a nonempty mask')
        self.lengths = np.diff(np.r_[self.starts, len(r)])
        bc = ref / ref.sum(axis=1, keepdims=True)
        self.log_bc, self.ref = np.log(bc[r, c]), ref[r, c]
        self.q_pi, self.q_dr = table['q_pi'][obs][r, c], table['q_dr'][obs][r, c]
        logging = table['logging'][obs]
        self.logging = logging[r, c]
        self.logging_off = np.where(mask, 0., logging).sum(axis=1)  # logging mass of the off-mask actions
        flat = np.full(mask.shape, -1, dtype=np.int64)
        flat[r, c] = np.arange(len(r))
        a, self.on = table['a'][obs], table['on_mask'][obs]
        self.pos_a = np.where(self.on, flat[np.arange(self.n), a], 0)
        self.b_safe = np.where(self.on, table['b_a'][obs], 1.)
        self.sha8 = table['sha8'][obs]

    def permutation(self, seed, k):
        """Flat index permutation of draw k: perm_rng(seed, k, sha8) inside every row's mask segment."""
        out = np.arange(len(self.row))
        for i in np.flatnonzero(self.lengths > 1):
            s, m = self.starts[i], self.lengths[i]
            out[s:s + m] = s + perm_rng(seed, k, self.sha8[i]).permutation(m)
        return out

    def tilt(self, q, tau):
        """pi on the flat entries ([..., E]) at temperature tau (the reference itself at tau = inf)."""
        if tau == INF:
            return np.broadcast_to(self.ref, np.shape(q)).copy()
        qmax = np.maximum.reduceat(q, self.starts, axis=-1)
        lw = self.log_bc + (q - np.take(qmax, self.row, axis=-1)) / tau
        lmax = np.maximum.reduceat(lw, self.starts, axis=-1)
        w = np.exp(lw - np.take(lmax, self.row, axis=-1))
        return w / np.take(np.add.reduceat(w, self.starts, axis=-1), self.row, axis=-1)

    def kl_rows(self, pi):
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(pi > 0, pi * np.log(pi / self.ref), 0.)
        return np.add.reduceat(terms, self.starts, axis=-1)

    def mean_kl(self, pi):
        return float(self.kl_rows(pi).mean())

    def segment(self, values):
        return np.add.reduceat(values, self.starts, axis=-1)

    def identity(self, pi, q_hat, model, reward):
        """The row identity of one flat policy: S per row = logging_off model + sum over the mask entries of
        logging(a) row_term(a; G') (``row_term``, the estimator's own term); max |S|."""
        safe = np.where(self.logging > 0, self.logging, 1.)
        terms = row_term(np.take(model, self.row), pi / safe, self.ref / safe, self.logging > 0, reward, q_hat)
        s = self.logging_off * model + self.segment(np.where(self.logging > 0, self.logging * terms, 0.))
        return float(np.max(np.abs(s))) if len(s) else 0.


def common_tau(flat, perms, target, tau, *, factor, iterations, tol, chunk=25):
    """One placebo temperature for every draw: log-space bisection on [tau/factor, factor tau] of the mean KL over the
    draws and decisions to ``target`` within ``tol`` (relative); outside the attainable range NOT_MATCHABLE (tau_p =
    tau). Returns (tau_p, mean KL at tau_p, status)."""
    if not target > 0:
        return INF, 0., 'MATCHED'

    def mean_kl(tp):
        values = [flat.mean_kl(flat.tilt(flat.q_pi[perms[i:i + chunk]], tp)) for i in range(0, len(perms), chunk)]
        sizes = [len(perms[i:i + chunk]) for i in range(0, len(perms), chunk)]
        return float(np.dot(values, sizes) / sum(sizes))
    lo, hi = tau / factor, tau * factor
    k_lo, k_hi = mean_kl(lo), mean_kl(hi)
    if not (k_lo >= target >= k_hi):
        return tau, mean_kl(tau), 'NOT_MATCHABLE'
    for tp, value in ((lo, k_lo), (hi, k_hi)):
        if abs(value - target) <= tol * target:
            return tp, value, 'MATCHED'
    for _ in range(iterations):
        mid = math.sqrt(lo * hi)
        value = mean_kl(mid)
        if abs(value - target) <= tol * target:
            return mid, value, 'MATCHED'
        lo, hi = (mid, hi) if value > target else (lo, mid)
    return mid, value, 'NOT_CONVERGED'


class PermSetup:
    """The G-free part of the perm placebo: draws, the common placebo temperatures, the placebo row identity and the
    placebo/real KL ratio within D_1se."""
    def __init__(self, flat, perms, tau_p, record, identity, seed):
        self.flat, self.perms, self.tau_p, self.record, self.identity, self.seed = flat, perms, tau_p, record, identity, seed


def perm_setup(table, *, draws_k, seed, kl_tol, factor, iterations, checkpoint=None):
    """Perm draws (perm_rng per draw and row), one common placebo temperature per tau matching T's mean KL on 'all' (the
    estimation-eligible decisions), the row identity of every perm and estimator_stress policy (EstimatorSuspect on a
    violation) and the KL ratio placebo/real within D_1se (D membership from the real q_pi)."""
    tick = checkpoint or (lambda: None)
    flat = FlatRows(table)
    obs = estimation_eligible(table)
    f = table['families'].index('temperature')
    d1se = table['D'][obs, [margin_key(m) for m in table['margins']].index('1se')]
    worst_real = 0.  # the flat arithmetic against the kl_policy columns of the scan, every row and tau
    for j, tau in enumerate(table['taus']):
        pi = flat.tilt(flat.q_pi, tau)
        worst_real = max(worst_real, float(np.abs(np.where(flat.on, pi[flat.pos_a] / flat.b_safe, 0.)
                                                  - table['rho'][obs, f, j]).max()),
                         float(np.abs(flat.segment((pi - flat.ref) * flat.q_dr) - table['model_dr'][obs, f, j]).max()))
    if not worst_real <= IDENTITY_TOL:
        raise EstimatorSuspect(f'the vectorised policy differs from kl_policy on the real family: {worst_real!r}')
    perms = np.stack([flat.permutation(seed, k) for k in range(draws_k)]) if draws_k else np.zeros((0, len(flat.row)),
                                                                                                    dtype=np.int64)
    tick()
    record, tau_p, worst, worst_perm = {}, [], 0., 0.
    sample = np.arange(0, flat.n, max(1, flat.n // 500))  # every k-th row: at least 500 rows (all when fewer)
    for j, tau in enumerate(table['taus']):
        target = float(table['kl'][obs, f, j].mean())
        tp, value, status = (common_tau(flat, perms, target, tau, factor=factor, iterations=iterations, tol=kl_tol)
                             if tau != INF and draws_k else (INF if tau == INF else tau, 0., 'MATCHED' if tau == INF
                                                                                        else 'NO_DRAWS'))
        tau_p.append(tp)
        kl_d = []
        for k in range(draws_k):
            q_perm, dr_perm = flat.q_pi[perms[k]], flat.q_dr[perms[k]]
            pi = flat.tilt(q_perm, tp)
            model = flat.segment((pi - flat.ref) * flat.q_dr)
            model_s = flat.segment((pi - flat.ref) * dr_perm)
            for g in IDENTITY_G:
                worst = max(worst, flat.identity(pi, flat.q_dr, model, g), flat.identity(pi, dr_perm, model_s, g))
            if d1se.any():
                kl_d.append(float(flat.kl_rows(pi)[d1se].mean()))
        if not worst <= IDENTITY_TOL:
            raise EstimatorSuspect(f'row identity failed for a perm / estimator_stress policy at tau {tau_key(tau)}: '
                                   f'max |S| = {worst!r}')
        if tp != INF and draws_k:
            worst_perm = max(worst_perm, perm_vector_check(flat, perms, tp, sample))
        real_d = float(table['kl'][obs, f, j][d1se].mean()) if d1se.any() else None
        record[tau_key(tau)] = {'target_mean_kl': target, 'tau_p': None if tp == INF else tp, 'mean_kl': value,
                                'relative_kl_mismatch': None if not target > 0 else abs(value / target - 1),
                                'status': status, 'mean_kl_D1se_placebo': float(np.mean(kl_d)) if kl_d else None,
                                'mean_kl_D1se_real': real_d,
                                'kl_ratio_D1se_placebo_over_real': (float(np.mean(kl_d)) / real_d
                                                                    if kl_d and real_d else None)}
        tick()
    if not worst_perm <= IDENTITY_TOL:
        raise EstimatorSuspect(f'perm policy vectors differ from kl_policy: max |diff| = {worst_perm!r}')
    return PermSetup(flat, perms, tau_p, record, {
        'max_abs': worst, 'G': list(IDENTITY_G), 'tolerance': IDENTITY_TOL,
        'policies': 'perm and estimator_stress, every draw and tau',
        'kl_policy_check': {'real_family_every_row_max_abs': worst_real,
                            'perm_vectors_max_abs': worst_perm, 'perm_rows_checked': int(len(sample)),
                            'perm_draws_checked': sorted({0, max(draws_k - 1, 0)}) if draws_k else [],
                            'rule': 'flat placebo arithmetic == rollout_policy.kl_policy within 1e-12'}}, seed)


def perm_vector_check(flat, perms, tau, sample):
    """max |flat perm policy - kl_policy| over the sampled rows of the first and last draw at placebo temperature tau."""
    worst = 0.
    for k in sorted({0, len(perms) - 1}):
        q_perm = flat.q_pi[perms[k]]
        pi = flat.tilt(q_perm, tau)
        for i in sample:
            s, m = flat.starts[i], flat.lengths[i]
            q = np.full(flat.mask_full.shape[1], -np.inf)
            q[flat.col[s:s + m]] = q_perm[s:s + m]
            expected = kl_policy(q, flat.ref_full[i], flat.mask_full[i], tau)
            got = np.zeros_like(expected)
            got[flat.col[s:s + m]] = pi[s:s + m]
            worst = max(worst, float(np.abs(got - expected).max()))
    return worst


def perm_placebo(engine, table, setup, *, subsets, checkpoint=None):
    """(a) perm and (e) estimator_stress cells at the common placebo temperatures of ``setup`` (q_dr kept / q_dr
    permuted alike). Returns ({subset: perm DrawMean}, {subset: stress DrawMean})."""
    flat, taus = setup.flat, table['taus']
    f = table['families'].index('temperature')
    rho_ref = table['rho_ref'][estimation_eligible(table)]
    perm = {s: DrawMean() for s in subsets}
    stress = {s: DrawMean() for s in subsets}
    for k in range(len(setup.perms)):
        if checkpoint is not None:
            checkpoint()
        q_perm, dr_perm = flat.q_pi[setup.perms[k]], flat.q_dr[setup.perms[k]]
        model, model_s, rho = (np.zeros((flat.n, len(taus))) for _ in range(3))
        for j, tp in enumerate(setup.tau_p):
            if tp == INF:
                rho[:, j] = rho_ref
                continue
            pi = flat.tilt(q_perm, tp)
            model[:, j] = flat.segment((pi - flat.ref) * flat.q_dr)
            model_s[:, j] = flat.segment((pi - flat.ref) * dr_perm)
            rho[:, j] = np.where(flat.on, pi[flat.pos_a] / flat.b_safe, 0.)
        qa = np.where(flat.on, flat.q_dr[flat.pos_a], 0.)
        qa_s = np.where(flat.on, dr_perm[flat.pos_a], 0.)
        for sub in subsets:
            perm[sub].add(engine.cell(engine.parts(f, 'dr', model_dr=model, model_pi=False, rho=rho, qa=qa,
                                                   dr_uses_pi=False), 'dr', sub))
            stress[sub].add(engine.cell(engine.parts(f, 'dr', model_dr=model_s, model_pi=False, rho=rho, qa=qa_s,
                                                     dr_uses_pi=False), 'dr', sub))
    return perm, stress


def other_game_candidates(pitcher, game, usable):
    """{PA: donor PAs}: the usable PAs of the same first-row pitcher in other games (never the PA itself or its game);
    a PA without any is absent (it keeps its own dG)."""
    groups = {}
    for i in np.flatnonzero(usable):
        groups.setdefault(pitcher[i], []).append(i)
    out = {}
    for members in groups.values():
        members = np.array(members, dtype=np.int64)
        for i in members:
            others = members[game[members] != game[i]]
            if len(others):
                out[int(i)] = others
    return out


def outcome_shuffle_null(engine, *, draws_k, seed, subsets, checkpoint=None):
    """DESCRIPTIVE: dG = G - WE_start of a donor PA of the same first-row pitcher in ANOTHER game (drawn with replacement
    by SeedSequence([seed, k]); never the PA itself or its game; a PA whose pitcher has no other-game eligible PA, or
    without WE_start, keeps its own dG), G' = WE_start(own) + dG(donor); the temperature family's DR and IPS-only
    re-evaluated. Returns ({(variant, subset): DrawMean}, counts)."""
    t = engine.t
    f = engine.families.index('temperature')
    pa = t['pa_index']
    starts = np.flatnonzero(np.r_[True, pa[1:] != pa[:-1]])
    owner = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(pa)]))
    reward, we_start = t['G'][starts], t['we_start'][starts]
    pitcher, game = t['pitcher'][starts], t['game'][starts]
    delta = reward - we_start
    usable = np.isfinite(we_start)
    candidates = other_game_candidates(pitcher, game, usable)
    receivers = np.array(sorted(candidates), dtype=np.int64)
    out = {(v, s): DrawMean() for v in ('dr', 'ips') for s in subsets}
    for k in range(draws_k):
        if checkpoint is not None:
            checkpoint()
        u = np.random.default_rng(np.random.SeedSequence([int(seed), int(k)])).random(len(starts))
        shuffled = reward.copy()
        for i in receivers:
            cands = candidates[i]
            shuffled[i] = we_start[i] + delta[cands[int(u[i] * len(cands))]]
        g = shuffled[owner]
        for (variant, sub), acc in out.items():
            acc.add(engine.cell(engine.parts(f, variant, reward=g), variant, sub))
    return out, {'pas': int(len(starts)), 'pas_with_other_game_donors': int(len(receivers)),
                 'pas_keeping_own_dG': int(len(starts) - len(receivers)),
                 'pas_without_we_start': int((~usable).sum()), 'label': 'descriptive (sets no label)'}


# ---------------------------------------------------------------- composition

def inf_identity(table):
    """tau = inf is the reference exactly: every family's model terms are 0 and rho' == rho_r bit for bit (D = 0)."""
    j = table['taus'].index(INF)
    n = len(table['observed'])
    record = {key: (float(np.abs(table[key][:, :, j]).max()) if n else 0.) for key in ('model_dr', 'model_pi')}
    record['max_abs_rho_minus_rho_ref'] = (float(np.abs(table['rho'][:, :, j] - table['rho_ref'][:, None]).max())
                                           if n else 0.)
    record['zero'] = all(v == 0. for v in record.values())
    return record


def analysis(table, prediag_record, *, rule, perm=None, checkpoint=None):
    """Every G-dependent section on the shared resampling and the inputs of ``decide``. ``rule``: primary tau,
    variants, main_families, margins, bootstrap {draws, seeds}, invalid_share_max, minima, placebos, sensitivities,
    alpha, mde multipliers; ``perm``: the G-free PermSetup (perm_setup). Returns (sections, per_tau.csv sections,
    decision inputs)."""
    tick = checkpoint or (lambda: None)
    eng = Engine(table, draws=rule['bootstrap']['draws'], seeds=rule['bootstrap']['seeds'],
                 invalid_share_max=rule['invalid_share_max'], minima=rule['minima'])
    taus, draws, share = table['taus'], eng.draws, rule['invalid_share_max']
    gates = prediag_record['gates']
    gate_all = {fam: {key: entry['all'] for key, entry in gates[fam].items()} for fam in gates}
    primary_key = tau_key(rule['primary_tau'])
    out, csv, cells = {'tau_inf_identity': prediag_record['tau_inf_identity']}, [], {}
    dose = {}
    for fam in rule['main_families']:
        dose[fam] = {}
        for variant in rule['variants']:
            tick()
            cells[(fam, variant)] = eng.family_cell(fam, variant, 'all')
            dose[fam][variant] = eng.record(cells[(fam, variant)], gate_all[fam])
            csv.append(('dose_response', fam, variant, 'all', dose[fam][variant]))
    out['dose_response'] = {**dose, 'note': 'shrinkage is descriptive only'}
    real = cells[('temperature', 'dr')]
    subsets = {'all': dose['temperature']['dr']}
    for sub in [*[f'D:{margin_key(m)}' for m in rule['margins']], 'not_D:0', 'two_strike']:
        tick()
        cells[('temperature', 'dr', sub)] = eng.family_cell('temperature', 'dr', sub)
        sub_gates = None if sub not in gates['temperature'][primary_key] else {
            key: entry[sub] for key, entry in gates['temperature'].items()}
        subsets[sub] = eng.record(cells[('temperature', 'dr', sub)], sub_gates)
        csv.append(('subsets', 'temperature', 'dr', sub, subsets[sub]))
    out['subsets'] = {'family': 'temperature', 'variant': 'dr', 'confident_slice': 'D:1se', 'by_subset': subsets}
    tick()
    pa_level = eng.record(eng.family_cell('temperature', 'dr', 'pa_first_D:0'), gate_all['temperature'])
    out['pa_level'] = {'pa_first_D:0': pa_level}
    csv.append(('pa_level', 'temperature', 'dr', 'pa_first_D:0', pa_level))
    confirm = {name: cells[('temperature', 'dr')] if sub == 'all' else cells[('temperature', 'dr', sub)]
               for name, sub in CONFIRMATORY.items()}
    sens = {'variants': 'IPS-only, SN-DR and the switch caps are the dose_response variants of each family'}
    spos = {}
    for name, sub in CONFIRMATORY.items():
        tick()
        spos[name] = eng.family_cell('temperature', 'dr', 'no_positivity_pa' if sub == 'all'
                                     else f'{sub}&no_positivity_pa')
    sens['s_pos'] = {'rule': 'whole PAs containing a logging-positivity node excluded; retention = same sign and the '
                             '95% CI excludes 0 in the same direction in both bootstraps',
                     'by_contrast': {n: eng.record(c, None) for n, c in spos.items()}}
    csv.append(('sensitivities', 'temperature', 'dr', 'no_positivity_pa', sens['s_pos']['by_contrast']['P']))
    if rule['sensitivities']['cross_replicate']:
        tick()
        cell = eng.family_cell('cross_replicate', 'dr', 'all')
        sens['cross_replicate'] = {'rule': 'policy from q_reference, DR with q_planning',
                                   'cell': eng.record(cell, gate_all.get('cross_replicate')),
                                   'paired_temperature_minus_cross': eng.record(paired(real, cell))}
        csv.append(('sensitivities', 'cross_replicate', 'dr', 'all', sens['cross_replicate']['cell']))
    out['sensitivities'] = sens
    pl = {'reading_rule': 'paired differences real - placebo on the shared bootstrap resampling; bands (2.5-97.5% of the '
                          'K draw means) only for the stochastic placebos', 'policy_side_only': True,
          'inf': prediag_record['tau_inf_identity'],
          'donor': {'status': 'NOT_IMPLEMENTED', 'reason': 'a same-count / same-mask donor does not exist for every '
                    'decision; dropping or defaulting those changes the placebo population (needs a design decision)'}}
    confirm_subs = list(CONFIRMATORY.values())
    if rule['placebos']['neg']:
        pl['neg'] = {}
        for name, sub in CONFIRMATORY.items():
            tick()
            cell = eng.family_cell('neg', 'dr', sub)
            pl['neg'][name] = {'cell': eng.record(cell), 'paired_real_minus_neg': eng.record(paired(confirm[name], cell))}
        csv.append(('placebos', 'neg', 'dr', 'all', pl['neg']['P']['cell']))
    perm_paired = {}
    if perm is not None and len(perm.perms):
        accs_perm, accs_stress = perm_placebo(eng, table, perm, subsets=confirm_subs, checkpoint=tick)
        for label, accs in (('perm', accs_perm), ('estimator_stress', accs_stress)):
            entry = {'draws': len(perm.perms), 'seed': perm.seed, 'tau_p': perm.record, 'row_identity': perm.identity,
                     'by_contrast': {}}
            for name, sub in CONFIRMATORY.items():
                mean = accs[sub].cell()
                diff = paired(confirm[name], mean)
                if label == 'perm':
                    perm_paired[name] = diff
                entry['by_contrast'][name] = {'band': accs[sub].band(taus), 'mean': eng.record(mean),
                                              'paired_real_minus_mean': eng.record(diff)}
            pl[label] = entry
            csv.append(('placebos', label, 'dr', 'all', entry['by_contrast']['P']['mean']))
    if rule['placebos']['outcome_shuffle_null_draws']:
        p = rule['placebos']
        accs, info = outcome_shuffle_null(eng, draws_k=p['outcome_shuffle_null_draws'],
                                          seed=p['outcome_shuffle_null_seed'], subsets=confirm_subs, checkpoint=tick)
        j = taus.index(rule['primary_tau'])
        entry = {'draws': p['outcome_shuffle_null_draws'], 'seed': p['outcome_shuffle_null_seed'], **info,
                 'scope': f'tau {primary_key} x confirmatory contrasts x (dr, ips)', 'by_contrast': {}}
        for name, sub in CONFIRMATORY.items():
            entry['by_contrast'][name] = {}
            for variant in ('dr', 'ips'):
                cell = accs[(variant, sub)].cell().at(j, draws, share, primary_key)
                excludes = all((cell['delta'][u]['ci95'] is not None and
                                (cell['delta'][u]['ci95'][0] > 0 or cell['delta'][u]['ci95'][1] < 0)) for u in UNITS)
                entry['by_contrast'][name][variant] = {**cell, 'ci_excludes_zero_in_both_bootstraps': excludes}
        pl['outcome_shuffle_null'] = entry
    out['placebos'] = pl
    used = [k for k, ok in evaluable_taus(prediag_record, 'temperature').items() if ok and k != 'inf']
    out['trend'] = {'temperature': trend(real, taus, used, draws, share)}
    if 'shrinkage' in rule['main_families']:
        used_s = [k for k, ok in evaluable_taus(prediag_record, 'shrinkage').items() if ok and k != 'inf']
        out['trend']['shrinkage'] = trend(cells[('shrinkage', 'dr')], taus, used_s, draws, share)
    out['max_t'] = {'temperature': max_t(real, taus, used, draws, share, rule['alpha'])}
    out['bounds'] = {fam: bounds(table, fam) for fam in rule['main_families']}
    out['bounds']['H_P'] = {'family': 'temperature', 'tau': primary_key, 'subset': 'all', 'variant': 'dr',
                            'bounds': out['bounds']['temperature']['bounds'][primary_key]}
    out['bootstrap'] = {'draws': draws, 'seeds': dict(rule['bootstrap']['seeds']), 'invalid_share_max': share,
                        'sample_minima': dict(rule['minima']), 'clusters': {u: b.n_clusters for u, b in eng.boots.items()},
                        'ci_levels': list(CI_LEVELS), 'pitcher_cluster': "the PA's first-row pitcher (A2 convention)",
                        'replicate_invalid': 'no decision of the subset in the replicate, or a non-finite statistic',
                        'paired': 'one multiplicity matrix per unit for every family, variant, subset, tau and placebo',
                        'trend_seed_use': 'registered; unused: trend and max-t use the shared replicates (paired)'}
    inputs = decision_inputs(eng, confirm, perm_paired, spos, real, out['max_t']['temperature'], used, prediag_record,
                             rule)
    return out, csv, inputs


def decision_inputs(eng, confirm, perm_paired, spos, real, maxt, used, prediag_record, rule):
    """Every number ``decide`` reads: the confirmatory cells at the primary tau and, for path (ii), the studentized
    statistic, the perm paired CI and S-pos of every tau in E (subset all)."""
    j = eng.taus.index(rule['primary_tau'])
    key = tau_key(rule['primary_tau'])
    keys = [tau_key(t) for t in eng.taus]
    contrasts = {}
    for name, sub in CONFIRMATORY.items():
        cell = confirm[name].at(j, eng.draws, eng.share, key)
        gate = prediag_record['gates']['temperature'][key][sub]
        reasons = []
        if not gate['pass']:
            reasons.append('eligibility gate failed: ' + ', '.join(k for k, ok in gate['checks'].items() if not ok)
                           if 'checks' in gate else 'eligibility gate failed')
        if cell['below_minima']:
            reasons.append(f'sample minima: games {cell["games"]}, PAs {cell["pas"]}, pitchers {cell["pitchers"]}')
        for unit in UNITS:
            if not cell['delta'][unit]['valid']:
                reasons.append(f'{unit} bootstrap INVALID ({cell["delta"][unit]["invalid_replicates"]} invalid)')
        perm = perm_paired.get(name)
        perm_cell = None if perm is None else perm.at(j, eng.draws, eng.share, key)
        spos_cell = spos[name].at(j, eng.draws, eng.share, key)
        contrasts[name] = {
            'subset': sub, 'evaluable': not reasons, 'not_evaluable_reasons': reasons, 'gate': gate,
            'delta': cell['delta']['mean'], 'se': {u: cell['delta'][u]['se'] for u in UNITS},
            'ci95': {u: cell['delta'][u]['ci95'] for u in UNITS},
            'p_plus': {u: cell['delta'][u].get('p_plus') for u in UNITS},
            'p_minus': {u: cell['delta'][u].get('p_minus') for u in UNITS},
            'perm_paired_ci': None if perm_cell is None else {u: perm_cell['delta'][u]['ci95'] for u in UNITS},
            's_pos': {'delta': spos_cell['delta']['mean'], 'ci': {u: spos_cell['delta'][u]['ci95'] for u in UNITS}}}
    path_ii = {}
    t_values, se_tau = maxt.get('t') or {}, maxt.get('se_tau') or {}
    for k in used:
        jj = keys.index(k)
        c = real.at(jj, eng.draws, eng.share, k)
        perm = perm_paired.get('P')
        perm_cell = None if perm is None else perm.at(jj, eng.draws, eng.share, k)
        spos_cell = spos['P'].at(jj, eng.draws, eng.share, k)
        path_ii[k] = {'delta': c['delta']['mean'], 'se_tau': se_tau.get(k),
                      'studentized': t_values.get(k),  # the max-t statistic itself: delta / max(SE_game, SE_pitcher)
                      'perm_paired_ci': None if perm_cell is None else {u: perm_cell['delta'][u]['ci95'] for u in UNITS},
                      's_pos': {'delta': spos_cell['delta']['mean'],
                                'ci': {u: spos_cell['delta'][u]['ci95'] for u in UNITS}}}
    return {'study': {'estimator_suspect': False, 'reasons': [],
                      'note': 'INVALID and ESTIMATOR_SUSPECT stop the run before estimates (failed.json)'},
            'contrasts': contrasts,
            'max_t': {'E': list(used), 'p': maxt.get('p'), 'significant': maxt.get('significant', False)},
            'path_ii': path_ii, 'alpha': rule['alpha'], 'mde_multipliers': dict(rule['mde_multipliers']),
            'primary_tau': key, 'stage_label': rule['stage_label'],
            'any_main_family_tau_evaluable': bool(prediag_record['any_main_family_tau_evaluable'])}


# ---------------------------------------------------------------- the decision procedure

def _ci_above(ci, zero=0.):
    return ci is not None and ci[0] > zero


def _ci_below(ci, zero=0.):
    return ci is not None and ci[1] < zero


def _gain_evidence(perm_ci, s_pos):
    """(perm paired CI > 0 in both bootstraps, S-pos keeps a gain, S-pos keeps a harm)."""
    perm_ok = perm_ci is not None and all(_ci_above(perm_ci.get(u)) for u in UNITS)
    plus = s_pos is not None and s_pos['delta'] is not None and s_pos['delta'] > 0 and all(
        _ci_above(s_pos['ci'].get(u)) for u in UNITS)
    minus = s_pos is not None and s_pos['delta'] is not None and s_pos['delta'] < 0 and all(
        _ci_below(s_pos['ci'].get(u)) for u in UNITS)
    return perm_ok, plus, minus


def holm(p_values, alpha):
    """Explicit two-step step-down over the ordered contrasts: if the smaller p > alpha/2 stop (nothing rejected), else
    reject it and test the other at alpha."""
    names = list(p_values)
    ordered = sorted(names, key=lambda n: (p_values[n], names.index(n)))
    steps, rejected = [], []
    for rank, name in enumerate(ordered):
        threshold = alpha / (len(ordered) - rank)
        ok = p_values[name] <= threshold
        steps.append({'contrast': name, 'p': p_values[name], 'threshold': threshold, 'rejected': ok})
        if not ok:
            break
        rejected.append(name)
    return {'steps': steps, 'rejected': sorted(rejected)}


def decide(inputs):
    """The registered decision procedure (SPEC DELTA 2 item 9, DELTA 3 items 4-6) on the decision inputs; every
    intermediate number is returned. Study level first (INVALID, ESTIMATOR_SUSPECT). Contrasts P and P_D: one-sided p =
    max over the game and pitcher bootstraps (valid replicates only), p = 1 for a NOT_EVALUABLE contrast, m = 2 fixed,
    Holm step-down at one-sided alpha. GAIN: a Holm rejection on p+, the contrast's own perm paired CI > 0 in both
    bootstraps and S-pos retention; HARM: a Holm rejection on p- with S-pos retention; a rejection without that evidence
    is NO_DIFFERENCE_ESTABLISHED:SENSITIVE; otherwise NO_DIFFERENCE_ESTABLISHED with MDE = 2.8016 SE and 3.083 SE (Holm
    first step), SE = the larger bootstrap SE. Any HARM (also next to a GAIN) -> HARM_ESCALATE, no stage 2. Stage 2: no
    study label and (GAIN contrasts, else path (ii): |E| >= 2, a significant max-t, and at the tau in E with the largest
    studentized statistic (ties -> the larger tau) the perm paired CI > 0 and S-pos retention, subset all)."""
    alpha = inputs['alpha']
    study = ('INVALID' if inputs['study'].get('invalid') else
             'ESTIMATOR_SUSPECT' if inputs['study'].get('estimator_suspect') else None)
    names = list(inputs['contrasts'])
    p = {'plus': {}, 'minus': {}}
    for name, c in inputs['contrasts'].items():
        for side in p:
            values = list(c[f'p_{side}'].values())
            p[side][name] = (max(values) if c['evaluable'] and values and all(v is not None for v in values) else 1.)
    tests = {side: holm(p[side], alpha) for side in p}
    stage = inputs['stage_label']
    _require(stage in STAGE_LABELS, f'stage label must be one of {STAGE_LABELS}')
    gain, harm_label = f'ONESTEP_GAIN_{stage}', f'ONESTEP_HARM_{stage}'
    contrasts, flags = {}, []
    for name, c in inputs['contrasts'].items():
        perm_ok, keeps_gain, keeps_harm = _gain_evidence(c.get('perm_paired_ci'), c.get('s_pos'))
        rec = {'p_plus': p['plus'][name], 'p_minus': p['minus'][name], 'delta': c['delta'], 'se': c['se'],
               'ci95': c['ci95'], 'perm_paired_ci_above_zero_both': perm_ok, 's_pos_keeps_gain': keeps_gain,
               's_pos_keeps_harm': keeps_harm}
        if not c['evaluable']:
            rec['label'] = f'NOT_EVALUABLE:{name}'
            rec['reasons'] = c['not_evaluable_reasons']
        elif name in tests['plus']['rejected']:
            rec['label'] = gain if (perm_ok and keeps_gain) else 'NO_DIFFERENCE_ESTABLISHED:SENSITIVE'
        elif name in tests['minus']['rejected']:
            rec['label'] = harm_label if keeps_harm else 'NO_DIFFERENCE_ESTABLISHED:SENSITIVE'
        else:
            rec['label'] = 'NO_DIFFERENCE_ESTABLISHED'
        ses = [v for v in c['se'].values() if v is not None]
        rec['mde'] = ({f'x{m}': m * max(ses) for m in inputs['mde_multipliers'].values()}
                      if rec['label'] == 'NO_DIFFERENCE_ESTABLISHED' and ses else None)
        contrasts[name] = rec
    harm = any(r['label'] == harm_label for r in contrasts.values())
    gains = [n for n in names if contrasts[n]['label'] == gain]
    if harm:
        flags.append('HARM_ESCALATE')
    if not inputs.get('any_main_family_tau_evaluable', True):
        flags.append('NOT_EVALUABLE_ALL')
    path = {'E': list(inputs['max_t']['E']), 'max_t_significant': bool(inputs['max_t'].get('significant')),
            'applied': False, 'tau': None, 'reason': None}
    candidates = {k: v for k, v in inputs.get('path_ii', {}).items() if v.get('studentized') is not None}
    if len(path['E']) < 2:
        path['reason'] = '|E| < 2: path (ii) not applied'
    elif not path['max_t_significant']:
        path['reason'] = 'max-t not significant'
    elif not candidates:
        path['reason'] = 'no studentized statistic in E'
    else:
        best = max(candidates, key=lambda k: (candidates[k]['studentized'], float(k)))
        perm_ok, keeps_gain, _ = _gain_evidence(candidates[best]['perm_paired_ci'], candidates[best]['s_pos'])
        path.update(tau=best, studentized=candidates[best]['studentized'], perm_paired_ci_above_zero_both=perm_ok,
                    s_pos_keeps_gain=keeps_gain, applied=perm_ok and keeps_gain,
                    reason=None if perm_ok and keeps_gain else 'perm or S-pos condition failed at the chosen tau')
    enter = study is None and not harm and (bool(gains) or path['applied'])
    if not enter:
        chosen = []
    elif gains:
        chosen = [{'contrast': n, 'family': 'temperature', 'tau': inputs.get('primary_tau'),
                   'subset': inputs['contrasts'][n]['subset']} for n in gains]
    else:
        chosen = [{'contrast': 'path_ii', 'family': 'temperature', 'tau': path['tau'], 'subset': 'all',
                   'studentized': path['studentized']}]
    reason = ('study-level label' if study else 'HARM (HARM_ESCALATE)' if harm else gain if gains else
              'path (ii)' if path['applied'] else 'no GAIN and path (ii) not applied')
    return {'study_label': study, 'study_reasons': inputs['study'].get('reasons', []),
            'study_summary': [label for label in (study, *flags) if label], 'stage_label': stage,
            'alpha_one_sided': alpha, 'holm': tests, 'contrasts': contrasts, 'max_t': {**inputs['max_t'], 'alpha': alpha},
            'path_ii': path, 'flags': flags, 'stage2': {'enter': enter, 'contrasts': chosen, 'reason': reason}}


# ---------------------------------------------------------------- output helpers

def per_tau_rows(sections):
    """Flat rows for per_tau.csv: [(section, family, variant, subset, record)] -> one row per tau."""
    rows = []
    for section, family, variant, subset, record in sections:
        for key, cell in record.items():
            gate = cell.get('gate') or {}
            row = {'section': section, 'family': family, 'variant': variant, 'subset': subset, 'tau': key,
                   'n': cell.get('n'), 'gate_pass': gate.get('pass'), 'descriptive': cell.get('descriptive')}
            for stat in STATS:
                entry = cell.get(stat)
                entry = entry if isinstance(entry, dict) else {}  # R / R_pi may be R_UNDETERMINED
                row[f'{stat}_mean'] = entry.get('mean')
                for unit in UNITS:
                    ci = (entry.get(unit) or {}).get('ci95')
                    row[f'{stat}_{unit}_lo'] = None if ci is None else ci[0]
                    row[f'{stat}_{unit}_hi'] = None if ci is None else ci[1]
            rows.append(row)
    return rows


def jsonable(value):
    """Plain JSON values: numpy -> python, NaN/inf -> None; anything else is refused."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, str):
        return value
    raise IntegrityError(f'not a JSON value: {type(value).__name__}')
