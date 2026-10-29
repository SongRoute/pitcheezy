"""EXP-OPE-1STEP-01: one-step dose-response OPE on a sealed decision ledger (side experiment, spec v2, 2026-10-29).

Design: docs/pm/research/side-onestep-ope-proposal-2026-10-29.md and SPEC DELTA v2 (independent design review
REVIEW-DESIGN-SONNET-001). Pure functions over loaded inputs; scripts/run_ope_onestep.py owns the pins, the
frame/WE load and the files. Sealed-path modules are imported, never modified.

Estimand. For a decision t (pre-decision history H_t), a target law pi and the recorded habit pi_ref (the masked
TRAIN BC ``reference``): A_t(pi) = sum_a [pi(a|H_t) - pi_ref(a|H_t)] Q_b(H_t, a) with Q_b(H_t, a) = E[G | H_t,
a_t = a, the rest of the PA as logged]: one pitch from pi, the rest of the PA as the pitcher behaves, minus the same
with the habit's pitch. G is the registered PA-end reward (policy_requests.pa_outcome, structural-end-v1, next-row
state: frozen C0 initial-defender WE, or the final result). The primary is the mean of A_t over the decision
population (WE per decision) at the sealed tau 0.1; ``pa_first_D:0`` is the mean over PAs of the deviation at the PA's
first D_0 decision (WE per PA). Not a policy value, not the full-PA L1-R deviation, not a causal effect.

Two Q arrays of every row. q_pi = result['q_planning'] builds the policy families, D and the policy-side placebos:
the recorded candidate is P3 from the PLANNING Q (policy_runtime.build_runtime -> ``candidate``:
``q, diagnostics = improvement.q_values(state)``, ``p = kl_policy(q, reference.probabilities(state), mask, tau)``;
``q`` is recorded as ``q_planning``, ``q_eval`` as ``q_reference``; policy_tau._decision recomputes the same law).
q_dr = result['q_reference'] (the registered DR q-hat, policy_estimator._step; the evaluation-seed Q, M-7) enters
only the model term and the residual. Identity check on every row with a result: kl_policy(q_pi, reference, mask,
sealed tau) equals the recorded candidate and pi(a_t)/b(a_t) equals the recorded rho_candidate (<= tolerance).

Population and row rules. Decision rows of E0 PAs (sealed pa_values ``in_population``) whose PA-end reward G is
observed; PAs without one (NO_TERMINAL, TERMINAL_VALUE_MISSING: end_state_gap, ...) are excluded WHOLE and counted by
reason (their rows only enter the worst-case bounds with g in {0, 1}). Per row, with M_t = sum_mask (pi(a) -
reference(a)) q_dr(a), rho' = pi(a_t)/b(a_t), rho_r = reference(a_t)/b(a_t), b the recorded full-vocabulary logging
law: SUPPORTED: D_t = M_t + C_t, C_t = (rho' - rho_r)(G - q_dr[a_t]); OUTSIDE_POLICY_SUPPORT (a_t off the mask,
b(a_t) > 0) and UNSUPPORTED_LOGGING_POSITIVITY (b(a_t) = 0): D_t = M_t, C_t = 0, q_dr[a_t] is never read (not
recorded off the mask), no ratio is formed. These three statuses depend on where a_t fell, so all three stay in
(dropping one selects on a_t; enumerated toy test). Every other status (MID_PA, UNKNOWN_PITCHER, NO_LOGGED_ACTION,
...) has no result at H_t: excluded with counts. With a correct logging law E[D_t | H_t] = A_t for any q_dr.

Families (same tau grid, same estimator): ``temperature`` pi = kl_policy(q_pi, reference, mask, tau) (rollout_policy
.kl_policy, imported); ``shrinkage`` q~ = qbar + c (q_pi - qbar), c = max(0, 1 - mean_mask(se^2)/var_mask(q_pi)),
se = q_planning_diff_se, sample variance, c = 0 (mask < 2 or var 0) -> pi = reference exactly; sensitivity
``cross_replicate`` (policy from q_dr, DR with q_pi); placebo ``neg`` (-q_pi). tau = inf is the reference exactly.

D_k (tau-free): argmax_mask q_pi != argmax_mask reference and q_pi[best] - q_pi[ref_best] > k for k in {0, 0.005 WE,
'1se'}; '1se' = q_planning_diff_se[ref_best], the SE of Q(ref_best) - Q(best) (paired_difference_se is 0 at the Q
argmax itself, so the review's ``[best]`` would make D_1se = D_0). A tie at either maximum -> outside every D.

Prediag (no G, written before any G-dependent number): per family x tau decision ESS (Kish) of rho', PA weights =
product of rho' over the PA's eligible decisions with PA ESS, PA ESS ratio candidate/reference and game ESS
(policy_tau.tau_table arithmetic; D106/D107 read the ESS ratio at the PA level), noise ratio max_mask(c se)/tau
q50/q90, mean KL, mean M (per row class), max rho' share. Evaluable: PA ESS ratio >= 0.5, noise q90 <= 0.5, game ESS
>= 30, PA ESS >= 100 (registered). Trend (OLS slope on log(1/tau)) and the one-sided joint max-t use evaluable finite
taus only; other rows are reported as descriptive.

Placebos (policy side; q_dr fixed): ``perm`` (K draws, SeedSequence([placebo_perm_seed, k, int(row sha256[:8], 16)]),
placebo temperature bisected per draw so that its mean KL matches the temperature family's at each tau, via the
row-wise kl_policy mirror ``tilt_rows`` that is checked against kl_policy on every decision), ``neg``, tau = inf
(identity), the outcome-shuffle null (G permuted across PAs within a first-row pitcher; DR and IPS), and
``estimator_stress`` (q_dr permuted with the same permutation, reported apart). Real - placebo differences are paired
on the shared replicates; the band is the 2.5-97.5% range of the K draw means. The other-pitcher donor placebo is
NOT_IMPLEMENTED. Sensitivities: IPS-only, SN-DR, switch at rho' caps, S-pos (whole PAs with a positivity node out),
cross_replicate. Bounds: the excluded no-reward PAs added back with g in {0, 1} in their correction terms.

Inference: cluster bootstraps by game and by the PA's first-row pitcher; decisions are never resampled alone (a PA
lies in one game and has one first-row pitcher). One multiplicity matrix per unit (A2 ope_bootstrap.Resampler
conventions: clusters sorted by str, default_rng(seed).integers(C, size=(draws, C))) serves every statistic, so
differences are paired. Denominators are recounted per replicate; a replicate is invalid when the subset has no
decision, fewer than the registered distinct games (or PAs) / pitchers; a statistic whose invalid share exceeds the
registered maximum is INVALID (A2 5.3). Percentile 95% CI, SE = SD (ddof 1) of the valid replicates.
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

CONTRACT = 'EXP-OPE-1STEP-01/onestep-v2'
INF = math.inf
Q_PI, Q_DR, SE_KEY = 'q_planning', 'q_reference', 'q_planning_diff_se'
RESULT_STATUSES = (SUPPORTED, OUTSIDE_POLICY_SUPPORT, LOGGING_POSITIVITY)
HALTING_KINDS = ('conflict', 'malformed', 'aborted')
MAIN_FAMILIES = ('temperature', 'shrinkage')
FAMILY_Q = {'temperature': ('q_pi', 'q_dr'), 'shrinkage': ('shrunk', 'q_dr'), 'cross_replicate': ('q_dr', 'q_pi'),
            'neg': ('-q_pi', 'q_dr')}  # (policy Q, DR q-hat)
UNITS = ('game', 'pitcher')
CI_LEVELS = (0.025, 0.975)
STATS = ('delta', 'M', 'C', 'R')
ESTIMAND = {
    'primary': 'temperature family, tau 0.1 (the sealed policy), subset all, DR: mean over the decision population of '
               'A_t = sum_a [pi(a) - reference(a)] Q_b(H_t, a); WE per decision',
    'pa_first_D:0': 'mean over PAs of the deviation at the first D_0 decision (0 without one); WE per PA',
    'population': 'E0 PAs with an observed PA-end reward (whole PAs otherwise excluded); rows with a recorded result '
                  '(SUPPORTED, OUTSIDE_POLICY_SUPPORT, UNSUPPORTED_LOGGING_POSITIVITY)',
    'row_rules': 'SUPPORTED: D = M + (rho\' - rho_r)(G - q_dr[a]); off-mask rows: D = M; M = sum_mask (pi - '
                 'reference) q_dr', 'q_pi': Q_PI, 'q_dr': Q_DR, 'units': 'frozen C0 initial-defender WE',
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
    newline-terminated rows) without constructing a Ledger, which would create a missing file. ``sha256`` is the hash
    of the bytes streamed, set when the stream is exhausted (compared with the pin by the caller)."""
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


def _on_mask(values, mask, name, finite=True):
    _require(isinstance(values, list) and len(values) == len(mask), f'recorded {name} required (candidate runtime)')
    _require(all(v is None for v, m in zip(values, mask) if not m), f'{name} must not be recorded off the mask')
    out = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    _require(not finite or np.isfinite(out[mask]).all(), f'{name} must be finite exactly on the mask')
    return out


def decision_arrays(row, n):
    """Validated arrays of a row with a recorded result: policy_estimator._step (positivity rows included: logging[a] =
    0, both laws 0 at a, recorded rho exactly 0) plus q_pi (exactly on the mask) and the paired-difference SE."""
    status = row.get('status')
    _require(status in RESULT_STATUSES, f'{status}: no recorded result is expected for this status')
    policies, q_dr, mask, a, rho = est._step(row, n, positivity=True)
    result = row['result']
    q_pi = _on_mask(result.get(Q_PI), mask, Q_PI)
    _require(all((v is None) != bool(m) for v, m in zip(result[Q_PI], mask)), f'{Q_PI} must be recorded exactly on the mask')
    se = _on_mask(result.get(SE_KEY), mask, SE_KEY)
    _require((se[mask] >= 0).all(), 'paired-difference SE must be nonnegative (search samples >= 2)')
    logging = np.asarray(result['logging'], dtype=np.float64)
    reference = policies['reference']
    on = bool(mask[a])
    rho_ref = float(reference[a] / logging[a]) if on else 0.
    _require(rho_ref == rho['reference'], 'reference ratio differs from the validated recorded ratio')
    return {'status': status, 'mask': mask, 'logging': logging, 'reference': reference, 'candidate': policies['candidate'],
            'q_dr': q_dr, 'q_pi': q_pi, 'se': se, 'a': int(a), 'on_mask': on, 'b_a': float(logging[a]),
            'rho_ref': rho_ref, 'rho_candidate': float(rho['candidate']), 'q_source': result.get('q_source')}


# ---------------------------------------------------------------- policies, terms, D, placebo inputs

def tilt(q, reference, mask, tau):
    """pi_tau = rollout_policy.kl_policy(q, reference, mask, tau) with q = -inf off the mask (policy_tau._decision);
    tau = inf is the reference exactly."""
    if tau == INF:
        return np.array(reference, dtype=np.float64, copy=True)
    return kl_policy(np.where(mask, q, -np.inf), reference, mask, tau)


def tilt_rows(q, reference, mask, tau):
    """Row-wise mirror of kl_policy for already validated rows ([n, V]; q, reference, mask): the same arithmetic
    (rollout_policy.probabilities renormalisation, log(bc) + (q - max q)/tau, exp(lw - max lw), normalise). Used only
    by the vectorised placebos; Scan/aggregate verify it against kl_policy on every decision of the temperature family."""
    if tau == INF:
        return np.array(reference, dtype=np.float64, copy=True)
    bc = reference / reference.sum(axis=1, keepdims=True)
    qmax = np.where(mask, q, -np.inf).max(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        lw = np.where(mask, np.log(bc) + (q - qmax) / tau, -np.inf)
    w = np.where(mask, np.exp(lw - lw.max(axis=1, keepdims=True)), 0.)
    return w / w.sum(axis=1, keepdims=True)


def policy_tau(row, tau, n=None):
    """pi_tau of one ledger row (q_pi, recorded reference and mask)."""
    d = decision_arrays(row, len(row['result']['logging']) if n is None else n)
    return tilt(d['q_pi'], d['reference'], d['mask'], tau)


def shrink_factor(q, se, mask):
    """c = max(0, 1 - mean_mask(se^2) / var_mask(q)) (sample variance); c = 0 for a mask below 2 actions or var 0."""
    qm, sm = np.asarray(q)[mask], np.asarray(se)[mask]
    if len(qm) < 2:
        return 0.
    var = float(np.var(qm, ddof=1))
    if not var > 0:
        return 0.
    return max(0., 1. - float(np.mean(sm ** 2)) / var)


def shrunk(q, se, mask):
    """(q~, c): q~ = qbar + (q - qbar) c on the mask (NaN off it)."""
    c = shrink_factor(q, se, mask)
    qm = np.asarray(q)[mask]
    out = np.full(len(mask), np.nan)
    out[mask] = qm.mean() + (qm - qm.mean()) * c
    return out, c


def family_q(name, d):
    """(policy Q, DR q-hat, shrinkage c) of one decision for a family (FAMILY_Q)."""
    _require(name in FAMILY_Q, f'unknown family {name}')
    if name == 'temperature':
        return d['q_pi'], d['q_dr'], 1.
    if name == 'shrinkage':
        q, c = shrunk(d['q_pi'], d['se'], d['mask'])
        return q, d['q_dr'], c
    if name == 'cross_replicate':
        return d['q_dr'], d['q_pi'], 1.
    return -d['q_pi'], d['q_dr'], 1.


def family_policy(name, d, tau, fq=None):
    """(pi, DR q-hat, c) of a family at tau; ``fq`` = a precomputed family_q(name, d)."""
    q, q_hat, c = family_q(name, d) if fq is None else fq
    pi = np.array(d['reference'], dtype=np.float64, copy=True) if (tau == INF or c == 0.) else tilt(q, d['reference'],
                                                                                                     d['mask'], tau)
    return pi, q_hat, c


def onestep_terms(pi, d, reward, q_hat=None):
    """(M, C, IPS, rho') of one decision for target law ``pi`` (q_hat defaults to q_dr). Off the mask C = IPS = rho' = 0
    and q_hat[a_t] is never read."""
    q_hat = d['q_dr'] if q_hat is None else q_hat
    mask = d['mask']
    model = float((pi[mask] - d['reference'][mask]) @ q_hat[mask])
    if not d['on_mask']:
        return model, 0., 0., 0.
    rho_pi = float(pi[d['a']] / d['b_a'])
    diff = rho_pi - d['rho_ref']
    return model, diff * (float(reward) - float(q_hat[d['a']])), diff * float(reward), rho_pi


def kl(pi, reference, mask):
    p, r = pi[mask], reference[mask]
    pos = p > 0
    return float(np.sum(p[pos] * np.log(p[pos] / r[pos])))


def disagreement(q, reference, mask, se, margins):
    """(flags per margin, tie_q, tie_ref): argmax_mask q != argmax_mask reference and q[best] - q[ref_best] > k; k =
    '1se' is se[ref_best]. A tie at either maximum keeps the decision outside every D."""
    qm, rm, sm = np.asarray(q)[mask], np.asarray(reference)[mask], np.asarray(se)[mask]
    tie_q, tie_r = int((qm == qm.max()).sum()) > 1, int((rm == rm.max()).sum()) > 1
    if tie_q or tie_r:
        return [False] * len(margins), tie_q, tie_r
    best, ref_best = int(np.argmax(qm)), int(np.argmax(rm))
    if best == ref_best:
        return [False] * len(margins), False, False
    gap = float(qm[best] - qm[ref_best])
    return [gap > (float(sm[ref_best]) if m == '1se' else float(m)) for m in margins], False, False


def perm_rng(seed, k, sha8):
    """The registered perm RNG of one row and draw: SeedSequence([placebo_perm_seed, k, int(row sha256[:8], 16)])."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(k), int(sha8)]))


def permuted(q, mask, seed, k, sha8):
    """q with its mask values permuted by perm_rng (the multiset over the mask is kept)."""
    idx = np.flatnonzero(mask)
    out = np.array(q, dtype=np.float64, copy=True)
    out[idx] = np.asarray(q)[idx[perm_rng(seed, k, sha8).permutation(len(idx))]]
    return out


# ---------------------------------------------------------------- sealed pa_values cross-check, frame facts

CATEGORICAL = ('status', 'node', 'kind', 'reason', 'game', 'in_population', 'validity_violation', 'resolution', 'l2')
NUMERIC = ('candidate', 'reference', 'delta', 'weight_candidate', 'weight_reference', 'terminal_residual')
PAIRS = ('candidate_bounds', 'reference_bounds', 'delta_bounds')
PA_VALUE_COLUMNS = ('pa_id', 'end_kind', 'flags', *CATEGORICAL, *NUMERIC, *PAIRS)


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


def sealed_rows(records):
    """{pa_id: row} of the sealed pa_values records (file order kept), the columns the cross-check reads."""
    out = {}
    for record in records:
        row = {k: plain(record.get(k)) for k in PA_VALUE_COLUMNS}
        _require(isinstance(row['pa_id'], str) and row['pa_id'] not in out, 'pa_values: PA ids must be unique strings')
        _require(row['in_population'] == (row['status'] in (est.COMPLETE, est.CENSORED)),
                 f'pa_values {row["pa_id"]}: in_population differs from its status')
        out[row['pa_id']] = row
    return out


def estimator_info(pa_row, facts):
    """policy_estimator.pa_value facts of one PA: E0, start reason and structural problem from the sealed row (what the
    registered runner recorded), the PA end (reward, kind, reason, end, end_kind, flags) from the frame. A structural
    problem hidden behind an earlier refusal or outside E0 does not change the classification and is left out."""
    status, reason, node = pa_row['status'], pa_row['reason'], pa_row['node']
    problem = index = None
    if status == est.NO_DECISION:
        problem, index = 'no_decision', 0
    elif status == est.UNSUBMITTABLE:
        problem, index = reason, 0
    elif (status == est.CENSORED and pa_row['kind'] == est.REFUSED and isinstance(reason, str)
          and reason.startswith('structural:')):
        _require(node is not None, f'pa_values {pa_row["pa_id"]}: structural censoring without a node')
        problem, index = reason.split(':', 1)[1], int(node)
    return {'game': facts['game'], 'in_population': pa_row['in_population'],
            'start_reason': reason if status == est.EXCLUDED_PRE_START else None, 'problem': problem,
            'problem_index': index, **{k: facts[k] for k in ('reward', 'kind', 'reason', 'end', 'end_kind', 'flags')}}


def _same_number(a, b, tol):
    if a is None or b is None:
        return a is None and b is None, 0.
    diff = abs(float(a) - float(b))
    return diff <= tol, diff


def _moved(a, b):
    if a is None or b is None:
        return (a is None) != (b is None)
    return abs(float(a) - float(b)) > 1e-9


class FactsCheck:
    """Per PA, policy_estimator.pa_value re-run on the ledger rows and the frame facts must reproduce the sealed
    pa_values row (categorical fields exactly, values within ``tol``); a PA whose recorded values move when its reward
    is perturbed has its frame reward verified by the sealed record."""
    def __init__(self, censoring, tol):
        _require(censoring in est.CENSORING and isinstance(tol, float) and 0 <= tol < 1e-6, 'cross-check settings')
        self.censoring, self.tol = censoring, tol
        self.max_abs = {k: 0. for k in (*NUMERIC, *PAIRS)}
        self.pas, self.reward_verified, self.reward_unverified, self.missing_reward = 0, set(), 0, Counter()

    @staticmethod
    def _values(row):
        return [row.get(k) for k in NUMERIC] + [v for k in PAIRS for v in (row.get(k) or [None, None])]

    def check(self, pa_id, decisions, info, pa_row, facts):
        got = est.pa_value(decisions, info, self.censoring)
        for key in CATEGORICAL:
            expected, value = pa_row[key], plain(got.get(key))
            same = _same_number(expected, value, 0.)[0] if key == 'node' else expected == value
            _require(same, f'pa_values cross-check {pa_id}: {key} {expected!r} != recomputed {value!r}')
        for key in NUMERIC:
            same, diff = _same_number(pa_row[key], plain(got.get(key)), self.tol)
            _require(same, f'pa_values cross-check {pa_id}: {key} {pa_row[key]!r} != recomputed {got.get(key)!r}')
            self.max_abs[key] = max(self.max_abs[key], diff)
        for key in PAIRS:
            expected, value = pa_row[key], plain(got.get(key))
            _require((expected is None) == (value is None), f'pa_values cross-check {pa_id}: {key} presence differs')
            if expected is not None:
                _require(len(expected) == len(value) == 2, f'pa_values cross-check {pa_id}: {key} must be a pair')
                for e, v in zip(expected, value):
                    same, diff = _same_number(e, v, self.tol)
                    _require(same, f'pa_values cross-check {pa_id}: {key} {expected!r} != recomputed {value!r}')
                    self.max_abs[key] = max(self.max_abs[key], diff)
        _require(pa_row['end_kind'] == facts['end_kind'], f'pa_values cross-check {pa_id}: end_kind differs')
        _require(list(pa_row['flags'] or []) == list(facts['flags'] or []), f'pa_values cross-check {pa_id}: flags differ')
        self.pas += 1
        reward = facts['reward']
        if reward is None:
            self.missing_reward[f'{facts["kind"]}:{facts["reason"]}'] += 1
            return
        moved = est.pa_value(decisions, {**info, 'reward': reward + .25 if reward <= .5 else reward - .25}, self.censoring)
        if any(_moved(a, b) for a, b in zip(self._values(got), self._values(moved))):
            self.reward_verified.add(pa_id)
        else:
            self.reward_unverified += 1

    def record(self):
        return {'rule': 'policy_estimator.pa_value re-run on the ledger rows + frame facts == sealed pa_values row',
                'censoring': self.censoring, 'tolerance': self.tol, 'pas': self.pas, 'max_abs_diff': dict(self.max_abs),
                'reward_verified_pas': len(self.reward_verified),
                'reward_not_reaching_a_recorded_value_pas': self.reward_unverified,
                'missing_reward_pas': dict(self.missing_reward)}


def frame_facts(frame, blocks, games_table, defense):
    """{pa_id: facts} in block order: the PA end exactly as the registered runner computes it
    (run_policy_validation.outcome_function -> policy_requests.pa_outcome(frame, positions, components.defense_we,
    games, rule='structural-end-v1', score_source='next_row'); the store's frame is the loaded frame object) and
    pa_manifest's start pitcher and date (pre-decision)."""
    from . import policy_requests as preq
    out = {}
    for pa_id, positions in blocks:
        _require(pa_id not in out, f'PA {pa_id} appears in more than one block')
        outcome = preq.pa_outcome(frame, positions, defense, games_table, rule='structural-end-v1',
                                  score_source='next_row')
        manifest = preq.pa_manifest(frame, positions)
        out[pa_id] = {'game': int(frame.game_pk.iloc[positions[0]]), 'start_pitcher': manifest['start_pitcher'],
                      'date': manifest['date'], 'reward': outcome['reward'], 'kind': outcome['kind'],
                      'reason': outcome['reason'], 'end': outcome['end'], 'end_kind': outcome['end_kind'],
                      'flags': list(outcome['flags'])}
    return out


# ---------------------------------------------------------------- the streamed pass

class Scan:
    """One streamed pass in PA order (policy_estimator.by_pa): row validation, the identity check, D, the population
    accounting, the sealed facts cross-check and, for every kept row (eligible status in an E0 PA, observed reward or
    not), the per-family model term, rho', KL and q-hat at a_t per tau (kl_policy), plus the padded vectors the
    vectorised placebos need. G is stored, never aggregated here."""
    def __init__(self, *, taus, sealed_tau, identity_tolerance, statuses, families, margins, expected_q_source=None,
                 runtime_sha256=None):
        self.taus = check_grid(taus, sealed_tau)
        _require(isinstance(identity_tolerance, float) and 0 <= identity_tolerance < 1e-6, 'identity tolerance')
        _require(statuses and set(statuses) <= set(RESULT_STATUSES) and len(set(statuses)) == len(statuses),
                 f'eligible statuses must be distinct values of {RESULT_STATUSES}')
        _require(families and families[0] == 'temperature' and set(families) <= set(FAMILY_Q)
                 and len(set(families)) == len(families), 'families: temperature first, known names')
        _require(margins and margins[0] == 0 and all(m == '1se' or (isinstance(m, (int, float)) and m >= 0)
                                                     for m in margins), 'D margins: 0 first, then numbers or "1se"')
        self.sealed_tau, self.tol, self.statuses = sealed_tau, identity_tolerance, tuple(statuses)
        self.families, self.margins, self.expected_q_source = list(families), list(margins), expected_q_source
        self.runtime_sha256 = runtime_sha256
        self.n_vocab = None
        self.by_status, self.excluded, self.kept_status = Counter(), Counter(), Counter()
        self.q_source, self.mask_sizes, self.pa = Counter(), Counter(), Counter()
        self.pa_no_reward = Counter()
        self.identity = {'tau': sealed_tau, 'rows_checked': 0, 'max_abs_diff_candidate': 0., 'max_abs_diff_rho': 0.,
                         'tolerance': identity_tolerance}
        self.d_rows = {'rows': 0, 'tie_q': 0, 'tie_ref': 0, **{f'D:{margin_key(m)}': 0 for m in self.margins}}
        self.cols = {k: [] for k in ('pa_index', 'pa_id', 'game', 'pitcher', 'decision_index', 'balls', 'strikes',
                                     'status', 'on_mask', 'observed', 'G', 'rho_ref', 'b_a', 'a', 'sha8', 'D', 'tie',
                                     'c', 'noise', 'model', 'rho', 'kl', 'qa', 'mask', 'ref', 'q_pi', 'q_dr', 'pa_pos',
                                     'pa_out', 'first_eligible', 'first_D0')}

    def run(self, decisions, pas, *, sealed=None, facts_check=None, checkpoint=None):
        """``pas``: {pa_id: facts} in the ledger's PA order (game, in_population, start_pitcher, reward, kind, reason,
        end_kind, flags, pa_status); ``sealed`` {pa_id: pa_values row} with ``facts_check``."""
        for k, (pa_id, info, rows) in enumerate(est.by_pa(decisions, pas)):
            if checkpoint is not None and k % 200 == 0:
                checkpoint()
            self._pa(k, pa_id, info, sorted(rows, key=lambda d: d['decision_index']), sealed, facts_check)
        return self

    def _pa(self, k, pa_id, info, rows, sealed, facts_check):
        self.pa['pas'] += 1
        if facts_check is not None:
            facts_check.check(pa_id, rows, estimator_info(sealed[pa_id], info), sealed[pa_id], info)
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
            self._identity(row, d)
            flags, tie_q, tie_r = disagreement(d['q_pi'], d['reference'], d['mask'], d['se'], self.margins)
            self.d_rows['rows'] += 1
            self.d_rows['tie_q'] += tie_q
            self.d_rows['tie_ref'] += tie_r
            for m, flag in zip(self.margins, flags):
                self.d_rows[f'D:{margin_key(m)}'] += flag
            if status not in self.statuses:
                self.excluded[f'status:{status}'] += 1
                continue
            if not e0:
                self.excluded['pa_not_in_E0'] += 1
                continue
            if reward is None:
                self.excluded[f'no_reward_pa:{info["kind"]}:{info["reason"]}'] += 1
            self._keep(k, pa_id, info, row, d, flags, tie_q or tie_r, reward)
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

    def _identity(self, row, d):
        if d['q_source'] is not None:
            self.q_source[str(d['q_source'])] += 1
        _require(self.expected_q_source is None or d['q_source'] == self.expected_q_source,
                 f'row q_source {d["q_source"]!r} differs from the expected DR q-hat source {self.expected_q_source!r}')
        pi = tilt(d['q_pi'], d['reference'], d['mask'], self.sealed_tau)
        diff = float(np.max(np.abs(pi - d['candidate'])))
        rho = float(pi[d['a']] / d['b_a']) if d['on_mask'] else 0.
        rho_diff = abs(rho - d['rho_candidate'])
        self.identity['rows_checked'] += 1
        self.identity['max_abs_diff_candidate'] = max(self.identity['max_abs_diff_candidate'], diff)
        self.identity['max_abs_diff_rho'] = max(self.identity['max_abs_diff_rho'], rho_diff)
        _require(diff <= self.tol and rho_diff <= self.tol,
                 f'identity check failed at {row["request_id"]}: |pi - candidate| = {diff!r}, |rho - rho_candidate| = '
                 f'{rho_diff!r}')

    def _keep(self, k, pa_id, info, row, d, flags, tie, reward):
        c = self.cols
        self.kept_status[d['status']] += 1
        self.mask_sizes[int(d['mask'].sum())] += 1
        for key, value in (('pa_index', k), ('pa_id', pa_id), ('game', int(info['game'])),
                           ('pitcher', str(info['start_pitcher'])), ('decision_index', int(row['decision_index'])),
                           ('balls', int(row['balls'])), ('strikes', int(row['strikes'])), ('status', d['status']),
                           ('on_mask', d['on_mask']), ('observed', reward is not None),
                           ('G', np.nan if reward is None else float(reward)), ('rho_ref', d['rho_ref']),
                           ('b_a', d['b_a']), ('a', d['a']), ('sha8', int(row['sha256'][:8], 16)), ('D', list(flags)),
                           ('tie', tie), ('mask', d['mask']), ('ref', d['reference']), ('q_pi', d['q_pi']),
                           ('q_dr', d['q_dr'])):
            c[key].append(value)
        shape = (len(self.families), len(self.taus))
        model, rho, kls = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        qa, noise = np.zeros(len(self.families)), np.zeros(len(self.families))
        se_max = float(d['se'][d['mask']].max())
        shrink = 1.
        for f, name in enumerate(self.families):
            fq = family_q(name, d)
            for j, tau in enumerate(self.taus):
                pi, q_hat, cf = family_policy(name, d, tau, fq)
                model[f, j], _, _, rho[f, j] = onestep_terms(pi, d, 0., q_hat)
                kls[f, j] = kl(pi, d['reference'], d['mask'])
            qa[f] = float(q_hat[d['a']]) if d['on_mask'] else 0.
            noise[f] = cf * se_max
            shrink = cf if name == 'shrinkage' else shrink
        c['c'].append(shrink if 'shrinkage' in self.families else np.nan)
        for key, value in (('model', model), ('rho', rho), ('kl', kls), ('qa', qa), ('noise', noise)):
            c[key].append(value)

    def table(self):
        """Per kept decision, compact arrays: ids, cluster keys, count, status, flags, G (NaN when unobserved), rho_r,
        and [n, families, taus] model term / rho' / KL; [n, families] q-hat at a_t and noise numerators; [n, V]
        vectors for the placebos."""
        c, n = self.cols, len(self.cols['pa_index'])
        v = self.n_vocab or 0
        f, t = len(self.families), len(self.taus)
        out = {'n': n, 'families': list(self.families), 'taus': list(self.taus), 'margins': list(self.margins),
               'pa_id': np.array(c['pa_id'], dtype=object), 'pitcher': np.array(c['pitcher'], dtype=object),
               'status': np.array(c['status'], dtype=object),
               'D': np.array(c['D'], dtype=bool).reshape(n, len(self.margins))}
        for key, dtype in (('pa_index', np.int64), ('game', np.int64), ('decision_index', np.int64), ('balls', np.int64),
                           ('strikes', np.int64), ('on_mask', bool), ('observed', bool), ('G', np.float64),
                           ('rho_ref', np.float64), ('b_a', np.float64), ('a', np.int64), ('sha8', np.int64),
                           ('tie', bool), ('c', np.float64), ('pa_pos', bool), ('pa_out', bool),
                           ('first_eligible', bool), ('first_D0', bool)):
            out[key] = np.array(c[key], dtype=dtype)
        for key, shape in (('model', (f, t)), ('rho', (f, t)), ('kl', (f, t)), ('qa', (f,)), ('noise', (f,)),
                           ('mask', (v,)), ('ref', (v,)), ('q_pi', (v,)), ('q_dr', (v,))):
            out[key] = np.stack(c[key]) if n else np.zeros((0, *shape))
        return out

    def records(self, table=None):
        table = self.table() if table is None else table
        obs = table['observed']
        n = int(obs.sum())
        status = table['status'][obs]
        pitchers = Counter(table['pitcher'][obs].tolist())
        d0 = table['D'][obs, 0] if n else np.zeros(0, bool)
        by_pitcher = {}
        for p, flag in zip(table['pitcher'][obs].tolist(), d0.tolist()):
            entry = by_pitcher.setdefault(p, [0, 0])
            entry[0] += 1
            entry[1] += int(flag)
        shares = np.array([b / a for a, b in by_pitcher.values()]) if by_pitcher else np.zeros(0)
        return {
            'decision_rows': sum(self.by_status.values()), 'by_status': dict(self.by_status),
            'q_source': dict(self.q_source), 'vocabulary_size': self.n_vocab,
            'identity_check': {**self.identity, 'q_pi': Q_PI, 'pass': True,
                               'rule': 'kl_policy(q_planning, reference, mask, sealed tau) == recorded candidate and '
                                       'pi(a)/b(a) == recorded rho_candidate'},
            'counts': {'eligible_decisions': n, 'kept_decisions_by_status': dict(self.kept_status),
                       'eligible_by_status': dict(Counter(status.tolist())),
                       'excluded_decisions': dict(sorted(self.excluded.items())), 'eligible_statuses': list(self.statuses),
                       'pas': dict(self.pa), 'e0_pas_without_reward': dict(sorted(self.pa_no_reward.items())),
                       'unobserved_reward_decisions': int((~obs).sum()),
                       'mask_sizes': {str(k): v for k, v in sorted(self.mask_sizes.items())},
                       'two_strike_decisions': int((table['strikes'][obs] == 2).sum()),
                       'p_out': {'decision': float(np.mean(status == OUTSIDE_POLICY_SUPPORT)) if n else None,
                                 'pa': (self.pa['eligible_pas_with_outside'] / self.pa['eligible_pas']
                                        if self.pa['eligible_pas'] else None)},
                       'p_pos': {'decision': float(np.mean(status == LOGGING_POSITIVITY)) if n else None,
                                 'pa': (self.pa['eligible_pas_with_positivity'] / self.pa['eligible_pas']
                                        if self.pa['eligible_pas'] else None)}},
            'disagreement': {'definition': 'argmax_mask q_planning != argmax_mask reference and q[best] - q[ref_best] > k; '
                                           "k = '1se' is q_planning_diff_se[ref_best]; ties outside every D",
                             'margins': [margin_key(m) for m in self.margins], 'result_rows': dict(self.d_rows),
                             'eligible': {f'D:{margin_key(m)}': int(table['D'][obs, i].sum()) for i, m in
                                          enumerate(self.margins)} if n else {},
                             'p_D': {f'D:{margin_key(m)}': float(table['D'][obs, i].mean()) for i, m in
                                     enumerate(self.margins)} if n else {},
                             'ties_eligible': int(table['tie'][obs].sum()),
                             'by_pitcher': {'pitchers': len(pitchers),
                                            'p_D0_quantiles': ({str(q): float(np.quantile(shares, q)) for q in
                                                                (.1, .25, .5, .75, .9)} if len(shares) else None),
                                            'counts': {p: {'n': a, 'n_D0': b} for p, (a, b) in sorted(by_pitcher.items())}}}}


# ---------------------------------------------------------------- prediag (no G)

def kish(weights):
    w = np.asarray(weights, dtype=np.float64)
    s2 = float((w ** 2).sum())
    return None if s2 == 0 else float(w.sum() ** 2 / s2)


def _pa_products(values, pa_index):
    """Per-PA products of per-decision values (rows of a PA are contiguous, in ledger order)."""
    starts = np.flatnonzero(np.r_[True, pa_index[1:] != pa_index[:-1]])
    return np.multiply.reduceat(values, starts), starts


def prediag(table, thresholds):
    """Per family x tau, on the observed eligible decisions and without any G: decision ESS of rho' (Kish) and its
    ratio to the reference, PA-level weights = product of rho' over the PA's eligible decisions with PA ESS, PA ESS
    ratio candidate/reference and game ESS (policy_tau.tau_table arithmetic, policy_estimator.effective_sample_size),
    noise-ratio quantiles of max_mask(c se)/tau, mean KL, mean model term (all, per row class), max rho' share; p_D,
    p_out, p_pos. Evaluable: PA ESS ratio >= ess_ratio_min AND noise q90 <= noise_q90_max AND game ESS >= game_ess_min
    AND PA ESS >= pa_ess_min."""
    obs = table['observed']
    n = int(obs.sum())
    _require(n > 0, 'no eligible decision with an observed reward: the one-step estimate is undefined')
    pa = table['pa_index'][obs]
    rho_ref = table['rho_ref'][obs]
    status = table['status'][obs]
    ref_w, starts = _pa_products(rho_ref, pa)
    pa_games = table['game'][obs][starts]
    ref_pa_ess = est.effective_sample_size(ref_w)
    common = {'decisions': n, 'pas': int(len(starts)), 'games': int(len(set(pa_games.tolist()))),
              'pitchers': int(len(set(table['pitcher'][obs].tolist()))),
              'p_D': {f'D:{margin_key(m)}': float(table['D'][obs, i].mean()) for i, m in enumerate(table['margins'])},
              'p_out': {'decision': float(np.mean(status == OUTSIDE_POLICY_SUPPORT)),
                        'pa': float(np.mean(table['pa_out'][obs][starts]))},
              'p_pos': {'decision': float(np.mean(status == LOGGING_POSITIVITY)),
                        'pa': float(np.mean(table['pa_pos'][obs][starts]))},
              'reference': {'decision_ess': kish(rho_ref), 'pa_ess': ref_pa_ess,
                            'game_ess': est.effective_sample_size(ref_w, pa_games)},
              'thresholds': dict(thresholds)}
    out = {}
    for f, fam in enumerate(table['families']):
        rows = {}
        for j, tau in enumerate(table['taus']):
            rho = table['rho'][obs, f, j]
            w, _ = _pa_products(rho, pa)
            pa_ess, game_ess = est.effective_sample_size(w), est.effective_sample_size(w, pa_games)
            ratio = None if (pa_ess is None or not ref_pa_ess) else pa_ess / ref_pa_ess
            noise = np.zeros(n) if tau == INF else table['noise'][obs, f] / tau
            q50, q90 = (float(np.quantile(noise, q)) for q in (.5, .9))
            model = table['model'][obs, f, j]
            dec_ess = kish(rho)
            entry = {'decision_ess': dec_ess, 'decision_ess_ratio': (None if dec_ess is None or not common['reference'][
                         'decision_ess'] else dec_ess / common['reference']['decision_ess']),
                     'pa_ess': pa_ess, 'pa_ess_ratio_candidate_reference': ratio, 'game_ess': game_ess,
                     'noise_ratio_q50': q50, 'noise_ratio_q90': q90, 'mean_kl': float(table['kl'][obs, f, j].mean()),
                     'mean_M': float(model.mean()),
                     'mean_M_by_class': {s: (float(model[status == s].mean()) if (status == s).any() else None)
                                         for s in RESULT_STATUSES},
                     'max_rho_share': (None if not rho.sum() > 0 else float(rho.max() / rho.sum())),
                     'max_rho': float(rho.max())}
            entry['evaluable'] = bool(ratio is not None and game_ess is not None and pa_ess is not None
                                      and ratio >= thresholds['ess_ratio_min'] and q90 <= thresholds['noise_q90_max']
                                      and game_ess >= thresholds['game_ess_min'] and pa_ess >= thresholds['pa_ess_min'])
            rows[tau_key(tau)] = entry
        out[fam] = rows
    shrink = table['c'][obs]
    return {'common': common, 'families': out, 'noise_numerator': 'max_mask(q_planning_diff_se) x c (c = 1 except '
            'shrinkage; cross_replicate uses q_planning_diff_se as a proxy)',
            'shrinkage_c': ({str(q): float(np.quantile(shrink, q)) for q in (0., .1, .5, .9, 1.)}
                            if np.isfinite(shrink).all() else None),
            'shrinkage_c_zero_share': float(np.mean(shrink == 0)) if np.isfinite(shrink).all() else None}


def evaluable(prediag_record, family):
    return {key: row['evaluable'] for key, row in prediag_record['families'][family].items()}


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
        self.drawn = (self.m > 0).astype(np.float64)

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


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


class Cell:
    """One statistic family over the tau grid: point values [T] and per-unit replicates / validity [draws, T]."""
    def __init__(self, point, reps, valid, info):
        self.point, self.reps, self.valid, self.info = point, reps, valid, info

    def record(self, taus, evaluable_by_tau, invalid_share_max, draws):
        out = {}
        for j, tau in enumerate(taus):
            key = tau_key(tau)
            ev = evaluable_by_tau.get(key) if evaluable_by_tau is not None else None
            cell = {**self.info, 'evaluable': ev, 'descriptive': ev is False}
            for stat, values in self.point.items():
                entry = {'mean': _finite(values[j])}
                for unit in self.reps:
                    if self.info.get('below_minimum', {}).get(unit):
                        entry[unit] = {'valid': False, 'ci95': None, 'se': None, 'invalid_replicates': draws,
                                       'reason': 'INVALID: subset below the registered minimum games/PAs/pitchers'}
                    else:
                        entry[unit] = interval(self.reps[unit][stat][:, j], self.valid[unit][stat][:, j], draws,
                                               invalid_share_max)
                cell[stat] = entry
            out[key] = cell
        return out


class Engine:
    """Shared-resampling statistics over the observed eligible decisions (one multiplicity matrix per unit)."""
    def __init__(self, table, *, draws, seeds, minimum, invalid_share_max):
        _require(set(seeds) >= set(UNITS), 'one registered bootstrap seed per unit (game, pitcher)')
        _require(set(minimum) == {'games', 'pitchers', 'pas'} and all(type(v) is int and v >= 1 for v in minimum.values()),
                 'bootstrap minimum: games, pitchers, pas')
        obs = table['observed']
        self.n_all = len(obs)
        self.t = {k: (v[obs] if isinstance(v, np.ndarray) and v.shape[:1] == (self.n_all,) else v)
                  for k, v in table.items()}
        self.n = int(obs.sum())
        _require(self.n > 0, 'no eligible decision with an observed reward')
        self.taus, self.families = table['taus'], table['families']
        self.draws, self.minimum, self.share = draws, dict(minimum), invalid_share_max
        keys = {'game': self.t['game'].tolist(), 'pitcher': self.t['pitcher'].tolist()}
        self.boots = {u: ClusterBootstrap(keys[u], unit=u, draws=draws, seed=seeds[u]) for u in UNITS}
        self._validity = {}

    # subsets: (numerator rows, denominator rows)
    def subset(self, name):
        t = self.t
        ones = np.ones(self.n, dtype=bool)
        if name == 'all':
            return ones, ones
        if name.startswith('D:') or name.startswith('not_D:'):
            margin = name.split(':', 1)[1]
            index = [margin_key(m) for m in t['margins']].index(margin)
            d = t['D'][:, index]
            return (d, d) if name.startswith('D:') else (~d, ~d)
        if name == 'two_strike':
            two = t['strikes'] == 2
            return two, two
        if name == 'no_positivity_pa':
            keep = ~t['pa_pos']
            return keep, keep
        if name == 'pa_first_D:0':
            return t['first_D0'], t['first_eligible']
        raise IntegrityError(f'unknown subset {name}')

    def validity(self, name):
        """Per unit: replicate validity [draws] (subset nonempty, distinct games >= minimum and PAs >= minimum for the
        game bootstrap; distinct pitchers >= minimum for the pitcher bootstrap) and the full-sample minimum flags."""
        if name in self._validity:
            return self._validity[name]
        num, den = self.subset(name)
        rows = np.flatnonzero(den)
        pa = self.t['pa_index'][rows]
        first = np.zeros(self.n)
        first[rows[np.r_[True, pa[1:] != pa[:-1]]] if len(rows) else rows] = 1.
        out = {'n': int(num.sum()), 'n_denominator': int(den.sum()), 'n_pas': int(first.sum()), 'units': {},
               'below_minimum': {}}
        for unit, boot in self.boots.items():
            present = (boot.sums(den.astype(np.float64)) > 0).astype(np.float64)
            count = boot.replicate(den.astype(np.float64))
            distinct = boot.drawn @ present
            least = self.minimum['games' if unit == 'game' else 'pitchers']
            valid = (count > 0) & (distinct >= least)
            below = present.sum() < least
            if unit == 'game':
                valid &= boot.replicate(first) >= self.minimum['pas']
                below = below or first.sum() < self.minimum['pas']
            out['units'][unit] = valid
            out['below_minimum'][unit] = bool(below)
        self._validity[name] = out
        return out

    def parts(self, f, variant, model=None, rho=None, qa=None, reward=None):
        """Per-row parts [n, T] of a variant for family index f (or explicit placebo arrays)."""
        t = self.t
        model = t['model'][:, f, :] if model is None else model
        rho = t['rho'][:, f, :] if rho is None else rho
        qa = t['qa'][:, f] if qa is None else qa
        reward = t['G'] if reward is None else reward
        on = t['on_mask'][:, None]
        w = (rho - t['rho_ref'][:, None]) * on
        resid = np.where(on, reward[:, None] - qa[:, None], 0.)
        if variant == 'dr':
            return {'M': model, 'C': w * resid}
        if variant.startswith('switch:'):
            cap = float(variant.split(':', 1)[1])
            return {'M': model, 'C': w * resid * (rho <= cap)}
        if variant == 'ips':
            return {'I': w * np.where(on, reward[:, None], 0.)}
        if variant == 'sndr':
            ref = np.broadcast_to(t['rho_ref'][:, None] * on, rho.shape)
            return {'M': model, 'pr': rho * resid, 'p': rho, 'rr': ref * resid, 'r': ref}
        raise IntegrityError(f'unknown variant {variant}')

    @staticmethod
    def stats(variant, s, count):
        count = np.asarray(count, dtype=np.float64)[..., None]
        with np.errstate(divide='ignore', invalid='ignore'):
            if variant == 'ips':
                delta = s['I'] / count
                return {'delta': delta}, np.ones(delta.shape, dtype=bool)
            model = s['M'] / count
            if variant == 'sndr':
                corr = s['pr'] / s['p'] - s['rr'] / s['r']
                ok = (s['p'] > 0) & (s['r'] > 0)
            else:
                corr = s['C'] / count
                ok = np.ones(model.shape, dtype=bool)
            delta = model + corr
            return {'delta': delta, 'M': model, 'C': corr, 'R': delta / model}, ok

    def cell(self, parts, variant, sub):
        num, den = self.subset(sub)
        val = self.validity(sub)
        w = num.astype(np.float64)[:, None]
        point, ok = self.stats(variant, {k: (v * w).sum(axis=0) for k, v in parts.items()}, den.sum())
        reps, valid = {}, {}
        for unit, boot in self.boots.items():
            stats, ok_u = self.stats(variant, {k: boot.replicate(v * w) for k, v in parts.items()},
                                     boot.replicate(den.astype(np.float64)))
            reps[unit] = stats
            valid[unit] = {s: val['units'][unit][:, None] & ok_u & np.isfinite(v) for s, v in stats.items()}
        info = {'n': val['n'], 'n_denominator': val['n_denominator'], 'n_pas': val['n_pas'],
                'unit': 'WE/PA' if sub.startswith('pa_') else 'WE/decision', 'below_minimum': val['below_minimum']}
        return Cell(point, reps, valid, info)

    def family_cell(self, family, variant, sub):
        return self.cell(self.parts(self.families.index(family), variant), variant, sub)

    def record(self, cell, evaluable_by_tau=None):
        return cell.record(self.taus, evaluable_by_tau, self.share, self.draws)


def paired(a, b, stat='delta'):
    """a - b on the shared replicates (both valid)."""
    return Cell({stat: a.point[stat] - b.point[stat]},
                {u: {stat: a.reps[u][stat] - b.reps[u][stat]} for u in a.reps},
                {u: {stat: a.valid[u][stat] & b.valid[u][stat]} for u in a.reps}, dict(a.info))


class DrawMean:
    """Streaming mean over placebo draws of one statistic, replicate by replicate (paired with every other cell on the
    shared resampling), plus each draw's full-sample points (for the 2.5-97.5% band). No draw's replicates are kept."""
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
            entry = {}
            for name, values in self.points.items():
                column = np.array([v[j] for v in values], dtype=np.float64)
                column = column[np.isfinite(column)]
                entry[name] = ({'mean': float(column.mean()), 'band_2.5_97.5': [float(np.quantile(column, .025)),
                                                                                 float(np.quantile(column, .975))]}
                               if len(column) else None)
            out[tau_key(t)] = entry
        return {'draws': self.k, 'per_tau': out}


# ---------------------------------------------------------------- trend, max-t, bounds

def trend(cell, taus, evaluable_by_tau, draws, invalid_share_max, stat='delta'):
    """OLS slope of the per-tau point estimate on log(1/tau) over the evaluable finite taus, with the shared-bootstrap
    percentile CI per unit (a replicate is valid when every used tau is valid)."""
    used = [j for j, t in enumerate(taus) if t != INF and evaluable_by_tau.get(tau_key(t))]
    out = {'x': 'log(1/tau)', 'taus': [tau_key(taus[j]) for j in used], 'slope': None}
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


def max_t(cell, taus, evaluable_by_tau, draws, invalid_share_max, stat='delta'):
    """One-sided joint max-t over the evaluable finite taus: T = max_j point_j / se_j; the null law of
    max_j (rep_j - point_j) / se_j from the shared bootstrap; p = P(max* >= T)."""
    used = [j for j, t in enumerate(taus) if t != INF and evaluable_by_tau.get(tau_key(t))]
    out = {'taus': [tau_key(taus[j]) for j in used], 'one_sided': 'H0: every delta <= 0'}
    if not used:
        return {**out, 'reason': 'no evaluable finite tau'}
    point = cell.point[stat][used]
    for unit in cell.reps:
        reps = cell.reps[unit][stat][:, used]
        ok = cell.valid[unit][stat][:, used].all(axis=1)
        if (draws - ok.sum()) / draws > invalid_share_max or ok.sum() < 2:
            out[unit] = {'valid': False, 'reason': 'INVALID: invalid replicate share above the registered maximum'}
            continue
        se = reps[ok].std(axis=0, ddof=1)
        if not (se > 0).all():
            out[unit] = {'valid': False, 'reason': 'zero bootstrap SE at a used tau'}
            continue
        t_obs = point / se
        null = ((reps[ok] - point) / se).max(axis=1)
        out[unit] = {'valid': True, 't': {tau_key(taus[j]): float(v) for j, v in zip(used, t_obs)},
                     'max_t': float(t_obs.max()), 'p_value': float(np.mean(null >= t_obs.max())),
                     'critical_95': float(np.quantile(null, .95)), 'replicates': int(ok.sum())}
    return out


def bounds(table, family):
    """Worst-case bounds of the subset-all DR mean when the excluded no-reward PAs are added back: their correction
    terms take g in {0, 1} (one unknown per PA, the extreme per PA), the denominator counts their decisions."""
    f = table['families'].index(family)
    obs, on = table['observed'], table['on_mask']
    w = (table['rho'][:, f, :] - table['rho_ref'][:, None]) * on[:, None]
    qa = np.where(on, table['qa'][:, f], 0.)
    model = table['model'][:, f, :]
    observed_sum = (model[obs] + w[obs] * (table['G'][obs][:, None] - qa[obs][:, None])).sum(axis=0)
    un = ~obs
    out = {'family': family, 'unobserved_decisions': int(un.sum()),
           'unobserved_pas': int(len(set(table['pa_index'][un].tolist()))), 'rule': 'g in {0, 1} per excluded PA'}
    lo, hi = observed_sum.copy(), observed_sum.copy()
    if un.any():
        pa = table['pa_index'][un]
        starts = np.flatnonzero(np.r_[True, pa[1:] != pa[:-1]])
        base = np.add.reduceat(model[un] - w[un] * qa[un][:, None], starts, axis=0)
        slope = np.add.reduceat(w[un], starts, axis=0)
        lo += (base + np.minimum(0., slope)).sum(axis=0)
        hi += (base + np.maximum(0., slope)).sum(axis=0)
    total = len(obs)
    out['bounds'] = {tau_key(t): [_finite(lo[j] / total), _finite(hi[j] / total)] for j, t in enumerate(table['taus'])}
    return out


# ---------------------------------------------------------------- placebos (vectorised)

def _kl_rows(pi, reference, mask):
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(mask & (pi > 0), pi * np.log(pi / reference), 0.)
    return terms.sum(axis=1)


def match_tau(q, reference, mask, target, tol, lo_limit=1e-6, hi_limit=1e6):
    """Placebo temperature whose mean KL(pi_p || reference) is within ``tol`` (relative) of ``target``: log-space
    bisection (mean KL decreases in tau). Returns (tau_p, mean KL, matched)."""
    if not target > 0:
        return INF, 0., True

    def mean_kl(tau):
        return float(_kl_rows(tilt_rows(q, reference, mask, tau), reference, mask).mean())
    lo, hi = 1e-2, 1e-1
    k_lo, k_hi = mean_kl(lo), mean_kl(hi)
    while k_lo < target and lo > lo_limit:
        lo, hi, k_hi = lo / 8, lo, k_lo
        k_lo = mean_kl(lo)
    while k_hi > target and hi < hi_limit:
        lo, k_lo, hi = hi, k_hi, hi * 8
        k_hi = mean_kl(hi)
    for tau, value in ((lo, k_lo), (hi, k_hi)):
        if abs(value - target) <= tol * target:
            return tau, value, True
    if not (k_lo >= target >= k_hi):
        tau, value = (lo, k_lo) if k_lo < target else (hi, k_hi)
        return tau, value, False
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        value = mean_kl(mid)
        if abs(value - target) <= tol * target:
            return mid, value, True
        lo, hi = (mid, hi) if value > target else (lo, mid)
    return mid, value, False


def placebo_rows(table):
    """The observed eligible rows' vectors for the vectorised placebos."""
    obs = table['observed']
    return {k: table[k][obs] for k in ('mask', 'ref', 'q_pi', 'q_dr', 'a', 'b_a', 'sha8', 'on_mask', 'rho_ref')}


def batch_mirror_check(table, rows):
    """Max |row-wise mirror - kl_policy| over the temperature family's model terms and rho' (every observed decision
    and tau): the vectorised placebos are only trusted when it is within the identity tolerance."""
    f = table['families'].index('temperature')
    obs = table['observed']
    q_dr = np.where(rows['mask'], rows['q_dr'], 0.)
    idx = np.arange(len(rows['a']))
    worst = 0.
    for j, tau in enumerate(table['taus']):
        pi = tilt_rows(rows['q_pi'], rows['ref'], rows['mask'], tau)
        model = ((pi - rows['ref']) * q_dr).sum(axis=1)
        rho = np.where(rows['on_mask'], pi[idx, rows['a']] / np.where(rows['on_mask'], rows['b_a'], 1.), 0.)
        worst = max(worst, float(np.abs(model - table['model'][obs, f, j]).max()),
                    float(np.abs(rho - table['rho'][obs, f, j]).max()))
    return worst


def perm_placebo(engine, table, *, draws_k, seed, kl_tol, checkpoint=None):
    """(a) perm and (e) estimator_stress: q_pi permuted over the mask inside each decision (perm_rng per draw and row),
    the placebo temperature matched per draw and tau so that its mean KL equals the temperature family's (within
    ``kl_tol``); (a) keeps q_dr, (e) permutes q_dr with the same permutation. Returns (perm DrawMean, stress DrawMean,
    the tau_p summary)."""
    rows = placebo_rows(table)
    obs = table['observed']
    f = table['families'].index('temperature')
    taus = table['taus']
    targets = [float(table['kl'][obs, f, j].mean()) for j in range(len(taus))]
    n = len(rows['a'])
    idx = np.arange(n)
    masks = [np.flatnonzero(m) for m in rows['mask']]
    q_dr_masked = np.where(rows['mask'], rows['q_dr'], 0.)
    b_safe = np.where(rows['on_mask'], rows['b_a'], 1.)
    perm, stress, record = DrawMean(), DrawMean(), {'tau_p': [], 'kl': [], 'matched': []}
    for k in range(draws_k):
        if checkpoint is not None:
            checkpoint()
        q_perm, dr_perm = rows['q_pi'].copy(), rows['q_dr'].copy()
        for i, where in enumerate(masks):
            order = where[perm_rng(seed, k, rows['sha8'][i]).permutation(len(where))]
            q_perm[i, where] = rows['q_pi'][i, order]
            dr_perm[i, where] = rows['q_dr'][i, order]
        dr_perm_masked = np.where(rows['mask'], dr_perm, 0.)
        model, model_s, rho = (np.zeros((n, len(taus))) for _ in range(3))
        tau_p, kls, matched = [], [], []
        for j, tau in enumerate(taus):
            if tau == INF:
                tau_p.append(INF), kls.append(0.), matched.append(True)
                rho[:, j] = rows['rho_ref']
                continue
            tp, value, ok = match_tau(q_perm, rows['ref'], rows['mask'], targets[j], kl_tol)
            tau_p.append(tp), kls.append(value), matched.append(ok)
            pi = tilt_rows(q_perm, rows['ref'], rows['mask'], tp)
            model[:, j] = ((pi - rows['ref']) * q_dr_masked).sum(axis=1)
            model_s[:, j] = ((pi - rows['ref']) * dr_perm_masked).sum(axis=1)
            rho[:, j] = np.where(rows['on_mask'], pi[idx, rows['a']] / b_safe, 0.)
        qa = np.where(rows['on_mask'], rows['q_dr'][idx, rows['a']], 0.)
        qa_s = np.where(rows['on_mask'], dr_perm[idx, rows['a']], 0.)
        perm.add(engine.cell(engine.parts(f, 'dr', model=model, rho=rho, qa=qa), 'dr', 'all'))
        stress.add(engine.cell(engine.parts(f, 'dr', model=model_s, rho=rho, qa=qa_s), 'dr', 'all'))
        record['tau_p'].append(tau_p)
        record['kl'].append(kls)
        record['matched'].append(matched)
    return perm, stress, _tau_p_summary(record, taus, targets)


def _tau_p_summary(record, taus, targets):
    out = {}
    for j, t in enumerate(taus):
        tp = np.array([row[j] for row in record['tau_p']], dtype=np.float64)
        kls = np.array([row[j] for row in record['kl']], dtype=np.float64)
        finite = tp[np.isfinite(tp)]
        out[tau_key(t)] = {'target_mean_kl': targets[j], 'tau_p_mean': _finite(finite.mean()) if len(finite) else None,
                           'tau_p_min': _finite(finite.min()) if len(finite) else None,
                           'tau_p_max': _finite(finite.max()) if len(finite) else None,
                           'max_relative_kl_mismatch': (None if not targets[j] > 0 else
                                                        float(np.abs(kls / targets[j] - 1).max())),
                           'unmatched_draws': int(sum(not row[j] for row in record['matched']))}
    return out


def outcome_shuffle(engine, *, draws_k, seed, checkpoint=None):
    """(d) G permuted across the eligible PAs within the same first-row pitcher (SeedSequence([seed, k]); pitchers in
    str order), the temperature family's DR and IPS-only re-evaluated: expectation 0 for an unbiased estimator."""
    t = engine.t
    f = engine.families.index('temperature')
    pa = t['pa_index']
    starts = np.flatnonzero(np.r_[True, pa[1:] != pa[:-1]])
    owner = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(pa)]))
    reward_pa, pitcher_pa = t['G'][starts], t['pitcher'][starts]
    groups = {}
    for i, p in enumerate(pitcher_pa.tolist()):
        groups.setdefault(p, []).append(i)
    order = sorted(groups, key=str)
    cells = {'dr': DrawMean(), 'ips': DrawMean()}
    for k in range(draws_k):
        if checkpoint is not None:
            checkpoint()
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(k)]))
        shuffled = reward_pa.copy()
        for p in order:
            members = np.array(groups[p])
            shuffled[members] = reward_pa[members[rng.permutation(len(members))]]
        reward = shuffled[owner]
        for variant in cells:
            cells[variant].add(engine.cell(engine.parts(f, variant, reward=reward), variant, 'all'))
    return cells


# ---------------------------------------------------------------- output helpers

def per_tau_rows(sections):
    """Flat rows for per_tau.csv: [(section, family, variant, subset, record)] -> one row per tau."""
    rows = []
    for section, family, variant, subset, record in sections:
        for key, cell in record.items():
            row = {'section': section, 'family': family, 'variant': variant, 'subset': subset, 'tau': key,
                   'n': cell.get('n'), 'evaluable': cell.get('evaluable'), 'descriptive': cell.get('descriptive')}
            for stat in STATS:
                entry = cell.get(stat) or {}
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


# ---------------------------------------------------------------- composition

def inf_identity(table):
    """tau = inf is the reference exactly: every family's model term is 0 and rho' == rho_r bit for bit (D = 0)."""
    j = table['taus'].index(INF)
    n = len(table['observed'])
    record = {'max_abs_model': float(np.abs(table['model'][:, :, j]).max()) if n else 0.,
              'max_abs_rho_minus_rho_ref': float(np.abs(table['rho'][:, :, j] - table['rho_ref'][:, None]).max()) if n
              else 0.}
    _require(record['max_abs_model'] == 0. and record['max_abs_rho_minus_rho_ref'] == 0.,
             f'tau = inf identity failed (pi = reference must give D = 0 exactly): {record}')
    return record


def analysis(table, prediag_record, *, primary, variants, main_families, margins, bootstrap, minimum, invalid_share_max,
             placebos, sensitivities, identity_tolerance, checkpoint=None):
    """Every G-dependent section on the shared resampling: primary, dose_response, subsets, pa_level, trend, max_t,
    bounds, sensitivities and placebos. Returns (sections, per_tau.csv sections)."""
    tick = checkpoint or (lambda: None)
    eng = Engine(table, draws=bootstrap['draws'], seeds=bootstrap['seeds'], minimum=minimum,
                 invalid_share_max=invalid_share_max)
    taus = table['taus']
    ev = {fam: evaluable(prediag_record, fam) for fam in table['families']}
    out = {'tau_inf_identity': inf_identity(table)}
    csv, cells = [], {}
    dose = {}
    for fam in main_families:
        dose[fam] = {}
        for variant in variants:
            tick()
            cells[(fam, variant)] = eng.family_cell(fam, variant, 'all')
            dose[fam][variant] = eng.record(cells[(fam, variant)], ev[fam])
            csv.append(('dose_response', fam, variant, 'all', dose[fam][variant]))
    out['dose_response'] = dose
    real = cells[('temperature', 'dr')]
    key = tau_key(primary['tau'])
    out['primary'] = {**{k: v for k, v in primary.items() if k != 'tau'}, 'tau': key, 'evaluable': ev['temperature'][key],
                      'cell': dose[primary['family']][primary['variant']][key]}
    subsets = {'all': dose['temperature']['dr']}
    for sub in [*[f'D:{margin_key(m)}' for m in margins], 'not_D:0', 'two_strike']:
        tick()
        subsets[sub] = eng.record(eng.family_cell('temperature', 'dr', sub), ev['temperature'])
        csv.append(('subsets', 'temperature', 'dr', sub, subsets[sub]))
    out['subsets'] = {'family': 'temperature', 'variant': 'dr', 'confident_slice': 'D:1se', 'by_subset': subsets}
    tick()
    pa_level = eng.record(eng.family_cell('temperature', 'dr', 'pa_first_D:0'), ev['temperature'])
    out['pa_level'] = {'pa_first_D:0': pa_level}
    csv.append(('pa_level', 'temperature', 'dr', 'pa_first_D:0', pa_level))
    out['trend'] = {fam: trend(cells[(fam, 'dr')], taus, ev[fam], eng.draws, invalid_share_max) for fam in main_families}
    out['max_t'] = {fam: max_t(cells[(fam, 'dr')], taus, ev[fam], eng.draws, invalid_share_max) for fam in main_families}
    out['bounds'] = {fam: bounds(table, fam) for fam in main_families}
    sens = {'variants': 'IPS-only, SN-DR and the switch caps are the dose_response variants of each family'}
    if sensitivities['s_pos']:
        tick()
        cell = eng.family_cell('temperature', 'dr', 'no_positivity_pa')
        sens['s_pos'] = {'rule': 'whole PAs containing a logging-positivity node excluded',
                         'cell': eng.record(cell, ev['temperature'])}
        csv.append(('sensitivities', 'temperature', 'dr', 'no_positivity_pa', sens['s_pos']['cell']))
    if sensitivities['cross_replicate']:
        tick()
        cell = eng.family_cell('cross_replicate', 'dr', 'all')
        sens['cross_replicate'] = {'rule': 'policy from q_reference, DR with q_planning',
                                   'cell': eng.record(cell, ev['cross_replicate']),
                                   'paired_temperature_minus_cross': eng.record(paired(real, cell), ev['temperature'])}
        csv.append(('sensitivities', 'cross_replicate', 'dr', 'all', sens['cross_replicate']['cell']))
    out['sensitivities'] = sens
    pl = {'reading_rule': 'paired differences real - placebo on the shared bootstrap resampling; band = 2.5-97.5% of '
                          'the K placebo-draw means', 'inf': out['tau_inf_identity'], 'policy_side_only': True,
          'donor': {'status': 'NOT_IMPLEMENTED', 'reason': 'a same-count / same-mask donor does not exist for every '
                    'decision; dropping or defaulting those decisions changes the placebo population (needs a design '
                    'decision)'}}
    if placebos['neg']:
        tick()
        cell = eng.family_cell('neg', 'dr', 'all')
        pl['neg'] = {'cell': eng.record(cell, ev['temperature']),
                     'paired_real_minus_neg': eng.record(paired(real, cell), ev['temperature'])}
        csv.append(('placebos', 'neg', 'dr', 'all', pl['neg']['cell']))
    if placebos['perm_draws']:
        rows = placebo_rows(table)
        mirror = batch_mirror_check(table, rows)
        _require(mirror <= identity_tolerance, f'row-wise kl_policy mirror differs from kl_policy by {mirror!r}')
        perm, stress, tau_p = perm_placebo(eng, table, draws_k=placebos['perm_draws'], seed=placebos['perm_seed'],
                                           kl_tol=placebos['kl_tolerance'], checkpoint=tick)
        for name, acc in (('perm', perm), ('estimator_stress', stress)):
            mean = acc.cell()
            pl[name] = {'draws': acc.k, 'seed': placebos['perm_seed'], 'kl_tolerance': placebos['kl_tolerance'],
                        'q_dr': 'fixed' if name == 'perm' else 'permuted with the same permutation (estimator stress)',
                        'tau_p': tau_p, 'band': acc.band(taus), 'mean': eng.record(mean, ev['temperature']),
                        'paired_real_minus_mean': eng.record(paired(real, mean), ev['temperature'])}
            csv.append(('placebos', name, 'dr', 'all', pl[name]['mean']))
        pl['batch_mirror_max_abs_diff'] = mirror
    if placebos['outcome_shuffle_draws']:
        shuffled = outcome_shuffle(eng, draws_k=placebos['outcome_shuffle_draws'],
                                   seed=placebos['outcome_shuffle_seed'], checkpoint=tick)
        pl['outcome_shuffle'] = {'draws': placebos['outcome_shuffle_draws'], 'seed': placebos['outcome_shuffle_seed'],
                                 'rule': 'G permuted across eligible PAs within the same first-row pitcher'}
        for variant, acc in shuffled.items():
            base = cells.get(('temperature', variant)) or eng.family_cell('temperature', variant, 'all')
            mean = acc.cell()
            pl['outcome_shuffle'][variant] = {'band': acc.band(taus), 'mean': eng.record(mean, ev['temperature']),
                                              'paired_real_minus_mean': eng.record(paired(base, mean), ev['temperature'])}
            csv.append(('placebos', f'outcome_shuffle:{variant}', variant, 'all', pl['outcome_shuffle'][variant]['mean']))
    out['placebos'] = pl
    out['bootstrap'] = {'draws': eng.draws, 'seeds': dict(bootstrap['seeds']), 'invalid_share_max': invalid_share_max,
                        'minimum': dict(minimum), 'clusters': {u: b.n_clusters for u, b in eng.boots.items()},
                        'ci_levels': list(CI_LEVELS), 'pitcher_cluster': "the PA's first-row pitcher (A2 convention)",
                        'paired': 'one multiplicity matrix per unit for every family, variant, subset, tau and placebo',
                        'trend_seed_use': 'registered; unused: trend and max-t are computed on the shared game/pitcher '
                                          'replicates so they stay paired with every curve'}
    return out, csv
