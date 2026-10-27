"""EXP-OPE-RES-01 R2 estimator variants: an ADDITIONAL code path beside policy_estimator.

Every variant keeps the registered per-decision recursion V_t = v_hat(H_t) + rho~_t (r_t + V_{t+1} -
q_hat(H_t, a_t)) and the L1-R censoring of ``policy_estimator`` (affine base + slope x c); only rho
changes, identically for both policies:
- R2_0 current DR: rho~ = rho (no clipping, no self-normalisation);
- R2_1 DR-shrinkage (Su et al. 2020): rho~ = lambda rho / (rho^2 + lambda); lambda = inf is R2_0;
- R2_2 Switch-DR (Wang et al. 2017): rho~ = rho 1[rho <= M]; a switched step uses the model value
  v_hat(H_t) and drops every later correction (the zero branch of the recursion); M = inf is R2_0;
- R2_3 self-normalised DR (WDR form): V_0 = sum_t (W_t v_t - W_{t+1} q_t) + W_T r with every W_t
  replaced by W_t / n_bar_t, n_bar_t the mean over ALL E0 PAs of the depth-t cumulative weight (a PA
  holds its last W after its line ends; a censored PA holds W at its node k).
The classification (status, node, resolution, the point set P) is always the registered one, from
the recorded ratios; the steps are re-validated by ``policy_estimator._steps``. With the identity
transform the rows and the aggregate equal ``policy_estimator.pa_value``/``estimate`` (L1-R)
bit for bit (tested). lambda/M selection (``select_parameter``) is outcome-dependent and runs only
inside the sealed evaluator (ope_sealed) or inside a synthetic replication.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from . import policy_estimator as est
from .ope_redesign_registry import INF, LAMBDA_GRID, M_GRID, Z_BOUND, cluster_se, select_minimizer
from .policy_artifacts import IntegrityError, _require

POLICIES = est.POLICIES
KINDS = ('identity', 'shrink', 'switch', 'sn')
POINT_ENDS = ('complete', 'shared_point')


# ---------------------------------------------------------------- transforms
def transform(kind, param=None):
    """rho -> rho~ for one variant; None is the identity (the exact same float flows through)."""
    _require(kind in ('identity', 'shrink', 'switch'), f'unknown rho transform {kind!r}')
    if kind == 'identity' or param == INF:
        return None
    lam = float(param)
    _require(math.isfinite(lam) and lam > 0, f'{kind}: registered positive finite parameter or inf required')
    if kind == 'shrink':
        return lambda r: r if r == 0 else lam * r / (r * r + lam)
    return lambda r: r if r <= lam else 0.0


def variant_key(kind, param=None):
    return f'{kind}:{"inf" if param == INF else param}' if kind in ('shrink', 'switch') else kind


# ---------------------------------------------------------------- compact validated lines
def compact(steps):
    """(v_candidate, v_reference, q_at_logged, rho_candidate, rho_reference) per validated step,
    with exactly the floats ``policy_estimator.affine`` uses."""
    out = []
    for policies, q, mask, a, rho in steps:
        v = tuple(float(policies[name][mask] @ q[mask]) for name in POLICIES)
        out.append((v[0], v[1], q[a], rho['candidate'], rho['reference']))
    return tuple(out)


def pa_line(decisions, info, *, natural=est.H_K_REFUSALS):
    """The registered L1-R classification of one PA and the validated steps of its value line.

    ``end``: complete (c = reward), shared_point (both policies end on one known value, or a zero
    slope), interval_shared (one unknown end value shared by both policies), interval_independent
    (the D-5 worst-case residual), outside_e0, no_decision."""
    status, k, kind, reason = est.classify(decisions, info)
    if status == 'FAILED':
        raise IntegrityError(f'FAILED decision: {reason}; a halted ledger is not estimated')
    line = {'status': status, 'node': k, 'kind': kind, 'reason': reason, 'game': info['game'],
            'in_population': status in (est.COMPLETE, est.CENSORED), 'validity_violation': reason == est.POSITIVITY,
            'resolution': None, 'steps': (), 'reward': None, 'reward_source': None, 'terminal_residual': None}
    if status in (est.EXCLUDED_PRE_START, est.UNSUBMITTABLE):
        return {**line, 'end': 'outside_e0'}
    if status == est.NO_DECISION:
        return {**line, 'end': 'no_decision', 'reward': info.get('reward')}
    if status == est.COMPLETE:
        steps = est._steps(decisions)
        _, q, mask, a, _ = steps[-1]
        reward = info['reward']
        _require(isinstance(reward, float) and np.isfinite(reward) and 0 <= reward <= 1, 'terminal WE outside [0, 1]')
        return {**line, 'end': 'complete', 'steps': compact(steps), 'reward': reward, 'reward_source': 'observed',
                'terminal_residual': float(info['reward'] - q[a]) if mask[a] else None}
    if kind == est.REFUSED and reason == est.POSITIVITY:
        steps = est._steps(decisions[:k + 1], positivity=True)
        return {**line, 'end': 'shared_point', 'steps': compact(steps), 'reward': 0., 'reward_source': 'placeholder',
                'resolution': 'rho_zero'}
    steps = est._steps(decisions[:k])
    if steps and all(est.affine(steps, name)[1] == 0 for name in POLICIES):
        return {**line, 'end': 'shared_point', 'steps': compact(steps), 'reward': 0., 'reward_source': 'placeholder',
                'resolution': 'rho_zero_before_node'}
    if kind == est.REFUSED and reason in natural:
        reward = info.get('reward')
        if reward is None:
            return {**line, 'end': 'interval_shared', 'steps': compact(steps), 'resolution': 'natural_course'}
        return {**line, 'end': 'shared_point', 'steps': compact(steps), 'reward': float(reward),
                'reward_source': 'observed', 'resolution': 'natural_course'}
    if kind == est.TERMINAL_VALUE_MISSING:
        return {**line, 'end': 'interval_shared', 'steps': compact(steps), 'resolution': 'shared_end_unknown'}
    return {**line, 'end': 'interval_independent', 'steps': compact(steps), 'resolution': 'worst_case_residual'}


def is_point(line):
    """An E0 line whose per-policy values are both unknown-free (complete or shared point). NOT the point
    set P: P is defined from the baseline rows' paired bounds (registry.in_point_set)."""
    return line['in_population'] and line['end'] in POINT_ENDS


# ---------------------------------------------------------------- value lines
def affine_t(steps, which, f=None):
    """(base, slope) of V_0 = base + slope c; ``policy_estimator.affine`` with rho~ = f(rho)."""
    base, slope = 0., 1.
    for step in reversed(steps):
        v, qa, r = step[which], step[2], step[3 + which]
        if f is not None:
            r = f(r)
        base, slope = (v, 0.) if r == 0 else (v + r * (base - qa), r * slope)
    return base, slope


def affine_sn(steps, which, u):
    """Self-normalised (base, slope): each W_t carries u_t = 1 / n_bar_t (u = 1 gives affine_t)."""
    T = len(steps)
    base, slope = 0., float(u[T])
    for t in reversed(range(T)):
        v, qa, r = steps[t][which], steps[t][2], steps[t][3 + which]
        base, slope = (u[t] * v, 0.) if r == 0 else (u[t] * v + r * (base - u[t + 1] * qa), r * slope)
    return base, slope


def _lines_of(line, f, u):
    """Per-policy (base, slope); ``u`` = (u_candidate, u_reference) inverse SN normalisers or None."""
    _require(f is None or u is None, 'self-normalisation applies to the plain ratios only')
    return [affine_t(line['steps'], i, f) if u is None else affine_sn(line['steps'], i, u[i]) for i in (0, 1)]


def line_value(line, f=None, u=None, reward=None):
    """pa_value-style values of one line (``reward`` overrides an observed end value: R1 arms)."""
    end = line['end']
    if end == 'outside_e0':
        return {**{f'{n}_bounds': [0., 1.] for n in POLICIES}, 'delta_bounds': [-1., 1.]}
    if end == 'no_decision':
        value = line['reward'] if reward is None else reward
        bounds = [0., 1.] if value is None else [float(value)] * 2
        return {**{f'{n}_bounds': list(bounds) for n in POLICIES}, 'delta_bounds': [0., 0.]}
    (bc, sc), (br, sr) = _lines_of(line, f, u)
    if end in POINT_ENDS:
        c = line['reward'] if reward is None or line['reward_source'] != 'observed' else float(reward)
        row = {'candidate': bc + sc * c, 'weight_candidate': sc, 'reference': br + sr * c, 'weight_reference': sr}
        row['delta'] = row['candidate'] - row['reference']
        row.update({f'{n}_bounds': [row[n]] * 2 for n in POLICIES}, delta_bounds=[row['delta']] * 2)
        return row
    out = {'candidate_bounds': [bc, bc + sc], 'reference_bounds': [br, br + sr]}
    if end == 'interval_shared':
        if f is None and u is None:  # the registered formula: bit-identical to policy_estimator._interval (baseline P)
            ends = [bc - br, bc + sc - br - sr]
        else:  # a variant (review round 3 #1): the shared end cancels exactly when the transformed slopes are equal
            ends = shared_end_bounds(bc, br, sc, sr)
        out['delta_bounds'] = [min(ends), max(ends)]
    else:
        out['delta_bounds'] = [bc - br - sr, bc + sc - br]
    out.update(weight_candidate=sc, weight_reference=sr)
    return out


def shared_end_bounds(bc, br, sc, sr):
    """Paired delta ends of a variant line with one unknown end value c in [0, 1] shared by both policies:
    delta(c) = (bc - br) + (sc - sr) c. Equal transformed slopes cancel algebraically (a point, no rounding
    split); unequal slopes give [bc - br, (bc - br) + (sc - sr)] in either order. No tolerance is used."""
    d = bc - br
    return [d, d] if sc == sr else [d, d + (sc - sr)]


def pa_value_variant(decisions, info, f=None, u=None, *, natural=est.H_K_REFUSALS):
    """``policy_estimator.pa_value(..., 'l1r')`` through the variant path (bit-identical when f is None)."""
    return row_of(pa_line(decisions, info, natural=natural), f, u)


def row_of(line, f=None, u=None, reward=None):
    row = {k: line[k] for k in ('status', 'node', 'kind', 'reason', 'game', 'in_population', 'validity_violation')}
    row.update(line_value(line, f, u, reward))
    if line['status'] == est.COMPLETE:
        row['terminal_residual'] = line['terminal_residual']
    if line['status'] == est.CENSORED:
        row['resolution'] = line['resolution']
    row['l2'] = line['status'] == est.COMPLETE or row.get('resolution') in ('rho_zero', 'rho_zero_before_node')
    return row


# ---------------------------------------------------------------- self-normalisation
def weight_paths(lines, f=None):
    """Per E0 line and policy: cumulative weights W_0..W_T (W_0 = 1) of its value line."""
    out = []
    for line in lines:
        paths = []
        for which in (0, 1):
            w, path = 1., [1.]
            for step in line['steps']:
                r = step[3 + which] if f is None else f(step[3 + which])
                w = w * r
                path.append(w)
            paths.append(path)
        out.append(paths)
    return out


def sn_normalizers(lines):
    """n_bar_t (t = 0..D) per policy over ALL E0 lines: each line holds its last W after its end
    (stopped martingale; a censored line holds W at its node). Returns (n_bar [2, D+1], valid)."""
    e0 = [line for line in lines if line['in_population']]
    _require(e0, 'self-normalisation needs E0 PAs')
    paths = weight_paths(e0)
    depth = max(len(p[0]) for p in paths) - 1
    n_bar = np.zeros((2, depth + 1))
    for p in paths:
        for which in (0, 1):
            w = np.asarray(p[which], dtype=np.float64)
            n_bar[which, :len(w)] += w
            n_bar[which, len(w):] += w[-1]
    n_bar /= len(e0)
    valid = bool(np.isfinite(n_bar).all() and (n_bar > 0).all())
    return n_bar, valid


def inverse_normalizers(n_bar):
    _require(np.isfinite(n_bar).all() and (n_bar > 0).all(), 'a zero or nonfinite self-normaliser: R2_3 invalid')
    return 1.0 / n_bar


def sn_coefficients(line, which, depth):
    """V = base_coef . u + slope_coef * u_T * c for the SN value line (u = inverse normalisers)."""
    coef, w = np.zeros(depth + 1), 1.
    for t, step in enumerate(line['steps']):
        v, qa, r = step[which], step[2], step[3 + which]
        coef[t] += w * v
        w_next = w * r
        if w_next != 0:
            coef[t + 1] -= w_next * qa
        w = w_next
    return coef, w, len(line['steps'])


# ---------------------------------------------------------------- aggregate (mirror of estimate, L1-R)
def estimate_variant(ledger_decisions, pas, *, f=None, sn=False, draws, seed, invalid_share_max, minimum, ess_gate=None,
                     natural=est.H_K_REFUSALS):
    """``policy_estimator.estimate(..., censoring='l1r')`` through the variant path: identical keys
    and values (bit for bit) for the identity transform; the secondary and strata tables are not
    produced. With ``sn`` the game bootstrap is not reported (normalisers must be recomputed per
    resample: ope_bootstrap)."""
    lines, rows = [], []
    for pa_id, info, decisions in est.by_pa(ledger_decisions, pas):
        decisions = sorted(decisions, key=lambda d: d['decision_index'])
        line = pa_line(decisions, info, natural=natural)
        lines.append((pa_id, info, line))
    u = None
    if sn:
        n_bar, valid = sn_normalizers([line for _, _, line in lines])
        _require(valid, 'a zero or nonfinite self-normaliser: R2_3 invalid')
        u = (inverse_normalizers(n_bar[0]), inverse_normalizers(n_bar[1]))
    for pa_id, info, line in lines:
        row = row_of(line, f, u)
        row.update(pa_id=pa_id, end_kind=info.get('end_kind'), flags=list(info.get('flags') or ()),
                   strata=dict(info.get('strata') or {}))
        rows.append(row)
    e0 = [r for r in rows if r['in_population']]
    complete = [r for r in e0 if r.get('l2', r['status'] == est.COMPLETE)]
    result = {'estimand': est.ESTIMAND, 'pas': len(rows), 'status': dict(Counter(r['status'] for r in rows)),
              'reasons': dict(Counter(f'{r["status"]}:{r["reason"]}' for r in rows if r['status'] != est.COMPLETE)),
              'censor_kinds': dict(Counter(r['kind'] for r in e0 if r['status'] == est.CENSORED)),
              'censor_nodes': dict(Counter(str(r['node']) for r in e0 if r['status'] == est.CENSORED)),
              'validity_violation_pas': sum(r['validity_violation'] for r in rows),
              'layers': {'L0_all_pas': est._layer(rows), 'L1_start_population': est._layer(e0)},
              'population_value': None, 'causal_effect': None}
    residual = [r for r in e0 if r.get('resolution') in ('worst_case_residual', 'worst_case_residual_gap')]
    result['censoring_rule'] = {
        'rule': 'L1-R', 'natural_course_refusals': list(natural), 'gap_delta': None, 'excluded_pas': 0,
        'resolutions': dict(Counter(r['resolution'] for r in e0 if r['status'] == est.CENSORED)),
        'worst_case_residual_reasons': dict(Counter(f"{r['kind']}:{r['reason']}" for r in residual)),
        'worst_case_residual_share': len(residual) / len(e0) if e0 else None,
        'worst_case_residual_definition': list(est.WORST_CASE_RESIDUAL),
        'estimand': 'regime: pi until the first H_k-fixed event, then logged behaviour; no verdict on V(pi) itself'}
    result['ess_all_e0'] = {name: {'pa': est.effective_sample_size([r[f'weight_{name}'] for r in e0]),
                                   'game': est.effective_sample_size([r[f'weight_{name}'] for r in e0],
                                                                     [r['game'] for r in e0])} for name in POLICIES}
    if complete:
        deltas = [r['delta'] for r in complete]
        weights = {name: [r[f'weight_{name}'] for r in complete] for name in POLICIES}
        games = [r['game'] for r in complete]
        ess = {name: {'pa': est.effective_sample_size(w), 'game': est.effective_sample_size(w, games)}
               for name, w in weights.items()}
        gate = {level: (None if any(ess[name][level] is None for name in POLICIES)
                        else min(ess[name][level] for name in POLICIES)) for level in ('pa', 'game')}
        result['layers']['L2_complete_conditional'] = {
            'pas': len(complete), 'delta_mean': float(np.mean(deltas)),
            'candidate_mean': float(np.mean([r['candidate'] for r in complete])),
            'reference_mean': float(np.mean([r['reference'] for r in complete])),
            'observed_mean_end_value': (lambda ends: float(np.mean(ends)) if ends else None)(
                [pas[r['pa_id']]['reward'] for r in complete if pas[r['pa_id']].get('reward') is not None]),
            'selection': 'post-treatment selected (complete and supported to the end); no judgement'}
        result['ess'] = {**ess, 'gate_min_candidate_reference': gate}
        if ess_gate is not None:
            weak = any(v is None or v < ess_gate[level] for level, v in gate.items())
            result['ess']['label'] = 'UNCONFIRMED_WEAK_OVERLAP' if weak else None
    else:
        result['layers']['L2_complete_conditional'] = None
        result['ess'] = {'gate_min_candidate_reference': {'pa': None, 'game': None},
                         'label': None if ess_gate is None else 'UNCONFIRMED_WEAK_OVERLAP'}
    if e0:
        censored = [r for r in e0 if r['status'] == est.CENSORED]
        result['censoring'] = {'observed_share': len(censored) / len(e0), **{
            f'estimated_mass_{name}': sum(r[f'weight_{name}'] for r in censored) / len(e0) for name in POLICIES},
            'ess_of_censored_weights': {name: {
                'pa': est.effective_sample_size([r[f'weight_{name}'] for r in censored]) if censored else None,
                'game': est.effective_sample_size([r[f'weight_{name}'] for r in censored],
                                                  [r['game'] for r in censored]) if censored else None} for name in POLICIES}}
        result['bootstrap'] = None if sn else est.game_bootstrap(e0, draws=draws, seed=seed,
                                                                 invalid_share_max=invalid_share_max, minimum=minimum)
    return result, rows


# ---------------------------------------------------------------- per-pair point deltas and selection
def point_deltas(lines, f=None, u=None, rewards=None):
    """Delta_i on the point set P (lines already restricted to P), with optional R1 reward overrides: the
    paired delta when its bounds coincide under this variant, NaN when the variant breaks the baseline's
    slope cancellation (the pair then has no P mean for that value)."""
    out = np.empty(len(lines))
    for i, line in enumerate(lines):
        reward = None if rewards is None else rewards[i]
        lo, hi = line_value(line, f, u, reward)['delta_bounds']
        out[i] = lo if lo == hi else np.nan
    return out


def selection_objective(deltas, reference, clusters):
    """B_hat^2 + V_hat with B_hat = |mean(D(v) - D(inf))| + 1.96 SE_cl(D(v) - D(inf)) and V_hat =
    SE_cl(D(v))^2 (design R2_1; the pinned closed-form cluster SE)."""
    diff = np.asarray(deltas) - np.asarray(reference)
    bias = abs(float(diff.mean())) + Z_BOUND * cluster_se(diff, clusters)
    variance = cluster_se(deltas, clusters) ** 2
    value = bias * bias + variance
    return value if math.isfinite(value) else math.inf


def select_parameter(kind, deltas_by_value, clusters):
    """Registered lambda / M choice from {grid value: Delta_i on P}: minimise B_hat^2 + V_hat; ties
    (relative tolerance 1e-12 of the finite minimum) -> the larger value (toward inf). Returns
    (selected value or None, objectives)."""
    grid = LAMBDA_GRID if kind == 'shrink' else M_GRID
    _require(kind in ('shrink', 'switch') and set(deltas_by_value) == set(grid), 'one delta vector per grid value')
    reference = deltas_by_value[INF]
    objectives = [selection_objective(deltas_by_value[v], reference, clusters) for v in grid]
    return select_minimizer(grid, objectives), dict(zip((variant_key(kind, v) for v in grid), objectives))


def trigger_rate(lines, kind, param):
    """Share of decision steps whose rho~ differs from rho (Switch: rho > M), per policy (weight diagnostic)."""
    f = transform(kind, param)
    counts, changed = [0, 0], [0, 0]
    for line in lines:
        for step in line['steps']:
            for which in (0, 1):
                r = step[3 + which]
                counts[which] += 1
                if f is not None and f(r) != r:
                    changed[which] += 1
    return {name: (changed[i] / counts[i] if counts[i] else None) for i, name in enumerate(POLICIES)}


def mean_abs_shrink(lines, param):
    f = transform('shrink', param)
    values = [[], []]
    for line in lines:
        for step in line['steps']:
            for which in (0, 1):
                r = step[3 + which]
                values[which].append(0. if f is None else abs(f(r) - r))
    return {name: (float(np.mean(values[i])) if values[i] else None) for i, name in enumerate(POLICIES)}
