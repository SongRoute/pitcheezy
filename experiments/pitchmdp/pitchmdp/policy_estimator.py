"""ARM-B per-decision sequential DR over ledger rows (ML-POLICY-ESTIMATOR-v2, COOP-018, D93).

Implements the D89 §5 formula on the rows ``PolicyRuntime`` records: for pi in {candidate,
reference}, ``V_T = 0`` and ``V_t = v(H_t) + rho_t (r_t + V_{t+1} - q(H_t, a_t))`` with the reward
only at the PA's last decision, ``rho_t = pi(a_t|H_t) / pi_b_hat(a_t|H_t)`` (full-vocabulary
logging law, never renormalised), no clipping and no self-normalisation. ``q`` is the recorded
reference-continuation Q (evaluation seed when registered, M-7); for the candidate it is the
declared control variate (D89 §5). Every input is re-validated against the ledger values.

Populations and layers (D-2, D-5): L0 = every regular-season PA of the window (bounds only; a PA
outside the pre-decision start population E0 or unsubmittable at its first row takes [0, 1] per
policy); L1 = E0 (primary: bounds); L2 = E0 PAs that are complete (named post-treatment-selected
conditional mean, no judgement). A PA censored at node k (a refusal or structural defect at k,
an unobserved PA end, or an unobservable end value) is bounded inside the recursion: V_0 is
affine in the unknown node value c in [0, 1] with slope prod_{t<k} rho_t (D-5 C). The bound is
valid in expectation when censoring is fixed by H_k or is the shared end value, and the
candidate path needs pi_b_hat correct; LOGGING_POSITIVITY nodes violate that and are reported
separately. Nothing here identifies a causal effect.

``censoring='l1r'`` (L1-R, COOP-021/022; default ``'worst_case'`` is the registered D-5 above and
reproduces the sealed <=2025 S6):
- LOGGING_POSITIVITY node recorded by runtime v3: both policies give the logged action no mass, so
  rho_k = 0 exactly and V_k = v_pi(H_k) from the recorded pi and Q (point; slope 0, no reward).
  The node needs no unknown; pi_b_hat misspecification is a separate bias path (b_V, V3, S-B).
- refusal fixed by the pre-decision history H_k (``natural``, default ``H_K_REFUSALS``): regime
  "pi until the first H_k-fixed event, then the logged behaviour" -- both policies end on the
  observed PA end value (point; a shared unknown when unobserved), as in the D-4 secondary. No
  verdict on V(pi) itself. Assumption for no-pitch calls: P(no-pitch | H_k, type) = P(no-pitch | H_k).
- every other censoring (``WORST_CASE_RESIDUAL``) keeps the D-5 bound (independent unknowns), or,
  with ``gap_delta`` (sensitivity S-C only), a shared level plus |V_c - V_r| <= delta at the node.
- a censored PA whose ratio products are already 0 for both policies is a point value.
L2 in L1-R holds every E0 PA with an unknown-free point value from ratios alone (complete, rho = 0
at a positivity node or before the censoring node; COOP-019 F4). ``exclude`` moves PAs out of E0
(sensitivity S-B only).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .policy_artifacts import IntegrityError, _require

EVALUATED = ('SUPPORTED', 'OUTSIDE_POLICY_SUPPORT')
FATAL = ('FAILED_INTEGRITY', 'FAILED_RUNTIME')
START_REFUSALS = ('UNSUPPORTED_INCOMPLETE_START', 'UNSUPPORTED_UNKNOWN_PITCHER', 'UNSUPPORTED_EMPTY_SUPPORT',
                  'UNSUPPORTED_PITCHER_HAND')
POSITIVITY = 'UNSUPPORTED_LOGGING_POSITIVITY'
H_K_REFUSALS = ('UNSUPPORTED_NO_LOGGED_ACTION', 'UNSUPPORTED_UNKNOWN_PITCHER', 'UNSUPPORTED_PITCHER_HAND',
                'UNSUPPORTED_EMPTY_SUPPORT', 'UNSUPPORTED_INCONSISTENT_HISTORY')  # fixed before the choice at k
CENSORING = ('worst_case', 'l1r')
WORST_CASE_RESIDUAL = ('NO_TERMINAL', 'UNSUPPORTED_MISSING_ACTION_LABEL', 'structural:*',
                       'any other REFUSED reason outside POSITIVITY and the natural-course list')  # G-R numerator
POLICIES = ('candidate', 'reference')
COMPLETE, CENSORED = 'COMPLETE', 'CENSORED'
EXCLUDED_PRE_START, UNSUBMITTABLE = 'EXCLUDED_PRE_START', 'UNSUBMITTABLE'
NO_DECISION = 'NO_DECISION'  # a PA without any pitcher decision: both policies act identically, delta = 0
REFUSED, NO_TERMINAL, TERMINAL_VALUE_MISSING = 'REFUSED', 'NO_TERMINAL', 'TERMINAL_VALUE_MISSING'
ESTIMAND = {
    'L0_all_pas': 'bounds on the mean paired value difference over every regular-season PA of the window',
    'L1_start_population': 'primary: bounds over the pre-decision start population E0 (first-row pre-pitch '
                           'fields and frozen TRAIN artifacts only)',
    'L2_complete_conditional': 'E0 PAs complete and supported to the end: post-treatment-selected conditional '
                               'mean, descriptive, no judgement',
    'units': 'frozen C0 initial-defender WE; V(candidate) - V(reference)'}


def _probabilities(values, n, name):
    p = np.asarray(values, dtype=np.float64)
    _require(p.shape == (n,) and np.isfinite(p).all() and (p >= 0).all() and abs(p.sum() - 1) <= 1e-9,
             f'{name}: invalid probability vector')
    return p


def _step(row, n, positivity=False):
    """Validated arrays of one evaluated decision row (or, with ``positivity``, of a runtime-v3
    LOGGING_POSITIVITY row: logging[a] = 0, both policies 0 at a, recorded rho exactly 0)."""
    result = row.get('result')
    if positivity and row.get('status') == POSITIVITY:
        return _positivity_step(row, n)
    _require(row.get('status') in EVALUATED and isinstance(result, dict), 'only evaluated decisions enter the estimator')
    mask = np.asarray(result['mask'], dtype=bool)
    _require(mask.shape == (n,) and mask.any(), 'policy mask shape')
    logging = _probabilities(result['logging'], n, 'logging law')
    policies = {name: _probabilities(result[name], n, name) for name in POLICIES}
    for name, p in policies.items():
        _require((p[~mask] == 0).all(), f'{name}: mass outside the policy mask')
    q_values = result.get('q_reference')
    _require(isinstance(q_values, list) and len(q_values) == n, 'recorded reference Q required (candidate runtime)')
    _require(all((v is None) != bool(m) for v, m in zip(q_values, mask)), 'Q must be recorded exactly on the mask')
    q = np.array([np.nan if v is None else v for v in q_values], dtype=np.float64)
    _require(np.isfinite(q[mask]).all(), 'non-finite Q on the mask')
    a = result.get('logged_index')
    _require(type(a) is int and 0 <= a < n and logging[a] > 0, 'logged action with positive logging mass required')
    rho = {}
    for name, p in policies.items():
        recorded = result.get(f'rho_{name}')
        _require(isinstance(recorded, float) and np.isclose(recorded, p[a] / logging[a], rtol=1e-12, atol=0),
                 f'{name}: recorded ratio differs from the recorded probabilities')
        rho[name] = p[a] / logging[a]
    _require(bool(mask[a]) == (row['status'] == 'SUPPORTED'), 'status differs from the logged action support')
    return policies, q, mask, a, rho


def _positivity_step(row, n):
    result = row.get('result')
    _require(isinstance(result, dict), 'LOGGING_POSITIVITY row without a recorded result: runtime v3 required for L1-R')
    mask = np.asarray(result['mask'], dtype=bool)
    _require(mask.shape == (n,) and mask.any(), 'policy mask shape')
    logging = _probabilities(result['logging'], n, 'logging law')
    policies = {name: _probabilities(result[name], n, name) for name in POLICIES}
    for name, p in policies.items():
        _require((p[~mask] == 0).all(), f'{name}: mass outside the policy mask')
    q_values = result.get('q_reference')
    _require(isinstance(q_values, list) and len(q_values) == n, 'recorded reference Q required (candidate runtime)')
    _require(all((v is None) != bool(m) for v, m in zip(q_values, mask)), 'Q must be recorded exactly on the mask')
    q = np.array([np.nan if v is None else v for v in q_values], dtype=np.float64)
    _require(np.isfinite(q[mask]).all(), 'non-finite Q on the mask')
    a = result.get('logged_index')
    _require(type(a) is int and 0 <= a < n and logging[a] == 0 and not mask[a]
             and all(p[a] == 0 for p in policies.values()), 'positivity row: logged action must be off every support')
    _require(all(result.get(f'rho_{name}') == 0.0 for name in POLICIES), 'positivity row: recorded rho must be 0')
    return policies, q, mask, a, {name: 0. for name in POLICIES}


def _steps(decisions, positivity=False):
    _require([d['decision_index'] for d in decisions] == list(range(len(decisions))),
             'decision rows must be the ordered PA prefix')
    _require(len({d['pa_id'] for d in decisions}) <= 1, 'decision rows from more than one PA')
    if not decisions:
        return []
    n = len(decisions[0]['result']['logging'])
    return [_step(d, n, positivity) for d in decisions]


def affine(steps, name):
    """(base, slope): V_0 = base + slope * c for the unknown node value c entering after the last
    step (the terminal reward plus V_T = 0 for a complete PA). slope = prod rho >= 0."""
    base, slope = 0., 1.
    for policies, q, mask, a, rho in reversed(steps):
        v = float(policies[name][mask] @ q[mask])
        r = rho[name]
        base, slope = (v, 0.) if r == 0 else (v + r * (base - q[a]), r * slope)
    return base, slope


def pa_dr(decisions, reward):
    """Sequential DR values of one complete PA from its ordered decision rows and terminal WE."""
    _require(isinstance(reward, float) and np.isfinite(reward) and 0 <= reward <= 1, 'terminal WE outside [0, 1]')
    _require(decisions, 'decision rows must be the complete ordered PA')
    steps = _steps(decisions)
    out = {}
    for name in POLICIES:
        base, slope = affine(steps, name)
        out[name], out[f'weight_{name}'] = base + slope * reward, slope
    out['delta'] = out['candidate'] - out['reference']
    return out


def _interval(steps, shared):
    """Per-policy [lo, hi] and the paired delta interval for an unknown node value in [0, 1]."""
    lines = {name: affine(steps, name) for name in POLICIES}
    out = {f'{name}_bounds': [b, b + s] for name, (b, s) in lines.items()}
    (bc, sc), (br, sr) = lines['candidate'], lines['reference']
    if shared:  # one end value shared by both policies (TERMINAL_VALUE_MISSING)
        ends = [bc - br, bc + sc - br - sr]
        out['delta_bounds'] = [min(ends), max(ends)]
    else:
        out['delta_bounds'] = [bc - br - sr, bc + sc - br]
    out.update({f'weight_{name}': s for name, (_, s) in lines.items()})
    return out


def _interval_gap(steps, delta):
    """S-C: a node level u in [0, 1] shared by both policies plus |V_c - V_r| <= delta."""
    out = _interval(steps, shared=True)
    (bc, sc), (br, sr) = (affine(steps, name) for name in POLICIES)
    out['delta_bounds'] = [bc - br + min(0., sc - sr) - sc * delta, bc - br + max(0., sc - sr) + sc * delta]
    return out


def _shared_end(steps, reward):
    """Per-policy point values when both policies end on the same observed value after ``steps``."""
    row = {}
    for name in POLICIES:
        base, slope = affine(steps, name)
        row[name], row[f'weight_{name}'] = base + slope * reward, slope
        row[f'{name}_bounds'] = [row[name]] * 2
    row['delta'] = row['candidate'] - row['reference']
    row['delta_bounds'] = [row['delta']] * 2
    return row


def classify(decisions, info):
    """One PA -> (status, node k, kind, reason). ``info``: runner facts for the PA (start
    population, structural problem and index, reward/reason/kind of the PA end)."""
    statuses = [d['status'] for d in decisions]
    if any(s in FATAL for s in statuses):
        return 'FAILED', None, None, next(s for s in statuses if s in FATAL)
    problem, index = info.get('problem'), info.get('problem_index')
    if problem == 'no_decision':
        _require(not decisions, 'a PA without a decision row cannot have ledger rows')
        return NO_DECISION, None, None, problem
    if problem is not None and index == 0:
        _require(not decisions, 'a PA unsubmittable at its first row cannot have ledger rows')
        return UNSUBMITTABLE, None, None, problem
    _require(decisions, 'a submitted PA needs its ledger rows')
    if not info['in_population']:
        return EXCLUDED_PRE_START, None, None, info.get('start_reason')
    _require(statuses[0] not in START_REFUSALS, 'E0 start population disagrees with the runtime decision 0')
    refused = [k for k, s in enumerate(statuses) if s not in EVALUATED]
    if problem is not None:
        _require(len(decisions) == index, 'a structural defect ends the submitted prefix exactly')
    if refused:
        k = refused[0]
        _require(all(s == 'UNSUPPORTED_MID_PA' for s in statuses[k + 1:]), 'refusals after the first must be sticky')
        return CENSORED, k, REFUSED, statuses[k]
    if problem is not None:
        return CENSORED, index, REFUSED, f'structural:{problem}'
    if info.get('reward') is None:
        kind = info.get('kind')
        _require(kind in (NO_TERMINAL, TERMINAL_VALUE_MISSING), 'unknown PA-end kind')
        return CENSORED, len(decisions), kind, info.get('reason')
    return COMPLETE, None, None, None


def pa_value(decisions, info, censoring='worst_case', *, natural=H_K_REFUSALS, gap_delta=None):
    """Per-PA row: status, node, bounds or point values (and weights) for both policies."""
    _require(censoring in CENSORING, 'unknown censoring rule')
    status, k, kind, reason = classify(decisions, info)
    row = {'status': status, 'node': k, 'kind': kind, 'reason': reason, 'game': info['game'],
           'in_population': status in (COMPLETE, CENSORED), 'validity_violation': reason == POSITIVITY}
    if status in (EXCLUDED_PRE_START, UNSUBMITTABLE):
        row.update({f'{name}_bounds': [0., 1.] for name in POLICIES}, delta_bounds=[-1., 1.])
    elif status == NO_DECISION:  # no action to change: the same (possibly unknown) end value for both
        end = [0., 1.] if info.get('reward') is None else [float(info['reward'])] * 2
        row.update({f'{name}_bounds': list(end) for name in POLICIES}, delta_bounds=[0., 0.])
    elif status == COMPLETE:
        values = pa_dr(decisions, float(info['reward']))
        row.update(values)
        _, q, mask, a, _ = _steps(decisions)[-1]
        row['terminal_residual'] = float(info['reward'] - q[a]) if mask[a] else None  # D-6 note 5 diagnostic
        row.update({f'{name}_bounds': [values[name]] * 2 for name in POLICIES}, delta_bounds=[values['delta']] * 2)
    elif status == CENSORED and censoring == 'l1r':
        row.update(_l1r_censored(decisions, info, k, kind, reason, natural, gap_delta))
    elif status == CENSORED:
        row.update(_interval(_steps(decisions[:k]), shared=kind == TERMINAL_VALUE_MISSING))
    if censoring == 'l1r':
        row['l2'] = status == COMPLETE or row.get('resolution') in ('rho_zero', 'rho_zero_before_node')
    return row


def _l1r_censored(decisions, info, k, kind, reason, natural, gap_delta):
    if kind == REFUSED and reason == POSITIVITY:
        steps = _steps(decisions[:k + 1], positivity=True)  # last step has rho = 0: V_k = v_pi(H_k)
        return {**_shared_end(steps, 0.), 'resolution': 'rho_zero'}  # slope 0: the end value never enters
    steps = _steps(decisions[:k])
    if steps and all(affine(steps, name)[1] == 0 for name in POLICIES):  # rho = 0 before the node
        return {**_shared_end(steps, 0.), 'resolution': 'rho_zero_before_node'}
    if kind == REFUSED and reason in natural:
        reward = info.get('reward')
        return {**(_interval(steps, shared=True) if reward is None else _shared_end(steps, float(reward))),
                'resolution': 'natural_course'}
    if kind == TERMINAL_VALUE_MISSING:
        return {**_interval(steps, shared=True), 'resolution': 'shared_end_unknown'}
    if gap_delta is not None:
        return {**_interval_gap(steps, gap_delta), 'resolution': 'worst_case_residual_gap'}
    return {**_interval(steps, shared=False), 'resolution': 'worst_case_residual'}


def secondary_value(decisions, info, primary):
    """D-4 secondary estimand (descriptive): after the first pitcher change t* (from the ledger's
    pitcher field) both policies follow the logged behaviour, so V_{t*} is the observed end value
    (a shared unknown when unobserved) and refusals at or after t* do not censor."""
    pitchers = [d['pitcher'] for d in decisions]
    change = next((t for t, p in enumerate(pitchers) if p != pitchers[0]), None) if pitchers else None
    if 'first_pitcher_change_index' in info and decisions:
        manifest = info['first_pitcher_change_index']
        expected = manifest if manifest is not None and manifest < len(decisions) else None
        _require(change == expected, 'ledger pitcher change differs from the request manifest')
    if change is None or not primary['in_population'] or (primary['node'] is not None and primary['node'] < change):
        return {**primary, 'secondary_change': change}
    steps = _steps(decisions[:change])
    if info.get('reward') is None:
        row = {**_interval(steps, shared=True), 'status': CENSORED, 'kind': TERMINAL_VALUE_MISSING}
    else:
        row = {'status': COMPLETE, **_shared_end(steps, float(info['reward']))}
    return {**row, 'game': info['game'], 'in_population': True, 'secondary_change': change}


def effective_sample_size(weights, games=None):
    """(sum w)^2 / sum w^2 over PAs, or over per-game weight sums when ``games`` is given."""
    w = np.asarray(weights, dtype=np.float64)
    _require(w.ndim == 1 and np.isfinite(w).all() and (w >= 0).all(), 'importance weights invalid')
    if games is not None:
        _, inverse = np.unique(np.asarray(games), return_inverse=True)
        w = np.bincount(inverse, weights=w)
    return None if not (w ** 2).sum() else float(w.sum() ** 2 / (w ** 2).sum())


def _layer(rows):
    n = len(rows)
    if not n:
        return None
    out = {'pas': n}
    for name in (*POLICIES, 'delta'):
        lo = sum(r[f'{name}_bounds'][0] for r in rows) / n
        hi = sum(r[f'{name}_bounds'][1] for r in rows) / n
        out[f'{name}_bounds'] = [lo, hi]
    return out


def game_bootstrap(rows, *, draws, seed, invalid_share_max, minimum):
    """Whole-game bootstrap (all PAs of a resampled game, paired across quantities): the L1 bound
    endpoints and the L2 conditional mean. A replicate without a complete PA is invalid for L2;
    their share above ``invalid_share_max`` nulls the L2 interval. Below the registered minimum
    games / PA starts, no interval is reported (P8 POLICY_INFERENCE rule)."""
    _require(type(draws) is int and draws >= 1 and type(seed) is int, 'registered draws/seed required')
    _require(isinstance(invalid_share_max, (int, float)) and not isinstance(invalid_share_max, bool)
             and 0 <= invalid_share_max <= 1, 'registered invalid share required')
    games = sorted({r['game'] for r in rows}, key=str)
    out = {'draws': draws, 'seed': seed, 'games': len(games), 'pa_starts': len(rows), 'unit': 'game'}
    if len(games) < minimum['games'] or len(rows) < minimum['pa_starts']:
        return {**out, 'inference': None, 'reason': 'below registered minimum games / PA starts'}
    index = {g: i for i, g in enumerate(games)}
    g = np.array([index[r['game']] for r in rows])
    complete = np.array([r.get('l2', r['status'] == COMPLETE) for r in rows])
    per = lambda values: np.bincount(g, weights=np.asarray(values, dtype=np.float64), minlength=len(games))
    n_all, n_c = per(np.ones(len(rows))), per(complete)
    lo, hi = per([r['delta_bounds'][0] for r in rows]), per([r['delta_bounds'][1] for r in rows])
    d = per([r['delta'] if r.get('l2', r['status'] == COMPLETE) else 0. for r in rows])
    sample = np.random.default_rng(seed).integers(len(games), size=(draws, len(games)))
    counts = n_all[sample].sum(axis=1)
    boot_lo, boot_hi = lo[sample].sum(axis=1) / counts, hi[sample].sum(axis=1) / counts
    nc = n_c[sample].sum(axis=1)
    valid = nc > 0
    invalid = int((~valid).sum())
    conditional = d[sample].sum(axis=1)[valid] / nc[valid]
    q = lambda v: np.quantile(v, [.025, .975]).tolist()
    return {**out, 'L1_lower_endpoint_ci95': q(boot_lo), 'L1_upper_endpoint_ci95': q(boot_hi),
            'L2_invalid_replicates': invalid,
            'L2_conditional_ci95': None if not valid.any() or invalid / draws > invalid_share_max else q(conditional)}


def estimate(ledger_decisions, pas, *, draws, seed, invalid_share_max, minimum, ess_gate=None,
             censoring='worst_case', natural=H_K_REFUSALS, gap_delta=None, exclude=frozenset()):
    """Aggregate the ledger into per-PA values, layered bounds, bootstrap, ESS and diagnostics.

    ``ledger_decisions``: decision rows in ledger order (``Ledger.decisions()``).
    ``pas``: {pa_id: facts} for EVERY PA of the window: ``game``, ``in_population`` and
    ``start_reason`` (E0), ``problem``/``problem_index`` (structural defect), ``reward``/
    ``reason``/``kind`` (PA end), optional ``first_pitcher_change_index``, ``end_kind``,
    ``flags`` and ``strata``. A ledger PA missing from ``pas`` is an integrity failure; so is any
    FAILED_* decision (a halted ledger is never aggregated). ``censoring``, ``natural``,
    ``gap_delta``, ``exclude``: see the module docstring; the defaults reproduce the registered D-5.
    """
    by_pa = {}
    for row in ledger_decisions:
        by_pa.setdefault(row['pa_id'], []).append(row)
    _require(set(by_pa) <= set(pas), 'every submitted PA needs its facts')
    rows, secondary = [], []
    for pa_id, info in pas.items():
        decisions = sorted(by_pa.get(pa_id, []), key=lambda d: d['decision_index'])
        if pa_id in exclude:  # S-B: pre-decision exclusion from E0 (sensitivity only)
            _require(censoring == 'l1r', 'E0 exclusions are an L1-R sensitivity')
            info = {**info, 'in_population': False, 'start_reason': 'revealed_new_pitch_before_pa'}
        row = pa_value(decisions, info, censoring, natural=natural, gap_delta=gap_delta)
        if row['status'] == 'FAILED':
            raise IntegrityError(f'FAILED decision in PA {pa_id}: {row["reason"]}; a halted ledger is not estimated')
        row.update(pa_id=pa_id, end_kind=info.get('end_kind'), flags=list(info.get('flags') or ()),
                   strata=dict(info.get('strata') or {}))
        rows.append(row)
        secondary.append({**secondary_value(decisions, info, row), 'pa_id': pa_id})
    e0 = [r for r in rows if r['in_population']]
    complete = [r for r in e0 if r.get('l2', r['status'] == COMPLETE)]
    result = {'estimand': ESTIMAND, 'pas': len(rows), 'status': dict(Counter(r['status'] for r in rows)),
              'reasons': dict(Counter(f'{r["status"]}:{r["reason"]}' for r in rows if r['status'] != COMPLETE)),
              'censor_kinds': dict(Counter(r['kind'] for r in e0 if r['status'] == CENSORED)),
              'censor_nodes': dict(Counter(str(r['node']) for r in e0 if r['status'] == CENSORED)),
              'validity_violation_pas': sum(r['validity_violation'] for r in rows),
              'layers': {'L0_all_pas': _layer(rows), 'L1_start_population': _layer(e0)},
              'population_value': None, 'causal_effect': None}
    if censoring == 'l1r':  # the registered default output keeps its keys
        residual = [r for r in e0 if r.get('resolution') in ('worst_case_residual', 'worst_case_residual_gap')]
        result['censoring_rule'] = {
            'rule': 'L1-R', 'natural_course_refusals': list(natural), 'gap_delta': gap_delta, 'excluded_pas': len(exclude),
            'resolutions': dict(Counter(r['resolution'] for r in e0 if r['status'] == CENSORED)),
            'worst_case_residual_reasons': dict(Counter(f"{r['kind']}:{r['reason']}" for r in residual)),
            'worst_case_residual_share': len(residual) / len(e0) if e0 else None,
            'worst_case_residual_definition': list(WORST_CASE_RESIDUAL),
            'estimand': 'regime: pi until the first H_k-fixed event, then logged behaviour; no verdict on V(pi) itself'}
        result['ess_all_e0'] = {name: {'pa': effective_sample_size([r[f'weight_{name}'] for r in e0]),  # COOP-019 F3
                                       'game': effective_sample_size([r[f'weight_{name}'] for r in e0],
                                                                     [r['game'] for r in e0])} for name in POLICIES}
    if complete:
        deltas = [r['delta'] for r in complete]
        weights = {name: [r[f'weight_{name}'] for r in complete] for name in POLICIES}
        games = [r['game'] for r in complete]
        ess = {name: {'pa': effective_sample_size(w), 'game': effective_sample_size(w, games)}
               for name, w in weights.items()}
        gate = {level: (None if any(ess[name][level] is None for name in POLICIES)
                        else min(ess[name][level] for name in POLICIES)) for level in ('pa', 'game')}
        result['layers']['L2_complete_conditional'] = {
            'pas': len(complete), 'delta_mean': float(np.mean(deltas)),
            'candidate_mean': float(np.mean([r['candidate'] for r in complete])),
            'reference_mean': float(np.mean([r['reference'] for r in complete])),
            'observed_mean_end_value': float(np.mean([pas[r['pa_id']]['reward'] for r in complete
                                                      if pas[r['pa_id']].get('reward') is not None] or [np.nan])),
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
        censored = [r for r in e0 if r['status'] == CENSORED]
        result['censoring'] = {'observed_share': len(censored) / len(e0), **{
            f'estimated_mass_{name}': sum(r[f'weight_{name}'] for r in censored) / len(e0) for name in POLICIES},
            'ess_of_censored_weights': {name: {
                'pa': effective_sample_size([r[f'weight_{name}'] for r in censored]) if censored else None,
                'game': effective_sample_size([r[f'weight_{name}'] for r in censored],
                                              [r['game'] for r in censored]) if censored else None} for name in POLICIES}}
        result['bootstrap'] = game_bootstrap(e0, draws=draws, seed=seed, invalid_share_max=invalid_share_max,
                                             minimum=minimum)
    changed = [s for s in secondary if s['in_population'] and s.get('secondary_change') is not None]
    same = {r['pa_id'] for r in complete}

    def with_ess(part):
        layer = _layer(part)
        points = [r for r in part if r['status'] == COMPLETE]
        if layer is not None:
            values = [effective_sample_size([r[f'weight_{n}'] for r in points]) for n in POLICIES] if points else [None]
            layer['ess_min_candidate_reference'] = None if None in values else min(values)
        return layer
    result['secondary_natural_course_after_pitcher_change'] = {
        'label': 'descriptive; not in the primary family',
        'changed_pas': len(changed),
        'all_secondary_evaluable': with_ess([s for s in secondary if s['in_population']]),
        'primary_complete_set': with_ess([s for s in secondary if s['pa_id'] in same])}
    result['strata'] = strata_table(rows)
    return result, rows


def strata_table(rows):
    """Descriptive L3 per registered stratum value: PAs, share inside E0, completes, mean delta
    bounds and min(candidate, reference) ESS over the complete PAs."""
    out = {}
    keys = sorted({k for r in rows for k in r['strata']} | {'end_kind'})
    for key in keys:
        groups = {}
        for r in rows:
            value = r['end_kind'] if key == 'end_kind' else r['strata'].get(key)
            groups.setdefault(str(value), []).append(r)
        table = {}
        for value, part in sorted(groups.items()):
            complete = [r for r in part if r['status'] == COMPLETE]
            ess = [effective_sample_size([r[f'weight_{n}'] for r in complete]) for n in POLICIES] if complete else [None]
            inside = _layer([r for r in part if r['in_population']])
            residuals = [r['terminal_residual'] for r in complete if r.get('terminal_residual') is not None]
            table[value] = {'pas': len(part), 'share_in_population': sum(r['in_population'] for r in part) / len(part),
                            'complete': len(complete), 'l0_delta_bounds': _layer(part)['delta_bounds'],
                            'l1_delta_bounds': None if inside is None else inside['delta_bounds'],
                            'ess_min_candidate_reference': None if None in ess else min(ess),
                            'mean_terminal_residual': float(np.mean(residuals)) if residuals else None}
        out[key] = table
    return out


def revealed_new_pitch(ledger_decisions, pas):
    """S-B (sensitivity only): PAs whose first-row pitcher already threw an action outside his TRAIN
    BC support (a LOGGING_POSITIVITY row) in a strictly earlier PA -- an earlier game date, or the
    same game with a smaller at-bat number. Pre-decision information only; the order (game_date,
    game_pk, at_bat_number) comes from ``pas[pa]['date']`` and the ``game:at_bat`` PA id."""
    first, events = {}, {}
    for row in ledger_decisions:
        if row['decision_index'] == 0:
            first[row['pa_id']] = row['pitcher']
        if row['status'] == POSITIVITY:
            events.setdefault(row['pitcher'], []).append(row['pa_id'])

    def key(pa_id):
        game, ab = (int(x) for x in str(pa_id).split(':'))
        return str(pas[pa_id]['date']), game, ab
    out = set()
    for pa_id, pitcher in first.items():
        date, game, ab = key(pa_id)
        if any(d < date or (g == game and b < ab) for d, g, b in map(key, events.get(pitcher, ()))):
            out.add(pa_id)
    return frozenset(out)


LABELS = ('FAILED_INTEGRITY', 'NOT_DECIDABLE_CENSORING', 'UNCONFIRMED_WEAK_OVERLAP', 'IMPROVEMENT_SUPPORTED',
          'IMPROVEMENT_SUPPORTED_STATISTICAL', 'HARM_SUPPORTED', 'NO_EVIDENCE_OF_IMPROVEMENT')


def decide(result, *, mei, b_v, gr_cap, invalid_share_max, pair_pass):
    """D108/D109 rule restated on L1-R (COOP-021/022): one label from the L1 Delta_lo / Delta_hi
    endpoint CIs, in the order integrity -> G-R -> overlap -> improvement tiers -> harm. Beside the
    G-R share s the approximate passable effect is printed: the worst-case residual adds a width of
    about 2 s, so a pass needs roughly Delta > MEI (+ b_V) + s + 1.96 SE."""
    boot = result.get('bootstrap') or {}
    share = (result.get('censoring_rule') or {}).get('worst_case_residual_share')
    lo, hi = boot.get('L1_lower_endpoint_ci95'), boot.get('L1_upper_endpoint_ci95')
    invalid = boot.get('L2_invalid_replicates')
    out = {'mei': mei, 'b_v': b_v, 'gr_cap': gr_cap, 'worst_case_residual_share': share,
           'passable_effect_approx': None if share is None else
           f'Delta > {mei} (+ b_V {b_v}) + s {share:.6f} + 1.96 SE; worst-case residual width ~ 2 s = {2 * share:.6f}',
           'delta_lo_ci95': lo, 'delta_hi_ci95': hi}
    if not pair_pass:
        return {**out, 'label': 'FAILED_INTEGRITY'}
    if share is None or share > gr_cap:
        return {**out, 'label': 'NOT_DECIDABLE_CENSORING'}
    overlap = ((result.get('ess') or {}).get('label') is None and lo is not None and hi is not None
               and invalid is not None and invalid / boot['draws'] <= invalid_share_max)
    if not overlap:
        return {**out, 'label': 'UNCONFIRMED_WEAK_OVERLAP'}
    if lo[0] > mei + b_v:
        label = 'IMPROVEMENT_SUPPORTED'
    elif lo[0] > mei:
        label = 'IMPROVEMENT_SUPPORTED_STATISTICAL'
    elif hi[1] < 0:
        label = 'HARM_SUPPORTED'
    else:
        label = 'NO_EVIDENCE_OF_IMPROVEMENT'
    return {**out, 'label': label}


def v2_bias_record(v2, mei, v3_law='tempered_alpha_0.5'):
    """b_V = max over the V2 seeds of |gap| + 1.96 se, and the V3 misspecification flag (max |gap| of
    the registered V3 law above MEI), from the sealed S5 v2.json (COOP-020 B1)."""
    v2_runs = [r for r in v2['runs'] if r['check'] == 'V2']
    v3_runs = [r for r in v2['runs'] if r['check'] == 'V3' and r['law'] == v3_law]
    _require(v2_runs and v3_runs, 'sealed V2/V3 runs required for b_V and the V3 flag')
    v3 = max(abs(r['delta_gap']) for r in v3_runs)
    return {'b_v': max(abs(r['delta_gap']) + 1.96 * r['delta_gap_se'] for r in v2_runs), 'v3_law': v3_law,
            'v3_max_abs_gap': v3, 'v3_flag': 'pi_b_misspecification_sensitivity_exceeds_MEI' if v3 > mei else None}
