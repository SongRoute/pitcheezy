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


def _step(row, n):
    """Validated arrays of one evaluated decision row."""
    result = row.get('result')
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


def _steps(decisions):
    _require([d['decision_index'] for d in decisions] == list(range(len(decisions))),
             'decision rows must be the ordered PA prefix')
    _require(len({d['pa_id'] for d in decisions}) <= 1, 'decision rows from more than one PA')
    if not decisions:
        return []
    n = len(decisions[0]['result']['logging'])
    return [_step(d, n) for d in decisions]


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


def pa_value(decisions, info):
    """Per-PA row: status, node, bounds or point values (and weights) for both policies."""
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
        row.update({f'{name}_bounds': [values[name]] * 2 for name in POLICIES}, delta_bounds=[values['delta']] * 2)
    elif status == CENSORED:
        row.update(_interval(_steps(decisions[:k]), shared=kind == TERMINAL_VALUE_MISSING))
    return row


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
        reward = float(info['reward'])
        row = {'status': COMPLETE}
        for name in POLICIES:
            base, slope = affine(steps, name)
            row[name], row[f'weight_{name}'] = base + slope * reward, slope
            row[f'{name}_bounds'] = [row[name]] * 2
        row['delta'] = row['candidate'] - row['reference']
        row['delta_bounds'] = [row['delta']] * 2
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
    complete = np.array([r['status'] == COMPLETE for r in rows])
    per = lambda values: np.bincount(g, weights=np.asarray(values, dtype=np.float64), minlength=len(games))
    n_all, n_c = per(np.ones(len(rows))), per(complete)
    lo, hi = per([r['delta_bounds'][0] for r in rows]), per([r['delta_bounds'][1] for r in rows])
    d = per([r['delta'] if r['status'] == COMPLETE else 0. for r in rows])
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


def estimate(ledger_decisions, pas, *, draws, seed, invalid_share_max, minimum, ess_gate=None):
    """Aggregate the ledger into per-PA values, layered bounds, bootstrap, ESS and diagnostics.

    ``ledger_decisions``: decision rows in ledger order (``Ledger.decisions()``).
    ``pas``: {pa_id: facts} for EVERY PA of the window: ``game``, ``in_population`` and
    ``start_reason`` (E0), ``problem``/``problem_index`` (structural defect), ``reward``/
    ``reason``/``kind`` (PA end), optional ``first_pitcher_change_index``, ``end_kind``,
    ``flags`` and ``strata``. A ledger PA missing from ``pas`` is an integrity failure; so is any
    FAILED_* decision (a halted ledger is never aggregated).
    """
    by_pa = {}
    for row in ledger_decisions:
        by_pa.setdefault(row['pa_id'], []).append(row)
    _require(set(by_pa) <= set(pas), 'every submitted PA needs its facts')
    rows, secondary = [], []
    for pa_id, info in pas.items():
        decisions = sorted(by_pa.get(pa_id, []), key=lambda d: d['decision_index'])
        row = pa_value(decisions, info)
        if row['status'] == 'FAILED':
            raise IntegrityError(f'FAILED decision in PA {pa_id}: {row["reason"]}; a halted ledger is not estimated')
        row.update(pa_id=pa_id, end_kind=info.get('end_kind'), flags=list(info.get('flags') or ()),
                   strata=dict(info.get('strata') or {}))
        rows.append(row)
        secondary.append({**secondary_value(decisions, info, row), 'pa_id': pa_id})
    e0 = [r for r in rows if r['in_population']]
    complete = [r for r in e0 if r['status'] == COMPLETE]
    result = {'estimand': ESTIMAND, 'pas': len(rows), 'status': dict(Counter(r['status'] for r in rows)),
              'reasons': dict(Counter(f'{r["status"]}:{r["reason"]}' for r in rows if r['status'] != COMPLETE)),
              'censor_kinds': dict(Counter(r['kind'] for r in e0 if r['status'] == CENSORED)),
              'censor_nodes': dict(Counter(str(r['node']) for r in e0 if r['status'] == CENSORED)),
              'validity_violation_pas': sum(r['validity_violation'] for r in rows),
              'layers': {'L0_all_pas': _layer(rows), 'L1_start_population': _layer(e0)},
              'population_value': None, 'causal_effect': None}
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
            'observed_mean_end_value': float(np.mean([pas[r['pa_id']]['reward'] for r in complete])),
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
            f'estimated_mass_{name}': sum(r[f'weight_{name}'] for r in censored) / len(e0) for name in POLICIES}}
        result['bootstrap'] = game_bootstrap(e0, draws=draws, seed=seed, invalid_share_max=invalid_share_max,
                                             minimum=minimum)
    changed = [s for s in secondary if s['in_population'] and s.get('secondary_change') is not None]
    same = {r['pa_id'] for r in complete}
    result['secondary_natural_course_after_pitcher_change'] = {
        'label': 'descriptive; not in the primary family',
        'changed_pas': len(changed),
        'all_secondary_evaluable': _layer([s for s in secondary if s['in_population']]),
        'primary_complete_set': _layer([s for s in secondary if s['pa_id'] in same])}
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
            table[value] = {'pas': len(part), 'share_in_population': sum(r['in_population'] for r in part) / len(part),
                            'complete': len(complete), 'delta_bounds': _layer(part)['delta_bounds'],
                            'ess_min_candidate_reference': None if None in ess else min(ess)}
        out[key] = table
    return out
