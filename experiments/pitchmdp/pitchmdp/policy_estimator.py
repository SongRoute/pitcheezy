"""ARM-B per-decision sequential DR over ledger rows (ML-POLICY-ESTIMATOR-v1, COOP-018).

Implements the D89 §5 formula on the rows ``PolicyRuntime`` records: for pi in {candidate,
reference}, ``V_T = 0`` and ``V_t = v(H_t) + rho_t (r_t + V_{t+1} - q(H_t, a_t))`` with the
reward only at the PA's last decision, ``rho_t = pi(a_t|H_t) / pi_b_hat(a_t|H_t)`` (full-vocabulary
logging law, never renormalised), no clipping and no self-normalisation. ``q`` is the recorded
reference-continuation Q; for the candidate it is the declared control variate (D89 §5), so
candidate unbiasedness rests on the logging-law path alone. Every input is re-validated against
the ledger values (fail closed). PAs that are not evaluable keep their place in the denominator
and enter only the per-policy [0, 1] worst-case bound. Nothing here identifies a causal effect.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .policy_artifacts import IntegrityError, _require

EVALUATED = ('SUPPORTED', 'OUTSIDE_POLICY_SUPPORT')
FATAL = ('FAILED_INTEGRITY', 'FAILED_RUNTIME')
POLICIES = ('candidate', 'reference')
COMPLETE = 'COMPLETE'
INCOMPLETE_NO_TERMINAL = 'INCOMPLETE_NO_TERMINAL'
UNSUPPORTED = 'UNSUPPORTED'
FAILED = 'FAILED'
ESTIMAND = ('evaluable-PA conditional mean of V(candidate) - V(reference) in frozen C0 initial-defender WE; '
            'population value only as worst-case bounds')


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


def pa_dr(decisions, reward):
    """Sequential DR values of one evaluable PA from its ordered decision rows and terminal WE."""
    _require(isinstance(reward, float) and np.isfinite(reward) and 0 <= reward <= 1, 'terminal WE outside [0, 1]')
    _require(decisions and [d['decision_index'] for d in decisions] == list(range(len(decisions))),
             'decision rows must be the complete ordered PA')
    _require(len({d['pa_id'] for d in decisions}) == 1, 'decision rows from more than one PA')
    n = len(decisions[0]['result']['logging'])
    steps = [_step(d, n) for d in decisions]
    out = {}
    for name in POLICIES:
        value, weight = 0., 1.
        for t in reversed(range(len(steps))):
            policies, q, mask, a, rho = steps[t]
            p = policies[name]
            r = reward if t == len(steps) - 1 else 0.
            value = float(p[mask] @ q[mask]) + (0. if rho[name] == 0 else rho[name] * (r + value - q[a]))
            weight *= rho[name]
        out[name], out[f'weight_{name}'] = value, weight
    out['delta'] = out['candidate'] - out['reference']
    return out


def pa_status(decisions, outcome):
    """Ledger decision rows + PA outcome -> one PA status (the refusal reason is kept separately)."""
    statuses = [d['status'] for d in decisions]
    if any(s in FATAL for s in statuses):
        return FAILED, next(s for s in statuses if s in FATAL)
    refused = [s for s in statuses if s not in EVALUATED]
    if refused:
        return UNSUPPORTED, refused[0]
    if outcome.get('reward') is None:
        return INCOMPLETE_NO_TERMINAL, outcome.get('reason')
    return COMPLETE, None


def game_bootstrap(deltas, games, *, draws, seed):
    """PA-weighted whole-game percentile bootstrap of the mean paired delta (fixed nuisances)."""
    deltas, games = np.asarray(deltas, dtype=np.float64), np.asarray(games)
    _require(deltas.ndim == 1 and deltas.shape == games.shape and np.isfinite(deltas).all(), 'bootstrap inputs')
    _require(type(draws) is int and draws >= 1 and type(seed) is int, 'registered draws/seed required')
    unique, inverse = np.unique(games, return_inverse=True)
    if len(unique) < 2:
        return {'games': int(len(unique)), 'ci95': None, 'draws': draws, 'seed': seed}
    sums, counts = np.bincount(inverse, weights=deltas), np.bincount(inverse)
    sample = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique)))
    boot = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
    return {'games': int(len(unique)), 'ci95': np.quantile(boot, [.025, .975]).tolist(), 'draws': draws, 'seed': seed}


def effective_sample_size(weights, games=None):
    """(sum w)^2 / sum w^2 over PAs, or over per-game weight sums when ``games`` is given."""
    w = np.asarray(weights, dtype=np.float64)
    _require(w.ndim == 1 and np.isfinite(w).all() and (w >= 0).all(), 'importance weights invalid')
    if games is not None:
        _, inverse = np.unique(np.asarray(games), return_inverse=True)
        w = np.bincount(inverse, weights=w)
    return None if not (w ** 2).sum() else float(w.sum() ** 2 / (w ** 2).sum())


def estimate(ledger_decisions, outcomes, *, draws, seed):
    """Aggregate the ledger into per-PA DR values, status denominators, bounds and diagnostics.

    ``ledger_decisions``: decision rows in ledger order (``Ledger.decisions()``).
    ``outcomes``: {pa_id: {'game': ..., 'reward': float | None, 'reason': ...}} for every PA
    submitted (an outcome missing for a submitted PA is an integrity failure).
    """
    by_pa = {}
    for row in ledger_decisions:
        by_pa.setdefault(row['pa_id'], []).append(row)
    _require(set(by_pa) == set(outcomes), 'every submitted PA needs exactly one outcome record')
    rows, reasons = [], Counter()
    for pa_id, decisions in by_pa.items():
        decisions = sorted(decisions, key=lambda d: d['decision_index'])
        status, reason = pa_status(decisions, outcomes[pa_id])
        row = {'pa_id': pa_id, 'game': outcomes[pa_id]['game'], 'status': status}
        if status == COMPLETE:
            row.update(pa_dr(decisions, outcomes[pa_id]['reward']))
        else:
            reasons[f'{status}:{reason}'] += 1
        rows.append(row)
    complete = [r for r in rows if r['status'] == COMPLETE]
    n, others = len(rows), len(rows) - len(complete)
    delta_sum = sum(r['delta'] for r in complete)
    result = {'estimand': ESTIMAND, 'pas': n, 'status': dict(Counter(r['status'] for r in rows)),
              'non_evaluable_reasons': dict(reasons), 'evaluable_pas': len(complete),
              'population_delta_bounds': [(delta_sum - others) / n, (delta_sum + others) / n] if n else None,
              'population_value': None, 'causal_effect': None}
    if complete:
        deltas = [r['delta'] for r in complete]
        games = [r['game'] for r in complete]
        result['conditional'] = {
            'delta_mean': float(np.mean(deltas)), 'delta_pp': 100 * float(np.mean(deltas)),
            'candidate_mean': float(np.mean([r['candidate'] for r in complete])),
            'reference_mean': float(np.mean([r['reference'] for r in complete])),
            'bootstrap': game_bootstrap(deltas, games, draws=draws, seed=seed),
            'ess': {name: {'pa': effective_sample_size([r[f'weight_{name}'] for r in complete]),
                           'game': effective_sample_size([r[f'weight_{name}'] for r in complete], games)}
                    for name in POLICIES}}
    else:
        result['conditional'] = None
    return result, rows
