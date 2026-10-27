"""EXP-OPE-RES-01 R5 pre-diagnostics (reported before any Delta center) and the placebo SE check.

a agreement, b ESS, c weights, d positivity/censoring, g self-consistency use no reward (weights and
policies only); e (MDE) and f (SE decomposition) are widths/variances from ope_bootstrap; h compares
the SD of K = 100 reward-free-constructed placebo Delta_hat with their mean bootstrap SE (a flag,
never a decision input); i reports the R1 surrogate paths and the TRAIN residuals. Individual
placebo Delta_hat values are sealed. Also the vectorized padded DR used by the placebo and TAB
(bit-identical, element by element, to the scalar recursion of ``policy_estimator.affine``).
"""
from __future__ import annotations

import math

import numpy as np

from . import policy_estimator as est
from .ope_redesign_registry import (B_V, MDE_MULTIPLIER, MEI, PLACEBO_FLAG, PLACEBO_KL, PLACEBOS, role_sequence)
from .ope_variants import mean_abs_shrink, transform, trigger_rate, weight_paths
from .policy_artifacts import _require

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


# ---------------------------------------------------------------- vectorized DR (padded)
def padded(per_pa_steps, width):
    """Stack ragged per-PA step arrays [T_i, width] into [n, L, width] (zeros beyond T_i) and T."""
    lengths = np.array([len(s) for s in per_pa_steps], dtype=np.int64)
    out = np.zeros((len(per_pa_steps), max(int(lengths.max(initial=0)), 1), width))
    for i, s in enumerate(per_pa_steps):
        if len(s):
            out[i, :len(s)] = s
    return out, lengths


def affine_padded(v, qa, rho, lengths, u=None):
    """Vectorized (base, slope) of the per-decision recursion for n PAs; u (SN) indexed by depth."""
    n, L = v.shape
    base = np.zeros(n)
    slope = np.ones(n) if u is None else np.full(n, np.nan)
    if u is not None:
        slope = np.asarray(u, dtype=np.float64)[lengths]
    with np.errstate(invalid='ignore'):
        for t in range(L - 1, -1, -1):
            active = t < lengths
            r = rho[:, t]
            if u is None:
                nb = np.where(r == 0, v[:, t], v[:, t] + r * (base - qa[:, t]))
            else:
                nb = np.where(r == 0, u[t] * v[:, t], u[t] * v[:, t] + r * (base - u[t + 1] * qa[:, t]))
            ns = np.where(r == 0, 0., r * slope)
            base = np.where(active, nb, base)
            slope = np.where(active, ns, slope)
    return base, slope


# ---------------------------------------------------------------- decision-level arrays from the ledger
class DecisionTable:
    """Evaluated decision rows of the E0 PAs (both policies' probabilities, mask, logging, q-hat)."""
    def __init__(self, entries):
        self.entries = entries  # list of dicts: pa, pitcher, balls, strikes, mask, logging, cand, ref, q, a
        if entries:
            self.mask = np.array([e['mask'] for e in entries], dtype=bool)
            self.logging = np.array([e['logging'] for e in entries], dtype=np.float64)
            self.cand = np.array([e['candidate'] for e in entries], dtype=np.float64)
            self.ref = np.array([e['reference'] for e in entries], dtype=np.float64)
            self.a = np.array([e['a'] for e in entries], dtype=np.int64)
            self.q = np.array([[np.nan if v is None else v for v in e['q']] for e in entries], dtype=np.float64)
            self.positivity = np.array([e['positivity'] for e in entries], dtype=bool)

    @classmethod
    def from_ledger(cls, decisions, e0_pas):
        entries = []
        for row in decisions:
            usable = row['status'] in est.EVALUATED or row['status'] == est.POSITIVITY
            if row['pa_id'] not in e0_pas or not usable or not isinstance(row.get('result'), dict):
                continue
            r = row['result']
            entries.append({'pa': row['pa_id'], 'pitcher': str(row.get('pitcher')), 'balls': row.get('balls'),
                            'strikes': row.get('strikes'), 'mask': r['mask'], 'logging': r['logging'],
                            'candidate': r['candidate'], 'reference': r['reference'], 'q': r['q_reference'],
                            'a': r['logged_index'], 'index': row['decision_index'],
                            'positivity': row['status'] == est.POSITIVITY})
        return cls(entries)

    def __len__(self):
        return len(self.entries)

    def rho(self, probabilities):
        """pi(a) / pi_b_hat(a) at the logged action; exactly 0 on a LOGGING_POSITIVITY row (as recorded)."""
        rows = np.arange(len(self))
        logged = self.logging[rows, self.a]
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.positivity | (logged == 0), 0., probabilities[rows, self.a] / logged)


def _kl(p, q, mask):
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(mask & (p > 0), p * np.log(p / q), 0.)
    return terms.sum(axis=1)


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {**{f'q{int(round(q * 100)):02d}': float(np.quantile(values, q)) for q in QUANTILES},
            'mean': float(values.mean()), 'max': float(values.max())}


def agreement(table):
    """R5 a: top-1 = logged rates, mean pi(logged), TV/KL(candidate || reference), top-1 agreement."""
    if not len(table):
        return None
    rows = np.arange(len(table))
    top_c, top_r = table.cand.argmax(axis=1), table.ref.argmax(axis=1)
    tv = 0.5 * np.abs(table.cand - table.ref).sum(axis=1)
    kl = _kl(table.cand, table.ref, table.mask)
    dist = lambda v: {k: _quantiles(v)[k] for k in ('mean', 'q50', 'q90', 'q99')}
    return {'decisions': len(table),
            'top1_equals_logged': {'candidate': float((top_c == table.a).mean()), 'reference': float((top_r == table.a).mean())},
            'mean_pi_logged': {'candidate': float(table.cand[rows, table.a].mean()),
                               'reference': float(table.ref[rows, table.a].mean())},
            'tv_candidate_reference': dist(tv), 'kl_candidate_reference': dist(kl),
            'top1_agreement_candidate_reference': float((top_c == top_r).mean())}


def weight_summary(lines):
    """R5 c: rho quantiles, max and zero share per policy over the value-line steps of the E0 PAs."""
    rho = [[step[3 + w] for line in lines for step in line['steps']] for w in (0, 1)]
    out = {}
    for name, values in zip(est.POLICIES, rho):
        values = np.asarray(values, dtype=np.float64)
        out[name] = None if not len(values) else {**_quantiles(values), 'zero_share': float((values == 0).mean())}
    return out


def ess_by_variant(lines, games, grids, sn_inverse=None):
    """R5 b: PA and game ESS of the PA weights (slopes) for every variant and grid value, both
    policies, over all E0 / complete / censored PAs; no label marks the selected value. With
    ``sn_inverse`` (u_c, u_r) the self-normalised weights W_T / n_bar_T are added as 'sn'."""
    out = {}
    complete = np.array([line['status'] == est.COMPLETE for line in lines])
    censored = np.array([line['status'] == est.CENSORED for line in lines])
    games = np.asarray([str(g) for g in games])
    weights = {key: np.array([[p[w_][-1] for w_ in (0, 1)] for p in weight_paths(lines, f)]) for key, f in grids.items()}
    if sn_inverse is not None:
        weights['sn'] = np.array([[p[w_][-1] * sn_inverse[w_][len(p[w_]) - 1] for w_ in (0, 1)]
                                  for p in weight_paths(lines)])
    for key, w in weights.items():
        entry = {}
        for name, col in zip(est.POLICIES, (0, 1)):
            part = {}
            for subset, sel in (('all_e0', np.ones(len(lines), bool)), ('complete', complete), ('censored', censored)):
                values = w[sel, col]
                part[subset] = {'pa': est.effective_sample_size(values) if sel.any() else None,
                                'game': est.effective_sample_size(values, games[sel]) if sel.any() else None}
            entry[name] = part
        out[key] = entry
    return out


def grid_weight_diagnostics(lines, lambda_grid, m_grid):
    """R5 c per grid value: Switch trigger rate per M and mean |rho~ - rho| per lambda."""
    return {'switch_trigger_rate': {str(m): trigger_rate(lines, 'switch', m) for m in m_grid},
            'shrink_mean_abs_change': {str(lam): mean_abs_shrink(lines, lam) for lam in lambda_grid}}


def positivity_censoring(result, rows):
    """R5 d: positivity count, resolution counts, s, per-game worst-case PA distribution (counts only)."""
    rule = result.get('censoring_rule') or {}
    e0 = [r for r in rows if r['in_population']]
    worst = {}
    for r in e0:
        if r.get('resolution') == 'worst_case_residual':
            worst[str(r['game'])] = worst.get(str(r['game']), 0) + 1
    per_game = {}
    for count in worst.values():
        per_game[str(count)] = per_game.get(str(count), 0) + 1
    games_with_e0 = len({str(r['game']) for r in e0})
    per_game['0'] = games_with_e0 - len(worst)
    return {'logging_positivity_pas': int(result.get('validity_violation_pas', 0)),
            'resolutions': dict(rule.get('resolutions') or {}), 'worst_case_residual_share': rule.get('worst_case_residual_share'),
            'worst_case_pas_per_game_distribution': dict(sorted(per_game.items(), key=lambda kv: int(kv[0]))),
            'censored_share': (result.get('censoring') or {}).get('observed_share')}


def logging_mass(table):
    if not len(table):
        return None
    outside = 1 - np.where(table.mask, table.logging, 0.).sum(axis=1)
    target = np.maximum(np.where(~table.mask, table.cand, 0.).sum(axis=1), np.where(~table.mask, table.ref, 0.).sum(axis=1))
    return {'mean_logging_mass_outside_policy_mask': float(outside.mean()),
            'target_mass_outside_policy_mask_max': float(target.max()),
            'logged_outside_policy_mask_share': float((~table.mask[np.arange(len(table)), table.a]).mean())}


def mde(boot_se_point, boot_se_lower, s):
    """R5 e (blinded widths only): MDE_80 on P = 2.801585 SE; current-rule MDE = MEI + s + 2.801585 SE_lo (+ b_V)."""
    point = None if boot_se_point is None else MDE_MULTIPLIER * boot_se_point
    rule = None if boot_se_lower is None or s is None else MEI + s + MDE_MULTIPLIER * boot_se_lower
    return {'mde80_point': point, 'mde80_current_rule': rule,
            'mde80_current_rule_bias_adjusted': None if rule is None else rule + B_V}


def self_consistency(lines, table, resampler):
    """R5 g: mean rho over the logged evaluated decisions and mean PA-end cumulative W per policy (both
    have expectation 1 under a correct pi_b_hat), with game-bootstrap 95% CIs (reward-free)."""
    from .ope_bootstrap import interval
    final = np.array([[p[w][-1] for w in (0, 1)] for p in weight_paths(lines)])
    out = {}
    rows = np.arange(len(table))
    pa_index = {line['pa_id']: i for i, line in enumerate(lines)} if lines and 'pa_id' in lines[0] else None
    for col, name in enumerate(est.POLICIES):
        rho = table.rho(table.cand if name == 'candidate' else table.ref)
        entry = {'mean_rho_logged': float(rho.mean()) if len(rho) else None,
                 'mean_pa_end_weight': float(final[:, col].mean()) if len(final) else None}
        if resampler is not None and resampler.eligible:
            m = resampler.m
            den = m @ resampler.sums(np.ones(resampler.n_rows))
            w_ci = interval((m @ resampler.sums(final[:, col])) / den, den > 0, resampler.draws)
            entry['mean_pa_end_weight_ci95'] = w_ci['ci95']
            if pa_index is not None and len(rho):
                rows_of = np.array([pa_index[e['pa']] for e in table.entries])
                cluster = resampler.row_cluster[rows_of]
                num = m @ np.bincount(cluster, weights=rho, minlength=resampler.n_clusters)
                cnt = m @ np.bincount(cluster, minlength=resampler.n_clusters).astype(np.float64)
                with np.errstate(divide='ignore', invalid='ignore'):
                    entry['mean_rho_logged_ci95'] = interval(num / cnt, cnt > 0, resampler.draws)['ci95']
        out[name] = entry
    return out


# ---------------------------------------------------------------- R5 h placebo
def _bucket(balls, strikes):
    if balls is None or strikes is None:
        return 1
    return 0 if strikes > balls else (1 if strikes == balls else 2)


def placebo_policy(ref, mask, xi, tau):
    with np.errstate(over='ignore'):
        logits = np.where(mask, np.log(np.where(mask, ref, 1.)) + xi / tau, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    w = np.where(mask, np.exp(logits), 0.)
    return w / w.sum(axis=1, keepdims=True)


def tune_tau(ref, mask, xi, target=PLACEBO_KL):
    """tau_p with mean KL(pi_p || pi_r) = target (bisection on log tau; KL decreases in tau)."""
    kl = lambda tau: float(_kl(placebo_policy(ref, mask, xi, tau), ref, mask).mean())
    lo, hi = 1e-4, 1e4
    _require(kl(lo) > target > kl(hi), 'placebo KL target outside the bisection bracket')
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if kl(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1 < 1e-12:
            break
    return math.sqrt(lo * hi)


def placebo_check(point_lines_steps, table, resamplers, *, rewards, placebos=PLACEBOS):
    """K placebo tilts pi_p proportional to pi_r exp(xi / tau_p), xi ~ N(0, 1) fixed per (pitcher,
    pitch type, count bucket) from seed role 'placebo' substream [k]; Delta_hat_k = mean over P of the
    DR difference V(pi_p) - V(pi_r) (same q-hat, rewards and logging law; no planner). Returns the
    public SD ratio per bootstrap and the sealed individual Delta_hat values.

    ``point_lines_steps``: per P line the decision-table row indices of its value-line steps;
    ``resamplers``: {boot: (Resampler over the E0 rows, E0 indices of the P lines)}; ``rewards``: end
    value per P line (0 for a zero-slope line)."""
    keys = sorted({(e['pitcher'], _bucket(e['balls'], e['strikes'])) for e in table.entries})
    key_index = {k: i for i, k in enumerate(keys)}
    row_key = np.array([key_index[(e['pitcher'], _bucket(e['balls'], e['strikes']))] for e in table.entries])
    n_actions = table.mask.shape[1]
    rows = np.arange(len(table))
    v_ref = np.nansum(np.where(table.mask, table.ref * np.nan_to_num(table.q), 0.), axis=1)
    qa = table.q[rows, table.a]
    rho_ref = table.rho(table.ref)
    steps = [np.asarray(s, dtype=np.int64) for s in point_lines_steps]
    lengths = np.array([len(s) for s in steps], dtype=np.int64)
    L = max(int(lengths.max(initial=0)), 1)
    grid = np.full((len(steps), L), -1, dtype=np.int64)
    for i, s in enumerate(steps):
        grid[i, :len(s)] = s
    take = lambda values: np.where(grid >= 0, values[np.maximum(grid, 0)], 0.)
    ends = np.asarray(rewards, dtype=np.float64)
    base_r, slope_r = affine_padded(take(v_ref), take(qa), take(rho_ref), lengths)
    value_r = base_r + slope_r * ends
    deltas, ses, taus = [], {name: [] for name in resamplers}, []
    for k in range(placebos):
        xi = np.random.default_rng(role_sequence('placebo', k)).standard_normal((len(keys), n_actions))[row_key]
        tau = tune_tau(table.ref, table.mask, xi)
        pi_p = placebo_policy(table.ref, table.mask, xi, tau)
        v_p = np.nansum(np.where(table.mask, pi_p * np.nan_to_num(table.q), 0.), axis=1)
        rho_p = table.rho(pi_p)
        base_p, slope_p = affine_padded(take(v_p), take(qa), take(rho_p), lengths)
        delta_i = (base_p + slope_p * ends) - value_r
        deltas.append(float(delta_i.mean()))
        taus.append(tau)
        for name, (resampler, p_rows) in resamplers.items():
            if resampler is None or not resampler.eligible:
                continue
            m = resampler.m
            num = m @ resampler.sums(delta_i, p_rows)
            den = m @ resampler.sums(np.ones(len(p_rows)), p_rows)
            with np.errstate(divide='ignore', invalid='ignore'):
                rep = num / den
            ok = (den > 0) & np.isfinite(rep)
            ses[name].append(float(np.std(rep[ok], ddof=1)) if ok.sum() > 1 else math.nan)
    sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else math.nan
    ratios = {name: (sd / float(np.mean(v)) if v and np.isfinite(v).all() and np.mean(v) > 0 else None)
              for name, v in ses.items()}
    flag = any(r is not None and r > PLACEBO_FLAG for r in ratios.values())
    return ({'placebos': placebos, 'sd_ratio': ratios, 'flag': 'SE_CALIBRATION_WARNING' if flag else None,
             'decision_input': False},
            {'placebo_delta_hat': deltas, 'tau_p': taus, 'placebo_bootstrap_se': ses, 'placebo_sd': sd})


def r1_variance(point_lines, last_q, rewards_by_component):
    """Var(r - q_hat_last) vs Var(r~ - q_hat_last) over complete PAs with the logged action on the mask."""
    out = {}
    for component, rewards in rewards_by_component.items():
        values = [r - q for r, q, line in zip(rewards, last_q, point_lines)
                  if line['status'] == est.COMPLETE and q is not None]
        out[component] = float(np.var(values, ddof=1)) if len(values) > 1 else None
    return out


def standard_grids(lambda_grid, m_grid):
    grids = {'identity': None}
    grids.update({f'shrink:{lam}': transform('shrink', lam) for lam in lambda_grid})
    grids.update({f'switch:{m}': transform('switch', m) for m in m_grid})
    return grids
