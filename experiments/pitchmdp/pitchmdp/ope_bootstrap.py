"""EXP-OPE-RES-01 R4 cluster bootstraps on the point set P (game and first-row pitcher).

The resampling units are the clusters of the E0 PAs, ordered as ``policy_estimator.game_bootstrap``
orders games (sorted by str) and drawn with ``default_rng(seed).integers(G, size=(draws, G))``, so the
game bootstrap resamples exactly the games the registered S6 bootstrap resampled (seed 3060201743);
the pitcher bootstrap uses its own registered seed. A replicate is the multiplicity vector m of the
clusters; every pair is evaluated on the same replicates (paired). For the self-normalised pair the
normalisers n_bar_t are recomputed in every replicate from the resampled E0 PAs (never fixed at the
full-sample value). Validity (design 5.3): fewer clusters than the minimum (30 games / 30 pitchers)
or fewer than 50 E0 PA starts or P PAs -> no interval; an invalid replicate (empty resampled P, a zero
or nonfinite normaliser, a nonfinite statistic) counts against the 0.05 invalid share, above which
the interval is null; a nonfinite endpoint or lower > upper makes that CI invalid.
"""
from __future__ import annotations

import math

import numpy as np

from .ope_redesign_registry import CI_LEVELS, INVALID_SHARE_MAX, MIN_PA_STARTS
from .ope_variants import POINT_ENDS, sn_coefficients, weight_paths
from .policy_artifacts import _require


class Resampler:
    """Cluster multiplicities of one registered bootstrap over the E0 PAs."""
    def __init__(self, keys, *, unit, draws, seed, minimum_clusters, minimum_pa_starts=MIN_PA_STARTS,
                 invalid_share_max=INVALID_SHARE_MAX):
        _require(type(draws) is int and draws >= 1 and type(seed) is int, 'registered draws/seed required')
        self.unit, self.draws, self.seed = unit, draws, seed
        self.order = sorted(set(keys), key=str)
        index = {k: i for i, k in enumerate(self.order)}
        self.row_cluster = np.array([index[k] for k in keys], dtype=np.int64)
        self.n_clusters, self.n_rows = len(self.order), len(keys)
        self.minimum = {'clusters': minimum_clusters, 'pa_starts': minimum_pa_starts}
        self.invalid_share_max = invalid_share_max
        self.eligible = self.n_clusters >= minimum_clusters and self.n_rows >= minimum_pa_starts
        self._m = None

    @property
    def m(self):
        if self._m is None:
            sample = np.random.default_rng(self.seed).integers(self.n_clusters, size=(self.draws, self.n_clusters))
            flat = (np.arange(self.draws)[:, None] * self.n_clusters + sample).ravel()
            self._m = np.bincount(flat, minlength=self.draws * self.n_clusters).reshape(
                self.draws, self.n_clusters).astype(np.float64)
        return self._m

    def sums(self, values, rows=None):
        """Per-cluster sums of per-row values ([n] or [n, K]) over ``rows`` (indices into the E0 rows)."""
        rows = np.arange(self.n_rows) if rows is None else np.asarray(rows, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        cluster = self.row_cluster[rows]
        if values.ndim == 1:
            return np.bincount(cluster, weights=values, minlength=self.n_clusters)
        out = np.zeros((self.n_clusters, values.shape[1]))
        np.add.at(out, cluster, values)
        return out

    def record(self):
        return {'unit': self.unit, 'draws': self.draws, 'seed': self.seed, 'clusters': self.n_clusters,
                'pa_starts': self.n_rows, 'minimum': self.minimum}


def interval(replicates, valid, draws, invalid_share_max=INVALID_SHARE_MAX):
    """Percentile 95% CI, half-width and SE (SD of the valid replicates, ddof 1) of one statistic."""
    replicates = np.asarray(replicates, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(replicates)
    invalid = int(draws - ok.sum())
    out = {'invalid_replicates': invalid, 'ci95': None, 'half_width': None, 'se': None, 'valid': False}
    if invalid / draws > invalid_share_max or ok.sum() < 2:
        return {**out, 'reason': 'invalid replicate share above the registered maximum'}
    lo, hi = (float(v) for v in np.quantile(replicates[ok], CI_LEVELS))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        return {**out, 'reason': 'nonfinite or reversed CI'}
    return {**out, 'ci95': [lo, hi], 'half_width': (hi - lo) / 2, 'se': float(np.std(replicates[ok], ddof=1)),
            'valid': True, 'reason': None}


class PairInputs:
    """Everything a pair contributes to the bootstrap: per-E0-row point deltas (P rows), endpoint
    bounds (all E0 rows), or for the SN pair the linear coefficients in the inverse normalisers."""
    def __init__(self, name, *, deltas=None, lo=None, hi=None, sn=None):
        self.name, self.deltas, self.lo, self.hi, self.sn = name, deltas, lo, hi, sn


def fixed_pair(name, rows, point):
    """A non-SN pair from its E0 rows (pa_value-style dicts, E0 order) and the P membership mask."""
    lo = np.array([r['delta_bounds'][0] for r in rows], dtype=np.float64)
    hi = np.array([r['delta_bounds'][1] for r in rows], dtype=np.float64)
    deltas = np.where(point, lo, np.nan)
    _require(np.array_equal(lo[point], hi[point]), f'{name}: a P row is not a point')
    return PairInputs(name, deltas=deltas, lo=lo, hi=hi)


def sn_pair(name, lines, point, rewards=None):
    """The SN pair: per E0 line, per policy, base coefficients over depth and the slope coefficient
    (W_T at depth T) so that V = coef . u (+ W_T u_T c); rewards override observed ends (R1 arms)."""
    paths = weight_paths(lines)
    depth = max(len(p[0]) for p in paths) - 1
    n = len(lines)
    held = np.zeros((2, n, depth + 1))
    coef = np.zeros((2, n, depth + 1))
    slope = np.zeros((2, n))
    end = np.zeros(n, dtype=np.int64)
    for i, (line, path) in enumerate(zip(lines, paths)):
        for which in (0, 1):
            w = np.asarray(path[which], dtype=np.float64)
            held[which, i, :len(w)] = w
            held[which, i, len(w):] = w[-1]
            c, s, T = sn_coefficients(line, which, depth)
            coef[which, i], slope[which, i], end[i] = c, s, T
    known = np.array([line['end'] in POINT_ENDS for line in lines])
    reward = np.array([(line['reward'] if rewards is None or line['reward_source'] != 'observed' else rewards[i])
                       if known[i] else 0. for i, line in enumerate(lines)], dtype=np.float64)
    full = coef.copy()
    full[:, np.arange(n), end] += slope * reward[None]
    kinds = np.array([line['end'] for line in lines], dtype=object)
    _require(np.array_equal(known, point), f'{name}: P differs from the point lines')
    return PairInputs(name, sn={'held': held, 'coef': coef, 'full': full, 'slope': slope, 'end': end,
                                'point': known, 'kinds': kinds, 'depth': depth})


def sn_values(pair, u):
    """Full-sample SN per-row values for inverse normalisers u [2, D+1]: (deltas on P, lo, hi)."""
    s = pair.sn
    point_values = np.einsum('pnd,pd->pn', s['full'], u)
    deltas = np.where(s['point'], point_values[0] - point_values[1], np.nan)
    base = np.einsum('pnd,pd->pn', s['coef'], u)
    slope = s['slope'] * u[np.arange(2)[:, None], s['end'][None]]
    lo, hi = _bounds(base, slope, s['kinds'], s['point'], deltas)
    return deltas, lo, hi


def _bounds(base, slope, kinds, point, deltas):
    bc, br, sc, sr = base[0], base[1], slope[0], slope[1]
    shared_a, shared_b = bc - br, bc + sc - br - sr
    lo = np.where(kinds == 'interval_shared', np.minimum(shared_a, shared_b), bc - br - sr)
    hi = np.where(kinds == 'interval_shared', np.maximum(shared_a, shared_b), bc + sc - br)
    lo = np.where(point, deltas, lo)
    hi = np.where(point, deltas, hi)
    return lo, hi


def run_pairs(resampler, pairs, point, baseline_name, *, chunk=500):
    """Replicates of every pair on the same multiplicities: the P mean, the L1-R endpoints, the
    paired shift to the baseline on P, and the point/censored split of the lower endpoint over the
    common denominator N (R5 f). Returns {pair: {statistic: replicates}}, validity per replicate."""
    point = np.asarray(point, dtype=bool)
    draws = resampler.draws
    p_rows = np.flatnonzero(point)
    n_p = resampler.sums(np.ones(len(p_rows)), p_rows)
    n_all = resampler.sums(np.ones(resampler.n_rows))
    m = resampler.m
    den_p, den_all = m @ n_p, m @ n_all
    out, valid = {}, {}
    base = next(p for p in pairs if p.name == baseline_name)
    _require(base.sn is None, 'the baseline is the fixed (non-SN) pair')
    base_num = m @ resampler.sums(base.deltas[p_rows], p_rows)
    for pair in pairs:
        if pair.sn is None:
            num = m @ resampler.sums(pair.deltas[p_rows], p_rows)
            lo_num = m @ resampler.sums(pair.lo)
            hi_num = m @ resampler.sums(pair.hi)
            cens_rows = np.flatnonzero(~point)
            cens_num = m @ resampler.sums(pair.lo[cens_rows], cens_rows) if len(cens_rows) else np.zeros(draws)
            ok = (den_p > 0) & (den_all > 0)
        else:
            num, lo_num, hi_num, cens_num, ok = _sn_replicates(resampler, pair, point, den_all, chunk)
            ok &= den_p > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            stats = {'mean_p': num / den_p, 'shift_p': (num - base_num) / den_p, 'lower_endpoint': lo_num / den_all,
                     'upper_endpoint': hi_num / den_all, 'point_part_n': num / den_all,
                     'censored_part_n': cens_num / den_all}
        good = ok & np.all([np.isfinite(v) for v in stats.values()], axis=0)
        out[pair.name], valid[pair.name] = stats, good
    return out, valid


def _sn_replicates(resampler, pair, point, den_all, chunk):
    s = pair.sn
    m = resampler.m
    held = [resampler.sums(s['held'][w]) for w in (0, 1)]  # [G, D+1]
    n_bar = [m @ h for h in held]
    with np.errstate(divide='ignore', invalid='ignore'):
        n_bar = [nb / den_all[:, None] for nb in n_bar]
        u = [1.0 / nb for nb in n_bar]
    ok = (den_all > 0) & np.all([np.isfinite(nb).all(axis=1) & (nb > 0).all(axis=1) for nb in n_bar], axis=0)
    p_rows = np.flatnonzero(point)
    full = [m @ resampler.sums(s['full'][w][p_rows], p_rows) for w in (0, 1)]  # [draws, D+1]
    with np.errstate(invalid='ignore'):
        num = np.sum(u[0] * full[0], axis=1) - np.sum(u[1] * full[1], axis=1)
    cens = np.flatnonzero(~point)
    lo_cens = np.zeros(resampler.draws)
    hi_cens = np.zeros(resampler.draws)
    for i in cens:  # the bounded rows are few; each is evaluated in every replicate
        g = resampler.row_cluster[i]
        with np.errstate(invalid='ignore'):
            base = [u[w] @ s['coef'][w][i] for w in (0, 1)]
            slope = [s['slope'][w][i] * u[w][:, s['end'][i]] for w in (0, 1)]
        bc, br, sc, sr = base[0], base[1], slope[0], slope[1]
        if s['kinds'][i] == 'interval_shared':
            a, b = bc - br, bc + sc - br - sr
            lo_i, hi_i = np.minimum(a, b), np.maximum(a, b)
        else:
            lo_i, hi_i = bc - br - sr, bc + sc - br
        lo_cens += m[:, g] * lo_i
        hi_cens += m[:, g] * hi_i
    return num, num + lo_cens, num + hi_cens, lo_cens, ok


def summarize(stats, valid, resampler):
    """Public-shaped summary per pair (widths, SEs, variances only) and the sealed centers/CIs."""
    draws = resampler.draws
    if not resampler.eligible:
        return {'eligible': False, 'reason': 'below the registered minimum clusters / PA starts', **resampler.record()}
    out = {'eligible': True, **resampler.record()}
    for name, s in stats.items():
        ok = valid[name]
        point, shift = interval(s['mean_p'], ok, draws), interval(s['shift_p'], ok, draws)
        lower, upper = interval(s['lower_endpoint'], ok, draws), interval(s['upper_endpoint'], ok, draws)
        good = ok & np.isfinite(s['point_part_n']) & np.isfinite(s['censored_part_n'])
        if good.sum() >= 2 and (draws - good.sum()) / draws <= INVALID_SHARE_MAX:
            cov = np.cov(s['point_part_n'][good], s['censored_part_n'][good], ddof=1)
            decomposition = {'var_point': float(cov[0, 0]), 'var_censored': float(cov[1, 1]),
                             'cov_point_censored': float(cov[0, 1]),
                             'var_lower_endpoint': float(np.var(s['lower_endpoint'][good], ddof=1)), 'valid': True}
        else:
            decomposition = {'valid': False}
        out[name] = {'mean_p': point, 'shift_p': shift, 'lower_endpoint': lower, 'upper_endpoint': upper,
                     'decomposition': decomposition}
    return out


def pigeonhole(game_keys, pitcher_keys, pairs, point, *, draws, seed, chunk=200):
    """Descriptive two-way (game x pitcher) reweighting bootstrap (Owen 2007, unverified): PA weight =
    game multiplicity x pitcher multiplicity; the SN normalisers are recomputed with the same weights.
    Sealed only (not in the public allowlist)."""
    point = np.asarray(point, dtype=bool)
    games, g_idx = np.unique(np.asarray([str(k) for k in game_keys]), return_inverse=True)
    pitchers, p_idx = np.unique(np.asarray([str(k) for k in pitcher_keys]), return_inverse=True)
    rng = np.random.default_rng(seed)
    results = {p.name: [] for p in pairs}
    done = 0
    while done < draws:
        n = min(chunk, draws - done)
        mg = _mult(rng.integers(len(games), size=(n, len(games))), len(games))
        mp = _mult(rng.integers(len(pitchers), size=(n, len(pitchers))), len(pitchers))
        w = mg[:, g_idx] * mp[:, p_idx]
        wp = w[:, point]
        den = wp.sum(axis=1)
        for pair in pairs:
            if pair.sn is None:
                num = wp @ pair.deltas[point]
            else:
                s = pair.sn
                tot = w.sum(axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    u = [1.0 / ((w @ s['held'][k]) / tot[:, None]) for k in (0, 1)]
                    num = (np.sum(u[0] * (wp @ s['full'][0][point]), axis=1)
                           - np.sum(u[1] * (wp @ s['full'][1][point]), axis=1))
            with np.errstate(divide='ignore', invalid='ignore'):
                results[pair.name].append(num / den)
        done += n
    return {name: interval(np.concatenate(v), np.ones(draws, dtype=bool), draws) for name, v in results.items()}


def _mult(sample, n):
    rows = sample.shape[0]
    flat = (np.arange(rows)[:, None] * n + sample).ravel()
    return np.bincount(flat, minlength=rows * n).reshape(rows, n).astype(np.float64)
