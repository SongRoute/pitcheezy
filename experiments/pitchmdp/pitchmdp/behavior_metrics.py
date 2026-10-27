"""18-class probability metrics of EXP-BC-CAL-01 (design section 7), no data access.

Same definitions as ``matrix_metrics`` (which is fixed at 10 classes): natural-log NLL clipped at
1e-12, multiclass Brier summed over classes, top-label ECE over 10 equal-width confidence bins
(bin = min(floor(10 * confidence), 9)). The class count comes from the probability width, so a
10-class input reproduces ``matrix_metrics.prediction_metrics``. Paired candidate-minus-B0
inference is the whole-game bootstrap of ``matrix_metrics.paired_game_comparison`` (same draws,
chunking, null-centered one-sided p); the ECE and near-zero (P1) differences are recomputed
exactly per replicate from per-game sufficient statistics (bin counts, confidence sums and hit
sums; near-zero counts). Intervals condition on fixed predictions.
"""
from __future__ import annotations

import numpy as np

from .matrix_metrics import holm_adjust  # noqa: F401  (re-exported: the registered Holm rule)

NLL_CLIP = 1e-12


def validate(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 2 or len(p) != len(y):
        raise ValueError('Incompatible labels and probability shape')
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError('Labels outside the declared class contract')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and in [0,1]')
    if len(p) and not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError('Probability mass differs from one')
    return y.astype(np.int64, copy=False), p


def pitch_losses(labels, probabilities):
    y, p = validate(labels, probabilities)
    selected = p[np.arange(len(y)), y]
    nll = -np.log(np.clip(selected, NLL_CLIP, 1))
    brier = np.square(p).sum(axis=1) - 2 * selected + 1
    return np.column_stack([nll, brier])


def _bins(confidence, bins):
    return np.minimum((confidence * bins).astype(int), bins - 1)


def top_label_ece(labels, probabilities, bins=10):
    y, p = validate(labels, probabilities)
    if not len(y):
        return None
    confidence, hit = p.max(axis=1), (p.argmax(axis=1) == y).astype(np.float64)
    b = _bins(confidence, bins)
    return float(np.abs(np.bincount(b, hit, bins) - np.bincount(b, confidence, bins)).sum() / len(y))


def classwise_ece(labels, probabilities, bins=10):
    """matrix_metrics class ECE per class: sum_b share_b * |observed_b - mean p_b|."""
    y, p = validate(labels, probabilities)
    out = []
    for c in range(p.shape[1]):
        actual = (y == c).astype(np.float64)
        b = _bins(p[:, c], bins)
        out.append(float(np.abs(np.bincount(b, actual, bins) - np.bincount(b, p[:, c], bins)).sum() / len(y))
                   if len(y) else None)
    return out


def near_zero_share(labels, probabilities, threshold):
    y, p = validate(labels, probabilities)
    return float((p[np.arange(len(y)), y] < threshold).mean()) if len(y) else None


def prediction_summary(labels, probabilities, *, p1, p2, bins=10):
    y, p = validate(labels, probabilities)
    if not len(y):
        return {'n': 0, 'nll': None, 'brier': None, 'top_label_ece10': None, 'accuracy': None, 'p1_share': None,
                'p2_share': None, 'classwise_ece': None}
    losses = pitch_losses(y, p)
    classes = classwise_ece(y, p, bins)
    return {'n': int(len(y)), 'nll': float(losses[:, 0].mean()), 'brier': float(losses[:, 1].mean()),
            'top_label_ece10': top_label_ece(y, p, bins), 'accuracy': float((p.argmax(axis=1) == y).mean()),
            'p1_share': near_zero_share(y, p, p1), 'p2_share': near_zero_share(y, p, p2),
            'classwise_ece': {'per_class': classes, 'mean': float(np.mean(classes)), 'max': float(np.max(classes))}}


def _game_bin_stats(y, p, inverse, games, bins):
    """Per game [games, 3 * bins]: bin counts | confidence sums | hit sums."""
    confidence, hit = p.max(axis=1), (p.argmax(axis=1) == y).astype(np.float64)
    index = inverse * bins + _bins(confidence, bins)
    size = games * bins
    parts = [np.bincount(index, minlength=size), np.bincount(index, confidence, size), np.bincount(index, hit, size)]
    return np.column_stack([part.reshape(games, bins) for part in parts]).astype(np.float64)


def _ece_from(stats, total, bins):
    confidence, hits = stats[..., bins:2 * bins], stats[..., 2 * bins:]
    return np.abs(hits - confidence).sum(axis=-1) / total


def paired_game_bootstrap(labels, candidate, control, game_ids, *, draws, seed, p1_threshold, bins=10):
    """Candidate minus control on the same pitches: NLL/Brier (delta, CI95, null-centered one-sided
    p_less exactly as matrix_metrics.paired_game_comparison), top-label ECE10 and near-zero share P1
    (delta, CI95 recomputed per replicate from per-game sufficient statistics)."""
    if draws < 100:
        raise ValueError('At least 100 bootstrap draws required')
    y, pc = validate(labels, candidate)
    _, p0 = validate(labels, control)
    if pc.shape != p0.shape:
        raise ValueError('candidate and control must cover the same pitches and classes')
    delta = pitch_losses(y, pc) - pitch_losses(y, p0)
    games = np.asarray(game_ids)
    if games.ndim != 1 or len(games) != len(delta):
        raise ValueError('One game ID is required for every paired pitch')
    base = {'n': int(len(y)), 'draws': draws, 'seed': seed, 'unit': 'game',
            'estimand': 'pitch-weighted mean paired difference (candidate - control)'}
    if not len(y):
        return {**base, 'games': 0, 'status': 'unmeasured', 'nll': None, 'brier': None, 'ece': None, 'p1': None}
    unique, inverse = np.unique(games, return_inverse=True)
    if any(g is None or str(g).lower() in ('nan', 'nat', '<na>') for g in unique):
        raise ValueError('Missing game ID')
    g = len(unique)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.column_stack([np.bincount(inverse, weights=delta[:, c]) for c in range(2)])
    observed = sums.sum(axis=0) / counts.sum()
    stats = {name: _game_bin_stats(y, p, inverse, g, bins) for name, p in (('candidate', pc), ('control', p0))}
    near = {name: np.bincount(inverse, (p[np.arange(len(y)), y] < p1_threshold).astype(np.float64), g)
            for name, p in (('candidate', pc), ('control', p0))}
    total = counts.sum()
    ece_obs = {name: float(_ece_from(s.sum(axis=0), total, bins)) for name, s in stats.items()}
    p1_obs = {name: float(v.sum() / total) for name, v in near.items()}
    result = {**base, 'games': int(g)}
    if g < 2:
        result['status'] = 'insufficient_games'
        for c, name in enumerate(('nll', 'brier')):
            result[name] = {'delta': float(observed[c]), 'ci95': None, 'p_less': None}
        result['ece'] = {'delta': ece_obs['candidate'] - ece_obs['control'], 'ci95': None, **ece_obs}
        result['p1'] = {'delta': p1_obs['candidate'] - p1_obs['control'], 'ci95': None, **p1_obs}
        return result
    rng = np.random.default_rng(seed)
    replicates = np.empty((draws, 2), dtype=np.float64)
    ece_delta = np.empty(draws, dtype=np.float64)
    p1_delta = np.empty(draws, dtype=np.float64)
    near_delta = near['candidate'] - near['control']
    chunk = max(1, min(256, 2_000_000 // g))  # matrix_metrics chunking: identical random index stream
    for start in range(0, draws, chunk):
        n = min(chunk, draws - start)
        indices = rng.integers(0, g, size=(n, g))
        replicates[start:start + n] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
        multiplicity = np.stack([np.bincount(row, minlength=g) for row in indices]).astype(np.float64)
        rows = multiplicity @ counts
        ece_delta[start:start + n] = (_ece_from(multiplicity @ stats['candidate'], rows, bins)
                                      - _ece_from(multiplicity @ stats['control'], rows, bins))
        p1_delta[start:start + n] = (multiplicity @ near_delta) / rows
    result['status'] = 'measured'
    for c, name in enumerate(('nll', 'brier')):
        centered = replicates[:, c] - observed[c]
        result[name] = {'delta': float(observed[c]), 'ci95': np.quantile(replicates[:, c], [.025, .975]).tolist(),
                        'p_less': float((1 + (centered <= observed[c]).sum()) / (draws + 1))}
    result['ece'] = {'delta': ece_obs['candidate'] - ece_obs['control'],
                     'ci95': np.quantile(ece_delta, [.025, .975]).tolist(), **ece_obs}
    result['p1'] = {'delta': p1_obs['candidate'] - p1_obs['control'],
                    'ci95': np.quantile(p1_delta, [.025, .975]).tolist(), **p1_obs}
    return result


def group_calibration(labels, probabilities, vocabulary, groups):
    """SC-1: observed share minus predicted share per code and per registered pitch group."""
    y, p = validate(labels, probabilities)
    if not len(y):
        return None
    observed = np.bincount(y, minlength=p.shape[1]) / len(y)
    predicted = p.mean(axis=0)
    codes = {code: {'observed': float(observed[i]), 'predicted': float(predicted[i]),
                    'observed_minus_predicted': float(observed[i] - predicted[i])} for i, code in enumerate(vocabulary)}
    out = {}
    for name, members in groups.items():
        index = [list(vocabulary).index(code) for code in members]
        o, q = float(observed[index].sum()), float(predicted[index].sum())
        out[name] = {'observed': o, 'predicted': q, 'observed_minus_predicted': o - q}
    return {'n': int(len(y)), 'groups': out, 'codes': codes}


def ratio_bins(labels, probabilities, edges):
    """SC-3: observed / expected over every (decision, code) pair per predicted-probability bin
    [e0, e1), ..., [e_{k-1}, e_k] (the last bin closed)."""
    y, p = validate(labels, probabilities)
    edges = list(edges)
    flat = p.ravel()
    observed = np.zeros_like(p)
    observed[np.arange(len(y)), y] = 1.
    index = np.digitize(flat, edges[1:-1], right=False)
    out = []
    for b in range(len(edges) - 1):
        inside = index == b
        expected, seen = float(flat[inside].sum()), float(observed.ravel()[inside].sum())
        out.append({'bin': [edges[b], edges[b + 1]], 'pairs': int(inside.sum()), 'observed': seen,
                    'expected': expected, 'observed_over_expected': seen / expected if expected > 0 else None})
    return out


def strata_metrics(labels, arms, strata, *, p1, p2, bins=10):
    """Descriptive per-stratum summaries: {stratum: {value: {arm: summary}}}."""
    y = np.asarray(labels, dtype=np.int64)
    out = {}
    for key, values in strata.items():
        text = np.array([str(v) for v in values], dtype=object)
        table = {}
        for value in sorted(set(text.tolist())):
            inside = text == value
            table[value] = {arm: _brief(y[inside], p[inside], p1, p2, bins) for arm, p in arms.items()}
        out[key] = table
    return out


def _brief(y, p, p1, p2, bins):
    if not len(y):
        return {'n': 0}
    summary = prediction_summary(y, p, p1=p1, p2=p2, bins=bins)
    return {k: summary[k] for k in ('n', 'nll', 'brier', 'top_label_ece10', 'p1_share', 'accuracy')}
