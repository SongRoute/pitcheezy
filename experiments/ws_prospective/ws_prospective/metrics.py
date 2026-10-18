"""Pre-registered metrics (fixed before any label is read).

* NLL: mean of -log(max(p_y, EPS)), natural log.
* Brier: mean over pitches of sum_k (p_k - 1[y=k])^2 (multiclass, range 0..2).
* ECE: top-label confidence, 15 equal-width bins on [0,1], weighted |acc-conf|.
* top-1: mean 1[argmax p == y] (ties -> lowest leaf index, numpy argmax).
* Paired difference: metric(candidate) - metric(comparator) on the identical
  pitch set (intersection of both models' eligible forecasts).
* CI: percentile bootstrap that resamples *games* (clusters) with replacement,
  fixed seed and B from the registration; per replicate the metric is
  recomputed on the pooled pitches of the drawn games (ECE included).
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

EPS = 1e-12
ECE_BINS = 15


def nll(P: np.ndarray, y: np.ndarray) -> float:
    p = P[np.arange(len(y)), y]
    return float(np.mean(-np.log(np.clip(p, EPS, 1.0))))


def brier(P: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def ece(P: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(y))


def top1(P: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(P.argmax(axis=1) == y))


METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "nll": nll, "brier": brier, "ece": ece, "top1": top1,
}


def validate_probs(P: np.ndarray, tol: float = 1e-6) -> None:
    if P.ndim != 2 or np.any(~np.isfinite(P)) or np.any(P < -tol):
        raise ValueError("invalid probability matrix")
    if np.max(np.abs(P.sum(axis=1) - 1.0)) > 1e-4:
        raise ValueError("rows do not sum to 1")


def cluster_bootstrap_paired(P_a: np.ndarray, P_b: np.ndarray, y: np.ndarray,
                             games: Sequence, *, seed: int, B: int,
                             alpha: float = 0.05) -> Dict[str, dict]:
    """Paired candidate-minus-comparator difference with game-cluster CI."""
    games = np.asarray(games)
    uniq = np.unique(games)
    members = [np.flatnonzero(games == g) for g in uniq]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    out: Dict[str, dict] = {}
    for name, fn in METRICS.items():
        a, b = fn(P_a, y), fn(P_b, y)
        reps = np.empty(B)
        for i in range(B):
            sel = np.concatenate([members[j] for j in draws[i]])
            reps[i] = fn(P_a[sel], y[sel]) - fn(P_b[sel], y[sel])
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        out[name] = {"candidate": a, "comparator": b, "diff": a - b,
                     "ci_low": float(lo), "ci_high": float(hi),
                     "n_pitches": int(len(y)), "n_games": int(len(uniq)),
                     "B": int(B), "seed": int(seed)}
    return out
