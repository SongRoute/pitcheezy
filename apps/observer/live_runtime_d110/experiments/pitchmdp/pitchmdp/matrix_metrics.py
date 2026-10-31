"""Auditable ten-class metrics and paired whole-game uncertainty for ML matrices.

Bootstrap intervals condition on the supplied predictions. They do not incorporate
model fitting, calibration, data selection, or hyperparameter-selection uncertainty.
"""
from __future__ import annotations

import numpy as np


def validate_probabilities(labels, probabilities, n_classes=10):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != (len(y), n_classes):
        raise ValueError("Incompatible labels and probability shape")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= n_classes)):
        raise ValueError("Labels outside the declared class contract")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Probability mass differs from one")
    return y.astype(np.int64, copy=False), p


def pitch_losses(labels, probabilities):
    y, p = validate_probabilities(labels, probabilities)
    selected = p[np.arange(len(y)), y]
    nll = -np.log(np.clip(selected, 1e-12, 1))
    brier = np.square(p).sum(axis=1) - 2 * selected + 1
    return np.column_stack([nll, brier])


def prediction_metrics(labels, probabilities, bins=10):
    if bins != 10:
        raise ValueError("This protocol fixes ten calibration bins")
    y, p = validate_probabilities(labels, probabilities)
    if not len(y):
        return {"n": 0, "log_loss": None, "brier_multiclass": None,
                "accuracy": None, "macro_f1_observed_classes": None, "classes": []}
    losses = pitch_losses(y, p)
    predicted = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = predicted == y
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    for b in range(bins):
        mask = index == b
        reliability.append({"bin": b, "n": int(mask.sum()),
                            "accuracy": float(correct[mask].mean()) if mask.any() else None,
                            "confidence": float(confidence[mask].mean()) if mask.any() else None})
    ece = sum(row["n"] / len(y) * abs(row["accuracy"] - row["confidence"])
              for row in reliability if row["n"])
    classes = []
    for c in range(p.shape[1]):
        actual, chosen = y == c, predicted == c
        tp, support, positive = int((actual & chosen).sum()), int(actual.sum()), int(chosen.sum())
        precision = tp / positive if positive else None
        recall = tp / support if support else None
        # No observed class means its F1 is unmeasured, not an artificial zero.
        f1 = 2 * tp / (support + positive) if support else None
        per_bin = np.minimum((p[:, c] * bins).astype(int), bins - 1)
        class_ece = sum(float(mask.mean()) * abs(float(actual[mask].mean()) - float(p[mask, c].mean()))
                        for b in range(bins) if (mask := per_bin == b).any())
        classes.append({"class": c, "support": support, "predicted_count": positive,
                        "precision": precision, "recall": recall, "f1": f1,
                        "observed_rate": support / len(y), "predicted_rate": float(p[:, c].mean()),
                        "brier": float(np.square(p[:, c] - actual).mean()),
                        "ece": class_ece})
    measured_f1 = [row["f1"] for row in classes if row["f1"] is not None]
    return {"n": len(y), "log_loss": float(losses[:, 0].mean()),
            "brier_multiclass": float(losses[:, 1].mean()), "accuracy": float(correct.mean()),
            "macro_f1_observed_classes": float(np.mean(measured_f1)),
            "macro_f1_class_count": len(measured_f1), "top_label_ece10": float(ece),
            "reliability": reliability, "classes": classes}


def paired_game_comparison(labels, candidate, control, game_ids, *, draws=10000, seed=20260924):
    """Candidate minus control: negative NLL/Brier differences are improvements.

    Each replicate samples complete games, then divides the loss sum by the
    sampled pitch count. The one-sided test is a null-centered cluster bootstrap,
    not the fraction of uncentered replicates on the other side of zero.
    """
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws required")
    delta = pitch_losses(labels, candidate) - pitch_losses(labels, control)
    games = np.asarray(game_ids)
    if games.ndim != 1 or len(games) != len(delta):
        raise ValueError("One game ID is required for every paired pitch")
    if not len(delta):
        return {"n": 0, "games": 0, "status": "unmeasured", "nll": None, "brier": None}
    unique, inverse = np.unique(games, return_inverse=True)
    if any(g is None or str(g).lower() in ("nan", "nat", "<na>") for g in unique):
        raise ValueError("Missing game ID")
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.column_stack([np.bincount(inverse, weights=delta[:, c]) for c in range(2)])
    observed = sums.sum(axis=0) / counts.sum()
    result = {"n": len(delta), "games": len(unique), "draws": draws, "seed": seed,
              "estimand": "pitch-weighted mean paired loss difference",
              "uncertainty": "conditional on fixed predictions; whole-game bootstrap",
              "test": "one-sided improvement; null-centered whole-game bootstrap with plus-one correction"}
    if len(unique) < 2:
        result["status"] = "insufficient_games"
        for c, name in enumerate(("nll", "brier")):
            result[name] = {"delta": float(observed[c]), "ci95": None, "p_less": None}
        return result
    rng = np.random.default_rng(seed)
    replicates = np.empty((draws, 2), dtype=np.float64)
    # Bound index memory for whole-league cohorts as well as the small C6.
    chunk = max(1, min(256, 2_000_000 // len(unique)))
    for start in range(0, draws, chunk):
        n = min(chunk, draws - start)
        indices = rng.integers(0, len(unique), size=(n, len(unique)))
        replicates[start:start+n] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    result["status"] = "measured"
    for c, name in enumerate(("nll", "brier")):
        centered = replicates[:, c] - observed[c]
        result[name] = {"delta": float(observed[c]),
                        "ci95": np.quantile(replicates[:, c], [.025, .975]).tolist(),
                        "p_less": float((1 + (centered <= observed[c]).sum()) / (draws + 1))}
    return result


def holm_adjust(p_values):
    """Adjusted values in original order; missing tests keep their family slot."""
    values = list(p_values)
    measured = [(i, float(p)) for i, p in enumerate(values) if p is not None]
    if any(not np.isfinite(p) or not 0 <= p <= 1 for _, p in measured):
        raise ValueError("Invalid p-value")
    measured.sort(key=lambda item: item[1])
    adjusted = [None] * len(values)
    running = 0.
    for rank, (index, p) in enumerate(measured):
        running = min(1., max(running, (len(values) - rank) * p))
        adjusted[index] = running
    return adjusted


def prediction_decision(comparison, p_holm, seed_deltas, *, minimum_improvement=.003,
                        brier_margin=.001, stage="screen"):
    if stage not in ("screen", "final"):
        raise ValueError("Unknown comparison stage")
    expected, required_seed_improvements = (3, 2) if stage == "screen" else (5, 4)
    if len(seed_deltas) != expected or not np.isfinite(np.asarray(seed_deltas, float)).all():
        raise ValueError(f"Complete {expected}-seed finite paired differences required")
    if p_holm is not None and (not np.isfinite(p_holm) or not 0 <= p_holm <= 1):
        raise ValueError("Invalid adjusted p-value")
    if comparison.get("status") != "measured" or p_holm is None:
        return {"status": "unmeasured", "reason": "paired inference unavailable"}
    nll, brier = comparison["nll"], comparison["brier"]
    criteria = {"practical_improvement": nll["delta"] <= -minimum_improvement,
                "paired_ci_below_zero": nll["ci95"][1] < 0,
                "holm_significant": p_holm <= .05,
                "brier_noninferior": brier["ci95"][1] <= brier_margin,
                "seed_direction_stable": sum(x < 0 for x in seed_deltas) >= required_seed_improvements}
    if all(criteria.values()):
        status = "predictive_improvement"
    elif nll["ci95"][0] > 0 or brier["ci95"][0] > brier_margin:
        status = "worse_or_guardrail_failure"
    else:
        status = "inconclusive"
    return {"status": status, "criteria": criteria, "p_holm": p_holm,
            "seed_deltas": list(seed_deltas), "policy_effect": None,
            "scope": "prediction only; no policy or unmeasured-group improvement claim"}
