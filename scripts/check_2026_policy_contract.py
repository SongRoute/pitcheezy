"""SYNTHETIC-ONLY checks for MLB-2026-POLICY-PREPARATION-v1 (COOP-015, M1 preparation).

Builds a toy plate appearance in memory. Loads no data, model, cache or artifact, takes no
arguments and never calls a 2026 evaluation. Passing proves the stated semantics on the toy,
not identification, transport, overlap or any real policy identity.

Run: PYTHONPATH=src:experiments/pitchmdp python scripts/check_2026_policy_contract.py
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


class IntegrityError(ValueError):
    """FAILED_INTEGRITY: bad mapping/probability/value. Stop; never repair or substitute."""


class Unsupported(Exception):
    """Legitimate refusal (empty support, unknown pitcher, zero logging mass). Stays in the denominator."""


def vocabulary_sha256(vocabulary):
    return hashlib.sha256("\n".join(vocabulary).encode()).hexdigest()


def policy_row(probs, mask, vocabulary, vocabulary_sha256_pin, atol=1e-9):
    """Validate one callable-policy output. Returns probs unchanged; no masking or renormalisation."""
    vocabulary = tuple(vocabulary)
    if vocabulary != tuple(sorted(set(vocabulary))) or vocabulary_sha256(vocabulary) != vocabulary_sha256_pin:
        raise IntegrityError("action vocabulary order/hash mismatch")
    p, m = np.asarray(probs, dtype=np.float64), np.asarray(mask)
    if p.shape != (len(vocabulary),) or m.shape != p.shape or m.dtype != bool:
        raise IntegrityError("probability/mask shape or dtype mismatch")
    if not m.any():
        raise Unsupported("empty_support")
    if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > atol or (p[~m] != 0).any():
        raise IntegrityError("invalid probabilities or mass outside registered support")
    return p


def action_index(vocabulary, label):
    if label not in vocabulary:
        raise IntegrityError(f"unmapped action {label!r}")
    return vocabulary.index(label)


def logging_ratio(pi_e, pi_b):
    """Unclipped per-decision ratio. Candidate mass 0 is valid; logging mass 0 is a refusal."""
    if not (np.isfinite(pi_e) and 0 <= pi_e <= 1 and np.isfinite(pi_b) and 0 <= pi_b <= 1):
        raise IntegrityError("logging/evaluation probability missing or outside [0,1]")
    if pi_b == 0:
        raise Unsupported("logging_positivity")
    return pi_e / pi_b


def we_value(v):
    if not (np.isfinite(v) and 0 <= v <= 1):
        raise IntegrityError("WE value outside [0,1]")
    return float(v)


def we_contrast(v_candidate, v_reference):
    """defense-we-pa-v1: dimensionless initial-defender WE; positive favours the defender."""
    v_candidate, v_reference = we_value(v_candidate), we_value(v_reference)
    delta = v_candidate - v_reference
    return {"delta": delta, "delta_pp": 100 * delta, "offense_delta": -delta}


def sequential_dr(steps, terminal_we, pi, pi_b, q_hat, clip=None):
    """Per-decision DR, reward only at PA end: V_T=0, V_t = v(H_t) + rho_t (r_t + V_{t+1} - q(H_t,a_t)).

    Inputs are validated (full rows, terminal WE, clip); the estimate itself may leave [0,1].
    """
    if not steps:
        raise Unsupported("empty_or_incomplete_pa")
    terminal_we = we_value(terminal_we)
    if clip is not None and not (np.isfinite(clip) and clip > 0):
        raise IntegrityError("clip must be finite and positive")
    value = 0.0
    for t in reversed(range(len(steps))):
        history, action = steps[t]
        p, pb = (np.asarray(f(history), dtype=np.float64) for f in (pi, pi_b))
        p, pb = (policy_row(x, x > 0, VOCAB, SHA) for x in (p, pb))
        a = action_index(VOCAB, action)
        rho = logging_ratio(p[a], pb[a])
        rho = rho if clip is None else min(rho, clip)
        q = np.array([q_hat(history, b) for b in VOCAB], dtype=np.float64)
        if not np.isfinite(q).all():
            raise IntegrityError("non-finite q_hat")
        reward = terminal_we if t == len(steps) - 1 else 0.0
        value = float(p @ q) + rho * (reward + value - q[a])
    return value


# ---------------------------------------------------------------- toy PA (history-dependent)
VOCAB = ("CH", "FF", "SL")
SHA = vocabulary_sha256(VOCAB)
TERMINAL_WE = {"K": .62, "out": .60, "walk": .53, "hit": .44}  # absolute initial-defender WE, PA end
OUTCOME = {  # P(ball, strike, out, hit | action, strikes)
    ("CH", 0): (.40, .30, .18, .12), ("CH", 1): (.35, .30, .22, .13),
    ("FF", 0): (.30, .40, .15, .15), ("FF", 1): (.30, .30, .20, .20),
    ("SL", 0): (.45, .35, .12, .08), ("SL", 1): (.38, .40, .14, .08),
}


def _count(history):
    return sum(o == "ball" for _, o in history), sum(o == "strike" for _, o in history)


def _end(outcome, balls, strikes):
    if outcome in ("out", "hit"):
        return outcome
    if outcome == "ball" and balls == 1:
        return "walk"
    if outcome == "strike" and strikes == 1:
        return "K"
    return None


def _law(table):
    def pi(history):
        _, strikes = _count(history)
        last = history[-1][0] if history else "<START>"
        p = np.array(table.get((strikes, last), table[strikes]), dtype=float)
        return policy_row(p, p > 0, VOCAB, SHA)
    return pi


PI_B = _law({0: (.2, .5, .3), 1: (.25, .35, .4), (0, "FF"): (.3, .3, .4)})       # true logging law
PI_B_WRONG = _law({0: (.4, .3, .3), 1: (.2, .6, .2)})                             # misspecified, positive
PI_REF = _law({0: (.25, .45, .3), 1: (.2, .4, .4)})                               # stands in for TRAIN BC
PI_REF_COPY = _law({0: (.25, .45, .3), 1: (.2, .4, .4)})                          # independent object, same law
PI_CAND = _law({0: (0., .6, .4), 1: (.1, .2, .7), (1, "SL"): (.3, .3, .4)})       # zero CH mass at 0 strikes


def true_q(pi):
    def q(history, action):
        balls, strikes = _count(history)
        total = 0.0
        for outcome, prob in zip(("ball", "strike", "out", "hit"), OUTCOME[action, strikes]):
            end = _end(outcome, balls, strikes)
            nxt = history + ((action, outcome),)
            total += prob * (TERMINAL_WE[end] if end else float(pi(nxt) @ [q(nxt, b) for b in VOCAB]))
        return total
    return q


def value(pi):
    q = true_q(pi)
    return float(pi(()) @ [q((), b) for b in VOCAB])


def wrong_q(history, action):
    return .30 + .05 * len(history) + .04 * VOCAB.index(action)


def trajectories(pi_b):
    """Every PA path under the logging law with its exact probability: (steps, terminal WE, prob)."""
    out = []

    def walk(history, prob):
        balls, strikes = _count(history)
        for a, pa in zip(VOCAB, pi_b(history)):
            for outcome, po in zip(("ball", "strike", "out", "hit"), OUTCOME[a, strikes]):
                w, nxt = prob * pa * po, history + ((a, outcome),)
                if w == 0:
                    continue
                end = _end(outcome, balls, strikes)
                if end:
                    out.append((tuple((nxt[:i], nxt[i][0]) for i in range(len(nxt))), TERMINAL_WE[end], w))
                else:
                    walk(nxt, w)
    walk((), 1.0)
    return out


def expected_dr(pi, pi_b_model, q_hat, clip=None):
    return sum(w * sequential_dr(s, r, pi, pi_b_model, q_hat, clip) for s, r, w in trajectories(PI_B))


def run_checks():
    paths = trajectories(PI_B)
    results = {"n_toy_paths": len(paths), "path_mass": sum(w for *_, w in paths)}
    for name, pi in (("candidate", PI_CAND), ("reference", PI_REF)):
        truth = value(pi)
        results[name] = {
            "true_value": truth,
            "err_correct_propensity_wrong_q": expected_dr(pi, PI_B, wrong_q) - truth,
            "err_wrong_propensity_correct_q": expected_dr(pi, PI_B_WRONG, true_q(pi)) - truth,
            "err_both_wrong": expected_dr(pi, PI_B_WRONG, wrong_q) - truth,
            "err_clip_1p5_wrong_q": expected_dr(pi, PI_B, wrong_q, clip=1.5) - truth,
            "err_clip_1p5_correct_q": expected_dr(pi, PI_B, true_q(pi), clip=1.5) - truth,
        }
    q_ref = true_q(PI_REF)
    results["max_abs_same_policy_paired_delta"] = max(
        abs(sequential_dr(s, r, PI_REF_COPY, PI_B, wrong_q) - sequential_dr(s, r, PI_REF, PI_B, wrong_q)) for s, r, _ in paths)
    results["contrast"] = we_contrast(value(PI_CAND), value(PI_REF))
    results["q_ref_root"] = [q_ref((), b) for b in VOCAB]
    return results


if __name__ == "__main__":
    r = run_checks()
    tol = 1e-12
    gates = {k: bool(v) for k, v in {
        "path_mass_is_one": abs(r["path_mass"] - 1) < tol,
        "dr_exact_correct_propensity": all(abs(r[k]["err_correct_propensity_wrong_q"]) < tol for k in ("candidate", "reference")),
        "dr_exact_correct_q": all(abs(r[k]["err_wrong_propensity_correct_q"]) < tol for k in ("candidate", "reference")),
        "dr_biased_when_both_wrong": all(abs(r[k]["err_both_wrong"]) > 1e-6 for k in ("candidate", "reference")),
        "clip_biased_with_wrong_q": abs(r["candidate"]["err_clip_1p5_wrong_q"]) > 1e-6,
        "same_policy_paired_delta_zero": r["max_abs_same_policy_paired_delta"] == 0,
    }.items()
    }
    print(json.dumps({"synthetic_only": True, "gates": gates, "all_pass": all(gates.values()), "values": r}, indent=1))
    raise SystemExit(0 if all(gates.values()) else 1)
