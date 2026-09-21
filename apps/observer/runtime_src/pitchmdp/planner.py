"""Finite plate-appearance planning with a fixed defensive-team terminal value.

Only count and preceding pitch type evolve before a terminal event. A two-strike
foul keeps the count but *does* replace the preceding pitch type. No recorded
future pitch is used by this solver.
"""
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


OUTCOMES = (
    "ball", "strike", "foul", "out", "single", "double", "triple",
    "home_run", "hbp", "double_play",
)
TERMINALS = ("walk", "strikeout", *OUTCOMES[3:])


@dataclass
class PlanResult:
    values: np.ndarray
    q_values: np.ndarray
    baseline_values: np.ndarray
    myopic_q_values: np.ndarray
    policy: np.ndarray
    diagnostics: dict

    def topk(self, balls: int, strikes: int, prev: int, k: int = 5) -> list[dict]:
        q = self.q_values[balls, strikes, prev]
        baseline = float(self.baseline_values[balls, strikes, prev])
        return [
            {"action_index": int(a), "value": float(q[a]),
             "delta_vs_baseline": float(q[a] - baseline),
             "one_step_then_baseline_value": float(self.myopic_q_values[balls, strikes, prev, a])}
            for a in np.argsort(-q, kind="stable")[:k]
        ]


def solve_pa(
    probabilities: np.ndarray,
    terminal_values: Mapping[str, float],
    action_next_prev: Sequence[int],
    baseline_policy: np.ndarray | None = None,
    *, tolerance: float = 1e-12,
    max_iterations: int = 10000,
) -> PlanResult:
    """Solve a PA, evaluate a fixed policy, and improve that policy for one pitch.

    ``probabilities`` has shape (4, 3, n_previous_types, n_actions, 10),
    in ``OUTCOMES`` order. ``action_next_prev[a]`` is the preceding-type
    index after action a, for *every* nonterminal outcome including a foul.
    Terminal values are absolute defensive win probabilities (or a consistent
    alternative utility) keyed by ``TERMINALS``. The baseline defaults to
    uniform actions, or accepts shape (n_actions,) / (4,3,n_previous,n_actions).

    Exact linear solves evaluate the baseline's two-strike foul loops; the
    optimal Bellman equation is a contraction over preceding pitch types.
    Other counts are evaluated by backwards count order. Pure endless-foul
    actions are rejected so that every candidate policy ends a PA almost surely.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 5 or p.shape[:2] != (4, 3) or p.shape[-1] != len(OUTCOMES):
        raise ValueError("probabilities must have shape (4,3,n_previous,n_actions,10)")
    nprev, na = p.shape[2:4]
    if not nprev or not na or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and nonnegative with nonempty actions/states")
    mass_error = float(np.max(np.abs(p.sum(axis=-1) - 1)))
    if mass_error > 1e-6:
        raise ValueError(f"probability rows must sum to one; max error {mass_error}")
    # Remove insignificant float32 accumulation error, not missing probability mass.
    p = p / p.sum(axis=-1, keepdims=True)
    next_prev = np.asarray(action_next_prev, dtype=np.int64)
    if next_prev.shape != (na,) or (next_prev < 0).any() or (next_prev >= nprev).any():
        raise ValueError("action_next_prev must contain valid preceding-type indices, one per action")
    if np.max(p[:, 2, :, :, 2]) >= 1 - 1e-12:
        raise ValueError("pure two-strike foul actions have no guaranteed terminal PA outcome")
    terminal = np.array([terminal_values[name] for name in TERMINALS], dtype=float)
    if not np.isfinite(terminal).all():
        raise ValueError("terminal values must be finite")
    weights = np.full((4, 3, nprev, na), 1 / na) if baseline_policy is None else np.asarray(baseline_policy, dtype=float)
    if weights.shape == (na,):
        weights = np.broadcast_to(weights, (4, 3, nprev, na))
    if weights.shape != (4, 3, nprev, na) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid baseline policy dimensions/probabilities")
    if not np.allclose(weights.sum(axis=-1), 1, atol=1e-8):
        raise ValueError("baseline policy rows must sum to one")
    weights = weights / weights.sum(axis=-1, keepdims=True)
    values = np.zeros((4, 3, nprev))
    baseline = np.zeros_like(values)
    q = np.zeros((4, 3, nprev, na))
    qb = np.zeros_like(q)
    total_iterations = 0
    residual = 0.0

    def fixed_part(b, s, v):
        row = p[b, s]
        result = np.einsum("pao,o->pa", row[:, :, 3:], terminal[2:])
        result += row[:, :, 0] * (terminal[0] if b == 3 else v[b + 1, s, next_prev][None, :])
        result += row[:, :, 1] * (terminal[1] if s == 2 else v[b, s + 1, next_prev][None, :])
        if s < 2:
            result += row[:, :, 2] * v[b, s + 1, next_prev][None, :]
        return result

    for b in range(3, -1, -1):
        for s in range(2, -1, -1):
            base_q = fixed_part(b, s, baseline)
            opt_q = fixed_part(b, s, values)
            if s == 2:
                foul = p[b, s, :, :, 2]
                transition = np.zeros((nprev, nprev))
                for a in range(na):
                    transition[:, next_prev[a]] += weights[b, s, :, a] * foul[:, a]
                rhs = np.sum(weights[b, s] * base_q, axis=-1)
                baseline[b, s] = np.linalg.solve(np.eye(nprev) - transition, rhs)
                qb[b, s] = base_q + foul * baseline[b, s, next_prev][None, :]
                # Starting from baseline reduces iterations and preserves a lower bound.
                v = baseline[b, s].copy()
                for iteration in range(1, max_iterations + 1):
                    next_v = np.max(opt_q + foul * v[next_prev][None, :], axis=-1)
                    error = float(np.max(np.abs(next_v - v)))
                    v = next_v
                    if error <= tolerance:
                        break
                else:
                    raise RuntimeError("two-strike foul Bellman iteration did not converge")
                total_iterations += iteration
                values[b, s] = v
                q[b, s] = opt_q + foul * v[next_prev][None, :]
                residual = max(residual, float(np.max(np.abs(v - q[b, s].max(axis=-1)))))
            else:
                qb[b, s] = base_q
                baseline[b, s] = np.sum(weights[b, s] * base_q, axis=-1)
                q[b, s] = opt_q
                values[b, s] = np.max(opt_q, axis=-1)
    return PlanResult(
        values, q, baseline, qb, np.argmax(q, axis=-1),
        {"max_input_probability_mass_error": mass_error,
         "max_bellman_residual": residual, "foul_iterations": total_iterations,
         "minimum_full_minus_baseline": float(np.min(values - baseline)),
         "state_count": int(values.size), "action_count": na,
         "myopic_definition": "optimize one pitch then follow the fixed baseline policy"},
    )


def plan_pa(
    actions: Sequence[Mapping],
    previous_pitch_types: Sequence[str],
    predictor: Callable[[int, int, str, Sequence[Mapping]], np.ndarray],
    terminal_value: Callable[[str], float] | Mapping[str, float],
    baseline_policy: np.ndarray | None = None,
) -> PlanResult:
    """Build the finite tensor using model-generated future counts and pitch types."""
    p = np.empty((4, 3, len(previous_pitch_types), len(actions), len(OUTCOMES)))
    for b in range(4):
        for s in range(3):
            for previous_index, previous_type in enumerate(previous_pitch_types):
                p[b, s, previous_index] = predictor(b, s, previous_type, actions)
    index = {name: i for i, name in enumerate(previous_pitch_types)}
    next_prev = [index[action["pitch_type"]] for action in actions]
    terminal = terminal_value if isinstance(terminal_value, Mapping) else {name: terminal_value(name) for name in TERMINALS}
    return solve_pa(p, terminal, next_prev, baseline_policy)
