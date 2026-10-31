"""Bounded sequence-aware lookahead with a supplied full-PA tail value.

Only observed past pitches and simulated deliveries/outcomes enter histories.
Every outcome and delivery quadrature point is integrated inside the horizon;
the remaining PA is approximated by ``tail_value``. This is not an exact
sequence-history PA MDP, nor does deeper lookahead necessarily improve an
inconsistent tail model. Game context stays fixed until the supplied terminal
utilities are reached, so callers must restrict use to supported PAs without
mid-PA runner/game-state changes.
"""
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .planner import OUTCOMES, TERMINALS


@dataclass
class LookaheadResult:
    q_values: np.ndarray
    one_step_q_values: np.ndarray
    policy: int
    continuation_reference: float
    diagnostics: dict

    @property
    def value(self) -> float:
        return float(self.q_values[self.policy])

    def topk(self, k: int = 5) -> list[dict]:
        return [
            {"action_index": int(a), "value": float(self.q_values[a]),
             "one_step_then_tail_value": float(self.one_step_q_values[a]),
             "delta_vs_continuation_reference": float(self.q_values[a] - self.continuation_reference)}
            for a in np.argsort(-self.q_values, kind="stable")[:k]
        ]


@dataclass
class _State:
    history: np.ndarray
    balls: int
    strikes: int


def solve_lookahead(
    history: np.ndarray,
    balls: int,
    strikes: int,
    actions: np.ndarray,
    predictor: Callable[[Sequence[np.ndarray], np.ndarray, np.ndarray], np.ndarray],
    terminal_values: Mapping[str, float],
    tail_value: Callable[[int, int, np.ndarray], float],
    *,
    weights: np.ndarray | None = None,
    append_token: Callable[[np.ndarray, np.ndarray, str, int, int], np.ndarray] | None = None,
    max_depth: int = 2,
    history_length: int = 5,
    max_prediction_rows: int = 250_000,
) -> LookaheadResult:
    """Choose an action before its uncertain physical delivery is realized.

    ``actions[A,K,F]`` contains K *joint* physical delivery vectors per action,
    with ``weights[K]`` or ``weights[A,K]`` (uniform by default). All candidate
    actions are available at every simulated node. Supply already selected
    candidates; the solver does not silently prune actions or probability mass.

    ``predictor(histories, counts[N,2], deliveries[N,F])`` returns ``[N,10]``
    in ``OUTCOMES`` order. It is called once per depth with all branches batched.
    Histories contain only previous pitches, never the current delivery. The
    callback can close over fixed player/game context. ``append_token`` returns
    a full updated history, given the previous history, delivered vector,
    outcome name, and *pre-pitch* balls/strikes. By default only the physical
    delivery is appended. An outcome/count-aware model must supply its encoder.
    Updated histories are independently copied and truncated to the last
    ``history_length`` tokens; empty histories use shape ``(0, token_width)``.

    ``tail_value(balls, strikes, history)`` supplies absolute defensive utility
    at horizon leaves, e.g. a separately evaluated memoryless baseline PA
    policy. ``terminal_values`` may come from ``game.terminal_values`` and must
    use the same utility scale. The root tail value is reported as a reference,
    not asserted to be this sequence model's evaluated baseline policy.
    """
    if (not isinstance(max_depth, (int, np.integer)) or max_depth < 1 or
            not isinstance(history_length, (int, np.integer)) or history_length < 1 or
            not isinstance(max_prediction_rows, (int, np.integer)) or max_prediction_rows < 1):
        raise ValueError("depth, history length and prediction budget must be positive integers")
    if (not isinstance(balls, (int, np.integer)) or not 0 <= balls <= 3 or
            not isinstance(strikes, (int, np.integer)) or not 0 <= strikes <= 2):
        raise ValueError("balls/strikes must be legal nonterminal integer counts")
    h = np.asarray(history, dtype=float)
    deliveries = np.asarray(actions, dtype=float)
    if h.ndim != 2 or not h.shape[1] or not np.isfinite(h).all():
        raise ValueError("history must be a finite matrix, including for empty history")
    if deliveries.ndim != 3 or not all(deliveries.shape) or not np.isfinite(deliveries).all():
        raise ValueError("actions must be finite with nonempty shape (actions,quadrature,features)")
    na, nk, nf = deliveries.shape
    if append_token is None and h.shape[1] != nf:
        raise ValueError("default history tokens must have the delivery feature width")
    w = np.full((na, nk), 1 / nk) if weights is None else np.asarray(weights, dtype=float)
    if w.shape == (nk,):
        w = np.broadcast_to(w, (na, nk))
    if w.shape != (na, nk) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("quadrature weights must be nonnegative with shape (K,) or (A,K)")
    if not np.allclose(w.sum(axis=-1), 1, rtol=0, atol=1e-8):
        raise ValueError("quadrature weights must sum to one per action")
    w = w / w.sum(axis=-1, keepdims=True)
    terminal = np.array([terminal_values[name] for name in TERMINALS], dtype=float)
    if not np.isfinite(terminal).all():
        raise ValueError("terminal values must be finite")
    diagnostics = {"max_depth": int(max_depth), "history_length": int(history_length),
                   "action_count": na, "quadrature_points_per_action": nk,
                   "prediction_rows": 0, "prediction_calls": 0, "tail_calls": 0,
                   "max_input_probability_mass_error": 0.,
                   "method": "bounded sequence lookahead with supplied PA continuation",
                   "exact_full_history_pa_mdp": False,
                   "one_step_definition": "one optimized pitch then supplied tail value"}

    def frozen(array):
        result = np.array(array, dtype=float, copy=True)
        result.setflags(write=False)
        return result

    root = _State(frozen(h[-history_length:]), int(balls), int(strikes))

    def tail(state):
        value = float(tail_value(state.balls, state.strikes, state.history))
        diagnostics["tail_calls"] += 1
        if not np.isfinite(value):
            raise ValueError("tail values must be finite")
        return value

    def child(state, delivery, outcome, next_balls, next_strikes):
        updated = (np.concatenate([state.history, delivery[None, :]], axis=0)
                   if append_token is None else
                   np.asarray(append_token(state.history, delivery, outcome,
                                           state.balls, state.strikes), dtype=float))
        if (updated.ndim != 2 or not len(updated) or updated.shape[1] != h.shape[1]
                or not np.isfinite(updated).all()):
            raise ValueError("append_token must return a finite nonempty history of the original token width")
        return _State(frozen(updated[-history_length:]), next_balls, next_strikes)

    def evaluate(states, depth):
        n = len(states)
        rows = n * na * nk
        if diagnostics["prediction_rows"] + rows > max_prediction_rows:
            raise ValueError("lookahead exceeds max_prediction_rows; reduce depth, actions or quadrature")
        histories = [state.history for state in states for _ in range(na * nk)]
        counts = np.repeat(np.array([(state.balls, state.strikes) for state in states]), na * nk, axis=0)
        candidates = frozen(np.tile(deliveries.reshape(-1, nf), (n, 1)))
        p = np.asarray(predictor(histories, counts, candidates), dtype=float)
        diagnostics["prediction_calls"] += 1
        diagnostics["prediction_rows"] += rows
        if p.shape != (rows, len(OUTCOMES)) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("predictor must return finite nonnegative probabilities with shape (N,10)")
        error = float(np.max(np.abs(p.sum(axis=-1) - 1)))
        if error > 1e-6:
            raise ValueError("predictor probability rows must sum to one")
        diagnostics["max_input_probability_mass_error"] = max(
            diagnostics["max_input_probability_mass_error"], error)
        p = (p / p.sum(axis=-1, keepdims=True)).reshape(n, na, nk, len(OUTCOMES))
        utility = np.einsum("nako,o->nak", p[..., 3:], terminal[2:])
        children, links = [], []
        for i, state in enumerate(states):
            if state.balls == 3:
                utility[i] += p[i, ..., 0] * terminal[0]
            if state.strikes == 2:
                utility[i] += p[i, ..., 1] * terminal[1]
            for a in range(na):
                for k in range(nk):
                    if w[a, k] == 0:
                        continue
                    for outcome in range(3):
                        probability = p[i, a, k, outcome]
                        if probability == 0 or (outcome == 0 and state.balls == 3) or (outcome == 1 and state.strikes == 2):
                            continue
                        next_balls = state.balls + int(outcome == 0)
                        next_strikes = min(2, state.strikes + int(outcome in (1, 2)))
                        children.append(child(state, candidates[(i * na + a) * nk + k], OUTCOMES[outcome],
                                              next_balls, next_strikes))
                        links.append((i, a, k, probability))
        # Reuse leaf values for the root one-step comparator when depth == 1.
        leaf_values = [tail(s) for s in children] if depth == 1 or depth == max_depth else None
        one_step = utility.copy() if depth == max_depth else None
        if one_step is not None:
            for (i, a, k, probability), value in zip(links, leaf_values):
                one_step[i, a, k] += probability * value
        if children:
            future = leaf_values if depth == 1 else evaluate(children, depth - 1)[0].max(axis=-1)
            for (i, a, k, probability), value in zip(links, future):
                utility[i, a, k] += probability * value
        return np.sum(utility * w[None, ...], axis=-1), (None if one_step is None else
                                                        np.sum(one_step * w[None, ...], axis=-1))

    reference = tail(root)
    q, one_step = evaluate([root], max_depth)
    return LookaheadResult(q[0], one_step[0], int(np.argmax(q[0])), reference, diagnostics)
