"""Pitch-type policy primitives for model-internal, bounded-MC PA experiments.

P1 improves the first decision under BC continuation. P2 repeats this rollout
improvement every pitch. P3 softens the same Q^BC toward BC; it is NOT a KL
Bellman solution. No method claims an exact optimal PA policy or causal value.
The caller owns TRAIN provenance verification, frozen model selection, legal
as-of context encoding, PA eligibility, calibration and experiment manifests.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Mapping, Sequence

import numpy as np

from .planner import OUTCOMES, TERMINALS


@dataclass(frozen=True)
class PastPitch:
    action: str
    physics: tuple[float, ...]
    outcome: str
    balls: int
    strikes: int

    def __post_init__(self):
        if self.outcome not in (*OUTCOMES, "unknown") or not self.action or not np.isfinite(self.physics).all():
            raise ValueError("invalid generated/observed past pitch")
        _count(self.balls, self.strikes)


def _count(balls, strikes):
    if (not isinstance(balls, (int, np.integer)) or not 0 <= balls <= 3 or
            not isinstance(strikes, (int, np.integer)) or not 0 <= strikes <= 2):
        raise ValueError("illegal nonterminal count")


@dataclass(frozen=True)
class PAState:
    """Only pre-pitch information; no slot for the current realized action.

    context_key identifies the caller's frozen pre-pitch context, never a future
    row. Outcome callbacks can close over that context. Full simulated history
    is retained for reproducible planning RNG; adapters may encode last H5.
    """
    balls: int
    strikes: int
    pitcher: str
    batter_side: str
    history: tuple[PastPitch, ...] = ()
    context_key: str = ""

    def __post_init__(self):
        _count(self.balls, self.strikes)
        if not isinstance(self.history, tuple) or not all(isinstance(x, PastPitch) for x in self.history):
            raise ValueError("history must be an immutable tuple of past pitches")


@dataclass(frozen=True)
class BCRecord:
    state: PAState
    action: str
    split: str


class CategoricalBC:
    """TRAIN-only hierarchical categorical BC; support is not causal overlap.

    Frequency comparator uses pitcher frequencies on exactly the same support.
    BC adds count, batter side and preceding observed/generated pitch type.
    Unknown pitchers use TRAIN league support/frequencies, explicitly flagged.
    """
    def __init__(self, prior_strength=20., minimum_action_count=1):
        if not np.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if not isinstance(minimum_action_count, int) or minimum_action_count < 1:
            raise ValueError("minimum_action_count must be positive integer")
        self.prior_strength = float(prior_strength)
        self.minimum_action_count = minimum_action_count

    @staticmethod
    def _key(state):
        previous = state.history[-1].action if state.history else "<START>"
        return state.pitcher, state.balls, state.strikes, state.batter_side, previous

    def fit(self, records: Sequence[BCRecord]):
        # Single-pass iterable support avoids materializing millions of states.
        league, pitchers, cells = Counter(), defaultdict(Counter), defaultdict(Counter)
        for row in records:
            if row.split != "train" or not row.action:
                raise ValueError("BC fit requires exclusively TRAIN records with actions")
            league[row.action] += 1
            pitchers[row.state.pitcher][row.action] += 1
            cells[self._key(row.state)][row.action] += 1
        if not league:
            raise ValueError("BC fit requires nonempty exclusively TRAIN records")
        self.actions = tuple(sorted(league))
        self.league, self.pitchers, self.cells = league, pitchers, cells
        return self

    def support(self, state):
        counts = self.pitchers.get(state.pitcher, self.league)
        mask = np.array([counts[a] >= self.minimum_action_count for a in self.actions])
        # An empty known-pitcher support is a reported abstention, not permission
        # to invent a different repertoire from unseen player data.
        return mask

    def probabilities(self, state, *, frequency=False):
        mask = self.support(state)
        if not mask.any():
            raise ValueError("no TRAIN-supported actions; abstain")
        counts = self.pitchers.get(state.pitcher, self.league)
        prior = np.array([counts[a] for a in self.actions], dtype=float) * mask
        prior /= prior.sum()
        if frequency:
            return prior
        cell = self.cells.get(self._key(state), {})
        p = (np.array([cell.get(a, 0) for a in self.actions]) + self.prior_strength * prior) * mask
        return p / p.sum()

    def fallback(self, state):
        return state.pitcher not in self.pitchers


def probabilities(values, width):
    p = np.asarray(values, dtype=np.float64)
    if p.shape[-1:] != (width,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid probabilities")
    if not np.allclose(p.sum(axis=-1), 1, rtol=0, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    return p / p.sum(axis=-1, keepdims=True)


def calibrated_conditional(logits, temperatures, frequency, neural_weight):
    """Per-seed temperature, probability mean, then frozen frequency blend.

    logits[S,N,10] are conditional on the SAME selected physical delivery.
    neural_weight=1 means neural only. Integrating these over all 400 draws
    recovers the ordinary integrated ensemble+blend by linearity.
    """
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(temperatures, dtype=np.float64)
    if z.ndim != 3 or z.shape[2] != 10 or not z.shape[0] or not np.isfinite(z).all():
        raise ValueError("logits must be finite [seeds,rows,10]")
    if t.shape != (z.shape[0],) or not np.isfinite(t).all() or (t <= 0).any():
        raise ValueError("one positive temperature per seed required")
    if not np.isfinite(neural_weight) or not 0 <= neural_weight <= 1:
        raise ValueError("neural_weight must lie in [0,1]")
    f = probabilities(frequency, 10)
    if f.shape != z.shape[1:]:
        raise ValueError("frequency must have shape [rows,10]")
    z = z / t[:, None, None]
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return probabilities(neural_weight * p.mean(axis=0) + (1-neural_weight) * f, 10)


@dataclass(frozen=True)
class DeliveryPool:
    """Joint physical vectors; provenance assertions must be audited by caller."""
    values: np.ndarray
    source_split: str
    source_hash: str
    draw_key: str

    def __post_init__(self):
        v = np.array(self.values, dtype=float, copy=True)
        if v.ndim != 2 or v.shape[0] != 400 or not v.shape[1] or not np.isfinite(v).all():
            raise ValueError("exactly 400 finite joint TRAIN draws required")
        if self.source_split != "train" or not self.source_hash or not self.draw_key:
            raise ValueError("TRAIN provenance/hash/draw key required")
        v.setflags(write=False)
        object.__setattr__(self, "values", v)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RowBudget:
    limit: int
    seed_count: int = 3
    conditional_rows: int = 0
    network_rows: int = 0
    calls: int = 0

    def __post_init__(self):
        if any(not isinstance(v, int) or v < 1 for v in (self.limit, self.seed_count)):
            raise ValueError("positive integer row limit and seed count required")

    def consume(self, rows):
        if self.conditional_rows + rows > self.limit:
            raise BudgetExceeded("conditional row budget exceeded; preserve partial run, do not shrink protocol")
        self.conditional_rows += rows
        self.network_rows += rows * self.seed_count
        self.calls += 1


def _draw(p, u):
    if not np.isfinite(u) or not 0 <= u < 1:
        raise ValueError("uniform must lie in [0,1)")
    return min(int(np.searchsorted(np.cumsum(p), u, side="right")), len(p)-1)


@dataclass(frozen=True)
class Step:
    state: PAState | None
    value: float | None
    action: str
    physics: tuple[float, ...]
    outcome: str


class JointSimulator:
    """Batched coherent delivery→conditional outcome→generated history.

    predictor(states, actions, physics[N,F]) returns calibrated conditional
    [N,10]. pool(state, action) returns frozen exact400 TRAIN draws. terminal
    utilities are absolute PA-end WE for the initial defending team; caller can
    use game.terminal_values. Supported PAs exclude mid-PA state changes.
    """
    def __init__(self, pool: Callable, predictor: Callable, terminal: Callable, budget: RowBudget):
        self.pool, self.predictor, self.terminal, self.budget = pool, predictor, terminal, budget

    def step(self, states, actions, uniforms):
        if len(states) != len(actions):
            raise ValueError("one action per state required")
        u = np.asarray(uniforms, dtype=float)
        if u.shape != (len(states), 2) or not np.isfinite(u).all() or (u < 0).any() or (u >= 1).any():
            raise ValueError("delivery/outcome uniforms must have shape [N,2]")
        if not states:
            return []
        delivery = np.stack([self.pool(s, a).values[int(x*400)] for s, a, x in zip(states, actions, u[:, 0])])
        self.budget.consume(len(states))
        p = probabilities(self.predictor(states, actions, delivery), 10)
        if p.shape != (len(states), 10):
            raise ValueError("predictor must return [N,10]")
        result = []
        for s, a, d, probs, x in zip(states, actions, delivery, p, u[:, 1]):
            outcome = OUTCOMES[_draw(probs, x)]
            terminal = ("walk" if outcome == "ball" and s.balls == 3 else
                        "strikeout" if outcome == "strike" and s.strikes == 2 else
                        outcome if outcome in OUTCOMES[3:] else None)
            physical = tuple(float(v) for v in d)
            if terminal:
                value = _we(self.terminal(s, terminal))
                result.append(Step(None, value, a, physical, outcome))
            else:
                token = PastPitch(a, physical, outcome, s.balls, s.strikes)
                child = PAState(s.balls + int(outcome == "ball"),
                                min(2, s.strikes + int(outcome in ("strike", "foul"))),
                                s.pitcher, s.batter_side, s.history + (token,), s.context_key)
                result.append(Step(child, None, a, physical, outcome))
        return result


def _we(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("defensive WE must lie in [0,1]")
    return value


def planning_seed(state, actions, base_seed):
    """Stable full-state/support key; independent of evaluation random streams."""
    payload = [int(base_seed), state.context_key, state.pitcher, state.batter_side,
               state.balls, state.strikes, list(actions),
               [[p.action, list(p.physics), p.outcome, p.balls, p.strikes] for p in state.history]]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass
class Rollouts:
    values: np.ndarray
    truncated: np.ndarray
    pitches: np.ndarray

    @property
    def lower(self):
        return np.where(self.truncated, 0., self.values)

    @property
    def upper(self):
        return np.where(self.truncated, 1., self.values)


def rollouts(simulator, states, policy, uniforms, cutoff, *, first_actions=None):
    """Common evaluator, batched across trajectories at each pitch.

    uniforms[N,cap,3]: policy, delivery, outcome; preallocation preserves CRN
    when paired policies have different trajectory lengths. cutoff supplies a
    shared explicit tail approximation, never an unreported observed future.
    policy(state, step) returns (action_names, normalized probabilities).
    """
    u = np.asarray(uniforms, dtype=float)
    n = len(states)
    if (u.ndim != 3 or u.shape[0] != n or u.shape[2] != 3 or u.shape[1] < 1 or
            not np.isfinite(u).all() or (u < 0).any() or (u >= 1).any()):
        raise ValueError("invalid rollout uniforms")
    if first_actions is not None and len(first_actions) != n:
        raise ValueError("first_actions must match states")
    current = list(states)
    values, lengths = np.zeros(n), np.zeros(n, dtype=int)
    for depth in range(u.shape[1]):
        active = [i for i, s in enumerate(current) if s is not None]
        if not active:
            break
        actions = []
        batched = None
        if hasattr(policy, "batch_probabilities") and not (depth == 0 and first_actions is not None):
            names, batched = policy.batch_probabilities([current[i] for i in active], depth)
            batched = probabilities(batched, len(names))
            if batched.shape != (len(active), len(names)):
                raise ValueError("batched policy probabilities must have shape [states,actions]")
        for row_index, i in enumerate(active):
            if depth == 0 and first_actions is not None:
                actions.append(first_actions[i])
            else:
                if batched is None:
                    names, p = policy(current[i], depth)
                    p = probabilities(p, len(names))
                    if p.ndim != 1:
                        raise ValueError("policy probabilities must be a vector")
                else:
                    p = batched[row_index]
                actions.append(names[_draw(p, u[i, depth, 0])])
        steps = simulator.step([current[i] for i in active], actions, u[active, depth, 1:])
        for i, step in zip(active, steps):
            lengths[i] += 1
            current[i] = step.state
            if step.value is not None:
                values[i] = step.value
    truncated = np.array([s is not None for s in current])
    for i in np.flatnonzero(truncated):
        values[i] = _we(cutoff(current[i]))
    return Rollouts(values, truncated, lengths)


def kl_policy(q, bc, mask, tau):
    """Argmax_pi E_pi Q - tau KL(pi||BC), for supplied Q^BC only."""
    q, bc, mask = np.asarray(q), np.asarray(bc), np.asarray(mask, dtype=bool)
    if q.ndim != 1 or bc.shape != q.shape or mask.shape != q.shape or not mask.any():
        raise ValueError("Q/BC/support dimensions invalid")
    bc = probabilities(bc, len(q))
    if not np.isfinite(q[mask]).all() or (bc[mask] <= 0).any() or (bc[~mask] != 0).any():
        raise ValueError("BC must be positive exactly on support and Q finite there")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive in absolute WE units")
    weights = np.zeros(len(q))
    log_weights = np.log(bc[mask]) + (q[mask] - q[mask].max()) / tau
    weights[mask] = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


class RolloutImprovement:
    """One-step policy improvement by bounded Monte Carlo Q^BC.

    Search reuses the same random tensor across candidate actions and is keyed
    only by full legal state/support/search seed. Evaluation RNG is external.
    Max search rows per decision = supported_actions * samples * pitch_cap.
    P2/P3 repeat this approximation until PA end; neither solves optimal PA DP.
    """
    def __init__(self, bc, simulator, cutoff, *, samples=8, pitch_cap=16, seed=701, cache_size=4096):
        if any(not isinstance(v, int) or v < 1 for v in (samples, pitch_cap)):
            raise ValueError("positive integer samples/pitch_cap required")
        self.bc, self.simulator, self.cutoff = bc, simulator, cutoff
        self.samples, self.pitch_cap, self.seed = samples, pitch_cap, seed
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self.cache_size, self._cache = cache_size, {}
        self.diagnostics = {"decisions": 0, "search_rollouts": 0, "search_truncated": 0,
                            "search_conditional_rows": 0, "cache_hits": 0}

    def baseline(self, state, depth=0):
        return self.bc.actions, self.bc.probabilities(state)

    def q_values(self, state):
        names = [a for a, ok in zip(self.bc.actions, self.bc.support(state)) if ok]
        if not names:
            raise ValueError("no TRAIN-supported actions; abstain")
        key = planning_seed(state, names, self.seed)
        # Include the exact state too: a hash collision cannot reuse another Q.
        cache_key = (state, tuple(names), key)
        if cache_key in self._cache:
            self.diagnostics["cache_hits"] += 1
            q, diagnostics = self._cache[cache_key]
            return q.copy(), {k: v.copy() if isinstance(v, np.ndarray) else v
                              for k, v in diagnostics.items()}
        before = self.simulator.budget.conditional_rows
        rng = np.random.default_rng(key)
        common = rng.random((self.samples, self.pitch_cap, 3))
        u = np.tile(common, (len(names), 1, 1))
        result = rollouts(self.simulator, [state] * len(u), self.baseline, u, self.cutoff,
                          first_actions=[a for a in names for _ in range(self.samples)])
        q = np.full(len(self.bc.actions), -np.inf)
        se = np.full(len(q), np.nan)
        lower, upper = q.copy(), q.copy()
        for a, values, lo, hi in zip(names, result.values.reshape(-1, self.samples),
                                     result.lower.reshape(-1, self.samples), result.upper.reshape(-1, self.samples)):
            i = self.bc.actions.index(a)
            q[i] = values.mean()
            se[i] = values.std(ddof=1)/np.sqrt(self.samples) if self.samples > 1 else np.nan
            lower[i], upper[i] = lo.mean(), hi.mean()
        self.diagnostics["decisions"] += 1
        self.diagnostics["search_rollouts"] += len(u)
        self.diagnostics["search_truncated"] += int(result.truncated.sum())
        self.diagnostics["search_conditional_rows"] += self.simulator.budget.conditional_rows - before
        diagnostics = {"mc_se": se, "truncation_lower": lower, "truncation_upper": upper,
                       "truncated": int(result.truncated.sum()), "rollouts": len(u)}
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = (q.copy(), {k: v.copy() if isinstance(v, np.ndarray) else v
                                               for k, v in diagnostics.items()})
        return q, diagnostics

    def policy(self, mode, *, tau=None):
        if mode not in ("P0", "frequency", "P1", "P2", "P3"):
            raise ValueError("unknown policy")
        if mode == "P3" and (tau is None or not np.isfinite(tau) or tau <= 0):
            raise ValueError("P3 requires a frozen positive tau")
        def choose(state, depth):
            if mode in ("P0", "frequency") or (mode == "P1" and depth > 0):
                return self.bc.actions, self.bc.probabilities(state, frequency=mode == "frequency")
            q, _ = self.q_values(state)
            if mode == "P3":
                p = kl_policy(q, self.bc.probabilities(state), self.bc.support(state), tau)
            else:
                p = np.zeros(len(q))
                p[int(np.argmax(q))] = 1
            return self.bc.actions, p
        return choose


def paired_summary(candidate: Rollouts, control: Rollouts):
    """Conditional MC error, not game/bootstrap or causal inference.

    Supply [rollouts] for ONE fixed PA, or [PAstarts,rollouts] for a fixed panel.
    The latter stratifies MC variance by PA; between-PA variation belongs in a
    separate game/PA sampling analysis, not in simulation MC standard error.
    """
    if (candidate.values.shape != control.values.shape or candidate.values.ndim not in (1, 2)
            or candidate.values.shape[-1] < 2):
        raise ValueError("at least two matched simulation trajectories required")
    delta = candidate.values - control.values
    grouped = delta[None, :] if delta.ndim == 1 else delta
    mc_se = np.sqrt(np.sum(grouped.var(axis=1, ddof=1)/grouped.shape[1]))/len(grouped)
    return {"mean_delta_we": float(delta.mean()),
            "paired_mc_se": float(mc_se),
            "truncation_delta_lower": float((candidate.lower-control.upper).mean()),
            "truncation_delta_upper": float((candidate.upper-control.lower).mean()),
            "candidate_truncation_rate": float(candidate.truncated.mean()),
            "control_truncation_rate": float(control.truncated.mean()),
            "model_internal_only": True, "observational_ope": None, "causal_effect": None}
