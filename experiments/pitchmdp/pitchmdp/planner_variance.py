"""EXP-STAB-02 (track A1): Rao-Blackwellized first-step planning Q, frozen-history DP and the metrics.

``RaoBlackwellFirstStep`` is an ADDITIONAL diagnostic planner next to the registered
``RolloutImprovement`` (which is not touched): for the first pitch of every candidate action it
replaces the drawn outcome by the exact expectation over the 10 outcome classes of the planner's own
conditional law (the bound G0 five-member mean probabilities given the sampled delivery). Everything
else is the registered search: the same ``planning_seed`` key, the same ``rng.random((S, cap, 3))``
tensor shared by every candidate action (CRN), the delivery index ``int(U[i,0,1]*400)``, BC
continuation through the registered ``rollouts`` with the parent sample's ``U[i,1:,:]``, cap 12 in
total and the same cutoff. With the same seed it is the conditional expectation of the registered
estimator given the delivery and continuation uniforms (same estimand, no larger variance), and its
selected-outcome path (the outcome ``U[i,0,2]`` picks) reproduces the registered sample exactly.

Nothing here is a registered policy identity, reads an outcome or chooses data. This module is
imported only by ``scripts/diagnose_planner_variance.py`` and the synthetic tests, so the
S2-certified component identity (``policy_identity.project_sources``) does not change.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from . import planner_variance_registry as registry
from .planner import OUTCOMES, TERMINALS, solve_pa
from .rollout_policy import (PAState, PastPitch, RowBudget, Step, _draw, _we, kl_policy, planning_seed,
                             probabilities, rollouts)

FIRST_PITCH_CHUNK = 6
ROOT_DRAWS = 400
NODE_STRIDE = 10  # frozen-history DP: delivery indices 0, 10, ..., 390 at non-root nodes
START = '<START>'


class SearchIntegrityError(ValueError):
    """The RB search could not be traced or reproduced exactly (stop; never repair)."""


def terminal_event(state, outcome):
    """PA-end event of ``outcome`` in ``state``; JointSimulator.step's rule (rollout_policy.py:245-247)."""
    return ("walk" if outcome == "ball" and state.balls == 3 else
            "strikeout" if outcome == "strike" and state.strikes == 2 else
            outcome if outcome in OUTCOMES[3:] else None)


def child_state(state, action, physical, outcome):
    """Generated child after a non-terminal outcome; JointSimulator.step's child (rollout_policy.py:253-256)."""
    token = PastPitch(action, physical, outcome, state.balls, state.strikes)
    return PAState(state.balls + int(outcome == "ball"),
                   min(2, state.strikes + int(outcome in ("strike", "foul"))),
                   state.pitcher, state.batter_side, state.history + (token,), state.context_key)


def conditional_rows(simulator, states, actions, delivery_uniforms):
    """Front half of JointSimulator.step (rollout_policy.py:237-241): selected deliveries, one row-budget
    charge per row, the predictor's conditional law [N,10] given each selected delivery."""
    delivery = np.stack([simulator.pool(s, a).values[int(x*400)]
                         for s, a, x in zip(states, actions, delivery_uniforms)])
    simulator.budget.consume(len(states))
    p = probabilities(simulator.predictor(states, actions, delivery), 10)
    if p.shape != (len(states), 10):
        raise ValueError("predictor must return [N,10]")
    return delivery, p


def _uniforms(values, shape):
    u = np.asarray(values, dtype=float)
    if u.shape != shape or not np.isfinite(u).all() or (u < 0).any() or (u >= 1).any():
        raise ValueError("uniforms must be finite in [0,1) with shape " + str(shape))
    return u


def expand_first_pitch(simulator, state, actions, delivery_uniforms):
    """Every outcome of each first pitch ``actions[r]`` from ``state`` with its sampled delivery.

    Returns (deliveries [N,F], law [N,10], terminal values [N,10] (NaN where the PA continues),
    children [N][10] (child state or None)). Same deliveries, budget charge and children as
    ``JointSimulator.step``; the predictor sees the rows in the given order (one call)."""
    u = _uniforms(delivery_uniforms, (len(actions),))
    if not len(actions):
        return np.zeros((0, 0)), np.zeros((0, 10)), np.zeros((0, 10)), []
    delivery, p = conditional_rows(simulator, [state] * len(actions), actions, u)
    values = np.full((len(actions), 10), np.nan)
    children = []
    for r, (action, d) in enumerate(zip(actions, delivery)):
        physical = tuple(float(v) for v in d)
        kids = []
        for k, outcome in enumerate(OUTCOMES):
            event = terminal_event(state, outcome)
            if event:
                values[r, k] = _we(simulator.terminal(state, event))
                kids.append(None)
            else:
                kids.append(child_state(state, action, physical, outcome))
        children.append(kids)
    return delivery, p, values, children


class _TracingSimulator:
    """Diagnostic twin of ``JointSimulator.step`` on a scratch row budget.

    It goes through the bound pool, predictor and terminal of ``simulator`` and records, for every
    continuation trajectory, the smallest distance between an outcome uniform and the outcome CDF
    (the boundary margin of a possible float-rounding flip). Used only to explain a reconstruction
    mismatch; the RB values it produces must equal the untraced search exactly."""
    def __init__(self, simulator):
        self.pool, self.predictor, self.terminal = simulator.pool, simulator.predictor, simulator.terminal
        self.budget = RowBudget(sys.maxsize, seed_count=simulator.budget.seed_count)
        self.current, self.active, self.margin = [], [], np.zeros(0)

    def begin(self, states):
        self.current, self.active = list(states), list(range(len(states)))
        self.margin = np.full(len(states), np.inf)

    def step(self, states, actions, uniforms):
        if len(states) != len(actions):
            raise ValueError("one action per state required")
        u = _uniforms(uniforms, (len(states), 2))
        if not states:
            return []
        ids = self.active
        if len(ids) != len(states) or any(self.current[t] is not s for t, s in zip(ids, states)):
            raise SearchIntegrityError('traced continuation lost track of its trajectories')
        delivery, p = conditional_rows(self, states, actions, u[:, 0])
        result = []
        for t, s, a, d, probs, x in zip(ids, states, actions, delivery, p, u[:, 1]):
            self.margin[t] = min(self.margin[t], float(np.min(np.abs(np.cumsum(probs) - x))))
            outcome = OUTCOMES[_draw(probs, x)]
            event = terminal_event(s, outcome)
            physical = tuple(float(v) for v in d)
            if event:
                result.append(Step(None, _we(self.terminal(s, event)), a, physical, outcome))
            else:
                result.append(Step(child_state(s, a, physical, outcome), None, a, physical, outcome))
        for t, step in zip(ids, result):
            self.current[t] = step.state
        self.active = [t for t, step in zip(ids, result) if step.state is not None]
        return result


class RaoBlackwellFirstStep:
    """Q_RB(a) = (1/S) sum_i sum_o p(o | s0, a, d_i(a)) W_i(a, o) (design section 4.2).

    d_i(a) = pool(s0, a)[int(U_i[0,1]*400)]; W = terminal WE for a PA-ending outcome, else the value of a
    BC rollout from the child with U_i[1:, :] (shared by every action and every outcome child of sample
    i), cap ``pitch_cap`` pitches in total (first pitch + cap-1), same cutoff. Same constructor
    arguments and ``q_values`` return type as ``RolloutImprovement`` (support off the mask is -inf), so
    ``BoundComponents.verify(planner)`` checks its wiring. Not a registered policy identity."""
    def __init__(self, bc, simulator, cutoff, *, samples, pitch_cap, seed, first_pitch_chunk=FIRST_PITCH_CHUNK):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (samples, pitch_cap, first_pitch_chunk)):
            raise ValueError("positive integer samples/pitch_cap/first_pitch_chunk required")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("integer planning seed required")
        self.bc, self.simulator, self.cutoff = bc, simulator, cutoff
        self.samples, self.pitch_cap, self.seed = samples, pitch_cap, seed
        self.first_pitch_chunk = first_pitch_chunk
        self.diagnostics = {"decisions": 0, "first_pitch_rows": 0, "continuation_rollouts": 0,
                            "continuation_truncated": 0, "search_conditional_rows": 0}

    def baseline(self, state, depth=0):
        return self.bc.actions, self.bc.probabilities(state)

    def identity(self):
        return {"q": "RaoBlackwellFirstStep MC of the reference continuation",
                "first_step": "exact over the 10 outcomes given the sampled delivery",
                "samples": self.samples, "pitch_cap": self.pitch_cap, "seed": self.seed,
                "seed_rule": "planning_seed(state, support, seed) -> default_rng(key).random((samples, pitch_cap, 3)), "
                             "shared by every candidate action (CRN)",
                "first_pitch_chunk": self.first_pitch_chunk, "policy_identity_registered": False}

    def q_values(self, state):
        return self._search(state, self.simulator, self.simulator, count=True)

    def trace(self, state):
        """The same search with a tracing twin on a scratch budget; adds per-sample selected-path margins."""
        tracer = _TracingSimulator(self.simulator)
        return self._search(state, tracer, tracer, count=False)

    def _search(self, state, first, continuation, *, count):
        names = [a for a, ok in zip(self.bc.actions, self.bc.support(state)) if ok]
        if not names:
            raise ValueError("no TRAIN-supported actions; abstain")
        key = planning_seed(state, names, self.seed)
        S, cap, n = self.samples, self.pitch_cap, len(names)
        common = np.random.default_rng(key).random((S, cap, 3))
        before = first.budget.conditional_rows
        law = np.empty((n, S, 10))
        values = np.full((n, S, 10), np.nan)
        kids = [[None] * S for _ in range(n)]
        for low in range(0, S, self.first_pitch_chunk):
            cells = [(j, i) for j in range(n) for i in range(low, min(S, low + self.first_pitch_chunk))]
            _, p, v, c = expand_first_pitch(first, state, [names[j] for j, _ in cells],
                                            common[[i for _, i in cells], 0, 1])
            for (j, i), p_row, v_row, c_row in zip(cells, p, v, c):
                law[j, i], values[j, i], kids[j][i] = p_row, v_row, c_row
        order = [(j, i, k) for j in range(n) for i in range(S) for k in range(10) if kids[j][i][k] is not None]
        truncated = np.zeros((n, S, 10), dtype=bool)
        position = {}
        if order:
            children = [kids[j][i][k] for j, i, k in order]
            if cap > 1:
                if hasattr(continuation, 'begin'):
                    continuation.begin(children)
                result = rollouts(continuation, children, self.baseline, common[[i for _, i, _ in order], 1:, :],
                                  self.cutoff)
                child_values, child_truncated = result.values, result.truncated
            else:
                child_values = np.array([_we(self.cutoff(child)) for child in children])
                child_truncated = np.ones(len(children), dtype=bool)
            for slot, ((j, i, k), value, cut) in enumerate(zip(order, child_values, child_truncated)):
                values[j, i, k], truncated[j, i, k] = value, cut
                position[(j, i, k)] = slot
        if not np.isfinite(values).all():
            raise SearchIntegrityError('RB outcome value missing')
        weighted = law * values
        z = np.sum(weighted, axis=-1)
        terminal = np.array([[[kids[j][i][k] is None for k in range(10)] for i in range(S)] for j in range(n)])
        first_outcome = np.array([[_draw(law[j, i], common[i, 0, 2]) for i in range(S)] for j in range(n)], dtype=int)
        selected = np.take_along_axis(values, first_outcome[..., None], axis=-1)[..., 0]
        width = len(self.bc.actions)
        q = np.full(width, -np.inf)
        se = np.full(width, np.nan)
        full = lambda fill, dtype=float: np.full((width, S), fill, dtype=dtype)  # noqa: E731
        sample_values, selected_values = full(np.nan), full(np.nan)
        terminal_term, continuation_term = full(np.nan), full(np.nan)
        outcome_index, outcome_terminal = full(-1, int), full(False, bool)
        margins = full(np.inf) if isinstance(continuation, _TracingSimulator) else None
        for j, action in enumerate(names):
            r = self.bc.actions.index(action)
            sample_values[r] = z[j]
            q[r] = z[j].mean()
            se[r] = z[j].std(ddof=1) / np.sqrt(S) if S > 1 else np.nan
            selected_values[r] = selected[j]
            terminal_term[r] = np.sum(np.where(terminal[j], weighted[j], 0.), axis=-1)
            continuation_term[r] = np.sum(np.where(terminal[j], 0., weighted[j]), axis=-1)
            outcome_index[r] = first_outcome[j]
            outcome_terminal[r] = terminal[j, np.arange(S), first_outcome[j]]
            if margins is not None:
                for i in range(S):
                    slot = position.get((j, i, int(first_outcome[j, i])))
                    if slot is not None and cap > 1:
                        margins[r, i] = continuation.margin[slot]
        rows = first.budget.conditional_rows - before
        diagnostics = {"mc_se": se, "sample_values": sample_values, "selected_values": selected_values,
                       "terminal_term": terminal_term, "continuation_term": continuation_term,
                       "first_outcome": outcome_index, "first_outcome_terminal": outcome_terminal,
                       "rollouts": n * S, "children": len(order), "truncated_children": int(truncated.sum()),
                       "truncated_weight": float(np.sum(law * truncated)), "rows": int(rows)}
        if margins is not None:
            diagnostics["selected_margin"] = margins
        if count:
            self.diagnostics["decisions"] += 1
            self.diagnostics["first_pitch_rows"] += n * S
            self.diagnostics["continuation_rollouts"] += len(order)
            self.diagnostics["continuation_truncated"] += int(truncated.sum())
            self.diagnostics["search_conditional_rows"] += int(rows)
        return q, diagnostics


class FrozenHistoryDP:
    """Optional FH-DP arm (design section 4.4): exact count DP with the predictor history frozen at H0.

    Every node (b, s) reachable from the decision count uses the real pre-decision history H0 with the
    node count; its outcome law is the mean over deliveries (all 400 at the root, indices 0, 10, ..., 390
    elsewhere); BC is evaluated per (count, previous type); ``planner.solve_pa`` with the BC baseline;
    Q_DP(a) = myopic_q_values[b0, s0, prev0, a]. No pitch cap and no cutoff: a different estimand from
    the capped MC planners. Deterministic (no seed). ``cutoff`` is kept only for the wiring check."""
    def __init__(self, bc, simulator, cutoff):
        self.bc, self.simulator, self.cutoff = bc, simulator, cutoff
        self.diagnostics = {"decisions": 0, "search_conditional_rows": 0}

    def identity(self):
        return {"q": "FrozenHistoryDP: solve_pa over reachable counts, predictor history frozen at the decision",
                "root_draws": ROOT_DRAWS, "node_draws": list(range(0, ROOT_DRAWS, NODE_STRIDE)),
                "pitch_cap": None, "seed": None, "policy_identity_registered": False}

    def _bc_state(self, state, balls, strikes, previous, own):
        if own:  # the decision's own previous pitch (or the PA start): the real history
            return PAState(balls, strikes, state.pitcher, state.batter_side, state.history, state.context_key)
        token = PastPitch(previous, (0.0,), 'unknown', 0, 0)  # BC reads only the previous type (rollout_policy.py:84-86)
        return PAState(balls, strikes, state.pitcher, state.batter_side, (token,), state.context_key)

    def q_values(self, state):
        mask = np.asarray(self.bc.support(state), dtype=bool)
        names = [a for a, ok in zip(self.bc.actions, mask) if ok]
        if not names:
            raise ValueError("no TRAIN-supported actions; abstain")
        n, root = len(names), (state.balls, state.strikes)
        counts = [(b, s) for b in range(state.balls, 4) for s in range(state.strikes, 3)]
        nodes = {c: PAState(c[0], c[1], state.pitcher, state.batter_side, state.history, state.context_key)
                 for c in counts}
        rows_s, rows_a, rows_d, spans = [], [], [], {}
        for c in counts:
            draws = range(ROOT_DRAWS) if c == root else range(0, ROOT_DRAWS, NODE_STRIDE)
            for j, action in enumerate(names):
                pool = self.simulator.pool(nodes[c], action).values
                start = len(rows_s)
                for index in draws:
                    rows_s.append(nodes[c])
                    rows_a.append(action)
                    rows_d.append(pool[index])
                spans[(c, j)] = (start, len(rows_s))
        before = self.simulator.budget.conditional_rows
        self.simulator.budget.consume(len(rows_s))
        p = probabilities(self.simulator.predictor(rows_s, rows_a, np.stack(rows_d)), 10)
        if p.shape != (len(rows_s), 10):
            raise ValueError("predictor must return [N,10]")
        mean = {key: p[lo:hi].mean(axis=0) for key, (lo, hi) in spans.items()}
        own = state.history[-1].action if state.history else START
        previous = list(names) + ([own] if own not in names else [])
        own_index = previous.index(own)
        law = np.empty((4, 3, len(previous), n, 10))
        baseline = np.empty((4, 3, len(previous), n))
        reachable = set(counts)
        for b in range(4):
            for s in range(3):
                c = (b, s) if (b, s) in reachable else root
                for j in range(n):
                    law[b, s, :, j] = mean[(c, j)]
                for k, prev in enumerate(previous):
                    bc_state = self._bc_state(state, c[0], c[1], prev, k == own_index)
                    baseline[b, s, k] = np.asarray(self.bc.probabilities(bc_state))[mask]
        terminal = {name: _we(self.simulator.terminal(state, name)) for name in TERMINALS}
        plan = solve_pa(law, terminal, [previous.index(a) for a in names], baseline)
        q = np.full(len(self.bc.actions), -np.inf)
        q[np.flatnonzero(mask)] = plan.myopic_q_values[root[0], root[1], own_index]
        rows = self.simulator.budget.conditional_rows - before
        self.diagnostics["decisions"] += 1
        self.diagnostics["search_conditional_rows"] += int(rows)
        return q, {"rows": int(rows), "nodes": len(counts), "previous_slots": len(previous),
                   "solve": plan.diagnostics}


# ---------------------------------------------------------------- per-decision metrics

def law(q, reference, mask, tau):
    """P3 law of a planning Q exactly as diagnose_policy_predictor_swap.law / policy_tau recompute it."""
    mask = np.asarray(mask, dtype=bool)
    return kl_policy(np.where(mask, q, -np.inf), reference, mask, tau)


def pair(law_a, law_b):
    """(top-1 of the two laws agrees, total variation)."""
    return bool(np.argmax(law_a) == np.argmax(law_b)), .5 * float(np.abs(np.asarray(law_a) - np.asarray(law_b)).sum())


def kl_to_reference(p, reference, mask):
    p, reference, mask = np.asarray(p), np.asarray(reference), np.asarray(mask, dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.sum(p[mask] * np.log(p[mask] / reference[mask]), where=p[mask] > 0))


def top_two(p):
    order = np.argsort(-np.asarray(p), kind='stable')
    return int(order[0]), int(order[1])


def effective_gap(q, reference, mask, tau):
    """g_eff = tau*log(pi[j1]/pi[j2]) = (Q[j1]-Q[j2]) + tau*log(BC[j1]/BC[j2]) for the top-2 law actions.
    None when the support has one action."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return None
    j1, j2 = top_two(law(q, reference, mask, tau))
    g = float((q[j1] - q[j2]) + tau * np.log(reference[j1] / reference[j2]))
    if g < 0:  # j1 is the law's argmax, so g >= 0 up to float rounding (a tie)
        if g < -1e-12:
            raise ValueError('effective gap of the law top-2 is negative')
        g = 0.0
    return g


def q_gap(q, mask):
    """Raw Q gap of the two largest Q on the support (None when the support has one action)."""
    values = np.sort(np.asarray(q, dtype=float)[np.asarray(mask, dtype=bool)])[::-1]
    return None if len(values) < 2 else float(values[0] - values[1])


def gap_bin(g, edges):
    """Index k of [edges[k], edges[k+1]) (the last bin open); -1 for a one-action support (g None)."""
    if g is None:
        return -1
    if not np.isfinite(g) or g < edges[0]:
        raise ValueError('a gap is a finite nonnegative number')
    return int(np.searchsorted(np.asarray(edges[1:], dtype=float), g, side='right'))


def paired_difference_variance(sample_values, mask, anchor):
    """v = sum_{a != anchor} Var_i[Y_i(a) - Y_i(anchor)] / S over the support (ddof 1): the variance of the
    Q contrasts against the common anchor action (design section 5-6)."""
    values = np.asarray(sample_values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    S = values.shape[1]
    if S < 2:
        return float('nan')
    rows = [values[a] - values[anchor] for a in np.flatnonzero(mask) if a != anchor]
    return float(sum(np.var(d, ddof=1) for d in rows) / S) if rows else 0.0


def residual_terms(diagnostics, mask):
    """RB residual (design section 5-9): sums over the support of Var_i(T), Var_i(C), Cov_i(T, C), Var_i(Z)."""
    mask = np.asarray(mask, dtype=bool)
    t, c, z = (np.asarray(diagnostics[k])[mask] for k in ('terminal_term', 'continuation_term', 'sample_values'))
    if t.shape[1] < 2:
        return {'var_terminal': float('nan'), 'var_continuation': float('nan'), 'cov': float('nan'),
                'var_sample': float('nan')}
    cov = sum(float(np.cov(t[k], c[k], ddof=1)[0, 1]) for k in range(len(t)))
    return {'var_terminal': float(np.var(t, axis=1, ddof=1).sum()),
            'var_continuation': float(np.var(c, axis=1, ddof=1).sum()), 'cov': cov,
            'var_sample': float(np.var(z, axis=1, ddof=1).sum())}


def max_finite(values, mask):
    v = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else float('nan')


# ---------------------------------------------------------------- A0 first-step evidence (observation only)

class FirstStepObserver:
    """Delegating view of a bound ``JointSimulator``: it exposes the same pool, predictor, terminal and budget
    objects (so ``BoundComponents.verify`` still sees the bound wiring) and every step runs the registered
    ``JointSimulator.step`` unchanged; the first step after ``arm()`` (the depth-0 batch of one search, rows in
    action-major, sample-minor order) is recorded as the outcome index of every row."""
    def __init__(self, simulator):
        self.inner = simulator
        self.pool, self.predictor, self.terminal = simulator.pool, simulator.predictor, simulator.terminal
        self.budget = simulator.budget
        self.armed, self.first = False, None

    def arm(self):
        self.armed, self.first = True, None

    def step(self, states, actions, uniforms):
        steps = self.inner.step(states, actions, uniforms)
        if self.armed:
            self.armed = False
            self.first = [OUTCOMES.index(step.outcome) for step in steps]
        return steps


class A0Witness:
    """First-pitch outcome, sample values and planner time of every search of one registered
    ``RolloutImprovement`` (the runtime's own seed-0 planner or a fresh A0 planner), captured during the original
    call: the planner's simulator is replaced by a ``FirstStepObserver`` over the same bound components and its
    ``q_values`` by a recording call (instance attributes only; the registered classes and sources are unchanged
    and no search is repeated)."""
    def __init__(self, improvement):
        self.improvement = improvement
        self.observer = FirstStepObserver(improvement.simulator)
        improvement.simulator = self.observer
        self._search = improvement.q_values
        improvement.q_values = self._record
        self.last = None

    def _record(self, state):
        self.observer.arm()
        started = time.perf_counter()
        q, diagnostics = self._search(state)
        elapsed = (time.perf_counter() - started) * 1000.
        first, self.observer.armed = self.observer.first, False
        self.last = (state, q, diagnostics, first, elapsed)
        return q, diagnostics

    def discard(self):
        self.last = None

    def take(self, state):
        """Evidence of the search just made for ``state``: {'q', 'first' [V,S] (-1 off the support),
        'samples' [V,S], 'ms'}."""
        if self.last is None or self.last[0] is not state:
            raise SearchIntegrityError('no A0 search was observed for this decision')
        _, q, diagnostics, first, ms = self.last
        self.last = None
        planner = self.improvement
        names = [a for a, ok in zip(planner.bc.actions, planner.bc.support(state)) if ok]
        S = planner.samples
        if first is None or len(first) != len(names) * S:
            raise SearchIntegrityError('the A0 search left no complete first-step evidence (cache hit?)')
        out = np.full((len(planner.bc.actions), S), -1, dtype=np.int8)
        for j, action in enumerate(names):
            out[planner.bc.actions.index(action)] = first[j * S:(j + 1) * S]
        return {'q': np.array(q, dtype=np.float64), 'first': out,
                'samples': np.array(diagnostics['sample_values'], dtype=np.float64), 'ms': ms}


# ---------------------------------------------------------------- reconstruction (integrity)

def selected_q(diagnostics, samples):
    """Mean over the first ``samples`` selected-outcome values per action, computed exactly like the
    registered estimator's ``values.mean()`` over one action's contiguous sample row."""
    values = diagnostics['selected_values']
    out = np.full(len(values), -np.inf)
    for r in range(len(values)):
        row = np.ascontiguousarray(values[r, :samples])
        if np.isfinite(row).all():
            out[r] = row.mean()
    return out


def reconstruction_compare(diagnostics, a0, mask, samples):
    """Initial comparison (no tracing) of the first ``samples`` RB selected-outcome paths with the A0 search of
    the same seed. ``a0``: {'q', 'first', 'samples'} captured during the A0 call. The A0 evidence must reproduce
    the A0 Q exactly; a path whose RB first-pitch outcome index differs from A0's is a first-pitch flip (zero
    allowed); a path with the same first outcome but a different value is a differing (continuation) path."""
    mask = np.asarray(mask, dtype=bool)
    rows = [int(r) for r in np.flatnonzero(mask)]
    a0_first, a0_samples = np.asarray(a0['first']), np.asarray(a0['samples'], dtype=np.float64)
    a0_q = np.asarray(a0['q'], dtype=np.float64)
    for r in rows:
        if not np.ascontiguousarray(a0_samples[r, :samples]).mean() == a0_q[r]:
            raise SearchIntegrityError('the captured A0 evidence does not reproduce the A0 Q')
    rb_first, rb_values = diagnostics['first_outcome'], diagnostics['selected_values']
    flips = [(r, i) for r in rows for i in range(samples) if int(rb_first[r, i]) != int(a0_first[r, i])]
    differing = [(r, i) for r in rows for i in range(samples)
                 if int(rb_first[r, i]) == int(a0_first[r, i]) and not rb_values[r, i] == a0_samples[r, i]]
    return {'mismatch': bool(flips or differing), 'first_pitch_flips': flips, 'differing_paths': differing,
            'path_margins': None, 'margin': None}


def classify_paths(compare, diagnostics, trace):
    """Boundary margins of exactly the differing paths, from the traced re-run of the same RB search (which must
    reproduce the untraced search exactly). A path whose RB first outcome ends the PA has no continuation draw:
    margin inf."""
    for key in ('sample_values', 'selected_values', 'first_outcome'):
        if not np.array_equal(trace[key], diagnostics[key], equal_nan=key != 'first_outcome'):
            raise SearchIntegrityError('the traced RB search does not reproduce the RB search')
    margins = [float(np.inf if trace['first_outcome_terminal'][r, i] else trace['selected_margin'][r, i])
               for r, i in compare['differing_paths']]
    return {**compare, 'path_margins': margins, 'margin': max(margins) if margins else None}


class ReconstructionTally:
    """Registered reconstruction rule: zero first-pitch flips and no differing path with a boundary margin above
    ``margin_max`` (immediate stop); at most ``share_max`` of the compared decisions with a differing path
    (checked by ``finish``)."""
    def __init__(self, margin_max, share_max, first_pitch_flips_max=0):
        self.margin_max, self.share_max, self.flips_max = margin_max, share_max, first_pitch_flips_max
        self.decisions, self.mismatched, self.flips, self.margins, self.records = 0, 0, 0, [], []

    def add(self, request_id, checks):
        """``checks``: {label: reconstruction_compare/classify_paths result} of one decision."""
        self.decisions += 1
        flips = sum(len(c['first_pitch_flips']) for c in checks.values())
        if flips:
            self.flips += flips
            self.records.append({'request_id': request_id, 'first_pitch_flips': flips})
            if self.flips > self.flips_max:
                raise SearchIntegrityError(f'first-pitch outcome flip in the reconstruction of {request_id}')
        bad = {label: c for label, c in checks.items() if c['differing_paths']}
        if not bad:
            return
        if any(c['margin'] is None for c in bad.values()):
            raise SearchIntegrityError(f'differing reconstruction paths of {request_id} were not traced')
        self.mismatched += 1
        margin = max(c['margin'] for c in bad.values())
        self.margins.append(margin)
        self.records.append({'request_id': request_id, 'checks': sorted(bad),
                             'paths': {label: len(c['differing_paths']) for label, c in bad.items()},
                             'margin': None if np.isinf(margin) else margin})
        if margin > self.margin_max:
            raise SearchIntegrityError(f'reconstruction mismatch with margin {margin} > {self.margin_max} at {request_id}')

    def finish(self):
        share = self.mismatched / self.decisions if self.decisions else 0.0
        if share > self.share_max:
            raise SearchIntegrityError(f'reconstruction mismatches {self.mismatched}/{self.decisions} exceed '
                                       f'{self.share_max:.1%}')
        return {'decisions': self.decisions, 'mismatched_decisions': self.mismatched, 'share': share,
                'first_pitch_flips': self.flips,
                'max_margin': max([m for m in self.margins if np.isfinite(m)], default=None),
                'mismatches': self.records}


# ---------------------------------------------------------------- whole-game bootstrap

class GameBootstrap:
    """EXP-STAB-01 ``summarize`` draw table: ``np.unique(games)`` order, ``default_rng(seed).integers(0, G,
    (draws, G))``, ratio of per-game sums; paired differences use the same draws. A replicate whose
    denominator is 0 leaves the ratio undefined (NaN); intervals use the valid replicates."""
    def __init__(self, games, *, draws, seed):
        self.unique, self.index = np.unique(np.asarray(games), return_inverse=True)
        self.picks = np.random.default_rng(seed).integers(0, len(self.unique), (draws, len(self.unique)))
        self.weight = np.bincount(self.index, minlength=len(self.unique)).astype(float)
        self.n = len(self.index)

    def _sums(self, values):
        return np.bincount(self.index, weights=np.asarray(values, dtype=float), minlength=len(self.unique))

    def sums(self, values):
        """Per-replicate sums of ``values`` (to compose a statistic on the same draws)."""
        return self._sums(values)[self.picks].sum(axis=1)

    def ci(self, values):
        boot = self._sums(values)[self.picks].sum(axis=1) / self.weight[self.picks].sum(axis=1)
        return [float(v) for v in np.quantile(boot, (.025, .975))]

    def subset(self, values, include):
        """Point value, interval and valid-replicate count of the mean of ``values`` over ``include``."""
        include = np.asarray(include, dtype=bool)
        if not include.any():
            return {'decisions': 0, 'value': None, 'ci95': None, 'valid_replicates': 0}
        values = np.where(include, np.asarray(values, dtype=float), 0.)
        num = self._sums(values)[self.picks].sum(axis=1)
        den = self._sums(include.astype(float))[self.picks].sum(axis=1)
        valid = den > 0
        ratio = num[valid] / den[valid]
        return {'decisions': int(include.sum()), 'value': float(values[include].mean()),
                'ci95': [float(v) for v in np.quantile(ratio, (.025, .975))], 'valid_replicates': int(valid.sum())}

    def ratio(self, numerator, denominator):
        """Point value and interval of sum(numerator)/sum(denominator) (NaN replicates dropped)."""
        num = self._sums(numerator)[self.picks].sum(axis=1)
        den = self._sums(denominator)[self.picks].sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = num / den
        boot = boot[np.isfinite(boot)]
        total = float(np.sum(denominator))
        return {'value': float(np.sum(numerator)) / total if total else None,
                'ci95': [float(v) for v in np.quantile(boot, (.025, .975))] if len(boot) else None,
                'valid_replicates': int(len(boot))}


def primary(top1_same, tv, boot):
    """Registered primary metrics of one seed-0 vs seed-5 pair (design section 5-1, 5-2)."""
    same = np.asarray(top1_same, dtype=float)
    tv = np.asarray(tv, dtype=float)
    above = registry.TV_SHARE_ABOVE
    return {'decisions': len(tv), 'games': len(boot.unique), 'top1_agreement': float(same.mean()),
            'top1_agreement_ci95': boot.ci(same), 'mean_tv': float(tv.mean()), 'mean_tv_ci95': boot.ci(tv),
            'tv_quantiles': {str(q): float(np.quantile(tv, q)) for q in registry.TV_QUANTILES},
            f'share_tv_above_{above}': float((tv > above).mean())}


def paired_gain(a, b, boot):
    """Mean and interval of the per-decision difference a - b on the same draws."""
    gain = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {'mean': float(gain.mean()), 'ci95': boot.ci(gain)}
