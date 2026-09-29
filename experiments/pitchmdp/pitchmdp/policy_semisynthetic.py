"""V2/V3 model-world semi-synthetic OPE checks (COOP-018).

A declared simulator world W: the bound G0 predictor, the frozen WE terminal/cutoff, and a delivery
law for EVERY action the logging law can choose (D89 V2 precondition). The world uses the action's
exact TRAIN pool when one exists (the policy's own pool); for a logging-only action without one it
uses the registered world rule (D-10): ``'fallback'`` = JointDelivery's league fallback draws, or
``'refuse'`` = V2 cannot run for that start (reported, never renormalised onto the mask).

Logged PAs are generated in W from registered PA starts under a KNOWN law, submitted to the same
candidate runtime (ledger), and the sequential DR estimate is compared with the Monte Carlo truth of
each policy in W. V2 generates under the full-vocabulary TRAIN BC (the estimator's pi_b_hat), so any
gap is implementation error or MC noise; V3 generates under another law while the estimator keeps
pi_b_hat (logging-law transport sensitivity). Model-internal only: no observed outcome, no causal claim.
"""
from __future__ import annotations

import numpy as np

from .policy_artifacts import _require
from .policy_estimator import estimate
from .policy_runtime import DecisionRequest
from .rollout_policy import DeliveryPool, JointSimulator, _draw, rollouts

WORLD_RULES = ('fallback', 'refuse')


class WorldRefused(Exception):
    """The registered world rule cannot deliver an action the logging law chose."""


def world_pool(components, rule):
    """Delivery law of W: the policy's exact pool, else the registered rule for logging-only actions."""
    _require(rule in WORLD_RULES, 'registered V2 world rule required (D-10)')
    inputs = components.inputs
    fallback = DeliveryPool(inputs.delivery.fallback, 'train', inputs.source_hash, 'world-league-fallback')

    def pool(state, action):
        try:
            return inputs.pool(state, action)
        except ValueError:
            if rule == 'refuse' or action not in inputs.type_map:
                raise WorldRefused(f'no world delivery for logging action {action!r}')
            return fallback
    return pool


def generate_pa(simulator, start, law, rng, cap):
    """One logged PA in W. Returns (states, actions, terminal WE or None if truncated, last state)."""
    state, states, actions = start, [], []
    for _ in range(cap):
        names, p = law(state)
        action = names[_draw(np.asarray(p, dtype=np.float64), rng.random())]
        step, = simulator.step([state], [action], rng.random((1, 2)))
        states.append(state)
        actions.append(action)
        if step.state is None:
            return states, actions, step.value, None
        state = step.state
    return states, actions, None, state


def policy_truth(simulator, start, policy, cutoff, *, rollouts_per_start, cap, seed):
    """MC value of a policy from ``start`` in W (same cap and cutoff tail as the generated logs)."""
    uniforms = np.random.default_rng(seed).random((rollouts_per_start, cap, 3))
    values = rollouts(simulator, [start] * rollouts_per_start, policy, uniforms, cutoff).values
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def run_world(runtime, components, starts, *, law, rule, logs_per_start, cap, truth_rollouts, seed, draws, budget):
    """Generate logs in W, estimate by DR through the runtime ledger, and compare with the MC truth."""
    _require(runtime.components is components and runtime.improvement is not None, 'candidate runtime required')
    world = JointSimulator(world_pool(components, rule), components.g0, components.we.terminal, budget)
    rng = np.random.default_rng(seed)
    outcomes, refused = {}, []
    for i, start in enumerate(starts):
        for j in range(logs_per_start):
            pa_id = f'w{i}:{j}'
            try:
                states, actions, value, last = generate_pa(world, start, law, rng, cap)
            except WorldRefused as reason:
                refused.append((pa_id, str(reason)))
                continue
            for t, (state, action) in enumerate(zip(states, actions)):
                runtime.submit(DecisionRequest(f'{pa_id}:{t}', pa_id, t, state, action, runtime.sha256))
            reward = value if last is None else float(components.we.cutoff(last))
            outcomes[pa_id] = {'game': f'start{i}', 'reward': reward, 'truncated': last is not None}
    result, rows = estimate(runtime.ledger.decisions(), outcomes, draws=draws, seed=seed)
    improvement = runtime.improvement
    reference_policy = lambda state, depth: (improvement.bc.actions, improvement.bc.probabilities(state))
    candidate_policy = lambda state, depth: (improvement.bc.actions, runtime.candidate(state)[0])
    truth = []
    for i, start in enumerate(starts):
        row = {}
        for name, policy in (('candidate', candidate_policy), ('reference', reference_policy)):
            row[name], row[f'{name}_mc_se'] = policy_truth(world, start, policy, components.we.cutoff,
                                                            rollouts_per_start=truth_rollouts, cap=cap, seed=seed + i)
        truth.append(row)
    runtime.verify_components()
    by_start = {}
    for row in rows:
        if row['status'] == 'COMPLETE':
            by_start.setdefault(int(row['pa_id'][1:].split(':')[0]), []).append(row)
    comparison = []
    for i, row in enumerate(truth):
        found = by_start.get(i, [])
        estimates = {name: float(np.mean([r[name] for r in found])) if found else None for name in ('candidate', 'reference')}
        comparison.append({'start': i, 'logs': len(found), 'truth': row, 'dr_mean': estimates,
                           'dr_se': {name: float(np.std([r[name] for r in found], ddof=1) / np.sqrt(len(found)))
                                     if len(found) > 1 else None for name in ('candidate', 'reference')}})
    complete = [c for c in comparison if c['logs']]
    delta_truth = float(np.mean([c['truth']['candidate'] - c['truth']['reference'] for c in complete])) if complete else None
    delta_dr = float(np.mean([c['dr_mean']['candidate'] - c['dr_mean']['reference'] for c in complete])) if complete else None
    return {'world_rule': rule, 'starts': len(starts), 'logs_per_start': logs_per_start, 'world_refused_logs': len(refused),
            'estimate': result, 'per_start': comparison, 'delta_truth_mean': delta_truth, 'delta_dr_mean': delta_dr,
            'delta_gap': None if delta_truth is None else delta_dr - delta_truth,
            'interpretation': 'model-internal semi-synthetic check; no observed outcome; not a causal effect'}
