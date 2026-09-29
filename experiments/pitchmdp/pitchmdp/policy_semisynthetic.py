"""V2/V3 model-world semi-synthetic OPE checks (COOP-018, D93).

A declared simulator world W: the bound G0 predictor, the policy's own TRAIN delivery pools and
the frozen WE terminal/cutoff. D-10 (absorb_off_mask): a logging action outside the policy mask
M (the runtime's own ``reference.support``) ends the PA at once with an absorbing value; both
target laws put zero mass there, so rho = 0 for both policies and no later row, reward or
transition enters the DR value, the weights or the world truth (plain DR, no clipping or
self-normalisation; checked by an invariance test). No log is ever dropped. Logs are generated
from registered PA starts in E0 under a KNOWN law with one random substream per PA, submitted to
the same candidate runtime (ledger), estimated by the sequential DR and compared with the Monte
Carlo truth of each policy in W. V2 generates under the full-vocabulary TRAIN BC (the
estimator's pi_b_hat); V3 under another registered law (frequency or tempered BC restricted to
pi_b_hat > 0) while the estimator keeps pi_b_hat. Model-internal only: no observed outcome, no
causal claim; the world shares the policy's context rows, so profile staleness is not tested.
"""
from __future__ import annotations

import numpy as np

from .policy_artifacts import IntegrityError, _require
from .policy_estimator import estimate
from .policy_runtime import OUTSIDE_POLICY_SUPPORT, DecisionRequest
from .rollout_policy import JointSimulator, _draw, rollouts

WORLD_RULE = 'absorb_off_mask'


def tempered_law(bc, alpha):
    """V3: pi_b_hat^alpha renormalised on pi_b_hat > 0 (alpha registered)."""
    def law(state):
        p = bc.probabilities(state)
        w = np.where(p > 0, p ** alpha, 0.)
        return bc.actions, w / w.sum()
    return law


def frequency_law(bc):
    """V3: the pitcher's TRAIN frequency law on the same BC support (D89)."""
    return lambda state: (bc.actions, bc.probabilities(state, frequency=True))


def _law(law, runtime, state):
    names, p = law(state)
    p = np.asarray(p, dtype=np.float64)
    logging = runtime.bc.probabilities(state)
    _require(tuple(names) == tuple(runtime.bc.actions) and p.shape == logging.shape and np.isfinite(p).all()
             and (p >= 0).all() and abs(p.sum() - 1) <= 1e-9, 'generating law must be a probability vector over the BC')
    _require(not (p[logging == 0] > 0).any(), 'generating law must stay inside pi_b_hat > 0')
    return p


def generate_pa(runtime, simulator, start, law, rng, cap):
    """One logged PA in W: (states, actions, end, last_state). ``end`` is ('terminal', WE),
    ('absorbed', None) after an off-mask action, or ('cap', None) when the cap was reached."""
    state, states, actions = start, [], []
    for _ in range(cap):
        p = _law(law, runtime, state)
        action = runtime.bc.actions[_draw(p, rng.random())]
        states.append(state)
        actions.append(action)
        if not runtime.reference.support(state)[runtime.bc.actions.index(action)]:
            return states, actions, ('absorbed', None), state
        step, = simulator.step([state], [action], rng.random((1, 2)))
        if step.state is None:
            return states, actions, ('terminal', step.value), None
        state = step.state
    return states, actions, ('cap', None), state


def policy_truth(simulator, start, policies, cutoff, *, rollouts_per_start, cap, seed):
    """Paired MC truth of the policies from ``start`` in W (common random numbers, same cap and
    cutoff tail as the generated logs). Returns means, s.e. and the paired delta s.e."""
    uniforms = np.random.default_rng(seed).random((rollouts_per_start, cap, 3))
    values = {name: rollouts(simulator, [start] * rollouts_per_start, policy, uniforms, cutoff).values
              for name, policy in policies.items()}
    n = np.sqrt(rollouts_per_start)
    delta = values['candidate'] - values['reference']
    return {**{name: float(v.mean()) for name, v in values.items()},
            **{f'{name}_mc_se': float(v.std(ddof=1) / n) for name, v in values.items()},
            'delta': float(delta.mean()), 'delta_mc_se': float(delta.std(ddof=1) / n)}


def run_world(runtime, components, starts, *, law, law_identity, logs_per_start, cap, truth_rollouts, seed, draws,
              budget, tolerance=None, absorbing_value='cutoff'):
    """Generate logs in W, estimate by DR through the runtime ledger and compare with the truth.

    ``starts``: [(state, pitcher_hand)] registered PA starts; a start outside E0 is reported and
    not generated. Acceptance (M-6): the pooled delta gap's 95% interval contains 0 and |gap| is
    within the registered ``tolerance`` (None: report only). Per-start gaps are diagnostics.
    """
    _require(runtime.components is components and runtime.improvement is not None, 'candidate runtime required')
    world = JointSimulator(components.inputs.pool, components.g0, components.we.terminal, budget)
    pas, excluded, ends = {}, [], {'terminal': 0, 'absorbed': 0, 'cap': 0}
    for i, (start, hand) in enumerate(starts):
        inside, reason = runtime.start_population(start, hand)
        if not inside:
            excluded.append({'start': i, 'reason': reason})
            continue
        for j in range(logs_per_start):
            pa_id = f'w{i}:{j}'
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0, i, j]))
            states, actions, (end, value), last = generate_pa(runtime, world, start, law, rng, cap)
            rows = [runtime.submit(DecisionRequest(f'{pa_id}:{t}', pa_id, t, state, action, runtime.sha256, hand))
                    for t, (state, action) in enumerate(zip(states, actions))]
            _require(all(r['status'] in ('SUPPORTED', OUTSIDE_POLICY_SUPPORT) for r in rows),
                     'a generated in-E0 log was refused by the runtime')
            outside = [r['status'] == OUTSIDE_POLICY_SUPPORT for r in rows]
            if end == 'absorbed':
                _require(outside[-1] and not any(outside[:-1]) and rows[-1]['result']['rho_candidate'] == 0.
                         and rows[-1]['result']['rho_reference'] == 0., 'absorbed PA must end with one rho = 0 row')
                value = float(components.we.cutoff(last)) if absorbing_value == 'cutoff' else float(absorbing_value)
            else:
                _require(not any(outside), 'an off-mask action must absorb')
                if end == 'cap':
                    value = float(components.we.cutoff(last))
            ends[end] += 1
            pas[pa_id] = {'game': f'start{i}', 'in_population': True, 'reward': value, 'end_kind': end}
    if not pas:
        raise IntegrityError('no registered start is inside E0')
    result, rows = estimate(runtime.ledger.decisions(), pas, draws=draws, seed=seed, invalid_share_max=1.,
                            minimum={'games': 1, 'pa_starts': 1})
    improvement = runtime.improvement
    policies = {'reference': lambda state, depth: (improvement.bc.actions, improvement.bc.probabilities(state)),
                'candidate': lambda state, depth: (improvement.bc.actions, runtime.candidate(state)[0])}
    by_start = {}
    for row in rows:
        by_start.setdefault(int(row['pa_id'][1:].split(':')[0]), []).append(row)
    comparison = []
    for i, found in sorted(by_start.items()):
        truth = policy_truth(world, starts[i][0], policies, components.we.cutoff, rollouts_per_start=truth_rollouts,
                             cap=cap, seed=int(np.random.SeedSequence([seed, 1, i]).generate_state(1)[0]))
        deltas = np.array([r['delta'] for r in found])
        dr = {name: float(np.mean([r[name] for r in found])) for name in ('candidate', 'reference')}
        dr_se = {name: float(np.std([r[name] for r in found], ddof=1) / np.sqrt(len(found))) for name in dr}
        comparison.append({'start': i, 'logs': len(found), 'truth': truth, 'dr_mean': dr, 'dr_se': dr_se,
                           'delta_dr': float(deltas.mean()), 'delta_dr_se': float(deltas.std(ddof=1) / np.sqrt(len(found))),
                           'delta_gap': float(deltas.mean()) - truth['delta']})
    runtime.verify_components()
    gap = float(np.mean([c['delta_gap'] for c in comparison]))
    gap_se = float(np.sqrt(sum(c['delta_dr_se'] ** 2 + c['truth']['delta_mc_se'] ** 2 for c in comparison))
                   / len(comparison))
    interval = [gap - 1.96 * gap_se, gap + 1.96 * gap_se]
    contains = interval[0] <= 0 <= interval[1]
    return {'world': {'rule': WORLD_RULE, 'absorbing_value': absorbing_value, 'law': law_identity, 'seed': seed,
                      'cap': cap, 'logs_per_start': logs_per_start, 'truth_rollouts': truth_rollouts,
                      'estimator': 'D89 plain sequential DR; no clipping or self-normalisation',
                      'runtime_sha256': runtime.sha256},
            'starts': len(starts), 'excluded_starts': excluded, 'ends': ends, 'estimate': result, 'per_start': comparison,
            'delta_gap': gap, 'delta_gap_se': gap_se, 'delta_gap_ci95': interval,
            'accept': None if tolerance is None else bool(contains and abs(gap) <= tolerance),
            'interval_contains_zero': contains,
            'interpretation': 'model-internal semi-synthetic check; no observed outcome; not a causal effect'}
