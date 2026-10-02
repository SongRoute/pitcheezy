"""Draft runner: stability of the frozen G0 P3 law across planning seeds and sample counts.

Additive diagnostic in the EXP-P18-001 family (``diagnose_policy_seed_noise.py``).  The predictor
stays the bound G0, the reference law, support table, delivery pools, WE terminal/cutoff, tau and
the sealed June decision states are unchanged; only the planner's search seed and, optionally,
its sample count vary.  Every variant is compared with the sealed seed-0 law of the same decision
and the per-decision laws are saved so that an offline aggregation
(``diagnose_policy_stability_aggregate.py``) can compute cross-seed statistics without touching
data again.

Origin (disclosed): drafted 2026-10-03 after the EXP-P17-001 (D152) and EXP-P18-001 (D155)
results were read.  Its reading rules are P18-informed; they are not prospectively registered
relative to those observations.  No verdict is computed here and nothing in this file changes
the registered P17/P18 verdicts, the policy identity, the service policy or any gate.

No outcome, reward, WE result or events kind is read.  Nothing is registered as a policy identity.

Status: NOT executed on real data in the drafting session.  The pure parts (config check, variant
plan, per-decision comparison, nesting property) have synthetic tests; the data path needs a
PM-run smoke before any registered run.

``--smoke`` limits the games and the replay to check the wiring; it writes to an output ending
``-smoke`` and never writes the repository result.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.policy_artifacts import load_support_table, load_train_bc
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as prt
from pitchmdp.rollout_policy import JointSimulator, RolloutImprovement, RowBudget
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock
from run_ml_delivery_pool import git_commit, peak
import run_policy_validation as rpv
import diagnose_policy_predictor_swap as swap

PROTOCOL = 'policy_seed_stability_diagnostic_v1'
EXPERIMENT_ID = re.compile(r'^EXP-P\d+-\d{3}$')
SOURCES = (*swap.SOURCES, 'scripts/diagnose_policy_seed_noise.py', 'scripts/diagnose_policy_seed_stability.py')
SAME_AS_P18 = ('policy_registration', 'sealed_g0_run', 'settings', 'data_rules')
SEED_BASES = ('planning_main', 'planning_v2', 'extension')
LAW_WIDTH = 10
P18_CONDITIONAL_ROWS = 3_805_035   # measured: EXP-P18-001, one 6-sample variant over 40,702 decisions (results/EXP-P18-001.json)
P18_SAMPLES = 6
ROW_BUDGET_MARGIN = 1.2
require = swap.require


def row_budget_floor(plan):
    """Smallest per-variant row cap that leaves a margin over the measured P18 rows, scaled by sample count."""
    largest = max(v['samples'] for v in plan)
    return int(np.ceil(ROW_BUDGET_MARGIN * P18_CONDITIONAL_ROWS * largest / P18_SAMPLES))


def registered_seeds(policy):
    """Planning seeds the policy registration names: main seed and the planning_v2 set."""
    seeds = policy['seeds']
    return {'planning_main': int(seeds['planning_main']), 'planning_v2': [int(s) for s in seeds['planning_v2']]}


def _int(value):
    return type(value) is int


def variant_plan(variants, settings, seeds):
    """Validate the variant list against the sealed setting and the registered seeds.

    Returns the plan with ``shares_prefix_draws`` filled in: a variant that keeps the sealed seed
    and raises the sample count reuses the sealed common random numbers as a prefix (numpy draws
    sequentially), so its distance to the sealed law is a nested, not an independent, comparison.
    """
    require(isinstance(variants, list) and variants, 'At least one variant is required')
    require(_int(settings['seed']) and _int(settings['samples']) and settings['samples'] > 0, 'Sealed setting must be integer')
    sealed = (settings['seed'], settings['samples'])
    labels, pairs, plan = set(), set(), []
    for v in variants:
        label = v.get('label')
        require(isinstance(label, str) and re.fullmatch(r'[A-Za-z0-9_-]+', label or ''), f'Variant label invalid: {label!r}')
        require(label not in labels, f'Duplicate variant label: {label}')
        seed, samples, basis = v.get('seed'), v.get('samples', settings['samples']), v.get('seed_basis')
        require(_int(seed) and seed >= 0, f'{label}: planning seed must be a non-negative integer')
        require(_int(samples) and samples >= 1, f'{label}: samples must be a positive integer')
        require((seed, samples) != sealed, f'{label}: identical to the sealed setting (seed {sealed[0]}, samples {sealed[1]})')
        require((seed, samples) not in pairs, f'{label}: duplicate (seed, samples) pair')
        require(basis in SEED_BASES, f'{label}: seed_basis must be one of {SEED_BASES}')
        if basis == 'planning_main':
            require(seed == seeds['planning_main'], f'{label}: planning_main basis but seed differs from the registered main seed')
        elif basis == 'planning_v2':
            require(seed in seeds['planning_v2'], f'{label}: planning_v2 basis but seed {seed} is not a registered planning seed')
        else:
            require(seed != seeds['planning_main'] and seed not in seeds['planning_v2'],
                    f'{label}: extension basis but seed {seed} is a registered planning seed')
        labels.add(label)
        pairs.add((seed, samples))
        plan.append({'label': label, 'seed': seed, 'samples': samples, 'seed_basis': basis,
                     'kind': 'samples' if samples != settings['samples'] else 'seed',
                     'shares_prefix_draws': seed == settings['seed'] and samples > settings['samples']})
    return plan


def config_check(config, seeds, smoke=False):
    """A full run needs an assigned experiment id, REGISTERED status and enabled execution.

    A smoke (three games, no repository result) may run from a DRAFT with an UNASSIGNED id so the
    wiring can be checked before an id is assigned; everything else is validated the same way.
    """
    require(config.get('protocol') == PROTOCOL, 'Diagnostic settings differ from the protocol')
    experiment, status = config.get('experiment_id'), config.get('status')
    if smoke:
        require(experiment == 'UNASSIGNED' or (isinstance(experiment, str) and EXPERIMENT_ID.fullmatch(experiment)),
                f'Experiment id malformed: {experiment!r}')
        require(status in ('DRAFT', 'REGISTERED'), 'Config status must be DRAFT or REGISTERED for a smoke')
    else:
        require(isinstance(experiment, str) and EXPERIMENT_ID.fullmatch(experiment),
                f'Experiment id not assigned or malformed: {experiment!r} (drafts are not runnable)')
        require(status == 'REGISTERED', 'Config status must be REGISTERED (not DRAFT)')
    require(isinstance(config.get('origin'), str) and 'P18' in config['origin'] and 'after' in config['origin'],
            'The P18-informed origin must be disclosed in the config')
    for key in (*SAME_AS_P18, 'replay', 'row_budget', 'variants', 'bootstrap'):
        require(config.get(key), 'Registered input missing: ' + key)
    replay, bootstrap = config['replay'], config['bootstrap']
    require(_int(replay.get('decisions')) and replay['decisions'] >= 1 and replay.get('atol', 0) > 0, 'Replay rule required')
    require(_int(bootstrap.get('draws')) and bootstrap['draws'] > 0 and _int(bootstrap.get('seed')), 'Bootstrap draws and seed required')
    require(not config.get('verdict_rule') and not config.get('stop_rule'),
            'This diagnostic is descriptive: no verdict or stop rule may be attached')
    require(smoke or config['execution']['enabled'] is True, 'Execution not enabled in the registered config')
    plan = variant_plan(config['variants'], config['settings'], seeds)
    floor = row_budget_floor(plan)
    require(_int(config['row_budget']) and config['row_budget'] >= floor,
            f'row_budget must be an integer >= {floor} ({ROW_BUDGET_MARGIN}x the measured EXP-P18-001 rows scaled to the largest sample count)')
    return plan


def compare_decision(planners, state, q_sealed, reference, mask, tau):
    """Per-decision rows for every variant planner against the sealed planning Q.

    ``planners`` maps a variant label to an object with ``q_values(state) -> (q, diagnostics)``.
    Returns the sealed law row and one row per variant; laws are kept so an offline aggregation can
    compute cross-variant distances.
    """
    p_sealed = swap.law(q_sealed, reference, mask, tau)
    sealed_row = {'law': p_sealed, 'top1': int(np.argmax(p_sealed)), 'mask': np.asarray(mask, dtype=bool)}
    rows = {}
    for label, planner in planners.items():
        q, _ = planner.q_values(state)
        p = swap.law(q, reference, mask, tau)
        metrics = swap.decision_metrics(p, p_sealed, reference, mask)
        rows[label] = {'tv': metrics['tv'], 'top1_same': metrics['top1_same'],
                       'kl_variant_reference': metrics['kl_mix_reference'], 'kl_sealed_reference': metrics['kl_g0_reference'],
                       'max_abs_q_difference': float(np.abs(q[mask] - q_sealed[mask]).max()),
                       'law': p, 'top1': int(np.argmax(p))}
    return sealed_row, rows


def identity(config_path, local_path, experiment):
    return {'config_sha256': hash_file(config_path), 'local_config_sha256': hash_file(local_path),
            'experiment_id': experiment, 'git_commit': git_commit(),
            'source_hashes': {rel: hash_file(PROJECT / rel) for rel in SOURCES}}


def _stack(rows, key, dtype=None):
    return np.array([r[key] for r in rows], dtype=dtype)


def run(config, plan, local, config_path, local_path, output, smoke):
    started = time.perf_counter()
    experiment = config['experiment_id']
    reg = rpv.load_registration(REPO / config['policy_registration']['config'],
                                [REPO / path for path in config['policy_registration']['addenda']])
    require(reg['config_sha256'] == config['policy_registration']['config_sha256'], 'Policy registration config changed')
    policy, settings = reg['config'], config['settings']
    require(plan == variant_plan(config['variants'], settings, registered_seeds(policy)), 'Variant plan differs from the registration')
    freeze = rpv.registered_json(reg, 'tau_freeze')
    root, header, ledger = swap.sealed_ledger(config)
    require(freeze['final_identity_sha256'] == config['sealed_g0_run']['final_identity_sha256']
            and hash_file(root / 'tau_freeze.json') == rpv.registered_path(reg, 'tau_freeze')[1],
            'Sealed run is not the registered tau freeze')
    require(freeze['selected_tau'] == settings['tau'] and freeze['setting'] == {'samples': settings['samples'],
            'pitch_cap': settings['pitch_cap']} and policy['seeds']['planning_main'] == settings['seed'],
            'Diagnostic settings differ from the sealed G0 run')
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'started.json', {'identity': identity(config_path, local_path, experiment), 'registration_chain': reg['chain'],
                                   'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': smoke,
                                   'outcomes_read': False, 'origin': config['origin'], 'variants': plan})
    inputs = rpv.load_inputs(policy, local)
    frame, store = inputs['frame'], inputs['store']
    no_pitch = frozenset(policy['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
    artifact = load_train_bc(*rpv.registered_path(reg, 'bc'))
    spec = policy['le2025_validation_plan']['stages']['S3b_tau_select']
    games = rpv.select_games(frame, 'blend', spec['n_games'], rpv.role_seed(policy, 'selection_salt'))
    require(games == freeze['games'], 'Selected June games differ from the sealed run')
    if smoke:
        games = games[:swap.SMOKE['games']]
    blocks = rpv.game_blocks(frame, games, 'blend')
    snapshot, as_of, _ = rpv.load_style_snapshot(*rpv.registered_path(reg, 'style_blend'))
    components = swap.bind(reg, inputs, blocks, artifact, snapshot, as_of)
    support, _ = load_support_table(*rpv.registered_path(reg, 'support'), artifact)
    reference = prt.MaskedReference(artifact.bc, support)
    we, budget = components.we, int(config['row_budget'])

    def planner(seed, samples):
        return RolloutImprovement(reference, JointSimulator(components.inputs.pool, components.g0, we.terminal,
                                  RowBudget(budget, seed_count=5)), we.cutoff,
                                  samples=samples, pitch_cap=settings['pitch_cap'], seed=seed)
    replay_planner = planner(settings['seed'], settings['samples'])
    planners = {v['label']: planner(v['seed'], v['samples']) for v in plan}
    replay_limit = swap.SMOKE['replay_decisions'] if smoke else config['replay']['decisions']
    tau, replay, seen = settings['tau'], [], 0
    sealed_rows, variant_rows, request_ids, row_games, row_strikes = [], {v['label']: [] for v in plan}, [], [], []
    for _, positions in blocks:
        requests, _, _ = preq.pa_requests(store, positions, header['runtime_sha256'], no_pitch)
        for request in requests:
            sealed = ledger.get(request.request_id)
            fingerprint = canonical_hash({'request': request.fingerprint(),
                                          'context_sha256': components.context_sha256(request.state)})
            require(sealed is not None and sealed['fingerprint'] == fingerprint,
                    f'Request differs from the sealed ledger: {request.request_id}')
            seen += 1
            if sealed['status'] not in prt.EVALUATED:
                continue
            state, result = request.state, sealed['result']
            mask, ref = reference.support(state), reference.probabilities(state)
            require(np.array_equal(mask, np.asarray(result['mask'], dtype=bool))
                    and np.allclose(ref, result['reference'], rtol=0, atol=1e-12)
                    and components.context_sha256(state) == sealed['context_sha256'],
                    f'Reference law or context differs from the sealed ledger: {request.request_id}')
            q_sealed = np.array([np.nan if v is None else v for v in result['q_planning']], dtype=np.float64)
            if len(replay) < replay_limit:  # the sealed setting must reproduce the sealed Q on this machine
                q_again, _ = replay_planner.q_values(state)
                replay.append(float(np.abs(q_again[mask] - q_sealed[mask]).max()))
                require(replay[-1] <= config['replay']['atol'], f'G0 replay differs from the sealed planning Q: {request.request_id}')
            sealed_row, rows = compare_decision(planners, state, q_sealed, ref, mask, tau)
            sealed_rows.append(sealed_row)
            for label, row in rows.items():
                variant_rows[label].append(row)
            request_ids.append(request.request_id)
            row_games.append(int(frame.game_pk.iloc[positions[0]]))
            row_strikes.append(int(state.strikes))
    components.verify()
    require(smoke or seen == len(ledger), 'Not every sealed request was regenerated')
    require(len(replay) == replay_limit, 'Replay did not reach the registered decision count')
    evaluated = sum(r['status'] in prt.EVALUATED for r in ledger.values())
    require(smoke or len(sealed_rows) == evaluated == config['sealed_g0_run']['evaluated_decisions'], 'Evaluated decision count differs')
    scalar = ('tv', 'top1_same', 'kl_variant_reference', 'kl_sealed_reference', 'max_abs_q_difference')
    for rows in variant_rows.values():
        require(all(np.isfinite([r[k] for k in scalar]).all() for r in rows), 'Non-finite comparison value')
    keys = {'request_id': np.array(request_ids), 'game_pk': np.array(row_games), 'strikes': np.array(row_strikes)}
    np.savez_compressed(output / 'sealed.npz', **keys, law=_stack(sealed_rows, 'law'), top1=_stack(sealed_rows, 'top1'),
                        mask=_stack(sealed_rows, 'mask'))
    per_variant = {}
    for v in plan:
        rows = variant_rows[v['label']]
        np.savez_compressed(output / f"decisions-{v['label']}.npz", **keys, law=_stack(rows, 'law'), top1=_stack(rows, 'top1'),
                            **{k: _stack(rows, k) for k in scalar})
        per_variant[v['label']] = {**v, 'decisions_file': f"decisions-{v['label']}.npz",
                                   'mean_tv_point': float(np.mean([r['tv'] for r in rows])),
                                   'top1_agreement_point': float(np.mean([r['top1_same'] for r in rows])),
                                   'conditional_rows': planners[v['label']].simulator.budget.conditional_rows}
    result = {'experiment_id': experiment, 'protocol': PROTOCOL, 'smoke': smoke, 'settings': settings,
              'sealed_g0_run': config['sealed_g0_run'], 'origin': config['origin'],
              'replay': {'decisions': len(replay), 'max_abs_q_difference': max(replay), 'atol': config['replay']['atol']},
              'requests_regenerated': seen, 'decisions_compared': len(sealed_rows), 'variants': per_variant,
              'row_budget': int(config['row_budget']), 'bootstrap': config['bootstrap'],
              'comparison': None, 'verdict': None,
              'aggregation': 'offline: scripts/diagnose_policy_stability_aggregate.py over this directory (descriptive only)',
              'cost': {'seconds': time.perf_counter() - started, 'peak_rss_bytes': peak(),
                       'replay_conditional_rows': replay_planner.simulator.budget.conditional_rows},
              'git_commit': git_commit(), 'outcomes_read': False, 'policy_identity_registered': False,
              'limits': ['Drafted after the EXP-P17-001 and EXP-P18-001 results were read (P18-informed reading rules)',
                         'Model-internal diagnostic of the planning law under a fixed predictor; not a policy value estimate',
                         'Variants that keep the sealed seed and raise the sample count reuse the sealed draws as a prefix (nested comparison)',
                         'All variant-vs-sealed distances share the one sealed seed-0 side; they are not independent replicates',
                         'Same-model drafting; the independent review gate is not passed'],
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    dump(output / 'results.json', result)
    print('SEED_STABILITY_COMPLETE', {'decisions': len(sealed_rows), 'smoke': smoke, 'variants': len(plan),
                                      'seconds': round(result['cost']['seconds'], 1)},
          {k: round(v['mean_tv_point'], 4) for k, v in per_variant.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='three games and a short replay; output must end -smoke')
    args = p.parse_args()
    config, local = read_json(args.config), read_json(args.local_config)
    reg_config = read_json(REPO / config['policy_registration']['config'])
    plan = config_check(config, registered_seeds(reg_config), args.smoke)
    experiment = config['experiment_id']
    output = args.output.resolve()
    require(args.smoke == output.name.endswith('-smoke'), 'Smoke runs write only to an output ending -smoke, and only smoke runs may')
    if not args.smoke:  # a smoke may run from the draft file; a full run only from the registered config
        require(output.name == experiment, 'Output directory must be named after the experiment')
        require(args.config.resolve() == (REPO / 'configs' / (experiment + '.yaml')).resolve(), 'Use the registered config file')
        require(not (REPO / 'results' / (experiment + '.json')).exists(), 'Preserve the completed repository result')
    root = check_location(local, output)
    validate_native_runtime()
    with heavy_lock(root):
        run(config, plan, local, args.config.resolve(), args.local_config.resolve(), output, args.smoke)


if __name__ == '__main__':
    main()
