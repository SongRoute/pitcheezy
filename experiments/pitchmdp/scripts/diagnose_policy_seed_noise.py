"""EXP-P18-001: search-noise floor of the P3 law, to read the EXP-P17-001 predictor-swap result.

Additive diagnostic, registered AFTER the EXP-P17-001 result was read (post-hoc; disclosed).
EXP-P17-001 found that swapping the planner's predictor moves the P3 law by a mean TV of 0.0323.
The planning Q is a six-sample Monte Carlo estimate, so part of that distance can be search noise.
Here the predictor is unchanged (the bound G0) and only the planning seed changes: on the same
sealed June decisions the P3 law is recomputed from a seed-1 planning Q and compared with the
sealed seed-0 law.  The paired excess of the swap distance over this seed distance is the
registered quantity.

No outcome, reward, WE result or events kind is read.  Nothing is registered as a policy identity.

``--smoke`` limits the games to check the wiring; it writes to an output ending ``-smoke`` and
neither computes the comparison nor writes the repository result.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
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

EXPERIMENT = 'EXP-P18-001'
SOURCES = (*swap.SOURCES, 'scripts/diagnose_policy_seed_noise.py')
SAME_AS_SWAP = ('policy_registration', 'sealed_g0_run', 'settings', 'bootstrap', 'data_rules')
BEYOND, WITHIN = 'PREDICTOR_EFFECT_BEYOND_NOISE', 'WITHIN_SEARCH_NOISE'
require = swap.require


def config_check(config, smoke=False):
    require(config['protocol'] == 'policy_seed_noise_diagnostic_v1' and config['experiment_id'] == EXPERIMENT,
            'Diagnostic settings differ from registered protocol')
    rule = config['verdict_rule']
    require(rule.get('excess_mean_tv_at_least', 0) > 0, 'Registered verdict rule required')
    require(type(config['noise_seed']) is int and config['noise_seed'] != config['settings']['seed'],
            'The noise seed must differ from the sealed planning seed')
    for key in ('swap_run', *SAME_AS_SWAP):
        require(config.get(key), 'Registered input missing: ' + key)
    require(smoke or config['execution']['enabled'] is True, 'Execution not enabled in the registered config')
    return config


def verdict(tv_swap, tv_seed, same_swap, same_seed, games, rule, *, draws, seed):
    """Paired excess of the swap distance over the seed distance, whole-game bootstrap."""
    tv_swap, tv_seed = np.asarray(tv_swap, dtype=float), np.asarray(tv_seed, dtype=float)
    same_swap, same_seed = np.asarray(same_swap, dtype=float), np.asarray(same_seed, dtype=float)
    unique, index = np.unique(np.asarray(games), return_inverse=True)
    columns = (tv_swap - tv_seed, same_seed - same_swap, tv_seed, np.ones(len(tv_swap)))
    sums = np.stack([np.bincount(index, weights=v, minlength=len(unique)) for v in columns])
    picks = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    boot = sums[:, picks].sum(axis=2)
    interval = lambda k: [float(v) for v in np.quantile(boot[k] / boot[3], (.025, .975))]
    excess = float((tv_swap - tv_seed).mean())
    result = {'decisions': len(tv_swap), 'games': len(unique),
              'swap': {'mean_tv': float(tv_swap.mean()), 'top1_agreement': float(same_swap.mean())},
              'seed_noise': {'mean_tv': float(tv_seed.mean()), 'mean_tv_ci95': interval(2),
                             'top1_agreement': float(same_seed.mean()),
                             'tv_quantiles': {str(q): float(np.quantile(tv_seed, q)) for q in (.5, .9, .99, 1.)},
                             'share_tv_above_0.05': float((tv_seed > .05).mean())},
              'excess_mean_tv': excess, 'excess_mean_tv_ci95': interval(0),
              'excess_top1_disagreement': float((same_seed - same_swap).mean()),
              'excess_top1_disagreement_ci95': interval(1),
              'swap_to_seed_tv_ratio': float(tv_swap.mean() / tv_seed.mean()) if tv_seed.mean() > 0 else None}
    beyond = excess >= rule['excess_mean_tv_at_least'] and result['excess_mean_tv_ci95'][0] > 0
    result['verdict'] = {**rule, 'status': BEYOND if beyond else WITHIN}
    return result


def identity(local_path):
    return {'config_sha256': hash_file(REPO / 'configs' / (EXPERIMENT + '.yaml')), 'local_config_sha256': hash_file(local_path),
            'git_commit': git_commit(), 'source_hashes': {rel: hash_file(PROJECT / rel) for rel in SOURCES}}


def swap_decisions(config):
    """Per-decision EXP-P17-001 distances, pinned by hash."""
    spec = config['swap_run']
    root = Path(spec['path'])
    require(hash_file(root / 'decisions.npz') == spec['decisions_sha256']
            and hash_file(root / 'results.json') == spec['results_sha256'], 'EXP-P17-001 output changed')
    registered = read_json(root / 'started.json')['identity']['config_sha256']
    frozen = read_json(REPO / 'configs' / 'EXP-P17-001.yaml')
    require(hash_file(REPO / 'configs' / 'EXP-P17-001.yaml') == registered, 'EXP-P17-001 config changed since its run')
    for key in SAME_AS_SWAP:
        require(frozen[key] == config[key], 'The noise run must not change the swap setting: ' + key)
    with np.load(root / 'decisions.npz', allow_pickle=False) as data:
        return {k: data[k].copy() for k in ('request_id', 'game_pk', 'tv', 'top1_same')}


def run(config, local, local_path, output, smoke):
    started = time.perf_counter()
    reg = rpv.load_registration(REPO / config['policy_registration']['config'],
                                [REPO / path for path in config['policy_registration']['addenda']])
    require(reg['config_sha256'] == config['policy_registration']['config_sha256'], 'Policy registration config changed')
    policy, settings = reg['config'], config['settings']
    freeze = rpv.registered_json(reg, 'tau_freeze')
    _, header, ledger = swap.sealed_ledger(config)
    require(freeze['selected_tau'] == settings['tau'] and freeze['setting'] == {'samples': settings['samples'],
            'pitch_cap': settings['pitch_cap']} and policy['seeds']['planning_main'] == settings['seed'],
            'Diagnostic settings differ from the sealed G0 run')
    previous = swap_decisions(config)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'started.json', {'identity': identity(local_path), 'registration_chain': reg['chain'],
                                   'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': smoke,
                                   'outcomes_read': False, 'registered_after_swap_result': True})
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
    we = components.we
    planner = RolloutImprovement(reference, JointSimulator(components.inputs.pool, components.g0, we.terminal,
                                 RowBudget(int(config['row_budget']), seed_count=5)), we.cutoff,
                                 samples=settings['samples'], pitch_cap=settings['pitch_cap'], seed=config['noise_seed'])
    tau, rows, request_ids, row_games = settings['tau'], [], [], []
    for _, positions in blocks:
        requests, _, _ = preq.pa_requests(store, positions, header['runtime_sha256'], no_pitch)
        for request in requests:
            sealed = ledger.get(request.request_id)
            fingerprint = canonical_hash({'request': request.fingerprint(),
                                          'context_sha256': components.context_sha256(request.state)})
            require(sealed is not None and sealed['fingerprint'] == fingerprint,
                    f'Request differs from the sealed ledger: {request.request_id}')
            if sealed['status'] not in prt.EVALUATED:
                continue
            state, result = request.state, sealed['result']
            mask, ref = reference.support(state), reference.probabilities(state)
            require(np.array_equal(mask, np.asarray(result['mask'], dtype=bool))
                    and np.allclose(ref, result['reference'], rtol=0, atol=1e-12),
                    f'Reference law differs from the sealed ledger: {request.request_id}')
            q_sealed = np.array([np.nan if v is None else v for v in result['q_planning']], dtype=np.float64)
            q_noise, _ = planner.q_values(state)
            rows.append({**swap.decision_metrics(swap.law(q_noise, ref, mask, tau), swap.law(q_sealed, ref, mask, tau), ref, mask),
                         'max_abs_q_difference': float(np.abs(q_noise[mask] - q_sealed[mask]).max())})
            request_ids.append(request.request_id)
            row_games.append(int(frame.game_pk.iloc[positions[0]]))
    components.verify()
    require(all(np.isfinite(list(r.values())).all() for r in rows), 'Non-finite comparison value')
    comparison = None
    if not smoke:  # the comparison is opened only by the full registered run
        require(np.array_equal(np.array(request_ids), previous['request_id'])
                and np.array_equal(np.array(row_games), previous['game_pk']), 'Decisions differ from the EXP-P17-001 run')
        np.savez_compressed(output / 'decisions.npz', request_id=np.array(request_ids), game_pk=np.array(row_games),
                            **{k: np.array([r[k] for r in rows]) for k in rows[0]})
        comparison = verdict(previous['tv'], [r['tv'] for r in rows], previous['top1_same'], [r['top1_same'] for r in rows],
                             row_games, config['verdict_rule'], draws=config['bootstrap']['draws'], seed=config['bootstrap']['seed'])
    result = {'experiment_id': EXPERIMENT, 'smoke': smoke, 'settings': settings, 'noise_seed': config['noise_seed'],
              'swap_run': config['swap_run'], 'sealed_g0_run': config['sealed_g0_run'], 'decisions_compared': len(rows),
              'comparison': comparison,
              'mean_max_abs_q_difference': None if smoke else float(np.mean([r['max_abs_q_difference'] for r in rows])),
              'cost': {'seconds': time.perf_counter() - started, 'peak_rss_bytes': peak(),
                       'conditional_rows': planner.simulator.budget.conditional_rows},
              'git_commit': git_commit(), 'outcomes_read': False, 'policy_identity_registered': False,
              'limits': ['Registered after the EXP-P17-001 result was read (post-hoc analysis)',
                         'An independent-seed distance overstates the noise inside the common-random-number swap comparison: '
                         'BEYOND is conservative, WITHIN does not prove a zero predictor effect',
                         'One alternative seed; model-internal diagnostic, not a policy value estimate',
                         'Same-model review only; the independent review gate is not passed'],
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    dump(output / 'results.json', result)
    if not smoke:
        repo_result = REPO / 'results' / (EXPERIMENT + '.json')
        require(not repo_result.exists(), 'Preserve the completed repository result')
        shutil.copyfile(output / 'results.json', repo_result)
    print('SEED_NOISE_COMPLETE', {'decisions': len(rows), 'smoke': smoke, 'seconds': round(result['cost']['seconds'], 1)},
          '' if smoke else {k: comparison[k] for k in ('excess_mean_tv', 'excess_mean_tv_ci95')},
          '' if smoke else comparison['verdict']['status'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='three games; output must end -smoke')
    args = p.parse_args()
    config, local = config_check(read_json(args.config), args.smoke), read_json(args.local_config)
    output = args.output.resolve()
    require(args.smoke == output.name.endswith('-smoke'), 'Smoke runs write only to an output ending -smoke, and only smoke runs may')
    require(args.smoke or output.name == EXPERIMENT, 'Output directory must be named after the experiment')
    require(args.config.resolve() == (REPO / 'configs' / (EXPERIMENT + '.yaml')).resolve(), 'Use the registered config file')
    root = check_location(local, output)
    validate_native_runtime()
    with heavy_lock(root):
        run(config, local, args.local_config.resolve(), output, args.smoke)


if __name__ == '__main__':
    main()
