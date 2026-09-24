"""Strict TRAIN-only NNBC/IQL/CQL jobs and frozen cross-model policy evaluation.

No real work at import time. All CLI stages use the existing exclusive heavy
lock. Final update count is registered only after resource-only TRAIN profiles.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import pickle
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd
import torch
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_policy_artifacts import artifact_names, is_appledouble
from pitchmdp.matrix_policy import PolicyInputs, context_key
from pitchmdp.matrix_offline_rl import (FIXED, METHODS, audit_trajectories, terminal_rewards,
    LazyFeatures, OfflinePolicy, AmortizedPolicy, rl_comparisons)
from pitchmdp.rollout_policy import PAState, Rollouts, rollouts, paired_summary
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import heavy_lock, check_location, assert_hashes, artifact_hashes
from run_ml_sharing import load_data
from run_ml_policy import (SOURCES as POLICY_SOURCES, identity as policy_identity, verify as verify_policy,
                          resources, TimedBudget, finish_runtime)

SOURCES = list(dict.fromkeys([*POLICY_SOURCES, 'pitchmdp/matrix_offline_rl.py', 'scripts/run_ml_offline_rl.py']))
SEEDS = (0, 1, 2)


def source_hashes(): return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    return {**policy_identity(config, local_path), 'source_hashes': source_hashes()}


def config_check(config):
    if (config['protocol'] != 'ml_offline_rl_prepare_v1' or config['fixed'] != FIXED
        or config['seeds'] != list(SEEDS) or config['profile_updates'] != 256 or config['prepare_seconds'] != 3600
        or config['terminal_missing_next'] != 'exclude_whole_pa_no_exceptions'
        or config['comparator_rule'] != 'June_P3_if_mean_ge_P2_else_P2'
        or config['multiplicity'] != 'joint_four_imputed_and_worst_case_Holm'):
        raise ValueError('RL preparation differs from fixed first-screen contract')
    for field in ('parent_policy_preparation_sha256', 'parent_tuning_results_sha256'):
        if len(config[field]) != 64: raise ValueError('Frozen parent hash required')
    return config


def seal(directory):
    files = artifact_names(directory, exclude=('manifest.json',))
    dump(directory / 'manifest.json', {'artifact_hashes': artifact_hashes(directory, files)})


def verify_sealed(directory):
    manifest = read_json(directory / 'manifest.json')
    names = set(artifact_names(directory, exclude=('manifest.json',)))
    if names != set(manifest['artifact_hashes']): raise ValueError('Sealed RL artifact family changed')
    assert_hashes(directory, manifest['artifact_hashes'])
    return manifest


def fresh(directory, payload):
    if directory.exists(): raise ValueError('Preserve existing/completed/failed RL stage: '+str(directory))
    directory.mkdir(parents=True)
    dump(directory / 'started.json', {**payload, 'utc': datetime.now(timezone.utc).isoformat()})


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected: raise ValueError('RL preparation identity changed')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest: raise ValueError('RL frozen dependency changed: '+name)
    return prep


def prepare(config, local_path, local, output, expected):
    if output.exists() and any(output.iterdir()): verify(output, expected); return
    parent = Path(config['parent_policy_run']).resolve(); check_location(local, parent)
    if parent == output or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError('RL run must be a distinct sibling')
    pc = read_json(parent / 'registered_config.json')
    policy_prep = verify_policy(parent, policy_identity(pc, local_path))
    if hash_file(parent / 'preparation.json') != config['parent_policy_preparation_sha256']:
        raise ValueError('Parent policy preparation changed')
    tuning_path = parent / 'stages/blend-control/results.json'
    if hash_file(tuning_path) != config['parent_tuning_results_sha256']:
        raise ValueError('Frozen June tuning changed')
    # The common policy audit helper verifies sealed June per-PA arrays, recomputes
    # tau/comparator, and freezes the same dependency used by both DEV worlds.
    from run_ml_policy import verify_tuning
    policy_execution = read_json(Path(config['parent_policy_execution_config']))
    tuning, tuning_dependency = verify_tuning(parent, policy_execution)
    if tuning['rl_comparator'] not in ('P2', 'P3'): raise ValueError('June comparator not frozen')
    external = {str(parent / 'preparation.json'): hash_file(parent / 'preparation.json'),
                str(tuning_path): hash_file(tuning_path), **policy_prep['external_hashes']}
    for name, digest in policy_prep['artifact_hashes'].items(): external[str(parent / name)] = digest
    june = parent / 'stages/blend-control'
    # Include every sealed tuning artifact, not only the selected scalar result.
    for name in artifact_names(june):
        p = june / name
        external[str(p)] = hash_file(p)
    fresh(output, {'identity': expected})
    started = time.perf_counter(); prepare_deadline = time.monotonic()+config['prepare_seconds']
    with (parent / 'inputs.pkl').open('rb') as stream: saved = pickle.load(stream)
    parent_g_prep = read_json(parent / 'parent_preparation.json')
    store, context, parts, aux = load_data(local, Path(policy_prep['parent_run']), parent_g_prep)
    frame, train = store.frame, parts['train']
    templates = train.drop_duplicates(['pitcher', 'p_throws', 'stand'])
    support = PolicyInputs(templates, context, aux['delivery'], parent_g_prep['features']['tokens']['type_vocabulary'],
                          parent_g_prep['artifact_hashes']['aux.pkl'], saved['bc'])
    keys = {(str(int(row.pitcher)), row.p_throws, row.stand): context_key(row) for _, row in templates.iterrows()}
    def mask(row):
        pid = str(int(row.pitcher))
        return support.support(PAState(int(row.balls), int(row.strikes), pid, str(row.stand), (),
                                      keys[(pid, row.p_throws, row.stand)]))
    roles = {int(r['pitcher']): r['train_role'] for r in parent_g_prep['panel']['train_players']}
    records, audit = audit_trajectories(frame, train.index, saved['bc'].actions, mask, roles, deadline=prepare_deadline)
    audit.to_parquet(output / 'trajectory_audit.parquet', index=False)
    report = {'raw_train_pa': len(audit), 'complete_d100_pa': int(audit.complete_d100.sum()),
        'retained_pa': int(audit.retained.sum()), 'retained_transitions': len(records),
        'reasons': audit.reasons.str.get_dummies(sep='|').sum().astype(int).to_dict(),
        'role_terminal_representativeness': audit.groupby(['train_role', 'terminal_class'], dropna=False).agg(
            requested_pa=('retained', 'size'), complete_d100_pa=('complete_d100', 'sum'), retained_pa=('retained', 'sum')).reset_index().to_dict('records')}
    dump(output / 'trajectory_audit.json', report)
    if not records: raise ValueError('No strict TRAIN trajectories remain; review concrete audit before weakening rules')
    with Path(policy_prep['we_path']).open('rb') as stream: values = pickle.load(stream)
    rewards = terminal_rewards(records, frame, values['we'], values['advancement'])
    if time.monotonic() > prepare_deadline: raise TimeoutError('RL preparation wall limit exceeded before feature bank')
    positions = LazyFeatures.write(output / 'features', store, context, records, rewards)
    if time.monotonic() > prepare_deadline: raise TimeoutError('RL preparation wall limit exceeded after feature bank')
    frame.iloc[positions][[*KEY, 'game_date', 'pitcher']].to_parquet(output / 'train_keys.parquet', index=False)
    for name in SOURCES:
        target = output / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    dump(output / 'registered_config.json', config)
    dump(output / 'parent_policy_config.json', pc)
    dump(output / 'parent_policy_preparation.json', policy_prep)
    dump(output / 'parent_policy_execution.json', policy_execution)
    dump(output / 'tuning_dependency.json', tuning_dependency)
    files = artifact_names(output)
    dump(output / 'preparation.json', {'identity': expected, 'artifact_hashes': artifact_hashes(output, files),
        'external_hashes': external, 'parent_policy_run': str(parent), 'comparator': tuning['rl_comparator'],
        'tuning_dependency': tuning_dependency, 'trajectory_audit': report, 'train_keys_sha256': ordered_key_hash(frame.iloc[positions]),
        'preparation_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'feature_bank_bytes': sum(p.stat().st_size for p in (output / 'features').glob('*.npy') if not is_appledouble(p)),
        'preparation_memory_note': 'Transition records and token/context payload are materialized before mmap write; lazy minibatch gathering applies to fits.',
        'actions': list(saved['bc'].actions), 'feature_width': LazyFeatures(output / 'features').width,
        'seconds': time.perf_counter()-started,
        'strict_missing_terminal_next_exceptions': 0, 'value_spec_version': 'defense-we-pa-v1',
        'reward': 'zero intermediate; frozen expected-advancement WE at observed terminal event; gamma1',
        'information': 'past H5+mask+safe context; current token and last2 routing IDs omitted'})
    print('RL_PREPARED', report['retained_pa'], 'PA', len(records), 'transitions', flush=True)


def train_job(bank, method, seed, updates, destination, seconds_limit):
    started = time.perf_counter()
    model = None
    sampler = np.random.default_rng(seed)
    try:
        model = OfflinePolicy(method, bank.width, bank.support.shape[1], seed)
        with (destination / 'updates.jsonl').open('x') as log:
            for step in range(updates):
                if time.perf_counter()-started > seconds_limit: raise TimeoutError('RL fit/profile wall ceiling reached')
                indices = sampler.integers(bank.size, size=FIXED['batch_size'])
                metrics = model.train_step(bank.batch(indices))
                log.write(json.dumps({'update': step+1, **metrics}, allow_nan=False)+'\n')
                if (step+1) % 100 == 0: log.flush()
        model.save(destination / 'model.pt')
        if time.perf_counter()-started > seconds_limit:
            raise TimeoutError('RL fit/profile wall ceiling exceeded at completion')
    except Exception as error:
        failure = {'method': method, 'seed': seed, 'updates': model.updates if model is not None else 0,
            'seconds': time.perf_counter()-started, 'completed': False, 'error': str(error),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        # Record charged wall time before a best-effort interrupted checkpoint,
        # which itself can fail during OOM/disk errors. Caller logs cover SIGKILL.
        dump(destination / 'runtime.json', failure)
        if model is not None:
            try: model.save(destination / 'interrupted_model.pt')
            except Exception as checkpoint_error:
                failure['checkpoint_error'] = str(checkpoint_error)
            finally:
                failure['seconds'] = time.perf_counter()-started
                dump(destination / 'runtime.json', failure)
        raise
    report = {'method': method, 'seed': seed, 'updates': updates, 'transition_draws': updates*FIXED['batch_size'],
        'seconds': time.perf_counter()-started, 'device': model.device, 'fixed': FIXED,
        'trainable_networks': {name: sum(p.numel() for p in net.parameters()) for name, net in model.networks.items()},
        'target_networks': list(model.targets), 'optimizer_steps': updates*len(model.networks),
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'policy_quality_screen': False}
    dump(destination / 'runtime.json', report)
    return report


def profile(config, output, prep):
    dest = output / 'profile'; fresh(dest, {'preparation_sha256': hash_file(output / 'preparation.json'), 'quality_scores_read': False})
    bank = LazyFeatures(output / 'features'); reports = {}
    for method in METHODS:
        directory = dest / method; directory.mkdir()
        reports[method] = train_job(bank, method, 0, config['profile_updates'], directory, 1800)
        print('RL_PROFILE', method, reports[method]['seconds'], 'seconds', flush=True)
    dump(dest / 'results.json', {'profile_only': True, 'updates': config['profile_updates'], 'reports': reports,
        'projection_seconds_20k_all3seeds': sum(r['seconds']/r['updates']*20000*3 for r in reports.values()),
        'training_quality_not_used_for_setting_selection': True})
    seal(dest)


def execution_check(execution, output):
    if (execution['protocol'] != 'ml_offline_rl_execution_v1'
        or execution['preparation_sha256'] != hash_file(output / 'preparation.json')
        or execution['profile_result_sha256'] != hash_file(output / 'profile/results.json')
        or type(execution['updates']) is not int or not 1 <= execution['updates'] <= 20000
        or execution['member_seconds'] != 3600 or execution['family_seconds'] != 10800):
        raise ValueError('RL update count/resource registration differs')
    verify_sealed(output / 'profile')
    frozen = output / 'final_execution.json'
    if frozen.exists():
        if read_json(frozen) != execution: raise ValueError('RL final execution settings already frozen')
    else: dump(frozen, execution)
    return execution


def family_seconds(output):
    """Charge failed attempts too; an unaccounted hard kill blocks more fits."""
    spent = 0.
    for started in (output / 'members').glob('*/*/started.json') if (output / 'members').exists() else ():
        runtime = started.parent / 'runtime.json'
        if not runtime.exists():
            raise ValueError('Unaccounted interrupted RL member; preserve caller elapsed ledger before proceeding: '+str(started.parent))
        seconds = read_json(runtime)['seconds']
        if not np.isfinite(seconds) or seconds < 0: raise ValueError('Invalid RL elapsed-time ledger')
        spent += seconds
    return spent


def fit(config, output, prep, execution, method, seed):
    if method not in METHODS or seed not in SEEDS: raise ValueError('Unregistered RL member')
    spent = family_seconds(output)
    remaining = execution['family_seconds']-spent
    if remaining <= 0: raise TimeoutError('RL family wall ceiling exhausted')
    dest = output / 'members' / method / f'seed{seed}'
    fresh(dest, {'preparation_sha256': hash_file(output / 'preparation.json'), 'execution_sha256': canonical_hash(execution),
                 'method': method, 'seed': seed})
    report = train_job(LazyFeatures(output / 'features'), method, seed, execution['updates'], dest,
                       min(execution['member_seconds'], remaining))
    dump(dest / 'state.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
        'execution_sha256': canonical_hash(execution), 'method': method, 'seed': seed, 'completed': True,
        'family_seconds_before': spent, 'seconds': report['seconds']})
    seal(dest)
    print('RL_FIT_COMPLETE', method, seed, flush=True)


def verify_family(output, prep, execution):
    """Complete nine-member metadata/payload audit before any network is built."""
    expected_prep, expected_execution = hash_file(output / 'preparation.json'), canonical_hash(execution)
    dependencies = {}
    for method in METHODS:
        for seed in SEEDS:
            member = output / 'members' / method / f'seed{seed}'
            manifest = verify_sealed(member)
            state, started = read_json(member / 'state.json'), read_json(member / 'started.json')
            expected = dict(preparation_sha256=expected_prep, execution_sha256=expected_execution,
                            method=method, seed=seed)
            if any(state.get(k) != v or started.get(k) != v for k, v in expected.items()) or state.get('completed') is not True:
                raise ValueError('RL full-family member identity differs')
            payload = torch.load(member / 'model.pt', map_location='cpu', weights_only=False)
            expected_model = dict(format='matrix_offline_rl_v1', method=method, seed=seed,
                width=prep['feature_width'], n_actions=len(prep['actions']), updates=execution['updates'], fixed=FIXED)
            if any(payload.get(k) != v for k, v in expected_model.items()):
                raise ValueError('RL checkpoint method/seed/dimensions/updates differ')
            for net in [*payload['networks'].values(), *payload['targets'].values()]:
                if any(not bool(torch.isfinite(value).all()) for value in net.values()):
                    raise ValueError('RL completed checkpoint contains nonfinite weights')
            dependencies[str(member / 'manifest.json')] = hash_file(member / 'manifest.json')
            dependencies.update({str(member / name): digest for name, digest in manifest['artifact_hashes'].items()})
    return dependencies


def evaluate(config, output, prep, execution, world):
    from run_ml_policy import verify_stage, verify_tuning
    parent = Path(prep['parent_policy_run'])
    pc, pp = read_json(output / 'parent_policy_config.json'), read_json(output / 'parent_policy_preparation.json')
    pe = read_json(output / 'parent_policy_execution.json')
    if verify_tuning(parent, pe)[1] != prep['tuning_dependency']: raise ValueError('June comparator/tau dependency changed')
    parent_stage = parent / f'stages/dev-{world}'
    verify_stage(parent, 'dev-'+world)
    parent_results = read_json(parent_stage / 'results.json')
    if parent_results['execution_sha256'] != canonical_hash(pe): raise ValueError('Parent DEV execution changed')
    dependencies = verify_family(output, prep, execution)
    dest = output / 'evaluation' / world
    fresh(dest, {'execution_sha256': canonical_hash(execution), 'policy_execution_sha256': canonical_hash(pe)})
    started = time.perf_counter(); budget = TimedBudget(pe['row_budget'], pe['seconds_budget'])
    saved, inputs, bc, we, outcome_models, simulators = resources(parent, pp, pc, budget)
    fitted = {}
    for method in METHODS:
        fitted[method] = []
        for seed in SEEDS:
            member = output / 'members' / method / f'seed{seed}'
            fitted[method].append(OfflinePolicy.load(member / 'model.pt'))
    policy_map = {method: AmortizedPolicy(inputs, bc, fitted[method]) for method in METHODS}
    for method in METHODS:
        for seed, model in zip(SEEDS, fitted[method]): policy_map[f'{method}-seed{seed}'] = AmortizedPolicy(inputs, bc, [model])
    values, flags = {k: [] for k in policy_map}, {k: [] for k in policy_map}
    values['planner'], flags['planner'] = [], []
    games, keys = [], []
    for key, game in zip(parent_results['pa_keys'], parent_results['game_ids']):
        state = saved['states'][key]; g, pa, _ = map(int, key.split(':'))
        uniforms = np.random.default_rng(np.random.SeedSequence([pe['evaluation_seed'], 1, g, pa])).random(
            (pe['evaluation_rollouts'], pe['evaluation_cap'], 3))
        parent_archive = parent_stage / (key.replace(':', '-')+'.npz')
        with np.load(parent_archive) as original:
            values['planner'].append(original[prep['comparator']+'_values'].copy())
            flags['planner'].append(original[prep['comparator']+'_truncated'].copy())
        archive = {'planner_values': values['planner'][-1], 'planner_truncated': flags['planner'][-1]}
        for name, policy in policy_map.items():
            result = rollouts(simulators[world], [state]*len(uniforms), policy, uniforms, we.cutoff)
            values[name].append(result.values); flags[name].append(result.truncated)
            archive.update({name+'_values': result.values, name+'_truncated': result.truncated, name+'_pitches': result.pitches})
        np.savez_compressed(dest / (key.replace(':', '-')+'.npz'), **archive)
        games.append(int(game)); keys.append(key)
        print('RL_EVALUATE_CASE', world, key, 'rows', budget.conditional_rows, flush=True)
    values = {k: np.stack(v) for k, v in values.items()}; flags = {k: np.stack(v) for k, v in flags.items()}
    runtime = finish_runtime(dest, started, budget, outcome_models)
    family = rl_comparisons(values, flags, games, draws=pe['bootstrap_draws'], seed=pe['bootstrap_seed'])
    diagnostics = {name: paired_summary(Rollouts(values[name], flags[name], np.empty(0)),
        Rollouts(values['planner'], flags['planner'], np.empty(0))) for name in policy_map}
    dependencies[str(parent_stage / 'manifest.json')] = hash_file(parent_stage / 'manifest.json')
    dump(dest / 'results.json', {'execution_sha256': canonical_hash(execution), 'parent_policy_execution_sha256': canonical_hash(pe),
        'world': world, 'comparator': prep['comparator'], 'tuning_dependency': prep['tuning_dependency'],
        'requested_pa_starts': parent_results['requested_pa_starts'], 'selected_pa_starts': parent_results['selected_pa_starts'],
        'supported_selected': len(keys), 'selected_unsupported_reasons': parent_results['selected_unsupported_reasons'],
        'pa_keys': keys, 'game_ids': games, 'primary_joint_four_family': family, 'per_policy_mc_diagnostics': diagnostics,
        'mean_we': {k: float(v.mean()) for k, v in values.items()}, 'runtime': runtime,
        'actor_or_q_forward_rows': {method: [m.inference_rows for m in models] for method, models in fitted.items()},
        'inference': 'Control-world primary; candidate-world sensitivity; no independent empirical evaluator',
        'policy_adoption': None, 'observational_ope': None, 'causal_effect': None, 'dependencies': dependencies})
    seal(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare'); sub.add_parser('profile')
    fit_parser = sub.add_parser('fit'); fit_parser.add_argument('--method', choices=METHODS, required=True)
    fit_parser.add_argument('--seed', type=int, choices=SEEDS, required=True); fit_parser.add_argument('--execution-config', type=Path, required=True)
    ev = sub.add_parser('evaluate'); ev.add_argument('--world', choices=('control', 'candidate'), required=True)
    ev.add_argument('--execution-config', type=Path, required=True)
    args = parser.parse_args(); config, local = config_check(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve(); root = check_location(local, output)
    validate_native_runtime(); expected = identity(config, args.local_config)
    with heavy_lock(root):
        try:
            if args.command == 'prepare': prepare(config, args.local_config, local, output, expected)
            else:
                prep = verify(output, expected)
                if args.command == 'profile': profile(config, output, prep)
                elif args.command == 'fit': fit(config, output, prep, execution_check(read_json(args.execution_config), output), args.method, args.seed)
                else: evaluate(config, output, prep, execution_check(read_json(args.execution_config), output), args.world)
            if source_hashes() != expected['source_hashes']: raise ValueError('RL sources changed during stage')
        except Exception as error:
            if output.is_dir():
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                dump(output / ('failure-'+stamp+'.json'), {'command': args.command, 'error': str(error),
                    'error_type': type(error).__name__, 'complete': False, 'preserve_partial_artifacts': True})
            raise


if __name__ == '__main__': main()
