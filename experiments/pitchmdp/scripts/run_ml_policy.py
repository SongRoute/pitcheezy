"""Frozen G cross-model policy prepare/profile/P0/rollout stages; no new NN fit.

Preparation and execution registrations are separate: resource-only May profile
precedes final counts/tau registration. Every CLI stage acquires the heavy lock.
Interrupted stages are preserved and require a fresh explicitly reviewed attempt.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd
from pitchmdp.data import KEY, hash_file
from pitchmdp.game import GameState
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_policy import (PolicyInputs, SupportedBC, FrozenGEnsemble, FrozenWE,
    context_key, safe_rows, state_from_row, fit_bc, select_pa_requests, game_policy_comparisons)
from pitchmdp.rollout_policy import (PAState, PastPitch, RowBudget, BudgetExceeded,
    JointSimulator, RolloutImprovement, Rollouts, rollouts, paired_summary)
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_sharing import (SOURCES as SHARING_SOURCES, identity as sharing_identity,
    verify as verify_sharing, load_data, predictor, SEEDS)
from run_ml_transfer import chosen_comparison, CONTROLS
from run_ml_matrix import heavy_lock, check_location, assert_hashes, artifact_hashes

SOURCES = list(dict.fromkeys([*SHARING_SOURCES, 'pitchmdp/rollout_policy.py',
    'pitchmdp/matrix_policy.py', 'pitchmdp/game.py', 'scripts/run_ml_policy.py', 'scripts/run_ml_transfer.py']))
WE_SOURCE_FILES = ('pitchmdp/game.py', 'scripts/build_minimal_pitch_service.py', 'scripts/run_temporal_blend.py')


def source_hashes(): return {name: hash_file(PROJECT / name) for name in SOURCES}


def config_check(config):
    if (config['protocol'] != 'ml_policy_prepare_v1' or config['seeds'] != list(SEEDS)
        or config['draws'] != 400 or config['value_spec_version'] != 'defense-we-pa-v1'
        or config['common_support'] != 'bc_and_token_vocab_and_type_delivery_all_counts'
        or config['evaluator'] != 'mapped_control' or config['bc'] != {'prior_strength': 20., 'minimum_action_count': 1}):
        raise ValueError('Unregistered policy preparation contract')
    if CONTROLS.get(config['candidate']) != config['control']:
        raise ValueError('Candidate/control mapping differs')
    for field in ('parent_preparation_sha256', 'parent_analysis_sha256', 'we_contract_sha256'):
        if len(config[field]) != 64: raise ValueError('Frozen hash required: '+field)
    for split in ('temperature', 'blend', 'dev'):
        if type(config['maximum_requested_starts'][split]) is not int or config['maximum_requested_starts'][split] < 1:
            raise ValueError('Positive selected-request maxima required')
    return config


def identity(config, local_path):
    return {**sharing_identity(config, local_path), 'source_hashes': source_hashes()}


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected: raise ValueError('Policy preparation identity changed')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest: raise ValueError('Frozen policy dependency changed: '+name)
    return prep


def verify_we(config):
    path = Path(config['we_contract']).resolve()
    if hash_file(path) != config['we_contract_sha256']: raise ValueError('WE contract identity changed')
    contract = read_json(path)
    if (contract['contract_version'] != 'C0-v1' or contract['value_spec_version'] != 'defense-we-pa-v1'
        or contract['training_cutoff'] != '2025-04-30'):
        raise ValueError('Compatible frozen C0 WE continuation required; do not fit substitute')
    bundle = Path(contract['bundle_path'])
    external = {str(path): hash_file(path)}
    for name in ('bundle_manifest.json', 'source_hashes.json'):
        expected = contract['sha256'][str(bundle / name)]
        if hash_file(bundle / name) != expected: raise ValueError('WE lineage metadata changed')
        external[str(bundle / name)] = expected
    source = read_json(bundle / 'source_hashes.json')
    for name in WE_SOURCE_FILES:
        if hash_file(PROJECT / name) != source[name]:
            raise ValueError('Frozen WE TRAIN-window/source lineage differs: '+name)
        external[str(PROJECT / name)] = source[name]
    manifest = read_json(bundle / 'bundle_manifest.json')['sha256']
    expected = contract['bundle_files']['game_values.pkl']
    if manifest['game_values.pkl'] != expected or hash_file(bundle / 'game_values.pkl') != expected:
        raise ValueError('Frozen WE/advancement bytes changed; no substitute fit authorized')
    external[str(bundle / 'game_values.pkl')] = expected
    return str(bundle / 'game_values.pkl'), external


def prepare(config, local_path, local, output, expected):
    if output.exists() and any(output.iterdir()):
        verify(output, expected); print('POLICY_PREPARED', flush=True); return
    parent = Path(config['parent_run']).resolve()
    check_location(local, parent)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Policy run must be a distinct sibling')
    parent_config = read_json(parent / 'registered_config.json')
    prep = verify_sharing(parent, sharing_identity(parent_config, local_path))
    if hash_file(parent / 'preparation.json') != config['parent_preparation_sha256']:
        raise ValueError('Parent preparation hash changed')
    results = parent / 'analysis/panel/results.json'
    if hash_file(results) != config['parent_analysis_sha256']: raise ValueError('Parent analysis hash changed')
    analysis = read_json(results)
    chosen = chosen_comparison(analysis)
    if (any(config[k] != chosen[k] for k in ('candidate', 'control'))
        or config['selection_status'] != chosen['status']):
        raise ValueError('Policy outcome-model selection rule changed')
    manifest_path = parent / 'analysis/panel/manifest.json'
    manifest = read_json(manifest_path)
    if manifest['results_sha256'] != config['parent_analysis_sha256']: raise ValueError('Parent analysis manifest differs')
    external = {**manifest['inputs'], str(results): hash_file(results), str(manifest_path): hash_file(manifest_path)}
    we_path, we_external = verify_we(config)
    external.update(we_external)
    external[str(parent / 'preparation.json')] = hash_file(parent / 'preparation.json')
    for cell in (chosen['candidate'], chosen['control']):
        for seed in SEEDS:
            member = parent / 'members' / cell / f'seed{seed}'
            state = read_json(member / 'prediction_state.json')
            if state['identity'] != {'preparation_sha256': canonical_hash(prep), 'cell': cell, 'seed': seed}:
                raise ValueError('Frozen G member identity differs')
            assert_hashes(member, state['artifact_hashes'])
            external.update(state['dependencies'])
            for name, digest in state['artifact_hashes'].items(): external[str(member / name)] = digest
    for name, digest in external.items():
        if hash_file(Path(name)) != digest: raise ValueError('Dependency changed before policy preparation: '+name)
    output.mkdir(parents=True)
    started = time.perf_counter()
    dump(output / 'preparing.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'identity': expected})
    store, context, parts, aux = load_data(local, parent, prep)
    bc = fit_bc(store, parts['train'], **config['bc'])
    frame, selections, positions = store.frame, {}, set()
    for split, count in config['maximum_requested_starts'].items():
        table = select_pa_requests(frame, prep['panel'], split, count, config['selection_seed'])
        selections[split] = table
        positions.update(int(x) for x in table.loc[table.selected, 'position'])
    rows = frame.loc[sorted(positions)]
    inputs = PolicyInputs(rows, context, aux['delivery'], prep['features']['tokens']['type_vocabulary'],
                          prep['artifact_hashes']['aux.pkl'], bc)
    states, coverage = {}, {}
    for split, table in selections.items():
        reasons = []
        for request in table.itertuples(index=False):
            if not request.selected: reasons.append('not_selected_by_fixed_budget'); continue
            row = frame.iloc[request.position]
            reason = None
            try:
                if int(row.pitch_number) != 1 or int(row.balls) != 0 or int(row.strikes) != 0:
                    reason = 'incomplete_pa_start'
                elif pd.isna(row.supported_pa) or not bool(row.supported_pa):
                    reason = 'retrospectively_unsupported_pa'
                else:
                    GameState.from_row(row)
                    if row.stand not in ('L', 'R') or row.p_throws not in ('L', 'R'):
                        raise ValueError('Invalid pre-pitch handedness')
                    s = state_from_row(store, request.position)
                    if not inputs.support(s).any(): reason = 'no_common_train_action_support'
                    else: states[s.context_key] = s
            except (ValueError, TypeError): reason = 'invalid_legal_state_or_past_history'
            reasons.append(reason)
        table['reason'] = reasons
        table['supported'] = table.selected & table.reason.isna()
        table.to_parquet(output / f'{split}_requests.parquet', index=False)
        coverage[split] = {'requested_pa_starts': len(table), 'selected_pa_starts': int(table.selected.sum()),
            'supported_selected': int(table.supported.sum()), 'reason_counts': table.reason.value_counts().to_dict(),
            'requested_start_key_hash': canonical_hash(table[KEY].to_numpy().tolist()),
            'selection_uses_outcomes': False, 'supported_pa_is_retrospective': True}
    # P0 uses every requested panel pitch. Current type is copied only as label;
    # previous type is obtained from the strict history index, never shifted
    # across PA boundaries or after dropping unsupported rows.
    for split in ('blend', 'dev'):
        requested = frame.loc[frame.split.eq(split) & frame.pitcher.isin(prep['panel']['pitcher_ids'])]
        previous = store.indices[requested.index, -1]
        prior_types = frame.pitch_type.to_numpy()
        table = requested[[*KEY, 'pitcher', 'p_throws', 'stand', 'balls', 'strikes']].copy()
        table['observed_action'] = requested.pitch_type.to_numpy()
        table['previous_action'] = [str(prior_types[i]) if i >= 0 and pd.notna(prior_types[i]) else '<START>' for i in previous]
        table.to_parquet(output / f'p0_{split}.parquet', index=False)
        # One safe representative context per pitcher/hand/side suffices for
        # support, which is intersected across ALL counts and ignores game state.
        templates = requested.drop_duplicates(['pitcher', 'p_throws', 'stand'])
        safe_rows(templates).to_parquet(output / f'p0_{split}_contexts.parquet', index=False)
    with (output / 'inputs.pkl').open('wb') as stream:
        pickle.dump({'rows': safe_rows(rows.loc[[context_key(row) in states for _, row in rows.iterrows()]]), 'states': states, 'bc': bc, 'context': context,
                     'delivery': aux['delivery'], 'baseline': aux['baseline']}, stream)
    for name in SOURCES:
        target = output / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    dump(output / 'registered_config.json', config)
    dump(output / 'parent_preparation.json', prep)
    dump(output / 'parent_config.json', parent_config)
    dump(output / 'parent_analysis.json', analysis)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    if source_hashes() != expected['source_hashes']: raise ValueError('Policy source changed during prepare')
    dump(output / 'preparation.json', {'identity': expected, 'external_hashes': external,
        'artifact_hashes': artifact_hashes(output, files), 'parent_run': str(parent), 'we_path': we_path,
        'coverage': coverage, 'selection': chosen, 'seconds': time.perf_counter()-started,
        'bc_train_pitches': sum(bc.league.values()), 'bc_actions': list(bc.actions),
        'we_train_scope': '2023-05-15 through 2025-04-30 inclusive; pinned builder+assign_fold(2025)',
        'we_prior_dev_exposure': 'Frozen service WE passed a 2025 DEV Brier gate during original build',
        'policy_effect': None, 'observational_ope': None})
    print('POLICY_PREPARED', coverage, flush=True)


class TimedBudget(RowBudget):
    def __init__(self, limit, seconds):
        super().__init__(limit, seed_count=3)
        self.deadline = time.monotonic()+seconds
    def consume(self, rows):
        if time.monotonic() > self.deadline: raise BudgetExceeded('Policy wall budget exceeded; preserve partial stage')
        super().consume(rows)


def resources(output, prep, config, budget):
    with (output / 'inputs.pkl').open('rb') as stream: saved = pickle.load(stream)
    parent_prep = read_json(output / 'parent_preparation.json')
    parent_config = read_json(output / 'parent_config.json')
    analysis = read_json(output / 'parent_analysis.json')
    inputs = PolicyInputs(saved['rows'], saved['context'], saved['delivery'],
        parent_prep['features']['tokens']['type_vocabulary'], parent_prep['artifact_hashes']['aux.pkl'], saved['bc'])
    with Path(prep['we_path']).open('rb') as stream: game_values = pickle.load(stream)
    we = FrozenWE(inputs, game_values)
    models = {}
    for role in ('candidate', 'control'):
        cell = config[role]; members, temperatures = [], []
        for seed in SEEDS:
            model, dependencies = predictor(parent_config, Path(prep['parent_run']), parent_prep, cell, seed)
            for name, digest in dependencies.items():
                if prep['external_hashes'].get(name) != digest: raise ValueError('G model dependency changed')
            members.append(model)
            temperatures.append(read_json(Path(prep['parent_run']) / 'members' / cell / f'seed{seed}' / 'calibration.json')['delivery_temperature'])
        models[role] = FrozenGEnsemble(inputs, members, temperatures, saved['baseline'],
            parent_prep['baseline_temperature']['temperature'], analysis['reports'][cell]['selection']['model_weight'])
    simulators = {role: JointSimulator(inputs.pool, model, we.terminal, budget) for role, model in models.items()}
    return saved, inputs, SupportedBC(inputs), we, models, simulators


def begin_stage(output, name, payload):
    path = output / 'stages' / name
    if path.exists(): raise ValueError('Preserve existing/completed/failed policy stage: '+str(path))
    path.mkdir(parents=True)
    dump(path / 'started.json', {**payload, 'preparation_sha256': hash_file(output / 'preparation.json'), 'started_utc': datetime.now(timezone.utc).isoformat()})
    return path


def finish_runtime(destination, started, budget, models):
    payload = {'seconds': time.perf_counter()-started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'conditional_rows': budget.conditional_rows, 'seed_predictor_rows': budget.network_rows,
        'actual_neural_subnetwork_rows': {k: v.actual_neural_network_rows for k, v in models.items()},
        'prediction_calls': budget.calls}
    dump(destination / 'runtime.json', payload)
    return payload


def profile(config, output, prep):
    spec = config['profile']
    destination = begin_stage(output, 'profile', {'spec': spec, 'quality_scores_read': False})
    started = time.perf_counter(); budget = TimedBudget(spec['row_budget'], spec['seconds_budget'])
    saved, inputs, bc, we, models, simulators = resources(output, prep, config, budget)
    table = pd.read_parquet(output / 'temperature_requests.parquet')
    selected = table.loc[table.selected].head(spec['requested_starts'])
    used = 0
    for row in selected.loc[selected.supported].itertuples(index=False):
        state = saved['states'][':'.join(str(int(getattr(row, k))) for k in KEY)]
        planner = RolloutImprovement(bc, simulators['candidate'], we.cutoff,
            samples=spec['search_rollouts'], pitch_cap=spec['search_cap'], seed=spec['planning_seed'])
        uniforms = np.random.default_rng(spec['evaluation_seed']+int(row.selection_rank)).random((spec['evaluation_rollouts'], spec['evaluation_cap'], 3))
        # Actual model cost, but no outcome values, policy ranking or NLL exposed.
        rollouts(simulators['control'], [state]*len(uniforms), planner.policy('P2'), uniforms, we.cutoff)
        used += 1
        print('POLICY_PROFILE_CASE', used, 'conditional_rows', budget.conditional_rows, flush=True)
    if not used: raise ValueError('No supported May profile starts; cannot register execution from empty profile')
    runtime = finish_runtime(destination, started, budget, models)
    dump(destination / 'result.json', {'quality_scores_read': False, 'split': 'May16-31 temperature only',
        'requested_starts': len(selected), 'supported_used': used, 'runtime': runtime,
        'conditional_rows_per_second_including_load': budget.conditional_rows/max(runtime['seconds'], 1e-9),
        'note': 'Resource feasibility only; final execution registration must be frozen separately.'})


def p0_evaluate(config, output, prep):
    destination = begin_stage(output, 'p0', {'model': 'TRAIN categorical BC', 'current_pitch_type': 'label only'})
    started = time.perf_counter()
    with (output / 'inputs.pkl').open('rb') as stream: saved = pickle.load(stream)
    parent_prep = read_json(output / 'parent_preparation.json')
    results = {}
    for split in ('blend', 'dev'):
        rows = pd.read_parquet(output / f'p0_{split}.parquet')
        templates = pd.read_parquet(output / f'p0_{split}_contexts.parquet')
        inputs = PolicyInputs(templates, saved['context'], saved['delivery'],
            parent_prep['features']['tokens']['type_vocabulary'], parent_prep['artifact_hashes']['aux.pkl'], saved['bc'])
        bc = SupportedBC(inputs)
        keys = {(str(int(row.pitcher)), row.p_throws, row.stand): context_key(row) for _, row in templates.iterrows()}
        probability = np.zeros((len(rows), len(bc.actions))); frequency = probability.copy()
        supported, observed_supported, fallback = np.zeros((3, len(rows)), dtype=bool)
        labels = np.full(len(rows), -1, dtype=int)
        for i, row in enumerate(rows.itertuples(index=False)):
            if row.observed_action in bc.actions: labels[i] = bc.actions.index(row.observed_action)
            try:
                history = () if row.previous_action == '<START>' else (PastPitch(row.previous_action, (), 'unknown', 0, 0),)
                state = PAState(int(row.balls), int(row.strikes), str(int(row.pitcher)), row.stand, history,
                               keys[(str(int(row.pitcher)), row.p_throws, row.stand)])
                probability[i], frequency[i] = bc.probabilities(state), bc.probabilities(state, frequency=True)
                supported[i], fallback[i] = True, bc.fallback(state)
                observed_supported[i] = labels[i] >= 0 and probability[i, labels[i]] > 0
            except (ValueError, KeyError, TypeError): pass
        valid = np.flatnonzero(observed_supported)
        metrics = {name: float(-np.log(p[valid, labels[valid]]).mean()) if len(valid) else None
                   for name, p in (('bc_supported_action_nll', probability), ('frequency_supported_action_nll', frequency))}
        results[split] = {**metrics, 'requested_pitches': len(rows), 'supported_requests': int(supported.sum()),
            'supported_observed_labels': len(valid), 'unsupported_observed_labels': int((~observed_supported).sum()),
            'unknown_pitcher_fallbacks': int(fallback.sum()),
            'full_requested_action_nll': metrics['bc_supported_action_nll'] if observed_supported.all() else None,
            'full_requested_nll_note': 'Null when any observed label lacks common action support; finite subset NLL is conditional.',
            'action_agreement_is_not_policy_value': True}
        np.savez_compressed(destination / f'{split}.npz', keys=rows[KEY].to_numpy(), labels=labels,
            bc=probability, frequency=frequency, supported=supported, observed_supported=observed_supported, fallback=fallback)
    dump(destination / 'results.json', {'results': results, 'seconds': time.perf_counter()-started, 'policy_effect': None})


def execution_check(execution, output, config):
    if (execution['protocol'] != 'ml_policy_execution_v1'
        or execution['preparation_sha256'] != hash_file(output / 'preparation.json')
        or execution['profile_result_sha256'] != hash_file(output / 'stages/profile/result.json')
        or execution['tau_grid'] != [.001, .003, .01, .03]
        or execution['bootstrap_draws'] != 10000 or execution['bootstrap_seed'] != 20260924):
        raise ValueError('Final policy execution registration differs')
    for name in ('search_rollouts', 'search_cap', 'evaluation_rollouts', 'evaluation_cap', 'row_budget', 'seconds_budget'):
        if type(execution[name]) is not int or execution[name] < 1: raise ValueError('Invalid execution resource: '+name)
    if execution['evaluation_rollouts'] < 2: raise ValueError('At least two MC evaluation trajectories required')
    for split in ('blend', 'dev'):
        if not 0 < execution['requested_starts'][split] <= config['maximum_requested_starts'][split]:
            raise ValueError('Final PA requests exceed frozen selection prefix')
    return execution


def policy_run(config, output, prep, execution, split, world):
    if split == 'blend' and world != 'control': raise ValueError('Tune only in the common control world')
    destination = begin_stage(output, f'{split}-{world}', {'execution': execution, 'world': world, 'split': split})
    started = time.perf_counter(); budget = TimedBudget(execution['row_budget'], execution['seconds_budget'])
    saved, inputs, bc, we, models, simulators = resources(output, prep, config, budget)
    table = pd.read_parquet(output / f'{split}_requests.parquet')
    selected = table.loc[table.selected].head(execution['requested_starts'][split])
    if split == 'blend':
        policies = [('P0', None), ('P2', None), *[(f'P3-tau{tau}', tau) for tau in execution['tau_grid']]]
    else:
        tuning_path = output / 'stages/blend-control/results.json'
        tuning = read_json(tuning_path)
        if tuning['execution_sha256'] != canonical_hash(execution): raise ValueError('Tune/DEV execution differs')
        dump(destination / 'tuning_dependency.json', {'path': str(tuning_path), 'sha256': hash_file(tuning_path)})
        policies = [('P0', None), ('P1', None), ('P2', None), ('P3', tuning['selected_tau'])]
    values = {name: [] for name, _ in policies}; truncation = {name: [] for name, _ in policies}
    games, pa_keys, diagnostics = [], [], []
    for row in selected.loc[selected.supported].itertuples(index=False):
        key = ':'.join(str(int(getattr(row, k))) for k in KEY); state = saved['states'][key]
        planner = RolloutImprovement(bc, simulators['candidate'], we.cutoff,
            samples=execution['search_rollouts'], pitch_cap=execution['search_cap'], seed=execution['planning_seed'])
        uniforms = np.random.default_rng(np.random.SeedSequence([execution['evaluation_seed'],
            0 if split == 'blend' else 1, int(row.game_pk), int(row.at_bat_number)])).random(
                (execution['evaluation_rollouts'], execution['evaluation_cap'], 3))
        archive = {}
        for name, tau in policies:
            mode = 'P3' if tau is not None else name
            result = rollouts(simulators[world], [state]*len(uniforms), planner.policy(mode, tau=tau), uniforms, we.cutoff)
            values[name].append(result.values); truncation[name].append(result.truncated)
            archive.update({name+'_values': result.values, name+'_truncated': result.truncated, name+'_pitches': result.pitches})
            print('POLICY_CASE', split, world, key, name, 'rows', budget.conditional_rows, flush=True)
        np.savez_compressed(destination / (key.replace(':', '-')+'.npz'), **archive)
        games.append(int(row.game_pk)); pa_keys.append(key); diagnostics.append(dict(planner.diagnostics))
        dump(destination / 'progress.json', {'completed_pa': len(games), 'conditional_rows': budget.conditional_rows,
            'last_pa': key, 'policy_effect': None})
    if not games: raise ValueError('No supported selected PA starts; preserve denominator, do not replace')
    arrays = {name: Rollouts(np.stack(values[name]), np.stack(truncation[name]), np.empty((0,))) for name, _ in policies}
    comparisons = {name+'_minus_P0': paired_summary(result, arrays['P0']) for name, result in arrays.items() if name != 'P0'}
    if split == 'dev':
        comparisons.update({'P2_minus_P1': paired_summary(arrays['P2'], arrays['P1']),
                            'P3_minus_P2': paired_summary(arrays['P3'], arrays['P2'])})
    means = {name: float(result.values.mean()) for name, result in arrays.items()}
    chosen = None
    if split == 'blend': chosen = max(execution['tau_grid'], key=lambda tau: (means[f'P3-tau{tau}'], tau))
    runtime = finish_runtime(destination, started, budget, models)
    dump(destination / 'results.json', {'execution_sha256': canonical_hash(execution), 'split': split, 'world': world,
        'requested_pa_starts': len(table), 'selected_pa_starts': len(selected), 'supported_selected': len(games),
        'selected_unsupported_reasons': selected.loc[~selected.supported].reason.value_counts().to_dict(),
        'pa_keys': pa_keys, 'game_ids': games, 'mean_original_defensive_we': means, 'comparisons': comparisons,
        'selected_tau': chosen,
        'rl_comparator': ('P3' if means[f'P3-tau{chosen}'] >= means['P2'] else 'P2') if split == 'blend' else None,
        'runtime': runtime, 'planner_diagnostics': diagnostics,
        'delivery_tier_draw_calls': {str(k): int(v) for k, v in inputs.pool_tiers.items()},
        'primary_family': game_policy_comparisons({k: r.values for k, r in arrays.items()}, games,
            truncated={k: r.truncated for k, r in arrays.items()},
            draws=execution['bootstrap_draws'], seed=execution['bootstrap_seed']) if split == 'dev' else None,
        'family_interpretation': 'Primary cross-model internal' if world == 'control' else 'Sensitivity only; not a second success family',
        'model_internal_only': True, 'evaluator_independent_data': False, 'observational_ope': None,
        'causal_effect': None, 'policy_adoption': None,
        'sampling_inference': 'PA-weighted whole-game bootstrap of simulated conditional means; fixed model, calibration and selection; separate MC SE',
        'truncation_tail': 'Frozen WE at initial game state, independent of count; bounds also reported'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--local-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'profile', 'p0-evaluate'): sub.add_parser(command)
    run = sub.add_parser('policy-run')
    run.add_argument('--execution-config', required=True, type=Path)
    run.add_argument('--split', required=True, choices=('blend', 'dev'))
    run.add_argument('--world', required=True, choices=('control', 'candidate'))
    args = parser.parse_args()
    config, local = config_check(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve(); root = check_location(local, output)
    validate_native_runtime(); expected = identity(config, args.local_config)
    with heavy_lock(root):
        try:
            if args.command == 'prepare': prepare(config, args.local_config, local, output, expected)
            else:
                prep = verify(output, expected)
                if args.command == 'profile': profile(config, output, prep)
                elif args.command == 'p0-evaluate': p0_evaluate(config, output, prep)
                else: policy_run(config, output, prep, execution_check(read_json(args.execution_config), output, config), args.split, args.world)
            if source_hashes() != expected['source_hashes']:
                raise ValueError('Policy source changed during stage; results invalid pending audit')
        except Exception as error:
            # Fresh timestamped audit; never overwrite a completed or failed stage.
            if output.is_dir():
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                dump(output / ('failure-'+stamp+'.json'), {'command': args.command,
                    'error_type': type(error).__name__, 'error': str(error),
                    'completed': False, 'preserve_partial_artifacts': True})
            raise


if __name__ == '__main__': main()
