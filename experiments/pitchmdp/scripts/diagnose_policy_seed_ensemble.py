"""EXP-STAB-01: does averaging K planning seeds make the G0 P3 top-1 recommendation stable?

Stability / noise-floor study (D155 side observation: changing only the planning seed 0 -> 1 moved
the top-1 pitch type in 11.3% of sealed June decisions).  The predictor stays G0 and every other
policy setting stays the registered one (tau 0.1, samples 6, pitch cap 12).  On 2025 July-September
DEV games that the validation chain never selected (S5 and S6 games excluded), the planning Q is
computed with ten planning seeds 0..9.  For K in {1, 3, 5} the K-seed ensemble Q of set A
(seeds 0..K-1) and of the disjoint set B (seeds 5..5+K-1) is the per-action mean planning Q; each
gives a P3 law, and the two laws are compared (top-1 agreement, TV).  K = 1 ({0} vs {5}) is the
D155-style control.

The decision statuses come from the registered runtime (seed 0 planner, no evaluation seed, no
outcome).  No reward, WE result, events kind or next-row state is read; no policy value is estimated
and nothing is registered as a policy identity.

``--smoke`` limits the games to check the wiring and the cost; it writes to an output ending
``-smoke`` and neither computes the comparison nor writes the repository result.
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
import pandas as pd
from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.policy_artifacts import load_style_snapshot, load_support_table, load_train_bc
from pitchmdp import policy_identity as pid
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as prt
from pitchmdp.rollout_policy import JointSimulator, RolloutImprovement, RowBudget
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock
from run_ml_delivery_pool import git_commit, peak
import run_policy_validation as rpv
import diagnose_policy_predictor_swap as swap

EXPERIMENT = 'EXP-STAB-01'
SOURCES = (*swap.SOURCES, 'scripts/diagnose_policy_seed_ensemble.py')
SMOKE_GAMES = 3
require = swap.require


class DeadlinePassed(RuntimeError):
    pass


def config_check(config, smoke=False):
    require(config['protocol'] == 'policy_seed_ensemble_stability_v1' and config['experiment_id'] == EXPERIMENT,
            'Diagnostic settings differ from registered protocol')
    sets = config['seed_sets']
    a, b = sets['A'], sets['B']
    require(len(a) == len(b) == max(sets['K']) and not set(a) & set(b) and all(type(s) is int for s in a + b),
            'Two disjoint seed lists of length max(K) required')
    require(all(type(k) is int and 1 <= k <= len(a) for k in sets['K']) and sets['K'][0] == 1,
            'K must start with the K = 1 control')
    require(config['settings']['seed'] == a[0], 'Set A must start with the registered planning seed')
    rule = config['verdict_rule']
    require(0 < rule['stable_if_top1_agreement_ci95_lower_at_least'] <= 1, 'Registered verdict rule required')
    for key in ('policy_registration', 'settings', 'games', 'bootstrap', 'deadline_utc', 'row_budget_per_seed'):
        require(config.get(key), 'Registered input missing: ' + key)
    require(smoke or config['execution']['enabled'] is True, 'Execution not enabled in the registered config')
    return config


def ensemble_law(qs, reference, mask, tau):
    """P3 law of the per-action mean planning Q of the given seeds."""
    return swap.law(np.mean(qs, axis=0), reference, mask, tau)


def compare(q_by_seed, reference, mask, tau, sets):
    """Per-decision A-vs-B distances for every K (Q-mean primary, law-mean secondary)."""
    out = {}
    for k in sets['K']:
        qa = [q_by_seed[s] for s in sets['A'][:k]]
        qb = [q_by_seed[s] for s in sets['B'][:k]]
        pa, pb = ensemble_law(qa, reference, mask, tau), ensemble_law(qb, reference, mask, tau)
        la = np.mean([swap.law(q, reference, mask, tau) for q in qa], axis=0)
        lb = np.mean([swap.law(q, reference, mask, tau) for q in qb], axis=0)
        out[f'K{k}_tv'] = .5 * float(np.abs(pa - pb).sum())
        out[f'K{k}_top1_same'] = bool(np.argmax(pa) == np.argmax(pb))
        out[f'K{k}_lawmean_tv'] = .5 * float(np.abs(la - lb).sum())
        out[f'K{k}_lawmean_top1_same'] = bool(np.argmax(la) == np.argmax(lb))
        out[f'K{k}_max_abs_q_difference'] = float(np.abs(np.mean(qa, 0)[mask] - np.mean(qb, 0)[mask]).max())
    return out


def summarize(columns, games, sets, rule, *, draws, seed):
    """Point values, whole-game bootstrap intervals and the registered verdict."""
    games = np.asarray(games)
    unique, index = np.unique(games, return_inverse=True)
    picks = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    weight = np.bincount(index, minlength=len(unique)).astype(float)
    wboot = weight[picks].sum(axis=1)

    def ci(values):
        sums = np.bincount(index, weights=np.asarray(values, dtype=float), minlength=len(unique))
        return [float(v) for v in np.quantile(sums[picks].sum(axis=1) / wboot, (.025, .975))]

    per_k = {}
    for k in sets['K']:
        tv, same = np.asarray(columns[f'K{k}_tv']), np.asarray(columns[f'K{k}_top1_same'], dtype=float)
        ltv, lsame = np.asarray(columns[f'K{k}_lawmean_tv']), np.asarray(columns[f'K{k}_lawmean_top1_same'], dtype=float)
        agreement_ci = ci(same)
        per_k[str(k)] = {'seeds_A': sets['A'][:k], 'seeds_B': sets['B'][:k],
                         'top1_agreement': float(same.mean()), 'top1_agreement_ci95': agreement_ci,
                         'mean_tv': float(tv.mean()), 'mean_tv_ci95': ci(tv),
                         'tv_quantiles': {str(q): float(np.quantile(tv, q)) for q in (.5, .9, .99, 1.)},
                         'share_tv_above_0.05': float((tv > .05).mean()),
                         'mean_max_abs_q_difference': float(np.mean(columns[f'K{k}_max_abs_q_difference'])),
                         'secondary_law_mean': {'top1_agreement': float(lsame.mean()), 'mean_tv': float(ltv.mean())},
                         'stable': bool(agreement_ci[0] >= rule['stable_if_top1_agreement_ci95_lower_at_least'])}
    base = np.asarray(columns['K1_top1_same'], dtype=float)
    base_tv = np.asarray(columns['K1_tv'])
    paired = {}
    for k in sets['K'][1:]:
        gain = np.asarray(columns[f'K{k}_top1_same'], dtype=float) - base
        drop = base_tv - np.asarray(columns[f'K{k}_tv'])
        paired[f'K{k}_minus_K1'] = {'top1_agreement_gain': float(gain.mean()), 'top1_agreement_gain_ci95': ci(gain),
                                    'mean_tv_reduction': float(drop.mean()), 'mean_tv_reduction_ci95': ci(drop)}
    stable = [k for k in sets['K'] if per_k[str(k)]['stable']]
    largest = max(sets['K'])
    improves = largest > 1 and paired[f'K{largest}_minus_K1']['top1_agreement_gain_ci95'][0] > 0
    status = f'STABLE_AT_K{min(stable)}' if stable else f'NOT_STABLE_UP_TO_K{largest}'
    return {'decisions': len(base), 'games': len(unique), 'by_K': per_k, 'paired_vs_K1': paired,
            'verdict': {**rule, 'status': status, 'averaging_improves_agreement': bool(improves),
                        'policy_value': 'NOT_MEASURED (2026-10-20 Song decision: stability only)'}}


def identity(local_path):
    return {'config_sha256': hash_file(REPO / 'configs' / (EXPERIMENT + '.yaml')), 'local_config_sha256': hash_file(local_path),
            'git_commit': git_commit(), 'source_hashes': {rel: hash_file(PROJECT / rel) for rel in SOURCES}}


def stab_games(frame, policy, config, smoke):
    """DEV games outside the S5 and S6 selections, by the registered month-proportional salted-hash rule."""
    salt = rpv.role_seed(policy, 'selection_salt')
    stages = policy['le2025_validation_plan']['stages']
    used = set(rpv.select_games(frame, 'dev', stages['S6_V4']['n_games'], salt))
    used |= set(rpv.select_games(frame, 'dev', stages['S5_V2_V3']['n_games'], salt))
    spec = config['games']
    require(spec['split'] == 'dev' and spec['exclude'] == ['S5_V2_V3', 'S6_V4'], 'Registered game rule differs')
    rest = frame.loc[~frame.game_pk.isin(used)]
    games = rpv.select_games(rest, 'dev', spec['n_games'], spec['salt'])
    require(not set(games) & used, 'A selected game was used by the validation chain')
    months = pd.to_datetime(frame.loc[frame.game_pk.isin(games) & frame.split.eq('dev'), 'game_date'])
    require(months.min() >= pd.Timestamp('2025-07-01') and months.max() < pd.Timestamp('2025-10-01'),
            'Selected games fall outside July-September 2025')
    record = {'excluded_games': len(used), 'selected': len(games), 'games_sha256': canonical_hash(sorted(games)),
              'first_date': str(months.min().date()), 'last_date': str(months.max().date())}
    return (games[:SMOKE_GAMES] if smoke else games), record


def run(config, local, local_path, output, smoke):
    started = time.perf_counter()
    deadline = pd.Timestamp(config['deadline_utc'])
    reg = rpv.load_registration(REPO / config['policy_registration']['config'],
                                [REPO / path for path in config['policy_registration']['addenda']])
    require(reg['config_sha256'] == config['policy_registration']['config_sha256'], 'Policy registration config changed')
    policy, settings, sets = reg['config'], config['settings'], config['seed_sets']
    freeze = rpv.registered_json(reg, 'tau_freeze')
    require(freeze['selected_tau'] == settings['tau'] and freeze['setting'] == {'samples': settings['samples'],
            'pitch_cap': settings['pitch_cap']} and policy['seeds']['planning_main'] == settings['seed'],
            'Settings differ from the registered tau freeze')
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'started.json', {'identity': identity(local_path), 'registration_chain': reg['chain'],
                                   'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': smoke,
                                   'outcomes_read': False})
    inputs = rpv.load_inputs(policy, local)
    frame, store = inputs['frame'], inputs['store']
    no_pitch = frozenset(policy['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
    games, game_record = stab_games(frame, policy, config, smoke)
    blocks = rpv.game_blocks(frame, games, 'dev')
    bc_path, bc_sha = rpv.registered_path(reg, 'bc')
    support_path, support_sha = rpv.registered_path(reg, 'support')
    artifact = load_train_bc(bc_path, bc_sha)
    snapshot, as_of, content = load_style_snapshot(*rpv.registered_path(reg, 'style_dev'))
    require(as_of == policy['le2025_validation_plan']['stages']['S1b_style_snapshot']['as_of_exclusive']['dev'],
            'Style as-of differs from registration')
    first = pd.to_datetime(frame.game_date.iloc[np.concatenate([p for _, p in blocks])]).min()
    require(first >= pd.Timestamp(as_of), 'Evaluation blocks start before the style as-of date')
    components = swap.bind(reg, inputs, blocks, artifact, snapshot, as_of)
    budgets = {s: RowBudget(int(config['row_budget_per_seed']), seed_count=5) for s in sets['A'] + sets['B']}
    runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, output / 'ledger-dev-0.jsonl',
                                components=components, budget=budgets[settings['seed']], tau=settings['tau'],
                                samples=settings['samples'], pitch_cap=settings['pitch_cap'], seed=settings['seed'],
                                hand_registry=rpv.registered_path(reg, 'hands'),
                                provenance={'style_snapshot_sha256': content, 'as_of_exclusive': as_of,
                                            'experiment': EXPERIMENT})
    reference, we = runtime.reference, components.we
    planners = {settings['seed']: runtime.improvement}
    for s in sets['A'] + sets['B']:
        if s != settings['seed']:
            planners[s] = RolloutImprovement(reference, JointSimulator(components.inputs.pool, components.g0, we.terminal,
                                             budgets[s]), we.cutoff, samples=settings['samples'],
                                             pitch_cap=settings['pitch_cap'], seed=s)
    replay_check = RolloutImprovement(reference, JointSimulator(components.inputs.pool, components.g0, we.terminal,
                                      RowBudget(int(config['row_budget_per_seed']), seed_count=5)), we.cutoff,
                                      samples=settings['samples'], pitch_cap=settings['pitch_cap'], seed=settings['seed'])
    tau, columns, row_games, request_ids, replay = settings['tau'], {}, [], [], []
    statuses, done_games, last = {}, set(), None
    for pa_id, positions in blocks:
        if pd.Timestamp.now(tz='UTC') >= deadline:
            raise DeadlinePassed(f'registered deadline {deadline} passed after {len(done_games)} games')
        game = int(frame.game_pk.iloc[positions[0]])
        requests, _, _ = preq.pa_requests(store, positions, runtime.sha256, no_pitch)
        for request in requests:
            row = runtime.submit(request)
            statuses[row['status']] = statuses.get(row['status'], 0) + 1
            if row['status'] not in prt.EVALUATED:
                continue
            state, result = request.state, row['result']
            mask, ref = reference.support(state), reference.probabilities(state)
            require(np.array_equal(mask, np.asarray(result['mask'], dtype=bool))
                    and np.allclose(ref, result['reference'], rtol=0, atol=1e-12), 'Runtime reference differs')
            q_runtime = np.array([np.nan if v is None else v for v in result['q_planning']], dtype=np.float64)
            q_by_seed = {s: planner.q_values(state)[0] for s, planner in planners.items()}
            require(np.array_equal(q_by_seed[settings['seed']][mask], q_runtime[mask]), 'Seed-0 Q differs from the ledger')
            if len(replay) < config['replay']['decisions']:
                replay.append(float(np.abs(replay_check.q_values(state)[0][mask] - q_runtime[mask]).max()))
                require(replay[-1] <= config['replay']['atol'], 'Fresh seed-0 planner does not reproduce the runtime Q')
            require(all(np.isfinite(q[mask]).all() for q in q_by_seed.values()), 'Non-finite planning Q on the support')
            for key, value in compare(q_by_seed, ref, mask, tau, sets).items():
                columns.setdefault(key, []).append(value)
            row_games.append(game)
            request_ids.append(request.request_id)
        if game != last:
            done_games.add(game)
            last = game
            if len(done_games) % 10 == 0:
                print('PROGRESS', {'games': len(done_games), 'decisions': len(request_ids),
                                   'seconds': round(time.perf_counter() - started)}, flush=True)
    runtime.verify_components()
    components.verify()
    require(len(replay) == min(config['replay']['decisions'], len(request_ids)), 'Replay count short')
    summary = None
    if not smoke:
        np.savez_compressed(output / 'decisions.npz', request_id=np.array(request_ids), game_pk=np.array(row_games),
                            **{k: np.array(v) for k, v in columns.items()})
        summary = summarize(columns, row_games, sets, config['verdict_rule'], draws=config['bootstrap']['draws'],
                            seed=config['bootstrap']['seed'])
    dataset = inputs['parent'].get('dataset_identity', {})
    result = {'experiment_id': EXPERIMENT, 'smoke': smoke, 'settings': settings, 'seed_sets': sets,
              'games': game_record, 'requests': sum(statuses.values()), 'statuses': statuses,
              'decisions_compared': len(request_ids), 'comparison': summary,
              'replay': {'decisions': len(replay), 'max_abs_q_difference': max(replay) if replay else None},
              'data_version': {'processed_sha256': dataset.get('processed_sha256'),
                               'sources': dataset.get('sources'), 'split': 'dev (2025-07-01..2025-09-30)',
                               'style_snapshot_sha256': content, 'components_sha256': components.sha256,
                               'ledger_sha256': hash_file(output / 'ledger-dev-0.jsonl')},
              'cost': {'seconds': time.perf_counter() - started, 'peak_rss_bytes': peak(),
                       'conditional_rows_by_seed': {str(s): b.conditional_rows for s, b in budgets.items()}},
              'git_commit': git_commit(), 'outcomes_read': False, 'policy_identity_registered': False,
              'limits': ['Stability of the planning law only; no policy value or OPE estimate',
                         'DEV July-September 2025 is exposed development data (no outcome read here)',
                         'The 0.95 agreement threshold is a judgement value borrowed from the D153 service criterion',
                         'Ensemble = per-action mean planning Q of K seeds; equivalent to more search samples, '
                         'not a registered policy identity'],
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    dump(output / 'results.json', result)
    if not smoke:
        repo_result = REPO / 'results' / (EXPERIMENT + '.json')
        require(not repo_result.exists(), 'Preserve the completed repository result')
        shutil.copyfile(output / 'results.json', repo_result)
    print('SEED_ENSEMBLE_COMPLETE', {'decisions': len(request_ids), 'smoke': smoke,
                                     'seconds': round(result['cost']['seconds'], 1)},
          '' if smoke else summary['verdict']['status'], flush=True)


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
        try:
            run(config, local, args.local_config.resolve(), output, args.smoke)
        except DeadlinePassed as stop:
            dump(output / 'aborted.json', {'reason': str(stop), 'utc': datetime.now(timezone.utc).isoformat(),
                                           'partial_results_discarded': True})
            print('SEED_ENSEMBLE_ABORTED', str(stop), flush=True)
            raise SystemExit(3)


if __name__ == '__main__':
    main()
