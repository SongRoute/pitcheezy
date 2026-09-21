"""Frozen-model sensitivity of hypothetical sequence/WE recommendations.

No training, policy-effect estimation, or intended-target identification occurs.
The same TRAIN-supported actions are held fixed across all scenarios for a case.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd

from pitchmdp.game import GameState, terminal_values
from pitchmdp.planner import solve_pa
from pitchmdp.sequence_model import SequenceModel
from pitchmdp.sequence_planner import solve_lookahead
from recommend_sequence import (action_deliveries, project_illegal_double_play,
                                select_actions, sequence_predictor, sha256)


SIGMAS = (.15, .30, .45)


def squared_distance_medoid(pool):
    """Return one retained vector minimizing sum of squared distances.

    Only standardized nonlocation channels enter the distance. This is the
    squared-Euclidean medoid, equivalent to the member nearest the pool mean.
    The retained pool already contains frozen train-only imputations; selecting
    a member cannot certify that every original physical measurement existed.
    """
    pool = np.asarray(pool, dtype=float)
    if pool.ndim != 2 or pool.shape[1] != 8 or not len(pool) or not np.isfinite(pool).all():
        raise ValueError('Expected a nonempty finite TRAIN pool with eight channels')
    features = pool[:, :6]
    distances = ((features - features.mean(axis=0)) ** 2).sum(axis=1)
    index = int(np.argmin(distances))
    return index, pool[index].copy(), float(distances[index])


def controlled_points(row, actions, delivery, normalizer, sigma, physics):
    """Keep identical target quadrature while changing nonlocation assumptions."""
    if physics not in ('mean', 'train_medoid'):
        raise ValueError('physics must be mean or train_medoid')
    points, weights, tiers = action_deliveries(row, actions, delivery, normalizer, sigma)
    keys = list(dict.fromkeys(k for tier in delivery.TIERS for k in tier))
    records = [{**{k: row[k] for k in keys}, 'pitch_type': a['pitch_type']} for a in actions]
    pools, second_tiers = delivery.sample(pd.DataFrame(records))
    if not np.array_equal(tiers, second_tiers):
        raise ValueError('Delivery pool selection changed within a scenario')
    details = []
    for index, pool in enumerate(pools):
        medoid_index, medoid, distance = squared_distance_medoid(pool)
        if physics == 'train_medoid':
            points[index, :, :6] = medoid[:6]
        nonlocation = np.asarray(points[index, 0, :6], dtype=float)
        spin = nonlocation[2:4] * normalizer.scale[2:4] + normalizer.mean[2:4]
        details.append({'action_index': index, 'pool_tier': int(tiers[index]),
                        'pool_rows': len(pool), 'medoid_pool_row_index': medoid_index,
                        'medoid_squared_distance_to_mean': distance,
                        'nonlocation_standardized': nonlocation.tolist(),
                        'spin_sine_cosine_norm_after_inverse_standardization': float(np.linalg.norm(spin)),
                        'pool_sha256': hashlib.sha256(np.ascontiguousarray(pool).tobytes()).hexdigest()})
    return points, weights, details


def summarize_stability(scenarios, reference_id='mean_sigma0.30_depth2'):
    by_id = {s['scenario_id']: s for s in scenarios}
    reference = by_id[reference_id]
    reference_action = reference['best_action_index']
    winners = Counter(s['best_action_index'] for s in scenarios)
    for scenario in scenarios:
        value_by_action = {a['action_index']: a['model_internal_defense_we'] for a in scenario['ranked_actions']}
        scenario['same_best_action_as_reference'] = scenario['best_action_index'] == reference_action
        # All values in this subtraction use the *same* scenario's model/kernel.
        scenario['reference_action_shortfall_in_scenario_pp'] = float(
            100 * (scenario['best_model_internal_defense_we'] - value_by_action[reference_action]))
    return {'reference_scenario': reference_id, 'reference_best_action_index': reference_action,
            'scenario_count': len(scenarios), 'unique_best_actions': len(winners),
            'all_scenarios_same_best_action': len(winners) == 1,
            'reference_action_best_in_scenarios': winners[reference_action],
            'best_action_counts': {str(action): count for action, count in sorted(winners.items())},
            'largest_reference_action_shortfall_pp': max(s['reference_action_shortfall_in_scenario_pp'] for s in scenarios),
            'best_value_range_across_scenarios': [min(s['best_model_internal_defense_we'] for s in scenarios),
                                                max(s['best_model_internal_defense_we'] for s in scenarios)],
            'interpretation': 'Sensitivity of a fixed model and assumed control kernels; no observed or causal policy gain.'}


def evaluate_case(model, encoders, planning, row, max_actions=6):
    if int(row.pitch_number) != 1 or int(row.balls) != 0 or int(row.strikes) != 0:
        raise ValueError('Only saved 0-0 first-pitch cases are supported; later pitches require observed history')
    if 'supported_pa' in row and not bool(row.supported_pa):
        raise ValueError('Unsupported PA: mid-PA game state may change')
    available = planning['actions'][(int(row.pitcher), str(row.stand))]
    actions = select_actions(available, max_actions)
    if not actions:
        raise ValueError('No supported actions')
    state = GameState.from_row(row)
    terminal = terminal_values(state, planning['we'], planning['advancement'])
    count_frame = pd.DataFrame([{**row.to_dict(), 'balls': b, 'strikes': s}
                               for b in range(4) for s in range(3)])
    base_p = project_illegal_double_play(planning['baseline'].predict(count_frame), row)
    tail = solve_pa(base_p.reshape(4, 3, 1, 1, 10), terminal, [0]).baseline_values[:, :, 0]
    predictor = sequence_predictor(model, encoders['context'], row)
    usage = np.array([a['training_pitch_type_n'] / sum(b['pitch_type'] == a['pitch_type'] for b in actions)
                      for a in actions], dtype=float)
    usage /= usage.sum()
    scenarios, computations = [], []
    for physics in ('mean', 'train_medoid'):
        for sigma in SIGMAS:
            start = time.perf_counter()
            points, weights, detail = controlled_points(row, actions, encoders['delivery'], encoders['normalizer'], sigma, physics)
            plan = solve_lookahead(np.empty((0, 8)), 0, 0, points, predictor, terminal,
                                  lambda b, s, h: tail[b, s], weights=weights, max_depth=2)
            computation_id = f'{physics}_sigma{sigma:.2f}'
            computations.append({'computation_id': computation_id, 'solver': plan.diagnostics,
                                 'seconds': time.perf_counter() - start, 'delivery_details': detail,
                                 'depth_one_reuses_root_one_step_q': True})
            for depth, q in ((1, plan.one_step_q_values), (2, plan.q_values)):
                order = np.argsort(-q, kind='stable')
                ranking = [{**actions[int(i)], 'action_index': int(i),
                            'model_internal_defense_we': float(q[i]), 'root_usage_weight': float(usage[i])}
                           for i in order]
                scenarios.append({'scenario_id': f'{computation_id}_depth{depth}',
                                  'computation_id': computation_id, 'physics': physics,
                                  'sigma_ft': sigma, 'max_depth': depth,
                                  'best_action_index': int(order[0]),
                                  'best_model_internal_defense_we': float(q[order[0]]),
                                  'best_minus_runner_up_pp': float(100 * (q[order[0]] - q[order[1]])) if len(order) > 1 else None,
                                  'root_usage_weighted_model_internal_value': float(usage @ q),
                                  'ranked_actions': ranking})
    stability = summarize_stability(scenarios)
    return {'input': {name: int(row[name]) for name in ['game_pk', 'pitcher', 'batter', 'at_bat_number', 'pitch_number', 'balls', 'strikes', 'outs_when_up', 'bases']},
            'game_date': str(pd.Timestamp(row.game_date).date()), 'stand': str(row.stand),
            'available_supported_actions': len(available), 'selected_actions': actions,
            'actual_history_length': 0, 'logged_current_physics_used': False,
            'separate_count_frequency_tail_at_root': float(tail[0, 0]),
            'start_defense_we': float(planning['we'].predict_defense(state, state.defender_is_home)),
            'terminal_values': terminal, 'conditional_temperature': float(model.temperature),
            'stability': stability, 'scenarios': scenarios, 'computations': computations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--pitcher', type=int)
    parser.add_argument('--max-actions', type=int, default=6)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with (args.run / 'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    with (args.run / 'planning_context.pkl').open('rb') as stream:
        planning = pickle.load(stream)
    rows = planning['rows']
    if args.pitcher is not None:
        rows = rows[rows.pitcher.eq(args.pitcher)]
    if not len(rows):
        raise SystemExit('No saved cases for this pitcher')
    model = SequenceModel.load(args.run / 'all_transformer.pt')
    cases = []
    for _, row in rows.iterrows():
        case = evaluate_case(model, encoders, planning, row, args.max_actions)
        cases.append(case)
        print(json.dumps({'pitcher': int(row.pitcher), 'stability': case['stability']}), flush=True)
    sources = [Path(__file__).resolve(), PROJECT / 'scripts/recommend_sequence.py',
               *(PROJECT / 'pitchmdp' / name for name in ['sequence_planner.py', 'sequence_model.py',
                 'sequence_data.py', 'sequence_delivery.py', 'game.py', 'planner.py', 'model.py', 'archetypes.py'])]
    suffix = f'_{args.pitcher}' if args.pitcher is not None else ''
    output = args.output or args.run / f'sequence_policy_sensitivity{suffix}.json'
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run': str(args.run.resolve()),
              'source_hashes': {str(p.relative_to(PROJECT)): sha256(p) for p in sources},
              'input_artifact_sha256': {name: sha256(args.run / name) for name in ['all_transformer.pt', 'encoders.pkl', 'planning_context.pkl']},
              'design': {'sigmas_ft': list(SIGMAS), 'depths': [1, 2], 'physics': ['mean', 'train_medoid'],
                         'max_actions': args.max_actions, 'scenario_count_per_case': 12,
                         'candidate_selection': 'Fixed before model scoring: TRAIN local support, first target per type then rounds',
                         'depth_one_computation': 'Exact root one-step Q returned by the same depth-two evaluation',
                         'medoid': 'Retained TRAIN-pool member minimizing summed squared standardized nonlocation distance; pool has frozen TRAIN imputations'},
              'limitations': [
                  'Model-internal sensitivity only; no observed, causal, or off-policy policy gain is estimated.',
                  'Targets/control noise are assumed, not identified from logged realized locations.',
                  'Both physics variants fix nonlocation channels across future counts and omit their execution variability.',
                  'A retained TRAIN medoid preserves one joint nonlocation vector after frozen imputation; it does not ensure originally complete measurements.',
                  'Leaves use a separate count/hand-frequency PA tail, not an evaluated sequence policy.',
                  'Root usage-weighted values randomize only the first action; later branches are optimized.',
                  'All saved cases begin at 0-0: explicit depth two cannot reach walk/strikeout and sees at most one simulated prior token.',
                  'training_local_n counts observed pitches within 0.65 ft of a target, not logged intended targets.',
                  'Conditional calibration temperature is fixed across target-control scenarios; calibration under these kernels is unverified.'
              ],
              'cases': cases}
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(output)
    for source in sources:
        destination = args.run / 'sensitivity_source' / source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    print(json.dumps({'output': str(output.resolve()), 'case_count': len(cases),
                      'cases_with_identical_top_action_in_all_scenarios': sum(c['stability']['all_scenarios_same_best_action'] for c in cases)}), flush=True)


if __name__ == '__main__':
    main()
