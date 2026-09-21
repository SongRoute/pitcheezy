"""Replay saved first-PA cases with bounded physical-sequence/WE lookahead."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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


def select_actions(actions, max_actions=6):
    """Round-robin supported targets by type, ordered only by TRAIN support."""
    if max_actions < 1:
        raise ValueError('max_actions must be positive')
    by_type = {}
    for action in actions:
        by_type.setdefault(action['pitch_type'], []).append(dict(action))
    for group in by_type.values():
        group.sort(key=lambda a: (-a['training_local_n'], a['target_x_ft'], a['target_z_ft']))
    groups = sorted(by_type.values(), key=lambda g: (-g[0]['training_local_n'], g[0]['pitch_type']))
    selected = []
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank < len(group):
                selected.append(group[rank])
                if len(selected) == max_actions:
                    return selected
    return selected


def action_deliveries(row, actions, delivery, normalizer, sigma=.3):
    """Typical TRAIN physics with an explicitly assumed target location kernel.

    This planning demonstration holds nonlocation physics at its pool mean.
    It does not pretend the nine points are the trained joint-delivery model.
    The pool is selected at the initial count and held fixed within lookahead.
    """
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be finite and nonnegative')
    keys = list(dict.fromkeys(k for tier in delivery.TIERS for k in tier))
    records = [{**{k: row[k] for k in keys}, 'pitch_type': a['pitch_type']} for a in actions]
    pools, levels = delivery.sample(pd.DataFrame(records))
    typical = pools.mean(axis=1)
    offsets = np.sqrt(3) * np.array([-1., 0., 1.]) * sigma
    axis_weights = np.array([1/6, 2/3, 1/6])
    weights = np.outer(axis_weights, axis_weights).ravel()
    points = np.repeat(typical[:, None, :], 9, axis=1)
    for a, action in enumerate(actions):
        for k, (ix, iz) in enumerate(np.ndindex(3, 3)):
            points[a, k, 6] = (action['target_x_ft'] + offsets[ix] - normalizer.mean[6]) / normalizer.scale[6]
            points[a, k, 7] = (action['target_z_ft'] + offsets[iz] - normalizer.mean[7]) / normalizer.scale[7]
    return points, weights, levels


def project_illegal_double_play(probabilities, row):
    p = np.asarray(probabilities, dtype=float).copy()
    if int(row.outs_when_up) == 2 or int(row.bases) == 0:
        p[..., 3] += p[..., 9]
        p[..., 9] = 0
    return p


def sequence_predictor(model, context_encoder, row):
    fixed_context = context_encoder.transform(pd.DataFrame([row]))[0]

    def predict(histories, counts, current):
        n, nf = current.shape
        tokens = np.zeros((n, 6, nf), dtype=np.float32)
        valid = np.zeros((n, 6), dtype=bool)
        for i, history in enumerate(histories):
            previous = history[-5:]
            if len(previous):
                tokens[i, 5-len(previous):5] = previous
                valid[i, 5-len(previous):5] = True
        tokens[:, -1] = current
        valid[:, -1] = True
        context = np.repeat(fixed_context[None, :], n, axis=0)
        context[:, 0], context[:, 1] = counts[:, 0] / 3, counts[:, 1] / 2
        return project_illegal_double_play(model.predict((tokens, valid, context)), row)

    return predict


def recommend(model, encoders, planning, row, *, max_actions=6, max_depth=2, sigma=.3):
    start = time.perf_counter()
    if int(row.pitch_number) != 1 or int(row.balls) != 0 or int(row.strikes) != 0:
        raise ValueError('Saved replay requires an actual 0-0 first pitch; later pitches require observed histories')
    if 'supported_pa' in row and not bool(row.supported_pa):
        raise ValueError('Sequence planning only supports PAs without intervening game-state changes')
    available = planning['actions'][(int(row.pitcher), str(row.stand))]
    actions = select_actions(available, max_actions)
    if not actions:
        raise ValueError('No TRAIN-supported actions for the saved pitcher/hand case')
    state = GameState.from_row(row)
    terminal = terminal_values(state, planning['we'], planning['advancement'])
    count_frame = pd.DataFrame([{**row.to_dict(), 'balls': b, 'strikes': s}
                               for b in range(4) for s in range(3)])
    base_p = project_illegal_double_play(planning['baseline'].predict(count_frame), row)
    count_plan = solve_pa(base_p.reshape(4, 3, 1, 1, 10), terminal, [0])
    points, weights, levels = action_deliveries(row, actions, encoders['delivery'], encoders['normalizer'], sigma)
    predictor = sequence_predictor(model, encoders['context'], row)
    result = solve_lookahead(np.empty((0, points.shape[-1])), int(row.balls), int(row.strikes),
                            points, predictor, terminal,
                            lambda b, s, h: count_plan.baseline_values[b, s, 0],
                            weights=weights, max_depth=max_depth)
    # These weights randomize the root action only; later branches are optimized.
    usage = np.array([a['training_pitch_type_n'] / sum(b['pitch_type'] == a['pitch_type'] for b in actions)
                      for a in actions], dtype=float)
    usage /= usage.sum()
    ranking = []
    for index in np.argsort(-result.q_values, kind='stable'):
        index = int(index)
        ranking.append({**actions[index], 'action_index': index,
                        'bounded_defense_we': float(result.q_values[index]),
                        'one_pitch_then_count_tail_we': float(result.one_step_q_values[index]),
                        'root_usage_weight': float(usage[index]), 'delivery_pool_tier': int(levels[index])})
    one_step_best = int(np.argmax(result.one_step_q_values))
    return {
        'input': {'game_pk': int(row.game_pk), 'game_date': str(pd.Timestamp(row.game_date).date()),
                  'at_bat_number': int(row.at_bat_number), 'pitch_number': int(row.pitch_number),
                  'pitcher': int(row.pitcher), 'batter': int(row.batter), 'stand': str(row.stand),
                  'balls': int(row.balls), 'strikes': int(row.strikes), 'game_state': asdict(state)},
        'actual_history_length': 0, 'logged_current_physics_used': False,
        'candidate_count_before_support_selection': len(available), 'candidate_count': len(actions),
        'candidate_selection': 'TRAIN local support only; best target per type, then subsequent targets in rounds',
        'control_sigma_ft': sigma, 'start_defense_we': float(planning['we'].predict_defense(state, state.defender_is_home)),
        'bounded_defense_we': result.value,
        'separate_count_frequency_tail_at_root': result.continuation_reference,
        'root_usage_weighted_bounded_value': float(usage @ result.q_values),
        'root_usage_definition': 'TRAIN type usage, uniform selected targets within type; root only, optimized later branches',
        'one_pitch_then_count_tail_best_we': float(result.one_step_q_values.max()),
        'one_pitch_best_action_index': one_step_best,
        'bounded_best_action_index': result.policy,
        'bounded_first_action_equals_one_pitch': result.policy == one_step_best,
        'bounded_minus_one_pitch_pp': float(100 * (result.value - result.one_step_q_values.max())),
        'ranked_actions': ranking, 'terminal_values': terminal, 'solver': result.diagnostics,
        'conditional_temperature': float(model.temperature),
        'temperature_rationale': 'Conditional calibrated model for hypothetical physical tokens; observational delivery-mixture temperature is not used for the different target-control kernel.',
        'assumptions': [
            'Bounded physical-history lookahead, not an exact full-history optimal PA policy.',
            'Leaves use a separate TRAIN count/hand-frequency PA model; this is not a same-model sequence baseline.',
            'Nonlocation physical channels use the mean of a TRAIN joint pool chosen at root count, held fixed through lookahead.',
            'Target location uses independent Gaussian axes and 3x3 Gauss-Hermite quadrature; intended targets/control are unobserved.',
            'Only simulated deliveries update future history; game state is fixed until terminal events.',
            'Model-internal hypothetical values do not establish causal benefit or observational off-policy performance.',
        ],
        'seconds': time.perf_counter() - start,
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--pitcher', type=int)
    parser.add_argument('--max-actions', type=int, default=6)
    parser.add_argument('--max-depth', type=int, choices=(1, 2), default=2)
    parser.add_argument('--sigma', type=float, default=.3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with (args.run / 'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    with (args.run / 'planning_context.pkl').open('rb') as stream:
        planning = pickle.load(stream)
    model = SequenceModel.load(args.run / 'all_transformer.pt')
    rows = planning['rows']
    if args.pitcher is not None:
        rows = rows[rows.pitcher.eq(args.pitcher)]
    if not len(rows):
        raise SystemExit('No saved representative PA for this pitcher.')
    results = [recommend(model, encoders, planning, row, max_actions=args.max_actions,
                         max_depth=args.max_depth, sigma=args.sigma) for _, row in rows.iterrows()]
    sources = [Path(__file__).resolve(), *(PROJECT / 'pitchmdp' / name for name in
                ['sequence_planner.py', 'sequence_model.py', 'sequence_data.py', 'sequence_delivery.py', 'game.py', 'planner.py'])]
    source_hashes = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}
    artifact_hashes = {name: sha256(args.run / name)
                       for name in ['all_transformer.pt', 'encoders.pkl', 'planning_context.pkl']}
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run': str(args.run.resolve()),
              'source_hashes': source_hashes, 'input_artifact_sha256': artifact_hashes, 'recommendations': results}
    filename = 'sequence_recommendations' + (f'_{args.pitcher}' if args.pitcher is not None else '') + '.json'
    output = args.output or args.run / filename
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(output)
    snapshot = args.run / 'planning_source'
    for path in sources:
        destination = snapshot / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({'output': str(output.resolve()), 'cases': len(results),
                      'summaries': [{'pitcher': r['input']['pitcher'], 'bounded_defense_we': r['bounded_defense_we'],
                                     'best_action': r['ranked_actions'][0], 'seconds': r['seconds']} for r in results]}, indent=2))


if __name__ == '__main__':
    main()
