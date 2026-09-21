"""Freeze actual-history DEV examples, then perform bounded sequence/WE replay.

Selection uses chronological keys and pre-pitch state, beyond the experiment's
existing eligibility filter. Current/future physical features and outcomes are
removed from replay records before any recommendation is computed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from pitchmdp.archetypes import HISTORY_COLUMNS, add_batter_style_history
from pitchmdp.game import GameState, terminal_values
from pitchmdp.model import eligible
from pitchmdp.planner import solve_pa
from pitchmdp.recommend import supported_actions
from pitchmdp.sequence_data import KEY, PHYSICAL_COLUMNS, build_history_indices, prepare_frame
from pitchmdp.sequence_model import SequenceModel
from pitchmdp.sequence_planner import solve_lookahead
from recommend_sequence import project_illegal_double_play, select_actions, sequence_predictor, sha256
from run_sequence_policy_sensitivity import controlled_points


REPLAY_COLUMNS = ('game_pk', 'game_date', 'at_bat_number', 'pitch_number', 'pitcher', 'batter',
                  'balls', 'strikes', 'stand', 'p_throws', 'inning', 'inning_topbot', 'outs_when_up',
                  'bases', 'home_score', 'away_score', 'supported_pa', *HISTORY_COLUMNS)


def select_case_positions(frame, eligible_mask, pitcher_ids):
    """Return deterministic state-selected positions; never rank by model/Q."""
    mask = np.asarray(eligible_mask, dtype=bool)
    if mask.shape != (len(frame),):
        raise ValueError('Eligibility mask must align to the complete frame')
    chronological = frame.assign(_position=np.arange(len(frame))).sort_values(['game_date', *KEY])
    cohort = chronological[mask[chronological._position] & chronological.split.eq('dev') &
                           chronological.pitcher.isin(pitcher_ids) &
                           chronological.pitcher.eq(chronological.starter_pitcher)]
    selected, unavailable = [], []
    for pitcher in sorted(map(int, pitcher_ids)):
        pool = cohort[cohort.pitcher.eq(pitcher)]
        conditions = {
            'two_strikes': pool.strikes.eq(2) & pool.pitch_number.gt(1),
            'runners_and_outs_pa_start': pool.pitch_number.eq(1) & pool.bases.ne(0) & pool.outs_when_up.ge(1),
        }
        for kind, condition in conditions.items():
            matches = pool.loc[condition]
            if len(matches):
                selected.append((kind, int(matches.iloc[0]._position)))
            else:
                unavailable.append({'pitcher': pitcher, 'kind': kind})
    return selected, unavailable


def observed_history(frame, history_indices, position, normalizer):
    """Read only preceding rows in the same PA; exclude current/future physics."""
    index = np.asarray(history_indices[position], dtype=int)
    valid = index >= 0
    prior = index[valid]
    row = frame.iloc[position]
    if len(prior):
        previous = frame.iloc[prior]
        if (np.any(prior >= position) or not previous.game_pk.eq(row.game_pk).all() or
                not previous.at_bat_number.eq(row.at_bat_number).all() or
                not previous.pitch_number.lt(row.pitch_number).all() or
                not previous.pitch_number.is_monotonic_increasing):
            raise ValueError('History contains current/future pitches or crosses PA boundaries')
        tokens = normalizer.transform(previous)
        keys = previous[KEY].astype(np.int64).to_numpy().tolist()
    else:
        tokens = np.empty((0, 8), dtype=np.float32)
        keys = []
    if tokens.shape != (len(prior), 8) or not np.isfinite(tokens).all():
        raise ValueError('Observed normalized history must be finite with eight channels')
    return tokens, keys, valid.tolist()


def replay_row(row):
    """Whitelist pre-pitch context; action generation replaces the type sentinel."""
    clean = row.loc[list(REPLAY_COLUMNS)].copy()
    clean['pitch_type'] = 'UNOBSERVED_CURRENT_TYPE'
    forbidden = set(PHYSICAL_COLUMNS) | {'events', 'description', 'outcome', 'final_home_win'}
    if forbidden.intersection(clean.index):
        raise AssertionError('Current physical/outcome columns entered replay context')
    return clean


def prepare_cases(run, encoders, max_actions):
    local = json.loads((PROJECT / 'configs/local.json').read_text())
    cohort = json.loads((run / 'cohort_manifest.json').read_text())
    frame = prepare_frame(local)
    data_identity = frame.attrs['sequence_data_identity']
    add_batter_style_history(frame)
    mask = eligible(frame)
    selected, unavailable = select_case_positions(frame, mask, cohort['pitcher_ids'])
    indices = build_history_indices(frame)
    train_all = frame.loc[frame.split.eq('train') & mask]
    actions_by_key, cases, details = {}, [], []
    for kind, position in selected:
        original = frame.iloc[position]
        row = replay_row(original)
        history, history_keys, valid = observed_history(frame, indices, position, encoders['normalizer'])
        if kind == 'two_strikes' and not len(history):
            raise ValueError('Selected mid-PA example has no observed history')
        key = (int(row.pitcher), str(row.stand))
        if key not in actions_by_key:
            actions_by_key[key] = supported_actions(train_all, *key)
        available = actions_by_key[key]
        actions = select_actions(available, max_actions)
        if not actions:
            raise ValueError(f'No TRAIN-supported candidates for selected case {key}; do not silently select another case')
        context = encoders['context'].transform(pd.DataFrame([row]))[0]
        case_id = f'{int(row.pitcher)}_{kind}'
        cases.append({'case_id': case_id, 'kind': kind, 'row': row, 'history': history,
                      'history_keys': history_keys, 'history_valid_mask': valid,
                      'context': context, 'actions': actions, 'available_action_count': len(available)})
        details.append({'case_id': case_id, 'kind': kind, 'root_key': [int(row[k]) for k in KEY],
                        'context_row': row.to_dict(), 'context_vector': context.tolist(),
                        'observed_history_keys': history_keys, 'history_valid_mask': valid,
                        'observed_history_normalized': history.tolist(), 'selected_actions': actions,
                        'available_supported_action_count': len(available)})
    # Persist both before loading a neural checkpoint or computing any model Q.
    prepared = run / 'midpa_cases.pkl'
    temporary = prepared.with_suffix('.pkl.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(cases, stream)
    temporary.replace(prepared)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'selection': 'First chronological eligible cohort DEV row per pitcher with strikes=2,pitch_number>1; plus first pitch_number=1,bases!=0,outs>=1 when available',
                'eligibility_caveat': 'Existing eligible() requires supported PA, outcome label and observed current location availability; no additional outcome/winner or model-score selection',
                'history_contract': 'Same-PA strictly preceding pitches only; current/future raw physical measurements excluded from saved replay rows',
                'source_sha256': sha256(Path(__file__).resolve()), 'max_actions': max_actions,
                'encoders_sha256': sha256(run / 'encoders.pkl'),
                'cohort_manifest_sha256': sha256(run / 'cohort_manifest.json'),
                'prepared_pickle_sha256': sha256(prepared), 'verified_data_identity': data_identity,
                'unavailable_cases': unavailable, 'cases': details}
    target = run / 'midpa_selection.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(target)
    return cases, manifest


def load_prepared(run, max_actions):
    manifest = json.loads((run / 'midpa_selection.json').read_text())
    checks = {'source_sha256': sha256(Path(__file__).resolve()),
              'encoders_sha256': sha256(run / 'encoders.pkl'),
              'cohort_manifest_sha256': sha256(run / 'cohort_manifest.json'),
              'prepared_pickle_sha256': sha256(run / 'midpa_cases.pkl')}
    if any(manifest.get(k) != v for k, v in checks.items()) or manifest['max_actions'] != max_actions:
        raise ValueError('Prepared examples or their source/encoder/cohort changed; explicitly regenerate with --prepare-only')
    with (run / 'midpa_cases.pkl').open('rb') as stream:
        cases = pickle.load(stream)
    return cases, manifest


def evaluate_case(model, encoders, planning, case, physics='mean'):
    start = time.perf_counter()
    row, history, actions = case['row'], case['history'], case['actions']
    if not 0 <= int(row.balls) <= 3 or not 0 <= int(row.strikes) <= 2:
        raise ValueError('Invalid observed count')
    if int(row.pitch_number) > 1 and not len(history):
        raise ValueError('Later pitches require nonempty actual observed history')
    if not bool(row.supported_pa):
        raise ValueError('Unsupported PA')
    context = encoders['context'].transform(pd.DataFrame([row]))[0]
    if not np.array_equal(context, case['context']):
        raise ValueError('Frozen pre-pitch context changed between selection and evaluation')
    state = GameState.from_row(row)
    terminal = terminal_values(state, planning['we'], planning['advancement'])
    count_frame = pd.DataFrame([{**row.to_dict(), 'balls': b, 'strikes': s}
                               for b in range(4) for s in range(3)])
    base_p = project_illegal_double_play(planning['baseline'].predict(count_frame), row)
    tail = solve_pa(base_p.reshape(4, 3, 1, 1, 10), terminal, [0]).baseline_values[:, :, 0]
    points, weights, delivery_details = controlled_points(row, actions, encoders['delivery'], encoders['normalizer'], .3, physics)
    predict = sequence_predictor(model, encoders['context'], row)
    root_p = predict([history] * (len(actions) * 9),
                     np.tile([int(row.balls), int(row.strikes)], (len(actions) * 9, 1)), points.reshape(-1, 8))
    root_p = np.einsum('ako,k->ao', root_p.reshape(len(actions), 9, 10), weights)
    result = solve_lookahead(history, int(row.balls), int(row.strikes), points, predict, terminal,
                            lambda b, s, h: tail[b, s], weights=weights, max_depth=2)
    usage = np.array([a['training_pitch_type_n'] / sum(b['pitch_type'] == a['pitch_type'] for b in actions)
                      for a in actions], dtype=float)
    usage /= usage.sum()
    ranked = [{**actions[int(i)], 'action_index': int(i), 'bounded_model_internal_defense_we': float(result.q_values[i]),
               'one_pitch_then_count_tail_we': float(result.one_step_q_values[i]),
               'root_usage_weight': float(usage[i]),
               'root_strikeout_probability': float(root_p[i, 1]) if int(row.strikes) == 2 else 0.,
               'root_walk_probability': float(root_p[i, 0]) if int(row.balls) == 3 else 0.}
              for i in np.argsort(-result.q_values, kind='stable')]
    return {'case_id': case['case_id'], 'kind': case['kind'], 'physics': physics,
            'input': row.to_dict(), 'actual_history_length': len(history),
            'observed_history_keys': case['history_keys'], 'history_valid_mask': case['history_valid_mask'],
            'logged_current_physics_used': False, 'logged_future_physics_used': False,
            'bounded_model_internal_defense_we': result.value,
            'one_pitch_then_count_tail_best_we': float(result.one_step_q_values.max()),
            'separate_count_frequency_tail_at_root': result.continuation_reference,
            'start_defense_we': float(planning['we'].predict_defense(state, state.defender_is_home)),
            'root_usage_weighted_bounded_value': float(usage @ result.q_values),
            'bounded_best_action_index': result.policy, 'one_pitch_best_action_index': int(result.one_step_q_values.argmax()),
            'ranked_actions': ranked, 'terminal_values': terminal, 'delivery_details': delivery_details,
            'solver': result.diagnostics, 'seconds': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='Save selection/history/context before any neural inference; explicitly regenerate if already present')
    parser.add_argument('--max-actions', type=int, default=6)
    parser.add_argument('--physics', choices=['mean', 'train_medoid'], default='mean')
    args = parser.parse_args()
    with (args.run / 'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    if args.prepare_only or not (args.run / 'midpa_selection.json').exists():
        cases, manifest = prepare_cases(args.run, encoders, args.max_actions)
    else:
        cases, manifest = load_prepared(args.run, args.max_actions)
    print(json.dumps({'prepared_cases': len(cases), 'unavailable': manifest['unavailable_cases'],
                      'selection_manifest': str(args.run / 'midpa_selection.json')}), flush=True)
    if args.prepare_only:
        return
    with (args.run / 'planning_context.pkl').open('rb') as stream:
        planning = pickle.load(stream)
    model = SequenceModel.load(args.run / 'all_transformer.pt')
    results = []
    for case in cases:
        result = evaluate_case(model, encoders, planning, case, args.physics)
        results.append(result)
        print(json.dumps({'case_id': result['case_id'], 'history_length': result['actual_history_length'],
                          'count': [int(case['row'].balls), int(case['row'].strikes)],
                          'best_action': result['ranked_actions'][0]}), flush=True)
    sources = [Path(__file__).resolve(), PROJECT / 'scripts/recommend_sequence.py',
               PROJECT / 'scripts/run_sequence_policy_sensitivity.py',
               *(PROJECT / 'pitchmdp' / n for n in ['sequence_data.py', 'sequence_model.py', 'sequence_delivery.py',
                 'sequence_planner.py', 'game.py', 'planner.py', 'model.py', 'archetypes.py', 'recommend.py'])]
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run': str(args.run.resolve()),
              'source_hashes': {str(p.relative_to(PROJECT)): sha256(p) for p in sources},
              'input_artifact_sha256': {n: sha256(args.run / n) for n in ['all_transformer.pt', 'encoders.pkl', 'planning_context.pkl', 'midpa_selection.json', 'midpa_cases.pkl']},
              'conditional_temperature': float(model.temperature), 'selection': manifest['selection'],
              'limitations': ['Actual preceding history is replayed, then only hypothetical deliveries enter future history.',
                              'Depth-two bounded approximation with separate count/hand-frequency PA continuation; not full-history PA optimality.',
                              'Examples are selected before Q computation; they do not estimate population policy benefit.',
                              'Targets use assumed sigma0.3 independent Gaussian axes and nine-point quadrature; intended target/control is unidentified.',
                              'Nonlocation physical means or retained TRAIN medoids are held fixed across future counts.',
                              'Root usage weights randomize only the first action; later branches are optimized.',
                              'Conditional calibration under these counterfactual control kernels is unverified.',
                              'Game context stays fixed until terminal outcomes; only supported PAs are included.',
                              'WE values are model-internal estimates, not measured causal or observational off-policy improvements.'],
              'recommendations': results}
    output = args.run / f'sequence_midpa_{args.physics}.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(output)
    for source in sources:
        target = args.run / 'midpa_source' / source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(json.dumps({'output': str(output), 'cases': len(results)}), flush=True)


if __name__ == '__main__':
    main()
