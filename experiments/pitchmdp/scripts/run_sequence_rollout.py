"""Same-sequence fixed-policy continuation with independent root selection/evaluation."""
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

from pitchmdp.game import GameState, terminal_values
from pitchmdp.planner import OUTCOMES, TERMINALS
from pitchmdp.sequence_model import SequenceModel
from recommend_sequence import sequence_predictor, sha256
from run_sequence_midpa import load_prepared
from run_sequence_policy_sensitivity import controlled_points

PROTOCOL = PROJECT / 'docs/SEQUENCE_ROLLOUT_PROTOCOL.md'


def common_uniforms(n_rollouts, max_pitches, seed):
    if n_rollouts < 2 or max_pitches < 1:
        raise ValueError('At least two rollouts and one pitch are required')
    return np.random.default_rng(seed).random((max_pitches, n_rollouts, 3))


def categorical(probabilities, uniforms):
    p = np.asarray(probabilities, dtype=float)
    cdf = np.cumsum(p / p.sum(axis=-1, keepdims=True), axis=-1)
    cdf[..., -1] = 1.
    return np.sum(np.asarray(uniforms)[..., None] >= cdf, axis=-1)


def simulate(history, balls, strikes, actions, delivery_weights, policy_weights,
             predictor, terminal_utilities, uniforms):
    """Vectorized forced-root trajectories sharing indexed randomness by step.

    Returns zero-imputed lower returns and censor flags, never a tail estimate.
    Each root action has the same trajectory indices. A terminated arm does not
    consume or shift another arm's randomness. All later actions use one fixed
    policy, independent of the simulated state; outcome probabilities still use
    the actual simulated counts/history.
    """
    action = np.asarray(actions, dtype=float)
    h = np.asarray(history, dtype=float)
    u = np.asarray(uniforms, dtype=float)
    if (action.ndim != 3 or not all(action.shape) or not np.isfinite(action).all() or
            h.ndim != 2 or h.shape[1] != action.shape[-1] or not np.isfinite(h).all()):
        raise ValueError('Finite action[A,K,F] and history[H,F] arrays are required')
    if (not isinstance(balls, (int, np.integer)) or not 0 <= balls <= 3 or
            not isinstance(strikes, (int, np.integer)) or not 0 <= strikes <= 2):
        raise ValueError('Counts must be legal integers')
    if u.ndim != 3 or u.shape[2] != 3 or not u.shape[0] or u.shape[1] < 2 or not np.isfinite(u).all() or ((u < 0) | (u >= 1)).any():
        raise ValueError('Uniforms must have shape (positive max_pitches, at least 2 trajectories, 3), in [0,1)')
    na, nk, nf = action.shape
    max_pitches, n, _ = u.shape
    w = np.asarray(delivery_weights, dtype=float)
    if w.shape == (nk,):
        w = np.broadcast_to(w, (na, nk))
    policy = np.asarray(policy_weights, dtype=float)
    for value, shape in ((w, (na, nk)), (policy, (na,))):
        if value.shape != shape or not np.isfinite(value).all() or (value < 0).any() or not np.allclose(value.sum(axis=-1), 1, atol=1e-8, rtol=0):
            raise ValueError('Delivery and policy weights must be valid normalized probabilities')
    utility = np.array([terminal_utilities[t] for t in TERMINALS], dtype=float)
    if not np.isfinite(utility).all() or ((utility < 0) | (utility > 1)).any():
        raise ValueError('Terminal defensive WE must lie in [0,1] for censor bounds')
    h = h[-5:]
    histories = np.zeros((na, n, 5, nf), dtype=float)
    histories[:, :, :len(h)] = h
    lengths = np.full((na, n), len(h), dtype=np.int8)
    b = np.full((na, n), balls, dtype=np.int8)
    s = np.full((na, n), strikes, dtype=np.int8)
    alive = np.ones((na, n), dtype=bool)
    returns = np.zeros((na, n), dtype=float)
    event = np.full((na, n), -1, dtype=np.int8)
    pitches = np.zeros((na, n), dtype=np.int32)
    prediction_rows, prediction_calls, mass_error = 0, 0, 0.
    for step in range(max_pitches):
        arm, trajectory = np.nonzero(alive)
        if not len(arm):
            break
        chosen = arm if step == 0 else categorical(policy, u[step, trajectory, 0])
        delivery = categorical(w[chosen], u[step, trajectory, 1])
        current = action[chosen, delivery]
        past = [histories[a, t, :lengths[a, t]] for a, t in zip(arm, trajectory)]
        p = np.asarray(predictor(past, np.column_stack((b[arm, trajectory], s[arm, trajectory])), current), dtype=float)
        if p.shape != (len(arm), len(OUTCOMES)) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('Predictor must return finite nonnegative probabilities [N,10]')
        error = float(np.max(np.abs(p.sum(axis=-1) - 1)))
        if error > 1e-6:
            raise ValueError('Predictor probabilities must sum to one')
        mass_error = max(mass_error, error)
        prediction_calls += 1
        prediction_rows += len(arm)
        outcome = categorical(p, u[step, trajectory, 2])
        old_b, old_s = b[arm, trajectory], s[arm, trajectory]
        is_walk = (outcome == 0) & (old_b == 3)
        is_k = (outcome == 1) & (old_s == 2)
        terminal = (outcome >= 3) | is_walk | is_k
        terminal_index = np.where(is_walk, 0, np.where(is_k, 1, outcome - 1))
        ta, tt = arm[terminal], trajectory[terminal]
        returns[ta, tt] = utility[terminal_index[terminal]]
        event[ta, tt] = terminal_index[terminal]
        alive[ta, tt] = False
        pitches[arm, trajectory] = step + 1
        continuation = ~terminal
        ca, ct = arm[continuation], trajectory[continuation]
        result = outcome[continuation]
        b[ca, ct] += result == 0
        s[ca, ct] = np.minimum(2, s[ca, ct] + np.isin(result, [1, 2]))
        delivered = current[continuation]
        old_length = lengths[ca, ct]
        full = old_length == 5
        fa, ft = ca[full], ct[full]
        histories[fa, ft, :-1] = histories[fa, ft, 1:]
        histories[ca, ct, np.minimum(old_length, 4)] = delivered
        lengths[ca, ct] = np.minimum(old_length + 1, 5)
    return {'lower_returns': returns, 'censored': alive, 'terminal_event_index': event,
            'pitches': pitches, 'uniforms': u,
            'diagnostics': {'prediction_rows': prediction_rows, 'prediction_calls': prediction_calls,
                            'max_input_probability_mass_error': mass_error,
                            'max_pitches': max_pitches, 'rollouts_per_root_action': n}}


def mean_se_interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite paired trajectory values are required')
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {'mean': mean, 'mc_standard_error': se, 'mc95_normal': [mean - 1.96 * se, mean + 1.96 * se]}


def summarize(selection, evaluation, weights):
    """Select on one stream; estimate selected-vs-mixture contrast on the other."""
    weights = np.asarray(weights, dtype=float)
    score = selection['lower_returns'].mean(axis=1)
    selected = int(np.argmax(score))
    low = evaluation['lower_returns']
    high = low + evaluation['censored']
    baseline_low, baseline_high = weights @ low, weights @ high
    paired_low_imputed = low[selected] - baseline_low
    # The selected arm occurs in the baseline too, so its common unknown
    # censored utility cancels with weight w_selected before bounding.
    coefficients = -weights.copy()
    coefficients[selected] += 1
    positive, negative = np.maximum(coefficients, 0), np.minimum(coefficients, 0)
    lower_stats = mean_se_interval(positive @ low + negative @ high)
    upper_stats = mean_se_interval(positive @ high + negative @ low)
    point = mean_se_interval(paired_low_imputed)
    return {'selected_action_index': selected,
            'selection_definition': 'argmax independent selection-stream zero-imputed lower-return mean; fixed-order tie break',
            'selection_lower_means': score.tolist(),
            'selection_upper_means': (selection['lower_returns'] + selection['censored']).mean(axis=1).tolist(),
            'selection_censor_fraction': selection['censored'].mean(axis=1).tolist(),
            'evaluation_action_lower_means': low.mean(axis=1).tolist(),
            'evaluation_action_upper_means': high.mean(axis=1).tolist(),
            'evaluation_action_censor_fraction': evaluation['censored'].mean(axis=1).tolist(),
            'selected_value_bounds': [float(low[selected].mean()), float(high[selected].mean())],
            'baseline_value_bounds': [float(baseline_low.mean()), float(baseline_high.mean())],
            'paired_zero_imputed_contrast': point,
            'paired_censor_contrast_bounds': [lower_stats['mean'], upper_stats['mean']],
            'paired_censor_expanded_mc95_normal': [lower_stats['mean'] - 1.96 * lower_stats['mc_standard_error'],
                                                  upper_stats['mean'] + 1.96 * upper_stats['mc_standard_error']],
            'censor_bound_definition': 'Apply [0,1] censor bounds after canceling the selected arm shared with baseline: coefficient (1-w_selected) for selected and -w_a for other arms',
            'mc_uncertainty_scope': 'Simulation noise conditional on a fixed selected action, case, fitted model and control kernel; not model/data/game-sampling/causal uncertainty',
            'baseline_definition': 'Fixed TRAIN type-usage action mixture at root and every continuation pitch; root mixture enumerated with common random numbers',
            'continuation_definition': 'Same physical-sequence model to PA terminal under fixed policy; no separate count tail',
            'full_pa_optimal_policy': False}


def policy_weights(actions):
    weights = np.array([a['training_pitch_type_n'] / sum(b['pitch_type'] == a['pitch_type'] for b in actions)
                        for a in actions], dtype=float)
    return weights / weights.sum()


def main():
    overall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--selection-rollouts', type=int, default=1024)
    parser.add_argument('--evaluation-rollouts', type=int, default=4096)
    parser.add_argument('--max-pitches', type=int, default=32)
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.selection_rollouts < 2 or args.evaluation_rollouts < 2 or args.max_pitches < 1 or (args.case_limit is not None and args.case_limit < 1):
        parser.error('Positive max-pitches/case-limit and at least two rollouts per stream required')
    local = json.loads((PROJECT / 'configs/local.json').read_text())
    artifact_root = Path(local['artifact_root']).resolve()
    run = args.run.resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not run.is_relative_to(artifact_root):
        raise SystemExit('Use a run under the mounted configured SSD artifact root')
    # A pinned preparation is mandatory; never silently choose or regenerate cases.
    manifest = json.loads((run / 'midpa_selection.json').read_text())
    cases, _ = load_prepared(run, manifest['max_actions'])
    if args.case_limit is not None:
        cases = cases[:args.case_limit]
    destination = args.output_dir or run / datetime.now(timezone.utc).strftime('sequence-rollout-%Y%m%dT%H%M%SZ')
    if not destination.resolve().is_relative_to(artifact_root):
        raise SystemExit('Rollout artifacts must remain under configured SSD artifact root')
    destination.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), PROJECT / 'scripts/run_sequence_midpa.py',
               PROJECT / 'scripts/recommend_sequence.py', PROJECT / 'scripts/run_sequence_policy_sensitivity.py',
               *(PROJECT / 'pitchmdp' / n for n in ['sequence_planner.py', 'sequence_model.py', 'sequence_data.py',
                 'sequence_delivery.py', 'game.py', 'planner.py', 'archetypes.py', 'model.py', 'recommend.py'])]
    config = {'run': str(run), 'selection_rollouts_per_action': args.selection_rollouts,
              'evaluation_rollouts_per_action': args.evaluation_rollouts, 'selection_seed': 142,
              'evaluation_seed': 242, 'max_pitches': args.max_pitches, 'case_limit': args.case_limit,
              'case_ids': [c['case_id'] for c in cases], 'physics': 'mean', 'sigma_ft': .3,
              'protocol_sha256': sha256(PROTOCOL),
              'source_hashes': {str(p.relative_to(PROJECT)): sha256(p) for p in sources},
              'input_artifact_sha256': {n: sha256(run / n) for n in ['all_transformer.pt', 'encoders.pkl', 'planning_context.pkl', 'midpa_cases.pkl', 'midpa_selection.json']}}
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    shutil.copyfile(PROTOCOL, destination / PROTOCOL.name)
    for source in sources:
        target = destination / 'source' / source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Configuration, protocol and source snapshots exist before neural inference.
    with (run / 'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    with (run / 'planning_context.pkl').open('rb') as stream:
        planning = pickle.load(stream)
    model = SequenceModel.load(run / 'all_transformer.pt')
    if model.seed != 42 or model.kind != 'transformer' or model.n_classes != 10:
        raise ValueError('This frozen protocol requires the original seed-42 ten-outcome Transformer')
    output = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
              'terminal_event_order': list(TERMINALS), 'conditional_temperature': float(model.temperature),
              'cases': [], 'limitations': [
                  'Single-root policy improvement under a fixed same-sequence continuation; no fully optimal PA policy.',
                  'Root action is selected using independent simulation noise; model/context/control misspecification remains.',
                  'Model-internal contrasts do not establish actual, causal or off-policy policy benefit.',
                  'Censored trajectories retain [0,1] unknown WE bounds; no separate continuation model is inserted.',
                  'Game-state changes within supported PAs are excluded; predetermined terminal WE averages runner advancement.',
                  'Target-control kernel is assumed; nonlocation TRAIN-pool means stay fixed through the PA, including their averaged spin channels.',
                  'Same seed design is reused across cases; Monte Carlo case intervals must not be treated as independent sample evidence.'
              ]}
    print('ROLLOUT_DIR=' + str(destination), flush=True)
    for case in cases:
        start = time.perf_counter()
        row, actions = case['row'], case['actions']
        if not np.array_equal(encoders['context'].transform(pd.DataFrame([row]))[0], case['context']):
            raise ValueError('Frozen pre-pitch context changed after case preparation')
        state = GameState.from_row(row)
        terminal = terminal_values(state, planning['we'], planning['advancement'])
        points, delivery_weights, details = controlled_points(row, actions, encoders['delivery'], encoders['normalizer'], .3, 'mean')
        predictor = sequence_predictor(model, encoders['context'], row)
        weights = policy_weights(actions)
        streams = {}
        for name, n, seed in [('selection', args.selection_rollouts, 142), ('evaluation', args.evaluation_rollouts, 242)]:
            streams[name] = simulate(case['history'], int(row.balls), int(row.strikes), points,
                                     delivery_weights, weights, predictor, terminal,
                                     common_uniforms(n, args.max_pitches, seed))
            archive = destination / f'{case["case_id"]}_{name}.npz'
            np.savez_compressed(archive, **{k: v for k, v in streams[name].items() if isinstance(v, np.ndarray)})
        result = summarize(streams['selection'], streams['evaluation'], weights)
        result.update({'case_id': case['case_id'], 'kind': case['kind'], 'input': row.to_dict(),
                       'actual_history_length': len(case['history']), 'observed_history_keys': case['history_keys'],
                       'actions': actions, 'policy_weights': weights.tolist(), 'terminal_utilities': terminal,
                       'delivery_details': details, 'seconds': time.perf_counter() - start,
                       'streams': {name: {**item['diagnostics'], 'archive': f'{case["case_id"]}_{name}.npz',
                                          'archive_sha256': sha256(destination / f'{case["case_id"]}_{name}.npz')}
                                   for name, item in streams.items()}})
        output['cases'].append(result)
        temporary = destination / 'results.json.tmp'
        temporary.write_text(json.dumps(output, indent=2, default=str) + '\n')
        temporary.replace(destination / 'results.json')
        print(json.dumps({'case_id': case['case_id'], 'selected_action_index': result['selected_action_index'],
                          'paired_contrast': result['paired_zero_imputed_contrast'],
                          'evaluation_censor_fraction': result['evaluation_action_censor_fraction'],
                          'seconds': result['seconds']}), flush=True)
    source_hashes_end = {str(p.relative_to(PROJECT)): sha256(p) for p in sources}
    inputs_end = {name: sha256(run / name) for name in config['input_artifact_sha256']}
    protocol_end = sha256(PROTOCOL)
    integrity_ok = (source_hashes_end == config['source_hashes'] and
                    inputs_end == config['input_artifact_sha256'] and protocol_end == config['protocol_sha256'])
    runtime = {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'seconds': time.perf_counter() - overall_start, 'device': model.device,
               'cases_completed': len(output['cases']), 'integrity_ok': integrity_ok,
               'source_hashes_end': source_hashes_end, 'input_artifact_sha256_end': inputs_end,
               'protocol_sha256_end': protocol_end}
    (destination / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    if not integrity_ok:
        raise SystemExit('Source, input artifact or protocol changed during execution; inspect runtime.json')
    print('COMPLETE ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
