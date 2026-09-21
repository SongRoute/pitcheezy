"""Saved-artifact-only post-hoc diagnosis of conditional-score/policy-value disagreement.

No model inference, fitting, candidate tuning, selection, or deployment. Intervals
resample whole games and describe this reused evaluation set, not causal policy
effects or independent model/evaluator uncertainty.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-improvement-v2')
SIGMAS = ('0.3', '0.45', '0.65')
SEED = 20260921
REPLICATES = 2000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def cluster_stats(difference, game_ids, *, replicates=REPLICATES, seed=SEED):
    """Row-weighted mean with whole-game cluster resampling; no one-game pseudo-CI."""
    values = np.asarray(difference, dtype=float)
    games = np.asarray(game_ids)
    if not len(values) or values.ndim != 1 or games.shape != values.shape or not np.isfinite(values).all():
        raise ValueError('Finite paired differences and matching nonempty game IDs required')
    unique, inverse = np.unique(games, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    interval = None
    if len(unique) >= 2:
        draws = np.random.default_rng(seed).integers(0, len(unique), (replicates, len(unique)))
        bootstrap = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        interval = np.quantile(bootstrap, [.025, .975]).tolist()
    return {'difference': float(values.mean()), 'ci95': interval, 'rows': len(values), 'games': len(unique),
            'bootstrap_replicates': replicates if interval else 0, 'seed': seed,
            'cluster_caution': 'no_interval_one_game' if len(unique) == 1 else
                               ('few_games_descriptive_only' if len(unique) < 10 else 'posthoc_reused_evaluation'),
            'negative_fraction': float((values < 0).mean())}


def group_table(frame, columns, value_columns, global_rows=None):
    result = []
    for key, rows in frame.groupby(columns, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        label = {name: value.item() if isinstance(value, np.generic) else value for name, value in zip(columns, key)}
        item = {'group': label, 'rows': len(rows), 'games': int(rows.game_id.nunique()), 'contrasts': {}}
        for value_column in value_columns:
            item['contrasts'][value_column] = cluster_stats(rows[value_column], rows.game_id)
            if global_rows:
                item['contrasts'][value_column]['contribution_to_overall_mean'] = float(rows[value_column].sum()/global_rows)
        result.append(item)
    return result


def support_bin(count):
    return '0' if count == 0 else ('1_to_19' if count < 20 else '20_or_more')


def policy_frame(records, observed_rows):
    """Join hand from saved observed rows only; unknown hands remain explicit."""
    if len({(row['game_id'], row['pa_id']) for row in records}) != len(records):
        raise ValueError('Duplicate saved PA policy rows')
    hand_lookup = {}
    for key, rows in observed_rows.groupby(['game_pk', 'at_bat_number', 'pitcher'], sort=False):
        hands = rows.stand.dropna().unique()
        hand_lookup[tuple(map(int, key))] = str(hands[0]) if len(hands) == 1 else 'ambiguous'
    ready, unavailable = [], []
    for row in records:
        base = {key: row[key] for key in ('game_id','pa_id','pitcher','date','inning','outs','bases')}
        base['hand'] = hand_lookup.get((row['game_id'],row['pa_id'],row['pitcher']), 'unknown_not_in_saved_proxy_rows')
        base['root_count'] = '0-0_by_saved_selection_protocol'
        if row['status'] != 'ready':
            unavailable.append(base | {'reason': 'no_common_supported_action_grid_saved_reason_not_detailed'})
            continue
        if row['first_action_different'] and not row['policy_different']:
            raise ValueError('Changed root action contradicts identical full policy')
        baseline, adaptive = row['chosen']['baseline'], row['chosen']['adaptive']
        base.update(action_count=row['action_count'], baseline_type=baseline['type'], baseline_zone=baseline['zone'],
                    adaptive_type=adaptive['type'], adaptive_zone=adaptive['zone'],
                    baseline_judge_cell_count=baseline['judge_observed_cell_count'],
                    adaptive_judge_cell_count=adaptive['judge_observed_cell_count'],
                    minimum_selected_judge_support=support_bin(min(baseline['judge_observed_cell_count'], adaptive['judge_observed_cell_count'])),
                    first_action_different=bool(row['first_action_different']), policy_different=bool(row['policy_different']),
                    change_pattern='root_changed' if row['first_action_different'] else
                        ('same_root_changed_later' if row['policy_different'] else 'identical_policy'))
        for sigma in SIGMAS:
            evaluation = row['evaluated'][sigma]
            if not all(np.isfinite(value) and 0 <= value <= 1 for value in evaluation.values()):
                raise ValueError('Invalid saved model-internal WE')
            base['we_delta_pp_'+sigma] = 100*(evaluation['adaptive']-evaluation['baseline'])
            if not row['policy_different'] and abs(base['we_delta_pp_'+sigma]) > 1e-10:
                raise ValueError('Identical policies must have identical values under a common evaluator')
        ready.append(base)
    return pd.DataFrame(ready), unavailable


def conditional_frame(proxy, adaptive, rows):
    required = ['game_pk','at_bat_number','pitch_number','pitcher','pitch_type','stand','balls','strikes','outs_when_up','bases']
    if not set(required).issubset(rows) or rows.duplicated(required[:3]).any():
        raise ValueError('Saved observed rows require unique pitch keys and descriptive state')
    for archive in (proxy, adaptive):
        np.testing.assert_array_equal(archive['game_ids'], rows.game_pk.to_numpy())
    for key in ('at_bat_number', 'pitch_number'):
        np.testing.assert_array_equal(proxy[key], rows[key].to_numpy())
    np.testing.assert_array_equal(proxy['y'], adaptive['y'])
    narrow, broad = proxy['sigma030_supported'].astype(bool), proxy['sigma045_supported'].astype(bool)
    expected = np.where(narrow[:, None], proxy['sigma030'], proxy['sigma045'])
    np.testing.assert_allclose(adaptive['p'], expected, rtol=0., atol=1e-14)
    np.testing.assert_array_equal(adaptive['supported'], narrow | broad)
    y = np.asarray(proxy['y'], int)
    baseline, candidate = proxy['sigma045'], adaptive['p']
    for prediction in (baseline, candidate):
        if prediction.shape != (len(rows), 10) or not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('Invalid saved conditional probabilities')
        np.testing.assert_allclose(prediction.sum(axis=1), 1., atol=1e-6)
    onehot = np.eye(10)[y]
    frame = rows.rename(columns={'game_pk':'game_id','outs_when_up':'outs','stand':'hand'}).copy()
    frame['log_loss_delta'] = -np.log(np.clip(candidate[np.arange(len(y)), y], 1e-15, 1))+np.log(np.clip(baseline[np.arange(len(y)), y], 1e-15, 1))
    frame['brier_delta'] = np.square(candidate-onehot).sum(axis=1)-np.square(baseline-onehot).sum(axis=1)
    frame['fallback'] = np.where(narrow, 'narrow_kernel', np.where(broad, 'broad_kernel_fallback', 'type_only_fallback'))
    frame['count'] = frame.balls.astype(str)+'-'+frame.strikes.astype(str)
    if np.any(np.abs(frame.loc[~narrow, 'log_loss_delta']) > 1e-12):
        raise ValueError('Non-narrow adaptive predictions must equal broad baseline')
    return frame


def diagnose(source, proxies, output):
    source, proxies, output = Path(source).resolve(), Path(proxies).resolve(), Path(output).resolve()
    if output.exists() or output in (source, proxies) or source.is_relative_to(output) or proxies.is_relative_to(output):
        raise ValueError('Use a new output directory distinct from all prior artifacts')
    inputs = [source/name for name in ('policy_rows.json','policy_summary.json','conditional_validation.json','spec.json','source_hashes.json')]
    inputs += [proxies/name for name in ('spec.json','data_identity.json','evaluator_validation.json','proxy_validation.json','source_hashes.json')]
    for partition in ('diagnosis','guard','evaluation'):
        inputs.extend([source/f'{partition}_adaptive_predictions.npz', proxies/f'{partition}_proxy_predictions.npz',
                       proxies/f'{partition}_proxy_rows.parquet'])
    hashes = {str(path): sha256(path) for path in inputs}
    output.mkdir(parents=True, exist_ok=False)
    config = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'post_hoc': True,
              'source': str(source), 'conditional_source': str(proxies), 'input_hashes': hashes,
              'script_sha256': sha256(Path(__file__)), 'bootstrap_replicates': REPLICATES, 'seed': SEED,
              'no_inference_refit_tuning_selection_or_deployment': True,
              'uncertainty': 'Whole-game paired bootstrap; reused evaluation and unadjusted descriptive subgroups. No causal or evaluator-model uncertainty.'}
    write(output/'config.json', config)
    records = json.loads((source/'policy_rows.json').read_text())
    saved_summary = json.loads((source/'policy_summary.json').read_text())
    evaluation_rows = pd.read_parquet(proxies/'evaluation_proxy_rows.parquet')
    policy, unavailable = policy_frame(records, evaluation_rows)
    if policy.empty:
        raise ValueError('No ready saved policy rows')
    policy_columns = ['we_delta_pp_'+sigma for sigma in SIGMAS]
    overall = {column: cluster_stats(policy[column], policy.game_id) for column in policy_columns}
    for sigma in SIGMAS:
        np.testing.assert_allclose(overall['we_delta_pp_'+sigma]['difference'],
                                   saved_summary['paired_adaptive_minus_baseline_pp'][sigma]['difference'], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(overall['we_delta_pp_'+sigma]['ci95'],
                                   saved_summary['paired_adaptive_minus_baseline_pp'][sigma]['ci95'], atol=1e-12, rtol=0.)
    policy_strata = {'pitcher': ['pitcher'], 'hand': ['hand'], 'pitcher_hand': ['pitcher','hand'],
                    'base_out': ['outs','bases'], 'change_pattern': ['change_pattern'],
                    'judge_support': ['minimum_selected_judge_support'],
                    'adaptive_type': ['adaptive_type'], 'adaptive_zone': ['adaptive_zone'],
                    'adaptive_type_zone': ['adaptive_type','adaptive_zone'], 'baseline_zone': ['baseline_zone']}
    policy_tables = {name: group_table(policy, columns, policy_columns, len(policy)) for name, columns in policy_strata.items()}
    changes = policy.loc[policy.first_action_different]
    policy_tables['changed_root_transition'] = group_table(changes,
        ['baseline_type','baseline_zone','adaptive_type','adaptive_zone'], policy_columns, len(policy)) if len(changes) else []
    signs = np.sign(policy[policy_columns].to_numpy())
    same_root = policy.loc[policy.change_pattern.eq('same_root_changed_later')]
    availability = {'requested_pas': len(records), 'ready_pas': len(policy), 'unavailable_pas': len(unavailable),
                    'ready_fraction': len(policy)/len(records), 'missing_hand_ready_pas': int(policy.hand.eq('unknown_not_in_saved_proxy_rows').sum()),
                    'unavailable_rows': unavailable,
                    'kernel_fallback_at_selected_policy_actions': None,
                    'kernel_fallback_missing_reason': 'ESS/mass/narrow fallback masks were not saved by the original policy runner'}
    conditional = {}
    for partition in ('diagnosis','guard','evaluation'):
        rows = pd.read_parquet(proxies/f'{partition}_proxy_rows.parquet')
        with np.load(proxies/f'{partition}_proxy_predictions.npz') as proxy, np.load(source/f'{partition}_adaptive_predictions.npz') as adaptive:
            observed = conditional_frame(proxy, adaptive, rows)
        value_columns = ['log_loss_delta','brier_delta']
        conditional[partition] = {'overall': {column: cluster_stats(observed[column], observed.game_id) for column in value_columns},
            'groups': {name: group_table(observed, columns, value_columns, len(observed)) for name, columns in {
                'pitcher':['pitcher'], 'hand':['hand'], 'base_out':['outs','bases'], 'count':['count'],
                'observed_type':['pitch_type'], 'fallback':['fallback']}.items()},
            'scope': 'Actual observed type/location prediction; these are not candidate-action policy values'}
        observed.to_parquet(output/f'{partition}_conditional_differences.parquet', index=False)
    policy.to_parquet(output/'policy_differences.parquet', index=False)
    missing = {
        'policy_root_count': 'All roots are0-0 by original saved selection code; count-stratified policy loss unavailable.',
        'continuation_policy_counts': 'All12count action indices and Q/probability arrays were not saved.',
        'kernel_support_fallback': 'Only common-action count and root July judge cell counts were saved; no ESS/mass masks.',
        'evaluator_disagreement': 'Three values share one independent judge and vary assumed execution sigma; own-model/judge probability disagreement cannot be recovered.',
        'minimum_rerun_specification': 'Replay only the same454 saved ready PA keys with frozen models/evaluator. Save all12count policies/Q, selected-action own/judge probabilities, action labels, ESS/mass and fallback masks. No fitting, new candidates, outcome selection, or deployment.'}
    result = {'post_hoc': True, 'primary_sigma': '0.45', 'policy_overall': overall, 'availability': availability,
        'policy_groups': policy_tables, 'conditional': conditional,
        'execution_assumption_sensitivity': {'same_judge_different_execution_sigmas': True,
            'negative_at_all_three_fraction': float((signs < 0).all(axis=1).mean()),
            'sign_disagreement_fraction': float((np.ptp(signs, axis=1) > 0).mean()),
            'not_independent_evaluators': True},
        'continuation_evidence': {'same_root_changed_later_pas': len(same_root),
            'primary_contrast': cluster_stats(same_root['we_delta_pp_0.45'], same_root.game_id) if len(same_root) else None,
            'interpretation': 'When root action is identical but full policy value changes under the same judge, later-count policy choices account for that value difference; no individual count attribution is available.'},
        'missing_fields': missing,
        'interpretation_limits': ['Conditional log loss averages recorded actual locations and outcomes; policy WE evaluates optimized hypothetical actions and continuation under another model.',
            'Improved observed-location scores do not by themselves validate action rankings or intended-target effects.',
            'Selected-zone subgroups are post-policy descriptive groups, not randomized location interventions.',
            'All subgroups and intervals reuse inspected data; no multiplicity correction, causal claim, candidate tuning, or deployment decision.']}
    write(output/'diagnosis.json', result)
    lines = ['# Adaptive location: saved-artifact post-hoc diagnosis', '',
             'No inference, refitting, tuning, new candidate, or deployment. Whole-game intervals reuse the inspected evaluation set.', '',
             '| Reference execution sigma | Adaptive minus baseline WE (pp) | 95% game interval |', '|---|---:|---|']
    for sigma in SIGMAS:
        stats = overall['we_delta_pp_'+sigma]
        lines.append(f"| {sigma} | {stats['difference']:.6f} | {stats['ci95']} |")
    lines += ['', f"Ready PAs: {len(policy)}/{len(records)}. Missing saved hand join: {availability['missing_hand_ready_pas']}.",
              f"Same root action but changed later policy: {len(same_root)} PAs; see continuation_evidence and change_pattern groups.",
              '', 'All policy roots are0-0. Count-specific tables describe conditional observed-pitch scores only.',
              'The three policy references vary execution sigma under the same judge; they are not independent evaluator models.',
              'Kernel support/fallback by chosen policy action and own-model/judge probability disagreements were not archived.',
              'Full stratified statistics, subgroup cautions, missing-field inventory and minimal replay specification: diagnosis.json.']
    for name in ('pitcher','hand','change_pattern','judge_support','adaptive_zone'):
        lines += ['', '## Descriptive primary-policy groups: '+name, '',
                  '| Group | PAs / games | WE difference (pp) | Game interval | Contribution to overall mean |',
                  '|---|---:|---:|---|---:|']
        for item in policy_tables[name]:
            stats = item['contrasts']['we_delta_pp_0.45']
            label = ', '.join(f'{key}={value}' for key, value in item['group'].items())
            interval = 'unavailable (one game)' if stats['ci95'] is None else f"[{stats['ci95'][0]:.6f}, {stats['ci95'][1]:.6f}]"
            lines.append(f"| {label} | {item['rows']} / {item['games']} | {stats['difference']:.6f} | {interval} | {stats['contribution_to_overall_mean']:.6f} |")
        lines.append('Unadjusted post-hoc subgroups; fewer than10games have especially unstable cluster intervals.')
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    if hashes != {str(path): sha256(path) for path in inputs} or config['script_sha256'] != sha256(Path(__file__)):
        raise RuntimeError('Diagnostic source/input artifacts changed during analysis')
    write(output/'runtime.json', {'complete': True, 'integrity_ok': True, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                                 'diagnosis_sha256': sha256(output/'diagnosis.json')})
    return {'output': str(output), 'ready_pas': len(policy), 'primary_delta_pp': overall['we_delta_pp_0.45']['difference'],
            'same_root_changed_later_pas': len(same_root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'adaptive-location-v1')
    parser.add_argument('--proxies', type=Path, default=ROOT/'location-evaluation-v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT/datetime.now(timezone.utc).strftime('adaptive-location-diagnosis-%Y%m%dT%H%M%SZ')).resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not output.is_relative_to(ROOT):
        raise ValueError('New output on approved mounted v2 SSD required')
    print(json.dumps(diagnose(args.source, args.proxies, output)))


if __name__ == '__main__':
    main()
