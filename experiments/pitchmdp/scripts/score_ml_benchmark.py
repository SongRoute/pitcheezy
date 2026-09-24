"""Complete-family scoring for the preregistered seven-architecture screen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np

from pitchmdp.data import hash_file
from pitchmdp.matrix_benchmark import CELLS, SEEDS, validate_config, member_identity
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, paired_game_comparison, holm_adjust, prediction_decision
from run_ml_benchmark import read_json, verify, identity, validate_native_runtime, member_dir, dump
from run_ml_matrix import assert_hashes, check_location
from score_ml_matrix import archive, assert_aligned, require_complete, summarize_cell, group_report


def comparison_family(config):
    control = config['registration']['control']
    candidates = config['registration']['primary_candidates']
    if control != 'A0-MLP' or candidates != ['A1-linear', 'A2-lightgbm', 'A3-lstm',
                                           'A4-gru', 'A5-melville', 'A6-transformer']:
        raise ValueError('Comparison family differs from preregistration')
    return control, candidates


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    control, candidates = comparison_family(config)
    require_complete(output, CELLS, SEEDS)
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Scoring output exists; preserve previous analysis')
    baseline = archive(output / 'baseline_predictions.npz')
    assert_aligned(baseline, baseline)
    started = time.perf_counter()
    inputs = {str(output / name): hash_file(output / name) for name in ('preparation.json', 'baseline_predictions.npz')}
    members, costs = {}, {}
    for cell in CELLS:
        members[cell], costs[cell] = [], []
        for seed in SEEDS:
            dest = member_dir(output, cell, seed)
            fit = read_json(dest / 'fit_state.json')
            if fit['identity'] != member_identity(prep, cell, seed):
                raise ValueError('Architecture member identity differs')
            assert_hashes(dest, fit['artifact_hashes'])
            pred = read_json(dest / 'prediction_state.json')
            if pred['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
                raise ValueError('Prediction references changed fit')
            assert_hashes(dest, pred['artifact_hashes'])
            path = dest / 'predictions.npz'
            member = archive(path)
            assert_aligned(member, baseline)
            members[cell].append(member)
            for name in ('fit_state.json', 'prediction_state.json', 'fit.json',
                         'prediction_runtime.json', 'predictions.npz'):
                inputs[str(dest / name)] = hash_file(dest / name)
            costs[cell].append({'seed': seed, 'fit': read_json(dest / 'fit.json'),
                               'prediction': read_json(dest / 'prediction_runtime.json')})
    reports, predictions = {}, {}
    for cell in CELLS:
        reports[cell], predictions[cell] = summarize_cell(members[cell], baseline)
    comparisons = []
    for candidate in candidates:
        paired = paired_game_comparison(baseline['dev_y'], predictions[candidate]['primary'],
                    predictions[control]['primary'], baseline['dev_game_pk'])
        deltas = [float((pitch_losses(baseline['dev_y'], a) - pitch_losses(baseline['dev_y'], b))[:, 0].mean())
                  for a, b in zip(predictions[candidate]['seed_primary'], predictions[control]['seed_primary'])]
        comparisons.append({'candidate': candidate, 'control': control, 'paired': paired, 'seed_deltas': deltas})
    for comparison, adjusted in zip(comparisons, holm_adjust([c['paired']['nll']['p_less'] for c in comparisons])):
        comparison['decision'] = prediction_decision(comparison['paired'], adjusted, comparison['seed_deltas'])
    cost_summary = {cell: {'total_load_fit_temperature_seconds': sum(m['fit']['seconds_total'] for m in row),
                           'total_prediction_seconds': sum(m['prediction']['seconds'] for m in row),
                           'maximum_peak_rss_bytes': max(max(m['fit']['peak_rss_bytes'], m['prediction']['peak_rss_bytes']) for m in row)}
                    for cell, row in costs.items()}
    eligible = [c['candidate'] for c in comparisons if c['decision']['status'] == 'predictive_improvement']
    ranking = sorted(eligible, key=lambda c: (reports[c]['primary']['log_loss'], cost_summary[c]['total_load_fit_temperature_seconds']))[:2]
    parent = read_json(output / 'parent_preparation.json')
    cohort = parent['scope']['regular']['cohort_ids']
    result = {'protocol': config['protocol'], 'config_sha256': canonical_hash(config),
        'n': len(baseline['dev_y']), 'games': len(np.unique(baseline['dev_game_pk'])),
        'frequency': prediction_metrics(baseline['dev_y'], baseline['dev']),
        'reports': reports, 'primary_comparisons': comparisons,
        'descriptive_gru_minus_lstm': paired_game_comparison(baseline['dev_y'], predictions['A4-gru']['primary'],
                                    predictions['A3-lstm']['primary'], baseline['dev_game_pk']),
        'followup_candidates': ranking, 'costs': costs, 'cost_summary': cost_summary,
        'per_pitcher': group_report(baseline, predictions, cohort),
        'class_reporting': {'minimum_events': 30, 'eligible': [int((baseline['dev_y'] == i).sum()) >= 30 for i in range(10)]},
        'policy_effect': None, 'whole_mlb_robustness': None, 'formal_cost_improvement': None,
        'limits': ['C6 retrospective eligible previously exposed DEV only',
                   'Fixed-prediction conditional game bootstrap excludes full training and selection uncertainty',
                   'Shared-task adaptations, not faithful original-paper reproductions',
                   'No independent final-set, reliever-wide, or causal recommendation evidence'],
        'input_hashes': inputs, 'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in
            (Path(__file__), PROJECT / 'scripts/score_ml_matrix.py', PROJECT / 'pitchmdp/matrix_metrics.py',
             PROJECT / 'scripts/run_sequence_calibration.py')},
        'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')},
        **{cell + '_' + name: value for cell, values in predictions.items() for name, value in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
         'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print(json.dumps({'nll': {c: r['primary']['log_loss'] for c, r in reports.items()}, 'followup_candidates': ranking}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    score(validate_config(read_json(args.config)), args.local_config, args.output.resolve())


if __name__ == '__main__':
    main()
