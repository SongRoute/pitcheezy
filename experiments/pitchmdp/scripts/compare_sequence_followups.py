"""CPU-only new-seed sensitivity and stronger-baseline comparisons.

All fitted artifacts stay immutable. Both seed sets, both legality regimes and
all predeclared frequency baselines are retained; none is selected using DEV.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from audit_sequence_robustness import independent_bootstrap, scores, sha
from diagnose_sequence_legality import condition_on_legality


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness', required=True, type=Path)
    parser.add_argument('--frequency', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    robustness, frequency = args.robustness.resolve(), args.frequency.resolve()
    config = read(robustness/'config.json')
    base = Path(config['base_run'])
    root = base.parent.parent
    assert all(path.is_relative_to(root) for path in [robustness, frequency])
    output = (args.output or robustness/datetime.now(timezone.utc).strftime('followup-comparisons-%Y%m%dT%H%M%SZ')).resolve()
    assert output.is_relative_to(root) and not any(path.is_relative_to(output) for path in [base, robustness, frequency])
    output.mkdir(parents=True, exist_ok=False)
    selection = read(frequency/'selection.json')
    primary = selection['chosen_baseline']
    inputs = {str(path): sha(path) for path in [robustness/'config.json', frequency/'config.json',
                                              frequency/'selection.json', frequency/'heldout_predictions.npz']}
    with np.load(frequency/'heldout_predictions.npz', allow_pickle=False) as saved:
        y, keys, games, impossible = [saved[name].copy() for name in ['y', 'pitch_keys', 'game_pk', 'impossible_dp']]
        assert not ((y == 9) & impossible).any()
        baseline_p = {name+'__'+variant: saved[name+'__'+variant].copy()
                      for name in selection['baselines'] for variant in ['raw', 'tempered']}
    models = {}
    coverage = {}
    for variant in config['variants']:
        coverage[variant] = []
        for seed in config['model_seeds']:
            directory = robustness/'models'/f'seed{seed}'/variant
            if not (directory/'result.json').exists():
                continue
            result = read(directory/'result.json')
            assert result['seed'] == seed and result['variant'] == variant
            path = directory/'predictions.npz'
            assert sha(path) == result['artifact_hashes']['predictions.npz']
            for file in [path, directory/'result.json']:
                inputs[str(file)] = sha(file)
            with np.load(path, allow_pickle=False) as saved:
                for name, value in [('y', y), ('pitch_keys', keys), ('game_pk', games)]:
                    np.testing.assert_array_equal(saved[name], value)
                p = saved['delivery_integrated_calibrated'].copy()
            models[variant, seed] = {'raw': scores(y, p), 'legal': scores(y, condition_on_legality(p, impossible))}
            coverage[variant].append(seed)
    baseline_scores = {name: {'raw': scores(y, p), 'legal': scores(y, condition_on_legality(p, impossible))}
                       for name, p in baseline_p.items()}
    report = {'scope': 'Exploratory new-seed sensitivity and stronger-baseline audit on previously inspected DEV; no fitting or DEV selection.',
              'estimand': 'Equal-seed average of single-model per-pitch losses, not ensemble probabilities.',
              'n': len(y), 'games': len(np.unique(games)), 'coverage': coverage,
              'complete_all_variants': all(seeds == config['model_seeds'] for seeds in coverage.values()),
              'calibration_selected_frequency_baseline': primary,
              'frequency_uncertainty': 'Frequency model fixed; crossed resampling captures neural seed variation and shared game sampling, not refitting on alternative training data.',
              'seed_sets': {}}
    for label, expected in [('all_five', [42, 43, 44, 45, 46]), ('new_four', [43, 44, 45, 46])]:
        target = report['seed_sets'][label] = {'expected_seeds': expected, 'contrasts': {}, 'strong_baselines': {}}
        for left_name, right_name in [('full_transformer', 'flatten_mlp'), ('full_transformer', 'capacity_mlp'),
                                      ('no_game_context', 'full_transformer'), ('no_batter_style', 'full_transformer'),
                                      ('no_clusters', 'full_transformer')]:
            matched = [seed for seed in expected if (left_name, seed) in models and (right_name, seed) in models]
            if not matched:
                continue
            name = left_name+'_minus_'+right_name
            contrast = {'seeds': matched, 'complete': matched == expected}
            for regime in ['raw', 'legal']:
                left = {metric: np.stack([models[left_name, seed][regime][metric] for seed in matched]) for metric in ['log_loss', 'brier_multiclass']}
                right = {metric: np.stack([models[right_name, seed][regime][metric] for seed in matched]) for metric in left}
                contrast[regime] = independent_bootstrap(left, right, games, matched)
            target['contrasts'][name] = contrast
        matched = [seed for seed in expected if ('full_transformer', seed) in models]
        for name in baseline_scores:
            comparison = {'seeds': matched, 'complete': matched == expected,
                          'direction': 'full Transformer minus frequency baseline; negative favors Transformer',
                          'primary_baseline': name == primary+'__tempered'}
            for regime in ['raw', 'legal']:
                left = {metric: np.stack([models['full_transformer', seed][regime][metric] for seed in matched]) for metric in ['log_loss', 'brier_multiclass']}
                right = {metric: np.repeat(baseline_scores[name][regime][metric][None], len(matched), axis=0) for metric in left}
                comparison[regime] = independent_bootstrap(left, right, games, matched)
            target['strong_baselines'][name] = comparison
    original_new = robustness/'new_seed_replication.json'
    if original_new.exists():
        earlier = read(original_new)
        inputs[str(original_new)] = sha(original_new)
        verified = []
        for name, result in earlier.items():
            actual = report['seed_sets']['new_four']['contrasts'].get(name)
            if actual is None:
                continue
            assert actual['seeds'] == result['seeds']
            for metric in ['log_loss', 'brier_multiclass']:
                for key in ['model_minus_reference', 'game_only_bootstrap95', 'crossed_seed_game_bootstrap95']:
                    np.testing.assert_allclose(actual['raw'][metric][key], result[metric][key], rtol=1e-7, atol=1e-10)
            verified.append(name)
        report['root_new_seed_results_verified'] = verified
    for path, expected in inputs.items():
        assert sha(Path(path)) == expected
    source_hashes = {}
    for name in ['compare_sequence_followups.py', 'audit_sequence_robustness.py', 'diagnose_sequence_legality.py']:
        source = Path(__file__).with_name(name)
        source_hashes[name] = sha(source)
        shutil.copyfile(source, output/name)
    report['source_hashes'], report['input_hashes'] = source_hashes, inputs
    (output/'comparisons.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'complete_all_variants': report['complete_all_variants'],
                      'models': len(models), 'root_new_seed_verified': report.get('root_new_seed_results_verified', [])}, indent=2))


if __name__ == '__main__':
    main()
