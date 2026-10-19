"""Independent CPU artifact audit; no model imports, inference, or raw-data access.

Partial runs are explicitly incomplete. --require-phase exits nonzero until that
phase and all its models are complete, while still saving the available audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

SEEDS = [42, 43, 44, 45, 46]
MASKS = {'full_transformer': [], 'flatten_mlp': [], 'no_game_context': list(range(2, 9)),
         'no_batter_style': list(range(11, 28)), 'capacity_mlp': [], 'no_clusters': list(range(23, 28))}
REGIMES = ['delivery_integrated_calibrated', 'delivery_integrated_uncalibrated',
           'conditional_calibrated', 'conditional_uncalibrated']
PHASES = {'replication': ['full_transformer', 'flatten_mlp', 'no_game_context', 'no_batter_style'],
          'capacity': ['full_transformer', 'capacity_mlp'], 'clusters': ['full_transformer', 'no_clusters']}
CONTRASTS = [('full_transformer', 'flatten_mlp'), ('no_game_context', 'full_transformer'),
             ('no_batter_style', 'full_transformer'), ('full_transformer', 'capacity_mlp'),
             ('no_clusters', 'full_transformer')]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def scores(y, p):
    p = np.asarray(p, dtype=float)
    assert p.shape == (len(y), 10) and np.isfinite(p).all() and np.all(p >= 0)
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-5)
    return {'log_loss': -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)),
            'brier_multiclass': np.square(p-np.eye(10)[y]).sum(1)}


def independent_bootstrap(model_scores, reference_scores, games, seeds, replicates=2000, bootstrap_seed=42):
    """Loop over explicit draws, jointly selecting seed IDs and whole game sums."""
    groups = [np.flatnonzero(games == game) for game in np.unique(games)]
    counts = np.array([len(group) for group in groups])
    rng = np.random.default_rng(bootstrap_seed)
    game_draws = rng.integers(len(groups), size=(replicates, len(groups)))
    seed_draws = rng.integers(len(seeds), size=(replicates, len(seeds)))
    result = {}
    for metric in model_scores:
        left, right = np.asarray(model_scores[metric]), np.asarray(reference_scores[metric])
        delta = left-right
        sums = np.column_stack([delta[:, group].sum(1) for group in groups])
        fixed, crossed = [], []
        for drawn_games, drawn_seeds in zip(game_draws, seed_draws):
            denominator = len(seeds)*counts[drawn_games].sum()
            fixed.append(sums[:, drawn_games].sum()/denominator)
            crossed.append(sums[np.ix_(drawn_seeds, drawn_games)].sum()/denominator)
        result[metric] = {'mean_model': float(left.mean()), 'mean_reference': float(right.mean()),
                          'model_minus_reference': float(delta.mean()),
                          'per_seed_difference': {str(seed): float(value) for seed, value in zip(seeds, delta.mean(1))},
                          'game_only_bootstrap95': np.quantile(fixed, [.025, .975]).tolist(),
                          'crossed_seed_game_bootstrap95': np.quantile(crossed, [.025, .975]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness', required=True, type=Path)
    parser.add_argument('--require-phase', choices=list(PHASES))
    args = parser.parse_args()
    output = args.robustness.resolve()
    config, data = read(output/'config.json'), read(output/'data.json')
    base, ablations = Path(config['base_run']), Path(config['seed42_ablations'])
    assert config['model_seeds'] == SEEDS and config['sample_encoder_delivery_seed'] == 42
    assert config['allowed_raw_seasons'] == [2023, 2024, 2025] and config['raw_2026_access'] is False
    assert set(config['variants']) == set(MASKS)
    project = Path(__file__).resolve().parents[1]
    current_source_changes = []
    for name, expected in config['source_hashes'].items():
        assert sha(output/'source'/name) == expected, 'Frozen source mismatch: '+name
        if sha(project/name) != expected:
            current_source_changes.append(name)
    assert 'docs/NEXT_EXPERIMENTS_PROTOCOL.md' in config['source_hashes']
    for path, expected in config['reference_hashes'].items():
        assert sha(Path(path)) == expected, 'Reference changed: '+path
    for variant, removed in MASKS.items():
        declared = config['variants'][variant]
        assert declared['zero_context_indices'] == removed
        assert declared['zero_context_channels'] == [config['context_channels'][i] for i in removed]
        assert not set(removed).intersection([0, 1, 9, 10])
        assert declared['architecture'] == ('flatten_mlp' if variant in ['flatten_mlp', 'capacity_mlp'] else 'transformer')
        assert declared['width'] == (234 if variant == 'capacity_mlp' else 128)
    base_samples, ab_data = read(base/'samples.json'), read(ablations/'data.json')
    assert data['ordered_rows_hash'] == base_samples['rows_hash'] == ab_data['rows_hash']
    assert data['delivery_calibration_rows_hash'] == ab_data['delivery_calibration_rows_hash']
    assert data['data_identity'] == ab_data['data_identity']
    for key in ['normalizer', 'context', 'delivery']:
        assert data[key] == ab_data[key]
    assert data['train_rows'] == base_samples['train_used'] and data['calibration_rows'] == base_samples['calibration_used']
    assert data['dev_rows'] == base_samples['dev_rows'] and data['dev_games'] == base_samples['dev_games']
    with np.load(base/'heldout_predictions.npz', allow_pickle=False) as saved:
        y, keys, games = saved['y'], saved['pitch_keys'], saved['game_pk']
    assert hashlib.sha256(keys.astype(np.int64).tobytes()).hexdigest() == data['ordered_rows_hash']['dev']
    # Read one atomic summary snapshot; a fit may finish before its next summary.
    summary = read(output/'summary.json')
    report = {'status': 'partial', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'requested_phase': args.require_phase, 'sources_and_references_verified': True,
              'protocol_sha256': config['source_hashes']['docs/NEXT_EXPERIMENTS_PROTOCOL.md'],
              'data_samples_encoders_and_masks_verified': True,
              'current_workspace_source_changes': current_source_changes,
              'no_training_inference_or_raw_data_access': True,
              'estimator': 'Equal-seed average of per-pitch model losses; not probability ensembling.',
              'uncertainty': 'Whole-game and matched-seed resampling; five seeds do not imply unseen confirmation.',
              'models': {}, 'comparisons': {}, 'summary_pending': []}
    records, measured, months = {}, {}, None
    for variant in MASKS:
        for seed in SEEDS:
            directory = output/'models'/f'seed{seed}'/variant
            result_path = directory/'result.json'
            if not result_path.exists():
                continue
            result = read(result_path)
            assert result['seed'] == seed and result['variant'] == variant
            assert set(result['artifact_hashes']) == {'model.pt', 'predictions.npz'}
            for name, expected in result['artifact_hashes'].items():
                assert sha(directory/name) == expected, 'Completed model changed: '+str(directory/name)
            assert Path(result['checkpoint_path']).resolve() == directory/'model.pt'
            assert result['zero_context_indices'] == MASKS[variant]
            training = result['training']
            expected_count = 278874 if variant == 'capacity_mlp' else 99946 if variant == 'flatten_mlp' else 278762
            assert training['seed'] == seed and training['parameter_count'] == expected_count
            assert training['training_rows'] == data['train_rows'] and training['calibration_rows'] == data['calibration_rows']
            assert training['delivery_calibration_rows'] == config['base_config']['delivery_calibration_rows']
            assert 1 <= training['best_epoch'] <= training['epochs_run'] <= config['base_config']['epochs']
            assert training['network'] == {'kind': config['variants'][variant]['architecture'], 'n_context': 28,
                                            'n_physical': 8, 'n_classes': 10, 'width': config['variants'][variant]['width'], 'length': 6}
            if seed == 42 and variant in PHASES['replication']:
                origin = result['origin']
                assert origin['mode'] == 'reused_without_fit_or_inference'
                source = base/f'all_{"transformer" if variant == "full_transformer" else variant}.pt' if variant in ['full_transformer', 'flatten_mlp'] else ablations/f'{variant}.pt'
                assert Path(origin['source_checkpoint']) == source
                assert sha(source) == origin['source_sha256'] == result['artifact_hashes']['model.pt']
                original_result = (read(base/'all_count_results.json')['transformer' if variant == 'full_transformer' else variant]
                                   if variant in ['full_transformer', 'flatten_mlp'] else
                                   read(ablations/'ablation_results.json')['variants'][variant])
                assert training == original_result['training'], 'Seed42 training record differs from its reused source'
            else:
                assert result['origin']['mode'] == 'new_fit'
            record_scores, means = {}, {}
            with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
                for key, expected in [('y', y), ('pitch_keys', keys), ('game_pk', games)]:
                    np.testing.assert_array_equal(saved[key], expected)
                if months is None:
                    months = saved['game_month'].copy()
                    assert set(months) == {'2025-07', '2025-08', '2025-09'}
                    for month in np.unique(months):
                        assert data['monthly_counts'][month] == {'pitches': int((months == month).sum()),
                                                                  'games': len(np.unique(games[months == month]))}
                np.testing.assert_array_equal(saved['game_month'], months)
                for regime in REGIMES:
                    if seed == 42 and variant in PHASES['replication']:
                        if variant in ['full_transformer', 'flatten_mlp']:
                            kind = 'transformer' if variant == 'full_transformer' else variant
                            original_file = base/'heldout_predictions.npz' if regime == REGIMES[0] else base/'audit_supplemental/all_count_calibration_predictions.npz'
                            original_key = kind if regime == REGIMES[0] else kind+'__'+regime
                        else:
                            original_file = ablations/'heldout_predictions.npz'
                            original_key = variant+{'delivery_integrated_calibrated': '', 'delivery_integrated_uncalibrated': '_uncalibrated',
                                                    'conditional_calibrated': '_conditional_calibrated', 'conditional_uncalibrated': '_conditional_uncalibrated'}[regime]
                        with np.load(original_file, allow_pickle=False) as original_predictions:
                            np.testing.assert_array_equal(saved[regime], original_predictions[original_key])
                    values = scores(y, saved[regime])
                    means[regime] = {metric: float(per_pitch.mean()) for metric, per_pitch in values.items()}
                    for metric, value in means[regime].items():
                        np.testing.assert_allclose(value, result['metrics'][regime][metric], rtol=1e-8, atol=1e-9)
                    for month in np.unique(months):
                        mask = months == month
                        for metric, per_pitch in values.items():
                            np.testing.assert_allclose(per_pitch[mask].mean(), result['monthly'][month]['metrics'][regime][metric], rtol=1e-8, atol=1e-9)
                    if regime == REGIMES[0]:
                        record_scores = values
            records[variant, seed], measured[variant, seed] = result, record_scores
            report['models'][f'{variant}/seed{seed}'] = {'metrics': means, 'origin': result['origin'],
                                                        'result_sha256': sha(result_path), 'artifact_hashes': result['artifact_hashes']}
    for variant in MASKS:
        selected = [seed for seed in SEEDS if (variant, seed) in records]
        if not selected:
            continue
        existing = summary['variants'].get(variant)
        if existing is None or existing['completed_seeds'] != selected:
            report['summary_pending'].append(variant)
            continue
        assert existing['complete'] == (selected == SEEDS)
        for regime in REGIMES:
            for metric in ['log_loss', 'brier_multiclass']:
                values = [records[variant, seed]['metrics'][regime][metric] for seed in selected]
                np.testing.assert_allclose(np.mean(values), existing['metrics'][regime][metric]['mean'], atol=1e-12)
                if len(selected) > 1:
                    np.testing.assert_allclose(np.std(values, ddof=1), existing['metrics'][regime][metric]['seed_std'], atol=1e-12)
                for month in np.unique(months):
                    monthly_mean = np.mean([records[variant, seed]['monthly'][month]['metrics'][regime][metric] for seed in selected])
                    np.testing.assert_allclose(monthly_mean, existing['monthly'][month]['metrics'][regime][metric], atol=1e-12)
    for variant, reference in CONTRASTS:
        selected = [seed for seed in SEEDS if (variant, seed) in measured and (reference, seed) in measured]
        if not selected:
            continue
        name = variant+'_minus_'+reference
        left = {metric: np.stack([measured[variant, seed][metric] for seed in selected]) for metric in ['log_loss', 'brier_multiclass']}
        right = {metric: np.stack([measured[reference, seed][metric] for seed in selected]) for metric in left}
        comparison = independent_bootstrap(left, right, games, selected, config['bootstrap_replicates'], config['bootstrap_seed'])
        comparison.update(seeds=selected, games=len(np.unique(games)), n=len(y), complete=selected == SEEDS)
        report['comparisons'][name] = comparison
        existing = summary['comparisons'].get(name)
        if existing is None or existing['seeds'] != selected:
            report['summary_pending'].append(name)
            continue
        for metric in left:
            for key in ['mean_model', 'mean_reference', 'model_minus_reference', 'game_only_bootstrap95', 'crossed_seed_game_bootstrap95']:
                np.testing.assert_allclose(comparison[metric][key], existing[metric][key], rtol=1e-7, atol=1e-10)
            assert comparison[metric]['per_seed_difference'].keys() == existing[metric]['per_seed_difference'].keys()
            np.testing.assert_allclose(list(comparison[metric]['per_seed_difference'].values()), list(existing[metric]['per_seed_difference'].values()), rtol=1e-7, atol=1e-10)
    report['phase_status'] = {}
    for phase, variants in PHASES.items():
        runtime_path = output/f'runtime_{phase}.json'
        complete_models = all((variant, seed) in records for variant in variants for seed in SEEDS)
        runtime_verified = False
        if runtime_path.exists():
            runtime = read(runtime_path)
            assert runtime['phase'] == phase and runtime['source_hashes_end'] == config['source_hashes']
            assert complete_models, 'Phase completion marker without all required models: '+phase
            runtime_verified = True
        report['phase_status'][phase] = {'all_models_complete': complete_models, 'runtime_verified': runtime_verified,
                                          'status': 'passed' if complete_models and runtime_verified else 'incomplete'}
    requested_complete = (report['phase_status'][args.require_phase]['status'] == 'passed' if args.require_phase else
                          all(item['status'] == 'passed' for item in report['phase_status'].values()))
    if requested_complete and not report['summary_pending']:
        report['status'] = 'passed_requested_phase' if args.require_phase else 'passed_all_phases'
    if args.require_phase and not requested_complete:
        report['status'] = 'incomplete_requested_phase'
    report['completed_models'] = len(records)
    destination = output/'independent_audit'
    destination.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), destination/'audit_sequence_robustness.py')
    report['audit_source_sha256'] = sha(destination/'audit_sequence_robustness.py')
    name = 'audit_'+(args.require_phase or 'progress')+'.json'
    (destination/name).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'models': len(records), 'report': str(destination/name)}, indent=2))
    if args.require_phase and report['status'] != 'passed_requested_phase':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
