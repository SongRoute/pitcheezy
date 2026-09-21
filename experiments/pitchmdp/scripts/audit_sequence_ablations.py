"""Independent CPU-only audit of frozen sequence ablation artifacts."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil

import numpy as np

from audit_sequence_run import digest, paired_bootstrap


def metric_values(y, p):
    p = np.asarray(p, dtype=np.float64)
    assert p.shape == (len(y), 10) and np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-5)
    return {'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            'brier_multiclass': float(np.square(p-np.eye(10)[y]).sum(1).mean()),
            'accuracy': float((p.argmax(1) == y).mean())}


def check_mask_class(source, masks):
    """Execute only the archived context-mask class against a synthetic encoder."""
    tree = ast.parse(source)
    selected = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MaskedContext')
    namespace = {'VARIANTS': {key: tuple(value['zero_indices']) for key, value in masks['variants'].items()},
                 'CONTEXT_SCHEMA': masks['schema']}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), '<archived MaskedContext>', 'exec'), namespace)
    matrix = np.arange(1, 85, dtype=np.float32).reshape(3, 28)
    class Encoder:
        def report(self):
            return {'features': masks['schema']}
        def transform(self, frame):
            return matrix
    original = matrix.copy()
    for variant, details in masks['variants'].items():
        masked = namespace['MaskedContext'](Encoder(), variant).transform(None)
        keep = np.array(details['keep_mask'], dtype=bool)
        np.testing.assert_array_equal(masked[:, keep], original[:, keep])
        np.testing.assert_array_equal(masked[:, ~keep], 0.)
        np.testing.assert_array_equal(matrix, original)
        np.testing.assert_array_equal(masked[:, [0, 1, 9, 10]], original[:, [0, 1, 9, 10]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation', required=True, type=Path)
    args = parser.parse_args()
    output = args.ablation.resolve()
    config = json.loads((output/'config.json').read_text())
    data = json.loads((output/'data.json').read_text())
    masks = json.loads((output/'context_masks.json').read_text())
    base = Path(config['base_run'])
    assert set(config['variants']) == {'no_game_context', 'no_batter_style'}, 'Expected the two prespecified default ablations'
    expected_masks = {'no_game_context': list(range(2, 9)), 'no_batter_style': list(range(11, 28))}
    assert config['zero_context_indices'] == expected_masks
    assert len(masks['channels']) == 28
    for variant in config['variants']:
        assert masks['variants'][variant]['zero_indices'] == expected_masks[variant]
        assert masks['variants'][variant]['keep_mask'] == [j not in expected_masks[variant] for j in range(28)]
    for name, sha in config['source_hashes'].items():
        assert digest(output/'source'/name) == sha, 'Ablation frozen source mismatch: '+name
    for name, sha in config['reference_artifact_hashes'].items():
        assert digest(base/name) == sha, 'Original reference mutated: '+name
    original_sources = json.loads((base/'source_hashes.json').read_text())
    for name, sha in original_sources.items():
        assert config['source_hashes'][name] == sha
        assert digest(base/'source'/name) == sha
    check_mask_class((output/'source/scripts/run_sequence_ablations.py').read_text(), masks)
    samples = json.loads((base/'samples.json').read_text())
    original = json.loads((base/'all_count_results.json').read_text())
    assert data['rows_hash'] == samples['rows_hash']
    for key in ['train_used', 'calibration_used', 'dev_rows', 'dev_games']:
        assert data[key] == samples[key], key
    physics_identity = json.loads((base/'audit_supplemental/sequence_physics.json').read_text())['identity']
    assert data['data_identity'] == physics_identity
    assert data['normalizer'] == json.loads((base/'audit_supplemental/normalizer_report.json').read_text())
    assert data['context'] == samples['context'] and data['delivery'] == samples['delivery']
    report = {'status': 'waiting_for_both_variants', 'mask_synthetic_check': 'passed',
              'same_ordered_samples': True, 'same_normalizer_context_delivery': True,
              'frozen_sources_verified': config['source_hashes'],
              'immutable_reference_hashes_verified': config['reference_artifact_hashes'],
              'no_training_or_model_inference': True,
              'scope': 'Post-DEV exploratory comparisons; intervals measure game-sampling uncertainty conditional on one fitted seed.',
              'legal_state_contract': 'Only response-network context arrays are masked; original frame, legal count, handedness and physical history remain unchanged.'}
    if (output/'runtime.json').exists():
        runtime = json.loads((output/'runtime.json').read_text())
        assert runtime['source_hashes_end'] == config['source_hashes']
        results = json.loads((output/'ablation_results.json').read_text())
        assert set(results['variants']) == set(config['variants'])
        report['results'] = {}
        with np.load(base/'heldout_predictions.npz', allow_pickle=False) as old, np.load(output/'heldout_predictions.npz', allow_pickle=False) as saved:
            for key in ['y', 'game_pk', 'pitch_keys']:
                np.testing.assert_array_equal(saved[key], old[key])
            np.testing.assert_array_equal(saved['full_transformer'], old['transformer'])
            np.testing.assert_array_equal(saved['count_hand_baseline'], old['count_hand'])
            y, games = saved['y'], saved['game_pk']
            for variant in config['variants']:
                item = results['variants'][variant]
                training = item['training']
                for key in ['parameter_count', 'training_rows', 'calibration_rows', 'network', 'seed']:
                    assert training[key] == original['transformer']['training'][key], key
                assert training['epochs_run'] <= config['base_config']['epochs']
                assert training['best_epoch'] <= training['epochs_run']
                assert training['delivery_calibration_rows'] == original['transformer']['training']['delivery_calibration_rows']
                scores = {}
                for suffix, metric_key in [('', 'primary_delivery_integrated'),
                                          ('_uncalibrated', 'delivery_integrated_uncalibrated'),
                                          ('_conditional_calibrated', 'conditional_current_physics_diagnostic'),
                                          ('_conditional_uncalibrated', 'conditional_current_physics_uncalibrated')]:
                    scores[metric_key] = metric_values(y, saved[variant+suffix])
                    for key, value in scores[metric_key].items():
                        np.testing.assert_allclose(value, item[metric_key][key], rtol=1e-7, atol=1e-8)
                comparisons = {}
                for reference, name in [('full_transformer', 'paired_ablation_minus_full'),
                                        ('count_hand_baseline', 'paired_ablation_minus_baseline')]:
                    comparison = paired_bootstrap(y, saved[variant], saved[reference], games, config['bootstrap_replicates'])
                    for key in ['model_minus_reference_log_loss', 'bootstrap95', 'games']:
                        np.testing.assert_allclose(comparison[key], item[name][key], rtol=1e-6, atol=1e-8)
                    comparisons[name] = comparison
                report['results'][variant] = {'metrics': scores, 'comparisons': comparisons,
                                               'checkpoint_sha256': digest(output/f'{variant}.pt')}
        report.update(status='passed', source_unchanged_during_execution=True,
                      predictions_sha256=digest(output/'heldout_predictions.npz'))
    shutil.copyfile(Path(__file__).resolve(), output/'audit_sequence_ablations.py')
    # Preserve the independently implemented bootstrap helper used by this audit.
    shutil.copyfile(Path(__file__).with_name('audit_sequence_run.py'), output/'audit_sequence_run.py')
    report['audit_source_sha256'] = {name: digest(output/name) for name in ['audit_sequence_ablations.py', 'audit_sequence_run.py']}
    (output/'independent_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'audit': str(output/'independent_audit.json')}, indent=2))


if __name__ == '__main__':
    main()
