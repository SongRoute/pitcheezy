"""CPU artifact audit of fixed-checkpoint integration; supports partial snapshots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from audit_sequence_robustness import SEEDS, MASKS, REGIMES, CONTRASTS, read, sha, scores, independent_bootstrap
from audit_sequence_frequency_blend import arrays
from audit_sequence_calibration import independent_optimum, key_hash


def unchanged_checkpoint(old, new):
    for key in ['network_config', 'seed', 'temperature']:
        assert old[key] == new[key], 'Checkpoint identity changed: '+key
    assert old['state_dict'].keys() == new['state_dict'].keys()
    digest = hashlib.sha256()
    for key, tensor in sorted(old['state_dict'].items()):
        assert tensor.device.type == new['state_dict'][key].device.type == 'cpu'
        assert tensor.dtype == new['state_dict'][key].dtype and torch.equal(tensor, new['state_dict'][key]), key
        array = tensor.numpy()
        for value in [key.encode(), str(array.dtype).encode(), str(array.shape).encode(), array.tobytes()]:
            digest.update(value)
    return digest.hexdigest()


def compare_arrays(y, p, q, games, seeds):
    left = [scores(y, row) for row in p]
    right = [scores(y, row) for row in q]
    return independent_bootstrap({k: np.stack([x[k] for x in left]) for k in left[0]},
                                 {k: np.stack([x[k] for x in right]) for k in right[0]}, games, seeds)


def verify_comparison(actual, expected):
    for metric, fields in actual.items():
        for key, value in fields.items():
            if key == 'per_seed_difference':
                assert set(value) == set(expected[metric][key])
                for seed, number in value.items():
                    np.testing.assert_allclose(number, expected[metric][key][seed], atol=1e-10)
            else:
                np.testing.assert_allclose(value, expected[metric][key], atol=1e-10)


def ensemble_audit(folder, original, models):
    cal, dev = arrays(folder/'calibration_predictions.npz'), arrays(folder/'predictions.npz')
    result, config = read(folder/'results.json'), read(folder/'config.json')
    assert config['seeds'] == SEEDS and config['delivery_draws'] == 100
    assert len(cal['y']) == 12000 and len(dev['y']) == 7276
    reference_cal = arrays(original/'calibration-20260921T031450Z/calibration_predictions.npz')
    for field in ['y', 'pitch_keys', 'game_pk', 'count_hand']:
        np.testing.assert_array_equal(cal[field], reference_cal[field])
    for split in [cal, dev]:
        np.testing.assert_array_equal(split['seed_predictions'].mean(0), split['ensemble'])
    assert key_hash(cal['pitch_keys']) == result['calibration_row_hash']
    report = {'status': 'passed', 'weights': {}, 'comparisons': {}}
    for i, seed in enumerate(SEEDS):
        directory = folder.parent/'models'/f'seed{seed}'/'full_transformer'
        member = arrays(directory/'blend_calibration_predictions.npz')
        for field in ['y', 'pitch_keys', 'game_pk']:
            np.testing.assert_array_equal(cal[field], member[field])
            np.testing.assert_array_equal(dev[field], models[('full_transformer', seed)][field])
        np.testing.assert_array_equal(cal['seed_predictions'][i], member['probabilities'])
        np.testing.assert_array_equal(dev['seed_predictions'][i], models[('full_transformer', seed)][REGIMES[0]])
    for objective, item in result['blends'].items():
        optimum = independent_optimum(cal['y'], cal['ensemble'], cal['count_hand'], objective)
        weight = item['model_weight']
        np.testing.assert_allclose(weight, optimum['independent_weight'], atol=2e-5, rtol=0)
        np.testing.assert_array_equal(weight*dev['ensemble']+(1-weight)*dev['count_hand'], dev['blend_'+objective])
        report['weights'][objective] = {**optimum, 'saved_weight': weight}
        for reference, field in [('ensemble', 'paired_vs_ensemble'), ('count_hand', 'paired_vs_baseline')]:
            contrast = compare_arrays(dev['y'], dev['blend_'+objective][None], dev[reference][None], dev['game_pk'], ['fixed_ensemble'])
            np.testing.assert_allclose(contrast['log_loss']['model_minus_reference'], item[field]['model_minus_reference_log_loss'], atol=1e-7)
            np.testing.assert_allclose(contrast['log_loss']['game_only_bootstrap95'], item[field]['bootstrap95'], atol=1e-7)
            report['comparisons'][objective+'_minus_'+reference] = contrast
    for name in ['ensemble', 'count_hand', 'blend_log_loss', 'blend_brier_multiclass']:
        expected = result[name] if not name.startswith('blend_') else result['blends'][name.removeprefix('blend_')]['dev_metrics']
        for metric, value in scores(dev['y'], dev[name]).items():
            np.testing.assert_allclose(value.mean(), expected[metric], atol=1e-10)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration', required=True, type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    folder = args.integration.resolve()
    config, data = read(folder/'config.json'), read(folder/'data.json')
    original = Path(config['robustness_run'])
    assert config['seeds'] == SEEDS and set(config['variants']) == set(MASKS)
    assert config['draws'] == 100 and config['neural_training'] is False
    for path, expected in config['source_hashes'].items():
        assert sha(folder/'source'/path) == expected
    for path, expected in config['reference_hashes'].items():
        assert sha(Path(path)) == expected
    old_data = read(original/'data.json')
    for field in ['data_identity', 'ordered_rows_hash', 'delivery_calibration_rows_hash', 'dev_rows', 'dev_games']:
        assert data[field] == old_data[field]
    assert data['delivery_calibration_rows_hash'] == config['temperature_row_hash']
    assert data['blend_calibration_rows_hash'] == config['blend_calibration_row_hash']
    base = arrays(Path(config['base_run'])/'heldout_predictions.npz')
    summary = read(folder/'summary.json') if (folder/'summary.json').exists() else {}
    report = {'status': 'partial', 'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'models': {},
              'comparisons': {}, 'comparisons100_minus25': {}, 'new_seeds43_46': {},
              'scope': 'Fixed checkpoints, fixed delivery pools; paired seed/game intervals exclude pool Monte Carlo uncertainty. Previously inspected DEV.',
              'limitation': 'Temperature-fit CAL logits are not archived; temperature metadata and checkpoint consistency verified, optimizer not independently replayed.'}
    models, old_models = {}, {}
    for variant in MASKS:
        for seed in SEEDS:
            directory = folder/'models'/f'seed{seed}'/variant
            if not (directory/'result.json').exists():
                continue
            item = read(directory/'result.json')
            # The runner publishes result.json before adding the final CAL artifact hashes.
            if 'calibration100.json' not in item['artifact_hashes']:
                continue
            assert item['seed'] == seed and item['variant'] == variant and item['zero_context_indices'] == MASKS[variant]
            for name, expected in item['artifact_hashes'].items():
                assert sha(directory/name) == expected
            source = original/'models'/f'seed{seed}'/variant
            old = torch.load(source/'model.pt', map_location='cpu', weights_only=False)
            new = torch.load(directory/'model.pt', map_location='cpu', weights_only=False)
            tensor_digest = unchanged_checkpoint(old, new)
            assert tensor_digest == item['origin']['neural_tensor_sha256']
            assert item['origin']['original_checkpoint_sha256'] == sha(source/'model.pt')
            fit = read(directory/'calibration100.json')
            assert fit == item['origin']['temperature_fit'] == new['report']['integration100_recalibration']
            assert new['delivery_temperature'] == fit['temperature'] and .5 <= fit['temperature'] <= 2.5
            assert new['report']['neural_training_in_this_run'] is False
            for key, value in old['report'].items():
                assert new['report'][key] == value
            current, previous = arrays(directory/'predictions.npz'), arrays(source/'predictions.npz')
            for field in ['y', 'pitch_keys', 'game_pk']:
                np.testing.assert_array_equal(current[field], base[field])
                np.testing.assert_array_equal(current[field], previous[field])
            np.testing.assert_array_equal(current['game_month'], previous['game_month'])
            for regime in REGIMES:
                if regime.startswith('conditional'):
                    np.testing.assert_array_equal(current[regime], previous[regime])
                values = scores(current['y'], current[regime])
                for metric, per_pitch in values.items():
                    np.testing.assert_allclose(per_pitch.mean(), item['metrics'][regime][metric], atol=1e-10)
                    for month in np.unique(current['game_month']):
                        selected = current['game_month'] == month
                        np.testing.assert_allclose(per_pitch[selected].mean(), item['monthly'][month]['metrics'][regime][metric], atol=1e-10)
            models[(variant, seed)], old_models[(variant, seed)] = current, previous
            report['models'][variant+'/'+str(seed)] = {'tensors_identical': True, 'tensor_sha256': tensor_digest,
                'temperature100': fit['temperature'], 'temperature25': old['delivery_temperature'],
                'metrics': {metric: float(values.mean()) for metric, values in scores(current['y'], current[REGIMES[0]]).items()}}
    y, games = base['y'], base['game_pk']
    for left, right in CONTRASTS:
        seeds = [seed for seed in SEEDS if (left, seed) in models and (right, seed) in models]
        if not seeds:
            continue
        p, q = (np.stack([models[(variant, seed)][REGIMES[0]] for seed in seeds]) for variant in [left, right])
        name = left+'_minus_'+right
        actual = compare_arrays(y, p, q, games, seeds)
        if name in summary.get('comparisons', {}) and summary['comparisons'][name]['seeds'] == seeds:
            verify_comparison(actual, summary['comparisons'][name])
        report['comparisons'][name] = {'seeds': seeds, 'complete': seeds == SEEDS, **actual}
        if seeds == SEEDS:
            report['new_seeds43_46'][name] = compare_arrays(y, p[1:], q[1:], games, SEEDS[1:])
    for variant in MASKS:
        seeds = [seed for seed in SEEDS if (variant, seed) in models]
        if not seeds:
            continue
        p = np.stack([models[(variant, seed)][REGIMES[0]] for seed in seeds])
        q = np.stack([old_models[(variant, seed)][REGIMES[0]] for seed in seeds])
        actual = compare_arrays(y, p, q, games, seeds)
        if seeds == SEEDS and variant in summary.get('comparisons100_minus25', {}):
            verify_comparison(actual, summary['comparisons100_minus25'][variant])
        report['comparisons100_minus25'][variant] = {'seeds': seeds, **actual}
    if (folder/'calibration100/runtime.json').exists():
        report['ensemble_calibration'] = ensemble_audit(folder/'calibration100', original, models)
    if (folder/'convergence.json').exists():
        convergence = read(folder/'convergence.json')
        p400 = arrays(folder/'predictions400.npz')
        for field in ['y', 'pitch_keys', 'game_pk']:
            np.testing.assert_array_equal(p400[field], base[field])
        assert sha(folder/'delivery_draw400.pkl') == convergence['pool400_sha256']
        assert read(folder/'calibration400.json') == convergence['temperature400']
        assert .5 <= convergence['temperature400']['temperature'] <= 2.5
        for count, archive in [('25', old_models[('full_transformer', 42)]), ('100', models[('full_transformer', 42)]), ('400', p400)]:
            for regime in REGIMES[:2]:
                for metric, values in scores(y, archive[regime]).items():
                    np.testing.assert_allclose(values.mean(), convergence['draws'][count][regime][metric], atol=1e-10)
        actual = compare_arrays(y, p400[REGIMES[0]][None], models[('full_transformer', 42)][REGIMES[0]][None], games, [42])
        verify_comparison(actual, convergence['paired400_minus100'])
        report['convergence'] = {**convergence, 'independently_recomputed400_minus100': actual}
    complete = len(models) == 30 and 'convergence' in report and (folder/'runtime.json').exists()
    if complete:
        runtime = read(folder/'runtime.json')
        assert runtime['integrity_ok'] and runtime['neural_training'] is False
        assert runtime['source_hashes_end'] == config['source_hashes'] and runtime['reference_hashes_end'] == config['reference_hashes']
        report['status'] = 'passed'
    report['model_count'] = len(models)
    destination = folder/'independent_audit'
    destination.mkdir(exist_ok=True)
    for name in [Path(__file__).name, 'audit_sequence_robustness.py', 'audit_sequence_frequency_blend.py', 'audit_sequence_calibration.py']:
        shutil.copyfile(Path(__file__).with_name(name), destination/name)
        report.setdefault('audit_source_hashes', {})[name] = sha(destination/name)
    for name in ['config.json', 'data.json', 'summary.json', 'runtime.json', 'convergence.json']:
        if (folder/name).exists():
            report.setdefault('artifact_hashes', {})[name] = sha(folder/name)
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'models': len(models), 'report': str(destination/'audit.json')}, indent=2))
    if args.require_complete and not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
