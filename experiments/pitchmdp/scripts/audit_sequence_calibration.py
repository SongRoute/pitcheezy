"""CPU-only audit of saved ensemble/blend arrays; independent optimality checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from audit_sequence_robustness import independent_bootstrap, scores, sha


def read(path):
    return json.loads(path.read_text())


def key_hash(keys):
    return hashlib.sha256(np.asarray(keys, dtype=np.int64).tobytes()).hexdigest()


def independent_optimum(y, model, baseline, objective):
    model, baseline = np.asarray(model, float), np.asarray(baseline, float)
    difference = model-baseline
    if objective == 'brier_multiclass':
        denominator = np.square(difference).sum()
        weight = float(np.clip(((np.eye(10)[y]-baseline)*difference).sum()/denominator, 0., 1.)) if denominator else 0.
    else:
        delta = difference[np.arange(len(y)), y]
        initial = baseline[np.arange(len(y)), y]
        def derivative(w):
            return float(-np.mean(delta/(initial+w*delta)))
        weight = 0. if derivative(0.) >= 0 else 1. if derivative(1.) <= 0 else float(brentq(derivative, 0., 1., xtol=1e-13))
    def loss(w):
        return float(scores(y, w*model+(1-w)*baseline)[objective].mean())
    return {'independent_weight': weight, 'independent_calibration_score': loss(weight),
            'endpoint_baseline_score': loss(0.), 'endpoint_ensemble_score': loss(1.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True, type=Path)
    args = parser.parse_args()
    output = args.calibration.resolve()
    config, result = read(output/'config.json'), read(output/'results.json')
    assert (output/'runtime.json').exists(), 'Calibration run must be complete'
    base, robustness = Path(config['base_run']), Path(config['robustness_run'])
    assert config['seeds'] == [42, 43, 44, 45, 46]
    for name, expected in config['source_hashes'].items():
        assert sha(output/'source'/name) == expected
    for name, expected in config['reference_hashes'].items():
        assert sha(Path(name)) == expected
    with np.load(output/'predictions.npz', allow_pickle=False) as saved:
        dev = {name: saved[name].copy() for name in saved.files}
    with np.load(output/'calibration_predictions.npz', allow_pickle=False) as saved:
        cal = {name: saved[name].copy() for name in saved.files}
    assert len(dev['y']) == 7276 and len(cal['y']) == 12000
    for saved in [dev, cal]:
        assert saved['seed_predictions'].shape == (5, len(saved['y']), 10)
        np.testing.assert_array_equal(saved['seed_predictions'].mean(0), saved['ensemble'])
        for p in saved['seed_predictions']:
            scores(saved['y'], p)
        scores(saved['y'], saved['count_hand'])
    for i, seed in enumerate(config['seeds']):
        directory = robustness/'models'/f'seed{seed}'/'full_transformer'
        item = read(directory/'result.json')
        assert item['seed'] == seed and item['variant'] == 'full_transformer'
        assert sha(directory/'predictions.npz') == item['artifact_hashes']['predictions.npz']
        with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
            for identifier in ['y', 'pitch_keys', 'game_pk']:
                np.testing.assert_array_equal(dev[identifier], saved[identifier])
            np.testing.assert_allclose(dev['seed_predictions'][i], saved['delivery_integrated_calibrated'], rtol=1e-7, atol=1e-7)
    with np.load(base/'heldout_predictions.npz', allow_pickle=False) as saved:
        for identifier in ['y', 'pitch_keys', 'game_pk']:
            np.testing.assert_array_equal(dev[identifier], saved[identifier])
        np.testing.assert_allclose(dev['count_hand'], saved['count_hand'], rtol=0, atol=0)
    # Saved CAL/DEV labels and row identities must match the approved processed log.
    sys.path.insert(0, str(base/'source'))
    from pitchmdp.model import eligible, outcome_labels
    processed = base.parent.parent/'processed/pitches.parquet'
    quality = read(base/'audit_supplemental/data_quality.json')
    assert sha(processed) == quality['processed_sha256']
    columns = ['game_pk', 'at_bat_number', 'pitch_number', 'game_date', 'split', 'description', 'events',
               'strikes', 'balls', 'supported_pa', 'pitch_type', 'plate_x', 'plate_z']
    keys = columns[:3]
    frame = pd.read_parquet(processed, columns=columns).sort_values(['game_date', *keys], ignore_index=True)
    assert frame.game_date.dt.year.isin([2023, 2024, 2025]).all()
    base_cfg = read(base/'config.json')
    available = frame[frame.split.eq('calibration') & eligible(frame)]
    all_cal = available.sample(min(len(available), base_cfg['calibration_rows']), random_state=42).sort_index()
    temperature = all_cal.sample(base_cfg['delivery_calibration_rows'], random_state=42)
    blend = all_cal.loc[~all_cal.index.isin(temperature.index)]
    np.testing.assert_array_equal(cal['pitch_keys'], blend[keys].to_numpy())
    np.testing.assert_array_equal(cal['game_pk'], blend.game_pk.to_numpy())
    np.testing.assert_array_equal(cal['y'], outcome_labels(blend))
    assert key_hash(cal['pitch_keys']) == result['calibration_row_hash']
    assert key_hash(dev['pitch_keys']) == result['dev_row_hash']
    assert set(cal['game_pk']).isdisjoint(dev['game_pk'])
    report = {'status': 'passed', 'five_fixed_seed_ensemble_reconstructed': True,
              'dev_seed_predictions_match_archives': True, 'cal_labels_keys_dates_verified': True,
              'original_4000_temperature_rows_excluded': True, 'source_references_verified': True,
              'calibration_n': len(cal['y']), 'dev_n': len(dev['y']), 'dev_games': len(np.unique(dev['game_pk'])),
              'scope': 'Game uncertainty conditional on the fitted five-model ensemble and saved blend weights; not CAL-fit or seed-fit uncertainty.',
              'metadata_correction': 'Original paired limits fields contain stale one-seed boilerplate; the experiment-level interval_scope correctly describes the fixed ensemble.',
              'calibration_limitation': 'All original16000CAL rows were used in early stopping; remaining12000 are not independent validation.',
              'blends': {}, 'comparisons': {}}
    for name in ['ensemble', 'count_hand']:
        values = scores(dev['y'], dev[name])
        for metric, per_pitch in values.items():
            np.testing.assert_allclose(per_pitch.mean(), result[name][metric], rtol=1e-8, atol=1e-9)
    for objective in ['log_loss', 'brier_multiclass']:
        item = result['blends'][objective]
        independent = independent_optimum(cal['y'], cal['ensemble'], cal['count_hand'], objective)
        weight = item['model_weight']
        assert np.isclose(weight+item['baseline_weight'], 1.) and 0 <= weight <= 1
        np.testing.assert_allclose(weight, independent['independent_weight'], rtol=0, atol=2e-5)
        cal_score = scores(cal['y'], weight*cal['ensemble'].astype(float)+(1-weight)*cal['count_hand'])[objective].mean()
        np.testing.assert_allclose(cal_score, item['calibration_score'], rtol=0, atol=1e-10)
        assert cal_score <= min(independent['endpoint_baseline_score'], independent['endpoint_ensemble_score'])+1e-10
        assert cal_score-independent['independent_calibration_score'] < 1e-9
        p = weight*dev['ensemble']+(1-weight)*dev['count_hand']
        np.testing.assert_array_equal(p, dev['blend_'+objective])
        values = scores(dev['y'], p)
        for metric, per_pitch in values.items():
            np.testing.assert_allclose(per_pitch.mean(), item['dev_metrics'][metric], rtol=1e-8, atol=1e-9)
        report['blends'][objective] = {**independent, 'saved_weight': weight,
                                      'calibration_optimality_gap': float(cal_score-independent['independent_calibration_score']),
                                      'dev_metrics': {metric: float(per_pitch.mean()) for metric, per_pitch in values.items()}}
        for reference, original_key in [('ensemble', 'paired_vs_ensemble'), ('count_hand', 'paired_vs_baseline')]:
            right = scores(dev['y'], dev[reference])
            comparison = independent_bootstrap({metric: value[None] for metric, value in values.items()},
                                               {metric: value[None] for metric, value in right.items()},
                                               dev['game_pk'], ['fixed_ensemble'])
            np.testing.assert_allclose(comparison['log_loss']['model_minus_reference'], item[original_key]['model_minus_reference_log_loss'], rtol=0, atol=1e-7)
            np.testing.assert_allclose(comparison['log_loss']['game_only_bootstrap95'], item[original_key]['bootstrap95'], rtol=0, atol=1e-7)
            report['comparisons']['blend_'+objective+'_minus_'+reference] = comparison
    ensemble_scores, baseline_scores = scores(dev['y'], dev['ensemble']), scores(dev['y'], dev['count_hand'])
    comparison = independent_bootstrap({metric: value[None] for metric, value in ensemble_scores.items()},
                                       {metric: value[None] for metric, value in baseline_scores.items()},
                                       dev['game_pk'], ['fixed_ensemble'])
    for key, original_key in [('model_minus_reference', 'model_minus_reference_log_loss'), ('game_only_bootstrap95', 'bootstrap95')]:
        np.testing.assert_allclose(comparison['log_loss'][key], result['ensemble_paired_vs_baseline'][original_key], atol=1e-7, rtol=0)
    report['comparisons']['ensemble_minus_count_hand'] = comparison
    for name in ['predictions.npz', 'calibration_predictions.npz', 'results.json', 'config.json', 'runtime.json']:
        report.setdefault('audited_artifact_hashes', {})[name] = sha(output/name)
    destination = output/'independent_audit'
    destination.mkdir(exist_ok=True)
    for name in ['audit_sequence_calibration.py', 'audit_sequence_robustness.py']:
        shutil.copyfile(Path(__file__).with_name(name), destination/name)
        report.setdefault('audit_source_hashes', {})[name] = sha(destination/name)
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'report': str(destination/'audit.json'), 'weights': report['blends']}, indent=2))


if __name__ == '__main__':
    main()
