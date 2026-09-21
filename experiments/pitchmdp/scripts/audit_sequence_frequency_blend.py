"""Independent CPU audit of strong-frequency ensemble blending and intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

from audit_sequence_calibration import independent_optimum, key_hash
from audit_sequence_robustness import sha, scores


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def paired_again(y, p, q, games):
    """Separate vectorized cluster resampling, not the experiment's bootstrap helper."""
    left, right = scores(y, p), scores(y, q)
    unique, inverse = np.unique(games, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(42).integers(0, len(unique), size=(2000, len(unique)))
    result = {}
    for metric in left:
        delta = left[metric]-right[metric]
        totals = np.bincount(inverse, weights=delta)
        bootstrap = totals[draws].sum(1)/counts[draws].sum(1)
        result[metric] = {'mean_model': float(left[metric].mean()), 'mean_reference': float(right[metric].mean()),
                          'model_minus_reference': float(delta.mean()),
                          'game_only_bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist()}
    return result


def assert_comparison(actual, reported):
    for metric in actual:
        for key, value in actual[metric].items():
            np.testing.assert_allclose(value, reported['metrics'][metric][key], atol=1e-10, rtol=1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend', required=True, type=Path)
    args = parser.parse_args()
    folder = args.blend.resolve()
    config, selection, result = read(folder/'config.json'), read(folder/'selection.json'), read(folder/'results.json')
    runtime = read(folder/'runtime.json')
    assert runtime['neural_training'] is False and runtime['neural_inference'] is False
    for relative, expected in config['source_hashes'].items():
        assert sha(folder/'source'/relative) == expected
    for path, expected in config['reference_hashes'].items():
        assert sha(Path(path)) == expected
    assert sha(folder/'predictions.npz') == result['predictions_sha256']
    calibration, frequency, base = (Path(config[key]) for key in ['calibration_run', 'frequency_run', 'base_run'])
    root_audit = read(calibration/'independent_audit/audit.json')
    assert root_audit['status'] == 'passed'
    original_selection = read(frequency/'selection.json')
    assert selection['original4000_selection'] == original_selection
    chosen = min(original_selection['baselines'], key=lambda name: original_selection['baselines'][name]['calibrated_log_loss'])
    assert chosen == config['selected_frequency_baseline'] == selection['baseline_chosen_on_original4000']
    assert selection['dev_selection'] is False
    dev, cal = arrays(folder/'predictions.npz'), arrays(folder/'calibration_predictions.npz')
    root_dev, root_cal, freq = (arrays(path) for path in [calibration/'predictions.npz', calibration/'calibration_predictions.npz', frequency/'heldout_predictions.npz'])
    for name in ['y', 'pitch_keys', 'game_pk']:
        np.testing.assert_array_equal(dev[name], root_dev[name])
        np.testing.assert_array_equal(dev[name], freq[name])
        np.testing.assert_array_equal(cal[name], root_cal[name])
    assert len(cal['y']) == 12000 and len(dev['y']) == 7276
    assert key_hash(cal['pitch_keys']) == selection['blend_calibration_rows_hash']
    assert selection['temperature_rows_hash'] == read(frequency/'data.json')['calibration_rows_hash']
    np.testing.assert_array_equal(cal['ensemble'], root_cal['ensemble'])
    np.testing.assert_array_equal(dev['ensemble'], root_dev['ensemble'])
    np.testing.assert_array_equal(dev['selected_frequency_baseline'], freq[chosen+'__tempered'])
    for name, original in [('old_count_baseline', 'count_hand'), ('count_blend_log_loss', 'blend_log_loss'),
                           ('count_blend_brier_multiclass', 'blend_brier_multiclass')]:
        np.testing.assert_array_equal(dev[name], root_dev[original])
    # Independently restore CAL predictions using saved tables and pre-pitch keys.
    processed = base.parent.parent/'processed/pitches.parquet'
    assert sha(processed) == read(frequency/'config.json')['processed_sha256']
    key_columns = ['game_pk', 'at_bat_number', 'pitch_number']
    count_columns = ['balls', 'strikes', 'stand', 'p_throws']
    type_columns = ['pitch_type', *count_columns]
    pitcher_columns = ['pitcher', *type_columns]
    frame = pd.read_parquet(processed, columns=key_columns+pitcher_columns+['game_date'])
    assert frame.game_date.dt.year.isin([2023, 2024, 2025]).all() and not frame.duplicated(key_columns).any()
    query = frame.set_index(key_columns).reindex(pd.MultiIndex.from_arrays(cal['pitch_keys'].T, names=key_columns))
    assert not query.isna().any().any()
    with (frequency/(chosen+'.pkl')).open('rb') as stream:
        table = pickle.load(stream)
    prediction = np.array([table['count_hand_table'].get(tuple(row), table['global_probability'])
                           for row in query[count_columns].itertuples(index=False, name=None)])
    for table_name, columns in [('type_table', type_columns), ('pitcher_table', pitcher_columns)]:
        if table_name not in table or table[table_name] is None:
            continue
        local = table[table_name].reindex(pd.MultiIndex.from_frame(query[columns])).to_numpy()
        observed = np.isfinite(local).all(1)
        prediction[observed] = local[observed]
    z = np.log(np.clip(prediction, 1e-12, 1.))/config['baseline_temperature']
    exponential = np.exp(z-z.max(1, keepdims=True))
    prediction = exponential/exponential.sum(1, keepdims=True)
    np.testing.assert_allclose(prediction, cal['selected_frequency_baseline'], atol=1e-14, rtol=1e-13)
    report = {'status': 'passed', 'sources_references_keys_verified': True, 'cal_frequency_table_predictions_reconstructed': True,
              'cal_chosen_baseline': chosen, 'calibration_n': len(cal['y']), 'dev_n': len(dev['y']),
              'weights': {}, 'paired_comparisons_checked': 0, 'all_four_frequency_baselines': {},
              'scope': 'Whole-game intervals condition on fitted ensemble, baseline selection, and CAL weights; omit training/selection/weight-fit uncertainty.',
              'limitations': ['Previously inspected DEV and CAL reused for early stopping.',
                             'No claim of Brier dominance over every fixed strong baseline.']}
    y, games = dev['y'], dev['game_pk']
    for objective, item in selection['fitted_blends'].items():
        independent = independent_optimum(cal['y'], cal['ensemble'], cal['selected_frequency_baseline'], objective)
        weight = item['model_weight']
        np.testing.assert_allclose(weight, independent['independent_weight'], atol=2e-5, rtol=0)
        cal_score = scores(cal['y'], weight*cal['ensemble'].astype(float)+(1-weight)*cal['selected_frequency_baseline'])[objective].mean()
        np.testing.assert_allclose(cal_score, item['calibration_score'], atol=1e-10, rtol=0)
        assert cal_score <= min(independent['endpoint_baseline_score'], independent['endpoint_ensemble_score'])+1e-10
        assert cal_score-independent['independent_calibration_score'] < 1e-9
        reconstructed = weight*dev['ensemble'].astype(float)+(1-weight)*dev['selected_frequency_baseline']
        np.testing.assert_array_equal(reconstructed, dev['strong_blend_'+objective])
        report['weights'][objective] = {**independent, 'saved_weight': weight}
    impossible = dev['impossible_dp']
    assert not (impossible & (y == 9)).any()
    for name, reported in result['metrics'].items():
        for metric, per_pitch in scores(y, dev[name]).items():
            np.testing.assert_allclose(per_pitch.mean(), reported[metric], atol=1e-10)
        legal = np.asarray(dev[name], float).copy()
        legal[impossible, 9] = 0
        legal[impossible] /= legal[impossible].sum(1, keepdims=True)
        np.testing.assert_array_equal(legal, dev[name+'__legal'])
        for metric, per_pitch in scores(y, legal).items():
            np.testing.assert_allclose(per_pitch.mean(), result['posthoc_legal']['metrics'][name][metric], atol=1e-10)
    for suffix, section in [('', result), ('__legal', result['posthoc_legal'])]:
        for name, comparisons in section['paired'].items():
            for comparison_name, reported in comparisons.items():
                reference = comparison_name.removeprefix('minus_')
                assert_comparison(paired_again(y, dev[name+suffix], dev[reference+suffix], games), reported)
                report['paired_comparisons_checked'] += 1
    all_baselines = result['all_frequency_baselines_descriptive']['baselines']
    assert set(all_baselines) == set(original_selection['baselines']) and len(all_baselines) == 4
    for name, info in all_baselines.items():
        reviewed = {}
        for suffix, field, regime in [('', 'tempered', 'original'), ('__legal', 'tempered_legal', 'posthoc_legal')]:
            reference = freq[name+'__'+field]
            for metric, per_pitch in scores(y, reference).items():
                np.testing.assert_allclose(per_pitch.mean(), info[field][metric], atol=1e-10)
            for objective in ['log_loss', 'brier_multiclass']:
                blend_name = 'strong_blend_'+objective
                actual = paired_again(y, dev[blend_name+suffix], reference, games)
                assert_comparison(actual, info['strong_blend_comparisons'][blend_name][regime])
                reviewed[blend_name+'/'+regime] = actual
                report['paired_comparisons_checked'] += 1
        report['all_four_frequency_baselines'][name] = reviewed
    destination = folder/'independent_audit'
    destination.mkdir(exist_ok=True)
    for name in ['audit_sequence_frequency_blend.py', 'audit_sequence_calibration.py', 'audit_sequence_robustness.py']:
        shutil.copyfile(Path(__file__).with_name(name), destination/name)
        report.setdefault('audit_source_hashes', {})[name] = sha(destination/name)
    for name in ['config.json', 'selection.json', 'predictions.npz', 'calibration_predictions.npz', 'results.json', 'runtime.json']:
        report.setdefault('audited_artifact_hashes', {})[name] = sha(folder/name)
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'report': str(destination/'audit.json'),
                      'paired_comparisons_checked': report['paired_comparisons_checked'], 'weights': report['weights']}, indent=2))


if __name__ == '__main__':
    main()
