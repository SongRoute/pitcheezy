"""Independent CPU audit of fixed-context baseline and 100-draw ensemble blending."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from audit_sequence_calibration import independent_optimum, key_hash
from audit_sequence_frequency_blend import arrays, read, paired_again, assert_comparison
from audit_sequence_robustness import sha, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend', required=True, type=Path)
    folder = parser.parse_args().blend.resolve()
    config, selection, results = (read(folder/name) for name in ['config.json', 'selection.json', 'results.json'])
    assert read(folder/'runtime.json')['no_neural_inference']
    assert config['draws'] == 100 and config['seeds'] == [42, 43, 44, 45, 46]
    assert config['baseline_reselected'] is False and config['baseline_refitted'] is False
    for relative, digest in config['source_hashes'].items():
        assert sha(folder/'source'/relative) == digest
    for path, digest in config['reference_hashes'].items():
        assert sha(Path(path)) == digest
    assert sha(folder/'predictions.npz') == results['predictions_sha256']
    context, ensemble = Path(config['context25_run']), Path(config['calibration100_run'])
    assert read(context/'independent_audit/audit.json')['status'] == 'passed'
    assert read(ensemble.parent/'independent_audit/audit.json')['status'] == 'passed'
    original_selection = read(context/'selection.json')
    assert selection['baseline_selection_preserved'] == original_selection['baseline']
    assert selection['dev_selection'] is False
    chosen = config['fixed_baseline']
    assert chosen == original_selection['baseline']['chosen'] == 'league_context'
    assert config['baseline_temperature'] == original_selection['baseline']['candidates'][chosen]['temperature']
    cal, dev = arrays(folder/'calibration_predictions.npz'), arrays(folder/'predictions.npz')
    old_cal, old_dev = arrays(context/'calibration_predictions.npz'), arrays(context/'predictions.npz')
    new_cal, new_dev = arrays(ensemble/'calibration_predictions.npz'), arrays(ensemble/'predictions.npz')
    temperature = arrays(context/'temperature_predictions.npz')
    for current, old, new in [(cal, old_cal, new_cal), (dev, old_dev, new_dev)]:
        for field in ['y', 'pitch_keys', 'game_pk']:
            np.testing.assert_array_equal(current[field], old[field])
            np.testing.assert_array_equal(current[field], new[field])
        np.testing.assert_array_equal(current['ensemble100'], new['ensemble'])
        np.testing.assert_array_equal(new['ensemble'], new['seed_predictions'].mean(0))
    assert len(cal['y']) == 12000 and len(dev['y']) == 7276 and len(temperature['y']) == 4000
    groups = [set(map(tuple, archive['pitch_keys'])) for archive in [temperature, cal, dev]]
    assert all(groups[i].isdisjoint(groups[j]) for i, j in [(0, 1), (1, 2), (0, 2)])
    assert key_hash(cal['pitch_keys']) == selection['weight_calibration_row_hash']
    assert key_hash(temperature['pitch_keys']) == selection['temperature_row_hash']
    np.testing.assert_array_equal(cal['fixed_context_baseline'], old_cal['selected_baseline'])
    np.testing.assert_array_equal(dev['fixed_context_baseline'], old_dev[chosen+'__tempered'])
    np.testing.assert_array_equal(dev['context25_primary'], old_dev['context_blend_log_loss'])
    np.testing.assert_array_equal(dev['context25_secondary'], old_dev['context_blend_brier_multiclass'])
    np.testing.assert_array_equal(dev['impossible_dp'], old_dev['impossible_dp'])
    report = {'status': 'passed', 'weights': {}, 'paired_comparisons_checked': 0,
              'scope': 'Fixed five-model ensemble, baseline, CAL selection and weights; game intervals omit fitting and numerical integration uncertainty.'}
    for objective, info in selection['weights'].items():
        optimum = independent_optimum(cal['y'], cal['ensemble100'], cal['fixed_context_baseline'], objective)
        weight = info['model_weight']
        np.testing.assert_allclose(weight, optimum['independent_weight'], atol=2e-5, rtol=0)
        score = scores(cal['y'], weight*cal['ensemble100'].astype(float)+(1-weight)*cal['fixed_context_baseline'])[objective].mean()
        np.testing.assert_allclose(score, info['calibration_score'], atol=1e-10)
        assert score-optimum['independent_calibration_score'] < 1e-9
        assert score <= min(optimum['endpoint_baseline_score'], optimum['endpoint_ensemble_score'])+1e-10
        np.testing.assert_array_equal(weight*dev['ensemble100'].astype(float)+(1-weight)*dev['fixed_context_baseline'], dev['context100_blend_'+objective])
        report['weights'][objective] = {**optimum, 'saved_weight': weight}
    impossible = dev['impossible_dp']
    assert not (impossible & (dev['y'] == 9)).any()
    for name, metrics in results['metrics'].items():
        for metric, values in scores(dev['y'], dev[name]).items():
            np.testing.assert_allclose(values.mean(), metrics[metric], atol=1e-10)
        legal = dev[name].astype(float).copy()
        legal[impossible, 9] = 0
        legal[impossible] /= legal[impossible].sum(1, keepdims=True)
        np.testing.assert_array_equal(legal, dev[name+'__legal'])
        for metric, values in scores(dev['y'], legal).items():
            np.testing.assert_allclose(values.mean(), results['posthoc_legal']['metrics'][name][metric], atol=1e-10)
    for suffix, section in [('', results), ('__legal', results['posthoc_legal'])]:
        for name, comparisons in section['paired'].items():
            for reference, expected in comparisons.items():
                reference = reference.removeprefix('minus_')
                actual = paired_again(dev['y'], dev[name+suffix], dev[reference+suffix], dev['game_pk'])
                assert_comparison(actual, expected)
                report['paired_comparisons_checked'] += 1
    for reference in ['fixed_context_baseline', 'context25_primary', 'ensemble100']:
        report.setdefault('primary_comparisons', {})[reference] = paired_again(dev['y'], dev['context100_blend_log_loss'], dev[reference], dev['game_pk'])
    destination = folder/'independent_audit'
    destination.mkdir(exist_ok=True)
    for name in [Path(__file__).name, 'audit_sequence_calibration.py', 'audit_sequence_frequency_blend.py', 'audit_sequence_robustness.py']:
        shutil.copyfile(Path(__file__).with_name(name), destination/name)
        report.setdefault('audit_source_hashes', {})[name] = sha(destination/name)
    for name in ['config.json', 'results.json', 'selection.json', 'predictions.npz', 'calibration_predictions.npz', 'runtime.json']:
        report.setdefault('artifact_hashes', {})[name] = sha(folder/name)
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
