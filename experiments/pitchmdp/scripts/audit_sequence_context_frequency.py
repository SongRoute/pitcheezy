"""CPU-only independent reconstruction of context-frequency artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.model import eligible, outcome_labels
from audit_sequence_calibration import independent_optimum, key_hash
from audit_sequence_frequency_blend import arrays, read, paired_again, assert_comparison
from audit_sequence_robustness import sha, scores

KEY = ['game_pk', 'at_bat_number', 'pitch_number']
COUNT = ['balls', 'strikes', 'stand', 'p_throws']
TYPE = ['pitch_type', *COUNT]
CONTEXT = [*TYPE, 'outs_when_up', 'bases']


def predict(table, frame):
    parent = table.get('type_parent', table)
    p = np.array([parent['count_hand_table'].get(tuple(row), parent['global_probability'])
                  for row in frame[COUNT].itertuples(index=False, name=None)])
    for item, columns in [(parent.get('type_table'), TYPE), (table.get('context_table'), CONTEXT),
                          (table.get('pitcher_context_table'), ['pitcher', *CONTEXT])]:
        if item is not None:
            local = item.reindex(pd.MultiIndex.from_frame(frame[columns])).to_numpy()
            seen = np.isfinite(local).all(1)
            p[seen] = local[seen]
    return p


def check_metrics(y, p, expected):
    for metric, values in scores(y, p).items():
        np.testing.assert_allclose(values.mean(), expected[metric], atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    folder = parser.parse_args().context.resolve()
    config, selection, result = (read(folder/name) for name in ['config.json', 'selection.json', 'results.json'])
    assert read(folder/'runtime.json')['neural_inference'] is False
    for relative, digest in config['source_hashes'].items():
        assert sha(folder/'source'/relative) == digest
    for path, digest in config['reference_hashes'].items():
        assert sha(Path(path)) == digest
    assert sha(folder/'predictions.npz') == result['predictions_sha256']
    frequency, old, calibration = (Path(config[key]) for key in ['frequency_run', 'old_type_blend_run', 'calibration_run'])
    assert read(old/'independent_audit/audit.json')['status'] == 'passed'
    base = Path(read(frequency/'config.json')['base_run'])
    processed = base.parent.parent/'processed/pitches.parquet'
    assert sha(processed) == config['processed_sha256']
    columns = list(dict.fromkeys(KEY+['pitcher']+CONTEXT+['game_date', 'split', 'description', 'events', 'supported_pa', 'plate_x', 'plate_z']))
    frame = pd.read_parquet(processed, columns=columns)
    assert frame.game_date.dt.year.isin([2023, 2024, 2025]).all() and not frame.duplicated(KEY).any()
    train = frame[frame.split.eq('train') & eligible(frame)].copy()
    assert train.game_date.between('2023-05-15', '2025-04-30').all()
    assert key_hash(train[KEY].to_numpy()) == read(folder/'data.json')['full_train_rows_hash']
    train['y'] = outcome_labels(train)
    indexed = frame.set_index(KEY)
    dev, cal, temp = (arrays(folder/name) for name in ['predictions.npz', 'calibration_predictions.npz', 'temperature_predictions.npz'])
    source_dev, source_cal = (arrays(path) for path in [calibration/'predictions.npz', calibration/'calibration_predictions.npz'])
    assert key_hash(temp['pitch_keys']) == read(frequency/'data.json')['calibration_rows_hash']
    queries = {}
    for name, archive, source in [('dev', dev, source_dev), ('cal', cal, source_cal), ('temp', temp, None)]:
        if source is not None:
            for field in ['y', 'pitch_keys', 'game_pk']:
                np.testing.assert_array_equal(archive[field], source[field])
        query = indexed.reindex(pd.MultiIndex.from_arrays(archive['pitch_keys'].T, names=KEY))
        assert query.pitch_type.notna().all()
        np.testing.assert_array_equal(outcome_labels(query), archive['y'])
        queries[name] = query
    assert len(cal['y']) == 12000 and len(temp['y']) == 4000 and len(dev['y']) == 7276
    assert not set(map(tuple, cal['pitch_keys'])) & set(map(tuple, temp['pitch_keys']))
    for archive, source in [(dev, source_dev), (cal, source_cal)]:
        np.testing.assert_array_equal(archive['ensemble'], source['ensemble'])
        np.testing.assert_allclose(source['ensemble'], source['seed_predictions'].mean(0), atol=1e-7)
    old_dev = arrays(old/'predictions.npz')
    for here, there in [('old_type_blend_log_loss', 'strong_blend_log_loss'), ('old_type_blend_brier', 'strong_blend_brier_multiclass')]:
        np.testing.assert_array_equal(dev[here], old_dev[there])
    report = {'status': 'passed', 'training_tables_reconstructed': {}, 'temperature_checks': {}, 'blend_weights': {},
              'paired_comparisons_checked': 0, 'scope': 'Exploratory fixed-ensemble, CAL-selected context-baseline and blend; whole-game intervals omit fitting and model-selection uncertainty.'}
    candidates = selection['baseline']['candidates']
    chosen = min(config['baseline_candidates'], key=lambda name: candidates[name]['calibrated_log_loss'])
    assert chosen == selection['baseline']['chosen']
    report['chosen'] = chosen
    tables = {}
    for name, info in result['baselines'].items():
        path = Path(info['model_path'])
        assert sha(path) == info['model_sha256']
        with path.open('rb') as stream:
            table = pickle.load(stream)
        tables[name] = table
        assert info['training_rows'] == len(train)
        for field, keys in [('context_table', CONTEXT), ('pitcher_context_table', ['pitcher', *CONTEXT])]:
            if table.get(field) is None:
                continue
            counts = train.groupby(keys+['y'], sort=True, dropna=False).size().unstack('y', fill_value=0).reindex(columns=range(10), fill_value=0)
            parent_table = table['type_parent'] if field == 'context_table' else {**table, 'pitcher_context_table': None}
            parent = predict(parent_table, counts.index.to_frame(index=False))
            expected = (counts.to_numpy()+100*parent)/(counts.sum(1).to_numpy()[:, None]+100)
            np.testing.assert_array_equal(table[field].index, counts.index)
            np.testing.assert_array_equal(table[field].to_numpy(), expected)
            report['training_tables_reconstructed'][name+'/'+field] = len(counts)
        temperature = candidates[name]['temperature']
        for split, archive in [('temp', temp), ('dev', dev)]:
            p = predict(table, queries[split])
            np.testing.assert_array_equal(p, archive[name+'__raw'])
            np.testing.assert_array_equal(softmax(np.log(np.clip(p, 1e-12, 1.))/temperature, axis=1), archive[name+'__tempered'])
        logits = np.log(np.clip(temp[name+'__raw'], 1e-12, 1.))
        def derivative(t):
            p = softmax(logits/t, axis=1)
            return float((logits[np.arange(len(p)), temp['y']]-(p*logits).sum(1)).mean()/t**2)
        optimum = .5 if derivative(.5) >= 0 else (2.5 if derivative(2.5) <= 0 else brentq(derivative, .5, 2.5))
        np.testing.assert_allclose(temperature, optimum, atol=1e-5, rtol=0)
        check_metrics(temp['y'], temp[name+'__tempered'], {'log_loss': candidates[name]['calibrated_log_loss'], 'brier_multiclass': scores(temp['y'], temp[name+'__tempered'])['brier_multiclass'].mean()})
        report['temperature_checks'][name] = {'saved': temperature, 'independent_optimum': optimum}
        for regime in ['raw', 'tempered']:
            check_metrics(dev['y'], dev[name+'__'+regime], info['metrics'][regime])
    expected = softmax(np.log(np.clip(predict(tables[chosen], queries['cal']), 1e-12, 1.))/candidates[chosen]['temperature'], axis=1)
    np.testing.assert_array_equal(expected, cal['selected_baseline'])
    for objective, info in selection['blend_weights'].items():
        independent = independent_optimum(cal['y'], cal['ensemble'], cal['selected_baseline'], objective)
        weight = info['model_weight']
        np.testing.assert_allclose(weight, independent['independent_weight'], atol=2e-5, rtol=0)
        score = scores(cal['y'], weight*cal['ensemble'].astype(float)+(1-weight)*cal['selected_baseline'])[objective].mean()
        assert score-independent['independent_calibration_score'] < 1e-9
        assert score <= min(independent['endpoint_baseline_score'], independent['endpoint_ensemble_score'])+1e-10
        key = 'context_blend_'+objective
        np.testing.assert_array_equal(weight*dev['ensemble'].astype(float)+(1-weight)*dev[chosen+'__tempered'], dev[key])
        check_metrics(dev['y'], dev[key], result['blends'][key]['metrics'])
        report['blend_weights'][objective] = {**independent, 'saved_weight': weight}
    impossible = (queries['dev'].outs_when_up.eq(2) | queries['dev'].bases.eq(0)).to_numpy()
    np.testing.assert_array_equal(impossible, dev['impossible_dp'])
    assert not (impossible & (dev['y'] == 9)).any()
    for name, metric in result['posthoc_legal']['metrics'].items():
        p = dev[name].astype(float).copy()
        p[impossible, 9] = 0
        p[impossible] /= p[impossible].sum(1, keepdims=True)
        np.testing.assert_array_equal(p, dev[name+'__legal'])
        check_metrics(dev['y'], p, metric)
    for suffix, section in [('', result), ('__legal', result['posthoc_legal'])]:
        for name, comparisons in section['paired'].items():
            for ref, reported in comparisons.items():
                reference = ref.removeprefix('minus_')
                actual = paired_again(dev['y'], dev[name+suffix], dev[reference+suffix], dev['game_pk'])
                assert_comparison(actual, reported)
                report['paired_comparisons_checked'] += 1
    report['primary_vs_context'] = paired_again(dev['y'], dev['context_blend_log_loss'], dev[chosen+'__tempered'], dev['game_pk'])
    report['primary_vs_context_legal'] = paired_again(dev['y'], dev['context_blend_log_loss__legal'], dev[chosen+'__tempered__legal'], dev['game_pk'])
    for objective in ['log_loss', 'brier_multiclass']:
        delta = scores(dev['y'], dev[chosen+'__tempered'])[objective]-scores(dev['y'], dev['context_blend_log_loss'])[objective]
        dp = dev['y'] == 9
        report.setdefault('double_play_contribution', {})[objective] = {'rows': int(dp.sum()), 'fraction_total_gain': float(delta[dp].sum()/delta.sum()), 'non_dp_mean_gain': float(delta[~dp].mean())}
    destination = folder/'independent_audit'
    destination.mkdir(exist_ok=True)
    for name in [Path(__file__).name, 'audit_sequence_frequency_blend.py', 'audit_sequence_calibration.py', 'audit_sequence_robustness.py']:
        shutil.copyfile(Path(__file__).with_name(name), destination/name)
        report.setdefault('audit_source_hashes', {})[name] = sha(destination/name)
    for name in ['config.json', 'selection.json', 'results.json', 'runtime.json', 'predictions.npz', 'calibration_predictions.npz', 'temperature_predictions.npz']:
        report.setdefault('audited_artifact_hashes', {})[name] = sha(folder/name)
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
