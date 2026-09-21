"""CPU independent exact-pool membership and ragged-logit probability audit."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.model import eligible, outcome_labels
from audit_sequence_frequency_blend import arrays, read
from audit_sequence_robustness import sha, scores
from audit_sequence_integration import unchanged_checkpoint, compare_arrays, verify_comparison

KEY = ['game_pk', 'at_bat_number', 'pitch_number']
TIERS = [('pitch_type', 'p_throws', 'stand'),
         ('pitch_type', 'p_throws', 'stand', 'balls', 'strikes'),
         ('pitcher', 'pitch_type', 'p_throws', 'stand'),
         ('pitcher', 'pitch_type', 'p_throws', 'stand', 'balls', 'strikes')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', required=True, type=Path)
    folder = parser.parse_args().diagnostic.resolve()
    config, data, result, runtime = (read(folder/name) for name in ['config.json', 'data.json', 'results.json', 'runtime.json'])
    assert config['model_seed'] == 42 and config['selection'] == 'none'
    assert config['calibration_inference'] is False and config['temperature_refitting'] is False and config['neural_training'] is False
    assert runtime['integrity_ok'] and runtime['calibration_inference'] is False
    assert runtime['source_hashes_end'] == config['source_hashes'] and runtime['reference_hashes_end'] == config['reference_hashes']
    for path, digest in config['source_hashes'].items():
        assert sha(folder/'source'/path) == digest
    for path, digest in config['reference_hashes'].items():
        assert sha(Path(path)) == digest
    for path, digest in result['artifact_hashes'].items():
        assert sha(folder/path) == digest
    base, integration = Path(config['base_run']), Path(config['integration_run'])
    assert data['data_identity'] == read(integration/'data.json')['data_identity']
    temperature = read(integration/'calibration400.json')['temperature']
    assert config['common_temperature'] == result['common_temperature'] == temperature
    checkpoint = torch.load(config['neural_checkpoint'], map_location='cpu', weights_only=False)
    digest = unchanged_checkpoint(checkpoint, checkpoint)
    assert digest == result['neural_tensor_sha256'] == runtime['neural_tensor_sha256_end']
    saved, exact = arrays(folder/'predictions.npz'), arrays(folder/'exact_dev_logits.npz')
    original = arrays(base/'heldout_predictions.npz')
    for field in ['y', 'pitch_keys', 'game_pk']:
        np.testing.assert_array_equal(saved[field], original[field])
        np.testing.assert_array_equal(exact[field], original[field])
    processed = base.parent.parent/'processed/pitches.parquet'
    assert sha(processed) == data['data_identity']['processed_sha256']
    columns = list(dict.fromkeys(KEY+list(TIERS[-1])+['game_date', 'split', 'description', 'events', 'supported_pa', 'plate_x', 'plate_z']))
    frame = pd.read_parquet(processed, columns=columns).sort_values(['game_date', *KEY], ignore_index=True)
    assert frame.game_date.dt.year.isin([2023, 2024, 2025]).all()
    train = frame[frame.split.eq('train') & eligible(frame)]
    assert train.game_date.between('2023-05-15', '2025-04-30').all()
    assert len(train) == data['full_eligible_train_rows']
    queries = frame.set_index(KEY).reindex(pd.MultiIndex.from_arrays(saved['pitch_keys'].T, names=KEY))
    np.testing.assert_array_equal(outcome_labels(queries), saved['y'])
    assert queries.split.eq('dev').all()
    positions, offsets, tiers = exact['training_frame_positions'], exact['row_offsets'], exact['pool_tier']
    assert offsets[0] == 0 and offsets[-1] == len(positions) == data['exact_candidate_pairs'] == 1330692
    assert len(offsets) == len(saved['y'])+1 and np.all(np.diff(offsets) > 0) and np.all(tiers >= 0)
    assert train.index.is_unique and train.index.get_indexer(positions).min() >= 0
    np.testing.assert_array_equal(frame[KEY].to_numpy()[positions], exact['training_pitch_keys'])
    np.testing.assert_array_equal(1/np.diff(offsets), exact['uniform_weight_per_query'])
    grouped = [train.groupby(list(keys), observed=True).groups for keys in TIERS]
    query_values = queries[list(TIERS[-1])].to_dict('records')
    for i, row in enumerate(query_values):
        expected, level, key = train.index.to_numpy(), -1, ()
        for candidate_level in [3, 2, 1, 0]:
            keys = TIERS[candidate_level]
            candidate_key = tuple(row[name] for name in keys)
            pool = grouped[candidate_level].get(candidate_key, [])
            if len(pool) >= (20 if 'balls' in keys else 50):
                expected, level, key = np.asarray(pool), candidate_level, candidate_key
                break
        assert tiers[i] == level
        np.testing.assert_array_equal(positions[offsets[i]:offsets[i+1]], expected)
        declared = data['pool_keys_per_query'][i]
        assert declared['tier'] == level and declared['keys'] == list(TIERS[level]) and declared['values'] == list(key)
    profile = arrays(Path(config['profile_run'])/'row_costs.npz')
    np.testing.assert_array_equal(profile['dev_pool_rows'], np.diff(offsets))
    np.testing.assert_array_equal(profile['dev_tier'], tiers)
    logits = exact['logits'].astype(float)
    assert logits.shape == (1330692, 10) and np.isfinite(logits).all()
    report = {'status': 'passed', 'exact_membership_pairs_verified': len(positions), 'dev_rows': len(saved['y']),
              'global_fallback_rows': 0, 'temperature': temperature, 'metrics': {}, 'sampled_minus_exact': {},
              'scope': 'Exact uniform empirical TRAIN-pool integration for original T42 at one frozen temperature; not population truth or unseen validation.',
              'limits': 'Sampled logits are not archived; their raw outputs are cross-checked against prior runs and all saved sampled metrics are recomputed.'}
    for regime, divisor in [('raw', 1.), ('fixed_temperature', temperature)]:
        # Explicit segment loop, independent from runner reduceat implementation.
        p = np.stack([softmax(logits[start:end]/divisor, axis=1).mean(0) for start, end in zip(offsets[:-1], offsets[1:])])
        np.testing.assert_allclose(p, saved['exact_'+regime], atol=1e-13, rtol=1e-13)
    original_robustness = Path(read(integration/'config.json')['robustness_run'])
    prior = {'draw25': arrays(original_robustness/'models/seed42/full_transformer/predictions.npz'),
             'draw100': arrays(integration/'models/seed42/full_transformer/predictions.npz'),
             'draw400': arrays(integration/'predictions400.npz')}
    for name in ['draw25', 'draw100', 'draw400', 'exact']:
        for regime in ['raw', 'fixed_temperature']:
            p = saved[name+'_'+regime]
            values = scores(saved['y'], p)
            report['metrics'][name+'/'+regime] = {metric: float(v.mean()) for metric, v in values.items()}
            for metric, v in values.items():
                np.testing.assert_allclose(v.mean(), result['metrics'][name][regime][metric], atol=1e-10)
            if name != 'exact':
                distance = np.abs(p-saved['exact_'+regime]).sum(1)
                expected = result['probability_l1_to_exact'][name][regime]
                np.testing.assert_allclose(distance.mean(), expected['mean'], atol=1e-12)
                for quantile, value in expected['quantiles'].items():
                    np.testing.assert_allclose(np.quantile(distance, float(quantile)), value, atol=1e-12)
        if name != 'exact':
            np.testing.assert_allclose(saved[name+'_raw'], prior[name]['delivery_integrated_uncalibrated'], atol=3e-7, rtol=1e-5)
            actual = compare_arrays(saved['y'], saved[name+'_fixed_temperature'][None], saved['exact_fixed_temperature'][None], saved['game_pk'], [42])
            verify_comparison(actual, result['sampled_minus_exact'][name])
            report['sampled_minus_exact'][name] = actual
    np.testing.assert_allclose(saved['draw400_fixed_temperature'], prior['draw400']['delivery_integrated_calibrated'], atol=3e-7, rtol=1e-5)
    report['probability_l1_to_exact'] = result['probability_l1_to_exact']
    destination = folder/'independent_audit'
    destination.mkdir(exist_ok=True)
    for name in [Path(__file__).name, 'audit_sequence_integration.py', 'audit_sequence_frequency_blend.py', 'audit_sequence_calibration.py', 'audit_sequence_robustness.py']:
        shutil.copyfile(Path(__file__).with_name(name), destination/name)
        report.setdefault('audit_source_hashes', {})[name] = sha(destination/name)
    for name in ['config.json', 'data.json', 'results.json', 'runtime.json', 'exact_dev_logits.npz', 'predictions.npz']:
        report.setdefault('artifact_hashes', {})[name] = sha(folder/name)
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'report': str(destination/'audit.json'), 'metrics': report['metrics']}, indent=2))


if __name__ == '__main__':
    main()
