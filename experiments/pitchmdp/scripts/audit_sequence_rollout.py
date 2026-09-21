"""Independent CPU reconstruction of archived rollout estimates and randomness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.output.resolve()
    data = json.loads((run/'results.json').read_text())
    config = json.loads((run/'config.json').read_text())
    runtime = json.loads((run/'runtime.json').read_text())
    assert runtime['integrity_ok'] and len(data['cases']) == runtime['cases_completed'] == 12
    assert config == data['config']
    for relative, expected in config['source_hashes'].items():
        assert digest(run/'source'/relative) == expected
    assert digest(run/'SEQUENCE_ROLLOUT_PROTOCOL.md') == config['protocol_sha256']
    for name, expected in config['input_artifact_sha256'].items():
        assert digest(Path(config['run'])/name) == expected
    rows = []
    for case in data['cases']:
        weights = np.asarray(case['policy_weights'])
        np.testing.assert_allclose(weights.sum(), 1.)
        utility = np.asarray([case['terminal_utilities'][key] for key in data['terminal_event_order']])
        streams = {}
        for name, count_key, seed_key in [('selection', 'selection_rollouts_per_action', 'selection_seed'),
                                          ('evaluation', 'evaluation_rollouts_per_action', 'evaluation_seed')]:
            info = case['streams'][name]
            path = run/info['archive']
            assert digest(path) == info['archive_sha256']
            with np.load(path, allow_pickle=False) as loaded:
                streams[name] = {key: loaded[key] for key in loaded.files}
            item = streams[name]
            expected_uniforms = np.random.default_rng(config[seed_key]).random(
                (config['max_pitches'], config[count_key], 3))
            np.testing.assert_array_equal(item['uniforms'], expected_uniforms)
            assert item['lower_returns'].shape == (len(weights), config[count_key])
            assert (item['pitches'] > 0).all() and (item['pitches'] <= config['max_pitches']).all()
            censored = item['censored']
            assert np.array_equal(censored, item['terminal_event_index'] == -1)
            assert (item['lower_returns'][censored] == 0).all()
            np.testing.assert_allclose(item['lower_returns'][~censored], utility[item['terminal_event_index'][~censored]])
            np.testing.assert_allclose(censored.mean(1), case[name+'_censor_fraction'] if name == 'selection' else case['evaluation_action_censor_fraction'])
        selected = int(np.argmax(streams['selection']['lower_returns'].mean(1)))
        assert selected == case['selected_action_index']
        eval_data = streams['evaluation']
        low = eval_data['lower_returns']
        high = low + eval_data['censored']
        paired = low[selected] - weights @ low
        mean = float(paired.mean())
        se = float(paired.std(ddof=1)/np.sqrt(len(paired)))
        np.testing.assert_allclose([mean, se], [case['paired_zero_imputed_contrast']['mean'], case['paired_zero_imputed_contrast']['mc_standard_error']], atol=1e-14)
        np.testing.assert_allclose([mean-1.96*se, mean+1.96*se], case['paired_zero_imputed_contrast']['mc95_normal'], atol=1e-14)
        coefficients = -weights.copy()
        coefficients[selected] += 1
        minimum = np.maximum(coefficients, 0) @ low + np.minimum(coefficients, 0) @ high
        maximum = np.maximum(coefficients, 0) @ high + np.minimum(coefficients, 0) @ low
        np.testing.assert_allclose([minimum.mean(), maximum.mean()], case['paired_censor_contrast_bounds'], atol=1e-14)
        rows.append({'case_id': case['case_id'], 'selected_index': selected,
                     'contrast_we_percentage_points': mean*100,
                     'mc95_we_percentage_points': [(mean-1.96*se)*100, (mean+1.96*se)*100],
                     'selection_censored': int(streams['selection']['censored'].sum()),
                     'evaluation_censored': int(eval_data['censored'].sum())})
    output = run/'independent_audit'
    output.mkdir(exist_ok=True)
    report = {'status': 'passed', 'cases': rows, 'source_hashes_verified': True,
              'separate_selection_evaluation_streams_verified': config['selection_seed'] != config['evaluation_seed'],
              'scope': 'CPU arithmetic and archived provenance only; no causal/model/control validation',
              'auditor_sha256': digest(Path(__file__))}
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copyfile(__file__, output/Path(__file__).name)
    print(json.dumps({'status': report['status'], 'cases': len(rows),
                      'censored': sum(row['selection_censored']+row['evaluation_censored'] for row in rows)}))


if __name__ == '__main__':
    main()
