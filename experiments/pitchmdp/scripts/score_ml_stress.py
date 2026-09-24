"""Score the complete frozen two-model, three-seed, ten-scenario T3 family."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, validate_probabilities
from pitchmdp.matrix_group_metrics import masks
from pitchmdp.matrix_stress import SCENARIOS
from pitchmdp.matrix_stress_metrics import (FAMILY_SIZE, DRAWS, frozen_predictions,
                                          bounded_comparison, combined_status)
from run_ml_stress import config_check, identity, verify, SEEDS
from run_ml_matrix import assert_hashes, check_location
from run_ml_benchmark import read_json, dump, validate_native_runtime
from score_ml_matrix import archive


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    registered = config['registration']['robustness']
    if registered != {'family_size': FAMILY_SIZE, 'draws': DRAWS,
                      'nll_margin': .010, 'brier_margin': .002, 'family_alpha': .05}:
        raise ValueError('Stress bound family differs from preregistration')
    cells = [config['candidate'], config['control']]
    locations = {(scenario, cell, seed): output / 'members' / scenario / cell / f'seed{seed}'
                 for scenario in SCENARIOS for cell in cells for seed in SEEDS}
    if any(not (p / 'prediction_state.json').is_file() for p in locations.values()):
        raise ValueError('Complete 60-member stress family required before scores')
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted stress analysis')
    start = time.perf_counter()
    baseline = archive(output / 'baseline_predictions.npz')
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    keys, y, games = baseline['dev_keys'], baseline['dev_y'], baseline['dev_game_pk']
    if (len(np.unique(keys, axis=0)) != len(keys) or
        not np.array_equal(keys[:, 0], games) or
        not np.array_equal(metadata[KEY].to_numpy(np.int64), keys)):
        raise ValueError('Invalid stress evaluation keys')
    archived = read_json(output / 'parent_analysis.json')
    parent = archive(Path(prep['parent_run']) / 'analysis' / 'panel' / 'predictions.npz')
    if not np.array_equal(parent['keys'], keys) or not np.array_equal(parent['y'], y):
        raise ValueError('Frozen parent analysis rows differ')
    inputs = {str(output / name): hash_file(output / name) for name in
              ('preparation.json', 'baseline_predictions.npz', 'dev_metadata.parquet', 'parent_analysis.json')}
    members, costs = {}, {}
    for (scenario, cell, seed), dest in locations.items():
        state = read_json(dest / 'prediction_state.json')
        if state['identity'] != {'preparation_sha256': hash_file(output / 'preparation.json'),
                                 'cell': cell, 'seed': seed, 'scenario': scenario}:
            raise ValueError('Stress member identity differs')
        assert_hashes(dest, state['artifact_hashes'])
        values = archive(dest / 'predictions.npz')
        for suffix in ('keys', 'y', 'game_pk', 'pitcher'):
            if not np.array_equal(values['dev_' + suffix], baseline['dev_' + suffix]):
                raise ValueError('Stress metadata unpaired: ' + suffix)
        for field in ('dev', 'dev_raw'):
            validate_probabilities(y, values[field])
        members[scenario, cell, seed] = values
        runtime = read_json(dest / 'runtime.json')
        if runtime['no_recalibration'] is not True:
            raise ValueError('Stress calibration must stay frozen')
        if scenario == 'clean' and (runtime['clean_archive_reused'] or
            runtime['clean_maximum_probability_error'] is None or
            runtime['clean_maximum_probability_error'] > 1e-6):
            raise ValueError('Fresh clean equivalence was not verified')
        costs[f'{scenario}/{cell}/{seed}'] = runtime
        for name in ('prediction_state.json', 'predictions.npz', 'runtime.json'):
            inputs[str(dest / name)] = hash_file(dest / name)
    predictions = {(scenario, cell): frozen_predictions([members[scenario, cell, s] for s in SEEDS],
                    baseline, archived['reports'][cell]) for scenario in SCENARIOS for cell in cells}
    clean_errors = {}
    for cell in cells:
        error = float(np.abs(predictions['clean', cell]['primary'] - parent[cell + '_primary']).max())
        if error > 1e-6:
            raise ValueError('Clean archived ensemble/blend equivalence failed')
        clean_errors[cell] = error
    reports, relative, stability = {}, {}, {}
    group_masks = {'overall': np.ones(len(y), dtype=bool), **masks(metadata)}
    for scenario in SCENARIOS:
        reports[scenario] = {cell: {name: prediction_metrics(y, values[name])
                                   for name in ('primary', 'calibrated', 'raw')}
                             for cell in cells for values in [predictions[scenario, cell]]}
        if scenario == 'clean':
            continue
        a, b = (predictions[scenario, c] for c in cells)
        relative[scenario] = {name: bounded_comparison(y, a['primary'], b['primary'], games, mask)
                              for name, mask in group_masks.items()}
        relative[scenario]['seed_nll_deltas'] = [float((pitch_losses(y, x)-pitch_losses(y, z))[:, 0].mean())
                                                 for x, z in zip(a['seed_primary'], b['seed_primary'])]
        stability[scenario] = {cell: bounded_comparison(y, predictions[scenario, cell]['primary'],
                              predictions['clean', cell]['primary'], games, group_masks['overall']) for cell in cells}
    result = {'experiment_id': config['experiment_id'], 'scope': 'Cpanel input stress',
        'candidate': cells[0], 'control': cells[1], 'n': len(y), 'games': len(np.unique(games)),
        'reports': reports, 'relative': relative, 'stability': stability,
        'relative_R': combined_status([v for scenarios in relative.values()
                                      for name, v in scenarios.items() if name != 'seed_nll_deltas']),
        'within_model_stability': {cell: combined_status([v[cell] for v in stability.values()]) for cell in cells},
        'multiplicity': {**registered, 'method': 'One-sided Bonferroni percentile game-bootstrap upper bounds; all missing slots retained',
                         'seed': 20260924, 'definition': '9 scenarios * ((overall+12 groups) + 2 within-model) * 2 losses'},
        'clean_equivalence_maximum_error': clean_errors, 'costs_and_exposure': costs,
        'class_reporting': {'minimum_events': 30, 'eligible': [int((y == i).sum()) >= 30 for i in range(10)]},
        'coverage': prep['coverage'], 'policy_effect': None, 'whole_mlb_robustness': None,
        'limits': ['Hypothetical fixed stress intensities, not an empirical field-error distribution',
            'Candidate selection used earlier Cpanel DEV; no independent confirmation',
            'Conditional bootstrap excludes fit/calibration/selection uncertainty; extreme percentile bounds have Monte Carlo error',
            'Missing groups remain unconfirmed; any measured failure takes precedence',
            'Unknown-pitcher scenario hides encoder/routing features while delivery keeps original identity'],
        'input_hashes': inputs, 'seconds': time.perf_counter()-start,
        'scored_utc': datetime.now(timezone.utc).isoformat(),
        'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in
            (Path(__file__), PROJECT / 'pitchmdp/matrix_stress_metrics.py',
             PROJECT / 'pitchmdp/matrix_group_metrics.py', PROJECT / 'pitchmdp/matrix_metrics.py')}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz', keys=keys, y=y, game_pk=games,
        **{scenario+'_'+cell+'_'+name: array for (scenario, cell), values in predictions.items() for name, array in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print({'relative_R': result['relative_R'], 'stability': result['within_model_stability']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    score(config_check(read_json(a.config)), a.local_config, a.output.resolve())
