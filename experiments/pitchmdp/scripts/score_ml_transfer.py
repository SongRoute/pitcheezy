"""Evaluate one frozen G comparison on all registered eligible MLB DEV pitches."""
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
from pitchmdp.matrix_metrics import (prediction_metrics, pitch_losses, validate_probabilities,
    paired_game_comparison, holm_adjust, prediction_decision)
from pitchmdp.matrix_group_metrics import masks, bootstrap_difference
from pitchmdp.matrix_panel import group_reporting_status
from pitchmdp.matrix_stress_metrics import frozen_predictions, combined_status
from run_ml_transfer import config_check, identity, verify, SEEDS
from run_ml_matrix import assert_hashes, check_location
from run_ml_benchmark import read_json, dump, validate_native_runtime
from score_ml_matrix import archive, require_complete, group_report


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    cells = [config['candidate'], config['control']]
    if config['registration']['R'] != {'family_size': 24, 'draws': 100000,
                                      'nll_margin': .010, 'brier_margin': .002, 'family_alpha': .05}:
        raise ValueError('Transfer robustness family differs')
    require_complete(output, cells, SEEDS)
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted transfer analysis')
    start = time.perf_counter()
    saved = archive(output / 'baseline_predictions.npz')
    baseline = {'dev'+name[len('mlb_dev'):]: value for name, value in saved.items() if name.startswith('mlb_dev')}
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    keys, y, games = baseline['dev_keys'], baseline['dev_y'], baseline['dev_game_pk']
    if (len(np.unique(keys, axis=0)) != len(keys) or not np.array_equal(keys[:, 0], games) or
        not np.array_equal(metadata[KEY].to_numpy(np.int64), keys)):
        raise ValueError('Whole-MLB metadata keys differ')
    archived = read_json(output / 'parent_analysis.json')
    inputs = {str(output / name): hash_file(output / name) for name in
              ('preparation.json', 'baseline_predictions.npz', 'dev_metadata.parquet', 'parent_analysis.json')}
    members, costs = {}, {}
    for cell in cells:
        members[cell], costs[cell] = [], []
        for seed in SEEDS:
            dest = output / 'members' / cell / f'seed{seed}'
            state = read_json(dest / 'prediction_state.json')
            if state['identity'] != {'preparation_sha256': hash_file(output / 'preparation.json'), 'cell': cell, 'seed': seed}:
                raise ValueError('Transfer member identity differs')
            assert_hashes(dest, state['artifact_hashes'])
            member = archive(dest / 'predictions.npz')
            for suffix in ('keys', 'y', 'game_pk', 'pitcher'):
                if not np.array_equal(member['dev_'+suffix], baseline['dev_'+suffix]):
                    raise ValueError('Whole-MLB member unpaired: '+suffix)
            for name in ('dev', 'dev_raw'):
                validate_probabilities(y, member[name])
            parent_state = read_json(dest / 'parent_prediction_state.json')
            parent_prep = read_json(output / 'parent_preparation.json')
            from pitchmdp.matrix_data import canonical_hash
            if parent_state['identity'] != {'preparation_sha256': canonical_hash(parent_prep), 'cell': cell, 'seed': seed}:
                raise ValueError('Transfer parent member identity differs')
            for path, digest in parent_state['dependencies'].items():
                if hash_file(Path(path)) != digest:
                    raise ValueError('Transfer model dependency changed')
                inputs[path] = digest
            members[cell].append(member)
            costs[cell].append(read_json(dest / 'prediction_runtime.json'))
            inputs[str(dest / 'prediction_state.json')] = hash_file(dest / 'prediction_state.json')
            for name, digest in state['artifact_hashes'].items():
                inputs[str(dest / name)] = digest
    predictions = {cell: frozen_predictions(members[cell], baseline, archived['reports'][cell]) for cell in cells}
    candidate, control = (predictions[c] for c in cells)
    pair = paired_game_comparison(y, candidate['primary'], control['primary'], games)
    seed_loss = [(pitch_losses(y, a)-pitch_losses(y, b))[:, 0]
                 for a, b in zip(candidate['seed_primary'], control['seed_primary'])]
    low = metadata.train_volume.eq('low').to_numpy(bool)
    low_gate = group_reporting_status(games, low)
    low_pair = paired_game_comparison(y[low], candidate['primary'][low], control['primary'][low], games[low]) if low_gate['status'] == 'reporting_eligible' else None
    adjusted = holm_adjust([pair['nll']['p_less'], low_pair['nll']['p_less'] if low_pair else None])
    N = prediction_decision(pair, adjusted[0], [float(v.mean()) for v in seed_loss])
    G = {'status': 'unmeasured', 'reporting': low_gate}
    if low_pair:
        low_N = prediction_decision(low_pair, adjusted[1], [float(v[low].mean()) for v in seed_loss])
        whole_ok = pair['nll']['ci95'][1] <= .001 and pair['brier']['ci95'][1] <= .001
        G.update(status='group_improvement' if low_N['status'] == 'predictive_improvement' and whole_ok else 'inconclusive',
                 low_group=low_N, paired=low_pair, whole_population_noninferior=whole_ok)
    loss = pitch_losses(y, candidate['primary'])-pitch_losses(y, control['primary'])
    robust = {}
    for name, mask in masks(metadata).items():
        gate = group_reporting_status(games, mask)
        measured = bootstrap_difference(loss[mask], games[mask], draws=100000, upper_alpha=.05/24) if gate['status'] == 'reporting_eligible' else None
        robust[name] = {'reporting': gate, 'paired': measured,
                       'passed': bool(measured['simultaneous_upper'][0] <= .010 and measured['simultaneous_upper'][1] <= .002) if measured else None}
    reports = {cell: {name: prediction_metrics(y, value[name]) for name in ('primary', 'calibrated', 'raw')}
               for cell, value in predictions.items()}
    slices = {name: metadata[column].eq(value).to_numpy(bool) for name, column, value in
        [('cpanel', 'in_cpanel', True), ('outside_cpanel', 'in_cpanel', False),
         ('seen_pitcher', 'seen_pitcher', True), ('unseen_pitcher', 'seen_pitcher', False),
         ('seen_batter', 'seen_batter', True), ('unseen_batter', 'seen_batter', False)]}
    slices.update({'month_'+str(month): metadata.month.eq(month).to_numpy(bool) for month in sorted(metadata.month.unique())})
    sliced = {name: {'n': int(mask.sum()), 'games': len(np.unique(games[mask])),
                    'metrics': {c: prediction_metrics(y[mask], v['primary'][mask]) for c, v in predictions.items()},
                    'status': 'descriptive'} for name, mask in slices.items()}
    players = group_report(baseline, predictions, sorted(metadata.pitcher.unique()))
    macro = {cell: {metric: float(np.mean([p['metrics'][cell][metric] for p in players.values()]))
                    for metric in ('log_loss', 'brier_multiclass')} for cell in cells}
    result = {'experiment_id': config['experiment_id'], 'scope': 'Cmlb previously exposed eligible DEV',
        'selection': prep['selection'], 'n': len(y), 'games': len(np.unique(games)),
        'frequency': prediction_metrics(y, baseline['dev']), 'reports': reports,
        'paired': pair, 'seed_nll_deltas': [float(v.mean()) for v in seed_loss], 'N': N, 'G': G,
        'R': {'status': combined_status(list(robust.values())), 'groups': robust,
              'family_size': 24, 'draws': 100000, 'family_alpha': .05},
        'multiplicity': {'primary_nll_tests': 2, 'method': 'Holm; missing low-group slot retained', 'adjusted_p': adjusted},
        'slices': sliced, 'per_pitcher': players, 'pitcher_macro': macro,
        'coverage': prep['coverage'], 'individual_eligibility': prep['individual_eligibility'],
        'costs': costs, 'class_reporting': {'minimum_events': 30, 'eligible': [int((y == i).sum()) >= 30 for i in range(10)]},
        'policy_effect': None, 'independent_confirmation': None,
        'limits': ['Includes selection Cpanel and exposed2025DEV; not an independent confirmation',
            'No fullMLB recalibration or refitting; only existing panel personal experts are available',
            'New-player histories are online observed context, not strict entity zero-shot',
            'Retrospective eligibility differs from pre-pitch availability',
            'Conditional bootstrap omits fit/calibration/selection uncertainty; R missing groups remain unconfirmed'],
        'input_hashes': inputs, 'seconds': time.perf_counter()-start, 'scored_utc': datetime.now(timezone.utc).isoformat(),
        'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in
             (Path(__file__), PROJECT / 'pitchmdp/matrix_stress_metrics.py', PROJECT / 'pitchmdp/matrix_group_metrics.py',
              PROJECT / 'pitchmdp/matrix_metrics.py', PROJECT / 'scripts/score_ml_matrix.py')}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz', keys=keys, y=y, game_pk=games, pitcher=baseline['dev_pitcher'],
        **{cell+'_'+name: value for cell, values in predictions.items() for name, value in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print({'N': N['status'], 'G': G['status'], 'R': result['R']['status']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    score(config_check(read_json(a.config)), a.local_config, a.output.resolve())
