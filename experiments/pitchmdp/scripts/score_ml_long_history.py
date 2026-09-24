"""Complete three-arm, nine-member Cpanel long-history family analysis."""
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
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, paired_game_comparison, holm_adjust, prediction_decision
from pitchmdp.matrix_group_metrics import guardrails
from pitchmdp.matrix_panel import group_reporting_status
from pitchmdp.matrix_long_experiment import CELLS, SEEDS
from run_ml_long_history import config_check, identity, verify
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, check_location
from score_ml_matrix import archive, assert_aligned, require_complete, summarize_cell, group_report


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    if config['registration']['primary_candidates'] != ['F4-32', 'F4-128']:
        raise ValueError('Long-history comparison family changed')
    require_complete(output, CELLS, SEEDS)
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve previous long-history analysis')
    started = time.perf_counter()
    baseline = archive(output / 'baseline_predictions.npz')
    assert_aligned(baseline, baseline)
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), baseline['dev_keys']):
        raise ValueError('Long-history grouping keys differ')
    inputs = {str(output / name): hash_file(output / name) for name in
              ('preparation.json', 'baseline_predictions.npz', 'dev_metadata.parquet')}
    members, costs = {}, {}
    for cell in CELLS:
        members[cell], costs[cell] = [], []
        for seed in SEEDS:
            dest = output / 'members' / cell / f'seed{seed}'
            fitted = read_json(dest / 'fit_state.json')
            saved = fitted['identity']
            if (saved['parent_preparation_sha256'] != hash_file(output / 'preparation.json') or
                saved['cell'] != cell or saved['seed'] != seed or saved['budget'] != config['budget']):
                raise ValueError('Long-history fit identity differs')
            for split, record in prep['samples'].items():
                if saved['samples'][split] != {'n': record['n'], 'rows_sha256': record['rows_sha256']}:
                    raise ValueError('Long-history fitted ordered samples differ')
            assert_hashes(dest, fitted['artifact_hashes'])
            pred = read_json(dest / 'prediction_state.json')
            if pred['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
                raise ValueError('Long-history predictions reference changed fit')
            assert_hashes(dest, pred['artifact_hashes'])
            member = archive(dest / 'predictions.npz')
            assert_aligned(member, baseline)
            members[cell].append(member)
            costs[cell].append({'seed': seed, 'fit': read_json(dest / 'fit.json'),
                               'prediction': read_json(dest / 'prediction_runtime.json')})
            for name in ('fit_state.json', 'fit.json', 'prediction_state.json', 'prediction_runtime.json', 'predictions.npz'):
                inputs[str(dest / name)] = hash_file(dest / name)
    reports, predictions = {}, {}
    for cell in CELLS:
        reports[cell], predictions[cell] = summarize_cell(members[cell], baseline)
    control = predictions['F4-H0']
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    low = metadata.train_volume.eq('low').to_numpy(bool)
    gate = group_reporting_status(games, low)
    comparisons, p_values = [], []
    for candidate in ['F4-32', 'F4-128']:
        values = predictions[candidate]
        paired = paired_game_comparison(y, values['primary'], control['primary'], games)
        low_pair = paired_game_comparison(y[low], values['primary'][low], control['primary'][low], games[low]) if gate['status'] == 'reporting_eligible' else None
        losses = [(pitch_losses(y, a) - pitch_losses(y, b))[:, 0]
                  for a, b in zip(values['seed_primary'], control['seed_primary'])]
        comparisons.append({'candidate': candidate, 'control': 'F4-H0', 'paired': paired,
            'seed_deltas': [float(d.mean()) for d in losses], 'low_paired': low_pair, 'low_reporting': gate,
            'low_seed_deltas': [float(d[low].mean()) for d in losses] if low_pair else None,
            'robustness': guardrails(y, values['primary'], control['primary'], games, metadata, candidate_family_size=2)})
        p_values.extend([paired['nll']['p_less'], low_pair['nll']['p_less'] if low_pair else None])
    adjusted = holm_adjust(p_values)
    for i, c in enumerate(comparisons):
        c['N'] = prediction_decision(c['paired'], adjusted[2*i], c['seed_deltas'])
        if c['low_paired'] is None:
            c['G'] = {'status': 'unmeasured', 'reason': 'Low-TRAIN-pitcher reporting threshold not met'}
        else:
            decision = prediction_decision(c['low_paired'], adjusted[2*i+1], c['low_seed_deltas'])
            whole_ok = c['paired']['nll']['ci95'][1] <= .001 and c['paired']['brier']['ci95'][1] <= .001
            c['G'] = {'low_group': decision, 'whole_population_noninferior': whole_ok,
                      'status': 'group_improvement' if decision['status'] == 'predictive_improvement' and whole_ok else 'inconclusive'}
    result = {'experiment_id': config['experiment_id'], 'n': len(y), 'games': len(np.unique(games)),
        'frequency': prediction_metrics(y, baseline['dev']), 'reports': reports, 'comparisons': comparisons,
        'costs': costs, 'coverage': prep['coverage'], 'multiplicity': {'tests': 4, 'method': 'Holm NLL hypotheses, overall and low-TRAIN-pitcher group; missing slots retained', 'p_adjusted': adjusted},
        'per_pitcher': group_report(baseline, predictions, prep['panel']['pitcher_ids']),
        'class_reporting': {'minimum_events': 30, 'eligible': [int((y == i).sum()) >= 30 for i in range(10)]},
        'policy_effect': None, 'whole_mlb_robustness': None, 'independent_confirmation': None,
        'limits': ['Capacity-matched dual-stream MLP adaptation, not faithful paper reproduction',
            'Long stream excludes current PA; same-day other games excluded, prior-date doubleheaders key-ordered',
            'Cpanel retrospective eligible previously exposed DEV; R missinggroups remain unconfirmed',
            'Low-group target is TRAIN pitcher volume, not a claim about rare-batter generalization',
            'Conditional bootstrap omits fitting/calibration/selection uncertainty'],
        'input_hashes': inputs, 'seconds': time.perf_counter() - started,
        'scored_utc': datetime.now(timezone.utc).isoformat(),
        'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in
             (Path(__file__), PROJECT / 'scripts/score_ml_matrix.py', PROJECT / 'pitchmdp/matrix_metrics.py',
              PROJECT / 'pitchmdp/matrix_group_metrics.py')}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')},
        **{cell + '_' + name: value for cell, values in predictions.items() for name, value in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
         'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print({c: r['primary']['log_loss'] for c, r in reports.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    score(config_check(read_json(a.config)), a.local_config, a.output.resolve())


if __name__ == '__main__':
    main()
