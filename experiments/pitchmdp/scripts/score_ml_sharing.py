"""Score a complete pitcher-sharing family on its TRAIN-selected Cpanel."""
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
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, paired_game_comparison, holm_adjust, prediction_decision
from pitchmdp.matrix_panel import group_reporting_status
from pitchmdp.matrix_group_metrics import guardrails, volume_interaction
from run_ml_matrix import assert_hashes, check_location
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_sharing import config_check, identity, verify, CELLS, SEEDS
from score_ml_matrix import archive, assert_aligned, require_complete, summarize_cell, group_report

COMPARISONS = [('G1-personal', 'G0-global'), ('G2-feature', 'G0-global'),
               ('G3-cluster', 'G2-feature'), ('G4-partial', 'G2-feature')]


def logical_costs(units, fitted):
    modes = {'G0-global': {'global'}, 'G1-personal': {'global', 'personal'},
             'G2-feature': {'global', 'feature'}, 'G3-cluster': {'global', 'cluster'},
             'G4-partial': {'global', 'cluster', 'personal'}}
    result = {}
    for cell, allowed in modes.items():
        selected = [name for name, spec in units.items() if spec['mode'] in allowed]
        result[cell] = {'units_per_seed': selected, 'logical_fits': len(selected)*len(SEEDS),
            'total_fit_seconds': sum(fitted[str(seed)][name]['seconds_total']
                                     for seed in SEEDS for name in selected)}
    return result


def followup_candidates(comparisons, reports, costs):
    eligible = [c['candidate'] for c in comparisons
                if (c['N']['status'] == 'predictive_improvement' or c['G']['status'] == 'group_improvement')
                and c['robustness']['status'] != 'failed']
    return sorted(eligible, key=lambda c: (reports[c]['primary']['log_loss'], costs[c]['total_fit_seconds']))[:2]


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    if config['registration']['primary_comparisons'] != [list(x) for x in COMPARISONS]:
        raise ValueError('G comparison family differs from preregistration')
    if config['registration']['R']['bootstrap_draws'] != 100000:
        raise ValueError('G robust upper bounds require the registered 100000 draws')
    require_complete(output, CELLS, SEEDS)
    destination = output / 'analysis' / 'panel'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted scoring output')
    started = time.perf_counter()
    baseline_path = output / 'baseline_predictions.npz'
    baseline = archive(baseline_path)
    assert_aligned(baseline, baseline)
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), baseline['dev_keys']):
        raise ValueError('Unpaired group metadata')
    inputs = {str(output / name): hash_file(output / name) for name in
              ('preparation.json', 'baseline_predictions.npz', 'dev_metadata.parquet')}
    costs = {}
    for seed in SEEDS:
        costs[str(seed)] = {}
        for unit, spec in prep['units'].items():
            fit_dir = output / 'fits' / f'seed{seed}' / unit
            state = read_json(fit_dir / 'state.json')
            if state['identity'] != {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'unit': unit, 'spec': spec}:
                raise ValueError('Sharing fit-cost identity differs')
            assert_hashes(fit_dir, state['artifact_hashes'])
            costs[str(seed)][unit] = read_json(fit_dir / 'fit.json')
            for name in ('state.json', 'fit.json'):
                inputs[str(fit_dir / name)] = hash_file(fit_dir / name)
    cell_costs = logical_costs(prep['units'], costs)
    members = {}
    for cell in CELLS:
        members[cell] = []
        for seed in SEEDS:
            dest = output / 'members' / cell / f'seed{seed}'
            state = read_json(dest / 'prediction_state.json')
            inputs[str(dest / 'prediction_state.json')] = hash_file(dest / 'prediction_state.json')
            if state['identity'] != {'preparation_sha256': canonical_hash(prep), 'cell': cell, 'seed': seed}:
                raise ValueError('Sharing prediction identity differs')
            assert_hashes(dest, state['artifact_hashes'])
            for name, digest in state['dependencies'].items():
                if hash_file(Path(name)) != digest:
                    raise ValueError('Composite model dependency changed')
                inputs[name] = digest
            member = archive(dest / 'predictions.npz')
            assert_aligned(member, baseline)
            members[cell].append(member)
            for name, digest in state['artifact_hashes'].items():
                inputs[str(dest / name)] = digest
    reports, predictions = {}, {}
    for cell in CELLS:
        reports[cell], predictions[cell] = summarize_cell(members[cell], baseline)
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    low = metadata.train_volume.eq('low').to_numpy(bool)
    low_gate = group_reporting_status(games, low)
    comparisons = []
    p_values = []
    for candidate, control in COMPARISONS:
        a, b = predictions[candidate], predictions[control]
        whole = paired_game_comparison(y, a['primary'], b['primary'], games)
        seed_losses = [(pitch_losses(y, x) - pitch_losses(y, z))[:, 0]
                       for x, z in zip(a['seed_primary'], b['seed_primary'])]
        low_pair = paired_game_comparison(y[low], a['primary'][low], b['primary'][low], games[low]) if low_gate['status'] == 'reporting_eligible' else None
        comparisons.append({'candidate': candidate, 'control': control, 'paired': whole,
            'seed_deltas': [float(d.mean()) for d in seed_losses], 'low_reporting': low_gate,
            'low_paired': low_pair, 'low_seed_deltas': [float(d[low].mean()) for d in seed_losses] if low_pair else None,
            'robustness': guardrails(y, a['primary'], b['primary'], games, metadata),
            'volume_interaction': volume_interaction(y, a['primary'], b['primary'], games, metadata)})
        p_values.extend([whole['nll']['p_less'], low_pair['nll']['p_less'] if low_pair else None])
    adjusted = holm_adjust(p_values)
    for i, c in enumerate(comparisons):
        c['N'] = prediction_decision(c['paired'], adjusted[2*i], c['seed_deltas'])
        if c['low_paired'] is None:
            c['G'] = {'status': 'unmeasured', 'reason': 'Low-volume group reporting threshold not met'}
        else:
            decision = prediction_decision(c['low_paired'], adjusted[2*i+1], c['low_seed_deltas'])
            whole_guard = c['paired']['nll']['ci95'][1] <= .001 and c['paired']['brier']['ci95'][1] <= .001
            c['G'] = {'low_group': decision, 'whole_population_noninferior': whole_guard,
                      'status': 'group_improvement' if decision['status'] == 'predictive_improvement' and whole_guard else 'inconclusive'}
    temporal = {}
    for month in sorted(metadata.month.unique()):
        mask = metadata.month.eq(month).to_numpy(bool)
        temporal[str(month)] = {'n': int(mask.sum()), 'games': len(np.unique(games[mask])),
            'metrics': {cell: prediction_metrics(y[mask], p['primary'][mask]) for cell, p in predictions.items()},
            'paired': {candidate + '_minus_' + control: paired_game_comparison(y[mask],
                predictions[candidate]['primary'][mask], predictions[control]['primary'][mask], games[mask])
                for candidate, control in COMPARISONS}, 'status': 'descriptive development-time slice'}
    result = {'experiment_id': config['experiment_id'], 'scope': 'Cpanel', 'n': len(y),
        'games': len(np.unique(games)), 'panel_size': prep['panel']['panel_size'],
        'frequency': prediction_metrics(y, baseline['dev']), 'reports': reports,
        'comparisons': comparisons, 'multiplicity': {'tests': 8, 'method': 'Holm, overall and low-group hypotheses together', 'adjusted_p': adjusted},
        'per_pitcher': group_report(baseline, predictions, prep['panel']['pitcher_ids']),
        'class_reporting': {'minimum_events': 30, 'eligible': [int((y == i).sum()) >= 30 for i in range(10)]},
        'temporal': temporal, 'fit_units': costs, 'logical_cell_costs': cell_costs,
        'unique_actual_fits': len(prep['units'])*len(SEEDS),
        'followup_candidates': followup_candidates(comparisons, reports, cell_costs),
        'selection_note': 'N/G passes ranked by primary NLL then dependency fit time, maximum two; measured R failure excludes promotion, missing R stays unconfirmed.',
        'coverage': prep['coverage'],
        'individual_eligibility': prep['individual_eligibility'], 'input_hashes': inputs,
        'policy_effect': None, 'whole_mlb_robustness': None, 'independent_confirmation': None,
        'limits': ['TRAIN-selected development panel, no replacement of absent players',
            'Retrospective eligible subset; overlapping exclusion reasons reported separately',
            'Static TRAIN-fitted player representation, not within-TRAIN expanding statistics',
            'G4 is fixed probability shrinkage, not fitted hierarchical Bayesian parameters',
            'Missing R group remains unconfirmed; no unseen-player zero-shot claim',
            'Conditional bootstrap excludes training/calibration/selection uncertainty'],
        'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat(),
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
    print({cell: value['primary']['log_loss'] for cell, value in reports.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    score(config_check(read_json(args.config)), args.local_config, args.output.resolve())


if __name__ == '__main__':
    main()
