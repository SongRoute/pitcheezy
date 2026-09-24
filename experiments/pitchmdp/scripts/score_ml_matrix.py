"""Score only complete registered prediction families, with exact paired keys."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np

from pitchmdp.data import hash_file
from pitchmdp.matrix_data import CELLS, canonical_hash, validate_config
from pitchmdp.matrix_metrics import (prediction_metrics, pitch_losses, validate_probabilities,
    paired_game_comparison, holm_adjust, prediction_decision)
from run_ml_matrix import (read_json, assert_hashes, manifest_identity, verify_prepare,
    member_dir, member_identity, atomic_json, check_location)
from run_sequence_calibration import fit_blend


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def assert_aligned(candidate, reference):
    for part in ('blend', 'dev'):
        keys = reference[part + '_keys']
        if keys.ndim != 2 or keys.shape[1] != 3 or len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError('Repeated or invalid reference pitch keys')
        for suffix in ('keys', 'y', 'game_pk', 'pitcher'):
            name = part + '_' + suffix
            if not np.array_equal(candidate[name], reference[name]):
                raise ValueError('Unpaired prediction metadata: ' + name)
        if not np.array_equal(keys[:, 0], reference[part + '_game_pk']):
            raise ValueError('Pitch key and game metadata disagree')
        for suffix in ('', '_raw'):
            validate_probabilities(reference[part + '_y'], candidate[part + suffix])


def require_complete(output, cells, seeds):
    missing = [f'{cell}/seed{seed}' for cell in cells for seed in seeds
               if not (output / 'members' / cell / f'seed{seed}' / 'prediction_state.json').is_file()]
    if missing:
        raise ValueError('Family incomplete; scores remain unopened: ' + ', '.join(missing))


def summarize_cell(members, baseline):
    y, cy = baseline['dev_y'], baseline['blend_y']
    ensemble = {part: np.mean([m[part] for m in members], axis=0) for part in ('blend', 'dev')}
    selection = fit_blend(cy, ensemble['blend'], baseline['blend'], 'log_loss')
    w = selection['model_weight']
    primary = w * ensemble['dev'] + (1 - w) * baseline['dev']
    seed_primary, seed_reports = [], []
    for member in members:
        chosen = fit_blend(cy, member['blend'], baseline['blend'], 'log_loss')
        sw = chosen['model_weight']
        predicted = sw * member['dev'] + (1 - sw) * baseline['dev']
        seed_primary.append(predicted)
        seed_reports.append({'blend_selection': chosen,
            'raw': prediction_metrics(y, member['dev_raw']),
            'calibrated': prediction_metrics(y, member['dev']),
            'primary': prediction_metrics(y, predicted)})
    raw_ensemble = np.mean([m['dev_raw'] for m in members], axis=0)
    return {'selection': selection, 'raw_ensemble': prediction_metrics(y, raw_ensemble),
            'calibrated_ensemble': prediction_metrics(y, ensemble['dev']),
            'primary': prediction_metrics(y, primary), 'seeds': seed_reports}, {
            'primary': primary, 'calibrated': ensemble['dev'], 'raw': raw_ensemble,
            'seed_primary': np.stack(seed_primary)}


def group_report(baseline, predictions, cohort_ids):
    result = {}
    for pid in cohort_ids:
        mask = baseline['dev_pitcher'] == pid
        n, games = int(mask.sum()), len(np.unique(baseline['dev_game_pk'][mask]))
        result[str(pid)] = {'n': n, 'games': games, 'eligible_for_group_claim': n >= 500 and games >= 30,
            'metrics': {cell: prediction_metrics(baseline['dev_y'][mask], values['primary'][mask])
                        for cell, values in predictions.items()}}
    return result


def legacy_gate(output, config, baseline, reports, members_by_cell, input_hashes):
    root = Path(config['registration']['base_run']) / '2025'
    if not root.is_dir():
        root = Path(config['registration']['base_run']) / 'fold2025'
    old = read_json(root / 'results.json')
    input_hashes[str(root / 'results.json')] = hash_file(root / 'results.json')
    tests = []
    for cell, kind in (('R2-MLP', 'flatten_mlp'), ('R2-TF', 'transformer')):
        for i, seed in enumerate(CELLS[cell][1]):
            path = root / f'{kind}_{seed}_predictions.npz'
            hashes = read_json(root / f'{kind}_{seed}_hashes.json')
            if hash_file(path) != hashes['predictions']:
                raise ValueError('Archived prediction hash differs')
            input_hashes[str(path)] = hash_file(path)
            previous = archive(path)
            for part in ('blend', 'dev'):
                if not np.array_equal(previous[part + '_keys'], baseline[part + '_keys']):
                    raise ValueError('Legacy sample differs from reproduction')
            # Archived float32 marginal sums use the historical audit contract.
            # Do not silently normalize them or claim they passed the new mass check.
            p = np.asarray(previous['dev'], dtype=np.float64)
            y = baseline['dev_y']
            old_nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
            recorded = old['seed_metrics'][kind][i]['log_loss']
            new_nll = reports[cell]['seeds'][i]['calibrated']['log_loss']
            tests.append({'cell': cell, 'seed': seed, 'archived_nll': old_nll,
                'recorded_nll': recorded, 'retrained_nll': new_nll,
                'recomputed_error': abs(old_nll - recorded), 'retraining_difference': new_nll - old_nll,
                'recomputation_pass': abs(old_nll - recorded) <= 1e-8,
                'retraining_pass': abs(new_nll - old_nll) <= .001,
                'maximum_probability_difference': float(np.abs(members_by_cell[cell][i]['dev'] - p).max()),
                'historical_probability_mass_error': float(np.abs(p.sum(1) - 1).max())})
    return {'passed': all(t['recomputation_pass'] and t['retraining_pass'] for t in tests),
            'members': tests, 'claim': 'reproduction only; no candidate superiority claim'}


def score(config, local_path, output, family):
    local = read_json(local_path)
    check_location(local, output)
    prep = verify_prepare(output, manifest_identity(config, local_path))
    cells = ['R2-MLP', 'R2-TF'] if family == 'legacy' else ['D1-25', 'D1-50', 'D1-100']
    seeds = list(CELLS[cells[0]][1])
    require_complete(output, cells, seeds)
    destination = output / 'analysis' / family
    if destination.exists():
        raise ValueError('Analysis output already exists; preserve prior scoring')
    start = time.perf_counter()
    scope = 'legacy' if family == 'legacy' else 'regular'
    baseline_path = output / f'{scope}_baseline_predictions.npz'
    baseline = archive(baseline_path)
    assert_aligned(baseline, baseline)
    inputs = {str(output / 'preparation.json'): hash_file(output / 'preparation.json'),
              str(baseline_path): hash_file(baseline_path)}
    members_by_cell, costs = {}, {}
    # Validate the complete family before computing any scores.
    for cell in cells:
        members_by_cell[cell], costs[cell] = [], []
        for seed in seeds:
            dest = member_dir(output, cell, seed)
            fitstate = read_json(dest / 'fit_state.json')
            if fitstate['identity'] != member_identity(prep, cell, seed):
                raise ValueError('Fit member identity differs')
            assert_hashes(dest, fitstate['artifact_hashes'])
            state = read_json(dest / 'prediction_state.json')
            if state['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
                raise ValueError('Prediction fit identity differs')
            assert_hashes(dest, state['artifact_hashes'])
            path = dest / 'predictions.npz'
            member = archive(path)
            assert_aligned(member, baseline)
            members_by_cell[cell].append(member)
            inputs[str(path)] = hash_file(path)
            costs[cell].append({'seed': seed, 'fit': read_json(dest / 'fit.json'),
                                'prediction': read_json(dest / 'prediction_runtime.json')})
    reports, predictions = {}, {}
    for cell in cells:
        reports[cell], predictions[cell] = summarize_cell(members_by_cell[cell], baseline)
    result = {'family': family, 'cells': cells, 'seeds': seeds,
        'frequency': prediction_metrics(baseline['dev_y'], baseline['dev']),
        'reports': reports, 'costs': costs, 'input_hashes': inputs,
        'config_sha256': canonical_hash(config),
        'scoring_sources': {str(path.relative_to(PROJECT)): hash_file(path) for path in
            [Path(__file__), PROJECT / 'pitchmdp/matrix_metrics.py', PROJECT / 'scripts/run_sequence_calibration.py']},
        'n': len(baseline['dev_y']), 'games': len(np.unique(baseline['dev_game_pk'])),
        'policy_effect': None, 'whole_mlb_robustness': None,
        'limits': ['C6 retrospective eligible DEV only', 'Previously exposed development season',
                   'Bootstrap conditions on fixed fitted and calibrated predictors',
                   'No starter/reliever-wide or causal recommendation claim']}
    cohort_ids = prep['scope']['regular']['cohort_ids']
    result['per_pitcher'] = group_report(baseline, predictions, cohort_ids)
    result['class_reporting'] = {'minimum_events': 30,
        'eligible': [int((baseline['dev_y'] == i).sum()) >= 30 for i in range(10)]}
    if family == 'legacy':
        result['reproduction_gate'] = legacy_gate(output, config, baseline, reports, members_by_cell, inputs)
    else:
        comparisons = []
        for candidate, control in config['registration']['primary_comparisons']:
            comparison = paired_game_comparison(baseline['dev_y'], predictions[candidate]['primary'],
                predictions[control]['primary'], baseline['dev_game_pk'])
            deltas = [float((pitch_losses(baseline['dev_y'], a) - pitch_losses(baseline['dev_y'], b))[:, 0].mean())
                for a, b in zip(predictions[candidate]['seed_primary'], predictions[control]['seed_primary'])]
            comparisons.append({'candidate': candidate, 'control': control,
                                'paired': comparison, 'seed_deltas': deltas})
        adjusted = holm_adjust([c['paired']['nll']['p_less'] for c in comparisons])
        for comparison, p in zip(comparisons, adjusted):
            comparison['decision'] = prediction_decision(comparison['paired'], p, comparison['seed_deltas'])
        result['primary_comparisons'] = comparisons
        result['descriptive_D100_minus_D50'] = paired_game_comparison(baseline['dev_y'],
            predictions['D1-100']['primary'], predictions['D1-50']['primary'], baseline['dev_game_pk'])
    result['seconds'] = time.perf_counter() - start
    result['scored_utc'] = datetime.now(timezone.utc).isoformat()
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')},
        **{cell + '_' + name: value for cell, values in predictions.items() for name, value in values.items()})
    atomic_json(destination / 'results.json', result)
    atomic_json(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print(json.dumps({'output': str(destination), 'nll': {c: r['primary']['log_loss'] for c, r in reports.items()},
                      'reproduction_gate': result.get('reproduction_gate', {}).get('passed')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=['legacy', 'data'], required=True)
    args = parser.parse_args()
    score(validate_config(read_json(args.config)), args.local_config, args.output.resolve(), args.family)


if __name__ == '__main__':
    main()
