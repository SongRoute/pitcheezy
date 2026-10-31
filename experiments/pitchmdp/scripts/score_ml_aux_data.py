"""Complete restricted-auxiliary follow-up against immutable D1 references."""
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
from pitchmdp.data import hash_file
from pitchmdp.matrix_aux_data import SEEDS, validate_config
from pitchmdp.matrix_metrics import (prediction_metrics, pitch_losses,
    paired_game_comparison, holm_adjust, prediction_decision)
from run_ml_aux_data import identity, verify, member_identity
from run_ml_matrix import assert_hashes, check_location, member_identity as parent_member_identity
from run_sequence_pilot import dump
from run_ml_benchmark import read_json, validate_native_runtime
from score_ml_matrix import archive, assert_aligned, require_complete, summarize_cell, group_report

CELLS = ('D2-25', 'D1-25', 'D1-100')
COMPARISONS = [['D1-100', 'D2-25'], ['D1-25', 'D2-25']]


def verify_archived_reference(archived, baseline, predictions):
    for name in ('keys', 'y', 'game_pk', 'pitcher'):
        if not np.array_equal(archived[name], baseline['dev_' + name]):
            raise ValueError('D1 archived analysis metadata differs: ' + name)
    for cell in ('D1-25', 'D1-100'):
        for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
            if not np.array_equal(archived[cell + '_' + kind], predictions[cell][kind]):
                raise ValueError('D1 reference analysis is not byte-identical')


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    registration = config['registration']
    if (registration['primary_comparisons'] != COMPARISONS or
            registration['bootstrap'] != {'draws': 10000, 'seed': 20260924,
                'unit': 'whole game', 'estimand': 'pitch-weighted paired mean loss'} or
            registration['N'] != {'delta_nll_max': -.003, 'nll_ci95_upper_max': 0,
                'brier_ci95_upper_max': .001, 'required_negative_seeds': 2}):
        raise ValueError('D2 scoring family differs from registration')
    parent = Path(prep['parent_run'])
    parent_prep = read_json(parent / 'preparation.json')
    require_complete(output, ['D2-25'], SEEDS)
    require_complete(parent, ['D1-25', 'D1-50', 'D1-100'], SEEDS)
    analysis = parent / 'analysis' / 'data'
    manifest = read_json(analysis / 'manifest.json')
    for name, suffix in (('results', '.json'), ('predictions', '.npz')):
        if hash_file(analysis / (name + suffix)) != manifest[name + '_sha256']:
            raise ValueError('D1 completed analysis changed')
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('D1 analysis input changed')
    assert_hashes(parent, parent_prep['artifact_hashes'])
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve prior or interrupted D2 analysis')
    started = time.perf_counter()
    restricted = archive(output / 'baseline_predictions.npz')
    full_aux = archive(parent / 'regular_baseline_predictions.npz')
    assert_aligned(restricted, full_aux)
    inputs = {str(output / name): hash_file(output / name)
              for name in ('preparation.json', 'baseline_predictions.npz')}
    inputs.update({str(analysis / name): hash_file(analysis / name)
                   for name in ('manifest.json', 'results.json', 'predictions.npz')})
    inputs[str(parent / 'regular_baseline_predictions.npz')] = hash_file(parent / 'regular_baseline_predictions.npz')
    members, costs = {}, {}
    for cell in CELLS:
        root = output if cell == 'D2-25' else parent
        members[cell], costs[cell] = [], []
        for seed in SEEDS:
            dest = root / 'members' / cell / f'seed{seed}'
            fit = read_json(dest / 'fit_state.json')
            expected = member_identity(prep, seed) if root == output else parent_member_identity(parent_prep, cell, seed)
            if fit['identity'] != expected:
                raise ValueError('D2 family fit identity differs')
            assert_hashes(dest, fit['artifact_hashes'])
            predicted = read_json(dest / 'prediction_state.json')
            if predicted['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
                raise ValueError('D2 family prediction references changed fit')
            assert_hashes(dest, predicted['artifact_hashes'])
            member = archive(dest / 'predictions.npz')
            assert_aligned(member, restricted)
            members[cell].append(member)
            costs[cell].append({'seed': seed, 'fit': read_json(dest / 'fit.json'),
                'prediction': read_json(dest / 'prediction_runtime.json'), 'reused': root == parent})
            for name in ('fit_state.json', 'prediction_state.json', 'fit.json', 'prediction_runtime.json', 'predictions.npz'):
                inputs[str(dest / name)] = hash_file(dest / name)
    reports, predictions = {}, {}
    for cell in CELLS:
        # Baseline quantity is part of the intervention: never reblend a D1
        # reference with D2's newly fitted restricted frequency probabilities.
        baseline = restricted if cell == 'D2-25' else full_aux
        reports[cell], predictions[cell] = summarize_cell(members[cell], baseline)
    verify_archived_reference(archive(analysis / 'predictions.npz'), full_aux, predictions)
    y, games = restricted['dev_y'], restricted['dev_game_pk']
    comparisons = []
    for candidate, control in COMPARISONS:
        paired = paired_game_comparison(y, predictions[candidate]['primary'], predictions[control]['primary'], games)
        seed_deltas = [float((pitch_losses(y, a) - pitch_losses(y, b))[:, 0].mean())
            for a, b in zip(predictions[candidate]['seed_primary'], predictions[control]['seed_primary'])]
        comparisons.append({'candidate': candidate, 'control': control, 'paired': paired, 'seed_deltas': seed_deltas})
    for comparison, adjusted in zip(comparisons, holm_adjust([c['paired']['nll']['p_less'] for c in comparisons])):
        comparison['decision'] = prediction_decision(comparison['paired'], adjusted, comparison['seed_deltas'])
    result = {'experiment_id': config['experiment_id'], 'cells': list(CELLS), 'seeds': list(SEEDS),
        'n': len(y), 'games': len(np.unique(games)), 'reports': reports,
        'primary_comparisons': comparisons, 'costs': costs, 'new_fits': 3, 'reused_fits': 6,
        'frequency': {'restricted_aux': prediction_metrics(y, restricted['dev']),
                      'full_aux': prediction_metrics(y, full_aux['dev'])},
        'per_pitcher': group_report(restricted, predictions, parent_prep['scope']['regular']['cohort_ids']),
        'policy_effect': None, 'whole_mlb_robustness': None,
        'limits': ['Adaptive follow-up on exposed C6 DEV after D1 results',
            'Auxiliary intervention jointly changes normalization, dated histories, context, frequency and delivery',
            'Pre-TRAIN history and later observed dates remain available in both arms',
            'Same epoch/early-stop rule does not equalize updates or compute',
            'Fixed-prediction bootstrap excludes fit/calibration/selection uncertainty'],
        'input_hashes': inputs, 'seconds': time.perf_counter()-started,
        'scored_utc': datetime.now(timezone.utc).isoformat(),
        'scoring_sources': {str(path.relative_to(PROJECT)): hash_file(path) for path in
            (Path(__file__), PROJECT / 'pitchmdp/matrix_metrics.py', PROJECT / 'scripts/score_ml_matrix.py',
             PROJECT / 'scripts/run_sequence_calibration.py')}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: restricted['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')},
        **{cell + '_' + kind: values for cell, kinds in predictions.items() for kind, values in kinds.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print({'nll': {cell: report['primary']['log_loss'] for cell, report in reports.items()},
           'decisions': [comparison['decision']['status'] for comparison in comparisons]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    score(validate_config(read_json(args.config)), args.local_config, args.output.resolve())
