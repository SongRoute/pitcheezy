"""Complete enriched-H5 MLP/Transformer by D25/D100 factorial analysis."""
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
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses
from pitchmdp.matrix_interaction import CELLS, SEEDS, validate_config, member_identity
from pitchmdp.matrix_interaction_metrics import ORDER, factorial_contrasts, interaction_decision
from pitchmdp.matrix_benchmark import CELLS as ARCH_CELLS, member_identity as architecture_member_identity
from run_ml_interaction import identity, verify
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, check_location
from score_ml_matrix import archive, assert_aligned, require_complete, summarize_cell, group_report


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    if config['registration']['interaction_gate'] != {'minimum_absolute_nll_interaction': .003,
         'ci95_excludes_zero': True, 'two_sided_p_max': .05, 'required_same_direction_seeds': 2}:
        raise ValueError('Factorial primary gate differs from registration')
    require_complete(output, CELLS, SEEDS)
    parent = Path(prep['architecture_run'])
    require_complete(parent, ARCH_CELLS, SEEDS)
    parent_prep = read_json(parent / 'preparation.json')
    parent_analysis = parent / 'analysis'
    manifest = read_json(parent_analysis / 'manifest.json')
    for name, suffix in (('results', '.json'), ('predictions', '.npz')):
        if hash_file(parent_analysis / (name+suffix)) != manifest[name+'_sha256']:
            raise ValueError('Parent architecture analysis changed')
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('Parent architecture input changed')
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted factorial analysis')
    started = time.perf_counter()
    baseline = archive(output / 'baseline_predictions.npz')
    assert_aligned(baseline, baseline)
    references = {'MLP25': (output, 'I25-MLP'), 'MLP100': (parent, 'A0-MLP'),
                  'TF25': (output, 'I25-TF'), 'TF100': (parent, 'A6-transformer')}
    inputs = {str(output / name): hash_file(output / name) for name in ('preparation.json', 'baseline_predictions.npz')}
    inputs.update({str(parent_analysis / name): hash_file(parent_analysis / name)
                   for name in ('results.json', 'predictions.npz', 'manifest.json')})
    members, costs = {}, {}
    for label in ORDER:
        root, cell = references[label]
        members[label], costs[label] = [], []
        for seed in SEEDS:
            dest = root / 'members' / cell / f'seed{seed}'
            fit = read_json(dest / 'fit_state.json')
            expected = member_identity(prep, cell, seed) if root == output else architecture_member_identity(parent_prep, cell, seed)
            if fit['identity'] != expected:
                raise ValueError('Factorial fit identity differs')
            assert_hashes(dest, fit['artifact_hashes'])
            pred = read_json(dest / 'prediction_state.json')
            if pred['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
                raise ValueError('Factorial predictions reference changed fit')
            assert_hashes(dest, pred['artifact_hashes'])
            member = archive(dest / 'predictions.npz')
            assert_aligned(member, baseline)
            members[label].append(member)
            costs[label].append({'seed': seed, 'fit': read_json(dest / 'fit.json'),
                                 'prediction': read_json(dest / 'prediction_runtime.json'), 'reused': root == parent})
            for name in ('fit_state.json', 'prediction_state.json', 'fit.json', 'prediction_runtime.json', 'predictions.npz'):
                inputs[str(dest / name)] = hash_file(dest / name)
    reports, predictions = {}, {}
    for label in ORDER:
        reports[label], predictions[label] = summarize_cell(members[label], baseline)
    archived = archive(parent_analysis / 'predictions.npz')
    for name in ('keys', 'y', 'game_pk', 'pitcher'):
        if not np.array_equal(archived[name], baseline['dev_' + name]):
            raise ValueError('Reused D100 analysis metadata is not byte-identical: ' + name)
    for label, cell in (('MLP100', 'A0-MLP'), ('TF100', 'A6-transformer')):
        for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
            if not np.array_equal(predictions[label][kind], archived[cell+'_'+kind]):
                raise ValueError('Reused D100 analysis is not byte-identical')
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    loss = np.stack([pitch_losses(y, predictions[label]['primary']) for label in ORDER], axis=1)
    contrasts = factorial_contrasts(loss, games)
    seed_effects = []
    for seed in SEEDS:
        means = [pitch_losses(y, predictions[label]['seed_primary'][seed])[:, 0].mean() for label in ORDER]
        seed_effects.append(float(means[3]-means[2]-means[1]+means[0]))
    decision = interaction_decision(contrasts['contrasts']['interaction']['nll'], seed_effects)
    cohort = read_json(output / 'parent_preparation.json')['scope']['regular']['cohort_ids']
    result = {'experiment_id': config['experiment_id'], 'scope': 'C6 exposed DEV factorial screen',
        'n': len(y), 'games': len(np.unique(games)), 'frequency': prediction_metrics(y, baseline['dev']),
        'reports': reports, 'factorial': contrasts, 'interaction_decision': decision,
        'primary_family': 'One two-sided NLL interaction; main/simple effects and Brier descriptive',
        'per_pitcher': group_report(baseline, predictions, cohort), 'costs': costs,
        'new_fits': 6, 'reused_fits': 6, 'policy_effect': None, 'whole_mlb_robustness': None,
        'limits': ['Same D100 auxiliary at both result-model data levels',
            'Interaction is for full temperature/June-blend pipelines, not uncalibrated architecture alone',
            'Same epoch/early-stop budgets do not imply equal optimizer updates or wall time',
            'Fixed-prediction conditional bootstrap omits training/calibration/selection uncertainty'],
        'input_hashes': inputs, 'seconds': time.perf_counter()-started,
        'scored_utc': datetime.now(timezone.utc).isoformat(),
        'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in
             (Path(__file__), PROJECT / 'pitchmdp/matrix_interaction_metrics.py',
              PROJECT / 'pitchmdp/matrix_metrics.py', PROJECT / 'scripts/score_ml_matrix.py',
              PROJECT / 'scripts/run_sequence_calibration.py')}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: baseline['dev_'+name] for name in ('keys', 'y', 'game_pk', 'pitcher')},
        **{cell+'_'+name: value for cell, values in predictions.items() for name, value in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print({'interaction': contrasts['contrasts']['interaction']['nll'], 'decision': decision})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    score(validate_config(read_json(a.config)), a.local_config, a.output.resolve())
