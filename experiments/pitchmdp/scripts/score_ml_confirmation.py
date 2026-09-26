"""Complete five-seed G confirmation with unchanged three-seed references."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import numpy as np
import pandas as pd
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_confirmation import SEEDS, validate_config
from pitchmdp.matrix_confirmation_metrics import SCORING, compare_family
from pitchmdp.matrix_metrics import prediction_metrics, paired_game_comparison
from run_ml_confirmation import identity, verify, resolve_member
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, check_location
from score_ml_matrix import archive, assert_aligned, summarize_cell, group_report


def verify_reference_metadata(archived, baseline):
    for name in ('keys', 'y', 'game_pk', 'pitcher'):
        if not np.array_equal(archived[name], baseline['dev_' + name]):
            raise ValueError('Archived G metadata differs: ' + name)


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    if config['registration']['c1_scoring'] != SCORING or prep['primary_comparisons'] != config['primary_comparisons']:
        raise ValueError('C1 fixed scoring family differs')
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve previous or interrupted C1 analysis')
    parent = Path(prep['parent_run'])
    parent_prep = read_json(parent / 'preparation.json')
    assert_hashes(parent, parent_prep['artifact_hashes'])
    previous = parent / 'analysis' / 'panel'
    manifest = read_json(previous / 'manifest.json')
    if hash_file(previous / 'manifest.json') != config['parent_analysis_sha256']:
        raise ValueError('C1 parent panel analysis manifest changed')
    for name, suffix in (('results', '.json'), ('predictions', '.npz')):
        if hash_file(previous / (name + suffix)) != manifest[name + '_sha256']:
            raise ValueError('G analysis archive changed')
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('G analysis input changed')
    started = time.perf_counter()
    baseline = archive(output / 'parent_baseline_predictions.npz')
    assert_aligned(baseline, baseline)
    metadata = pd.read_parquet(parent / 'dev_metadata.parquet')
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), baseline['dev_keys']):
        raise ValueError('C1 group metadata keys differ')
    for name in ('game_pk', 'pitcher'):
        if not np.array_equal(metadata[name].to_numpy(np.int64), baseline['dev_' + name]):
            raise ValueError('C1 group metadata differs: ' + name)
    archived = archive(previous / 'predictions.npz')
    verify_reference_metadata(archived, baseline)
    inputs = {str(output / name): hash_file(output / name)
              for name in ('preparation.json', 'parent_baseline_predictions.npz')}
    inputs[str(parent / 'dev_metadata.parquet')] = hash_file(parent / 'dev_metadata.parquet')
    inputs.update({str(previous / name): hash_file(previous / name)
                   for name in ('manifest.json', 'results.json', 'predictions.npz')})
    members, member_costs, fit_catalog = {}, {}, {}
    # Resolve every requested member and all underlying units before scoring.
    resolved = {(cell, seed): resolve_member(config, output, prep, cell, seed)
                for cell in config['cells'] for seed in SEEDS}
    for cell in config['cells']:
        members[cell], member_costs[cell] = [], []
        for seed in SEEDS:
            reference = resolved[cell, seed]
            inputs.update(reference['hashes'])
            folder = Path(reference['directory'])
            state = read_json(Path(reference['state_path']))
            member = archive(Path(reference['predictions_path']))
            assert_aligned(member, baseline)
            members[cell].append(member)
            inputs[reference['state_path']] = hash_file(Path(reference['state_path']))
            inputs.update({str(folder / name): digest for name, digest in state['artifact_hashes'].items()})
            inputs.update(state['dependencies'])
            fit_paths = sorted({str(Path(path).parent / 'fit.json') for path in state['dependencies']
                                if 'fits' in Path(path).parts})
            if not fit_paths:
                raise ValueError('C1 calibrated member lacks explicit fit costs')
            for path in fit_paths:
                fit_catalog[path] = {'reused': reference['reused'], 'fit': read_json(Path(path))}
                inputs[path] = hash_file(Path(path))
            runtime = read_json(folder / 'prediction_runtime.json')
            member_costs[cell].append({'seed': seed, 'reused': reference['reused'],
                'source_run': reference['source_run'], 'fit_paths': fit_paths, 'prediction': runtime})
    reports, predictions, l6, saved = {}, {}, {}, {}
    for cell in config['cells']:
        by_size = {}
        for size in (1, 3, 5):
            report, probability = summarize_cell(members[cell][:size], baseline)
            by_size[str(size)] = {'report': report, 'predictions': probability}
            saved[f'{cell}_L6_{size}_primary'] = probability['primary']
        for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
            if not np.array_equal(by_size['3']['predictions'][kind], archived[cell + '_' + kind]):
                raise ValueError('Three-seed G reference reconstruction changed')
        reports[cell], predictions[cell] = by_size['5']['report'], by_size['5']['predictions']
        l6[cell] = {'status': 'descriptive_only_no_independent_confirmation',
            'fixed_seed_sets': {'1': [0], '3': [0, 1, 2], '5': list(SEEDS)},
            'ensembles': {size: value['report'] for size, value in by_size.items()},
            'paired_3_minus_1': paired_game_comparison(baseline['dev_y'], by_size['3']['predictions']['primary'],
                by_size['1']['predictions']['primary'], baseline['dev_game_pk']),
            'paired_5_minus_3': paired_game_comparison(baseline['dev_y'], by_size['5']['predictions']['primary'],
                by_size['3']['predictions']['primary'], baseline['dev_game_pk'])}
    comparisons, multiplicity = compare_family(baseline['dev_y'], predictions,
        baseline['dev_game_pk'], metadata, config['primary_comparisons'])
    cost_summary = {}
    for cell, rows in member_costs.items():
        paths = sorted({path for row in rows for path in row['fit_paths']})
        cost_summary[cell] = {'logical_fits': len(paths),
            'logical_fit_seconds': sum(fit_catalog[path]['fit']['seconds_total'] for path in paths),
            'new_fit_seconds': sum(fit_catalog[path]['fit']['seconds_total'] for path in paths if not fit_catalog[path]['reused']),
            'prediction_seconds': sum(row['prediction']['seconds'] for row in rows)}
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    result = {'experiment_id': config['experiment_id'], 'scope': 'Cpanel exposed DEV',
        'selection_status': config['selection_status'], 'cells': config['cells'], 'seeds': list(SEEDS),
        'n': len(y), 'games': len(np.unique(games)), 'reports': reports, 'comparisons': comparisons,
        'multiplicity': multiplicity, 'L6': l6, 'frequency': prediction_metrics(y, baseline['dev']),
        'per_pitcher': group_report(baseline, predictions, prep['panel']['pitcher_ids']),
        'member_costs': member_costs, 'logical_cell_costs': cost_summary, 'fit_catalog': fit_catalog,
        'unique_new_fits': sum(not row['reused'] for row in fit_catalog.values()),
        'unique_reused_fits': sum(row['reused'] for row in fit_catalog.values()),
        'coverage': parent_prep['coverage'], 'policy_effect': None, 'independent_confirmation': None,
        'formal_cost_improvement': None,
        'limits': ['Five seeds do not undo upstream selection or DEV exposure',
            'The same three-seed members and dependencies are reused byte-identically',
            'Five-seed June blend is newly fitted; L6 uses its own June blend for each fixed ensemble size',
            'Baseline-only stability has no candidate superiority test',
            'Conditional game bootstrap excludes fitting/calibration/selection uncertainty',
            'R upper-bound failure is failure to establish noninferiority, not proof of harm'],
        'input_hashes': inputs, 'seconds': time.perf_counter()-started,
        'scored_utc': datetime.now(timezone.utc).isoformat(),
        'scoring_sources': {str(path.relative_to(PROJECT)): hash_file(path) for path in
            (Path(__file__), PROJECT / 'pitchmdp/matrix_confirmation_metrics.py',
             PROJECT / 'pitchmdp/matrix_metrics.py', PROJECT / 'pitchmdp/matrix_group_metrics.py',
             PROJECT / 'scripts/score_ml_matrix.py', PROJECT / 'scripts/run_sequence_calibration.py')}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')}, **saved,
        **{cell + '_' + kind: value for cell, values in predictions.items() for kind, value in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    print({'nll': {cell: report['primary']['log_loss'] for cell, report in reports.items()},
           'selection_status': config['selection_status']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    score(validate_config(read_json(args.config)), args.local_config, args.output.resolve())
