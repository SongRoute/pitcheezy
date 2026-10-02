"""Frozen Cpanel long-batter-history F4 preparation, feasibility and execution.

The caller enforces 600 seconds per profile command and 7200 seconds per member.
All operations acquire the existing matrix heavy lock. DEV scores remain closed
until the independent scorer validates the complete three-arm/three-seed family.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import pandas as pd

from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_benchmark import select_keys
from pitchmdp.matrix_long_history import MatrixLongHistoryStore, LazyPitchBatch
from pitchmdp.matrix_long_experiment import CELLS, SEEDS, DEFAULT_BUDGET, fit_member, predict_member, _delivery_hash
from pitchmdp.matrix_long_profile import PROFILE_SPEC, profile_batches, resource_projection
from pitchmdp.matrix_sharing import ContinuousPitcherContext
from run_ml_matrix import check_location, heavy_lock, assert_hashes, artifact_hashes
from run_ml_benchmark import read_json, dump, regular_frame, identity as base_identity, validate_native_runtime
from run_ml_sharing import SOURCES as SHARING_SOURCES

SPLITS = ('train', 'earlystop', 'temperature', 'blend', 'dev')
SOURCES = list(dict.fromkeys([*SHARING_SOURCES, 'pitchmdp/matrix_long_history.py',
    'pitchmdp/matrix_lazy_model.py', 'pitchmdp/matrix_long_experiment.py',
    'pitchmdp/matrix_long_profile.py', 'scripts/run_ml_long_history.py']))


def config_check(config):
    required = {'protocol', 'experiment_id', 'parent_run', 'parent_preparation_sha256',
                'seeds', 'width', 'budget', 'draws', 'device', 'profile'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'registration'}:
        raise ValueError('Invalid F4 configuration schema')
    if config['protocol'] != 'ml_long_history_v2' or not isinstance(config['experiment_id'], str) or not config['experiment_id'].strip():
        raise ValueError('Registered F4 protocol and experiment ID required')
    if not isinstance(config['parent_run'], str) or not config['parent_run']:
        raise ValueError('Frozen sharing parent run required')
    digest = config['parent_preparation_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Parent preparation SHA256 required')
    if config['seeds'] != list(SEEDS) or config['width'] != 128 or config['draws'] != 400 or config['budget'] != DEFAULT_BUDGET:
        raise ValueError('F4 fixes all three seeds, max128 capacity, 400 draws, and 30/5/256/.0005 budget')
    if config['profile'] != PROFILE_SPEC:
        raise ValueError('Fresh F4 preparation requires the exact version2 resource-profile registration')
    if config['device'] not in ('auto', 'cpu', 'mps') or ('registration' in config and not isinstance(config['registration'], dict)):
        raise ValueError('Invalid device or registration')
    return config


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    return {**base_identity(config, local_path), 'source_hashes': source_hashes()}


def auxiliary_identity(aux):
    """Pin fitted normalization and the actual frozen delivery vectors."""
    return {'normalizer_sha256': canonical_hash(aux['normalizer'].report()),
            'delivery_sha256': _delivery_hash(aux['delivery'])}


def verify(output, expected):
    preparation = read_json(output / 'preparation.json')
    if preparation['identity'] != expected:
        raise ValueError('F4 source/config/environment differs from preparation')
    assert_hashes(output, preparation['artifact_hashes'])
    return preparation


def _source_frame(local, directory):
    # This is the inherited ML1 derived-data identity, not the sharing model state.
    return regular_frame(local, read_json(directory / 'parent_preparation.json'))


def _load(output, local, preparation, length):
    frame = _source_frame(local, output)
    with (output / 'aux.pkl').open('rb') as stream:
        aux = pickle.load(stream)
    tokens = preparation['features']['h5']
    store = MatrixLongHistoryStore.from_frame(frame, normalizer=aux['normalizer'],
                    type_vocabulary=tokens['type_vocabulary'], long_length=length)
    context = ContinuousPitcherContext(aux['context'], preparation['clusters'])
    if store.base.report() != tokens or context.report() != preparation['features']['context']:
        raise ValueError('Common F4 feature specification changed')
    parts = {name: select_keys(frame, pd.read_parquet(output / record['path']), record)
             for name, record in preparation['samples'].items()}
    batches = {name: LazyPitchBatch(store, context, part.index.to_numpy()) for name, part in parts.items()}
    if aux['delivery'].draws != 400:
        raise ValueError('Frozen delivery budget changed')
    return batches, aux


def prepare(config, local, output, expected):
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        print('LONG_HISTORY_PREPARED', output, flush=True)
        return
    parent = Path(config['parent_run']).resolve()
    check_location(local, parent)
    if output == parent or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError('F4 must use a distinct sibling run')
    if hash_file(parent / 'preparation.json') != config['parent_preparation_sha256']:
        raise ValueError('Sharing parent preparation changed')
    shared = read_json(parent / 'preparation.json')
    assert_hashes(parent, shared['artifact_hashes'])
    for name, expected_hash in shared['identity']['source_hashes'].items():
        if hash_file(PROJECT / name) != expected_hash:
            raise ValueError('Frozen sharing implementation changed: ' + name)
    if shared['identity'].get('config_sha256') is None or 'panel' not in shared or 'clusters' not in shared:
        raise ValueError('Parent must be a registered sharing preparation')
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    dump(output / 'registered_config.json', config)
    for source, destination in [('preparation.json', 'sharing_preparation.json'),
          ('parent_preparation.json', 'parent_preparation.json'), ('aux.pkl', 'aux.pkl'),
          ('baseline_predictions.npz', 'baseline_predictions.npz'), ('panel.json', 'panel.json'),
          ('clusters.json', 'clusters.json'), ('blend_metadata.parquet', 'blend_metadata.parquet'),
          ('dev_metadata.parquet', 'dev_metadata.parquet')]:
        shutil.copyfile(parent / source, output / destination)
    samples = {}
    for name in SPLITS:
        record = shared['samples'][name]
        target = name + '_keys.parquet'
        shutil.copyfile(parent / record['path'], output / target)
        samples[name] = {**record, 'path': target}
    frame = _source_frame(local, output)
    with (output / 'aux.pkl').open('rb') as stream:
        aux = pickle.load(stream)
    store = MatrixLongHistoryStore.from_frame(frame, normalizer=aux['normalizer'],
                    type_vocabulary=shared['features']['tokens']['type_vocabulary'], long_length=128)
    context = ContinuousPitcherContext(aux['context'], shared['clusters'])
    features = {'h5': store.base.report(), 'long_stream': store.report(), 'context': context.report(),
                'adaptation': 'dual-stream MLP with fixed max128 capacity; fresh matched H0 control required',
                'capacity_shared_across_cells': True}
    if features['context'] != shared['features']['context']:
        raise ValueError('F4 continuous context differs from sharing parent')
    for record in samples.values():
        select_keys(frame, pd.read_parquet(output / record['path']), record)
    dump(output / 'features.json', features)
    if source_hashes() != expected['source_hashes']:
        raise ValueError('F4 sources changed during prepare')
    files = [str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()]
    report = {'identity': expected, 'parent_run': str(parent),
              'parent_preparation_sha256': config['parent_preparation_sha256'],
              'samples': samples, 'features': features, 'auxiliary_hashes': auxiliary_identity(aux),
              'clusters': shared['clusters'], 'panel': shared['panel'],
              'coverage': shared['coverage'], 'cells': CELLS, 'seeds': list(SEEDS),
              'artifact_hashes': artifact_hashes(output, files), 'seconds': time.perf_counter() - start,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False,
              'resource_limits': {'profile_command_seconds': 600, 'member_command_seconds': 7200,
                                  'enforcement': 'caller subprocess timeout; common heavy lock inside runner'}}
    dump(output / 'preparation.json', report)
    print('LONG_HISTORY_PREPARED', output, flush=True)


def profile(config, local, output, preparation, cell):
    destination = output / 'profiles' / cell
    expected = {'preparation_sha256': hash_file(output / 'preparation.json'), 'cell': cell,
                'profile_version': PROFILE_SPEC['version']}
    if destination.exists():
        manifest = read_json(destination / 'manifest.json')
        if manifest['identity'] != expected:
            raise ValueError('Existing profile identity changed')
        assert_hashes(destination, manifest['artifact_hashes'])
        print('LONG_HISTORY_PROFILE_COMPLETE', cell, flush=True)
        return
    destination.mkdir(parents=True)
    started = time.perf_counter()
    batches, aux = _load(output, local, preparation, CELLS[cell])
    load_seconds = time.perf_counter() - started
    print('LONG_HISTORY_PROFILE_START', cell, flush=True)
    report = profile_batches(batches, aux['delivery'], device=None if config['device'] == 'auto' else config['device'],
                             width=config['width'], load_seconds=load_seconds)
    report['seconds_total'] = time.perf_counter() - started
    if source_hashes() != preparation['identity']['source_hashes']:
        raise ValueError('F4 sources changed during profile')
    dump(destination / 'profile.json', report)
    dump(destination / 'manifest.json', {'identity': expected, 'artifact_hashes': artifact_hashes(destination, ['profile.json'])})
    print('LONG_HISTORY_PROFILE_COMPLETE', cell, json.dumps(report['projection']), flush=True)


def verify_profile(output, cell):
    destination = output / 'profiles' / cell
    manifest = read_json(destination / 'manifest.json')
    if manifest['identity'] != {'preparation_sha256': hash_file(output / 'preparation.json'), 'cell': cell,
                                'profile_version': PROFILE_SPEC['version']}:
        raise ValueError('A matching feasibility profile is required before a full fit')
    if set(manifest['artifact_hashes']) != {'profile.json'}:
        raise ValueError('Exact version2 profile artifact family required')
    assert_hashes(destination, manifest['artifact_hashes'])
    result = read_json(destination / 'profile.json')
    if result['profile_spec'] != PROFILE_SPEC or result['long_length'] != CELLS[cell]:
        raise ValueError('Profile arm/version/settings differ')
    preparation = read_json(output / 'preparation.json')
    populations = {name: int(preparation['samples'][name]['n']) for name in SPLITS}
    if result['population_counts'] != populations:
        raise ValueError('Profile population sizes differ from preparation')
    if ({name: item['n'] for name, item in result['samples'].items()} != {'train': 65536, 'earlystop': 2048, 'temperature': 16}
        or {name: item['n'] for name, item in result['warmup']['samples'].items()} != {'train': 8192, 'evaluation_train': 2048}
        or result['epochs'] != 4 or result['warmup']['epochs'] != 1 or result['warmup']['optimizer_updates'] != 32):
        raise ValueError('Profile actual sample/epoch/update counts differ')
    computed = resource_projection(warmup_seconds=result['warmup_seconds'], measured_fit_seconds=result['measured_fit_seconds'],
        measured_updates=result['optimizer_updates'], full_train_rows=populations['train'], full_earlystop_rows=populations['earlystop'],
        temperature_seconds=result['temperature_seconds'], temperature_rows=populations['temperature'],
        inference_seconds=result['inference_seconds'], blend_rows=populations['blend'], dev_rows=populations['dev'], load_seconds=result['load_seconds'])
    if computed != result['projection']:
        raise ValueError('Saved resource projection differs from frozen formula')
    if not computed['within_limit']:
        raise ValueError('Local member projection exceeds 7200 seconds; register a resource plan before execution')
    return result


def verify_family_budget(output, budget):
    """All three profiles plus frozen prior-cost ledger gate EVERY full member."""
    required = {'protocol', 'preparation_sha256', 'profile_manifest_sha256', 'owner_budget_ledger',
                'prior_seconds', 'family_seconds'}
    if (set(budget) != required or budget['protocol'] != 'ml_long_family_budget_v2'
        or budget['preparation_sha256'] != hash_file(output / 'preparation.json')
        or set(budget['profile_manifest_sha256']) != set(CELLS) or budget['family_seconds'] != 28800
        or not isinstance(budget['prior_seconds'], (int, float)) or isinstance(budget['prior_seconds'], bool)
        or not math.isfinite(budget['prior_seconds']) or budget['prior_seconds'] <= 0):
        raise ValueError('Exact preparation/all-arm profile/prior-cost/family28800 registration required')
    reports = {}
    for cell in CELLS:
        manifest = output / 'profiles' / cell / 'manifest.json'
        if hash_file(manifest) != budget['profile_manifest_sha256'][cell]:
            raise ValueError('Frozen all-three profile manifest changed')
        reports[cell] = verify_profile(output, cell)
    first = reports[next(iter(CELLS))]
    if any(r['samples'] != first['samples'] or r['warmup']['samples'] != first['warmup']['samples'] for r in reports.values()):
        raise ValueError('All three arms must use identical measured/warmup row hashes')
    record = budget['owner_budget_ledger']
    if set(record) != {'path', 'sha256'} or hash_file(Path(record['path'])) != record['sha256']:
        raise ValueError('Owner prior-budget ledger changed')
    ledger = read_json(Path(record['path']))
    categories = {'preparation', 'profiles', 'cold_failed_attempts', 'equivalence'}
    if (set(ledger) != {'protocol', 'preparation_sha256', 'entries', 'elapsed_seconds_total'}
        or ledger['protocol'] != 'ml_long_owner_budget_ledger_v1' or ledger['preparation_sha256'] != budget['preparation_sha256']
        or {row['category'] for row in ledger['entries']} != categories):
        raise ValueError('Owner ledger must cover preparation/profiles/cold-failed/equivalence costs')
    attempts = set()
    for row in ledger['entries']:
        if (set(row) != {'category', 'attempt', 'seconds', 'evidence'} or not isinstance(row['attempt'], str)
            or not row['attempt'] or row['attempt'] in attempts or not row['evidence']
            or not isinstance(row['seconds'], (int, float)) or isinstance(row['seconds'], bool)
            or not math.isfinite(row['seconds']) or row['seconds'] < 0):
            raise ValueError('Distinct, evidence-backed finite prior-cost entries required')
        attempts.add(row['attempt'])
        for evidence in row['evidence']:
            if set(evidence) != {'path', 'sha256'} or hash_file(Path(evidence['path'])) != evidence['sha256']:
                raise ValueError('Prior-cost process evidence changed')
    prior = math.fsum(row['seconds'] for row in ledger['entries'])
    if (not math.isclose(prior, ledger['elapsed_seconds_total'], rel_tol=0, abs_tol=1e-6)
        or not math.isclose(prior, budget['prior_seconds'], rel_tol=0, abs_tol=1e-6)):
        raise ValueError('Prior seconds do not match owner ledger entries')
    nine = len(SEEDS)*sum(r['projection']['load_fit_temperature_prediction_seconds'] for r in reports.values())
    if prior+nine > budget['family_seconds']:
        raise ValueError('Prior costs plus projected nine-member family exceed28800seconds; explicitly replan')
    frozen = output / 'family_budget.json'
    projection = {'prior_seconds': prior, 'nine_member_projection_seconds': nine,
        'total_projected_seconds': prior+nine, 'family_seconds': budget['family_seconds'],
        'per_arm_member_seconds': {cell: r['projection']['load_fit_temperature_prediction_seconds'] for cell, r in reports.items()},
        'runtime_enforcement': 'Owner subprocess hard timeouts and authoritative cumulative ledger required; projection is not a guarantee'}
    if frozen.exists():
        if read_json(frozen) != budget: raise ValueError('Family budget already frozen; do not overwrite')
        if read_json(output / 'family_budget_projection.json') != projection:
            raise ValueError('Frozen family resource projection changed')
    else:
        projection_path = output / 'family_budget_projection.json'
        if projection_path.exists(): raise ValueError('Incomplete family-budget freeze requires review; preserve it')
        dump(projection_path, projection)
        dump(frozen, budget)
    return reports


def run_member(config, local, output, preparation, cell, seed, command, family_budget):
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError('Unregistered F4 member')
    verify_family_budget(output, family_budget)
    batches, aux = _load(output, local, preparation, CELLS[cell])
    directory = output / 'members' / cell / f'seed{seed}'
    device = None if config['device'] == 'auto' else config['device']
    started = time.perf_counter()
    if command == 'fit':
        fit_member(directory, parent_preparation_sha256=hash_file(output / 'preparation.json'),
                   batches=batches, frozen_delivery=aux['delivery'], cell=cell, seed=seed,
                   budget=config['budget'], width=config['width'], device=device)
    elif command == 'predict':
        predict_member(directory, batches=batches, frozen_delivery=aux['delivery'], device=device)
    else:
        raise ValueError('Unknown member command')
    print('LONG_HISTORY_MEMBER_COMPLETE', command, cell, seed, time.perf_counter() - started, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('status')
    probe = sub.add_parser('profile'); probe.add_argument('--cell', choices=list(CELLS), required=True)
    for name in ('fit', 'predict'):
        command = sub.add_parser(name)
        command.add_argument('--cell', choices=list(CELLS), required=True)
        command.add_argument('--seed', choices=list(SEEDS), type=int, required=True)
        command.add_argument('--family-budget', type=Path, required=True)
    args = parser.parse_args()
    config, local = config_check(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, args.local_config)
    if args.command == 'status':
        verify(output, expected)
        for cell in CELLS:
            for seed in SEEDS:
                member = output / 'members' / cell / f'seed{seed}'
                state = 'predicted' if (member / 'prediction_state.json').exists() else 'fitted' if (member / 'fit_state.json').exists() else 'pending'
                print(cell, seed, state)
        return
    with heavy_lock(root):
        if args.command == 'prepare':
            prepare(config, local, output, expected)
        else:
            prepared = verify(output, expected)
            if args.command == 'profile': profile(config, local, output, prepared, args.cell)
            else: run_member(config, local, output, prepared, args.cell, args.seed, args.command, read_json(args.family_budget))


if __name__ == '__main__':
    main()
