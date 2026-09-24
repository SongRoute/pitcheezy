"""Frozen Cpanel long-batter-history F4 preparation, feasibility and execution.

The caller enforces 600 seconds per profile command and 7200 seconds per member.
All operations acquire the existing matrix heavy lock. DEV scores remain closed
until the independent scorer validates the complete three-arm/three-seed family.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from pitchmdp.matrix_long_profile import profile_batches
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
                'seeds', 'width', 'budget', 'draws', 'device'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'registration'}:
        raise ValueError('Invalid F4 configuration schema')
    if config['protocol'] != 'ml_long_history_v1' or not isinstance(config['experiment_id'], str) or not config['experiment_id'].strip():
        raise ValueError('Registered F4 protocol and experiment ID required')
    if not isinstance(config['parent_run'], str) or not config['parent_run']:
        raise ValueError('Frozen sharing parent run required')
    digest = config['parent_preparation_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Parent preparation SHA256 required')
    if config['seeds'] != list(SEEDS) or config['width'] != 128 or config['draws'] != 400 or config['budget'] != DEFAULT_BUDGET:
        raise ValueError('F4 fixes all three seeds, max128 capacity, 400 draws, and 30/5/256/.0005 budget')
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
    expected = {'preparation_sha256': hash_file(output / 'preparation.json'), 'cell': cell}
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
                             width=config['width'])
    report['load_seconds'] = load_seconds
    report['seconds_total'] = time.perf_counter() - started
    report['projection']['load_fit_temperature_prediction_seconds'] = load_seconds + report['projection']['fit_temperature_prediction_seconds']
    report['projection']['within_limit'] = report['projection']['load_fit_temperature_prediction_seconds'] <= 7200
    if source_hashes() != preparation['identity']['source_hashes']:
        raise ValueError('F4 sources changed during profile')
    dump(destination / 'profile.json', report)
    dump(destination / 'manifest.json', {'identity': expected, 'artifact_hashes': artifact_hashes(destination, ['profile.json'])})
    print('LONG_HISTORY_PROFILE_COMPLETE', cell, json.dumps(report['projection']), flush=True)


def verify_profile(output, cell):
    destination = output / 'profiles' / cell
    manifest = read_json(destination / 'manifest.json')
    if manifest['identity'] != {'preparation_sha256': hash_file(output / 'preparation.json'), 'cell': cell}:
        raise ValueError('A matching feasibility profile is required before a full fit')
    assert_hashes(destination, manifest['artifact_hashes'])
    result = read_json(destination / 'profile.json')
    if not result['projection']['within_limit']:
        raise ValueError('Local member projection exceeds 7200 seconds; register a resource plan before execution')


def run_member(config, local, output, preparation, cell, seed, command):
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError('Unregistered F4 member')
    if command == 'fit':
        verify_profile(output, cell)
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
            else: run_member(config, local, output, prepared, args.cell, args.seed, args.command)


if __name__ == '__main__':
    main()
