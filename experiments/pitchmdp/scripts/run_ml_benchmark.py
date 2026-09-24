"""Prepare and execute a registered common-input ML2 architecture family.

No DEV scores or candidate selection occur here. Every stage shares the ML
matrix heavy-process lock. Parent samples and auxiliary bytes are immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import resource
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd
import scipy
import torch

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_benchmark import (CELLS, SEEDS, validate_config, select_keys,
                                       member_identity, predict_streamed)
from pitchmdp.matrix_data import canonical_hash, load_verified_processed_cache
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.model import outcome_labels
from run_ml_matrix import (SOURCES as PARENT_SOURCES, assert_hashes, artifact_hashes,
                           check_location, heavy_lock)
from run_temporal_blend import assign_fold
from run_sequence_pilot import arrays

SOURCES = list(dict.fromkeys([*PARENT_SOURCES, 'pitchmdp/matrix_features.py',
           'pitchmdp/matrix_models.py', 'pitchmdp/matrix_benchmark.py', 'scripts/run_ml_benchmark.py']))


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    return {rel: hash_file(PROJECT / rel) for rel in SOURCES}


def identity(config, local_path):
    try:
        lgb_version = importlib.metadata.version('lightgbm')
    except importlib.metadata.PackageNotFoundError:
        lgb_version = None
    return {'config_sha256': canonical_hash(config), 'local_config_sha256': hash_file(local_path),
            'source_hashes': source_hashes(), 'python': str(Path(sys.executable).resolve()),
            'python_version': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'lightgbm': lgb_version,
            'dynamic_library_path': os.environ.get('DYLD_LIBRARY_PATH', ''),
            'pythonpath': os.environ.get('PYTHONPATH', '')}


def validate_native_runtime():
    # Apple torch and Homebrew LightGBM otherwise initialize two OpenMP copies.
    # Resolve one runtime before Python starts; never use the unsafe duplicate flag.
    if sys.platform == 'darwin':
        torch_lib = (Path(torch.__file__).resolve().parent / 'lib').resolve()
        configured = [Path(value).resolve() for value in os.environ.get('DYLD_LIBRARY_PATH', '').split(':') if value]
        if torch_lib not in configured:
            raise ValueError(f'Launch with DYLD_LIBRARY_PATH={torch_lib} so torch and LightGBM share one OpenMP runtime')
    if os.environ.get('KMP_DUPLICATE_LIB_OK', '').lower() in ('true', '1', 'yes'):
        raise ValueError('Unsafe duplicate OpenMP override is forbidden for benchmark runs')


def parent_preparation(config, local, output):
    root = Path(local['artifact_root']).resolve()
    parent = Path(config['parent_run']).resolve()
    if not parent.is_relative_to(root / 'runs' / 'ML-MATRIX-20260924'):
        raise ValueError('Parent must be an approved ML matrix run')
    if output == parent or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError('Architecture output and parent must be distinct sibling runs')
    if hash_file(parent / 'preparation.json') != config['parent_preparation_sha256']:
        raise ValueError('Parent preparation differs from registration')
    prep = read_json(parent / 'preparation.json')
    assert_hashes(parent, prep['artifact_hashes'])
    # Pickled auxiliary objects and reconstructed features must retain their implementation.
    for relative, expected in prep['identity']['source_hashes'].items():
        if hash_file(PROJECT / relative) != expected:
            raise ValueError('Parent implementation changed: ' + relative)
    samples = prep['scope']['regular']['samples']
    if samples['d100']['rows_sha256'] != samples['full_train']['rows_sha256']:
        raise ValueError('Parent D100 differs from the full regular-season eligible pool')
    return parent, prep


def regular_frame(local, parent):
    raw = load_verified_processed_cache(local)
    if raw.attrs['sequence_data_identity'] != parent['dataset_identity']:
        raise ValueError('Dataset differs from parent')
    if raw.attrs['matrix_source_provenance'] != parent['source_provenance']:
        raise ValueError('Source provenance differs from parent')
    frame = raw.loc[raw.game_type.eq('R')].copy().reset_index(drop=True)
    return assign_fold(add_batter_style_history(frame), 2025)


def verify(output, expected_identity):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected_identity:
        raise ValueError('Architecture configuration, environment, or sources changed')
    assert_hashes(output, prep['artifact_hashes'])
    return prep


def prepare(config, local, output, expected_identity):
    if output.exists() and any(output.iterdir()):
        verify(output, expected_identity)
        print('BENCHMARK_PREPARED', output, flush=True)
        return
    parent_path, parent = parent_preparation(config, local, output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for rel in SOURCES:
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    dump(output / 'registered_config.json', config)
    # Preserve the parent's provenance as a hashed, self-contained local artifact.
    shutil.copyfile(parent_path / 'preparation.json', output / 'parent_preparation.json')
    shutil.copyfile(parent_path / 'regular_aux.pkl', output / 'aux.pkl')
    shutil.copyfile(parent_path / 'regular_baseline_predictions.npz', output / 'baseline_predictions.npz')
    records = parent['scope']['regular']['samples']
    samples = {}
    for name in ('train', 'earlystop', 'temperature', 'blend', 'dev'):
        original = records['d100' if name == 'train' else name]
        target = f'{name}_keys.parquet'
        shutil.copyfile(parent_path / original['path'], output / target)
        samples[name] = {**original, 'path': target}
    frame = regular_frame(local, parent)
    with (output / 'aux.pkl').open('rb') as stream:
        aux = pickle.load(stream)
    if aux['delivery'].draws != config['draws']:
        raise ValueError('Parent delivery draw budget differs')
    store = MatrixHistoryStore.from_frame(frame, normalizer=aux['normalizer'], history_length=config['history_length'])
    # Fail preparation if any copied sample cannot be reconstructed exactly.
    for record in samples.values():
        select_keys(frame, pd.read_parquet(output / record['path']), record)
    features = {'tokens': store.report(), 'context': aux['context'].report(),
                'auxiliary_scope': parent['scope']['regular']['aux'],
                'no_retrospective_support_input': True}
    dump(output / 'features.json', features)
    if source_hashes() != expected_identity['source_hashes']:
        raise ValueError('Source changed during architecture preparation')
    files = [str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()]
    report = {'identity': expected_identity, 'parent_run': str(parent_path),
              'parent_preparation_sha256': config['parent_preparation_sha256'],
              'samples': samples, 'features': features,
              'cells': CELLS, 'seeds': list(SEEDS), 'dataset_identity': parent['dataset_identity'],
              'source_provenance': parent['source_provenance'],
              'artifact_hashes': artifact_hashes(output, files),
              'seconds': time.perf_counter() - start,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scored': False,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT, text=True).strip())}
    dump(output / 'preparation.json', report)
    print('BENCHMARK_PREPARED', output, flush=True)


def load_data(local, output, preparation):
    parent = read_json(output / 'parent_preparation.json')
    frame = regular_frame(local, parent)
    with (output / 'aux.pkl').open('rb') as stream:
        aux = pickle.load(stream)
    tokens = preparation['features']['tokens']
    store = MatrixHistoryStore.from_frame(frame, normalizer=aux['normalizer'],
                history_length=tokens['history_length'], type_vocabulary=tokens['type_vocabulary'])
    if store.report() != tokens:
        raise ValueError('Enriched feature contract changed')
    parts = {name: select_keys(frame, pd.read_parquet(output / record['path']), record)
             for name, record in preparation['samples'].items()}
    return store, parts, aux


def member_dir(output, cell, seed):
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError('Unregistered architecture cell/seed')
    return output / 'members' / cell / f'seed{seed}'


def fit(config, local, output, preparation, cell, seed):
    destination = member_dir(output, cell, seed)
    member = member_identity(preparation, cell, seed)
    statepath = destination / 'fit_state.json'
    if statepath.exists():
        state = read_json(statepath)
        if state['identity'] != member:
            raise ValueError('Completed architecture member changed')
        assert_hashes(destination, state['artifact_hashes'])
        print('BENCHMARK_FIT_COMPLETE', cell, seed, flush=True)
        return
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Incomplete member requires failure review; refusing overwrite')
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    store, parts, aux = load_data(local, output, preparation)
    train, early = parts['train'], parts['earlystop']
    params = config['lightgbm'] if CELLS[cell] == 'lightgbm' else config['neural']
    model = MatrixModel(CELLS[cell], seed=seed, width=config['width'],
                        device=None if config['device'] == 'auto' else config['device'],
                        lightgbm_params=config['lightgbm']['params'])
    model.fit(arrays(store, aux['context'], train.index.to_numpy()), outcome_labels(train),
              arrays(store, aux['context'], early.index.to_numpy()), outcome_labels(early),
              **{key: value for key, value in params.items() if key != 'params'})
    fitted_seconds = time.perf_counter() - start
    temperature = parts['temperature']
    aux['delivery'].calibrate(model, store, aux['context'], temperature.index.to_numpy(), outcome_labels(temperature))
    calibrated_seconds = time.perf_counter() - start - fitted_seconds
    model.save(destination / 'model.pt')
    dump(destination / 'fit.json', {'report': model.report, 'load_and_fit_seconds': fitted_seconds,
                'temperature_seconds': calibrated_seconds, 'seconds_total': time.perf_counter() - start,
                'train_rows_sha256': preparation['samples']['train']['rows_sha256'],
                'temperature_rows_sha256': preparation['samples']['temperature']['rows_sha256'],
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    if source_hashes() != preparation['identity']['source_hashes']:
        raise ValueError('Source changed during architecture fit')
    dump(statepath, {'identity': member, 'artifact_hashes': artifact_hashes(destination, ['model.pt', 'fit.json']),
                     'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('BENCHMARK_FIT_COMPLETE', cell, seed, flush=True)


def predict(config, local, output, preparation, cell, seed):
    destination = member_dir(output, cell, seed)
    fitted = read_json(destination / 'fit_state.json')
    if fitted['identity'] != member_identity(preparation, cell, seed):
        raise ValueError('Fit identity changed')
    assert_hashes(destination, fitted['artifact_hashes'])
    statepath = destination / 'prediction_state.json'
    if statepath.exists():
        state = read_json(statepath)
        if state['fit_state_sha256'] != hash_file(destination / 'fit_state.json'):
            raise ValueError('Prediction references changed fit')
        assert_hashes(destination, state['artifact_hashes'])
        print('BENCHMARK_PREDICT_COMPLETE', cell, seed, flush=True)
        return
    if (destination / 'predictions.npz').exists():
        raise ValueError('Uncommitted predictions require failure review')
    start = time.perf_counter()
    store, parts, aux = load_data(local, output, preparation)
    model = MatrixModel.load(destination / 'model.pt', device=None if config['device'] == 'auto' else config['device'])
    if model.kind != CELLS[cell] or model.seed != seed:
        raise ValueError('Checkpoint architecture/seed differs')
    predictions, tier_counts = {}, {}
    for split in ('blend', 'dev'):
        part = parts[split]
        p, raw, levels = predict_streamed(model, aux['delivery'], store, aux['context'], part.index.to_numpy())
        predictions.update({split: p, split + '_raw': raw, split + '_delivery_level': levels,
                            split + '_keys': part[KEY].to_numpy(np.int64), split + '_y': outcome_labels(part),
                            split + '_game_pk': part.game_pk.to_numpy(np.int64),
                            split + '_pitcher': part.pitcher.to_numpy(np.int64)})
        unique, counts = np.unique(levels, return_counts=True)
        tier_counts[split] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    np.savez_compressed(destination / 'predictions.npz', **predictions)
    dump(destination / 'prediction_runtime.json', {'seconds': time.perf_counter() - start,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'delivery_tier_counts': tier_counts, 'draw_logit_chunk_pitches': 64, 'dev_scored': False})
    if source_hashes() != preparation['identity']['source_hashes']:
        raise ValueError('Source changed during architecture prediction')
    dump(statepath, {'fit_state_sha256': hash_file(destination / 'fit_state.json'),
                    'artifact_hashes': artifact_hashes(destination, ['predictions.npz', 'prediction_runtime.json']),
                    'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('BENCHMARK_PREDICT_COMPLETE', cell, seed, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    commands.add_parser('status')
    for name in ('fit', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('--cell', choices=list(CELLS), required=True)
        command.add_argument('--seed', choices=list(SEEDS), type=int, required=True)
    args = parser.parse_args()
    config, local = validate_config(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected_identity = identity(config, args.local_config)
    if expected_identity['lightgbm'] is None:
        raise ValueError('All-family environment must expose installed LightGBM before preparation')
    if args.command == 'status':
        verify(output, expected_identity)
        for cell in CELLS:
            for seed in SEEDS:
                member = member_dir(output, cell, seed)
                state = 'predicted' if (member / 'prediction_state.json').exists() else 'fitted' if (member / 'fit_state.json').exists() else 'pending'
                print(cell, seed, state)
        return
    with heavy_lock(root):
        if args.command == 'prepare':
            prepare(config, local, output, expected_identity)
            return
        prep = verify(output, expected_identity)
        (fit if args.command == 'fit' else predict)(config, local, output, prep, args.cell, args.seed)


if __name__ == '__main__':
    main()
