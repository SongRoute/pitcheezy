"""Artifact adapters for a caller-registered lazy F4 experiment member.

The caller freezes population/auxiliary inputs and holds the shared heavy lock.
These adapters never load raw data, choose a population, or calculate DEV scores.
They accept the same frozen LazyPitchBatch objects for every capacity-matched arm.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

import numpy as np

from .data import KEY, hash_file
from .matrix_data import canonical_hash, ordered_key_hash
from .matrix_lazy_model import LazyMatrixModel, LazyJointDelivery
from .matrix_long_history import LazyPitchBatch
from .model import outcome_labels

CELLS = {'F4-H0': 0, 'F4-32': 32, 'F4-128': 128}
SEEDS = (0, 1, 2)
DEFAULT_BUDGET = {'epochs': 30, 'patience': 5, 'batch_size': 256, 'learning_rate': .0005}


def _dump(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def _read(path):
    return json.loads(path.read_text())


def _hashes(directory, names):
    return {name: hash_file(directory / name) for name in names}


def _verify(directory, hashes):
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or hash_file(path) != expected:
            raise ValueError('Lazy member artifact changed: ' + name)


def _source_hashes():
    root = Path(__file__).parent
    names = ('matrix_long_experiment.py', 'matrix_long_history.py', 'matrix_lazy_model.py',
             'matrix_features.py', 'model.py', 'sequence_data.py', 'sequence_delivery.py')
    return {name: hash_file(root / name) for name in names}


def _delivery_hash(delivery):
    digest = hashlib.sha256()
    digest.update(canonical_hash(delivery.report).encode('ascii'))
    for key, values in sorted(delivery.pools.items(), key=lambda pair: repr(pair[0])):
        digest.update(repr(key).encode('utf-8'))
        array = np.ascontiguousarray(values)
        digest.update(str((array.shape, str(array.dtype))).encode('ascii'))
        digest.update(memoryview(array))
    digest.update(memoryview(np.ascontiguousarray(delivery.fallback)))
    return digest.hexdigest()


def _identity(parent_preparation_sha256, batches, cell, seed, budget, width, device, frozen_delivery):
    if (len(parent_preparation_sha256) != 64 or any(c not in '0123456789abcdef' for c in parent_preparation_sha256)
            or cell not in CELLS or seed not in SEEDS):
        raise ValueError('Frozen parent identity, registered F4 cell and screen seed required')
    if set(batches) != {'train', 'earlystop', 'temperature', 'blend', 'dev'}:
        raise ValueError('All five frozen split batches are required')
    if not all(isinstance(b, LazyPitchBatch) and b.store.long_length == CELLS[cell] for b in batches.values()):
        raise ValueError('Lazy batches must have the member history length')
    game_sets = [set(b.frame().game_pk) for b in batches.values()]
    if any(not len(b) for b in batches.values()) or any(a & b for i, a in enumerate(game_sets) for b in game_sets[i + 1:]):
        raise ValueError('Registered split batches must be nonempty and game-disjoint')
    if any(b.current is not None or b.candidate_pitch_types is not None for b in batches.values()):
        raise ValueError('Registered observed-action experiment batches cannot contain query overrides')
    if any(not callable(getattr(batch.context, 'report', None)) for batch in batches.values()):
        raise ValueError('Frozen context encoders must expose their fitted report')
    contexts = [canonical_hash(b.context.report()) for b in batches.values()]
    normalizers = [canonical_hash(b.store.normalizer.report()) for b in batches.values()]
    if len(set(contexts)) != 1 or len(set(normalizers)) != 1:
        raise ValueError('All splits must share frozen context and physical preprocessing')
    records = {name: {'n': len(batch), 'rows_sha256': ordered_key_hash(batch.frame())} for name, batch in batches.items()}
    return {'parent_preparation_sha256': parent_preparation_sha256, 'cell': cell, 'seed': seed,
            'samples': records, 'budget': dict(budget), 'width': width, 'device_requested': device,
            'features': batches['train'].store.report(), 'source_hashes': _source_hashes(),
            'context_sha256': contexts[0], 'normalizer_sha256': normalizers[0],
            'delivery_sha256': _delivery_hash(frozen_delivery)}


def fit_member(directory, *, parent_preparation_sha256, batches, frozen_delivery, cell, seed,
               budget=None, width=128, device=None):
    """Fit one registered member; caller must hold the protocol heavy lock."""
    directory = Path(directory)
    budget = DEFAULT_BUDGET if budget is None else budget
    identity = _identity(parent_preparation_sha256, batches, cell, seed, budget, width, device, frozen_delivery)
    if frozen_delivery.draws != 400:
        raise ValueError('Registered F4 experiments require the shared 400 delivery draws')
    statepath = directory / 'fit_state.json'
    if statepath.exists():
        state = _read(statepath)
        if state['identity'] != identity:
            raise ValueError('Completed lazy member identity changed')
        _verify(directory, state['artifact_hashes'])
        return state
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Incomplete lazy member requires failure review; refusing overwrite')
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = LazyMatrixModel(seed=seed, width=width, device=device).fit(
        batches['train'], outcome_labels(batches['train'].frame()),
        batches['earlystop'], outcome_labels(batches['earlystop'].frame()), **budget)
    fitted = time.perf_counter()
    LazyJointDelivery(frozen_delivery).calibrate(model, batches['temperature'], outcome_labels(batches['temperature'].frame()))
    model.save(directory / 'model.pt')
    _dump(directory / 'fit.json', {'report': model.report, 'fit_seconds': fitted - start,
                                  'temperature_seconds': time.perf_counter() - fitted,
                                  'seconds_total': time.perf_counter() - start})
    if _source_hashes() != identity['source_hashes']:
        raise ValueError('Lazy implementation changed during fit')
    state = {'identity': identity, 'artifact_hashes': _hashes(directory, ['model.pt', 'fit.json'])}
    _dump(statepath, state)
    return state


def predict_member(directory, *, batches, frozen_delivery, device=None):
    """Save raw/calibrated keyed probabilities only; no DEV scoring or selection."""
    directory = Path(directory)
    fitted = _read(directory / 'fit_state.json')
    _verify(directory, fitted['artifact_hashes'])
    identity = fitted['identity']
    observed = _identity(identity['parent_preparation_sha256'], batches, identity['cell'], identity['seed'],
                         identity['budget'], identity['width'], identity['device_requested'], frozen_delivery)
    if observed != identity or frozen_delivery.draws != 400:
        raise ValueError('Lazy prediction inputs differ from registered fit')
    statepath = directory / 'prediction_state.json'
    if statepath.exists():
        state = _read(statepath)
        if state['fit_state_sha256'] != hash_file(directory / 'fit_state.json'):
            raise ValueError('Predictions reference a changed fit')
        _verify(directory, state['artifact_hashes'])
        return state
    if (directory / 'predictions.npz').exists():
        raise ValueError('Uncommitted predictions require failure review')
    start = time.perf_counter()
    model = LazyMatrixModel.load(directory / 'model.pt', device=device)
    delivery = LazyJointDelivery(frozen_delivery)
    values, tier_counts = {}, {}
    for name in ('blend', 'dev'):
        batch = batches[name]
        frame = batch.frame()
        p, raw, levels = delivery.predict(model, batch)
        values.update({name: p, name + '_raw': raw, name + '_delivery_level': levels,
                       name + '_keys': frame[KEY].to_numpy(np.int64), name + '_y': outcome_labels(frame),
                       name + '_game_pk': frame.game_pk.to_numpy(np.int64),
                       name + '_pitcher': frame.pitcher.to_numpy(np.int64)})
        keys, counts = np.unique(levels, return_counts=True)
        tier_counts[name] = {str(int(k)): int(v) for k, v in zip(keys, counts)}
    np.savez_compressed(directory / 'predictions.npz', **values)
    _dump(directory / 'prediction_runtime.json', {'seconds': time.perf_counter() - start,
                    'delivery_tier_counts': tier_counts, 'model_batch_size': delivery.model_batch_size,
                    'pitch_chunk': delivery.pitch_chunk, 'dev_scored': False})
    if _source_hashes() != identity['source_hashes']:
        raise ValueError('Lazy implementation changed during prediction')
    state = {'fit_state_sha256': hash_file(directory / 'fit_state.json'),
             'artifact_hashes': _hashes(directory, ['predictions.npz', 'prediction_runtime.json'])}
    _dump(statepath, state)
    return state
