"""Frozen common-input architecture benchmark contracts and prediction helpers.

This module does not select models or calculate DEV scores. The first benchmark
uses D100/H5/C6; later populations and contexts require a new protocol/version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import KEY
from .matrix_data import canonical_hash, ordered_key_hash

CELLS = {'A0-MLP': 'flatten_mlp', 'A1-linear': 'linear', 'A2-lightgbm': 'lightgbm',
         'A3-lstm': 'lstm', 'A4-gru': 'gru', 'A5-melville': 'melville',
         'A6-transformer': 'transformer'}
SEEDS = (0, 1, 2)
NEURAL = {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}
LIGHTGBM = {'epochs': 300, 'patience': 30, 'learning_rate': .05,
            'params': {'num_leaves': 31, 'min_data_in_leaf': 100, 'lambda_l2': 1.}}


def validate_config(config):
    required = {'protocol', 'experiment_id', 'parent_run', 'parent_preparation_sha256',
                'seeds', 'data_sample', 'history_length', 'draws', 'device', 'width',
                'neural', 'lightgbm'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'registration'}:
        raise ValueError('Invalid architecture benchmark config schema')
    if config['protocol'] != 'ml_architecture_v1' or not str(config['experiment_id']).strip():
        raise ValueError('Unregistered protocol or missing experiment ID')
    digest = config['parent_preparation_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('A frozen parent preparation SHA256 is required')
    if not isinstance(config['parent_run'], str) or not config['parent_run']:
        raise ValueError('A parent run is required')
    if (config['seeds'] != list(SEEDS) or config['data_sample'] != 'd100' or
            config['history_length'] != 5 or config['draws'] != 400 or config['width'] != 128):
        raise ValueError('First architecture family fixes seeds 0..2, D100, H5, 400 draws, width 128')
    if config['device'] not in ('auto', 'cpu', 'mps'):
        raise ValueError('Supported local device: auto, cpu, mps')
    if config['neural'] != NEURAL or config['lightgbm'] != LIGHTGBM:
        raise ValueError('Representative training configurations are fixed before DEV comparison')
    if 'registration' in config and not isinstance(config['registration'], dict):
        raise ValueError('Registration must be an object')
    return config


def select_keys(frame, keys, record):
    """Select an exact ordered sample without dropping, duplicating or sorting it."""
    if len(keys) != record['n'] or ordered_key_hash(keys) != record['rows_sha256']:
        raise ValueError('Prepared key identity changed')
    if frame.duplicated(KEY).any() or frame[KEY].isna().any().any():
        raise ValueError('Source frame has repeated or missing pitch keys')
    positions = pd.MultiIndex.from_frame(frame[KEY]).get_indexer(pd.MultiIndex.from_frame(keys[KEY]))
    if (positions < 0).any():
        raise ValueError('Prepared pitch keys are absent from source frame')
    selected = frame.iloc[positions]
    if ordered_key_hash(selected) != record['rows_sha256']:
        raise ValueError('Ordered source selection differs')
    return selected


def member_identity(preparation, cell, seed):
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError('Unregistered architecture member')
    return {'preparation_sha256': canonical_hash(preparation), 'cell': cell, 'kind': CELLS[cell],
            'seed': seed, 'train_rows_sha256': preparation['samples']['train']['rows_sha256'],
            'feature_sha256': canonical_hash(preparation['features'])}


def integrated_probabilities(logits, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[-1] != 10 or logits.shape[1] < 1 or not np.isfinite(logits).all():
        raise ValueError('Expected finite [pitch,draw,10] logits')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Expected positive finite temperature')
    result = tuple(softmax(logits / t, axis=-1).mean(1) for t in (temperature, 1.))
    for p in result:
        if ((p < 0).any() or (p > 1).any() or not np.isfinite(p).all() or
                not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0)):
            raise ValueError('Invalid integrated probability mass')
    return result


def predict_streamed(model, delivery, store, context, rows, chunk_size=64):
    """Keep only one small block of draw logits; output probabilities are float64."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or chunk_size < 1:
        raise ValueError('Expected row vector and positive chunk size')
    calibrated, raw = np.empty((len(rows), 10)), np.empty((len(rows), 10))
    levels = np.empty(len(rows), dtype=np.int64)
    for begin in range(0, len(rows), chunk_size):
        selected = rows[begin:begin + chunk_size]
        logits, tier = delivery.logits(model, store, context, selected)
        if logits.shape != (len(selected), delivery.draws, 10) or np.asarray(tier).shape != (len(selected),):
            raise ValueError('Prediction shape differs from frozen delivery contract')
        end = begin + len(selected)
        calibrated[begin:end], raw[begin:end] = integrated_probabilities(logits, model.delivery_temperature)
        levels[begin:end] = tier
    return calibrated, raw, levels
