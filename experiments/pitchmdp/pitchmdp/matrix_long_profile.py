"""Small TRAIN/early-stop/temperature feasibility probe for lazy F4 models.

The caller holds the shared heavy lock and supplies a 600-second subprocess
limit. No DEV features or labels are gathered, and no quality scores are saved.
"""
from __future__ import annotations

import resource
import time

import numpy as np
import torch

from .matrix_data import ordered_key_hash
from .matrix_lazy_model import LazyMatrixModel, LazyJointDelivery
from .model import outcome_labels


LIMITS = {'train': 8192, 'earlystop': 2048, 'temperature': 16}


def profile_batches(batches, delivery, *, device=None, width=128):
    if delivery.draws != 400:
        raise ValueError('Feasibility requires the frozen 400-draw delivery pool')
    selected = {}
    for name, limit in LIMITS.items():
        original = batches[name]
        if not len(original):
            raise ValueError('Profile fit/calibration splits must be nonempty')
        # Evenly spaced chronological rows cover the frozen date range.
        positions = np.linspace(0, len(original) - 1, min(limit, len(original)), dtype=np.int64)
        selected[name] = original.subset(positions)
    samples = {name: {'n': len(batch), 'rows_sha256': ordered_key_hash(batch.frame())}
               for name, batch in selected.items()}
    start = time.perf_counter()
    model = LazyMatrixModel(seed=0, width=width, device=device).fit(
        selected['train'], outcome_labels(selected['train'].frame()),
        selected['earlystop'], outcome_labels(selected['earlystop'].frame()),
        epochs=2, patience=2, batch_size=256, learning_rate=.0005)
    if model.device == 'mps':
        torch.mps.synchronize()
    fit_seconds = time.perf_counter() - start
    marginal = LazyJointDelivery(delivery)
    start = time.perf_counter()
    marginal.calibrate(model, selected['temperature'], outcome_labels(selected['temperature'].frame()))
    if model.device == 'mps':
        torch.mps.synchronize()
    temperature_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probability, _, _ = marginal.predict(model, selected['temperature'])
    if model.device == 'mps':
        torch.mps.synchronize()
    inference_seconds = time.perf_counter() - start
    # Cost is only a rough scaling projection, never a promise of wall-clock time.
    full_fit = fit_seconds / 2 * 30 * len(batches['train']) / len(selected['train'])
    full_temperature = temperature_seconds * len(batches['temperature']) / len(selected['temperature'])
    full_prediction = inference_seconds * (len(batches['blend']) + len(batches['dev'])) / len(selected['temperature'])
    return {'samples': samples, 'device': model.device, 'epochs': 2, 'batch_size': 256,
            'parameter_count': model.report['parameter_count'], 'network': model.net.config,
            'long_length': batches['train'].store.long_length, 'draws': 400,
            'fit_seconds': fit_seconds, 'temperature_seconds': temperature_seconds,
            'inference_seconds': inference_seconds, 'inference_query_pitches': len(selected['temperature']),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_current_allocated_bytes': int(torch.mps.current_allocated_memory()) if model.device == 'mps' else None,
            'mps_driver_allocated_bytes': int(torch.mps.driver_allocated_memory()) if model.device == 'mps' else None,
            'maximum_probability_sum_error': float(np.abs(probability.sum(1) - 1).max()),
            'max_expanded_training_rows': model.report['max_expanded_training_rows'],
            'projection': {'full_fit_seconds_at_30_epochs': full_fit,
                           'temperature_seconds': full_temperature, 'blend_dev_prediction_seconds': full_prediction,
                           'fit_temperature_prediction_seconds': full_fit + full_temperature + full_prediction,
                           'limit_seconds': 7200, 'within_limit': full_fit + full_temperature + full_prediction <= 7200},
            'limits': ['Linear extrapolation from a small sample; actual data load, allocation, convergence and backend behavior may differ',
                       'Caller must enforce a 600-second profile subprocess cap and 7200-second member cap'],
            'dev_features_gathered': False, 'dev_scores_read': False, 'quality_scores_omitted': True}
