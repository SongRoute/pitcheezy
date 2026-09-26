"""Warmup-separated TRAIN resource profile for a fresh F4 preparation only.

The caller holds the shared heavy lock and supplies a 600-second subprocess
limit. No DEV features or labels are gathered, and no quality scores are saved.
Full model, training budget, batch size, seeds and 400-draw inference are fixed.
"""
from __future__ import annotations

import gc
import math
import resource
import time

import numpy as np
import torch

from .matrix_data import ordered_key_hash
from .matrix_lazy_model import LazyMatrixModel, LazyJointDelivery
from .model import outcome_labels


PROFILE_SPEC = {
    'version': 'f4_resource_profile_v2', 'warmup_train': 8192, 'warmup_epochs': 1,
    'warmup_eval_train': 2048, 'measured_train': 65536, 'measured_epochs': 4,
    'measured_patience': 4, 'earlystop': 2048, 'temperature': 16, 'seed': 0,
    'width': 128, 'batch_size': 256, 'learning_rate': .0005, 'draws': 400,
    'full_epochs': 30, 'sample_rule': 'chronological_linspace_no_shrink',
    'projection': 'cold_once_plus_all_measured_fit_cost_per_update_times_full30_updates',
    'data_loads_per_member': 2, 'profile_seconds': 600, 'member_seconds': 7200,
}
LIMITS = {'train': PROFILE_SPEC['measured_train'], 'earlystop': PROFILE_SPEC['earlystop'],
          'temperature': PROFILE_SPEC['temperature']}


def _synchronize(device):
    if device == 'mps': torch.mps.synchronize()


def _memory(device):
    return {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_current_allocated_bytes': int(torch.mps.current_allocated_memory()) if device == 'mps' else None,
            'mps_driver_allocated_bytes': int(torch.mps.driver_allocated_memory()) if device == 'mps' else None}


def _select(original, count):
    if len(original) < count:
        raise ValueError('Insufficient rows for exact resource-profile sample; do not shrink it')
    return original.subset(np.linspace(0, len(original)-1, count, dtype=np.int64))


def _sample(batch):
    return {'n': len(batch), 'rows_sha256': ordered_key_hash(batch.frame())}


def resource_projection(*, warmup_seconds, measured_fit_seconds, measured_updates, full_train_rows, full_earlystop_rows,
                        temperature_seconds, temperature_rows, inference_seconds, blend_rows, dev_rows,
                        load_seconds=0.):
    expected_updates = PROFILE_SPEC['measured_epochs']*math.ceil(PROFILE_SPEC['measured_train']/PROFILE_SPEC['batch_size'])
    if measured_updates != expected_updates or full_train_rows < PROFILE_SPEC['measured_train']:
        raise ValueError('Measured/full update-count contract differs')
    measured_validation_ratio = math.ceil(PROFILE_SPEC['earlystop']/256)/math.ceil(PROFILE_SPEC['measured_train']/256)
    full_validation_ratio = math.ceil(full_earlystop_rows/256)/math.ceil(full_train_rows/256)
    if full_earlystop_rows < PROFILE_SPEC['earlystop'] or full_validation_ratio > measured_validation_ratio:
        raise ValueError('Full early-stop batch ratio exceeds measured conservative profile ratio')
    values = (warmup_seconds, measured_fit_seconds, temperature_seconds, inference_seconds, load_seconds)
    if any(not np.isfinite(v) or v < 0 for v in values) or measured_fit_seconds <= 0:
        raise ValueError('Finite nonnegative resource measurements required')
    full_updates = PROFILE_SPEC['full_epochs']*math.ceil(full_train_rows/PROFILE_SPEC['batch_size'])
    measured_rate = measured_fit_seconds/measured_updates
    scaled_fit = measured_rate*full_updates
    full_fit = warmup_seconds+scaled_fit
    full_temperature = temperature_seconds*temperature_rows/PROFILE_SPEC['temperature']
    full_prediction = inference_seconds*(blend_rows+dev_rows)/PROFILE_SPEC['temperature']
    total = full_fit+full_temperature+full_prediction
    loads = PROFILE_SPEC['data_loads_per_member']*load_seconds
    return {'full_epochs_assumed': PROFILE_SPEC['full_epochs'], 'full_train_rows': int(full_train_rows),
            'full_optimizer_updates': int(full_updates), 'measured_optimizer_updates': int(measured_updates),
            'full_earlystop_rows': int(full_earlystop_rows),
            'measured_validation_batches_per_training_update': measured_validation_ratio,
            'full_validation_batches_per_training_update': full_validation_ratio,
            'seconds_per_measured_update_including_earlystop': measured_rate,
            'cold_warmup_seconds_added_once': warmup_seconds, 'scaled_measured_fit_seconds': scaled_fit,
            'full_fit_seconds_at_30_epochs': full_fit, 'temperature_seconds': full_temperature,
            'blend_dev_prediction_seconds': full_prediction, 'fit_temperature_prediction_seconds': total,
            'one_load_seconds': load_seconds, 'data_loads_per_member': PROFILE_SPEC['data_loads_per_member'],
            'load_seconds': loads, 'load_fit_temperature_prediction_seconds': loads+total,
            'limit_seconds': PROFILE_SPEC['member_seconds'],
            'within_limit': loads+total <= PROFILE_SPEC['member_seconds']}


def profile_batches(batches, delivery, *, device=None, width=128, load_seconds=0.):
    if delivery.draws != PROFILE_SPEC['draws'] or width != PROFILE_SPEC['width']:
        raise ValueError('Profile fixes width128 and frozen400-draw delivery pools')
    selected = {name: _select(batches[name], count) for name, count in LIMITS.items()}
    warm_train = _select(batches['train'], PROFILE_SPEC['warmup_train'])
    # Existing fit always evaluates once per epoch. Keep this discarded warmup
    # entirely within TRAIN, using a fixed subset of its own training sample.
    warm_eval = _select(warm_train, PROFILE_SPEC['warmup_eval_train'])
    for batch in (warm_train, warm_eval, selected['train']):
        frame = batch.frame()
        if 'split' not in frame or not frame.split.eq('train').all():
            raise ValueError('Warmup and measured optimizer rows must be TRAIN only')
    samples = {name: _sample(batch) for name, batch in selected.items()}
    warm_samples = {'train': _sample(warm_train), 'evaluation_train': _sample(warm_eval)}
    start = time.perf_counter()
    warmup = LazyMatrixModel(seed=0, width=width, device=device).fit(
        warm_train, outcome_labels(warm_train.frame()), warm_eval, outcome_labels(warm_eval.frame()),
        epochs=1, patience=1, batch_size=256, learning_rate=.0005)
    _synchronize(warmup.device)
    warm_report = {'samples': warm_samples, 'epochs': warmup.report['epochs_run'],
        'optimizer_updates': warmup.report['optimizer_updates'], 'evaluation_source': 'TRAIN subset only',
        'memory': _memory(warmup.device), 'discarded_before_measured_model_construction': True}
    if warm_report['epochs'] != 1 or warm_report['optimizer_updates'] != 32:
        raise ValueError('Warmup must perform exactly one8192-row epoch/32updates')
    del warmup
    gc.collect()
    warmup_seconds = time.perf_counter()-start
    warm_report['seconds'] = warmup_seconds
    # New object/init and optimizer: no warmup weights or optimizer state reused.
    start = time.perf_counter()
    model = LazyMatrixModel(seed=0, width=width, device=device).fit(
        selected['train'], outcome_labels(selected['train'].frame()),
        selected['earlystop'], outcome_labels(selected['earlystop'].frame()),
        epochs=4, patience=4, batch_size=256, learning_rate=.0005)
    _synchronize(model.device)
    fit_seconds = time.perf_counter()-start
    if model.report['epochs_run'] != 4 or model.report['optimizer_updates'] != 1024:
        raise ValueError('Measured profile must perform exactly four65536-row epochs/1024updates')
    measured_memory = _memory(model.device)
    marginal = LazyJointDelivery(delivery)
    start = time.perf_counter()
    marginal.calibrate(model, selected['temperature'], outcome_labels(selected['temperature'].frame()))
    _synchronize(model.device)
    temperature_seconds = time.perf_counter()-start
    temperature_memory = _memory(model.device)
    start = time.perf_counter()
    probability, _, _ = marginal.predict(model, selected['temperature'])
    _synchronize(model.device)
    inference_seconds = time.perf_counter()-start
    return {'profile_spec': PROFILE_SPEC, 'samples': samples, 'device': model.device,
            'population_counts': {name: len(batch) for name, batch in batches.items()},
            'epochs': 4, 'batch_size': 256, 'warmup': warm_report, 'warmup_seconds': warmup_seconds,
            'optimizer_updates': model.report['optimizer_updates'],
            'parameter_count': model.report['parameter_count'], 'network': model.net.config,
            'long_length': batches['train'].store.long_length, 'draws': 400,
            'load_seconds': load_seconds, 'fit_seconds': fit_seconds, 'measured_fit_seconds': fit_seconds,
            'temperature_seconds': temperature_seconds, 'inference_seconds': inference_seconds,
            'inference_query_pitches': len(selected['temperature']), **_memory(model.device),
            'stage_memory': {'measured_fit': measured_memory, 'temperature': temperature_memory,
                             'inference': _memory(model.device)},
            'maximum_probability_sum_error': float(np.abs(probability.sum(1)-1).max()),
            'max_expanded_training_rows': model.report['max_expanded_training_rows'],
            'projection': resource_projection(warmup_seconds=warmup_seconds, measured_fit_seconds=fit_seconds,
                measured_updates=model.report['optimizer_updates'], full_train_rows=len(batches['train']),
                full_earlystop_rows=len(batches['earlystop']),
                temperature_seconds=temperature_seconds, temperature_rows=len(batches['temperature']),
                inference_seconds=inference_seconds, blend_rows=len(batches['blend']), dev_rows=len(batches['dev']),
                load_seconds=load_seconds),
            'limits': ['Resource-only larger-sample extrapolation; not a promise of full runtime',
                      'All measured fit cost, including four early-stop passes, is scaled per optimizer update; no early-stop savings assumed',
                      'Cold warmup cost is added once even though it includes a discarded TRAIN fit/evaluation',
                      'Peak RSS is process-lifetime high-water mark; MPS allocator values are stage snapshots',
                      'Caller must enforce600-second profile subprocess cap and7200-second member cap'],
            'dev_features_gathered': False, 'dev_scores_read': False, 'quality_scores_omitted': True,
            'profile_weights_reused_as_full_member': False}
