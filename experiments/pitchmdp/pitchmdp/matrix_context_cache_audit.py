"""Audit stages for the optional F4 observed-context cache (COOP-001).

Backend-agnostic stage helpers; the CLI fixes the registered device, samples,
budget and tolerances. Only TRAIN/early-stop/May batches are accepted: no DEV
or June rows, no quality metric, no full member fit and no adoption occur here.
Frozen scientific sources are used as building blocks and are not modified.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import itertools
import math
import time

import numpy as np
import torch
from torch import nn

from .archetypes import STYLE_COLUMNS
from .matrix_data import ordered_key_hash
from .matrix_lazy_model import LazyMatrixModel, LazyJointDelivery, DualStreamNetwork, _tensors
from .matrix_long_equivalence import array_hash, rng_state, synchronize, memory
from .matrix_long_history import LazyPitchBatch
from .matrix_long_profile import PROFILE_SPEC, resource_projection
from .matrix_observed_context_cache import FrozenObservedContext
from .model import outcome_labels

REGISTERED_SAMPLES = {'train': 65536, 'train_evaluation': 2048, 'earlystop': 2048, 'temperature': 16}
WARMUP_TRAIN = 8192
STAGE1_ORDERS = ('original', 'repeated', 'reverse', 'uneven_chunks')
STAGE1_MUST_FAIL = ('count_change', 'style_change', 'undeclared_row')
STAGE2_TOLERANCES = {'input_context': 'bitwise',
                     'forward_logits': {'atol': 2e-5, 'rtol': 0.},
                     'loss': {'atol': 2e-6, 'rtol': 0.},
                     'gradients': {'atol': 1e-6, 'rtol': 1e-5},
                     'updated_weights': {'atol': 1e-6, 'rtol': 1e-5}}
STAGE3_TOLERANCES = {'selected_state': {'atol': 1e-6, 'rtol': 1e-5},
                     'history_nll': {'atol': 2e-6, 'rtol': 0.},
                     'delivery_temperature': {'atol': 1e-3, 'rtol': 0.},
                     'probability': {'atol': 2e-5, 'rtol': 0.},
                     'probability_sum': {'atol': 1e-6, 'rtol': 0.},
                     'calibration_objective': {'atol': 2e-6, 'rtol': 0.}}
STAGE3_BUDGET = {'epochs': 4, 'patience': 4, 'batch_size': 256, 'learning_rate': .0005}
ACCEPTED_SPLITS = ('train', 'earlystop', 'temperature')
PRIOR_LEDGER_PROTOCOL = 'ml_long_owner_budget_ledger_v2'
PRIOR_LEDGER_CATEGORIES = {'preparation', 'profiles', 'cold_failed_attempts', 'equivalence', 'cache_audit'}


def _linspace(batch, count):
    if len(batch) < count:
        raise ValueError('Insufficient rows for the registered chronological sample; do not shrink it')
    return batch.subset(np.linspace(0, len(batch)-1, count, dtype=np.int64))


def select_samples(batches, counts=REGISTERED_SAMPLES, warmup_train=WARMUP_TRAIN):
    """v3 profile selectors: linspace over each split; evaluation rows are a linspace of the TRAIN warmup."""
    if set(batches) != set(ACCEPTED_SPLITS):
        raise ValueError('Only TRAIN/earlystop/May-temperature batches are accepted; DEV/June/blend are forbidden')
    if set(counts) != set(REGISTERED_SAMPLES):
        raise ValueError('Sample count registry differs')
    warm = _linspace(batches['train'], warmup_train)
    selected = {'train': _linspace(batches['train'], counts['train']),
                'train_evaluation': _linspace(warm, counts['train_evaluation']),
                'earlystop': _linspace(batches['earlystop'], counts['earlystop']),
                'temperature': _linspace(batches['temperature'], counts['temperature'])}
    for batch in (warm, selected['train'], selected['train_evaluation']):
        frame = batch.frame()
        if 'split' not in frame or not frame.split.eq('train').all():
            raise ValueError('Warmup, measured and evaluation optimizer rows must be TRAIN only')
    return selected, warm


def sample_records(selected):
    return {name: {'n': int(len(batch)), 'rows_sha256': ordered_key_hash(batch.frame())}
            for name, batch in selected.items()}


def check_sample_identity(records, registered):
    for name, record in records.items():
        expected = registered[name]
        if record['n'] != expected['n'] or record['rows_sha256'] != expected['rows_sha256']:
            raise ValueError('Registered sample selector identity differs: ' + name)


def cached_batch(batch, cache):
    return LazyPitchBatch(batch.store, cache, batch.rows, batch.current, batch.candidate_pitch_types)


def build_cache(base, frame, rows, *, device='cpu', chunk_size=8192):
    """Construct the exact cache; RNG state and the scientific report must be unchanged."""
    if isinstance(base, FrozenObservedContext):
        raise ValueError('Original encoder required; do not cache a cache')
    before = rng_state(device)
    synchronize(device)
    started = time.perf_counter()
    cache = FrozenObservedContext(base, frame, rows, chunk_size=chunk_size)
    seconds = time.perf_counter()-started
    if rng_state(device) != before:
        raise ValueError('Cache construction changed a global RNG state')
    if cache.report() != base.report():
        raise ValueError('Cache changed the scientific context report')
    return cache, {'construction_seconds': seconds, 'construction_rows': int(len(np.unique(rows))),
                   'rng_state_unchanged': True, 'cache_report': cache.cache_report(), 'memory': memory(device)}


def _bitwise(a, b, label):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError(f'Cached output differs bitwise from the original: {label}')
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'differing_bits': 0, 'sha256': array_hash(a)}


def _must_fail(action, match):
    try:
        action()
    except ValueError as error:
        if match in str(error):
            return {'rejected': True, 'error': str(error)}
        raise ValueError('Guard failed with an unexpected message: ' + str(error))
    raise ValueError('Cache guard did not reject a changed or undeclared observed row')


def stage1_context(selected, *, chunk_size=8192, repetitions=3, uneven_chunk_sizes=(1, 7, 256, 1000),
                   timing_batch_size=256, device='cpu'):
    """Bitwise context checks in four orders, override invariance, guard rejection and timing."""
    if repetitions < 2 or not uneven_chunk_sizes or any(int(s) < 1 for s in uneven_chunk_sizes):
        raise ValueError('At least two alternating timing repetitions and positive uneven chunk sizes required')
    train, evaluation = selected['train'], selected['train_evaluation']
    store, base, frame = train.store, train.context, train.store.frame
    union = np.union1d(train.rows, evaluation.rows)
    cache, build = build_cache(base, frame, union, device=device, chunk_size=chunk_size)

    def compare(rows, label, sizes):
        """Bounded-memory bitwise comparison: rows are consumed in the given chunk-size cycle."""
        digest, begin, count, cycle = hashlib.sha256(), 0, 0, itertools.cycle(int(s) for s in sizes)
        while begin < len(rows):
            size = next(cycle)
            query = frame.iloc[rows[begin:begin+size]]
            record = _bitwise(np.asarray(base.transform(query), dtype=np.float32), cache.transform(query), label)
            digest.update(record['sha256'].encode()); begin += size; count += 1
        return {'rows': int(len(rows)), 'chunks': count, 'chunk_sizes': [int(s) for s in sizes],
                'differing_bits': 0, 'chunk_sha256_chain': digest.hexdigest()}
    checks = {}
    for sample, batch in (('train', train), ('train_evaluation', evaluation)):
        rows = batch.rows
        checks[sample] = {'rows_sha256': ordered_key_hash(batch.frame()),
                          'original': compare(rows, f'{sample} original order', (chunk_size,)),
                          'repeated': compare(np.tile(rows, 2), f'{sample} repeated rows', (chunk_size,)),
                          'reverse': compare(rows[::-1], f'{sample} reverse order', (chunk_size,)),
                          'uneven_chunks': compare(rows, f'{sample} uneven chunks', uneven_chunk_sizes)}
    vocabulary = list(store.base.type_vocabulary)
    if len(vocabulary) < 2:
        raise ValueError('Two TRAIN pitch types are required for candidate overrides')
    n = len(evaluation.rows)
    candidates = np.array([vocabulary[i % len(vocabulary)] for i in range(n)])
    current = (np.arange(n*8, dtype=np.float32).reshape(n, 8) % 7)/7.-.5
    left = LazyPitchBatch(store, base, evaluation.rows, current=current, candidate_pitch_types=candidates)
    right = cached_batch(left, cache)
    overrides = {name: _bitwise(a, b, 'override gather ' + name) for name, a, b in
                 zip(('h5', 'h5_valid', 'long', 'long_valid', 'context'), left.gather(), right.gather())}
    probe = frame.iloc[evaluation.rows[:4]]
    count_changed = probe.copy(); count_changed['balls'] = count_changed['balls'].to_numpy()+1
    style_changed = probe.copy(); style_changed[STYLE_COLUMNS[0]] = style_changed[STYLE_COLUMNS[0]].to_numpy()+1.
    outside = np.setdiff1d(selected['earlystop'].rows, union)
    if not len(outside):
        raise ValueError('An undeclared observed row is required for the guard probe')
    must_fail = {'count_change': _must_fail(lambda: cache.transform(count_changed), 'dependency changed'),
                 'style_change': _must_fail(lambda: cache.transform(style_changed), 'dependency changed'),
                 'undeclared_row': _must_fail(lambda: cache.transform(frame.iloc[outside[:1]]), 'not registered')}
    minibatches = [evaluation.rows[b:b+timing_batch_size] for b in range(0, n, timing_batch_size)]

    def context_only(context):
        for rows in minibatches: context.transform(frame.iloc[rows])

    def lazy_gather(context):
        for rows in minibatches: LazyPitchBatch(store, context, rows).gather()
    timing, schedule = {}, []
    for kind, action in (('context_transform', context_only), ('lazy_gather', lazy_gather)):
        timing[kind] = {'original': [], 'cached': []}
        action(base); action(cache)
        for repetition in range(repetitions):
            order = ('original', 'cached') if repetition % 2 == 0 else ('cached', 'original')
            for label in order:
                started = time.perf_counter()
                action(base if label == 'original' else cache)
                timing[kind][label].append(time.perf_counter()-started)
                schedule.append([kind, repetition, label])
    return {'protocol': 'f4_cache_audit_stage1_v1', 'device': device, 'long_length': store.long_length,
            'construction': build, 'orders': list(STAGE1_ORDERS), 'samples_checked': ['train', 'train_evaluation'],
            'checks': checks, 'tolerance': 'bitwise', 'check_chunk_size': chunk_size,
            'overrides': overrides, 'override_candidate_types': sorted(set(candidates.tolist())),
            'must_fail': must_fail, 'all_bitwise_identical': True, 'all_guards_rejected': True,
            'timing': {'protocol': 'each path warmed once; alternating original/cached order per repetition',
                       'rows_per_repetition': n, 'minibatch_size': timing_batch_size,
                       'repetitions': repetitions, 'seconds': timing, 'schedule': schedule},
            'memory': memory(device), 'dev_features_gathered': False, 'dev_scores_read': False}


def _difference(a, b, tolerance):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError('Compared arrays differ in shape')
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    gap = float(np.abs(a-b).max()) if a.size else 0.
    passed = finite and bool(np.allclose(a, b, rtol=tolerance['rtol'], atol=tolerance['atol']))
    return {'max_absolute_difference': gap, 'finite': finite, 'pass': passed}


def _difference_map(a, b, tolerance):
    if set(a) != set(b):
        raise ValueError('Compared parameter families differ')
    values = {name: _difference(a[name], b[name], tolerance) for name in a}
    return {'max_absolute_difference': max((v['max_absolute_difference'] for v in values.values()), default=0.),
            'pass': all(v['pass'] for v in values.values()), 'parameters': values}


def stage2_backend(train, *, device, minibatches=8, batch_size=256, seed=0, width=128,
                   learning_rate=.0005, chunk_size=8192, tolerances=STAGE2_TOLERANCES):
    """Identical init, identical minibatch order: forward, loss, gradients, AdamW-updated weights."""
    if minibatches < 1 or batch_size < 1 or len(train) < minibatches*batch_size:
        raise ValueError('Registered minibatch count exceeds the fixed TRAIN sample')
    store, base = train.store, train.context
    order = np.random.default_rng(seed).permutation(len(train))  # same generator/order as LazyMatrixModel.fit epoch1
    steps = [order[begin:begin+batch_size] for begin in range(0, minibatches*batch_size, batch_size)]
    cache, build = build_cache(base, store.frame, train.rows[np.concatenate(steps)], device=device, chunk_size=chunk_size)
    labels = outcome_labels(train.frame())
    one = train.subset(np.array([0])).gather()
    torch.manual_seed(seed)
    reference = DualStreamNetwork(one[0].shape[-1], one[-1].shape[-1], width).to(device)
    optimized = deepcopy(reference)
    initial = {n: p.detach().cpu().numpy().copy() for n, p in reference.named_parameters()}
    for name, parameter in optimized.named_parameters():
        _bitwise(initial[name], parameter.detach().cpu().numpy(), 'initial weights ' + name)
    paths = {'original': (reference, train), 'cached': (optimized, cached_batch(train, cache))}
    optimizers = {label: torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=.01)
                  for label, (net, _) in paths.items()}
    reports = []
    for number, indices in enumerate(steps, start=1):
        outputs = {}
        for label, (net, batch) in paths.items():
            before = rng_state(device)
            tensors = _tensors(batch.subset(indices), device)
            target = torch.as_tensor(labels[indices].astype(np.int64), device=device)
            weight = torch.ones(len(indices), device=device)
            net.train()
            optimizers[label].zero_grad(set_to_none=True)
            logits = net(*tensors)
            loss = (nn.functional.cross_entropy(logits, target, reduction='none')*weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizers[label].step()
            synchronize(device)
            if rng_state(device) != before:
                raise ValueError('Training step changed a global RNG state outside the explicit update')
            outputs[label] = {'context': tensors[-1].detach().cpu().numpy(), 'logits': logits.detach().cpu().numpy(),
                              'loss': float(loss.item()),
                              'gradients': {n: p.grad.detach().cpu().numpy().copy() for n, p in net.named_parameters()},
                              'weights': {n: p.detach().cpu().numpy().copy() for n, p in net.named_parameters()}}
        a, b = outputs['original'], outputs['cached']
        reports.append({'step': number, 'rows': int(len(indices)),
                        'input_context': _bitwise(a['context'], b['context'], f'step{number} input context'),
                        'forward_logits': _difference(a['logits'], b['logits'], tolerances['forward_logits']),
                        'loss': {'original': a['loss'], 'cached': b['loss'],
                                 **_difference([a['loss']], [b['loss']], tolerances['loss'])},
                        'gradients': _difference_map(a['gradients'], b['gradients'], tolerances['gradients']),
                        'updated_weights': _difference_map(a['weights'], b['weights'], tolerances['updated_weights'])})
    keys = ('forward_logits', 'loss', 'gradients', 'updated_weights')
    return {'protocol': 'f4_cache_audit_stage2_v1', 'device': device, 'long_length': store.long_length,
            'seed': seed, 'width': width, 'batch_size': batch_size, 'learning_rate': learning_rate,
            'minibatches': minibatches, 'identical_initialization': True, 'construction': build,
            'tolerances': deepcopy(tolerances), 'steps': reports,
            'equivalent': all(step[key]['pass'] for step in reports for key in keys),
            'rng_state_unchanged_by_steps': True, 'memory': memory(device),
            'dev_features_gathered': False, 'dev_scores_read': False}


def stage3_fit(selected, warm_train, delivery, *, device, width=128, budget=STAGE3_BUDGET,
               cache=None, path='original'):
    """v3-style resource fit: discarded TRAIN warmup, fresh fit, May calibration/inference."""
    if delivery.draws != PROFILE_SPEC['draws']:
        raise ValueError('Frozen 400-draw delivery pool required')
    if (cache is None) != (path == 'original'):
        raise ValueError('Path label must match the presence of a cache')
    batches = {name: selected[name] for name in ('train', 'train_evaluation', 'earlystop', 'temperature')}
    batches['warmup_train'] = warm_train
    if cache is not None:
        batches = {name: cached_batch(batch, cache) for name, batch in batches.items()}
    labels = {name: outcome_labels(batch.frame()) for name, batch in batches.items()}
    synchronize(device)
    started = time.perf_counter()
    warmup = LazyMatrixModel(seed=0, width=width, device=device).fit(
        batches['warmup_train'], labels['warmup_train'], batches['train_evaluation'], labels['train_evaluation'],
        epochs=1, patience=1, batch_size=budget['batch_size'], learning_rate=budget['learning_rate'])
    synchronize(device)
    warm = {'epochs': warmup.report['epochs_run'], 'optimizer_updates': warmup.report['optimizer_updates'],
            'evaluation_source': 'TRAIN subset only', 'discarded_before_measured_model_construction': True,
            'memory': memory(device)}
    del warmup
    gc.collect()
    warm['seconds'] = time.perf_counter()-started
    started = time.perf_counter()
    model = LazyMatrixModel(seed=0, width=width, device=device).fit(
        batches['train'], labels['train'], batches['earlystop'], labels['earlystop'], **budget)
    synchronize(device)
    fit_seconds = time.perf_counter()-started
    fit_memory = memory(device)
    marginal = LazyJointDelivery(delivery)
    started = time.perf_counter()
    marginal.calibrate(model, batches['temperature'], labels['temperature'])
    synchronize(device)
    temperature_seconds = time.perf_counter()-started
    started = time.perf_counter()
    probability, raw, tiers = marginal.predict(model, batches['temperature'])
    synchronize(device)
    inference_seconds = time.perf_counter()-started
    state = {key: value.detach().cpu().numpy().copy() for key, value in model.net.state_dict().items()}
    result = {'protocol': 'f4_cache_audit_stage3_v1', 'path': path, 'device': model.device,
              'long_length': batches['train'].store.long_length, 'width': width, 'budget': dict(budget),
              'warmup': warm, 'warmup_seconds': warm['seconds'], 'fit_seconds': fit_seconds,
              'temperature_seconds': temperature_seconds, 'inference_seconds': inference_seconds,
              'epochs_run': model.report['epochs_run'], 'best_epoch': model.report['best_epoch'],
              'optimizer_updates': model.report['optimizer_updates'], 'history': model.report['history'],
              'delivery_temperature': model.delivery_temperature,
              'calibration_integrated_log_loss': model.report['calibration_integrated_log_loss'],
              'parameter_count': model.report['parameter_count'], 'network': model.net.config,
              'draws': delivery.draws, 'inference_query_pitches': int(len(batches['temperature'])),
              'maximum_probability_sum_error': float(np.abs(probability.sum(1)-1).max()),
              'stage_memory': {'measured_fit': fit_memory, 'inference': memory(device)},
              'state_sha256': {key: array_hash(value) for key, value in state.items()},
              'profile_weights_reused_as_full_member': False, 'quality_scores_omitted': True,
              'dev_features_gathered': False, 'dev_scores_read': False}
    artifacts = {'state': state, 'probability': probability, 'raw': raw, 'tiers': tiers}
    return result, artifacts


def compare_stage3(original, original_artifacts, cached, cached_artifacts, tolerances=STAGE3_TOLERANCES):
    """Fixed-tolerance comparison of selected state, training history and delivery outputs."""
    if original['path'] != 'original' or cached['path'] != 'cached':
        raise ValueError('Original and cached stage3 results required in that order')
    if (original['long_length'] != cached['long_length'] or original['device'] != cached['device']
            or original['budget'] != cached['budget'] or original['width'] != cached['width']):
        raise ValueError('Paired fits differ in arm, device, width or budget')
    counts = {key: original[key] == cached[key] for key in ('epochs_run', 'best_epoch', 'optimizer_updates')}
    counts['warmup_updates'] = original['warmup']['optimizer_updates'] == cached['warmup']['optimizer_updates']
    history = {}
    if len(original['history']) != len(cached['history']):
        history['pass'] = False
    else:
        for key in ('train_weighted_nll', 'earlystop_conditional_nll'):
            history[key] = _difference([row[key] for row in original['history']], [row[key] for row in cached['history']],
                                       tolerances['history_nll'])
        history['pass'] = all(history[key]['pass'] for key in ('train_weighted_nll', 'earlystop_conditional_nll'))
    state = _difference_map(original_artifacts['state'], cached_artifacts['state'], tolerances['selected_state'])
    temperature = _difference([original['delivery_temperature']], [cached['delivery_temperature']],
                              tolerances['delivery_temperature'])
    probability = _difference(original_artifacts['probability'], cached_artifacts['probability'], tolerances['probability'])
    raw = _difference(original_artifacts['raw'], cached_artifacts['raw'], tolerances['probability'])
    sums = _difference(np.concatenate([original_artifacts['probability'].sum(1), cached_artifacts['probability'].sum(1)]),
                       np.ones(2*len(original_artifacts['probability'])), tolerances['probability_sum'])
    tiers = bool(np.array_equal(original_artifacts['tiers'], cached_artifacts['tiers']))
    objective = _difference([original['calibration_integrated_log_loss']], [cached['calibration_integrated_log_loss']],
                            tolerances['calibration_objective'])
    equivalent = (all(counts.values()) and history['pass'] and state['pass'] and temperature['pass']
                  and probability['pass'] and raw['pass'] and sums['pass'] and tiers and objective['pass'])
    return {'protocol': 'f4_cache_audit_stage3_comparison_v1', 'long_length': original['long_length'],
            'device': original['device'], 'tolerances': deepcopy(tolerances), 'counts_equal': counts,
            'history': history, 'selected_state': state, 'delivery_temperature': temperature,
            'calibrated_probability': probability, 'raw_probability': raw, 'probability_sum': sums,
            'calibration_objective': {'original': original['calibration_integrated_log_loss'],
                                      'cached': cached['calibration_integrated_log_loss'], **objective},
            'fallback_tiers_equal': tiers, 'equivalent': bool(equivalent),
            'timing': {path: {key: result[key] for key in ('warmup_seconds', 'fit_seconds', 'temperature_seconds', 'inference_seconds')}
                       for path, result in (('original', original), ('cached', cached))},
            'interpretation': 'Paired resource-fit numerical screen; no quality claim, no full-member validity'}


def cost_summary(parent_profiles, stage3, constructions, populations, *, prior_seconds, audit_seconds,
                 member_seconds=PROFILE_SPEC['member_seconds'], family_seconds=28800, members=9):
    """Honest ledger: measured costs, frozen-formula projections, unknowns stay unknown."""
    if not (math.isfinite(prior_seconds) and prior_seconds >= 0 and math.isfinite(audit_seconds) and audit_seconds >= 0):
        raise ValueError('Finite nonnegative prior and audit cost required')
    arms, lower = {}, 0.
    for cell, pair in stage3.items():
        projections = {}
        for path, result in pair.items():
            projections[path] = resource_projection(
                warmup_seconds=result['warmup_seconds'], measured_fit_seconds=result['fit_seconds'],
                measured_updates=result['optimizer_updates'], full_train_rows=populations['train'],
                full_earlystop_rows=populations['earlystop'], temperature_seconds=result['temperature_seconds'],
                temperature_rows=populations['temperature'], inference_seconds=result['inference_seconds'],
                blend_rows=populations['blend'], dev_rows=populations['dev'], load_seconds=result['load_seconds'])
        cached = projections['cached']['load_fit_temperature_prediction_seconds']
        lower += members/len(stage3)*cached
        arms[cell] = {'original_v3_profile_projection_seconds': parent_profiles[cell]['projection']['load_fit_temperature_prediction_seconds'],
                      'audit_original_projection_seconds': projections['original']['load_fit_temperature_prediction_seconds'],
                      'audit_cached_projection_seconds_excluding_cache_construction': cached,
                      'projections': projections,
                      'measured_cache_construction': constructions[cell],
                      'full_population_cache_construction_seconds': {'fit_process': 'unmeasured', 'prediction_process': 'unmeasured'},
                      'cached_member_total_seconds': None,
                      'cached_member_within_limit': None,
                      'note': 'Two processes per member each build their own cache; the full-population build cost is not measured by this audit and is not extrapolated'}
    projection = prior_seconds+audit_seconds+lower
    return {'protocol': 'f4_cache_audit_cost_summary_v2', 'members': members, 'full_epochs': PROFILE_SPEC['full_epochs'],
            'prior_seconds': prior_seconds, 'audit_seconds': audit_seconds, 'arms': arms,
            'family_projection_excluding_cache_construction_seconds': projection,
            'family_projection_gate_excluding_cache_construction': projection <= family_seconds,
            'projection_meaning': 'Frozen-formula model projection plus ledgers, excluding unmeasured cache costs; not a guaranteed or minimum physical runtime',
            'family_projection_seconds': None, 'family_within_budget': None,
            'member_limit_seconds': member_seconds, 'family_seconds': family_seconds,
            'decision': 'not_adopted',
            'reason': ('Registered projection gate excluding cache construction fails' if projection > family_seconds else
                       'Full-population cache construction for fit and prediction processes is unmeasured; root must time it before any adoption')}


def validate_prior_ledger(ledger, expected_preparation_sha256, hash_file):
    """Owner ledger v2: five categories, unique evidence-backed finite entries, exact total."""
    required = {'protocol', 'preparation_sha256', 'entries', 'elapsed_seconds_total'}
    optional = {'preparation_path', 'category_seconds', 'entry_count'}
    if not isinstance(ledger, dict) or not required <= set(ledger) or set(ledger)-required-optional:
        raise ValueError('Owner ledger must carry protocol/preparation_sha256/entries/elapsed_seconds_total')
    if ledger['protocol'] != PRIOR_LEDGER_PROTOCOL: raise ValueError('Owner ledger protocol differs')
    if ledger['preparation_sha256'] != expected_preparation_sha256: raise ValueError('Owner ledger belongs to another parent preparation')
    entries = ledger['entries']
    if not isinstance(entries, list) or not entries: raise ValueError('Owner ledger needs entries')
    if {row.get('category') for row in entries} != PRIOR_LEDGER_CATEGORIES:
        raise ValueError('Owner ledger must cover exactly preparation/profiles/cold_failed_attempts/equivalence/cache_audit')
    attempts, by_category = set(), {name: 0. for name in PRIOR_LEDGER_CATEGORIES}
    for row in entries:
        fields = set(row)
        if not {'category', 'attempt', 'seconds', 'evidence'} <= fields or fields-{'category', 'attempt', 'seconds', 'evidence', 'step', 'exit_code'}:
            raise ValueError('Owner ledger entry fields differ')
        seconds = row['seconds']
        if (not isinstance(row['attempt'], str) or not row['attempt'] or row['attempt'] in attempts
                or not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('Distinct nonempty attempts with finite nonnegative seconds required')
        attempts.add(row['attempt'])
        if not isinstance(row['evidence'], list) or not row['evidence']: raise ValueError('Owner ledger entry needs evidence')
        for evidence in row['evidence']:
            if set(evidence) != {'path', 'sha256'} or hash_file(evidence['path']) != evidence['sha256']:
                raise ValueError('Owner ledger evidence changed: ' + str(evidence.get('path')))
        by_category[row['category']] += seconds
    total = math.fsum(row['seconds'] for row in entries)
    if not math.isclose(total, ledger['elapsed_seconds_total'], rel_tol=0, abs_tol=1e-6):
        raise ValueError('Owner ledger total differs from its entries')
    if 'entry_count' in ledger and ledger['entry_count'] != len(entries): raise ValueError('Owner ledger entry count differs')
    if 'category_seconds' in ledger and (set(ledger['category_seconds']) != PRIOR_LEDGER_CATEGORIES or any(
            not math.isclose(ledger['category_seconds'][name], by_category[name], rel_tol=0, abs_tol=1e-6) for name in PRIOR_LEDGER_CATEGORIES)):
        raise ValueError('Owner ledger category totals differ')
    return total
