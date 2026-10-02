"""Bounded backend numerical screen for optional F4 long-encoding reuse.

Callers hold the common heavy lock and preserve immutable inputs/checkpoints.
No DEV query lists/features or model-quality ranking are accepted by this helper;
the source loader may hold the full processed frame for as-of history links.
"""
from pathlib import Path
import hashlib
import json
import resource
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import torch

from .data import KEY, hash_file
from .matrix_data import ordered_key_hash
from .matrix_lazy_model import LazyJointDelivery, LazyMatrixModel
from .matrix_long_history import LazyPitchBatch
from .model import outcome_labels

TOLERANCES = dict(logits=2e-5, fixed_probability=2e-6, probability_sum=1e-6,
                  fitted_temperature=1e-3, refitted_probability=2e-5, objective=2e-6)
LIMITS = dict(train=8192, earlystop=2048, temperature=16)
FIXED_TEMPERATURES = (.5, 1., 2.5)
UNKNOWN_ACTION = '__F4_AUDIT_UNKNOWN__'


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


def array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
    digest.update(memoryview(value)); return digest.hexdigest()


def synchronize(device):
    if device == 'mps': torch.mps.synchronize()


def memory(device):
    return dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mps_current_allocated_bytes=int(torch.mps.current_allocated_memory()) if device == 'mps' else None,
        mps_driver_allocated_bytes=int(torch.mps.driver_allocated_memory()) if device == 'mps' else None)


def weights_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.net.state_dict().items():
        digest.update(name.encode()); digest.update(array_hash(tensor.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def rng_state(device):
    numpy = np.random.get_state()
    return {'numpy': (numpy[0], array_hash(numpy[1]), *numpy[2:]),
            'torch_cpu': array_hash(torch.get_rng_state().numpy()),
            'torch_mps': array_hash(torch.mps.get_rng_state().cpu().numpy()) if device == 'mps' else None}


def objective(logits, labels, temperature):
    # Literal registered float32 conditional-logit temperature objective.
    if logits.dtype != np.float32: raise ValueError('Temperature logits must remain float32')
    p = softmax(logits/temperature, axis=-1).mean(1)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def fitted_temperature(logits, labels):
    fit = minimize_scalar(lambda t: objective(logits, labels, t), bounds=(.5, 2.5), method='bounded')
    if not fit.success or not np.isfinite(fit.x) or not np.isfinite(fit.fun):
        raise ValueError('Temperature optimizer failed')
    return float(fit.x)


def probabilities(logits, temperature):
    return softmax(logits.astype(np.float64)/temperature, axis=-1).mean(1)


def differences(reference, optimized, labels, tolerances=TOLERANCES):
    if (reference.shape != optimized.shape or reference.shape != (len(labels), 400, 10) or
            not np.isfinite(reference).all() or not np.isfinite(optimized).all()):
        raise ValueError('Finite aligned query-by400-by10 logits required')
    reference_t, optimized_t = fitted_temperature(reference, labels), fitted_temperature(optimized, labels)
    fixed, sums = {}, []
    for t in FIXED_TEMPERATURES:
        a, b = probabilities(reference, t), probabilities(optimized, t)
        fixed[str(t)] = float(np.abs(a-b).max())
        sums.extend((float(np.abs(a.sum(1)-1).max()), float(np.abs(b.sum(1)-1).max())))
    a, b = probabilities(reference, reference_t), probabilities(optimized, optimized_t)
    sums.extend((float(np.abs(a.sum(1)-1).max()), float(np.abs(b.sum(1)-1).max())))
    values = dict(logits=float(np.abs(reference-optimized).max()), fixed_probability=max(fixed.values()),
        probability_sum=max(sums), fitted_temperature=abs(reference_t-optimized_t),
        refitted_probability=float(np.abs(a-b).max()),
        objective=abs(objective(reference, labels, reference_t)-objective(optimized, labels, optimized_t)))
    return dict(max_absolute_differences=values, fixed_temperature_probability_differences=fixed,
        fitted_temperatures={'reference': reference_t, 'optimized': optimized_t},
        tolerance_pass={k: values[k] <= tolerances[k] for k in tolerances},
        equivalent=all(values[k] <= tolerances[k] for k in tolerances))


class AuditedDelivery:
    """Assert exact frozen vector/tier correspondence at every chunked sample."""
    def __init__(self, delivery, batch):
        self.delivery, self.draws = delivery, delivery.draws
        self.vectors, self.tiers = delivery.sample(batch.frame())
        self.lookup = {self.key(row): i for i, row in enumerate(batch.frame().itertuples())}
        if len(self.lookup) != len(batch): raise ValueError('Unique row/action queries required')
        self.calls = self.sampled_query_rows = 0

    @staticmethod
    def key(row): return tuple(getattr(row, name) for name in (*KEY, 'pitch_type'))

    def sample(self, frame):
        vectors, tiers = self.delivery.sample(frame)
        indices = [self.lookup[self.key(row)] for row in frame.itertuples()]
        if not np.array_equal(vectors, self.vectors[indices]) or not np.array_equal(tiers, self.tiers[indices]):
            raise ValueError('Pool vectors or fallback tiers depend on query order/chunk/path')
        self.calls += 1; self.sampled_query_rows += len(frame)
        return vectors, tiers


def timed_logits(model, delivery, batch, reuse, *, pitch_chunk=16, model_batch_size=256):
    checked = AuditedDelivery(delivery, batch)
    rng_before, weights_before = rng_state(model.device), weights_hash(model)
    counts = dict(long_token_rows=0, h5_rows=0, head_rows=0)
    def hook(name, multiplier=1):
        def count(module, args): counts[name] += len(args[0])*multiplier
        return count
    handles = [model.net.long_path.register_forward_pre_hook(hook('long_token_rows', 128)),
        model.net.h5_path.register_forward_pre_hook(hook('h5_rows')),
        model.net.head.register_forward_pre_hook(hook('head_rows'))]
    synchronize(model.device); before = memory(model.device); started = time.perf_counter()
    try:
        logits, tiers = LazyJointDelivery(checked, pitch_chunk, model_batch_size, reuse_long=reuse).logits(model, batch)
        synchronize(model.device)
        elapsed = time.perf_counter()-started
    finally:
        for handle in handles: handle.remove()
    expected = len(batch)*(1 if reuse else 400)*128
    if counts != dict(long_token_rows=expected, h5_rows=len(batch)*400, head_rows=len(batch)*400):
        raise ValueError('Network row counts differ from registered reuse/reference computation')
    if not np.array_equal(tiers, checked.tiers): raise ValueError('Output fallback-tier correspondence differs')
    if rng_state(model.device) != rng_before: raise ValueError('Inference changed a global RNG state')
    if weights_hash(model) != weights_before: raise ValueError('Inference changed shared network weights/buffers')
    return logits, tiers, dict(seconds=elapsed, network_rows=counts,
        sampled_query_rows=checked.sampled_query_rows, sample_calls=checked.calls,
        before=before, after=memory(model.device), pool_vectors_sha256=array_hash(checked.vectors),
        fallback_tiers_sha256=array_hash(checked.tiers), pitch_chunk=pitch_chunk, model_batch_size=model_batch_size,
        rng_state_unchanged=True, weights_sha256=weights_before)


def audit_batches(batches, delivery, destination, *, device='mps', seconds_limit=1200):
    """Fit one profile checkpoint, compare both paths; never accept DEV batches."""
    if set(batches) != set(LIMITS) or delivery.draws != 400:
        raise ValueError('Only TRAIN/earlystop/May-temperature and exact400 pools are accepted')
    destination = Path(destination)
    start = time.monotonic()
    def deadline():
        if time.monotonic()-start > seconds_limit: raise TimeoutError('Equivalence audit cooperative wall cap exceeded')
    selected = {}
    for name, count in LIMITS.items():
        if len(batches[name]) < count: raise ValueError('Do not silently shrink registered profile samples')
        positions = np.linspace(0, len(batches[name])-1, count, dtype=np.int64)
        selected[name] = batches[name].subset(positions)
        selected[name].frame()[[*KEY, 'game_date']].to_parquet(destination/(name+'_keys.parquet'), index=False)
    pd_dates = selected['temperature'].frame().game_date.astype('datetime64[ns]')
    if not ((pd_dates >= np.datetime64('2025-05-01')) & (pd_dates < np.datetime64('2025-06-01'))).all():
        raise ValueError('Numerical probes must use the frozen May temperature split')
    samples = {name: dict(n=len(value), rows_sha256=ordered_key_hash(value.frame())) for name, value in selected.items()}
    dump(destination/'samples.json', samples)
    synchronize(device); fit_start = time.perf_counter()
    model = LazyMatrixModel(seed=0, width=128, device=device).fit(selected['train'], outcome_labels(selected['train'].frame()),
        selected['earlystop'], outcome_labels(selected['earlystop'].frame()), epochs=2, patience=2, batch_size=256, learning_rate=.0005)
    synchronize(device)
    fit_seconds = time.perf_counter()-fit_start
    if model.report['epochs_run'] != 2 or model.report['optimizer_updates'] != 64:
        raise ValueError('Registered 8192-row/two-epoch profile computation changed')
    # Both paths use this exact network; fitted temperatures never mutate it.
    model.save(destination/'profile_checkpoint.pt')
    checkpoint_hash = hash_file(destination/'profile_checkpoint.pt')
    deadline()
    query = selected['temperature']; labels = outcome_labels(query.frame())
    reference, tiers, rt = timed_logits(model, delivery, query, False); deadline()
    optimized, other_tiers, ot = timed_logits(model, delivery, query, True); deadline()
    if not np.array_equal(tiers, other_tiers): raise ValueError('Reference/optimized fallback tiers differ')
    if rt['pool_vectors_sha256'] != ot['pool_vectors_sha256']:
        raise ValueError('Reference/optimized physical pools differ')
    result = differences(reference, optimized, labels)
    reverse = np.arange(len(query)-1, -1, -1)
    reordered, reverse_tiers, order_runtime = timed_logits(model, delivery, query.subset(reverse), True,
                                                         pitch_chunk=7, model_batch_size=257)
    deadline()
    order_difference = float(np.abs(reordered-reference[reverse]).max())
    order_probability_difference = max(float(np.abs(probabilities(reordered, t)-probabilities(reference[reverse], t)).max())
                                       for t in FIXED_TEMPERATURES)
    if not np.array_equal(reverse_tiers, tiers[reverse]): raise ValueError('Reversed query fallback correspondence differs')
    # Two fixed rows, first two TRAIN vocabulary types, plus unknown fallback.
    if len(query.store.base.type_vocabulary) < 2 or UNKNOWN_ACTION in query.store.base.type_vocabulary:
        raise ValueError('Two TRAIN types and a distinct unknown fallback sentinel required')
    names = list(query.store.base.type_vocabulary[:2])+[UNKNOWN_ACTION]
    rows = np.repeat(query.rows[:2], len(names)); actions = np.tile(names, 2)
    candidate = LazyPitchBatch(query.store, query.context, rows, candidate_pitch_types=actions)
    ar, at, art = timed_logits(model, delivery, candidate, False, pitch_chunk=3, model_batch_size=257); deadline()
    ao, bt, aot = timed_logits(model, delivery, candidate, True, pitch_chunk=3, model_batch_size=257); deadline()
    if not np.array_equal(at, bt): raise ValueError('Candidate fallback correspondence differs')
    if art['pool_vectors_sha256'] != aot['pool_vectors_sha256']:
        raise ValueError('Reference/optimized candidate physical pools differ')
    action_difference = float(np.abs(ar-ao).max())
    action_probability_difference = max(float(np.abs(probabilities(ar, t)-probabilities(ao, t)).max())
                                        for t in FIXED_TEMPERATURES)
    probe_sum_error = max(float(np.abs(probabilities(logits, t).sum(1)-1).max())
                          for logits in (reordered, ar, ao) for t in FIXED_TEMPERATURES)
    unknown = actions == UNKNOWN_ACTION
    if not (at[unknown] == -1).all(): raise ValueError('Synthetic unknown type did not use frozen fallback')
    # Preserve logits/probabilities rather than present a quality comparison.
    ref_t, opt_t = result['fitted_temperatures'].values()
    np.savez_compressed(destination/'numerics.npz', reference_logits=reference, optimized_logits=optimized,
        reference_raw=probabilities(reference, 1.), optimized_raw=probabilities(optimized, 1.),
        reference_refitted=probabilities(reference, ref_t), optimized_refitted=probabilities(optimized, opt_t),
        reversed_optimized_logits=reordered, action_reference_logits=ar, action_optimized_logits=ao,
        candidate_rows=rows, candidate_actions=actions, tiers=tiers, candidate_tiers=at, labels=labels,
        **{f'{side}_fixed_{t}': probabilities(logits, t) for side, logits in [('reference', reference), ('optimized', optimized)] for t in FIXED_TEMPERATURES})
    if hash_file(destination/'profile_checkpoint.pt') != checkpoint_hash: raise ValueError('Shared weights changed during audit')
    deadline()
    result.update(samples=samples, device=model.device, long_length=query.store.long_length,
        checkpoint_sha256=checkpoint_hash, fit_seconds=fit_seconds, seconds_total=time.monotonic()-start,
        fit_epochs=model.report['epochs_run'], fit_optimizer_updates=model.report['optimizer_updates'],
        runtime={'reference': rt, 'optimized': ot, 'reordered_optimized': order_runtime,
                 'candidate_reference': art, 'candidate_optimized': aot},
        probes={'reordered_logit_difference': order_difference, 'candidate_logit_difference': action_difference,
                'reordered_probability_difference': order_probability_difference,
                'candidate_probability_difference': action_probability_difference,
                'probability_sum_error': probe_sum_error,
                'candidate_query_rows': len(rows), 'unknown_fallback_rows': int(unknown.sum())},
        tolerances=TOLERANCES, dev_features_gathered=False, dev_scores_read=False,
        interpretation='Backend numerical screen on May16 and six candidate probes; not proof for every input or a quality comparison')
    result['equivalent'] &= (order_difference <= TOLERANCES['logits'] and action_difference <= TOLERANCES['logits']
                             and order_probability_difference <= TOLERANCES['fixed_probability']
                             and action_probability_difference <= TOLERANCES['fixed_probability']
                             and probe_sum_error <= TOLERANCES['probability_sum'])
    dump(destination/'results.json', result)
    return result
