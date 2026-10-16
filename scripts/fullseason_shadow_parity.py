"""Registered <=100-query CPU implementation parity, never model evaluation."""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from experiments.ws_fullseason.audit import gates, sha, checked_json
from experiments.ws_fullseason.calibrate_worker import header_check, file_digest
from experiments.ws_fullseason.data import key_digest

ATOL = 1e-7
RTOL = 1e-6
QUERY_CAP = 100
ROW_CAP = 720505
FEATURES = 199
LENGTH = 32
SEQ_MEMBERS = {'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter',
               'pitch_type', 'outcome10', 'balls', 'strikes', 'stand'}
SELECTED_MEMBERS = SEQ_MEMBERS - {'stand'}


def fixed_indices(rows):
    if type(rows) is not int or not 1 <= rows <= ROW_CAP:
        raise ValueError('row bound')
    count = min(QUERY_CAP, rows)
    return np.array([k * (rows - 1) // max(1, count - 1) for k in range(count)], np.int64)


def query_context(s, queries):
    """Only 100 query rows; O(100*N) avoids an N*32 full context allocation."""
    out = np.full((len(queries), LENGTH), -1, np.int32)
    p, g, a, n = (s[k] for k in ('pitcher', 'game_pk', 'at_bat_number', 'pitch_number'))
    for row, i in enumerate(queries):
        earlier = np.flatnonzero((p == p[i]) & (g == g[i]) &
            ((a < a[i]) | ((a == a[i]) & (n < n[i]))))
        order = np.lexsort((n[earlier], a[earlier]))[::-1]
        chosen = earlier[order[:LENGTH]]
        out[row, :len(chosen)] = chosen
    return out


def make_tokens(s, queries, ctx):
    valid = ctx >= 0; safe = np.maximum(ctx, 0)
    tokens = {'valid': valid}
    for target, source in [('type', 'pitch_type'), ('out', 'outcome10'), ('balls', 'balls'), ('strikes', 'strikes')]:
        tokens[target] = np.where(valid, s[source][safe], 0).astype(np.int64)
    tokens['same'] = ((s['batter'][safe] == s['batter'][queries, None]) & valid).astype(np.int64)
    tokens['pos'] = np.broadcast_to(np.arange(1, LENGTH + 1), ctx.shape).copy()
    tokens['context_keys'] = np.stack([s[n][safe] for n in ('game_pk', 'at_bat_number', 'pitch_number')], axis=-1).astype(np.int64)
    tokens['context_pitcher'] = s['pitcher'][safe].astype(np.int64)
    tokens['query_pitcher'] = s['pitcher'][queries].astype(np.int64)
    return tokens


def reference_subset(x, s, queries, ctx):
    """Remap every original query/context index; no selected query loses history."""
    selected = np.unique(np.concatenate((queries, ctx[ctx >= 0])))
    local_queries = np.searchsorted(selected, queries)
    local_ctx = np.full((len(selected), LENGTH), -1, np.int32)
    valid = ctx >= 0
    mapped = np.full_like(ctx, -1); mapped[valid] = np.searchsorted(selected, ctx[valid])
    if not np.array_equal(selected[mapped[valid]], ctx[valid]):
        raise ValueError('context remap')
    local_ctx[local_queries] = mapped
    local_x = np.zeros((len(selected), FEATURES), np.float32)
    local_x[local_queries] = x
    local_s = {name: values[selected] for name, values in s.items()}
    keys = np.column_stack([local_s[n] for n in ('game_pk', 'at_bat_number', 'pitch_number')]).astype(np.int64)
    # base.Data's training-label fields are unused by predict; never open corpus labels.
    c = {'y10': np.zeros(len(selected), np.int64), 'narrow_whiff': np.zeros(len(selected)),
         'ground_ball_per_pitch': np.zeros(len(selected))}
    return local_x, c, local_s, local_ctx, keys, local_queries


def compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all() or not np.allclose(actual, expected, atol=ATOL, rtol=RTOL)):
        raise ValueError('frozen parity tolerance')
    return float(np.max(np.abs(actual - expected), initial=0))


def pinned(binding, stack, cap):
    p = Path(binding['path']); size = binding['bytes']
    if type(size) is not int or not 0 < size <= cap or not stat.S_ISREG(p.lstat().st_mode):
        raise ValueError('regular nonsymlink bounded input')
    stream = stack.enter_context(p.open('rb'))
    if os.fstat(stream.fileno()).st_size != size or file_digest(stream) != binding['sha256']:
        raise ValueError('input byte custody')
    return stream


def reference_state(binding):
    import torch
    from pitcheezy.seq_shadow.fullseason import read_artifact
    raw = read_artifact(binding, 2 << 20)
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def unchanged(stream, binding):
    if os.fstat(stream.fileno()).st_size != binding['bytes'] or file_digest(stream) != binding['sha256']:
        raise ValueError('input changed during parity')


def feature_view(stream, rows, size):
    version = np.lib.format.read_magic(stream)
    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers: raise ValueError('NPY version')
    shape, fortran, dtype = readers[version](stream, max_header_size=10000)
    offset = stream.tell()
    if shape != (rows, FEATURES) or dtype != np.dtype('float32') or offset + math.prod(shape) * 4 != size:
        raise ValueError('encoded feature header')
    return np.memmap(stream, dtype=dtype, mode='r', offset=offset, shape=shape, order='F' if fortran else 'C')


def run(regpath, grantpath):
    reg, recipe = gates(regpath, grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    import torch
    from experiments.ws_seq import train_seq as base
    from experiments.ws_campaign import train as campaign
    from experiments.ws_campaign.models import build_model
    from pitcheezy.seq_shadow import fullseason
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    rows = reg['expected_rows']; queries = fixed_indices(rows)
    if rows != ROW_CAP or reg['query_rule'] != 'floor(k*(N-1)/99), k=0..99' or reg['tolerance'] != {'atol': ATOL, 'rtol': RTOL}:
        raise ValueError('frozen parity rule')
    if reg['adapter_config'] not in reg['source_pins']:
        raise ValueError('pinned adapter configuration required')
    config = checked_json(reg['adapter_config'])
    if config['recipe']['path'] != reg['recipe'] or config['recipe']['sha256'] != sha(reg['recipe']):
        raise ValueError('same exact reference and adapter recipe')
    with ExitStack() as stack:
        # Whole x and sequence bytes hash before headers or arrays; no corpus access.
        xf = pinned(reg['inputs']['features'], stack, 600 << 20)
        sf = pinned(reg['inputs']['sequence'], stack, 32 << 20)
        header_check(sf, SEQ_MEMBERS, {name: ((rows,), 'iu') for name in SELECTED_MEMBERS})
        with np.load(sf, allow_pickle=False) as archive:
            s = {name: archive[name] for name in SELECTED_MEMBERS}
        keys = np.column_stack([s[n] for n in ('game_pk', 'at_bat_number', 'pitch_number')]).astype(np.int64)
        if key_digest(keys) != reg['expected_keys_sha256']:
            raise ValueError('accepted ordered key identity')
        ctx = query_context(s, queries); tokens = make_tokens(s, queries, ctx)
        mapped = feature_view(xf, rows, reg['inputs']['features']['bytes'])
        x = np.array(mapped[queries], dtype=np.float32, copy=True)
        del mapped
        if not np.isfinite(x).all(): raise ValueError('finite encoded queries')
        ensemble = fullseason.load(config)
        actual = ensemble.predict(x, tokens, keys[queries])
        lx, c, ls, lc, lk, lq = reference_subset(x, s, queries, ctx)
        tensors = base.Data(torch, torch.device('cpu'), lx, c, ls, lc)
        ti = torch.as_tensor(lq, dtype=torch.long); tk = torch.as_tensor(lk, dtype=torch.long)
        reference = {}; differences = {}
        for seed in (0, 1, 2):
            model = build_model(recipe, 'T', seed).cpu().float()
            model.load_state_dict(reference_state(config['state'][str(seed)]), strict=True)
            z = campaign.predict(model, tensors, tk, ti, len(queries))
            reference[seed] = campaign.probabilities(z)
            differences[str(seed)] = compare(actual['raw_by_seed'][seed], reference[seed])
        mean = np.mean([reference[seed] for seed in (0, 1, 2)], axis=0)
        differences['raw_mean'] = compare(actual['raw_mean'], mean)
        cal = ensemble.calibrator
        logits = np.log(np.clip(mean, 1e-15, 1)) / cal['temperature']
        logits -= logits.max(axis=1, keepdims=True); calibrated = np.exp(logits); calibrated /= calibrated.sum(axis=1, keepdims=True)
        differences['calibrated'] = compare(actual['calibrated'], calibrated)
        unchanged(xf, reg['inputs']['features']); unchanged(sf, reg['inputs']['sequence'])
    receipt = {'status': 'COMPLETED', 'stage': 'CPU_IMPLEMENTATION_PARITY_ONLY',
        'registration_sha256': sha(regpath), 'grant_sha256': sha(grantpath), 'source_commit': reg['source_commit'],
        'queries': len(queries), 'query_indices_sha256': hashlib.sha256(queries.astype('<i8').tobytes()).hexdigest(),
        'selected_keys_sha256': key_digest(keys[queries]), 'historical_context_edges': int((ctx >= 0).sum()),
        'reference_subset_rows': len(lx), 'tolerance': reg['tolerance'], 'maximum_absolute_differences': differences,
        'fits': 0, 'calibration_fits': 0, 'performance_metrics': False, 'raw_probabilities_written': False,
        'same_cpu_reference_only': True, 'actual_raw91_profile_online_parity': False,
        'decode_scope': 'x selected100; nine sequence members all rows including outcome-derived historical token values; no corpus/y7 evaluation labels; full sequence outcome member includes query/current/future rows internally, but only strict-prior outcomes enter tokens',
        'data_role': '2026 EXPOSED DEVELOPMENT', 'arrival': 'UNKNOWN',
        'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
    (Path(reg['output'])/'receipt.pending.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--registration', required=True); p.add_argument('--grant', required=True)
    a=p.parse_args(); run(Path(a.registration), Path(a.grant))
