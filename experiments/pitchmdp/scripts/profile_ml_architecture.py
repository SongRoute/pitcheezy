"""Bounded synthetic CPU/MPS feasibility profile; acquires the shared heavy lock.

No source data, labels, DEV scores, or checkpoints are read. Invoke only through
the sole heavy execution owner before scheduling full architecture fits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import torch

from pitchmdp.matrix_models import MatrixModel
from pitchmdp.data import hash_file
from pitchmdp.matrix_benchmark import CELLS, integrated_probabilities
from run_ml_matrix import heavy_lock, check_location
from run_ml_benchmark import dump, validate_native_runtime


def input_memory_bytes(rows, length, token_channels, context_channels=28):
    return int(rows * (length * (token_channels * 4 + 1) + context_channels * 4 + 8 + 4))


def profile(kind, rows, device, token_channels, width=128):
    if not 20 <= rows <= 8192 or token_channels < 20 or device not in ('cpu', 'mps'):
        raise ValueError('Synthetic profiles require 20..8192 rows, enriched tokens, explicit CPU/MPS')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    if kind == 'lightgbm':
        validate_native_runtime()
    rng = np.random.default_rng(20260924)
    tokens = rng.normal(size=(rows, 6, token_channels)).astype(np.float32)
    tokens[:, -1, -11:] = 0.
    valid = np.ones((rows, 6), dtype=bool)
    valid[:rows // 2, :2] = False
    tokens[~valid] = 0.
    context = rng.normal(size=(rows, 28)).astype(np.float32)
    labels = np.arange(rows, dtype=np.int64) % 10
    arrays = tokens, valid, context
    model = MatrixModel(kind, seed=0, width=width, device=device,
                        lightgbm_params={'num_leaves': 31, 'min_data_in_leaf': 100, 'lambda_l2': 1.})
    start = time.perf_counter()
    model.fit(arrays, labels, arrays, labels, epochs=2, patience=2, batch_size=256,
              learning_rate=.05 if kind == 'lightgbm' else .0005)
    if device == 'mps':
        torch.mps.synchronize()
    fit_seconds = time.perf_counter() - start
    # One 400-draw query block: no large real-data arrays or long histories.
    query_pitches = min(8, rows)
    draw_arrays = tuple(np.repeat(value[:query_pitches], 400, axis=0) for value in arrays)
    start = time.perf_counter()
    logits = model.logits(draw_arrays).reshape(query_pitches, 400, 10)
    p, raw = integrated_probabilities(logits, 1.)
    if device == 'mps':
        torch.mps.synchronize()
    inference_seconds = time.perf_counter() - start
    return {'kind': kind, 'device_requested': device, 'model_report': model.report,
            'synthetic_only': True, 'source_sha256': hash_file(Path(__file__)),
            'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'rows': rows, 'epochs': 2, 'batch_size': 256,
            'fit_seconds': fit_seconds, 'query_pitches': query_pitches, 'draws': 400,
            'inference_seconds': inference_seconds,
            'inference_seconds_per_pitch': inference_seconds / query_pitches,
            'max_probability_sum_error': float(np.abs(p.sum(1) - 1).max()),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_current_allocated_bytes': int(torch.mps.current_allocated_memory()) if device == 'mps' else None,
            'mps_driver_allocated_bytes': int(torch.mps.driver_allocated_memory()) if device == 'mps' else None,
            'synthetic_input_bytes': input_memory_bytes(rows, 6, token_channels),
            'projection_rule': 'fit time scaling is a rough screen; add data load/CAL/inference/save; validate real profile before a 7200-second run',
            'not_a_quality_result': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New profile directory under the matrix protocol root')
    parser.add_argument('--cell', choices=list(CELLS), required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--token-channels', type=int, default=37)
    args = parser.parse_args()
    local = json.loads(args.local_config.read_text())
    output = args.output.resolve()
    root = check_location(local, output)
    with heavy_lock(root):
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        result = profile(CELLS[args.cell], args.rows, args.device, args.token_channels)
        result['seconds_total'] = time.perf_counter() - started
        dump(output / 'profile.json', result)
        print('PROFILE_COMPLETE', args.cell, output, flush=True)


if __name__ == '__main__':
    main()
