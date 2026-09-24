"""Bounded real TRAIN/temperature-only architecture profile, never DEV scores."""
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
from pitchmdp.data import hash_file
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_benchmark import CELLS, validate_config, predict_streamed
from pitchmdp.model import outcome_labels
from run_ml_matrix import heavy_lock, check_location
from run_ml_benchmark import identity, verify, load_data, read_json, dump, validate_native_runtime
from run_sequence_pilot import arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell', choices=list(CELLS), required=True)
    args = parser.parse_args()
    config, local = validate_config(read_json(args.config)), read_json(args.local_config)
    root = check_location(local, args.output)
    validate_native_runtime()
    with heavy_lock(root):
        if args.output.exists():
            raise ValueError('Preserve previous profile output')
        args.output.mkdir(parents=True)
        started = time.perf_counter()
        prep = verify(args.run, identity(config, args.local_config))
        store, parts, aux = load_data(local, args.run, prep)
        train = parts['train'].iloc[:8192]
        early = parts['earlystop'].iloc[:2048]
        model = MatrixModel(CELLS[args.cell], seed=0, width=128,
                            lightgbm_params=config['lightgbm']['params'])
        ta = arrays(store, aux['context'], train.index.to_numpy())
        ea = arrays(store, aux['context'], early.index.to_numpy())
        ready = time.perf_counter()
        model.fit(ta, outcome_labels(train), ea, outcome_labels(early), epochs=2, patience=2,
                  batch_size=1024, learning_rate=.05 if model.kind == 'lightgbm' else .0005)
        fit_seconds = time.perf_counter() - ready
        query = parts['temperature'].iloc[:64]
        infer_start = time.perf_counter()
        p, raw, tiers = predict_streamed(model, aux['delivery'], store, aux['context'], query.index.to_numpy())
        infer_seconds = time.perf_counter() - infer_start
        max_epochs = 300 if model.kind == 'lightgbm' else 30
        report = {'cell': args.cell, 'real_profile_only': True, 'dev_scores_read': False,
            'train_rows': len(train), 'earlystop_rows': len(early), 'query_rows': len(query),
            'fit_seconds': fit_seconds, 'inference_seconds': infer_seconds,
            'load_arrays_seconds': ready - started, 'total_seconds': time.perf_counter() - started,
            'rough_full_fit_linear_projection_seconds': fit_seconds * len(parts['train']) / len(train) * max_epochs / 2,
            'projection_limit': 'Small-sample linear approximation; not a guaranteed bound, tree scaling and startup differ.',
            'model_report': model.report, 'token_shape': list(ta[0].shape),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'maximum_mass_error': float(np.abs(p.sum(1) - 1).max()),
            'preparation_sha256': hash_file(args.run / 'preparation.json'),
            'source_sha256': hash_file(Path(__file__))}
        dump(args.output / 'profile.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
