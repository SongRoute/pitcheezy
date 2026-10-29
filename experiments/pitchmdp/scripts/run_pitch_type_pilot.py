#!/usr/bin/env python3
"""Explicit registered runner: exclusive output, CPU limits, exclusive task CPU lock."""
import argparse
from contextlib import ExitStack
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pitchmdp.pitch_type_pilot import digest, evaluate_frame, load_registered, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--cpu-lock', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.config) != args.config_sha256: raise ValueError('configuration pin mismatch')
    c = json.loads(args.config.read_text()); validate_config(c)
    args.output.mkdir(parents=False, exist_ok=False)
    started = time.time()
    resource.setrlimit(resource.RLIMIT_AS, (c['budget']['address_space_gib']*1024**3,)*2)
    signal.alarm(c['budget']['wall_seconds'])
    with ExitStack() as stack:
        lock = stack.enter_context(open(args.cpu_lock, 'a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, ledger, audit = evaluate_frame(load_registered(args.input, c), c)
        result['provenance'] = {'source_commit': args.source_commit, 'config_sha256': args.config_sha256,
            'data_sha256': c['source']['sha256'], 'python': os.sys.version.split()[0],
            'packages': {name: importlib.metadata.version(name) for name in ('numpy','pandas','pyarrow','scikit-learn','scipy')},
            'source_sha256': {str(path): digest(path) for path in sorted(Path('experiments/pitchmdp').rglob('*.py'))},
            'elapsed_seconds': time.time()-started,
            'max_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        with (args.output/'exclusions.jsonl').open('x') as f:
            for row in ledger: f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        audit.to_parquet(args.output/'predictions.parquet', index=False)
        with (args.output/'metrics.json').open('x') as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
        print(json.dumps({'id':c['id'],'completed':True,'elapsed_seconds':result['provenance']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
