#!/usr/bin/env python3
"""Explicit registered runner: exclusive output, CPU limits, shared GPU locks."""
import argparse
from contextlib import ExitStack
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pitchmdp.pitch_type_pilot import digest, evaluate_frame, load_registered, validate_config

GPU_IDS = ['GPU-edb6bc12-c9aa-d79b-d3e7-9bc28118e4d4', 'GPU-0cc63040-5522-7fea-20d8-93b9196ce019']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--lock-dir', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.config) != args.config_sha256: raise ValueError('configuration pin mismatch')
    c = json.loads(args.config.read_text()); validate_config(c)
    args.output.mkdir(parents=False, exist_ok=False)
    started = time.time()
    resource.setrlimit(resource.RLIMIT_AS, (c['budget']['address_space_gib']*1024**3,)*2)
    signal.alarm(c['budget']['wall_seconds'])
    with ExitStack() as stack:
        for uuid in sorted(GPU_IDS):
            lock = stack.enter_context(open(args.lock_dir / f'{uuid}.lock', 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        mapping = dict(line.replace(' ', '').split(',') for line in observed.strip().splitlines())
        if [mapping.get(str(i)) for i in (0, 1)] != GPU_IDS: raise RuntimeError('GPU index/UUID changed')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if any(uuid in apps for uuid in GPU_IDS): raise RuntimeError('allocated GPU has active compute process')
        result, ledger, audit = evaluate_frame(load_registered(args.input, c), c)
        result['provenance'] = {'source_commit': args.source_commit, 'config_sha256': args.config_sha256,
            'data_sha256': c['source']['sha256'], 'python': os.sys.version.split()[0],
            'packages': {name: importlib.metadata.version(name) for name in ('numpy','pandas','pyarrow','scikit-learn','scipy')},
            'source_sha256': {str(path.relative_to(Path.cwd())): digest(path) for path in sorted(Path('experiments/pitchmdp').rglob('*.py'))},
            'elapsed_seconds': time.time()-started}
        with (args.output/'exclusions.jsonl').open('x') as f:
            for row in ledger: f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        audit.to_parquet(args.output/'predictions.parquet', index=False)
        with (args.output/'metrics.json').open('x') as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
        print(json.dumps({'id':c['id'],'completed':True,'elapsed_seconds':result['provenance']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
