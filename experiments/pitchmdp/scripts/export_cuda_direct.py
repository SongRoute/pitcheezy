"""DRAFT Mac-side exporter and scorer for DRAFT-CUDA-DIRECT-20261004.

export  Reuses the frozen loaders unchanged (run_ml_delivery_pool.load = verify_parent
        + load_parent + guard_dates; run_ml_direct_recency.verify_control;
        run_ml_direct_head.direct_arrays + matrix_sharing.training_arrays;
        run_ml_direct_workload.read_raw + workload_columns) and writes the exact
        EXP-P15-001 direct-head inputs per split as plain .npy arrays plus a SHA256
        manifest with lineage.  Nothing is learned here: token standardization,
        batter archetypes and pitcher profile come from the TRAIN-only EXP-P4-001
        preparation (pinned).  The output directory must not exist; it is written
        under a temporary name, each array atomically, then renamed.
estimate  Byte estimate from shapes and frozen sample counts (metadata only).
score   After all priority-1 CUDA tasks are pulled back: frozen recency.describe
        (June frequency blend, paired whole-game bootstrap, fixed 1/2 G0 mix) with
        the CUDA control as the uniform arm.  Descriptive repeated-DEV only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np

import run_cuda_direct as cuda

FORMAT = 'cuda_direct_export_v1'
DTYPES = {'tokens': 'float32', 'valid': 'bool', 'context': 'float32', 'workload': 'float32', 'keys': 'int64',
          'y': 'int64', 'game_pk': 'int64', 'pitcher': 'int64', 'date': 'int32'}
SOURCES = ['scripts/export_cuda_direct.py', 'scripts/run_cuda_direct.py', 'scripts/run_ml_direct_workload.py',
           'scripts/run_ml_direct_head.py', 'scripts/run_ml_delivery_pool.py', 'pitchmdp/workload_features.py',
           'pitchmdp/matrix_sharing.py']


def row_bytes(shapes):
    length, n_token, n_context, n_workload = (shapes[k] for k in ('length', 'n_token', 'n_context', 'n_workload'))
    return length * n_token * 4 + length + n_context * 4 + n_workload * 4 + 3 * 8 + 3 * 8 + 4


def estimate_bytes(counts, shapes):
    per = row_bytes(shapes)
    by_split = {split: int(n) * per for split, n in counts.items()}
    return {'row_bytes': per, 'by_split': by_split, 'total': sum(by_split.values())}


def check_arrays(split, arrays, shapes, columns_n):
    n = len(arrays['y'])
    expected = {'tokens': (n, shapes['length'], shapes['n_token']), 'valid': (n, shapes['length']),
                'context': (n, shapes['n_context']), 'workload': (n, columns_n), 'keys': (n, 3), 'y': (n,),
                'game_pk': (n,), 'pitcher': (n,), 'date': (n,)}
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape or str(arrays[name].dtype) != DTYPES[name]:
            raise ValueError(f'{split}/{name}: {arrays[name].shape} {arrays[name].dtype} differs from {shape} {DTYPES[name]}')
    if not np.isfinite(arrays['tokens']).all() or not np.isfinite(arrays['context']).all() \
            or not np.isfinite(arrays['workload']).all():
        raise ValueError(f'{split}: nonfinite input')
    if np.any(arrays['tokens'][:, -1, :cuda.N_PHYSICAL] != 0):
        raise ValueError(f'{split}: current physics not zero')
    if not np.isin(arrays['y'], range(10)).all():
        raise ValueError(f'{split}: labels outside the ten-class contract')
    if not np.array_equal(arrays['keys'][:, 0], arrays['game_pk']):
        raise ValueError(f'{split}: key and game metadata disagree')
    if n and int(arrays['date'].max()) // 10000 > 2025:
        raise ValueError(f'{split}: 2026 row')


def write_export(output, splits, lineage, shapes, columns, expect_bytes=None):
    """Write {split: {array: ndarray}} as .npy + manifest; refuse an existing output."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} exists; exports are immutable')
    counts = {s: len(a['y']) for s, a in splits.items()}
    estimate = estimate_bytes(counts, shapes)
    if expect_bytes is not None and estimate['total'] != int(expect_bytes):
        raise ValueError(f'Export size {estimate["total"]} differs from the expected {expect_bytes}')
    for split, arrays in splits.items():
        check_arrays(split, arrays, shapes, len(columns))
    work = output.with_name(output.name + f'.tmp-{os.getpid()}')
    work.mkdir(parents=True)
    manifest = {'format': FORMAT, 'shapes': dict(shapes), 'workload_columns': list(columns),
                'estimate_bytes': estimate, 'lineage': lineage, 'splits': {},
                'created_utc': datetime.now(timezone.utc).isoformat()}
    for split, arrays in splits.items():
        (work / split).mkdir()
        files = {}
        for name in DTYPES:
            rel = f'{split}/{name}.npy'
            tmp = work / f'{rel}.tmp'
            with open(tmp, 'wb') as handle:
                np.save(handle, np.ascontiguousarray(arrays[name]), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, work / rel)
            files[name] = {'path': rel, 'shape': list(arrays[name].shape), 'dtype': DTYPES[name],
                           'sha256': cuda.sha256_file(work / rel)}
        dates = arrays['date']
        manifest['splits'][split] = {
            'n': counts[split], 'files': files,
            'rows_sha256': hashlib.sha256(np.ascontiguousarray(arrays['keys'], dtype=np.int64).tobytes()).hexdigest(),
            'games': int(len(np.unique(arrays['game_pk']))),
            'min_date': int(dates.min()) if len(dates) else None, 'max_date': int(dates.max()) if len(dates) else None,
            'max_year': int(dates.max()) // 10000 if len(dates) else 0}
    cuda.write_json_atomic(work / 'manifest.json', manifest)
    if output.exists():
        raise FileExistsError(f'{output} appeared during export')
    os.rename(work, output)
    return cuda.sha256_file(output / 'manifest.json')


# ------------------------------------------------------------------ real export (root runs; heavy)

def collect(cuda_config, local_path):
    """Frozen-loader arrays for every split; the only path that reads real data."""
    import pandas as pd
    from pitchmdp.data import KEY, hash_file
    from pitchmdp.matrix_data import ordered_key_hash
    from pitchmdp.matrix_sharing import training_arrays
    from pitchmdp.model import outcome_labels
    from run_ml_benchmark import read_json
    from run_ml_delivery_pool import load
    import run_ml_direct_head as direct
    import run_ml_direct_recency as recency
    import run_ml_direct_workload as workload
    source = cuda_config['source_configs']['workload']
    if hash_file(REPO / source['path']) != source['sha256']:
        raise ValueError('Workload source config changed')
    wconfig = workload.config_check(read_json(REPO / source['path']), smoke=True)  # schema only; nothing is run
    if list(cuda_config['export']['workload_columns']) != list(workload.COLUMNS):
        raise ValueError('Workload columns differ from run_ml_direct_workload')
    local = read_json(local_path)
    recency.verify_control(wconfig)
    parent, pprep, store, context, parts, _ = load(local, wconfig)
    values = workload.workload_columns(workload.read_raw(local), store.frame).astype(np.float32)
    splits = {}
    for split in cuda_config['export']['splits']:
        part = parts[split]
        if ordered_key_hash(part) != pprep['samples'][split]['rows_sha256']:
            raise ValueError(f'{split}: frozen sample identity differs')
        rows = part.index.to_numpy()
        tokens, valid, ctx = training_arrays(direct.direct_arrays(store, context, rows))
        dates = pd.to_datetime(part.game_date)
        splits[split] = {'tokens': np.asarray(tokens, dtype=np.float32), 'valid': np.asarray(valid, dtype=bool),
                         'context': np.asarray(ctx, dtype=np.float32), 'workload': values[rows],
                         'keys': part[KEY].to_numpy(np.int64), 'y': outcome_labels(part).astype(np.int64),
                         'game_pk': part.game_pk.to_numpy(np.int64), 'pitcher': part.pitcher.to_numpy(np.int64),
                         'date': (dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day).to_numpy(np.int32)}
    train_rows = parts['train'].index.to_numpy()
    lineage = {'parent_run': str(parent), 'parent_preparation_sha256': wconfig['parent_preparation_sha256'],
               'control_run': wconfig['control']['run'], 'control_preparation_sha256': wconfig['control']['preparation_sha256'],
               'workload_config_sha256': source['sha256'],
               'raw_file_sha256': workload.raw_hashes(local), 'direct_source_sha256': direct.source_hashes(),
               'export_source_sha256': {rel: hash_file(PROJECT / rel) for rel in SOURCES},
               'git_commit': _git_commit(), 'normalization': cuda_config['export']['normalization'],
               'train_workload_report': workload.feature_report(values, train_rows),
               'samples': {s: pprep['samples'][s] for s in cuda_config['export']['splits']}}
    return splits, lineage


def _git_commit():
    import subprocess
    try:
        head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
        return {'head': head, 'dirty': bool(dirty.strip())}
    except Exception as error:  # recorded, never silently assumed
        return {'error': repr(error)}


# ------------------------------------------------------------------ Mac scoring (after pull)

def load_members(root, arm, seeds):
    members = []
    for seed in seeds:
        dest = Path(root) / arm / str(seed)
        cuda.verify_task(dest)
        with np.load(dest / 'predictions.npz', allow_pickle=False) as data:
            members.append({k: data[k].copy() for k in data.files})
    return members


def score(cuda_config, pulled, destination):
    """Descriptive repeated-DEV comparison of every complete arm vs the CUDA control."""
    from run_ml_benchmark import read_json
    from run_ml_delivery_pool import members
    import run_ml_direct_head as direct
    import run_ml_direct_recency as recency
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Preserve completed or interrupted scoring output')
    seeds = cuda_config['seeds']
    first = [a['name'] for a in cuda_config['arms'] if a['priority'] == 1]
    for arm in first:
        for seed in seeds:
            if not (Path(pulled) / arm / str(seed) / 'state.json').is_file():
                raise ValueError('Incomplete family: every priority-1 arm and seed must finish before any comparison')
    wconfig = read_json(REPO / cuda_config['source_configs']['workload']['path'])
    parent, _ = direct.verify_parent(wconfig)
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as data:
        baseline = {k: data[k].copy() for k in data.files}
    g0 = members(parent, 'members/' + direct.CELL + '/seed{seed}', 'prediction_state.json')
    control = load_members(pulled, 'control', seeds)
    result = {'family': cuda_config['experiment_id'], 'status': cuda_config['status'], 'arms': {}, 'decision': None,
              'control': 'CUDA control (same device, no workload); frozen MPS EXP-P15-001 not compared',
              'limits': ['Descriptive repeated-DEV comparison; Cpanel DEV already exposed',
                         'No winner selection; priority-2 arms are reported only when all their seeds completed',
                         'Prediction only; no policy claim; no 2026 rows']}
    for arm in [a['name'] for a in cuda_config['arms'] if a['name'] != 'control']:
        if not all((Path(pulled) / arm / str(s) / 'state.json').is_file() for s in seeds):
            result['arms'][arm] = 'incomplete: not compared'
            continue
        result['arms'][arm] = recency.describe(load_members(pulled, arm, seeds), control, g0, baseline,
                                               wconfig['mix']['direct_weight'])
    destination.mkdir(parents=True)
    cuda.write_json_atomic(destination / 'descriptive.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--local-config', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True, help='new directory; must not exist')
    e.add_argument('--expect-bytes', type=int, help='refuse if the computed size differs')
    s = sub.add_parser('estimate')
    s.add_argument('--parent-preparation', type=Path, required=True, help='EXP-P4-001/preparation.json (metadata only)')
    c = sub.add_parser('score')
    c.add_argument('--pulled', type=Path, required=True, help='pulled CUDA family directory (arm/seed/...)')
    c.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    config = cuda.config_check(cuda.read_json(args.config))
    shapes, columns = config['export']['shapes'], config['export']['workload_columns']
    if args.command == 'estimate':
        samples = cuda.read_json(args.parent_preparation)['samples']
        print(json.dumps(estimate_bytes({s: samples[s]['n'] for s in config['export']['splits']}, shapes), indent=2))
    elif args.command == 'export':
        from run_ml_matrix import check_location, heavy_lock
        output = args.output.resolve()
        root = check_location(cuda.read_json(args.local_config), output)  # under runs/ML-MATRIX-20260924
        if output.exists():
            raise FileExistsError(f'{output} exists; exports are immutable')
        with heavy_lock(root):  # same single Mac heavy queue as the matrix runners
            splits, lineage = collect(config, args.local_config)
            lineage['cuda_config_sha256'] = cuda.sha256_file(args.config)
            digest = write_export(output, splits, lineage, shapes, columns, args.expect_bytes)
        print('CUDA_EXPORT_COMPLETE', output, 'manifest_sha256', digest, flush=True)
    else:
        result = score(config, args.pulled, args.destination)
        print('CUDA_DESCRIBED', {a: (r if isinstance(r, str) else
                                     {k: r[k]['after_june_blend']['paired']['nll'] for k in ('direct', 'mix')})
                                 for a, r in result['arms'].items()}, flush=True)


if __name__ == '__main__':
    main()
