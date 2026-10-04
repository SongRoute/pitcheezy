"""Timing-only warm steady-state TRAIN probe for the DRAFT TabM-residual family (Amendment 2).

Pre-registration: docs/AMENDMENT2-TABM-RESIDUAL-TIMING-PROBE-20261004.md (committed with this
script, before any probe run).  Purpose: replace the smoke-only projection that triggered the
pre-registered 1800 s gate (SMOKE-TABM-RESIDUAL-b9f28dc-20261004.md, kept unchanged) by a
MEASURED warm per-epoch time of every arm on the allocated GPUs, then a conservative projected
full-task and family cost.  Nothing here fits, saves, calibrates, predicts or scores a model of
the family.

Data: ONLY the TRAIN split is loaded (``load_split(..., 'train')``).  The early-stop argument
of the frozen ``fit`` loop is filled with the first ``n_earlystop`` TRAIN rows (same shape and
count as the real early-stop split, read from the manifest record ``n``), so the per-epoch
validation cost is measured without opening earlystop/temperature/blend/dev/mlb_dev.  The
post-fit prediction cost is measured as a forward pass over TRAIN rows whose count equals the
summed temperature+blend+dev+mlb_dev row counts (also from the manifest).  No loss value,
history, best epoch, logit or probability is written.

Per (GPU, arm), fixed before any run:
  warm-up   : frozen ``model.fit`` for WARMUP_EPOCHS epochs (fresh model, discarded)
  measured  : a NEW frozen ``model.fit`` for MEASURE_EPOCHS epochs (fresh model; patience 5 cannot
              trigger within 2 epochs); seconds = ``report['seconds']`` (includes net build,
              TRAIN tensor upload and per-epoch validation, so it over-states a steady epoch)
  predict   : ``model.logits`` over PREDICT rows of TRAIN after the measured fit
Same code path as the family: ``run_cuda_direct.build_model`` / ``arm_inputs`` and each arm's own
frozen ``fit``; batch size / learning rate / patience from the config ``training`` block.

CLI (server):
  timing_probe_tabm_residual.py --config C --config-sha256 S --export E --export-manifest-sha256 M \\
      --output OUT-timing --lock-dir L --gpu UUID_A --gpu UUID_B run
Each GPU gets one worker process (all 5 arms in config order), concurrently, each under the
common exclusive per-GPU flock; a busy lock refuses (nothing waits).  Per-worker hard timeout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402  numpy only at import; torch only after bind_gpu

SELF = 'scripts/timing_probe_tabm_residual.py'
QUEUE = 'scripts/cuda_direct_queue.py'
PROTOCOL = 'tabm_residual_timing_probe_v1'
FAMILY_CONFIG = 'DRAFT-CUDA-TABM-RESIDUAL-20261004'
PROBE_SEED = 1000  # not a registered seed of any family; timing does not depend on the seed
WARMUP_EPOCHS = 1
MEASURE_EPOCHS = 2
PREDICT_SPLITS = ('temperature', 'blend', 'dev', 'mlb_dev')
EPOCH_CAP = 60
TASK_GATE_SECONDS = 1800
FAMILY_BUDGET_SECONDS = 36000
FIXED_ALLOWANCE_SECONDS = 30.  # assumption: process start/imports, temperature fit, save, hashing, rename
WORKER_TIMEOUT_SECONDS = 1500
FORBIDDEN_OUTPUT_KEYS = ('history', 'best_epoch', 'train_weighted_nll', 'earlystop_conditional_nll', 'loss')


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_config(config):
    """The DRAFT TabM-residual family config only (any status); returns the arm list in config order."""
    rc.config_check(config)
    if config.get('experiment_id') != FAMILY_CONFIG or not rc.uses_tabm_residual(config):
        raise ValueError('Timing probe is defined only for the DRAFT TabM-residual family config')
    if [a['name'] for a in config['arms']] != ['control', 'periodic_residual_counts', 'tabm_counts',
                                                'tabm_residual_counts', 'tabm_counts_d10']:
        raise ValueError('Arm list differs from the pre-registered 5 arms')
    if rc.training_budget(config)['epochs'] != EPOCH_CAP or config['training']['epochs'] != EPOCH_CAP:
        raise ValueError('Epoch cap differs from the pre-registered 60')
    if PROBE_SEED in config['seeds']:
        raise ValueError('Probe seed must not be a registered seed')
    return [a['name'] for a in config['arms']]


def row_counts(manifest):
    """(n_train, n_earlystop, n_predict) from manifest records; refuses a TRAIN smaller than either stand-in."""
    splits = manifest['splits']
    n_train, n_early = int(splits['train']['n']), int(splits['earlystop']['n'])
    n_predict = sum(int(splits[s]['n']) for s in PREDICT_SPLITS)
    if not 0 < n_early <= n_train or not 0 < n_predict <= n_train:
        raise ValueError('TRAIN has fewer rows than an early-stop or prediction stand-in')
    return n_train, n_early, n_predict


def take(inputs, labels, n):
    return tuple(a[:n] for a in inputs), labels[:n]


def probe_arm(config, arm_name, train, columns, n_early, n_predict, device, seed=PROBE_SEED, sync=lambda: None,
              reset_peak=lambda: None, peak=lambda: None):
    """Warm-up fit, measured fit and TRAIN forward pass of one arm.  Returns timing fields only."""
    import numpy as np
    arm = rc.arm_spec(config, arm_name)
    spec = config['training']
    t0 = time.perf_counter()
    inputs = rc.arm_inputs(train, arm, columns)
    labels = np.array(train['y'])
    prepare = time.perf_counter() - t0
    early, early_y = take(inputs, labels, n_early)
    kwargs = dict(patience=spec['patience'], batch_size=spec['batch_size'], learning_rate=spec['learning_rate'])
    record = {'arm': arm_name, 'seed': seed, 'prepare_seconds': prepare, 'train_rows': int(len(labels)),
              'validation_standin_rows': int(n_early), 'predict_standin_rows': int(n_predict)}
    for phase, epochs in (('warmup', WARMUP_EPOCHS), ('measured', MEASURE_EPOCHS)):
        reset_peak()
        model = rc.build_model(arm, seed, device)
        sync()
        start = time.perf_counter()
        model.fit(inputs, labels, early, early_y, epochs=epochs, **kwargs)
        sync()
        wall = time.perf_counter() - start
        if rc.model_device_type(model) != device.split(':')[0]:
            raise RuntimeError('Model is not on the requested device')
        report = model.report
        if report['epochs_run'] != epochs:
            raise RuntimeError(f'{phase}: {report["epochs_run"]} epochs run, expected {epochs}')
        record[phase] = {'epochs': epochs, 'optimizer_updates': int(report['optimizer_updates']),
                         'fit_report_seconds': float(report['seconds']), 'wall_seconds': wall,
                         'peak_device_bytes': peak()}
    record['parameter_count'] = int(sum(p.numel() for p in model.net.parameters()))
    predict_inputs = tuple(a[:n_predict] for a in inputs)
    sync()
    start = time.perf_counter()
    model.logits(predict_inputs)  # values discarded
    sync()
    record['predict_seconds'] = time.perf_counter() - start
    m = record['measured']
    record['measured_seconds_per_epoch'] = m['fit_report_seconds'] / m['epochs']
    record['measured_seconds_per_update'] = m['fit_report_seconds'] / m['optimizer_updates']
    del model
    return record


def project(record, startup_seconds):
    """Conservative full-task projection at the 60-epoch cap (assumption fields named)."""
    per_epoch = record['measured_seconds_per_epoch']
    return {'startup_seconds_measured': startup_seconds, 'prepare_seconds_measured': record['prepare_seconds'],
            'fit_seconds_projected': EPOCH_CAP * per_epoch, 'predict_seconds_measured': record['predict_seconds'],
            'fixed_allowance_seconds_assumed': FIXED_ALLOWANCE_SECONDS,
            'task_seconds_projected': startup_seconds + record['prepare_seconds'] + EPOCH_CAP * per_epoch
            + record['predict_seconds'] + FIXED_ALLOWANCE_SECONDS}


def summarize(workers, seeds_per_arm):
    """Per arm: slowest GPU projection; family = sum over arms of seeds x slowest (every fit at the cap)."""
    arms = {}
    for worker in workers:
        for record in worker['arms']:
            p = project(record, worker['startup_seconds'])
            arms.setdefault(record['arm'], []).append({'gpu': worker['gpu'], **p,
                                                       'measured_seconds_per_epoch': record['measured_seconds_per_epoch']})
    names = list(arms)
    if any(len(arms[n]) != len(workers) for n in names):
        raise ValueError('Every arm must be measured on every GPU')
    slowest = {n: max(r['task_seconds_projected'] for r in arms[n]) for n in names}
    family = sum(seeds_per_arm * s for s in slowest.values())
    new = slowest.get('tabm_residual_counts')
    return {'per_arm': arms, 'slowest_task_seconds_projected': slowest, 'family_worker_seconds_projected': family,
            'tabm_residual_task_seconds_projected': new,
            'gate': {'task_seconds_max': TASK_GATE_SECONDS, 'family_seconds_max': FAMILY_BUDGET_SECONDS,
                     'tabm_residual_within': new is not None and new <= TASK_GATE_SECONDS,
                     'every_arm_within': all(s <= TASK_GATE_SECONDS for s in slowest.values()),
                     'family_within': family <= FAMILY_BUDGET_SECONDS},
            'label': 'measurement (warm TRAIN epochs, TRAIN forward pass) + projection (60 x epoch, assumptions named); '
                     'not a fit, not a score; no DEV/earlystop/temperature/blend/mlb_dev rows opened'}


def no_forbidden_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in key for word in FORBIDDEN_OUTPUT_KEYS):
                raise ValueError('Probe output carries a loss/selection field: ' + key)
            no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            no_forbidden_keys(item)
    return value


def source_hashes():
    return {**rc.own_hashes(), SELF: rc.sha256_file(PROJECT / SELF), QUEUE: rc.sha256_file(PROJECT / QUEUE)}


# ------------------------------------------------------------------ worker (one GPU)

def worker(args, config):
    started = time.perf_counter()
    runtime = rc.bind_gpu(args.gpu_uuid, config['device']['allowed_gpu_uuids'])
    import torch
    sources = source_hashes()
    if sources != json.loads(Path(args.output, 'preflight.json').read_text())['source_sha256']:
        raise RuntimeError('Sources differ from the preflight')
    manifest = rc.verify_export(args.export, config, args.export_manifest_sha256)  # same full hash as a real task
    _, n_early, n_predict = row_counts(manifest)
    train = rc.load_split(args.export, manifest, 'train')
    startup = time.perf_counter() - started
    arms = []
    for name in check_config(config):
        arms.append(probe_arm(config, name, train, config['export']['workload_columns'], n_early, n_predict, 'cuda',
                              sync=torch.cuda.synchronize, reset_peak=torch.cuda.reset_peak_memory_stats,
                              peak=lambda: int(torch.cuda.max_memory_allocated())))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during the probe')
    result = {'gpu': args.gpu_uuid, 'runtime': runtime, 'startup_seconds': startup, 'arms': arms,
              'seconds_total': time.perf_counter() - started, 'finished_utc': utc()}
    rc.write_json_atomic(Path(args.output) / f'worker-{args.gpu_uuid}.json', no_forbidden_keys(result))
    print('TIMING_PROBE_WORKER_DONE', args.gpu_uuid, f'{result["seconds_total"]:.1f}s', flush=True)


# ------------------------------------------------------------------ parent

def preflight(args, config):
    if rc.sha256_file(args.config) != args.config_sha256:
        raise ValueError('Config differs from the pinned SHA256')
    check_config(config)
    if not args.output.name.endswith('-timing'):
        raise ValueError('Probe output must end -timing')
    if len(set(args.gpu)) != len(args.gpu) or any(g not in config['device']['allowed_gpu_uuids'] for g in args.gpu):
        raise ValueError('GPU UUIDs must be distinct and in the config allow-list')
    manifest = rc.verify_export(args.export, config, args.export_manifest_sha256)
    return {'protocol': PROTOCOL, 'config_sha256': args.config_sha256, 'export_manifest_sha256': args.export_manifest_sha256,
            'source_sha256': source_hashes(), 'row_counts': dict(zip(('train', 'earlystop', 'predict'), row_counts(manifest))),
            'fixed': {'probe_seed': PROBE_SEED, 'warmup_epochs': WARMUP_EPOCHS, 'measure_epochs': MEASURE_EPOCHS,
                      'epoch_cap': EPOCH_CAP, 'fixed_allowance_seconds': FIXED_ALLOWANCE_SECONDS,
                      'predict_splits': list(PREDICT_SPLITS), 'worker_timeout_seconds': WORKER_TIMEOUT_SECONDS},
            'gpus': list(args.gpu), 'python': sys.executable, 'checked_utc': utc()}


def run(args, config):
    from cuda_direct_queue import GpuLock  # the common per-GPU flock of the family queue
    args.output.mkdir(parents=True, exist_ok=False)  # never reuse an output
    identity = preflight(args, config)
    rc.write_json_atomic(args.output / 'preflight.json', identity)
    locks = [GpuLock(args.lock_dir, g) for g in args.gpu]
    held = []
    try:
        for lock in locks:
            if not lock.acquire():
                raise RuntimeError(f'GPU lock busy: {lock.path}; nothing launched')
            held.append(lock)
        procs = {}
        for gpu in args.gpu:
            log = open(args.output / f'worker-{gpu}.log', 'w')
            command = [args.python, str(PROJECT / SELF), '--config', str(args.config), '--config-sha256', args.config_sha256,
                       '--export', str(args.export), '--export-manifest-sha256', args.export_manifest_sha256,
                       '--output', str(args.output), '--lock-dir', str(args.lock_dir), '--gpu', gpu,
                       'worker', '--gpu-uuid', gpu]
            procs[gpu] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
        codes = {}
        for gpu, proc in procs.items():
            try:
                codes[gpu] = proc.wait(timeout=max(1., deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                codes[gpu] = 'timeout'
        for gpu, proc in procs.items():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    finally:
        for lock in held:
            lock.release()
    workers = []
    for gpu in args.gpu:
        path = args.output / f'worker-{gpu}.json'
        if codes.get(gpu) == 0 and path.is_file():
            workers.append(json.loads(path.read_text()))
    complete = len(workers) == len(args.gpu)
    summary = {'protocol': PROTOCOL, 'exit_codes': {g: codes.get(g) for g in args.gpu}, 'complete': complete,
               'written_utc': utc(), 'preflight': identity}
    if complete:
        summary.update(summarize(workers, len(config['seeds'])))
    rc.write_json_atomic(args.output / 'summary.json', no_forbidden_keys(summary))
    print('TIMING_PROBE_DONE complete', complete, flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock-dir', type=Path, required=True)
    p.add_argument('--gpu', action='append', required=True)
    p.add_argument('--python', default=sys.executable)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('run')
    w = sub.add_parser('worker')
    w.add_argument('--gpu-uuid', required=True)
    args = p.parse_args(argv)
    args.output, args.config, args.export = args.output.resolve(), args.config.resolve(), args.export.resolve()
    config = json.loads(args.config.read_text())
    if args.command == 'run':
        run(args, config)
        return
    if args.gpu_uuid not in args.gpu:
        raise ValueError('Worker GPU is not one of the probe GPUs')
    try:
        worker(args, config)
    except BaseException as error:
        rc.write_json_atomic(args.output / f'worker-{args.gpu_uuid}-error.json',
                             {'error': repr(error), 'traceback': traceback.format_exc(), 'utc': utc()})
        raise


if __name__ == '__main__':
    main()
