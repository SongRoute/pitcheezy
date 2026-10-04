"""Timing-only warm steady-state TRAIN probe for the DRAFT PLE-embedding family (EXP-P35-001 DRAFT).

Pre-registration: docs/AMENDMENT-P35-TIMING-PROBE-20261004.md (committed with this script, before any probe
run) and DRAFT §6 "Timing gate before registration — DECIDED" (the P34 Amendment-2 method, gates unchanged).
Method, bounds and parent/worker structure are copied from scripts/timing_probe_tabm_residual.py at 9de203a
(P34 Amendment 2, incl. the N4/N5 max(wall, report)/2 rule and the N8 environment/host records); that file is
P34-only and is neither imported nor edited here.  Nothing here fits, saves, calibrates, predicts or scores a
model of the family.

Data: ONLY the TRAIN split is loaded (``load_split(..., 'train')``).  The early-stop argument of the frozen
``fit`` loop is filled with the first ``n_earlystop`` TRAIN rows (manifest record ``n``), so the per-epoch
validation cost is measured without opening earlystop/temperature/blend/dev/mlb_dev.  The post-fit prediction
cost is a forward pass over TRAIN rows whose count equals the summed temperature+blend+dev+mlb_dev row counts
(also from the manifest).  No loss value, history, best epoch, logit or probability is written.

Per (GPU, arm), fixed before any run:
  prepare   : ``arm_inputs`` of the full TRAIN split and, for the two PLE arms, ONE ``compute_bins`` call on those
              TRAIN inputs (the same call the frozen PLE ``fit`` makes; its seconds are recorded separately too)
  warm-up   : frozen ``model.fit`` for WARMUP_EPOCHS epochs (fresh model, discarded)
  measured  : a NEW frozen ``model.fit`` for MEASURE_EPOCHS epochs (fresh model; patience 5 cannot trigger within
              2 epochs); seconds per epoch = max(wall time, ``report['seconds']``) / 2.  For PLE arms the wall time
              also contains the fit's own binning (the frozen fit recomputes bins before its timer starts), so the
              60-epoch projection carries that binning ~30 times: conservative, recorded, not credited.
  predict   : ``model.logits`` over PREDICT rows of TRAIN after the measured fit
Same code path as the family: ``run_cuda_direct.build_model`` / ``arm_inputs`` and each arm's own frozen ``fit``;
batch size / learning rate / patience from the config ``training`` block.  Probe seed 34 (outside 29-33).

CLI (server; launch env OMP/MKL/OPENBLAS_NUM_THREADS=2 is required):
  timing_probe_ple_residual.py --config C --config-sha256 S --export E --export-manifest-sha256 M \\
      --output OUT-timing --lock-dir L --gpu UUID_A --gpu UUID_B run
Each GPU gets one worker process (all 5 arms in config order), concurrently, each under the common exclusive
per-GPU flock; a busy lock refuses (nothing waits).  Per-worker hard timeout.
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

SELF = 'scripts/timing_probe_ple_residual.py'
QUEUE = 'scripts/cuda_direct_queue.py'
PROTOCOL = 'ple_residual_timing_probe_v1'
FAMILY_PREFIX = 'DRAFT-CUDA-PLE-RESIDUAL-'
CONFIG_SHA256 = 'f0ae4e9116d3efe90d119fcc84616172bde183a981206caacc4804d68a31b9b2'  # DRAFT-CUDA-PLE-RESIDUAL-20261004
ARMS = ['control', 'periodic_residual_counts', 'ple_residual_counts', 'tabm_residual_counts',
        'ple_tabm_residual_counts']
PLE_ARMS = ('ple_residual_counts', 'ple_tabm_residual_counts')
FAMILY_SEEDS = (29, 30, 31, 32, 33)
PROBE_SEED = 34  # outside the family seeds 29-33; timing does not depend on the seed
WARMUP_EPOCHS = 1
MEASURE_EPOCHS = 2
PREDICT_SPLITS = ('temperature', 'blend', 'dev', 'mlb_dev')
EPOCH_CAP = 60
TASK_GATE_SECONDS = 1800
FAMILY_BUDGET_SECONDS = 36000
MARGINAL_FRACTION = .1  # N8: a projection within 10 % of the task gate is reported as marginal (gate unchanged)
FIXED_ALLOWANCE_SECONDS = 30.  # assumption: process start/imports, temperature fit, save, hashing, rename
WORKER_TIMEOUT_SECONDS = 1500
LAUNCH_ENV = {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}  # N8, = the family queue
FORBIDDEN_OUTPUT_KEYS = ('history', 'best_epoch', 'train_weighted_nll', 'earlystop_conditional_nll', 'loss',
                         'logit', 'probabilit', 'nll', 'brier', 'ece', 'score')
SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)  # as cuda_direct_queue: cleanup before any lock release


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_seed(seed, config):
    if seed in FAMILY_SEEDS or seed in config['seeds']:
        raise ValueError('Probe seed must be outside the family seeds 29-33')
    return seed


def check_config(config):
    """The DRAFT PLE-embedding family config only (any status); returns the arm list in config order."""
    rc.config_check(config)
    if not str(config.get('experiment_id', '')).startswith(FAMILY_PREFIX) or not rc.uses_ple(config):
        raise ValueError('Timing probe is defined only for the DRAFT PLE-embedding family config')
    if [a['name'] for a in config['arms']] != ARMS:
        raise ValueError('Arm list differs from the pre-registered 5 arms')
    if rc.training_budget(config)['epochs'] != EPOCH_CAP or config['training']['epochs'] != EPOCH_CAP:
        raise ValueError('Epoch cap differs from the pre-registered 60')
    if list(config['seeds']) != list(FAMILY_SEEDS):
        raise ValueError('Family seeds differ from the pre-registered 29-33')
    check_seed(PROBE_SEED, config)
    return list(ARMS)


def check_config_file(path, pinned):
    """The exact reviewed DRAFT file: the given pin must be CONFIG_SHA256 and the file must hash to it."""
    if pinned != CONFIG_SHA256:
        raise ValueError('Config pin is not the reviewed DRAFT-CUDA-PLE-RESIDUAL sha256')
    if rc.sha256_file(path) != pinned:
        raise ValueError('Config differs from the pinned SHA256')


def check_env(environ=None):
    environ = os.environ if environ is None else environ
    seen = {k: environ.get(k) for k in LAUNCH_ENV}
    if seen != LAUNCH_ENV:
        raise ValueError(f'Launch environment differs from the fixed N8 env {LAUNCH_ENV}: {seen}')
    return seen


def host_snapshot():
    """Host state for the record (N8): nproc, load average, GPU memory/utilisation and compute apps."""
    snap = {'utc': utc(), 'nproc': os.cpu_count(), 'loadavg': list(os.getloadavg()),
            'env': {k: os.environ.get(k) for k in LAUNCH_ENV}}
    queries = {'gpus': ['--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'],
               'compute_apps': ['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader']}
    for name, query in queries.items():
        try:
            out = subprocess.run(['nvidia-smi', *query], capture_output=True, text=True, timeout=30)
            snap['nvidia_smi_' + name] = {'returncode': out.returncode, 'lines': out.stdout.splitlines()}
        except (OSError, subprocess.TimeoutExpired) as error:
            snap['nvidia_smi_' + name] = {'error': repr(error)}
    return snap


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


def ple_bins(arm_name, inputs):
    """The bins the frozen PLE fit computes from this TRAIN argument (same call, same arguments)."""
    from pitchmdp import direct_periodic_model as periodic
    from pitchmdp import direct_ple_model as plm
    from pitchmdp.direct_ple_registry import ARMS as PLE_SPECS
    spec = PLE_SPECS[arm_name]
    return plm.compute_bins(periodic._no_current_physics(inputs), spec['context_continuous'], spec['n_bins'])


def probe_arm(config, arm_name, train, columns, n_early, n_predict, device, seed=PROBE_SEED, sync=lambda: None,
              reset_peak=lambda: None, peak=lambda: None):
    """Warm-up fit, measured fit and TRAIN forward pass of one arm.  Returns timing fields only."""
    import numpy as np
    check_seed(seed, config)
    arm = rc.arm_spec(config, arm_name)
    spec = config['training']
    t0 = time.perf_counter()
    inputs = rc.arm_inputs(train, arm, columns)
    labels = np.array(train['y'])
    bins_seconds, bins_hash = None, None
    if rc.is_ple_arm(arm):
        from pitchmdp import direct_ple_model as plm
        t1 = time.perf_counter()
        bins = ple_bins(arm_name, inputs)
        bins_seconds = time.perf_counter() - t1
        bins_hash, counts = plm.bins_sha256(bins), plm.check_bins(bins)
    prepare = time.perf_counter() - t0
    early, early_y = take(inputs, labels, n_early)
    kwargs = dict(patience=spec['patience'], batch_size=spec['batch_size'], learning_rate=spec['learning_rate'])
    record = {'arm': arm_name, 'seed': seed, 'prepare_seconds': prepare, 'train_rows': int(len(labels)),
              'validation_standin_rows': int(n_early), 'predict_standin_rows': int(n_predict)}
    if bins_hash is not None:
        record['ple'] = {'bins_seconds': bins_seconds, 'bins_sha256': bins_hash, 'bin_count_total': int(sum(counts)),
                         'degenerate_columns': int(sum(t == 0 for t in counts)),
                         'past_valid_rows': rc.ple_bin_rows(inputs)['past_valid_rows']}
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
        if bins_hash is not None and model.net.config['ple_bins_sha256'] != bins_hash:
            raise RuntimeError(f'{phase}: the fit computed other bins than the timed prepare call')
        record[phase] = {'epochs': epochs, 'optimizer_updates': int(report['optimizer_updates']),
                         'fit_report_seconds': float(report['seconds']), 'wall_seconds': wall,
                         'peak_device_bytes': peak()}
    record['parameter_count'] = int(sum(p.numel() for p in model.net.parameters()))
    if bins_hash is not None:
        record['ple']['effective_parameter_count'] = int(model.net.config['ple_effective_parameter_count'])
    predict_inputs = tuple(a[:n_predict] for a in inputs)
    sync()
    start = time.perf_counter()
    model.logits(predict_inputs)  # values discarded
    sync()
    record['predict_seconds'] = time.perf_counter() - start
    m = record['measured']
    record['measured_seconds_per_epoch'] = max(m['wall_seconds'], m['fit_report_seconds']) / m['epochs']
    record['measured_seconds_per_update'] = m['fit_report_seconds'] / m['optimizer_updates']
    del model
    return record


def project(record, startup_seconds):
    """Conservative full-task projection at the 60-epoch cap (assumption fields named)."""
    per_epoch = record['measured_seconds_per_epoch']
    out = {'startup_seconds_measured': startup_seconds, 'prepare_seconds_measured': record['prepare_seconds'],
           'fit_seconds_projected': EPOCH_CAP * per_epoch, 'predict_seconds_measured': record['predict_seconds'],
           'fixed_allowance_seconds_assumed': FIXED_ALLOWANCE_SECONDS,
           'task_seconds_projected': startup_seconds + record['prepare_seconds'] + EPOCH_CAP * per_epoch
           + record['predict_seconds'] + FIXED_ALLOWANCE_SECONDS}
    if 'ple' in record:  # information only, not credited: the fit's own binning inside the measured wall, x cap/2
        out['ple_binning_seconds_inside_projection_not_credited'] = \
            EPOCH_CAP / MEASURE_EPOCHS * record['ple']['bins_seconds']
    return out


def summarize(workers, seeds_per_arm):
    """Per arm: slowest GPU projection; family = sum over arms of seeds x slowest (every fit at the cap)."""
    arms = {}
    for worker in workers:
        for record in worker['arms']:
            p = project(record, worker['startup_seconds'])
            arms.setdefault(record['arm'], []).append({'gpu': worker['gpu'], **p,
                                                       'measured_seconds_per_epoch': record['measured_seconds_per_epoch']})
    names = list(arms)
    if names != ARMS or any(len(arms[n]) != len(workers) for n in names):
        raise ValueError('Every pre-registered arm must be measured on every GPU')
    slowest = {n: max(r['task_seconds_projected'] for r in arms[n]) for n in names}
    family = sum(seeds_per_arm * s for s in slowest.values())
    new = {n: slowest[n] for n in PLE_ARMS}
    margin = MARGINAL_FRACTION * TASK_GATE_SECONDS
    return {'per_arm': arms, 'slowest_task_seconds_projected': slowest, 'family_worker_seconds_projected': family,
            'ple_task_seconds_projected': new,
            'marginal_arms': [n for n in names if abs(slowest[n] - TASK_GATE_SECONDS) <= margin],
            'gate': {'task_seconds_max': TASK_GATE_SECONDS, 'family_seconds_max': FAMILY_BUDGET_SECONDS,
                     'ple_arms_within': all(s <= TASK_GATE_SECONDS for s in new.values()),
                     'every_arm_within': all(s <= TASK_GATE_SECONDS for s in slowest.values()),
                     'family_within': family <= FAMILY_BUDGET_SECONDS},
            'label': 'measurement (warm TRAIN epochs, TRAIN binning, TRAIN forward pass) + projection (60 x epoch, '
                     'assumptions named); not a family fit or score; only TRAIN loaded; other split files '
                     'byte-hashed for integrity only'}


def no_forbidden_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in key.lower() for word in FORBIDDEN_OUTPUT_KEYS):
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
    check_env()
    runtime = rc.bind_gpu(args.gpu_uuid, config['device']['allowed_gpu_uuids'])
    import torch
    sources = source_hashes()
    if sources != json.loads(Path(args.output, 'preflight.json').read_text())['source_sha256']:
        raise RuntimeError('Sources differ from the preflight')
    try:
        check_config_file(args.config, args.config_sha256)
    except ValueError as error:
        raise RuntimeError(str(error)) from None
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
    check_config_file(args.config, args.config_sha256)
    check_config(config)
    env = check_env()
    if not args.output.name.endswith('-timing') or 'probe' not in args.output.name:
        raise ValueError('Probe output must be a NEW <...>probe<...>-timing directory (never a family runs/ dir)')
    allowed = config['device']['allowed_gpu_uuids']
    if len(set(args.gpu)) != len(args.gpu) or sorted(args.gpu) != sorted(allowed):
        raise ValueError('GPU UUIDs must be exactly the two distinct UUIDs of the config allow-list')
    manifest = rc.verify_export(args.export, config, args.export_manifest_sha256)
    return {'protocol': PROTOCOL, 'config_sha256': args.config_sha256, 'export_manifest_sha256': args.export_manifest_sha256,
            'source_sha256': source_hashes(), 'row_counts': dict(zip(('train', 'earlystop', 'predict'), row_counts(manifest))),
            'fixed': {'probe_seed': PROBE_SEED, 'warmup_epochs': WARMUP_EPOCHS, 'measure_epochs': MEASURE_EPOCHS,
                      'epoch_cap': EPOCH_CAP, 'fixed_allowance_seconds': FIXED_ALLOWANCE_SECONDS,
                      'predict_splits': list(PREDICT_SPLITS), 'worker_timeout_seconds': WORKER_TIMEOUT_SECONDS,
                      'launch_env': env},
            'gpus': list(args.gpu), 'lock_dir': str(args.lock_dir), 'python': sys.executable, 'checked_utc': utc()}


def stop_group(proc, grace=20):
    """SIGTERM the worker's own session group, SIGKILL after the grace; returns once the leader is reaped."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    except ProcessLookupError:
        proc.wait()


def _interrupt(signum, frame):
    for sig in SIGNALS:
        signal.signal(sig, signal.SIG_IGN)  # cleanup must not be interrupted again
    raise KeyboardInterrupt(f'signal {signum}')


def run(args, config):
    from cuda_direct_queue import GpuLock  # the common per-GPU flock of the family queue
    args.output.mkdir(parents=True, exist_ok=False)  # never reuse an output
    identity = preflight(args, config)
    identity['host_pre'] = host_snapshot()
    rc.write_json_atomic(args.output / 'preflight.json', identity)
    locks = [GpuLock(args.lock_dir, g) for g in args.gpu]
    held, procs, logs, codes, previous = [], {}, [], {}, {}
    try:
        for sig in SIGNALS:  # SIGTERM/SIGHUP (timeout, ssh drop) must reach the finally below
            previous[sig] = signal.signal(sig, _interrupt)
    except ValueError:  # not the main thread (tests); exceptions are still cleaned up below
        previous = {}
    try:
        for lock in locks:
            if not lock.acquire():
                raise RuntimeError(f'GPU lock busy: {lock.path}; nothing launched')
            held.append(lock)
        for gpu in args.gpu:
            logs.append(open(args.output / f'worker-{gpu}.log', 'w'))
            command = [args.python, str(PROJECT / SELF), '--config', str(args.config), '--config-sha256', args.config_sha256,
                       '--export', str(args.export), '--export-manifest-sha256', args.export_manifest_sha256,
                       '--output', str(args.output), '--lock-dir', str(args.lock_dir), '--gpu', args.gpu[0],
                       '--gpu', args.gpu[1], 'worker', '--gpu-uuid', gpu]
            procs[gpu] = subprocess.Popen(command, stdout=logs[-1], stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
        for gpu, proc in procs.items():
            try:
                codes[gpu] = proc.wait(timeout=max(1., deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                codes[gpu] = 'timeout'
    finally:
        for proc in procs.values():  # never release a GPU lock while an own worker group is alive
            stop_group(proc)
        for log in logs:
            log.close()
        for lock in held:
            lock.release()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    workers = []
    for gpu in args.gpu:
        path = args.output / f'worker-{gpu}.json'
        if codes.get(gpu) == 0 and path.is_file():
            workers.append(json.loads(path.read_text()))
    complete = len(workers) == len(args.gpu)
    summary = {'protocol': PROTOCOL, 'exit_codes': {g: codes.get(g) for g in args.gpu}, 'complete': complete,
               'written_utc': utc(), 'preflight': identity, 'host_post': host_snapshot()}
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
        if not run(args, config)['complete']:
            raise SystemExit(1)
        return
    if args.gpu_uuid not in args.gpu:
        raise ValueError('Worker GPU is not one of the probe GPUs')
    signal.alarm(WORKER_TIMEOUT_SECONDS + 60)  # own bound: a worker orphaned by a SIGKILLed parent still ends
    try:
        worker(args, config)
    except BaseException as error:
        rc.write_json_atomic(args.output / f'worker-{args.gpu_uuid}-error.json',
                             {'error': repr(error), 'traceback': traceback.format_exc(), 'utc': utc()})
        raise


if __name__ == '__main__':
    main()
