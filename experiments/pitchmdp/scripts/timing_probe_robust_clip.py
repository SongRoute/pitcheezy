"""Timing-only warm steady-state TRAIN probe for the DRAFT robust-clip family (tentative EXP-P36-001).

Pre-registration: docs/AMENDMENT-P36-TIMING-PROBE-20261004.md (committed with this script, before any probe run) and
PREREG-DRAFT-EXP-P36-001 §4 gate (ii) (replaced from same_model_review a23e77b M1; the P34 Amendment-2 method, gates
unchanged).  Method, bounds and parent/worker structure are copied from scripts/timing_probe_ple_residual.py at
08da77f (P35 probe, reviewed 7569d2f, run 3b317f2: N4/N5 max(wall, report)/2, N8 environment/host records, F1-F8);
that file is P35-only and is neither imported nor edited here.  Nothing here fits a family task, saves, calibrates,
scores or keeps a model of the family.

Probed arms (config order): residual_counts and robust_clip_residual_counts.  control and periodic_residual_counts are
NOT re-probed: the summary takes their P35 probe projections 464 s / 1,046 s
(docs/RESULT-P35-TIMING-PROBE-3b317f2-20261004.md:30-31) as fixed constants.

Data: ONLY the TRAIN split is loaded (``load_split(..., 'train')``).  The early-stop argument of the frozen ``fit``
loop is filled with the first ``n_earlystop`` TRAIN rows (manifest record ``n``), so the per-epoch validation cost is
measured without opening earlystop/temperature/blend/dev/mlb_dev.  The post-fit prediction cost is a forward pass
over TRAIN rows whose count equals the summed temperature+blend+dev+mlb_dev row counts (also from the manifest),
exactly as the P35 probe.  No loss value, history, best epoch, logit or probability is written.

Per (GPU, arm), fixed before any run:
  prepare   : ``arm_inputs`` of the full TRAIN split and, for the robust-clip arm, ONE ``compute_stats`` call on those
              TRAIN inputs (the same call with the same arguments the frozen robust-clip ``fit`` makes; its seconds
              are recorded separately too, with the statistics hash, rule counts, rule columns and rows)
  warm-up   : frozen ``model.fit`` for WARMUP_EPOCHS epochs (fresh model, discarded)
  measured  : a NEW frozen ``model.fit`` for MEASURE_EPOCHS epochs (fresh model; patience 5 cannot trigger within
              2 epochs); seconds per epoch = max(wall time, ``report['seconds']``) / 2.  For the robust-clip arm the
              wall time also contains the fit's own ``compute_stats`` (the module computes it before the frozen
              fit's timer starts), so the 60-epoch projection carries it ~30 times: conservative, recorded, not
              credited.
  predict   : ``model.logits`` over PREDICT rows of TRAIN after the measured fit
  memory    : peak device bytes per fit (reset before each fit) and peak host RSS
              (``resource.getrusage(RUSAGE_SELF).ru_maxrss``, a process-lifetime peak; unit recorded)
Same code path as the family: ``run_cuda_direct.build_model`` / ``arm_inputs`` and each arm's own frozen ``fit``;
batch size / learning rate / patience from the config ``training`` block.  Probe seed 39 only (outside 0-38).

CLI (server; launch env OMP/MKL/OPENBLAS_NUM_THREADS=2 is required):
  timing_probe_robust_clip.py --config C --config-sha256 S --export E --export-manifest-sha256 M \\
      --output OUT-timing --lock-dir L --gpu UUID_A --gpu UUID_B run
Each GPU gets one worker process (both probed arms in config order), concurrently, each under the common exclusive
per-GPU flock; a busy lock refuses (nothing waits).  Per-worker hard timeout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402  numpy only at import; torch only after bind_gpu

SELF = 'scripts/timing_probe_robust_clip.py'
QUEUE = 'scripts/cuda_direct_queue.py'
PROTOCOL = 'robust_clip_timing_probe_v1'
FAMILY_PREFIX = 'DRAFT-CUDA-ROBUST-CLIP-'
CONFIG_SHA256 = '644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a'  # DRAFT-CUDA-ROBUST-CLIP-20261005
MODULE = 'pitchmdp/direct_robust_clip_model.py'
MODULE_SHA256 = 'a3d6cbcef8350586a23d1a6d35213e36ba8e0ccf21288288464f8ce7dcbf52fa'  # reviewed 55938c7, pinned a23e77b
CONFIG_ARMS = ['control', 'residual_counts', 'periodic_residual_counts', 'robust_clip_residual_counts']
ARMS = ['residual_counts', 'robust_clip_residual_counts']  # probed, config order
RC_ARM = 'robust_clip_residual_counts'
# Not re-probed: P35 probe projections (slower GPU, 60-epoch cap), fixed constants with their citation.
FIXED_TASK_SECONDS = {'control': 464., 'periodic_residual_counts': 1046.}
FIXED_SOURCE = 'docs/RESULT-P35-TIMING-PROBE-3b317f2-20261004.md:30-31'
FAMILY_SEEDS = (34, 35, 36, 37, 38)
PROBE_SEED = 39  # outside the family seeds 34-38 and every earlier block 0-33; timing does not depend on the seed
WARMUP_EPOCHS = 1
MEASURE_EPOCHS = 2
PREDICT_SPLITS = ('temperature', 'blend', 'dev', 'mlb_dev')  # row counts only; the rows predicted are TRAIN rows
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
    """Seed 39 only; refused if it falls in the family seeds or any earlier family block."""
    from pitchmdp.direct_robust_clip_registry import EARLIER_SEEDS  # torch-free
    used = set(FAMILY_SEEDS) | set(config['seeds']) | {s for block in EARLIER_SEEDS.values() for s in block}
    if seed != PROBE_SEED or seed in used:
        raise ValueError('Probe seed must be 39, outside the family seeds 34-38 and every earlier block')
    return seed


def check_module_pin():
    """The reviewed robust-clip module: registry pin and file both equal MODULE_SHA256 (torch-free)."""
    from pitchmdp import direct_robust_clip_registry as reg
    if reg.MODULE_SHA256 != MODULE_SHA256:
        raise ValueError('Registry module pin is not the reviewed robust-clip module sha256')
    if rc.sha256_file(PROJECT / MODULE) != MODULE_SHA256:
        raise ValueError('Robust-clip module differs from the pinned sha256')
    return MODULE_SHA256


def check_config(config):
    """The DRAFT robust-clip family config only (any status); returns the probed arm list in config order."""
    rc.config_check(config)
    if not str(config.get('experiment_id', '')).startswith(FAMILY_PREFIX) or not rc.uses_robust_clip(config):
        raise ValueError('Timing probe is defined only for the DRAFT robust-clip family config')
    if [a['name'] for a in config['arms']] != CONFIG_ARMS:
        raise ValueError('Arm list differs from the pre-registered 4 arms')
    if rc.training_budget(config)['epochs'] != EPOCH_CAP or config['training']['epochs'] != EPOCH_CAP:
        raise ValueError('Epoch cap differs from the pre-registered 60')
    if list(config['seeds']) != list(FAMILY_SEEDS):
        raise ValueError('Family seeds differ from the pre-registered 34-38')
    check_seed(PROBE_SEED, config)
    return list(ARMS)


def check_config_file(path, pinned):
    """The exact reviewed DRAFT file: the given pin must be CONFIG_SHA256 and the file must hash to it."""
    if pinned != CONFIG_SHA256:
        raise ValueError('Config pin is not the reviewed DRAFT-CUDA-ROBUST-CLIP sha256')
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


def host_peak_rss():
    """Peak host RSS of this process so far (RUSAGE_SELF; cumulative over earlier arms of the same worker)."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    linux = sys.platform.startswith('linux')  # Linux reports KiB, macOS bytes
    return {'ru_maxrss': value, 'unit': 'KiB' if linux else 'bytes', 'bytes': value * 1024 if linux else value}


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


def rc_spec_continuous():
    from pitchmdp.direct_robust_clip_registry import ARMS as RC_SPECS  # torch-free
    return RC_SPECS[RC_ARM]['context_continuous']


def rc_stats(inputs):
    """The statistics the frozen robust-clip fit computes from this TRAIN argument (same call, same arguments)."""
    from pitchmdp import direct_periodic_model as periodic
    from pitchmdp import direct_robust_clip_model as rcm
    return rcm.compute_stats(periodic._no_current_physics(inputs), rc_spec_continuous())


def probe_arm(config, arm_name, train, columns, n_early, n_predict, device, seed=PROBE_SEED, sync=lambda: None,
              reset_peak=lambda: None, peak=lambda: None, rss=host_peak_rss):
    """Warm-up fit, measured fit and TRAIN forward pass of one arm.  Returns timing/memory fields only."""
    import numpy as np
    check_seed(seed, config)
    if arm_name not in ARMS:
        raise ValueError('Arm is not probed: ' + str(arm_name))
    arm = rc.arm_spec(config, arm_name)
    spec = config['training']
    t0 = time.perf_counter()
    inputs = rc.arm_inputs(train, arm, columns)
    labels = np.array(train['y'])
    stats_seconds, stats_hash = None, None
    if rc.is_robust_clip_arm(arm):
        from pitchmdp import direct_robust_clip_model as rcm
        from pitchmdp.direct_robust_clip_registry import check_stats_rows
        t1 = time.perf_counter()
        center, scale, rules = rc_stats(inputs)
        stats_seconds = time.perf_counter() - t1
        stats_hash = rcm.stats_sha256(center, scale)
        length = inputs[0].shape[1]
        labelled = dict(zip(rcm.feature_labels(length, rc_spec_continuous()), rules, strict=True))
        rule_counts = {r: rules.count(r) for r in ('iqr', 'range', 'constant')}
    prepare = time.perf_counter() - t0
    early, early_y = take(inputs, labels, n_early)
    kwargs = dict(patience=spec['patience'], batch_size=spec['batch_size'], learning_rate=spec['learning_rate'])
    record = {'arm': arm_name, 'seed': seed, 'prepare_seconds': prepare, 'train_rows': int(len(labels)),
              'validation_standin_rows': int(n_early), 'predict_standin_rows': int(n_predict),
              'host_peak_rss_after_prepare': rss()}
    if stats_hash is not None:
        record['rc'] = {'compute_stats_seconds': stats_seconds, 'stats_sha256': stats_hash,
                        'rule_counts': rule_counts,
                        'rule_columns': {r: [c for c, x in labelled.items() if x == r] for r in ('range', 'constant')},
                        'stats_rows': check_stats_rows(rc.ple_bin_rows(inputs), length),
                        'module_sha256': check_module_pin()}
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
        if stats_hash is not None and (report.get('rc_stats_sha256') != stats_hash
                                       or report.get('rc_rule_counts') != rule_counts):
            raise RuntimeError(f'{phase}: the fit computed other robust-scale statistics than the timed prepare call')
        record[phase] = {'epochs': epochs, 'optimizer_updates': int(report['optimizer_updates']),
                         'fit_report_seconds': float(report['seconds']), 'wall_seconds': wall,
                         'peak_device_bytes': peak(), 'host_peak_rss': rss()}
    record['parameter_count'] = int(sum(p.numel() for p in model.net.parameters()))
    predict_inputs = tuple(a[:n_predict] for a in inputs)
    sync()
    start = time.perf_counter()
    model.logits(predict_inputs)  # values discarded
    sync()
    record['predict_seconds'] = time.perf_counter() - start
    record['peak_device_bytes_measured_and_predict'] = peak()  # reset before the measured fit
    record['host_peak_rss'] = rss()
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
    if 'rc' in record:  # information only, not credited: the fit's own compute_stats inside the measured wall, x cap/2
        out['compute_stats_seconds_measured'] = record['rc']['compute_stats_seconds']
        out['compute_stats_seconds_inside_projection_not_credited'] = \
            EPOCH_CAP / MEASURE_EPOCHS * record['rc']['compute_stats_seconds']
    return out


def summarize(workers, seeds_per_arm, exit_codes):
    """Per probed arm: slowest GPU projection; control/periodic: the fixed P35 constants; family = sum over the 4 arms
    of seeds x slowest (every fit at the cap)."""
    arms = {}
    for worker in workers:
        for record in worker['arms']:
            p = project(record, worker['startup_seconds'])
            arms.setdefault(record['arm'], []).append({'gpu': worker['gpu'], **p,
                                                       'measured_seconds_per_epoch': record['measured_seconds_per_epoch']})
    names = list(arms)
    if names != ARMS or any(len(arms[n]) != len(workers) for n in names):
        raise ValueError('Every probed arm must be measured on every GPU')
    measured = {n: max(r['task_seconds_projected'] for r in arms[n]) for n in names}
    slowest = {n: FIXED_TASK_SECONDS[n] if n in FIXED_TASK_SECONDS else measured[n] for n in CONFIG_ARMS}
    family = sum(seeds_per_arm * s for s in slowest.values())
    margin = MARGINAL_FRACTION * TASK_GATE_SECONDS
    return {'per_arm': arms, 'slowest_task_seconds_projected': slowest, 'family_worker_seconds_projected': family,
            'probed_task_seconds_projected': measured,
            'fixed_task_seconds': {'arms': dict(FIXED_TASK_SECONDS), 'source': FIXED_SOURCE, 'measured_here': False},
            'marginal_arms': [n for n in CONFIG_ARMS if abs(slowest[n] - TASK_GATE_SECONDS) <= margin],
            'gate': {'task_seconds_max': TASK_GATE_SECONDS, 'family_seconds_max': FAMILY_BUDGET_SECONDS,
                     'every_arm_within': all(s <= TASK_GATE_SECONDS for s in slowest.values()),
                     'family_within': family <= FAMILY_BUDGET_SECONDS,
                     'probe_exit_zero': bool(exit_codes) and all(c == 0 for c in exit_codes.values())},
            'label': 'measurement (warm TRAIN epochs, TRAIN robust-scale statistics, TRAIN forward pass) + projection '
                     '(60 x epoch, assumptions named); control/periodic_residual_counts are the P35 probe constants; '
                     'not a family fit or score; only TRAIN loaded; other split files byte-hashed for integrity only'}


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
    check_config_file(args.config, args.config_sha256)  # DRAFT sha and module pin before anything else
    check_module_pin()
    check_env()
    runtime = rc.bind_gpu(args.gpu_uuid, config['device']['allowed_gpu_uuids'])
    import torch
    sources = source_hashes()
    if sources != json.loads(Path(args.output, 'preflight.json').read_text())['source_sha256']:
        raise RuntimeError('Sources differ from the preflight')
    manifest = rc.verify_export(args.export, config, args.export_manifest_sha256)  # same full hash as a real task
    _, n_early, n_predict = row_counts(manifest)
    train = rc.load_split(args.export, manifest, 'train')
    startup = time.perf_counter() - started
    rss_startup = host_peak_rss()
    arms = []
    for name in check_config(config):
        arms.append(probe_arm(config, name, train, config['export']['workload_columns'], n_early, n_predict, 'cuda',
                              sync=torch.cuda.synchronize, reset_peak=torch.cuda.reset_peak_memory_stats,
                              peak=lambda: int(torch.cuda.max_memory_allocated())))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during the probe')
    result = {'gpu': args.gpu_uuid, 'runtime': runtime, 'startup_seconds': startup,
              'host_peak_rss_after_startup': rss_startup, 'arms': arms,
              'seconds_total': time.perf_counter() - started, 'finished_utc': utc()}
    rc.write_json_atomic(Path(args.output) / f'worker-{args.gpu_uuid}.json', no_forbidden_keys(result))
    print('TIMING_PROBE_WORKER_DONE', args.gpu_uuid, f'{result["seconds_total"]:.1f}s', flush=True)


# ------------------------------------------------------------------ parent

def preflight(args, config):
    check_config_file(args.config, args.config_sha256)  # DRAFT sha and module pin before anything else
    module = check_module_pin()
    check_config(config)
    env = check_env()
    if not args.output.name.endswith('-timing') or 'probe' not in args.output.name:
        raise ValueError('Probe output must be a NEW <...>probe<...>-timing directory (never a family runs/ dir)')
    allowed = config['device']['allowed_gpu_uuids']
    if len(set(args.gpu)) != len(args.gpu) or sorted(args.gpu) != sorted(allowed):
        raise ValueError('GPU UUIDs must be exactly the two distinct UUIDs of the config allow-list')
    manifest = rc.verify_export(args.export, config, args.export_manifest_sha256)
    return {'protocol': PROTOCOL, 'config_sha256': args.config_sha256, 'export_manifest_sha256': args.export_manifest_sha256,
            'module_sha256': module, 'source_sha256': source_hashes(),
            'row_counts': dict(zip(('train', 'earlystop', 'predict'), row_counts(manifest))),
            'fixed': {'probe_seed': PROBE_SEED, 'probed_arms': list(ARMS), 'warmup_epochs': WARMUP_EPOCHS,
                      'measure_epochs': MEASURE_EPOCHS, 'epoch_cap': EPOCH_CAP,
                      'fixed_allowance_seconds': FIXED_ALLOWANCE_SECONDS,
                      'fixed_task_seconds': dict(FIXED_TASK_SECONDS), 'fixed_task_seconds_source': FIXED_SOURCE,
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
    exit_codes = {g: codes.get(g) for g in args.gpu}
    summary = {'protocol': PROTOCOL, 'exit_codes': exit_codes, 'complete': complete,
               'written_utc': utc(), 'preflight': identity, 'host_post': host_snapshot()}
    if complete:
        summary.update(summarize(workers, len(config['seeds']), exit_codes))
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
