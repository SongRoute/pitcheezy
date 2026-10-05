"""Same-GPU3 two-slot scheduler and the predeclared serial-vs-concurrent TRAIN-only benchmark (AUTHOR ONLY).

Additive module; the existing queue (cuda_direct_queue.py) and every P37/P38 command stay max 1 fit.  One
supervisor owns ONE common exclusive flock for GPU3 (the same GpuLock file the P37/P38 queues use) for the whole
run; children never touch the lock (no re-acquire deadlock) and are bound only through CUDA_VISIBLE_DEVICES=GPU3.
At most two children run at once, each in its own session (pgid == pid); on timeout/failure/interrupt only those
own groups are stopped, and the lock is released only after every own group is gone.  Fail closed on any other
GPU, a busy lock, a live Pitcheezy GPU process, a non-terminal P37 state, or insufficient host memory / VRAM.

    python scripts/gpu3_two_slot.py bench --spec docs/launch/gpu3-two-slot-bench-001.json --out DIR

The benchmark is not a performance-research experiment: no DEV/2026 rows, no scoring, no policy conclusion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_direct_queue import GpuLock  # the same lock file/semantics as the P37/P38 queues

GPU3 = 'GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9'
RELEASED = ('GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99',)  # GPU4, returned to another project
MAX_SLOTS = 2  # hard cap; 3-4 slots are not part of this protocol
TERMINAL = ('complete', 'incomplete', 'blocked')
OTHER_GPU_WORK = 'cuda_direct_queue.py|score_cuda_direct|timing_probe_|gpu3_bench_child|run-p3[0-9]-pipeline'
GIB = 1 << 30


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, value):
    """Exclusive create: an existing artifact is never overwritten."""
    with open(path, 'x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def check_gpu(uuid, slots):
    if uuid in RELEASED or uuid != GPU3:
        raise ValueError(f'Only GPU3 {GPU3} is permitted; refused {uuid}')
    if not 1 <= int(slots) <= MAX_SLOTS:
        raise ValueError(f'slots must be 1 or 2, got {slots}')


# ------------------------------------------------------------------ preflight (fail closed)

def meminfo_available(path='/proc/meminfo'):
    for line in Path(path).read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable not found')


def nvidia_gpu(uuid):
    """(free bytes, utilization %, used bytes) of exactly this UUID; never queries another GPU by index."""
    out = subprocess.run(['nvidia-smi', f'--id={uuid}', '--query-gpu=uuid,memory.free,utilization.gpu,memory.used',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20, check=True).stdout
    seen, free, util, used = [x.strip() for x in out.strip().split(',')]
    if seen != uuid:
        raise RuntimeError(f'nvidia-smi answered for {seen}, not {uuid}')
    return int(free) * (1 << 20), int(util), int(used) * (1 << 20)


def nvidia_apps():
    """[(gpu_uuid, pid)] of all compute apps (used to prove no own child appears on another GPU)."""
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    return [(u.strip(), int(p)) for u, p in (line.split(',') for line in out.strip().splitlines() if line.strip())]


def pgrep(pattern):
    out = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def preflight(spec, *, mem_available=meminfo_available, gpu=nvidia_gpu, apps=nvidia_apps, procs=pgrep):
    """Return the preflight record or raise; every check is finite and fails closed."""
    res, uuid = spec['resources'], spec['gpu_uuid']
    check_gpu(uuid, spec['concurrent_slots'])
    for path in spec['wait_for_terminal']:
        state = json.loads(Path(path).read_text())
        if state.get('status') not in TERMINAL:
            raise RuntimeError(f'{path}: status {state.get("status")!r} is not terminal; GPU3 not handed over')
    others = procs(OTHER_GPU_WORK)
    if others:
        raise RuntimeError(f'Pitcheezy GPU work still running: pids {others}')
    on_gpu3 = [pid for u, pid in apps() if u == uuid]
    if on_gpu3:
        raise RuntimeError(f'GPU3 has compute processes {on_gpu3}; refusing to share with them')
    slots = int(spec['concurrent_slots'])
    host_need = slots * res['host_bytes_per_child'] + res['host_margin_bytes']
    host_free = int(mem_available())
    if host_free < host_need:
        raise RuntimeError(f'Host MemAvailable {host_free} < required {host_need}')
    vram_free, util, used = gpu(uuid)
    vram_need = slots * res['vram_bytes_per_child'] + res['vram_margin_bytes']
    if vram_free < vram_need:
        raise RuntimeError(f'GPU3 free {vram_free} < required {vram_need}')
    return {'utc': utc(), 'host_available': host_free, 'host_required': host_need, 'gpu3_free': vram_free,
            'gpu3_required': vram_need, 'gpu3_util': util, 'gpu3_used': used}


# ------------------------------------------------------------------ supervisor

class Sampler(threading.Thread):
    """1 s samples of GPU3 only (util, used memory) and the compute-app list while children run."""

    def __init__(self, uuid, gpu=nvidia_gpu, apps=nvidia_apps, period=1.):
        super().__init__(daemon=True)
        self.uuid, self.gpu, self.apps, self.period = uuid, gpu, apps, period
        self.samples, self.app_rows, self.errors, self.halt = [], [], [], threading.Event()

    def run(self):
        while not self.halt.is_set():
            try:
                _, util, used = self.gpu(self.uuid)
                self.samples.append({'t': time.monotonic(), 'util': util, 'used': used})
                self.app_rows.extend(self.apps())
            except Exception as error:  # sampling is evidence, not control: record and continue
                self.errors.append(repr(error))
            self.halt.wait(self.period)


class TwoSlotSupervisor:
    def __init__(self, lock_dir, uuid, *, grace=20., poll=.2):
        check_gpu(uuid, 1)
        self.lock, self.uuid, self.grace, self.poll = GpuLock(lock_dir, uuid), uuid, grace, poll
        self.owned, self.running = False, {}

    def acquire(self):
        """Non-blocking: a busy common lock (P37/P38/other queue) is a refusal, never a wait loop."""
        if not self.lock.acquire():
            raise RuntimeError(f'Common GPU3 lock {self.lock.path} is held by another owner')
        self.owned = True

    def release(self):
        if self.running:
            raise RuntimeError('Refusing to release the GPU3 lock while own children are alive')
        if self.owned:
            self.lock.release()
            self.owned = False

    def stop_own(self, pid):
        """Stop only a group this supervisor created (pgid == own child pid)."""
        record = self.running.get(pid)
        if record is None:
            return
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + self.grace
                while time.monotonic() < deadline and not self._reap(pid, block=False):
                    time.sleep(.05)
                if pid in self.running:
                    os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if pid in self.running:
            self._reap(pid, block=True)

    def _reap(self, pid, block):
        try:
            got, status, usage = os.wait4(pid, 0 if block else os.WNOHANG)
        except ChildProcessError:
            got, status, usage = pid, 0, None
        if got == 0:
            return False
        record = self.running.pop(pid)
        record['log'].close()
        record.update(returncode=os.waitstatus_to_exitcode(status) if usage else None,
                      ended=time.monotonic(), finished_utc=utc(),
                      cpu_seconds=(usage.ru_utime + usage.ru_stime) if usage else None,
                      max_rss_bytes=(usage.ru_maxrss * 1024) if usage else None)
        del record['log']
        self.done.append(record)
        return True

    def run(self, tasks, command_for, out_dir, slots, *, task_timeout, total_timeout):
        """Run tasks with at most `slots` live children; return per-task records (in completion order)."""
        check_gpu(self.uuid, slots)
        if not self.owned:
            raise RuntimeError('The supervisor must own the common GPU3 lock before launching')
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)  # unique; no partial resume into an old mode dir
        pending, self.done, self.max_live = list(tasks), [], 0
        start = time.monotonic()
        try:
            while pending or self.running:
                while pending and len(self.running) < slots:
                    task = pending.pop(0)
                    log = open(out_dir / f'{task["id"]}.log', 'x')
                    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': self.uuid, 'PYTHONUNBUFFERED': '1'}
                    proc = subprocess.Popen(command_for(task, out_dir), stdout=log, stderr=subprocess.STDOUT,
                                            env=env, start_new_session=True)
                    self.running[proc.pid] = {'task': task, 'pid': proc.pid, 'pgid': proc.pid, 'log': log,
                                              'started': time.monotonic(), 'started_utc': utc(),
                                              'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES']}
                    self.max_live = max(self.max_live, len(self.running))
                for pid in list(self.running):
                    self._reap(pid, block=False)
                now = time.monotonic()
                for pid, record in list(self.running.items()):
                    if now - record['started'] > task_timeout or now - start > total_timeout:
                        self.stop_own(pid)
                        self.done[-1]['timed_out'] = True
                if any(r.get('returncode') not in (0, None) or r.get('timed_out') for r in self.done):
                    for pid in list(self.running):
                        self.stop_own(pid)
                    pending.clear()
                time.sleep(self.poll)
        except BaseException:
            for pid in list(self.running):
                self.stop_own(pid)
            raise
        for record in self.done:
            record['wall_seconds'] = record['ended'] - record['started']
        return {'tasks': self.done, 'wall_seconds': time.monotonic() - start, 'max_live': self.max_live,
                'completed': sum(r.get('returncode') == 0 and not r.get('timed_out') for r in self.done)}


# ------------------------------------------------------------------ predeclared comparison

COMPARED = ('state_dict_sha256', 'tensor_sha256', 'initial_param_sha256', 'first_permutation_sha256', 'history',
            'best_epoch', 'epochs_run', 'optimizer_updates', 'train_inputs_sha256', 'earlystop_inputs_sha256',
            'source_sha256', 'arm', 'seed', 'training_rows', 'earlystop_rows')
EXCLUDED = ('seconds', 'fit_seconds', 'report_seconds', 'total_seconds', 'peak_device_bytes', 'pid', 'runtime')


def compare(spec, serial, concurrent):
    """Predeclared rule (spec['decision']): exact equality of every COMPARED field per task, both modes 2/2,
    every child bound to GPU3 only, no OOM, and serial_wall / concurrent_wall >= the declared gain."""
    rule, errors = spec['decision'], []
    expected = {(t['arm'], t['seed']) for t in spec['tasks']}
    modes = {'serial': serial, 'concurrent': concurrent}
    results = {}
    for mode, run in modes.items():
        if run['completed'] != len(spec['tasks']) or len(run['tasks']) != len(spec['tasks']):
            errors.append(f'{mode}: {run["completed"]}/{len(spec["tasks"])} tasks completed')
        results[mode] = {}
        for record in run['tasks']:
            result = record.get('result') or {}
            key = (result.get('arm'), result.get('seed'))
            if key not in expected or key in results[mode]:
                errors.append(f'{mode}: unexpected or duplicate task {key}')
                continue
            results[mode][key] = result
            if record.get('oom') or result.get('oom'):
                errors.append(f'{mode} {key}: out of memory')
            rt = result.get('runtime', {})
            if rt.get('bound_uuid') != spec['gpu_uuid'] or rt.get('cuda_visible_devices') != spec['gpu_uuid'] \
                    or not rt.get('device_uuid') or str(rt['device_uuid']).lower() not in spec['gpu_uuid'].lower() \
                    or record.get('cuda_visible_devices') != spec['gpu_uuid']:
                errors.append(f'{mode} {key}: not proven bound to GPU3 only')
            if result.get('source_sha256') != spec['pins']['source_sha256']:
                errors.append(f'{mode} {key}: source identity differs from the pin')
            if result.get('train_only') is not True:
                errors.append(f'{mode} {key}: not a TRAIN-only result')
        if run.get('foreign_gpu_apps'):
            errors.append(f'{mode}: own child seen on another GPU {run["foreign_gpu_apps"]}')
    if serial.get('max_live', 0) > 1 or concurrent.get('max_live', 0) > MAX_SLOTS:
        errors.append('slot cap exceeded')
    if concurrent.get('max_live', 0) < 2:
        errors.append('concurrent mode never ran two children at once')
    mismatches = []
    for key in sorted(expected):
        a, b = results['serial'].get(key), results['concurrent'].get(key)
        if a is None or b is None:
            continue
        for field in COMPARED:
            if field not in a or a.get(field) != b.get(field):
                mismatches.append(f'{key}: {field}')
    if rule['equivalence'] != 'exact' or rule['tolerance'] != 0:
        errors.append('only the predeclared exact rule (tolerance 0) is supported')
    gain = serial['wall_seconds'] / concurrent['wall_seconds'] if concurrent.get('wall_seconds') else 0.
    fit_gain = (sum(r['fit_seconds'] for r in results['serial'].values())
                / max(max((r['fit_seconds'] for r in results['concurrent'].values()), default=0), 1e-9)
                if results['serial'] and results['concurrent'] else None)
    equivalent = not mismatches and not errors
    adopt = equivalent and gain >= rule['min_throughput_gain']
    return {'equivalent': equivalent, 'mismatches': mismatches, 'errors': errors,
            'throughput_gain_total_wall': gain, 'fit_wall_ratio_info_only': fit_gain,
            'min_throughput_gain': rule['min_throughput_gain'],
            'verdict': 'adoption_candidate_for_review' if adopt else 'retain_serial_record_negative'}


# ------------------------------------------------------------------ benchmark CLI

def child_command(spec, spec_path):
    def command(task, out_dir):
        return [spec['python'], str(Path(__file__).resolve().parent / 'gpu3_bench_child.py'), '--spec', str(spec_path),
                '--arm', task['arm'], '--seed', str(task['seed']), '--result', str(Path(out_dir) / f'{task["id"]}.json')]
    return command


def attach_results(run, out_dir, own_pids_apps):
    for record in run['tasks']:
        path = Path(out_dir) / f'{record["task"]["id"]}.json'
        record['result'] = json.loads(path.read_text()) if path.exists() else None
        log = (Path(out_dir) / f'{record["task"]["id"]}.log')
        record['oom'] = log.exists() and 'out of memory' in log.read_text(errors='replace').lower()
    pids = {r['pid'] for r in run['tasks']}
    run['foreign_gpu_apps'] = sorted({(u, p) for u, p in own_pids_apps if p in pids and u != GPU3})
    return run


def bench(spec_path, out, *, preflight_fn=preflight, sampler_cls=Sampler, command_for=None):
    spec_path, out = Path(spec_path).resolve(), Path(out)
    spec = json.loads(spec_path.read_text())
    if spec.get('protocol') != 'gpu3_two_slot_bench_v1':
        raise ValueError('Unknown benchmark protocol')
    out.mkdir(parents=True, exist_ok=False)  # unique run directory; never reused or overwritten
    write_new_json(out / 'spec-copy.json', {'sha256': sha256_file(spec_path), 'spec': spec})
    tasks = [{**t, 'id': f'{t["arm"]}-{t["seed"]}'} for t in spec['tasks']]
    command_for = command_for or child_command(spec, spec_path)
    sup = TwoSlotSupervisor(spec['lock_dir'], spec['gpu_uuid'])
    record = {'started_utc': utc(), 'spec_sha256': sha256_file(spec_path), 'host': os.uname().nodename}
    sup.acquire()  # ONE acquisition held across both modes: no P37/P38/other queue can interleave
    try:
        record['preflight'] = preflight_fn(spec)
        budget, start = spec['budget'], time.monotonic()
        for mode in spec['mode_order']:  # predeclared order (concurrent first = page-cache bias AGAINST it)
            slots = 1 if mode == 'serial' else int(spec['concurrent_slots'])
            left = budget['total_wall_cap_seconds'] - (time.monotonic() - start)
            if left < budget['min_seconds_to_start_mode']:
                raise RuntimeError(f'{mode}: {left:.0f}s left of the wall cap; report instead of shrinking')
            sampler = sampler_cls(spec['gpu_uuid'])
            sampler.start()
            try:
                run = sup.run(tasks, command_for, out / mode, slots, task_timeout=budget['task_timeout_seconds'],
                              total_timeout=left)
            finally:
                sampler.halt.set()
                sampler.join(5)
            run = attach_results(run, out / mode, sampler.app_rows)
            run['gpu3_samples'] = sampler.samples
            run['gpu3_peak_used_bytes'] = max((s['used'] for s in sampler.samples), default=None)
            run['gpu3_peak_util'] = max((s['util'] for s in sampler.samples), default=None)
            run['sampler_errors'] = sampler.errors
            record[mode] = run
            if run['completed'] != len(tasks):
                break
        record['total_wall_seconds'] = time.monotonic() - start
        if 'serial' in record and 'concurrent' in record:
            record['decision'] = compare(spec, record['serial'], record['concurrent'])
        else:
            record['decision'] = {'verdict': 'retain_serial_record_negative', 'errors': ['a mode did not finish']}
    finally:
        sup.release()
        record['finished_utc'] = utc()
        write_new_json(out / 'bench.json', record)
    return record


def interrupted(signum, frame):
    """SIGTERM/SIGINT/SIGHUP (e.g. the outer `timeout`) become an exception, so run() stops its own child groups
    (they live in their own sessions and would otherwise outlive the supervisor on GPU3) and bench() releases the
    lock and still writes bench.json."""
    raise RuntimeError(f'supervisor interrupted by signal {signum}')


def install_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bench')
    b.add_argument('--spec', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    install_signal_handlers()
    result = bench(args.spec, args.out)
    print('GPU3_TWO_SLOT_BENCH', result['decision']['verdict'], flush=True)


if __name__ == '__main__':
    main()
