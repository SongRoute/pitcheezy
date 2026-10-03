"""DRAFT persistent queue for DRAFT-CUDA-DIRECT-20261004 on newsrv (stdlib only).

Predeclared tasks = registered arms x seeds (priority 1 first).  At most two GPUs,
each from the config allow-list, each guarded by an exclusive flock; one task per
GPU.  Every task runs run_cuda_direct.py in its OWN process group bound to its GPU
UUID.  Fail closed before launching anything if the config/export/runner identity
does not match.  Clock rules (KST): no new task after `no_new_fit_after`; own
process groups still running at `kill_own_after` are terminated (SIGTERM, then
SIGKILL); the summary is written immediately, before `aggregate_by`.  A task that
fails twice (failure, timeout, missing state) is final and never retried; attempts
persist across queue restarts.  Per-task timeout and a family wall budget bound the
run.  The queue never promotes, pushes or deploys anything and never touches a
process it did not start.

  python scripts/cuda_direct_queue.py --config ../../configs/DRAFT-CUDA-DIRECT-20261004.json \
      --config-sha256 <sha> --export <dir> --export-manifest-sha256 <sha> --output <family dir> \
      --queue-dir <dir> --gpu GPU-c334d905-... --gpu GPU-ae803a24-... run [--priority 1] [--smoke]
  ... summary        (rewrite summary.json from status files; no launch)
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / 'scripts' / 'run_cuda_direct.py'
KST = timezone(timedelta(hours=9))  # no DST
FINAL = ('done', 'failed_final', 'killed_deadline')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


class Deadlines:
    def __init__(self, queue):
        day = datetime.strptime(queue['date'], '%Y-%m-%d').replace(tzinfo=KST)

        def at(text):
            hour, minute = (int(x) for x in text.split(':'))
            return day.replace(hour=hour, minute=minute)
        self.no_new, self.kill, self.aggregate = at(queue['no_new_fit_after']), at(queue['kill_own_after']), at(queue['aggregate_by'])
        if not self.no_new < self.kill < self.aggregate:
            raise ValueError('Deadlines must be ordered: no new fit < kill < aggregate')

    def may_launch(self, now):
        return now < self.no_new

    def must_kill(self, now):
        return now >= self.kill


def plan(config, priorities=(1, 2)):
    """Predeclared, ordered task list: priority, then seed, then registered arm order."""
    arms = [a for a in config['arms'] if a['priority'] in priorities]
    order = sorted(((a['priority'], seed, i, a['name']) for i, a in enumerate(arms) for seed in config['seeds']))
    return [{'id': f'{name}-{seed}', 'arm': name, 'seed': seed, 'priority': p} for p, seed, _, name in order]


class GpuLock:
    def __init__(self, lock_dir, uuid):
        Path(lock_dir).mkdir(parents=True, exist_ok=True)
        self.path, self.handle = Path(lock_dir) / f'{uuid}.lock', None

    def acquire(self):
        handle = open(self.path, 'a+')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.seek(0), handle.truncate(), handle.write(f'{os.getpid()} {utc()}\n'), handle.flush()
        self.handle = handle
        return True

    def release(self):
        if self.handle:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class Queue:
    def __init__(self, config, queue_dir, gpus, command_for, is_complete, *, lock_dir=None, priorities=(1, 2),
                 clock=lambda: datetime.now(KST), sleep=time.sleep, poll=5., grace=20.):
        q = config['queue']
        allowed = config['device']['allowed_gpu_uuids']
        if not gpus or len(gpus) > min(2, config['device']['max_parallel']) or len(set(gpus)) != len(gpus) \
                or any(g not in allowed for g in gpus):
            raise ValueError('One or two distinct GPUs from the allow-list only')
        self.config, self.dir, self.gpus = config, Path(queue_dir), list(gpus)
        self.command_for, self.is_complete = command_for, is_complete
        self.deadlines, self.clock, self.sleep, self.poll, self.grace = Deadlines(q), clock, sleep, poll, grace
        self.timeout, self.budget, self.max_attempts = q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']
        if self.max_attempts != 2:
            raise ValueError('Exactly two attempts per task')
        for sub in ('status', 'logs'):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.locks = {g: GpuLock(lock_dir or self.dir / 'locks', g) for g in self.gpus}
        self.tasks = plan(config, priorities)
        self.running = {}  # gpu -> (task, Popen, started monotonic, log handle)
        self.family_seconds = 0.

    # ---- status
    def status_path(self, task):
        return self.dir / 'status' / f'{task["id"]}.json'

    def status(self, task):
        path = self.status_path(task)
        if path.exists():
            return json.loads(path.read_text())
        return {**task, 'state': 'pending', 'attempts': 0, 'stages': [{'stage': 'declared', 'utc': utc()}]}

    def update(self, task, state=None, **details):
        record = self.status(task)
        if state:
            record['state'] = state
        record.update({k: v for k, v in details.items() if k != 'stage'})
        record['stages'].append({'stage': details.get('stage', state), 'utc': utc(),
                                 **{k: v for k, v in details.items() if k != 'stage'}})
        write_json(self.status_path(task), record)
        return record

    # ---- launch / reap
    def next_task(self):
        busy = {t['id'] for t, *_ in self.running.values()}
        for task in self.tasks:
            record = self.status(task)
            if task['id'] in busy or record['state'] in FINAL:
                continue
            if self.is_complete(task):
                self.update(task, 'done', stage='found_complete')
                continue
            if record['attempts'] >= self.max_attempts:
                self.update(task, 'failed_final', stage='attempts_exhausted')
                continue
            return task
        return None

    def launch(self, gpu, task):
        record = self.status(task)
        attempt = record['attempts'] + 1
        log_path = self.dir / 'logs' / f'{task["id"]}-attempt{attempt}.log'
        if log_path.exists():
            raise FileExistsError(f'{log_path} exists; logs are never overwritten')
        log = open(log_path, 'x')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1'}
        command = self.command_for(task, gpu)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                   cwd=str(PROJECT))
        self.running[gpu] = (task, process, time.monotonic(), log)
        self.update(task, 'running', stage='launched', attempts=attempt, gpu=gpu, pid=process.pid, pgid=process.pid,
                    log=str(log_path), command=command)

    def stop_group(self, process):
        """Terminate only the process group this queue created (pgid == child pid)."""
        if process.poll() is not None:
            return
        try:
            if os.getpgid(process.pid) != process.pid:
                return  # not the session we started
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(self.grace)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except ProcessLookupError:
            pass

    def finish(self, gpu, state=None, stage=None):
        task, process, started, log = self.running.pop(gpu)
        elapsed = time.monotonic() - started
        self.family_seconds += elapsed
        log.close()
        self.locks[gpu].release()
        code = process.returncode
        if state is None:
            ok = code == 0 and self.is_complete(task)
            attempts = self.status(task)['attempts']
            state = 'done' if ok else ('failed_final' if attempts >= self.max_attempts else 'pending')
            stage = 'finished' if ok else ('failed' if code else 'missing_state')
        elif state == 'pending' and self.status(task)['attempts'] >= self.max_attempts:
            state = 'failed_final'
        self.update(task, state, stage=stage, returncode=code, seconds=elapsed)

    def reap(self, now):
        for gpu in list(self.running):
            task, process, started, _ = self.running[gpu]
            if process.poll() is not None:
                self.finish(gpu)
            elif time.monotonic() - started > self.timeout:
                self.stop_group(process)
                self.finish(gpu, 'pending', 'timeout_killed')

    def kill_all(self):
        for gpu in list(self.running):
            self.stop_group(self.running[gpu][1])
            self.finish(gpu, 'killed_deadline', 'deadline_killed')

    def run(self):
        queue_lock = GpuLock(self.dir, 'queue')
        if not queue_lock.acquire():
            raise RuntimeError('Another queue instance owns this queue directory')
        try:
            write_json(self.dir / 'plan.json', {'tasks': self.tasks, 'gpus': self.gpus, 'created_utc': utc(),
                                                'deadlines_kst': [str(self.deadlines.no_new), str(self.deadlines.kill),
                                                                  str(self.deadlines.aggregate)]})
            while True:
                now = self.clock()
                self.reap(now)
                if self.deadlines.must_kill(now):
                    self.kill_all()
                    break
                launching = self.deadlines.may_launch(now) and self.family_seconds < self.budget
                if launching:
                    for gpu in self.gpus:
                        if gpu in self.running:
                            continue
                        task = self.next_task()
                        if task is None:
                            break
                        if not self.locks[gpu].acquire():
                            self.update(task, None, stage='gpu_lock_busy', gpu=gpu)
                            continue
                        try:
                            self.launch(gpu, task)
                        except Exception as error:
                            self.locks[gpu].release()
                            self.update(task, None, stage='launch_error', error=repr(error))
                            raise
                if not self.running and (not launching or self.next_task() is None):
                    break
                self.sleep(self.poll)
            for task in self.tasks:
                if self.status(task)['state'] not in FINAL:
                    reason = 'deadline_no_new_fit' if not self.deadlines.may_launch(self.clock()) else 'family_budget'
                    self.update(task, 'not_run', stage=reason)
            return self.summary()
        finally:
            queue_lock.release()

    def summary(self):
        records = [self.status(t) for t in self.tasks]
        first = [r for r in records if r['priority'] == 1]
        result = {'family': self.config['experiment_id'], 'written_utc': utc(), 'written_kst': str(self.clock()),
                  'on_time': self.clock() < self.deadlines.aggregate,
                  'counts': {s: sum(r['state'] == s for r in records) for s in sorted({r['state'] for r in records})},
                  'priority1_complete': all(r['state'] == 'done' for r in first),
                  'comparison_allowed': all(r['state'] == 'done' for r in first),
                  'family_seconds': self.family_seconds,
                  'tasks': [{k: r.get(k) for k in ('id', 'arm', 'seed', 'priority', 'state', 'attempts', 'gpu',
                                                     'returncode', 'seconds', 'log')} for r in records],
                  'note': 'No promotion, push or deployment. Descriptive scoring happens on the Mac after pull.'}
        write_json(self.dir / 'summary.json', result)
        return result


# ------------------------------------------------------------------ CLI

def preflight(args, config):
    """Fail closed before any launch."""
    if sha256_file(args.config) != args.config_sha256:
        raise ValueError('Config differs from the pinned SHA256')
    sys.path.insert(0, str(PROJECT / 'scripts'))
    import run_cuda_direct as runner  # numpy only at import; torch is never imported here
    runner.config_check(config)
    runner.verify_export(args.export, config, args.export_manifest_sha256)
    if args.smoke != args.output.name.endswith('-smoke'):
        raise ValueError('Smoke runs write only to an output ending -smoke, and only smoke runs may')
    return {'runner_sha256': sha256_file(RUNNER), 'config_sha256': args.config_sha256,
            'export_manifest_sha256': args.export_manifest_sha256, 'python': sys.executable, 'checked_utc': utc()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--queue-dir', type=Path, required=True)
    p.add_argument('--lock-dir', type=Path, help='shared per-GPU lock directory (default <queue-dir>/locks)')
    p.add_argument('--gpu', action='append', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--python', default=sys.executable)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--priority', type=int, action='append', choices=(1, 2))
    sub.add_parser('summary')
    args = p.parse_args(argv)
    args.output = args.output.resolve()
    config = json.loads(Path(args.config).read_text())
    sys.path.insert(0, str(PROJECT / 'scripts'))
    import run_cuda_direct as runner

    def command_for(task, gpu):
        command = [args.python, str(RUNNER), '--config', str(args.config.resolve()), '--export', str(args.export.resolve()),
                   '--export-manifest-sha256', args.export_manifest_sha256, '--output', str(args.output),
                   '--gpu-uuid', gpu, 'task', '--arm', task['arm'], '--seed', str(task['seed'])]
        return command + (['--smoke'] if args.smoke else [])

    def is_complete(task):
        dest = runner.task_dir(args.output, task['arm'], task['seed'], args.smoke)
        if not (dest / 'state.json').is_file():
            return False
        runner.verify_task(dest)
        return True

    queue = Queue(config, args.queue_dir, args.gpu, command_for, is_complete, lock_dir=args.lock_dir,
                  priorities=tuple(getattr(args, 'priority', None) or (1, 2)))
    if args.command == 'summary':
        print(json.dumps(queue.summary(), indent=2))
        return
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    try:
        identity = preflight(args, config)
    except Exception as error:
        write_json(args.queue_dir / f'preflight-failed-{os.getpid()}.json', {'error': repr(error), 'utc': utc()})
        raise
    write_json(args.queue_dir / f'preflight-{os.getpid()}.json', identity)
    result = queue.run()
    print('CUDA_QUEUE_DONE', json.dumps(result['counts']), 'comparison_allowed', result['comparison_allowed'], flush=True)


if __name__ == '__main__':
    main()
