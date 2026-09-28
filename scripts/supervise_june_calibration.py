"""Fixed ten-stage June calibration queue; workers own the shared heavy lock.

Official cost is monotonic Popen-before to wait/reap, including failed launches
and termination. Supervisor postflight and terminal-write tails are separate.
No resume, retry, partial-output reuse, or scientific config mutation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

PROTOCOL = "june_calibration_execution_plan_v1"
REGISTRATION_PROTOCOL = "june_calibration_execution_registration_v1"
ORDER = [("prepare", None), ("profile", None)] + [("predict", s) for s in range(5)] + [("fit", None), ("apply", None), ("score", None)]
CAPS = {"prepare": 300, "profile": 600, "predict": 600, "fit": 300, "apply": 300, "score": 300}
COMMON_FIELDS = ("coord_root", "code_root", "code_commit", "scientific_config_path", "scientific_config_sha256", "config_path", "config_sha256", "source_hashes", "supervisor_sha256", "python_executable", "output_dir", "worker_heavy_lock", "stages")
class Interrupted(BaseException):
    pass


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def good_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def atomic(path: Path, value: dict) -> None:
    require(not path.exists(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists(), f"refusing overwrite: {path}")
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def terminate(process: subprocess.Popen, grace: float = 5.0) -> int | None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


@contextmanager
def ignore_cleanup_signals():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, signal.SIG_IGN)
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def commit(value):
    return isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def leaf_checks(plan, job):
    checks = job['completion_checks']
    require(isinstance(checks, dict) and all(isinstance(k, str) and type(v) in (str, int, bool) for k, v in checks.items()), 'invalid completion checks')
    wanted = {'status': 'complete', 'stage': job['stage'], 'config_sha256': plan['config_sha256'],
              'registered_code_commit': plan['code_commit'], 'execution_head': plan['code_commit']}
    if job['stage'] == 'predict':
        wanted['seed'] = job['seed']
    if job['stage'] == 'fit':
        wanted.update(optimizer_calls=36, replay_complete=True)
    if job['stage'] == 'score':
        wanted['contrast_additivity_passed'] = True
    for leaf, value in wanted.items():
        require(any(k.split('.')[-1] == leaf and type(v) is type(value) and v == value for k, v in checks.items()), f'missing completion identity: {leaf}')


def validate(plan, path):
    require(set(plan) == set(COMMON_FIELDS) | {'protocol', 'registration_path', 'registration_sha256'}, 'invalid plan schema')
    require(plan['protocol'] == PROTOCOL, 'wrong protocol')
    coord = Path(plan['coord_root'])
    require(coord.is_absolute() and path.is_absolute() and path.resolve().parent == coord.resolve(), 'plan/coord path mismatch')
    require(commit(plan['code_commit']), 'invalid code C')
    for key in ('config_sha256', 'scientific_config_sha256', 'registration_sha256', 'supervisor_sha256'):
        require(good_hash(plan[key]), f'invalid {key}')
    for key in ('code_root', 'config_path', 'scientific_config_path', 'python_executable', 'worker_heavy_lock', 'registration_path', 'output_dir'):
        require(Path(plan[key]).is_absolute(), f'absolute {key} required')
    require(Path(plan['registration_path']).parent == coord, 'registration outside coord')
    require(Path(plan['worker_heavy_lock']).name == '.heavy.lock', 'shared heavy lock required')
    require(Path(plan['python_executable']) == Path('/Users/song/Projects/pitcheezy/.venv/bin/python'), 'exact root venv required')
    require(isinstance(plan['source_hashes'], dict) and plan['source_hashes'], 'source closure required')
    for p, h in plan['source_hashes'].items():
        require(Path(p).is_absolute() and Path(p).is_relative_to(Path(plan['code_root'])) and good_hash(h), 'invalid source pin')
    jobs = plan['stages']
    require(isinstance(jobs, list) and len(jobs) == 10, 'ten stages required')
    require([(j['stage'], j['seed']) for j in jobs] == ORDER, 'fixed stage/seed order required')
    require(len({j['id'] for j in jobs}) == 10, 'duplicate job id')
    require(len({j['completion_path'] for j in jobs}) == 10, 'duplicate completion path')
    for job in jobs:
        require(set(job) == {'id', 'stage', 'seed', 'argv', 'completion_path', 'completion_checks', 'artifact_hashes_key'}, 'job schema')
        require(isinstance(job['id'], str) and job['id'] and all(c.isalnum() or c in '-_' for c in job['id']), 'unsafe job id')
        require(job['seed'] is None or type(job['seed']) is int, 'seed must be integer')
        argv = job['argv']
        require(isinstance(argv, list) and all(isinstance(v, str) for v in argv), 'argv must be strings')
        expected = [plan['python_executable'], argv[1] if len(argv) > 1 else '', '--config', plan['config_path'], '--output-dir', plan['output_dir'], '--stage', job['stage']]
        if job['seed'] is not None:
            expected += ['--seed', str(job['seed'])]
        require(argv == expected and argv[1] in plan['source_hashes'], 'argv differs from fixed worker command')
        require(Path(job['completion_path']).is_absolute() and Path(job['completion_path']).is_relative_to(Path(plan['output_dir'])), 'completion outside output')
        require(isinstance(job['artifact_hashes_key'], str) and job['artifact_hashes_key'], 'artifact hashes key required')
        leaf_checks(plan, job)


def preflight(plan, fresh=False):
    require(sha(Path(__file__)) == plan['supervisor_sha256'], 'supervisor source changed')
    require(sha(Path(plan['registration_path'])) == plan['registration_sha256'], 'registration changed')
    registration = read(Path(plan['registration_path']))
    require(set(registration) == set(COMMON_FIELDS) | {'protocol', 'registration_commit_d', 'created_utc'}, 'registration schema')
    require(registration['protocol'] == REGISTRATION_PROTOCOL and commit(registration['registration_commit_d']), 'registration identity')
    require(isinstance(registration['created_utc'], str) and registration['created_utc'], 'registration timestamp')
    for key in COMMON_FIELDS:
        require(registration[key] == plan[key], f'registration mismatch: {key}')
    for path_key, hash_key in [('config_path', 'config_sha256'), ('scientific_config_path', 'scientific_config_sha256')]:
        require(sha(Path(plan[path_key])) == plan[hash_key], f'{path_key} changed')
    execution = read(Path(plan['config_path']))
    require(execution.get('enabled') is True and 'execution' not in execution, 'execution config requires top-level enabled')
    require(execution.get('code_commit_c') == plan['code_commit'], 'execution config C mismatch')
    require(execution.get('scientific_config') == {'path': plan['scientific_config_path'], 'sha256': plan['scientific_config_sha256']}, 'execution/scientific binding mismatch')
    scientific = read(Path(plan['scientific_config_path']))
    require(scientific['budget']['family_seconds'] == 3600 and scientific['budget']['profile_gate']['tail_reserve_seconds'] == 900, 'scientific budget mismatch')
    for source, digest in plan['source_hashes'].items():
        require(sha(Path(source)) == digest, f'source changed: {source}')
    require(Path(plan['python_executable']).is_file() and Path(plan['worker_heavy_lock']).is_file(), 'runtime/lock missing')
    current = subprocess.run(['git', '-C', plan['code_root'], 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    require(current == plan['code_commit'], 'execution HEAD mismatch')
    dirty = subprocess.run(['git', '-C', plan['code_root'], 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True).stdout.strip()
    require(not dirty, 'tracked execution changes')
    if fresh:
        output = Path(plan['output_dir'])
        require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'output not fresh')


def dotted(node, key):
    for part in key.split('.'):
        require(isinstance(node, dict) and part in node, f'missing manifest field: {key}')
        node = node[part]
    return node


def completion(plan, job):
    path = Path(job['completion_path'])
    manifest = read(path)
    for key, expected in job['completion_checks'].items():
        value = dotted(manifest, key)
        require(type(value) is type(expected) and value == expected, f'completion mismatch: {key}')
    artifacts = dotted(manifest, job['artifact_hashes_key'])
    require(isinstance(artifacts, dict) and artifacts, 'artifact hashes missing')
    stage_root = path.parent.resolve()
    require(stage_root.is_relative_to(Path(plan['output_dir']).resolve()), 'stage directory escapes output')
    resolved = set()
    for name, digest in artifacts.items():
        require(isinstance(name, str) and name and '\\' not in name, 'invalid relative artifact name')
        relative = Path(name)
        require(not relative.is_absolute() and all(part not in ('', '.', '..') for part in name.split('/')),
                'invalid relative artifact reference')
        artifact = path.parent / relative
        target = artifact.resolve()
        require(target.is_relative_to(stage_root) and target != path.resolve() and target not in resolved
                and good_hash(digest), 'invalid artifact reference')
        resolved.add(target)
        require(artifact.is_file() and sha(artifact) == digest, f'artifact changed: {name}')
    return sha(path)


def budget(job, ends):
    require(all(finite(e['popen_wait_seconds']) for e in ends), 'invalid recorded wall')
    remaining = 3600 - sum(e['popen_wait_seconds'] for e in ends)
    if job['stage'] != 'predict':
        require(remaining >= CAPS[job['stage']], 'insufficient full stage cap')
        return float(CAPS[job['stage']]), {'remaining_seconds': remaining}
    members = [e['popen_wait_seconds'] for e in ends if e['stage'] == 'predict']
    if not members:
        profiles = [e for e in ends if e['stage'] == 'profile']
        require(len(profiles) == 1, 'official profile missing')
        estimate = profiles[0]['popen_wait_seconds'] * 104970 / 8192
    else:
        estimate = max(members)
    k = 5 - len(members)
    require(2 * estimate <= 600 and 2 * k * estimate + 900 <= remaining, 'profile/member budget gate refused')
    timeout = min(600, remaining - 900)
    require(finite(timeout) and timeout > 0, 'prediction reserve exhausted')
    return float(timeout), {'remaining_seconds': remaining, 'estimate_seconds': estimate, 'remaining_members': k, 'reserve_seconds': 900}


@contextmanager
def queue_lock(coord):
    with (coord / '.queue.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_worker(plan, job, coord, cap, plan_hash, gate):
    evidence = {'job_id': job['id'], 'stage': job['stage'], 'seed': job['seed'],
                'plan_sha256': plan_hash, 'registration_sha256': plan['registration_sha256'],
                'config_sha256': plan['config_sha256'], 'code_commit': plan['code_commit'],
                'source_hashes': plan['source_hashes'], 'supervisor_sha256': plan['supervisor_sha256'],
                'cap_seconds': cap, 'gate': gate}
    atomic(coord / 'ledger' / 'starts' / (job['id'] + '.json'), {**evidence, 'argv': job['argv'], 'at_utc': utc()})
    process = None
    begun = None
    observed = None
    elapsed = 0.0
    outcome = 'failed'
    exit_code = None
    digest = None
    note = ''
    old = {}
    def interrupted(_sig, _frame):
        raise Interrupted()
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.getsignal(sig)
        signal.signal(sig, interrupted)
    try:
        with (coord / 'ledger' / (job['id'] + '.stdout.log')).open('xb') as stdout, (coord / 'ledger' / (job['id'] + '.stderr.log')).open('xb') as stderr:
            begun = time.monotonic()
            process = subprocess.Popen(job['argv'], cwd=plan['code_root'], stdout=stdout, stderr=stderr, start_new_session=True)
            startup = time.monotonic() - begun
            require(finite(startup), 'invalid monotonic launch wall')
            exit_code = process.wait(timeout=max(0.0, cap - startup))
            observed = time.monotonic()
            elapsed = observed - begun
            require(finite(elapsed), 'invalid monotonic worker wall')
            if elapsed > cap:
                outcome, note = 'timeout', 'full worker wall exceeds cap'
            elif exit_code == 0:
                digest = completion(plan, job)
                outcome = 'completed'
            else:
                note = f'worker exit {exit_code}'
    except subprocess.TimeoutExpired:
        outcome, note = 'timeout', 'worker timeout; cleanup charged'
    except Interrupted:
        outcome, note = 'interrupted', 'SIGINT/SIGTERM'
    except BaseException as error:
        note = f'{type(error).__name__}: {error}'
    finally:
        with ignore_cleanup_signals():
            try:
                if observed is None:
                    if process is not None:
                        exit_code = terminate(process)
                    observed = time.monotonic()
                    elapsed = observed - begun if begun is not None else 0.0
                valid_clock = finite(elapsed)
                # An invalid clock cannot create free budget or a successful stage.
                if not valid_clock:
                    elapsed, outcome, digest = cap, 'failed', None
                    note += '; invalid clock; reserved cap charged'
                postflight = time.monotonic() - observed
                postflight = postflight if finite(postflight) else 0.0
                end = {**evidence, 'outcome': outcome, 'exit_code': exit_code,
                       'popen_wait_seconds': elapsed, 'within_cap': elapsed <= cap,
                       'clock_valid': valid_clock, 'completion_path': job['completion_path'],
                       'completion_sha256': digest, 'note': note,
                       'postflight_seconds': postflight, 'at_utc': utc()}
                atomic(coord / 'ledger' / 'ends' / (job['id'] + '.json'), end)
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
    return end


def run(path):
    path = Path(path)
    plan = read(path)
    validate(plan, path)
    coord = Path(plan['coord_root'])
    coord.mkdir(parents=True, exist_ok=True)
    plan_hash = sha(path)
    with queue_lock(coord):
        ledger = coord / 'ledger'
        require(not ledger.exists(), 'single attempt already exists; no retry/resume')
        ledger.mkdir()
        ends = []
        failure = None
        try:
            preflight(plan, fresh=True)
            for job in plan['stages']:
                preflight(plan)
                # Every predecessor requires its successful ledger AND current artifact binding.
                for previous, end in zip(plan['stages'], ends):
                    require(end['outcome'] == 'completed' and end['within_cap'] and end['clock_valid'], 'predecessor not successful')
                    require(completion(plan, previous) == end['completion_sha256'], 'predecessor manifest changed')
                cap, gate = budget(job, ends)
                require(not Path(job['completion_path']).exists(), 'stage output already exists')
                end = run_worker(plan, job, coord, cap, plan_hash, gate)
                ends.append(end)
                if end['outcome'] != 'completed':
                    failure = end['note'] or end['outcome']
                    break
            if failure is None:
                for previous, end in zip(plan['stages'], ends):
                    require(completion(plan, previous) == end['completion_sha256'], 'final artifact verification failed')
        except BaseException as error:
            failure = f'{type(error).__name__}: {error}'
            atomic(ledger / 'refusal.json', {'at_utc': utc(), 'reason': failure, 'plan_sha256': plan_hash,
                   'registration_sha256': plan['registration_sha256'], 'config_sha256': plan['config_sha256'],
                   'source_hashes': plan['source_hashes'], 'completed_or_failed_launches': len(ends)})
        result = {'protocol': PROTOCOL, 'outcome': 'completed' if failure is None and len(ends) == 10 else 'failed',
                  'plan_sha256': plan_hash, 'registration_sha256': plan['registration_sha256'],
                  'config_sha256': plan['config_sha256'], 'code_commit': plan['code_commit'],
                  'popen_wait_seconds': sum(e['popen_wait_seconds'] for e in ends),
                  'family_cap_seconds': 3600, 'jobs': ends, 'note': failure, 'at_utc': utc()}
        atomic(coord / 'status.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run'))
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        plan = read(args.plan)
        validate(plan, args.plan)
        preflight(plan, fresh=True)
        result = {'outcome': 'preflight_passed', 'plan_sha256': sha(args.plan)}
    else:
        result = run(args.plan)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['outcome'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    sys.exit(main())
