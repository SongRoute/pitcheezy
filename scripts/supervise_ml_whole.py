"""Durable, fail-closed process supervisor for a registered whole-MLB plan.

The worker owns the shared ML .heavy.lock. This process holds a separate queue
flock for its entire run; never let a second supervisor launch concurrently.
All worker wall time is measured from immediately before Popen through wait.
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

PROTOCOL = 'ml_whole_supervisor_v1'
PHASES = ('evaluation', 'improvement')


class SupervisorInterrupted(BaseException):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def valid_commit(value):
    return isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def child_of(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def validate(plan, plan_path):
    required = {'protocol', 'repo', 'registration_root', 'repo_commit', 'python_executable', 'ledger_dir',
                'artifact_root', 'worker_heavy_lock', 'termination_grace_seconds',
                'budgets_seconds', 'stages'}
    if set(plan) != required or plan['protocol'] != PROTOCOL:
        raise ValueError('Invalid supervisor plan schema')
    repo = Path(plan['repo']).resolve()
    registration = Path(plan['registration_root']).resolve()
    python = Path(plan['python_executable']).absolute()
    ledger = Path(plan['ledger_dir']).resolve()
    root = Path(plan['artifact_root']).resolve()
    if not repo.is_dir() or not registration.is_dir() or not python.is_file() or not root.is_dir():
        raise ValueError('Missing registered repo, Python, or artifact root')
    if registration == repo:
        raise ValueError('Registration root must be distinct from frozen execution repo')
    if python != registration / '.venv' / 'bin' / 'python':
        raise ValueError('All stages must use the registration root venv Python')
    if not valid_commit(plan['repo_commit']) or not positive(plan['termination_grace_seconds']):
        raise ValueError('Invalid source commit or termination grace')
    if plan['budgets_seconds'] != {'evaluation': 7200, 'improvement': 3600}:
        raise ValueError('Whole-MLB phase budgets differ from registered caps')
    if not child_of(ledger, root / 'runs') or ledger == root / 'runs':
        raise ValueError('Ledger must be a distinct child of the artifact runs root')
    if Path(plan['worker_heavy_lock']).resolve() != root / 'runs' / 'ML-MATRIX-20260924' / '.heavy.lock':
        raise ValueError('Worker heavy lock differs from shared ML lock')
    if not isinstance(plan['stages'], list) or not plan['stages']:
        raise ValueError('Ordered stages required')
    seen = set()
    phase_index = -1
    for item in plan['stages']:
        if set(item) != {'id', 'phase', 'argv', 'config_path', 'config_sha256',
                         'source_hashes', 'completion_path', 'max_seconds', 'gates'}:
            raise ValueError('Invalid stage schema')
        ident, phase = item['id'], item['phase']
        if not isinstance(ident, str) or not ident or not all(c.isalnum() or c in '-_' for c in ident) or ident in seen:
            raise ValueError('Invalid or duplicate stage ID')
        seen.add(ident)
        if phase not in PHASES or PHASES.index(phase) < phase_index:
            raise ValueError('Stage phases must be ordered evaluation then improvement')
        phase_index = PHASES.index(phase)
        if not positive(item['max_seconds']) or item['max_seconds'] > plan['budgets_seconds'][phase]:
            raise ValueError('Invalid stage cap')
        argv = item['argv']
        if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(a, str) for a in argv) or Path(argv[0]).absolute() != python:
            raise ValueError('Every worker must use the registered Python')
        if not child_of(argv[1], repo) or not Path(argv[1]).is_file():
            raise ValueError('Worker script must be a repo file')
        if str(Path(item['config_path']).resolve()) not in argv:
            raise ValueError('Worker argv must use the pinned stage config')
        if not child_of(item['completion_path'], root / 'runs') or not child_of(item['config_path'], registration):
            raise ValueError('Stage output/config outside registered roots')
        if not valid_hash(item['config_sha256']) or not isinstance(item['source_hashes'], dict) or not item['source_hashes']:
            raise ValueError('Stage source/config pins required')
        if str(Path(argv[1]).resolve()) not in item['source_hashes']:
            raise ValueError('Worker script must have a source hash pin')
        for path, digest in item['source_hashes'].items():
            if not child_of(path, repo) or not valid_hash(digest):
                raise ValueError('Invalid stage source pin')
        if not isinstance(item['gates'], list):
            raise ValueError('Stage gates must be a list')
        for gate in item['gates']:
            if set(gate) != {'source_stage_id', 'source', 'field', 'multiplier', 'reserve_seconds'}:
                raise ValueError('Invalid resource gate schema')
            if gate['source_stage_id'] not in seen - {ident} or gate['source'] not in ('completion', 'ledger_end'):
                raise ValueError('Resource gate must bind an earlier completed stage')
            if not positive(gate['multiplier']) or not nonnegative(gate['reserve_seconds']):
                raise ValueError('Invalid resource gate multiplier or reserve')
            if not isinstance(gate['field'], str) or not gate['field']:
                raise ValueError('Invalid gate field')
    if not child_of(plan_path, registration):
        raise ValueError('Registered plan must reside in registration root')
    return plan


def preflight(plan, stage, charged):
    actual = subprocess.run(['git', '-C', plan['repo'], 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if actual != plan['repo_commit']:
        raise RuntimeError('Registered source commit changed')
    if sha(stage['config_path']) != stage['config_sha256']:
        raise RuntimeError('Registered stage config changed')
    for path, digest in stage['source_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Registered source file changed: ' + path)
    for gate in stage['gates']:
        source_stage = next(s for s in plan['stages'] if s['id'] == gate['source_stage_id'])
        path = (Path(source_stage['completion_path']) if gate['source'] == 'completion' else
                Path(plan['ledger_dir']) / 'ends' / (source_stage['id'] + '.json'))
        node = read(path)
        for key in gate['field'].split('.'):
            node = node[key]
        remaining = plan['budgets_seconds'][stage['phase']] - charged[stage['phase']]
        if not nonnegative(node) or node * gate['multiplier'] + gate['reserve_seconds'] > remaining:
            raise RuntimeError('Registered resource gate failed: ' + gate['field'])


@contextmanager
def queue_lock(ledger):
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / '.queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ledger_files(folder):
    if not folder.exists():
        return []
    files = []
    for path in folder.iterdir():
        if path.name.startswith('._') and path.is_file() and path.read_bytes()[:4] == b'\x00\x05\x16\x07':
            continue
        if not path.is_file() or path.suffix != '.json':
            raise RuntimeError('Unexpected ledger file: ' + str(path))
        files.append(path)
    return files


def ledger_state(plan, plan_hash):
    ledger = Path(plan['ledger_dir'])
    starts = ledger / 'starts'
    ends = ledger / 'ends'
    start_files = ledger_files(starts)
    end_files = ledger_files(ends)
    if {p.name for p in end_files} - {p.name for p in start_files}:
        raise RuntimeError('Orphan terminal record')
    ids = {p.stem for p in start_files}
    expected_prefix = [s['id'] for s in plan['stages'][:len(ids)]]
    if ids != set(expected_prefix) or len(ids) != len(start_files):
        raise RuntimeError('Ledger stages are not an exact registered prefix')
    jobs = []
    charged = {phase: 0.0 for phase in PHASES}
    for stage in plan['stages'][:len(ids)]:
        path = starts / (stage['id'] + '.json')
        a = read(path)
        if a.get('plan_sha256') != plan_hash or a.get('stage_id') != path.stem or not positive(a.get('cap_seconds')):
            raise RuntimeError('Invalid or foreign start record')
        if a.get('phase') != stage['phase'] or a['cap_seconds'] != stage['max_seconds']:
            raise RuntimeError('Start record does not match plan')
        endpoint = ends / path.name
        b = read(endpoint) if endpoint.exists() else None
        if b is not None:
            elapsed = b.get('elapsed_seconds')
            if (b.get('protocol') != PROTOCOL or b.get('stage_id') != a['stage_id'] or b.get('plan_sha256') != plan_hash
                    or b.get('phase') != a['phase'] or b.get('cap_seconds') != a['cap_seconds']
                    or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                    or b.get('outcome') not in ('completed', 'failed', 'timeout')
                    or (b['outcome'] == 'completed' and (b.get('exit_code') != 0
                        or elapsed > a['cap_seconds']
                        or b.get('completion_sha256') != sha(stage['completion_path'])))):
                raise RuntimeError('Invalid terminal record or changed completion artifact')
        charged[stage['phase']] += max(b['elapsed_seconds'], 0.0) if b else stage['max_seconds']
        jobs.append((a, b))
    return jobs, charged


def terminate_group(process, grace):
    """Reap the direct child and kill every remaining process in its group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        code = process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        code = None
    # A child may exit after TERM while a grandchild ignores it. Always send
    # KILL to the group after grace, even if the direct child has exited.
    if code is not None and grace:
        time.sleep(grace)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait()


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def interrupt_cleanup():
    previous = {}
    def handler(number, _frame):
        raise SupervisorInterrupted(f'supervisor signal {number}')
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.getsignal(number)
        signal.signal(number, handler)
    try:
        yield
    finally:
        for number, original in previous.items():
            signal.signal(number, original)


def run_stage(plan, plan_hash, stage_id):
    ledger = Path(plan['ledger_dir'])
    with queue_lock(ledger):
        jobs, charged = ledger_state(plan, plan_hash)
        if any(b is None or b['outcome'] != 'completed' for _, b in jobs):
            raise RuntimeError('Failed or unresolved stage requires manual review; no automatic retry')
        completed = [a['stage_id'] for a, _ in jobs]
        next_index = len(completed)
        if next_index >= len(plan['stages']) or plan['stages'][next_index]['id'] != stage_id:
            raise RuntimeError('Stage ordering or duplicate launch rejected')
        stage = plan['stages'][next_index]
        if Path(stage['completion_path']).exists():
            raise RuntimeError('Completion artifact exists without successful ledger record')
        phase = stage['phase']
        if charged[phase] + stage['max_seconds'] > plan['budgets_seconds'][phase]:
            raise RuntimeError('Remaining phase budget cannot reserve stage cap')
        preflight(plan, stage, charged)
        start_record = {'protocol': PROTOCOL, 'plan_sha256': plan_hash, 'stage_id': stage_id,
                        'phase': phase, 'cap_seconds': stage['max_seconds'],
                        'argv': stage['argv'], 'started_utc': utc()}
        start_path = ledger / 'starts' / (stage_id + '.json')
        end_path = ledger / 'ends' / (stage_id + '.json')
        if start_path.exists() or end_path.exists():
            raise RuntimeError('Stage already reserved')
        atomic(start_path, start_record)
        origin = time.monotonic()
        process = None
        outcome, code, note = 'failed', None, None
        with interrupt_cleanup():
            try:
                env = os.environ.copy()
                env['PITCHEEZY_HEAVY_LOCK'] = plan['worker_heavy_lock']
                process = subprocess.Popen(stage['argv'], cwd=plan['repo'], env=env, start_new_session=True)
                grace = min(plan['termination_grace_seconds'], stage['max_seconds'])
                remaining = stage['max_seconds'] - (time.monotonic() - origin) - grace
                if remaining <= 0:
                    outcome, code, note = 'timeout', terminate_group(process, max(stage['max_seconds'] - (time.monotonic() - origin), 0)), 'launch exhausted cap'
                else:
                    try:
                        code = process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        outcome, code, note = 'timeout', terminate_group(process, min(grace, max(stage['max_seconds'] - (time.monotonic() - origin), 0))), 'stage cap reached'
                    else:
                        if group_alive(process.pid):
                            terminate_group(process, min(grace, max(stage['max_seconds'] - (time.monotonic() - origin), 0)))
                            note = 'worker left descendant processes in its group'
                        elif code == 0 and Path(stage['completion_path']).is_file():
                            outcome = 'completed'
                        else:
                            note = 'worker failed or completion artifact missing'
            except BaseException as exc:
                note = repr(exc)
                if process is not None:
                    code = terminate_group(process, min(plan['termination_grace_seconds'], max(stage['max_seconds'] - (time.monotonic() - origin), 0)))
            elapsed = time.monotonic() - origin
            if elapsed > stage['max_seconds'] and outcome == 'completed':
                outcome, note = 'timeout', 'full process wall exceeded cap'
            completion_hash = sha(stage['completion_path']) if outcome == 'completed' else None
            atomic(end_path, {'protocol': PROTOCOL, 'plan_sha256': plan_hash, 'stage_id': stage_id,
                              'phase': phase, 'outcome': outcome, 'exit_code': code,
                              'elapsed_seconds': elapsed, 'cap_seconds': stage['max_seconds'],
                              'completion_path': stage['completion_path'], 'completion_sha256': completion_hash,
                              'note': note, 'ended_utc': utc()})
        if outcome != 'completed':
            raise RuntimeError(f'{stage_id}: {outcome}: {note}; inspect ledger before any retry')
        return {'stage': stage_id, 'elapsed_seconds': elapsed, 'completion_sha256': completion_hash}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('run')
    step.add_argument('--stage', required=True)
    sub.add_parser('status')
    args = parser.parse_args(argv)
    plan = validate(read(args.plan), args.plan)
    if Path(sys.executable).absolute() != Path(plan['python_executable']).absolute():
        raise RuntimeError('Supervisor and workers must use the same registered root venv Python')
    plan_hash = canonical(plan)
    if args.command == 'status':
        jobs, charged = ledger_state(plan, plan_hash)
        print(json.dumps({'charged_seconds': charged, 'jobs': [{'stage': a['stage_id'], 'outcome': b['outcome'] if b else 'unresolved_reserved_at_cap'} for a, b in jobs]}, indent=2))
        return 0
    result = run_stage(plan, plan_hash, args.stage)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
