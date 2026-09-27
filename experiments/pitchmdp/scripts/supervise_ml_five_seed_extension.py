"""Supervisor and authoritative wall ledger for the G0/F1 five-seed bundle.

Every command of both arms (frozen C1 runner/scorer and the additive F1
extension) is launched here as a worker in its own process group. Before
launch, under an exclusive ledger lock, the supervisor verifies all bundle
pins, enforces the frozen sequence and the per-arm budget (7200 s C1 share,
7200 s F1-extension share, no transfer), reserves the effective cap with a
durable start record, then kills the whole process group at the cap
(SIGTERM grace, then SIGKILL) and writes exactly one terminal record with
the full caller wall. Unresolved starts keep reserving their cap. Workers,
not the supervisor, hold the shared ``.heavy.lock``; runner-internal timings
are never added to this ledger. C1 score is refused until all ten member
predictions verify. Refusals are recorded in the ledger as ``refused`` jobs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

from pitchmdp.data import hash_file
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.matrix_five_seed_extension import (COMMANDS, NEW_SEEDS, PARENT_CELL, SEQUENCE, SHARE_SECONDS, STAGE_CAPS,
    MEMBER_LIMIT_SECONDS, arm_obligation, effective_cap, launch_gate, ledger_totals, member_fit_seconds, sequence_step)
import run_ml_five_seed_extension as runner
from run_ml_benchmark import read_json, dump

LEDGER_PROTOCOL = 'ml_g0_f1_five_seed_wall_ledger_v1'
EXIT_REFUSED, EXIT_TIMEOUT = 2, 124
PRE_PROFILE = ('c1-prepare', 'c1-profile', 'profile-full', 'profile-masked')


# ---------------------------------------------------------------- ledger

@contextmanager
def ledger_lock(ledger):
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / '.ledger.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def ledger_jobs(ledger):
    """Every durable start; an end record resolves it. Only verified AppleDouble sidecars are skipped."""
    jobs = []
    folder = ledger / 'jobs'
    for record in (sorted(p for p in folder.glob('*.json') if not is_appledouble(p)) if folder.exists() else []):
        start = read_json(record)
        if start.get('protocol') != LEDGER_PROTOCOL:
            raise ValueError('Foreign ledger record: ' + record.name)
        end_path = ledger / 'ends' / record.name
        end = read_json(end_path) if end_path.exists() else None
        jobs.append({**{k: start[k] for k in ('job_id', 'command', 'arm', 'stage', 'seed', 'cap_seconds', 'argv')},
                     'ended': end is not None, 'outcome': end['outcome'] if end else 'unresolved_reserved_at_cap',
                     'exit_code': end['exit_code'] if end else None,
                     'elapsed_seconds': end['elapsed_seconds'] if end else None})
    return jobs


def ledger_state(ledger, shares):
    jobs = ledger_jobs(ledger)
    arms = {arm: ledger_totals(jobs, arm) for arm in shares}
    for arm, totals in arms.items():
        totals['share_seconds'] = shares[arm]
        totals['remaining_seconds'] = shares[arm] - totals['charged_seconds']
    return {'jobs': jobs, 'arms': arms,
            'combined_charged_seconds': math.fsum(t['charged_seconds'] for t in arms.values())}


def write_start(ledger, record):
    (ledger / 'jobs').mkdir(exist_ok=True)
    (ledger / 'ends').mkdir(exist_ok=True)
    dump(ledger / 'jobs' / (record['job_id'] + '.json'), record)


def write_end(ledger, job_id, outcome, exit_code, elapsed, cap, note):
    with ledger_lock(ledger):
        target = ledger / 'ends' / (job_id + '.json')
        if target.exists():
            raise RuntimeError('Job already has a terminal outcome; never overwrite it')
        dump(target, {'job_id': job_id, 'outcome': outcome, 'exit_code': exit_code, 'elapsed_seconds': elapsed,
                      'cap_seconds': cap, 'within_cap': elapsed <= cap, 'ended_utc': datetime.now(timezone.utc).isoformat(),
                      'note': note})


def new_job_id(command, arm, seed):
    return f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}-{arm}-{command}-seed{seed if seed is not None else "na"}-{uuid.uuid4().hex[:8]}'


# ---------------------------------------------------------------- sequence state

def completion_paths(bundle, local):
    c1_out, ext_out = runner.arm_output(bundle, local, 'c1'), runner.arm_output(bundle, local, 'f1ext')
    ledger = runner.ledger_location(bundle, local)
    paths = {('c1-prepare', None): c1_out / 'preparation.json', ('c1-profile', None): c1_out / 'profile' / 'state.json',
             ('profile-full', None): ledger / 'profiles' / 'full' / 'state.json',
             ('profile-masked', None): ledger / 'profiles' / 'masked' / 'state.json',
             ('freeze-third-parent', None): ext_out / 'third_parent' / 'manifest.json',
             ('f1-prepare', None): ext_out / 'preparation.json', ('c1-score', None): c1_out / 'analysis' / 'manifest.json',
             ('f1-score', None): ext_out / 'analysis' / 'five_seed' / 'manifest.json'}
    for seed in NEW_SEEDS:
        paths[('c1-fit', seed)] = c1_out / 'fits' / f'seed{seed}' / 'global' / 'state.json'
        paths[('c1-predict', seed)] = c1_out / 'members' / PARENT_CELL / f'seed{seed}' / 'prediction_state.json'
        paths[('f1-fit', seed)] = ext_out / 'members' / 'masked' / f'seed{seed}' / 'fit_state.json'
        paths[('f1-predict', seed)] = ext_out / 'members' / 'masked' / f'seed{seed}' / 'prediction_state.json'
    return paths


def completed_steps(bundle, local):
    paths = completion_paths(bundle, local)
    return {key for key, path in paths.items() if path.is_file()}


def require_sequence(command, seed, done):
    """Every earlier frozen step must be complete and this step must not be."""
    step = sequence_step(command, seed)
    index = SEQUENCE.index(step)
    missing = [f'{c}:{s}' for c, _, _, s in SEQUENCE[:index] if (c, s) not in done]
    if missing:
        raise RuntimeError('Frozen sequence violated; incomplete predecessors: ' + ', '.join(missing))
    if (command, seed) in done:
        raise RuntimeError(f'{command} seed={seed} already complete; nothing to launch (no reruns)')
    return step


def arm_completed_stages(done, arm):
    stages = set()
    for command, step_arm, stage, seed in SEQUENCE:
        if step_arm == arm and (command, seed) in done:
            stages.add((stage, seed))
    return stages


# ---------------------------------------------------------------- gates

def projections(bundle, local, expected):
    """Matched-profile projections per arm once both profiles exist; None before."""
    out = {}
    for arm, name in (('c1', 'full'), ('f1ext', 'masked')):
        folder = runner.profile_dir(bundle, local, name)
        if not (folder / 'state.json').is_file():
            return None
        result, _ = runner.load_profile(bundle, local, name, expected)
        out[arm] = result['projection']
    return out


def c1_gates(bundle, c1_config, local, local_path):
    """Frozen C1 profile gate plus both matched profiles before the first fit of either arm."""
    import run_ml_confirmation as c1
    c1_out = runner.arm_output(bundle, local, 'c1')
    state = read_json(c1_out / 'profile' / 'state.json')
    if state['preparation_sha256'] != hash_file(c1_out / 'preparation.json'):
        raise RuntimeError('C1 mandatory profile belongs to another preparation')
    c1.resource_gate(c1_config, read_json(c1_out / 'profile' / 'profile.json'))
    evidence = runner.ledger_location(bundle, local) / 'profiles' / 'c1_mandatory_loader.json'
    if not evidence.is_file() or read_json(evidence)['c1_profile_state_sha256'] != hash_file(c1_out / 'profile' / 'state.json'):
        raise RuntimeError('C1 mandatory profile must be produced by the restricted-loader entry point (evidence missing)')


def plan_launch(bundle, c1_config, ext, local, local_path, command, seed, state):
    """Effective cap, projection and obligation for one command; raises RuntimeError to refuse."""
    _, arm, stage, _ = sequence_step(command, seed)
    grace = bundle['termination_grace_seconds']
    totals = state['arms'][arm]
    expected = runner.identity(ext, local_path, bundle)
    done = completed_steps(bundle, local)
    projected = projections(bundle, local, expected)
    if command not in PRE_PROFILE and projected is None:
        raise RuntimeError('Both matched profiles must complete before any fit, prediction, freeze, prepare or score')
    member_remaining = None
    if stage == 'predict':
        member_remaining = MEMBER_LIMIT_SECONDS - member_fit_seconds(state['jobs'], arm, seed)
        if member_remaining <= 0:
            raise RuntimeError(f'Member seed {seed} fit walls already exhaust the 7200-second fit+predict limit')
    cap = effective_cap(stage, totals['remaining_seconds'], member_remaining=member_remaining, grace=grace)
    if projected is None:
        projected_command, obligation = 0., 0.
    else:
        projection = projected[arm]
        projected_command = {'prepare': projection['prepare_seconds'], 'freeze': projection['prepare_seconds'],
                             'fit': projection['fit_command_seconds'], 'predict': projection['predict_command_seconds'],
                             'score': projection['score_seconds'], 'profile': 0.}[stage]
        obligation = arm_obligation(projection, arm_completed_stages(done, arm), arm)
        if stage == 'fit':
            c1_gates(bundle, c1_config, local, local_path)
        if stage == 'predict' and projected_command > member_remaining:
            raise RuntimeError(f'Projected predict {projected_command:.1f}s exceeds the member remaining {member_remaining:.1f}s')
    gate = launch_gate(totals, totals['share_seconds'], stage, projected_command=projected_command, remaining_obligation=obligation)
    if projected_command > cap - grace:
        raise RuntimeError(f'{stage}: projected {projected_command:.1f}s does not fit the effective cap {cap:.1f}s minus grace')
    return {'arm': arm, 'stage': stage, 'cap_seconds': cap, 'grace_seconds': grace, **gate}


def worker_argv(bundle_path, local_path, bundle, local, command, seed):
    python = sys.executable
    c1_out = runner.arm_output(bundle, local, 'c1')
    c1_cfg = runner.resolve_path(bundle['configs']['c1']['path'])
    old = [python, str(PROJECT / 'scripts' / 'run_ml_confirmation.py'), '--config', str(c1_cfg), '--local-config', str(local_path),
           '--output', str(c1_out)]
    new = [python, str(PROJECT / 'scripts' / 'run_ml_five_seed_extension.py'), '--bundle', str(bundle_path), '--local-config', str(local_path)]
    if command == 'c1-prepare':
        return [*old, 'prepare']
    if command == 'c1-fit':
        return [*old, 'fit', '--seed', str(seed)]
    if command == 'c1-predict':
        return [*old, 'predict', '--cell', PARENT_CELL, '--seed', str(seed)]
    if command == 'c1-profile':
        return [*new, 'c1-profile-restricted']
    if command in ('profile-full', 'profile-masked'):
        return [*new, 'profile', '--arm', command.split('-')[1]]
    if command == 'freeze-third-parent':
        return [*new, 'freeze-third-parent']
    if command == 'f1-prepare':
        return [*new, 'prepare']
    if command in ('f1-fit', 'f1-predict'):
        return [*new, command.split('-')[1], '--seed', str(seed)]
    if command == 'c1-score':
        return [*new, 'c1-score']
    if command == 'f1-score':
        return [python, str(PROJECT / 'scripts' / 'score_ml_five_seed_extension.py'), '--bundle', str(bundle_path), '--local-config', str(local_path)]
    raise ValueError('Unknown bundle command: ' + command)


# ---------------------------------------------------------------- process-group supervision

def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_group(child, pgid, grace):
    """SIGTERM the whole group, wait half the grace, SIGKILL, wait the rest; only wait proves the child exited."""
    deadline = time.monotonic() + grace
    notes, code = [], None
    for name, number in (('SIGTERM', signal.SIGTERM), ('SIGKILL', signal.SIGKILL)):
        try:
            os.killpg(pgid, number)
        except ProcessLookupError:
            notes.append(f'{name}: group already gone')
        except BaseException as error:
            notes.append(f'{name} failed: {error!r}')
        try:
            remaining = max(deadline - time.monotonic(), 0.)
            code = child.wait(timeout=remaining / 2 if name == 'SIGTERM' else remaining)
            notes.append(f'{name} reaped within grace')
            break
        except subprocess.TimeoutExpired:
            notes.append(f'{name} grace expired')
    reaped = code is not None
    if reaped and _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            notes.append('orphaned group members killed')
        except ProcessLookupError:
            pass
    return code, reaped, '; '.join(notes)


def supervise(bundle_path, local_path, command, seed, *, popen=None):
    started = time.monotonic()
    bundle = runner.load_bundle(bundle_path, real=True)
    runner.verify_bundle_pins(bundle)
    c1_config, ext = runner.load_configs(bundle)
    local = read_json(local_path)
    ledger = runner.ledger_location(bundle, local)
    shares = {arm: record['share_seconds'] for arm, record in bundle['arms'].items()}
    argv = worker_argv(bundle_path, local_path, bundle, local, command, seed)
    grace = bundle['termination_grace_seconds']
    with ledger_lock(ledger):
        state = ledger_state(ledger, shares)
        _, arm, stage, _ = sequence_step(command, seed)
        try:
            done = completed_steps(bundle, local)
            require_sequence(command, seed, done)
            if command == 'c1-score':
                runner.verify_ten_predictions(bundle, ext, c1_config, local_path, local)
            plan = plan_launch(bundle, c1_config, ext, local, local_path, command, seed, state)
        except (RuntimeError, ValueError) as error:
            job_id = new_job_id(command, arm, seed)
            record = {'protocol': LEDGER_PROTOCOL, 'job_id': job_id, 'command': command, 'arm': arm, 'stage': stage, 'seed': seed,
                      'cap_seconds': 0., 'argv': argv, 'launched_utc': datetime.now(timezone.utc).isoformat(), 'supervisor_pid': os.getpid(),
                      'refused': True, 'reason': f'{type(error).__name__}: {error}', 'arm_state_before': state['arms'][arm]}
            write_start(ledger, record)
            dump(ledger / 'ends' / (job_id + '.json'), {'job_id': job_id, 'outcome': 'refused', 'exit_code': EXIT_REFUSED,
                 'elapsed_seconds': time.monotonic() - started, 'cap_seconds': 0., 'within_cap': True,
                 'ended_utc': datetime.now(timezone.utc).isoformat(), 'note': record['reason']})
            print('FIVE_SEED_JOB_REFUSED', job_id, record['reason'], flush=True)
            return 'refused', EXIT_REFUSED
        job_id = new_job_id(command, arm, seed)
        cap = plan['cap_seconds']
        write_start(ledger, {'protocol': LEDGER_PROTOCOL, 'job_id': job_id, 'command': command, 'arm': arm, 'stage': stage, 'seed': seed,
                             'cap_seconds': cap, 'grace_seconds': grace, 'argv': argv, 'plan': plan, 'bundle_sha256': runner.canonical_hash(bundle),
                             'arm_state_before': state['arms'][arm], 'launched_utc': datetime.now(timezone.utc).isoformat(),
                             'supervisor_pid': os.getpid(), 'refused': False})
    outcome, code, note, child, reaped, pgid = 'failed', None, '', None, False, None
    launch = popen or (lambda argv_: subprocess.Popen(argv_, start_new_session=True))
    try:
        remaining = cap - (time.monotonic() - started)
        if remaining <= grace:
            outcome, code, note = 'timeout', EXIT_TIMEOUT, 'Effective cap exhausted before worker launch'
        else:
            child = launch(argv)
            pgid = child.pid
            try:
                code = child.wait(timeout=max(cap - (time.monotonic() - started) - grace, 0.))
                reaped = True
                outcome = 'completed' if code == 0 else 'failed'
                if _group_alive(pgid):
                    os.killpg(pgid, signal.SIGKILL)
                    note = 'orphaned group members killed after worker exit'
            except subprocess.TimeoutExpired:
                code, reaped, note = terminate_group(child, pgid, min(grace, max(cap - (time.monotonic() - started), 0.)))
                if not reaped:
                    raise RuntimeError('Timed-out worker could not be reaped; start remains unresolved and reserves its cap')
                outcome = 'timeout'
    except BaseException as error:
        if child is not None and not reaped:
            code, reaped, cleanup = terminate_group(child, pgid, min(grace, max(cap - (time.monotonic() - started), 0.)))
            note = cleanup
        note = 'launch/supervision failure: ' + repr(error) + ('; ' + note if note else '')
        if reaped or child is None:
            code = code if code is not None else -1
        raise
    finally:
        elapsed = time.monotonic() - started
        if child is None or reaped:
            write_end(ledger, job_id, outcome, code, elapsed, cap, note)
            print('FIVE_SEED_JOB', job_id, outcome, code, f'{elapsed:.3f}s', flush=True)
        else:
            print('FIVE_SEED_JOB_UNRESOLVED', job_id, note, flush=True)
    return outcome, code


def status(bundle_path, local_path):
    bundle = runner.load_bundle(bundle_path, real=True)
    runner.verify_bundle_pins(bundle)
    local = read_json(local_path)
    ledger = runner.ledger_location(bundle, local)
    state = ledger_state(ledger, {arm: record['share_seconds'] for arm, record in bundle['arms'].items()})
    done = completed_steps(bundle, local)
    for command, arm, stage, seed in SEQUENCE:
        print(f'{command:22s} seed={seed} arm={arm:6s} {"complete" if (command, seed) in done else "pending"}')
    for arm, totals in state['arms'].items():
        print(arm, 'ended %.1f reserved %.1f remaining %.1f unresolved %s' % (
            totals['ended_seconds'], totals['reserved_seconds'], totals['remaining_seconds'], totals['unresolved_jobs']))
    print('combined_charged_seconds %.1f of %d' % (state['combined_charged_seconds'], bundle['combined_seconds']))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in (*COMMANDS, 'status'):
        command = sub.add_parser(name)
        if name in ('c1-fit', 'c1-predict', 'f1-fit', 'f1-predict'):
            command.add_argument('--seed', type=int, choices=NEW_SEEDS, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    bundle_path, local_path = args.bundle.resolve(), args.local_config.resolve()
    if args.command == 'status':
        status(bundle_path, local_path)
        return
    outcome, code = supervise(bundle_path, local_path, args.command, getattr(args, 'seed', None))
    sys.exit(EXIT_TIMEOUT if outcome == 'timeout' else (EXIT_REFUSED if outcome == 'refused' else code))


if __name__ == '__main__':
    main()
