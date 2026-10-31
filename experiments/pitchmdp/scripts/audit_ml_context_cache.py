"""Real-data audit runner for the optional F4 observed-context cache (COOP-001).

Fresh sibling audit directory only; original v3 preparation/profiles are read
for identity and metadata and never modified. Every command is launched by a
supervisor that records a durable start in a registered wall ledger outside
the audit directory, gates the launch on the frozen family budget, kills the
worker at the registered cap (SIGTERM grace, then SIGKILL) and records one
terminal outcome with the full caller wall. The worker holds the shared heavy
lock. Only TRAIN/early-stop/May rows are loaded. No DEV/June cache, no quality
metric, no full member fit and no adoption decision are produced here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
REPO = PROJECT.parents[1]

import numpy as np
import torch

from pitchmdp.data import hash_file
from pitchmdp.matrix_long_experiment import CELLS
from pitchmdp.matrix_policy_artifacts import is_appledouble, artifact_names
from pitchmdp.matrix_context_cache_audit import (REGISTERED_SAMPLES, WARMUP_TRAIN, STAGE1_ORDERS, STAGE1_MUST_FAIL,
    STAGE2_TOLERANCES, STAGE3_TOLERANCES, STAGE3_BUDGET, select_samples, sample_records, check_sample_identity,
    build_cache, stage1_context, stage2_backend, stage3_fit, compare_stage3, cost_summary, validate_prior_ledger)
from run_ml_long_history import SOURCES as F4_SOURCES
from audit_ml_long_equivalence import load_batches
from run_ml_benchmark import read_json, dump, identity as base_identity, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock, assert_hashes, artifact_hashes

PROTOCOL = 'f4_observed_context_cache_audit_v2'
SOURCES = list(dict.fromkeys([*F4_SOURCES, 'pitchmdp/matrix_long_equivalence.py', 'scripts/audit_ml_long_equivalence.py',
    'pitchmdp/matrix_observed_context_cache.py', 'pitchmdp/matrix_policy_artifacts.py',
    'pitchmdp/matrix_context_cache_audit.py', 'scripts/audit_ml_context_cache.py']))
CONTRACTS = {'optional_cache': 'docs/contracts/ML-F4-CONTEXT-CACHE-OPTIONAL-v1.md',
             'runner': 'docs/contracts/ML-F4-CONTEXT-CACHE-RUNNER-v1.md'}
COMMANDS = ('prepare', 'stage1', 'stage2', 'stage3', 'compare', 'summary')
PATHS = ('original', 'cached')
PROFILE_CAP_SECONDS = 600      # historical per-profile command cap; a 7200 member cap never authorizes a 7200 profile
EXIT_NOT_EQUIVALENT = 3
HEX = set('0123456789abcdef')


def _sha(value, length=64):
    return isinstance(value, str) and len(value) == length and set(value) <= HEX


def _pin(value, length=64, *, real):
    if value is None:
        if real: raise ValueError('Null identity pin is not allowed in a real audit; freeze the registration first')
        return
    if not _sha(value, length): raise ValueError('Identity pin must be a lowercase hex digest')


def _seconds(value, upper=None, *, allow_zero=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (value >= 0 if allow_zero else value > 0) and (upper is None or value <= upper))


def config_check(config, *, real=True):
    required = {'protocol', 'experiment_id', 'contracts', 'historical_registration', 'repo_commit', 'parent',
                'sources', 'device', 'arms', 'samples', 'stage1', 'stage2', 'stage3', 'limits',
                'prior_cost_ledger', 'adoption'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Exact F4 cache-audit v2 configuration schema required')
    if config['protocol'] != PROTOCOL or config['experiment_id'] != 'EXP-P4-002' or config['adoption'] is not None:
        raise ValueError('Registered protocol/experiment and null adoption required')
    if config['device'] != 'mps' or config['arms'] != list(CELLS):
        raise ValueError('Audit fixes the actual MPS backend and all three arms')
    _pin(config['repo_commit'], 40, real=real)
    contracts = config['contracts']
    if set(contracts) != set(CONTRACTS) or any(set(record) != {'path', 'sha256'} or record['path'] != CONTRACTS[name]
                                                 for name, record in contracts.items()):
        raise ValueError('Optional-cache and runner contract paths must both be pinned')
    for record in contracts.values(): _pin(record['sha256'], real=real)
    registration = config['historical_registration']
    if set(registration) != {'config', 'sha256'} or registration['config'] != 'configs/EXP-P4-002-v3-cache-audit.yaml':
        raise ValueError('Historical v1 registration must be pinned and preserved')
    _pin(registration['sha256'], real=real)
    parent = config['parent']
    if set(parent) != {'run', 'config', 'config_sha256', 'preparation_sha256', 'profiles'} or not parent['run'] or not parent['config']:
        raise ValueError('Parent v3 run/config/preparation/profile pins required')
    _pin(parent['config_sha256'], real=real); _pin(parent['preparation_sha256'], real=real)
    if set(parent['profiles']) != set(CELLS):
        raise ValueError('All three parent profiles must be pinned')
    for record in parent['profiles'].values():
        if set(record) != {'profile_sha256', 'manifest_sha256'}: raise ValueError('Profile pins need profile and manifest digests')
        _pin(record['profile_sha256'], real=real); _pin(record['manifest_sha256'], real=real)
    if set(config['sources']) != set(SOURCES):
        raise ValueError('Source pins must cover exactly the audited source family')
    for digest in config['sources'].values(): _pin(digest, real=real)
    samples = config['samples']
    if (set(samples) != {'rule', 'warmup_train', 'train', 'train_evaluation', 'earlystop', 'temperature', 'dev_or_june_cache'}
            or samples['rule'] != 'chronological_linspace_no_shrink' or samples['warmup_train'] != WARMUP_TRAIN
            or samples['dev_or_june_cache'] != 'forbidden'):
        raise ValueError('Registered chronological selectors and forbidden DEV/June cache required')
    for name, count in REGISTERED_SAMPLES.items():
        record = samples[name]
        keys = {'n', 'rows_sha256', 'draws'} if name == 'temperature' else {'n', 'rows_sha256'}
        if set(record) != keys or record['n'] != count or (name == 'temperature' and record['draws'] != 400):
            raise ValueError('Sample counts/draws are fixed: 65536/2048/2048/16x400')
        _pin(record['rows_sha256'], real=real)
    one = config['stage1']
    if (set(one) != {'orders', 'samples', 'must_fail', 'tolerance', 'chunk_size', 'timing_repetitions', 'uneven_chunk_sizes', 'timing_minibatch_size'}
            or one['orders'] != list(STAGE1_ORDERS) or one['samples'] != ['train', 'train_evaluation']
            or one['must_fail'] != list(STAGE1_MUST_FAIL) or one['tolerance'] != 'bitwise'
            or type(one['chunk_size']) is not int or one['chunk_size'] < 1 or type(one['timing_repetitions']) is not int
            or one['timing_repetitions'] < 2 or not one['uneven_chunk_sizes'] or any(type(s) is not int or s < 1 for s in one['uneven_chunk_sizes'])
            or one['timing_minibatch_size'] != 256):
        raise ValueError('Stage1 orders on both samples, bitwise tolerance, chunking and alternating timing are fixed')
    two = config['stage2']
    if (set(two) != {'seed', 'width', 'minibatches', 'batch_size', 'learning_rate', 'tolerances', 'rng_state', 'identical_initialization_and_minibatch_order'}
            or two['seed'] != 0 or two['width'] != 128 or two['batch_size'] != 256 or two['learning_rate'] != .0005
            or type(two['minibatches']) is not int or two['minibatches'] < 1 or two['tolerances'] != STAGE2_TOLERANCES
            or two['identical_initialization_and_minibatch_order'] is not True):
        raise ValueError('Stage2 fixes seed0 width128 batch256 lr.0005 and the registered non-relaxable tolerances')
    three = config['stage3']
    if (set(three) != {'budget', 'warmup', 'draws', 'tolerances', 'profile_models_reused_in_full_experiment', 'scientific_budget_unchanged'}
            or three['budget'] != STAGE3_BUDGET or three['warmup'] != {'train': WARMUP_TRAIN, 'epochs': 1, 'evaluation': 'train_evaluation'}
            or three['draws'] != 400 or three['tolerances'] != STAGE3_TOLERANCES
            or three['profile_models_reused_in_full_experiment'] is not False
            or three['scientific_budget_unchanged'] != '30/5/256/.0005'):
        raise ValueError('Stage3 fixes the v3 4-epoch/patience4/256/.0005 resource fit, May16x400 and registered tolerances')
    limits = config['limits']
    caps = limits.get('command_seconds', {})
    if (set(limits) != {'command_seconds', 'termination_grace_seconds', 'wall_ledger_dir', 'profile_command_cap_seconds',
                        'single_member_wall_limit_seconds', 'family_wall_budget_seconds',
                        'no_seed_data_epoch_batch_draw_reduction', 'full_fit_started'}
            or limits['profile_command_cap_seconds'] != PROFILE_CAP_SECONDS
            or limits['single_member_wall_limit_seconds'] != 7200 or limits['family_wall_budget_seconds'] != 28800
            or limits['no_seed_data_epoch_batch_draw_reduction'] is not True or limits['full_fit_started'] is not False
            or set(caps) != set(COMMANDS) or not all(_seconds(caps[name], 7200) for name in COMMANDS)
            or not _seconds(caps['stage3'], PROFILE_CAP_SECONDS)
            or not _seconds(limits['termination_grace_seconds']) or limits['termination_grace_seconds'] >= min(caps.values())
            or not isinstance(limits['wall_ledger_dir'], str) or not limits['wall_ledger_dir']):
        raise ValueError('Per-command caps (stage3 within the historical 600-second profile cap, grace inside every cap), '
                         'a registered wall ledger directory and the frozen 7200/28800 limits are required')
    ledger = config['prior_cost_ledger']
    if ledger is None:
        if real: raise ValueError('Owner prior-cost ledger pin required for a real audit')
    elif set(ledger) != {'path', 'sha256'} or not _sha(ledger['sha256']):
        raise ValueError('Prior-cost ledger needs a path and digest')
    return config


def source_hashes():
    return {name: hash_file(PROJECT/name) for name in SOURCES}


def contract_hashes():
    return {name: hash_file(REPO/path) for name, path in CONTRACTS.items()}


def identity(config, local_path):
    return {**base_identity(config, local_path), 'source_hashes': source_hashes(),
            'contract_hashes': contract_hashes(), 'repo_commit': config['repo_commit']}


def verify_pins(config):
    hashes = source_hashes()
    for name, digest in config['sources'].items():
        if hashes[name] != digest: raise ValueError('Pinned audit source changed: ' + name)
    for name, record in config['contracts'].items():
        if hash_file(REPO/record['path']) != record['sha256']: raise ValueError('Pinned contract changed: ' + name)
    registration = config['historical_registration']
    if hash_file(REPO/registration['config']) != registration['sha256']:
        raise ValueError('Historical v1 registration file changed; preserve it')
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != config['repo_commit']:
        raise ValueError('Checked-out commit differs from the registered repo_commit')


def prior_seconds(config):
    record = config['prior_cost_ledger']
    if hash_file(Path(record['path'])) != record['sha256']: raise ValueError('Owner prior-cost ledger changed')
    return validate_prior_ledger(read_json(Path(record['path'])), config['parent']['preparation_sha256'], lambda p: hash_file(Path(p)))


def verify_parent(config, local, repo=REPO):
    """Pin the untouched v3 run: preparation, three profiles, config, frozen sources, sample hashes."""
    pins = config['parent']
    parent = Path(pins['run']).resolve()
    check_location(local, parent)
    if hash_file(repo/pins['config']) != pins['config_sha256']:
        raise ValueError('Parent v3 configuration file changed')
    if hash_file(parent/'preparation.json') != pins['preparation_sha256']:
        raise ValueError('Parent v3 preparation changed')
    prep = read_json(parent/'preparation.json')
    assert_hashes(parent, prep['artifact_hashes'])
    if prep['features']['long_stream']['version'] != 'batter_dual_stream_v2' or prep.get('dev_scores_read') is not False:
        raise ValueError('Parent must be the reviewed v3 preparation with closed DEV scores')
    for name, digest in prep['identity']['source_hashes'].items():
        if hash_file(parent/'source'/name) != digest: raise ValueError('Frozen v3 source copy changed: ' + name)
        if hash_file(PROJECT/name) != digest: raise ValueError('v3 pinned source differs from the working tree: ' + name)
    profiles = {}
    for cell, record in pins['profiles'].items():
        directory = parent/'profiles'/cell
        if hash_file(directory/'manifest.json') != record['manifest_sha256'] or hash_file(directory/'profile.json') != record['profile_sha256']:
            raise ValueError('Parent v3 profile evidence changed: ' + cell)
        manifest = read_json(directory/'manifest.json')
        if manifest['identity'] != {'preparation_sha256': pins['preparation_sha256'], 'cell': cell, 'profile_version': 'f4_resource_profile_v2'}:
            raise ValueError('Parent profile identity differs: ' + cell)
        assert_hashes(directory, manifest['artifact_hashes'])
        profile = read_json(directory/'profile.json')
        recorded = {**{name: profile['samples'][name] for name in ('train', 'earlystop', 'temperature')},
                    'train_evaluation': profile['warmup']['samples']['evaluation_train']}
        registered = {name: {'n': config['samples'][name]['n'], 'rows_sha256': config['samples'][name]['rows_sha256']}
                      for name in REGISTERED_SAMPLES}
        if (profile['long_length'] != CELLS[cell] or profile['dev_scores_read'] is not False
                or profile['warmup']['samples']['train']['n'] != WARMUP_TRAIN
                or {n: {'n': r['n'], 'rows_sha256': r['rows_sha256']} for n, r in recorded.items()} != registered):
            raise ValueError('Registered sample selectors differ from the v3 profile samples: ' + cell)
        profiles[cell] = profile
    return parent, prep, profiles


# ---------------------------------------------------------------- outputs and seals

def fresh(directory):
    """Refuse any prior content except verified AppleDouble sidecars."""
    directory = Path(directory)
    if directory.exists():
        if not directory.is_dir(): raise ValueError('Output path exists and is not a directory')
        leftovers = [p for p in directory.rglob('*') if p.is_file() and not is_appledouble(p)]
        if leftovers: raise ValueError(f'Preserve existing output {directory}; choose a fresh attempt')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def seal(directory, stage_identity):
    files = artifact_names(directory, exclude=('manifest.json',))
    dump(directory/'manifest.json', {'identity': stage_identity, 'artifact_hashes': artifact_hashes(directory, files)})
    return hash_file(directory/'manifest.json')


def sealed(directory, stage_identity):
    manifest = read_json(directory/'manifest.json')
    files = set(artifact_names(directory, exclude=('manifest.json',)))
    if manifest['identity'] != stage_identity or files != set(manifest['artifact_hashes']):
        raise ValueError('Sealed stage identity/artifact family differs: ' + str(directory))
    assert_hashes(directory, manifest['artifact_hashes'])
    return hash_file(directory/'manifest.json')


def stage_root(output, command, arm=None, path=None):
    directory = output/command
    if arm is not None: directory = directory/arm
    if path is not None: directory = directory/path
    return directory


def stage_identity(registration, command, arm, path, attempt, dependencies=None):
    return {'registration_sha256': registration, 'stage': command, 'arm': arm, 'path': path, 'attempt': attempt,
            'dependencies': dict(dependencies or {})}


def stage_passed(command, result):
    if command == 'stage1': return bool(result['all_bitwise_identical'] and result['all_guards_rejected'])
    if command in ('stage2', 'compare'): return bool(result['equivalent'])
    return True


def completed_attempt(output, registration, command, arm=None, path=None, dependencies=None):
    """Exactly one sealed attempt is the evidence; failed attempts are preserved and ignored."""
    root = stage_root(output, command, arm, path)
    candidates = sorted(p for p in root.glob('attempt*') if (p/'manifest.json').is_file()) if root.exists() else []
    if len(candidates) != 1:
        raise ValueError(f'Exactly one sealed {command} attempt required for {arm}/{path}; found {len(candidates)}')
    directory = candidates[0]
    attempt = int(directory.name[len('attempt'):])
    return directory, sealed(directory, stage_identity(registration, command, arm, path, attempt, dependencies))


def passed_predecessor(output, registration, command, arm, dependencies=None):
    """A sealed predecessor whose own recorded verdict passed; its manifest digest binds the successor."""
    directory, digest = completed_attempt(output, registration, command, arm, None, dependencies)
    result = read_json(directory/'results.json')
    if not stage_passed(command, result):
        raise ValueError(f'Predecessor {command} for {arm} is sealed but did not pass; no costly successor stage')
    return digest


def predecessors(output, registration, command, arm):
    """Registered dependency chain: stage2 <- stage1; stage3 <- stage1, stage2; compare <- both stage3 paths."""
    if command in ('stage1', 'prepare', 'summary'): return {}
    chain = {f'stage1/{arm}': passed_predecessor(output, registration, 'stage1', arm)}
    if command == 'stage2': return chain
    chain[f'stage2/{arm}'] = passed_predecessor(output, registration, 'stage2', arm, {f'stage1/{arm}': chain[f'stage1/{arm}']})
    if command == 'stage3': return chain
    fit_chain = dict(chain)  # both stage3 paths were sealed against the stage1+stage2 chain only
    for path in PATHS:
        _, digest = completed_attempt(output, registration, 'stage3', arm, path, fit_chain)
        chain[f'stage3/{arm}/{path}'] = digest
    return chain


# ---------------------------------------------------------------- wall ledger (supervisor-owned)

def ledger_location(local, output, ledger_dir):
    """Registered ledger directory: under the artifact root, disjoint from the sealed audit output."""
    root = Path(local['artifact_root']).resolve()
    ledger = Path(ledger_dir).resolve(); output = Path(output).resolve()
    if not ledger.is_relative_to(root) or ledger == root or ledger.is_relative_to(output) or output.is_relative_to(ledger):
        raise ValueError('Wall ledger must live under the artifact root and outside the audit output')
    return ledger


@contextmanager
def ledger_lock(ledger):
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger/'.ledger.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def ledger_state(ledger):
    """Every started job is charged: its final wall if ended, otherwise its full cap (reserved until resolved)."""
    jobs = []
    starts = sorted(p for p in (ledger/'jobs').glob('*.json') if not is_appledouble(p)) if (ledger/'jobs').exists() else []
    for record in starts:
        start = read_json(record)
        end_path = ledger/'ends'/record.name
        end = read_json(end_path) if end_path.exists() else None
        charged = end['elapsed_seconds'] if end else start['cap_seconds']
        jobs.append({'job_id': start['job_id'], 'command': start['command'], 'arm': start['arm'], 'path': start['path'],
                     'attempt': start['attempt'], 'cap_seconds': start['cap_seconds'], 'ended': end is not None,
                     'outcome': end['outcome'] if end else 'unresolved_reserved_at_cap',
                     'exit_code': end['exit_code'] if end else None, 'charged_seconds': charged})
    return {'jobs': jobs, 'charged_seconds': math.fsum(j['charged_seconds'] for j in jobs),
            'ended_seconds': math.fsum(j['charged_seconds'] for j in jobs if j['ended']),
            'reserved_seconds': math.fsum(j['charged_seconds'] for j in jobs if not j['ended']),
            'unresolved_jobs': [j['job_id'] for j in jobs if not j['ended']]}


def start_job(ledger, config, prior, command, arm, path, attempt, argv):
    """Under the ledger lock: budget gate, single-heavy gate, unique attempt, durable start before launch."""
    cap = config['limits']['command_seconds'][command]
    family = config['limits']['family_wall_budget_seconds']
    with ledger_lock(ledger):
        state = ledger_state(ledger)
        if state['unresolved_jobs']:
            raise RuntimeError('Unresolved audit job reserves its cap; resolve it (write its end record) before launching: '
                               + ', '.join(state['unresolved_jobs']))
        if any((j['command'], j['arm'], j['path'], j['attempt']) == (command, arm, path, attempt) for j in state['jobs']):
            raise RuntimeError('This command/arm/path/attempt was already launched; use a new attempt number')
        projected = prior+state['charged_seconds']+cap
        if projected > family:
            raise RuntimeError(f'Launch refused: prior {prior:.3f}s + audit {state["charged_seconds"]:.3f}s + cap {cap}s '
                               f'exceeds the frozen {family}s family budget')
        job_id = f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}-{command}-{arm or "all"}-{path or "na"}-attempt{attempt}-{uuid.uuid4().hex[:8]}'
        (ledger/'jobs').mkdir(exist_ok=True); (ledger/'ends').mkdir(exist_ok=True)
        dump(ledger/'jobs'/(job_id+'.json'), {'protocol': 'f4_cache_audit_wall_ledger_v1', 'job_id': job_id,
            'command': command, 'arm': arm, 'path': path, 'attempt': attempt, 'cap_seconds': cap,
            'grace_seconds': config['limits']['termination_grace_seconds'], 'argv': argv,
            'prior_seconds': prior, 'audit_charged_before_launch_seconds': state['charged_seconds'],
            'family_seconds': family, 'launched_utc': datetime.now(timezone.utc).isoformat(), 'supervisor_pid': os.getpid()})
    return job_id, cap


def end_job(ledger, job_id, outcome, exit_code, elapsed, cap, note):
    with ledger_lock(ledger):
        target = ledger/'ends'/(job_id+'.json')
        if target.exists(): raise RuntimeError('Job already has a terminal outcome; never overwrite it')
        dump(target, {'job_id': job_id, 'outcome': outcome, 'exit_code': exit_code, 'elapsed_seconds': elapsed,
                      'cap_seconds': cap, 'within_cap': elapsed <= cap, 'ended_utc': datetime.now(timezone.utc).isoformat(), 'note': note})


def terminate_and_reap(child, grace):
    """Share one bounded grace across TERM and KILL; only wait proves exit."""
    deadline = time.monotonic() + grace
    notes = []
    for signal_name, action in (('SIGTERM', child.terminate), ('SIGKILL', child.kill)):
        try:
            action()
        except BaseException as error:
            notes.append(f'{signal_name} failed: {error!r}')
        try:
            remaining = max(deadline - time.monotonic(), 0.)
            code = child.wait(timeout=remaining / 2 if signal_name == 'SIGTERM' else remaining)
            notes.append(f'{signal_name} reaped within grace')
            return code, True, '; '.join(notes)
        except subprocess.TimeoutExpired:
            notes.append(f'{signal_name} grace expired')
        except BaseException as error:
            notes.append(f'{signal_name} reap failed: {error!r}')
    return None, False, '; '.join(notes)


@contextmanager
def supervision_signals():
    """Turn terminal interrupts into catchable cleanup while supervising a child."""
    state = {'launching': False, 'pending': []}
    if threading.current_thread() is not threading.main_thread():
        yield state
        return
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(number, frame):
        if state['launching']:
            state['pending'].append(number)  # let Popen return so its child can be reaped
            return
        raise KeyboardInterrupt(f'Supervisor received signal {number}')
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield state
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def supervise(command, arm, path, attempt, config, local, output, argv, *, popen=subprocess.Popen):
    """One job, one terminal outcome only after the launched worker is reaped."""
    started = time.monotonic()  # includes ledger location, prior-hash validation and durable start
    ledger = ledger_location(local, output, config['limits']['wall_ledger_dir'])
    prior = prior_seconds(config)
    job_id, cap = start_job(ledger, config, prior, command, arm, path, attempt, argv)
    grace = config['limits']['termination_grace_seconds']
    outcome, code, note = 'failed', None, ''
    child, reaped, cleanup_attempted = None, False, False
    with supervision_signals() as signals:
        try:
            remaining = cap - (time.monotonic()-started)
            if remaining <= 0:
                outcome = 'timeout'
                code = 124
                note = 'Caller cap exhausted before worker launch'
            else:
                signals['launching'] = True
                try:
                    child = popen(argv)
                finally:
                    signals['launching'] = False
                if signals['pending']:
                    raise KeyboardInterrupt(f'Supervisor received signal {signals["pending"][0]} during launch')
                try:
                    code = child.wait(timeout=max(cap-(time.monotonic()-started)-grace, 0.))
                    reaped = True
                    outcome = 'completed' if code == 0 else ('not_equivalent' if code == EXIT_NOT_EQUIVALENT else 'failed')
                except subprocess.TimeoutExpired:
                    cleanup_attempted = True
                    code, reaped, note = terminate_and_reap(child, min(grace, max(cap-(time.monotonic()-started), 0.)))
                    if not reaped:
                        raise RuntimeError('Timed-out worker could not be reaped; start remains unresolved')
                    outcome = 'timeout'
        except BaseException as error:
            if child is not None and not reaped and not cleanup_attempted:
                cleanup_attempted = True
                code, reaped, cleanup_note = terminate_and_reap(child, min(grace, max(cap-(time.monotonic()-started), 0.)))
                note = cleanup_note
            note = 'launch/supervision failure: ' + repr(error) + ('; ' + note if note else '')
            if reaped or child is None:
                code = code if code is not None else -1
            raise
        finally:
            elapsed = time.monotonic()-started
            if child is None or reaped:
                end_job(ledger, job_id, outcome, code, elapsed, cap, note)
                print('CACHE_AUDIT_JOB', job_id, outcome, code, f'{elapsed:.3f}s', flush=True)
            else:
                print('CACHE_AUDIT_JOB_UNRESOLVED', job_id, note, flush=True)
    return outcome, code


# ---------------------------------------------------------------- worker commands

def prepare(config, local, output, expected):
    if output.exists() and any(p.is_file() and not is_appledouble(p) for p in output.rglob('*')):
        raise ValueError('Preserve the existing audit directory; register a fresh sibling')
    parent, prep, profiles = verify_parent(config, local)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output) or 'members' in output.parts:
        raise ValueError('Audit requires a distinct sibling outside the parent and any member directory')
    prior = prior_seconds(config)
    output.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, target)
    for name, relative in CONTRACTS.items():
        target = output/'contracts'/Path(relative).name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, target)
    dump(output/'registered_config.json', config)
    dump(output/'original_preparation.json', prep)
    dump(output/'parent_profiles.json', profiles)
    populations = {name: int(prep['samples'][name]['n']) for name in ('train', 'earlystop', 'temperature', 'blend', 'dev')}
    dump(output/'registration.json', {'identity': expected, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent': config['parent'], 'contracts': config['contracts'], 'population_counts': populations,
        'prior_cost_ledger': {**config['prior_cost_ledger'], 'seconds': prior},
        'wall_ledger_dir': config['limits']['wall_ledger_dir'],
        'interpretation': 'Resource/numerical audit of an unadopted cache; no DEV/June cache, no quality, no full fit'})
    dump(output/'registration_manifest.json', {'identity': expected, 'artifact_hashes': artifact_hashes(output, artifact_names(output))})
    print('CACHE_AUDIT_PREPARED', output, flush=True)


def verify_registration(output, expected):
    manifest = read_json(output/'registration_manifest.json')
    if manifest['identity'] != expected: raise ValueError('Audit config/source/contract/environment changed since registration')
    assert_hashes(output, manifest['artifact_hashes'])
    return hash_file(output/'registration_manifest.json')


def _select(config, parent, prep, local, arm, started, limit):
    batches, aux = load_batches(parent, prep, local, CELLS[arm])
    load_seconds = time.monotonic()-started
    selected, warm = select_samples(batches)
    records = sample_records(selected)
    check_sample_identity(records, config['samples'])
    if time.monotonic()-started > limit: raise TimeoutError('Deadline exhausted during input load')
    return selected, warm, aux, records, load_seconds


def run_stage(config, local, output, expected, command, arm, path, attempt):
    """Returns the stage's own pass verdict; results are sealed even when the verdict fails."""
    registration = verify_registration(output, expected)
    parent, prep, _ = verify_parent(config, local)
    dependencies = predecessors(output, registration, command, arm)
    directory = fresh(stage_root(output, command, arm, path)/f'attempt{attempt}')
    identity_record = stage_identity(registration, command, arm, path, attempt, dependencies)
    limit = config['limits']['command_seconds'][command]
    dump(directory/'started.json', {**identity_record, 'started_utc': datetime.now(timezone.utc).isoformat(),
                                    'deadline_seconds': limit, 'pid': os.getpid()})
    started = time.monotonic()
    device = config['device']
    try:
        selected, warm, aux, records, load_seconds = _select(config, parent, prep, local, arm, started, limit)
        if command == 'stage1':
            one = config['stage1']
            result = stage1_context(selected, chunk_size=one['chunk_size'], repetitions=one['timing_repetitions'],
                                    uneven_chunk_sizes=one['uneven_chunk_sizes'], timing_batch_size=one['timing_minibatch_size'], device=device)
        elif command == 'stage2':
            two = config['stage2']
            result = stage2_backend(selected['train'], device=device, minibatches=two['minibatches'], batch_size=two['batch_size'],
                                    seed=two['seed'], width=two['width'], learning_rate=two['learning_rate'],
                                    chunk_size=config['stage1']['chunk_size'], tolerances=two['tolerances'])
        else:
            cache, construction = None, None
            if path == 'cached':
                rows = np.unique(np.concatenate([warm.rows, *[b.rows for b in selected.values()]]))
                cache, construction = build_cache(selected['train'].context, selected['train'].store.frame, rows,
                                                  device=device, chunk_size=config['stage1']['chunk_size'])
            result, artifacts = stage3_fit(selected, warm, aux['delivery'], device=device, width=config['stage2']['width'],
                                           budget=config['stage3']['budget'], cache=cache, path=path)
            result['construction'] = construction
            np.savez_compressed(directory/'artifacts.npz', probability=artifacts['probability'], raw=artifacts['raw'],
                                tiers=artifacts['tiers'], **{'state.'+k: v for k, v in artifacts['state'].items()})
        if result.get('device', device) != device:
            raise ValueError('Stage ran on a different device than registered')
        result.update(arm=arm, path=path, attempt=attempt, dependencies=dependencies, load_seconds=load_seconds, samples=records,
                      deadline_seconds=limit, seconds_total=time.monotonic()-started, adoption=None)
        if result['seconds_total'] > limit: raise TimeoutError('Registered command deadline exceeded')
        if source_hashes() != expected['source_hashes']: raise ValueError('Audit source changed during the stage')
        dump(directory/'results.json', result)
        seal(directory, identity_record)
        passed = stage_passed(command, result)
        print('CACHE_AUDIT_STAGE_COMPLETE', command, arm, path, 'passed' if passed else 'NOT_EQUIVALENT', flush=True)
        return passed
    except BaseException as error:
        dump(directory/'failure.json', {**identity_record, 'seconds': time.monotonic()-started, 'error': repr(error),
                                        'preserve_partial': True, 'equivalence': None})
        raise


def _load_artifacts(directory):
    with np.load(directory/'artifacts.npz') as data:
        return {'probability': data['probability'], 'raw': data['raw'], 'tiers': data['tiers'],
                'state': {k[len('state.'):]: data[k] for k in data.files if k.startswith('state.')}}


def compare(config, output, expected, arm, attempt):
    registration = verify_registration(output, expected)
    dependencies = predecessors(output, registration, 'compare', arm)
    evidence = {path: stage_root(output, 'stage3', arm, path) for path in PATHS}
    directory = fresh(stage_root(output, 'compare', arm)/f'attempt{attempt}')
    identity_record = stage_identity(registration, 'compare', arm, None, attempt, dependencies)
    started = time.monotonic()
    try:
        found = {path: completed_attempt(output, registration, 'stage3', arm, path,
                                         {k: v for k, v in dependencies.items() if not k.startswith('stage3/')})[0] for path in PATHS}
        results = {path: read_json(found[path]/'results.json') for path in PATHS}
        artifacts = {path: _load_artifacts(found[path]) for path in PATHS}
        for path in PATHS:
            if results[path]['device'] != config['device'] or results[path]['long_length'] != CELLS[arm]:
                raise ValueError('Stage3 evidence device/arm differs from registration')
        result = compare_stage3(results['original'], artifacts['original'], results['cached'], artifacts['cached'],
                                tolerances=config['stage3']['tolerances'])
        result.update(arm=arm, attempt=attempt, dependencies=dependencies, seconds_total=time.monotonic()-started, adoption=None)
        dump(directory/'results.json', result)
        seal(directory, identity_record)
        print('CACHE_AUDIT_COMPARE', arm, 'passed' if result['equivalent'] else 'NOT_EQUIVALENT', flush=True)
        return bool(result['equivalent'])
    except BaseException as error:
        dump(directory/'failure.json', {**identity_record, 'seconds': time.monotonic()-started, 'error': repr(error), 'preserve_partial': True})
        raise


def summary(config, local, output, expected, attempt):
    registration = verify_registration(output, expected)
    directory = fresh(stage_root(output, 'summary')/f'attempt{attempt}')
    identity_record = stage_identity(registration, 'summary', None, None, attempt)
    started = time.monotonic()
    try:
        stages, dependencies, constructions, stage3 = {}, {}, {}, {}
        for arm in CELLS:
            chain = predecessors(output, registration, 'compare', arm)
            found, digest = completed_attempt(output, registration, 'compare', arm, None, chain)
            comparison = read_json(found/'results.json')
            if not comparison['equivalent']: raise ValueError('Stage3 comparison did not pass for ' + arm)
            if comparison['tolerances'] != config['stage3']['tolerances']: raise ValueError('Stage3 tolerances differ')
            dependencies.update(chain); dependencies[f'compare/{arm}'] = digest
            stages[arm] = {'compare': comparison}
            for command in ('stage1', 'stage2'):
                found, _ = completed_attempt(output, registration, command, arm, None,
                                             {} if command == 'stage1' else {f'stage1/{arm}': chain[f'stage1/{arm}']})
                result = read_json(found/'results.json')
                if result['long_length'] != CELLS[arm] or result['device'] != config['device']:
                    raise ValueError('Stage evidence arm/device differs: ' + command)
                if command == 'stage2' and result['tolerances'] != config['stage2']['tolerances']: raise ValueError('Stage2 tolerances differ')
                stages[arm][command] = result
            stage3[arm] = {}
            for path in PATHS:
                found, _ = completed_attempt(output, registration, 'stage3', arm, path, {k: v for k, v in chain.items() if not k.startswith('stage3/')})
                stage3[arm][path] = read_json(found/'results.json')
            constructions[arm] = {'stage1_union_rows': stages[arm]['stage1']['construction'],
                                  'stage3_fit_process_rows': stage3[arm]['cached']['construction']}
        registration_record = read_json(output/'registration.json')
        ledger = ledger_state(ledger_location(local, output, config['limits']['wall_ledger_dir']))
        prior = prior_seconds(config)
        costs = cost_summary(read_json(output/'parent_profiles.json'), stage3, constructions, registration_record['population_counts'],
                             prior_seconds=prior, audit_seconds=ledger['charged_seconds'])
        result = {'protocol': 'f4_cache_audit_summary_v2', 'all_stages_complete': True,
                  'stage1_all_bitwise': all(stage_passed('stage1', s['stage1']) for s in stages.values()),
                  'stage2_all_equivalent': all(s['stage2']['equivalent'] for s in stages.values()),
                  'stage3_all_equivalent': all(s['compare']['equivalent'] for s in stages.values()),
                  'per_arm': {arm: {'stage1_bitwise': stage_passed('stage1', s['stage1']), 'stage2_equivalent': s['stage2']['equivalent'],
                                    'stage3_equivalent': s['compare']['equivalent'], 'stage1_timing': s['stage1']['timing']['seconds'],
                                    'stage3_timing': s['compare']['timing']} for arm, s in stages.items()},
                  'costs': costs, 'wall_ledger': {**ledger,
                      'treatment': 'Supervisor-owned caller wall per launched job; unresolved jobs, including this summary while it runs, are charged at their full cap (conservative); the summary job\'s final wall is written by its supervisor after this file is sealed'},
                  'dependencies': dependencies, 'dev_scores_read': False, 'adoption': None, 'full_fit_started': False,
                  'seconds_total': time.monotonic()-started,
                  'scope': 'Numerical/resource screen of an unadopted cache; root decides adoption and any fresh F4 version separately'}
        dump(directory/'results.json', result)
        seal(directory, identity_record)
        print('CACHE_AUDIT_SUMMARY', result['stage1_all_bitwise'], result['stage2_all_equivalent'], result['stage3_all_equivalent'], costs['decision'], flush=True)
    except BaseException as error:
        dump(directory/'failure.json', {**identity_record, 'seconds': time.monotonic()-started, 'error': repr(error), 'preserve_partial': True})
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True, help='configs/EXP-P4-002-v3-cache-audit-v2.yaml (JSON)')
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='fresh sibling audit directory under the ML matrix run root')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest='command', required=True)
    helps = {'prepare': 'register: pin parent/config/source/contract/environment identities',
             'stage1': 'bitwise context/override/guard checks and timing',
             'stage2': 'MPS forward/loss/gradient/updated-weight equivalence (needs passed stage1)',
             'stage3': 'paired v3-style resource fit on one path (needs passed stage1+2; 600-second profile cap)',
             'compare': 'compare sealed original/cached stage3 attempts for one arm',
             'summary': 'all-arm summary and honest cost ledger; no adoption (needs three passed comparisons)'}
    for name in COMMANDS:
        command = sub.add_parser(name, help=helps[name])
        if name in ('stage1', 'stage2', 'stage3', 'compare'):
            command.add_argument('--arm', choices=list(CELLS), required=True)
        if name == 'stage3':
            command.add_argument('--path', choices=list(PATHS), required=True)
        command.add_argument('--attempt', type=int, required=True,
                             help='fresh attempt number; existing attempts and ledger jobs are never overwritten')
    return parser


def worker_argv(argv_tail):
    """Parent options precede the subcommand; argparse rejects --worker after subcommand arguments."""
    return [sys.executable, str(Path(__file__).resolve()), '--worker', *argv_tail]


def main():
    args = build_parser().parse_args()
    config = config_check(read_json(args.config), real=True)
    local = read_json(args.local_config)
    output = args.output.resolve()
    if not args.worker:
        outcome, code = supervise(args.command, getattr(args, 'arm', None), getattr(args, 'path', None), args.attempt,
                                  config, local, output, worker_argv(sys.argv[1:]))
        sys.exit(124 if outcome == 'timeout' else code)
    root = check_location(local, output)
    validate_native_runtime()
    if not torch.backends.mps.is_available(): raise RuntimeError('Actual MPS backend required for the registered real audit')
    verify_pins(config)
    expected = identity(config, args.local_config)
    with heavy_lock(root):
        if args.command == 'prepare': prepare(config, local, output, expected); passed = True
        elif args.command in ('stage1', 'stage2'): passed = run_stage(config, local, output, expected, args.command, args.arm, None, args.attempt)
        elif args.command == 'stage3': passed = run_stage(config, local, output, expected, 'stage3', args.arm, args.path, args.attempt)
        elif args.command == 'compare': passed = compare(config, output, expected, args.arm, args.attempt)
        else: summary(config, local, output, expected, args.attempt); passed = True
    sys.exit(0 if passed else EXIT_NOT_EQUIVALENT)


if __name__ == '__main__':
    main()
