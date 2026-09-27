"""Real-data audit runner for the optional F4 observed-context cache (COOP-001).

Fresh sibling audit directory only; original v3 preparation/profiles are read
for identity and metadata and never modified. Every command holds the shared
heavy lock inside a worker process that a supervisor kills at the registered
deadline. Only TRAIN/early-stop/May rows are loaded. No DEV/June cache, no
quality metric, no full member fit and no adoption decision are produced here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
REPO = PROJECT.parents[1]

import numpy as np
import torch

from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_long_experiment import CELLS
from pitchmdp.matrix_policy_artifacts import is_appledouble, artifact_names
from pitchmdp.matrix_context_cache_audit import (REGISTERED_SAMPLES, WARMUP_TRAIN, STAGE1_ORDERS, STAGE1_MUST_FAIL,
    STAGE2_TOLERANCES, STAGE3_TOLERANCES, STAGE3_BUDGET, select_samples, sample_records, check_sample_identity,
    build_cache, stage1_context, stage2_backend, stage3_fit, compare_stage3, cost_summary)
from run_ml_long_history import SOURCES as F4_SOURCES
from audit_ml_long_equivalence import load_batches
from run_ml_benchmark import read_json, dump, identity as base_identity, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock, assert_hashes, artifact_hashes

PROTOCOL = 'f4_observed_context_cache_audit_v2'
SOURCES = list(dict.fromkeys([*F4_SOURCES, 'pitchmdp/matrix_long_equivalence.py', 'scripts/audit_ml_long_equivalence.py',
    'pitchmdp/matrix_observed_context_cache.py', 'pitchmdp/matrix_policy_artifacts.py',
    'pitchmdp/matrix_context_cache_audit.py', 'scripts/audit_ml_context_cache.py']))
COMMANDS = ('prepare', 'stage1', 'stage2', 'stage3', 'compare', 'summary')
PATHS = ('original', 'cached')
HEX = set('0123456789abcdef')


def _sha(value, length=64):
    return isinstance(value, str) and len(value) == length and set(value) <= HEX


def _pin(value, length=64, *, real):
    if value is None:
        if real: raise ValueError('Null identity pin is not allowed in a real audit; freeze the registration first')
        return
    if not _sha(value, length): raise ValueError('Identity pin must be a lowercase hex digest')


def _seconds(value, upper=None):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
            and (upper is None or value <= upper))


def config_check(config, *, real=True):
    required = {'protocol', 'experiment_id', 'contract', 'historical_registration', 'repo_commit', 'parent',
                'sources', 'device', 'arms', 'samples', 'stage1', 'stage2', 'stage3', 'limits',
                'prior_cost_ledger', 'adoption'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Exact F4 cache-audit v2 configuration schema required')
    if config['protocol'] != PROTOCOL or config['experiment_id'] != 'EXP-P4-002' or config['adoption'] is not None:
        raise ValueError('Registered protocol/experiment and null adoption required')
    if config['device'] != 'mps' or config['arms'] != list(CELLS):
        raise ValueError('Audit fixes the actual MPS backend and all three arms')
    _pin(config['repo_commit'], 40, real=real)
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
    if (set(one) != {'orders', 'must_fail', 'tolerance', 'chunk_size', 'timing_repetitions', 'uneven_chunk_sizes', 'timing_minibatch_size'}
            or one['orders'] != list(STAGE1_ORDERS) or one['must_fail'] != list(STAGE1_MUST_FAIL) or one['tolerance'] != 'bitwise'
            or type(one['chunk_size']) is not int or one['chunk_size'] < 1 or type(one['timing_repetitions']) is not int
            or one['timing_repetitions'] < 2 or not one['uneven_chunk_sizes'] or any(type(s) is not int or s < 1 for s in one['uneven_chunk_sizes'])
            or one['timing_minibatch_size'] != 256):
        raise ValueError('Stage1 orders, bitwise tolerance, chunking and alternating timing are fixed')
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
    if (set(limits) != {'command_seconds', 'single_member_wall_limit_seconds', 'family_wall_budget_seconds',
                        'no_seed_data_epoch_batch_draw_reduction', 'full_fit_started'}
            or limits['single_member_wall_limit_seconds'] != 7200 or limits['family_wall_budget_seconds'] != 28800
            or limits['no_seed_data_epoch_batch_draw_reduction'] is not True or limits['full_fit_started'] is not False
            or set(limits['command_seconds']) != set(COMMANDS)
            or not all(_seconds(limits['command_seconds'][name], 7200) for name in COMMANDS)):
        raise ValueError('Per-command deadlines within the 7200-second member cap and the frozen 28800 family budget required')
    ledger = config['prior_cost_ledger']
    if ledger is None:
        if real: raise ValueError('Owner prior-cost ledger pin required for a real audit')
    elif set(ledger) != {'path', 'sha256'} or not _sha(ledger['sha256']):
        raise ValueError('Prior-cost ledger needs a path and digest')
    return config


def source_hashes():
    return {name: hash_file(PROJECT/name) for name in SOURCES}


def identity(config, local_path):
    return {**base_identity(config, local_path), 'source_hashes': source_hashes(), 'repo_commit': config['repo_commit']}


def verify_pins(config):
    hashes = source_hashes()
    for name, digest in config['sources'].items():
        if hashes[name] != digest: raise ValueError('Pinned audit source changed: ' + name)
    registration = config['historical_registration']
    if hash_file(REPO/registration['config']) != registration['sha256']:
        raise ValueError('Historical v1 registration file changed; preserve it')
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != config['repo_commit']:
        raise ValueError('Checked-out commit differs from the registered repo_commit')


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


def stage_identity(registration, command, arm, path, attempt):
    return {'registration_sha256': registration, 'stage': command, 'arm': arm, 'path': path, 'attempt': attempt}


def completed_attempt(output, registration, command, arm=None, path=None):
    """Exactly one sealed attempt is the evidence; failed attempts are preserved and ignored."""
    root = stage_root(output, command, arm, path)
    candidates = sorted(p for p in root.glob('attempt*') if (p/'manifest.json').is_file()) if root.exists() else []
    if len(candidates) != 1:
        raise ValueError(f'Exactly one sealed {command} attempt required for {arm}/{path}; found {len(candidates)}')
    directory = candidates[0]
    attempt = int(directory.name[len('attempt'):])
    return directory, sealed(directory, stage_identity(registration, command, arm, path, attempt))


def prepare(config, local, output, expected):
    if output.exists() and any(p.is_file() and not is_appledouble(p) for p in output.rglob('*')):
        raise ValueError('Preserve the existing audit directory; register a fresh sibling')
    parent, prep, profiles = verify_parent(config, local)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output) or 'members' in output.parts:
        raise ValueError('Audit requires a distinct sibling outside the parent and any member directory')
    output.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, target)
    dump(output/'registered_config.json', config)
    dump(output/'original_preparation.json', prep)
    dump(output/'parent_profiles.json', profiles)
    populations = {name: int(prep['samples'][name]['n']) for name in ('train', 'earlystop', 'temperature', 'blend', 'dev')}
    dump(output/'registration.json', {'identity': expected, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent': config['parent'], 'population_counts': populations,
        'interpretation': 'Resource/numerical audit of an unadopted cache; no DEV/June cache, no quality, no full fit'})
    dump(output/'registration_manifest.json', {'identity': expected, 'artifact_hashes': artifact_hashes(output, artifact_names(output))})
    print('CACHE_AUDIT_PREPARED', output, flush=True)


def verify_registration(output, expected):
    manifest = read_json(output/'registration_manifest.json')
    if manifest['identity'] != expected: raise ValueError('Audit config/source/environment changed since registration')
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
    registration = verify_registration(output, expected)
    parent, prep, _ = verify_parent(config, local)
    directory = fresh(stage_root(output, command, arm, path)/f'attempt{attempt}')
    identity_record = stage_identity(registration, command, arm, path, attempt)
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
        result.update(arm=arm, path=path, attempt=attempt, load_seconds=load_seconds, samples=records,
                      deadline_seconds=limit, seconds_total=time.monotonic()-started, adoption=None)
        if result['seconds_total'] > limit: raise TimeoutError('Registered command deadline exceeded')
        if source_hashes() != expected['source_hashes']: raise ValueError('Audit source changed during the stage')
        dump(directory/'results.json', result)
        seal(directory, identity_record)
        print('CACHE_AUDIT_STAGE_COMPLETE', command, arm, path, result.get('equivalent', result.get('all_bitwise_identical')), flush=True)
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
    evidence = {path: completed_attempt(output, registration, 'stage3', arm, path) for path in PATHS}
    directory = fresh(stage_root(output, 'compare', arm)/f'attempt{attempt}')
    identity_record = stage_identity(registration, 'compare', arm, None, attempt)
    started = time.monotonic()
    try:
        results = {path: read_json(evidence[path][0]/'results.json') for path in PATHS}
        artifacts = {path: _load_artifacts(evidence[path][0]) for path in PATHS}
        for path in PATHS:
            if results[path]['device'] != config['device'] or results[path]['long_length'] != CELLS[arm]:
                raise ValueError('Stage3 evidence device/arm differs from registration')
        result = compare_stage3(results['original'], artifacts['original'], results['cached'], artifacts['cached'],
                                tolerances=config['stage3']['tolerances'])
        result.update(arm=arm, attempt=attempt, dependencies={path: evidence[path][1] for path in PATHS},
                      seconds_total=time.monotonic()-started, adoption=None)
        dump(directory/'results.json', result)
        seal(directory, identity_record)
        print('CACHE_AUDIT_COMPARE', arm, result['equivalent'], flush=True)
    except BaseException as error:
        dump(directory/'failure.json', {**identity_record, 'seconds': time.monotonic()-started, 'error': repr(error), 'preserve_partial': True})
        raise


def audit_costs(output):
    """Every attempt, sealed or failed or killed, is a real cost of this audit."""
    total, entries = 0., []
    for command in ('stage1', 'stage2', 'stage3', 'compare'):
        root = output/command
        if not root.exists(): continue
        for record in sorted(root.rglob('*.json')):
            if record.name in ('results.json', 'failure.json', 'timeout.json') and record.parent.name.startswith('attempt'):
                value = read_json(record)
                seconds = value.get('seconds_total', value.get('seconds', value.get('elapsed_seconds')))
                if seconds is None or not _seconds(seconds+1e-9): raise ValueError('Attempt cost record lacks finite seconds: ' + str(record))
                total += seconds; entries.append({'path': str(record.relative_to(output)), 'seconds': seconds})
    return total, entries


def prior_seconds(config):
    record = config['prior_cost_ledger']
    if hash_file(Path(record['path'])) != record['sha256']: raise ValueError('Owner prior-cost ledger changed')
    ledger = read_json(Path(record['path']))
    if ledger.get('protocol') != 'ml_long_owner_budget_ledger_v1' or not ledger.get('entries'):
        raise ValueError('Owner ledger protocol/entries differ')
    total = math.fsum(row['seconds'] for row in ledger['entries'])
    if not math.isclose(total, ledger['elapsed_seconds_total'], rel_tol=0, abs_tol=1e-6): raise ValueError('Owner ledger total differs from entries')
    return total


def summary(config, output, expected, attempt):
    registration = verify_registration(output, expected)
    directory = fresh(stage_root(output, 'summary')/f'attempt{attempt}')
    identity_record = stage_identity(registration, 'summary', None, None, attempt)
    started = time.monotonic()
    try:
        stages, dependencies, constructions, stage3 = {}, {}, {}, {}
        for arm in CELLS:
            stages[arm] = {}
            for command in ('stage1', 'stage2', 'compare'):
                found, digest = completed_attempt(output, registration, command, arm)
                result = read_json(found/'results.json')
                if result['long_length'] != CELLS[arm] or result.get('device', config['device']) != config['device']:
                    raise ValueError('Stage evidence arm/device differs: ' + command)
                if command == 'stage2' and result['tolerances'] != config['stage2']['tolerances']: raise ValueError('Stage2 tolerances differ')
                if command == 'compare' and result['tolerances'] != config['stage3']['tolerances']: raise ValueError('Stage3 tolerances differ')
                stages[arm][command] = result; dependencies[f'{command}/{arm}'] = digest
            stage3[arm] = {}
            for path in PATHS:
                found, digest = completed_attempt(output, registration, 'stage3', arm, path)
                stage3[arm][path] = read_json(found/'results.json'); dependencies[f'stage3/{arm}/{path}'] = digest
            constructions[arm] = {'stage1_union_rows': stages[arm]['stage1']['construction'],
                                  'stage3_fit_process_rows': stage3[arm]['cached']['construction']}
        registration_record = read_json(output/'registration.json')
        audit_seconds, entries = audit_costs(output)
        costs = cost_summary(read_json(output/'parent_profiles.json'), stage3, constructions, registration_record['population_counts'],
                             prior_seconds=prior_seconds(config), audit_seconds=audit_seconds)
        result = {'protocol': 'f4_cache_audit_summary_v1', 'all_stages_complete': True,
                  'stage1_all_bitwise': all(s['stage1']['all_bitwise_identical'] and s['stage1']['all_guards_rejected'] for s in stages.values()),
                  'stage2_all_equivalent': all(s['stage2']['equivalent'] for s in stages.values()),
                  'stage3_all_equivalent': all(s['compare']['equivalent'] for s in stages.values()),
                  'per_arm': {arm: {'stage1_bitwise': s['stage1']['all_bitwise_identical'], 'stage2_equivalent': s['stage2']['equivalent'],
                                    'stage3_equivalent': s['compare']['equivalent'], 'stage1_timing': s['stage1']['timing']['seconds'],
                                    'stage3_timing': s['compare']['timing']} for arm, s in stages.items()},
                  'costs': costs, 'audit_cost_entries': entries, 'dependencies': dependencies,
                  'dev_scores_read': False, 'adoption': None, 'full_fit_started': False,
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
    sub.add_parser('prepare', help='register: pin parent/config/source/environment identities')
    for name in ('stage1', 'stage2'):
        command = sub.add_parser(name, help={'stage1': 'bitwise context/override/guard checks and timing',
                                             'stage2': 'MPS forward/loss/gradient/updated-weight equivalence'}[name])
        command.add_argument('--arm', choices=list(CELLS), required=True)
        command.add_argument('--attempt', type=int, required=True, help='fresh attempt number; existing attempts are never overwritten')
    three = sub.add_parser('stage3', help='paired v3-style resource fit on one path')
    three.add_argument('--arm', choices=list(CELLS), required=True)
    three.add_argument('--path', choices=list(PATHS), required=True)
    three.add_argument('--attempt', type=int, required=True)
    comparison = sub.add_parser('compare', help='compare sealed original/cached stage3 attempts for one arm')
    comparison.add_argument('--arm', choices=list(CELLS), required=True)
    comparison.add_argument('--attempt', type=int, required=True)
    final = sub.add_parser('summary', help='all-arm summary and honest cost ledger; no adoption')
    final.add_argument('--attempt', type=int, required=True)
    return parser


def supervise(args, config):
    """Hard deadline: the worker (which holds the heavy lock) is terminated then killed."""
    limit = config['limits']['command_seconds'][args.command]
    argv = [sys.executable, str(Path(__file__).resolve()), '--worker', *sys.argv[1:]]  # parent option precedes the subcommand
    started = time.monotonic()
    child = subprocess.Popen(argv)
    try:
        code = child.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        child.terminate()
        try: child.wait(timeout=15)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        elapsed = time.monotonic()-started
        output = args.output.resolve()
        arm, path, attempt = getattr(args, 'arm', None), getattr(args, 'path', None), getattr(args, 'attempt', None)
        target = stage_root(output, args.command, arm, path)/f'attempt{attempt}' if attempt is not None else output
        record = {'protocol': 'f4_cache_audit_caller_outcome_v1', 'outcome': 'timeout', 'command': args.command, 'arm': arm,
                  'path': path, 'attempt': attempt, 'elapsed_seconds': elapsed, 'limit_seconds': limit, 'argv': argv,
                  'killed_utc': datetime.now(timezone.utc).isoformat(), 'preserve_partial': True}
        if target.is_dir(): dump(target/'timeout.json', record)
        else: print('CACHE_AUDIT_TIMEOUT', record, file=sys.stderr, flush=True)
        print('CACHE_AUDIT_TIMEOUT', args.command, arm, path, attempt, elapsed, flush=True)
        sys.exit(124)
    sys.exit(code)


def main():
    args = build_parser().parse_args()
    config = config_check(read_json(args.config), real=True)
    if not args.worker:
        supervise(args, config)
        return
    local = read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    if not torch.backends.mps.is_available(): raise RuntimeError('Actual MPS backend required for the registered real audit')
    verify_pins(config)
    expected = identity(config, args.local_config)
    with heavy_lock(root):
        if args.command == 'prepare': prepare(config, local, output, expected)
        elif args.command in ('stage1', 'stage2'): run_stage(config, local, output, expected, args.command, args.arm, None, args.attempt)
        elif args.command == 'stage3': run_stage(config, local, output, expected, 'stage3', args.arm, args.path, args.attempt)
        elif args.command == 'compare': compare(config, output, expected, args.arm, args.attempt)
        else: summary(config, output, expected, args.attempt)


if __name__ == '__main__':
    main()
