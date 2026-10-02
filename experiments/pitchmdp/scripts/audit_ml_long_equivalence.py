"""Optional MPS numerical audit; original F4 profiles must precede registration.

Distinct audit directory only; never write valid full-member outputs. The sole
owner supplies a subprocess timeout. No actual work occurs at import time.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import math
import pickle
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
import pandas as pd
import torch

from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_benchmark import select_keys
from pitchmdp.matrix_long_history import MatrixLongHistoryStore, LazyPitchBatch
from pitchmdp.matrix_long_equivalence import TOLERANCES, LIMITS, FIXED_TEMPERATURES, audit_batches
from pitchmdp.matrix_long_experiment import CELLS
from pitchmdp.matrix_sharing import ContinuousPitcherContext
from run_ml_long_history import SOURCES as F4_SOURCES, auxiliary_identity
from run_ml_benchmark import read_json, dump, regular_frame, identity as base_identity, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock, assert_hashes, artifact_hashes

REFERENCE_LAZY_SHA256 = '8db3d08d4bfab5539baffb1450290a71800a5ed8482c11b18c9c713c8812515b'
OPTIMIZED_LAZY_SHA256 = '4203aa36099c0fb315b40a541b25d7cd475023025a90d8a557f490cabb86fb78'
LAZY = 'pitchmdp/matrix_lazy_model.py'
SOURCES = [*F4_SOURCES, 'pitchmdp/matrix_long_equivalence.py', 'scripts/audit_ml_long_equivalence.py']


def config_check(config):
    required = {'protocol', 'original_f4_run', 'original_preparation_sha256', 'original_profile_evidence',
                'device', 'cells', 'tolerances', 'limits', 'fixed_temperatures', 'cell_seconds'}
    if set(config) != required or config['protocol'] != 'ml_long_equivalence_v1':
        raise ValueError('Exact optional-equivalence registration required')
    if (config['device'] != 'mps' or config['cells'] != list(CELLS) or config['tolerances'] != TOLERANCES
        or config['limits'] != LIMITS or config['fixed_temperatures'] != list(FIXED_TEMPERATURES)
        or config['cell_seconds'] != 1200 or set(config['original_profile_evidence']) != set(CELLS)):
        raise ValueError('Audit fixes MPS, all three arms, sample counts, tolerances and resource cap')
    records = [{'sha256': config['original_preparation_sha256']}, *config['original_profile_evidence'].values()]
    for record in records:
        digest = record['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Frozen original preparation/profile evidence SHA256 required')
    evidence = config['original_profile_evidence'].values()
    if (any(set(record) != {'kind', 'path', 'sha256'} or record['kind'] not in
             ('completed_profile_manifest', 'caller_failure_ledger') for record in evidence)
        or len({str(Path(record['path']).resolve()) for record in evidence}) != len(CELLS)):
        raise ValueError('Distinct typed per-arm original-profile evidence required')
    return config


def source_hashes(): return {name: hash_file(PROJECT/name) for name in SOURCES}


def identity(config, local_path):
    return {**base_identity(config, local_path), 'source_hashes': source_hashes()}


def seal(directory, expected):
    files = [str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json']
    dump(directory/'manifest.json', {'identity': expected, 'artifact_hashes': artifact_hashes(directory, files)})


def sealed(directory, expected):
    value = read_json(directory/'manifest.json')
    files = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if value['identity'] != expected or files != set(value['artifact_hashes']):
        raise ValueError('Sealed audit stage identity/artifact family differs')
    assert_hashes(directory, value['artifact_hashes'])
    return value


def verify_profile_evidence(config, parent, prep):
    for cell, record in config['original_profile_evidence'].items():
        evidence = Path(record['path']).resolve()
        if hash_file(evidence) != record['sha256']:
            raise ValueError('Original profile outcome/timeout evidence changed')
        value = read_json(evidence)
        expected = {'preparation_sha256': config['original_preparation_sha256'], 'cell': cell}
        if record['kind'] == 'completed_profile_manifest':
            if evidence != (parent/'profiles'/cell/'manifest.json').resolve() or value['identity'] != expected:
                raise ValueError('Completed original profile arm/preparation identity differs')
            if set(value['artifact_hashes']) != {'profile.json'}:
                raise ValueError('Original profile artifact family differs')
            assert_hashes(evidence.parent, value['artifact_hashes'])
            report = read_json(evidence.parent/'profile.json')
            if report['long_length'] != CELLS[cell] or report['dev_scores_read'] is not False:
                raise ValueError('Original profile arm or no-DEV-score contract differs')
        else:
            required = {'protocol', 'original_f4_run', 'preparation_sha256', 'cell', 'command',
                        'outcome', 'elapsed_seconds', 'source_hashes'}
            if (set(value) != required or value['protocol'] != 'ml_long_profile_caller_outcome_v1'
                or any(value[k] != v for k, v in expected.items()) or value['command'] != 'profile'
                or Path(value['original_f4_run']).resolve() != parent
                or value['outcome'] not in ('timeout', 'failure')
                or not isinstance(value['elapsed_seconds'], (float, int)) or isinstance(value['elapsed_seconds'], bool)
                or not math.isfinite(value['elapsed_seconds']) or value['elapsed_seconds'] <= 0
                or value['source_hashes'] != prep['identity']['source_hashes']):
                raise ValueError('Caller profile failure ledger arm/preparation/source/elapsed identity differs')


def original(config, local):
    parent = Path(config['original_f4_run']).resolve(); check_location(local, parent)
    if hash_file(parent/'preparation.json') != config['original_preparation_sha256']:
        raise ValueError('Original F4 preparation changed')
    prep = read_json(parent/'preparation.json'); assert_hashes(parent, prep['artifact_hashes'])
    if prep['features']['long_stream']['version'] != 'batter_dual_stream_v2':
        raise ValueError('Original F4 preparation must include reviewed identity-only repair')
    expected = prep['identity']['source_hashes']
    if expected[LAZY] != REFERENCE_LAZY_SHA256 or hash_file(PROJECT/LAZY) != OPTIMIZED_LAZY_SHA256:
        raise ValueError('Only the reviewed optional lazy-model source change is allowed')
    for name, digest in expected.items():
        if hash_file(parent/'source'/name) != digest:
            raise ValueError('Original frozen source copy changed: '+name)
        if name != LAZY and hash_file(PROJECT/name) != digest:
            raise ValueError('Non-optimized original source changed: '+name)
    verify_profile_evidence(config, parent, prep)
    return parent, prep


def prepare(config, local, output, expected):
    if output.exists(): raise ValueError('Preserve existing audit directory; no overwrite/resume')
    parent, prep = original(config, local)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Audit requires a distinct sibling, outside valid full members')
    if 'members' in output.parts: raise ValueError('Audit output cannot be a member directory')
    output.mkdir(parents=True)
    for name in SOURCES:
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, target)
    dump(output/'registered_config.json', config)
    dump(output/'original_preparation.json', prep)
    dump(output/'registration.json', {'identity': expected, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'original_preparation_sha256': config['original_preparation_sha256'],
        'interpretation': 'Backend numerical screen only; no full-member validity or DEV quality claims'})
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    dump(output/'registration_manifest.json', {'identity': expected, 'artifact_hashes': artifact_hashes(output, files)})


def verify_registration(output, expected):
    manifest = read_json(output/'registration_manifest.json')
    if manifest['identity'] != expected: raise ValueError('Audit source/config/environment changed')
    assert_hashes(output, manifest['artifact_hashes'])
    return hash_file(output/'registration_manifest.json')


def load_batches(parent, prep, local, length):
    frame = regular_frame(local, read_json(parent/'parent_preparation.json'))
    with (parent/'aux.pkl').open('rb') as stream: aux = pickle.load(stream)
    if auxiliary_identity(aux) != prep['auxiliary_hashes']: raise ValueError('Frozen auxiliary identity changed')
    store = MatrixLongHistoryStore.from_frame(frame, normalizer=aux['normalizer'],
        type_vocabulary=prep['features']['h5']['type_vocabulary'], long_length=length)
    context = ContinuousPitcherContext(aux['context'], prep['clusters'])
    if store.base.report() != prep['features']['h5'] or context.report() != prep['features']['context']:
        raise ValueError('Frozen feature specification changed')
    batches = {}
    for name in LIMITS:  # Never read DEV/blend query lists or gather their inputs.
        record = prep['samples'][name]
        selected = select_keys(frame, pd.read_parquet(parent/record['path']), record)
        batches[name] = LazyPitchBatch(store, context, selected.index.to_numpy())
    return batches, aux


def run_cell(config, local, output, expected, cell):
    registration = verify_registration(output, expected)
    parent, prep = original(config, local)
    directory = output/'cells'/cell
    if directory.exists(): raise ValueError('Preserve completed or interrupted audit cell')
    directory.mkdir(parents=True)
    stage = {'registration_sha256': registration, 'cell': cell}
    dump(directory/'started.json', stage)
    start = time.monotonic()
    try:
        batches, aux = load_batches(parent, prep, local, CELLS[cell])
        remaining = config['cell_seconds']-(time.monotonic()-start)
        if remaining <= 0: raise TimeoutError('Audit cap exhausted during input load')
        result = audit_batches(batches, aux['delivery'], directory, device='mps', seconds_limit=remaining)
        result['load_and_audit_seconds'] = time.monotonic()-start
        if result['load_and_audit_seconds'] > config['cell_seconds']: raise TimeoutError('Audit cap exceeded')
        if source_hashes() != expected['source_hashes']: raise ValueError('Audit source changed during cell')
        dump(directory/'results.json', result)
        seal(directory, stage)
        print('LONG_EQUIVALENCE_CELL', cell, result['equivalent'], flush=True)
    except Exception as error:
        dump(directory/'failure.json', {'seconds': time.monotonic()-start, 'error': str(error),
             'preserve_partial': True, 'equivalence': None})
        raise


def summarize(output, expected):
    registration = verify_registration(output, expected)
    destination = output/'summary'
    if destination.exists(): raise ValueError('Preserve existing summary')
    rows, dependencies = {}, {}
    for cell, length in CELLS.items():
        directory = output/'cells'/cell
        sealed(directory, {'registration_sha256': registration, 'cell': cell})
        result = read_json(directory/'results.json')
        if result['device'] != 'mps' or result['long_length'] != length or result['tolerances'] != TOLERANCES:
            raise ValueError('Audit family arm/device/tolerance changed')
        rows[cell] = result
        dependencies[cell] = hash_file(directory/'manifest.json')
    destination.mkdir()
    dump(destination/'results.json', {'all_three_complete': True,
        'all_three_equivalent': all(row['equivalent'] for row in rows.values()),
        'cells': rows, 'dependencies': dependencies, 'dev_scores_read': False,
        'adoption': None, 'scope': 'May16 backend numerical screen; no automatic adoption or universal equivalence claim'})
    seal(destination, {'registration_sha256': registration, 'family': list(CELLS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare'); sub.add_parser('summarize')
    cell = sub.add_parser('cell'); cell.add_argument('--cell', choices=list(CELLS), required=True)
    args = parser.parse_args(); config = config_check(read_json(args.config)); local = read_json(args.local_config)
    output = args.output.resolve(); root = check_location(local, output)
    validate_native_runtime()
    if not torch.backends.mps.is_available(): raise RuntimeError('Actual MPS backend required for registered real equivalence audit')
    expected = identity(config, args.local_config)
    with heavy_lock(root):
        if args.command == 'prepare': prepare(config, local, output, expected)
        elif args.command == 'cell': run_cell(config, local, output, expected, args.cell)
        else:
            original(config, local)
            summarize(output, expected)


if __name__ == '__main__': main()
