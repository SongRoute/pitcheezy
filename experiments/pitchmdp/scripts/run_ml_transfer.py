"""Freeze a G comparison before six unchanged full-MLB predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
from pitchmdp.data import hash_file
from run_ml_sharing import (SOURCES as SHARING_SOURCES, SEEDS, identity as sharing_identity,
                            verify as verify_sharing, predict as predict_sharing)
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from score_ml_matrix import archive

SOURCES = [*SHARING_SOURCES, 'scripts/run_ml_transfer.py']
CONTROLS = {'G1-personal': 'G0-global', 'G2-feature': 'G0-global',
            'G3-cluster': 'G2-feature', 'G4-partial': 'G2-feature'}


def chosen_comparison(analysis):
    ranked = analysis['followup_candidates']
    promoted = bool(ranked)
    if not ranked:
        ranked = sorted(CONTROLS, key=lambda c: (analysis['reports'][c]['primary']['log_loss'],
                            analysis['logical_cell_costs'][c]['total_fit_seconds']))
    candidate = ranked[0]
    return {'candidate': candidate, 'control': CONTROLS[candidate],
            'status': 'screen_promoted' if promoted else 'diagnostic_only_not_promoted'}


def config_check(config):
    if (config['protocol'] != 'ml_transfer_v1' or config['seeds'] != list(SEEDS) or
        config['scope'] != 'Cmlb' or CONTROLS.get(config['candidate']) != config['control']):
        raise ValueError('Unregistered whole-MLB comparison')
    for name in ('parent_preparation_sha256', 'parent_analysis_sha256'):
        value = config[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Frozen parent hashes required')
    return config


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    return {**sharing_identity(config, local_path), 'source_hashes': source_hashes()}


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('Transfer source/config/environment changed')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Frozen transfer dependency changed')
    return prep


def prepare(config, local_path, local, output, expected):
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        print('TRANSFER_PREPARED', flush=True)
        return
    parent = Path(config['parent_run']).resolve()
    check_location(local, parent)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Transfer must use a separate sibling run')
    parent_config = read_json(parent / 'registered_config.json')
    shared = verify_sharing(parent, sharing_identity(parent_config, local_path))
    analysis_dir = parent / 'analysis' / 'panel'
    if (hash_file(parent / 'preparation.json') != config['parent_preparation_sha256'] or
        hash_file(analysis_dir / 'results.json') != config['parent_analysis_sha256']):
        raise ValueError('Transfer parent hashes differ')
    analysis = read_json(analysis_dir / 'results.json')
    chosen = chosen_comparison(analysis)
    if any(config[name] != chosen[name] for name in ('candidate', 'control')) or config['selection_status'] != chosen['status']:
        raise ValueError('Transfer comparison differs from predeclared selection rule')
    manifest = read_json(analysis_dir / 'manifest.json')
    if manifest['results_sha256'] != config['parent_analysis_sha256']:
        raise ValueError('Parent results manifest differs')
    external = dict(manifest['inputs'])
    for name in ('results.json', 'manifest.json', 'predictions.npz'):
        external[str(analysis_dir / name)] = hash_file(analysis_dir / name)
    if external[str(analysis_dir / 'predictions.npz')] != manifest['predictions_sha256']:
        raise ValueError('Parent predictions manifest differs')
    for cell in (config['candidate'], config['control']):
        for seed in SEEDS:
            member = parent / 'members' / cell / f'seed{seed}'
            state = read_json(member / 'prediction_state.json')
            assert_hashes(member, state['artifact_hashes'])
            for name, digest in state['artifact_hashes'].items():
                external[str(member / name)] = digest
            external.update(state['dependencies'])
    for name, digest in external.items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Parent input changed before transfer')
    output.mkdir(parents=True)
    for name in SOURCES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    for source, target in [('preparation.json', 'parent_preparation.json'),
        ('registered_config.json', 'parent_config.json'), ('mlb_dev_metadata.parquet', 'dev_metadata.parquet'),
        ('baseline_predictions.npz', 'baseline_predictions.npz'),
        ('analysis/panel/results.json', 'parent_analysis.json')]:
        shutil.copyfile(parent / source, output / target)
    dump(output / 'registered_config.json', config)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    dump(output / 'preparation.json', {'identity': expected, 'parent_run': str(parent),
        'external_hashes': external, 'artifact_hashes': artifact_hashes(output, files),
        'samples': {'dev': shared['samples']['mlb_dev']}, 'coverage': shared['coverage']['mlb_dev'],
        'selection': chosen, 'panel': shared['panel'], 'individual_eligibility': shared['individual_eligibility'],
        'scores_read': False})
    print('TRANSFER_PREPARED', flush=True)


def predict(config, local, output, prep, cell, seed):
    if cell not in (config['candidate'], config['control']) or seed not in SEEDS:
        raise ValueError('Unregistered transfer member')
    dest = output / 'members' / cell / f'seed{seed}'
    expected = {'preparation_sha256': hash_file(output / 'preparation.json'), 'cell': cell, 'seed': seed}
    if (dest / 'prediction_state.json').exists():
        state = read_json(dest / 'prediction_state.json')
        if state['identity'] != expected:
            raise ValueError('Transfer identity differs')
        assert_hashes(dest, state['artifact_hashes'])
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Incomplete transfer output requires failure review')
    parent = Path(prep['parent_run'])
    start = time.perf_counter()
    # Parent function checks calibrated panel identity/dependencies before the
    # existing optional full-population path. No source, weights or CAL changes.
    predict_sharing(read_json(output / 'parent_config.json'), local, parent,
                    read_json(output / 'parent_preparation.json'), cell, seed, mlb=True)
    member = parent / 'members' / cell / f'seed{seed}'
    state = read_json(member / 'mlb_prediction_state.json')
    assert_hashes(member, state['artifact_hashes'])
    raw = archive(member / 'mlb_predictions.npz')
    values = {'dev' + name[len('mlb_dev'):]: value for name, value in raw.items() if name.startswith('mlb_dev')}
    dest.mkdir(parents=True)
    np.savez_compressed(dest / 'predictions.npz', **values)
    shutil.copyfile(member / 'mlb_prediction_runtime.json', dest / 'prediction_runtime.json')
    dump(dest / 'parent_prediction_state.json', state)
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Transfer source changed during inference')
    dump(dest / 'prediction_state.json', {'identity': expected,
        'artifact_hashes': artifact_hashes(dest, ['predictions.npz', 'prediction_runtime.json', 'parent_prediction_state.json']),
        'seconds_including_parent': time.perf_counter()-start})
    print('TRANSFER_PREDICT_COMPLETE', cell, seed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    q = sub.add_parser('predict')
    q.add_argument('--cell', choices=list(set(CONTROLS) | set(CONTROLS.values())), required=True)
    q.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a = p.parse_args()
    config, local = config_check(read_json(a.config)), read_json(a.local_config)
    output = a.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, a.local_config)
    with heavy_lock(root):
        if a.command == 'prepare': prepare(config, a.local_config, local, output, expected)
        else: predict(config, local, output, verify(output, expected), a.cell, a.seed)
