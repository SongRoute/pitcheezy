"""Additive G0 scorer with explicit immutable artifact inputs and a fresh destination."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd
from pitchmdp.data import hash_file
from pitchmdp.matrix_g0_whole_metrics import (frozen_five_predictions, aligned_population,
                                            replay_panel, evaluate_whole)
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, check_location, heavy_lock
from score_ml_matrix import archive


SCORING = {'protocol': 'g0_whole_scoring_v1', 'seeds': [0, 1, 2, 3, 4],
    'bootstrap_draws': 10000, 'bootstrap_seed': 20260924, 'primary_slots': ['whole', 'non_cpanel'],
    'holm_alpha': .05, 'nll_delta_max': -.003, 'nll_ci_upper_max': 0., 'brier_ci_upper_max': .001,
    'negative_seeds_required': 4, 'minimum_games': 30, 'minimum_pitches': 500,
    'R': {'family_size': 24, 'draws': 100000, 'alpha': .05, 'nll_margin': .010, 'brier_margin': .002},
    'panel_replay_atol': 1e-6}


def score_artifacts(member_root, baseline_path, metadata_path, c1_results_path,
                    c1_predictions_path, destination, *, provenance=None, extra_inputs=None):
    """Caller verifies registered config/preparation/runtime and owns the wall ledger.

    Baseline archive is the pinned original G archive containing both dev and
    mlb_dev arrays. Member files are the new output's dev-named whole-MLB arrays.
    """
    member_root, destination = Path(member_root), Path(destination)
    if destination.exists():
        raise ValueError('Preserve completed or interrupted analysis; fresh destination required')
    paths = [Path(p) for p in (baseline_path, metadata_path, c1_results_path, c1_predictions_path)]
    inputs = {str(p.resolve()): hash_file(p) for p in paths}
    inputs.update(extra_inputs or {})
    for path, digest in inputs.items():
        if hash_file(Path(path)) != digest:
            raise ValueError('Registered input changed before scoring')
    saved = archive(Path(baseline_path))
    baseline = {'dev'+k[len('mlb_dev'):]: v for k, v in saved.items() if k.startswith('mlb_dev')}
    panel_frequency = {k: v for k, v in saved.items() if k.startswith('dev')}
    metadata = pd.read_parquet(metadata_path)
    members = []
    for seed in range(5):
        member = member_root / 'members' / 'G0-global' / f'seed{seed}'
        state_path = member / 'prediction_state.json'
        state = read_json(state_path)
        if state['identity']['seed'] != seed or state['identity']['cell'] != 'G0-global':
            raise ValueError('Member seed/cell identity differs')
        assert_hashes(member, state['artifact_hashes'])
        inputs[str(state_path.resolve())] = hash_file(state_path)
        for name, digest in state['artifact_hashes'].items():
            inputs[str((member / name).resolve())] = digest
        for name, digest in state.get('dependencies', {}).items():
            if hash_file(Path(name)) != digest:
                raise ValueError('Frozen model dependency changed')
            inputs[str(Path(name).resolve())] = digest
        members.append(archive(member / 'predictions.npz'))
    predictions = frozen_five_predictions(members, baseline, read_json(Path(c1_results_path))['reports']['G0-global'])
    _, indices, _ = aligned_population(baseline, metadata, panel_frequency)
    archived = archive(Path(c1_predictions_path))
    panel = {name: archived['G0-global_'+name] for name in ('raw', 'calibrated', 'primary', 'seed_primary')}
    replay = replay_panel(predictions, panel, indices)
    # All artifact identities and replay are checked before any new loss is opened.
    result = evaluate_whole(baseline, metadata, predictions, panel_frequency)
    result.update(provenance or {})
    result.update(replay=replay, input_hashes=inputs, scored_utc=datetime.now(timezone.utc).isoformat())
    for path, digest in inputs.items():
        if hash_file(Path(path)) != digest:
            raise ValueError('Input changed while scoring')
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz', keys=baseline['dev_keys'], y=baseline['dev_y'],
        game_pk=baseline['dev_game_pk'], pitcher=baseline['dev_pitcher'], **predictions)
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs,
        'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in (
            Path(__file__), PROJECT/'pitchmdp/matrix_g0_whole_metrics.py', PROJECT/'pitchmdp/matrix_metrics.py',
            PROJECT/'pitchmdp/matrix_group_metrics.py', PROJECT/'pitchmdp/matrix_panel.py')}})
    return result


def score_registered(config_path, local_path, output):
    # Imports are delayed so the pure artifact scorer remains independently testable.
    from pitchmdp.matrix_g0_whole import validate_config
    from run_ml_g0_whole import identity, verify, verify_member, SOURCES as WORKER_SOURCES
    config = validate_config(read_json(config_path), real=True)
    registration = config['registration']
    if registration.get('scoring') != SCORING:
        raise ValueError('Standalone G0 scoring rules differ from preregistration')
    if Path(registration['output']).resolve() != output.resolve():
        raise ValueError('Registered output differs')
    code = subprocess.check_output(['git', '-C', str(PROJECT.parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    if code != registration['execution_code_commit']:
        raise ValueError('Frozen execution code commit changed')
    required_sources = set(WORKER_SOURCES) | {
        'scripts/score_ml_g0_whole.py', 'scripts/score_ml_matrix.py',
        'pitchmdp/matrix_g0_whole_metrics.py', 'pitchmdp/matrix_metrics.py',
        'pitchmdp/matrix_group_metrics.py', 'pitchmdp/matrix_panel.py', 'pitchmdp/data.py'}
    if not required_sources <= set(registration.get('sources', {})):
        raise ValueError('Incomplete registered inference/scoring source closure')
    source_inputs = {}
    for name, digest in registration['sources'].items():
        path = (PROJECT / name).resolve()
        if not path.is_relative_to(PROJECT) or hash_file(path) != digest:
            raise ValueError('Frozen source registration changed: '+name)
        source_inputs[str(path)] = digest
    local = read_json(local_path)
    root = check_location(local, output)
    validate_native_runtime()
    with heavy_lock(root):
        prep = verify(output, identity(config, local_path))
        for seed in range(5):
            verify_member(output, prep, seed)
        inputs = {**prep['external_hashes'], **source_inputs,
                  str(config_path.resolve()): hash_file(config_path),
                  str(local_path.resolve()): hash_file(local_path),
                  str(output/'preparation.json'): hash_file(output/'preparation.json')}
        for name, digest in prep['artifact_hashes'].items():
            inputs[str(output/name)] = digest
        runtimes = [read_json(output/'members'/'G0-global'/f'seed{s}'/'prediction_runtime.json') for s in range(5)]
        provenance = {'experiment_id': config['experiment_id'], 'seeds': list(range(5)),
            'execution_code_commit': code, 'config_sha256': hash_file(config_path),
            'registered_scoring': SCORING, 'coverage': prep['coverage'],
            'new_fits': 0, 'reused_fits': 5, 'prediction_runtimes': runtimes,
            'delivery_tier_counts': {str(s): row['delivery_tier_counts'] for s,row in enumerate(runtimes)}}
        return score_artifacts(output, output/'baseline_predictions.npz', output/'dev_metadata.parquet',
            output/'parent_c1_analysis_results.json',
            Path(prep['parents']['c1']['run'])/'analysis'/'predictions.npz', output/'analysis',
            provenance=provenance, extra_inputs=inputs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'local-config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    score_registered(a.config.resolve(), a.local_config.resolve(), a.output.resolve())
