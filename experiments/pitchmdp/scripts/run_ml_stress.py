"""Immutable Cpanel stress predictions for a previously selected G comparison."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import time
import resource

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_stress import protocol, SCENARIOS, StressHistoryStore, predict_stress_streamed
from run_ml_sharing import (SOURCES as SHARING_SOURCES, identity as sharing_identity,
                            verify as verify_sharing, load_data, predictor, SEEDS)
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from score_ml_matrix import archive

SOURCES = [*SHARING_SOURCES, 'pitchmdp/matrix_stress.py', 'scripts/run_ml_stress.py']
CONTROLS = {'G1-personal': 'G0-global', 'G2-feature': 'G0-global',
            'G3-cluster': 'G2-feature', 'G4-partial': 'G2-feature'}


def exposure_report(store, rows, scenario):
    """Count query-history occurrences once, independently of delivery draws."""
    wrapper = StressHistoryStore(store, scenario)
    counts = {'present_occurrences': 0, 'dropped_occurrences': 0,
              'changed_occurrences': 0, 'query_rows': len(rows)}
    present_keys, changed_keys = set(), set()
    for start in range(0, len(rows), 512):
        selected = np.asarray(rows[start:start + 512], dtype=np.int64)
        current = np.zeros((len(selected), 8), dtype=np.float32)
        original, valid = store.gather(selected, current)
        perturbed, kept = wrapper.gather(selected, current)
        present = valid[:, :-1]
        dropped = present & ~kept[:, :-1]
        changed = present & ((original[:, :-1] != perturbed[:, :-1]).any(2) | dropped)
        history = store.indices[selected]
        counts['present_occurrences'] += int(present.sum())
        counts['dropped_occurrences'] += int(dropped.sum())
        counts['changed_occurrences'] += int(changed.sum())
        present_keys.update(history[present].tolist())
        changed_keys.update(history[changed].tolist())
    counts.update(unique_present_pitches=len(present_keys), unique_changed_pitches=len(changed_keys))
    denominator = counts['present_occurrences']
    counts['dropped_fraction'] = counts['dropped_occurrences'] / denominator if denominator else None
    counts['changed_fraction'] = counts['changed_occurrences'] / denominator if denominator else None
    counts['context_outage'] = scenario if scenario.startswith('unknown_') and scenario != 'unknown_type_outcome' else None
    return counts


def config_check(config):
    if (config['protocol'] != 'ml_stress_v1' or config['seeds'] != list(SEEDS)
        or config['stress_protocol'] != protocol() or config['scope'] != 'Cpanel'
        or CONTROLS.get(config['candidate']) != config['control']):
        raise ValueError('Stress comparison differs from registered protocol')
    for key in ('parent_preparation_sha256', 'parent_analysis_sha256'):
        if len(config[key]) != 64:
            raise ValueError('Frozen parent artifacts required')
    return config


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    return {**sharing_identity(config, local_path), 'source_hashes': source_hashes()}


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('Stress config/source/environment changed')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Frozen stress dependency changed')
    return prep


def prepare(config, local_path, local, output, expected):
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        print('STRESS_PREPARED', flush=True)
        return
    parent = Path(config['parent_run']).resolve()
    check_location(local, parent)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Stress run must be a sibling of sharing run')
    parent_config = read_json(parent / 'registered_config.json')
    shared = verify_sharing(parent, sharing_identity(parent_config, local_path))
    analysis = parent / 'analysis' / 'panel'
    if (hash_file(parent / 'preparation.json') != config['parent_preparation_sha256'] or
        hash_file(analysis / 'results.json') != config['parent_analysis_sha256']):
        raise ValueError('Stress selection/preparation hashes differ')
    parent_result = read_json(analysis / 'results.json')
    ranked = parent_result['followup_candidates']
    selected_status = 'screen_promoted' if ranked else 'diagnostic_only_not_promoted'
    if not ranked:
        ranked = sorted(CONTROLS, key=lambda c: (parent_result['reports'][c]['primary']['log_loss'],
                               parent_result['logical_cell_costs'][c]['total_fit_seconds']))
    if config['candidate'] != ranked[0] or config['selection_status'] != selected_status:
        raise ValueError('Stress comparison differs from registered common selection rule')
    manifest = read_json(analysis / 'manifest.json')
    if manifest['results_sha256'] != config['parent_analysis_sha256'] or hash_file(analysis / 'predictions.npz') != manifest['predictions_sha256']:
        raise ValueError('Parent analysis integrity failed')
    external = dict(manifest['inputs'])
    external[str(analysis / 'results.json')] = config['parent_analysis_sha256']
    external[str(analysis / 'manifest.json')] = hash_file(analysis / 'manifest.json')
    external[str(analysis / 'predictions.npz')] = manifest['predictions_sha256']
    for cell in (config['candidate'], config['control']):
        for seed in SEEDS:
            dest = parent / 'members' / cell / f'seed{seed}'
            state = read_json(dest / 'prediction_state.json')
            assert_hashes(dest, state['artifact_hashes'])
            for name, digest in state['artifact_hashes'].items():
                external[str(dest / name)] = digest
            external.update(state['dependencies'])
    for name, digest in external.items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Parent input changed before stress preparation')
    output.mkdir(parents=True)
    for name in SOURCES:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, destination)
    for source, target in [(parent / 'preparation.json', 'parent_preparation.json'),
        (parent / 'registered_config.json', 'parent_config.json'),
        (parent / 'dev_metadata.parquet', 'dev_metadata.parquet'),
        (parent / 'baseline_predictions.npz', 'baseline_predictions.npz'),
        (analysis / 'results.json', 'parent_analysis.json')]:
        shutil.copyfile(source, output / target)
    dump(output / 'registered_config.json', config)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    dump(output / 'preparation.json', {'identity': expected, 'parent_run': str(parent),
        'external_hashes': external, 'artifact_hashes': artifact_hashes(output, files),
        'samples': {'dev': shared['samples']['dev']}, 'coverage': shared['coverage'],
        'candidate': config['candidate'], 'control': config['control'], 'seeds': list(SEEDS),
        'selection_status': selected_status,
        'stress_protocol': protocol(), 'stress_scores_read': False,
        'frequency_component': 'Frozen league/type/count/outs/base baseline has no perturbed historical/player-profile features; unchanged by these scenarios.'})
    print('STRESS_PREPARED', flush=True)


def predict(config, local, output, prep, cell, seed, scenario):
    if cell not in (config['candidate'], config['control']) or seed not in SEEDS or scenario not in SCENARIOS:
        raise ValueError('Unregistered stress member')
    dest = output / 'members' / scenario / cell / f'seed{seed}'
    expected = {'preparation_sha256': hash_file(output / 'preparation.json'), 'cell': cell, 'seed': seed, 'scenario': scenario}
    if (dest / 'prediction_state.json').exists():
        state = read_json(dest / 'prediction_state.json')
        if state['identity'] != expected:
            raise ValueError('Stress member identity differs')
        assert_hashes(dest, state['artifact_hashes'])
        print('STRESS_PREDICT_COMPLETE', cell, seed, scenario, flush=True)
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Incomplete stress member requires failure review')
    start = time.perf_counter()
    parent = Path(prep['parent_run'])
    member = parent / 'members' / cell / f'seed{seed}'
    archived = archive(member / 'predictions.npz')
    parent_config = read_json(output / 'parent_config.json')
    parent_prep = read_json(output / 'parent_preparation.json')
    store, context, parts, aux = load_data(local, parent, parent_prep)
    # All frequency keys are unaffected by the registered stress scenarios.
    if aux['baseline'].include_pitcher or aux['baseline'].parent_type_model.include_pitcher:
        raise ValueError('Unexpected pitcher-aware frequency baseline under outage scenario')
    model, _ = predictor(parent_config, parent, parent_prep, cell, seed)
    model.delivery_temperature = read_json(member / 'calibration.json')['delivery_temperature']
    part = parts['dev']
    if not np.array_equal(part[KEY].to_numpy(np.int64), archived['dev_keys']):
        raise ValueError('Stress query keys differ from clean parent')
    calibrated, raw, levels = predict_stress_streamed(model, aux['delivery'], store, context,
                                                    part.index.to_numpy(), scenario)
    values = {name: value for name, value in archived.items() if name.startswith('dev_')}
    values.update(dev=calibrated, dev_raw=raw, dev_delivery_level=levels)
    if not np.array_equal(levels, archived['dev_delivery_level']):
        raise ValueError('Stress must preserve delivery support and tiers')
    clean_error = None
    if scenario == 'clean':
        clean_error = max(float(np.abs(values[name] - archived[name]).max()) for name in ('dev', 'dev_raw'))
        if clean_error > 1e-6:
            raise ValueError('Fresh clean inference differs from parent beyond registered tolerance')
    exposure = exposure_report(store, part.index.to_numpy(), scenario)
    dest.mkdir(parents=True)
    np.savez_compressed(dest / 'predictions.npz', **values)
    dump(dest / 'runtime.json', {'seconds': time.perf_counter() - start,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'clean_archive_reused': False, 'clean_maximum_probability_error': clean_error,
        'clean_equivalence_tolerance': 1e-6, 'exposure': exposure,
        'no_recalibration': True, 'source_member': str(member)})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Stress sources changed during inference')
    dump(dest / 'prediction_state.json', {'identity': expected,
        'artifact_hashes': artifact_hashes(dest, ['predictions.npz', 'runtime.json'])})
    print('STRESS_PREDICT_COMPLETE', cell, seed, scenario, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    q = sub.add_parser('predict')
    q.add_argument('--cell', choices=list(set(CONTROLS) | set(CONTROLS.values())), required=True)
    q.add_argument('--seed', type=int, choices=SEEDS, required=True)
    q.add_argument('--scenario', choices=SCENARIOS, required=True)
    a = p.parse_args()
    config, local = config_check(read_json(a.config)), read_json(a.local_config)
    output = a.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, a.local_config)
    with heavy_lock(root):
        if a.command == 'prepare': prepare(config, a.local_config, local, output, expected)
        else: predict(config, local, output, verify(output, expected), a.cell, a.seed, a.scenario)


if __name__ == '__main__':
    main()
