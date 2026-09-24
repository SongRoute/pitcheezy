"""Additive five-seed C1 extension for a registered subset of frozen G cells.

Seeds 0–2 are read-only parent references. This runner fits only the union of
needed G units at seeds 3–4 and recalibrates those new members on the same May
panel. Scoring is a separate complete-family operation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_confirmation import (BUDGET, CELLS, MODES, NEW_SEEDS, SEEDS,
    member_identity, required_units, unit_identity, validate_config)
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import SharingPredictor, training_arrays
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.model import outcome_labels
from run_ml_benchmark import (identity as architecture_identity, read_json,
    validate_native_runtime)
from run_ml_matrix import artifact_hashes, assert_hashes, check_location, heavy_lock
from run_ml_sharing import (SOURCES as G_SOURCES, load_data as load_g_data,
    config_check as g_config_check, source_hashes as g_source_hashes)
from run_sequence_pilot import arrays, dump
from score_ml_matrix import archive, assert_aligned


SOURCES = list(dict.fromkeys([*G_SOURCES, 'pitchmdp/matrix_confirmation.py',
                              'pitchmdp/matrix_confirmation_metrics.py',
                              'pitchmdp/matrix_group_metrics.py',
                              'pitchmdp/matrix_metrics.py',
                              'scripts/score_ml_matrix.py',
                              'scripts/run_ml_confirmation.py']))


def source_hashes():
    return {rel: hash_file(PROJECT / rel) for rel in SOURCES}


def identity(config, local_path):
    return {**architecture_identity(config, local_path), 'source_hashes': source_hashes()}


def resolve_member(config, output, prep, cell, seed):
    """Return immutable state/archive locations for an ordered five-seed cell."""
    if cell not in prep['cells'] or seed not in SEEDS:
        raise ValueError('Unregistered C1 member')
    reused = seed not in NEW_SEEDS
    root = Path(prep['parent_run']) if reused else output
    folder = root / 'members' / cell / f'seed{seed}'
    state_path = folder / 'prediction_state.json'
    state = read_json(state_path)
    expected = ({'preparation_sha256': canonical_hash(read_json(root / 'preparation.json')),
                 'cell': cell, 'seed': seed} if reused else member_identity(prep, cell, seed))
    if state['identity'] != expected:
        raise ValueError('C1 member prediction identity differs')
    assert_hashes(folder, state['artifact_hashes'])
    for path, digest in state['dependencies'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('C1 member fit dependency changed: ' + path)
    result = {'directory': str(folder), 'state_path': str(state_path),
              'predictions_path': str(folder / 'predictions.npz'),
              'source_run': str(root), 'reused': reused,
              'state_sha256': hash_file(state_path),
              'predictions_sha256': hash_file(folder / 'predictions.npz')}
    return result


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('C1 config, source or native environment changed')
    assert_hashes(output, prep['artifact_hashes'])
    for path, digest in prep['external_hashes'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('C1 parent artifact changed: ' + path)
    return prep


def _parent(config, local, output):
    root = Path(local['artifact_root']).resolve() / 'runs' / 'ML-MATRIX-20260924'
    parent = Path(config['parent_run']).resolve()
    if (not parent.is_relative_to(root) or parent == output or
            parent.is_relative_to(output) or output.is_relative_to(parent)):
        raise ValueError('C1 parent must be a distinct protocol sibling')
    prep_path = parent / 'preparation.json'
    analysis = parent / 'analysis' / 'panel'
    manifest_path = analysis / 'manifest.json'
    if (hash_file(prep_path) != config['parent_preparation_sha256'] or
            hash_file(manifest_path) != config['parent_analysis_sha256']):
        raise ValueError('Frozen G preparation or complete panel analysis SHA differs')
    prep = read_json(prep_path)
    assert_hashes(parent, prep['artifact_hashes'])
    for rel, digest in prep['identity']['source_hashes'].items():
        if hash_file(PROJECT / rel) != digest:
            raise ValueError('Frozen G implementation changed: ' + rel)
    registered = g_config_check(read_json(parent / 'registered_config.json'))
    if prep['identity']['config_sha256'] != canonical_hash(registered):
        raise ValueError('G registered config differs from preparation')
    for name in ('draws', 'individual_tau', 'cluster_tau'):
        if registered[name] != config[name]:
            raise ValueError('C1 setting differs from G parent: ' + name)
    training = registered['registration']['training']
    if (training['kind'] != config['kind'] or training['width'] != config['width'] or
            {k: training[k] for k in BUDGET} != config['budget']):
        raise ValueError('C1 neural training differs from G parent')
    manifest = read_json(manifest_path)
    for name, suffix in (('results', '.json'), ('predictions', '.npz')):
        if hash_file(analysis / (name + suffix)) != manifest[name + '_sha256']:
            raise ValueError('G complete-family analysis changed')
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('G analysis input changed: ' + path)
    return parent, prep, registered, analysis


def _reuse(config, parent, parent_prep):
    """Validate all selected 0–2 calibrated members and common Cpanel keys."""
    common = archive(parent / 'baseline_predictions.npz')
    reused, external = {}, {}
    for cell in config['cells']:
        for seed in SEEDS[:3]:
            folder = parent / 'members' / cell / f'seed{seed}'
            state_path = folder / 'prediction_state.json'
            state = read_json(state_path)
            if state['identity'] != {'preparation_sha256': canonical_hash(parent_prep),
                                    'cell': cell, 'seed': seed}:
                raise ValueError('G reused member identity differs')
            assert_hashes(folder, state['artifact_hashes'])
            for path, digest in state['dependencies'].items():
                if hash_file(Path(path)) != digest:
                    raise ValueError('G reused fit dependency changed')
                external[path] = digest
                unit_folder = Path(path).parent
                fit_state_path = unit_folder / 'state.json'
                fit_state = read_json(fit_state_path)
                unit = unit_folder.name
                if (fit_state['identity'] != {'preparation_sha256': canonical_hash(parent_prep),
                        'seed': seed, 'unit': unit, 'spec': parent_prep['units'][unit]}):
                    raise ValueError('G reused unit training identity differs')
                assert_hashes(unit_folder, fit_state['artifact_hashes'])
                external[str(fit_state_path)] = hash_file(fit_state_path)
                for rel in fit_state['artifact_hashes']:
                    external[str(unit_folder / rel)] = hash_file(unit_folder / rel)
            member = archive(folder / 'predictions.npz')
            assert_aligned(member, common)
            files = [state_path, *[folder / name for name in state['artifact_hashes']]]
            external.update({str(path): hash_file(path) for path in files})
            reused[f'{cell}/seed{seed}'] = {'prediction_state_sha256': hash_file(state_path),
                'predictions_sha256': hash_file(folder / 'predictions.npz'),
                'calibration_sha256': hash_file(folder / 'calibration.json')}
    return reused, external


def prepare(config, local, output, expected):
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        print('CONFIRMATION_PREPARED', output, flush=True)
        return
    parent, gprep, gconfig, analysis = _parent(config, local, output)
    selected, external = _reuse(config, parent, gprep)
    units = required_units(gprep['units'], config['cells'])
    started = time.perf_counter()
    output.mkdir(parents=True)
    dump(output / 'registered_config.json', config)
    for rel in SOURCES:
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    # These copies make the exact common calibration and evaluation denominator
    # reviewable without rewriting any G member or auxiliary object.
    for name in ('preparation.json', 'registered_config.json', 'baseline_predictions.npz',
                 'aux.pkl', 'panel.json', 'clusters.json'):
        shutil.copyfile(parent / name, output / ('parent_' + name))
    for name in ('manifest.json', 'results.json'):
        shutil.copyfile(analysis / name, output / ('parent_analysis_' + name))
    external.update({str(parent / 'preparation.json'): hash_file(parent / 'preparation.json'),
                     str(analysis / 'manifest.json'): hash_file(analysis / 'manifest.json'),
                     str(analysis / 'results.json'): hash_file(analysis / 'results.json'),
                     str(analysis / 'predictions.npz'): hash_file(analysis / 'predictions.npz')})
    files = [str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()]
    report = {'identity': expected, 'parent_run': str(parent),
        'parent_preparation_sha256': hash_file(parent / 'preparation.json'),
        'parent_analysis_sha256': hash_file(analysis / 'manifest.json'),
        'cells': config['cells'], 'seeds': list(SEEDS), 'primary_comparisons': config['primary_comparisons'],
        'selection_status': config['selection_status'], 'units': units, 'reused_members': selected,
        'samples': gprep['samples'], 'features': gprep['features'], 'panel': gprep['panel'],
        'clusters': gprep['clusters'],
        'external_hashes': external, 'artifact_hashes': artifact_hashes(output, files),
        'new_unit_count_per_seed': len(units), 'new_fit_count': len(units) * len(NEW_SEEDS),
        'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': time.perf_counter()-started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'dev_scores_read': False}
    if source_hashes() != expected['source_hashes']:
        raise ValueError('C1 source changed during preparation')
    dump(output / 'preparation.json', report)
    print('CONFIRMATION_PREPARED', config['cells'], len(units), 'units/seed', flush=True)


def load_data(local, output, prep):
    parent = Path(prep['parent_run'])
    gprep = read_json(parent / 'preparation.json')
    return load_g_data(local, parent, gprep)


def _unit_rows(parts, context, spec):
    train, early = parts['train'], parts['earlystop']
    if spec['mode'] == 'cluster':
        tc = train.pitcher.map(context.mapping).fillna(-1)
        ec = early.pitcher.map(context.mapping).fillna(-1)
        train, early = train.loc[tc.eq(spec['id'])], early.loc[ec.eq(spec['id'])]
    elif spec['mode'] == 'personal':
        train, early = train.loc[train.pitcher.eq(spec['id'])], early.loc[early.pitcher.eq(spec['id'])]
    if (len(train), len(early)) != (spec['train_n'], spec['earlystop_n']):
        raise ValueError('C1 unit population differs from frozen G')
    return train, early


def profile(config, local, output, prep):
    destination = output / 'profile'
    if destination.exists():
        state = read_json(destination / 'state.json')
        if state['preparation_sha256'] != hash_file(output / 'preparation.json'):
            raise ValueError('C1 profile belongs to another preparation')
        assert_hashes(destination, state['artifact_hashes'])
        print('CONFIRMATION_PROFILE_COMPLETE', flush=True)
        return
    destination.mkdir(parents=True)
    started = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep)
    times, representatives = {}, {}
    for mode in sorted({spec['mode'] for spec in prep['units'].values()}):
        unit, spec = next((name, value) for name, value in prep['units'].items() if value['mode'] == mode)
        train, early = _unit_rows(parts, context, spec)
        train, early = train.iloc[:8192], early.iloc[:2048]
        before = time.perf_counter()
        model = MatrixModel('flatten_mlp', seed=3, width=128,
                            device=None if config['device'] == 'auto' else config['device'])
        enrich = mode in ('feature', 'cluster')
        model.fit(training_arrays(arrays(store, context, train.index.to_numpy()), enrich), outcome_labels(train),
                  training_arrays(arrays(store, context, early.index.to_numpy()), enrich), outcome_labels(early),
                  epochs=2, patience=2, batch_size=1024, learning_rate=.0005)
        representatives[mode] = model
        times[mode] = {'unit': unit, 'train_rows': len(train), 'earlystop_rows': len(early),
                       'two_epoch_seconds': time.perf_counter()-before}
    # May-only calibration and inference exercise the original 400-draw
    # physical integration with a newly seeded representative model.
    temp = parts['temperature'].iloc[:64]
    probe = SharingPredictor('G0-global', representatives['global'], prep['clusters'])
    before = time.perf_counter()
    aux['delivery'].calibrate(probe, store, context, temp.index.to_numpy(), outcome_labels(temp))
    calibration = time.perf_counter()-before
    before = time.perf_counter()
    p, _, _ = predict_streamed(probe, aux['delivery'], store, context, temp.index.to_numpy())
    inference = time.perf_counter()-before
    projection = sum(times[spec['mode']]['two_epoch_seconds'] * spec['train_n'] /
                     max(times[spec['mode']]['train_rows'], 1) * 15 for spec in prep['units'].values())
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('C1 source changed during profile')
    parent = Path(prep['parent_run'])
    parent_prediction = {cell: [read_json(parent / 'members' / cell / f'seed{seed}' /
                        'prediction_runtime.json')['seconds'] for seed in SEEDS[:3]]
                         for cell in prep['cells']}
    dump(destination / 'profile.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
        'per_mode': times, 'may_rows': len(temp), 'draws': config['draws'],
        'may_calibration_seconds': calibration, 'may_inference_seconds': inference,
        'parent_full_cpanel_prediction_seconds': parent_prediction,
        'maximum_mass_error': float(np.abs(p.sum(1)-1).max()),
        'rough_per_seed_fit_projection_seconds': projection,
        'two_seed_fit_projection_seconds': 2*projection,
        'seconds_total': time.perf_counter()-started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'dev_scores_read': False})
    dump(destination / 'state.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
         'artifact_hashes': artifact_hashes(destination, ['profile.json'])})
    print('CONFIRMATION_PROFILE_COMPLETE', flush=True)


def fit(config, local, output, prep, seed):
    if seed not in NEW_SEEDS:
        raise ValueError('Only seeds3/4 can be newly fitted')
    profile_state = read_json(output / 'profile' / 'state.json')
    if profile_state['preparation_sha256'] != hash_file(output / 'preparation.json'):
        raise ValueError('C1 matching profile required')
    assert_hashes(output / 'profile', profile_state['artifact_hashes'])
    store, context, parts, _ = load_data(local, output, prep)
    for unit, spec in prep['units'].items():
        folder = output / 'fits' / f'seed{seed}' / unit
        state_path = folder / 'state.json'
        expected = unit_identity(prep, unit, seed)
        if state_path.exists():
            state = read_json(state_path)
            if state['identity'] != expected:
                raise ValueError('C1 completed unit identity changed')
            assert_hashes(folder, state['artifact_hashes'])
            continue
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Incomplete C1 unit requires failure review')
        folder.mkdir(parents=True, exist_ok=True)
        before = time.perf_counter()
        train, early = _unit_rows(parts, context, spec)
        enrich = spec['mode'] in ('feature', 'cluster')
        model = MatrixModel(config['kind'], seed=seed, width=config['width'],
                            device=None if config['device'] == 'auto' else config['device'])
        model.fit(training_arrays(arrays(store, context, train.index.to_numpy()), enrich), outcome_labels(train),
                  training_arrays(arrays(store, context, early.index.to_numpy()), enrich), outcome_labels(early),
                  **config['budget'])
        model.save(folder / 'model.pt')
        dump(folder / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter()-before,
             'train_rows_sha256': ordered_key_hash(train), 'earlystop_rows_sha256': ordered_key_hash(early),
             'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if source_hashes() != prep['identity']['source_hashes']:
            raise ValueError('C1 source changed during fit')
        dump(state_path, {'identity': expected,
             'artifact_hashes': artifact_hashes(folder, ['model.pt', 'fit.json'])})
        print('CONFIRMATION_FIT_COMPLETE', seed, unit, flush=True)


def _predictor(config, output, prep, cell, seed):
    models = {'personal': {}, 'cluster': {}}
    dependencies = {}
    for unit, spec in prep['units'].items():
        if spec['mode'] not in MODES[cell]:
            continue
        folder = output / 'fits' / f'seed{seed}' / unit
        state = read_json(folder / 'state.json')
        if state['identity'] != unit_identity(prep, unit, seed):
            raise ValueError('C1 unit fit identity changed')
        assert_hashes(folder, state['artifact_hashes'])
        dependencies[str(folder / 'model.pt')] = hash_file(folder / 'model.pt')
        model = MatrixModel.load(folder / 'model.pt', device=None if config['device'] == 'auto' else config['device'])
        if spec['mode'] in ('personal', 'cluster'):
            models[spec['mode']][spec['id']] = model
        else:
            models[spec['mode']] = model
    return SharingPredictor(cell, models['global'], prep['clusters'],
        feature_model=models.get('feature'), personal_models=models['personal'],
        cluster_models=models['cluster'], individual_tau=config['individual_tau'],
        cluster_tau=config['cluster_tau']), dependencies


def predict(config, local, output, prep, cell, seed):
    if cell not in prep['cells'] or seed not in NEW_SEEDS:
        raise ValueError('Only registered cells seeds3/4 are new predictions')
    folder = output / 'members' / cell / f'seed{seed}'
    state_path = folder / 'prediction_state.json'
    expected = member_identity(prep, cell, seed)
    if state_path.exists():
        state = read_json(state_path)
        if state['identity'] != expected:
            raise ValueError('C1 prediction identity changed')
        assert_hashes(folder, state['artifact_hashes'])
        for path, digest in state['dependencies'].items():
            if hash_file(Path(path)) != digest:
                raise ValueError('C1 prediction fit dependency changed')
        print('CONFIRMATION_PREDICT_COMPLETE', cell, seed, flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Incomplete C1 member requires failure review')
    started = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep)
    model, dependencies = _predictor(config, output, prep, cell, seed)
    folder.mkdir(parents=True, exist_ok=True)
    temp = parts['temperature']
    aux['delivery'].calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
    dump(folder / 'calibration.json', model.report)
    values, tiers = {}, {}
    for split in ('blend', 'dev'):
        part = parts[split]
        calibrated, raw, levels = predict_streamed(model, aux['delivery'], store, context, part.index.to_numpy())
        values.update({split: calibrated, split+'_raw': raw, split+'_delivery_level': levels,
            split+'_keys': part[KEY].to_numpy(np.int64), split+'_y': outcome_labels(part),
            split+'_game_pk': part.game_pk.to_numpy(np.int64),
            split+'_pitcher': part.pitcher.to_numpy(np.int64)})
        unique, counts = np.unique(levels, return_counts=True)
        tiers[split] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    np.savez_compressed(folder / 'predictions.npz', **values)
    dump(folder / 'prediction_runtime.json', {'seconds': time.perf_counter()-started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'delivery_tier_counts': tiers, 'dev_scored': False})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('C1 source changed during prediction')
    dump(state_path, {'identity': expected, 'dependencies': dependencies,
        'artifact_hashes': artifact_hashes(folder, ['calibration.json', 'predictions.npz',
                                                   'prediction_runtime.json'])})
    print('CONFIRMATION_PREDICT_COMPLETE', cell, seed, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'profile', 'status'):
        commands.add_parser(name)
    fit_parser = commands.add_parser('fit')
    fit_parser.add_argument('--seed', type=int, choices=NEW_SEEDS, required=True)
    predict_parser = commands.add_parser('predict')
    predict_parser.add_argument('--cell', choices=CELLS, required=True)
    predict_parser.add_argument('--seed', type=int, choices=NEW_SEEDS, required=True)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    local = read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, args.local_config)
    if args.command == 'status':
        prep = verify(output, expected)
        for cell in prep['cells']:
            for seed in SEEDS:
                if seed not in NEW_SEEDS:
                    state = 'reused'
                else:
                    folder = output / 'members' / cell / f'seed{seed}'
                    state = 'predicted' if (folder / 'prediction_state.json').exists() else 'pending'
                print(cell, seed, state)
        return
    with heavy_lock(root):
        if args.command == 'prepare':
            prepare(config, local, output, expected)
        else:
            prep = verify(output, expected)
            if args.command == 'profile': profile(config, local, output, prep)
            elif args.command == 'fit': fit(config, local, output, prep, args.seed)
            else: predict(config, local, output, prep, args.cell, args.seed)


if __name__ == '__main__':
    main()
