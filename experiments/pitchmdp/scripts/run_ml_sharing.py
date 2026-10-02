"""Registered TRAIN-only pitcher-sharing family on a frozen representative panel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
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
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_panel import select_panel, evaluation_metadata
from pitchmdp.matrix_sharing import (fit_pitcher_clusters, SharingContext, SharingPredictor,
                                   ContinuousPitcherContext, training_arrays)
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_benchmark import predict_streamed
from run_ml_benchmark import (SOURCES as ARCH_SOURCES, read_json, dump, regular_frame,
                              validate_native_runtime, load_data as load_arch_data)
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_sequence_pilot import arrays
from run_sequence_frequency_baselines import fit_temperature, temperature_predictions

CELLS = ('G0-global', 'G1-personal', 'G2-feature', 'G3-cluster', 'G4-partial')
SEEDS = (0, 1, 2)
SOURCES = list(dict.fromkeys([*ARCH_SOURCES, 'pitchmdp/matrix_panel.py',
    'pitchmdp/matrix_sharing.py', 'scripts/run_ml_sharing.py']))


def config_check(config):
    if (config['protocol'] != 'ml_sharing_v1' or config['seeds'] != list(SEEDS)
        or config['k'] != 4 or config['panel_per_cell'] != 4 or config['draws'] != 400
        or config['minimum_personal_train'] != 500 or config['minimum_personal_earlystop'] != 20
        or config['individual_tau'] != 1000 or config['cluster_tau'] != 10000):
        raise ValueError('Sharing settings differ from registered protocol')
    return config


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    from run_ml_benchmark import identity as arch_identity
    return {**arch_identity(config, local_path), 'source_hashes': source_hashes()}


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('Sharing source/config/environment changed')
    assert_hashes(output, prep['artifact_hashes'])
    for path, digest in prep['external_hashes'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('Parent artifact changed: ' + path)
    return prep


def save_keys(output, parts):
    records = {}
    for name, part in parts.items():
        path = name + '_keys.parquet'
        part[KEY].to_parquet(output / path, index=False)
        records[name] = {'path': path, 'n': len(part), 'games': int(part.game_pk.nunique()),
                         'rows_sha256': ordered_key_hash(part)}
    return records


def prepare(config, local, output, expected):
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        print('SHARING_PREPARED', output, flush=True)
        return
    parent = Path(config['parent_run']).resolve()
    check_location(local, parent)
    if output == parent or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError('Sharing output must be a distinct sibling')
    if hash_file(parent / 'preparation.json') != config['parent_preparation_sha256']:
        raise ValueError('Architecture preparation changed')
    arch = read_json(parent / 'preparation.json')
    assert_hashes(parent, arch['artifact_hashes'])
    for rel, digest in arch['identity']['source_hashes'].items():
        if hash_file(PROJECT / rel) != digest:
            raise ValueError('Frozen architecture implementation changed')
    external = {str(parent / 'preparation.json'): hash_file(parent / 'preparation.json')}
    started = time.perf_counter()
    store, inherited, aux = load_arch_data(local, parent, arch)
    frame, train = store.frame, inherited['train']
    panel = select_panel(train, starter_source=frame.loc[frame.split.eq('train')],
                         c6_ids=read_json(parent / 'parent_preparation.json')['scope']['regular']['cohort_ids'])
    clusters = fit_pitcher_clusters(train)
    ok = eligible(frame)
    selected = frame.pitcher.isin(panel['pitcher_ids'])
    parts = {name: inherited[name] for name in ('train', 'earlystop')}
    for name in ('temperature', 'blend', 'dev'):
        parts[name] = frame.loc[frame.split.eq(name) & selected & ok]
        if not len(parts[name]):
            raise ValueError('Empty panel split; no DEV-based replacement permitted')
    parts['mlb_dev'] = frame.loc[frame.split.eq('dev') & ok]
    units = {name: {'mode': name, 'train_n': len(train), 'earlystop_n': len(parts['earlystop'])}
             for name in ('global', 'feature')}
    eligibility = []
    train_c = train.pitcher.map(lambda pid: clusters['pitcher_cluster'][str(int(pid))])
    early_c = parts['earlystop'].pitcher.map(lambda pid: clusters['pitcher_cluster'].get(str(int(pid)), -1))
    for c in range(4):
        n, e = int(train_c.eq(c).sum()), int(early_c.eq(c).sum())
        if n >= 500 and e >= 20:
            units[f'cluster{c}'] = {'mode': 'cluster', 'id': c, 'train_n': n, 'earlystop_n': e}
    for pid in panel['pitcher_ids']:
        n, e = int(train.pitcher.eq(pid).sum()), int(parts['earlystop'].pitcher.eq(pid).sum())
        fit = n >= 500 and e >= 20
        eligibility.append({'pitcher': pid, 'train_n': n, 'earlystop_n': e, 'fit': fit,
                            'fallback_reason': None if fit else 'insufficient TRAIN or early-stop sample'})
        if fit:
            units[f'personal{pid}'] = {'mode': 'personal', 'id': pid, 'train_n': n, 'earlystop_n': e}
    output.mkdir(parents=True)
    for rel in SOURCES:
        path = output / 'source' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)
    for src, dst in [('preparation.json', 'architecture_preparation.json'),
                     ('parent_preparation.json', 'parent_preparation.json'), ('aux.pkl', 'aux.pkl')]:
        shutil.copyfile(parent / src, output / dst)
    dump(output / 'registered_config.json', config)
    dump(output / 'panel.json', panel)
    dump(output / 'clusters.json', clusters)
    records = save_keys(output, parts)
    coverage = {}
    for split in ('temperature', 'blend', 'dev'):
        requested = frame.loc[frame.split.eq(split) & selected]
        coverage[split] = {'requested_pitches': len(requested), 'eligible_pitches': len(parts[split]),
            'unsupported_pa': int((~requested.supported_pa.fillna(False)).sum()),
            'unmapped_outcome': int((outcome_labels(requested) < 0).sum()),
            'missing_type': int(requested.pitch_type.isna().sum()),
            'missing_coordinates': int(requested[['plate_x', 'plate_z']].isna().any(axis=1).sum()),
            'note': 'Reason counts overlap; retrospective eligible denominator is not pre-pitch availability.'}
    for name in ('blend', 'dev', 'mlb_dev'):
        evaluation_metadata(parts[name], panel).to_parquet(output / (name + '_metadata.parquet'), index=False)
    raw_dev = frame.loc[frame.split.eq('dev')]
    coverage['mlb_dev'] = {'requested_pitches': len(raw_dev), 'eligible_pitches': len(parts['mlb_dev'])}
    baseline = aux['baseline']
    chosen = fit_temperature(baseline.predict(parts['temperature']), outcome_labels(parts['temperature']))
    saved = {}
    for name in ('blend', 'dev', 'mlb_dev'):
        part = parts[name]
        raw = baseline.predict(part)
        saved[name + '_raw'] = raw
        saved[name] = temperature_predictions(raw, chosen['temperature'])
        for label, values in [('keys', part[KEY].to_numpy(np.int64)), ('y', outcome_labels(part)),
                               ('game_pk', part.game_pk.to_numpy(np.int64)), ('pitcher', part.pitcher.to_numpy(np.int64))]:
            saved[name + '_' + label] = values
    np.savez_compressed(output / 'baseline_predictions.npz', **saved)
    files = [str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()]
    prep = {'identity': expected, 'external_hashes': external, 'samples': records,
        'features': {**arch['features'], 'context': ContinuousPitcherContext(aux['context'], clusters).report(),
                     'parent_context': arch['features']['context']},
        'units': units, 'individual_eligibility': eligibility,
        'panel': panel, 'clusters': clusters, 'baseline_temperature': chosen, 'coverage': coverage,
        'artifact_hashes': artifact_hashes(output, files), 'seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False}
    if source_hashes() != expected['source_hashes']:
        raise ValueError('Sources changed during sharing prepare')
    dump(output / 'preparation.json', prep)
    print('SHARING_PREPARED', len(panel['pitcher_ids']), 'players', len(units), 'units per seed', flush=True)


def load_data(local, output, prep):
    from pitchmdp.matrix_features import MatrixHistoryStore
    from pitchmdp.matrix_benchmark import select_keys
    parent = read_json(output / 'parent_preparation.json')
    frame = regular_frame(local, parent)
    with (output / 'aux.pkl').open('rb') as stream:
        aux = pickle.load(stream)
    tokens = prep['features']['tokens']
    store = MatrixHistoryStore.from_frame(frame, normalizer=aux['normalizer'], history_length=5,
                                        type_vocabulary=tokens['type_vocabulary'])
    context = SharingContext(aux['context'], prep['clusters'])
    parts = {name: select_keys(frame, pd.read_parquet(output / record['path']), record)
             for name, record in prep['samples'].items()}
    return store, context, parts, aux


def profile(config, local, output, prep):
    dest = output / 'profile'
    if dest.exists():
        state = read_json(dest / 'state.json')
        if state['preparation_sha256'] != hash_file(output / 'preparation.json'):
            raise ValueError('Profile parent changed')
        assert_hashes(dest, state['artifact_hashes'])
        print('SHARING_PROFILE_COMPLETE', flush=True)
        return
    start = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep)
    train, early = parts['train'].iloc[:8192], parts['earlystop'].iloc[:2048]
    ta = arrays(store, context, train.index.to_numpy())
    ea = arrays(store, context, early.index.to_numpy())
    times, models = {}, {}
    for name, enriched in [('global', False), ('feature', True)]:
        before = time.perf_counter()
        model = MatrixModel('flatten_mlp', seed=0, width=128).fit(training_arrays(ta, enriched), outcome_labels(train),
            training_arrays(ea, enriched), outcome_labels(early), epochs=2, patience=2, batch_size=1024, learning_rate=.0005)
        times[name] = time.perf_counter() - before
        models[name] = model
    query = parts['temperature'].iloc[:64]
    wrapped = SharingPredictor('G2-feature', models['global'], prep['clusters'], feature_model=models['feature'])
    before = time.perf_counter()
    p, _, _ = predict_streamed(wrapped, aux['delivery'], store, context, query.index.to_numpy())
    inference = time.perf_counter() - before
    projected = max(times.values()) * sum(u['train_n'] for u in prep['units'].values()) / len(train) * 15
    result = {'train_rows': len(train), 'earlystop_rows': len(early), 'query_rows': len(query),
        'epochs': 2, 'fit_seconds': times, 'inference_seconds': inference,
        'rough_per_seed_fit_batch_projection_seconds': projected,
        'resource_gate': projected < 7200,
        'projection_note': 'Conservative linear 30-epoch extrapolation, not a guaranteed bound; caller also enforces timeout.',
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'maximum_mass_error': float(np.abs(p.sum(1) - 1).max()),
        'seconds_total': time.perf_counter() - start, 'dev_scores_read': False}
    dest.mkdir()
    dump(dest / 'profile.json', result)
    dump(dest / 'state.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
         'artifact_hashes': artifact_hashes(dest, ['profile.json'])})
    print('SHARING_PROFILE_COMPLETE', result, flush=True)


def fit(config, local, output, prep, seed):
    profile_state = read_json(output / 'profile' / 'state.json')
    if profile_state['preparation_sha256'] != hash_file(output / 'preparation.json'):
        raise ValueError('Matching sharing resource profile required')
    assert_hashes(output / 'profile', profile_state['artifact_hashes'])
    if read_json(output / 'profile' / 'profile.json')['resource_gate'] is not True:
        raise ValueError('Sharing fit resource gate requires revised execution plan')
    store, context, parts, aux = load_data(local, output, prep)
    train, early = parts['train'], parts['earlystop']
    mapping = context.mapping
    tc, ec = train.pitcher.map(mapping).fillna(-1), early.pitcher.map(mapping).fillna(-1)
    for unit, spec in prep['units'].items():
        destination = output / 'fits' / f'seed{seed}' / unit
        expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'unit': unit, 'spec': spec}
        if (destination / 'state.json').exists():
            state = read_json(destination / 'state.json')
            if state['identity'] != expected:
                raise ValueError('Fitted sharing unit changed')
            assert_hashes(destination, state['artifact_hashes'])
            continue
        if destination.exists() and any(destination.iterdir()):
            raise ValueError('Incomplete unit requires failure review')
        destination.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        tr, er = train, early
        if spec['mode'] == 'cluster':
            tr, er = train.loc[tc.eq(spec['id'])], early.loc[ec.eq(spec['id'])]
        elif spec['mode'] == 'personal':
            tr, er = train.loc[train.pitcher.eq(spec['id'])], early.loc[early.pitcher.eq(spec['id'])]
        enrich = spec['mode'] in ('feature', 'cluster')
        ta = training_arrays(arrays(store, context, tr.index.to_numpy()), enrich)
        ea = training_arrays(arrays(store, context, er.index.to_numpy()), enrich)
        model = MatrixModel('flatten_mlp', seed=seed, width=128).fit(ta, outcome_labels(tr), ea, outcome_labels(er),
                epochs=30, patience=5, batch_size=1024, learning_rate=.0005)
        model.save(destination / 'model.pt')
        dump(destination / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter() - start,
            'train_rows_sha256': ordered_key_hash(tr), 'earlystop_rows_sha256': ordered_key_hash(er),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if source_hashes() != prep['identity']['source_hashes']:
            raise ValueError('Sources changed during sharing fit')
        dump(destination / 'state.json', {'identity': expected,
             'artifact_hashes': artifact_hashes(destination, ['model.pt', 'fit.json'])})
        print('SHARING_FIT_COMPLETE', seed, unit, flush=True)
        del ta, ea, model


def predictor(config, output, prep, cell, seed):
    global_model = None
    personal, clusters, feature = {}, {}, None
    needed = {'G0-global': [], 'G1-personal': ['personal'], 'G2-feature': ['feature'],
              'G3-cluster': ['cluster'], 'G4-partial': ['personal', 'cluster']}[cell]
    needed = [*needed, 'global']
    dependencies = {}
    for name, spec in prep['units'].items():
        if spec['mode'] not in needed:
            continue
        dest = output / 'fits' / f'seed{seed}' / name
        state = read_json(dest / 'state.json')
        if state['identity'] != {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'unit': name, 'spec': spec}:
            raise ValueError('Sharing fit identity differs')
        assert_hashes(dest, state['artifact_hashes'])
        dependencies[str(dest / 'model.pt')] = hash_file(dest / 'model.pt')
        model = MatrixModel.load(dest / 'model.pt')
        if spec['mode'] == 'global': global_model = model
        elif spec['mode'] == 'personal': personal[spec['id']] = model
        elif spec['mode'] == 'cluster': clusters[spec['id']] = model
        else: feature = model
    return SharingPredictor(cell, global_model, prep['clusters'], feature_model=feature,
        personal_models=personal, cluster_models=clusters, individual_tau=config['individual_tau'],
        cluster_tau=config['cluster_tau']), dependencies


def predict(config, local, output, prep, cell, seed, mlb=False):
    dest = output / 'members' / cell / f'seed{seed}'
    name = 'mlb_prediction_state.json' if mlb else 'prediction_state.json'
    expected = {'preparation_sha256': canonical_hash(prep), 'cell': cell, 'seed': seed}
    if (dest / name).exists():
        state = read_json(dest / name)
        if state['identity'] != expected:
            raise ValueError('Sharing prediction identity changed')
        assert_hashes(dest, state['artifact_hashes'])
        for path, digest in state['dependencies'].items():
            if hash_file(Path(path)) != digest:
                raise ValueError('Sharing predictor dependency changed')
        print('SHARING_PREDICT_COMPLETE', cell, seed, flush=True)
        return
    start = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep)
    model, dependencies = predictor(config, output, prep, cell, seed)
    dest.mkdir(parents=True, exist_ok=True)
    if mlb:
        # Freeze the exact panel-selected calibration for whole-MLB confirmation.
        panel_state = read_json(dest / 'prediction_state.json')
        if panel_state['identity'] != expected or panel_state['dependencies'] != dependencies:
            raise ValueError('MLB prediction differs from frozen panel model')
        assert_hashes(dest, panel_state['artifact_hashes'])
        model.delivery_temperature = read_json(dest / 'calibration.json')['delivery_temperature']
    else:
        temp = parts['temperature']
        aux['delivery'].calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
        dump(dest / 'calibration.json', model.report)
    values, counts = {}, {}
    for split in (('mlb_dev',) if mlb else ('blend', 'dev')):
        part = parts[split]
        p, raw, levels = predict_streamed(model, aux['delivery'], store, context, part.index.to_numpy())
        values.update({split: p, split + '_raw': raw, split + '_delivery_level': levels,
            split + '_keys': part[KEY].to_numpy(np.int64), split + '_y': outcome_labels(part),
            split + '_game_pk': part.game_pk.to_numpy(np.int64), split + '_pitcher': part.pitcher.to_numpy(np.int64)})
        unique, n = np.unique(levels, return_counts=True)
        counts[split] = {str(int(k)): int(v) for k, v in zip(unique, n)}
    prefix = 'mlb_' if mlb else ''
    archive_name = prefix + 'predictions.npz'
    runtime_name = prefix + 'prediction_runtime.json'
    if (dest / archive_name).exists():
        raise ValueError('Incomplete prediction archive requires failure review')
    np.savez_compressed(dest / archive_name, **values)
    dump(dest / runtime_name, {'seconds': time.perf_counter() - start,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'delivery_tier_counts': counts})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during sharing prediction')
    dump(dest / name, {'identity': expected, 'dependencies': dependencies,
        'artifact_hashes': artifact_hashes(dest, [archive_name, runtime_name, 'calibration.json'])})
    print('SHARING_PREDICT_COMPLETE', cell, seed, 'mlb' if mlb else 'panel', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('profile')
    f = sub.add_parser('fit'); f.add_argument('--seed', type=int, choices=SEEDS, required=True)
    q = sub.add_parser('predict'); q.add_argument('--seed', type=int, choices=SEEDS, required=True)
    q.add_argument('--cell', choices=CELLS, required=True); q.add_argument('--mlb', action='store_true')
    args = p.parse_args()
    config, local = config_check(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, args.local_config)
    with heavy_lock(root):
        if args.command == 'prepare': prepare(config, local, output, expected)
        else:
            prep = verify(output, expected)
            if args.command == 'profile': profile(config, local, output, prep)
            elif args.command == 'fit': fit(config, local, output, prep, args.seed)
            else: predict(config, local, output, prep, args.cell, args.seed, args.mlb)


if __name__ == '__main__':
    main()
