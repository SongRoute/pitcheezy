"""Prepare, fit and archive T2 exclusion folds; never select or score DEV models.

Pitcher uses one common global fallback. Batter uses the frozen G comparison.
Each member archives paired Z/W/O predictions with ONE allowed-May temperature
and ONE set of June predictions for the later, common blend. Prefix-only pitcher
centroid routing is intentionally outside this first execution protocol.
"""
from __future__ import annotations

import argparse
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
from pitchmdp.matrix_data import canonical_hash, load_verified_processed_cache, ordered_key_hash
from pitchmdp.matrix_benchmark import select_keys, predict_streamed
from pitchmdp.matrix_generalization import (GeneralizationFold, REGIMES, select_heldout_pitchers,
    select_heldout_batters, audit_new_matchups)
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_panel import evaluation_metadata
from pitchmdp.matrix_sharing import SharingContext, SharingPredictor, fit_pitcher_clusters, training_arrays
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_data import PhysicalNormalizer
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceContext
from run_ml_benchmark import read_json, dump, identity as base_identity, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_sharing import SOURCES as SHARING_SOURCES, identity as sharing_identity, verify as verify_sharing
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline, fit_temperature, temperature_predictions
from run_sequence_context_frequency import ContextFrequencyBaseline
from run_sequence_pilot import arrays
from run_temporal_blend import assign_fold

SEEDS, AXES = (0, 1, 2), ('pitcher', 'batter')
BUDGET = {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}
CONTROLS = {'G1-personal': 'G0-global', 'G2-feature': 'G0-global',
            'G3-cluster': 'G2-feature', 'G4-partial': 'G2-feature'}
MODES = {'G0-global': {'global'}, 'G1-personal': {'global', 'personal'},
         'G2-feature': {'global', 'feature'}, 'G3-cluster': {'global', 'cluster'},
         'G4-partial': {'global', 'cluster', 'personal'}}
SOURCES = list(dict.fromkeys([*SHARING_SOURCES, 'pitchmdp/matrix_generalization.py',
                             'scripts/run_ml_generalization.py']))


def config_check(config):
    required = {'protocol', 'experiment_id', 'parent_run', 'parent_preparation_sha256', 'parent_analysis_sha256',
                'candidate', 'control', 'selection_basis', 'selection_status', 'scope', 'seeds', 'axes', 'regimes',
                'prefix_games', 'selector_seed', 'draws', 'width', 'budget', 'device', 'adaptation'}
    if not isinstance(config, dict) or set(config) - required - {'registration'} or not required <= set(config):
        raise ValueError('Invalid T2 configuration schema')
    if (config['protocol'] != 'ml_generalization_v1' or config['scope'] != 'Cpanel'
            or config['seeds'] != list(SEEDS) or config['axes'] != list(AXES)
            or config['regimes'] != list(REGIMES) or config['prefix_games'] != 2
            or config['selector_seed'] != 20260924 or config['draws'] != 400
            or config['width'] != 128 or config['budget'] != BUDGET
            or config['adaptation'] != 'fixed_prefix_context_only_v1'
            or CONTROLS.get(config['candidate']) != config['control']):
        raise ValueError('T2 axes or selected G comparison differ from registered protocol')
    for name in ('parent_preparation_sha256', 'parent_analysis_sha256'):
        value = config[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Exact frozen G preparation and analysis SHA256 required')
    if any(not isinstance(config[name], str) or not config[name].strip()
           for name in ('experiment_id', 'parent_run', 'selection_basis')):
        raise ValueError('Run identity and pre-T2 selection rationale required')
    if config['device'] not in ('auto', 'cpu', 'mps'):
        raise ValueError('Local device must be auto/cpu/mps')
    if config['selection_status'] not in ('screen_promoted', 'diagnostic_only_not_promoted'):
        raise ValueError('Explicit promoted versus diagnostic-only selection status required')
    if 'registration' in config and not isinstance(config['registration'], dict):
        raise ValueError('Registration must be an object')
    return config


def chosen_comparison(analysis):
    """Recheck the common T2/T3/T4 rule using frozen Cpanel results only."""
    comparisons = analysis['comparisons']
    if [(c['candidate'], c['control']) for c in comparisons] != list(CONTROLS.items()):
        raise ValueError('Frozen G analysis must contain its exact four comparisons')
    ranks = {}
    for cell in CONTROLS:
        nll = analysis['reports'][cell]['primary']['log_loss']
        seconds = analysis['logical_cell_costs'][cell]['total_fit_seconds']
        if not np.isfinite(nll) or not np.isfinite(seconds) or nll < 0 or seconds < 0:
            raise ValueError('Finite Cpanel loss and logical fit costs required for selection')
        ranks[cell] = (nll, seconds)
    eligible = [c['candidate'] for c in comparisons
                if (c['N']['status'] == 'predictive_improvement' or c['G']['status'] == 'group_improvement')
                and c['robustness']['status'] != 'failed']
    ranked = sorted(eligible, key=ranks.__getitem__)[:2]
    if analysis['followup_candidates'] != ranked:
        raise ValueError('Frozen G follow-up ranking violates the predeclared rule')
    candidate = ranked[0] if ranked else min(CONTROLS, key=ranks.__getitem__)
    return {'candidate': candidate, 'control': CONTROLS[candidate],
            'status': 'screen_promoted' if ranked else 'diagnostic_only_not_promoted'}


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    return {**base_identity(config, local_path), 'source_hashes': source_hashes()}


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('T2 source/config/environment changed')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Frozen G selection dependency changed')
    return prep


def primitive_frame(local, data_prep):
    frame = load_verified_processed_cache(local)
    if (frame.attrs['sequence_data_identity'] != data_prep['dataset_identity'] or
            frame.attrs['matrix_source_provenance'] != data_prep['source_provenance']):
        raise ValueError('T2 primitive source differs from the frozen G data lineage')
    # Crucially, do not call regular_frame: it computes unsanitized cumulative
    # statistics. The fold helper reconstructs these AFTER observation masking.
    return assign_fold(frame.loc[frame.game_type.eq('R')].copy().reset_index(drop=True), 2025)


def plan_units(cells, train, early, clusters, panel_ids):
    required = set().union(*(MODES[cell] for cell in cells))
    units = {mode: {'mode': mode, 'train_n': len(train), 'earlystop_n': len(early)}
             for mode in ('global', 'feature') if mode in required}
    eligibility = []
    for mode, ids in [('cluster', range(4)), ('personal', panel_ids)]:
        if mode not in required:
            continue
        for entity in ids:
            if mode == 'cluster':
                mapping = {int(pid): c for pid, c in clusters['pitcher_cluster'].items()}
                tm, em = train.pitcher.map(mapping).eq(entity), early.pitcher.map(mapping).eq(entity)
            else:
                tm, em = train.pitcher.eq(entity), early.pitcher.eq(entity)
            n, e = int(tm.sum()), int(em.sum())
            allowed = n >= 500 and e >= 20
            eligibility.append({'mode': mode, 'id': int(entity), 'train_n': n, 'earlystop_n': e,
                                'fit': allowed, 'fallback': None if allowed else 'insufficient_allowed_fit_observations'})
            if allowed:
                units[f'{mode}{entity}'] = {'mode': mode, 'id': int(entity), 'train_n': n, 'earlystop_n': e}
    return units, eligibility


def _save_samples(directory, parts):
    result = {}
    for name, frame in parts.items():
        path = name + '_keys.parquet'
        frame[KEY].to_parquet(directory / path, index=False)
        result[name] = {'path': path, 'n': len(frame), 'games': int(frame.game_pk.nunique()),
                        'rows_sha256': ordered_key_hash(frame)}
    return result


def build_fold(config, frame, axis, selection, parent_parts, panel):
    """Pure in-memory statistical preparation; no neural training or DEV metric."""
    fold = GeneralizationFold(frame, axis, selection['ids'])
    reconstructed = fold.reconstruct('Z')
    features = reconstructed.feature_frame
    parts = {}
    for split in ('train', 'earlystop', 'temperature', 'blend'):
        rows = parent_parts[split].index.to_numpy()
        parts[split] = features.iloc[rows[fold.allowed_fit[split][rows] & fold.truth_eligible[rows]]]
        if not len(parts[split]):
            raise ValueError('T2 exclusion leaves an empty fitting/CAL partition: ' + split)
    # D100 includes the full eligible TRAIN pool. Reject an unregistered reduced
    # parent sample rather than silently fitting auxiliaries on extra examples.
    expected_train = frame.loc[fold.allowed_fit['train'] & fold.truth_eligible]
    if ordered_key_hash(parts['train']) != ordered_key_hash(expected_train):
        raise ValueError('T2 requires exact ordered eligible D100 TRAIN after exclusions')
    requested = fold.target_eval & frame.pitcher.isin(panel['pitcher_ids']).to_numpy()
    parts['dev'] = features.loc[requested & fold.truth_eligible]
    normalizer_train = features.loc[fold.allowed_fit['train']]
    normalizer = PhysicalNormalizer().fit(normalizer_train)
    context = SequenceContext().fit(parts['train'])
    delivery = JointDelivery().fit(parts['train'], normalizer, draws=400, seed=42)
    baseline = ContextFrequencyBaseline(HierarchicalFrequencyBaseline().fit(parts['train'])).fit(parts['train'])
    clusters = fit_pitcher_clusters(parts['train'])
    vocabulary = sorted(normalizer_train.pitch_type.dropna().astype(str).unique())
    cells = ['G0-global'] if axis == 'pitcher' else [config['candidate'], config['control']]
    units, eligibility = plan_units(cells, parts['train'], parts['earlystop'], clusters, panel['pitcher_ids'])
    aux = {'normalizer': normalizer, 'context': context, 'delivery': delivery, 'baseline': baseline}
    report = {'axis': axis, 'selection': selection, 'cells': cells, 'units': units,
        'individual_cluster_eligibility': eligibility, 'clusters': clusters,
        'type_vocabulary': vocabulary, 'prefix': fold.prefix_records,
        'coverage': {'requested_pitches': int(requested.sum()), 'eligible_pitches': len(parts['dev']),
                     'omitted_pitches': int(requested.sum()) - len(parts['dev'])},
        'active': bool(len(parts['dev'])), 'normalizer_train_keys_sha256': ordered_key_hash(normalizer_train),
        'features': {'normalizer': normalizer.report(), 'context': SharingContext(context, clusters).report()},
        'aux_refit': 'all auxiliaries refitted; normalizer/type vocabulary use allowed whole-PA TRAIN, others eligible D100 TRAIN',
        'pitcher_adaptation': 'unknown league routing in Z/W/O; O may change permitted batter context but does not adapt pitcher profile, delivery or experts',
        'calibration': 'one allowed-May temperature per cell/seed and identical June predictions for all regimes',
        'evaluation_metadata': 'canonical TRAIN count/seen/role from actual sanitized supervised TRAIN; volume uses frozen G thresholds; selection_* preserves original panel strata',
        'truth': 'original primitive labels/eligibility; sanitized statistical primitives never determine targets'}
    return fold, parts, aux, report


def _baseline_artifacts(parts, aux):
    temp = parts['temperature']
    selected = fit_temperature(aux['baseline'].predict(temp), outcome_labels(temp))
    values = {}
    for split in ('blend', 'dev'):
        part = parts[split]
        raw = aux['baseline'].predict(part) if len(part) else np.empty((0, 10))
        values[split + '_raw'] = raw
        values[split] = temperature_predictions(raw, selected['temperature']) if len(part) else raw.copy()
        values.update(_metadata_values(part, split))
    return values, selected


def _metadata_values(frame, prefix):
    return {prefix + '_keys': frame[KEY].to_numpy(np.int64), prefix + '_y': outcome_labels(frame),
            prefix + '_pitcher': frame.pitcher.to_numpy(np.int64), prefix + '_batter': frame.batter.to_numpy(np.int64),
            prefix + '_game_pk': frame.game_pk.to_numpy(np.int64)}



def fold_metadata(frame, panel, actual_train):
    """Separate frozen panel-selection strata from actual supervised TRAIN exposure."""
    result = evaluation_metadata(frame, panel)
    for name in ('train_pitches', 'train_role', 'train_volume', 'seen_pitcher', 'seen_batter'):
        result['selection_' + name] = result[name]
    counts = actual_train.groupby('pitcher').size()
    result['train_pitches'] = frame.pitcher.map(counts).fillna(0).to_numpy(np.int64)
    result['seen_pitcher'] = frame.pitcher.isin(counts.index).to_numpy()
    result['seen_batter'] = frame.batter.isin(actual_train.batter.unique()).to_numpy()
    q25, q75 = (panel['volume_thresholds'][name] for name in ('q25', 'q75'))
    n = result.train_pitches.to_numpy()
    result['train_volume'] = np.select([n == 0, n <= q25, n <= q75], ['zero', 'low', 'middle'], default='high')
    # Role is descriptive observed TRAIN role, not the original selection role.
    shares = actual_train.assign(_starter=actual_train.pitcher.eq(actual_train.starter_pitcher)).groupby('pitcher')._starter.mean()
    result['train_role'] = ['unseen' if int(pid) not in shares else
                            'starter' if shares.loc[int(pid)] >= .5 else 'relief' for pid in frame.pitcher]
    return result

def prepare(config, local_path, local, output, expected):
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        return
    parent = Path(config['parent_run']).resolve()
    check_location(local, parent)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('T2 output must be a distinct sibling of the G run')
    shared_config = read_json(parent / 'registered_config.json')
    shared = verify_sharing(parent, sharing_identity(shared_config, local_path))
    analysis = parent / 'analysis' / 'panel'
    manifest = read_json(analysis / 'manifest.json')
    if (hash_file(parent / 'preparation.json') != config['parent_preparation_sha256']
            or hash_file(analysis / 'results.json') != config['parent_analysis_sha256']
            or manifest['results_sha256'] != config['parent_analysis_sha256']
            or hash_file(analysis / 'predictions.npz') != manifest['predictions_sha256']):
        raise ValueError('Frozen G preparation/selection identity changed')
    result = read_json(analysis / 'results.json')
    chosen = chosen_comparison(result)
    if (any(config[name] != chosen[name] for name in ('candidate', 'control'))
            or config['selection_status'] != chosen['status']):
        raise ValueError('Selected T2 comparison differs from the common transfer rule')
    external = {**manifest['inputs'], str(analysis / 'results.json'): config['parent_analysis_sha256'],
                str(analysis / 'manifest.json'): hash_file(analysis / 'manifest.json'),
                str(analysis / 'predictions.npz'): manifest['predictions_sha256'],
                str(parent / 'preparation.json'): config['parent_preparation_sha256']}
    for name, digest in external.items():
        if hash_file(Path(name)) != digest:
            raise ValueError('G selection dependency changed before T2 preparation')
    data_prep = read_json(parent / 'parent_preparation.json')
    frame = primitive_frame(local, data_prep)
    parent_parts = {name: select_keys(frame, pd.read_parquet(parent / record['path']), record)
                    for name, record in shared['samples'].items() if name != 'mlb_dev'}
    panel = shared['panel']
    selections = {'pitcher': select_heldout_pitchers(panel), 'batter': select_heldout_batters(parent_parts['train'])}
    output.mkdir(parents=True)
    for name in SOURCES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    dump(output / 'registered_config.json', config)
    dump(output / 'data_preparation.json', data_prep)
    dump(output / 'sharing_preparation.json', shared)
    shutil.copyfile(analysis / 'results.json', output / 'selection_analysis.json')
    folds = {}
    for axis in AXES:
        directory = output / axis
        directory.mkdir()
        if not selections[axis]['ids']:
            folds[axis] = {'active': False, 'selection': selections[axis], 'reason': 'empty_TRAIN_hash_cohort', 'units': {}, 'cells': []}
            continue
        fold, parts, aux, report = build_fold(config, frame, axis, selections[axis], parent_parts, panel)
        report['samples'] = _save_samples(directory, parts)
        with (directory / 'aux.pkl').open('wb') as stream:
            pickle.dump(aux, stream)
        baseline, report['baseline_temperature'] = _baseline_artifacts(parts, aux)
        np.savez_compressed(directory / 'baseline_predictions.npz', **baseline)
        fold_metadata(parts['dev'], panel, parts['train']).to_parquet(directory / 'dev_metadata.parquet', index=False)
        frame.loc[fold.prefix, [*KEY, axis]].to_parquet(directory / 'exposure_keys.parquet', index=False)
        folds[axis] = report
        del fold, parts, aux, baseline
    raw_panel = frame.loc[frame.split.eq('dev') & frame.pitcher.isin(panel['pitcher_ids'])]
    natural = audit_new_matchups(raw_panel, history_sources={'complete_verified_history': frame}, fitting_train=parent_parts['train'])
    requested = raw_panel.loc[natural['mask']]
    positions = pd.MultiIndex.from_frame(requested[KEY]).get_indexer(pd.MultiIndex.from_frame(parent_parts['dev'][KEY]))
    kept = np.flatnonzero(positions >= 0)
    with np.load(analysis / 'predictions.npz', allow_pickle=False) as archive:
        if not np.array_equal(archive['keys'], parent_parts['dev'][KEY].to_numpy(np.int64)):
            raise ValueError('Natural matchup parent keys differ')
        if not np.array_equal(archive['y'], outcome_labels(parent_parts['dev'])):
            raise ValueError('Natural matchup original ground truth differs from parent archive')
        values = {name: archive[name][kept] for name in ('keys', 'y', 'game_pk', 'pitcher')}
        for cell in (config['candidate'], config['control']):
            for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
                name = cell + '_' + kind
                values[name] = archive[name][:, kept] if kind == 'seed_primary' else archive[name][kept]
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as archive:
        if not np.array_equal(archive['dev_keys'], parent_parts['dev'][KEY].to_numpy(np.int64)):
            raise ValueError('Natural matchup frequency baseline keys differ')
        values['frequency'] = archive['dev'][kept]
    values['batter'] = parent_parts['dev'].iloc[kept].batter.to_numpy(np.int64)
    np.savez_compressed(output / 'natural_matchup_predictions.npz', **values)
    fold_metadata(parent_parts['dev'].iloc[kept], panel, parent_parts['train']).to_parquet(
        output / 'natural_matchup_metadata.parquet', index=False)
    natural['report']['eligible_archived_pitches'] = len(kept)
    dump(output / 'natural_matchup_audit.json', natural['report'])
    if source_hashes() != expected['source_hashes']:
        raise ValueError('T2 source changed during preparation')
    files = [str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()]
    dump(output / 'preparation.json', {'identity': expected, 'folds': folds, 'panel': panel, 'selection': chosen,
        'artifact_hashes': artifact_hashes(output, files), 'external_hashes': external,
        'natural_matchup': natural['report'], 'new_DEV_scores_read': False,
        'resource_limits': {'profile_seconds': 600, 'per_seed_fit_seconds': 7200,
                            'enforcement': 'caller subprocess timeouts; shared heavy lock in CLI'}})


def load_fold(local, output, prep, axis, regime='Z'):
    spec = prep['folds'][axis]
    if not spec['active']:
        raise ValueError('T2 fold is unmeasured; no replacement allowed')
    frame = primitive_frame(local, read_json(output / 'data_preparation.json'))
    fold = GeneralizationFold(frame, axis, spec['selection']['ids'])
    reconstructed = fold.reconstruct(regime)
    with (output / axis / 'aux.pkl').open('rb') as stream:
        aux = pickle.load(stream)
    store = reconstructed.history_store(normalizer=aux['normalizer'], type_vocabulary=spec['type_vocabulary'])
    context = SharingContext(aux['context'], spec['clusters'])
    parts = {name: select_keys(store.frame, pd.read_parquet(output / axis / record['path']), record)
             for name, record in spec['samples'].items()}
    return fold, store, context, parts, aux


def _unit_parts(parts, spec, clusters):
    train, early = parts['train'], parts['earlystop']
    if spec['mode'] == 'cluster':
        mapping = {int(pid): c for pid, c in clusters['pitcher_cluster'].items()}
        return train.loc[train.pitcher.map(mapping).eq(spec['id'])], early.loc[early.pitcher.map(mapping).eq(spec['id'])]
    if spec['mode'] == 'personal':
        return train.loc[train.pitcher.eq(spec['id'])], early.loc[early.pitcher.eq(spec['id'])]
    return train, early


def _device(config):
    return None if config['device'] == 'auto' else config['device']


def profile(config, local, output, prep, axis):
    dest = output / axis / 'profile'
    expected = {'preparation_sha256': hash_file(output / 'preparation.json'), 'axis': axis}
    if (dest / 'state.json').exists():
        state = read_json(dest / 'state.json')
        if state['identity'] != expected:
            raise ValueError('T2 profile parent differs')
        assert_hashes(dest, state['artifact_hashes'])
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Interrupted T2 profile requires review')
    start = time.perf_counter()
    _, store, context, parts, aux = load_fold(local, output, prep, axis)
    train = parts['train'].iloc[np.linspace(0, len(parts['train']) - 1, min(8192, len(parts['train'])), dtype=int)]
    early = parts['earlystop'].iloc[np.linspace(0, len(parts['earlystop']) - 1, min(2048, len(parts['earlystop'])), dtype=int)]
    ta, ea = arrays(store, context, train.index.to_numpy()), arrays(store, context, early.index.to_numpy())
    timings, models = {}, {}
    enriched_needed = any(s['mode'] in ('feature', 'cluster') for s in prep['folds'][axis]['units'].values())
    for name, enriched in [('global', False), *([('feature', True)] if enriched_needed else [])]:
        before = time.perf_counter()
        model = MatrixModel('flatten_mlp', seed=0, width=128, device=_device(config)).fit(
            training_arrays(ta, enriched), outcome_labels(train), training_arrays(ea, enriched), outcome_labels(early),
            epochs=2, patience=2, batch_size=1024, learning_rate=.0005)
        timings[name], models[name] = time.perf_counter() - before, model
    query = parts['temperature'].iloc[:16]
    model = SharingPredictor('G2-feature' if enriched_needed else 'G0-global', models['global'],
                             prep['folds'][axis]['clusters'], feature_model=models.get('feature'))
    before = time.perf_counter()
    aux['delivery'].calibrate(model, store, context, query.index.to_numpy(), outcome_labels(query))
    predict_streamed(model, aux['delivery'], store, context, query.index.to_numpy(), chunk_size=16)
    infer_seconds = time.perf_counter() - before
    units = prep['folds'][axis]['units']
    projected_fit = max(timings.values()) * sum(s['train_n'] for s in units.values()) / len(train) * 15
    projected_predict = infer_seconds * (len(parts['temperature']) + len(parts['blend']) + 3*len(parts['dev'])) / len(query)
    report = {'fit_seconds': timings, 'train_rows': len(train), 'earlystop_rows': len(early),
        'temperature_rows': len(query), 'projected_per_seed_fit_seconds': projected_fit,
        'projected_per_cell_prediction_seconds': projected_predict,
        'resource_gate': projected_fit <= 7200 and projected_predict <= 7200,
        'seconds_total': time.perf_counter() - start, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'dev_features_gathered': False, 'quality_scores_emitted': False,
        'note': 'Two-epoch linear extrapolation; caller must enforce 600s profile and 7200s member command timeouts'}
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('T2 sources changed during profiling')
    dest.mkdir(parents=True)
    dump(dest / 'profile.json', report)
    dump(dest / 'state.json', {'identity': expected, 'artifact_hashes': artifact_hashes(dest, ['profile.json'])})


def _profile_gate(output, axis):
    dest = output / axis / 'profile'
    state = read_json(dest / 'state.json')
    if state['identity'] != {'preparation_sha256': hash_file(output / 'preparation.json'), 'axis': axis}:
        raise ValueError('Matching T2 resource profile required')
    assert_hashes(dest, state['artifact_hashes'])
    if read_json(dest / 'profile.json')['resource_gate'] is not True:
        raise ValueError('T2 resource profile exceeds the registered per-command budget')


def fit(config, local, output, prep, axis, seed):
    if axis not in AXES or seed not in SEEDS:
        raise ValueError('Unregistered T2 fit member')
    _profile_gate(output, axis)
    _, store, context, parts, _ = load_fold(local, output, prep, axis)
    spec = prep['folds'][axis]
    for unit, record in spec['units'].items():
        dest = output / axis / 'fits' / f'seed{seed}' / unit
        expected = {'preparation_sha256': hash_file(output / 'preparation.json'), 'axis': axis,
                    'seed': seed, 'unit': unit, 'spec': record}
        if (dest / 'state.json').exists():
            state = read_json(dest / 'state.json')
            if state['identity'] != expected:
                raise ValueError('T2 fit identity changed')
            assert_hashes(dest, state['artifact_hashes'])
            continue
        if dest.exists() and any(dest.iterdir()):
            raise ValueError('Interrupted T2 fit requires review')
        start = time.perf_counter()
        tr, er = _unit_parts(parts, record, spec['clusters'])
        enriched = record['mode'] in ('feature', 'cluster')
        ta = training_arrays(arrays(store, context, tr.index.to_numpy()), enriched)
        ea = training_arrays(arrays(store, context, er.index.to_numpy()), enriched)
        model = MatrixModel('flatten_mlp', seed=seed, width=128, device=_device(config)).fit(
            ta, outcome_labels(tr), ea, outcome_labels(er), **config['budget'])
        dest.mkdir(parents=True)
        model.save(dest / 'model.pt')
        dump(dest / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter() - start,
            'train_keys_sha256': ordered_key_hash(tr), 'earlystop_keys_sha256': ordered_key_hash(er),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if source_hashes() != prep['identity']['source_hashes']:
            raise ValueError('T2 sources changed during fit')
        dump(dest / 'state.json', {'identity': expected, 'artifact_hashes': artifact_hashes(dest, ['model.pt', 'fit.json'])})
        del model, ta, ea


def _predictor(config, output, prep, axis, cell, seed):
    spec = prep['folds'][axis]
    if cell not in spec['cells']:
        raise ValueError('Cell is absent from the selected T2 fold')
    global_model, feature, personal, clusters, dependencies = None, None, {}, {}, {}
    for unit, record in spec['units'].items():
        if record['mode'] not in MODES[cell]:
            continue
        dest = output / axis / 'fits' / f'seed{seed}' / unit
        state = read_json(dest / 'state.json')
        if state['identity'] != {'preparation_sha256': hash_file(output / 'preparation.json'), 'axis': axis,
                                  'seed': seed, 'unit': unit, 'spec': record}:
            raise ValueError('T2 predictor dependency identity changed')
        assert_hashes(dest, state['artifact_hashes'])
        dependencies[str(dest / 'state.json')] = hash_file(dest / 'state.json')
        for name, digest in state['artifact_hashes'].items():
            dependencies[str(dest / name)] = digest
        model = MatrixModel.load(dest / 'model.pt', device=_device(config))
        if record['mode'] == 'global': global_model = model
        elif record['mode'] == 'feature': feature = model
        elif record['mode'] == 'personal': personal[record['id']] = model
        else: clusters[record['id']] = model
    return SharingPredictor(cell, global_model, spec['clusters'], feature_model=feature,
        personal_models=personal, cluster_models=clusters, individual_tau=1000., cluster_tau=10000.), dependencies


def predict(config, local, output, prep, axis, cell, seed):
    if axis not in AXES or seed not in SEEDS or cell not in prep['folds'][axis]['cells']:
        raise ValueError('Unregistered T2 prediction member')
    _profile_gate(output, axis)
    dest = output / axis / 'members' / cell / f'seed{seed}'
    expected = {'preparation_sha256': hash_file(output / 'preparation.json'), 'axis': axis, 'cell': cell, 'seed': seed}
    if (dest / 'prediction_state.json').exists():
        state = read_json(dest / 'prediction_state.json')
        if state['identity'] != expected:
            raise ValueError('T2 prediction identity changed')
        assert_hashes(dest, state['artifact_hashes'])
        for name, digest in state['dependencies'].items():
            if hash_file(Path(name)) != digest:
                raise ValueError('T2 prediction checkpoint changed')
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Interrupted T2 prediction requires review')
    start = time.perf_counter()
    model, dependencies = _predictor(config, output, prep, axis, cell, seed)
    fold, store, context, parts, aux = load_fold(local, output, prep, axis, 'Z')
    temp = parts['temperature']
    aux['delivery'].calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
    blend = parts['blend']
    p, raw, levels = predict_streamed(model, aux['delivery'], store, context, blend.index.to_numpy())
    values = {**_metadata_values(blend, 'blend'), 'blend': p, 'blend_raw': raw, 'blend_delivery_level': levels,
              **_metadata_values(parts['dev'], 'dev')}
    for regime in REGIMES:
        if regime != 'Z':
            # Reuse the verified primitive pool; only observation statistics and
            # guarded history visibility change. No auxiliary object is refitted.
            reconstructed = fold.reconstruct(regime)
            store = reconstructed.history_store(normalizer=aux['normalizer'], type_vocabulary=prep['folds'][axis]['type_vocabulary'])
        rows = parts['dev'].index.to_numpy()
        p, raw, levels = predict_streamed(model, aux['delivery'], store, context, rows)
        values.update({regime: p, regime + '_raw': raw, regime + '_delivery_level': levels})
        if regime != 'Z' and not np.array_equal(levels, values['Z_delivery_level']):
            raise ValueError('T2 context-only adaptation must preserve delivery draws/tiers')
    dest.mkdir(parents=True)
    np.savez_compressed(dest / 'predictions.npz', **values)
    dump(dest / 'calibration.json', model.report)
    dump(dest / 'runtime.json', {'seconds_total': time.perf_counter() - start,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'regimes': list(REGIMES), 'calibration_shared_across_regimes': True,
        'pitcher_expert_adaptation': False, 'new_DEV_scores_read': False})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('T2 sources changed during prediction')
    dump(dest / 'prediction_state.json', {'identity': expected, 'dependencies': dependencies,
        'artifact_hashes': artifact_hashes(dest, ['predictions.npz', 'calibration.json', 'runtime.json'])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare'); commands.add_parser('status')
    for command in ('profile', 'fit', 'predict'):
        sub = commands.add_parser(command)
        sub.add_argument('--axis', choices=AXES, required=True)
        if command != 'profile': sub.add_argument('--seed', type=int, choices=SEEDS, required=True)
        if command == 'predict': sub.add_argument('--cell', choices=MODES, required=True)
    args = parser.parse_args()
    config, local = config_check(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, args.local_config)
    if args.command == 'status':
        prep = verify(output, expected)
        print({axis: {'active': fold['active'], 'units_per_seed': len(fold['units']), 'cells': fold['cells']}
               for axis, fold in prep['folds'].items()})
        return
    with heavy_lock(root):
        if args.command == 'prepare': prepare(config, args.local_config, local, output, expected)
        else:
            prep = verify(output, expected)
            if args.command == 'profile': profile(config, local, output, prep, args.axis)
            elif args.command == 'fit': fit(config, local, output, prep, args.axis, args.seed)
            else: predict(config, local, output, prep, args.axis, args.cell, args.seed)


if __name__ == '__main__':
    main()
