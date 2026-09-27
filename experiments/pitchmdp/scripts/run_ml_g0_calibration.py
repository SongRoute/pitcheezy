"""Registered, additive stage-3 G0 probability correction worker.

This worker never trains a network or writes into either parent. An external
supervisor owns the 3600-second authoritative monotonic wall ledger. Mutating
commands acquire the shared ML matrix heavy lock inside this worker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]

import numpy as np
import pandas as pd
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_g0_calibration import (
    fixed_spec, validate_config as validate_calibration_config, config_sha256,
    array_sha256, weakness_indicators, fit_class_bias_family,
    fit_volume_blend_family, load_blend_optimizer, apply_class_bias,
    apply_volume_blend,
)
from pitchmdp.matrix_g0_whole_metrics import aligned_population, evaluate_candidates
from pitchmdp.matrix_metrics import validate_probabilities
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock
from score_ml_matrix import archive

SLOTS = ('I1', 'I2')
PREDICTORS = ('ensemble', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4')
STAGE2_FILES = ('preparation.json', 'analysis/manifest.json',
                'analysis/results.json', 'analysis/predictions.npz')
C1_FILES = ('registered_config.json', 'preparation.json', 'analysis/manifest.json',
            'analysis/results.json', 'analysis/predictions.npz',
            'parent_baseline_predictions.npz')
G_FILES = ('registered_config.json', 'preparation.json', 'panel.json', 'baseline_predictions.npz',
           'blend_metadata.parquet', 'mlb_dev_metadata.parquet')
SCORING = {
    'protocol': 'g0_calibration_scoring_v1',
    'seeds': [0, 1, 2, 3, 4],
    'primary_slots': ['I1', 'I2'],
    'bootstrap_draws': 10000,
    'bootstrap_seed': 20260924,
    'holm_alpha': 0.05,
    'nll_delta_max': -0.003,
    'nll_ci_upper_max': 0.0,
    'brier_ci_upper_max': 0.001,
    'negative_seeds_required': 4,
    'minimum_games': 30,
    'minimum_pitches': 500,
    'R': {'family_size': 52, 'draws': 100000, 'alpha': 0.05,
          'nll_margin': 0.010, 'brier_margin': 0.002},
    'inactive_p': 1.0,
}
REQUIRED_SOURCES = {
    'pitchmdp/matrix_g0_calibration.py', 'pitchmdp/matrix_g0_whole_metrics.py',
    'pitchmdp/matrix_panel.py', 'pitchmdp/matrix_metrics.py',
    'pitchmdp/matrix_group_metrics.py', 'scripts/run_ml_g0_calibration.py',
    'scripts/score_ml_g0_whole.py', 'scripts/run_sequence_calibration.py',
    'pitchmdp/data.py', 'scripts/score_ml_matrix.py',
    'scripts/run_ml_benchmark.py', 'scripts/run_ml_matrix.py',
}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _git_sha(value):
    return isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{label} must have exactly {sorted(keys)}')


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def validate_config(config):
    """No scored stage-2 hash belongs in the preregistered config."""
    _exact(config, {'experiment_id', 'output', 'stage2', 'c1', 'g', 'calibration',
                    'source_commit', 'sources', 'contract', 'budget_seconds', 'scoring'}, 'stage3 config')
    if config['experiment_id'] != 'EXP-P11-002' or config['budget_seconds'] != 3600:
        raise ValueError('Wrong stage3 identity or budget')
    _exact(config['stage2'], {'experiment_id', 'run', 'config_file', 'config_sha256',
                              'source_commit', 'required_files'}, 'stage2 declaration')
    if config['stage2']['experiment_id'] != 'EXP-P11-001' or \
            tuple(config['stage2']['required_files']) != STAGE2_FILES:
        raise ValueError('Stage2 identity or required artifact inventory changed')
    for parent, names, files in (('c1', {'run', 'files'}, C1_FILES),
                                  ('g', {'run', 'files'}, G_FILES)):
        _exact(config[parent], names, parent)
        if not isinstance(config[parent]['files'], dict) or set(config[parent]['files']) != set(files):
            raise ValueError(parent + ' file inventory changed')
        if any(not _sha(digest) for digest in config[parent]['files'].values()):
            raise ValueError(parent + ' frozen file SHA256 pins are required')
    validate_calibration_config(config['calibration'])
    if _canonical(config['scoring']) != _canonical(SCORING):
        raise ValueError('Stage3 scoring settings differ from the fixed registered family')
    if config['calibration']['fit_population'] == {}:
        raise ValueError('June identity must be frozen before stage2 quality')
    if not _sha(config['stage2']['config_sha256']) or not _git_sha(config['source_commit']) or \
            not _git_sha(config['stage2']['source_commit']):
        raise ValueError('Invalid source/config commit pin')
    if not isinstance(config['sources'], dict) or not REQUIRED_SOURCES <= set(config['sources']):
        raise ValueError('Missing required source closure')
    if any(not _sha(v) for v in config['sources'].values()):
        raise ValueError('Malformed source pin')
    _exact(config['contract'], {'path', 'sha256'}, 'contract')
    if not _sha(config['contract']['sha256']):
        raise ValueError('Malformed contract pin')
    if not Path(config['output']).is_absolute() or any(
            not Path(config[k]['run']).is_absolute() for k in ('stage2', 'c1', 'g')):
        raise ValueError('Registered output and parent paths must be absolute')
    roots = [Path(config[k]['run']).resolve() for k in ('stage2', 'c1', 'g')]
    output = Path(config['output']).resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in roots):
        raise ValueError('Stage3 output must be disjoint from immutable parents')
    if len(set(roots)) != len(roots):
        raise ValueError('Parent roots must be distinct')
    return config


def verify_config_pins(config, config_path):
    validate_config(config)
    if subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() != config['source_commit']:
        raise ValueError('Checked-out source commit differs')
    for name, digest in config['sources'].items():
        path = (PROJECT / name).resolve()
        if not path.is_relative_to(PROJECT) or hash_file(path) != digest:
            raise ValueError('Pinned source changed: ' + name)
    contract = config['contract']
    if hash_file(_path(contract['path'])) != contract['sha256']:
        raise ValueError('Pinned contract changed')
    if hash_file(_path(config['stage2']['config_file'])) != config['stage2']['config_sha256']:
        raise ValueError('Stage2 preregistered config changed')
    return {'config_sha256': hash_file(config_path), 'config_canonical_sha256': _canonical(config),
            'source_commit': config['source_commit'], 'sources': config['sources'],
            'contract': contract, 'calibration_sha256': config_sha256(config['calibration'])}


def _assert_hashes(records):
    for filename, digest in records.items():
        if not _sha(digest) or hash_file(Path(filename)) != digest:
            raise ValueError('Frozen dependency changed: ' + filename)


def _verify_prep_hashes(root, prep):
    artifacts = prep.get('artifact_hashes')
    external = prep.get('external_hashes')
    if not isinstance(artifacts, dict) or not isinstance(external, dict):
        raise ValueError('Parent preparation lacks frozen artifact or external hashes')
    _assert_hashes({str((root / name).resolve()): digest for name, digest in artifacts.items()})
    _assert_hashes(external)


def _declared(parent, names):
    run = Path(parent['run']).resolve()
    if not run.is_dir():
        raise ValueError('Missing parent root: ' + str(run))
    records = {str((run / name).resolve()): hash_file(run / name) for name in names}
    if isinstance(parent.get('files'), dict):
        for name, expected in parent['files'].items():
            if records[str((run / name).resolve())] != expected:
                raise ValueError('Frozen old parent changed: ' + name)
    return records


def bind_stage2(config):
    """Materialize actual completed stage2 digests without editing preregistered config."""
    root = Path(config['stage2']['run']).resolve()
    records = _declared(config['stage2'], STAGE2_FILES)
    prep = read_json(root / 'preparation.json')
    _verify_prep_hashes(root, prep)
    stage2_config = read_json(_path(config['stage2']['config_file']))
    if config['stage2']['source_commit'] != config['source_commit'] or \
            stage2_config.get('registration', {}).get('execution_code_commit') != config['source_commit'] or \
            stage2_config.get('experiment_id') != 'EXP-P11-001' or \
            prep.get('identity', {}).get('config_sha256') != _canonical(stage2_config):
        raise ValueError('Stage2 preregistered config/source identity differs')
    stage2_sources = stage2_config['registration']['sources']
    worker_sources = prep['identity'].get('source_hashes')
    if not isinstance(stage2_sources, dict) or not isinstance(worker_sources, dict) or \
            any(stage2_sources.get(name) != digest for name, digest in worker_sources.items()):
        raise ValueError('Stage2 worker sources differ from preregistration')
    for name, digest in stage2_sources.items():
        source = PROJECT / name
        if hash_file(source) != digest:
            raise ValueError('Stage2 preregistered source changed: ' + name)
        if name in worker_sources and hash_file(root / 'source' / name) != digest:
            raise ValueError('Stage2 frozen worker snapshot changed: ' + name)
    result = read_json(root / 'analysis/results.json')
    manifest = read_json(root / 'analysis/manifest.json')
    if prep.get('experiment_id') != 'EXP-P11-001':
        raise ValueError('Stage2 preparation identity differs')
    if manifest['results_sha256'] != records[str((root / 'analysis/results.json').resolve())] or \
            manifest['predictions_sha256'] != records[str((root / 'analysis/predictions.npz').resolve())]:
        raise ValueError('Stage2 analysis manifest differs from files')
    if result.get('replay', {}).get('passed') is not True or result.get('n') != 311721:
        raise ValueError('Stage2 five-member replay/whole population not complete')
    if result.get('independent_confirmation') is not None or result.get('held_out_confirmation') is not False:
        raise ValueError('Stage2 development boundary changed')
    _assert_hashes(manifest.get('inputs', {}))
    for seed in range(5):
        member = root / 'members/G0-global' / f'seed{seed}'
        state_path = member / 'prediction_state.json'
        state = read_json(state_path)
        if state.get('identity', {}).get('seed') != seed or state['identity'].get('cell') != 'G0-global':
            raise ValueError('Stage2 member order/identity differs')
        for name, digest in state['artifact_hashes'].items():
            records[str((member / name).resolve())] = digest
        records[str(state_path.resolve())] = hash_file(state_path)
        _assert_hashes(state.get('dependencies', {}))
        records.update(state.get('dependencies', {}))
    _assert_hashes(records)
    return {'experiment_id': 'EXP-P11-001', 'run': str(root),
            'preregistered_config_sha256': config['stage2']['config_sha256'],
            'preregistered_source_commit': config['stage2']['source_commit'],
            'artifacts': records, 'analysis_inputs': manifest['inputs'],
            'bound_utc': datetime.now(timezone.utc).isoformat()}


def bind_old_parents(config):
    records = {}
    for label, names in (('c1', C1_FILES), ('g', G_FILES)):
        records.update(_declared(config[label], names))
    c1 = Path(config['c1']['run']).resolve()
    g = Path(config['g']['run']).resolve()
    if read_json(c1 / 'registered_config.json').get('experiment_id') != 'EXP-P10-001' or \
            read_json(g / 'registered_config.json').get('experiment_id') != 'EXP-P4-001':
        raise ValueError('Old parent experiment identity differs')
    _verify_prep_hashes(c1, read_json(c1 / 'preparation.json'))
    c1_manifest = read_json(c1 / 'analysis/manifest.json')
    if c1_manifest['results_sha256'] != records[str((c1 / 'analysis/results.json').resolve())] or \
            c1_manifest['predictions_sha256'] != records[str((c1 / 'analysis/predictions.npz').resolve())]:
        raise ValueError('C1 analysis manifest disagrees')
    _assert_hashes(c1_manifest['inputs'])
    records.update(c1_manifest['inputs'])
    g_prep = read_json(g / 'preparation.json')
    _verify_prep_hashes(g, g_prep)
    return records


def _june(config, old_hashes):
    """Read only existing June Cpanel arrays; return a copyable fixed input set."""
    g = Path(config['g']['run']).resolve()
    c1 = Path(config['c1']['run']).resolve()
    base = archive(c1 / 'parent_baseline_predictions.npz')
    keys = np.asarray(base['blend_keys'], dtype=np.int64)
    y = np.asarray(base['blend_y'], dtype=np.int64)
    if keys.shape != (4821, 3) or y.shape != (4821,) or \
            array_sha256(keys, np.int64) != config['calibration']['fit_population']['ordered_keys_sha256'] or \
            array_sha256(y, np.int64) != config['calibration']['fit_population']['labels_sha256']:
        raise ValueError('June population differs from preregistered pins')
    if not np.array_equal(keys[:, 0], base['blend_game_pk']):
        raise ValueError('June key/game alignment differs')
    frequency = np.asarray(base['blend'], dtype=np.float64)
    validate_probabilities(y, frequency)
    metadata = pd.read_parquet(g / 'blend_metadata.parquet')
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), keys) or \
            not np.array_equal(metadata['pitcher'].to_numpy(np.int64), base['blend_pitcher']):
        raise ValueError('June metadata and probabilities differ')
    panel = read_json(g / 'panel.json')
    lookup = {int(row['pitcher']): row['train_volume'] for row in panel['train_players']}
    groups = np.asarray([lookup.get(int(pid), 'zero') for pid in base['blend_pitcher']])
    if not np.array_equal(groups, metadata['train_volume'].to_numpy()):
        raise ValueError('June TRAIN-volume grouping differs from original panel')
    c1_manifest = read_json(c1 / 'analysis/manifest.json')
    members = []
    for seed in range(5):
        suffix = f'/members/G0-global/seed{seed}/predictions.npz'
        found = [p for p in c1_manifest['inputs'] if p.endswith(suffix)]
        if len(found) != 1:
            raise ValueError('C1 June member identity is missing or ambiguous: ' + str(seed))
        path = Path(found[0])
        if hash_file(path) != c1_manifest['inputs'][found[0]]:
            raise ValueError('June member changed')
        item = archive(path)
        for tag in ('keys', 'y', 'game_pk', 'pitcher'):
            if not np.array_equal(item['blend_' + tag], base['blend_' + tag]):
                raise ValueError('June member rows differ')
        validate_probabilities(y, item['blend'])
        members.append(item['blend'])
        old_hashes[str(path.resolve())] = hash_file(path)
    report = read_json(c1 / 'analysis/results.json')['reports']['G0-global']
    weights = [report['selection']['model_weight'],
               *[row['blend_selection']['model_weight'] for row in report['seeds']]]
    if len(weights) != 6 or not all(np.isfinite(w) and 0 <= w <= 1 for w in weights):
        raise ValueError('Archived five-seed June weights differ')
    calibrated = np.mean(members, axis=0)
    primary = weights[0]*calibrated + (1-weights[0])*frequency
    seed_primary = np.stack([w*p + (1-w)*frequency for w, p in zip(weights[1:], members)])
    return dict(keys=keys, y=y, game_pk=base['blend_game_pk'], pitcher=base['blend_pitcher'],
                frequency=frequency, groups=groups, calibrated=calibrated,
                seed_calibrated=np.stack(members), primary=primary,
                seed_primary=seed_primary, weights=np.asarray(weights))


def _volume_diagnostics(stage2_result, june):
    groups = stage2_result['R']['groups']
    slices = stage2_result['slices']
    out = []
    for name in ('zero', 'low', 'middle', 'high'):
        mask = june['groups'] == name
        source = groups['volume_' + name]
        report = source['reporting']
        descriptive = slices['volume_' + name]
        delta = None if descriptive['n'] == 0 else (descriptive['g0']['log_loss'] - descriptive['frequency']['log_loss'])
        out.append({'group': name, 'june_games': int(len(np.unique(june['game_pk'][mask]))),
                    'june_pitches': int(mask.sum()), 'dev_games': report['games'],
                    'dev_pitches': report['n'],
                    'dev_g0_minus_frequency_nll': delta})
    return out


def _class_diagnostics(stage2_result):
    rows = stage2_result['reports']['primary']['classes']
    return [{'class': row['class'], 'events': row['support'],
             'observed_prevalence': row['observed_rate'],
             'predicted_prevalence': row['predicted_rate']} for row in rows]


def prepare(config, config_path, local_path):
    validate_native_runtime()
    pins = verify_config_pins(config, config_path)
    output = Path(config['output']).resolve()
    local = read_json(local_path)
    check_location(local, output)
    if output.exists():
        raise ValueError('Stage3 output exists; preserve and review, then use a fresh registered run')
    stage2 = bind_stage2(config)
    old = bind_old_parents(config)
    june = _june(config, old)
    stage2_result = read_json(Path(config['stage2']['run']) / 'analysis/results.json')
    indicators = weakness_indicators(class_diagnostics=_class_diagnostics(stage2_result),
                                     volume_diagnostics=_volume_diagnostics(stage2_result, june),
                                     config=config['calibration'])
    activation = {slot: indicators[slot]['status'] == 'met' for slot in SLOTS}
    _assert_hashes(stage2['artifacts'])
    _assert_hashes(old)
    output.mkdir(parents=True)
    dump(output / 'stage2_binding.json', stage2)
    np.savez_compressed(output / 'june_inputs.npz', **june)
    dump(output / 'activation.json', {'indicators': indicators, 'activation': activation,
        'stage2_binding_sha256': hash_file(output / 'stage2_binding.json'),
        'stage2_result_sha256': stage2['artifacts'][str((Path(config['stage2']['run'])/'analysis/results.json').resolve())],
        'status': 'sealed_pre_fit'})
    dump(output / 'preparation.json', {'identity': pins, 'old_hashes': old,
        'stage2_binding_sha256': hash_file(output / 'stage2_binding.json'),
        'june_inputs_sha256': hash_file(output / 'june_inputs.npz'),
        'activation_sha256': hash_file(output / 'activation.json'),
        'prepared_utc': datetime.now(timezone.utc).isoformat()})


def verify_prepared(config, config_path, output):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != verify_config_pins(config, config_path):
        raise ValueError('Stage3 source/config identity changed')
    _assert_hashes(prep['old_hashes'])
    for name, field in (('stage2_binding.json', 'stage2_binding_sha256'),
                        ('june_inputs.npz', 'june_inputs_sha256'),
                        ('activation.json', 'activation_sha256')):
        if hash_file(output / name) != prep[field]:
            raise ValueError('Sealed stage3 preparation changed: ' + name)
    binding = read_json(output / 'stage2_binding.json')
    _assert_hashes(binding['artifacts'])
    _assert_hashes(binding['analysis_inputs'])
    activation = read_json(output / 'activation.json')
    if activation['status'] != 'sealed_pre_fit' or set(activation['activation']) != set(SLOTS) or \
            any(type(activation['activation'][s]) is not bool for s in SLOTS) or \
            activation['stage2_binding_sha256'] != prep['stage2_binding_sha256']:
        raise ValueError('Activation is not sealed and complete')
    return prep, activation['activation']


def _ordered(values):
    return {name: values[0] if i == 0 else values[1][i-1] for i, name in enumerate(PREDICTORS)}


def fit(config, config_path):
    output = Path(config['output']).resolve()
    prep, activation = verify_prepared(config, config_path, output)
    if (output / 'fits').exists():
        raise ValueError('Fit directory exists; preserve partial or completed fit')
    june = archive(output / 'june_inputs.npz')
    cc = config['calibration']
    fitted = {}
    if activation['I1']:
        predictions = _ordered((june['primary'], june['seed_primary']))
        fitted['I1'] = fit_class_bias_family(predictions=predictions, prediction_keys=june['keys'],
            labels=june['y'], keys=june['keys'], split_token='june_cpanel', config=cc)
    if activation['I2']:
        calibrated = _ordered((june['calibrated'], june['seed_calibrated']))
        weights = dict(zip(PREDICTORS, map(float, june['weights'])))
        fitted['I2'] = fit_volume_blend_family(calibrated=calibrated, prediction_keys=june['keys'],
            frequency=june['frequency'], frequency_keys=june['keys'], labels=june['y'],
            keys=june['keys'], groups=june['groups'], global_weights=weights,
            split_token='june_cpanel', config=cc, blend_optimizer=load_blend_optimizer(cc))
    failed = any(len(row['fits']) != 6 or any(
        fit['fit_success'] is not True or (slot == 'I1' and fit['optimizer_report']['scipy_success'] is not True)
        for fit in row['fits']) for slot, row in fitted.items())
    directory = output / 'fits'
    directory.mkdir()
    for slot, item in fitted.items():
        dump(directory / (slot + '.json'), item)
    dump(directory / 'manifest.json', {'activation_sha256': prep['activation_sha256'],
        'june_inputs_sha256': prep['june_inputs_sha256'],
        'files': {slot: hash_file(directory / (slot + '.json')) for slot in fitted},
        'inactive_slots': [slot for slot in SLOTS if not activation[slot]],
        'status': 'failed_preserved' if failed else 'sealed_before_dev_apply'})
    if failed:
        raise ValueError('Calibration fit failed; preserved parameters and manifest, no candidate application')


def verify_fits(output, activation):
    manifest = read_json(output / 'fits/manifest.json')
    if manifest['status'] != 'sealed_before_dev_apply' or set(manifest['files']) != {s for s in SLOTS if activation[s]} or \
            manifest['inactive_slots'] != [s for s in SLOTS if not activation[s]] or \
            manifest['activation_sha256'] != hash_file(output / 'activation.json') or \
            manifest['june_inputs_sha256'] != hash_file(output / 'june_inputs.npz'):
        raise ValueError('Fit manifest does not match sealed activation/June inputs')
    fits = {}
    for slot, digest in manifest['files'].items():
        path = output / 'fits' / (slot + '.json')
        if hash_file(path) != digest:
            raise ValueError('Frozen fit parameters changed')
        row = read_json(path)
        if row.get('predictors') != list(PREDICTORS) or len(row['fits']) != 6 or \
                [fit.get('predictor') for fit in row['fits']] != list(PREDICTORS) or any(
                    fit['fit_success'] is not True or (slot == 'I1' and fit['optimizer_report']['scipy_success'] is not True)
                    for fit in row['fits']):
            raise ValueError('Incomplete or permuted calibration fit')
        fits[slot] = row
    return manifest, fits


def _select_npz(path, names):
    with np.load(path, allow_pickle=False) as saved:
        if not set(names) <= set(saved.files):
            raise ValueError('Frozen archive misses required arrays')
        return {name: saved[name].copy() for name in names}


def _probabilities(values, n, name):
    p = np.asarray(values)
    if p.shape != (n, 10) or not np.issubdtype(p.dtype, np.floating) or \
            not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or \
            not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError(name + ' is not a valid ten-class simplex')
    return p


def _whole_inputs(config, *, include_labels):
    """Application excludes DEV labels. Scoring opens them only after all outputs seal."""
    stage2 = Path(config['stage2']['run']).resolve()
    g = Path(config['g']['run']).resolve()
    names = ('keys', 'game_pk', 'pitcher', 'primary', 'calibrated', 'raw', 'seed_primary')
    saved = _select_npz(stage2 / 'analysis/predictions.npz', (*names, 'y') if include_labels else names)
    baseline_names = ('mlb_dev_keys', 'mlb_dev_game_pk', 'mlb_dev_pitcher', 'mlb_dev')
    if include_labels:
        baseline_names += ('mlb_dev_y', 'dev_keys', 'dev_y', 'dev_game_pk', 'dev_pitcher')
    frequency = _select_npz(g / 'baseline_predictions.npz', baseline_names)
    metadata = pd.read_parquet(g / 'mlb_dev_metadata.parquet')
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), saved['keys']):
        raise ValueError('Whole-MLB metadata ordering differs')
    for name in ('keys', 'game_pk', 'pitcher'):
        if not np.array_equal(saved[name], frequency['mlb_dev_'+name]):
            raise ValueError('Stage2 archive rows differ from frozen G baseline: ' + name)
    if not np.array_equal(saved['keys'][:, 0], saved['game_pk']):
        raise ValueError('Pitch key/game alignment differs')
    n = len(saved['keys'])
    for name in ('primary', 'calibrated', 'raw'):
        _probabilities(saved[name], n, name)
    _probabilities(frequency['mlb_dev'], n, 'frequency')
    if saved['seed_primary'].shape != (5, n, 10):
        raise ValueError('Stage2 seed archive shape differs')
    for i, p in enumerate(saved['seed_primary']):
        _probabilities(p, n, 'seed'+str(i))
    panel_lookup = {int(row['pitcher']): row['train_volume'] for row in read_json(g / 'panel.json')['train_players']}
    groups = np.asarray([panel_lookup.get(int(pid), 'zero') for pid in saved['pitcher']])
    if not np.array_equal(groups, metadata['train_volume'].to_numpy()):
        raise ValueError('Whole-MLB TRAIN-volume grouping differs from original G panel')
    if include_labels:
        if not np.array_equal(saved['y'], frequency['mlb_dev_y']):
            raise ValueError('Stage2 labels differ from frozen G baseline')
        base = {'dev'+k[len('mlb_dev'):]: v for k, v in frequency.items() if k.startswith('mlb_dev')}
        panel = {k: v for k, v in frequency.items() if k.startswith('dev')}
        aligned_population(base, metadata, panel)
    return saved, frequency['mlb_dev'], metadata, groups


def _apply_slot(config, config_path, slot):
    output = Path(config['output']).resolve()
    _, activation = verify_prepared(config, config_path, output)
    if slot not in SLOTS or activation[slot] is not True:
        raise ValueError('Only an activated candidate may be applied')
    fit_manifest, fits = verify_fits(output, activation)
    destination = output / 'candidates' / slot
    if destination.exists():
        raise ValueError('Preserve existing candidate application')
    saved, frequency, _, groups = _whole_inputs(config, include_labels=False)
    cc = config['calibration']
    params = fits[slot]['fits']
    if slot == 'I1':
        primary = apply_class_bias(probabilities=saved['primary'], parameters=params[0], config=cc)
        seeds = np.stack([apply_class_bias(probabilities=p, parameters=params[i+1], config=cc)
                          for i, p in enumerate(saved['seed_primary'])])
    else:
        stage2 = Path(config['stage2']['run']).resolve()
        members = []
        for seed in range(5):
            item = _select_npz(stage2 / 'members/G0-global' / f'seed{seed}/predictions.npz',
                               ('dev_keys', 'dev_game_pk', 'dev_pitcher', 'dev'))
            for name, key in (('dev_keys', 'keys'), ('dev_game_pk', 'game_pk'), ('dev_pitcher', 'pitcher')):
                if not np.array_equal(item[name], saved[key]):
                    raise ValueError('Whole member order differs')
            _probabilities(item['dev'], len(saved['keys']), 'stage2 member')
            members.append(item['dev'])
        primary = apply_volume_blend(calibrated=saved['calibrated'], calibrated_keys=saved['keys'],
            frequency=frequency, frequency_keys=saved['keys'], keys=saved['keys'], groups=groups,
            parameters=params[0], config=cc)
        seeds = np.stack([apply_volume_blend(calibrated=p, calibrated_keys=saved['keys'],
            frequency=frequency, frequency_keys=saved['keys'], keys=saved['keys'], groups=groups,
            parameters=params[i+1], config=cc) for i, p in enumerate(members)])
    _probabilities(primary, len(saved['keys']), 'candidate primary')
    for i, p in enumerate(seeds):
        _probabilities(p, len(saved['keys']), 'candidate seed'+str(i))
    verify_prepared(config, config_path, output)
    verify_fits(output, activation)
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz', keys=saved['keys'], primary=primary,
                        seed_primary=seeds)
    dump(destination / 'manifest.json', {'slot': slot, 'predictions_sha256': hash_file(destination / 'predictions.npz'),
        'fit_manifest_sha256': hash_file(output / 'fits/manifest.json'),
        'fit_sha256': fit_manifest['files'][slot], 'activation_sha256': hash_file(output / 'activation.json')})


def apply(config, config_path):
    """One fixed queue command completes both slots, including explicit inactive markers."""
    output = Path(config['output']).resolve()
    _, activation = verify_prepared(config, config_path, output)
    verify_fits(output, activation)
    if (output / 'applications.json').exists() or (output / 'candidates').exists():
        raise ValueError('Preserve completed or interrupted candidate application')
    applications = {}
    for slot in SLOTS:
        if not activation[slot]:
            applications[slot] = {'status': 'not_activated', 'candidate_manifest_sha256': None}
            continue
        _apply_slot(config, config_path, slot)
        manifest = output / 'candidates' / slot / 'manifest.json'
        applications[slot] = {'status': 'applied', 'candidate_manifest_sha256': hash_file(manifest)}
    verify_prepared(config, config_path, output)
    verify_fits(output, activation)
    dump(output / 'applications.json', {'activation_sha256': hash_file(output / 'activation.json'),
        'fit_manifest_sha256': hash_file(output / 'fits/manifest.json'),
        'slots': applications, 'status': 'complete'})


def verify_applications(output, activation):
    record = read_json(output / 'applications.json')
    if record['status'] != 'complete' or record['activation_sha256'] != hash_file(output / 'activation.json') or \
            record['fit_manifest_sha256'] != hash_file(output / 'fits/manifest.json') or \
            list(record['slots']) != list(SLOTS):
        raise ValueError('Application inventory incomplete or changed')
    for slot in SLOTS:
        item = record['slots'][slot]
        folder = output / 'candidates' / slot
        if activation[slot]:
            if item['status'] != 'applied' or item['candidate_manifest_sha256'] != hash_file(folder / 'manifest.json'):
                raise ValueError('Active candidate application changed')
        elif item != {'status': 'not_activated', 'candidate_manifest_sha256': None} or folder.exists():
            raise ValueError('Inactive candidate must retain an explicit no-op slot')
    return record


def score(config, config_path):
    output = Path(config['output']).resolve()
    _, activation = verify_prepared(config, config_path, output)
    fit_manifest, _ = verify_fits(output, activation)
    verify_applications(output, activation)
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted stage3 score')
    saved, _, metadata, _ = _whole_inputs(config, include_labels=True)
    candidates = {}
    inputs = {str((output / 'preparation.json').resolve()): hash_file(output / 'preparation.json'),
              str((output / 'stage2_binding.json').resolve()): hash_file(output / 'stage2_binding.json'),
              str((output / 'activation.json').resolve()): hash_file(output / 'activation.json'),
              str((output / 'june_inputs.npz').resolve()): hash_file(output / 'june_inputs.npz'),
              str((output / 'fits/manifest.json').resolve()): hash_file(output / 'fits/manifest.json'),
              str((output / 'applications.json').resolve()): hash_file(output / 'applications.json')}
    for slot in SLOTS:
        folder = output / 'candidates' / slot
        if not activation[slot]:
            if folder.exists():
                raise ValueError('Inactive candidate output is forbidden')
            continue
        manifest_path = folder / 'manifest.json'
        manifest = read_json(manifest_path)
        prediction_path = folder / 'predictions.npz'
        if manifest['slot'] != slot or manifest['predictions_sha256'] != hash_file(prediction_path) or \
                manifest['fit_manifest_sha256'] != inputs[str((output / 'fits/manifest.json').resolve())] or \
                manifest['fit_sha256'] != fit_manifest['files'][slot] or \
                manifest['activation_sha256'] != hash_file(output / 'activation.json'):
            raise ValueError('Active candidate is incomplete or changed')
        item = archive(prediction_path)
        if not np.array_equal(item['keys'], saved['keys']):
            raise ValueError('Candidate keys differ')
        if item['seed_primary'].shape != (5, len(saved['y']), 10):
            raise ValueError('Candidate seed shape differs')
        candidates[slot] = {'primary': item['primary'], 'seed_primary': item['seed_primary']}
        inputs[str(manifest_path.resolve())] = hash_file(manifest_path)
        inputs[str(prediction_path.resolve())] = hash_file(prediction_path)
    result = evaluate_candidates(saved['y'], saved['game_pk'], metadata,
        {'primary': saved['primary'], 'seed_primary': saved['seed_primary']}, candidates, activation,
        draws=config['scoring']['bootstrap_draws'], r_draws=config['scoring']['R']['draws'])
    result.update(experiment_id=config['experiment_id'], registered_scoring=config['scoring'], stage2_binding_sha256=hash_file(output / 'stage2_binding.json'),
                  activation_sha256=hash_file(output / 'activation.json'),
                  fit_manifest_sha256=hash_file(output / 'fits/manifest.json'),
                  input_hashes=inputs, scored_utc=datetime.now(timezone.utc).isoformat())
    verify_prepared(config, config_path, output)
    _assert_hashes(inputs)
    destination.mkdir(parents=True)
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'inputs': inputs, 'source_hashes': config['sources']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--local-config', required=True, type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'fit', 'apply', 'score', 'status'):
        sub.add_parser(name)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    local = read_json(args.local_config)
    output = Path(config['output']).resolve()
    check_location(local, output)
    if args.command == 'status':
        _, active = verify_prepared(config, args.config, output)
        print(json.dumps({'experiment_id': config['experiment_id'], 'activation': active,
            'fits': (output/'fits/manifest.json').exists(), 'applications': (output/'applications.json').exists(),
            'analysis': (output/'analysis/manifest.json').exists()}, sort_keys=True))
        return
    with heavy_lock(check_location(local, output)):
        if args.command == 'prepare':
            prepare(config, args.config, args.local_config)
        elif args.command == 'fit':
            validate_native_runtime()
            fit(config, args.config)
        elif args.command == 'apply':
            validate_native_runtime()
            apply(config, args.config)
        elif args.command == 'score':
            validate_native_runtime()
            score(config, args.config)


if __name__ == '__main__':
    main()
