"""F1 batter-representation bridge runner: prepare, profile, fit and predict only.

The full arm reuses the preserved G0-global members of the frozen G run
(EXP-P4-001) byte for byte. Only three masked members are new: context
``[11:28]`` is zero in training, early stopping, May temperature calibration
and inference. Normalizer, batter statistics, pitcher representation,
frequency baseline and delivery are copied from the parent, never refitted.
No DEV score is computed here; scoring is ``score_ml_bridge.py``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_bridge import (BATTER_START, BATTER_STOP, CONTEXT_WIDTH, ROUTING_WIDTH,
                                    batter_train_volume, check_context_layout, fit_masked,
                                    masked_g0_predictor, masked_training_arrays, network_signature)
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.matrix_sharing import SharingPredictor
from pitchmdp.model import outcome_labels
from run_ml_benchmark import read_json, dump, identity as base_identity, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_sharing import (SOURCES as SHARING_SOURCES, identity as sharing_identity,
                            verify as verify_sharing, load_data as sharing_load_data)
from run_sequence_pilot import arrays
from score_ml_matrix import archive, assert_aligned, summarize_cell

SEEDS = (0, 1, 2)
PARTS = ('train', 'earlystop', 'temperature', 'blend', 'dev')
PARENT_EXPERIMENT = 'EXP-P4-001'
PARENT_CELL = 'G0-global'
EXPECTED_SAMPLES = {'train': 1252824, 'earlystop': 16000, 'temperature': 2603,
                    'blend': 4821, 'dev': 12334, 'dev_games': 328}
BUDGET = {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}
PROFILE = {'train_rows': 65536, 'epochs': 2, 'earlystop_rows': 2048, 'temperature_rows': 64}
LIMITS = {'profile_wall_limit_seconds': 600, 'member_wall_limit_seconds': 7200,
          'family_wall_budget_seconds': 14400}
MASK = {'start': BATTER_START, 'stop': BATTER_STOP}
BOOTSTRAP = {'draws': 10000, 'seed': 20260924, 'unit': 'whole game',
             'estimand': 'pitch-weighted paired mean loss'}
DECISION = {'delta_nll_max': -.003, 'nll_ci95_upper_max': 0, 'one_sided_p_max': .05,
            'brier_ci95_upper_max': .001, 'required_negative_seeds': 2}
ROBUSTNESS = {'groups': 12, 'metrics': 2, 'comparisons': 1, 'family_size': 24, 'family_alpha': .05,
              'draws': 100000, 'seed': 20260924, 'minimum_games': 30, 'minimum_pitches': 500,
              'nll_margin': .01, 'brier_margin': .002}
BATTER_VOLUME = {'quantile': .25, 'method': 'linear', 'boundary': 'low_inclusive',
                 'scope': 'D100 TRAIN positive batter pitch counts'}
FULL_PROBE_ROWS = 64
FULL_PROBE_ATOL = 1e-6
SOURCES = list(dict.fromkeys([*SHARING_SOURCES, 'pitchmdp/matrix_metrics.py',
                             'scripts/run_sequence_calibration.py', 'scripts/score_ml_matrix.py',
                             'pitchmdp/matrix_group_metrics.py', 'pitchmdp/matrix_bridge.py',
                             'scripts/run_ml_bridge.py', 'scripts/score_ml_bridge.py']))
FIXED = {'protocol': 'ml_bridge_v1', 'parent_experiment_id': PARENT_EXPERIMENT,
         'parent_cell': PARENT_CELL, 'scope': 'Cpanel', 'seeds': list(SEEDS), 'draws': 400,
         'kind': 'flatten_mlp', 'width': 128, 'budget': BUDGET, 'mask': MASK,
         'context_width': CONTEXT_WIDTH, 'routing_width': ROUTING_WIDTH,
         'expected_samples': EXPECTED_SAMPLES, 'profile': PROFILE, 'limits': LIMITS,
         'bootstrap': BOOTSTRAP, 'decision': DECISION, 'robustness': ROBUSTNESS,
         'batter_volume': BATTER_VOLUME, 'device': 'auto'}
REQUIRED = {*FIXED, 'experiment_id', 'parent_run', 'parent_preparation_sha256',
            'parent_analysis_sha256'}


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def config_check(config):
    if not isinstance(config, dict) or not REQUIRED <= set(config) or set(config) - REQUIRED - {'registration'}:
        raise ValueError('Invalid F1 bridge configuration schema')
    for name, value in FIXED.items():
        if config[name] != value:
            raise ValueError('F1 bridge setting differs from registered protocol: ' + name)
    for name in ('parent_preparation_sha256', 'parent_analysis_sha256'):
        if not _hex(config[name]):
            raise ValueError('Exact frozen G preparation and analysis SHA256 required: ' + name)
    for name in ('experiment_id', 'parent_run'):
        if not isinstance(config[name], str) or not config[name].strip() or 'ROOT_TO_ASSIGN' in config[name]:
            raise ValueError('Registered run identity required: ' + name)
    if 'registration' in config and not isinstance(config['registration'], dict):
        raise ValueError('Registration must be an object')
    return config


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    return {**base_identity(config, local_path), 'source_hashes': source_hashes()}


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('F1 bridge source/config/environment changed from preparation')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Frozen G dependency changed: ' + name)
    return prep


# ---------------------------------------------------------------- ledger

STAGE_CAPS = {'prepare': LIMITS['member_wall_limit_seconds'], 'profile': LIMITS['profile_wall_limit_seconds'],
              'fit': LIMITS['member_wall_limit_seconds'], 'predict': LIMITS['member_wall_limit_seconds'],
              'score': LIMITS['member_wall_limit_seconds']}
_ACTIVE = {}


def ledger_entries(output):
    path = output / 'ledger.jsonl'
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ledger_total(output, *, exclude=None):
    """Actual new spending, including failed, interrupted and killed attempts.

    A ``start`` without an ``end`` (SIGKILL, OOM, power loss) is charged
    conservatively: the wall time until the next ledger start, capped at the
    stage's registered caller timeout (or the cap if no later start exists).
    ``exclude`` omits the currently running command.
    """
    entries = ledger_entries(output)
    ended = {e['id']: e for e in entries if e['event'] == 'end'}
    starts = [e for e in entries if e['event'] == 'start' and e['id'] != exclude]
    total = 0.
    for i, entry in enumerate(starts):
        if entry['id'] in ended:
            total += ended[entry['id']]['seconds']
            continue
        cap = STAGE_CAPS[entry['stage']]
        later = [e['unix'] for e in starts[i + 1:]]
        total += min(cap, later[0] - entry['unix']) if later else cap
    return float(total)


def ledger_attempts(output):
    entries = ledger_entries(output)
    ended = {e['id']: e for e in entries if e['event'] == 'end'}
    return [{**e, 'status': ended[e['id']]['status'] if e['id'] in ended else 'unterminated',
             'seconds': ended[e['id']]['seconds'] if e['id'] in ended else None,
             'error': ended.get(e['id'], {}).get('error')}
            for e in entries if e['event'] == 'start']


def _append_ledger(output, entry):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def active_elapsed(output):
    """Seconds already spent by the running command (including verification)."""
    active = _ACTIVE.get(str(output))
    return (time.perf_counter() - active['clock'], active['id']) if active else (0., None)


@contextmanager
def ledger_stage(output, stage, seed=None):
    if stage not in STAGE_CAPS:
        raise ValueError('Unregistered ledger stage')
    identifier = uuid.uuid4().hex
    clock = time.perf_counter()
    _append_ledger(output, {'event': 'start', 'id': identifier, 'stage': stage, 'seed': seed,
                            'unix': time.time(), 'utc': datetime.now(timezone.utc).isoformat()})
    _ACTIVE[str(output)] = {'id': identifier, 'clock': clock}
    try:
        yield
    except BaseException as error:
        _append_ledger(output, {'event': 'end', 'id': identifier, 'status': 'failed',
                                'seconds': time.perf_counter() - clock,
                                'error': f'{type(error).__name__}: {error}',
                                'utc': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        _ACTIVE.pop(str(output), None)
    _append_ledger(output, {'event': 'end', 'id': identifier, 'status': 'completed',
                            'seconds': time.perf_counter() - clock,
                            'utc': datetime.now(timezone.utc).isoformat()})


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


# ---------------------------------------------------------------- full-arm reuse

def _normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def verify_full_reconstruction(members, baseline, stored, stored_report):
    """Recompute G0 June blends from preserved members; reject any changed value.

    ``stored`` is the frozen G panel analysis archive and ``stored_report`` its
    ``reports['G0-global']``. Exact equality is required because the same
    members, baseline and frozen numerical environment are used.
    """
    if len(members) != len(SEEDS):
        raise ValueError('Three preserved G0 members required')
    for member in members:
        assert_aligned(member, baseline)
    for name in ('keys', 'y', 'game_pk', 'pitcher'):
        if not np.array_equal(stored[name], baseline['dev_' + name]):
            raise ValueError('Preserved G0 analysis metadata differs: ' + name)
    report, values = summarize_cell(members, baseline)
    for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
        name = PARENT_CELL + '_' + kind
        if name not in stored or stored[name].shape != values[kind].shape or not np.array_equal(stored[name], values[kind]):
            raise ValueError('Preserved G0 prediction reconstruction differs: ' + kind)
    if _normalized(report['selection']) != _normalized(stored_report['selection']):
        raise ValueError('Preserved G0 June ensemble weight differs')
    if len(stored_report['seeds']) != len(members):
        raise ValueError('Preserved G0 seed reports incomplete')
    for mine, frozen in zip(report['seeds'], stored_report['seeds']):
        if _normalized(mine['blend_selection']) != _normalized(frozen['blend_selection']):
            raise ValueError('Preserved G0 per-seed June weight differs')
    for kind in ('primary', 'calibrated_ensemble', 'raw_ensemble'):
        for metric in ('log_loss', 'brier_multiclass'):
            if report[kind][metric] != stored_report[kind][metric]:
                raise ValueError('Preserved G0 metric reconstruction differs: ' + kind)
    return report, values


def full_member_paths(parent, seed):
    return {'fit_dir': parent / 'fits' / f'seed{seed}' / 'global',
            'member_dir': parent / 'members' / PARENT_CELL / f'seed{seed}'}


def _verify_full_members(parent, shared):
    """Identity/hash checks of the preserved G0 fits and predictions."""
    external, reuse = {}, []
    spec = shared['units']['global']
    if spec['mode'] != 'global':
        raise ValueError('Frozen G global unit changed')
    for seed in SEEDS:
        paths = full_member_paths(parent, seed)
        fit_dir, member = paths['fit_dir'], paths['member_dir']
        state = read_json(fit_dir / 'state.json')
        if state['identity'] != {'preparation_sha256': canonical_hash(shared), 'seed': seed,
                                 'unit': 'global', 'spec': spec}:
            raise ValueError('Preserved G0 fit identity differs')
        assert_hashes(fit_dir, state['artifact_hashes'])
        prediction = read_json(member / 'prediction_state.json')
        if prediction['identity'] != {'preparation_sha256': canonical_hash(shared), 'cell': PARENT_CELL, 'seed': seed}:
            raise ValueError('Preserved G0 prediction identity differs')
        assert_hashes(member, prediction['artifact_hashes'])
        model_path = str(fit_dir / 'model.pt')
        if prediction['dependencies'] != {model_path: hash_file(fit_dir / 'model.pt')}:
            raise ValueError('Preserved G0 prediction depends on another model')
        for folder, names in ((fit_dir, ['state.json', *state['artifact_hashes']]),
                              (member, ['prediction_state.json', *prediction['artifact_hashes']])):
            for name in names:
                external[str(folder / name)] = hash_file(folder / name)
        calibration = read_json(member / 'calibration.json')
        fitted = read_json(fit_dir / 'fit.json')
        reuse.append({'seed': seed, 'fit_dir': str(fit_dir), 'member_dir': str(member),
                      'delivery_temperature': calibration['delivery_temperature'],
                      'logical_fit_seconds': fitted['seconds_total'],
                      'device': fitted['report']['device'],
                      'logical_prediction_seconds': read_json(member / 'prediction_runtime.json')['seconds']})
    return external, reuse


# ---------------------------------------------------------------- prepare

def prepare(config, local_path, local, output, expected):
    if (output / 'preparation.json').exists():
        verify(output, expected)
        print('BRIDGE_PREPARED', output, flush=True)
        return
    if output.exists() and any(p.name != 'ledger.jsonl' and not is_appledouble(p) for p in output.iterdir()):
        raise ValueError('Incomplete F1 preparation requires failure review')
    _prepare(config, local_path, local, output, expected)
    print('BRIDGE_PREPARED', output, flush=True)


def _prepare(config, local_path, local, output, expected):
    started = time.perf_counter()
    parent = Path(config['parent_run']).resolve()
    check_location(local, parent)
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('F1 output must be a distinct sibling of the G run')
    shared_config = read_json(parent / 'registered_config.json')
    if shared_config.get('experiment_id') != PARENT_EXPERIMENT:
        raise ValueError('F1 parent must be the registered G run')
    shared = verify_sharing(parent, sharing_identity(shared_config, local_path))
    analysis = parent / 'analysis' / 'panel'
    manifest = read_json(analysis / 'manifest.json')
    if (hash_file(parent / 'preparation.json') != config['parent_preparation_sha256']
            or hash_file(analysis / 'results.json') != config['parent_analysis_sha256']
            or manifest['results_sha256'] != config['parent_analysis_sha256']
            or hash_file(analysis / 'predictions.npz') != manifest['predictions_sha256']):
        raise ValueError('Frozen G preparation/analysis identity changed')
    external = {**manifest['inputs'], str(analysis / 'results.json'): config['parent_analysis_sha256'],
                str(analysis / 'manifest.json'): hash_file(analysis / 'manifest.json'),
                str(analysis / 'predictions.npz'): manifest['predictions_sha256'],
                str(parent / 'preparation.json'): config['parent_preparation_sha256'],
                str(parent / 'registered_config.json'): hash_file(parent / 'registered_config.json')}
    member_hashes, reuse = _verify_full_members(parent, shared)
    for name, digest in member_hashes.items():
        if name in external and external[name] != digest:
            raise ValueError('G analysis manifest and preserved member disagree: ' + name)
    external.update(member_hashes)
    for name, digest in external.items():
        if hash_file(Path(name)) != digest:
            raise ValueError('G dependency changed before F1 preparation: ' + name)
    records = {}
    for name in PARTS:
        record = shared['samples'][name]
        if record['n'] != EXPECTED_SAMPLES[name]:
            raise ValueError('Frozen G sample size differs from F1 registration: ' + name)
        records[name] = record
    if records['dev']['games'] != EXPECTED_SAMPLES['dev_games']:
        raise ValueError('Frozen Cpanel DEV game count differs from F1 registration')
    baseline = archive(parent / 'baseline_predictions.npz')
    assert_aligned(baseline, baseline)
    members = [archive(Path(item['member_dir']) / 'predictions.npz') for item in reuse]
    stored = archive(analysis / 'predictions.npz')
    frozen = read_json(analysis / 'results.json')
    full_report, _ = verify_full_reconstruction(members, baseline, stored, frozen['reports'][PARENT_CELL])
    output.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    dump(output / 'registered_config.json', config)
    dump(output / 'sharing_preparation.json', shared)
    for src, dst in [('parent_preparation.json', 'parent_preparation.json'), ('aux.pkl', 'aux.pkl'),
                     ('baseline_predictions.npz', 'baseline_predictions.npz'),
                     ('dev_metadata.parquet', 'dev_metadata.parquet'),
                     *[(records[name]['path'], records[name]['path']) for name in PARTS]]:
        shutil.copyfile(parent / src, output / dst)
        if hash_file(output / dst) != hash_file(parent / src):
            raise ValueError('Copied frozen G artifact differs: ' + src)
    view = {'features': shared['features'], 'clusters': shared['clusters'], 'samples': records}
    store, context, parts, aux = sharing_load_data(local, output, view)
    for name in PARTS:
        if ordered_key_hash(parts[name]) != records[name]['rows_sha256']:
            raise ValueError('Rebuilt F1 sample differs from frozen G: ' + name)
    layout = check_context_layout(aux['context'], context, parts['train'].iloc[:16])
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    if (not np.array_equal(metadata[KEY].to_numpy(np.int64), parts['dev'][KEY].to_numpy(np.int64))
            or not np.array_equal(metadata.batter.to_numpy(np.int64), parts['dev'].batter.to_numpy(np.int64))):
        raise ValueError('Frozen DEV metadata and batter identities are unpaired')
    volume = batter_train_volume(parts['train'].batter.to_numpy(), quantile=BATTER_VOLUME['quantile'])
    dump(output / 'batter_train_volume.json', volume)
    del store, context, parts, aux
    if source_hashes() != expected['source_hashes']:
        raise ValueError('Sources changed during F1 preparation')
    files = [str(p.relative_to(output)) for p in output.rglob('*')
             if p.is_file() and p.name != 'ledger.jsonl' and not is_appledouble(p)]
    prep = {'identity': expected, 'parent_run': str(parent), 'external_hashes': external,
            'samples': records, 'features': shared['features'], 'clusters': shared['clusters'],
            'panel': shared['panel'],
            'full_reuse': {'cell': PARENT_CELL, 'members': reuse,
                           'june_selection': full_report['selection'],
                           'june_seed_selection': [s['blend_selection'] for s in full_report['seeds']],
                           'reconstruction': 'exact equality of raw/calibrated/primary/seed_primary and weights'},
            'context_layout': layout,
            'mask': {'channels': [BATTER_START, BATTER_STOP], 'width': CONTEXT_WIDTH,
                     'routing_width': ROUTING_WIDTH,
                     'retained': {'[0:11]': 'game state and handedness', '[28:52]': 'pitcher representation and TRAIN count'},
                     'scope': 'training, early stopping, calibration and inference'},
            'refit': {'normalizer': False, 'batter_statistics': False, 'pitcher_representation': False,
                      'frequency_baseline': False, 'delivery': False,
                      'masked_new_fits': ['network', 'May delivery temperature', 'June seed/ensemble blend']},
            'batter_volume': {k: v for k, v in volume.items() if k != 'counts'},
            'artifact_hashes': artifact_hashes(output, files),
            'seconds': time.perf_counter() - started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'prepared_utc': datetime.now(timezone.utc).isoformat(),
            'masked_dev_scores_computed': False,
            'full_scores_read': 'frozen public G0 analysis only, for exact reconstruction'}
    dump(output / 'preparation.json', prep)


def load_data(local, output, prep, parts=PARTS):
    view = {'features': prep['features'], 'clusters': prep['clusters'],
            'samples': {name: prep['samples'][name] for name in parts}}
    return sharing_load_data(local, output, view)


def _new_model(config, seed):
    # Device is pinned to 'auto' (the G0 default) by config_check and verified
    # against the preserved G0 device before any fit or prediction.
    return MatrixModel(config['kind'], seed=seed, width=config['width'])


def _require_parent_device(prep, seed):
    item = prep['full_reuse']['members'][SEEDS.index(seed)]
    current = MatrixModel().device
    if item['seed'] != seed or item['device'] != current:
        raise ValueError(f'F1 device {current} differs from preserved G0 device {item["device"]}')
    return current


# ---------------------------------------------------------------- profile

def project_costs(measured, samples, prepare_seconds):
    """Linear extrapolation from the fresh profile; a planning estimate, not a bound."""
    rows = PROFILE['train_rows'] + PROFILE['earlystop_rows']
    per_epoch = measured['fit_seconds'] / PROFILE['epochs']
    fit = per_epoch * (samples['train'] + samples['earlystop']) / rows * BUDGET['epochs']
    calibrate = measured['calibration_seconds'] / PROFILE['temperature_rows'] * samples['temperature']
    per_row = measured['inference_seconds'] / PROFILE['temperature_rows']
    predict = per_row * (samples['blend'] + samples['dev'] + FULL_PROBE_ROWS)
    # Every command pays startup/verification overhead and a data load
    # (the profile loads only TRAIN/early/May rows).
    overhead = measured['command_overhead_seconds'] + measured['load_seconds']
    fit_command = overhead + fit
    predict_command = overhead + calibrate + predict
    member = fit_command + predict_command
    family = prepare_seconds + measured['wall_seconds'] + len(SEEDS) * member
    return {'fit_command_seconds': fit_command, 'predict_command_seconds': predict_command,
            'member_seconds': member, 'family_seconds': family,
            'member_gate': member <= LIMITS['member_wall_limit_seconds'],
            'family_gate': family <= LIMITS['family_wall_budget_seconds'],
            'note': ('Linear 30-epoch, all CAL/DEV row and per-command overhead/load extrapolation; '
                     'profile load excludes blend/DEV rows; caller enforces timeouts.')}


def profile(config, local, output, prep):
    dest = output / 'profile'
    if (dest / 'state.json').exists():
        state = read_json(dest / 'state.json')
        if state['preparation_sha256'] != hash_file(output / 'preparation.json'):
            raise ValueError('F1 profile parent changed')
        assert_hashes(dest, state['artifact_hashes'])
        print('BRIDGE_PROFILE_COMPLETE', flush=True)
        return
    if dest.exists() and any(not is_appledouble(p) for p in dest.iterdir()):
        raise ValueError('Incomplete F1 profile requires failure review')
    overhead, _ = active_elapsed(output)
    start = time.perf_counter()
    # Profile never selects blend or DEV rows.
    store, context, parts, aux = load_data(local, output, prep, parts=('train', 'earlystop', 'temperature'))
    load_seconds = time.perf_counter() - start
    train = parts['train'].iloc[:PROFILE['train_rows']]
    early = parts['earlystop'].iloc[:PROFILE['earlystop_rows']]
    query = parts['temperature'].iloc[:PROFILE['temperature_rows']]
    before = time.perf_counter()
    model = fit_masked(_new_model(config, 0), arrays(store, context, train.index.to_numpy()),
                       outcome_labels(train), arrays(store, context, early.index.to_numpy()),
                       outcome_labels(early), epochs=PROFILE['epochs'], patience=PROFILE['epochs'],
                       batch_size=BUDGET['batch_size'], learning_rate=BUDGET['learning_rate'])
    fit_seconds = time.perf_counter() - before
    predictor = masked_g0_predictor(model, prep['clusters'])
    before = time.perf_counter()
    aux['delivery'].calibrate(predictor, store, context, query.index.to_numpy(), outcome_labels(query))
    calibration_seconds = time.perf_counter() - before
    before = time.perf_counter()
    p, _, _ = predict_streamed(predictor, aux['delivery'], store, context, query.index.to_numpy())
    inference_seconds = time.perf_counter() - before
    wall = time.perf_counter() - start + overhead
    if wall > LIMITS['profile_wall_limit_seconds']:
        raise TimeoutError('F1 real-data profile exceeded registered 600-second limit')
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during F1 profile')
    measured = {'command_overhead_seconds': overhead, 'load_seconds': load_seconds, 'fit_seconds': fit_seconds,
                'calibration_seconds': calibration_seconds, 'inference_seconds': inference_seconds,
                'wall_seconds': wall}
    prepare_seconds = sum(a['seconds'] if a['seconds'] is not None else STAGE_CAPS['prepare']
                          for a in ledger_attempts(output) if a['stage'] == 'prepare')
    result = {**PROFILE, 'draws': config['draws'], 'device': model.device, 'measured': measured,
              'projection': project_costs(measured, {k: prep['samples'][k]['n'] for k in PARTS}, prepare_seconds),
              'maximum_mass_error': float(np.abs(p.sum(1) - 1).max()),
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'blend_rows_selected': False, 'dev_rows_selected': False, 'dev_scores_read': False,
              'profiled_utc': datetime.now(timezone.utc).isoformat()}
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest / 'profile.json', result)
    dump(dest / 'state.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
         'artifact_hashes': artifact_hashes(dest, ['profile.json'])})
    print('BRIDGE_PROFILE_COMPLETE', result['projection'], flush=True)


def _profile_projection(output):
    state_path = output / 'profile' / 'state.json'
    if not state_path.is_file():
        raise ValueError('Registered F1 profile must complete before any full fit')
    state = read_json(state_path)
    if state['preparation_sha256'] != hash_file(output / 'preparation.json'):
        raise ValueError('Matching F1 resource profile required')
    assert_hashes(output / 'profile', state['artifact_hashes'])
    projection = read_json(output / 'profile' / 'profile.json')['projection']
    if projection['member_gate'] is not True or projection['family_gate'] is not True:
        raise ValueError('F1 projected cost exceeds 7200-second member or 14400-second family budget')
    return projection


def check_family_budget(output, upcoming):
    """Ledger (excluding this command) + this command's elapsed + projection."""
    elapsed, active = active_elapsed(output)
    spent = ledger_total(output, exclude=active) + elapsed
    if spent + upcoming > LIMITS['family_wall_budget_seconds']:
        raise ValueError(f'F1 family ledger {spent:.0f}s plus projected {upcoming:.0f}s exceeds 14400s')
    return spent


# ---------------------------------------------------------------- fit / predict

def member_dir(output, seed):
    if seed not in SEEDS:
        raise ValueError('Unregistered F1 seed')
    return output / 'members' / 'masked' / f'seed{seed}'


def member_identity(prep, seed):
    return {'preparation_sha256': canonical_hash(prep), 'arm': 'masked', 'seed': seed,
            'mask': [BATTER_START, BATTER_STOP], 'train_rows_sha256': prep['samples']['train']['rows_sha256']}


def _parent_model(prep, seed):
    item = prep['full_reuse']['members'][SEEDS.index(seed)]
    if item['seed'] != seed:
        raise ValueError('Preserved G0 member order changed')
    return MatrixModel.load(Path(item['fit_dir']) / 'model.pt', device='cpu')


def check_input_shape(train_arrays, network):
    """Before fitting: masked inputs must build exactly the preserved G0 network."""
    tokens, _, context = train_arrays
    planned = {'kind': 'flatten_mlp', 'n_context': int(context.shape[1]), 'n_token': int(tokens.shape[2]),
               'length': int(tokens.shape[1]), 'width': 128, 'n_classes': 10}
    if planned != dict(network):
        raise ValueError(f'Masked input contract {planned} differs from preserved G0 {dict(network)}')
    return planned


def fit(config, local, output, prep, seed):
    projection = _profile_projection(output)
    dest = member_dir(output, seed)
    statepath = dest / 'fit_state.json'
    membership = member_identity(prep, seed)
    if statepath.exists():
        state = read_json(statepath)
        if state['identity'] != membership:
            raise ValueError('F1 completed fit identity changed')
        assert_hashes(dest, state['artifact_hashes'])
        print('BRIDGE_FIT_COMPLETE', seed, flush=True)
        return
    if dest.exists() and any(not is_appledouble(p) for p in dest.iterdir()):
        raise ValueError('Incomplete F1 member requires failure review')
    device = _require_parent_device(prep, seed)
    check_family_budget(output, projection['member_seconds'])
    parent_signature = network_signature(_parent_model(prep, seed))
    start = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep, parts=('train', 'earlystop'))
    train, early = parts['train'], parts['earlystop']
    train_arrays = arrays(store, context, train.index.to_numpy())
    early_arrays = arrays(store, context, early.index.to_numpy())
    check_input_shape(masked_training_arrays(early_arrays), parent_signature['network'])
    dest.mkdir(parents=True, exist_ok=True)
    model = fit_masked(_new_model(config, seed), train_arrays, outcome_labels(train),
                       early_arrays, outcome_labels(early), **BUDGET, checkpoint=dest / 'best_training.pt')
    signature = network_signature(model)
    if signature != parent_signature or model.device != device:
        raise ValueError('Masked network shape/parameter count/device differs from preserved G0')
    model.save(dest / 'model.pt')
    dump(dest / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter() - start,
         'command_seconds_total': active_elapsed(output)[0],
         'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'train_rows_sha256': ordered_key_hash(train), 'earlystop_rows_sha256': ordered_key_hash(early),
         'mask': [BATTER_START, BATTER_STOP], 'network_signature': signature,
         'matches_preserved_g0_signature': True,
         'checkpoint_note': 'best_training.pt is the best early-stop state written at fit end'})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during F1 fit')
    dump(statepath, {'identity': membership,
         'artifact_hashes': artifact_hashes(dest, ['model.pt', 'best_training.pt', 'fit.json']),
         'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('BRIDGE_FIT_COMPLETE', seed, flush=True)


def full_path_probe(prep, seed, store, context, aux, blend):
    """Recompute the first preserved G0 blend chunk through the shared wrapper."""
    item = prep['full_reuse']['members'][SEEDS.index(seed)]
    model = MatrixModel.load(Path(item['fit_dir']) / 'model.pt', device=item['device'])
    predictor = SharingPredictor(PARENT_CELL, model, prep['clusters'])
    predictor.delivery_temperature = item['delivery_temperature']
    rows = blend.index.to_numpy()[:FULL_PROBE_ROWS]
    probability, _, _ = predict_streamed(predictor, aux['delivery'], store, context, rows)
    with np.load(Path(item['member_dir']) / 'predictions.npz', allow_pickle=False) as frozen:
        preserved = frozen['blend'][:FULL_PROBE_ROWS]
        if not np.array_equal(frozen['blend_keys'][:FULL_PROBE_ROWS], blend[KEY].to_numpy(np.int64)[:FULL_PROBE_ROWS]):
            raise ValueError('Full-path probe rows differ from preserved G0 blend keys')
    difference = float(np.abs(probability - preserved).max())
    if not difference <= FULL_PROBE_ATOL:
        raise ValueError(f'Shared G0 numerical path no longer reproduces preserved predictions ({difference:.3g})')
    return {'rows': len(rows), 'maximum_absolute_difference': difference, 'tolerance': FULL_PROBE_ATOL}


def predict(config, local, output, prep, seed):
    projection = _profile_projection(output)
    dest = member_dir(output, seed)
    fitted = read_json(dest / 'fit_state.json')
    if fitted['identity'] != member_identity(prep, seed):
        raise ValueError('F1 fit identity changed')
    assert_hashes(dest, fitted['artifact_hashes'])
    statepath = dest / 'prediction_state.json'
    if statepath.exists():
        state = read_json(statepath)
        if state['identity'] != member_identity(prep, seed) or state['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
            raise ValueError('F1 prediction differs from fitted checkpoint')
        assert_hashes(dest, state['artifact_hashes'])
        print('BRIDGE_PREDICT_COMPLETE', seed, flush=True)
        return
    if any((dest / name).exists() for name in ('predictions.npz', 'calibration.json', 'prediction_runtime.json')):
        raise ValueError('Uncommitted F1 predictions require failure review')
    device = _require_parent_device(prep, seed)
    fit_seconds = read_json(dest / 'fit.json')['command_seconds_total']
    if fit_seconds + projection['predict_command_seconds'] > LIMITS['member_wall_limit_seconds']:
        raise ValueError('F1 member fit+predict projection exceeds 7200 seconds')
    check_family_budget(output, projection['predict_command_seconds'])
    start = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep, parts=('temperature', 'blend', 'dev'))
    inner = MatrixModel.load(dest / 'model.pt', device=device)
    if inner.kind != config['kind'] or inner.seed != seed:
        raise ValueError('F1 checkpoint architecture/seed changed')
    probe = full_path_probe(prep, seed, store, context, aux, parts['blend'])
    predictor = masked_g0_predictor(inner, prep['clusters'])
    temperature = parts['temperature']
    aux['delivery'].calibrate(predictor, store, context, temperature.index.to_numpy(), outcome_labels(temperature))
    dump(dest / 'calibration.json', predictor.report)
    baseline = archive(output / 'baseline_predictions.npz')
    values, counts = {}, {}
    for split in ('blend', 'dev'):
        part = parts[split]
        p, raw, levels = predict_streamed(predictor, aux['delivery'], store, context, part.index.to_numpy())
        values.update({split: p, split + '_raw': raw, split + '_delivery_level': levels,
                       split + '_keys': part[KEY].to_numpy(np.int64), split + '_y': outcome_labels(part),
                       split + '_game_pk': part.game_pk.to_numpy(np.int64),
                       split + '_pitcher': part.pitcher.to_numpy(np.int64)})
        unique, n = np.unique(levels, return_counts=True)
        counts[split] = {str(int(k)): int(v) for k, v in zip(unique, n)}
    assert_aligned(values, baseline)
    np.savez_compressed(dest / 'predictions.npz', **values)
    dump(dest / 'prediction_runtime.json', {'seconds': time.perf_counter() - start,
         'command_seconds_total': active_elapsed(output)[0], 'device': device,
         'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'delivery_tier_counts': counts, 'full_path_probe': probe, 'dev_scored': False})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during F1 prediction')
    dump(statepath, {'identity': member_identity(prep, seed),
         'fit_state_sha256': hash_file(dest / 'fit_state.json'),
         'artifact_hashes': artifact_hashes(dest, ['predictions.npz', 'prediction_runtime.json', 'calibration.json']),
         'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('BRIDGE_PREDICT_COMPLETE', seed, flush=True)


def status(output, expected):
    verify(output, expected)
    spent = ledger_total(output)
    print('ledger_seconds', round(spent, 1), 'family_remaining', round(LIMITS['family_wall_budget_seconds'] - spent, 1))
    print('profile', 'complete' if (output / 'profile' / 'state.json').exists() else 'pending')
    for seed in SEEDS:
        folder = member_dir(output, seed)
        state = ('predicted' if (folder / 'prediction_state.json').exists()
                 else 'fitted' if (folder / 'fit_state.json').exists() else 'pending')
        remaining = ''
        if state == 'fitted':
            used = read_json(folder / 'fit.json')['command_seconds_total']
            remaining = ' predict_timeout_remaining %.0f' % (LIMITS['member_wall_limit_seconds'] - used)
        print(seed, state + remaining)


def check_registered_output(config, output):
    registered = config.get('registration', {}).get('output')
    if registered is not None and Path(registered).resolve() != output:
        raise ValueError('Output differs from registration.output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'profile', 'status'):
        commands.add_parser(name)
    for name in ('fit', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    config, local = config_check(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve()
    check_registered_output(config, output)
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, args.local_config)
    if args.command == 'status':
        status(output, expected)
        return
    signal.signal(signal.SIGTERM, _terminate)
    with heavy_lock(root):
        # The ledger covers verification, loading and every failure mode.
        with ledger_stage(output, args.command, getattr(args, 'seed', None)):
            if args.command == 'prepare':
                prepare(config, args.local_config, local, output, expected)
            else:
                prep = verify(output, expected)
                if args.command == 'profile':
                    profile(config, local, output, prep)
                elif args.command == 'fit':
                    fit(config, local, output, prep, args.seed)
                else:
                    predict(config, local, output, prep, args.seed)


if __name__ == '__main__':
    main()
