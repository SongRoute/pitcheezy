"""DRAFT family: periodic numerical embeddings for the direct pre-pitch head.

Additive runner on top of the sealed EXP-P15-001 direct head.  One factor
changes: justified continuous inputs (past-token physics, continuous context)
enter the unchanged flatten MLP head through fixed PLR-lite periodic embeddings
(pitchmdp/direct_periodic_model.py; Gorishniy et al. 2022 adaptation).
Categorical one-hots, the mask, game-state ordinals/flags and the zero
current-pitch physics stay raw.  Inputs, splits, optimizer/budget, unweighted
May early stop, May temperature, June blend and DEV are those of EXP-P15-001.
Controls: the frozen EXP-P15-001 direct head and the frozen fixed 1/2 mix of
that head with G0-global.

  run   --seeds 0 [--mlb] [--smoke] [--device auto|mps|cpu|cuda]
        prepare -> fit -> May temperature -> Cpanel blend/dev predict
        [-> whole-MLB 1-draw predict]; completed steps verified, partial steps refused.
  score after all three seeds: descriptive repeated-DEV comparison only,
        written under <output>/analysis/.  No N decision, no results file.

``--device auto`` keeps the control's rule (mps if available, else cpu).  An
explicit device is part of the identity; cpu/cuda (or auto resolving to cpu)
scores are not comparable to the MPS-trained control without a same-device control arm.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
from pitchmdp.data import KEY, hash_file
from pitchmdp.direct_periodic_model import ARCHITECTURE, DirectPeriodicModel, context_map, parameter_count
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_sharing import SharingPredictor, training_arrays
from pitchmdp.model import outcome_labels
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_delivery_pool import git_commit, load, members, peak, state_ok
import run_ml_direct_head as direct
import run_ml_direct_recency as recency  # reused: frozen-control check and descriptive scoring

SEEDS = (0, 1, 2)
PROTOCOL = 'ml_direct_periodic_v1'
SELF = 'scripts/run_ml_direct_periodic.py'
OWN_SOURCES = [SELF, 'pitchmdp/direct_periodic_model.py', 'pitchmdp/direct_residual_model.py',
               'scripts/run_ml_direct_recency.py']
BUDGET = {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005}
DEVICES = ('auto', 'mps', 'cpu', 'cuda')


def own_hashes():
    return {name: hash_file(PROJECT / name) for name in OWN_SOURCES}


def config_check(config, smoke=False):
    if config['protocol'] != PROTOCOL or config['seeds'] != list(SEEDS):
        raise ValueError('Periodic settings differ from the runner protocol')
    if config['control']['experiment_id'] != 'EXP-P15-001' or config['mix']['direct_weight'] != .5:
        raise ValueError('Controls are the frozen EXP-P15-001 direct head and its fixed 1/2 G0 mix')
    if config['control_training'] != {'kind': 'flatten_mlp', 'width': 128, **BUDGET}:
        raise ValueError('Control architecture/budget must equal EXP-P15-001')
    arch = config['candidate_architecture']
    if {k: arch[k] for k in ARCHITECTURE} != ARCHITECTURE or arch['training'] != BUDGET or arch['hpo_trials'] != 0:
        raise ValueError('Candidate architecture or budget differs from the fixed registered network')
    shape, columns = arch['input_contract'], arch['input_map']
    mapped = context_map(columns['pitcher_profile_columns'])
    if mapped['n_context'] != shape['n_context'] or columns['context_continuous'] != mapped['continuous'] \
            or columns['context_raw'] != mapped['raw']:
        raise ValueError('Registered continuous/raw context map differs from the profile layout')
    if arch['parameter_count'] != parameter_count(**shape, context_continuous=columns['context_continuous']):
        raise ValueError('Registered parameter count differs from the network')
    if not config.get('parent_preparation_sha256') or not config['control'].get('preparation_sha256'):
        raise ValueError('Parent and control preparation hashes must be pinned')
    if not smoke and (config['execution']['enabled'] is not True or config['status'] != 'REGISTERED'):
        raise ValueError('Execution not enabled in a registered config')
    return config


def identity(config, local_path, device='auto'):
    """Direct-head identity (shared input/split/predict path) plus this runner, model source, network and device."""
    if device not in DEVICES:
        raise ValueError('Unknown device option')
    return {**direct.identity(config, local_path), 'periodic_source_hashes': own_hashes(),
            'architecture': dict(ARCHITECTURE), 'input_map': config['candidate_architecture']['input_map'],
            'device_option': device,
            'resolved_device': DirectPeriodicModel(device=None if device == 'auto' else device).device}


def layout_check(config, pprep):
    """Frozen parent pitcher-profile columns must give exactly the registered continuous/raw context map."""
    columns = config['candidate_architecture']['input_map']
    if list(pprep['clusters'].get('columns', ())) != columns['pitcher_profile_columns']:
        raise ValueError('Parent pitcher-profile columns differ from the registered context map')
    return columns['context_continuous']


def sources_unchanged(prep):
    if direct.source_hashes() != prep['identity']['source_hashes'] or own_hashes() != prep['identity']['periodic_source_hashes']:
        raise ValueError('Sources changed during the periodic run')


def model_device(prep):
    option = prep['identity']['device_option']
    return None if option == 'auto' else option


# ------------------------------------------------------------------ run

def prepare(config, data, output, expected):
    if (output / 'preparation.json').exists():
        prep = read_json(output / 'preparation.json')
        if prep['identity'] != expected:
            raise ValueError('Periodic source/config/environment/device changed')
        assert_hashes(output, prep['artifact_hashes'])
        return prep
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete preparation requires failure review')
    started = time.perf_counter()
    _, pprep, store, _, _, _ = data
    layout_check(config, pprep)
    output.mkdir(parents=True, exist_ok=True)
    for rel in (*direct.SOURCES, *OWN_SOURCES):
        path = output / 'source' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)
    dump(output / 'registered_config.json', config)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    prep = {'identity': expected, 'parent_preparation_sha256': config['parent_preparation_sha256'],
            'control_preparation_sha256': config['control']['preparation_sha256'],
            'physical_channels_zeroed': int(store.physical.shape[1]),
            'comparable_device_to_control': expected['resolved_device'] == 'mps',
            'samples': dict(pprep['samples']), 'artifact_hashes': artifact_hashes(output, files),
            'seconds': time.perf_counter() - started, 'peak_rss_bytes': peak(), 'git_commit': git_commit(),
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False}
    dump(output / 'preparation.json', prep)
    print('PERIODIC_PREPARED', expected['device_option'], flush=True)
    return prep


def fit(config, data, output, prep, seed, smoke):
    dest = output / str(seed)
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke}
    if state_ok(dest, 'fit_state.json', expected):
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Incomplete fit requires failure review')
    _, pprep, store, context, parts, _ = data
    continuous = layout_check(config, pprep)
    train, early = parts['train'], parts['earlystop']
    if smoke:
        train, early = train.iloc[:direct.SMOKE['train']], early.iloc[:direct.SMOKE['earlystop']]
    arch = config['candidate_architecture']
    start = time.perf_counter()
    ta = training_arrays(direct.direct_arrays(store, context, train.index.to_numpy()))
    ea = training_arrays(direct.direct_arrays(store, context, early.index.to_numpy()))
    array_seconds = time.perf_counter() - start
    shape = {'n_context': int(ta[2].shape[1]), 'n_token': int(ta[0].shape[2]), 'length': int(ta[0].shape[1])}
    if shape != arch['input_contract']:
        raise ValueError(f'Input contract {shape} differs from the registered EXP-P15-001 inputs')
    model = DirectPeriodicModel(seed, continuous, device=model_device(prep)).fit(
        ta, outcome_labels(train), ea, outcome_labels(early),
        epochs=direct.SMOKE['epochs'] if smoke else BUDGET['epochs'], patience=BUDGET['patience'],
        batch_size=BUDGET['batch_size'], learning_rate=BUDGET['learning_rate'])
    if model.report['parameter_count'] != arch['parameter_count']:
        raise ValueError('Fitted parameter count differs from the registered count')
    dest.mkdir(parents=True, exist_ok=True)
    model.save(dest / 'model.pt')
    dump(dest / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter() - start,
         'array_seconds': array_seconds, 'context_width': shape['n_context'],
         'train_rows_sha256': ordered_key_hash(train), 'earlystop_rows_sha256': ordered_key_hash(early),
         'earlystop_metric': 'pre-pitch NLL (current physics zero), unweighted', 'peak_rss_bytes': peak()})
    sources_unchanged(prep)
    dump(dest / 'fit_state.json', {'identity': expected, 'artifact_hashes': artifact_hashes(dest, ['model.pt', 'fit.json'])})
    print('PERIODIC_FIT_COMPLETE', seed, model.report['device'], flush=True)
    del ta, ea, model


def predict(data, output, prep, seed, smoke, mlb=False):
    """Mirror of run_ml_direct_head.predict; only the model loader (DirectPeriodicModel) differs."""
    dest = output / str(seed)
    name = 'mlb_prediction_state.json' if mlb else 'prediction_state.json'
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke,
                'model_sha256': hash_file(dest / 'model.pt')}
    if state_ok(dest, name, expected):
        return
    prefix = 'mlb_' if mlb else ''
    artifacts = [prefix + 'predictions.npz', prefix + 'prediction_runtime.json'] + ([] if mlb else ['calibration.json'])
    if any((dest / rel).exists() for rel in artifacts):
        raise ValueError('Incomplete prediction artifacts require failure review before recalibration')
    _, pprep, store, context, parts, _ = data
    start = time.perf_counter()
    model = SharingPredictor(direct.CELL, DirectPeriodicModel.load(dest / 'model.pt', model_device(prep)), pprep['clusters'])
    delivery = direct.NoPhysicsDelivery(store.physical.shape[1])
    limit = (lambda part, n: part.iloc[:n]) if smoke else (lambda part, n: part)
    if mlb:
        if not state_ok(dest, 'prediction_state.json', expected):
            raise ValueError('Whole-MLB prediction requires the frozen Cpanel calibration')
        model.delivery_temperature = read_json(dest / 'calibration.json')['delivery_temperature']
    else:
        temp = limit(parts['temperature'], direct.SMOKE['temperature'])
        delivery.calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
        dump(dest / 'calibration.json', model.report)
    values = {}
    for split in (('mlb_dev',) if mlb else ('blend', 'dev')):
        part = limit(parts[split], direct.SMOKE['predict'])
        p, raw, levels = predict_streamed(model, delivery, store, context, part.index.to_numpy())
        if not np.all(levels == direct.NO_PHYSICS_LEVEL) or not np.allclose(p.sum(1), 1., atol=1e-8):
            raise ValueError('Direct prediction used a delivery pool or lost probability mass')
        values.update({split: p, split + '_raw': raw, split + '_delivery_level': levels,
            split + '_keys': part[KEY].to_numpy(np.int64), split + '_y': outcome_labels(part),
            split + '_game_pk': part.game_pk.to_numpy(np.int64), split + '_pitcher': part.pitcher.to_numpy(np.int64)})
    prefix = 'mlb_' if mlb else ''
    if (dest / (prefix + 'predictions.npz')).exists():
        raise ValueError('Incomplete prediction archive requires failure review')
    np.savez_compressed(dest / (prefix + 'predictions.npz'), **values)
    dump(dest / (prefix + 'prediction_runtime.json'), {'seconds': time.perf_counter() - start, 'peak_rss_bytes': peak(),
         'draws_per_pitch': 1, 'rows': {k: len(v) for k, v in values.items() if k.endswith('_y')}})
    sources_unchanged(prep)
    names = [prefix + 'predictions.npz', prefix + 'prediction_runtime.json'] + ([] if mlb else ['calibration.json'])
    dump(dest / name, {'identity': expected, 'artifact_hashes': artifact_hashes(dest, names)})
    print('PERIODIC_PREDICT_COMPLETE', seed, 'mlb' if mlb else 'panel', flush=True)


def run(config, local, output, expected, smoke, mlb, seeds):
    timings = {}
    before = time.perf_counter()
    data = load(local, config)
    timings['load'] = time.perf_counter() - before
    prep = prepare(config, data, output, expected)
    for seed in seeds:
        for step, call in (('fit', lambda: fit(config, data, output, prep, seed, smoke)),
                           ('predict', lambda: predict(data, output, prep, seed, smoke)),
                           ('mlb_predict', lambda: predict(data, output, prep, seed, smoke, mlb=True) if mlb else None)):
            before = time.perf_counter()
            call()
            timings[f'{step}_seed{seed}'] = time.perf_counter() - before
            sources_unchanged(prep)
    print('PERIODIC_RUN_COMPLETE', {'seconds': timings, 'peak_rss_bytes': peak()}, flush=True)
    return timings


# ------------------------------------------------------------------ descriptive scoring

def score(config, local_path, output, device='auto'):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted scoring output')
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != identity(config, local_path, device):
        raise ValueError('Periodic source/config/environment/device changed')
    for seed in SEEDS:
        if not (output / str(seed) / 'prediction_state.json').is_file():
            raise ValueError('Incomplete family: every seed must finish before any comparison')
    parent, _ = direct.verify_parent(config)
    control = recency.verify_control(config)
    started = time.perf_counter()
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as data:
        baseline = {k: data[k].copy() for k in data.files}
    candidate = members(output, '{seed}', 'prediction_state.json')
    uniform = members(control, '{seed}', 'prediction_state.json')
    g0 = members(parent, 'members/' + direct.CELL + '/seed{seed}', 'prediction_state.json')
    weight = config['mix']['direct_weight']
    result = {'cpanel_dev': recency.describe(candidate, uniform, g0, baseline, weight)}
    have_mlb = all((output / str(s) / 'mlb_prediction_state.json').is_file() for s in SEEDS)
    if have_mlb:
        whole = Path(config['whole_mlb_reference'])
        result['mlb_dev'] = recency.describe_mlb(members(output, '{seed}', 'mlb_prediction_state.json'),
                                                 members(control, '{seed}', 'mlb_prediction_state.json'),
                                                 members(whole, 'members/' + direct.CELL + '/seed{seed}', 'prediction_state.json'),
                                                 weight)
    fits = {str(s): read_json(output / str(s) / 'fit.json') for s in SEEDS}
    result = {'family': config['experiment_id'], 'status': config['status'], **result,
              'embedding': {'fixed_before_scores': dict(ARCHITECTURE), 'embedded_scalars': config['candidate_architecture']['n_embedded']},
              'controls': {'direct': 'frozen EXP-P15-001 flatten_mlp', 'mix': 'frozen EXP-P15-001 + G0-global, fixed 1/2'},
              'parameter_count': {'candidate': config['candidate_architecture']['parameter_count'],
                                  'control': config['control_parameter_count']},
              'costs_seconds': {s: {'fit': f['seconds_total'], 'best_epoch': f['report']['best_epoch'], 'device': f['report']['device'],
                                    'predict': read_json(output / s / 'prediction_runtime.json')['seconds']} for s, f in fits.items()},
              'comparable_device_to_control': prep['comparable_device_to_control'],
              'decision': None, 'git_commit': git_commit(), 'seconds': time.perf_counter() - started,
              'limits': ['Descriptive repeated-DEV comparison; Cpanel and whole-MLB DEV already exposed',
                         'Capacity and representation are confounded (2.02x parameters); no equal-parameter control',
                         'No N/G decision until the root registers rules', 'Prediction only; no policy claim; no 2026 rows'],
              'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir()
    dump(destination / 'descriptive.json', result)
    print('PERIODIC_DESCRIBED', {a: result['cpanel_dev'][a]['after_june_blend']['paired']['nll'] for a in ('direct', 'mix')}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=DEVICES, default='auto', help='auto = mps if available else cpu (control rule)')
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    r.add_argument('--mlb', action='store_true', help='also predict the frozen whole-MLB DEV (one draw)')
    r.add_argument('--smoke', action='store_true', help='row/epoch-limited wiring check; output must end -smoke')
    sub.add_parser('score')
    args = p.parse_args()
    smoke = getattr(args, 'smoke', False)
    config, local = config_check(read_json(args.config), smoke), read_json(args.local_config)
    output = args.output.resolve()
    if smoke != output.name.endswith('-smoke'):
        raise ValueError('Smoke runs write only to an output ending -smoke, and only smoke runs may')
    if not smoke and output.name != config['experiment_id']:
        raise ValueError('Output directory must be named after the experiment')
    root = check_location(local, output)
    validate_native_runtime()
    with heavy_lock(root):
        if args.command == 'run':
            recency.verify_control(config)
            run(config, local, output, identity(config, args.local_config, args.device), smoke, args.mlb, args.seeds)
        else:
            score(config, args.local_config, output, args.device)


if __name__ == '__main__':
    main()
