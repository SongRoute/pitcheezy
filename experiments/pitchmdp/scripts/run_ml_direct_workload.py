"""DRAFT family: observed-workload context scalars for the direct pre-pitch head (E4 / MX-F6).

Additive runner on top of the sealed EXP-P15-001 direct head.  One factor
changes: fixed workload scalars, rebuilt from keys/date/pitcher of ALL raw
2023-2025 Statcast records (before eligibility/cohort filters), are appended to
the numeric context.  Two pre-fixed variants (definitions fixed before any
outcome evaluation, no feature HPO):

  counts : prior_records_per100   = Statcast records of this pitcher earlier in this game / 100 (unclipped)
  block  : counts plus thruorder_minus1_over3 = (rebuilt times through order - 1) / 3,
           log_rest_days = log1p(clip(days since previous date, 0, 180)) / log 181 (missing -> 0),
           rest_missing  = 1 if no earlier date in the loaded 2023-2025 raw records

The count includes every Statcast record (automatic non-pitch events may be
present); it is observed workload, not physiological fatigue.  Every value uses
earlier keys of the same game or earlier dates only (same-day previous games are
omitted by definition, pitchmdp/workload_features.rebuild).  Savant workload
columns are never read here.  Inputs otherwise equal EXP-P15-001 (current-pitch
physics zero), as do model, budget, May early stop/temperature, June blend and DEV.
The extra scalars enter before the 7 sharing columns (5 cluster flags, cluster id,
routing id) that training_arrays strips, so the global head sees them and routing
is unchanged.  Controls: frozen EXP-P15-001 and its fixed 1/2 G0 mix (seeds 0-2).

  run   --variant counts --seeds 0 [--mlb] [--smoke] [--device auto|mps|cpu|cuda]
  score after every variant x seed is complete: descriptive repeated-DEV comparison
        under <output>/analysis/.  No N decision, no repository results file.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import SharingPredictor, training_arrays
from pitchmdp.model import outcome_labels
from pitchmdp.workload_features import rebuild
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_delivery_pool import git_commit, load, members, peak, state_ok
import run_ml_direct_head as direct
import run_ml_direct_recency as recency  # reused: frozen-control check and descriptive scoring

SEEDS = (0, 1, 2)
PROTOCOL = 'ml_direct_workload_v1'
SELF = 'scripts/run_ml_direct_workload.py'
OWN_SOURCES = [SELF, 'pitchmdp/workload_features.py', 'scripts/run_ml_direct_recency.py']
COLUMNS = ('prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing')
VARIANTS = {'counts': COLUMNS[:1], 'block': COLUMNS}
RAW_COLUMNS = ['game_date', *KEY, 'pitcher']
YEARS = (2023, 2024, 2025)
ALLOWLIST = [f'statcast_{y}.parquet' for y in YEARS]
REST_CAP = 180
SHARING_TAIL = 7  # 5 cluster flags, cluster id, routing id (stripped by training_arrays)
TRAINING = {'kind': 'flatten_mlp', 'width': 128, 'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005}
DEVICES = ('auto', 'mps', 'cpu', 'cuda')


# ------------------------------------------------------------------ features

def read_raw(local):
    """Keys/date/pitcher of every allowlisted 2023-2025 raw record; nothing else is opened."""
    if list(local['raw_allowlist']) != ALLOWLIST:
        raise ValueError('Raw allowlist must be exactly the 2023-2025 Statcast files')
    root, frames = Path(local['raw_root']), []
    for name, year in zip(ALLOWLIST, YEARS):
        part = pd.read_parquet(root / name, columns=RAW_COLUMNS)
        if not pd.to_datetime(part.game_date).dt.year.eq(year).all():
            raise ValueError(f'{name}: rows outside season {year}')
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def raw_hashes(local):
    root = Path(local['raw_root'])
    return {name: hash_file(root / name) for name in ALLOWLIST}


def workload_columns(raw, rows):
    """All four fixed scalars for `rows` (frozen parent rows, any order), keyed exactly to raw records."""
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError('Workload source must contain only keys, date and pitcher')
    if rows[KEY].isna().any().any() or rows.duplicated(KEY).any():
        raise ValueError('Parent keys must be complete and unique')
    built = pd.concat([raw, rebuild(raw)], axis=1)  # rebuild rejects missing/duplicate raw keys
    merged = rows[[*KEY, 'game_date', 'pitcher']].reset_index(drop=True).merge(
        built, on=KEY, how='left', validate='one_to_one', suffixes=('', '_raw'))
    if merged.prior_pitches_game.isna().any():
        raise ValueError('Parent rows without an exact raw Statcast key')
    same_date = pd.to_datetime(merged.game_date).dt.normalize().eq(pd.to_datetime(merged.game_date_raw).dt.normalize())
    if not (same_date.all() and merged.pitcher.astype(np.int64).eq(merged.pitcher_raw.astype(np.int64)).all()):
        raise ValueError('Parent and raw records disagree on date or pitcher')
    days = merged.rebuilt_days_since
    missing = days.isna()
    rest = np.log1p(days.clip(0, REST_CAP).fillna(0.)) / np.log(REST_CAP + 1)
    return np.column_stack([merged.prior_pitches_game.to_numpy(float) / 100.,
                            (merged.rebuilt_thruorder.to_numpy(float) - 1.) / 3.,
                            np.where(missing, 0., rest), missing.to_numpy(float)])


class WorkloadContext:
    """Base sharing context with fixed workload scalars inserted before the 7 sharing columns.

    `values` is aligned to the positions of the frozen store frame; every lookup
    re-checks the exact pitch keys, so rows may arrive in any order.
    """

    def __init__(self, base, store_frame, values):
        if not store_frame.index.equals(pd.RangeIndex(len(store_frame))) or len(values) != len(store_frame):
            raise ValueError('Workload values must align to the positional store frame')
        self.base, self.values = base, np.asarray(values, dtype=np.float32)
        self.keys = store_frame[KEY].to_numpy(np.int64)

    def transform(self, frame):
        out = self.base.transform(frame)
        if not self.values.shape[1]:
            return out  # zero extra columns: exactly the control path
        position = frame.index.to_numpy()
        if not np.array_equal(frame[KEY].to_numpy(np.int64), self.keys[position]):
            raise ValueError('Workload rows misaligned with the frozen store keys')
        if out.shape[1] <= SHARING_TAIL:
            raise ValueError('Sharing context requires cluster and routing metadata')
        return np.concatenate([out[:, :-SHARING_TAIL], self.values[position].astype(out.dtype),
                               out[:, -SHARING_TAIL:]], axis=1)


def feature_report(values, rows):
    """Input distribution on TRAIN rows only (no outcomes)."""
    report = {}
    for i, name in enumerate(COLUMNS):
        v = values[rows, i]
        report[name] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()),
                        'quantiles_0_50_99_100': [float(q) for q in np.quantile(v, [0, .5, .99, 1])]}
    report['rest_missing']['rate'] = float(values[rows, 3].mean())
    return report


def added_parameters(n_added, width=TRAINING['width']):
    """Flatten MLP: each context scalar adds one input to Linear(n_flat, 2*width)."""
    return n_added * 2 * width


# ------------------------------------------------------------------ config / identity

def own_hashes():
    return {name: hash_file(PROJECT / name) for name in OWN_SOURCES}


def config_check(config, smoke=False):
    if config['protocol'] != PROTOCOL or config['seeds'] != list(SEEDS):
        raise ValueError('Workload settings differ from the runner protocol')
    if {v['name']: tuple(v['columns']) for v in config['variants']} != VARIANTS:
        raise ValueError('Variant columns differ from the pre-fixed definitions')
    if any(v['added_parameters'] != added_parameters(len(v['columns'])) for v in config['variants']):
        raise ValueError('Registered added parameter counts differ from the network')
    if config['feature_rules']['raw_columns'] != RAW_COLUMNS or config['feature_rules']['rest_cap_days'] != REST_CAP \
            or config['feature_rules']['raw_allowlist'] != ALLOWLIST or config['feature_rules']['hpo_trials'] != 0:
        raise ValueError('Feature source or definitions differ from the runner')
    if config['control']['experiment_id'] != 'EXP-P15-001' or config['mix']['direct_weight'] != .5:
        raise ValueError('Controls are the frozen EXP-P15-001 direct head and its fixed 1/2 G0 mix')
    if config['unchanged_from_p15']['training'] != TRAINING or config['unchanged_from_p15'].get('loss_weighting') != 'uniform':
        raise ValueError('Architecture/budget/uniform loss must equal EXP-P15-001')
    if not config.get('parent_preparation_sha256') or not config['control'].get('preparation_sha256'):
        raise ValueError('Parent and control preparation hashes must be pinned')
    if not smoke and (config['execution']['enabled'] is not True or config['status'] != 'REGISTERED'):
        raise ValueError('Execution not enabled in a registered config')
    return config


def identity(config, local_path, variant, device='auto'):
    """Direct-head identity plus this runner, feature layout, raw-file hashes and device."""
    if device not in DEVICES:
        raise ValueError('Unknown device option')
    return {**direct.identity(config, local_path), 'workload_source_hashes': own_hashes(),
            'variant': variant, 'feature_layout': {'columns': list(VARIANTS[variant]),
            'insert': f'before the last {SHARING_TAIL} sharing columns', 'raw_columns': RAW_COLUMNS,
            'rest_cap_days': REST_CAP, 'dtype': 'float32'},
            'raw_file_sha256': raw_hashes(read_json(local_path)), 'device_option': device,
            'resolved_device': MatrixModel(device=None if device == 'auto' else device).device}


def sources_unchanged(prep):
    if direct.source_hashes() != prep['identity']['source_hashes'] or own_hashes() != prep['identity']['workload_source_hashes']:
        raise ValueError('Sources changed during the workload run')


def model_device(prep):
    option = prep['identity']['device_option']
    return None if option == 'auto' else option


def workload_data(local, data, variant):
    """Swap the sharing context for the workload wrapper; also return all four values for reports."""
    parent, pprep, store, context, parts, aux = data
    years = pd.to_datetime(store.frame.game_date).dt.year
    if years.isna().any() or not years.isin(YEARS).all():
        raise ValueError('Frozen rows outside 2023-2025')
    values = workload_columns(read_raw(local), store.frame)
    keep = [COLUMNS.index(c) for c in VARIANTS[variant]]
    return (parent, pprep, store, WorkloadContext(context, store.frame, values[:, keep]), parts, aux), values


# ------------------------------------------------------------------ run

def prepare(config, data, values, output, expected):
    selected = [COLUMNS.index(c) for c in expected['feature_layout']['columns']]
    feature_sha = hashlib.sha256(np.ascontiguousarray(values[:, selected], dtype=np.float32).tobytes()).hexdigest()
    if (output / 'preparation.json').exists():
        prep = read_json(output / 'preparation.json')
        if prep['identity'] != expected:
            raise ValueError('Workload source/config/environment/variant/device changed')
        if prep['feature_sha256'] != feature_sha:
            raise ValueError('Rebuilt workload features differ from the prepared ones')
        assert_hashes(output, prep['artifact_hashes'])
        return prep
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete preparation requires failure review')
    started = time.perf_counter()
    _, pprep, store, _, parts, _ = data
    output.mkdir(parents=True, exist_ok=True)
    for rel in (*direct.SOURCES, *OWN_SOURCES):
        path = output / 'source' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)
    dump(output / 'registered_config.json', config)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    n_added = len(selected)
    prep = {'identity': expected, 'parent_preparation_sha256': config['parent_preparation_sha256'],
            'control_preparation_sha256': config['control']['preparation_sha256'],
            'physical_channels_zeroed': int(store.physical.shape[1]), 'feature_sha256': feature_sha,
            'added_context_width': n_added, 'added_parameters': added_parameters(n_added),
            'train_feature_report': feature_report(values, parts['train'].index.to_numpy()),
            'train_rows_sha256': ordered_key_hash(parts['train']), 'samples': dict(pprep['samples']),
            'comparable_device_to_control': expected['resolved_device'] == 'mps',
            'artifact_hashes': artifact_hashes(output, files), 'seconds': time.perf_counter() - started,
            'peak_rss_bytes': peak(), 'git_commit': git_commit(),
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False}
    dump(output / 'preparation.json', prep)
    print('WORKLOAD_PREPARED', expected['variant'], expected['device_option'], flush=True)
    return prep


def fit(config, data, output, prep, seed, smoke):
    dest = output / str(seed)
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke}
    if state_ok(dest, 'fit_state.json', expected):
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Incomplete fit requires failure review')
    _, _, store, context, parts, _ = data
    train, early = parts['train'], parts['earlystop']
    if smoke:
        train, early = train.iloc[:direct.SMOKE['train']], early.iloc[:direct.SMOKE['earlystop']]
    spec = config['unchanged_from_p15']['training']
    start = time.perf_counter()
    ta = training_arrays(direct.direct_arrays(store, context, train.index.to_numpy()))
    ea = training_arrays(direct.direct_arrays(store, context, early.index.to_numpy()))
    array_seconds = time.perf_counter() - start
    model = MatrixModel(spec['kind'], seed=seed, width=spec['width'], device=model_device(prep)).fit(
        ta, outcome_labels(train), ea, outcome_labels(early),
        epochs=direct.SMOKE['epochs'] if smoke else spec['epochs'], patience=spec['patience'],
        batch_size=spec['batch_size'], learning_rate=spec['learning_rate'])
    dest.mkdir(parents=True, exist_ok=True)
    model.save(dest / 'model.pt')
    dump(dest / 'fit.json', {'report': model.report, 'device': model.device, 'seconds_total': time.perf_counter() - start,
         'array_seconds': array_seconds, 'context_width': int(ta[2].shape[1]),
         'parameter_count': int(sum(p.numel() for p in model.net.parameters())),
         'added_parameters': prep['added_parameters'],
         'train_rows_sha256': ordered_key_hash(train), 'earlystop_rows_sha256': ordered_key_hash(early),
         'peak_rss_bytes': peak()})
    sources_unchanged(prep)
    dump(dest / 'fit_state.json', {'identity': expected, 'artifact_hashes': artifact_hashes(dest, ['model.pt', 'fit.json'])})
    print('WORKLOAD_FIT_COMPLETE', prep['identity']['variant'], seed, model.device, flush=True)
    del ta, ea, model


def predict(data, output, prep, seed, smoke, mlb=False):
    """Mirror of run_ml_direct_head.predict with the workload context and partial-artifact refusal."""
    dest = output / str(seed)
    name = 'mlb_prediction_state.json' if mlb else 'prediction_state.json'
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke,
                'model_sha256': hash_file(dest / 'model.pt')}
    if state_ok(dest, name, expected):
        return
    prefix = 'mlb_' if mlb else ''
    names = [prefix + 'predictions.npz', prefix + 'prediction_runtime.json'] + ([] if mlb else ['calibration.json'])
    if any((dest / rel).exists() for rel in names):
        raise ValueError('Incomplete prediction artifacts require failure review before recalibration')
    _, pprep, store, context, parts, _ = data
    start = time.perf_counter()
    model = SharingPredictor(direct.CELL, MatrixModel.load(dest / 'model.pt', model_device(prep)), pprep['clusters'])
    delivery = direct.NoPhysicsDelivery(store.physical.shape[1])
    limit = (lambda part, n: part.iloc[:n]) if smoke else (lambda part, n: part)
    if mlb:
        if not state_ok(dest, 'prediction_state.json', expected):
            raise ValueError('Whole-MLB prediction requires the frozen Cpanel calibration')
        model.delivery_temperature = read_json(dest / 'calibration.json')['delivery_temperature']
    else:
        temp = limit(parts['temperature'], direct.SMOKE['temperature'])
        delivery.calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
    values = {}
    for split in (('mlb_dev',) if mlb else ('blend', 'dev')):
        part = limit(parts[split], direct.SMOKE['predict'])
        p, raw, levels = predict_streamed(model, delivery, store, context, part.index.to_numpy())
        if not np.all(levels == direct.NO_PHYSICS_LEVEL) or not np.allclose(p.sum(1), 1., atol=1e-8):
            raise ValueError('Direct prediction used a delivery pool or lost probability mass')
        values.update({split: p, split + '_raw': raw, split + '_delivery_level': levels,
            split + '_keys': part[KEY].to_numpy(np.int64), split + '_y': outcome_labels(part),
            split + '_game_pk': part.game_pk.to_numpy(np.int64), split + '_pitcher': part.pitcher.to_numpy(np.int64)})
    if not mlb:
        dump(dest / 'calibration.json', model.report)
    np.savez_compressed(dest / (prefix + 'predictions.npz'), **values)
    dump(dest / (prefix + 'prediction_runtime.json'), {'seconds': time.perf_counter() - start, 'peak_rss_bytes': peak(),
         'draws_per_pitch': 1, 'rows': {k: len(v) for k, v in values.items() if k.endswith('_y')}})
    sources_unchanged(prep)
    dump(dest / name, {'identity': expected, 'artifact_hashes': artifact_hashes(dest, names)})
    print('WORKLOAD_PREDICT_COMPLETE', prep['identity']['variant'], seed, 'mlb' if mlb else 'panel', flush=True)


def run(config, local, output, expected, smoke, mlb, seeds):
    timings = {}
    before = time.perf_counter()
    data, values = workload_data(local, load(local, config), expected['variant'])
    timings['load'] = time.perf_counter() - before
    prep = prepare(config, data, values, output, expected)
    for seed in seeds:
        for step, call in (('fit', lambda: fit(config, data, output, prep, seed, smoke)),
                           ('predict', lambda: predict(data, output, prep, seed, smoke)),
                           ('mlb_predict', lambda: predict(data, output, prep, seed, smoke, mlb=True) if mlb else None)):
            before = time.perf_counter()
            call()
            timings[f'{step}_seed{seed}'] = time.perf_counter() - before
            sources_unchanged(prep)
    print('WORKLOAD_RUN_COMPLETE', {'variant': expected['variant'], 'seconds': timings, 'peak_rss_bytes': peak()}, flush=True)
    return timings


# ------------------------------------------------------------------ descriptive scoring

def score(config, local_path, output, device='auto'):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted scoring output')
    for variant in VARIANTS:
        prep = read_json(output / variant / 'preparation.json')
        if prep['identity'] != identity(config, local_path, variant, device):
            raise ValueError(f'{variant}: source/config/environment/device changed')
        for seed in SEEDS:
            if not (output / variant / str(seed) / 'prediction_state.json').is_file():
                raise ValueError('Incomplete family: every variant and seed must finish before any comparison')
    parent, _ = direct.verify_parent(config)
    control = recency.verify_control(config)
    started = time.perf_counter()
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as data:
        baseline = {k: data[k].copy() for k in data.files}
    uniform = members(control, '{seed}', 'prediction_state.json')
    g0 = members(parent, 'members/' + direct.CELL + '/seed{seed}', 'prediction_state.json')
    have_mlb = all((output / v / str(s) / 'mlb_prediction_state.json').is_file() for v in VARIANTS for s in SEEDS)
    if have_mlb:
        uniform_mlb = members(control, '{seed}', 'mlb_prediction_state.json')
        g0_mlb = members(Path(config['whole_mlb_reference']), 'members/' + direct.CELL + '/seed{seed}', 'prediction_state.json')
    weight, variants = config['mix']['direct_weight'], {}
    for variant in VARIANTS:
        candidate = members(output / variant, '{seed}', 'prediction_state.json')
        prep = read_json(output / variant / 'preparation.json')
        fits = {str(s): read_json(output / variant / str(s) / 'fit.json') for s in SEEDS}
        variants[variant] = {'cpanel_dev': recency.describe(candidate, uniform, g0, baseline, weight),
                             'added_context_width': prep['added_context_width'], 'added_parameters': prep['added_parameters'],
                             'parameter_count': {s: f['parameter_count'] for s, f in fits.items()},
                             'train_feature_report': prep['train_feature_report'],
                             'comparable_device_to_control': prep['comparable_device_to_control'],
                             'costs_seconds': {s: {'fit': f['seconds_total'], 'best_epoch': f['report']['best_epoch'],
                                                   'device': f['device']} for s, f in fits.items()}}
        if have_mlb:
            variants[variant]['mlb_dev'] = recency.describe_mlb(members(output / variant, '{seed}', 'mlb_prediction_state.json'),
                                                                uniform_mlb, g0_mlb, weight)
    result = {'family': config['experiment_id'], 'status': config['status'], 'variants': variants,
              'controls': {'direct': 'frozen EXP-P15-001 (no workload inputs)', 'mix': 'frozen EXP-P15-001 + G0-global, fixed 1/2'},
              'decision': None, 'git_commit': git_commit(), 'seconds': time.perf_counter() - started,
              'limits': ['Descriptive repeated-DEV comparison; Cpanel and whole-MLB DEV already exposed',
                         'Observed Statcast record counts and rest, not physiological fatigue',
                         'No N/G decision and no winner selection until the root registers rules',
                         'Prediction only; no policy claim; no 2026 rows'],
              'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir()
    dump(destination / 'descriptive.json', result)
    print('WORKLOAD_DESCRIBED', {v: {a: r['cpanel_dev'][a]['after_june_blend']['paired']['nll'] for a in ('direct', 'mix')}
                                 for v, r in variants.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='family directory; variants are sub-directories')
    p.add_argument('--device', choices=DEVICES, default='auto', help='auto = mps if available else cpu (control rule)')
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--variant', choices=list(VARIANTS), required=True)
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
        raise ValueError('Output directory must be named after the experiment family')
    root = check_location(local, output)
    validate_native_runtime()
    with heavy_lock(root):
        if args.command == 'run':
            recency.verify_control(config)
            run(config, local, output / args.variant, identity(config, args.local_config, args.variant, args.device),
                smoke, args.mlb, args.seeds)
        else:
            score(config, args.local_config, output, args.device)


if __name__ == '__main__':
    main()
