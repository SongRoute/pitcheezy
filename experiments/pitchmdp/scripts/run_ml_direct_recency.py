"""DRAFT family: recency-weighted TRAIN loss for the direct pre-pitch head (matrix row MX-D4).

Additive runner on top of the sealed EXP-P15-001 direct head.  One factor
changes: each TRAIN pitch gets loss weight exp2(-age / half_life), age in days
before the fixed TRAIN anchor 2025-04-30, normalised to mean one.  Weights
depend only on game_date (never on outcomes).  Model, inputs (current-pitch
physics zero), splits, budget, May early stop, May temperature, June blend and
DEV are those of EXP-P15-001; early stopping stays unweighted.  The half-life
grid is a pre-fixed family of variants, each compared only with its fixed
controls: the frozen uniform EXP-P15-001 direct head and the frozen fixed 1/2
mix of that head with G0-global (G15 mix, seeds 0-2).

  run   --variant hl365 --seeds 0 [--mlb] [--smoke]
        prepare -> fit -> May temperature -> Cpanel blend/dev predict
        [-> whole-MLB 1-draw predict]; per variant+seed state, resumable.
  score after every registered variant x seed is complete: descriptive repeated-
        DEV comparisons only, written under <output>/analysis/.  No N decision,
        no repository results file, no policy claim.

Outputs: <output>/<variant>/{preparation.json, source/, <seed>/...}.
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
import pandas as pd
from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import training_arrays
from pitchmdp.model import outcome_labels
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_delivery_pool import git_commit, load, members, peak, state_ok
import run_ml_direct_head as direct

SEEDS = (0, 1, 2)
VARIANTS = {'hl180': 180, 'hl365': 365, 'hl730': 730}
ANCHOR = '2025-04-30'
PROTOCOL = 'ml_direct_recency_v1'
SELF = 'scripts/run_ml_direct_recency.py'


# ------------------------------------------------------------------ weights

def recency_weights(dates, anchor=ANCHOR, half_life_days=365):
    """exp2(-age/half_life), age in whole days before the anchor, normalised to mean one."""
    if isinstance(half_life_days, bool) or not isinstance(half_life_days, (int, float)) \
            or not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError('Half-life must be a positive finite number of days')
    dates = pd.to_datetime(pd.Series(dates), errors='coerce')
    if dates.isna().any():
        raise ValueError('Missing or unparseable TRAIN dates')
    age = (pd.Timestamp(anchor) - dates.dt.normalize()).dt.days.to_numpy(dtype=float)
    if (age < 0).any():
        raise ValueError('TRAIN rows after the recency anchor are forbidden')
    weight = np.exp2(-age / float(half_life_days))
    return weight / weight.mean()


def weight_report(dates, weight):
    years = pd.to_datetime(pd.Series(dates)).dt.year.to_numpy()
    total = weight.sum()
    return {'rows': int(len(weight)), 'mean': float(weight.mean()), 'min': float(weight.min()),
            'max': float(weight.max()), 'ess': float(total ** 2 / np.square(weight).sum()),
            'ess_ratio': float(total ** 2 / np.square(weight).sum() / len(weight)),
            'share_by_year': {str(y): float(weight[years == y].sum() / total) for y in np.unique(years)},
            'rows_by_year': {str(y): int((years == y).sum()) for y in np.unique(years)}}


# ------------------------------------------------------------------ config / identity

def config_check(config, smoke=False):
    if config['protocol'] != PROTOCOL or config['seeds'] != list(SEEDS):
        raise ValueError('Recency settings differ from the runner protocol')
    if {v['name']: v['half_life_days'] for v in config['variants']} != VARIANTS:
        raise ValueError('Half-life grid differs from the pre-fixed family')
    if config['anchor'] != ANCHOR or config['weight_rule'] != 'exp2(-age_days/half_life_days), mean 1, TRAIN only':
        raise ValueError('Recency anchor or weight rule differs from the runner')
    if config['control']['experiment_id'] != 'EXP-P15-001' or config['mix']['direct_weight'] != .5:
        raise ValueError('Controls are the frozen EXP-P15-001 direct head and its fixed 1/2 G0 mix')
    if config['unchanged_from_p15']['training'] != {'kind': 'flatten_mlp', 'width': 128, 'epochs': 30, 'patience': 5,
                                                     'batch_size': 1024, 'learning_rate': 0.0005}:
        raise ValueError('Architecture/budget must equal EXP-P15-001')
    if not config.get('parent_preparation_sha256') or not config['control'].get('preparation_sha256'):
        raise ValueError('Parent and control preparation hashes must be pinned')
    if not smoke and (config['execution']['enabled'] is not True or config['status'] != 'REGISTERED'):
        raise ValueError('Execution not enabled in a registered config')
    return config


def verify_control(config):
    """Frozen uniform control: pinned preparation, same direct-head source as now."""
    control = Path(config['control']['run'])
    if hash_file(control / 'preparation.json') != config['control']['preparation_sha256']:
        raise ValueError('Frozen EXP-P15-001 control preparation changed')
    prep = read_json(control / 'preparation.json')
    if prep['identity']['source_hashes'] != direct.source_hashes():
        raise ValueError('Direct-head source differs from the frozen control; the candidate would not share its path')
    assert_hashes(control, prep['artifact_hashes'])
    return control


def identity(config, local_path, variant):
    """Direct-head identity (its source_hashes, checked inside direct.predict) plus this runner and variant."""
    return {**direct.identity(config, local_path), 'recency_source_sha256': hash_file(PROJECT / SELF),
            'variant': variant, 'half_life_days': VARIANTS[variant], 'anchor': ANCHOR}


def sources_unchanged(prep):
    if direct.source_hashes() != prep['identity']['source_hashes'] or \
            hash_file(PROJECT / SELF) != prep['identity']['recency_source_sha256']:
        raise ValueError('Sources changed during the recency run')


# ------------------------------------------------------------------ run

def prepare(config, data, output, expected):
    if (output / 'preparation.json').exists():
        prep = read_json(output / 'preparation.json')
        if prep['identity'] != expected:
            raise ValueError('Recency source/config/environment/variant changed')
        assert_hashes(output, prep['artifact_hashes'])
        return prep
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete preparation requires failure review')
    started = time.perf_counter()
    _, pprep, store, _, parts, _ = data
    train = parts['train']
    weight = recency_weights(train.game_date, half_life_days=expected['half_life_days'])
    output.mkdir(parents=True, exist_ok=True)
    for rel in (*direct.SOURCES, SELF):
        path = output / 'source' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)
    dump(output / 'registered_config.json', config)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    prep = {'identity': expected, 'parent_preparation_sha256': config['parent_preparation_sha256'],
            'control_preparation_sha256': config['control']['preparation_sha256'],
            'physical_channels_zeroed': int(store.physical.shape[1]),
            'train_rows_sha256': ordered_key_hash(train), 'train_weights': weight_report(train.game_date, weight),
            'samples': dict(pprep['samples']), 'artifact_hashes': artifact_hashes(output, files),
            'seconds': time.perf_counter() - started, 'peak_rss_bytes': peak(), 'git_commit': git_commit(),
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False}
    dump(output / 'preparation.json', prep)
    print('RECENCY_PREPARED', expected['variant'], prep['train_weights']['ess_ratio'], flush=True)
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
    weight = recency_weights(train.game_date, half_life_days=prep['identity']['half_life_days'])
    spec = config['unchanged_from_p15']['training']
    start = time.perf_counter()
    ta = training_arrays(direct.direct_arrays(store, context, train.index.to_numpy()))
    ea = training_arrays(direct.direct_arrays(store, context, early.index.to_numpy()))
    array_seconds = time.perf_counter() - start
    model = MatrixModel(spec['kind'], seed=seed, width=spec['width']).fit(
        ta, outcome_labels(train), ea, outcome_labels(early),
        epochs=direct.SMOKE['epochs'] if smoke else spec['epochs'], patience=spec['patience'],
        batch_size=spec['batch_size'], learning_rate=spec['learning_rate'], sample_weight=weight)
    dest.mkdir(parents=True, exist_ok=True)
    model.save(dest / 'model.pt')
    dump(dest / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter() - start,
         'array_seconds': array_seconds, 'context_width': int(ta[2].shape[1]),
         'train_rows_sha256': ordered_key_hash(train), 'earlystop_rows_sha256': ordered_key_hash(early),
         'train_weights': weight_report(train.game_date, weight), 'earlystop_weighting': 'uniform',
         'peak_rss_bytes': peak()})
    sources_unchanged(prep)
    dump(dest / 'fit_state.json', {'identity': expected, 'artifact_hashes': artifact_hashes(dest, ['model.pt', 'fit.json'])})
    print('RECENCY_FIT_COMPLETE', prep['identity']['variant'], seed, flush=True)
    del ta, ea, model


def run(config, local, output, expected, smoke, mlb, seeds):
    timings = {}
    before = time.perf_counter()
    data = load(local, config)
    timings['load'] = time.perf_counter() - before
    prep = prepare(config, data, output, expected)
    for seed in seeds:
        for step, call in (('fit', lambda: fit(config, data, output, prep, seed, smoke)),
                           ('predict', lambda: direct.predict(data, output, prep, seed, smoke)),
                           ('mlb_predict', lambda: direct.predict(data, output, prep, seed, smoke, mlb=True) if mlb else None)):
            before = time.perf_counter()
            call()
            timings[f'{step}_seed{seed}'] = time.perf_counter() - before
            sources_unchanged(prep)
    print('RECENCY_RUN_COMPLETE', {'variant': expected['variant'], 'seconds': timings, 'peak_rss_bytes': peak()}, flush=True)
    return timings


# ------------------------------------------------------------------ descriptive scoring

def describe(candidate, uniform, g0, baseline, weight):
    """Candidate direct/mix vs the matching fixed control on Cpanel DEV after the June blend (descriptive)."""
    from score_ml_matrix import assert_aligned, summarize_cell
    for member in (*candidate, *uniform, *g0):
        assert_aligned(member, baseline)
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    arms = {'direct': (candidate, uniform),
            'mix': (direct.mixed(candidate, g0, weight), direct.mixed(uniform, g0, weight))}
    result = {}
    for arm, (a, b) in arms.items():
        rep_a, sa = summarize_cell(a, baseline)
        rep_b, sb = summarize_cell(b, baseline)
        result[arm] = {'after_june_blend': direct.paired(y, games, sa['primary'], sb['primary'], sa['seed_primary'], sb['seed_primary']),
                       'before_blend': direct.paired(y, games, sa['calibrated'], sb['calibrated'],
                                                     [m['dev'] for m in a], [m['dev'] for m in b]),
                       'candidate_report': rep_a, 'control_report': rep_b}
    return result


def describe_mlb(candidate, uniform, g0_mlb, weight):
    y, games, keys = g0_mlb[0]['dev_y'], g0_mlb[0]['dev_game_pk'], g0_mlb[0]['dev_keys']
    for m in (*candidate, *uniform):
        if not (np.array_equal(m['mlb_dev_keys'], keys) and np.array_equal(m['mlb_dev_y'], y)):
            raise ValueError('Unpaired whole-MLB predictions')
    a, b, g = [m['mlb_dev'] for m in candidate], [m['mlb_dev'] for m in uniform], [m['dev'] for m in g0_mlb]
    am, bm = [weight * x + (1 - weight) * z for x, z in zip(a, g)], [weight * x + (1 - weight) * z for x, z in zip(b, g)]
    return {'direct': direct.paired(y, games, np.mean(a, 0), np.mean(b, 0), a, b),
            'mix': direct.paired(y, games, np.mean(am, 0), np.mean(bm, 0), am, bm),
            'scope': 'calibrated three-seed mean, frozen May temperature, no June blend'}


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    destination = output / 'analysis'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted scoring output')
    for variant in VARIANTS:
        prep = read_json(output / variant / 'preparation.json')
        if prep['identity'] != identity(config, local_path, variant):
            raise ValueError(f'{variant}: source/config/environment changed')
        for seed in SEEDS:
            if not (output / variant / str(seed) / 'prediction_state.json').is_file():
                raise ValueError('Incomplete family: every variant and seed must finish before any comparison')
    parent, _ = direct.verify_parent(config)
    control = verify_control(config)
    started = time.perf_counter()
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as data:
        baseline = {k: data[k].copy() for k in data.files}
    uniform = members(control, '{seed}', 'prediction_state.json')
    g0 = members(parent, 'members/' + direct.CELL + '/seed{seed}', 'prediction_state.json')
    have_mlb = all((output / v / str(s) / 'mlb_prediction_state.json').is_file() for v in VARIANTS for s in SEEDS)
    if have_mlb:
        whole = Path(config['whole_mlb_reference'])
        uniform_mlb = members(control, '{seed}', 'mlb_prediction_state.json')
        g0_mlb = members(whole, 'members/' + direct.CELL + '/seed{seed}', 'prediction_state.json')
    weight, variants = config['mix']['direct_weight'], {}
    for variant in VARIANTS:
        candidate = members(output / variant, '{seed}', 'prediction_state.json')
        variants[variant] = {'cpanel_dev': describe(candidate, uniform, g0, baseline, weight)}
        if have_mlb:
            variants[variant]['mlb_dev'] = describe_mlb(members(output / variant, '{seed}', 'mlb_prediction_state.json'),
                                                        uniform_mlb, g0_mlb, weight)
        variants[variant]['costs_seconds'] = {str(s): {
            'fit': read_json(output / variant / str(s) / 'fit.json')['seconds_total'],
            'predict': read_json(output / variant / str(s) / 'prediction_runtime.json')['seconds'],
            'mlb_predict': read_json(output / variant / str(s) / 'mlb_prediction_runtime.json')['seconds'] if have_mlb else None}
            for s in SEEDS}
        variants[variant]['train_weights'] = read_json(output / variant / 'preparation.json')['train_weights']
    result = {'family': config['experiment_id'], 'status': config['status'], 'variants': variants,
              'controls': {'direct': 'frozen EXP-P15-001 (uniform weights)', 'mix': 'frozen EXP-P15-001 + G0-global, fixed 1/2'},
              'decision': None, 'git_commit': git_commit(), 'seconds': time.perf_counter() - started,
              'limits': ['Descriptive repeated-DEV comparison; Cpanel and whole-MLB DEV already exposed',
                         'No N/G decision and no winner selection until the root registers rules',
                         'Prediction only; no policy claim; no 2026 rows'],
              'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir()
    dump(destination / 'descriptive.json', result)
    print('RECENCY_DESCRIBED', {v: {a: r['cpanel_dev'][a]['after_june_blend']['paired']['nll'] for a in ('direct', 'mix')}
                                for v, r in variants.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='family directory; variants are sub-directories')
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
            verify_control(config)
            run(config, local, output / args.variant, identity(config, args.local_config, args.variant),
                smoke, args.mlb, args.seeds)
        else:
            score(config, args.local_config, output)


if __name__ == '__main__':
    main()
