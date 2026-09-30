"""EXP-P13-001: G0-global plus TRAIN-only pitcher arsenal features (one change vs G0).

Additive runner.  Reuses the frozen EXP-P4-001 (G) preparation read-only: same
keys, store, aux delivery model, clusters and frequency baseline.  Only the
context gains ``ArsenalContext`` and the network gains the candidate arsenal
row (``CandidateArsenalModel``).  Training, May temperature calibration and the
400-draw ``predict_streamed`` path are the G0 ones.

  run   : prepare (arsenal fit on TRAIN) -> per seed fit -> Cpanel predict
          [-> whole-MLB predict with --mlb]; resumable, completed steps verified.
  score : paired vs the frozen G0-global members (EXP-P4-001 Cpanel, and
          EXP-P11-001 whole-MLB when --mlb predictions exist); writes
          analysis/results.json and results/EXP-P13-001.json in the repository.

``--smoke`` limits rows (fit/earlystop/calibration/prediction) and epochs so the
wiring can be timed end to end; it only writes to an output ending ``-smoke``
and never scores.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_arsenal import ArsenalContext, CandidateArsenalModel, fit_pitcher_arsenal, with_candidate_arsenal
from pitchmdp.matrix_sharing import SharingPredictor, training_arrays
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, paired_game_comparison, prediction_decision
from pitchmdp.model import outcome_labels
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_sharing import SOURCES as SHARING_SOURCES, load_data as load_parent
from run_sequence_pilot import arrays

SEEDS = (0, 1, 2)
SOURCES = [*SHARING_SOURCES, 'pitchmdp/matrix_arsenal.py', 'scripts/run_ml_arsenal.py']
SMOKE = {'train': 8192, 'earlystop': 2048, 'temperature': 256, 'predict': 256, 'epochs': 2}
LAST_YEAR = 2025  # Absolute rule: no 2026 rows for fit, calibration or selection.


def config_check(config, smoke=False):
    if config['protocol'] != 'ml_arsenal_v1' or config['seeds'] != list(SEEDS):
        raise ValueError('Arsenal settings differ from registered protocol')
    if config['baseline']['experiment_id'] != 'EXP-P4-001' or config['baseline']['cell'] != 'G0-global':
        raise ValueError('Baseline must be the frozen G0-global')
    if not smoke and config['execution']['enabled'] is not True:
        raise ValueError('Execution not enabled in the registered config')
    if not config.get('parent_preparation_sha256'):
        raise ValueError('Parent preparation hash must be registered')
    return config


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    from run_ml_benchmark import identity as arch_identity
    return {**arch_identity(config, local_path), 'source_hashes': source_hashes()}


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def guard_dates(parts):
    for name, part in parts.items():
        years = pd.to_datetime(part.game_date).dt.year
        if years.isna().any() or int(years.max()) > LAST_YEAR:
            raise ValueError(f'{name}: rows after {LAST_YEAR} are forbidden')


def verify_parent(config):
    parent = Path(config['parent_run'])
    if hash_file(parent / 'preparation.json') != config['parent_preparation_sha256']:
        raise ValueError('G0 parent preparation changed')
    prep = read_json(parent / 'preparation.json')
    assert_hashes(parent, prep['artifact_hashes'])
    for rel, digest in prep['identity']['source_hashes'].items():
        if hash_file(PROJECT / rel) != digest:
            raise ValueError('Frozen G0 implementation changed: ' + rel)
    return parent, prep


def load(local, config, arsenal=None):
    parent, pprep = verify_parent(config)
    store, sharing, parts, aux = load_parent(local, parent, pprep)
    guard_dates(parts)
    return pprep, store, ArsenalContext(sharing, arsenal), parts, aux


def prepare(config, data, output, expected):
    if (output / 'preparation.json').exists():
        prep = read_json(output / 'preparation.json')
        if prep['identity'] != expected:
            raise ValueError('Arsenal source/config/environment changed')
        assert_hashes(output, prep['artifact_hashes'])
        return prep
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete preparation requires failure review')
    started = time.perf_counter()
    pprep, _, _, parts, _ = data
    vocabulary = pprep['features']['tokens']['type_vocabulary']
    arsenal = fit_pitcher_arsenal(parts['train'], vocabulary, tau=float(config['arsenal']['tau']))
    output.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        path = output / 'source' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)
    dump(output / 'registered_config.json', config)
    dump(output / 'arsenal.json', arsenal)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    prep = {'identity': expected, 'parent_preparation_sha256': config['parent_preparation_sha256'],
            'arsenal_sha256': canonical_hash(arsenal), 'n_types': len(vocabulary),
            'samples': {k: v for k, v in pprep['samples'].items()},
            'artifact_hashes': artifact_hashes(output, files), 'seconds': time.perf_counter() - started,
            'peak_rss_bytes': peak(), 'git_commit': git_commit(),
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False}
    dump(output / 'preparation.json', prep)
    print('ARSENAL_PREPARED', prep['seconds'], flush=True)
    return prep


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def state_ok(dest, name, expected):
    if not (dest / name).exists():
        return False
    state = read_json(dest / name)
    if state['identity'] != expected:
        raise ValueError(f'{dest / name} identity changed')
    assert_hashes(dest, state['artifact_hashes'])
    return True


def fit(config, data, output, prep, seed, smoke):
    dest = output / str(seed)
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke}
    if state_ok(dest, 'fit_state.json', expected):
        return
    if (dest / 'model.pt').exists():
        raise ValueError('Incomplete fit requires failure review')
    _, store, context, parts, _ = data
    train, early = parts['train'], parts['earlystop']
    if smoke:
        train, early = train.iloc[:SMOKE['train']], early.iloc[:SMOKE['earlystop']]
    spec = config['unchanged_from_g0']['training']
    n_types = prep['n_types']
    start = time.perf_counter()
    ta = with_candidate_arsenal(training_arrays(arrays(store, context, train.index.to_numpy())), n_types)
    ea = with_candidate_arsenal(training_arrays(arrays(store, context, early.index.to_numpy())), n_types)
    array_seconds = time.perf_counter() - start
    model = MatrixModel(spec['kind'], seed=seed, width=spec['width']).fit(
        ta, outcome_labels(train), ea, outcome_labels(early),
        epochs=SMOKE['epochs'] if smoke else spec['epochs'], patience=spec['patience'],
        batch_size=spec['batch_size'], learning_rate=spec['learning_rate'])
    dest.mkdir(parents=True, exist_ok=True)
    model.save(dest / 'model.pt')
    dump(dest / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter() - start,
         'array_seconds': array_seconds, 'context_width': int(ta[2].shape[1]),
         'train_rows_sha256': ordered_key_hash(train), 'earlystop_rows_sha256': ordered_key_hash(early),
         'peak_rss_bytes': peak()})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during arsenal fit')
    dump(dest / 'fit_state.json', {'identity': expected, 'artifact_hashes': artifact_hashes(dest, ['model.pt', 'fit.json'])})
    print('ARSENAL_FIT_COMPLETE', seed, flush=True)
    del ta, ea, model


def predict(data, output, prep, seed, smoke, mlb=False):
    dest = output / str(seed)
    name = 'mlb_prediction_state.json' if mlb else 'prediction_state.json'
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke,
                'model_sha256': hash_file(dest / 'model.pt')}
    if state_ok(dest, name, expected):
        return
    pprep, store, context, parts, aux = data
    start = time.perf_counter()
    model = SharingPredictor('G0-global', CandidateArsenalModel(MatrixModel.load(dest / 'model.pt'), prep['n_types']),
                             pprep['clusters'])
    limit = (lambda part, n: part.iloc[:n]) if smoke else (lambda part, n: part)
    if mlb:
        if not state_ok(dest, 'prediction_state.json', {**expected}):
            raise ValueError('Whole-MLB prediction requires the frozen Cpanel calibration')
        model.delivery_temperature = read_json(dest / 'calibration.json')['delivery_temperature']
    else:
        temp = limit(parts['temperature'], SMOKE['temperature'])
        aux['delivery'].calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
        dump(dest / 'calibration.json', model.report)
    values, counts = {}, {}
    for split in (('mlb_dev',) if mlb else ('blend', 'dev')):
        part = limit(parts[split], SMOKE['predict'])
        p, raw, levels = predict_streamed(model, aux['delivery'], store, context, part.index.to_numpy())
        values.update({split: p, split + '_raw': raw, split + '_delivery_level': levels,
            split + '_keys': part[KEY].to_numpy(np.int64), split + '_y': outcome_labels(part),
            split + '_game_pk': part.game_pk.to_numpy(np.int64), split + '_pitcher': part.pitcher.to_numpy(np.int64)})
        unique, n = np.unique(levels, return_counts=True)
        counts[split] = {str(int(k)): int(v) for k, v in zip(unique, n)}
        if not np.allclose(p.sum(1), 1., atol=1e-8):
            raise ValueError('Prediction mass differs')
    prefix = 'mlb_' if mlb else ''
    if (dest / (prefix + 'predictions.npz')).exists():
        raise ValueError('Incomplete prediction archive requires failure review')
    np.savez_compressed(dest / (prefix + 'predictions.npz'), **values)
    dump(dest / (prefix + 'prediction_runtime.json'), {'seconds': time.perf_counter() - start,
         'peak_rss_bytes': peak(), 'delivery_tier_counts': counts, 'rows': {k: len(v) for k, v in values.items() if k.endswith('_y')}})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during arsenal prediction')
    names = [prefix + 'predictions.npz', prefix + 'prediction_runtime.json'] + ([] if mlb else ['calibration.json'])
    dump(dest / name, {'identity': expected, 'artifact_hashes': artifact_hashes(dest, names)})
    print('ARSENAL_PREDICT_COMPLETE', seed, 'mlb' if mlb else 'panel', flush=True)


def run(config, local, output, expected, smoke, mlb, seeds):
    timings = {}
    before = time.perf_counter()
    data = load(local, config)
    timings['load'] = time.perf_counter() - before
    before = time.perf_counter()
    prep = prepare(config, data, output, expected)
    arsenal = read_json(output / 'arsenal.json')
    if canonical_hash(arsenal) != prep['arsenal_sha256']:
        raise ValueError('Arsenal table changed')
    data[2].arsenal = arsenal  # flag on: ArsenalContext now inserts the block
    timings['prepare'] = time.perf_counter() - before
    for seed in seeds:
        for step, call in (('fit', lambda: fit(config, data, output, prep, seed, smoke)),
                           ('predict', lambda: predict(data, output, prep, seed, smoke)),
                           ('mlb_predict', lambda: predict(data, output, prep, seed, smoke, mlb=True) if mlb else None)):
            before = time.perf_counter()
            call()
            timings[f'{step}_seed{seed}'] = time.perf_counter() - before
    print('ARSENAL_RUN_COMPLETE', {'seconds': timings, 'peak_rss_bytes': peak()}, flush=True)
    return timings


# ------------------------------------------------------------------ scoring

def members(root, cell_dir, name, expected_hashes=True):
    result = []
    for seed in SEEDS:
        dest = root / cell_dir.format(seed=seed)
        state = read_json(dest / name)
        assert_hashes(dest, state['artifact_hashes'])
        archive_name = 'mlb_predictions.npz' if name.startswith('mlb_') else 'predictions.npz'
        with np.load(dest / archive_name, allow_pickle=False) as data:
            result.append({k: data[k].copy() for k in data.files})
    return result


def compare(y, games, candidate, control, seeds_a, seeds_b, n_rule):
    paired = paired_game_comparison(y, candidate, control, games)
    seed_deltas = [float((pitch_losses(y, a) - pitch_losses(y, b))[:, 0].mean()) for a, b in zip(seeds_a, seeds_b)]
    decision = prediction_decision(paired, paired['nll']['p_less'], seed_deltas,
                                   minimum_improvement=-n_rule['delta_nll_max'], brier_margin=n_rule['brier_ci95_upper_max'])
    return {'paired': paired, 'seed_deltas': seed_deltas, 'N': decision,
            'candidate': prediction_metrics(y, candidate), 'control': prediction_metrics(y, control)}


def score_arrays(mine, g0, baseline, n_rule, mine_mlb=None, g0_mlb=None):
    """Pure scoring on loaded archives (synthetic-testable)."""
    from score_ml_matrix import assert_aligned, summarize_cell
    for member in (*mine, *g0):
        assert_aligned(member, baseline)
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    rep_a, pa = summarize_cell(mine, baseline)
    rep_b, pb = summarize_cell(g0, baseline)
    result = {'cpanel_dev': {'n': len(y), 'games': len(np.unique(games)),
              'reports': {'G13-arsenal': rep_a, 'G0-global': rep_b},
              'comparison': compare(y, games, pa['primary'], pb['primary'], pa['seed_primary'], pb['seed_primary'], n_rule),
              'multiplicity': 'single preregistered comparison; unadjusted one-sided p'}}
    if mine_mlb is not None:
        y, games, keys = g0_mlb[0]['dev_y'], g0_mlb[0]['dev_game_pk'], g0_mlb[0]['dev_keys']
        for m in mine_mlb:
            if not (np.array_equal(m['mlb_dev_keys'], keys) and np.array_equal(m['mlb_dev_y'], y)):
                raise ValueError('Unpaired whole-MLB predictions')
        for m in g0_mlb:
            if not np.array_equal(m['dev_keys'], keys):
                raise ValueError('Unpaired G0 whole-MLB members')
        a, b = [m['mlb_dev'] for m in mine_mlb], [m['dev'] for m in g0_mlb]
        result['mlb_dev'] = {'n': len(y), 'games': len(np.unique(games)),
            'comparison': compare(y, games, np.mean(a, 0), np.mean(b, 0), a, b, n_rule),
            'scope': 'secondary descriptive: calibrated three-seed mean, frozen May temperature, no June blend'}
    return result


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != identity(config, local_path):
        raise ValueError('Arsenal source/config/environment changed')
    destination, repo_result = output / 'analysis', REPO / 'results' / (config['experiment_id'] + '.json')
    if destination.exists() or repo_result.exists():
        raise ValueError('Preserve completed or interrupted scoring output')
    parent, _ = verify_parent(config)
    started = time.perf_counter()
    for seed in SEEDS:
        if not (output / str(seed) / 'prediction_state.json').is_file():
            raise ValueError('Incomplete family; scores remain unopened')
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as data:
        baseline = {k: data[k].copy() for k in data.files}
    mine = members(output, '{seed}', 'prediction_state.json')
    g0 = members(parent, 'members/G0-global/seed{seed}', 'prediction_state.json')
    mine_mlb = g0_mlb = None
    if all((output / str(s) / 'mlb_prediction_state.json').is_file() for s in SEEDS):
        mine_mlb = members(output, '{seed}', 'mlb_prediction_state.json')
        g0_mlb = members(Path(config['whole_mlb_reference']), 'members/G0-global/seed{seed}', 'prediction_state.json')
    result = score_arrays(mine, g0, baseline, config['metrics']['N'], mine_mlb, g0_mlb)
    costs = {str(s): {'fit': read_json(output / str(s) / 'fit.json')['seconds_total'],
                      'predict': read_json(output / str(s) / 'prediction_runtime.json')['seconds']} for s in SEEDS}
    result = {'experiment_id': config['experiment_id'], 'baseline': config['baseline'], 'seeds': list(SEEDS),
              'git_commit': git_commit(), 'preparation_sha256': hash_file(output / 'preparation.json'),
              'parent_preparation_sha256': config['parent_preparation_sha256'],
              'data_identity': read_json(parent / 'parent_preparation.json')['dataset_identity'],
              **result, 'costs_seconds': costs, 'policy_effect': None,
              'limits': ['Prediction only; no policy claim', 'Cpanel DEV previously exposed (development evaluation)',
                         'Conditional bootstrap excludes training/calibration uncertainty'],
              'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir()
    dump(destination / 'results.json', result)
    repo_result.parent.mkdir(exist_ok=True)
    shutil.copyfile(destination / 'results.json', repo_result)
    print('ARSENAL_SCORED', result['cpanel_dev']['comparison']['paired']['nll'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    r.add_argument('--mlb', action='store_true', help='also predict the frozen whole-MLB DEV')
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
            run(config, local, output, identity(config, args.local_config), smoke, args.mlb, args.seeds)
        else:
            score(config, args.local_config, output)


if __name__ == '__main__':
    main()
