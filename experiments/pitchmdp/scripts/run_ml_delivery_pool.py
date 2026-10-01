"""EXP-P14-001: frozen G0-global with league-only delivery pools (one change vs G0).

Inference-only additive runner.  Reuses the frozen EXP-P4-001 (G) preparation
and the three G0-global member weights read-only.  The only change is the
JointDelivery pool tier used for the 400-draw pre-pitch integration: tiers
above ``delivery.max_tier`` (the pitcher-specific pools) are removed, so every
pitch integrates over the league type/hand[/count] pool.  May temperature
calibration, June frequency blend and scoring follow the G0 procedure.

  replay : validity gate. Unrestricted pools + the frozen G0 temperature must
           reproduce the stored G0 member DEV probabilities on a fixed row
           prefix (reads only the already-scored G0 archive).
  run    : prepare -> per seed May calibration -> Cpanel blend/dev predict
           [-> whole-MLB predict with --mlb]; resumable, completed steps verified.
  score  : paired vs the frozen G0-global members (EXP-P4-001 Cpanel, and
           EXP-P11-001 whole-MLB when --mlb predictions exist); writes
           analysis/results.json and results/EXP-P14-001.json in the repository.

``--smoke`` limits calibration/prediction rows so the wiring can be timed end
to end; it only writes to an output ending ``-smoke`` and never scores.
"""
from __future__ import annotations
import argparse
import copy
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
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, paired_game_comparison, prediction_decision
from pitchmdp.model import outcome_labels
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_sharing import SOURCES as SHARING_SOURCES, load_data as load_parent, predictor

SEEDS = (0, 1, 2)
CELL = 'G0-global'
ARM = 'G14-league-pool'
SOURCES = [*SHARING_SOURCES, 'scripts/run_ml_delivery_pool.py']
SMOKE = {'temperature': 256, 'predict': 256}
LAST_YEAR = 2025  # Absolute rule: no 2026 rows for fit, calibration or selection.


def config_check(config, smoke=False):
    if config['protocol'] != 'ml_delivery_pool_v1' or config['seeds'] != list(SEEDS):
        raise ValueError('Delivery-pool settings differ from registered protocol')
    if config['baseline']['experiment_id'] != 'EXP-P4-001' or config['baseline']['cell'] != CELL:
        raise ValueError('Baseline must be the frozen G0-global')
    if config['delivery']['max_tier'] not in (0, 1):
        raise ValueError('Candidate must keep league tiers only')
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


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def restrict_delivery(delivery, max_tier):
    """Same fitted draws, pools above ``max_tier`` removed.  The parent object is not modified."""
    limited = copy.copy(delivery)
    limited.pools = {key: value for key, value in delivery.pools.items() if key[0] <= max_tier}
    if not limited.pools or len(limited.pools) == len(delivery.pools):
        raise ValueError('Restriction must remove the pitcher pools and keep league pools')
    return limited


def pool_report(delivery):
    levels, counts = np.unique([key[0] for key in delivery.pools], return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(levels, counts)}


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


def load(local, config):
    parent, pprep = verify_parent(config)
    store, context, parts, aux = load_parent(local, parent, pprep)
    guard_dates(parts)
    return parent, pprep, store, context, parts, aux


def g0_model(parent, pprep, seed):
    """Frozen G0-global member; weights and fit identity verified by the sharing runner."""
    model, dependencies = predictor(read_json(parent / 'registered_config.json'), parent, pprep, CELL, seed)
    return model, dependencies


def prepare(config, data, output, expected):
    if (output / 'preparation.json').exists():
        prep = read_json(output / 'preparation.json')
        if prep['identity'] != expected:
            raise ValueError('Delivery-pool source/config/environment changed')
        assert_hashes(output, prep['artifact_hashes'])
        return prep
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete preparation requires failure review')
    started = time.perf_counter()
    _, pprep, _, _, _, aux = data
    limited = restrict_delivery(aux['delivery'], config['delivery']['max_tier'])
    output.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        path = output / 'source' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)
    dump(output / 'registered_config.json', config)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    prep = {'identity': expected, 'parent_preparation_sha256': config['parent_preparation_sha256'],
            'delivery': {'max_tier': config['delivery']['max_tier'], 'draws': int(aux['delivery'].draws),
                         'parent_pools_by_tier': pool_report(aux['delivery']), 'kept_pools_by_tier': pool_report(limited)},
            'samples': {k: v for k, v in pprep['samples'].items()},
            'artifact_hashes': artifact_hashes(output, files), 'seconds': time.perf_counter() - started,
            'peak_rss_bytes': peak(), 'git_commit': git_commit(),
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False}
    dump(output / 'preparation.json', prep)
    print('DELIVERY_POOL_PREPARED', prep['delivery'], flush=True)
    return prep


def state_ok(dest, name, expected):
    if not (dest / name).exists():
        return False
    state = read_json(dest / name)
    if state['identity'] != expected:
        raise ValueError(f'{dest / name} identity changed')
    assert_hashes(dest, state['artifact_hashes'])
    return True


def predict(config, data, output, prep, seed, smoke, mlb=False):
    dest = output / str(seed)
    name = 'mlb_prediction_state.json' if mlb else 'prediction_state.json'
    parent, pprep, store, context, parts, aux = data
    model, dependencies = g0_model(parent, pprep, seed)
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke, 'dependencies': dependencies}
    if state_ok(dest, name, expected):
        return
    start = time.perf_counter()
    delivery = restrict_delivery(aux['delivery'], config['delivery']['max_tier'])
    limit = (lambda part, n: part.iloc[:n]) if smoke else (lambda part, n: part)
    dest.mkdir(parents=True, exist_ok=True)
    if mlb:
        if not state_ok(dest, 'prediction_state.json', expected):
            raise ValueError('Whole-MLB prediction requires the frozen Cpanel calibration')
        model.delivery_temperature = read_json(dest / 'calibration.json')['delivery_temperature']
    else:
        temp = limit(parts['temperature'], SMOKE['temperature'])
        delivery.calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
        dump(dest / 'calibration.json', model.report)
    values, counts = {}, {}
    for split in (('mlb_dev',) if mlb else ('blend', 'dev')):
        part = limit(parts[split], SMOKE['predict'])
        p, raw, levels = predict_streamed(model, delivery, store, context, part.index.to_numpy())
        if int(levels.max()) > config['delivery']['max_tier']:
            raise ValueError('A pitcher pool was used despite the restriction')
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
        raise ValueError('Sources changed during delivery-pool prediction')
    names = [prefix + 'predictions.npz', prefix + 'prediction_runtime.json'] + ([] if mlb else ['calibration.json'])
    dump(dest / name, {'identity': expected, 'artifact_hashes': artifact_hashes(dest, names)})
    print('DELIVERY_POOL_PREDICT_COMPLETE', seed, 'mlb' if mlb else 'panel', flush=True)


def run(config, local, output, expected, smoke, mlb, seeds):
    timings = {}
    before = time.perf_counter()
    data = load(local, config)
    timings['load'] = time.perf_counter() - before
    prep = prepare(config, data, output, expected)
    for seed in seeds:
        for step, wanted in (('predict', False), ('mlb_predict', True)):
            if wanted and not mlb:
                continue
            before = time.perf_counter()
            predict(config, data, output, prep, seed, smoke, mlb=wanted)
            timings[f'{step}_seed{seed}'] = time.perf_counter() - before
    print('DELIVERY_POOL_RUN_COMPLETE', {'seconds': timings, 'peak_rss_bytes': peak()}, flush=True)
    return timings


def replay(config, local, rows, tolerance):
    """Unrestricted pools + frozen G0 temperature must reproduce the stored G0 member (validity V)."""
    parent, pprep, store, context, parts, aux = load(local, config)
    part = parts['dev'].iloc[:rows]
    report = {}
    for seed in SEEDS:
        dest = parent / 'members' / CELL / f'seed{seed}'
        model, _ = g0_model(parent, pprep, seed)
        model.delivery_temperature = read_json(dest / 'calibration.json')['delivery_temperature']
        p, _, levels = predict_streamed(model, aux['delivery'], store, context, part.index.to_numpy())
        with np.load(dest / 'predictions.npz', allow_pickle=False) as stored:
            if not np.array_equal(stored['dev_keys'][:rows], part[KEY].to_numpy(np.int64)):
                raise ValueError('Replay rows differ from the stored G0 archive')
            error = float(np.abs(stored['dev'][:rows] - p).max())
            same_levels = bool(np.array_equal(stored['dev_delivery_level'][:rows], levels))
        report[str(seed)] = {'max_abs_error': error, 'same_delivery_levels': same_levels}
        if error > tolerance or not same_levels:
            raise ValueError(f'G0 replay failed for seed {seed}: {report[str(seed)]}')
    print('DELIVERY_POOL_REPLAY_PASS', {'rows': rows, 'tolerance': tolerance, 'seeds': report}, flush=True)
    return report


# ------------------------------------------------------------------ scoring

def members(root, cell_dir, name):
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


def tier_slices(y, games, candidate, control, g0_levels):
    """Descriptive: paired NLL difference by the tier the G0 control used for the same pitch."""
    loss = (pitch_losses(y, candidate) - pitch_losses(y, control))[:, 0]
    result = {}
    for label, mask in (('g0_league_pool_tiers_le1', g0_levels <= 1), ('g0_pitcher_pool_tiers_ge2', g0_levels >= 2)):
        result[label] = {'n': int(mask.sum()), 'games': int(len(np.unique(games[mask]))),
                         'delta_nll': float(loss[mask].mean()) if mask.any() else None}
    return result


def score_arrays(mine, g0, baseline, n_rule, mine_mlb=None, g0_mlb=None, mlb_frequency=None):
    """Pure scoring on loaded archives (synthetic-testable)."""
    from score_ml_matrix import assert_aligned, summarize_cell
    for member in (*mine, *g0):
        assert_aligned(member, baseline)
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    rep_a, pa = summarize_cell(mine, baseline)
    rep_b, pb = summarize_cell(g0, baseline)
    result = {'cpanel_dev': {'n': len(y), 'games': len(np.unique(games)),
              'reports': {ARM: rep_a, CELL: rep_b},
              'comparison': compare(y, games, pa['primary'], pb['primary'], pa['seed_primary'], pb['seed_primary'], n_rule),
              'before_blend': compare(y, games, pa['calibrated'], pb['calibrated'],
                                      [m['dev'] for m in mine], [m['dev'] for m in g0], n_rule),
              'by_g0_tier_primary': tier_slices(y, games, pa['primary'], pb['primary'], g0[0]['dev_delivery_level']),
              'multiplicity': 'single preregistered comparison (comparison); before_blend and tier slices are descriptive'}}
    if mine_mlb is not None:
        y, games, keys = g0_mlb[0]['dev_y'], g0_mlb[0]['dev_game_pk'], g0_mlb[0]['dev_keys']
        for m in mine_mlb:
            if not (np.array_equal(m['mlb_dev_keys'], keys) and np.array_equal(m['mlb_dev_y'], y)):
                raise ValueError('Unpaired whole-MLB predictions')
        for m in g0_mlb:
            if not np.array_equal(m['dev_keys'], keys):
                raise ValueError('Unpaired G0 whole-MLB members')
        a, b = [m['mlb_dev'] for m in mine_mlb], [m['dev'] for m in g0_mlb]
        mean_a, mean_b = np.mean(a, 0), np.mean(b, 0)
        result['mlb_dev'] = {'n': len(y), 'games': len(np.unique(games)),
            'comparison': compare(y, games, mean_a, mean_b, a, b, n_rule),
            'by_g0_tier_before_blend': tier_slices(y, games, mean_a, mean_b, g0_mlb[0]['dev_delivery_level']),
            'scope': 'secondary descriptive: calibrated three-seed mean, frozen May temperature, no June blend'}
        if mlb_frequency is not None:
            if not np.array_equal(mlb_frequency['mlb_dev_keys'], keys):
                raise ValueError('Unpaired whole-MLB frequency baseline')
            f = mlb_frequency['mlb_dev']
            wa, wb = rep_a['selection']['model_weight'], rep_b['selection']['model_weight']
            sa = [s['blend_selection']['model_weight'] * m + (1 - s['blend_selection']['model_weight']) * f
                  for s, m in zip(rep_a['seeds'], a)]
            sb = [s['blend_selection']['model_weight'] * m + (1 - s['blend_selection']['model_weight']) * f
                  for s, m in zip(rep_b['seeds'], b)]
            blended_a, blended_b = wa * mean_a + (1 - wa) * f, wb * mean_b + (1 - wb) * f
            result['mlb_dev_blended'] = {'n': len(y), 'games': len(np.unique(games)),
                'model_weight': {ARM: wa, CELL: wb},
                'comparison': compare(y, games, blended_a, blended_b, sa, sb, n_rule),
                'by_g0_tier_primary': tier_slices(y, games, blended_a, blended_b, g0_mlb[0]['dev_delivery_level']),
                'frequency': prediction_metrics(y, f),
                'scope': 'secondary descriptive: each arm blended with the frozen frequency baseline using its own '
                         'Cpanel June three-seed weight (per-seed weights for seed deltas)'}
    return result


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != identity(config, local_path):
        raise ValueError('Delivery-pool source/config/environment changed')
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
    g0 = members(parent, 'members/' + CELL + '/seed{seed}', 'prediction_state.json')
    mine_mlb = g0_mlb = frequency = None
    if all((output / str(s) / 'mlb_prediction_state.json').is_file() for s in SEEDS):
        whole = Path(config['whole_mlb_reference'])
        mine_mlb = members(output, '{seed}', 'mlb_prediction_state.json')
        g0_mlb = members(whole, 'members/' + CELL + '/seed{seed}', 'prediction_state.json')
        with np.load(whole / 'baseline_predictions.npz', allow_pickle=False) as data:
            frequency = {k: data[k].copy() for k in ('mlb_dev', 'mlb_dev_keys')}
    result = score_arrays(mine, g0, baseline, config['metrics']['N'], mine_mlb, g0_mlb, frequency)
    costs = {str(s): {name: read_json(output / str(s) / (prefix + 'prediction_runtime.json'))['seconds']
                      for name, prefix in (('predict', ''), ('mlb_predict', 'mlb_'))
                      if (output / str(s) / (prefix + 'prediction_runtime.json')).is_file()} for s in SEEDS}
    result = {'experiment_id': config['experiment_id'], 'baseline': config['baseline'], 'seeds': list(SEEDS),
              'one_change': config['one_change'], 'delivery': prep['delivery'],
              'git_commit': git_commit(), 'preparation_sha256': hash_file(output / 'preparation.json'),
              'parent_preparation_sha256': config['parent_preparation_sha256'],
              'data_identity': read_json(parent / 'parent_preparation.json')['dataset_identity'],
              'new_fits': 0, **result, 'costs_seconds': costs, 'policy_effect': None,
              'limits': ['Prediction only; no policy claim', 'Cpanel DEV previously exposed (development evaluation)',
                         'Hypothesis came from a post-hoc look at the same exposed DEV (D130)',
                         'Conditional bootstrap excludes training/calibration uncertainty'],
              'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir()
    dump(destination / 'results.json', result)
    repo_result.parent.mkdir(exist_ok=True)
    shutil.copyfile(destination / 'results.json', repo_result)
    print('DELIVERY_POOL_SCORED', result['cpanel_dev']['comparison']['paired']['nll'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    r.add_argument('--mlb', action='store_true', help='also predict the frozen whole-MLB DEV')
    r.add_argument('--smoke', action='store_true', help='row-limited wiring check; output must end -smoke')
    sub.add_parser('replay')
    sub.add_parser('score')
    args = p.parse_args()
    smoke = getattr(args, 'smoke', False) or args.command == 'replay'
    config, local = config_check(read_json(args.config), smoke), read_json(args.local_config)
    output = args.output.resolve()
    if args.command != 'replay':
        if smoke != output.name.endswith('-smoke'):
            raise ValueError('Smoke runs write only to an output ending -smoke, and only smoke runs may')
        if not smoke and output.name != config['experiment_id']:
            raise ValueError('Output directory must be named after the experiment')
    root = check_location(local, output)
    validate_native_runtime()
    with heavy_lock(root):
        if args.command == 'run':
            run(config, local, output, identity(config, args.local_config), smoke, args.mlb, args.seeds)
        elif args.command == 'replay':
            replay(config, local, config['replay']['rows'], config['replay']['tolerance'])
        else:
            score(config, args.local_config, output)


if __name__ == '__main__':
    main()
