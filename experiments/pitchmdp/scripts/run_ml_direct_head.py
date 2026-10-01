"""EXP-P15-001/002: direct pre-pitch head (no current-pitch physics) vs frozen G0-global.

Additive runner (matrix row MX-F8).  G0 learns outcome | context, candidate type,
current-pitch physics and integrates 400 TRAIN delivery draws at pre-pitch time.
The direct head is the same flatten MLP, data, splits and budget, but the eight
current-pitch physical channels are zero in training and inference, so one
forward pass gives the pre-pitch distribution.  Past pitches of the same plate
appearance keep their observed physics; the candidate pitch type stays.

  run   : prepare -> per seed fit -> May temperature -> Cpanel blend/dev predict
          [-> whole-MLB predict with --mlb]; resumable, completed steps verified.
  score : registered family of two comparisons against the frozen G0-global
          members: EXP-P15-001 direct head, EXP-P15-002 equal-weight mix of the
          direct head and G0 (no extra fit).  Writes analysis/results.json and
          results/EXP-P15-001.json in the repository.

``--smoke`` limits rows and epochs so the wiring can be timed end to end; it only
writes to an output ending ``-smoke`` and never scores.
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
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_metrics import holm_adjust, paired_game_comparison, pitch_losses, prediction_decision, prediction_metrics
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import SharingPredictor, training_arrays
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_delivery import JointDelivery
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_sharing import SOURCES as SHARING_SOURCES
from run_ml_delivery_pool import git_commit, load, members, peak, state_ok, tier_slices, verify_parent

SEEDS = (0, 1, 2)
CELL = 'G0-global'
DIRECT, MIX = 'G15-direct', 'G15-direct-g0-mix'
IDS = {DIRECT: 'EXP-P15-001', MIX: 'EXP-P15-002'}
SOURCES = [*SHARING_SOURCES, 'scripts/run_ml_delivery_pool.py', 'scripts/run_ml_direct_head.py']
SMOKE = {'train': 8192, 'earlystop': 2048, 'temperature': 256, 'predict': 256, 'epochs': 2}
NO_PHYSICS_LEVEL = -2  # delivery_level marker: no pool was used


def config_check(config, smoke=False):
    if config['protocol'] != 'ml_direct_head_v1' or config['seeds'] != list(SEEDS):
        raise ValueError('Direct-head settings differ from registered protocol')
    if config['baseline']['experiment_id'] != 'EXP-P4-001' or config['baseline']['cell'] != CELL:
        raise ValueError('Baseline must be the frozen G0-global')
    if config['family'] != [{'experiment_id': IDS[DIRECT], 'arm': DIRECT}, {'experiment_id': IDS[MIX], 'arm': MIX}]:
        raise ValueError('Registered family differs from the runner')
    if config['mix']['direct_weight'] != .5:
        raise ValueError('The mix weight is fixed at one half')
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


class NoPhysicsDelivery(JointDelivery):
    """One 'draw' of all-zero current physics: the direct head never sees the current delivery."""
    draws = 1

    def __init__(self, channels):
        self.channels = channels

    def sample(self, frame):
        return (np.zeros((len(frame), 1, self.channels), dtype=np.float32),
                np.full(len(frame), NO_PHYSICS_LEVEL, dtype=np.int64))


def direct_arrays(store, context, rows):
    """G0 input contract with the current token's physical channels forced to zero."""
    rows = np.asarray(rows)
    channels = store.physical.shape[1]
    tokens, valid = store.gather(rows, current=np.zeros((len(rows), channels), dtype=np.float32))
    if np.any(tokens[:, -1, :channels] != 0):
        raise ValueError('Current-pitch physics leaked into the direct head input')
    return tokens, valid, context.transform(store.frame.iloc[rows])


def prepare(config, data, output, expected):
    if (output / 'preparation.json').exists():
        prep = read_json(output / 'preparation.json')
        if prep['identity'] != expected:
            raise ValueError('Direct-head source/config/environment changed')
        assert_hashes(output, prep['artifact_hashes'])
        return prep
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete preparation requires failure review')
    started = time.perf_counter()
    _, pprep, store, _, _, _ = data
    output.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        path = output / 'source' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)
    dump(output / 'registered_config.json', config)
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
    prep = {'identity': expected, 'parent_preparation_sha256': config['parent_preparation_sha256'],
            'physical_channels_zeroed': int(store.physical.shape[1]),
            'samples': {k: v for k, v in pprep['samples'].items()},
            'artifact_hashes': artifact_hashes(output, files), 'seconds': time.perf_counter() - started,
            'peak_rss_bytes': peak(), 'git_commit': git_commit(),
            'prepared_utc': datetime.now(timezone.utc).isoformat(), 'dev_scores_read': False}
    dump(output / 'preparation.json', prep)
    print('DIRECT_HEAD_PREPARED', flush=True)
    return prep


def fit(config, data, output, prep, seed, smoke):
    dest = output / str(seed)
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke}
    if state_ok(dest, 'fit_state.json', expected):
        return
    if (dest / 'model.pt').exists():
        raise ValueError('Incomplete fit requires failure review')
    _, _, store, context, parts, _ = data
    train, early = parts['train'], parts['earlystop']
    if smoke:
        train, early = train.iloc[:SMOKE['train']], early.iloc[:SMOKE['earlystop']]
    spec = config['unchanged_from_g0']['training']
    start = time.perf_counter()
    ta = training_arrays(direct_arrays(store, context, train.index.to_numpy()))
    ea = training_arrays(direct_arrays(store, context, early.index.to_numpy()))
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
         'earlystop_metric': 'pre-pitch NLL (current physics zero), not the conditional NLL G0 early-stops on',
         'peak_rss_bytes': peak()})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during direct-head fit')
    dump(dest / 'fit_state.json', {'identity': expected, 'artifact_hashes': artifact_hashes(dest, ['model.pt', 'fit.json'])})
    print('DIRECT_HEAD_FIT_COMPLETE', seed, flush=True)
    del ta, ea, model


def predict(data, output, prep, seed, smoke, mlb=False):
    dest = output / str(seed)
    name = 'mlb_prediction_state.json' if mlb else 'prediction_state.json'
    expected = {'preparation_sha256': canonical_hash(prep), 'seed': seed, 'smoke': smoke,
                'model_sha256': hash_file(dest / 'model.pt')}
    if state_ok(dest, name, expected):
        return
    _, pprep, store, context, parts, _ = data
    start = time.perf_counter()
    model = SharingPredictor(CELL, MatrixModel.load(dest / 'model.pt'), pprep['clusters'])
    delivery = NoPhysicsDelivery(store.physical.shape[1])
    limit = (lambda part, n: part.iloc[:n]) if smoke else (lambda part, n: part)
    if mlb:
        if not state_ok(dest, 'prediction_state.json', expected):
            raise ValueError('Whole-MLB prediction requires the frozen Cpanel calibration')
        model.delivery_temperature = read_json(dest / 'calibration.json')['delivery_temperature']
    else:
        temp = limit(parts['temperature'], SMOKE['temperature'])
        delivery.calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
        dump(dest / 'calibration.json', model.report)
    values = {}
    for split in (('mlb_dev',) if mlb else ('blend', 'dev')):
        part = limit(parts[split], SMOKE['predict'])
        p, raw, levels = predict_streamed(model, delivery, store, context, part.index.to_numpy())
        if not np.all(levels == NO_PHYSICS_LEVEL) or not np.allclose(p.sum(1), 1., atol=1e-8):
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
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during direct-head prediction')
    names = [prefix + 'predictions.npz', prefix + 'prediction_runtime.json'] + ([] if mlb else ['calibration.json'])
    dump(dest / name, {'identity': expected, 'artifact_hashes': artifact_hashes(dest, names)})
    print('DIRECT_HEAD_PREDICT_COMPLETE', seed, 'mlb' if mlb else 'panel', flush=True)


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
    print('DIRECT_HEAD_RUN_COMPLETE', {'seconds': timings, 'peak_rss_bytes': peak()}, flush=True)
    return timings


# ------------------------------------------------------------------ scoring

def mixed(direct, g0, weight, parts=('blend', 'dev')):
    """Member-wise probability mix of the direct head and the G0 member of the same seed."""
    result = []
    for a, b in zip(direct, g0):
        member = dict(b)
        for part in parts:
            for suffix in ('', '_raw'):
                member[part + suffix] = weight * a[part + suffix] + (1 - weight) * b[part + suffix]
        result.append(member)
    return result


def paired(y, games, candidate, control, seeds_a, seeds_b):
    comparison = paired_game_comparison(y, candidate, control, games)
    seed_deltas = [float((pitch_losses(y, a) - pitch_losses(y, b))[:, 0].mean()) for a, b in zip(seeds_a, seeds_b)]
    return {'paired': comparison, 'seed_deltas': seed_deltas,
            'candidate': prediction_metrics(y, candidate), 'control': prediction_metrics(y, control)}


def decide(rows, n_rule, holm=True):
    """Attach N decisions; the registered primary family is Holm-adjusted across its rows."""
    raw = [row['paired']['nll']['p_less'] for row in rows.values()]
    adjusted = holm_adjust(raw) if holm else raw
    for row, p in zip(rows.values(), adjusted):
        row['N'] = prediction_decision(row['paired'], p, row['seed_deltas'],
                                       minimum_improvement=-n_rule['delta_nll_max'], brier_margin=n_rule['brier_ci95_upper_max'])
    return rows


def efficiency(row, c_rule, candidate_seconds, reference_seconds):
    """Registered C rule for the direct head: non-inferior loss and cheaper whole-MLB inference."""
    reduction = None if not candidate_seconds or not reference_seconds else 1 - sum(candidate_seconds) / sum(reference_seconds)
    criteria = {'nll_noninferior': row['paired']['nll']['ci95'][1] <= c_rule['nll_ci95_upper_max'],
                'brier_noninferior': row['paired']['brier']['ci95'][1] <= c_rule['brier_ci95_upper_max'],
                'cost_reduced': reduction is not None and reduction >= c_rule['min_cost_reduction']}
    return {'status': 'efficiency_improvement' if all(criteria.values()) else 'not_shown', 'criteria': criteria,
            'cost_reduction': reduction, 'candidate_seconds': candidate_seconds, 'reference_seconds': reference_seconds,
            'cost_scope': c_rule['cost_metric']}


def score_arrays(direct, g0, baseline, config, direct_mlb=None, g0_mlb=None, mlb_frequency=None,
                 direct_seconds=None, reference_seconds=None):
    """Pure scoring on loaded archives (synthetic-testable)."""
    from score_ml_matrix import assert_aligned, summarize_cell
    n_rule, weight = config['metrics']['N'], config['mix']['direct_weight']
    for member in (*direct, *g0):
        assert_aligned(member, baseline)
    arms = {DIRECT: direct, MIX: mixed(direct, g0, weight)}
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    rep_g0, p_g0 = summarize_cell(g0, baseline)
    reports, primary, before = {CELL: rep_g0}, {}, {}
    summaries = {}
    for arm, arm_members in arms.items():
        reports[arm], summaries[arm] = summarize_cell(arm_members, baseline)
        primary[arm] = paired(y, games, summaries[arm]['primary'], p_g0['primary'], summaries[arm]['seed_primary'], p_g0['seed_primary'])
        before[arm] = paired(y, games, summaries[arm]['calibrated'], p_g0['calibrated'],
                             [m['dev'] for m in arm_members], [m['dev'] for m in g0])
        primary[arm]['by_g0_tier'] = tier_slices(y, games, summaries[arm]['primary'], p_g0['primary'], g0[0]['dev_delivery_level'])
    result = {'cpanel_dev': {'n': len(y), 'games': len(np.unique(games)), 'reports': reports,
              'comparison': decide(primary, n_rule), 'before_blend': decide(before, n_rule, holm=False),
              'multiplicity': 'registered family of two one-sided tests (comparison), Holm alpha .05; before_blend is descriptive'}}
    if direct_mlb is not None:
        y, games, keys = g0_mlb[0]['dev_y'], g0_mlb[0]['dev_game_pk'], g0_mlb[0]['dev_keys']
        for m in direct_mlb:
            if not (np.array_equal(m['mlb_dev_keys'], keys) and np.array_equal(m['mlb_dev_y'], y)):
                raise ValueError('Unpaired whole-MLB predictions')
        for m in g0_mlb:
            if not np.array_equal(m['dev_keys'], keys):
                raise ValueError('Unpaired G0 whole-MLB members')
        if not np.array_equal(mlb_frequency['mlb_dev_keys'], keys):
            raise ValueError('Unpaired whole-MLB frequency baseline')
        f, b = mlb_frequency['mlb_dev'], [m['dev'] for m in g0_mlb]
        a = {DIRECT: [m['mlb_dev'] for m in direct_mlb]}
        a[MIX] = [weight * d + (1 - weight) * g for d, g in zip(a[DIRECT], b)]
        wb = rep_g0['selection']['model_weight']
        sb = [s['blend_selection']['model_weight'] * m + (1 - s['blend_selection']['model_weight']) * f
              for s, m in zip(rep_g0['seeds'], b)]
        blended_b = wb * np.mean(b, 0) + (1 - wb) * f
        plain, blended = {}, {}
        for arm, values in a.items():
            wa = reports[arm]['selection']['model_weight']
            sa = [s['blend_selection']['model_weight'] * m + (1 - s['blend_selection']['model_weight']) * f
                  for s, m in zip(reports[arm]['seeds'], values)]
            plain[arm] = paired(y, games, np.mean(values, 0), np.mean(b, 0), values, b)
            blended[arm] = paired(y, games, wa * np.mean(values, 0) + (1 - wa) * f, blended_b, sa, sb)
            blended[arm]['model_weight'] = {arm: wa, CELL: wb}
            blended[arm]['by_g0_tier'] = tier_slices(y, games, wa * np.mean(values, 0) + (1 - wa) * f, blended_b,
                                                     g0_mlb[0]['dev_delivery_level'])
        result['mlb_dev'] = {'n': len(y), 'games': len(np.unique(games)), 'comparison': decide(plain, n_rule, holm=False),
            'scope': 'secondary descriptive: calibrated three-seed mean, frozen May temperature, no June blend'}
        result['mlb_dev_blended'] = {'n': len(y), 'games': len(np.unique(games)), 'comparison': decide(blended, n_rule, holm=False),
            'frequency': prediction_metrics(y, f),
            'scope': 'secondary descriptive: each arm blended with the frozen frequency baseline using its own '
                     'Cpanel June three-seed weight (per-seed weights for seed deltas)'}
    result['C'] = {DIRECT: efficiency(result['cpanel_dev']['comparison'][DIRECT], config['metrics']['C'],
                                      direct_seconds, reference_seconds)}
    return result


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != identity(config, local_path):
        raise ValueError('Direct-head source/config/environment changed')
    destination, repo_result = output / 'analysis', REPO / 'results' / (IDS[DIRECT] + '.json')
    if destination.exists() or repo_result.exists():
        raise ValueError('Preserve completed or interrupted scoring output')
    parent, _ = verify_parent(config)
    started = time.perf_counter()
    for seed in SEEDS:
        if not (output / str(seed) / 'prediction_state.json').is_file():
            raise ValueError('Incomplete family; scores remain unopened')
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as data:
        baseline = {k: data[k].copy() for k in data.files}
    direct = members(output, '{seed}', 'prediction_state.json')
    g0 = members(parent, 'members/' + CELL + '/seed{seed}', 'prediction_state.json')
    direct_mlb = g0_mlb = frequency = direct_seconds = reference_seconds = None
    if all((output / str(s) / 'mlb_prediction_state.json').is_file() for s in SEEDS):
        whole, reference = Path(config['whole_mlb_reference']), Path(config['cost_reference_run'])
        direct_mlb = members(output, '{seed}', 'mlb_prediction_state.json')
        g0_mlb = members(whole, 'members/' + CELL + '/seed{seed}', 'prediction_state.json')
        with np.load(whole / 'baseline_predictions.npz', allow_pickle=False) as data:
            frequency = {k: data[k].copy() for k in ('mlb_dev', 'mlb_dev_keys')}
        direct_seconds = [read_json(output / str(s) / 'mlb_prediction_runtime.json')['seconds'] for s in SEEDS]
        reference_seconds = [read_json(reference / str(s) / 'mlb_prediction_runtime.json')['seconds'] for s in SEEDS]
    result = score_arrays(direct, g0, baseline, config, direct_mlb, g0_mlb, frequency, direct_seconds, reference_seconds)
    costs = {str(s): {'fit': read_json(output / str(s) / 'fit.json')['seconds_total'],
                      'best_epoch': read_json(output / str(s) / 'fit.json')['report']['best_epoch'],
                      'predict': read_json(output / str(s) / 'prediction_runtime.json')['seconds']} for s in SEEDS}
    result = {'experiment_ids': list(IDS.values()), 'family': config['family'], 'baseline': config['baseline'],
              'seeds': list(SEEDS), 'one_change': config['one_change'], 'mix': config['mix'],
              'git_commit': git_commit(), 'preparation_sha256': hash_file(output / 'preparation.json'),
              'parent_preparation_sha256': config['parent_preparation_sha256'],
              'data_identity': read_json(parent / 'parent_preparation.json')['dataset_identity'],
              'new_fits': len(SEEDS), **result, 'costs_seconds': costs, 'policy_effect': None,
              'limits': ['Prediction only; no policy claim', 'Cpanel DEV previously exposed (development evaluation)',
                         'Pitch-type-only pre-pitch marginal; no target-location input (MX-F8 first comparison)',
                         'Conditional bootstrap excludes training/calibration uncertainty'],
              'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir()
    dump(destination / 'results.json', result)
    repo_result.parent.mkdir(exist_ok=True)
    shutil.copyfile(destination / 'results.json', repo_result)
    print('DIRECT_HEAD_SCORED', {arm: row['paired']['nll'] for arm, row in result['cpanel_dev']['comparison'].items()}, flush=True)


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
