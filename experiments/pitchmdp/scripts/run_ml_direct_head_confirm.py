"""EXP-P16-001: five-seed and whole-MLB confirmation of the EXP-P15-002 fixed mix.

Additive runner.  EXP-P15-002 (equal-weight mix of the direct pre-pitch head and
the frozen G0-global member of the same seed) passed N on three seeds.  Its
registered reading asks for a five-seed check and a whole-MLB confirmation before
any G0 change.  Nothing about the model, data, splits or mix changes here:

  run   : fit and predict the direct head for seeds 3 and 4 only, with the
          unchanged EXP-P15-001 fit/predict functions; seeds 0-2 are the sealed
          EXP-P15-001 archives and are never refitted.
  score : five-seed scoring against the frozen G0-global members (seeds 0-2 from
          EXP-P4-001, seeds 3-4 from EXP-P10-001, whole MLB from EXP-P11-001).
          The mix is confirmed only if it passes the final-stage N rule (4/5
          seeds) on both co-primary comparisons.  Writes analysis/results.json
          and results/EXP-P16-001.json in the repository.

``--smoke`` limits rows and epochs for seed 3 only; it writes to an output ending
``-smoke`` and never scores.
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
from pitchmdp.data import hash_file
from pitchmdp.matrix_metrics import holm_adjust, prediction_decision, prediction_metrics
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, check_location, heavy_lock
from run_ml_delivery_pool import git_commit, peak, tier_slices, verify_parent
import run_ml_direct_head as base

NAME = 'scripts/run_ml_direct_head_confirm.py'
BASE_SOURCES = tuple(base.SOURCES)
# The reused prepare/fit/predict steps hash base.SOURCES; adding this file makes them cover it too.
base.SOURCES = [*BASE_SOURCES, NAME]

EXPERIMENT = 'EXP-P16-001'
FROZEN_SEEDS, NEW_SEEDS = (0, 1, 2), (3, 4)
SEEDS = FROZEN_SEEDS + NEW_SEEDS
CELL, DIRECT, MIX = base.CELL, base.DIRECT, base.MIX
SAME_AS_FROZEN = ('unchanged_from_g0', 'one_change', 'mix', 'parent_run', 'parent_preparation_sha256',
                  'whole_mlb_reference', 'leak_guard', 'data_rules')
PASS = 'predictive_improvement'


def config_check(config, smoke=False):
    if config['protocol'] != 'ml_direct_head_confirm_v1' or config['experiment_id'] != EXPERIMENT:
        raise ValueError('Confirmation settings differ from registered protocol')
    if config['seeds'] != list(SEEDS) or config['new_fit_seeds'] != list(NEW_SEEDS):
        raise ValueError('Confirmation uses seeds 0-4 and fits only seeds 3 and 4')
    if config['confirmed_arm'] != MIX or config['mix']['direct_weight'] != .5:
        raise ValueError('The confirmed arm is the fixed equal-weight mix')
    if config['baseline']['cell'] != CELL:
        raise ValueError('Baseline must be the frozen G0-global')
    if config['metrics']['N'].get('required_negative_seeds') != 4:
        raise ValueError('Final-stage N needs four of five seeds')
    for key in ('parent_preparation_sha256', 'frozen_direct_run', 'g0_extra_seeds_run', 'whole_mlb_reference'):
        if not config.get(key):
            raise ValueError('Registered input missing: ' + key)
    if not smoke and config['execution']['enabled'] is not True:
        raise ValueError('Execution not enabled in the registered config')
    return config


def verify_frozen_direct(config):
    """Sealed EXP-P15-001 run: same registered settings and the same fit/predict implementation."""
    spec = config['frozen_direct_run']
    root = Path(spec['path'])
    if hash_file(root / 'preparation.json') != spec['preparation_sha256']:
        raise ValueError('Frozen direct-head preparation changed')
    prep = read_json(root / 'preparation.json')
    assert_hashes(root, prep['artifact_hashes'])
    if set(prep['identity']['source_hashes']) != set(BASE_SOURCES):
        raise ValueError('Frozen direct-head source list differs from the runner')
    for rel, digest in prep['identity']['source_hashes'].items():
        if hash_file(PROJECT / rel) != digest:
            raise ValueError('Direct-head implementation changed since the frozen run: ' + rel)
    frozen = read_json(root / 'registered_config.json')
    for key in SAME_AS_FROZEN:
        if frozen[key] != config[key]:
            raise ValueError('Confirmation must not change the frozen setting: ' + key)
    return root


def verify_g0_extra(config):
    """EXP-P10-001 holds G0-global seeds 3 and 4, fitted on the same parent preparation."""
    spec = config['g0_extra_seeds_run']
    root = Path(spec['path'])
    if hash_file(root / 'preparation.json') != spec['preparation_sha256']:
        raise ValueError('G0 extra-seed preparation changed')
    if read_json(root / 'preparation.json')['parent_preparation_sha256'] != config['parent_preparation_sha256']:
        raise ValueError('G0 extra seeds were not fitted on the registered parent')
    return root


def verify_whole_mlb(config):
    root = Path(config['whole_mlb_reference'])
    if hash_file(root / 'preparation.json') != config['whole_mlb_preparation_sha256']:
        raise ValueError('Whole-MLB reference preparation changed')
    return root


def run(config, local, output, expected, smoke):
    verify_frozen_direct(config)
    timings = {}
    before = time.perf_counter()
    data = base.load(local, config)
    timings['load'] = time.perf_counter() - before
    prep = base.prepare(config, data, output, expected)
    for seed in (NEW_SEEDS[:1] if smoke else NEW_SEEDS):
        for step, call in (('fit', lambda: base.fit(config, data, output, prep, seed, smoke)),
                           ('predict', lambda: base.predict(data, output, prep, seed, smoke)),
                           ('mlb_predict', lambda: base.predict(data, output, prep, seed, smoke, mlb=True))):
            before = time.perf_counter()
            call()
            timings[f'{step}_seed{seed}'] = time.perf_counter() - before
    print('DIRECT_CONFIRM_RUN_COMPLETE', {'seconds': timings, 'peak_rss_bytes': peak()}, flush=True)
    return timings


# ------------------------------------------------------------------ scoring
def member(root, rel, name):
    dest = root / rel
    state = read_json(dest / name)
    assert_hashes(dest, state['artifact_hashes'])
    with np.load(dest / ('mlb_predictions.npz' if name.startswith('mlb_') else 'predictions.npz'), allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def decide(rows, n_rule, holm):
    """Final-stage N (five seeds, four must improve); Holm across the rows when asked."""
    raw = [row['paired']['nll']['p_less'] for row in rows.values()]
    adjusted = holm_adjust(raw) if holm else raw
    for row, p in zip(rows.values(), adjusted):
        row['N'] = prediction_decision(row['paired'], p, row['seed_deltas'], minimum_improvement=-n_rule['delta_nll_max'],
                                       brier_margin=n_rule['brier_ci95_upper_max'], stage='final')
    return rows


def score_arrays(direct, g0, baseline, config, direct_mlb, g0_mlb, mlb_frequency):
    """Pure five-seed scoring on loaded archives (synthetic-testable).  Same quantities as EXP-P15-001."""
    from score_ml_matrix import assert_aligned, summarize_cell
    if not len(direct) == len(g0) == len(direct_mlb) == len(g0_mlb) == len(SEEDS):
        raise ValueError('Confirmation needs all five seeds of every arm')
    n_rule, weight = config['metrics']['N'], config['mix']['direct_weight']
    for item in (*direct, *g0):
        assert_aligned(item, baseline)
    arms = {DIRECT: direct, MIX: base.mixed(direct, g0, weight)}
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    rep_g0, p_g0 = summarize_cell(g0, baseline)
    reports, primary, before = {CELL: rep_g0}, {}, {}
    for arm, arm_members in arms.items():
        reports[arm], summary = summarize_cell(arm_members, baseline)
        primary[arm] = base.paired(y, games, summary['primary'], p_g0['primary'], summary['seed_primary'], p_g0['seed_primary'])
        before[arm] = base.paired(y, games, summary['calibrated'], p_g0['calibrated'],
                                  [m['dev'] for m in arm_members], [m['dev'] for m in g0])
        primary[arm]['by_g0_tier'] = tier_slices(y, games, summary['primary'], p_g0['primary'], g0[0]['dev_delivery_level'])
    result = {'cpanel_dev': {'n': len(y), 'games': len(np.unique(games)), 'reports': reports,
              'comparison': decide(primary, n_rule, holm=True), 'before_blend': decide(before, n_rule, holm=False),
              'multiplicity': 'Holm alpha .05 across the two arms, as in EXP-P15-001; before_blend is descriptive'}}
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
        plain[arm] = base.paired(y, games, np.mean(values, 0), np.mean(b, 0), values, b)
        blended[arm] = base.paired(y, games, wa * np.mean(values, 0) + (1 - wa) * f, blended_b, sa, sb)
        blended[arm]['model_weight'] = {arm: wa, CELL: wb}
        blended[arm]['by_g0_tier'] = tier_slices(y, games, wa * np.mean(values, 0) + (1 - wa) * f, blended_b,
                                                 g0_mlb[0]['dev_delivery_level'])
    result['mlb_dev'] = {'n': len(y), 'games': len(np.unique(games)), 'comparison': decide(plain, n_rule, holm=False),
        'scope': 'descriptive: calibrated five-seed mean, frozen May temperature, no June blend'}
    result['mlb_dev_blended'] = {'n': len(y), 'games': len(np.unique(games)), 'comparison': decide(blended, n_rule, holm=False),
        'frequency': prediction_metrics(y, f),
        'scope': 'co-primary for the mix: each arm blended with the frozen frequency baseline using its own '
                 'Cpanel June five-seed weight (per-seed weights for seed deltas); no multiplicity adjustment '
                 'because both co-primary comparisons must pass'}
    checks = {'cpanel_dev_after_blend': result['cpanel_dev']['comparison'][MIX]['N']['status'],
              'whole_mlb_dev_blended': result['mlb_dev_blended']['comparison'][MIX]['N']['status']}
    result['confirmation'] = {'arm': MIX, 'co_primary': checks,
        'status': 'confirmed' if all(status == PASS for status in checks.values()) else 'not_confirmed',
        'rule': 'confirmed only if the mix is a final-stage predictive_improvement on both co-primary comparisons',
        'direct_head_rows': 'descriptive in this experiment'}
    return result


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != base.identity(config, local_path):
        raise ValueError('Confirmation source/config/environment changed')
    destination, repo_result = output / 'analysis', REPO / 'results' / (EXPERIMENT + '.json')
    if destination.exists() or repo_result.exists():
        raise ValueError('Preserve completed or interrupted scoring output')
    parent, _ = verify_parent(config)
    frozen, extra, whole = verify_frozen_direct(config), verify_g0_extra(config), verify_whole_mlb(config)
    started = time.perf_counter()
    for seed in NEW_SEEDS:
        for name in ('prediction_state.json', 'mlb_prediction_state.json'):
            if not (output / str(seed) / name).is_file():
                raise ValueError('Incomplete confirmation; scores remain unopened')
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as data:
        baseline = {k: data[k].copy() for k in data.files}
    with np.load(whole / 'baseline_predictions.npz', allow_pickle=False) as data:
        frequency = {k: data[k].copy() for k in ('mlb_dev', 'mlb_dev_keys')}
    roots = {**{s: frozen for s in FROZEN_SEEDS}, **{s: output for s in NEW_SEEDS}}
    g0_roots = {**{s: parent for s in FROZEN_SEEDS}, **{s: extra for s in NEW_SEEDS}}
    direct = [member(roots[s], str(s), 'prediction_state.json') for s in SEEDS]
    direct_mlb = [member(roots[s], str(s), 'mlb_prediction_state.json') for s in SEEDS]
    g0 = [member(g0_roots[s], f'members/{CELL}/seed{s}', 'prediction_state.json') for s in SEEDS]
    g0_mlb = [member(whole, f'members/{CELL}/seed{s}', 'prediction_state.json') for s in SEEDS]
    result = score_arrays(direct, g0, baseline, config, direct_mlb, g0_mlb, frequency)
    costs = {str(s): {'fit': read_json(roots[s] / str(s) / 'fit.json')['seconds_total'],
                      'best_epoch': read_json(roots[s] / str(s) / 'fit.json')['report']['best_epoch'],
                      'predict': read_json(roots[s] / str(s) / 'prediction_runtime.json')['seconds'],
                      'mlb_predict': read_json(roots[s] / str(s) / 'mlb_prediction_runtime.json')['seconds'],
                      'source_run': roots[s].name} for s in SEEDS}
    result = {'experiment_id': EXPERIMENT, 'confirms': config['confirms'], 'baseline': config['baseline'],
              'seeds': list(SEEDS), 'new_fit_seeds': list(NEW_SEEDS), 'mix': config['mix'],
              'git_commit': git_commit(), 'preparation_sha256': hash_file(output / 'preparation.json'),
              'parent_preparation_sha256': config['parent_preparation_sha256'],
              'frozen_direct_preparation_sha256': config['frozen_direct_run']['preparation_sha256'],
              'data_identity': read_json(parent / 'parent_preparation.json')['dataset_identity'],
              'new_fits': len(NEW_SEEDS), **result, 'costs_seconds': costs, 'policy_effect': None,
              'limits': ['Prediction only; no policy claim',
                         'DEV previously exposed, including the three-seed result of this same candidate: '
                         'this checks seed stability and the whole-MLB population, not new data',
                         'Pitch-type-only pre-pitch marginal; no target-location input',
                         'Conditional bootstrap excludes training/calibration uncertainty',
                         'The mix still needs the 400-draw G0 integration; no inference cost reduction'],
              'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat()}
    destination.mkdir()
    dump(destination / 'results.json', result)
    repo_result.parent.mkdir(exist_ok=True)
    shutil.copyfile(destination / 'results.json', repo_result)
    print('DIRECT_CONFIRM_SCORED', result['confirmation'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
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
            run(config, local, output, base.identity(config, args.local_config), smoke)
        else:
            score(config, args.local_config, output)


if __name__ == '__main__':
    main()
