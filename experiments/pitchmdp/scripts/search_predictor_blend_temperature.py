"""DRAFT exploratory predictor search: direct-head weight x scalar temperature (77 fixed combos).

Additive runner.  No fitting, no inference: it reuses the sealed <=2025 May-calibrated
probabilities of G0-global (EXP-P4-001 / EXP-P10-001 / EXP-P11-001) and the direct
pre-pitch head (EXP-P15-001 / EXP-P16-001), and scores a pinned grid at the
EXP-P15-002 mix stage:

  member_s(w, T) = temper(w * direct_s + (1 - w) * G0_s, T),   temper(p, T) ~ p ** (1 / T)

then the unchanged five-seed mean and June frequency blend (score_ml_matrix.summarize_cell,
weight selected on June Cpanel rows only) for the Cpanel DEV, and the EXP-P16-001
whole-MLB construction (each candidate's own Cpanel June weight against the frozen
frequency baseline).  w = 0 / 1 / .5 at T = 1 are G0-global, G15-direct and the frozen
G15 mix; they must reproduce results/EXP-P16-001.json before any candidate is scored.

This is repeated-DEV exploration: descriptive rankings only, no bootstrap, no p-values,
no promotion.  ``run`` needs a REGISTERED config with a PM-assigned ID and writes a fresh
output; ``smoke`` exercises the same code on synthetic arrays only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import itertools
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
from pitchmdp.data import hash_file
from pitchmdp.matrix_metrics import pitch_losses, prediction_metrics
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock
from run_ml_delivery_pool import git_commit, peak
from run_ml_direct_head import mixed, SOURCES as DIRECT_SOURCES
from run_ml_direct_head_confirm import member
from score_ml_matrix import assert_aligned, summarize_cell

NAME = 'scripts/search_predictor_blend_temperature.py'
PROTOCOL = 'predictor_blend_temperature_search_v1'
SEEDS = (0, 1, 2, 3, 4)
WEIGHTS = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)
TEMPERATURES = (.8, .9, .95, 1, 1.05, 1.1, 1.2)
BASELINES = {'G0-global': (0, 1), 'G15-direct': (1, 1), 'G15-direct-g0-mix': (.5, 1)}
G0, G15 = 'G0-global', 'G15-direct-g0-mix'
ID = re.compile(r'^EXP-P\d+-\d{3}$')


def config_check(config, full):
    if config['protocol'] != PROTOCOL or config['seeds'] != list(SEEDS):
        raise ValueError('Search settings differ from the pinned protocol')
    grid = config['grid']
    if tuple(grid['direct_weight']) != WEIGHTS or tuple(grid['temperature']) != TEMPERATURES or grid['count'] != 77:
        raise ValueError('The 11 x 7 grid is pinned')
    if {k: (v['direct_weight'], v['temperature']) for k, v in config['baselines'].items()} != BASELINES:
        raise ValueError('Baseline identities are pinned')
    if full and not (config['status'] == 'REGISTERED' and config['execution']['enabled'] is True
                     and ID.match(str(config.get('experiment_id')))):
        raise ValueError('Real-data run needs a REGISTERED config with a PM-assigned experiment ID')
    return config


def candidates():
    return list(itertools.product(WEIGHTS, TEMPERATURES))


def label(w, t):
    return f'w{w:.1f}_T{t:.2f}'


def temper(p, t):
    """Scalar temperature on log-probabilities (Guo et al. 2017 applied to a probability vector)."""
    if t == 1:
        return p
    q = np.power(p, 1.0 / t)
    return q / q.sum(axis=1, keepdims=True)


def candidate_members(direct, g0, w, t):
    members = mixed(direct, g0, w, parts=('blend', 'dev'))
    for m in members:
        for part in ('blend', 'dev'):
            m[part] = temper(m[part], t)
    return members


def mean_losses(y, p):
    return pitch_losses(y, p).mean(axis=0)


# ------------------------------------------------------------------ Cpanel
def cpanel(direct, g0, baseline, w, t, periods):
    """Exact EXP-P16-001 Cpanel path for one (w, T): five-seed mean, June blend, per-seed rows."""
    report, pred = summarize_cell(candidate_members(direct, g0, w, t), baseline)
    y = baseline['dev_y']
    groups = {}
    for name in np.unique(periods):
        mask = periods == name
        groups[str(name)] = {'n': int(mask.sum()), 'loss_sum': pitch_losses(y[mask], pred['primary'][mask]).sum(0)}
    return {'metrics': report['primary'], 'june_weight': report['selection']['model_weight'],
            'seed_june_weights': [s['blend_selection']['model_weight'] for s in report['seeds']],
            'mean': mean_losses(y, pred['primary']),
            'seed_mean': np.stack([mean_losses(y, p) for p in pred['seed_primary']]), 'periods': groups}


# ------------------------------------------------------------------ whole MLB (streamed)
def mlb_stream(direct_mlb, g0_mlb, frequency, combos, weights, periods, chunk_rows, deadline=None):
    """One deterministic pass over row chunks for all combos.  weights[c] = (ensemble, per-seed) June weights."""
    y, n = g0_mlb[0]['dev_y'], len(frequency)
    names = np.unique(periods)
    code = np.searchsorted(names, periods)
    out = {c: {'loss': np.zeros(2), 'seed_loss': np.zeros((len(SEEDS), 2)),
               'period_loss': np.zeros((len(names), 2)),
               'bin_n': np.zeros(10), 'bin_conf': np.zeros(10), 'bin_correct': np.zeros(10)} for c in combos}
    for start in range(0, n, chunk_rows):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('max_wall_seconds exceeded; no result written')
        rows = slice(start, min(start + chunk_rows, n))
        yc, fc, gc = y[rows], frequency[rows], code[rows]
        d = [m['mlb_dev'][rows] for m in direct_mlb]
        g = [m['dev'][rows] for m in g0_mlb]
        for c in combos:
            w, t = c
            ew, sw = weights[c]
            members = [temper(w * a + (1 - w) * b, t) for a, b in zip(d, g)]
            blended = ew * np.mean(members, axis=0) + (1 - ew) * fc
            losses = pitch_losses(yc, blended)
            acc = out[c]
            acc['loss'] += losses.sum(0)
            for k in range(2):
                acc['period_loss'][:, k] += np.bincount(gc, losses[:, k], minlength=len(names))
            for i, (s, m) in enumerate(zip(sw, members)):
                acc['seed_loss'][i] += pitch_losses(yc, s * m + (1 - s) * fc).sum(0)
            conf = blended.max(1)
            b = np.minimum((conf * 10).astype(int), 9)
            acc['bin_n'] += np.bincount(b, minlength=10)
            acc['bin_conf'] += np.bincount(b, conf, minlength=10)
            acc['bin_correct'] += np.bincount(b, (blended.argmax(1) == yc).astype(float), minlength=10)
    counts = np.bincount(code, minlength=len(names))
    for acc in out.values():
        acc['mean'] = acc['loss'] / n
        acc['seed_mean'] = acc['seed_loss'] / n
        nz = acc['bin_n'] > 0
        acc['ece10'] = float(np.sum(acc['bin_n'][nz] / n * np.abs(acc['bin_correct'][nz] / acc['bin_n'][nz]
                                                                   - acc['bin_conf'][nz] / acc['bin_n'][nz])))
        acc['periods'] = {str(k): {'n': int(cnt), 'loss_sum': acc['period_loss'][i]}
                          for i, (k, cnt) in enumerate(zip(names, counts))}
    return out


# ------------------------------------------------------------------ gate
def reproduction_gate(cp, mlb, reference, tol):
    """Compare the three T = 1 baselines with the sealed EXP-P16-001 numbers."""
    checks = []

    def check(name, ours, theirs):
        ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
        diff = float(np.max(np.abs(ours - theirs))) if ours.shape == theirs.shape else float('inf')
        checks.append({'check': name, 'max_abs_diff': diff, 'passed': diff <= tol})

    rep, comp = reference['cpanel_dev']['reports'], reference['cpanel_dev']['comparison']
    for arm, combo in BASELINES.items():
        check(f'cpanel/{arm}/log_loss', cp[combo]['mean'][0], rep[arm]['primary']['log_loss'])
        check(f'cpanel/{arm}/brier', cp[combo]['mean'][1], rep[arm]['primary']['brier_multiclass'])
        check(f'cpanel/{arm}/june_weight', cp[combo]['june_weight'], rep[arm]['selection']['model_weight'])
    g0 = BASELINES[G0]
    blended = reference['mlb_dev_blended']['comparison']
    for arm in ('G15-direct', G15):
        combo = BASELINES[arm]
        check(f'cpanel/{arm}/seed_deltas_vs_G0', cp[combo]['seed_mean'][:, 0] - cp[g0]['seed_mean'][:, 0],
              comp[arm]['seed_deltas'])
        row = blended[arm]
        check(f'mlb/{arm}/log_loss', mlb[combo]['mean'][0], row['candidate']['log_loss'])
        check(f'mlb/{arm}/brier', mlb[combo]['mean'][1], row['candidate']['brier_multiclass'])
        check(f'mlb/{arm}/control_log_loss', mlb[g0]['mean'][0], row['control']['log_loss'])
        check(f'mlb/{arm}/control_brier', mlb[g0]['mean'][1], row['control']['brier_multiclass'])
        check(f'mlb/{arm}/seed_deltas_vs_G0', mlb[combo]['seed_mean'][:, 0] - mlb[g0]['seed_mean'][:, 0],
              row['seed_deltas'])
    return {'passed': all(c['passed'] for c in checks), 'abs_tolerance': tol, 'checks': checks}


# ------------------------------------------------------------------ summaries
def period_table(stats, controls):
    table = {}
    for name, row in stats['periods'].items():
        n = row['n']
        table[name] = {'n': n, 'nll': float(row['loss_sum'][0] / n), 'brier': float(row['loss_sum'][1] / n),
                       **{f'delta_nll_vs_{k}': float((row['loss_sum'][0] - c['periods'][name]['loss_sum'][0]) / n)
                          for k, c in controls.items()}}
    return table


def describe(stats, controls, cpanel_metrics=None):
    row = {'nll': float(stats['mean'][0]), 'brier': float(stats['mean'][1])}
    for k, c in controls.items():
        row[f'delta_nll_vs_{k}'] = float(stats['mean'][0] - c['mean'][0])
        row[f'delta_brier_vs_{k}'] = float(stats['mean'][1] - c['mean'][1])
        seed = stats['seed_mean'] - c['seed_mean']
        row[f'seed_delta_nll_vs_{k}'] = [float(v) for v in seed[:, 0]]
        row[f'seed_delta_brier_vs_{k}'] = [float(v) for v in seed[:, 1]]
        row[f'seeds_nll_better_than_{k}'] = int((seed[:, 0] < 0).sum())
    row['ece10'] = cpanel_metrics['top_label_ece10'] if cpanel_metrics else stats['ece10']
    row['periods'] = period_table(stats, controls)
    return row


def rank(rows):
    keys = [('cpanel', 'delta_nll_vs_G15'), ('mlb', 'delta_nll_vs_G15'),
            ('cpanel', 'delta_brier_vs_G15'), ('mlb', 'delta_brier_vs_G15')]
    names = list(rows)
    values = np.array([[rows[n][pop][k] for pop, k in keys] for n in names])
    ranks = values.argsort(0, kind='stable').argsort(0, kind='stable') + 1
    pareto = [not any(np.all(values[j] <= values[i]) and np.any(values[j] < values[i])
                      for j in range(len(names)) if j != i) for i in range(len(names))]
    for i, n in enumerate(names):
        rows[n]['rank'] = {'cpanel_nll': int(ranks[i, 0]), 'mlb_nll': int(ranks[i, 1]),
                           'cpanel_brier': int(ranks[i, 2]), 'mlb_brier': int(ranks[i, 3]),
                           'mean_nll_rank': float(ranks[i, :2].mean()), 'pareto_four_objective': bool(pareto[i])}
    return sorted(names, key=lambda n: (rows[n]['rank']['mean_nll_rank'], n))


def coverage(y, games, pitchers, periods):
    names, counts = np.unique(periods, return_counts=True)
    return {'n': len(y), 'games': int(len(np.unique(games))), 'pitchers': int(len(np.unique(pitchers))),
            'class_support': [int(v) for v in np.bincount(y, minlength=10)], 'rows_dropped': 0,
            'periods': {str(k): {'n': int(c), 'games': int(len(np.unique(games[periods == k])))}
                        for k, c in zip(names, counts)}}


def search(direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb, reference,
           tol=1e-9, chunk_rows=65536, deadline=None):
    """Pure search on loaded arrays (synthetic-testable).  Gate first; candidates only if it passes."""
    if not len(direct) == len(g0) == len(direct_mlb) == len(g0_mlb) == len(SEEDS):
        raise ValueError('All five seeds of every arm are required')
    for item in (*direct, *g0):
        assert_aligned(item, baseline)
    y, games, keys = g0_mlb[0]['dev_y'], g0_mlb[0]['dev_game_pk'], g0_mlb[0]['dev_keys']
    for m in direct_mlb:
        if not (np.array_equal(m['mlb_dev_keys'], keys) and np.array_equal(m['mlb_dev_y'], y)):
            raise ValueError('Unpaired whole-MLB predictions')
    for m in g0_mlb:
        if not (np.array_equal(m['dev_keys'], keys) and np.array_equal(m['dev_y'], y)):
            raise ValueError('Unpaired G0 whole-MLB members')
    if not np.array_equal(frequency['mlb_dev_keys'], keys):
        raise ValueError('Unpaired whole-MLB frequency baseline')
    if len(periods_cp) != len(baseline['dev_y']) or len(periods_mlb) != len(y):
        raise ValueError('Period labels must cover every DEV row')
    f = frequency['mlb_dev']

    def run(combos):
        cp = {c: cpanel(direct, g0, baseline, *c, periods_cp) for c in combos}
        weights = {c: (cp[c]['june_weight'], cp[c]['seed_june_weights']) for c in combos}
        return cp, mlb_stream(direct_mlb, g0_mlb, f, combos, weights, periods_mlb, chunk_rows, deadline)

    cp, mlb = run(list(BASELINES.values()))
    gate = reproduction_gate(cp, mlb, reference, tol)
    result = {'reproduction_gate': gate,
              'coverage': {'cpanel_dev': coverage(baseline['dev_y'], baseline['dev_game_pk'], baseline['dev_pitcher'], periods_cp),
                           'whole_mlb_dev': coverage(y, games, g0_mlb[0]['dev_pitcher'], periods_mlb)},
              'seeds': list(SEEDS)}
    if not gate['passed']:
        return result
    cp, mlb = run(candidates())
    controls = {'G0': BASELINES[G0], 'G15': BASELINES[G15]}
    rows = {}
    for c in candidates():
        rows[label(*c)] = {'direct_weight': c[0], 'temperature': c[1],
                           'is_baseline': next((k for k, v in BASELINES.items() if v == c), None),
                           'june_weight': cp[c]['june_weight'], 'seed_june_weights': cp[c]['seed_june_weights'],
                           'cpanel': describe(cp[c], {k: cp[v] for k, v in controls.items()}, cp[c]['metrics']),
                           'mlb': describe(mlb[c], {k: mlb[v] for k, v in controls.items()})}
    order = rank(rows)
    result.update({'candidates': rows, 'order_by_mean_nll_rank': order, 'candidate_count': len(rows),
                   'pareto_four_objective': [n for n in order if rows[n]['rank']['pareto_four_objective']]})
    return result


# ------------------------------------------------------------------ real-data loading
def load_member(spec):
    root = Path(spec['dir'])
    if hash_file(root / spec['state']) != spec['state_sha256']:
        raise ValueError('Pinned member state changed: ' + str(root / spec['state']))
    return member(root.parent, root.name, spec['state'])


def load_npz(spec, keys):
    if hash_file(Path(spec['path'])) != spec['sha256']:
        raise ValueError('Pinned frequency archive changed: ' + spec['path'])
    with np.load(spec['path'], allow_pickle=False) as data:
        return {k: data[k].copy() for k in keys}


def game_periods(local, config, *game_arrays):
    """game_pk -> YYYY-MM from the pinned <=2025 raw files (two columns only)."""
    import pyarrow.parquet as pq
    allow, root = set(local['raw_allowlist']), Path(local['raw_root'])
    frames = []
    for spec in config['periods']['source_files']:
        name = spec['file']
        if '2026' in name or name not in allow:
            raise ValueError('Period source must be an allowlisted <=2025 file: ' + name)
        if hash_file(root / name) != spec['sha256']:
            raise ValueError('Period source changed: ' + name)
        table = pq.read_table(root / name, columns=['game_pk', 'game_date'])
        frames.append((np.asarray(table['game_pk'].to_numpy(), np.int64),
                       np.asarray(table['game_date'].cast('string').to_numpy(zero_copy_only=False), str)))
    pk = np.concatenate([a for a, _ in frames])
    month = np.char.ljust(np.concatenate([b for _, b in frames]).astype('<U7'), 7)
    pairs = np.unique(np.stack([pk.astype(str), month], 1), axis=0)
    if len(np.unique(pairs[:, 0])) != len(pairs):
        raise ValueError('A game_pk maps to more than one month')
    lookup = dict(zip(pairs[:, 0].astype(np.int64).tolist(), pairs[:, 1].tolist()))
    out = []
    for games in game_arrays:
        missing = set(np.unique(games).tolist()) - lookup.keys()
        if missing:
            raise ValueError(f'{len(missing)} DEV games have no pinned game_date')
        if any(lookup[g] >= '2026' for g in np.unique(games).tolist()):
            raise ValueError('DEV row dated 2026')
        out.append(np.array([lookup[g] for g in games.tolist()]))
    return out


def real_inputs(config, local):
    inp = config['inputs']
    if hash_file(Path(inp['parent_run']) / 'preparation.json') != inp['parent_preparation_sha256']:
        raise ValueError('Parent preparation changed')
    members = inp['members']
    direct = [load_member(members[str(s)]['direct']) for s in SEEDS]
    g0 = [load_member(members[str(s)]['g0']) for s in SEEDS]
    direct_mlb = [load_member(members[str(s)]['direct_mlb']) for s in SEEDS]
    g0_mlb = [load_member(members[str(s)]['g0_mlb']) for s in SEEDS]
    baseline = load_npz(inp['cpanel_frequency'], [f'{p}{k}' for p in ('blend', 'dev')
                                                  for k in ('', '_raw', '_keys', '_y', '_game_pk', '_pitcher')])
    frequency = load_npz(inp['mlb_frequency'], ('mlb_dev', 'mlb_dev_keys'))
    periods_cp, periods_mlb = game_periods(local, config, baseline['dev_game_pk'], g0_mlb[0]['dev_game_pk'])
    return direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb


# ------------------------------------------------------------------ synthetic smoke
def synthetic(n_blend=600, n_dev=900, n_mlb=4000, seed=7, games=40):
    """Synthetic five-seed archives with the real key/field layout (no real data)."""
    rng = np.random.default_rng(seed)

    def probs(n):
        p = rng.dirichlet(np.ones(10), size=n)
        return p

    def part(n, prefix, offset):
        game = np.sort(rng.integers(0, games, n)) + offset
        keys = np.stack([game, np.arange(n), np.zeros(n, int)], 1)
        return {f'{prefix}_keys': keys, f'{prefix}_y': rng.integers(0, 10, n),
                f'{prefix}_game_pk': game, f'{prefix}_pitcher': rng.integers(0, 5, n)}
    meta = {**part(n_blend, 'blend', 1000), **part(n_dev, 'dev', 2000)}
    baseline = {**meta, **{f'{p}{s}': probs(len(meta[p + '_y'])) for p in ('blend', 'dev') for s in ('', '_raw')}}

    def cell():
        out = []
        for _ in SEEDS:
            out.append({**meta, **{f'{p}{s}': probs(len(meta[p + '_y'])) for p in ('blend', 'dev') for s in ('', '_raw')},
                        'dev_delivery_level': np.zeros(n_dev, int)})
        return out
    direct, g0 = cell(), cell()
    mlb = part(n_mlb, 'dev', 5000)
    g0_mlb = [{**mlb, 'dev': probs(n_mlb), 'dev_raw': probs(n_mlb), 'dev_delivery_level': np.zeros(n_mlb, int)}
              for _ in SEEDS]
    direct_mlb = [{'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev_y': mlb['dev_y'], 'mlb_dev': probs(n_mlb)} for _ in SEEDS]
    frequency = {'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev': np.tile(rng.dirichlet(np.ones(10)), (n_mlb, 1))}
    month = lambda g: np.where(g % 3 == 0, '2025-07', np.where(g % 3 == 1, '2025-08', '2025-09'))
    return direct, g0, baseline, direct_mlb, g0_mlb, frequency, month(meta['dev_game_pk']), month(mlb['dev_game_pk'])


def reference_from(direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb):
    """Build an EXP-P16-001-shaped reference with the independent run_ml_direct_head_confirm.score_arrays."""
    import run_ml_direct_head_confirm as confirm
    import run_ml_direct_head as base
    config = {'metrics': {'N': {'delta_nll_max': -.003, 'brier_ci95_upper_max': .001}}, 'mix': {'direct_weight': .5}}
    # Bootstrap-free stand-ins: the reference needs the point metrics and seed deltas only.
    saved = base.paired, confirm.decide, confirm.tier_slices
    base.paired = lambda y, games, a, b, sa, sb: {
        'paired': {'nll': {'p_less': 1.0}}, 'seed_deltas': [float((pitch_losses(y, u) - pitch_losses(y, v))[:, 0].mean())
                                                           for u, v in zip(sa, sb)],
        'candidate': prediction_metrics(y, a), 'control': prediction_metrics(y, b)}
    confirm.decide = lambda rows, n_rule, holm: {k: {**v, 'N': {'status': None}} for k, v in rows.items()}
    confirm.tier_slices = lambda *a: None
    try:
        return confirm.score_arrays(direct, g0, baseline, config, direct_mlb, g0_mlb, frequency)
    finally:
        base.paired, confirm.decide, confirm.tier_slices = saved


def write_result(output, result, config, extra):
    output.mkdir(parents=True)
    dump(output / 'registered_config.json', config)
    dump(output / 'results.json', {**extra, **result})


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def provenance(config_path):
    return {'runner': NAME, 'runner_sha256': hash_file(PROJECT / NAME), 'config_sha256': hash_file(config_path),
            'git_commit': git_commit(), 'python': platform.python_version(), 'numpy': np.__version__,
            'source_hashes': {name: hash_file(PROJECT / name) for name in sorted(set([*DIRECT_SOURCES, NAME,
                'scripts/run_ml_direct_head_confirm.py', 'scripts/score_ml_matrix.py']))},
            'started_utc': datetime.now(timezone.utc).isoformat()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('command', choices=('smoke', 'run'))
    args = p.parse_args()
    full = args.command == 'run'
    config, local = config_check(read_json(args.config), full), read_json(args.local_config)
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output exists; refusing to overwrite')
    started = time.monotonic()
    deadline = started + config['execution']['max_wall_seconds']
    meta = provenance(args.config)
    meta['local_config_sha256'] = hash_file(args.local_config)
    if not full:
        if not output.name.endswith('-smoke'):
            raise ValueError('Smoke output must end -smoke')
        arrays = synthetic()
        reference = jsonable(reference_from(*arrays))
        result = search(*arrays, reference, config['reproduction_gate']['abs_tolerance'], 1000, deadline)
        write_result(output, jsonable(result), config, {'mode': 'synthetic_smoke', 'provenance': meta,
                     'seconds': time.monotonic() - started, 'peak_rss_bytes': peak()})
        print('SEARCH_SMOKE_DONE', {'gate': result['reproduction_gate']['passed'],
                                    'candidates': result.get('candidate_count')}, flush=True)
        return
    if output.name != config['experiment_id']:
        raise ValueError('Output directory must be named after the assigned experiment ID')
    root = check_location(local, output)
    validate_native_runtime()
    ref_path = REPO / config['reproduction_gate']['reference']
    if hash_file(ref_path) != config['reproduction_gate']['reference_sha256']:
        raise ValueError('Reproduction reference changed')
    with heavy_lock(root):
        before = time.monotonic()
        arrays = real_inputs(config, local)
        load_seconds = time.monotonic() - before
        result = search(*arrays, read_json(ref_path), config['reproduction_gate']['abs_tolerance'],
                        config['execution']['chunk_rows'], deadline)
        extra = {'experiment_id': config['experiment_id'], 'mode': 'exploratory_repeated_dev',
                 'nature': config['nature'], 'provenance': meta,
                 'inputs': config['inputs'], 'reproduction_reference_sha256': config['reproduction_gate']['reference_sha256'],
                 'seconds': {'load': load_seconds, 'total': time.monotonic() - started}, 'peak_rss_bytes': peak(),
                 'limits': ['Exploratory search on repeatedly exposed 2025 DEV; no promotion, no p-values',
                            'Rankings are descriptive and subject to 77-way selection optimism',
                            'Prediction only; no policy claim (D155 stays closed)']}
        write_result(output, jsonable(result), config, extra)
        repo_result = REPO / 'results' / (config['experiment_id'] + '.json')
        if not repo_result.exists():
            shutil.copyfile(output / 'results.json', repo_result)
    if not result['reproduction_gate']['passed']:
        raise ValueError('Reproduction gate failed; saved diagnostic result, no candidate search')
    print('SEARCH_DONE', {'gate': result['reproduction_gate']['passed'], 'candidates': result.get('candidate_count')},
          flush=True)


if __name__ == '__main__':
    main()
