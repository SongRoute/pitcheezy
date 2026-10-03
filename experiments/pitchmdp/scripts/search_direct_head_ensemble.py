"""DRAFT exploratory search: equal-probability direct-head ensembles x G0 mix weight (66 fixed candidates).

Additive runner.  No fitting, no inference, no temperature grid: it reuses the saved,
already May-calibrated seed-0/1/2 probabilities of six direct pre-pitch heads

  P15-uniform (frozen EXP-P15-001), P20-hl180 / hl365 / hl730 (EXP-P20-001),
  P21-residual (EXP-P21-001), P22-periodic (EXP-P22-001)

and of the frozen G0-global members.  For every seed s and every fixed subset S
(6 singletons + 15 unordered pairs + the all-six set = 22) and direct weight w in
{.25, .5, .75} (66 candidates):

  member_s(S, w) = w * mean_{k in S} direct_{k,s} + (1 - w) * G0_s     (same seed only)

then the unchanged three-seed mean and June frequency blend of the P19 path
(search_predictor_blend_temperature.cpanel -> score_ml_matrix.summarize_cell) on
Cpanel DEV, and on whole-MLB DEV the same members blended with the frozen frequency
baseline using the candidate's own Cpanel June weight (P15/P16/P19 construction; this
differs from the unblended whole-MLB secondary of the P20-P22 runners).  The frozen
three-seed G0, uniform direct and uniform 1/2 mix must reproduce results/EXP-P15-001.json
before any candidate is scored.  ({P15-uniform}, .5) is the frozen control row.

Repeated-DEV exploration: all candidates kept, descriptive ranks only, no bootstrap,
no p-values, no winner.  ``pins`` reads only metadata/state JSON of finished families
and prints the hashes the root pins before registration; ``run`` needs a REGISTERED
config with every pin filled; ``smoke`` uses synthetic arrays only.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import re
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_metrics import pitch_losses, prediction_metrics
from run_ml_benchmark import read_json, validate_native_runtime
from run_ml_matrix import assert_hashes, check_location, heavy_lock
from run_ml_delivery_pool import peak
from run_ml_direct_head_confirm import member
from score_ml_matrix import assert_aligned
import search_predictor_blend_temperature as p19

NAME = 'scripts/search_direct_head_ensemble.py'
PROTOCOL = 'direct_head_ensemble_search_v1'
SEEDS = (0, 1, 2)
FAMILIES = ('P15-uniform', 'P20-hl180', 'P20-hl365', 'P20-hl730', 'P21-residual', 'P22-periodic')
WEIGHTS = (.25, .5, .75)
CONTROL = (('P15-uniform',), .5)
G0_KEY = (('P15-uniform',), 0)          # w = 0: the subset is irrelevant, pure G0-global
DIRECT_KEY = (('P15-uniform',), 1)      # gate-only arm: frozen uniform direct head alone
GATE = {'G0-global': G0_KEY, 'G15-direct': DIRECT_KEY, 'G15-direct-g0-mix': CONTROL}
PARTS = ('blend', 'blend_raw', 'dev', 'dev_raw')
MLB_KEEP = ('mlb_dev', 'mlb_dev_keys', 'mlb_dev_y')
STATES = {'fit_state.json': ('model.pt', 'fit.json'),
          'prediction_state.json': ('predictions.npz', 'calibration.json'),
          'mlb_prediction_state.json': ('mlb_predictions.npz',)}
OWN_SOURCE_FIELD = {'ml_direct_recency_v1': None, 'ml_direct_residual_v1': 'residual_source_hashes',
                    'ml_direct_periodic_v1': 'periodic_source_hashes'}
ID = re.compile(r'^EXP-P\d+-\d{3}$')


# ------------------------------------------------------------------ grid
def subsets():
    return [(f,) for f in FAMILIES] + list(itertools.combinations(FAMILIES, 2)) + [FAMILIES]


def candidates():
    return [(s, w) for s in subsets() for w in WEIGHTS]


def label(subset, w):
    return ('all6' if subset == FAMILIES else '+'.join(subset)) + f'_w{w:.2f}'


def config_check(config, full):
    if config['protocol'] != PROTOCOL or config['seeds'] != list(SEEDS):
        raise ValueError('Search settings differ from the pinned protocol')
    grid = config['grid']
    if tuple(grid['families']) != FAMILIES or tuple(grid['direct_weight']) != WEIGHTS \
            or grid['subset_count'] != 22 or grid['count'] != 66 or grid['temperature'] is not None:
        raise ValueError('The 22-subset x 3-weight grid is pinned (no temperature)')
    if [f['name'] for f in config['families']] != list(FAMILIES):
        raise ValueError('Family list/order is pinned')
    if full:
        if not (config['status'] == 'REGISTERED' and config['execution']['enabled'] is True
                and ID.match(str(config.get('experiment_id')))):
            raise ValueError('Real-data run needs a REGISTERED config with a root-assigned experiment ID')
        for f in config['families']:
            if not f.get('preparation_sha256') or any(not f.get('state_sha256', {}).get(str(s), {}).get(n)
                                                      for s in SEEDS for n in STATES):
                raise ValueError(f"{f['name']}: preparation/state hashes must be pinned before a run")
    return config


# ------------------------------------------------------------------ ensemble semantics
def average(by_family, subset, parts=PARTS):
    """Equal-probability mean of the subset's members, seed by seed (never across seeds)."""
    first = by_family[subset[0]]
    if len(subset) == 1:
        return first
    out = []
    for i in range(len(SEEDS)):
        m = dict(first[i])
        for part in parts:
            m[part] = np.mean([by_family[f][i][part] for f in subset], axis=0)
        out.append(m)
    return out


def check_probabilities(name, p, n=None):
    if not (isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 10 and (n is None or len(p) == n)):
        raise ValueError(f'{name}: not an (n, 10) probability array')
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.max(np.abs(p.sum(1) - 1)) > 1e-6:
        raise ValueError(f'{name}: not a probability array')


def validate(direct, g0, baseline, direct_mlb, g0_mlb, frequency):
    """Every component: three seeds, frozen Cpanel keys/labels, whole-MLB keys/labels, probabilities."""
    if set(direct) != set(FAMILIES) or set(direct_mlb) != set(FAMILIES):
        raise ValueError('All six direct families are required')
    if not len(g0) == len(g0_mlb) == len(SEEDS) or any(len(direct[f]) != len(SEEDS) or len(direct_mlb[f]) != len(SEEDS)
                                                         for f in FAMILIES):
        raise ValueError('Exactly three seeds of every component are required')
    for name, items in [('G0', g0), *direct.items()]:
        for i, m in enumerate(items):
            assert_aligned(m, baseline)
            for part in PARTS:
                check_probabilities(f'{name}/seed{SEEDS[i]}/{part}', m[part], len(baseline[part.split('_')[0] + '_y']))
    y, keys = g0_mlb[0]['dev_y'], g0_mlb[0]['dev_keys']
    for i, m in enumerate(g0_mlb):
        if not (np.array_equal(m['dev_keys'], keys) and np.array_equal(m['dev_y'], y)):
            raise ValueError('Unpaired G0 whole-MLB members')
        check_probabilities(f'G0-mlb/seed{SEEDS[i]}', m['dev'], len(y))
    for name, items in direct_mlb.items():
        for i, m in enumerate(items):
            if not (np.array_equal(m['mlb_dev_keys'], keys) and np.array_equal(m['mlb_dev_y'], y)):
                raise ValueError(f'{name}/seed{SEEDS[i]}: unpaired whole-MLB predictions')
            check_probabilities(f'{name}-mlb/seed{SEEDS[i]}', m['mlb_dev'], len(y))
    if not np.array_equal(frequency['mlb_dev_keys'], keys):
        raise ValueError('Unpaired whole-MLB frequency baseline')
    check_probabilities('frequency-mlb', frequency['mlb_dev'], len(y))


# ------------------------------------------------------------------ scoring
def cpanel_all(direct, g0, baseline, keys, periods):
    """Exact P19 Cpanel path (T = 1) for each (subset, w): three-seed mean, June blend, per-seed rows."""
    return {k: p19.cpanel(average(direct, k[0]), g0, baseline, k[1], 1, periods) for k in keys}


def mlb_stream(direct_mlb, g0_mlb, frequency, keys, weights, periods, chunk_rows, deadline=None):
    """One pass over row chunks.  weights[k] = (ensemble, per-seed) Cpanel June weights of that candidate."""
    y, n = g0_mlb[0]['dev_y'], len(frequency)
    names = np.unique(periods)
    code = np.searchsorted(names, periods)
    out = {k: {'loss': np.zeros(2), 'seed_loss': np.zeros((len(SEEDS), 2)), 'period_loss': np.zeros((len(names), 2)),
               'bin_n': np.zeros(10), 'bin_conf': np.zeros(10), 'bin_correct': np.zeros(10)} for k in keys}
    for start in range(0, n, chunk_rows):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('max_wall_seconds exceeded; no result written')
        rows = slice(start, min(start + chunk_rows, n))
        yc, fc, gc = y[rows], frequency[rows], code[rows]
        g = [m['dev'][rows] for m in g0_mlb]
        chunk = {f: [{'mlb_dev': m['mlb_dev'][rows]} for m in direct_mlb[f]] for f in FAMILIES}
        cache = {}
        for k in keys:
            subset, w = k
            if subset not in cache:
                cache[subset] = [m['mlb_dev'] for m in average(chunk, subset, ('mlb_dev',))]
            members = [w * a + (1 - w) * b for a, b in zip(cache[subset], g)]
            ew, sw = weights[k]
            blended = ew * np.mean(members, axis=0) + (1 - ew) * fc
            losses = pitch_losses(yc, blended)
            acc = out[k]
            acc['loss'] += losses.sum(0)
            for j in range(2):
                acc['period_loss'][:, j] += np.bincount(gc, losses[:, j], minlength=len(names))
            for i, (s, m) in enumerate(zip(sw, members)):
                acc['seed_loss'][i] += pitch_losses(yc, s * m + (1 - s) * fc).sum(0)
            conf = blended.max(1)
            b = np.minimum((conf * 10).astype(int), 9)
            acc['bin_n'] += np.bincount(b, minlength=10)
            acc['bin_conf'] += np.bincount(b, conf, minlength=10)
            acc['bin_correct'] += np.bincount(b, (blended.argmax(1) == yc).astype(float), minlength=10)
    counts = np.bincount(code, minlength=len(names))
    for acc in out.values():
        acc['mean'], acc['seed_mean'] = acc['loss'] / n, acc['seed_loss'] / n
        nz = acc['bin_n'] > 0
        acc['ece10'] = float(np.sum(acc['bin_n'][nz] / n * np.abs(acc['bin_correct'][nz] / acc['bin_n'][nz]
                                                                   - acc['bin_conf'][nz] / acc['bin_n'][nz])))
        acc['periods'] = {str(k): {'n': int(c), 'loss_sum': acc['period_loss'][i]} for i, (k, c) in enumerate(zip(names, counts))}
    return out


def search(direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb, reference,
           tol=1e-9, chunk_rows=65536, deadline=None):
    """Pure search on loaded arrays (synthetic-testable).  Gate first; candidates only if it passes."""
    validate(direct, g0, baseline, direct_mlb, g0_mlb, frequency)
    y = g0_mlb[0]['dev_y']
    if len(periods_cp) != len(baseline['dev_y']) or len(periods_mlb) != len(y):
        raise ValueError('Period labels must cover every DEV row')
    f = frequency['mlb_dev']

    def run(keys):
        cp = cpanel_all(direct, g0, baseline, keys, periods_cp)
        weights = {k: (cp[k]['june_weight'], cp[k]['seed_june_weights']) for k in keys}
        return cp, mlb_stream(direct_mlb, g0_mlb, f, keys, weights, periods_mlb, chunk_rows, deadline)

    cp, mlb = run(list(GATE.values()))
    # p19.reproduction_gate addresses arms by its (w, T) keys; map ours onto them.
    alias = {p19.BASELINES[arm]: key for arm, key in GATE.items()}
    gate = p19.reproduction_gate({a: cp[k] for a, k in alias.items()}, {a: mlb[k] for a, k in alias.items()}, reference, tol)
    result = {'reproduction_gate': gate, 'seeds': list(SEEDS),
              'coverage': {'cpanel_dev': p19.coverage(baseline['dev_y'], baseline['dev_game_pk'], baseline['dev_pitcher'], periods_cp),
                           'whole_mlb_dev': p19.coverage(y, g0_mlb[0]['dev_game_pk'], g0_mlb[0]['dev_pitcher'], periods_mlb)}}
    if not gate['passed']:
        return result
    grid = candidates()
    ccp, cmlb = run(grid)
    for k in (CONTROL,):     # the control row must be the gate's frozen mix, bit for bit
        if not (np.array_equal(ccp[k]['mean'], cp[k]['mean']) and np.array_equal(cmlb[k]['mean'], mlb[k]['mean'])):
            raise ValueError('Control candidate differs from the reproduced frozen mix')
    controls = {'G0': (cp[G0_KEY], mlb[G0_KEY]), 'G15': (cp[CONTROL], mlb[CONTROL])}
    rows = {}
    for k in grid:
        subset, w = k
        rows[label(*k)] = {'subset': list(subset), 'subset_size': len(subset), 'direct_weight': w,
                           'is_control': k == CONTROL, 'june_weight': ccp[k]['june_weight'],
                           'seed_june_weights': ccp[k]['seed_june_weights'],
                           'cpanel': p19.describe(ccp[k], {n: c[0] for n, c in controls.items()}, ccp[k]['metrics']),
                           'mlb': p19.describe(cmlb[k], {n: c[1] for n, c in controls.items()})}
    order = p19.rank(rows)
    result.update({'candidates': rows, 'order_by_mean_nll_rank': order, 'candidate_count': len(rows),
                   'pareto_four_objective': [n for n in order if rows[n]['rank']['pareto_four_objective']],
                   'controls': {'G0': 'frozen G0-global three-seed mean, own Cpanel June blend',
                                'G15': 'frozen EXP-P15-001 uniform direct + G0, member-wise 1/2, seeds 0-2 (row ' + label(*CONTROL) + ')'},
                   'mlb_scope': 'whole-MLB DEV after each candidate\'s own Cpanel June three-seed weight (P15/P16/P19 '
                                'construction); NOT the unblended calibrated-mean secondary of the P20-P22 runners'})
    return result


# ------------------------------------------------------------------ fail-closed family verification
def verify_seed(name, root, prep, seed, pins=None):
    dest, prep_hash = root / str(seed), canonical_hash(prep)
    for state_name, required in STATES.items():
        path = dest / state_name
        if not path.is_file():
            raise ValueError(f'{name}: seed {seed} incomplete ({state_name} missing)')
        state = read_json(path)
        ident = state.get('identity', {})
        if ident.get('smoke') is not False:
            raise ValueError(f'{name}: seed {seed} {state_name} is a smoke or unlabelled state')
        if ident.get('seed') != seed or ident.get('preparation_sha256') != prep_hash:
            raise ValueError(f'{name}: seed {seed} {state_name} belongs to another preparation/seed')
        if state_name != 'fit_state.json' and ident.get('model_sha256') != hash_file(dest / 'model.pt'):
            raise ValueError(f'{name}: seed {seed} predictions were made by another model')
        if not set(required) <= set(state.get('artifact_hashes', {})):
            raise ValueError(f'{name}: seed {seed} {state_name} lacks required artifacts')
        assert_hashes(dest, state['artifact_hashes'])
        pinned = (pins or {}).get(str(seed), {}).get(state_name)
        if pinned and hash_file(path) != pinned:
            raise ValueError(f'{name}: seed {seed} {state_name} differs from the pinned state')
    device = read_json(dest / 'fit.json').get('report', {}).get('device')
    if device != 'mps':
        raise ValueError(f'{name}: seed {seed} fitted on {device!r}; only MPS is comparable to the MPS control')


def verify_family(spec, control_prep):
    """Completed, non-smoke, registered, MPS, unchanged-source family of three seeds.  Returns its preparation."""
    root, name = Path(spec['dir']), spec['name']
    if any(part.endswith('-smoke') for part in root.parts):
        raise ValueError(f'{name}: smoke output refused')
    if not (root / 'preparation.json').is_file():
        raise ValueError(f'{name}: no preparation; family not run')
    if spec.get('preparation_sha256') and hash_file(root / 'preparation.json') != spec['preparation_sha256']:
        raise ValueError(f'{name}: preparation differs from the pinned one')
    prep = read_json(root / 'preparation.json')
    ident = prep['identity']
    if spec['protocol'] == 'frozen_control':
        if spec.get('preparation_sha256') is None:
            raise ValueError('The frozen control preparation must be pinned')
    else:
        registered = read_json(root / 'registered_config.json')
        if ident.get('config_sha256') != spec['config_canonical_sha256'] or canonical_hash(registered) != spec['config_canonical_sha256']:
            raise ValueError(f'{name}: family config differs from the registered one')
        if registered['experiment_id'] != spec['experiment_id'] or registered['protocol'] != spec['protocol'] \
                or registered['status'] != 'REGISTERED' or registered['control']['preparation_sha256'] != control_prep['sha256']:
            raise ValueError(f'{name}: registration/protocol/control identity differs')
        if 'registered_config.json' not in prep['artifact_hashes']:
            raise ValueError(f'{name}: registered config not covered by preparation hashes')
        if ident['source_hashes'] != control_prep['prep']['identity']['source_hashes']:
            raise ValueError(f'{name}: shared direct-head source differs from the frozen control')
        snapshot = {rel: prep['artifact_hashes'].get('source/' + rel) for rel in (*ident['source_hashes'], *spec['own_source_sha256'])}
        expected = {**ident['source_hashes'], **spec['own_source_sha256']}
        if snapshot != expected:
            raise ValueError(f'{name}: original source snapshot differs from the pinned sources')
        field = OWN_SOURCE_FIELD[spec['protocol']]
        if field is None:
            if (ident.get('variant'), ident.get('half_life_days'), ident.get('anchor'), ident.get('recency_source_sha256')) != \
                    (spec['variant'], spec['half_life_days'], '2025-04-30', spec['own_source_sha256']['scripts/run_ml_direct_recency.py']):
                raise ValueError(f'{name}: recency variant/source identity differs')
        else:
            arch = registered['candidate_architecture']
            if ident.get(field) != spec['own_source_sha256'] or not ident.get('architecture') \
                    or ident['architecture'] != {k: arch[k] for k in ident['architecture']} \
                    or ('input_map' in ident and ident['input_map'] != arch['input_map']):
                raise ValueError(f'{name}: model source/architecture identity differs')
            if ident.get('resolved_device') != 'mps' or ident.get('device_option') not in ('auto', 'mps') \
                    or prep.get('comparable_device_to_control') is not True:
                raise ValueError(f'{name}: not an MPS family; not comparable to the MPS control')
    assert_hashes(root, prep['artifact_hashes'])
    for seed in SEEDS:
        verify_seed(name, root, prep, seed, spec.get('state_sha256'))
    return prep


def verify_all(config):
    specs = config['families']
    control = specs[0]
    prep = read_json(Path(control['dir']) / 'preparation.json')
    control_prep = {'prep': prep, 'sha256': control['preparation_sha256']}
    return {s['name']: verify_family(s, control_prep) for s in specs}


def pins(config):
    """Hashes of the finished families' preparation and state files (JSON metadata only, no predictions)."""
    verify_all(config)
    out = {}
    for s in config['families']:
        root = Path(s['dir'])
        out[s['name']] = {'preparation_sha256': hash_file(root / 'preparation.json'),
                          'state_sha256': {str(seed): {n: hash_file(root / str(seed) / n) for n in STATES} for seed in SEEDS}}
    return out


# ------------------------------------------------------------------ real-data loading
def real_inputs(config, local):
    verify_all(config)
    inp = config['inputs']
    if hash_file(Path(inp['parent_run']) / 'preparation.json') != inp['parent_preparation_sha256']:
        raise ValueError('Parent preparation changed')
    direct, direct_mlb = {}, {}
    for spec in config['families']:
        root = Path(spec['dir'])
        direct[spec['name']] = [member(root, str(s), 'prediction_state.json') for s in SEEDS]
        direct_mlb[spec['name']] = [{k: v for k, v in member(root, str(s), 'mlb_prediction_state.json').items() if k in MLB_KEEP}
                                    for s in SEEDS]
    g0 = [p19.load_member(inp['g0'][str(s)]) for s in SEEDS]
    g0_mlb = [p19.load_member(inp['g0_mlb'][str(s)]) for s in SEEDS]
    baseline = p19.load_npz(inp['cpanel_frequency'], [f'{p}{k}' for p in ('blend', 'dev')
                                                      for k in ('', '_raw', '_keys', '_y', '_game_pk', '_pitcher')])
    frequency = p19.load_npz(inp['mlb_frequency'], ('mlb_dev', 'mlb_dev_keys'))
    periods_cp, periods_mlb = p19.game_periods(local, config, baseline['dev_game_pk'], g0_mlb[0]['dev_game_pk'])
    return direct, g0, baseline, direct_mlb, g0_mlb, frequency, periods_cp, periods_mlb


# ------------------------------------------------------------------ synthetic smoke
def synthetic(n_blend=300, n_dev=400, n_mlb=1500, seed=11, games=25):
    """Synthetic three-seed archives for six families with the real key/field layout (no real data)."""
    rng = np.random.default_rng(seed)
    probs = lambda n: rng.dirichlet(np.ones(10), size=n)

    def part(n, prefix, offset):
        game = np.sort(rng.integers(0, games, n)) + offset
        return {f'{prefix}_keys': np.stack([game, np.arange(n), np.zeros(n, int)], 1), f'{prefix}_y': rng.integers(0, 10, n),
                f'{prefix}_game_pk': game, f'{prefix}_pitcher': rng.integers(0, 5, n)}
    meta = {**part(n_blend, 'blend', 1000), **part(n_dev, 'dev', 2000)}
    filled = lambda: {**meta, **{p: probs(len(meta[p.split('_')[0] + '_y'])) for p in PARTS}, 'dev_delivery_level': np.zeros(n_dev, int)}
    baseline = filled()
    direct = {f: [filled() for _ in SEEDS] for f in FAMILIES}
    g0 = [filled() for _ in SEEDS]
    mlb = part(n_mlb, 'dev', 5000)
    g0_mlb = [{**mlb, 'dev': probs(n_mlb), 'dev_raw': probs(n_mlb), 'dev_delivery_level': np.zeros(n_mlb, int)} for _ in SEEDS]
    direct_mlb = {f: [{'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev_y': mlb['dev_y'], 'mlb_dev': probs(n_mlb)} for _ in SEEDS]
                  for f in FAMILIES}
    frequency = {'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev': np.tile(rng.dirichlet(np.ones(10)), (n_mlb, 1))}
    month = lambda g: np.where(g % 3 == 0, '2025-07', np.where(g % 3 == 1, '2025-08', '2025-09'))
    return direct, g0, baseline, direct_mlb, g0_mlb, frequency, month(meta['dev_game_pk']), month(mlb['dev_game_pk'])


def reference_from(direct, g0, baseline, direct_mlb, g0_mlb, frequency, *_):
    """EXP-P15-001-shaped reference from the independent three-seed run_ml_direct_head.score_arrays."""
    import run_ml_direct_head as base
    config = {'metrics': {'N': {'delta_nll_max': -.003, 'brier_ci95_upper_max': .001}, 'C': {}}, 'mix': {'direct_weight': .5}}
    saved = base.paired, base.decide, base.tier_slices, base.efficiency
    base.paired = lambda y, games, a, b, sa, sb: {
        'seed_deltas': [float((pitch_losses(y, u) - pitch_losses(y, v))[:, 0].mean()) for u, v in zip(sa, sb)],
        'candidate': prediction_metrics(y, a), 'control': prediction_metrics(y, b)}
    base.decide = lambda rows, n_rule, holm=True: rows
    base.tier_slices = lambda *a: None
    base.efficiency = lambda *a: None
    try:
        return base.score_arrays(direct['P15-uniform'], g0, baseline, config, direct_mlb['P15-uniform'], g0_mlb, frequency)
    finally:
        base.paired, base.decide, base.tier_slices, base.efficiency = saved


def provenance(config_path):
    meta = p19.provenance(config_path)
    meta['runner'], meta['runner_sha256'] = NAME, hash_file(PROJECT / NAME)
    meta['source_hashes'] = {**meta['source_hashes'], NAME: hash_file(PROJECT / NAME),
                             p19.NAME: hash_file(PROJECT / p19.NAME), 'scripts/run_ml_direct_head.py': hash_file(PROJECT / 'scripts/run_ml_direct_head.py')}
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('command', choices=('smoke', 'pins', 'run'))
    args = p.parse_args()
    full = args.command == 'run'
    config, local = config_check(read_json(args.config), full), read_json(args.local_config)
    if args.command == 'pins':
        print(json.dumps(pins(config), indent=1))
        return
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output exists; refusing to overwrite')
    started = time.monotonic()
    deadline = started + config['execution']['max_wall_seconds']
    meta = {**provenance(args.config), 'local_config_sha256': hash_file(args.local_config)}
    if not full:
        if not output.name.endswith('-smoke'):
            raise ValueError('Smoke output must end -smoke')
        arrays = synthetic()
        result = search(*arrays, p19.jsonable(reference_from(*arrays)), config['reproduction_gate']['abs_tolerance'], 257, deadline)
        p19.write_result(output, p19.jsonable(result), config, {'mode': 'synthetic_smoke', 'provenance': meta,
                         'seconds': time.monotonic() - started, 'peak_rss_bytes': peak()})
        print('ENSEMBLE_SMOKE_DONE', {'gate': result['reproduction_gate']['passed'], 'candidates': result.get('candidate_count')}, flush=True)
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
        extra = {'experiment_id': config['experiment_id'], 'mode': 'exploratory_repeated_dev', 'nature': config['nature'],
                 'provenance': meta, 'families': config['families'], 'inputs': config['inputs'],
                 'reproduction_reference_sha256': config['reproduction_gate']['reference_sha256'],
                 'seconds': {'load': load_seconds, 'total': time.monotonic() - started}, 'peak_rss_bytes': peak(),
                 'limits': config['limits']}
        p19.write_result(output, p19.jsonable(result), config, extra)
        repo_result = REPO / 'results' / (config['experiment_id'] + '.json')
        if not repo_result.exists():
            shutil.copyfile(output / 'results.json', repo_result)
    if not result['reproduction_gate']['passed']:
        raise ValueError('Reproduction gate failed; saved diagnostic result, no candidate search')
    print('ENSEMBLE_DONE', {'gate': True, 'candidates': result['candidate_count']}, flush=True)


if __name__ == '__main__':
    main()
