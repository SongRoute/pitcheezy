"""EXP-WS-SEQ-001 (E4) evaluation on the Mac after `gpu pull`: neural arms N0 / N1 vs the trees H0, H2 (E1) and P (E1b).

For each arm and seed and for the ensemble of the available seeds (mean of calibrated seven-leaf probabilities):
NLL / Brier / top-label ECE (10 bins) on the registered EVAL rows, paired whole-game bootstrap (eop-001 contract),
strata S0..S3 (pitcher FIT rows) and B0/B1 (batter FIT rows). PRIMARY: N1 - P NLL; co-primary: N1 - N0 NLL.
2026 EXPOSED DEVELOPMENT; advisory labels only; never a winner/adoption claim. Numpy + scipy only (no torch).

  <runner python> -I experiments/ws_seq/evaluate_seq.py --registration <REGISTRATION.json> --runs <pulled runs dir>
        [--inputs data/ws-seq/inputs] --out <new result dir>
  <runner python> -I experiments/ws_seq/evaluate_seq.py --selfcheck [--selfcheck-out FILE]    (closed forms, synthetic)

BLOCKED unless the registration is READY_FOR_REVIEW, free of PENDING_E1B placeholders, and REVIEW.json (verdict PASS,
distinct reviewer, attempt id, matching source sha256) exists. Every number in result.json is computed in this run.
"""
import os, sys
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP_ID = 'EXP-WS-SEQ-001'
READY = 'READY_FOR_REVIEW'
PENDING = 'PENDING_E1B'
CONFIG_JSON = REPO / 'configs' / (EXP_ID + '.json')
CONFIG_YAML = REPO / 'configs' / (EXP_ID + '.yaml')
SOURCES = {'experiments/ws_seq/prepare_inputs.py': HERE / 'prepare_inputs.py', 'experiments/ws_seq/train_seq.py': HERE / 'train_seq.py',
           'experiments/ws_seq/evaluate_seq.py': HERE / 'evaluate_seq.py', 'configs/EXP-WS-SEQ-001.json': CONFIG_JSON, 'configs/EXP-WS-SEQ-001.yaml': CONFIG_YAML}
NEURAL = ('N0', 'N1')
TREES = ('H0', 'H2', 'P')
HEADS = ('parent', 'narrow_whiff', 'ground_ball_per_pitch')
ROLES = ('tuning', 'calibration', 'evaluation')
BIP = (3, 4, 5, 6, 7, 9)
BINS = 10
EDGES = [i / BINS for i in range(BINS + 1)]
CLIP = 1e-15
RUN_FILES = ('margins.npz', 'temperatures.json', 'seven-leaf-evaluation.npy', 'curve.json', 'model-state.pt')


class Stop(Exception):
    """A registered STOP: no result.json."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False, default=str, ensure_ascii=False) + '\n')


def own_hashes(reg_path):
    return {'REGISTRATION.json': sha(reg_path), **{k: sha(p) for k, p in SOURCES.items()}}


def pending_fields(obj, at=''):
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in pending_fields(v, f'{at}.{k}' if at else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in pending_fields(v, f'{at}[{i}]')]
    return [at] if obj == PENDING else []


def load_registration(path):
    reg = json.loads(Path(path).read_text()); b = reg['bootstrap']
    ok = [reg['id'] == EXP_ID, b['unit'] == 'game_pk', reg['metrics']['bins'] == BINS, reg['metrics']['nll_clip'] == CLIP, list(reg['arms']) == list(NEURAL),
          set(reg['evaluation_inputs']) == {'eval_meta', 'seven_leaf_H0', 'seven_leaf_H2', 'seven_leaf_P'}, b['primary_pair'] == ['N1', 'P'], b['co_primary_pair'] == ['N1', 'N0']]
    if not all(ok):
        raise SystemExit('BLOCKED: REGISTRATION.json does not match this source: ' + str(ok))
    return reg


def review_gate(reg, reg_path):
    if reg.get('status') != READY:
        raise SystemExit('BLOCKED: registration status is not exactly ' + READY)
    pend = pending_fields(reg)
    if pend:
        raise SystemExit('BLOCKED: registration still carries ' + PENDING + ' at: ' + ', '.join(pend))
    p = Path(reg['review']['path'])
    if not p.exists():
        raise SystemExit('BLOCKED: independent REVIEW.json (verdict PASS, matching source sha256) required at ' + str(p))
    r = json.loads(p.read_text())
    if r.get('verdict') != 'PASS' or r.get('distinct_from_author') is not True or r.get('attempt_id') != reg['attempt_id']:
        raise SystemExit('BLOCKED: review verdict / distinct reviewer / attempt id')
    if r.get('sources_sha256') != own_hashes(reg_path):
        raise SystemExit('BLOCKED: reviewed source sha256 differ from the current REGISTRATION.json / scripts / configs')
    return sha(p)


# ---------------------------------------------------------------- leaves (copy of scale_safe.leaves, sha256 1f2cd2e2...5234, lines 132-136)
def leaves(p10, q, r):
    s = p10[:, list(BIP)].sum(1); out = np.column_stack([p10[:, 0], p10[:, 2], p10[:, 8], p10[:, 1] * q, p10[:, 1] * (1 - q), s * r, s * (1 - r)])
    if not np.isfinite(out).all() or np.abs(out.sum(1) - 1).max() > 1e-9:
        raise Stop('STOP seven-leaf coherence')
    return out


def seven_from_margins(z, T=None):
    from scipy.special import softmax, expit
    T = T or {h: 1.0 for h in HEADS}
    return leaves(softmax(z['parent'] / T['parent'], axis=1), expit(z['narrow_whiff'] / T['narrow_whiff']), expit(z['ground_ball_per_pitch'] / T['ground_ball_per_pitch']))


# ---------------------------------------------------------------- metrics
# row_loss, top, ece, reliability, classwise_ece, arm_metrics, bootstrap, paired_block, assign_bins: copied from
# /Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-research-e1e3-001/e1-recency/recency_run.py (lines 492-596,
# sha256 af1947c0841921902e51025433d4b6e555190e7755400e0e0a67aadd652f232d), which copied row_loss/top/ece/reliability
# verbatim from compare_e_op.py (sha256 933c8b7bcace7ab36a18626b76a2ef63134ab4bed62bf2c011d5a840dd115577, eop-001 contract).
# Not imported. Differences: paired_block takes the pair list for the classwise point difference unchanged; nothing else.
def row_loss(q, y):
    i = np.arange(len(y)); oh = np.zeros_like(q); oh[i, y] = 1
    return -np.log(np.clip(q[i, y], CLIP, 1)), ((q - oh) ** 2).sum(1)


def top(q, y):
    conf = q.max(1); return conf, (q.argmax(1) == y).astype(float), np.minimum((conf * BINS).astype(int), BINS - 1)


def ece(q, y):
    conf, hit, b = top(q, y); return float(np.abs(np.bincount(b, conf - hit, BINS)).sum() / len(y))


def reliability(q, y, leaf_names):
    out = {}
    for k, name in enumerate(leaf_names):
        p = q[:, k]; b = np.minimum((p * BINS).astype(int), BINS - 1); n = np.bincount(b, minlength=BINS); sp = np.bincount(b, p, BINS); so = np.bincount(b, (y == k).astype(float), BINS)
        out[name] = [{'bin': i, 'lo': EDGES[i], 'hi': EDGES[i + 1], 'n': int(n[i]), 'mean_predicted': float(sp[i] / n[i]) if n[i] else None, 'observed_frequency': float(so[i] / n[i]) if n[i] else None} for i in range(BINS)]
    return out


def classwise_ece(q, y):
    """SECONDARY: mean over the seven leaves of the ten-bin calibration gap of that leaf's probability."""
    total = 0.
    for k in range(q.shape[1]):
        p = q[:, k]; b = np.minimum((p * BINS).astype(int), BINS - 1)
        total += np.abs(np.bincount(b, p - (y == k), BINS)).sum() / len(y)
    return float(total / q.shape[1])


def arm_metrics(q, y):
    ll, bs = row_loss(q, y)
    return {'rows': int(len(y)), 'nll': float(ll.mean()), 'brier': float(bs.mean()), 'ece_top_label_10bin': ece(q, y), 'ece_classwise_10bin': classwise_ece(q, y)}


def bootstrap(Q, y, games, pairs, draws, seed, chunk):
    """Whole-game paired bootstrap conditional on fixed predictions: each draw resamples the G games with replacement;
    NLL/Brier difference = sum of the sampled games' loss-difference sums / sum of their pitch counts; ECE from summed
    per-game bin sums. All pairs share the same draws. Returns {'a-b': {metric: [lo, hi]}} percentile 95%."""
    u, inv = np.unique(games, return_inverse=True); G = len(u); n = np.bincount(inv, minlength=G).astype(float); per = {}
    for a, q in Q.items():
        ll, bs = row_loss(q, y); conf, hit, b = top(q, y); gap = np.zeros((G, BINS)); np.add.at(gap, (inv, b), conf - hit)
        per[a] = (np.bincount(inv, ll, G), np.bincount(inv, bs, G), gap)
    rng = np.random.default_rng(seed); out = {f'{a}-{b}': {m: np.empty(draws) for m in ('nll', 'brier', 'ece')} for a, b in pairs}
    for s in range(0, draws, chunk):
        pick = rng.integers(0, G, (min(chunk, draws - s), G)); N = n[pick].sum(1); sl = slice(s, s + len(pick))
        gaps = {a: np.abs(per[a][2][pick].sum(1)).sum(1) for a in Q}
        for a, b in pairs:
            o = out[f'{a}-{b}']
            o['nll'][sl] = (per[a][0] - per[b][0])[pick].sum(1) / N
            o['brier'][sl] = (per[a][1] - per[b][1])[pick].sum(1) / N
            o['ece'][sl] = (gaps[a] - gaps[b]) / N
    ci = {k: {m: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for m, v in d.items()} for k, d in out.items()}
    return ci, {'games': int(G), 'rows': int(len(y)), 'draws': int(draws), 'seed': int(seed), 'unit': 'game_pk'}


def paired_block(Q, y, games, pairs, bcfg, draws):
    """Point differences on the given rows plus bootstrap CI95 for NLL, Brier and top-label ECE; classwise ECE point only."""
    used = sorted({a for p in pairs for a in p}); Q = {a: Q[a] for a in Q}
    m = {a: arm_metrics(q, y) for a, q in Q.items()}
    ci, unit = bootstrap({a: Q[a] for a in used}, y, games, pairs, draws, bcfg['seed'], bcfg['chunk'])
    out = {}
    for a, b in pairs:
        k = f'{a}-{b}'
        out[k] = {'nll': {'difference': m[a]['nll'] - m[b]['nll'], 'ci95': ci[k]['nll']}, 'brier': {'difference': m[a]['brier'] - m[b]['brier'], 'ci95': ci[k]['brier']},
                  'ece_top_label_10bin': {'difference': m[a]['ece_top_label_10bin'] - m[b]['ece_top_label_10bin'], 'ci95': ci[k]['ece']},
                  'ece_classwise_10bin': {'difference': m[a]['ece_classwise_10bin'] - m[b]['ece_classwise_10bin'], 'ci95': None}}
    return m, out, unit


def assign_bins(counts, bins):
    """Stratum name per row from registered [name, lo, hi|None] bins; every row must land in exactly one bin."""
    lab = np.full(len(counts), '', object)
    for name, lo, hi in bins:
        m = (counts >= lo) & (counts <= (hi if hi is not None else np.iinfo(np.int64).max))
        if (lab[m] != '').any():
            raise Stop('STOP overlapping stratum bins')
        lab[m] = name
    if (lab == '').any():
        raise Stop('STOP row outside every stratum bin')
    return lab


def fit_rows_of(fit_ids, eval_ids):
    """Number of FIT rows carrying each EVAL row's id (same definition as recency_run.fit_rows_of)."""
    u, c = np.unique(fit_ids, return_counts=True); pos = np.searchsorted(u, eval_ids); pos = np.clip(pos, 0, max(len(u) - 1, 0))
    return np.where(u[pos] == eval_ids, c[pos], 0).astype(np.int64) if len(u) else np.zeros(len(eval_ids), np.int64)


def advisory(reg, paired, stop_triggered):
    """Advisory text only (no adoption meaning)."""
    mg = reg['advisory']['non_inferiority_margins']
    ni = {k: {'does_not_degrade_nll': bool(paired[k]['nll']['ci95'][1] <= mg['nll']), 'does_not_degrade_brier': bool(paired[k]['brier']['ci95'][1] <= mg['brier'])} for k in ('N1-P', 'N0-P')}
    if stop_triggered:
        return {'experiment_status': 'STOP_RULE', 'SEQUENCE_HELPS_NLL': None, 'BEATS_TREE_P': None, 'non_inferiority': ni,
                'note': 'registered stop rule triggered on seed 0: seeds 1,2 were not run; the seed-0 numbers are descriptive and carry no label'}
    return {'experiment_status': 'COMPLETED', 'SEQUENCE_HELPS_NLL': bool(paired['N1-N0']['nll']['ci95'][1] < 0), 'BEATS_TREE_P': bool(paired['N1-P']['nll']['ci95'][1] < 0), 'non_inferiority': ni}


# ---------------------------------------------------------------- loading
def verified(spec, what):
    p = Path(spec['path'])
    if not p.is_file() or sha(p) != spec['sha256']:
        raise SystemExit('BLOCKED: registered evaluation input missing or sha256 mismatch: ' + what)
    return p


def load_runs(runs, reg, manifest, manifest_sha, C, synthetic):
    """Every <runs>/<arm>-s<seed>/ with a receipt: verify hashes and bindings, rebuild the leaves from the margins."""
    runs = Path(runs); ev = C['evaluation']; tu = C['tuning']; n_ev = int(ev.sum()); found = {a: {} for a in NEURAL}; tol = reg['gates']['evaluate']['leaves_recompute_max_abs']
    train_sha = sha(HERE / 'train_seq.py')
    for arm in NEURAL:
        for seed in reg['seeds']['training']:
            d = runs / f'{arm}-s{seed}'
            if not (d / 'receipt.json').is_file():
                continue
            r = json.loads((d / 'receipt.json').read_text())
            bind = [r['id'] == EXP_ID, r['arm'] == arm, r['seed'] == seed, r['manifest_sha256'] == manifest_sha, r['train_seq_sha256'] == train_sha == manifest['bound_sources']['train_seq.py'],
                    r['config_sha256'] == manifest['bound_sources']['config'], synthetic or r['config_sha256'] == sha(CONFIG_JSON),
                    all(sha(d / f) == r['outputs_sha256'][f] for f in RUN_FILES), r['rows'] == {k: int(C[k].sum()) for k in ('fit',) + ROLES}]
            if not all(bind):
                raise Stop(f'STOP run {d.name}: receipt binding (id/arm/seed/manifest/train_seq/config/output sha256/rows) failed: {bind}')
            with np.load(d / 'margins.npz', allow_pickle=False) as z:
                M = {role: {h: np.array(z[f'{role}__{h}'], np.float64) for h in HEADS} for role in ROLES}
            T = json.loads((d / 'temperatures.json').read_text())['temperatures']; q = np.load(d / 'seven-leaf-evaluation.npy', allow_pickle=False)
            if q.shape != (n_ev, 7) or q.dtype != np.float64 or M['evaluation']['parent'].shape != (n_ev, 10) or T != r['temperatures']:
                raise Stop(f'STOP run {d.name}: shapes / temperatures')
            gap = float(np.abs(seven_from_margins(M['evaluation'], T) - q).max())
            sel = float(np.mean(row_loss(seven_from_margins(M['tuning']), C['y7'][tu])[0]))
            if gap > tol or abs(sel - r['best_select_seven_leaf_nll_uncalibrated']) > reg['gates']['evaluate']['select_nll_recompute_max_abs']:
                raise Stop(f'STOP run {d.name}: leaves or SELECT NLL do not reproduce from the margins (leaf gap {gap}, select {sel})')
            found[arm][seed] = {'q': q, 'receipt': r, 'leaves_recompute_max_abs': gap, 'select_nll_recomputed': sel, 'dir': str(d)}
    return found


def seed_policy(found, reg):
    """Registered: either seed 0 alone with the stop rule triggered, or all registered seeds with the stop rule passed."""
    s0, s1 = sorted(found['N0']), sorted(found['N1']); want = list(reg['seeds']['training']); ref = reg['stop_rule']['reference_select_nll']
    if s0 != s1 or 0 not in s1:
        raise Stop(f'STOP seeds differ between arms or seed 0 missing: N0 {s0}, N1 {s1}')
    sr = found['N1'][0]['receipt']['stop_rule']
    if not sr['applies'] or sr['reference_select_nll'] != ref:
        raise Stop('STOP N1 seed 0 was not trained with the registered stop-rule reference ' + repr(ref))
    triggered = not found['N1'][0]['select_nll_recomputed'] < ref
    if triggered != sr['triggered']:
        raise Stop('STOP stop-rule decision recomputed from the margins differs from the receipt')
    if triggered and s1 != [0]:
        raise Stop('STOP stop rule triggered on seed 0 but other seeds exist: ' + str(s1))
    if not triggered and s1 != want:
        raise Stop(f'STOP stop rule passed but seeds {s1} != registered {want} (evaluate only after all registered seeds finished)')
    return s1, triggered


def evaluate(reg, reg_path, runs, inputs, out, draws=None):
    synthetic = 'synthetic_selfcheck' in reg
    if synthetic and not all('selfcheck-tmp' in Path(p).parts for p in (reg_path, runs, inputs, out)):
        raise SystemExit('BLOCKED: a synthetic registration may only be used under a selfcheck-tmp directory')
    review_sha = review_gate(reg, reg_path); out = Path(out)
    if (out / 'result.json').exists():
        raise SystemExit('BLOCKED: result.json exists (a rerun needs a fresh --out and a recorded reason): ' + str(out))
    E = reg['evaluation_inputs']; paths = {k: verified(E[k], k) for k in E}; inputs = Path(inputs); mp = inputs / 'manifest.json'
    if not mp.is_file():
        raise SystemExit('BLOCKED: inputs/manifest.json missing: ' + str(mp))
    manifest = json.loads(mp.read_text()); manifest_sha = sha(mp)
    for f in ('corpus-subset.npz', 'seq-inputs.npz'):
        if sha(inputs / f) != manifest['files'][f]['sha256']:
            raise SystemExit('BLOCKED: local input package differs from its manifest: ' + f)
    bound = [manifest['id'] == EXP_ID, manifest['synthetic'] == synthetic, manifest['source_sha256']['corpus'] == reg['inputs']['corpus']['sha256'],
             manifest['source_sha256']['x_prequential'] == reg['inputs']['x_prequential']['sha256'], synthetic or manifest['registration_sha256'] == sha(reg_path)]
    if not all(bound):
        raise SystemExit('BLOCKED: input manifest is not bound to this registration (id/synthetic/corpus/x_prequential/registration sha256): ' + str(bound))
    with np.load(inputs / 'corpus-subset.npz', allow_pickle=False) as z:
        C = {k: np.array(z[k]) for k in z.files}
    with np.load(inputs / 'seq-inputs.npz', allow_pickle=False) as z:
        pit_all, bat_all = np.array(z['pitcher'], np.int64), np.array(z['batter'], np.int64)
    with np.load(paths['eval_meta'], allow_pickle=False) as z:
        meta = {k: np.array(z[k]) for k in ('keys', 'y7', 'games', 'pitcher', 'batter', 'fit_pitcher_rows', 'fit_batter_rows')}
    ev = C['evaluation']; y = C['y7'][ev]; games = C['games'][ev]; n = len(y); pitcher = pit_all[ev]
    same = [np.array_equal(meta['keys'], C['keys'][ev]), np.array_equal(meta['y7'], y), np.array_equal(meta['games'], games), np.array_equal(meta['pitcher'], pitcher), np.array_equal(meta['batter'], bat_all[ev]),
            np.array_equal(meta['fit_pitcher_rows'], fit_rows_of(pit_all[C['fit']], pitcher)), np.array_equal(meta['fit_batter_rows'], fit_rows_of(bat_all[C['fit']], bat_all[ev]))]
    if not all(same):
        raise Stop('STOP eval-meta.npz is not the EVAL slice of the corpus (keys/y7/games/pitcher/batter/FIT-row counts): ' + str(same))
    den = {'rows': int(n), 'games': int(len(np.unique(games))), 'leaf_counts': np.bincount(y, minlength=7).tolist(), 'pitchers': int(len(np.unique(pitcher)))}
    exp = reg['expected_denominators']
    if (den['rows'], den['games'], den['leaf_counts']) != (exp['role_rows']['EVAL'], exp['eval_games'], exp['eval_leaf_counts']):
        raise Stop('STOP EVAL denominators differ from the registration: ' + json.dumps(den))
    Q = {}
    for a in TREES:
        q = np.load(paths['seven_leaf_' + a], allow_pickle=False)
        if q.shape != (n, 7) or not np.isfinite(q).all() or np.abs(q.sum(1) - 1).max() > 1e-9:
            raise Stop('STOP tree baseline leaves shape / coherence: ' + a)
        Q[a] = np.asarray(q, np.float64)
    found = load_runs(runs, reg, manifest, manifest_sha, C, synthetic); seeds, triggered = seed_policy(found, reg)
    for a in NEURAL:
        for s in seeds:
            Q[f'{a}-s{s}'] = found[a][s]['q']
        Q[a] = np.mean([found[a][s]['q'] for s in seeds], axis=0)   # ensemble = mean of the calibrated seven-leaf probabilities
    b = reg['bootstrap']; draws = draws or b['draws']
    pairs = [tuple(p) for p in b['pairs']] + [p for s in seeds for p in ((f'N1-s{s}', 'P'), (f'N1-s{s}', f'N0-s{s}'), (f'N0-s{s}', 'P'))]
    overall, paired, unit = paired_block(Q, y, games, pairs, b, draws)
    for a in TREES + NEURAL:
        overall[a]['reliability_per_leaf'] = reliability(Q[a], y, reg['metrics']['leaf_order'])
    strata = {}; spairs = [tuple(p) for p in b['stratum_pairs']]; unstable = reg['strata']['small_n_unstable']; core = {a: Q[a] for a in TREES + NEURAL}
    for key, counts in (('pitcher_fit_rows', meta['fit_pitcher_rows']), ('batter_fit_rows', meta['fit_batter_rows'])):
        lab = assign_bins(counts, reg['strata'][key]['bins']); strata[key] = {}
        for name, _, _ in reg['strata'][key]['bins']:
            m = lab == name; rows = int(m.sum())
            if rows == 0:
                strata[key][name] = {'rows': 0, 'games': 0, 'pitchers': 0, 'note': 'empty stratum: nothing computed'}; continue
            sm, sp, su = paired_block({a: q[m] for a, q in core.items()}, y[m], games[m], spairs, b, draws)
            strata[key][name] = {'rows': rows, 'games': su['games'], 'pitchers': int(len(np.unique(pitcher[m]))), 'small_n_unstable': bool(rows < unstable['rows_below'] or su['games'] < unstable['games_below']), 'arms': sm, 'paired': sp}
        if sum(v['rows'] for v in strata[key].values()) != n:
            raise Stop('STOP stratum rows do not reconcile to the EVAL rows: ' + key)
    adv = advisory(reg, paired, triggered)
    runs_report = {f'{a}-s{s}': {k: found[a][s]['receipt'].get(k) for k in ('epochs_run', 'best_epoch', 'best_select_seven_leaf_nll_uncalibrated', 'early_stopped', 'wall_cap_hit', 'elapsed_seconds', 'parameters',
                                                                               'temperatures', 'stop_rule', 'device', 'gpu_name', 'gpu_uuid', 'peak_gpu_memory_bytes', 'versions', 'job')}
                   | {'leaves_recompute_max_abs': found[a][s]['leaves_recompute_max_abs'], 'select_nll_recomputed': found[a][s]['select_nll_recomputed'], 'dir': found[a][s]['dir']} for a in NEURAL for s in seeds}
    result = {'id': reg['id'], 'attempt_id': reg['attempt_id'], 'synthetic': synthetic,
              'label': '2026 EXPOSED DEVELOPMENT; research model; EVAL is not an independent TEST; no winner, adoption or policy meaning', 'claim_ceiling': reg['claim_ceiling'],
              'sign': 'arm - baseline; negative = lower loss / lower ECE for the arm', 'denominators': den, 'seeds_evaluated': seeds,
              'ensemble': 'N0 / N1 = mean over the evaluated seeds of the calibrated seven-leaf probabilities', 'stop_rule': {'reference_select_nll': reg['stop_rule']['reference_select_nll'], 'triggered': triggered,
                                                                                                                             'N1_seed0_best_select_nll': found['N1'][0]['select_nll_recomputed']},
              'arms': overall, 'paired': paired,
              'primary': {'estimand': 'N1 (ensemble) - P overall EVAL seven-leaf NLL', **paired['N1-P']['nll']}, 'co_primary': {'estimand': 'N1 (ensemble) - N0 (ensemble) overall EVAL seven-leaf NLL (sequence effect)', **paired['N1-N0']['nll']},
              'advisory': adv, 'advisory_meaning': reg['advisory']['meaning'], 'bootstrap': dict(unit, contract=b['contract'], ci=b['ci']), 'bin_edges': EDGES, 'strata': strata, 'runs': runs_report,
              'nondeterminism_caveat': reg['nondeterminism_caveat'], 'bindings': {'registration_sha256': sha(reg_path), 'review_sha256': review_sha, 'manifest_sha256': manifest_sha, 'sources_sha256': own_hashes(reg_path),
                                                                                'evaluation_inputs_sha256': {k: E[k]['sha256'] for k in E}},
              'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': __import__('scipy').__version__}, 'utc': time.time(),
              'scientific_decision': None, 'adoption': False, 'automatic_followup': False}
    out.mkdir(parents=True, exist_ok=True)
    for a in NEURAL:
        np.save(out / f'seven-leaf-{a}-ensemble.npy', Q[a], allow_pickle=False)
    save_json(out / 'result.json', result)
    return result


# ---------------------------------------------------------------- selfcheck: closed forms (adapted from recency_run.closed_forms)
def brute_bootstrap(Q, y, games, pair, draws, seed, chunk):
    u = np.unique(games); G = len(u); rows = [np.flatnonzero(games == x) for x in u]; rng = np.random.default_rng(seed); out = {m: [] for m in ('nll', 'brier', 'ece')}
    for s in range(0, draws, chunk):
        for pick in rng.integers(0, G, (min(chunk, draws - s), G)):
            i = np.concatenate([rows[j] for j in pick]); a, b = (arm_metrics(Q[k][i], y[i]) for k in pair)
            out['nll'].append(a['nll'] - b['nll']); out['brier'].append(a['brier'] - b['brier']); out['ece'].append(a['ece_top_label_10bin'] - b['ece_top_label_10bin'])
    return {m: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for m, v in out.items()}


def selfcheck(reg):
    out = {}
    q = np.zeros((100, 7)); y = np.zeros(100, int); q[:40, 0] = .75; q[:40, 1] = .25; y[20:40] = 1; q[40:, 0] = .55; q[40:, 1] = .45
    assert abs(ece(q, y) - .37) < 1e-12 and abs(classwise_ece(q, y) - .74 / 7) < 1e-12
    out['ece_closed_form'] = {'value': ece(q, y), 'expected': .37}; out['classwise_closed_form'] = {'value': classwise_ece(q, y), 'expected': .74 / 7}
    ll, bs = row_loss(q, y)
    assert abs(ll[0] + np.log(.75)) < 1e-12 and abs(bs[0] - 2 * .25 ** 2) < 1e-12 and abs(bs[20] - 2 * .75 ** 2) < 1e-12 and top(np.eye(7)[:1], np.array([0]))[2][0] == 9
    out['row_loss_closed_form'] = True
    p = np.array([[.30, .20, .10, .05, .05, .05, .05, .05, .10, .05]]); want = np.array([.30, .10, .10, .05, .15, .12, .18])
    assert np.allclose(leaves(p, np.array([.25]), np.array([.4]))[0], want, atol=1e-15); out['leaves_closed_form'] = True
    rng = np.random.default_rng(5); games = np.repeat(np.arange(9), [3, 5, 8, 2, 6, 4, 7, 5, 9]); n = len(games); yb = rng.integers(0, 7, n)
    Q = {a: rng.dirichlet(np.ones(7) * 2, n) for a in ('N0', 'N1', 'P')}; prs = [('N1', 'P'), ('N1', 'N0'), ('N0', 'P')]
    ci, unit = bootstrap(Q, yb, games, prs, 60, 20261211, 7); worst = 0.
    for pair in prs:
        ref = brute_bootstrap(Q, yb, games, pair, 60, 20261211, 7)
        worst = max(worst, max(abs(a - b) for m in ref for a, b in zip(ref[m], ci['-'.join(pair)][m])))
    assert worst < 1e-12 and unit == {'games': 9, 'rows': n, 'draws': 60, 'seed': 20261211, 'unit': 'game_pk'}
    out['bootstrap_vs_row_level_brute_force_max_abs_diff'] = worst
    same, _ = bootstrap({'P': Q['P'], 'N1': Q['P']}, yb, games, [('N1', 'P')], 40, 1, 500)
    assert all(v == [0., 0.] for v in same['N1-P'].values()); out['bootstrap_identical_arms_zero'] = True
    g2 = np.repeat(np.arange(5), 4); y2 = np.tile([0, 1, 0, 1], 5)
    qa = np.tile(np.array([[.6, .4, 0, 0, 0, 0, 0], [.3, .7, 0, 0, 0, 0, 0], [.8, .2, 0, 0, 0, 0, 0], [.45, .55, 0, 0, 0, 0, 0]]), (5, 1))
    qb = np.tile(np.array([[.5, .5, 0, 0, 0, 0, 0]]), (20, 1)); qb[:, 0] += 1e-9; qb[:, 1] -= 1e-9
    _, p2, _ = paired_block({'P': qb, 'N1': qa}, y2, g2, [('N1', 'P')], {'seed': 3, 'chunk': 500}, 50)
    assert all(max(abs(p2['N1-P'][k]['difference'] - v) for v in p2['N1-P'][k]['ci95']) < 1e-12 for k in ('nll', 'brier', 'ece_top_label_10bin'))
    out['bootstrap_degenerate_equal_games'] = True
    blk = lambda lo, hi: {'nll': {'ci95': [lo, hi]}, 'brier': {'ci95': [lo, hi]}}
    a1 = advisory(reg, {'N1-N0': blk(-.002, -.001), 'N1-P': blk(-.001, .001), 'N0-P': blk(-.001, .0005)}, False)
    assert a1['SEQUENCE_HELPS_NLL'] is True and a1['BEATS_TREE_P'] is False and a1['non_inferiority']['N1-P'] == {'does_not_degrade_nll': True, 'does_not_degrade_brier': False}
    assert a1['non_inferiority']['N0-P'] == {'does_not_degrade_nll': True, 'does_not_degrade_brier': True}
    a2 = advisory(reg, {'N1-N0': blk(-.002, -.001), 'N1-P': blk(-.003, -.001), 'N0-P': blk(0, .01)}, True)
    assert a2['experiment_status'] == 'STOP_RULE' and a2['SEQUENCE_HELPS_NLL'] is None and a2['BEATS_TREE_P'] is None
    out['advisory_labels'] = True
    lab = assign_bins(np.array([0, 1, 199, 200, 999, 1000, 50000]), reg['strata']['pitcher_fit_rows']['bins']).tolist()
    assert lab == ['S0', 'S1', 'S1', 'S2', 'S2', 'S3', 'S3'] and assign_bins(np.array([0, 1, 7]), reg['strata']['batter_fit_rows']['bins']).tolist() == ['B0', 'B1', 'B1']
    assert fit_rows_of(np.array([5, 5, 9]), np.array([5, 7, 9, 1, 11])).tolist() == [2, 0, 1, 0, 0]
    out['strata_bins'] = lab
    assert pending_fields({'a': [1, {'b': PENDING}], 'c': PENDING}) == ['a[1].b', 'c']; out['pending_scan'] = True
    return {'id': EXP_ID, 'mode': 'evaluate_seq --selfcheck (closed forms)', 'checks': out, 'verdict': 'SELFCHECK_PASS'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--registration', required=True); ap.add_argument('--runs'); ap.add_argument('--inputs', default=str(REPO / 'data' / 'ws-seq' / 'inputs')); ap.add_argument('--out')
    ap.add_argument('--selfcheck', action='store_true'); ap.add_argument('--selfcheck-out')
    args = ap.parse_args(); reg = load_registration(args.registration)
    if args.selfcheck:
        r = selfcheck(reg)
        if args.selfcheck_out:
            save_json(args.selfcheck_out, r)
        print(json.dumps({'verdict': r['verdict'], 'checks': sorted(r['checks'])})); return 0
    if not args.runs or not args.out:
        ap.print_usage(sys.stderr); return 2
    try:
        r = evaluate(reg, args.registration, args.runs, args.inputs, args.out)
    except Stop as e:
        print(str(e), file=sys.stderr); return 6
    print(json.dumps({'primary': r['primary'], 'co_primary': r['co_primary'], 'advisory': r['advisory'], 'seeds_evaluated': r['seeds_evaluated'], 'result': str(Path(args.out) / 'result.json')}, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
