"""calibrate_logits.py: post-hoc calibration of K seeds of three-head margins (parent-10 + two binary heads) and of
their seven-leaf ensemble. No classifier refit. Written for EXP-WS-SEQ-CAL-001; reusable (EXP-WS-SEQ-002 calls this
exact file with its own REGISTRATION.json / REVIEW.json in another directory).

Arms, per seed, every calibrator fit on that seed's CAL margins only:
  C0 recorded scalar temperature per head (temperatures.json; must reproduce the run's seven-leaf-evaluation.npy)
  C1 vector scaling of the parent margins + Platt scaling of the two binary heads
  C2 Dirichlet calibration (ODIR, lambda by game-grouped CV inside the fit rows) of the parent + the C1 Platt heads
  C3 per-leaf isotonic regression (PAV) of the C0 seven-leaf probabilities, clipped and renormalised
Calibrator selection (per seed, CAL only): game-grouped K-fold CV of every arm; selected = argmin CV mean top-label
ECE among the arms whose CV mean seven-leaf NLL <= C0's + registered margin; ties -> C0.
Ensemble of an arm = mean over the seeds of that arm's calibrated seven-leaf rows; SEL = mean over the seeds of each
seed's own selected arm. EVAL is scored for every arm (per seed and ensemble).

Usage (the registration is the anchor: REVIEW.json, SELFCHECK.json, PLAN.json and the attempt live in ITS directory):
  python -I calibrate_logits.py --selfcheck --registration REG
  python -I calibrate_logits.py --plan    --runs DIR --arm-prefix N1 --seeds 0,1,2 --corpus corpus.npz
        --eval-meta eval-meta.npz --baselines P=<npy>[,NAME=<npy>...] [--reference-ensemble <npy>]
        --registration REG --out <REG dir>/<attempt root>/<attempt id>
  python -I calibrate_logits.py --execute <same arguments as --plan>
Every command-line value must equal the registered one (the registration pins path + bytes + sha256 of every input);
a mismatch is BLOCKED. --execute refuses unless the registration is READY_FOR_REVIEW and an independent REVIEW.json
(verdict PASS, reviewer != author, sources_sha256 == current sha256 of THIS file and of the registration) sits next
to the registration, and the attempt directory does not exist.

Every hyperparameter is read from the registration (nothing numeric about the experiment is hard-coded here except
the constants of the verbatim-copied metric functions, which the registration repeats and the gate compares).
Data role of the callers so far: 2026 EXPOSED DEVELOPMENT. Output carries advisory labels only: no winner, no
adoption, no independent TEST. stdlib + numpy + scipy only.
"""
import os
import sys

sys.dont_write_bytecode = True
THREADS = '2'  # OMP/OPENBLAS/VECLIB/MKL threads, set before numpy is imported (same as E3)
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = THREADS

import argparse
import copy
import errno
import fcntl
import hashlib
import json
import platform
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, softmax

SELF = Path(__file__).resolve()
SCRIPT_KEY = 'calibrate_logits.py'
READY = 'READY_FOR_REVIEW'
HEADS = ('parent', 'narrow_whiff', 'ground_ball_per_pitch')
ROLES = ('tuning', 'calibration', 'evaluation')
ARMS = ('C0', 'C1', 'C2', 'C3')
SEL = 'SEL'
SEED_PAIRS = (('C1', 'C0'), ('C2', 'C0'), ('C3', 'C0'), ('C2', 'C1'))  # (arm, reference): difference = arm - reference
RUN_FILES = ('margins.npz', 'temperatures.json', 'seven-leaf-evaluation.npy', 'receipt.json')
SYNTH_DIR = 'selfcheck-tmp'  # a registration carrying 'synthetic_selfcheck' may only be used under a dir of this name
K10 = 10


class Stop(Exception):
    """Explicit STOP after the attempt was claimed (a terminal.json with success=false is written)."""


# =====================================================================================================================
# VERBATIM COPIES (pure functions only; the source modules carry gates and are never imported).
# Taken from the reviewed E3 file
#   /Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-research-e1e3-001/e3-cal/cal_repair.py
#   sha256 0daf2574192563dfdd72f3571f6d2ba7ed60f388969f987584466531ad492b5e, lines 67-109,
# which copied them verbatim from
#   scale_safe.py   sha256 1f2cd2e221bcff53c9f2b1b1ee7ceae0476f02bf5559a21e31a8b520f94b5234 lines 132-136 (BIP, leaves)
#   compare_e_op.py sha256 933c8b7bcace7ab36a18626b76a2ef63134ab4bed62bf2c011d5a840dd115577 lines 25-27, 81-93, 97-109
# bootstrap_eop_reference (draws/seed as required arguments) is used ONLY by --selfcheck as the reference that
# bootstrap_multi() below must reproduce exactly.
# =====================================================================================================================
BIP=(3,4,5,6,7,9)
def leaves(p10,q,r):
 s=p10[:,list(BIP)].sum(1);out=np.column_stack([p10[:,0],p10[:,2],p10[:,8],p10[:,1]*q,p10[:,1]*(1-q),s*r,s*(1-r)])
 if not np.isfinite(out).all() or np.abs(out.sum(1)-1).max()>1e-9:raise ValueError('STOP seven-leaf coherence')
 return out

LEAVES=['ball','foul','hit_by_pitch','narrow_whiff','other_strike','ground_ball_in_play','other_in_play']
BINS=10;EDGES=[i/BINS for i in range(BINS+1)]  # bin(p)=min(floor(p*10),9): [0,.1),[.1,.2),...,[.9,1.0]; frozen before payload
CLIP=1e-15

def row_loss(q,y):
 i=np.arange(len(y));oh=np.zeros_like(q);oh[i,y]=1
 return -np.log(np.clip(q[i,y],CLIP,1)),((q-oh)**2).sum(1)
def top(q,y):
 conf=q.max(1);return conf,(q.argmax(1)==y).astype(float),np.minimum((conf*BINS).astype(int),BINS-1)
def ece(q,y):
 conf,hit,b=top(q,y);return float(np.abs(np.bincount(b,conf-hit,BINS)).sum()/len(y))
def reliability(q,y):
 out={}
 for k,name in enumerate(LEAVES):
  p=q[:,k];b=np.minimum((p*BINS).astype(int),BINS-1);n=np.bincount(b,minlength=BINS);sp=np.bincount(b,p,BINS);so=np.bincount(b,(y==k).astype(float),BINS)
  out[name]=[{'bin':i,'lo':EDGES[i],'hi':EDGES[i+1],'n':int(n[i]),'mean_predicted':float(sp[i]/n[i]) if n[i] else None,'observed_frequency':float(so[i]/n[i]) if n[i] else None} for i in range(BINS)]
 return out

def bootstrap_eop_reference(qo,qn,y,games,draws,seed,chunk=500):
 """Whole-game paired bootstrap, fit100 - OLD, conditional on fixed predictions. Each draw: sum of sampled games'
 pitch losses / sum of sampled games' pitch counts (pitch-level estimand); ECE recomputed from per-game bin sums."""
 u,inv=np.unique(games,return_inverse=True);G=len(u);n=np.bincount(inv,minlength=G).astype(float);d={}
 for a,q in (('old',qo),('fit100',qn)):
  ll,bs=row_loss(q,y);conf,hit,b=top(q,y);gap=np.zeros((G,BINS));np.add.at(gap,(inv,b),conf-hit)
  d[a]=(np.bincount(inv,ll,G),np.bincount(inv,bs,G),gap)
 rng=np.random.default_rng(seed);out={m:np.empty(draws) for m in ('nll','brier','ece')}
 for s in range(0,draws,chunk):
  pick=rng.integers(0,G,(min(chunk,draws-s),G));N=n[pick].sum(1);sl=slice(s,s+len(pick))
  out['nll'][sl]=(d['fit100'][0]-d['old'][0])[pick].sum(1)/N;out['brier'][sl]=(d['fit100'][1]-d['old'][1])[pick].sum(1)/N
  out['ece'][sl]=(np.abs(d['fit100'][2][pick].sum(1)).sum(1)-np.abs(d['old'][2][pick].sum(1)).sum(1))/N
 return {m:{'ci95_percentile':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for m,v in out.items()}|{'games':int(G),'rows':int(len(y)),'draws':draws,'seed':seed,'unit':'game_pk'}
# =====================================================================================================================
# END OF VERBATIM COPIES.
# The next block (utilities, labels, optimisation, PAV) is copied from the same reviewed cal_repair.py (lines 116-173,
# 177-197, 240-355, 380-413) with these differences only: Stop text prefixes, fit names passed by the caller,
# select_lambda() takes the fold id of every row instead of building the folds itself (so the same function serves the
# full-CAL fit and the fits inside the selection CV). Arithmetic is unchanged.
# =====================================================================================================================


# ---------- small utilities ----------
def sha(p):
    """sha256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(p, x):
    """Write strict JSON (NaN/inf refused)."""
    Path(p).write_text(json.dumps(x, indent=1, allow_nan=False) + '\n')


def peak_rss_bytes():
    """ru_maxrss of this process in bytes (macOS reports bytes, Linux kilobytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == 'darwin' else r * 1024)


def free_memory_bytes(page_bytes, text=None):
    """(free + inactive + speculative) pages x page_bytes from `LC_ALL=C vm_stat` (README gate). `text` is for tests."""
    if text is None:
        text = subprocess.run(['/usr/bin/vm_stat'], env={'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}, capture_output=True,
                              text=True, check=True, timeout=20).stdout
    m = re.search(r'page size of (\d+) bytes', text)
    if not m or int(m.group(1)) != page_bytes:
        raise Stop('STOP vm_stat page size is not the registered %d bytes' % page_bytes)
    pages = 0
    for name in ('Pages free', 'Pages inactive', 'Pages speculative'):
        m = re.search(r'^%s:\s+(\d+)\.' % name, text, re.M)
        if not m:
            raise Stop('STOP vm_stat field missing: ' + name)
        pages += int(m.group(1))
    return pages * page_bytes


# ---------- labels ----------
def derive_y7(y10, whiff, gb):
    """Seven-leaf label implied by the parent-10 label and the two head labels (same map as leaves()):
    0 ball(y10=0), 1 foul(2), 2 HBP(8), 3 whiff(y10=1 & whiff), 4 other strike(y10=1 & ~whiff), 5 GB BIP, 6 non-GB BIP."""
    y7 = np.full(len(y10), -1, np.int64)
    y7[y10 == 0] = 0
    y7[y10 == 2] = 1
    y7[y10 == 8] = 2
    y7[(y10 == 1) & (whiff == 1)] = 3
    y7[(y10 == 1) & (whiff == 0)] = 4
    bip = np.isin(y10, BIP)
    y7[bip & (gb == 1)] = 5
    y7[bip & (gb == 0)] = 6
    return y7


def check_probs(q, name, tol):
    """Coherence gate for every arm: finite, inside [0,1], rows sum to 1 within tol."""
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() \
            or np.abs(q.sum(1) - 1).max() > tol:
        raise Stop('STOP seven-leaf coherence (finite / [0,1] / row sum): ' + name)


# ---------- secondary metrics ----------
def classwise_ece(q, y):
    """Per leaf k: sum_b (n_b/N)|mean q_k,b - freq_k,b| = sum_b |sum_{i in b}(q_ik - 1[y_i=k])| / N with the same 10
    equal-width bins (bin=min(floor(10*q_k),9)). Returns the mean and the max over the seven leaves and each leaf."""
    n = len(y)
    per = []
    for k in range(7):
        p = q[:, k]
        b = np.minimum((p * BINS).astype(int), BINS - 1)
        per.append(float(np.abs(np.bincount(b, p - (y == k), BINS)).sum() / n))
    return {'mean_over_leaves': float(np.mean(per)), 'max_over_leaves': float(np.max(per)), 'per_leaf': dict(zip(LEAVES, per))}


def top_label_reliability(q, y):
    """Report-only table behind the primary ECE: per confidence bin n, mean confidence, accuracy."""
    conf, hit, b = top(q, y)
    n = np.bincount(b, minlength=BINS)
    sc = np.bincount(b, conf, BINS)
    sh = np.bincount(b, hit, BINS)
    return [{'bin': i, 'lo': EDGES[i], 'hi': EDGES[i + 1], 'n': int(n[i]),
             'mean_confidence': float(sc[i] / n[i]) if n[i] else None,
             'accuracy': float(sh[i] / n[i]) if n[i] else None} for i in range(BINS)]


def advisory_labels(ci, margins):
    """Pre-registered advisory labels for one arm vs C0 from the paired CI95 upper bounds. No adoption meaning."""
    e = ci['ece'][1] < margins['ece_improved_upper_lt']
    n = ci['nll'][1] <= margins['nll_non_inferior_upper_le']
    b = ci['brier'][1] <= margins['brier_non_inferior_upper_le']
    return {'ece': 'ECE_IMPROVED' if e else 'ECE_NOT_IMPROVED', 'nll': 'NLL_NON_INFERIOR' if n else 'NLL_NOT_NON_INFERIOR',
            'brier': 'BRIER_NON_INFERIOR' if b else 'BRIER_NOT_NON_INFERIOR',
            'overall': 'CANDIDATE' if (e and n and b) else 'NOT_CANDIDATE'}


# ---------- optimisation (C1, C2) ----------
def _softmax_nll(s, y, rows):
    """Mean multiclass NLL of logits s (n,K) and the softmax probabilities, in one stable pass."""
    m = s.max(1, keepdims=True)
    e = np.exp(s - m)
    se = e.sum(1, keepdims=True)
    return float(np.mean(m[:, 0] + np.log(se[:, 0]) - s[rows, y])), e / se


def vector_objective(z, y):
    """C1 parent: logits = z*w + b (w, b in R^K). theta = [w, b]. Returns fun(theta) -> (mean NLL, analytic gradient)."""
    n, K = z.shape
    rows = np.arange(n)
    oh = np.zeros((n, K))
    oh[rows, y] = 1

    def fun(t):
        f, p = _softmax_nll(z * t[:K] + t[K:], y, rows)
        g = (p - oh) / n
        return f, np.concatenate([(g * z).sum(0), g.sum(0)])
    return fun


def platt_objective(z, y):
    """Binary head: logit = a*z + b. theta = [a, b]. Mean Bernoulli NLL and analytic gradient."""
    y = y.astype(np.float64)

    def fun(t):
        s = t[0] * z + t[1]
        r = expit(s) - y
        return float(np.mean(np.logaddexp(0, s) - y * s)), np.array([np.mean(r * z), np.mean(r)])
    return fun


def dirichlet_objective(u, y, lam, mu):
    """C2 parent: logits = u W^T + b with u = log_softmax(z). theta = [W.ravel(), b].
    Objective = mean NLL + lam * sum_{i != j} W_ij^2 + mu * sum_j b_j^2 (ODIR, exactly as written in BRIEF-E3)."""
    n, K = u.shape
    rows = np.arange(n)
    oh = np.zeros((n, K))
    oh[rows, y] = 1
    off = 1.0 - np.eye(K)

    def fun(t):
        W = t[:K * K].reshape(K, K)
        b = t[K * K:]
        f, p = _softmax_nll(u @ W.T + b, y, rows)
        g = (p - oh) / n
        Wo = W * off
        return f + lam * float((Wo ** 2).sum()) + mu * float((b ** 2).sum()), \
            np.concatenate([(g.T @ u + 2 * lam * Wo).ravel(), g.sum(0) + 2 * mu * b])
    return fun


def lbfgs(fun, x0, opt, name):
    """scipy L-BFGS-B with the registered options. STOP unless scipy reports success and everything is finite."""
    r = minimize(fun, np.asarray(x0, np.float64), jac=True, method='L-BFGS-B',
                 options={'gtol': opt['gtol'], 'ftol': opt['ftol'], 'maxiter': opt['maxiter'], 'maxfun': opt['maxfun'],
                          'maxls': opt['maxls'], 'maxcor': opt['maxcor']})
    f, g = fun(r.x)
    info = {'success': bool(r.success), 'status': int(r.status), 'message': str(r.message), 'iterations': int(r.nit),
            'function_evaluations': int(r.nfev), 'objective': float(f), 'gradient_max_abs': float(np.abs(g).max())}
    if not r.success or not np.isfinite(f) or not np.isfinite(r.x).all():
        raise Stop('STOP optimiser not converged: %s: %s' % (name, json.dumps(info)))
    return r.x, info


def require_classes(y, k, name):
    """A class absent from the fit rows sends its bias to -inf; refuse with an exact reason instead."""
    c = np.bincount(y, minlength=k)
    if len(c) != k or (c == 0).any():
        raise Stop('STOP class absent from fit rows: %s counts %s' % (name, c.tolist()))


def fit_vector(z, y, T, opt, name):
    """C1 parent vector scaling, init w = 1/T (the C0 solution), b = 0."""
    require_classes(y, K10, name)
    x, info = lbfgs(vector_objective(z, y), np.r_[np.full(K10, 1.0 / T), np.zeros(K10)], opt, name)
    return {'w': x[:K10], 'b': x[K10:]}, info


def fit_platt(z, y, T, opt, name):
    """Platt scaling of one binary head on its own rows, init a = 1/T, b = 0."""
    require_classes(y.astype(np.int64), 2, name)
    x, info = lbfgs(platt_objective(z, y), np.array([1.0 / T, 0.0]), opt, name)
    return {'a': float(x[0]), 'b': float(x[1])}, info


def fit_dirichlet(u, y, lam, mu, T, opt, name):
    """C2 parent Dirichlet calibration, init W = I/T, b = 0 (the C0 solution: softmax(log_softmax(z)/T) = softmax(z/T))."""
    require_classes(y, K10, name)
    x, info = lbfgs(dirichlet_objective(u, y, lam, mu), np.r_[(np.eye(K10) / T).ravel(), np.zeros(K10)], opt, name)
    return {'W': x[:K10 * K10].reshape(K10, K10), 'b': x[K10 * K10:]}, info


def cv_folds(games, seed, k):
    """Registered fold assignment: sort unique game_pks, permute with default_rng(seed).permutation, split into k
    contiguous chunks (np.array_split). Returns the fold index of every row and the games of each fold."""
    perm = np.random.default_rng(seed).permutation(np.unique(games))
    chunks = np.array_split(perm, k)
    fold = np.full(len(games), -1, np.int64)
    for f, c in enumerate(chunks):
        fold[np.isin(games, c)] = f
    if (fold < 0).any() or any(len(c) == 0 for c in chunks):
        raise Stop('STOP CV fold assignment')
    return fold, chunks


def select_lambda(u, y, fold, T, spec, opt, guard, tag):
    """Game-grouped CV over the registered lambda grid on the given rows; `fold` is each row's (game-grouped) fold id
    and every distinct id is held out once. Held-out score = UNREGULARISED mean parent-10 NLL on the held-out fold; a
    lambda's score = unweighted mean of its fold scores; lowest wins; exact ties -> larger lambda. (E3 rule.)"""
    ids = np.unique(fold).tolist()
    if len(ids) < 2:
        raise Stop('STOP lambda CV needs at least two folds: ' + tag)
    table = []
    for lam in spec['lambda_grid']:
        scores, infos = [], []
        for f in ids:
            tr, ho = fold != f, fold == f
            par, info = fit_dirichlet(u[tr], y[tr], lam, lam * spec['mu_over_lambda'], T, opt, '%s C2 CV lambda=%g fold=%d' % (tag, lam, f))
            nll, _ = _softmax_nll(u[ho] @ par['W'].T + par['b'], y[ho], np.arange(int(ho.sum())))
            scores.append(nll)
            infos.append(info)
            guard('%s C2 CV lambda=%g fold=%d' % (tag, lam, f))
        table.append({'lambda': lam, 'mu': lam * spec['mu_over_lambda'], 'heldout_fold_ids': ids, 'heldout_nll_by_fold': scores,
                      'mean_heldout_nll': float(np.mean(scores)), 'optimiser_by_fold': infos})
    best = min(table, key=lambda r: (r['mean_heldout_nll'], -r['lambda']))['lambda']
    return best, {'selected_lambda': best, 'table': table}


# ---------- isotonic (C3) ----------
def pav_fit(x, y):
    """Isotonic (non-decreasing) least-squares regression of y on x by pool-adjacent-violators, ties in x pooled first.
    Returns blocks with strictly increasing values: left/right = min/max x of the block, value = mean y, weight = rows."""
    order = np.argsort(x, kind='stable')
    xs = x[order]
    ys = np.asarray(y, np.float64)[order]
    ux, start, cnt = np.unique(xs, return_index=True, return_counts=True)
    sy = np.add.reduceat(ys, start)
    S, Wt, L, R = [], [], [], []  # stack of blocks: sum of y, weight, left x, right x
    for s, w, v in zip(sy.tolist(), cnt.tolist(), ux.tolist()):
        S.append(s); Wt.append(w); L.append(v); R.append(v)
        while len(S) > 1 and S[-2] * Wt[-1] >= S[-1] * Wt[-2]:  # mean[-2] >= mean[-1]: pool (equal means pooled too)
            s2 = S.pop(); w2 = Wt.pop(); L.pop(); r2 = R.pop()
            S[-1] += s2; Wt[-1] += w2; R[-1] = r2
    return {'left': np.array(L), 'right': np.array(R), 'value': np.array(S) / np.array(Wt), 'weight': np.array(Wt, np.int64)}


def pav_apply(blocks, x):
    """Piecewise-constant lookup: block index = searchsorted(block right edges, x, side='left'), clipped to the last
    block. So x <= first right edge -> first block, x > last right edge -> last block, x in a gap -> the upper block."""
    i = np.minimum(np.searchsorted(blocks['right'], x, side='left'), len(blocks['right']) - 1)
    return blocks['value'][i]


def isotonic_fit(q_cal, y7_cal):
    """One PAV fit per leaf: 1[y7==k] on the C0 probability of leaf k."""
    return [pav_fit(q_cal[:, k], (y7_cal == k).astype(np.float64)) for k in range(7)]


def isotonic_apply(blocks, q, clip):
    """Map each leaf, clip to [clip, 1-clip], renormalise rows to sum to one."""
    m = np.column_stack([pav_apply(blocks[k], q[:, k]) for k in range(7)])
    m = np.clip(m, clip, 1 - clip)
    return m / m.sum(1, keepdims=True)


# =====================================================================================================================
# NEW CODE of this script: K seeds, calibrator selection by CV, ensemble, per-leaf reliability, strata.
# =====================================================================================================================
def assign_bins(counts, bins):
    """Stratum name per row from registered [name, lo, hi|None] bins; every row must land in exactly one bin.
    (Copy of experiments/ws_seq/evaluate_seq.py assign_bins, sha256 7a192df7...f646.)"""
    lab = np.full(len(counts), '', object)
    for name, lo, hi in bins:
        m = (counts >= lo) & (counts <= (hi if hi is not None else np.iinfo(np.int64).max))
        if (lab[m] != '').any():
            raise Stop('STOP overlapping stratum bins')
        lab[m] = name
    if (lab == '').any():
        raise Stop('STOP row outside every stratum bin')
    return lab


def take(d, m):
    """Row subset of a dict of equally long arrays."""
    return {k: v[m] for k, v in d.items()}


def probs_c0(z, T):
    """Exactly train_seq.seven_leaf / evaluate_seq.seven_from_margins: leaves(softmax(zp/Tp), expit(zw/Tw), expit(zg/Tg))."""
    return leaves(softmax(z['parent'] / T['parent'], axis=1), expit(z['narrow_whiff'] / T['narrow_whiff']),
                  expit(z['ground_ball_per_pitch'] / T['ground_ball_per_pitch']))


def fit_arms(z, lab, fold, T, reg, guard, tag):
    """Fit C1, C2 and C3 on the given rows ONLY. z: the three head margins; lab: y10, y7, whiff, gb; fold: game-grouped
    fold id of every given row (used by the C2 lambda CV; every distinct id is held out once)."""
    A, opt = reg['arms'], reg['optimiser']
    y10 = lab['y10']
    wr, br = y10 == 1, np.isin(y10, BIP)
    vec, vinfo = fit_vector(z['parent'], y10, T['parent'], opt, tag + ' C1 parent')
    pw, winfo = fit_platt(z['narrow_whiff'][wr], lab['whiff'][wr], T['narrow_whiff'], opt, tag + ' Platt narrow_whiff')
    pg, ginfo = fit_platt(z['ground_ball_per_pitch'][br], lab['gb'][br], T['ground_ball_per_pitch'], opt, tag + ' Platt ground_ball_per_pitch')
    guard(tag + ' C1')
    u = log_softmax(z['parent'], axis=1)
    lam, lcv = select_lambda(u, y10, fold, T['parent'], A['C2'], opt, guard, tag)
    dr, dinfo = fit_dirichlet(u, y10, lam, lam * A['C2']['mu_over_lambda'], T['parent'], opt, '%s C2 refit lambda=%g' % (tag, lam))
    guard(tag + ' C2')
    blocks = isotonic_fit(probs_c0(z, T), lab['y7'])
    guard(tag + ' C3')
    return {'T': T, 'vec': vec, 'pw': pw, 'pg': pg, 'dir': dr, 'lam': lam, 'blocks': blocks, 'clip': A['C3']['clip'],
            'info': {'rows': int(len(y10)), 'whiff_rows': int(wr.sum()), 'bip_rows': int(br.sum()), 'vector': vinfo,
                     'platt_narrow_whiff': winfo, 'platt_ground_ball_per_pitch': ginfo, 'dirichlet': dinfo, 'lambda_cv': lcv}}


def apply_arms(model, z):
    """Seven-leaf probabilities of all four arms for the rows of z under one fitted model."""
    q0 = probs_c0(z, model['T'])
    qw = expit(model['pw']['a'] * z['narrow_whiff'] + model['pw']['b'])
    qg = expit(model['pg']['a'] * z['ground_ball_per_pitch'] + model['pg']['b'])
    c1 = leaves(softmax(z['parent'] * model['vec']['w'] + model['vec']['b'], axis=1), qw, qg)
    c2 = leaves(softmax(log_softmax(z['parent'], axis=1) @ model['dir']['W'].T + model['dir']['b'], axis=1), qw, qg)
    return {'C0': q0, 'C1': c1, 'C2': c2, 'C3': isotonic_apply(model['blocks'], q0, model['clip'])}


def select_arm(means, nll_margin):
    """Registered selection rule. means: {arm: {'nll': CV mean seven-leaf NLL, 'ece': CV mean top-label ECE}}.
    Eligible = arms with CV mean NLL <= C0's + nll_margin (C0 always is); selected = lowest CV mean ECE among the
    eligible; an exact ECE tie goes to the earlier arm in (C0, C1, C2, C3), so ties -> C0."""
    ceiling = means['C0']['nll'] + nll_margin
    eligible = [a for a in ARMS if means[a]['nll'] <= ceiling]
    return min(eligible, key=lambda a: (means[a]['ece'], ARMS.index(a))), eligible, ceiling


def cv_select(z, lab, fold, T, reg, guard, tag):
    """Game-grouped CV of every arm inside CAL. For each fold f the WHOLE fitting procedure (C1, the C2 lambda CV over
    the remaining folds, the C2 refit, C3) runs on the rows of the other folds only and scores fold f. C0 is the
    recorded scalar temperature (not refit). Returns the CV table, the selected arm and the out-of-fold rows."""
    ids = np.unique(fold).tolist()
    n = len(fold)
    oof = {a: np.full((n, 7), np.nan) for a in ARMS}
    by = {a: {'nll': [], 'brier': [], 'ece': []} for a in ARMS}
    lam_by_fold = []
    for f in ids:
        tr, ho = fold != f, fold == f
        model = fit_arms(take(z, tr), take(lab, tr), fold[tr], T, reg, guard, '%s selCV fold=%d' % (tag, f))
        q = apply_arms(model, take(z, ho))
        lam_by_fold.append(model['lam'])
        for a in ARMS:
            check_probs(q[a], '%s selCV fold=%d %s' % (tag, f, a), reg['metrics']['row_sum_tolerance'])
            ll, bs = row_loss(q[a], lab['y7'][ho])
            by[a]['nll'].append(float(ll.mean()))
            by[a]['brier'].append(float(bs.mean()))
            by[a]['ece'].append(ece(q[a], lab['y7'][ho]))
            oof[a][ho] = q[a]
        guard('%s selCV fold=%d' % (tag, f))
    per, means = {}, {}
    for a in ARMS:
        ll, bs = row_loss(oof[a], lab['y7'])
        means[a] = {'nll': float(np.mean(by[a]['nll'])), 'ece': float(np.mean(by[a]['ece']))}
        per[a] = {'nll_by_fold': by[a]['nll'], 'brier_by_fold': by[a]['brier'], 'ece_top_label_10bin_by_fold': by[a]['ece'],
                  'cv_mean_nll': means[a]['nll'], 'cv_mean_brier': float(np.mean(by[a]['brier'])), 'cv_mean_ece_top_label_10bin': means[a]['ece'],
                  'pooled_out_of_fold_report_only': {'nll': float(ll.mean()), 'brier': float(bs.mean()), 'ece_top_label_10bin': ece(oof[a], lab['y7'])}}
    selected, eligible, ceiling = select_arm(means, reg['selection']['nll_margin'])
    return {'fold_ids': ids, 'per_arm': per, 'c2_lambda_by_outer_fold': lam_by_fold, 'nll_margin': reg['selection']['nll_margin'],
            'nll_ceiling_c0_plus_margin': ceiling, 'eligible': eligible, 'selected': selected}, oof


def model_params(m):
    """JSON form of one fitted model (all four arms)."""
    i = m['info']
    return {'C0': {'temperatures': dict(m['T'])},
            'C1': {'parent': {'w': m['vec']['w'].tolist(), 'b': m['vec']['b'].tolist(), 'b_centered': (m['vec']['b'] - m['vec']['b'].mean()).tolist(),
                              'fit_rows': i['rows'], 'optimiser': i['vector'], 'note': 'softmax is invariant to a constant added to b; b_centered = b - mean(b)'},
                   'heads': {'narrow_whiff': dict(m['pw'], fit_rows=i['whiff_rows'], optimiser=i['platt_narrow_whiff']),
                             'ground_ball_per_pitch': dict(m['pg'], fit_rows=i['bip_rows'], optimiser=i['platt_ground_ball_per_pitch'])}},
            'C2': {'parent': {'lambda': m['lam'], 'W': m['dir']['W'].tolist(), 'b': m['dir']['b'].tolist(), 'fit_rows': i['rows'], 'optimiser': i['dirichlet']},
                   'lambda_cv': i['lambda_cv'], 'heads': 'identical to C1 heads (same fitted a, b)'},
            'C3': {'clip': m['clip'], 'per_leaf': {name: {'blocks': int(len(b['value'])), 'left': b['left'].tolist(), 'right': b['right'].tolist(),
                                                           'value': b['value'].tolist(), 'weight': b['weight'].tolist()} for name, b in zip(LEAVES, m['blocks'])}}}


def build_predictions(S, reg, guard=lambda stage: None):
    """All probabilities of the experiment. S: seeds (list), runs[seed] = {T, z[role][head], persisted (EVAL seven-leaf)},
    lab[split] = {y10, y7, whiff, gb} and games[split] for split in (calibration, evaluation), reference (array|None).
    Returns per seed: Q[arm][split], cv table, selected arm, parameters, gates; and the EVAL ensembles incl. SEL."""
    cal, ev = 'calibration', 'evaluation'
    G, tol = reg['gates'], reg['metrics']['row_sum_tolerance']
    fold, chunks = cv_folds(S['games'][cal], reg['selection']['cv_seed'], reg['selection']['cv_folds'])
    out = {'seeds': {}, 'folds': {'cv_seed': reg['selection']['cv_seed'], 'cv_folds': reg['selection']['cv_folds'],
                                  'fold_rows': np.bincount(fold, minlength=len(chunks)).tolist(), 'fold_games': [int(len(c)) for c in chunks],
                                  'fold_game_pks': [np.sort(c).tolist() for c in chunks]}}
    for s in S['seeds']:
        R, tag = S['runs'][s], 'seed=%d' % s
        for h in HEADS:
            if not (isinstance(R['T'][h], float) and np.isfinite(R['T'][h]) and R['T'][h] > 0):
                raise Stop('STOP recorded temperature %s %s' % (tag, h))
        for sp in (cal, ev):
            n = len(S['lab'][sp]['y7'])
            if R['z'][sp]['parent'].shape != (n, K10) or any(R['z'][sp][h].shape != (n,) for h in HEADS[1:]) \
                    or not all(np.isfinite(R['z'][sp][h]).all() for h in HEADS):
                raise Stop('STOP margin shape or non-finite: %s %s' % (tag, sp))
        q0 = probs_c0(R['z'][ev], R['T'])
        diff = float(np.abs(q0 - R['persisted']).max()) if R['persisted'].shape == q0.shape else float('inf')
        if not diff <= G['c0_reproduction_max_abs']:
            raise Stop('STOP C0 does not reproduce the persisted seven-leaf evaluation array, %s: max abs diff %r' % (tag, diff))
        model = fit_arms(R['z'][cal], S['lab'][cal], fold, R['T'], reg, guard, tag + ' full CAL')
        Q = {a: {} for a in ARMS}
        for sp in (cal, ev):
            q = apply_arms(model, R['z'][sp])
            for a in ARMS:
                check_probs(q[a], '%s %s %s' % (tag, a, sp), tol)
                Q[a][sp] = q[a]
        cv, _ = cv_select(R['z'][cal], S['lab'][cal], fold, R['T'], reg, guard, tag)
        out['seeds'][s] = {'Q': Q, 'cv': cv, 'selected': cv['selected'], 'parameters': model_params(model),
                           'gate': {'c0_vs_persisted_evaluation_max_abs': diff, 'tolerance': G['c0_reproduction_max_abs']}}
        guard(tag)
    ens = {a: np.mean([out['seeds'][s]['Q'][a][ev] for s in S['seeds']], axis=0) for a in ARMS}
    ens[SEL] = np.mean([out['seeds'][s]['Q'][out['seeds'][s]['selected']][ev] for s in S['seeds']], axis=0)
    for a, q in ens.items():
        check_probs(q, 'ensemble ' + a, tol)
    if S.get('reference') is not None:
        d = float(np.abs(ens['C0'] - S['reference']).max()) if S['reference'].shape == ens['C0'].shape else float('inf')
        out['reference_gate'] = {'checked': True, 'c0_ensemble_vs_reference_max_abs': d, 'tolerance': G['reference_ensemble_max_abs']}
        if not d <= G['reference_ensemble_max_abs']:
            raise Stop('STOP the C0 ensemble does not reproduce the registered reference ensemble: max abs diff %r' % d)
    else:
        out['reference_gate'] = {'checked': False, 'note': 'no reference ensemble registered'}
    out['ensemble'] = ens
    return out


# ---------- metrics / reporting ----------
def leaf_block(q, y, k, th):
    """STEP5-style block for one leaf (one-vs-rest): 10-bin ECE, mean predicted vs base rate, reliability table with the
    gap, and the descriptive checks over populated bins (n >= populated_bin_min_n)."""
    p = q[:, k]
    hit = (y == k).astype(float)
    b = np.minimum((p * BINS).astype(int), BINS - 1)
    n = np.bincount(b, minlength=BINS)
    sp = np.bincount(b, p, BINS)
    so = np.bincount(b, hit, BINS)
    rel = [{'bin': i, 'lo': EDGES[i], 'hi': EDGES[i + 1], 'n': int(n[i]), 'mean_predicted': float(sp[i] / n[i]) if n[i] else None,
            'observed_frequency': float(so[i] / n[i]) if n[i] else None,
            'gap_observed_minus_predicted': float((so[i] - sp[i]) / n[i]) if n[i] else None} for i in range(BINS)]
    pop = [i for i in range(BINS) if n[i] >= th['populated_bin_min_n']]
    obs = [so[i] / n[i] for i in pop]
    viol = [[pop[j], pop[j + 1]] for j in range(len(pop) - 1) if not obs[j + 1] > obs[j]]
    gaps = [abs(float((so[i] - sp[i]) / n[i])) for i in pop]
    e = float(np.abs(np.bincount(b, p - hit, BINS)).sum() / len(y))
    return {'ece_10bin': e, 'mean_predicted': float(p.mean()), 'base_rate': float(hit.mean()),
            'mean_predicted_minus_base_rate': float(p.mean() - hit.mean()), 'reliability': rel,
            'checks': {'populated_min_n': th['populated_bin_min_n'], 'populated_bins': pop,
                       'monotone': None if len(pop) < 2 else not viol, 'monotone_violations': viol,
                       'max_abs_bin_gap': max(gaps) if gaps else None, 'max_abs_bin_gap_threshold': th['max_abs_bin_gap'],
                       'bin_gap_ok': None if not gaps else bool(max(gaps) <= th['max_abs_bin_gap']),
                       'leaf_ece_threshold': th['leaf_ece_max'], 'leaf_ece_ok': bool(e <= th['leaf_ece_max'])}}


def arm_summary(q, y):
    """NLL, Brier, primary top-label ECE and the classwise ECE of one arm on one set of rows."""
    ll, bs = row_loss(q, y)
    return {'rows': int(len(y)), 'nll': float(ll.mean()), 'brier': float(bs.mean()), 'ece_top_label_10bin': ece(q, y),
            'classwise_ece_10bin': classwise_ece(q, y)}


def arm_full(q, y, th):
    """arm_summary + the top-label reliability table + the per-leaf blocks (all seven leaves)."""
    return dict(arm_summary(q, y), top_label_reliability=top_label_reliability(q, y),
                leaves={name: leaf_block(q, y, k, th) for k, name in enumerate(LEAVES)})


def bootstrap_multi(arms, pairs, y, games, draws, seed, chunk, leaf_idx=(), abs_arms=()):
    """Whole-game paired bootstrap (eop-001 contract) for several (arm, reference) pairs under ONE stream of game
    draws. Per draw and arm: NLL/Brier = sum of sampled games' row losses / sampled rows; top-label ECE = sum over the
    10 bins of |sum of sampled games' (conf - hit) bin sums| / sampled rows; for a leaf k in leaf_idx the one-vs-rest
    leaf ECE from per-game bin sums of (q_k - 1[y=k]) and the mean gap (mean predicted - base rate). Difference = arm
    - reference. For nll/brier/ece the arithmetic is the verbatim eop bootstrap (checked in --selfcheck).
    Returns (pair CIs, absolute CIs of abs_arms, info)."""
    u, inv = np.unique(games, return_inverse=True)
    G = len(u)
    n = np.bincount(inv, minlength=G).astype(float)
    need = [a for a in arms if a in {x for p in pairs for x in p} | set(abs_arms)]
    d = {}
    for a in need:
        q = arms[a]
        ll, bs = row_loss(q, y)
        conf, hit, b = top(q, y)
        gap = np.zeros((G, BINS))
        np.add.at(gap, (inv, b), conf - hit)
        lg = {}
        for k in leaf_idx:
            p = q[:, k]
            g = np.zeros((G, BINS))
            np.add.at(g, (inv, np.minimum((p * BINS).astype(int), BINS - 1)), p - (y == k))
            lg[k] = g
        d[a] = (np.bincount(inv, ll, G), np.bincount(inv, bs, G), gap, lg)
    rng = np.random.default_rng(seed)
    keys = ['nll', 'brier', 'ece'] + [('leaf_ece', k) for k in leaf_idx]
    P = {'%s-%s' % p: {m: np.empty(draws) for m in keys} for p in pairs}
    A = {a: {m: np.empty(draws) for m in keys + [('leaf_gap', k) for k in leaf_idx]} for a in abs_arms}
    for s in range(0, draws, chunk):
        pick = rng.integers(0, G, (min(chunk, draws - s), G))
        N = n[pick].sum(1)
        sl = slice(s, s + len(pick))
        tops = {a: np.abs(d[a][2][pick].sum(1)).sum(1) for a in need}
        lsum = {a: {k: d[a][3][k][pick].sum(1) for k in leaf_idx} for a in need}
        for a, r in pairs:
            o = P['%s-%s' % (a, r)]
            o['nll'][sl] = (d[a][0] - d[r][0])[pick].sum(1) / N
            o['brier'][sl] = (d[a][1] - d[r][1])[pick].sum(1) / N
            o['ece'][sl] = (tops[a] - tops[r]) / N
            for k in leaf_idx:
                o[('leaf_ece', k)][sl] = (np.abs(lsum[a][k]).sum(1) - np.abs(lsum[r][k]).sum(1)) / N
        for a in abs_arms:
            o = A[a]
            o['nll'][sl] = d[a][0][pick].sum(1) / N
            o['brier'][sl] = d[a][1][pick].sum(1) / N
            o['ece'][sl] = tops[a] / N
            for k in leaf_idx:
                o[('leaf_ece', k)][sl] = np.abs(lsum[a][k]).sum(1) / N
                o[('leaf_gap', k)][sl] = lsum[a][k].sum(1) / N
    q95 = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]

    def pack(o):
        r = {m: q95(o[m]) for m in ('nll', 'brier', 'ece')}
        r['leaf_ece'] = {LEAVES[k]: q95(o[('leaf_ece', k)]) for k in leaf_idx}
        if ('leaf_gap', leaf_idx[0] if leaf_idx else None) in o:
            r['leaf_gap'] = {LEAVES[k]: q95(o[('leaf_gap', k)]) for k in leaf_idx}
        return r
    return {p: pack(o) for p, o in P.items()}, {a: pack(o) for a, o in A.items()}, \
        {'games': int(G), 'rows': int(len(y)), 'draws': int(draws), 'seed': int(seed), 'chunk': int(chunk), 'unit': 'game_pk',
         'ci': 'percentile 95% (numpy quantile .025/.975, linear interpolation)'}


MKEY = {'nll': 'nll', 'brier': 'brier', 'ece': 'ece_top_label_10bin'}


def paired_entry(ma, mr, ci):
    """Point difference arm - reference (from the row-level metrics) + the bootstrap CI95 of that difference."""
    out = {MKEY[m]: {'difference': ma[MKEY[m]] - mr[MKEY[m]], 'ci95_percentile': ci[m]} for m in MKEY}
    if ci.get('leaf_ece'):
        out['leaf_ece_10bin'] = {name: {'difference': ma['classwise_ece_10bin']['per_leaf'][name] - mr['classwise_ece_10bin']['per_leaf'][name],
                                        'ci95_percentile': c} for name, c in ci['leaf_ece'].items()}
    return out


def compute(S, reg, guard=lambda stage: None):
    """Fit, select, ensemble, score. Returns the result.json payload. S as in build_predictions plus meta (pitcher,
    fit_pitcher_rows, fit_batter_rows on EVAL rows) and baselines {name: (n_eval,7) probabilities}."""
    cal, ev = 'calibration', 'evaluation'
    B = build_predictions(S, reg, guard)
    y, games, th, bs = S['lab'][ev]['y7'], S['games'][ev], reg['fan_display']['thresholds'], reg['bootstrap']
    boot = lambda arms, pairs, yy, gg, **kw: bootstrap_multi(arms, pairs, yy, gg, bs['draws'], bs['seed'], bs['chunk'], **kw)
    den = {sp: {'rows': int(len(S['lab'][sp]['y7'])), 'games': int(len(np.unique(S['games'][sp]))),
                'leaf_counts': np.bincount(S['lab'][sp]['y7'], minlength=7).tolist()} for sp in (cal, ev)}
    den[ev]['pitchers'] = int(len(np.unique(S['meta']['pitcher'])))

    # ---- per seed: every arm vs that seed's C0 (fresh bootstrap stream per call, same seed) ----
    seeds = {}
    for s in S['seeds']:
        X = B['seeds'][s]
        m = {a: {'calibration_in_sample': arm_summary(X['Q'][a][cal], S['lab'][cal]['y7']), 'evaluation': arm_full(X['Q'][a][ev], y, th)} for a in ARMS}
        ci, _, _ = boot({a: X['Q'][a][ev] for a in ARMS}, SEED_PAIRS, y, games)
        seeds[str(s)] = {'temperatures': dict(S['runs'][s]['T']), 'cal_selected': X['selected'], 'selection_cv': X['cv'], 'gate': X['gate'],
                         'arms': m, 'paired_evaluation': {p: paired_entry(m[p[:2]][ev], m[p[3:]][ev], ci[p]) for p in ci},
                         'advisory_labels_vs_C0': {a: advisory_labels(ci['%s-C0' % a], reg['margins']) for a in ARMS[1:]},
                         'parameters': X['parameters']}
        guard('report seed=%d' % s)

    # ---- ensemble: C0..C3 and SEL vs the C0 ensemble; SEL and C0 vs every baseline; per-leaf CIs ----
    E = dict(B['ensemble'])
    names = list(ARMS) + [SEL]
    m = {a: arm_full(E[a], y, th) for a in names}
    bm = {b: arm_full(q, y, th) for b, q in S['baselines'].items()}
    pairs = [(a, 'C0') for a in names[1:]] + [('C2', 'C1')] + [(a, b) for b in S['baselines'] for a in (SEL, 'C0')]
    ci, ab, binfo = boot(dict(E, **S['baselines']), pairs, y, games, leaf_idx=tuple(range(7)), abs_arms=tuple(names))
    allm = dict(m, **bm)
    for a in names:
        m[a]['ci95_percentile'] = {MKEY[k]: ab[a][k] for k in MKEY}
        for name in LEAVES:
            m[a]['leaves'][name]['ece_ci95'] = ab[a]['leaf_ece'][name]
            m[a]['leaves'][name]['mean_gap_ci95'] = ab[a]['leaf_gap'][name]
    paired = {'%s-%s' % p: paired_entry(allm[p[0]], allm[p[1]], ci['%s-%s' % p]) for p in pairs}
    fan = {name: {a: {'ece_10bin': m[a]['leaves'][name]['ece_10bin'], 'ece_ci95': m[a]['leaves'][name]['ece_ci95'],
                      'mean_predicted_minus_base_rate': m[a]['leaves'][name]['mean_predicted_minus_base_rate'],
                      'mean_gap_ci95': m[a]['leaves'][name]['mean_gap_ci95'], 'checks': m[a]['leaves'][name]['checks'],
                      'paired_vs_C0': None if a == 'C0' else paired['%s-C0' % a]['leaf_ece_10bin'][name]} for a in names}
           for name in reg['fan_display']['leaves']}
    guard('report ensemble')

    # ---- strata (descriptive; paired CI for SEL - C0 only; fresh stream per stratum) ----
    strata, unstable = {}, reg['strata']['small_n_unstable']
    core = dict(E, **S['baselines'])
    for key, counts in (('pitcher_fit_rows', S['meta']['fit_pitcher_rows']), ('batter_fit_rows', S['meta']['fit_batter_rows'])):
        lab = assign_bins(counts, reg['strata'][key]['bins'])
        strata[key] = {}
        for name, _, _ in reg['strata'][key]['bins']:
            k = lab == name
            rows = int(k.sum())
            if rows == 0:
                strata[key][name] = {'rows': 0, 'games': 0, 'pitchers': 0, 'note': 'empty stratum: nothing computed'}
                continue
            sm = {a: {x: v for x, v in arm_summary(q[k], y[k]).items() if x != 'classwise_ece_10bin'} for a, q in core.items()}
            sci, _, su = boot({SEL: E[SEL][k], 'C0': E['C0'][k]}, [(SEL, 'C0')], y[k], games[k])
            strata[key][name] = {'rows': rows, 'games': su['games'], 'pitchers': int(len(np.unique(S['meta']['pitcher'][k]))),
                                 'small_n_unstable': bool(rows < unstable['rows_below'] or su['games'] < unstable['games_below']),
                                 'arms': sm, 'paired': {'SEL-C0': paired_entry(sm[SEL], sm['C0'], sci['SEL-C0'])}}
        if sum(v['rows'] for v in strata[key].values()) != den[ev]['rows']:
            raise Stop('STOP stratum rows do not reconcile to the EVAL rows: ' + key)
        guard('report strata ' + key)

    sel = {str(s): B['seeds'][s]['selected'] for s in S['seeds']}
    return {'id': reg['id'], 'attempt_id': reg['attempt']['id'], 'synthetic': 'synthetic_selfcheck' in reg,
            'data_role': reg['data_role'], 'claim_ceiling': reg['claim_ceiling'], 'evaluation_exposure': reg['evaluation_exposure'],
            'sign': 'difference = arm - reference on EVAL; negative = lower loss / lower ECE for the arm',
            'arm_prefix': S['arm_prefix'], 'seeds_evaluated': list(S['seeds']), 'leaf_order': LEAVES, 'bin_edges': EDGES,
            'definitions': {
                'nll': 'mean of -log(clip(q[y7], 1e-15, 1))', 'brier': 'mean over rows of the sum over 7 leaves of squared error',
                'ece_top_label_10bin': 'sum over 10 equal-width bins of |sum(conf - hit)| / n, bin = min(floor(10*conf), 9) (compare_e_op.ece) [PRIMARY]',
                'leaf_ece_10bin': 'one-vs-rest for one leaf: sum over 10 equal-width bins of |sum(q_leaf - 1[y = leaf])| / n, bin = min(floor(10*q_leaf), 9)',
                'reliability_table': 'per leaf, 10 equal-width bins: n, mean predicted, observed frequency, gap = observed - predicted',
                'ensemble': 'mean over the seeds of the calibrated seven-leaf rows of the same arm',
                'SEL': "mean over the seeds of each seed's own cal_selected arm (the calibrator the registered rule would deploy)",
                'ci': 'whole-game bootstrap (eop-001): ratio of sampled games\' sums to sampled games\' pitch count, ECE from summed per-game bin sums, percentile 95%; differences use the same sampled games for both arms; every bootstrap call starts a fresh default_rng(seed)'},
            'denominators': den, 'cv_folds_in_calibration': B['folds'],
            'gates': {'per_seed_c0_reproduction': {str(s): B['seeds'][s]['gate'] for s in S['seeds']}, 'reference_ensemble': B['reference_gate'],
                      'inputs': S.get('input_gates')},
            'cal_selected': sel, 'cal_selected_rule': reg['selection']['rule'],
            'seeds': seeds,
            'ensemble': {'arms': m, 'paired_evaluation': paired,
                         'advisory_labels_vs_C0': {a: advisory_labels(ci['%s-C0' % a], reg['margins']) for a in names[1:]},
                         'fan_display_leaves': fan, 'baselines_report_only': bm,
                         'note': 'SEL equals the C0 ensemble exactly (differences 0, CI [0,0]) when every seed selects C0'},
            'strata': strata, 'bootstrap': binfo, 'fan_display_thresholds': th,
            'advisory_label_margins': reg['margins'],
            'advisory_note': 'pre-registered descriptive labels on EXPOSED DEVELOPMENT rows; not a winner, not adoption; arms are compared without multiplicity correction',
            'calibration_split_note': 'calibration_in_sample metrics are in-sample for C1/C2/C3 and descriptive only; selection_cv is out-of-fold inside CAL'}


# ---------- registration / gates / IO ----------
def load_registration(path):
    return json.loads(Path(path).read_text())


def constants_match(reg):
    """The registration repeats the constants baked into the copied functions and the fixed structure; drift is refused."""
    m, bad = reg['metrics'], []
    if m['bins'] != BINS or m['nll_clip'] != CLIP or reg['leaf_order'] != LEAVES or reg['bip_parent_classes'] != list(BIP):
        bad.append('metric constants / leaf order / BIP')
    if sorted(reg['arms']) != sorted(ARMS) or reg['paired_comparisons']['per_seed'] != ['%s-%s' % p for p in SEED_PAIRS]:
        bad.append('arms / per-seed pairs')
    if reg['selection']['cv_seed'] != reg['arms']['C2']['cv_seed'] or reg['selection']['cv_folds'] != reg['arms']['C2']['cv_folds']:
        bad.append('selection CV folds must be the C2 lambda CV folds')
    if not set(reg['fan_display']['leaves']) <= set(LEAVES):
        bad.append('fan display leaves')
    if reg['caps']['threads'] != int(THREADS):
        bad.append('threads')
    if reg['sources'][0] != SCRIPT_KEY or len(reg['sources']) != 2:
        bad.append('sources')
    x = reg['inputs']
    if sorted(x['run_files']) != sorted('%s-s%d' % (x['arm_prefix'], s) for s in x['seeds']) or len(set(x['seeds'])) != len(x['seeds']) or not x['seeds'] \
            or any(sorted(v) != sorted(RUN_FILES) for v in x['run_files'].values()):
        bad.append('run_files do not match arm_prefix / seeds / the four run files')
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', b) or b in ARMS + (SEL,) for b in x['baselines']):
        bad.append('baseline names')
    return bad


def input_files(reg):
    """Every registered input file: name -> {path, sha256, bytes}. Order: runs, corpus, eval-meta, baselines, reference."""
    x = reg['inputs']
    out = {}
    for run in sorted(x['run_files']):
        for f in RUN_FILES:
            out['%s/%s' % (run, f)] = dict(x['run_files'][run][f], path=str(Path(x['runs_dir']) / run / f))
    out['corpus'] = x['corpus']
    out['eval_meta'] = x['eval_meta']
    for b, v in x['baselines'].items():
        out['baseline:' + b] = v
    if x.get('reference_ensemble'):
        out['reference_ensemble'] = x['reference_ensemble']
    return {k: {'path': v['path'], 'sha256': v['sha256'], 'bytes': v['bytes']} for k, v in out.items()}


def cli_mismatches(reg, reg_path, a):
    """The command line must repeat the registered inputs and attempt directory exactly (resolved paths)."""
    x, same = reg['inputs'], lambda p, q: p is not None and q is not None and Path(p).resolve() == Path(q).resolve()
    bad = []
    if not same(a['runs'], x['runs_dir']):
        bad.append('--runs')
    if a['arm_prefix'] != x['arm_prefix']:
        bad.append('--arm-prefix')
    if a['seeds'] != list(x['seeds']):
        bad.append('--seeds')
    if not same(a['corpus'], x['corpus']['path']):
        bad.append('--corpus')
    if not same(a['eval_meta'], x['eval_meta']['path']):
        bad.append('--eval-meta')
    if list(a['baselines']) != list(x['baselines']) or not all(same(p, x['baselines'][b]['path']) for b, p in a['baselines'].items()):
        bad.append('--baselines')
    ref = x.get('reference_ensemble')
    if (a['reference_ensemble'] is None) != (not ref) or (ref and not same(a['reference_ensemble'], ref['path'])):
        bad.append('--reference-ensemble')
    if not same(a['out'], attempt_dir(reg, reg_path)):
        bad.append('--out (must be <registration dir>/%s/%s)' % (reg['attempt']['root'], reg['attempt']['id']))
    return bad


def args_from_registration(reg, reg_path):
    """The argument set that cli_mismatches() accepts (used by --plan to print the command and by --selfcheck)."""
    x = reg['inputs']
    return {'runs': x['runs_dir'], 'arm_prefix': x['arm_prefix'], 'seeds': list(x['seeds']), 'corpus': x['corpus']['path'],
            'eval_meta': x['eval_meta']['path'], 'baselines': {b: v['path'] for b, v in x['baselines'].items()},
            'reference_ensemble': x['reference_ensemble']['path'] if x.get('reference_ensemble') else None,
            'out': str(attempt_dir(reg, reg_path))}


def command_line(reg, reg_path, mode):
    a = args_from_registration(reg, reg_path)
    c = [reg['python']['executable'], '-I', str(SELF), mode, '--runs', a['runs'], '--arm-prefix', a['arm_prefix'], '--seeds', ','.join(map(str, a['seeds'])),
         '--corpus', a['corpus'], '--eval-meta', a['eval_meta'], '--baselines', ','.join('%s=%s' % kv for kv in a['baselines'].items())]
    if a['reference_ensemble']:
        c += ['--reference-ensemble', a['reference_ensemble']]
    return ' '.join("'%s'" % t if ' ' in t else t for t in c + ['--registration', str(reg_path), '--out', a['out']])


def attempt_dir(reg, reg_path):
    return Path(reg_path).resolve().parent / reg['attempt']['root'] / reg['attempt']['id']


def synthetic_guard(reg, reg_path):
    """A fabricated (selfcheck) registration can never be pointed at anything outside a selfcheck temp directory."""
    if 'synthetic_selfcheck' in reg:
        paths = [reg_path, reg['lock']['path']] + [v['path'] for v in input_files(reg).values()]
        if not all(SYNTH_DIR in Path(p).resolve().parts for p in paths):
            raise SystemExit('BLOCKED: a synthetic registration may only be used under a %s directory' % SYNTH_DIR)


def review_gate(reg, reg_path):
    """BLOCKED (SystemExit, nothing written) unless the registration is frozen and independently reviewed. The review
    file is read from the REGISTRATION's directory and must bind the sha256 of this script and of the registration."""
    if reg.get('status') != READY:
        raise SystemExit('BLOCKED: registration status is not exactly ' + READY)
    bad = constants_match(reg)
    if bad:
        raise SystemExit('BLOCKED: registration does not match the source constants: ' + '; '.join(bad))
    if reg['sources'][1] != Path(reg_path).name:
        raise SystemExit('BLOCKED: registration sources[1] is not the registration file name')
    rp = Path(reg_path).parent / reg['review']['file']
    if not rp.is_file():
        raise SystemExit('BLOCKED: independent %s required next to the registration' % reg['review']['file'])
    rev = json.loads(rp.read_text())
    src = {SCRIPT_KEY: sha(SELF), Path(reg_path).name: sha(reg_path)}
    author, reviewer = reg.get('author'), rev.get('reviewer')
    if rev.get('verdict') != 'PASS':
        raise SystemExit('BLOCKED: review verdict is not PASS')
    if not isinstance(author, str) or not isinstance(reviewer, str) or not reviewer.strip() \
            or reviewer.strip().casefold() == author.strip().casefold():
        raise SystemExit('BLOCKED: reviewer missing or reviewer == author')
    if rev.get('sources_sha256') != src:
        raise SystemExit('BLOCKED: review sources_sha256 != current sha256 of %s and %s' % (SCRIPT_KEY, Path(reg_path).name))
    return {'review_sha256': sha(rp), 'sources_sha256': src, 'reviewer': reviewer}


def npz_members(path, spec):
    """Load exactly the registered members and check shape and dtype."""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for m, want in spec.items():
            if m not in z.files:
                raise Stop('STOP missing member %s in %s' % (m, path))
            a = z[m]
            if list(a.shape) != want['shape'] or str(a.dtype) != want['dtype']:
                raise Stop('STOP member shape/dtype %s in %s: %s %s' % (m, path, a.shape, a.dtype))
            out[m] = a
    return out


def load_inputs(reg):
    """Read the registered arrays (after their sha256 was verified) and apply the binding / label / denominator gates.
    Contract: every margins member is in masked-corpus row order of its role (train_seq.py writes it that way); the
    SELECT-NLL receipt gate and the temperature fit-row gate are the registered alignment proofs."""
    x, G, exp = reg['inputs'], reg['gates'], reg['expected']
    C = npz_members(x['corpus']['path'], x['corpus']['members'])
    M = npz_members(x['eval_meta']['path'], x['eval_meta']['members'])
    lab, games, rows = {}, {}, {}
    for role in ROLES:
        mask = C[role]
        y10 = C['y10'][mask]
        w, gb, y7 = C['narrow_whiff'][mask].astype(np.int64), C['ground_ball_per_pitch'][mask].astype(np.int64), C['y7'][mask]
        if not (np.isin(w, (0, 1)).all() and np.isin(gb, (0, 1)).all()) or not np.array_equal(derive_y7(y10, w, gb), y7) \
                or y7.min() < 0 or y7.max() > 6:
            raise Stop('STOP label coherence: y7 != leaf map of (y10, narrow_whiff, ground_ball_per_pitch) on ' + role)
        lab[role], games[role], rows[role] = {'y10': y10, 'y7': y7, 'whiff': w, 'gb': gb}, C['games'][mask], int(mask.sum())
    if (C['calibration'] & C['evaluation']).any() or (C['tuning'] & (C['calibration'] | C['evaluation'])).any():
        raise Stop('STOP role masks overlap')
    if rows['tuning'] != exp['tuning_rows']:
        raise Stop('STOP tuning rows %d != registered %d' % (rows['tuning'], exp['tuning_rows']))
    for sp in ('calibration', 'evaluation'):
        den = {'rows': rows[sp], 'games': int(len(np.unique(games[sp]))), 'leaf_counts': np.bincount(lab[sp]['y7'], minlength=7).tolist()}
        if den != {k: exp[sp][k] for k in den}:
            raise Stop('STOP denominators %s %s != registered' % (sp, json.dumps(den)))
    if np.intersect1d(games['calibration'], games['evaluation']).size:
        raise Stop('STOP a game_pk appears in both CAL and EVAL')
    ev = C['evaluation']
    same = [np.array_equal(M['keys'], C['keys'][ev]), np.array_equal(M['y7'], lab['evaluation']['y7']),
            np.array_equal(M['y10'], lab['evaluation']['y10']), np.array_equal(M['games'], games['evaluation'])]
    if not all(same) or (M['fit_pitcher_rows'] < 0).any() or (M['fit_batter_rows'] < 0).any():
        raise Stop('STOP eval-meta is not the EVAL slice of the corpus (keys/y7/y10/games) or has negative counts: ' + str(same))
    n_ev = rows['evaluation']
    cal_fit_rows = {'parent': rows['calibration'], 'narrow_whiff': int((lab['calibration']['y10'] == 1).sum()),
                    'ground_ball_per_pitch': int(np.isin(lab['calibration']['y10'], BIP).sum())}
    runs, ig = {}, {}
    for s in x['seeds']:
        name = '%s-s%d' % (x['arm_prefix'], s)
        d = Path(x['runs_dir']) / name
        r = json.loads((d / 'receipt.json').read_text())
        tj = json.loads((d / 'temperatures.json').read_text())
        T = tj.get('temperatures')
        bind = [r.get('seed') == s, all(r.get(k) == v for k, v in x['receipt_expect'].items()),
                all(r.get('outputs_sha256', {}).get(f) == x['run_files'][name][f]['sha256'] for f in RUN_FILES[:3]),
                isinstance(T, dict) and set(T) == set(HEADS) and r.get('temperatures') == T,
                all(r.get('rows', {}).get(k) == rows[k] for k in ROLES)]
        if not all(bind):
            raise Stop('STOP run %s: receipt binding (seed / expected fields / output sha256 / temperatures / role rows) failed: %s' % (name, bind))
        if G['temperature_fit_rows'] and (tj.get('split') != 'calibration' or tj.get('fit_rows') != cal_fit_rows):
            raise Stop('STOP run %s: temperatures.json split / fit_rows != the CAL head rows of the corpus %s' % (name, json.dumps(cal_fit_rows)))
        Z = npz_members(d / 'margins.npz', x['margins_members'])
        z = {role: {h: Z['%s__%s' % (role, h)] for h in HEADS} for role in ROLES}
        for role in ROLES:
            if z[role]['parent'].shape != (rows[role], K10) or any(z[role][h].shape != (rows[role],) for h in HEADS[1:]):
                raise Stop('STOP run %s: margins rows != corpus %s rows' % (name, role))
        ig[name] = {}
        if G['select_nll_receipt_field']:
            got = float(row_loss(probs_c0(z['tuning'], {h: 1.0 for h in HEADS}), lab['tuning']['y7'])[0].mean())
            want = r.get(G['select_nll_receipt_field'])
            ig[name]['select_nll_uncalibrated'] = {'recomputed': got, 'receipt': want, 'tolerance': G['select_nll_max_abs']}
            if not isinstance(want, float) or not abs(got - want) <= G['select_nll_max_abs']:
                raise Stop('STOP run %s: uncalibrated SELECT seven-leaf NLL recomputed from the tuning margins %r != receipt %r (row alignment)' % (name, got, want))
        pers = np.load(d / 'seven-leaf-evaluation.npy', allow_pickle=False)
        if list(pers.shape) != x['seven_leaf_evaluation']['shape'] or str(pers.dtype) != x['seven_leaf_evaluation']['dtype'] or pers.shape != (n_ev, 7):
            raise Stop('STOP run %s: persisted seven-leaf shape/dtype' % name)
        runs[s] = {'T': {h: float(T[h]) for h in HEADS}, 'z': {sp: z[sp] for sp in ('calibration', 'evaluation')}, 'persisted': pers}
        ig[name]['receipt'] = {k: r.get(k) for k in ('id', 'arm', 'seed', 'config_sha256', 'train_seq_sha256', 'manifest_sha256') if k in r}
    base = {}
    for b, v in x['baselines'].items():
        q = np.load(v['path'], allow_pickle=False)
        if q.shape != (n_ev, 7) or q.dtype != np.float64:
            raise Stop('STOP baseline shape/dtype: ' + b)
        check_probs(q, 'baseline ' + b, reg['metrics']['row_sum_tolerance'])
        base[b] = q
    ref = None
    if x.get('reference_ensemble'):
        ref = np.load(x['reference_ensemble']['path'], allow_pickle=False)
        if ref.shape != (n_ev, 7) or ref.dtype != np.float64:
            raise Stop('STOP reference ensemble shape/dtype')
    return {'arm_prefix': x['arm_prefix'], 'seeds': list(x['seeds']), 'runs': runs,
            'lab': {sp: lab[sp] for sp in ('calibration', 'evaluation')}, 'games': {sp: games[sp] for sp in ('calibration', 'evaluation')},
            'meta': {k: M[k] for k in ('pitcher', 'fit_pitcher_rows', 'fit_batter_rows')}, 'baselines': base, 'reference': ref,
            'input_gates': dict(ig, calibration_head_fit_rows=cal_fit_rows)}


def acquire_lock(spec):
    """flock(LOCK_EX|LOCK_NB) on the EXISTING common heavy lock, opened read-only; busy -> retry; never create/delete."""
    lp = Path(spec['path'])
    if not lp.is_file():
        raise Stop('STOP common heavy lock file absent (never created by this program)')
    fh = lp.open('rb')
    t0 = time.monotonic()
    tries = 0
    while True:
        tries += 1
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh, {'wait_seconds': time.monotonic() - t0, 'tries': tries}
        except OSError as ex:
            if ex.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                fh.close()
                raise Stop('STOP common heavy lock flock error: %r' % (ex,)) from None
        if time.monotonic() - t0 + spec['retry_seconds'] > spec['max_wait_seconds']:
            fh.close()
            raise Stop('STOP common heavy lock busy after %d tries / %.0f s' % (tries, time.monotonic() - t0))
        time.sleep(spec['retry_seconds'])


def _alarm(signum, frame):
    raise Stop('STOP wall cap (SIGALRM)')


def execute(reg_path, cli):
    """The real run. Order: synthetic guard, review gate, command-line == registration, memory preflight (all BLOCKED,
    nothing written) -> claim the attempt -> lock (retry) -> memory gate -> wall alarm -> bytes + sha256 of every
    input -> arrays + gates -> compute -> result.json + terminal.json."""
    reg_path = Path(reg_path).resolve()
    reg = load_registration(reg_path)
    synthetic_guard(reg, reg_path)
    gates = review_gate(reg, reg_path)
    bad = cli_mismatches(reg, reg_path, cli)
    if bad:
        raise SystemExit('BLOCKED: command line differs from the registration: ' + ', '.join(bad))
    caps = reg['caps']
    try:
        pre = free_memory_bytes(caps['vm_stat_page_bytes'])
    except Stop as ex:
        raise SystemExit('BLOCKED: ' + str(ex)) from None
    if pre < caps['min_free_memory_bytes']:
        raise SystemExit('BLOCKED: free memory %d < %d bytes (attempt not claimed; retry later)' % (pre, caps['min_free_memory_bytes']))
    att = attempt_dir(reg, reg_path)
    try:
        att.parent.mkdir(parents=True, exist_ok=True)
        att.mkdir()  # exclusive: fails if the attempt was ever claimed
    except FileExistsError:
        raise SystemExit('BLOCKED: attempt %s exists (never overwritten)' % reg['attempt']['id']) from None
    t0 = time.monotonic()
    (att / 'CLAIM').write_text(json.dumps({'pid': os.getpid(), 'utc': time.time(), 'attempt_id': reg['attempt']['id'], 'argv': sys.argv}) + '\n')
    state = {'inputs_sha256': {}, 'lock': None, 'free_memory_bytes_gate': None, 'compute_started': None}
    common = lambda: {'id': reg['id'], 'attempt_id': reg['attempt']['id'], 'elapsed_seconds': time.monotonic() - t0, 'peak_rss_bytes': peak_rss_bytes(),
                      'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
                      'threads_env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')},
                      'inputs_sha256': state['inputs_sha256'], 'registration_path': str(reg_path),
                      'registration_sha256': gates['sources_sha256'][reg_path.name], 'calibrate_logits_sha256': gates['sources_sha256'][SCRIPT_KEY],
                      'review_sha256': gates['review_sha256'], 'reviewer': gates['reviewer'], 'lock': state['lock'],
                      'free_memory_bytes_preflight': pre, 'free_memory_bytes_gate': state['free_memory_bytes_gate'],
                      'caps': 'wall: SIGALRM handler after lock acquisition + stage checks; RSS: ru_maxrss sampled at stage boundaries'}

    def guard(stage):
        if state['compute_started'] is not None and time.monotonic() - state['compute_started'] > caps['wall_seconds']:
            raise Stop('STOP wall cap at ' + stage)
        if peak_rss_bytes() > caps['max_rss_bytes']:
            raise Stop('STOP RSS cap at ' + stage)

    fh, old_handler = None, None
    try:
        fh, state['lock'] = acquire_lock(reg['lock'])
        state['free_memory_bytes_gate'] = free_memory_bytes(caps['vm_stat_page_bytes'])
        if state['free_memory_bytes_gate'] < caps['min_free_memory_bytes']:
            raise Stop('STOP memory gate: free %d < %d bytes before touching arrays' % (state['free_memory_bytes_gate'], caps['min_free_memory_bytes']))
        state['compute_started'] = time.monotonic()
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(caps['wall_seconds']))
        for name, i in input_files(reg).items():  # byte size + sha256 of every input BEFORE any array is opened
            p = Path(i['path'])
            if not p.is_file() or p.stat().st_size != i['bytes']:
                raise Stop('STOP input absent / byte size differs: ' + name)
            state['inputs_sha256'][name] = sha(p)
            if state['inputs_sha256'][name] != i['sha256']:
                raise Stop('STOP input sha256 mismatch: ' + name)
        guard('hash')
        S = load_inputs(reg)
        guard('load')
        res = compute(S, reg, guard)
        save(att / 'result.json', res)
        guard('result')
        save(att / 'terminal.json', dict(common(), success=True, rc=0, status='CALIBRATE_LOGITS_COMPUTED', result_sha256=sha(att / 'result.json'),
                                         cal_selected=res['cal_selected'],
                                         meaning='descriptive EXPOSED DEVELOPMENT calibration comparison; advisory labels only; no winner or adoption'))
    except BaseException as ex:
        signal.alarm(0)
        (att / 'result.json').unlink(missing_ok=True)
        save(att / 'terminal.json', dict(common(), success=False, rc=1, status='STOP', error=repr(ex), result_sha256=None))
        raise
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        if fh is not None:
            fh.close()  # releases the flock; the lock file itself is never created, written or deleted
    return att


def plan(reg_path, cli):
    """Registration echo + byte-level sha256 verification of every input + the command-line check. Loads NO array,
    takes no lock, never claims the attempt."""
    reg_path = Path(reg_path).resolve()
    reg = load_registration(reg_path)
    synthetic_guard(reg, reg_path)
    inputs, ok = {}, True
    for name, i in input_files(reg).items():
        p = Path(i['path'])
        e = {'path': i['path'], 'exists': p.is_file(), 'registered_sha256': i['sha256'], 'registered_bytes': i['bytes']}
        if e['exists']:
            e['bytes'] = p.stat().st_size
            e['sha256'] = sha(p)
        e['match'] = bool(e['exists'] and e['bytes'] == i['bytes'] and e['sha256'] == i['sha256'])
        ok = ok and e['match']
        inputs[name] = e
    bad = constants_match(reg)
    cbad = cli_mismatches(reg, reg_path, cli)
    rp = reg_path.parent / reg['review']['file']
    att = attempt_dir(reg, reg_path)
    try:
        free = free_memory_bytes(reg['caps']['vm_stat_page_bytes'])
    except Exception as ex:  # report-only here; --execute gates on it
        free = repr(ex)
    return {'pass': bool(ok and not bad and not cbad and reg.get('status') == READY and reg['sources'][1] == reg_path.name),
            'mode': 'plan (no array loaded, no lock taken, no attempt claimed)', 'utc': time.time(),
            'sources_sha256': {SCRIPT_KEY: sha(SELF), reg_path.name: sha(reg_path)}, 'script_path': str(SELF),
            'registration_status': reg.get('status'), 'registration_vs_source_constants': bad or 'match',
            'command_line_vs_registration': cbad or 'match', 'inputs': inputs, 'inputs_checked': len(inputs), 'all_inputs_match': ok,
            'review_file': str(rp), 'review_file_present': rp.is_file(), 'attempt_dir': str(att), 'attempt_dir_exists': att.exists(),
            'lock_path': reg['lock']['path'], 'lock_file_exists_stat_only': Path(reg['lock']['path']).is_file(),
            'free_memory_bytes_now_report_only': free, 'min_free_memory_bytes': reg['caps']['min_free_memory_bytes'],
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
            'execute_command': command_line(reg, reg_path, '--execute'), 'registration': reg}


# =====================================================================================================================
# --selfcheck: synthetic / fabricated only
# =====================================================================================================================
def synth(n_fit, n_tun, n_cal, n_eval, g_cal, g_eval, seeds, seed):
    """Fabricated corpus-shaped arrays (roles interleaved in corpus order) and, per model seed, deliberately
    miscalibrated margins of all rows = shared signal + seed noise. Fabricated numbers; never a result."""
    r = np.random.default_rng(seed)
    n = n_fit + n_tun + n_cal + n_eval
    role = np.r_[np.zeros(n_fit, np.int64), np.ones(n_tun, np.int64), np.full(n_cal, 2), np.full(n_eval, 3)]
    games = np.r_[1000 + np.arange(n_fit) % 7, 1500 + np.arange(n_tun) % 9, 2000 + np.arange(n_cal) % g_cal, 3000 + np.arange(n_eval) % g_eval]
    perm = r.permutation(n)
    role, games = role[perm], games[perm]
    prior = np.log(np.array([.36, .26, .18, .08, .03, .02, .012, .02, .012, .026]))
    zp = r.normal(0, 1.0, (n, K10)) + prior
    w_true = np.linspace(.8, 1.3, K10)
    b_true = np.array([.1, -.1, .05, 0, .1, -.05, 0, .05, -.1, 0])
    y10 = (softmax(zp * w_true + b_true, axis=1).cumsum(1) < r.random((n, 1))).sum(1).clip(0, K10 - 1).astype(np.int64)
    zw, zg = r.normal(0, 1.5, n), r.normal(0, 1.5, n)
    whiff = ((y10 == 1) & (r.random(n) < expit(1.2 * zw - .3))).astype(np.int8)
    gb = (np.isin(y10, BIP) & (r.random(n) < expit(.8 * zg + .2))).astype(np.int8)
    pitcher, batter = 500 + r.integers(0, 40, n), 900 + r.integers(0, 60, n)
    pitcher[(role == 3) & (r.random(n) < .1)] = 499  # an EVAL-only pitcher (stratum S0)
    batter[(role == 3) & (r.random(n) < .1)] = 899   # an EVAL-only batter (stratum B0)
    z = {s: {'parent': zp + r.normal(0, .25, (n, K10)), 'narrow_whiff': zw + r.normal(0, .3, n), 'ground_ball_per_pitch': zg + r.normal(0, .3, n)} for s in seeds}
    T = {s: {'parent': .97 + .02 * i, 'narrow_whiff': .95 + .03 * i, 'ground_ball_per_pitch': 1.02 - .02 * i} for i, s in enumerate(seeds)}
    return {'role': role, 'games': games, 'y10': y10, 'whiff': whiff, 'gb': gb, 'y7': derive_y7(y10, whiff, gb), 'pitcher': pitcher, 'batter': batter,
            'keys': np.column_stack([games, np.arange(n), np.zeros(n, np.int64)]).astype(np.int64), 'z': z, 'T': T, 'seeds': list(seeds), 'w_true': w_true}


def fit_rows_of(fit_ids, ids):
    """Number of FIT rows carrying each id (selfcheck fixture only)."""
    u, c = np.unique(fit_ids, return_counts=True)
    pos = np.clip(np.searchsorted(u, ids), 0, len(u) - 1)
    return np.where(u[pos] == ids, c[pos], 0).astype(np.int64)


def synth_S(Sy, baselines=('P',), reference=True, seed=99):
    """compute()-shaped dict from a synth() corpus (no files)."""
    r = np.random.default_rng(seed)
    cal, ev = Sy['role'] == 2, Sy['role'] == 3
    lab = {sp: {'y10': Sy['y10'][m], 'y7': Sy['y7'][m], 'whiff': Sy['whiff'][m].astype(np.int64), 'gb': Sy['gb'][m].astype(np.int64)}
           for sp, m in (('calibration', cal), ('evaluation', ev))}
    runs = {}
    for s in Sy['seeds']:
        z = {sp: {h: Sy['z'][s][h][m] for h in HEADS} for sp, m in (('calibration', cal), ('evaluation', ev))}
        runs[s] = {'T': dict(Sy['T'][s]), 'z': z, 'persisted': probs_c0(z['evaluation'], Sy['T'][s])}
    base = {}
    for b in baselines:
        q = softmax(np.log(runs[Sy['seeds'][0]]['persisted']) * .9 + r.normal(0, .15, (int(ev.sum()), 7)), axis=1)
        base[b] = q
    return {'arm_prefix': 'SYN', 'seeds': list(Sy['seeds']), 'runs': runs, 'lab': lab, 'games': {'calibration': Sy['games'][cal], 'evaluation': Sy['games'][ev]},
            'meta': {'pitcher': Sy['pitcher'][ev], 'fit_pitcher_rows': fit_rows_of(Sy['pitcher'][Sy['role'] == 0], Sy['pitcher'][ev]),
                     'fit_batter_rows': fit_rows_of(Sy['batter'][Sy['role'] == 0], Sy['batter'][ev])},
            'baselines': base, 'reference': np.mean([runs[s]['persisted'] for s in Sy['seeds']], axis=0) if reference else None}


def refresh_receipt(d):
    """Re-bind a fabricated run's receipt to its (possibly tampered) files."""
    r = json.loads((d / 'receipt.json').read_text())
    r['outputs_sha256'] = {f: sha(d / f) for f in RUN_FILES[:3]}
    save(d / 'receipt.json', r)


def write_fixture(fx, Sy, prefix, baselines=('P', 'N0'), reference=True):
    """Fabricated input FILES in the registered layout: <fx>/runs/<prefix>-s<seed>/{margins.npz, temperatures.json,
    seven-leaf-evaluation.npy, receipt.json}, corpus.npz, eval-meta.npz, baseline arrays, reference ensemble, lock."""
    S = synth_S(Sy, baselines, reference)
    masks = {'fit': Sy['role'] == 0, 'tuning': Sy['role'] == 1, 'calibration': Sy['role'] == 2, 'evaluation': Sy['role'] == 3}
    ev = masks['evaluation']
    np.savez(fx / 'corpus.npz', y10=Sy['y10'], y7=Sy['y7'], keys=Sy['keys'], games=Sy['games'], narrow_whiff=Sy['whiff'], ground_ball_per_pitch=Sy['gb'], **masks)
    np.savez(fx / 'eval-meta.npz', keys=Sy['keys'][ev], y7=Sy['y7'][ev], y10=Sy['y10'][ev], games=Sy['games'][ev], pitcher=Sy['pitcher'][ev], batter=Sy['batter'][ev],
             fit_pitcher_rows=S['meta']['fit_pitcher_rows'], fit_batter_rows=S['meta']['fit_batter_rows'])
    cal10 = Sy['y10'][masks['calibration']]
    for s in Sy['seeds']:
        d = fx / 'runs' / ('%s-s%d' % (prefix, s))
        d.mkdir(parents=True)
        np.savez(d / 'margins.npz', **{'%s__%s' % (role, h): Sy['z'][s][h][masks[role]] for role in ROLES for h in HEADS})
        save(d / 'temperatures.json', {'temperatures': Sy['T'][s], 'split': 'calibration',
                                      'fit_rows': {'parent': int(len(cal10)), 'narrow_whiff': int((cal10 == 1).sum()), 'ground_ball_per_pitch': int(np.isin(cal10, BIP).sum())}})
        np.save(d / 'seven-leaf-evaluation.npy', S['runs'][s]['persisted'])
        zt = {h: Sy['z'][s][h][masks['tuning']] for h in HEADS}
        save(d / 'receipt.json', {'id': 'FABRICATED', 'arm': prefix, 'seed': s, 'temperatures': Sy['T'][s], 'rows': {k: int(m.sum()) for k, m in masks.items()},
                                  'best_select_seven_leaf_nll_uncalibrated': float(row_loss(probs_c0(zt, {h: 1.0 for h in HEADS}), Sy['y7'][masks['tuning']])[0].mean()),
                                  'outputs_sha256': {}})
        refresh_receipt(d)
    for b in baselines:
        np.save(fx / ('baseline-%s.npy' % b), S['baselines'][b])
    if reference:
        np.save(fx / 'reference.npy', S['reference'])
    (fx / 'lock').write_bytes(b'')
    return S


def fab_registration(real, fx, prefix, seeds):
    """The REAL registration with only paths / hashes / shapes / expected denominators / lock / memory floor / attempt /
    author replaced (and the synthetic marker added), so the fabricated execute exercises the registered settings."""
    g = copy.deepcopy(real)
    ent = lambda p: {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
    x = g['inputs']
    x.update(runs_dir=str(fx / 'runs'), arm_prefix=prefix, seeds=list(seeds), receipt_expect={'id': 'FABRICATED', 'arm': prefix})
    x['run_files'] = {'%s-s%d' % (prefix, s): {f: {k: v for k, v in ent(fx / 'runs' / ('%s-s%d' % (prefix, s)) / f).items() if k != 'path'} for f in RUN_FILES} for s in seeds}
    with np.load(fx / 'runs' / ('%s-s%d' % (prefix, seeds[0])) / 'margins.npz') as z:
        x['margins_members'] = {m: {'shape': list(z[m].shape), 'dtype': str(z[m].dtype)} for m in x['margins_members']}
    for name, f in (('corpus', 'corpus.npz'), ('eval_meta', 'eval-meta.npz')):
        with np.load(fx / f) as z:
            x[name] = dict(ent(fx / f), members={m: {'shape': list(z[m].shape), 'dtype': str(z[m].dtype)} for m in x[name]['members']})
    x['baselines'] = {p.name[9:-4]: ent(p) for p in sorted(fx.glob('baseline-*.npy'), reverse=True)}
    x['reference_ensemble'] = ent(fx / 'reference.npy') if (fx / 'reference.npy').exists() else None
    with np.load(fx / 'corpus.npz') as z:
        for sp in ('calibration', 'evaluation'):
            m = z[sp]
            g['expected'][sp] = {'rows': int(m.sum()), 'games': int(len(np.unique(z['games'][m]))), 'leaf_counts': np.bincount(z['y7'][m], minlength=7).tolist()}
        g['expected']['tuning_rows'] = int(z['tuning'].sum())
        x['seven_leaf_evaluation'] = {'shape': [int(z['evaluation'].sum()), 7], 'dtype': 'float64'}
    g['lock'] = {'path': str(fx / 'lock'), 'retry_seconds': 0.05, 'max_wait_seconds': 0.2}
    g['caps']['min_free_memory_bytes'] = 0
    g['attempt'] = {'root': 'attempts', 'id': 'fab-001'}
    g['author'] = 'fabricated author'
    g['synthetic_selfcheck'] = True
    return g


def pav_bruteforce(x, y):
    """Independent O(n^3) reference: pool ties with a dict, then the min-max formula
    fit_i = max_{j<=i} min_{k>=i} weighted mean(y[j..k]) of the isotonic regression. Returns the fit per input row."""
    pool = {}
    for xv, yv in zip(x.tolist(), y.tolist()):
        s, w = pool.get(xv, (0.0, 0))
        pool[xv] = (s + yv, w + 1)
    keys = sorted(pool)
    fit = {}
    for i, xv in enumerate(keys):
        best = -np.inf
        for j in range(i + 1):
            low = np.inf
            for k in range(i, len(keys)):
                s = sum(pool[keys[t]][0] for t in range(j, k + 1))
                w = sum(pool[keys[t]][1] for t in range(j, k + 1))
                low = min(low, s / w)
            best = max(best, low)
        fit[xv] = best
    return np.array([fit[v] for v in x.tolist()])


def fd_error(fun, x, h=1e-6):
    """Max abs difference between the analytic gradient and central finite differences, relative to max(1,|grad|)."""
    _, g = fun(x)
    num = np.empty_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        num[i] = (fun(x + e)[0] - fun(x - e)[0]) / (2 * h)
    return float(np.abs(num - g).max() / max(1.0, np.abs(g).max()))


def selfcheck(reg_path):
    """Synthetic-only verification under the hyperparameters of the given registration. Returns the receipt payload."""
    t0 = time.monotonic()
    reg_path = Path(reg_path).resolve()
    res, detail = {}, {}
    real = load_registration(reg_path)
    opt, A = real['optimiser'], real['arms']
    # audit: no registered input / lock / real attempt path may be opened by this mode
    watch = [v['path'] for v in input_files(real).values()] + [real['lock']['path'], str(reg_path.parent / real['attempt']['root'])]
    opened = []
    sys.addaudithook(lambda ev, a: opened.append(os.fsdecode(a[0])) if ev == 'open' and a and isinstance(a[0], (str, bytes)) else None)
    res['registration_matches_source_constants'] = constants_match(real) == [] and real['status'] == READY and real['sources'][1] == reg_path.name
    th = real['fan_display']['thresholds']

    # --- 1. copied metric functions: closed forms -------------------------------------------------------------
    y = np.arange(14) % 7
    q = np.full((14, 7), 1 / 7)
    m = arm_full(q, y, th)
    res['metric_closed_form_uniform'] = abs(m['nll'] - np.log(7)) < 1e-12 and abs(m['brier'] - 6 / 7) < 1e-12 and abs(m['ece_top_label_10bin']) < 1e-12 \
        and m['leaves']['ball']['reliability'][1]['n'] == 14 and abs(m['classwise_ece_10bin']['mean_over_leaves']) < 1e-12 \
        and reliability(q, y)['ball'][1]['n'] == 14
    q = np.zeros((4, 7))
    q[:, 0] = 1.
    y = np.array([0, 0, 0, 1])
    m = arm_full(q, y, th)
    cw = m['classwise_ece_10bin']
    res['ece_p1_last_bin_and_nll_clip'] = m['leaves']['ball']['reliability'][9]['n'] == 4 and abs(m['ece_top_label_10bin'] - .25) < 1e-12 and abs(m['nll'] + np.log(CLIP) / 4) < 1e-9 \
        and m['top_label_reliability'][9]['n'] == 4 and abs(m['top_label_reliability'][9]['accuracy'] - .75) < 1e-12
    res['classwise_ece_closed_form'] = abs(cw['per_leaf']['ball'] - .25) < 1e-12 and abs(cw['per_leaf']['foul'] - .25) < 1e-12 and abs(cw['mean_over_leaves'] - .5 / 7) < 1e-12 and abs(cw['max_over_leaves'] - .25) < 1e-12
    lb = m['leaves']['ball']
    res['leaf_block_closed_form'] = abs(lb['ece_10bin'] - .25) < 1e-12 and lb['mean_predicted'] == 1.0 and lb['base_rate'] == .75 and abs(lb['mean_predicted_minus_base_rate'] - .25) < 1e-12 \
        and abs(lb['reliability'][9]['gap_observed_minus_predicted'] + .25) < 1e-12 and lb['checks']['populated_bins'] == [] and lb['checks']['monotone'] is None and lb['checks']['max_abs_bin_gap'] is None
    r = np.random.default_rng(11)
    q = r.dirichlet(np.ones(7) * .7, 500)
    y = r.integers(0, 7, 500)
    ref = 0.
    for b in range(10):
        s = [i for i in range(500) if min(int(q[i].max() * 10), 9) == b]
        if s:
            ref += abs(np.mean([q[i].max() for i in s]) - np.mean([float(q[i].argmax() == y[i]) for i in s])) * len(s) / 500
    refc = []
    for k in range(7):
        t = 0.
        for b in range(10):
            s = [i for i in range(500) if min(int(q[i, k] * 10), 9) == b]
            if s:
                t += abs(np.mean([q[i, k] for i in s]) - np.mean([float(y[i] == k) for i in s])) * len(s) / 500
        refc.append(t)
    cw = classwise_ece(q, y)
    res['ece_and_classwise_match_loop_reference'] = abs(ece(q, y) - ref) < 1e-12 and np.allclose([cw['per_leaf'][n] for n in LEAVES], refc, atol=1e-12) \
        and abs(cw['mean_over_leaves'] - np.mean(refc)) < 1e-12 and abs(cw['max_over_leaves'] - max(refc)) < 1e-12
    # leaf block vs loop reference incl. the populated-bin checks (low populated threshold so that bins qualify)
    th2 = dict(th, populated_bin_min_n=20)
    ok = True
    for k in range(7):
        blk = leaf_block(q, y, k, th2)
        pop, obs, gaps = [], [], []
        for b in range(10):
            s = [i for i in range(500) if min(int(q[i, k] * 10), 9) == b]
            if len(s) >= 20:
                pop.append(b)
                obs.append(np.mean([float(y[i] == k) for i in s]))
                gaps.append(abs(obs[-1] - np.mean([q[i, k] for i in s])))
        viol = [[pop[j], pop[j + 1]] for j in range(len(pop) - 1) if not obs[j + 1] > obs[j]]
        c = blk['checks']
        ok = ok and abs(blk['ece_10bin'] - refc[k]) < 1e-12 and c['populated_bins'] == pop and c['monotone_violations'] == viol \
            and c['monotone'] == (None if len(pop) < 2 else not viol) and abs(c['max_abs_bin_gap'] - max(gaps)) < 1e-12 \
            and c['bin_gap_ok'] == (max(gaps) <= th['max_abs_bin_gap']) and c['leaf_ece_ok'] == (refc[k] <= th['leaf_ece_max'])
    res['leaf_block_matches_loop_reference'] = bool(ok)

    # --- 2. leaves / labels / C0 and the initial points of C1 and C2 ------------------------------------------
    Sy = synth(1500, 1200, 4000, 5000, 20, 25, (0, 1, 2), 3)
    S = synth_S(Sy, ('P', 'N0'))
    T0 = Sy['T'][0]
    zc = S['runs'][0]['z']['calibration']
    zp, zw, zg = zc['parent'][:50], zc['narrow_whiff'][:50], zc['ground_ball_per_pitch'][:50]
    q0 = probs_c0({'parent': zp, 'narrow_whiff': zw, 'ground_ball_per_pitch': zg}, T0)
    ok = True
    for i in range(50):
        a = zp[i] / T0['parent']
        p = np.exp(a - a.max())
        p /= p.sum()
        qw = 1 / (1 + np.exp(-zw[i] / T0['narrow_whiff']))
        qg = 1 / (1 + np.exp(-zg[i] / T0['ground_ball_per_pitch']))
        s = p[3] + p[4] + p[5] + p[6] + p[7] + p[9]
        ok = ok and np.allclose(q0[i], [p[0], p[2], p[8], p[1] * qw, p[1] * (1 - qw), s * qg, s * (1 - qg)], atol=1e-13)
    res['c0_leaves_match_scalar_reference'] = bool(ok)
    y7ref = []
    for a, w, g in zip(Sy['y10'].tolist(), Sy['whiff'].tolist(), Sy['gb'].tolist()):
        y7ref.append({0: 0, 2: 1, 8: 2}.get(a, (3 if w else 4) if a == 1 else (5 if g else 6)))
    res['derive_y7_matches_loop_reference'] = np.array_equal(Sy['y7'], y7ref) and set(np.unique(Sy['y7'])) == set(range(7))
    p_c0 = softmax(zp / T0['parent'], axis=1)
    p_c1 = softmax(zp * np.full(K10, 1 / T0['parent']) + np.zeros(K10), axis=1)
    p_c2 = softmax(log_softmax(zp, axis=1) @ (np.eye(K10) / T0['parent']).T + np.zeros(K10), axis=1)
    res['c1_c2_initial_point_equals_c0'] = np.abs(p_c1 - p_c0).max() < 1e-14 and np.abs(p_c2 - p_c0).max() < 1e-12

    # --- 3. PAV ---------------------------------------------------------------------------------------------
    r = np.random.default_rng(5)
    worst, mono, sse_ok, blocks_ok = 0., True, True, True
    for case in range(300):
        n = int(r.integers(1, 13))
        x = np.round(r.random(n), 1) if case % 2 else r.random(n)  # every other case has heavy ties
        yb = (r.random(n) < .2 + .6 * x).astype(float) if case % 3 else r.normal(size=n)
        bl = pav_fit(x, yb)
        fit = pav_apply(bl, x)
        worst = max(worst, float(np.abs(fit - pav_bruteforce(x, yb)).max()))
        o = np.argsort(x, kind='stable')
        mono = mono and bool((np.diff(fit[o]) >= -1e-15).all()) and bool((np.diff(bl['value']) > 0).all())
        blocks_ok = blocks_ok and int(bl['weight'].sum()) == n and bool((bl['left'] <= bl['right']).all()) and bool((bl['right'][:-1] < bl['left'][1:]).all())
        for _ in range(20):  # no random monotone candidate (function of x) may beat the PAV squared error
            cand = np.sort(r.normal(yb.mean(), 1, len(np.unique(x))))[np.searchsorted(np.unique(x), x)]
            sse_ok = sse_ok and ((fit - yb) ** 2).sum() <= ((cand - yb) ** 2).sum() + 1e-12
    detail['pav_max_abs_diff_vs_bruteforce'] = worst
    res['pav_matches_bruteforce_minmax_300_cases'] = worst < 1e-12
    res['pav_monotone_and_block_structure'] = mono and blocks_ok
    res['pav_sse_not_beaten_by_random_monotone'] = bool(sse_ok)
    bl = pav_fit(np.array([.1, .1, .2, .3, .6, .7]), np.array([0., 1, 0, 0, 1, 1.]))  # pooled: [.1,.3]->1/4, [.6,.7]->1
    res['pav_hand_case_and_lookup_rule'] = np.allclose(bl['value'], [.25, 1.]) and np.allclose(bl['left'], [.1, .6]) and np.allclose(bl['right'], [.3, .7]) \
        and np.allclose(pav_apply(bl, np.array([0., .1, .3, .30001, .45, .6, .7, .99])), [.25, .25, .25, 1, 1, 1, 1, 1])
    qc = probs_c0(zc, T0)
    yc = S['lab']['calibration']['y7']
    blocks = isotonic_fit(qc, yc)
    q3 = isotonic_apply(blocks, qc, A['C3']['clip'])
    raw = np.column_stack([pav_apply(blocks[k], qc[:, k]) for k in range(7)])
    res['isotonic_insample_mean_preserved_and_rows_normalised'] = np.allclose(raw.mean(0), np.bincount(yc, minlength=7) / len(raw), atol=1e-12) \
        and np.abs(q3.sum(1) - 1).max() < 1e-12 and q3.min() > 0 and all((np.diff(pav_apply(blocks[k], np.linspace(0, 1, 2001))) >= 0).all() for k in range(7))

    # --- 4. analytic gradients vs central finite differences ------------------------------------------------
    r = np.random.default_rng(7)
    zs, ys = r.normal(size=(300, K10)), r.integers(0, K10, 300)
    us = log_softmax(zs, axis=1)
    zb, yb = r.normal(size=300), r.integers(0, 2, 300)
    detail['gradient_fd_relative_error'] = {
        'vector': fd_error(vector_objective(zs, ys), r.normal(size=2 * K10)),
        'platt': fd_error(platt_objective(zb, yb), r.normal(size=2)),
        'dirichlet': fd_error(dirichlet_objective(us, ys, .03, .07), r.normal(size=K10 * K10 + K10) * .5)}
    res['analytic_gradients_match_finite_differences'] = all(v < 1e-6 for v in detail['gradient_fd_relative_error'].values())

    # --- 5. recovery of known transforms -------------------------------------------------------------------
    r = np.random.default_rng(13)
    n = 60000
    z = r.normal(0, 1.2, (n, K10))
    w_true = np.linspace(.7, 1.4, K10)
    b_true = np.linspace(-.3, .3, K10)
    yv = (softmax(z * w_true + b_true, axis=1).cumsum(1) < r.random((n, 1))).sum(1).clip(0, K10 - 1)
    par, info = fit_vector(z, yv, 1.0, opt, 'selfcheck vector')
    f_true = vector_objective(z, yv)(np.r_[w_true, b_true])[0]
    detail['vector_recovery'] = {'max_abs_w_error': float(np.abs(par['w'] - w_true).max()),
                                 'max_abs_centered_b_error': float(np.abs(par['b'] - par['b'].mean() - (b_true - b_true.mean())).max()),
                                 'nll_fit_minus_nll_true': info['objective'] - f_true, 'optimiser': info}
    res['vector_scaling_recovers_known_transform'] = detail['vector_recovery']['max_abs_w_error'] < .08 and detail['vector_recovery']['max_abs_centered_b_error'] < .10 \
        and info['objective'] <= f_true + 1e-12 and info['gradient_max_abs'] < 1e-4
    zh = r.normal(0, 1.5, n)
    yh = (r.random(n) < expit(1.3 * zh - .4)).astype(np.int64)
    ph, info = fit_platt(zh, yh, 1.0, opt, 'selfcheck platt')
    detail['platt_recovery'] = {'a': ph['a'], 'b': ph['b'], 'true': [1.3, -.4], 'optimiser': info}
    res['platt_recovers_known_transform'] = abs(ph['a'] - 1.3) < .04 and abs(ph['b'] + .4) < .04 and info['gradient_max_abs'] < 1e-4
    W_true = np.eye(K10) * 1.2 + r.normal(0, .08, (K10, K10))
    bd_true = r.normal(0, .15, K10)
    u = log_softmax(z, axis=1)
    yd = (softmax(u @ W_true.T + bd_true, axis=1).cumsum(1) < r.random((n, 1))).sum(1).clip(0, K10 - 1)
    lam = 1e-6
    par, info = fit_dirichlet(u, yd, lam, lam, 1.0, opt, 'selfcheck dirichlet')
    ut = log_softmax(r.normal(0, 1.2, (5000, K10)), axis=1)
    perr = float(np.abs(softmax(ut @ par['W'].T + par['b'], axis=1) - softmax(ut @ W_true.T + bd_true, axis=1)).max())
    f_fit = dirichlet_objective(u, yd, 0., 0.)(np.r_[par['W'].ravel(), par['b']])[0]
    f_tru = dirichlet_objective(u, yd, 0., 0.)(np.r_[W_true.ravel(), bd_true])[0]
    detail['dirichlet_recovery'] = {'max_abs_probability_error_on_fresh_points': perr, 'nll_fit_minus_nll_true': f_fit - f_tru, 'optimiser': info}
    res['dirichlet_recovers_known_transform'] = perr < .10 and f_fit <= f_tru + 1e-9 and info['gradient_max_abs'] < 1e-4

    # --- 6. CV folds, lambda CV, selection rule, leak-free selection CV ---------------------------------------
    sel = real['selection']
    g = S['games']['calibration']
    f1, ch1 = cv_folds(g, sel['cv_seed'], sel['cv_folds'])
    f2, _ = cv_folds(g[::-1].copy(), sel['cv_seed'], sel['cv_folds'])
    res['cv_folds_grouped_by_game_and_deterministic'] = all(len(np.unique(f1[g == x])) == 1 for x in np.unique(g)) and np.array_equal(f1[::-1], f2) \
        and [len(c) for c in ch1] == [4, 4, 4, 4, 4] and np.array_equal(np.sort(np.concatenate(ch1)), np.unique(g)) \
        and np.array_equal(np.concatenate(ch1), np.random.default_rng(sel['cv_seed']).permutation(np.unique(g)))
    labc = S['lab']['calibration']
    uc = log_softmax(zc['parent'], axis=1)
    noguard = lambda stage: None
    lam_full, lcv = select_lambda(uc, labc['y10'], f1, T0['parent'], A['C2'], opt, noguard, 'selfcheck')
    row = lcv['table'][2]
    par, _ = fit_dirichlet(uc[f1 != 3], labc['y10'][f1 != 3], row['lambda'], row['lambda'] * A['C2']['mu_over_lambda'], T0['parent'], opt, 'selfcheck direct')
    direct = _softmax_nll(uc[f1 == 3] @ par['W'].T + par['b'], labc['y10'][f1 == 3], np.arange(int((f1 == 3).sum())))[0]
    want = min(lcv['table'], key=lambda t: (t['mean_heldout_nll'], -t['lambda']))['lambda']
    res['lambda_cv_table_matches_direct_fit_and_rule'] = row['heldout_nll_by_fold'][3] == direct and lam_full == want and [t['lambda'] for t in lcv['table']] == A['C2']['lambda_grid'] \
        and all(t['heldout_fold_ids'] == [0, 1, 2, 3, 4] and abs(t['mean_heldout_nll'] - np.mean(t['heldout_nll_by_fold'])) < 1e-15 for t in lcv['table'])
    mk = lambda **kw: {a: {'nll': kw[a][0], 'ece': kw[a][1]} for a in ARMS}
    mgn = sel['nll_margin']
    res['selection_rule_cases'] = \
        select_arm(mk(C0=(1.5, .010), C1=(1.5, .008), C2=(1.5, .006), C3=(1.5, .007)), mgn)[0] == 'C2' \
        and select_arm(mk(C0=(1.5, .010), C1=(1.5, .008), C2=(1.5 + mgn * 1.01, .001), C3=(1.5, .009)), mgn)[:2] == ('C1', ['C0', 'C1', 'C3']) \
        and select_arm(mk(C0=(1.5, .010), C1=(1.5 + mgn, .010), C2=(1.5, .010), C3=(1.5, .010)), mgn)[0] == 'C0' \
        and select_arm(mk(C0=(1.5, .010), C1=(1.6, .001), C2=(1.6, .001), C3=(1.6, .001)), mgn)[:2] == ('C0', ['C0']) \
        and select_arm(mk(C0=(1.5, .010), C1=(1.5, .005), C2=(1.4, .005), C3=(1.5, .006)), mgn)[0] == 'C1' \
        and select_arm(mk(C0=(1.5, .010), C1=(1.5 + mgn, .009), C2=(1.7, .0), C3=(1.7, .0)), mgn)[0] == 'C1'
    cv1, oof1 = cv_select(zc, labc, f1, T0, real, noguard, 'selfcheck')
    lab_t = {k: v.copy() for k, v in labc.items()}  # scramble ALL labels of fold 2: its held-out predictions must not move
    ho = f1 == 2
    pr = np.random.default_rng(1).permutation(int(ho.sum()))
    for k in lab_t:
        lab_t[k][ho] = lab_t[k][ho][pr]
    cv2, oof2 = cv_select(zc, lab_t, f1, T0, real, noguard, 'selfcheck tampered')
    res['selection_cv_heldout_rows_never_used_for_fitting'] = all(np.array_equal(oof1[a][ho], oof2[a][ho]) for a in ARMS) \
        and all(not np.array_equal(oof1[a][~ho], oof2[a][~ho]) for a in ARMS[1:]) and np.array_equal(oof1['C0'], oof2['C0'])
    means = {a: {'nll': cv1['per_arm'][a]['cv_mean_nll'], 'ece': cv1['per_arm'][a]['cv_mean_ece_top_label_10bin']} for a in ARMS}
    res['selection_cv_table_consistent'] = all(np.isfinite(oof1[a]).all() for a in ARMS) and cv1['selected'] == select_arm(means, mgn)[0] \
        and all(abs(cv1['per_arm'][a]['cv_mean_nll'] - np.mean(cv1['per_arm'][a]['nll_by_fold'])) < 1e-15
                and abs(cv1['per_arm'][a]['nll_by_fold'][1] - row_loss(oof1[a][f1 == 1], labc['y7'][f1 == 1])[0].mean()) < 1e-15
                and abs(cv1['per_arm'][a]['ece_top_label_10bin_by_fold'][4] - ece(oof1[a][f1 == 4], labc['y7'][f1 == 4])) < 1e-15 for a in ARMS) \
        and np.array_equal(oof1['C0'], qc) and len(cv1['c2_lambda_by_outer_fold']) == 5
    # the fit inside outer fold 0 is the registered procedure on the other folds (inner lambda CV = 4 folds)
    tr = f1 != 0
    mdl = fit_arms(take(zc, tr), take(labc, tr), f1[tr], T0, real, noguard, 'selfcheck outer0')
    res['outer_fold_fit_uses_inner_lambda_cv_on_training_folds_only'] = mdl['lam'] == cv1['c2_lambda_by_outer_fold'][0] \
        and mdl['info']['lambda_cv']['table'][0]['heldout_fold_ids'] == [1, 2, 3, 4] and mdl['info']['rows'] == int(tr.sum()) \
        and np.array_equal(apply_arms(mdl, take(zc, ~tr))['C2'], oof1['C2'][~tr])

    # --- 7. build_predictions: ensemble, SEL, reference gate, determinism ------------------------------------
    B1 = build_predictions(S, real)
    B2 = build_predictions(S, real)
    ev = 'evaluation'
    res['build_is_deterministic'] = all(np.array_equal(B1['ensemble'][a], B2['ensemble'][a]) for a in B1['ensemble']) \
        and all(B1['seeds'][s]['selected'] == B2['seeds'][s]['selected'] for s in S['seeds'])
    res['ensemble_is_mean_of_seed_rows_and_sel_follows_selection'] = \
        all(np.array_equal(B1['ensemble'][a], np.mean([B1['seeds'][s]['Q'][a][ev] for s in S['seeds']], axis=0)) for a in ARMS) \
        and np.array_equal(B1['ensemble'][SEL], np.mean([B1['seeds'][s]['Q'][B1['seeds'][s]['selected']][ev] for s in S['seeds']], axis=0)) \
        and B1['reference_gate']['checked'] and B1['reference_gate']['c0_ensemble_vs_reference_max_abs'] == 0.0 \
        and all(B1['seeds'][s]['gate']['c0_vs_persisted_evaluation_max_abs'] == 0.0 for s in S['seeds'])
    res['seed0_full_cal_fit_equals_direct_fit'] = np.array_equal(B1['seeds'][0]['Q']['C1']['calibration'],
                                                                 apply_arms(fit_arms(zc, labc, f1, T0, real, noguard, 'direct'), zc)['C1']) \
        and B1['seeds'][0]['parameters']['C2']['parent']['lambda'] == lam_full and B1['seeds'][0]['cv'] == cv1
    S_bad = dict(S, reference=S['reference'] + np.r_[2e-12, -2e-12, np.zeros(5)])
    try:
        build_predictions(S_bad, real)
        res['reference_ensemble_gate_stops_at_2e-12'] = False
    except Stop as ex:
        res['reference_ensemble_gate_stops_at_2e-12'] = 'STOP the C0 ensemble does not reproduce' in str(ex)
    detail['synthetic_selected_arms'] = {str(s): B1['seeds'][s]['selected'] for s in S['seeds']}

    # --- 8. bootstrap ----------------------------------------------------------------------------------------
    r = np.random.default_rng(17)
    ne = 900
    gm = np.sort(r.integers(0, 14, ne))
    yv = r.integers(0, 7, ne)
    arms = {a: r.dirichlet(np.ones(7) * (1 + i), ne) for i, a in enumerate(ARMS)}
    ci, ab, _ = bootstrap_multi(arms, SEED_PAIRS, yv, gm, 200, 5, 64, leaf_idx=(3, 5), abs_arms=('C0', 'C3'))
    ci_plain, _, _ = bootstrap_multi(arms, SEED_PAIRS, yv, gm, 200, 5, 64)
    ok_ref, ok_replay = True, True
    for a, b in SEED_PAIRS:
        refb = bootstrap_eop_reference(arms[b], arms[a], yv, gm, 200, 5, 64)
        ok_ref = ok_ref and all(ci['%s-%s' % (a, b)][m] == refb[m]['ci95_percentile'] == ci_plain['%s-%s' % (a, b)][m] for m in ('nll', 'brier', 'ece'))
    u = np.unique(gm)
    rng = np.random.default_rng(5)
    rep = {p: {'nll': [], 'brier': [], 'ece': [], 3: [], 5: []} for p in ci}
    rab = {a: {'nll': [], 'brier': [], 'ece': [], ('e', 3): [], ('e', 5): [], ('g', 3): [], ('g', 5): []} for a in ab}
    lece = lambda qq, yy, k: classwise_ece(qq, yy)['per_leaf'][LEAVES[k]]
    for s in range(0, 200, 64):
        for pick in rng.integers(0, len(u), (min(64, 200 - s), len(u))):
            rows = np.concatenate([np.flatnonzero(gm == u[p]) for p in pick])
            for a, b in SEED_PAIRS:
                la, ba = row_loss(arms[a][rows], yv[rows])
                lb_, bb = row_loss(arms[b][rows], yv[rows])
                o = rep['%s-%s' % (a, b)]
                o['nll'].append(la.mean() - lb_.mean())
                o['brier'].append(ba.mean() - bb.mean())
                o['ece'].append(ece(arms[a][rows], yv[rows]) - ece(arms[b][rows], yv[rows]))
                for k in (3, 5):
                    o[k].append(lece(arms[a][rows], yv[rows], k) - lece(arms[b][rows], yv[rows], k))
            for a in ab:
                la, ba = row_loss(arms[a][rows], yv[rows])
                rab[a]['nll'].append(la.mean())
                rab[a]['brier'].append(ba.mean())
                rab[a]['ece'].append(ece(arms[a][rows], yv[rows]))
                for k in (3, 5):
                    rab[a][('e', k)].append(lece(arms[a][rows], yv[rows], k))
                    rab[a][('g', k)].append(arms[a][rows, k].mean() - (yv[rows] == k).mean())
    qq = lambda v: np.quantile(v, [.025, .975])
    for p in ci:
        for m in ('nll', 'brier', 'ece'):
            ok_replay = ok_replay and np.allclose(ci[p][m], qq(rep[p][m]), atol=1e-12)
        for k in (3, 5):
            ok_replay = ok_replay and np.allclose(ci[p]['leaf_ece'][LEAVES[k]], qq(rep[p][k]), atol=1e-12)
    for a in ab:
        for m in ('nll', 'brier', 'ece'):
            ok_replay = ok_replay and np.allclose(ab[a][m], qq(rab[a][m]), atol=1e-12)
        for k in (3, 5):
            ok_replay = ok_replay and np.allclose(ab[a]['leaf_ece'][LEAVES[k]], qq(rab[a][('e', k)]), atol=1e-12) \
                and np.allclose(ab[a]['leaf_gap'][LEAVES[k]], qq(rab[a][('g', k)]), atol=1e-12)
    res['bootstrap_multi_equals_verbatim_eop_bootstrap'] = bool(ok_ref)
    res['bootstrap_incl_leaf_ece_and_absolute_ci_equals_row_level_replay_200_draws'] = bool(ok_replay) and sorted(ci['C1-C0']['leaf_ece']) == sorted([LEAVES[3], LEAVES[5]])
    same = dict(arms, X=arms['C0'].copy())
    cz, _, _ = bootstrap_multi(same, [('X', 'C0')], yv, gm, 200, 5, 64, leaf_idx=(3,))
    res['identical_arms_give_zero_difference_ci'] = cz['X-C0'] == {'nll': [0.0, 0.0], 'brier': [0.0, 0.0], 'ece': [0.0, 0.0], 'leaf_ece': {LEAVES[3]: [0.0, 0.0]}}
    mg = real['margins']
    lab = lambda e, n, b: advisory_labels({'ece': [-1, e], 'nll': [-1, n], 'brier': [-1, b]}, mg)
    res['advisory_label_rules'] = lab(-1e-9, mg['nll_non_inferior_upper_le'], mg['brier_non_inferior_upper_le'])['overall'] == 'CANDIDATE' \
        and lab(0.0, 0, 0) == {'ece': 'ECE_NOT_IMPROVED', 'nll': 'NLL_NON_INFERIOR', 'brier': 'BRIER_NON_INFERIOR', 'overall': 'NOT_CANDIDATE'} \
        and lab(-1, mg['nll_non_inferior_upper_le'] + 1e-9, 0)['nll'] == 'NLL_NOT_NON_INFERIOR' and lab(-1, 0, mg['brier_non_inferior_upper_le'] + 1e-9)['overall'] == 'NOT_CANDIDATE'
    bins = real['strata']['pitcher_fit_rows']['bins']
    res['stratum_bins'] = assign_bins(np.array([0, 1, 199, 200, 999, 1000, 10 ** 6]), bins).tolist() == ['S0', 'S1', 'S1', 'S2', 'S2', 'S3', 'S3'] \
        and assign_bins(np.array([0, 1, 5000]), real['strata']['batter_fit_rows']['bins']).tolist() == ['B0', 'B1', 'B1']

    # --- 9. memory parser --------------------------------------------------------------------------------------
    txt = 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      10.\nPages active:   99.\nPages inactive:   20.\nPages speculative:   3.\n'
    ok = free_memory_bytes(16384, txt) == 33 * 16384
    for bad in (txt.replace('16384', '4096'), txt.replace('Pages inactive', 'Pages xinactive')):
        try:
            free_memory_bytes(16384, bad)
            ok = False
        except Stop:
            pass
    res['vm_stat_parser_and_page_size_gate'] = ok and isinstance(free_memory_bytes(real['caps']['vm_stat_page_bytes']), int)

    # --- 10. fabricated end-to-end execute + gate negatives (temp dir only) ----------------------------------
    with tempfile.TemporaryDirectory(prefix='calibrate-logits-') as tmp0:
        tmp = Path(tmp0) / SYNTH_DIR
        tmp.mkdir()
        fx = tmp / 'fx'
        fx.mkdir()
        write_fixture(fx, Sy, 'N1')
        run0 = lambda b: b / 'runs' / 'N1-s0'

        def case(name, want, reg=lambda g: None, review=lambda v: None, drop_review=False, files=None, pre=lambda d: None, cli=lambda a: None,
                 base=None, prefix='N1', seeds=(0, 1, 2), root=None, check=lambda o, t, g: True):
            """One fabricated run in its own dir. want=None: must succeed. 'BLOCKED..': SystemExit and NO attempt dir.
            'STOP..': Stop, attempt claimed, terminal success=false, no result.json."""
            d = (root or tmp) / name
            d.mkdir(parents=True)
            base = base or fx
            if files is not None:  # tampered private copy of the fixture
                shutil.copytree(base, d / 'fx')
                base = d / 'fx'
                files(base)
            g = fab_registration(real, base, prefix, seeds)
            reg(g)
            save(d / reg_path.name, g)
            v = {'verdict': 'PASS', 'reviewer': 'fabricated reviewer', 'sources_sha256': {SCRIPT_KEY: sha(SELF), reg_path.name: sha(d / reg_path.name)}}
            review(v)
            if not drop_review:
                save(d / g['review']['file'], v)
            a = args_from_registration(g, d / reg_path.name)
            cli(a)
            hold = pre(d)
            att = d / 'attempts' / 'fab-001'
            try:
                execute(d / reg_path.name, a)
            except SystemExit as ex:
                res[name] = want is not None and want.startswith('BLOCKED') and want in str(ex) and (name == 'second_claim_refused' or not att.exists())
                return d
            except (Stop, ValueError) as ex:
                t = json.loads((att / 'terminal.json').read_text()) if (att / 'terminal.json').exists() else {}
                res[name] = want is not None and want.startswith('STOP') and want in str(ex) and t.get('success') is False and t.get('rc') == 1 \
                    and want in t.get('error', '') and not (att / 'result.json').exists()
                return d
            finally:
                if hold is not None:
                    hold.close()
            t = json.loads((att / 'terminal.json').read_text())
            o = json.loads((att / 'result.json').read_text())
            names = list(ARMS) + [SEL]
            res[name] = want is None and t['success'] is True and t['rc'] == 0 and t['result_sha256'] == sha(att / 'result.json') \
                and sorted(f.name for f in att.iterdir()) == ['CLAIM', 'result.json', 'terminal.json'] \
                and t['registration_sha256'] == sha(d / reg_path.name) and t['calibrate_logits_sha256'] == sha(SELF) and set(t['inputs_sha256']) == set(input_files(g)) \
                and o['synthetic'] is True and o['seeds_evaluated'] == list(seeds) and sorted(o['seeds']) == sorted(str(s) for s in seeds) \
                and sorted(o['ensemble']['arms']) == sorted(names) and o['cal_selected'] == t['cal_selected'] and all(v in ARMS for v in o['cal_selected'].values()) \
                and all(o['gates']['per_seed_c0_reproduction'][str(s)]['c0_vs_persisted_evaluation_max_abs'] == 0.0 for s in seeds) \
                and sorted(o['ensemble']['advisory_labels_vs_C0']) == sorted(names[1:]) and all(sorted(o['seeds'][str(s)]['advisory_labels_vs_C0']) == ['C1', 'C2', 'C3'] for s in seeds) \
                and sorted(o['ensemble']['fan_display_leaves']) == sorted(real['fan_display']['leaves']) \
                and all(len(o['ensemble']['arms'][a]['leaves'][l]['ece_ci95']) == 2 for a in names for l in LEAVES) \
                and sorted(o['strata']['pitcher_fit_rows']) == ['S0', 'S1', 'S2', 'S3'] and sorted(o['strata']['batter_fit_rows']) == ['B0', 'B1'] \
                and o['bootstrap']['draws'] == real['bootstrap']['draws'] and o['bootstrap']['seed'] == real['bootstrap']['seed'] \
                and bool(check(o, t, g))
            return d

        def main_check(o, t, g):
            e = o['ensemble']
            ok = o['denominators']['evaluation']['rows'] == 5000 and o['denominators']['calibration']['games'] == 20 \
                and o['gates']['reference_ensemble']['checked'] and o['gates']['reference_ensemble']['c0_ensemble_vs_reference_max_abs'] == 0.0 \
                and sorted(e['baselines_report_only']) == ['N0', 'P'] and all(k in e['paired_evaluation'] for k in ('C1-C0', 'C2-C0', 'C3-C0', 'SEL-C0', 'C2-C1', 'SEL-P', 'C0-P', 'SEL-N0', 'C0-N0')) \
                and o['strata']['pitcher_fit_rows']['S0']['rows'] > 0 and o['strata']['batter_fit_rows']['B0']['rows'] > 0 \
                and sum(v['rows'] for v in o['strata']['pitcher_fit_rows'].values()) == 5000 \
                and all(o['seeds'][s]['parameters']['C2']['parent']['lambda'] in A['C2']['lambda_grid'] for s in o['seeds']) \
                and all(o['seeds'][s]['arms']['C1']['calibration_in_sample']['nll'] <= o['seeds'][s]['arms']['C0']['calibration_in_sample']['nll'] + 1e-12 for s in o['seeds'])
            # every reported point difference is the difference of the reported arm metrics, and CIs bracket sensibly
            for k, v in e['paired_evaluation'].items():
                a, b = k.split('-')
                ma = e['arms'].get(a) or e['baselines_report_only'][a]
                mb = e['arms'].get(b) or e['baselines_report_only'][b]
                ok = ok and v['nll']['difference'] == ma['nll'] - mb['nll'] and v['nll']['ci95_percentile'][0] <= v['nll']['ci95_percentile'][1] \
                    and v['leaf_ece_10bin']['narrow_whiff']['difference'] == ma['leaves']['narrow_whiff']['ece_10bin'] - mb['leaves']['narrow_whiff']['ece_10bin']
            if len(set(o['cal_selected'].values())) == 1:  # SEL must then coincide with that arm's ensemble
                a = next(iter(o['cal_selected'].values()))
                ok = ok and e['arms'][SEL]['nll'] == e['arms'][a]['nll'] and e['arms'][SEL]['ece_top_label_10bin'] == e['arms'][a]['ece_top_label_10bin']
            detail['fabricated_execute'] = {'elapsed_seconds': t['elapsed_seconds'], 'cal_selected': o['cal_selected'], 'result_json_bytes': len(json.dumps(o)),
                                            'note': 'fabricated numbers; never a result'}
            return ok

        d = case('fabricated_execute_all_gates_ok', None, check=main_check)
        try:
            execute(d / reg_path.name, args_from_registration(load_registration(d / reg_path.name), d / reg_path.name))
            res['second_claim_refused'] = False
        except SystemExit as ex:
            res['second_claim_refused'] = 'attempt fab-001 exists' in str(ex)
        pl = plan(d / reg_path.name, args_from_registration(load_registration(d / reg_path.name), d / reg_path.name))
        res['plan_on_fabricated_registration'] = pl['pass'] and pl['all_inputs_match'] and pl['inputs_checked'] == 3 * 4 + 2 + 2 + 1 and pl['attempt_dir_exists'] \
            and '--execute' in pl['execute_command'] and '--reference-ensemble' in pl['execute_command']
        # the same script on another layout: other prefix, two non-consecutive seeds, one baseline, no reference ensemble
        fxg = tmp / 'fx-generic'
        fxg.mkdir()
        Sg = synth(1500, 1200, 4000, 5000, 20, 25, (3, 7), 4)
        write_fixture(fxg, Sg, 'G05', baselines=('P',), reference=False)
        case('generic_layout_other_prefix_two_seeds_no_reference', None, base=fxg, prefix='G05', seeds=(3, 7),
             check=lambda o, t, g: o['arm_prefix'] == 'G05' and o['gates']['reference_ensemble']['checked'] is False and sorted(o['ensemble']['baselines_report_only']) == ['P']
             and 'reference_ensemble' not in t['inputs_sha256'] and 'G05-s7/margins.npz' in t['inputs_sha256'])
        case('status_not_ready_refused', 'BLOCKED: registration status', reg=lambda g: g.update(status='DRAFT'))
        case('review_missing_refused', 'BLOCKED: independent REVIEW.json required', drop_review=True)
        case('review_rework_refused', 'BLOCKED: review verdict', review=lambda v: v.update(verdict='REWORK'))
        case('review_by_author_refused', 'BLOCKED: reviewer missing or reviewer == author', review=lambda v: v.update(reviewer=' Fabricated Author '))
        case('review_wrong_script_hash_refused', 'BLOCKED: review sources_sha256', review=lambda v: v['sources_sha256'].update({SCRIPT_KEY: '0' * 64}))
        case('review_wrong_registration_hash_refused', 'BLOCKED: review sources_sha256', review=lambda v: v['sources_sha256'].update({reg_path.name: '0' * 64}))
        case('registration_constant_drift_refused', 'BLOCKED: registration does not match', reg=lambda g: g['metrics'].update(bins=15))
        case('selection_folds_differ_from_c2_folds_refused', 'BLOCKED: registration does not match', reg=lambda g: g['selection'].update(cv_seed=1))
        case('memory_floor_refused_before_claim', 'BLOCKED: free memory', reg=lambda g: g['caps'].update(min_free_memory_bytes=1 << 60))
        case('cli_runs_dir_differs_refused', 'BLOCKED: command line differs from the registration: --runs', cli=lambda a: a.update(runs=str(fxg / 'runs')))
        case('cli_seeds_differ_refused', 'BLOCKED: command line differs from the registration: --seeds', cli=lambda a: a.update(seeds=[0, 1]))
        case('cli_baseline_differs_refused', 'BLOCKED: command line differs from the registration: --baselines', cli=lambda a: a['baselines'].update(P=str(fx / 'baseline-N0.npy')))
        case('cli_out_differs_refused', 'BLOCKED: command line differs from the registration: --out', cli=lambda a: a.update(out=str(tmp / 'elsewhere')))
        with tempfile.TemporaryDirectory(prefix='calibrate-logits-outside-') as other:
            case('synthetic_registration_outside_selfcheck_dir_refused', 'BLOCKED: a synthetic registration may only', root=Path(other))
        case('input_hash_mismatch_stops', 'STOP input sha256 mismatch', reg=lambda g: g['inputs']['corpus'].update(sha256='0' * 64))

        def t_persist(b, refresh=True):
            a = np.load(run0(b) / 'seven-leaf-evaluation.npy')
            a[3, 0] += 2e-12
            a[3, 1] -= 2e-12
            np.save(run0(b) / 'seven-leaf-evaluation.npy', a)
            if refresh:
                refresh_receipt(run0(b))
        case('c0_not_reproduced_at_2e-12_stops', 'STOP C0 does not reproduce the persisted', files=t_persist)
        case('receipt_output_hash_binding_stops', 'STOP run N1-s0: receipt binding', files=lambda b: t_persist(b, False))

        def t_ref(b):
            a = np.load(b / 'reference.npy')
            a[0, 0] += 5e-12
            a[0, 1] -= 5e-12
            np.save(b / 'reference.npy', a)
        case('reference_ensemble_mismatch_stops', 'STOP the C0 ensemble does not reproduce the registered reference', files=t_ref)

        def t_tuning(b):
            z = dict(np.load(run0(b) / 'margins.npz'))
            z['tuning__parent'][[0, 1]] = z['tuning__parent'][[1, 0]]
            np.savez(run0(b) / 'margins.npz', **z)
            refresh_receipt(run0(b))
        case('tuning_row_order_breaks_select_nll_gate_stops', 'STOP run N1-s0: uncalibrated SELECT seven-leaf NLL', files=t_tuning)

        def t_fitrows(b):
            j = json.loads((run0(b) / 'temperatures.json').read_text())
            j['fit_rows']['narrow_whiff'] += 1
            save(run0(b) / 'temperatures.json', j)
            refresh_receipt(run0(b))
        case('temperature_fit_rows_mismatch_stops', 'STOP run N1-s0: temperatures.json split / fit_rows', files=t_fitrows)

        def t_temp(b):
            j = json.loads((run0(b) / 'temperatures.json').read_text())
            j['temperatures']['parent'] = 0.5
            save(run0(b) / 'temperatures.json', j)
            refresh_receipt(run0(b))
        case('temperature_differs_from_receipt_stops', 'STOP run N1-s0: receipt binding', files=t_temp)
        case('receipt_expected_field_stops', 'STOP run N1-s0: receipt binding', reg=lambda g: g['inputs']['receipt_expect'].update(arm='N9'))

        def t_label(b):
            z = dict(np.load(b / 'corpus.npz'))
            i = np.flatnonzero(z['calibration'] & (z['y7'] == 0))[0]
            z['y7'][i] = 1
            np.savez(b / 'corpus.npz', **z)
        case('label_incoherence_stops', 'STOP label coherence', files=t_label)

        def t_meta(b):
            z = dict(np.load(b / 'eval-meta.npz'))
            z['games'][0] += 1
            np.savez(b / 'eval-meta.npz', **z)
        case('eval_meta_not_eval_slice_stops', 'STOP eval-meta is not the EVAL slice', files=t_meta)

        def t_base(b):
            a = np.load(b / 'baseline-P.npy')
            a[0] *= 1.001
            np.save(b / 'baseline-P.npy', a)
        case('baseline_incoherent_stops', 'STOP seven-leaf coherence', files=t_base)
        case('expected_leaf_counts_stop', 'STOP denominators evaluation', reg=lambda g: g['expected']['evaluation'].update(leaf_counts=[1, 2, 3, 4, 5, 6, 7]))
        case('member_shape_stops', 'STOP member shape/dtype', reg=lambda g: g['inputs']['margins_members']['evaluation__narrow_whiff'].update(shape=[4999]))
        case('optimiser_budget_exhausted_stops', 'STOP optimiser not converged', reg=lambda g: g['optimiser'].update(maxiter=2))
        case('rss_cap_stops', 'STOP RSS cap', reg=lambda g: g['caps'].update(max_rss_bytes=1 << 20))

        def hold_lock(d):
            fh = (d / 'fx' / 'lock').open('rb')
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        d = case('lock_busy_retries_then_stops', 'STOP common heavy lock busy', files=lambda b: None, pre=hold_lock)
        res['lock_busy_left_lock_file_untouched'] = (d / 'fx' / 'lock').is_file() and (d / 'fx' / 'lock').stat().st_size == 0
        d = case('lock_absent_stops_and_is_not_created', 'STOP common heavy lock file absent', files=lambda b: (b / 'lock').unlink())
        res['lock_absent_stops_and_is_not_created'] = res['lock_absent_stops_and_is_not_created'] and not (d / 'fx' / 'lock').exists()
        signal.alarm(0)

    # --- 11. registered-scale synthetic benchmark of the whole compute() (fabricated values) -------------------
    exp, x = real['expected'], real['inputs']
    tb = time.monotonic()
    Sb = synth_S(synth(3000, 3000, exp['calibration']['rows'], exp['evaluation']['rows'], exp['calibration']['games'], exp['evaluation']['games'], tuple(x['seeds']), 23),
                 tuple(x['baselines']), bool(x.get('reference_ensemble')))
    ob = compute(Sb, real)
    its, grads, msgs = [], [], set()

    def walk(o):
        if isinstance(o, dict):
            if 'iterations' in o and 'gradient_max_abs' in o:
                its.append(o['iterations']); grads.append(o['gradient_max_abs']); msgs.add(o['message'])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk({s: ob['seeds'][s]['parameters'] for s in ob['seeds']})
    detail['benchmark_fabricated_registered_scale'] = {
        'seeds': len(x['seeds']), 'cal_rows': exp['calibration']['rows'], 'eval_rows': exp['evaluation']['rows'], 'draws': ob['bootstrap']['draws'],
        'seconds': time.monotonic() - tb, 'peak_rss_bytes': peak_rss_bytes(), 'max_optimiser_iterations_full_cal_fits': int(max(its)), 'registered_maxiter': opt['maxiter'],
        'optimiser_stop_messages': sorted(msgs), 'max_final_gradient_abs_full_cal_fits': float(max(grads)), 'cal_selected': ob['cal_selected'],
        'selected_lambda_per_seed': {s: ob['seeds'][s]['parameters']['C2']['parent']['lambda'] for s in ob['seeds']},
        'result_json_bytes': len(json.dumps(ob, allow_nan=False)),
        'note': 'fabricated values at the registered seed/row/game/draw counts; optimiser fits inside the selection CV are not in the iteration statistics (they would have raised STOP if not converged); never a result'}
    res['benchmark_within_registered_caps'] = detail['benchmark_fabricated_registered_scale']['seconds'] < real['caps']['wall_seconds'] and peak_rss_bytes() < real['caps']['max_rss_bytes'] and max(its) < opt['maxiter']

    hits = [p for p in opened if any(p == w or p.startswith(w + '/') for w in watch)]
    res['zero_registered_input_lock_or_attempt_path_opened'] = not hits and not attempt_dir(real, reg_path).exists()
    res = {k: bool(v) for k, v in res.items()}
    return {'pass': all(res.values()), 'checks': res, 'cases': len(res), 'detail': detail, 'wall_seconds': time.monotonic() - t0,
            'peak_rss_bytes': peak_rss_bytes(), 'sources_sha256': {SCRIPT_KEY: sha(SELF), reg_path.name: sha(reg_path)}, 'script_path': str(SELF),
            'registration_path': str(reg_path), 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
            'scope': 'synthetic in-memory arrays and fabricated temp-dir files only; no registered input, lock, run directory, corpus or score was opened; not a data result and not a review'}


def parse_cli(argv):
    ap = argparse.ArgumentParser(description='Post-hoc calibration of K seeds of three-head margins + ensemble (registration-anchored).')
    mode = ap.add_mutually_exclusive_group(required=True)
    for m in ('selfcheck', 'plan', 'execute'):
        mode.add_argument('--' + m, action='store_true')
    ap.add_argument('--registration', required=True, help='REGISTRATION.json; its directory holds REVIEW.json, SELFCHECK.json, PLAN.json and the attempt')
    ap.add_argument('--runs')
    ap.add_argument('--arm-prefix')
    ap.add_argument('--seeds')
    ap.add_argument('--corpus')
    ap.add_argument('--eval-meta')
    ap.add_argument('--baselines', help='NAME=path[,NAME=path...]')
    ap.add_argument('--reference-ensemble')
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    if a.selfcheck:
        return 'selfcheck', a.registration, None
    miss = [k for k in ('runs', 'arm_prefix', 'seeds', 'corpus', 'eval_meta', 'baselines', 'out') if getattr(a, k) is None]
    if miss:
        ap.error('--plan/--execute need: ' + ', '.join('--' + k.replace('_', '-') for k in miss))
    try:
        seeds = [int(s) for s in a.seeds.split(',')]
        base = dict(kv.split('=', 1) for kv in a.baselines.split(','))
    except ValueError:
        ap.error('--seeds must be comma-separated integers and --baselines NAME=path[,NAME=path...]')
    return ('plan' if a.plan else 'execute'), a.registration, {
        'runs': a.runs, 'arm_prefix': a.arm_prefix, 'seeds': seeds, 'corpus': a.corpus, 'eval_meta': a.eval_meta, 'baselines': base,
        'reference_ensemble': a.reference_ensemble, 'out': a.out}


if __name__ == '__main__':
    mode, regp, cli = parse_cli(sys.argv[1:])
    regdir = Path(regp).resolve().parent
    if mode == 'selfcheck':
        out = selfcheck(regp)
        save(regdir / 'SELFCHECK.json', out)
        print(json.dumps({k: out[k] for k in ('pass', 'cases', 'wall_seconds', 'peak_rss_bytes')}))
        print(json.dumps(out['checks'], indent=1))
        sys.exit(0 if out['pass'] else 1)
    if mode == 'plan':
        out = plan(regp, cli)
        save(regdir / 'PLAN.json', out)
        print(json.dumps({k: v for k, v in out.items() if k != 'registration'}, indent=1))
        sys.exit(0 if out['pass'] else 1)
    try:
        att = execute(regp, cli)
    except Stop as ex:
        sys.exit(str(ex) if str(ex).startswith('STOP') else 'STOP ' + str(ex))
    print(json.dumps({'success': True, 'attempt': str(att)}))
    sys.exit(0)
