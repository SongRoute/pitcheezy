"""Verbatim copies of the scoring functions of cuda-score-1120 scripts/svc_p9/eval_prediction.py (sha 1c49e173...).

Copied (not imported) so this package does not put another worktree's ``src`` on sys.path. ``verify_reference()``
re-reads the reference file, checks its sha256 and that every copied function's source text is identical; score.py
calls it before reading any label. Constants below are the reference module's values (checked too).
"""
from __future__ import annotations

import ast
import hashlib
import pathlib

import numpy as np

from pitcheezy.interfaces.pitch_types import PITCH_NAMES, STATCAST_TO_PITCH_ID

EPS = 1e-3
NLL_CLIP = 1e-12
BINS = 10
REPLICATES, SEED = 2000, 20261110
UNIFORM = [1.0 / 9] * 9
K = len(PITCH_NAMES)
COPIED = ('label', 'frozen_frequency', 'floor', 'row_parts', 'pooled', 'game_weights', 'paired_contrast')
CONSTANTS = ('EPS = 1e-3', 'NLL_CLIP = 1e-12', 'BINS = 10', 'REPLICATES, SEED = 2000, 20261110', 'UNIFORM = [1.0 / 9] * 9',
             'K = len(PITCH_NAMES)')


def label(code):
    """9-group index of an actual pitch type, or None (excluded as a target)."""
    if code is None:
        return None
    return STATCAST_TO_PITCH_ID.get(str(code))

def frozen_frequency(table, pitcher, balls, strikes):
    """gpu14h_prepitch/run.py:frequency for one row (alpha 4; global -> pitcher -> pitcher:balls:strikes)."""
    if table['alpha'] != 4.0:
        raise ValueError('frozen frequency alpha')
    count = np.asarray(table['global'], np.float64)
    globalp = (count + 4 / 9) / (count.sum() + 4)
    pc = table['pitcher'].get(str(pitcher)); cc = table['context'].get(f'{pitcher}:{balls}:{strikes}')
    pp = globalp if pc is None else (np.asarray(pc) + 4 * globalp) / (sum(pc) + 4)
    return pp if cc is None else (np.asarray(cc) + 4 * pp) / (sum(cc) + 4)

def floor(q, eps=EPS):
    q = np.asarray(q, np.float64)
    return (1 - eps) * q + eps / K

def row_parts(q, y, clip=None):
    q = np.asarray(q, np.float64); y = np.asarray(y, np.int64)
    truth = q[np.arange(len(y)), y]
    with np.errstate(divide='ignore'):
        nll = -np.log(np.clip(truth, clip, 1.0)) if clip else -np.log(truth)
    onehot = np.eye(K)[y]
    brier = ((q - onehot) ** 2).sum(1)
    conf = q.max(1); hit = (q.argmax(1) == y).astype(np.float64)
    b = np.minimum((conf * BINS).astype(int), BINS - 1)
    return {'nll': nll, 'brier': brier, 'gap': conf - hit, 'bin': b, 'top1': hit, 'zero_truth': (truth == 0)}

def pooled(parts, weights=None):
    w = np.ones(len(parts['nll'])) if weights is None else np.asarray(weights, np.float64)
    total = w.sum()
    cells = np.zeros(BINS); np.add.at(cells, parts['bin'], w * parts['gap'])
    return {'nll': float(w @ parts['nll'] / total), 'brier': float(w @ parts['brier'] / total),
            'ece10': float(np.abs(cells).sum() / total), 'top1': float(w @ parts['top1'] / total)}

def game_weights(games, replicates=REPLICATES, seed=SEED):
    """Row weights per replicate: each game drawn with replacement; a row's weight = its game's draw count."""
    ids, inverse = np.unique(np.asarray(games), return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        yield counts[inverse].astype(np.float64)

def paired_contrast(q_cand, q_base, y, games, *, clip=None, replicates=REPLICATES, seed=SEED):
    a, b = row_parts(q_cand, y, clip), row_parts(q_base, y, clip)
    point = {k: pooled(a)[k] - pooled(b)[k] for k in ('nll', 'brier', 'ece10', 'top1')}
    reps = {k: [] for k in point}
    for w in game_weights(games, replicates, seed):
        pa, pb = pooled(a, w), pooled(b, w)
        for k in reps:
            reps[k].append(pa[k] - pb[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in reps.items()}
    return {'point': point, 'ci95': ci, 'candidate': pooled(a), 'baseline': pooled(b),
            'zero_truth': {'candidate': int(a['zero_truth'].sum()), 'baseline': int(b['zero_truth'].sum())}}


def verify_reference(path, sha256):
    """Reference file sha, identical source of every copied function, and the constants (raises SystemExit)."""
    raw = pathlib.Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise SystemExit('eval_prediction.py reference differs from its pin')
    ref, here = raw.decode(), pathlib.Path(__file__).read_text()
    def segs(text):
        t = ast.parse(text)
        return {n.name: ast.get_source_segment(text, n) for n in t.body if isinstance(n, ast.FunctionDef)}
    a, b = segs(ref), segs(here)
    bad = [n for n in COPIED if a.get(n) is None or a.get(n) != b.get(n)]
    lines = set(ref.splitlines())
    bad += [c for c in CONSTANTS if c not in lines]
    if bad:
        raise SystemExit(f'copied scoring code differs from eval_prediction.py: {bad}')
    return {'path': str(path), 'sha256': sha256, 'functions': list(COPIED)}
