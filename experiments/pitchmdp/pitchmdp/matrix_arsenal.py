"""EXP-P13-001: TRAIN-only per-pitch-type pitcher arsenal features for G0.

Additive to ``matrix_sharing``; frozen G0 sources are untouched. With the flag
off (``arsenal=None``) ``ArsenalContext`` returns the ``SharingContext`` array
unchanged, so the G0 path is byte-identical.

Context block (pitcher-static, action-independent, inserted before the seven
routing columns so ``training_arrays``/``SharingPredictor`` work unchanged):
for every type t of the frozen token vocabulary (T types), in vocabulary order,
nine channels ``ARSENAL_CHANNELS``. Candidate block: the same nine channels of
the row whose type is the current (candidate) token, gathered by
``with_candidate_arsenal`` after routing is stripped; unknown type -> zeros.
Total added width = 9 * (T + 1).

Shrinkage (per pitcher p, type t; n = TRAIN count, N = pitcher TRAIN count):
  share    = (n + tau * league_share_t) / (N + tau)
  physical = (sum_finite + tau * league_mean_t) / (n_finite + tau)
  reliability = n / (n + tau)   (kept raw, not standardized)
Primary fastball = the most-used of FASTBALLS (ties in FASTBALLS order); a
pitcher with none uses the league-most-used fastball type (its row is then the
league mean). Differences use the shrunk rows. Channels 0..7 are standardized
with TRAIN pitcher x type cell mean/std. Unknown pitchers get the n=0 limit,
i.e. the league per-type row (the existing league fallback).
"""
from __future__ import annotations

import numpy as np

from .matrix_sharing import CLUSTER_FEATURES

ARSENAL_CHANNELS = ('share', 'effective_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
                    'd_speed_vs_fastball', 'd_pfx_x_vs_fastball', 'd_pfx_z_vs_fastball', 'reliability')
FASTBALLS = ('FF', 'SI', 'FC')
WIDTH = len(ARSENAL_CHANNELS)
# Index into CLUSTER_FEATURES of the columns differenced against the fastball.
_DIFF = (0, 2, 3)


def _rows(n, finite_sum, finite_n, big_n, league_share, league_mean, fb_index, tau):
    """Raw [P, T, 9] arsenal rows before standardization."""
    share = (n + tau * league_share) / (big_n[:, None] + tau)
    physical = (finite_sum + tau * league_mean) / (finite_n + tau)
    fastball = physical[np.arange(len(n)), fb_index]
    diff = physical[:, :, _DIFF] - fastball[:, None, _DIFF]
    return np.concatenate([share[..., None], physical, diff, (n / (n + tau))[..., None]], axis=2)


def fit_pitcher_arsenal(train, type_vocabulary, *, tau=50.):
    if not len(train) or not train.split.eq('train').all():
        raise ValueError('Arsenal fit requires nonempty TRAIN only')
    if train.pitcher.isna().any():
        raise ValueError('Pitcher identity required')
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('Shrinkage tau must be positive')
    types = tuple(str(t) for t in type_vocabulary)
    if len(set(types)) != len(types):
        raise ValueError('Duplicate pitch-type vocabulary')
    fastballs = [types.index(t) for t in FASTBALLS if t in types]
    if not fastballs:
        raise ValueError('Primary fastball requires FF, SI or FC in the vocabulary')
    pitchers = np.sort(train.pitcher.astype(np.int64).unique())
    p_index = np.searchsorted(pitchers, train.pitcher.to_numpy(np.int64))
    t_map = {t: i for i, t in enumerate(types)}
    # Same string rule as MatrixHistoryStore: missing/unseen types are unknown.
    t_index = np.array([t_map.get(str(v), -1) for v in train.pitch_type], dtype=np.int64)
    values = train[list(CLUSTER_FEATURES)].to_numpy(float)
    finite = np.isfinite(values)
    big_n = np.bincount(p_index, minlength=len(pitchers)).astype(float)
    known = t_index >= 0  # Out-of-vocabulary/missing types count in N only.
    P, T, F = len(pitchers), len(types), len(CLUSTER_FEATURES)
    cell = p_index[known] * T + t_index[known]
    n = np.bincount(cell, minlength=P * T).reshape(P, T).astype(float)
    finite_sum = np.stack([np.bincount(cell, np.where(finite[known, f], values[known, f], 0.), P * T)
                           for f in range(F)], axis=1).reshape(P, T, F)
    finite_n = np.stack([np.bincount(cell, finite[known, f].astype(float), P * T)
                         for f in range(F)], axis=1).reshape(P, T, F)
    type_n = n.sum(0)
    league_share = type_n / len(train)
    column_mean = np.array([values[finite[:, f], f].mean() if finite[:, f].any() else 0. for f in range(F)])
    league_mean = np.where(finite_n.sum(0) > 0, finite_sum.sum(0) / np.maximum(finite_n.sum(0), 1), column_mean)
    league_fb = max(fastballs, key=lambda i: (type_n[i], -fastballs.index(i)))
    fb_counts = n[:, fastballs]
    fb_index = np.where(fb_counts.max(1) > 0, np.asarray(fastballs)[fb_counts.argmax(1)], league_fb)
    raw = _rows(n, finite_sum, finite_n, big_n, league_share, league_mean, fb_index, tau)
    zero = np.zeros((1, T))
    fallback = _rows(zero, np.zeros((1, T, F)), np.zeros((1, T, F)), np.zeros(1),
                     league_share, league_mean, np.array([league_fb]), tau)[0]
    cells = raw[..., :-1].reshape(-1, WIDTH - 1)
    mean, scale = cells.mean(0), cells.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)

    def standardize(rows):
        return np.concatenate([(rows[..., :-1] - mean) / scale, rows[..., -1:]], axis=-1)

    table, fallback = standardize(raw), standardize(fallback)
    return {'version': 'pitcher_arsenal_v1', 'tau': float(tau), 'type_vocabulary': list(types),
            'channels': list(ARSENAL_CHANNELS), 'fastballs': list(FASTBALLS),
            'league_share': league_share.tolist(), 'league_mean': league_mean.tolist(),
            'mean': mean.tolist(), 'scale': scale.tolist(), 'fallback': fallback.reshape(-1).tolist(),
            'pitcher_arsenal': {str(int(p)): row.reshape(-1).tolist() for p, row in zip(pitchers, table)},
            'pitcher_primary_fastball': {str(int(p)): types[i] for p, i in zip(pitchers, fb_index)},
            'pitcher_counts': {str(int(p)): int(c) for p, c in zip(pitchers, big_n)},
            'scope': 'eligible TRAIN only; per pitcher x vocabulary type share/physical means shrunk to league per-type means',
            'unseen_rule': 'league per-type row (n=0 shrinkage limit); no future-player statistics'}


class ArsenalContext:
    """SharingContext plus the arsenal block before the seven routing columns."""
    def __init__(self, sharing, arsenal=None):
        self.sharing, self.arsenal = sharing, arsenal

    def transform(self, frame):
        context = self.sharing.transform(frame)
        if self.arsenal is None:
            return context
        table, fallback = self.arsenal['pitcher_arsenal'], self.arsenal['fallback']
        block = np.array([table.get(str(int(pid)), fallback) for pid in frame.pitcher], dtype=np.float32)
        return np.column_stack([context[:, :-7], block, context[:, -7:]]).astype(np.float32)

    def report(self):
        report = self.sharing.report()
        if self.arsenal is None:
            return report
        from .matrix_data import canonical_hash
        n_types = len(self.arsenal['type_vocabulary'])
        return {**report, 'arsenal_sha256': canonical_hash(self.arsenal),
                'arsenal': f'{n_types} types x {WIDTH} channels {list(ARSENAL_CHANNELS)} before routing columns',
                'arsenal_unknown': self.arsenal['unseen_rule']}


def with_candidate_arsenal(arrays, n_types):
    """Append the candidate type's arsenal row; arrays must have routing stripped."""
    tokens, valid, context = arrays
    width = n_types * WIDTH
    if tokens.shape[2] < 9 + n_types or context.shape[1] < width:
        raise ValueError('Arsenal block or type channels missing')
    candidate = tokens[:, -1, 9:9 + n_types]  # channel 8 is the unknown type.
    table = context[:, -width:].reshape(len(context), n_types, WIDTH)
    row = np.einsum('nt,ntc->nc', candidate, table).astype(np.float32)
    return tokens, valid, np.column_stack([context, row]).astype(np.float32)


class CandidateArsenalModel:
    """Model wrapper used for both fit inputs and SharingPredictor inference."""
    def __init__(self, model, n_types):
        self.model, self.n_types = model, n_types

    def logits(self, arrays):
        return self.model.logits(with_candidate_arsenal(arrays, self.n_types))
