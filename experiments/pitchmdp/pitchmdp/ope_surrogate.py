"""EXP-OPE-RES-01 R1 luck-removed reward surrogates (WE units), TRAIN-only tables, calibration.

R1-a ("xWE-class"): r_a = game.terminal_values(s_K, we, advancement)[e] for the PA's terminal class
e = game._EVENT_GROUP[events] among the nine TERMINALS (s_K = pre-pitch state of the PA's last row);
a terminal event outside the nine classes passes the realized reward through.
R1-b ("xWE-BIP"): r_b = sum_e P_hat(e | x, dp) * terminal_values(s_K)[e] for a hit_into_play last row
with finite launch_speed and launch_angle (x = EV x LA x spray bins, dp = runner on first and < 2
outs); every other end uses R1-a. P_hat is a hierarchical Dirichlet shrinkage table fitted on TRAIN
(2023-05-15..2025-04-30) rows only, with m chosen by game_pk-parity 2-fold cross-validated log-loss.
The same frozen C0 WE and EmpiricalAdvancement the planner and q-hat use give terminal_values.

Only rows that already have an observed R1-0 reward get a surrogate, so the censoring classification
(and with it the point set P) is identical for every R1 arm. The input facts are never modified:
``surrogate_rewards`` returns new per-PA rewards with a path tag. The violation calibration
(kappa_p, zeta_p) and the TRAIN exclusion diagnostic delta_g are TRAIN-only and cross-fitted.
Nothing here reads files; callers pass already-guarded rows.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .game import _EVENT_GROUP, GameState, terminal_values
from .ope_redesign_registry import (BIP_CLASSES, EV_EDGES, HIT_CLASSES, LA_EDGES, M_SMOOTH_GRID, N_CELLS, N_EV, N_LA,
                                    N_SPRAY, SPRAY_LIMIT_DEG, SPRAY_ORIGIN, TERMINALS9, TRAIN_EVENT_MAP, TRAIN_WINDOW,
                                    pitch_group, select_minimizer)
from .policy_artifacts import IntegrityError, _require

N_CLASSES = len(BIP_CLASSES)
DP_INDEX = BIP_CLASSES.index('double_play')
OUT_INDEX = BIP_CLASSES.index('out')
HIT_MASK = np.array([c in HIT_CLASSES for c in BIP_CLASSES])


# ---------------------------------------------------------------- bins
def spray_angle(hc_x, hc_y, stand):
    """phi = atan((hc_x - 125.42) / (198.27 - hc_y)) in degrees, signed pull-positive by batter stand
    (negated for R, kept for L); NaN when hc_x, hc_y or stand is missing or phi is nonfinite."""
    x = np.asarray(hc_x, dtype=np.float64)
    y = np.asarray(hc_y, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.degrees(np.arctan((x - SPRAY_ORIGIN[0]) / (SPRAY_ORIGIN[1] - y)))
    side = np.asarray(pd.Series(np.asarray(stand, dtype=object)).astype(object).to_numpy())
    sign = np.where(side == 'R', -1.0, np.where(side == 'L', 1.0, np.nan))
    out = phi * sign
    return np.where(np.isfinite(out), out, np.nan)


def spray_bin(phi):
    phi = np.asarray(phi, dtype=np.float64)
    return np.where(~np.isfinite(phi), 3, np.where(phi > SPRAY_LIMIT_DEG, 0, np.where(phi < -SPRAY_LIMIT_DEG, 2, 1)))


def _edge_bin(values, edges):
    v = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(v), np.searchsorted(np.asarray(edges, dtype=np.float64), v, side='right'), -1)


def dp_flag(bases, outs):
    return ((np.asarray(bases, dtype=np.int64) & 1) > 0) & (np.asarray(outs, dtype=np.int64) < 2)


def quality_cells(launch_speed, launch_angle, hc_x, hc_y, stand, bases, outs):
    """960-cell index ((EV * 10 + LA) * 4 + spray) * 2 + dp, or -1 when EV or LA is missing."""
    ev, la = _edge_bin(launch_speed, EV_EDGES), _edge_bin(launch_angle, LA_EDGES)
    spray = spray_bin(spray_angle(hc_x, hc_y, stand))
    x = (ev * N_LA + la) * N_SPRAY + spray
    cell = x * 2 + dp_flag(bases, outs).astype(np.int64)
    return np.where((ev >= 0) & (la >= 0), cell, -1)


def cell_parts(cell):
    cell = np.asarray(cell, dtype=np.int64)
    dp, x = cell % 2, cell // 2
    return x // (N_LA * N_SPRAY), (x // N_SPRAY) % N_LA, x % N_SPRAY, dp


# ---------------------------------------------------------------- TRAIN guards and mapping
def guard_train(dates, *, window=TRAIN_WINDOW):
    """Stop before any output if a row outside TRAIN (or any 2026 row) reaches table fitting."""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    _require(len(d) > 0 and d.notna().all(), 'TRAIN rows need dates')
    _require(not (d.dt.year >= 2026).any(), 'a 2026 row reached a TRAIN table fit; stopping before outputs')
    _require(bool(d.min() >= pd.Timestamp(window[0]) and d.max() <= pd.Timestamp(window[1])),
             'a row outside TRAIN reached a table fit; stopping before outputs')
    return d


def train_bip_rows(frame):
    """TRAIN in-play rows for the R1-b table: hit_into_play with finite EV and LA, event mapped by the
    registered TRAIN mapping (dp-ineligible double plays -> out, the implementation choice). Returns
    (cells, classes, index into frame, report)."""
    guard_train(frame['game_date'])
    bip = frame['description'].eq('hit_into_play').to_numpy()
    events = frame['events'].astype(object).to_numpy()
    mapped = np.array([TRAIN_EVENT_MAP.get(e) if isinstance(e, str) else None for e in events], dtype=object)
    cells = quality_cells(frame['launch_speed'], frame['launch_angle'], frame['hc_x'], frame['hc_y'], frame['stand'],
                          _bases(frame), frame['outs_when_up'])
    usable = bip & (cells >= 0) & np.array([m is not None for m in mapped])
    classes = np.full(len(frame), -1, dtype=np.int64)
    for k, name in enumerate(BIP_CLASSES):
        classes[usable & (mapped == name)] = k
    dp = cells % 2
    remapped = usable & (classes == DP_INDEX) & (dp == 0)
    classes[remapped] = OUT_INDEX
    report = {'train_rows': int(len(frame)), 'bip_rows': int(bip.sum()),
              'bip_missing_ev_or_la': int((bip & (cells < 0)).sum()),
              'bip_unmapped_event': int((bip & (cells >= 0) & np.array([m is None for m in mapped])).sum()),
              'bip_used': int(usable.sum()), 'dp_ineligible_double_play_to_out': int(remapped.sum()),
              'spray_missing_among_used': int((usable & (cell_parts(np.maximum(cells, 0))[2] == 3)).sum()),
              'mapping_counts': {str(k): int(v) for k, v in pd.Series(events[usable]).value_counts().items()}}
    index = np.flatnonzero(usable)
    return cells[index], classes[index], index, report


def _bases(frame):
    if 'bases' in frame:
        return frame['bases'].to_numpy(np.int64)
    return sum(((frame[c].notna().to_numpy()) << i).astype(np.int64) for i, c in enumerate(('on_1b', 'on_2b', 'on_3b')))


# ---------------------------------------------------------------- hierarchical shrinkage table
def class_counts(cells, classes, weights=None):
    cells, classes = np.asarray(cells, dtype=np.int64), np.asarray(classes, dtype=np.int64)
    _require(cells.shape == classes.shape and ((cells >= 0) & (cells < N_CELLS)).all()
             and ((classes >= 0) & (classes < N_CLASSES)).all(), 'cells/classes out of range')
    return np.bincount(cells * N_CLASSES + classes, weights=weights, minlength=N_CELLS * N_CLASSES).reshape(N_CELLS, N_CLASSES)


def fit_table(counts, m):
    """P(e | EV, LA, spray, dp) <- P(e | EV, LA, dp) <- P(e | LA, dp) <- P(e | dp) with m pseudo-counts per
    level; the root is the maximum likelihood P(e | dp). Double-play mass is 0 at dp = 0."""
    n0 = np.asarray(counts, dtype=np.float64).reshape(N_EV, N_LA, N_SPRAY, 2, N_CLASSES)
    _require(np.isfinite(n0).all() and (n0 >= 0).all() and m > 0, 'counts and a positive m required')
    _require(n0[:, :, :, 0, DP_INDEX].sum() == 0, 'double-play counts at dp = 0 (map them to out first)')
    n1 = n0.sum(axis=2)  # EV, LA, dp
    n2 = n1.sum(axis=0)  # LA, dp
    n3 = n2.sum(axis=0)  # dp
    totals = n3.sum(axis=1)
    _require((totals > 0).all(), 'both dp strata need TRAIN rows for the root level')
    p3 = n3 / totals[:, None]
    p2 = (n2 + m * p3[None]) / (n2.sum(axis=-1, keepdims=True) + m)
    p1 = (n1 + m * p2[None]) / (n1.sum(axis=-1, keepdims=True) + m)
    p0 = (n0 + m * p1[:, :, None]) / (n0.sum(axis=-1, keepdims=True) + m)
    table = p0.reshape(N_CELLS, N_CLASSES)
    _require(np.allclose(table.sum(axis=1), 1, rtol=0, atol=1e-12) and (table[0::2, DP_INDEX] == 0).all(),
             'table rows must be probabilities with no double play at dp = 0')
    return table


def log_loss(table, counts):
    """Total negative log-likelihood of the counts under the table (inf if an observed class has 0 mass)."""
    counts = np.asarray(counts, dtype=np.float64)
    with np.errstate(divide='ignore'):
        logp = np.log(table)
    seen = counts > 0
    if (table[seen] <= 0).any():
        return math.inf
    return float(-(counts[seen] * logp[seen]).sum())


def cv_log_loss(fold_counts, m):
    """game_pk-parity 2-fold CV: fit on one fold, score the other, both directions; the pooled held-out
    negative log-likelihood divided by the held-out row count."""
    a, b = fold_counts
    total = 0.
    for train, test in ((a, b), (b, a)):
        total += log_loss(fit_table(train, m), test)
    return total / float(np.sum(a) + np.sum(b))


def select_m(fold_counts, grid=M_SMOOTH_GRID):
    losses = []
    for m in grid:
        try:
            losses.append(cv_log_loss(fold_counts, m))
        except IntegrityError:
            losses.append(math.inf)
    chosen = select_minimizer(grid, losses)
    _require(chosen is not None, 'every m has a nonfinite cross-validated log-loss')
    return chosen, dict(zip((str(m) for m in grid), losses))


def fit_r1b(cells, classes, folds, *, m=None):
    """TRAIN table: m selected by CV unless fixed (the smoke fixes m and computes no log-loss)."""
    folds = np.asarray(folds, dtype=np.int64)
    _require(set(np.unique(folds)) <= {0, 1}, 'folds must be the game_pk parity (0/1)')
    fold_counts = [class_counts(cells[folds == f], classes[folds == f]) for f in (0, 1)]
    losses = None
    if m is None:
        m, losses = select_m(fold_counts)
    table = fit_table(fold_counts[0] + fold_counts[1], m)
    return {'table': table, 'm': m, 'cv_log_loss': losses, 'rows': int(len(cells))}


def table_sha256(table, m):
    arr = np.ascontiguousarray(np.asarray(table, dtype='<f8'))
    head = json.dumps({'shape': list(arr.shape), 'm': m, 'classes': list(BIP_CLASSES), 'cells': N_CELLS},
                      sort_keys=True).encode()
    return hashlib.sha256(head + b'\0' + arr.tobytes()).hexdigest()


# ---------------------------------------------------------------- surrogate values
class TerminalValues:
    """Cached game.terminal_values(state, we, advancement) for the frozen C0 WE (the planner's values)."""
    def __init__(self, we, advancement):
        self.we, self.advancement, self.cache = we, advancement, {}

    def __call__(self, state):
        if state not in self.cache:
            self.cache[state] = terminal_values(state, self.we, self.advancement)
        return self.cache[state]


def r1a_value(tv, event):
    e = _EVENT_GROUP.get(event) if isinstance(event, str) else None
    return (float(tv[e]), 'class') if e in TERMINALS9 else (None, 'pass_through')


def r1b_value(tv, table, cell):
    p = table[int(cell)]
    return float(sum(p[k] * tv[name] for k, name in enumerate(BIP_CLASSES)))


def surrogate_rewards(facts, rows, terminal, table):
    """{pa_id: {'R1_a': (reward, path), 'R1_b': (reward, path)}} for every PA with an observed R1-0
    reward. ``rows``: {pa_id: last-row mapping (events, description, launch_speed, launch_angle,
    hc_x, hc_y, stand, pre-pitch game state)}; ``terminal``: state -> terminal_values dict."""
    out = {}
    for pa_id, info in facts.items():
        reward = info.get('reward')
        if reward is None:
            continue
        row = rows[pa_id]
        state = GameState.from_row(row)
        tv = terminal(state)
        r_a, path_a = r1a_value(tv, row.get('events'))
        if r_a is None:
            r_a = float(reward)
        cell = int(quality_cells([row.get('launch_speed')], [row.get('launch_angle')], [row.get('hc_x')],
                                 [row.get('hc_y')], [row.get('stand')], [state.bases], [state.outs])[0])
        if row.get('description') == 'hit_into_play' and cell >= 0:
            r_b, path_b = r1b_value(tv, table, cell), 'bip_quality'
        elif row.get('description') == 'hit_into_play':
            r_b, path_b = r_a, 'bip_missing_ev_la_fallback'
        else:
            r_b, path_b = r_a, f'non_bip_{path_a}'
        for value in (r_a, r_b):
            _require(math.isfinite(value) and 0 <= value <= 1, 'surrogate WE outside [0, 1]')
        out[pa_id] = {'R1_a': (r_a, path_a), 'R1_b': (r_b, path_b),
                      'spray_missing': bool(path_b == 'bip_quality' and cell_parts(cell)[2] == 3)}
    return out


def path_shares(surrogates):
    """R5-i: shares of the surrogate paths among PAs with an observed reward (counts and shares only)."""
    n = len(surrogates)
    count = lambda key, path: sum(v[key][1] == path for v in surrogates.values())
    bip = count('R1_b', 'bip_quality')
    return {'pas_with_reward': n,
            'bip_with_ev_la': bip, 'bip_missing_ev_la': count('R1_b', 'bip_missing_ev_la_fallback'),
            'non_bip_class': count('R1_b', 'non_bip_class'), 'pass_through': count('R1_a', 'pass_through'),
            'spray_missing_among_bip': sum(v['spray_missing'] for v in surrogates.values()),
            'shares': {'bip_with_ev_la': bip / n if n else None,
                       'bip_missing_ev_la': count('R1_b', 'bip_missing_ev_la_fallback') / n if n else None,
                       'non_bip_class': count('R1_b', 'non_bip_class') / n if n else None,
                       'pass_through': count('R1_a', 'pass_through') / n if n else None}}


# ---------------------------------------------------------------- violation calibration (TRAIN only)
def hit_probability(table, cells):
    return table[np.asarray(cells, dtype=np.int64)][:, HIT_MASK].sum(axis=1)


def group_residuals(values, groups, names):
    """Per group: mean residual (delta_g of the design), SE_g = SD (ddof 1) / sqrt(n_g), n_g."""
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups, dtype=object)
    out = {}
    for g in names:
        v = values[groups == g]
        out[g] = {'n': int(len(v)), 'mean_residual': float(v.mean()) if len(v) else None,
                  'se': float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None}
    return out


def crossfit_hit_residuals(cells, classes, folds, groups, m):
    """kappa calibration input: residual 1[hit] - P_hat_{other fold}(hit | x, dp), both directions."""
    folds = np.asarray(folds, dtype=np.int64)
    residual = np.empty(len(cells))
    for f in (0, 1):
        held = folds == f
        table = fit_table(class_counts(cells[~held], classes[~held]), m)
        residual[held] = HIT_MASK[classes[held]].astype(np.float64) - hit_probability(table, cells[held])
    return residual


def _sign(value):
    return 0 if value is None or value == 0 else (1 if value > 0 else -1)


def _bisect(gap, target, *, rel_tol=1e-10, upper_max=1000.0):
    """Smallest-bracket bisection for gap(t) = target on t >= 0 (gap(0) = 0, nondecreasing)."""
    _require(math.isfinite(target) and target >= 0, 'calibration target must be finite')
    if target == 0:
        return 0.0
    hi = 1.0
    while gap(hi) < target:
        hi *= 2
        _require(hi <= upper_max, 'calibration target not reachable inside the registered bracket')
    lo = 0.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if gap(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= rel_tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def kappa_world_residual(kappa, hit_prob, groups, signs, names):
    """World group residual E[P_kappa(hit | x, dp, g) - P_hat(hit | x, dp)] over the TRAIN rows of g."""
    out = {}
    for g in names:
        h = hit_prob[groups == g]
        w = math.exp(kappa * signs[g])
        out[g] = float(np.mean(h * w / (h * w + 1 - h) - h)) if len(h) else 0.0
    return out


def calibrate_kappa(hit_prob, groups, diagnostics, names):
    """kappa_p: the world's max |group residual| over ``names`` equals max_g(|delta_g| + 1.96 SE_g)."""
    present = [g for g in names if diagnostics[g]['n'] > 1]
    _require(present, 'no pitch group with TRAIN in-play rows for the kappa calibration')
    target = max(abs(diagnostics[g]['mean_residual']) + 1.96 * diagnostics[g]['se'] for g in present)
    signs = {g: _sign(diagnostics[g]['mean_residual']) for g in names}
    gap = lambda k: max(abs(v) for v in kappa_world_residual(k, hit_prob, groups, signs, present).values())
    return {'kappa_p': _bisect(gap, target), 'target': target, 'signs': signs, 'groups': present}


def tilted_out_mass(p_score, zeta, sign):
    """Scoring-outcome mass of the C0 out distribution after the zeta tilt (identity at zeta = 0)."""
    if zeta == 0 or sign == 0 or p_score <= 0 or p_score >= 1:
        return p_score
    return float(expit(logit(p_score) + zeta * sign))


def zeta_world_residual(zeta, p_score, run_gap, groups, signs, names):
    out = {}
    for g in names:
        sel = groups == g
        if not sel.any():
            out[g] = 0.0
            continue
        moved = np.array([tilted_out_mass(p, zeta, signs[g]) for p in p_score[sel]]) - p_score[sel]
        out[g] = float(np.mean(moved * run_gap[sel]))
    return out


def calibrate_zeta(p_score, run_gap, groups, diagnostics, names):
    """zeta_p: same rule for the run-scoring residual of outs with a runner on third and < 2 outs.
    ``p_score``/``run_gap``: per TRAIN row, the C0 scoring mass and E[runs | scoring] - E[runs | not]."""
    present = [g for g in names if diagnostics[g]['n'] > 1]
    _require(present, 'no pitch group with eligible TRAIN outs for the zeta calibration')
    target = max(abs(diagnostics[g]['mean_residual']) + 1.96 * diagnostics[g]['se'] for g in present)
    signs = {g: _sign(diagnostics[g]['mean_residual']) for g in names}
    gap = lambda z: max(abs(v) for v in zeta_world_residual(z, p_score, run_gap, groups, signs, present).values())
    return {'zeta_p': _bisect(gap, target), 'target': target, 'signs': signs, 'groups': present}


def out_scoring_split(state, advancement, we):
    """C0 'out' advancement split for the zeta layer: [(prob, runs, defensive WE of the next state)],
    the scoring mass, E[runs | scoring] - E[runs | not scoring]."""
    outcomes = []
    for prob, nxt in advancement.distribution(state, 'out'):
        before = state.home_score if state.half == 'Bot' else state.away_score
        after = nxt.home_score if state.half == 'Bot' else nxt.away_score
        outcomes.append((float(prob), int(after - before), float(we.predict_defense(nxt, state.defender_is_home))))
    p_score = sum(p for p, runs, _ in outcomes if runs > 0)
    mean = lambda part: (sum(p * r for p, r, _ in part) / sum(p for p, _, _ in part)) if part else 0.0
    score = [o for o in outcomes if o[1] > 0]
    other = [o for o in outcomes if o[1] <= 0]
    return outcomes, float(p_score), float(mean(score) - mean(other))


def group_of_types(pitch_types, names):
    groups = np.array([pitch_group(t) if isinstance(t, str) else None for t in pitch_types], dtype=object)
    return np.where(np.isin(groups, list(names)), groups, None)
