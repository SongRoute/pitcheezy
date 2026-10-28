"""EXP-OPE-RES-02 A2b-S3 sealed evaluator (25 rows), its S6 12-row mode for G-INT-EVAL and the integrity record.

Copied from the EXP-OPE-RES-01 ``ope_sealed.rehearsal_evaluate`` (design 9.1: the A2 evaluator is bound to its 12
pairs, its bootstrap seeds and its IDs) with the row list, the denominator axis, the B2 rho recomputation, the A2b
seeds and IDs. Every computation of a B0 row is the A2 code path line for line, so on the same ledger and seeds the
12 B0 rows equal ``ope_sealed.rehearsal_evaluate`` bit for bit (tested); ``ope_surrogate``, ``ope_variants``,
``ope_bootstrap``, ``ope_diagnostics`` and ``policy_estimator`` are imported unchanged.

Modes (``ope_registry_a2b``):
- ``REHEARSAL`` (the new DEV 150 ledger, A2b seeds): the 25 rows - the 12 B0 rows (= the A2 pairs), the 12 B2 rows
  (the same lines with rho recomputed from pi_b_hat_B2: ``decisions_b2``; descriptive under D192 2 3) and the
  selection-free R1_b x R2_1 x B0 row at lambda = 100, whose public-form items (pairs, S5, R5 e / f) are written to
  the sealed file only (``sealed['lambda_fixed_row_public']``, opened with ``unblinded.json`` at A2b-UB; G-CODE
  MAJOR-1: next to the R1_b x R2_1 row it would reveal a selected lambda = 100 before the decision, and it is a
  per-grid-value half-width, design 9.5); the public file holds the 24 ``PUBLIC_ROWS``;
- ``S6_GINT_EVAL`` (the S6 record, the A2 seeds): the 12 B0 rows ONLY - the 13 B2 and lambda-fixed rows are refused
  (never computed or written on S6, design 9.3 / review R3-MINOR-1).

The point set P, the rewards and the end classification come from the B0 baseline; a B2 line must keep the B0
classification (only its ratios change; rho0 stays 0 under any denominator), else a deterministic integrity
failure. Every signed real-data quantity goes to the sealed document only; the public document holds exactly the
typed allowlist (``ope_registry_a2b.validate_public``). ``gint_eval_record`` compares the S6-mode output with the
unblinded EXP-OPE-RES-01 record (public and unblinded values) in the registered absolute 1e-12 form and writes only
pass/fail, max |delta|, the tolerance and the key lists.
"""
from __future__ import annotations

import math

import numpy as np

from . import ope_bootstrap as ob
from . import ope_diagnostics as od
from . import ope_registry_a2b as rb
from . import ope_variants as ov
from . import policy_estimator as est
from .ope_redesign_registry import INVALID_SHARE_MAX, MIN_GAMES, MIN_PA_STARTS, MIN_PITCHERS, in_point_set
from .ope_sealed import _bits, _jsonable, canonical_bytes, sha256_bytes, unblind  # noqa: F401  (re-exported)
from .policy_artifacts import IntegrityError, _require

BOOTS = rb.BOOTS
INF = rb.INF
NOTES = {'R5_f_normalization': 'the point part of R5 f uses the common denominator N (all E0 PAs); H_point is a mean '
                               'over P, a different normalisation',
         'coverage': '95% bounds are nominal per row and cell, not simultaneous over the selected tuple',
         'point_set': 'P = E0 PAs whose B0 baseline paired delta bounds are equal; a row under which a P PA is not a '
                      'point has no P mean (its P statistics are invalid); the B2 rows use the same P',
         'b2_rows': 'B2 rows recompute rho = pi / pi_b_hat_B2 on the same lines (descriptive: D192 decision 2 '
                    'option 3); R5 a, d, h and i are denominator-free and computed once (B0); R5 b, c and g are '
                    'computed per denominator (B2: public items 10-11)',
         'lambda_fixed_row': 'R1_b x R2_1 x B0 with lambda fixed at 100 (no selection; review NOTE N1); its public-form '
                             'items are sealed until A2b-UB (G-CODE MAJOR-1) and are never a gate or decision input',
         'pigeonhole': 'the descriptive two-way bootstrap is run only with a registered seed (the S6 mode, A2 seed); '
                       'the A2b registration lists no pigeonhole role, so the rehearsal records NOT_RUN'}


def _label(value):
    return 'inf' if value == INF else str(value)


def _row(r1, r2, den):
    return rb.row_id(r1, r2, den)


# ---------------------------------------------------------------- placebo (A2 code, injectable stream)
def placebo_check(point_lines_steps, table, resamplers, *, rewards, stream, placebos=od.PLACEBOS, checkpoint=None):
    """``ope_diagnostics.placebo_check`` with the placebo substream injected: ``stream(k)`` -> SeedSequence (A2:
    [20261027, 1, k]; A2b: [20261028, 2, k]). With the A2 stream it is bit-identical to the A2 function (tested)."""
    keys = sorted({(e['pitcher'], od._bucket(e['balls'], e['strikes'])) for e in table.entries})
    key_index = {k: i for i, k in enumerate(keys)}
    row_key = np.array([key_index[(e['pitcher'], od._bucket(e['balls'], e['strikes']))] for e in table.entries])
    n_actions = table.mask.shape[1]
    rows = np.arange(len(table))
    v_ref = np.nansum(np.where(table.mask, table.ref * np.nan_to_num(table.q), 0.), axis=1)
    qa = table.q[rows, table.a]
    rho_ref = table.rho(table.ref)
    steps = [np.asarray(s, dtype=np.int64) for s in point_lines_steps]
    lengths = np.array([len(s) for s in steps], dtype=np.int64)
    L = max(int(lengths.max(initial=0)), 1)
    grid = np.full((len(steps), L), -1, dtype=np.int64)
    for i, s in enumerate(steps):
        grid[i, :len(s)] = s
    take = lambda values: np.where(grid >= 0, values[np.maximum(grid, 0)], 0.)
    ends = np.asarray(rewards, dtype=np.float64)
    base_r, slope_r = od.affine_padded(take(v_ref), take(qa), take(rho_ref), lengths)
    value_r = base_r + slope_r * ends
    deltas, ses, taus = [], {name: [] for name in resamplers}, []
    for k in range(placebos):
        if checkpoint is not None:
            checkpoint()
        xi = np.random.default_rng(stream(k)).standard_normal((len(keys), n_actions))[row_key]
        tau = od.tune_tau(table.ref, table.mask, xi)
        pi_p = od.placebo_policy(table.ref, table.mask, xi, tau)
        v_p = np.nansum(np.where(table.mask, pi_p * np.nan_to_num(table.q), 0.), axis=1)
        rho_p = table.rho(pi_p)
        base_p, slope_p = od.affine_padded(take(v_p), take(qa), take(rho_p), lengths)
        delta_i = (base_p + slope_p * ends) - value_r
        deltas.append(float(delta_i.mean()))
        taus.append(tau)
        for name, (resampler, p_rows) in resamplers.items():
            if resampler is None or not resampler.eligible:
                continue
            m = resampler.m
            num = m @ resampler.sums(delta_i, p_rows)
            den = m @ resampler.sums(np.ones(len(p_rows)), p_rows)
            with np.errstate(divide='ignore', invalid='ignore'):
                rep = num / den
            ok = (den > 0) & np.isfinite(rep)
            ses[name].append(float(np.std(rep[ok], ddof=1)) if ok.sum() > 1 else math.nan)
    sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else math.nan
    ratios = {name: (sd / float(np.mean(v)) if v and np.isfinite(v).all() and np.mean(v) > 0 else None)
              for name, v in ses.items()}
    flag = any(r is not None and r > od.PLACEBO_FLAG for r in ratios.values())
    return ({'placebos': placebos, 'sd_ratio': ratios, 'flag': 'SE_CALIBRATION_WARNING' if flag else None,
             'decision_input': False},
            {'placebo_delta_hat': deltas, 'tau_p': taus, 'placebo_bootstrap_se': ses, 'placebo_sd': sd})


# ---------------------------------------------------------------- lines per denominator
def build_lines(decisions, facts, checkpoint):
    lines, by_pa = [], {}
    for n, (pa_id, info, rows) in enumerate(est.by_pa(decisions(), facts)):
        if n % 1000 == 0:
            checkpoint()
        rows = sorted(rows, key=lambda d: d['decision_index'])
        line = ov.pa_line(rows, info)
        line['pa_id'] = pa_id
        line['pitcher'] = info.get('start_pitcher') or (rows[0].get('pitcher') if rows else None)
        lines.append(line)
        by_pa[pa_id] = rows
    return lines, by_pa


LINE_INVARIANT = ('pa_id', 'status', 'node', 'kind', 'reason', 'game', 'in_population', 'validity_violation',
                  'resolution', 'end', 'reward', 'reward_source', 'pitcher')


def check_b2_lines(lines_b0, lines_b2):
    """A B2 line is the B0 line with recomputed ratios only: the classification, ends, rewards and every step's
    v_c, v_r and q_a are identical (bit for bit), the step count too; rho0 stays 0. Else integrity failure."""
    _require(len(lines_b0) == len(lines_b2), 'B2 lines differ in number from the B0 lines (deterministic integrity)')
    for b0, b2 in zip(lines_b0, lines_b2):
        _require(all(_bits(b0.get(k)) == _bits(b2.get(k)) for k in LINE_INVARIANT),
                 f'B2 classification differs from B0 on PA {b0.get("pa_id")} (deterministic integrity)')
        _require(len(b0['steps']) == len(b2['steps']), f'B2 step count differs on PA {b0.get("pa_id")}')
        for s0, s2 in zip(b0['steps'], b2['steps']):
            _require(_bits(s0[:3]) == _bits(s2[:3]), f'B2 values or q-hat differ on PA {b0.get("pa_id")}')
            _require(all((a == 0) == (b == 0) for a, b in zip(s0[3:], s2[3:])),
                     f'B2 changes a zero ratio on PA {b0.get("pa_id")} (rho0 is denominator-free)')
    return True


def _component_pairs(den, e0, p_lines, point, rewards, p_rewards, p_clusters, sealed):
    """The EXP-OPE-RES-01 per-component loop (R2_0, the lambda / M grids and their selection, the SN row) for one
    denominator's lines; row names carry the denominator. Returns (pairs, inf_ok, sn inverse normalisers or None,
    sn_valid)."""
    pairs, inf_ok = [], True
    n_bar, sn_valid = ov.sn_normalizers(e0)
    u = (1 / n_bar[0], 1 / n_bar[1]) if sn_valid else None
    for comp in rb.R1:
        rw = rewards[comp]
        rows_r20 = [ov.row_of(line, None, None, None if rw is None else rw[i]) for i, line in enumerate(e0)]
        pairs.append(ob.fixed_pair(_row(comp, 'R2_0', den), rows_r20, point))
        base_d = ov.point_deltas(p_lines, None, None, p_rewards[comp])
        for kind, r2, grid in (('shrink', 'R2_1', rb.LAMBDA_GRID), ('switch', 'R2_2', rb.M_GRID)):
            by_value = {v: ov.point_deltas(p_lines, ov.transform(kind, v), None, p_rewards[comp]) for v in grid}
            inf_ok &= bool(np.max(np.abs(by_value[INF] - base_d), initial=0.) <= rb.INTEGRITY_TOL)
            chosen, objectives = ov.select_parameter(kind, by_value, p_clusters)
            name = _row(comp, r2, den)
            sealed['selection'][name] = {'selected': None if chosen is None else _label(chosen),
                                         'objectives': objectives}
            sealed['grid_points'][name] = {_label(v): float(by_value[v].mean()) for v in grid}
            if chosen is None:
                pairs.append(ob.PairInputs(name, deltas=np.full(len(e0), np.nan), lo=np.full(len(e0), np.nan),
                                           hi=np.full(len(e0), np.nan)))
                continue
            f = ov.transform(kind, chosen)
            rows_v = [ov.row_of(line, f, None, None if rw is None else rw[i]) for i, line in enumerate(e0)]
            pairs.append(ob.fixed_pair(name, rows_v, point))
            for v in grid:  # sealed per-grid-value pairs (half-widths stay sealed)
                if v != chosen:
                    fv = ov.transform(kind, v)
                    rows_g = [ov.row_of(line, fv, None, None if rw is None else rw[i]) for i, line in enumerate(e0)]
                    pairs.append(ob.fixed_pair(f'grid:{name}:{_label(v)}', rows_g, point))
        name = _row(comp, 'R2_3', den)
        if sn_valid:
            pairs.append(ob.sn_pair(name, e0, point, None if rw is None else rw, u=np.vstack(u)))
        else:
            pairs.append(ob.PairInputs(name, deltas=np.full(len(e0), np.nan), lo=np.full(len(e0), np.nan),
                                       hi=np.full(len(e0), np.nan)))
    return pairs, inf_ok, u, sn_valid


def _invalid_pairs(names, n):
    return [ob.PairInputs(name, deltas=np.full(n, np.nan), lo=np.full(n, np.nan), hi=np.full(n, np.nan))
            for name in names]


def _resolve_seeds(mode, seeds):
    base = dict(rb.EVALUATOR_SEEDS[mode])
    if mode == rb.MODE_S6:
        _require(seeds is None or dict(seeds) == base, 'the S6 mode runs with the EXP-OPE-RES-01 seeds only (G-INT-EVAL)')
        return base
    if seeds is not None:
        _require(set(seeds) == set(base), 'evaluator seeds: game, pitcher, placebo, pigeonhole')
        return dict(seeds)
    return base


# ---------------------------------------------------------------- the evaluator
def evaluate(decisions, facts, *, mode, surrogates, train_residuals, r1_paths, gint, decisions_b2=None, b2=None,
             rows=None, seeds=None, draws=rb.DRAWS, minimum_games=MIN_GAMES, minimum_pitchers=MIN_PITCHERS,
             minimum_pa_starts=MIN_PA_STARTS, placebos=None, ess_gate=None, checkpoint=None):
    """The blinded A2b evaluation. ``decisions()``: a fresh stream of the (B0) ledger decision rows; ``facts``:
    {pa_id: info} in ledger order; ``surrogates``: ope_surrogate.surrogate_rewards; ``gint``: the record of the
    integrity gates that ran first (required); ``decisions_b2()``: the same rows with pi_b_hat_B2 (REHEARSAL, valid
    B2); ``b2``: the B2 denominator record (validity, A3 weight diagnostics; REHEARSAL); ``rows``: the requested rows
    (default: every row of the mode; the S6 mode refuses any of the 13 B2 / lambda-fixed rows). Returns (public,
    sealed, notes)."""
    tick = checkpoint if checkpoint is not None else (lambda: None)
    _require(mode in (rb.MODE_S6, rb.MODE_REHEARSAL), f'unknown evaluator mode {mode!r}')
    allowed = rb.S6_MODE_ROWS if mode == rb.MODE_S6 else rb.ROWS
    requested = tuple(allowed if rows is None else rows)
    refused = [r for r in requested if r not in allowed]
    _require(not refused, f'the {mode} mode refuses rows {refused} (never computed or written on S6)')
    _require(set(requested) == set(allowed), f'the {mode} mode computes exactly its registered rows')
    if mode == rb.MODE_S6:
        _require(decisions_b2 is None and b2 is None, 'the S6 mode refuses every B2 input (13 rows not computed on S6)')
    else:
        _require(b2 is not None and isinstance(b2.get('validity'), dict), 'the rehearsal needs the B2 denominator record')
        _require((decisions_b2 is not None) == bool(b2['validity'].get('pass')),
                 'B2 rows need the B2 decision stream exactly when B2 is valid')
    _require(gint is not None, 'the integrity record is required (the integrity gates run before the evaluation)')
    seeds = _resolve_seeds(mode, seeds)
    # 1. the registered baseline path (policy_estimator, L1-R) with the mode's game seed
    boot = {'draws': draws, 'seed': seeds['game'], 'invalid_share_max': INVALID_SHARE_MAX,
            'minimum': {'games': minimum_games, 'pa_starts': minimum_pa_starts}}
    result0, rows0 = est.estimate(decisions(), facts, **boot, ess_gate=ess_gate, censoring='l1r')
    tick()
    # 2. variant lines, identity reproduction (bit for bit)
    lines, by_pa = build_lines(decisions, facts, tick)
    for line, row in zip(lines, rows0):
        mine = ov.row_of(line)
        _require(all(_bits(mine.get(k)) == _bits(row.get(k)) for k in mine),
                 f'variant path differs from the registered estimator on PA {line["pa_id"]} (deterministic integrity)')
    e0 = [line for line in lines if line['in_population']]
    _require(e0, 'no E0 PA in the rehearsal')
    e0_rows = [row for line, row in zip(lines, rows0) if line['in_population']]
    point = np.array([in_point_set(row) for row in e0_rows])  # P from the B0 BASELINE paired bounds
    p_index = np.flatnonzero(point)
    p_lines = [line for line, keep in zip(e0, point) if keep]
    games = [line['game'] for line in e0]
    pitchers = [line['pitcher'] for line in e0]
    _require(all(p is not None for p in pitchers), 'every E0 PA needs its first-row pitcher')
    p_clusters = [line['game'] for line in p_lines]
    # 3. rewards per R1 component (observed ends only; denominator-free)
    rewards = {'R1_0': None}
    for comp in ('R1_a', 'R1_b'):
        rewards[comp] = [surrogates[line['pa_id']][comp][0] if line['reward_source'] == 'observed' else line['reward']
                         for line in e0]
    p_rewards = {c: (None if r is None else [r[i] for i in p_index]) for c, r in rewards.items()}
    # 4. per denominator: R2_0, the lambda / M grids and their selection (sealed), the SN row
    sealed = {'selection': {}, 'grid_points': {}, 'gint': gint, 'notes': NOTES, 'mode': mode,
              'seeds': {k: (list(v) if isinstance(v, tuple) else v) for k, v in seeds.items()}}
    pairs, inf_ok, u, sn_valid = _component_pairs('B0', e0, p_lines, point, rewards, p_rewards, p_clusters, sealed)
    lines_b2 = e0_b2 = u_b2 = None
    sn_valid_b2 = None
    if mode == rb.MODE_REHEARSAL:
        rw = rewards['R1_b']  # the selection-free lambda = 100 row (B0)
        f100 = ov.transform('shrink', rb.LAMBDA_FIXED_VALUE)
        rows_fixed = [ov.row_of(line, f100, None, rw[i]) for i, line in enumerate(e0)]
        pairs.append(ob.fixed_pair(rb.LAMBDA_FIXED_ROW, rows_fixed, point))
        if b2['validity']['pass']:
            all_b2, _ = build_lines(decisions_b2, facts, tick)
            check_b2_lines(lines, all_b2)
            lines_b2 = all_b2
            e0_b2 = [line for line in lines_b2 if line['in_population']]
            p_lines_b2 = [line for line, keep in zip(e0_b2, point) if keep]
            b2_pairs, inf_b2, u_b2, sn_valid_b2 = _component_pairs('B2', e0_b2, p_lines_b2, point, rewards, p_rewards,
                                                                   p_clusters, sealed)
            pairs.extend(b2_pairs)
            inf_ok &= inf_b2
        else:
            pairs.extend(_invalid_pairs(rb.B2_ROWS, len(e0)))
    _require(inf_ok, 'lambda = inf / M = inf variants do not reproduce their R2_0 row (deterministic integrity)')
    sealed['p_nonpoint'] = {pi.name: pi.p_nonpoint for pi in pairs}
    sealed['p_counts'] = {'p': int(point.sum()), 'p_known_end': int(sum(ov.is_point(l) for l in p_lines)),
                          'p_shared_unknown_end': int(sum(not ov.is_point(l) for l in p_lines))}
    # 5. bootstraps (paired, same replicates for every row)
    resamplers = {'game': ob.Resampler(games, unit='game', draws=draws, seed=seeds['game'],
                                       minimum_clusters=minimum_games, minimum_pa_starts=minimum_pa_starts),
                  'pitcher': ob.Resampler(pitchers, unit='pitcher', draws=draws, seed=seeds['pitcher'],
                                          minimum_clusters=minimum_pitchers, minimum_pa_starts=minimum_pa_starts)}
    eligible_p = int(point.sum()) >= minimum_pa_starts
    summaries = {}
    for b, resampler in resamplers.items():
        stats, valid = ob.run_pairs(resampler, pairs, point, rb.BASELINE, checkpoint=tick)
        if not eligible_p:  # the minimum-P rule: P statistics invalid for every row alike
            valid = {name: {**v, 'p': np.zeros_like(v['p']), 'decomposition': np.zeros_like(v['decomposition'])}
                     for name, v in valid.items()}
        summaries[b] = ob.summarize(stats, valid, resampler)
    # 6. public and sealed per row
    public = {'experiment': rb.EXPERIMENT_ID, 'stage': 'A2b-S3', 'r3_status': rb.R3_STATUS_DECIDED,
              'pairs': {}, 'S5': {}, 'R5': {}, 'r1_variance': {}, 'integrity': {}, 'validity': {}}
    held = {'pairs': {}, 'S5': {}, 'R5': {'e': {}, 'f': {}}}  # public-form items of the rows sealed until A2b-UB
    target = lambda row: held if row in rb.SEALED_UNTIL_UNBLIND_ROWS else public
    sealed['pairs'] = {}
    for row in requested:
        entry = {'H_point': {}, 'SE': {}, 'endpoint_half_width': {'lower': {}, 'upper': {}}, 'valid': {},
                 'invalid_replicates': {}}
        s5, sealed_row = {}, {}
        for b in BOOTS:
            summary = summaries[b]
            stats = summary.get(row) if summary.get('eligible') else None
            point_ci = None if stats is None else stats['mean_p']
            ok = bool(point_ci is not None and point_ci['valid'])
            entry['H_point'][b] = point_ci['half_width'] if ok else None
            entry['SE'][b] = point_ci['se'] if ok else None
            entry['valid'][b] = ok
            entry['invalid_replicates'][b] = None if point_ci is None else point_ci['invalid_replicates']
            for side in ('lower', 'upper'):
                ci = None if stats is None else stats[f'{side}_endpoint']
                entry['endpoint_half_width'][side][b] = ci['half_width'] if ci is not None and ci['valid'] else None
            shift = None if stats is None else stats['shift_p']
            if row == rb.BASELINE:
                s5[b] = 'BASELINE'
            elif shift is None or not shift['valid']:
                s5[b] = 'INVALID'
            elif shift['half_width'] == 0 or shift['se'] == 0:
                s5[b] = 'INCONCLUSIVE'  # a degenerate (zero-width) shift interval is never a pass
            else:
                s5[b] = 'PASS' if shift['ci95'][0] <= 0 <= shift['ci95'][1] else 'FAIL'
            sealed_row[b] = None if stats is None else {
                'mean_p_ci95': stats['mean_p']['ci95'], 'shift_p_ci95': stats['shift_p']['ci95'],
                'lower_endpoint_ci95': stats['lower_endpoint']['ci95'], 'upper_endpoint_ci95': stats['upper_endpoint']['ci95']}
        target(row)['pairs'][row] = entry
        target(row)['S5'][row] = s5
        sealed['pairs'][row] = {'bootstrap': sealed_row}
    # 7. full-sample centers (sealed only)
    u_of = {'B0': u, 'B2': u_b2}
    for pair_inputs in pairs:
        if pair_inputs.sn is not None:
            den = 'B2' if pair_inputs.name.endswith('xB2') else 'B0'
            if u_of[den] is None:
                continue
            deltas, lo, hi, _ = ob.sn_values(pair_inputs, np.vstack(u_of[den]))
        else:
            deltas, lo, hi = pair_inputs.deltas, pair_inputs.lo, pair_inputs.hi
        on_p = deltas[point]
        center = {'mean_p': float(on_p.mean()) if len(on_p) and np.isfinite(on_p).all() else None,
                  'l1r_bounds': [float(np.mean(lo)), float(np.mean(hi))]}
        sealed['pairs'].setdefault(pair_inputs.name, {})['center'] = center
    base_center = sealed['pairs'][rb.BASELINE]['center']['mean_p']
    for row in requested:
        own = sealed['pairs'][row].get('center', {}).get('mean_p')
        if row != rb.BASELINE:
            sealed['pairs'][row]['shift_center'] = None if own is None or base_center is None else own - base_center
    sealed['grid_bootstrap'] = {b: {name: s for name, s in summaries[b].items() if str(name).startswith('grid:')}
                                for b in BOOTS if summaries[b].get('eligible')}
    # 8. R5 diagnostics (reward-free or widths only) and R1 variance; B0 sections exactly as EXP-OPE-RES-01
    table = od.DecisionTable.from_ledger((r for rows_ in by_pa.values() for r in rows_), {l['pa_id'] for l in e0})
    grids = od.standard_grids(rb.LAMBDA_GRID, rb.M_GRID)
    public['R5']['a'] = od.agreement(table)
    public['R5']['b'] = od.ess_by_variant(e0, games, grids, sn_inverse=u)
    public['R5']['c'] = {'weights': od.weight_summary(e0), **od.grid_weight_diagnostics(e0, rb.LAMBDA_GRID, rb.M_GRID),
                         'logging_mass': od.logging_mass(table)}
    public['R5']['d'] = od.positivity_censoring(result0, rows0)
    s_share = (result0.get('censoring_rule') or {}).get('worst_case_residual_share')
    public['R5']['e'], public['R5']['f'] = {}, {}
    for row in requested:
        r5 = target(row)['R5']
        r5['e'][row], r5['f'][row] = {}, {}
        for b in BOOTS:
            summary = summaries[b]
            stats = summary.get(row) if summary.get('eligible') else None
            se_p = None if stats is None or not stats['mean_p']['valid'] else stats['mean_p']['se']
            se_lo = None if stats is None or not stats['lower_endpoint']['valid'] else stats['lower_endpoint']['se']
            r5['e'][row][b] = od.mde(se_p, se_lo, s_share)
            dec = None if stats is None else stats['decomposition']
            r5['f'][row][b] = ({k: dec[k] for k in ('var_point', 'var_censored', 'var_lower_endpoint')}
                                         | {'cov_point_censored': dec['cov_point_censored']}) if dec and dec['valid'] else {
                'var_point': None, 'var_censored': None, 'var_lower_endpoint': None, 'cov_point_censored': None}
    public['R5']['g'] = od.self_consistency(e0, table, resamplers['game'])
    # placebo on the B0 baseline (reward enters only through the same ends as the baseline)
    step_rows = {}
    for i, entry in enumerate(table.entries):
        step_rows.setdefault(entry['pa'], {})[entry['index']] = i
    p_steps, placebo_lines, placebo_index = [], [], []
    for line, index in zip(p_lines, p_index):
        if not ov.is_point(line):  # a shared unknown end: a placebo tilt breaks the slope cancellation
            continue
        rows_of = step_rows.get(line['pa_id'], {})
        p_steps.append([rows_of[t] for t in range(len(line['steps']))])
        placebo_lines.append(line)
        placebo_index.append(index)
    placebo_public, placebo_sealed = placebo_check(
        p_steps, table, {b: (resamplers[b] if eligible_p else None, np.array(placebo_index, dtype=np.int64))
                         for b in BOOTS},
        rewards=[line['reward'] for line in placebo_lines], stream=rb.placebo_stream(seeds['placebo']), checkpoint=tick,
        **({} if placebos is None else {'placebos': placebos}))
    placebo_sealed['population'] = 'P PAs with a known end (complete or shared point)'
    public['R5']['h'] = {'sd_ratio': placebo_public['sd_ratio'], 'placebos': placebo_public['placebos'],
                         'flag': placebo_public['flag'], 'decision_input': False}
    sealed['placebo'] = placebo_sealed
    public['R5']['i'] = {'paths': r1_paths, 'train_residuals': train_residuals}
    last_q = []
    for line in p_lines:
        last_q.append(line['reward'] - line['terminal_residual'] if line['terminal_residual'] is not None else None)
    obs = [line['reward_source'] == 'observed' for line in p_lines]
    comp_rewards = {'R1_0': [line['reward'] for line in p_lines]}
    for comp in ('R1_a', 'R1_b'):
        comp_rewards[comp] = p_rewards[comp]
    public['r1_variance'] = od.r1_variance(p_lines, last_q, comp_rewards)
    observed = lambda values: np.array([v for v, o in zip(values, obs) if o], dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        sealed['corr_r_r_tilde'] = {comp: float(np.corrcoef(observed(comp_rewards['R1_0']),
                                                             observed(comp_rewards[comp]))[0, 1])
                                    if sum(obs) > 2 else None for comp in ('R1_a', 'R1_b')}
    if seeds['pigeonhole'] is not None:
        main_pairs = [pi for pi in pairs if pi.name in rb.B0_ROWS and (pi.sn is None or u is not None)]
        sealed['pigeonhole'] = ob.pigeonhole(games, pitchers, main_pairs, point, draws=draws, seed=seeds['pigeonhole'],
                                             checkpoint=tick)  # descriptive, sealed only
    else:
        sealed['pigeonhole'] = {'status': 'NOT_RUN', 'reason': 'no registered A2b pigeonhole seed role (design 9.4)'}
    public['integrity'] = {'G_INT': 'PASS', 'variant_identity': 'PASS', 'inf_variants_reproduce_baseline': 'PASS',
                           'sn_normalizer_full_sample_valid': bool(sn_valid)}
    public['validity'] = {'point_set_meets_minimum': bool(eligible_p),
                          'game_bootstrap_eligible': bool(summaries['game'].get('eligible')),
                          'pitcher_bootstrap_eligible': bool(summaries['pitcher'].get('eligible'))}
    sealed['baseline_result'] = {k: result0.get(k) for k in ('layers', 'bootstrap', 'censoring_rule', 'ess_all_e0', 'ess')}
    sealed['counts'] = {'e0': len(e0), 'p': int(point.sum())}
    # 9. the B2 denominator (public items 10-11; descriptive under D192 2 3)
    if mode == rb.MODE_REHEARSAL:
        public['B2'] = b2_public(b2, e0_b2, by_pa_b2=None if lines_b2 is None else _by_pa(decisions_b2, facts),
                                 e0_ids={l['pa_id'] for l in e0}, games=games, resampler=resamplers['game'],
                                 grids=grids, u_b2=u_b2, sn_valid_b2=sn_valid_b2)
        public['flags'] = {'B2_PRECISION_PASS_DIAGNOSTIC': b2_precision_pass(public)}
        public['counts'] = {'rows': len(public['pairs'])}  # the 24 public rows (the lambda-fixed row is sealed)
        sealed['lambda_fixed_row_public'] = lambda_fixed_row_public(held, public)
    require_descriptive(public, sealed, requested, mode)
    return public, sealed, NOTES


LAMBDA_FIXED_COMPARED_ROW = 'R1_bxR2_1xB0'  # the selected-lambda row the fixed row would be read against


def lambda_fixed_row_public(held, public):
    """The lambda = 100 row's public-form items (pairs, S5, R5 e / f), sealed until A2b-UB and opened with
    ``unblinded.json`` (G-CODE MAJOR-1). ``equals_R1_bxR2_1xB0``: whether they are bit-identical to the public
    R1_b x R2_1 row (i.e. whether lambda = 100 was selected) - descriptive, written only here, never a gate or
    decision input."""
    row = rb.LAMBDA_FIXED_ROW
    _require(set(held['pairs']) == {row} and set(held['S5']) == {row}, 'the sealed-until-unblind rows are the lambda-fixed '
                                                                        'row only (stage integrity)')
    items = lambda pairs, s5, r5, name: [_bits(pairs[name]), _bits(s5[name]), _bits(r5['e'][name]), _bits(r5['f'][name])]
    same = (items(held['pairs'], held['S5'], held['R5'], row)
            == items(public['pairs'], public['S5'], public['R5'], LAMBDA_FIXED_COMPARED_ROW))
    return {'row': row, 'lambda': rb.LAMBDA_FIXED_VALUE, 'pairs': held['pairs'][row], 'S5': held['S5'][row],
            'R5': {'e': held['R5']['e'][row], 'f': held['R5']['f'][row]},
            'equals_R1_bxR2_1xB0': bool(same), 'decision_input': False,
            'disclosure': 'sealed until A2b-UB (G-CODE MAJOR-1); opened in unblinded.json with the selection'}


def _by_pa(decisions, facts):
    return {pa_id: sorted(rows, key=lambda d: d['decision_index']) for pa_id, _, rows in est.by_pa(decisions(), facts)}


B2_A3_KEYS = ('ess', 'rho_quantiles', 'w_T_quantiles', 'rho_gt_large_share', 'w_T_gt_large_share')


def b2_public(b2, e0_b2, *, by_pa_b2, e0_ids, games, resampler, grids, u_b2, sn_valid_b2):
    """Items 10 and 11: the B2 validity counts, the A3 weight diagnostics (E5, ESS, rho / W quantiles and tails,
    KL(B2 || B0), weight shift D) and, when B2 is valid, R5 b, c and g under the B2 denominator."""
    validity = b2['validity']
    out = {'validity': {k: validity[k] for k in ('pass', 'decisions_checked', 'zero_mass_with_positive_pi',
                                                 'nonfinite_rho', 'support_differs_from_b0')}}
    weights = b2.get('weights')
    if weights is not None:
        out['E5'] = {'pass': bool(weights['E5']['pass']), 'checks': dict(weights['E5']['checks']),
                     'thresholds': dict(weights['E5']['thresholds'])}
        for key in B2_A3_KEYS:
            out[key] = {p: weights['policies'][p][key] for p in ('candidate', 'reference')}
        out['large_weight_threshold'] = weights.get('large_weight_threshold')
        kl = weights.get('kl_new_vs_b0') or {}
        out['kl_b2_vs_b0'] = {'mean': None if kl.get('mean') is None or not math.isfinite(kl['mean']) else kl['mean'],
                              'finite_only_mean': kl.get('finite_only_mean'), 'decisions': kl.get('decisions'),
                              'infinite_decisions': kl.get('infinite_decisions')}
        shift = weights.get('weight_shift_D') or {}
        out['weight_shift_D'] = {k: shift.get(k) for k in ('candidate', 'reference', 'paired', 'paired_below_mei')}
        ratio = b2.get('ess_ratio_vs_b0')
        if ratio is not None:
            out['ess_ratio_vs_b0'] = ratio
    if e0_b2 is not None:
        table_b2 = od.DecisionTable.from_ledger((r for rows in by_pa_b2.values() for r in rows), e0_ids)
        out['R5_b'] = od.ess_by_variant(e0_b2, games, grids, sn_inverse=u_b2)
        out['R5_c'] = {'weights': od.weight_summary(e0_b2)}
        out['R5_g'] = od.self_consistency(e0_b2, table_b2, resampler)
        out['sn_normalizer_full_sample_valid'] = bool(sn_valid_b2)
    return out


def b2_precision_pass(public):
    """B2_PRECISION_PASS_DIAGNOSTIC: a (descriptive) B2 row passes (ii) under both bootstraps."""
    base = public['pairs'][rb.BASELINE]
    h_base = {b: base['H_point'][b] if base['valid'][b] else None for b in BOOTS}
    for row in rb.B2_ROWS:
        entry = public['pairs'].get(row)
        if entry is None:
            continue
        h = {b: entry['H_point'][b] if entry['valid'][b] else None for b in BOOTS}
        if rb.condition_ii(h, h_base)['pass']:
            return True
    return False


REQUIRED_PUBLIC = (('R5', 'a'), ('R5', 'b'), ('R5', 'c'), ('R5', 'd'), ('R5', 'e'), ('R5', 'f'), ('R5', 'g'),
                   ('R5', 'h'), ('R5', 'i'), ('r1_variance',), ('integrity',), ('validity',))
REQUIRED_SEALED = ('selection', 'grid_points', 'grid_bootstrap', 'pairs', 'placebo', 'corr_r_r_tilde', 'pigeonhole',
                   'baseline_result', 'p_nonpoint', 'p_counts', 'gint', 'notes', 'seeds', 'mode')


def require_descriptive(public, sealed, rows, mode):
    """Every descriptive output the design requires (a missing one is a stage integrity failure, never silent)."""
    for path in REQUIRED_PUBLIC + ((('B2',), ('flags',)) if mode == rb.MODE_REHEARSAL else ()):
        node = public
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        _require(node is not None, f'descriptive output missing: {".".join(path)} (stage integrity)')
    for key in REQUIRED_SEALED:
        _require(sealed.get(key) is not None, f'sealed descriptive output missing: {key} (stage integrity)')
    shown = [r for r in rows if r not in rb.SEALED_UNTIL_UNBLIND_ROWS]
    _require(set(public['pairs']) == set(shown) and set(public['S5']) == set(shown)
             and set(public['R5']['e']) == set(shown) and set(public['R5']['f']) == set(shown),
             'public rows differ from the mode (the lambda-fixed row is never public before A2b-UB)')
    if any(r in rb.SEALED_UNTIL_UNBLIND_ROWS for r in rows):
        held = sealed.get('lambda_fixed_row_public')
        _require(isinstance(held, dict) and held.get('row') == rb.LAMBDA_FIXED_ROW
                 and all(held.get(k) is not None for k in ('pairs', 'S5', 'R5')),
                 'sealed lambda-fixed row items missing (stage integrity)')
    sn_ok = {'B0': bool(public['integrity'].get('sn_normalizer_full_sample_valid')),
             'B2': bool((public.get('B2') or {}).get('sn_normalizer_full_sample_valid'))}
    b2_valid = bool(((public.get('B2') or {}).get('validity') or {}).get('pass'))
    for row in rows:
        record = sealed['pairs'].get(row) or {}
        den = rb.denominator_of(row)
        _require('bootstrap' in record, f'sealed bootstrap record missing: {row} (stage integrity)')
        _require('center' in record or (row.endswith('R2_3x' + den) and not sn_ok[den]),
                 f'sealed center missing: {row} (stage integrity)')
        _require(row == rb.BASELINE or 'shift_center' in record, f'sealed paired shift missing: {row} (stage integrity)')
        if den == 'B2' and not b2_valid:  # every B2 row is a public row
            _require(not any(public['pairs'][row]['valid'].values()), f'an invalid B2 denominator gave a valid {row}')
    return True


def seal(public, sealed):
    """Bytes of centers.sealed.json and the public document carrying only its sha256 (strict A2b allowlist)."""
    sealed_bytes = canonical_bytes(_jsonable(sealed))
    public = {**public, 'sealed_sha256': sha256_bytes(sealed_bytes)}
    rb.validate_public(public)
    return canonical_bytes(public), sealed_bytes


# ---------------------------------------------------------------- G-INT-EVAL (A2b-S3, before the 25 rows)
PUBLIC_NOT_COMPARABLE = ('experiment', 'stage', 'r3_status', 'sealed_sha256')  # differ by design (IDs, seal)


def a2_view(public, sealed):
    """The S6-mode output in the EXP-OPE-RES-01 record layout: the public part (row names -> A2 pair names, minus
    the IDs) and the unblinded part (selection; per pair bootstrap CIs, center and shift center), JSON-normalised
    like the A2 sealed file (nonfinite -> null)."""
    rename = lambda row: rb.a2_pair(row) or row
    pub = {k: v for k, v in public.items() if k not in PUBLIC_NOT_COMPARABLE and k not in ('B2', 'flags', 'counts')}
    pub = _jsonable(pub)
    pub['pairs'] = {rename(r): v for r, v in pub['pairs'].items()}
    pub['S5'] = {rename(r): v for r, v in pub['S5'].items()}
    for section in ('e', 'f'):
        pub['R5'][section] = {rename(r): v for r, v in pub['R5'][section].items()}
    unblinded = {'selection': {rename(k): v for k, v in sealed['selection'].items() if k.endswith('xB0')},
                 'pairs': {}}
    for row in rb.S6_MODE_ROWS:
        record = sealed['pairs'][row]
        unblinded['pairs'][rename(row)] = {k: record[k] for k in ('bootstrap', 'center', 'shift_center') if k in record}
    return {'public': pub, 'unblinded': _jsonable(unblinded)}


def a2_record(results):
    """The comparable EXP-OPE-RES-01 record (results file): S3_public minus its IDs and UB_unblinded's selection and
    pairs (the decision-record sha is not an evaluator output)."""
    public = {k: v for k, v in results['S3_public'].items() if k not in PUBLIC_NOT_COMPARABLE}
    unblinded = {k: results['UB_unblinded'][k] for k in ('selection', 'pairs')}
    return {'public': public, 'unblinded': unblinded}


def gint_eval_record(public_s6, sealed_s6, results, *, reference):
    """G-INT-EVAL integrity record (public item 13): the S6-mode output against the unblinded A2 record, absolute
    1e-12 for floats, equality for counts, strings and the selected lambda / M (inf by equality). Only pass/fail,
    max |delta|, the tolerance and the key lists are written (no center or value)."""
    _require(set(public_s6['pairs']) == set(rb.S6_MODE_ROWS), 'G-INT-EVAL takes the 12-row S6 mode output only')
    compared, not_comparable, mismatched, max_delta = rb.compare_documents(a2_view(public_s6, sealed_s6),
                                                                           a2_record(results))
    record = {'experiment': rb.EXPERIMENT_ID, 'check': 'G-INT-EVAL', 'result': 'FAIL' if mismatched else 'PASS',
              'max_abs_delta': float(max_delta), 'tolerance': dict(rb._TOLERANCE), 'compared_keys': compared,
              'not_comparable_keys': not_comparable, 'mismatched_keys': mismatched, 'rows': list(rb.S6_MODE_ROWS),
              'refused_rows': list(rb.S6_REFUSED_ROWS), 'reference': dict(reference)}
    rb.validate_integrity_record(record, 'gint_eval')
    return record


def require_gint_pass(record, check):
    if record['result'] != 'PASS':
        raise IntegrityError(f'{check} failed (deterministic integrity): {len(record["mismatched_keys"])} mismatched '
                             f'keys, max |delta| {record["max_abs_delta"]}')
    return record
