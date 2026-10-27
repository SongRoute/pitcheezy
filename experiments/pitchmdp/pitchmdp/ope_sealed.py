"""EXP-OPE-RES-01 A2-S3 sealed evaluator, the G-INT gate and the strict public file (design section 3).

``gint_gate`` runs FIRST, on the untouched registered path (``policy_estimator.estimate`` on the sealed
ledger and the recomputed facts), before any surrogate reward or variant is computed: every value the
design lists (L1-R bounds, both endpoint CIs, L2 CI and mean, the worst-case share s, the resolution
counts, both policies' ESS) must reproduce the recorded S6 values within 1e-12 (exact for counts), both
the values quoted in the registration and the pinned S6-dr-a5 record; a value absent from a reference is
recorded as not comparable. ``rehearsal_evaluate`` then computes EVERY signed real-data quantity
internally (Delta centers and CIs per pair and grid value, paired shifts and their CIs (S-5), lambda/M
objectives and selections, per-grid half-widths, placebo Delta_hat, corr(r, r~), pigeonhole); they go
to the sealed document only. The public document holds exactly the (key, type) pairs of the strict
registry allowlist (``ope_redesign_registry.validate_public``); everything else is sealed. ``unblind``
opens the sealed file only after a committed decision record.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from . import ope_bootstrap as ob
from . import ope_diagnostics as od
from . import ope_variants as ov
from . import policy_estimator as est
from .ope_redesign_registry import (BASELINE, DRAWS, GAME_BOOTSTRAP_SEED, INF, INTEGRITY_TOL, INVALID_SHARE_MAX,
                                    LAMBDA_GRID, M_GRID, MIN_GAMES, MIN_PA_STARTS, MIN_PITCHERS, PAIRS, R1, ROLE_VALUES,
                                    in_point_set, pair_id)
from .ope_redesign_registry import validate_public  # noqa: F401  (the one strict allowlist, re-exported)
from .policy_artifacts import IntegrityError, _require

BOOTS = ('game', 'pitcher')
NOTES = {'R5_f_normalization': 'the point part of R5 f uses the common denominator N (all E0 PAs); H_point is a mean '
                               'over P, a different normalisation',
         'coverage': '95% bounds are nominal per pair and cell, not simultaneous over the selected pair',
         'point_set': 'P = E0 PAs whose baseline paired delta bounds are equal; a pair under which a P PA is not a '
                      'point has no P mean (its P statistics are invalid)'}


def canonical_bytes(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + '\n').encode('utf-8')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- G-INT
GINT_KEYS = ('l1r_delta_bounds', 'l1r_lower_endpoint_ci95', 'l1r_upper_endpoint_ci95', 'l2_ci95', 'l2_delta_mean',
             'worst_case_residual_share_s', 'resolutions', 'ess_all_e0_candidate', 'ess_all_e0_reference')


def gint_values(result):
    """The design's G-INT quantities of one estimate result (policy_estimator.estimate, L1-R)."""
    rule = result.get('censoring_rule') or {}
    boot = result.get('bootstrap') or {}
    l2 = (result.get('layers') or {}).get('L2_complete_conditional') or {}
    ess = result.get('ess_all_e0') or {}
    return {'l1r_delta_bounds': ((result.get('layers') or {}).get('L1_start_population') or {}).get('delta_bounds'),
            'l1r_lower_endpoint_ci95': boot.get('L1_lower_endpoint_ci95'),
            'l1r_upper_endpoint_ci95': boot.get('L1_upper_endpoint_ci95'),
            'l2_ci95': boot.get('L2_conditional_ci95'), 'l2_delta_mean': l2.get('delta_mean'),
            'worst_case_residual_share_s': rule.get('worst_case_residual_share'), 'resolutions': rule.get('resolutions'),
            'ess_all_e0_candidate': ess.get('candidate'), 'ess_all_e0_reference': ess.get('reference')}


def _same(value, expected):
    if isinstance(expected, dict) and all(isinstance(v, int) and not isinstance(v, bool) for v in expected.values()):
        return value == expected  # counts: exact
    if isinstance(expected, dict):
        return isinstance(value, dict) and set(value) >= set(expected) and all(
            value[k] is not None and abs(value[k] - expected[k]) <= INTEGRITY_TOL for k in expected)
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(
            abs(a - b) <= INTEGRITY_TOL for a, b in zip(value, expected))
    return value is not None and abs(value - expected) <= INTEGRITY_TOL


def gint_compare(result, reference):
    """Mismatching keys (1e-12; counts exact) of the recomputed baseline against one reference."""
    got = gint_values(result)
    return [key for key, expected in reference.items() if expected is not None and not _same(got.get(key), expected)]


def gint_record(result, references):
    """Per reference: compared keys, mismatches and the listed keys it cannot compare (absent)."""
    out = {}
    for name, reference in references.items():
        present = {k: v for k, v in (reference or {}).items() if k in GINT_KEYS and v is not None}
        out[name] = {'compared': sorted(present), 'not_comparable': sorted(set(GINT_KEYS) - set(present)),
                     'mismatches': gint_compare(result, present)}
    return out


def gint_gate(decisions, facts, *, references, draws=DRAWS, minimum_games=MIN_GAMES, minimum_pa_starts=MIN_PA_STARTS,
              ess_gate=None):
    """G-INT on the untouched registered path, before any surrogate or variant computation. Raises
    IntegrityError on any mismatch; returns (result0, rows0, record)."""
    boot = {'draws': draws, 'seed': GAME_BOOTSTRAP_SEED, 'invalid_share_max': INVALID_SHARE_MAX,
            'minimum': {'games': minimum_games, 'pa_starts': minimum_pa_starts}}
    result0, rows0 = est.estimate(decisions(), facts, **boot, ess_gate=ess_gate, censoring='l1r')
    record = gint_record(result0, references)
    failures = {name: entry['mismatches'] for name, entry in record.items() if entry['mismatches']}
    if failures:
        raise IntegrityError(f'G-INT failed (deterministic integrity): {failures}; no new arm is computed')
    covered = set().union(*(set(entry['compared']) for entry in record.values())) if record else set()
    record['listed_keys_never_compared'] = sorted(set(GINT_KEYS) - covered)
    return result0, rows0, record


def registered_gint_reference(config):
    """The recorded S6-v3 values the registration quotes (S6-dr-a5 reproduces S6-dr-a4 exactly)."""
    s6 = config['mde_from_records']['S6_rehearsal_150_dev_games']
    return {'l1r_delta_bounds': s6['l1r_delta_bounds']['value'],
            'l1r_lower_endpoint_ci95': s6['l1r_lower_endpoint_ci95']['value'], 'l2_ci95': s6['l2_ci95']['value'],
            'l2_delta_mean': config['mde_from_records']['reference_magnitudes']['exposed_point_estimates']['S6_l2_delta_mean'],
            'worst_case_residual_share_s': s6['worst_case_residual_share_s']['value'],
            'resolutions': s6['resolutions']['value'], 'ess_all_e0_candidate': s6['ess_all_e0_candidate']['value']}


def recorded_gint_reference(dr_record):
    """Every listed G-INT value of the pinned S6-dr-a5 record (dr.json = the estimate result it sealed)."""
    return gint_values(dr_record)


def _bits(value):
    return json.dumps(value, sort_keys=True, default=lambda v: float(v)) if value is not None else None


def rehearsal_evaluate(decisions, facts, *, baseline, surrogates, train_residuals, r1_paths, r3_status,
                       draws=DRAWS, minimum_games=MIN_GAMES, minimum_pitchers=MIN_PITCHERS,
                       minimum_pa_starts=MIN_PA_STARTS, placebos=None, gint=None, checkpoint=None):
    """The blinded rehearsal after the G-INT gate. ``baseline`` = (result0, rows0) from ``gint_gate``;
    ``decisions()`` returns a fresh stream of the sealed ledger's decision rows; ``facts``: {pa_id:
    info} in ledger order (start_pitcher for the pitcher bootstrap); ``surrogates``:
    ope_surrogate.surrogate_rewards output (computed after the gate); ``checkpoint``: the runner's memory
    and hang guard, called inside every long loop. Returns (public, sealed, notes)."""
    tick = checkpoint if checkpoint is not None else (lambda: None)
    _require(gint is not None, 'the G-INT record is required (G-INT runs before the rehearsal)')
    result0, rows0 = baseline
    # 2. variant lines, identity reproduction (bit for bit)
    lines, by_pa = [], {}
    for n, (pa_id, info, rows) in enumerate(est.by_pa(decisions(), facts)):
        if n % 1000 == 0:
            tick()
        rows = sorted(rows, key=lambda d: d['decision_index'])
        line = ov.pa_line(rows, info)
        line['pa_id'] = pa_id
        line['pitcher'] = info.get('start_pitcher') or (rows[0].get('pitcher') if rows else None)
        lines.append(line)
        by_pa[pa_id] = rows
    for line, row in zip(lines, rows0):
        mine = ov.row_of(line)
        _require(all(_bits(mine.get(k)) == _bits(row.get(k)) for k in mine),
                 f'variant path differs from the registered estimator on PA {line["pa_id"]} (deterministic integrity)')
    e0 = [line for line in lines if line['in_population']]
    _require(e0, 'no E0 PA in the rehearsal')
    e0_rows = [row for line, row in zip(lines, rows0) if line['in_population']]
    point = np.array([in_point_set(row) for row in e0_rows])  # P from the BASELINE paired bounds (registry)
    p_lines = [line for line, keep in zip(e0, point) if keep]
    games = [line['game'] for line in e0]
    pitchers = [line['pitcher'] for line in e0]
    _require(all(p is not None for p in pitchers), 'every E0 PA needs its first-row pitcher')
    p_clusters = [line['game'] for line in p_lines]
    # 3. rewards per R1 component (observed ends only)
    rewards = {'R1_0': None}
    for comp in ('R1_a', 'R1_b'):
        rewards[comp] = [surrogates[line['pa_id']][comp][0] if line['reward_source'] == 'observed' else line['reward']
                         for line in e0]
    p_rewards = {c: (None if r is None else [r[i] for i in np.flatnonzero(point)]) for c, r in rewards.items()}
    # 4. per R1 component: R2_0, the lambda/M grids and their selection (sealed), the SN pair
    sealed = {'selection': {}, 'grid_points': {}, 'gint': gint, 'notes': NOTES}
    pairs, inf_ok = [], True
    n_bar, sn_valid = ov.sn_normalizers(e0)
    u = (1 / n_bar[0], 1 / n_bar[1]) if sn_valid else None
    for comp in R1:
        tick()
        rw = rewards[comp]
        rows_r20 = [ov.row_of(line, None, None, None if rw is None else rw[i]) for i, line in enumerate(e0)]
        pairs.append(ob.fixed_pair(pair_id(comp, 'R2_0'), rows_r20, point))
        base_d = ov.point_deltas(p_lines, None, None, p_rewards[comp])
        for kind, r2, grid in (('shrink', 'R2_1', LAMBDA_GRID), ('switch', 'R2_2', M_GRID)):
            by_value = {v: ov.point_deltas(p_lines, ov.transform(kind, v), None, p_rewards[comp]) for v in grid}
            inf_ok &= bool(np.max(np.abs(by_value[INF] - base_d), initial=0.) <= INTEGRITY_TOL)
            chosen, objectives = ov.select_parameter(kind, by_value, p_clusters)
            name = pair_id(comp, r2)
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
        name = pair_id(comp, 'R2_3')
        if sn_valid:
            pairs.append(ob.sn_pair(name, e0, point, None if rw is None else rw, u=np.vstack(u)))
        else:
            pairs.append(ob.PairInputs(name, deltas=np.full(len(e0), np.nan), lo=np.full(len(e0), np.nan),
                                       hi=np.full(len(e0), np.nan)))
    _require(inf_ok, 'lambda = inf / M = inf variants do not reproduce the baseline (deterministic integrity)')
    sealed['p_nonpoint'] = {pi.name: pi.p_nonpoint for pi in pairs}
    sealed['p_counts'] = {'p': int(point.sum()), 'p_known_end': int(sum(ov.is_point(l) for l in p_lines)),
                          'p_shared_unknown_end': int(sum(not ov.is_point(l) for l in p_lines))}
    # 5. bootstraps (paired, same replicates for every pair)
    resamplers = {'game': ob.Resampler(games, unit='game', draws=draws, seed=GAME_BOOTSTRAP_SEED,
                                       minimum_clusters=minimum_games, minimum_pa_starts=minimum_pa_starts),
                  'pitcher': ob.Resampler(pitchers, unit='pitcher', draws=draws, seed=ROLE_VALUES['pitcher_bootstrap'],
                                          minimum_clusters=minimum_pitchers, minimum_pa_starts=minimum_pa_starts)}
    eligible_p = int(point.sum()) >= minimum_pa_starts
    summaries = {}
    for b, resampler in resamplers.items():
        stats, valid = ob.run_pairs(resampler, pairs, point, BASELINE, checkpoint=tick)
        summary = ob.summarize(stats, valid, resampler)
        if not eligible_p:
            summary = {'eligible': False, 'reason': 'P below the registered minimum PA starts', **resampler.record()}
        summaries[b] = summary
    # 6. public and sealed per pair
    public = {'experiment': 'EXP-OPE-RES-01', 'stage': 'A2-S3', 'r3_status': r3_status,
              'pairs': {}, 'S5': {}, 'R5': {}, 'r1_variance': {}, 'integrity': {}, 'validity': {}}
    sealed['pairs'] = {}
    for pair in PAIRS:
        entry = {'H_point': {}, 'SE': {}, 'endpoint_half_width': {'lower': {}, 'upper': {}}, 'valid': {},
                 'invalid_replicates': {}}
        s5, sealed_pair = {}, {}
        for b in BOOTS:
            summary = summaries[b]
            stats = summary.get(pair) if summary.get('eligible') else None
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
            if pair == BASELINE:
                s5[b] = 'BASELINE'
            elif shift is None or not shift['valid']:
                s5[b] = 'INVALID'
            elif shift['half_width'] == 0 or shift['se'] == 0:
                s5[b] = 'INCONCLUSIVE'  # a degenerate (zero-width) shift interval is never a pass (review round 2 #17)
            else:
                s5[b] = 'PASS' if shift['ci95'][0] <= 0 <= shift['ci95'][1] else 'FAIL'
            sealed_pair[b] = None if stats is None else {
                'mean_p_ci95': stats['mean_p']['ci95'], 'shift_p_ci95': stats['shift_p']['ci95'],
                'lower_endpoint_ci95': stats['lower_endpoint']['ci95'], 'upper_endpoint_ci95': stats['upper_endpoint']['ci95']}
        public['pairs'][pair] = entry
        public['S5'][pair] = s5
        sealed['pairs'][pair] = {'bootstrap': sealed_pair}
    # 7. full-sample centers (sealed only)
    for pair_inputs in pairs:
        if pair_inputs.sn is not None:
            if u is None:
                continue
            deltas, lo, hi, _ = ob.sn_values(pair_inputs, np.vstack(u))
        else:
            deltas, lo, hi = pair_inputs.deltas, pair_inputs.lo, pair_inputs.hi
        on_p = deltas[point]
        center = {'mean_p': float(on_p.mean()) if len(on_p) and np.isfinite(on_p).all() else None,  # no P mean: None
                  'l1r_bounds': [float(np.mean(lo)), float(np.mean(hi))]}
        sealed['pairs'].setdefault(pair_inputs.name, {})['center'] = center
    base_center = sealed['pairs'][BASELINE]['center']['mean_p']
    for pair in PAIRS:
        own = sealed['pairs'][pair].get('center', {}).get('mean_p')
        if pair != BASELINE:
            sealed['pairs'][pair]['shift_center'] = None if own is None or base_center is None else own - base_center
    sealed['grid_bootstrap'] = {b: {name: s for name, s in summaries[b].items() if str(name).startswith('grid:')}
                                for b in BOOTS if summaries[b].get('eligible')}
    # 8. R5 diagnostics (reward-free or widths only) and R1 variance
    table = od.DecisionTable.from_ledger((r for rows in by_pa.values() for r in rows), {l['pa_id'] for l in e0})
    grids = od.standard_grids(LAMBDA_GRID, M_GRID)
    public['R5']['a'] = od.agreement(table)
    public['R5']['b'] = od.ess_by_variant(e0, games, grids, sn_inverse=u)
    public['R5']['c'] = {'weights': od.weight_summary(e0), **od.grid_weight_diagnostics(e0, LAMBDA_GRID, M_GRID),
                         'logging_mass': od.logging_mass(table)}
    public['R5']['d'] = od.positivity_censoring(result0, rows0)
    s_share = (result0.get('censoring_rule') or {}).get('worst_case_residual_share')
    public['R5']['e'], public['R5']['f'] = {}, {}
    for pair in PAIRS:
        public['R5']['e'][pair], public['R5']['f'][pair] = {}, {}
        for b in BOOTS:
            summary = summaries[b]
            stats = summary.get(pair) if summary.get('eligible') else None
            se_p = None if stats is None or not stats['mean_p']['valid'] else stats['mean_p']['se']
            se_lo = None if stats is None or not stats['lower_endpoint']['valid'] else stats['lower_endpoint']['se']
            public['R5']['e'][pair][b] = od.mde(se_p, se_lo, s_share)
            dec = None if stats is None else stats['decomposition']
            public['R5']['f'][pair][b] = ({k: dec[k] for k in ('var_point', 'var_censored', 'var_lower_endpoint')}
                                          | {'cov_point_censored': dec['cov_point_censored']}) if dec and dec['valid'] else {
                'var_point': None, 'var_censored': None, 'var_lower_endpoint': None, 'cov_point_censored': None}
    public['R5']['g'] = od.self_consistency(e0, table, resamplers['game'])
    # placebo on the baseline pair (reward enters only through the same ends as the baseline)
    step_rows = {}
    for i, entry in enumerate(table.entries):
        step_rows.setdefault(entry['pa'], {})[entry['index']] = i
    p_steps, placebo_lines, placebo_index = [], [], []
    for k, (line, index) in enumerate(zip(p_lines, np.flatnonzero(point))):
        if not ov.is_point(line):  # a shared unknown end: a placebo tilt breaks the slope cancellation
            continue
        rows_of = step_rows.get(line['pa_id'], {})
        p_steps.append([rows_of[t] for t in range(len(line['steps']))])
        placebo_lines.append(line)
        placebo_index.append(index)
    placebo_public, placebo_sealed = od.placebo_check(
        p_steps, table, {b: (resamplers[b] if eligible_p else None, np.array(placebo_index, dtype=np.int64))
                         for b in BOOTS},
        rewards=[line['reward'] for line in placebo_lines], checkpoint=tick,
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
    observed = lambda values: np.array([v for v, o in zip(values, obs) if o], dtype=np.float64)  # P ends observed
    with np.errstate(invalid='ignore', divide='ignore'):
        sealed['corr_r_r_tilde'] = {comp: float(np.corrcoef(observed(comp_rewards['R1_0']),
                                                             observed(comp_rewards[comp]))[0, 1])
                                    if sum(obs) > 2 else None for comp in ('R1_a', 'R1_b')}
    main_pairs = [pi for pi in pairs if pi.name in PAIRS and (pi.sn is None or u is not None)]
    sealed['pigeonhole'] = ob.pigeonhole(games, pitchers, main_pairs, point, draws=draws,
                                         seed=ROLE_VALUES['pigeonhole'], checkpoint=tick)  # descriptive, sealed only
    public['integrity'] = {'G_INT': 'PASS', 'variant_identity': 'PASS', 'inf_variants_reproduce_baseline': 'PASS',
                           'sn_normalizer_full_sample_valid': bool(sn_valid)}
    public['validity'] = {'point_set_meets_minimum': bool(eligible_p),
                          'game_bootstrap_eligible': bool(summaries['game'].get('eligible')),
                          'pitcher_bootstrap_eligible': bool(summaries['pitcher'].get('eligible'))}
    sealed['baseline_result'] = {k: result0.get(k) for k in ('layers', 'bootstrap', 'censoring_rule', 'ess_all_e0', 'ess')}
    sealed['counts'] = {'e0': len(e0), 'p': int(point.sum())}
    require_descriptive(public, sealed)
    return public, sealed, NOTES


REQUIRED_PUBLIC = (('R5', 'a'), ('R5', 'b'), ('R5', 'c'), ('R5', 'd'), ('R5', 'e'), ('R5', 'f'), ('R5', 'g'),
                   ('R5', 'h'), ('R5', 'i'), ('r1_variance',), ('integrity',), ('validity',))
REQUIRED_SEALED = ('selection', 'grid_points', 'grid_bootstrap', 'pairs', 'placebo', 'corr_r_r_tilde', 'pigeonhole',
                   'baseline_result', 'p_nonpoint', 'p_counts', 'gint', 'notes')


def require_descriptive(public, sealed):
    """Every descriptive output the design requires is present (a missing one is a stage integrity
    failure, never silent): R5 a-i, Var(r - q) vs Var(r~ - q), the integrity flags, and in the sealed file
    the selections and objectives, per-grid points and half-widths, the paired shifts with their CIs
    (Delta_R1 - Delta_R0 included), corr(r, r~), the pigeonhole bootstrap and the placebo values."""
    for path in REQUIRED_PUBLIC:
        node = public
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        _require(node is not None, f'descriptive output missing: {".".join(path)} (stage integrity)')
    for key in REQUIRED_SEALED:
        _require(sealed.get(key) is not None, f'sealed descriptive output missing: {key} (stage integrity)')
    sn_ok = bool(public['integrity'].get('sn_normalizer_full_sample_valid'))
    for pair in PAIRS:
        record = sealed['pairs'].get(pair) or {}
        _require('bootstrap' in record, f'sealed bootstrap record missing: {pair} (stage integrity)')
        _require('center' in record or (pair.endswith('R2_3') and not sn_ok),
                 f'sealed center missing: {pair} (stage integrity)')
        _require(pair == BASELINE or 'shift_center' in record, f'sealed paired shift missing: {pair} (stage integrity)')
    return True


def _label(value):
    return 'inf' if value == INF else str(value)


def seal(public, sealed):
    """Bytes of the sealed file and the public document carrying only the sealed sha256."""
    sealed_bytes = canonical_bytes(_jsonable(sealed))
    public = {**public, 'sealed_sha256': sha256_bytes(sealed_bytes)}
    validate_public(public)
    return canonical_bytes(public), sealed_bytes


def _jsonable(node):
    if isinstance(node, dict):
        return {str(k): _jsonable(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(v) for v in node]
    if isinstance(node, np.ndarray):
        return [_jsonable(v) for v in node.tolist()]
    if isinstance(node, (np.floating, float)):
        value = float(node)
        return value if math.isfinite(value) else None
    if isinstance(node, (np.integer,)):
        return int(node)
    return node


def unblind(sealed_bytes, sealed_sha256, decision_record_bytes, decision_record_sha256, *, committed):
    """Open the sealed file only after the decision record is committed (``committed`` = the bytes at
    HEAD equal the record) and both hashes match their stage seals."""
    _require(committed is True, 'the decision record must be committed before unblinding')
    _require(sha256_bytes(decision_record_bytes) == decision_record_sha256, 'decision record differs from its seal')
    _require(sha256_bytes(sealed_bytes) == sealed_sha256, 'sealed file differs from its A2-S3 seal')
    return json.loads(sealed_bytes)
