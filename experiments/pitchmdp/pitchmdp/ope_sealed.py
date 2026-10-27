"""EXP-OPE-RES-01 A2-S3 sealed evaluator and the typed public allowlist (design section 3, blinding).

``rehearsal_evaluate`` computes EVERY signed real-data quantity internally: Delta centers and CIs per
pair and grid value, paired shifts Delta_pair - Delta_baseline with their CIs (S-5), lambda/M
objective values and the selected values, per-grid-value half-widths, placebo Delta_hat values,
corr(r, r~). They go to the sealed document only (written as centers.sealed.json; only its sha256 is
printed). The public document is built from the typed allowlist only and ``validate_public`` rejects
any key or value outside its public type (in particular any signed effect or shift and any endpoint
of an effect or shift interval, and any key naming a selected value). Before any new arm is
computed the evaluator runs the deterministic integrity gates: G-INT (the registered estimator on the
sealed ledger reproduces the recorded S6 values within 1e-12), the variant path reproducing every
baseline row bit for bit, and the lambda = inf / M = inf variants reproducing the baseline within
1e-12. ``unblind`` opens the sealed file only after a committed decision record.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from . import ope_bootstrap as ob
from . import ope_diagnostics as od
from . import ope_variants as ov
from . import policy_estimator as est
from .ope_redesign_registry import (BASELINE, DRAWS, GAME_BOOTSTRAP_SEED, INF, INTEGRITY_TOL, INVALID_SHARE_MAX,
                                    LAMBDA_GRID, M_GRID, MIN_GAMES, MIN_PA_STARTS, MIN_PITCHERS, PAIRS, PUBLIC_TYPES,
                                    R1, ROLE_VALUES, pair_id)
from .policy_artifacts import IntegrityError, _require

BOOTS = ('game', 'pitcher')
PASS_FAIL = ('PASS', 'FAIL', 'INVALID', 'BASELINE')
FLAGS = ('SE_CALIBRATION_WARNING', 'ENABLED_DIAGNOSTIC_ONLY', 'NOT_RUN:R3', 'EXP-OPE-RES-01', 'A2-S3', 'PASS', 'FAIL')
_HEX64 = re.compile(r'[0-9a-f]{64}')
W, P, RF, CS, WV, G, T, V, F = PUBLIC_TYPES  # the nine registered public types, in allowlist order

# Typed allowlist: (path pattern, public type). '*' = any key; '**' = any subtree.
PAIR, BOOT = '<pair>', '<boot>'
SCHEMA = (
    (('experiment',), F), (('stage',), F), (('r3_status',), F), (('sealed_sha256',), F),
    (('pairs', PAIR, 'H_point', BOOT), W), (('pairs', PAIR, 'SE', BOOT), W),
    (('pairs', PAIR, 'endpoint_half_width', 'lower', BOOT), W), (('pairs', PAIR, 'endpoint_half_width', 'upper', BOOT), W),
    (('pairs', PAIR, 'valid', BOOT), F), (('pairs', PAIR, 'invalid_replicates', BOOT), CS),
    (('S5', PAIR, BOOT), P),
    (('R5', 'a', '**'), RF), (('R5', 'b', '**'), RF), (('R5', 'c', '**'), RF),
    (('R5', 'd', '**'), CS), (('R5', 'i', 'paths', '**'), CS),
    (('R5', 'e', PAIR, BOOT, '*'), WV), (('R5', 'f', PAIR, BOOT, '*'), WV),
    (('R5', 'h', 'sd_ratio', BOOT), WV), (('R5', 'h', 'placebos'), CS), (('R5', 'h', 'flag'), F),
    (('R5', 'h', 'decision_input'), F),
    (('R5', 'g', '**'), G), (('R5', 'i', 'train_residuals', '**'), T),
    (('r1_variance', '*'), V),
    (('integrity', '*'), F), (('validity', '**'), F),
)


def _match(pattern, path):
    if pattern and pattern[-1] == '**':
        head = pattern[:-1]
        return len(path) > len(head) and all(_seg(p, s) for p, s in zip(head, path))
    return len(pattern) == len(path) and all(_seg(p, s) for p, s in zip(pattern, path))


def _seg(pattern, segment):
    if pattern == PAIR:
        return segment in PAIRS or (segment.startswith('R3x') and segment[3:] in PAIRS)
    if pattern == BOOT:
        return segment in BOOTS
    return pattern == '*' or pattern == segment


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _type_ok(kind, value, key):
    if value is None:
        return True
    if kind in (W, V):
        return _number(value) and value >= 0
    if kind == P:
        return value in PASS_FAIL
    if kind == RF:
        return _number(value) and value >= 0
    if kind == CS:
        return (isinstance(value, int) and not isinstance(value, bool) and value >= 0) or (
            isinstance(value, float) and math.isfinite(value) and 0 <= value <= 1)
    if kind == WV:
        return _number(value) and (value >= 0 or str(key).startswith('cov_'))
    if kind == G:
        return _number(value) or (isinstance(value, list) and len(value) == 2 and all(map(_number, value)))
    if kind == T:
        return _number(value)
    if kind == F:
        return isinstance(value, bool) or (isinstance(value, str) and (_HEX64.fullmatch(value) or value in FLAGS))
    return False


EMPTY = object()


def _leaves(node, path=()):
    """(path, value) of every leaf; an empty container is a leaf too (its path must still be allowlisted)."""
    if isinstance(node, dict) and node:
        for key, value in node.items():
            _require(isinstance(key, str), 'public keys must be strings')
            yield from _leaves(value, (*path, key))
    elif isinstance(node, list) and node and not (len(node) == 2 and all(map(_number, node))):
        for i, value in enumerate(node):
            yield from _leaves(value, (*path, str(i)))
    elif isinstance(node, (dict, list)) and not node:
        yield path, EMPTY
    else:
        yield path, node


def validate_public(document):
    """Every leaf of the public document has one allowlisted path and a value of its public type.
    Refuses unknown keys, signed values outside the R5 g / R5 i types, interval endpoints outside R5 g,
    and any key naming a selection (the selected lambda / M is sealed)."""
    for path, value in _leaves(document):
        _require(not any('select' in str(p).lower() or 'center' in str(p).lower() or 'delta' == str(p).lower()
                         for p in path), f'public key names a sealed quantity: {".".join(path)}')
        kinds = [kind for pattern, kind in SCHEMA if _match(pattern, path)]
        _require(kinds, f'public key outside the typed allowlist: {".".join(path)}')
        _require(value is EMPTY or _type_ok(kinds[0], value, path[-1]),
                 f'public value outside its type {kinds[0]!r}: {".".join(path)}')
    return True


def canonical_bytes(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + '\n').encode('utf-8')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- G-INT
def gint_compare(result, reference):
    """Registered baseline values vs the recomputed baseline (1e-12; counts exact). Returns failures."""
    rule = result.get('censoring_rule') or {}
    boot = result.get('bootstrap') or {}
    l2 = (result.get('layers') or {}).get('L2_complete_conditional') or {}
    got = {'l1r_delta_bounds': result['layers']['L1_start_population']['delta_bounds'],
           'l1r_lower_endpoint_ci95': boot.get('L1_lower_endpoint_ci95'), 'l2_ci95': boot.get('L2_conditional_ci95'),
           'l2_delta_mean': l2.get('delta_mean'), 'worst_case_residual_share_s': rule.get('worst_case_residual_share'),
           'resolutions': rule.get('resolutions'), 'ess_all_e0_candidate': (result.get('ess_all_e0') or {}).get('candidate')}
    failures = []
    for key, expected in reference.items():
        value = got.get(key)
        if isinstance(expected, dict) and key == 'resolutions':
            ok = value == expected
        elif isinstance(expected, dict):
            ok = isinstance(value, dict) and set(value) >= set(expected) and all(
                value[k] is not None and abs(value[k] - expected[k]) <= INTEGRITY_TOL for k in expected)
        elif isinstance(expected, list):
            ok = isinstance(value, list) and len(value) == len(expected) and all(
                abs(a - b) <= INTEGRITY_TOL for a, b in zip(value, expected))
        else:
            ok = value is not None and abs(value - expected) <= INTEGRITY_TOL
        if not ok:
            failures.append(key)
    return failures


def registered_gint_reference(config):
    """The recorded S6-v3 values the registration quotes (S6-dr-a5 reproduces S6-dr-a4 exactly)."""
    s6 = config['mde_from_records']['S6_rehearsal_150_dev_games']
    return {'l1r_delta_bounds': s6['l1r_delta_bounds']['value'],
            'l1r_lower_endpoint_ci95': s6['l1r_lower_endpoint_ci95']['value'], 'l2_ci95': s6['l2_ci95']['value'],
            'l2_delta_mean': config['mde_from_records']['reference_magnitudes']['exposed_point_estimates']['S6_l2_delta_mean'],
            'worst_case_residual_share_s': s6['worst_case_residual_share_s']['value'],
            'resolutions': s6['resolutions']['value'], 'ess_all_e0_candidate': s6['ess_all_e0_candidate']['value']}


# ---------------------------------------------------------------- the evaluator
def _bits(value):
    return json.dumps(value, sort_keys=True, default=lambda v: float(v)) if value is not None else None


def rehearsal_evaluate(decisions, facts, *, surrogates, gint_reference, train_residuals, r1_paths, r3_status,
                       ess_gate=None, draws=DRAWS, minimum_games=MIN_GAMES, minimum_pitchers=MIN_PITCHERS,
                       minimum_pa_starts=MIN_PA_STARTS, placebos=None):
    """The blinded rehearsal. ``decisions()`` returns a fresh stream of the sealed ledger's decision
    rows; ``facts``: {pa_id: info} in ledger order (start_pitcher for the pitcher bootstrap);
    ``surrogates``: ope_surrogate.surrogate_rewards output. Returns (public, sealed, integrity)."""
    # 1. registered baseline path (byte-identical code) and G-INT before anything new is computed
    boot = {'draws': draws, 'seed': GAME_BOOTSTRAP_SEED, 'invalid_share_max': INVALID_SHARE_MAX,
            'minimum': {'games': minimum_games, 'pa_starts': minimum_pa_starts}}
    result0, rows0 = est.estimate(decisions(), facts, **boot, ess_gate=ess_gate, censoring='l1r')
    failures = gint_compare(result0, gint_reference) if gint_reference is not None else []
    if failures:
        raise IntegrityError(f'G-INT failed (deterministic integrity): {failures}; no new arm is computed')
    # 2. variant lines, identity reproduction (bit for bit)
    lines, by_pa = [], {}
    for pa_id, info, rows in est.by_pa(decisions(), facts):
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
    point = np.array([ov.is_point(line) for line in e0])
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
    sealed = {'selection': {}, 'grid_points': {}, 'gint': {'reference_keys': sorted(gint_reference or {}), 'failures': []}}
    pairs, inf_ok = [], True
    n_bar, sn_valid = ov.sn_normalizers(e0)
    u = (1 / n_bar[0], 1 / n_bar[1]) if sn_valid else None
    for comp in R1:
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
            pairs.append(ob.sn_pair(name, e0, point, None if rw is None else rw))
        else:
            pairs.append(ob.PairInputs(name, deltas=np.full(len(e0), np.nan), lo=np.full(len(e0), np.nan),
                                       hi=np.full(len(e0), np.nan)))
    _require(inf_ok, 'lambda = inf / M = inf variants do not reproduce the baseline (deterministic integrity)')
    # 5. bootstraps (paired, same replicates for every pair)
    resamplers = {'game': ob.Resampler(games, unit='game', draws=draws, seed=GAME_BOOTSTRAP_SEED,
                                       minimum_clusters=minimum_games, minimum_pa_starts=minimum_pa_starts),
                  'pitcher': ob.Resampler(pitchers, unit='pitcher', draws=draws, seed=ROLE_VALUES['pitcher_bootstrap'],
                                          minimum_clusters=minimum_pitchers, minimum_pa_starts=minimum_pa_starts)}
    eligible_p = int(point.sum()) >= minimum_pa_starts
    summaries = {}
    for b, resampler in resamplers.items():
        stats, valid = ob.run_pairs(resampler, pairs, point, BASELINE)
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
            deltas, lo, hi = ob.sn_values(pair_inputs, np.vstack(u))
        else:
            deltas, lo, hi = pair_inputs.deltas, pair_inputs.lo, pair_inputs.hi
        center = {'mean_p': float(np.nanmean(deltas[point])) if point.any() else None,
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
    p_steps = []
    for line in p_lines:
        rows_of = step_rows.get(line['pa_id'], {})
        p_steps.append([rows_of[t] for t in range(len(line['steps']))])
    p_index = np.flatnonzero(point)
    placebo_public, placebo_sealed = od.placebo_check(
        p_steps, table, {b: (resamplers[b] if eligible_p else None, p_index) for b in BOOTS},
        rewards=[line['reward'] for line in p_lines], **({} if placebos is None else {'placebos': placebos}))
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
    with np.errstate(invalid='ignore', divide='ignore'):
        sealed['corr_r_r_tilde'] = {comp: float(np.corrcoef(np.array(comp_rewards['R1_0'])[obs],
                                                             np.array(comp_rewards[comp])[obs])[0, 1])
                                    if sum(obs) > 2 else None for comp in ('R1_a', 'R1_b')}
    main_pairs = [pi for pi in pairs if pi.name in PAIRS and (pi.sn is None or u is not None)]
    sealed['pigeonhole'] = ob.pigeonhole(games, pitchers, main_pairs, point, draws=draws,
                                         seed=ROLE_VALUES['pigeonhole'])  # descriptive, sealed only
    public['integrity'] = {'G_INT': 'PASS', 'variant_identity': 'PASS', 'inf_variants_reproduce_baseline': 'PASS',
                           'sn_normalizer_full_sample_valid': bool(sn_valid)}
    public['validity'] = {'point_set_meets_minimum': bool(eligible_p),
                          'game_bootstrap_eligible': bool(summaries['game'].get('eligible')),
                          'pitcher_bootstrap_eligible': bool(summaries['pitcher'].get('eligible'))}
    sealed['baseline_result'] = {k: result0.get(k) for k in ('layers', 'bootstrap', 'censoring_rule', 'ess_all_e0', 'ess')}
    sealed['counts'] = {'e0': len(e0), 'p': int(point.sum())}
    return public, sealed, {'gint_failures': failures}


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
