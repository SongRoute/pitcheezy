"""EXP-OPE-RES-01 A2-S4 decision (design section 5), computed from the PUBLIC rehearsal file and the
synthetic results only (no center is read).

(ii) H_point(pair) <= (1 - X) H_point(baseline) under the game AND under the pitcher bootstrap.
(i) S-1 and S-4 pooled nets PASS (and the baseline's own S-1/S-4 PASS: shared nets), S-2 PASS for
pairs using R1_a/R1_b (gamma cells at 2 kappa_p and 2 zeta_p, plus the gamma_a invariant), S-3 PASS in
the four TAB acceptance cells, S-5 PASS under both bootstraps. Labels in precedence: FAILED_INTEGRITY,
NOT_EVALUABLE, ADOPT_FOR_2027_PROTOCOL:<pair>, BIAS_NOT_CERTIFIED:<pairs>, NO_PRECISION_GAIN. Statistical
outcomes are recorded as they are (no retry). Bounds are nominal per pair and cell (not simultaneous).
"""
from __future__ import annotations

import math

from .ope_redesign_registry import (BASELINE, R1, TAB_ACCEPTANCE, TOTAL_ORDER, X, select_pair, split_pair)
from .policy_artifacts import _require

BOOTS = ('game', 'pitcher')


def _width(public, pair, boot):
    entry = public['pairs'][pair]
    valid = entry['valid'][boot]
    value = entry['H_point'][boot]
    return value if valid and value is not None and math.isfinite(value) else None


def condition_ii(public, pair):
    """(ii) H_point(pair) <= (1 - X) H_point(baseline), evaluated directly under each bootstrap. A zero
    (degenerate) baseline or pair half-width is INCONCLUSIVE, never a pass; the ratio is reporting only."""
    base = {b: _width(public, BASELINE, b) for b in BOOTS}
    own = {b: _width(public, pair, b) for b in BOOTS}
    valid = all(v is not None for v in (*base.values(), *own.values()))
    degenerate = valid and any(base[b] == 0 or own[b] == 0 for b in BOOTS)
    ratio = {b: (own[b] / base[b] if valid and base[b] > 0 else None) for b in BOOTS}
    passed = valid and not degenerate and all(own[b] <= (1 - X) * base[b] for b in BOOTS)
    return {'valid': valid, 'degenerate': bool(degenerate), 'ratio': ratio, 'pass': bool(passed),
            'outcome': 'INVALID' if not valid else 'INCONCLUSIVE' if degenerate else 'PASS' if passed else 'FAIL'}


BLOCKING = ('INVALID', 'NOT_EVALUABLE', None)


def _cell(record):
    """(outcome, invalid) of one bias cell record (a dict with 'outcome', or a bare outcome string)."""
    if isinstance(record, dict):
        return record.get('outcome'), bool(record.get('invalid_repetition')) or record.get('outcome') in BLOCKING
    return record, record in BLOCKING


def condition_i(pair, public, tab, s5x):
    """Every registered bias cell of the pair; an invalid repetition, replication or missing cell makes the
    pair non-adoptable (mandatory statistic invalid) and never counts as PASS."""
    r1, _ = split_pair(pair)
    nets = s5x['nets']
    cells = {'S1': nets['S1'][pair], 'S4': nets['S4'][pair], 'S1_baseline': nets['S1'][BASELINE],
             'S4_baseline': nets['S4'][BASELINE]}
    for cell in TAB_ACCEPTANCE:
        cells[f"S3:{cell['id']}"] = tab['acceptance'][cell['id']][pair]
    if r1 in ('R1_a', 'R1_b'):
        s2 = (s5x['s2'].get(pair) or {})
        for label in ('2*kappa_p', '2*zeta_p'):
            cells[f'S2:{label}'] = s2.get(label)
        cells['S2:invariant'] = s2.get('invariant')
    cells['S5:game'] = public['S5'][pair]['game']
    cells['S5:pitcher'] = public['S5'][pair]['pitcher']
    parsed = {name: _cell(record) for name, record in cells.items()}
    valid = not any(invalid for _, invalid in parsed.values())
    outcomes = {name: outcome for name, (outcome, _) in parsed.items()}
    return {'cells': outcomes, 'invalid_cells': sorted(n for n, (_, bad) in parsed.items() if bad), 'valid': valid,
            'pass': bool(valid and all(o == 'PASS' for o in outcomes.values()))}


def decide(public, tab, s5x, *, integrity_failures=(), r3_status):
    """The registered label from the public file and the synthetic results (design 5.1/5.2)."""
    out = {'r3_status': r3_status, 'flags': [], 'pairs': {}, 'nominal_coverage': 'per pair and cell; not simultaneous'}
    flag = ((public.get('R5') or {}).get('h') or {}).get('flag')
    if flag:
        out['flags'].append(flag)
    if integrity_failures:
        return {**out, 'label': 'FAILED_INTEGRITY', 'integrity_failures': list(integrity_failures)}
    if any(_width(public, BASELINE, b) is None for b in BOOTS):
        return {**out, 'label': 'NOT_EVALUABLE'}
    h_star, adopt, precision = {}, [], []
    for pair in TOTAL_ORDER:
        ii = condition_ii(public, pair)
        i = condition_i(pair, public, tab, s5x)
        mandatory = ii['valid'] and i['valid']
        out['pairs'][pair] = {'ii': ii, 'i': i, 'mandatory_valid': mandatory}
        if ii['pass']:
            precision.append(pair)
        if mandatory and ii['pass'] and i['pass']:
            adopt.append(pair)
            h_star[pair] = max(_width(public, pair, b) for b in BOOTS)
    base_star = max(_width(public, BASELINE, b) for b in BOOTS)
    if adopt:
        chosen = select_pair(h_star, adopt, base_star)
        return {**out, 'label': f'ADOPT_FOR_2027_PROTOCOL:{chosen}', 'pair': chosen, 'h_star': h_star,
                'h_star_baseline': base_star}
    if precision:
        return {**out, 'label': 'BIAS_NOT_CERTIFIED:' + ','.join(p for p in TOTAL_ORDER if p in precision)}
    return {**out, 'label': 'NO_PRECISION_GAIN'}


def check_inputs(public, tab, s5x):
    """Structural check of the decision inputs (every pair and cell present)."""
    _require(set(public['pairs']) >= {BASELINE, *TOTAL_ORDER} and set(public['S5']) >= set(TOTAL_ORDER),
             'public file misses a pair')
    for cell in TAB_ACCEPTANCE:
        _require(set(tab['acceptance'][cell['id']]) >= {BASELINE, *TOTAL_ORDER}, f'TAB {cell["id"]} misses a pair')
    for net in ('S1', 'S4'):
        _require(set(s5x['nets'][net]) >= {BASELINE, *TOTAL_ORDER}, f'S5-X {net} misses a pair')
    for pair in TOTAL_ORDER:
        if split_pair(pair)[0] in R1[1:]:
            _require({'2*kappa_p', '2*zeta_p', 'invariant'} <= set(s5x['s2'].get(pair) or {}), f'S5-X S2 misses {pair}')
    return True
