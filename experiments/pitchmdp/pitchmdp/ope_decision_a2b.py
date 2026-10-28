"""EXP-OPE-RES-02 gate record A2b-G and the A2b-S4 decision (design 7, 8), from public files and synthetic records
only (no center is ever read).

- ``PublicView``: the A2b-S3 public file behind the strict typed allowlist; construction validates the whole file
  (a sealed document or any center / shift / selected-lambda key is refused) and every read is checked against the
  allowlist paths, so the gate and the decision cannot touch anything else.
- ``gate_record``: deterministic from the public file (and the reward-free new E0 count of A2b-L for the TAB reuse
  rule): a candidate goes to phase 2 iff (ii) passes under BOTH bootstraps with valid statistics and S-5 passes
  under both; the work list follows (R1_a -> S-2' 2 zeta_p; R1_b -> 2 kappa_p, 2 zeta_p and the descriptive
  kappa_p' cell; B0 tuples -> the TAB reuse rule). Under D192 2 3 the B2 rows are descriptive and never gated.
- ``decide``: the labels in precedence (FAILED_INTEGRITY, NOT_EVALUABLE, ADOPT_FOR_2027_PROTOCOL:<tuple>,
  DENOMINATOR_NOT_CERTIFIED:<B2 tuples> - reachable only under ADOPTION_ELIGIBLE*, i.e. never under the pinned
  DIAGNOSTIC_ONLY:DECIDED -, BIAS_NOT_CERTIFIED:<tuples> with failing families, NO_PRECISION_GAIN), the selection
  (H* = max of the two half-widths, the 2% tie window of H*(baseline), the registered total order) and the flags.
"""
from __future__ import annotations

import hashlib
import json
import math

from . import ope_redesign_registry as a2
from . import ope_registry_a2b as rb
from .policy_artifacts import IntegrityError, _require

BOOTS = rb.BOOTS
BLOCKING = ('INVALID', 'NOT_EVALUABLE', 'NOT_RUN', None)


class PublicView:
    """Read-only access to the A2b-S3 public file through the typed allowlist."""
    def __init__(self, document):
        _require(isinstance(document, dict), 'the gate reads the A2b-S3 public file (a JSON object)')
        rb.validate_public(document)  # a sealed file, a center or any unlisted key refuses here
        self._doc = document

    def get(self, *path):
        node = self._doc
        for part in path:
            _require(isinstance(node, dict) and part in node, f'public key missing: {".".join(path)}')
            node = node[part]
        if not isinstance(node, dict):
            _require(any(a2._schema_match(p, path) for p, _, _ in rb.PUBLIC_SCHEMA),
                     f'public read outside the typed allowlist refused: {".".join(path)}')
        else:
            _require(any(a2._prefix_match(p, path) for p, _, _ in rb.PUBLIC_SCHEMA),
                     f'public read outside the typed allowlist refused: {".".join(path)}')
        return node

    def width(self, row, boot):
        valid = self.get('pairs', row, 'valid', boot)
        value = self.get('pairs', row, 'H_point', boot)
        return value if valid and value is not None and math.isfinite(value) else None


def _canonical(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + '\n').encode('utf-8')


def condition_ii(view, row):
    return rb.condition_ii({b: view.width(row, b) for b in BOOTS}, {b: view.width(rb.BASELINE, b) for b in BOOTS})


def s5_pass(view, row):
    outcomes = {b: view.get('S5', row, b) for b in BOOTS}
    return all(v == 'PASS' for v in outcomes.values()), outcomes


def gate_record(public, *, e0_count, r3_status=rb.R3_STATUS_DECIDED, public_sha256=None):
    """A2b-G: deterministic from the public file (same input -> same bytes); no judgment."""
    view = PublicView(public)
    _require(r3_status == rb.R3_STATUS_DECIDED or r3_status in rb.ADOPTION_ELIGIBLE_STATUSES,
             f'unknown r3_status for the gate {r3_status!r}')
    candidates, gated = {}, []
    eligible_b2 = r3_status in rb.ADOPTION_ELIGIBLE_STATUSES  # never under the pinned DIAGNOSTIC_ONLY:DECIDED
    tuples = (*rb.CANDIDATES, *(rb.TOTAL_ORDER_APPENDED_OPTIONS_1_2 if eligible_b2 else ()))
    for row in tuples:
        ii = condition_ii(view, row)
        s5_ok, s5 = s5_pass(view, row)
        denominator_ok = True
        if rb.denominator_of(row) == 'B2':  # options 1/2 only: E5 and B2 validity gate the B2 tuples too
            denominator_ok = bool(view.get('B2', 'validity', 'pass')) and bool(view.get('B2', 'E5', 'pass'))
        passed = ii['pass'] and s5_ok and denominator_ok
        reason = None if passed else ('(ii) ' + ii['outcome'] if not ii['pass'] else
                                      'S-5 not PASS under both bootstraps' if not s5_ok else 'E5 or B2 validity')
        candidates[row] = {'ii': {'outcome': ii['outcome'], 'per_bootstrap': ii['per_bootstrap'], 'ratio': ii['ratio']},
                           'S5': s5, 'phase_2': bool(passed), 'reason': reason}
        if passed:
            gated.append(row)
    tab = rb.tab_reuse(e0_count)
    b0_gated = [t for t in gated if rb.denominator_of(t) == 'B0']
    record = {'experiment': rb.EXPERIMENT_ID, 'stage': 'A2b-G', 'r3_status': r3_status,
              'public_sha256': public_sha256, 'candidates': candidates, 'gated': gated,
              'work_list': {'S2_prime_cells': list(rb.s2p_cells_for(gated)),
                            'tab': {**tab, 'rerun': bool(b0_gated and not tab['reuse'])},
                            'b2_phase_2': [] if not eligible_b2 else [t for t in gated if rb.denominator_of(t) == 'B2']},
              'b2_rows': 'descriptive rows (DIAGNOSTIC_ONLY:DECIDED, D192 2 3): never gated',
              'b2_precision_pass_diagnostic': bool(view.get('flags', 'B2_PRECISION_PASS_DIAGNOSTIC')),
              'rule': 'phase 2 iff (ii) H_point <= 0.70 H_point(baseline) under the game and the pitcher bootstrap '
                      '(valid, not degenerate) and S-5 PASS under both; R1_a -> S-2\' 2 zeta_p; R1_b -> 2 kappa_p, '
                      '2 zeta_p (+ kappa_p\' descriptive); B0 -> TAB reuse unless the E0 count is outside +-20%',
              'empty': not gated}
    validate_gate_record(record)
    return record


GATE_KEYS = ('experiment', 'stage', 'r3_status', 'public_sha256', 'candidates', 'gated', 'work_list', 'b2_rows',
             'b2_precision_pass_diagnostic', 'rule', 'empty')


def validate_gate_record(record):
    """The gate record carries widths' ratios, pass/fail and the work list only (public item 12 type)."""
    _require(isinstance(record, dict) and set(record) == set(GATE_KEYS), 'gate record keys')
    for row, entry in record['candidates'].items():
        _require(row in rb.ROWS and set(entry) == {'ii', 'S5', 'phase_2', 'reason'}, 'gate record candidate entry')
        _require(set(entry['ii']) == {'outcome', 'per_bootstrap', 'ratio'}, 'gate record (ii) entry')
        for value in entry['ii']['ratio'].values():
            _require(value is None or (isinstance(value, float) and math.isfinite(value) and value >= 0),
                     'a gate ratio is a nonnegative width ratio')
    _require(all(c in rb.S2P_CELLS for c in record['work_list']['S2_prime_cells']), 'gate work list cells')
    return True


def gate_bytes(record):
    return _canonical(record)


def gate_sha256(record):
    return hashlib.sha256(gate_bytes(record)).hexdigest()


# ---------------------------------------------------------------- the decision
def _cell(record):
    if isinstance(record, dict):
        return record.get('outcome'), bool(record.get('invalid_repetition')) or record.get('outcome') in BLOCKING
    return record, record in BLOCKING


FAMILY_OF = {'TAB': 'TAB', 'S1': 'COARSE_NET', 'S4': 'COARSE_NET', 'S2': 'R1_SURROGATE', 'S5': 'REAL_SHIFT',
             'DEN': 'DENOMINATOR'}


def condition_i(row, view, *, tab, nets, s2_prime):
    """Every registered bias cell of a B0 candidate (design 5.6): the four TAB cells (reused A2 T-A or re-run), the
    reused A2 S-1 / S-4 nets of the pair and of the baseline, the S-2' cells of its R1 component (R1_a: 2 zeta_p;
    R1_b: 2 kappa_p and 2 zeta_p) with the gamma_a invariant, and S-5 under both bootstraps. A missing, invalid or
    not-run cell makes the tuple non-adoptable and never counts as PASS."""
    r1 = rb.split_row(row)[0]
    pair = rb.a2_pair(row)
    cells = {}
    for spec in rb.TAB_ACCEPTANCE:
        cells[f"TAB:{spec['id']}"] = ((tab.get(spec['id']) or {}).get(pair))
    cells['S1'] = (nets.get('S1') or {}).get(pair)
    cells['S4'] = (nets.get('S4') or {}).get(pair)
    cells['S1:baseline'] = (nets.get('S1') or {}).get(a2.BASELINE)
    cells['S4:baseline'] = (nets.get('S4') or {}).get(a2.BASELINE)
    for cell in rb.S2P_COMPONENT_CELLS.get(r1, ()):
        pooled = s2_prime.get(cell)
        cells[f'S2:{cell}'] = 'NOT_RUN' if pooled is None else (pooled['tuples'].get(row) or 'NOT_RUN')
    if r1 in ('R1_a', 'R1_b'):
        kappa = s2_prime.get('2*kappa_p')
        if r1 == 'R1_b':
            cells['S2:invariant'] = 'NOT_RUN' if kappa is None or kappa.get('invariant') is None else \
                kappa['invariant']['outcome']
    ok5, s5 = s5_pass(view, row)
    for b in BOOTS:
        cells[f'S5:{b}'] = s5[b]
    parsed = {name: _cell(record) for name, record in cells.items()}
    outcomes = {name: outcome for name, (outcome, _) in parsed.items()}
    invalid = sorted(n for n, (_, bad) in parsed.items() if bad)
    failing = sorted({FAMILY_OF[n.split(':')[0]] for n, o in outcomes.items() if o not in rb.PASSING})
    return {'cells': outcomes, 'invalid_cells': invalid, 'valid': not invalid,
            'pass': bool(not invalid and all(o in rb.PASSING for o in outcomes.values())), 'failing_families': failing}


def select(h_star, passing, baseline_h_star):
    """Among the passing tuples: H*_min, the 2% window of H*(baseline) (fixed reference), first in total order."""
    order = [t for t in (*rb.TOTAL_ORDER, *rb.TOTAL_ORDER_APPENDED_OPTIONS_1_2) if t in passing]
    if not order:
        return None
    _require(all(math.isfinite(h_star[t]) for t in order) and math.isfinite(baseline_h_star),
             'selection needs finite half-widths')
    h_min = min(h_star[t] for t in order)
    return [t for t in order if h_star[t] - h_min <= a2.TIE_WINDOW * baseline_h_star][0]


def decide(public, gate, *, tab, nets, s2_prime, integrity_failures=(), r3_status=rb.R3_STATUS_DECIDED,
           r3_stage_failed_integrity=False, denominator_cells=None):
    """The registered label (design 8.2) from the public file, the committed gate record and the phase-2 records.
    ``tab``: {T-A id: {A2 pair: cell record}} (reused A2 or re-run); ``nets``: the reused A2 S-1 / S-4 nets;
    ``s2_prime``: {cell: pooled record} of the opened cells; ``denominator_cells``: {B2 tuple: {cell: outcome}}
    (options 1/2 only; never under DIAGNOSTIC_ONLY:DECIDED)."""
    view = PublicView(public)
    _require(gate.get('experiment') == rb.EXPERIMENT_ID and gate.get('stage') == 'A2b-G', 'committed gate record required')
    out = {'experiment': rb.EXPERIMENT_ID, 'r3_status': r3_status, 'rehearsal_exposure': rb.REHEARSAL_EXPOSURE,
           'flags': [], 'tuples': {}, 'nominal_coverage': 'per tuple and cell; not simultaneous',
           'gate_record_sha256': gate_sha256(gate)}
    if view.get('R5', 'h', 'flag'):
        out['flags'].append('SE_CALIBRATION_WARNING')
    if view.get('flags', 'B2_PRECISION_PASS_DIAGNOSTIC'):
        out['flags'].append('B2_PRECISION_PASS_DIAGNOSTIC')
    if r3_stage_failed_integrity:
        out['flags'].append('r3_stage_failed_integrity')
    if integrity_failures:
        return {**out, 'label': 'FAILED_INTEGRITY', 'integrity_failures': list(integrity_failures)}
    if any(view.width(rb.BASELINE, b) is None for b in BOOTS):
        return {**out, 'label': 'NOT_EVALUABLE'}
    eligible_b2 = r3_status in rb.ADOPTION_ELIGIBLE_STATUSES and not r3_stage_failed_integrity
    h_star, adopt, precision, denominator_failed = {}, [], [], []
    for row in rb.CANDIDATES:
        ii = condition_ii(view, row)
        entry = {'ii': ii, 'gated': row in gate['gated']}
        if ii['pass']:
            precision.append(row)
            i = condition_i(row, view, tab=tab, nets=nets, s2_prime=s2_prime) if row in gate['gated'] else {
                'pass': False, 'valid': True, 'failing_families': ['REAL_SHIFT'],
                'cells': {f'S5:{b}': view.get('S5', row, b) for b in BOOTS}, 'invalid_cells': []}
            entry['i'] = i
            mandatory = ii['valid'] and i['valid']
            entry['mandatory_valid'] = mandatory
            if mandatory and i['pass']:
                adopt.append(row)
                h_star[row] = max(view.width(row, b) for b in BOOTS)
        out['tuples'][row] = entry
    if eligible_b2:  # options 1/2 only (not selected, D192): the label-4 rule, unreachable under the pinned status
        for row in rb.TOTAL_ORDER_APPENDED_OPTIONS_1_2:
            ii = condition_ii(view, row)
            if not ii['pass']:
                continue
            precision.append(row)
            entry = {'ii': ii, 'gated': row in gate['gated']}
            if not entry['gated']:  # stopped at the gate by S-5, E5 or B2 validity: a denominator condition
                entry['i'] = {'pass': False, 'valid': True, 'failing_families': ['DENOMINATOR'], 'invalid_cells': []}
                out['tuples'][row] = entry
                continue
            r1 = rb.split_row(row)[0]
            s2_ok = all(((s2_prime.get(c) or {}).get('components') or {}).get(r1, {}).get('outcome') in rb.PASSING
                        for c in rb.S2P_COMPONENT_CELLS.get(r1, ()))
            cells = (denominator_cells or {}).get(row) or {}
            den_ok = bool(cells) and all(v in rb.PASSING for v in cells.values())
            families = sorted({*(() if s2_ok else ('R1_SURROGATE',)), *(() if den_ok else ('DENOMINATOR',))})
            entry['i'] = {'pass': s2_ok and den_ok, 'valid': True, 'failing_families': families, 'invalid_cells': [],
                          'denominator_cells': cells}
            out['tuples'][row] = entry
            if s2_ok and den_ok:
                adopt.append(row)
                h_star[row] = max(view.width(row, b) for b in BOOTS)
            elif s2_ok:
                denominator_failed.append(row)
    base_star = max(view.width(rb.BASELINE, b) for b in BOOTS)
    if adopt:
        chosen = select(h_star, adopt, base_star)
        return {**out, 'label': f'ADOPT_FOR_2027_PROTOCOL:{chosen}', 'tuple': chosen, 'h_star': h_star,
                'h_star_baseline': base_star}
    if denominator_failed:  # unreachable under DIAGNOSTIC_ONLY:DECIDED (label kept in the precedence list)
        return {**out, 'label': 'DENOMINATOR_NOT_CERTIFIED:' + ','.join(denominator_failed)}
    if precision:
        families = {t: out['tuples'][t]['i']['failing_families'] for t in precision}
        return {**out, 'label': 'BIAS_NOT_CERTIFIED:' + ','.join(precision), 'failing_families': families}
    return {**out, 'label': 'NO_PRECISION_GAIN'}


def label_name(label):
    return label.split(':')[0]


def require_label(record):
    _require(label_name(record['label']) in [label_name(l) for l in rb.LABELS], f'unknown label {record["label"]!r}')
    if record['r3_status'] == rb.R3_STATUS_DECIDED:
        _require(not record['label'].startswith('DENOMINATOR_NOT_CERTIFIED'),
                 'DENOMINATOR_NOT_CERTIFIED is unreachable under DIAGNOSTIC_ONLY:DECIDED (deterministic integrity)')
    return record


def check_inputs(tab, nets, s2_prime, gate):
    """Structural check of the decision inputs: the reused TAB / net records cover every candidate, and an S-2'
    record exists for every cell the gate opened."""
    for spec in rb.TAB_ACCEPTANCE:
        _require(set(tab.get(spec['id']) or {}) >= {rb.a2_pair(t) for t in rb.CANDIDATES},
                 f'TAB {spec["id"]} misses a candidate')
    for net in ('S1', 'S4'):
        _require(set(nets.get(net) or {}) >= {a2.BASELINE, *(rb.a2_pair(t) for t in rb.CANDIDATES)},
                 f'S5-X {net} misses a candidate')
    missing = [c for c in gate['work_list']['S2_prime_cells'] if c in rb.S2P_DECISION_CELLS and c not in s2_prime]
    _require(not missing, f'the gate opened S-2\' cells without a pooled record: {missing}')
    return True


def describe_failure(error):
    return {'class': 'deterministic_integrity' if isinstance(error, IntegrityError) else 'error', 'error': str(error)}
