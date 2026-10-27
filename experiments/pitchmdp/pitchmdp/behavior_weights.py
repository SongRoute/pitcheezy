"""S6 ledger reader, E0, PA weights and OPE weight diagnostics of EXP-BC-CAL-01 (design section 7.4).

``read_ledger`` opens only a ``ledger*.jsonl`` file next to its ``manifest.json``: both sha256
equal the registration pins before and after reading (raw-byte hashing, a separate authorized
operation), the header and every decision row carry the pinned runtime identity, and every line is
read by a key-by-key scanner: only the registered allowlist is decoded (header: kind,
runtime_sha256, pins; decisions: identity/state fields and result.{mask, logging, reference,
candidate, logged_index, rho_reference, rho_candidate}); the value of every other member
(``q_*``, ``mc_se``, fingerprints, details, ...) is validated against the JSON grammar and skipped,
never decoded into a Python object (only member names of skipped objects are decoded, to refuse
duplicates). A malformed line (bad string, escape, number, literal or bracket structure), a
duplicate member at any nesting level or a non-UTF-8 line refuses the ledger. ``pa_values.parquet``,
``dr.json`` or any other path is refused. Rewards are never read.

E0 is rebuilt from each PA's first row with ``PolicyRuntime.start_population`` itself (borrowed,
not copied). Weights follow the L1-R rule of policy_estimator (``affine`` unrolled):
rho_t = pi(a_t) / pi_b_hat(a_t) on SUPPORTED rows, 0 on OUTSIDE_POLICY_SUPPORT rows (pi(a) = 0);
residual W_t = prod_{s<=t} rho_s over the evaluated prefix before the first refusal k; a
LOGGING_POSITIVITY refusal at k closes the PA with rho_k = 0; any other refusal keeps
prod_{s<k} rho_s. W_T (the affine slope) feeds ESS, the PA-weight quantiles and SC-2.
"""
from __future__ import annotations

import hashlib
import json
from json.decoder import scanstring
import math
from pathlib import Path
import re

import numpy as np

from .policy_artifacts import IntegrityError
from .policy_estimator import effective_sample_size
from .policy_runtime import MaskedReference, PolicyRuntime
from .rollout_policy import PAState

EVALUATED = ('SUPPORTED', 'OUTSIDE_POLICY_SUPPORT')
POSITIVITY = 'UNSUPPORTED_LOGGING_POSITIVITY'
POLICIES = ('candidate', 'reference')
ALLOWED_HEADER = ('runtime_sha256', 'pins')
ALLOWED_DECISION = ('kind', 'request_id', 'runtime_sha256', 'pa_id', 'decision_index', 'status', 'pitcher',
                    'pitcher_hand', 'batter_side', 'balls', 'strikes', 'history_actions', 'logged_action',
                    'context_key')
ALLOWED_RESULT = ('mask', 'logging', 'reference', 'candidate', 'logged_index', 'rho_reference', 'rho_candidate')
FORBIDDEN_FILES = ('pa_values.parquet', 'dr.json')
HEADER_SCHEMA = {'kind': None, **{k: None for k in ALLOWED_HEADER}}
DECISION_SCHEMA = {**{k: None for k in ALLOWED_DECISION}, 'result': {k: None for k in ALLOWED_RESULT}}
PARSING_DISCLOSURE = ('ledger lines are scanned key by key: allowlisted values are decoded (strict json raw_decode, '
                      'NaN / Infinity refused); the value of every other member is validated against the JSON grammar '
                      'and skipped, never decoded into a Python object (only the member names of skipped objects are '
                      'decoded, to refuse duplicates); a malformed or non-UTF-8 line or a duplicate member at any '
                      'level refuses the ledger (FAILED_INTEGRITY); the file bytes are sha256-hashed before and after '
                      'reading as a separate authorized operation')


class LedgerRefused(IntegrityError):
    """A forbidden file or an unpinned / changed ledger was requested (reader refuses)."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- allowlist scanner

def _reject_constant(name):
    raise IntegrityError(f'non-JSON constant {name} in a ledger line')


def _unique_members(pairs):
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise IntegrityError('duplicate member in an allowlisted ledger value')
    return dict(pairs)


# strict JSON for allowlisted values: NaN / Infinity refused, duplicate members refused at every level
_DECODER = json.JSONDecoder(parse_constant=_reject_constant, object_pairs_hook=_unique_members)
_WS = re.compile(r'[ \t\n\r]*')
_STRING = re.compile(r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"')
_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
_LITERALS = ('true', 'false', 'null')
_SCALAR_END = ' \t\n\r,]}'
MAX_DEPTH = 64


def _ws(text, i):
    return _WS.match(text, i).end()


def _boundary(text, end):
    if end < len(text) and text[end] not in _SCALAR_END:
        raise IntegrityError('invalid JSON scalar in a ledger line')
    return end


def _skip(text, i, depth=0):
    """Index just past the JSON value starting at ``text[i]``. The value is validated against the JSON
    grammar (strict strings and escapes, numbers, true / false / null, matched arrays and objects with
    commas and colons, no duplicate member in any object) but no value is decoded into a Python object;
    only the member names of skipped objects are decoded, for the duplicate check."""
    if depth > MAX_DEPTH:
        raise IntegrityError('ledger value nested too deeply')
    first = text[i] if i < len(text) else ''
    if first == '"':
        match = _STRING.match(text, i)
        if match is None:
            raise IntegrityError('invalid JSON string in a ledger line')
        return match.end()
    if first in '[{':
        close, i = (']' if first == '[' else '}'), _ws(text, i + 1)
        names = set()
        if text.startswith(close, i):
            return i + 1
        while True:
            if first == '{':
                if not text.startswith('"', i):
                    raise IntegrityError('expected a member name in a ledger object')
                end = _skip(text, i, depth + 1)
                try:
                    name, _ = scanstring(text, i + 1)
                except ValueError as error:
                    raise IntegrityError(f'malformed member name in a ledger line: {error}') from error
                if name in names:
                    raise IntegrityError('duplicate member in a skipped ledger value')
                names.add(name)
                i = _ws(text, end)
                if not text.startswith(':', i):
                    raise IntegrityError("expected ':' in a ledger object")
                i = _ws(text, i + 1)
            i = _ws(text, _skip(text, i, depth + 1))
            if text.startswith(',', i):
                i = _ws(text, i + 1)
                continue
            if text.startswith(close, i):
                return i + 1
            raise IntegrityError(f"expected ',' or '{close}' in a ledger line")
    for literal in _LITERALS:
        if text.startswith(literal, i):
            return _boundary(text, i + len(literal))
    match = _NUMBER.match(text, i)
    if match is None:
        raise IntegrityError('invalid JSON value in a ledger line')
    return _boundary(text, match.end())


def _decode(text, i):
    try:
        return _DECODER.raw_decode(text, i)
    except ValueError as error:
        raise IntegrityError(f'malformed allowlisted JSON value in a ledger line: {error}') from error


def scan_object(text, i, schema, skipped):
    """Decode only the members of the JSON object at ``text[i]`` named in ``schema`` (None = decode
    the value; a dict = scan the nested object with that schema); every other member's value is
    validated and skipped unread and its name counted in ``skipped``. Any grammar error or duplicate
    member raises IntegrityError (the ledger is refused). Returns (object, end index)."""
    i = _ws(text, i)
    if not text.startswith('{', i):
        raise IntegrityError('expected a JSON object in a ledger line')
    i = _ws(text, i + 1)
    out, seen = {}, set()
    if text.startswith('}', i):
        return out, i + 1
    while True:
        if not text.startswith('"', i):
            raise IntegrityError('expected a member name in a ledger line')
        try:
            key, i = scanstring(text, i + 1)
        except ValueError as error:
            raise IntegrityError(f'malformed member name in a ledger line: {error}') from error
        if key in seen:
            raise IntegrityError(f'duplicate member {key!r} in a ledger line')
        seen.add(key)
        i = _ws(text, i)
        if not text.startswith(':', i):
            raise IntegrityError("expected ':' in a ledger line")
        i = _ws(text, i + 1)
        if key in schema and schema[key] is None:
            out[key], i = _decode(text, i)
        elif key in schema:
            if text.startswith('null', i):
                out[key], i = None, _boundary(text, i + 4)
            else:
                out[key], i = scan_object(text, i, schema[key], skipped)
        else:
            i = _skip(text, i)
            skipped[key] = skipped.get(key, 0) + 1
        i = _ws(text, i)
        if text.startswith(',', i):
            i = _ws(text, i + 1)
            continue
        if text.startswith('}', i):
            return out, i + 1
        raise IntegrityError("expected ',' or '}' in a ledger line")


def scan_line(line, schema, skipped):
    try:
        text = line.decode('utf-8') if isinstance(line, bytes) else line
    except UnicodeDecodeError as error:
        raise IntegrityError('a ledger line is not UTF-8') from error
    row, end = scan_object(text, 0, schema, skipped)
    if text[_ws(text, end):]:
        raise IntegrityError('trailing data after a ledger row')
    return row


def read_ledger(path, *, manifest_path, ledger_sha256, manifest_sha256, runtime_sha256):
    """Allowlisted, pinned, read-only S6 ledger: (header, decisions, record)."""
    path, manifest_path = Path(path), Path(manifest_path)
    if path.name in FORBIDDEN_FILES or manifest_path.name in FORBIDDEN_FILES:
        raise LedgerRefused(f'forbidden file requested: {path.name}')
    if not (path.name.startswith('ledger') and path.suffix == '.jsonl'):
        raise LedgerRefused(f'only a ledger*.jsonl file may be read, not {path.name}')
    if manifest_path.name != 'manifest.json' or manifest_path.parent.resolve() != path.parent.resolve():
        raise LedgerRefused('the pinned manifest.json must sit next to the ledger')
    for value in (ledger_sha256, manifest_sha256, runtime_sha256):
        if not (isinstance(value, str) and len(value) == 64):
            raise LedgerRefused('ledger, manifest and runtime sha256 pins are required')
    before = {'ledger': file_sha256(path), 'manifest': file_sha256(manifest_path)}
    if before != {'ledger': ledger_sha256, 'manifest': manifest_sha256}:
        raise LedgerRefused('S6 ledger or manifest differs from its registration pin (FAILED_INTEGRITY)')
    header, decisions, kinds, skipped = None, [], {}, {}
    with path.open('rb') as stream:
        for seq, line in enumerate(stream):
            row = scan_line(line, HEADER_SCHEMA if seq == 0 else DECISION_SCHEMA, skipped)
            kind = row.get('kind')
            kinds[kind] = kinds.get(kind, 0) + 1
            if seq == 0:
                if kind != 'header':
                    raise IntegrityError('ledger header missing')
                header = {k: row[k] for k in ALLOWED_HEADER if k in row}
            elif kind == 'decision':
                if 'result' not in row:
                    row['result'] = None
                decisions.append(row)
            else:
                raise IntegrityError(f'ledger contains a {kind!r} row (halted ledger is never read)')
    after = {'ledger': file_sha256(path), 'manifest': file_sha256(manifest_path)}
    if after != before:
        raise IntegrityError('S6 ledger or manifest changed while it was read')
    if header is None or header.get('runtime_sha256') != runtime_sha256:
        raise IntegrityError('ledger header runtime_sha256 differs from the registered identity')
    wrong = sum(d.get('runtime_sha256') != runtime_sha256 for d in decisions)
    if wrong:
        raise IntegrityError(f'{wrong} decision rows carry another runtime_sha256')
    record = {'sha256_before': before, 'sha256_after': after, 'row_kinds': kinds, 'decisions': len(decisions),
              'fields_decoded': {'header': list(HEADER_SCHEMA), 'decision': list(ALLOWED_DECISION),
                                 'result': list(ALLOWED_RESULT)},
              'skipped_unread_members': dict(sorted(skipped.items())), 'parsing': PARSING_DISCLOSURE}
    return header, decisions, record


def by_pa(decisions):
    """[(pa_id, rows)] with each PA one contiguous run whose decision indices are 0..n-1."""
    out, seen = [], set()
    for row in decisions:
        if not out or out[-1][0] != row['pa_id']:
            if row['pa_id'] in seen:
                raise IntegrityError(f'ledger rows of PA {row["pa_id"]} are not contiguous')
            seen.add(row['pa_id'])
            out.append((row['pa_id'], []))
        out[-1][1].append(row)
    for pa_id, rows in out:
        if [r['decision_index'] for r in rows] != list(range(len(rows))):
            raise IntegrityError(f'PA {pa_id}: decision indices are not the ordered prefix 0..n-1')
    return out


class _StartRule:
    """``PolicyRuntime.start_population`` / ``_hand`` bound to the frozen B0 artifacts, without a ledger."""
    _hand = PolicyRuntime._hand
    start_population = PolicyRuntime.start_population

    def __init__(self, bc, support, hands):
        self.bc, self.reference, self.hand_registry, self.support_check = bc, MaskedReference(bc, support), hands, None


def e0_from_ledger(pas, bc, support, hands):
    """{pa_id: (in_E0, reason)} from the first row's pre-decision fields and frozen TRAIN artifacts."""
    rule, out = _StartRule(bc, support, hands), {}
    for pa_id, rows in pas:
        first = rows[0]
        if first['history_actions']:
            raise IntegrityError(f'PA {pa_id}: first decision has a history')
        state = PAState(int(first['balls']), int(first['strikes']), str(first['pitcher']), str(first['batter_side']),
                        (), str(first.get('context_key') or ''))
        out[pa_id] = rule.start_population(state, first.get('pitcher_hand'))
    return out


def _prefix(rows):
    statuses = [r['status'] for r in rows]
    k = next((i for i, s in enumerate(statuses) if s not in EVALUATED), len(rows))
    return k, k < len(rows) and statuses[k] == POSITIVITY


def pa_weights(rows, denominators=None, *, check_recorded=True):
    """Per policy: rho of the recursion steps (evaluated prefix, then rho = 0 at a positivity node),
    residual W_t and W_T. ``denominators``: {request_id: pi_b_hat_new(a_t)} for SUPPORTED rows;
    None = the recorded B0 logging law (and the recorded rho is checked when ``check_recorded``)."""
    k, positivity = _prefix(rows)
    out = {}
    for policy in POLICIES:
        rhos = []
        for row in rows[:k]:
            result = row['result']
            if not isinstance(result, dict):
                raise IntegrityError('an evaluated decision without a recorded result')
            a = result['logged_index']
            if type(a) is not int:
                raise IntegrityError('an evaluated decision without a logged action index')
            pi = float(result[policy][a])
            if row['status'] == 'OUTSIDE_POLICY_SUPPORT':
                if pi != 0:
                    raise IntegrityError('OUTSIDE_POLICY_SUPPORT row with policy mass on the logged action')
                rho = 0.
            else:
                logging = float(result['logging'][a])
                if not logging > 0:
                    raise IntegrityError('SUPPORTED row whose B0 logging law gives the action no mass')
                if check_recorded and denominators is None:
                    recorded = result.get(f'rho_{policy}')
                    if not (isinstance(recorded, float) and np.isclose(recorded, pi / logging, rtol=1e-12, atol=0)):
                        raise IntegrityError(f'{policy}: recorded ratio differs from the recorded probabilities')
                denominator = logging if denominators is None else float(denominators[row['request_id']])
                if not denominator > 0:
                    raise IntegrityError('a new pi_b_hat gives a SUPPORTED logged action no mass')
                rho = pi / denominator
            rhos.append(rho)
        if positivity:
            rhos.append(0.)
        residual = np.cumprod(rhos) if rhos else np.zeros(0)
        out[policy] = {'rho': rhos, 'residual': residual, 'w_T': float(residual[-1]) if len(residual) else 1.}
    return out


def r1b_logging(decisions, vector_bc):
    """Max |VectorBC(BC_P) - recorded logging vector| over every decision with a recorded result."""
    from .behavior_model import Keys, side_code
    rows = [d for d in decisions if isinstance(d.get('result'), dict) and d['result'].get('logging') is not None]
    if not rows:
        return {'rows': 0, 'max_abs_diff': None}
    keys = Keys([int(d['pitcher']) for d in rows], side_code([d['batter_side'] for d in rows]),
                [int(d['balls']) for d in rows], [int(d['strikes']) for d in rows],
                [str(d['history_actions'][-1]) if d['history_actions'] else '<START>' for d in rows])
    predicted = vector_bc.batch_keys(keys)
    recorded = np.array([d['result']['logging'] for d in rows], dtype=np.float64)
    if recorded.shape != predicted.shape:
        raise IntegrityError('recorded logging vectors are not over the registered 18 codes')
    return {'rows': len(rows), 'max_abs_diff': float(np.abs(predicted - recorded).max())}


def game_of(pa_id):
    return int(str(pa_id).split(':')[0])


def ess(weights, games):
    return {'pa': effective_sample_size(weights), 'game': effective_sample_size(weights, games)}


def r2_reproduction(pas, e0, expected, rtol):
    """ESS of the recorded B0 weights over all E0 vs the sealed S6-v3 ``ess_all_e0`` (relative)."""
    members = [(pa_id, rows) for pa_id, rows in pas if e0[pa_id][0]]
    games = [game_of(pa_id) for pa_id, _ in members]
    weights = {policy: [] for policy in POLICIES}
    for _, rows in members:
        w = pa_weights(rows)
        for policy in POLICIES:
            weights[policy].append(w[policy]['w_T'])
    actual = {policy: ess(weights[policy], games) for policy in POLICIES}
    checks = {}
    for policy in POLICIES:
        for level in ('pa', 'game'):
            value, target = actual[policy][level], expected[policy][level]
            checks[f'{policy}_{level}'] = value is not None and abs(value - target) <= rtol * abs(target)
    return {'e0_pas': len(members), 'actual': actual, 'expected': expected, 'rtol': rtol, 'checks': checks,
            'pass': all(checks.values())}


def _game_ratio_ci(values, games, *, draws, seed):
    """Whole-game bootstrap CI95 of a pitch/PA-weighted mean (ratio of resampled sums)."""
    values, games = np.asarray(values, dtype=np.float64), np.asarray(games)
    if not len(values):
        return None
    unique, inverse = np.unique(games, return_inverse=True)
    g = len(unique)
    sums, counts = np.bincount(inverse, values, g), np.bincount(inverse, minlength=g).astype(np.float64)
    if g < 2:
        return None
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    chunk = max(1, min(256, 2_000_000 // g))
    for start in range(0, draws, chunk):
        n = min(chunk, draws - start)
        indices = rng.integers(0, g, size=(n, g))
        out[start:start + n] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return np.quantile(out, [.025, .975]).tolist()


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {'q50': float(np.quantile(values, .5)), 'q90': float(np.quantile(values, .9)),
            'q99': float(np.quantile(values, .99)), 'max': float(values.max())}


def weight_diagnostics(pas, e0, *, new_vectors=None, baseline=None, large, mei, ess_min, draws, seed):
    """Diagnostics of one logging law on S6 E0 (None = recorded B0). ``new_vectors``: {request_id:
    pi_b_hat_new vector (18)} for every evaluated decision of an E0 PA; ``baseline`` = the B0
    output of this function (residual weights) for the weight shift D."""
    members = [(pa_id, rows) for pa_id, rows in pas if e0[pa_id][0]]
    games = [game_of(pa_id) for pa_id, _ in members]
    denominators, kl = None, []
    if new_vectors is not None:
        denominators = {}
        for _, rows in members:
            k, _ = _prefix(rows)
            for row in rows[:k]:
                vector = np.asarray(new_vectors[row['request_id']], dtype=np.float64)
                logging = np.asarray(row['result']['logging'], dtype=np.float64)
                denominators[row['request_id']] = float(vector[row['result']['logged_index']])
                support = vector > 0
                if (support & (logging == 0)).any():
                    kl.append(None)  # mass outside the B0 support: KL infinite, counted separately
                else:
                    kl.append(float((vector[support] * np.log(vector[support] / logging[support])).sum()))
    per_pa = [pa_weights(rows, denominators) for _, rows in members]
    out = {'e0_pas': len(members), 'policies': {}, 'residual': per_pa}
    for policy in POLICIES:
        w_T = [w[policy]['w_T'] for w in per_pa]
        rho = [r for w in per_pa for r in w[policy]['rho']]
        rho_games = [g for w, g in zip(per_pa, games) for _ in w[policy]['rho']]
        out['policies'][policy] = {
            'ess': ess(w_T, games) if members else {'pa': None, 'game': None},
            'rho_quantiles': _quantiles(rho), 'rho_gt_large_share': float(np.mean(np.asarray(rho) > large)) if rho else None,
            'w_T_quantiles': _quantiles(w_T), 'w_T_gt_large_share': float(np.mean(np.asarray(w_T) > large)) if w_T else None,
            'sc2_mean_w_T': float(np.mean(w_T)) if w_T else None,
            'sc2_mean_w_T_ci95': _game_ratio_ci(w_T, games, draws=draws, seed=seed),
            'sc2_mean_rho': float(np.mean(rho)) if rho else None,
            'sc2_mean_rho_ci95': _game_ratio_ci(rho, rho_games, draws=draws, seed=seed),
            'decisions_in_recursion': len(rho)}
    ratio = None
    c, r = (out['policies'][p]['ess'] for p in POLICIES)
    if c['pa'] is not None and r['pa']:
        ratio = {'pa': c['pa'] / r['pa'], 'game': None if not r['game'] else c['game'] / r['game']}
    out['candidate_over_reference_ess'] = ratio
    gates = {f'{p}_{level}': (out['policies'][p]['ess'][level] is not None
                              and out['policies'][p]['ess'][level] >= ess_min[level])
             for p in POLICIES for level in ('pa', 'game')}
    out['E5'] = {'thresholds': ess_min, 'checks': gates, 'pass': bool(members) and all(gates.values())}
    finite = [v for v in kl if v is not None]
    infinite = len(kl) - len(finite)
    finite_mean = float(np.mean(finite)) if finite else None
    # The registered statistic is the mean over every decision: infinite when any decision puts new
    # mass outside the B0 support (written as the string 'inf' in JSON).
    out['kl_new_vs_b0'] = None if new_vectors is None else {
        'mean': (math.inf if infinite else finite_mean), 'finite_only_mean': finite_mean, 'decisions': len(kl),
        'infinite_decisions': infinite}
    if baseline is not None:
        shift = {}
        for policy in POLICIES:
            totals = []
            for new, old in zip(per_pa, baseline['residual']):
                a, b = new[policy]['residual'], old[policy]['residual']
                if len(a) != len(b):
                    raise IntegrityError('baseline and new recursion lengths differ')
                totals.append(float(np.abs(a - b).sum()))
            shift[policy] = float(np.mean(totals)) if totals else None
        paired = None if any(v is None for v in shift.values()) else shift['candidate'] + shift['reference']
        out['weight_shift_D'] = {**shift, 'paired': paired, 'mei': mei,
                                 'paired_below_mei': None if paired is None else paired < mei,
                                 'reading': 'fixed mean contrast only: below MEI means the S6 empirical-mean contrast '
                                            '(point and each bound endpoint) moves by < MEI WE/PA under the bounded-'
                                            'residual assumption; no CI-based verdict or ESS gate guarantee'}
    return out


def public(diagnostics):
    """JSON-ready diagnostics (residual arrays dropped)."""
    return {k: v for k, v in diagnostics.items() if k != 'residual'}
