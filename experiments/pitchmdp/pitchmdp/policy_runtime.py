"""Policy runtime with an unsupported-request denominator ledger (ML-POLICY-RUNTIME-v2).

One call path: ``PolicyRuntime.submit(request)`` evaluates the full-vocabulary TRAIN BC
logging law, the masked reference, an optional pinned candidate, and appends exactly one
decision row per request ID to a hash-chained JSONL ledger before returning or raising.
It computes per-decision probabilities and ratios only. It is not an OPE runner: no
value, no estimand, no population value (always ``None``) and no 2026 data selection.

v2 (COOP-018, D93): cause-split no-action refusals (D-3), the TRAIN single-hand pitcher registry
(D-8), a candidate-mode context check before any data refusal, the logging-positivity refusal
before the candidate search, an explicit abort row, and the reference-continuation Q recorded
for the DR estimator from a separate evaluation seed when registered (M-7).

v3 (COOP-021/022, opt-in ``positivity_record=True``; the default keeps v2 byte for byte): the
logging-positivity check runs AFTER the candidate computation, and the refused row records the
policies, Q and ``rho_candidate = rho_reference = 0.0`` (both policies give the logged action no
mass, so the true ratio is 0 under any logging law that played it; no 0/0 is formed). Status,
stickiness and the refusal denominators are unchanged.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .matrix_data import canonical_hash
from .policy_artifacts import (IntegrityError, Unsupported, TrainBCArtifact, _canonical_bytes, bc_payload,
                               normalize_hand, support_table_payload)
from .planner import OUTCOMES
from .rollout_policy import PAState

SUPPORTED = 'SUPPORTED'
OUTSIDE_POLICY_SUPPORT = 'OUTSIDE_POLICY_SUPPORT'  # valid rho=0, logged action off the policy mask
UNKNOWN_PITCHER = 'UNSUPPORTED_UNKNOWN_PITCHER'
EMPTY_SUPPORT = 'UNSUPPORTED_EMPTY_SUPPORT'
LOGGING_POSITIVITY = 'UNSUPPORTED_LOGGING_POSITIVITY'
MID_PA = 'UNSUPPORTED_MID_PA'
NO_LOGGED_ACTION = 'UNSUPPORTED_NO_LOGGED_ACTION'  # registered no-pitch call (automatic ball/strike): no choice
MISSING_LABEL = 'UNSUPPORTED_MISSING_ACTION_LABEL'  # a real pitch whose type label is missing (D-3)
INCOMPLETE_START = 'UNSUPPORTED_INCOMPLETE_START'  # first recorded decision is not the 0-0 PA start
INCONSISTENT_HISTORY = 'UNSUPPORTED_INCONSISTENT_HISTORY'  # count does not follow the recorded previous outcome
PITCHER_HAND = 'UNSUPPORTED_PITCHER_HAND'  # TRAIN hand ambiguous, request hand missing, or differs (D-8)
FAILED_INTEGRITY = 'FAILED_INTEGRITY'
FAILED_RUNTIME = 'FAILED_RUNTIME'  # e.g. BudgetExceeded: fatal, recorded, not a refusal
EVALUATED = (SUPPORTED, OUTSIDE_POLICY_SUPPORT)
FATAL = (FAILED_INTEGRITY, FAILED_RUNTIME)
STATUSES = (SUPPORTED, OUTSIDE_POLICY_SUPPORT, UNKNOWN_PITCHER, EMPTY_SUPPORT, LOGGING_POSITIVITY,
            MID_PA, NO_LOGGED_ACTION, MISSING_LABEL, INCOMPLETE_START, INCONSISTENT_HISTORY, PITCHER_HAND,
            FAILED_INTEGRITY, FAILED_RUNTIME)
CONTRACT = 'ML-POLICY-RUNTIME-v2'
CONTRACT_V3 = 'ML-POLICY-RUNTIME-v3'  # positivity rows record pi, Q and rho = 0 (COOP-021/022)
ATOL = 1e-9
NO_PITCH = '<NO_PITCH>'  # logged label AND history action of a registered no-pitch row
MISSING = '<MISSING_LABEL>'  # logged label AND history action of a real pitch without a type label
SENTINELS = (NO_PITCH, MISSING)
HANDS = ('L', 'R')
AMBIGUOUS = 'AMBIGUOUS'


def next_count(balls, strikes, outcome):
    """Count after a non-terminal recorded outcome; None if the outcome ends the PA; 'unknown' is uncheckable."""
    if outcome == 'unknown':
        return 'unknown'
    if outcome == 'ball':
        return None if balls == 3 else (balls + 1, strikes)
    if outcome == 'strike':
        return None if strikes == 2 else (balls, strikes + 1)
    if outcome == 'foul':
        return balls, min(strikes + 1, 2)
    if outcome in OUTCOMES:
        return None  # ball in play or hit by pitch ends the PA
    raise IntegrityError(f'unknown outcome label {outcome!r}')


@dataclass(frozen=True)
class DecisionRequest:
    """Caller-supplied pre-decision context. ``logged_action`` is the observed label (or None
    for a pre-pitch service call, or a ``SENTINELS`` label for a row without a usable choice);
    it is only mapped for ratios, never substituted. ``pitcher_hand`` is the request row's
    p_throws (pre-decision; None when missing), compared with the pinned TRAIN hand registry."""
    request_id: str
    pa_id: str
    decision_index: int
    state: PAState
    logged_action: str | None
    runtime_sha256: str
    pitcher_hand: str | None = None

    def fingerprint(self):
        s = self.state
        return canonical_hash({'request_id': self.request_id, 'pa_id': self.pa_id,
            'decision_index': self.decision_index, 'logged_action': self.logged_action,
            'runtime_sha256': self.runtime_sha256, 'context_key': s.context_key, 'pitcher': s.pitcher,
            'pitcher_hand': self.pitcher_hand, 'batter_side': s.batter_side, 'balls': int(s.balls),
            'strikes': int(s.strikes),
            'history': [[h.action, h.outcome, h.balls, h.strikes, list(h.physics)] for h in s.history]})


_CRITICAL = {'depth': 0, 'deferred': None}


@contextmanager
def critical_section():
    """A ledger append runs inside. An interrupt requested meanwhile through ``defer_or_raise`` (the
    runner's hang guard) is raised only after the row is complete on disk AND in memory. CPython
    runs signal handlers in the main thread, which is the appending thread, so this holds whichever
    OS thread (e.g. a torch worker) received the signal; a per-thread signal mask would not."""
    _CRITICAL['depth'] += 1
    try:
        yield
    finally:
        _CRITICAL['depth'] -= 1
        error = _CRITICAL['deferred'] if _CRITICAL['depth'] == 0 else None
        if error is not None:
            _CRITICAL['deferred'] = None
            raise error


def defer_or_raise(error):
    """For signal handlers: raise ``error`` now, or right after the ledger append in progress."""
    if _CRITICAL['depth']:
        _CRITICAL['deferred'] = error
        return
    raise error


class Ledger:
    """Append-only JSONL; each row carries the previous row hash (tamper/truncation evident).

    ponytail: single-writer file; concurrent writers need an OS lock (fcntl) upgrade.
    """
    def __init__(self, path, header):
        self.path, self.rows = Path(path), []
        self.header = json.loads(_canonical_bytes({'kind': 'header', **header}))
        if self.path.exists():
            self._replay()
            if {k: v for k, v in self.rows[0].items() if k not in ('seq', 'prev', 'sha256')} != self.header:
                raise IntegrityError('ledger header pins differ from this runtime')
        else:
            self._append(self.header)

    def _replay(self):
        raw = self.path.read_bytes()
        if not raw.endswith(b'\n'):
            raise IntegrityError('ledger truncated mid-row; refuse to repair')
        prev = None
        for seq, line in enumerate(raw.splitlines(keepends=True)):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise IntegrityError(f'ledger row {seq} malformed') from error
            body = {k: v for k, v in row.items() if k != 'sha256'}
            if (row.get('seq') != seq or row.get('prev') != prev or _canonical_bytes(row) != line
                    or row.get('sha256') != canonical_hash(body)):
                raise IntegrityError(f'ledger row {seq} tampered, reordered or not canonical')
            prev = row['sha256']
            self.rows.append(row)
        if not self.rows or self.rows[0].get('kind') != 'header':
            raise IntegrityError('ledger header missing')

    def _append(self, record):
        body = {**record, 'seq': len(self.rows), 'prev': self.rows[-1]['sha256'] if self.rows else None}
        row = {**body, 'sha256': canonical_hash(body)}
        with critical_section():  # a hang-guard interrupt lands after the row is on disk AND in memory
            with self.path.open('ab') as stream:
                stream.write(_canonical_bytes(row))
                stream.flush()
                os.fsync(stream.fileno())
            self.rows.append(row)
        return row

    def decisions(self):
        return [r for r in self.rows if r['kind'] == 'decision']


class PolicyRuntime:
    """Wire a reloaded TRAIN BC, its support table and an optional candidate to a ledger.

    candidate: ``callable(state) -> probs[|V|]`` or ``(probs, q_record)`` where ``q_record`` holds
    the reference-continuation Q for the DR estimator (``q``, ``mc_se``, ``q_planning``,
    ``planning_diff_se``, ``source``), plus ``candidate_identity``; no default candidate or tau.
    hand_registry: pinned {pitcher: 'L'|'R'|'AMBIGUOUS'} over the BC pitchers (D-8), or None for
    synthetic runtimes without a hand check (the <=2025 runner always passes one).
    context_check: ``callable(state, pitcher_hand)`` raising ``IntegrityError`` when a candidate-mode
    request is not the bound context row (runs before any data refusal).
    support_check: ``callable(state, mask)`` raising ``IntegrityError`` when the pinned support
    table disagrees with the bound delivery pools (``BoundComponents.check_support``).
    context_digest: ``callable(state) -> sha256 | None`` of the bound context row; it joins the
    request fingerprint so a rebinding with other row values cannot replay stored rows.
    provenance: extra pins recorded in the ledger header (e.g. the style snapshot sha and as-of).
    """
    def __init__(self, bc_artifact, support, support_sha256, ledger_path, *, candidate=None,
                 candidate_identity=None, predictor_identity=None, support_check=None, context_digest=None,
                 context_check=None, hand_registry=None, hand_registry_sha256=None, provenance=None,
                 positivity_record=False):
        if (candidate is None) != (candidate_identity is None):
            raise IntegrityError('a candidate needs a pinned identity (and vice versa)')
        if (hand_registry is None) != (hand_registry_sha256 is None):
            raise IntegrityError('a hand registry needs its pin (and vice versa)')
        self.support_check, self.context_digest, self.context_check = support_check, context_digest, context_check
        self.components = self.improvement = None
        # Own private copies whose content re-hashes to the pins; later caller mutation
        # of the artifact/support objects cannot change the law behind this runtime sha.
        payload = bc_payload(bc_artifact.bc, bc_artifact.provenance)
        if canonical_hash(payload) != bc_artifact.sha256:
            raise IntegrityError('BC state differs from its pinned identity')
        bc_artifact = TrainBCArtifact(payload, bc_artifact.file_sha256)
        rows = [(p, s, np.asarray(m, dtype=bool)) for (p, s), m in support.items()]
        if canonical_hash(support_table_payload(bc_artifact, rows)) != support_sha256:
            raise IntegrityError('support table differs from its pinned identity')
        support = {key: mask.copy() for key, mask in support.items()}
        for mask in support.values(): mask.setflags(write=False)
        if hand_registry is not None:
            hand_registry = dict(hand_registry)
            if set(hand_registry) != set(bc_artifact.bc.pitchers) or not set(hand_registry.values()) <= {*HANDS, AMBIGUOUS}:
                raise IntegrityError('hand registry must cover exactly the BC pitchers with L/R/AMBIGUOUS')
            single = {p for p, hand in hand_registry.items() if hand != AMBIGUOUS}
            if {p for p, _ in support} != single:
                raise IntegrityError('support table pitchers must equal the single-hand registry pitchers')
        self.hand_registry = hand_registry
        self.positivity_record = bool(positivity_record)
        self.bc, self.vocabulary = bc_artifact.bc, bc_artifact.vocabulary
        self.support, self.candidate = support, candidate
        self.reference = MaskedReference(self.bc, support)
        self.pins = json.loads(_canonical_bytes({'bc': bc_artifact.identity(), 'support_table_sha256': support_sha256,
                     'logging_law': {'name': 'train_bc_full_vocabulary', 'bc_sha256': bc_artifact.sha256},
                     'reference': {'name': 'train_bc_masked_to_intervention_support', 'bc_sha256': bc_artifact.sha256,
                                   'support_table_sha256': support_sha256},
                     'hand_registry_sha256': hand_registry_sha256, 'statuses': list(STATUSES),
                     'candidate': candidate_identity, 'predictor': predictor_identity, 'provenance': provenance,
                     'population_value': None, 'contract': CONTRACT,
                     **({'contract': CONTRACT_V3, 'positivity_record': True} if self.positivity_record else {})}))
        self.sha256 = canonical_hash(self.pins)
        self.ledger = Ledger(ledger_path, {'runtime_sha256': self.sha256, 'pins': self.pins})
        self.by_request, self.pa = {}, {}
        for row in self.ledger.decisions():
            self._index(row)
        self.halted = any(r['kind'] in ('conflict', 'malformed', 'aborted') or r.get('status') in FATAL
                          for r in self.ledger.rows)

    def _index(self, row):
        self.by_request[row['request_id']] = row
        pa = self.pa.setdefault(row['pa_id'], {'next': 0, 'refused': None, 'last_logged': None})
        pa['next'] = row['decision_index'] + 1
        pa['last_logged'] = row['logged_action']
        if row['status'] not in EVALUATED and pa['refused'] is None:
            pa['refused'] = (row['decision_index'], row['status'])

    # ------------------------------------------------------------ evaluation
    def _row(self, probs, mask, name):
        p = np.asarray(probs, dtype=np.float64)
        if (p.shape != (len(self.vocabulary),) or not np.isfinite(p).all() or (p < 0).any()
                or abs(p.sum() - 1) > ATOL or (p[~mask] != 0).any()):
            raise IntegrityError(f'{name}: invalid probabilities or mass outside its support')
        return p

    def _q_record(self, record, mask):
        """Reference-continuation Q on the mask (None off the mask) for the DR estimator."""
        if not isinstance(record, dict) or set(record) != {'q', 'mc_se', 'q_planning', 'planning_diff_se', 'source'}:
            raise IntegrityError('candidate Q record fields differ')
        out = {'q_source': record['source']}
        for key, name, finite in (('q', 'q_reference', True), ('mc_se', 'q_mc_se', False),
                                  ('q_planning', 'q_planning', True), ('planning_diff_se', 'q_planning_diff_se', False)):
            v = np.asarray(record[key], dtype=np.float64)
            if v.shape != mask.shape or np.isfinite(v[~mask]).any() or (finite and not np.isfinite(v[mask]).all()):
                raise IntegrityError(f'{key} must be finite exactly on the policy mask')
            out[name] = [float(x) if m and np.isfinite(x) else None for x, m in zip(v, mask)]
        return out

    def _hand(self, state, hand):
        """D-8: the request's pitcher hand against the pinned TRAIN single-hand registry."""
        if self.hand_registry is None:
            return
        registered = self.hand_registry[state.pitcher]
        if registered == AMBIGUOUS:
            raise Unsupported(PITCHER_HAND, 'ambiguous_train')
        hand = normalize_hand(hand)
        if hand not in HANDS:
            raise Unsupported(PITCHER_HAND, 'missing_or_invalid')
        if hand != registered:
            raise Unsupported(PITCHER_HAND, 'mismatch')

    def start_population(self, state, hand):
        """E0/S membership (D-2, D-5) from the first row's pre-decision fields and frozen TRAIN
        artifacts only: 0-0 count, known pitcher, registered single hand, nonempty BC support and
        nonempty intervention mask. Independent of the logged action and of the refusal order."""
        if (state.balls, state.strikes) != (0, 0):
            return False, INCOMPLETE_START
        if self.bc.fallback(state):
            return False, UNKNOWN_PITCHER
        try:
            self._hand(state, hand)
        except Unsupported as refusal:
            return False, f'{PITCHER_HAND}:{refusal}'
        if not self.bc.support(state).any():
            return False, EMPTY_SUPPORT
        mask = self.reference.support(state)
        if self.support_check is not None:
            self.support_check(state, mask)
        if not mask.any():
            return False, EMPTY_SUPPORT
        return True, None

    def _evaluate(self, request, pa):
        state = request.state
        if request.runtime_sha256 != self.sha256:
            raise IntegrityError('request pinned to another runtime/policy identity')
        if request.logged_action not in (None, *SENTINELS) and request.logged_action not in self.vocabulary:
            raise IntegrityError(f'unknown logged action label {request.logged_action!r}')
        # Strict pre-decision history: exactly the earlier decisions of this PA, in order.
        if request.decision_index != pa['next'] or len(state.history) != request.decision_index:
            raise IntegrityError('decision index/history length is not the next strictly prior decision')
        if pa['next'] and state.history[-1].action != pa['last_logged']:
            raise IntegrityError('history does not end with the previously logged action of this PA')
        if self.context_check is not None:  # candidate mode: an unbound/mismatched row is never a data refusal
            self.context_check(state, request.pitcher_hand)
        if pa['refused'] is not None:
            raise Unsupported(MID_PA, f'PA refused at decision {pa["refused"][0]}: {pa["refused"][1]}')
        # Pre-decision data refusals (legitimate, in the denominator, sticky for the rest of the PA).
        if request.logged_action == NO_PITCH:
            raise Unsupported(NO_LOGGED_ACTION, 'registered no-pitch call: no pitcher choice at this row')
        if request.logged_action == MISSING:
            raise Unsupported(MISSING_LABEL, 'pitch without a type label')
        if request.decision_index == 0 and (state.balls, state.strikes) != (0, 0):
            raise Unsupported(INCOMPLETE_START, f'first decision at {state.balls}-{state.strikes}')
        if state.history:
            past = state.history[-1]
            follows = next_count(past.balls, past.strikes, past.outcome)
            if follows == 'unknown':  # a real pitch whose outcome the 10-class encoder cannot map
                raise Unsupported(INCONSISTENT_HISTORY, f'uncheckable previous outcome after {past.action}')
            if follows != (state.balls, state.strikes):
                raise Unsupported(INCONSISTENT_HISTORY, f'{past.balls}-{past.strikes} {past.outcome} -> '
                                  f'{state.balls}-{state.strikes}')
        if self.bc.fallback(state):
            raise Unsupported(UNKNOWN_PITCHER, 'no TRAIN history for this pitcher; league fallback refused')
        self._hand(state, request.pitcher_hand)
        logging_mask = self.bc.support(state)
        if not logging_mask.any():
            raise Unsupported(EMPTY_SUPPORT, 'no TRAIN BC support for this pitcher')
        logging = self._row(self.bc.probabilities(state), logging_mask, 'logging law')
        mask = self.reference.support(state)
        if self.support_check is not None:  # before the empty-support refusal: an empty mask can hide a pool mismatch
            self.support_check(state, mask)
        if not mask.any():
            raise Unsupported(EMPTY_SUPPORT, 'no intervention-supported action')
        if (mask & ~logging_mask).any():
            raise IntegrityError('intervention mask outside the TRAIN BC support')
        reference = self._row(self.reference.probabilities(state), mask, 'reference')
        a = None
        positivity = False
        if request.logged_action is not None:
            a = self.vocabulary.index(request.logged_action)
            positivity = logging[a] == 0  # estimated logging law gives the observed action no mass (always off the mask)
            if positivity and not self.positivity_record:
                raise Unsupported(LOGGING_POSITIVITY, request.logged_action)
        candidate, q_record = None, {}
        if self.candidate is not None:
            out = self.candidate(state)  # probabilities, or (probabilities, reference-Q record)
            candidate = self._row(out[0] if isinstance(out, tuple) else out, mask, 'candidate')
            if isinstance(out, tuple):
                q_record = self._q_record(out[1], mask)
        result = {'mask': mask.tolist(), 'logging': logging.tolist(), 'reference': reference.tolist(),
                  'candidate': None if candidate is None else candidate.tolist(),
                  'logging_mass_on_mask': float(logging[mask].sum()),
                  'logged_index': None, 'rho_reference': None, 'rho_candidate': None, **q_record}
        if a is None:
            return SUPPORTED, result
        if positivity:  # v3: pi(a) = 0 for both policies, so rho = 0 exactly; recorded, never divided
            if mask[a] or reference[a] != 0 or (candidate is not None and candidate[a] != 0):
                raise IntegrityError('logging-positivity action with policy mass or on the mask')
            result.update(logged_index=a, rho_reference=0.0, rho_candidate=None if candidate is None else 0.0)
            refusal = Unsupported(LOGGING_POSITIVITY, request.logged_action)
            refusal.result = result
            raise refusal
        result.update(logged_index=a, rho_reference=float(reference[a] / logging[a]),
                      rho_candidate=None if candidate is None else float(candidate[a] / logging[a]))
        return (SUPPORTED if mask[a] else OUTSIDE_POLICY_SUPPORT), result

    def submit(self, request):
        """Record-then-return. Duplicates with identical content replay the stored row;
        a changed retry of the same request ID is an integrity conflict (audited, not counted)."""
        try:
            if not isinstance(request, DecisionRequest) or not isinstance(request.state, PAState):
                raise IntegrityError('DecisionRequest with a PAState required')
            if (not isinstance(request.request_id, str) or not request.request_id or not isinstance(request.pa_id, str)
                    or not request.pa_id or type(request.decision_index) is not int
                    or not (request.pitcher_hand is None or isinstance(request.pitcher_hand, str))):
                raise IntegrityError('request/PA IDs must be nonempty strings, decision_index an int, hand str/None')
            fingerprint = request.fingerprint()
            context = self.context_digest(request.state) if self.context_digest is not None else None
            if context is not None:
                fingerprint = canonical_hash({'request': fingerprint, 'context_sha256': context})
        except Exception as failure:  # no usable ID: audit row outside the denominators, then halt
            self.ledger._append({'kind': 'malformed', 'detail': f'{type(failure).__name__}: {failure}'})
            self.halted = True
            raise IntegrityError('malformed request') from failure
        if request.request_id in self.by_request:
            stored = self.by_request[request.request_id]
            if stored['fingerprint'] == fingerprint:
                if stored['status'] in FATAL:
                    raise IntegrityError('replayed request had failed integrity: ' + stored['detail'])
                return stored
            self.ledger._append({'kind': 'conflict', 'request_id': request.request_id, 'fingerprint': fingerprint,
                                 'stored_fingerprint': stored['fingerprint']})
            self.halted = True
            raise IntegrityError('request ID reused with different content')
        pa = self.pa.get(request.pa_id, {'next': 0, 'refused': None, 'last_logged': None})
        error, status, result, detail = None, None, None, ''
        if self.halted:
            status, detail = FAILED_INTEGRITY, 'run halted by an earlier integrity failure or abort'
            error = IntegrityError(detail)
        else:
            try:
                status, result = self._evaluate(request, pa)
            except Unsupported as refusal:
                status, detail, result = refusal.status, str(refusal), getattr(refusal, 'result', None)
            except (IntegrityError, ValueError) as failure:
                status, detail, error = FAILED_INTEGRITY, str(failure), failure
            except Exception as failure:  # record before propagating; never lose the audit row
                status, detail, error = FAILED_RUNTIME, f'{type(failure).__name__}: {failure}', failure
        s = request.state
        row = self.ledger._append({'kind': 'decision', 'request_id': request.request_id, 'pa_id': request.pa_id,
            'decision_index': request.decision_index, 'fingerprint': fingerprint, 'status': status, 'detail': detail,
            'runtime_sha256': self.sha256, 'context_key': s.context_key, 'pitcher': s.pitcher,
            'pitcher_hand': request.pitcher_hand, 'batter_side': s.batter_side, 'balls': int(s.balls),
            'strikes': int(s.strikes), 'history_actions': [h.action for h in s.history],
            'logged_action': request.logged_action, 'context_sha256': context, 'result': result})
        self._index(row)
        if status in FATAL:
            self.halted = True
            if status == FAILED_RUNTIME:
                raise error
            raise IntegrityError(detail) from error
        return row

    def abort(self, reason):
        """Stage abort (wall cap, hang guard): an audit row outside the denominators; the run is HALTED."""
        self.ledger._append({'kind': 'aborted', 'detail': str(reason)})
        self.halted = True

    def summary(self):
        """Denominators from the ledger alone: every request ID once, every PA once."""
        rows = self.ledger.decisions()
        pas, first = {}, {}
        for row in rows:  # ledger order = decision order within each PA
            status = pas.setdefault(row['pa_id'], SUPPORTED)
            if row['status'] in FATAL:
                pas[row['pa_id']] = row['status']
            elif status == SUPPORTED and row['status'] not in EVALUATED:
                pas[row['pa_id']] = MID_PA if row['decision_index'] > 0 else row['status']
            if row['status'] not in EVALUATED and row['status'] != MID_PA:
                first.setdefault(row['pa_id'], row['status'])
        return {'requests': len(rows), 'request_status': dict(Counter(r['status'] for r in rows)),
                'pas': len(pas), 'pa_status': dict(Counter(pas.values())),
                'pa_first_refusal': dict(Counter(first.values())),
                'conflicts': sum(r['kind'] == 'conflict' for r in self.ledger.rows),
                'malformed': sum(r['kind'] == 'malformed' for r in self.ledger.rows),
                'aborted': sum(r['kind'] == 'aborted' for r in self.ledger.rows),
                'run_status': 'HALTED' if self.halted else 'OK',
                'ledger_rows': len(self.ledger.rows), 'ledger_head_sha256': self.ledger.rows[-1]['sha256'],
                'runtime_sha256': self.sha256, 'population_value': None}

    def verify_components(self):
        """Stage-end check before sealing results: for a candidate runtime, the pinned code,
        the bound component content and the simulator wiring are unchanged since binding."""
        if self.components is None:
            if self.candidate is not None:
                raise IntegrityError('a candidate without bound components cannot be verified')
            return True
        return self.components.verify(self.improvement)


class MaskedReference:
    """pi_ref = full TRAIN BC restricted to the frozen intervention support and renormalized.

    Same arithmetic as ``matrix_policy.SupportedBC`` but reads a pinned support table
    instead of live delivery pools. Usable as the ``bc`` of ``RolloutImprovement``.
    """
    def __init__(self, bc, support):
        self.bc, self.table, self.actions = bc, support, bc.actions

    def fallback(self, state): return self.bc.fallback(state)

    def support(self, state):
        return self.table.get((state.pitcher, state.batter_side), np.zeros(len(self.actions), bool)) & self.bc.support(state)

    def probabilities(self, state, *, frequency=False):
        p = self.bc.probabilities(state, frequency=frequency) * self.support(state)
        if p.sum() <= 0: raise ValueError('No common supported actions; abstain')
        return p / p.sum()


def paired_difference_se(sample_values, mask):
    """Per action, the s.e. of Q(a) - Q(argmax) from the common-random-number samples (0 at the
    argmax, None/NaN off the mask); the D-9 noise measure of the planning search."""
    values = np.asarray(sample_values, dtype=np.float64)
    out = np.full(len(mask), np.nan)
    if values.shape[1] < 2:
        return out
    means = np.where(mask, values.mean(axis=1), -np.inf)
    best = int(np.argmax(means))
    for i in np.flatnonzero(mask):
        out[i] = 0. if i == best else float((values[i] - values[best]).std(ddof=1) / np.sqrt(values.shape[1]))
    return out


def candidate_identity(components, support_identity, hands_identity, settings, evaluation_seed):
    """Complete ML-POLICY-IDENTITY-v1 candidate identity (context rows are not part of it; they
    enter each request fingerprint)."""
    identity = {**components.identity, 'support_table_sha256': support_identity, 'hand_registry_sha256': hands_identity,
                'search': {'policy': 'P3', 'reference': 'MaskedReference(TRAIN BC, pinned support table)',
                           'q': 'RolloutImprovement MC of the reference continuation', **settings},
                'dr_q': {'source': 'planning_reuse' if evaluation_seed is None else 'evaluation_seed',
                         'evaluation_seed': evaluation_seed}}
    identity['sha256'] = canonical_hash(identity)
    return identity


def build_runtime(bc_path, bc_sha256, support_path, support_sha256, ledger_path, *, components=None,
                  budget=None, tau=None, samples=None, pitch_cap=None, seed=None, evaluation_seed=None,
                  expected_identity_sha256=None, hand_registry=None, provenance=None, positivity_record=False):
    """Entry path: pinned BC + pinned support table (+ optional P3 candidate over bound components).

    Without ``components`` the runtime evaluates logging law and reference only. A candidate
    needs ``BoundComponents`` from ``policy_identity.bind_components``: pool, G0, WE terminal and
    cutoff come only from them, never as loose callables. tau/MC/budget have no defaults
    (unregistered). ``evaluation_seed`` (M-7) re-estimates the reference-continuation Q that the
    DR estimator uses with an independent search seed; without it the planning Q is reused as the
    declared control variate. ``hand_registry`` = (path, sha256) of the pinned TRAIN hand registry.
    The candidate identity is the complete ML-POLICY-IDENTITY-v1 identity; a registered
    ``expected_identity_sha256`` must match it exactly.
    """
    from .policy_artifacts import load_hand_registry, load_support_table, load_train_bc
    from .policy_identity import BoundComponents
    from .rollout_policy import JointSimulator, RolloutImprovement, kl_policy
    artifact = load_train_bc(bc_path, bc_sha256)
    support, support_identity = load_support_table(support_path, support_sha256, artifact)
    hands, hands_identity = (None, None) if hand_registry is None else load_hand_registry(*hand_registry, artifact)
    common = dict(hand_registry=hands, hand_registry_sha256=hands_identity, provenance=provenance,
                  positivity_record=positivity_record)
    if components is None:
        if expected_identity_sha256 is not None or evaluation_seed is not None:
            raise IntegrityError('a policy identity pin or evaluation seed needs bound candidate components')
        return PolicyRuntime(artifact, support, support_identity, ledger_path, **common)
    settings = {'tau': tau, 'samples': samples, 'pitch_cap': pitch_cap, 'seed': seed}
    if not isinstance(components, BoundComponents) or any(v is None for v in (*settings.values(), budget)):
        raise IntegrityError('candidate requires BoundComponents and explicit tau/MC/budget settings')
    if evaluation_seed is not None and (type(evaluation_seed) is not int or evaluation_seed == seed):
        raise IntegrityError('the evaluation seed must be an integer different from the planning seed')
    components.verify()
    if components.identity['bc']['bc_sha256'] != artifact.sha256:
        raise IntegrityError('bound components were built for another TRAIN BC')
    identity = candidate_identity(components, support_identity, hands_identity, settings, evaluation_seed)
    if expected_identity_sha256 is not None and identity['sha256'] != expected_identity_sha256:
        raise IntegrityError('complete policy identity differs from its registered pin')
    reference, we = MaskedReference(artifact.bc, support), components.we
    simulator = JointSimulator(components.inputs.pool, components.g0, we.terminal, budget)
    improvement = RolloutImprovement(reference, simulator, we.cutoff, samples=samples, pitch_cap=pitch_cap, seed=seed)
    improvement.policy('P3', tau=tau)  # validates tau exactly as the P3 policy does
    evaluator = None if evaluation_seed is None else RolloutImprovement(
        reference, simulator, we.cutoff, samples=samples, pitch_cap=pitch_cap, seed=evaluation_seed)

    def planning_policy(state):
        """The P3 law alone (planning Q only): what the candidate plays; no DR q-hat search."""
        q, _ = improvement.q_values(state)
        return kl_policy(q, reference.probabilities(state), reference.support(state), tau)

    def candidate(state):
        """P3 = kl_policy(Q_ref, pi_ref, M, tau) (RolloutImprovement.policy('P3')) from the planning Q,
        which is computed once; the DR q-hat of both policies (D89 §5) is the evaluation-seed Q when
        registered (M-7), else the planning Q (declared control variate)."""
        if reference.actions != artifact.vocabulary:
            raise IntegrityError('candidate action order differs from the BC vocabulary')
        mask = reference.support(state)
        q, diagnostics = improvement.q_values(state)
        p = kl_policy(q, reference.probabilities(state), mask, tau)
        q_eval, eval_diagnostics = (q, diagnostics) if evaluator is None else evaluator.q_values(state)
        off = np.where(mask, 0., np.nan)
        return p, {'q': q_eval + off, 'mc_se': eval_diagnostics['mc_se'] + off, 'q_planning': q + off,
                   'planning_diff_se': paired_difference_se(diagnostics['sample_values'], mask) + off,
                   'source': 'planning_reuse' if evaluator is None else 'evaluation_seed'}
    runtime = PolicyRuntime(artifact, support, support_identity, ledger_path, candidate=candidate,
                            candidate_identity=identity, predictor_identity=components.g0.identity,
                            support_check=components.check_support, context_digest=components.context_sha256,
                            context_check=components.check_context, **common)
    runtime.reference, runtime.improvement, runtime.components = reference, improvement, components
    runtime.evaluator, runtime.tau, runtime.planning_policy = evaluator, tau, planning_policy
    return runtime


def build_reference_pair_runtime(bc_path, bc_sha256, support_path, support_sha256, ledger_path, *, hand_registry=None,
                                 provenance=None, positivity_record=False):
    """M-10 / D89 V4(e): a candidate runtime whose candidate is an INDEPENDENT callable of the
    reference law (its own reloaded BC and support table, a separate MaskedReference). Submitted the
    same requests as the primary run, every complete PA must give a paired delta of exactly 0; the
    candidate path (request state, mask alignment, recorded rho_candidate, Q record) is exercised.
    The Q record is zero on the mask (any fixed q keeps DR unbiased; it only has to be common)."""
    from .policy_artifacts import load_hand_registry, load_support_table, load_train_bc
    artifact = load_train_bc(bc_path, bc_sha256)
    support, support_identity = load_support_table(support_path, support_sha256, artifact)
    independent = MaskedReference(load_train_bc(bc_path, bc_sha256).bc,
                                  load_support_table(support_path, support_sha256, artifact)[0])
    hands, hands_identity = (None, None) if hand_registry is None else load_hand_registry(*hand_registry, artifact)

    def candidate(state):
        mask = independent.support(state)
        zero, missing = np.where(mask, 0., np.nan), np.full(len(mask), np.nan)
        return independent.probabilities(state), {'q': zero, 'mc_se': missing, 'q_planning': zero,
                                                  'planning_diff_se': missing, 'source': 'paired_identity_zero_q'}
    return PolicyRuntime(artifact, support, support_identity, ledger_path, candidate=candidate,
                         candidate_identity={'name': 'cand=ref paired identity run (M-10, D89 V4 e)',
                                             'candidate': 'independent MaskedReference(TRAIN BC, pinned support table)'},
                         hand_registry=hands, hand_registry_sha256=hands_identity, provenance=provenance,
                         positivity_record=positivity_record)
