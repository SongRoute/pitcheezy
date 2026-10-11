"""Read-only view of the Step 5 SHADOW ledger for one fan sentence: "던지기 전 예상: 헛스윙 x% · 땅볼 y%".

What the model is. The frozen 2024 three-head model gives, for each CANDIDATE pitch type, seven outcome
probabilities of ONE pitch (they sum to 1: ball, foul, hit by pitch, swinging strike, other strike, ground ball in
play, other ball in play). It does not say which pitch type will be thrown and it is not a recommendation. This
module therefore only ever answers, AFTER a pitch, for the pitch type that WAS thrown, and only for the last pitch
of a plate appearance that ended in a key moment (``KEY_MOMENT_EVENTS_PROVISIONAL``). It never ranks pitch types,
never returns a cell of a type that was not thrown, and no text it builds names a "best" pitch.

Nothing is written, no model code is imported: stdlib ``sqlite3`` on the ledger file the shadow runner (or a
future live sidecar) appends to. Every refusal is ``None`` plus a machine-readable reason for the operator
(``explain`` / ``status``); nothing raises into a route. The principle is the one of ``rec_pairing``: an ambiguous
join shows NOTHING rather than a wrong pairing.

A pitch gets an expectation only if every rule holds (``ShadowExpectations.explain``):

 1. the display is switched on (``enabled``; default OFF) and the PA's terminal event is a key moment;
 2. exactly one ``shadow_forecasts`` row names this physical pitch ``(game_pk, at_bat_number,
    candidate_pitch_number)`` under the runner's own key ``game:PA:after{k-1}``, and it is ``scored``.
    Pitch identity: the runner numbers the pending pitch "pitches received so far + 1" and only when the feed's
    ``pitchNumber`` of the received pitches is exactly 1..n; the app's key uses the feed's ``pitchNumber`` too
    (``game_history``), so the numbers are the same number. An automatic ball/strike or a ``no_pitch`` event
    between two pitches does not move ``pitchNumber`` but does move the count, which rule 5 catches;
 3. the ledger does not disown the link: no ``matchup_changed_quarantine`` on any attempt of the plate appearance,
    no other pitcher/batter observed by any poll of that plate appearance (the runner's ``_conflict`` rule,
    available mid-game), no ``late`` order reconciliation, and, once the runner has reconciled the game, the
    pitch's label is ``had_attempt`` for exactly this attempt;
 4. the forecast row and the attempt record it cites agree (hash, outcome, leaf order, model kind);
 5. the forecast's own inputs equal the pitch as thrown, as the app tells it: pitcher, batter, inning, half, outs,
    balls, strikes, runners and score. A forecast held from before a substitution, a count correction, an
    automatic ball or a runner move is not this pitch's forecast;
 6. the cell of exactly the realized pitch type is ``scored`` in all three places that state it, and the ledger's
    own latest observation of the pitch (when it has one) names the same type;
 7. "recorded before the pitch" (``_timing``). ``live_before_pitch`` needs a decision stamped on the live observer
    clock from a live poll AND that decision (plus compute time and ``min_lead_s``) to be earlier than the wall
    time the RC store first received the pitch. A post-game replay ledger cannot prove that: it is refused unless
    ``allow_replayed``, and is then marked ``replayed_pre_pitch_inputs`` with wording that says so;
 8. only leaves with calibration evidence are shown, and only inside the probability range that evidence covers
    (``SHOWN_LEAVES``); an item outside it is dropped, never clipped. No item left: nothing.

Connection policy (measured with SQLite 3.53): a rollback-journal ledger (what the runner writes today) is opened
``mode=ro``: no side file is created, readers take a shared lock for well under a millisecond per call, and rows
committed by another process are visible. A WAL ledger is opened ``mode=ro`` only when its ``-wal`` and ``-shm``
already exist (a writer is attached, nothing new is created); with no side files it is opened ``immutable=1``
(``mode=ro`` alone would CREATE ``-wal``/``-shm`` next to a ledger we do not own); a non-empty ``-wal`` without
``-shm`` is refused, because ``immutable`` would silently skip committed rows such as a quarantine. Nothing here
can checkpoint or write. The tables are append-only (UPDATE/DELETE abort), so they are read incrementally.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
import sqlite3
import threading
from urllib.parse import quote

from .domain import PITCH_LABELS

KIND = 'step5_expectation'
LEAD_LIVE = '던지기 전 예상'
LEAD_REPLAYED = '던지기 전 정보로 본 예상'
NOTE = '2024 모델 · 참고'
NOTE_REPLAYED = '2024 모델 · 참고 · 경기 뒤 던지기 전 정보만으로 다시 계산'
# One line for the detail sheet: both leaves are per PITCH, not per swing and not per ball in play.
EXPLAIN = ('이 공 하나가 어떻게 끝날지 본 비율이에요. 헛스윙은 휘둘러서 못 맞힌 경우(파울팁 제외), '
           '땅볼은 이 공을 쳐서 땅볼 타구가 된 경우예요. 스윙했을 때만 따진 비율이 아니에요.')
BELOW_ONE = '1% 미만'

# The seven leaves in the order the runner stores ``seven_leaf`` (step5 pipeline.LEAVES; the attempt record repeats
# it as ``model.leaf_order`` and a record that states another order is refused).
LEAF_ORDER = ('ball', 'foul', 'hit_by_pitch', 'narrow_whiff', 'other_strike', 'ground_ball_in_play', 'other_in_play')
ACTIONS = ('CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'ST', 'SV')
MODEL_KIND = 'frozen_2024_comparator'  # the only model the note "2024 모델" is true for

# Leaves that may be shown, in a FIXED order (never sorted by value), with the half-open probability range
# [0, below) that results/EXP-WS-STEP5-CAL-001.json (design A, 78,136 scored rows of 2026 exposed development)
# supports. Only these two leaves were named by the registered acceptance rule (``thresholds.applies_to_leaves``);
# the other five have reliability tables but no acceptance check, so they are not shown at all.
#   narrow_whiff        bins [0,.1) n=36,140  [.1,.2) n=38,734  [.2,.3) n=3,256  [.3,.4) n=6   leaf ECE 0.0021
#   ground_ball_in_play bins [0,.1) n=59,873  [.1,.2) n=17,650  [.2,.3) n=611    [.3,.4) n=2   leaf ECE 0.0044
# Ground ball runs about 0.4 pp high on average (mean predicted 0.0765 vs base rate 0.0722). Its [.2,.3) bin has
# n=611, below the experiment's own "populated" threshold of 1,000 (gap -0.7 pp): the 30% cap is the PM's rule
# (decision B2); ``caps={'ground_ball_in_play': 0.2}`` is the stricter reading of the same evidence.
SHOWN_LEAVES = (
    {'leaf': 'narrow_whiff', 'label': '헛스윙', 'meaning': '이 공에 헛스윙', 'below': 0.30},
    {'leaf': 'ground_ball_in_play', 'label': '땅볼', 'meaning': '이 공을 쳐서 땅볼', 'below': 0.30})

# PROVISIONAL. Plate-appearance terminal ``eventType`` values that count as a key moment for THIS display only.
# docs/product/experience.md: strikeout / home run / double play are EXAMPLES of key-moment candidates, not a
# fixed list, and no exposure rule is adopted. These are the feed values ``game_history`` already knows for those
# three examples (in the 849831 capture only strikeout, home_run and grounded_into_double_play occur).
KEY_MOMENT_EVENTS_PROVISIONAL = frozenset((
    'strikeout', 'strikeout_double_play', 'home_run', 'grounded_into_double_play', 'double_play', 'triple_play',
    'sac_fly_double_play', 'sac_bunt_double_play'))

# Clock labels. ``observer_local``/``live_poll`` are what the accepted live reader (step5 reader.SnapshotReader)
# stamps; the replay label is what shadow_runner.py stamps on every decision of a post-capture replay.
LIVE_CLOCK, LIVE_SOURCE = 'observer_local', 'live_poll'
REPLAY_CLOCK, REPLAY_SOURCE = 'capture_receipt_times_replayed_post_hoc', 'replay_folder'
RECORDED_LIVE, RECORDED_REPLAYED = 'live_before_pitch', 'replayed_pre_pitch_inputs'

SITUATION = ('inning', 'half', 'outs', 'balls', 'strikes', 'bases', 'home_score', 'away_score')
# Refusal codes (operator only, never fan text). A code may carry a detail after a colon.
REASONS = (
    'disabled', 'bad_pitch_key', 'not_key_moment', 'realized_type_unknown', 'identity_unverified',
    'situation_unverified', 'ledger_missing', 'ledger_not_ready', 'ledger_busy', 'ledger_unreadable',
    'ledger_wal_unrecovered', 'ledger_too_large', 'no_forecast', 'ambiguous_forecast', 'forecast_abstained',
    'matchup_changed', 'unlinked', 'recorded_late', 'attempt_missing', 'ledger_inconsistent', 'leaf_order_unknown',
    'model_unknown', 'forecast_inputs_unknown', 'different_pitcher', 'different_batter', 'different_situation',
    'realized_type_disagrees', 'written_after_pitch_shown', 'cell_unsupported', 'cell_invalid',
    'provenance_unknown', 'replayed_not_allowed', 'pitch_first_seen_unknown', 'recorded_after_pitch_appeared',
    'above_supported_range', 'error')
# Refusals that can pass by themselves (the sidecar has not written the row yet, a read collided with a commit).
TRANSIENT = ('ledger_missing', 'ledger_not_ready', 'ledger_busy', 'ledger_unreadable', 'no_forecast',
             'pitch_first_seen_unknown', 'realized_type_disagrees')
# Words no text built here may contain (tests enforce it): this is not a recommendation or a pitch prediction.
FORBIDDEN_WORDS = ('추천', '예측 구종', '구종 예측', '최적', '가장 좋은', '던져야')


def is_key_moment(pa_event):
    """True when a plate appearance's terminal ``eventType`` is in the PROVISIONAL key-moment list."""
    return isinstance(pa_event, str) and pa_event in KEY_MOMENT_EVENTS_PROVISIONAL


def percent(probability):
    """``(pct, text)`` of a probability, rounded half up to a whole percent. Below 0.5% the text is "1% 미만"
    (pct 0): a displayed "0%" would claim an impossibility the model does not state."""
    pct = int(math.floor(probability*100+0.5))
    return (0, BELOW_ONE) if pct < 1 else (pct, f'{pct}%')


def _is_int(value):
    return type(value) is int


def _num(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _situation_thrown(situation):
    """The caller's pre-pitch situation (``game_history`` ``situation_before``) in comparable form, or None."""
    if not isinstance(situation, dict):
        return None
    bases = situation.get('bases')
    if not _is_int(bases) and isinstance(situation.get('runners'), dict):
        runners = situation['runners']
        bases = sum(bit for bit, base in ((1, 'first'), (2, 'second'), (4, 'third')) if runners.get(base))
    half = situation.get('half')
    half = {'top': 'Top', 'bot': 'Bot', 'bottom': 'Bot'}.get(half.lower()) if isinstance(half, str) else None
    out = {'inning': situation.get('inning'), 'half': half, 'outs': situation.get('outs'),
           'balls': situation.get('balls'), 'strikes': situation.get('strikes'), 'bases': bases,
           'home_score': situation.get('home_score'), 'away_score': situation.get('away_score')}
    return out if half and all(_is_int(out[k]) for k in SITUATION if k != 'half') else None


def _situation_forecast(state):
    """The same facts from the runner's ``pre_pitch_state`` (score is (home, away), bases (1B, 2B, 3B))."""
    if not isinstance(state, dict):
        return None
    bases, score, top = state.get('bases'), state.get('score'), state.get('top')
    if not (isinstance(bases, (list, tuple)) and len(bases) == 3 and all(type(b) is bool for b in bases)
            and isinstance(score, (list, tuple)) and len(score) == 2 and all(_is_int(s) for s in score)
            and type(top) is bool):
        return None
    out = {'inning': state.get('inning'), 'half': 'Top' if top else 'Bot', 'outs': state.get('outs'),
           'balls': state.get('balls'), 'strikes': state.get('strikes'),
           'bases': sum(bit for bit, on in zip((1, 2, 4), bases) if on),
           'home_score': score[0], 'away_score': score[1]}
    return out if all(_is_int(out[k]) for k in SITUATION if k != 'half') else None


def _seven(cell):
    """The seven probabilities of a scored cell, or None when they are not a probability vector."""
    leaf = cell.get('seven_leaf') if isinstance(cell, dict) else None
    if not isinstance(leaf, list) or len(leaf) != len(LEAF_ORDER):
        return None
    values = [_num(v) for v in leaf]
    if any(v is None or v < 0 or v > 1 for v in values) or abs(sum(values)-1) > 1e-6:
        return None
    return tuple(values)


class _Refuse(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class _Ledger:
    """Incremental, bounded index of ONE ledger file. Kept small on purpose: per pitch a few ints and short
    strings; the two heavy JSON columns of an attempt are parsed once, on first use, into a slim record held in an
    LRU (``max_cached``). Caller holds the owner's lock."""

    def __init__(self, path, owner):
        self.path, self.owner = str(path), owner
        self.db, self.identity = None, None
        self._reset()

    def _reset(self):
        self.marks = {'forecast': 0, 'later': 0, 'physical': 0, 'poll': 0, 'realized': 0}
        self.forecasts = {}     # (game, PA, pitch) -> [(attempt_key, decision, written_index, attempt_sha)]
        self.quarantined = set()  # (game, PA) with a matchup_changed_quarantine on any of its attempts
        self.late = set()       # attempt keys an order reconciliation labelled late
        self.physical = {}      # (game, PA, pitch) -> (label, attempt_key) of the latest reconciliation
        self.seen = {}          # (game, PA) -> {'pitcher': {ids}, 'batter': {ids}} every poll showed
        self.realized = {}      # (game, PA, pitch) -> [first observed index, latest pitch type]
        self.slim = OrderedDict()  # attempt_key -> slim attempt + forecast record
        self.rows, self.run = 0, None

    def close(self):
        if self.db is not None:
            try:
                self.db.close()
            except sqlite3.Error:
                pass
        self.db = None

    def _connect(self):
        """A read-only connection chosen so that no file is created next to the ledger (module docstring)."""
        try:
            stat = os.stat(self.path)
        except OSError:
            self.close()
            raise _Refuse('ledger_missing') from None
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity:  # another file under the same name: nothing held is about it
            self.close()
            self._reset()
            self.identity = identity
        if self.db is not None:
            return self.db, True
        try:
            with open(self.path, 'rb') as handle:
                header = handle.read(100)
        except OSError:
            raise _Refuse('ledger_unreadable:open') from None
        if len(header) < 100:
            raise _Refuse('ledger_not_ready')
        if header[:16] != b'SQLite format 3\x00':
            raise _Refuse('ledger_unreadable:not_sqlite')
        immutable = False
        if header[18] == 2 or header[19] == 2:  # WAL
            wal, shm = self.path+'-wal', self.path+'-shm'
            if not (os.path.exists(wal) and os.path.exists(shm)):
                try:
                    pending = os.path.getsize(wal) > 0
                except OSError:
                    pending = False
                if pending:
                    raise _Refuse('ledger_wal_unrecovered')
                immutable = True
        uri = 'file:'+quote(self.path)+'?mode=ro'+('&immutable=1' if immutable else '')
        db = sqlite3.connect(uri, uri=True, timeout=self.owner.busy_timeout_s, isolation_level=None,
                             check_same_thread=False)
        db.execute('PRAGMA query_only=ON')
        if not immutable:  # an immutable handle would never see a writer that attaches later: reopen every call
            self.db = db
        return db, not immutable

    def read(self, work):
        """Run ``work()`` inside one read transaction after taking in the rows appended since the last call."""
        db, keep = self._connect()
        try:
            db.execute('BEGIN')
            try:
                self._refresh(db)
                return work(db)
            finally:
                if db.in_transaction:
                    db.execute('ROLLBACK')
        except sqlite3.OperationalError as exc:
            text = str(exc).lower()
            self.close()
            if 'locked' in text or 'busy' in text:
                raise _Refuse('ledger_busy') from None
            if 'no such table' in text or 'no such column' in text:
                raise _Refuse('ledger_not_ready') from None
            raise _Refuse('ledger_unreadable:'+type(exc).__name__) from None
        except sqlite3.Error as exc:
            self.close()
            raise _Refuse('ledger_unreadable:'+type(exc).__name__) from None
        finally:
            if not keep:
                try:
                    db.close()
                except sqlite3.Error:
                    pass

    def _count(self, n=1):
        self.rows += n
        if self.rows > self.owner.max_index_rows:
            raise _Refuse('ledger_too_large')

    def _refresh(self, db):  # noqa: C901 - one append-only table per block
        cap, marks = self.owner.max_body_bytes, dict(self.marks)
        try:
            for rowid, key, game, ab, number, decision, index, sha in db.execute(
                    'SELECT rowid, attempt_key, game_pk, at_bat_number, candidate_pitch_number, decision, '
                    'written_at_replay_index, attempt_record_sha256 FROM shadow_forecasts WHERE rowid > ? ORDER BY rowid',
                    (marks['forecast'],)):
                self._count()
                self.forecasts.setdefault((game, ab, number), []).append((key, decision, index, sha))
                marks['forecast'] = rowid
            for row_id, key, kind, game, ab, body in db.execute(
                    "SELECT l.id, l.attempt_key, l.kind, p.game_pk, p.at_bat_number, CASE WHEN l.kind="
                    "'physical_order_reconciliation' AND length(l.body) <= ? THEN l.body END FROM later_observations l "
                    "JOIN pitch_attempts p ON p.attempt_key = l.attempt_key WHERE l.id > ? AND l.kind IN "
                    "('matchup_changed_quarantine', 'physical_order_reconciliation') ORDER BY l.id", (cap, marks['later'])):
                self._count()
                if kind == 'matchup_changed_quarantine':
                    self.quarantined.add((game, ab))
                else:
                    try:
                        label = json.loads(body).get('label') if body is not None else None
                    except (ValueError, AttributeError):
                        label = None
                    if label != 'provider_time_after_decision_unverified':  # late, unknown or unreadable
                        self.late.add(key)
                marks['later'] = row_id
            for row_id, game, ab, number, key, label in db.execute(
                    'SELECT id, game_pk, at_bat_number, pitch_number, attempt_key, label FROM physical_reconciliation '
                    'WHERE id > ? ORDER BY id', (marks['physical'],)):
                self._count()
                self.physical[(game, ab, number)] = (label, key)
                marks['physical'] = row_id
            for row_id, game, ab, body in db.execute(
                    'SELECT id, game_pk, at_bat_number, CASE WHEN length(body) <= ? THEN body END FROM poll_observations '
                    'WHERE id > ? ORDER BY id', (cap, marks['poll'])):
                try:
                    seen = json.loads(body) if body is not None else {}
                except ValueError:
                    seen = {}
                observed = seen.get('observed_matchup') if isinstance(seen, dict) else None
                shown = []
                if ab is not None:
                    shown += [(ab, seen.get('matchup')), (ab, observed)]
                elif isinstance(observed, dict) and _is_int(observed.get('at_bat_number')):
                    shown.append((observed['at_bat_number'], observed))
                for number, matchup in shown:
                    for role in ('pitcher', 'batter'):
                        pid = matchup.get(role) if isinstance(matchup, dict) else None
                        if _is_int(pid):
                            ids = self.seen.setdefault((game, number), {'pitcher': set(), 'batter': set()})[role]
                            if pid not in ids:
                                self._count()
                                ids.add(pid)
                marks['poll'] = row_id
            for row_id, game, ab, number, index, kind in db.execute(
                    'SELECT id, game_pk, at_bat_number, pitch_number, observed_at_replay_index, realized_pitch_type '
                    'FROM shadow_realized WHERE id > ? ORDER BY id', (marks['realized'],)):
                held = self.realized.get((game, ab, number))
                if held is None:
                    self._count()
                    self.realized[(game, ab, number)] = [index, kind]
                else:
                    held[0], held[1] = min(held[0], index) if _is_int(index) and _is_int(held[0]) else None, kind
                marks['realized'] = row_id
        except _Refuse:
            self.identity = None  # over the cap: drop everything, the next call starts over (and refuses again)
            raise
        self.marks = marks
        if self.run is None:
            row = db.execute('SELECT CASE WHEN length(body) <= ? THEN body END FROM shadow_run WHERE id = 1', (cap,)).fetchone()
            if row and row[0] is not None:
                try:
                    body = json.loads(row[0])
                    self.run = {k: body.get(k) for k in ('schema', 'mode', 'clock', 'game_pk')} if isinstance(body, dict) else {}
                except ValueError:
                    self.run = {}

    def attempt(self, db, key):
        """Slim immutable record of one attempt and its forecast row (parsed once, LRU)."""
        if key in self.slim:
            self.slim.move_to_end(key)
            return self.slim[key]
        cap = self.owner.max_body_bytes
        held = db.execute('SELECT game_pk, at_bat_number, last_pitch_number, outcome, decision_unix, sha, '
                          'CASE WHEN length(body) <= ? THEN body END FROM pitch_attempts WHERE attempt_key = ?', (cap, key)).fetchone()
        row = db.execute(
            'SELECT game_pk, at_bat_number, candidate_pitch_number, decision, outcome, written_at_replay_index, '
            'store_seq, feed_sha256, pitcher, batter, CASE WHEN length(action_probabilities) <= ? THEN action_probabilities END, '
            'model_kind, model_sha256, registry_sha256, encoder_sha256, profile_sha256, attempt_record_sha256, row_sha256, '
            'compute_ms FROM shadow_forecasts WHERE attempt_key = ?', (cap, key)).fetchone()
        if held is None or row is None:
            return None
        try:
            body, cells = json.loads(held[6]), json.loads(row[10])
            model_sha = json.loads(row[12]) if row[12] else {}
        except (TypeError, ValueError):
            body, cells, model_sha = None, None, {}
        ledger_cells = dict(db.execute('SELECT action, status FROM action_cells WHERE attempt_key = ?', (key,)))
        slim = {'valid': isinstance(body, dict) and isinstance(cells, list), 'attempt': held[:6], 'forecast': row[:10],
                'model_kind': row[11], 'compute_ms': _num(row[18]), 'ledger_cells': ledger_cells,
                'hashes': {'row_sha256': row[17], 'attempt_record_sha256': row[16], 'feed_sha256': row[7],
                           'profile_sha256': row[15], 'registry_sha256': row[13], 'encoder_sha256': row[14],
                           'model_sha256': model_sha if isinstance(model_sha, dict) else {}}}
        if slim['valid']:
            decision, snapshot, model = (body.get(k) if isinstance(body.get(k), dict) else {} for k in ('decision', 'snapshot', 'model'))
            own = {c.get('action'): c for c in body.get('cells') or [] if isinstance(c, dict)}
            slim |= {'outcome': body.get('outcome'), 'matchup': body.get('matchup'), 'state': body.get('pre_pitch_state'),
                     'leaf_order': model.get('leaf_order'), 'body_model_kind': model.get('kind'),
                     'decision_clock': decision.get('clock'), 'decision_unix': _num(decision.get('unix')),
                     'source': snapshot.get('source'), 'receipt_clock': snapshot.get('receipt_clock'),
                     # only scored cells are kept, and only where the forecast row and the attempt record agree
                     'cells': {c['action']: _seven(c) for c in cells
                               if isinstance(c, dict) and c.get('status') == 'scored' and isinstance(c.get('action'), str)
                               and (own.get(c['action']) or {}).get('status') == 'scored'
                               and (own.get(c['action']) or {}).get('seven_leaf') == c.get('seven_leaf')}}
        self.slim[key] = slim
        while len(self.slim) > self.owner.max_cached:
            self.slim.popitem(last=False)
        return slim


class ShadowExpectations:
    """Thread-safe, lazy reader of a Step 5 shadow ledger.

    ``ledger_path``: the ledger file, or a template with ``{game_pk}`` (one ledger per game, as the runner writes
    them). ``enabled`` is the display flag and is OFF by default. ``allow_replayed`` lets a post-game replay
    ledger answer (marked as such). ``min_lead_s``: how much earlier than the pitch's first receipt a live decision
    must be. ``caps``: stricter or other per-leaf upper bounds (``{'ground_ball_in_play': 0.2}``); a cap can only
    lower a bound of ``SHOWN_LEAVES``."""

    def __init__(self, ledger_path, *, enabled=False, allow_replayed=False, min_lead_s=1.0, caps=None,
                 max_cached=512, max_ledgers=4, max_index_rows=200_000, max_body_bytes=1 << 20, busy_timeout_s=0.05):
        self.ledger_path, self.enabled, self.allow_replayed = str(ledger_path), bool(enabled), bool(allow_replayed)
        self.min_lead_s, self.max_cached, self.max_ledgers = float(min_lead_s), int(max_cached), int(max_ledgers)
        self.max_index_rows, self.max_body_bytes, self.busy_timeout_s = int(max_index_rows), int(max_body_bytes), float(busy_timeout_s)
        self.below = {leaf['leaf']: min(leaf['below'], float((caps or {}).get(leaf['leaf'], leaf['below']))) for leaf in SHOWN_LEAVES}
        self._lock, self._ledgers = threading.RLock(), OrderedDict()
        self._counts, self._last = {}, None

    # -- public -------------------------------------------------------------------------------------------------
    def expectation(self, game_pk, at_bat_number, pitch_number, realized_pitch_type, *, pa_event, pitcher_id,
                    batter_id, situation_before, pitch_first_received=None):
        """The display dict for this pitch, or None. See ``explain`` for the arguments."""
        return self.explain(game_pk, at_bat_number, pitch_number, realized_pitch_type, pa_event=pa_event,
                            pitcher_id=pitcher_id, batter_id=batter_id, situation_before=situation_before,
                            pitch_first_received=pitch_first_received)[0]

    def for_pitch(self, pitch, *, pitch_first_received=None):
        """``expectation`` of a ``game_history`` pitch row. ``actual.event`` is set there only on the last pitch of a
        completed plate appearance that ended ON that pitch, which is exactly the pitch a key moment belongs to."""
        try:
            actual = pitch.get('actual') or {}
            game_pk = int(str(pitch['key']).split(':')[0])
            args = (game_pk, pitch['at_bat_number'], pitch['pitch_number'], actual.get('pitch_type'))
            names = {'pa_event': actual.get('event'), 'pitcher_id': (pitch.get('pitcher') or {}).get('id'),
                     'batter_id': (pitch.get('batter') or {}).get('id'), 'situation_before': pitch.get('situation_before')}
        except Exception:  # noqa: BLE001 - a row of another shape shows nothing
            self._note('bad_pitch_key')
            return None
        return self.expectation(*args, **names, pitch_first_received=pitch_first_received)

    def explain(self, game_pk, at_bat_number, pitch_number, realized_pitch_type, *, pa_event, pitcher_id, batter_id,
                situation_before, pitch_first_received=None):
        """``(dict | None, reason | None)``.

        ``game_pk:at_bat_number:pitch_number`` is the app's pitch key; ``realized_pitch_type`` the type code the
        app shows for it; ``pa_event`` the terminal ``eventType`` of the plate appearance this pitch ended (None
        for any other pitch); ``pitcher_id``/``batter_id``/``situation_before`` the pitch as thrown
        (``game_history`` row); ``pitch_first_received`` the wall time (unix s) the RC store first received a body
        holding this pitch (``pitch_events.first_received`` / ``display_state.pitch_receipts``), same host clock
        as the sidecar's."""
        try:
            with self._lock:
                try:
                    result = self._explain(game_pk, at_bat_number, pitch_number, realized_pitch_type, pa_event,
                                           pitcher_id, batter_id, situation_before, pitch_first_received)
                    self._note(None)
                    return result, None
                except _Refuse as refusal:
                    self._note(refusal.reason)
                    return None, refusal.reason
        except Exception as exc:  # noqa: BLE001 - never into a route
            reason = 'error:'+type(exc).__name__
            try:
                self._note(reason)
            except Exception:  # noqa: BLE001
                pass
            return None, reason

    def status(self):
        """Operator view: flags, refusal counts by code since start, the last refusal, what is held in memory."""
        with self._lock:
            return {'enabled': self.enabled, 'allow_replayed': self.allow_replayed, 'ledger_path': self.ledger_path,
                    'below': dict(self.below), 'min_lead_s': self.min_lead_s, 'last_reason': self._last,
                    'counts': dict(self._counts),
                    'ledgers': [{'path': held.path, 'indexed_rows': held.rows, 'cached_attempts': len(held.slim),
                                 'run': held.run, 'open': held.db is not None} for held in self._ledgers.values()]}

    def close(self):
        with self._lock:
            for held in self._ledgers.values():
                held.close()
            self._ledgers.clear()

    # -- internals ----------------------------------------------------------------------------------------------
    def _note(self, reason):
        code = 'shown' if reason is None else reason.split(':')[0]
        self._counts[code] = self._counts.get(code, 0)+1
        if reason is not None:
            self._last = reason

    def _ledger(self, game_pk):
        path = self.ledger_path.replace('{game_pk}', str(game_pk))
        held = self._ledgers.get(path)
        if held is None:
            held = self._ledgers[path] = _Ledger(path, self)
        self._ledgers.move_to_end(path)
        while len(self._ledgers) > self.max_ledgers:
            self._ledgers.popitem(last=False)[1].close()
        return held

    def _explain(self, game, ab, number, kind, pa_event, pitcher_id, batter_id, situation, first_received):  # noqa: C901
        if not self.enabled:
            raise _Refuse('disabled')
        if not (_is_int(game) and _is_int(ab) and _is_int(number) and game > 0 and ab >= 1 and 1 <= number <= 99):
            raise _Refuse('bad_pitch_key')
        if not is_key_moment(pa_event):
            raise _Refuse('not_key_moment')
        if kind not in ACTIONS:
            raise _Refuse('realized_type_unknown')
        if not (_is_int(pitcher_id) and _is_int(batter_id)):
            raise _Refuse('identity_unverified')
        thrown = _situation_thrown(situation)
        if thrown is None:
            raise _Refuse('situation_unverified')
        ledger, key = self._ledger(game), f'{game}:{ab}:after{number-1}'

        def work(db):
            rows = ledger.forecasts.get((game, ab, number), [])
            if not rows:
                raise _Refuse('no_forecast')
            if len(rows) != 1 or rows[0][0] != key:
                raise _Refuse('ambiguous_forecast')
            _, decision, written, attempt_sha = rows[0]
            if decision != 'scored':
                raise _Refuse('forecast_abstained')
            if (game, ab) in ledger.quarantined:
                raise _Refuse('matchup_changed:quarantine_in_plate_appearance')
            if key in ledger.late:
                raise _Refuse('recorded_late')
            label = ledger.physical.get((game, ab, number))
            if label is not None and label != ('had_attempt', key):
                raise _Refuse(f'unlinked:{label[0]}')
            slim = ledger.attempt(db, key)
            if slim is None:
                raise _Refuse('attempt_missing')
            if (not slim['valid'] or slim['attempt'][:3] != (game, ab, number-1) or slim['forecast'][:3] != (game, ab, number)
                    or slim['attempt'][5] != attempt_sha or slim['attempt'][3] != 'scored' or slim['outcome'] != 'scored'
                    or slim['forecast'][3:5] != ('scored', 'scored')):
                raise _Refuse('ledger_inconsistent')
            if tuple(slim['leaf_order'] or ()) != LEAF_ORDER:
                raise _Refuse('leaf_order_unknown')
            if slim['model_kind'] != MODEL_KIND or slim['body_model_kind'] != MODEL_KIND:
                raise _Refuse('model_unknown')
            matchup = slim['matchup'] if isinstance(slim['matchup'], dict) else {}
            if not (_is_int(matchup.get('pitcher')) and _is_int(matchup.get('batter'))) or \
                    slim['forecast'][8:10] != (matchup['pitcher'], matchup['batter']):
                raise _Refuse('forecast_inputs_unknown')
            if matchup['pitcher'] != pitcher_id:
                raise _Refuse('different_pitcher')
            if matchup['batter'] != batter_id:
                raise _Refuse('different_batter')
            seen = ledger.seen.get((game, ab), {})
            if any(ids - {matchup[role]} for role, ids in seen.items()):
                raise _Refuse('matchup_changed:another_player_observed_in_plate_appearance')
            inputs = _situation_forecast(slim['state'])
            if inputs is None:
                raise _Refuse('forecast_inputs_unknown')
            differing = [k for k in SITUATION if inputs[k] != thrown[k]]
            if differing:
                raise _Refuse('different_situation:'+','.join(differing))
            shown = ledger.realized.get((game, ab, number))
            if shown is not None and shown[1] is not None and shown[1] != kind:
                raise _Refuse('realized_type_disagrees')
            if shown is not None and not (_is_int(written) and _is_int(shown[0]) and written < shown[0]):
                raise _Refuse('written_after_pitch_shown')
            if slim['ledger_cells'].get(kind) != 'scored' or kind not in slim['cells']:
                raise _Refuse('cell_unsupported')
            seven = slim['cells'][kind]
            if seven is None:
                raise _Refuse('cell_invalid')
            recorded = self._timing(slim, shown, ledger.run, first_received)
            items, omitted = [], []
            for leaf in SHOWN_LEAVES:
                p = seven[LEAF_ORDER.index(leaf['leaf'])]
                if p >= self.below[leaf['leaf']]:
                    omitted.append(leaf['leaf'])
                    continue
                pct, text = percent(p)
                items.append({'leaf': leaf['leaf'], 'label': leaf['label'], 'meaning': leaf['meaning'], 'pct': pct,
                              'pct_text': text})
            if not items:
                raise _Refuse('above_supported_range')
            lead = LEAD_LIVE if recorded == RECORDED_LIVE else LEAD_REPLAYED
            hashes = slim['hashes']
            return {'kind': KIND, 'lead': lead, 'pitch_type': kind, 'pitch_label': PITCH_LABELS.get(kind, kind),
                    'items': items, 'text': lead+': '+' · '.join(f"{i['label']} {i['pct_text']}" for i in items),
                    'note': NOTE if recorded == RECORDED_LIVE else NOTE_REPLAYED, 'explain': EXPLAIN,
                    'basis': 'per_pitch', 'recorded': recorded, 'omitted': omitted,
                    'provenance': {'attempt_key': key, 'decision_unix': slim['decision_unix'],
                                   'decision_clock': slim['decision_clock'], 'source': slim['source'],
                                   'written_at_replay_index': written, 'run_mode': (ledger.run or {}).get('mode'),
                                   'model_kind': slim['model_kind'],
                                   'model': {k: str(v)[:12] for k, v in sorted(hashes['model_sha256'].items())},
                                   **{k: (v[:12] if isinstance(v, str) else None) for k, v in hashes.items()
                                      if k != 'model_sha256'}}}
        return ledger.read(work)

    def _timing(self, slim, shown, run, first_received):
        """``live_before_pitch`` or ``replayed_pre_pitch_inputs``; refuses otherwise.

        Live: the attempt says so positively (decision on the observer's own clock, from a live poll, in a run not
        declared a replay) and ``decision_unix + compute time + min_lead_s <= pitch_first_received``. The ledger
        has no separate commit time, so the decision instant plus the measured compute time stands for "written".
        Replay: the attempt carries the runner's post-hoc replay clock AND the ledger shows the forecast row was
        written from a body older than the first body that showed the pitch. That proves pre-pitch INPUTS, never a
        physical "before", so it is used only with ``allow_replayed`` and marked."""
        clock, source = slim['decision_clock'], slim['source']
        replay_run = isinstance(run, dict) and run.get('clock') == REPLAY_CLOCK
        if clock == LIVE_CLOCK and source == LIVE_SOURCE and slim['receipt_clock'] == LIVE_CLOCK and not replay_run:
            received, decided = _num(first_received), slim['decision_unix']
            if received is None or decided is None or slim['attempt'][4] is None or _num(slim['attempt'][4]) != decided:
                raise _Refuse('pitch_first_seen_unknown')
            if decided+max(slim['compute_ms'] or 0.0, 0.0)/1000+self.min_lead_s > received:
                raise _Refuse('recorded_after_pitch_appeared')
            return RECORDED_LIVE
        if clock == REPLAY_CLOCK and source == REPLAY_SOURCE:
            if not self.allow_replayed:
                raise _Refuse('replayed_not_allowed')
            if shown is None:  # the ledger never showed the pitch: "written before it was shown" is not proven
                raise _Refuse('written_after_pitch_shown:pitch_not_observed_in_ledger')
            return RECORDED_REPLAYED
        raise _Refuse('provenance_unknown')
