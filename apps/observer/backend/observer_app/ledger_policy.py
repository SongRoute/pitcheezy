"""A stand-in for the live ARM-B policy in an OFFLINE replay (``scripts/publisher_replay.py --policy ledger``):
every pending game state is answered with the recommendation the LIVE run's append-only ledger
(``served_recommendations``, ``live_store.LiveSnapshotStore``) recorded for that very state. Nothing is computed:
no model, no T7, no MPS. The replayed body therefore shows the recommendation that was on fans' screens during
the game, not a recomputation of it (DEMO-WS-MOMENTS-1029, Song's decision 2026-10-29 "장부 값으로 진행").

How a state is matched: the ledger keys a decision by ``state_sha256`` = sha256 of ``live_armb.state_key(feed)``,
every pre-pitch input the policy read. The replay hands this policy the same stored feed bodies the live run
polled, so the same state hashes to the same key; a body the live run never decided on has no row.

What this policy answers, in the shape ``demo_precompute.LivePolicy`` returns (``main.armb_view`` reads it):

* a row for the state -> ``pre`` = the row's ``displayed`` (status ``ready``, the stored candidates, no change),
  ``policy_identity`` = the row's (the live policy's own 64-hex identity: the candidates ARE its output);
  ``pitch.status`` is ``LEDGER_RECORDED``, a name of this module, never a ``policy_runtime`` status;
* no row, and the pending pitcher is in the registered support map -> the policy's refusal for that pitcher
  (``UNSUPPORTED_UNKNOWN_PITCHER`` ...), so the screen shows 추천 없음 with the same sentence as live and the A2
  reference card stays eligible. The map is an INPUT of the run (a JSON file registered by sha256, built from the
  10/17 coverage recount, ``armb_static``), not a rule re-implemented here;
* no row, any other pitcher -> 추천 없음 with ``NO_ROW_REASON`` (``pitch.status`` ``LEDGER_NO_ROW``): live had no
  recorded recommendation for this state (not computed in time, or the state was gated), and this replay never
  invents one. ``display_state`` turns it into ``unsupported / policy_refused`` with that sentence.

``pa_pitches`` is empty and ``previous``/``home_we_now`` are None: the ledger stores neither the at-bat's earlier
recomputations nor a win expectancy, so the body's ``win_expectancy`` slot reads ``unavailable`` and the screen
shows 미측정 rather than a number this replay did not have. Counters (``counters``) say how each call was answered.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

# ``live_armb.state_key`` and ``display_state.pending_key`` are imported when a state is answered, not here: the
# replay command validates the ledger inputs (``load_rows``, ``load_support``) BEFORE it pins the app's folders,
# and importing the app's modules before that would be refused (``publisher_replay.execute``).
SUPPORT_SCHEMA = 'pitcheezy-ledger-support-v1'
# the three pitcher-level refusals the A2 card is attached to (display_state.POLICY_REASONS); nothing else is accepted
UNSUPPORTED = ('UNSUPPORTED_UNKNOWN_PITCHER', 'UNSUPPORTED_PITCHER_HAND', 'UNSUPPORTED_EMPTY_SUPPORT')
READY_STATUS, NO_ROW_STATUS = 'LEDGER_RECORDED', 'LEDGER_NO_ROW'
NO_ROW_REASON = '경기 중 이 상황에 기록된 추천이 없어요.'  # <= 60 chars, Korean, no code: display_state shows it as is
DISPLAYED = ('status', 'reason', 'key', 'recommendation', 'policy_identity')  # live_ledger.DISPLAYED


class LedgerUnusable(ValueError):
    """The ledger copy or the support map cannot be the input of a replay."""


def load_rows(path, game_pk):
    """``{state_sha256: row}`` of the ``first`` (displayable) rows of ``game_pk`` in the live store copy at
    ``path``, opened read-only and immutable (no lock, no -wal/-shm beside it). Refuses an empty ledger, a row
    whose ``displayed`` is not a ready recommendation with candidates, two rows for one state, or more than one
    policy identity: this replay stands for exactly one live policy."""
    path = Path(path)
    if not path.is_file():
        raise LedgerUnusable('ledger copy missing')
    if any(Path(str(path)+suffix).exists() and Path(str(path)+suffix).stat().st_size for suffix in ('-wal',)):
        raise LedgerUnusable('the ledger copy has a non-empty WAL')
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro&immutable=1', uri=True, isolation_level=None)
    try:
        db.execute('PRAGMA query_only=1')
        try:
            found = db.execute("SELECT id, pitch_key, decided, body FROM served_recommendations WHERE game_pk=? AND outcome='first' ORDER BY id",
                               (int(game_pk),)).fetchall()
        except sqlite3.Error as error:
            raise LedgerUnusable(f'the ledger cannot be read: {type(error).__name__}') from None
    finally:
        db.close()
    rows, identities = {}, set()
    for row_id, pitch_key, decided, body in found:
        try:
            record = json.loads(body)
        except ValueError:
            raise LedgerUnusable(f'ledger row {row_id} is not JSON') from None
        displayed, state = record.get('displayed') or {}, record.get('state_sha256')
        if (record.get('schema') != 'served-recommendation-v1' or not isinstance(state, str) or len(state) != 64
                or set(displayed) != set(DISPLAYED) or displayed.get('status') != 'ready' or displayed.get('key') != pitch_key
                or not ((displayed.get('recommendation') or {}).get('candidates'))
                or not isinstance(record.get('policy_identity'), str) or len(record['policy_identity']) != 64
                or record['policy_identity'][:8] != displayed.get('policy_identity')):
            raise LedgerUnusable(f'ledger row {row_id} is not a displayable ready recommendation')
        if state in rows:
            raise LedgerUnusable(f'two first rows for one state ({state[:16]})')
        rows[state] = {'id': row_id, 'pitch_key': pitch_key, 'decided': decided, 'state_sha256': state,
                       'policy_identity': record['policy_identity'], 'displayed': displayed}
        identities.add(record['policy_identity'])
    if not rows:
        raise LedgerUnusable(f'no first row for game {game_pk}')
    if len(identities) != 1:
        raise LedgerUnusable(f'{len(identities)} policy identities in the ledger; one is expected')
    return rows


def load_support(path):
    """``{pitcher id: status}`` from a support map file (``SUPPORT_SCHEMA``): ``pitchers`` is ``{"<id>": {"status":
    one of UNSUPPORTED, ...}}``. Supported pitchers are simply absent. Anything else refuses."""
    try:
        body = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as error:
        raise LedgerUnusable(f'support map unreadable: {type(error).__name__}') from None
    if not isinstance(body, dict) or body.get('schema') != SUPPORT_SCHEMA or not isinstance(body.get('pitchers'), dict):
        raise LedgerUnusable('support map is not a pitcheezy-ledger-support-v1 file')
    support = {}
    for key, entry in body['pitchers'].items():
        status = entry.get('status') if isinstance(entry, dict) else None
        if not str(key).isdigit() or status not in UNSUPPORTED:
            raise LedgerUnusable(f'support map entry {key!r}: status must be one of {UNSUPPORTED}')
        support[int(key)] = status
    return support


def pending_pitcher(feed):
    """Id of the pitcher the pending pitch belongs to: the open at-bat's, else (between batters) the defense's."""
    live = feed.get('liveData') or {}
    plays = live.get('plays') or {}
    last = plays.get('currentPlay') or ((plays.get('allPlays') or []) or [{}])[-1]
    matchup = ((last.get('matchup') or {}).get('pitcher') or {}).get('id')
    defense = (((live.get('linescore') or {}).get('defense') or {}).get('pitcher') or {}).get('id')
    complete = (last.get('about') or {}).get('isComplete', True)
    chosen = defense if complete else matchup
    return chosen if chosen is not None else (matchup if complete else defense)


class LedgerPolicy:
    """Callable like ``demo_precompute.LivePolicy``: ``policy(feed) -> result``. ``rows`` from ``load_rows``,
    ``support`` from ``load_support``; ``game_pk`` is the one game this policy answers for (another game -> None,
    as the live policy returns None for a feed without rows)."""

    def __init__(self, rows, support, *, game_pk):
        self.rows, self.support, self.game_pk = dict(rows), dict(support or {}), int(game_pk)
        self.identity = next(iter(self.rows.values()))['policy_identity']
        self.counters = Counter()
        self.binds = 0  # read by the replay's summary: this policy binds nothing
        self.answered = []  # (state_sha256[:16], pitch_key, how) in call order, for the run's record

    def identity_short(self):
        return self.identity[:8]

    def __call__(self, feed):
        from . import display_state  # the pending pitch's name, computed the way the server names it
        from .live_armb import state_key
        if int(feed['gameData']['game']['pk']) != self.game_pk:
            self.counters['other_game'] += 1
            return None
        key = display_state.pending_key(feed)
        if key is None:
            self.counters['no_pending_pitch'] += 1
            return None
        state = hashlib.sha256(state_key(feed).encode()).hexdigest()
        row = self.rows.get(state)
        if row is not None:
            displayed = row['displayed']
            how = 'ledger' if displayed['key'] == key else 'ledger_key_differs'
            pitch = {'key': displayed['key'], 'pitch_number': _number(displayed['key']), 'status': READY_STATUS,
                     'pre': {'status': 'ready', 'reason': displayed['reason'], 'recommendation': copy.deepcopy(displayed['recommendation'])},
                     'actual': None}
            key = displayed['key']
        else:
            status = self.support.get(pending_pitcher(feed))
            how = 'unsupported' if status else 'no_row'
            pitch = {'key': key, 'pitch_number': _number(key), 'status': status or NO_ROW_STATUS,
                     'pre': {'status': 'unsupported', 'reason': None if status else NO_ROW_REASON, 'recommendation': None},
                     'actual': None}
        self.counters[how] += 1
        self.answered.append((state[:16], key, how))
        return {'key': key, 'pitch': pitch, 'pa_pitches': [], 'previous': None, 'home_we_now': None,
                'policy_identity': self.identity, 'seconds': {'bind': 0.0, 'total': 0.0}}

    def record(self):
        """What this policy did, for the run's summary (counts and names only, no candidates)."""
        return {'kind': 'ledger', 'live_policy_identity': self.identity, 'rows': len(self.rows),
                'unsupported_pitchers': {str(k): v for k, v in sorted(self.support.items())},
                'answers': dict(self.counters), 'rows_used': len({s for s, _, how in self.answered if how.startswith('ledger')}),
                'rows_never_asked': len(self.rows)-len({row['state_sha256'][:16] for row in self.rows.values()}
                                                       & {s for s, _, how in self.answered if how.startswith('ledger')})}


def _number(key):
    try:
        return int(str(key).rsplit(':', 1)[1])
    except (IndexError, ValueError):
        return None
