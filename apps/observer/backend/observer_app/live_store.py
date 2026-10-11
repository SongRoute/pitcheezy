"""Optional SQLite (WAL) cache of live GUMBO snapshots so a restart resumes every viewer's delay buffer.

Derived data only: the MLB feed and the recorder's ``{gamePk}_{stamp}.json.gz`` files stay authoritative,
and this database can be deleted and rebuilt from them (``ingest``). Off unless the service is started with
``PITCHEEZY_OBSERVER_LIVE_STORE=<path>``; without it ``LiveFeed`` keeps its in-memory buffer only.

A snapshot belongs to a *slot*: the wall time of the newest game event it contains (``event_wall``, the
source time, as ``LiveFeed`` computes it). Each slot keeps its first arrival (``first_fetched``) and an
append-only list of versions: ``version`` 1 is the first content seen, every content change for the newest
slot without a newer game event is a correction (``version`` + 1). ``seq`` orders commits per database and is
the replay cursor. Ingest outcomes for one game, under one ``BEGIN IMMEDIATE`` transaction:

* newer game event than the newest slot -> ``new`` slot
* same or older event, older or unorderable ``metaData.timeStamp`` than the newest slot's valid one -> ``stale``
  (out of order: the snapshot is not kept, its changed pitches still reach the pitch ledger; see ``_stale_stamp``)
* same content as the newest slot's current version -> ``duplicate`` (no write)
* otherwise -> ``correction`` of the newest slot (first-seen times kept, as the in-memory buffer does)

``served_recommendations`` is the append-only ledger of pre-pitch recommendations the serving path was about to
display (``append_recommendation``, called through ``live_ledger.admit`` before the response is built). It is
separate from the NextPitchContext tables below and never claims model eligibility or a verified next pitch.

``context_decisions`` / ``context_reconciliations`` hold append-only NextPitchContext decisions and their later,
separate unverified candidate observations (``append_decision`` / ``append_reconciliation``); no other reader uses them.

    python -m observer_app.live_store ingest --db D --dir apps/observer/live_snapshots [--game 849844] [--limit N]
    python -m observer_app.live_store status --db D
    python -m observer_app.live_store backup --db D --to D.bak
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from datetime import datetime
import gzip
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import zlib

from .live_feed import LiveFeed, event_wall

LOGGER = logging.getLogger('observer_app.live_store')
SCHEMA = '''
CREATE TABLE IF NOT EXISTS snapshots (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk INTEGER NOT NULL,
    event_wall REAL NOT NULL,
    version INTEGER NOT NULL,
    feed_ts TEXT,
    sha TEXT NOT NULL,
    first_fetched REAL NOT NULL,
    fetched REAL NOT NULL,
    source TEXT NOT NULL,
    body BLOB NOT NULL,
    requested REAL,
    UNIQUE(game_pk, event_wall, version));
CREATE INDEX IF NOT EXISTS snapshots_game_event ON snapshots(game_pk, event_wall, version);
CREATE TRIGGER IF NOT EXISTS snapshots_append_only
    BEFORE UPDATE ON snapshots BEGIN SELECT RAISE(ABORT, 'append-only snapshot'); END;
CREATE TABLE IF NOT EXISTS pitch_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk INTEGER NOT NULL,
    provider_event_id TEXT,
    at_bat_number INTEGER,
    pitch_number INTEGER,
    outcome TEXT NOT NULL,
    reason TEXT,
    revision INTEGER,
    provider_ts TEXT,
    provider_event_time TEXT,
    event_wall REAL NOT NULL,
    sha TEXT NOT NULL,
    first_received REAL,
    received REAL NOT NULL,
    receipt_clock TEXT NOT NULL,
    store_seq INTEGER NOT NULL,
    body BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS pitch_events_id ON pitch_events(game_pk, provider_event_id, id);
CREATE TRIGGER IF NOT EXISTS pitch_events_no_update
    BEFORE UPDATE ON pitch_events BEGIN SELECT RAISE(ABORT, 'append-only pitch event'); END;
CREATE TRIGGER IF NOT EXISTS pitch_events_no_delete
    BEFORE DELETE ON pitch_events BEGIN SELECT RAISE(ABORT, 'append-only pitch event'); END;
CREATE TABLE IF NOT EXISTS context_decisions (
    decision_id TEXT PRIMARY KEY,
    game_pk INTEGER NOT NULL,
    decision_at TEXT NOT NULL,
    category TEXT NOT NULL,
    store_seq INTEGER,
    sha TEXT NOT NULL,
    body BLOB NOT NULL);
CREATE TRIGGER IF NOT EXISTS context_decisions_no_update
    BEFORE UPDATE ON context_decisions BEGIN SELECT RAISE(ABORT, 'append-only decision'); END;
CREATE TRIGGER IF NOT EXISTS context_decisions_no_delete
    BEFORE DELETE ON context_decisions BEGIN SELECT RAISE(ABORT, 'append-only decision'); END;
CREATE TABLE IF NOT EXISTS context_reconciliations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL REFERENCES context_decisions(decision_id),
    revision INTEGER NOT NULL,
    reconciled_at TEXT NOT NULL,
    sha TEXT NOT NULL,
    body BLOB NOT NULL,
    UNIQUE(decision_id, revision));
CREATE TRIGGER IF NOT EXISTS context_reconciliations_no_update
    BEFORE UPDATE ON context_reconciliations BEGIN SELECT RAISE(ABORT, 'append-only reconciliation'); END;
CREATE TRIGGER IF NOT EXISTS context_reconciliations_no_delete
    BEFORE DELETE ON context_reconciliations BEGIN SELECT RAISE(ABORT, 'append-only reconciliation'); END;
CREATE TABLE IF NOT EXISTS served_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    game_pk INTEGER NOT NULL,
    pitch_key TEXT,
    outcome TEXT NOT NULL,
    decided REAL NOT NULL,
    decision_clock TEXT NOT NULL,
    store_seq INTEGER NOT NULL,
    sha TEXT NOT NULL,
    body BLOB NOT NULL,
    UNIQUE(decision_id, sha));
CREATE INDEX IF NOT EXISTS served_recommendations_game ON served_recommendations(game_pk, id);
CREATE TRIGGER IF NOT EXISTS served_recommendations_no_update
    BEFORE UPDATE ON served_recommendations BEGIN SELECT RAISE(ABORT, 'append-only served recommendation'); END;
CREATE TRIGGER IF NOT EXISTS served_recommendations_no_delete
    BEFORE DELETE ON served_recommendations BEGIN SELECT RAISE(ABORT, 'append-only served recommendation'); END;
'''
ACCEPTED = "outcome IN ('new', 'correction')"
# Test-only snapshot source. No production writer emits it (``PersistentLiveFeed``: 'live_poll'/'memory_recovery',
# ``ingest_dir``: 'recorded:*'), so a live, recorded or Final provider body can never carry KNOWN pre-pitch state
# until a separately reviewed real provider contract exists. Bound to the committed row, not to a caller flag.
SYNTHETIC_SOURCE = 'synthetic_test_pre_pitch'
SYNTHETIC_SCHEMA = 'synthetic-pre-pitch-v1'
SYNTHETIC_EVIDENCE = 'synthetic_test_state_not_provider'
# The only receipt clock a stored snapshot source can truthfully claim; 'UNKNOWN' (abstention) is always truthful.
RECEIPT_CLOCK = {'live_poll': 'observer_local', SYNTHETIC_SOURCE: 'synthetic_test_clock'}
# 'eligible' is never persisted: two required fields are deliberately UNSOURCED and no model mapping exists.
DECISION_CATEGORIES = ('feature_unavailable', 'ambiguous', 'unsupported')


def _int_in(low, high):
    return lambda v: type(v) is int and low <= v <= high


# The only field -> path mapping that can be KNOWN, each with its validator (no bool-as-int).
SYNTHETIC_PRE_PITCH = {
    'balls': ('balls', _int_in(0, 3)), 'strikes': ('strikes', _int_in(0, 2)), 'outs_when_up': ('outs', _int_in(0, 2)),
    'inning': ('inning', _int_in(1, 30)), 'inning_topbot': ('inning_topbot', lambda v: v in ('Top', 'Bottom')),
    'home_score': ('home_score', _int_in(0, 99)), 'away_score': ('away_score', _int_in(0, 99)),
    'bases': ('bases', lambda v: type(v) is list and all(type(b) is int for b in v) and v == sorted(set(v))
              and set(v) <= {1, 2, 3}),
    'stand': ('stand', lambda v: v in ('L', 'R')), 'p_throws': ('p_throws', lambda v: v in ('L', 'R')),
    'pitcher': ('pitcher', _int_in(1, 10**7)), 'batter': ('batter', _int_in(1, 10**7))}


def synthetic_pre_pitch(feed, source, field):
    """(value, None) for a valid synthetic pre-pitch ``field`` of a committed ``SYNTHETIC_SOURCE`` body, else
    (None, reason). Never reads plays, results, physics or provider linescore."""
    if source != SYNTHETIC_SOURCE:
        return None, 'not_synthetic_test_source'
    state = feed.get('syntheticPrePitch') if isinstance(feed, dict) else None
    if not isinstance(state, dict) or state.get('schema') != SYNTHETIC_SCHEMA:
        return None, 'synthetic_envelope_absent'
    key, valid = SYNTHETIC_PRE_PITCH[field]
    if key not in state or state[key] is None:
        return None, 'synthetic_field_absent'
    return (state[key], None) if valid(state[key]) else (None, 'invalid_synthetic_value')


def _provider_ts(value):
    """GUMBO ``metaData.timeStamp`` (``%Y%m%d_%H%M%S`` UTC) as a datetime, or None if absent or malformed."""
    if not isinstance(value, str) or not re.fullmatch(r'\d{8}_\d{6}', value):
        return None
    try:
        return datetime.strptime(value, '%Y%m%d_%H%M%S')
    except ValueError:
        return None


def _feed_ts(feed):
    meta = feed.get('metaData') if isinstance(feed, dict) else None
    return meta.get('timeStamp') if isinstance(meta, dict) else None


def _stale_stamp(feed_ts, newest_ts, newer_event):
    """True if a snapshot stamped ``feed_ts`` whose event is not newer must not replace the newest one stamped
    ``newest_ts``. Two valid provider stamps compare as times. A missing or malformed stamp never replaces a valid
    one, and a valid stamp always replaces an unorderable one. If neither parses, ordering is unknown: the legacy
    raw string compare is kept (only refuses when both are non-empty strings and ``feed_ts`` sorts first), so
    otherwise the later arrival is selected. A newer event is always kept as its own slot whatever its stamp, so
    a newer slot's selected snapshot may carry an older or unorderable provider revision: the pitch ledger then
    refuses any changed pitch under it (``stale_revision`` / ``unordered_revision``)."""
    if newer_event:
        return False
    new, old = _provider_ts(feed_ts), _provider_ts(newest_ts)
    if new and old:
        return new < old
    if new or old:
        return not new
    return bool(isinstance(feed_ts, str) and isinstance(newest_ts, str) and feed_ts and newest_ts and feed_ts < newest_ts)


def _provider_order(db, game_pk, feed_ts, seq, sha):
    """Provider-revision order of the committed version ``seq`` (content ``sha``), which local receipt order does
    not imply: 'unorderable' (missing/malformed stamp), 'stale' (an earlier committed version of the game carries a
    later valid stamp, e.g. a newer-event slot selected under an older revision), 'same_stamp_conflict' (an earlier
    committed version carries the same valid stamp with different content, so only receipt order separates them)
    or 'ordered'. Only versions committed before ``seq`` are consulted, so restore agrees with ingest."""
    if not _provider_ts(feed_ts):
        return 'unorderable'
    # ponytail: O(prior versions) per call, O(n^2) per game (~0.6 ms/ingest at 3000 versions); if that grows,
    # keep a per-game running MAX(valid feed_ts) instead of rescanning.
    later = db.execute('SELECT DISTINCT feed_ts FROM snapshots WHERE game_pk=? AND seq<? AND feed_ts>?',
                       (game_pk, seq, feed_ts)).fetchall()  # valid stamps sort as text; parse filters the rest
    if any(_provider_ts(ts) for ts, in later):
        return 'stale'
    if db.execute('SELECT 1 FROM snapshots WHERE game_pk=? AND seq<? AND feed_ts=? AND sha<>? LIMIT 1',
                  (game_pk, seq, feed_ts, sha)).fetchone():
        return 'same_stamp_conflict'
    return 'ordered'


def _pitches(feed):
    """(provider_event_id, at_bat_number, pitch_number, playEvent, reason) for every pitch in the feed. The
    official key is GUMBO ``atBatIndex``+1 / ``pitchNumber``; ``playId`` is the provider's id, not a promised
    stable one. A malformed item gets ``reason`` and no key, so it is quarantined alone."""
    def number(value, low):
        return value if type(value) is int and value >= low else None
    live = feed.get('liveData') if isinstance(feed, dict) else None
    plays = live.get('plays') if isinstance(live, dict) else None
    plays = plays.get('allPlays') if isinstance(plays, dict) else None
    for play in plays if isinstance(plays, list) else []:
        about = play.get('about') if isinstance(play, dict) else None
        index = number(about.get('atBatIndex') if isinstance(about, dict) else None, 0)
        events = play.get('playEvents') if isinstance(play, dict) else None
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict):
                yield None, None, None, event, 'malformed_event'
                continue
            if not event.get('isPitch'):
                continue
            pid, pitch = event.get('playId'), number(event.get('pitchNumber'), 1)
            pid = pid.strip() if isinstance(pid, str) else str(pid) if type(pid) is int else None
            if not pid or index is None or pitch is None:
                yield pid or None, None, None, event, 'incomplete_identity'
            else:
                yield pid, index+1, pitch, event, None


def _canonical(feed):
    return json.dumps(feed, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


class LiveSnapshotStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript(SCHEMA)
            if 'requested' not in [row[1] for row in db.execute('PRAGMA table_info(snapshots)')]:
                try:  # stores made before the receipt split
                    db.execute('ALTER TABLE snapshots ADD COLUMN requested REAL')
                except sqlite3.OperationalError as exc:  # another process migrated it between the two statements
                    if 'duplicate column' not in str(exc):
                        raise

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA busy_timeout=5000')
        return db

    @contextmanager
    def _write(self):
        """One serialized write transaction: committed whole or not at all."""
        with closing(self.connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
            except BaseException:
                db.execute('ROLLBACK')
                raise
            db.execute('COMMIT')

    def ingest(self, game_pk, feed, *, fetched, source, slot_event=None, first_fetched=None, requested=None):
        """Returns (outcome, entry); ``entry`` is the ``LiveFeed`` buffer entry for the slot (None if stale).

        ``fetched`` is the receipt: the wall time the response was completely received. ``requested`` (optional)
        is when that request was sent; it is kept per version (``version_requested``) and never used as a receipt."""
        game_pk, body = int(game_pk), _canonical(feed)
        event, sha = event_wall(feed) if slot_event is None else slot_event, hashlib.sha256(body).hexdigest()
        event = fetched if event is None else min(event, fetched)
        feed_ts = _feed_ts(feed)
        with self._write() as db:
            newest = db.execute('SELECT event_wall, version, feed_ts, sha, first_fetched, seq, fetched, source, requested FROM snapshots WHERE game_pk=? '
                                'ORDER BY event_wall DESC, version DESC LIMIT 1', (game_pk,)).fetchone()
            if newest is None or event > newest[0]:
                outcome, slot, version, first = 'new', event, 1, fetched if first_fetched is None else first_fetched
            elif _stale_stamp(feed_ts, newest[2], False):
                # Selection is unchanged and the stale snapshot body is not kept, but its changed pitches still get
                # a receipt (quarantined unless new). ``store_seq`` then names the selected version, not this arrival.
                self._record_pitches(db, game_pk, feed, feed_ts=feed_ts, event=newest[0], received=fetched,
                                     source=source, seq=newest[5])
                return 'stale', None
            elif sha == newest[3]:
                return 'duplicate', {'game_pk': game_pk, 'feed': feed, 'event': newest[0], 'fetched': newest[4],
                                     'seq': newest[5], 'version': newest[1], 'sha': newest[3],
                                     'version_fetched': newest[6], 'source': newest[7],
                                     'requested': newest[8] if newest[1] == 1 else None, 'version_requested': newest[8],
                                     'provider_order': _provider_order(db, game_pk, newest[2], newest[5], newest[3])}
            else:
                outcome, slot, version, first = 'correction', newest[0], newest[1]+1, newest[4]
            db.execute('INSERT INTO snapshots(game_pk, event_wall, version, feed_ts, sha, first_fetched, fetched, source, body, '
                       'requested) VALUES (?,?,?,?,?,?,?,?,?,?)', (game_pk, slot, version, feed_ts, sha, first, fetched, source,
                                                                   gzip.compress(body, compresslevel=5), requested))
            seq = db.execute('SELECT last_insert_rowid()').fetchone()[0]
            self._record_pitches(db, game_pk, feed, feed_ts=feed_ts, event=slot, received=fetched,
                                 source=source, seq=seq)
            order = _provider_order(db, game_pk, feed_ts, seq, sha)
        return outcome, {'game_pk': game_pk, 'feed': feed, 'event': slot, 'fetched': first, 'seq': seq,
                         'version': version, 'sha': sha, 'version_fetched': fetched, 'source': source,
                         'requested': requested if outcome == 'new' and first == fetched else None,
                         'version_requested': requested, 'provider_order': order}

    def _record_pitches(self, db, game_pk, feed, *, feed_ts, event, received, source, seq):
        """Append each pitch whose provider id is new or whose payload changed under a strictly later, valid
        provider ``timeStamp``. A malformed item, same id with a different official key, a key already held by
        another id, or a changed payload under an equal, older or unparseable ``timeStamp`` is quarantined
        (kept, never accepted). Every snapshot that reaches this store is seen, including a ``stale`` one that
        was not selected; for that one ``event_wall``/``store_seq`` reference the selected version it lost to, not
        a stored copy of the arrival. Corrections superseded while the store was unreachable are not recoverable,
        so this is a pitch receipt ledger, not a complete log of raw snapshots. ``first_received`` is the first
        *recorded* receipt, an upper bound on the local first receipt: after a store outage it is the last in-memory
        receipt (``memory_recovery``), after a crash the restart poll time."""
        clock = 'observer_local' if source == 'live_poll' else 'unverified'
        for pid, at_bat, number, payload, reason in _pitches(feed):
            body = _canonical(payload)
            sha = hashlib.sha256(body).hexdigest()
            current = holder = None
            if not reason:
                current = db.execute(f'SELECT at_bat_number, pitch_number, sha, provider_ts, revision, first_received '
                                     f'FROM pitch_events WHERE game_pk=? AND provider_event_id=? AND {ACCEPTED} '
                                     'ORDER BY id DESC LIMIT 1', (game_pk, pid)).fetchone()
                holder = db.execute(f'SELECT provider_event_id FROM pitch_events WHERE game_pk=? AND at_bat_number=? '
                                    f'AND pitch_number=? AND {ACCEPTED} AND provider_event_id IS NOT ?',
                                    (game_pk, at_bat, number, pid)).fetchone()
            revision, first = None, None
            if reason:
                pass
            elif holder:
                reason = 'ambiguous_remap'
            elif current is None:
                revision, first = 1, received
            elif (current[0], current[1]) != (at_bat, number):
                reason = 'conflicting_key'
            elif current[2] == sha:
                continue
            elif not _provider_ts(feed_ts) or not _provider_ts(current[3]):
                reason = 'unordered_revision'
            elif _provider_ts(feed_ts) == _provider_ts(current[3]):
                reason = 'same_revision_hash_mismatch'
            elif _provider_ts(feed_ts) < _provider_ts(current[3]):
                reason = 'stale_revision'
            else:
                revision, first = current[4]+1, current[5]
            if reason and db.execute('SELECT 1 FROM pitch_events WHERE game_pk=? AND provider_event_id IS ? AND sha=? '
                                     'AND reason=?', (game_pk, pid, sha, reason)).fetchone():
                continue
            db.execute('INSERT INTO pitch_events(game_pk, provider_event_id, at_bat_number, pitch_number, outcome, reason, '
                       'revision, provider_ts, provider_event_time, event_wall, sha, first_received, received, receipt_clock, '
                       'store_seq, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (game_pk, pid, at_bat, number, 'quarantine' if reason else 'new' if revision == 1 else 'correction',
                        reason, revision, feed_ts, payload.get('startTime') if isinstance(payload, dict) else None, event, sha, first, received, clock, seq,
                        gzip.compress(body, compresslevel=5)))

    def pitch_events(self, game_pk):
        """Every pitch row in append order plus the accepted key coverage per at-bat (synthetic reconciliation
        only: gaps say which pitch numbers this store never saw, not that the game is complete).

        ``history`` is always UNKNOWN: revisions never fetched (network gap) or fetched but superseded/lost while
        the store was down or the process crashed leave no trace, and no raw arrival log exists, so this ledger
        cannot prove every intermediate revision. It is derived from committed rows, so a restart reports the
        same; ``as_of`` is the reconciliation boundary."""
        keys = ('id', 'provider_event_id', 'at_bat_number', 'pitch_number', 'outcome', 'reason', 'revision',
                'provider_ts', 'provider_event_time', 'event_wall', 'sha', 'first_received', 'received',
                'receipt_clock', 'store_seq')
        with closing(self.connect()) as db:
            rows = [dict(zip(keys, row)) for row in db.execute(
                f'SELECT {", ".join(keys)} FROM pitch_events WHERE game_pk=? ORDER BY id', (int(game_pk),))]
            max_seq, recovered = db.execute("SELECT MAX(seq), SUM(source='memory_recovery') FROM snapshots "
                                            'WHERE game_pk=?', (int(game_pk),)).fetchone()
        history = {'status': 'UNKNOWN',
                   'reasons': ['intermediate_revisions_unlogged'] + (['store_outage_recovered'] if recovered else []),
                   'as_of': {'store_seq': max_seq, 'pitch_event_id': rows[-1]['id'] if rows else None}}
        coverage = {}
        for row in rows:
            if row['outcome'] == 'new':
                coverage.setdefault(row['at_bat_number'], []).append(row['pitch_number'])
        gaps = {ab: sorted(set(range(1, max(n)+1))-set(n)) for ab, n in coverage.items()}
        return {'events': rows, 'coverage': {ab: sorted(n) for ab, n in coverage.items()},
                'gaps': {ab: g for ab, g in gaps.items() if g},
                'missing_at_bats': sorted(set(range(1, max(coverage, default=0)+1))-set(coverage)),
                'history': history}

    def append_decision(self, record):
        """Persist one ``next-pitch-context-v1`` decision record (``service_research.next_pitch_context``) exactly
        once, as its canonical bytes. Its snapshot must be a version committed here with the same slot version and
        sha, received at or before ``decision_at``, its ``source`` must be the stored row's and its receipt clock that
        source's clock (``RECEIPT_CLOCK``) or 'UNKNOWN', and every KNOWN field must cite that version on that clock
        with ``decision_clock`` equal to it. An abstaining record ('ambiguous'/'unsupported') may truthfully carry an
        UNKNOWN or mismatched clock and is appended and counted; 'eligible' is refused. Returns 'appended', or
        'duplicate' for the same bytes; a refused or different body under a stored ``decision_id`` raises and the
        stored one is kept. Committed with ``synchronous=FULL`` like the snapshots: a crash before this call loses
        the decision, so this is a record of persisted decisions, not proof of every decision made."""
        body, snap = _canonical(record), record['snapshot']
        decided = datetime.fromisoformat(record['decision_at']).timestamp()
        category = record['opportunity']['category']
        if category not in DECISION_CATEGORIES:
            raise ValueError(f'Decision category {category!r} is not persisted by this provisional contract')
        abstains = category != 'feature_unavailable'
        with self._write() as db:
            row = clock = None
            if snap is not None:
                row = db.execute('SELECT version, sha, fetched, source, body FROM snapshots WHERE seq=? AND game_pk=?',
                                 (snap['store_seq'], record['game_pk'])).fetchone()
                if row is None or (row[0], row[1]) != (snap['slot_version'], snap['feed_sha256']) or row[2] > decided:
                    raise ValueError('Decision snapshot is not a committed version received by decision_at')
                clock = snap.get('receipt_clock')
                if snap.get('source') != row[3] or clock not in ('UNKNOWN', RECEIPT_CLOCK.get(row[3])):
                    raise ValueError('Decision snapshot source/receipt clock is not the stored row\'s')
            if not abstains and (clock in (None, 'UNKNOWN') or record['decision_clock'] != clock):
                raise ValueError('Only an abstaining decision may carry an unverified or mismatched clock')
            stored = None
            for name, field in record['fields'].items():
                if field['status'] != 'KNOWN':
                    continue
                prov = field['provenance'] or {}
                if snap is None or (prov.get('store_seq'), prov.get('feed_sha256')) != (snap['store_seq'], snap['feed_sha256']):
                    raise ValueError(f'KNOWN {name} does not cite the decision snapshot')
                if abstains or prov.get('source') != 'selected_snapshot' or prov.get('receipt_clock') != clock:
                    raise ValueError(f'KNOWN {name} source/receipt clock is not the decision snapshot\'s')
                # Re-derive from the stored body: only the fixed synthetic path, same value, same validator.
                if name not in SYNTHETIC_PRE_PITCH or prov.get('evidence') != SYNTHETIC_EVIDENCE or \
                        prov.get('path') != ['syntheticPrePitch', SYNTHETIC_PRE_PITCH[name][0]]:
                    raise ValueError(f'KNOWN {name} is not the fixed synthetic pre-pitch path')
                stored = stored or json.loads(gzip.decompress(row[4]))
                value, why = synthetic_pre_pitch(stored, row[3], name)
                if why or type(value) is not type(field['value']) or value != field['value']:
                    raise ValueError(f'KNOWN {name} is not the stored synthetic value ({why or "value_mismatch"})')
            held = db.execute('SELECT body FROM context_decisions WHERE decision_id=?', (record['decision_id'],)).fetchone()
            if held:
                if bytes(held[0]) == body:
                    return 'duplicate'
                raise ValueError('A conflicting decision exists for this decision_id; the stored record is kept')
            db.execute('INSERT INTO context_decisions(decision_id, game_pk, decision_at, category, store_seq, sha, body) '
                       'VALUES (?,?,?,?,?,?,?)', (record['decision_id'], record['game_pk'], record['decision_at'],
                                                  record['opportunity']['category'], snap and snap['store_seq'],
                                                  hashlib.sha256(body).hexdigest(), body))
        return 'appended'

    def append_reconciliation(self, decision_id, *, reconciled_at, evidence):
        """Append one later candidate observation for a stored decision as a separate revision; the decision (and its
        UNKNOWN target) is never rewritten. ``evidence`` are accepted ``pitch_events`` ids of one key, each recorded
        on the observer clock, first received after ``decision_at`` and by ``reconciled_at`` (> ``decision_at``); the
        decision itself must be on the observer clock too (its ``decision_clock`` and snapshot receipt clock), else
        the two times are not comparable and this refuses. A ledger row received later proves only a later
        observation, never the immediately next physical pitch: ``candidate_target`` is always ``UNVERIFIED_NEXT``.
        Accepted ``new`` rows of other keys first received after the decision and no later than the cited one are
        listed as ``supplementary_observations`` only; receipt order is not physical order, so they never change the
        status. Each revision links the previous one (``previous``), saying whether it supersedes a different key.
        Returns the stored body."""
        at = datetime.fromisoformat(reconciled_at)
        if at.tzinfo is None:
            raise ValueError('A timezone-aware reconciled_at is required')
        with self._write() as db:
            held = db.execute('SELECT game_pk, decision_at, body FROM context_decisions WHERE decision_id=?',
                              (decision_id,)).fetchone()
            if held is None:
                raise ValueError('Unknown decision_id')
            record = json.loads(held[2])
            if record['decision_clock'] != 'observer_local' or \
                    (record['snapshot'] or {}).get('receipt_clock') != 'observer_local':
                raise ValueError('Decision clock is not observer_local; it is not comparable with observer receipts')
            decided = datetime.fromisoformat(held[1]).timestamp()
            if at.timestamp() <= decided:
                raise ValueError('reconciled_at must be after decision_at')
            keys = ('pitch_event_id', 'at_bat_number', 'pitch_number', 'outcome', 'revision', 'sha', 'first_received',
                    'received', 'receipt_clock', 'store_seq')
            rows = [db.execute(f'SELECT id, at_bat_number, pitch_number, outcome, revision, sha, first_received, received, '
                               f'receipt_clock, store_seq FROM pitch_events WHERE id=? AND game_pk=? AND {ACCEPTED}',
                               (int(i), held[0])).fetchone() for i in evidence]
            if not rows or None in rows:
                raise ValueError('Evidence must be accepted pitch ledger rows of this game')
            rows = [dict(zip(keys, row)) for row in rows]
            if any(r['receipt_clock'] != 'observer_local' for r in rows):
                raise ValueError('Evidence receipt clock is unverified')
            if any(not decided < r['first_received'] <= r['received'] <= at.timestamp() for r in rows):
                raise ValueError('Evidence must be first received after the decision and by reconciled_at')
            if len({(r['at_bat_number'], r['pitch_number']) for r in rows}) != 1:
                raise ValueError('Evidence rows name conflicting keys')
            key = (rows[0]['at_bat_number'], rows[0]['pitch_number'])
            cited_first = min(r['first_received'] for r in rows)
            between = db.execute("SELECT id, at_bat_number, pitch_number, first_received, receipt_clock FROM pitch_events WHERE game_pk=? "
                                 "AND outcome='new' AND first_received>? AND first_received<=? "
                                 'AND NOT (at_bat_number=? AND pitch_number=?) ORDER BY id',
                                 (held[0], decided, cited_first, *key)).fetchall()
            prior = db.execute('SELECT revision, body FROM context_reconciliations WHERE decision_id=? '
                               'ORDER BY revision DESC LIMIT 1', (decision_id,)).fetchone()
            previous = None
            if prior:
                last = json.loads(prior[1])['candidate_target']
                same = (last['at_bat_number'], last['pitch_number']) == key
                previous = {'revision': prior[0], 'at_bat_number': last['at_bat_number'],
                            'pitch_number': last['pitch_number'],
                            'relation': 'same_key_later_observation' if same else 'supersedes_different_key'}
            body = _canonical({'schema': 'next-pitch-reconciliation-v3', 'decision_id': decision_id,
                               'revision': (prior[0] if prior else 0)+1, 'reconciled_at': at.isoformat(),
                               'decision_clock': 'observer_local', 'evidence': rows, 'previous': previous,
                               'candidate_observation': {'pitch_event_ids': [r['pitch_event_id'] for r in rows],
                                                         'first_received': cited_first, 'receipt_clock': 'observer_local'},
                               'supplementary_observations': {
                                   'relation': 'receipt_order_not_physical_order_proof',
                                   'rows': [dict(zip(('pitch_event_id', 'at_bat_number', 'pitch_number',
                                                      'first_received', 'receipt_clock'), r)) for r in between]},
                               'candidate_target': {
                                   'game_pk': held[0], 'at_bat_number': key[0], 'pitch_number': key[1],
                                   'status': 'UNVERIFIED_NEXT',
                                   'basis': 'accepted_row_first_received_after_decision',
                                   'reason': 'later_observation_not_proof_of_immediate_next_pitch'}})
            revision = json.loads(body)['revision']
            db.execute('INSERT INTO context_reconciliations(decision_id, revision, reconciled_at, sha, body) VALUES (?,?,?,?,?)',
                       (decision_id, revision, at.isoformat(), hashlib.sha256(body).hexdigest(), body))
        return json.loads(body)

    def decision_history(self, decision_id):
        """The stored decision bytes (and parsed record) plus every reconciliation revision in append order."""
        with closing(self.connect()) as db:
            held = db.execute('SELECT body FROM context_decisions WHERE decision_id=?', (decision_id,)).fetchone()
            later = db.execute('SELECT body FROM context_reconciliations WHERE decision_id=? ORDER BY revision',
                               (decision_id,)).fetchall()
        if held is None:
            return None
        return {'decision': bytes(held[0]), 'record': json.loads(held[0]),
                'reconciliations': [json.loads(body) for body, in later]}

    def decision_counts(self, game_pk):
        """Local decision opportunities by category, one per stored ``decision_id``; not a physical-pitch count."""
        with closing(self.connect()) as db:
            rows = db.execute('SELECT category, COUNT(*) FROM context_decisions WHERE game_pk=? GROUP BY category',
                              (int(game_pk),)).fetchall()
        return {'unit': 'local_decision_opportunity', 'decisions': sum(n for _, n in rows), 'categories': dict(rows)}

    def append_recommendation(self, *, game_pk, state_sha, policy_identity, pitch_key, snapshot, displayed, decided,
                              feed_ok):
        """Persist the pre-pitch recommendation about to be displayed, before it is displayed. Returns
        (outcome, record) where ``record['displayed']`` is what may be shown:

        * 'first' — nothing was stored for this decision; ``displayed`` is now the stored one.
        * 'duplicate' — the same content is already stored (another viewer or poll); nothing is written.
        * 'conflict' — a different content was stored first for the same decision. The new content is appended
          as a ``conflict`` row (kept, never displayed) and the *first* stored record is returned, so a viewer
          never sees the recommendation for one decision silently change.

        A decision is ``(game_pk, state_sha, policy_identity)``: ``state_sha`` hashes every pre-pitch input the
        policy reads (``live_armb.state_key``), so a provider correction of an earlier pitch is a new decision and
        a new row for the same ``pitch_key``, never a rewrite. ``pitch_key`` is the policy's name for the pending
        pitch, not a provider id, and the ledger never says which physical pitch followed (``target`` stays
        UNKNOWN). ``snapshot`` must be a version committed here (same ``seq``/``version``/``sha``) and received at
        or before ``decided``; anything else raises ``ValueError`` and nothing is written. ``decision_clock`` is
        'observer_local' only for a real upstream poll (``RECEIPT_CLOCK``), else 'unverified'. Committed with
        ``synchronous=FULL``: a crash before this returns means the recommendation was not displayed either.
        Both times are the observer's wall clock: if it steps back between the receipt and the decision this
        refuses (fail-closed) until it has passed the receipt again. The usual repeat (another viewer or poll of
        the same decision) is answered from a plain read, without taking the write lock."""
        game_pk = int(game_pk)
        shown = _canonical(displayed)
        decision_id = hashlib.sha256(_canonical([game_pk, state_sha, policy_identity])).hexdigest()

        def look(db):
            row = db.execute('SELECT version, sha, fetched, source FROM snapshots WHERE seq=? AND game_pk=?',
                             (snapshot.get('seq'), game_pk)).fetchone()
            if row is None or (row[0], row[1]) != (snapshot.get('version'), snapshot.get('sha')) or row[2] > decided:
                raise ValueError('Recommendation snapshot is not a committed version received by the decision time')
            held = db.execute("SELECT body FROM served_recommendations WHERE decision_id=? AND outcome='first'",
                              (decision_id,)).fetchone()
            return row, json.loads(held[0]) if held else None

        with closing(self.connect()) as db:
            _, first = look(db)
        if first is not None and _canonical(first['displayed']) == shown:
            return 'duplicate', first
        with self._write() as db:
            row, first = look(db)  # again, now serialized with every other writer
            if first is not None and _canonical(first['displayed']) == shown:
                return 'duplicate', first
            clock = RECEIPT_CLOCK.get(row[3], 'unverified')
            record = {'schema': 'served-recommendation-v1', 'decision_id': decision_id, 'game_pk': game_pk,
                      'pitch_key': pitch_key, 'state_sha256': state_sha, 'policy_identity': policy_identity,
                      'decided': decided, 'decision_clock': clock, 'feed_ok': bool(feed_ok),
                      'snapshot': {'store_seq': snapshot['seq'], 'slot_version': row[0], 'feed_sha256': row[1],
                                   'source': row[3], 'received': row[2],
                                   'provider_order': snapshot.get('provider_order')},
                      'displayed': json.loads(shown),
                      'target': {'status': 'UNKNOWN', 'reason': 'next_pitch_not_observed_at_decision'}}
            body = _canonical(record)
            sha = hashlib.sha256(shown).hexdigest()
            outcome = 'first' if first is None else 'conflict'
            if outcome == 'conflict' and db.execute('SELECT 1 FROM served_recommendations WHERE decision_id=? AND sha=?',
                                                    (decision_id, sha)).fetchone():
                return 'conflict', first  # this differing content is already on record
            db.execute('INSERT INTO served_recommendations(decision_id, game_pk, pitch_key, outcome, decided, '
                       'decision_clock, store_seq, sha, body) VALUES (?,?,?,?,?,?,?,?,?)',
                       (decision_id, game_pk, pitch_key, outcome, decided, clock, snapshot['seq'], sha, body))
        return outcome, record if first is None else first

    def served_recommendations(self, game_pk):
        """Every ledger row of a game in append order (``first`` rows were displayable, ``conflict`` rows never)."""
        with closing(self.connect()) as db:
            rows = db.execute('SELECT id, outcome, body FROM served_recommendations WHERE game_pk=? ORDER BY id',
                              (int(game_pk),)).fetchall()
        return [{'id': i, 'outcome': outcome} | json.loads(body) for i, outcome, body in rows]

    def restore(self, game_pk, wall, max_delay_s):
        """The in-memory buffer ``LiveFeed`` would hold at ``wall``: the current version of every slot newer
        than ``wall - max_delay_s`` plus the newest one at or before it. Slots newer than a viewer's delay are
        kept but, as in memory, only served once that viewer's delay has passed."""
        bound = wall-max_delay_s
        with closing(self.connect()) as db:
            floor = db.execute('SELECT MAX(event_wall) FROM snapshots WHERE game_pk=? AND event_wall<=?',
                               (game_pk, bound)).fetchone()[0]
            rows = db.execute(
                'SELECT event_wall, first_fetched, body, seq, version, sha, fetched, source, feed_ts, requested FROM snapshots s WHERE game_pk=? AND event_wall>=? AND version='
                '(SELECT MAX(version) FROM snapshots WHERE game_pk=s.game_pk AND event_wall=s.event_wall) ORDER BY event_wall',
                (game_pk, bound if floor is None else floor)).fetchall()
            return [{'game_pk': int(game_pk), 'feed': json.loads(gzip.decompress(body)), 'event': event, 'fetched': first,
                     'seq': seq, 'version': version, 'sha': sha, 'version_fetched': received,
                     'source': source, 'restored': True, 'provider_order': _provider_order(db, game_pk, feed_ts, seq, sha),
                     'requested': requested if version == 1 and first == received else None, 'version_requested': requested}
                    for event, first, body, seq, version, sha, received, source, feed_ts, requested in rows]

    def read_after(self, game_pk, after=0, *, until_event, limit=100):
        """Committed versions of ``game_pk`` with ``seq > after`` whose game event is at or before
        ``until_event``, oldest first, and the cursor to pass next time. Writers are serialized and a
        correction only ever targets the newest slot, so per game ``seq`` order is ``event_wall`` order:
        the delay bound cuts a suffix and the cursor never skips a row it has not returned."""
        with closing(self.connect()) as db:
            rows = db.execute('SELECT seq, event_wall, version, feed_ts, sha, first_fetched, fetched, source FROM snapshots '
                              'WHERE game_pk=? AND seq>? AND event_wall<=? ORDER BY seq LIMIT ?',
                              (int(game_pk), int(after), until_event, int(limit))).fetchall()
        keys = ('seq', 'event_wall', 'version', 'feed_ts', 'sha', 'first_fetched', 'fetched', 'source')
        items = [dict(zip(keys, row)) for row in rows]
        return {'items': items, 'cursor': items[-1]['seq'] if items else int(after)}

    def status(self):
        with closing(self.connect()) as db:
            games = db.execute('SELECT game_pk, COUNT(*), COUNT(DISTINCT event_wall), MAX(seq), MAX(fetched) '
                               'FROM snapshots GROUP BY game_pk ORDER BY game_pk').fetchall()
            check = db.execute('PRAGMA quick_check').fetchone()[0]
        return {'path': str(self.path), 'quick_check': check,
                'games': [{'game_pk': g, 'versions': n, 'slots': s, 'max_seq': m, 'last_fetched': f} for g, n, s, m, f in games]}

    def backup(self, target):
        """Consistent online copy (SQLite backup API), safe while the service writes."""
        target = Path(target)
        if target.exists():
            raise FileExistsError(target)
        with closing(self.connect()) as source, closing(sqlite3.connect(target)) as dest:
            source.backup(dest)
        return target


# What ties a buffer entry to the committed version whose bytes its ``feed`` is.
IDENTITY = ('seq', 'version', 'sha', 'version_fetched', 'source', 'game_pk', 'provider_order', 'version_requested')


class PersistentLiveFeed(LiveFeed):
    """``LiveFeed`` whose delay buffer is written through to a ``LiveSnapshotStore`` and rebuilt from it the first
    time a game is asked for after a restart. The memory buffer changes only after the commit; if the store
    fails, serving falls back to the plain in-memory behaviour and ``store_error`` says so."""

    def __init__(self, transport, *, store, source='live_poll', **options):
        """``source`` labels every polled snapshot row. Only 'live_poll' (a real upstream poll) puts pitch
        receipts on the 'observer_local' clock; a replayed recording must say so ('replay_poll') and is recorded
        with an 'unverified' receipt clock, because its arrival times are the replay's, not the game's."""
        super().__init__(transport, **options)
        self.source = source
        self.store, self.store_error, self.stale = store, None, {}
        self._unsaved = set()
        self._restore_failed = False

    def _buffer(self, game_pk):
        if game_pk not in self.buffers:
            try:
                self.buffers[game_pk] = self.store.restore(game_pk, self.wall(), self.max_delay_s)
            except (sqlite3.Error, OSError, EOFError, ValueError, zlib.error) as exc:
                self._restore_failed = True
                self._failed('restore', exc)
                self.buffers[game_pk] = []
        return self.buffers[game_pk]

    def _failed(self, operation, exc):
        self.store_error = f'{operation}: {type(exc).__name__}'
        LOGGER.warning('live store %s failed; serving from memory only', operation, exc_info=True)

    def _accept(self, game_pk, feed, wall, buffer, requested=None):
        # Memory may be ahead of disk during an outage, so stale input is also judged against memory, but only
        # when the store is down: once the flush below makes disk >= memory, ingest decides and keeps the receipt.
        stale = bool(buffer and _stale_stamp(_feed_ts(feed), _feed_ts(buffer[-1]['feed']),
                                             (event_wall(feed) or wall) > buffer[-1]['event']))
        try:
            if game_pk in self._unsaved:
                # ponytail: resync only the bounded delay window; pruned outage history is not recoverable.
                for saved in buffer:
                    _, kept = self.store.ingest(game_pk, saved['feed'], fetched=saved.get('version_fetched', saved['fetched']),
                                                source='memory_recovery', slot_event=saved['event'], first_fetched=saved['fetched'],
                                                requested=saved.get('version_requested', saved.get('requested')))
                    if kept and kept['event'] == saved['event']:  # now committed: it has a store identity again
                        saved.update({key: kept[key] for key in IDENTITY})
                self._unsaved.discard(game_pk)
            outcome, entry = self.store.ingest(game_pk, feed, fetched=wall, source=self.source, requested=requested)
        except (sqlite3.Error, OSError) as exc:
            self._unsaved.add(game_pk)
            self._failed('ingest', exc)
            if stale:  # received but never persisted: the known store-outage loss bound
                self.stale[game_pk] = self.stale.get(game_pk, 0)+1
                return
            changed = not buffer or feed != buffer[-1]['feed']
            super()._accept(game_pk, feed, wall, buffer, requested)
            if changed:
                buffer[-1]['version_fetched'], buffer[-1]['version_requested'] = wall, requested
                for key in ('seq', 'version', 'sha', 'provider_order'):  # memory-only content: no committed identity
                    buffer[-1].pop(key, None)
            return
        if not self._unsaved and not self._restore_failed:
            self.store_error = None
        if outcome == 'stale':  # an older feed answered late: never rewind what viewers already saw
            self.stale[game_pk] = self.stale.get(game_pk, 0)+1
        elif buffer and entry['event'] == buffer[-1]['event']:
            buffer[-1]['feed'] = feed
            if outcome == 'correction' or buffer[-1].get('sha') != entry['sha']:  # the latter: identity lost in an outage
                buffer[-1].update({key: entry[key] for key in IDENTITY})
        elif not buffer or entry['event'] > buffer[-1]['event']:
            buffer.append(entry)

    def sync_extra(self, game_pk, entry):
        return {'restored': bool(entry and entry.get('restored')), 'store_ok': self.store_error is None,
                'stale_dropped': self.stale.get(int(game_pk), 0)}


def ingest_dir(store, directory, *, game=None, limit=None):
    """Feeds recorded snapshots (read only) through ``ingest`` in recording order. The recorder writes each
    file right after its fetch, so the file's mtime stands in for the arrival time."""
    files = sorted(Path(directory).glob(f'{game or "[0-9]*"}_*.json*'))
    counts = {}
    for path in files[:limit]:
        game_pk = int(re.match(r'(\d+)_', path.name)[1])
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as handle:
            feed = json.load(handle)
        outcome, _ = store.ingest(game_pk, feed, fetched=path.stat().st_mtime, source=f'recorded:{path.name}')
        counts[outcome] = counts.get(outcome, 0)+1
    return {'files': len(files[:limit]), 'outcomes': counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['ingest', 'status', 'backup'])
    parser.add_argument('--db', required=True)
    parser.add_argument('--dir')
    parser.add_argument('--game', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--to')
    args = parser.parse_args(argv)
    store = LiveSnapshotStore(args.db)
    if args.command == 'ingest':
        result = ingest_dir(store, args.dir, game=args.game, limit=args.limit)
    elif args.command == 'backup':
        result = {'backup': str(store.backup(args.to))}
    else:
        result = store.status()
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
