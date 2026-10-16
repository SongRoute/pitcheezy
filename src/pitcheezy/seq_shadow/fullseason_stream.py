"""Isolated normalized-pitch state for the fullseason seven-leaf shadow.

This module does not normalize a feed or establish its completeness. The caller
must supply accepted-row quality and complete stored-prefix evidence under its
registered normalizer. Captured receipt != upstream first availability; neither
retrospective inputs nor an observed pitch type establish a pre-pitch forecast.

Existing SeqTables binds N1 forecasts to Step5 pitch_attempts, so it cannot store
this state. Reuse its canonical hashing and append-only SQLite pattern instead.
One DB per game/normalizer/evidence kind. No observer or collector connection.
"""
from __future__ import annotations

from datetime import date
import json
import sqlite3

import numpy as np
from experiments.ws_campaign.source.first_small_features import TYPES
from .ledger import canonical, sha
from .tokens import FIELDS

KEY = ('game_pk', 'at_bat_number', 'pitch_number')
# Exact frozen scale_safe.CONTEXT; current realized outcome/physics are absent.
CONTEXT = KEY + ('game_date', 'pitcher', 'batter', 'pitch_type', 'balls', 'strikes',
    'outs_when_up', 'inning', 'inning_topbot', 'home_score', 'away_score',
    'on_1b', 'on_2b', 'on_3b', 'stand', 'p_throws')
SCHEMA = CONTEXT + ('previous_type', 'previous_valid')
ACCEPTED = ('pitcher', 'batter', 'pitch_type', 'outcome10', 'balls', 'strikes')
KINDS = ('synthetic', 'captured', 'retrospective')
MAX_KEYS = 10000
MAX_EVENTS = 100000
MAX_DECISIONS = 100000
_SCHEMA = '''
CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
 seq INTEGER PRIMARY KEY, ab INTEGER NOT NULL, pitch INTEGER NOT NULL,
 event_id TEXT NOT NULL, revision INTEGER NOT NULL, received_ns INTEGER NOT NULL,
 first_received_ns INTEGER NOT NULL, status TEXT NOT NULL, pitcher INTEGER,
 body TEXT NOT NULL, digest TEXT NOT NULL, UNIQUE(ab,pitch,revision), UNIQUE(event_id,revision));
CREATE INDEX IF NOT EXISTS latest_pitch ON events(ab,pitch,seq DESC);
CREATE INDEX IF NOT EXISTS event_identity ON events(event_id,seq DESC);
CREATE TABLE IF NOT EXISTS receipt_clock (id INTEGER PRIMARY KEY CHECK(id=1), latest_ns INTEGER NOT NULL);
CREATE TRIGGER IF NOT EXISTS clock_no_regression BEFORE UPDATE ON receipt_clock
 WHEN NEW.latest_ns < OLD.latest_ns BEGIN SELECT RAISE(ABORT,'receipt clock regressed'); END;
CREATE TRIGGER IF NOT EXISTS clock_no_delete BEFORE DELETE ON receipt_clock
 BEGIN SELECT RAISE(ABORT,'receipt clock cannot be deleted'); END;
CREATE TABLE IF NOT EXISTS decisions (
 decision_id TEXT PRIMARY KEY, request_sha TEXT NOT NULL, received_ns INTEGER NOT NULL,
 body TEXT NOT NULL);
''' + ''.join(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{verb}
 BEFORE {verb} ON {table} BEGIN SELECT RAISE(ABORT,'append-only shadow state'); END;
''' for table in ('identity', 'events', 'decisions') for verb in ('UPDATE', 'DELETE'))


def _integer(value, low=0, high=(1 << 53)-1):
    return type(value) is int and low <= value <= high


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _text(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def _key(key):
    if not isinstance(key, (tuple, list)) or len(key) != 3 or not all(_integer(v, 1) for v in key):
        raise ValueError('positive integer physical key required')
    return tuple(key)


def _row(row):
    if not isinstance(row, dict) or set(row) != {'status', 'reason', 'accepted'}:
        raise ValueError('exact normalized row schema required')
    if row['status'] not in ('accepted', 'rejected', 'unknown') or not _text(row['reason']):
        raise ValueError('explicit row fate/reason required')
    r = row['accepted']
    if row['status'] != 'accepted':
        if r is not None: raise ValueError('unaccepted row cannot supply tokens')
    elif (not isinstance(r, dict) or set(r) != set(ACCEPTED)
            or not all(_integer(r[k], 1) for k in ('pitcher', 'batter'))
            or r['pitch_type'] not in TYPES or not _integer(r['outcome10'], 0, 9)
            or not _integer(r['balls'], 0, 3) or not _integer(r['strikes'], 0, 2)):
        raise ValueError('accepted token vocabulary/identity')
    return canonical(row)


def _query(query, game, game_date):
    if not isinstance(query, dict) or set(query) != set(CONTEXT):
        raise ValueError('exact pre-pitch query schema; no realization fields')
    key = _key([query[k] for k in KEY])
    if key[0] != game or query['game_date'] != game_date:
        raise ValueError('query game/date identity')
    for k in ('pitcher', 'batter', 'inning'):
        if not _integer(query[k], 1): raise ValueError('query identity/inning')
    for k, hi in (('balls', 3), ('strikes', 2), ('outs_when_up', 2)):
        if not _integer(query[k], 0, hi): raise ValueError('query count')
    for k in ('home_score', 'away_score'):
        if not _integer(query[k]): raise ValueError('query score')
    for k in ('on_1b', 'on_2b', 'on_3b'):
        if query[k] is not None and not _integer(query[k], 1): raise ValueError('query runner')
    if (query['pitch_type'] not in TYPES or query['stand'] not in ('L', 'R')
            or query['p_throws'] not in ('L', 'R') or query['inning_topbot'] not in ('Top', 'Bot')):
        raise ValueError('query vocabulary')
    return key


class StreamState:
    """Single-owner serialized API; explicit close releases only this connection.

    ponytail: SQLite latest-version queries scan a bounded game (<=10k keys,
    <=100k revisions/decisions), not an in-memory season. Multi-game service needs routing,
    not a larger global cache. The fixed caps refuse intake instead of evicting
    evidence needed for restart/first-decision replay.
    """
    def __init__(self, path, *, game_pk, game_date, normalizer_sha256, evidence_kind):
        if (not _integer(game_pk, 1) or evidence_kind not in KINDS
                or not _digest(normalizer_sha256) or date.fromisoformat(game_date).isoformat() != game_date):
            raise ValueError('explicit game/normalizer/evidence identity')
        self.identity = dict(game_pk=game_pk, game_date=game_date,
            normalizer_sha256=normalizer_sha256, evidence_kind=evidence_kind)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.executescript(_SCHEMA)
            with self.db:
                found = self.db.execute('SELECT body FROM identity WHERE id=1').fetchone()
                if found and found['body'] != canonical(self.identity):
                    raise ValueError('stream identity changed')
                if not found: self.db.execute('INSERT INTO identity VALUES(1,?)', (canonical(self.identity),))
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def _clock(self, received_ns):
        if not _integer(received_ns, 1, (1 << 63)-1): raise ValueError('receipt integer nanoseconds required')
        prior = self.db.execute('''SELECT MAX(t) FROM (
            SELECT MAX(received_ns) t FROM events UNION ALL SELECT MAX(received_ns) t FROM decisions
            UNION ALL SELECT latest_ns t FROM receipt_clock)''').fetchone()[0]
        if prior is not None and received_ns < prior:
            raise ValueError('receipt clock regressed; do not backdate availability')

    def _advance_clock(self, received_ns):
        # One bounded mutable watermark, separate from immutable observations.
        # A duplicate receipt is still observed time, including across restart.
        self.db.execute('INSERT INTO receipt_clock VALUES(1,?) ON CONFLICT(id) DO UPDATE SET latest_ns=excluded.latest_ns', (received_ns,))

    def ingest(self, event_id, key, revision, row, *, received_ns):
        key = _key(key); body = _row(row)
        if key[0] != self.identity['game_pk'] or not _text(event_id) or not _integer(revision, 1):
            raise ValueError('event identity/revision')
        digest = sha(row)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            self._clock(received_ns)
            old = self.db.execute('SELECT * FROM events WHERE ab=? AND pitch=? ORDER BY seq DESC LIMIT 1', key[1:]).fetchone()
            alias = self.db.execute('SELECT ab,pitch FROM events WHERE event_id=? LIMIT 1', (event_id,)).fetchone()
            if (old and old['event_id'] != event_id) or (alias and tuple(alias) != key[1:]):
                raise ValueError('physical/provider identity reassignment')
            if old:
                if revision == old['revision'] and digest == old['digest']:
                    self._advance_clock(received_ns)
                    return {'status': 'duplicate', 'seq': old['seq'], 'first_received_ns': old['first_received_ns']}
                if revision <= old['revision']: raise ValueError('stale or unordered revision')
            count = self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            if count >= MAX_EVENTS or (not old and self.db.execute('SELECT COUNT(*) FROM (SELECT 1 FROM events GROUP BY ab,pitch)').fetchone()[0] >= MAX_KEYS):
                raise ValueError('bounded game ledger capacity')
            first = old['first_received_ns'] if old else received_ns
            r = row['accepted']
            cur = self.db.execute('''INSERT INTO events(ab,pitch,event_id,revision,received_ns,
                first_received_ns,status,pitcher,body,digest) VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (*key[1:], event_id, revision, received_ns, first, row['status'],
                 r['pitcher'] if r else None, body, digest))
            self._advance_clock(received_ns)
            return {'status': 'correction' if old else 'new', 'seq': cur.lastrowid, 'first_received_ns': first}

    def decide(self, decision_id, query, *, pitch_type_basis, coverage, received_ns):
        """Freeze one decision's query/context. Repeating its ID returns its original snapshot.

        coverage is caller evidence of a complete *stored* normalized prefix, not
        completeness of a provider or upstream arrival. Current key/outcome never
        enters the query/context. No return value claims a prospective forecast.
        """
        key = _query(query, self.identity['game_pk'], self.identity['game_date'])
        if not _text(decision_id) or pitch_type_basis not in ('hypothetical_candidate', 'observed_actual'):
            raise ValueError('decision identity and conditional pitch-type basis required')
        if (not isinstance(coverage, dict) or set(coverage) != {'complete', 'before_key', 'source_sha256'}
                or type(coverage['complete']) is not bool or _key(coverage['before_key']) != key
                or not _digest(coverage['source_sha256'])):
            raise ValueError('explicit prefix evidence binding required')
        request = dict(query=query, pitch_type_basis=pitch_type_basis, coverage=coverage)
        request_sha = sha(request)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT * FROM decisions WHERE decision_id=?', (decision_id,)).fetchone()
            if old:
                if old['request_sha'] != request_sha: raise ValueError('decision identity reused for changed request')
                return json.loads(old['body'])
            self._clock(received_ns)
            if self.db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0] >= MAX_DECISIONS:
                raise ValueError('bounded game decision capacity')
            # Every latest version strictly before this key; correction availability
            # is this transaction's receipt sequence, never retroactively backdated.
            latest = ''' FROM events e WHERE (e.ab,e.pitch)<(?,?) AND e.seq=(
                SELECT MAX(v.seq) FROM events v WHERE v.ab=e.ab AND v.pitch=e.pitch)'''
            unknown = self.db.execute('SELECT 1'+latest+" AND status='unknown' LIMIT 1", key[1:]).fetchone()
            reason = 'stored_prefix_incomplete' if not coverage['complete'] else ('prior_quality_unknown' if unknown else None)
            seq = self.db.execute('SELECT COALESCE(MAX(seq),0) FROM events').fetchone()[0]
            result = dict(status='abstain' if reason else 'ready', reason=reason,
                identity=dict(self.identity), evidence=dict(coverage), pitch_type_basis=pitch_type_basis,
                prospective_claim=False, received_ns=received_ns, event_seq=seq,
                query=None, tokens=None)
            if reason is None:
                previous = self.db.execute('SELECT body FROM events WHERE ab=? AND pitch=? ORDER BY seq DESC LIMIT 1',
                                           (key[1], key[2]-1)).fetchone()
                previous = json.loads(previous['body']) if previous else None
                valid = previous is not None and previous['status'] == 'accepted'
                result['query'] = dict(query, previous_type=previous['accepted']['pitch_type'] if valid else 'UNKNOWN', previous_valid=valid)
                past = self.db.execute('SELECT e.*'+latest+" AND status='accepted' AND pitcher=? ORDER BY ab DESC,pitch DESC LIMIT 32",
                                       (*key[1:], query['pitcher'])).fetchall()
                t = {k: np.zeros((1, 32), np.int64) for k in FIELDS}
                t['pos'][0] = np.arange(1, 33); t['valid'] = np.zeros((1, 32), bool)
                t['context_keys'] = np.zeros((1, 32, 3), np.int64)
                t['context_pitcher'] = np.zeros((1, 32), np.int64)
                t['query_pitcher'] = np.array([query['pitcher']], np.int64)
                for j, event in enumerate(past):
                    r = json.loads(event['body'])['accepted']
                    for name, value in zip(FIELDS, (TYPES.index(r['pitch_type']), r['outcome10'], r['balls'], r['strikes'], int(r['batter'] == query['batter']), j+1)):
                        t[name][0, j] = value
                    t['valid'][0, j] = True
                    t['context_keys'][0, j] = (key[0], event['ab'], event['pitch'])
                    t['context_pitcher'][0, j] = r['pitcher']
                result['tokens'] = {k: v.tolist() for k, v in t.items()}
            result['snapshot_sha256'] = sha(result)
            self.db.execute('INSERT INTO decisions VALUES(?,?,?,?)', (decision_id, request_sha, received_ns, canonical(result)))
            return result


def arrays(snapshot):
    """Materialize the small immutable snapshot for Ensemble.predict/ProfileBundle.raw91."""
    if snapshot['status'] != 'ready': raise ValueError('abstained snapshot has no model inputs')
    body = dict(snapshot); digest = body.pop('snapshot_sha256')
    if sha(body) != digest: raise ValueError('snapshot changed')
    return {k: np.asarray(v, dtype=bool if k == 'valid' else np.int64) for k, v in snapshot['tokens'].items()}
