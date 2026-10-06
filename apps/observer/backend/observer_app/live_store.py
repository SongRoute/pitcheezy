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
* same or older event, older ``metaData.timeStamp`` than the newest slot -> ``stale`` (out of order; dropped)
* same content as the newest slot's current version -> ``duplicate`` (no write)
* otherwise -> ``correction`` of the newest slot (first-seen times kept, as the in-memory buffer does)

    python -m observer_app.live_store ingest --db D --dir apps/observer/live_snapshots [--game 849844] [--limit N]
    python -m observer_app.live_store status --db D
    python -m observer_app.live_store backup --db D --to D.bak
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
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
    UNIQUE(game_pk, event_wall, version));
CREATE INDEX IF NOT EXISTS snapshots_game_event ON snapshots(game_pk, event_wall, version);
CREATE TRIGGER IF NOT EXISTS snapshots_append_only
    BEFORE UPDATE ON snapshots BEGIN SELECT RAISE(ABORT, 'append-only snapshot'); END;
'''


def _canonical(feed):
    return json.dumps(feed, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


class LiveSnapshotStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript(SCHEMA)

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

    def ingest(self, game_pk, feed, *, fetched, source, slot_event=None):
        """Returns (outcome, entry); ``entry`` is the ``LiveFeed`` buffer entry for the slot (None if stale)."""
        game_pk, body = int(game_pk), _canonical(feed)
        event, sha = event_wall(feed) if slot_event is None else slot_event, hashlib.sha256(body).hexdigest()
        event = fetched if event is None else min(event, fetched)
        feed_ts = (feed.get('metaData') or {}).get('timeStamp')
        with self._write() as db:
            newest = db.execute('SELECT event_wall, version, feed_ts, sha, first_fetched FROM snapshots WHERE game_pk=? '
                                'ORDER BY event_wall DESC, version DESC LIMIT 1', (game_pk,)).fetchone()
            if newest is None or event > newest[0]:
                outcome, slot, version, first = 'new', event, 1, fetched
            elif feed_ts and newest[2] and feed_ts < newest[2]:
                return 'stale', None
            elif sha == newest[3]:
                return 'duplicate', {'feed': feed, 'event': newest[0], 'fetched': newest[4]}
            else:
                outcome, slot, version, first = 'correction', newest[0], newest[1]+1, newest[4]
            db.execute('INSERT INTO snapshots(game_pk, event_wall, version, feed_ts, sha, first_fetched, fetched, source, body) '
                       'VALUES (?,?,?,?,?,?,?,?,?)', (game_pk, slot, version, feed_ts, sha, first, fetched, source,
                                                      gzip.compress(body, compresslevel=5)))
        return outcome, {'feed': feed, 'event': slot, 'fetched': first}

    def restore(self, game_pk, wall, max_delay_s):
        """The in-memory buffer ``LiveFeed`` would hold at ``wall``: the current version of every slot newer
        than ``wall - max_delay_s`` plus the newest one at or before it. Slots newer than a viewer's delay are
        kept but, as in memory, only served once that viewer's delay has passed."""
        bound = wall-max_delay_s
        with closing(self.connect()) as db:
            floor = db.execute('SELECT MAX(event_wall) FROM snapshots WHERE game_pk=? AND event_wall<=?',
                               (game_pk, bound)).fetchone()[0]
            rows = db.execute(
                'SELECT event_wall, first_fetched, body FROM snapshots s WHERE game_pk=? AND event_wall>=? AND version='
                '(SELECT MAX(version) FROM snapshots WHERE game_pk=s.game_pk AND event_wall=s.event_wall) ORDER BY event_wall',
                (game_pk, bound if floor is None else floor)).fetchall()
        return [{'feed': json.loads(gzip.decompress(body)), 'event': event, 'fetched': first, 'restored': True}
                for event, first, body in rows]

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


class PersistentLiveFeed(LiveFeed):
    """``LiveFeed`` whose delay buffer is written through to a ``LiveSnapshotStore`` and rebuilt from it the first
    time a game is asked for after a restart. The memory buffer changes only after the commit; if the store
    fails, serving falls back to the plain in-memory behaviour and ``store_error`` says so."""

    def __init__(self, transport, *, store, **options):
        super().__init__(transport, **options)
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

    def _accept(self, game_pk, feed, wall, buffer):
        # Reject stale input against memory too: it may be ahead of disk after an outage.
        feed_ts = (feed.get('metaData') or {}).get('timeStamp')
        newest_ts = (buffer[-1]['feed'].get('metaData') or {}).get('timeStamp') if buffer else None
        stale = bool(buffer and (event_wall(feed) or wall) <= buffer[-1]['event']
                     and feed_ts and newest_ts and feed_ts < newest_ts)
        try:
            if game_pk in self._unsaved:
                # ponytail: resync only the bounded delay window; pruned outage history is not recoverable.
                for saved in buffer:
                    self.store.ingest(game_pk, saved['feed'], fetched=saved['fetched'], source='memory_recovery', slot_event=saved['event'])
                self._unsaved.discard(game_pk)
            if stale:
                self.stale[game_pk] = self.stale.get(game_pk, 0)+1
                if not self._unsaved and not self._restore_failed:
                    self.store_error = None
                return
            outcome, entry = self.store.ingest(game_pk, feed, fetched=wall, source='live_poll')
        except (sqlite3.Error, OSError) as exc:
            self._unsaved.add(game_pk)
            self._failed('ingest', exc)
            if stale:
                self.stale[game_pk] = self.stale.get(game_pk, 0)+1
                return
            return super()._accept(game_pk, feed, wall, buffer)
        if not self._unsaved and not self._restore_failed:
            self.store_error = None
        if outcome == 'stale':  # an older feed answered late: never rewind what viewers already saw
            self.stale[game_pk] = self.stale.get(game_pk, 0)+1
        elif buffer and entry['event'] == buffer[-1]['event']:
            buffer[-1]['feed'] = feed
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
