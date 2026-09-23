"""SQLite persistence for replay cursors, immutable recommendations and leased jobs."""
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import uuid


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, game_id INTEGER NOT NULL, pa_id INTEGER NOT NULL,
                    dataset_identity TEXT NOT NULL, cursor INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS recommendations (
                    session_id TEXT NOT NULL REFERENCES sessions(id), pitch_id TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(session_id,pitch_id));
                CREATE TRIGGER IF NOT EXISTS recommendations_immutable
                    BEFORE UPDATE ON recommendations BEGIN SELECT RAISE(ABORT,'immutable recommendation'); END;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, pitch_id TEXT NOT NULL, version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until REAL, lease_token TEXT, revision INTEGER NOT NULL DEFAULT 1,
                    message TEXT, cv_status TEXT NOT NULL DEFAULT 'unavailable:no_media', UNIQUE(pitch_id,version));
                CREATE TABLE IF NOT EXISTS manual_annotations (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id), pitch_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS recommendation_meta (
                    session_id TEXT NOT NULL, pitch_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(session_id,pitch_id),
                    FOREIGN KEY(session_id,pitch_id) REFERENCES recommendations(session_id,pitch_id));
                CREATE TABLE IF NOT EXISTS event_results (
                    session_id TEXT NOT NULL REFERENCES sessions(id), pitch_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(session_id,pitch_id,revision));
                CREATE TRIGGER IF NOT EXISTS event_results_immutable
                    BEFORE UPDATE ON event_results BEGIN SELECT RAISE(ABORT,'immutable event result'); END;
            ''')
            if 'cv_status' not in {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}:
                db.execute("ALTER TABLE jobs ADD COLUMN cv_status TEXT NOT NULL DEFAULT 'unavailable:no_media'")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=30000')
        return db

    @contextmanager
    def transaction(self, *, write=True):
        db = self.connect()
        try:
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def save_recommendation(db, session_id, pitch_id, recommendation):
        db.execute('INSERT OR IGNORE INTO recommendations VALUES (?,?,?)',
                   (session_id, pitch_id, json.dumps(recommendation, ensure_ascii=False, allow_nan=False)))
        db.execute('INSERT OR IGNORE INTO recommendation_meta VALUES (?,?,?)',
                   (session_id, pitch_id, datetime.now(timezone.utc).isoformat()))

    @staticmethod
    def recommendation(db, session_id, pitch_id):
        row = db.execute('SELECT payload FROM recommendations WHERE session_id=? AND pitch_id=?',
                         (session_id, pitch_id)).fetchone()
        return json.loads(row['payload']) if row else None

    @staticmethod
    def event_result(db, session_id, pitch_id):
        row = db.execute('SELECT payload FROM event_results WHERE session_id=? AND pitch_id=? ORDER BY revision DESC LIMIT 1',
                         (session_id, pitch_id)).fetchone()
        return json.loads(row['payload']) if row else None

    @staticmethod
    def save_event_result(db, session_id, pitch_id, result):
        db.execute('INSERT OR IGNORE INTO event_results VALUES (?,?,?,?)',
                   (session_id, pitch_id, result['revision'], json.dumps(result, ensure_ascii=False, allow_nan=False)))

    @staticmethod
    def enqueue(db, pitch_id, version):
        db.execute('INSERT OR IGNORE INTO jobs(id,pitch_id,version) VALUES (?,?,?)',
                   (str(uuid.uuid4()), pitch_id, version))

    def claim(self, lease_seconds=30, max_attempts=2, now=None):
        now = time.time() if now is None else now
        with self.transaction() as db:
            db.execute("""UPDATE jobs SET status='failed',lease_token=NULL,lease_until=NULL,
                       revision=revision+1,message=? WHERE status='running' AND lease_until<=? AND attempts>=?""",
                       ('분석 작업이 중단되었습니다. 영상이 없어 자동 분석은 제공되지 않습니다.', now, max_attempts))
            row = db.execute("""SELECT * FROM jobs WHERE attempts<? AND
                (status='queued' OR (status='running' AND lease_until<=?)) ORDER BY rowid LIMIT 1""",
                (max_attempts, now)).fetchone()
            if row is None:
                return None
            token = str(uuid.uuid4())
            db.execute("""UPDATE jobs SET status='running',attempts=attempts+1,lease_until=?,
                       lease_token=?,revision=revision+1 WHERE id=?""", (now+lease_seconds, token, row['id']))
            return dict(db.execute('SELECT * FROM jobs WHERE id=?', (row['id'],)).fetchone())

    def finish(self, job_id, lease_token, status='unavailable', message=None, now=None,
               cv_status='unavailable:no_media'):
        if status not in ('unavailable', 'failed'):
            raise ValueError('No-media worker cannot claim successful CV')
        if cv_status not in ('unavailable:no_media', 'unavailable:adapter_not_configured'):
            raise ValueError('No video adapter is configured')
        now = time.time() if now is None else now
        with self.transaction() as db:
            result = db.execute("""UPDATE jobs SET status=?,message=?,cv_status=?,lease_token=NULL,lease_until=NULL,
                    revision=revision+1 WHERE id=? AND status='running' AND lease_token=? AND lease_until>?""",
                    (status, message, cv_status, job_id, lease_token, now))
            return result.rowcount == 1
