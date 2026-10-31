"""Append-only SQLite tables of the sequence-model shadow, in the same file as the accepted Step 5 ``DecisionLedger``
tables and the Step 5 shadow tables (``shadow_run``, ``shadow_forecasts``, ``shadow_realized``).

  seq_run        id = 1, the run information as canonical JSON
  seq_forecasts  ONE immutable row per pitch attempt: the N1 forecast made from the body of the first decision
                 (decision 'scored'), or an abstention carrying the Step 5 reason (decision 'abstain')

UPDATE and DELETE abort on both. ``row_sha256`` covers every column except the wall-clock timings, so two replays
of the same bodies give the same value. The Step 5 ``shadow_forecasts.action_probabilities`` written by THIS run
come from placeholder heads and must not be read; ``seq_forecasts`` deliberately stores the Step 5 cell statuses
without any ``seven_leaf``.
"""
from __future__ import annotations

import hashlib
import json

SEQ_TABLES = ('seq_run', 'seq_forecasts')
TIMING_COLUMNS = ('compute_ms', 'tokens_ms', 'forward_ms', 'calibration_ms', 'features_ms')
SEQ_SCHEMA = '''
CREATE TABLE IF NOT EXISTS seq_run (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS seq_forecasts (
    attempt_key TEXT PRIMARY KEY REFERENCES pitch_attempts(attempt_key),
    game_pk INTEGER NOT NULL,
    at_bat_number INTEGER NOT NULL,
    candidate_pitch_number INTEGER NOT NULL,
    written_at_replay_index INTEGER NOT NULL,
    replay_file TEXT,
    feed_sha256 TEXT,
    pitcher INTEGER,
    batter INTEGER,
    context_sha256 TEXT,
    vectors_sha256 TEXT,
    x_seq_sha256 TEXT,
    tokens_json TEXT,
    tokens_sha256 TEXT,
    decision TEXT NOT NULL CHECK (decision IN ('scored', 'abstain')),
    reason TEXT,
    step5_outcome TEXT NOT NULL,
    step5_cells_status_json TEXT NOT NULL,
    logits_json TEXT,
    c0_ensemble_json TEXT,
    sel_ensemble_json TEXT,
    cal_arms_json TEXT,
    model_sha256_json TEXT,
    calibrator_sha256 TEXT,
    config_sha256 TEXT NOT NULL,
    source_sha256_json TEXT NOT NULL,
    row_sha256 TEXT NOT NULL,
    compute_ms REAL,
    tokens_ms REAL,
    forward_ms REAL,
    calibration_ms REAL,
    features_ms REAL,
    CHECK ((decision = 'scored') = (sel_ensemble_json IS NOT NULL)));
'''+''.join(f'''
CREATE TRIGGER IF NOT EXISTS {table}_no_{verb} BEFORE {verb} ON {table}
    BEGIN SELECT RAISE(ABORT, 'append-only seq shadow ledger'); END;''' for table in SEQ_TABLES for verb in ('update', 'delete'))
HASHED_COLUMNS = ('attempt_key', 'game_pk', 'at_bat_number', 'candidate_pitch_number', 'written_at_replay_index',
                  'replay_file', 'feed_sha256', 'pitcher', 'batter', 'context_sha256', 'vectors_sha256', 'x_seq_sha256',
                  'tokens_json', 'tokens_sha256', 'decision', 'reason', 'step5_outcome', 'step5_cells_status_json',
                  'logits_json', 'c0_ensemble_json', 'sel_ensemble_json', 'cal_arms_json', 'model_sha256_json',
                  'calibrator_sha256', 'config_sha256', 'source_sha256_json')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class SeqTables:
    """Writer of the two tables on the connection of an open ``DecisionLedger`` (autocommit, synchronous=FULL: every
    INSERT is its own committed transaction)."""

    def __init__(self, db, run_info, *, config_sha256, source_sha256, model_sha256, calibrator_sha256):
        self.db = db
        self.db.executescript(SEQ_SCHEMA)
        self.config_sha256 = config_sha256
        self.source_sha256_json = canonical(source_sha256)
        self.model_sha256_json = canonical(model_sha256)
        self.calibrator_sha256 = calibrator_sha256
        self.db.execute('INSERT INTO seq_run(id, body) VALUES (1, ?)', (canonical(run_info).encode(),))

    @staticmethod
    def cells_status(record):
        """Step 5 per-action support of the attempt WITHOUT probabilities (placeholder heads in this run)."""
        return [{'action': c.get('action'), 'status': c.get('status'), 'reason': c.get('reason'),
                 'prior_pitch_count': c.get('prior_pitch_count')} for c in record['cells']]

    def _insert(self, row, timings):
        missing = [k for k in HASHED_COLUMNS if k not in row]
        if missing or set(row) != set(HASHED_COLUMNS):
            raise ValueError('seq forecast row columns: '+','.join(missing or sorted(set(row)-set(HASHED_COLUMNS))))
        row = dict(row)
        row['row_sha256'] = sha({k: row[k] for k in HASHED_COLUMNS})  # timings are not part of the forecast identity
        for name in TIMING_COLUMNS:
            row[name] = (timings or {}).get(name)
        names = list(row)
        self.db.execute(f"INSERT INTO seq_forecasts({','.join(names)}) VALUES ({','.join('?'*len(names))})",
                        [row[n] for n in names])
        return row['row_sha256']

    def _base(self, record, index, replay_file, context_sha256):
        matchup, snapshot = record.get('matchup') or {}, record.get('snapshot') or {}
        return {'attempt_key': record['attempt_key'], 'game_pk': record['game_pk'],
                'at_bat_number': record['target']['candidate_at_bat_number'],
                'candidate_pitch_number': record['target']['candidate_pitch_number'],
                'written_at_replay_index': index, 'replay_file': replay_file,
                'feed_sha256': snapshot.get('feed_sha256'), 'pitcher': matchup.get('pitcher'),
                'batter': matchup.get('batter'), 'context_sha256': context_sha256,
                'vectors_sha256': record.get('vectors_sha256'), 'step5_outcome': record['outcome'],
                'step5_cells_status_json': canonical(self.cells_status(record)),
                'config_sha256': self.config_sha256, 'source_sha256_json': self.source_sha256_json}

    def append_scored(self, record, index, replay_file, context_sha256, *, x_seq_sha256, tokens, forecast, timings):
        """The forecast row of a first decision. ``tokens``: ``tokens.to_json`` value; ``forecast``: adapter output."""
        row = self._base(record, index, replay_file, context_sha256) | {
            'x_seq_sha256': x_seq_sha256, 'tokens_json': canonical(tokens), 'tokens_sha256': sha(tokens),
            'decision': 'scored', 'reason': None, 'logits_json': canonical(forecast['logits']),
            'c0_ensemble_json': canonical(forecast['c0_ensemble'].tolist()),
            'sel_ensemble_json': canonical(forecast['sel_ensemble'].tolist()),
            'cal_arms_json': canonical(forecast['arms']), 'model_sha256_json': self.model_sha256_json,
            'calibrator_sha256': self.calibrator_sha256}
        return self._insert(row, timings)

    def append_abstain(self, record, index, replay_file, context_sha256, reason, timings=None):
        row = self._base(record, index, replay_file, context_sha256) | {
            'x_seq_sha256': None, 'tokens_json': None, 'tokens_sha256': None, 'decision': 'abstain', 'reason': reason,
            'logits_json': None, 'c0_ensemble_json': None, 'sel_ensemble_json': None, 'cal_arms_json': None,
            'model_sha256_json': self.model_sha256_json, 'calibrator_sha256': self.calibrator_sha256}
        return self._insert(row, timings)

    def sync_terminalized(self, index):
        """Abstain rows for attempts the Step 5 ledger holds without a first decision of their own (a pending
        prefix closed by an observed event or by the replay end), so that seq_forecasts has one row per attempt."""
        rows = self.db.execute('SELECT body FROM pitch_attempts WHERE attempt_key NOT IN '
                               '(SELECT attempt_key FROM seq_forecasts) ORDER BY at_bat_number, last_pitch_number').fetchall()
        for body, in rows:
            record = json.loads(body)
            self.append_abstain(record, index, None, None, f"{record['outcome']}:{record['reason']}")
        return len(rows)
