"""Append-only SQLite forecast ledger (one file per game).

Design follows the accepted capture-native forecast persistence pattern
(immutable intent first, forecast second, first write wins):

* every table rejects UPDATE and DELETE with a trigger;
* an *intent* row is written before the forecaster is called, keyed by
  (game_pk, slot_key, target key, model_id) -> a restart can never write a second forecast
  for the same slot/model, it only finds the earlier intent;
* a restart opens a new *session* row; nothing from an earlier session is
  rewritten;
* each forecast records wall-clock UTC (ns) and the host monotonic clock (ns),
  plus the context as-of (last accepted pitch uid and its receipt time) and a
  provisional timing class.  The authoritative PRE_RECEIPT / POST_HOC decision is
  re-derived at the single analysis look from the store's first-receipt time.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

SCHEMA_VERSION = "rs01-ledger-v1"

_TABLES = ("ledger_meta", "sessions", "intents", "forecasts", "receipts", "events")

_DDL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  started_wall_ns INTEGER NOT NULL,
  started_mono_ns INTEGER NOT NULL,
  pid INTEGER NOT NULL,
  host TEXT NOT NULL,
  code_commit TEXT NOT NULL,
  registration_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS intents (
  intent_id TEXT PRIMARY KEY,
  game_pk INTEGER NOT NULL,
  slot_key TEXT NOT NULL,
  slot_ordinal INTEGER NOT NULL,
  target_at_bat_number INTEGER NOT NULL,
  target_pitch_number INTEGER NOT NULL,
  target_pitcher_id INTEGER,
  target_batter_id INTEGER,
  mid_pa_substitution INTEGER NOT NULL DEFAULT 0,
  model_id TEXT NOT NULL,
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  context_asof_pitch_uid TEXT,
  context_asof_receipt_wall_ns INTEGER,
  context_sha256 TEXT NOT NULL,
  context_source_seq INTEGER,
  context_source_fetched_ns INTEGER,
  created_wall_ns INTEGER NOT NULL,
  created_mono_ns INTEGER NOT NULL,
  UNIQUE (game_pk, slot_key, target_at_bat_number, target_pitch_number, model_id)
);
CREATE TABLE IF NOT EXISTS forecasts (
  intent_id TEXT PRIMARY KEY REFERENCES intents(intent_id),
  status TEXT NOT NULL CHECK (status IN ('OK', 'ABSTAIN', 'ERROR')),
  abstain_reason TEXT,
  leaf_order TEXT NOT NULL,
  probs_json TEXT,  -- {candidate_pitch_type or '*': [p_leaf...]}
  generated_wall_ns INTEGER NOT NULL,
  generated_mono_ns INTEGER NOT NULL,
  compute_ns INTEGER NOT NULL,
  provisional_timing TEXT NOT NULL
    CHECK (provisional_timing IN ('PRE_RECEIPT', 'POST_HOC_REPLAY'))
);
CREATE TABLE IF NOT EXISTS receipts (
  game_pk INTEGER NOT NULL,
  pitch_uid TEXT NOT NULL,
  store_first_receipt_wall_ns INTEGER,
  observed_wall_ns INTEGER NOT NULL,
  observed_mono_ns INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  PRIMARY KEY (game_pk, pitch_uid)
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  wall_ns INTEGER NOT NULL,
  kind TEXT NOT NULL,
  detail_json TEXT NOT NULL
);
"""


def _immutability_triggers() -> str:
    parts = []
    for t in _TABLES:
        for op in ("UPDATE", "DELETE"):
            parts.append(
                f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
                f"BEGIN SELECT RAISE(ABORT, 'rs01 ledger is append-only: {op} on {t}'); END;"
            )
    return "\n".join(parts)


class LedgerImmutableError(RuntimeError):
    pass


def intent_id_for(game_pk: int, slot_key: str, model_id: str, target: tuple = ()) -> str:
    raw = f"{int(game_pk)}|{slot_key}|{'/'.join(map(str, target))}|{model_id}".encode()
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


@dataclass(frozen=True)
class Clock:
    """Injectable clock so tests can simulate timing; production uses the host."""

    def wall_ns(self) -> int:
        return time.time_ns()

    def mono_ns(self) -> int:
        return time.monotonic_ns()


class Ledger:
    def __init__(self, path: Path, *, game_pk: int, registration_sha256: str,
                 code_commit: str, meta: Optional[dict] = None,
                 clock: Optional[Clock] = None):
        self.path = Path(path)
        self.game_pk = int(game_pk)
        self.clock = clock or Clock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path), isolation_level=None)
        mode = self.con.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if mode.lower() != "delete":  # single-file rollback journal, as in the accepted design
            raise LedgerImmutableError(f"ledger journal mode {mode!r} != delete")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(_DDL + _immutability_triggers())
        base_meta = {"schema_version": SCHEMA_VERSION, "game_pk": str(self.game_pk),
                     "registration_sha256": registration_sha256}
        base_meta.update({k: (v if isinstance(v, str) else json.dumps(v, sort_keys=True))
                          for k, v in (meta or {}).items()})
        self._pin_meta(base_meta)
        self.session_id = uuid.uuid4().hex
        self.con.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
            (self.session_id, self.clock.wall_ns(), self.clock.mono_ns(), os.getpid(),
             socket.gethostname(), code_commit, registration_sha256))

    # -- meta: first write wins; a restart with different frozen inputs is refused
    def _pin_meta(self, meta: dict) -> None:
        for k, v in meta.items():
            row = self.con.execute("SELECT value FROM ledger_meta WHERE key=?", (k,)).fetchone()
            if row is None:
                self.con.execute("INSERT INTO ledger_meta VALUES (?,?)", (k, v))
            elif row[0] != v:
                raise LedgerImmutableError(
                    f"ledger meta {k!r} is frozen as {row[0]!r}; restart supplied {v!r}")

    def meta(self) -> dict:
        return dict(self.con.execute("SELECT key, value FROM ledger_meta"))

    def event(self, kind: str, **detail) -> None:
        self.con.execute("INSERT INTO events (session_id, wall_ns, kind, detail_json) VALUES (?,?,?,?)",
                         (self.session_id, self.clock.wall_ns(), kind,
                          json.dumps(detail, sort_keys=True, default=str)))

    def has_intent(self, slot_key: str, model_id: str, target: tuple) -> bool:
        iid = intent_id_for(self.game_pk, slot_key, model_id, target)
        return self.con.execute("SELECT 1 FROM intents WHERE intent_id=?", (iid,)).fetchone() is not None

    def write_intent(self, *, slot_key: str, slot_ordinal: int, target_at_bat_number: int,
                     target_pitch_number: int, model_id: str,
                     target_pitcher_id: Optional[int] = None, target_batter_id: Optional[int] = None,
                     mid_pa_substitution: int = 0,
                     context_asof_pitch_uid: Optional[str],
                     context_asof_receipt_wall_ns: Optional[int],
                     context_sha256: str, context_source_seq: Optional[int] = None,
                     context_source_fetched_ns: Optional[int] = None) -> Optional[str]:
        """Insert the intent; returns intent_id, or None if one already exists."""
        iid = intent_id_for(self.game_pk, slot_key, model_id,
                            (int(target_at_bat_number), int(target_pitch_number)))
        cur = self.con.execute(
            "INSERT OR IGNORE INTO intents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (iid, self.game_pk, slot_key, int(slot_ordinal), int(target_at_bat_number),
             int(target_pitch_number), target_pitcher_id, target_batter_id,
             int(mid_pa_substitution), model_id, self.session_id,
             context_asof_pitch_uid, context_asof_receipt_wall_ns, context_sha256,
             context_source_seq, context_source_fetched_ns,
             self.clock.wall_ns(), self.clock.mono_ns()))
        return iid if cur.rowcount == 1 else None

    def write_forecast(self, intent_id: str, *, status: str, leaf_order: Sequence[str],
                       probs: Optional[dict], abstain_reason: Optional[str],
                       generated_wall_ns: int, generated_mono_ns: int, compute_ns: int,
                       provisional_timing: str) -> bool:
        cur = self.con.execute(
            "INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?,?,?,?,?,?)",
            (intent_id, status, abstain_reason, json.dumps(list(leaf_order)),
             None if probs is None else json.dumps(
                 {str(k): [float(x) for x in v] for k, v in probs.items()}, sort_keys=True),
             int(generated_wall_ns), int(generated_mono_ns), int(compute_ns),
             provisional_timing))
        return cur.rowcount == 1

    def record_receipt(self, pitch_uid: str, store_first_receipt_wall_ns: Optional[int]) -> None:
        self.con.execute(
            "INSERT OR IGNORE INTO receipts VALUES (?,?,?,?,?,?)",
            (self.game_pk, pitch_uid, store_first_receipt_wall_ns, self.clock.wall_ns(),
             self.clock.mono_ns(), self.session_id))

    def close(self) -> None:
        self.con.close()


def read_rows(path: Path, table: str) -> list[dict]:
    """Read-only access for metadata checks and the sealed analysis."""
    if table not in _TABLES:
        raise ValueError(table)
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table}")]
    finally:
        con.close()


def iter_ledgers(root: Path) -> Iterable[Path]:
    return sorted(Path(root).glob("game_*/ledger*.sqlite"))
