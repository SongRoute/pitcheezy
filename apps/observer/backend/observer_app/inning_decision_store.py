"""Immutable, file-imported historical inning decisions."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from typing import Any

from .inning_result import ANCHOR_KIND, MAX_SAFE_INTEGER, convert_inning_result, validate_inning_result


SCHEMA = "inning-decision-v1"
MODE = "historical_decision_review"
PHASE = "before_pitching_change"
REVISION = 1
_ID_PATTERN = re.compile(r"inning-decision-[0-9a-f]{64}\Z")
_UTC_PATTERN = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|\+00:00)\Z")
_PITCH_PATTERN = re.compile(r"[1-9]\d*:[1-9]\d*:1\Z")
_LINKAGE_KEYS = {"game_pk", "keep_pitcher_id", "official_game_date", "anchor_kind", "anchor_time_utc", "anchor_action_index", "first_observed_pitch_id"}
_STATE_KEYS = {"date", "inning", "topbot", "outs", "bases", "home_score", "away_score", "balls", "strikes"}


class DecisionNotFound(Exception):
    pass


class DecisionConflict(Exception):
    pass


class DecisionCorrupt(Exception):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and minimum <= value <= MAX_SAFE_INTEGER, f"{name} must be a safe integer >= {minimum}")
    return value


def _utc(value: Any) -> str:
    _require(type(value) is str and _UTC_PATTERN.fullmatch(value) is not None, "anchor_time_utc must be UTC ISO8601")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("anchor_time_utc must be UTC ISO8601") from exc
    return instant.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def validate_context(value: Any) -> dict:
    """Return a normalized, strict decision context."""
    _require(type(value) is dict and set(value) == {"phase", "linkage", "initial_state"}, "context fields")
    phase = value["phase"]
    _require(type(phase) is str and bool(phase.strip()), "phase must be a nonempty string")
    linkage = value["linkage"]
    _require(type(linkage) is dict and set(linkage) == _LINKAGE_KEYS, "linkage fields")
    game = _integer(linkage["game_pk"], "game_pk", 1)
    _integer(linkage["keep_pitcher_id"], "keep_pitcher_id", 1)
    game_date = linkage["official_game_date"]
    _require(type(game_date) is str and re.fullmatch(r"\d{4}-\d\d-\d\d", game_date) is not None, "official_game_date")
    try:
        date.fromisoformat(game_date)
    except ValueError as exc:
        raise ValueError("official_game_date") from exc
    _require(linkage["anchor_kind"] == ANCHOR_KIND, "anchor_kind")
    anchor_time = _utc(linkage["anchor_time_utc"])
    _integer(linkage["anchor_action_index"], "anchor_action_index")
    pitch = linkage["first_observed_pitch_id"]
    _require(type(pitch) is str and _PITCH_PATTERN.fullmatch(pitch) is not None, "first_observed_pitch_id")
    _require(int(pitch.split(":", 1)[0]) == game, "first_observed_pitch_id game")
    state = value["initial_state"]
    _require(type(state) is dict and set(state) == _STATE_KEYS, "initial_state fields")
    _require(type(state["date"]) is str and state["date"] == game_date, "initial_state date")
    _integer(state["inning"], "inning", 1)
    for key, maximum in (("outs", 2), ("bases", 7), ("balls", 3), ("strikes", 2)):
        _require(_integer(state[key], key) <= maximum, f"{key} range")
    for key in ("home_score", "away_score"):
        _integer(state[key], key)
    _require(type(state["topbot"]) is str and state["topbot"] in ("Top", "Bot"), "topbot")
    return {"phase": phase, "linkage": {**linkage, "anchor_time_utc": anchor_time}, "initial_state": dict(state)}


def _decision_id(linkage: dict) -> str:
    identity = {key: linkage[key] for key in ("game_pk", "anchor_kind", "anchor_time_utc", "anchor_action_index", "first_observed_pitch_id")}
    return "inning-decision-" + hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def _read_row(row: Any) -> tuple[dict, dict]:
    try:
        raw_context = json.loads(row["context_json"])
        raw_result = json.loads(row["result_json"])
        context = validate_context(raw_context)
        result = validate_inning_result(raw_result)
        if context["phase"] != PHASE:
            raise ValueError("stored phase")
        if _canonical(context) != row["context_json"] or _canonical(result) != row["result_json"]:
            raise ValueError("noncanonical row")
        normalized_result_linkage = {**result["linkage"], "anchor_time_utc": _utc(result["linkage"]["anchor_time_utc"])}
        if context["linkage"] != normalized_result_linkage or context["initial_state"] != result["provenance"]["evaluation_identity"]["initial_state_and_count"]:
            raise ValueError("context/result mismatch")
        if result["status"] != "bounded":
            raise ValueError("non-source result")
        if row["decision_id"] != _decision_id(context["linkage"]):
            raise ValueError("decision ID mismatch")
        if type(row["game_id"]) is not int or row["game_id"] != context["linkage"]["game_pk"]:
            raise ValueError("game ID mismatch")
        if type(row["revision"]) is not int or row["revision"] != REVISION:
            raise ValueError("revision mismatch")
        if row["result_sha256"] != hashlib.sha256(row["result_json"].encode("utf-8")).hexdigest():
            raise ValueError("result SHA mismatch")
        _utc(row["created_at"])
        return context, result
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise DecisionCorrupt("stored inning decision is corrupt") from exc


class InningDecisionRepository:
    def __init__(self, store):
        self.store = store
        with store.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS inning_decisions (
                decision_id TEXT PRIMARY KEY, game_id INTEGER NOT NULL, revision INTEGER NOT NULL,
                context_json TEXT NOT NULL, result_json TEXT NOT NULL, result_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL)""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS inning_decisions_no_update
                BEFORE UPDATE ON inning_decisions BEGIN SELECT RAISE(ABORT, 'immutable inning decision'); END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS inning_decisions_no_delete
                BEFORE DELETE ON inning_decisions BEGIN SELECT RAISE(ABORT, 'immutable inning decision'); END""")
            db.execute("CREATE INDEX IF NOT EXISTS inning_decisions_game ON inning_decisions(game_id, decision_id)")

    def import_result(self, source_path, anchor_path) -> str:
        result = validate_inning_result(convert_inning_result(source_path, anchor_path))
        if result["status"] != "bounded":
            raise ValueError("only bounded source evaluations may be imported")
        result = json.loads(_canonical(result))
        context = validate_context({"phase": PHASE, "linkage": result["linkage"], "initial_state": result["provenance"]["evaluation_identity"]["initial_state_and_count"]})
        decision_id = _decision_id(context["linkage"])
        context_json = _canonical(context)
        result_json = _canonical(result)
        result_sha = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        with self.store.transaction() as db:
            old = db.execute("SELECT * FROM inning_decisions WHERE decision_id=?", (decision_id,)).fetchone()
            if old is not None:
                _read_row(old)
                if old["context_json"] != context_json or old["result_json"] != result_json:
                    raise DecisionConflict("decision already registered with different context or result")
                return decision_id
            db.execute("""INSERT INTO inning_decisions
                (decision_id, game_id, revision, context_json, result_json, result_sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", (decision_id, context["linkage"]["game_pk"], REVISION, context_json, result_json, result_sha, created_at))
        return decision_id

    def list_decisions(self, game_id) -> dict:
        _integer(game_id, "game_id", 1)
        with self.store.transaction(write=False) as db:
            rows = db.execute("SELECT * FROM inning_decisions WHERE game_id=? ORDER BY decision_id", (game_id,)).fetchall()
            decisions = []
            for row in rows:
                context, _ = _read_row(row)
                decisions.append({"decision_id": row["decision_id"], "revision": REVISION, "context": context})
        return {"schema_version": SCHEMA, "mode": MODE, "decisions": decisions}

    def resolve(self, decision_id, revision, context) -> dict:
        _require(type(decision_id) is str and _ID_PATTERN.fullmatch(decision_id) is not None, "decision_id")
        _integer(revision, "revision", 1)
        expected = validate_context(context)
        with self.store.transaction(write=False) as db:
            row = db.execute("SELECT * FROM inning_decisions WHERE decision_id=?", (decision_id,)).fetchone()
            if row is None:
                raise DecisionNotFound("inning decision not found")
            stored_context, result = _read_row(row)
            if revision != REVISION or expected != stored_context:
                raise DecisionConflict("inning decision context or revision mismatch")
        return {"schema_version": SCHEMA, "mode": MODE, "decision_id": decision_id,
                "revision": REVISION, "context": stored_context, "result": result}
