"""Validated, file-backed C inning result conversion. No evaluator is run here."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


TOLERANCE = 1e-8
MAX_SAFE_INTEGER = 2**53 - 1
ANCHOR_KIND = "immediately_before_logged_pitching_substitution_action"
ASSUMPTIONS = [
    "keep_pitcher_fixed", "prechange_lineup_fixed",
    "frozen_train_repertoire_policy", "no_future_substitutions",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(value: Any, keys: set[str], name: str) -> dict:
    _require(type(value) is dict and set(value) == keys, f"{name} fields")
    return value


def _number(value: Any, name: str, *, probability: bool = False) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{name} must be finite number")
    if probability:
        _require(0 <= value <= 1, f"{name} outside probability unit")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and minimum <= value <= MAX_SAFE_INTEGER, f"{name} must be safe integer >= {minimum}")
    return value


def _sha(value: Any, name: str) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{name} SHA256")


def _utc(value: Any, name: str) -> datetime:
    _require(type(value) is str and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|\+00:00)", value) is not None, f"{name} UTC ISO8601")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} UTC ISO8601") from exc
    _require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, f"{name} UTC")
    return result.astimezone(timezone.utc)


def validate_inning_result(payload: Any) -> dict:
    """Return a valid v1 payload unchanged; raise ValueError for invalid data."""
    p = _fields(payload, {"schema_version", "kind", "status", "reason", "linkage", "scope", "estimate", "coverage", "assumptions", "profiles", "replacement", "provenance"}, "result")
    _require(p["schema_version"] == "inning-result-v1", "schema_version")
    _require(p["kind"] == "conditional_keep", "kind")
    _require(p["status"] in ("bounded", "unavailable"), "status")
    _require(type(p["reason"]) is str and bool(p["reason"].strip()), "reason")
    l = _fields(p["linkage"], {"game_pk", "keep_pitcher_id", "official_game_date", "anchor_kind", "anchor_time_utc", "anchor_action_index", "first_observed_pitch_id"}, "linkage")
    _integer(l["game_pk"], "game_pk", 1)
    _integer(l["keep_pitcher_id"], "keep_pitcher_id", 1)
    _require(type(l["official_game_date"]) is str and re.fullmatch(r"\d{4}-\d\d-\d\d", l["official_game_date"]) is not None, "official_game_date")
    try:
        date.fromisoformat(l["official_game_date"])
    except ValueError as exc:
        raise ValueError("official_game_date") from exc
    _require(l["anchor_kind"] == ANCHOR_KIND, "anchor_kind")
    _utc(l["anchor_time_utc"], "anchor_time_utc")
    _integer(l["anchor_action_index"], "anchor_action_index")
    _require(type(l["first_observed_pitch_id"]) is str and re.fullmatch(r"[1-9]\d*:[1-9]\d*:1", l["first_observed_pitch_id"]) is not None, "first_observed_pitch_id")
    pitch_game = int(l["first_observed_pitch_id"].split(":")[0])
    _require(pitch_game == l["game_pk"], "first_observed_pitch_id game")
    s = _fields(p["scope"], {"horizon", "stopping_boundary", "value_target", "perspective", "initial_defender", "unit", "additive_to_pa"}, "scope")
    for key, expected in {"horizon": "inning_end", "stopping_boundary": "current_half_inning_or_game_end", "value_target": "final_game_win_probability", "perspective": "initial_defense", "unit": "probability", "additive_to_pa": False}.items():
        _require(type(s[key]) is type(expected) and s[key] == expected, f"scope.{key}")
    _require(s["initial_defender"] in ("home", "away"), "initial_defender")
    e = _fields(p["estimate"], {"point", "lower", "upper", "interval_kind"}, "estimate")
    _require(e["point"] is None and e["interval_kind"] == "unresolved_mass_bound", "estimate point or interval kind")
    c = _fields(p["coverage"], {"resolved_mass", "unresolved_mass", "unresolved_reasons", "model_calls"}, "coverage")
    _require(type(c["unresolved_reasons"]) is dict, "unresolved_reasons")
    if p["status"] == "bounded":
        lower = _number(e["lower"], "lower", probability=True)
        upper = _number(e["upper"], "upper", probability=True)
        resolved = _number(c["resolved_mass"], "resolved_mass", probability=True)
        unresolved = _number(c["unresolved_mass"], "unresolved_mass", probability=True)
        _integer(c["model_calls"], "model_calls")
        _require(lower <= upper and lower <= resolved + TOLERANCE, "bound ordering")
        _require(abs(resolved + unresolved - 1) <= TOLERANCE, "coverage mass")
        _require(abs(upper - lower - unresolved) <= TOLERANCE, "unresolved bound width")
        _require(all(type(k) is str and k for k in c["unresolved_reasons"]), "unresolved reason codes")
        total = sum(_number(v, f"unresolved_reasons.{k}", probability=True) for k, v in c["unresolved_reasons"].items())
        _require(abs(total - unresolved) <= TOLERANCE, "unresolved reasons mass")
    else:
        _require(e["lower"] is None and e["upper"] is None, "unavailable estimate must be null")
        _require(c["resolved_mass"] is None and c["unresolved_mass"] is None and c["model_calls"] is None and c["unresolved_reasons"] == {}, "unavailable coverage must be null")
    _require(type(p["assumptions"]) is list and p["assumptions"] == ASSUMPTIONS, "assumptions")
    profiles = _fields(p["profiles"], {"default_batter_ids"}, "profiles")
    ids = profiles["default_batter_ids"]
    _require(type(ids) is list, "default_batter_ids")
    for item in ids:
        _integer(item, "default_batter_id", 1)
    _require(len(ids) == len(set(ids)), "duplicate default batter")
    replacement = _fields(p["replacement"], {"status", "value_pp", "interval_pp", "reason"}, "replacement")
    _require(replacement == {"status": "unavailable", "value_pp": None, "interval_pp": None, "reason": "actual_eligible_substitutes_unverified"}, "replacement")
    prov = _fields(p["provenance"], {"usage", "source_result_sha256", "source_anchor_sha256", "model_bundle_sha256", "evaluation_identity"}, "provenance")
    _require(prov["usage"] == "historical_research", "provenance usage")
    for key in ("source_result_sha256", "source_anchor_sha256", "model_bundle_sha256"):
        _sha(prov[key], key)
    identity = _fields(prov["evaluation_identity"], {"provider_identity", "initial_state_and_count", "lineup_sha256", "evaluation_config_sha256", "policy_id", "horizon", "initial_defender"}, "evaluation_identity")
    for key in ("provider_identity", "lineup_sha256", "evaluation_config_sha256"):
        _sha(identity[key], f"evaluation_identity.{key}")
    _require(identity["policy_id"] == "frozen_train_repertoire_frequency_v1", "evaluation policy")
    _require(identity["horizon"] == s["horizon"] and identity["initial_defender"] == s["initial_defender"], "evaluation identity scope")
    _require(identity["provider_identity"] == prov["model_bundle_sha256"], "evaluation identity bundle")
    state = _fields(identity["initial_state_and_count"], {"date", "inning", "topbot", "outs", "bases", "home_score", "away_score", "balls", "strikes"}, "initial state")
    _require(state["date"] == l["official_game_date"], "evaluation date linkage")
    _integer(state["inning"], "inning", 1)
    for key, maximum in (("outs", 2), ("bases", 7), ("balls", 3), ("strikes", 2)):
        _integer(state[key], key)
        _require(state[key] <= maximum, f"{key} range")
    for key in ("home_score", "away_score"):
        _integer(state[key], key)
    _require(state["topbot"] in ("Top", "Bot"), "initial half inning")
    _require(s["initial_defender"] == ("home" if state["topbot"] == "Top" else "away"), "initial defender and half inning")
    return p


def _read_json(path: str | Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    _require(type(data) is dict, "source JSON object")
    return data, hashlib.sha256(raw).hexdigest()


def convert_inning_result(source_path: str | Path, anchor_path: str | Path) -> dict:
    """Convert the frozen bounded evaluation and its decision anchor."""
    source, source_sha = _read_json(source_path)
    anchor, anchor_sha = _read_json(anchor_path)
    _require(source.get("case_kind") == "conditional_fixed_prechange_lineup_keep_pitcher" and source.get("status") == "bounded", "source case kind or status")
    _require(source.get("decision_anchor") == anchor.get("anchor") == ANCHOR_KIND, "decision anchor kind")
    _require(source.get("source_anchor_sha256") == anchor_sha, "source anchor SHA256 mismatch")
    _sha(source.get("source_roster_sha256"), "source roster SHA256")
    _sha(anchor.get("original_packet_sha256"), "anchor original packet SHA256")
    _require(source.get("source_roster_sha256") == anchor.get("original_packet_sha256"), "source roster provenance mismatch")
    _integer(source.get("game_pk"), "source game", 1)
    _integer(anchor.get("game_pk"), "anchor game", 1)
    _integer(source.get("pitcher_id"), "source pitcher", 1)
    _integer(anchor.get("keep_pitcher_id"), "anchor pitcher", 1)
    _require(source.get("game_pk") == anchor.get("game_pk") and source.get("pitcher_id") == anchor.get("keep_pitcher_id"), "game or pitcher mismatch")
    identity = source.get("evaluation_identity")
    _require(type(identity) is dict, "evaluation_identity")
    _require(source.get("horizon") == identity.get("horizon") == "inning_end", "evaluation horizon")
    _require(source.get("initial_defender") == identity.get("initial_defender") and source.get("initial_defender") in ("home", "away"), "evaluation defender")
    _require(source.get("model_bundle_manifest_sha256") == identity.get("provider_identity"), "model bundle identity")
    _sha(source.get("model_bundle_manifest_sha256"), "model bundle")
    state = identity.get("initial_state_and_count")
    expected = anchor.get("state_as_of_anchor")
    _require(type(state) is dict and type(expected) is dict, "initial state")
    _require(state.get("topbot") in ("Top", "Bot") and source["initial_defender"] == ("home" if state["topbot"] == "Top" else "away"), "initial defender and half inning")
    for source_key, anchor_key in (("inning", "inning"), ("topbot", "half"), ("outs", "outs"), ("bases", "bases"), ("home_score", "home_score"), ("away_score", "away_score"), ("balls", "balls"), ("strikes", "strikes")):
        _require(state.get(source_key) == expected.get(anchor_key) and type(state.get(source_key)) is type(expected.get(anchor_key)), f"initial state {source_key}")
    _require(type(state.get("date")) is str, "official game date")
    _require(anchor.get("cutoff_index_verified") is True and anchor.get("cutoff_timestamp_verified") is True, "anchor cutoff verification")
    _integer(anchor.get("anchor_action_index"), "anchor action index")
    _require(anchor.get("lineup_missing_reasons") == [], "lineup missing reasons")
    anchor_time = _utc(anchor.get("anchor_action_start_time_utc"), "anchor time")
    source_end = _utc(anchor.get("source_play_end_time_max"), "source play end")
    _require(source_end < anchor_time, "source play after anchor")
    max_play_index = _integer(anchor.get("source_play_index_max"), "source play index")
    lineup = anchor.get("lineup_as_of")
    _require(type(lineup) is dict and type(lineup.get("ordered_slots")) is list and len(lineup["ordered_slots"]) == 9, "anchor lineup")
    _require(source.get("lineup_slots_provided") == 9, "source lineup slots")
    slots = lineup["ordered_slots"]
    _require(all(type(x) is dict for x in slots), "lineup slot objects")
    _require(all(type(x.get("slot")) is int for x in slots) and [x.get("slot") for x in slots] == list(range(1, 10)), "lineup slots")
    start = _integer(lineup.get("next_slot_one_based"), "next slot", 1) - 1
    _require(start < 9, "next slot")
    rotated = slots[start:] + slots[:start]
    ids = [x.get("batter_id") for x in rotated]
    for batter_id in ids:
        _integer(batter_id, "lineup batter", 1)
    _require(len(set(ids)) == 9 and ids == source.get("lineup_batter_ids"), "prechange lineup mismatch")
    _require(lineup.get("next_batter_id") == ids[0], "next batter mismatch")
    _require(type(lineup.get("next_batter_id")) is int, "next batter id")
    _require(lineup.get("next_batter_stand_vs_keep") == rotated[0].get("stand_vs_keep"), "next batter stance mismatch")
    for slot in slots:
        _require(slot.get("stand_vs_keep") in ("L", "R"), "lineup stance")
        _require(_integer(slot.get("source_at_bat_index"), "source at bat index") <= max_play_index, "lineup source index after cutoff")
        evidence_list = slot.get("stand_evidence")
        _require(type(evidence_list) is list and bool(evidence_list), "stance evidence missing")
        for evidence in evidence_list:
            _require(type(evidence) is dict, "stance evidence object")
            _require(_integer(evidence.get("at_bat_index"), "stance evidence index") <= max_play_index, "stance evidence index after cutoff")
            _require(evidence.get("pitcher_id") == anchor.get("keep_pitcher_id") and type(evidence.get("pitcher_id")) is int, "stance evidence pitcher mismatch")
            _require(evidence.get("observed_stand") == slot["stand_vs_keep"], "stance evidence stand mismatch")
            _require(_utc(evidence.get("play_end_time"), "stance evidence") <= source_end, "stance evidence after source cutoff")
    sources = source.get("lineup_profile_sources")
    _require(type(sources) is list and len(sources) == 9 and all(type(x) is str and x in ("frozen_default_zero_reliability", "frozen_batter_snapshot") for x in sources), "profile sources")
    defaults = [batter_id for batter_id, profile_source in zip(ids, sources) if profile_source == "frozen_default_zero_reliability"]
    _require(defaults == source.get("default_profile_batter_ids"), "default profiles mismatch")
    actual = source.get("actual_replacement")
    _require(type(actual) is dict and actual.get("status") == "unavailable" and actual.get("horizon") == "inning_end" and actual.get("value_pp") is None and actual.get("reason") == "actual_eligible_substitutes_unverified", "actual replacement")
    _require(source.get("value") is None and type(source.get("value_interval")) is list and len(source["value_interval"]) == 2, "source interval")
    _require(type(anchor.get("decision_first_observed_pitch_id")) is str, "first observed pitch id")
    result = {
        "schema_version": "inning-result-v1", "kind": "conditional_keep", "status": "bounded", "reason": source.get("reason"),
        "linkage": {"game_pk": anchor["game_pk"], "keep_pitcher_id": anchor["keep_pitcher_id"], "official_game_date": state["date"], "anchor_kind": ANCHOR_KIND, "anchor_time_utc": anchor["anchor_action_start_time_utc"], "anchor_action_index": anchor["anchor_action_index"], "first_observed_pitch_id": anchor["decision_first_observed_pitch_id"]},
        "scope": {"horizon": "inning_end", "stopping_boundary": "current_half_inning_or_game_end", "value_target": "final_game_win_probability", "perspective": "initial_defense", "initial_defender": source["initial_defender"], "unit": "probability", "additive_to_pa": False},
        "estimate": {"point": None, "lower": source["value_interval"][0], "upper": source["value_interval"][1], "interval_kind": "unresolved_mass_bound"},
        "coverage": {"resolved_mass": source.get("absorbed_mass"), "unresolved_mass": source.get("unresolved_mass"), "unresolved_reasons": source.get("unresolved_reasons"), "model_calls": source.get("model_calls")},
        "assumptions": ASSUMPTIONS.copy(), "profiles": {"default_batter_ids": defaults},
        "replacement": {"status": "unavailable", "value_pp": None, "interval_pp": None, "reason": "actual_eligible_substitutes_unverified"},
        "provenance": {"usage": "historical_research", "source_result_sha256": source_sha, "source_anchor_sha256": anchor_sha, "model_bundle_sha256": source["model_bundle_manifest_sha256"], "evaluation_identity": identity},
    }
    return validate_inning_result(result)


def development_unavailable(bounded: dict) -> dict:
    """Derive a clearly synthetic missing-result example from validated linkage."""
    validate_inning_result(bounded)
    _require(bounded["status"] == "bounded", "development source must be bounded")
    result = json.loads(json.dumps(bounded))
    result["status"] = "unavailable"
    result["reason"] = "missing_evaluation_result"
    result["estimate"]["lower"] = result["estimate"]["upper"] = None
    result["coverage"] = {"resolved_mass": None, "unresolved_mass": None, "unresolved_reasons": {}, "model_calls": None}
    return validate_inning_result(result)
