"""Audit C-INNING-001 decision input admissibility; never run model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROSTER_PATH = Path("/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/replacement_packets.json")
LINEUP_PATH = Path("/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/lineup_supplement_v2.json")
ROSTER_SHA256 = "a327f1fecd7c9b97e4e2575328497520e373f9bcc325b28b341ae17303360371"
LINEUP_SHA256 = "70ec1a13869b7a73f6a2dc6eb97fc5b813c2a03e0300f3b67205018ad6cd4605"
SCREEN_PATH = ROOT / "results/EXP-C-INNING-001/roster_packet_screen.json"
SCREEN_SHA256 = "e33ba90e5b10c2641d9dc41f85ff4b7c17590fcd0a2f12feab204608597b2ecb"
OUTPUT = ROOT / "results/EXP-C-INNING-001/source_admissibility_audit.json"


def checked_json(path: Path, expected_sha: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"source hash changed: {path}")
    return json.loads(raw)


def main() -> None:
    roster = checked_json(ROSTER_PATH, ROSTER_SHA256)
    lineup = checked_json(LINEUP_PATH, LINEUP_SHA256)
    screen = checked_json(SCREEN_PATH, SCREEN_SHA256)
    by_game = [{int(row["game_pk"]): row for row in source} for source in
               (roster["decisions"], lineup["lineups"], screen["decisions"])]
    games = set(by_game[0])
    if len(games) != 6 or any(set(table) != games for table in by_game):
        raise ValueError("six fixed decision games do not align")
    decisions = []
    for game in sorted(games):
        r, l, s = (table[game] for table in by_game)
        if len({r["decision_first_observed_pitch_id"], l["decision_first_observed_pitch_id"],
                s["decision_first_observed_pitch_id"]}) != 1:
            raise ValueError("decision pitch keys do not align")
        supported = s["keep_evaluation_status"] in ("bounded", "complete")
        lineup_known = l["lineup_as_of"] is not None
        eligibility_known = r["actual_eligible_replacement_candidates"] is not None
        if game == 777063:
            if (l["missing_reason"] != "current_pa_pre_first_pitch_nonpitching_substitution" or
                    "offensive_substitution" not in l["current_pa_pre_first_pitch_action_types"]):
                raise ValueError("777063 pinch-hit timing evidence changed")
            reason = "invalid_for_replacement_current_batter_after_pitcher_decision"
        elif not supported:
            reason = "unsupported_keep_pitcher"
        else:
            reason = "unverified_predecision_lineup_or_availability"
        decisions.append({"game_pk": game, "decision_first_observed_pitch_id": r["decision_first_observed_pitch_id"],
                          "keep_pitcher_id": r["keep_pitcher_id"], "keep_model_supported": supported,
                          "predecision_order_verified": lineup_known,
                          "current_batter_at_pitcher_decision_verified": lineup_known,
                          "substitute_matchup_stances_verified": False,
                          "actual_eligible_substitute_verified": eligibility_known,
                          "usable_for_actual_replacement": False,
                          "replacement_value_pp": None, "reason": reason,
                          "lineup_missing_reason": l.get("missing_reason")})
    supported_games = [d["game_pk"] for d in decisions if d["keep_model_supported"]]
    lineup_games = [d["game_pk"] for d in decisions if d["predecision_order_verified"]]
    joint_games = sorted(set(supported_games) & set(lineup_games))
    if len(supported_games) != 1 or len(lineup_games) != 5 or joint_games:
        raise ValueError("admissibility counts differ from prespecified audit claim")
    payload = {"schema_version": 1, "audit_id": "C-INNING-001-source-admissibility-v1",
               "scope": "six fixed 2025 DEV observed pitching changes",
               "source_sha256": {"roster_packet": ROSTER_SHA256,
                                 "lineup_supplement_v2": LINEUP_SHA256,
                                 "saved_numeric_screen": SCREEN_SHA256},
               "counts": {"decision_games": 6, "keep_model_supported": 1,
                          "predecision_order_verified": 5,
                          "both_supported_keep_and_verified_order": 0,
                          "actually_eligible_substitute_verified": 0,
                          "usable_actual_replacement": 0},
               "supported_keep_games": supported_games, "verified_order_games": lineup_games,
               "joint_games": joint_games, "decisions": decisions,
               "prior_numeric_outputs_unchanged": True,
               "interpretation": "The 777063 keep one-PA screen conditions on a batter observed after the pitcher decision; it is invalid as an actual replacement-decision input. The independent two-out demonstration is unaffected."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
