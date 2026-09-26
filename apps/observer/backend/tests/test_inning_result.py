import copy
from pathlib import Path

import pytest

from observer_app.inning_result import convert_inning_result, development_unavailable, validate_inning_result


ROOT = Path(__file__).parents[4]
SOURCE = ROOT / "results/EXP-C-INNING-001/conditional_keep_777063.json"
ANCHOR = Path("/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/decision_anchor_777063.json")


@pytest.fixture(scope="module")
def bounded():
    return convert_inning_result(SOURCE, ANCHOR)


def test_actual_conversion_preserves_identity_date_and_mass(bounded):
    assert bounded["linkage"]["official_game_date"] == "2025-07-21"
    assert bounded["linkage"]["anchor_time_utc"].startswith("2025-07-22")
    assert bounded["profiles"]["default_batter_ids"] == [691785, 701350]
    assert bounded["estimate"]["lower"] == pytest.approx(.5210473531762911)
    assert bounded["coverage"]["unresolved_mass"] == pytest.approx(.013631509811148742)
    assert validate_inning_result(bounded) is bounded


@pytest.mark.parametrize("path,value", [
    (("estimate", "lower"), -0.1),
    (("estimate", "upper"), float("nan")),
    (("estimate", "lower"), True),
    (("estimate", "upper"), .7),
    (("coverage", "resolved_mass"), .8),
    (("coverage", "unresolved_reasons"), {"minimum_state_mass": .1}),
    (("scope", "unit"), "percentage_points"),
    (("scope", "additive_to_pa"), True),
    (("provenance", "model_bundle_sha256"), "0" * 64),
    (("linkage", "anchor_time_utc"), "2025-07-22T00:20:56.827+09:00"),
    (("linkage", "first_observed_pitch_id"), "777063:49:2"),
    (("linkage", "game_pk"), 2**53),
    (("linkage", "official_game_date"), "2025-07-22"),
    (("scope", "initial_defender"), "away"),
])
def test_invalid_numeric_units_identity_and_time_rejected(bounded, path, value):
    payload = copy.deepcopy(bounded)
    payload[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_inning_result(payload)


@pytest.mark.parametrize("field,value", [
    ("policy_id", "some_other_policy"),
    ("lineup_sha256", "bad"),
    ("evaluation_config_sha256", "bad"),
    ("provider_identity", "0" * 64),
])
def test_evaluation_identity_rejected(bounded, field, value):
    payload = copy.deepcopy(bounded)
    payload["provenance"]["evaluation_identity"][field] = value
    with pytest.raises(ValueError):
        validate_inning_result(payload)


@pytest.mark.parametrize("field,value", [
    ("inning", 0), ("topbot", "Bottom"), ("outs", 3), ("bases", 8),
    ("home_score", -1), ("balls", 4), ("strikes", True),
])
def test_evaluation_state_rejected(bounded, field, value):
    payload = copy.deepcopy(bounded)
    payload["provenance"]["evaluation_identity"]["initial_state_and_count"][field] = value
    with pytest.raises(ValueError):
        validate_inning_result(payload)


def test_utc_plus_zero_accepted(bounded):
    payload = copy.deepcopy(bounded)
    payload["linkage"]["anchor_time_utc"] = "2025-07-22T00:20:56.827+00:00"
    assert validate_inning_result(payload) is payload


@pytest.mark.parametrize("field", ["game_pk", "pitcher_id", "model_bundle_manifest_sha256", "initial_defender", "value_interval"])
def test_missing_source_fields_raise_value_error(tmp_path, field):
    import json
    source = json.loads(SOURCE.read_text())
    source.pop(field)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        convert_inning_result(source_path, ANCHOR)


def test_unavailable_has_no_numeric_result_and_retains_linkage(bounded):
    unavailable = development_unavailable(bounded)
    assert unavailable["reason"] == "missing_evaluation_result"
    assert unavailable["linkage"] == bounded["linkage"]
    assert unavailable["provenance"] == bounded["provenance"]
    assert unavailable["estimate"] == {"point": None, "lower": None, "upper": None, "interval_kind": "unresolved_mass_bound"}
    assert unavailable["coverage"] == {"resolved_mass": None, "unresolved_mass": None, "unresolved_reasons": {}, "model_calls": None}
    unavailable["coverage"]["model_calls"] = 0
    with pytest.raises(ValueError):
        validate_inning_result(unavailable)


@pytest.mark.parametrize("change", ["anchor_hash", "game", "state", "lineup", "profile", "profile_type", "cutoff", "late_evidence", "evidence_pitcher", "evidence_stance", "evidence_index", "source_index", "next_stance", "missing_lineup", "bundle", "horizon"])
def test_file_conversion_rejects_broken_crosslinks(tmp_path, change):
    import json
    source = json.loads(SOURCE.read_text())
    anchor = json.loads(ANCHOR.read_text())
    if change == "anchor_hash":
        source["source_anchor_sha256"] = "0" * 64
    elif change == "game":
        source["game_pk"] += 1
    elif change == "state":
        source["evaluation_identity"]["initial_state_and_count"]["outs"] = 2
    elif change == "lineup":
        source["lineup_batter_ids"][0] = 123
    elif change == "profile":
        source["default_profile_batter_ids"] = []
    elif change == "profile_type":
        source["lineup_profile_sources"][0] = []
    elif change == "cutoff":
        anchor["cutoff_timestamp_verified"] = False
    elif change == "late_evidence":
        anchor["lineup_as_of"]["ordered_slots"][0]["stand_evidence"][0]["play_end_time"] = "2025-07-22T01:00:00Z"
    elif change == "evidence_pitcher":
        anchor["lineup_as_of"]["ordered_slots"][0]["stand_evidence"][0]["pitcher_id"] = 123
    elif change == "evidence_stance":
        anchor["lineup_as_of"]["ordered_slots"][0]["stand_evidence"][0]["observed_stand"] = "R"
    elif change == "evidence_index":
        anchor["lineup_as_of"]["ordered_slots"][0]["stand_evidence"][0]["at_bat_index"] = 100
    elif change == "source_index":
        anchor["lineup_as_of"]["ordered_slots"][0]["source_at_bat_index"] = 100
    elif change == "next_stance":
        anchor["lineup_as_of"]["next_batter_stand_vs_keep"] = "R"
    elif change == "missing_lineup":
        anchor["lineup_missing_reasons"] = ["missing"]
    elif change == "bundle":
        source["evaluation_identity"]["provider_identity"] = "0" * 64
    else:
        source["evaluation_identity"]["horizon"] = "game_end"
    source_path = tmp_path / "source.json"
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(anchor))
    if change != "anchor_hash":
        import hashlib
        source["source_anchor_sha256"] = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        convert_inning_result(source_path, anchor_path)
