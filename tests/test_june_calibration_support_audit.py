import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("june_audit", ROOT / "scripts/audit_june_calibration_support.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
CONFIG = json.loads((ROOT / "configs/ML-JUNE-SUPPORT-AUDIT-v1.json").read_text())
RULE = CONFIG["support_rule"]


def test_registered_config_is_valid_and_june_2025_only():
    audit.validate_config(CONFIG)
    assert audit.validate_window(CONFIG["window"])[1].isoformat() == "2025-06-30"


@pytest.mark.parametrize("window", [
    {"season": 2026, "date_min": "2026-06-01", "date_max": "2026-06-30", "reject_year_min": 2026},
    {"season": 2025, "date_min": "2025-06-01", "date_max": "2026-01-01", "reject_year_min": 2026},
])
def test_window_reaching_2026_is_rejected(window):
    with pytest.raises(audit.AuditBoundaryError):
        audit.validate_window(window)


def test_rows_from_2026_reaching_filter_are_rejected():
    frame = pd.DataFrame({"game_date": ["2025-06-02", "2026-06-02"], "game_type": ["R", "R"]})
    with pytest.raises(audit.AuditBoundaryError):
        audit.filter_window(frame, CONFIG["window"])


def test_filter_keeps_june_regular_season_only():
    frame = pd.DataFrame({"game_date": ["2025-05-31", "2025-06-01", "2025-06-30", "2025-07-01", "2025-06-10"],
                          "game_type": ["R", "R", "R", "R", "S"]})
    kept = audit.filter_window(frame, CONFIG["window"])
    assert kept["game_date"].tolist() == ["2025-06-01", "2025-06-30"]


@pytest.mark.parametrize("column", ["events", "description", "plate_x", "pitch_outcome", "release_speed"])
def test_disallowed_column_selection(column):
    with pytest.raises(audit.AuditBoundaryError):
        audit.check_columns(["game_pk", column], CONFIG["allowed_parquet_columns"])
    bad = copy.deepcopy(CONFIG)
    bad["allowed_parquet_columns"].append(column)
    with pytest.raises(audit.AuditBoundaryError):
        audit.validate_config(bad)


def test_source_columns_outside_allowlist_rejected():
    bad = copy.deepcopy(CONFIG)
    bad["sources"]["blend_metadata"]["columns"].append("two_strikes")
    with pytest.raises(audit.AuditBoundaryError):
        audit.validate_config(bad)


def _frame(games, pitches_per_game, level="x"):
    rows = [{"game_pk": g, "g": level} for g in range(games) for _ in range(pitches_per_game)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("games,per_game,expected", [
    (30, 17, True),    # 30 games, 510 pitches
    (29, 20, False),   # pitches pass, games fail
    (50, 9, False),    # games pass, 450 pitches fail
    (30, 16, False),   # 480 pitches
])
def test_support_requires_games_and_pitches(games, per_game, expected):
    table = audit.support_table(_frame(games, per_game), "g", ["x"], RULE)
    assert table["x"]["supported"] is expected
    assert table["x"]["games"] == games


def test_bounded_status():
    yes, no = {"supported": True}, {"supported": False}
    assert audit.bounded_status(yes, yes) == "supported"
    assert audit.bounded_status(no, no) == "unsupported"
    assert audit.bounded_status(no, yes) == "indeterminate"
    assert audit.bounded_status(None, yes) == "indeterminate"


def test_duplicate_keys_rejected():
    frame = pd.DataFrame({"game_pk": [1, 1], "at_bat_number": [1, 1], "pitch_number": [1, 1]})
    with pytest.raises(audit.AuditBoundaryError):
        audit.assert_unique_keys(frame, "dup")


def test_groups_use_frozen_volume_and_game_starter():
    frame = pd.DataFrame({"pitcher": [1, 2, 3], "starter_pitcher": [1, 9, 9], "p_throws": ["L", "R", "R"]})
    grouped = audit.add_groups(frame, {1: "high", 2: "low"})
    assert grouped["volume"].tolist() == ["high", "low", "zero"]
    assert grouped["game_role"].tolist() == ["starter", "relief", "relief"]
    assert grouped["role_x_volume"].tolist() == ["starter|high", "relief|low", "relief|zero"]


def _panel(train_volume_for_171):
    return {"volume_thresholds": {"q25": 170.0, "q75": 1514.0, "method": "m"},
            "train_players": [{"pitcher": 1, "train_pitches": 170, "train_volume": "low"},
                              {"pitcher": 2, "train_pitches": 171, "train_volume": train_volume_for_171},
                              {"pitcher": 3, "train_pitches": 2000, "train_volume": "high"}]}


def test_frozen_cutoffs_are_checked_not_recomputed():
    out = audit.check_frozen_volume(_panel("middle"), ["zero", "low", "middle", "high"])
    assert (out["q25"], out["q75"]) == (170.0, 1514.0)
    assert out["volume_of"] == {1: "low", 2: "middle", 3: "high"}


def test_volume_label_mismatch_fails_closed():
    with pytest.raises(audit.AuditBoundaryError, match="frozen cutoffs"):
        audit.check_frozen_volume(_panel("low"), ["zero", "low", "middle", "high"])


def test_expected_count_mismatch_fails_closed_and_missing_stays_unknown():
    with pytest.raises(audit.AuditBoundaryError, match="mismatch"):
        audit.check_expected([("a", 4821, 4820)])
    out = audit.check_expected([("a", 4821, 4821), ("b", None, "unknown")])
    assert out["a"]["status"] == "match"
    assert out["b"]["status"] == "unknown"
    assert out["overall"] == "unknown_items_present" and out["unknown_items"] == ["b"]
    assert audit.check_expected([("a", 1, 1)])["overall"] == "passed"


def test_require_stops_on_false():
    with pytest.raises(audit.AuditBoundaryError, match="key order"):
        audit.require(False, "blend_keys and blend_metadata key order differ")


def _status(walls, status="completed", bad_step=False):
    steps = [{"stage": f"g0-predict-{i}", "phase": "evaluation", "status": "completed", "exit_code": 0,
              "worker_wall_seconds": w} for i, w in enumerate(walls)]
    steps.append({"stage": "g0-score", "phase": "evaluation", "status": "completed",
                  "exit_code": 1 if bad_step else 0, "worker_wall_seconds": 27.0})
    return {"status": status, "steps": steps, "authoritative_worker_charged_seconds": {"evaluation": 1.0},
            "full_outer_queue_wall_seconds": 2.0}


def test_cost_uses_official_worker_walls_and_separates_ledger():
    profile = {"draws": 400, "measured": {}, "projection": {"seconds_per_row": 0.001}}
    ledger = [{"event": "start", "id": "a", "stage": "predict", "seed": 0},
              {"event": "end", "id": "a", "status": "completed", "seconds": 9999.0}]
    walls = [311.721, 311.721, 311.721, 311.721, 311.721]
    out = audit.cost_projection(CONFIG, profile, _status(walls), ledger, {"x": 1000})
    est = out["extrapolated_inference"]["estimates"]["x"]
    assert est["profile_inference_only_seconds_5_members"] == pytest.approx(5.0)
    assert est["full_worker_linear_proxy_seconds_5_members"] == pytest.approx(5.0)
    assert out["measured_components"]["observed_predict_bound"]["sum_5_member_worker_wall_seconds"] == pytest.approx(1558.605)
    assert "NOT authoritative" in out["measured_components"]["internal_ledger_crosscheck"]["label"]
    assert out["unknown_costs"]["scoring_and_bootstrap"] == "미측정"
    for bad in (_status(walls[:4]), _status(walls, status="failed"), _status(walls, bad_step=True)):
        with pytest.raises(audit.AuditBoundaryError):
            audit.cost_projection(CONFIG, profile, bad, ledger, {"x": 1})
    with pytest.raises(audit.AuditBoundaryError):
        audit.cost_projection(CONFIG, {**profile, "draws": 100}, _status(walls), ledger, {"x": 1})


def test_all_declared_sources_are_hash_verified(tmp_path):
    good = tmp_path / "good.json"
    good.write_text("{}")
    sources = {"good": {"path": str(good), "sha256": audit.sha256_file(good)},
               "p11_preparation": {"path": str(good), "sha256": "0" * 64}}
    with pytest.raises(audit.AuditBoundaryError, match="hash mismatch"):
        audit.verify_sources(sources)
    assert set(CONFIG["sources"]) >= {"p11_preparation", "official_queue_status", "processed_pitches"}


def test_run_refuses_overwrite_before_reading_data(tmp_path):
    out = tmp_path / "existing.json"
    out.write_text("{}")
    with pytest.raises(audit.AuditBoundaryError, match="overwrite"):
        audit.run(ROOT / "configs/ML-JUNE-SUPPORT-AUDIT-v1.json", out)
    assert out.read_text() == "{}"


def test_output_outside_registered_roots_rejected(tmp_path):
    with pytest.raises(audit.AuditBoundaryError, match="outside registered roots"):
        audit.check_output(tmp_path / "new.json", CONFIG)
    run_dir = Path("/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-P11-001/new.json")
    with pytest.raises(audit.AuditBoundaryError):
        audit.check_output(run_dir, CONFIG)


def test_registered_script_hash_matches_committed_script():
    assert CONFIG["implementation"]["script_sha256"] == audit.sha256_file(ROOT / "scripts/audit_june_calibration_support.py")


def test_hash_mismatch_rejected(tmp_path):
    path = tmp_path / "x.parquet"
    path.write_bytes(b"abc")
    with pytest.raises(audit.AuditBoundaryError, match="hash mismatch"):
        audit.verified_path({"path": str(path), "sha256": "0" * 64})
