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


def test_frozen_cutoffs_are_checked_not_recomputed():
    panel = {"volume_thresholds": {"q25": 170.0, "q75": 1514.0, "method": "m"},
             "train_players": [{"pitcher": 1, "train_pitches": 170, "train_volume": "low"},
                               {"pitcher": 2, "train_pitches": 171, "train_volume": "low"},
                               {"pitcher": 3, "train_pitches": 2000, "train_volume": "high"}]}
    out = audit.check_frozen_volume(panel, ["zero", "low", "middle", "high"])
    assert out["q25"] == 170.0 and out["q75"] == 1514.0
    assert out["stored_label_vs_frozen_cutoff_mismatches"] == 1
    assert out["volume_of"] == {1: "low", 2: "low", 3: "high"}


def test_cost_requires_five_completed_members_and_matching_draws():
    profile = {"draws": 400, "measured": {}, "projection": {"seconds_per_row": 0.001}}
    ledger = []
    for seed in range(5):
        ledger += [{"event": "start", "id": str(seed), "stage": "predict", "seed": seed},
                   {"event": "end", "id": str(seed), "status": "completed", "seconds": 311.721}]
    out = audit.cost_projection(CONFIG, profile, ledger, {"x": 1000})
    est = out["extrapolated_inference"]["estimates"]["x"]
    assert est["five_member_seconds_from_profile_rate"] == pytest.approx(5.0)
    assert est["five_member_seconds_from_measured_ledger_rate"] == pytest.approx(5.0)
    assert out["unknown_costs"]["scoring_and_bootstrap"] == "미측정"
    with pytest.raises(audit.AuditBoundaryError):
        audit.cost_projection(CONFIG, profile, ledger[:-2], {"x": 1})
    with pytest.raises(audit.AuditBoundaryError):
        audit.cost_projection(CONFIG, {**profile, "draws": 100}, ledger, {"x": 1})


def test_run_refuses_overwrite_before_reading_data(tmp_path):
    out = tmp_path / "existing.json"
    out.write_text("{}")
    with pytest.raises(audit.AuditBoundaryError, match="overwrite"):
        audit.run(ROOT / "configs/ML-JUNE-SUPPORT-AUDIT-v1.json", out)
    assert out.read_text() == "{}"


def test_hash_mismatch_rejected(tmp_path):
    path = tmp_path / "x.parquet"
    path.write_bytes(b"abc")
    with pytest.raises(audit.AuditBoundaryError, match="hash mismatch"):
        audit.verified_path({"path": str(path), "sha256": "0" * 64})
