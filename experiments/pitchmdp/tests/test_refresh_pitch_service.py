"""Small synthetic refresh checks; no source datasets or checkpoints are loaded."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refresh_pitch_service as refresh
from minimal_pitch_service import RequestError, validate_request
from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS, add_batter_style_history


META = {"schema_version": 1, "training_cutoff": "2025-04-30",
        "pitchers": {"10": {"p_throws": "R", "pitch_types": ["FF", "SL"], "name": "Synthetic"},
                     "11": {"p_throws": "L", "pitch_types": ["FF"]}}}
REQUEST = {"date": "2025-08-16", "inning": 6, "topbot": "Top", "outs": 1,
           "bases": 1, "home_score": 2, "away_score": 2, "balls": 1, "strikes": 2,
           "pitcher_id": 10, "batter_id": 20, "batter_stand": "L"}


def history():
    rows = []
    for day, kind, count in [("2025-04-01", "SL", 30), ("2025-08-14", "SL", 19),
                             ("2025-08-15", "FF", 20), ("2025-08-15", "NEW", 25),
                             ("2025-08-16", "SL", 100)]:
        for index in range(count):
            rows.append({"batter": 20 if index % 2 else 21, "pitcher": 10,
                "game_date": day, "pitch_type": kind, "p_throws": "R",
                "description": "hit_into_play" if index % 3 else "swinging_strike",
                "events": "double" if index % 3 else "strikeout", "is_pa_terminal": True,
                "launch_angle": 8. if index % 2 else np.nan})
    return pd.DataFrame(rows)


@pytest.fixture
def snapshot():
    return refresh.build_sidecar(history(), "2025-08-16", META, {"synthetic": True})


def test_profile_matches_canonical_next_day_probe_and_includes_last_day():
    frame = history()
    prior = frame.loc[frame.game_date.lt("2025-08-16")].copy()
    probes = pd.DataFrame({"batter": [20, 21], "game_date": "2025-08-16",
                           "description": "", "events": "", "is_pa_terminal": False})
    expected = add_batter_style_history(pd.concat([prior, probes], ignore_index=True)).tail(2)
    actual = refresh.refresh_profiles(frame, "2025-08-16")
    for (_, row), batter in zip(expected.iterrows(), (20, 21)):
        profile = actual["profiles"][str(batter)]
        np.testing.assert_array_equal(profile["rates"], row[list(STYLE_COLUMNS)].to_numpy(float))
        np.testing.assert_array_equal(profile["reliabilities"], row[list(RELIABILITY_COLUMNS)].to_numpy(float))
    without_last_day = refresh.refresh_profiles(frame, "2025-08-15")
    assert actual["profiles"]["20"]["reliabilities"] != without_last_day["profiles"]["20"]["reliabilities"]
    assert actual["source_date_max"] == "2025-08-15"
    assert actual["source_rows_before_date"] == 94
    assert actual["default_profile"]["reliabilities"] == [0.] * 6


def test_same_date_future_and_row_order_cannot_change_snapshot():
    frame = history()
    expected = refresh.refresh_profiles(frame, "2025-08-16")
    frame.loc[frame.game_date.ge("2025-08-16"), ["events", "description", "batter"]] = ["home_run", "hit_into_play", 999]
    actual = refresh.refresh_profiles(frame.sample(frac=1, random_state=4), "2025-08-16")
    assert actual == expected
    assert "999" not in actual["profiles"]


def test_2026_effective_date_uses_only_2025_data_and_empty_pool_is_valid():
    frame = history()
    # Exercise the last permitted real data date, including the entire day.
    frame["game_date"] = "2025-12-31"
    current = refresh.refresh_profiles(frame, "2026-01-01")
    shifted = frame.copy()
    shifted["game_date"] = "2025-12-30"
    expected = refresh.refresh_profiles(shifted, "2025-12-31")
    assert current["profiles"]["20"]["rates"] == expected["profiles"]["20"]["rates"]
    assert current["profiles"]["20"]["reliabilities"] == expected["profiles"]["20"]["reliabilities"]
    assert current["source_date_max"] == "2025-12-31"
    empty = refresh.refresh_profiles(frame, "2023-01-01")
    assert empty["profiles"] == {} and empty["source_date_max"] is None
    assert empty["default_profile"]["reliabilities"] == [0.] * 6
    frame.loc[0, "game_date"] = "2026-01-01"
    with pytest.raises(ValueError, match="2023–2025"):
        refresh.refresh_profiles(frame, "2026-01-02")


def test_original_support_recent_threshold_and_missing_pitcher(snapshot):
    assert set(snapshot["pitchers"]) == {"10", "11"}
    pitcher = snapshot["pitchers"]["10"]
    assert pitcher["pitch_types"] == ["FF"]
    assert pitcher["masked_pitch_types"] == ["SL"]
    assert pitcher["unmodeled_recent_pitch_types"] == ["NEW"]
    assert pitcher["p_throws"] == "R"
    assert snapshot["repertoire_counts"]["10"] == {"FF": 20}
    assert not snapshot["pitchers"]["11"]["available"]
    stale = refresh.build_sidecar(history(), "2026-09-21", META)
    assert not stale["pitchers"]["10"]["available"]
    assert stale["pitchers"]["10"]["pitch_types"] == []


def test_handedness_cannot_be_invented_or_change_frozen_support():
    for hand in (None, "L"):
        frame = history()
        frame["p_throws"] = hand
        result = refresh.build_sidecar(frame, "2025-08-16", META)
        assert not result["pitchers"]["10"]["available"]


class FakeEngine:
    """Exercise adapter mutations without invoking torch or frozen checkpoints."""
    def __init__(self):
        self.metadata = deepcopy(META)
        self._lock = threading.RLock()

    def predict_counts(self, request):
        row = validate_request(request, self.metadata)
        p = np.full((4, 3, 1, len(row["pitch_types"]), 10), 1 / 9)
        p[..., -1] = 0  # Structural zero must survive optional calibration.
        return {**row, "probabilities": {"blend": p}}

    def recommend(self, request):
        row = self.predict_counts(request)
        return {"recommendations": row["pitch_types"], "profile_source": row["profile_source"],
                "profile_as_of": row["profile_as_of"], "assumptions": [],
                "probabilities": row["probabilities"]["blend"][0, 0, 0, 0].tolist()}


def test_wrapper_unknown_batter_flags_and_metadata_restoration(snapshot):
    engine = FakeEngine()
    original = engine.metadata
    response = refresh.recommend_with_sidecar(engine, {**REQUEST, "batter_id": 999}, snapshot)
    assert response["recommendations"] == ["FF"]
    assert "missing_batter_zero_reliability" in response["refresh"]["flags"]
    assert response["baseline_policy"] == "prior_90_day_supported_repertoire_frequency"
    assert engine.metadata is original and engine.metadata == META
    with pytest.raises(RequestError, match="Unsupported"):
        refresh.recommend_with_sidecar(engine, {**REQUEST, "pitcher_id": 999}, snapshot)
    with pytest.raises(RequestError) as caught:
        refresh.recommend_with_sidecar(engine, {**REQUEST, "pitcher_id": 11}, snapshot)
    assert caught.value.code == "unavailable_pitcher"


@pytest.mark.parametrize("request_date,code", [(None, "invalid_request"),
    ("2025-06-30", "model_date_conflict"), ("2025-08-15", "profile_date_conflict")])
def test_request_date_guards(snapshot, request_date, code):
    request = {**REQUEST, "date": request_date}
    with pytest.raises(RequestError) as caught:
        refresh.recommend_with_sidecar(FakeEngine(), request, snapshot)
    assert caught.value.code == code


@pytest.mark.parametrize("profile_date", [None, "2025-08-16", "2025-08-17"])
def test_explicit_profile_requires_strict_prior_date(snapshot, profile_date):
    profile = {"rates": [.5] * 6, "reliabilities": [.5] * 6, "as_of": profile_date}
    with pytest.raises(RequestError):
        refresh.recommend_with_sidecar(FakeEngine(), {**REQUEST, "batter_profile": profile}, snapshot)


def test_valid_explicit_profile_and_provenance_guard(snapshot):
    profile = {"rates": [.5] * 6, "reliabilities": [.5] * 6, "as_of": "2025-08-15"}
    request = {**REQUEST, "batter_profile": profile}
    response = refresh.recommend_with_sidecar(FakeEngine(), request, snapshot)
    assert response["profile_source"] == "explicit_prior_date_profile"
    assert "user_supplied_profile_provenance_not_independently_verified" in response["refresh"]["flags"]
    profile["source_date_max"] = "2025-08-16"
    with pytest.raises(RequestError):
        refresh.recommend_with_sidecar(FakeEngine(), request, snapshot)


def test_sidecar_hash_immutable_write_and_bundle_binding(snapshot, tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "VOLUME", tmp_path)
    monkeypatch.setattr(refresh, "RUNS", tmp_path)
    monkeypatch.setattr(Path, "is_mount", lambda self: self == tmp_path)
    output = tmp_path / "new-run"
    refresh.save_sidecar(snapshot, output)
    assert refresh.load_sidecar(output, META) == snapshot
    with pytest.raises(FileExistsError):
        refresh.save_sidecar(snapshot, output)
    with pytest.raises(ValueError, match="different original bundle"):
        refresh.load_sidecar(output, {**META, "changed": True})
    with pytest.raises(ValueError, match="directly under"):
        refresh.save_sidecar(snapshot, tmp_path / "new-run" / "nested")
    (output / "snapshot.json").write_text(json.dumps({**snapshot, "source_date_max": "2025-08-16"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        refresh.load_sidecar(output)


def calibration():
    return {"accepted": True, "bias": [1.] + [0.] * 9, "available_from": "2025-08-16",
            "original_metadata_sha256": refresh.metadata_hash(META)}


def test_calibration_applies_only_blend_preserves_zeros_and_restores_method(snapshot):
    engine = FakeEngine()
    unchanged = refresh.recommend_with_sidecar(engine, REQUEST, snapshot)
    response = refresh.recommend_with_sidecar(engine, REQUEST, snapshot, calibration())
    assert response["probabilities"][0] > unchanged["probabilities"][0]
    assert response["probabilities"][-1] == 0
    assert sum(response["probabilities"]) == pytest.approx(1.)
    assert response["outcome_calibration"]["applied"]
    assert "predict_counts" not in engine.__dict__ and engine.metadata == META
    rejected = refresh.recommend_with_sidecar(engine, REQUEST, snapshot, {"accepted": False})
    assert rejected["probabilities"] == unchanged["probabilities"]
    assert not rejected["outcome_calibration"]["applied"]


def test_calibration_date_binding_and_exception_restore(snapshot):
    repair = calibration()
    repair["available_from"] = "2025-08-17"
    with pytest.raises(RequestError) as caught:
        refresh.recommend_with_sidecar(FakeEngine(), REQUEST, snapshot, repair)
    assert caught.value.code == "calibration_date_conflict"
    repair = {**calibration(), "original_metadata_sha256": "wrong"}
    with pytest.raises(ValueError, match="different original bundle"):
        refresh.recommend_with_sidecar(FakeEngine(), REQUEST, snapshot, repair)
    engine = FakeEngine()
    original = engine.metadata
    def failure(request):
        raise RuntimeError("synthetic failure")
    engine.predict_counts = failure
    with pytest.raises(RuntimeError, match="synthetic failure"):
        refresh.recommend_with_sidecar(engine, REQUEST, snapshot, calibration())
    assert engine.metadata is original and engine.predict_counts is failure


def test_tampered_support_and_cutoff_rejected(snapshot):
    bad = deepcopy(snapshot)
    bad["pitchers"]["10"]["pitch_types"].append("NEW")
    bad["repertoire_counts"]["10"]["NEW"] = 25
    with pytest.raises(ValueError, match="unsupported model actions"):
        refresh.validate_sidecar(bad, META)
    bad = deepcopy(snapshot)
    bad["profile_cutoff"] = "2025-08-16"
    with pytest.raises(ValueError, match="cutoff"):
        refresh.validate_sidecar(bad, META)
