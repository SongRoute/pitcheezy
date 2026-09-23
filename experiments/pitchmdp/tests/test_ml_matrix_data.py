"""Small synthetic checks for preregistered game sampling and identities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from pitchmdp.data import RAW_ALLOWLIST, hash_file
from pitchmdp.matrix_data import (canonical_hash, load_verified_processed_cache,
                                  nested_game_samples, ordered_key_hash, validate_config)
from pitchmdp.sequence_data import CACHE_VERSION, SIDECAR_COLUMNS


def sample_frame() -> pd.DataFrame:
    rows = []
    for game in range(1, 25):
        month = "2024-04-01" if game <= 12 else "2024-05-01"
        for pitch in range(1, 9):
            rows.append({"game_pk": game, "at_bat_number": pitch, "pitch_number": 1,
                         "game_date": month, "pitcher": game if pitch <= (2 if game % 2 else 6) else game+100,
                         "starter_pitcher": game, "split": "train", "result": pitch % 3})
    return pd.DataFrame(rows)


def test_nested_game_samples_are_deterministic_complete_and_label_blind():
    frame = sample_frame()
    samples, report = nested_game_samples(frame)
    changed = frame.copy()
    changed["result"] = np.arange(len(changed))[::-1]
    repeated, repeated_report = nested_game_samples(changed)
    assert report == repeated_report
    assert all(np.array_equal(samples[key], repeated[key]) for key in samples)
    assert set(samples["d25"]) < set(samples["d50"]) < set(samples["d100"])
    assert len(samples["d100"]) == 24
    assert report["fractions"]["d25"]["pitches"] == 48
    assert report["fractions"]["d50"]["pitches"] == 96


def test_samples_reject_future_and_nontrain_rows():
    frame = sample_frame()
    frame.loc[0, "game_date"] = "2026-04-01"
    with pytest.raises(ValueError, match="dates"):
        nested_game_samples(frame)
    frame.loc[0, "game_date"] = "2024-04-01"
    frame.loc[0, "split"] = "dev"
    with pytest.raises(ValueError, match="TRAIN"):
        nested_game_samples(frame)


def test_ordered_pitch_identity_and_frozen_config():
    frame = sample_frame()
    assert ordered_key_hash(frame) != ordered_key_hash(frame.iloc[::-1])
    duplicate = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        ordered_key_hash(duplicate)
    config = {"protocol": "ml_matrix_v1", "experiment_id": "EXP-P2-001",
              "input_mode": "verified_processed_cache", "year": 2025,
              "sample_seed": 42, "draws": 400, "epochs": 30, "patience": 5,
              "batch_size": 1024, "learning_rate": .0005, "width": 128}
    assert validate_config(config) == config
    assert canonical_hash(config) == canonical_hash(dict(reversed(list(config.items()))))
    with pytest.raises(ValueError, match="training settings"):
        validate_config({**config, "draws": 400, "width": 64})


def test_verified_processed_cache_checks_bytes_keys_and_historical_source_identity(tmp_path):
    root = tmp_path / "artifacts"
    (root / "processed").mkdir(parents=True)
    (root / "cache").mkdir()
    (root / "reports").mkdir()
    processed = root / "processed/pitches.parquet"
    cache = root / "cache/sequence_physics.parquet"
    base = pd.DataFrame({"game_pk": [1, 1], "at_bat_number": [1, 1],
                         "pitch_number": [1, 2], "game_date": ["2025-04-01"] * 2})
    base.to_parquet(processed, index=False)
    sidecar = base[["game_pk", "at_bat_number", "pitch_number"]].iloc[::-1].copy()
    sidecar["effective_speed"] = [91., 92.]
    sidecar["spin_axis"] = [15., 30.]
    sidecar.to_parquet(cache, index=False)
    sources = [{"file": name, "bytes": 123, "sha256": "a" * 64} for name in RAW_ALLOWLIST]
    quality = {"sources": sources, "rows": 2, "processed_sha256": hash_file(processed)}
    (root / "reports/data_quality.json").write_text(json.dumps(quality))
    identity = {"cache_version": CACHE_VERSION, "sources": sources,
                "processed_sha256": quality["processed_sha256"], "rows": 2,
                "key": ["game_pk", "at_bat_number", "pitch_number"],
                "columns": list(SIDECAR_COLUMNS)}
    metadata = {"identity": identity, "sha256": hash_file(cache)}
    meta_path = root / "cache/sequence_physics.json"
    meta_path.write_text(json.dumps(metadata))
    local = {"artifact_root": str(root), "raw_root": str(tmp_path / "absent"),
             "raw_allowlist": RAW_ALLOWLIST}
    frame = load_verified_processed_cache(local)
    assert frame.effective_speed.tolist() == [92., 91.]
    assert frame.attrs["matrix_source_provenance"]["raw_hash_verification"] == "historical_only"
    assert frame.attrs["matrix_source_provenance"]["raw_available"] is False
    meta_path.write_text(json.dumps({**metadata, "sha256": "b" * 64}))
    with pytest.raises(ValueError, match="bytes"):
        load_verified_processed_cache(local)
    meta_path.write_text(json.dumps(metadata))
    quality["sources"][0]["sha256"] = "c" * 64
    (root / "reports/data_quality.json").write_text(json.dumps(quality))
    with pytest.raises(ValueError, match="identity"):
        load_verified_processed_cache(local)
