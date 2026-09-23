"""Pure sampling and identity helpers for the registered ML matrix pilot.

The pilot has one 2025 temporal fold.  Sampling uses game identity and
pre-pitch metadata only; labels are deliberately absent from this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import KEY, RAW_ALLOWLIST, hash_file
from .sequence_data import CACHE_VERSION, SIDECAR_COLUMNS, join_physics


LEGACY_SEEDS = (42, 43, 44, 45, 46)
SCREEN_SEEDS = (0, 1, 2)
CELLS = {
    "R2-MLP": ("flatten_mlp", LEGACY_SEEDS, "legacy"),
    "R2-TF": ("transformer", LEGACY_SEEDS, "legacy"),
    "D1-25": ("flatten_mlp", SCREEN_SEEDS, "d25"),
    "D1-50": ("flatten_mlp", SCREEN_SEEDS, "d50"),
    "D1-100": ("flatten_mlp", SCREEN_SEEDS, "d100"),
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def ordered_key_hash(frame: pd.DataFrame) -> str:
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Pitch keys must be complete and unique")
    return hashlib.sha256(frame[KEY].to_numpy(np.int64).tobytes()).hexdigest()


def validate_config(config: dict) -> dict:
    """Reject undefined axes before a costly data load or fit starts."""
    required = {"protocol", "experiment_id", "input_mode", "year", "sample_seed", "draws",
                "epochs", "patience", "batch_size", "learning_rate", "width"}
    if set(config) != required or config["protocol"] != "ml_matrix_v1":
        raise ValueError("Unsupported ML matrix config schema or protocol")
    if not isinstance(config["experiment_id"], str) or not config["experiment_id"].strip():
        raise ValueError("experiment_id is required")
    if config["input_mode"] != "verified_processed_cache":
        raise ValueError("Pilot requires explicit verified_processed_cache input_mode")
    if config["year"] != 2025 or config["sample_seed"] != 42 or config["draws"] != 400:
        raise ValueError("Pilot fixes the 2025 fold, sample seed 42 and 400 draws")
    if (config["epochs"], config["patience"], config["batch_size"],
        config["learning_rate"], config["width"]) != (30, 5, 1024, .0005, 128):
        raise ValueError("Pilot fixes the original neural training settings")
    return config


def load_verified_processed_cache(local: dict) -> pd.DataFrame:
    """Read byte-verified derived data when historical raw inputs are unavailable.

    This establishes the current processed/cache bytes against frozen manifests;
    it does not claim to reproduce ingestion from the historical raw files.
    """
    if local.get("raw_allowlist") != RAW_ALLOWLIST:
        raise ValueError("Configured raw allowlist differs from approved 2023-25 sources")
    root = Path(local["artifact_root"])
    quality = json.loads((root / "reports/data_quality.json").read_text())
    metadata = json.loads((root / "cache/sequence_physics.json").read_text())
    sources = quality.get("sources")
    if (not isinstance(sources, list) or [s.get("file") for s in sources] != RAW_ALLOWLIST or
            any(not isinstance(s.get("bytes"), int) or s["bytes"] <= 0 or
                not isinstance(s.get("sha256"), str) or len(s["sha256"]) != 64 for s in sources)):
        raise ValueError("Frozen quality manifest lacks approved historical raw identities")
    expected = {"cache_version": CACHE_VERSION, "sources": sources,
                "processed_sha256": quality["processed_sha256"], "rows": quality["rows"],
                "key": KEY, "columns": list(SIDECAR_COLUMNS)}
    if metadata.get("identity") != expected:
        raise ValueError("Physics cache identity differs from frozen quality manifest")
    processed = root / "processed/pitches.parquet"
    cache = root / "cache/sequence_physics.parquet"
    if hash_file(processed) != quality["processed_sha256"] or hash_file(cache) != metadata.get("sha256"):
        raise ValueError("Processed or physics cache bytes differ from their manifest")
    frame, sidecar = pd.read_parquet(processed), pd.read_parquet(cache)
    if len(frame) != quality["rows"] or len(sidecar) != quality["rows"]:
        raise ValueError("Derived data row count differs from frozen manifest")
    dates = pd.to_datetime(frame.game_date)
    if dates.isna().any() or not dates.dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Derived data contains missing or unapproved dates")
    result = join_physics(frame, sidecar)
    result.sort_values(["game_date", *KEY], inplace=True, ignore_index=True)
    result.attrs["sequence_data_identity"] = expected
    result.attrs["matrix_source_provenance"] = {
        "input_mode": "verified_processed_cache", "raw_available": all(
            (Path(local["raw_root"]) / source["file"]).is_file() for source in sources),
        "raw_hash_verification": "historical_only", "processed_cache_verification": "current_verified",
        "quality_manifest_sha256": hash_file(root / "reports/data_quality.json"),
        "cache_manifest_sha256": hash_file(root / "cache/sequence_physics.json"),
        "processed_sha256": quality["processed_sha256"], "cache_sha256": metadata["sha256"]}
    return result


def nested_game_samples(full: pd.DataFrame, seed: int = 42) -> tuple[dict[str, np.ndarray], dict]:
    """Return nested 25/50/100% TRAIN game sets, stratified by month and role mix.

    Role mix is the share of eligible pitches from the recorded starting pitcher.
    It is known before the outcome and only affects game inclusion.  Within each
    month/role stratum the seeded ranking is fixed; quarter and half prefixes
    therefore stay nested, including in small strata.
    """
    needed = set(KEY + ["game_date", "pitcher", "starter_pitcher"])
    if not needed.issubset(full.columns) or full.empty:
        raise ValueError("Eligible full TRAIN requires game/date/pitcher metadata")
    if full[KEY].isna().any().any() or full.duplicated(KEY).any():
        raise ValueError("Full TRAIN pitch keys must be complete and unique")
    if "split" in full and not full.split.eq("train").all():
        raise ValueError("D100 may contain TRAIN rows only")
    dates = pd.to_datetime(full.game_date)
    if dates.isna().any() or not dates.between("2023-05-15", "2025-04-30").all():
        raise ValueError("D100 dates outside approved TRAIN window")
    month = dates.dt.to_period("M").astype(str)
    starter = full.pitcher.eq(full.starter_pitcher).to_numpy()
    games = pd.DataFrame({"game_pk": full.game_pk.to_numpy(), "month": month.to_numpy(),
                          "starter": starter})
    grouped = games.groupby("game_pk", sort=True)
    summary = grouped.agg(month=("month", "first"), pitches=("starter", "size"),
                          starter_share=("starter", "mean"))
    if grouped.month.nunique().gt(1).any():
        raise ValueError("One game spans multiple recorded months")
    summary["role_band"] = pd.cut(summary.starter_share, bins=[-1., .25, .5, .75, 1.],
                                   labels=["0-25", "25-50", "50-75", "75-100"])
    rng = np.random.default_rng(seed)
    ranked_games: list[tuple[float, str, str, int]] = []
    for _, stratum in summary.groupby(["month", "role_band"], observed=True, sort=True):
        ids = stratum.index.to_numpy()
        ranked = ids[rng.permutation(len(ids))]
        for rank, game_id in enumerate(ranked):
            record = summary.loc[game_id]
            ranked_games.append(((rank+.5)/len(ids), str(record["month"]),
                                 str(record["role_band"]), int(game_id)))
    ranked_games.sort()
    n_games = len(ranked_games)
    selected25 = {item[3] for item in ranked_games[:int(np.ceil(n_games*.25))]}
    selected50 = {item[3] for item in ranked_games[:int(np.ceil(n_games*.50))]}
    all_ids = set(int(x) for x in summary.index)
    if not selected25 <= selected50 <= all_ids:
        raise AssertionError("Nested game sampling failed")
    result = {"d25": np.array(sorted(selected25), dtype=np.int64),
              "d50": np.array(sorted(selected50), dtype=np.int64),
              "d100": np.array(sorted(all_ids), dtype=np.int64)}
    report = {"sampling_seed": seed, "unit": "whole game", "strata": "month x starting-pitch share band",
              "role_bands": ["0-25", "25-50", "50-75", "75-100"], "fractions": {}}
    for name, ids in result.items():
        chosen = full.game_pk.isin(ids)
        report["fractions"][name] = {
            "games": len(ids), "game_fraction": len(ids)/len(all_ids),
            "pitches": int(chosen.sum()), "pitch_fraction": float(chosen.mean()),
            "pitchers": int(full.loc[chosen, "pitcher"].nunique()),
            "starter_pitch_fraction": float(full.loc[chosen, "pitcher"].eq(full.loc[chosen, "starter_pitcher"]).mean()),
            "game_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "rows_sha256": ordered_key_hash(full.loc[chosen]),
        }
    return result, report


def assert_output_path(path: Path, artifact_root: Path) -> None:
    output, root = path.resolve(), artifact_root.resolve()
    if output == root or not output.is_relative_to(root) or not Path("/Volumes/T7 Shield").is_mount():
        raise ValueError("New output must be a child of the mounted configured SSD root")
