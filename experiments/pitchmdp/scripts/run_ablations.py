"""Prespecified predictive Full/-B/-C contrasts on one fixed first-result run.

Uses the original five-epoch budget, random seed, training/calibration samples,
and complete eligible DEV pitch set. Checkpoint selection and temperature use
calibration only. DEV contrasts do not select a winner or estimate policy effects.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ["WANDB_MODE"] = "offline"

import numpy as np
import pandas as pd
import torch

from pitchmdp.data import hash_file
from pitchmdp.model import DeliveryDistribution, PitchModel, eligible, metrics, outcome_labels


def dump(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def key_fingerprint(frame: pd.DataFrame) -> str:
    keys = frame[["game_pk", "at_bat_number", "pitch_number"]].to_numpy(dtype="<i8")
    return hashlib.sha256(keys.tobytes(order="C")).hexdigest()


def fixed_samples(frame: pd.DataFrame, config: dict):
    """The exact selection expressions/order used by run_first_result.py."""
    fit_mask = eligible(frame)
    train_all = frame[(frame.split == "train") & fit_mask].copy()
    calibration_all = frame[(frame.split == "calibration") & fit_mask].copy()
    dev_cohort = frame[(frame.split == "dev") & frame.cohort_pitcher & frame.is_lad_start].copy()
    dev = dev_cohort[eligible(dev_cohort)].copy()
    train = pd.concat([
        train_all.sample(min(len(train_all), config["training_random_rows"]), random_state=config["seed"]),
        train_all[train_all.cohort_pitcher & train_all.is_lad_start],
    ]).drop_duplicates(["game_pk", "at_bat_number", "pitch_number"]).sort_index()
    calibration = calibration_all.sample(
        min(len(calibration_all), config["calibration_random_rows"]), random_state=config["seed"])
    dev_sample = dev.sample(min(len(dev), config["dev_max_rows"]), random_state=config["seed"]).sort_index()
    if len(dev_sample) != len(dev):
        raise ValueError("This ablation requires the full eligible DEV set; original run config subsamples it")
    if not (train.game_date.max() < calibration.game_date.min() < dev_sample.game_date.min()):
        raise ValueError("Date split overlaps")
    if set(train.game_pk) & set(dev_sample.game_pk):
        raise ValueError("Training and DEV games overlap")
    return train_all, train, calibration, dev_sample


def evaluate(model, delivery, dev, labels):
    started = time.perf_counter()
    primary = delivery.predict(model, dev)
    return {
        "primary_prepitch_type_conditional": metrics(labels, primary),
        "retrospective_actual_location_diagnostic": metrics(labels, model.predict(dev)),
        "per_pitcher_primary": {
            str(int(pitcher)): metrics(labels[dev.pitcher.to_numpy() == pitcher],
                                      primary[dev.pitcher.to_numpy() == pitcher])
            for pitcher in sorted(dev.pitcher.unique())
        },
        "evaluation_seconds": time.perf_counter() - started,
    }


def release_model(model) -> None:
    model.net.to("cpu")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing first-result SSD run directory")
    args = parser.parse_args()
    local = json.loads((PROJECT / "configs/local.json").read_text())
    root = Path(local["artifact_root"])
    volume = Path("/Volumes/T7 Shield")
    run = args.run.resolve()
    if not volume.is_mount() or not root.resolve().is_relative_to(volume.resolve()):
        raise SystemExit("Mounted T7 Shield required; no internal fallback")
    if not run.is_dir() or not run.is_relative_to(root.resolve()):
        raise SystemExit("--run must be an existing directory inside the configured SSD root")
    if Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Use the configured existing Python environment")
    config = json.loads((run / "config.json").read_text())
    if config["epochs"] != 5:
        raise SystemExit("This prespecified comparison requires the original five-epoch budget")
    checkpoint = run / "pitch_model.pt"
    if not checkpoint.is_file():
        raise SystemExit("Original Full checkpoint is required")
    data_path = root / "processed/pitches.parquet"
    original_data = json.loads((run / "data_quality.json").read_text())
    processed_hash = hash_file(data_path)
    if processed_hash != original_data["processed_sha256"]:
        raise SystemExit("Processed data changed since the Full run; refusing an unmatched comparison")
    original_prediction = json.loads((run / "prediction_metrics.json").read_text())
    original_coverage = json.loads((run / "coverage.json").read_text())
    source_hashes = {}
    for source in [Path(__file__), PROJECT / "pitchmdp/model.py", PROJECT / "configs/first_result.json"]:
        relative = source.relative_to(PROJECT)
        destination = run / "ablation_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[str(relative)] = hash_file(source)
    start = time.perf_counter()
    report = {
        "status": "preparing", "started_at_utc": now(), "run": str(run),
        "seed": int(config["seed"]), "epochs_per_variant": 5,
        "variant_order": ["full_existing_checkpoint", "minus_b", "minus_c"],
        "processed_sha256": processed_hash, "code_hashes": source_hashes,
        "feature_changes": {
            "full": "Batter ID, three strict-prior-date batter summaries, previous pitch type, current count and game state",
            "minus_b": "Remove batter ID, batter_pa_prior, batter_obp_prior, batter_k_prior; retain batter handedness",
            "minus_c": "Remove prev_pitch_type; retain count and current game state. Full uses only previous pitch type as its sequence feature",
        },
        "evaluation_contract": {
            "primary": "Pre-pitch conditional on observed pitch type, integrating over identical training-only delivery-location draws",
            "diagnostic_only": "Retrospective conditional prediction using actual delivered location",
            "selection": "Epoch and temperature selected exclusively on calibration; no model winner is selected from DEV",
            "interpretation": "Exploratory predictive contrasts; no strategy benefit or causal policy effect is estimated",
            "dev_prior_exposure": "This 2025 DEV period is not untouched project-level confirmation data",
        },
        "models": {}, "comparisons": {},
    }
    dump(run / "ablations.json", report)
    frame = pd.read_parquet(data_path)
    train_all, train, calibration, dev = fixed_samples(frame, config)
    del frame
    gc.collect()
    report["samples"] = {
        name: {"n": len(part), "key_sha256": key_fingerprint(part),
               "games": int(part.game_pk.nunique()),
               "dates": [str(part.game_date.min().date()), str(part.game_date.max().date())]}
        for name, part in [("train", train), ("calibration", calibration), ("dev", dev)]
    }
    for label, count_key in [("train", "train_used"), ("calibration", "calibration_used"), ("dev", "dev_used")]:
        if report["samples"][label]["n"] != original_coverage[count_key]:
            raise ValueError(f"{label} row count differs from original Full run")
    # Same training population, RNG seed, group iteration, and draw budget as Full.
    delivery = DeliveryDistribution().fit(train_all, draws=config["delivery_draws"], seed=config["seed"])
    del train_all
    gc.collect()
    labels = outcome_labels(dev)
    full = PitchModel.load(checkpoint)
    if full.encoder.variant != "full" or full.seed != config["seed"]:
        raise ValueError("Full checkpoint variant/seed does not match the run config")
    if len(full.training_report["history"]) != config["epochs"]:
        raise ValueError("Full checkpoint did not use the same epoch budget")
    full_result = evaluate(full, delivery, dev, labels)
    for family in ["primary_prepitch_type_conditional", "retrospective_actual_location_diagnostic"]:
        for measure in ["log_loss", "brier_multiclass"]:
            if not np.isclose(full_result[family][measure], original_prediction[family][measure], atol=1e-6, rtol=0):
                raise ValueError(f"Full {family} {measure} does not reproduce original evaluation")
    full_result["training"] = full.training_report
    full_result["checkpoint"] = str(checkpoint)
    full_result["checkpoint_reused"] = True
    report["models"]["full"] = full_result
    release_model(full)
    del full
    report["status"] = "training"
    dump(run / "ablations.json", report)
    print("Fixed samples and Full evaluation reproduced", report["samples"], flush=True)
    for variant in ["minus_b", "minus_c"]:
        variant_checkpoint = run / f"{variant}.pt"
        reused = variant_checkpoint.exists()
        fit_started = time.perf_counter()
        if reused:
            model = PitchModel.load(variant_checkpoint)
            if (model.encoder.variant != variant or model.seed != config["seed"] or
                    model.training_report["n_train"] != len(train) or
                    model.training_report["n_calibration"] != len(calibration) or
                    len(model.training_report["history"]) != config["epochs"]):
                raise ValueError(f"Existing {variant} checkpoint has an incompatible training contract")
        else:
            model = PitchModel(variant=variant, seed=config["seed"]).fit(train, calibration, epochs=config["epochs"])
            temporary = variant_checkpoint.with_suffix(".pt.tmp")
            model.save(temporary)
            temporary.replace(variant_checkpoint)
        fit_wall_seconds = time.perf_counter() - fit_started
        result = evaluate(model, delivery, dev, labels)
        result.update({"training": model.training_report, "checkpoint": str(variant_checkpoint),
                       "checkpoint_reused": reused, "fit_or_reload_wall_seconds": fit_wall_seconds})
        report["models"][variant] = result
        report["comparisons"][variant + "_minus_full"] = {
            "primary_log_loss_delta": result["primary_prepitch_type_conditional"]["log_loss"] - full_result["primary_prepitch_type_conditional"]["log_loss"],
            "primary_brier_delta": result["primary_prepitch_type_conditional"]["brier_multiclass"] - full_result["primary_prepitch_type_conditional"]["brier_multiclass"],
            "sign": "Positive means removing the feature group worsened this prediction metric",
            "exploratory_only": True,
        }
        dump(run / "ablations.json", report)
        print(variant, {"primary": result["primary_prepitch_type_conditional"]["log_loss"],
                        "training_seconds": model.training_report["seconds"]}, flush=True)
        release_model(model)
        del model
    report["status"] = "complete"
    report["finished_at_utc"] = now()
    report["pipeline_seconds"] = time.perf_counter() - start
    dump(run / "ablations.json", report)
    print("COMPLETE", run / "ablations.json", flush=True)


if __name__ == "__main__":
    main()
