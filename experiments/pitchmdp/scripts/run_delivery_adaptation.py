"""Exploratory as-of-game delivery shifts; fixed model, calibration-only selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_data import HistoryStore, prepare_frame
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_sequence_ablations import reconstruct_samples, rows_hash
from run_sequence_pilot import dump, paired

CHANNELS = (0, 1, 4, 5)
RAW_CHANNELS = ("effective_speed", "release_spin_rate", "pfx_x", "pfx_z")
GROUPS = ("game_pk", "pitcher", "pitch_type")
SETTINGS = {"static": float("inf"), "k10": 10., "k30": 30., "k100": 100.}
REFERENCE_MINIMUM = 50


def complete_physics(frame):
    raw = frame[list(RAW_CHANNELS)].to_numpy(dtype=np.float64, na_value=np.nan)
    return np.isfinite(raw).all(axis=1) & frame.pitch_type.notna().to_numpy() & frame.pitcher.notna().to_numpy()


def build_prior_physics(frame, physical):
    """All strictly earlier complete same-game/pitcher/type deliveries, across PAs.

    Return count[n] and standardized four-channel mean[n,4] in input row order.
    Sort by unique game/PA/pitch keys so shuffled input cannot introduce lookahead.
    Missing physics is not treated as an observed TRAIN-median delivery.
    """
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Unique complete pitch keys are required")
    dates = pd.to_datetime(frame.game_date)
    if dates.isna().any() or not dates.dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Only approved 2023–2025 rows are allowed")
    physical = np.asarray(physical)
    if physical.shape != (len(frame), 8) or not np.isfinite(physical).all():
        raise ValueError("Expected finite normalized physical rows [n,8]")
    order = np.lexsort(tuple(frame[key].to_numpy() for key in reversed(KEY)))
    work = frame.iloc[order][list(GROUPS)].reset_index(drop=True).copy()
    valid = complete_physics(frame)[order]
    work["observed"] = valid.astype(np.int32)
    names = [f"z{j}" for j in range(4)]
    for j, name in enumerate(names):
        work[name] = np.where(valid, physical[order, CHANNELS[j]], 0.).astype(np.float64)
    cumulative = work.groupby(list(GROUPS), sort=False, dropna=False)[["observed", *names]].cumsum()
    # Shift within groups rather than subtract current values; no cancellation
    # can make current/future feature mutations influence the prior sum.
    prior = cumulative.groupby([work[key] for key in GROUPS], sort=False, dropna=False).shift().fillna(0)
    counts = prior.observed.to_numpy(np.int32)
    means = prior[names].to_numpy()/np.maximum(counts[:, None], 1)
    output_count, output_mean = np.empty(len(frame), np.int32), np.empty((len(frame), 4), np.float32)
    output_count[order], output_mean[order] = counts, means
    return output_count, output_mean


def fit_reference(train, physical, minimum=REFERENCE_MINIMUM):
    """TRAIN-only eligible reference; stable pitcher/type, then type, then global."""
    if not len(train) or not train.split.eq("train").all() or not pd.to_datetime(train.game_date).between("2023-05-15", "2025-04-30").all():
        raise ValueError("Reference means require only approved TRAIN rows")
    physical = np.asarray(physical)
    if physical.shape != (len(train), 8) or not np.isfinite(physical).all():
        raise ValueError("Expected aligned finite physical values")
    valid = complete_physics(train)
    work = train.loc[valid, ["pitcher", "pitch_type"]].reset_index(drop=True).copy()
    if not len(work):
        raise ValueError("No complete TRAIN deliveries")
    names = [f"z{j}" for j in range(4)]
    work[names] = physical[valid][:, CHANNELS]
    grouped = work.groupby(["pitcher", "pitch_type"], sort=True)
    pitcher_type = grouped[names].mean()
    pitcher_type = pitcher_type.loc[grouped.size().ge(minimum)]
    return {"pitcher_type": pitcher_type, "type": work.groupby("pitch_type")[names].mean(),
            "global": work[names].mean().to_numpy(),
            "report": {"eligible_train_rows": len(train), "complete_train_rows": len(work),
                       "minimum_pitcher_type_count": minimum, "pitcher_type_groups": len(pitcher_type),
                       "type_groups": int(work.pitch_type.nunique()), "fit_date_max": str(pd.to_datetime(train.game_date).max().date())}}


def reference_for_rows(frame, reference):
    indexes = pd.MultiIndex.from_frame(frame[["pitcher", "pitch_type"]])
    player = reference["pitcher_type"].reindex(indexes).to_numpy()
    league = reference["type"].reindex(frame.pitch_type.to_numpy()).to_numpy()
    has_player, has_type = np.isfinite(player).all(axis=1), np.isfinite(league).all(axis=1)
    means = np.where(has_player[:, None], player, np.where(has_type[:, None], league, reference["global"]))
    # 2 pitcher/type, 1 league/type, 0 global. Origin is auditable per row.
    return means.astype(np.float32), np.where(has_player, 2, np.where(has_type, 1, 0)).astype(np.int8)


def adaptive_shift(counts, means, reference_means, k):
    counts = np.asarray(counts)
    if means.shape != reference_means.shape or means.shape != (len(counts), 4) or (counts < 0).any() or k <= 0:
        raise ValueError("Invalid as-of counts, reference shape or shrinkage")
    if np.isinf(k):
        return np.zeros_like(means, dtype=np.float32)
    shift = (means-reference_means)*(counts/(counts+k))[:, None]
    shift[counts == 0] = 0.
    return shift.astype(np.float32)


def shifted_delivery_logits(model, store, context, delivery, rows, shift, chunk_size=256):
    """Translate only four channels in unchanged TRAIN joint candidate vectors."""
    logits = []
    for begin in range(0, len(rows), chunk_size):
        selected = np.asarray(rows[begin:begin+chunk_size])
        frame = store.frame.iloc[selected]
        candidates, _ = delivery.sample(frame)
        candidates = candidates.copy()
        candidates[..., CHANNELS] += shift[selected, None, :]
        repeated = np.repeat(selected, delivery.draws)
        tokens, valid = store.gather(repeated, current=candidates.reshape(-1, 8))
        ctx = np.repeat(context.transform(frame), delivery.draws, axis=0)
        logits.append(model.logits((tokens, valid, ctx)).reshape(len(selected), delivery.draws, -1))
    return np.concatenate(logits)


def calibrate_temperature(logits, labels):
    def loss(temperature):
        p = softmax(logits/temperature, axis=-1).mean(1)
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.)).mean())
    fitted = minimize_scalar(loss, bounds=(.5, 2.5), method="bounded")
    return {"temperature": float(fitted.x), "calibrated_log_loss": float(fitted.fun), "uncalibrated_log_loss": loss(1.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Original fixed Transformer42 run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="CPU preparation only; no model inference")
    parser.add_argument("--check-draws", type=int, choices=[0, 100], default=100,
                        help="Prespecified static versus CAL-selected setting at 100 TRAIN draws; 0 disables sensitivity")
    args = parser.parse_args()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.is_relative_to(volume.resolve()) or Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Use the mounted configured SSD and existing Python")
    run = args.run.resolve()
    output = (args.output or run/datetime.now(timezone.utc).strftime("delivery-adaptation-%Y%m%dT%H%M%SZ")).resolve()
    if not run.is_relative_to(root) or not output.is_relative_to(root) or output == run or run.is_relative_to(output):
        raise SystemExit("Use distinct configured SSD output directory")
    cfg = json.loads((run/"config.json").read_text())
    original = json.loads((run/"source_hashes.json").read_text())
    if cfg["seed"] != 42 or json.loads((run/"runtime.json").read_text())["source_hashes_end"] != original:
        raise ValueError("Original fixed-seed source contract changed")
    for rel, digest in original.items():
        if hash_file(PROJECT/rel) != digest or hash_file(run/"source"/rel) != digest:
            raise ValueError(f"Original source changed: {rel}")
    protocol = PROJECT/"docs/DELIVERY_ADAPTATION_PROTOCOL.md"
    sources = {**original, "scripts/run_sequence_ablations.py": hash_file(PROJECT/"scripts/run_sequence_ablations.py"),
               str(Path(__file__).resolve().relative_to(PROJECT)): hash_file(Path(__file__)),
               str(protocol.relative_to(PROJECT)): hash_file(protocol)}
    reference_paths = [run/name for name in ("all_transformer.pt", "config.json", "encoders.pkl", "samples.json",
                                            "cohort_manifest.json", "heldout_predictions.npz")]
    reference_hashes = {str(path): hash_file(path) for path in reference_paths}
    manifest = {"base_run": str(run), "model_seed": 42, "training": "none; fixed checkpoint inference only",
                "settings": {name: None if np.isinf(k) else k for name, k in SETTINGS.items()},
                "primary_draws": 25, "sensitivity_draws": args.check_draws,
                "sensitivity_selection": "Keep the primary25 CAL-selected k; recalibrate only temperatures at100draws",
                "static_setting": "static (null k means infinity)", "reference_minimum": REFERENCE_MINIMUM,
                "adapted_standardized_channels": list(CHANNELS), "adapted_raw_variables": list(RAW_CHANNELS),
                "source_hashes": sources, "reference_hashes": reference_hashes,
                "selection": "minimum calibrated log loss on the original 4000 calibration rows; stable order breaks exact ties",
                "scope": "Exploratory previously inspected DEV; distribution adaptation, not causal mechanics or policy benefit",
                "raw_seasons": [2023, 2024, 2025], "raw_2026_access": False}
    output.mkdir(parents=True, exist_ok=False)
    dump(output/"config.json", manifest)
    for rel in sources:
        destination = output/"source"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, destination)
    shutil.copyfile(protocol, output/protocol.name)
    print("ADAPTATION_DIR="+str(output), flush=True)
    with (run/"encoders.pkl").open("rb") as stream:
        encoders = pickle.load(stream)
    if encoders["delivery"].draws != 25:
        raise ValueError("Primary adaptation protocol requires the original 25-draw pool")
    frame = prepare_frame(local)
    data_identity = frame.attrs["sequence_data_identity"]
    add_batter_style_history(frame)
    store = HistoryStore.from_frame(frame, normalizer=encoders["normalizer"])
    cohort = json.loads((run/"cohort_manifest.json").read_text())
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    samples = json.loads((run/"samples.json").read_text())
    actual = {name: rows_hash(part) for name, part in (("train", train), ("calibration", cal), ("dev", dev))}
    if actual != samples["rows_hash"]:
        raise ValueError("Frozen ordered sample identities changed")
    mixture = cal.sample(min(len(cal), cfg["delivery_calibration_rows"]), random_state=42).sort_index()
    train_all = frame[frame.split.eq("train") & eligible(frame)]
    reference = fit_reference(train_all, store.physical[train_all.index])
    counts, prior_mean = build_prior_physics(frame, store.physical)
    reference_mean, origins = reference_for_rows(frame, reference)
    with (output/"reference_means.pkl").open("wb") as stream:
        pickle.dump(reference, stream)
    data = {"data_identity": data_identity, "ordered_rows_hash": actual, "calibration_rows_hash": rows_hash(mixture),
            "reference": reference["report"], "reference_sha256": hash_file(output/"reference_means.pkl"),
            "dev_rows": len(dev), "dev_games": int(dev.game_pk.nunique()), "calibration_rows": len(mixture),
            "as_of_history": "all complete earlier game/pitcher/type physical deliveries across PAs, independent of eligibility",
            "dev_prior_count_quantiles": np.quantile(counts[dev.index], [0, .25, .5, .75, 1]).tolist(),
            "dev_zero_prior_rows": int((counts[dev.index] == 0).sum()),
            "dev_reference_origins": {name: int((origins[dev.index] == code).sum())
                                      for code, name in ((0, "global"), (1, "league_type"), (2, "pitcher_type"))}}
    dump(output/"data.json", data)
    selected_rows = np.r_[mixture.index.to_numpy(), dev.index.to_numpy()]
    np.savez_compressed(output/"as_of_inputs.npz", row_positions=selected_rows, pitch_keys=frame.iloc[selected_rows][KEY].to_numpy(),
                        counts=counts[selected_rows], prior_mean=prior_mean[selected_rows], reference_mean=reference_mean[selected_rows],
                        reference_origin=origins[selected_rows])
    if args.prepare_only:
        print("PREPARED; no model inference", flush=True)
        return
    model = SequenceModel.load(run/"all_transformer.pt")
    cy, dy = outcome_labels(mixture), outcome_labels(dev)
    selection = {"criterion": manifest["selection"], "rows": len(mixture), "settings": {}}
    for name, k in SETTINGS.items():
        shift = adaptive_shift(counts, prior_mean, reference_mean, k)
        logits = shifted_delivery_logits(model, store, encoders["context"], encoders["delivery"], mixture.index.to_numpy(), shift)
        selection["settings"][name] = calibrate_temperature(logits, cy)
        print("CALIBRATION", name, selection["settings"][name], flush=True)
    chosen = min(SETTINGS, key=lambda name: selection["settings"][name]["calibrated_log_loss"])
    selection["chosen_setting"] = chosen
    # Persist selection before evaluating any adapted DEV prediction.
    dump(output/"selection.json", selection)
    result = {"chosen_by_calibration": chosen, "settings": {}, "scope": manifest["scope"]}
    predictions = {}
    for name, k in SETTINGS.items():
        shift = adaptive_shift(counts, prior_mean, reference_mean, k)
        logits = shifted_delivery_logits(model, store, encoders["context"], encoders["delivery"], dev.index.to_numpy(), shift)
        p = softmax(logits/selection["settings"][name]["temperature"], axis=-1).mean(1)
        uncalibrated = softmax(logits, axis=-1).mean(1)
        predictions[name], predictions[name+"_uncalibrated"] = p, uncalibrated
        if name == "static":
            with np.load(run/"heldout_predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["pitch_keys"], dev[KEY].to_numpy())
                np.testing.assert_array_equal(saved["y"], dy)
                np.testing.assert_allclose(p, saved["transformer"], atol=1e-6, rtol=1e-5)
        result["settings"][name] = {"calibrated": classification_metrics(dy, p), "uncalibrated": classification_metrics(dy, uncalibrated),
                                    "paired_minus_static": paired(dy, p, predictions["static"], dev.game_pk, 2000),
                                    "shift_mean_abs_standardized": np.abs(shift[dev.index]).mean(axis=0).tolist(),
                                    "shift_max_abs_standardized": np.abs(shift[dev.index]).max(axis=0).tolist()}
        dump(output/"results.json", result)
        np.savez_compressed(output/"heldout_predictions.npz", y=dy, game_pk=dev.game_pk.to_numpy(), pitch_keys=dev[KEY].to_numpy(), **predictions)
        print("DEV", name, result["settings"][name]["calibrated"], result["settings"][name]["paired_minus_static"], flush=True)
    if args.check_draws:
        # Prespecified integration sensitivity: no second k search and no DEV selection.
        delivery100 = JointDelivery().fit(train_all, encoders["normalizer"], draws=100, seed=42)
        with (output/"delivery_draw100.pkl").open("wb") as stream:
            pickle.dump(delivery100, stream)
        sensitivity_names = ["static"] if chosen == "static" else ["static", chosen]
        sensitivity = {"draws": 100, "seed": 42, "selected_setting_from_primary25": chosen,
                       "k_retuned": False, "adaptive_contrast_available": chosen != "static",
                       "pool_report": delivery100.report, "pool_sha256": hash_file(output/"delivery_draw100.pkl"),
                       "calibration": {}, "settings": {}}
        for name in sensitivity_names:
            shift = adaptive_shift(counts, prior_mean, reference_mean, SETTINGS[name])
            logits = shifted_delivery_logits(model, store, encoders["context"], delivery100, mixture.index.to_numpy(), shift)
            sensitivity["calibration"][name] = calibrate_temperature(logits, cy)
            print("CALIBRATION100", name, sensitivity["calibration"][name], flush=True)
        dump(output/"selection_draw100.json", {key: value for key, value in sensitivity.items() if key != "settings"})
        sensitivity_predictions = {}
        for name in sensitivity_names:
            shift = adaptive_shift(counts, prior_mean, reference_mean, SETTINGS[name])
            logits = shifted_delivery_logits(model, store, encoders["context"], delivery100, dev.index.to_numpy(), shift)
            p = softmax(logits/sensitivity["calibration"][name]["temperature"], axis=-1).mean(1)
            uncalibrated = softmax(logits, axis=-1).mean(1)
            sensitivity_predictions[name], sensitivity_predictions[name+"_uncalibrated"] = p, uncalibrated
            sensitivity["settings"][name] = {"calibrated": classification_metrics(dy, p), "uncalibrated": classification_metrics(dy, uncalibrated),
                "paired_minus_static": paired(dy, p, sensitivity_predictions["static"], dev.game_pk, 2000)}
            dump(output/"sensitivity_draw100.json", sensitivity)
            np.savez_compressed(output/"sensitivity_draw100_predictions.npz", y=dy, game_pk=dev.game_pk.to_numpy(),
                                pitch_keys=dev[KEY].to_numpy(), **sensitivity_predictions)
            print("DEV100", name, sensitivity["settings"][name]["calibrated"],
                  sensitivity["settings"][name]["paired_minus_static"], flush=True)
    if any(hash_file(PROJECT/rel) != digest for rel, digest in sources.items()) or any(hash_file(Path(path)) != digest for path, digest in reference_hashes.items()):
        raise RuntimeError("Frozen sources or references changed during inference")
    dump(output/"runtime.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "selection_saved_before_dev_evaluation": True})
    print("COMPLETE "+str(output), flush=True)


if __name__ == "__main__":
    main()
