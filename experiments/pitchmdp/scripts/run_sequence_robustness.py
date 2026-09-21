"""Fixed-sample multi-seed replication and incremental cluster-feature experiment.

All fits are sequential. Samples, normalizer, batter encoder and delivery pools
retain seed 42; only network initialization/minibatch/dropout seed varies. This
reuses an inspected DEV and is an exploratory robustness study, not confirmation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy.special import softmax

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_data import HistoryStore, prepare_frame
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_sequence_pilot import arrays, dump
from run_sequence_ablations import CONTEXT_CHANNELS, CONTEXT_SCHEMA, reconstruct_samples, rows_hash


SEEDS = (42, 43, 44, 45, 46)
MASKS = {"full_transformer": (), "flatten_mlp": (), "no_game_context": tuple(range(2, 9)),
         "no_batter_style": tuple(range(11, 28)), "capacity_mlp": (), "no_clusters": tuple(range(23, 28))}
REPLICATION = ("full_transformer", "flatten_mlp", "no_game_context", "no_batter_style")
REGIMES = ("delivery_integrated_calibrated", "delivery_integrated_uncalibrated",
           "conditional_calibrated", "conditional_uncalibrated")
EXPLORATORY = ("Exploratory robustness on previously inspected 2025 DEV; not unseen confirmation. "
               "Five fixed model seeds with fixed samples, encoders and delivery pools; no DEV model selection.")


class FixedMaskedContext:
    def __init__(self, encoder, variant):
        if encoder.report()["features"] != CONTEXT_SCHEMA or variant not in MASKS:
            raise ValueError("Unknown context schema or variant")
        self.encoder, self.variant = encoder, variant

    def transform(self, frame):
        values = self.encoder.transform(frame).copy()
        if values.ndim != 2 or values.shape[1] != 28:
            raise ValueError("Expected unchanged 28-dimensional context")
        if MASKS[self.variant]:
            values[:, MASKS[self.variant]] = 0.
        return values


def per_pitch_scores(y, probabilities):
    """Score each seed separately; never average probabilities before scoring."""
    labels = np.asarray(y)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Labels must be a finite one-dimensional integer vector")
    y = labels.astype(np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 2:
        p = p[None]
    if p.ndim != 3 or p.shape[1] != len(y) or not len(y) or not p.shape[0] or p.shape[-1] < 2:
        raise ValueError("Expected nonempty [seed,pitch,class] predictions aligned with labels")
    if (not np.isfinite(p).all() or (p < 0).any() or
            not np.allclose(p.sum(-1), 1., atol=1e-5) or (y < 0).any() or (y >= p.shape[-1]).any()):
        raise ValueError("Invalid probabilities or labels")
    return {"log_loss": -np.log(np.clip(p[:, np.arange(len(y)), y], 1e-12, 1.)),
            "brier_multiclass": ((p-np.eye(p.shape[-1])[y])**2).sum(-1)}


def aggregate_comparison(y, probabilities, references, game_ids, seeds, replicates=2000, bootstrap_seed=42):
    """Paired pitch-weighted differences with game-only and crossed seed/game CIs.

    Each draw samples G whole games and S matched seeds independently. The same
    game multiset is used for every selected seed and both models. Thus games
    are never treated as S*G independent units. Conditional on these samples and
    this training procedure; five seeds cannot characterize all model uncertainty.
    """
    model, reference = per_pitch_scores(y, probabilities), per_pitch_scores(y, references)
    game_ids = np.asarray(game_ids)
    seeds = list(seeds)
    if model["log_loss"].shape != reference["log_loss"].shape or len(seeds) != model["log_loss"].shape[0]:
        raise ValueError("Model/reference seeds and prediction dimensions must match")
    if len(set(seeds)) != len(seeds) or game_ids.shape != (len(y),) or pd.isna(game_ids).any() or replicates < 2:
        raise ValueError("Expected unique matched seeds, aligned game IDs and at least two resamples")
    games, inverse = np.unique(game_ids, return_inverse=True)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(bootstrap_seed)
    game_draws = rng.integers(0, len(games), size=(replicates, len(games)))
    seed_draws = rng.integers(0, len(seeds), size=(replicates, len(seeds)))
    denominators = counts[game_draws].sum(axis=1)
    result = {"seeds": seeds, "n": len(y), "games": len(games), "replicates": replicates,
              "bootstrap_seed": bootstrap_seed, "direction": "model minus reference; negative favors model",
              "estimand": "mean of per-seed per-pitch losses; not probability ensemble loss",
              "bootstrap": "whole games shared across models/seeds; independently resampled matched seed IDs"}
    for metric in model:
        difference = model[metric]-reference[metric]
        sums = np.array([np.bincount(inverse, weights=row, minlength=len(games)) for row in difference])
        # [seed, replicate]: every retained seed sees the same sampled games.
        seed_game_means = sums[:, game_draws].sum(axis=-1)/denominators[None]
        game_only = seed_game_means.mean(axis=0)
        crossed = seed_game_means[seed_draws, np.arange(replicates)[:, None]].mean(axis=1)
        result[metric] = {"mean_model": float(model[metric].mean()), "mean_reference": float(reference[metric].mean()),
                          "model_minus_reference": float(difference.mean()),
                          "per_seed_difference": {str(seed): float(value) for seed, value in zip(seeds, difference.mean(axis=1))},
                          "game_only_bootstrap95": np.quantile(game_only, [.025, .975]).tolist(),
                          "crossed_seed_game_bootstrap95": np.quantile(crossed, [.025, .975]).tolist()}
    return result


def _verified_predictions(path, y, keys, games):
    with np.load(path, allow_pickle=False) as saved:
        for key, value in (("y", y), ("pitch_keys", keys), ("game_pk", games)):
            if not np.array_equal(saved[key], value):
                raise ValueError(f"Prediction identity mismatch: {path.name}/{key}")
        return {key: saved[key].copy() for key in saved.files if key not in ("y", "pitch_keys", "game_pk", "game_month")}


def _save_model_result(directory, model_path, probabilities, training, origin, y, keys, games, months, seed, variant):
    directory.mkdir(parents=True, exist_ok=True)
    if model_path != directory/"model.pt":
        shutil.copyfile(model_path, directory/"model.pt")
    metrics = {name: classification_metrics(y, probabilities[name]) for name in REGIMES}
    monthly = {month: {"n": int((months == month).sum()), "games": int(len(np.unique(games[months == month]))),
                       "metrics": {name: classification_metrics(y[months == month], probabilities[name][months == month])
                                   for name in REGIMES}} for month in sorted(np.unique(months))}
    temporary = directory/"predictions.tmp.npz"
    np.savez_compressed(temporary, y=y, pitch_keys=keys, game_pk=games, game_month=months, **probabilities)
    temporary.replace(directory/"predictions.npz")
    result = {"seed": seed, "variant": variant, "training": training, "origin": origin,
              "checkpoint_path": str((directory/"model.pt").resolve()), "predictions_file": str((directory/"predictions.npz").resolve()),
              "zero_context_indices": list(MASKS[variant]), "metrics": metrics, "monthly": monthly,
              "exploratory_status": EXPLORATORY,
              "artifact_hashes": {name: hash_file(directory/name) for name in ("model.pt", "predictions.npz")}}
    dump(directory/"result.json", result)
    return result


def _completed(directory, seed, variant):
    if not (directory/"result.json").exists():
        return None
    result = json.loads((directory/"result.json").read_text())
    if result["seed"] != seed or result["variant"] != variant:
        raise ValueError("Completed model identity mismatch")
    if any(hash_file(directory/name) != digest for name, digest in result["artifact_hashes"].items()):
        raise ValueError("Completed model artifacts changed")
    return result


def update_summary(output, y, keys, games, months):
    """Refresh descriptive/monthly summaries and paired seed/game comparisons."""
    records, predictions, per_seed = {}, {}, {}
    for variant in MASKS:
        for seed in SEEDS:
            directory = output/"models"/f"seed{seed}"/variant
            result = _completed(directory, seed, variant)
            if result is None:
                continue
            records.setdefault(variant, {})[seed] = result
            probabilities = _verified_predictions(directory/"predictions.npz", y, keys, games)
            predictions[(variant, seed)] = probabilities["delivery_integrated_calibrated"]
    summary = {"exploratory_status": EXPLORATORY, "expected_seeds": list(SEEDS),
               "primary_estimand": "equal-seed average of single-model per-pitch LL/Brier, not an ensemble",
               "n": len(y), "games": len(np.unique(games)), "variants": {}, "comparisons": {},
               "monthly_role": "descriptive stability check only; no monthly or overall DEV winner selection"}
    for variant, seed_records in records.items():
        selected = sorted(seed_records)
        summary["variants"][variant] = {"completed_seeds": selected, "complete": selected == list(SEEDS), "metrics": {}, "monthly": {}}
        target = summary["variants"][variant]
        for regime in REGIMES:
            target["metrics"][regime] = {metric: {"mean": float(np.mean(values)),
                "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "per_seed": {str(seed): float(value) for seed, value in zip(selected, values)}}
                for metric in ("log_loss", "brier_multiclass")
                for values in [[seed_records[seed]["metrics"][regime][metric] for seed in selected]]}
        for month in sorted(np.unique(months)):
            target["monthly"][month] = {"n": int((months == month).sum()), "games": len(np.unique(games[months == month])),
                "metrics": {regime: {metric: float(np.mean([seed_records[seed]["monthly"][month]["metrics"][regime][metric]
                                                            for seed in selected])) for metric in ("log_loss", "brier_multiclass")}
                            for regime in REGIMES}}
    contrasts = [("full_transformer", "flatten_mlp"), ("no_game_context", "full_transformer"),
                 ("no_batter_style", "full_transformer"), ("full_transformer", "capacity_mlp"), ("no_clusters", "full_transformer")]
    for model, reference in contrasts:
        matched = [seed for seed in SEEDS if (model, seed) in predictions and (reference, seed) in predictions]
        if not matched:
            continue
        name = model+"_minus_"+reference
        p = np.stack([predictions[(model, seed)] for seed in matched])
        q = np.stack([predictions[(reference, seed)] for seed in matched])
        contrast = aggregate_comparison(y, p, q, games, matched)
        contrast["complete"] = matched == list(SEEDS)
        contrast["monthly"] = {month: aggregate_comparison(y[months == month], p[:, months == month], q[:, months == month],
                                                           games[months == month], matched)
                               for month in sorted(np.unique(months))}
        summary["comparisons"][name] = contrast
        for i, seed in enumerate(matched):
            per_seed.setdefault(str(seed), {})[name] = aggregate_comparison(y, p[i], q[i], games, [seed])
    dump(output/"summary.json", summary)
    dump(output/"per_seed_results.json", per_seed)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True, help="Completed original seed-42 ablations")
    parser.add_argument("--phase", choices=["replication", "capacity", "clusters", "all"], default="replication")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze and validate without training or model inference")
    args = parser.parse_args()
    start = time.perf_counter()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.is_relative_to(volume.resolve()):
        raise SystemExit("Mounted configured SSD required")
    if Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Use existing configured Python; no installation")
    run, ablations = args.run.resolve(), args.ablations.resolve()
    if not all(path.is_relative_to(root) for path in (run, ablations)):
        raise SystemExit("Input runs must be on the configured SSD")
    output = (args.output or run/datetime.now(timezone.utc).strftime("robustness-%Y%m%dT%H%M%SZ")).resolve()
    if not output.is_relative_to(root) or any(output == path or path.is_relative_to(output) for path in (run, ablations)):
        raise SystemExit("Use a distinct output subdirectory on the configured SSD")
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    cfg = json.loads((run/"config.json").read_text())
    if cfg["seed"] != 42 or cfg["width"] != 128 or cfg["history_length"] != 5:
        raise SystemExit("Original sample/delivery seed42, width128 and past-five history required")
    sources = json.loads((run/"source_hashes.json").read_text())
    ab_cfg = json.loads((ablations/"config.json").read_text())
    if json.loads((run/"runtime.json").read_text())["source_hashes_end"] != sources:
        raise ValueError("Original run source identity changed during fitting")
    if json.loads((ablations/"runtime.json").read_text())["source_hashes_end"] != ab_cfg["source_hashes"]:
        raise ValueError("Seed42 ablation source identity changed during fitting")
    for origin, expected in ((run, sources), (ablations, ab_cfg["source_hashes"])):
        for rel, digest in expected.items():
            if hash_file(PROJECT/rel) != digest or hash_file(origin/"source"/rel) != digest:
                raise ValueError(f"Frozen source changed: {rel}")
    sources = {**sources, **ab_cfg["source_hashes"], str(Path(__file__).resolve().relative_to(PROJECT)): hash_file(Path(__file__))}
    protocol = PROJECT/"docs/NEXT_EXPERIMENTS_PROTOCOL.md"
    sources[str(protocol.relative_to(PROJECT))] = hash_file(protocol)
    reference_paths = [run/name for name in ("config.json", "samples.json", "cohort_manifest.json", "encoders.pkl",
                       "heldout_predictions.npz", "all_count_results.json", "all_transformer.pt", "all_flatten_mlp.pt",
                       "audit_supplemental/all_count_calibration_predictions.npz", "audit_supplemental/all_count_calibration_replay.json",
                       "audit_supplemental/data_quality.json")]
    reference_paths += [ablations/name for name in ("config.json", "data.json", "heldout_predictions.npz", "ablation_results.json",
                                                   "no_game_context.pt", "no_batter_style.pt")]
    reference_hashes = {str(path): hash_file(path) for path in reference_paths}
    manifest = {"protocol_version": 1, "base_run": str(run), "seed42_ablations": str(ablations), "base_config": cfg,
                "model_seeds": list(SEEDS), "sample_encoder_delivery_seed": 42,
                "variants": {key: {"architecture": "flatten_mlp" if key in ("flatten_mlp", "capacity_mlp") else "transformer",
                                    "width": 234 if key == "capacity_mlp" else cfg["width"],
                                    "zero_context_indices": list(mask), "zero_context_channels": [CONTEXT_CHANNELS[j] for j in mask]}
                             for key, mask in MASKS.items()}, "context_channels": CONTEXT_CHANNELS,
                "source_hashes": sources, "reference_hashes": reference_hashes, "bootstrap_replicates": 2000,
                "bootstrap_seed": 42, "exploratory_status": EXPLORATORY,
                "fixed_estimator": "mean of per-seed per-pitch losses; no probability ensembling",
                "allowed_raw_seasons": [2023, 2024, 2025], "raw_2026_access": False,
                "phase_policy": "replication then capacity then clusters; phase may change on resume without changing frozen experiment"}
    if args.resume:
        if json.loads((output/"config.json").read_text()) != manifest:
            raise ValueError("Resume source/config/reference identity changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        dump(output/"config.json", manifest)
        for rel in sources:
            destination = output/"source"/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/rel, destination)
        shutil.copyfile(protocol, output/"NEXT_EXPERIMENTS_PROTOCOL.md")
    lock = (output/"runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("Another phase is running in this output directory") from error
    print("ROBUSTNESS_DIR="+str(output), flush=True)

    samples = json.loads((run/"samples.json").read_text())
    cohort = json.loads((run/"cohort_manifest.json").read_text())
    ab_data = json.loads((ablations/"data.json").read_text())
    with (run/"encoders.pkl").open("rb") as stream:
        encoders = pickle.load(stream)
    context, delivery, normalizer = (encoders[key] for key in ("context", "delivery", "normalizer"))
    frame = prepare_frame(local)
    identity = frame.attrs["sequence_data_identity"]
    if identity != ab_data["data_identity"]:
        raise ValueError("Dataset identity differs from seed42 ablations")
    add_batter_style_history(frame)
    store = HistoryStore.from_frame(frame, normalizer=normalizer)
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("Original full-frame row positions must be preserved")
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    ordered_hashes = {name: rows_hash(part) for name, part in (("train", train), ("calibration", cal), ("dev", dev))}
    if ordered_hashes != samples["rows_hash"] or ordered_hashes != ab_data["rows_hash"]:
        raise ValueError("Ordered samples differ from the frozen original")
    mixture = cal.sample(min(len(cal), cfg["delivery_calibration_rows"]), random_state=42).sort_index()
    if rows_hash(mixture) != ab_data["delivery_calibration_rows_hash"]:
        raise ValueError("Delivery calibration sample changed")
    y, keys, games = outcome_labels(dev), dev[KEY].to_numpy(), dev.game_pk.to_numpy()
    months = pd.to_datetime(dev.game_date).dt.strftime("%Y-%m").to_numpy(dtype="U7")
    data_manifest = {"data_identity": identity, "ordered_rows_hash": ordered_hashes,
                     "delivery_calibration_rows_hash": rows_hash(mixture), "train_rows": len(train), "calibration_rows": len(cal),
                     "dev_rows": len(dev), "dev_games": len(np.unique(games)), "dev_dates": samples["dev_dates"],
                     "monthly_counts": {month: {"pitches": int((months == month).sum()), "games": len(np.unique(games[months == month]))}
                                        for month in sorted(np.unique(months))},
                     "normalizer": normalizer.report(), "context": context.report(), "delivery": delivery.report}
    normalized_manifest = json.loads(json.dumps(data_manifest))
    if (output/"data.json").exists() and json.loads((output/"data.json").read_text()) != normalized_manifest:
        raise ValueError("Resume data identity changed")
    dump(output/"data.json", data_manifest)
    base_predictions = _verified_predictions(run/"heldout_predictions.npz", y, keys, games)
    supplemental = _verified_predictions(run/"audit_supplemental/all_count_calibration_predictions.npz", y, keys, games)
    ab_predictions = _verified_predictions(ablations/"heldout_predictions.npz", y, keys, games)
    base_results = json.loads((run/"all_count_results.json").read_text())
    ab_results = json.loads((ablations/"ablation_results.json").read_text())["variants"]
    for variant in REPLICATION:
        directory = output/"models"/"seed42"/variant
        if _completed(directory, 42, variant):
            continue
        if variant in ("full_transformer", "flatten_mlp"):
            kind = "transformer" if variant == "full_transformer" else variant
            probabilities = {regime: supplemental[kind+"__"+regime] for regime in REGIMES}
            np.testing.assert_allclose(probabilities["delivery_integrated_calibrated"], base_predictions[kind], rtol=1e-5, atol=1e-6)
            probabilities["delivery_integrated_calibrated"] = base_predictions[kind]
            model_path, old = run/f"all_{kind}.pt", base_results[kind]
        else:
            probabilities = dict(zip(REGIMES, [ab_predictions[variant], ab_predictions[variant+"_uncalibrated"],
                                                ab_predictions[variant+"_conditional_calibrated"], ab_predictions[variant+"_conditional_uncalibrated"]]))
            model_path, old = ablations/f"{variant}.pt", ab_results[variant]
        measured = classification_metrics(y, probabilities["delivery_integrated_calibrated"])
        for metric in ("log_loss", "brier_multiclass"):
            if not np.isclose(measured[metric], old["primary_delivery_integrated"][metric], rtol=1e-10, atol=1e-10):
                raise ValueError("Seed42 reference metric differs from archived predictions")
        _save_model_result(directory, model_path, probabilities, old["training"],
                           {"mode": "reused_without_fit_or_inference", "source_checkpoint": str(model_path),
                            "source_sha256": hash_file(model_path)}, y, keys, games, months, 42, variant)
    # Config/source/data/ordered samples and all seed42 references are frozen before the first fit.
    update_summary(output, y, keys, games, months)
    if args.prepare_only:
        dump(output/"preparation.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "seconds": time.perf_counter()-start, "no_training_or_model_inference": True})
        print("PREPARED "+str(output), flush=True)
        return
    if args.phase in ("clusters", "capacity") and any(not _completed(output/"models"/f"seed{seed}"/"full_transformer", seed, "full_transformer")
                                         for seed in SEEDS):
        raise ValueError("Run replication first in this output directory: follow-up phases need all five full-Transformer references")
    train_arrays, cal_arrays, dev_arrays = (arrays(store, context, part.index.to_numpy()) for part in (train, cal, dev))
    labels, cy, mixture_y = outcome_labels(train), outcome_labels(cal), outcome_labels(mixture)
    jobs = []
    if args.phase in ("replication", "all"):
        jobs += [(seed, variant) for seed in SEEDS[1:] for variant in REPLICATION]
    if args.phase in ("capacity", "all"):
        jobs += [(seed, "capacity_mlp") for seed in SEEDS]
    if args.phase in ("clusters", "all"):
        jobs += [(seed, "no_clusters") for seed in SEEDS]
    for seed, variant in jobs:
        directory = output/"models"/f"seed{seed}"/variant
        if _completed(directory, seed, variant):
            print(f"REUSE completed seed={seed} variant={variant}", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        variant_arrays = []
        for original in (train_arrays, cal_arrays, dev_arrays):
            x = original[2].copy()
            if MASKS[variant]:
                x[:, MASKS[variant]] = 0.
            variant_arrays.append((original[0], original[1], x))
        kind = "flatten_mlp" if variant in ("flatten_mlp", "capacity_mlp") else "transformer"
        width = 234 if variant == "capacity_mlp" else cfg["width"]
        masked = FixedMaskedContext(context, variant)
        model_path = directory/"model.pt"
        print(f"MODEL seed={seed} variant={variant}", flush=True)
        # A fit completed before interruption may be reused; partial optimizer checkpoints are diagnostic only.
        if model_path.exists():
            model = SequenceModel.load(model_path)
            metadata = json.loads((directory/"checkpoint_identity.json").read_text())
            if metadata != {"seed": seed, "variant": variant, "sha256": hash_file(model_path)}:
                raise ValueError("Incomplete result checkpoint identity changed")
        else:
            model = SequenceModel(kind, seed, width, n_classes=10).fit(
                variant_arrays[0], labels, variant_arrays[1], cy, epochs=cfg["epochs"], patience=cfg["patience"],
                batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"], checkpoint=directory/"best_training.pt")
            delivery.calibrate(model, store, masked, mixture.index.to_numpy(), mixture_y)
            model.save(model_path)
            dump(directory/"checkpoint_identity.json", {"seed": seed, "variant": variant, "sha256": hash_file(model_path)})
        expected_count = 278874 if variant == "capacity_mlp" else base_results[kind]["training"]["parameter_count"]
        if (model.seed != seed or model.kind != kind or model.report["parameter_count"] != expected_count or
                model.report["network"]["n_context"] != 28 or model.report["network"]["width"] != width):
            raise ValueError("Model architecture, seed or parameter count differs from protocol")
        conditional_logits = model.logits(variant_arrays[2])
        delivery_logits, _ = delivery.logits(model, store, masked, dev.index.to_numpy())
        probabilities = {"conditional_calibrated": softmax(conditional_logits/model.temperature, axis=-1),
                         "conditional_uncalibrated": softmax(conditional_logits, axis=-1),
                         "delivery_integrated_calibrated": softmax(delivery_logits/model.delivery_temperature, axis=-1).mean(1),
                         "delivery_integrated_uncalibrated": softmax(delivery_logits, axis=-1).mean(1)}
        result = _save_model_result(directory, model_path, probabilities, model.report, {"mode": "new_fit"},
                                    y, keys, games, months, seed, variant)
        print("RESULT", seed, variant, result["metrics"]["delivery_integrated_calibrated"], flush=True)
        del model, variant_arrays, conditional_logits, delivery_logits
        gc.collect()
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        update_summary(output, y, keys, games, months)
    end_hashes = {rel: hash_file(PROJECT/rel) for rel in sources}
    if end_hashes != sources or any(hash_file(Path(path)) != digest for path, digest in reference_hashes.items()):
        raise RuntimeError("Frozen sources or reference artifacts changed during execution")
    dump(output/f"runtime_{args.phase}.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
                                             "seconds": time.perf_counter()-start, "source_hashes_end": end_hashes})
    print("COMPLETE "+args.phase+" "+str(output), flush=True)


if __name__ == "__main__":
    main()
