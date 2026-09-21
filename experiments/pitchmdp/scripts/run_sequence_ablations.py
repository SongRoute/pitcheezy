"""Exploratory matched context ablations after inspection of the original DEV.

Keep the full 28-dimensional input and identical parameter count; zero selected
context channels during fitting, calibration, conditional and integrated inference.
Original models, encoders and run artifacts remain immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
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
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_data import HistoryStore, prepare_frame
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_sequence_pilot import arrays, dump, paired


VARIANTS = {"no_game_context": tuple(range(2, 9)),
            "no_batter_style": tuple(range(11, 28)),
            "no_game_or_batter": tuple(range(2, 9)) + tuple(range(11, 28))}
CONTEXT_SCHEMA = ["balls", "strikes", "outs", "inning", "defensive_score_lead", "defender_is_home",
                  "first_base", "second_base", "third_base", "batter_left", "pitcher_left",
                  "six_batter_style_rates", "six_style_reliabilities", "five_soft_memberships"]
STYLE_NAMES = ("contact", "swing", "walk", "strikeout", "isolated_power", "groundball")
CONTEXT_CHANNELS = (CONTEXT_SCHEMA[:11] + [f"batter_style_{name}" for name in STYLE_NAMES] +
                    [f"batter_style_{name}_reliability" for name in STYLE_NAMES] +
                    [f"soft_membership_{j}" for j in range(5)])
EXPLORATORY = ("Post-DEV-inspection exploratory ablation; not an independent confirmatory test. "
               "Same one-seed fitted-model protocol; game-bootstrap intervals exclude training uncertainty.")


class MaskedContext:
    """Apply the same fixed channel intervention through every context call."""
    def __init__(self, encoder, variant):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown ablation: {variant}")
        if encoder.report()["features"] != CONTEXT_SCHEMA:
            raise ValueError("Context schema changed; refusing positional masking")
        self.encoder, self.variant = encoder, variant
        self.zero_indices = VARIANTS[variant]

    def transform(self, frame):
        values = self.encoder.transform(frame).copy()
        if values.ndim != 2 or values.shape[1] != 28:
            raise ValueError("Matched ablations require the original 28 context dimensions")
        values[:, self.zero_indices] = 0.
        return values


def rows_hash(frame):
    return hashlib.sha256(frame[KEY].to_numpy(np.int64).tobytes()).hexdigest()


def reconstruct_samples(frame, cfg, cohort):
    """Replicate original sampling, then validate each ordered row hash externally."""
    mask = eligible(frame)
    target = frame.pitcher.isin(cohort["pitcher_ids"]) & frame.pitcher.eq(frame.starter_pitcher)
    train_all = frame[frame.split.eq("train") & mask]
    train = pd.concat([train_all.sample(min(len(train_all), cfg["train_random_rows"]), random_state=cfg["seed"]),
                       train_all[target.reindex(train_all.index)]]).drop_duplicates(KEY).sort_index()
    cal_all = frame[frame.split.eq("calibration") & mask]
    cal = cal_all.sample(min(len(cal_all), cfg["calibration_rows"]), random_state=cfg["seed"]).sort_index()
    dev = frame[frame.split.eq("dev") & mask & target].copy()
    if not (train.game_date.max() < cal.game_date.min() < dev.game_date.min()):
        raise ValueError("Chronological splits are not disjoint")
    return train, cal, dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed original sequence run")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS),
                        default=["no_game_context", "no_batter_style"])
    parser.add_argument("--output", type=Path, help="Distinct SSD output directory; default inside original run")
    parser.add_argument("--resume", action="store_true", help="Reuse completed ablation checkpoints with identical identities")
    parser.add_argument("--prepare-only", action="store_true", help="Verify rows and save manifests; no fitting or model inference")
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("Variants must be unique")
    started = time.perf_counter()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.is_relative_to(volume.resolve()):
        raise SystemExit("Mounted configured SSD required")
    if Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Use existing configured Python")
    run = args.run.resolve()
    if not run.is_relative_to(root):
        raise SystemExit("Original run must be on the configured SSD")
    output = (args.output or run/datetime.now(timezone.utc).strftime("ablations-%Y%m%dT%H%M%SZ")).resolve()
    if not output.is_relative_to(root) or output == run or run.is_relative_to(output):
        raise SystemExit("Ablations need a distinct SSD subdirectory; cannot overwrite the original run")
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    cfg = json.loads((run/"config.json").read_text())
    samples = json.loads((run/"samples.json").read_text())
    cohort = json.loads((run/"cohort_manifest.json").read_text())
    original_hashes = json.loads((run/"source_hashes.json").read_text())
    runtime = json.loads((run/"runtime.json").read_text())
    if runtime["source_hashes_end"] != original_hashes:
        raise SystemExit("Original source hashes changed during its execution")
    for rel, digest in original_hashes.items():
        if hash_file(PROJECT/rel) != digest or hash_file(run/"source"/rel) != digest:
            raise SystemExit(f"Original source differs from saved run: {rel}")
    if cfg["seed"] != 42 or cfg["history_length"] != 5:
        raise SystemExit("This matched protocol requires seed 42 and five previous pitches")
    references = ["config.json", "samples.json", "cohort_manifest.json", "encoders.pkl",
                  "all_count_results.json", "heldout_predictions.npz", "all_transformer.pt"]
    reference_hashes = {name: hash_file(run/name) for name in references}
    source_hashes = {**original_hashes,
                     str(Path(__file__).resolve().relative_to(PROJECT)): hash_file(Path(__file__))}
    manifest = {"base_run": str(run), "base_config": cfg, "variants": args.variants,
                "zero_context_indices": {key: list(VARIANTS[key]) for key in args.variants},
                "retained_for_every_variant": ["balls", "strikes", "batter_left", "pitcher_left", "physical_sequence"],
                "context_dimension": 28, "same_architecture_and_parameter_count": True,
                "source_hashes": source_hashes, "reference_artifact_hashes": reference_hashes,
                "bootstrap_replicates": 2000, "exploratory_status": EXPLORATORY,
                "comparison_direction": "ablation minus full Transformer; positive log loss means full features helped"}
    if args.resume:
        if json.loads((output/"config.json").read_text()) != manifest:
            raise SystemExit("Ablation source/config/reference identity changed; start a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        dump(output/"config.json", manifest)
        for rel in source_hashes:
            destination = output/"source"/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/rel, destination)
    print("ABLATION_DIR="+str(output), flush=True)
    dump(output/"context_masks.json", {"channels": CONTEXT_CHANNELS, "schema": CONTEXT_SCHEMA,
                                       "variants": {key: {"zero_indices": list(VARIANTS[key]),
                                                          "zero_channels": [CONTEXT_CHANNELS[j] for j in VARIANTS[key]],
                                                          "keep_mask": [j not in VARIANTS[key] for j in range(28)]}
                                                    for key in args.variants}})

    with (run/"encoders.pkl").open("rb") as stream:
        encoders = pickle.load(stream)
    context, delivery, normalizer = (encoders[key] for key in ("context", "delivery", "normalizer"))
    frame = prepare_frame(local)
    identity = frame.attrs["sequence_data_identity"]
    add_batter_style_history(frame)
    store = HistoryStore.from_frame(frame, normalizer=normalizer)
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("Store frame must retain contiguous original row positions")
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    actual_hashes = {name: rows_hash(part) for name, part in [("train", train), ("calibration", cal), ("dev", dev)]}
    if actual_hashes != samples["rows_hash"]:
        raise ValueError("Reconstructed ordered samples differ from original run")
    mixture_cal = cal.sample(min(len(cal), cfg["delivery_calibration_rows"]), random_state=cfg["seed"]).sort_index()
    labels, cy, dy = (outcome_labels(part) for part in (train, cal, dev))
    with np.load(run/"heldout_predictions.npz", allow_pickle=False) as saved:
        for key, values in [("pitch_keys", dev[KEY].to_numpy()), ("game_pk", dev.game_pk.to_numpy()), ("y", dy)]:
            if not np.array_equal(saved[key], values):
                raise ValueError(f"Saved reference predictions disagree with reconstructed {key}")
        full_p, base_p = saved["transformer"].copy(), saved["count_hand"].copy()
    base_results = json.loads((run/"all_count_results.json").read_text())
    for predicted, reported in [(full_p, base_results["transformer"]["primary_delivery_integrated"]),
                                (base_p, base_results["count_hand_baseline"])]:
        measured = classification_metrics(dy, predicted)
        for metric in ("log_loss", "brier_multiclass"):
            if not np.isclose(measured[metric], reported[metric], atol=1e-10, rtol=1e-10):
                raise ValueError(f"Saved reference metrics disagree with predictions: {metric}")
    data_report = {"data_identity": identity, "rows_hash": actual_hashes,
                   "delivery_calibration_rows_hash": rows_hash(mixture_cal),
                   "train_used": len(train), "calibration_used": len(cal), "dev_rows": len(dev),
                   "dev_games": int(dev.game_pk.nunique()), "normalizer": normalizer.report(),
                   "context": context.report(), "delivery": delivery.report,
                   "reference_parameter_count": base_results["transformer"]["training"]["parameter_count"]}
    if (output/"data.json").exists() and json.loads((output/"data.json").read_text()) != json.loads(json.dumps(data_report)):
        raise ValueError("Ablation data identity changed")
    dump(output/"data.json", data_report)
    for variant in args.variants:
        masked = MaskedContext(context, variant)
        original = context.transform(dev.iloc[:4])
        changed = masked.transform(dev.iloc[:4])
        keep = [j for j in range(28) if j not in VARIANTS[variant]]
        if not (np.array_equal(changed[:, keep], original[:, keep]) and np.all(changed[:, VARIANTS[variant]] == 0)):
            raise ValueError("Ablation mask changed unselected context channels")
    if args.prepare_only:
        dump(output/"preparation.json", {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "seconds": time.perf_counter()-started, "training_started": False})
        print("PREPARED; no training or model inference: "+str(output), flush=True)
        return

    train_arrays, cal_arrays, dev_arrays = (arrays(store, context, part.index.to_numpy()) for part in (train, cal, dev))
    results = {"exploratory_status": EXPLORATORY, "full_transformer": classification_metrics(dy, full_p),
               "count_hand_baseline": classification_metrics(dy, base_p), "variants": {}}
    predictions = {"full_transformer": full_p, "count_hand_baseline": base_p}
    for variant in args.variants:
        masked = MaskedContext(context, variant)
        variant_arrays = []
        for original_arrays in (train_arrays, cal_arrays, dev_arrays):
            x = original_arrays[2].copy()
            x[:, VARIANTS[variant]] = 0.
            variant_arrays.append((original_arrays[0], original_arrays[1], x))
        checkpoint = output/f"{variant}.pt"
        if checkpoint.exists():
            model = SequenceModel.load(checkpoint)
        else:
            model = SequenceModel("transformer", cfg["seed"], cfg["width"], n_classes=10).fit(
                variant_arrays[0], labels, variant_arrays[1], cy, epochs=cfg["epochs"], patience=cfg["patience"],
                batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"],
                checkpoint=output/f"{variant}_best_training.pt")
            delivery.calibrate(model, store, masked, mixture_cal.index.to_numpy(), outcome_labels(mixture_cal))
            model.save(checkpoint)
        if model.report["parameter_count"] != data_report["reference_parameter_count"]:
            raise ValueError("Ablation parameter count differs from the full Transformer")
        conditional_logits = model.logits(variant_arrays[2])
        conditional = softmax(conditional_logits/model.temperature, axis=-1)
        conditional_uncalibrated = softmax(conditional_logits, axis=-1)
        integrated_logits, _ = delivery.logits(model, store, masked, dev.index.to_numpy())
        p = softmax(integrated_logits/getattr(model, "delivery_temperature", model.temperature), axis=-1).mean(1)
        integrated_uncalibrated = softmax(integrated_logits, axis=-1).mean(1)
        predictions[variant] = p
        predictions[variant+"_uncalibrated"] = integrated_uncalibrated
        predictions[variant+"_conditional_calibrated"] = conditional
        predictions[variant+"_conditional_uncalibrated"] = conditional_uncalibrated
        results["variants"][variant] = {
            "zero_context_indices": list(VARIANTS[variant]), "training": model.report,
            "primary_delivery_integrated": classification_metrics(dy, p),
            "delivery_integrated_uncalibrated": classification_metrics(dy, integrated_uncalibrated),
            "conditional_current_physics_diagnostic": classification_metrics(dy, conditional),
            "conditional_current_physics_uncalibrated": classification_metrics(dy, conditional_uncalibrated),
            "paired_ablation_minus_full": paired(dy, p, full_p, dev.game_pk, 2000),
            "paired_ablation_minus_baseline": paired(dy, p, base_p, dev.game_pk, 2000),
            "per_pitcher": {str(int(pid)): classification_metrics(dy[dev.pitcher.to_numpy() == pid],
                                                                  p[dev.pitcher.to_numpy() == pid])
                            for pid in cohort["pitcher_ids"]}}
        dump(output/"ablation_results.json", results)
        np.savez_compressed(output/"heldout_predictions.npz", y=dy, game_pk=dev.game_pk.to_numpy(),
                            pitch_keys=dev[KEY].to_numpy(), **predictions)
        print("ABLATION", variant, results["variants"][variant]["primary_delivery_integrated"],
              results["variants"][variant]["paired_ablation_minus_full"], flush=True)
        del model, variant_arrays
        gc.collect()
        # The next variant starts after the previous fit/inference has completed.
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    end_hashes = {rel: hash_file(PROJECT/rel) for rel in source_hashes}
    if end_hashes != source_hashes or any(hash_file(run/name) != digest for name, digest in reference_hashes.items()):
        raise RuntimeError("Source or original artifacts changed during ablation execution")
    dump(output/"runtime.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
                                "seconds": time.perf_counter()-started, "source_hashes_end": end_hashes,
                                "exploratory_status": EXPLORATORY})
    print("COMPLETE "+str(output), flush=True)


if __name__ == "__main__":
    main()
