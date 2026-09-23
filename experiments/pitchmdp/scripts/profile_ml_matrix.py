"""Bounded, explicit ML matrix hardware profile on TRAIN and early-stop only.

Requires a verified preparation.  Fits two disposable 2-epoch models on the
first 8,192 registered LEGACY TRAIN pitches and 2,048 early-stop pitches,
then times 400-draw integration on 256 early-stop pitches.  No DEV rows,
scores, model selection, or checkpoints are written by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import resource
import signal
import sys
import time
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash, load_verified_processed_cache, ordered_key_hash, validate_config
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_data import HistoryStore
from pitchmdp.sequence_model import SequenceModel
from run_ml_matrix import (_select_by_keys, atomic_json, check_location, heavy_lock,
                           manifest_identity, read_json, verify_prepare)
from run_sequence_pilot import arrays
from run_temporal_blend import assign_fold


TRAIN_LIMIT = 8192
EARLY_LIMIT = 2048
INTEGRATION_LIMIT = 256
CPU_FORWARD_LIMIT = 32
PROFILE_EPOCHS = 2


def profile(config: dict, local: dict, output: Path, prep: dict, max_seconds: int) -> dict:
    start = time.perf_counter()
    raw = load_verified_processed_cache(local)
    if raw.attrs["sequence_data_identity"] != prep["dataset_identity"] or raw.attrs["matrix_source_provenance"] != prep["source_provenance"]:
        raise ValueError("Derived data identity changed since preparation")
    # The verified source reader checks full derived-file bytes. Keep all later
    # dates out of the actual profile frame before history/features are built.
    prior = raw.loc[pd.to_datetime(raw.game_date).le("2025-05-15")].copy().reset_index(drop=True)
    del raw
    frame = assign_fold(add_batter_style_history(prior), 2025)
    records = prep["scope"]["legacy"]["samples"]
    train = _select_by_keys(frame, output, records["train"]).iloc[:TRAIN_LIMIT]
    early = _select_by_keys(frame, output, records["earlystop"]).iloc[:EARLY_LIMIT]
    if len(train) != TRAIN_LIMIT or len(early) != EARLY_LIMIT:
        raise ValueError("Profile requires enough registered TRAIN and early-stop rows")
    with (output / "legacy_aux.pkl").open("rb") as stream:
        aux = pickle.load(stream)
    store = HistoryStore.from_frame(frame, normalizer=aux["normalizer"])
    context = aux["context"]
    train_arrays = arrays(store, context, train.index.to_numpy())
    early_arrays = arrays(store, context, early.index.to_numpy())
    labels, early_labels = outcome_labels(train), outcome_labels(early)
    if (labels < 0).any() or (early_labels < 0).any():
        raise ValueError("Profile sample includes an ineligible outcome")
    result = {"profile_source_sha256": hash_file(Path(__file__)),
              "preparation_sha256": hash_file(output / "preparation.json"),
              "config_sha256": canonical_hash(config), "train_rows_sha256": ordered_key_hash(train),
              "earlystop_rows_sha256": ordered_key_hash(early),
              "hardware": {"torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
                           "python": str(Path(sys.executable).resolve())},
              "limits": {"train": TRAIN_LIMIT, "earlystop": EARLY_LIMIT,
                         "integrated_inference": INTEGRATION_LIMIT, "cpu_forward": CPU_FORWARD_LIMIT,
                         "epochs": PROFILE_EPOCHS, "max_seconds": max_seconds},
              "models": {}}
    for kind in ("flatten_mlp", "transformer"):
        if time.perf_counter() - start >= max_seconds:
            raise TimeoutError("Profile wall budget reached before next architecture")
        model = SequenceModel(kind, seed=42, width=config["width"])
        began_fit = time.perf_counter()
        model.fit(train_arrays, labels, early_arrays, early_labels,
                  epochs=PROFILE_EPOCHS, patience=PROFILE_EPOCHS,
                  batch_size=config["batch_size"], learning_rate=config["learning_rate"])
        fit_seconds = time.perf_counter() - began_fit
        if time.perf_counter() - start >= max_seconds:
            raise TimeoutError("Profile wall budget reached before integrated inference")
        began_infer = time.perf_counter()
        logits, levels = aux["delivery"].logits(model, store, context, early.index.to_numpy()[:INTEGRATION_LIMIT])
        infer_seconds = time.perf_counter() - began_infer
        if logits.shape != (INTEGRATION_LIMIT, config["draws"], 10):
            raise ValueError("Profile delivery inference differs from 400-draw contract")
        tier, counts = np.unique(levels, return_counts=True)
        entry = {"fit_seconds": fit_seconds, "integrated_inference_seconds": infer_seconds,
                 "seconds_per_integrated_pitch": infer_seconds / INTEGRATION_LIMIT,
                 "epochs_run": model.report["epochs_run"],
                 "optimizer_updates": int(np.ceil(TRAIN_LIMIT/config["batch_size"])) * model.report["epochs_run"],
                 "parameter_count": model.report["parameter_count"], "device": model.device,
                 "delivery_tier_counts": {str(int(k)): int(v) for k, v in zip(tier, counts)},
                 "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if torch.backends.mps.is_available() and time.perf_counter() - start < max_seconds:
            model.net = model.net.to("cpu")
            model.device = "cpu"
            began_cpu = time.perf_counter()
            cpu_logits = model.logits(tuple(value[:CPU_FORWARD_LIMIT] for value in early_arrays))
            if cpu_logits.shape != (CPU_FORWARD_LIMIT, 10):
                raise ValueError("CPU forward profile produced invalid shape")
            entry["cpu_forward_seconds_32"] = time.perf_counter() - began_cpu
        result["models"][kind] = entry
        print("PROFILE_DONE", kind, {key: entry[key] for key in ("fit_seconds", "integrated_inference_seconds", "device")}, flush=True)
        del model, logits
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    result["total_seconds"] = time.perf_counter() - start
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["scope"] = "Disposable profile only; TRAIN and early-stop rows, no DEV access or score"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Completed matrix preparation directory")
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 600:
        parser.error("max-seconds must be 1..600")
    config = validate_config(read_json(args.config))
    local = read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    identity = manifest_identity(config, args.local_config)
    with heavy_lock(root):
        prep = verify_prepare(output, identity)
        destination = output / "profile"
        if destination.exists():
            raise ValueError("Profile destination already exists; use a fresh preparation/run for a new profile")
        destination.mkdir()
        previous_handler = signal.getsignal(signal.SIGALRM)
        def deadline(_number, _frame):
            raise TimeoutError("Profile wall-clock deadline reached")
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        try:
            result = profile(config, local, output, prep, args.max_seconds)
            atomic_json(destination / "profile.json", result)
        except Exception as exc:
            atomic_json(destination / "failure.json", {"error_type": type(exc).__name__,
                       "message": str(exc), "failed_utc": datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
