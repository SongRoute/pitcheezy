"""D2-25 restricted-auxiliary ML follow-up; preparation, fit and prediction only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import canonical_hash, load_verified_processed_cache, ordered_key_hash
from pitchmdp.matrix_aux_data import PARTS, SEEDS, restrict_train_games, validate_config
from pitchmdp.model import outcome_labels, eligible
from pitchmdp.sequence_data import HistoryStore
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceContext, SequenceModel
from run_ml_matrix import (SOURCES as PARENT_SOURCES, artifact_hashes, assert_hashes,
                           check_location, heavy_lock, integrated_probabilities,
                           member_identity as parent_member_identity, read_json,
                           _select_by_keys)
from run_sequence_pilot import arrays, dump
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline, fit_temperature, temperature_predictions
from run_sequence_context_frequency import ContextFrequencyBaseline


SOURCES = list(dict.fromkeys([*PARENT_SOURCES, "pitchmdp/matrix_aux_data.py",
                              "scripts/run_ml_aux_data.py"]))


def source_hashes() -> dict[str, str]:
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config: dict, local_path: Path) -> dict:
    return {"config_sha256": canonical_hash(config), "local_config_sha256": hash_file(local_path),
            "source_hashes": source_hashes(), "python": str(Path(sys.executable).resolve())}


def member_dir(output: Path, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError("Unregistered D2 seed")
    return output / "members" / "D2-25" / f"seed{seed}"


def member_identity(prep: dict, seed: int) -> dict:
    return {"preparation_sha256": canonical_hash(prep), "cell": "D2-25", "seed": seed,
            "train_rows_sha256": prep["samples"]["train"]["rows_sha256"],
            "aux_sha256": prep["artifact_hashes"]["aux.pkl"]}


def verify(output: Path, expected: dict) -> dict:
    prep = read_json(output / "preparation.json")
    if prep["identity"] != expected:
        raise ValueError("D2 config, source or runtime changed from preparation")
    assert_hashes(output, prep["artifact_hashes"])
    for name, digest in prep["external_hashes"].items():
        if hash_file(Path(name)) != digest:
            raise ValueError("D2 parent artifact changed: " + name)
    return prep


def _parent(config: dict, local: dict, output: Path) -> tuple[Path, dict, dict[str, str]]:
    parent = Path(config["parent_run"]).resolve()
    protocol = Path(local["artifact_root"]).resolve() / "runs" / "ML-MATRIX-20260924"
    if (not parent.is_relative_to(protocol) or parent == output or
            parent.is_relative_to(output) or output.is_relative_to(parent)):
        raise ValueError("D2 requires a distinct registered sibling parent run")
    prep_path = parent / "preparation.json"
    if hash_file(prep_path) != config["parent_preparation_sha256"]:
        raise ValueError("P2 parent preparation differs from registered SHA256")
    prep = read_json(prep_path)
    assert_hashes(parent, prep["artifact_hashes"])
    for name, digest in prep["identity"]["source_hashes"].items():
        if hash_file(PROJECT / name) != digest:
            raise ValueError("Frozen P2 implementation changed: " + name)
    if prep["dataset_identity_sha256"] != canonical_hash(prep["dataset_identity"]):
        raise ValueError("P2 dataset identity changed")
    external = {str(prep_path): hash_file(prep_path)}
    common = {}
    with np.load(parent / "regular_baseline_predictions.npz", allow_pickle=False) as baseline:
        for split in ("blend", "dev"):
            for suffix in ("_keys", "_y", "_game_pk", "_pitcher"):
                common[split + suffix] = baseline[split + suffix].copy()
            record = prep["scope"]["regular"]["samples"][split]
            if not np.array_equal(common[split + "_keys"],
                                  pd.read_parquet(parent / record["path"])[KEY].to_numpy(np.int64)):
                raise ValueError("P2 baseline and frozen evaluation keys differ")
    for cell in ("D1-25", "D1-100"):
        for seed in SEEDS:
            folder = parent / "members" / cell / f"seed{seed}"
            fit_path, pred_path = folder / "fit_state.json", folder / "prediction_state.json"
            fitted, predicted = read_json(fit_path), read_json(pred_path)
            if fitted["identity"] != parent_member_identity(prep, cell, seed):
                raise ValueError("D1 reference member identity differs from P2 parent")
            if predicted["fit_state_sha256"] != hash_file(fit_path):
                raise ValueError("D1 reference predictions differ from fit")
            assert_hashes(folder, fitted["artifact_hashes"])
            assert_hashes(folder, predicted["artifact_hashes"])
            with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
                for name, expected_values in common.items():
                    if not np.array_equal(archive[name], expected_values):
                        raise ValueError("D1 reference prediction keys/truth differ: " + name)
                for split in ("blend", "dev"):
                    for name in (split, split + "_raw"):
                        probability = archive[name]
                        if (probability.shape != (len(common[split + "_y"]), 10) or
                                not np.isfinite(probability).all() or (probability < 0).any() or
                                (probability > 1).any() or
                                not np.allclose(probability.sum(axis=1), 1, atol=1e-6, rtol=0)):
                            raise ValueError("D1 reference probability archive is invalid")
            for path in (fit_path, pred_path):
                external[str(path)] = hash_file(path)
            for rel in list(fitted["artifact_hashes"]) + list(predicted["artifact_hashes"]):
                path = folder / rel
                external[str(path)] = hash_file(path)
    return parent, prep, external


def _construct(local: dict, parent: Path, parent_prep: dict, selected: np.ndarray):
    raw = load_verified_processed_cache(local)
    if (raw.attrs["sequence_data_identity"] != parent_prep["dataset_identity"] or
            raw.attrs["matrix_source_provenance"] != parent_prep["source_provenance"]):
        raise ValueError("Derived source bytes/provenance differ from P2 preparation")
    frame, history = restrict_train_games(raw, selected)
    records = parent_prep["scope"]["regular"]["samples"]
    parts = {"train": _select_by_keys(frame, parent, records["d25"])}
    for name in PARTS[1:]:
        parts[name] = _select_by_keys(frame, parent, records[name])
    if not parts["train"].split.eq("train").all() or not eligible(parts["train"]).all():
        raise ValueError("D2 train contains ineligible or non-TRAIN pitches")
    if not set(parts["train"].game_pk.astype(int)) == set(selected.tolist()):
        raise ValueError("D2 supervised TRAIN differs from selected whole games")
    return frame, parts, history


def _source_copy(output: Path) -> None:
    for rel in SOURCES:
        path = output / "source" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, path)


def prepare(config: dict, local: dict, output: Path, expected: dict) -> None:
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        print("D2_PREPARED", output, flush=True)
        return
    parent, parent_prep, external = _parent(config, local, output)
    selected = np.load(parent / "d25_game_ids.npy", allow_pickle=False)
    sample = parent_prep["scope"]["regular"]["samples"]["d25"]
    if sample["games"] != len(selected):
        raise ValueError("P2 D25 game sample differs from registered game IDs")
    start = time.perf_counter()
    frame, parts, history = _construct(local, parent, parent_prep, selected)
    # Physical normalization uses every retained regular TRAIN row, whereas
    # context, frequency and delivery see only eligible selected TRAIN rows.
    store = HistoryStore.from_frame(frame, history_length=config["history_length"])
    train = parts["train"]
    context = SequenceContext().fit(train)
    delivery = JointDelivery().fit(train, store.normalizer, draws=config["draws"], seed=config["sample_seed"])
    baseline = ContextFrequencyBaseline(HierarchicalFrequencyBaseline().fit(train)).fit(train)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "registered_config.json", config)
    _source_copy(output)
    with (output / "aux.pkl").open("wb") as stream:
        pickle.dump({"normalizer": store.normalizer, "context": context,
                     "delivery": delivery, "baseline": baseline}, stream)
    samples = {}
    for name, part in parts.items():
        rel = f"{name}_keys.parquet"
        part[KEY].to_parquet(output / rel, index=False)
        samples[name] = {"path": rel, "n": len(part), "games": int(part.game_pk.nunique()),
                         "rows_sha256": ordered_key_hash(part)}
        parent_name = "d25" if name == "train" else name
        if samples[name]["rows_sha256"] != parent_prep["scope"]["regular"]["samples"][parent_name]["rows_sha256"]:
            raise ValueError("D2 evaluation or TRAIN ordered keys differ from frozen P2")
    chosen = fit_temperature(baseline.predict(parts["temperature"]), outcome_labels(parts["temperature"]))
    baseline_archive = {}
    for name in ("blend", "dev"):
        part = parts[name]
        raw = baseline.predict(part)
        baseline_archive[name + "_raw"] = raw
        baseline_archive[name] = temperature_predictions(raw, chosen["temperature"])
        baseline_archive[name + "_keys"] = part[KEY].to_numpy(np.int64)
        baseline_archive[name + "_y"] = outcome_labels(part)
        baseline_archive[name + "_game_pk"] = part.game_pk.to_numpy(np.int64)
        baseline_archive[name + "_pitcher"] = part.pitcher.to_numpy(np.int64)
    with np.load(parent / "regular_baseline_predictions.npz", allow_pickle=False) as previous:
        for name in ("blend", "dev"):
            for suffix in ("_keys", "_y", "_game_pk", "_pitcher"):
                if not np.array_equal(baseline_archive[name + suffix], previous[name + suffix]):
                    raise ValueError("D2 CAL/DEV keys, truth or metadata differ from D1")
    np.savez_compressed(output / "baseline_predictions.npz", **baseline_archive)
    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    report = {"identity": expected, "parent_run": str(parent),
              "parent_preparation_sha256": hash_file(parent / "preparation.json"),
              "dataset_identity": parent_prep["dataset_identity"],
              "source_provenance": parent_prep["source_provenance"],
              "history": history, "samples": samples,
              "aux": {"normalizer": store.normalizer.report(), "context": context.report(),
                      "delivery": delivery.report, "baseline": baseline.report,
                      "normalizer_fit_scope": "all raw selected regular TRAIN game rows, including ineligible",
                      "other_aux_fit_scope": "eligible D25 TRAIN only"},
              "baseline_temperature_selection": chosen,
              "external_hashes": external, "artifact_hashes": artifact_hashes(output, files),
              "seconds": time.perf_counter()-start,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "prepared_utc": datetime.now(timezone.utc).isoformat(), "dev_scored": False}
    dump(output / "preparation.json", report)
    print("D2_PREPARED", output, flush=True)


def load_data(local: dict, output: Path, prep: dict):
    parent = Path(prep["parent_run"])
    selected = np.load(parent / "d25_game_ids.npy", allow_pickle=False)
    parent_prep = read_json(parent / "preparation.json")
    frame, parts, history = _construct(local, parent, parent_prep, selected)
    if history != prep["history"]:
        raise ValueError("D2 rebuilt history/source population changed")
    for name in PARTS:
        if ordered_key_hash(parts[name]) != prep["samples"][name]["rows_sha256"]:
            raise ValueError("D2 prepared sample changed")
    with (output / "aux.pkl").open("rb") as stream:
        aux = pickle.load(stream)
    store = HistoryStore.from_frame(frame, normalizer=aux["normalizer"], history_length=5)
    return store, parts, aux


def profile(config: dict, local: dict, output: Path, prep: dict) -> None:
    """Bounded real-data TRAIN/CAL cost probe; never reads DEV targets."""
    destination = output / "profile"
    report_path = destination / "profile.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["preparation_sha256"] != canonical_hash(prep):
            raise ValueError("D2 cost profile uses another preparation")
        print("D2_PROFILE_COMPLETE", output, flush=True)
        return
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Incomplete D2 profile requires failure review")
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    store, parts, aux = load_data(local, output, prep)
    train = parts["train"].iloc[:8192]
    early = parts["earlystop"].iloc[:2048]
    temperature = parts["temperature"].iloc[:64]
    model = SequenceModel("flatten_mlp", seed=0, width=128)
    if config["device"] != "auto":
        model.device = config["device"]
    model.fit(arrays(store, aux["context"], train.index.to_numpy()), outcome_labels(train),
              arrays(store, aux["context"], early.index.to_numpy()), outcome_labels(early),
              epochs=2, patience=2, batch_size=1024, learning_rate=.0005)
    fit_seconds = time.perf_counter()-start
    aux["delivery"].calibrate(model, store, aux["context"],
                              temperature.index.to_numpy(), outcome_labels(temperature))
    calibration_seconds = time.perf_counter()-start-fit_seconds
    logits, tiers = aux["delivery"].logits(model, store, aux["context"],
                                          parts["train"].iloc[:256].index.to_numpy())
    integrated_probabilities(logits, model.delivery_temperature)
    prediction_seconds = time.perf_counter()-start-fit_seconds-calibration_seconds
    if source_hashes() != prep["identity"]["source_hashes"]:
        raise ValueError("D2 source changed during profile")
    dump(report_path, {"preparation_sha256": canonical_hash(prep), "train_rows": len(train),
         "earlystop_rows": len(early), "temperature_rows": len(temperature),
         "prediction_train_rows": 256, "draws": config["draws"],
         "fit_seconds": fit_seconds, "calibration_seconds": calibration_seconds,
         "prediction_seconds": prediction_seconds, "wall_seconds": time.perf_counter()-start,
         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "delivery_tier_counts": {str(int(level)): int((tiers == level).sum()) for level in np.unique(tiers)},
         "dev_scores_read": False, "profiled_utc": datetime.now(timezone.utc).isoformat()})
    print("D2_PROFILE_COMPLETE", output, flush=True)


def fit(config: dict, local: dict, output: Path, prep: dict, seed: int) -> None:
    dest = member_dir(output, seed)
    statepath = dest / "fit_state.json"
    membership = member_identity(prep, seed)
    if statepath.exists():
        state = read_json(statepath)
        if state["identity"] != membership:
            raise ValueError("D2 completed fit identity changed")
        assert_hashes(dest, state["artifact_hashes"])
        print("D2_FIT_COMPLETE", seed, flush=True)
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Incomplete D2 member requires failure review")
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    store, parts, aux = load_data(local, output, prep)
    train, early, temperature = (parts[name] for name in ("train", "earlystop", "temperature"))
    model = SequenceModel(config["kind"], seed=seed, width=config["width"])
    if config["device"] != "auto":
        model.device = config["device"]
    model.fit(arrays(store, aux["context"], train.index.to_numpy()), outcome_labels(train),
              arrays(store, aux["context"], early.index.to_numpy()), outcome_labels(early),
              **config["budget"], checkpoint=dest / "best_training.pt")
    aux["delivery"].calibrate(model, store, aux["context"],
                              temperature.index.to_numpy(), outcome_labels(temperature))
    model.save(dest / "model.pt")
    dump(dest / "fit.json", {"report": model.report, "seconds_total": time.perf_counter()-start,
         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "train_rows_sha256": ordered_key_hash(train),
         "temperature_rows_sha256": ordered_key_hash(temperature)})
    if source_hashes() != prep["identity"]["source_hashes"]:
        raise ValueError("D2 source changed during fit")
    dump(statepath, {"identity": membership,
         "artifact_hashes": artifact_hashes(dest, ["model.pt", "best_training.pt", "fit.json"]),
         "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("D2_FIT_COMPLETE", seed, flush=True)


def predict(local: dict, output: Path, prep: dict, seed: int) -> None:
    dest = member_dir(output, seed)
    fitted = read_json(dest / "fit_state.json")
    if fitted["identity"] != member_identity(prep, seed):
        raise ValueError("D2 fit identity changed")
    assert_hashes(dest, fitted["artifact_hashes"])
    statepath = dest / "prediction_state.json"
    if statepath.exists():
        state = read_json(statepath)
        if state["fit_state_sha256"] != hash_file(dest / "fit_state.json"):
            raise ValueError("D2 prediction differs from fitted checkpoint")
        assert_hashes(dest, state["artifact_hashes"])
        print("D2_PREDICT_COMPLETE", seed, flush=True)
        return
    if (dest / "predictions.npz").exists():
        raise ValueError("Uncommitted D2 predictions require failure review")
    start = time.perf_counter()
    store, parts, aux = load_data(local, output, prep)
    model = SequenceModel.load(dest / "model.pt")
    if model.kind != "flatten_mlp" or model.seed != seed:
        raise ValueError("D2 checkpoint architecture/seed changed")
    values, tiers = {}, {}
    for name in ("blend", "dev"):
        part = parts[name]
        logits, levels = aux["delivery"].logits(model, store, aux["context"], part.index.to_numpy())
        if logits.shape != (len(part), 400, 10):
            raise ValueError("D2 delivery logits differ from registered 400-draw shape")
        values[name], values[name + "_raw"] = integrated_probabilities(logits, model.delivery_temperature)
        values[name + "_delivery_level"] = levels
        values[name + "_keys"] = part[KEY].to_numpy(np.int64)
        values[name + "_y"] = outcome_labels(part)
        values[name + "_game_pk"] = part.game_pk.to_numpy(np.int64)
        values[name + "_pitcher"] = part.pitcher.to_numpy(np.int64)
        unique, counts = np.unique(levels, return_counts=True)
        tiers[name] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    np.savez_compressed(dest / "predictions.npz", **values)
    dump(dest / "prediction_runtime.json", {"seconds": time.perf_counter()-start,
         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "delivery_tier_counts": tiers, "dev_scored": False})
    if source_hashes() != prep["identity"]["source_hashes"]:
        raise ValueError("D2 source changed during prediction")
    dump(statepath, {"fit_state_sha256": hash_file(dest / "fit_state.json"),
         "artifact_hashes": artifact_hashes(dest, ["predictions.npz", "prediction_runtime.json"]),
         "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("D2_PREDICT_COMPLETE", seed, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "status", "profile"):
        commands.add_parser(name)
    for name in ("fit", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    local = read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    expected = identity(config, args.local_config)
    if args.command == "status":
        prep = verify(output, expected)
        for seed in SEEDS:
            folder = member_dir(output, seed)
            print(seed, "predicted" if (folder / "prediction_state.json").exists()
                  else "fitted" if (folder / "fit_state.json").exists() else "pending")
        return
    with heavy_lock(root):
        if args.command == "prepare":
            prepare(config, local, output, expected)
        elif args.command == "profile":
            profile(config, local, output, verify(output, expected))
        elif args.command == "fit":
            fit(config, local, output, verify(output, expected), args.seed)
        else:
            predict(local, output, verify(output, expected), args.seed)


if __name__ == "__main__":
    main()
