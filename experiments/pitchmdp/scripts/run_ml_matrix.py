"""Registered 2025 ML0/R2 and ML1/D1 fit and prediction adapter.

No stage chooses a model from DEV.  ``prepare`` freezes samples and auxiliary
objects, ``fit`` trains one requested cell/seed, and ``predict`` saves keyed
CAL/DEV probabilities.  Family scoring is a separate, batch-complete step.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import json
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import torch

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import (CELLS, assert_output_path, canonical_hash,
                                  load_verified_processed_cache, nested_game_samples,
                                  ordered_key_hash, validate_config)
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_data import HistoryStore
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceContext, SequenceModel
from run_sequence_pilot import arrays, dump
from run_temporal_blend import assign_fold, samples_for, select_cohort
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline, fit_temperature, temperature_predictions
from run_sequence_context_frequency import ContextFrequencyBaseline


PROTOCOL_DIR = "ML-MATRIX-20260924"
SOURCES = ["pitchmdp/archetypes.py", "pitchmdp/data.py", "pitchmdp/matrix_data.py",
           "pitchmdp/model.py", "pitchmdp/sequence_data.py", "pitchmdp/sequence_delivery.py",
           "pitchmdp/sequence_model.py", "scripts/run_ml_matrix.py", "scripts/run_sequence_pilot.py",
           "scripts/run_temporal_blend.py", "scripts/run_sequence_frequency_baselines.py",
           "scripts/run_sequence_context_frequency.py"]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, value)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_hashes() -> dict[str, str]:
    return {rel: hash_file(PROJECT / rel) for rel in SOURCES}


def artifact_hashes(root: Path, names: list[str]) -> dict[str, str]:
    return {name: hash_file(root / name) for name in names}


def assert_hashes(root: Path, hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or hash_file(path) != digest:
            raise ValueError(f"Artifact identity changed: {name}")


@contextmanager
def heavy_lock(root: Path):
    lock_path = root / "runs" / PROTOCOL_DIR / ".heavy.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another ML matrix heavy job holds {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_location(local: dict, output: Path) -> Path:
    root = Path(local["artifact_root"]).resolve()
    assert_output_path(output, root)
    protocol = root / "runs" / PROTOCOL_DIR
    if not output.resolve().is_relative_to(protocol.resolve()) or output.resolve() == protocol.resolve():
        raise ValueError(f"Output must be a distinct child of {protocol}")
    return root


def manifest_identity(config: dict, local_path: Path) -> dict:
    return {"config_sha256": canonical_hash(config), "local_config_sha256": hash_file(local_path),
            "source_hashes": source_hashes(), "python": str(Path(sys.executable).resolve()),
            "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}


def verify_prepare(output: Path, identity: dict) -> dict:
    report = read_json(output / "preparation.json")
    if report["identity"] != identity:
        raise ValueError("Config, source, local path, or environment differs from preparation")
    assert_hashes(output, report["artifact_hashes"])
    return report


def fold_frame(raw: pd.DataFrame, scope: str, ids: list[int] | None = None):
    if scope == "regular":
        if "game_type" not in raw:
            raise ValueError("Regular-season D1 requires game_type")
        raw = raw.loc[raw.game_type.eq("R")].copy().reset_index(drop=True)
    elif scope != "legacy":
        raise ValueError("Unknown data scope")
    frame = assign_fold(add_batter_style_history(raw), 2025)
    if ids is None:
        ids, cohort = select_cohort(frame)
    else:
        cohort = {"pitcher_ids": ids, "selection": "Frozen LEGACY 2025 TRAIN cohort"}
    full, parts = samples_for(frame, ids)
    return frame, ids, cohort, full, parts


def _save_keys(directory: Path, parts: dict[str, pd.DataFrame], prefix: str) -> dict[str, dict]:
    manifest = {}
    for name, part in parts.items():
        rel = f"{prefix}/{name}_keys.parquet"
        directory.joinpath(rel).parent.mkdir(parents=True, exist_ok=True)
        part[KEY].to_parquet(directory / rel, index=False)
        manifest[name] = {"path": rel, "n": len(part), "games": int(part.game_pk.nunique()),
                          "rows_sha256": ordered_key_hash(part),
                          "date_min": str(part.game_date.min()), "date_max": str(part.game_date.max())}
    return manifest


def _fit_aux(frame: pd.DataFrame, full: pd.DataFrame, destination: Path, config: dict) -> dict:
    # HistoryStore fits physical normalization on every TRAIN row in this scope,
    # including rows that are ineligible as supervised outcomes. This preserves
    # the original temporal runner's preprocessing contract.
    store = HistoryStore.from_frame(frame)
    context = SequenceContext().fit(full)
    delivery = JointDelivery().fit(full, store.normalizer, draws=config["draws"], seed=config["sample_seed"])
    baseline = ContextFrequencyBaseline(HierarchicalFrequencyBaseline().fit(full)).fit(full)
    aux = {"context": context, "delivery": delivery, "baseline": baseline,
           "normalizer": store.normalizer}
    with destination.open("wb") as stream:
        pickle.dump(aux, stream)
    return {"normalizer": store.normalizer.report(), "context": context.report(),
            "delivery": delivery.report, "baseline": baseline.report,
            "normalizer_fit_scope": "all TRAIN rows including ineligible rows in this population",
            "other_aux_fit_scope": "eligible D100 TRAIN rows"}


def _baseline_artifacts(output: Path, scope: str, parts: dict[str, pd.DataFrame]) -> dict:
    with (output / f"{scope}_aux.pkl").open("rb") as stream:
        baseline = pickle.load(stream)["baseline"]
    temperature = parts["temperature"]
    chosen = fit_temperature(baseline.predict(temperature), outcome_labels(temperature))
    archive = {}
    for name in ("blend", "dev"):
        part = parts[name]
        archive[name] = temperature_predictions(baseline.predict(part), chosen["temperature"])
        archive[name + "_keys"] = part[KEY].to_numpy(np.int64)
        archive[name + "_y"] = outcome_labels(part)
        archive[name + "_game_pk"] = part.game_pk.to_numpy(np.int64)
        archive[name + "_pitcher"] = part.pitcher.to_numpy(np.int64)
    np.savez_compressed(output / f"{scope}_baseline_predictions.npz", **archive)
    return {"temperature_selection": chosen,
            "temperature_rows_sha256": ordered_key_hash(temperature),
            "note": "Saved keyed probabilities only; DEV scores are deferred until full batch scoring"}


def prepare(config: dict, local: dict, output: Path, identity: dict) -> None:
    if output.exists() and any(output.iterdir()):
        verify_prepare(output, identity)
        print("PREPARED", output, flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        snapshot = output / "source" / rel
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, snapshot)
    started = time.perf_counter()
    raw = load_verified_processed_cache(local)
    data_identity = raw.attrs["sequence_data_identity"]
    source_provenance = raw.attrs["matrix_source_provenance"]
    if pd.to_datetime(raw.game_date).max() > pd.Timestamp("2025-12-31"):
        raise ValueError("2026 data are outside the permitted development population")
    legacy, ids, cohort, legacy_full, legacy_parts = fold_frame(raw.copy(), "legacy")
    legacy_manifest = _save_keys(output, {"full_train": legacy_full, **legacy_parts}, "legacy")
    legacy_aux = _fit_aux(legacy, legacy_full, output / "legacy_aux.pkl", config)
    legacy_baseline = _baseline_artifacts(output, "legacy", legacy_parts)
    del legacy, legacy_full, legacy_parts
    gc.collect()
    regular, _, _, regular_full, regular_parts = fold_frame(raw, "regular", ids)
    games, sampling = nested_game_samples(regular_full, seed=config["sample_seed"])
    regular_manifest = _save_keys(output, {"full_train": regular_full,
                                          **{k: regular_full.loc[regular_full.game_pk.isin(v)] for k, v in games.items()},
                                          **{k: v for k, v in regular_parts.items() if k != "train"}}, "regular")
    regular_aux = _fit_aux(regular, regular_full, output / "regular_aux.pkl", config)
    regular_baseline = _baseline_artifacts(output, "regular", regular_parts)
    for name, ids_array in games.items():
        np.save(output / f"{name}_game_ids.npy", ids_array, allow_pickle=False)
    scope = {"legacy": {"cohort": cohort, "samples": legacy_manifest, "aux": legacy_aux,
                         "baseline": legacy_baseline},
             "regular": {"cohort_ids": ids, "samples": regular_manifest, "aux": regular_aux,
                         "baseline": regular_baseline, "nested_game_sampling": sampling}}
    atomic_json(output / "samples.json", scope)
    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    report = {"identity": identity, "dataset_identity": data_identity,
              "source_provenance": source_provenance,
              "dataset_identity_sha256": canonical_hash(data_identity), "scope": scope,
              "artifact_hashes": artifact_hashes(output, files), "seconds": time.perf_counter()-started,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "prepared_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": False, "dev_scored": False}
    atomic_json(output / "preparation.json", report)
    print("PREPARED", output, flush=True)


def _select_by_keys(frame: pd.DataFrame, output: Path, record: dict) -> pd.DataFrame:
    keys = pd.read_parquet(output / record["path"])
    if len(keys) != record["n"] or keys.duplicated(KEY).any():
        raise ValueError("Prepared keys changed")
    index = pd.MultiIndex.from_frame(frame[KEY])
    wanted = pd.MultiIndex.from_frame(keys[KEY])
    positions = index.get_indexer(wanted)
    if (positions < 0).any():
        raise ValueError("Prepared sample absent from source frame")
    part = frame.iloc[positions]
    if ordered_key_hash(part) != record["rows_sha256"]:
        raise ValueError("Prepared ordered pitch sample changed")
    return part


def load_cell_data(local: dict, output: Path, prep: dict, cell: str):
    _, _, sample = CELLS[cell]
    scope = "legacy" if sample == "legacy" else "regular"
    raw = load_verified_processed_cache(local)
    if raw.attrs["sequence_data_identity"] != prep["dataset_identity"]:
        raise ValueError("Source data identity changed since preparation")
    if raw.attrs["matrix_source_provenance"] != prep["source_provenance"]:
        raise ValueError("Derived source provenance changed since preparation")
    frame = assign_fold(add_batter_style_history(raw if scope == "legacy" else
                        raw.loc[raw.game_type.eq("R")].copy().reset_index(drop=True)), 2025)
    records = prep["scope"][scope]["samples"]
    train_name = "train" if sample == "legacy" else sample
    names = [train_name, "earlystop", "temperature", "blend", "dev"]
    parts = {name: _select_by_keys(frame, output, records[name]) for name in names}
    with (output / f"{scope}_aux.pkl").open("rb") as stream:
        aux = pickle.load(stream)
    store = HistoryStore.from_frame(frame, normalizer=aux["normalizer"])
    return store, parts, aux


def member_dir(output: Path, cell: str, seed: int) -> Path:
    if cell not in CELLS or seed not in CELLS[cell][1]:
        raise ValueError("Unsupported registered cell/seed")
    return output / "members" / cell / f"seed{seed}"


def member_identity(prep: dict, cell: str, seed: int) -> dict:
    kind, _, sample = CELLS[cell]
    return {"preparation_sha256": canonical_hash(prep), "cell": cell, "seed": seed,
            "kind": kind, "sample": sample,
            "train_rows_sha256": prep["scope"]["legacy" if sample == "legacy" else "regular"]["samples"]["train" if sample == "legacy" else sample]["rows_sha256"]}


def fit(config: dict, local: dict, output: Path, prep: dict, cell: str, seed: int) -> None:
    dest = member_dir(output, cell, seed)
    identity = member_identity(prep, cell, seed)
    statepath = dest / "fit_state.json"
    if statepath.exists():
        state = read_json(statepath)
        if state["identity"] != identity:
            raise ValueError("Completed member identity changed")
        assert_hashes(dest, state["artifact_hashes"])
        print("FIT_COMPLETE", cell, seed, flush=True)
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Incomplete member directory requires manual failure review; no implicit overwrite")
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    store, parts, aux = load_cell_data(local, output, prep, cell)
    context = aux["context"]
    train = parts["train" if CELLS[cell][2] == "legacy" else CELLS[cell][2]]
    early = parts["earlystop"]
    model = SequenceModel(CELLS[cell][0], seed=seed, width=config["width"]).fit(
        arrays(store, context, train.index.to_numpy()), outcome_labels(train),
        arrays(store, context, early.index.to_numpy()), outcome_labels(early),
        epochs=config["epochs"], patience=config["patience"], batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], checkpoint=dest / "best_training.pt")
    temp = parts["temperature"]
    aux["delivery"].calibrate(model, store, context, temp.index.to_numpy(), outcome_labels(temp))
    model.save(dest / "model.pt")
    atomic_json(dest / "fit.json", {"report": model.report, "seconds_total": time.perf_counter()-started,
                                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   "optimizer_updates": int(np.ceil(len(train)/config["batch_size"])) * model.report["epochs_run"],
                                   "batch_size": config["batch_size"],
                                   "train_rows_sha256": ordered_key_hash(train),
                                   "temperature_rows_sha256": ordered_key_hash(temp)})
    if source_hashes() != prep["identity"]["source_hashes"]:
        raise ValueError("Runner source changed during fit")
    atomic_json(statepath, {"identity": identity,
                            "artifact_hashes": artifact_hashes(dest, ["model.pt", "best_training.pt", "fit.json"]),
                            "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("FIT_COMPLETE", cell, seed, flush=True)


def predict(local: dict, output: Path, prep: dict, cell: str, seed: int) -> None:
    dest = member_dir(output, cell, seed)
    state = read_json(dest / "fit_state.json")
    if state["identity"] != member_identity(prep, cell, seed):
        raise ValueError("Fit identity changed")
    assert_hashes(dest, state["artifact_hashes"])
    predstate = dest / "prediction_state.json"
    if predstate.exists():
        saved = read_json(predstate)
        if saved["fit_state_sha256"] != hash_file(dest / "fit_state.json"):
            raise ValueError("Prediction uses changed fit state")
        assert_hashes(dest, saved["artifact_hashes"])
        print("PREDICT_COMPLETE", cell, seed, flush=True)
        return
    if (dest / "predictions.npz").exists():
        raise ValueError("Uncommitted prediction artifact requires manual failure review")
    started = time.perf_counter()
    store, parts, aux = load_cell_data(local, output, prep, cell)
    model = SequenceModel.load(dest / "model.pt")
    if model.kind != CELLS[cell][0] or model.seed != seed:
        raise ValueError("Checkpoint model/seed differs from cell")
    values = {}
    for name in ("blend", "dev"):
        part = parts[name]
        p = aux["delivery"].predict(model, store, aux["context"], part.index.to_numpy())
        if p.shape != (len(part), 10) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError("Integrated prediction probabilities are invalid")
        values[name] = p
        values[name + "_keys"] = part[KEY].to_numpy(np.int64)
        values[name + "_y"] = outcome_labels(part)
        values[name + "_game_pk"] = part.game_pk.to_numpy(np.int64)
        values[name + "_pitcher"] = part.pitcher.to_numpy(np.int64)
    np.savez_compressed(dest / "predictions.npz", **values)
    if source_hashes() != prep["identity"]["source_hashes"]:
        raise ValueError("Runner source changed during prediction")
    atomic_json(dest / "prediction_runtime.json", {"seconds": time.perf_counter()-started,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "note": "Pre-pitch 400-draw joint delivery integration; no logged current physics at inference"})
    atomic_json(predstate, {"fit_state_sha256": hash_file(dest / "fit_state.json"),
               "artifact_hashes": artifact_hashes(dest, ["predictions.npz", "prediction_runtime.json"]),
               "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("PREDICT_COMPLETE", cell, seed, flush=True)


def status(output: Path, identity: dict) -> None:
    prep = verify_prepare(output, identity)
    for cell, (_, seeds, _) in CELLS.items():
        for seed in seeds:
            dest = member_dir(output, cell, seed)
            fit_state = (dest / "fit_state.json").exists()
            prediction_state = (dest / "prediction_state.json").exists()
            print(cell, seed, "predicted" if prediction_state else "fitted" if fit_state else "pending")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("status")
    for name in ("fit", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--cell", choices=tuple(CELLS), required=True)
        command.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    local = read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    identity = manifest_identity(config, args.local_config)
    if args.command == "status":
        status(output, identity)
        return
    with heavy_lock(root):
        if args.command == "prepare":
            prepare(config, local, output, identity)
            return
        prep = verify_prepare(output, identity)
        member_dir(output, args.cell, args.seed)
        if args.command == "fit":
            fit(config, local, output, prep, args.cell, args.seed)
        else:
            predict(local, output, prep, args.cell, args.seed)


if __name__ == "__main__":
    main()
