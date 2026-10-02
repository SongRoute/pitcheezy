"""Registered enriched-H5 D25 MLP/Transformer cells for the ML4 2x2 comparison.

D100 checkpoints and probabilities remain the frozen EXP-P3-001 members. This
runner fits only the six D25 members and never scores DEV or chooses a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_interaction import (CELLS, REFERENCE_CELLS, SEEDS,
                                         check_reference_archive, check_nested_training, member_identity, validate_config)
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_benchmark import (predict_streamed, CELLS as ARCH_CELLS,
    member_identity as reference_member_identity, validate_config as validate_architecture_config)
from pitchmdp.model import outcome_labels
from run_ml_matrix import assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_benchmark import (SOURCES as ARCH_SOURCES, identity as architecture_identity,
                              load_data, read_json, dump, validate_native_runtime)
from run_sequence_pilot import arrays


SOURCES = list(dict.fromkeys([*ARCH_SOURCES, "pitchmdp/matrix_interaction.py",
                              "pitchmdp/matrix_metrics.py", "scripts/run_ml_interaction.py"]))


def source_hashes() -> dict[str, str]:
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config: dict, local_path: Path) -> dict:
    return {**architecture_identity(config, local_path), "source_hashes": source_hashes()}


def verify(output: Path, expected: dict) -> dict:
    preparation = read_json(output / "preparation.json")
    if preparation["identity"] != expected:
        raise ValueError("Interaction config, source or native environment changed")
    assert_hashes(output, preparation["artifact_hashes"])
    for path, digest in preparation["external_hashes"].items():
        if hash_file(Path(path)) != digest:
            raise ValueError("Frozen parent artifact changed: " + path)
    return preparation


def _reference_artifacts(parent: Path, preparation: dict) -> tuple[dict, dict[str, str]]:
    members, hashes = {}, {}
    expected_keys = {name: pd.read_parquet(parent / preparation["samples"][name]["path"])[KEY].to_numpy(np.int64)
                     for name in ("blend", "dev")}
    with np.load(parent / 'baseline_predictions.npz', allow_pickle=False) as baseline:
        expected_labels = {name: baseline[name + '_y'].copy() for name in ('blend', 'dev')}
        metadata = {name + '_' + field: baseline[name + '_' + field].copy()
                    for name in ('blend', 'dev') for field in ('game_pk', 'pitcher')}
        check_reference_archive(baseline, expected_keys, expected_labels)
    for candidate in REFERENCE_CELLS.values():
        for seed in SEEDS:
            folder = parent / "members" / candidate / f"seed{seed}"
            fitted = read_json(folder / "fit_state.json")
            predicted = read_json(folder / "prediction_state.json")
            if fitted['identity'] != reference_member_identity(preparation, candidate, seed):
                raise ValueError('D100 reference member identity differs from its P3 preparation')
            assert_hashes(folder, fitted["artifact_hashes"])
            assert_hashes(folder, predicted["artifact_hashes"])
            if predicted["fit_state_sha256"] != hash_file(folder / "fit_state.json"):
                raise ValueError("D100 reference prediction no longer matches checkpoint")
            with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
                check_reference_archive(archive, expected_keys, expected_labels)
                for name, expected in metadata.items():
                    if not np.array_equal(archive[name], expected):
                        raise ValueError('D100 reference metadata differs from its baseline')
            paths = [folder / "fit_state.json", folder / "prediction_state.json"]
            paths += [folder / rel for rel in fitted["artifact_hashes"]]
            paths += [folder / rel for rel in predicted["artifact_hashes"]]
            members[f"{candidate}/seed{seed}"] = {"directory": str(folder),
                "fit_state_sha256": hash_file(folder / "fit_state.json"),
                "prediction_state_sha256": hash_file(folder / "prediction_state.json"),
                "prediction_sha256": hash_file(folder / "predictions.npz")}
            hashes.update({str(path): hash_file(path) for path in paths})
    return members, hashes


def validate_parent_science(config, architecture_config, architecture):
    validate_architecture_config(architecture_config)
    if canonical_hash(architecture_config) != architecture['identity']['config_sha256']:
        raise ValueError('P3 registered configuration differs from its preparation')
    for name in ('seeds', 'history_length', 'draws', 'width', 'device', 'neural'):
        if architecture_config[name] != config[name]:
            raise ValueError('I1 scientific setting differs from frozen P3: ' + name)
    if (architecture_config['parent_preparation_sha256'] != config['parent_data_preparation_sha256']
            or Path(architecture_config['parent_run']).resolve() != Path(config['parent_data_run']).resolve()):
        raise ValueError('I1 data parent differs from the actual P3 lineage')


def prepare(config: dict, local: dict, output: Path, expected: dict) -> None:
    if output.exists() and any(output.iterdir()):
        verify(output, expected)
        print("INTERACTION_PREPARED", output, flush=True)
        return
    root = Path(local["artifact_root"]).resolve()
    protocol = root / "runs" / "ML-MATRIX-20260924"
    arch_path = Path(config["parent_architecture_run"]).resolve()
    data_path = Path(config["parent_data_run"]).resolve()
    for path in (arch_path, data_path):
        if not path.is_relative_to(protocol) or path == output or path.is_relative_to(output) or output.is_relative_to(path):
            raise ValueError("Interaction requires distinct registered sibling parent runs")
    arch_prep_path, data_prep_path = arch_path / "preparation.json", data_path / "preparation.json"
    if (hash_file(arch_prep_path) != config["parent_architecture_preparation_sha256"] or
            hash_file(data_prep_path) != config["parent_data_preparation_sha256"]):
        raise ValueError("Parent preparations differ from registered hashes")
    architecture, data = read_json(arch_prep_path), read_json(data_prep_path)
    assert_hashes(arch_path, architecture["artifact_hashes"])
    assert_hashes(data_path, data["artifact_hashes"])
    architecture_config = read_json(arch_path / 'registered_config.json')
    validate_parent_science(config, architecture_config, architecture)
    missing = [f'{cell}/seed{seed}' for cell in ARCH_CELLS for seed in SEEDS
               if not (arch_path / 'members' / cell / f'seed{seed}' / 'prediction_state.json').is_file()]
    if missing:
        raise ValueError('Complete P3 family required before I1 preparation: ' + ', '.join(missing))
    for rel, digest in architecture["identity"]["source_hashes"].items():
        if hash_file(PROJECT / rel) != digest:
            raise ValueError("Frozen architecture implementation changed: " + rel)
    if architecture["dataset_identity"] != data["dataset_identity"] or architecture["source_provenance"] != data["source_provenance"]:
        raise ValueError("Data and architecture parents use different source bytes")
    d25 = data["scope"]["regular"]["samples"]["d25"]
    if not d25["n"] < architecture["samples"]["train"]["n"]:
        raise ValueError("Registered D25 must be smaller than D100")
    check_nested_training(pd.read_parquet(data_path / d25['path']),
                          pd.read_parquet(arch_path / architecture['samples']['train']['path']))
    for split in ("earlystop", "temperature", "blend", "dev"):
        if data["scope"]["regular"]["samples"][split]["rows_sha256"] != architecture["samples"][split]["rows_sha256"]:
            raise ValueError("D25 comparison requires identical CAL/DEV pitch keys")
    references, external = _reference_artifacts(arch_path, architecture)
    external.update({str(arch_prep_path): hash_file(arch_prep_path), str(data_prep_path): hash_file(data_prep_path)})
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    dump(output / 'registered_config.json', config)
    dump(output / 'architecture_config.json', architecture_config)
    for rel in SOURCES:
        snapshot = output / "source" / rel
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, snapshot)
    shutil.copyfile(arch_path / "parent_preparation.json", output / "parent_preparation.json")
    shutil.copyfile(arch_path / "aux.pkl", output / "aux.pkl")
    shutil.copyfile(arch_path / "baseline_predictions.npz", output / "baseline_predictions.npz")
    shutil.copyfile(arch_path / "features.json", output / "features.json")
    shutil.copyfile(data_path / d25["path"], output / "train_keys.parquet")
    samples = {"train": {**d25, "path": "train_keys.parquet"}}
    for split in ("earlystop", "temperature", "blend", "dev"):
        record = architecture["samples"][split]
        target = f"{split}_keys.parquet"
        shutil.copyfile(arch_path / record["path"], output / target)
        samples[split] = {**record, "path": target}
    features = architecture["features"]
    staged = {"samples": samples, "features": features}
    # Reconstruct all ordered samples and the exact enriched feature contract
    # before accepting the preparation; load_data does not calculate DEV scores.
    _, parts, _ = load_data(local, output, staged)
    if len(parts["train"]) != d25["n"]:
        raise ValueError("Reconstructed D25 size changed")
    if source_hashes() != expected["source_hashes"]:
        raise ValueError("Interaction implementation changed during preparation")
    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    report = {"identity": expected, "architecture_run": str(arch_path), "data_run": str(data_path),
              "architecture_preparation_sha256": hash_file(arch_prep_path),
              "data_preparation_sha256": hash_file(data_prep_path),
              "samples": samples, "features": features, "cells": CELLS, "seeds": list(SEEDS),
              "d100_references": references, "external_hashes": external,
              "artifact_hashes": artifact_hashes(output, files),
              "seconds": time.perf_counter() - start,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "prepared_utc": datetime.now(timezone.utc).isoformat(), "dev_scored": False,
              "comparison": "Only D25 training keys change; P3 D100 members and auxiliaries remain frozen"}
    dump(output / "preparation.json", report)
    print("INTERACTION_PREPARED", output, flush=True)


def member_dir(output: Path, cell: str, seed: int) -> Path:
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError("Unregistered interaction member")
    return output / "members" / cell / f"seed{seed}"


def fit(config: dict, local: dict, output: Path, prep: dict, cell: str, seed: int) -> None:
    dest = member_dir(output, cell, seed)
    membership = member_identity(prep, cell, seed)
    statepath = dest / "fit_state.json"
    if statepath.exists():
        state = read_json(statepath)
        if state["identity"] != membership:
            raise ValueError("Completed interaction fit identity changed")
        assert_hashes(dest, state["artifact_hashes"])
        print("INTERACTION_FIT_COMPLETE", cell, seed, flush=True)
        return
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Incomplete interaction member requires failure review")
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    store, parts, aux = load_data(local, output, prep)
    train, early, temperature = (parts[name] for name in ("train", "earlystop", "temperature"))
    model = MatrixModel(CELLS[cell], seed=seed, width=config["width"],
                        device=None if config["device"] == "auto" else config["device"])
    model.fit(arrays(store, aux["context"], train.index.to_numpy()), outcome_labels(train),
              arrays(store, aux["context"], early.index.to_numpy()), outcome_labels(early),
              **config["neural"])
    fit_seconds = time.perf_counter() - start
    aux["delivery"].calibrate(model, store, aux["context"], temperature.index.to_numpy(), outcome_labels(temperature))
    temperature_seconds = time.perf_counter() - start - fit_seconds
    model.save(dest / "model.pt")
    dump(dest / "fit.json", {"report": model.report, "load_and_fit_seconds": fit_seconds,
         "temperature_seconds": temperature_seconds, "seconds_total": time.perf_counter() - start,
         "train_rows_sha256": prep["samples"]["train"]["rows_sha256"],
         "temperature_rows_sha256": prep["samples"]["temperature"]["rows_sha256"],
         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    if source_hashes() != prep["identity"]["source_hashes"]:
        raise ValueError("Interaction source changed during fit")
    dump(statepath, {"identity": membership,
         "artifact_hashes": artifact_hashes(dest, ["model.pt", "fit.json"]),
         "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("INTERACTION_FIT_COMPLETE", cell, seed, flush=True)


def predict(config: dict, local: dict, output: Path, prep: dict, cell: str, seed: int) -> None:
    dest = member_dir(output, cell, seed)
    fitted = read_json(dest / "fit_state.json")
    if fitted["identity"] != member_identity(prep, cell, seed):
        raise ValueError("Interaction fit identity changed")
    assert_hashes(dest, fitted["artifact_hashes"])
    statepath = dest / "prediction_state.json"
    if statepath.exists():
        saved = read_json(statepath)
        if saved["fit_state_sha256"] != hash_file(dest / "fit_state.json"):
            raise ValueError("Interaction prediction uses changed fit")
        assert_hashes(dest, saved["artifact_hashes"])
        print("INTERACTION_PREDICT_COMPLETE", cell, seed, flush=True)
        return
    if (dest / "predictions.npz").exists():
        raise ValueError("Uncommitted interaction predictions require review")
    start = time.perf_counter()
    store, parts, aux = load_data(local, output, prep)
    model = MatrixModel.load(dest / "model.pt", device=None if config["device"] == "auto" else config["device"])
    if model.kind != CELLS[cell] or model.seed != seed:
        raise ValueError("Interaction checkpoint kind/seed changed")
    values, tier_counts = {}, {}
    for name in ("blend", "dev"):
        part = parts[name]
        p, raw, levels = predict_streamed(model, aux["delivery"], store, aux["context"], part.index.to_numpy())
        values.update({name: p, name + "_raw": raw, name + "_delivery_level": levels,
                       name + "_keys": part[KEY].to_numpy(np.int64),
                       name + "_y": outcome_labels(part),
                       name + "_game_pk": part.game_pk.to_numpy(np.int64),
                       name + "_pitcher": part.pitcher.to_numpy(np.int64)})
        levels_unique, counts = np.unique(levels, return_counts=True)
        tier_counts[name] = {str(int(k)): int(v) for k, v in zip(levels_unique, counts)}
    np.savez_compressed(dest / "predictions.npz", **values)
    dump(dest / "prediction_runtime.json", {"seconds": time.perf_counter() - start,
         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "delivery_tier_counts": tier_counts, "dev_scored": False})
    if source_hashes() != prep["identity"]["source_hashes"]:
        raise ValueError("Interaction source changed during prediction")
    dump(statepath, {"fit_state_sha256": hash_file(dest / "fit_state.json"),
         "artifact_hashes": artifact_hashes(dest, ["predictions.npz", "prediction_runtime.json"]),
         "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("INTERACTION_PREDICT_COMPLETE", cell, seed, flush=True)


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
        command.add_argument("--seed", choices=tuple(SEEDS), type=int, required=True)
    args = parser.parse_args()
    config, local = validate_config(read_json(args.config)), read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, args.local_config)
    if args.command == "status":
        verify(output, expected)
        for cell in CELLS:
            for seed in SEEDS:
                folder = member_dir(output, cell, seed)
                state = "predicted" if (folder / "prediction_state.json").exists() else (
                    "fitted" if (folder / "fit_state.json").exists() else "pending")
                print(cell, seed, state)
        return
    with heavy_lock(root):
        if args.command == "prepare":
            prepare(config, local, output, expected)
            return
        prep = verify(output, expected)
        if args.command == "fit":
            fit(config, local, output, prep, args.cell, args.seed)
        else:
            predict(config, local, output, prep, args.cell, args.seed)


if __name__ == "__main__":
    main()
