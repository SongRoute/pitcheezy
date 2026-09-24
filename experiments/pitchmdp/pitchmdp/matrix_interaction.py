"""Frozen D25 cells for the registered ML4 architecture-by-data interaction."""
from __future__ import annotations

import numpy as np

from .matrix_benchmark import NEURAL, SEEDS
from .matrix_data import canonical_hash


CELLS = {"I25-MLP": "flatten_mlp", "I25-TF": "transformer"}
REFERENCE_CELLS = {"I25-MLP": "A0-MLP", "I25-TF": "A6-transformer"}


def validate_config(config: dict) -> dict:
    required = {"protocol", "experiment_id", "parent_architecture_run",
                "parent_architecture_preparation_sha256", "parent_data_run",
                "parent_data_preparation_sha256", "seeds", "data_sample",
                "history_length", "draws", "width", "device", "neural"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"registration"}:
        raise ValueError("Invalid interaction config schema")
    if config["protocol"] != "ml_interaction_v1" or not isinstance(config["experiment_id"], str) or not config["experiment_id"].strip():
        raise ValueError("Unregistered interaction protocol or experiment")
    for name in ("parent_architecture_preparation_sha256", "parent_data_preparation_sha256"):
        digest = config[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Frozen parent preparation SHA256 required")
    for name in ("parent_architecture_run", "parent_data_run"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError("Frozen parent run path required")
    if (config["seeds"] != list(SEEDS) or config["data_sample"] != "d25" or
            config["history_length"] != 5 or config["draws"] != 400 or
            config["width"] != 128 or config["neural"] != NEURAL or
            config["device"] not in ("auto", "cpu", "mps")):
        raise ValueError("Interaction D25, H5, draws, seeds and optimizer must match ML2")
    if "registration" in config and not isinstance(config["registration"], dict):
        raise ValueError("Interaction registration must be an object")
    return config


def member_identity(preparation: dict, cell: str, seed: int) -> dict:
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError("Unregistered interaction cell/seed")
    return {"preparation_sha256": canonical_hash(preparation), "cell": cell,
            "kind": CELLS[cell], "seed": seed,
            "d25_train_rows_sha256": preparation["samples"]["train"]["rows_sha256"],
            "feature_sha256": canonical_hash(preparation["features"]),
            "reference_cell": REFERENCE_CELLS[cell],
            "architecture_preparation_sha256": preparation["architecture_preparation_sha256"]}


def check_reference_archive(archive, expected_keys: dict, expected_labels: dict) -> None:
    """Require D100 members to share the exact keyed CAL/DEV denominator."""
    for name in ("blend", "dev"):
        keys = expected_keys[name]
        if (not np.array_equal(archive[name + "_keys"], keys) or
                archive[name].shape != (len(keys), 10) or
                archive[name + "_raw"].shape != (len(keys), 10)):
            raise ValueError("D100 reference raw/calibrated predictions use different keys")
        if name not in expected_labels:
            expected_labels[name] = archive[name + "_y"].copy()
        elif not np.array_equal(archive[name + "_y"], expected_labels[name]):
            raise ValueError("D100 reference outcome labels differ between members")
