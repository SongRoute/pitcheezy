"""Synthetic identity checks for the registered D25 architecture cells."""
from __future__ import annotations

import pytest
import numpy as np

from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_interaction import (CELLS, REFERENCE_CELLS, check_reference_archive,
                                         member_identity, validate_config)


def config():
    return {"protocol": "ml_interaction_v1", "experiment_id": "EXP-P5-001",
            "parent_architecture_run": "/approved/architecture",
            "parent_architecture_preparation_sha256": "a" * 64,
            "parent_data_run": "/approved/data",
            "parent_data_preparation_sha256": "b" * 64,
            "seeds": [0, 1, 2], "data_sample": "d25", "history_length": 5,
            "draws": 400, "width": 128, "device": "auto",
            "neural": {"epochs": 30, "patience": 5, "batch_size": 1024,
                       "learning_rate": .0005},
            "registration": {"primary": "paired 2x2 difference in differences"}}


def test_config_freezes_only_registered_axis():
    assert validate_config(config()) == config()
    for changed in ({"data_sample": "d50"}, {"draws": 25}, {"seeds": [0]},
                    {"neural": {**config()["neural"], "epochs": 5}}):
        with pytest.raises(ValueError, match="must match ML2"):
            validate_config({**config(), **changed})
    with pytest.raises(ValueError, match="SHA256"):
        validate_config({**config(), "parent_data_preparation_sha256": "bad"})
    with pytest.raises(ValueError, match="schema"):
        validate_config({**config(), "unregistered_execution_axis": True})


def test_member_identity_pins_d25_features_and_reference():
    prep = {"samples": {"train": {"rows_sha256": "c" * 64}},
            "features": {"tokens": {"token_channels": 38}},
            "architecture_preparation_sha256": "a" * 64}
    before = member_identity(prep, "I25-MLP", 0)
    assert before["kind"] == CELLS["I25-MLP"]
    assert before["reference_cell"] == REFERENCE_CELLS["I25-MLP"]
    assert before["feature_sha256"] == canonical_hash(prep["features"])
    changed = {**prep, "samples": {"train": {"rows_sha256": "d" * 64}}}
    assert member_identity(changed, "I25-MLP", 0) != before
    with pytest.raises(ValueError, match="Unregistered"):
        member_identity(prep, "I25-MLP", 42)


def test_reference_archive_requires_common_keys_raw_predictions_and_labels():
    keys = {"blend": np.array([[1, 1, 1]]), "dev": np.array([[2, 1, 1]])}
    archive = {name: np.full((1, 10), .1) for name in ("blend", "dev")}
    archive.update({name + "_raw": np.full((1, 10), .1) for name in keys})
    archive.update({name + "_keys": value.copy() for name, value in keys.items()})
    archive.update({name + "_y": np.array([0]) for name in keys})
    labels = {}
    check_reference_archive(archive, keys, labels)
    check_reference_archive(archive, keys, labels)
    assert set(labels) == {"blend", "dev"}
    with pytest.raises(ValueError, match="different keys"):
        check_reference_archive({**archive, "dev_keys": np.array([[3, 1, 1]])}, keys, labels)
    with pytest.raises(ValueError, match="labels differ"):
        check_reference_archive({**archive, "dev_y": np.array([1])}, keys, labels)
