"""Registered D2-25 restricted-auxiliary data contract.

Only complete, unselected regular-season TRAIN games are removed. All earlier
pre-TRAIN and later observed rows remain available for past-only histories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY, add_history
from .archetypes import add_batter_style_history
from .matrix_data import ordered_key_hash, canonical_hash


SEEDS = (0, 1, 2)
PARTS = ("train", "earlystop", "temperature", "blend", "dev")


def validate_config(config: dict) -> dict:
    required = {"protocol", "experiment_id", "parent_run", "parent_preparation_sha256",
                "training_games", "seeds", "history_length", "kind", "width", "draws",
                "sample_seed", "device", "budget"}
    if not isinstance(config, dict) or not required <= config.keys() or set(config) - required - {"registration"}:
        raise ValueError("Invalid restricted-auxiliary config schema")
    if config["protocol"] != "ml_aux_data_v1" or config["experiment_id"] != "EXP-P6-001":
        raise ValueError("Unsupported restricted-auxiliary protocol")
    digest = config["parent_preparation_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Parent preparation SHA256 required")
    if not isinstance(config["parent_run"], str) or not config["parent_run"]:
        raise ValueError("Parent run path required")
    fixed = {"training_games": "d25", "seeds": list(SEEDS), "history_length": 5,
             "kind": "flatten_mlp", "width": 128, "draws": 400, "sample_seed": 42}
    if any(config[name] != value for name, value in fixed.items()):
        raise ValueError("D2 fixed sample/model settings differ from registration")
    if config["device"] not in ("auto", "cpu", "mps") or config["budget"] != {
            "epochs": 30, "patience": 5, "batch_size": 1024, "learning_rate": .0005}:
        raise ValueError("D2 optimizer/device setting differs from registration")
    if "registration" in config and not isinstance(config["registration"], dict):
        raise ValueError("Registration must be a structured object")
    return config


def restrict_train_games(frame: pd.DataFrame, selected_games: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Remove unselected TRAIN games, then rebuild dated and within-PA histories.

    `frame` is the already sorted physical-cache join. Historical `split` and
    history columns may be present; the fresh dated split alone determines the
    omission mask. The selected game array comes from the parent D25 artifact.
    """
    from scripts.run_temporal_blend import assign_fold

    if frame.empty or "game_type" not in frame or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Restricted history needs complete unique raw pitch keys and game type")
    ids = np.asarray(selected_games, dtype=np.int64)
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("Selected TRAIN games must be nonempty and unique")
    regular = frame.loc[frame.game_type.eq("R")].copy().reset_index(drop=True)
    dated = assign_fold(regular, 2025)
    by_game = dated.groupby("game_pk", sort=False).split.nunique()
    if by_game.gt(1).any():
        raise ValueError("One whole game spans multiple temporal splits")
    train_games = set(dated.loc[dated.split.eq("train"), "game_pk"].astype(int))
    if not set(ids.tolist()) <= train_games:
        raise ValueError("Selected games are not a subset of regular TRAIN games")
    keep = ~dated.split.eq("train") | dated.game_pk.isin(ids)
    retained = dated.loc[keep].copy().reset_index(drop=True)
    if retained.loc[retained.split.eq("train"), "game_pk"].nunique() != len(ids):
        raise ValueError("TRAIN restriction did not retain precisely selected whole games")
    # add_history overwrites the precomputed within-PA and batter cumulative
    # columns. Styles overwrite six dated rates and six reliability columns.
    # Neither transform sees an omitted TRAIN game or a later calendar date.
    add_history(retained)
    add_batter_style_history(retained)
    if retained[KEY].isna().any().any() or retained.duplicated(KEY).any():
        raise ValueError("Rebuilt history changed pitch identity")
    report = {"source_regular_rows": len(dated), "retained_rows": len(retained),
              "omitted_train_rows": int((~keep).sum()),
              "selected_train_games": len(ids), "retained_train_raw_rows": int(retained.split.eq("train").sum()),
              "retained_ordered_key_sha256": ordered_key_hash(retained),
              "selected_games_sha256": canonical_hash(sorted(ids.tolist())),
              "pretrain_rows": int(retained.game_date.lt(pd.Timestamp("2023-05-15")).sum()),
              "posttrain_rows": int(retained.game_date.gt(pd.Timestamp("2025-04-30")).sum())}
    return retained, report
