"""E4 workload candidates rebuilt from frozen ordered pitch keys (feasibility only).

Every value for a row uses only rows of the same pitcher that come strictly
before it: earlier pitches of the same game, or games on earlier dates. The
current pitch is never counted. Savant's own n_thruorder_pitcher and
pitcher_days_since_prev_game are compared against these rebuilds, never used
blindly. *_days_until_next_game describe the future and are forbidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["game_pk", "at_bat_number", "pitch_number"]
SAVANT_COLUMNS = ("n_thruorder_pitcher", "pitcher_days_since_prev_game")
FORBIDDEN_COLUMNS = ("pitcher_days_until_next_game", "batter_days_until_next_game")
INPUT_COLUMNS = ("game_date", *KEY, "pitcher", *SAVANT_COLUMNS)
TRAIN_START, TRAIN_END = "2023-05-15", "2025-04-30"


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    if set(FORBIDDEN_COLUMNS).intersection(frame.columns):
        raise ValueError("Future-looking workload columns must not be loaded")
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Pitch keys must be complete and unique")
    out = frame.copy()
    out["game_date"] = pd.to_datetime(out.game_date).dt.normalize()
    if out.groupby("game_pk").game_date.nunique().gt(1).any():
        raise ValueError("A game_pk spans several dates")
    return out.sort_values(["game_date", *KEY], kind="stable")


def rebuild(frame: pd.DataFrame) -> pd.DataFrame:
    """Return prior_pitches_game, rebuilt_thruorder, rebuilt_days_since aligned to frame.index."""
    f = _ordered(frame)
    by_game = f.groupby(["game_pk", "pitcher"], sort=False)
    prior = by_game.cumcount()
    # Batters faced so far by this pitcher in this game, counted at PA starts.
    new_pa = ~f.duplicated(["game_pk", "pitcher", "at_bat_number"])
    faced_before = new_pa.astype(np.int64).groupby([f.game_pk, f.pitcher], sort=False).cumsum() - 1
    thruorder = faced_before // 9 + 1
    # Days since the pitcher's last earlier date in the loaded rows; no earlier date -> missing.
    days = f[["pitcher", "game_date"]].drop_duplicates().sort_values(["pitcher", "game_date"])
    days["rebuilt_days_since"] = days.groupby("pitcher").game_date.diff().dt.days
    out = f[["pitcher", "game_date"]].merge(days, on=["pitcher", "game_date"], how="left")
    out.index = f.index
    out["prior_pitches_game"] = prior.astype(np.int64)
    out["rebuilt_thruorder"] = thruorder.astype(np.int64)
    return out[["prior_pitches_game", "rebuilt_thruorder", "rebuilt_days_since"]].reindex(frame.index)


def audit(frame: pd.DataFrame, start: str = TRAIN_START, end: str = TRAIN_END) -> dict:
    """Column quality on TRAIN rows only; earlier loaded rows act only as history."""
    built = rebuild(frame)
    date = pd.to_datetime(frame.game_date).dt.normalize()
    if date.gt(pd.Timestamp(end)).any():
        raise ValueError("Rows after TRAIN end must not be loaded")
    train = date.ge(pd.Timestamp(start))
    f, b = frame.loc[train], built.loc[train]
    report = {"train_rows": int(train.sum()), "train_start": start, "train_end": end, "columns": {}}
    for name, rebuilt in (("n_thruorder_pitcher", "rebuilt_thruorder"),
                          ("pitcher_days_since_prev_game", "rebuilt_days_since")):
        value = pd.to_numeric(f[name], errors="coerce") if name in f else pd.Series(np.nan, index=f.index)
        both = value.notna() & b[rebuilt].notna()
        report["columns"][name] = {
            "present": name in f, "missing_rate": float(value.isna().mean()) if len(f) else None,
            "rebuild_missing_rate": float(b[rebuilt].isna().mean()) if len(f) else None,
            "agreement_rate": float(value[both].eq(b[rebuilt][both]).mean()) if both.any() else None,
            "quantiles": [float(q) for q in value.quantile([0, .5, .99, 1])] if value.notna().any() else None,
        }
    report["columns"]["prior_pitches_game"] = {
        "present": True, "missing_rate": 0.0,
        "quantiles": [float(q) for q in b.prior_pitches_game.quantile([0, .5, .99, 1])] if len(f) else None,
    }
    return report
