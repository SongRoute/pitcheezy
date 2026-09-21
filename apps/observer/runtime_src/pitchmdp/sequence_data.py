"""Generated prediction runtime: exact selected definitions from a hash-verified capture.
Training methods may remain for class identity; the service does not invoke them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PHYSICAL_COLUMNS = ("effective_speed", "release_spin_rate", "spin_axis",
                    "pfx_x", "pfx_z", "plate_x", "plate_z")

PHYSICAL_CHANNELS = ("effective_speed", "release_spin_rate", "spin_axis_sin",
                     "spin_axis_cos", "pfx_x", "pfx_z", "plate_x", "plate_z")

def _physical_values(frame: pd.DataFrame) -> np.ndarray:
    raw = frame[list(PHYSICAL_COLUMNS)].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    raw[~np.isfinite(raw)] = np.nan
    angle = np.deg2rad(raw[:, 2] % 360.)
    return np.column_stack((raw[:, :2], np.sin(angle), np.cos(angle), raw[:, 3:]))

class PhysicalNormalizer:
    """Train-only per-channel median imputation and standardization, then frozen."""

    def fit(self, train: pd.DataFrame) -> "PhysicalNormalizer":
        if not len(train) or "split" not in train or not train.split.eq("train").all():
            raise ValueError("Physical normalization requires only explicitly marked train rows")
        if "game_date" in train:
            dates = pd.to_datetime(train.game_date)
            if not dates.between("2023-05-15", "2025-04-30").all():
                raise ValueError("Physical normalization dates fall outside TRAIN")
        values = _physical_values(train)
        self.observed_count = np.isfinite(values).sum(axis=0)
        self.fill = np.array([np.nanmedian(values[:, j]) if count else 0.
                              for j, count in enumerate(self.observed_count)])
        values = np.where(np.isfinite(values), values, self.fill)
        self.mean = values.mean(axis=0)
        self.scale = np.maximum(values.std(axis=0), 1e-6)
        self.n_train = len(train)
        self.fit_date_max = str(pd.to_datetime(train.game_date).max().date()) if "game_date" in train else None
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = _physical_values(frame)
        values = np.where(np.isfinite(values), values, self.fill)
        return np.ascontiguousarray((values-self.mean)/self.scale, dtype=np.float32)

    def report(self) -> dict:
        return {"raw_columns": list(PHYSICAL_COLUMNS), "channels": list(PHYSICAL_CHANNELS),
                "paper_adaptation": "spin axis degrees expanded to sine/cosine; 7 variables, 8 channels",
                "imputation": "train per-channel median; zero if entirely missing in train",
                "standardization": "train imputed population mean/std; minimum std 1e-6",
                "fill": self.fill.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "observed_train_counts": self.observed_count.tolist(), "n_train": self.n_train,
                "fit_date_max": self.fit_date_max}
