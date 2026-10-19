"""Past-five physical tokens plus an explicitly conditional current-pitch token.

Seven paper variables become eight channels because spin_axis is represented by
sine/cosine. No outcomes, player IDs, or future rows enter these physical tokens.
The current realization is appropriate for conditional outcome prediction only;
pre-pitch decisions must supply/integrate a candidate realization separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import KEY, PA_KEY, RAW_ALLOWLIST, hash_file, utc_now, write_json


PHYSICAL_COLUMNS = ("effective_speed", "release_spin_rate", "spin_axis",
                    "pfx_x", "pfx_z", "plate_x", "plate_z")
PHYSICAL_CHANNELS = ("effective_speed", "release_spin_rate", "spin_axis_sin",
                     "spin_axis_cos", "pfx_x", "pfx_z", "plate_x", "plate_z")
SIDECAR_COLUMNS = ("effective_speed", "spin_axis")
CACHE_VERSION = 1


def join_physics(frame: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Align raw physical columns by pitch identity, never by source row order."""
    if frame[KEY].isna().any().any() or sidecar[KEY].isna().any().any():
        raise ValueError("Missing pitch keys")
    if frame.duplicated(KEY).any() or sidecar.duplicated(KEY).any():
        raise ValueError("Duplicate pitch keys in processed frame or physics sidecar")
    if set(SIDECAR_COLUMNS).intersection(frame.columns):
        raise ValueError("Processed frame already contains sidecar columns")
    if len(frame) != len(sidecar):
        raise ValueError("Physics sidecar and processed frame have different key counts")
    result = frame.merge(sidecar[KEY + list(SIDECAR_COLUMNS)], on=KEY, how="left",
                         sort=False, validate="one_to_one", indicator=True)
    if not result.pop("_merge").eq("both").all():
        raise ValueError("Physics sidecar is missing processed pitch keys")
    return result


def prepare_frame(config: dict | Path | str) -> pd.DataFrame:
    """Load all processed rows and verified missing physics without rewriting them.

    The existing data_quality manifest pins original and processed SHA256 values.
    Only three explicitly approved source paths are opened, including on cache
    reuse. Cache data and metadata must both match the pinned dataset identity.
    """
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    if config["raw_allowlist"] != RAW_ALLOWLIST:
        raise ValueError("Only the three exact approved raw files may be opened")
    root, volume = Path(config["artifact_root"]), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.resolve().is_relative_to(volume.resolve()):
        raise RuntimeError("Mounted T7 Shield artifact root is required")
    quality = json.loads((root / "reports/data_quality.json").read_text())
    expected = quality["sources"]
    if [source["file"] for source in expected] != RAW_ALLOWLIST:
        raise ValueError("Data-quality manifest does not pin the approved source list")
    processed = root / "processed/pitches.parquet"
    processed_hash = hash_file(processed)
    if processed_hash != quality["processed_sha256"]:
        raise ValueError("Processed data hash differs from data-quality manifest")
    for source in expected:
        path = Path(config["raw_root"]) / source["file"]
        if path.stat().st_size != source["bytes"] or hash_file(path) != source["sha256"]:
            raise ValueError(f"Raw source identity changed: {source['file']}")
    identity = {"cache_version": CACHE_VERSION, "sources": expected,
                "processed_sha256": processed_hash, "rows": quality["rows"],
                "key": KEY, "columns": list(SIDECAR_COLUMNS)}
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "sequence_physics.parquet"
    metadata = cache_dir / "sequence_physics.json"
    valid_cache = False
    if cache.exists() and metadata.exists():
        old = json.loads(metadata.read_text())
        valid_cache = old.get("identity") == identity and old.get("sha256") == hash_file(cache)
    if valid_cache:
        sidecar = pd.read_parquet(cache)
    else:
        pieces = []
        for source in expected:
            path = Path(config["raw_root"]) / source["file"]
            columns = KEY + ["game_date"] + list(SIDECAR_COLUMNS)
            schema = pq.read_schema(path)
            if not set(columns).issubset(schema.names):
                raise ValueError(f"Missing sequence variables in {source['file']}")
            raw = pq.read_table(path, columns=columns).to_pandas()
            dates = pd.to_datetime(raw.pop("game_date"))
            year = int(source["file"].split("_")[1].split(".")[0])
            if dates.isna().any() or not dates.dt.year.eq(year).all():
                raise ValueError("Unexpected or forbidden dates in approved source")
            pieces.append(raw)
        sidecar = pd.concat(pieces, ignore_index=True)
        del pieces
    frame = pd.read_parquet(processed)
    if len(frame) != quality["rows"]:
        raise ValueError("Processed row count differs from data-quality manifest")
    dates = pd.to_datetime(frame.game_date)
    if dates.isna().any() or not dates.dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Unexpected or forbidden processed dates")
    frame = join_physics(frame, sidecar)
    frame.sort_values(["game_date", *KEY], inplace=True, ignore_index=True)
    if not valid_cache:
        temporary = cache.with_suffix(".parquet.tmp")
        sidecar.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(cache)
        write_json(metadata, {"created_at_utc": utc_now(), "identity": identity,
                              "sha256": hash_file(cache)})
    frame.attrs["sequence_data_identity"] = identity
    return frame


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


def build_history_indices(frame: pd.DataFrame, history_length: int = 5) -> np.ndarray:
    """Return prior row positions oldest-to-newest, with -1 left padding.

    Input must have contiguous PAs in chronological pitch order. Row positions
    refer to this exact full frame, before filtering supervised examples.
    """
    if not isinstance(history_length, int) or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    if frame[KEY].isna().any().any():
        raise ValueError("Missing pitch keys")
    n = len(frame)
    if n > np.iinfo(np.int32).max:
        raise ValueError("Too many rows for int32 history indices")
    if not n:
        return np.empty((0, history_length), dtype=np.int32)
    game = frame.game_pk.to_numpy()
    pa = frame.at_bat_number.to_numpy()
    pitch = frame.pitch_number.to_numpy()
    first = np.r_[True, (game[1:] != game[:-1]) | (pa[1:] != pa[:-1])]
    if pd.MultiIndex.from_frame(frame.loc[first, PA_KEY]).has_duplicates:
        raise ValueError("Plate appearances must be contiguous")
    if np.any((~first[1:]) & (pitch[1:] <= pitch[:-1])):
        raise ValueError("Pitch order within each PA must be strictly increasing")
    if "game_date" in frame and not pd.to_datetime(frame.game_date).is_monotonic_increasing:
        raise ValueError("Frame dates must be chronological")
    positions = np.arange(n, dtype=np.int32)
    start = np.maximum.accumulate(np.where(first, positions, 0))
    indices = positions[:, None] - np.arange(history_length, 0, -1, dtype=np.int32)
    indices[indices < start[:, None]] = -1
    return np.ascontiguousarray(indices)


@dataclass
class HistoryStore:
    """Compact row features plus row references; expand tokens only per batch."""

    frame: pd.DataFrame
    physical: np.ndarray
    indices: np.ndarray
    normalizer: PhysicalNormalizer

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, normalizer: PhysicalNormalizer | None = None,
                   history_length: int = 5) -> "HistoryStore":
        if normalizer is None:
            normalizer = PhysicalNormalizer().fit(frame.loc[frame.split.eq("train")])
        return cls(frame, normalizer.transform(frame),
                   build_history_indices(frame, history_length), normalizer)

    def gather(self, rows, current: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return [past...,current] float32 tokens and bool mask (True=valid).

        ``current`` overrides the last token with already-normalized candidate
        physics. Omitting it uses actual current physics (conditional evaluation).
        Padded history is exactly zero, and must be excluded using the mask.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= len(self.frame)):
            raise IndexError("Sequence row positions are outside the store")
        history = self.indices[rows]
        valid = history >= 0
        tokens = np.zeros((len(rows), history.shape[1]+1, self.physical.shape[1]), dtype=np.float32)
        tokens[:, :-1] = self.physical[np.maximum(history, 0)]
        tokens[:, :-1][~valid] = 0.
        if current is None:
            current = self.physical[rows]
        current = np.asarray(current, dtype=np.float32)
        if current.shape != (len(rows), self.physical.shape[1]) or not np.isfinite(current).all():
            raise ValueError("Candidate current physics must be finite normalized [batch,8] values")
        tokens[:, -1] = current
        return tokens, np.column_stack((valid, np.ones(len(rows), dtype=bool)))
