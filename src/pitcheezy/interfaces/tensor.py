"""자산 1 — 전이 확률 텐서 (transition → policy). docs/interface-spec.md.

P[투수, state, action, outcome] float32 dense + valid[P,S,A] bool + n_obs[P,S,A] int32 + 룩업 parquet 3개 + meta.json + sha256.txt
저장 runs/{실험ID}/s{seed}/transition/ (홀드아웃용은 transition_holdout/)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._io import save_array, verify_sha256, write_sha256
from .grid import N_ACTIONS
from .outcomes import N_OUTCOMES, outcomes_table
from .states import n_states, states_table

SPEC_VERSION = "v1"
DEFAULT_REPERTOIRE_MIN_PITCHES = 100
DEFAULT_ROW_SUM_TOL = 1e-5

META_REQUIRED_KEYS: tuple[str, ...] = (
    "spec_version",
    "data_version",
    "season_window",
    "holdout_split",  # "season:2025" / "none"
    "model_arch",
    "seed",
    "train_commit",
    "K",
    "cluster_file_version",
    "pitch_type_map_version",
    "repertoire_min_pitches",
    "row_sum_tol",
    "holdout_nll",
    "holdout_ece",
    "holdout_ece_hr",
    "excluded_pitchers",
)  # C·context_kind 는 필수가 아니다 (옛 산출물 호환. 없으면 C=1)
PITCHERS_COLUMNS = ("pitcher_idx", "mlbam_id", "name", "n_pitches_train")
FILES = ("P.npy", "valid.npy", "n_obs.npy", "states.parquet", "outcomes.parquet", "pitchers.parquet", "meta.json")


@dataclass
class TransitionTensor:
    P: np.ndarray  # float32 [P, S, A, O]
    valid: np.ndarray  # bool [P, S, A]
    n_obs: np.ndarray  # int32 [P, S, A]
    pitchers: pd.DataFrame  # PITCHERS_COLUMNS
    meta: dict
    states: pd.DataFrame = field(default=None)  # 없으면 meta.K 로 생성
    outcomes: pd.DataFrame = field(default=None)  # 없으면 스펙 표

    def __post_init__(self) -> None:
        if self.states is None:
            self.states = states_table(int(self.meta["K"]), self.C)
        if self.outcomes is None:
            self.outcomes = outcomes_table()

    @property
    def n_pitchers(self) -> int:
        return int(self.P.shape[0])

    @property
    def K(self) -> int:
        return int(self.meta["K"])

    @property
    def C(self) -> int:
        """맥락 수. 옛 산출물(키 없음)은 1."""
        return int(self.meta.get("C", 1))

    @property
    def context_kind(self) -> str | None:
        """맥락 종류. C=1 이거나 옛 산출물이면 None."""
        return self.meta.get("context_kind")

    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        save_array(d / "P.npy", self.P, np.float32)
        np.save(d / "valid.npy", np.asarray(self.valid, dtype=bool))
        save_array(d / "n_obs.npy", self.n_obs, np.int32)
        self.states.to_parquet(d / "states.parquet", index=False)
        self.outcomes.to_parquet(d / "outcomes.parquet", index=False)
        self.pitchers.to_parquet(d / "pitchers.parquet", index=False)
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        write_sha256(d, FILES)

    @classmethod
    def load(cls, d: Path, *, check_hash: bool = True, mmap: bool = False) -> "TransitionTensor":
        d = Path(d)
        if check_hash:
            verify_sha256(d, FILES)
        mm = "r" if mmap else None
        return cls(
            P=np.load(d / "P.npy", mmap_mode=mm),
            valid=np.load(d / "valid.npy"),
            n_obs=np.load(d / "n_obs.npy"),
            pitchers=pd.read_parquet(d / "pitchers.parquet"),
            meta=json.loads((d / "meta.json").read_text(encoding="utf-8")),
            states=pd.read_parquet(d / "states.parquet"),
            outcomes=pd.read_parquet(d / "outcomes.parquet"),
        )


def expected_shape(n_pitchers: int, K: int, C: int = 1) -> tuple[int, int, int, int]:
    return (n_pitchers, n_states(K, C), N_ACTIONS, N_OUTCOMES)
