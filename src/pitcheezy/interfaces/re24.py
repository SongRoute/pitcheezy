"""공통 자산 0 — RE24 테이블 (docs/interface-spec.md).

RE24[base_out_id]              float64 [24]   24상태 기대득점
dRE24[terminal_idx, base_out_id] float64 [8, 24]  종결 결과 8종(outcomes.TERMINAL 순서) × 24상태의 학습 창 평균 ΔRE24
버전 = 시즌 창 + 계산 커밋. 한 실험 안에서 모두 같은 버전.
저장: {dir}/RE24.npy, dRE24.npy, meta.json (필수 키 META_REQUIRED_KEYS), sha256.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .outcomes import N_TERMINAL, TERMINAL, OUTCOME_NAMES
from .states import N_BASE_OUT
from ._io import read_sha256, sha256_file, verify_sha256, write_sha256

META_REQUIRED_KEYS: tuple[str, ...] = ("re24_version", "data_version", "season_window", "compute_commit", "n_plate_appearances")
FILES = ("RE24.npy", "dRE24.npy", "meta.json")


@dataclass
class RE24Table:
    RE24: np.ndarray
    dRE24: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def terminal_names(self) -> tuple[str, ...]:
        return tuple(OUTCOME_NAMES[i] for i in TERMINAL)

    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "RE24.npy", np.asarray(self.RE24, dtype=np.float64))
        np.save(d / "dRE24.npy", np.asarray(self.dRE24, dtype=np.float64))
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        write_sha256(d, FILES)

    @classmethod
    def load(cls, d: Path, *, check_hash: bool = True) -> "RE24Table":
        d = Path(d)
        if check_hash:
            verify_sha256(d, FILES)
        return cls(
            RE24=np.load(d / "RE24.npy"),
            dRE24=np.load(d / "dRE24.npy"),
            meta=json.loads((d / "meta.json").read_text(encoding="utf-8")),
        )


def validate(t: RE24Table) -> list[str]:
    """계약 위반 목록. 비어 있으면 통과."""
    p: list[str] = []
    if t.RE24.shape != (N_BASE_OUT,):
        p.append(f"RE24 형상 {t.RE24.shape} ≠ ({N_BASE_OUT},)")
    if t.dRE24.shape != (N_TERMINAL, N_BASE_OUT):
        p.append(f"dRE24 형상 {t.dRE24.shape} ≠ ({N_TERMINAL}, {N_BASE_OUT})")
    for name, a in (("RE24", t.RE24), ("dRE24", t.dRE24)):
        if a.dtype != np.float64:
            p.append(f"{name} dtype {a.dtype} ≠ float64")
        if np.isnan(a).any():
            p.append(f"{name} 에 NaN")
    if t.RE24.shape == (N_BASE_OUT,) and (t.RE24 < 0).any():
        p.append("RE24 에 음수")
    missing = [k for k in META_REQUIRED_KEYS if k not in t.meta]
    if missing:
        p.append(f"meta 필수 키 없음: {missing}")
    return p


__all__ = ["RE24Table", "validate", "META_REQUIRED_KEYS", "FILES", "sha256_file", "read_sha256"]
