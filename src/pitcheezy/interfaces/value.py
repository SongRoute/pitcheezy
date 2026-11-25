"""자산 2 — 가치함수 (policy → ope·decomp). docs/interface-spec.md.

Q.npy [P, S, A] float32 (단위 RE24), V.npy [P, S] = valid 행동 위 max, policy.npy [P, S, A] = 완화 후 분포
조회는 lookup() 으로만. Phase 0 mode="snap", "bilinear" 는 자리만. V(의도)·V(실제)는 같은 mode.
저장 runs/{실험ID}/s{seed}/value/ + meta.json + sha256.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._io import save_array, verify_sha256, write_sha256
from .grid import action_id, loc_id

LOOKUP_MODES = ("snap", "bilinear")
META_REQUIRED_KEYS: tuple[str, ...] = (
    "transition_dir",
    "transition_sha256",
    "re24_version",
    "dre24_version",
    "terminal_reward",  # 종결 보상 축 설명
    "in_play_reward",
    "gamma",  # 1
    "relax",  # {"method": "softmax"|"topk", ...}
    "lookup_mode",
    "seed",
    "train_commit",
)
FILES = ("Q.npy", "V.npy", "policy.npy", "meta.json")


@dataclass
class ValueBundle:
    Q: np.ndarray  # float32 [P, S, A]
    V: np.ndarray  # float32 [P, S]
    policy: np.ndarray  # float32 [P, S, A]
    meta: dict

    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        save_array(d / "Q.npy", self.Q, np.float32)
        np.save(d / "V.npy", np.asarray(self.V, dtype=np.float32))
        save_array(d / "policy.npy", self.policy, np.float32)
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        write_sha256(d, FILES)

    @classmethod
    def load(cls, d: Path, *, check_hash: bool = True, mmap: bool = False) -> "ValueBundle":
        d = Path(d)
        if check_hash:
            verify_sha256(d, FILES)
        mm = "r" if mmap else None
        return cls(
            Q=np.load(d / "Q.npy", mmap_mode=mm),
            V=np.load(d / "V.npy"),
            policy=np.load(d / "policy.npy", mmap_mode=mm),
            meta=json.loads((d / "meta.json").read_text(encoding="utf-8")),
        )


def lookup(Q: np.ndarray, pitcher_idx, state_id, pitch_id, plate_x, z_norm, mode: str = "snap"):
    """Q[pitcher, state, action(pitch, 위치)] 조회. 위치는 연속값을 받아 격자로 맞춘다.

    snap: 위치를 셀로 스냅해 그 셀의 Q. 같은 셀이면 V(의도)=V(실제) → 실책 0.
    bilinear: Phase 0 에서는 자리만 (NotImplementedError).
    """
    if mode not in LOOKUP_MODES:
        raise ValueError(f"mode 는 {LOOKUP_MODES} 중 하나: {mode!r}")
    if mode == "bilinear":
        raise NotImplementedError("bilinear 룩업은 Phase 0 범위 밖 (자리만)")
    aid = action_id(pitch_id, loc_id(plate_x, z_norm))
    out = Q[np.asarray(pitcher_idx, dtype=np.int64), np.asarray(state_id, dtype=np.int64), aid]
    return float(out) if np.ndim(out) == 0 else out
