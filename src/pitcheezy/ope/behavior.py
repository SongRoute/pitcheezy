"""행동 정책 π_b(a | s, 투수) 추정 — 2026 로그 데이터에서 계층 평활 (잠정 D18).

층: L0 league(a|s) → base league(a|count) / L1 pitcher(a|count) / L2 pitcher(a|s) ∝ L0 × L1/league(count).
transition.smoothing 을 행동 축(V=225, A축은 크기 1)으로 재사용. 바닥 확률 floor 를 더해 로그된 행동이 0 이 되지 않게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitcheezy.interfaces import states as S
from pitcheezy.interfaces.grid import N_ACTIONS
from pitcheezy.transition.smoothing import smooth_hierarchical


def count_actions(df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int) -> np.ndarray:
    S_ = S.n_states(K)
    ok = df["action_id"].to_numpy() >= 0
    p = df["pitcher_idx"].to_numpy(dtype=np.int64)[ok]
    s = state_id[ok].astype(np.int64)
    a = df["action_id"].to_numpy(dtype=np.int64)[ok]
    n = np.bincount((p * S_ + s) * N_ACTIONS + a, minlength=n_pitchers * S_ * N_ACTIONS)
    return n.reshape(n_pitchers, S_, 1, N_ACTIONS).astype(np.int32)


def fit_behavior(df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int, *, alpha: float, floor: float = 1e-6) -> np.ndarray:
    """π_b [P, S, A] float32."""
    n = count_actions(df, state_id, n_pitchers, K)
    cid = S.decode_state(np.arange(S.n_states(K)), K)[0]
    pb = smooth_hierarchical(
        n, count_of_state=cid, group_of_action=np.zeros(1, dtype=np.int64), n_counts=S.N_COUNTS, n_groups=1, alpha=alpha,
        rule_mask=None, out_dtype=np.float64,
    )[:, :, 0, :]
    pb = pb + floor
    pb /= pb.sum(-1, keepdims=True)
    return pb.astype(np.float32)
