"""행동 정책 π_b(a | s, 투수) — 2026 로그 데이터에서 계층 평활 + 경기 단위 교차 적합 (잠정 D18).

같은 데이터로 π_b 를 맞추고 그 데이터의 비를 계산하면 로그된 행동의 π_b 가 부풀어 가중치가 쪼그라든다 (첫 실행에서 E[ρ]≈0.24).
→ 경기 game_pk 를 n_folds 로 나눠, 폴드 k 의 투구는 나머지 폴드로 맞춘 π_b 를 쓴다. 결과는 투구별 π_b(로그 행동) 배열.
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
    """π_b [P, S, A] float32 (전체 데이터, 교차 적합 없음 — 진단용)."""
    n = count_actions(df, state_id, n_pitchers, K)
    cid = S.decode_state(np.arange(S.n_states(K)), K)[0]
    pb = smooth_hierarchical(n, count_of_state=cid, group_of_action=np.zeros(1, dtype=np.int64), n_counts=S.N_COUNTS, n_groups=1,
                             alpha=alpha, rule_mask=None, out_dtype=np.float64)[:, :, 0, :]
    pb = pb + floor
    pb /= pb.sum(-1, keepdims=True)
    return pb.astype(np.float32)


def behavior_logged_crossfit(df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int, *, alpha: float, n_folds: int, floor: float = 1e-6) -> np.ndarray:
    """투구별 π_b(로그 행동 | s, 투수) [len(df)]. 행동 없는 투구는 NaN. 폴드 = game_pk % n_folds."""
    out = np.full(len(df), np.nan)
    fold = (df["game_pk"].to_numpy(dtype=np.int64) % n_folds)
    for k in range(n_folds):
        tr = fold != k
        pb = fit_behavior(df[tr], state_id[tr], n_pitchers, K, alpha=alpha, floor=floor)
        te = np.where(~tr & (df["action_id"].to_numpy() >= 0))[0]
        out[te] = pb[df["pitcher_idx"].to_numpy(dtype=np.int64)[te], state_id[te].astype(np.int64), df["action_id"].to_numpy(dtype=np.int64)[te]]
    return out
