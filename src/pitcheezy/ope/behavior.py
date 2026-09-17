"""행동 정책 π_b(a | s, 투수) — 인수분해 계층 추정 + 경기 단위 교차 적합 (잠정 D18).

π_b(a = (구종 g, 위치 l) | s, p) = π_type(g | s, p) × π_loc(l | g, c(s), p)
  π_type:  league(g)  → pitcher(g)  → pitcher(c, g)  → pitcher(s, g)        각 층 (n + α·상위) / (N + α)
  π_loc:   league(g, l) → pitcher(g, l) → pitcher(c, g, l)                  주자아웃은 위치엔 안 씀 (표본)
행동 축 225 를 통째로 평활하면 희귀 구종 셀이 바닥(1e-6)으로 떨어져 IPS 비가 폭발했다 (첫 실행). 인수분해로 투수가 실제 던지는
구종은 어느 카운트에서도 바닥으로 가지 않는다.
교차 적합: 같은 데이터로 맞추면 로그 행동의 π_b 가 부풀어 E[ρ]≈0.24 로 붕괴 → game_pk % n_folds 폴드로 나눠 밖에서 맞춘다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitcheezy.interfaces import states as S
from pitcheezy.interfaces.grid import N_ACTIONS, N_LOC
from pitcheezy.interfaces.pitch_types import N_PITCH


def _shrink(n: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    out = (n + alpha * prior) / (n.sum(-1, keepdims=True) + alpha)
    s = out.sum(-1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1.0), 0.0)


def fit_behavior(df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int, *, alpha: float, floor: float = 1e-6) -> np.ndarray:
    """π_b [P, S, A] float32."""
    S_ = S.n_states(K)
    C = S.N_COUNTS
    cid_of_state = S.decode_state(np.arange(S_), K)[0]
    ok = df["action_id"].to_numpy() >= 0
    p = df["pitcher_idx"].to_numpy(dtype=np.int64)[ok]
    s = state_id[ok].astype(np.int64)
    c = cid_of_state[s]
    g = df["pitch_id"].to_numpy(dtype=np.int64)[ok]
    l = df["loc_id"].to_numpy(dtype=np.int64)[ok]
    # --- 구종 선택
    n_psg = np.bincount((p * S_ + s) * N_PITCH + g, minlength=n_pitchers * S_ * N_PITCH).reshape(n_pitchers, S_, N_PITCH).astype(float)
    n_pcg = np.zeros((n_pitchers, C, N_PITCH)); np.add.at(n_pcg, (slice(None), cid_of_state), n_psg)
    n_pg = n_pcg.sum(1)
    league_g = n_pg.sum(0); league_g = league_g / league_g.sum()
    t_pg = _shrink(n_pg, league_g[None], alpha)  # [P, G]
    t_pcg = _shrink(n_pcg, t_pg[:, None, :], alpha)  # [P, C, G]
    t_psg = _shrink(n_psg, t_pcg[:, cid_of_state, :], alpha)  # [P, S, G]
    # --- 구종 내 위치
    n_pcgl = np.bincount(((p * C + c) * N_PITCH + g) * N_LOC + l, minlength=n_pitchers * C * N_PITCH * N_LOC).reshape(n_pitchers, C, N_PITCH, N_LOC).astype(float)
    n_pgl = n_pcgl.sum(1)  # [P, G, L]
    n_gl = n_pgl.sum(0)
    league_gl = _shrink(n_gl, np.full((N_PITCH, N_LOC), 1.0 / N_LOC), alpha)  # [G, L]
    l_pgl = _shrink(n_pgl, league_gl[None], alpha)  # [P, G, L]
    l_pcgl = _shrink(n_pcgl, l_pgl[:, None, :, :], alpha)  # [P, C, G, L]
    # --- 결합 [P, S, G, L] → [P, S, A]
    pb = t_psg[:, :, :, None] * l_pcgl[:, cid_of_state, :, :]
    pb = pb.reshape(n_pitchers, S_, N_ACTIONS) + floor
    pb /= pb.sum(-1, keepdims=True)
    return pb.astype(np.float32)


def behavior_logged_crossfit(df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int, *, alpha: float, n_folds: int, floor: float = 1e-6) -> np.ndarray:
    """투구별 π_b(로그 행동 | s, 투수) [len(df)]. 행동 없는 투구는 NaN. 폴드 = game_pk % n_folds. n_folds ≤ 1 이면 교차 적합 없음."""
    out = np.full(len(df), np.nan)
    p_all = df["pitcher_idx"].to_numpy(dtype=np.int64)
    a_all = df["action_id"].to_numpy(dtype=np.int64)
    if n_folds <= 1:
        pb = fit_behavior(df, state_id, n_pitchers, K, alpha=alpha, floor=floor)
        te = np.where(a_all >= 0)[0]
        out[te] = pb[p_all[te], state_id[te].astype(np.int64), a_all[te]]
        return out
    fold = df["game_pk"].to_numpy(dtype=np.int64) % n_folds
    for k in range(n_folds):
        tr = fold != k
        pb = fit_behavior(df[tr], state_id[tr], n_pitchers, K, alpha=alpha, floor=floor)
        te = np.where(~tr & (a_all >= 0))[0]
        out[te] = pb[p_all[te], state_id[te].astype(np.int64), a_all[te]]
    return out


def tilt(pb: np.ndarray, Q: np.ndarray, support: np.ndarray, tau: float) -> np.ndarray:
    """π_b 에 Q 를 기울인 정책 (KL 제약 개선, design.md ③ 보수화 축을 P0 로 앞당김 — 잠정 D20):
    π_e(a|s) ∝ π_b(a|s) · exp(Q(s,a)/τ), support 밖 0. τ→∞ 면 π_b, τ→0 이면 π_b 지지 위 greedy."""
    z = np.where(support, Q.astype(np.float64) / tau, -np.inf)
    z = z - np.where(support.any(-1, keepdims=True), np.max(np.where(support, z, -np.inf), axis=-1, keepdims=True), 0.0)
    e = np.where(support, pb.astype(np.float64) * np.exp(z), 0.0)
    s = e.sum(-1, keepdims=True)
    return np.where(s > 0, e / np.where(s > 0, s, 1.0), 0.0)


def crossfit_logged(
    df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int, *, alpha: float, n_folds: int,
    tilts: dict[str, tuple[np.ndarray, np.ndarray, float]] | None = None, floor: float = 1e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """교차 적합. 반환 (pb_logged [N], {tilt 이름: pe_logged [N]}). tilts = {이름: (Q, support, τ)} 는 같은 폴드의 π_b 로 기울인 π_e 의 로그 행동 확률."""
    tilts = tilts or {}
    pb_out = np.full(len(df), np.nan)
    pe_out = {k: np.full(len(df), np.nan) for k in tilts}
    p_all = df["pitcher_idx"].to_numpy(dtype=np.int64)
    a_all = df["action_id"].to_numpy(dtype=np.int64)
    s_all = state_id.astype(np.int64)
    fold = df["game_pk"].to_numpy(dtype=np.int64) % max(n_folds, 1)
    for k in range(max(n_folds, 1)):
        tr = fold != k if n_folds > 1 else np.ones(len(df), dtype=bool)
        pb = fit_behavior(df[tr], state_id[tr], n_pitchers, K, alpha=alpha, floor=floor)
        te = np.where((~tr if n_folds > 1 else tr) & (a_all >= 0))[0]
        pb_out[te] = pb[p_all[te], s_all[te], a_all[te]]
        for name, (Q, support, tau) in tilts.items():
            pe = tilt(pb, Q, support, tau)
            pe_out[name][te] = pe[p_all[te], s_all[te], a_all[te]]
    return pb_out, pe_out
