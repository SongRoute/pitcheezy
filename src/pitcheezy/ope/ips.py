"""IPS 계열 OPE (design.md §0-b). 에피소드 = 타석, 보상 = 타석의 실제 ΔRE24 (투수 관점 = −ΔRE24).

투구 t 의 비 ρ_t = π_e(a_t|s_t) / π_b(a_t|s_t). 행동 없는 투구(구종 제외·위치 결측)는 ρ = 1.
공통 지지: π_e 는 valid ∩ 평가 시즌 레퍼토리(투수×구종 ≥ support_min 구) 위로 재정규화. 로그 행동이 그 밖이면 ρ = 0 (타석 가중치 0).
추정량 (모두 자기정규화 SN, 경기 단위 부트스트랩 CI)
  traj     w = Π_t clip(ρ_t, c)     타석 궤적 IPS. e2e 저울 (잠정 D19). c=None 은 noclip
  onestep  w = Σ_t clip(ρ_t, c)     1스텝 편차: 한 투구만 π_e 로 바꾸고 나머지는 π_b. 분산 낮음, 정책 개선 정리의 조건. 보조
진단: ESS, max w, 평균 w (π_b 가 맞고 지지가 같으면 traj 의 평균 w ≈ 1), ρ=0 타석 비율.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SORT = ["game_pk", "at_bat_number", "pitch_number"]


def restrict_support(policy: np.ndarray, support: np.ndarray) -> np.ndarray:
    """policy [P,S,A] 를 support [P,S,A] bool 위로 재정규화. 지지 없는 상태는 0."""
    q = np.where(support, policy.astype(np.float64), 0.0)
    s = q.sum(-1, keepdims=True)
    return np.where(s > 0, q / np.where(s > 0, s, 1.0), 0.0)


def logged_probs(df: pd.DataFrame, state_id: np.ndarray, pi_e: np.ndarray) -> np.ndarray:
    """정책 배열 [P,S,A] → 투구별 π_e(로그 행동) [N] (행동 없으면 NaN)."""
    out = np.full(len(df), np.nan)
    a = df["action_id"].to_numpy(dtype=np.int64)
    m = a >= 0
    out[m] = pi_e[df["pitcher_idx"].to_numpy(dtype=np.int64)[m], state_id.astype(np.int64)[m], a[m]]
    return out


def pa_weights(df: pd.DataFrame, pe_logged: np.ndarray, pb_logged: np.ndarray, *, clip: float | None, slice_mask: np.ndarray | None = None) -> pd.DataFrame:
    """타석별 (game_pk, at_bat_number, w_traj, w_onestep, w_onestep_slice, n_decisions, n_zero). pe_logged·pb_logged·slice_mask 는 df 위치 정렬.
    slice_mask (bool[N]) 가 있으면 그 투구의 결정만 더한 1스텝 가중치 w_onestep_slice 도 낸다 (예: 2스트라이크 결정)."""
    d = df.sort_values(SORT, kind="stable")
    order = d.index.to_numpy()
    pel = pe_logged[order]
    pbl = pb_logged[order]
    sl = np.ones(len(d), dtype=bool) if slice_mask is None else slice_mask[order]
    a = d["action_id"].to_numpy(dtype=np.int64)
    has = (a >= 0) & ~np.isnan(pbl) & ~np.isnan(pel)
    rho = np.ones(len(d), dtype=np.float64)
    r = pel[has] / np.maximum(pbl[has], 1e-12)
    if clip is not None:
        r = np.minimum(r, clip)
    rho[has] = r
    rho_dec = np.where(has, rho, 0.0)
    g = pd.DataFrame({"game_pk": d["game_pk"].to_numpy(), "at_bat_number": d["at_bat_number"].to_numpy(), "rho": rho, "rho_dec": rho_dec, "rho_slice": np.where(sl, rho_dec, 0.0),
                      "has": has, "has_slice": has & sl, "zero": has & (rho == 0)})
    out = g.groupby(["game_pk", "at_bat_number"], sort=True).agg(w_traj=("rho", "prod"), w_onestep=("rho_dec", "sum"), w_onestep_slice=("rho_slice", "sum"),
                                                                  n_decisions=("has", "sum"), n_decisions_slice=("has_slice", "sum"), n_zero=("zero", "sum")).reset_index()
    return out


def estimate(w: np.ndarray, r: np.ndarray) -> dict:
    sw = w.sum()
    return {
        "ips": float((w * r).mean()),
        "snips": float((w * r).sum() / sw) if sw > 0 else float("nan"),
        "ess": float(sw ** 2 / (w ** 2).sum()) if sw > 0 else 0.0,
        "ess_frac": float(sw ** 2 / (w ** 2).sum() / len(w)) if sw > 0 else 0.0,
        "w_max": float(w.max()), "w_mean": float(w.mean()),
        "n_pa": int(len(w)), "frac_zero_w": float((w == 0).mean()),
    }


def bootstrap(w: np.ndarray, r: np.ndarray, games: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """자기정규화 추정량의 경기 클러스터 부트스트랩."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(games, return_inverse=True)
    n_g = len(ug)
    wr_g = np.bincount(inv, weights=w * r, minlength=n_g)
    w_g = np.bincount(inv, weights=w, minlength=n_g)
    est = np.empty(n_boot)
    for b in range(n_boot):
        m = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g)).astype(float)
        den = (m * w_g).sum()
        est[b] = (m * wr_g).sum() / den if den > 0 else np.nan
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "boot_std": float(np.nanstd(est)), "n_boot": int(n_boot), "n_games": int(n_g), "seed": int(seed)}
