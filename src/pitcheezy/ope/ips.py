"""IPS 계열 OPE (design.md §0-b). 에피소드 = 타석, 보상 = 타석의 실제 ΔRE24 (투수 관점 = −ΔRE24).

투구 t 의 비 ρ_t = π_e(a_t|s_t) / π_b(a_t|s_t). 행동 없는 투구(구종 제외·위치 결측)는 ρ = 1.
타석 가중치 w = Π_t clip(ρ_t, 0, c) (c = None 이면 클립 없음).
  IPS   = mean(w·r)
  SNIPS = Σ w·r / Σ w   (자기정규화, 주 추정량 — 잠정 D19)
진단: ESS = (Σw)²/Σw², max w, ρ=0 타석 비율(π_e 지지 밖 행동), 타석 수.
CI: 경기 단위 클러스터 부트스트랩(시드), 95% 백분위.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SORT = ["game_pk", "at_bat_number", "pitch_number"]


def pa_weights(df: pd.DataFrame, state_id: np.ndarray, pi_e: np.ndarray, pi_b: np.ndarray, *, clip: float | None) -> pd.DataFrame:
    """타석별 (game_pk, at_bat_number, w, log_w, n_decisions, n_zero)."""
    d = df.sort_values(SORT, kind="stable")
    order = d.index.to_numpy()
    sid = state_id[order]
    p = d["pitcher_idx"].to_numpy(dtype=np.int64)
    a = d["action_id"].to_numpy(dtype=np.int64)
    has = a >= 0
    rho = np.ones(len(d), dtype=np.float64)
    pe = pi_e[p[has], sid[has], a[has]].astype(np.float64)
    pb = pi_b[p[has], sid[has], a[has]].astype(np.float64)
    r = pe / np.maximum(pb, 1e-12)
    if clip is not None:
        r = np.minimum(r, clip)
    rho[has] = r
    key = d[["game_pk", "at_bat_number"]].reset_index(drop=True)
    g = pd.DataFrame({"game_pk": key["game_pk"], "at_bat_number": key["at_bat_number"], "rho": rho, "has": has, "zero": has & (rho == 0)})
    out = g.groupby(["game_pk", "at_bat_number"], sort=True).agg(w=("rho", "prod"), n_decisions=("has", "sum"), n_zero=("zero", "sum")).reset_index()
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


def bootstrap(w: np.ndarray, r: np.ndarray, games: np.ndarray, *, n_boot: int, seed: int, key: str = "snips") -> dict:
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(games, return_inverse=True)
    n_g = len(ug)
    # 경기별 합계로 빠르게: Σ_g m_g · (Σ_{i∈g} w r), Σ_g m_g · Σ w
    wr_g = np.bincount(inv, weights=w * r, minlength=n_g)
    w_g = np.bincount(inv, weights=w, minlength=n_g)
    n_pa_g = np.bincount(inv, minlength=n_g).astype(float)
    est = np.empty(n_boot)
    for b in range(n_boot):
        m = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g)).astype(float)
        if key == "snips":
            den = (m * w_g).sum()
            est[b] = (m * wr_g).sum() / den if den > 0 else np.nan
        else:
            est[b] = (m * wr_g).sum() / (m * n_pa_g).sum()
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "boot_std": float(np.nanstd(est)), "n_boot": int(n_boot), "n_games": int(n_g), "seed": int(seed)}
