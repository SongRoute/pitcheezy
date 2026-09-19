"""이중 강건(DR) OPE — IPS 에 모델의 Q̂ 를 제어변량으로 붙여 분산을 줄인다 (γ=1, 보상은 종결 투구에만).

보상이 타석 끝에서만 나오고 γ=1 이라 어느 투구에서 보든 남은 보상(return-to-go)은 그 타석의 r_j 하나다.
따라서 Q̂(p, s, a) = E[r_j | s, a] 가 그대로 제어변량이 된다.

1스텝 DR (onestep_dr) — 한 투구만 π_e 로 바꾸고 나머지는 π_b. 기존 onestep SNIPS 의 DR 판.
    DR1 = Σ_j (Σ_i ρ_i·r_j − Σ_i ρ_i·q̂_b(p_i,s_i,a_i)) / Σ_j Σ_i ρ_i  +  Σ_j Σ_i v̂_e(p_i,s_i) / Σ_j n_dec_j
  q̂_b 는 행동 정책의 연속가치(V_b = policy_evaluation(π_b))에서 만든다. π_e 와 무관한 고정 제어변량이라
  교차 적합이 필요 없다 (편향은 붙지 않고 분산만 줄인다). v̂_e(p,s) = Σ_a π_e(a|s)·q̂_b(p,s,a).

궤적 DR (traj_dr) — Jiang & Li (2016), γ=1:
    DR_j = v̂_e(s_1) + Σ_{t=1..T} ρ_{1:t}·( r_t + v̂_e(s_{t+1}) − q̂_t ),  r_t = r_j (t=T) / 0,  v̂_e(s_{T+1}) = 0
  q̂_t = q̂_e(p,s_t,a_t) (결정) / v̂_e(p,s_t) (행동 없는 투구). q̂_e·v̂_e 는 π_e 의 모델 내 가치.
  WDR (Thomas & Brunskill 2016) 은 ρ_{1:t} 를 그 스텝까지 간 타석들의 평균으로 나눈 것 (스텝별 자기정규화).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitcheezy.ope.ips import SORT, restrict_support
from pitcheezy.policy import vi as VI

TERM_COLS = ("sum_rho", "sum_rho_q", "sum_v", "n_dec")


def q_from_v(P: np.ndarray, R: np.ndarray, nxt: np.ndarray, V: np.ndarray, *, chunk: int = 16) -> np.ndarray:
    """Q[p,s,a] = Σ_o P[p,s,a,o]·(R[s,o] + 1[nxt≥0]·V[p, nxt[s,o]]) float64. P 는 mmap 가능 → 투수 청크로."""
    n_p = P.shape[0]
    Q = np.empty(P.shape[:3])
    nxt_safe = np.where(nxt >= 0, nxt, 0)
    is_next = (nxt >= 0)[None]
    for lo in range(0, n_p, chunk):
        hi = min(lo + chunk, n_p)
        Vn = np.where(is_next, V[lo:hi][:, nxt_safe], 0.0)  # [c, S, O]
        Q[lo:hi] = np.einsum("psao,pso->psa", np.asarray(P[lo:hi], dtype=np.float64), R[None] + Vn)
    return Q


def behavior_q(P: np.ndarray, pb_full: np.ndarray, support: np.ndarray, R: np.ndarray, nxt: np.ndarray, *, chunk: int = 16) -> np.ndarray:
    """1스텝 DR 의 제어변량 q̂_b [P,S,A]. π_b 를 공통 지지 위로 재정규화한 뒤 모델 안에서 평가. 정책 메뉴와 무관 → 스테이지당 한 번."""
    V_b = VI.policy_evaluation(P, restrict_support(pb_full, support), R, nxt, chunk=chunk)
    return q_from_v(P, R, nxt, V_b, chunk=chunk)


def dr_inputs(P: np.ndarray, pol_arr: np.ndarray, q_b: np.ndarray, R: np.ndarray, nxt: np.ndarray, *, V_e: np.ndarray | None = None, chunk: int = 16):
    """정책 하나의 DR 재료 (v_e_b [P,S], V_e [P,S], q_e [P,S,A]).

    v_e_b = Σ_a π_e(a|s)·q̂_b  (1스텝 DR 의 직접법 항), V_e·q_e = π_e 의 모델 내 가치 (궤적 DR).
    V_e 를 넘기면 policy_evaluation 을 다시 돌리지 않는다 (stage_ope 의 model_value 가 이미 계산).
    stage_ope 와 scripts/ope_compare.py 가 같은 재료를 쓰게 하는 단일 출처.
    """
    v_e_b = np.einsum("psa,psa->ps", pol_arr.astype(np.float64), q_b)
    if V_e is None:
        V_e = VI.policy_evaluation(P, pol_arr, R, nxt, chunk=chunk)
    return v_e_b, V_e, q_from_v(P, R, nxt, V_e, chunk=chunk)


# ---------------------------------------------------------------- 타석 조각
def _sorted_view(df: pd.DataFrame, state_id: np.ndarray, *arrays: np.ndarray):
    """df 를 (game_pk, at_bat_number, pitch_number) 로 정렬한 뒤 (d, 위치정렬된 배열들, 타석 코드 inv, 타석 키, 타석 내 순번 pos)."""
    d = df.sort_values(SORT, kind="stable")
    o = d.index.to_numpy()
    gk = d[["game_pk", "at_bat_number"]].to_numpy()
    if len(d) == 0:
        return d, [state_id[o]] + [a[o] for a in arrays], np.zeros(0, np.int64), gk, np.zeros(0, np.int64)
    chg = np.empty(len(d), dtype=bool)
    chg[0] = True
    chg[1:] = (gk[1:] != gk[:-1]).any(1)
    inv = np.cumsum(chg) - 1
    start = np.flatnonzero(chg)
    pos = np.arange(len(d)) - start[inv]  # 타석 내 0-기반 투구 순번
    return d, [state_id[o].astype(np.int64)] + [a[o] for a in arrays], inv, gk[chg], pos


def onestep_dr_terms(df: pd.DataFrame, state_id: np.ndarray, rho: np.ndarray, has: np.ndarray, q_b: np.ndarray, v_e: np.ndarray) -> pd.DataFrame:
    """타석별 1스텝 DR 조각 (game_pk, at_bat_number, sum_rho, sum_rho_q, sum_v, n_dec). rho·has 는 df 위치 정렬 (ips.pitch_ratios).

    sum_rho = Σ_i ρ_i, sum_rho_q = Σ_i ρ_i·q̂_b(p_i,s_i,a_i), sum_v = Σ_i v̂_e(p_i,s_i), n_dec = 결정 수 (모두 행동 있는 투구만).
    r 은 여기 안 들어간다 — DR1 = Σ(sum_rho·r − sum_rho_q)/Σ sum_rho + Σ sum_v/Σ n_dec 로 나중에 붙인다 (estimate_dr1).
    """
    d, (s_, rho_, has_), inv, keys, _ = _sorted_view(df, state_id, rho, has)
    n = len(keys)
    p_ = d["pitcher_idx"].to_numpy(dtype=np.int64)
    a_ = d["action_id"].to_numpy(dtype=np.int64)
    hd = has_.astype(bool)
    rq = np.zeros(len(d))
    vv = np.zeros(len(d))
    if hd.any():
        rq[hd] = rho_[hd] * q_b[p_[hd], s_[hd], a_[hd]]
        vv[hd] = v_e[p_[hd], s_[hd]]
    return pd.DataFrame({
        "game_pk": keys[:, 0], "at_bat_number": keys[:, 1],
        "sum_rho": np.bincount(inv, weights=np.where(hd, rho_, 0.0), minlength=n),
        "sum_rho_q": np.bincount(inv, weights=rq, minlength=n),
        "sum_v": np.bincount(inv, weights=vv, minlength=n),
        "n_dec": np.bincount(inv, weights=hd.astype(float), minlength=n),
    })


def traj_dr_terms(df: pd.DataFrame, state_id: np.ndarray, rho: np.ndarray, has: np.ndarray, q_e: np.ndarray, v_e: np.ndarray, *, r_pa: np.ndarray | None = None) -> pd.DataFrame:
    """타석별 궤적 DR 조각 (game_pk, at_bat_number, base_plain, base_wdr, w_last, w_last_wdr [, dr_plain, dr_wdr]).

    보상이 종결 투구에만 붙으므로 DR_j = base_j + w_last_j·r_j 로 r 을 분리할 수 있다 (traj_dr_values).
    base_j = v̂_e(s_1) + Σ_t ρ_{1:t}·(v̂_e(s_{t+1}) − q̂_t),  w_last_j = ρ_{1:T}.
    WDR 은 ρ_{1:t} 를 mean_{j: T_j ≥ t} ρ_{1:t}^{(j)} 로 나눈 것 (선두 v̂_e(s_1) 항은 그대로 = w_0 정규화가 1).
    정규화 상수는 넘긴 df 의 모든 타석에서 계산한다 (다투수 타석 제외 전 — 차이는 무시할 수준).
    r_pa 를 주면 (정렬된 타석 키 순서) dr_plain·dr_wdr 도 같이 낸다.
    """
    d, (s_, rho_, has_), inv, keys, pos = _sorted_view(df, state_id, rho, has)
    n = len(keys)
    p_ = d["pitcher_idx"].to_numpy(dtype=np.int64)
    a_ = d["action_id"].to_numpy(dtype=np.int64)
    hd = has_.astype(bool)
    # ρ_{1:t} — [타석, 스텝] 행렬에 흩뿌린 뒤 누적곱 (0 이 섞여도 안전, 타석 길이는 20 남짓)
    T = int(pos.max()) + 1 if n else 0
    M = np.ones((n, T))
    M[inv, pos] = rho_
    C = np.cumprod(M, axis=1)
    seen = np.zeros((n, T), dtype=bool)
    seen[inv, pos] = True
    cnt = seen.sum(0)
    norm = np.divide((C * seen).sum(0), np.where(cnt > 0, cnt, 1), out=np.ones(T), where=cnt > 0)
    w_t = C[inv, pos]
    w_t_wdr = np.where(norm[pos] > 0, w_t / np.where(norm[pos] > 0, norm[pos], 1.0), 0.0)
    # q̂_t, v̂_e(s_{t+1}) — 같은 타석의 실제 다음 투구, 종결 뒤는 0
    q_t = v_e[p_, s_].copy()
    if hd.any():
        q_t[hd] = q_e[p_[hd], s_[hd], a_[hd]]
    same = np.zeros(len(d), dtype=bool)
    same[:-1] = inv[1:] == inv[:-1]
    v_next = np.zeros(len(d))
    v_next[same] = v_e[p_[1:][same[:-1]], s_[1:][same[:-1]]]
    step = v_next - q_t
    last = np.flatnonzero(~same) if len(d) else np.zeros(0, dtype=np.int64)
    first = np.flatnonzero(pos == 0) if len(d) else np.zeros(0, dtype=np.int64)
    v1 = np.zeros(n)
    v1[inv[first]] = v_e[p_[first], s_[first]]
    out = pd.DataFrame({
        "game_pk": keys[:, 0], "at_bat_number": keys[:, 1],
        "base_plain": v1 + np.bincount(inv, weights=w_t * step, minlength=n),
        "base_wdr": v1 + np.bincount(inv, weights=w_t_wdr * step, minlength=n),
        "w_last": np.zeros(n), "w_last_wdr": np.zeros(n),
    })
    out.loc[inv[last], "w_last"] = w_t[last]
    out.loc[inv[last], "w_last_wdr"] = w_t_wdr[last]
    if r_pa is not None:
        if len(r_pa) != n:
            raise ValueError(f"r_pa 길이 {len(r_pa)} ≠ 타석 수 {n}")
        out["dr_plain"], out["dr_wdr"] = traj_dr_values(out, np.asarray(r_pa, dtype=float))
    return out


def traj_dr_values(terms: pd.DataFrame, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dr_plain, dr_wdr) = base + w_last·r. terms 는 traj_dr_terms 결과(필터·병합 뒤여도 됨), r 은 그 행 순서."""
    return (terms["base_plain"].to_numpy() + terms["w_last"].to_numpy() * r,
            terms["base_wdr"].to_numpy() + terms["w_last_wdr"].to_numpy() * r)


# ---------------------------------------------------------------- 추정·부트스트랩
def _weight_diag(w: np.ndarray) -> dict:
    sw = w.sum()
    s2 = (w ** 2).sum()
    return {"ess": float(sw ** 2 / s2) if s2 > 0 else 0.0, "ess_frac": float(sw ** 2 / s2 / len(w)) if s2 > 0 and len(w) else 0.0,
            "w_max": float(w.max()) if len(w) else 0.0, "w_mean": float(w.mean()) if len(w) else 0.0,
            "n_pa": int(len(w)), "frac_zero_w": float((w == 0).mean()) if len(w) else 0.0}


def _parts_dr1(terms: pd.DataFrame, r: np.ndarray):
    """(A, B, C, N) — A = sum_rho·r − sum_rho_q, B = sum_rho, C = sum_v, N = n_dec."""
    B = terms["sum_rho"].to_numpy(dtype=float)
    return B * r - terms["sum_rho_q"].to_numpy(dtype=float), B, terms["sum_v"].to_numpy(dtype=float), terms["n_dec"].to_numpy(dtype=float)


def estimate_dr1(terms: pd.DataFrame, r: np.ndarray) -> dict:
    """1스텝 DR 점추정. snips = DR1 (자기정규화), ips = 같은 값을 결정 수로만 나눈 것, dm = 직접법 항, corr = 보정 항."""
    A, B, C, N = _parts_dr1(terms, np.asarray(r, dtype=float))
    sB, sN = B.sum(), N.sum()
    corr = float(A.sum() / sB) if sB > 0 else float("nan")
    dm = float(C.sum() / sN) if sN > 0 else float("nan")
    return {"ips": float((A.sum() + C.sum()) / sN) if sN > 0 else float("nan"), "snips": corr + dm, "dm": dm, "corr": corr, **_weight_diag(B)}


def bootstrap_dr1(terms: pd.DataFrame, r: np.ndarray, games: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """경기 클러스터 다항 부트스트랩 (IPS.bootstrap 과 같은 방식). est = Σm·A/Σm·B + Σm·C/Σm·N."""
    rng = np.random.default_rng(seed)
    A, B, C, N = _parts_dr1(terms, np.asarray(r, dtype=float))
    ug, inv = np.unique(games, return_inverse=True)
    n_g = len(ug)
    Ag, Bg, Cg, Ng = (np.bincount(inv, weights=x, minlength=n_g) for x in (A, B, C, N))
    est = np.empty(n_boot)
    for b in range(n_boot):
        m = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g)).astype(float)
        db, dn = (m * Bg).sum(), (m * Ng).sum()
        est[b] = ((m * Ag).sum() / db if db > 0 else np.nan) + ((m * Cg).sum() / dn if dn > 0 else np.nan)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "boot_std": float(np.nanstd(est)), "n_boot": int(n_boot), "n_games": int(n_g), "seed": int(seed)}


def estimate_traj_dr(dr_wdr: np.ndarray, dr_plain: np.ndarray, w_last: np.ndarray) -> dict:
    """궤적 DR 점추정. snips = WDR 평균, ips = 정규화 없는 DR 평균. 가중치 진단은 ρ_{1:T} 로."""
    return {"ips": float(np.mean(dr_plain)), "snips": float(np.mean(dr_wdr)), **_weight_diag(np.asarray(w_last, dtype=float))}


def bootstrap_traj_dr(dr: np.ndarray, games: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """타석별 DR 값 평균의 경기 클러스터 부트스트랩. 근사: WDR 정규화 상수는 원 표본 값으로 고정 (재표본마다 다시 계산하지 않음)."""
    rng = np.random.default_rng(seed)
    dr = np.asarray(dr, dtype=float)
    ug, inv = np.unique(games, return_inverse=True)
    n_g = len(ug)
    Sg = np.bincount(inv, weights=dr, minlength=n_g)
    Ng = np.bincount(inv, minlength=n_g).astype(float)
    est = np.empty(n_boot)
    for b in range(n_boot):
        m = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g)).astype(float)
        dn = (m * Ng).sum()
        est[b] = (m * Sg).sum() / dn if dn > 0 else np.nan
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "boot_std": float(np.nanstd(est)), "n_boot": int(n_boot), "n_games": int(n_g), "seed": int(seed)}
