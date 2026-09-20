"""OPE 재료를 투수 블록 단위로 만든다 (D37).

OPE 가 쓰는 [P,S,A] 배열(π_b·π_e·q̂_b·q̂_e)은 투수끼리 독립이고, 추정량에 들어가는 것은 로그된 투구 행의 값뿐이다.
→ 투수 BLOCK 명씩 [B,S,A] 로 만들고 행 단위 벡터만 남긴다. K=6×C=7 에서 [P,S,A] float64 하나가 4.8GB 라 통째로는 못 올린다.
블록 안 계산은 기존 함수(BH.tilt·IPS.restrict_support·VI.policy_evaluation·DR.dr_inputs …)를 그대로 쓴다. BLOCK 은 그 함수들의 chunk(16)와 같아
맥락 C>1 경로는 통짜 계산과 같은 반복 횟수로 멈춘다 (C=1 의 policy_evaluation 은 전 투수 공통 수렴 판정이라 tol 1e-9 안에서만 같다).

정책 명세 spec: ("tilt", τ) π_b·exp(Q/τ) — 교차 적합 폴드의 π_b 로 만든다 / ("relax", method, kwargs) VI.relax(Q) / ("stored",) value/policy.npy
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitcheezy.ope import behavior as BH
from pitcheezy.ope import dr as DR
from pitcheezy.ope import ips as IPS
from pitcheezy.policy import vi as VI

BLOCK = 16
BLOCK_BYTES = 1e9  # 블록의 P 를 float64 로 올린 크기 상한. VI·policy_evaluation 은 이 크기 임시 배열을 3~4개 든다


def block_size(n_states: int, n_actions: int, n_outcomes: int) -> int:
    """투수 블록 크기. 지금까지의 실험 크기(S ≤ 2016)는 16 그대로 (= VI chunk, 값 불변), K=6×C=7 (S=12096) 은 4."""
    return int(max(1, min(BLOCK, BLOCK_BYTES // (n_states * n_actions * n_outcomes * 8))))


@dataclass
class PolicyRows:
    pe: np.ndarray  # [N] π_e(로그 행동). 행동 없으면 NaN
    pe_coarse: np.ndarray | None = None
    V_row: np.ndarray | None = None  # V_e[p_i, s_i] (모델 내 가치·궤적 DR)
    qe_row: np.ndarray | None = None  # q̂_e[p_i, s_i, a_i]
    veb_row: np.ndarray | None = None  # Σ_a π_e·q̂_b [p_i, s_i] (1스텝 DR 직접법 항)


@dataclass
class Rows:
    pb: np.ndarray  # [N] 교차 적합 π_b(로그 행동)
    pb_coarse: np.ndarray | None
    policies: dict[str, PolicyRows] = field(default_factory=dict)
    qb_row: np.ndarray | None = None  # q̂_b[p_i, s_i, a_i]
    n_support: int = 0
    n_valid: int = 0


_POLICY_FIELDS = ("pe", "pe_coarse", "V_row", "qe_row", "veb_row")


def save_rows(path, rows: Rows, key: str) -> None:
    """행 단위 재료 캐시. 집계 모델은 시드가 OPE 부트스트랩에만 쓰여 재료가 시드와 무관 → s0 에 한 번 저장하고 다른 시드는 읽는다."""
    arrs = {"pb": rows.pb, "n": np.array([rows.n_support, rows.n_valid]), "key": np.array(key), "names": np.array(list(rows.policies))}
    for k in ("pb_coarse", "qb_row"):
        if getattr(rows, k) is not None:
            arrs[k] = getattr(rows, k)
    for i, pr in enumerate(rows.policies.values()):
        for k in _POLICY_FIELDS:
            if getattr(pr, k) is not None:
                arrs[f"p{i}_{k}"] = getattr(pr, k)
    np.savez(path, **arrs)


def load_rows(path, key: str) -> Rows | None:
    """캐시가 없거나 key(OPE 설정·정책 메뉴·전이/가치 해시)가 다르면 None."""
    try:
        z = np.load(path, allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if str(z["key"]) != key:
        return None
    rows = Rows(pb=z["pb"], pb_coarse=z["pb_coarse"] if "pb_coarse" in z else None, qb_row=z["qb_row"] if "qb_row" in z else None, n_support=int(z["n"][0]), n_valid=int(z["n"][1]))
    for i, name in enumerate(z["names"].tolist()):
        rows.policies[name] = PolicyRows(**{k: (z[f"p{i}_{k}"] if f"p{i}_{k}" in z else None) for k in _POLICY_FIELDS})
    return rows


def parse_spec(name: str):
    """메뉴 이름 → spec. behavior 는 None."""
    if name == "behavior":
        return None
    if name.startswith("tilt_t"):
        return ("tilt", float(name[6:]))
    if name == "uniform" or name == "greedy":
        return ("relax", name, {})
    if name.startswith("topk"):
        return ("relax", "topk", {"top_k": int(name[4:])})
    if name.startswith("softmax_t"):
        return ("relax", "softmax", {"temperature": float(name[9:])})
    raise ValueError(f"메뉴 항목 모름: {name}")


def compute_rows(
    ev: pd.DataFrame, sid: np.ndarray, n_pitchers: int, K: int, C: int, *, specs: dict[str, tuple], valid: np.ndarray, rep_ok: np.ndarray, Q: np.ndarray,
    alpha: float, n_folds: int, groups: np.ndarray | None = None, stored_policy: np.ndarray | None = None,
    P: np.ndarray | None = None, R: np.ndarray | None = None, nxt: np.ndarray | None = None, with_value: bool = False, with_dr: bool = False, block: int | None = None,
) -> Rows:
    """ev 행 순서의 행 단위 OPE 재료. valid·Q·stored_policy·P 는 mmap 이어도 된다 (블록만 실체화).

    rep_ok [P, A] = 평가 시즌 레퍼토리 (공통 지지 = valid ∩ rep_ok). with_value 는 V_e (P·R·nxt 필요), with_dr 는 거기에 q̂_b·q̂_e·v̂_e_b 까지.
    """
    if (with_value or with_dr) and (P is None or R is None or nxt is None):
        raise ValueError("with_value·with_dr 는 P·R·nxt 가 필요함")
    if block is None:
        block = block_size(valid.shape[1], valid.shape[2], P.shape[3] if P is not None else 1) if P is not None else BLOCK
    N = len(ev)
    p_all = ev["pitcher_idx"].to_numpy(dtype=np.int64)
    a_all = ev["action_id"].to_numpy(dtype=np.int64)
    s_all = sid.astype(np.int64)
    nf = max(n_folds, 1)
    fold = ev["game_pk"].to_numpy(dtype=np.int64) % nf
    fac = []
    for k in range(nf):
        tr = fold != k if n_folds > 1 else np.ones(N, dtype=bool)
        fac.append(BH.behavior_factors(ev[tr], sid[tr], n_pitchers, K, alpha=alpha, C=C))
    fac_full = BH.behavior_factors(ev, sid, n_pitchers, K, alpha=alpha, C=C) if (with_value or with_dr) else None

    nan = lambda: np.full(N, np.nan)  # noqa: E731
    out = Rows(pb=nan(), pb_coarse=nan() if groups is not None else None, qb_row=np.zeros(N) if with_dr else None)
    for name in specs:
        out.policies[name] = PolicyRows(pe=nan(), pe_coarse=nan() if groups is not None else None, V_row=np.zeros(N) if (with_value or with_dr) else None,
                                        qe_row=np.zeros(N) if with_dr else None, veb_row=np.zeros(N) if with_dr else None)
    order = np.argsort(p_all, kind="stable")
    bounds = np.searchsorted(p_all[order], np.arange(n_pitchers + 1))
    for lo in range(0, n_pitchers, block):
        hi = min(lo + block, n_pitchers)
        rows = np.sort(order[bounds[lo]:bounds[hi]])  # 이 블록 투수의 행 (ev 순서)
        vd = np.asarray(valid[lo:hi])
        sup = vd & rep_ok[lo:hi][:, None, :]
        out.n_support += int(sup.sum()); out.n_valid += int(vd.sum())
        if len(rows) == 0:
            continue
        pl, sl, al = p_all[rows] - lo, s_all[rows], a_all[rows]
        has = al >= 0
        Qb = np.asarray(Q[lo:hi])
        # --- 교차 적합: 폴드 k 의 π_b 로 폴드 k 의 시험 행
        for k in range(nf):
            te = has & (fold[rows] == k) if n_folds > 1 else has
            if not te.any():
                continue
            pbk = fac[k].block(lo, hi)
            idx = (pl[te], sl[te], al[te])
            out.pb[rows[te]] = pbk[idx]
            if groups is not None:
                out.pb_coarse[rows[te]] = IPS.coarsen(pbk, groups)[idx]
            for name, spec in specs.items():
                if spec[0] != "tilt":
                    continue
                pe = BH.tilt(pbk, Qb, sup, spec[1])
                out.policies[name].pe[rows[te]] = pe[idx]
                if groups is not None:
                    out.policies[name].pe_coarse[rows[te]] = IPS.coarsen(pe, groups)[idx]
        # --- 정책 배열 (전체 표본 π_b) → 비-tilt 의 π_e(로그), 모델 내 가치, DR
        need_full = with_value or with_dr
        pbf = fac_full.block(lo, hi) if need_full else None
        Pc = np.asarray(P[lo:hi]) if need_full else None
        ev_b = ev.iloc[rows] if with_dr else None
        q_b = None
        if with_dr:
            q_b = DR.behavior_q(Pc, pbf, sup, R, nxt)
            out.qb_row[rows] = DR.gather_rows(ev_b, sl, q_b, np.zeros(q_b.shape[:2]), lo=lo)[0]
        for name, spec in specs.items():
            pr = out.policies[name]
            if spec[0] == "tilt":
                pol = BH.tilt(pbf, Qb, sup, spec[1]) if need_full else None
            else:
                base = np.asarray(stored_policy[lo:hi]) if spec[0] == "stored" else VI.relax(Qb.astype(np.float64), vd, method=spec[1], **spec[2])
                pol = IPS.restrict_support(base, sup)
                idx = (pl[has], sl[has], al[has])
                pr.pe[rows[has]] = pol[idx]
                if groups is not None:
                    pr.pe_coarse[rows[has]] = IPS.coarsen(pol, groups)[idx]
            if not need_full:
                continue
            V_e = VI.policy_evaluation(Pc, pol, R, nxt)
            pr.V_row[rows] = V_e[pl, sl]
            if with_dr:
                v_e_b, _, q_e = DR.dr_inputs(Pc, pol.astype(np.float32), q_b, R, nxt, V_e=V_e)
                pr.qe_row[rows] = DR.gather_rows(ev_b, sl, q_e, V_e, lo=lo)[0]
                pr.veb_row[rows] = v_e_b[pl, sl]
    return out
