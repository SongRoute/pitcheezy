"""가치 반복 (타석 MDP, γ=1) → interfaces.value.ValueBundle. design.md §1-e, §3 Phase 0.

보상(투수 관점) R[s, o] = −dRE24[o, base_out(s)] (종결) / 0 (비종결. 중간 shaping 은 P0 미사용 — 최적 정책 불변)
Q[p, s, a] = Σ_o P[p, s, a, o] · (R[s, o] + V[p, next(s, o)]),  V[p, s] = max_{a valid} Q[p, s, a],  종결 다음은 0.
2스트라이크 파울은 자기 전이 → 고정점까지 반복 (수축: 파울 확률 < 1).
valid 행동이 없는 상태는 V = 0 (플래그로 기록).
완화: softmax(Q/τ) 또는 top-k 균등, valid 위에서만.
"""

from __future__ import annotations

import numpy as np

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import states as S


def reward_table(dRE24: np.ndarray, K: int) -> np.ndarray:
    """R [S, O] 투수 관점 (= −ΔRE24). dRE24 [8, 24] 는 outcomes.TERMINAL 순서."""
    S_ = S.n_states(K)
    _, bid, _ = S.decode_state(np.arange(S_), K)
    R = np.zeros((S_, O.N_OUTCOMES), dtype=np.float64)
    for i, o in enumerate(O.TERMINAL):
        R[:, o] = -dRE24[i, bid]
    return R


def next_state_table(K: int) -> np.ndarray:
    """next [S, O] 다음 state_id, 종결·불허면 −1."""
    S_ = S.n_states(K)
    cid, bid, kid = S.decode_state(np.arange(S_), K)
    nxt = np.full((S_, O.N_OUTCOMES), -1, dtype=np.int64)
    for s in range(S_):
        for o in O.NONTERMINAL:
            try:
                c2 = O.next_count(int(cid[s]), o)
            except ValueError:
                continue  # 규칙상 불가 (P 도 0)
            nxt[s, o] = S.state_id(c2, int(bid[s]), int(kid[s]), K)
    return nxt


def value_iteration(P: np.ndarray, valid: np.ndarray, R: np.ndarray, nxt: np.ndarray, *, tol: float = 1e-9, max_iter: int = 200, chunk: int = 16):
    """P [P,S,A,O] float32 (mmap 가능), valid [P,S,A] → Q [P,S,A], V [P,S] float64, iters, max_delta. 투수는 서로 독립이라 청크로."""
    n_p, S_, A, O_ = P.shape
    Q = np.empty((n_p, S_, A))
    V = np.zeros((n_p, S_))
    nxt_safe = np.where(nxt >= 0, nxt, 0)
    is_next = (nxt >= 0)[None, :, :]
    iters_max, delta_max = 0, 0.0
    for lo in range(0, n_p, chunk):
        hi = min(lo + chunk, n_p)
        Pf = np.asarray(P[lo:hi], dtype=np.float64)
        vd = valid[lo:hi]
        has_valid = vd.any(-1)
        Vc = np.zeros((hi - lo, S_))
        for it in range(max_iter):
            Vn = np.where(is_next, Vc[:, nxt_safe], 0.0)
            Qc = np.einsum("psao,pso->psa", Pf, R[None] + Vn)
            Vnew = np.where(has_valid, np.where(vd, Qc, -np.inf).max(-1), 0.0)
            delta = float(np.abs(Vnew - Vc).max())
            Vc = Vnew
            if delta < tol:
                break
        Q[lo:hi], V[lo:hi] = Qc, Vc
        iters_max, delta_max = max(iters_max, it + 1), max(delta_max, delta)
    return Q, V, iters_max, delta_max


def relax(Q: np.ndarray, valid: np.ndarray, *, method: str = "softmax", temperature: float = 0.05, top_k: int = 5) -> np.ndarray:
    """완화 정책 [P,S,A] float32. valid 밖 0, valid 있는 상태는 합 1."""
    if method == "softmax":
        z = np.where(valid, Q / temperature, -np.inf)
        z = z - np.where(valid.any(-1, keepdims=True), z.max(-1, keepdims=True), 0.0)
        e = np.where(valid, np.exp(z), 0.0)
    elif method == "topk":
        z = np.where(valid, Q, -np.inf)
        kth = -np.sort(-z, axis=-1)[..., min(top_k, z.shape[-1]) - 1: min(top_k, z.shape[-1])]
        e = (valid & (z >= kth)).astype(np.float64)
    elif method == "greedy":
        z = np.where(valid, Q, -np.inf)
        e = (valid & (z >= z.max(-1, keepdims=True))).astype(np.float64)
    elif method == "uniform":
        e = valid.astype(np.float64)
    else:
        raise ValueError(f"알 수 없는 완화: {method}")
    s = e.sum(-1, keepdims=True)
    return np.where(s > 0, e / np.where(s > 0, s, 1.0), 0.0).astype(np.float32)


def policy_evaluation(P: np.ndarray, policy: np.ndarray, R: np.ndarray, nxt: np.ndarray, *, tol: float = 1e-9, max_iter: int = 200, chunk: int = 16) -> np.ndarray:
    """모델 안에서 고정 정책의 가치 V^π [P, S] (진단: 모델이 말하는 값 vs OPE 가 말하는 값 → 착취 폭)."""
    n_p, S_, A, O_ = P.shape
    Pmix = np.empty((n_p, S_, O_))
    for lo in range(0, n_p, chunk):
        hi = min(lo + chunk, n_p)
        Pmix[lo:hi] = np.einsum("psao,psa->pso", np.asarray(P[lo:hi], dtype=np.float64), policy[lo:hi].astype(np.float64))
    V = np.zeros((n_p, S_))
    nxt_safe = np.where(nxt >= 0, nxt, 0)
    is_next = (nxt >= 0)[None]
    for _ in range(max_iter):
        Vn = np.where(is_next, V[:, nxt_safe], 0.0)
        Vnew = (Pmix * (R[None] + Vn)).sum(-1)
        delta = float(np.abs(Vnew - V).max())
        V = Vnew
        if delta < tol:
            break
    return V
