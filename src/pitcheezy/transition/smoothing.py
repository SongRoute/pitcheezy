"""계층 디리클레 평활 (전이 텐서·π_b 공용). 잠정 결정 D16.

셀 [p, s, a, :] 의 분포를 세 층으로 추정한다 (V = 마지막 축, 결과 O 또는 행동 A):
  L0 league(s, a)      : (n0 + α·base[c(s)]) / (N0 + α)         base = 리그 count 조건부 분포
  L1 pitcher(c, g)     : (n1 + α·league(c, g)) / (N1 + α)       g = 행동의 거친 그룹(구종 pitch_id), 주자아웃·위치 붕괴
  L2 pitcher(s, a)     : (n2 + α·m) / (N2 + α),  m ∝ L0(s,a) × L1(p,c,g) / league(c,g)
      → 리그의 (상태, 행동) 분포에 투수의 (카운트, 구종) 수준 편차를 비율로 곱한 것. 투수 표본이 없으면 m, 많으면 실측.
rule_mask [S, V] (bool) 가 있으면 불허 셀은 모든 층에서 0 으로 두고 재정규화.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def _norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    s = x.sum(axis=axis, keepdims=True)
    return np.where(s > 0, x / np.where(s > 0, s, 1.0), 0.0)


def smooth_hierarchical(
    n: np.ndarray, *, count_of_state: np.ndarray, group_of_action: np.ndarray, n_counts: int, n_groups: int,
    alpha: float, rule_mask: np.ndarray | None = None, out_dtype=np.float32,
) -> np.ndarray:
    """n [P, S, A, V] int → 확률 [P, S, A, V]. 메모리를 위해 투수 축은 루프."""
    P_, S, A, V = n.shape
    n = n.astype(np.float64, copy=False)
    mask = np.ones((S, V), dtype=bool) if rule_mask is None else rule_mask.astype(bool)
    # base: 리그 count 조건부
    n_league_sa = n.sum(axis=0)  # [S, A, V]
    n_league_c = np.zeros((n_counts, V))
    np.add.at(n_league_c, count_of_state, n_league_sa.sum(axis=1))
    base_c = _norm(n_league_c * _mask_c(mask, count_of_state, n_counts))  # [C, V]
    # 표본이 전혀 없는 count (없어야 정상) → 허용 셀 균등
    empty = base_c.sum(-1) == 0
    if empty.any():
        base_c[empty] = _norm(_mask_c(mask, count_of_state, n_counts)[empty].astype(float))
    # L0 league(s, a)
    prior0 = base_c[count_of_state][:, None, :]  # [S, 1, V]
    L0 = (n_league_sa + alpha * prior0) / (n_league_sa.sum(-1, keepdims=True) + alpha)
    L0 = _norm(L0 * mask[:, None, :])
    # league(c, g)
    n_league_cg = np.zeros((n_counts, n_groups, V))
    np.add.at(n_league_cg, (count_of_state[:, None], group_of_action[None, :]), n_league_sa)
    Lcg = (n_league_cg + alpha * base_c[:, None, :]) / (n_league_cg.sum(-1, keepdims=True) + alpha)
    Lcg = _norm(Lcg * _mask_c(mask, count_of_state, n_counts)[:, None, :])
    out = np.empty((P_, S, A, V), dtype=out_dtype)
    for p in range(P_):
        n2 = n[p]  # [S, A, V]
        n1 = np.zeros((n_counts, n_groups, V))
        np.add.at(n1, (count_of_state[:, None], group_of_action[None, :]), n2)
        L1 = (n1 + alpha * Lcg) / (n1.sum(-1, keepdims=True) + alpha)  # [C, G, V]
        ratio = L1 / np.maximum(Lcg, EPS)  # 투수 편차 비율
        m = _norm(L0 * ratio[count_of_state][:, group_of_action, :] * mask[:, None, :])
        L2 = (n2 + alpha * m) / (n2.sum(-1, keepdims=True) + alpha)
        out[p] = _norm(L2 * mask[:, None, :])
    return out


def _mask_c(mask: np.ndarray, count_of_state: np.ndarray, n_counts: int) -> np.ndarray:
    """상태 마스크 [S, V] → count 마스크 [C, V] (같은 count 의 상태는 같은 마스크)."""
    m = np.zeros((n_counts, mask.shape[1]), dtype=bool)
    for c in range(n_counts):
        rows = mask[count_of_state == c]
        m[c] = rows.any(axis=0) if len(rows) else True
    return m
