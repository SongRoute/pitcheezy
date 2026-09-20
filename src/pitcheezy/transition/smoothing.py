"""계층 디리클레 평활 (전이 텐서·π_b 공용). 잠정 결정 D16.

셀 [p, s, a, :] 의 분포 (V = 마지막 축, 결과 O 또는 행동 A) 를 아래 층으로 추정한다. 각 층 = (n + α·상위층) / (N + α).
  base   league(c)         리그 count 조건부
  Lca    league(c, a)      ← base            주자아웃을 뺀 리그 (상태×행동 셀보다 표본 24배)
  L0     league(s, a)      ← Lca[c(s)]
  Lcg    league(c, g)      ← base            g = 행동의 거친 그룹(구종)
  L1     pitcher(c, g)     ← Lcg             투수 수준은 (카운트, 구종) 로 거칠게
  L2     pitcher(s, a)     ← m ∝ L0(s,a) × L1(p,c,g) / Lcg(c,g)   리그 (상태,행동) 분포에 투수의 (카운트,구종) 편차를 비율로
rule_mask [S, V] (bool) 가 있으면 불허 셀은 모든 층에서 0 으로 두고 재정규화.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def _norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    s = x.sum(axis=axis, keepdims=True)
    return np.where(s > 0, x / np.where(s > 0, s, 1.0), 0.0)


def _shrink(n: np.ndarray, prior: np.ndarray, alpha: float, mask=None) -> np.ndarray:
    out = (n + alpha * prior) / (n.sum(-1, keepdims=True) + alpha)
    return _norm(out * mask) if mask is not None else _norm(out)


class LeagueLayers:
    """리그 층(base·Lca·L0·Lcg)과 마스크. 투수 축과 무관 → 한 번 만들고 pitcher_layer 가 투수마다 쓴다."""

    def __init__(self, n_sa: np.ndarray, *, count_of_state: np.ndarray, group_of_action: np.ndarray, n_counts: int, n_groups: int, alpha: float, rule_mask: np.ndarray | None = None):
        S, A, V = n_sa.shape
        self.cs, self.group, self.n_counts, self.n_groups = count_of_state, group_of_action, n_counts, n_groups
        self.mask_s = np.ones((S, V), dtype=bool) if rule_mask is None else rule_mask.astype(bool)
        self.mask_c = mask_c = _mask_c(self.mask_s, count_of_state, n_counts)
        cs = count_of_state
        n_c = np.zeros((n_counts, V)); np.add.at(n_c, cs, n_sa.sum(axis=1))
        base = _norm(n_c * mask_c)
        empty = base.sum(-1) == 0
        if empty.any():
            base[empty] = _norm(mask_c[empty].astype(float))
        # Lca league(c, a)
        n_ca = np.zeros((n_counts, A, V)); np.add.at(n_ca, cs, n_sa)
        Lca = _shrink(n_ca, base[:, None, :], alpha, mask_c[:, None, :])
        # L0 league(s, a)
        self.L0 = _shrink(n_sa, Lca[cs], alpha, self.mask_s[:, None, :])
        # Lcg league(c, g)
        n_cg = np.zeros((n_counts, n_groups, V)); np.add.at(n_cg, (cs[:, None], group_of_action[None, :]), n_sa)
        self.Lcg = _shrink(n_cg, base[:, None, :], alpha, mask_c[:, None, :])


def pitcher_layer(n2: np.ndarray, lg: LeagueLayers, alpha_pitcher: float) -> np.ndarray:
    """투수 한 명의 n2 [S, A, V] float64 → L2 [S, A, V] float64."""
    cs, group = lg.cs, lg.group
    n1 = np.zeros((lg.n_counts, lg.n_groups, n2.shape[-1])); np.add.at(n1, (cs[:, None], group[None, :]), n2)
    L1 = _shrink(n1, lg.Lcg, alpha_pitcher, lg.mask_c[:, None, :])
    ratio = L1 / np.maximum(lg.Lcg, EPS)
    m = _norm(lg.L0 * ratio[cs][:, group, :] * lg.mask_s[:, None, :])
    return _shrink(n2, m, alpha_pitcher, lg.mask_s[:, None, :])


def smooth_hierarchical(
    n: np.ndarray, *, count_of_state: np.ndarray, group_of_action: np.ndarray, n_counts: int, n_groups: int,
    alpha: float, rule_mask: np.ndarray | None = None, out_dtype=np.float32, league_only: bool = False, alpha_pitcher: float | None = None,
) -> np.ndarray:
    """n [P, S, A, V] int → 확률 [P, S, A, V]. 투수 축은 루프 (메모리). league_only=True 면 L0 를 [1, S, A, V] 로 돌려준다.
    alpha 는 리그 층(Lca·L0·Lcg), alpha_pitcher 는 투수 층(L1·L2). None 이면 같은 값 (EXP-P0-005 이전 동작).
    dense n 을 못 올리는 크기는 LeagueLayers + pitcher_layer 를 직접 쓴다 (transition/count.py, D37)."""
    ap = alpha if alpha_pitcher is None else float(alpha_pitcher)
    P_, S, A, V = n.shape
    # 리그 합은 투수별 누적 (전체 float64 캐스트 금지: K=6 이면 7.5GB)
    n_sa = np.zeros((S, A, V), dtype=np.float64)
    for p in range(P_):
        n_sa += n[p]
    lg = LeagueLayers(n_sa, count_of_state=count_of_state, group_of_action=group_of_action, n_counts=n_counts, n_groups=n_groups, alpha=alpha, rule_mask=rule_mask)
    if league_only:
        return lg.L0[None].astype(out_dtype)
    out = np.empty((P_, S, A, V), dtype=out_dtype)
    for p in range(P_):
        out[p] = pitcher_layer(n[p].astype(np.float64), lg, ap)
    return out


def _mask_c(mask: np.ndarray, count_of_state: np.ndarray, n_counts: int) -> np.ndarray:
    m = np.zeros((n_counts, mask.shape[1]), dtype=bool)
    for c in range(n_counts):
        rows = mask[count_of_state == c]
        m[c] = rows.any(axis=0) if len(rows) else True
    return m
