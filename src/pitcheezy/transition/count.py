"""전이 모델 ⓐ — 투수별 집계 + 계층 평활 (design.md ② ⓐ, Phase 0). 산출은 interfaces.tensor.TransitionTensor.

n[p, s, a, o] = 학습 창에서 (투수, 상태, 행동) 뒤에 결과 o 가 난 횟수 (행동 있는 투구만).
P = smoothing.smooth_hierarchical(n, alpha). 규칙 마스크 적용. valid[p, s, a] = 투수의 구종(pitch_id) 학습 투구 수 ≥ repertoire_min.
holdout: 2023–24 학습 → 2025 NLL·ECE. 2025 에만 있는 투수는 집계 제외·목록 기록.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import states as S
from pitcheezy.interfaces.grid import N_ACTIONS, N_LOC, decode_action
from pitcheezy.interfaces.pitch_types import N_PITCH
from pitcheezy.interfaces.tensor import DEFAULT_REPERTOIRE_MIN_PITCHES, DEFAULT_ROW_SUM_TOL, SPEC_VERSION, TransitionTensor
from pitcheezy.transition.smoothing import smooth_hierarchical

ECE_BINS = 15


def count_transitions(df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int) -> np.ndarray:
    """n [P, S, A, O] 정수. action_id ≥ 0, outcome_id ≥ 0 인 투구만. 투수별로 세서 전체 int64 임시 배열을 만들지 않는다 (K=6 이면 7.5GB)."""
    S_ = S.n_states(K)
    ok = (df["action_id"].to_numpy() >= 0) & (df["outcome_id"].to_numpy() >= 0)
    p = df["pitcher_idx"].to_numpy(dtype=np.int64)[ok]
    s = state_id[ok].astype(np.int64)
    a = df["action_id"].to_numpy(dtype=np.int64)[ok]
    o = df["outcome_id"].to_numpy(dtype=np.int64)[ok]
    cell = (s * N_ACTIONS + a) * O.N_OUTCOMES + o
    dtype = np.int16 if S_ * N_ACTIONS * O.N_OUTCOMES * n_pitchers > 2**30 else np.int32
    n = np.zeros((n_pitchers, S_, N_ACTIONS, O.N_OUTCOMES), dtype=dtype)
    order = np.argsort(p, kind="stable")
    p_sorted, cell_sorted = p[order], cell[order]
    bounds = np.searchsorted(p_sorted, np.arange(n_pitchers + 1))
    for i in range(n_pitchers):
        c = cell_sorted[bounds[i]:bounds[i + 1]]
        if len(c):
            cnt = np.bincount(c, minlength=S_ * N_ACTIONS * O.N_OUTCOMES)
            if cnt.max() >= np.iinfo(dtype).max:
                raise OverflowError(f"셀 관측 수가 {dtype} 범위를 넘음")
            n[i] = cnt.reshape(S_, N_ACTIONS, O.N_OUTCOMES).astype(dtype)
    return n


def repertoire_counts(df: pd.DataFrame, n_pitchers: int) -> np.ndarray:
    """[P, N_PITCH] 투수별 구종 투구 수 (행동 있는 투구)."""
    ok = df["action_id"].to_numpy() >= 0
    p = df["pitcher_idx"].to_numpy(dtype=np.int64)[ok]
    g = df["pitch_id"].to_numpy(dtype=np.int64)[ok]
    return np.bincount(p * N_PITCH + g, minlength=n_pitchers * N_PITCH).reshape(n_pitchers, N_PITCH)


def fit(
    df: pd.DataFrame, state_id: np.ndarray, pitchers: pd.DataFrame, K: int, *, alpha: float, alpha_pitcher: float | None = None,
    pitcher_group: str = "pitch", repertoire_min: int = DEFAULT_REPERTOIRE_MIN_PITCHES, valid_states: np.ndarray | None = None, meta: dict | None = None,
) -> TransitionTensor:
    """학습 표 → TransitionTensor. valid_states [S] bool 로 B1 처럼 쓰지 않는 상태 행을 통째로 무효화."""
    n_p = len(pitchers)
    S_ = S.n_states(K)
    n = count_transitions(df, state_id, n_p, K)
    cid = S.decode_state(np.arange(S_), K)[0]
    group = decode_action(np.arange(N_ACTIONS))[0]
    if pitcher_group == "pitch":  # 투수 층 = (카운트, 구종 9)
        pgroup, n_groups = group, N_PITCH
    elif pitcher_group == "coarse":  # 투수 층 = (카운트, 구종×3×3 구역 81) — EXP-P0-006
        from pitcheezy.ope.ips import coarse_groups
        pgroup, n_groups = coarse_groups(), 81
    else:
        raise ValueError(f"pitcher_group 모름: {pitcher_group}")
    P = smooth_hierarchical(
        n, count_of_state=cid, group_of_action=pgroup, n_counts=S.N_COUNTS, n_groups=n_groups,
        alpha=alpha, alpha_pitcher=alpha_pitcher, rule_mask=O.rule_mask_table()[cid],
    )
    rep = repertoire_counts(df, n_p)  # [P, 9]
    valid = (rep >= repertoire_min)[:, group]  # [P, A]
    valid = np.broadcast_to(valid[:, None, :], (n_p, S_, N_ACTIONS)).copy()
    if valid_states is not None:
        valid &= np.asarray(valid_states, dtype=bool)[None, :, None]
    P[~valid] = 0.0
    n_obs = np.empty(n.shape[:3], dtype=np.int32)
    for i in range(n_p):
        n_obs[i] = n[i].sum(-1, dtype=np.int64)
    m = {
        "spec_version": SPEC_VERSION, "model_arch": "count_hierarchical_dirichlet", "K": int(K),
        "repertoire_min_pitches": int(repertoire_min), "row_sum_tol": DEFAULT_ROW_SUM_TOL, "alpha": float(alpha), "alpha_pitcher": None if alpha_pitcher is None else float(alpha_pitcher), "pitcher_group": pitcher_group,
        "n_train_pitches_with_action": int(n.sum()), "holdout_nll": None, "holdout_ece": None, "holdout_ece_hr": None,
        "excluded_pitchers": [],
    }
    if meta:
        m.update(meta)
    return TransitionTensor(P=P.astype(np.float32), valid=valid, n_obs=n_obs, pitchers=pitchers, meta=m)


def holdout_metrics(t: TransitionTensor, df: pd.DataFrame, state_id: np.ndarray) -> dict:
    """홀드아웃 투구의 NLL·ECE. valid 행의 (행동 있음, 결과 있음) 투구만."""
    ok = (df["action_id"].to_numpy() >= 0) & (df["outcome_id"].to_numpy() >= 0)
    p = df["pitcher_idx"].to_numpy(dtype=np.int64)[ok]
    s = state_id[ok].astype(np.int64)
    a = df["action_id"].to_numpy(dtype=np.int64)[ok]
    o = df["outcome_id"].to_numpy(dtype=np.int64)[ok]
    v = t.valid[p, s, a]
    p, s, a, o = p[v], s[v], a[v], o[v]
    probs = np.empty((len(o), O.N_OUTCOMES), dtype=np.float64)
    for i in range(0, len(o), 200_000):  # mmap 텐서에서 청크로 gather
        probs[i:i + 200_000] = t.P[p[i:i + 200_000], s[i:i + 200_000], a[i:i + 200_000]]
    p_obs = probs[np.arange(len(o)), o]
    nll = float(-np.log(np.maximum(p_obs, 1e-12)).mean())
    ece = []
    for k in range(O.N_OUTCOMES):
        ece.append(_ece(probs[:, k], (o == k).astype(float)))
    return {
        "holdout_nll": nll, "holdout_ece": float(np.mean(ece)), "holdout_ece_hr": float(ece[O.HR]),
        "holdout_ece_by_outcome": {O.OUTCOME_NAMES[k]: float(e) for k, e in enumerate(ece)},
        "holdout_n_pitches": int(len(o)), "holdout_n_invalid_rows": int((~v).sum()),
        "holdout_accuracy_top1": float((probs.argmax(1) == o).mean()),
    }


def _ece(conf: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    n = np.bincount(idx, minlength=bins).astype(float)
    sc = np.bincount(idx, weights=conf, minlength=bins)
    sy = np.bincount(idx, weights=y, minlength=bins)
    nz = n > 0
    return float((n[nz] / n.sum() * np.abs(sc[nz] / n[nz] - sy[nz] / n[nz])).sum())
