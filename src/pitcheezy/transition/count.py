"""전이 모델 ⓐ — 투수별 집계 + 계층 평활 (design.md ② ⓐ, Phase 0). 산출은 interfaces.tensor.TransitionTensor.

n[p, s, a, o] = 학습 창에서 (투수, 상태, 행동) 뒤에 결과 o 가 난 횟수 (행동 있는 투구만).
P = 계층 평활(smoothing.LeagueLayers + pitcher_layer, smooth_hierarchical 과 같은 값). 규칙 마스크 적용. valid[p, s, a] = 투수의 구종(pitch_id) 학습 투구 수 ≥ repertoire_min.
fit 은 투수 한 명씩 세고 평활해 바로 쓴다 — dense n[P,S,A,O] 를 만들지 않는다. out_dir 을 주면 P·n_obs 를 그 디렉터리의 npy memmap 에 쓴다 (K=6×C=7 은 P 26GB, D37).
holdout: 2023–24 학습 → 2025 NLL·ECE. 2025 에만 있는 투수는 집계 제외·목록 기록.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import states as S
from pitcheezy.interfaces.grid import N_ACTIONS, N_LOC, decode_action
from pitcheezy.interfaces.pitch_types import N_PITCH
from pitcheezy.interfaces.tensor import DEFAULT_REPERTOIRE_MIN_PITCHES, DEFAULT_ROW_SUM_TOL, SPEC_VERSION, TransitionTensor
from pitcheezy.transition.smoothing import LeagueLayers, pitcher_layer

ECE_BINS = 15


def count_transitions(df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, K: int, C: int = 1) -> np.ndarray:
    """n [P, S, A, O] 정수. action_id ≥ 0, outcome_id ≥ 0 인 투구만. 투수별로 세서 전체 int64 임시 배열을 만들지 않는다 (K=6 이면 7.5GB)."""
    S_ = S.n_states(K, C)
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


class _PitcherCells:
    """(투수별로 정렬한) 셀 번호. dense n[P,S,A,O] 없이 리그 합과 투수 한 명의 n[S,A,O] 를 만든다."""

    def __init__(self, df: pd.DataFrame, state_id: np.ndarray, n_pitchers: int, S_: int):
        ok = (df["action_id"].to_numpy() >= 0) & (df["outcome_id"].to_numpy() >= 0)
        p = df["pitcher_idx"].to_numpy(dtype=np.int64)[ok]
        cell = (state_id[ok].astype(np.int64) * N_ACTIONS + df["action_id"].to_numpy(dtype=np.int64)[ok]) * O.N_OUTCOMES + df["outcome_id"].to_numpy(dtype=np.int64)[ok]
        order = np.argsort(p, kind="stable")
        self.cell = cell[order]
        self.bounds = np.searchsorted(p[order], np.arange(n_pitchers + 1))
        self.shape = (S_, N_ACTIONS, O.N_OUTCOMES)
        self.size = S_ * N_ACTIONS * O.N_OUTCOMES

    def league(self) -> np.ndarray:
        return np.bincount(self.cell, minlength=self.size).reshape(self.shape).astype(np.float64)

    def pitcher(self, i: int) -> np.ndarray:
        return np.bincount(self.cell[self.bounds[i]:self.bounds[i + 1]], minlength=self.size).reshape(self.shape)


def _alloc(out_dir: Path | None, name: str, shape: tuple, dtype) -> np.ndarray:
    if out_dir is None:
        return np.empty(shape, dtype=dtype)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(Path(out_dir) / name, mode="w+", dtype=dtype, shape=shape)


def repertoire_counts(df: pd.DataFrame, n_pitchers: int) -> np.ndarray:
    """[P, N_PITCH] 투수별 구종 투구 수 (행동 있는 투구)."""
    ok = df["action_id"].to_numpy() >= 0
    p = df["pitcher_idx"].to_numpy(dtype=np.int64)[ok]
    g = df["pitch_id"].to_numpy(dtype=np.int64)[ok]
    return np.bincount(p * N_PITCH + g, minlength=n_pitchers * N_PITCH).reshape(n_pitchers, N_PITCH)


def fit(
    df: pd.DataFrame, state_id: np.ndarray, pitchers: pd.DataFrame, K: int, *, alpha: float, alpha_pitcher: float | None = None,
    pitcher_group: str = "pitch", repertoire_min: int = DEFAULT_REPERTOIRE_MIN_PITCHES, valid_states: np.ndarray | None = None, meta: dict | None = None,
    C: int = 1, context_kind: str | None = None, out_dir: Path | None = None,
) -> TransitionTensor:
    """학습 표 → TransitionTensor. valid_states [S] bool 로 B1 처럼 쓰지 않는 상태 행을 통째로 무효화.

    out_dir 을 주면 P·n_obs 는 out_dir/P.npy·n_obs.npy 의 memmap (그 뒤 t.save(out_dir) 는 두 파일을 다시 쓰지 않는다). 값은 out_dir 유무와 무관하게 같다.
    """
    n_p = len(pitchers)
    S_ = S.n_states(K, C)
    cells = _PitcherCells(df, state_id, n_p, S_)
    cid = S.decode_state_full(np.arange(S_), K, C)[0]
    group = decode_action(np.arange(N_ACTIONS))[0]
    if pitcher_group == "pitch":  # 투수 층 = (카운트, 구종 9)
        pgroup, n_groups = group, N_PITCH
    elif pitcher_group == "coarse":  # 투수 층 = (카운트, 구종×3×3 구역 81) — EXP-P0-006
        from pitcheezy.ope.ips import coarse_groups
        pgroup, n_groups = coarse_groups(), 81
    elif pitcher_group == "action":  # 투수 층 = (카운트, 행동 225) — EXP-P0-008. 주자아웃만 붕괴
        pgroup, n_groups = np.arange(N_ACTIONS), N_ACTIONS
    else:
        raise ValueError(f"pitcher_group 모름: {pitcher_group}")
    lg = LeagueLayers(cells.league(), count_of_state=cid, group_of_action=pgroup, n_counts=S.N_COUNTS, n_groups=n_groups, alpha=alpha, rule_mask=O.rule_mask_table()[cid])
    ap = alpha if alpha_pitcher is None else float(alpha_pitcher)
    rep = repertoire_counts(df, n_p)  # [P, 9]
    valid = (rep >= repertoire_min)[:, group]  # [P, A]
    valid = np.broadcast_to(valid[:, None, :], (n_p, S_, N_ACTIONS)).copy()
    if valid_states is not None:
        valid &= np.asarray(valid_states, dtype=bool)[None, :, None]
    P = _alloc(out_dir, "P.npy", (n_p, S_, N_ACTIONS, O.N_OUTCOMES), np.float32)
    n_obs = _alloc(out_dir, "n_obs.npy", (n_p, S_, N_ACTIONS), np.int32)
    n_total = 0
    for i in range(n_p):
        n2 = cells.pitcher(i)
        row = pitcher_layer(n2.astype(np.float64), lg, ap).astype(np.float32)
        row[~valid[i]] = 0.0
        P[i] = row
        n_obs[i] = n2.sum(-1)
        n_total += int(n2.sum())
    m = {
        "spec_version": SPEC_VERSION, "model_arch": "count_hierarchical_dirichlet", "K": int(K), "C": int(C), "context_kind": context_kind,
        "repertoire_min_pitches": int(repertoire_min), "row_sum_tol": DEFAULT_ROW_SUM_TOL, "alpha": float(alpha), "alpha_pitcher": None if alpha_pitcher is None else float(alpha_pitcher), "pitcher_group": pitcher_group,
        "n_train_pitches_with_action": n_total, "holdout_nll": None, "holdout_ece": None, "holdout_ece_hr": None,
        "excluded_pitchers": [],
    }
    if meta:
        m.update(meta)
    return TransitionTensor(P=P, valid=valid, n_obs=n_obs, pitchers=pitchers, meta=m)


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
