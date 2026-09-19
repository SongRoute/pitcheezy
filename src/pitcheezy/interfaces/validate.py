"""검증 계약 (tests/interfaces 가 이 함수를 호출한다). docs/interface-spec.md "검증 계약".

전이 텐서: 형상·dtype / NaN 없음 / valid 행 합 1±tol / valid=False 행 합 0 / 규칙 마스크 셀 0 / 룩업 표 크기 = 축 길이 /
          meta 필수 키 / sha256 일치(load 시)
가치함수: 형상·dtype / NaN 없음 / V = valid 행동 위 max Q / policy 는 valid 위 분포(합 1, 비valid 0) / meta 필수 키
"""

from __future__ import annotations

import numpy as np

from . import tensor as T
from . import value as V
from .grid import ACTIONS_COLUMNS, N_ACTIONS  # noqa: F401 - re-export for tests
from .outcomes import N_OUTCOMES, OUTCOMES_COLUMNS, rule_mask_table
from .states import N_COUNT_BASE_OUT, STATES_COLUMNS, STATES_COLUMNS_V1, decode_state_full, n_states

P_CHUNK = 16  # 행 합·마스크 검사의 투수 청크 (봉우리 메모리)


def validate_transition(t: T.TransitionTensor, *, row_sum_tol: float | None = None) -> list[str]:
    """계약 위반 목록. 비어 있으면 통과. 재정규화 없음 (실패는 실패)."""
    p: list[str] = []
    meta = t.meta if isinstance(t.meta, dict) else {}

    missing = [k for k in T.META_REQUIRED_KEYS if k not in meta]
    if missing:
        p.append(f"meta 필수 키 없음: {missing}")
    if "K" not in meta:
        return p + ["meta.K 없음 — 형상 검증 불가"]
    K = int(meta["K"])
    C = t.C
    tol = float(meta.get("row_sum_tol", T.DEFAULT_ROW_SUM_TOL)) if row_sum_tol is None else row_sum_tol

    n_p = t.P.shape[0] if t.P.ndim == 4 else -1
    shape = T.expected_shape(max(n_p, 0), K, C)
    if t.P.shape != shape:
        p.append(f"P 형상 {t.P.shape} ≠ {shape}")
    if t.P.dtype != np.float32:
        p.append(f"P dtype {t.P.dtype} ≠ float32")
    if t.valid.shape != shape[:3]:
        p.append(f"valid 형상 {t.valid.shape} ≠ {shape[:3]}")
    if t.valid.dtype != np.bool_:
        p.append(f"valid dtype {t.valid.dtype} ≠ bool")
    if t.n_obs.shape != shape[:3]:
        p.append(f"n_obs 형상 {t.n_obs.shape} ≠ {shape[:3]}")
    if t.n_obs.dtype != np.int32:
        p.append(f"n_obs dtype {t.n_obs.dtype} ≠ int32")
    if p:
        return p  # 형상이 틀리면 아래 검사는 의미 없음

    # P 전체 검사(NaN·음수·행 합·규칙 마스크)는 투수 청크로 돈다.
    # 전체 텐서를 float64 로 복사하면 C=7 에서 9GB → 봉우리 메모리를 O(chunk × S × A × O) 로 묶는다. (D35)
    S = n_states(K, C)
    count_of_state = decode_state_full(np.arange(S), K, C)[0]
    allowed = rule_mask_table()[count_of_state]  # [S, O]
    has_nan = has_neg = False
    n_bad_valid = n_bad_invalid = n_viol = 0
    max_dev = 0.0
    for lo in range(0, n_p, P_CHUNK):
        hi = min(lo + P_CHUNK, n_p)
        blk = np.asarray(t.P[lo:hi])  # mmap 이면 여기서만 실체화
        has_nan = has_nan or bool(np.isnan(blk).any())
        has_neg = has_neg or bool((blk < 0).any())
        row = blk.sum(axis=-1, dtype=np.float64)
        vd = t.valid[lo:hi]
        n_bad_valid += int((vd & (np.abs(row - 1.0) > tol)).sum())
        if vd.any():
            max_dev = max(max_dev, float(np.abs(row[vd] - 1).max()))
        n_bad_invalid += int(((~vd) & (row != 0.0)).sum())
        n_viol += int(((blk != 0) & ~allowed[None, :, None, :]).sum())
    if has_nan:
        p.append("P 에 NaN")
    if has_neg:
        p.append("P 에 음수")
    if (t.n_obs < 0).any():
        p.append("n_obs 에 음수")
    if n_bad_valid:
        p.append(f"valid 행 합 1±{tol} 위반 {n_bad_valid}개 (최대 편차 {max_dev:.2e})")
    if n_bad_invalid:
        p.append(f"valid=False 행 합 0 위반 {n_bad_invalid}개")
    if n_viol:
        p.append(f"규칙 마스크 셀 ≠ 0: {n_viol}개")

    # 룩업 표
    cols = tuple(t.states.columns)
    # 맥락 이전(v1) 산출물은 context_id 열이 없다 → C=1 이면 그대로 통과 (context_id ≡ 0). C>1 은 5열을 요구한다.
    cols_ok = cols == STATES_COLUMNS or (C == 1 and cols == STATES_COLUMNS_V1)
    if not cols_ok or len(t.states) != S:
        want = list(STATES_COLUMNS) + ([f"(또는 v1 {list(STATES_COLUMNS_V1)})"] if C == 1 else [])
        p.append(f"states 표 {list(cols)}×{len(t.states)} ≠ {want}×{S}")
    elif not (t.states["state_id"].to_numpy() == np.arange(S)).all():
        p.append("states.state_id 가 0..S−1 순서가 아님")
    if tuple(t.outcomes.columns) != OUTCOMES_COLUMNS or len(t.outcomes) != N_OUTCOMES:
        p.append(f"outcomes 표 {list(t.outcomes.columns)}×{len(t.outcomes)} ≠ {list(OUTCOMES_COLUMNS)}×{N_OUTCOMES}")
    if tuple(t.pitchers.columns) != T.PITCHERS_COLUMNS or len(t.pitchers) != n_p:
        p.append(f"pitchers 표 {list(t.pitchers.columns)}×{len(t.pitchers)} ≠ {list(T.PITCHERS_COLUMNS)}×{n_p}")
    elif not (t.pitchers["pitcher_idx"].to_numpy() == np.arange(n_p)).all():
        p.append("pitchers.pitcher_idx 가 0..P−1 순서가 아님")
    return p


def validate_value(v: V.ValueBundle, valid: np.ndarray | None = None, *, tol: float = 1e-5) -> list[str]:
    """가치함수 계약. valid[P,S,A] 를 주면 V=max·policy 지지 검사까지."""
    p: list[str] = []
    missing = [k for k in V.META_REQUIRED_KEYS if k not in v.meta]
    if missing:
        p.append(f"meta 필수 키 없음: {missing}")
    if v.Q.ndim != 3 or v.Q.shape[2] != N_ACTIONS:
        p.append(f"Q 형상 {v.Q.shape} ≠ [P, S, {N_ACTIONS}]")
        return p
    P_, S = v.Q.shape[:2]
    if S % N_COUNT_BASE_OUT != 0:
        p.append(f"Q 의 S={S} 가 288 의 배수가 아님")
    for name, a, shp in (("Q", v.Q, (P_, S, N_ACTIONS)), ("V", v.V, (P_, S)), ("policy", v.policy, (P_, S, N_ACTIONS))):
        if a.shape != shp:
            p.append(f"{name} 형상 {a.shape} ≠ {shp}")
        if a.dtype != np.float32:
            p.append(f"{name} dtype {a.dtype} ≠ float32")
    if p:
        return p
    if np.isnan(v.V).any() or np.isnan(v.policy).any():
        p.append("V/policy 에 NaN")
    if valid is not None:
        if valid.shape != v.Q.shape:
            return p + [f"valid 형상 {valid.shape} ≠ {v.Q.shape}"]
        has = valid.any(axis=-1)
        qmax = np.where(valid, v.Q.astype(np.float64), -np.inf).max(axis=-1)
        if (np.abs(qmax[has] - v.V[has]) > tol).any():
            p.append("V ≠ valid 행동 위 max Q")
        if (v.policy[~valid] != 0).any():
            p.append("policy 가 valid=False 행동에 질량")
        psum = v.policy.astype(np.float64).sum(axis=-1)
        if (np.abs(psum[has] - 1.0) > tol).any():
            p.append(f"policy 합 1±{tol} 위반 (valid 행동 있는 상태)")
    return p
