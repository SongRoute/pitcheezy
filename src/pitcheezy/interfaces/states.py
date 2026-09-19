"""상태 축 — docs/interface-spec.md 자산 1 "상태".

state_id = ((count_id × 24 + base_out_id) × K + cluster_id) × C + context_id, S = 288 × K × C
count_id = 볼 × 3 + 스트라이크 (0..11)
base_out_id = 아웃 × 8 + 주자 비트마스크(1루=1, 2루=2, 3루=4) (0..23). 전이 텐서·Q·RE24 모두 이 id
cluster_id 0..K−1, K ≤ 8, 좌우 층화 필수 (군집 배정 파일 쪽 책임)
context_id 0..C−1 — 시퀀스 맥락 (v1.1). C=1 이면 v1 과 같은 id (맥락 없음)
  v0 맥락 = 직전 구의 구종 계열: 0 타석 첫 구(또는 직전 구가 행동 제외) / 1 속구 / 2 변화 / 3 오프스피드
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BALLS = 4
N_STRIKES = 3
N_COUNTS = N_BALLS * N_STRIKES  # 12
N_OUTS = 3
N_BASE_OUT = N_OUTS * 8  # 24
N_COUNT_BASE_OUT = N_COUNTS * N_BASE_OUT  # 288
MAX_K = 8
MAX_C = 4
N_CONTEXT_V0 = 4  # 맥락 v0 의 C
CONTEXT_KIND_V0 = "prev_pitch_family"

STATES_COLUMNS = ("state_id", "count_id", "base_out_id", "cluster_id", "context_id")
STATES_COLUMNS_V1 = ("state_id", "count_id", "base_out_id", "cluster_id")  # 맥락 이전(v1) 산출물. C=1 일 때만 허용 (context_id = 0 으로 봄)


def n_states(K: int, C: int = 1) -> int:
    check_K(K)
    check_C(C)
    return N_COUNT_BASE_OUT * K * C


def check_K(K: int) -> None:
    if not (1 <= int(K) <= MAX_K) or int(K) != K:
        raise ValueError(f"K 는 1..{MAX_K} 정수여야 함: {K!r}")


def check_C(C: int) -> None:
    if not (1 <= int(C) <= MAX_C) or int(C) != C:
        raise ValueError(f"C 는 1..{MAX_C} 정수여야 함: {C!r}")


def context_family_of_pitch(pid):
    """pitch_id → 맥락 v0 (구종 계열). FF·SI·FC=1, SL·ST·CU=2, CH·FS·OT=3. 행동 없음(−1)은 0."""
    g = np.asarray(pid, dtype=np.int64)
    if (g >= 9).any():
        raise ValueError("pitch_id 0..8 범위 밖")
    out = np.where(g < 0, 0, g // 3 + 1)
    return int(out) if out.ndim == 0 else out


# ---------------------------------------------------------------- count
def count_id(balls, strikes):
    """볼 × 3 + 스트라이크. 스칼라·배열 모두. 범위 밖이면 ValueError."""
    b = np.asarray(balls, dtype=np.int64)
    s = np.asarray(strikes, dtype=np.int64)
    if ((b < 0) | (b >= N_BALLS)).any() or ((s < 0) | (s >= N_STRIKES)).any():
        raise ValueError("볼 0..3, 스트라이크 0..2 범위 밖")
    out = b * N_STRIKES + s
    return int(out) if out.ndim == 0 else out


def decode_count(cid):
    """count_id → (볼, 스트라이크)."""
    c = np.asarray(cid, dtype=np.int64)
    if ((c < 0) | (c >= N_COUNTS)).any():
        raise ValueError("count_id 0..11 범위 밖")
    b, s = c // N_STRIKES, c % N_STRIKES
    return (int(b), int(s)) if c.ndim == 0 else (b, s)


# ---------------------------------------------------------------- base_out
def base_out_id(outs, on_1b, on_2b, on_3b):
    """아웃 × 8 + (1루=1, 2루=2, 3루=4). 주자 인자는 bool 또는 Statcast on_Xb(주자 MLBAM id, 결측=없음)."""
    o = np.asarray(outs, dtype=np.int64)
    if ((o < 0) | (o >= N_OUTS)).any():
        raise ValueError("아웃 0..2 범위 밖")
    r1 = _runner_flag(on_1b)
    r2 = _runner_flag(on_2b)
    r3 = _runner_flag(on_3b)
    out = o * 8 + r1 + 2 * r2 + 4 * r3
    return int(out) if out.ndim == 0 else out


def _runner_flag(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == bool:
        return a.astype(np.int64)
    if a.dtype.kind in "iu":
        return (a > 0).astype(np.int64)
    # float(NaN=없음) 또는 object(None=없음)
    f = pd.to_numeric(pd.Series(a.ravel()), errors="coerce").to_numpy()
    return (~np.isnan(f) & (f > 0)).astype(np.int64).reshape(a.shape)


def decode_base_out(bid):
    """base_out_id → (아웃, 1루, 2루, 3루) bool."""
    b = np.asarray(bid, dtype=np.int64)
    if ((b < 0) | (b >= N_BASE_OUT)).any():
        raise ValueError("base_out_id 0..23 범위 밖")
    o, m = b // 8, b % 8
    r1, r2, r3 = (m & 1) > 0, (m & 2) > 0, (m & 4) > 0
    if b.ndim == 0:
        return int(o), bool(r1), bool(r2), bool(r3)
    return o, r1, r2, r3


# ---------------------------------------------------------------- state
def state_id(cid, bid, cluster, K: int, ctx=0, C: int = 1):
    check_K(K)
    check_C(C)
    c = np.asarray(cid, dtype=np.int64)
    b = np.asarray(bid, dtype=np.int64)
    k = np.asarray(cluster, dtype=np.int64)
    if ((c < 0) | (c >= N_COUNTS)).any():
        raise ValueError("count_id 0..11 범위 밖")
    if ((b < 0) | (b >= N_BASE_OUT)).any():
        raise ValueError("base_out_id 0..23 범위 밖")
    if ((k < 0) | (k >= K)).any():
        raise ValueError(f"cluster_id 0..{K - 1} 범위 밖")
    x = np.asarray(ctx, dtype=np.int64)
    if ((x < 0) | (x >= C)).any():
        raise ValueError(f"context_id 0..{C - 1} 범위 밖")
    out = ((c * N_BASE_OUT + b) * K + k) * C + x
    return int(out) if out.ndim == 0 else out


def decode_state(sid, K: int, C: int = 1):
    """state_id → (count_id, base_out_id, cluster_id). C > 1 이면 (…, context_id) 4-튜플."""
    c, b, k, x = decode_state_full(sid, K, C)
    return (c, b, k) if C == 1 else (c, b, k, x)


def decode_state_full(sid, K: int, C: int = 1):
    """state_id → (count_id, base_out_id, cluster_id, context_id). C 와 무관하게 늘 4-튜플."""
    check_K(K)
    check_C(C)
    s = np.asarray(sid, dtype=np.int64)
    if ((s < 0) | (s >= n_states(K, C))).any():
        raise ValueError(f"state_id 0..{n_states(K, C) - 1} 범위 밖")
    x = s % C
    k = (s // C) % K
    cb = s // (C * K)
    c, b = cb // N_BASE_OUT, cb % N_BASE_OUT
    return (int(c), int(b), int(k), int(x)) if s.ndim == 0 else (c, b, k, x)


def states_table(K: int, C: int = 1) -> pd.DataFrame:
    """states.parquet 내용. 행 = state_id 순서 (0..S−1). C=1 이면 context_id 는 전부 0."""
    S = n_states(K, C)
    sid = np.arange(S, dtype=np.int32)
    c, b, k, x = decode_state_full(sid, K, C)
    return pd.DataFrame(
        {
            "state_id": sid,
            "count_id": c.astype(np.int32),
            "base_out_id": b.astype(np.int32),
            "cluster_id": k.astype(np.int32),
            "context_id": x.astype(np.int32),
        }
    )
