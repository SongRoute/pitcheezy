"""상태 축 — docs/interface-spec.md 자산 1 "상태".

state_id = (count_id × 24 + base_out_id) × K + cluster_id, S = 288 × K
count_id = 볼 × 3 + 스트라이크 (0..11)
base_out_id = 아웃 × 8 + 주자 비트마스크(1루=1, 2루=2, 3루=4) (0..23). 전이 텐서·Q·RE24 모두 이 id
cluster_id 0..K−1, K ≤ 8, 좌우 층화 필수 (군집 배정 파일 쪽 책임)
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

STATES_COLUMNS = ("state_id", "count_id", "base_out_id", "cluster_id")


def n_states(K: int) -> int:
    check_K(K)
    return N_COUNT_BASE_OUT * K


def check_K(K: int) -> None:
    if not (1 <= int(K) <= MAX_K) or int(K) != K:
        raise ValueError(f"K 는 1..{MAX_K} 정수여야 함: {K!r}")


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
def state_id(cid, bid, cluster, K: int):
    check_K(K)
    c = np.asarray(cid, dtype=np.int64)
    b = np.asarray(bid, dtype=np.int64)
    k = np.asarray(cluster, dtype=np.int64)
    if ((c < 0) | (c >= N_COUNTS)).any():
        raise ValueError("count_id 0..11 범위 밖")
    if ((b < 0) | (b >= N_BASE_OUT)).any():
        raise ValueError("base_out_id 0..23 범위 밖")
    if ((k < 0) | (k >= K)).any():
        raise ValueError(f"cluster_id 0..{K - 1} 범위 밖")
    out = (c * N_BASE_OUT + b) * K + k
    return int(out) if out.ndim == 0 else out


def decode_state(sid, K: int):
    """state_id → (count_id, base_out_id, cluster_id)."""
    check_K(K)
    s = np.asarray(sid, dtype=np.int64)
    if ((s < 0) | (s >= n_states(K))).any():
        raise ValueError(f"state_id 0..{n_states(K) - 1} 범위 밖")
    k = s % K
    cb = s // K
    c, b = cb // N_BASE_OUT, cb % N_BASE_OUT
    return (int(c), int(b), int(k)) if s.ndim == 0 else (c, b, k)


def states_table(K: int) -> pd.DataFrame:
    """states.parquet 내용. 행 = state_id 순서 (0..S−1)."""
    S = n_states(K)
    sid = np.arange(S, dtype=np.int32)
    c, b, k = decode_state(sid, K)
    return pd.DataFrame(
        {
            "state_id": sid,
            "count_id": c.astype(np.int32),
            "base_out_id": b.astype(np.int32),
            "cluster_id": k.astype(np.int32),
        }
    )
