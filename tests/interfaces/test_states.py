"""상태 축 계약: count_id / base_out_id / state_id 공식과 범위, 왕복."""

import numpy as np
import pytest

from pitcheezy.interfaces import states as S


def test_constants():
    assert (S.N_COUNTS, S.N_BASE_OUT, S.N_COUNT_BASE_OUT, S.MAX_K) == (12, 24, 288, 8)
    assert S.n_states(1) == 288 and S.n_states(8) == 2304


def test_count_id_formula_and_roundtrip():
    assert S.count_id(0, 0) == 0
    assert S.count_id(3, 2) == 11
    assert S.count_id(2, 1) == 2 * 3 + 1
    for c in range(12):
        assert S.count_id(*S.decode_count(c)) == c
    with pytest.raises(ValueError):
        S.count_id(4, 0)
    with pytest.raises(ValueError):
        S.count_id(0, 3)


def test_base_out_id_formula():
    assert S.base_out_id(0, False, False, False) == 0
    assert S.base_out_id(0, True, False, False) == 1
    assert S.base_out_id(0, False, True, False) == 2
    assert S.base_out_id(0, False, False, True) == 4
    assert S.base_out_id(2, True, True, True) == 23
    # Statcast on_Xb: MLBAM id 또는 결측
    assert S.base_out_id(1, 543037, np.nan, None) == 8 + 1
    arr = S.base_out_id(np.array([0, 2]), np.array([np.nan, 1.0]), np.array([np.nan, np.nan]), np.array([5.0, np.nan]))
    assert arr.tolist() == [4, 17]
    for b in range(24):
        o, r1, r2, r3 = S.decode_base_out(b)
        assert S.base_out_id(o, r1, r2, r3) == b
    with pytest.raises(ValueError):
        S.base_out_id(3, False, False, False)


@pytest.mark.parametrize("K", [1, 2, 8])
def test_state_id_roundtrip_and_table(K):
    S_ = S.n_states(K)
    sid = np.arange(S_)
    c, b, k = S.decode_state(sid, K)
    assert (S.state_id(c, b, k, K) == sid).all()
    assert (k < K).all() and (c < 12).all() and (b < 24).all()
    # 스펙 공식: (count × 24 + base_out) × K + cluster
    assert S.state_id(11, 23, K - 1, K) == S_ - 1
    assert S.state_id(1, 0, 0, K) == 24 * K
    t = S.states_table(K)
    assert tuple(t.columns) == S.STATES_COLUMNS and len(t) == S_
    assert (t["state_id"].to_numpy() == sid).all()


def test_state_id_rejects_out_of_range():
    with pytest.raises(ValueError):
        S.state_id(0, 0, 2, 2)
    with pytest.raises(ValueError):
        S.n_states(9)
    with pytest.raises(ValueError):
        S.decode_state(288, 1)
