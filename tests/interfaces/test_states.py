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


# ---------------------------------------------------------------- 시퀀스 맥락 (v1.1)
def test_context_constants_and_family_table():
    assert (S.MAX_C, S.N_CONTEXT_V0, S.CONTEXT_KIND_V0) == (4, 4, "prev_pitch_family")
    assert [S.context_family_of_pitch(g) for g in range(9)] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert S.context_family_of_pitch(-1) == 0  # 행동 없음(PO·IN·UN·null)
    assert S.context_family_of_pitch(np.array([-1, 0, 2, 3, 5, 6, 8])).tolist() == [0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError):
        S.context_family_of_pitch(9)
    with pytest.raises(ValueError):
        S.n_states(1, 5)
    with pytest.raises(ValueError):
        S.n_states(1, 0)


@pytest.mark.parametrize("K", [1, 2])
@pytest.mark.parametrize("C", [1, 4])
def test_state_id_context_roundtrip(K, C):
    S_ = S.n_states(K, C)
    assert S_ == 288 * K * C
    sid = np.arange(S_)
    c, b, k, x = S.decode_state_full(sid, K, C)
    assert (S.state_id(c, b, k, K, x, C) == sid).all()
    assert (x < C).all() and (k < K).all() and (c < 12).all() and (b < 24).all()
    assert len(S.decode_state(sid, K, C)) == (3 if C == 1 else 4)
    if C == 1:  # v1 공식과 완전히 같은 id
        assert (sid == (c * 24 + b) * K + k).all()
        assert (S.state_id(c, b, k, K) == S.state_id(c, b, k, K, 0, 1)).all()
    else:
        assert (sid // C == S.state_id(c, b, k, K)).all()  # 맥락은 뒤에 접힌다
    with pytest.raises(ValueError):
        S.state_id(0, 0, 0, K, C, C)  # ctx 범위 밖


def test_states_table_context_column():
    t = S.states_table(1, 4)
    assert tuple(t.columns) == S.STATES_COLUMNS and len(t) == 1152
    assert t["context_id"].to_numpy()[:8].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert (t["state_id"].to_numpy() == np.arange(1152)).all()
    assert (S.states_table(2)["context_id"].to_numpy() == 0).all()  # C=1 은 전부 0
