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
    assert (S.MAX_C, S.N_CONTEXT_V0, S.CONTEXT_KIND_V0) == (7, 4, "prev_pitch_family")
    assert [S.context_family_of_pitch(g) for g in range(9)] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert S.context_family_of_pitch(-1) == 0  # 행동 없음(PO·IN·UN·null)
    assert S.context_family_of_pitch(np.array([-1, 0, 2, 3, 5, 6, 8])).tolist() == [0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError):
        S.context_family_of_pitch(9)
    with pytest.raises(ValueError):
        S.n_states(1, 8)
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



# ---------------------------------------------------------------- 맥락 레지스트리·v1 (v1.2)
def test_context_kind_registry():
    assert S.CONTEXT_KINDS == {"prev_pitch_family": 4, "prev_pitch_family_zone": 7}
    assert (S.CONTEXT_KIND_V1, S.N_CONTEXT_V1) == ("prev_pitch_family_zone", 7)
    assert S.MAX_C == max(S.CONTEXT_KINDS.values())
    assert S.n_context(S.CONTEXT_KIND_V0) == 4 and S.n_context(S.CONTEXT_KIND_V1) == 7
    with pytest.raises(ValueError):
        S.n_context("prev_pitch")


def test_in_zone_edges():
    # loc_id = z행 × 5 + x열. 가운데 3×3(행·열 1..3)만 존 안, 0·4 행/열은 밖
    for zr in range(5):
        for xc in range(5):
            want = 1 <= zr <= 3 and 1 <= xc <= 3
            assert bool(S.in_zone(zr * 5 + xc)) is want
    assert int(S.in_zone(np.arange(25)).sum()) == 9


def test_context_of_action_v0_ignores_loc():
    for lid in (0, 12, 24):
        assert [S.context_of_action(g, lid, S.CONTEXT_KIND_V0) for g in range(9)] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert S.context_of_action(-1, 12, S.CONTEXT_KIND_V0) == 0
    assert S.context_of_action(np.array([-1, 0, 3, 6]), None, S.CONTEXT_KIND_V0).tolist() == [0, 1, 2, 3]
    assert S.context_of_action(4) == 2  # 기본 kind = v0, loc 없이도


def test_context_of_action_v1_table():
    K1 = S.CONTEXT_KIND_V1
    assert S.context_of_action(-1, 12, K1) == 0  # 행동 제외
    assert S.context_of_action(0, -1, K1) == 0  # 위치 없음 → action_id < 0 과 같은 취급
    inside, outside = 12, 0  # 12 = (2,2) 존 안, 0 = (0,0) 존 밖
    for g, fam in [(0, 1), (2, 1), (3, 2), (5, 2), (6, 3), (8, 3)]:
        assert S.context_of_action(g, outside, K1) == 1 + (fam - 1) * 2
        assert S.context_of_action(g, inside, K1) == 1 + (fam - 1) * 2 + 1
    # 존 경계: 0행·4행·0열·4열은 밖
    for lid in (0, 4, 20, 24, 2, 10, 14, 22):
        assert S.context_of_action(0, lid, K1) == 1
    for lid in (6, 7, 8, 11, 12, 13, 16, 17, 18):
        assert S.context_of_action(0, lid, K1) == 2
    # 벡터화: 행동 축 전체가 1..6, 0 은 안 나온다
    pid, lid = np.divmod(np.arange(225), 25)
    out = S.context_of_action(pid, lid, K1)
    assert out.shape == (225,) and set(np.unique(out).tolist()) == {1, 2, 3, 4, 5, 6}
    assert int((out % 2 == 0).sum()) == 9 * 9  # 존 안 = 구종 9 × 존 안 칸 9


def test_context_of_action_v1_requires_loc():
    with pytest.raises(ValueError):
        S.context_of_action(0, None, S.CONTEXT_KIND_V1)
    with pytest.raises(ValueError):
        S.context_of_action(0, 12, "prev_pitch_family_quadrant")


@pytest.mark.parametrize("K", [1, 2])
def test_state_id_context_roundtrip_c7(K):
    S_ = S.n_states(K, 7)
    assert S_ == 288 * K * 7
    sid = np.arange(S_)
    c, b, k, x = S.decode_state_full(sid, K, 7)
    assert (S.state_id(c, b, k, K, x, 7) == sid).all()
    assert set(np.unique(x).tolist()) == set(range(7))
    t = S.states_table(K, 7)
    assert list(t.columns) == list(S.STATES_COLUMNS) and len(t) == S_
