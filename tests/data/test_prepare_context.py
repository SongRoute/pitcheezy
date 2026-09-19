"""맥락 계산: 같은 타석 직전 구에서 나온 맥락 id. 행 순서와 무관하게 (game_pk, at_bat_number, pitch_number) 로 본다. v0 / v1."""

import numpy as np
import pandas as pd
import pytest

from pitcheezy.data import prepare as PR
from pitcheezy.interfaces import states as S

# 타석 (1,1): FF(0) → SL(3) → 행동 없음(−1) → CH(6),  타석 (1,2): FS(7) → FF(0)
ROWS = [(1, 1, 1, 0), (1, 1, 2, 3), (1, 1, 3, -1), (1, 1, 4, 6), (1, 2, 1, 7), (1, 2, 2, 0)]
EXPECTED = [0, 1, 2, 0, 0, 3]  # 첫 구·직전 구 행동 없음 → 0


def _df(order=None) -> pd.DataFrame:
    d = pd.DataFrame(ROWS, columns=["game_pk", "at_bat_number", "pitch_number", "pitch_id"])
    d["count_id"], d["base_out_id"], d["cluster_id"] = 0, 0, 0
    d["expected_ctx"] = EXPECTED
    return d if order is None else d.iloc[order].reset_index(drop=True)


def test_context_ids_row_order_independent():
    for order in (None, [4, 0, 3, 5, 1, 2], [5, 4, 3, 2, 1, 0]):
        d = _df(order)
        assert PR.context_ids(d).tolist() == d["expected_ctx"].tolist()


def test_context_ids_empty():
    assert len(PR.context_ids(_df().iloc[:0])) == 0


def test_state_ids_folds_context():
    d = _df([3, 1, 5, 0, 4, 2])
    sid = PR.state_ids(d, 1, C=4, context_kind=S.CONTEXT_KIND_V0)
    assert (sid % 4 == d["expected_ctx"].to_numpy()).all()
    assert (sid // 4 == PR.state_ids(d, 1)).all()  # 나머지 성분은 C=1 과 같다
    assert (np.asarray(S.decode_state_full(sid, 1, 4)[3]) == d["expected_ctx"].to_numpy()).all()


def test_state_ids_requires_context_kind():
    with pytest.raises(ValueError):
        PR.state_ids(_df(), 1, C=4)
    with pytest.raises(ValueError):
        PR.state_ids(_df(), 1, C=4, context_kind="something_else")


# ---------------------------------------------------------------- 맥락 v1 (계열 × 존 안/밖)
# 타석 (1,1): FF 존안(12) → SL 존밖(0) → 행동 없음(−1, loc 도 −1) → CH 존안(6),  타석 (1,2): FS 존밖(24) → FF 존안(18)
ROWS_V1 = [
    (1, 1, 1, 0, 12),
    (1, 1, 2, 3, 0),
    (1, 1, 3, -1, -1),
    (1, 1, 4, 6, 6),
    (1, 2, 1, 7, 24),
    (1, 2, 2, 0, 18),
]
# 규칙 1 + (계열−1)·2 + 존안: 첫 구 0 / 직전 FF 존안 → 2 / 직전 SL 존밖 → 3 / 직전 행동 없음 → 0 / 첫 구 0 / 직전 FS 존밖 → 5
EXPECTED_V1 = [0, 2, 3, 0, 0, 5]


def _df_v1(order=None) -> pd.DataFrame:
    d = pd.DataFrame(ROWS_V1, columns=["game_pk", "at_bat_number", "pitch_number", "pitch_id", "loc_id"])
    d["count_id"], d["base_out_id"], d["cluster_id"] = 0, 0, 0
    d["expected_ctx"] = EXPECTED_V1
    return d if order is None else d.iloc[order].reset_index(drop=True)


def test_context_ids_v1_table():
    for order in (None, [4, 0, 3, 5, 1, 2], [5, 4, 3, 2, 1, 0]):
        d = _df_v1(order)
        assert PR.context_ids(d, S.CONTEXT_KIND_V1).tolist() == d["expected_ctx"].tolist()
    assert len(PR.context_ids(_df_v1().iloc[:0], S.CONTEXT_KIND_V1)) == 0


def test_context_ids_v1_is_v0_refinement():
    """v1 을 계열로 되돌리면 v0 과 같아야 한다 (0 은 0, 그 밖에는 (ctx−1)//2 + 1)."""
    d = _df_v1()
    v1 = PR.context_ids(d, S.CONTEXT_KIND_V1)
    v0 = PR.context_ids(d)
    assert np.where(v1 == 0, 0, (v1 - 1) // 2 + 1).tolist() == v0.tolist()


def test_state_ids_folds_context_v1():
    d = _df_v1([3, 1, 5, 0, 4, 2])
    sid = PR.state_ids(d, 1, C=7, context_kind=S.CONTEXT_KIND_V1)
    assert (sid % 7 == d["expected_ctx"].to_numpy()).all()
    assert (sid // 7 == PR.state_ids(d, 1)).all()
    assert (np.asarray(S.decode_state_full(sid, 1, 7)[3]) == d["expected_ctx"].to_numpy()).all()


def test_state_ids_rejects_mismatched_c():
    with pytest.raises(ValueError):
        PR.state_ids(_df_v1(), 1, C=4, context_kind=S.CONTEXT_KIND_V1)
    with pytest.raises(ValueError):
        PR.state_ids(_df_v1(), 1, C=7, context_kind=S.CONTEXT_KIND_V0)


def test_context_ids_v0_unchanged_by_registry():
    """v0 회귀: 변경 전 인라인 규칙(직전 구 pitch_id → 계열)과 완전히 같아야 한다."""
    d = _df_v1([2, 0, 5, 1, 4, 3])
    n = len(d)
    g = d["game_pk"].to_numpy(dtype=np.int64)
    ab = d["at_bat_number"].to_numpy(dtype=np.int64)
    order = np.lexsort((d["pitch_number"].to_numpy(dtype=np.int64), ab, g))
    pid = d["pitch_id"].to_numpy(dtype=np.int64)[order]
    same = np.zeros(n, dtype=bool)
    same[1:] = (g[order][1:] == g[order][:-1]) & (ab[order][1:] == ab[order][:-1])
    prev = np.concatenate(([-1], pid[:-1]))
    want = np.empty(n, dtype=np.int64)
    want[order] = np.where(same, S.context_family_of_pitch(prev), 0)
    assert PR.context_ids(d).tolist() == want.tolist()
    assert PR.context_ids(d, S.CONTEXT_KIND_V0).tolist() == want.tolist()
