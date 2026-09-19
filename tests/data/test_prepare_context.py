"""맥락 v0 계산: 같은 타석 직전 구의 구종 계열. 행 순서와 무관하게 (game_pk, at_bat_number, pitch_number) 로 본다."""

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
