"""결과 축 계약: 11종, Statcast 매핑, count_rule, 규칙 마스크."""

import numpy as np
import pytest

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces.states import count_id


def test_axis():
    assert O.N_OUTCOMES == 11 and O.N_TERMINAL == 8
    assert O.OUTCOME_NAMES[:3] == ("ball", "strike", "foul")
    assert O.TERMINAL == (3, 4, 5, 6, 7, 8, 9, 10)
    t = O.outcomes_table()
    assert tuple(t.columns) == O.OUTCOMES_COLUMNS and len(t) == 11
    assert t["terminal"].tolist() == [False] * 3 + [True] * 8
    assert t["count_rule"].tolist()[:3] == ["ball", "strike", "foul"]


@pytest.mark.parametrize(
    "desc,events,expected",
    [
        ("ball", None, O.BALL),
        ("blocked_ball", None, O.BALL),
        ("automatic_ball", None, O.BALL),
        ("called_strike", None, O.STRIKE),
        ("swinging_strike_blocked", None, O.STRIKE),
        ("foul_tip", None, O.STRIKE),
        ("missed_bunt", None, O.STRIKE),
        ("automatic_strike", None, O.STRIKE),
        ("foul", None, O.FOUL),
        ("foul_bunt", None, O.FOUL),
        ("swinging_strike", "strikeout", O.K),
        ("foul_tip", "strikeout", O.K),
        ("foul_bunt", "strikeout", O.K),
        ("called_strike", "strikeout_double_play", O.K),
        ("ball", "walk", O.BB),
        ("ball", "intent_walk", O.BB),
        ("hit_by_pitch", "hit_by_pitch", O.HBP),
        ("hit_into_play", "single", O.SINGLE),
        ("hit_into_play", "double", O.DOUBLE),
        ("hit_into_play", "triple", O.TRIPLE),
        ("hit_into_play", "home_run", O.HR),
        ("hit_into_play", "field_out", O.IN_PLAY_OUT),
        ("hit_into_play", "grounded_into_double_play", O.IN_PLAY_OUT),
        ("hit_into_play", "sac_fly", O.IN_PLAY_OUT),
        ("hit_into_play", "field_error", O.IN_PLAY_OUT),
        ("hit_into_play", "catcher_interf", O.IN_PLAY_OUT),
        ("hit_into_play", "fielders_choice", O.IN_PLAY_OUT),
        # 주자 사건은 투구 결과에 영향 없음
        ("ball", "caught_stealing_2b", O.BALL),
        ("called_strike", "pickoff_1b", O.STRIKE),
        ("nonsense", None, None),
        (None, None, None),
    ],
)
def test_statcast_mapping(desc, events, expected):
    assert O.outcome_id(desc, events) == expected


def test_outcome_ids_array():
    arr = O.outcome_ids(np.array(["ball", "hit_into_play", "zzz"], dtype=object), np.array([None, "single", None], dtype=object))
    assert arr.tolist() == [O.BALL, O.SINGLE, -1]


def test_next_count_rules():
    assert O.next_count(count_id(0, 0), O.BALL) == count_id(1, 0)
    assert O.next_count(count_id(0, 0), O.STRIKE) == count_id(0, 1)
    assert O.next_count(count_id(1, 1), O.FOUL) == count_id(1, 2)
    assert O.next_count(count_id(1, 2), O.FOUL) == count_id(1, 2)  # 2스트 파울은 유지
    for oid in O.TERMINAL:
        assert O.next_count(count_id(3, 2), oid) is None
    with pytest.raises(ValueError):
        O.next_count(count_id(3, 0), O.BALL)
    with pytest.raises(ValueError):
        O.next_count(count_id(0, 2), O.STRIKE)


def test_rule_mask():
    m = O.rule_mask_table()
    assert m.shape == (12, 11)
    for c in range(12):
        b, s = divmod(c, 3)
        assert m[c, O.BALL] == (b < 3)
        assert m[c, O.BB] == (b == 3)
        assert m[c, O.STRIKE] == (s < 2)
        assert m[c, O.K] == (s == 2)
        assert m[c, [O.FOUL, O.HBP, O.SINGLE, O.DOUBLE, O.TRIPLE, O.HR, O.IN_PLAY_OUT]].all()
    assert (O.rule_mask(count_id(3, 2)) == m[11]).all()
