"""RE24·dRE24 계산 규칙: 손으로 만든 두 하프이닝 + 실데이터 fixture 항등식."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitcheezy.data import re24 as R
from pitcheezy.interfaces.outcomes import TERMINAL, HR, K, SINGLE, IN_PLAY_OUT
from pitcheezy.interfaces.states import base_out_id

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "statcast_2024_d20260911-s2325_p2.parquet"


def _pitch(game, inning, tb, ab, pn, outs, r1, r2, r3, score, post, desc, ev=None):
    return dict(game_pk=game, inning=inning, inning_topbot=tb, at_bat_number=ab, pitch_number=pn, outs_when_up=outs,
                on_1b=r1, on_2b=r2, on_3b=r3, bat_score=score, post_bat_score=post, description=desc, events=ev)


def _toy() -> pd.DataFrame:
    n = np.nan
    rows = [
        # 경기 1, 1회초: 1B(무주자) → HR(1루) 2점 → K, K, K. 이닝 득점 2
        _pitch(1, 1, "Top", 1, 1, 0, n, n, n, 0, 0, "ball"),
        _pitch(1, 1, "Top", 1, 2, 0, n, n, n, 0, 0, "hit_into_play", "single"),
        _pitch(1, 1, "Top", 2, 1, 0, 10, n, n, 0, 2, "hit_into_play", "home_run"),
        _pitch(1, 1, "Top", 3, 1, 0, n, n, n, 2, 2, "swinging_strike", "strikeout"),
        _pitch(1, 1, "Top", 4, 1, 1, n, n, n, 2, 2, "swinging_strike", "strikeout"),
        _pitch(1, 1, "Top", 5, 1, 2, n, n, n, 2, 2, "swinging_strike", "strikeout"),
        # 1회말: 3 아웃 무득점
        _pitch(1, 1, "Bot", 6, 1, 0, n, n, n, 0, 0, "hit_into_play", "field_out"),
        _pitch(1, 1, "Bot", 7, 1, 1, n, n, n, 0, 0, "hit_into_play", "field_out"),
        _pitch(1, 1, "Bot", 8, 1, 2, n, n, n, 0, 0, "hit_into_play", "field_out"),
        # 2회초 = 경기 마지막 하프이닝 (제외 대상): 1점 나고 끝
        _pitch(1, 2, "Top", 9, 1, 0, n, n, n, 2, 3, "hit_into_play", "home_run"),
    ]
    return pd.DataFrame(rows)


def test_toy_re24_and_dre24():
    r = R.compute(_toy(), exclude_last_half_inning=True)
    assert r.n_half_innings == 3 and r.n_half_innings_excluded == 1
    assert r.n_plate_appearances == 8
    e0 = base_out_id(0, False, False, False)
    e1 = base_out_id(1, False, False, False)
    e2 = base_out_id(2, False, False, False)
    r1_0 = base_out_id(0, True, False, False)
    # 0아웃 무주자 시작 타석 = ab1(이닝 끝까지 2점), ab3(홈런 뒤, 0점), ab6(0점) → 2/3. [1루 0아웃] = 2, 1·2아웃 무주자 = 0
    assert r.RE24[e0] == pytest.approx(2 / 3) and r.RE24_n[e0] == 3
    assert r.RE24[r1_0] == pytest.approx(2.0) and r.RE24_n[r1_0] == 1
    assert r.RE24[e1] == pytest.approx(0.0) and r.RE24[e2] == pytest.approx(0.0)
    t = {o: i for i, o in enumerate(TERMINAL)}
    # 1B(0아웃 무주자): RE[1루 0아웃] + 0 − RE[무주자 0아웃] = 2 − 2/3
    assert r.dRE24[t[SINGLE], e0] == pytest.approx(2 - 2 / 3)
    # HR(1루 0아웃): RE[무주자 0아웃] + 2 − RE[1루 0아웃] = 2/3 + 2 − 2
    assert r.dRE24[t[HR], r1_0] == pytest.approx(2 / 3)
    # K(0아웃 무주자): RE[1아웃 무주자] − RE[0아웃 무주자] = −2/3 ; K(2아웃, 이닝 종료): 0 − 0
    assert r.dRE24[t[K], e0] == pytest.approx(-2 / 3)
    assert r.dRE24[t[K], e2] == pytest.approx(0.0)
    assert r.dRE24[t[IN_PLAY_OUT], e0] == pytest.approx(-2 / 3)
    # 표본 없는 셀은 NaN (make_re24 가 min_cell_n 으로 거른다)
    assert np.isnan(r.dRE24).sum() > 0


def test_toy_including_last_half_inning():
    r = R.compute(_toy(), exclude_last_half_inning=False)
    assert r.n_half_innings_excluded == 0 and r.n_plate_appearances == 9
    e0 = base_out_id(0, False, False, False)
    assert r.RE24[e0] == pytest.approx((2 + 0 + 0 + 1) / 4)


def test_fixture_runs_and_identities():
    """실데이터 두 경기: 무주자 HR 은 정확히 +1, 홈런·볼넷 등 항등식."""
    df = pd.read_parquet(FIXTURE, columns=list(R.COLUMNS))
    r = R.compute(df)
    assert r.n_plate_appearances > 30
    assert (r.RE24 >= 0).all()
