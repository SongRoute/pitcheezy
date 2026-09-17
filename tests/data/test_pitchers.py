"""레귤러 선발 필터: 아웃 귀속·이닝·선발 판정(토이) + fixture 로 Webb 2024-03-28 한 경기 이닝."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitcheezy.data import pitchers as P

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "statcast_2024_d20260911-s2325_p2.parquet"


def _row(game, inning, tb, ab, pn, outs, pitcher, pt="FF"):
    return dict(game_pk=game, game_year=2024, inning=inning, inning_topbot=tb, at_bat_number=ab, pitch_number=pn,
                outs_when_up=outs, pitcher=pitcher, player_name=f"P{pitcher}", p_throws="R", pitch_type=pt)


def test_toy_outs_and_starts():
    rows = [
        # 1회초 투수 1: 타석 3개, 아웃 0→1→2→(이닝 끝 3) = 3아웃. 두 번째 타석은 두 투구
        _row(1, 1, "Top", 1, 1, 0, 1), _row(1, 1, "Top", 2, 1, 1, 1), _row(1, 1, "Top", 2, 2, 1, 1, None), _row(1, 1, "Top", 3, 1, 2, 1),
        # 1회말 투수 2: 타석 2개 (아웃 0→2, 이닝 끝 3) = 3아웃
        _row(1, 1, "Bot", 4, 1, 0, 2), _row(1, 1, "Bot", 5, 1, 2, 2),
        # 2회초 투수 3(구원): 타석 1개, 아웃 0 → 이닝 끝 = 3아웃
        _row(1, 2, "Top", 6, 1, 0, 3),
    ]
    st = P.season_table(pd.DataFrame(rows))
    by = st.set_index("mlbam_id")
    assert by.loc[1, "outs"] == 3 and by.loc[1, "ip"] == pytest.approx(1.0) and by.loc[1, "starts"] == 1
    assert by.loc[1, "pitches"] == 4 and by.loc[1, "pitches_action"] == 3
    assert by.loc[2, "outs"] == 3 and by.loc[2, "starts"] == 1
    assert by.loc[3, "outs"] == 3 and by.loc[3, "starts"] == 0 and by.loc[3, "games"] == 1
    pool = P.build_pool(st, min_ip=1.0)
    assert set(pool["mlbam_id"]) == {1, 2, 3} and "q2024" in pool.columns
    assert P.build_pool(st, min_ip=1.5).empty
    with pytest.raises(ValueError):
        P.build_pool(st, 1.0, rule="all_seasons")


def test_fixture_webb_one_game():
    """Webb 2024-03-28 SF@SD 선발. 전체 경기 데이터로 확인: 1~6회 투구, 6회 2아웃 마지막 타석 삼진으로 이닝 종료 → 6.0 이닝, 선발 1."""
    df = pd.read_parquet(FIXTURE, columns=list(P.COLUMNS))
    st = P.season_table(df).set_index("mlbam_id")
    assert st.loc[657277, "starts"] == 1
    assert st.loc[657277, "ip"] == pytest.approx(6.0)
    assert st.loc[657277, "pitches"] == 98
