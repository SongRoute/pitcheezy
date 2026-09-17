"""실제 Statcast 행(tests/fixtures)을 인터페이스 매핑에 통과시킨다: 모든 투구가 state·outcome 을 받고, 행동 제외는 구종 null 뿐."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitcheezy.interfaces import grid as G
from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import pitch_types as PT
from pitcheezy.interfaces import states as S

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "statcast_2024_d20260911-s2325_p2.parquet"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def test_fixture_shape(df):
    assert len(df) == 196 and df["pitcher"].nunique() == 2
    assert set(df["p_throws"]) == {"L", "R"}


def test_every_pitch_gets_state(df):
    cid = S.count_id(df["balls"].to_numpy(), df["strikes"].to_numpy())
    bid = S.base_out_id(df["outs_when_up"].to_numpy(), df["on_1b"].to_numpy(), df["on_2b"].to_numpy(), df["on_3b"].to_numpy())
    sid = S.state_id(cid, bid, np.zeros(len(df), dtype=int), 1)
    assert sid.min() >= 0 and sid.max() < S.n_states(1)
    # 첫 투구는 0-0 카운트
    first = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby(["game_pk", "at_bat_number"]).head(1)
    assert (first["balls"] == 0).all() and (first["strikes"] == 0).all()


def test_every_pitch_gets_outcome_and_terminals_match_events(df):
    oid = O.outcome_ids(df["description"].to_numpy(dtype=object), df["events"].to_numpy(dtype=object))
    assert (oid >= 0).all(), df.loc[oid < 0, ["description", "events"]].drop_duplicates()
    term = O.is_terminal(oid)
    has_event = df["events"].notna().to_numpy()
    # 종결 ⇒ events 있음. (events 있음 ⇒ 종결은 주자 사건 때문에 성립하지 않을 수 있음)
    assert (~term | has_event).all()
    # 타석 마지막 투구는 종결(주자 사건으로 끊긴 타석 제외)
    last = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby(["game_pk", "at_bat_number"]).tail(1)
    last_oid = O.outcome_ids(last["description"].to_numpy(dtype=object), last["events"].to_numpy(dtype=object))
    frac = O.is_terminal(last_oid).mean()
    assert frac > 0.9


def test_next_count_matches_data(df):
    """비종결 결과의 count_rule 로 만든 다음 카운트가 실제 다음 투구의 카운트와 같다."""
    d = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    oid = O.outcome_ids(d["description"].to_numpy(dtype=object), d["events"].to_numpy(dtype=object))
    cid = S.count_id(d["balls"].to_numpy(), d["strikes"].to_numpy())
    same_pa = (d["at_bat_number"].shift(-1) == d["at_bat_number"]).to_numpy() & (d["game_pk"].shift(-1) == d["game_pk"]).to_numpy()
    checked = 0
    for i in np.where(same_pa)[0]:
        if O.is_terminal(oid[i]):
            continue
        assert O.next_count(int(cid[i]), int(oid[i])) == cid[i + 1], (d.iloc[i][["balls", "strikes", "description"]].tolist(), d.iloc[i + 1][["balls", "strikes"]].tolist())
        checked += 1
    assert checked > 100


def test_actions_only_missing_for_excluded_pitch_types(df):
    pid = PT.pitch_ids(df["pitch_type"].to_numpy(dtype=object))
    excluded = df["pitch_type"].isna() | df["pitch_type"].isin(PT.EXCLUDED_PITCH_CODES)
    assert ((pid < 0) == excluded.to_numpy()).all()
    ok = pid >= 0
    zn = G.z_norm(df["plate_z"].to_numpy(), df["sz_bot"].to_numpy(), df["sz_top"].to_numpy())
    has_loc = ~np.isnan(zn) & df["plate_x"].notna().to_numpy()
    aid = G.action_id(pid[ok & has_loc], G.loc_id(df["plate_x"].to_numpy()[ok & has_loc], zn[ok & has_loc]))
    assert aid.min() >= 0 and aid.max() < G.N_ACTIONS
    assert (ok & has_loc).sum() >= 0.95 * len(df)
