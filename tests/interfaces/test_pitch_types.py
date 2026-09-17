"""구종 병합표 계약: 9종, 스펙 병합 규칙, 제외 코드."""

import numpy as np

from pitcheezy.interfaces import pitch_types as P


def test_nine_groups_in_spec_order():
    assert P.N_PITCH == 9
    assert P.PITCH_NAMES == ("FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS", "OT")
    assert P.PITCH_TYPE_MAP_VERSION == "v1"


def test_merge_rules():
    assert P.pitch_id("FF") == P.pitch_id("FA") == 0
    assert P.pitch_id("SI") == 1
    assert P.pitch_id("FC") == 2
    assert P.pitch_id("SL") == P.pitch_id("SV") == 3
    assert P.pitch_id("ST") == 4
    assert P.pitch_id("CU") == P.pitch_id("KC") == P.pitch_id("CS") == 5
    assert P.pitch_id("CH") == 6
    assert P.pitch_id("FS") == P.pitch_id("FO") == 7
    assert P.pitch_id("KN") == P.pitch_id("SC") == P.pitch_id("EP") == 8


def test_excluded_codes_return_none():
    for code in ("PO", "IN", "UN", None, float("nan"), "ZZ"):
        assert P.pitch_id(code) is None
    arr = P.pitch_ids(np.array(["FF", "PO", None, "KC"], dtype=object))
    assert arr.tolist() == [0, -1, -1, 5]


def test_table_covers_every_group():
    t = P.pitch_types_table()
    assert len(t) == 9 and set(t["pitch_id"]) == set(range(9))
    assert all(t["statcast_codes"].str.len() > 0)
