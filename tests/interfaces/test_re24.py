"""RE24 테이블 계약: RE24[24], dRE24[8, 24], float64, NaN 없음, meta 키, sha256."""

import numpy as np
import pytest

from pitcheezy.interfaces import re24 as R
from pitcheezy.interfaces.outcomes import OUTCOME_NAMES, TERMINAL


def make_table():
    rng = np.random.default_rng(0)
    return R.RE24Table(
        RE24=rng.random(24),
        dRE24=rng.normal(size=(8, 24)),
        meta={"re24_version": "test", "data_version": "test", "season_window": "2023-2025", "compute_commit": "0", "n_plate_appearances": 1},
    )


def test_valid_table():
    t = make_table()
    assert R.validate(t) == []
    assert t.terminal_names == tuple(OUTCOME_NAMES[i] for i in TERMINAL)


def test_violations():
    t = make_table()
    t.RE24 = t.RE24[:23]
    assert any("RE24 형상" in p for p in R.validate(t))
    t = make_table()
    t.dRE24 = t.dRE24.astype(np.float32)
    assert any("dtype" in p for p in R.validate(t))
    t = make_table()
    t.dRE24[0, 0] = np.nan
    assert any("NaN" in p for p in R.validate(t))
    t = make_table()
    del t.meta["compute_commit"]
    assert any("compute_commit" in p for p in R.validate(t))


def test_roundtrip_and_hash(tmp_path):
    t = make_table()
    t.save(tmp_path)
    t2 = R.RE24Table.load(tmp_path)
    assert np.array_equal(t2.dRE24, t.dRE24) and t2.meta == t.meta
    (tmp_path / "RE24.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256"):
        R.RE24Table.load(tmp_path)
