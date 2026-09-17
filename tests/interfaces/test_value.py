"""가치함수 계약: 형상·dtype, V = valid 위 max Q, policy 지지, lookup snap/bilinear."""

import numpy as np
import pytest

from pitcheezy.interfaces import value as V
from pitcheezy.interfaces.grid import action_id, loc_id
from pitcheezy.interfaces.validate import validate_value


def test_fixture_value_passes(small_tensor, small_value):
    assert validate_value(small_value, small_tensor.valid) == []


def test_save_load_roundtrip(tmp_path, small_tensor, small_value):
    d = tmp_path / "value"
    small_value.save(d)
    v2 = V.ValueBundle.load(d)
    assert validate_value(v2, small_tensor.valid) == []
    assert np.array_equal(v2.Q, small_value.Q)
    (d / "Q.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256"):
        V.ValueBundle.load(d)


def test_v_must_be_max_over_valid(small_tensor, small_value):
    small_value.V[0, 0] += 1.0
    assert any("max" in p for p in validate_value(small_value, small_tensor.valid))


def test_policy_support_and_sum(small_tensor, small_value):
    idx = tuple(np.argwhere(~small_tensor.valid)[0])
    small_value.policy[idx] = 0.1
    probs = validate_value(small_value, small_tensor.valid)
    assert any("valid=False" in p for p in probs) or any("합" in p for p in probs)


def test_meta_keys(small_tensor, small_value):
    del small_value.meta["lookup_mode"]
    assert any("lookup_mode" in p for p in validate_value(small_value, small_tensor.valid))


def test_lookup_snap_matches_cell(small_value):
    Q = small_value.Q
    x, z, pid = 0.1, 0.5, 3
    aid = action_id(pid, loc_id(x, z))
    assert V.lookup(Q, 1, 7, pid, x, z) == pytest.approx(float(Q[1, 7, aid]))
    # 같은 셀 안이면 값이 같다 (V(의도)=V(실제) → 실책 0)
    assert V.lookup(Q, 1, 7, pid, 0.2, 0.4) == V.lookup(Q, 1, 7, pid, x, z)
    # 배열 조회
    out = V.lookup(Q, np.array([0, 1]), np.array([7, 7]), np.array([pid, pid]), np.array([x, x]), np.array([z, z]))
    assert out.shape == (2,)


def test_lookup_bilinear_placeholder(small_value):
    with pytest.raises(NotImplementedError):
        V.lookup(small_value.Q, 0, 0, 0, 0.0, 0.5, mode="bilinear")
    with pytest.raises(ValueError):
        V.lookup(small_value.Q, 0, 0, 0, 0.0, 0.5, mode="nearest")
