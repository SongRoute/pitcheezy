"""위치 격자·행동 축 계약: 경계, 클립, 좌우 반전 없음, action_id 공식."""

import numpy as np
import pytest

from pitcheezy.interfaces import grid as G


def test_constants():
    assert (G.N_X, G.N_Z, G.N_LOC, G.N_ACTIONS) == (5, 5, 25, 225)
    assert G.X_EDGES == (-0.83, -0.28, 0.28, 0.83)
    assert G.Z_EDGES == (0.0, 1 / 3, 2 / 3, 1.0)


def test_x_bins_and_clip():
    # 포수 시점 절대 좌표, 좌우 반전 없음: 음수 = 열 0 쪽
    assert G.x_col(-5.0) == 0 and G.x_col(-0.9) == 0
    assert G.x_col(-0.5) == 1
    assert G.x_col(0.0) == 2
    assert G.x_col(0.5) == 3
    assert G.x_col(0.9) == 4 and G.x_col(5.0) == 4
    # 경계값은 오른쪽 칸
    assert G.x_col(-0.83) == 1 and G.x_col(0.83) == 4


def test_z_norm_and_bins():
    assert G.z_norm(2.5, 1.5, 3.5) == pytest.approx(0.5)
    assert np.isnan(G.z_norm(2.0, 3.0, 3.0))
    assert G.z_row(-1.0) == 0
    assert G.z_row(0.1) == 1
    assert G.z_row(0.5) == 2
    assert G.z_row(0.9) == 3
    assert G.z_row(1.5) == 4
    assert G.z_row(0.0) == 1 and G.z_row(1.0) == 4


def test_nan_position_rejected():
    with pytest.raises(ValueError):
        G.x_col(np.nan)
    with pytest.raises(ValueError):
        G.loc_id(np.array([0.0, np.nan]), np.array([0.5, 0.5]))


def test_loc_and_action_formula_roundtrip():
    assert G.loc_id(0.0, 0.5) == 2 * 5 + 2
    for l in range(25):
        zr, xc = G.decode_loc(l)
        assert zr * 5 + xc == l
    for a in range(225):
        p, l = G.decode_action(a)
        assert G.action_id(p, l) == a
        assert a == p * 25 + l
    with pytest.raises(ValueError):
        G.action_id(9, 0)
    with pytest.raises(ValueError):
        G.action_id(-1, 0)  # 제외 구종은 행동이 될 수 없음


def test_loc_center_inside_cell():
    for l in range(25):
        x, z = G.loc_center(l)
        assert G.loc_id(x, z) == l
