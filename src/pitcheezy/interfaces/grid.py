"""위치 격자와 행동 축 — docs/interface-spec.md 자산 1 "행동".

action_id = pitch_id × 25 + loc_id, A = 225
loc_id = z행 × 5 + x열 (0..24)
  x: plate_x 경계 −0.83 / −0.28 / +0.28 / +0.83 ft, 바깥 클립. 포수 시점 절대 좌표, 좌우 반전 없음
  z: z_norm = (plate_z − sz_bot) / (sz_top − sz_bot), 경계 0 / ⅓ / ⅔ / 1, 바깥 클립
"""

from __future__ import annotations

import numpy as np

from .pitch_types import N_PITCH

X_EDGES: tuple[float, ...] = (-0.83, -0.28, 0.28, 0.83)
Z_EDGES: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
N_X = len(X_EDGES) + 1  # 5
N_Z = len(Z_EDGES) + 1  # 5
N_LOC = N_X * N_Z  # 25
N_ACTIONS = N_PITCH * N_LOC  # 225

ACTIONS_COLUMNS = ("action_id", "pitch_id", "loc_id", "z_row", "x_col")


def z_norm(plate_z, sz_bot, sz_top):
    """(plate_z − sz_bot) / (sz_top − sz_bot). sz_top ≤ sz_bot 이면 NaN."""
    pz = np.asarray(plate_z, dtype=np.float64)
    lo = np.asarray(sz_bot, dtype=np.float64)
    hi = np.asarray(sz_top, dtype=np.float64)
    h = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(h > 0, (pz - lo) / h, np.nan)
    return float(out) if out.ndim == 0 else out


def x_col(plate_x):
    """0..4. 경계값은 오른쪽 칸 (np.digitize right=False). NaN 이면 ValueError."""
    return _bin(plate_x, X_EDGES)


def z_row(zn):
    """0..4. 경계값은 위 칸. NaN 이면 ValueError."""
    return _bin(zn, Z_EDGES)


def _bin(v, edges):
    a = np.asarray(v, dtype=np.float64)
    if np.isnan(a).any():
        raise ValueError("위치 결측 (NaN) — 행동 축에 넣기 전에 걸러야 함")
    out = np.digitize(a, edges)  # 0..len(edges), 바깥은 자동 클립
    return int(out) if out.ndim == 0 else out


def loc_id(plate_x, zn):
    out = np.asarray(z_row(zn)) * N_X + np.asarray(x_col(plate_x))
    return int(out) if out.ndim == 0 else out


def decode_loc(lid):
    l = np.asarray(lid, dtype=np.int64)
    if ((l < 0) | (l >= N_LOC)).any():
        raise ValueError("loc_id 0..24 범위 밖")
    zr, xc = l // N_X, l % N_X
    return (int(zr), int(xc)) if l.ndim == 0 else (zr, xc)


def action_id(pid, lid):
    p = np.asarray(pid, dtype=np.int64)
    l = np.asarray(lid, dtype=np.int64)
    if ((p < 0) | (p >= N_PITCH)).any():
        raise ValueError("pitch_id 0..8 범위 밖 (제외 구종은 행동으로 만들 수 없음)")
    if ((l < 0) | (l >= N_LOC)).any():
        raise ValueError("loc_id 0..24 범위 밖")
    out = p * N_LOC + l
    return int(out) if out.ndim == 0 else out


def decode_action(aid):
    a = np.asarray(aid, dtype=np.int64)
    if ((a < 0) | (a >= N_ACTIONS)).any():
        raise ValueError("action_id 0..224 범위 밖")
    p, l = a // N_LOC, a % N_LOC
    return (int(p), int(l)) if a.ndim == 0 else (p, l)


def loc_center(lid):
    """셀 중심 (plate_x ft, z_norm). 바깥 칸은 경계에서 안쪽 칸 폭만큼 바깥. snap 룩업·그림용."""
    zr, xc = decode_loc(lid)
    return _center(np.asarray(xc), X_EDGES), _center(np.asarray(zr), Z_EDGES)


def _center(idx, edges):
    e = np.asarray(edges)
    w = np.diff(e)
    centers = np.concatenate([[e[0] - w[0] / 2], (e[:-1] + e[1:]) / 2, [e[-1] + w[-1] / 2]])
    out = centers[idx]
    return float(out) if out.ndim == 0 else out
