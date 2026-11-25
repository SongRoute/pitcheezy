"""구종 축 — docs/interface-spec.md 자산 1 "행동" 의 pitch_id.

pitch_id 0..8: FF{FF,FA} / SI / FC / SL{SL,SV} / ST / CU{CU,KC,CS} / CH / FS{FS,FO} / OT{KN,SC,EP}
PO·IN·UN·null(그리고 매핑에 없는 코드)은 행동에서 제외 (카운트만 진행). 제외 비율은 data/versions.md 에 기록.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PITCH_TYPE_MAP_VERSION = "v1"  # 바꾸면 meta.pitch_type_map_version 도 바뀐다

PITCH_NAMES: tuple[str, ...] = ("FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS", "OT")
N_PITCH = len(PITCH_NAMES)  # 9

STATCAST_TO_PITCH_ID: dict[str, int] = {
    "FF": 0, "FA": 0,
    "SI": 1,
    "FC": 2,
    "SL": 3, "SV": 3,
    "ST": 4,
    "CU": 5, "KC": 5, "CS": 5,
    "CH": 6,
    "FS": 7, "FO": 7,
    "KN": 8, "SC": 8, "EP": 8,
}
EXCLUDED_PITCH_CODES: frozenset[str] = frozenset({"PO", "IN", "UN"})


def pitch_id(code: Optional[str]) -> Optional[int]:
    """Statcast pitch_type 코드 → pitch_id. 제외 대상(PO·IN·UN·결측·미지 코드)은 None."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    return STATCAST_TO_PITCH_ID.get(str(code))


def pitch_ids(codes) -> np.ndarray:
    """배열 버전. 제외 대상은 −1."""
    s = pd.Series(np.asarray(codes, dtype=object).ravel())
    out = s.map(STATCAST_TO_PITCH_ID).to_numpy(dtype=float)
    out = np.where(np.isnan(out), -1, out).astype(np.int64)
    return out.reshape(np.shape(codes))


def pitch_types_table() -> pd.DataFrame:
    """pitch_id ↔ 이름 ↔ Statcast 코드 목록."""
    codes = {i: [] for i in range(N_PITCH)}
    for code, i in STATCAST_TO_PITCH_ID.items():
        codes[i].append(code)
    return pd.DataFrame(
        {
            "pitch_id": np.arange(N_PITCH, dtype=np.int32),
            "name": list(PITCH_NAMES),
            "statcast_codes": [",".join(codes[i]) for i in range(N_PITCH)],
        }
    )
