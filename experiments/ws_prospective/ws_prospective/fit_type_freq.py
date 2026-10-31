"""Type-conditional FIT frequency comparator (REVIEW-001 B2, primary comparator in v2).

Two parts:

* ``compute_counts`` - a pre-Game-1 AGGREGATE step over the T_FINAL FIT window
  (the model's own training rows, role 2026 EXPOSED DEVELOPMENT): seven-leaf
  counts per pitch type from the prepared package (corpus-subset.npz ``y7``,
  ``fit``; seq-inputs.npz ``pitch_type`` index into the frozen TYPES).  It
  verifies the package manifest shas, refuses unless the per-type counts sum
  to the registered FIT marginal [258626, ...]/720505, and writes only counts.
  It is NOT run in the preparation step (it reads 2026 rows in aggregate); it
  needs its own approval, then its output JSON sha is pinned in the executable
  registration before Game 1.
* ``FitTypeFreq7`` - the adapter: for each registered candidate type with a
  non-zero FIT total, p = (count + alpha) / (total + 7 * alpha), alpha = 0.5
  (fixed); types with zero FIT rows are omitted (graded as
  REALIZED_TYPE_NOT_IN_CANDIDATES), never zero-filled.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import LEAVES
from .adapters import FIT_COUNTS, FIT_TOTAL, AdapterBase, PinMismatch, sha256_file
from .context import PrePitchContext

SCHEMA = "rs01-fit-type-counts-v1"
ALPHA = 0.5
TYPES19 = ('UNKNOWN', 'CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO',
           'SC', 'SI', 'SL', 'ST', 'SV', 'UN')  # first_small_features.TYPES (frozen recipe order)


def compute_counts(package_dir: Path, *, expected_manifest_sha256: str) -> dict:
    package_dir = Path(package_dir)
    man_path = package_dir / "manifest.json"
    if sha256_file(man_path) != expected_manifest_sha256:
        raise PinMismatch("package manifest sha differs")
    man = json.loads(man_path.read_text())
    for name in ("corpus-subset.npz", "seq-inputs.npz"):
        if sha256_file(package_dir / name) != man["files"][name]["sha256"]:
            raise PinMismatch(f"{name} sha differs from manifest")
    with np.load(package_dir / "corpus-subset.npz", allow_pickle=False) as c, \
            np.load(package_dir / "seq-inputs.npz", allow_pickle=False) as s:
        y7, fit, ptype = c["y7"], c["fit"].astype(bool), s["pitch_type"]
        keys_c, keys_s = c["keys"], np.stack([s["game_pk"], s["at_bat_number"], s["pitch_number"]], 1)
    if len(y7) != len(ptype) or not np.array_equal(keys_c.astype(np.int64), keys_s.astype(np.int64)):
        raise ValueError("corpus and sequence rows are not aligned")
    sel = fit & (y7 >= 0)
    counts = np.zeros((len(TYPES19), 7), dtype=np.int64)
    np.add.at(counts, (ptype[sel].astype(np.int64), y7[sel].astype(np.int64)), 1)
    if counts.sum(0).tolist() != list(FIT_COUNTS) or int(counts.sum()) != FIT_TOTAL:
        raise ValueError("per-type counts do not reproduce the registered FIT marginal")
    return {"schema": SCHEMA, "leaf_order": list(LEAVES), "types": list(TYPES19),
            "counts": {t: counts[i].tolist() for i, t in enumerate(TYPES19)},
            "total": FIT_TOTAL, "fit_marginal_check": list(FIT_COUNTS),
            "source": {"package_dir": str(package_dir), "manifest_sha256": expected_manifest_sha256},
            "alpha": ALPHA, "rows_read": "aggregate counts only; no row-level output"}


def write_counts(obj: dict, dest: Path) -> str:
    raw = (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode()
    with open(dest, "xb") as f:
        f.write(raw)
    return hashlib.sha256(raw).hexdigest()


class FitTypeFreq7(AdapterBase):
    model_id = "FIT_TYPE_FREQ_7"

    def __init__(self, counts: Dict[str, List[int]], types: List[str], alpha: float = ALPHA):
        self.vectors: Dict[str, List[float]] = {}
        for t in types:
            c = counts.get(t)
            if c is None or len(c) != 7 or sum(c) <= 0:
                continue
            den = sum(c) + 7 * alpha
            self.vectors[t] = [(x + alpha) / den for x in c]
        if not self.vectors:
            raise ValueError("no candidate type has FIT rows")

    @classmethod
    def build(cls, reg: dict, game: dict, out_dir: Path) -> "FitTypeFreq7":
        src = reg["model_sources"][cls.model_id]
        if sha256_file(src["counts_path"]) != src["counts_sha256"]:
            raise PinMismatch("FIT type counts file changed")
        body = json.loads(Path(src["counts_path"]).read_text())
        if (body.get("schema") != SCHEMA or body.get("leaf_order") != list(LEAVES)
                or [sum(v[i] for v in body["counts"].values()) for i in range(7)] != list(FIT_COUNTS)):
            raise PinMismatch("FIT type counts do not match the registered marginal/leaf order")
        types = list(src.get("candidate_pitch_types") or reg["candidate_pitch_types"])
        return cls(body["counts"], types, float(src.get("alpha", ALPHA)))

    def predict(self, ctx: PrePitchContext) -> Dict[str, List[float]]:
        return {t: list(v) for t, v in self.vectors.items()}
