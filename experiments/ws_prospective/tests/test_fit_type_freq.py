"""B2 comparator: synthetic package only (no real FIT rows)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ws_prospective import fit_type_freq as F  # noqa: E402
from ws_prospective.adapters import FIT_COUNTS, PinMismatch, resolve  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _package(tmp, *, break_sum=False):
    rng = np.random.default_rng(0)
    rows = []  # build rows whose FIT y7 marginal equals FIT_COUNTS exactly (scaled down is not
    # allowed by the check, so use the real totals with random types)
    y7 = np.concatenate([np.full(c, k) for k, c in enumerate(FIT_COUNTS)])
    if break_sum:
        y7 = y7[:-1]
    n = len(y7)
    ptype = rng.integers(1, len(F.TYPES19), size=n).astype(np.int16)
    keys = np.stack([np.full(n, 1), np.arange(n) // 10 + 1, np.arange(n) % 10 + 1], 1).astype("<i8")
    extra = 5  # OOF rows must be ignored
    y7 = np.concatenate([y7, np.zeros(extra, int)])
    ptype = np.concatenate([ptype, np.ones(extra, np.int16)])
    keys = np.concatenate([keys, np.stack([np.full(extra, 2), np.arange(extra) + 1,
                                           np.ones(extra)], 1).astype("<i8")])
    fit = np.concatenate([np.ones(n, bool), np.zeros(extra, bool)])
    np.savez(tmp / "corpus-subset.npz", keys=keys, y7=y7, fit=fit)
    np.savez(tmp / "seq-inputs.npz", game_pk=keys[:, 0].astype(np.int32),
             at_bat_number=keys[:, 1].astype(np.int32), pitch_number=keys[:, 2].astype(np.int16),
             pitch_type=ptype)
    man = {"files": {k: {"sha256": _sha(tmp / k)} for k in ("corpus-subset.npz", "seq-inputs.npz")}}
    (tmp / "manifest.json").write_text(json.dumps(man))
    return ptype[:n], y7[:n]


def test_counts_reproduce_marginal_and_adapter_vectors(tmp_path):
    ptype, y7 = _package(tmp_path)
    obj = F.compute_counts(tmp_path, expected_manifest_sha256=_sha(tmp_path / "manifest.json"))
    ff = obj["counts"]["FF"]
    i = F.TYPES19.index("FF")
    assert ff == np.bincount(y7[ptype == i], minlength=7).tolist()
    digest = F.write_counts(obj, tmp_path / "FIT-TYPE-COUNTS.json")
    reg = {"candidate_pitch_types": ["FF", "SL", "XX"],
           "model_sources": {"FIT_TYPE_FREQ_7": {"counts_path": str(tmp_path / "FIT-TYPE-COUNTS.json"),
                                                 "counts_sha256": digest}}}
    a = resolve("FIT_TYPE_FREQ_7").build(reg, {}, None)
    out = a.predict(None)
    assert set(out) == {"FF", "SL"}  # unknown/zero types omitted, never zero-filled
    assert out["FF"] == pytest.approx([(c + .5) / (sum(ff) + 3.5) for c in ff])
    assert all(abs(sum(v) - 1) < 1e-12 for v in out.values())
    with pytest.raises(FileExistsError):
        F.write_counts(obj, tmp_path / "FIT-TYPE-COUNTS.json")


def test_counts_refuse_when_marginal_differs_or_pins_change(tmp_path):
    _package(tmp_path, break_sum=True)
    with pytest.raises(ValueError):
        F.compute_counts(tmp_path, expected_manifest_sha256=_sha(tmp_path / "manifest.json"))
    with pytest.raises(PinMismatch):
        F.compute_counts(tmp_path, expected_manifest_sha256="0" * 64)
