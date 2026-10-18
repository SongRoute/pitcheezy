"""Seal for the single performance look.

Labels and metrics may be computed only when *all* hold:

1. the registration file's sha256 equals the value pinned in every ledger;
2. current UTC time >= ``analysis.seal.not_before_utc`` (after the last
   Series game + the label-freeze delay), or an explicit early-unseal entry
   allowed by ``analysis.seal.early_unseal_allowed_if`` is present in the
   unseal file (Series ended early is still covered by the date rule; no other
   early reason is accepted by default);
3. an UNSEAL.json written by the named analyst exists, quoting the
   registration sha256 and the analyst id from the registration;
4. no LOOK.done marker exists yet (single look). The marker is created with
   O_EXCL *before* any label is read, so a crashed look still counts as the
   look and must be reported as such.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Optional


class SealedError(PermissionError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_utc(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {s}")
    return t.astimezone(dt.timezone.utc)


def check_unsealed(registration_path: Path, job_dir: Path, *,
                   now_utc: Optional[dt.datetime] = None) -> dict:
    """Return the parsed registration if the look may proceed, else raise."""
    reg_sha = file_sha256(registration_path)
    reg = json.loads(Path(registration_path).read_text())
    seal = reg["analysis"]["seal"]
    if reg.get("review", {}).get("status") != "PASS":
        raise SealedError("registration has no independent-review PASS")
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now < _parse_utc(seal["not_before_utc"]):
        raise SealedError(f"sealed until {seal['not_before_utc']}")
    unseal_path = Path(job_dir) / seal["unseal_file"]
    if not unseal_path.exists():
        raise SealedError(f"missing {unseal_path.name}")
    unseal = json.loads(unseal_path.read_text())
    if unseal.get("registration_sha256") != reg_sha:
        raise SealedError("UNSEAL does not quote the current registration sha256")
    if unseal.get("analyst") != seal["analyst"]:
        raise SealedError("UNSEAL analyst differs from the registered analyst")
    if (Path(job_dir) / seal["look_marker"]).exists():
        raise SealedError("the single performance look was already taken")
    reg["_sha256"] = reg_sha
    return reg


def take_look(job_dir: Path, marker_name: str, payload: dict) -> Path:
    """Create the look marker atomically; a second call raises."""
    p = Path(job_dir) / marker_name
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as e:
        raise SealedError("the single performance look was already taken") from e
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    return p
