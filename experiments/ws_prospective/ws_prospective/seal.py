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
    not_before = _parse_utc(seal["not_before_utc"])
    if now < not_before:
        raise SealedError(f"sealed until {seal['not_before_utc']}")
    unseal_path = Path(job_dir) / seal["unseal_file"]
    if not unseal_path.exists():
        raise SealedError(f"missing {unseal_path.name}")
    unseal = json.loads(unseal_path.read_text())
    # Addenda are separate files listed (with sha) in UNSEAL.json; the only
    # field they may carry is a LATER not_before_utc.
    for add in unseal.get("addenda", []):
        ap = Path(job_dir) / add["file"]
        if not ap.exists() or file_sha256(ap) != add.get("sha256"):
            raise SealedError(f"addendum {add.get('file')} missing or sha mismatch")
        body = json.loads(ap.read_text())
        if set(body) - {"not_before_utc", "reason"}:
            raise SealedError("addendum may only move not_before_utc later")
        if "not_before_utc" in body:
            nb = _parse_utc(body["not_before_utc"])
            if nb < not_before:
                raise SealedError("addendum may only move not_before_utc later")
            not_before = nb
            if now < not_before:
                raise SealedError(f"sealed until {body['not_before_utc']} (addendum)")
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


# ---------------------------------------------------------------- per-game EXEC files
PLACEHOLDER_MARK = "<PLACEHOLDER"
_GAME_ROW_KEYS = ("official_date", "first_pitch_utc", "store_path", "profile_path", "profile_sha256")


def normalized_common(reg: dict) -> dict:
    """Registration content that must be identical across the per-game EXEC
    files: everything except execution.games[] rows, of which only the ordered
    game_pk list is kept."""
    import copy
    body = copy.deepcopy(reg)
    games = body.get("execution", {}).pop("games", [])
    body["execution"]["_game_pks"] = [g.get("game_pk") for g in games]
    return body


def _row_filled(row: dict) -> bool:
    return all(k in row and PLACEHOLDER_MARK not in str(row[k]) for k in _GAME_ROW_KEYS)


def check_unsealed_set(registration_paths, job_dir: Path, *,
                       now_utc: Optional[dt.datetime] = None) -> dict:
    """Per-game EXEC files (PM 2026-10-19, REVIEW-EXEC-001 B-2 option 1a).

    UNSEAL.json lists {"registrations": [{"file", "sha256", "game_pk"}, ...]}; the
    set of listed shas must equal the set of files passed.  All files must share
    the same parent sha and be identical outside execution.games[] rows (same
    ordered game_pk list); each file's own game row must be filled, and a game
    row filled in several files must be identical everywhere.  Seal date,
    analyst, addenda and the single-look marker are then checked exactly as in
    check_unsealed, using the common content.  Returns the common registration
    with execution.games = each game's row from its own file, "_exec" =
    {game_pk: sha} and "_sha256" = the Game-1 file's sha."""
    paths = [Path(p) for p in registration_paths]
    if not paths:
        raise SealedError("no registration files")
    shas = [file_sha256(p) for p in paths]
    regs = [json.loads(p.read_text()) for p in paths]
    if len(set(shas)) != len(shas):
        raise SealedError("duplicate registration file")
    commons = [normalized_common(r) for r in regs]
    if any(c != commons[0] for c in commons[1:]):
        raise SealedError("per-game EXEC files differ outside execution.games[]")
    if len({json.dumps(r.get("parent"), sort_keys=True) for r in regs}) != 1:
        raise SealedError("per-game EXEC files do not share one parent")
    common = commons[0]
    unseal_path = Path(job_dir) / common["analysis"]["seal"]["unseal_file"]
    if not unseal_path.exists():
        raise SealedError(f"missing {unseal_path.name}")
    listed = json.loads(unseal_path.read_text()).get("registrations", [])
    by_sha = {x.get("sha256"): x for x in listed}
    if set(by_sha) != set(shas) or len(listed) != len(shas):
        raise SealedError("UNSEAL registrations do not list exactly these EXEC files")
    owner = {}
    for p, s, r in zip(paths, shas, regs):
        pk = int(by_sha[s]["game_pk"])
        if pk in owner:
            raise SealedError(f"two EXEC files claim game {pk}")
        rows = {int(g["game_pk"]): g for g in r["execution"]["games"]
                if str(g.get("game_pk")).isdigit() or isinstance(g.get("game_pk"), int)}
        if pk not in rows or not _row_filled(rows[pk]):
            raise SealedError(f"{p.name}: own game {pk} row missing or not filled")
        owner[pk] = (s, rows[pk])
    for r in regs:
        for g in r["execution"]["games"]:
            pk = int(g["game_pk"])
            if _row_filled(g) and pk in owner and g != owner[pk][1]:
                raise SealedError(f"game {pk} row differs between EXEC files")
    # seal checks on a temporary single file of the common content
    import tempfile
    first = min(owner, key=lambda pk: [int(x) for x in common["execution"]["_game_pks"]].index(pk))
    merged = json.loads(json.dumps(regs[0]))
    merged["execution"]["games"] = [owner[int(pk)][1] for pk in common["execution"]["_game_pks"]
                                    if int(pk) in owner]
    with tempfile.TemporaryDirectory() as td:
        tmp_reg = Path(td) / "COMMON.json"
        tmp_reg.write_text(json.dumps(merged))
        tmp_job = Path(td) / "job"
        tmp_job.mkdir()
        unseal = json.loads(unseal_path.read_text())
        for add in unseal.get("addenda", []):  # addenda live in the real job dir
            src = Path(job_dir) / add["file"]
            if src.exists():
                (tmp_job / add["file"]).write_bytes(src.read_bytes())
        unseal["registration_sha256"] = file_sha256(tmp_reg)
        (tmp_job / common["analysis"]["seal"]["unseal_file"]).write_text(json.dumps(unseal))
        if (Path(job_dir) / common["analysis"]["seal"]["look_marker"]).exists():
            raise SealedError("the single performance look was already taken")
        reg = check_unsealed(tmp_reg, tmp_job, now_utc=now_utc)
    reg["_exec"] = {pk: s for pk, (s, _) in owner.items()}
    reg["_sha256"] = owner[first][0]
    return reg
