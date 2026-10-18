"""Command line for RS-01.

  python -m ws_prospective.cli verify   --registration R
  python -m ws_prospective.cli run      --registration R --game PK --store STORE --out DIR --proc A
  python -m ws_prospective.cli ops-check --out DIR
  python -m ws_prospective.cli analyze  --registration R --job JOB --games GAMES.json

``run`` refuses unless the registration says execute=true, has an independent
review PASS, lists the game, and every pinned artifact hash still matches.
It lowers its own priority, stops on a STOP file in --out (no signals needed),
and never writes outside --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .seal import file_sha256


def verify_artifacts(reg: dict) -> list[str]:
    bad = []
    for a in reg.get("artifacts", []):
        p = Path(a["path"])
        if not p.exists():
            bad.append(f"missing {p}")
        elif a.get("sha256") and file_sha256(p) != a["sha256"]:
            bad.append(f"sha mismatch {p}")
    return bad


def _code_commit() -> str:
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def free_memory_mb() -> float | None:
    """macOS: free + inactive + speculative pages from vm_stat (None if unknown)."""
    try:
        txt = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    import re
    m = re.search(r"page size of (\d+) bytes", txt)
    if not m:
        return None
    page = int(m.group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        mm = re.search(key + r":\s+(\d+)", txt)
        pages += int(mm.group(1)) if mm else 0
    return pages * page / 2**20


def resource_breach(out: Path, lim: dict) -> str | None:
    import shutil
    free = free_memory_mb()
    if free is not None and free < lim["min_free_memory_mb"]:
        return f"free memory {free:.0f} MB < {lim['min_free_memory_mb']}"
    if shutil.disk_usage(out).free / 2**30 < lim["min_free_disk_gb"]:
        return "free disk below limit"
    used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if used / 2**20 > lim["max_output_mb"]:
        return f"output {used / 2**20:.0f} MB > {lim['max_output_mb']}"
    return None


def cmd_verify(a) -> int:
    reg = json.loads(Path(a.registration).read_text())
    bad = verify_artifacts(reg)
    print(json.dumps({"registration_sha256": file_sha256(a.registration), "problems": bad}, indent=2))
    return 1 if bad else 0


def cmd_run(a) -> int:
    from .adapters import resolve
    from .ledger import Ledger
    from .runner import GameRunner, Limits
    from .store_source import LiveStoreSource

    reg_path = Path(a.registration)
    reg = json.loads(reg_path.read_text())
    if reg.get("execute") is not True or reg.get("review", {}).get("status") != "PASS":
        print("refused: registration is not executable (execute!=true or no review PASS)")
        return 2
    games = {int(g["game_pk"]): g for g in reg.get("execution", {}).get("games", [])}
    if int(a.game) not in games:
        print("refused: game not listed in registration.execution.games")
        return 2
    bad = verify_artifacts(reg)
    if bad:
        print("refused: " + "; ".join(bad))
        return 2
    g = games[int(a.game)]
    commit = _code_commit()
    if commit.endswith("-dirty") or commit != reg["execution"].get("runner_commit", commit):
        print(f"refused: runner commit {commit} != registered")
        return 2
    try:
        os.nice(int(reg["resources"].get("nice", 10)))
    except OSError:
        pass
    out = Path(a.out)
    procs = reg["execution"]["processes"]
    if a.proc not in procs:
        print(f"refused: process {a.proc!r} not registered")
        return 2
    led = Ledger(out / f"game_{a.game}" / f"ledger-{a.proc}.sqlite", game_pk=int(a.game),
                 registration_sha256=file_sha256(reg_path), code_commit=commit,
                 meta={"profile_sha256": g["profile_sha256"], "store_path": str(a.store)})
    forecasters = [resolve(m).build(reg, g, out) for m in procs[a.proc]["models"]]
    lim = reg["resources"]
    runner = GameRunner(int(a.game), LiveStoreSource(Path(a.store), int(a.game)), forecasters, led,
                        g["profile_sha256"],
                        Limits(max_rss_mb=lim["max_rss_mb"], max_compute_ms=lim["max_compute_ms"],
                               poll_interval_s=lim["poll_interval_s"],
                               record_post_hoc_replays=False))
    stop = out / "STOP"

    last_check = [0.0]

    def sleep(s):
        import time
        if stop.exists():
            led.event("STOP_FILE")
            raise SystemExit(0)
        if time.monotonic() - last_check[0] > 60:
            last_check[0] = time.monotonic()
            reason = resource_breach(out, lim)
            if reason:
                led.event("RESOURCE_ABORT", reason=reason)
                raise SystemExit(3)
        time.sleep(s)

    runner.run(sleep=sleep)
    return 0


def cmd_ops(a) -> int:
    from .analyzer import ops_check
    from .ledger import iter_ledgers
    print(json.dumps({str(p): ops_check(p) for p in iter_ledgers(Path(a.out))}, indent=2))
    return 0


def cmd_analyze(a) -> int:
    from .analyzer import sealed_look
    from .store_source import final_label_rows
    from . import labels
    games = {int(k): {"ledgers": [Path(x) for x in v["ledgers"]], "store": Path(v["store"])}
             for k, v in json.loads(Path(a.games).read_text()).items()}
    sealed_look(Path(a.registration), Path(a.job), games, leaf_of=labels.leaf_of,
                type_of=labels.type_of, label_loader=final_label_rows)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ws_prospective")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("--registration", required=True)
    r = sub.add_parser("run")
    for k in ("--registration", "--game", "--store", "--out", "--proc"):
        r.add_argument(k, required=True)
    o = sub.add_parser("ops-check"); o.add_argument("--out", required=True)
    n = sub.add_parser("analyze")
    for k in ("--registration", "--job", "--games"):
        n.add_argument(k, required=True)
    a = p.parse_args(argv)
    return {"verify": cmd_verify, "run": cmd_run, "ops-check": cmd_ops,
            "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
