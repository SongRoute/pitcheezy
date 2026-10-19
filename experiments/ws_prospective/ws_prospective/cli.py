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

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")  # before any numeric import (CPU, single thread)


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


def runtime_versions() -> dict:
    """python / numpy / torch / lightgbm versions and thread settings actually loaded."""
    import platform
    v = {"python": platform.python_version(),
         "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    for name in ("numpy", "pandas", "torch", "lightgbm", "scipy"):
        mod = sys.modules.get(name)
        if mod is not None:
            v[name] = getattr(mod, "__version__", "unknown")
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)
        v["torch_threads"] = torch.get_num_threads()
    return v


def cmd_backup_store(a) -> int:
    """B4: after the launcher finished the date, copy the store (sqlite online
    backup) into the RS-01 job folder and append its sha256 to STORE-BACKUPS.jsonl."""
    import datetime as dt
    from .store_source import backup_store
    dest = Path(a.dest)
    digest = backup_store(Path(a.store), dest)
    rec = {"store": str(Path(a.store)), "copy": str(dest), "sha256": digest,
           "bytes": dest.stat().st_size, "at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "games": [int(x) for x in a.games.split(",")] if a.games else []}
    with open(Path(a.record), "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps(rec))
    return 0


def _in_protected_window(now_utc=None) -> bool:
    """10/19 rehearsal window 05:53-10:00 KST (= 10/18 20:53 - 10/19 01:00 UTC)."""
    import datetime as dt
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    lo = dt.datetime(2026, 10, 18, 20, 53, tzinfo=dt.timezone.utc)
    hi = dt.datetime(2026, 10, 19, 1, 0, tzinfo=dt.timezone.utc)
    return lo <= now < hi


def cmd_dryrun(a) -> int:
    """Metadata-only replay of a finished game's BACKUP store copy (DRYRUN-REGISTRATION).

    Builds the registered adapters, replays snapshots in write order
    (LiveStoreSource(max_seq)), never stores probability vectors, never reads
    labels, and writes DRYRUN-SUMMARY-<proc>.json with counts, abstain reasons,
    PA-first coverage, compute ms, peak RSS and runtime versions."""
    import datetime as dt
    from .adapters import resolve
    from .analyzer import ops_check
    from .ledger import Ledger
    from .runner import GameRunner, Limits
    from .store_source import LiveStoreSource, open_ro, snapshot_seqs

    plan_path = Path(a.plan)
    plan = json.loads(plan_path.read_text())
    if plan.get("execute") is not True or plan.get("review", {}).get("status") != "PASS":
        print("refused: dry-run plan not approved (execute!=true or no review PASS)")
        return 2
    if _in_protected_window():
        print("refused: inside the 10/19 05:53-10:00 KST rehearsal window")
        return 2
    src_reg_path = Path(plan["model_registration"]["path"])
    if file_sha256(src_reg_path) != plan["model_registration"]["sha256"]:
        print("refused: model registration sha differs from the plan")
        return 2
    reg = json.loads(src_reg_path.read_text())
    inp = plan["input"]
    store = Path(a.store)
    if inp.get("store_copy_sha256") and file_sha256(store) != inp["store_copy_sha256"]:
        print("refused: store copy sha differs from the plan (use the backup copy, never the live store)")
        return 2
    out = Path(a.out)
    if out.exists():
        print("refused: fresh output directory required")
        return 2
    procs = plan["processes"]
    if a.proc not in procs or sys.executable != procs[a.proc]["python"]:
        print("refused: process/interpreter not registered in the plan")
        return 2
    try:
        os.nice(int(plan["bounds"].get("nice", 10)))
    except OSError:
        pass
    game = {"game_pk": int(inp["game_pk"]), "official_date": inp["official_date"],
            "store_path": str(store), "profile_path": inp.get("profile_path"),
            "profile_sha256": inp.get("profile_sha256")}
    out.mkdir(parents=True)
    forecasters = [resolve(m).build(reg, game, out) for m in procs[a.proc]["models"]]
    versions = runtime_versions()
    led = Ledger(out / f"game_{game['game_pk']}" / f"ledger-{a.proc}.sqlite", game_pk=game["game_pk"],
                 registration_sha256=file_sha256(plan_path), code_commit=_code_commit(),
                 meta={"kind": "DRYRUN_METADATA_ONLY", "store_path": str(store),
                       "profile_sha256": str(game["profile_sha256"]), "runtime_versions": versions})
    b = plan["bounds"]
    src = LiveStoreSource(store, game["game_pk"], max_seq=0)
    runner = GameRunner(game["game_pk"], src, forecasters, led, str(game["profile_sha256"]),
                        Limits(max_rss_mb=b["max_rss_mb"], max_compute_ms=b["max_compute_ms"],
                               record_post_hoc_replays=False, store_probabilities=False))
    import time
    t0 = time.monotonic()
    seqs = snapshot_seqs(store, game["game_pk"])
    led.event("DRYRUN_START", n_snapshots=len(seqs))
    stopped = None
    for seq in seqs:
        if time.monotonic() - t0 > b["max_wall_s"]:
            stopped = "MAX_WALL"
            break
        reason = resource_breach(out, {"min_free_memory_mb": b["min_free_memory_mb"],
                                       "min_free_disk_gb": b["min_free_disk_gb"],
                                       "max_output_mb": b["max_output_mb"]})
        if reason:
            stopped = "RESOURCE:" + reason
            break
        src.max_seq = seq
        runner.step()
    led.event("RUN_STOP", stopped=stopped, peak_rss_mb=round(_peak_rss_mb(), 1))
    led.close()
    con = open_ro(store)
    try:
        n_pitches, n_pa_first = con.execute(
            "SELECT COUNT(DISTINCT at_bat_number || '-' || pitch_number), "
            "COUNT(DISTINCT CASE WHEN pitch_number=1 THEN at_bat_number END) FROM pitch_events "
            "WHERE game_pk=? AND outcome IN ('new','correction')", (game["game_pk"],)).fetchone()
    finally:
        con.close()
    summary = {"kind": "DRYRUN_METADATA_ONLY_NO_LABELS_NO_METRICS", "proc": a.proc,
               "plan_sha256": file_sha256(plan_path), "store_copy_sha256": file_sha256(store),
               "game_pk": game["game_pk"], "n_snapshots": len(seqs), "stopped": stopped,
               "store_pitch_keys": n_pitches, "store_pa_first_keys": n_pa_first,
               "wall_s": round(time.monotonic() - t0, 1), "peak_rss_mb": round(_peak_rss_mb(), 1),
               "runtime_versions": versions,
               "ledger": ops_check(led.path),
               "note": "replay timing classes are not meaningful; coverage = intents per store pitch key",
               "at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    (out / f"DRYRUN-SUMMARY-{a.proc}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"summary": str(out / f"DRYRUN-SUMMARY-{a.proc}.json"), "stopped": stopped}))
    return 0 if stopped is None else 3


def _peak_rss_mb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


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
    if str(Path(a.store)) != str(g.get("store_path")):
        print("refused: --store differs from registration.execution.games[].store_path")
        return 2
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
    if sys.executable != procs[a.proc].get("python", sys.executable):
        print(f"refused: interpreter {sys.executable} != registered for process {a.proc}")
        return 2
    forecasters = [resolve(m).build(reg, g, out) for m in procs[a.proc]["models"]]
    versions = runtime_versions()
    want = procs[a.proc].get("versions")
    if want and any(versions.get(k) != v for k, v in want.items()):
        print(f"refused: runtime versions {versions} != registered {want}")
        return 2
    led = Ledger(out / f"game_{a.game}" / f"ledger-{a.proc}.sqlite", game_pk=int(a.game),
                 registration_sha256=file_sha256(reg_path), code_commit=commit,
                 meta={"profile_sha256": g["profile_sha256"], "store_path": str(a.store),
                       "process": a.proc, "models": procs[a.proc]["models"],
                       "runtime_versions": versions})
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
    games = {int(k): {"ledgers": [Path(x) for x in v["ledgers"]], "store": Path(v["store"]),
                      "store_sha256": v["store_sha256"]}
             for k, v in json.loads(Path(a.games).read_text()).items()}
    regs = [Path(x) for x in a.registration]
    sealed_look(regs[0] if len(regs) == 1 else regs, Path(a.job), games, leaf_of=labels.leaf_of,
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
    d = sub.add_parser("dryrun")
    for k in ("--plan", "--store", "--out", "--proc"):
        d.add_argument(k, required=True)
    b = sub.add_parser("backup-store")
    for k in ("--store", "--dest", "--record"):
        b.add_argument(k, required=True)
    b.add_argument("--games", default="")
    n = sub.add_parser("analyze")
    n.add_argument("--registration", required=True, nargs="+",
                   help="one EXEC file, or every per-game EXEC file (listed in UNSEAL.json)")
    for k in ("--job", "--games"):
        n.add_argument(k, required=True)
    a = p.parse_args(argv)
    return {"verify": cmd_verify, "run": cmd_run, "ops-check": cmd_ops,
            "backup-store": cmd_backup_store, "dryrun": cmd_dryrun,
            "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
