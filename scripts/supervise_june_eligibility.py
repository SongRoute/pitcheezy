"""Single-attempt, registered June eligibility preparation supervisor.

The worker owns the shared .heavy.lock. This supervisor owns only its queue
lock, verifies registration and source pins, and charges worker Popen-to-wait
wall against a fixed 600-second cap. It never retries or runs a model.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


COORD = Path(__file__).resolve().parent
RUNS = COORD.parents[1]
PROTOCOL = "june_eligibility_single_supervisor_v1"
REGISTRATION_PROTOCOL = "june_eligibility_registration_v1"
JOB = "june-prepare"
CAP = 600
COMMON_FIELDS = ("code_root", "code_commit", "config_path", "config_sha256",
                 "source_hashes", "supervisor_sha256", "python_executable", "argv",
                 "output_dir", "completion_path", "completion_checks", "worker_heavy_lock",
                 "cap_seconds")


class Interrupted(BaseException):
    pass


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def good_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def atomic(path: Path, value: dict) -> None:
    require(not path.exists(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists(), f"refusing overwrite: {path}")
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate(plan: dict, plan_path: Path) -> None:
    required = {"protocol", "registration_path", "registration_sha256", "config_path",
                "config_sha256", "supervisor_sha256", "source_hashes", "code_root",
                "code_commit", "python_executable", "argv", "output_dir", "completion_path",
                "completion_checks", "worker_heavy_lock", "cap_seconds"}
    require(set(plan) == required and plan["protocol"] == PROTOCOL, "invalid plan schema")
    require(plan_path.resolve().parent == COORD, "plan outside coordination root")
    require(plan["cap_seconds"] == CAP and type(plan["cap_seconds"]) is int, "cap must be 600 seconds")
    for field in ("registration_sha256", "config_sha256", "supervisor_sha256"):
        require(good_hash(plan[field]), f"invalid {field}")
    require(isinstance(plan["code_commit"], str) and len(plan["code_commit"]) == 40
            and all(c in "0123456789abcdef" for c in plan["code_commit"]), "invalid code commit")
    registration = Path(plan["registration_path"])
    config = Path(plan["config_path"])
    code_root = Path(plan["code_root"])
    python = Path(plan["python_executable"])
    output = Path(plan["output_dir"])
    completion = Path(plan["completion_path"])
    lock = Path(plan["worker_heavy_lock"])
    require(registration.is_absolute() and registration.parent == COORD and registration.is_file(), "registration path")
    require(config.is_absolute() and config.is_file(), "config path")
    require(code_root.is_absolute() and code_root.is_dir(), "code root")
    require(python.is_absolute() and python.is_file(), "Python executable")
    require(output == RUNS / "ML-JUNE-ELIGIBILITY-v1", "output differs from registered fresh run")
    require(completion == output / "manifest.json", "completion must be output manifest")
    require(lock == RUNS / ".heavy.lock" and lock.is_file(), "wrong shared heavy lock")
    sources = plan["source_hashes"]
    require(isinstance(sources, dict) and sources, "source pins required")
    for path, digest in sources.items():
        source = Path(path)
        require(source.is_absolute() and source.is_file() and source.is_relative_to(code_root)
                and good_hash(digest), f"invalid source pin: {path}")
    argv = plan["argv"]
    require(isinstance(argv, list) and len(argv) >= 4 and all(isinstance(s, str) for s in argv), "invalid argv")
    require(argv[0] == str(python) and Path(argv[1]).is_file() and argv[1] in sources,
            "worker script must use pinned Python and source")
    require(str(config) in argv and str(output) in argv, "worker argv lacks config/output")
    validate_completion_checks(plan)


def validate_completion_checks(plan: dict) -> None:
    checks = plan["completion_checks"]
    require(isinstance(checks, dict) and checks and all(isinstance(k, str) and isinstance(v, (str, int, bool))
            for k, v in checks.items()), "completion checks required")
    require(any(k.split(".")[-1] == "status" and v == "complete" for k, v in checks.items()),
            "completion status check required")
    require(any(k.split(".")[-1] == "config_sha256" and v == plan["config_sha256"] for k, v in checks.items()),
            "completion config identity check required")
    for leaf in ("registered_code_commit", "execution_head"):
        require(any(k.split(".")[-1] == leaf and v == plan["code_commit"] for k, v in checks.items()),
                f"completion {leaf} identity check required")


def validate_registration(plan: dict) -> None:
    registration = read(Path(plan["registration_path"]))
    required = set(COMMON_FIELDS) | {"protocol", "registration_commit_d", "created_utc"}
    require(set(registration) == required and registration["protocol"] == REGISTRATION_PROTOCOL,
            "invalid registration schema")
    commit = registration["registration_commit_d"]
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in "0123456789abcdef" for c in commit), "invalid registration D commit")
    require(isinstance(registration["created_utc"], str) and registration["created_utc"],
            "registration creation time missing")
    for key in COMMON_FIELDS:
        require(registration[key] == plan[key], f"registration/plan mismatch: {key}")


def preflight(plan: dict) -> None:
    require(sha(Path(__file__)) == plan["supervisor_sha256"], "supervisor source hash changed")
    require(sha(Path(plan["registration_path"])) == plan["registration_sha256"], "registration hash changed")
    validate_registration(plan)
    require(sha(Path(plan["config_path"])) == plan["config_sha256"], "config hash changed")
    for path, expected in plan["source_hashes"].items():
        require(sha(Path(path)) == expected, f"source hash changed: {path}")
    current = subprocess.run(["git", "-C", plan["code_root"], "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    require(current == plan["code_commit"], "registered code commit changed")
    dirty = subprocess.run(["git", "-C", plan["code_root"], "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    require(not dirty, "execution checkout has tracked changes")
    output = Path(plan["output_dir"])
    require(not output.exists() or not any(output.iterdir()), "output directory is not fresh")


@contextmanager
def queue_lock():
    with (COORD / ".queue.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def terminate(process: subprocess.Popen, grace: float = 5.0) -> int | None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


@contextmanager
def ignore_cleanup_signals():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, signal.SIG_IGN)
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def completion_sha(plan: dict) -> str:
    path = Path(plan["completion_path"])
    require(path.is_file(), "worker exited zero without completion manifest")
    manifest = read(path)
    require(isinstance(manifest, dict), "completion manifest is not an object")
    for key, expected in plan["completion_checks"].items():
        node = manifest
        for part in key.split("."):
            require(isinstance(node, dict) and part in node, f"completion field missing: {key}")
            node = node[part]
        require(type(node) is type(expected) and node == expected, f"completion field mismatch: {key}")
    return sha(path)


def run(plan_path: Path) -> dict:
    plan = read(plan_path)
    validate(plan, plan_path)
    plan_hash = sha(plan_path)
    start = COORD / "ledger" / "starts" / (JOB + ".json")
    end = COORD / "ledger" / "ends" / (JOB + ".json")
    refusal = COORD / "ledger" / "refusal.json"
    with queue_lock():
        require(not start.exists() and not end.exists() and not refusal.exists(), "attempt already exists; no automatic retry")
        try:
            preflight(plan)
        except BaseException as error:
            atomic(refusal, {"protocol": PROTOCOL, "status": "refused_preflight", "at_utc": utc(),
                             "plan_sha256": plan_hash, "registration_sha256": plan["registration_sha256"],
                             "config_sha256": plan["config_sha256"], "supervisor_sha256": plan["supervisor_sha256"],
                             "source_hashes": plan["source_hashes"],
                             "reason": f"{type(error).__name__}: {error}"})
            raise
        atomic(start, {"protocol": PROTOCOL, "job_id": JOB, "plan_sha256": plan_hash,
                       "registration_sha256": plan["registration_sha256"],
                       "config_sha256": plan["config_sha256"], "source_hashes": plan["source_hashes"],
                       "supervisor_sha256": plan["supervisor_sha256"], "code_commit": plan["code_commit"],
                       "cap_seconds": CAP, "argv": plan["argv"], "worker_heavy_lock": plan["worker_heavy_lock"],
                       "at_utc": utc()})
        outcome = "failed"
        exit_code: int | None = None
        note = ""
        artifact_hash: str | None = None
        process: subprocess.Popen | None = None
        begun: float | None = None
        elapsed = 0.0
        postflight_started: float | None = None
        old = {}

        def interrupted(_signum, _frame):
            raise Interrupted()

        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.getsignal(sig)
            signal.signal(sig, interrupted)
        try:
            stdout_path = COORD / "ledger" / "worker.stdout.log"
            stderr_path = COORD / "ledger" / "worker.stderr.log"
            require(not stdout_path.exists() and not stderr_path.exists(), "worker logs already exist")
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                begun = time.monotonic()
                process = subprocess.Popen(plan["argv"], stdout=stdout, stderr=stderr,
                                           start_new_session=True, cwd=plan["code_root"])
                try:
                    exit_code = process.wait(timeout=CAP)
                    elapsed = time.monotonic() - begun
                    postflight_started = time.monotonic()
                    if elapsed > CAP:
                        outcome, note = "timeout", "worker Popen-to-wait wall exceeded 600 seconds"
                    elif exit_code == 0:
                        artifact_hash = completion_sha(plan)
                        outcome = "completed"
                    else:
                        note = f"worker exit code {exit_code}"
                except subprocess.TimeoutExpired:
                    outcome, note = "timeout", "worker Popen-to-wait timeout"
                    with ignore_cleanup_signals():
                        exit_code = terminate(process)
                    elapsed = time.monotonic() - begun
        except Interrupted:
            outcome, note = "interrupted", "supervisor received SIGINT/SIGTERM"
            if process is not None:
                with ignore_cleanup_signals():
                    exit_code = terminate(process)
            if postflight_started is None:
                elapsed = time.monotonic() - begun if begun is not None else 0.0
        except BaseException as error:
            note = f"{type(error).__name__}: {error}"
            if process is not None and process.poll() is None:
                with ignore_cleanup_signals():
                    exit_code = terminate(process)
            if postflight_started is None:
                elapsed = time.monotonic() - begun if begun is not None else 0.0
        finally:
            for sig in old:
                signal.signal(sig, signal.SIG_IGN)
            try:
                atomic(end, {"protocol": PROTOCOL, "job_id": JOB, "plan_sha256": plan_hash,
                             "registration_sha256": plan["registration_sha256"],
                             "config_sha256": plan["config_sha256"], "source_hashes": plan["source_hashes"],
                             "supervisor_sha256": plan["supervisor_sha256"], "cap_seconds": CAP,
                             "outcome": outcome, "exit_code": exit_code,
                             "popen_wait_seconds": elapsed, "completion_path": plan["completion_path"],
                             "completion_sha256": artifact_hash,
                             "postflight_seconds": (time.monotonic() - postflight_started
                                                    if postflight_started is not None else 0.0),
                             "note": note, "at_utc": utc()})
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
    return read(end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        plan = read(args.plan)
        validate(plan, args.plan)
        preflight(plan)
        print(json.dumps({"protocol": PROTOCOL, "status": "preflight_passed", "plan_sha256": sha(args.plan)}))
        return 0
    result = run(args.plan)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["outcome"] == "completed" else 124 if result["outcome"] == "timeout" else 1


if __name__ == "__main__":
    sys.exit(main())
