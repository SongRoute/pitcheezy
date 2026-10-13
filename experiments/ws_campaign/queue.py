"""Finite, detached 87-job campaign queue. This process only manages metadata."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from experiments.ws_campaign import runtime

GPU = "/Users/song/.local/bin/gpu"
TERMINAL = {"COMPLETED", "FAILED", "QUARANTINED"}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temp = stream.name
    os.replace(temp, path)


def command(*argv):
    return subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False, timeout=120, cwd=runtime.ROOT).stdout


def validate(manifest_path, launch_grant_path):
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest["jobs"]
    validate_inventory(jobs)
    root_grant = json.loads(launch_grant_path.read_text())
    if root_grant.get("recovery_amendment_sha256"):
        jobs = [job for job in jobs if job["arm"] == "K"]
        runtime.require(len(jobs) == 12 and sum(j["calibrations"] for j in jobs) == 36, "exact K recovery scope")
    for job in jobs:
        runtime.load_job(manifest_path, launch_grant_path, job["id"])
    return jobs


def validate_inventory(jobs):
    runtime.require(len(jobs) == 87 and len({j["id"] for j in jobs}) == 87, "87 unique slots")
    counts = {"supervised": 0, "pretrain": 0, "teacher": 0}
    for job in jobs:
        counts[job["kind"]] += 1
    runtime.require(counts == {"supervised": 78, "pretrain": 6, "teacher": 3}, "registered 78/6/3 counts")
    runtime.require(sum(j["calibrations"] for j in jobs) == 228, "registered CAL total")
    ids = {j["id"] for j in jobs}
    runtime.require(all(set(j["depends_on"]) <= ids - {j["id"]} for j in jobs), "dependency IDs")
    pending = {j["id"]: set(j["depends_on"]) for j in jobs}
    done = set()
    while pending:
        ready = {name for name, deps in pending.items() if deps <= done}
        runtime.require(bool(ready), "acyclic dependency graph")
        done |= ready
        for name in ready:
            del pending[name]
    attempts = [j["attempt_id"] for j in jobs]
    outputs = [j["output"] for j in jobs]
    runtime.require(len(set(attempts)) == 87 and len(set(outputs)) == 87, "unique attempts and outputs")
    runtime.require(all(j["attempt_id"] and j["gpu_uuid"] and j["common_lock"] for j in jobs),
                    "complete GPU ownership bindings")


def gpu_status():
    output = command(GPU, "st")
    found = {}
    for index, used, total, util in re.findall(r"^GPU(\d+)\s+(\d+)\s*/\s*(\d+) MiB\s+util\s+(\d+)%", output, re.M):
        found[int(index)] = {"used_mib": int(used), "total_mib": int(total), "util": int(util)}
    runtime.require(found, "gpu st failed or format changed")
    return found


def idle_gpu(job, available):
    card = available.get(int(job["gpu_index"]))
    return card is not None and card["used_mib"] <= 1024 and card["util"] <= 10


def remote_status(job_id):
    info = command(GPU, "info", job_id)
    match = re.search(r"상태:\s+(\S+)", info)
    runtime.require(match is not None, "gpu info unavailable: " + job_id)
    return match.group(1)


def pull_json(gpu_id, server_file):
    with tempfile.TemporaryDirectory(prefix="ws-campaign-receipt-") as temp:
        run = subprocess.run([GPU, "pull", gpu_id, server_file, temp], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        runtime.require(run.returncode == 0, "metadata pull failed: " + run.stdout)
        path = Path(temp) / server_file.lstrip("/")
        runtime.require(path.is_file(), "metadata pull absent")
        return json.loads(path.read_text()), runtime.sha(path)


def certify(job, gpu_id):
    receipt, receipt_hash = pull_json(gpu_id, job["output"] + "/receipt.json")
    sup, sup_hash = pull_json(gpu_id, job["output"] + "/SUPERVISOR.json")
    runtime.require(receipt.get("status") == "COMPLETED" and sup.get("status") == "COMPLETED"
                    and sup.get("receipt_sha256") == receipt_hash
                    and receipt.get("job_id") == job["id"]
                    and receipt.get("attempt_id") == job["attempt_id"]
                    and receipt.get("grant_sha256") == job["grant_sha256"]
                    and receipt.get("input_manifest_sha256") == job["input_manifest_sha256"]
                    and receipt.get("exposure_ledger_sha256") == job["exposure_ledger_sha256"]
                    and receipt.get("eval_scores_opened") is False, "metadata-only integrity certification")
    return {"receipt_sha256": receipt_hash, "supervisor_sha256": sup_hash}


def quarantine_failed_dependencies(state, jobs):
    changed = False
    lookup = {job["id"]: job for job in jobs}
    for name, item in state["jobs"].items():
        if item["status"] == "PENDING" and any(
                state["jobs"][dep]["status"] in {"FAILED", "QUARANTINED"}
                for dep in lookup[name]["depends_on"]):
            item.update(status="QUARANTINED", reason="failed dependency")
            changed = True
    return changed


def launch(job, manifest_path, grant_path, available):
    index = int(job["gpu_index"])
    runtime.require(idle_gpu(job, available), "GPU not idle at dispatch")
    result = command(GPU, "run", "-n", "ws-" + job["id"], "-g", str(index), "--",
                     "python", "-m", "experiments.ws_campaign.runtime", "--manifest",
                     str(manifest_path.relative_to(runtime.ROOT)), "--launch-grant",
                     str(grant_path.relative_to(runtime.ROOT)), "--job", job["id"])
    match = re.search(r"잡 시작:\s+(\S+)", result)
    runtime.require(match is not None and re.search(r"상태:\s+(running|ok)\b", result),
                    "gpu run did not confirm dispatch: " + result)
    return match.group(1)


def assert_resumable(state, manifest_hash, launch_hash):
    runtime.require(state["manifest_sha256"] == manifest_hash
                    and state["launch_grant_sha256"] == launch_hash, "queue state binding")
    runtime.require(not any(item["status"] == "CLAIMED" for item in state["jobs"].values()),
                    "unresolved launch claim; human audit required")


def run_queue(manifest_path, launch_grant_path, state_path, poll_seconds):
    runtime.require(manifest_path.is_relative_to(runtime.ROOT)
                    and launch_grant_path.is_relative_to(runtime.ROOT), "snapshot must contain manifest and grant")
    jobs = validate(manifest_path, launch_grant_path)
    cfg = json.loads((runtime.ROOT / json.loads(manifest_path.read_text())["config"]).read_text())
    runtime.require(cfg["budgets"]["max_concurrent_jobs"] == 5
                    and cfg["budgets"]["max_jobs_per_gpu"] == 1
                    and cfg["budgets"]["retries"] == 0
                    and cfg["budgets"]["max_campaign_wall_seconds"] > 0, "global resource budget")
    allowed = {str(path.relative_to(runtime.ROOT)) for path in (manifest_path, launch_grant_path)}
    allowed.update(job["grant"] for job in jobs)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=runtime.ROOT, text=True, capture_output=True, check=True).stdout
    runtime.require(all(line[:2] == "??" and line[3:] in allowed for line in status.splitlines()),
                    "source snapshot has unreviewed changes")
    manifest_hash = runtime.sha(manifest_path)
    launch_hash = runtime.sha(launch_grant_path)
    with (state_path.with_suffix(".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            assert_resumable(state, manifest_hash, launch_hash)
        else:
            state = {"manifest_sha256": manifest_hash, "launch_grant_sha256": launch_hash,
                     "created_unix": time.time(), "jobs": {j["id"]: {"status": "PENDING"} for j in jobs},
                     "global_stop": None}
            atomic(state_path, state)
        lookup = {j["id"]: j for j in jobs}
        while True:
            state["heartbeat_unix"] = time.time()
            if state["heartbeat_unix"] - state["created_unix"] > cfg["budgets"]["max_campaign_wall_seconds"]:
                state["global_stop"] = "campaign wall budget reached"
            if state["global_stop"]:
                atomic(state_path, state)
                return 87
            changed = False
            for name, item in state["jobs"].items():
                if item["status"] != "RUNNING":
                    continue
                try:
                    status = remote_status(item["gpu_job_id"])
                    if status == "running":
                        continue
                    if status != "ok":
                        item.update(status="FAILED", reason="gpu state " + status)
                    else:
                        item.update(status="COMPLETED", certification=certify(lookup[name], item["gpu_job_id"]))
                    changed = True
                except Exception as exc:
                    state["global_stop"] = "monitor/certification failure: " + str(exc)
                    break
            changed |= quarantine_failed_dependencies(state, jobs)
            if len(jobs) == 12 and any(state["jobs"][j["id"]]["status"] in {"FAILED", "QUARANTINED"}
                                       for j in jobs if j["seed"] == 0):
                for j in jobs:
                    if j["seed"] > 0 and state["jobs"][j["id"]]["status"] == "PENDING":
                        state["jobs"][j["id"]].update(status="QUARANTINED", reason="K2 seed0 integrity gate failed")
            if not state["global_stop"]:
                active = {int(lookup[name]["gpu_index"]) for name, item in state["jobs"].items()
                          if item["status"] == "RUNNING"}
                for job in jobs:
                    item = state["jobs"][job["id"]]
                    index = int(job["gpu_index"])
                    if item["status"] != "PENDING" or index in active or len(active) >= (4 if len(jobs) == 12 else 5):
                        continue
                    if len(jobs) == 12 and job["seed"] > 0 and any(
                            state["jobs"][j["id"]]["status"] != "COMPLETED" for j in jobs if j["seed"] == 0):
                        continue
                    if any(state["jobs"][dep]["status"] != "COMPLETED" for dep in job["depends_on"]):
                        continue
                    try:
                        # Recheck immutable files and occupancy immediately before every submission.
                        _, _, job_grant, _ = runtime.load_job(manifest_path, launch_grant_path, job["id"])
                        remaining = (state["created_unix"] + cfg["budgets"]["max_campaign_wall_seconds"]
                                     - time.time())
                        if remaining < job_grant["max_wall_seconds"] + 10:
                            state["global_stop"] = "insufficient campaign wall budget for next job"
                            break
                        available = gpu_status()
                        if not idle_gpu(job, available):
                            continue
                        item.update(status="CLAIMED", claimed_unix=time.time())
                        atomic(state_path, state)
                        gpu_job_id = launch(job, manifest_path, launch_grant_path, available)
                        item.update(status="RUNNING", gpu_job_id=gpu_job_id, launched_unix=time.time())
                        active.add(index)
                        changed = True
                    except Exception as exc:
                        item.update(status="FAILED", reason="launch failure, attempt consumed: " + str(exc))
                        state["global_stop"] = "launch failure: " + job["id"]
                        changed = True
                        break
            atomic(state_path, state)
            if all(item["status"] in TERMINAL for item in state["jobs"].values()):
                return 0 if all(item["status"] == "COMPLETED" for item in state["jobs"].values()) else 87
            time.sleep(poll_seconds)


def detach(manifest_path, launch_grant_path, state_path, poll_seconds):
    """Return after one launch is recorded; the queue continues in its own session."""
    runtime.require(not state_path.exists(), "detached campaign already has state")
    validate(manifest_path, launch_grant_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = state_path.with_suffix(".log")
    with log_path.open("x") as log:
        child = subprocess.Popen([sys.executable, "-m", "experiments.ws_campaign.queue",
                                  "--manifest", str(manifest_path), "--launch-grant", str(launch_grant_path),
                                  "--state", str(state_path), "--poll-seconds", str(poll_seconds)],
                                 cwd=runtime.ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    until = time.monotonic() + 120
    while time.monotonic() < until:
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            runtime.require(not state["global_stop"], "detached launch stopped: " + str(state["global_stop"]))
            if any(item["status"] in {"RUNNING", "COMPLETED"} for item in state["jobs"].values()):
                print(json.dumps({"queue_pid": child.pid, "state": str(state_path), "log": str(log_path),
                                  "first_launch_recorded": True, "heartbeat_unix": state.get("heartbeat_unix")}))
                return 0
        runtime.require(child.poll() is None, "detached queue exited before first launch")
        time.sleep(0.5)
    runtime.require(False, "detached queue did not record first launch within 120 seconds")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--launch-grant", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--detach", action="store_true")
    args = p.parse_args()
    try:
        runtime.require(args.poll_seconds >= 10, "poll cadence")
        paths = (args.manifest.resolve(), args.launch_grant.resolve(), args.state.resolve())
        return detach(*paths, args.poll_seconds) if args.detach else run_queue(*paths, args.poll_seconds)
    except (runtime.Stop, OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 87


if __name__ == "__main__":
    sys.exit(main())
