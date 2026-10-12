"""One GPU job: fail-closed grant checks, owned-child supervision, no scoring."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.ws_seq.train_tm002 import supervise_command

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "experiments/ws_campaign/runtime.py",
    "experiments/ws_campaign/queue.py",
    "experiments/ws_campaign/train.py",
    "experiments/ws_campaign/models.py",
    "experiments/ws_campaign/data.py",
    "configs/EXP-WS-CAMPAIGN-001.json",
    "experiments/ws_seq/train_seq.py",
    "experiments/ws_seq/tm002.py",
    "experiments/ws_seq/train_tm002.py",
)


class Stop(RuntimeError):
    pass


def require(ok, reason):
    if not ok:
        raise Stop("STOP " + reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read_bound(path, digest):
    require(Path(path).is_file() and sha(path) == digest, "bound file missing or changed: " + str(path))
    return json.loads(Path(path).read_text())


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def load_job(manifest_path, root_grant_path, job_id):
    root_grant = json.loads(root_grant_path.read_text())
    require(root_grant.get("execute") is True and root_grant.get("verdict") == "PASS"
            and root_grant.get("distinct_from_author") is True
            and root_grant.get("author") and root_grant.get("reviewer")
            and root_grant["author"] != root_grant["reviewer"], "campaign launch grant")
    require(root_grant.get("manifest_sha256") == sha(manifest_path), "campaign manifest hash")
    manifest = json.loads(manifest_path.read_text())
    cfg_path = ROOT / manifest["config"]
    require(cfg_path.resolve().is_relative_to(ROOT) and cfg_path.is_file(), "config path")
    cfg = read_bound(cfg_path, manifest["config_sha256"])
    require(manifest.get("campaign_id") == cfg.get("id"), "campaign ID")
    jobs = manifest["jobs"]
    require(len(jobs) == 87 and len({j["id"] for j in jobs}) == 87, "exact 87 unique jobs")
    registered = {j["id"]: j for j in cfg["jobs"]}
    require(len(registered) == 87 and set(registered) == {j["id"] for j in jobs}, "registered job inventory")
    job = next((j for j in jobs if j["id"] == job_id), None)
    require(job is not None, "unknown job")
    require(all(job.get(key) == registered[job_id].get(key)
                for key in ("id", "kind", "block", "arm", "seed", "window", "depends_on", "calibrations")),
            "scientific job differs from registration")
    grant_path = ROOT / job["grant"]
    require(grant_path.resolve().is_relative_to(ROOT), "grant path escapes snapshot")
    grant = read_bound(grant_path, job["grant_sha256"])
    require(grant.get("execute") is True and grant.get("verdict") == "PASS"
            and grant.get("distinct_from_author") is True and grant.get("author")
            and grant.get("reviewer") and grant["author"] != grant["reviewer"], "job independent review")
    require(grant.get("job_id") == job_id and grant.get("campaign_id") == cfg["id"]
            and grant.get("config_sha256") == manifest["config_sha256"]
            and grant.get("sources_sha256") == source_hashes()
            and grant.get("attempt_id") == job.get("attempt_id")
            and grant.get("output") == job.get("output")
            and grant.get("gpu_index") == job.get("gpu_index")
            and grant.get("gpu_uuid") == job.get("gpu_uuid")
            and grant.get("common_lock") == job.get("common_lock")
            and grant.get("input_manifest_sha256") == job.get("input_manifest_sha256")
            and grant.get("exposure_ledger_sha256") == job.get("exposure_ledger_sha256")
            and grant.get("commit") == job.get("commit"), "job grant binding")
    budgets = cfg["budgets"]
    require(0 < grant.get("max_wall_seconds", 0) <= budgets["per_job_seconds"]
            and 0 < grant.get("max_rss_bytes", 0) <= budgets["host_rss_bytes"]
            and budgets["gpu_reserved_bytes"] > 0, "registered resource limits")
    require(Path(job["output"]).is_absolute() and Path(job["inputs"]).is_absolute()
            and Path(job["input_manifest"]).is_absolute()
            and Path(job["exposure_ledger"]).is_absolute()
            and Path(job["common_lock"]).is_absolute(), "absolute server paths")
    return cfg, job, grant, grant_path


def worker(args, cfg, job, grant, grant_path):
    require(sys.platform == "linux" and os.environ.get("GPU_JOB_ID") and os.environ.get("GPU_JOB_DIR"),
            "gpu run Linux worker only")
    env = Path(os.environ["GPU_JOB_DIR"]) / "job.env"
    values = dict(line.strip().split("=", 1) for line in env.read_text().splitlines() if "=" in line)
    # Grants are issued after the reviewed source commit, so gpu snapshots them as
    # untracked bytes. Their exact hashes and every executable source are checked.
    require(grant["commit"].startswith(values.get("COMMIT", "missing:"))
            and len(grant["commit"]) == 40, "source commit")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(grant["gpu_index"]), "selected GPU index")
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(grant["gpu_index"]),
                                    "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).strip()
    require(uuid == grant["gpu_uuid"], "physical GPU UUID")
    require(sha(job["input_manifest"]) == grant["input_manifest_sha256"], "input manifest")
    require(sha(job["exposure_ledger"]) == grant["exposure_ledger_sha256"], "exposure ledger")
    lock_path = Path(grant["common_lock"])
    require(lock_path.is_file(), "existing common GPU lock")
    out = Path(job["output"])
    require(not out.exists(), "output already exists")
    dependencies = {}
    for dep in job["depends_on"]:
        binding = grant.get("dependencies", {}).get(dep)
        require(binding and Path(binding["directory"]).is_absolute(), "dependency binding " + dep)
        receipt_path = Path(binding["directory"]) / "receipt.json"
        supervisor_path = Path(binding["directory"]) / "SUPERVISOR.json"
        require(receipt_path.is_file() and supervisor_path.is_file(), "dependency files " + dep)
        receipt = json.loads(receipt_path.read_text())
        supervisor = json.loads(supervisor_path.read_text())
        require(receipt.get("status") == "COMPLETED" and supervisor.get("status") == "COMPLETED"
                and supervisor.get("receipt_sha256") == sha(receipt_path)
                and receipt.get("job_id") == dep and receipt.get("attempt_id") == binding.get("attempt_id")
                and receipt.get("grant_sha256") == binding.get("grant_sha256")
                and receipt.get("eval_scores_opened") is False, "dependency completion " + dep)
        dependencies[dep] = binding["directory"]
    with lock_path.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not out.exists(), "output claimed while waiting for lock")
        out.mkdir(parents=True, exist_ok=False)
        from experiments.ws_campaign import train
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one CUDA device")
        actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
        require(actual and actual.removeprefix("GPU-") == uuid.removeprefix("GPU-"), "torch device UUID")
        cap = cfg["budgets"]["gpu_reserved_bytes"]
        total = torch.cuda.get_device_properties(0).total_memory
        require(cap < total, "GPU allocator cap lacks headroom")
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        metadata = {"job_id": job["id"], "attempt_id": job["attempt_id"], "commit": grant["commit"],
                    "grant_sha256": sha(grant_path), "input_manifest_sha256": grant["input_manifest_sha256"],
                    "exposure_ledger_sha256": grant["exposure_ledger_sha256"], "gpu_uuid": uuid,
                    "dependencies": dependencies, "eval_scores_opened": False}
        receipt = train.run_job(job, cfg, Path(job["inputs"]), out, metadata)
        require(isinstance(receipt, dict) and receipt.get("status") == "COMPLETED"
                and receipt.get("eval_scores_opened") is False, "worker completion contract")
        receipt.update(metadata)
        tmp = out / "receipt.json.tmp"
        tmp.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, out / "receipt.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--launch-grant", type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    try:
        cfg, job, grant, grant_path = load_job(args.manifest, args.launch_grant, args.job)
        if args.worker:
            worker(args, cfg, job, grant, grant_path)
            return 0
        require(sys.platform == "linux" and os.environ.get("GPU_JOB_ID"), "gpu run supervisor only")
        cmd = [sys.executable, "-m", "experiments.ws_campaign.runtime", "--manifest", str(args.manifest),
               "--launch-grant", str(args.launch_grant), "--job", args.job, "--worker"]
        return supervise_command(cmd, Path(job["output"]), grant["max_wall_seconds"],
                                 grant["max_rss_bytes"], sha(grant_path))
    except (Stop, OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 87


if __name__ == "__main__":
    sys.exit(main())
