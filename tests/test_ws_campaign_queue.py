"""Synthetic metadata checks only; no GPU, server, or real inputs."""
import json
from pathlib import Path
import tempfile

import pytest

from experiments.ws_campaign import queue, runtime


def inventory():
    jobs = []
    for kind, count in (("supervised", 78), ("pretrain", 6), ("teacher", 3)):
        for i in range(count):
            name = f"{kind}-{i}"
            jobs.append({"id": name, "kind": kind, "depends_on": [],
                         "calibrations": 3 if kind == "supervised" else 0,
                         "attempt_id": name, "output": "/out/" + name,
                         "gpu_uuid": "GPU-fixture", "common_lock": "/locks/gpu"})
    # One registered leaf7 seed uses one CAL instead of three; the fixed total is 228.
    for job in jobs[:3]:
        job["calibrations"] = 1
    return jobs


def test_inventory_is_finite_and_dependency_acyclic():
    jobs = inventory()
    queue.validate_inventory(jobs)
    jobs[1]["depends_on"] = [jobs[0]["id"]]
    queue.validate_inventory(jobs)
    jobs[0]["depends_on"] = [jobs[1]["id"]]
    with pytest.raises(runtime.Stop, match="acyclic"):
        queue.validate_inventory(jobs)


def test_duplicate_attempt_and_excess_calibration_stop():
    jobs = inventory()
    jobs[1]["attempt_id"] = jobs[0]["attempt_id"]
    with pytest.raises(runtime.Stop, match="unique attempts"):
        queue.validate_inventory(jobs)
    jobs[1]["attempt_id"] = jobs[1]["id"]
    jobs[3]["calibrations"] += 1
    with pytest.raises(runtime.Stop, match="CAL total"):
        queue.validate_inventory(jobs)


def test_gpu_st_real_spacing_fixture(monkeypatch):
    monkeypatch.setattr(queue, "command", lambda *_: "== GPU ==\nGPU0       3/ 24564 MiB  util   0%\nGPU1    1024/ 24564 MiB  util  10%\n")
    assert queue.gpu_status()[1] == {"used_mib": 1024, "total_mib": 24564, "util": 10}


def test_registered_87_slot_manifest_and_grants_bind_without_payload_access():
    config = runtime.ROOT / "configs/EXP-WS-CAMPAIGN-001.json"
    cfg = json.loads(config.read_text())
    with tempfile.TemporaryDirectory(dir=runtime.ROOT, prefix="campaign-grant-test-") as tmp:
        root = Path(tmp)
        jobs = []
        for i, scientific in enumerate(cfg["jobs"]):
            job = dict(scientific, attempt_id=f"fixture-{i:02d}", output=f"/tmp/campaign-fixture/{i:02d}",
                       inputs="/tmp/campaign-fixture/inputs", input_manifest="/tmp/campaign-fixture/inputs/manifest.json",
                       input_manifest_sha256="a" * 64, exposure_ledger="/tmp/campaign-fixture/exposure.json",
                       exposure_ledger_sha256="b" * 64, commit="c" * 40,
                       gpu_index=i % 5, gpu_uuid=f"GPU-fixture-{i % 5}",
                       common_lock=f"/tmp/campaign-fixture/lock-{i % 5}")
            grant = {"execute": True, "verdict": "PASS", "distinct_from_author": True,
                     "author": "author", "reviewer": "reviewer", "job_id": job["id"],
                     "campaign_id": cfg["id"], "data_role": cfg["data_role"],
                     "data_version": cfg["data_version"], "window": job["window"],
                     "config_sha256": runtime.sha(config), "sources_sha256": runtime.source_hashes(),
                     "attempt_id": job["attempt_id"], "output": job["output"],
                     "gpu_index": job["gpu_index"], "gpu_uuid": job["gpu_uuid"],
                     "common_lock": job["common_lock"], "input_manifest_sha256": job["input_manifest_sha256"],
                     "exposure_ledger_sha256": job["exposure_ledger_sha256"], "commit": job["commit"],
                     "max_wall_seconds": cfg["budgets"]["teacher_seconds"] if job["kind"] == "teacher" else cfg["budgets"]["per_job_seconds"],
                     "max_rss_bytes": cfg["budgets"]["host_rss_bytes"],
                     "min_free_host_bytes": cfg["budgets"]["host_rss_bytes"] * 5 + (8 << 30)}
            path = root / f"grant-{i:02d}.json"
            path.write_text(json.dumps(grant))
            job["grant"] = str(path.relative_to(runtime.ROOT))
            job["grant_sha256"] = runtime.sha(path)
            jobs.append(job)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"campaign_id": cfg["id"], "config": str(config.relative_to(runtime.ROOT)),
                                        "config_sha256": runtime.sha(config), "jobs": jobs}))
        launch = root / "launch.json"
        launch.write_text(json.dumps({"execute": True, "verdict": "PASS", "distinct_from_author": True,
                                      "author": "author", "reviewer": "reviewer",
                                      "manifest_sha256": runtime.sha(manifest)}))
        assert len(queue.validate(manifest, launch)) == 87


def test_failed_arm_quarantines_descendants_without_stopping_unrelated_work():
    jobs = [{"id": "seed0", "depends_on": []},
            {"id": "seed1", "depends_on": ["seed0"]},
            {"id": "teacher", "depends_on": ["seed1"]},
            {"id": "other", "depends_on": []}]
    state = {"jobs": {"seed0": {"status": "FAILED"}, "seed1": {"status": "PENDING"},
                      "teacher": {"status": "PENDING"}, "other": {"status": "PENDING"}}}
    assert queue.quarantine_failed_dependencies(state, jobs)
    assert [state["jobs"][key]["status"] for key in ("seed1", "teacher", "other")] == [
        "QUARANTINED", "QUARANTINED", "PENDING"]


def test_unresolved_claim_cannot_relaunch_on_restart():
    state = {"manifest_sha256": "a", "launch_grant_sha256": "b",
             "jobs": {"one": {"status": "CLAIMED"}}}
    with pytest.raises(runtime.Stop, match="unresolved launch claim"):
        queue.assert_resumable(state, "a", "b")
    state["jobs"]["one"]["status"] = "RUNNING"
    queue.assert_resumable(state, "a", "b")
