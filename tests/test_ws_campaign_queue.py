"""Synthetic metadata checks only; no GPU, server, or real inputs."""
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
