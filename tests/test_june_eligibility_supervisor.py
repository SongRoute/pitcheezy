"""Synthetic ledger/timeout tests; never invokes the June worker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "supervise_june_eligibility.py"
SPEC = importlib.util.spec_from_file_location("supervise_june_eligibility", SCRIPT)
assert SPEC and SPEC.loader
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def fixture_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path, Path]:
    coord = tmp_path / "coord"
    coord.mkdir()
    monkeypatch.setattr(supervisor, "COORD", coord)
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "manifest.json"
    plan = {
        "argv": ["python", "worker.py", "--config", "config.json", "--output", str(output)],
        "output_dir": str(output), "completion_path": str(manifest),
        "completion_checks": {"status": "complete", "config_sha256": "b" * 64,
                              "registered_code_commit": "e" * 40, "execution_head": "e" * 40},
        "registration_sha256": "a" * 64,
        "config_sha256": "b" * 64, "source_hashes": {"worker.py": "c" * 64},
        "supervisor_sha256": "d" * 64, "code_commit": "e" * 40,
        "code_root": str(tmp_path), "config_path": "config.json",
        "python_executable": "python", "cap_seconds": 600,
        "worker_heavy_lock": str(tmp_path / ".heavy.lock"),
    }
    plan_path = coord / "execution-plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(supervisor, "validate", lambda *_: None)
    return plan, plan_path, manifest


def test_success_binds_completion_and_refuses_second_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, plan_path, manifest = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)
    manifest.write_text(json.dumps(plan["completion_checks"]))
    launches = []

    class Child:
        pid = 2_000_000_000

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: launches.append((a, kw)) or Child())
    end = supervisor.run(plan_path)
    assert end["outcome"] == "completed", end["note"]
    assert end["completion_sha256"] == supervisor.sha(manifest)
    assert end["registration_sha256"] == plan["registration_sha256"]
    assert len(launches) == 1
    with pytest.raises(ValueError, match="no automatic retry"):
        supervisor.run(plan_path)
    assert len(launches) == 1


def test_preflight_refusal_is_durable_without_worker_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _plan, plan_path, _manifest = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "preflight", lambda *_: (_ for _ in ()).throw(ValueError("hash changed")))
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_a, **_k: pytest.fail("worker launched"))
    with pytest.raises(ValueError, match="hash changed"):
        supervisor.run(plan_path)
    assert supervisor.read(supervisor.COORD / "ledger/refusal.json")["status"] == "refused_preflight"
    assert not (supervisor.COORD / "ledger/starts/june-prepare.json").exists()
    with pytest.raises(ValueError, match="no automatic retry"):
        supervisor.run(plan_path)


def test_timeout_is_charged_and_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _plan, plan_path, _manifest = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)

    class Child:
        pid = 2_000_000_000

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_a, **_k: Child())
    def repeated_signal_cleanup(_process):
        import os
        import signal
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        return -15

    monkeypatch.setattr(supervisor, "terminate", repeated_signal_cleanup)
    end = supervisor.run(plan_path)
    assert end["outcome"] == "timeout", end["note"]
    assert end["popen_wait_seconds"] >= 0
    assert end["completion_sha256"] is None
    assert (supervisor.COORD / "ledger/starts/june-prepare.json").exists()


def test_postflight_failure_preserves_worker_wall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, plan_path, _manifest = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)

    class Child:
        pid = 2_000_000_000

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_a, **_k: Child())
    monkeypatch.setattr(supervisor, "completion_sha", lambda *_: (_ for _ in ()).throw(ValueError("bad manifest")))
    ticks = iter([0.0, 2.0, 2.0, 20.0])
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(ticks))
    end = supervisor.run(plan_path)
    assert end["outcome"] == "failed"
    assert end["popen_wait_seconds"] == 2.0
    assert end["postflight_seconds"] == 18.0
    assert "bad manifest" in end["note"]


def test_registration_mismatch_and_missing_completion_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, _plan_path, _manifest = fixture_plan(tmp_path, monkeypatch)
    registration = {key: plan[key] for key in supervisor.COMMON_FIELDS}
    registration.update({"protocol": supervisor.REGISTRATION_PROTOCOL,
                         "registration_commit_d": "f" * 40, "created_utc": "2026-09-27T00:00:00Z"})
    registration_path = supervisor.COORD / "registration.json"
    registration_path.write_text(json.dumps(registration))
    plan["registration_path"] = str(registration_path)
    supervisor.validate_registration(plan)
    registration["source_hashes"] = {"worker.py": "0" * 64}
    registration_path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="registration/plan mismatch: source_hashes"):
        supervisor.validate_registration(plan)
    checks = plan["completion_checks"].copy()
    del checks["execution_head"]
    plan["completion_checks"] = checks
    with pytest.raises(ValueError, match="completion execution_head identity"):
        supervisor.validate_completion_checks(plan)
