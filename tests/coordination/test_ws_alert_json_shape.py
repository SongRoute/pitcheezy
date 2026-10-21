"""Opt-in regression: non-object status JSON must not suppress unrelated alerts.

Run with --source-root CHECKOUT --sha256 REVIEWED_WS_ALERT_SHA --out NEW_DIRECTORY.
Only invented files are read. Actual collect/evaluate/run_once run in dry-run mode;
network, process creation, journal writes and notifier sends are guarded. No default
config, real publisher output, operations status, profile log or notifier is used.
"""
import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.request

SOURCE = SOURCE_SHA = OUTPUT = None
NOW = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()


class ForbiddenIO(BaseException):
    pass


class AlertJsonShapeTests(unittest.TestCase):
    def setUp(self):
        if SOURCE is None:
            self.skipTest("explicit --source-root, --sha256 and --out required")
        raw = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_SHA)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="cx47-synthetic-"))).resolve()
        self.blocked, self.observed = [], {}
        self.addCleanup(self.save_evidence)

        def forbid(*_args, **_kwargs):
            self.blocked.append("forbidden network/process/notifier/journal boundary")
            raise ForbiddenIO("real I/O forbidden")

        for obj, names in ((socket, ("socket", "create_connection", "getaddrinfo")),
                           (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                           (os, ("system", "popen", "posix_spawn", "posix_spawnp")),
                           (urllib.request, ("urlopen",))):
            for name in names:
                if hasattr(obj, name):
                    self.stack.enter_context(patch.object(obj, name, forbid))
        self.stack.enter_context(patch.object(os, "environ", {}))
        self.w = types.ModuleType("ws_alert_candidate")
        self.w.__file__ = str(SOURCE)
        exec(compile(raw, str(SOURCE), "exec"), self.w.__dict__)
        for name in ("MacNotifier", "NtfyNotifier", "IMessageNotifier", "FakeNotifier"):
            self.stack.enter_context(patch.object(getattr(self.w, name), "send", forbid))
        self.stack.enter_context(patch.object(self.w.JsonlNotifier, "log", forbid))

    def tearDown(self):
        self.assertEqual(self.blocked, [])

    def save_evidence(self):
        result = {"source_sha256": SOURCE_SHA, "forbidden_boundary_attempts": self.blocked,
                  "observed": self.observed}
        (OUTPUT / (self._testMethodName + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def check_role(self, role):
        out = self.root / "publisher"
        version = out / "v1"
        (version / "games").mkdir(parents=True)
        paths = {"health": version / "health.json", "games": version / "games" / "2030-01-01.json",
                 "ops": self.root / "ops.json", "state": self.root / "publisher.state.json"}
        # No active game: the unrelated disk warning is ungated and needs no hold.
        values = {"health": {"published_at": "2030-01-01T00:00:00.000Z", "game_checks": {}},
                  "games": {"games": []}, "ops": {}, "state": {}}
        values[role] = [1]
        for key, value in values.items():
            paths[key].write_text(json.dumps(value))
        cfg = deepcopy(self.w.BUILTIN)
        cfg.update(cli_out=True, schedule=[])
        cfg["paths"] = {"out_dir": str(out), "state_path": str(paths["state"]), "ops_status": str(paths["ops"]),
                        "profile_log": None, "alert_dir": str(self.root / "alerts"), "disk_paths": []}
        captured = io.StringIO()
        self.observed = {"role": role, "input": [1], "now": NOW, "dry_run": True}

        def collect(config, now):
            obs = self.w.collect(config, now, memory=lambda: None, disk=lambda _paths: {"synthetic-disk": 0})
            self.observed["input_errors"] = {key: obs.get(key) for key in ("health_error", "state_error", "ops_error")}
            return obs

        try:
            with redirect_stdout(captured):
                self.w.run_once(cfg, now=NOW, dry_run=True, collect_fn=collect)
        except Exception as error:
            self.observed["error"] = {"type": type(error).__name__, "message": str(error)}
            self.observed["output"] = captured.getvalue()
            raise
        result = json.loads(captured.getvalue())
        self.observed["result"] = result
        self.assertIn("disk_low", [item["key"] for item in result["active"]],
                      "a malformed input must not suppress the independent disk warning")
        self.assertFalse((self.root / "alerts").exists(), "dry-run must not write watcher state/journal")

    def test_health_list_preserves_disk_warning(self):
        self.check_role("health")

    def test_games_list_preserves_disk_warning(self):
        self.check_role("games")

    def test_ops_list_preserves_disk_warning(self):
        self.check_role("ops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.source_root.resolve() / "apps/observer/ops/ws_alert.py"
    SOURCE_SHA, OUTPUT = args.sha256, args.out.resolve()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        parser.error("source SHA mismatch")
    OUTPUT.mkdir(parents=False, exist_ok=False)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
