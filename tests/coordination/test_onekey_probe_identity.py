"""Opt-in, pinned-source probe identity regressions; synthetic inputs and blocked I/O.

Run directly with --source PATH --sha256 SHA --out NEW_DIRECTORY. Ordinary discovery
skips. Existing acceptance setup supplies the source pin, fake clock and I/O guards;
none of its test methods run. The original runner/gateway are never executed.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import check_onekey_receipt as checker
import test_onekey_acceptance as acceptance

SOURCE = SOURCE_SHA = OUTPUT = None
TOKEN = "cx04-fake-bearer-only-0123456789abcdefghijklm"


class ProbeIdentityTests(unittest.TestCase):
    iso = staticmethod(acceptance.Acceptance.iso)
    advance = acceptance.Acceptance.advance

    def setUp(self):
        if SOURCE is None:
            self.skipTest("explicit --source, --sha256 and --out required")
        acceptance.SOURCE, acceptance.SOURCE_SHA = SOURCE, SOURCE_SHA
        acceptance.Acceptance.setUp(self)
        self.heads, self.children, self.observations = [], [], []
        self.revoked = False
        self.start = self.clock[0]
        self.out = self.tmp / "out"
        # Registered before test assertions so failing regressions preserve evidence.
        self.addCleanup(self.save_evidence)

    def tearDown(self):
        acceptance.Acceptance.tearDown(self)

    def save_evidence(self):
        destination = OUTPUT / self._testMethodName
        destination.mkdir()
        for name in ("plan.json", "grant.json", "wrapper-log.json", "PUBLIC-SUMMARY.md"):
            path = self.out / name
            if path.exists():
                (destination / name).write_bytes(path.read_bytes())
        result = {"source_sha256": SOURCE_SHA, "head_stubs": self.heads,
                  "runner_stubs": len(self.children), "observations": self.observations,
                  "forbidden_boundary_attempts": self.blocked,
                  "boundary": "Synthetic client/build/runner and HEAD stubs; actual validate_pre, "
                              "cmd_run, cmd_probe, assess, finish and checker. No gateway execution."}
        (destination / "observations.json").write_text(json.dumps(result, indent=2) + "\n")

    def preconditions(self):
        at = self.iso(self.start)
        return {
            "schema": "pitcheezy-onekey-preconditions-v1", "run_id": "cx46-synthetic", "game_key": "123456",
            "registration_sha256": hashlib.sha256(self.w.REGISTRATION.read_bytes()).hexdigest(),
            "approvals": {"song_go": {"reference": "synthetic-only", "at_utc": at},
                          "independent_review": {"reference": "synthetic-only", "at_utc": at,
                                                 "reviewer": "synthetic-reviewer"}},
            "clock": {"ntp_offset_s": 0, "measured_at_utc": at},
            "teammate_confirmation": {
                "at_utc": at, "reference_private": "synthetic-only", "deployed_source_commit": "0" * 40,
                "deployed_runtime_sha256": {n: {"raw": raw, "lf": lf}
                                            for n, (raw, lf) in self.w.EXPECTED_RUNTIME.items()},
                "scope_mode": "scoped", "scope_feature_deployed": True, "expiry_feature_deployed": True,
                "publish_key_scope": "v1/game/123456.json", "publish_expires_at": self.iso(self.start + 1200),
                "upload_enabled_for_window": True, "data_read_enabled": True,
                "notice_switch_off": True, "limiter_allows_18_in_120s": True},
            "game_key_fresh_attested_by_teammate": True,
            "test_window": {"start_utc": self.iso(self.start - 60), "end_utc": self.iso(self.start + 1800)},
            "credentials": {"bearer_dedicated_new": True, "bearer_delivered_privately": True,
                            "viewer_session_dedicated": True, "viewer_session_delivered_privately": True,
                            "stored_on_disk": False},
            "single_writer": {"teammate_confirms_no_other_writer_on_key": True},
            "operators": {"ours_alias": "synthetic-ours", "theirs_alias": "synthetic-theirs"},
            "abort_contact": {"private_channel_agreed": True, "stop_signal_agreed": True,
                              "teammate_can_toggle_upload_off": True},
            "post_test": {"teammate_will_toggle_upload_off": True, "teammate_will_replace_token_digest": True,
                          "teammate_will_clear_scope": True},
            "target": {"upload_base": "https://synthetic.invalid/_publish", "public_origin": "https://synthetic.invalid"},
            "client": {"commit": "0" * 40, "worktree": str(self.tmp), "python": "never-executed-python"},
            "probes": {"out_of_scope_head": True, "expired_head": True, "revoked_head": True},
        }

    def head(self, _pre, relative, _token, _cafile):
        # Pinned partner gateway 48ec4065, ops/public-host/gateway.mjs:97-100:
        # scope mismatch returns 403 before expiry is evaluated. Auth precedes scope.
        status = 401 if self.revoked else 403 if relative != "v1/game/123456.json" or self.clock[0] >= self.start + 1200 else 200
        self.heads.append({"relative": relative, "elapsed_s": self.clock[0] - self.start, "status": status})
        return status, {"content_type": "application/json", "cf_mitigated_present": False, "cf_ray_present": False}

    def child(self, argv, **_kwargs):
        self.children.append("synthetic-runner")
        destination = Path(argv[argv.index("--output") + 1])
        self.assertEqual(destination, self.out / "runner")
        destination.mkdir()
        receipt = {"result": acceptance.PASS, "elapsed_s": 0.1, "remote_cleanup": "confirmed_tombstone_retained",
                   "owned_worker_stopped": True, "pending_paths": 0,
                   "requests": [{"method": item.split(":")[0],
                                 "result": "upload_conflict" if item.endswith(":upload_conflict") else "confirmed"}
                                for item in checker.METHODS]}
        (destination / "receipt.json").write_text(json.dumps(receipt))
        return types.SimpleNamespace(returncode=0, communicate=lambda **_kw: (json.dumps({"result": acceptance.PASS}), ""))

    def observe(self, stage, rc, refusal=None):
        log = json.loads((self.out / "wrapper-log.json").read_text())
        result = checker.check(log, "required", "required", "REAL_HOST")
        self.observations.append({"stage": stage, "rc": rc, "refusal": refusal, "completion": log["completion"],
                                  "missing_evidence": log["missing_evidence"], "failed_evidence": log["failed_evidence"],
                                  "checker": result})
        return log, result

    def main_run(self):
        pre = self.preconditions()
        self.assertEqual(self.w.validate_pre(pre, False), [], "actual real-mode precondition validation")
        plan = dict(pre["target"], game_key=pre["game_key"])
        grant = {"plan_sha256": self.w.sha256(self.w.canonical(plan)), "expires_epoch": self.start + 1800}
        credentials = iter((TOKEN, "session=cx46-fake-cookie-only"))
        with ExitStack() as stack:
            for obj, name, value in (
                (self.w, "check_client", lambda _pre: {"commit": "0" * 40}),
                (self.w, "build_plan_grant", lambda *_args: (plan, grant)),
                (self.w, "head", self.head), (subprocess, "Popen", self.child),
                (getpass, "getpass", lambda _label: next(credentials)),
                (sys, "stdin", acceptance.TTY("SKIP\n")),
                (sys, "stdout", acceptance.TTY()), (sys, "stderr", acceptance.TTY()),
            ):
                stack.enter_context(patch.object(obj, name, value))
            rc = self.w.cmd_run(types.SimpleNamespace(out=str(self.out), ledger=str(self.tmp / "ledger.json"),
                                                      dry_run=False), pre, False, None)
        log, _check = self.observe("main", rc)
        self.assertEqual((rc, log["missing_evidence"]), (4, ["expired", "revoked"]))
        return deepcopy(pre)

    def deferred(self, pre, kind):
        self.assertEqual(self.w.validate_pre(pre, False), [], "actual validation of changed input")
        refusal = None
        with ExitStack() as stack:
            for obj, name, value in (
                (self.w, "head", self.head), (getpass, "getpass", lambda _label: TOKEN),
                (sys, "stdin", acceptance.TTY()), (sys, "stdout", acceptance.TTY()), (sys, "stderr", acceptance.TTY()),
            ):
                stack.enter_context(patch.object(obj, name, value))
            try:
                rc = self.w.cmd_probe(types.SimpleNamespace(out=str(self.out), kind=kind), pre, False, None)
            except self.w.Refuse as error:
                rc, refusal = 2, str(error)
        log, checked = self.observe(kind, rc, refusal)
        return refusal, rc, log, checked

    def test_control_completes_with_original_identity(self):
        pre = self.main_run()
        self.clock[0] = self.start + 1210
        self.assertIsNone(self.deferred(pre, "expired")[0])
        self.revoked = True
        refusal, rc, log, checked = self.deferred(pre, "revoked")
        self.assertEqual((refusal, rc, log["completion"]), (None, 0, "COMPLETE"))
        self.assertEqual(checked["status"], "LOCAL_EVIDENCE_CONSISTENT")
        self.assertIn("SAME_TARGET_BINDING", checked["unverified"])
        self.assertEqual((len(self.heads), len(self.children)), (3, 1))

    def test_changed_game_key_scope_refused_before_request(self):
        pre = self.main_run()
        pre["game_key"] = "123457"
        pre["teammate_confirmation"]["publish_key_scope"] = "v1/game/123457.json"
        self.clock[0] = self.start + 1210
        before = len(self.heads)
        refusal, _rc, _log, _checked = self.deferred(pre, "expired")
        if refusal is None:
            # Retain the unrepaired source's entire verdict, not just expired.pass.
            self.revoked = True
            self.deferred(pre, "revoked")
        self.assertEqual(len(self.heads) - before, 0, "changed key/scope must refuse before any deferred HEAD")
        self.assertIsNotNone(refusal)

    def test_changed_expiry_refused_before_request(self):
        pre = self.main_run()
        pre["teammate_confirmation"]["publish_expires_at"] = self.iso(self.start + 600)
        self.clock[0] = self.start + 610
        before = len(self.heads)
        refusal, _rc, _log, _checked = self.deferred(pre, "expired")
        self.assertEqual(len(self.heads) - before, 0, "changed expiry must refuse before consuming a post slot")
        self.assertIsNotNone(refusal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    SOURCE, SOURCE_SHA, OUTPUT = args.source.resolve(), args.sha256, args.out.resolve()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        parser.error("source SHA mismatch")
    OUTPUT.mkdir(parents=False, exist_ok=False)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
