"""Opt-in, source-pinned synthetic contract regressions; ordinary discovery skips.

Run directly with --source PATH --sha256 REVIEWED_SHA. No real runner or transport.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import getpass
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


SOURCE = None
SOURCE_SHA = None
PASS = "PASS_SYNTHETIC_HTTPS"


class ForbiddenIO(BaseException):
    """Bypasses the wrapper's broad Exception handler: never a fake refusal."""


class TTY(io.StringIO):
    def isatty(self):
        return True


@unittest.skipUnless(SOURCE is not None, "explicit --source and --sha256 required")
class Acceptance(unittest.TestCase):
    def setUp(self):
        raw = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_SHA,
                         "source drift: refuse execution; review and explicitly pin candidate")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.blocked = []

        def forbid(*args, **kwargs):
            self.blocked.append("network/process attempt")
            raise ForbiddenIO("real network/process execution forbidden")

        # All wrapper stdlib imports are already loaded before these guards.
        for obj, names in ((socket, ("socket", "create_connection", "getaddrinfo")),
                           (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                           (os, ("system", "popen", "posix_spawn", "posix_spawnp"))):
            for name in names:
                if hasattr(obj, name):
                    self.stack.enter_context(patch.object(obj, name, forbid))
        self.stack.enter_context(patch.object(os, "environ", {}))
        self.w = types.ModuleType("onekey_candidate")
        self.w.__file__ = str(SOURCE)
        exec(compile(raw, str(SOURCE), "exec"), self.w.__dict__)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="cx04-")))
        self.clock = [1893456000.0]  # synthetic 2030-01-01; never wall-clock waits
        self.stack.enter_context(patch.object(time, "time", lambda: self.clock[0]))
        self.stack.enter_context(patch.object(time, "sleep", self.advance))
        self.stack.enter_context(patch.object(self.w, "now_utc", lambda: self.iso(self.clock[0])))

    def tearDown(self):
        self.assertEqual(self.blocked, [], "a forbidden boundary was reached")

    @staticmethod
    def iso(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def advance(self, seconds):
        self.clock[0] += seconds

    def scenario(self, *, expired=True, expiry_after=600, answer="REVOKED", input_delay=0):
        """Only pin/build, credentials, clock, HEAD transport and child are fixtures.

        Actual cmd_run/check_window/read_secrets/probe/ledger/log/scan decide outcomes.
        The child fixture is a successful runner receipt, not runner verification.
        """
        now = self.clock[0]
        pre = {
            "run_id": "cx04-synthetic", "game_key": "123456",
            "client": {"python": "never-executed-python", "worktree": str(self.tmp)},
            "test_window": {"start_utc": self.iso(now - 60), "end_utc": self.iso(now + 1800)},
            "teammate_confirmation": {"scope_mode": "scoped", "publish_expires_at": self.iso(now + expiry_after)},
            "probes": {"out_of_scope_head": True, "expired_head": expired, "revoked_head": True},
        }
        plan = {"upload_base": "https://synthetic.invalid/_publish/",
                "public_origin": "https://synthetic.invalid"}
        grant = {"plan_sha256": "0" * 64, "expires_epoch": now + 1800}
        args = types.SimpleNamespace(out=str(self.tmp / "out"), ledger=str(self.tmp / "ledger.json"), dry_run=False)
        heads, children = [], []
        credentials = iter(("cx04-fake-bearer-only-0123456789abcdefghijklm", "session=cx04-fake-cookie-only"))

        def prompt(_label):
            self.advance(input_delay)
            return next(credentials)

        def head(_pre, relative, _token, _cafile):
            heads.append((relative, self.clock[0]))
            # Scope key is forbidden; original key expires, then operator revokes it.
            status = 403 if relative.endswith("123457.json") else (401 if operator.revoked else 403)
            return status, {"content_type": "application/json", "cf_mitigated_present": False,
                            "cf_ray_present": False}

        class Operator(TTY):
            revoked = False

            def readline(self, *a, **kw):
                self.revoked = answer == "REVOKED"
                return super().readline(*a, **kw)

        operator = Operator(answer + "\n")

        def child(argv, **kwargs):
            children.append(self.clock[0])
            dest = Path(argv[argv.index("--output") + 1])
            self.assertEqual(dest, self.tmp / "out" / "runner")
            dest.mkdir()
            methods = "HEAD HEAD PUT HEAD GET HEAD DELETE HEAD HEAD PUT HEAD GET DELETE HEAD HEAD DELETE HEAD".split()
            receipt = {"result": PASS, "elapsed_s": 0.1, "remote_cleanup": "confirmed_tombstone_retained",
                       "owned_worker_stopped": True, "pending_paths": 0,
                       "requests": [{"method": m, "result": "upload_conflict" if i == 12 else "confirmed"}
                                    for i, m in enumerate(methods)]}
            (dest / "receipt.json").write_text(json.dumps(receipt))
            return types.SimpleNamespace(returncode=0, communicate=lambda **kw: (json.dumps({"result": PASS}), ""))

        with ExitStack() as run:
            for obj, name, value in (
                (self.w, "check_client", lambda _pre: {"commit": "0" * 40}),
                (self.w, "build_plan_grant", lambda *_: (plan, grant)),
                (self.w, "head", head), (subprocess, "Popen", child),
                (getpass, "getpass", prompt), (sys, "stdin", operator),
                (sys, "stdout", TTY()), (sys, "stderr", TTY()),
            ):
                run.enter_context(patch.object(obj, name, value))
            try:
                rc = self.w.cmd_run(args, pre, False, None)
            except self.w.Refuse:
                rc = 2  # A pre-network window refusal is an allowed repair.
        path = self.tmp / "out" / "wrapper-log.json"
        log = json.loads(path.read_text()) if path.exists() else {}
        return rc, log, heads, children

    def assert_incomplete(self, rc, log):
        self.assertNotEqual(log.get("result"), "SECRET_FOUND_IN_OUTPUT")
        self.assertFalse(log.get("stop_code", "").startswith("internal_error:"),
                         "an internal error is not an acceptance-contract repair")
        self.assertFalse(rc == 0 and log.get("result") == PASS,
                         "required evidence absent but final rc/result still advertise PASS; "
                         f"observed probes={[p['kind'] for p in log.get('probes', [])]}")

    def test_control_complete(self):
        rc, log, heads, children = self.scenario()
        self.assertEqual((rc, log.get("result")), (0, PASS))
        self.assertEqual([(p["kind"], p["status"], p["pass"]) for p in log["probes"]],
                         [("out_of_scope", 403, True), ("expired", 403, True), ("revoked", 401, True)])
        self.assertEqual(log["main_requests"], 18)
        self.assertEqual((len(heads), len(children)), (3, 1))

    def test_required_revocation_skip_is_incomplete(self):
        rc, log, heads, children = self.scenario(expired=False, answer="SKIP")
        self.assert_incomplete(rc, log)

    def test_required_expiry_deferred_is_incomplete(self):
        rc, log, heads, children = self.scenario(expiry_after=1200)
        self.assert_incomplete(rc, log)

    def test_delayed_credentials_recheck_before_any_request(self):
        # Two getpass calls each advance 1,801s: beyond window AND scope expiry.
        _rc, _log, heads, children = self.scenario(expired=False, answer="SKIP", input_delay=1801)
        self.assertEqual((len(heads), len(children)), (0, 0),
                         "expired prompt wait must stop before HEAD and runner invocation")

    def probe403(self, content_type, mitigated):
        meta = {"content_type": content_type, "cf_mitigated_present": mitigated, "cf_ray_present": False}
        with patch.object(self.w, "head", return_value=(403, meta)) as head:
            row = self.w.probe("out_of_scope", {"game_key": "123456"}, "fake-probe-token", None)
        self.assertEqual(head.call_count, 1)
        self.assertEqual(row["status"], 403)
        return row["pass"]

    def test_control_plain403(self):
        self.assertTrue(self.probe403("application/json", False))

    def test_control_cf_mitigated403(self):
        self.assertFalse(self.probe403("text/html", True))

    def test_html403_without_cf_header_is_not_scope_evidence(self):
        self.assertFalse(self.probe403("text/html; charset=utf-8", False),
                         "HTML403 without cf-mitigated must not count as gateway scope evidence")

    def test_clock_offset_requires_finite_bounded_number(self):
        at = "2030-01-01T00:00:00.000Z"
        pre = {
            "schema": "pitcheezy-onekey-preconditions-v1", "run_id": "cx17-synthetic", "game_key": "123456",
            "approvals": {"song_go": {"reference": "synthetic-only", "at_utc": at},
                          "independent_review": {"reference": "synthetic-only", "at_utc": at,
                                                 "reviewer": "synthetic-reviewer"}},
            "clock": {"ntp_offset_s": 0, "measured_at_utc": at},
            "teammate_confirmation": {
                "at_utc": at, "reference_private": "synthetic-only", "deployed_source_commit": "0" * 40,
                "deployed_runtime_sha256": {name: {"raw": raw, "lf": lf}
                                            for name, (raw, lf) in self.w.EXPECTED_RUNTIME.items()},
                "scope_mode": "scoped", "scope_feature_deployed": True, "expiry_feature_deployed": True,
                "publish_key_scope": "v1/game/123456.json", "publish_expires_at": "2030-01-01T01:00:00.000Z",
                "upload_enabled_for_window": True, "data_read_enabled": True,
                "notice_switch_off": True, "limiter_allows_18_in_120s": True},
            "game_key_fresh_attested_by_teammate": True,
            "test_window": {"start_utc": at, "end_utc": "2030-01-01T00:30:00.000Z"},
            "credentials": {"bearer_dedicated_new": True, "bearer_delivered_privately": True,
                            "viewer_session_dedicated": True, "viewer_session_delivered_privately": True,
                            "stored_on_disk": False},
            "single_writer": {"teammate_confirms_no_other_writer_on_key": True},
            "operators": {"ours_alias": "synthetic-ours", "theirs_alias": "synthetic-theirs"},
            "abort_contact": {"private_channel_agreed": True, "stop_signal_agreed": True,
                              "teammate_can_toggle_upload_off": True},
            "post_test": {"teammate_will_toggle_upload_off": True, "teammate_will_replace_token_digest": True,
                          "teammate_will_clear_scope": True},
            "target": {"upload_base": "https://localhost/_publish", "public_origin": "https://localhost"},
            "client": {"commit": "0" * 40, "worktree": str(self.tmp), "python": "never-executed-python"},
            "probes": {"out_of_scope_head": True, "expired_head": True, "revoked_head": True},
        }
        self.assertEqual(self.w.validate_pre(pre, harness=True), [], "valid synthetic baseline")
        cases = [("zero", 0, True), ("negative_boundary", -5, True), ("positive_boundary", 5, True),
                 ("nan", float("nan"), False), ("positive_infinity", float("inf"), False),
                 ("negative_infinity", float("-inf"), False), ("above", 5.001, False),
                 ("below", -5.001, False), ("true_bool", True, False), ("false_bool", False, False),
                 ("string", "0", False)]
        for name, value, accepted in cases:
            with self.subTest(clock_case=name):
                pre["clock"]["ntp_offset_s"] = value
                errors = self.w.validate_pre(pre, harness=True)
                self.assertEqual([e for e in errors if not e.startswith("clock.")], [])
                self.assertEqual(len(errors), 0 if accepted else 1,
                                 "clock must be a finite int/float within inclusive [-5, 5]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    options, remaining = parser.parse_known_args()
    if len(options.sha256) != 64 or any(c not in "0123456789abcdef" for c in options.sha256):
        parser.error("--sha256 must be the reviewed lowercase SHA-256")
    SOURCE, SOURCE_SHA = options.source.resolve(), options.sha256
    # Explicit CLI enables the class; imported discovery never reads external source.
    Acceptance.__unittest_skip__ = False
    unittest.main(argv=[sys.argv[0], *remaining])
