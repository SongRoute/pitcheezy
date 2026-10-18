"""Generated JSON only: no original wrapper, private logs, network or subprocess."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import check_onekey_receipt as checker


def fixture(scope=True, expiry=True, upload_off=False):
    probes = []
    for kind in (["out_of_scope"] if scope else []) + (["expired"] if expiry else []) + ["revoked"] + (["upload_off"] if upload_off else []):
        status = {"out_of_scope": 403, "expired": 403, "revoked": 401, "upload_off": 503}[kind]
        probes.append({"kind": kind, "status": status, "expected": [status], "pass": True,
                       "cf_mitigated_present": False, "content_type": "application/json"})
        if kind == "upload_off": probes[-1]["revocation_evidence"] = False
    return {"schema": "pitcheezy-onekey-wrapper-log-v1", "mode": "LOOPBACK_HARNESS", "result": checker.PASS,
            "credentials_recorded": False, "main_requests": 17 + int(scope), "probes": probes,
            "runner": {"exit": 0, "result": checker.PASS, "requests": 17, "elapsed_s": 0.1,
                       "owned_worker_stopped": True, "pending_paths": 0, "remote_cleanup": "confirmed_tombstone_retained",
                       "stop_code": None, "methods": [
                           "HEAD:ok", "HEAD:ok", "PUT:ok", "HEAD:ok", "GET:ok", "HEAD:ok", "DELETE:ok", "HEAD:ok",
                           "HEAD:ok", "PUT:ok", "HEAD:ok", "GET:ok", "DELETE:upload_conflict", "HEAD:ok", "HEAD:ok", "DELETE:ok", "HEAD:ok"]}}


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cx06-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def cli(self, raw, *extra):
        path = self.root / "synthetic.json"
        path.write_text(raw)
        output = io.StringIO()
        with redirect_stdout(output):
            rc = checker.main([str(path), "--scope", "required", "--expiry", "required", "--mode", "LOOPBACK_HARNESS", *extra])
        result = json.loads(output.getvalue())
        self.assertEqual(result["unverified"], checker.UNVERIFIED)
        return rc, result

    def test_selected_combinations_and_modes(self):
        for scope in (False, True):
            for expiry in (False, True):
                for off in (False, True):
                    for mode in ("LOOPBACK_HARNESS", "REAL_HOST"):
                        with self.subTest(scope=scope, expiry=expiry, off=off, mode=mode):
                            log = fixture(scope, expiry, off); log["mode"] = mode
                            result = checker.check(log, "required" if scope else "not-selected", "required" if expiry else "not-selected", mode)
                            self.assertEqual(result["status"], "LOCAL_EVIDENCE_CONSISTENT")
        self.assertEqual(self.cli(json.dumps(fixture(upload_off=True)))[0], 0)

    def test_missing_required_evidence_is_incomplete(self):
        for kind in ("out_of_scope", "expired", "revoked"):
            with self.subTest(kind=kind):
                log = fixture(); log["probes"] = [p for p in log["probes"] if p["kind"] != kind]
                if kind == "out_of_scope": log["main_requests"] = 17
                rc, result = self.cli(json.dumps(log))
                self.assertEqual((rc, result["status"]), (1, "INCOMPLETE"))
                self.assertIn("REQUIRED_" + kind.upper() + "_MISSING", result["reasons"])

    def test_runner_failures_are_not_complete(self):
        for key, value in (("exit", 2), ("requests", 16), ("elapsed_s", 120), ("elapsed_s", -1),
                           ("pending_paths", 1), ("owned_worker_stopped", False), ("remote_cleanup", "unknown"),
                           ("stop_code", "synthetic_stop"), ("result", "FAIL")):
            with self.subTest(key=key, value=value):
                log = fixture(); log["runner"][key] = value
                self.assertEqual(self.cli(json.dumps(log))[1]["status"], "INCOMPLETE")

    def test_bool_numeric_and_nonfinite_are_invalid(self):
        for key in ("exit", "requests", "elapsed_s", "pending_paths"):
            log = fixture(); log["runner"][key] = False
            self.assertEqual(self.cli(json.dumps(log))[1]["status"], "INVALID")
        for constant in ("NaN", "Infinity", "-Infinity", "1e999"):
            self.assertEqual(self.cli('{"value":' + constant + '}')[1]["reasons"], ["JSON_NONFINITE_NUMBER"])
        for field in ("status", "expected"):
            log = fixture(); log["probes"][0][field] = True if field == "status" else [True]
            self.assertEqual(self.cli(json.dumps(log))[1]["status"], "INVALID")

    def test_probe_selection_and_contradictions(self):
        for change in ("duplicate", "unknown", "order", "html", "edge", "status", "expected", "pass_type", "off_as_revoke"):
            with self.subTest(change=change):
                log = fixture(upload_off=True); row = log["probes"][0]
                if change == "duplicate": log["probes"].append(deepcopy(row))
                if change == "unknown": row["kind"] = "unsupported"
                if change == "order": log["probes"].reverse()
                if change == "html": row["content_type"] = "Text/HTML; charset=utf-8"
                if change == "edge": row["cf_mitigated_present"] = True
                if change == "status": row["status"] = 503
                if change == "expected": row["expected"] = [401]
                if change == "pass_type": row["pass"] = 1
                if change == "off_as_revoke": log["probes"][-1]["revocation_evidence"] = True
                self.assertEqual(self.cli(json.dumps(log))[1]["status"], "INVALID")
        log = fixture(); log["probes"][0]["content_type"] = None
        self.assertEqual(self.cli(json.dumps(log))[0], 0)  # Absence is not gateway proof; always unverified.
        self.assertEqual(self.cli(json.dumps(fixture()), "--scope", "not-selected")[1]["status"], "INVALID")
        self.assertEqual(self.cli(json.dumps(fixture()), "--expiry", "not-selected")[1]["status"], "INVALID")

    def test_count_method_mode_and_top_level_guards(self):
        cases = [("main_requests", 19), ("main_requests", 17), ("main_requests", True),
                 ("mode", "REAL_HOST"), ("schema", "other"), ("credentials_recorded", True),
                 ("probes", {}), ("runner", [])]
        for key, value in cases:
            log = fixture(); log[key] = value
            self.assertEqual(self.cli(json.dumps(log))[1]["status"], "INVALID")
        log = fixture(); log["runner"]["methods"][12] = "DELETE:ok"
        self.assertEqual(self.cli(json.dumps(log))[1]["reasons"], ["RUNNER_METHOD_SEQUENCE"])
        log = fixture(); log["result"] = "FAIL_OR_STOPPED"
        self.assertEqual(self.cli(json.dumps(log))[0], 1)
        self.assertEqual(self.cli('[]')[1]["reasons"], ["LOG_NOT_OBJECT"])

    def test_input_bounds_duplicates_symlinks_and_no_raw_echo(self):
        for raw, reason in (('{"x":1,"x":2}', "JSON_DUPLICATE_KEY"),
                            ('{"x":{"y":1,"y":2}}', "JSON_DUPLICATE_KEY"),
                            ('x' * (checker.MAX_BYTES + 1), "INPUT_TOO_LARGE"),
                            ('{"secret":"SYNTHETIC_DO_NOT_ECHO",', "INPUT_UNREADABLE_OR_INVALID_JSON")):
            rc, result = self.cli(raw)
            self.assertEqual((rc, result["reasons"]), (2, [reason]))
            self.assertNotIn("SYNTHETIC_DO_NOT_ECHO", json.dumps(result))
        real = self.root / "real.json"; real.write_text('{}')
        linked = self.root / "linked.json"; linked.symlink_to(real)
        folder = self.root / "linked-dir"; folder.symlink_to(self.root, target_is_directory=True)
        for path in (linked, folder / "real.json"):
            with self.assertRaisesRegex(checker.InputRejected, "INPUT_SYMLINK"): checker.load_log(path)
        with self.assertRaisesRegex(checker.InputRejected, "INPUT_NOT_REGULAR"): checker.load_log(self.root)

    def test_cli_requires_all_declarations_without_echo(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            checker.main(["SYNTHETIC_PRIVATE_PATH"])
        self.assertEqual(raised.exception.code, 2)
        self.assertNotIn("SYNTHETIC_PRIVATE_PATH", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["reasons"], ["CLI_ARGUMENTS_INVALID"])


if __name__ == "__main__":
    unittest.main()
