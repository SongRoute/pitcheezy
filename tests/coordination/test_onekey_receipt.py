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


def fixture_v4(scope=True, expiry=True):
    log = fixture(scope, expiry)
    log.update(schema="pitcheezy-onekey-wrapper-log-v2", runner_pass=True,
               post_requests=1 + int(expiry), required_evidence={"out_of_scope": scope, "expired": expiry, "revoked": True},
               main_result="MAIN_PASS", completion="COMPLETE", revocation="PROVEN_401",
               missing_evidence=[], failed_evidence=[])
    for row in log["probes"]:
        row["gateway_answer"] = True
        if row["kind"] == "revoked": row["same_token_as_main_run"] = True
    return log


def deferred_row(kind="revoked", status=401):
    return {"kind": kind, "status": status, "expected": [{"revoked": 401, "expired": 403, "out_of_scope": 403, "upload_off": 503}[kind]],
            "pass": True, "content_type": "application/json", "cf_mitigated_present": False,
            "gateway_answer": True, "deferred": True, "same_token_as_main_run": True}


def missing_revocation(log):
    log.update(result="MAIN_PASS_INCOMPLETE", completion="INCOMPLETE", revocation="NOT_CHECKED",
               missing_evidence=["revoked"], failed_evidence=[])


class V4ReceiptTests(unittest.TestCase):
    def check(self, log):
        result = checker.check(log, "required" if log["required_evidence"]["out_of_scope"] else "not-selected",
                               "required" if log["required_evidence"]["expired"] else "not-selected", log["mode"])
        self.assertEqual(result["basis"], "V4_SINGLE_WRAPPER_LOG_EMBEDDED_RUNNER_ONLY")
        self.assertEqual(result["unverified"], checker.UNVERIFIED + ["SAME_TARGET_BINDING"])
        return result

    def test_complete_selections_and_cli(self):
        for scope in (False, True):
            for expiry in (False, True):
                for mode in ("LOOPBACK_HARNESS", "REAL_HOST"):
                    log = fixture_v4(scope, expiry); log["mode"] = mode
                    self.assertEqual(self.check(log)["status"], "LOCAL_EVIDENCE_CONSISTENT")
        with tempfile.TemporaryDirectory(prefix="cx21-") as folder:
            path = Path(folder).resolve() / "generated.json"; path.write_text(json.dumps(fixture_v4()))
            output = io.StringIO()
            with redirect_stdout(output):
                rc = checker.main([str(path), "--scope", "required", "--expiry", "required", "--mode", "LOOPBACK_HARNESS"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "LOCAL_EVIDENCE_CONSISTENT")

    def test_skip_and_transport_failure_then_recovery(self):
        for transport in (False, True):
            log = fixture_v4(expiry=False)
            log["probes"].pop(); log["post_requests"] = 0
            if transport:
                log["probes"].append({"kind": "revoked", "status": None, "error": "OSError", "pass": False,
                                      "expected": [401], "same_token_as_main_run": True})
                log["post_requests"] = 1
            missing_revocation(log)
            self.assertEqual(self.check(log)["status"], "INCOMPLETE")
            before = deepcopy(log["probes"])
            log["probes"].append(deferred_row()); log["post_requests"] += 1
            log.update(result=checker.PASS, completion="COMPLETE", revocation="PROVEN_401", missing_evidence=[])
            self.assertEqual(self.check(log)["status"], "LOCAL_EVIDENCE_CONSISTENT")
            self.assertEqual(log["probes"][:-1], before)

    def test_latest_pass_fail_and_missing_win(self):
        for last in ("fail", "missing"):
            log = fixture_v4(expiry=False)
            row = deferred_row(status=503); row["pass"] = False
            if last == "missing":
                row = {"kind": "revoked", "status": None, "error": "operator_interrupt", "pass": False,
                       "expected": [401], "deferred": True, "same_token_as_main_run": True}
                missing_revocation(log)
            else:
                log.update(result="FAIL_OR_STOPPED", completion="FAILED", revocation="NOT_PROVEN", failed_evidence=["revoked"])
            log["probes"].append(row); log["post_requests"] += 1
            self.assertEqual(self.check(log)["status"], "INCOMPLETE")
            log["probes"].append(deferred_row()); log["post_requests"] += 1
            log.update(result=checker.PASS, completion="COMPLETE", revocation="PROVEN_401", missing_evidence=[], failed_evidence=[])
            self.assertEqual(self.check(log)["status"], "LOCAL_EVIDENCE_CONSISTENT")

    def test_deferred_scope_and_optional_evidence_use_post_budget(self):
        for kind in ("out_of_scope", "expired"):
            log = fixture_v4(scope=False, expiry=False)
            log["probes"].append(deferred_row(kind, 403)); log["post_requests"] += 1
            self.assertEqual(log["main_requests"], 17)
            self.assertEqual(self.check(log)["status"], "LOCAL_EVIDENCE_CONSISTENT")
        log = fixture_v4(expiry=False)
        for _ in range(2): log["probes"].append(deferred_row())
        log["post_requests"] = 3
        self.assertEqual(self.check(log)["status"], "LOCAL_EVIDENCE_CONSISTENT")
        log["probes"].append(deferred_row()); log["post_requests"] = 4
        self.assertIn("TOTAL_BUDGET_EXCEEDED", self.check(log)["reasons"])
        for field, value in (("post_requests", 0), ("post_requests", True), ("main_requests", 17)):
            log = fixture_v4(); log[field] = value
            self.assertEqual(self.check(log)["status"], "INVALID")

    def test_summary_required_and_runner_types(self):
        for key, value in (("runner_pass", 1), ("runner_pass", False), ("main_result", "MAIN_FAIL_OR_STOPPED"),
                           ("completion", "INCOMPLETE"), ("result", "MAIN_PASS_INCOMPLETE"), ("revocation", "NOT_CHECKED"),
                           ("missing_evidence", ["revoked"]), ("failed_evidence", ["expired"]), ("required_evidence", {})):
            log = fixture_v4(); log[key] = value
            self.assertEqual(self.check(log)["status"] if key != "required_evidence" else checker.check(log, "required", "required", "LOOPBACK_HARNESS")["status"], "INVALID")
        for value in (1, None, [], "true"):
            log = fixture_v4(); log["required_evidence"]["revoked"] = value
            self.assertEqual(self.check(log)["status"], "INVALID")
        log = fixture_v4(); log["runner"]["exit"] = 2; log["runner_pass"] = False
        log.update(main_result="MAIN_FAIL_OR_STOPPED", completion="FAILED", result="FAIL_OR_STOPPED")
        self.assertEqual(self.check(log)["status"], "INCOMPLETE")
        log["runner"] = None
        self.assertEqual(self.check(log)["status"], "INCOMPLETE")

    def test_gateway_identity_and_probe_shape(self):
        for key, value in (("same_token_as_main_run", False), ("gateway_answer", False), ("cf_mitigated_present", True),
                           ("content_type", "text/plain"), ("content_type", "application/jsonp"), ("content_type", None),
                           ("status", True), ("status", 401.0), ("expected", [True]), ("deferred", "yes"), ("pass", 1)):
            log = fixture_v4(); log["probes"][-1][key] = value
            self.assertEqual(self.check(log)["status"], "INVALID")
        log = fixture_v4(); row = log["probes"][-1]
        row.update(content_type="text/html", gateway_answer=False, **{"pass": False})
        log.update(result="FAIL_OR_STOPPED", completion="FAILED", revocation="NOT_PROVEN", failed_evidence=["revoked"])
        self.assertEqual(self.check(log)["status"], "INCOMPLETE")
        for value in (" Application/JSON ; charset=utf-8", "application/json"):
            log = fixture_v4(); log["probes"][-1]["content_type"] = value
            self.assertEqual(self.check(log)["status"], "LOCAL_EVIDENCE_CONSISTENT")
        log = fixture_v4(); log["probes"][-1] = {"kind": "revoked", "status": None, "expected": [401], "pass": False, "same_token_as_main_run": True}
        missing_revocation(log)
        self.assertIn("PROBE_ERROR_MISSING", self.check(log)["reasons"])

    def test_upload_off_never_replaces_revocation_and_secret_override(self):
        log = fixture_v4(expiry=False); log["probes"].pop(); log["post_requests"] = 1
        row = deferred_row("upload_off", 503); row["revocation_evidence"] = False; log["probes"].append(row)
        missing_revocation(log)
        self.assertEqual(self.check(log)["status"], "INCOMPLETE")
        log["probes"].append(deferred_row()); log["post_requests"] += 1
        log.update(result=checker.PASS, completion="COMPLETE", revocation="PROVEN_401", missing_evidence=[])
        self.assertEqual(self.check(log)["status"], "LOCAL_EVIDENCE_CONSISTENT")
        row["revocation_evidence"] = True
        self.assertEqual(self.check(log)["status"], "INVALID")
        log = fixture_v4(); log.update(secret_found=True, result="SECRET_FOUND_IN_OUTPUT", completion="FAILED")
        self.assertEqual(self.check(log)["status"], "INCOMPLETE")
        log["secret_found"] = "true"
        self.assertEqual(self.check(log)["status"], "INVALID")

    def test_checker_stricter_runner_policy_and_history_structure(self):
        for key, value in (("elapsed_s", 120), ("stop_code", "synthetic_stop")):
            log = fixture_v4(); log["runner"][key] = value
            self.assertEqual(self.check(log)["status"], "INCOMPLETE")
        log = fixture_v4(); log["runner"]["methods"][12] = "DELETE:ok"
        self.assertEqual(self.check(log)["status"], "INVALID")
        log = fixture_v4(expiry=False); log["probes"].append(deferred_row()); log["post_requests"] += 1
        log["probes"][1]["pass"] = True; log["probes"][1]["status"] = 503
        self.assertEqual(self.check(log)["status"], "INVALID")  # Later pass cannot erase an earlier contradiction.


if __name__ == "__main__":
    unittest.main()
