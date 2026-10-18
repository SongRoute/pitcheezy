"""Offline v2/v4 wrapper-log consistency check; never a host test or GO decision."""
import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys

MAX_BYTES = 256 * 1024  # Input safety bound, not a registered experiment budget.
PASS = "PASS_SYNTHETIC_HTTPS"
METHODS = [f"{m}:{'upload_conflict' if i == 12 else 'ok'}" for i, m in enumerate(
    "HEAD HEAD PUT HEAD GET HEAD DELETE HEAD HEAD PUT HEAD GET DELETE HEAD HEAD DELETE HEAD".split())]
UNVERIFIED = ["DEPLOY_GO", "WHOLE_JOINT_TEST_COMPLETION", "INDEPENDENT_SECRET_SCAN",
              "FIRST_REQUEST_WINDOW", "STANDALONE_RUN_IDENTITY", "PARTNER_CONFIRMATION",
              "ACTUAL_GATEWAY_OR_WIRE", "ARTIFACT_AUTHENTICITY"]


def verdict(invalid=(), incomplete=(), basis="V2_SINGLE_WRAPPER_LOG_EMBEDDED_RUNNER_ONLY"):
    return {"status": "INVALID" if invalid else "INCOMPLETE" if incomplete else "LOCAL_EVIDENCE_CONSISTENT",
            "reasons": sorted(set(invalid) | set(incomplete)),
            "unverified": UNVERIFIED + (["SAME_TARGET_BINDING"] if basis == "V4_SINGLE_WRAPPER_LOG_EMBEDDED_RUNNER_ONLY" else []),
            "basis": basis}


class InputRejected(ValueError):
    pass


def load_log(path):
    """One bounded regular file. No directory scan, external receipt, or import."""
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise InputRejected("INPUT_SYMLINK")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise InputRejected("INPUT_NOT_REGULAR")
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if info.st_size > MAX_BYTES:
            raise InputRejected("INPUT_TOO_LARGE")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InputRejected("INPUT_TOO_LARGE")

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise InputRejected("JSON_DUPLICATE_KEY")
            out[key] = value
        return out

    def finite(text):
        number = float(text)
        if not math.isfinite(number):
            raise InputRejected("JSON_NONFINITE_NUMBER")
        return number

    def constant(_text):
        raise InputRejected("JSON_NONFINITE_NUMBER")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=finite, parse_constant=constant)


def check_runner(log, invalid, incomplete):
    runner = log.get("runner")
    if runner is None:
        incomplete.append("RUNNER_MISSING")
    elif type(runner) is not dict:
        invalid.append("RUNNER_NOT_OBJECT")
    else:
        for key, expected in (("exit", 0), ("requests", 17), ("pending_paths", 0)):
            if type(runner.get(key)) is not int: invalid.append("RUNNER_" + key.upper() + "_TYPE")
            elif runner[key] != expected: incomplete.append("RUNNER_" + key.upper() + "_NOT_ACCEPTED")
        elapsed = runner.get("elapsed_s")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed): invalid.append("RUNNER_ELAPSED_TYPE")
        elif not 0 <= elapsed < 120: incomplete.append("RUNNER_ELAPSED_OUTSIDE_BUDGET")
        if type(runner.get("result")) is not str: invalid.append("RUNNER_RESULT_TYPE")
        elif runner["result"] != PASS: incomplete.append("RUNNER_NOT_PASS")
        if type(runner.get("owned_worker_stopped")) is not bool: invalid.append("WORKER_STOP_TYPE")
        elif not runner["owned_worker_stopped"]: incomplete.append("WORKER_STOP_UNCONFIRMED")
        if type(runner.get("remote_cleanup")) is not str: invalid.append("CLEANUP_TYPE")
        elif runner["remote_cleanup"] != "confirmed_tombstone_retained": incomplete.append("CLEANUP_UNCONFIRMED")
        if "stop_code" not in runner: invalid.append("RUNNER_STOP_CODE_MISSING")
        elif runner["stop_code"] is not None: incomplete.append("RUNNER_STOPPED")
        if runner.get("methods") != METHODS: invalid.append("RUNNER_METHOD_SEQUENCE")



def check_v1(log, scope, expiry, mode):
    invalid, incomplete = [], []
    if scope not in ("required", "not-selected") or expiry not in ("required", "not-selected") or mode not in ("LOOPBACK_HARNESS", "REAL_HOST"):
        return verdict(["SELECTION_INVALID"])
    if type(log) is not dict:
        return verdict(["LOG_NOT_OBJECT"])
    if log.get("schema") != "pitcheezy-onekey-wrapper-log-v1": invalid.append("SCHEMA_MISMATCH")
    if log.get("mode") != mode: invalid.append("MODE_MISMATCH")
    if log.get("credentials_recorded") is not False: invalid.append("CREDENTIALS_FLAG_INVALID")
    if type(log.get("result")) is not str: invalid.append("WRAPPER_RESULT_INVALID")
    elif log["result"] != PASS: incomplete.append("WRAPPER_NOT_PASS")
    if log.get("stop_code") is not None: incomplete.append("WRAPPER_STOPPED")

    check_runner(log, invalid, incomplete)

    probes = log.get("probes")
    if type(probes) is not list:
        return verdict(invalid + ["PROBES_NOT_LIST"], incomplete)
    if len(probes) > 4: invalid.append("PROBE_COUNT_EXCEEDED")
    supported = {"out_of_scope": 403, "expired": 403, "revoked": 401, "upload_off": 503}
    seen, order = set(), []
    for row in probes:
        if type(row) is not dict:
            invalid.append("PROBE_NOT_OBJECT"); continue
        kind = row.get("kind")
        if type(kind) is not str or kind not in supported:
            invalid.append("PROBE_KIND_UNSUPPORTED"); continue
        if kind in seen: invalid.append("PROBE_DUPLICATE")
        seen.add(kind); order.append(kind)
        expected = supported[kind]
        if type(row.get("expected")) is not list or len(row["expected"]) != 1 or type(row["expected"][0]) is not int or row["expected"][0] != expected:
            invalid.append("PROBE_EXPECTED_INVALID")
        status, passed, mitigated = row.get("status"), row.get("pass"), row.get("cf_mitigated_present")
        if type(status) is not int: invalid.append("PROBE_STATUS_TYPE")
        elif status != expected: incomplete.append("PROBE_STATUS_NOT_ACCEPTED")
        if type(passed) is not bool: invalid.append("PROBE_PASS_TYPE")
        elif not passed: incomplete.append("PROBE_NOT_PASS")
        if type(mitigated) is not bool: invalid.append("PROBE_CF_FLAG_TYPE")
        elif mitigated: incomplete.append("PROBE_EDGE_RESPONSE")
        content = row.get("content_type")
        if content is not None and type(content) is not str: invalid.append("PROBE_CONTENT_TYPE_INVALID")
        html = type(content) is str and "html" in content.lower()
        if html: incomplete.append("PROBE_HTML_RESPONSE")
        if passed is True and (type(status) is not int or status != expected or mitigated is True or html):
            invalid.append("PROBE_PASS_CONTRADICTION")
        if kind == "upload_off" and row.get("revocation_evidence") is not False:
            invalid.append("UPLOAD_OFF_REVOCATION_FLAG_INVALID")
    if order != [kind for kind in supported if kind in seen]: invalid.append("PROBE_ORDER_INVALID")
    for kind, selection in (("out_of_scope", scope), ("expired", expiry)):
        if selection == "required" and kind not in seen: incomplete.append("REQUIRED_" + kind.upper() + "_MISSING")
        if selection == "not-selected" and kind in seen: invalid.append("UNSELECTED_" + kind.upper() + "_PRESENT")
    if "revoked" not in seen: incomplete.append("REQUIRED_REVOKED_MISSING")
    main = log.get("main_requests")
    expected_main = 17 + int("out_of_scope" in seen)
    post = sum(type(row) is dict and row.get("kind") in ("expired", "revoked", "upload_off") for row in probes)
    if type(main) is not int: invalid.append("MAIN_COUNT_TYPE")
    elif main != expected_main or main > 18: invalid.append("MAIN_COUNT_INCONSISTENT")
    if post > 3 or (type(main) is int and main + post > 21): invalid.append("TOTAL_BUDGET_EXCEEDED")
    return verdict(invalid, incomplete)


def check_v2(log, scope, expiry, mode):
    """v4 source's log-v2 semantics, plus the stricter offline runner checks."""
    invalid, incomplete = [], []
    basis = "V4_SINGLE_WRAPPER_LOG_EMBEDDED_RUNNER_ONLY"
    if scope not in ("required", "not-selected") or expiry not in ("required", "not-selected") or mode not in ("LOOPBACK_HARNESS", "REAL_HOST"):
        return verdict(["SELECTION_INVALID"], basis=basis)
    if log.get("mode") != mode: invalid.append("MODE_MISMATCH")
    if log.get("credentials_recorded") is not False: invalid.append("CREDENTIALS_FLAG_INVALID")
    if log.get("stop_code") is not None: incomplete.append("WRAPPER_STOPPED")
    if "secret_found" in log and type(log["secret_found"]) is not bool: invalid.append("SECRET_FOUND_TYPE")
    if log.get("secret_found") is True: incomplete.append("SECRET_FOUND")
    required = {"out_of_scope": scope == "required", "expired": expiry == "required", "revoked": True}
    req = log.get("required_evidence")
    if type(req) is not dict or set(req) != set(required) or any(type(req[k]) is not bool or req[k] != required[k] for k in required):
        invalid.append("REQUIRED_EVIDENCE_MISMATCH")
    check_runner(log, invalid, incomplete)
    runner = log.get("runner")
    # This expression mirrors cmd_run; elapsed/methods/stop are additional checker policy.
    runner_pass = type(runner) is dict and runner.get("result") == PASS and type(runner.get("requests")) is int and runner["requests"] == 17 and type(runner.get("exit")) is int and runner["exit"] == 0 and runner.get("owned_worker_stopped") is True and type(runner.get("pending_paths")) is int and runner["pending_paths"] == 0 and runner.get("remote_cleanup") == "confirmed_tombstone_retained"
    main, post = log.get("main_requests"), log.get("post_requests")
    if type(main) is not int or main < 0: invalid.append("MAIN_COUNT_TYPE")
    if type(post) is not int or post < 0: invalid.append("POST_COUNT_TYPE")
    if type(main) is int and main > 18 or type(post) is int and post > 3 or type(main) is int and type(post) is int and main + post > 21:
        invalid.append("TOTAL_BUDGET_EXCEEDED")
    runner_pass = runner_pass and type(main) is int and main <= 18
    if type(log.get("runner_pass")) is not bool: invalid.append("RUNNER_PASS_TYPE")
    elif log["runner_pass"] != runner_pass: invalid.append("RUNNER_PASS_CONTRADICTION")
    probes = log.get("probes")
    if type(probes) is not list:
        return verdict(invalid + ["PROBES_NOT_LIST"], incomplete, basis)
    expected_status = {"out_of_scope": 403, "expired": 403, "revoked": 401, "upload_off": 503}
    latest, initial, deferred_seen = {}, [], False
    main_rows = post_rows = 0
    for row in probes:
        if type(row) is not dict:
            invalid.append("PROBE_NOT_OBJECT"); continue
        kind = row.get("kind")
        if type(kind) is not str or kind not in expected_status:
            invalid.append("PROBE_KIND_UNSUPPORTED"); continue
        deferred = row.get("deferred", False)
        if type(deferred) is not bool: invalid.append("PROBE_DEFERRED_TYPE")
        if deferred is True:
            deferred_seen = True
        else:
            if deferred_seen: invalid.append("PROBE_ORDER_INVALID")
            initial.append(kind)
        if kind == "out_of_scope" and deferred is not True: main_rows += 1
        else: post_rows += 1
        if kind in ("out_of_scope", "expired") and deferred is not True and not required[kind]: invalid.append("UNSELECTED_" + kind.upper() + "_PRESENT")
        if kind == "revoked" or deferred is True:
            if row.get("same_token_as_main_run") is not True: invalid.append("PROBE_SAME_TOKEN_UNCONFIRMED")
        expected = expected_status[kind]
        if type(row.get("expected")) is not list or len(row["expected"]) != 1 or type(row["expected"][0]) is not int or row["expected"][0] != expected:
            invalid.append("PROBE_EXPECTED_INVALID")
        status, passed = row.get("status"), row.get("pass")
        if "status" not in row or status is not None and (type(status) is not int or not 100 <= status <= 599): invalid.append("PROBE_STATUS_TYPE")
        if type(passed) is not bool: invalid.append("PROBE_PASS_TYPE")
        if status is None:
            if type(row.get("error")) is not str or not row["error"]: invalid.append("PROBE_ERROR_MISSING")
            if passed is not False or row.get("gateway_answer", False) is not False: invalid.append("PROBE_PASS_CONTRADICTION")
            latest[kind] = "missing"
        else:
            content, mitigated = row.get("content_type"), row.get("cf_mitigated_present")
            if content is not None and type(content) is not str: invalid.append("PROBE_CONTENT_TYPE_INVALID")
            if type(mitigated) is not bool: invalid.append("PROBE_CF_FLAG_TYPE")
            gateway = type(content) is str and content.split(";")[0].strip().lower() == "application/json" and mitigated is False
            if type(row.get("gateway_answer")) is not bool or row["gateway_answer"] != gateway: invalid.append("PROBE_GATEWAY_CONTRADICTION")
            accepted = type(status) is int and status == expected and gateway
            if type(passed) is bool and passed != accepted: invalid.append("PROBE_PASS_CONTRADICTION")
            if "error" in row: invalid.append("PROBE_ERROR_WITH_STATUS")
            latest[kind] = "pass" if accepted else "fail"
        # Transport-error upload_off rows return before this flag is added by the source.
        if kind == "upload_off" and status is not None and row.get("revocation_evidence") is not False:
            invalid.append("UPLOAD_OFF_REVOCATION_FLAG_INVALID")
    if initial != [kind for kind in ("out_of_scope", "expired", "revoked") if kind in initial]: invalid.append("PROBE_ORDER_INVALID")
    if main_rows > 1: invalid.append("MAIN_PROBE_DUPLICATE")
    if type(post) is int and post != post_rows: invalid.append("POST_COUNT_INCONSISTENT")
    if type(runner) is dict and type(runner.get("requests")) is int and type(main) is int and main != runner["requests"] + main_rows:
        invalid.append("MAIN_COUNT_INCONSISTENT")
    main_pass = runner_pass and (not required["out_of_scope"] or latest.get("out_of_scope") == "pass")
    missing = sorted(k for k in ("expired", "revoked") if required[k] and latest.get(k, "missing") == "missing")
    failed = sorted(k for k in ("expired", "revoked") if required[k] and latest.get(k) == "fail")
    completion = "FAILED" if not main_pass or failed or log.get("secret_found") is True else "INCOMPLETE" if missing else "COMPLETE"
    result = {"FAILED": "FAIL_OR_STOPPED", "INCOMPLETE": "MAIN_PASS_INCOMPLETE", "COMPLETE": PASS}[completion]
    if log.get("secret_found") is True and log.get("result") == "SECRET_FOUND_IN_OUTPUT": result = "SECRET_FOUND_IN_OUTPUT"
    summaries = {"main_result": "MAIN_PASS" if main_pass else "MAIN_FAIL_OR_STOPPED", "completion": completion,
                 "result": result, "revocation": {"pass": "PROVEN_401", "missing": "NOT_CHECKED", "fail": "NOT_PROVEN"}[latest.get("revoked", "missing")],
                 "missing_evidence": missing, "failed_evidence": failed}
    for key, expected in summaries.items():
        if type(log.get(key)) is not type(expected) or log[key] != expected: invalid.append("SUMMARY_" + key.upper() + "_CONTRADICTION")
    if not main_pass: incomplete.append("MAIN_NOT_PASS")
    incomplete.extend("REQUIRED_" + kind.upper() + "_MISSING" for kind in missing)
    incomplete.extend("REQUIRED_" + kind.upper() + "_FAILED" for kind in failed)
    return verdict(invalid, incomplete, basis)


def check(log, scope, expiry, mode):
    if type(log) is dict and log.get("schema") == "pitcheezy-onekey-wrapper-log-v2":
        return check_v2(log, scope, expiry, mode)
    return check_v1(log, scope, expiry, mode)


class Parser(argparse.ArgumentParser):
    def error(self, _message):
        print(json.dumps(verdict(["CLI_ARGUMENTS_INVALID"])))
        raise SystemExit(2)


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--scope", choices=("required", "not-selected"), required=True)
    parser.add_argument("--expiry", choices=("required", "not-selected"), required=True)
    parser.add_argument("--mode", choices=("LOOPBACK_HARNESS", "REAL_HOST"), required=True)
    args = parser.parse_args(argv)
    try:
        result = check(load_log(args.log), args.scope, args.expiry, args.mode)
    except InputRejected as error:
        result = verdict([str(error)])  # Only our fixed reason codes, never raw input.
    except (OSError, ValueError, UnicodeError, RecursionError, OverflowError):
        result = verdict(["INPUT_UNREADABLE_OR_INVALID_JSON"])
    print(json.dumps(result, sort_keys=True))
    return {"LOCAL_EVIDENCE_CONSISTENT": 0, "INCOMPLETE": 1, "INVALID": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
