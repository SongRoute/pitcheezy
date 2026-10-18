"""Offline v2 wrapper-log consistency check; never a host test or GO decision."""
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


def verdict(invalid=(), incomplete=()):
    return {"status": "INVALID" if invalid else "INCOMPLETE" if incomplete else "LOCAL_EVIDENCE_CONSISTENT",
            "reasons": sorted(set(invalid) | set(incomplete)), "unverified": UNVERIFIED,
            "basis": "V2_SINGLE_WRAPPER_LOG_EMBEDDED_RUNNER_ONLY"}


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


def check(log, scope, expiry, mode):
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
