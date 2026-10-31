#!/usr/bin/env python3
"""L-onekey: one-key synthetic HTTPS joint test wrapper around apps/observer/scripts/host_rehearsal.py.

CANDIDATE v7 (ONEKEY-PROBE-01, not registered, not delivered): onekey_wrapper-v6.final2-000e4ba0.py
(sha256 f41a7b27...) + the probe identity binding below. It reads REGISTRATION-v7.candidate.json, which does not
exist, so it cannot run until the PM re-pins it into L-onekey (tools/repin.py), reviews and re-delivers it.
Binding: cmd_run stores a salted digest of game_key, scope_mode, publish_key_scope and publish_expires_at in
wrapper-log.json; cmd_probe refuses (before reading the token or sending any HEAD) when the PRECONDITIONS given to
the deferred probe differ. This is a local consistency refusal only; it does not verify server-side evidence.

Refuses unless a filled PRECONDITIONS.json passes every check. Secrets (publish bearer, viewer session cookie)
are read with getpass and placed only in the runner child's environment; they are never written to disk or
printed. No automatic retry, no cleanup after an unknown result.

  check     validate PRECONDITIONS.json only (no network, no secrets)
  run       [--dry-run] validate, build plan/grant, then (unless --dry-run) out-of-scope probe -> runner -> optional
            expired probe -> revoked probe after the operator types REVOKED
            result: PASS_SYNTHETIC_HTTPS (rc 0) only when main test AND every required post check passed;
            MAIN_PASS_INCOMPLETE (rc 4) when evidence is still missing; FAIL_OR_STOPPED (rc 2) otherwise
  probe     deferred post-window HEAD appended to the main run's log (--out = that run folder); the re-typed
            token must match the main run's salted fingerprint (kind: out_of_scope | expired | revoked | upload_off)
            revoked passes only on 401 (token digest replaced while uploads are still ON); upload_off (503) is a
            record that uploads were switched off afterwards, never revocation evidence.

--loopback-harness-cert CERT switches to the owned loopback gateway harness (localhost targets, scope
local_synthetic, credentials may come from pre-set env). It never touches a real host.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
DEFAULT_LEDGER = HERE / "USED-KEYS.json"
REGISTRATION = HERE / "REGISTRATION-v7.candidate.json"   # candidate: unregistered on purpose (see docstring)
RUNNER_REL = "apps/observer/scripts/host_rehearsal.py"
SOURCE_ALLOWLIST = ("apps/observer/backend/observer_app/publisher.py",
                    "apps/observer/backend/observer_app/upload_worker.py",
                    "apps/observer/publish_config.json", RUNNER_REL)
BUILD_REL = "apps/observer/web/dist-public"
GATEWAY_PIN = "48ec406589982547e298ff2ee4bc2d64ed0216c0"
PARTNER_PACK_COMMIT = "3ea7ad69781daec8e80ce257ea39633ad6b2770a"
TOKEN_ENV, COOKIE_ENV = "PITCHEEZY_REHEARSAL_UPLOAD_TOKEN", "PITCHEEZY_REHEARSAL_ACCESS_COOKIE"
# Teammate pack 3ea7ad6 runtime (= PR42 48ec406 after newline normalization). raw = pack bytes (CRLF), lf = normalized.
EXPECTED_RUNTIME = {
    "worker.mjs": ("9f3d818686bc6af847bf4c292658a1c71f22f4dd72f6282c8b00e84e0ca33d1b",
                   "ba6bb83bbbbd8ba4727feb6b870fbe85c0c8c5c9bc65ef837aa2703235f3f985"),
    "gateway.mjs": ("fbc1f5d652c124f6827d4065caaf62411f3eba0c311da747ad8473c5d71f7ca0",
                    "3fd0217beae30ae23ca548c309e2535b66e80024728b10ddb249eec08f2dd2b6"),
    "notice.mjs": ("20433bdb8723fe5f9073e39d8f466f00a7fc7e95b26fb304903103473b62733a",
                   "d94b5c22ef66b38b3bb824e028fcaa92075856014bbc7e2e28bd3a85bdb64247"),
    "access.mjs": ("2513d96b0950b8917d45a37e6b4c4a7d605df820a8e294e9b30b8a7beb6a4707",
                   "67c3865be7754a9cf172075db67f60a8bc2cd076203799b0bf64a8fe0ef1219a"),
    "telemetry.mjs": ("545569404a68e8a5159984c3b328e4982c217247d1cd5af01e1831165ebf6739",
                      "c90d0f23dbf80eb965d343bef8fb75129cf3ce7ee9e64fbe6f683228ad0f646a"),
}
POST_BUDGET = 3            # post-window HEADs: expired (opt) + revoked + upload_off (opt)
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43,128}")   # gateway bearer shape (same as HttpSink)
MAIN_BUDGET = 18           # agreed: requests in the main window (1 out-of-scope probe + runner's 17-step sequence)
MIN_MARGIN_S = 180         # runner supervisor 114 s + slack must fit before window end and scope expiry
MAX_EXPIRY_WAIT_S = 900
EXPIRY_CLOCK_MARGIN_S = 10  # gateway compares against the edge clock; wait this long past PUBLISH_EXPIRES_AT
MAX_NTP_OFFSET_S = 5
OP_TIMEOUT_S = 5
ISO_MS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")
SHA = re.compile(r"[0-9a-f]{64}")
LOCAL_HOSTS = ("localhost", "127.0.0.1")


class Refuse(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def epoch(text: str) -> float:
    if not isinstance(text, str) or not ISO_MS.fullmatch(text):
        raise Refuse(f"time_not_iso_ms_utc:{text!r}")
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc).timestamp()


def is_true(value) -> bool:
    return value is True


def filled(value) -> bool:
    return isinstance(value, str) and value.strip() != "" and not value.startswith("<")


def validate_pre(pre: dict, harness: bool) -> list[str]:
    """Return every unmet precondition (empty list = ready). Pure: no network, no secrets."""
    p: list[str] = []
    if pre.get("schema") != "pitcheezy-onekey-preconditions-v1":
        p.append("schema")
    reg = pre.get("registration_sha256")
    if not harness and (not REGISTRATION.is_file() or reg != sha256(REGISTRATION.read_bytes())):
        p.append(f"registration_sha256 (must equal sha256 of {REGISTRATION.name})")
    if not re.fullmatch(r"[a-z0-9-]{8,64}", str(pre.get("run_id") or "")):
        p.append("run_id")
    ap = pre.get("approvals") or {}
    for name in ("song_go", "independent_review"):
        a = ap.get(name) or {}
        if not filled(a.get("reference")) or not filled(a.get("at_utc")):
            p.append(f"approvals.{name}")
    clock = pre.get("clock") or {}
    if type(clock.get("ntp_offset_s")) not in (int, float) or not (-MAX_NTP_OFFSET_S <= clock["ntp_offset_s"] <= MAX_NTP_OFFSET_S) \
            or not filled(clock.get("measured_at_utc")):
        p.append(f"clock.ntp_offset_s/measured_at_utc (record 'sntp time.apple.com' offset; |offset| <= {MAX_NTP_OFFSET_S} s)")
    if not filled((ap.get("independent_review") or {}).get("reviewer")) or \
            (ap.get("independent_review") or {}).get("reviewer") == "claude_author_L":
        p.append("approvals.independent_review.reviewer (different agent than claude_author_L)")
    tc = pre.get("teammate_confirmation") or {}
    for key in ("at_utc", "reference_private"):
        if not filled(tc.get(key)):
            p.append(f"teammate_confirmation.{key}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(tc.get("deployed_source_commit") or "")):
        p.append("teammate_confirmation.deployed_source_commit (40 hex)")
    hashes = tc.get("deployed_runtime_sha256") or {}
    for name, (raw, lf) in EXPECTED_RUNTIME.items():
        got = hashes.get(name) or {}
        if got.get("lf") != lf or got.get("raw") not in (raw, lf):
            p.append(f"teammate_confirmation.deployed_runtime_sha256.{name} (raw and lf must both be recorded and match)")
    mode = tc.get("scope_mode")
    key = str(pre.get("game_key") or "")
    if not re.fullmatch(r"[1-9][0-9]{0,10}", key):   # <= 11 digits so key+1 stays a valid runner key
        p.append("game_key (1-11 digits)")
    if mode == "scoped":
        if not (is_true(tc.get("scope_feature_deployed")) and is_true(tc.get("expiry_feature_deployed"))):
            p.append("teammate_confirmation.scope/expiry_feature_deployed")
        if tc.get("publish_key_scope") != f"v1/game/{key}.json":
            p.append("teammate_confirmation.publish_key_scope (must be v1/game/<game_key>.json)")
        try:
            epoch(tc.get("publish_expires_at"))
        except Refuse:
            p.append("teammate_confirmation.publish_expires_at (UTC ms, e.g. 2026-10-20T11:30:00.000Z)")
    elif mode == "unscoped":
        hw = (pre.get("approvals") or {}).get("song_accepts_hostwide_bearer") or {}
        if not (isinstance(hw, dict) and filled(hw.get("at_utc")) and filled(hw.get("reference"))):
            p.append("approvals.song_accepts_hostwide_bearer {at_utc, reference} (required for unscoped mode)")
    else:
        p.append("teammate_confirmation.scope_mode (scoped|unscoped)")
    for flag in ("upload_enabled_for_window", "data_read_enabled", "notice_switch_off", "limiter_allows_18_in_120s"):
        if not is_true(tc.get(flag)):
            p.append(f"teammate_confirmation.{flag}")
    if not is_true(pre.get("game_key_fresh_attested_by_teammate")):
        p.append("game_key_fresh_attested_by_teammate")
    w = pre.get("test_window") or {}
    try:
        if epoch(w.get("start_utc")) >= epoch(w.get("end_utc")):
            p.append("test_window (start < end)")
    except Refuse:
        p.append("test_window.start_utc/end_utc (UTC ms)")
    cr = pre.get("credentials") or {}
    for flag in ("bearer_dedicated_new", "bearer_delivered_privately", "viewer_session_dedicated",
                 "viewer_session_delivered_privately"):
        if not is_true(cr.get(flag)):
            p.append(f"credentials.{flag}")
    if cr.get("stored_on_disk") is not False:
        p.append("credentials.stored_on_disk (must be false)")
    if not is_true((pre.get("single_writer") or {}).get("teammate_confirms_no_other_writer_on_key")):
        p.append("single_writer.teammate_confirms_no_other_writer_on_key")
    op = pre.get("operators") or {}
    for side in ("ours_alias", "theirs_alias"):
        if not filled(op.get(side)):
            p.append(f"operators.{side}")
    ab = pre.get("abort_contact") or {}
    for flag in ("private_channel_agreed", "stop_signal_agreed", "teammate_can_toggle_upload_off"):
        if not is_true(ab.get(flag)):
            p.append(f"abort_contact.{flag}")
    pt = pre.get("post_test") or {}
    for flag in ("teammate_will_toggle_upload_off", "teammate_will_replace_token_digest"):
        if not is_true(pt.get(flag)):
            p.append(f"post_test.{flag}")
    if mode == "scoped" and not is_true(pt.get("teammate_will_clear_scope")):
        p.append("post_test.teammate_will_clear_scope")
    t = pre.get("target") or {}
    try:
        up, pub = urlsplit(t.get("upload_base") or ""), urlsplit(t.get("public_origin") or "")
        ok = (up.scheme == pub.scheme == "https" and up.hostname and up.netloc == pub.netloc
              and up.path == "/_publish" and pub.path in ("", "/") and not (up.query or up.fragment or up.username))
        local = up.hostname in LOCAL_HOSTS
        if not ok or local != harness:
            p.append("target (https same origin, upload_base=<origin>/_publish; localhost only in harness mode)")
    except ValueError:
        p.append("target")
    c = pre.get("client") or {}
    if not re.fullmatch(r"[0-9a-f]{40}", str(c.get("commit") or "")):
        p.append("client.commit (40 hex)")
    if not filled(c.get("worktree")) or not filled(c.get("python")):
        p.append("client.worktree/python")
    pr = pre.get("probes") or {}
    for flag in ("out_of_scope_head", "expired_head", "revoked_head"):
        if not isinstance(pr.get(flag), bool):
            p.append(f"probes.{flag} (true/false)")
    if mode != "scoped" and (pr.get("out_of_scope_head") or pr.get("expired_head")):
        p.append("probes: out_of_scope/expired need scope_mode=scoped")
    if pr.get("revoked_head") is not True:
        p.append("probes.revoked_head (must be true)")
    return p


def check_client(pre: dict) -> dict:
    wt = Path(pre["client"]["worktree"])
    head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != pre["client"]["commit"]:
        raise Refuse("client_commit_mismatch")
    if subprocess.run(["git", "-C", str(wt), "diff", "--quiet", "HEAD", "--", *SOURCE_ALLOWLIST]).returncode != 0:
        raise Refuse("client_pinned_sources_dirty")
    if f"GATEWAY = '{GATEWAY_PIN}'" not in (wt / RUNNER_REL).read_text():
        raise Refuse("runner_gateway_pin_mismatch")
    return {"commit": head, "source_sha256": {n: sha256((wt / n).read_bytes()) for n in SOURCE_ALLOWLIST}}


def build_plan_grant(pre: dict, client: dict, harness: bool) -> tuple[dict, dict]:
    wt = Path(pre["client"]["worktree"])
    build = wt / BUILD_REL
    plan = {
        "schema": "pitcheezy-dedicated-https-rehearsal-v1",
        "publisher_base": "1f0260fa0d2da40b229507d5ca15f45d7c13593d",
        "gateway_commit": GATEWAY_PIN,
        "run_id": pre["run_id"],
        "game_key": str(pre["game_key"]),
        "upload_base": pre["target"]["upload_base"],
        "public_origin": pre["target"]["public_origin"],
        "scope": "local_synthetic" if harness else "approved_https_synthetic",
        "execute_enabled": True,
        "token_env": TOKEN_ENV,
        "cookie_env": COOKIE_ENV,
        "source_sha256": client["source_sha256"],
        "public_build": str(build),
        "public_assets": {q.relative_to(build).as_posix(): sha256(q.read_bytes())
                          for q in sorted(build.rglob("*")) if q.is_file()},
    }
    if not harness:
        pins = json.loads(REGISTRATION.read_text())["client_pins"]
        if plan["source_sha256"] != pins["source_sha256"] or plan["public_assets"] != pins["public_assets_sha256"]:
            raise Refuse("client_files_differ_from_registration_pins")
    end = epoch(pre["test_window"]["end_utc"])
    ap = pre["approvals"]
    grant = {
        "plan_sha256": sha256(canonical(plan)),
        "authority": (f"song_go {ap['song_go']['reference']} @ {ap['song_go']['at_utc']}; "
                      f"review {ap['independent_review']['reviewer']} {ap['independent_review']['reference']}; "
                      f"teammate {pre['teammate_confirmation']['reference_private']}"),
        "expires_epoch": min(end, time.time() + 3500),
        "execute_synthetic_only": True, "exclusive_writer": True, "dedicated_target": True,
        "read_upload_enabled": True, "cookie_access_allowed": True,
    }
    return plan, grant


def check_window(pre: dict) -> None:
    t = time.time()
    w = pre["test_window"]
    if t < epoch(w["start_utc"]):
        raise Refuse("before_test_window")
    if t + MIN_MARGIN_S > epoch(w["end_utc"]):
        raise Refuse("too_close_to_window_end")
    if pre["teammate_confirmation"]["scope_mode"] == "scoped" and \
            t + MIN_MARGIN_S > epoch(pre["teammate_confirmation"]["publish_expires_at"]):
        raise Refuse("too_close_to_publish_expires_at")


def ledger_load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"schema": "pitcheezy-onekey-used-keys-v1", "keys": []}


def ledger_mark(path: Path, key: str, run_id: str, status: str) -> None:
    data = ledger_load(path)
    rows = [r for r in data["keys"] if not (r["game_key"] == key and r["run_id"] == run_id)]
    rows.append({"game_key": key, "run_id": run_id, "status": status, "at_utc": now_utc()})
    data["keys"] = rows
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_secrets(harness: bool) -> tuple[str, str]:
    if harness and os.environ.get(TOKEN_ENV) and os.environ.get(COOKIE_ENV):
        return os.environ[TOKEN_ENV], os.environ[COOKIE_ENV]
    if not harness and (TOKEN_ENV in os.environ or COOKIE_ENV in os.environ):
        raise Refuse("credential_preset_in_environment (type it at the prompt instead)")
    require_tty(harness)
    token = read_token(harness)
    cookie = getpass.getpass("관람 세션 쿠키 'name=value' 붙여넣기 (화면에 안 보임): ").strip()
    if not cookie or len(cookie) > 4096 or any(ord(ch) < 32 or ord(ch) > 126 for ch in cookie):
        raise Refuse("credential_format_invalid")
    return token, cookie


def require_tty(harness: bool) -> None:
    # Real mode: secrets are typed only by a human in a local terminal; never via an agent's captured shell.
    if not harness and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise Refuse("credential_prompt_requires_tty")


def read_token(harness: bool) -> str:
    if harness and os.environ.get(TOKEN_ENV):
        return os.environ[TOKEN_ENV]
    if not harness and TOKEN_ENV in os.environ:
        raise Refuse("credential_preset_in_environment (type it at the prompt instead)")
    require_tty(harness)
    token = getpass.getpass("발행 bearer 토큰 붙여넣기 (화면에 안 보임): ").strip()
    if not token or len(token) > 128 or any(ord(ch) < 33 or ord(ch) > 126 for ch in token):
        raise Refuse("credential_format_invalid")
    return token


def head(pre: dict, relative: str, token: str, cafile: str | None) -> tuple[int, dict]:
    """Exactly one HEAD to <origin>/_publish/<relative>; returns status + non-secret headers that tell a gateway
    answer from an edge (Cloudflare) answer. No redirects, no retry, no body logged."""
    url = urlsplit(pre["target"]["upload_base"])
    context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    conn = http.client.HTTPSConnection(url.hostname, url.port, timeout=OP_TIMEOUT_S, context=context)
    try:
        conn.request("HEAD", "/_publish/" + relative, headers={"Authorization": "Bearer " + token})
        response = conn.getresponse()
        response.read(1)
        meta = {"content_type": response.getheader("Content-Type"),
                "cache_control": response.getheader("Cache-Control"),
                "cf_mitigated_present": response.getheader("cf-mitigated") is not None,
                "cf_ray_present": response.getheader("cf-ray") is not None}
        return response.status, meta
    finally:
        conn.close()


def probe(kind: str, pre: dict, token: str, cafile: str | None) -> dict:
    key = int(pre["game_key"])
    relative = f"v1/game/{key + 1 if kind == 'out_of_scope' else key}.json"
    # revoked: only 401 proves the token digest was replaced (503 is also upload off / scope half-cleared /
    # limiter or storage unavailable, indistinguishable on a bodyless HEAD). upload_off is a record, not evidence.
    expected = {"out_of_scope": (403,), "expired": (403,), "revoked": (401,), "upload_off": (503,)}[kind]
    started = now_utc()
    try:
        status, meta = head(pre, relative, token, cafile)
    except KeyboardInterrupt:
        return {"kind": kind, "at_utc": started, "status": None, "error": "operator_interrupt", "pass": False,
                "expected": list(expected)}
    except (OSError, http.client.HTTPException, ssl.SSLError) as error:
        return {"kind": kind, "at_utc": started, "status": None, "error": type(error).__name__, "pass": False,
                "expected": list(expected)}
    # F4: only a gateway answer counts. The pinned gateway always sends application/json on these statuses;
    # an HTML/other body or a Cloudflare mitigation marker is an edge answer -> not evidence (unconfirmed).
    gateway_answer = is_gateway_json(meta.get("content_type")) and not meta.get("cf_mitigated_present")
    row = {"kind": kind, "at_utc": started, "status": status, "expected": list(expected),
           "gateway_answer": gateway_answer, "pass": status in expected and gateway_answer} | meta
    if kind == "upload_off":
        row["revocation_evidence"] = False
    return row


def is_gateway_json(content_type) -> bool:
    return isinstance(content_type, str) and content_type.split(";")[0].strip().lower() == "application/json"


def scan_for_secrets(directory: Path, secrets: tuple[str, ...]) -> list[str]:
    hits = []
    for path in directory.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            if any(s and s.encode() in data for s in secrets):
                hits.append(str(path.relative_to(directory)))
    return hits


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def identity_digest(pre: dict, salt: str) -> str:
    """ONEKEY-PROBE-01: the values later judgements depend on, bound at the main run (salted: game key not exposed)."""
    tc = pre["teammate_confirmation"]
    identity = {"game_key": str(pre["game_key"]), "scope_mode": tc.get("scope_mode"),
                "publish_key_scope": tc.get("publish_key_scope"), "publish_expires_at": tc.get("publish_expires_at")}
    return sha256(salt.encode() + b"\n" + canonical(identity))


def token_fingerprint(token: str, salt: str) -> str:
    """Salted digest kept only in the local log so a re-entered token can be shown to be the SAME bearer."""
    return sha256((salt + "\n" + token).encode())


def assess(log: dict) -> None:
    """F2: three separate verdicts.
    main_result  : out-of-scope probe (if required) + runner 17-step sequence with worker stopped, nothing pending,
                   cleanup confirmed, main window <= 18 requests.
    revocation   : PROVEN_401 (same bearer as the main run, gateway 401) | NOT_CHECKED | NOT_PROVEN.
    completion   : COMPLETE only if main passed and every required post-window evidence passed;
                   INCOMPLETE if something required is still missing (SKIP, deferred expiry, interrupt);
                   FAILED if main failed or a required post-window check failed.
    result/rc    : PASS_SYNTHETIC_HTTPS/0 only for COMPLETE; MAIN_PASS_INCOMPLETE/4; FAIL_OR_STOPPED/2.
    N21: per probe kind the LAST row decides (a later passing deferred probe supersedes an earlier failed one);
         every row is kept in the log and listed in the public summary.
    N22: a row without an HTTP status (transport error, timeout, interrupt) is 'missing', not 'fail'."""
    req = log["required_evidence"]
    rows = {}
    for row in log["probes"]:
        rows.setdefault(row["kind"], []).append(row)

    def verdict(kind):
        got = rows.get(kind, [])
        if not got:
            return "missing"
        last = got[-1]
        if last.get("pass"):
            return "pass"
        return "missing" if last.get("status") is None else "fail"

    main_pass = bool(log.get("runner_pass")) and log["main_requests"] <= MAIN_BUDGET and \
        (not req["out_of_scope"] or verdict("out_of_scope") == "pass")
    post = {"expired": verdict("expired")} if req["expired"] else {}
    post["revoked"] = verdict("revoked")
    log["main_result"] = "MAIN_PASS" if main_pass else "MAIN_FAIL_OR_STOPPED"
    log["revocation"] = {"pass": "PROVEN_401", "missing": "NOT_CHECKED", "fail": "NOT_PROVEN"}[post["revoked"]]
    log["missing_evidence"] = sorted(k for k, v in post.items() if v == "missing")
    log["failed_evidence"] = sorted(k for k, v in post.items() if v == "fail")
    if not main_pass or log["failed_evidence"] or log.get("secret_found"):
        log["completion"], log["result"] = "FAILED", "FAIL_OR_STOPPED"
    elif log["missing_evidence"]:
        log["completion"], log["result"] = "INCOMPLETE", "MAIN_PASS_INCOMPLETE"
    else:
        log["completion"], log["result"] = "COMPLETE", "PASS_SYNTHETIC_HTTPS"


def exit_code(log: dict) -> int:
    return {"COMPLETE": 0, "INCOMPLETE": 4}.get(log["completion"], 2)


def public_summary(log: dict) -> str:
    lines = [f"COLLAB-001 한 키 합성 공동 시험 결과 ({log['mode']})", "",
             f"- 전체 판정: {log['result']} (완료 {log.get('completion')})",
             f"- 본 시험: {log.get('main_result')} · 회수 증거: {log.get('revocation')}"
             + (f" · 아직 없는 필수 증거: {', '.join(log['missing_evidence'])}" if log.get("missing_evidence") else "")
             + (f" · 실패한 필수 증거: {', '.join(log['failed_evidence'])}" if log.get("failed_evidence") else ""),
             f"- 시작/끝(UTC): {log['started_utc']} / {log.get('finished_utc')}",
             f"- 본 시험 창 요청 수: {log['main_requests']} (상한 {MAIN_BUDGET}) · 후속 확인 요청 수: "
             f"{log.get('post_requests', 0)} (상한 {POST_BUDGET})"]
    r = log.get("runner") or {}
    if r:
        lines.append(f"- runner: {r.get('result')} · 요청 {r.get('requests')} · 소요 {r.get('elapsed_s')}초 · "
                     f"정리 {r.get('remote_cleanup')} · worker 종료 {r.get('owned_worker_stopped')} · "
                     f"남은 대기 {r.get('pending_paths')}" + (f" · 중단 코드 {r['stop_code']}" if r.get("stop_code") else ""))
        if r.get("methods"):
            lines.append(f"- 요청 순서: {' '.join(r['methods'])}")
    for pr in log.get("probes", []):
        lines.append(f"- {pr['kind']} HEAD{' (나중 실행)' if pr.get('deferred') else ''}: 상태 {pr.get('status')} "
                     f"(기대 {pr.get('expected')}, gateway 응답 {pr.get('gateway_answer')}) → "
                     f"{'통과' if pr.get('pass') else '실패/미확인'}")
    for note in log.get("notes", []):
        lines.append(f"- 메모: {note}")
    lines += ["", "PASS_SYNTHETIC_HTTPS는 본 시험·필수 후속 확인(회수 401 등)이 모두 끝났을 때만 붙습니다. "
              "MAIN_PASS_INCOMPLETE는 본 시험만 통과하고 필수 증거가 남은 상태입니다.",
              "runner 단계별 상태 코드(200/404/412)는 runner가 코드로 강제한 것(영수증 기준)이고, 직접 측정한 코드는 probe 줄뿐입니다.",
              "회수 증거는 본 시험과 같은 토큰의 gateway 401만 인정합니다(503은 업로드 꺼짐 기록일 뿐).",
              "주소·토큰·쿠키·게임 키 값은 적지 않았습니다. 이 기록은 실경기·일반 공개 승인이 아닙니다."]
    return "\n".join(lines) + "\n"


def finish(out: Path, log: dict, secrets: tuple[str, ...]) -> int:
    assess(log)
    log["finished_utc"] = now_utc()
    write_json(out / "wrapper-log.json", log)
    (out / "PUBLIC-SUMMARY.md").write_text(public_summary(log))
    hits = scan_for_secrets(out, secrets)
    if hits:
        log["secret_found"] = True
        log["result"], log["completion"] = "SECRET_FOUND_IN_OUTPUT", "FAILED"
        write_json(out / "wrapper-log.json", log)
        print(json.dumps({"result": "SECRET_FOUND_IN_OUTPUT", "files": hits}))
        return 3
    print(json.dumps({"result": log["result"], "completion": log["completion"], "main_result": log["main_result"],
                      "revocation": log["revocation"], "missing_evidence": log["missing_evidence"],
                      "main_requests": log["main_requests"], "post_requests": log.get("post_requests", 0),
                      "probes": [(p["kind"], p.get("status"), p["pass"]) for p in log["probes"]],
                      "summary": str(out / "PUBLIC-SUMMARY.md")}, ensure_ascii=False))
    return exit_code(log)


def cmd_run(args, pre: dict, harness: bool, cafile: str | None) -> int:
    client = check_client(pre)
    plan, grant = build_plan_grant(pre, client, harness)
    out = Path(args.out)
    if out.exists():
        raise Refuse("out_dir_exists (use a new --out folder; never reuse a run folder)")
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_json(out / "plan.json", plan)          # contains the target URL: local only, never posted
    write_json(out / "grant.json", grant)
    python, runner = pre["client"]["python"], str(Path(pre["client"]["worktree"]) / RUNNER_REL)
    child_env = {k: v for k, v in os.environ.items()
                 if not k.startswith("PITCHEEZY_") and k not in ("SSL_CERT_FILE", "SSL_CERT_DIR")}
    if args.dry_run:
        offline = subprocess.run([python, runner, "--plan", str(out / "plan.json")], capture_output=True,
                                 text=True, env=child_env, timeout=60)
        result = {"result": "DRY_RUN_READY" if offline.returncode == 0 else "DRY_RUN_REFUSED",
                  "runner_offline_validate": json.loads(offline.stdout or "{}"), "network_requests": 0,
                  "plan_sha256": grant["plan_sha256"]}
        write_json(out / "dry-run.json", result)
        print(json.dumps(result))
        return 0 if offline.returncode == 0 else 2
    check_window(pre)
    key, ledger = str(pre["game_key"]), Path(args.ledger)
    if any(r["game_key"] == key for r in ledger_load(ledger)["keys"]):
        raise Refuse("game_key_already_used (never reuse a key)")
    token, cookie = read_secrets(harness)
    token_shape_ok = bool(TOKEN_RE.fullmatch(token))
    if not token_shape_ok:
        # N23 (final2, Song 2026-10-21): hard refusal BEFORE ledger_mark (key not spent) and BEFORE any request.
        # Only a STOPPED log without any credential-derived value (no fingerprint) is written.
        del token, cookie
        write_json(out / "wrapper-log.json", {
            "schema": "pitcheezy-onekey-wrapper-log-v2", "mode": "LOOPBACK_HARNESS" if harness else "REAL_HOST",
            "run_id": pre["run_id"], "plan_sha256": grant["plan_sha256"],
            "target_sha256": sha256(canonical([plan["upload_base"], plan["public_origin"]])),
            "client_commit": client["commit"], "started_utc": now_utc(), "finished_utc": now_utc(),
            "main_requests": 0, "post_requests": 0, "probes": [], "credentials_recorded": False, "runner_pass": False,
            "token_shape_ok": False, "key_spent": False, "stop_code": "token_shape_invalid",
            "result": "STOPPED", "completion": "FAILED",
            "notes": ["토큰 모양이 gateway 형식([A-Za-z0-9_-]{43,128})과 다름: 거부. 요청 0건, 키 장부 기록 안 함(키 미사용). "
                      "토큰을 다시 받아 새 --out 폴더로 처음부터 다시 실행"]})
        raise Refuse("token_shape_invalid")
    check_window(pre)   # F3: the human prompt may have taken long; re-check before the first request
    salt = os.urandom(16).hex()
    log = {"schema": "pitcheezy-onekey-wrapper-log-v2", "mode": "LOOPBACK_HARNESS" if harness else "REAL_HOST",
           "run_id": pre["run_id"], "plan_sha256": grant["plan_sha256"],
           "target_sha256": sha256(canonical([plan["upload_base"], plan["public_origin"]])),
           "client_commit": client["commit"], "started_utc": now_utc(), "main_requests": 0, "post_requests": 0,
           "probes": [], "notes": [], "credentials_recorded": False, "runner_pass": False,
           "required_evidence": {"out_of_scope": pre["probes"]["out_of_scope_head"] is True,
                                 "expired": pre["probes"]["expired_head"] is True, "revoked": True},
           "token_fingerprint": {"salt": salt, "sha256": token_fingerprint(token, salt)},
           "identity_sha256": identity_digest(pre, salt),
           "token_shape_ok": token_shape_ok, "result": "STOPPED", "completion": "FAILED"}
    ledger_mark(ledger, key, pre["run_id"], "attempt_started")   # before any network: the key is spent
    try:
        if pre["probes"]["out_of_scope_head"]:
            row = probe("out_of_scope", pre, token, cafile)
            log["probes"].append(row)
            log["main_requests"] += 1
            if row.get("error") == "operator_interrupt":
                raise Refuse("operator_interrupt")
            if not row["pass"]:
                log["notes"].append("범위 밖 키가 gateway 403이 아님: 범위 강제 미확인 → runner 실행 안 함")
                raise Refuse("scope_not_enforced")
        check_window(pre)   # F3: and again immediately before the runner's first request
        env = dict(child_env, **{TOKEN_ENV: token, COOKIE_ENV: cookie})
        if cafile:
            env["SSL_CERT_FILE"] = cafile
        proc = subprocess.Popen([python, runner, "--plan", str(out / "plan.json"), "--grant", str(out / "grant.json"),
                                 "--execute", "--output", str(out / "runner")],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            stdout, _stderr = proc.communicate(timeout=125)
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
            # The runner parent got the same SIGINT and reaps its own process group; give it time, kill only if stuck.
            try:
                proc.communicate(timeout=10)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                proc.kill()
                proc.communicate()
            interrupted = isinstance(error, KeyboardInterrupt)
            log["notes"].append(("운영자 Ctrl+C" if interrupted else "runner 감독 시간 초과") +
                                ": 원격 상태 미확인, 정리·재시도 없음, 팀원과 읽기 전용 대조 필요")
            log["main_requests"] += MAIN_BUDGET
            raise Refuse("operator_interrupt" if interrupted else "runner_timeout")
        receipt_path = out / "runner" / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        n = len(receipt.get("requests", [])) if receipt else None
        stdout = stdout or ""
        log["runner"] = {"exit": proc.returncode, "stdout": json.loads(stdout) if stdout.strip().startswith("{") else None,
                         "result": receipt.get("result", "UNKNOWN"), "requests": n,
                         "methods": [f"{r['method']}:{'ok' if r['result'] == 'confirmed' else r['result']}"
                                     for r in receipt.get("requests", [])],
                         "elapsed_s": receipt.get("elapsed_s"), "remote_cleanup": receipt.get("remote_cleanup"),
                         "stop_code": receipt.get("stop_code"),
                         "owned_worker_stopped": receipt.get("owned_worker_stopped"),
                         "pending_paths": receipt.get("pending_paths")}
        log["main_requests"] += n if n is not None else MAIN_BUDGET
        if n is None:
            log["notes"].append("runner 영수증 없음: 요청 수를 상한으로 간주, 원격 상태 미확인")
        # F2: runner success also requires its worker to be gone, nothing pending and the final tombstone confirmed.
        log["runner_pass"] = (receipt.get("result") == "PASS_SYNTHETIC_HTTPS" and n == 17 and proc.returncode == 0
                              and receipt.get("owned_worker_stopped") is True and receipt.get("pending_paths") == 0
                              and receipt.get("remote_cleanup") == "confirmed_tombstone_retained")
        if receipt and receipt.get("result") == "PASS_SYNTHETIC_HTTPS" and not log["runner_pass"]:
            log["notes"].append("runner는 PASS라 했지만 worker 종료/대기 0/정리 확인/요청 17 중 하나가 빠짐 → 본 시험 실패")
        if log["main_requests"] > MAIN_BUDGET:
            log["notes"].append("본 시험 창 요청 수가 합의 상한 초과")
            log["runner_pass"] = False
        if log["runner_pass"] and pre["probes"]["expired_head"]:
            wait = epoch(pre["teammate_confirmation"]["publish_expires_at"]) + EXPIRY_CLOCK_MARGIN_S - time.time()
            if wait <= MAX_EXPIRY_WAIT_S:
                print(json.dumps({"waiting_for_expiry_s": round(max(0, wait))}), flush=True)
                try:
                    time.sleep(max(0, wait))
                    log["post_requests"] += 1          # N24: count the attempt, answered or not
                    row = probe("expired", pre, token, cafile)
                    log["probes"].append(row)
                except KeyboardInterrupt:   # N15: keep going to log, scan and the revocation prompt
                    log["notes"].append("만료 확인 건너뜀(운영자 중단): 필수 증거 미완")
            else:
                log["notes"].append("만료까지 15분 초과: 만료 확인은 probe --kind expired로 따로(그 전까지 미완)")
    except Refuse as error:
        log["stop_code"] = str(error)
    except Exception as error:  # still write the log, scan for secrets and offer the revocation check
        log["stop_code"] = f"internal_error:{type(error).__name__}"
        log["notes"].append("래퍼 내부 오류: 원격 상태 미확인, 재시도·정리 없음")
    finally:
        assess(log)
        ledger_mark(ledger, key, pre["run_id"], log["result"])
    # Credential revocation check happens whatever the outcome; the token stays only in this process's memory.
    print(json.dumps({"main_result": log["main_result"],
                      "next_ko": "팀원이 업로드는 켠 채로 토큰 지문만 교체했다고 하면 REVOKED(401이어야 통과), "
                                 "건너뛰려면 SKIP(그러면 전체 판정은 미완)"},
                     ensure_ascii=False), flush=True)
    try:
        sys.stderr.write("REVOKED / SKIP: ")
        sys.stderr.flush()
        answer = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        answer = "SKIP"
    if answer == "REVOKED":
        log["post_requests"] += 1                      # N24: count the attempt, answered or not
        row = probe("revoked", pre, token, cafile)
        row["same_token_as_main_run"] = True
        log["probes"].append(row)
        if row.get("status") is None:                  # N22: neutral wording, stays recoverable
            log["notes"].append("회수 확인 요청이 응답 없이 끝남(전송 오류·시간 초과·중단): 미확인. 업로드를 끄기 전에 같은 토큰으로 다시 확인")
        elif not row["pass"]:
            log["notes"].append("회수 확인 HEAD가 gateway 401이 아님: 회수 증거 없음, 팀원에게 즉시 알림")
    else:
        log["notes"].append("우리 쪽 회수 확인 HEAD 안 함: 전체 판정 미완. 업로드를 끄기 전에 probe --kind revoked --out <이 폴더>로 같은 토큰을 다시 넣어 확인")
    rc = finish(out, log, (token, cookie))
    ledger_mark(ledger, key, pre["run_id"], log["result"])   # the key stays spent; status is informational
    del token, cookie
    return rc


def cmd_probe(args, pre: dict, harness: bool, cafile: str | None) -> int:
    """A deferred post-window check, appended to the run's own log so completion is re-judged in one place."""
    require_tty(harness)
    out = Path(args.out)
    log_path = out / "wrapper-log.json"
    if not log_path.is_file():
        raise Refuse("run_log_missing (--out must be the run folder of the main run)")
    log = json.loads(log_path.read_text())
    if log.get("run_id") != pre["run_id"] or "token_fingerprint" not in log:
        raise Refuse("run_log_mismatch")
    if log.get("target_sha256") != sha256(canonical([pre["target"]["upload_base"], pre["target"]["public_origin"]])):
        raise Refuse("run_target_mismatch")
    # ONEKEY-PROBE-01: game_key / scope / expiry must be the main run's, checked before any token prompt or HEAD.
    if log.get("identity_sha256") is None:
        raise Refuse("run_identity_missing (main run log has no identity binding)")
    if identity_digest(pre, log["token_fingerprint"]["salt"]) != log["identity_sha256"]:
        raise Refuse("run_identity_mismatch (game_key/scope/expiry differ from the main run)")
    if args.kind == "expired" and time.time() < epoch(pre["teammate_confirmation"]["publish_expires_at"]) + EXPIRY_CLOCK_MARGIN_S:
        raise Refuse("not_yet_expired")
    if args.kind in ("expired", "out_of_scope") and pre["teammate_confirmation"]["scope_mode"] != "scoped":
        raise Refuse("probe_needs_scoped_mode")
    if log.get("post_requests", 0) >= POST_BUDGET:
        raise Refuse("post_window_budget_exhausted")
    token = read_token(harness)
    # N16: a re-entered token must have the gateway's bearer shape AND be the very bearer of the main run;
    # a mistyped or different token would also earn 401 and prove nothing.
    if not TOKEN_RE.fullmatch(token):
        raise Refuse("credential_format_invalid (expected [A-Za-z0-9_-]{43,128})")
    fp = log["token_fingerprint"]
    if token_fingerprint(token, fp["salt"]) != fp["sha256"]:
        raise Refuse("token_differs_from_main_run")
    log["post_requests"] = log.get("post_requests", 0) + 1   # N24: count the attempt, answered or not
    row = probe(args.kind, pre, token, cafile) | {"deferred": True, "same_token_as_main_run": True}
    log["probes"].append(row)
    log.setdefault("notes", []).append(f"나중 실행 {args.kind} 확인 추가({row['at_utc']})")
    return finish(out, log, (token,))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "run", "probe"))
    parser.add_argument("--pre", required=True, help="filled PRECONDITIONS.json")
    parser.add_argument("--out", help="new run directory (run) / probe output directory (probe)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kind", choices=("out_of_scope", "expired", "revoked", "upload_off"))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--loopback-harness-cert", help="owned loopback harness only")
    args = parser.parse_args(argv)
    harness = args.loopback_harness_cert is not None
    cafile = args.loopback_harness_cert
    try:
        if not harness:
            if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
                raise Refuse("custom_tls_trust_not_allowed")
            if Path(args.ledger).resolve() != DEFAULT_LEDGER:
                raise Refuse("real_mode_requires_default_ledger")
        raw = Path(args.pre).read_bytes()
        if len(raw) > 65536:
            raise Refuse("preconditions_too_large")
        pre = json.loads(raw)
        problems = validate_pre(pre, harness)
        if problems:
            print(json.dumps({"result": "REFUSED_PRECONDITIONS", "missing": problems, "network_requests": 0},
                             ensure_ascii=False, indent=1))
            return 2
        if args.command == "check":
            print(json.dumps({"result": "PRECONDITIONS_OK", "network_requests": 0,
                              "preconditions_sha256": sha256(raw)}))
            return 0
        if not args.out:
            raise Refuse("--out required")
        if args.command == "run":
            return cmd_run(args, pre, harness, cafile)
        if not args.kind:
            raise Refuse("--kind required")
        return cmd_probe(args, pre, harness, cafile)
    except Refuse as error:
        print(json.dumps({"result": "REFUSED", "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
