// CX39: pinned synthetic bytes through a gateway handler, never an HTTP server.
import assert from "node:assert/strict";
import { readFileSync, lstatSync, realpathSync } from "node:fs";
import { createHash, randomBytes } from "node:crypto";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { syncBuiltinESMExports } from "node:module";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import tls from "node:tls";
import dgram from "node:dgram";
import childProcess from "node:child_process";

assert.equal(process.argv.length, 4, "usage: node test.mjs <pinned-pack> <pinned-host>");
const [pack, host] = process.argv.slice(2).map(x => resolve(x));
process.env = {}; // No inherited operational credentials enter the fixture environment.
const denied = [];
const forbid = name => () => { denied.push(name); throw new Error(`forbidden: ${name}`); };
globalThis.fetch = forbid("fetch");
globalThis.WebSocket = forbid("WebSocket");
for (const [mod, names] of [[http, ["request", "get", "createServer"]], [https, ["request", "get", "createServer"]],
  [net, ["connect", "createConnection", "createServer"]], [tls, ["connect", "createServer"]],
  [dgram, ["createSocket"]], [childProcess, ["spawn", "spawnSync", "exec", "execSync", "execFile", "execFileSync", "fork"]]]) {
  for (const name of names) mod[name] = forbid(name);
}
net.Socket.prototype.connect = forbid("Socket.connect");
net.Server.prototype.listen = forbid("Server.listen");
syncBuiltinESMExports(); // Guards precede every partner module import.
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const pins = {};
function pinned(root, relative, expected, maxBytes = 50000) {
  const file = join(root, relative), stat = lstatSync(file);
  assert.ok(stat.isFile() && !stat.isSymbolicLink() && stat.size <= maxBytes);
  assert.equal(realpathSync(file), file, "no symlinked path");
  const bytes = readFileSync(file);
  assert.equal(digest(bytes), expected, relative);
  pins[relative] = { bytes: bytes.length, sha256: expected };
  return bytes;
}
const manifest = JSON.parse(pinned(pack, "MANIFEST.json", "7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93"));
assert.equal(manifest.source_commit, "6282c63bd699d9cdf5d7110fff2ce530b7221c4f");
const readme = pinned(pack, "README.md", "0cb5492b71866008fd74e1f0add60088960c6cba6d89eac58d5589533acff80c").toString();
assert.ok(readme.includes("six synthetic examples"));
const examplePath = "apps/observer/publish_contract/v1/examples/game-live-unsupported-reference.json";
const schemaPath = "apps/observer/publish_contract/v1/schema/game.schema.json";
const body = pinned(pack, examplePath, "9e34b37959b1221bbbc2a7d55b30a053981a6d73743183b695ed56ba3b7604f2");
const schemaBytes = pinned(pack, schemaPath, "0fc2bc10d723ddc05391b5aa3b9e8674624660b42722b08523bca12ebde35251");
for (const file of [examplePath, schemaPath]) assert.deepEqual(manifest.files[file], pins[file]);
const schema = JSON.parse(schemaBytes), original = JSON.parse(body);
// Narrow static schema checks, not a general JSON Schema validator.
assert.equal(schema.$schema, "https://json-schema.org/draft/2020-12/schema");
assert.equal(schema.properties.next.properties.reference.properties.kind.const, "profile_reference");
assert.ok(!schema.properties.next.required.includes("reference"));
function semantics(value) {
  assert.equal(value.game.game_pk, 900200);
  assert.equal(value.next.rec.status, "unsupported");
  assert.deepEqual(value.next.rec.candidates, []);
  assert.equal(value.next.rec.provenance, "absent");
  assert.equal(value.next.rec.policy, null);
  assert.equal(value.next.rec.recorded_at, null);
  assert.equal(value.next.reference.kind, "profile_reference");
  assert.equal(value.next.reference.reason_code, "unsupported/profile_reference");
  assert.equal(value.next.reference.chip, "참고 · 기록 통계");
  assert.ok(value.next.reference.note.includes("추천이나 다음 공 예측이 아닙니다"));
  assert.equal(value.pitches.length, 10);
  for (const pitch of value.pitches) assert.deepEqual(pitch.rec.pairing, { rule: "no_recorded_recommendation", status: "none" });
}
semantics(original);
const hostPins = {
  "gateway.mjs": "3fd0217beae30ae23ca548c309e2535b66e80024728b10ddb249eec08f2dd2b6",
  "notice.mjs": "d94b5c22ef66b38b3bb824e028fcaa92075856014bbc7e2e28bd3a85bdb64247",
  "access.mjs": "67c3865be7754a9cf172075db67f60a8bc2cd076203799b0bf64a8fe0ef1219a",
  "telemetry.mjs": "c90d0f23dbf80eb965d343bef8fb75129cf3ce7ee9e64fbe6f683228ad0f646a",
  "worker.mjs": "ba6bb83bbbbd8ba4727feb6b870fbe85c0c8c5c9bc65ef837aa2703235f3f985",
};
for (const [file, sha] of Object.entries(hostPins)) {
  const source = pinned(host, file, sha).toString();
  assert.equal(lstatSync(join(host, file)).mode & 0o777, 0o400);
  if (file === "gateway.mjs") assert.deepEqual([...source.matchAll(/^import .* from "([^"\n]+)";/gm)].map(x => x[1]), ["./telemetry.mjs", "./notice.mjs", "./access.mjs"]);
  else if (file !== "worker.mjs") assert.ok(!/^import\b/m.test(source));
}
const { default: gateway } = await import(pathToFileURL(join(host, "gateway.mjs")));
// Minimal R2 CAS model reused from the pinned partner worker.test.mjs pattern.
const key = "v1/game/900200.json";
class Bucket {
  objects = new Map(); writes = 0;
  snapshot(k) {
    assert.ok(k === key || k === "_notice/state.json");
    const row = this.objects.get(k);
    return row ? { ...row, customMetadata: { ...row.customMetadata }, arrayBuffer: async () => row.bytes.slice().buffer } : null;
  }
  async get(k) { return this.snapshot(k); }
  async head(k) { return this.snapshot(k); }
  async put(k, bytes, options) {
    assert.equal(k, key); assert.ok(options.onlyIf instanceof Headers);
    const current = this.objects.get(k), match = options.onlyIf.get("If-Match");
    if (match && (!current || current.httpEtag !== match)) return null;
    if (options.onlyIf.get("If-None-Match") === "*" && current) return null;
    assert.equal(digest(bytes), options.sha256);
    this.objects.set(k, { bytes: bytes.slice(), size: bytes.length, customMetadata: { ...options.customMetadata },
      httpEtag: '"' + createHash("md5").update(bytes).digest("hex") + '"' });
    this.writes++; return this.snapshot(k);
  }
}
const token = randomBytes(32).toString("base64url"), bucket = new Bucket();
let limiterCalls = 0;
const env = { DATA_READ_ENABLED: "true", UPLOAD_ENABLED: "true", PUBLISH_BUCKET: bucket,
  PUBLISH_TOKEN_SHA256: digest(token), PUBLISH_KEY_SCOPE: key,
  PUBLISH_EXPIRES_AT: new Date(Date.now() + 60000).toISOString(),
  PUBLISH_LIMITER: { limit: async ({ key: limiterKey }) => { assert.equal(limiterKey, "publisher"); limiterCalls++; return { success: true }; } } };
const requests = [];
async function call(method, privateRoute, status, condition = {}) {
  assert.ok(requests.length < 8);
  const route = (privateRoute ? "/_publish/" : "/") + key;
  const headers = privateRoute ? { Authorization: `Bearer ${token}`, ...condition } : {};
  if (method === "PUT") Object.assign(headers, { "Content-Type": "application/json", "X-Content-SHA256": digest(body) });
  const row = { method, route, expected_status: status }; requests.push(row);
  const response = await gateway.fetch(new Request("https://synthetic.example.test" + route,
    { method, headers, ...(method === "PUT" ? { body } : {}) }), env);
  row.status = response.status; row.etag = response.headers.get("ETag");
  row.body = new Uint8Array(await response.arrayBuffer()); row.body_sha256 = digest(row.body);
  assert.equal(row.status, status);
  if (status === 200 && row.etag) assert.match(row.etag, /^"[a-f0-9]{32}"$/);
  return { row, headers: response.headers };
}
function payload(row, etag) {
  assert.equal(row.etag, etag); assert.equal(row.body_sha256, digest(body));
  assert.deepEqual(Buffer.from(row.body), body);
  const value = JSON.parse(Buffer.from(row.body)); semantics(value); assert.deepEqual(value, original);
}
await call("HEAD", true, 404);
const created = (await call("PUT", true, 200, { "If-None-Match": "*" })).row;
assert.equal(JSON.parse(Buffer.from(created.body)).schema_validated, false);
payload((await call("GET", false, 200)).row, created.etag);
const withdrawn = (await call("DELETE", true, 200, { "If-Match": created.etag })).row;
assert.notEqual(withdrawn.etag, created.etag);
assert.equal(JSON.parse(Buffer.from(withdrawn.body)).deleted, true);
assert.deepEqual(JSON.parse(Buffer.from((await call("GET", false, 404)).row.body)), { error: "not_found" });
const tombstone = await call("HEAD", true, 200);
assert.equal(tombstone.row.etag, withdrawn.etag); assert.equal(tombstone.headers.get("X-Publish-Deleted"), "true");
const restored = (await call("PUT", true, 200, { "If-Match": withdrawn.etag })).row;
assert.notEqual(restored.etag, created.etag); assert.notEqual(restored.etag, withdrawn.etag);
assert.equal(JSON.parse(Buffer.from(restored.body)).deleted, false);
payload((await call("GET", false, 200)).row, restored.etag);
assert.deepEqual(requests.map(x => x.method), ["HEAD", "PUT", "GET", "DELETE", "GET", "HEAD", "PUT", "GET"]);
assert.equal(bucket.writes, 3); assert.equal(limiterCalls, 5); assert.deepEqual(denied, []);
// Re-read only the exact whitelist to show that input/source pins stayed unchanged.
for (const [file, pin] of Object.entries(pins)) pinned(Object.hasOwn(hostPins, file) ? host : pack, file, pin.sha256);
console.log(JSON.stringify({ result: "PASS_SYNTHETIC_GATEWAY_HANDLER", cases: 1,
  mode: "direct Request/Response with in-memory Bucket and mock limiter; no HTTP transport",
  node: process.version, pins, request_count: requests.length, limiter_calls: limiterCalls, storage_writes: bucket.writes,
  requests: requests.map(({ body: ignored, ...row }) => row), denied_attempts: denied,
  network_requests: 0, body_sha256: digest(body), body_bytes: body.length,
  schema_validation: "static reference definition checks only; no full validator",
  viewer_gate: "disabled synthetic fixture; no session/revocation test", freshness_evaluated: false }, null, 2));
