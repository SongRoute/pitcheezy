// UI-DAY preview server (2026-10-21, Song's UI day). Shows the PUBLIC screen on published JSON files without the publisher running.
// It builds nothing and never writes: it serves an already built dist-public/ plus one fixture folder (the `v1/` layout of the public
// publish contract), with a chosen server clock in the `Date` header (the screen reads "now" only from that header, never the device).
//
//   npm run build:public && node tests/ui-preview.mjs                 # loopback, random port, committed fixture, fresh clock
//   node tests/ui-preview.mjs --phone                                  # Tailscale address of this Mac, port 8780 (open on the phone)
//   node tests/ui-preview.mjs --fixture <dir> --clock fixture          # <dir> holds v1/ (or is v1/ itself)
//   node tests/ui-preview.mjs --ws                                     # World Series replay (fixtures/publish-ws: G1 849805 + G3 849804 + G4 849803), fixture clock
//   node tests/ui-preview.mjs --ws-moment first-rec [--ws]            # one replayed MID-GAME moment of G4 (fixtures/publish-ws-moments/849803/<label>), clock now
//   node tests/ui-preview.mjs --dev                                    # hot reload: also starts `vite --mode public` reading this server
//
// --fixture can be given more than once: the folders are layered (a game or games file comes from the first folder that has it) and
// the three index files are joined when answered (tests/publish-join.mjs: dates.json `dates`, manifest.json `live_dates`, health.json
// `game_checks` and its lists; every other field from the folder whose file was published last). One folder is served exactly as
// before. tests/publish-merge.mjs writes the same join into one new folder for a static host.
// --ws             the three World Series publish folders copied byte for byte from the live runs of 10/24, 10/27 and 10/28 KST (see
//                  fixtures/publish-ws/README.md); --clock defaults to fixture with it. All games are final: no `next`, no live state.
// --ws-moment      a moment folder of fixtures/publish-ws-moments/849803 (README there: REPLAYED bodies from the live run's store, the
//                  recommendation being the one the live ledger recorded; not live-published bytes). Served first, so with --ws its game
//                  and date files win and the game shows mid-game with `next`; --clock defaults to now with it.
//
// --clock now      (default) real Date header; the publish stamps of each answer (publish.published_at/sequence, health/manifest
//                  published_at/sequence, health game_checks checked_at/sequence_*) are moved to "now" and the sequence steps once per
//                  publish interval, so the screen behaves as if the publisher were running and never shows "갱신 지연". Content is unchanged.
// --clock fixture  Date header frozen 5 s after the fixture's newest published_at (same as tests/ui-r1-shots.mjs). The screen keeps its
//                  own clock running from the first answer, so a live game turns "갱신 지연" about 30 s after the page opens. For screenshots.
// --clock raw      real Date header, files untouched: an old fixture shows the late / stale states at once.
//
// Ports 8766-8772 belong to the running services and are refused. Default host is 127.0.0.1; --phone binds only the Tailscale address.
// --exit-after <s> stops by itself (used for a quick self-test). Files are never modified on disk.
// Day guide (outside the repo): /Users/song/Projects/pitcheezy-pm-jobs/20261021-ui-day/UI-DAY-KIT.md
import { spawn, execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { INDEX as INDEX_FILES, joinIndex } from './publish-join.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.join(here, '..');
const USAGE = 'usage: node tests/ui-preview.mjs [--fixture <dir>]... [--ws] [--ws-moment <label>]... [--dist <dir>] [--clock now|fixture|raw] [--host <ip>] [--port <n>] [--phone] [--dev] [--dev-port <n>] [--exit-after <s>]';
const WS = ['2026-10-24', '2026-10-27', '2026-10-28'].map((d) => path.join(here, 'fixtures', 'publish-ws', d));   // G1 849805, G3 849804, G4 849803
const MOMENTS = path.join(here, 'fixtures', 'publish-ws-moments', '849803');   // replayed mid-game bodies of G4 (README there): not live-published bytes

const args = { fixtures: [], ws: false, moments: [], dist: process.env.PZ_WEB_DIST_PUBLIC ?? path.join(web, 'dist-public'),
  clock: null, host: null, port: null, phone: false, dev: false, devPort: null, exitAfter: null };
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i], next = () => { const v = process.argv[++i]; if (v === undefined) die(`${a} needs a value`); return v; };
  if (a === '--fixture') args.fixtures.push(path.resolve(next()));
  else if (a === '--ws') args.ws = true;
  else if (a === '--ws-moment') args.moments.push(next());
  else if (a === '--dist') args.dist = path.resolve(next());
  else if (a === '--clock') args.clock = next();
  else if (a === '--host') args.host = next();
  else if (a === '--port') args.port = Number(next());
  else if (a === '--phone') args.phone = true;
  else if (a === '--dev') args.dev = true;
  else if (a === '--dev-port') args.devPort = Number(next());
  else if (a === '--exit-after') args.exitAfter = Number(next());
  else if (a === '-h' || a === '--help') { console.log(USAGE); process.exit(0); }
  else die(`unknown option ${a}\n${USAGE}`);
}
function die(msg) { console.error(msg); process.exit(2); }
for (const label of args.moments) {   // a moment goes FIRST: its game and date files win over the final ones of --ws
  if (!/^[a-z0-9-]+$/.test(label) || !existsSync(path.join(MOMENTS, label, 'v1', 'manifest.json'))) die(`no such moment: ${label} (see ${path.join(MOMENTS, 'README.md')})`);
  args.fixtures.push(path.join(MOMENTS, label));
}
if (args.ws) args.fixtures.push(...WS);
if (!args.fixtures.length) args.fixtures.push(path.join(here, 'fixtures', 'publish'));
args.clock ??= args.moments.length ? 'now' : args.ws ? 'fixture' : 'now';   // a moment is a live body: with the fixture clock it would read 갱신 지연 after 30 s
if (!['now', 'fixture', 'raw'].includes(args.clock)) die('--clock must be now, fixture or raw');

// ---- where ----
const RESERVED = (p) => p >= 8766 && p <= 8772;
if (args.phone && !args.host) {
  let ip = '';
  for (const bin of ['tailscale', '/Applications/Tailscale.app/Contents/MacOS/Tailscale']) {
    try { ip = execFileSync(bin, ['ip', '-4'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim().split('\n')[0]; if (ip) break; } catch { /* next */ }
  }
  if (!/^\d+\.\d+\.\d+\.\d+$/.test(ip)) die('Tailscale address not found. Pass it yourself: --phone --host <100.x.y.z>');
  args.host = ip;
}
args.host ??= '127.0.0.1';
args.port ??= args.phone ? 8780 : 0;
if (!Number.isInteger(args.port) || args.port < 0 || RESERVED(args.port)) die(`port ${args.port} refused (8766-8772 are the running services)`);
if (args.devPort !== null && (!Number.isInteger(args.devPort) || RESERVED(args.devPort))) die(`dev port ${args.devPort} refused`);

// ---- what ----
const v1s = args.fixtures.map((dir) => existsSync(path.join(dir, 'v1', 'manifest.json')) ? path.join(dir, 'v1')
  : existsSync(path.join(dir, 'manifest.json')) ? dir : die(`no v1/manifest.json under ${dir}`));
if (!args.dev && !existsSync(path.join(args.dist, 'index.html'))) die(`no ${path.join(args.dist, 'index.html')} — run: npm run build:public  (or use --dev)`);
const readJson = (p) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return null; } };
const manifest = readJson(path.join(v1s[0], 'manifest.json')) ?? {};
const fixtureAt = Math.max(...v1s.flatMap((d) => ['manifest.json', 'health.json'].map((f) => readJson(path.join(d, f)) ?? {}))
  .flatMap((j) => [j.published_at, j.publish?.published_at]).map((s) => Date.parse(s ?? '')).filter(Number.isFinite), 0);
const intervalS = Number(manifest.publish_interval_s) > 0 ? Number(manifest.publish_interval_s) : 10;
const names = (sub) => [...new Set(v1s.flatMap((d) => existsSync(path.join(d, sub)) ? readdirSync(path.join(d, sub)).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)) : []))];
const games = names('game'), dates = names('games');

// several folders: the index files are joined per answer (never on disk) by tests/publish-join.mjs, the rule publish-merge.mjs writes
const INDEX = INDEX_FILES.map((f) => `/${f}`);
const joined = (name) => joinIndex(name.slice(1), v1s.map((d) => readJson(path.join(d, name))));   // fresh parses: nothing on disk changes

// ---- clock ----
const started = Date.now();
const steps = () => Math.floor((Date.now() - started) / (intervalS * 1000));
const isoNow = () => new Date(Date.now() - 1000).toISOString().replace(/\.\d{3}Z$/, 'Z');   // 1 s back: never "ahead" of the Date header
function freshen(body) {
  const at = isoNow(), add = steps();
  const seq = (o, k) => { if (o && Number.isSafeInteger(o[k])) o[k] += add; };
  const stamp = (o) => { if (!o || typeof o !== 'object') return; if (typeof o.published_at === 'string') o.published_at = at; seq(o, 'sequence'); };
  stamp(body); stamp(body.publish);                      // top level only exists on health/manifest; publish{} on every file
  if (typeof body.publish_seq === 'number') body.publish_seq += add;
  for (const key of ['game_checks']) {
    const checks = body.schema?.includes('health') ? body[key] : null;
    if (checks && typeof checks === 'object' && !Array.isArray(checks)) for (const c of Object.values(checks)) {
      if (!c || typeof c !== 'object') continue;
      if (typeof c.checked_at === 'string') c.checked_at = at;
      seq(c, 'sequence_checked'); seq(c, 'file_sequence');
    }
  }
  return body;
}
const dateHeader = () => new Date(args.clock === 'fixture' ? fixtureAt + 5000 : Date.now()).toUTCString();

// ---- server ----
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json',
  '.png': 'image/png', '.woff2': 'font/woff2', '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json' };
const inside = (root, p) => { const r = path.resolve(root, '.' + p); return r === root || r.startsWith(root + path.sep) ? r : null; };
const server = http.createServer((req, res) => {
  res.setHeader('Date', dateHeader());
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Access-Control-Allow-Origin', '*');                      // --dev: the vite page reads us cross-origin
  res.setHeader('Access-Control-Expose-Headers', 'Date');                  // ...and must be able to read the clock
  res.setHeader('Access-Control-Allow-Headers', 'If-None-Match');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405); return res.end(); }
  let rel; try { rel = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch { res.writeHead(400); return res.end(); }
  rel = rel.replace(/^\/pub(?=\/)/, '');
  const m = rel.match(/^(?:.*)?\/v1(\/.*)$/);
  if (m) {
    const body = v1s.length > 1 && INDEX.includes(m[1]) ? joined(m[1]) : null;
    const disk = body ? null : v1s.map((d) => inside(d, m[1])).find((p) => p && existsSync(p) && statSync(p).isFile());
    if (!body && !disk) { res.writeHead(404, { 'content-type': 'text/plain' }); return res.end('not found'); }
    let data = body ? Buffer.from(JSON.stringify(body)) : readFileSync(disk);
    if (args.clock === 'now' && (body || disk.endsWith('.json'))) { const j = body ?? readJson(disk); if (j) data = Buffer.from(JSON.stringify(freshen(j))); }
    res.writeHead(200, { 'content-type': body ? TYPES['.json'] : TYPES[path.extname(disk)] ?? 'application/octet-stream' }); return res.end(req.method === 'HEAD' ? undefined : data);
  }
  if (args.dev) { res.writeHead(404, { 'content-type': 'text/plain' }); return res.end('files only (--dev): open the vite address'); }
  const file = inside(args.dist, rel);
  const target = file && existsSync(file) && statSync(file).isFile() ? file : path.join(args.dist, 'index.html');
  res.writeHead(200, { 'content-type': TYPES[path.extname(target)] ?? 'application/octet-stream' });
  res.end(req.method === 'HEAD' ? undefined : readFileSync(target));
});
server.on('error', (e) => die(`cannot listen on ${args.host}:${args.port}: ${e.message}`));
await new Promise((r) => server.listen(args.port, args.host, r));
const port = server.address().port, origin = `http://${args.host}:${port}`;

let child = null, page = `${origin}/pub/`;
if (args.dev) {
  const devPort = args.devPort ?? (args.phone ? 8781 : 5173);
  const vite = path.join(web, 'node_modules', '.bin', 'vite');
  child = spawn(vite, ['--mode', 'public', '--host', args.host, '--port', String(devPort), '--strictPort'],
    { cwd: web, stdio: 'inherit', env: { ...process.env, VITE_PUBLISH_BASE: origin } });
  child.on('exit', (code) => { console.log(`vite ended (${code})`); finish(); });
  page = `http://${args.host}:${devPort}/`;
}
console.log(`\nfixture  ${v1s.join('\n         ')}${v1s.length > 1 ? '\n         (layered; dates/manifest/health joined per answer)' : ''}`);
console.log(`clock    ${args.clock}${args.clock === 'fixture' ? ` (Date header ${new Date(fixtureAt + 5000).toISOString()})` : ''}; publish interval ${intervalS}s`);
console.log(`files    ${origin}/v1/   (${games.length} game files, dates: ${dates.join(', ') || '-'})`);
console.log(`\nopen     ${page}`);
for (const g of games) console.log(`         ${page}?game=${g}`);
for (const d of dates) console.log(`         ${page}?date=${d}`);
console.log('\nstop with Ctrl-C');

let done = false;
function finish() { if (done) return; done = true; if (child && child.exitCode === null) child.kill('SIGTERM'); server.closeAllConnections(); server.close(() => process.exit(0)); }
process.on('SIGINT', finish); process.on('SIGTERM', finish);
if (args.exitAfter) setTimeout(finish, args.exitAfter * 1000);
