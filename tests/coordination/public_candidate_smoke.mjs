// Explicit pinned public candidate + invented JSON -> owned loopback HTTP -> Chromium.
// No publisher/model/real-game fixture import, build, installation, or deployment.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { lstatSync, readFileSync, writeFileSync, mkdirSync, rmSync, symlinkSync } from 'node:fs';
import { dirname, extname, join, parse, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const [manifestArg, expectedManifest, expectedSource, buildArg, playwrightArg, outArg] = process.argv.slice(2);
assert.equal(process.argv.length, 8, 'manifest manifestSHA sourceSHA build playwright/index.mjs NEW_output required');
const sha = b => createHash('sha256').update(b).digest('hex');
const hex = (s, n) => typeof s === 'string' && new RegExp(`^[0-9a-f]{${n}}$`).test(s);
assert.ok(hex(expectedManifest, 64) && hex(expectedSource, 40));
function noLinks(path) {
  const absolute = resolve(path), root = parse(absolute).root;
  let at = root;
  for (const part of absolute.slice(root.length).split(sep).filter(Boolean)) {
    at = join(at, part);
    assert.ok(!lstatSync(at).isSymbolicLink(), `symlink forbidden: ${at}`);
  }
  return absolute;
}
function bytes(path, max) {
  noLinks(path);
  const s = lstatSync(path);
  assert.ok(s.isFile() && s.size > 0 && s.size <= max, `bounded regular file required: ${path}`);
  return readFileSync(path);
}
function pinnedAssets(manifestPath, manifestSHA, sourceSHA, buildRoot) {
  const raw = bytes(manifestPath, 128 * 1024);
  assert.equal(sha(raw), manifestSHA, 'manifest SHA mismatch');
  const manifest = JSON.parse(raw);
  assert.equal(manifest.source_commit, sourceSHA, 'source pin mismatch');
  const members = manifest.public_build_members, names = Object.keys(members ?? {}).sort();
  assert.equal(names.length, 4, 'exactly four public assets required');
  assert.ok(names.includes('index.html') && names.includes('favicon.svg'));
  assert.equal(names.filter(n => /^assets\/index-[A-Za-z0-9_-]+\.js$/.test(n)).length, 1);
  assert.equal(names.filter(n => /^assets\/index-[A-Za-z0-9_-]+\.css$/.test(n)).length, 1);
  const allowed = new Map(), hashes = {};
  const root = noLinks(buildRoot);
  for (const name of names) {
    assert.ok(name === 'index.html' || name === 'favicon.svg' || /^assets\/index-[A-Za-z0-9_-]+\.(js|css)$/.test(name), 'asset path outside allowlist');
    assert.deepEqual(manifest.files?.[`public-build/${name}`], members[name], 'manifest asset entries disagree');
    const file = resolve(root, name);
    assert.ok(file.startsWith(root + sep), 'asset path escape');
    const data = bytes(file, 1024 * 1024);
    assert.equal(data.length, members[name].bytes, `asset length: ${name}`);
    assert.equal(sha(data), members[name].sha256, `asset SHA: ${name}`);
    allowed.set('/' + name, data); hashes[name] = sha(data);
  }
  allowed.set('/', allowed.get('/index.html'));
  return { allowed, hashes };
}

// Refusal controls use only newly generated tiny files under this run's output.
function pinControls(out) {
  const dir = join(out, 'pin-controls'); mkdirSync(dir); mkdirSync(join(dir, 'assets'));
  const names = ['index.html', 'favicon.svg', 'assets/index-test.js', 'assets/index-test.css'];
  const members = Object.fromEntries(names.map(n => {
    const data = Buffer.from(`synthetic ${n}`); writeFileSync(join(dir, n), data, { flag: 'wx' });
    return [n, { bytes: data.length, sha256: sha(data) }];
  }));
  const manifest = { source_commit: '0'.repeat(40), public_build_members: members,
    files: Object.fromEntries(names.map(n => [`public-build/${n}`, members[n]])) };
  const path = join(dir, 'manifest.json'), raw = Buffer.from(JSON.stringify(manifest));
  writeFileSync(path, raw, { flag: 'wx' });
  const check = () => pinnedAssets(path, sha(raw), manifest.source_commit, dir);
  check();
  writeFileSync(path, Buffer.concat([raw, Buffer.from(' ')]));
  assert.throws(check, /manifest SHA mismatch/); writeFileSync(path, raw);
  writeFileSync(join(dir, 'index.html'), 'changed');
  assert.throws(check, /asset length|asset SHA/); writeFileSync(join(dir, 'index.html'), 'synthetic index.html');
  const js = join(dir, 'assets/index-test.js'); rmSync(js); symlinkSync('../index.html', js);
  assert.throws(check, /symlink forbidden/); rmSync(js); writeFileSync(js, 'synthetic assets/index-test.js');
  rmSync(js); assert.throws(check, /ENOENT/); writeFileSync(js, 'synthetic assets/index-test.js');
  const bad = { ...manifest, public_build_members: { ...members, '../escape.js': members['index.html'] } };
  const badRaw = Buffer.from(JSON.stringify(bad)); writeFileSync(path, badRaw);
  assert.throws(() => pinnedAssets(path, sha(badRaw), manifest.source_commit, dir), /exactly four/);
  rmSync(dir, { recursive: true });
  return ['valid_control', 'manifest_tamper_refused', 'asset_mismatch_refused', 'symlink_refused', 'missing_asset_refused', 'extra_escape_member_refused'];
}

// Source-shaped, entirely invented final game: no pitch/player/model rows are read.
function syntheticFiles() {
  const date = '2030-01-01', game = 990001, at = '2030-01-01T00:00:00.000Z';
  const publish = { contract: 'v1', sequence: 1, published_at: at, final: true };
  const source = { kind: 'archive', session_id: null };
  const card = { game_pk: game, date_kst: date, official_date: date, start: at, start_kst: '2030-01-01T09:00:00+09:00',
    away_team: 'Synthetic Away', home_team: 'Synthetic Home', away_abbr: 'SYN-A', home_abbr: 'SYN-H',
    away_score: 0, home_score: 0, status: 'final', status_text: 'Final', live: false, inning: null, half: null,
    game_number: 1, double_header: false, double_header_code: 'N', embargoed: false, has_history: false, venue: null };
  const detail = { schema: 'pitcheezy-service-game-v2', source, game: card, publish,
    current: { phase: 'final', batter: null, pitcher: null, situation: null, pa_key: null, context: null },
    cutoff: { index: -1, pitch_key: null }, feed: { as_of: at, ok: true, stale_s: 0 },
    linescore: null, next: null, pas: [], pitches: [], pregame: null, zone_bounds: { bottom: 1.5, top: 3.5 } };
  const bodies = {
    '/v1/manifest.json': { schema: 'pitcheezy-publish-manifest-v1', contract: 'v1', publish, published_at: at,
      commit: null, commit_dirty: null, publish_interval_s: 60, fixed_delay_s: 0, live_dates: [], provisional: ['synthetic-only'],
      files: { manifest: 'v1/manifest.json', dates: 'v1/dates.json', health: 'v1/health.json',
        games: 'v1/games/{date}.json', game: 'v1/game/{game_pk}.json', setup: 'none' } },
    '/v1/dates.json': { schema: 'pitcheezy-service-dates-v2', source, dates: [date], publish },
    [`/v1/games/${date}.json`]: { schema: 'pitcheezy-service-games-v2', source, date, games: [card], live_ok: true, publish },
    [`/v1/game/${game}.json`]: detail,
    '/v1/health.json': { schema: 'pitcheezy-publish-health-v1', publish, sequence: 1, published_at: at,
      publish_interval_s: 60, files_written: 5, unavailable_games: [], unavailable_dates: [], feed_failing_games: [] },
  };
  return { date, game, bodies, files: new Map(Object.entries(bodies).map(([p, b]) => [p, Buffer.from(JSON.stringify(b))])) };
}

// Pin validation happens before output creation, browser import, or any listening socket.
const assets = pinnedAssets(manifestArg, expectedManifest, expectedSource, buildArg);
const modulePath = noLinks(playwrightArg);
assert.equal(modulePath.split(sep).slice(-2).join('/'), 'playwright/index.mjs');
const output = resolve(outArg); noLinks(dirname(output)); mkdirSync(output, { mode: 0o700 });
const controls = pinControls(output), fixture = syntheticFiles();
writeFileSync(join(output, 'synthetic-files.json'), JSON.stringify(fixture.bodies, null, 2), { flag: 'wx' });
const allowed = new Map([...assets.allowed, ...fixture.files]);
const receipt = { schema: 'pitcheezy-current-public-candidate-smoke-v1', status: 'INCOMPLETE',
  source: expectedSource, manifestSHA: expectedManifest, buildHashes: assets.hashes,
  fixtureHashes: Object.fromEntries([...fixture.files].map(([p, b]) => [p, sha(b)])), controls,
  transport: 'owned_loopback_HTTP_not_TLS', requests: [], blocked: [], scenarios: [],
  requestCap: 60, executionBudgetMs: 80000, totalBudgetMs: 90000 };
let browser, browserServer, origin, fault = false, count = 0, routed = 0, timedOut = false;
let browserLaunchStarted = false, browserLaunchSettled = false;
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.svg': 'image/svg+xml', '.json': 'application/json' };
let rejectBudget;
const budget = new Promise((_, reject) => { rejectBudget = reject; });
const server = createServer((req, res) => {
  count++;
  if (count > 60) { res.writeHead(429); res.end(); rejectBudget(new Error('request cap exceeded')); return; }
  const url = new URL(req.url, origin);
  const permitted = req.method === 'GET' && req.headers.host === new URL(origin).host && allowed.has(url.pathname)
    && (url.pathname === '/' || !url.search);
  if (!permitted) { receipt.blocked.push('server:' + url.pathname); res.writeHead(403); res.end(); return; }
  const failed = fault && (url.pathname === `/v1/game/${fixture.game}.json` || url.pathname === '/v1/health.json');
  const body = failed ? Buffer.from('{"error":"synthetic_local_503"}') : allowed.get(url.pathname);
  const status = failed ? 503 : 200;
  receipt.requests.push({ path: url.pathname, method: req.method, status, sha256: sha(body), bytes: body.length });
  res.writeHead(status, { 'Content-Type': types[extname(url.pathname)] ?? types['.html'],
    'Cache-Control': 'no-store', 'Content-Security-Policy': "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; object-src 'none'; worker-src 'none'" });
  res.end(body);
});
const started = Date.now();
const deadline = setTimeout(() => { timedOut = true; rejectBudget(new Error('execution budget exceeded')); }, 80000);
async function bounded(promise, ms) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('owned resource cleanup timeout')), ms);
  })]); } finally { clearTimeout(timer); }
}
async function scenario(width, failing) {
  fault = failing;
  const context = await browser.newContext({ viewport: { width, height: 900 }, serviceWorkers: 'block' });
  try {
    await context.route('**/*', route => {
      const request = route.request(), url = new URL(request.url());
      if (url.origin !== origin || request.method() !== 'GET' || !allowed.has(url.pathname)) {
        receipt.blocked.push('browser:' + url.origin + url.pathname); return route.abort();
      }
      if (++routed > 60) { receipt.blocked.push('browser:request_cap'); rejectBudget(new Error('request cap exceeded')); return route.abort(); }
      return route.continue();
    });
    await context.routeWebSocket('**/*', ws => { receipt.blocked.push('websocket'); ws.close(); });
    const page = await context.newPage(), errors = [], begin = receipt.requests.length;
    page.setDefaultTimeout(10000); page.setDefaultNavigationTimeout(15000);
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto(`${origin}/?${failing ? `game=${fixture.game}` : `date=${fixture.date}`}`, { waitUntil: 'domcontentloaded' });
    if (failing) {
      await page.locator('.sv-alert[role=alert]').waitFor();
      assert.equal(await page.locator('.sv-rec-row').count(), 0);
    } else {
      const link = page.locator(`.sv-list .sv-card[href="./?game=${fixture.game}"]`);
      await link.waitFor(); await link.click();
      await page.locator('main.sv-live-layout').waitFor();
      assert.match(await page.locator('.sv-live').innerText(), /종료/);
      assert.equal(await page.locator('.sv-alert').count(), 0);
    }
    assert.deepEqual(errors, []); assert.deepEqual(receipt.blocked, []);
    const served = receipt.requests.slice(begin), gamePath = `/v1/game/${fixture.game}.json`;
    assert.ok(served.some(r => r.path === gamePath && r.status === (failing ? 503 : 200)));
    if (!failing) {
      for (const path of ['/v1/manifest.json', `/v1/games/${fixture.date}.json`, gamePath])
        assert.ok(served.some(r => r.path === path && r.sha256 === receipt.fixtureHashes[path]));
      assert.ok(served.every(r => r.status === 200));
    } else assert.ok(served.some(r => r.path === '/v1/health.json' && r.status === 503));
    await page.screenshot({ path: join(output, `${failing ? '503' : 'final'}-${width}.png`), fullPage: true });
    receipt.scenarios.push({ kind: failing ? '503_no_recommendation' : 'list_to_final', width, height: 900,
      requestCount: served.length, pageErrors: errors, fixtureGame: fixture.game });
  } finally { await bounded(context.close(), 3000); }
}
try {
  await Promise.race([(async () => {
    const { chromium } = await import(pathToFileURL(modulePath).href);
    assert.ok(!timedOut, 'budget expired before server start');
    await new Promise((ok, fail) => server.once('error', fail).listen(0, '127.0.0.1', ok));
    const port = server.address().port;
    assert.ok(port < 8766 || port > 8772, 'protected port forbidden');
    origin = `http://127.0.0.1:${port}`;
    assert.ok(!timedOut, 'budget expired before browser start');
    assert.ok(Date.now() - started < 60000, 'insufficient launch/cleanup margin');
    browserLaunchStarted = true;
    try {
      browserServer = await chromium.launchServer({ headless: true, host: '127.0.0.1', port: 0, timeout: 15000,
        args: ['--disable-background-networking', '--disable-component-update',
          '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1'],
        env: { HOME: process.env.HOME, PATH: process.env.PATH } });
    } finally { browserLaunchSettled = true; }
    if (timedOut) { await bounded(browserServer.kill(), 2000); throw new Error('browser completed after budget'); }
    const driverPort = Number(new URL(browserServer.wsEndpoint()).port);
    assert.ok(driverPort < 8766 || driverPort > 8772, 'protected driver port forbidden');
    browser = await chromium.connect(browserServer.wsEndpoint(), { timeout: 10000 });
    await scenario(390, false); await scenario(1440, false); await scenario(390, true);
  })(), budget]);
  assert.ok(count <= 60 && !timedOut); assert.deepEqual(receipt.blocked, []);
  receipt.status = 'PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP';
} catch (error) {
  receipt.error = String(error); process.exitCode = 1;
} finally {
  clearTimeout(deadline);
  if (browserServer) {
    try { await bounded(browserServer.close(), 5000); }
    catch (error) {
      receipt.cleanupError = String(error); process.exitCode = 1;
      // Only the BrowserServer handle created above; never process discovery or an unrelated PID.
      await bounded(browserServer.kill(), 2000).catch(e => { receipt.cleanupKillError = String(e); });
    }
  }
  server.closeAllConnections();
  if (server.listening) await bounded(new Promise(done => server.close(done)), 1000).catch(e => { receipt.cleanupError = String(e); });
  const ownedProcess = browserServer?.process();
  receipt.browserLaunchStarted = browserLaunchStarted; receipt.browserLaunchSettled = browserLaunchSettled;
  receipt.ownedBrowserClosed = !browserLaunchStarted || Boolean(browserLaunchSettled && ownedProcess
    && (ownedProcess.exitCode !== null || ownedProcess.signalCode !== null));
  receipt.ownedServerClosed = !server.listening;
  receipt.routedAttempts = routed; receipt.totalCalls = count; receipt.elapsedMs = Date.now() - started;
  if (!receipt.ownedBrowserClosed || !receipt.ownedServerClosed || receipt.elapsedMs > 90000 || receipt.cleanupError) {
    receipt.status = 'FAIL_CLEANUP_OR_BUDGET'; process.exitCode = 1;
  }
  writeFileSync(join(output, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ status: receipt.status, scenarios: receipt.scenarios.length,
    calls: count, ownedBrowserClosed: receipt.ownedBrowserClosed, ownedServerClosed: receipt.ownedServerClosed }));
}
