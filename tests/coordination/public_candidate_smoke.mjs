// Explicit pinned public candidate + invented JSON -> owned loopback HTTP -> Chromium.
// No publisher/model/real-game fixture import, build, installation, or deployment.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { lstatSync, readFileSync, writeFileSync, mkdirSync, rmSync, symlinkSync } from 'node:fs';
import { dirname, extname, join, parse, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const [manifestArg, expectedManifest, expectedSource, buildArg, playwrightArg, outArg, suiteArg] = process.argv.slice(2);
assert.ok(process.argv.length === 8 || (process.argv.length === 9 && ['--host-errors', '--live-states', '--health-freshness', '--withdrawal-recovery', '--partial-body'].includes(suiteArg)),
  'manifest manifestSHA sourceSHA build playwright/index.mjs NEW_output [--host-errors|--live-states|--health-freshness|--withdrawal-recovery|--partial-body] required');
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

// Invented source-shaped live game; no captured rows or publisher execution.
function liveFixtures(base) {
  const at = '2030-01-01T00:00:00.000Z', gamePath = `/v1/game/${base.game}.json`;
  return ['transition', 'unconfirmed', 'missing_display', 'unknown_display'].map(name => {
    const bodies = structuredClone(base.bodies), game = bodies[gamePath];
    game.source.kind = 'live'; game.publish.final = false;
    Object.assign(game.game, { status: 'live', status_text: 'In Progress', live: true, inning: 1, half: 'Top' });
    game.current = { phase: 'pre_pitch', pa_key: `${base.game}:1`, context: null,
      batter: { id: 990101, name: 'Synthetic Batter', side: 'R', jersey: null },
      pitcher: { id: 990102, name: 'Synthetic Pitcher', hand: 'R', jersey: null },
      situation: { inning: 1, half: 'Top', outs: 0, balls: 0, strikes: 0, bases: 0,
        away_score: 0, home_score: 0, runners: { first: false, second: false, third: false } } };
    game.next = { pitch_key: `${base.game}:1:1`, rec: { status: 'ready', reason: null,
      provenance: 'captured_live', policy: null, recorded_at: at,
      candidates: [{ rank: 1, pitch_type: 'FF', pitch_label: '포심 패스트볼', zone_id: 'middle', zone_label: '가운데',
        target: { x: 0, z: 2.25 }, detail: { probability: 1 } }] } };
    game.display_state = { state: name === 'unconfirmed' ? 'unconfirmed' : name === 'unknown_display' ? 'synthetic_future_state' : 'recommended',
      reason: 'synthetic_only', pitch_key: game.next.pitch_key, recommendation_shown: true,
      schema: 'synthetic-display-state-v1', label: '', message: null, detail: null,
      ledger: { status: 'synthetic_only', store_ok: true, outcome: null, why: null },
      feed: { ok: true, delayed: false, lag_s: null, failing_for_s: null, failing_since: null,
        lag_max_s: null, lag_measured: false, lag_threshold_s: 0, last_ok_at: at,
        provider_order: 'synthetic_only', received_at: at, thresholds_provisional: true }, correction: null };
    if (name === 'missing_display') delete game.display_state;
    bodies['/v1/games/2030-01-01.json'].games = [{ ...game.game, inning: null, half: null }];
    bodies['/v1/manifest.json'].live_dates = [base.date];
    for (const body of Object.values(bodies)) body.publish.final = false;
    const phases = (name === 'transition' ? [true, false, true] : [true]).map((feedOk, i) => {
      const phaseBodies = structuredClone(bodies), health = phaseBodies['/v1/health.json'];
      health.sequence = i + 1; health.publish = { ...health.publish, sequence: i + 1 };
      health.game_checks = { [base.game]: { checked_at: at, sequence_checked: i + 1, file_sequence: 1, feed_ok: feedOk } };
      return { name: name === 'transition' ? ['recommended', 'disconnected_held', 'recovered'][i] : name,
        state: name === 'transition' ? (feedOk ? 'recommended' : 'disconnected') : 'unconfirmed',
        held: !feedOk, visibleRows: name === 'transition' ? 1 : 0, bodies: phaseBodies };
    });
    return { name, phases };
  });
}

function healthFixtures(base) {
  const normal = liveFixtures(base)[0].phases[0];
  return [['stale_check', '2029-12-31T00:00:00.000Z'], ['future_check', '2030-01-02T00:00:00.000Z']].map(([name, at]) => {
    const phase = structuredClone(normal);
    phase.bodies['/v1/health.json'].game_checks[base.game].checked_at = at;
    Object.assign(phase, { name, state: 'disconnected', held: true, lineKind: 'publish-late' });
    return { name, phases: [phase] };
  });
}

function withdrawalFixtures(base) {
  const normal = liveFixtures(base)[0].phases[0], gamePath = `/v1/game/${base.game}.json`;
  const phases = ['recommended', 'withdrawn', 'old_health', 'missing_check', 'recovered'].map((name, i) => {
    const phase = structuredClone(normal), health = phase.bodies['/v1/health.json'];
    const sequence = [1, 2, 1, 3, 4][i];
    health.sequence = sequence; health.publish.sequence = sequence;
    health.game_checks[base.game].sequence_checked = sequence;
    if (i === 1) health.game_checks[base.game].file_sequence = null;
    if (i === 3) health.game_checks = {};
    if (i === 4) {
      health.game_checks[base.game].file_sequence = 4;
      phase.bodies[gamePath].publish.sequence = 4;
    }
    Object.assign(phase, { name, withdrawn: i > 0 && i < 4, visibleRows: i > 0 && i < 4 ? 0 : 1 });
    return phase;
  });
  return [{ name: 'withdrawal_recovery', phases }];
}

// Pin validation happens before output creation, browser import, or any listening socket.
const assets = pinnedAssets(manifestArg, expectedManifest, expectedSource, buildArg);
const modulePath = noLinks(playwrightArg);
assert.equal(modulePath.split(sep).slice(-2).join('/'), 'playwright/index.mjs');
const output = resolve(outArg); noLinks(dirname(output)); mkdirSync(output, { mode: 0o700 });
const controls = pinControls(output), fixture = syntheticFiles();
const liveCases = suiteArg === '--live-states' ? liveFixtures(fixture) : suiteArg === '--health-freshness' ? healthFixtures(fixture) : suiteArg === '--withdrawal-recovery' ? withdrawalFixtures(fixture) : [];
if (liveCases.length) writeFileSync(join(output, 'live-phases.json'), JSON.stringify(liveCases, null, 2), { flag: 'wx' });
writeFileSync(join(output, 'synthetic-files.json'), JSON.stringify(fixture.bodies, null, 2), { flag: 'wx' });
const allowed = new Map([...assets.allowed, ...fixture.files]);
const receipt = { schema: 'pitcheezy-current-public-candidate-smoke-v1', status: 'INCOMPLETE',
  source: expectedSource, manifestSHA: expectedManifest, buildHashes: assets.hashes,
  fixtureHashes: Object.fromEntries([...fixture.files].map(([p, b]) => [p, sha(b)])), controls,
  suite: suiteArg?.slice(2) ?? 'default', transport: 'owned_loopback_HTTP_not_TLS', requests: [], blocked: [], scenarios: [],
  requestCap: 60, executionBudgetMs: 80000, totalBudgetMs: 90000 };
let browser, browserServer, origin, fault = false, hostError = null, count = 0, routed = 0, timedOut = false;
let browserLaunchStarted = false, browserLaunchSettled = false;
let partialBody = false, partialClosed, resolvePartialClosed, partialTrigger = 'initial_navigation';
const partialResponses = [];
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
  if (partialBody && url.pathname === `/v1/game/${fixture.game}.json`) {
    const body = Buffer.from('{"schema":'), began = performance.now();
    const row = { path: url.pathname, method: req.method, status: 200, sha256: sha(body), bytes: body.length,
      partial: true, serverEnded: false, closed: false, startedMs: began, trigger: partialTrigger };
    receipt.requests.push(row); partialResponses.push(row);
    res.on('close', () => { row.closed = true; row.closeElapsedMs = performance.now() - began; resolvePartialClosed?.(); });
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.flushHeaders(); res.write(body);  // Deliberately no end: the client's body timeout must close this response.
    return;
  }
  const failed = (fault || hostError) && (url.pathname === `/v1/game/${fixture.game}.json` || url.pathname === '/v1/health.json');
  const body = failed ? (hostError?.body ?? Buffer.from('{"error":"synthetic_local_503"}')) : allowed.get(url.pathname);
  const status = failed ? (hostError?.status ?? 503) : 200;
  receipt.requests.push({ path: url.pathname, method: req.method, status, sha256: sha(body), bytes: body.length,
    ...(suiteArg === '--partial-body' ? { startedMs: performance.now(), trigger: partialTrigger } : {}) });
  res.writeHead(status, { ...(liveCases.length ? { Date: 'Tue, 01 Jan 2030 00:00:00 GMT' } : {}), 'Content-Type': types[extname(url.pathname)] ?? types['.html'],
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
const sentinel = 'REMOTE_DETAIL_MUST_NOT_RENDER_CX08';
const hostCases = [
  ['access_required', 401, null, '접근 확인이 필요해요. 페이지를 새로고침해 다시 접속해 주세요.'],
  ['maintenance', 503, 'maintenance', '서비스 점검 중이에요. 잠시 후 다시 확인해 주세요.'],
  ['data_hold', 503, 'data_hold', '자료를 확인하고 있어요. 확인이 끝나면 다시 보여 드려요.'],
  ['unknown_notice', 503, 'synthetic_unknown', '서비스 안내로 자료 제공이 잠시 중단됐어요. 잠시 후 다시 확인해 주세요.'],
  ['retry_final', 503, 'maintenance', '서비스 점검 중이에요. 잠시 후 다시 확인해 주세요.'],
].map(([name, status, id, expected]) => ({ name, status, expected,
  body: Buffer.from(JSON.stringify({ error: status === 401 ? 'access_required' : 'service_notice',
    message: sentinel, detail: sentinel, ...(id ? { notice: { id, message: sentinel, detail: sentinel } } : {}) })) }));
function activatePhase(phase) {
  for (const [path, body] of Object.entries(phase.bodies)) {
    assert.ok(fixture.files.has(path), 'phase outside synthetic allowlist');
    allowed.set(path, Buffer.from(JSON.stringify(body)));
  }
}
async function scenario(width, failing, hostCase = null, liveCase = null, partialCase = false) {
  fault = failing; hostError = hostCase;
  if (partialCase) { partialBody = true; partialClosed = new Promise(done => { resolvePartialClosed = done; }); }
  if (liveCase) activatePhase(liveCase.phases[0]);
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
    let navigations = 0;
    page.on('framenavigated', frame => { if (frame === page.mainFrame()) navigations++; });
    await page.goto(`${origin}/?${failing || hostCase || liveCase || partialCase ? `game=${fixture.game}` : `date=${fixture.date}`}`, { waitUntil: 'domcontentloaded' });
    let hostEvidence;
    if (partialCase) {
      const alert = page.locator('.sv-alert[role=alert]');
      await alert.waitFor({ timeout: 25000 });
      const alertAt = performance.now();
      assert.equal(partialResponses.length, 1, 'exactly one incomplete response before manual retry');
      const partial = partialResponses[0], alertElapsedMs = alertAt - partial.startedMs;
      assert.ok(alertElapsedMs >= 14000 && alertElapsedMs <= 25000, 'body timeout outside 14–25s test tolerance');
      assert.equal(await alert.locator('p').first().innerText(), '서버 응답을 읽지 못했어요.');
      assert.equal(await page.locator('.sv-rec-row').count(), 0);
      await bounded(partialClosed, 2000);
      assert.equal(partial.closed, true); assert.equal(partial.serverEnded, false);
      assert.ok(partial.closeElapsedMs >= 14000 && partial.closeElapsedMs <= 25000);
      await page.screenshot({ path: join(output, 'partial-timeout-390.png'), fullPage: true });
      const retryBegin = receipt.requests.length, retryAt = performance.now();
      assert.ok(retryAt - alertAt < 2000, 'manual retry must precede the 4s automatic retry');
      partialBody = false; partialTrigger = 'manual_retry';
      await alert.getByRole('button', { name: '다시 시도', exact: true }).click();
      await page.locator('main.sv-live-layout').waitFor();
      assert.ok(performance.now() - alertAt < 4000, 'recovery must precede automatic retry');
      assert.match(await page.locator('.sv-live').innerText(), /종료/);
      assert.equal(await page.locator('.sv-alert').count(), 0);
      assert.equal(await page.locator('.sv-rec-row').count(), 0);
      assert.equal(navigations, 1); assert.equal(partialResponses.length, 1);
      const recoveredRequests = receipt.requests.slice(retryBegin), gamePath = `/v1/game/${fixture.game}.json`;
      assert.ok(recoveredRequests.some(r => r.path === gamePath && r.sha256 === receipt.fixtureHashes[gamePath]));
      assert.ok(recoveredRequests.every(r => r.status === 200 && r.trigger === 'manual_retry' && r.startedMs >= retryAt));
      await page.screenshot({ path: join(output, 'partial-recovered-390.png'), fullPage: true });
      hostEvidence = { alertElapsedMs, closeElapsedMs: partial.closeElapsedMs, partialResponses: partialResponses.length,
        serverEndedPartialResponse: partial.serverEnded, partialResponseClosed: partial.closed, navigations,
        recoveryTrigger: 'manual_retry_button', retryAfterAlertMs: retryAt - alertAt, recoveredRequests,
        screenshots: ['partial-timeout-390.png', 'partial-recovered-390.png'] };
    } else if (liveCase) {
      const phases = [];
      let phaseBegin = begin;
      for (const [i, phase] of liveCase.phases.entries()) {
        if (i) {
          phaseBegin = receipt.requests.length; activatePhase(phase);
          if (liveCase.name === 'withdrawal_recovery') {
            const response = page.waitForResponse(r => r.url() === `${origin}/v1/health.json`);
            await (liveCase.phases[i - 1].withdrawn
              ? page.locator('.sv-unavailable').getByRole('button', { name: '다시 확인', exact: true })
              : page.locator('.sv-head .sv-live')).click();
            assert.equal(await (await response).finished(), null, 'health response must finish');
            await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
          } else await page.locator('.sv-head .sv-live').click();
        }
        if (phase.withdrawn) {
          const unavailable = page.locator('.sv-unavailable');
          await unavailable.waitFor();
          assert.equal(await unavailable.locator('[role=status]').first().innerText(),
            '이 경기는 지금 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.');
          assert.equal(await page.locator('.sv-rec-row').count(), 0);
          assert.equal(await page.locator('.sv-head .sv-live.on').count(), 0);
        } else {
        const state = page.locator(`.sv-state[data-state="${phase.state}"]${phase.held ? '.held' : ':not(.held)'}`);
        await state.waitFor();
        assert.equal(await state.locator('.sv-rec-row').count(), phase.visibleRows);
        if (phase.visibleRows) {
          assert.ok(await state.locator('.sv-rec-row').first().isVisible());
          if (phase.held) {
            assert.match(await state.locator('.sv-summary-label').innerText(), /마지막으로 받은 추천/);
            assert.match(await page.locator('.sv-line-text').innerText(), phase.lineKind === 'publish-late' ? /갱신 지연/ : /연결 끊김/);
            if (phase.lineKind) {
              assert.equal(await page.locator('.sv-notice.line').getAttribute('data-kind'), phase.lineKind);
              assert.match(await state.locator('.sv-chip').innerText(), /^⚠\uFE0E\s*갱신 지연$/);
            }
            assert.equal(await page.locator('.sv-head .sv-live.on').count(), 0);
          } else await page.locator('.sv-head .sv-live.on').waitFor();
        }
        }
        const servedPhase = receipt.requests.slice(phaseBegin), healthPath = '/v1/health.json';
        const expectedHashes = Object.fromEntries(Object.entries(phase.bodies).map(([p, b]) => [p, sha(Buffer.from(JSON.stringify(b)))]));
        assert.ok(servedPhase.some(r => r.path === healthPath && r.sha256 === expectedHashes[healthPath]));
        if (phase.withdrawn) assert.ok(!servedPhase.some(r => r.path === `/v1/game/${fixture.game}.json`), 'withdrawn game must not refetch old recommendations');
        if (liveCase.name === 'withdrawal_recovery') assert.equal(navigations, 1, 'one document must retain withdrawal/cache state');
        if (!i || liveCase.name === 'withdrawal_recovery' && !phase.withdrawn) assert.ok(servedPhase.some(r => r.path === `/v1/game/${fixture.game}.json` && r.sha256 === expectedHashes[`/v1/game/${fixture.game}.json`]));
        assert.ok(servedPhase.every(r => r.status === 200));
        for (const r of servedPhase) if (fixture.files.has(r.path)) assert.equal(r.sha256, expectedHashes[r.path]);
        const screenshot = `${liveCase.name}-${phase.name}-390.png`;
        await page.screenshot({ path: join(output, screenshot), fullPage: true });
        phases.push({ name: phase.name, expectedState: phase.withdrawn ? 'unavailable' : phase.state, ...(liveCase.name === 'withdrawal_recovery' ? { navigations, withdrawn: phase.withdrawn } : {}), visibleRows: phase.visibleRows, held: phase.held,
          payloadCandidateCount: phase.bodies[`/v1/game/${fixture.game}.json`].next.rec.candidates.length,
          requestCount: servedPhase.length, requests: servedPhase, fixtureHashes: expectedHashes, screenshot,
          trigger: i ? (liveCase.phases[i - 1].withdrawn ? 'manual_retry_button' : 'manual_live_button') : 'initial_navigation' });
      }
      hostEvidence = { phases, syntheticHTTPDate: 'Tue, 01 Jan 2030 00:00:00 GMT' };
    } else if (hostCase) {
      const alert = page.locator('.sv-alert[role=alert]');
      await alert.waitFor();
      assert.equal(await alert.locator('p').first().innerText(), hostCase.expected);
      assert.equal(await page.locator('.sv-rec-row').count(), 0);
      assert.ok(!(await page.locator('body').innerText()).includes(sentinel));
      const errorRequests = receipt.requests.slice(begin).filter(r => r.path === `/v1/game/${fixture.game}.json` || r.path === '/v1/health.json');
      assert.ok(errorRequests.some(r => r.path === `/v1/game/${fixture.game}.json`));
      assert.ok(errorRequests.every(r => r.status === hostCase.status && r.sha256 === sha(hostCase.body)));
      const initialScreenshot = `${hostCase.name}-390.png`;
      await page.screenshot({ path: join(output, initialScreenshot), fullPage: true });
      hostEvidence = { expectedRender: hostCase.expected, initialRecommendationRows: 0, remoteSentinelVisible: false,
        errorBodySHA: sha(hostCase.body), errorRequests, screenshots: [initialScreenshot], recovered: false };
      if (hostCase.name === 'retry_final') {
        const retryBegin = receipt.requests.length;
        hostError = null;
        await alert.getByRole('button', { name: '다시 시도', exact: true }).click();
        await page.locator('main.sv-live-layout').waitFor();
        assert.match(await page.locator('.sv-live').innerText(), /종료/);
        assert.equal(await page.locator('.sv-alert').count(), 0);
        assert.ok(!(await page.locator('body').innerText()).includes(sentinel));
        const recoveredRequests = receipt.requests.slice(retryBegin);
        const gamePath = `/v1/game/${fixture.game}.json`;
        assert.ok(recoveredRequests.some(r => r.path === gamePath && r.status === 200 && r.sha256 === receipt.fixtureHashes[gamePath]));
        assert.ok(recoveredRequests.every(r => r.status === 200));
        await page.screenshot({ path: join(output, 'retry_final-recovered-390.png'), fullPage: true });
        Object.assign(hostEvidence, { recovered: true, recoveryTrigger: 'manual_retry_button', recoveredRequests,
          consumedFixtureSHA: receipt.fixtureHashes[gamePath] });
        hostEvidence.screenshots.push('retry_final-recovered-390.png');
      }
    } else if (failing) {
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
    assert.ok(served.some(r => r.path === gamePath && r.status === (hostCase?.status ?? (failing ? 503 : 200))));
    if (!failing && !hostCase && !liveCase && !partialCase) {
      for (const path of ['/v1/manifest.json', `/v1/games/${fixture.date}.json`, gamePath])
        assert.ok(served.some(r => r.path === path && r.sha256 === receipt.fixtureHashes[path]));
      assert.ok(served.every(r => r.status === 200));
    } else if (!hostCase && !liveCase && !partialCase) assert.ok(served.some(r => r.path === '/v1/health.json' && r.status === 503));
    if (!hostCase && !liveCase && !partialCase) await page.screenshot({ path: join(output, `${failing ? '503' : 'final'}-${width}.png`), fullPage: true });
    receipt.scenarios.push({ kind: liveCase?.name ?? hostCase?.name ?? (partialCase ? 'partial_body_timeout_recovery' : failing ? '503_no_recommendation' : 'list_to_final'), width, height: 900,
      requestCount: served.length, pageErrors: errors, fixtureGame: fixture.game, ...(hostEvidence ?? {}) });
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
    if (liveCases.length) { for (const liveCase of liveCases) await scenario(390, false, null, liveCase); }
    else if (suiteArg === '--partial-body') await scenario(390, false, null, null, true);
    else if (suiteArg) { for (const hostCase of hostCases) await scenario(390, false, hostCase); }
    else { await scenario(390, false); await scenario(1440, false); await scenario(390, true); }
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
  if (suiteArg === '--partial-body') {
    receipt.partialResponsesClosed = partialResponses.every(r => r.closed);
    if (!receipt.partialResponsesClosed) { receipt.cleanupError = 'owned partial response remained open'; process.exitCode = 1; }
  }
  receipt.routedAttempts = routed; receipt.totalCalls = count; receipt.elapsedMs = Date.now() - started;
  if (!receipt.ownedBrowserClosed || !receipt.ownedServerClosed || receipt.elapsedMs > 90000 || receipt.cleanupError) {
    receipt.status = 'FAIL_CLEANUP_OR_BUDGET'; process.exitCode = 1;
  }
  writeFileSync(join(output, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ status: receipt.status, scenarios: receipt.scenarios.length,
    calls: count, ownedBrowserClosed: receipt.ownedBrowserClosed, ownedServerClosed: receipt.ownedServerClosed }));
}
