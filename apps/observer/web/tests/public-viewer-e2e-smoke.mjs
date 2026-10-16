// Synthetic publisher output -> exact public build -> owned localhost HTTPS -> Chromium.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer, request as httpsRequest } from 'node:https';
import { lstatSync, readFileSync, writeFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';

const [buildArg, outArg, keyArg, certArg, privateKeyArg, receiptArg, gatewayPortArg] = process.argv.slice(2);
assert.ok(receiptArg, 'build out game cert key receipt required');
const build = resolve(buildArg), out = resolve(outArg), game = Number(keyArg);
assert.ok(Number.isSafeInteger(game) && game > 0);
const gatewayPort = gatewayPortArg === undefined ? null : Number(gatewayPortArg);
assert.ok(gatewayPort === null || (Number.isSafeInteger(gatewayPort) && gatewayPort > 0 && gatewayPort < 65536));
const sh = data => createHash('sha256').update(data).digest('hex');
const assets = {
  '/': ['index.html', 'bccdf2b83ff613d40ae15b766fa0b056a9ec1d894f5ed13caa96859cadd5b8fb'],
  '/index.html': ['index.html', 'bccdf2b83ff613d40ae15b766fa0b056a9ec1d894f5ed13caa96859cadd5b8fb'],
  '/favicon.svg': ['favicon.svg', 'bbfd72e5d077bd545b2cf568e5267c6637598d5253746f5b395e0d5f3fb144d9'],
  '/assets/index-D5DvN5mB.js': ['assets/index-D5DvN5mB.js', '7444fdb7f01e9d8729ec39eb954871d9b7801ec9bdeb02375052b35e67d74231'],
  '/assets/index-Dw6b8jjM.css': ['assets/index-Dw6b8jjM.css', '3745f095a47d6cea5475b52533bf5a1aab820d9d14bf6c5932f45760e7024a4f'],
};
const allowed = new Map();
for (const [url, [rel, expected]] of Object.entries(assets)) {
  const path = join(build, rel);
  assert.ok(lstatSync(path).isFile() && !lstatSync(path).isSymbolicLink());
  const data = readFileSync(path);
  assert.equal(sh(data), expected, `public build ${rel}`);
  allowed.set(url, data);
}
const v1 = ['manifest.json', 'dates.json', 'health.json', 'games/2026-10-05.json', `game/${game}.json`];
const runnerHashes = {};
for (const rel of v1) {
  const path = join(out, 'v1', rel);
  assert.ok(lstatSync(path).isFile() && !lstatSync(path).isSymbolicLink());
  const data = readFileSync(path);
  assert.ok(data.length > 0 && data.length < 2 ** 20);
  allowed.set(`/v1/${rel}`, data);
  runnerHashes[`v1/${rel}`] = sh(data);
}
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json' };
let fault = false;
const requests = [];
let totalCalls = 0;
function gatewayGet(path, cookie) {
  return new Promise((resolve, reject) => {
    const request = httpsRequest({ hostname: '127.0.0.1', port: gatewayPort, servername: 'localhost', path,
      method: 'GET', ca: readFileSync(certArg), timeout: 5000, headers: { Cookie: cookie ?? '' } }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => resolve({ status: response.statusCode, body: Buffer.concat(chunks),
        contentType: response.headers['content-type'], etag: response.headers.etag }));
    });
    request.on('timeout', () => request.destroy(new Error('gateway timeout')));
    request.on('error', reject);
    request.end();
  });
}
const server = createServer({ key: readFileSync(privateKeyArg), cert: readFileSync(certArg) }, async (req, res) => {
  if (++totalCalls > 60) { res.writeHead(429); res.end(); return; }
  const url = new URL(req.url, 'https://localhost');
  const path = url.pathname;
  const override = fault && (path === `/v1/game/${game}.json` || path === '/v1/health.json');
  try {
    const forwarded = gatewayPort && !override && path.startsWith('/v1/') && allowed.has(path)
      ? await gatewayGet(path, req.headers.cookie) : null;
    const body = override ? Buffer.from('{"error":"synthetic_local_503"}') : forwarded?.body ?? allowed.get(path);
    const status = override ? 503 : forwarded?.status ?? (body ? 200 : 404);
    const data = body ?? Buffer.from('not found');
    const expected = allowed.get(path);
    if (forwarded && (status !== 200 || sh(data) !== sh(expected))) throw new Error('gateway stored bytes mismatch');
    requests.push({ method: req.method, path, status, sha256: sh(data), bytes: data.length, fault: override,
      gatewayStored: Boolean(forwarded), cookiePresent: Boolean(req.headers.cookie) });
    res.writeHead(status, { 'Content-Type': forwarded?.contentType ?? (path === '/' ? mime['.html'] : mime[extname(path)] ?? 'text/plain'),
      ...(forwarded?.etag ? { ETag: forwarded.etag } : {}), 'Cache-Control': 'no-store' });
    res.end(data);
  } catch (error) {
    requests.push({ method: req.method, path, status: 502, error: String(error) });
    res.writeHead(502); res.end();
  }
});
let browser;
const receipt = { schema: 'pitcheezy-public-viewer-e2e-synthetic-v1', status: 'INCOMPLETE', game,
  transport: gatewayPort === null ? 'direct_static' : 'pinned_local_gateway_via_front',
  buildHashes: Object.fromEntries(Object.entries(assets).filter(([k]) => k !== '/').map(([k, [, sha]]) => [k, sha])),
  runnerHashes, requests, scenarios: [] };
try {
  await new Promise((done, fail) => server.once('error', fail).listen(0, 'localhost', done));
  const origin = `https://localhost:${server.address().port}`;
  browser = await chromium.launch({ headless: true });
  for (const width of [390, 1440]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, ignoreHTTPSErrors: true });
    const page = await context.newPage(), external = [], pageErrors = [], origins = [];
    if (gatewayPort !== null) await context.addCookies([{ name: 'synthetic-session', value: 'local-only', url: origin,
      httpOnly: true, sameSite: 'Strict' }]);
    page.on('pageerror', e => pageErrors.push(String(e)));
    await page.route('**/*', route => {
      const requested = new URL(route.request().url()); origins.push(requested.origin);
      if (requested.origin !== origin) { external.push(requested.origin); return route.abort(); }
      return route.continue();
    });
    const start = requests.length;
    await page.goto(`${origin}/?date=2026-10-05`, { waitUntil: 'domcontentloaded', timeout: 15000 });
    await page.locator(`.sv-list .sv-card[href="./?game=${game}"]`).waitFor({ timeout: 15000 });
    await page.locator(`.sv-list .sv-card[href="./?game=${game}"]`).click();
    await page.locator('main.sv-live-layout').waitFor({ timeout: 15000 }).catch(async error => {
      receipt.debugBody = (await page.locator('body').innerText()).slice(0, 800);
      receipt.debugPageErrors = pageErrors;
      throw error;
    });
    assert.match(await page.locator('body').innerText(), /경기가 끝났어요/);
    assert.equal(await page.locator('.sv-alert').count(), 0);
    assert.deepEqual(pageErrors, []); assert.deepEqual(external, []);
    const served = requests.slice(start);
    assert.ok(served.some(r => r.path === `/v1/game/${game}.json` && r.status === 200 && r.sha256 === runnerHashes[`v1/game/${game}.json`]));
    assert.ok(served.some(r => r.path === '/v1/manifest.json' && r.status === 200));
    assert.ok(served.some(r => r.path === '/v1/games/2026-10-05.json' && r.status === 200));
    assert.ok(served.every(r => r.method === 'GET' && r.status === 200));
    if (gatewayPort !== null) assert.ok(served.filter(r => r.path.startsWith('/v1/')).every(r => r.gatewayStored && r.cookiePresent));
    assert.ok(served.every(r => assets[r.path] || allowed.has(r.path)));
    assert.ok(origins.length && origins.every(value => value === origin));
    receipt.scenarios.push({ kind: 'browser_consumed_runner_output', width, browserUrl: page.url(), origins, pageErrors, requestCount: served.length });
    await context.close();
  }
  fault = true;
  const context = await browser.newContext({ viewport: { width: 390, height: 900 }, ignoreHTTPSErrors: true });
  const page = await context.newPage(), external = [], origins = [];
  if (gatewayPort !== null) await context.addCookies([{ name: 'synthetic-session', value: 'local-only', url: origin,
    httpOnly: true, sameSite: 'Strict' }]);
  await page.route('**/*', route => {
    const requested = new URL(route.request().url()); origins.push(requested.origin);
    if (requested.origin !== origin) { external.push(requested.origin); return route.abort(); }
    return route.continue();
  });
  const start = requests.length;
  await page.goto(`${origin}/?game=${game}`, { waitUntil: 'domcontentloaded', timeout: 15000 });
  await page.locator('.sv-alert[role=alert]').waitFor({ timeout: 15000 }).catch(async error => {
    receipt.debugBody = (await page.locator('body').innerText()).slice(0, 800);
    throw error;
  });
  assert.equal(await page.locator('.sv-rec-row').count(), 0);
  const served = requests.slice(start);
  assert.ok(served.some(r => r.path === `/v1/game/${game}.json` && r.status === 503 && r.fault));
  assert.ok(served.some(r => r.path === '/v1/health.json' && r.status === 503 && r.fault));
  assert.deepEqual(external, []); assert.ok(origins.every(value => value === origin));
  assert.ok(served.every(r => r.method === 'GET' && (assets[r.path] || allowed.has(r.path))));
  receipt.scenarios.push({ kind: 'synthetic_error_override', browserUrl: page.url(), origins, alert: await page.locator('.sv-alert').innerText(),
    faultedRequests: served.filter(r => r.fault).length });
  await context.close();
  assert.ok(totalCalls <= 60);
  receipt.status = 'PASS_SYNTHETIC_LOCAL_HTTPS';
} catch (error) {
  receipt.error = String(error);
  throw error;
} finally {
  if (browser) await browser.close();
  if (server.listening) await new Promise(resolve => server.close(resolve));
  receipt.ownedServerClosed = !server.listening;
  receipt.totalCalls = totalCalls;
  writeFileSync(receiptArg, JSON.stringify(receipt, null, 2) + '\n');
}
