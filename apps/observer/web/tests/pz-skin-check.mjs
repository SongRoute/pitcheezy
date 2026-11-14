// PZ-SKIN check (2026-11-15): the teammate colour theme + 3D batter (pz-skin/) on the public game screen, and fixes ①~④.
//   PZ_SKIN=3d PZ_BATTER_MODEL=batter-ctl npm run build:public && node tests/pz-skin-check.mjs [shotDir]   (PZ_SKIN defaults to off since review P2-4)
//   PZ_WEB_DIST_PUBLIC=<dir>  test another public build (default dist-public; built with PZ_SKIN=3d)
//   PZ_CHECK_PORT=<8880-8889> loopback port of the fake static host (default 8884)
// Synthetic loopback only: tests/fixtures/publish/v1 (stamps moved to "now"), every other origin aborted. No real 3D model exists yet:
//   model '404'  → every pz/batter3d/assets/* request answers 404 (today's state)
//   model 'fake' → any *.glb request gets a tiny generated glTF (one triangle, clips idle/contact_hit1/miss/take/hit) so the
//                  attach / react / unmount paths run with real three.js + WebGL (SwiftShader). Meta files always 404.
// What it checks: tags and pz/ files; ③ list screen downloads no 3D code, game screen does; 404 → silhouette stays, no errors,
// bounded retries; fake model → attaches, silhouette hidden, first screen does not move; ① tab return / online re-read does not
// replay; a new pitch does; ④ box inside 320/344/359/360 screens, zone painted above it; ② contrast; 10/26 show/hide rules.
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.resolve(process.env.PZ_WEB_DIST_PUBLIC ?? path.join(here, '..', 'dist-public'));
const v1 = path.join(here, 'fixtures', 'publish', 'v1');
const PK = 849843, LIVE_PK = 990003;
const plainDist = path.join(here, '..', 'dist-public-plain');   // same source, PZ_SKIN=off (= the c47 screen), built below
const noModelDist = path.join(here, '..', 'dist-public-nomodel');   // PZ_SKIN=3d without PZ_BATTER_MODEL (review P2-3), built below
const PORT = Number(process.env.PZ_CHECK_PORT ?? 8884);
if (!(PORT >= 8880 && PORT <= 8889)) throw Error('PZ_CHECK_PORT must be 8880-8889');
const shots = process.argv[2]; if (shots) mkdirSync(shots, { recursive: true });
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.glb': 'model/gltf-binary' };
let failures = 0, passes = 0;
const ok = (cond, message, extra) => { if (cond) { passes++; console.log('ok  ', message); } else { failures++; console.error('FAIL', message, extra !== undefined ? JSON.stringify(extra) : ''); } };
async function waited(message, wait) { try { await wait(); ok(true, message); return true; } catch (e) { ok(false, `${message} — ${String(e).split('\n')[0].slice(0, 160)}`); return false; } }

// ---------- contrast (WCAG 2.x relative luminance) ----------
const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const rgb = (s) => { if (s.startsWith('#')) { const n = parseInt(s.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; } return s.match(/[\d.]+/g).slice(0, 3).map(Number); };
const L = (s) => { const [r, g, b] = rgb(s); return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b); };
const cr = (a, b) => { const x = L(a), y = L(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };

// ---------- a tiny valid glb: one triangle node + rotation clips (no skin; batter3d.js falls back to whole-model bounds) ----------
function fakeGlb() {
  const clips = [['idle', 1.0], ['contact_hit1', 2.4], ['miss', 2.6], ['take', 1.6], ['hit', 2.4]];
  const parts = [], views = [], accessors = [];
  let off = 0;
  const push = (f32, extra) => { const b = Buffer.from(f32.buffer); views.push({ buffer: 0, byteOffset: off, byteLength: b.length }); accessors.push({ bufferView: views.length - 1, componentType: 5126, ...extra }); parts.push(b); off += b.length; return accessors.length - 1; };
  const pos = push(new Float32Array([-0.3, 0, 0, 0.3, 0, 0, 0, 1.8, 0]), { count: 3, type: 'VEC3', min: [-0.3, 0, 0], max: [0.3, 1.8, 0] });
  const animations = clips.map(([name, d]) => {
    const t = push(new Float32Array([0, d]), { count: 2, type: 'SCALAR', min: [0], max: [d] });
    const q = push(new Float32Array([0, 0, 0, 1, 0, 0.7071068, 0, 0.7071068]), { count: 2, type: 'VEC4' });
    return { name, samplers: [{ input: t, output: q, interpolation: 'LINEAR' }], channels: [{ sampler: 0, target: { node: 0, path: 'rotation' } }] };
  });
  const bin = Buffer.concat(parts);
  const json = { asset: { version: '2.0', generator: 'pz-skin-check fake' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: 'Body', mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: pos }, material: 0 }] }], materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.85, 0.1, 0.1, 1] } }],
    buffers: [{ byteLength: bin.length }], bufferViews: views, accessors, animations };
  let js = Buffer.from(JSON.stringify(json)); if (js.length % 4) js = Buffer.concat([js, Buffer.alloc(4 - js.length % 4, 0x20)]);
  const binP = bin.length % 4 ? Buffer.concat([bin, Buffer.alloc(4 - bin.length % 4)]) : bin;
  const head = Buffer.alloc(12); head.writeUInt32LE(0x46546c67, 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(12 + 8 + js.length + 8 + binP.length, 8);
  const ch = (len, type) => { const b = Buffer.alloc(8); b.writeUInt32LE(len, 0); b.writeUInt32LE(type, 4); return b; };
  return Buffer.concat([head, ch(js.length, 0x4e4f534a), js, ch(binP.length, 0x004e4942), binP]);
}
const GLB = fakeGlb();

// ---------- fake static host ----------
const GAME_SEQ = JSON.parse(readFileSync(path.join(v1, 'game', `${PK}.json`), 'utf8')).publish.sequence;   // the fixture file's own sequence (freshen adds the same step)
const host = { model: '404', asks: [], synthetic: null, checks: false, slowGameMs: 0 };
const started = Date.now();
const isoNow = () => new Date(Date.now() - 1000).toISOString().replace(/\.\d{3}Z$/, 'Z');
function freshen(body) {
  const at = isoNow(), add = Math.floor((Date.now() - started) / 10000);
  const stamp = (o) => { if (!o || typeof o !== 'object') return; if (typeof o.published_at === 'string') o.published_at = at; if (Number.isSafeInteger(o.sequence)) o.sequence += add; };
  stamp(body); stamp(body.publish);
  if (body.schema?.includes('health') && body.game_checks && typeof body.game_checks === 'object') for (const c of Object.values(body.game_checks)) {
    if (!c || typeof c !== 'object') continue;
    if (typeof c.checked_at === 'string') c.checked_at = at;
    for (const k of ['sequence_checked', 'file_sequence']) if (Number.isSafeInteger(c[k])) c[k] += add;
  }
  if (host.checks && body.schema?.includes('health')) body.game_checks = { [PK]: { checked_at: at, sequence_checked: Math.max(body.sequence, GAME_SEQ + add), file_sequence: GAME_SEQ + add, feed_ok: true } };   // ① masked re-read scene
  return body;
}
const server = createServer((req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  const u = new URL(req.url, 'http://x');
  host.asks.push(u.pathname);
  const m = u.pathname.match(/\/v1(\/.*)$/);
  if (m) {
    const disk = path.resolve(v1, '.' + m[1]);
    if (!disk.startsWith(v1 + path.sep) || !existsSync(disk) || !statSync(disk).isFile()) { res.writeHead(404); return res.end('not found'); }
    const send = () => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(freshen(JSON.parse(readFileSync(disk, 'utf8'))))); };
    if (host.slowGameMs && (m[1].startsWith('/game/') || m[1] === '/health.json')) { setTimeout(send, host.slowGameMs); return; }   // the re-read after 'online' takes a while → the screen masks (확인 중)
    return send();
  }
  if (u.pathname.includes('/pz/batter3d/assets/')) {
    if (host.model === 'fake' && u.pathname.endsWith('.glb')) { res.writeHead(200, { 'content-type': TYPES['.glb'] }); return res.end(GLB); }
    res.writeHead(404, { 'content-type': 'text/plain' }); return res.end('not found');
  }
  for (const [pre, root] of [['/__plain__/', plainDist], ['/__nomodel__/', noModelDist]]) if (u.pathname.startsWith(pre)) {
    const f = path.resolve(root, '.' + u.pathname.slice(pre.length - 1));
    const t = f.startsWith(root + path.sep) && existsSync(f) && statSync(f).isFile() ? f : path.join(root, 'index.html');
    res.writeHead(200, { 'content-type': TYPES[path.extname(t)] ?? 'application/octet-stream' }); return res.end(readFileSync(t));
  }
  const file = path.resolve(dist, '.' + u.pathname);
  if (file.startsWith(dist + path.sep) && existsSync(file) && statSync(file).isFile()) { res.writeHead(200, { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream' }); return res.end(readFileSync(file)); }
  if (u.pathname === '/__synthetic__.html' && host.synthetic) { res.writeHead(200, { 'content-type': TYPES['.html'] }); return res.end(host.synthetic); }
  if (path.extname(u.pathname)) { res.writeHead(404); return res.end('not found'); }   // missing files are 404 (a static host), pages fall back to index.html
  res.writeHead(200, { 'content-type': TYPES['.html'] }); res.end(readFileSync(path.join(dist, 'index.html')));
});
await new Promise((r, j) => { server.once('error', j); server.listen(PORT, '127.0.0.1', r); });
const origin = `http://127.0.0.1:${PORT}`;

for (const [out, env] of [[plainDist, { PZ_SKIN: 'off' }], [noModelDist, { PZ_SKIN: '3d', PZ_BATTER_MODEL: '' }]])
  execFileSync(process.execPath, [path.join(here, '../node_modules/vite/bin/vite.js'), 'build', '--mode', 'public', '--outDir', out, '--logLevel', 'error'], { cwd: path.join(here, '..'), env: { ...process.env, ...env }, stdio: 'inherit' });
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
async function open(width, height, { reduceMotion = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2, hasTouch: true, isMobile: true, reducedMotion: reduceMotion ? 'reduce' : 'no-preference' });
  const page = await context.newPage();
  const errors = [], external = [], requests = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error' && !/404|Failed to load resource/.test(m.text())) errors.push('console: ' + m.text()); });
  page.on('request', (r) => requests.push(new URL(r.url()).pathname));
  await context.route('**/*', (route) => { const u = new URL(route.request().url()); if (u.origin !== origin) { external.push(u.href); return route.abort('blockedbyclient'); } return route.continue(); });
  return { context, page, errors, external, requests };
}
const IS3D = /\/pz\/(pz-batter\.js|batter3d\/)/;
const status = (page) => page.evaluate(() => window.PZSongSkin ? window.PZSongSkin.status() : null);
const silhouette = (page) => page.evaluate(() => { const s = document.querySelector('.sv-game .sv-batband svg'); return s ? getComputedStyle(s).visibility : 'none'; });
const sideScroll = (page) => page.evaluate(() => document.scrollingElement.scrollWidth - innerWidth);

try {
  // ---------- 0. build output (3D build: PZ_SKIN=3d; colour-only build: PZ_SKIN=theme, detected from index.html) ----------
  const html = readFileSync(path.join(dist, 'index.html'), 'utf8');
  const mode3d = html.includes('pz/pz-boot.js');
  console.log(`mode: ${mode3d ? '3d (colour + 3D batter)' : 'theme (colour only)'} — ${dist}`);
  const cssApp = html.lastIndexOf('<link rel="stylesheet" crossorigin href="./assets/'), cssPz = html.indexOf('<link rel="stylesheet" href="./pz/pz-theme.css">');
  ok(cssPz > 0 && cssApp >= 0 && cssApp < cssPz, 'index.html: pz-theme.css linked after the app CSS', { cssApp, cssPz });
  ok(html.includes('<meta name="theme-color" content="#091811" />'), 'index.html: theme-color #091811');
  if (mode3d) {
    ok(/<script type="importmap">\{ "imports": \{ "three": "\.\/pz\/batter3d\/vendor\/three\.module\.js", "three\/addons\/": "\.\/pz\/batter3d\/vendor\/addons\/" \} \}<\/script>/.test(html), 'index.html: import map tag');
    ok(html.indexOf('type="importmap"') < html.indexOf('<script type="module"'), 'index.html: import map before the first module script');
    ok(html.includes('<script type="module" src="./pz/pz-boot.js"></script>') && !html.includes('pz-batter.js'), 'index.html: ③ only the small boot script is static (no pz-batter.js tag)');
    for (const f of ['pz/pz-theme.css', 'pz/pz-boot.js', 'pz/pz-batter.js', 'pz/batter3d/batter3d.js', 'pz/batter3d/vendor/three.module.js', 'pz/batter3d/vendor/three.core.js',
      'pz/batter3d/vendor/addons/loaders/GLTFLoader.js', 'pz/batter3d/vendor/addons/utils/BufferGeometryUtils.js', 'pz/batter3d/vendor/addons/utils/SkeletonUtils.js', 'pz/batter3d/vendor/LICENSE'])
      ok(existsSync(path.join(dist, f)), `dist: ${f}`);
    const assets = path.join(dist, 'pz', 'batter3d', 'assets');
    const models = existsSync(assets) ? readdirSync(assets).filter((f) => /\.(glb|meta\.json)$/.test(f)) : [];
    ok(models.length === 0 || !!process.env.PZ_BATTER_ASSETS_DIR, `dist: no model / meta files in pz/batter3d/assets (${models.join(',') || 'none'})`);
  } else {
    ok(!html.includes('importmap') && (html.match(/<script /g) || []).length === 1, 'index.html (colour only): no import map, only the app script');
    ok(!existsSync(path.join(dist, 'pz', 'batter3d')) && !existsSync(path.join(dist, 'pz', 'pz-batter.js')) && !existsSync(path.join(dist, 'pz', 'pz-boot.js')), 'dist (colour only): no 3D files');
  }
  const css = readFileSync(path.join(dist, 'pz', 'pz-theme.css'), 'utf8');

  // ---------- ② contrast (tokens in pz-theme.css, then what the page computes) ----------
  const play = [...css.matchAll(/--sv-g1-play:\s*(#[0-9a-f]{6})/gi)].map((m) => m[1].toLowerCase());
  ok(play.length === 2 && play.every((c) => c === play[0]), `② --sv-g1-play set the same in game and sheet tokens (${play.join(',')})`);
  for (const [bg, name] of [['#ffffff', 'paper'], ['#eef8dc', 'selected row'], ['#eef3e8', 'page background']])
    ok(cr(play[0], bg) >= 3, `② in-play mark vs ${name} ${bg}: ${cr(play[0], bg).toFixed(2)}:1 ≥ 3:1`);
  ok(!/ffb782/i.test(css.replace(/\/\*[\s\S]*?\*\//g, '')), '② the old #ffb782 is no longer used by any rule (comments aside)');

  if (!mode3d) {   // colour only: neither screen asks for anything under pz/ but the theme
    const { context, page, errors, requests } = await open(390, 844);
    await page.goto(origin + '/'); await page.locator('.sv-list').waitFor({ timeout: 15000 });
    const card = page.locator(`a.sv-card[href*="${PK}"]`).first(); await card.click(); await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 }); await page.waitForTimeout(1500);
    const pz = [...new Set(requests.filter((p) => p.includes('/pz/')))];
    ok(pz.length === 1 && pz[0].endsWith('/pz/pz-theme.css'), `colour only: list → game asks only pz/pz-theme.css under pz/ (${pz.join(',')})`);
    ok(await silhouette(page) === 'visible' && await page.locator('.pz-bat3d').count() === 0, 'colour only: silhouette visible, no 3D box');
    ok(errors.length === 0, `colour only: no page errors (${errors.length})`, errors.slice(0, 3));
    await context.close();
  }
  if (mode3d) {
  // ---------- ③ list screen downloads no 3D code; the game screen does ----------
    host.model = '404';
    {
      const { context, page, errors, external, requests } = await open(390, 844);
      await page.goto(origin + '/');
      await page.locator('.sv-list').waitFor({ timeout: 15000 });
      await page.waitForTimeout(1500);
      ok(requests.some((p) => p.endsWith('/pz/pz-boot.js')) && requests.some((p) => p.endsWith('/pz/pz-theme.css')), '③ list: boot script and theme are loaded');
      ok(!requests.some((p) => IS3D.test(p)), '③ list: no pz-batter.js / batter3d / three.js request', requests.filter((p) => IS3D.test(p)));
      ok(await page.evaluate(() => window.PZSongSkinBoot?.status().requested === false), '③ list: boot has not requested the 3D code');
      const card = page.locator(`a.sv-card[href*="${PK}"]`).first();
      if (await card.count()) {
        await card.click();
        await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
        await waited('③ list → game: 3D code requested once the zone row exists', () => page.waitForFunction(() => window.PZSongSkinBoot?.status().loaded === true, null, { timeout: 15000 }));
        ok(requests.filter((p) => p.endsWith('/pz/pz-batter.js')).length === 1 && requests.some((p) => p.endsWith('/vendor/three.module.js')) && requests.some((p) => p.endsWith('/vendor/three.core.js')), '③ list → game: pz-batter.js once, three.js module + core fetched');
      } else ok(false, '③ list shows the fixture game card');
      ok(errors.length === 0, `③ list/game: no page errors (${errors.length})`, errors.slice(0, 3));
      ok(external.length === 0, '③ no request leaves the loopback host', external.slice(0, 3));
      await context.close();
    }

    // ---------- review P2-3: a 3D build without a model name loads no 3D code at all (0 requests), silhouette stays ----------
    {
      const { context, page, errors, requests } = await open(390, 844);
      await page.goto(`${origin}/__nomodel__/?game=${PK}`);
      await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
      await page.waitForTimeout(2500);
      for (let i = 0; i < 2; i++) { await page.evaluate(() => window.dispatchEvent(new Event('online'))); await page.waitForTimeout(600); }
      const boot = await page.evaluate(() => window.PZSongSkinBoot?.status() ?? null);
      const bad = requests.filter((p) => IS3D.test(p) || p.endsWith('.glb') || p.endsWith('.meta.json'));
      ok(boot && boot.skipped && !boot.requested && bad.length === 0, `P2-3 no model name: boot skipped (${boot?.skipped}), 3D / model requests ${bad.length}`, bad.slice(0, 3));
      ok(requests.some((p) => p.endsWith('/pz/pz-theme.css')) && await silhouette(page) === 'visible' && await page.locator('.pz-bat3d').count() === 0, 'P2-3 no model name: theme on, silhouette visible, no 3D box');
      ok(errors.length === 0, `P2-3 no model name: no page errors (${errors.length})`, errors.slice(0, 3));
      await context.close();
    }

    // ---------- model 404 (today): silhouette stays, nothing breaks, retries bounded ----------
    for (const [w, h] of [[390, 844], [320, 568]]) {
      const { context, page, errors, requests } = await open(w, h);
      await page.goto(`${origin}/?game=${PK}`);
      await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
      await page.waitForFunction(() => window.PZSongSkinBoot?.status().loaded === true, null, { timeout: 15000 }).catch(() => {});
      await page.waitForTimeout(4000);
      const glb1 = requests.filter((p) => p.endsWith('.glb')).length;
      for (let i = 0; i < 3; i++) { await page.evaluate(() => window.dispatchEvent(new Event('online'))); await page.waitForTimeout(800); }
      await page.waitForTimeout(3000);
      const glb2 = requests.filter((p) => p.endsWith('.glb')).length, st = await status(page);
      const id = `404 ${w}x${h}`;
      ok(st && st.attached === false && st.fails >= 1 && st.fails <= 3, `${id}: not attached, fails counted ${st?.fails} (≤ 3)`, st && { last: st.last });
      ok(glb2 <= 3 && glb2 === glb1, `${id}: model asked ${glb1} → ${glb2} times (≤ 3, no new asks after re-reads)`);
      ok(await silhouette(page) === 'visible', `${id}: silhouette stays visible`);
      ok(await page.locator('.pz-bat3d').count() === 0 && await page.locator('.sv-zrow.pz-has3d').count() === 0, `${id}: no 3D box, no pz-has3d left behind`);
      ok(await sideScroll(page) <= 0, `${id}: no side scroll`);
      ok(errors.length === 0, `${id}: no page errors (${errors.length})`, errors.slice(0, 3));
      if (shots) await page.screenshot({ path: path.join(shots, `404-${w}.png`) });
      await context.close();
    }

    // ---------- fake model: attach, first screen still, ① no replay on re-read / tab return, new pitch plays ----------
    host.model = 'fake';
    {
      const { context, page, errors } = await open(390, 844);
      await page.goto(`${origin}/?game=${PK}`);
      await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
      const attached = await waited('fake: 3D attaches with a valid glb (real three.js + WebGL)', () => page.waitForFunction(() => window.PZSongSkin?.status().attached === true, null, { timeout: 20000 }));
      if (attached) {
        ok(await silhouette(page) === 'hidden' && await page.locator('.sv-zrow.pz-has3d .pz-bat3d canvas').count() === 1, 'fake: silhouette hidden, one canvas in the box');
        await page.waitForTimeout(1200);
        const s0 = await status(page);
        ok(s0.played === 0, `fake: first screen does not play a reaction (played ${s0.played})`);
        // P2-1 (review): picking a swinging-strike row, then a foul row, each plays exactly one reaction (the fake model has miss and contact)
        const lineText = () => page.evaluate(() => document.querySelector('.sv-game .sv-actual-line')?.innerText ?? '');
        const pick = async (re) => {
          if ((await page.locator('.sv-atbat-toggle').getAttribute('aria-expanded')) !== 'true') await page.locator('.sv-atbat-toggle').click();
          const labels = await page.locator('.sv-pitch-row').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label') || ''));
          const i = labels.findIndex((l) => re.test(l)); if (i < 0) { console.log('note at-bat rows:', labels.join(' | ')); return null; }
          await page.locator('.sv-pitch-row').nth(i).click(); await page.waitForTimeout(400);
          const close = page.locator('.sv-sheet button', { hasText: '닫기' }); if (await close.count()) await close.first().click();
          await page.waitForTimeout(900); return labels[i];
        };
        const s1 = await status(page);
        const missRow = await pick(/헛스윙/), s2 = await status(page);
        ok(missRow && s2.played === s1.played + 1 && s2.last === 'miss', `P2-1 picking "${missRow}" plays exactly one miss (played ${s1.played} → ${s2.played}, last ${s2.last})`);
        const foulRow = await pick(/파울|삼진/), s3 = await status(page);
        ok(foulRow && s3.played === s2.played + 1 && s3.last === 'contact', `P2-1 picking "${foulRow}" plays exactly one contact (played ${s2.played} → ${s3.played}, last ${s3.last})`);
        // P2-2 (review): the real screen blanks the actual-pitch line while a re-read is masked (확인 중: maskPublished → "실제 투구 기록 없음")
        // and then shows the same pitch again. Health announces a check for this game; 'online' starts a re-read that answers 2.5 s late.
        host.checks = true; await page.waitForTimeout(11000);   // a health read with the check (10 s cadence)
        const before = await lineText(), s4 = await status(page);
        host.slowGameMs = 2500;
        await page.evaluate(() => window.dispatchEvent(new Event('online')));
        const seen = []; for (let k = 0; k < 25; k++) { seen.push((await lineText()).slice(0, 30) + '/' + (await page.evaluate(() => document.querySelector('.sv-hd-status')?.innerText ?? '-'))); await page.waitForTimeout(100); }
        console.log('note line during re-read:', [...new Set(seen)].join(' | '), '| asks:', host.asks.slice(-6).join(' '));
        const blanked = await page.waitForFunction(() => /실제 투구 기록 없음/.test(document.querySelector('.sv-game .sv-actual-line')?.innerText ?? ''), null, { timeout: 3000 }).then(() => true, () => false);
        ok(blanked, '① real screen: the re-read masks the line ("실제 투구 기록 없음") — the scene that replayed before the fix');
        const back = await page.waitForFunction((t) => (document.querySelector('.sv-game .sv-actual-line')?.innerText ?? '') === t, before, { timeout: 8000 }).then(() => true, () => false);
        host.slowGameMs = 0; host.checks = false;
        await page.waitForTimeout(1200);
        const s5 = await status(page);
        ok(back && s5.played === s4.played, `① real screen: same pitch back after the masked re-read → no replay (back ${back}, played ${s4.played} → ${s5.played})`);
      }
      ok(errors.length === 0, `fake: no page errors (${errors.length})`, errors.slice(0, 3));
      if (shots) await page.screenshot({ path: path.join(shots, 'fake-390.png') });
      // leaving the game (list) unmounts the 3D box
      await page.goto(origin + '/'); await page.locator('.sv-list').waitFor({ timeout: 15000 }); await page.waitForTimeout(500);
      ok(await page.locator('.pz-bat3d').count() === 0, 'fake: list screen has no 3D box');
      await context.close();
    }

    // ---------- ① deterministic DOM-level regression (synthetic page, same pz-batter.js, fake model) ----------
    {
      const { context, page, errors } = await open(390, 844);
      const A = '<span class="sv-al-label">실제 투구 · 3구 · Kim</span><i class="sv-g1 strike">3</i><b class="sv-actual-name">포심</b><span>헛스윙</span>';
      const B = '<span class="sv-al-label">실제 투구 · 4구 · Kim</span><i class="sv-g1 strike">4</i><b class="sv-actual-name">슬라이더</b><span>파울</span>';   // contact: the fake model (no meta) has no take clip
      const EMPTY = '<span class="sv-al-label">실제 투구 기록 없음</span>';
      const CONF = '<span class="sv-al-label">실제 투구 · 4구 · Kim</span><i class="sv-g1 unknown">4</i><b class="sv-actual-name">슬라이더</b><span>확인 중</span>';
      host.synthetic = `<!doctype html><html><head><meta charset="utf-8">${html.match(/<script type="importmap">[^<]*<\/script>/)[0]}<link rel="stylesheet" href="./pz/pz-theme.css"></head><body>
        <div class="sv-page sv-game"><main><div class="sv-zrow bat-r" style="position:relative;height:240px;--sv-zone-mid:50%"><div class="sv-batband at-right"><svg width="40" height="120"></svg></div><div class="sv-zone-wrap" style="position:relative"></div></div>
        <p class="sv-actual-line" id="line">${A}</p></main></div><script type="module" src="./pz/pz-batter.js"></script></body></html>`;
      await page.goto(`${origin}/__synthetic__.html`);
      const set = (h) => page.evaluate((h) => { document.getElementById('line').innerHTML = h; }, h);
      const att = await waited('① synthetic: attaches', () => page.waitForFunction(() => window.PZSongSkin?.status().attached === true, null, { timeout: 20000 }));
      if (att) {
        const played = async () => (await status(page)).played;
        const step = async (h) => { await set(h); await page.waitForTimeout(250); return played(); };
        const p0 = await played();
        ok(p0 === 0, `① synthetic: pitch on screen at attach does not play (${p0})`);
        ok(await step(EMPTY) === 0 && await step(A) === 0, '① synthetic: A → 기록 없음 → A does not replay');
        ok(await step(CONF) === 0 && await step(A) === 0, '① synthetic: A → 확인 중 → A does not replay');
        ok(await step(B) === 1, '① synthetic: A → B plays once');
        ok(await step(EMPTY) === 1 && await step(B) === 1, '① synthetic: B → 기록 없음 → B does not replay');
        ok(await step(A) === 2, '① synthetic: B → A (picked again) plays');
        await page.evaluate(() => { document.querySelector('.sv-zrow').className = 'sv-zrow bat-l'; }); await page.waitForTimeout(300);
        const st = await status(page);
        ok(st.side === 'L' && await page.locator('.pz-bat3d.side-r').count() === 0, '① synthetic: batter side follows bat-l');
      }
      // review P2-2 hole: 3D attaches while the line is a no-result line; the same pitch coming back must not play
      host.synthetic = host.synthetic.replace(/<p class="sv-actual-line" id="line">[\s\S]*?<\/p>/, `<p class="sv-actual-line" id="line">${EMPTY}</p>`);
      const page2 = await context.newPage(); page2.on('pageerror', (e) => errors.push(String(e)));
      await page2.goto(`${origin}/__synthetic__.html`);
      if (await waited('① synthetic (attach on a blank line): attaches', () => page2.waitForFunction(() => window.PZSongSkin?.status().attached === true, null, { timeout: 20000 }))) {
        const set2 = (h) => page2.evaluate((h) => { document.getElementById('line').innerHTML = h; }, h);
        const played2 = async () => (await page2.evaluate(() => window.PZSongSkin.status().played));
        await set2(A); await page2.waitForTimeout(250);
        ok(await played2() === 0, '① synthetic: attached on "기록 없음", then the pitch appears → no play (it was on screen before)');
        await set2(EMPTY); await page2.waitForTimeout(250); await set2(A); await page2.waitForTimeout(250);
        ok(await played2() === 0, '① synthetic: … → 기록 없음 → same pitch → still no play');
        await set2(B); await page2.waitForTimeout(250);
        ok(await played2() === 1, '① synthetic: a different pitch then plays once');
      }
      ok(errors.length === 0, `① synthetic: no page errors (${errors.length})`, errors.slice(0, 3));
      await context.close();
    }

    // ---------- ④ box geometry (real attach with fake model at each width, both sides by class) + zone painted above ----------
    for (const w of [320, 344, 359, 360, 390, 402]) {
      const { context, page, errors } = await open(w, w < 360 ? 640 : 760);
      await page.goto(`${origin}/?game=${PK}`);
      await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
      if (!(await waited(`④ ${w}: 3D attached`, () => page.waitForFunction(() => window.PZSongSkin?.status().attached === true, null, { timeout: 20000 })))) { await context.close(); continue; }
      for (const side of ['R', 'L']) {
        const r = await page.evaluate((side) => {
          const box = document.querySelector('.pz-bat3d'); box.classList.toggle('side-r', side === 'R');
          const b = box.getBoundingClientRect(), zw = document.querySelector('.sv-zrow .sv-zone-wrap');
          return { l: b.left, r: b.right, w: b.width, h: b.height, vw: innerWidth, sw: document.scrollingElement.scrollWidth - innerWidth, zBox: getComputedStyle(box).zIndex, zZone: getComputedStyle(zw).zIndex, posZone: getComputedStyle(zw).position };
        }, side);
        ok(r.l >= 0 && r.r <= r.vw && r.sw <= 0, `④ ${w} side ${side}: box inside the screen (${r.l.toFixed(1)}..${r.r.toFixed(1)} of ${r.vw}, side scroll ${r.sw})`);
        ok(r.w === 140 && r.h === 210, `④ ${w} side ${side}: box size unchanged 140×210 (${r.w}×${r.h})`);
        ok(Number(r.zZone) > Number(r.zBox) && r.posZone === 'relative', `④ ${w} side ${side}: zone (z ${r.zZone}) above the 3D box (z ${r.zBox})`);
      }
      // paint: fill the box red; the zone frame's right edge where it overlaps the box must still show the zone stroke, not red
      const probe = await page.evaluate(() => {
        const box = document.querySelector('.pz-bat3d'); box.classList.add('side-r');
        const red = document.createElement('div'); red.id = 'pz-red'; red.style.cssText = 'position:absolute;inset:0;background:#ff0000'; box.append(red);
        const zb = document.querySelector('.sv-zrow .sv-zone-box'); const z = zb.getBoundingClientRect(), b = box.getBoundingClientRect();
        return { x: z.right, y: (Math.max(z.top, b.top) + Math.min(z.bottom, b.bottom)) / 2, overlap: z.right > b.left && z.left < b.right };
      });
      if (probe.overlap) {
        const png = await page.screenshot({ clip: { x: probe.x - 1, y: probe.y - 1, width: 2, height: 2 } });   // 4×4 device px; (2,2) = the stroke centre (SVG bbox = geometry edge)
        const px = await page.evaluate(async (b64) => { const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode(); const c = document.createElement('canvas'); c.width = img.width; c.height = img.height; const x = c.getContext('2d'); x.drawImage(img, 0, 0); return [...x.getImageData(2, 2, 1, 1).data]; }, png.toString('base64'));
        ok(!(px[0] > 200 && px[1] < 60 && px[2] < 60), `④ ${w}: zone frame painted over the box where they overlap (pixel ${px.slice(0, 3).join(',')})`);
      } else ok(false, `④ ${w}: the box should reach the zone frame here (no overlap to paint-check)`);
      await page.evaluate(() => document.getElementById('pz-red')?.remove());
      ok(errors.length === 0, `④ ${w}: no page errors (${errors.length})`, errors.slice(0, 3));
      if (shots) await page.screenshot({ path: path.join(shots, `fake-${w}.png`) });
      await context.close();
    }
  }

  // ---------- ② what the page paints: list mark and zone mark of an in-play pitch, legend ----------
  {
    const { context, page } = await open(390, 844);
    await page.goto(`${origin}/?game=${PK}`);
    await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
    const c = await page.evaluate(() => {
      const g = document.querySelector('.sv-game');
      const i = document.createElement('i'); i.className = 'sv-g1 inplay'; i.textContent = '7'; g.append(i);
      const s = getComputedStyle(i), out = { listBg: s.backgroundColor, listFg: s.color };
      const ring = document.createElement('i'); ring.className = 'sv-g1 ball ring'; g.append(ring); out.ring = getComputedStyle(ring).boxShadow;
      const k = document.querySelector('.sv-zone-legend .k.inplay'); out.legend = k ? getComputedStyle(k).backgroundColor : null;
      const ac = document.querySelector('.sv-game .sv-actual.g1.inplay > circle:not(.sv-ring)'); out.zoneFill = ac ? getComputedStyle(ac).fill : null;
      const an = document.querySelector('.sv-game .sv-actual.g1.inplay .sv-g1-num'); out.zoneNum = an ? getComputedStyle(an).fill : null;
      i.remove(); ring.remove(); return out;
    });
    ok(cr(c.listBg, '#ffffff') >= 3, `② list in-play mark ${c.listBg} vs white ${cr(c.listBg, '#ffffff').toFixed(2)}:1 ≥ 3`);
    ok(cr(c.listFg, c.listBg) >= 4.5, `② list in-play number ${c.listFg} on ${c.listBg} ${cr(c.listFg, c.listBg).toFixed(2)}:1 ≥ 4.5`);
    const ringColor = (c.ring.match(/rgb\([^)]*\)/g) || []).pop();
    ok(ringColor && cr(ringColor, '#ffffff') >= 3, `② recent-pitch ring ${ringColor} vs white ${ringColor ? cr(ringColor, '#ffffff').toFixed(2) : '?'}:1 ≥ 3`);
    if (c.legend) ok(cr(c.legend, '#ffffff') >= 3, `② legend in-play dot ${c.legend} ${cr(c.legend, '#ffffff').toFixed(2)}:1 ≥ 3`); else ok(false, '② legend in-play dot present');
    if (c.zoneFill) { ok(cr(c.zoneFill, '#ffffff') >= 3, `② zone in-play mark ${c.zoneFill} ${cr(c.zoneFill, '#ffffff').toFixed(2)}:1 ≥ 3`); ok(c.zoneNum && cr(c.zoneNum, c.zoneFill) >= 4.5, `② zone in-play number ${c.zoneNum} ${c.zoneNum ? cr(c.zoneNum, c.zoneFill).toFixed(2) : '?'}:1 ≥ 4.5`); }
    else console.log('note ② zone in-play mark not on screen at this pitch — zone mark contrast not measured here (list / legend / token checks above)');
    await context.close();
  }

  // ---------- 10/26 show / hide rules with the skin on (handoff §8) ----------
  const notes = [];
  const look = async (page) => page.evaluate(() => {
    const main = document.querySelector('.sv-page'); const c = main.cloneNode(true); c.querySelectorAll('.sv-backdrop, .sv-sheet').forEach((x) => x.remove());
    const ab = document.querySelector('.sv-atbat-head')?.getBoundingClientRect(), nav = document.querySelector('.sv-bottom-nav')?.getBoundingClientRect();
    return { vScroll: document.scrollingElement.scrollHeight - innerHeight, hScroll: document.scrollingElement.scrollWidth - innerWidth,
      toggle: document.querySelector('.sv-atbat-toggle')?.getAttribute('aria-expanded') ?? null, list: document.querySelectorAll('.sv-atbat-list .sv-pitch-row').length,
      nav: [...document.querySelectorAll('.sv-bottom-nav button')].map((b) => b.innerText.trim()),
      switches: document.querySelectorAll('input[type=checkbox], input[type=range], [role=switch]').length,
      prevNext: [...c.querySelectorAll('button')].filter((b) => /^[‹›]$/.test(b.innerText.trim())).length,
      abBottom: ab ? ab.bottom : null, navTop: nav ? nav.top : null, text: c.innerText };
  });
  for (const pk of [PK, LIVE_PK]) for (const [w, h] of [[402, 681], [360, 620]]) {
    const id = `show/hide ${pk} ${w}x${h}`;
    const got = {};
    for (const which of ['skin', 'plain']) {
      const { context, page, errors } = await open(w, h);
      await page.goto(`${origin}${which === 'plain' ? '/__plain__' : ''}/?game=${pk}`);
      await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
      await page.waitForTimeout(1500);
      got[which] = await look(page);
      ok(errors.length === 0, `${id} ${which}: no page errors (${errors.length})`, errors.slice(0, 3));
      if (shots) await page.screenshot({ path: path.join(shots, `showhide-${which}-${pk}-${w}x${h}.png`) });
      await context.close();
    }
    const r = got.skin, p = got.plain;
    ok(r.vScroll <= p.vScroll, `${id}: skin adds no vertical scroll (skin ${r.vScroll}px, plain c47 ${p.vScroll}px)`);
    if (r.vScroll > 1 || p.vScroll > 1) notes.push(`${id}: first screen scrolls ${r.vScroll}px with the skin, ${p.vScroll}px without (rule: 0) — pre-existing in the plain build`);
    if (r.abBottom !== null && r.navTop !== null && r.abBottom > r.navTop) notes.push(`${id}: "이번 타석" line bottom ${r.abBottom.toFixed(0)} under the bottom bar top ${r.navTop.toFixed(0)} (plain: ${p.abBottom?.toFixed(0)} / ${p.navTop?.toFixed(0)})`);
    ok(r.hScroll <= 0, `${id}: no side scroll (${r.hScroll}px)`);
    ok(r.toggle === 'false' && r.list === 0, `${id}: "이번 타석" folded by default (aria-expanded ${r.toggle}, ${r.list} rows)`);
    ok(r.nav.length === 4 && r.nav.join('|') === '추천|선수|구종|기록', `${id}: bottom menu has 4 items (${r.nav.join('|')})`);
    ok(r.switches === 0, `${id}: no user switches (${r.switches})`);
    ok(r.prevNext === 0, `${id}: no ‹ › pitch buttons on the main screen (${r.prevNext})`);
    ok(!/스포일러|TV 중계보다|자동 진행|투구 애니메이션|라인업|선택 투구/.test(r.text), `${id}: no in-screen notice / V4 control words`);
    ok(r.text === p.text, `${id}: same visible text as the plain c47 screen`);
  }
  for (const n of notes) console.log('note', n);

  if (mode3d) {
  // ---------- reduced motion: attaches but never plays ----------
    {
      const { context, page, errors } = await open(390, 844, { reduceMotion: true });
      await page.goto(`${origin}/?game=${PK}`);
      await page.locator('.sv-game .sv-zrow').waitFor({ timeout: 15000 });
      if (await waited('reduced motion: attaches', () => page.waitForFunction(() => window.PZSongSkin?.status().attached === true, null, { timeout: 20000 }))) {
        await page.waitForTimeout(800);
        const st = await status(page);
        ok(st.state && st.state.idleLoop === false, `reduced motion: no idle loop (${st.state?.idleLoop})`);
      }
      ok(errors.length === 0, `reduced motion: no page errors (${errors.length})`, errors.slice(0, 3));
      await context.close();
    }
  }
} catch (e) {
  ok(false, `unexpected: ${String(e && e.stack || e).slice(0, 400)}`);
} finally {
  await browser.close(); server.close();
}
console.log(`\n${passes} ok, ${failures} FAIL`);
process.exit(failures ? 1 : 0);
