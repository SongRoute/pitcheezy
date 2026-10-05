// D2 (ui-calm-polish-1006): the 이닝별 기록 sheet of a SELECTED past pitch shows that pitch's `linescore_before`, never the
// latest linescore — a future scoring event (home 7 runs in the 1st, after the selected pitch) must not appear. A pitch without
// the field shows the 'not ready' text, not the latest line. Route-mocked on tests/fixtures/service-849843.json (final) with
// in-memory linescores; no server, screenshots only to argv[2]. Headless Chromium, not Safari.
//   npx vite build --outDir <tmp> && PZ_WEB_DIST=<tmp> node tests/boxscore-asof-check.mjs [outDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const out = process.argv[2]; if (out) mkdirSync(out, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
let failures = 0;
const ok = (cond, msg, info) => { if (!cond) failures++; console.log(cond ? 'ok  ' : 'FAIL', msg, cond ? '' : JSON.stringify(info)); };
const line = (rows, away, home) => ({ basis: 'revealed_events', innings: rows.map(([a, h], i) => ({ inning: i + 1, away: a, home: h })),
  teams: { away: { runs: away[0], hits: away[1], errors: away[2], bb_hbp: away[3] }, home: { runs: home[0], hits: home[1], errors: home[2], bb_hbp: home[3] } } });
const body = JSON.parse(readFileSync(path.join(here, 'fixtures', 'service-849843.json'), 'utf8')).final;
body.linescore = line([[0, 7], [0, 0], [0, 1]], [0, 2, 0, 1], [8, 11, 1, 3]);           // latest: home scored 7 in the 1st (later play)
body.pitches.forEach((p) => { if (p.index <= 1) p.linescore_before = line([[0, null]], [0, 0, 0, 0], [0, 0, 0, 0]); });  // 1회초 before the scoring
// index 2 deliberately has no linescore_before (older backend): must not fall back to the latest line

const browser = await chromium.launch();
const c = await browser.newContext({ viewport: { width: 375, height: 667 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: 'ko-KR' });
const p = await c.newPage();
await p.route('http://pz.test/**', (route) => {
  const u = new URL(route.request().url());
  if (u.pathname === '/api/service/games/849843') return route.fulfill({ json: body });
  if (u.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this test' } });
  const f = path.join(dist, u.pathname); const t = existsSync(f) && statSync(f).isFile() ? f : path.join(dist, 'index.html');
  return route.fulfill({ body: readFileSync(t), contentType: TYPES[path.extname(t)] ?? 'application/octet-stream' });
});
await p.goto('http://pz.test/watch/849843'); await p.waitForSelector('.sv-board'); await p.waitForTimeout(250);
const box = async (shot) => { await p.locator('.sv-inning').click(); await p.waitForSelector('.sv-sheet');
  const t = await p.evaluate(() => ({ rows: [...document.querySelectorAll('.sv-box tbody tr')].map((r) => [...r.children].map((x) => x.textContent)), text: document.querySelector('.sv-sheet-body').textContent }));
  if (out) await p.screenshot({ path: path.join(out, shot) }); await p.keyboard.press('Escape'); await p.waitForTimeout(150); return t; };
await p.locator('.sv-strip .sv-card').nth(1).click(); await p.waitForTimeout(200);   // selected: 1회초 2구 (has linescore_before)
let b = await box('asof-selected.png');
ok(b.rows.length === 2, 'selected pitch shows a box score', b);
ok(b.rows[1]?.[1] === '–' && !b.rows[1].includes('7') && !b.rows[1].includes('8'), 'home 1st inning not yet played: –, no later 7/8 runs', b.rows[1]);
ok(b.rows[1]?.slice(-4).join(',') === '0,0,0,0', 'home R/H/E/사사구 as of the selected pitch', b.rows[1]);
await p.locator('.sv-strip .sv-card').nth(2).click(); await p.waitForTimeout(200);   // index 2: no field
b = await box('asof-missing.png');
ok(b.rows.length === 0 && /준비 중/.test(b.text) && !/\b8\b/.test(b.text.replace(/\d+회/g, '')), 'missing field → not-ready text, never the latest line', b);
await p.locator('.sv-live').click(); await p.waitForTimeout(300);                    // latest view keeps the latest linescore
b = await box('asof-latest.png');
ok(b.rows[1]?.[1] === '7' && b.rows[1]?.includes('8'), 'latest view still shows the latest linescore', b.rows[1]);
await browser.close();
console.log(failures ? `${failures} FAIL` : 'all ok'); process.exit(failures ? 1 : 0);
