// API demo cache regressions rescued from the obsolete service-ui-smoke interaction flow.
// Existing fixture, synthetic session reset, normal polling only; no product/API/fixture-file changes.
// npm run build && node tests/demo-cache-smoke.mjs
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url)), dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const fx = JSON.parse(readFileSync(path.join(here, 'fixtures/service-849843.json'), 'utf8'));
const reset = structuredClone(fx.live);
reset.source.session_id = 'other';
reset.pitches = reset.pitches.slice(0, 5); reset.pas = reset.pas.slice(0, 2);
reset.cutoff.index = 99999; // forward cursor: session identity, not cutoff regression, must cause the full refetch
const last = reset.pitches.at(-1);
reset.current = { ...reset.current, pa_key: last.pa_key, batter: last.batter, pitcher: last.pitcher, situation: last.situation };
let current = fx.live, phase = 'initial', passed = 0, failed = 0;
const asks = [], errors = [], types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const ok = (c, text, detail) => { c ? passed++ : failed++; console.log(c ? 'ok  ' : 'FAIL', text, c ? '' : JSON.stringify(detail)); };
const browser = await chromium.launch(), page = await browser.newPage({ viewport: { width: 390, height: 844 }, locale: 'ko-KR' });
try {
  page.on('pageerror', e => errors.push(String(e)));
  await page.route('**/*', route => {
    const u = new URL(route.request().url()); if (u.origin !== 'http://pz.test') return route.abort();
    if (u.pathname.startsWith('/api/')) {
      asks.push({ phase, after: u.searchParams.get('after'), demo: u.searchParams.get('demo') });
      if (u.pathname !== '/api/service/games/849843') return route.fulfill({ status: 404, json: {} });
      const answer = structuredClone(current), after = u.searchParams.get('after');
      if (after !== null) answer.pitches = answer.pitches.filter(p => p.index > Number(after));
      return route.fulfill({ json: answer });
    }
    const file = path.join(dist, u.pathname), asset = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(asset), contentType: types[path.extname(asset)] ?? 'application/octet-stream' });
  });
  await page.goto('http://pz.test/watch/849843?demo=fixture');
  await page.locator('.sv-zone-wrap .sv-zone').waitFor();
  // This sheet counts every cached pitch for the displayed pitcher, including earlier plate appearances.
  await page.locator('.sv-bottom-nav').getByRole('button', { name: '구종', exact: true }).click();
  const count = payload => payload.pitches.filter(p => p.pitcher.id === payload.current.pitcher.id && p.actual?.pitch_type).length;
  const shownCount = async expected => {
    await page.waitForFunction(n => document.querySelector('.sv-sheet-body > p')?.textContent.includes(`구종이 확인된 ${n}구 기준`), expected, { timeout: 7000 });
    return page.locator('.sv-sheet-body > p').first().innerText();
  };
  await shownCount(count(current));
  ok(asks[0].after === null, 'initial load has no incremental cursor', asks);
  phase = 'more'; current = fx.live_more;
  await shownCount(count(current));
  ok(asks.some(a => a.phase === 'more' && a.after === String(fx.live_cut)), 'new pitches are requested incrementally from the prior cursor', asks);
  ok(asks.filter(a => a.phase === 'more').every(a => a.after !== null), 'incremental merge retains earlier pitches without a full refetch', asks);
  phase = 'back'; current = fx.live;
  await shownCount(count(current));
  const back = asks.filter(a => a.phase === 'back');
  ok(back.length >= 2 && back[0].after === String(fx.live_more.cutoff.index) && back.some(a => a.after === null), 'backward cutoff triggers a full refetch', back);
  ok((await shownCount(count(current))).includes(`${count(current)}구`), 'seek back removes later cached pitches');
  phase = 'new-session'; current = reset;
  await shownCount(count(current));
  const session = asks.filter(a => a.phase === 'new-session');
  ok(session.length >= 2 && session[0].after === String(fx.live_cut) && session.some(a => a.after === null), 'new session with a forward cutoff triggers a full refetch', session);
  ok((await shownCount(count(reset))).includes(`${count(reset)}구`), 'new session drops old cached pitches');
  await page.keyboard.press('Escape');
  await page.locator('.sv-bottom-nav').getByRole('button', { name: '기록', exact: true }).click();
  ok(await page.getByRole('dialog').locator('.sv-recent-pas > li').count() === reset.pas.length, 'new session also replaces the old plate appearances');
  ok(asks.every(a => a.demo === 'fixture'), 'every API request preserves the URL demo binding', asks);
  ok(errors.length === 0, 'no page errors', errors);
} finally { await browser.close(); }
console.log(`${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
