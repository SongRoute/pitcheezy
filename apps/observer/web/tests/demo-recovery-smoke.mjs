// Isolated demo-route recovery check on the built UI; no running service or model is touched.
// npm run build && node tests/demo-recovery-smoke.mjs
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(dir, '..', 'dist');
const fixture = JSON.parse(readFileSync(path.join(dir, 'fixtures', 'service-849843.json'), 'utf8'));
const date = '2026-09-30';
const game = fixture.live.game;
const browser = await chromium.launch();
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };

for (const [width, height] of [[320, 568], [375, 553]]) {
  const context = await browser.newContext({ viewport: { width, height }, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  const errors = []; page.on('pageerror', e => errors.push(String(e)));
  let failList = true, failDetail = true, detailCalls = 0;
  await page.route('http://pz.test/**', route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/service/dates') return route.fulfill({ json: { dates: [date] } });
    if (url.pathname === '/api/service/games') return failList
      ? route.fulfill({ status: 503, json: { detail: '목록 연결에 실패했어요.' } })
      : route.fulfill({ json: { date, games: [game], live_ok: true } });
    if (url.pathname === '/api/service/games/849843') {
      detailCalls++;
      if (failDetail) return route.fulfill({ status: 503, json: { detail: '기록 연결에 실패했어요.' } });
      const body = structuredClone(fixture.live);
      const after = url.searchParams.get('after');
      if (after !== null) body.pitches = body.pitches.filter(p => p.index > Number(after));
      return route.fulfill({ json: body });
    }
    if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in fixture' } });
    const file = path.join(dist, url.pathname);
    const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
    return route.fulfill({ body: readFileSync(target), contentType: mime[path.extname(target)] ?? 'application/octet-stream' });
  });

  await page.goto(`http://pz.test/watch?date=${date}`);
  await page.getByRole('alert').getByText('목록 연결에 실패했어요.').waitFor();
  const listRetry = page.getByRole('button', { name: '다시 시도' });
  let box = await listRetry.boundingBox();
  assert(box && box.width >= 44 && box.height >= 44 && box.x + box.width <= width && box.y + box.height <= height, `${width}: list retry visible and tappable ${JSON.stringify(box)}`);
  failList = false;
  await listRetry.click();
  await page.locator('.sv-card[href="/watch/849843"]').waitFor();
  assert.equal(await page.getByRole('alert').count(), 0, `${width}: list error cleared`);

  await page.locator('.sv-card[href="/watch/849843"]').click();
  await page.getByRole('alert').getByText('기록 연결에 실패했어요.').waitFor();
  const detailRetry = page.getByRole('button', { name: '다시 시도' });
  box = await detailRetry.boundingBox();
  assert(box && box.width >= 44 && box.height >= 44 && box.x + box.width <= width && box.y + box.height <= height, `${width}: detail retry visible and tappable ${JSON.stringify(box)}`);
  failDetail = false;
  await detailRetry.click();
  await page.locator('.sv-zone-wrap').waitFor();
  assert.equal(await page.getByRole('alert').count(), 0, `${width}: detail error cleared`);
  assert(detailCalls >= 2, `${width}: detail retried`);
  assert.deepEqual(errors, [], `${width}: no page errors`);
  await context.close();
  console.log(`ok ${width}x${height} list and detail retry`);
}

await browser.close();
