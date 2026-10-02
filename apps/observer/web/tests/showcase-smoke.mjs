// Synthetic local regression: video/scene identity, no pre-reveal result, missing mapping, network failure.
// The real MLB video is never downloaded by this test; playback verification is a separate rehearsal.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { readFileSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, '..', 'dist');
const source = JSON.parse(readFileSync(path.join(here, '../src/showcase-scenes.json'), 'utf8'));
assert.equal(new Set(source.scenes.map(s => s.pitchId)).size, 39);
assert(source.scenes.every(s => Number.isFinite(s.releaseSeconds) && s.releaseSeconds > 3));
const decision = (index, ab, number) => ({ index, pa_id: `849843:${ab}`, at_bat_number: ab, pitch_number: number,
  situation: { inning: 1, half: 'Top', outs: 0, balls: 0, strikes: 0, bases: 0, home_score: 0, away_score: 0 },
  pitcher: { id: 1, name: 'Pitcher', hand: 'R' }, batter: { id: 2, name: `Batter ${ab}`, side: 'R' },
  status: 'UNSUPPORTED', pre: { status: 'unsupported', reason: '합성 검사: 추천 없음', recommendation: null } });
const timeline = { schema: 'pitcheezy-watch-along-v1', badge: '검증 전 실험 버전',
  game: { game_pk: 849843, away_team: 'Chicago Cubs', home_team: 'San Diego Padres', date: '2026-09-29' },
  decisions: [decision(0, 1, 1), decision(1, 1, 2), decision(2, 1, 3), decision(3, 2, 5)],
  policy: { name: 'synthetic', identity_sha256: '0'.repeat(64), tau: .1, note: '합성 검사' },
  location: { note: '근사', zone_bounds: { bottom: 1.6, top: 3.39 } }, we_note: '합성 검사',
  coverage: { ready: 0, pitch_decisions: 4 } };
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const browser = await chromium.launch();
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    let failApi = false, videoRequests = 0;
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'showcase.test') { videoRequests++; return route.abort(); }
      if (url.pathname === '/api/watch/849843') return route.fulfill({ status: failApi ? 503 : 200, json: failApi ? { detail: '자료 없음' } : timeline });
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'synthetic fixture' } });
      const file = path.join(dist, url.pathname);
      const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
      return route.fulfill({ body: readFileSync(target), contentType: types[path.extname(target)] ?? 'application/octet-stream' });
    });
    await page.goto('http://showcase.test/showcase?scene=0');
    await page.locator('.sc-compare iframe').waitFor();
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=2', 'use pitch identity, not scene ordinal');
    assert.equal(videoRequests, 0, 'video loads only on viewer request');
    const frame = page.frameLocator('.sc-compare iframe');
    await frame.getByText('합성 검사: 추천 없음').waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'no actual pitch before explicit reveal');
    assert.equal(await frame.locator('.wa-guide').count(), 0, 'outer instructions replace embedded onboarding');
    assert.equal(await frame.locator('.wa-innings, .wa-nav, .wa-step').count(), 0, 'embedded scene cannot navigate independently');
    await frame.locator('.wa-page').press('ArrowLeft');
    const inner = page.frames().find(f => f.url().includes('/watch?'));
    assert(inner.url().endsWith('&embed=1#i=2'), 'keyboard cannot change embedded pitch');
    await page.getByLabel('비교할 장면').selectOption('1');
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=3');
    await page.getByLabel('비교할 장면').selectOption('2');
    await page.getByText('이 장면에 대응하는 투구 자료가 없습니다. 다른 장면을 선택해 주세요.').waitFor();
    assert.equal(await page.locator('iframe').count(), 0, 'missing mapping cannot retain prior pitch');
    await page.getByLabel('비교할 장면').selectOption('0');
    await page.getByRole('button', { name: '공식 경기 영상 불러오기' }).click();
    await page.getByRole('alert').filter({ hasText: '영상 재생을 확인하지 못했어요' }).waitFor();
    assert.equal(await page.getByRole('button', { name: '이 공 다시 보기' }).isDisabled(), true);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal overflow');
    if (process.argv[2]) await page.screenshot({ path: `${process.argv[2]}/showcase-${width}.png`, fullPage: true });
    failApi = true;
    await page.goto('http://showcase.test/showcase?scene=invalid');
    await page.locator('.sc-compare [role=alert]').waitFor();
    assert.equal(await page.locator('iframe').count(), 0);
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log('showcase smoke: ok (synthetic video failure and timeline fixtures, desktop + phone)');
} finally { await browser.close(); }
