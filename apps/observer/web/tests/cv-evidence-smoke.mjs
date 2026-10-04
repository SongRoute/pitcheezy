// Synthetic local check of the CV M3 evidence card on /showcase: the pinned JSON against the PM receipt (when its path is
// given), denominators and limitation wording, collapsed by default, no video/recommendation links, desktop + phone layout.
// Usage: node tests/cv-evidence-smoke.mjs [screenshot-dir]   env: DIST (default ../dist), CV_M3_RECEIPT (optional receipt JSON)
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { createHash } from 'node:crypto';
import { readFileSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.DIST ?? path.join(here, '..', 'dist');
const ev = JSON.parse(readFileSync(path.join(here, '../src/cv-m3-evidence.json'), 'utf8'));
const hypot = ([a, b]) => Math.hypot(a, b);

// Internal consistency: pooled = sum of games; abstentions are counted, not dropped.
for (const k of ['frames', 'aiOutput', 'aiAbstained', 'compared', 'personMarked', 'aiAbstainedPersonMarked'])
  assert.equal(ev.games.reduce((s, g) => s + g[k], 0), ev.pooled[k], `pooled ${k}`);
for (const g of [ev.pooled, ...ev.games]) assert.equal(g.aiOutput + g.aiAbstained, g.frames, 'output + abstain = frames');
assert(ev.examples.every(e => e.videoLinked === false), 'no example is linked to video');

if (process.env.CV_M3_RECEIPT) {  // exact identity with the PM receipt (no rounding in the pinned copy)
  const raw = readFileSync(process.env.CV_M3_RECEIPT), r = JSON.parse(raw);
  assert.equal(createHash('sha256').update(raw).digest('hex'), ev.source.receiptSha256, 'receipt sha256');
  assert.equal(r.source_commit, ev.source.teamCommit);
  const pc = r.pooled_counts, pd = r.pooled_distances;
  assert.deepEqual([pc.frames, pc.assistant_estimated, pc.assistant_abstained, pc.both_marked, pc.person_marked, pc.person_only],
    [ev.pooled.frames, ev.pooled.aiOutput, ev.pooled.aiAbstained, ev.pooled.compared, ev.pooled.personMarked, ev.pooled.aiAbstainedPersonMarked]);
  assert.deepEqual([pd.pixels.median, pd.pixels.p90, pd.output_feet.median, pd.output_feet.p90],
    [ev.pooled.pixels.median, ev.pooled.pixels.p90, ev.pooled.outputFeet.median, ev.pooled.outputFeet.p90]);
  for (const g of ev.games) {
    const rg = r.per_game.find(x => x.game_pk === g.gamePk), c = rg.counts, d = rg.distances;
    assert.deepEqual([c.frames, c.assistant_estimated, c.assistant_abstained, c.both_marked, d.pixels.median, d.pixels.p90, d.output_feet.median, d.output_feet.p90,
      rg.camera_check_reported.error_status, rg.camera_check_reported.rms_feet],
    [g.frames, g.aiOutput, g.aiAbstained, g.compared, g.pixels.median, g.pixels.p90, g.outputFeet.median, g.outputFeet.p90, g.cameraCheck.status, g.cameraCheck.rmsFeet], `game ${g.gamePk}`);
  }
  const frame = Object.fromEntries(r.per_frame.map(f => [f.pitch_id, f]));
  assert.deepEqual(ev.examples.slice(0, 3).map(e => e.pitchId), [r.examples.representative, r.examples.best, r.examples.worst]);
  for (const e of ev.examples) {
    // Derived in Python; JS Math.hypot may differ in the last bit.
    assert(Math.abs(e.pixels - hypot(frame[e.pitchId].pixels)) < 1e-12 && Math.abs(e.outputFeet - hypot(frame[e.pitchId].output_feet)) < 1e-12, e.pitchId);
  }
  assert.equal(r.service_validator.rows_passed, ev.source.validatorRowsPassed);
  console.log('receipt identity: ok');
} else console.log('receipt identity: skipped (CV_M3_RECEIPT not set)');

const timeline = { schema: 'pitcheezy-watch-along-v1', badge: '검증 전 실험 버전',
  game: { game_pk: 849843, away_team: 'Chicago Cubs', home_team: 'San Diego Padres', date: '2026-09-29' }, decisions: [],
  policy: { name: 'synthetic', identity_sha256: '0'.repeat(64), tau: .1, note: '합성 검사' },
  location: { note: '근사', zone_bounds: { bottom: 1.6, top: 3.39 } }, we_note: '합성 검사', coverage: { ready: 0, pitch_decisions: 0 } };
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const browser = await chromium.launch();
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = []; let outside = 0;
    page.on('pageerror', e => errors.push(String(e)));
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'showcase.test') { outside++; return route.abort(); }
      if (url.pathname === '/api/watch/849843') return route.fulfill({ json: timeline });
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'synthetic fixture' } });
      const file = path.join(dist, url.pathname);
      const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
      return route.fulfill({ body: readFileSync(target), contentType: types[path.extname(target)] ?? 'application/octet-stream' });
    });
    await page.goto('http://showcase.test/showcase?scene=0');
    const card = page.locator('details.cv-card');
    await card.waitFor();
    assert.equal(await card.evaluate(d => d.open), false, 'collapsed by default');
    const summary = await card.locator('summary').innerText();
    for (const t of ['CV 검증 결과', '평가 2경기 · 86장', 'AI 출력 58 / 기권 28', '사람과 비교한 58구의 미트 점 차이 중앙값 2.5px']) assert(summary.includes(t), `summary: ${t}`);
    assert(await card.locator('.cv-games').isHidden(), 'detail hidden while collapsed');
    await card.locator('summary').click();
    await card.locator('.cv-games').waitFor();
    const text = await card.innerText();
    for (const t of ['두 경기 86장 중 58장에서 좌표를 냈고', '아직 실제 의도나 목표 높이를 검증한 것은 아닙니다',
      'AI 출력 43/57장 · 기권 14장', 'AI 출력 15/29장 · 기권 14장', '미트 점 차이 중앙값 2.1px', '미트 점 차이 중앙값 5.8px',
      '출력 좌표 차이 중앙값 약 1.1인치', '출력 좌표 차이 중앙값 약 1.7인치', '카메라 미확인', '포구 공 13구 대조 RMS 0.15ft',
      'AI가 기권한 28장 중 26장은 사람이 미트를 표시', '높이는 미검증', '사람 판독자는 1명', '605구 중 86구',
      '849845:10:4', '849845:23:6', '849845:38:6', '823407:50:1', 'FOX 대표 · 카메라 미확인', '이동하지 않습니다', '팀 커밋 1687c1e'])
      assert(text.includes(t), `detail: ${t}`);
    // Every inch figure carries the same-transform disclaimer in the same block.
    assert.equal(await card.locator('em').filter({ hasText: '같은 변환 기준의 판독 차이 · 물리 정확도 아님' }).count(), 3);
    assert.equal(await card.locator('a, button, iframe, video').count(), 0, 'no navigation from the evidence card');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal overflow');
    const box = await card.boundingBox(); assert(box.x >= 0 && box.x + box.width <= width, 'card within viewport');
    if (process.argv[2]) {
      await card.scrollIntoViewIfNeeded();
      await card.screenshot({ path: `${process.argv[2]}/cv-card-open-${width}.png` });
      await page.screenshot({ path: `${process.argv[2]}/cv-card-page-${width}.png`, fullPage: true });
    }
    await card.locator('summary').click();
    assert.equal(await card.evaluate(d => d.open), false, 'collapses again');
    assert.equal(outside, 0, 'no request leaves the synthetic host'); assert.deepEqual(errors, []);
    await page.close();
  }
  console.log('cv evidence smoke: ok (desktop + phone, synthetic timeline)');
} finally { await browser.close(); }
