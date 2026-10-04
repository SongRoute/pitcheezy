// Local regression for the offline bundle's saved video copy (/showcase?media=local): serves a built dist with the real
// offline/serve_offline.py (--port 0) on a synthetic snapshot and a synthetic H.264/AAC mp4 made with ffmpeg, under the
// official file's name. Checks the default page still asks only the MLB URL (intercepted, never sent), that media=local
// asks /offline-media/<name> and gets 206, shows the local note, keeps the official link, keeps the scene param, and that
// the reveal stays hidden until clicked. Headless Chromium only: not Safari/iPhone, not the real MLB file.
// Usage: PZ_WEB_DIST=<built dist> node tests/showcase-local-media-smoke.mjs   (FFMPEG=<path> to override ffmpeg)
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST;
assert(dist, 'set PZ_WEB_DIST to a built dist outside any served directory');
const source = JSON.parse(readFileSync(path.join(here, '../src/showcase-scenes.json'), 'utf8'));
const [scene0, scene1] = source.scenes;
const name = new URL(source.mediaUrl).pathname.split('/').pop();
const local = `/offline-media/${name}`;
const near = (actual, expected, label) => assert(Math.abs(actual - expected) < .3, `${label}: ${actual} vs ${expected}`);

// Synthetic snapshot in the capture_snapshot.py layout (same fixture rows as showcase-smoke.mjs).
const tmp = mkdtempSync('/tmp/pz-offline-media-smoke-');
const decision = (index, ab, number) => ({ index, pa_id: `849843:${ab}`, at_bat_number: ab, pitch_number: number,
  situation: { inning: 1, half: 'Top', outs: 0, balls: 0, strikes: 0, bases: 0, home_score: 0, away_score: 0 },
  pitcher: { id: 1, name: 'Pitcher' }, batter: { id: 2, name: `Batter ${ab}`, side: 'R' },
  status: 'UNSUPPORTED', pre: { status: 'unsupported', reason: '합성 검사: 추천 없음', recommendation: null } });
const timeline = { schema: 'pitcheezy-watch-along-v1', badge: '검증 전 실험 버전',
  game: { game_pk: 849843, away_team: 'Chicago Cubs', home_team: 'San Diego Padres', date: '2026-09-29' },
  decisions: [decision(0, 1, 1), decision(1, 1, 2), decision(2, 1, 3), decision(3, 2, 5)],
  policy: { name: 'synthetic', identity_sha256: '0'.repeat(64), tau: .1, note: '합성 검사' },
  location: { note: '근사', zone_bounds: { bottom: 1.6, top: 3.39 } }, we_note: '합성 검사',
  coverage: { ready: 0, pitch_decisions: 4 } };
const reveal = index => ({ index, we: { home_before: .5, home_after: .5, home_delta: 0 },
  actual: { pitch_type: 'FF', pitch_label: `합성 실제 투구 ${index}`, result_label: '볼', event_label: null, play_text: null,
    speed_mph: 95, x: 0, z: 2.5, zone_label: 'heart' } });
const sha = b => createHash('sha256').update(b).digest('hex');
const snapshotFile = p => `api/${p.replace(/^\/+|\/+$/g, '').replaceAll('/', '__')}.json`;  // capture_snapshot.snapshot_file (no query)
const snap = path.join(tmp, 'snapshot');
mkdirSync(path.join(snap, 'api'), { recursive: true });
const files = Object.entries({ '/api/watch/849843': timeline, '/api/watch/849843/reveal/2': reveal(2),
  '/api/watch/849843/reveal/3': reveal(3) }).map(([p, value]) => {
  const body = Buffer.from(JSON.stringify(value));
  writeFileSync(path.join(snap, snapshotFile(p)), body);
  return { path: p, file: snapshotFile(p), bytes: body.length, sha256: sha(body) };
});
writeFileSync(path.join(snap, 'manifest.json'), JSON.stringify({ game_pk: 849843, scenes: 2, captured_kst: '합성 검사', files }));

// Synthetic mp4 long enough for scene 1's clip (release + 4 s), under the official file name.
const mediaDir = path.join(tmp, 'media');
mkdirSync(mediaDir);
const duration = Math.ceil(scene1.releaseSeconds + 6);
execFileSync(process.env.FFMPEG ?? '/opt/homebrew/bin/ffmpeg', ['-v', 'error', '-threads', '1',
  '-f', 'lavfi', '-i', `testsrc=duration=${duration}:size=160x90:rate=5`, '-f', 'lavfi', '-i', `sine=duration=${duration}`,
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart', path.join(mediaDir, name)]);
const mp4 = readFileSync(path.join(mediaDir, name));
writeFileSync(path.join(mediaDir, 'MEDIA-SHA256SUMS'), `${sha(mp4)}  ${name}\n`);

const server = spawn('python3', [path.join(here, '../offline/serve_offline.py'), '--dist', dist, '--snapshot', snap,
  '--media-dir', mediaDir, '--port', '0'], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUNBUFFERED: '1' } });
let serverLog = '';
server.stderr.on('data', d => { serverLog += d; });
const base = await new Promise((resolve, reject) => {
  let out = '';
  server.stdout.on('data', d => { out += d; const m = /(http:\/\/127\.0\.0\.1:\d+)\/showcase /.exec(out); if (m) resolve(m[1]); });
  server.on('exit', code => reject(new Error(`serve_offline.py exited ${code}: ${serverLog}`)));
});

const report = { tmp, mp4Bytes: mp4.length, decoded: {} };
let browser;
try {
  browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });  // inside try: a failed launch still stops the server
  // Direct HTTP: one range, an unknown name.
  let r = await fetch(`${base}${local}`, { headers: { Range: 'bytes=0-1' } });
  assert.equal(r.status, 206); assert.equal(r.headers.get('content-range'), `bytes 0-1/${mp4.length}`);
  assert.equal((await fetch(`${base}/offline-media/unknown.mp4`)).status, 404);

  async function open(query, width, { failLocal = false } = {}) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const seen = { mlb: 0, offline: [], other: 0, reveals: [], errors: [] };
    page.on('pageerror', e => seen.errors.push(String(e)));
    page.on('response', res => { if (res.url().includes('/offline-media/')) seen.offline.push({ status: res.status(), range: res.headers()['content-range'] }); });
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.href === source.mediaUrl) { seen.mlb++; return route.abort(); }  // never sent to MLB
      if (url.origin !== base) { seen.other++; return route.abort(); }
      if (failLocal && url.pathname.startsWith('/offline-media/')) return route.abort();
      const m = url.pathname.match(/\/reveal\/(\d+)$/);
      if (m) seen.reveals.push(Number(m[1]));
      return route.continue();
    });
    await page.goto(`${base}/showcase${query}`);
    await page.locator('.sc-compare iframe').waitFor();
    return { page, seen };
  }
  const note = '맥북에 저장한 같은 공식 영상 사본을 씁니다';
  const alertOf = page => page.getByRole('alert').filter({ hasText: '영상 재생을 확인하지 못했어요' });

  // A. Default: MLB URL only (intercepted), no local request, no local note.
  {
    const { page, seen } = await open('?scene=1', 1440);
    assert.equal(await page.getByText(note).count(), 0, 'no local note by default');
    await page.getByRole('button', { name: '공식 경기 영상 불러오기' }).click();
    await alertOf(page).waitFor();
    assert.equal(await page.locator('video').getAttribute('src'), source.mediaUrl, 'default src is the MLB URL');
    assert(seen.mlb >= 1, 'default asks the MLB URL'); assert.equal(seen.offline.length, 0, 'default never asks /offline-media/');
    assert.equal(await page.getByRole('link', { name: '공식 영상 따로 열기' }).getAttribute('href'), source.mediaUrl);
    assert.equal(seen.other, 0); assert.deepEqual(seen.errors, []);
    await page.close();
  }

  // B. media=local + scene: local src, 206, note, scene and reveal behaviour unchanged.
  for (const width of [1440, 390]) {
    const { page, seen } = await open('?media=local&scene=1', width);
    const frame = page.frameLocator('.sc-compare iframe');
    await page.getByText(note).waitFor();
    assert.equal(seen.offline.length, 0, 'video loads only on viewer request');
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=3', 'scene=1 kept with media=local');
    let params = new URL(page.url()).searchParams;
    assert.equal(params.get('media'), 'local'); assert.equal(params.get('scene'), '1');
    await page.getByRole('button', { name: '공식 경기 영상 불러오기' }).click();
    const outcome = await Promise.race([
      page.waitForFunction(() => !document.querySelector('.sc-primary')?.disabled, null, { timeout: 15000 }).then(() => 'ready'),
      alertOf(page).waitFor({ timeout: 15000 }).then(() => 'error')]);
    assert.equal(await page.locator('video').getAttribute('src'), local, 'local src');
    assert(seen.offline.length >= 1 && seen.offline.every(x => x.status === 206), `local media answered 206: ${JSON.stringify(seen.offline)}`);
    assert(/^bytes \d+-\d+\/\d+$/.test(seen.offline[0].range) && seen.offline[0].range.endsWith(`/${mp4.length}`));
    assert.equal(seen.mlb, 0, 'local mode never asks the MLB URL');
    report.decoded[width] = outcome;
    if (outcome === 'ready') {  // decodable here: same scene timing as the MLB file
      const media = () => page.locator('video').evaluate(v => ({ t: v.currentTime, paused: v.paused }));
      let m = await media(); near(m.t, scene1.releaseSeconds - 3, 'metadata seeks to scene start'); assert(m.paused);
      await page.getByRole('button', { name: '이 공 다시 보기' }).click();
      await frame.getByRole('button', { name: '실제 투구 공개' }).waitFor();
      m = await media(); near(m.t, scene1.releaseSeconds - 3, 'clip start'); assert(!m.paused, 'clip plays');
      await page.locator('video').evaluate((v, t) => { v.currentTime = t; }, scene1.releaseSeconds + 4.05);
      await page.waitForFunction(() => document.querySelector('video').paused);
    } else {
      assert.equal(await page.getByRole('link', { name: '공식 영상 따로 열기' }).getAttribute('href'), source.mediaUrl);
    }
    // Reveal: nothing before the click, the scene's own pitch after it.
    await frame.getByText('합성 검사: 추천 없음').waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'no actual pitch before explicit reveal');
    assert.deepEqual(seen.reveals, [], 'no reveal request before the click');
    if (width < 800) await page.locator('.sc-compare').evaluate(e => e.scrollIntoView({ block: 'end' }));
    else await page.evaluate(() => window.scrollTo(0, 0));
    await frame.getByRole('button', { name: '실제 투구 공개' }).click();
    await frame.locator('.wa-reveal').getByText('합성 실제 투구 3').waitFor();
    assert.deepEqual(seen.reveals, [3]);
    // Scene change keeps media=local and the local source.
    await page.getByLabel('비교할 장면').selectOption('0');
    await frame.locator('.wa-pitchno').getByText('3구째').waitFor();
    params = new URL(page.url()).searchParams;
    assert.equal(params.get('media'), 'local'); assert.equal(params.get('scene'), '0');
    assert.equal(await page.locator('video').getAttribute('src'), local);
    if (outcome === 'ready') near(await page.locator('video').evaluate(v => v.currentTime), scene0.releaseSeconds - 3, 'scene 0 seek');
    assert.equal(seen.other, 0); assert.deepEqual(seen.errors, []);
    await page.close();
  }

  // C. media=local with the local file unreachable: error keeps the official link and an in-place retry.
  {
    const { page, seen } = await open('?media=local&scene=1', 1440, { failLocal: true });
    await page.getByRole('button', { name: '공식 경기 영상 불러오기' }).click();
    await alertOf(page).waitFor();
    assert.equal(await page.getByRole('link', { name: '공식 영상 따로 열기' }).getAttribute('href'), source.mediaUrl);
    assert.equal(await alertOf(page).getByRole('button', { name: '영상 다시 불러오기' }).count(), 1);
    assert.equal(await page.locator('video').getAttribute('src'), local);
    assert.equal(seen.mlb, 0); assert.equal(seen.other, 0); assert.deepEqual(seen.errors, []);
    await page.close();
  }
  console.log(`showcase local-media smoke: ok ${JSON.stringify(report)}`);
} finally {
  await browser?.close();
  server.kill();
}
