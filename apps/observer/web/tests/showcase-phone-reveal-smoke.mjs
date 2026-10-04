// Synthetic local regression for the phone-width showcase: the comparison is stacked below the video, so its fixed
// reveal bar starts below the fold. Checks the jump buttons and the clip-end scroll bring the reveal bar into view,
// that neither reveals anything or requests a reveal, that a scene change mid-clip does not scroll, and that desktop
// is unchanged (jump buttons hidden, no scroll at clip end). Headless Chromium emulation only: not iPhone/Safari.
// Usage: PZ_WEB_DIST=<built dist> node tests/showcase-phone-reveal-smoke.mjs [screenshot dir]
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { readFileSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const source = JSON.parse(readFileSync(path.join(here, '../src/showcase-scenes.json'), 'utf8'));
const [scene0, scene1] = source.scenes;
const duration = scene1.releaseSeconds + 1.5;
const wav = (() => {  // 8 kHz 8-bit mono silence
  const n = Math.ceil(duration * 8000), b = Buffer.alloc(44 + n, 128);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22); b.writeUInt32LE(8000, 24); b.writeUInt32LE(8000, 28); b.writeUInt16LE(1, 32); b.writeUInt16LE(8, 34);
  b.write('data', 36); b.writeUInt32LE(n, 40); return b;
})();
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
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
const failures = [];
const check = (ok, label) => { if (!ok) failures.push(label); console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}`); };
try {
  for (const [width, height] of [[390, 844], [390, 664], [1440, 900]]) {
    const tag = `${width}x${height}`;
    const page = await browser.newPage({ viewport: { width, height } });
    const errors = [], reveals = [];
    page.on('pageerror', e => errors.push(String(e)));
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.href === source.mediaUrl) {
        const [, a, b] = /bytes=(\d+)-(\d*)/.exec(route.request().headers().range ?? '') ?? [];
        if (a === undefined) return route.fulfill({ body: wav, contentType: 'audio/wav', headers: { 'Accept-Ranges': 'bytes' } });
        const end = b ? Math.min(Number(b), wav.length - 1) : wav.length - 1;
        return route.fulfill({ status: 206, body: wav.subarray(Number(a), end + 1), contentType: 'audio/wav',
          headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${a}-${end}/${wav.length}` } });
      }
      if (url.hostname !== 'showcase.test') return route.abort();
      if (url.pathname === '/api/watch/849843') return route.fulfill({ json: timeline });
      const r = url.pathname.match(/^\/api\/watch\/849843\/reveal\/(\d+)$/);
      if (r) { reveals.push(Number(r[1])); return route.fulfill({ json: reveal(Number(r[1])) }); }
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'synthetic fixture' } });
      const file = path.join(dist, url.pathname);
      const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
      return route.fulfill({ body: readFileSync(target), contentType: types[path.extname(target)] ?? 'application/octet-stream' });
    });
    const frame = page.frameLocator('.sc-compare iframe');
    const revealButton = frame.getByRole('button', { name: '실제 투구 공개' });
    const down = page.getByRole('button', { name: '추천과 공개 버튼으로 내려가기 ↓' });
    const up = page.getByRole('button', { name: '↑ 영상' });
    const clip = page.getByRole('button', { name: '이 공 다시 보기' });
    // Fully inside the page viewport (frame element boxes are reported in main-page coordinates).
    const inView = async locator => {
      const box = await locator.boundingBox();
      return !!box && box.y >= 0 && box.y + box.height <= height;
    };
    const waitInView = async (locator, ms = 3000) => {
      const until = Date.now() + ms;
      while (Date.now() < until) { if (await inView(locator)) return true; await page.waitForTimeout(100); }
      return false;
    };
    // Earlier pitches of the same plate appearance are fetched as history; only the selected pitch's own reveal counts.
    const own = i => reveals.filter(r => r === i).length;
    const scrollY = () => page.evaluate(() => window.scrollY);
    const settle = async () => { let last = -1, y = await scrollY(); while (y !== last) { await page.waitForTimeout(150); last = y; y = await scrollY(); } return y; };

    await page.goto('http://showcase.test/showcase?scene=0');
    await page.locator('.sc-compare iframe').waitFor();
    await frame.getByText('합성 검사: 추천 없음').waitFor();
    await revealButton.waitFor();
    if (width < 800) {
      check(!(await inView(revealButton)), `${tag}: reveal bar starts below the fold (the problem this fixes)`);
      check(await down.isVisible(), `${tag}: jump-down button visible`);
      if (await down.isVisible()) {
        await down.click();
        check(await waitInView(revealButton), `${tag}: jump-down brings the reveal bar into view`);
        check(await up.isVisible() && await inView(up), `${tag}: ↑ 영상 visible in the comparison header`);
        if (await up.isVisible()) {
          await up.click(); await settle();
          check(await waitInView(page.locator('.sc-video')), `${tag}: ↑ 영상 brings the video back into view`);
        }
      }
      check(own(2) === 0 && await frame.locator('.wa-reveal').count() === 0, `${tag}: jumping reveals nothing`);
    } else {
      check(!(await down.isVisible()) && !(await up.isVisible()), `${tag}: jump buttons hidden on desktop`);
    }

    // Clip end: phone scrolls the reveal bar into view; desktop does not scroll.
    await page.evaluate(() => window.scrollTo(0, 0)); await settle();
    await page.getByRole('button', { name: '공식 경기 영상 불러오기' }).click();
    await page.waitForFunction(() => !document.querySelector('.sc-primary')?.disabled);
    await clip.click();
    const before = await settle();
    await page.locator('video').evaluate((v, t) => { v.currentTime = t; }, scene0.releaseSeconds + 4.05);
    await page.waitForFunction(() => document.querySelector('video').paused);
    if (width < 800) {
      check(await waitInView(revealButton), `${tag}: clip stop brings the reveal bar into view`);
    } else {
      check(await settle() === before, `${tag}: clip stop does not scroll on desktop`);
    }
    check(own(2) === 0 && await frame.locator('.wa-reveal').count() === 0, `${tag}: clip-end scroll reveals nothing`);
    await revealButton.click();
    await frame.locator('.wa-reveal').getByText('합성 실제 투구 2').waitFor();
    check(own(2) === 1, `${tag}: explicit reveal still works after the scroll`);

    // A scene change in the middle of a clip pauses without scrolling.
    await page.evaluate(() => window.scrollTo(0, 0)); await settle();
    await page.getByLabel('비교할 장면').selectOption('1');
    await frame.locator('.wa-pitchno').getByText('5구째').waitFor();
    check(await frame.locator('.wa-reveal').count() === 0, `${tag}: next scene starts unrevealed`);
    await clip.click();
    const mid = await settle();
    await page.getByRole('button', { name: '이전 장면' }).click();
    await page.waitForTimeout(600);
    check(await settle() === mid, `${tag}: scene change mid-clip does not scroll`);

    // Manual playback (native controls, no clip running) reaching the media end does not scroll.
    await page.evaluate(() => window.scrollTo(0, 0)); const manual = await settle();
    await page.locator('video').evaluate(async (v, t) => { v.currentTime = t; await v.play(); }, duration - .2);
    await page.waitForFunction(() => document.querySelector('video').ended);
    await page.waitForTimeout(600);
    check(await settle() === manual, `${tag}: manual playback to the media end (no clip) does not scroll`);

    // Natural media end inside a clip (scene 1 ends before its stop point).
    await page.getByLabel('비교할 장면').selectOption('1');
    await frame.locator('.wa-pitchno').getByText('5구째').waitFor();
    await page.evaluate(() => window.scrollTo(0, 0)); await settle();
    await clip.click();
    const beforeEnd = await settle();
    await page.locator('video').evaluate((v, t) => { v.currentTime = t; }, duration - .2);
    await page.waitForFunction(() => document.querySelector('video').ended);
    if (width < 800) check(await waitInView(revealButton), `${tag}: natural end brings the reveal bar into view`);
    else check(await settle() === beforeEnd, `${tag}: natural end does not scroll on desktop`);
    check(await frame.locator('.wa-reveal').count() === 0 && own(3) === 0, `${tag}: natural-end scroll reveals nothing`);

    check(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), `${tag}: no horizontal overflow`);
    check(errors.length === 0, `${tag}: no page errors ${JSON.stringify(errors)}`);
    if (process.argv[2]) await page.screenshot({ path: `${process.argv[2]}/phone-reveal-${tag}.png` });
    await page.close();
  }
} finally { await browser.close(); }
assert.deepEqual(failures, [], `showcase phone reveal smoke failed: ${failures.join('; ')}`);
console.log('showcase phone reveal smoke: ok (synthetic fixtures; 390x844, 390x664, 1440x900)');
