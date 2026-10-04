// Synthetic local regression: video/scene identity, embedded reveal and scene lock, copy-link identity,
// media loaded/play/clip-stop/ended/abort/error transitions, missing mapping, network failure,
// and in-place recovery from media, reveal and timeline failures without losing the scene or pre-reveal state.
// The real MLB video is never downloaded: the media URL is answered with a locally generated silent WAV.
// That exercises <video> element state handling only, not real H.264/HLS decoding, Safari or iPhone playback.
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
const [scene0, scene1] = source.scenes;
// Media ends before scene 1's clip stop (release + 4): scene 0 tests the clip stop, scene 1 the natural end.
const duration = scene1.releaseSeconds + 1.5;
assert(duration > scene0.releaseSeconds + 4);
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
const near = (actual, expected, label) => assert(Math.abs(actual - expected) < .3, `${label}: ${actual} vs ${expected}`);

// Synthetic play() needs no user gesture for the same-task clip+scene-switch check below.
const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], reveals = [];
    page.on('pageerror', e => errors.push(String(e)));
    let failApi = false, failMedia = false, failReveal = false, mediaRequests = 0, otherRequests = 0;
    await page.addInitScript(() => {  // http://showcase.test is not a secure context; record what would be copied
      Object.defineProperty(navigator, 'clipboard', { value: { writeText: async text => { window.__copied = text; } } });
    });
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.href === source.mediaUrl) {
        mediaRequests++;
        if (failMedia) return route.abort();
        // Seeking needs a range-capable host (as a CDN is assumed to be); without 206 Chromium cannot seek.
        const [, a, b] = /bytes=(\d+)-(\d*)/.exec(route.request().headers().range ?? '') ?? [];
        if (a === undefined) return route.fulfill({ body: wav, contentType: 'audio/wav', headers: { 'Accept-Ranges': 'bytes' } });
        const end = b ? Math.min(Number(b), wav.length - 1) : wav.length - 1;
        return route.fulfill({ status: 206, body: wav.subarray(Number(a), end + 1), contentType: 'audio/wav',
          headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${a}-${end}/${wav.length}` } });
      }
      if (url.hostname !== 'showcase.test') { otherRequests++; return route.abort(); }
      if (url.pathname === '/api/watch/849843') return route.fulfill({ status: failApi ? 503 : 200, json: failApi ? { detail: '자료 없음' } : timeline });
      const r = url.pathname.match(/^\/api\/watch\/849843\/reveal\/(\d+)$/);
      if (r && failReveal) return route.fulfill({ status: 503, json: { detail: '합성 공개 실패' } });
      if (r) { reveals.push(Number(r[1])); return route.fulfill({ json: reveal(Number(r[1])) }); }
      if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'synthetic fixture' } });
      const file = path.join(dist, url.pathname);
      const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
      return route.fulfill({ body: readFileSync(target), contentType: types[path.extname(target)] ?? 'application/octet-stream' });
    });
    const select = page.getByLabel('비교할 장면');
    const clip = page.getByRole('button', { name: '이 공 다시 보기' });
    const frame = page.frameLocator('.sc-compare iframe');
    const media = () => page.locator('video').evaluate(v => ({ t: v.currentTime, paused: v.paused, ended: v.ended, ready: v.readyState }));
    const alert = page.getByRole('alert').filter({ hasText: '영상 재생을 확인하지 못했어요' });

    await page.goto('http://showcase.test/showcase?scene=0');
    await page.locator('.sc-compare iframe').waitFor();
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=2', 'use pitch identity, not scene ordinal');
    await frame.getByText('합성 검사: 추천 없음').waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'no actual pitch before explicit reveal');
    assert.equal(await frame.locator('.wa-guide').count(), 0, 'outer instructions replace embedded onboarding');
    assert.equal(await frame.locator('.wa-innings, .wa-nav, .wa-step').count(), 0, 'embedded scene cannot navigate independently');
    await frame.locator('.wa-page').press('ArrowLeft');
    const inner = () => page.frames().find(f => f.url().includes('/watch?'));
    assert(inner().url().endsWith('&embed=1#i=2'), 'keyboard cannot change embedded pitch');

    // Copy link: the copied URL and the open link follow the selected pitch; the note resets on scene change.
    await page.getByRole('button', { name: '아이폰용 관전 링크 복사' }).click();
    await page.getByText('링크를 복사했어요').waitFor();
    assert.equal(await page.evaluate(() => window.__copied), 'http://showcase.test/watch?game=849843#i=2');
    await select.selectOption('1');
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=3');
    await page.getByText('링크를 복사했어요').waitFor({ state: 'detached', timeout: 2000 });  // note does not carry over
    assert.equal(await page.getByRole('link', { name: '이 공 관전 화면 열기' }).getAttribute('href'), '/watch?game=849843#i=3');
    await page.getByRole('button', { name: '아이폰용 관전 링크 복사' }).click();
    assert.equal(await page.evaluate(() => window.__copied), 'http://showcase.test/watch?game=849843#i=3');
    assert.equal(new URL(page.url()).searchParams.get('scene'), '1', 'page URL keeps the selected scene');
    await select.selectOption('0');

    // Video loads only on request, then seeks to the selected scene and waits paused.
    assert.equal(mediaRequests + otherRequests, 0, 'video loads only on viewer request');
    assert.equal(await clip.isDisabled(), true, 'clip disabled before video metadata');
    await page.getByRole('button', { name: '공식 경기 영상 불러오기' }).click();
    await page.waitForFunction(() => !document.querySelector('.sc-primary')?.disabled);
    assert.equal(await page.getByText('영상 정보를 불러오는 중').count(), 0);
    let m = await media();
    near(m.t, scene0.releaseSeconds - 3, 'metadata seeks to scene start'); assert(m.paused);

    // Real embedded click after the video controls: the frame fits the viewport so its fixed reveal bar is reachable
    // (desktop: from the page top; phone: after scrolling to the comparison section).
    // The CV evidence card below makes the desktop page scrollable; earlier clicks may have scrolled it, so return to the top.
    if (width < 800) await page.locator('.sc-compare').evaluate(e => e.scrollIntoView({ block: 'end' }));
    else await page.evaluate(() => window.scrollTo(0, 0));
    const box = await page.locator('.sc-compare').boundingBox();
    assert(box.y >= 0 && box.y + box.height <= 900, `comparison fits the viewport: ${JSON.stringify(box)}`);
    await frame.getByRole('button', { name: '실제 투구 공개' }).click();
    await frame.locator('.wa-reveal').getByText('합성 실제 투구 2').waitFor();
    assert(reveals.includes(2));
    // Scene lock: keys and a hash edit inside the frame cannot move the embedded pitch.
    for (const key of ['ArrowRight', 'ArrowLeft', 'ArrowRight']) await frame.locator('.wa-page').press(key);
    await inner().evaluate(() => { location.hash = '#i=0'; });
    await page.waitForTimeout(100);
    assert.equal(await frame.locator('.wa-pitchno').innerText(), '3구째', 'embedded pitch stays locked');
    assert.equal(await frame.locator('.wa-reveal h3').innerText(), '합성 실제 투구 2 · 95.0 mph');
    assert.equal(await frame.getByRole('button', { name: '공개 완료 · 다음 장면은 영상 옆에서 선택' }).isDisabled(), true);

    // Replaying the clip remounts the comparison (reveal hidden again) and stops ~4 s after release.
    await clip.click();
    await frame.getByRole('button', { name: '실제 투구 공개' }).waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'replay hides the previous reveal');
    m = await media(); near(m.t, scene0.releaseSeconds - 3, 'clip start'); assert(!m.paused, 'clip plays');
    await page.locator('video').evaluate((v, t) => { v.currentTime = t; }, scene0.releaseSeconds + 4.05);
    await page.waitForFunction(() => document.querySelector('video').paused);
    // Keyboard reveal needs focus inside the frame; a focused outer button would take Space instead.
    await frame.locator('.wa-board').click();
    await page.keyboard.press(' ');
    await frame.locator('.wa-reveal').waitFor();

    // Reselect: another scene and back remounts a fresh, unrevealed frame and re-seeks the paused video.
    await select.selectOption('1');
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=3');
    await frame.locator('.wa-pitchno').getByText('5구째').waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0);
    m = await media(); near(m.t, scene1.releaseSeconds - 3, 'scene 1 seek'); assert(m.paused);
    await select.selectOption('0');
    await frame.locator('.wa-pitchno').getByText('3구째').waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'reselected scene does not carry its earlier reveal');

    // Switching scene in the same task as starting the clip aborts play(); that must not read as a media failure.
    await page.evaluate(() => {
      document.querySelector('.sc-primary').click();
      [...document.querySelectorAll('.sc-controls button')].find(b => b.textContent === '다음 장면').click();
    });
    await page.waitForTimeout(300);
    assert.equal(await alert.count(), 0, 'aborted play() is not a video error');
    assert.equal(await select.inputValue(), '1');
    m = await media(); assert(m.paused, 'scene switch pauses the clip');
    assert.equal(await clip.isDisabled(), false);

    // Natural end of media inside a clip, then replay.
    await clip.click();
    await page.locator('video').evaluate((v, t) => { v.currentTime = t; }, duration - .2);
    await page.waitForFunction(() => document.querySelector('video').ended);
    assert.equal(await alert.count(), 0, 'ended is not an error');
    await clip.click();
    m = await media(); near(m.t, scene1.releaseSeconds - 3, 'replay after end'); assert(!m.paused && !m.ended);
    await page.locator('video').evaluate(v => v.pause());

    await select.selectOption('2');
    await page.getByText('이 장면에 대응하는 투구 자료가 없습니다. 다른 장면을 선택해 주세요.').waitFor();
    assert.equal(await page.locator('iframe').count(), 0, 'missing mapping cannot retain prior pitch');
    assert.equal(await clip.isDisabled(), true, 'no clip without a matching pitch');
    assert.equal(otherRequests, 0, 'no request leaves the synthetic host');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal overflow');
    if (process.argv[2]) await page.screenshot({ path: `${process.argv[2]}/showcase-${width}.png`, fullPage: true });

    failMedia = true;
    await page.goto('http://showcase.test/showcase?scene=0');
    await page.getByRole('button', { name: '공식 경기 영상 불러오기' }).click();
    await alert.waitFor();
    assert.equal(await clip.isDisabled(), true);
    // A transient media failure is retried in place: same scene, re-seeked, comparison still unrevealed.
    failMedia = false;
    await alert.getByRole('button', { name: '영상 다시 불러오기' }).click();
    await page.waitForFunction(() => !document.querySelector('.sc-primary')?.disabled);
    assert.equal(await alert.count(), 0, 'retry clears the media error');
    m = await media(); near(m.t, scene0.releaseSeconds - 3, 'retry seeks to scene start'); assert(m.paused);
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=2');
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'media retry does not reveal');

    // A failed reveal is announced inside the frame, shows no actual pitch, and a retry clears the stale error.
    failReveal = true;
    if (width < 800) await page.locator('.sc-compare').evaluate(e => e.scrollIntoView({ block: 'end' }));
    await frame.getByRole('button', { name: '실제 투구 공개' }).click();
    await frame.getByRole('alert').getByText('합성 공개 실패').waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'failed reveal shows no actual pitch');
    failReveal = false;
    await frame.getByRole('button', { name: '실제 투구 공개' }).click();
    await frame.locator('.wa-reveal').getByText('합성 실제 투구 2').waitFor();
    assert.equal(await frame.getByRole('alert').filter({ hasText: '합성 공개 실패' }).count(), 0, 'successful retry clears the reveal error');

    failApi = true;
    await page.goto('http://showcase.test/showcase?scene=invalid');
    await page.locator('.sc-compare [role=alert]').waitFor();
    assert.equal(await page.locator('iframe').count(), 0);
    // Timeline failure is retried in place; the comparison then mounts for the selected pitch, unrevealed.
    failApi = false;
    await page.locator('.sc-compare').getByRole('button', { name: '다시 시도' }).click();
    await page.locator('.sc-compare iframe').waitFor();
    assert.equal(await page.locator('iframe').getAttribute('src'), '/watch?game=849843&embed=1#i=2');
    await frame.getByText('합성 검사: 추천 없음').waitFor();
    assert.equal(await frame.locator('.wa-reveal').count(), 0, 'timeline retry does not reveal');
    assert.equal(await page.locator('.sc-compare [role=alert]').count(), 0);
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log('showcase smoke: ok (synthetic WAV media + timeline/reveal fixtures, desktop + phone)');
} finally { await browser.close(); }
