/** Real registered local-video UI checks. No tracker runs or annotation uploads. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8766/';
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-video-label-v1';
fs.mkdirSync(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
const page = await context.newPage();
const issues = [], mutations = [];
const report = { passed: false, base, checks: {}, issues, mutations };
page.on('pageerror', (error) => issues.push(String(error)));
page.on('console', (message) => { if (message.type() === 'error') issues.push(message.text()); });
page.on('request', (request) => { if (new URL(request.url()).pathname.startsWith('/api/video-lab') && !['GET', 'HEAD'].includes(request.method())) mutations.push(`${request.method()} ${request.url()}`); });

async function settledVideo() {
  await page.waitForFunction(() => { const element = document.querySelector('video'); return element && element.videoWidth > 0 && element.readyState >= 2 && !element.seeking; });
}
async function seek(time) {
  await page.getByLabel('시각 이동 (초)', { exact: true }).fill(String(time));
  await page.getByRole('button', { name: '이 시각으로 이동', exact: true }).click();
  await settledVideo();
  await page.waitForFunction((target) => Math.abs(document.querySelector('video').currentTime - target) < .05, time);
}
async function mark(time, name) {
  await seek(time);
  await page.getByRole('button', { name, exact: true }).click();
}
async function point(rx, ry) {
  const overlay = page.locator('.lab-video-stage canvas');
  const bounds = await overlay.boundingBox();
  assert.ok(bounds && bounds.width > 0 && bounds.height > 0);
  await overlay.click({ position: { x: bounds.width * rx, y: bounds.height * ry } });
}
async function exportJSON(filename) {
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: /라벨 JSON 내려받기/ }).click();
  const download = await pending;
  const file = path.join(output, filename);
  await download.saveAs(file);
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

try {
  const response = await page.request.get(new URL('/api/video-lab/catalog', base).href);
  assert.equal(response.status(), 200);
  const catalog = await response.json();
  const usable = catalog.clips.find((clip) => clip.usable_for_tracking);
  assert.ok(usable, 'A usable allowlisted real clip is required for the video-label UI smoke.');
  assert.ok(usable.fps > 0);
  await page.goto(new URL('/video-lab', base).href, { waitUntil: 'networkidle' });
  await page.getByLabel('검토할 영상', { exact: true }).selectOption(usable.id);
  await settledVideo();
  const dimensions = await page.locator('video').evaluate((element) => ({ width: element.videoWidth, height: element.videoHeight, duration: element.duration }));
  assert.ok(dimensions.duration > 6, 'The real clip should allow independent short-window boundary checks.');
  assert.equal(await page.getByRole('button', { name: /라벨 JSON 내려받기/ }).isDisabled(), true);
  await page.getByLabel('표시한 사람 / 라벨 출처', { exact: true }).selectOption('user_manual');

  await seek(0);
  await page.getByRole('button', { name: '1프레임 →', exact: true }).click();
  await settledVideo();
  const stepped = await page.locator('video').evaluate((element) => element.currentTime);
  assert.ok(Math.abs(stepped - 1 / usable.fps) < .003, `One-frame step ${stepped} differs from FPS ${usable.fps}`);
  report.checks.fps_step = stepped;

  await mark(1, '추적 시작 표시');
  await page.getByRole('button', { name: '추적 영역 표시', exact: true }).click();
  await settledVideo();
  await point(.3, .3);
  await point(.55, .55);
  await page.locator('.roi-readout').waitFor();
  await page.getByRole('button', { name: '네 모서리 표시', exact: true }).click();
  await settledVideo();
  for (const [x, y] of [[.15, .2], [.8, .2], [.8, .8], [.15, .8]]) await point(x, y);
  assert.equal(await page.locator('.corner-readout li').count(), 4);

  await mark(1, '추적 종료 표시');
  await mark(2.5, '릴리스 표시');
  assert.match(await page.locator('.lab-issues').innerText(), /시작 < 종료 < 릴리스/);
  assert.equal(await page.getByRole('button', { name: /라벨 JSON 내려받기/ }).isDisabled(), true);
  await mark(5, '추적 종료 표시');
  await mark(5.5, '릴리스 표시');
  assert.match(await page.locator('.lab-issues').innerText(), /최대 3초/);
  assert.equal(await page.getByRole('button', { name: /라벨 JSON 내려받기/ }).isDisabled(), true);
  await mark(2, '추적 종료 표시');
  await mark(2, '릴리스 표시');
  assert.match(await page.locator('.lab-issues').innerText(), /시작 < 종료 < 릴리스/);
  assert.equal(await page.getByRole('button', { name: /라벨 JSON 내려받기/ }).isDisabled(), true);
  await mark(2.5, '릴리스 표시');
  const payload = await exportJSON('manual-label-desktop.json');
  assert.equal(payload.schema_version, 1);
  assert.equal(payload.annotation_version, 1);
  assert.equal(payload.clip_id, usable.id);
  assert.equal(payload.clip_sha256, usable.clip_sha256);
  assert.equal(payload.source_url, usable.source_url);
  assert.equal(payload.pitch_id, null);
  assert.equal(payload.label_source, 'user_manual');
  assert.equal(payload.review_status, 'unreviewed');
  assert.equal('clip_path' in payload, false);
  assert.deepEqual(payload.image_dimensions, { width: dimensions.width, height: dimensions.height });
  assert.ok(Math.abs(payload.roi.x - dimensions.width * .3) < 2);
  assert.ok(Math.abs(payload.roi.y - dimensions.height * .3) < 2);
  assert.ok(Math.abs(payload.roi.w - dimensions.width * .25) < 2);
  assert.ok(Math.abs(payload.roi.h - dimensions.height * .25) < 2);
  assert.equal(payload.calibration_corners.length, 4);
  assert.ok(payload.seed_time < payload.end_time && payload.end_time < payload.release_time);
  assert.ok(payload.end_time - payload.seed_time <= 3);
  report.checks.flat_schema_and_natural_geometry = true;
  report.checks.interval_and_release_guards = true;
  await seek(payload.seed_time);
  await page.screenshot({ path: path.join(output, 'video-lab-desktop.png'), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  const screen = await page.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
  assert.ok(screen.scrollWidth <= screen.width + 1, JSON.stringify(screen));
  await page.getByLabel('표시한 사람 / 라벨 출처', { exact: true }).selectOption('assistant_visual_estimate');
  const mobile = await exportJSON('manual-label-mobile.json');
  assert.deepEqual(mobile.roi, payload.roi);
  assert.deepEqual(mobile.calibration_corners, payload.calibration_corners);
  assert.deepEqual(mobile.image_dimensions, payload.image_dimensions);
  assert.equal(mobile.label_source, 'assistant_visual_estimate');
  await page.screenshot({ path: path.join(output, 'video-lab-mobile.png'), fullPage: true });
  report.checks.mobile_geometry_unchanged = screen;

  await page.getByRole('button', { name: '보정점 지우기', exact: true }).click();
  const uncalibrated = await exportJSON('manual-label-no-calibration.json');
  assert.equal(uncalibrated.calibration_corners, null);
  await page.getByRole('button', { name: '이 클립 입력 초기화', exact: true }).click();
  assert.equal(await page.getByRole('button', { name: /라벨 JSON 내려받기/ }).isDisabled(), true);
  report.checks.optional_calibration_and_clear = true;

  const unavailable = catalog.clips.find((clip) => !clip.usable_for_tracking);
  if (unavailable) {
    await page.getByLabel('검토할 영상', { exact: true }).selectOption(unavailable.id);
    await page.locator('.lab-unavailable').waitFor();
    assert.ok((await page.locator('.lab-unavailable').innerText()).includes(unavailable.reason));
    assert.equal(await page.getByRole('button', { name: /라벨 JSON 내려받기/ }).isDisabled(), true);
    report.checks.unusable_video_honest_state = true;
  }
  assert.deepEqual(mutations, []);
  assert.deepEqual(issues, []);
  report.passed = true;
} finally {
  await browser.close();
  fs.writeFileSync(path.join(output, 'video-lab-smoke.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify(report));
