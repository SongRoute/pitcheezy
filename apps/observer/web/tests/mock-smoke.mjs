// mock API 모드(VITE_API_MOCK=1) 스모크. 백엔드 없이 `npm run build:mock && npm run preview` 또는 Vercel 프리뷰 URL 에 대해 돈다.
// 사용: MOCK_SMOKE_URL=http://127.0.0.1:4173/ node tests/mock-smoke.mjs   (URL 생략 시 preview 를 직접 띄움)
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const output = process.env.MOCK_SMOKE_OUTPUT || path.join(process.cwd(), 'runs', 'mock-smoke');
fs.mkdirSync(output, { recursive: true });
let base = process.env.MOCK_SMOKE_URL, preview = null;
if (!base) {
  preview = spawn('npx', ['vite', 'preview', '--host', '127.0.0.1', '--port', '4173', '--strictPort'], { stdio: 'ignore' });
  base = 'http://127.0.0.1:4173/';
  await new Promise((resolve) => setTimeout(resolve, 1500));
}
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH || undefined });
const issues = [];

async function openAt(width, height) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', (error) => issues.push(`pageerror: ${error}`));
  page.on('console', (message) => { if (message.type() === 'error') issues.push(`console: ${message.text()}`); });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.documentElement.dataset.apiMock === '1', null, { timeout: 10000 });
  await page.getByRole('combobox').nth(0).selectOption({ index: 1 });
  await page.getByRole('combobox').nth(1).selectOption({ index: 1 });
  await page.getByRole('button', { name: /타석 관전하기/ }).click();
  await page.locator('.candidate').first().waitFor();
  return page;
}

try {
  const desktop = await openAt(1280, 900);
  assert.equal(await desktop.locator('.candidate').count(), 3, '추천 3개');
  assert.match(await desktop.locator('body').innerText(), /mock/i, 'mock 안내 표시');
  await desktop.screenshot({ path: path.join(output, 'desktop-pitch1.png'), fullPage: true });
  for (let i = 0; i < 4; i++) { await desktop.getByRole('button', { name: /다음 실제 공 확인/ }).click(); await desktop.locator('.timeline-pitch').nth(i).waitFor(); }
  await desktop.locator('.terminal-panel').waitFor();
  assert.equal(await desktop.locator('.timeline-pitch').count(), 4, '4구 재생 완료');
  await desktop.locator('.zone-buttons button').first().click();
  await desktop.getByRole('button', { name: /목표 메모 저장/ }).click();
  await desktop.locator('.comparison').waitFor();
  assert.match(await desktop.locator('.comparison').innerText(), /직접 표시한 목표/);
  await desktop.screenshot({ path: path.join(output, 'desktop-complete.png'), fullPage: true });

  const mobile = await openAt(390, 844);
  assert.equal(await mobile.locator('.candidate').count(), 3);
  const overflow = await mobile.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.ok(overflow <= 0, `390px 가로 스크롤 ${overflow}px`);
  await mobile.screenshot({ path: path.join(output, 'mobile-pitch1.png'), fullPage: true });

  assert.deepEqual(issues, [], issues.join('\n'));
  console.log(`mock smoke OK → ${output}`);
} finally {
  await browser.close();
  preview?.kill();
}
