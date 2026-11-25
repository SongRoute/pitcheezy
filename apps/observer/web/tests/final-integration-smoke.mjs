import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = process.env.OBSERVER_SMOKE_URL || 'http://127.0.0.1:8768/';
const output = process.env.OBSERVER_SMOKE_OUTPUT || '/tmp/observer-final-integration';
fs.mkdirSync(output, { recursive: true });
const report = { passed: false, base, checks: {}, issues: [] };
const browser = await chromium.launch({ headless: true });
try {
  const api = await browser.newContext();
  const response = await api.request.get(new URL('/api/video-lab/annotations', base).href);
  assert.equal(response.status(), 200);
  const records = (await response.json()).annotations;
  assert.ok(records.length >= 4, 'Expected the four existing stored labels');
  const chosen = records.find(item => item.id === '8091cd25ed653ac9d927f7cd850050fdbe2289a413aea2fa3bb687c31b00fdc9');
  assert.ok(chosen, 'Expected original frozen stored label');
  const before = new Set(records.map(item => item.id));
  const track = await api.request.post(new URL(`/api/video-lab/annotations/${chosen.id}/track`, base).href);
  assert.equal(track.status(), 200);
  const tracked = await track.json();
  assert.equal(tracked.id, chosen.id);
  assert.ok(['tracked', 'abstained'].includes(tracked.result.status));
  const afterResponse = await api.request.get(new URL('/api/video-lab/annotations', base).href);
  const after = (await afterResponse.json()).annotations;
  assert.deepEqual(new Set(after.map(item => item.id)), before);
  report.checks.existing_label_tracking = { id: chosen.id, status: tracked.result.status, labels_preserved: true };
  await api.close();

  for (const [name, width, height] of [['desktop', 1440, 900], ['mobile', 390, 844]]) {
    const context = await browser.newContext({ viewport: { width, height } });
    const page = await context.newPage();
    page.on('pageerror', error => report.issues.push(`${name}: ${error}`));
    await page.goto(new URL('/video-lab', base).href, { waitUntil: 'networkidle' });
    const picker = page.getByLabel('저장된 라벨', { exact: true });
    await picker.waitFor();
    assert.equal(await picker.locator('option').count(), records.length + 1);
    await picker.selectOption(chosen.id);
    await page.getByRole('button', { name: '선택한 저장본 불러오기' }).click();
    await page.locator('.saved-provenance').waitFor();
    assert.match(await page.locator('.saved-provenance').innerText(), /검토 전/);
    await page.reload({ waitUntil: 'networkidle' });
    await page.getByLabel('저장된 라벨', { exact: true }).waitFor();
    assert.equal(await page.getByLabel('저장된 라벨', { exact: true }).locator('option').count(), records.length + 1);
    await page.getByLabel('저장된 라벨', { exact: true }).selectOption(chosen.id);
    await page.getByRole('button', { name: '선택한 저장본 불러오기' }).click();
    await page.locator('.tracking-result').waitFor();
    const savedResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/video-lab/annotations' && response.request().method() === 'POST');
    await page.getByRole('button', { name: '라벨을 로컬 서버에 저장' }).click();
    const savedAgain = await savedResponse;
    assert.equal(savedAgain.status(), 200);
    assert.equal((await savedAgain.json()).id, chosen.id);
    const trackedResponse = page.waitForResponse(response => new URL(response.url()).pathname === `/api/video-lab/annotations/${chosen.id}/track` && response.request().method() === 'POST');
    await page.getByRole('button', { name: '저장한 추적 결과 확인' }).click();
    const trackedAgain = await trackedResponse;
    assert.equal(trackedAgain.status(), 200);
    assert.equal((await trackedAgain.json()).id, chosen.id);
    const countResponse = await page.request.get(new URL('/api/video-lab/annotations', base).href);
    assert.equal((await countResponse.json()).annotations.length, records.length);
    assert.match(await page.locator('.tracking-status').innerText(), /프레임 처리/);
    assert.ok(await page.locator('.tracking-trace tbody tr').count() > 0);
    const labSize = await page.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
    assert.ok(labSize.scrollWidth <= labSize.viewport + 1, JSON.stringify(labSize));
    await page.screenshot({ path: path.join(output, `video-lab-${name}.png`), fullPage: true });
    report.checks[`video_lab_${name}`] = { save_button: true, track_button: true, ...labSize };

    await page.goto(base, { waitUntil: 'networkidle' });
    await page.getByRole('combobox').nth(0).selectOption({ index: 1 });
    await page.getByRole('combobox').nth(1).selectOption({ index: 1 });
    await page.getByRole('button', { name: /타석 관전하기/ }).click();
    const details = page.locator('.choice-explanation');
    await details.waitFor();
    await details.locator('summary').first().click();
    const distributions = details.locator('details');
    assert.equal(await distributions.count(), 3);
    for (const item of await distributions.all()) {
      await item.locator('summary').click();
      assert.equal(await item.locator('dl > div').count(), 10);
      const percentages = await item.locator('dd').allInnerTexts();
      assert.ok(percentages.every(value => /^\d+\.\d%$/.test(value)));
    }
    const size = await page.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
    assert.ok(size.scrollWidth <= size.viewport + 1, JSON.stringify(size));
    await page.screenshot({ path: path.join(output, `explanation-${name}.png`), fullPage: true });
    report.checks[`explanation_${name}`] = { candidates: 3, outcomes_each: 10, ...size };
    await context.close();
  }
  assert.deepEqual(report.issues, []);
  report.passed = true;
} catch (error) {
  report.issues.push(String(error));
  throw error;
} finally {
  await browser.close();
  fs.writeFileSync(path.join(output, 'final-integration-smoke.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify(report));
