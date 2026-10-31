// Current service UI: Korean word wrapping, long-token fallback and keyboard focus above the fixed navigation.
// Chromium with route-mocked existing fixture. 200% is explicit DOM text enlargement, not Safari or browser zoom.
// npm run build && node tests/ui-accessibility-smoke.mjs [screenshotDir]
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url)), dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const body = JSON.parse(readFileSync(path.join(here, 'fixtures/service-849843.json'), 'utf8')).live;
const longToken = 'LongUnbrokenExternalRecordIdentifier'.repeat(8);
body.notes = [...(body.notes ?? []), longToken]; // synthetic text; no fixture file is changed
const out = process.argv[2]; if (out) mkdirSync(out, { recursive: true });
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
let passed = 0, failed = 0;
const ok = (condition, text, detail) => { condition ? passed++ : failed++; console.log(condition ? 'ok  ' : 'FAIL', text, condition ? '' : JSON.stringify(detail)); };
const browser = await chromium.launch();
try {
  for (const [width, height] of [[320, 568], [390, 844]]) {
    const context = await browser.newContext({ viewport: { width, height }, locale: 'ko-KR' }), page = await context.newPage(), errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    await page.route('**/*', route => {
      const u = new URL(route.request().url()); if (u.origin !== 'http://pz.test') return route.abort();
      if (u.pathname === '/api/service/games/849843') {
        const answer = structuredClone(body), after = u.searchParams.get('after');
        if (after !== null) answer.pitches = answer.pitches.filter(p => p.index > Number(after));
        return route.fulfill({ json: answer });
      }
      if (u.pathname.startsWith('/api/')) return route.fulfill({ status: 404, json: {} });
      const file = path.join(dist, u.pathname), asset = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
      return route.fulfill({ body: readFileSync(asset), contentType: types[path.extname(asset)] ?? 'application/octet-stream' });
    });
    await page.goto('http://pz.test/watch/849843?demo=fixture');
    await page.locator('.sv-bottom-nav').waitFor(); await page.evaluate(() => document.fonts.ready);
    const focused = [];
    for (let i = 0; i < 40; i++) {
      await page.keyboard.press('Tab');
      const f = await page.evaluate(() => {
        const e = document.activeElement, b = e.getBoundingClientRect(), nav = document.querySelector('.sv-bottom-nav').getBoundingClientRect();
        return { cls: e.className, tag: e.tagName, nav: !!e.closest('.sv-bottom-nav'), top: b.top, bottom: b.bottom, fold: nav.top,
          hit: e.contains(document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2)) };
      });
      if (f.nav) break; focused.push(f);
    }
    ok(focused.some(f => f.tag === 'SUMMARY') && focused.some(f => f.cls.includes('sv-mobile-we')), `${width}: keyboard reaches the formerly obscured controls`, focused);
    ok(focused.length > 0 && focused.every(f => f.top >= -1 && f.bottom <= f.fold + 1 && f.hit), `${width}: every focused page control is visible above the tab bar`, focused);
    for (const scale of [1, 2]) for (const name of ['추천', '선수', '구종', '기록']) {
      const opener = page.locator('.sv-bottom-nav').getByRole('button', { name, exact: true });
      await opener.focus(); await page.keyboard.press('Enter'); await page.getByRole('dialog').waitFor();
      const enlarged = await page.evaluate(scale => {
        const sheet = document.querySelector('.sv-sheet');
        // Snapshot before applying styles so inherited sizes are not doubled twice. Include fixed-pixel fonts and close text.
        const text = [...sheet.querySelectorAll('*')].filter(e => [...e.childNodes].some(n => n.nodeType === Node.TEXT_NODE && n.textContent.trim()));
        const sizes = text.map(e => ({ e, font: parseFloat(getComputedStyle(e).fontSize), line: parseFloat(getComputedStyle(e).lineHeight) }));
        if (scale === 2) sizes.forEach(({ e, font, line }) => { e.style.fontSize = `${font * 2}px`; if (Number.isFinite(line)) e.style.lineHeight = `${line * 2}px`; });
        return sizes.map(({ e, font }) => ({ before: font, after: parseFloat(getComputedStyle(e).fontSize) }));
      }, scale);
      const m = await page.evaluate(longToken => {
        const sheet = document.querySelector('.sv-sheet'), b = sheet.querySelector('.sv-sheet-body'), br = b.getBoundingClientRect();
        const breaks = [], outside = [], walker = document.createTreeWalker(b, NodeFilter.SHOW_TEXT);
        let n;
        while ((n = walker.nextNode())) {
          const e = n.parentElement; if (!n.textContent.trim() || !e.getBoundingClientRect().height) continue;
          const range = document.createRange(); range.selectNodeContents(n);
          for (const r of range.getClientRects()) if (r.left < br.left - 1 || r.right > br.right + 1) outside.push(n.textContent.slice(0, 40));
          const css = getComputedStyle(e);
          const room = e.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight);
          for (const word of n.textContent.matchAll(/[가-힣]{2,}/g)) {
            let wordWidth = 0;
            for (let i = 0; i < word[0].length; i++) { range.setStart(n, word.index + i); range.setEnd(n, word.index + i + 1); wordWidth += range.getBoundingClientRect().width; }
            range.setStart(n, word.index); range.setEnd(n, word.index + word[0].length);
            // A token wider than its own line is allowed to use the anywhere fallback.
            if (wordWidth <= room && new Set([...range.getClientRects()].map(r => Math.round(r.top))).size > 1) breaks.push(word[0]);
          }
        }
        return { keepAll: getComputedStyle(b).wordBreak, overflowWrap: getComputedStyle(b).overflowWrap, breaks, outside,
          bodyWidth: b.clientWidth, scrollWidth: b.scrollWidth, longToken: b.textContent.includes(longToken), focusedDialog: document.activeElement === sheet };
      }, longToken);
      const tag = `${width} ${scale * 100}% ${name}`;
      ok(enlarged.length > 0 && enlarged.every(s => Math.abs(s.after - scale * s.before) < .01), `${tag}: actual computed text sizes match enlargement`, enlarged);
      ok(m.keepAll === 'keep-all' && m.breaks.length === 0, `${tag}: fitting Korean words stay whole`, m);
      ok(m.overflowWrap === 'anywhere' && m.outside.length === 0 && m.scrollWidth <= m.bodyWidth + 1, `${tag}: text remains within the sheet, with long-token fallback`, m);
      if (name === '추천') ok(m.longToken, `${tag}: synthetic unbroken token is actually rendered`, m);
      ok(m.focusedDialog, `${tag}: opening focuses the dialog`, m);
      await page.keyboard.press('Shift+Tab'); await page.keyboard.press('Tab');
      const close = page.getByRole('dialog').getByRole('button', { name: '닫기', exact: true });
      ok(await close.evaluate(e => e === document.activeElement && e.getBoundingClientRect().top >= 0 && e.getBoundingClientRect().bottom <= innerHeight), `${tag}: focus cycles to a visible close button`);
      if (out) await page.screenshot({ path: path.join(out, `${width}-${scale * 100}-${name}.png`) });
      await page.keyboard.press('Escape'); await page.getByRole('dialog').waitFor({ state: 'detached' });
      await page.evaluate(() => new Promise(requestAnimationFrame));
      ok(await opener.evaluate(e => e === document.activeElement), `${tag}: Escape restores the opener`);
    }
    ok(errors.length === 0, `${width}: no page errors`, errors);
    await context.close();
  }
} finally { await browser.close(); }
console.log(`${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
