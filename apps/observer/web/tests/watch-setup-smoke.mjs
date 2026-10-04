// Catcher setup (horizontal mitt position from the video, GET /api/watch/{game}/setup) on the watch-along page and inside
// the Showcase iframe: drawn only after this very pitch's reveal succeeds, matched on exact game:at_bat:pitch, 0 is a
// position, off-plate x is drawn off the plate (beyond the picture it is described, not pinned to the edge), abstained
// pitches say so, and a missing/404/503/aborted/other-game answer draws nothing. Synthetic data on the built dist.
//   PZ_WEB_DIST=/tmp/dist PLAYWRIGHT_BROWSERS_PATH=... node tests/watch-setup-smoke.mjs [screenshot-dir]
import { chromium } from 'playwright';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.PZ_WEB_DIST ?? path.join(here, '..', 'dist');
const source = JSON.parse(readFileSync(path.join(here, '..', 'src', 'showcase-scenes.json'), 'utf8'));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const shots = process.argv[2];
const fail = (message) => { console.error('FAIL', message); process.exitCode = 1; };
const px = (feet) => 90 + feet / 1.6 * 80;  // WatchAlong Zone x scale
const PLATE_LEFT = px(-0.83);

const candidate = { rank: 1, pitch_type: 'FF', pitch_label: '포심', zone_id: 'middle_middle', zone_label: '가운데 높이 중앙', target: { x: 0, z: 2.5 },
  detail: { probability: 0.5, reference_probability: 0.4, kernel_mass: null, kernel_ess: null } };
const decision = (game, index, ab, n, ready) => ({ index, pa_id: `${game}:${ab}`, at_bat_number: ab, pitch_number: n,
  situation: { inning: 1, half: 'Top', outs: 0, balls: 0, strikes: 0, bases: 0, home_score: 0, away_score: 0 },
  pitcher: { id: 1, name: '투수', hand: 'R' }, batter: { id: 100 + ab, name: `타자${ab}`, side: 'R' }, status: ready ? 'READY' : 'UNSUPPORTED',
  pre: ready ? { status: 'ready', reason: null, recommendation: { candidates: [candidate] } }
    : { status: 'unsupported', reason: '2025년 4월 이후 데뷔한 투수라 추천 대상이 아닙니다.', recommendation: null } });
const timeline = (game, rows) => ({ schema: 'pitcheezy-watch-along-v1', badge: '검증 전 실험 버전',
  game: { game_pk: game, date: '2025-09-01', game_type: 'R', away_team: 'Chicago Cubs', home_team: 'San Diego Padres' },
  policy: { name: 'synthetic', identity_sha256: '0'.repeat(64), tau: 0.5, note: '' }, location: { note: '근사', zone_bounds: { bottom: 1.6, top: 3.3899 } },
  we_note: '', coverage: { pitch_decisions: rows.length, ready: 0, ready_share_of_pitches: 0 }, decisions: rows.map(([ab, n, ready], i) => decision(game, i, ab, n, ready)) });
const reveal = (i) => ({ index: i, we: { home_before: 0.5, home_after: 0.5, home_delta: 0 },
  actual: { pitch_type: 'FF', pitch_label: `실제 ${i}`, result_label: '볼', event_label: null, play_text: null, speed_mph: 95, x: 0.2, z: 2.4, zone_label: '가운데' } });
const row = (ab, n, x, status = x === null ? 'unavailable' : 'estimated') => ({ pitch_id: `?:${ab}:${n}`, at_bat_number: ab, pitch_number: n, status, plate_x_feet: x });
const setup = (game, rows) => ({ schema: 'pitcheezy-setup-estimates-v1', badge: '포수 셋업 위치 추정(검토 전)', game_pk: game, pitches: rows });

// Game A: index → at_bat:pitch, ready?, what the setup file says about it
const GAME = 900300, OTHER = 900301;
const A = [[1, 1, true], [1, 2, true], [2, 1, false], [3, 1, true], [11, 1, true], [5, 1, true], [6, 1, false]];
const SETUP_A = [row(1, 1, 0), row(1, 2, -1.0), row(2, 1, 0.4), row(3, 1, null), row(5, 1, 1.3), row(6, 1, null),
  // decoys near index 4 (11:1, no line): a loose, swapped or separator-less key ("111" = 1:11 = 11:1) would pick one up
  row(11, 2, 0.6), row(1, 11, 0.6), row(111, 1, 0.6), row(11, 11, 0.6)];
const [scene0, scene1] = source.scenes;
const SHOW = timeline(source.gamePk, [[scene0.atBat, scene0.pitchNumber, true], [scene1.atBat, scene1.pitchNumber, false]]);
const SETUP_SHOW = setup(source.gamePk, [row(scene0.atBat, scene0.pitchNumber, 0.3), row(scene1.atBat, scene1.pitchNumber, null)]);

const browser = await chromium.launch();
const results = [];
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } });
    await context.addInitScript(() => { try { localStorage.setItem('pitcheezy.guide.v1', '1'); } catch { /* ignore */ } });
    let mode = 'ok', delayMs = 0, failNextReveal = false, setupCalls = 0;
    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      if (url.host !== 'pz.test') return route.abort();  // never the real media or anything else
      const p = url.pathname;
      if (p === `/api/watch/${GAME}`) return route.fulfill({ json: timeline(GAME, A) });
      if (p === `/api/watch/${OTHER}`) return route.fulfill({ json: timeline(OTHER, A) });
      if (p === `/api/watch/${source.gamePk}`) return route.fulfill({ json: SHOW });
      const s = p.match(/^\/api\/watch\/(\d+)\/setup$/);
      if (s) {
        setupCalls++;
        if (delayMs) await new Promise(r => setTimeout(r, delayMs));
        const game = Number(s[1]);
        if (game === source.gamePk) return route.fulfill({ json: SETUP_SHOW });
        if (game === OTHER || mode === '404') return route.fulfill({ status: 404, json: { detail: '이 경기의 포수 셋업 자료가 없습니다.' } });
        if (mode === '503') return route.fulfill({ status: 503, json: { detail: '포수 셋업 자료를 확인할 수 없습니다.' } });
        if (mode === 'abort') return route.abort();
        if (mode === 'wrong') return route.fulfill({ json: setup(OTHER, SETUP_A) });  // an answer for another game
        return route.fulfill({ json: setup(game, SETUP_A) });
      }
      const r = p.match(/^\/api\/watch\/\d+\/reveal\/(\d+)$/);
      if (r && failNextReveal) { failNextReveal = false; return route.fulfill({ status: 503, json: { detail: '잠시 후 다시 시도해 주세요.' } }); }
      if (r) return route.fulfill({ json: reveal(Number(r[1])) });
      if (p.startsWith('/api/')) return route.fulfill({ status: 404, json: { detail: 'not in this smoke' } });
      const file = path.join(dist, p);
      const target = existsSync(file) && statSync(file).isFile() ? file : path.join(dist, 'index.html');
      return route.fulfill({ body: readFileSync(target), contentType: TYPES[path.extname(target)] ?? 'application/octet-stream' });
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    const tag = (s) => `${width} ${s}`;
    const marks = (root = page) => root.locator('.wa-setup-mark line');
    const markX = async (root = page) => {
      const all = await marks(root).all();
      if (all.length !== 1) return { count: all.length };
      return { count: 1, x1: Number(await all[0].getAttribute('x1')), data: await root.locator('.wa-setup-mark').getAttribute('data-x') };
    };
    const noSetup = async (label, root = page) => {
      if (await root.locator('.wa-setup-mark, .wa-setup').count()) fail(tag(`${label}: setup shown`));
      if ((await root.locator('body').innerText()).includes('미트')) fail(tag(`${label}: mitt words on screen`));
    };
    const open = async (game, i, extra = '') => {
      await page.goto('about:blank'); setupCalls = 0;
      await page.goto(`http://pz.test/watch?game=${game}${extra}#i=${i}`);
      await page.locator('.wa-card').waitFor({ timeout: 8000 });
    };
    const settle = () => page.waitForTimeout(delayMs + 150);  // let the setup answer land
    const doReveal = async (root = page) => { await root.getByRole('button', { name: '실제 투구 공개' }).click(); await root.locator('.wa-reveal').waitFor({ timeout: 5000 }); };
    const expectMark = async (label, feet, root = page) => {
      const m = await markX(root);
      if (m.count !== 1 || Math.abs(m.x1 - px(feet)) > 0.01 || Number(m.data) !== feet) fail(tag(`${label}: mark ${JSON.stringify(m)} != ${feet}ft (${px(feet)})`));
      return m;
    };

    // A. setup slower than the page: nothing before the reveal, even after it arrives; 0 ft is drawn at the centre
    mode = 'ok'; delayMs = 800;
    await open(GAME, 0);
    await noSetup('A before setup arrives');
    await settle();
    await noSetup('A setup arrived, not revealed');
    await doReveal();
    const a = await expectMark('A x=0', 0);
    if (!(await page.getByText('점선 = 포수 미트 위치 추정').isVisible())) fail(tag('A legend missing'));
    const more = page.locator('.wa-setup-more');
    if (await more.evaluate(d => d.open)) fail(tag('A caveat open by default'));
    await more.locator('summary').click();
    const caveat = await more.innerText();
    if (!/정확도는 미측정/.test(caveat) || !/다른 두 경기/.test(caveat) || !/의도나 목표가 아니며/.test(caveat)) fail(tag(`A caveat text: ${caveat}`));
    if (setupCalls !== 1) fail(tag(`A setup requested ${setupCalls} times`));
    if (shots) await page.locator('.wa-card').screenshot({ path: path.join(shots, `setup-ready-${width}.png`) });
    results.push({ width, case: 'A zero', x1: a.x1 });

    // B. next pitch (off-plate x): nothing until its own reveal, then drawn left of the plate edge, not clamped onto it
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => window.location.hash === '#i=1');
    await noSetup('B before reveal');
    await doReveal();
    const b = await expectMark('B x=-1.0', -1.0);
    if (!(b.x1 < PLATE_LEFT)) fail(tag(`B off-plate mark drawn inside the plate: ${b.x1} >= ${PLATE_LEFT}`));
    // C. back to the revealed pitch 0: its own value, not pitch 1's
    await page.keyboard.press('ArrowLeft');
    await page.waitForFunction(() => window.location.hash === '#i=0');
    await expectMark('C back to 0', 0);
    results.push({ width, case: 'B off-plate', x1: b.x1, plateLeft: PLATE_LEFT });

    // D. no recommendation: after the reveal a zone appears with the mitt line; caption does not mention 추천
    delayMs = 0;
    await open(GAME, 2);
    await page.getByText('추천 없음').waitFor();
    await settle();
    await noSetup('D before reveal');
    await doReveal();
    await expectMark('D no-rec x=0.4', 0.4, page.locator('.wa-reveal'));
    const caption = await page.locator('.wa-reveal .wa-zone-caption').textContent();
    if (caption.includes('추천')) fail(tag(`D caption claims recommendations: ${caption}`));
    if (shots) await page.locator('.wa-card').screenshot({ path: path.join(shots, `setup-no-rec-${width}.png`) });

    // E/F/G/H. abstained, no line (with decoys around it), beyond the picture (1.3 ft: past the ball dot's 1.1 ft bound,
    // so mitt and ball never use different picture limits), no-rec + abstained
    for (const [i, label, text] of [[3, 'E abstained', '읽지 못해'], [4, 'F no line', null], [5, 'G beyond picture', '존 그림 밖(포수 기준 오른쪽)'], [6, 'H no-rec abstained', '읽지 못해']]) {
      await open(GAME, i); await settle(); await noSetup(`${label} before reveal`);
      await doReveal();
      if (await marks().count()) fail(tag(`${label}: a mark was drawn`));
      const shown = await page.locator('.wa-setup').count() ? await page.locator('.wa-setup').innerText() : null;
      if (text === null ? shown !== null : !shown?.includes(text)) fail(tag(`${label}: setup block ${JSON.stringify(shown)}`));
      if (i === 6 && await page.locator('.wa-reveal .wa-zone').count()) fail(tag('H abstained pitch drew a zone'));
      results.push({ width, case: label, block: shown?.split('\n')[0] ?? null });
    }

    // I. failed reveal draws nothing; the retry draws the mark
    await open(GAME, 0); await settle();
    failNextReveal = true;
    await page.getByRole('button', { name: '실제 투구 공개' }).click();
    await page.locator('.wa-error').waitFor({ timeout: 5000 });
    await noSetup('I failed reveal');
    await doReveal();
    await expectMark('I retry', 0);

    // J. reveal 0 and move on before the setup answer arrives: it lands on pitch 1 unrevealed and draws nothing there
    delayMs = 1200;
    await open(GAME, 0);
    await doReveal();
    if (await marks().count()) fail(tag('J mark before setup arrived'));
    await page.getByRole('button', { name: '다음 공 →' }).click();
    await page.waitForFunction(() => window.location.hash === '#i=1');
    await settle();
    await noSetup('J arrived on unrevealed pitch 1');
    await doReveal(); await expectMark('J pitch 1', -1.0);
    await page.keyboard.press('ArrowLeft'); await page.waitForFunction(() => window.location.hash === '#i=0');
    await expectMark('J back to 0', 0);
    delayMs = 0;

    // K. 404 / 503 / aborted / an answer for another game: nothing drawn, nothing said, no error
    for (const m of ['404', '503', 'abort', 'wrong']) {
      mode = m;
      await open(GAME, 0); await settle(); await doReveal();
      await noSetup(`K ${m}`);
      if (await page.locator('.wa-error').count()) fail(tag(`K ${m}: error on screen`));
      results.push({ width, case: `K ${m}`, marks: await marks().count() });
    }
    mode = 'ok';

    // L. game A then game B (same at_bat:pitch numbers, no setup file): nothing carries over
    await open(GAME, 0); await settle(); await doReveal(); await expectMark('L game A', 0);
    await open(OTHER, 0); await settle(); await doReveal();
    await noSetup('L game B');

    // M. embedded watch page: same rule
    await open(GAME, 1, '&embed=1'); await settle();
    await noSetup('M embedded before reveal');
    await page.keyboard.press(' ');
    await page.locator('.wa-reveal').waitFor({ timeout: 5000 });
    await expectMark('M embedded', -1.0);
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    if (scrollWidth > width) fail(tag(`M horizontal overflow ${scrollWidth}`));

    // N. Showcase iframe: scene 0 before/after reveal, scene 1 (abstained), back to scene 0 starts unrevealed again
    await page.goto('about:blank');
    await page.goto('http://pz.test/showcase?scene=0');
    const frameFor = async () => { await page.locator('.sc-compare iframe').waitFor(); await page.locator('.sc-compare iframe').scrollIntoViewIfNeeded(); const f = page.frameLocator('.sc-compare iframe'); await f.locator('.wa-card').waitFor({ timeout: 8000 }); await page.waitForTimeout(200); return f; };
    let f = await frameFor();
    await noSetup('N scene 0 before reveal', f);
    await doReveal(f);
    await expectMark('N scene 0', 0.3, f);
    await page.locator('.sc-label select').selectOption('1');
    f = await frameFor();
    await noSetup('N scene 1 before reveal', f);
    await doReveal(f);
    if (await marks(f).count()) fail(tag('N scene 1 (abstained) drew a mark'));
    if (!(await f.locator('.wa-setup').innerText()).includes('읽지 못해')) fail(tag('N scene 1 abstain line missing'));
    await page.locator('.sc-label select').selectOption('0');
    f = await frameFor();
    await noSetup('N back to scene 0 before reveal', f);
    await doReveal(f);
    await expectMark('N scene 0 again', 0.3, f);
    if (!(await page.locator('.cv-card').count())) fail(tag('N M3 card missing on Showcase'));
    results.push({ width, case: 'N showcase', ok: true });

    if (errors.length) fail(tag(`page errors ${errors.join(' | ')}`));
    await context.close();
  }
} finally { await browser.close(); }
console.log(JSON.stringify({ checked: results.length, results }));
if (!process.exitCode) console.log('ok');
