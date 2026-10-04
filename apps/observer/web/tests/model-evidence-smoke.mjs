// Synthetic local check of the model/recommendation evidence card on /showcase: the pinned JSON against the committed result
// files (when the repo path is given), separate prediction/policy wording and limits, collapsed by default, no links, layout.
// Usage: node tests/model-evidence-smoke.mjs [screenshot-dir]   env: DIST (default ../dist), MODEL_EVIDENCE_REPO (optional main repo root)
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { createHash } from 'node:crypto';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = process.env.DIST ?? path.join(here, '..', 'dist');
const ev = JSON.parse(readFileSync(path.join(here, '../src/model-evidence.json'), 'utf8'));
assert.equal(ev.researchMix.usedInService, false, 'research mix is not the service model');
assert.equal(ev.policy.label, 'NO_EVIDENCE_OF_IMPROVEMENT');
assert(ev.policy.deltaBounds[1] < ev.policy.improvementThreshold, 'upper bound below the improvement threshold');
// the card prints magnitudes with fixed words (낮음/작습니다), so pin the directions those words claim
assert(ev.prediction.deltaNll < 0 && ev.prediction.deltaNllCi95[1] < 0, 'G0 below frequency, whole interval');
assert(ev.researchMix.deltaNllVsG0 < 0 && ev.researchMix.deltaNllVsG0Ci95[1] < 0, 'mix below G0, whole interval');
assert(ev.prediction.frequency.ece10 < ev.prediction.g0.ece10, 'frequency ECE smaller than G0');
assert.equal(ev.policy.evaluatedPlateAppearances, ev.policy.plateAppearances - ev.policy.excludedUnknownPitcher, 'evaluated PA = all - excluded');

if (process.env.MODEL_EVIDENCE_REPO) {  // exact identity with the committed results (no rounding in the pinned copy)
  const read = rel => {
    const raw = readFileSync(path.join(process.env.MODEL_EVIDENCE_REPO, rel));
    assert.equal(createHash('sha256').update(raw).digest('hex'), ev.source.files[rel], `${rel} sha256`);
    return raw;
  };
  const p11 = JSON.parse(read('results/EXP-P11-001.json')), w = p11.estimands.whole, p = ev.prediction;
  assert.equal(p11.experiment_id, p.experimentId); assert.equal(p11.scope, p.scope);
  assert.deepEqual([p11.population.requested_pitches, p11.population.eligible_pitches], [p.requestedPitches, p.pitches], 'eligible of requested');
  assert.deepEqual([w.n, w.games, w.g0.log_loss, w.g0.brier_multiclass, w.g0.top_label_ece10, w.frequency.log_loss, w.frequency.brier_multiclass, w.frequency.top_label_ece10],
    [p.pitches, p.games, p.g0.nll, p.g0.brier, p.g0.ece10, p.frequency.nll, p.frequency.brier, p.frequency.ece10]);
  assert.deepEqual([w.paired.nll.delta, w.paired.nll.ci95, w.paired.brier.delta, w.paired.draws], [p.deltaNll, p.deltaNllCi95, p.deltaBrier, p.bootstrapDraws]);
  assert.deepEqual([w.seed_nll_deltas.length, w.N.negative_seeds], [p.seeds, p.negativeSeeds]);
  assert.equal(p11.descriptive.class_reporting.eligible.length, p.outcomeClasses, 'outcome classes');
  const p16 = JSON.parse(read('results/EXP-P16-001.json')), mb = p16.mlb_dev_blended, mc = mb.comparison['G15-direct-g0-mix'], m = ev.researchMix;
  assert.deepEqual([p16.experiment_id, mb.n, mb.games, mc.paired.nll.delta, mc.paired.nll.ci95, mc.seed_deltas.length, mc.seed_deltas.filter(x => x < 0).length],
    [m.experimentId, m.pitches, m.games, m.deltaNllVsG0, m.deltaNllVsG0Ci95, m.seeds, m.negativeSeeds]);
  const ope = JSON.parse(read('results/MLB-2026-OPE-v1.json')), o = ev.policy;
  assert.deepEqual([ope.id, ope.attempt, ope.games, ope.pas, ope.reasons['EXCLUDED_PRE_START:UNSUPPORTED_UNKNOWN_PITCHER'], ope.L1R_delta_bounds, ope.label, ope.labels.primary.mei],
    [o.id, o.attempt, o.games, o.plateAppearances, o.excludedUnknownPitcher, o.deltaBounds, o.label, o.improvementThreshold]);
  assert.equal(ope.status.COMPLETE + ope.status.CENSORED, o.evaluatedPlateAppearances, 'E0 = complete + censored');
  assert(ope.interpretation.includes('exposed-window'), 'OPE records its exposed window (D87)');
  const matrix = read('docs/ML_EXPERIMENT_MATRIX.md').toString();
  assert(matrix.includes('DEV `07-01~09-30`') && matrix.includes('TRAIN `2023-05-15~2025-04-30`'), 'DEV window in the matrix document');
  assert.deepEqual(p.devWindow, ['2025-07-01', '2025-09-30']);
  // conditioning sentences: code/contract facts read from the pinned commit (not the moving working tree)
  const c = ev.conditioning;
  assert.equal(c.deliveryDraws, 400);
  assert.equal(c.sourceCommit, ev.source.repoCommit, 'conditioning sources at the commit the card cites');
  // every sentence keeps its source: dropping a pin (or its anchors) must fail, not silently check less
  assert.deepEqual(Object.keys(c.files).sort(), ['docs/contracts/MLB-2026-POLICY-PREPARATION-v1.md',
    'experiments/pitchmdp/pitchmdp/matrix_features.py', 'experiments/pitchmdp/pitchmdp/matrix_g0_whole.py',
    'experiments/pitchmdp/pitchmdp/matrix_g0_whole_metrics.py', 'experiments/pitchmdp/pitchmdp/matrix_policy.py',
    'experiments/pitchmdp/pitchmdp/sequence_data.py', 'experiments/pitchmdp/pitchmdp/sequence_delivery.py',
    'experiments/pitchmdp/scripts/run_sequence_frequency_baselines.py'], 'conditioning source files');
  assert.deepEqual(Object.keys(c.anchors).sort(), Object.keys(c.files).sort(), 'one anchor list per pinned file');
  for (const [rel, sha] of Object.entries(c.files)) {
    assert(c.anchors[rel].length > 0, `${rel} has anchors`);
    const raw = execFileSync('git', ['-C', process.env.MODEL_EVIDENCE_REPO, 'show', `${c.sourceCommit}:${rel}`], { maxBuffer: 1 << 24 });
    assert.equal(createHash('sha256').update(raw).digest('hex'), sha, `${rel} sha256 at ${c.sourceCommit.slice(0, 7)}`);
    for (const anchor of c.anchors[rel]) assert(raw.toString().includes(anchor), `${rel} anchor: ${anchor}`);
  }
  console.log('source identity: ok');
} else console.log('source identity: skipped (MODEL_EVIDENCE_REPO not set)');

const timeline = { schema: 'pitcheezy-watch-along-v1', badge: '검증 전 실험 버전',
  game: { game_pk: 849843, away_team: 'Chicago Cubs', home_team: 'San Diego Padres', date: '2026-09-29' }, decisions: [],
  policy: { name: 'synthetic', identity_sha256: '0'.repeat(64), tau: .1, note: '합성 검사' },
  location: { note: '근사', zone_bounds: { bottom: 1.6, top: 3.39 } }, we_note: '합성 검사', coverage: { ready: 0, pitch_decisions: 0 } };
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const browser = await chromium.launch();
try {
  for (const width of [1440, 390, 360]) {
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
    const card = page.locator('details.model-card');
    await card.waitFor();
    assert.equal(await page.locator('details.model-card + details.cv-card').count(), 1, 'model card sits right before the CV card');
    assert.equal(await card.evaluate(d => d.open), false, 'collapsed by default');
    const summary = await card.locator('summary').innerText();
    for (const t of ['예측 모델·추천 검증 결과', '2025 평가 311,721구에서 상황 빈도 기준보다 로그손실 0.0137 낮음', '2026 정규시즌 1회 평가 2,429경기 · 개선 근거 없음'])
      assert(summary.includes(t), `summary: ${t}`);
    assert(await card.locator('.model-list').first().isHidden(), 'detail hidden while collapsed');
    if (process.argv[2]) await card.screenshot({ path: `${process.argv[2]}/model-card-closed-${width}.png` });
    await card.locator('summary').click();
    await card.locator('.model-list').first().waitFor();
    const text = await card.innerText();
    for (const t of ['예측이 더 정확한 것과 추천이 경기 결과를 좋게 만드는 것은 서로 다른 질문입니다', '결과 10종',
      '2025-07-01 ~ 2025-09-30 전체 MLB 중 평가 조건을 만족한 311,721구(전체 341,941구 중) · 1,161경기', '서비스 모델 1.4904 · 상황 빈도 기준 1.5041',
      '차이 −0.0137 (95% 구간 −0.0144 ~ −0.0129, 경기 단위 재표본 10,000회, 시드 5/5개 모두 빈도 기준보다 낮음)',
      '빈도 기준 0.0017이 서비스 모델 0.0087보다 작습니다', '로그손실이 0.0041 더 낮았습니다(95% 구간 −0.0044 ~ −0.0038, 시드 5/5)',
      '서비스와 추천에는 아직 쓰지 않습니다', '2026 정규시즌 2,429경기 · 184,288타석 중 시작 전 제외를 뺀 149,525타석', '타석당 수비 승리확률 차이 범위 −0.0012 ~ +0.0009',
      '노출 이력이 있어(D87), 처음 보는 자료에서의 평가는 아닙니다', '범위의 위쪽 끝(+0.0009)도 개선 문턱 +0.001보다 작아 개선 근거 없음',
      '해가 없다는 증명도 아닙니다', '위에서 뺀 34,763타석은 2025년까지 학습 기록이 없는 투수로 시작한 타석입니다',
      '화면에서도 이런 투수에게는 추천을 만들지 않습니다', '모든 면에서 나은 것은 아닙니다', '그 가정이 맞을 때만 인과 효과로 읽을 수 있습니다', '여러 번 본 개발용 자료', '공개 뒤 승리확률 변화는 추천의 효과를 보여 주지 않습니다',
      '독립 검토 전이라 이 카드에 넣지 않았습니다',
      '던질 구종이 정해졌을 때 다음 공 한 개의 결과 10종', '공의 위치·구속·회전 등은 던지기 전에 알 수 없어', '학습 기간 기록에서 뽑은 투구 400개로 평균합니다', '평가 투구마다 실제로 던진 구종을 넣어 계산했습니다',
      '추천은 던질 구종만 고릅니다', '타석이 끝날 때까지의 수비 승리확률을 모의 계산하고', '위치 표시는 그 투수의 2025년까지 실제 투구 분포에서 잡은 근사이며, 평가하지 않았습니다', '레포 b38bea6', 'EXP-P11-001 44dae07c', 'MLB-2026-OPE-v1 0bb8fcd0', '결정 D87·D131'])
      assert(text.includes(t), `detail: ${t}`);
    for (const t of ['학습·선택에는 쓰지', '예측은 좋아졌', '모두 개선)', '공이 들어갈 위치는'])  // overclaims removed in review (2026 exposure D87; 'better than frequency', not 'improved')
      assert(!text.includes(t), `overclaim: ${t}`);
    assert.equal(await card.locator('a, button, iframe, video').count(), 0, 'no navigation from the evidence card');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal overflow');
    const box = await card.boundingBox(); assert(box.x >= 0 && box.x + box.width <= width, 'card within viewport');
    if (process.argv[2]) {
      await card.scrollIntoViewIfNeeded();
      await card.screenshot({ path: `${process.argv[2]}/model-card-open-${width}.png` });
    }
    await card.locator('summary').click();
    assert.equal(await card.evaluate(d => d.open), false, 'collapses again');
    assert.equal(outside, 0, 'no request leaves the synthetic host'); assert.deepEqual(errors, []);
    await page.close();
  }
  console.log('model evidence smoke: ok (desktop 1440 + phone 390/360, synthetic timeline)');
} finally { await browser.close(); }
