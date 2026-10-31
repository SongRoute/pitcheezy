import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = resolve(web, '../../..');
const scratch = mkdtempSync(join(tmpdir(), 'inning-client-'));
const originalFetch = globalThis.fetch;
try {
  execFileSync(join(web, 'node_modules/.bin/tsc'), ['--target', 'ES2022', '--module', 'CommonJS', '--moduleResolution', 'node', '--skipLibCheck', '--strict', '--outDir', scratch, join(web, 'src/inningDecisionApi.ts')]);
  const api = createRequire(import.meta.url)(join(scratch, 'inningDecisionApi.js'));
  const catalog = JSON.parse(readFileSync(join(root, 'results/C-D-API-001/catalog.json')));
  const resolved = JSON.parse(readFileSync(join(root, 'results/C-D-API-001/resolved.json')));
  const selected = catalog.decisions[0];
  const reply = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
  const games = { schema_version: 'inning-decision-v1', mode: 'historical_decision_review', games: [{ game_id: 777063, date: '2025-07-21', decision_count: 1 }] };
  globalThis.fetch = async path => { assert.equal(path, '/api/inning-decision-games'); return reply(games); };
  assert.deepEqual(await api.listDecisionGames(), games.games);
  globalThis.fetch = async path => { assert.equal(path, '/api/inning-decisions?game_id=777063'); return reply(catalog); };
  assert.deepEqual(await api.listDecisions(777063), catalog.decisions);
  globalThis.fetch = async (path, init) => {
    assert.equal(path, `/api/inning-decisions/${selected.decision_id}/resolve`);
    assert.equal(init.method, 'POST');
    assert.deepEqual(JSON.parse(init.body), { revision: 1, context: selected.context });
    return reply(resolved);
  };
  assert.deepEqual(await api.resolveDecision(selected), resolved.result);
  for (const mutate of [
    x => { x.decision_id = 'inning-decision-' + 'a'.repeat(64); },
    x => { x.revision = 2; },
    x => { x.context.initial_state.outs = 1; },
    x => { x.result.linkage.keep_pitcher_id = 1; },
    x => { x.result.provenance.evaluation_identity.initial_state_and_count.home_score = 3; },
    x => { x.result.linkage.anchor_time_utc = '2025-07-22T00:22:07.083Z'; },
    x => { x.result.estimate.lower = 52.1; },
  ]) {
    const wrong = structuredClone(resolved); mutate(wrong);
    globalThis.fetch = async () => reply(wrong);
    await assert.rejects(api.resolveDecision(selected), /저장된 교체 기록의 형식/);
  }
  const wrongGame = structuredClone(catalog);
  wrongGame.decisions[0].context.linkage.game_pk = 777064;
  wrongGame.decisions[0].context.linkage.first_observed_pitch_id = '777064:49:1';
  globalThis.fetch = async () => reply(wrongGame);
  await assert.rejects(api.listDecisions(777063));
  globalThis.fetch = async () => reply({ ...games, games: [...games.games, ...games.games] });
  await assert.rejects(api.listDecisionGames());
  globalThis.fetch = async () => reply({ detail: '상태가 다릅니다.' }, 409);
  await assert.rejects(api.resolveDecision(selected), /상태가 다릅니다/);
  globalThis.fetch = async () => { throw new TypeError('offline'); };
  await assert.rejects(api.listDecisionGames(), /서버에 연결하지 못했어요/);
  console.log('inning decision client: passed real fixture identity, 7 response mismatch cases, wrong game, duplicates, 409, offline');
} finally { globalThis.fetch = originalFetch; rmSync(scratch, { recursive: true, force: true }); }
