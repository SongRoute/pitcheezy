// Synthetic regression for the review's shared result vocabulary and selected-pitch record boundaries.
// No game data, network or new test dependency. Render the actual React sheets, not copied display logic.
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url), { createElement } = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const built = await build({ stdin: { contents: "export * from './model'; export * from './LivePanels';", resolveDir: fileURLToPath(new URL('../src/service', import.meta.url)) }, bundle: true, write: false, format: 'cjs', platform: 'node', packages: 'external', jsx: 'automatic' });
const module = { exports: {} };
new Function('require', 'module', 'exports', built.outputFiles[0].text)(require, module, module.exports);
const { resultKind, pasAsOf, RecordSheet, PitchMixSheet } = module.exports;
let checks = 0;
for (const [kind, descriptions] of [
  ['ball', ['ball', 'blocked_ball', 'pitchout', 'hit_by_pitch', 'intent_ball', 'automatic_ball', 'Ball', 'Ball In Dirt', 'Hit By Pitch', 'Intentional Ball']],
  ['strike', ['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt', 'foul_pitchout', 'bunt_foul_tip', 'missed_bunt', 'automatic_strike', 'Called Strike', 'Swinging Strike', 'Swinging Strike (Blocked)', 'Foul', 'Foul Tip', 'Foul Bunt', 'Missed Bunt']],
  ['inplay', ['hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score', 'In play, out(s)', 'In play, no out', 'In play, run(s)']],
  ['unknown', [undefined, null, '', 'future_code', 'hit_into_play_unknown', 'Unknown Strike', 'Unknown Ball']],
]) for (const description of descriptions) { assert.equal(resultKind(description), kind, String(description)); checks++; }
const pitch = (index, pa, name, receipt) => ({ index, key: `p${index}`, pa_key: pa, at_bat_number: pa, pitch_number: index % 2 + 1,
  batter: { id: pa, name }, pitcher: { id: 9, name: '투수' }, situation: { inning: 1, half: 'Top' }, receipt,
  actual: { pitch_type: 'FF', pitch_label: '포심', speed_mph: 90, x: null, z: null } });
const pa = (id, first, last) => ({ pa_key: id, at_bat_number: id, inning: 1, half: 'Top', batter: `타자${id}`, pitcher: '투수', first, last, complete: true, result: '미래 결과', pitch_count: 9 });
const detail = { game: {}, current_context: null, zone_bounds: { bottom: 1.6, top: 3.4 },
  pitches: [pitch(0, 1, '첫 타자', { status: 'unconfirmed', corrected: true }), pitch(1, 1, '첫 타자', { status: 'received', corrected: true, reason: 'other_reason' }),
    pitch(2, 2, '다음 타자', { status: 'received', corrected: true, reason: 'held_last_accepted' }), pitch(3, 2, '다음 타자', { status: 'received', corrected: true })],
  pas: [pa(1, 0, null), pa(2, 2, 3), pa(3, 4, 5), { ...pa(0, null, null), result: '고의 볼넷', pitch_count: 0 }] };
const first = pasAsOf(detail, 0);
assert.deepEqual(first.map(p => p.pa_key), [1, 0]); checks++;
assert.deepEqual([first[0].complete, first[0].result, first[0].pitch_count], [false, '타석 진행 중', 1]); checks++;
const second = pasAsOf(detail, 2);
assert.equal(second[0], detail.pas[0], 'missing last on a strictly earlier PA does not erase its known result'); checks++;
assert.deepEqual([second[1].complete, second[1].result, second[1].pitch_count], [false, '타석 진행 중', 1]); checks++;
assert.equal(pasAsOf(detail, 3)[1], detail.pas[1], 'completed PA at its terminal pitch stays complete'); checks++;
assert.equal(pasAsOf(detail, null), detail.pas, 'latest view retains published full records'); checks++;
const oldWindow = globalThis.window;
globalThis.window = { matchMedia: () => ({ matches: false }) };
try {
  const render = (Component, extra) => renderToStaticMarkup(createElement(Component, { detail, fold: false, onClose() {}, onPick() {}, ...extra }));
  const firstHtml = render(RecordSheet, { asOfIndex: 0 });
  const laterHtml = render(RecordSheet, { asOfIndex: 2 });
  assert.match(firstHtml, /고른 공\(1회 초 · 첫 타자 · 1구\)까지/); checks++;
  assert.match(laterHtml, /고른 공\(1회 초 · 다음 타자 · 1구\)까지/); checks++;
  const allHtml = render(RecordSheet, { asOfIndex: null });
  assert.match(allHtml, /정정된 기록 4구\(투구 카드에 ‘정정됨’ 표시 1구, 카드에 표시하지 않은 3구\)/); checks++;
  assert.match(allHtml, /보류한 공 1구/); checks++;
  assert.doesNotMatch(allHtml, /보류 중이라 표시하지 않은/); checks++;
  assert.match(render(PitchMixSheet, { pitcherId: 9, pitcherName: '투수', asOfIndex: 2 }), /고른 공\(1회 초 · 다음 타자 · 1구\)까지/); checks++;
} finally { globalThis.window = oldWindow; }
console.log(`ui-review-followups-check: ${checks} checks passed`);
