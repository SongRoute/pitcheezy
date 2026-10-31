import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const scratch = mkdtempSync(join(tmpdir(), 'inning-result-test-'));
const tsc = join(webRoot, 'node_modules', '.bin', 'tsc');
const source = join(webRoot, 'src', 'inningResult.ts');

try {
  writeFileSync(join(scratch, 'package.json'), '{"type":"module"}\n');
  const compile = spawnSync(tsc, ['--target', 'ES2022', '--module', 'ES2022', '--moduleResolution', 'Bundler', '--skipLibCheck', '--strict', '--outDir', scratch, source], { encoding: 'utf8' });
  assert.equal(compile.status, 0, compile.stdout + compile.stderr);
  const { parseInningResult, buildInningPresentation } = await import(pathToFileURL(join(scratch, 'inningResult.js')));

  const hash = 'a'.repeat(64);
  const base = {
    schema_version: 'inning-result-v1', kind: 'conditional_keep', status: 'bounded', reason: 'unresolved_horizon_mass',
    linkage: { game_pk: 777063, keep_pitcher_id: 554430, official_game_date: '2025-07-21', anchor_kind: 'immediately_before_logged_pitching_substitution_action', anchor_time_utc: '2025-07-22T01:00:00Z', anchor_action_index: 1, first_observed_pitch_id: '777063:49:1' },
    scope: { horizon: 'inning_end', stopping_boundary: 'current_half_inning_or_game_end', value_target: 'final_game_win_probability', perspective: 'initial_defense', initial_defender: 'home', unit: 'probability', additive_to_pa: false },
    estimate: { point: null, lower: 0.521, upper: 0.535, interval_kind: 'unresolved_mass_bound' },
    coverage: { resolved_mass: 0.986, unresolved_mass: 0.014, unresolved_reasons: { maximum_pas: 0.014 }, model_calls: 85 },
    assumptions: ['keep_pitcher_fixed', 'prechange_lineup_fixed', 'frozen_train_repertoire_policy', 'no_future_substitutions'],
    profiles: { default_batter_ids: [691785, 701350] },
    replacement: { status: 'unavailable', value_pp: null, interval_pp: null, reason: 'actual_eligible_substitutes_unverified' },
    provenance: { usage: 'historical_research', source_result_sha256: hash, source_anchor_sha256: hash, model_bundle_sha256: hash, evaluation_identity: { provider_identity: hash, initial_state_and_count: { date: '2025-07-21', inning: 7, topbot: 'Top', outs: 0, bases: 0, home_score: 2, away_score: 2, balls: 0, strikes: 0 }, lineup_sha256: hash, evaluation_config_sha256: hash, policy_id: 'frozen_train_repertoire_frequency_v1', horizon: 'inning_end', initial_defender: 'home' } },
  };
  const clone = value => structuredClone(value);
  const bad = (name, change) => {
    const value = clone(base);
    change(value);
    assert.throws(() => parseInningResult(value), TypeError, name);
    assert.throws(() => buildInningPresentation(value), TypeError, `${name} presentation`);
  };

  assert.equal(parseInningResult(base), base);
  const presentation = buildInningPresentation(base);
  assert.equal(presentation.rangeLabel, '52.10%–53.50%');
  assert.equal(presentation.unresolvedLabel, '미해결 확률 1.40%');
  assert.equal(presentation.profileNote, '기본 프로필 2명');
  assert.equal(presentation.replacementLabel, '실제 교체 효과 미측정');
  assert.ok(presentation.assumptions.includes('현 투수 유지 + 당시 타순 유지'));
  assert.ok(presentation.assumptions.includes('기존 구종 사용 비율 유지'));
  assert.doesNotMatch(JSON.stringify(presentation), /%p|신뢰구간|교체 우위|midpoint|SHA|554430/);

  const unavailable = clone(base);
  unavailable.status = 'unavailable';
  unavailable.reason = 'missing_evaluation_result';
  unavailable.estimate.lower = null;
  unavailable.estimate.upper = null;
  unavailable.coverage.resolved_mass = null;
  unavailable.coverage.unresolved_mass = null;
  unavailable.coverage.unresolved_reasons = {};
  unavailable.coverage.model_calls = null;
  assert.equal(parseInningResult(unavailable), unavailable);
  const absent = buildInningPresentation(unavailable);
  assert.equal(absent.rangeLabel, null);
  assert.equal(absent.unresolvedLabel, null);
  assert.equal(absent.reasonLabel, '계산 결과 없음 · 평가 결과가 없습니다');
  const offset = clone(base);
  offset.linkage.anchor_time_utc = '2025-07-22T01:00:00+00:00';
  assert.equal(parseInningResult(offset), offset);
  const away = clone(base);
  away.scope.initial_defender = 'away';
  away.provenance.evaluation_identity.initial_defender = 'away';
  away.provenance.evaluation_identity.initial_state_and_count.topbot = 'Bot';
  assert.equal(parseInningResult(away), away);
  assert.match(buildInningPresentation(away).scope, /원정 수비팀/);

  bad('null payload', value => { value.estimate = null; });
  bad('unknown schema', value => { value.schema_version = 'inning-result-v2'; });
  bad('extra key', value => { value.value_pp = 10; });
  bad('percent instead of probability', value => { value.estimate.lower = 52.1; });
  bad('boolean number', value => { value.coverage.model_calls = true; });
  bad('missing lower', value => { value.estimate.lower = null; });
  bad('incorrect unresolved mass', value => { value.coverage.unresolved_mass = 0.1; });
  bad('incorrect reason mass', value => { value.coverage.unresolved_reasons.maximum_pas = 0.01; });
  bad('incorrect upper bound', value => { value.estimate.upper = 0.536; });
  bad('replacement effect', value => { value.replacement.value_pp = 1; });
  bad('mismatched defender', value => { value.provenance.evaluation_identity.initial_defender = 'away'; });
  bad('mismatched model', value => { value.provenance.evaluation_identity.provider_identity = 'b'.repeat(64); });
  bad('mismatched game date', value => { value.provenance.evaluation_identity.initial_state_and_count.date = '2025-07-22'; });
  bad('invalid date', value => { value.linkage.official_game_date = '2025-02-30'; });
  bad('invalid UTC calendar date', value => { value.linkage.anchor_time_utc = '2025-02-30T01:00:00Z'; });
  bad('invalid UTC clock', value => { value.linkage.anchor_time_utc = '2025-07-22T25:00:00Z'; });
  bad('non-UTC offset', value => { value.linkage.anchor_time_utc = '2025-07-22T10:00:00+09:00'; });
  bad('pitch ID for other game', value => { value.linkage.first_observed_pitch_id = '777064:49:1'; });
  bad('pitch ID at second pitch', value => { value.linkage.first_observed_pitch_id = '777063:49:2'; });
  bad('wrong policy', value => { value.provenance.evaluation_identity.policy_id = 'new_policy'; });
  bad('wrong half inning defender', value => { value.provenance.evaluation_identity.initial_state_and_count.topbot = 'Bot'; });
  bad('noncanonical half inning', value => { value.provenance.evaluation_identity.initial_state_and_count.topbot = 'Bottom'; });
  bad('duplicate profile', value => { value.profiles.default_batter_ids = [691785, 691785]; });
  bad('different assumptions', value => { value.assumptions.pop(); });
  bad('unavailable with stale value', value => { Object.assign(value, unavailable, { estimate: clone(base.estimate) }); });

  if (process.argv[2]) {
    const fixtureDir = resolve(process.argv[2]);
    const bounded = JSON.parse(readFileSync(join(fixtureDir, 'bounded.json'), 'utf8'));
    const missing = JSON.parse(readFileSync(join(fixtureDir, 'development_unavailable.json'), 'utf8'));
    assert.equal(parseInningResult(bounded), bounded);
    assert.equal(parseInningResult(missing), missing);
    const shown = buildInningPresentation(bounded);
    assert.equal(shown.rangeLabel, '52.10%–53.47%');
    assert.equal(shown.unresolvedLabel, '미해결 확률 1.36%');
    assert.equal(buildInningPresentation(missing).rangeLabel, null);
    assert.equal(buildInningPresentation(missing).reasonLabel, '계산 결과 없음 · 평가 결과가 없습니다');
  }
  if (process.argv[3]) {
    const response = JSON.parse(readFileSync(resolve(process.argv[3]), 'utf8'));
    assert.equal(response.schema_version, 'inning-decision-v1');
    assert.equal(response.mode, 'historical_decision_review');
    assert.equal(response.context.phase, 'before_pitching_change');
    assert.equal(response.context.linkage.game_pk, response.result.linkage.game_pk);
    assert.deepEqual(response.context.initial_state, response.result.provenance.evaluation_identity.initial_state_and_count);
    parseInningResult(response.result);
    const shown = buildInningPresentation(response.result);
    assert.equal(shown.rangeLabel, '52.10%–53.47%');
    assert.equal(shown.unresolvedLabel, '미해결 확률 1.36%');
    assert.equal(shown.replacementLabel, '실제 교체 효과 미측정');
  }
  console.log('inning result contract: ok');
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
