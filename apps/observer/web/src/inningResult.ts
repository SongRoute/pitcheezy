/** The isolated C → D conditional keep handoff. Values are probabilities, not percentage points. */
export type InningResult = {
  schema_version: 'inning-result-v1';
  kind: 'conditional_keep';
  status: 'bounded' | 'unavailable';
  reason: string;
  linkage: {
    game_pk: number;
    keep_pitcher_id: number;
    official_game_date: string;
    anchor_kind: 'immediately_before_logged_pitching_substitution_action';
    anchor_time_utc: string;
    anchor_action_index: number;
    first_observed_pitch_id: string;
  };
  scope: {
    horizon: 'inning_end';
    stopping_boundary: 'current_half_inning_or_game_end';
    value_target: 'final_game_win_probability';
    perspective: 'initial_defense';
    initial_defender: 'home' | 'away';
    unit: 'probability';
    additive_to_pa: false;
  };
  estimate: {
    point: null;
    lower: number | null;
    upper: number | null;
    interval_kind: 'unresolved_mass_bound';
  };
  coverage: {
    resolved_mass: number | null;
    unresolved_mass: number | null;
    unresolved_reasons: Record<string, number>;
    model_calls: number | null;
  };
  assumptions: [
    'keep_pitcher_fixed',
    'prechange_lineup_fixed',
    'frozen_train_repertoire_policy',
    'no_future_substitutions',
  ];
  profiles: { default_batter_ids: number[] };
  replacement: {
    status: 'unavailable';
    value_pp: null;
    interval_pp: null;
    reason: 'actual_eligible_substitutes_unverified';
  };
  provenance: {
    usage: 'historical_research';
    source_result_sha256: string;
    source_anchor_sha256: string;
    model_bundle_sha256: string;
    evaluation_identity: {
      provider_identity: string;
      initial_state_and_count: {
        date: string;
        inning: number;
        topbot: 'Top' | 'Bot';
        outs: number;
        bases: number;
        home_score: number;
        away_score: number;
        balls: number;
        strikes: number;
      };
      lineup_sha256: string;
      evaluation_config_sha256: string;
      policy_id: string;
      horizon: 'inning_end';
      initial_defender: 'home' | 'away';
    };
  };
};

export type InningPresentation = {
  title: string;
  scope: string;
  valueLabel: string;
  rangeLabel: string | null;
  unresolvedLabel: string | null;
  profileNote: string;
  assumptions: string[];
  replacementLabel: string;
  reasonLabel: string | null;
};

function invalid(path: string): never {
  throw new TypeError(`Invalid inning-result-v1: ${path}`);
}

function record(value: unknown, path: string, keys?: readonly string[]): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) invalid(path);
  const result = value as Record<string, unknown>;
  if (keys && (Object.keys(result).length !== keys.length || keys.some(key => !Object.hasOwn(result, key)))) invalid(path);
  return result;
}

function literal<T extends string | boolean | null>(value: unknown, expected: T, path: string): T {
  if (value !== expected) invalid(path);
  return expected;
}

function string(value: unknown, path: string): string {
  if (typeof value !== 'string' || !value.trim() || value !== value.trim()) invalid(path);
  return value;
}

function integer(value: unknown, path: string, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) invalid(path);
  return value;
}

function probability(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) invalid(path);
  return value;
}

function sha(value: unknown, path: string): string {
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value)) invalid(path);
  return value;
}

function date(value: unknown, path: string): string {
  const s = string(value, path);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s) || Number.isNaN(Date.parse(`${s}T00:00:00Z`)) || new Date(`${s}T00:00:00Z`).toISOString().slice(0, 10) !== s) invalid(path);
  return s;
}

function utcInstant(value: unknown, path: string): string {
  const s = string(value, path);
  const parts = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|\+00:00)$/.exec(s);
  if (!parts) invalid(path);
  date(parts[1], path);
  if (Number(parts[2]) > 23 || Number(parts[3]) > 59 || Number(parts[4]) > 59 || Number.isNaN(Date.parse(s))) invalid(path);
  return s;
}

function defender(value: unknown, path: string): 'home' | 'away' {
  if (value !== 'home' && value !== 'away') invalid(path);
  return value;
}

const assumptions = [
  'keep_pitcher_fixed', 'prechange_lineup_fixed',
  'frozen_train_repertoire_policy', 'no_future_substitutions',
] as const;
const close = (a: number, b: number): boolean => Math.abs(a - b) <= 1e-8;

/** Throws TypeError for any malformed payload; never turns a bad result into unavailable. */
export function parseInningResult(input: unknown): InningResult {
  const root = record(input, 'root', ['schema_version', 'kind', 'status', 'reason', 'linkage', 'scope', 'estimate', 'coverage', 'assumptions', 'profiles', 'replacement', 'provenance']);
  literal(root.schema_version, 'inning-result-v1', 'schema_version');
  literal(root.kind, 'conditional_keep', 'kind');
  const status = root.status;
  if (status !== 'bounded' && status !== 'unavailable') invalid('status');
  string(root.reason, 'reason');

  const linkage = record(root.linkage, 'linkage', ['game_pk', 'keep_pitcher_id', 'official_game_date', 'anchor_kind', 'anchor_time_utc', 'anchor_action_index', 'first_observed_pitch_id']);
  integer(linkage.game_pk, 'linkage.game_pk', 1);
  integer(linkage.keep_pitcher_id, 'linkage.keep_pitcher_id', 1);
  const officialDate = date(linkage.official_game_date, 'linkage.official_game_date');
  literal(linkage.anchor_kind, 'immediately_before_logged_pitching_substitution_action', 'linkage.anchor_kind');
  utcInstant(linkage.anchor_time_utc, 'linkage.anchor_time_utc');
  integer(linkage.anchor_action_index, 'linkage.anchor_action_index');
  const pitchId = string(linkage.first_observed_pitch_id, 'linkage.first_observed_pitch_id');
  if (!new RegExp(`^${linkage.game_pk}:[1-9]\\d*:1$`).test(pitchId)) invalid('linkage.first_observed_pitch_id');

  const scope = record(root.scope, 'scope', ['horizon', 'stopping_boundary', 'value_target', 'perspective', 'initial_defender', 'unit', 'additive_to_pa']);
  literal(scope.horizon, 'inning_end', 'scope.horizon');
  literal(scope.stopping_boundary, 'current_half_inning_or_game_end', 'scope.stopping_boundary');
  literal(scope.value_target, 'final_game_win_probability', 'scope.value_target');
  literal(scope.perspective, 'initial_defense', 'scope.perspective');
  const initialDefender = defender(scope.initial_defender, 'scope.initial_defender');
  literal(scope.unit, 'probability', 'scope.unit');
  literal(scope.additive_to_pa, false, 'scope.additive_to_pa');

  const estimate = record(root.estimate, 'estimate', ['point', 'lower', 'upper', 'interval_kind']);
  literal(estimate.point, null, 'estimate.point');
  literal(estimate.interval_kind, 'unresolved_mass_bound', 'estimate.interval_kind');
  const coverage = record(root.coverage, 'coverage', ['resolved_mass', 'unresolved_mass', 'unresolved_reasons', 'model_calls']);
  const reasons = record(coverage.unresolved_reasons, 'coverage.unresolved_reasons');
  let reasonSum = 0;
  for (const [code, mass] of Object.entries(reasons)) {
    if (!code.trim() || code !== code.trim()) invalid('coverage.unresolved_reasons');
    reasonSum += probability(mass, `coverage.unresolved_reasons.${code}`);
  }
  if (status === 'bounded') {
    const lower = probability(estimate.lower, 'estimate.lower');
    const upper = probability(estimate.upper, 'estimate.upper');
    const resolved = probability(coverage.resolved_mass, 'coverage.resolved_mass');
    const unresolved = probability(coverage.unresolved_mass, 'coverage.unresolved_mass');
    integer(coverage.model_calls, 'coverage.model_calls');
    if (lower > upper || lower > resolved + 1e-8 || !close(resolved + unresolved, 1) || !close(upper - lower, unresolved) || !close(reasonSum, unresolved)) invalid('bounded mass relationship');
  } else if (estimate.lower !== null || estimate.upper !== null || coverage.resolved_mass !== null || coverage.unresolved_mass !== null || coverage.model_calls !== null || Object.keys(reasons).length !== 0) {
    invalid('unavailable numeric fields');
  }

  const suppliedAssumptions = root.assumptions;
  if (!Array.isArray(suppliedAssumptions) || suppliedAssumptions.length !== assumptions.length || assumptions.some((item, i) => suppliedAssumptions[i] !== item)) invalid('assumptions');
  const profiles = record(root.profiles, 'profiles', ['default_batter_ids']);
  if (!Array.isArray(profiles.default_batter_ids)) invalid('profiles.default_batter_ids');
  const ids = profiles.default_batter_ids.map((id, i) => integer(id, `profiles.default_batter_ids.${i}`, 1));
  if (new Set(ids).size !== ids.length) invalid('profiles.default_batter_ids duplicates');

  const replacement = record(root.replacement, 'replacement', ['status', 'value_pp', 'interval_pp', 'reason']);
  literal(replacement.status, 'unavailable', 'replacement.status');
  literal(replacement.value_pp, null, 'replacement.value_pp');
  literal(replacement.interval_pp, null, 'replacement.interval_pp');
  literal(replacement.reason, 'actual_eligible_substitutes_unverified', 'replacement.reason');

  const provenance = record(root.provenance, 'provenance', ['usage', 'source_result_sha256', 'source_anchor_sha256', 'model_bundle_sha256', 'evaluation_identity']);
  literal(provenance.usage, 'historical_research', 'provenance.usage');
  sha(provenance.source_result_sha256, 'provenance.source_result_sha256');
  sha(provenance.source_anchor_sha256, 'provenance.source_anchor_sha256');
  const modelHash = sha(provenance.model_bundle_sha256, 'provenance.model_bundle_sha256');
  const identity = record(provenance.evaluation_identity, 'provenance.evaluation_identity', ['provider_identity', 'initial_state_and_count', 'lineup_sha256', 'evaluation_config_sha256', 'policy_id', 'horizon', 'initial_defender']);
  if (sha(identity.provider_identity, 'identity.provider_identity') !== modelHash) invalid('identity.provider_identity mismatch');
  sha(identity.lineup_sha256, 'identity.lineup_sha256');
  sha(identity.evaluation_config_sha256, 'identity.evaluation_config_sha256');
  literal(identity.policy_id, 'frozen_train_repertoire_frequency_v1', 'identity.policy_id');
  literal(identity.horizon, 'inning_end', 'identity.horizon');
  if (defender(identity.initial_defender, 'identity.initial_defender') !== initialDefender) invalid('identity.initial_defender mismatch');
  const state = record(identity.initial_state_and_count, 'identity.initial_state_and_count', ['date', 'inning', 'topbot', 'outs', 'bases', 'home_score', 'away_score', 'balls', 'strikes']);
  if (date(state.date, 'identity.initial_state_and_count.date') !== officialDate) invalid('identity date mismatch');
  integer(state.inning, 'identity.initial_state_and_count.inning', 1);
  if (state.topbot !== 'Top' && state.topbot !== 'Bot') invalid('identity.initial_state_and_count.topbot');
  if ((state.topbot === 'Top' ? 'home' : 'away') !== initialDefender) invalid('identity.initial_state_and_count defender mismatch');
  integer(state.outs, 'identity.initial_state_and_count.outs', 0, 2);
  integer(state.bases, 'identity.initial_state_and_count.bases', 0, 7);
  integer(state.home_score, 'identity.initial_state_and_count.home_score');
  integer(state.away_score, 'identity.initial_state_and_count.away_score');
  integer(state.balls, 'identity.initial_state_and_count.balls', 0, 3);
  integer(state.strikes, 'identity.initial_state_and_count.strikes', 0, 2);

  return input as InningResult;
}

const percent = (p: number): string => `${(p * 100).toFixed(2)}%`;

/** A deliberately small display model with no implied replacement effect or midpoint. */
export function buildInningPresentation(input: unknown): InningPresentation {
  const result = parseInningResult(input);
  const side = result.scope.initial_defender === 'home' ? '홈' : '원정';
  return {
    title: '조건부 이닝 전망',
    scope: `현재 반이닝 종료까지 · 초기 수비팀 관점 (${side} 수비팀)`,
    valueLabel: '조건을 고정한 경기 승리 확률 범위',
    rangeLabel: result.status === 'bounded' ? `${percent(result.estimate.lower!)}–${percent(result.estimate.upper!)}` : null,
    unresolvedLabel: result.status === 'bounded' ? `미해결 확률 ${percent(result.coverage.unresolved_mass!)}` : null,
    profileNote: `기본 프로필 ${result.profiles.default_batter_ids.length}명`,
    assumptions: ['현 투수 유지 + 당시 타순 유지', '기존 구종 사용 비율 유지', '이후 선수 교체 없음'],
    replacementLabel: '실제 교체 효과 미측정',
    reasonLabel: result.status === 'unavailable' ? (result.reason === 'missing_evaluation_result' ? '계산 결과 없음 · 평가 결과가 없습니다' : '계산 결과 없음 · 이유를 확인할 수 없습니다') : null,
  };
}
