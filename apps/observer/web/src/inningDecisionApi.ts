import { parseInningResult } from './inningResult';
import type { InningResult } from './inningResult';

export type DecisionGame = { game_id: number; date: string; decision_count: number };
export type DecisionContext = {
  phase: 'before_pitching_change';
  linkage: InningResult['linkage'];
  initial_state: InningResult['provenance']['evaluation_identity']['initial_state_and_count'];
};
export type DecisionSummary = { decision_id: string; revision: 1; context: DecisionContext };

function invalid(): never { throw new Error('저장된 교체 기록의 형식을 확인할 수 없습니다. 목록을 다시 불러와 주세요.'); }
function object(value: unknown, keys: string[]): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid();
  const row = value as Record<string, unknown>;
  if (Object.keys(row).length !== keys.length || keys.some(key => !Object.hasOwn(row, key))) invalid();
  return row;
}
function integer(value: unknown, minimum = 0, maximum = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > maximum) invalid();
  return value;
}
function date(value: unknown): string {
  if (typeof value !== 'string' || !/^\d{4}-\d\d-\d\d$/.test(value)) invalid();
  const parsed = new Date(`${value}T00:00:00Z`);
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value) invalid();
  return value;
}
function instant(value: unknown): string {
  if (typeof value !== 'string') invalid();
  const match = /^(\d{4}-\d\d-\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?(?:Z|\+00:00)$/.exec(value);
  if (!match) invalid();
  date(match[1]);
  if (+match[2] > 23 || +match[3] > 59 || +match[4] > 59) invalid();
  return `${match[1]}T${match[2]}:${match[3]}:${match[4]}.${(match[5] || '').padEnd(6, '0').slice(0, 6)}Z`;
}
function parseContext(input: unknown): DecisionContext {
  const context = object(input, ['phase', 'linkage', 'initial_state']);
  if (context.phase !== 'before_pitching_change') invalid();
  const linkage = object(context.linkage, ['game_pk', 'keep_pitcher_id', 'official_game_date', 'anchor_kind', 'anchor_time_utc', 'anchor_action_index', 'first_observed_pitch_id']);
  integer(linkage.game_pk, 1); integer(linkage.keep_pitcher_id, 1); integer(linkage.anchor_action_index);
  date(linkage.official_game_date); instant(linkage.anchor_time_utc);
  if (linkage.anchor_kind !== 'immediately_before_logged_pitching_substitution_action' || typeof linkage.first_observed_pitch_id !== 'string' || !new RegExp(`^${linkage.game_pk}:[1-9]\\d*:1$`).test(linkage.first_observed_pitch_id)) invalid();
  const state = object(context.initial_state, ['date', 'inning', 'topbot', 'outs', 'bases', 'home_score', 'away_score', 'balls', 'strikes']);
  if (date(state.date) !== linkage.official_game_date || (state.topbot !== 'Top' && state.topbot !== 'Bot')) invalid();
  integer(state.inning, 1); integer(state.outs, 0, 2); integer(state.bases, 0, 7);
  integer(state.home_score); integer(state.away_score); integer(state.balls, 0, 3); integer(state.strikes, 0, 2);
  return input as DecisionContext;
}
function summary(input: unknown): DecisionSummary {
  const row = object(input, ['decision_id', 'revision', 'context']);
  if (typeof row.decision_id !== 'string' || !/^inning-decision-[a-f0-9]{64}$/.test(row.decision_id) || row.revision !== 1) invalid();
  parseContext(row.context);
  return input as DecisionSummary;
}
function sameContext(left: DecisionContext, right: DecisionContext): boolean {
  const a = left.linkage, b = right.linkage;
  return left.phase === right.phase
    && (Object.keys(a) as (keyof typeof a)[]).every(key => key === 'anchor_time_utc' ? instant(a[key]) === instant(b[key]) : a[key] === b[key])
    && (Object.keys(left.initial_state) as (keyof typeof left.initial_state)[]).every(key => left.initial_state[key] === right.initial_state[key]);
}
function envelope(input: unknown, keys: string[]): Record<string, unknown> {
  const value = object(input, ['schema_version', 'mode', ...keys]);
  if (value.schema_version !== 'inning-decision-v1' || value.mode !== 'historical_decision_review') invalid();
  return value;
}

async function request(path: string, signal?: AbortSignal, body?: unknown): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(path, { signal, ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error('서버에 연결하지 못했어요. 잠시 후 다시 시도해 주세요.');
  }
  let data: unknown;
  try { data = await response.json(); } catch { throw new Error('서버 응답을 읽지 못했어요. 다시 시도해 주세요.'); }
  if (!response.ok) {
    const detail = data && typeof data === 'object' ? (data as { detail?: unknown }).detail : null;
    throw new Error(typeof detail === 'string' ? detail : '교체 기록을 불러오지 못했어요. 다시 시도해 주세요.');
  }
  return data;
}

export async function listDecisionGames(signal?: AbortSignal): Promise<DecisionGame[]> {
  const data = envelope(await request('/api/inning-decision-games', signal), ['games']);
  if (!Array.isArray(data.games)) invalid();
  const games = data.games.map(item => {
    const game = object(item, ['game_id', 'date', 'decision_count']);
    integer(game.game_id, 1); date(game.date); integer(game.decision_count, 1);
    return item as DecisionGame;
  });
  if (new Set(games.map(game => game.game_id)).size !== games.length) invalid();
  return games;
}

export async function listDecisions(gameId: number, signal?: AbortSignal): Promise<DecisionSummary[]> {
  integer(gameId, 1);
  const data = envelope(await request(`/api/inning-decisions?game_id=${gameId}`, signal), ['decisions']);
  if (!Array.isArray(data.decisions)) invalid();
  const decisions = data.decisions.map(summary);
  if (decisions.some(item => item.context.linkage.game_pk !== gameId) || new Set(decisions.map(item => item.decision_id)).size !== decisions.length) invalid();
  return decisions;
}

export async function resolveDecision(selected: DecisionSummary, signal?: AbortSignal): Promise<InningResult> {
  summary(selected);
  const data = envelope(await request(`/api/inning-decisions/${selected.decision_id}/resolve`, signal, { revision: selected.revision, context: selected.context }), ['decision_id', 'revision', 'context', 'result']);
  const context = parseContext(data.context);
  if (data.decision_id !== selected.decision_id || data.revision !== selected.revision || !sameContext(context, selected.context)) invalid();
  let result: InningResult;
  try { result = parseInningResult(data.result); } catch { return invalid(); }
  if (!sameContext(context, { phase: 'before_pitching_change', linkage: result.linkage, initial_state: result.provenance.evaluation_identity.initial_state_and_count })) invalid();
  return result;
}
