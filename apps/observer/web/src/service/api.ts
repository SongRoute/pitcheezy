/** Service API adapter. The ONLY place that knows backend response shapes; screens use model.ts types.
 *  Primary: the service contract (pm-jobs/20261005-ui-live-redesign/CONTRACT.md, requested as FRONTEND-REQUESTS R1/R2).
 *  Fallback (404 on the service route only): the existing `/api/live/games` + `/api/watch/{game}` feeds, so the screen
 *  still renders real data before the contract lands. The fallback never invents scores or recommendations. */
import type { Correction, Display, DisplayStateName, Receipt, LineTeam, Linescore, PlayerInfo, Actual, Bounds, Candidate, Current, Detail, GameCard, GameStatus, Half, Pa, PitchRow, Rec, Situation } from './model';
import { DISPLAY_STATES, groupPas, kstDate, markDoubleheaders, pitchName, plainZone, shiftDate } from './model';
import { learnViewer, normalizeViewer, viewerOn, withDelay } from './viewer';
import { normalizeExpectation, normalizeReference } from './extras';

export class FetchError extends Error { constructor(message: string, public status: number) { super(message); } }
export const TIMEOUT_MS = 15000;

/** Demo binding travels on the URL (`?demo=<session>`) and is forwarded to the API as-is; the screen never shows it. */
export const demoParam = () => new URLSearchParams(window.location.search).get('demo');
function withDemo(path: string) {
  const demo = demoParam();
  if (!demo) return path;
  return `${path}${path.includes('?') ? '&' : '?'}demo=${encodeURIComponent(demo)}`;
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const timeout = new AbortController();
  const timer = window.setTimeout(() => timeout.abort(), TIMEOUT_MS);
  const onAbort = () => timeout.abort();
  signal?.addEventListener('abort', onAbort);
  let response: Response;
  try { response = await fetch(`/api${withDemo(path)}`, { cache: 'no-store', signal: timeout.signal }); }
  catch (e) {
    if (signal?.aborted) throw e;
    throw new FetchError('서버에 연결하지 못했어요. 네트워크 연결을 확인해 주세요.', 0);
  } finally { window.clearTimeout(timer); signal?.removeEventListener('abort', onAbort); }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new FetchError(typeof body?.detail === 'string' ? body.detail : '자료를 불러오지 못했어요.', response.status);
  if (body === null) throw new FetchError('서버 응답을 읽지 못했어요.', response.status);
  return body as T;
}

// ---------- shared normalizers ----------
type Raw = Record<string, any>;
const num = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null;
function status(raw: Raw): { status: GameStatus; text: string } {
  const text = String(raw.status_text ?? raw.status ?? '');
  const s = String(raw.status ?? '').toLowerCase();
  if (raw.embargoed === true) return { status: 'other', text: '' };
  // The server's own decision (the `live` flag / a status kind) comes first: a mid-game 'Delayed', 'Suspended', 'Manager challenge' or
  // 'Review' stays live, and a 'Delayed Start' before the first pitch is not live. The status words are read only for rows that carry
  // neither (legacy schedule feed), and there 'Delayed Start' is a game that has not started.
  if (raw.live === true || s === 'live') return { status: 'live', text: text || '진행 중' };
  const decided = typeof raw.live === 'boolean' || ['final', 'scheduled', 'postponed', 'other'].includes(s);
  if (s === 'postponed' || /postponed|suspended|cancel/i.test(text)) return { status: 'postponed', text };
  if (s === 'final' || /final|game over|completed/i.test(text)) return { status: 'final', text };
  if (!decided && /in progress|manager challenge|review|delayed(?! start)/i.test(text)) return { status: 'live', text: text || '진행 중' };
  if (s === 'scheduled' || /scheduled|pre-game|warmup|delayed start/i.test(text)) return { status: 'scheduled', text };
  return { status: 'other', text };
}
export function normalizeCard(raw: Raw): GameCard {
  const st = status(raw);
  return { venue: raw.venue ?? null, away_abbr: raw.away_abbr ?? null, home_abbr: raw.home_abbr ?? null, game_pk: raw.game_pk, start: raw.start ?? null, kst_date: raw.date_kst ?? (raw.start ? kstDate(raw.start) : raw.date),
    status: st.status, status_text: st.text, away_team: raw.away_team, home_team: raw.home_team,
    away_score: num(raw.away_score), home_score: num(raw.home_score), inning: num(raw.inning), half: raw.half ?? null,
    game_number: num(raw.game_number) ?? 1, double_header: raw.double_header === true, embargoed: raw.embargoed === true };
}
function candidate(raw: Raw): Candidate {
  return { rank: raw.rank, pitch_type: raw.pitch_type, pitch_label: pitchName(raw.pitch_label, raw.pitch_type), zone_id: raw.zone_id ?? null, zone_label: plainZone(raw.zone_label ?? ''),
    target: raw.target && num(raw.target.x) !== null && num(raw.target.z) !== null ? { x: raw.target.x, z: raw.target.z } : null,
    probability: num(raw.probability ?? raw.detail?.probability) };
}
/** Archived pre-pitch recommendation. Only candidates the source actually carries (≤3 shown); nothing is padded. */
export function normalizeRec(raw: Raw | null | undefined, fallbackProvenance: Rec['provenance'] = 'absent'): Rec {
  if (!raw) return { status: 'missing', reason: null, candidates: [], provenance: 'absent', policy: null };
  const cands: Raw[] = raw.candidates ?? raw.recommendation?.candidates ?? [];
  const st = raw.status === 'ready' && cands.length ? 'ready' : raw.status === 'unsupported' || raw.status === 'unavailable' ? 'unsupported' : 'missing';
  const provenance = raw.provenance === 'captured_live' || raw.provenance === 'asof_replay' ? raw.provenance : st === 'missing' ? 'absent' : fallbackProvenance;
  return { status: st, reason: raw.reason ?? null, provenance, policy: raw.policy?.name ?? null, recorded_at: st === 'ready' && typeof raw.recorded_at === 'string' ? raw.recorded_at : null,
    candidates: st === 'ready' ? cands.map(candidate).sort((a, b) => a.rank - b.rank).slice(0, 3) : [] };
}
/** Receipt blanking: the server sends every field of an unconfirmed pitch as null except the '확인 중' labels. */
const BLANK_LABEL = '확인 중';
const isBlank = (raw: Raw) => raw.pitch_type == null && raw.pitch_label == null && raw.speed_mph == null && raw.x == null && raw.z == null
  && raw.description == null && raw.event == null;
const blankActual = (): Actual => ({ pitch_type: null, pitch_label: '', description: null, result_label: BLANK_LABEL, event: null, event_label: null,
  play_text: null, speed_mph: null, x: null, z: null, zone_label: '', setup_x: null });
function actual(raw: Raw | null | undefined): Actual | null {
  if (!raw) return null;
  if (isBlank(raw) && raw.result_label === BLANK_LABEL) return blankActual();  // only the server's own blanking; an archive row with no data keeps '구종 미상'
  return { pitch_type: raw.pitch_type ?? null, pitch_label: pitchName(raw.pitch_label, raw.pitch_type), description: raw.description ?? null,
    result_label: raw.result_label ?? '', event: raw.event ?? null, event_label: raw.event_label ?? null, play_text: raw.play_text ?? null,
    speed_mph: num(raw.speed_mph), x: num(raw.x), z: num(raw.z), zone_label: plainZone(raw.zone_label ?? ''),  // same plain words as a recommended zone
    setup_x: num(raw.catcher_setup?.plate_x_feet) };
}
/** Fail closed: any receipt status other than 'received' is unconfirmed. No `receipt` key (store off, archive) → undefined, except a
 *  row the server already blanked to '확인 중', which stays unconfirmed so it is never drawn as an observed pitch. */
function receipt(raw: Raw): Receipt | undefined {
  const r = raw.receipt;
  if (r && typeof r === 'object') return { status: r.status === 'received' ? 'received' : 'unconfirmed', corrected: r.corrected === true };
  return raw.actual && isBlank(raw.actual) && raw.actual.result_label === BLANK_LABEL ? { status: 'unconfirmed', corrected: false } : undefined;
}
function pitchRow(raw: Raw, provenance: Rec['provenance']): PitchRow {
  const before = raw.situation_before ?? raw.situation;
  const rc = receipt(raw);
  return { index: raw.index, key: raw.key ?? `${raw.pa_key ?? raw.pa_id}:${raw.pitch_number}`, pa_key: raw.pa_key ?? raw.pa_id,
    at_bat_number: raw.at_bat_number, pitch_number: raw.pitch_number,
    situation: { ...before, inning: raw.inning ?? before.inning, half: raw.half ?? before.half } as Situation,
    // an unconfirmed pitch is never paired with a recommendation, whatever the row carries (fail closed)
    batter: raw.batter, pitcher: raw.pitcher, rec: rc?.status === 'unconfirmed' ? normalizeRec(null) : normalizeRec(raw.rec ?? raw.pre, provenance),
    actual: rc?.status === 'unconfirmed' ? blankActual() : actual(raw.actual), receipt: rc,
    context: raw.context_before ?? null, line: linescoreRev3(raw.linescore_before), expectation: rc?.status === 'unconfirmed' ? null : normalizeExpectation(raw) };
}
/** Local copy used only when the server sends no `label` / `message` for a state (the server's text always wins). */
const STATE_COPY: Record<DisplayStateName, [string, string | null]> = {
  recommended: ['다음 공 추천', null],
  unconfirmed: ['확인 중', '경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.'],
  unsupported: ['추천 없음', '이 상황은 추천 대상이 아니에요.'],
  feed_delayed: ['자료 늦음', '경기 자료가 평소보다 늦게 도착했어요. 추천이 실제 경기보다 늦을 수 있어요.'],
  disconnected: ['연결 끊김', '경기 자료 연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.'],
  ledger_unavailable: ['추천 보류', '추천 기록을 저장하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.'],
  waiting: ['대기 중', '다음 공을 기다리고 있어요.'],
  unavailable: ['추천 없음', '지금은 추천을 제공하지 못해요.'],
  pregame: ['경기 전', '경기 시작 전이에요.'],
  final: ['경기 종료', '경기가 끝났어요.'] };
const SHOWING: DisplayStateName[] = ['recommended', 'feed_delayed', 'disconnected'];
/** Fail closed: an answer of the LIVE route without a usable `display_state` (missing, or a state name this build does not know) never
 *  shows a recommendation. The slot still says one calm thing: what the card and phase allow (경기 전 / 경기 종료 / 공수 교대 / 결과 확인 중),
 *  otherwise the default 확인 중 sentence. The pending pitch's key is kept so the D1 guard can say that a list went away. */
function fallbackDisplay(raw: Raw, card: GameCard): Display {
  const phase = raw.current?.phase;
  // 경기 전 only when nothing has been thrown and nothing is pending (the cutoff also covers an `?after=` answer, which lists no old pitch)
  const nothingYet = !(raw.pitches?.length) && (num(raw.cutoff?.index) ?? -1) < 0 && !raw.next;
  const [state, label, message]: [DisplayStateName, string, string | null] = card.status === 'final' ? ['final', ...STATE_COPY.final]
    : card.status !== 'live' && nothingYet ? ['pregame', ...STATE_COPY.pregame]
    : phase === 'between_innings' ? ['waiting', '공수 교대', '공수 교대 중이에요.'] : phase === 'resolving' ? ['waiting', '결과 확인 중', '방금 공의 결과를 기다리고 있어요.']
    : ['unconfirmed', ...STATE_COPY.unconfirmed];
  return { state, reason: phase === 'between_innings' && state === 'waiting' ? 'between_innings' : 'client_fail_closed', label, message,
    pitch_key: text(raw.display_state?.pitch_key) ?? text(raw.next?.pitch_key), shown: false, correction: null,
    feed: { ok: raw.feed?.ok !== false, delayed: false, lag_s: null } };
}
const CORRECTION_COPY: Record<Correction['kind'], string> = { replaced: '상황이나 기록이 바뀌어 추천이 달라졌어요.', withdrawn: '조금 전에 보여 드린 추천은 지금 표시하지 않아요.' };
const text = (v: unknown): string | null => typeof v === 'string' && v.trim() ? v.trim() : null;
/** `display_state` → Display. Anything that is not a known state (older backend, archive, demo, a future state name) → null, and the
 *  screen keeps its pre-contract behaviour. `reason` is kept for tests/diagnostics only; `detail` is dropped. */
export function normalizeDisplay(raw: Raw | null | undefined): Display | null {
  if (!raw || typeof raw !== 'object' || !(DISPLAY_STATES as readonly string[]).includes(raw.state)) return null;
  const state = raw.state as DisplayStateName;
  const c = raw.correction;
  const kind: Correction['kind'] | null = c && (c.kind === 'replaced' || c.kind === 'withdrawn') ? c.kind : null;
  const prev = c?.previous && typeof c.previous === 'object' ? c.previous as Raw : null;
  return { state, reason: text(raw.reason), label: text(raw.label) ?? STATE_COPY[state][0], message: text(raw.message) ?? STATE_COPY[state][1],
    // candidates may be on screen only when the server says so AND the state is one that carries a recommendation
    pitch_key: text(raw.pitch_key), shown: raw.recommendation_shown === true && SHOWING.includes(state),
    correction: kind ? { kind, message: text(c.message) ?? CORRECTION_COPY[kind],
      previous: prev && (text(prev.pitch_type) || text(prev.pitch_label)) ? { pitch_type: text(prev.pitch_type), pitch_label: text(prev.pitch_label), zone_id: text(prev.zone_id), zone_label: text(prev.zone_label) } : null } : null,
    feed: { ok: raw.feed?.ok !== false, delayed: raw.feed?.delayed === true, lag_s: num(raw.feed?.lag_s) } };
}
function pa(raw: Raw, unsure = false): Pa {
  return { pa_key: raw.pa_key, at_bat_number: raw.at_bat_number, inning: raw.inning, half: raw.half,
    batter: raw.batter?.name ?? '타자 미상', pitcher: raw.pitcher?.name ?? '투수 미상', first: raw.first_index, last: raw.last_index,
    batting_order: num(raw.batting_order), pitch_count: num(raw.pitch_count), complete: raw.complete === true, batter_id: typeof raw.batter?.id === 'number' ? raw.batter.id : null,
    // a plate appearance that holds an unconfirmed pitch: the server sends result_text '확인 중' (never its outcome)
    result: raw.result_text === BLANK_LABEL || unsure ? BLANK_LABEL : raw.complete === true ? raw.result_text ?? raw.event_label ?? '결과 기록 없음' : '타석 진행 중' };
}
/** main.py catch-all answer for an /api path the running backend does not have (older deployment). */
const ROUTE_MISSING = 'API 경로를 찾을 수 없습니다.';
const ZONE: Bounds = { bottom: 1.6, top: 3.3899 };
const PHASES: unknown[] = ['pregame', 'pre_pitch', 'resolving', 'between_innings', 'final'];

// ---------- list ----------
export type GameList = GameCard[] & { liveOk?: boolean };
export async function fetchGames(date: string, signal?: AbortSignal): Promise<GameList> {
  try {
    let body = await getJson<{ games: Raw[]; live_ok?: boolean }>(withDelay(`/service/games?date=${date}`), signal);
    if (!demoParam() && learnViewer(body)) body = await getJson(withDelay(`/service/games?date=${date}`), signal);  // viewer delay switched on: ask again at this device's delay
    return Object.assign(markDoubleheaders(body.games.map(normalizeCard).filter(g => g.kst_date === date))
      .sort((a, b) => (a.start ?? '').localeCompare(b.start ?? '') || a.game_pk - b.game_pk), { liveOk: body.live_ok !== false });
  } catch (e) { if (!(e instanceof FetchError) || e.status !== 404 || e.message !== ROUTE_MISSING || demoParam()) throw e; }
  // fallback: MLB schedule by US date; keep only games whose ORIGINAL start falls on the KST date
  const days = await Promise.all([shiftDate(date, -1), date].map(d => getJson<{ games: Raw[] }>(`/live/games?date=${d}`, signal)));
  const seen = new Set<number>();
  const games = days.flatMap(d => d.games).filter(g => !seen.has(g.game_pk) && seen.add(g.game_pk)).map(normalizeCard)
    .filter(g => g.kst_date === date);
  return markDoubleheaders(games).sort((a, b) => (a.start ?? '').localeCompare(b.start ?? '') || a.game_pk - b.game_pk);
}

// ---------- detail ----------
let serviceMissing = false;  // backend without /api/service (404 on the route itself): use the legacy feeds; never in demo
/** Full history from the backend. `after` asks only for newer pitches (contract §3); the caller merges with mergeDetail. */
export async function fetchDetail(game: number, signal?: AbortSignal, after?: number, delay: number | null = null): Promise<Detail> {
  if (!serviceMissing || demoParam()) {
    try {
      const raw = await getJson<Raw>(withDelay(`/service/games/${game}${after === undefined ? '' : `?after=${after}`}`, demoParam() ? null : delay), signal);
      return normalizeDetail(raw);
    } catch (e) {
      if (!(e instanceof FetchError) || e.status !== 404 || demoParam()) throw e;
      if (e.message !== ROUTE_MISSING) throw e;  // any other 404 = unknown game, shown as such
      serviceMissing = true;
    }
  }
  return fallbackDetail(game, signal);
}
/** Merge an `?after=` answer: keep older pitches unless the cutoff moved back (demo seek/reset), then refetch fully. */
export function mergeDetail(old: Detail, inc: Detail): Detail | null {
  if (inc.cutoff !== null && old.cutoff !== null && inc.cutoff < old.cutoff) return null;
  if (inc.session !== old.session) return null;  // contract rev2 client cache rule: a new demo session never merges with the old one
  if ((inc.viewer?.delay_s ?? null) !== (old.viewer?.delay_s ?? null) || inc.viewer?.buffering || old.viewer?.buffering) return null;  // another viewer delay = another moment: never merged
  // a receipt changed somewhere (a pitch became unconfirmed, corrected, or plain received again): take the full history again
  if ((inc.receipts_version ?? null) !== (old.receipts_version ?? null)) return null;
  const keep = old.pitches.filter(p => !inc.pitches.some(n => n.index === p.index));
  return { ...inc, pitches: [...keep, ...inc.pitches].sort((a, b) => a.index - b.index) };
}
/** CONTRACT-rev3 §3 `linescore` → screen shape. Cells stay null when the backend says not started / not reconstructible. */
function linescoreRev3(raw: Raw | null | undefined): Linescore | null {
  if (!raw || !Array.isArray(raw.innings) || !raw.teams) return null;
  const rows: Raw[] = raw.innings.filter((r: unknown) => r && typeof r === 'object');
  const team = (side: 'away' | 'home'): LineTeam => ({ innings: rows.map(r => num(r[side])),
    r: num(raw.teams[side]?.runs), h: num(raw.teams[side]?.hits), e: num(raw.teams[side]?.errors), bb_hbp: num(raw.teams[side]?.bb_hbp) });
  return { innings: rows.length, away: team('away'), home: team('home') };
}
/** Official stat strings ('.282', '1.014', '2.87') → numbers; anything else (missing, '-.--', '∞') → null = '–'. */
const statNum = (v: unknown): number | null => { const n = typeof v === 'string' && /^-?\d*\.?\d+$/.test(v.trim()) ? Number(v) : num(v); return n === null || Number.isNaN(n) ? null : n; };
/** CONTRACT-rev3 §4 flat `pregame.players[id]` (only revealed roles; a two-way player can carry era AND avg/ops) + game `jersey`
 *  from every person card in the response. The card's slot (pitcher/batter) decides which fields are shown. */
function playersRev3(raw: Raw): Record<string, PlayerInfo> | null {
  const out: Record<string, PlayerInfo> = {};
  const seen = (p: Raw | null | undefined) => { if (p && p.id != null) { const k = String(p.id); out[k] = { ...out[k], jersey: p.jersey ?? out[k]?.jersey ?? null }; } };
  for (const h of [raw.current, ...(raw.pitches ?? []), ...(raw.pas ?? [])]) { seen(h?.batter); seen(h?.pitcher); }
  for (const [id, st] of Object.entries((raw.pregame?.players ?? {}) as Record<string, Raw>))
    out[id] = { ...out[id], era: statNum(st.era), avg: statNum(st.avg), ops: statNum(st.ops), stats_as_of: st.as_of ?? raw.pregame?.as_of ?? null };
  return Object.keys(out).length ? out : null;
}
export function normalizeDetail(raw: Raw): Detail {
  const provenance: Rec['provenance'] = raw.source?.kind === 'live' ? 'captured_live' : 'asof_replay';
  const pitches = (raw.pitches ?? []).map((p: Raw) => pitchRow(p, provenance));
  const cur = raw.current;
  const game = normalizeCard(raw.game);
  const display = provenance === 'captured_live' ? normalizeDisplay(raw.display_state) ?? fallbackDisplay(raw, game) : null;
  // a plate appearance that holds an unconfirmed pitch reads 확인 중 here too, even when the server did not blank it
  const unsurePas = new Set(pitches.filter((p: PitchRow) => p.receipt?.status === 'unconfirmed').map((p: PitchRow) => p.pa_key));
  // The pending slot's key ('game:at_bat:pitch') names the open plate appearance when `current.pa_key` is not sent.
  const slot = /^(\d+:\d+):\d+$/.exec(display?.pitch_key ?? '');
  const current: Current | null = cur?.situation ? { phase: cur.phase ?? 'pre_pitch', situation: cur.situation, batter: cur.batter ?? null,
    pitcher: cur.pitcher ?? null, pa_key: cur.pa_key ?? (cur.phase === 'pre_pitch' && slot ? slot[1] : null) } : null;
  const nextRaw = raw.next?.rec ?? null;
  let next = raw.next ? normalizeRec(nextRaw, provenance === 'captured_live' ? 'captured_live' : 'asof_replay') : null;
  // the server says nothing is shown for the pending pitch: candidates never reach the screen, whatever `next.rec` carries
  if (next && display && !display.shown) next = { ...next, status: next.status === 'ready' ? 'missing' : next.status, candidates: [], provenance: next.status === 'ready' ? 'absent' : next.provenance };
  return { session_status: raw.source?.session_status ?? null, current_context: raw.current?.context ?? raw.current_context ?? null, game, pitches, pas: Array.isArray(raw.pas) ? raw.pas.map((p: Raw) => pa(p, unsurePas.has(p.pa_key))) : groupPas(pitches, current).map(p => unsurePas.has(p.pa_key) ? { ...p, result: BLANK_LABEL } : p), current,
    next, display, receipts_version: text(raw.receipts?.version), phase: PHASES.includes(cur?.phase) ? cur.phase : null, live_source: provenance === 'captured_live',
    cutoff: num(raw.cutoff?.index), session: raw.source?.session_id ?? null, feed: { as_of: raw.feed?.as_of ?? null, ok: raw.feed?.ok !== false, stale_s: num(raw.feed?.stale_s) },
    zone_bounds: raw.zone_bounds ?? ZONE, notes: (raw.notes ?? []).filter((n: unknown) => typeof n === 'string'),
    linescore_asof: linescoreRev3(raw.linescore), players: playersRev3(raw), viewer: normalizeViewer(raw), viewer_on: viewerOn(raw), reference: display?.state === 'unsupported' ? normalizeReference(raw) : null };
}

/** Finished-game fallback: the precomputed watch file holds each pre-pitch record and, per pitch, the actual. */
async function fallbackDetail(game: number, signal?: AbortSignal): Promise<Detail> {
  const timeline = await getJson<Raw>(`/watch/${game}`, signal);
  const decisions: Raw[] = timeline.decisions;
  const reveals: Raw[] = new Array(decisions.length);
  let next = 0;
  await Promise.all(Array.from({ length: 12 }, async () => {
    while (next < decisions.length) { const i = next++; reveals[i] = await getJson<Raw>(`/watch/${game}/reveal/${i}`, signal); }
  }));
  let start: string | null = null, scheduled: Raw | null = null;
  try {
    const sched = await getJson<{ games: Raw[] }>(`/live/games?date=${timeline.game.date}`, signal);
    scheduled = sched.games.find(g => g.game_pk === game) ?? null; start = scheduled?.start ?? null;
  } catch { /* start time stays unknown */ }
  const pitches = decisions.map((d, i) => pitchRow({ ...d, actual: reveals[i]?.actual }, 'asof_replay'));
  const notes = [timeline.policy?.note, timeline.location?.note, timeline.we_note].filter(Boolean);
  return { game: normalizeCard({ ...timeline.game, start, status: scheduled?.status ?? 'final', status_text: scheduled?.status ?? '경기 종료' }),
    pitches, pas: groupPas(pitches, null), current: null, next: null, cutoff: pitches.length - 1, session: null,
    feed: { as_of: null, ok: true, stale_s: null }, zone_bounds: timeline.location?.zone_bounds ?? ZONE, notes };
}

export type { Half };
