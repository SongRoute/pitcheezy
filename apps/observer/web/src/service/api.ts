/** Service API adapter. The ONLY place that knows backend response shapes; screens use model.ts types.
 *  Primary: the service contract (pm-jobs/20261005-ui-live-redesign/CONTRACT.md, requested as FRONTEND-REQUESTS R1/R2).
 *  Fallback (404 on the service route only): the existing `/api/live/games` + `/api/watch/{game}` feeds, so the screen
 *  still renders real data before the contract lands. The fallback never invents scores or recommendations. */
import type { Correction, Display, DisplayStateName, Receipt, LineTeam, Linescore, PlayerInfo, Actual, Bounds, Candidate, Current, Detail, GameCard, GameStatus, Half, Pa, PitchRow, Rec, Situation } from './model';
import { DISPLAY_STATES, groupPas, kstDate, markDoubleheaders, pitchName, plainZone, shiftDate } from './model';
import { learnViewer, normalizeViewer, viewerOn, withDelay } from './viewer';
import { normalizeExpectation, normalizeReference } from './extras';
import { PUBLISH_CONFIG, manifestPollMs, isCalendarDate, observeHostTime, observePublishTime, publishUrl, publishedPath, unwrapPublished } from './publish';
import type { PublishStamp, PublishHealth, PublishedFile } from './publish';

export class FetchError extends Error { constructor(message: string, public status: number) { super(message); } }
export const TIMEOUT_MS = 15000;

/** Host access/notice codes use local copy: never display an arbitrary host message. */
function hostErrorMessage(status: number, body: unknown): string | null {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return null;
  const error = body as Record<string, unknown>;
  if (status === 401 && error.error === 'access_required') return '접근 확인이 필요해요. 페이지를 새로고침해 다시 접속해 주세요.';
  if (status !== 503 || error.error !== 'service_notice') return null;
  const notice = error.notice;
  if (!notice || typeof notice !== 'object' || Array.isArray(notice)) return null;
  const id = (notice as Record<string, unknown>).id;
  if (id === 'maintenance') return '서비스 점검 중이에요. 잠시 후 다시 확인해 주세요.';
  if (id === 'data_hold') return '자료를 확인하고 있어요. 확인이 끝나면 다시 보여 드려요.';
  return typeof id === 'string' ? '서비스 안내로 자료 제공이 잠시 중단됐어요. 잠시 후 다시 확인해 주세요.' : null;
}

/** Demo binding travels on the URL (`?demo=<session>`) and is forwarded to the API as-is; the screen never shows it. */
export const demoParam = () => __PZ_STATIC__ ? null : new URLSearchParams(window.location.search).get('demo');  // the public bundle carries no demo binding
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
  if (!response.ok) throw new FetchError(hostErrorMessage(response.status, body) ?? (typeof body?.detail === 'string' ? body.detail : '자료를 불러오지 못했어요.'), response.status);
  if (body === null) throw new FetchError('서버 응답을 읽지 못했어요.', response.status);
  return body as T;
}

// ---------- static (public) mode: published files only; never `/api/**`, never `?demo=`, no `?after=` cursor ----------
/** One published file. The timeout and the caller's abort cover the WHOLE answer, body included: a body that stalls after the headers
 *  ends like an unreachable server instead of leaving the request (and every later poll) waiting for ever.
 *  A missing file keeps its HTTP status (404); an answer that is not JSON (half-written file, a host's HTML fallback page) is treated
 *  like an unreachable server (status 0) so the screen keeps its last state and asks again. */
async function getPublished<T>(file: PublishedFile, signal?: AbortSignal): Promise<{ body: T; stamp: Omit<PublishStamp, 'seenAt'> & { seq: number }; etag: string | null } | null> {
  const path = publishedPath(file);
  if (path === null) throw new FetchError('자료를 불러오지 못했어요.', 404);   // not a real date / game number: nothing is asked
  const url = publishUrl(path), last = applied.get(url);
  const timeout = new AbortController();
  const timer = window.setTimeout(() => timeout.abort(), PUBLISH_CONFIG.timeoutMs);
  const onAbort = () => timeout.abort();
  signal?.addEventListener('abort', onAbort);
  if (signal?.aborted) timeout.abort();
  try {
    let response: Response;
    try { response = await fetch(url, { cache: 'no-cache', headers: last?.etag ? { 'If-None-Match': last.etag } : undefined, signal: timeout.signal }); }
    catch (e) {
      if (signal?.aborted) throw e;
      throw new FetchError('서버에 연결하지 못했어요. 네트워크 연결을 확인해 주세요.', 0);
    }
    // the hosting server's own clock for this answer (Date, plus Age when a cache says how old its copy is); unreadable cross-origin unless exposed
    const date = Date.parse(response.headers.get('date') ?? ''), age = Number(response.headers.get('age') ?? '');
    if (Number.isFinite(date)) observeHostTime(date + (Number.isFinite(age) && age > 0 ? age * 1000 : 0));
    if (response.status === 304) {
      if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
      if (!last?.etag) throw new FetchError('서버 응답을 읽지 못했어요.', 0); // no validated body to reuse
      if (last.stamp.publishedAt) observePublishTime(Date.parse(last.stamp.publishedAt));
      return null; // reuse the already validated/normalized body; neither JSON parse nor shape runs again
    }
    if (!response.ok) {
      // Only the two host protocols need a body; other errors retain the bounded no-body path.
      const body = response.status === 401 || response.status === 503 ? await response.json().catch(() => null) : null;
      if (body === null) void response.body?.cancel().catch(() => undefined);
      if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
      throw new FetchError(hostErrorMessage(response.status, body) ?? '자료를 불러오지 못했어요.', response.status);
    }
    const raw = await response.json().catch(() => null);
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
    if (raw === null || typeof raw !== 'object') throw new FetchError('서버 응답을 읽지 못했어요.', 0);
    const p = unwrapPublished<T>(raw);
    if (p.seq === null) throw new FetchError('서버 응답을 읽지 못했어요.', 0);   // no integer sequence: cannot be ordered, so never applied
    if (p.publishedAt) observePublishTime(Date.parse(p.publishedAt));   // used as server time only when no hosting time is readable
    return { body: p.body, etag: response.headers.get('etag'), stamp: { seq: p.seq, publishedAt: p.publishedAt, final: p.final } };
  } finally { window.clearTimeout(timer); signal?.removeEventListener('abort', onAbort); }
}
/** Last applied publish per file. Contract: "화면은 순번이 뒤로 가는 파일을 버린다" — an answer whose sequence is LOWER than the one already
 *  applied (a cache serving an older copy, or an older answer arriving after a newer one) is discarded and the applied value stays.
 *  A sequence is the publish cycle that last wrote that file, so it is compared per file only, never across files. */
const applied = new Map<string, { stamp: PublishStamp & { seq: number }; value: unknown; etag: string | null; readAt: number }>();
async function readPublished<T, R>(file: PublishedFile, signal: AbortSignal | undefined, shape: (body: T) => R): Promise<{ value: R; stamp: PublishStamp }> {
  const got = await getPublished<T>(file, signal);
  const key = publishUrl(publishedPath(file)!), last = applied.get(key);
  if (got === null) { last!.readAt = performance.now(); return { value: last!.value as R, stamp: last!.stamp }; }
  if (last && got.stamp.seq < last.stamp.seq) return { value: last.value as R, stamp: last.stamp };
  let value: R;
  // a body that does not have the expected shape is an unreadable answer; the raw exception never reaches the screen
  try { value = shape(got.body); } catch (e) { throw e instanceof FetchError ? e : new FetchError('서버 응답을 읽지 못했어요.', 0); }
  // the same sequence seen again keeps the moment it was first seen
  const stamp = { ...got.stamp, seenAt: last && last.stamp.seq === got.stamp.seq ? last.stamp.seenAt : performance.now() };
  applied.set(key, { stamp, value, etag: got.etag, readAt: performance.now() });
  return { value, stamp };
}
/** Health additions are optional: today's publisher still supplies only the stamp and failure lists. */
export async function fetchPublishHealth(signal?: AbortSignal): Promise<PublishHealth> {
  const got = await readPublished<Raw, Pick<PublishHealth, 'gameChecks' | 'failingGames'>>({ kind: 'health' }, signal, b => ({
    gameChecks: b.game_checks && typeof b.game_checks === 'object' && !Array.isArray(b.game_checks) ? b.game_checks : null,
    failingGames: Array.isArray(b.feed_failing_games) ? b.feed_failing_games : [],
  }));
  return { ...got.stamp, ...got.value };
}
/** `v1/game/<pk>.json`: always the whole game (no cursor, no merge). */
export async function fetchPublishedDetail(game: number, signal?: AbortSignal): Promise<{ detail: Detail; stamp: PublishStamp }> {
  const got = await readPublished<Raw, Detail>({ kind: 'game', gamePk: game }, signal, body => {
    const detail = normalizeDetail(body);
    // fail closed: a file that holds another game is never drawn under this game's address
    if (detail.game.game_pk !== game) throw new FetchError('자료를 불러오지 못했어요.', 404);
    return detail;
  });
  return { detail: got.value, stamp: got.stamp };
}
/** `v1/games/<date>.json`. No file for a date = nothing published for it (a date outside the archive and the followed window). */
export async function fetchPublishedGames(date: string, signal?: AbortSignal): Promise<GameList> {
  try {
    return (await readPublished<{ games: unknown; live_ok?: boolean }, GameList>({ kind: 'games', date }, signal, body => {
      if (!Array.isArray(body.games)) throw new FetchError('서버 응답을 읽지 못했어요.', 0);
      // only rows that are objects with a positive integer game number, each game once (a link is built from that number)
      const seen = new Set<number>();
      const rows = (body.games as unknown[]).filter((r): r is Raw => r !== null && typeof r === 'object' && Number.isSafeInteger((r as Raw).game_pk)
        && (r as Raw).game_pk > 0 && !seen.has((r as Raw).game_pk) && seen.add((r as Raw).game_pk) !== undefined);
      return Object.assign(markDoubleheaders(rows.map(normalizeCard).filter(g => g.kst_date === date))
        .sort((a, b) => (a.start ?? '').localeCompare(b.start ?? '') || a.game_pk - b.game_pk), { liveOk: body.live_ok !== false });
    })).value;
  } catch (e) {
    if (e instanceof FetchError && e.status === 404) return Object.assign([], { liveOk: true });
    throw e;
  }
}
/** `v1/dates.json` lists archived-game dates only (as the server route does), so the dates the publisher is following now
 *  (`live_dates` of `v1/manifest.json`: today and its neighbours) are added. Either file may be missing; both missing = an error. */
export async function fetchPublishedManifest(signal?: AbortSignal) {
  return (await readPublished<Raw, { liveDates: string[]; pollMs: number | null; fixedDelayS: number | null }>({ kind: 'manifest' }, signal, b => ({
    liveDates: (Array.isArray(b.live_dates) ? b.live_dates : []).filter(isCalendarDate), pollMs: manifestPollMs(b.publish_interval_s),
    fixedDelayS: typeof b.fixed_delay_s === 'number' && Number.isFinite(b.fixed_delay_s) && b.fixed_delay_s >= 0 ? b.fixed_delay_s : null,
  }))).value;
}
export async function fetchPublishedDates(signal?: AbortSignal): Promise<{ dates: string[]; liveDates: string[]; pollMs: number | null; fixedDelayS: number | null }> {
  const [archived, manifest] = await Promise.allSettled([
    readPublished<Raw, string[]>({ kind: 'dates' }, signal, b => (Array.isArray(b.dates) ? b.dates : []).filter(isCalendarDate)),
    fetchPublishedManifest(signal)]);
  if (archived.status === 'rejected' && manifest.status === 'rejected') throw archived.reason;
  const liveDates = manifest.status === 'fulfilled' ? manifest.value.liveDates : [];
  const dates = [...new Set([...(archived.status === 'fulfilled' ? archived.value.value : []), ...liveDates])].sort().reverse();
  return { dates, liveDates, pollMs: manifest.status === 'fulfilled' ? manifest.value.pollMs : null, fixedDelayS: manifest.status === 'fulfilled' ? manifest.value.fixedDelayS : null };
}
/** Is this game on the list of any date the publisher is following now? true / false, or null when that could not be read.
 *  A game with no file that no followed list names (a demo runs on it, a wrong number, long over) is not waited for. */
export async function isGamePublishedSoon(game: number, signal?: AbortSignal): Promise<boolean | null> {
  let liveDates: string[];
  try {
    // Missing-game retries reuse the manifest read by the screen, rather than re-reading dates/manifest on every backoff tick.
    const cached = applied.get(publishUrl('manifest.json'));
    const manifest = cached?.value as Awaited<ReturnType<typeof fetchPublishedManifest>> | undefined;
    liveDates = manifest && performance.now() - cached!.readAt < Math.max(60000, manifest.pollMs ?? PUBLISH_CONFIG.livePollMs)
      ? manifest.liveDates : (await fetchPublishedManifest(signal)).liveDates;
  } catch { return null; }
  let unknown = false;
  for (const date of liveDates) {
    try { if ((await fetchPublishedGames(date, signal)).some(g => g.game_pk === game)) return true; } catch { unknown = true; }
  }
  return unknown ? null : false;
}
/** KST dates that have games: `GET /api/service/dates` (API mode). */
export async function fetchDates(signal?: AbortSignal): Promise<string[]> {
  if (__PZ_STATIC__) return (await fetchPublishedDates(signal)).dates;
  return (await getJson<{ dates: string[] }>('/service/dates', signal)).dates;
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
  return { game_type: raw.game_type ?? null, series_description: raw.series_description ?? null, series_game_number: num(raw.series_game_number), games_in_series: num(raw.games_in_series), probable_pitchers: raw.probable_pitchers ?? null, venue: raw.venue ?? null, away_abbr: raw.away_abbr ?? null, home_abbr: raw.home_abbr ?? null, game_pk: raw.game_pk, start: typeof raw.start === 'string' ? raw.start : null, kst_date: raw.date_kst ?? (raw.start ? kstDate(raw.start) : raw.date),
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
    setup_x: num(raw.catcher_setup?.plate_x_feet), tracking: raw.tracking ? { extension_feet: num(raw.tracking.extension_feet), spin_rate_rpm: num(raw.tracking.spin_rate_rpm), exit_velocity_mph: num(raw.tracking.exit_velocity_mph), launch_angle_deg: num(raw.tracking.launch_angle_deg), distance_feet: num(raw.tracking.distance_feet) } : null };
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
  if (__PZ_STATIC__) return fetchPublishedGames(date, signal);
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
  if (__PZ_STATIC__) return (await fetchPublishedDetail(game, signal)).detail;
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
  // Reindexing can collide with another known pitch's cursor. Reload the full history; never erase that neighbour.
  if (inc.pitches.some(n => old.pitches.some(p => (p.key === n.key) !== (p.index === n.index)))) return null;
  const keep = old.pitches.filter(p => !inc.pitches.some(n => n.key === p.key));
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
  return { win_expectancy: raw.win_expectancy?.status === 'available' && raw.win_expectancy.perspective === 'home' && num(raw.win_expectancy.home_probability) !== null && raw.win_expectancy.home_probability >= 0 && raw.win_expectancy.home_probability <= 1 ? { status: 'available', home_probability: raw.win_expectancy.home_probability } : null, session_status: raw.source?.session_status ?? null, current_context: raw.current?.context ?? raw.current_context ?? null, game, pitches, pas: Array.isArray(raw.pas) ? raw.pas.map((p: Raw) => pa(p, unsurePas.has(p.pa_key))) : groupPas(pitches, current).map(p => unsurePas.has(p.pa_key) ? { ...p, result: BLANK_LABEL } : p), current,
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
