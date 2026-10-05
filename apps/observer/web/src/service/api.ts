/** Service API adapter. The ONLY place that knows backend response shapes; screens use model.ts types.
 *  Primary: the service contract (pm-jobs/20261005-ui-live-redesign/CONTRACT.md, requested as FRONTEND-REQUESTS R1/R2).
 *  Fallback (404 on the service route only): the existing `/api/live/games` + `/api/watch/{game}` feeds, so the screen
 *  still renders real data before the contract lands. The fallback never invents scores or recommendations. */
import type { Actual, Bounds, Candidate, Current, Detail, GameCard, GameStatus, Half, Pa, PitchRow, Rec, Situation } from './model';
import { groupPas, kstDate, markDoubleheaders, plainZone, shiftDate } from './model';

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
  if (s === 'postponed' || /postponed|suspended|cancel/i.test(text)) return { status: 'postponed', text };
  if (raw.live === true || s === 'live' || /in progress|manager challenge|delayed|review/i.test(text)) return { status: 'live', text: text || '진행 중' };
  if (s === 'final' || /final|game over|completed/i.test(text)) return { status: 'final', text };
  if (s === 'scheduled' || /scheduled|pre-game|warmup/i.test(text)) return { status: 'scheduled', text };
  return { status: 'other', text };
}
export function normalizeCard(raw: Raw): GameCard {
  const st = status(raw);
  return { game_pk: raw.game_pk, start: raw.start ?? null, kst_date: raw.date_kst ?? (raw.start ? kstDate(raw.start) : raw.date),
    status: st.status, status_text: st.text, away_team: raw.away_team, home_team: raw.home_team,
    away_score: num(raw.away_score), home_score: num(raw.home_score), inning: num(raw.inning), half: raw.half ?? null,
    game_number: num(raw.game_number) ?? 1, double_header: raw.double_header === true, embargoed: raw.embargoed === true };
}
function candidate(raw: Raw): Candidate {
  return { rank: raw.rank, pitch_type: raw.pitch_type, pitch_label: raw.pitch_label, zone_id: raw.zone_id ?? null, zone_label: plainZone(raw.zone_label ?? ''),
    target: raw.target && num(raw.target.x) !== null && num(raw.target.z) !== null ? { x: raw.target.x, z: raw.target.z } : null,
    probability: num(raw.probability ?? raw.detail?.probability) };
}
/** Archived pre-pitch recommendation. Only candidates the source actually carries (≤3 shown); nothing is padded. */
export function normalizeRec(raw: Raw | null | undefined, fallbackProvenance: Rec['provenance'] = 'absent'): Rec {
  if (!raw) return { status: 'missing', reason: null, candidates: [], provenance: 'absent', policy: null };
  const cands: Raw[] = raw.candidates ?? raw.recommendation?.candidates ?? [];
  const st = raw.status === 'ready' && cands.length ? 'ready' : raw.status === 'unsupported' || raw.status === 'unavailable' ? 'unsupported' : 'missing';
  const provenance = raw.provenance === 'captured_live' || raw.provenance === 'asof_replay' ? raw.provenance : st === 'missing' ? 'absent' : fallbackProvenance;
  return { status: st, reason: raw.reason ?? null, provenance, policy: raw.policy?.name ?? null,
    candidates: st === 'ready' ? cands.map(candidate).sort((a, b) => a.rank - b.rank).slice(0, 3) : [] };
}
function actual(raw: Raw | null | undefined): Actual | null {
  if (!raw) return null;
  return { pitch_type: raw.pitch_type ?? null, pitch_label: raw.pitch_label ?? '구종 미상', description: raw.description ?? null,
    result_label: raw.result_label ?? '', event: raw.event ?? null, event_label: raw.event_label ?? null, play_text: raw.play_text ?? null,
    speed_mph: num(raw.speed_mph), x: num(raw.x), z: num(raw.z), zone_label: raw.zone_label ?? '',
    setup_x: num(raw.catcher_setup?.plate_x_feet) };
}
function pitchRow(raw: Raw, provenance: Rec['provenance']): PitchRow {
  const before = raw.situation_before ?? raw.situation;
  return { index: raw.index, key: raw.key ?? `${raw.pa_key ?? raw.pa_id}:${raw.pitch_number}`, pa_key: raw.pa_key ?? raw.pa_id,
    at_bat_number: raw.at_bat_number, pitch_number: raw.pitch_number,
    situation: { ...before, inning: raw.inning ?? before.inning, half: raw.half ?? before.half } as Situation,
    batter: raw.batter, pitcher: raw.pitcher, rec: normalizeRec(raw.rec ?? raw.pre, provenance), actual: actual(raw.actual) };
}
function pa(raw: Raw): Pa {
  return { pa_key: raw.pa_key, at_bat_number: raw.at_bat_number, inning: raw.inning, half: raw.half,
    batter: raw.batter?.name ?? '타자 미상', pitcher: raw.pitcher?.name ?? '투수 미상', first: raw.first_index, last: raw.last_index,
    complete: raw.complete === true,
    result: raw.complete === true ? raw.result_text ?? raw.event_label ?? '결과 기록 없음' : '타석 진행 중' };
}
/** main.py catch-all answer for an /api path the running backend does not have (older deployment). */
const ROUTE_MISSING = 'API 경로를 찾을 수 없습니다.';
const ZONE: Bounds = { bottom: 1.6, top: 3.3899 };

// ---------- list ----------
export type GameList = GameCard[] & { liveOk?: boolean };
export async function fetchGames(date: string, signal?: AbortSignal): Promise<GameList> {
  try {
    const body = await getJson<{ games: Raw[]; live_ok?: boolean }>(`/service/games?date=${date}`, signal);
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
export async function fetchDetail(game: number, signal?: AbortSignal, after?: number): Promise<Detail> {
  if (!serviceMissing || demoParam()) {
    try {
      const raw = await getJson<Raw>(`/service/games/${game}${after === undefined ? '' : `?after=${after}`}`, signal);
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
  const keep = old.pitches.filter(p => !inc.pitches.some(n => n.index === p.index));
  return { ...inc, pitches: [...keep, ...inc.pitches].sort((a, b) => a.index - b.index) };
}
export function normalizeDetail(raw: Raw): Detail {
  const provenance: Rec['provenance'] = raw.source?.kind === 'live' ? 'captured_live' : 'asof_replay';
  const pitches = (raw.pitches ?? []).map((p: Raw) => pitchRow(p, provenance));
  const cur = raw.current;
  const current: Current | null = cur?.situation ? { phase: cur.phase ?? 'pre_pitch', situation: cur.situation, batter: cur.batter ?? null,
    pitcher: cur.pitcher ?? null, pa_key: cur.pa_key ?? null } : null;
  const nextRaw = raw.next?.rec ?? null;
  return { game: normalizeCard(raw.game), pitches, pas: Array.isArray(raw.pas) ? raw.pas.map(pa) : groupPas(pitches, current), current,
    next: raw.next ? normalizeRec(nextRaw, provenance === 'captured_live' ? 'captured_live' : 'asof_replay') : null,
    cutoff: num(raw.cutoff?.index), session: raw.source?.session_id ?? null, feed: { as_of: raw.feed?.as_of ?? null, ok: raw.feed?.ok !== false, stale_s: num(raw.feed?.stale_s) },
    zone_bounds: raw.zone_bounds ?? ZONE, notes: (raw.notes ?? []).filter((n: unknown) => typeof n === 'string') };
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
