/** Public (static) data source: the build reads published JSON files instead of the backend API.
 *  Contract: docs/pm/service/public-publish-contract-draft.md (v0); file shapes: apps/observer/publish_contract/v1 (svc/ws-publisher).
 *
 *    <base>/v1/dates.json  <base>/v1/manifest.json  <base>/v1/games/<YYYY-MM-DD>.json  <base>/v1/game/<game_pk>.json  <base>/v1/health.json
 *
 *  The mode is the build-time constant `__PZ_STATIC__` (vite.config.ts: true when `VITE_PUBLISH_BASE` is non-empty). It is substituted as
 *  a literal, so every `if (__PZ_STATIC__)` / `if (!__PZ_STATIC__)` branch of the other mode is removed from the bundle: the public bundle
 *  holds no `/api`, cursor, demo-forwarding or operator code, and the API bundle behaves as before.
 *  This file holds the config values, the ONE place that builds a published file's URL, the ONE adapter that knows where the publish
 *  sequence / time sit in a file, the server-time estimate and the pure staleness rule. Fetching lives in api.ts. */
const env = import.meta.env;

/** URL or path that holds `v1/`; `.` = next to index.html. Trailing slashes are dropped. Build-time value, never from the page URL. */
export const PUBLISH_BASE: string = String(env.VITE_PUBLISH_BASE ?? '').trim().replace(/\/+$/, '');

const positive = (v: unknown, fallback: number) => { const n = typeof v === 'string' && v.trim() ? Number(v) : NaN; return Number.isFinite(n) && n > 0 ? n : fallback; };
/** Static-mode config (build-time env; `.env.public` carries the defaults). The manifest overrides polling fallbacks in static mode.
 *  PROVISIONAL: `staleAfterS` is NOT a measured value. It only reuses the screen's existing 30 s "connection is late" figure until the
 *  rehearsal measures the real publish interval and jitter; set `VITE_PUBLISH_STALE_S` from that measurement. */
export const PUBLISH_CONFIG = {
  livePollMs: positive(env.VITE_PUBLISH_POLL_MS, 10000),
  pregamePollMs: positive(env.VITE_PUBLISH_PREGAME_POLL_MS, 10000),
  listPollMs: positive(env.VITE_PUBLISH_LIST_POLL_MS, 10000),
  firstLoadRetryMs: positive(env.VITE_PUBLISH_RETRY_MS, 4000),
  timeoutMs: positive(env.VITE_PUBLISH_TIMEOUT_MS, 15000),
  lostAfterMs: positive(env.VITE_PUBLISH_LOST_MS, 30000),
  tickMs: positive(env.VITE_PUBLISH_TICK_MS, 5000),
  /** a screen pinned on a past pitch of a game that is not final still reads at this interval. PROVISIONAL; default = the pregame cadence. */
  pinnedPollMs: positive(env.VITE_PUBLISH_PINNED_POLL_MS, positive(env.VITE_PUBLISH_PREGAME_POLL_MS, 10000)),
  /** a game with no file yet: first re-ask after this long, doubling up to the max. PROVISIONAL (not measured). */
  missingRetryMs: positive(env.VITE_PUBLISH_MISSING_RETRY_MS, 10000),
  missingMaxMs: positive(env.VITE_PUBLISH_MISSING_MAX_MS, 60000),
  staleAfterS: positive(env.VITE_PUBLISH_STALE_S, 30),
};
/** A threshold is never shorter than this many poll intervals: with a slow poll (say 60 s) a 30 s threshold would otherwise report
 *  "late" / "disconnected" between two healthy reads. */
const MIN_POLLS = 3;
/** "no answer for this long = disconnected", for the given poll interval */
export const lostAfterMs = (pollMs: number) => Math.max(PUBLISH_CONFIG.lostAfterMs, MIN_POLLS * pollMs);
/** "no new publish for this long = 갱신 지연", for the given poll interval */
export const staleAfterMs = (pollMs: number) => Math.max(PUBLISH_CONFIG.staleAfterS * 1000, MIN_POLLS * pollMs);

/** A finite positive interval that the browser can schedule without timer overflow. Strings are not manifest numbers. */
export const manifestPollMs = (seconds: unknown): number | null => typeof seconds === 'number' && Number.isFinite(seconds)
  && seconds * 1000 >= 1 && seconds * 1000 <= 2147483647 ? seconds * 1000 : null;

// ---------- the one place that builds a published file's URL ----------
/** `YYYY-MM-DD` that is a real calendar date (round-trips), nothing else. */
export function isCalendarDate(v: unknown): v is string {
  if (typeof v !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(v)) return false;
  const t = Date.parse(`${v}T00:00:00Z`);
  return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === v;
}
export type PublishedFile = { kind: 'dates' | 'manifest' | 'health' } | { kind: 'games'; date: string } | { kind: 'game'; gamePk: number };
/** Only these five shapes can be asked for. A date must be a real calendar date and a game number a positive safe integer; anything else
 *  (a value taken from the page URL, say) yields null and no request is made. No query string is ever added. */
export function publishedPath(file: PublishedFile): string | null {
  if (file.kind === 'games') return isCalendarDate(file.date) ? `games/${encodeURIComponent(file.date)}.json` : null;
  if (file.kind === 'game') return Number.isSafeInteger(file.gamePk) && file.gamePk > 0 ? `game/${encodeURIComponent(String(file.gamePk))}.json` : null;
  return `${file.kind}.json`;
}
export const publishUrl = (path: string) => `${PUBLISH_BASE}/v1/${path}`;

// ---------- the one place that knows where the publish extras sit in a file ----------
export type Published<T> = { body: T; seq: number | null; publishedAt: string | null; final: boolean };
type Raw = Record<string, any>;
/** PRIMARY (what the publisher on svc/ws-publisher writes; apps/observer/publish_contract/v1): the endpoint body, unwrapped, plus ONE
 *  added top-level key
 *      "publish": { "contract": "v1", "sequence": N, "published_at": "…Z"[, "final": bool] }
 *  `sequence` is the publish cycle that last wrote THIS file, so it is only ever compared with the same file's earlier value.
 *  Two older guesses are still accepted so a change of mind is absorbed here only:
 *      { ...body, publish_seq, published_at }          and          { publish_seq, published_at, body: { ... } }
 *  A file without an integer sequence comes back with `seq: null`; the reader (api.ts) does not apply such a file (it cannot be ordered). */
export function unwrapPublished<T>(raw: unknown): Published<T> {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Raw;
  const p = r.publish !== null && typeof r.publish === 'object' ? r.publish as Raw : null;
  const enveloped = !p && r.body !== null && typeof r.body === 'object' && !Array.isArray(r.body) && ('publish_seq' in r || 'published_at' in r);
  const seq = p ? p.sequence : r.publish_seq, at = p ? p.published_at : r.published_at;
  return { body: (enveloped ? r.body : raw) as T, final: p?.final === true,
    seq: Number.isSafeInteger(seq) ? seq as number : null,   // an ordering needs an integer: a string or a fraction is "no sequence"
    publishedAt: typeof at === 'string' && Number.isFinite(Date.parse(at)) ? at : null };
}

// ---------- server time ----------
/** "What time is it for the servers" without the device's wall clock. Two kinds of reading, kept apart:
 *    host     the hosting `Date` header (+ `Age` when a cache says how old its copy is)
 *    publish  a file's own `published_at` (the publisher's clock)
 *  Each reading can only be at or before the true server time, and is kept as an offset to the page's monotonic clock
 *  (`performance.now()`), so time goes on between answers and while the same cached answer repeats.
 *  - Host time wins whenever any answer has exposed one. Its largest offset is kept: a stale `Date` from a cache cannot pull it back,
 *    and after a device sleep (the monotonic clock stood still) the next answer simply raises it.
 *  - Publish times are used only when no host time is readable, and only the readings of the last `windowMs` count (largest wins).
 *    So one `published_at` in the future (publisher clock ahead, a bad file) stops mattering once the window has passed, while a
 *    publisher that stops still shows a lag that grows up to the window. The caller passes a window longer than its threshold.
 *    This estimate starts at zero lag: a publisher that had already stopped before the page was opened is noticed one threshold later.
 *  The device clock is never used as "now". */
export type ServerClock = 'host' | 'elapsed';
let hostOffset: number | null = null;
let publishReadings: { offset: number; at: number }[] = [];
const KEEP_MS = 3600000, KEEP_MAX = 4000;   // bookkeeping bounds only (memory), not thresholds
export function observeHostTime(ms: number) {
  if (!Number.isFinite(ms)) return;
  const offset = ms - performance.now();
  if (hostOffset === null || offset > hostOffset) hostOffset = offset;
}
export function observePublishTime(ms: number) {
  if (!Number.isFinite(ms)) return;
  const at = performance.now();
  publishReadings.push({ offset: ms - at, at });
  if (publishReadings.length > KEEP_MAX || at - publishReadings[0].at > KEEP_MS) publishReadings = publishReadings.filter(r => at - r.at <= KEEP_MS).slice(-KEEP_MAX);
}
export function serverNow(windowMs: number): { ms: number; clock: ServerClock } | null {
  const t = performance.now();
  if (hostOffset !== null) return { ms: hostOffset + t, clock: 'host' };
  let best: number | null = null;
  // `r.at <= t`: a reading "from the future" of the monotonic clock is from before a clock step and is ignored
  for (const r of publishReadings) if (r.at <= t && t - r.at <= windowMs && (best === null || r.offset > best)) best = r.offset;
  return best === null ? null : { ms: best + t, clock: 'elapsed' };
}
/** the window to pass to `serverNow` for a screen that reads every `pollMs` */
export const clockWindowMs = (pollMs: number) => 2 * staleAfterMs(pollMs);

// ---------- staleness ("갱신 지연") ----------
/** `seenAt`: the page's monotonic time at which this sequence of this file was first applied. */
export type PublishStamp = { seq: number | null; publishedAt: string | null; final: boolean; seenAt: number };
export type GameCheck = { checked_at: string; sequence_checked: number; file_sequence: number | null; feed_ok: boolean };
export type PublishHealth = PublishStamp & { gameChecks: Record<string, unknown> | null; failingGames: unknown[] };
/** Presence and validity are separate: malformed checks use full-file polling, but can never assert health. */
export function gameCheck(health: PublishHealth | null, game: number): { present: boolean; check: GameCheck | null } {
  const present = health?.gameChecks !== null && health?.gameChecks !== undefined && Object.hasOwn(health.gameChecks, String(game));
  const c = present ? health!.gameChecks![String(game)] as Partial<GameCheck> | null : null;
  const seq = (v: unknown) => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0;
  const valid = c && typeof c === 'object' && !Array.isArray(c) && typeof c.checked_at === 'string' && Number.isFinite(Date.parse(c.checked_at))
    && seq(c.sequence_checked) && (c.file_sequence === null || (seq(c.file_sequence) && c.file_sequence! <= c.sequence_checked!)) && typeof c.feed_ok === 'boolean';
  return { present, check: valid ? c as GameCheck : null };
}
export type PublishLag = { lagMs: number; clock: 'publisher' | ServerClock };
/** How far behind the published state is; null = cannot be judged (no publish time seen).
 *  1. publisher clock only: health.json's last publish − this game file's publish time. The publisher rewrites a game file that is not
 *     final on every cycle and writes health.json last, so a gap means this file stopped while the publisher went on (the game is in
 *     health.json's `unavailable_games`). Never for a file marked `publish.final` (it keeps its old time on purpose). The caller
 *     passes `file` only for a game whose own status says it is in play; for any other game (pregame, delayed, postponed) only
 *     health.json is judged, so a fresh health.json is never late.
 *  2. server time now (see above) − the newest publish time seen (health.json, else the game file): the publisher itself stopped.
 *  A publish time that is AHEAD of server time (publisher clock wrong, a bad file) says nothing: it is left out of 1, and for 2 the
 *  lag is then the time since that publish was first seen on this page, so a publisher that stops is still noticed. */
export function publishLag(file: PublishStamp | null, health: PublishStamp | null, now: { ms: number; clock: ServerClock } | null): PublishLag | null {
  const ms = (s: PublishStamp | null) => s?.publishedAt ? Date.parse(s.publishedAt) : null;
  const fileAt = file?.final ? null : ms(file), healthAt = ms(health);
  const ahead = (at: number) => now !== null && at > now.ms;
  const out: PublishLag[] = [];
  if (fileAt !== null && healthAt !== null && !ahead(fileAt) && !ahead(healthAt)) out.push({ lagMs: healthAt - fileAt, clock: 'publisher' });
  const newest = healthAt !== null && (fileAt === null || healthAt >= fileAt) ? { at: healthAt, stamp: health! } : fileAt !== null ? { at: fileAt, stamp: file! } : null;
  if (now && newest) out.push(ahead(newest.at) ? { lagMs: performance.now() - newest.stamp.seenAt, clock: 'elapsed' } : { lagMs: now.ms - newest.at, clock: now.clock });
  return out.length ? out.reduce((a, b) => b.lagMs > a.lagMs ? b : a) : null;
}
/** New protocol: freshness is the health/check time, never the age of an unchanged game file. */
export function gameCheckLag(health: PublishHealth | null, check: GameCheck | null, now: { ms: number; clock: ServerClock } | null): PublishLag {
  if (!health?.publishedAt || !check || !now) return { lagMs: Infinity, clock: 'elapsed' };
  const healthAt = Date.parse(health.publishedAt), checkedAt = Date.parse(check.checked_at);
  // Verification cannot follow the health publication that announces it.
  if (checkedAt > healthAt) return { lagMs: Infinity, clock: now.clock };
  const healthAge = healthAt > now.ms ? performance.now() - health.seenAt : now.ms - healthAt;
  const checkAge = checkedAt > now.ms ? performance.now() - health.seenAt : now.ms - checkedAt;
  return { lagMs: Math.max(healthAge, checkAge), clock: now.clock };
}
export const isPublishStale = (lag: PublishLag | null, pollMs: number) => lag !== null && lag.lagMs > staleAfterMs(pollMs);

// ---------- links ----------
/** The public bundle is plain files on a static host: no `/watch/<id>` rewrite is assumed and no demo binding is carried.
 *  Links use the query form on the page's own directory (`?game=<pk>`, `?date=<d>`), which `gameIdFromPath` already reads. */
const home = () => env.BASE_URL || './';
export const publicListHref = (date?: string | null) => date && isCalendarDate(date) ? `${home()}?date=${encodeURIComponent(date)}` : home();
export const publicGameHref = (pk: number) => `${home()}?game=${encodeURIComponent(String(pk))}`;
