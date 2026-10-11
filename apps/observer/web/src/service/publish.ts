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
/** Static-mode config (build-time env; `.env.public` carries the defaults). Every default is the value the API-mode screen already uses.
 *  PROVISIONAL: `staleAfterS` is NOT a measured value. It only reuses the screen's existing 30 s "connection is late" figure until the
 *  rehearsal measures the real publish interval and jitter; set `VITE_PUBLISH_STALE_S` from that measurement. */
export const PUBLISH_CONFIG = {
  livePollMs: positive(env.VITE_PUBLISH_POLL_MS, 2000),
  pregamePollMs: positive(env.VITE_PUBLISH_PREGAME_POLL_MS, 10000),
  listPollMs: positive(env.VITE_PUBLISH_LIST_POLL_MS, 10000),
  firstLoadRetryMs: positive(env.VITE_PUBLISH_RETRY_MS, 4000),
  timeoutMs: positive(env.VITE_PUBLISH_TIMEOUT_MS, 15000),
  lostAfterMs: positive(env.VITE_PUBLISH_LOST_MS, 30000),
  tickMs: positive(env.VITE_PUBLISH_TICK_MS, 5000),
  staleAfterS: positive(env.VITE_PUBLISH_STALE_S, 30),
};
/** A threshold is never shorter than this many poll intervals: with a slow poll (say 60 s) a 30 s threshold would otherwise report
 *  "late" / "disconnected" between two healthy reads. */
const MIN_POLLS = 3;
/** "no answer for this long = disconnected", for the given poll interval */
export const lostAfterMs = (pollMs: number) => Math.max(PUBLISH_CONFIG.lostAfterMs, MIN_POLLS * pollMs);
/** "no new publish for this long = 갱신 지연", for the given poll interval */
export const staleAfterMs = (pollMs: number) => Math.max(PUBLISH_CONFIG.staleAfterS * 1000, MIN_POLLS * pollMs);

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
 *  A file with none of them is passed through with sequence / time unknown (no ordering or staleness can then be judged from it). */
export function unwrapPublished<T>(raw: unknown): Published<T> {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Raw;
  const p = r.publish !== null && typeof r.publish === 'object' ? r.publish as Raw : null;
  const enveloped = !p && r.body !== null && typeof r.body === 'object' && !Array.isArray(r.body) && ('publish_seq' in r || 'published_at' in r);
  const seq = p ? p.sequence : r.publish_seq, at = p ? p.published_at : r.published_at;
  return { body: (enveloped ? r.body : raw) as T, final: p?.final === true,
    seq: typeof seq === 'number' && Number.isFinite(seq) ? seq : null,
    publishedAt: typeof at === 'string' && Number.isFinite(Date.parse(at)) ? at : null };
}

// ---------- server time ----------
/** "What time is it for the servers" without the device's wall clock. Every answer gives a reading that can only be at or before the
 *  true server time: the hosting `Date` header (+ `Age` when a cache says how old its copy is), and a file's own `published_at`.
 *  A reading is kept as an offset to the page's monotonic clock (`performance.now()`), and the LARGEST offset wins, so
 *    - time goes on between answers and while the same cached answer repeats (the anchor is never reset by an older reading),
 *    - a stale `Date` from a cache cannot pull the estimate back,
 *    - a hosting time, when any answer exposes one, beats a publish time (it is later).
 *  Without any hosting time the estimate is "the newest publish time seen + time elapsed since": it starts at zero lag, so a publisher
 *  that had already stopped before the page was opened is noticed only one threshold later. The device clock is never used. */
export type ServerClock = 'host' | 'elapsed';
let anchor: { offset: number; clock: ServerClock } | null = null;
export function observeServerTime(ms: number | null, clock: ServerClock) {
  if (ms === null || !Number.isFinite(ms)) return;
  const offset = ms - performance.now();
  if (anchor === null || offset > anchor.offset) anchor = { offset, clock };
}
export const serverNow = (): { ms: number; clock: ServerClock } | null => anchor ? { ms: anchor.offset + performance.now(), clock: anchor.clock } : null;

// ---------- staleness ("갱신 지연") ----------
export type PublishStamp = { seq: number | null; publishedAt: string | null; final: boolean };
export type PublishLag = { lagMs: number; clock: 'publisher' | ServerClock };
/** How far behind the published state is; null = cannot be judged (no publish time seen).
 *  1. publisher clock only: health.json's last publish − this game file's publish time. The publisher rewrites a game file that is not
 *     final on every cycle and writes health.json last, so a gap means this file stopped while the publisher went on (the game is in
 *     health.json's `unavailable_games`). Never for a file marked `publish.final` (it keeps its old time on purpose).
 *  2. server time now (see above) − the newest publish time seen (health.json, else the game file): the publisher itself stopped. */
export function publishLag(file: PublishStamp | null, health: PublishStamp | null, now: { ms: number; clock: ServerClock } | null): PublishLag | null {
  const ms = (s: PublishStamp | null) => s?.publishedAt ? Date.parse(s.publishedAt) : null;
  const fileAt = file?.final ? null : ms(file), healthAt = ms(health);
  const out: PublishLag[] = [];
  if (fileAt !== null && healthAt !== null) out.push({ lagMs: healthAt - fileAt, clock: 'publisher' });
  const newest = Math.max(healthAt ?? -Infinity, fileAt ?? -Infinity);
  if (now && Number.isFinite(newest)) out.push({ lagMs: now.ms - newest, clock: now.clock });
  return out.length ? out.reduce((a, b) => b.lagMs > a.lagMs ? b : a) : null;
}
export const isPublishStale = (lag: PublishLag | null, pollMs: number) => lag !== null && lag.lagMs > staleAfterMs(pollMs);

// ---------- links ----------
/** The public bundle is plain files on a static host: no `/watch/<id>` rewrite is assumed and no demo binding is carried.
 *  Links use the query form on the page's own directory (`?game=<pk>`, `?date=<d>`), which `gameIdFromPath` already reads. */
const home = () => env.BASE_URL || './';
export const publicListHref = (date?: string | null) => date && isCalendarDate(date) ? `${home()}?date=${encodeURIComponent(date)}` : home();
export const publicGameHref = (pk: number) => `${home()}?game=${encodeURIComponent(String(pk))}`;
