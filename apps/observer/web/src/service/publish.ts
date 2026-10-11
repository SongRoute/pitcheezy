/** Public (static) data source: the build reads published JSON files instead of the backend API.
 *  Contract: docs/pm/service/public-publish-contract-draft.md (v0). Selected at BUILD time by `VITE_PUBLISH_BASE`; when that is unset or
 *  empty nothing in this file changes what the screens do (API mode).
 *
 *    <base>/v1/dates.json  <base>/v1/manifest.json  <base>/v1/games/<YYYY-MM-DD>.json  <base>/v1/game/<game_pk>.json  <base>/v1/health.json
 *
 *  This file holds the mode switch, the config values, the ONE adapter that knows where the publish sequence / time sit in a file,
 *  and the pure staleness rule. Fetching lives in api.ts. */
const env = import.meta.env;

/** Non-empty `VITE_PUBLISH_BASE` = static mode. A URL or a path; `.` = next to index.html. Trailing slashes are dropped. */
export const STATIC_MODE: boolean = Boolean(env.VITE_PUBLISH_BASE);
export const PUBLISH_BASE: string = String(env.VITE_PUBLISH_BASE ?? '').trim().replace(/\/+$/, '');
export const publishUrl = (file: string) => `${PUBLISH_BASE}/v1/${file}`;

const positive = (v: unknown, fallback: number) => { const n = typeof v === 'string' && v.trim() ? Number(v) : NaN; return Number.isFinite(n) && n > 0 ? n : fallback; };
/** Static-mode config (build-time env; `.env.public` carries the defaults). The three cadences default to the API-mode values.
 *  PROVISIONAL: `staleAfterS` is NOT a measured value. It only reuses the screen's existing 30 s "connection is late" figure until the
 *  rehearsal measures the real publish interval and jitter; set `VITE_PUBLISH_STALE_S` from that measurement. */
export const PUBLISH_CONFIG = {
  livePollMs: positive(env.VITE_PUBLISH_POLL_MS, 2000),
  pregamePollMs: positive(env.VITE_PUBLISH_PREGAME_POLL_MS, 10000),
  firstLoadRetryMs: positive(env.VITE_PUBLISH_RETRY_MS, 4000),
  staleAfterS: positive(env.VITE_PUBLISH_STALE_S, 30),
};

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

// ---------- staleness ("갱신 지연") ----------
/** What one read of a published file tells about time. `hostNow` = the hosting server's clock for that answer (HTTP `Date` + `Age`),
 *  null when the header cannot be read (cross-origin without `Access-Control-Expose-Headers: Date`); `clientAt` = this device's clock. */
export type PublishStamp = { seq: number | null; publishedAt: string | null; final: boolean; hostNow: number | null; clientAt: number };
export type PublishLag = { lagS: number; clock: 'publisher' | 'host' | 'client' };
/** How far behind the published state is, in seconds; null = cannot be judged (no publish time in the files).
 *  1. publisher clock only: health.json's last publish − this game file's publish time. The publisher rewrites a game file that is not
 *     final on every cycle and writes health.json last, so a gap means this file stopped while the publisher went on (the game is in
 *     health.json's `unavailable_games`). Never for a file marked `publish.final` (it keeps its old time on purpose) and only while
 *     the screen is following the game (`following`).
 *  2. "now" − the newest publish time seen (health.json, else the followed game file): the publisher itself stopped. "now" is the hosting
 *     server's clock when readable; otherwise the device clock (weak: a wrong phone clock shifts this). */
export function publishLag(file: PublishStamp | null, health: PublishStamp | null, following: boolean): PublishLag | null {
  const ms = (s: PublishStamp | null) => s?.publishedAt ? Date.parse(s.publishedAt) : null;
  const fileAt = following && !file?.final ? ms(file) : null, healthAt = ms(health);
  const out: PublishLag[] = [];
  if (fileAt !== null && healthAt !== null) out.push({ lagS: (healthAt - fileAt) / 1000, clock: 'publisher' });
  const ref = healthAt !== null ? health! : fileAt !== null ? file! : null;
  if (ref) {
    const newest = Math.max(healthAt ?? -Infinity, fileAt ?? -Infinity);
    out.push(ref.hostNow !== null ? { lagS: (ref.hostNow - newest) / 1000, clock: 'host' } : { lagS: (ref.clientAt - newest) / 1000, clock: 'client' });
  }
  return out.length ? out.reduce((a, b) => b.lagS > a.lagS ? b : a) : null;
}
export const isPublishStale = (lag: PublishLag | null) => lag !== null && lag.lagS > PUBLISH_CONFIG.staleAfterS;

// ---------- links ----------
/** The public bundle is plain files on a static host: no `/watch/<id>` rewrite is assumed and no demo binding is carried.
 *  Links use the query form on the page's own directory (`?game=<pk>`, `?date=<d>`), which `gameIdFromPath` already reads. */
const home = () => env.BASE_URL || './';
export const publicListHref = (date?: string | null) => date ? `${home()}?date=${encodeURIComponent(date)}` : home();
export const publicGameHref = (pk: number) => `${home()}?game=${pk}`;
