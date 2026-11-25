/** Preview builds only (`__PZ_REPLAY__`): answers the screen's published-file reads (`<base>/v1/*.json`) in the page, so the real
 *  reading path (api.ts → publish envelope, sequences, server time, lateness) runs unchanged against a replayed game.
 *    v1/dates.json        the archive date of the fixture game
 *    v1/manifest.json     live_dates = today (KST)
 *    v1/games/<today>     the replayed live game 990003
 *    v1/games/<archive>   the finished fixture game 849843
 *    v1/game/990003.json  the replay step now (frames.ts), rewritten every publish cycle
 *    v1/game/849843.json  the finished game as published (final)
 *    v1/health.json       written every cycle (last)
 *  Every other request goes to the real network unchanged. */
import { todayKst } from '../../service/model';
import { ARCHIVE_PK, FINAL_FILE, LIVE_PK, liveBody } from './frames';
import { STEPS, getReplay, startClock } from './control';

const CYCLE_MS = 2000;
const t0 = Date.now();
const ARCHIVE_DATE: string = FINAL_FILE.game.date_kst;
const ARCHIVE_AT = '2026-09-30T05:00:00Z';
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', date: new Date().toUTCString() } });
const notFound = () => json({ error: 'not_found' }, 404);

function cycle() {
  const s = getReplay();
  const at = s.network === 'stall' && s.stalledAt ? s.stalledAt : Date.now();
  const n = Math.floor((at - t0) / CYCLE_MS) + 1;
  return { seq: n, at: new Date(t0 + n * CYCLE_MS).toISOString().replace(/\.\d+Z$/, 'Z') };
}
const stamp = (body: Record<string, unknown>, c = cycle(), final?: boolean) => ({ ...body, publish: { contract: 'v1', sequence: c.seq, published_at: c.at, ...(final === undefined ? {} : { final }) } });

function liveNow() {
  const step = STEPS[getReplay().index];
  const body = liveBody(step);
  const today = todayKst();
  const start = new Date(t0 - 60000).toISOString().replace(/\.\d+Z$/, 'Z');
  body.game = { ...body.game, date_kst: today, official_date: today, start, start_kst: start };
  return body;
}
function answer(path: string): Response {
  const today = todayKst();
  if (path === 'health.json') { const c = cycle(); return json(stamp({ schema: 'pitcheezy-publish-health-v1', sequence: c.seq, published_at: c.at, publish_interval_s: CYCLE_MS / 1000,
    files_written: 6, feed_failing_games: [], unavailable_dates: [], unavailable_games: [] }, c)); }
  if (path === 'manifest.json') return json(stamp({ schema: 'pitcheezy-publish-manifest-v1', contract: 'v1', live_dates: [today], fixed_delay_s: 0,
    publish_interval_s: CYCLE_MS / 1000, provisional: ['replay'], commit: null, commit_dirty: null,
    files: { dates: 'v1/dates.json', games: 'v1/games/{date}.json', game: 'v1/game/{game_pk}.json', health: 'v1/health.json', manifest: 'v1/manifest.json', setup: 'v1/setup/{game_pk}.json' } }));
  if (path === 'dates.json') return json(stamp({ schema: 'pitcheezy-service-dates-v2', source: { kind: 'archive', session_id: null }, dates: [ARCHIVE_DATE] }));
  const games = /^games\/(\d{4}-\d{2}-\d{2})\.json$/.exec(path);
  if (games) {
    if (games[1] === today) return json(stamp({ schema: 'pitcheezy-service-games-v2', date: today, live_ok: true, source: { kind: 'live', session_id: null }, games: [liveNow().game] }));
    if (games[1] === ARCHIVE_DATE) return json(stamp({ schema: 'pitcheezy-service-games-v2', date: ARCHIVE_DATE, live_ok: true, source: { kind: 'archive', session_id: null }, games: [FINAL_FILE.game] }, { seq: 1, at: ARCHIVE_AT }));
    return notFound();
  }
  const game = /^game\/(\d+)\.json$/.exec(path);
  if (game && Number(game[1]) === LIVE_PK) { const body = liveNow(); return json(stamp(body, cycle(), body.game.status === 'final')); }
  if (game && Number(game[1]) === ARCHIVE_PK) return json(stamp(FINAL_FILE, { seq: 1, at: ARCHIVE_AT }, true));
  return notFound();
}

export function installReplay() {
  const real = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url, window.location.href);
    const m = /\/v1\/(.+\.json)$/.exec(url.pathname);
    if (!m) return real(input, init);
    await new Promise(r => setTimeout(r, 120));   // a little latency, as from a host
    if (init?.signal?.aborted) throw new DOMException('aborted', 'AbortError');
    if (getReplay().network === 'down') throw new TypeError('Failed to fetch');
    return answer(m[1]);
  };
  startClock();
}
