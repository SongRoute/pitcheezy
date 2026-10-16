/** Following one game: the reading, polling and lateness rules of the classic screen (src/service/GameScreen.tsx, svc/ws-static-web
 *  ef4e4df), moved into one hook so the v3 screens draw from the same state. Behaviour is kept line for line on purpose; only the
 *  past-pitch "pin" is gone: v3 opens a past pitch in a detail panel while the live view keeps following (so `auto` is always on).
 *  Any change here must be made in the classic screen too until one of the two is retired. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { FetchError, fetchDetail, fetchPublishHealth, fetchPublishedDetail, isGamePublishedSoon, mergeDetail } from '../service/api';
import { PUBLISH_CONFIG } from '../service/publish';
import type { PublishStamp } from '../service/publish';
import type { Detail } from '../service/model';

/** the classic screen's API-mode cadences (src/service/GameScreen.tsx POLL_MS / PREGAME_POLL_MS / FIRST_LOAD_RETRY_MS / STALE_MS) */
export const POLL_MS = 2000, PREGAME_POLL_MS = 10000, FIRST_LOAD_RETRY_MS = 4000, STALE_MS = 30000;
export const CADENCE = __PZ_STATIC__ ? { live: PUBLISH_CONFIG.livePollMs, pregame: PUBLISH_CONFIG.pregamePollMs, retry: PUBLISH_CONFIG.firstLoadRetryMs }
  : { live: POLL_MS, pregame: PREGAME_POLL_MS, retry: FIRST_LOAD_RETRY_MS };

export type Feed = {
  detail: Detail | null; error: { text: string; status: number } | null;
  lastOk: number | null; now: number; hidden: boolean; resumed: number;
  fileStamp: PublishStamp | null; health: PublishStamp | null; checking: boolean; busy: boolean;
  live: boolean; waitingToStart: boolean; pollMs: number;
  firstLoadRetry: boolean; gaveUp: boolean;
  reload: () => void; askAgain: () => void;
};

export function useGameFeed(gamePk: number): Feed {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<{ text: string; status: number } | null>(null);
  const [lastOk, setLastOk] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [hidden, setHidden] = useState(() => document.hidden);
  const [resumed, setResumed] = useState(0);
  const [busy, setBusy] = useState(false);
  const detailRef = useRef<Detail | null>(null);
  const [fileStamp, setFileStamp] = useState<PublishStamp | null>(null);
  const [health, setHealth] = useState<PublishStamp | null>(null);
  const [checking, setChecking] = useState(false);
  const inflight = useRef<AbortController | null>(null);
  const auto = true;

  const healthBusy = useRef<AbortController | null>(null);
  const loadHealth = useCallback(() => {
    if (healthBusy.current) return;
    const ctrl = healthBusy.current = new AbortController();
    fetchPublishHealth(ctrl.signal).then(h => { if (!ctrl.signal.aborted) setHealth(h); }, () => undefined)
      .finally(() => { if (healthBusy.current === ctrl) healthBusy.current = null; });
  }, []);
  const load = useCallback(async (full = false, force = false) => {
    if (inflight.current && !inflight.current.signal.aborted && !force) return;
    inflight.current?.abort();
    const ctrl = new AbortController(); inflight.current = ctrl;
    setBusy(true);
    try {
      if (__PZ_STATIC__) {
        const first = detailRef.current === null;
        if (!first) loadHealth();
        const got = await fetchPublishedDetail(gamePk, ctrl.signal);
        if (ctrl.signal.aborted) return;
        if (first) loadHealth();
        detailRef.current = got.detail; setDetail(got.detail); setFileStamp(got.stamp); setError(null); setLastOk(Date.now());
        return;
      }
      const old = detailRef.current;
      const last = old?.pitches.length ? old.pitches[old.pitches.length - 1].index : undefined;
      let next = await fetchDetail(gamePk, ctrl.signal, full || !old || last === undefined ? undefined : last);
      if (old && !full && last !== undefined) {
        const merged = mergeDetail(old, next);
        next = merged ?? await fetchDetail(gamePk, ctrl.signal);
      }
      if (ctrl.signal.aborted) return;
      detailRef.current = next; setDetail(next); setError(null); setLastOk(Date.now());
    } catch (e) {
      if (ctrl.signal.aborted) return;
      setError(e instanceof FetchError ? { text: e.message, status: e.status } : { text: '자료를 불러오지 못했어요.', status: 0 });
    } finally { if (inflight.current === ctrl) { inflight.current = null; setBusy(false); if (__PZ_STATIC__) setChecking(false); } }
  }, [gamePk, loadHealth]);

  useEffect(() => { void load(true, true); return () => { inflight.current?.abort(); healthBusy.current?.abort(); }; }, [load]);
  const live = detail ? detail.session_status !== 'ended' && (detail.game.status === 'live' || (detail.next !== null && detail.game.status !== 'final')) : false;
  const waitingToStart = detail !== null && !live && detail.session === null && detail.game.status !== 'final'
    && (detail.display?.state === 'pregame' || detail.live_source === true);
  const pollMs = live ? CADENCE.live : waitingToStart ? CADENCE.pregame : 0;
  useEffect(() => {
    if (!auto || !pollMs) return;
    const id = window.setInterval(() => { if (!document.hidden) void load(); }, pollMs);
    return () => window.clearInterval(id);
  }, [auto, pollMs, load]);
  const firstLoadRetry = detail === null && error !== null && (error.status === 0 || error.status >= 500);
  const missing = __PZ_STATIC__ && detail === null && error !== null && error.status === 404;
  const holdHidden = __PZ_STATIC__ && hidden;
  useEffect(() => {
    if (!firstLoadRetry || holdHidden) return;
    const id = window.setTimeout(() => void load(true), CADENCE.retry);
    return () => window.clearTimeout(id);
  }, [firstLoadRetry, holdHidden, error, load]);
  const missingTries = useRef(0);
  const [gaveUp, setGaveUp] = useState(false);
  useEffect(() => {
    if (!missing || holdHidden || gaveUp) return;
    const ctrl = new AbortController();
    const id = window.setTimeout(async () => {
      const soon = await isGamePublishedSoon(gamePk, ctrl.signal);
      if (ctrl.signal.aborted) return;
      if (soon === false) { setGaveUp(true); return; }
      missingTries.current += 1;
      void load(true);
    }, Math.min(PUBLISH_CONFIG.missingRetryMs * 2 ** missingTries.current, PUBLISH_CONFIG.missingMaxMs));
    return () => { window.clearTimeout(id); ctrl.abort(); };
  }, [missing, holdHidden, gaveUp, error, load, gamePk]);
  const askAgain = useCallback(() => { missingTries.current = 0; setGaveUp(false); void load(true, true); }, [load]);
  useEffect(() => { const id = window.setInterval(() => setNow(Date.now()), __PZ_STATIC__ ? PUBLISH_CONFIG.tickMs : 5000); return () => window.clearInterval(id); }, []);
  useEffect(() => {
    const recheck = () => { if (__PZ_STATIC__ && auto && pollMs && detailRef.current !== null) setChecking(true); };
    const onVis = () => { setHidden(document.hidden); if (!document.hidden) { setResumed(Date.now()); setNow(Date.now()); } if (!document.hidden && auto && pollMs) { recheck(); void load(); }
      if (__PZ_STATIC__ && !document.hidden && detailRef.current === null) void load(true); };
    const onOnline = () => { if (__PZ_STATIC__ && document.hidden) return; if (auto && (pollMs || detailRef.current === null)) { recheck(); void load(detailRef.current === null); } };
    document.addEventListener('visibilitychange', onVis); window.addEventListener('online', onOnline);
    let wall = Date.now(), mono = performance.now();
    const awake = !__PZ_STATIC__ ? 0 : window.setInterval(() => {
      const w = Date.now(), m = performance.now(), gap = (w - wall) - (m - mono); wall = w; mono = m;
      if (gap > 2 * PUBLISH_CONFIG.tickMs && !document.hidden && auto && pollMs) { setResumed(Date.now()); setNow(Date.now()); recheck(); void load(); }
    }, PUBLISH_CONFIG.tickMs);
    return () => { document.removeEventListener('visibilitychange', onVis); window.removeEventListener('online', onOnline); window.clearInterval(awake); };
  }, [auto, pollMs, load]);

  const reload = useCallback(() => void load(true, true), [load]);
  return { detail, error, lastOk, now, hidden, resumed, fileStamp, health, checking, busy, live, waitingToStart, pollMs, firstLoadRetry, gaveUp, reload, askAgain };
}
