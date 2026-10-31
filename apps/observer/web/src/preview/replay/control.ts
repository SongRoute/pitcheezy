/** Replay clock and the reviewer's overrides (preview builds only). One store shared by the fetch stand-in and the review panel. */
import { useSyncExternalStore } from 'react';
import { script } from './frames';
import type { Step } from './frames';

/** network: what the hosting answers. ok = normal; down = every read fails (client 연결 끊김); stall = files answer but stop
 *  changing (publisher stopped → 갱신 지연 after the threshold). */
export type Network = 'ok' | 'down' | 'stall';
export type ReplayState = { index: number; playing: boolean; stepMs: number; network: Network; stalledAt: number | null };
export const STEPS: Step[] = script();
/** a review link may start at a step: `?step=<n>&play=0&net=down|stall` (preview builds only) */
const q = new URLSearchParams(window.location.search);
const startNet = q.get('net');
// the public screen moves between list and game by real page loads (?game=…), so the replay position survives them in sessionStorage
const KEY = 'pitcheezy.preview.replay';
let saved: Partial<ReplayState> = {};
try { saved = JSON.parse(sessionStorage.getItem(KEY) ?? '{}') ?? {}; } catch { saved = {}; }
const startIndex = q.has('step') ? Number(q.get('step')) || 0 : typeof saved.index === 'number' ? saved.index : 0;
let state: ReplayState = { index: Math.max(0, Math.min(STEPS.length - 1, startIndex)), playing: q.has('play') ? q.get('play') !== '0' : saved.playing ?? true,
  stepMs: Number(q.get('speed')) > 0 ? Number(q.get('speed')) : 4000, network: startNet === 'down' || startNet === 'stall' ? startNet : 'ok',
  stalledAt: startNet === 'stall' ? Date.now() - 60000 : null };
const subs = new Set<() => void>();
const emit = () => subs.forEach(l => l());
export const getReplay = () => state;
export function setReplay(patch: Partial<ReplayState>) {
  const next = { ...state, ...patch };
  next.index = Math.max(0, Math.min(STEPS.length - 1, next.index));
  if (patch.network === 'stall' && state.network !== 'stall') next.stalledAt = Date.now();
  if (patch.network && patch.network !== 'stall') next.stalledAt = null;
  state = next; emit();
  try { sessionStorage.setItem(KEY, JSON.stringify({ index: state.index, playing: state.playing })); } catch { /* the replay still runs */ }
}
export const useReplay = () => useSyncExternalStore(l => { subs.add(l); return () => subs.delete(l); }, () => state, () => state);
let timer = 0;
export function startClock() {
  window.clearInterval(timer);
  let last = Date.now();
  timer = window.setInterval(() => {
    const now = Date.now();
    if (!state.playing || state.network === 'stall') { last = now; return; }
    if (now - last >= state.stepMs) { last = now; setReplay({ index: state.index + 1 >= STEPS.length ? 0 : state.index + 1 }); }
  }, 250);
}
