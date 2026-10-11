/** Viewer delay ("중계 맞추기") of THIS device: the fan's TV runs behind the ballpark, so the screen can ask the server for the
 *  game as it stood `delay_s` seconds ago. OFF unless BOTH hold:
 *  (a) build time: this is not the public build. The public service publishes one delay-0 body for every fan as static files, so
 *      its bundle (`VITE_PUBLISH_BASE` set) never asks for a delay and never shows a delay control, whatever a payload says;
 *  (b) run time: the server says so (`features.viewer_delay.enabled`, else `viewer.enabled`) — the owner's switch on the direct
 *      path to the server (phone demo, internal).
 *  While off nothing here is used: no `delay` parameter, no localStorage access, no prompt, nothing folded, no entry point.
 *  The switch is learned from an answer, per page load; the first request therefore carries no delay, and an answer that turns
 *  the feature on is never drawn: the caller asks again at this device's delay.
 *  The setting lives in localStorage (every access wrapped); a copy in memory keeps the screen working when storage is blocked. */
export const VIEWER_BUILD: boolean = !import.meta.env.VITE_PUBLISH_BASE;
export type ViewerSetting = {
  /** seconds this device watches behind; null = never set here → the server's default */
  delay_s: number | null;
  /** the fan set the delay themselves (tap sync, fine adjust, or "중계 없이 보기") */
  synced: boolean;
  /** the one-time sync question was already asked on this device */
  asked: boolean;
};
/** What the server sent about this viewer (live route only, feature on); null otherwise → the screen as before the feature. */
export type Viewer = { delay_s: number; default_s: number; max_delay_s: number | null; buffering: boolean; wait_s: number | null; prompt: boolean; fold: boolean };
type Raw = Record<string, any>;

const KEY = 'pitcheezy.viewer';
/** Used only when an answer names no default of its own (PROVISIONAL, same as live_config.json features.viewer_delay.default_s). */
export const FIRST_DEFAULT_S = 30;
let active = false;                      // the server's switch as last answered on this page (never true in the public build)
let memory: ViewerSetting | null = null;
let serverDefault: number | null = null;
const num = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null;

export function readViewer(): ViewerSetting {
  if (memory) return memory;
  if (!active) return { delay_s: null, synced: false, asked: false };   // feature off: storage is not touched
  let stored: Partial<ViewerSetting> | null = null;
  try { const raw = localStorage.getItem(KEY); stored = raw ? JSON.parse(raw) : null; } catch { stored = null; }
  const d = stored && typeof stored === 'object' ? stored : {};
  memory = { delay_s: num(d.delay_s) !== null && d.delay_s! >= 0 ? d.delay_s! : null, synced: d.synced === true, asked: d.asked === true };
  return memory;
}
export function writeViewer(patch: Partial<ViewerSetting>): ViewerSetting {
  if (!active) return readViewer();
  memory = { ...readViewer(), ...patch };
  try { localStorage.setItem(KEY, JSON.stringify(memory)); } catch { /* storage blocked: the setting lasts for this page */ }
  return memory;
}
/** The server's switch in an answer (game or list body). An older server says nothing → off. */
export const viewerOn = (raw: Raw | null | undefined): boolean => VIEWER_BUILD && (raw?.features?.viewer_delay?.enabled ?? raw?.viewer?.enabled) === true;
/** Take the switch (and the server's default) from an answer. True = it differs from what this page assumed when it asked. */
export function learnViewer(raw: Raw | null | undefined): boolean {
  const on = viewerOn(raw);
  const d = num(raw?.viewer?.default_s) ?? num(raw?.features?.viewer_delay?.default_s);
  if (on && d !== null) serverDefault = d;
  if (on === active) return false;
  active = on; if (!on) memory = null;
  return true;
}
/** The delay this device wants: its own setting, else the server's default. */
export const wantedDelay = (s: ViewerSetting = readViewer()) => s.delay_s ?? serverDefault ?? FIRST_DEFAULT_S;
/** The `delay` a request carries, or null for none (feature off as far as this page knows). */
export const requestDelay = (): number | null => active ? wantedDelay() : null;
export const withDelay = (path: string, delay: number | null = requestDelay()) => delay === null ? path : `${path}${path.includes('?') ? '&' : '?'}delay=${delay}`;
/** `viewer` of the live route with the feature on → Viewer; anything else (off, public build, archive, demo, older server) → null. */
export function normalizeViewer(raw: Raw): Viewer | null {
  const v = raw.viewer, f = raw.features?.viewer_delay;
  if (!viewerOn(raw) || !v || typeof v !== 'object' || raw.source?.kind !== 'live' || num(v.delay_s) === null) return null;
  return { delay_s: v.delay_s, default_s: num(v.default_s) ?? num(f?.default_s) ?? FIRST_DEFAULT_S, max_delay_s: num(v.max_delay_s), buffering: v.buffering === true,
    wait_s: num(v.wait_s), prompt: f?.sync_prompt !== false, fold: f?.fold_actual_until_synced !== false };
}

// ---------- one-time tap sync ----------
export type TapAnswer = { status: 'ok' | 'unclear' | 'wait'; delay_s: number | null; hint: 'tap_again' | 'reset' | 'no_pitch' | null; message: string | null; retry_s: number | null };
/** `ages`: how many seconds ago each tap was, measured on this device (no clock is compared with the server's). The answer is a
 *  number at most, never anything about the game. Throws when the server cannot be reached or refuses. */
export async function tapSync(game: number, ages: number[], current: number, sure: boolean): Promise<TapAnswer> {
  const response = await fetch(`/api/service/games/${game}/tap?ago=${ages.map(a => Math.max(0, a).toFixed(1)).join(',')}&current=${current}&sure=${sure ? 1 : 0}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(String(response.status));
  const raw: Raw = await response.json();
  const status = raw.status === 'ok' && num(raw.delay_s) !== null ? 'ok' : raw.status === 'wait' ? 'wait' : 'unclear';
  return { status, delay_s: status === 'ok' ? raw.delay_s : null, hint: raw.hint === 'tap_again' || raw.hint === 'reset' || raw.hint === 'no_pitch' ? raw.hint : null,
    message: typeof raw.message === 'string' && raw.message.trim() ? raw.message.trim() : null, retry_s: num(raw.retry_s) };
}
