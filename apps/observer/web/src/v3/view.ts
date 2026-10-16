/** What the live view shows, derived from the followed game (feed.ts). The rules are the classic screen's (GameScreen.tsx: stage,
 *  state slot, D1 "no silent flip" guard, connection line), for the "latest" view only; v3 shows past pitches in a detail panel.
 *  Nothing here invents a value: a field the published file does not carry stays null and is drawn as "—". */
import type { MutableRefObject } from 'react';
import { clockWindowMs, isPublishStale, lostAfterMs, publishLag, serverNow } from '../service/publish';
import type { PublishLag } from '../service/publish';
import type { Actual, Candidate, Detail, DisplayStateName, Pa, Person, PitchRow, PreviousPick, Rec, Situation } from '../service/model';
import { pitchName, plainZone } from '../service/model';
import type { Feed } from './feed';
import { STALE_MS } from './feed';

export type Drawn = { key: string | null; lastIdx: number; who: string; seen: Detail; sig: string; top: string; prev: { sig: string; top: string } | null };
/** The one connection story told to the viewer. `kind` drives colour; the words are always written. */
export type Conn = { kind: 'ok' | 'late' | 'lost' | 'checking'; label: string; detail: string | null };
export type Eff = { state: DisplayStateName; label: string; message: string | null };

export type GameView = {
  detail: Detail; live: boolean; final: boolean;
  situation: Situation | null; breakNow: boolean; ended: { inning: number; half: 'Top' | 'Bot' } | null; idle: boolean;
  batter: (Person & { side: string }) | null; pitcher: (Person & { hand: string }) | null; side: 'L' | 'R' | null;
  awayScore: number | null; homeScore: number | null;
  /** the recommendation the main zone shows: next pitch (live) or the last pitch's recorded one (not live) */
  rec: Rec | null; shownRec: boolean; cands: Candidate[]; nextNumber: number | null; nextPhase: boolean;
  stateSlot: boolean; eff: Eff | null; held: boolean; note: string | null; noteKind: string | null;
  lastPitch: PitchRow | null; paPitches: PitchRow[]; paKey: string | null; prevPa: Pa | null; currentPa: Pa | null;
  conn: Conn; lastOk: number | null; lagClock: PublishLag['clock'] | null;
  batterToday: { at_bats: number | null; hits: number | null; on_base: number | null } | null;
  pitcherToday: { pitches: number | null; strikeouts: number | null; earned_runs: number | null } | null;
  upcoming: { id: number | null; name: string | null; order: number }[];
  phase: string | null;
  /** an API-mode demo session (never in a public bundle): the chip says 시연, not 실시간, and shows no live dot */
  demo: boolean;
};

const CLIENT_LOST: Eff = { state: 'disconnected', label: '연결 끊김', message: '연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.' };
const PUBLISH_LATE: Eff = { state: 'disconnected', label: '갱신 지연', message: '새 내용이 늦어지고 있어요. 마지막으로 받은 내용을 보여 드려요.' };
const PUBLISH_CHECKING: Eff = { state: 'unconfirmed', label: '확인 중', message: '새 내용을 확인하고 있어요.' };
export const unconfirmed = (p: PitchRow | null | undefined) => p?.receipt?.status === 'unconfirmed';
const pickText = (pitch: string, zone: string | null | undefined) => zone ? `${pitch} · ${zone}` : pitch;
const previousText = (p: PreviousPick | null) => p ? pickText(pitchName(p.pitch_label, p.pitch_type), p.zone_label ? plainZone(p.zone_label) : null) : '';

export function deriveView(feed: Feed, detail: Detail, drawn: MutableRefObject<Drawn | null>): GameView {
  const { live, waitingToStart, pollMs, lastOk, now, hidden, resumed, error, fileStamp, health, checking } = feed;
  const { game } = detail;
  const pitches = detail.pitches;
  const lastPitch = pitches[pitches.length - 1] ?? null;
  const disp = detail.display ?? null;
  const breakNow = live && detail.live_source === true && (detail.phase === 'between_innings' || disp?.reason === 'between_innings');
  const ended = breakNow && lastPitch ? { inning: lastPitch.situation.inning, half: lastPitch.situation.half } : null;
  const situation: Situation | null = breakNow ? null : detail.current?.situation ?? lastPitch?.situation ?? null;
  const rec: Rec | null = live ? detail.next : lastPitch?.rec ?? null;
  const nextPhase = live;
  const paKey = detail.current?.pa_key ?? lastPitch?.pa_key ?? null;
  const paPitches = breakNow ? [] : pitches.filter(p => p.pa_key === paKey);
  const stateSlot = disp !== null && (nextPhase || disp.state === 'pregame');
  const slotNumber = Number(/^\d+:\d+:(\d+)$/.exec(disp?.pitch_key ?? '')?.[1] ?? NaN);
  const nextNumber = nextPhase && !breakNow ? (stateSlot && Number.isInteger(slotNumber) ? slotNumber
    : detail.current?.pa_key && lastPitch?.pa_key === detail.current.pa_key ? lastPitch.pitch_number + 1 : 1) : null;
  // a past pitch that is being re-checked is never paired with its recorded recommendation (fail closed)
  const pastUnconfirmed = !nextPhase && unconfirmed(lastPitch);
  const shownRec = stateSlot ? disp!.shown && rec?.status === 'ready' && rec.candidates.length > 0 : rec?.status === 'ready' && !pastUnconfirmed;
  const batter = breakNow ? null : detail.current?.batter ?? lastPitch?.batter ?? null;
  const pitcher = breakNow ? null : detail.current?.pitcher ?? lastPitch?.pitcher ?? null;
  const idle = detail.live_source === true && batter === null && pitcher === null;
  const side = batter?.side === 'R' || batter?.side === 'L' ? batter.side : null;

  const clientStale = live && !hidden && lastOk !== null && now - Math.max(lastOk, resumed) > (__PZ_STATIC__ ? lostAfterMs(pollMs) : STALE_MS);
  const feedDown = disp ? !disp.feed.ok : live && (detail.feed.ok === false || (detail.feed.stale_s ?? 0) > 30);
  const srv = __PZ_STATIC__ ? serverNow(clockWindowMs(pollMs)) : null;
  const startMs = game.start ? Date.parse(game.start) : NaN;
  const lag = !__PZ_STATIC__ ? null : live ? publishLag(fileStamp, health, srv)
    : waitingToStart && srv !== null && Number.isFinite(startMs) && srv.ms >= startMs ? publishLag(null, health, srv) : null;
  const lateNow = __PZ_STATIC__ && isPublishStale(lag, pollMs);
  const publishStale = lateNow && !checking, publishChecking = lateNow && checking;
  const demo = !__PZ_STATIC__ && detail.session !== null;
  const lagText = disp?.feed.delayed ? (disp.feed.lag_s !== null ? `${Math.floor(disp.feed.lag_s)}초 넘게 늦음` : '평소보다 늦음') : null;
  const conn: Conn = error || feedDown ? { kind: 'lost', label: '연결 끊김', detail: error ? '자동으로 다시 연결하고 있어요.' : '경기 자료 연결이 잠시 끊겼어요.' }
    : clientStale ? { kind: 'lost', label: '연결 끊김', detail: '마지막으로 받은 내용을 보여 드려요.' }
    : publishStale ? { kind: 'late', label: '갱신 지연', detail: PUBLISH_LATE.message }
    : publishChecking ? { kind: 'checking', label: '확인 중', detail: '새 내용을 확인하고 있어요.' }
    : lagText ? { kind: 'late', label: '자료 늦음', detail: `경기 자료가 ${lagText}. 실제 경기보다 늦을 수 있어요.` }
    : demo ? { kind: 'ok', label: detail.session_status === 'ended' ? '시연 종료' : detail.session_status === 'ready' ? '시연 대기' : detail.session_status === 'paused' ? '일시정지' : '시연 중', detail: null }
    : { kind: 'ok', label: live ? '실시간' : game.status === 'final' ? '경기 종료' : game.status === 'scheduled' ? '경기 전' : (disp?.label ?? '대기'), detail: null };

  // D1 "no silent flip" (classic screen, verbatim rules): server note first, otherwise compare with what was last drawn for this slot.
  let clientNote: string | null = null;
  if (stateSlot) {
    const key = disp!.pitch_key, lastIdx = lastPitch?.index ?? -1;
    const who = detail.current?.pa_key ?? (detail.current?.batter ? String(detail.current.batter.id) : '');
    const sig = shownRec ? rec!.candidates.map(c => `${c.pitch_type}@${c.zone_id ?? ''}`).join('|') : '';
    const top = shownRec ? pickText(rec!.candidates[0].pitch_label, rec!.candidates[0].zone_label) : '';
    let d = drawn.current;
    const other = !d || d.lastIdx !== lastIdx || (key !== null && d.key !== null && d.key !== key) || (key === null && who !== '' && d.who !== '' && d.who !== who);
    if (other) d = drawn.current = { key, lastIdx, who, seen: detail, sig, top, prev: null };
    else if (d!.seen !== detail) {
      d!.seen = detail; if (key !== null) d!.key = key; if (who !== '') d!.who = who;
      if (d!.sig !== sig) {
        const left = d!.sig !== '' ? { sig: d!.sig, top: d!.top } : d!.prev;
        d!.prev = left && left.sig !== sig ? left : null;
        d!.sig = sig; d!.top = top;
      }
    }
    if (d!.prev) clientNote = `${d!.sig ? '추천이 갱신됐어요' : '조금 전에 보여 드린 추천은 지금 표시하지 않아요'} (이전: ${d!.prev.top})`;
  }
  const fix = stateSlot ? disp!.correction : null;
  const fixPrev = previousText(fix?.previous ?? null);
  const note = fix ? `${fix.message}${fixPrev ? ` (이전: ${fixPrev})` : ''}` : clientNote;
  const clientLost = disp !== null && (error !== null || clientStale);
  const eff: Eff | null = !stateSlot ? null : clientLost ? CLIENT_LOST : publishStale ? PUBLISH_LATE : publishChecking ? PUBLISH_CHECKING
    : { state: disp!.state, label: disp!.label, message: disp!.message ?? (shownRec || disp!.state !== 'recommended' ? null : '추천을 준비하고 있어요.') };
  const held = stateSlot && shownRec && eff!.state === 'disconnected';
  const breakScore = (k: 'away' | 'home') => breakNow ? detail.linescore_asof?.[k].r ?? lastPitch?.situation[`${k}_score`] ?? null : null;
  const awayScore = game.away_score !== null ? game.away_score : situation?.away_score ?? breakScore('away');
  const homeScore = game.home_score !== null ? game.home_score : situation?.home_score ?? breakScore('home');
  const pasDone = detail.pas.filter(p => p.complete && p.pa_key !== paKey);
  const ctx = detail.current_context;
  return {
    detail, live, final: game.status === 'final', situation, breakNow, ended, idle, batter, pitcher, side, awayScore, homeScore,
    rec, shownRec: !!shownRec, cands: shownRec ? rec!.candidates : [], nextNumber, nextPhase,
    stateSlot, eff, held, note, noteKind: note ? (fix ? fix.kind : 'client') : null,
    lastPitch, paPitches, paKey, prevPa: pasDone[pasDone.length - 1] ?? null, currentPa: detail.pas.find(p => p.pa_key === paKey) ?? null,
    conn, lastOk, lagClock: publishStale ? lag!.clock : null,
    batterToday: ctx?.batter ?? null, pitcherToday: ctx?.pitcher ?? null, upcoming: ctx?.next_batters ?? [],
    phase: detail.current?.phase ?? detail.phase ?? null, demo,
  };
}

/** The recorded pre-pitch recommendation for one past pitch, with the classic fail-closed rule: an unconfirmed pitch has none. */
export function pastRec(p: PitchRow): { rec: Rec | null; actual: Actual | null; pending: boolean } {
  const pending = unconfirmed(p);
  return { rec: pending ? null : p.rec, actual: pending ? null : p.actual, pending };
}
