import { useEffect, useRef, useState } from 'react';
import { tapSync } from './viewer';
import { Sheet } from './Sheets';

/** 중계 맞추기: the fan's TV runs behind the ballpark, each by a different amount. One tap as the ball reaches the mitt on THEIR
 *  screen tells the server how far behind they are (a second tap on another pitch when one is not enough); ± fixes it by hand.
 *  The sheet only ever learns a number of seconds: nothing about the game comes back from a tap. */
export const TAP_GIVE_UP_S = 50;   // the server settles a tap within 45 s (PROVISIONAL, live_feed.TAP); a little longer, then stop asking
const FOLD_TEXT = '중계 맞추기 후 표시';
export { FOLD_TEXT };
type Phase = 'idle' | 'more' | 'wait' | 'ok' | 'fail';
const sleep = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms));

export default function SyncSheet({ gamePk, delay, max, synced, onDelay, onClose }: { gamePk: number; delay: number; max: number | null; synced: boolean;
  onDelay: (seconds: number) => void; onClose: () => void }) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [message, setMessage] = useState<string | null>(null);
  const taps = useRef<number[]>([]);   // performance.now() of each tap (this device's own clock only)
  const run = useRef(0);
  const now = useRef({ delay, synced }); now.current = { delay, synced };
  useEffect(() => () => { run.current++; }, []);
  const shown = Math.round(delay);

  const ask = async (id: number) => {
    const newest = taps.current[taps.current.length - 1];
    for (;;) {
      if (run.current !== id) return;
      const age = (t: number) => (performance.now() - t) / 1000;
      let answer = null;
      try { answer = await tapSync(gamePk, taps.current.map(age), now.current.delay, now.current.synced); } catch { answer = null; }
      if (run.current !== id) return;
      if (answer?.status === 'ok' && answer.delay_s !== null) { taps.current = []; setPhase('ok'); setMessage(null); onDelay(answer.delay_s); return; }
      if (answer === null || answer.status === 'wait') {
        if (age(newest) > TAP_GIVE_UP_S) { taps.current = []; setPhase('fail'); setMessage('이번에는 맞추지 못했어요. 다음 공에서 다시 눌러 주세요.'); return; }
        if (answer) { setPhase(answer.hint === 'tap_again' ? 'more' : 'wait'); setMessage(answer.message ?? '맞추는 중이에요. 잠시만 기다려 주세요.'); }
        await sleep((answer?.retry_s ?? 2) * 1000);
        continue;
      }
      if (answer.hint === 'tap_again') { setPhase('more'); setMessage(answer.message ?? '좋아요. 다음 공에서 한 번 더 눌러 주세요.'); return; }  // the taps so far are kept
      taps.current = []; setPhase('fail'); setMessage(answer.message ?? '이번에는 맞추지 못했어요. 다음 공에서 다시 눌러 주세요.');
      return;
    }
  };
  const onTap = () => {
    taps.current = [...taps.current, performance.now()].slice(-4);
    const first = taps.current.length < 2 && !synced;
    setPhase(first ? 'more' : 'wait');
    setMessage(first ? '좋아요. 다음 공에서 한 번 더 눌러 주세요.' : '맞추는 중이에요. 잠시만 기다려 주세요.');
    void ask(++run.current);
  };
  const adjust = (step: number) => { run.current++; taps.current = []; setPhase('ok'); setMessage(null); onDelay(Math.min(max ?? 600, Math.max(0, shown + step))); };
  const steps: [number, string][] = [[-5, '5초 덜 늦게'], [-1, '1초 덜 늦게'], [1, '1초 더 늦게'], [5, '5초 더 늦게']];

  return <Sheet title="중계 맞추기" onClose={onClose}>
    <section className="sv-sync" data-phase={phase}>
      <p className="sv-sync-why">TV 중계는 실제 경기보다 조금 늦어요. 내 중계에 맞춰 두면 결과가 TV보다 먼저 보이지 않아요.</p>
      <button className="sv-sync-tap" onClick={onTap} aria-describedby="sv-sync-status"><b>미트에 들어오는 순간 누르기</b><small>TV에서 공이 포수 미트에 들어올 때 한 번</small></button>
      <p className="sv-sync-status" id="sv-sync-status" role="status">{phase === 'ok' ? (shown > 0 ? `내 화면이 약 ${shown}초 늦어요` : '늦추지 않고 바로 보여 드려요')
        : message ?? (synced ? (shown > 0 ? `지금은 약 ${shown}초 늦춰서 보여 드리고 있어요.` : '지금은 늦추지 않고 바로 보여 드리고 있어요.') : `아직 맞추지 않았어요. 지금은 ${shown}초 늦춰서 보여 드려요.`)}</p>
      <div className="sv-sync-adjust" role="group" aria-label="조금씩 고치기">
        {steps.slice(0, 2).map(([step, label]) => <button key={step} className="sv-btn" onClick={() => adjust(step)} disabled={shown <= 0} aria-label={label}>{step}초</button>)}
        <span className="sv-sync-now" aria-label={`지금 ${shown}초 늦게 보기`}><b>{shown}</b>초 늦게</span>
        {steps.slice(2).map(([step, label]) => <button key={step} className="sv-btn" onClick={() => adjust(step)} disabled={max !== null && shown >= max} aria-label={label}>+{step}초</button>)}
      </div>
      <p className="sv-muted sv-sync-hint">화면이 TV보다 빠르면 +, 늦으면 −를 눌러 주세요.</p>
      <div className="sv-row sv-sync-foot">
        <button className="sv-btn" data-act="none" onClick={() => { run.current++; taps.current = []; setPhase('ok'); setMessage(null); onDelay(0); }}>중계 없이 보기</button>
        <button className="sv-btn" data-act="later" onClick={onClose}>{synced ? '닫기' : '나중에'}</button>
      </div>
    </section>
  </Sheet>;
}
