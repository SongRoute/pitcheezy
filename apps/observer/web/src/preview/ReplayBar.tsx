/** Replay panel for the public-screen preview: step through the replayed game, jump to the injected states, and make the hosting
 *  stop answering or stop changing. A vertical tab on the right edge; it hides while the screen has a sheet open. Preview only. */
import { createRoot } from 'react-dom/client';
import { useState } from 'react';
import { STEPS, setReplay, useReplay } from './replay/control';
import type { Network } from './replay/control';
import './replay-bar.css';

const KIND: Record<string, string> = { pregame: '경기 전', pitch: '투구 전', break: '공수 교대', final: '경기 종료' };
const INJECT: Record<string, string> = { none: '', unconfirmed: '확인 중', feed_delayed: '자료 늦음', disconnected: '연결 끊김', correction: '정정', unsupported: '추천 없음' };

function Bar() {
  const [open, setOpen] = useState(false);
  const r = useReplay();
  const step = STEPS[r.index];
  const jumps = STEPS.map((s, i) => ({ s, i })).filter(({ s }) => s.inject !== 'none' || s.kind !== 'pitch');
  return <div className={`pz-rb${open ? ' open' : ''}`}>
    <button className="pz-rb-tab" onClick={() => setOpen(o => !o)} aria-expanded={open}>{open ? '닫기' : '재생'}</button>
    {open && <div className="pz-rb-body" role="region" aria-label="재생 조절">
      <p className="pz-rb-note">미리보기 전용 · 실제 경기 849843(컵스-파드리스)의 첫 65구를 한 공씩 재생해요. 상태 일부는 일부러 넣은 예시예요.</p>
      <fieldset><legend>재생</legend>
        <button onClick={() => setReplay({ index: r.index - 1 })} aria-label="이전 단계">‹</button>
        <button aria-pressed={r.playing} onClick={() => setReplay({ playing: !r.playing })}>{r.playing ? '멈춤' : '재생'}</button>
        <button onClick={() => setReplay({ index: r.index + 1 })} aria-label="다음 단계">›</button>
        <span className="pz-rb-step">{r.index + 1}/{STEPS.length} · {KIND[step.kind]}{step.kind === 'pitch' ? ` ${step.k + 1}구째` : ''}{step.inject !== 'none' ? ` · ${INJECT[step.inject]}` : ''}</span>
      </fieldset>
      <fieldset><legend>속도</legend>{[2000, 4000, 8000].map(ms => <button key={ms} aria-pressed={r.stepMs === ms} onClick={() => setReplay({ stepMs: ms })}>{ms / 1000}초</button>)}</fieldset>
      <fieldset><legend>호스팅</legend>{([['ok', '정상'], ['down', '응답 없음'], ['stall', '발행 멈춤']] as [Network, string][]).map(([n, t]) =>
        <button key={n} aria-pressed={r.network === n} onClick={() => setReplay({ network: n })}>{t}</button>)}</fieldset>
      <fieldset><legend>바로 가기</legend>{jumps.slice(0, 14).map(({ s, i }) => <button key={i} aria-pressed={r.index === i} onClick={() => setReplay({ index: i, playing: false })}>
        {s.inject !== 'none' ? INJECT[s.inject] : KIND[s.kind]}{s.kind === 'pitch' ? ` ${s.k + 1}` : ''}</button>)}</fieldset>
      <fieldset><legend>페이지</legend><a href="?game=990003">재생 경기</a><a href="?game=849843">지난 경기</a><a href="?">경기 목록</a></fieldset>
    </div>}
  </div>;
}
export function mountReplayBar() {
  const host = document.createElement('div');
  host.id = 'pz-replay-bar';
  document.body.appendChild(host);
  createRoot(host).render(<Bar />);
}
