/** Review builds only (`__PZ_REVIEW__`): a small switcher for the variants under comparison and, in a replay build, the replay
 *  controls (step, play, speed, simulated hosting problems). Never part of a public bundle. */
import { useState } from 'react';
import { OPTION_LABELS, optionChoices, setOption, useOptions } from '../options';
import type { Options } from '../options';
import { STEPS, setReplay, useReplay } from '../replay/control';
import type { Network } from '../replay/control';
import './review.css';

const KEYS = Object.keys(OPTION_LABELS) as (keyof Options)[];
const KIND: Record<string, string> = { pregame: '경기 전', pitch: '투구 전', break: '공수 교대', final: '경기 종료' };
const INJECT: Record<string, string> = { unconfirmed: '확인 중', feed_delayed: '자료 늦음', disconnected: '연결 끊김(서버)', correction: '추천 정정', unsupported: '추천 없음' };

export default function Review() {
  const opts = useOptions();
  const r = useReplay();
  const [openPanel, setOpen] = useState(false);
  const step = STEPS[r.index];
  const jumps = STEPS.map((s, i) => ({ s, i })).filter(({ s }) => s.inject !== 'none' || s.kind !== 'pitch');
  return <div className={`v3-review${openPanel ? ' open' : ''}`}>
    <button className="v3-review-toggle" onClick={() => setOpen(o => !o)} aria-expanded={openPanel}>{openPanel ? '닫기' : '시안 설정'}</button>
    {openPanel && <div className="v3-review-body" role="region" aria-label="시안 설정">
      <p className="v3-review-note">미리보기 전용 · 실제 경기 849843(컵스–파드리스)의 첫 65구를 한 공씩 재생해요. 상태 일부는 일부러 넣은 예시예요.</p>
      {KEYS.map(k => <fieldset key={k}><legend>{OPTION_LABELS[k].title}</legend>
        {optionChoices(k).map(val => <button key={String(val)} aria-pressed={opts[k] === val} onClick={() => setOption(k, val as never)}>{(OPTION_LABELS[k].values as Record<string, string>)[String(val)]}</button>)}
      </fieldset>)}
      <fieldset><legend>재생</legend>
        <button onClick={() => setReplay({ index: r.index - 1 })} aria-label="이전 단계">‹</button>
        <button onClick={() => setReplay({ playing: !r.playing })}>{r.playing ? '멈춤' : '재생'}</button>
        <button onClick={() => setReplay({ index: r.index + 1 })} aria-label="다음 단계">›</button>
        <span className="v3-review-step">{r.index + 1}/{STEPS.length} · {KIND[step.kind]}{step.kind === 'pitch' ? ` ${step.k + 1}구째` : ''}{step.inject !== 'none' ? ` · ${INJECT[step.inject]}` : ''}</span>
      </fieldset>
      <fieldset><legend>속도</legend>{[2000, 4000, 8000].map(ms => <button key={ms} aria-pressed={r.stepMs === ms} onClick={() => setReplay({ stepMs: ms })}>{ms / 1000}초</button>)}</fieldset>
      <fieldset><legend>호스팅</legend>{([['ok', '정상'], ['down', '응답 없음'], ['stall', '발행 멈춤']] as [Network, string][]).map(([n, t]) =>
        <button key={n} aria-pressed={r.network === n} onClick={() => setReplay({ network: n })}>{t}</button>)}</fieldset>
      <fieldset><legend>바로 가기</legend>{jumps.slice(0, 14).map(({ s, i }) => <button key={i} aria-pressed={r.index === i} onClick={() => setReplay({ index: i, playing: false })}>
        {s.inject !== 'none' ? INJECT[s.inject] : KIND[s.kind]}{s.kind === 'pitch' ? ` ${s.k + 1}` : ''}</button>)}</fieldset>
    </div>}
  </div>;
}
