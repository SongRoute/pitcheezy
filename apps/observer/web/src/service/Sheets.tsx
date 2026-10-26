import { createContext, useContext, useEffect, useRef, useState } from 'react';
import type React from 'react';
import type { ReactNode } from 'react';
import { Fragment } from 'react';
import type { Actual, Candidate, Pa, PitchRow, Rec } from './model';
import { halfText, inningText, kstTime, policyShare, resultKind } from './model';
import Zone, { G1Dot, RankSq, ZoneLegend } from './Zone';
import type { Bounds } from './model';

/** Which sheet frame the tree below uses: 'c' = the 10/26 game-screen frame (half → full, dimmed, modal at every width); 'classic' =
 *  the frame the game list keeps (unchanged). */
export const SheetFrame = createContext<'classic' | 'c'>('classic');

/** Bottom sheet: the only place with inner scrolling. Esc / backdrop / 닫기 close it and focus returns to the opener.
 *  Frame C (UI 10/26 ⑧): opens at about half the screen so the zone stays visible; dragging the handle or title up, scrolling the
 *  body, or the handle button (keyboard: Enter/Space) opens it to full height; dragging down goes back to half, then closes. */
export function Sheet({ title, onClose, children, labelledBy = 'sv-sheet-title', headExtra = null, half = null }: { title: string; onClose: () => void; children: ReactNode; labelledBy?: string; headExtra?: ReactNode;
  /** frame C: this sheet's half height when its upper content needs more than the default (design 23/25: about 56% of the screen) */
  half?: string | null }) {
  const c = useContext(SheetFrame) === 'c';
  const box = useRef<HTMLDivElement>(null);
  const [wide, setWide] = useState(() => window.matchMedia('(min-width: 1200px)').matches);
  useEffect(() => { const mq = window.matchMedia('(min-width: 1200px)'); const change = () => setWide(mq.matches); mq.addEventListener('change', change); return () => mq.removeEventListener('change', change); }, []);
  const modal = c || !wide;
  const [full, setFull] = useState(false);
  const [dy, setDy] = useState(0);
  const drag = useRef<number | null>(null);
  const moved = useRef(false);  // a drag just ended on the handle: the click the browser sends after it is not a tap
  const closeRef = useRef(onClose); closeRef.current = onClose;
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); closeRef.current(); }
      if (e.key === 'Tab' && modal) {
        const nodes = [...(box.current?.querySelectorAll<HTMLElement>('button:not([disabled]),a[href],select,input,[tabindex="0"]') ?? [])];
        const first = nodes[0], last = nodes[nodes.length - 1];
        if (!first) { e.preventDefault(); return; }
        if (e.shiftKey && (document.activeElement === first || document.activeElement === box.current)) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && (document.activeElement === last || document.activeElement === box.current)) { e.preventDefault(); first.focus(); }
      }
    };
    window.addEventListener('keydown', onKey);
    if (modal) document.body.classList.add('sv-locked');
    return () => { window.removeEventListener('keydown', onKey); document.body.classList.remove('sv-locked'); opener?.focus?.(); };
  }, [modal]);
  // swipe on the handle / title row (buttons in the title row keep their taps)
  const down = (e: React.PointerEvent) => { if (!c || ((e.target as Element).closest('button') && !(e.target as Element).closest('.sv-sheet-grip'))) return; moved.current = false; drag.current = e.clientY; try { (e.currentTarget as Element).setPointerCapture(e.pointerId); } catch { /* not an active pointer: the gesture still works without capture */ } };
  const move = (e: React.PointerEvent) => { if (drag.current !== null) setDy(e.clientY - drag.current); };
  const up = (e: React.PointerEvent) => { if (drag.current === null) return; const d = e.clientY - drag.current; drag.current = null; setDy(0);
    if (Math.abs(d) > 8) moved.current = true;
    if (d < -40) setFull(true); else if (d > 60) { if (full) setFull(false); else onClose(); } };
  // a cancelled gesture (scroll taken over, system gesture) only resets: it never opens, shrinks or closes the sheet
  const cancel = () => { drag.current = null; moved.current = false; setDy(0); };
  const dragStyle = () => ({ transform: `translateY(${full ? Math.max(0, dy) : dy > 0 ? dy : 0}px)`, ...(dy < 0 && !full ? { height: `calc(var(--sv-sheet-half) + ${-dy}px)` } : {}) });
  const swipe = c ? { onPointerDown: down, onPointerMove: move, onPointerUp: up, onPointerCancel: cancel } : {};
  return <div className={`sv-backdrop${modal ? '' : ' nonmodal'}${c ? ' frame-c' : ''}`} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
    <div className={`sv-sheet${c ? ` frame-c ${full ? 'full' : 'half'}${dy ? ' dragging' : ''}` : ''}`} role="dialog" aria-modal={modal ? true : undefined} aria-labelledby={labelledBy} tabIndex={-1} ref={box}
      style={c && half ? { ['--sv-sheet-half' as string]: half, ...(dy ? dragStyle() : {}) } : c && dy ? { transform: `translateY(${full ? Math.max(0, dy) : dy > 0 ? dy : 0}px)`, ...(dy < 0 && !full ? { height: `calc(var(--sv-sheet-half) + ${-dy}px)` } : {}) } : undefined}>
      {c && <button className="sv-sheet-grip" {...swipe} onClick={() => { if (moved.current) { moved.current = false; return; } setFull(f => !f); }} aria-expanded={full} aria-label={full ? '시트 반만 보기' : '시트 크게 보기'}><i aria-hidden="true" /></button>}
      <div className="sv-sheet-head" {...swipe}><h2 id={labelledBy}>{title}</h2>{headExtra}<button className={c ? 'sv-sheet-close' : 'sv-btn'} onClick={onClose}>닫기</button></div>
      <div className="sv-sheet-body" onScroll={c && !full ? e => { if (e.currentTarget.scrollTop > 4) setFull(true); } : undefined}>{children}</div>
    </div>
  </div>;
}

export const LIMIT_LINE = '검증 전 실험 · 실제 선택보다 낫다는 근거가 없어요. 정책 선택 비중은 성공 확률이 아니에요.';
export const SPOILER_LINE = 'TV 중계보다 결과가 먼저 보일 수 있어요.';
export const publishDelayLine = (seconds: number | null) => seconds === null ? '' : seconds === 0 ? '공통 지연 설정 없이 발행돼요. ' : `공통 ${seconds}초 지연 설정으로 발행돼요. `;
/** Full explanations remain available from the visible short notices. */
const LIMITS = [
  '추천은 검증 전 실험 버전이에요. 2026 정규시즌 1회 평가에서는 추천이 실제 선택보다 낫다는 근거를 확인하지 못했어요.',
  '‘정책 선택 비중’은 추천 정책이 이 상황에서 그 구종을 고르는 비중이에요. 성공·적중 확률이 아니에요.',
  '추천 위치는 그 투수가 실제로 던진 공의 분포(2025년까지)로 잡은 대략 위치이며, 위치 자체는 평가되지 않았어요.',
  '지난 경기의 추천은 경기 뒤 그 공을 던지기 전 정보만으로 계산한 기록이에요. 경기 중 실시간으로 보여 준 추천이 아니에요.',
  '포수 미트 표시는 중계 영상에서 읽은 가로 위치 추정이며, 높이는 다루지 않아요. 실제 투구 뒤에만 보여요.',
  '날짜와 시간은 모두 한국 시간이에요. 경기 날짜는 원래 예정 시작 시각 기준이에요.',
  '공개 화면의 지연 설정은 개인의 TV·스트리밍과 맞춘 값이 아니에요. 자료 수신·전송에도 시간이 걸리며, 화면이 중계보다 먼저 결과를 보여 줄 수 있어요.',
];
export function InfoSheet({ onClose, notes }: { onClose: () => void; notes: string[] }) {
  return <Sheet title="이 화면의 기준과 한계" onClose={onClose}>
    <ul className="sv-notes">{[...LIMITS, ...notes.filter(n => !LIMITS.includes(n))].map(n => <li key={n}>{n}</li>)}</ul>
  </Sheet>;
}

export function PaSheet({ pas, selected, onPick, onClose, onBox, foldText = null }: { pas: Pa[]; selected: string | null; onPick: (pa: Pa) => void; onClose: () => void; onBox: () => void;
  /** viewer delay, device not synced yet: finished plate appearances do not say how they ended */
  foldText?: string | null }) {
  const current = pas.find(p => p.pa_key === selected) ?? pas[0];
  const key = (p: Pa) => `${p.inning}:${p.half}`;
  const [half, setHalf] = useState(current ? key(current) : '');
  const halves = [...new Map(pas.map(p => [key(p), p])).entries()];
  const visible = pas.filter(p => key(p) === half);
  return <Sheet title="타석 선택" onClose={onClose}>
    <div className="sv-pa-controls"><label>이닝 <select aria-label="이닝과 초말 선택" value={half} onChange={e => setHalf(e.target.value)}>
      {halves.map(([k,p]) => <option key={k} value={k}>{p.inning}회 {halfText(p.half)}</option>)}
    </select></label><button className="sv-btn" onClick={onBox}>박스스코어</button></div>
    <ol className="sv-pa-list">{visible.map(pa => <li key={pa.pa_key}>
      <button className={`sv-pa-item ${pa.complete ? '' : 'open'}`} aria-current={pa.pa_key === selected ? 'true' : undefined} onClick={() => onPick(pa)}>
        <span className="sv-pa-who">{pa.batting_order ? `${pa.batting_order}번 · ` : ''}<b>{pa.batter}</b>{pa.pa_key === selected && <small> · 보는 중</small>}</span>
        <span className="sv-pa-result">{foldText && pa.complete ? foldText : pa.result} · {pa.pitch_count ?? (Number.isFinite(pa.first) && Number.isFinite(pa.last) ? Math.max(0, pa.last - pa.first + 1) : '–')}구</span>
      </button></li>)}</ol>
    {!visible.length && <p className="sv-muted">공개된 타석이 아직 없어요.</p>}
  </Sheet>;
}

const PROVENANCE: Record<Rec['provenance'], string> = {
  captured_live: '경기 중 투구 전에 기록한 추천',
  asof_replay: '경기 뒤, 그 공을 던지기 전 정보만으로 계산한 추천',
  absent: '기록된 추천 없음',
};
/** The one calm line for a pitch without a recorded recommendation: the server's sentence when it sends one (the pre-2026-10-11 stock
 *  sentence counts as none), else ours. */
export const ABSENT_PAST = '이 공은 기록된 추천이 없어요.';
const LEGACY_ABSENT = '이 투구의 저장된 추천이 없습니다.';
export const absentLine = (rec: Rec, fallback = ABSENT_PAST) => rec.reason && rec.reason.trim() !== LEGACY_ABSENT ? rec.reason : fallback;
export const provenanceText = (rec: Rec) => PROVENANCE[rec.provenance];

/** The ranked table of a recorded recommendation (UI 10/26 R1/P2): rank square + type, rough location, policy share, then the sentence. */
function RecTable({ rec }: { rec: Rec }) {
  return <><table className="sv-table sv-rec-table"><thead><tr><th scope="col">순위 · 구종</th><th scope="col">대략 위치</th><th scope="col">정책 선택 비중</th></tr></thead>
    <tbody>{rec.candidates.map((c: Candidate) => <tr key={c.rank}><td><RankSq rank={c.rank} /><span className="sr-only">{c.rank}순위 </span>{c.pitch_label}</td><td>{c.zone_label || '위치 없음'}</td><td>{policyShare(c.probability)}</td></tr>)}</tbody></table>
    <p className="sv-muted">정책 선택 비중은 성공·적중 확률이 아니에요. 위치는 과거 투구 분포의 대략 위치이며, 칸별 가치 지도가 아니에요.</p></>;
}
const recordedLine = (rec: Rec) => `${provenanceText(rec)}${rec.recorded_at ? ` · ${kstTime(rec.recorded_at)} 기록` : ''}`;

/** ⑧-1 R1: the next pitch's (or the screen's current) recommendation. Half shows the table. */
export function DetailSheet({ onClose, rec, recTitle, actual, actualTitle, paResult, notes, speed, extra, bounds, recNote = null, actualPending = false, actualCorrected = false, pair = null, absentText = ABSENT_PAST, actualFolded = null, actualExtra = null }:
  { onClose: () => void; rec: Rec | null; recTitle: string; actual: Actual | null; actualTitle: string; paResult: string | null; notes: string[]; speed: (mph: number | null) => string; extra?: ReactNode; bounds?: Bounds;
    /** live display state: the server's sentence (and any change note) for the pending pitch */
    recNote?: string | null; actualPending?: boolean; actualCorrected?: boolean;
    /** viewer delay, device not synced yet: the thrown pitch is not described */
    actualFolded?: string | null;
    /** an optional quiet line that belongs to the thrown pitch (never shown while the pitch is folded or being confirmed) */
    actualExtra?: ReactNode;
    /** past pitch with a recorded recommendation: its first pick and the pitch that was thrown, side by side in words (never a score) */
    pair?: { rec: string; actual: string } | null; absentText?: string }) {
  return <Sheet title={actual ? `투구 상세 · ${actualTitle}` : recTitle} onClose={onClose}>
    <section className="sv-detail rec">
      {actual && <h3>{recTitle}</h3>}
      {recNote && <p>{recNote}</p>}
      {recNote && (rec === null || rec.status !== 'ready') ? null : rec === null ? <p className="sv-muted">이 시점에는 추천이 없어요.</p>
        : rec.status !== 'ready' ? <p>{rec.status === 'unsupported' ? rec.reason ?? '추천을 지원하지 않는 상황이에요.' : absentLine(rec, absentText)}</p>
        : <>
          {pair && <dl className="sv-dl sv-pair"><dt>추천 1순위</dt><dd>{pair.rec}</dd><dt>실제 투구</dt><dd>{pair.actual}</dd></dl>}
          {pair && <p className="sv-muted">투구 전에 기록된 추천과 실제로 던진 공을 나란히 보여 드려요. 맞고 틀림을 매긴 것이 아니에요.</p>}
          <p className="sv-muted sv-recorded">{recordedLine(rec)}</p>
          <RecTable rec={rec} />
        </>}
    </section>
    <section className="sv-detail act">
      <h3>{actualTitle}</h3>
      {actualFolded ? <p className="sv-muted">{actualFolded}. 화면 위쪽의 ‘중계 맞추기’에서 내 중계에 맞추면 구종·위치·결과를 보여 드려요.</p>
      : actualPending ? <p className="sv-muted">이 공을 다시 확인하고 있어요. 확인되면 구종과 위치를 보여 드려요.</p> : <>
      {actualCorrected && <p className="sv-muted">이 공의 기록은 처음 받은 뒤에 정정됐어요.</p>}
      {actual && bounds && <div className="sv-detail-zone"><Zone fullWindow bounds={bounds} candidates={rec?.status === 'ready' ? rec.candidates : []} actual={actual.x !== null && actual.z !== null ? {x:actual.x,z:actual.z} : null} actualLabel="" setupX={actual.setup_x} actualKind={resultKind(actual.description)} /></div>}
      {actual === null ? <p className="sv-muted">아직 던지지 않았어요.</p> : <ActualFacts actual={actual} speed={speed} paResult={paResult} withType />}{actual !== null && actualExtra}</>}
    </section>
    {notes.length > 0 && <section className="sv-detail"><h3>자료 설명</h3><ul className="sv-notes">{notes.map(n => <li key={n}>{n}</li>)}</ul></section>}
    {extra && <section className="sv-detail sv-links">{extra}</section>}
  </Sheet>;
}

/** 판정 · 원본 좌표 · measured values that exist (missing ones in one line) · PA result · source text. */
function ActualFacts({ actual, speed, paResult, withType = false }: { actual: Actual; speed: (mph: number | null) => string; paResult: string | null; withType?: boolean }) {
  const t = actual.tracking ?? null;
  const have: [string, string][] = [], miss: string[] = [];
  const add = (label: string, v: number | null | undefined, unit: string) => { if (v == null) miss.push(label); else have.push([label, `${v}${unit}`]); };
  add('회전수', t?.spin_rate_rpm, ' rpm'); add('익스텐션', t?.extension_feet, ' ft'); add('타구 속도', t?.exit_velocity_mph, ' mph'); add('발사각', t?.launch_angle_deg, '°'); add('비거리', t?.distance_feet, ' ft');
  const last = miss[miss.length - 1] ?? '', batchim = last && (last.charCodeAt(last.length - 1) - 0xac00) % 28 !== 0;
  return <><dl className="sv-dl sv-facts">
    {withType && <><dt>구종</dt><dd>{actual.pitch_label}{actual.speed_mph !== null ? ` · ${speed(actual.speed_mph)}` : ''}</dd></>}
    <dt>판정</dt><dd>{[actual.result_label || '기록 없음', actual.zone_label].filter(Boolean).join(' · ')}</dd>
    <dt>원본 좌표</dt><dd>{actual.x !== null && actual.z !== null ? `가로 ${actual.x.toFixed(2)} ft · 높이 ${actual.z.toFixed(2)} ft (포수 기준)` : '위치 정보 없음'}</dd>
    {have.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}
    {paResult && <><dt>타석 결과</dt><dd>{paResult}</dd></>}
    {actual.play_text && <><dt>원본 기록</dt><dd lang="en">{actual.play_text}</dd></>}
  </dl>{miss.length > 0 && <p className="sv-muted sv-missing">{miss.join('·')}{batchim ? '은' : '는'} 이 공 자료에 없어요</p>}</>;
}

/** ⑧-2 P2: one thrown pitch. Title row with ‹ › (previous / next pitch). Half shows the result line and the small zone. */
export function PitchSheet({ pitch, bounds, onClose, onPrev, onNext, paResult, speed, folded = null, pending = false, corrected = false, actualExtra = null }:
  { pitch: PitchRow; bounds: Bounds; onClose: () => void; onPrev: (() => void) | null; onNext: (() => void) | null; paResult: string | null; speed: (mph: number | null) => string;
    folded?: string | null; pending?: boolean; corrected?: boolean; actualExtra?: ReactNode }) {
  const a = pitch.actual, rec = pitch.rec, kind = resultKind(a?.description);
  const head = <span className="sv-sheet-nav"><button onClick={onPrev ?? undefined} disabled={!onPrev} aria-label="이전 공">‹</button><button onClick={onNext ?? undefined} disabled={!onNext} aria-label="다음 공">›</button></span>;
  return <Sheet title={`${inningText(pitch.situation)} · ${pitch.pitch_number}구 · ${pitch.batter.name ?? '타자'}`} onClose={onClose} headExtra={head}>
    <section className="sv-detail act">
      {folded ? <p className="sv-muted">{folded}. 화면 위쪽의 ‘중계 맞추기’에서 내 중계에 맞추면 구종·위치·결과를 보여 드려요.</p>
      : pending ? <p className="sv-muted">이 공을 다시 확인하고 있어요. 확인되면 구종과 위치를 보여 드려요.</p>
      : !a ? <p className="sv-muted">실제 투구 기록 없음</p> : <>
        <p className="sv-pitch-line"><G1Dot kind={kind} n={pitch.pitch_number} ring /><b>{a.pitch_label}{a.speed_mph !== null ? ` ${speed(a.speed_mph)}` : ''}</b><span>{[a.result_label, a.zone_label].filter(Boolean).join(' · ')}</span></p>
        {corrected && <p className="sv-muted">이 공의 기록은 처음 받은 뒤에 정정됐어요.</p>}
        <div className="sv-detail-zone small"><Zone fullWindow bounds={bounds} candidates={rec.status === 'ready' ? rec.candidates : []} actual={a.x !== null && a.z !== null ? { x: a.x, z: a.z } : null} actualLabel=""
          setupX={a.setup_x} actualKind={kind} actualNumber={pitch.pitch_number} /></div>
        <ZoneLegend side={null} actualOnly={kind} />
        <ActualFacts actual={a} speed={speed} paResult={paResult} />{actualExtra}</>}
    </section>
    <section className="sv-detail rec">
      <h3>이 공을 던지기 전 추천</h3>
      {pending ? <p>이 공을 확인하는 동안에는 추천을 함께 보여 드리지 않아요.</p>
        : rec.status !== 'ready' ? <p>{rec.status === 'unsupported' ? rec.reason ?? '추천을 지원하지 않는 상황이에요.' : absentLine(rec)}</p>
        : <><p className="sv-muted sv-recorded">{recordedLine(rec)}</p><RecTable rec={rec} /></>}
    </section>
  </Sheet>;
}
