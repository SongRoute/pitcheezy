import { useState } from 'react';
import type { Bounds, Detail, Pa, PitchRow } from './model';
import { heldReceipt, inningText, settledCorrection, short } from './model';
import { Sheet } from './Sheets';
import { PLATE_HALF, U, place, uz, zoneWindow } from './Zone';

/** The record sheets follow the pitch on screen (DEMO-WS-GAPS-1028): `asOfIndex` = the selected pitch's index, null = the latest.
 *  What a file carries per pitch (the pitch list itself) is cut at that index; what it carries once for the whole game (lineups, the
 *  box score) is shown as published, labelled with its basis, never recomputed from events here. */
const upTo = (pitches: PitchRow[], asOfIndex: number | null) => asOfIndex === null ? pitches : pitches.filter(p => p.index <= asOfIndex);
const asOfWords = (detail: Detail, asOfIndex: number | null) => {
  const p = asOfIndex === null ? null : detail.pitches.find(x => x.index === asOfIndex);
  return p ? `${inningText(p.situation)} ${p.pitch_number}구` : null;
};
/** Plate appearances up to the selected pitch (all of them for the latest); the one that holds that pitch is still open unless the
 *  pitch ended it, and then counts only the pitches thrown so far. Used by the record sheet and the game-context panel alike. */
export function pasAsOf(detail: Detail, asOfIndex: number | null): Pa[] {
  if (asOfIndex === null) return detail.pas;
  const shown = upTo(detail.pitches, asOfIndex);
  // a plate appearance with no pitch at all (an intentional walk and the like) has no pitch index: it belongs before the selected pitch
  // when its at-bat number is earlier than the selected pitch's
  const atBat = detail.pitches.find(p => p.index === asOfIndex)?.at_bat_number ?? null;
  return detail.pas.filter(p => typeof p.first === 'number' ? p.first <= asOfIndex : atBat !== null && p.at_bat_number < atBat)
    .map(p => typeof p.last === 'number' && p.last > asOfIndex ? { ...p, complete: false, result: '타석 진행 중', pitch_count: shown.filter(x => x.pa_key === p.pa_key).length } : p);
}

export function RecordSheet({ detail, fold, onClose, onPick, asOfIndex = null }: { detail: Detail; fold: boolean; onClose: () => void; onPick: (index: number) => void; asOfIndex?: number | null }) {
  const [tab, setTab] = useState('문자 로그');
  const context = detail.current_context;
  const shown = upTo(detail.pitches, asOfIndex), asOf = asOfWords(detail, asOfIndex);
  const pas = pasAsOf(detail, asOfIndex);
  // one line for the corrections instead of a tag on most cards (G3 of the 2026 World Series: 278 of 294 pitches were re-sent by the
  // feed). `corrected` = every pitch the publisher marked corrected; `tagged` = those that carry the card tag (settled, no hold);
  // `held` = pitches shown as their last accepted body (a held pitch may be corrected as well, so the sets overlap).
  const corrected = shown.filter(p => p.receipt?.corrected).length, held = shown.filter(p => heldReceipt(p.receipt)).length;
  const tagged = shown.filter(p => settledCorrection(p.receipt)).length;
  return <Sheet title="경기 기록" onClose={onClose}>
    <div className="sv-row">{['문자 로그', '라인업', '박스', '분석'].map(t => <button className="sv-btn" key={t} aria-pressed={tab === t} onClick={() => setTab(t)}>{t}</button>)}</div>
    {tab === '문자 로그' && <>{asOf && <p className="sv-muted sv-basis">고른 공({asOf})까지의 타석이에요. 마지막 공으로 가면 경기 전체가 보여요.</p>}
      <ol className="sv-recent-pas">{[...pas].reverse().map(p => <li key={p.pa_key}><button onClick={() => onPick(p.first)} disabled={!shown.some(x => x.index === p.first)}><b>{inningText(p)} · {p.batter}</b><span>{fold ? '중계를 맞춘 뒤 표시' : p.result} · {p.pitch_count ?? shown.filter(x => x.pa_key === p.pa_key).length}구</span></button></li>)}</ol></>}
    {tab === '라인업' && <>{asOf && <p className="sv-muted sv-basis">발행본의 명단은 경기 종료 시점 기준이에요(교체 포함). 고른 공 시점의 명단은 따로 없어요.</p>}
      {(['away', 'home'] as const).map(side => <section key={side}><h3>{short(detail.game[`${side}_team`])}</h3>{context?.lineups?.[side]?.length ? <ol>{context.lineups[side].map(p => <li key={`${p.order}:${p.id}`}>{p.order}번 · {p.name ?? '선수 미확인'}</li>)}</ol> : <p className="sv-muted">라인업 정보 없음</p>}</section>)}</>}
    {tab === '박스' && <>{asOf && <p className="sv-muted sv-basis">발행본의 박스는 경기 종료 시점 기준이에요. 고른 공 시점의 선수별 합계는 따로 없어요(선수 칸의 ‘오늘 기록’이 그 시점 값이에요).</p>}
      {(['away', 'home'] as const).map(side => <section key={side}><h3>{short(detail.game[`${side}_team`])}</h3>{context?.boxscore?.[side] ? <>
      <table className="sv-table"><thead><tr><th>타자</th><th>타수</th><th>안타</th><th>출루</th></tr></thead><tbody>{context.boxscore[side].batters.map(p => <tr key={p.id}><td>{p.name}</td><td>{p.at_bats ?? '—'}</td><td>{p.hits ?? '—'}</td><td>{p.on_base ?? '—'}</td></tr>)}</tbody></table>
      <table className="sv-table"><thead><tr><th>투수</th><th>투구</th><th>삼진</th><th>자책</th></tr></thead><tbody>{context.boxscore[side].pitchers.map(p => <tr key={p.id}><td>{p.name}</td><td>{p.pitches ?? '—'}</td><td>{p.strikeouts ?? '—'}</td><td>{p.earned_runs ?? '—'}</td></tr>)}</tbody></table></> : <p className="sv-muted">박스 기록 정보 없음</p>}</section>)}</>}
    {tab === '분석' && <div className="sv-unmeasured"><h3>분석 <span className="sv-tag">준비 중</span></h3><p>성분별 책임 분해·기대실점·타구 예상 결과는 아직 연결되지 않았어요.</p></div>}
    <p className="sv-muted">공개된 시점까지의 기록이며 이후 정정될 수 있어요. 출루는 안타·볼넷·몸에 맞는 공을 합한 값이에요.</p>
    {(corrected > 0 || held > 0) && <p className="sv-muted sv-corrections">{asOf ? '고른 공까지 ' : ''}정정된 기록 {corrected}구(투구 카드에 ‘정정됨’ 표시 {tagged}구{corrected > tagged ? `, 보류 중이라 표시하지 않은 ${corrected - tagged}구` : ''}){held ? ` · 공식 피드가 다시 보낸 내용을 보류한 공 ${held}구(마지막으로 받아들인 기록을 보여 드려요; 정정된 공과 겹칠 수 있어요)` : ''}</p>}
  </Sheet>;
}

/** Where today's pitches went: one dot per received pitch with a plate location, pitcher's view, the same window geometry as the main
 *  zone (Zone.tsx). Only `actual.x` / `actual.z` of the published file are drawn; a pitch beyond the window is a hollow marker at the
 *  edge, never a moved dot. Colour = pitch type (the legend names each type with its count). */
/** One fixed colour per Statcast pitch-type code, so a type keeps its colour whichever pitch or pitcher is on screen; a code not listed
 *  takes the next spare colour in legend order (fastballs warm, breaking balls cool, off-speed earth tones). */
const TYPE_COLOURS: Record<string, string> = { FF: '#b74710', SI: '#d9622b', FC: '#c33349', FA: '#b74710', SL: '#3559b5', ST: '#6b2fa0', SV: '#4a5ad0', CU: '#1e6a4e',
  KC: '#0d7f8c', CS: '#188652', CH: '#8a5a12', FS: '#a07b2a', FO: '#8a5a12', KN: '#7d8e95', EP: '#7d8e95', SC: '#7d8e95' };
const SPARE_COLOURS = ['#5c4300', '#2f6f8f', '#8e2335', '#7d8e95'];
export const scatterColour = (type: string, spare: number) => TYPE_COLOURS[type] ?? SPARE_COLOURS[Math.min(spare, SPARE_COLOURS.length - 1)];
export function PitchScatter({ bounds, pitches, groups }: { bounds: Bounds; pitches: PitchRow[]; groups: { type: string; label: string; n: number }[] }) {
  const w = zoneWindow(bounds, false, false, true);
  const left = -PLATE_HALF * U, right = PLATE_HALF * U, top = uz(bounds.top), bottom = uz(bounds.bottom);
  let spare = 0;
  const colour = new Map(groups.map(g => [g.type, scatterColour(g.type, g.type in TYPE_COLOURS ? 0 : spare++)]));
  const dots = pitches.flatMap(p => p.actual?.x == null || p.actual.z == null || !p.actual.pitch_type ? [] : [{ p, at: place({ x: p.actual.x, z: p.actual.z }, w) }]);
  const out = dots.filter(d => d.at.out).length;
  const r = 7, inset = r + 2;
  return <><div className="sv-mix-zone">
    <svg className="sv-zone sv-mix-scatter" viewBox={`${w.vb.x} ${w.vb.y} ${w.vb.w} ${w.vb.h}`} preserveAspectRatio="xMidYMid meet" role="img"
      aria-label={`오늘 던진 위치, 투수 시점, ${dots.length}구${out ? ` (화면 밖 ${out}구는 가장자리에 빈 동그라미)` : ''}: ${groups.map(g => `${g.label} ${g.n}구`).join(', ')}`}>
      <rect x={-w.xMax * U} y={uz(w.zHi)} width={2 * w.xMax * U} height={(w.zHi - w.zLo) * U} className="sv-zone-outside" />
      <rect x={left} y={top} width={right - left} height={bottom - top} className="sv-zone-box" />
      {[1, 2].map(i => <line key={`v${i}`} x1={left + (right - left) * i / 3} x2={left + (right - left) * i / 3} y1={top} y2={bottom} className="sv-zone-grid" />)}
      {[1, 2].map(i => <line key={`h${i}`} y1={top + (bottom - top) * i / 3} y2={top + (bottom - top) * i / 3} x1={left} x2={right} className="sv-zone-grid" />)}
      {dots.map(({ p, at }) => { const x = at.out ? at.x - at.dx * inset : at.x, y = at.out ? at.y + at.dz * inset : at.y;
        return <g key={p.key} className={`sv-mix-dot${at.out ? ' out' : ''}`} data-type={p.actual!.pitch_type!} data-x={p.actual!.x!} data-z={p.actual!.z!}>
          <circle cx={x} cy={y} r={r} style={{ fill: at.out ? 'none' : colour.get(p.actual!.pitch_type!), stroke: at.out ? colour.get(p.actual!.pitch_type!) : '#fff' }} />
          <title>{inningText(p.situation)} {p.pitch_number}구 · {p.actual!.pitch_label}{p.actual!.speed_mph !== null ? ` ${Math.round(p.actual!.speed_mph * 1.609344)} km/h` : ''} · {p.actual!.result_label || '기록 없음'}{at.out ? ' · 화면 밖' : ''}</title>
        </g>; })}
    </svg>
  </div>
  <p className="sv-mix-legend" aria-hidden="true">{groups.map(g => <span key={g.type}><i style={{ background: colour.get(g.type) }} />{g.label} {g.n}</span>)}<span className="sv-mix-note">투수 시점 · 회색은 존 밖{out ? ` · 빈 동그라미는 화면 밖 ${out}구` : ''}</span></p>
  </>;
}

export function PitchMixSheet({ detail, pitcherId, fold, onClose, asOfIndex = null }: { detail: Detail; pitcherId: number | null; fold: boolean; onClose: () => void; asOfIndex?: number | null }) {
  const asOf = asOfWords(detail, asOfIndex);
  const rows = fold ? [] : upTo(detail.pitches, asOfIndex).filter(p => p.pitcher.id === pitcherId && p.receipt?.status !== 'unconfirmed' && p.actual?.pitch_type);
  const groups = [...new Set(rows.map(p => p.actual!.pitch_type))].map(type => { const pitches = rows.filter(p => p.actual!.pitch_type === type); const speeds = pitches.flatMap(p => p.actual!.speed_mph === null ? [] : [p.actual!.speed_mph]); return { type: type!, label: pitches[0].actual!.pitch_label, n: pitches.length, speed: speeds.length ? speeds.reduce((a,b) => a+b, 0) / speeds.length * 1.609344 : null }; })
    .sort((a, b) => b.n - a.n);
  const located = rows.filter(p => p.actual!.x !== null && p.actual!.z !== null);
  return <Sheet title="구종 · 수신된 투구 기록" onClose={onClose}>
    {rows.length ? <><p>선택 투수의 수신된 기록 중 구종이 확인된 {rows.length}구 기준{asOf ? ` · 고른 공(${asOf})까지` : ''}</p><table className="sv-table"><thead><tr><th>구종</th><th>투구 수</th><th>관측 비율</th><th>평균 구속</th></tr></thead><tbody>{groups.map(g => <tr key={g.type}><td>{g.label}</td><td>{g.n}</td><td>{Math.round(g.n / rows.length * 100)}%</td><td>{g.speed === null ? '—' : `${Math.round(g.speed)} km/h`}</td></tr>)}</tbody></table><p className="sv-muted">경기 전체 투구가 모두 수신됐다는 뜻은 아니에요. 구속이 없는 공은 평균에서 제외해요.</p>
      <h3 className="sv-mix-title">오늘 던진 위치 <small>{located.length}구</small></h3>
      {located.length ? <PitchScatter bounds={detail.zone_bounds} pitches={located} groups={groups} /> : <p className="sv-muted">위치가 기록된 공이 없어요.</p>}
      <p className="sv-muted">발행된 투구의 포수 기준 좌표를 그대로 찍은 그림이에요(위치는 평가된 값이 아니에요).</p></> : <p className="sv-muted">{fold ? '중계를 맞춘 뒤 표시해요.' : '확인된 구종 기록이 없어요.'}</p>}
    <p className="sv-muted">시즌별 사용률·헛스윙률·피xwOBA는 아직 연결되지 않았어요.</p>
  </Sheet>;
}
