import { useState } from 'react';
import type { ReactNode } from 'react';
import type { Detail, Pa, PitchRow } from './model';
import { heldReceipt, inningText, settledCorrection, short, surname } from './model';
import Zone from './Zone';
import { Sheet } from './Sheets';

/** The record sheets follow the pitch on screen (DEMO-WS-GAPS-1028): `asOfIndex` = the selected pitch's index, null = the latest.
 *  What a file carries per pitch (the pitch list itself) is cut at that index; what it carries once for the whole game (lineups, the
 *  box score) is shown as published, labelled with its basis, never recomputed from events here. */
const upTo = (pitches: PitchRow[], asOfIndex: number | null) => asOfIndex === null ? pitches : pitches.filter(p => p.index <= asOfIndex);
const asOfWords = (detail: Detail, asOfIndex: number | null) => {
  const p = asOfIndex === null ? null : detail.pitches.find(x => x.index === asOfIndex);
  return p ? `${inningText(p.situation)} ${p.pitch_number}구` : null;
};
/** Plate appearances up to the selected pitch (all of them for the latest); the one that holds that pitch is still open unless the
 *  pitch ended it, and then counts only the pitches thrown so far. */
export function pasAsOf(detail: Detail, asOfIndex: number | null): Pa[] {
  if (asOfIndex === null) return detail.pas;
  const shown = upTo(detail.pitches, asOfIndex);
  // a plate appearance with no pitch at all (an intentional walk and the like) has no pitch index: it belongs before the selected pitch
  // when its at-bat number is earlier than the selected pitch's
  const atBat = detail.pitches.find(p => p.index === asOfIndex)?.at_bat_number ?? null;
  return detail.pas.filter(p => typeof p.first === 'number' ? p.first <= asOfIndex : atBat !== null && p.at_bat_number < atBat)
    .map(p => typeof p.last === 'number' && p.last > asOfIndex ? { ...p, complete: false, result: '타석 진행 중', pitch_count: shown.filter(x => x.pa_key === p.pa_key).length } : p);
}

export function RecordSheet({ detail, fold, onClose, onPick, headExtra, asOfIndex = null }: { detail: Detail; fold: boolean; onClose: () => void; onPick: (index: number) => void; asOfIndex?: number | null; headExtra?: ReactNode }) {
  const [tab, setTab] = useState('문자 로그');
  const context = detail.current_context;
  const shown = upTo(detail.pitches, asOfIndex), asOf = asOfWords(detail, asOfIndex);
  const pas = pasAsOf(detail, asOfIndex);
  // one line for the corrections instead of a tag on most cards (G3 of the 2026 World Series: 278 of 294 pitches were re-sent by the
  // feed). `corrected` = every pitch the publisher marked corrected; `tagged` = those that carry the card tag (settled, no hold);
  // `held` = pitches shown as their last accepted body (a held pitch may be corrected as well, so the sets overlap).
  const corrected = shown.filter(p => p.receipt?.corrected).length, held = shown.filter(p => heldReceipt(p.receipt)).length;
  const tagged = shown.filter(p => settledCorrection(p.receipt)).length;
  return <Sheet title="경기 기록" onClose={onClose} headExtra={headExtra}>
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
/** Today's pitch mix of one pitcher from the received records only (unconfirmed rows and rows without a type are left out). */
function pitchMix(detail: Detail, pitcherId: number | null, fold: boolean, asOfIndex: number | null = null) {
  const rows = fold ? [] : upTo(detail.pitches, asOfIndex).filter(p => p.pitcher.id === pitcherId && p.receipt?.status !== 'unconfirmed' && p.actual?.pitch_type);
  const groups = [...new Set(rows.map(p => p.actual!.pitch_type))].map(type => { const pitches = rows.filter(p => p.actual!.pitch_type === type); const speeds = pitches.flatMap(p => p.actual!.speed_mph === null ? [] : [p.actual!.speed_mph]); return { type: type!, label: pitches[0].actual!.pitch_label, n: pitches.length, speed: speeds.length ? speeds.reduce((a,b) => a+b, 0) / speeds.length * 1.609344 : null }; })
    .sort((a, b) => b.n - a.n);
  return { rows, groups };
}
/** 구종 / 투구 수 / 비율 / 평균 구속 — shared by the 구종 sheet and the player sheet's 투수 tab. */
export function MixTable({ detail, pitcherId, fold, heading = false, asOfIndex = null }: { detail: Detail; pitcherId: number | null; fold: boolean; heading?: boolean; asOfIndex?: number | null }) {
  const { rows, groups } = pitchMix(detail, pitcherId, fold, asOfIndex);
  return <>{heading && <h3 className="sv-sheet-h3">오늘 구종 · {rows.length}구</h3>}
    {rows.length ? <table className="sv-table sv-mix-table"><thead><tr><th scope="col">구종</th><th scope="col">투구 수</th><th scope="col">비율</th><th scope="col">평균 구속</th></tr></thead>
      <tbody>{groups.map(g => <tr key={g.type}><td>{g.label}</td><td>{g.n}</td><td>{Math.round(g.n / rows.length * 100)}%</td><td>{g.speed === null ? '—' : `${Math.round(g.speed)} km/h`}</td></tr>)}</tbody></table>
      : <p className="sv-muted">{fold ? '중계를 맞춘 뒤 표시해요.' : '확인된 구종 기록이 없어요.'}</p>}</>;
}
/** First letter of each type for the location map; two letters where two of this pitcher's types share one (커브 / 커터). */
const initials = (labels: string[]) => Object.fromEntries(labels.map(l => [l, labels.some(o => o !== l && o[0] === l[0]) ? l.slice(0, 2) : l[0]]));

/** ⑧-4 G3: today's mix (half = the table), then where today's pitches went, one letter per type. */
export function PitchMixSheet({ detail, pitcherId, pitcherName, fold, onClose, asOfIndex = null }: { detail: Detail; pitcherId: number | null; pitcherName: string | null; fold: boolean; onClose: () => void; asOfIndex?: number | null }) {
  const { rows, groups } = pitchMix(detail, pitcherId, fold, asOfIndex);
  const asOf = asOfWords(detail, asOfIndex);
  const placed = rows.filter(p => p.actual!.x !== null && p.actual!.z !== null);
  const abbr = initials(groups.map(g => g.label));
  return <Sheet title={`${pitcherName ? surname(pitcherName) : '투수'} 오늘 구종`} onClose={onClose} half="max(56dvh, min(420px, calc(100dvh - 56px)))">
    {rows.length > 0 && <p className="sv-muted">받은 기록 중 구종이 확인된 {rows.length}구 기준{asOf ? ` · 고른 공(${asOf})까지` : ''}</p>}
    <MixTable detail={detail} pitcherId={pitcherId} fold={fold} asOfIndex={asOfIndex} />
    {/* half = the table only (design 25); the map and the notes appear when the sheet is opened to full (design 26) */}
    {placed.length > 0 && <section className="sv-mix-map sv-full-only"><h3 className="sv-sheet-h3">오늘 던진 위치</h3>
      <div className="sv-detail-zone small"><Zone fullWindow bounds={detail.zone_bounds} candidates={[]} actual={null} actualLabel="" setupX={null}
        actuals={placed.map(p => ({ x: p.actual!.x!, z: p.actual!.z!, number: p.pitch_number, label: abbr[p.actual!.pitch_label] }))} /></div>
      <p className="sv-mix-key">{groups.map(g => `${abbr[g.label]}=${g.label}`).join(' · ')}</p></section>}
    {rows.length > 0 && <p className="sv-muted sv-full-only">경기 전체 투구가 모두 수신됐다는 뜻은 아니에요. 구속이 없는 공은 평균에서 제외해요.{placed.length < rows.length ? ` 위치 정보가 없는 ${rows.length - placed.length}구는 그림에서 뺐어요.` : ''}</p>}
    <p className="sv-muted sv-full-only">시즌별 사용률·헛스윙률·피xwOBA는 아직 연결되지 않았어요.</p>
  </Sheet>;
}
