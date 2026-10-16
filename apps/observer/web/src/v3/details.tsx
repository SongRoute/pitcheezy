import { useMemo, useState } from 'react';
import type { Detail, Pa, PitchRow, Rec } from '../service/model';
import { inningText, kstTime, short } from '../service/model';
import type { Options } from './options';
import type { GameView } from './view';
import { unconfirmed } from './view';
import Zone from './Zone';
import type { Open } from './parts';
import { zoneLabel, zonePitches } from './parts';
import { candidateWords, pointWords } from './zones';
import { callKind, callText, dash, era, handText, shareText, speedNum, stat3, toggleUnit, unitText, useUnit } from './format';

/** Fixed explanations (classic Sheets.tsx LIMITS, same words) — shown only on demand. */
export const LIMITS = [
  '추천은 검증 전 실험 버전이에요. 2026 정규시즌 1회 평가에서는 추천이 실제 선택보다 낫다는 근거를 확인하지 못했어요.',
  '‘정책 선택 비중’은 추천 정책이 이 상황에서 그 구종을 고르는 비중이에요. 성공·적중 확률이 아니에요.',
  '추천 위치는 그 투수가 실제로 던진 공의 분포(2025년까지)로 잡은 대략 위치이며, 위치 자체는 평가되지 않았어요.',
  '지난 경기의 추천은 경기 뒤 그 공을 던지기 전 정보만으로 계산한 기록이에요. 경기 중 실시간으로 보여 준 추천이 아니에요.',
  '포수 미트 표시는 중계 영상에서 읽은 가로 위치 추정이며, 높이는 다루지 않아요. 실제 투구 뒤에만 보여요.',
  '날짜와 시간은 모두 한국 시간이에요. 경기 날짜는 원래 예정 시작 시각 기준이에요.',
];
const NA = '—';
const provenanceText = (r: Rec) => r.provenance === 'captured_live' ? '투구 전에 기록' : r.provenance === 'asof_replay' ? '경기 뒤 같은 정보로 다시 계산' : '기록 없음';
const situText = (s: { inning: number; half: 'Top' | 'Bot'; outs: number; bases: number }) => {
  const r = [1, 2, 3].filter(b => (s.bases & (1 << (b - 1))) !== 0);
  return `${inningText(s)} · ${s.outs}아웃 · ${r.length ? `${r.join('·')}루` : '주자 없음'}`;
};

// ───────── recommendation ─────────
export function RecDetail({ v, opts }: { v: GameView; opts: Options }) {
  const rec = v.shownRec ? v.rec : null;
  const b = v.detail.zone_bounds;
  const s = v.situation;
  return <div className="v3-detail">
    <div className="v3-ctx">
      <b>{s ? `${situText(s)} · ${s.balls}-${s.strikes}` : v.breakNow ? '공수 교대 중' : '상황 —'}</b>
      <span>타자 {v.batter?.name ?? NA} {handText(v.batter?.side, 'b')} · 투수 {v.pitcher?.name ?? NA} {handText(v.pitcher?.hand, 'p')}</span>
      {rec && <span className="v3-muted">{v.nextPhase && v.nextNumber ? `${v.nextNumber}구 ` : ''}{provenanceText(rec)}{rec.recorded_at ? ` · ${kstTime(rec.recorded_at)}` : ''}{rec.policy ? ` · 정책 ${rec.policy}` : ''}</span>}
    </div>
    {!rec ? <p className="v3-empty-line">{v.eff?.message ?? (v.rec?.status === 'unsupported' ? v.rec.reason ?? '이 상황은 추천 대상이 아니에요.' : '지금 보여 드릴 추천이 없어요.')}</p> : <>
      <div className="v3-detail-zone"><Zone bounds={b} view={opts.view} mark={opts.zone} batter={v.side} recs={rec.candidates} pitches={zonePitches(v.paPitches)}
        label={zoneLabel(v, rec.candidates, v.paPitches, opts.view)} size="detail" /></div>
      <table className="v3-table">
        <thead><tr><th scope="col">순위</th><th scope="col">구종 · 위치</th><th scope="col" className="num">구종 선택 비중</th></tr></thead>
        <tbody>{rec.candidates.map(c => <tr key={c.rank}><td><i className={`v3-rank r${c.rank}`}>{c.rank}</i></td>
          <td><b>{c.pitch_label}</b> <small className="v3-code">{c.pitch_type}</small><br /><span className="v3-muted">{candidateWords(c, v.side, b)}</span></td>
          <td className="num"><span className="v3-bar"><i style={{ width: `${Math.round((c.probability ?? 0) * 100)}%` }} /></span>{shareText(c.probability)}</td></tr>)}</tbody>
      </table>
    </>}
    <div className="v3-explain">
      <h3>숫자와 위치의 뜻</h3>
      <p>순위와 비중은 추천 정책(ARM-B)이 이 상황에서 각 <b>구종</b>을 고르는 비중이에요. 성공·스트라이크 확률이 아니에요.</p>
      <p>위치는 그 투수가 실제로 던진 공의 분포로 잡은 대략적인 칸이에요. 위치 자체는 평가되지 않았어요.</p>
    </div>
    <div className="v3-explain"><h3>추천 이유</h3><p className="v3-muted">{NA} 모델이 근거를 함께 내보내는 경우에만 보여 드려요. 근거가 없으면 설명을 지어 넣지 않아요.</p></div>
  </div>;
}

// ───────── one pitch ─────────
export function PitchDetail({ v, opts, index, open }: { v: GameView; opts: Options; index: number; open: Open }) {
  const u = useUnit();
  const d = v.detail;
  const p = d.pitches.find(x => x.index === index) ?? null;
  if (!p) return <p className="v3-empty-line">이 공의 기록을 찾지 못했어요.</p>;
  const pa = d.pitches.filter(x => x.pa_key === p.pa_key);
  const upto = pa.filter(x => x.index <= p.index);
  const pending = unconfirmed(p);
  const a = pending ? null : p.actual;
  const rec = pending ? null : p.rec;
  const ready = rec?.status === 'ready' && rec.candidates.length > 0;
  const top = ready ? rec!.candidates[0] : null;
  const side = p.batter.side === 'R' || p.batter.side === 'L' ? p.batter.side as 'L' | 'R' : null;
  const b = d.zone_bounds;
  const nextInPa = pa.find(x => x.index > p.index);
  const after = nextInPa ? `${nextInPa.situation.balls}-${nextInPa.situation.strikes}` : a?.event ? '타석 종료' : NA;
  const at = d.pitches.indexOf(p);
  const prev = d.pitches[at - 1], next = d.pitches[at + 1];
  const paResult = a?.event ? d.pas.find(x => x.pa_key === p.pa_key && x.complete)?.result ?? callText(p) : null;
  const zv = { detail: d, side };
  return <div className="v3-detail">
    <div className="v3-pitch-nav">
      <button className="v3-btn ghost" disabled={!prev} onClick={() => prev && open('pitch', { pitch: prev.index })} aria-label="이전 공">‹</button>
      <div><b>{p.batter.name ?? '타자'} · {p.pitch_number}구</b><span className="v3-muted">{situText(p.situation)}</span></div>
      <button className="v3-btn ghost" disabled={!next} onClick={() => next && open('pitch', { pitch: next.index })} aria-label="다음 공">›</button>
    </div>
    <ol className="v3-pa-strip" aria-label="이 타석의 공">{pa.map(x => <li key={x.index}><button aria-pressed={x.index === p.index} onClick={() => open('pitch', { pitch: x.index })}>{x.pitch_number}</button></li>)}</ol>
    <div className="v3-count-move"><span>{p.situation.balls}-{p.situation.strikes}</span><i aria-hidden="true">→</i><span>{after}</span>
      <b className={`v3-call ${pending ? 'none' : callKind(a)}`}>{pending ? '확인 중' : paResult ?? callText(p)}</b></div>
    <div className="v3-detail-zone"><Zone bounds={b} view={opts.view} mark={opts.zone} batter={side} recs={ready ? rec!.candidates : []} pitches={zonePitches(upto)}
      label={zoneLabel(zv, ready ? rec!.candidates : [], upto, opts.view)} size="detail" /></div>
    {pending ? <p className="v3-empty-line">이 공을 다시 확인하고 있어요. 확인되는 동안에는 추천과 함께 보여 드리지 않아요.</p> : <>
      <table className="v3-table compare">
        <thead><tr><th scope="col" /><th scope="col">투구 전 추천 1순위</th><th scope="col">실제</th></tr></thead>
        <tbody>
          <tr><th scope="row">구종</th><td>{top ? top.pitch_label : rec?.status === 'unsupported' ? '추천 대상 아님' : '기록 없음'}</td><td><b>{a?.pitch_label ?? NA}</b></td></tr>
          <tr><th scope="row">위치</th><td>{top ? candidateWords(top, side, b) : NA}</td><td><b>{pointWords(a?.x ?? null, a?.z ?? null, b, side)}</b></td></tr>
          <tr><th scope="row">판정</th><td>{NA}</td><td><b>{callText(p)}</b></td></tr>
        </tbody>
      </table>
      {!ready && <p className="v3-muted small">{rec?.reason ?? '이 공 전에는 추천이 기록되지 않았어요. 추천은 투구 전에 기록된 것만 보여 드려요.'}</p>}
      {ready && rec!.candidates.length > 1 && <p className="v3-muted small">다른 후보: {rec!.candidates.slice(1).map(c => `${c.rank} ${c.pitch_label} · ${candidateWords(c, side, b)}`).join(', ')}</p>}
      <div className="v3-tiles">
        <div><small>구속</small><b>{speedNum(a?.speed_mph, u)}</b><small>{unitText(u)}</small></div>
        <div><small>회전수</small><b className="na">{NA}</b></div>
        <div><small>익스텐션</small><b className="na">{NA}</b></div>
        <div><small>무브먼트</small><b className="na">{NA}</b></div>
      </div>
      <div className="v3-explain"><h3>타구</h3><p className={a?.play_text ? '' : 'v3-muted'}>{a?.play_text ?? (a?.description === 'hit_into_play' ? NA : '타구 없음')}</p>
        {a?.event && <p className="v3-muted small">타구 속도·발사각 {NA}</p>}</div>
      <p className="v3-muted small">{p.receipt?.corrected ? '기록이 정정된 공이에요. ' : ''}{rec ? `추천: ${provenanceText(rec)}` : ''}</p>
    </>}
  </div>;
}

// ───────── players ─────────
type Today = { label: string; value: string }[];
function todayFor(d: Detail, who: 'batter' | 'pitcher', id: number | null, ctxPitches: number | null): Today {
  const done = d.pas.filter(p => p.complete);
  const ev = (pa: Pa) => d.pitches.find(x => x.index === pa.last)?.actual?.event ?? '';
  const hit = ['single', 'double', 'triple', 'home_run'], walk = ['walk', 'intent_walk', 'hit_by_pitch'], notAb = [...walk, 'sac_fly', 'sac_bunt', 'catcher_interf'];
  if (who === 'batter') {
    const mine = id === null ? [] : done.filter(pa => d.pitches.find(x => x.index === pa.first)?.batter.id === id);
    return [{ label: '타수', value: String(mine.filter(pa => !notAb.includes(ev(pa))).length) }, { label: '안타', value: String(mine.filter(pa => hit.includes(ev(pa))).length) },
      { label: '홈런', value: String(mine.filter(pa => ev(pa) === 'home_run').length) }, { label: '볼넷', value: String(mine.filter(pa => walk.includes(ev(pa))).length) },
      { label: '삼진', value: String(mine.filter(pa => /strikeout/.test(ev(pa))).length) }];
  }
  const mine = id === null ? [] : done.filter(pa => d.pitches.find(x => x.index === pa.first)?.pitcher.id === id);
  return [{ label: '투구 수', value: dash(ctxPitches ?? (id === null ? null : d.pitches.filter(x => x.pitcher.id === id).length)) }, { label: '상대 타자', value: String(mine.length) },
    { label: '피안타', value: String(mine.filter(pa => hit.includes(ev(pa))).length) }, { label: '볼넷', value: String(mine.filter(pa => walk.includes(ev(pa))).length) },
    { label: '삼진', value: String(mine.filter(pa => /strikeout/.test(ev(pa))).length) }, { label: '실점', value: NA }];
}
const DEFS: Record<string, string> = { 타율: '안타 ÷ 타수', 출루율: '(안타+볼넷+사구) ÷ (타수+볼넷+사구+희생플라이)', 장타율: '루타 ÷ 타수', OPS: '출루율 + 장타율',
  'wRC+': '리그·구장 보정 득점 창출력. 100 = 리그 평균', ERA: '9이닝당 자책점', WHIP: '이닝당 (볼넷+피안타)', FIP: '삼진·볼넷·사구·홈런만으로 본 평균자책', 탈삼진율: '삼진 ÷ 상대 타자', 볼넷율: '볼넷 ÷ 상대 타자' };
export function PlayerDetail({ v, layout, initial }: { v: GameView; layout: Options['stats']; initial: 'batter' | 'pitcher' }) {
  const [who, setWho] = useState(initial);
  const [defs, setDefs] = useState(false);
  const person = who === 'batter' ? v.batter : v.pitcher;
  const info = person ? v.detail.players?.[String(person.id)] : undefined;
  const asOf = info?.stats_as_of ?? null;
  // Periods side by side (spec §5). The published file carries only pre-game season AVG/OPS or ERA; every other cell is "—".
  const rows: { k: string; reg: string; post: string }[] = who === 'batter'
    ? [{ k: '타석', reg: NA, post: NA }, { k: '타율', reg: stat3(info?.avg), post: NA }, { k: '출루율', reg: NA, post: NA }, { k: '장타율', reg: NA, post: NA }, { k: 'OPS', reg: stat3(info?.ops), post: NA }, { k: 'wRC+', reg: NA, post: NA }]
    : [{ k: '투구 이닝', reg: NA, post: NA }, { k: 'ERA', reg: era(info?.era), post: NA }, { k: 'WHIP', reg: NA, post: NA }, { k: 'FIP', reg: NA, post: NA }, { k: '탈삼진율', reg: NA, post: NA }, { k: '볼넷율', reg: NA, post: NA }];
  const today = todayFor(v.detail, who, person?.id ?? null, who === 'pitcher' ? v.pitcherToday?.pitches ?? null : null);
  const g = v.detail.game;
  const top = (v.situation ?? v.lastPitch?.situation)?.half === 'Top';
  const team = who === 'batter' ? (top ? g.away_team : g.home_team) : (top ? g.home_team : g.away_team);
  return <div className="v3-detail">
    <div className="v3-seg" role="group" aria-label="선수 선택">
      {(['batter', 'pitcher'] as const).map(w => <button key={w} aria-pressed={who === w} onClick={() => setWho(w)}>{w === 'batter' ? '타자' : '투수'} · {(w === 'batter' ? v.batter : v.pitcher)?.name ?? NA}</button>)}
    </div>
    <div className="v3-person"><b>{person?.name ?? '선수 미확인'}</b><span>{short(team)} · {handText(who === 'batter' ? v.batter?.side : v.pitcher?.hand, who === 'batter' ? 'b' : 'p')}{info?.jersey ? ` · #${info.jersey}` : ''}</span></div>
    {layout === 'table' ? <table className="v3-table stats">
      <thead><tr><th scope="col">지표</th><th scope="col" className="num">정규시즌</th><th scope="col" className="num">포스트시즌</th></tr></thead>
      <tbody>{rows.map(r => <tr key={r.k}><th scope="row">{r.k}</th><td className={`num${r.reg === NA ? ' na' : ''}`}>{r.reg}</td><td className={`num${r.post === NA ? ' na' : ''}`}>{r.post}</td></tr>)}</tbody>
    </table> : <div className="v3-period-cards">
      {(['reg', 'post'] as const).map(k => <div key={k} className={`v3-period ${k}`}><h3>{k === 'reg' ? '정규시즌' : '포스트시즌'}</h3><small>{k === 'reg' ? (asOf ? `${asOf} 경기 전 기준` : '기준일 —') : '표본 —'}</small>
        <dl>{rows.map(r => <div key={r.k}><dt>{r.k}</dt><dd className={r[k] === NA ? 'na' : ''}>{r[k]}</dd></div>)}</dl></div>)}
    </div>}
    <p className="v3-muted small">{asOf ? `정규시즌 = ${asOf} 경기 전 시즌 기록.` : '시즌 기록 기준일 미확인.'} 발행 파일에 없는 값은 —로 표시해요.</p>
    <button className="v3-link" onClick={() => setDefs(x => !x)} aria-expanded={defs}>{defs ? '지표 설명 접기' : '지표 설명 보기'}</button>
    {defs && <dl className="v3-defs">{rows.filter(r => DEFS[r.k]).map(r => <div key={r.k}><dt>{r.k}</dt><dd>{DEFS[r.k]}</dd></div>)}</dl>}
    <h3 className="v3-h3">오늘</h3>
    <div className="v3-tiles">{today.map(t => <div key={t.label}><small>{t.label}</small><b className={t.value === NA ? 'na' : ''}>{t.value}</b></div>)}</div>
    <p className="v3-muted small">오늘 기록은 지금까지 발행된 플레이로 센 값이에요. 이후 기록 정정이 반영될 수 있어요.</p>
  </div>;
}

// ───────── arsenal ─────────
export function ArsenalDetail({ v }: { v: GameView }) {
  const u = useUnit();
  const pitcher = v.pitcher ?? v.lastPitch?.pitcher ?? null;
  const rows = useMemo(() => {
    const mine = v.detail.pitches.filter(p => pitcher && p.pitcher.id === pitcher.id && !unconfirmed(p) && p.actual?.pitch_type);
    const by = new Map<string, { label: string; n: number; speeds: number[] }>();
    for (const p of mine) { const a = p.actual!; const k = a.pitch_type!; const r = by.get(k) ?? { label: a.pitch_label, n: 0, speeds: [] }; r.n++; if (typeof a.speed_mph === 'number') r.speeds.push(a.speed_mph); by.set(k, r); }
    const total = mine.length;
    return { total, list: [...by.entries()].sort((a, b) => b[1].n - a[1].n).map(([k, r]) => ({ code: k, label: r.label, n: r.n, share: total ? r.n / total : 0, avg: r.speeds.length ? r.speeds.reduce((s, x) => s + x, 0) / r.speeds.length : null })) };
  }, [v.detail.pitches, pitcher]);
  return <div className="v3-detail">
    <div className="v3-person"><b>{pitcher?.name ?? '투수 미확인'}</b><span>{handText(pitcher?.hand, 'p')} · 오늘 {rows.total}구 관측</span></div>
    {rows.total === 0 ? <p className="v3-empty-line">오늘 관측된 공이 아직 없어요.</p> : <table className="v3-table">
      <thead><tr><th scope="col">구종</th><th scope="col" className="num">오늘 관측</th><th scope="col" className="num">시즌 사용률</th><th scope="col" className="num">오늘 평균</th></tr></thead>
      <tbody>{rows.list.map(r => <tr key={r.code}><th scope="row">{r.label} <small className="v3-code">{r.code}</small></th>
        <td className="num"><span className="v3-bar actual"><i style={{ width: `${Math.round(r.share * 100)}%` }} /></span>{r.n}구 · {Math.round(r.share * 100)}%</td>
        <td className="num na">{NA}</td><td className="num">{speedNum(r.avg, u)}<small> {unitText(u)}</small></td></tr>)}</tbody>
    </table>}
    <p className="v3-muted small">오늘 관측 비율은 오늘 던진 {rows.total}구만 센 값이라 시즌 사용률과 다를 수 있어요. 시즌 사용률과 구종별 성과는 발행 파일에 아직 없어요(—).</p>
  </div>;
}

// ───────── game record ─────────
export function RecordDetail({ v, open, focusPa }: { v: GameView; open: Open; focusPa?: string }) {
  const [tab, setTab] = useState<'log' | 'line' | 'box'>('log');
  const d = v.detail;
  const halves = useMemo(() => { const m = new Map<string, Pa[]>(); for (const p of d.pas) { const k = `${p.inning}:${p.half}`; m.set(k, [...(m.get(k) ?? []), p]); } return [...m.entries()].reverse(); }, [d.pas]);
  const ls = d.linescore_asof;
  const n = Math.max(9, ls?.innings ?? 0);
  const g = d.game;
  return <div className="v3-detail">
    <div className="v3-seg" role="group" aria-label="기록 보기">
      {([['log', '문자 기록'], ['line', '이닝 점수'], ['box', '라인업·박스']] as const).map(([k, t]) => <button key={k} aria-pressed={tab === k} onClick={() => setTab(k)}>{t}</button>)}
    </div>
    {tab === 'log' && (d.pas.length === 0 ? <p className="v3-empty-line">아직 기록된 타석이 없어요.</p> : halves.map(([k, list]) => <section key={k} className="v3-log-half">
      <h3>{inningText(list[0])}</h3>
      <ol>{[...list].reverse().map(pa => { const hasPitch = d.pitches.some(x => x.index === pa.first && x.pa_key === pa.pa_key);
        return <li key={pa.pa_key} className={pa.pa_key === focusPa ? 'focus' : ''}><button disabled={!hasPitch} onClick={() => hasPitch && open('pitch', { pitch: pa.first })}>
          <b>{pa.batter}</b><span>{pa.complete ? pa.result : pa.pa_key === v.paKey && v.live ? '타석 진행 중' : pa.result}</span><small>{pa.pitch_count ?? (pa.last - pa.first + 1)}구</small></button></li>; })}</ol>
    </section>))}
    {tab === 'line' && (ls ? <div className="v3-scroll"><table className="v3-table line">
      <thead><tr><th scope="col">팀</th>{Array.from({ length: n }, (_, i) => <th key={i} scope="col" className="num">{i + 1}</th>)}<th scope="col" className="num">R</th><th scope="col" className="num">H</th><th scope="col" className="num">E</th></tr></thead>
      <tbody>{(['away', 'home'] as const).map(k => <tr key={k}><th scope="row">{k === 'away' ? (g.away_abbr || short(g.away_team)) : (g.home_abbr || short(g.home_team))}</th>
        {Array.from({ length: n }, (_, i) => <td key={i} className="num">{ls[k].innings[i] ?? '–'}</td>)}<td className="num"><b>{ls[k].r ?? '–'}</b></td><td className="num">{ls[k].h ?? '–'}</td><td className="num">{ls[k].e ?? '–'}</td></tr>)}</tbody>
    </table></div> : <p className="v3-empty-line">이닝별 점수는 아직 없어요. 지금 점수는 {short(g.away_team)} {v.awayScore ?? '–'} : {v.homeScore ?? '–'} {short(g.home_team)}예요.</p>)}
    {tab === 'box' && <div>
      <p className="v3-empty-line">라인업과 박스스코어는 발행 파일에 아직 없어요 ({NA}).</p>
      {v.upcoming.length > 0 && <><h3 className="v3-h3">다음 타자</h3><ol className="v3-next-batters">{v.upcoming.map(b => <li key={b.order}>{b.order}번 · {b.name ?? NA}</li>)}</ol></>}
    </div>}
    <p className="v3-muted small">공개된 플레이까지만 보여 드려요.</p>
  </div>;
}

// ───────── win probability ─────────
export function WinProbDetail() {
  return <div className="v3-detail">
    <div className="v3-wp-empty" aria-hidden="true"><svg viewBox="0 0 320 120"><line x1="0" y1="60" x2="320" y2="60" /><text x="160" y="66" textAnchor="middle">승리확률 {NA}</text></svg></div>
    <p className="v3-empty-line">승리확률은 아직 제공되지 않아요.</p>
    <div className="v3-explain"><h3>제공되면 보여 드릴 것</h3><p>지금 확률과 경기 흐름 그래프, 크게 움직인 장면, 그리고 이 확률을 어떤 기준으로 계산했는지.</p></div>
  </div>;
}

// ───────── limits ─────────
export function InfoDetail({ notes }: { notes: string[] }) {
  const u = useUnit();
  return <div className="v3-detail">
    <ul className="v3-notes">{[...LIMITS, ...notes.filter(n => !LIMITS.includes(n))].map(n => <li key={n}>{n}</li>)}</ul>
    <button className="v3-btn" onClick={toggleUnit} aria-label={`속도 단위 바꾸기, 지금 ${unitText(u)}`}>속도 단위: {unitText(u)}</button>
  </div>;
}
export const pitchOf = (d: Detail, index: number): PitchRow | null => d.pitches.find(p => p.index === index) ?? null;
