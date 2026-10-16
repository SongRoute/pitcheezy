import type { ReactNode } from 'react';
import type { Candidate, PitchRow, Situation } from '../service/model';
import { inningText, short } from '../service/model';
import type { GameView } from './view';
import type { Options } from './options';
import Zone from './Zone';
import type { ZonePitch } from './Zone';
import { candidateWords, pointWords } from './zones';
import { callKind, callText, handText, speedNum, unitText, useUnit } from './format';
import { listHref } from './links';

export type PanelKind = 'rec' | 'pitch' | 'player' | 'arsenal' | 'record' | 'wp' | 'info';
export type Open = (kind: PanelKind, arg?: { pitch?: number; who?: 'batter' | 'pitcher'; pa?: string }) => void;

const teamName = (abbr: string | null | undefined, team: string) => abbr || short(team);

function Dots({ n, on, cls, label }: { n: number; on: number; cls: string; label: string }) {
  return <span className={`v3-dots ${cls}`} aria-hidden="true"><b>{label}</b>{Array.from({ length: n }, (_, i) => <i key={i} className={i < on ? 'on' : ''} />)}</span>;
}
export function Bases({ mask, size = 30 }: { mask: number; size?: number }) {
  const on = (b: number) => (mask & (1 << (b - 1))) !== 0;
  const runners = [1, 2, 3].filter(on);
  return <svg className="v3-bases" width={size} height={size * 0.78} viewBox="0 0 36 28" role="img" aria-label={`주자 ${runners.length ? runners.join('·') + '루' : '없음'}`}>
    {[[18, 7, 2], [29, 17, 1], [7, 17, 3]].map(([x, y, b]) => <rect key={b} x={x - 5} y={y - 5} width="10" height="10" transform={`rotate(45 ${x} ${y})`} className={on(b) ? 'on' : ''} />)}
  </svg>;
}
export function Count({ s }: { s: Situation }) {
  return <span className="v3-count" role="img" aria-label={`볼 ${s.balls}, 스트라이크 ${s.strikes}, ${s.outs}아웃`}>
    <Dots n={3} on={s.balls} cls="b" label="B" /><Dots n={2} on={s.strikes} cls="s" label="S" /><Dots n={2} on={s.outs} cls="o" label="O" />
  </span>;
}

/** the connection story as a small chip (header) */
export function ConnChip({ v }: { v: GameView }) {
  const { conn } = v;
  return <span className={`v3-chip conn ${conn.kind}`} data-conn={conn.kind}>{conn.kind === 'ok' && v.live && <i aria-hidden="true" />}{conn.label}</span>;
}

/** Dark scoreboard: away left, home right; inning, count, outs, bases; the connection chip. */
export function Scoreboard({ v, wide = false }: { v: GameView; wide?: boolean }) {
  const g = v.detail.game;
  const s = v.situation;
  const inning = v.final ? '경기 종료' : s ? inningText(s) : v.ended ? `${inningText(v.ended)} 종료` : g.status === 'scheduled' ? '경기 전' : '';
  const away = teamName(g.away_abbr, g.away_team), home = teamName(g.home_abbr, g.home_team);
  return <header className={`v3-board${wide ? ' wide' : ''}`}>
    <a className="v3-back" href={listHref()} aria-label="경기 목록으로">‹</a>
    <h1 className="v3-score" aria-label={`${short(g.away_team)} ${v.awayScore ?? '–'} 대 ${v.homeScore ?? '–'} ${short(g.home_team)}${inning ? `, ${inning}` : ''}`}>
      <span aria-hidden="true">{away}</span><strong aria-hidden="true">{v.awayScore ?? '–'}</strong><i aria-hidden="true">:</i><strong aria-hidden="true">{v.homeScore ?? '–'}</strong><span aria-hidden="true">{home}</span>
    </h1>
    <div className="v3-board-meta">
      <span className="v3-inning">{v.breakNow ? '공수 교대' : inning}</span>
      {s && !v.final && <Count s={s} />}
      {s && !v.final && <Bases mask={s.bases} />}
    </div>
    <ConnChip v={v} />
  </header>;
}

/** Below the scoreboard, only while the connection story is not "ok": what is on screen is not current, in plain words. */
export function ConnBanner({ v, busy, onRetry }: { v: GameView; busy: boolean; onRetry: () => void }) {
  if (v.conn.kind === 'ok') return null;
  const t = v.lastOk ? new Date(v.lastOk).toLocaleTimeString('en-GB', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit' }) : null;
  return <div className={`v3-banner ${v.conn.kind}`} data-kind={v.conn.kind} data-publish-clock={v.lagClock ?? undefined}>
    <p role="status"><b>{v.conn.label}</b>{v.conn.detail && <span> · {v.conn.detail}</span>}</p>
    {t && <span className="v3-banner-time" aria-hidden="true">{t} 수신</span>}
    <button className="v3-link" onClick={onRetry} disabled={busy} aria-label="지금 다시 불러오기">{busy ? '…' : '다시'}</button>
  </div>;
}

/** Batter vs pitcher with today's line (full) or names only (lite). Tapping opens the player panel. */
export function Matchup({ v, density, open }: { v: GameView; density: Options['density']; open: Open }) {
  if (v.idle || (!v.batter && !v.pitcher)) return <div className="v3-matchup idle"><span>{v.breakNow ? '다음 타자를 기다리고 있어요' : v.final ? '경기가 끝났어요' : '경기가 시작되면 타자와 투수를 보여 드려요'}</span></div>;
  const bt = v.batterToday, pt = v.pitcherToday;
  const n = (x: number | null | undefined) => typeof x === 'number' ? x : '–';
  return <div className="v3-matchup" role="group" aria-label="이번 타석">
    <button onClick={() => open('player', { who: 'batter' })} aria-haspopup="dialog">
      <small>타자</small><b>{v.batter?.name ?? '선수 미확인'}</b>
      {density === 'full' && <span>{handText(v.batter?.side, 'b')}{bt ? ` · 오늘 ${n(bt.at_bats)}타수 ${n(bt.hits)}안타` : ''}</span>}
    </button>
    <button onClick={() => open('player', { who: 'pitcher' })} aria-haspopup="dialog">
      <small>투수</small><b>{v.pitcher?.name ?? '선수 미확인'}</b>
      {density === 'full' && <span>{handText(v.pitcher?.hand, 'p')}{pt?.pitches != null ? ` · ${pt.pitches}구` : ''}</span>}
    </button>
  </div>;
}

export function zonePitches(list: PitchRow[]): ZonePitch[] {
  return list.map((p, i) => ({ n: p.pitch_number, x: p.receipt?.status === 'unconfirmed' ? null : p.actual?.x ?? null, z: p.receipt?.status === 'unconfirmed' ? null : p.actual?.z ?? null, latest: i === list.length - 1 }));
}
export function zoneLabel(v: Pick<GameView, 'detail' | 'side'>, cands: Candidate[], pitches: PitchRow[], view: Options['view']) {
  const b = v.detail.zone_bounds;
  const rec = cands.map(c => `추천 ${c.rank}순위 ${c.pitch_label} ${candidateWords(c, v.side, b)}`).join(', ');
  const act = pitches.map(p => p.receipt?.status === 'unconfirmed' ? `${p.pitch_number}구 확인 중` : `${p.pitch_number}구 ${p.actual?.pitch_label ?? ''} ${pointWords(p.actual?.x ?? null, p.actual?.z ?? null, b, v.side)}`).join(', ');
  return `${view === 'pitcher' ? '투수' : '포수'} 쪽에서 본 스트라이크존. ${rec || '표시된 추천 없음'}. ${act ? `이번 타석 ${act}` : '이번 타석 투구 없음'}.`;
}

/** Rank chips under the zone: rank + pitch name (+ location words in full density). Tapping opens the recommendation panel. */
export function Candidates({ v, density, open }: { v: GameView; density: Options['density']; open: Open }) {
  if (!v.cands.length) return null;
  return <div className="v3-cands" role="list">
    {v.cands.map(c => <button key={c.rank} role="listitem" className={`v3-cand r${c.rank}`} onClick={() => open('rec')} aria-haspopup="dialog"
      aria-label={`${c.rank}순위 ${c.pitch_label}, ${candidateWords(c, v.side, v.detail.zone_bounds)}. 자세히 보기`}>
      <i aria-hidden="true">{c.rank}</i><b>{c.pitch_label}</b>{density === 'full' && <small>{candidateWords(c, v.side, v.detail.zone_bounds)}</small>}
    </button>)}
  </div>;
}

/** The recommendation block: heading, the zone (hero), the chips, and ONE state sentence when nothing can be shown. */
export function RecBlock({ v, opts, open, side, compact = false }: { v: GameView; opts: Options; open: Open; side?: ReactNode; compact?: boolean }) {
  const title = v.nextPhase || (v.stateSlot && !v.final) ? `다음 공 추천${v.nextNumber ? ` · ${v.nextNumber}구` : ''}` : v.lastPitch ? `마지막 공 · 투구 전에 기록된 추천` : '추천';
  const count = v.situation ? `${v.situation.balls}-${v.situation.strikes}` : '';
  const pitches = v.paPitches;
  const stateText = v.eff && !v.shownRec ? v.eff : null;
  const zone = <Zone bounds={v.detail.zone_bounds} view={opts.view} mark={opts.zone} batter={v.side} recs={v.cands}
    pitches={zonePitches(pitches)} held={v.held} label={zoneLabel(v, v.cands, pitches, opts.view)} size="main" />;
  return <section className={`v3-rec${v.held ? ' held' : ''}${compact ? ' compact' : ''}`} aria-label={title} data-state={v.eff?.state ?? (v.shownRec ? 'recommended' : 'none')}>
    <div className="v3-rec-head">
      <h2>{v.held ? '마지막으로 받은 추천' : title}{count && v.nextPhase && <span className="v3-rec-count">{count}</span>}</h2>
      <button className="v3-link" onClick={() => open('rec')} aria-haspopup="dialog">자세히</button>
    </div>
    <div className={`v3-rec-stage${side ? ' with-side' : ''}`}>
      <div className="v3-zone-box">{zone}</div>
      {side}
    </div>
    {stateText && <div className="v3-state" role="status" data-state={stateText.state}>
      <span className={`v3-chip st ${stateText.state}`}>{stateText.label}</span>{stateText.message && <p>{stateText.message}</p>}</div>}
    {!stateText && !v.shownRec && <div className="v3-state" role="status"><p>{v.nextPhase ? (v.phase === 'between_innings' ? '공수 교대 중이에요.' : '추천 대기 중이에요.') : '이 공에는 기록된 추천이 없어요.'}</p></div>}
    <Candidates v={v} density={opts.density} open={open} />
    {v.note && <p className="v3-note" role="status" data-kind={v.noteKind ?? undefined}>{v.note}</p>}
    {opts.density === 'full' && !compact && <p className="v3-legend" aria-hidden="true"><span className="k rec" />추천 칸 <span className="k past" />이번 타석 공 <span className="k latest" />직전 공</p>}
  </section>;
}

/** This plate appearance's pitches: number · type · speed · call. Each opens the pitch panel. Three layouts (spec §3). */
export function PitchList({ v, layout, open, density }: { v: GameView; layout: Options['list']; open: Open; density: Options['density'] }) {
  const u = useUnit();
  const list = v.paPitches;
  const next = v.nextPhase && v.nextNumber && v.paKey === v.detail.current?.pa_key ? v.nextNumber : null;
  if (!list.length && !next) return layout === 'side' ? null : <div className={`v3-pitches ${layout} empty`}><p>{v.breakNow ? '공수 교대 중' : '이번 타석 투구 없음'}</p></div>;
  const item = (p: PitchRow, i: number) => {
    const latest = i === list.length - 1, pending = p.receipt?.status === 'unconfirmed';
    const a = pending ? null : p.actual;
    const speed = speedNum(a?.speed_mph, u);
    return <li key={p.index}><button className={`v3-pitch${latest ? ' latest' : ''}${pending ? ' pending' : ''}`} onClick={() => open('pitch', { pitch: p.index })} aria-haspopup="dialog"
      aria-label={pending ? `${p.pitch_number}구 확인 중` : `${p.pitch_number}구 ${a?.pitch_label ?? '구종 미상'} ${speed === '—' ? '' : `${speed}${unitText(u)} `}${callText(p)}${p.receipt?.corrected ? ', 정정됨' : ''}. 자세히 보기`}>
      <i className="v3-pn" aria-hidden="true">{p.pitch_number}</i>
      <b aria-hidden="true">{pending ? '확인 중' : a?.pitch_label ?? '구종 미상'}</b>
      {!pending && (layout !== 'row' || density === 'full') && <span className="v3-sp" aria-hidden="true">{speed}{layout === 'stack' && !pending && speed !== '—' ? <small> {unitText(u)}</small> : null}</span>}
      {!pending && <span className={`v3-call ${callKind(a)}`} aria-hidden="true">{callText(p)}</span>}
      {p.receipt?.corrected && !pending && <em className="v3-fixed" aria-hidden="true">정정</em>}
    </button></li>;
  };
  return <section className={`v3-pitches ${layout}`} aria-label="이번 타석 투구">
    {layout !== 'side' && <h3>이번 타석 {v.situation && list.length > 0 ? <span>{list.length}구 · {v.situation.balls}-{v.situation.strikes}</span> : null}</h3>}
    <ol>{list.map(item)}{next && <li><span className="v3-pitch next" aria-label={`다음 공 ${next}구, 아직 안 던짐`}><i className="v3-pn" aria-hidden="true">{next}</i><b aria-hidden="true">다음</b></span></li>}</ol>
  </section>;
}

export function PrevAb({ v, open }: { v: GameView; open: Open }) {
  const pa = v.prevPa;
  if (!pa) return null;
  return <button className="v3-prev" onClick={() => open('record', { pa: pa.pa_key })} aria-haspopup="dialog" aria-label={`직전 타석 ${pa.batter} ${pa.result}. 타석 기록 보기`}>
    <small>직전 타석</small><b>{pa.batter}</b><span>{pa.result}</span><em>기록 ›</em>
  </button>;
}

const ICON: Record<string, string> = {
  player: 'M10 3a3.2 3.2 0 1 0 0 6.4a3.2 3.2 0 1 0 0-6.4M4 17c0-3.3 2.7-5 6-5s6 1.7 6 5',
  arsenal: 'M10 3a7 7 0 1 0 0 14a7 7 0 1 0 0-14M6 5c2 3 2 7 0 10M14 5c-2 3-2 7 0 10',
  record: 'M5 5h10M5 10h10M5 15h7',
  wp: 'M3 15l4-5 3 3 6-8M3 17h14',
};
export function Dock({ open }: { open: Open }) {
  const items: [PanelKind, string][] = [['player', '선수'], ['arsenal', '구종'], ['record', '기록'], ['wp', '승리확률']];
  return <nav className="v3-dock" aria-label="더 보기">
    {items.map(([k, t]) => <button key={k} onClick={() => open(k)} aria-haspopup="dialog">
      <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d={ICON[k]} /></svg><span>{t}</span></button>)}
  </nav>;
}
