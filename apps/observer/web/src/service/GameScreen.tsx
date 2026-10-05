import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { FetchError, fetchDetail, mergeDetail } from './api';
import type { Actual, Candidate, Detail, Pa, PitchRow, Rec, Situation } from './model';
import { dateHeading, halfText, inningText, kstTime, policyShare, short } from './model';
import { teamColor } from './teams';
import Zone from './Zone';
import { DetailSheet, InfoSheet, PaSheet, provenanceText } from './Sheets';
import { StatusChip } from './WatchList';
import './service.css';

/** One game screen for live, demo and finished games (UI redesign 2026-10-05).
 *  Two independent states, never mixed: the GAME's status (LIVE dot, from the server) and the VIEWER's state
 *  (following the latest vs. looking at a past pitch; auto refresh on/off). */
export const POLL_MS = 2000;
export const STALE_MS = 30000;
type Selection = { kind: 'latest' } | { kind: 'pitch'; index: number };

const UNIT_KEY = 'pitcheezy.unit';
const readUnit = () => { try { return localStorage.getItem(UNIT_KEY) === 'mph' ? 'mph' : 'kmh'; } catch { return 'kmh'; } };
const speedText = (mph: number | null, unit: 'kmh' | 'mph') => mph === null ? '' : unit === 'kmh' ? `${Math.round(mph * 1.609344)}km/h` : `${mph.toFixed(1)}mph`;
const clockText = (ms: number) => new Date(ms).toLocaleTimeString('ko-KR', { timeZone: 'Asia/Seoul', hour12: false });
export const gameIdFromPath = () => {
  const m = /^\/(?:watch|game|live)\/(\d+)\/?$/.exec(window.location.pathname);
  const q = new URLSearchParams(window.location.search).get('game');
  const id = Number(m?.[1] ?? q);
  return Number.isInteger(id) && id > 0 ? id : null;
};

function Bases({ mask }: { mask: number }) {
  const on = (b: number) => (mask & (1 << (b - 1))) !== 0;
  const runners = [1, 2, 3].filter(on);
  return <svg className="sv-bases" viewBox="0 0 44 32" role="img" aria-label={`주자 ${runners.length ? runners.join('·') + '루' : '없음'}`}>
    {[[22, 6, 2], [36, 18, 1], [8, 18, 3]].map(([x, y, b]) => <rect key={b} x={x - 5.5} y={y - 5.5} width="11" height="11" transform={`rotate(45 ${x} ${y})`} className={on(b) ? 'on' : ''} />)}
  </svg>;
}
function Team({ team, batting }: { team: string; batting: boolean }) {
  return <span className={`sv-team ${batting ? 'atk' : ''}`} style={{ '--team': teamColor(team) } as CSSProperties}>
    <i aria-hidden="true" />{short(team)}{batting && <small aria-label="공격 중">▲공격</small>}</span>;
}

function Candidates({ rec, selected, onSelect, label, upcoming }: { rec: Rec; selected: number; onSelect: (rank: number) => void; label: string; upcoming: boolean }) {
  if (rec.status !== 'ready') return <p className="sv-norec">{rec.status === 'unsupported' ? rec.reason ?? '이 상황은 추천을 지원하지 않아요.'
    : upcoming ? `다음 공 추천이 없어요.${rec.reason ? ` ${rec.reason}` : ''}` : `이 공은 추천 기록이 없어요.${rec.reason ? ` ${rec.reason}` : ''}`}</p>;
  return <div className="sv-cands" role="radiogroup" aria-label={`${label} 후보`} style={{ '--n': rec.candidates.length } as CSSProperties}>
    {rec.candidates.map((c: Candidate) => <button key={c.rank} role="radio" aria-checked={c.rank === selected} className={`sv-cand ${c.rank === selected ? 'on' : ''}`}
      onClick={() => onSelect(c.rank)} aria-label={`${c.rank}순위 ${c.pitch_label}, ${c.zone_label || '위치 없음'}, 정책 선택 비중 ${policyShare(c.probability)}`}>
      <span className="sv-cand-rank" aria-hidden="true">{c.rank}</span>
      <span className="sv-cand-type">{c.pitch_label}</span>
      <span className="sv-cand-sub"><span className="sv-cand-zone">{c.zone_label || '위치 없음'}</span>
        <span className="sv-cand-share">{policyShare(c.probability)}</span></span>
    </button>)}
  </div>;
}

function ActualLine({ actual, unit, title, paResult, previous }: { actual: Actual | null; unit: 'kmh' | 'mph'; title: string; paResult: string | null; previous: boolean }) {
  return <div className="sv-actual-line">
    <span className={previous ? 'sv-key-prev' : 'sv-key-actual'} aria-hidden="true" /><b>{title}</b>
    {actual ? <span className="sv-actual-text">{actual.pitch_label}{actual.speed_mph !== null ? ` ${speedText(actual.speed_mph, unit)}` : ''} · <strong>{paResult ?? actual.result_label}</strong></span>
      : <span className="sv-muted">기록 없음</span>}
  </div>;
}

export default function GameScreen({ gamePk }: { gamePk: number }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<{ text: string; status: number } | null>(null);
  const [auto, setAuto] = useState(true);
  const [sel, setSel] = useState<Selection | null>(null);
  const [rank, setRank] = useState(1);
  const [lastOk, setLastOk] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [sheet, setSheet] = useState<'pa' | 'detail' | 'info' | null>(null);
  const [busy, setBusy] = useState(false);
  const [unit, setUnit] = useState<'kmh' | 'mph'>(readUnit);
  const detailRef = useRef<Detail | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const load = useCallback(async (full = false, force = false) => {
    // a background poll never cancels a slower answer still on its way (only a manual refresh does)
    if (inflight.current && !inflight.current.signal.aborted && !force) return;
    inflight.current?.abort();
    const ctrl = new AbortController(); inflight.current = ctrl;
    setBusy(true);
    try {
      const old = detailRef.current;
      const last = old?.pitches.length ? old.pitches[old.pitches.length - 1].index : undefined;
      let next = await fetchDetail(gamePk, ctrl.signal, full || !old || last === undefined ? undefined : last);
      if (old && !full && last !== undefined) {
        const merged = mergeDetail(old, next);
        next = merged ?? await fetchDetail(gamePk, ctrl.signal);  // cutoff moved back (demo seek/reset): take the full history again
      }
      if (ctrl.signal.aborted) return;
      detailRef.current = next; setDetail(next); setError(null); setLastOk(Date.now());
    } catch (e) {
      if (ctrl.signal.aborted) return;
      setError(e instanceof FetchError ? { text: e.message, status: e.status } : { text: '자료를 불러오지 못했어요.', status: 0 });
    } finally { if (inflight.current === ctrl) { inflight.current = null; setBusy(false); } }
  }, [gamePk]);

  useEffect(() => { void load(true, true); return () => inflight.current?.abort(); }, [load]);
  const live = detail ? detail.game.status === 'live' || (detail.next !== null && detail.game.status !== 'final') : false;
  // Polling follows the viewer's switch only; the game's LIVE status decides whether there is anything to poll.
  useEffect(() => {
    if (!auto || !live) return;
    const id = window.setInterval(() => { if (!document.hidden) void load(); }, POLL_MS);
    return () => window.clearInterval(id);
  }, [auto, live, load]);
  useEffect(() => { const id = window.setInterval(() => setNow(Date.now()), 5000); return () => window.clearInterval(id); }, []);
  useEffect(() => {  // coming back to the tab: refresh at once when following
    const onVis = () => { if (!document.hidden && auto && live) void load(); };
    document.addEventListener('visibilitychange', onVis); return () => document.removeEventListener('visibilitychange', onVis);
  }, [auto, live, load]);

  // default selection once data arrives: live → follow latest; finished → first pitch
  useEffect(() => { if (detail && sel === null) setSel(live ? { kind: 'latest' } : detail.pitches.length ? { kind: 'pitch', index: detail.pitches[0].index } : { kind: 'latest' }); }, [detail, sel, live]);
  // a pinned pitch that disappeared (demo seek back) falls back to latest
  useEffect(() => { if (detail && sel?.kind === 'pitch' && !detail.pitches.some(p => p.index === sel.index)) setSel({ kind: 'latest' }); }, [detail, sel]);
  useEffect(() => { setRank(1); }, [sel?.kind, sel?.kind === 'pitch' ? sel.index : -1, detail?.next?.candidates.map(c => c.pitch_type).join()]);
  useEffect(() => { if (detail) document.title = `${short(detail.game.away_team)} 대 ${short(detail.game.home_team)} · 피치이지`; }, [detail?.game.away_team, detail?.game.home_team]);

  const byIndex = useMemo(() => new Map(detail?.pitches.map(p => [p.index, p]) ?? []), [detail?.pitches]);
  if (!detail) {
    return <div className="sv-page"><TopBar /><main className="sv-center">{error
      ? <div className="sv-alert" role="alert"><p>{error.status === 423 ? '이 경기는 지금 기록을 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.' : error.text}</p>
        <div className="sv-row"><button className="sv-btn" onClick={() => void load(true)}>다시 시도</button><a className="sv-btn" href="/watch">경기 목록</a></div></div>
      : <p className="sv-muted" role="status">경기 기록을 불러오는 중…</p>}</main></div>;
  }

  const { game } = detail;
  const pitches = detail.pitches;
  const lastPitch = pitches[pitches.length - 1] ?? null;
  const pitch: PitchRow | null = sel?.kind === 'pitch' ? byIndex.get(sel.index) ?? null : null;
  const latest = pitch === null;  // a pin whose pitch is gone (demo seek back) shows latest until the effect resets it
  // What the stage shows
  const situation: Situation | null = latest ? detail.current?.situation ?? lastPitch?.situation ?? null : pitch!.situation;
  const batter = latest ? detail.current?.batter?.name ?? lastPitch?.batter.name : pitch!.batter.name;
  const pitcher = latest ? detail.current?.pitcher?.name ?? lastPitch?.pitcher.name : pitch!.pitcher.name;
  const rec: Rec | null = latest ? (live ? detail.next : lastPitch?.rec ?? null) : pitch!.rec;
  const actual: Actual | null = latest ? lastPitch?.actual ?? null : pitch!.actual;
  const nextPhase = latest && live;
  const candidate = rec?.status === 'ready' ? rec.candidates.find(c => c.rank === rank) ?? rec.candidates[0] ?? null : null;
  const shownPitch = latest ? lastPitch : pitch;
  const paKey = latest ? detail.current?.pa_key ?? lastPitch?.pa_key ?? null : pitch!.pa_key;
  const pa: Pa | null = detail.pas.find(p => p.pa_key === paKey) ?? null;
  const paPitches = pitches.filter(p => p.pa_key === paKey);
  const ending = shownPitch?.actual?.event ? detail.pas.find(p => p.pa_key === shownPitch.pa_key && p.complete)?.result ?? null : null;
  const nextNumber = nextPhase ? (detail.current?.pa_key && lastPitch?.pa_key === detail.current.pa_key ? lastPitch.pitch_number + 1 : 1) : null;
  const sameAtBatPrev = nextPhase && lastPitch && detail.current?.pa_key === lastPitch.pa_key;

  const stale = (live && auto && lastOk !== null && now - lastOk > STALE_MS) || (live && (detail.feed.ok === false || (detail.feed.stale_s ?? 0) > 30));
  const batting = situation ? (situation.half === 'Top' ? game.away_team : game.home_team) : null;
  const scored = situation ?? null;
  const awayScore = latest && game.away_score !== null ? game.away_score : scored?.away_score ?? null;
  const homeScore = latest && game.home_score !== null ? game.home_score : scored?.home_score ?? null;

  const pick = (index: number) => { setSel({ kind: 'pitch', index }); };
  const goLatest = () => { setSel({ kind: 'latest' }); setAuto(true); void load(false, true); };
  const stepPa = (dir: -1 | 1) => {
    const i = detail.pas.findIndex(p => p.pa_key === paKey);
    const target = detail.pas[i + dir];
    if (!target) return;
    if (typeof target.first === 'number' && byIndex.has(target.first)) pick(target.first); else goLatest();  // in-progress PA without a pitch yet
  };
  const paIdx = detail.pas.findIndex(p => p.pa_key === paKey);
  const recTitle = nextPhase ? `다음 공 추천${nextNumber ? ` · ${nextNumber}구` : ''}` : '이 공 던지기 전 추천';
  // the inning is left out when the board above already shows the same half-inning (it is repeated otherwise)
  const sameHalf = !!lastPitch && !!situation && lastPitch.situation.inning === situation.inning && lastPitch.situation.half === situation.half;
  const actualTitle = nextPhase ? `방금 던진 공${lastPitch ? ` · ${sameHalf ? '' : `${inningText(lastPitch.situation)} `}${lastPitch.pitch_number}구` : ''}` : `실제 투구${pitch ? ` · ${pitch.pitch_number}구` : ''}`;

  return <div className="sv-page sv-game">
    <TopBar right={<StatusChip g={game} />}
      title={`${dateHeading(game.kst_date).replace(/^\d{4}년 /, '')} ${kstTime(game.start)}${game.double_header ? ` · ${game.game_number}차전` : ''}`}
      label={`${dateHeading(game.kst_date)} ${kstTime(game.start)} 시작${game.double_header ? `, ${game.game_number}차전` : ''}, ${short(game.away_team)} 대 ${short(game.home_team)}`} />
    <section className="sv-board" aria-label="경기 상황" style={{ '--team': batting ? teamColor(batting) : 'transparent' } as CSSProperties}>
      <div className="sv-score">
        <Team team={game.away_team} batting={situation?.half === 'Top'} /><strong>{awayScore ?? '–'}</strong>
        <i aria-hidden="true">:</i><strong>{homeScore ?? '–'}</strong><Team team={game.home_team} batting={situation?.half === 'Bot'} />
      </div>
      {situation && <div className="sv-count">
        <strong>{inningText(situation)}</strong>
        <span aria-label={`볼 ${situation.balls} 스트라이크 ${situation.strikes}`}><b>B</b>{situation.balls} <b>S</b>{situation.strikes}</span>
        <span aria-label={`${situation.outs}아웃`}><b>O</b>{situation.outs}</span>
        <Bases mask={situation.bases} />
      </div>}
      <div className="sv-matchup"><span><em>투수</em>{pitcher ?? '—'}</span><span><em>타자</em>{batter ?? '—'}</span></div>
    </section>

    <section className={`sv-mode ${latest ? 'is-latest' : 'is-past'} ${stale || error ? 'is-stale' : ''}`} aria-live="polite">
      {stale || error ? <span className="sv-mode-text" role="alert">{error ? `연결 끊김 · ${error.text}` : '기록 갱신이 늦어지고 있어요'}{lastOk ? ` · 마지막 수신 ${clockText(lastOk)}` : ''}</span>
        : latest ? <span className="sv-mode-text">{live ? (auto ? '최신 따라가는 중' : '자동 갱신 꺼짐') : '경기 종료 · 마지막 공'}</span>
        : <span className="sv-mode-text">지난 기록 보는 중 · {inningText(pitch!.situation)} {pitch!.pitch_number}구</span>}
      <span className="sv-mode-actions">
        {!latest && <button className="sv-btn primary" onClick={goLatest}>{live ? '최신으로' : '마지막 공'}</button>}
        {live && <button className="sv-btn" aria-pressed={auto} onClick={() => setAuto(a => !a)} aria-label={auto ? '자동 갱신 끄기' : '자동 갱신 켜기'}>{auto ? '자동 ON' : '자동 OFF'}</button>}
        {(live || stale || error) && <button className="sv-btn" onClick={() => void load(stale || !!error, true)} disabled={busy} aria-label="지금 새로고침">{busy ? '…' : '새로고침'}</button>}
      </span>
    </section>

    <section className="sv-stage" aria-label={nextPhase ? '방금 던진 공과 다음 공 추천' : '이 공의 추천과 실제 투구'}>
      <div className="sv-split">
        <div className="sv-zone-col">
        <div className="sv-rec-head"><span className="sv-key-rec" aria-hidden="true" /><b>{recTitle}</b><small>% = 정책 선택 비중</small></div>
        <div className="sv-zone-wrap">
          <Zone bounds={detail.zone_bounds} candidate={candidate}
            actual={!nextPhase && actual && actual.x !== null && actual.z !== null ? { x: actual.x, z: actual.z } : null} actualLabel="실제"
            previous={sameAtBatPrev && actual && actual.x !== null && actual.z !== null ? { x: actual.x, z: actual.z } : null}
            setupX={!nextPhase ? actual?.setup_x ?? null : null} />
        </div>
        </div>
        <div className="sv-recbox">
          {rec ? <Candidates rec={rec} selected={candidate?.rank ?? 1} onSelect={setRank} label={recTitle} upcoming={!!nextPhase} />
            : <p className="sv-norec">{nextPhase ? (detail.current?.phase === 'between_innings' ? '공수 교대 중이에요.' : detail.current?.phase === 'resolving' ? '방금 공의 결과를 확인하는 중이에요.' : '다음 공 추천이 아직 없어요.') : '이 시점에는 추천이 없어요.'}</p>}
        </div>
      </div>
      <ActualLine actual={actual} unit={unit} title={actualTitle} paResult={ending} previous={!!nextPhase} />
    </section>

    <nav className="sv-nav" aria-label="타석과 투구 선택">
      <div className="sv-pa-row">
        <button className="sv-icon-btn" onClick={() => stepPa(-1)} disabled={paIdx <= 0} aria-label="이전 타석">‹</button>
        <button className="sv-pa-open" onClick={() => setSheet('pa')} aria-haspopup="dialog" aria-label={`전체 타석 목록 열기. 현재 ${pa ? `${pa.inning}회 ${halfText(pa.half)} ${pa.batter} 대 ${pa.pitcher}, ${pa.result}` : '타석 정보 없음'}`}>
          <span>{pa ? `${pa.inning}회 ${halfText(pa.half)} · ${pa.batter} @ ${pa.pitcher}` : '타석 목록'}</span>
          <small>{pa ? pa.result : ''} ▾</small>
        </button>
        <button className="sv-icon-btn" onClick={() => stepPa(1)} disabled={paIdx < 0 || paIdx >= detail.pas.length - 1} aria-label="다음 타석">›</button>
      </div>
      <div className="sv-pitches" role="group" aria-label="이 타석의 투구">
        {paPitches.map(p => <button key={p.index} className={`sv-pitch ${!latest && pitch?.index === p.index ? 'on' : ''}`} aria-pressed={!latest && pitch?.index === p.index}
          onClick={() => pick(p.index)} aria-label={`${p.pitch_number}구 ${p.actual?.pitch_label ?? ''} ${p.actual?.result_label ?? ''}`}>{p.pitch_number}</button>)}
        {nextPhase && nextNumber && paKey === detail.current?.pa_key && <button className="sv-pitch next on" aria-pressed="true" aria-label={`다음 공 ${nextNumber}구 추천 보기`} onClick={goLatest}>다음</button>}
        <span className="sv-spacer" />
        <button className="sv-btn" onClick={() => setSheet('detail')} aria-haspopup="dialog">자세히</button>
      </div>
    </nav>

    {sheet === 'pa' && <PaSheet pas={detail.pas} selected={paKey} onClose={() => setSheet(null)}
      onPick={p => { setSheet(null); if ((live && p.pa_key === detail.current?.pa_key && !p.complete) || typeof p.first !== 'number' || !byIndex.has(p.first)) goLatest(); else pick(p.first); }} />}
    {sheet === 'detail' && <DetailSheet onClose={() => setSheet(null)} rec={rec} recTitle={rec ? `${recTitle} (${provenanceText(rec)})` : recTitle}
      actual={actual} actualTitle={actualTitle} paResult={ending} notes={detail.notes} speed={mph => speedText(mph, unit)}
      extra={<div className="sv-row"><button className="sv-btn" onClick={() => setSheet('info')}>기준과 한계 보기</button>
        <button className="sv-btn" onClick={() => { const u = unit === 'kmh' ? 'mph' : 'kmh'; setUnit(u); try { localStorage.setItem(UNIT_KEY, u); } catch { /* ignore */ } }}
          aria-label={`속도 단위 바꾸기, 지금 ${unit === 'kmh' ? 'km/h' : 'mph'}`}>속도 단위: {unit === 'kmh' ? 'km/h' : 'mph'}</button></div>} />}
    {sheet === 'info' && <InfoSheet onClose={() => setSheet(null)} notes={detail.notes} />}
  </div>;
}

function TopBar({ right, title, label }: { right?: ReactNode; title?: string; label?: string }) {
  const demo = new URLSearchParams(window.location.search).get('demo');
  return <header className="sv-top">
    <a className="sv-back" href={demo ? `/watch?demo=${encodeURIComponent(demo)}` : '/watch'} aria-label="경기 목록으로">‹</a>
    {title ? <h1 className="sv-title" aria-label={label}>{title}</h1> : <a className="sv-brand small" href="/watch">Pitcheezy<span>.</span></a>}
    <span className="sv-top-right">{right}</span>
  </header>;
}
