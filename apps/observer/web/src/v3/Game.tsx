import { useEffect, useRef, useState } from 'react';
import type { Detail, Pa } from '../service/model';
import { short } from '../service/model';
import { useGameFeed } from './feed';
import { deriveView } from './view';
import type { Drawn, GameView } from './view';
import { useOptions } from './options';
import type { Options } from './options';
import { Bases, ConnBanner, Dock, Matchup, PitchList, PrevAb, RecBlock, Scoreboard } from './parts';
import type { Open, PanelKind } from './parts';
import Panel from './Panel';
import type { PanelMode } from './Panel';
import { ArsenalDetail, InfoDetail, PitchDetail, PlayerDetail, RecDetail, RecordDetail, WinProbDetail } from './details';
import { era, handText, stat3 } from './format';
import { listHref } from './links';

export type Width = 'phone' | 'tablet' | 'pc';
const widthOf = (w: number): Width => w >= 1180 ? 'pc' : w >= 720 ? 'tablet' : 'phone';
export function useWidth(): Width {
  const [w, setW] = useState(() => widthOf(window.innerWidth));
  useEffect(() => { const on = () => setW(widthOf(window.innerWidth)); window.addEventListener('resize', on); return () => window.removeEventListener('resize', on); }, []);
  return w;
}
type PanelState = { kind: PanelKind; pitch?: number; who?: 'batter' | 'pitcher'; pa?: string } | null;
const TITLES: Record<PanelKind, string> = { rec: '추천 자세히', pitch: '투구 자세히', player: '선수 기록', arsenal: '구종', record: '경기 기록', wp: '승리확률', info: '이 화면의 기준과 한계' };

export default function Game({ gamePk }: { gamePk: number }) {
  const feed = useGameFeed(gamePk);
  const opts = useOptions();
  const width = useWidth();
  const drawn = useRef<Drawn | null>(null);
  const [panel, setPanel] = useState<PanelState>(null);
  const open: Open = (kind, arg) => setPanel({ kind, ...arg });
  const d = feed.detail;
  useEffect(() => { if (d) document.title = `${short(d.game.away_team)} 대 ${short(d.game.home_team)} · 피치이지`; }, [d?.game.away_team, d?.game.home_team]);

  if (!d) {
    const e = feed.error;
    return <div className="v3 v3-page"><header className="v3-board"><a className="v3-back" href={listHref()} aria-label="경기 목록으로">‹</a><span className="v3-brand">Pitcheezy</span></header>
      <main className="v3-center">{e
        ? __PZ_STATIC__ && e.status === 404 ? <div className="v3-unavailable"><p role="status">이 경기는 지금 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.</p>
          <div className="v3-row"><button className="v3-btn" onClick={feed.askAgain}>다시 확인</button><a className="v3-btn ghost" href={listHref()}>경기 목록</a></div>
          {!feed.gaveUp && <p className="v3-muted small">올라오면 자동으로 보여 드려요.</p>}</div>
        : <div className="v3-alert" role="alert"><p>{e.status === 423 ? '이 경기는 지금 기록을 볼 수 없어요. 잠시 뒤 다시 확인해 주세요.' : e.text}</p>
          <div className="v3-row"><button className="v3-btn" onClick={feed.reload}>다시 시도</button><a className="v3-btn ghost" href={listHref()}>경기 목록</a></div>
          {feed.firstLoadRetry && <p className="v3-muted small">자동으로 다시 연결하고 있어요.</p>}</div>
        : <p className="v3-muted" role="status">경기 기록을 불러오는 중…</p>}</main></div>;
  }
  const v = deriveView(feed, d, drawn);
  const mode: PanelMode = width === 'pc' ? 'dock' : width === 'tablet' ? 'overlay' : 'sheet';
  const content = panel && <Panel mode={mode} title={panel.kind === 'pitch' ? TITLES.pitch : TITLES[panel.kind]} onClose={() => setPanel(null)}>
    {panel.kind === 'rec' && <RecDetail v={v} opts={opts} />}
    {panel.kind === 'pitch' && panel.pitch !== undefined && <PitchDetail v={v} opts={opts} index={panel.pitch} open={open} />}
    {panel.kind === 'player' && <PlayerDetail key={panel.who} v={v} layout={opts.stats} initial={panel.who ?? 'batter'} />}
    {panel.kind === 'arsenal' && <ArsenalDetail v={v} />}
    {panel.kind === 'record' && <RecordDetail v={v} open={open} focusPa={panel.pa} />}
    {panel.kind === 'wp' && <WinProbDetail />}
    {panel.kind === 'info' && <InfoDetail notes={d.notes} />}
  </Panel>;
  const cls = `v3 v3-game ${width} d-${opts.density} l-${opts.list}${panel && mode === 'dock' ? ' docked' : ''}`;
  if (width === 'phone') return <div className={cls}>
    <Scoreboard v={v} /><ConnBanner v={v} busy={feed.busy} onRetry={feed.reload} />
    <main className="v3-main">
      <Matchup v={v} density={opts.density} open={open} />
      <RecBlock v={v} opts={opts} open={open} side={opts.list === 'side' ? <PitchList v={v} layout="side" open={open} density={opts.density} /> : undefined} />
      {opts.list !== 'side' && <PitchList v={v} layout={opts.list} open={open} density={opts.density} />}
      {opts.density === 'full' && <PrevAb v={v} open={open} />}
      <InfoLink open={open} />
    </main>
    <Dock open={open} />
    {content}
  </div>;
  // tablet: two columns (recommendation | game); PC: players | recommendation | game, details docked on the right
  return <div className={cls}>
    <Scoreboard v={v} wide /><ConnBanner v={v} busy={feed.busy} onRetry={feed.reload} />
    <div className="v3-cols">
      {width === 'pc' && <PlayersCol v={v} open={open} />}
      <main className="v3-col v3-col-main">
        {width === 'tablet' && <Matchup v={v} density={opts.density} open={open} />}
        <RecBlock v={v} opts={opts} open={open} side={opts.list === 'side' ? <PitchList v={v} layout="side" open={open} density={opts.density} /> : undefined} />
        {opts.list !== 'side' && <PitchList v={v} layout={width === 'pc' && opts.list === 'row' ? 'stack' : opts.list} open={open} density={opts.density} />}
      </main>
      {!(panel && mode === 'dock') && <GameCol v={v} open={open} opts={opts} />}
      {mode === 'dock' && content}
    </div>
    {mode !== 'dock' && content}
  </div>;
}

function InfoLink({ open }: { open: Open }) {
  return <button className="v3-link v3-info-link" onClick={() => open('info')} aria-haspopup="dialog">이 화면의 기준과 한계</button>;
}

/** PC left column: batter and pitcher — season (pre-game) next to postseason, and today. */
function PlayersCol({ v, open }: { v: GameView; open: Open }) {
  const card = (who: 'batter' | 'pitcher') => {
    const p = who === 'batter' ? v.batter : v.pitcher;
    const info = p ? v.detail.players?.[String(p.id)] : undefined;
    const rows = who === 'batter' ? [['타율', stat3(info?.avg)], ['OPS', stat3(info?.ops)], ['wRC+', '—']] : [['ERA', era(info?.era)], ['WHIP', '—'], ['탈삼진율', '—']];
    const t = who === 'batter' ? v.batterToday : v.pitcherToday;
    const n = (x: number | null | undefined) => typeof x === 'number' ? x : '–';
    return <section className="v3-pcard" aria-label={who === 'batter' ? '타자' : '투수'}>
      <div className="v3-pcard-head"><small>{who === 'batter' ? '타자' : '투수'}</small><b>{p?.name ?? '—'}</b><span>{handText(who === 'batter' ? v.batter?.side : v.pitcher?.hand, who === 'batter' ? 'b' : 'p')}</span></div>
      <table className="v3-table mini"><thead><tr><th scope="col" /><th scope="col" className="num">정규</th><th scope="col" className="num">포스트</th></tr></thead>
        <tbody>{rows.map(([k, r]) => <tr key={k}><th scope="row">{k}</th><td className={`num${r === '—' ? ' na' : ''}`}>{r}</td><td className="num na">—</td></tr>)}</tbody></table>
      <p className="v3-today">오늘 {who === 'batter' ? `${n((t as GameView['batterToday'])?.at_bats)}타수 ${n((t as GameView['batterToday'])?.hits)}안타 ${n((t as GameView['batterToday'])?.on_base)}출루`
        : `${n((t as GameView['pitcherToday'])?.pitches)}구 ${n((t as GameView['pitcherToday'])?.strikeouts)}삼진`}</p>
      <button className="v3-link" onClick={() => open('player', { who })} aria-haspopup="dialog">전체 기록</button>
    </section>;
  };
  return <aside className="v3-col v3-col-players" aria-label="선수">{v.idle ? <p className="v3-muted">{v.breakNow ? '다음 타자를 기다리고 있어요' : '타석이 없어요'}</p> : <>{card('batter')}{card('pitcher')}</>}
    <button className="v3-link" onClick={() => open('arsenal')} aria-haspopup="dialog">{v.pitcher?.name ?? '투수'} 구종 보기</button></aside>;
}

/** Game column (tablet right, PC right): situation + linescore, recent plate appearances, win probability. */
function GameCol({ v, open, opts }: { v: GameView; open: Open; opts: Options }) {
  const d: Detail = v.detail;
  const ls = d.linescore_asof;
  const g = d.game;
  const s = v.situation;
  const recent: Pa[] = [...d.pas].filter(p => p.complete).slice(-5).reverse();
  return <aside className="v3-col v3-col-game" aria-label="경기">
    <section className="v3-gcard"><h3>경기 상황</h3>
      {ls ? <div className="v3-scroll"><table className="v3-table line compact"><thead><tr><th scope="col" />{Array.from({ length: Math.max(9, ls.innings) }, (_, i) => <th key={i} scope="col" className={`num${s?.inning === i + 1 ? ' cur' : ''}`}>{i + 1}</th>)}<th scope="col" className="num">R</th><th scope="col" className="num">H</th></tr></thead>
        <tbody>{(['away', 'home'] as const).map(k => <tr key={k}><th scope="row">{k === 'away' ? (g.away_abbr || short(g.away_team)) : (g.home_abbr || short(g.home_team))}</th>
          {Array.from({ length: Math.max(9, ls.innings) }, (_, i) => <td key={i} className="num">{ls[k].innings[i] ?? ''}</td>)}<td className="num"><b>{ls[k].r ?? '–'}</b></td><td className="num">{ls[k].h ?? '–'}</td></tr>)}</tbody></table></div>
        : <p className="v3-muted small">이닝별 점수 —</p>}
      {s && <p className="v3-situ"><Bases mask={s.bases} size={26} /><span>{s.outs}아웃 · {s.balls}-{s.strikes}</span></p>}
      {opts.density === 'full' && <PrevAb v={v} open={open} />}
    </section>
    <section className="v3-gcard"><h3>최근 타석</h3>
      {recent.length ? <ol className="v3-recent">{recent.map(p => <li key={p.pa_key}><button onClick={() => open('pitch', { pitch: p.first })}><b>{p.batter}</b><span>{p.result}</span></button></li>)}</ol> : <p className="v3-muted small">아직 없어요.</p>}
      <button className="v3-link" onClick={() => open('record')} aria-haspopup="dialog">경기 기록 전체</button>
    </section>
    <section className="v3-gcard"><h3>승리확률</h3><p className="v3-muted small">아직 제공되지 않아요 (—).</p>
      <button className="v3-link" onClick={() => open('wp')} aria-haspopup="dialog">무엇을 보여 줄지</button></section>
    <button className="v3-link v3-info-link" onClick={() => open('info')} aria-haspopup="dialog">이 화면의 기준과 한계</button>
  </aside>;
}
