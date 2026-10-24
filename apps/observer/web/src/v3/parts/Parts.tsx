/** Element sheet (preview builds only, `#parts`): every v3 element in every state it can be in, side by side, drawn by the real
 *  components from real replay frames (frames.ts → normalizeDetail → deriveView). Used to refine one element at a time; alternatives
 *  under review sit next to the current one (alts.tsx). Nothing here is a measurement and nothing here ships publicly. */
import type { ReactNode } from 'react';
import { normalizeCard, normalizeDetail } from '../../service/api';
import type { Detail } from '../../service/model';
import { FINAL_FILE, liveBody } from '../replay/frames';
import type { Step } from '../replay/frames';
import { STEPS } from '../replay/control';
import { POLL_MS } from '../feed';
import type { Feed } from '../feed';
import { deriveView } from '../view';
import type { GameView } from '../view';
import type { Options } from '../options';
import { DEFAULTS } from '../options';
import { Candidates, ConnBanner, Dock, Matchup, PitchList, PrevAb, RecBlock, Scoreboard } from '../parts';
import type { Open } from '../parts';
import Zone from '../Zone';
import { zoneLabel, zonePitches } from '../parts';
import { ArsenalDetail, PitchDetail, PlayerDetail, RecDetail, RecordDetail } from '../details';
import { LiveCard, Row } from '../List';
import Series from '../Series';
import { ScoreboardB, ScoreboardC } from './alts';
import './parts.css';

const noop: Open = () => {};
const find = (f: (s: Step) => boolean) => Math.max(0, STEPS.findIndex(f));
const AT = {
  pregame: 0,
  mid: 30,   // Suzuki's 4th pitch, 2-1 — the same frame the canvas screenshots use
  check: find(s => s.inject === 'unconfirmed'),
  late: find(s => s.inject === 'feed_delayed'),
  fix: find(s => s.inject === 'correction'),
  lost: find(s => s.inject === 'disconnected'),
  none: find(s => s.inject === 'unsupported'),
  brk: find(s => s.kind === 'break'),
  final: STEPS.length - 1,
};
function feedOf(d: Detail, over: Partial<Feed> = {}): Feed {
  const now = Date.now();
  return { detail: d, error: null, lastOk: now, now, hidden: false, resumed: 0, fileStamp: null, health: null, checking: false, busy: false,
    live: d.game.status === 'live', waitingToStart: d.game.status === 'scheduled', pollMs: POLL_MS, firstLoadRetry: false, gaveUp: false,
    reload: () => {}, askAgain: () => {}, ...over };
}
const viewAt = (i: number, over?: Partial<Feed>): GameView => { const d = normalizeDetail(liveBody(STEPS[i])); return deriveView(feedOf(d, over), d, { current: null }); };
const archive = (): GameView => { const d = normalizeDetail(FINAL_FILE); return deriveView(feedOf(d), d, { current: null }); };

const V = {
  pregame: viewAt(AT.pregame), mid: viewAt(AT.mid), check: viewAt(AT.check), late: viewAt(AT.late), fix: viewAt(AT.fix),
  lost: viewAt(AT.lost), none: viewAt(AT.none), brk: viewAt(AT.brk), final: viewAt(AT.final),
  gone: viewAt(AT.mid, { error: { text: '연결 끊김', status: 0 } }), archive: archive(),
};
const O = (o: Partial<Options> = {}): Options => ({ ...DEFAULTS, ...o });

/** One sample: a caption, then the element in a phone-wide frame carrying the classes the real screen has. */
function Item({ title, note, wide = false, density = 'full', list = 'row', children }: { title: string; note?: string; wide?: boolean; density?: Options['density']; list?: Options['list']; children: ReactNode }) {
  return <figure className={`parts-item${wide ? ' wide' : ''}`}>
    <figcaption><b>{title}</b>{note && <span>{note}</span>}</figcaption>
    <div className={`v3 v3-game ${wide ? 'pc' : 'phone'} d-${density} l-${list} parts-frame`}>{children}</div>
  </figure>;
}
function Section({ id, n, title, look, children }: { id: string; n: string; title: string; look: string; children: ReactNode }) {
  return <section className="parts-sec" id={id} aria-labelledby={`${id}-h`}>
    <h2 id={`${id}-h`}><span>{n}</span>{title}</h2><p className="parts-look">{look}</p>{children}
  </section>;
}
const Pad = ({ children }: { children: ReactNode }) => <div className="parts-pad">{children}</div>;

const BOARD_STATES: [string, GameView][] = [['경기 전', V.pregame], ['타석 중', V.mid], ['공수 교대', V.brk], ['자료 늦음', V.late], ['연결 끊김', V.lost], ['경기 종료', V.final]];
const SECTIONS: [string, string][] = [['p1', '① 점수판'], ['p2', '② 타자·투수'], ['p3', '③ 존'], ['p4', '④ 추천 후보'], ['p5', '⑤ 이번 타석'],
  ['p6', '⑥ 상태'], ['p7', '⑦ 하단 메뉴'], ['p8', '⑧ 상세 시트'], ['p9', '⑨ 경기 목록']];

export default function Parts() {
  const v = V.mid;
  const b = v.detail.zone_bounds;
  const live = normalizeCard(liveBody(STEPS[AT.mid]).game), sched = normalizeCard(liveBody(STEPS[AT.pregame]).game), fin = normalizeCard(FINAL_FILE.game);
  const go = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  return <div className="parts">
    <header className="parts-head">
      <h1>요소 시트</h1>
      <p>실제 경기 849843 재생 화면에서 뽑은 요소를 상태별로 나란히 놓았어요. 요소 하나씩 다듬고 고른 뒤 화면에 합쳐요.</p>
      <nav aria-label="요소로 이동">{SECTIONS.map(([id, t]) => <button key={id} onClick={() => go(id)}>{t}</button>)}
        <a href="#game-990003">화면으로 돌아가기</a></nav>
    </header>

    <Section id="p1" n="①" title="점수판" look="이번 차례: 세 안 중 하나를 골라 주세요. 같은 정보, 같은 문구이고 배치와 강조만 달라요.">
      {([['A · 지금', (x: GameView) => <Scoreboard v={x} />, '어두운 막대, 두 줄. 점수 크게, 카운트·주자는 둘째 줄 오른쪽.'],
        ['B · 중계 바', (x: GameView) => <ScoreboardB v={x} />, '중계 화면처럼 두 팀을 위아래로. 공격 팀 표시, 카운트는 큰 숫자, 아웃은 주자 아래 점.'],
        ['C · 밝은 점수판', (x: GameView) => <ScoreboardC v={x} />, '흰 바탕을 그대로 잇고 아래 구분선 하나. 점수 가운데 크게, 나머지는 한 줄 글로.']] as const).map(([t, draw, note]) =>
        <div className="parts-alt" key={t}><h3>{t}<span>{note}</span></h3>
          <div className="parts-row">{BOARD_STATES.map(([s, x]) => <Item key={s} title={s}>{draw(x)}</Item>)}</div></div>)}
      <div className="parts-row"><Item title="지금 · PC (넓은 화면)" wide><Scoreboard v={v} wide /></Item></div>
    </Section>

    <Section id="p2" n="②" title="타자·투수" look="이름 줄. 기본(오늘 기록 포함)과 필수만, 교대·종료 때 바뀌는 문구.">
      <div className="parts-row">
        <Item title="기본"><Pad><Matchup v={v} density="full" open={noop} /></Pad></Item>
        <Item title="필수만" density="lite"><Pad><Matchup v={v} density="lite" open={noop} /></Pad></Item>
        <Item title="공수 교대"><Pad><Matchup v={V.brk} density="full" open={noop} /></Pad></Item>
        <Item title="경기 종료"><Pad><Matchup v={V.final} density="full" open={noop} /></Pad></Item>
      </div>
    </Section>

    <Section id="p3" n="③" title="존" look="추천 칸과 실제 공 표시. 표시 방식 A·B·C와 시점(투수 쪽 기본, 포수 쪽).">
      <div className="parts-row">
        {(['A', 'B', 'C'] as const).map(m => <Item key={m} title={`표시 ${m}${m === 'B' ? ' (기본)' : ''}`} note={m === 'A' ? '모든 공에 번호' : m === 'B' ? '직전 공만 번호' : '번호는 오른쪽 라벨'}>
          <Pad><div className="parts-zone"><Zone bounds={b} view="pitcher" mark={m} batter={v.side} recs={v.cands} pitches={zonePitches(v.paPitches)} label={zoneLabel(v, v.cands, v.paPitches, 'pitcher')} /></div></Pad></Item>)}
        <Item title="포수 쪽 시점"><Pad><div className="parts-zone"><Zone bounds={b} view="catcher" mark="B" batter={v.side} recs={v.cands} pitches={zonePitches(v.paPitches)} label={zoneLabel(v, v.cands, v.paPitches, 'catcher')} /></div></Pad></Item>
        <Item title="연결 끊김 (마지막 추천 흐리게)"><Pad><div className="parts-zone"><Zone bounds={b} view="pitcher" mark="B" batter={V.lost.side} recs={V.lost.cands} pitches={zonePitches(V.lost.paPitches)} held label="연결 끊김" /></div></Pad></Item>
        <Item title="확인 중인 공 (그리지 않음)"><Pad><div className="parts-zone"><Zone bounds={b} view="pitcher" mark="B" batter={V.check.side} recs={V.check.cands} pitches={zonePitches(V.check.paPitches)} label="확인 중" /></div></Pad></Item>
      </div>
    </Section>

    <Section id="p4" n="④" title="추천 후보" look="존 아래 세 칸. 위치 글자가 있는 기본과 구종만 있는 필수만.">
      <div className="parts-row">
        <Item title="기본"><Pad><Candidates v={v} density="full" open={noop} /></Pad></Item>
        <Item title="필수만" density="lite"><Pad><Candidates v={v} density="lite" open={noop} /></Pad></Item>
      </div>
    </Section>

    <Section id="p5" n="⑤" title="이번 타석" look="던진 공 목록. 가로 한 줄(기본)·여러 줄·존 옆, 확인 중인 공, 빈 타석.">
      <div className="parts-row">
        <Item title="가로 한 줄 (기본)"><Pad><PitchList v={v} layout="row" open={noop} density="full" /></Pad></Item>
        <Item title="여러 줄" list="stack"><Pad><PitchList v={v} layout="stack" open={noop} density="full" /></Pad></Item>
        <Item title="존 옆" list="side"><Pad><RecBlock v={v} opts={O({ list: 'side' })} open={noop} side={<PitchList v={v} layout="side" open={noop} density="full" />} /></Pad></Item>
        <Item title="확인 중인 공"><Pad><PitchList v={V.check} layout="row" open={noop} density="full" /></Pad></Item>
        <Item title="공수 교대 (빈 타석)"><Pad><PitchList v={V.brk} layout="row" open={noop} density="full" /></Pad></Item>
        <Item title="직전 타석 줄"><Pad><PrevAb v={v} open={noop} /></Pad></Item>
      </div>
    </Section>

    <Section id="p6" n="⑥" title="상태" look="연결 안내 띠와 추천 칸의 상태 문장. 같은 상태를 두 번 말하지 않는지 함께 봐 주세요.">
      <div className="parts-row">
        <Item title="안내 띠 · 자료 늦음"><ConnBanner v={V.late} busy={false} onRetry={() => {}} /></Item>
        <Item title="안내 띠 · 연결 끊김"><ConnBanner v={V.lost} busy={false} onRetry={() => {}} /></Item>
        <Item title="안내 띠 · 읽기 실패"><ConnBanner v={V.gone} busy={false} onRetry={() => {}} /></Item>
      </div>
      <div className="parts-row">
        {([['추천', V.mid], ['확인 중', V.check], ['자료 늦음', V.late], ['연결 끊김', V.lost], ['정정', V.fix], ['추천 없음', V.none], ['공수 교대', V.brk], ['경기 전', V.pregame], ['경기 종료', V.final], ['지난 경기 기록', V.archive]] as const)
          .map(([t, x]) => <Item key={t} title={`추천 칸 · ${t}`}><Pad><RecBlock v={x} opts={O()} open={noop} /></Pad></Item>)}
      </div>
    </Section>

    <Section id="p7" n="⑦" title="하단 메뉴" look="폰 화면 맨 아래 고정 메뉴.">
      <div className="parts-row"><Item title="기본"><Dock open={noop} /></Item></div>
    </Section>

    <Section id="p8" n="⑧" title="상세 시트" look="시트 머리, 탭, 표, 숫자 칸, 설명. 시트 다섯 종류를 같은 틀에서 비교해요.">
      <div className="parts-row">
        {([['추천 자세히', <RecDetail v={v} opts={O()} />], ['투구 자세히', v.lastPitch ? <PitchDetail v={v} opts={O()} index={v.lastPitch.index} open={noop} /> : null],
          ['선수 · 표', <PlayerDetail v={v} layout="table" initial="batter" />], ['선수 · 카드', <PlayerDetail v={v} layout="card" initial="batter" />],
          ['구종', <ArsenalDetail v={v} />], ['경기 기록', <RecordDetail v={v} open={noop} />]] as [string, ReactNode][]).map(([t, body]) =>
          <Item key={t} title={t}><div className="v3-panel parts-sheet"><div className="v3-panel-head"><h2>{t}</h2><button className="v3-btn ghost" type="button">닫기</button></div>
            <div className="v3-panel-body">{body}</div></div></Item>)}
      </div>
    </Section>

    <Section id="p9" n="⑨" title="경기 목록" look="진행 중 경기 카드, 예정·종료 경기 줄, 시리즈 줄.">
      <div className="parts-row">
        <Item title="진행 중 경기"><Pad><LiveCard g={live} /></Pad></Item>
        <Item title="경기 줄 · 예정·종료"><Pad><ul><Row g={sched} /><Row g={fin} /></ul></Pad></Item>
        <Item title="시리즈 줄"><Pad><Series /></Pad></Item>
      </div>
    </Section>
  </div>;
}
