/** 주요 장면 분석 (owner 11/21 "화면에 넣자"; 11/22 D4-① 상위 10, D5-① 결과 카드와 함께). Data: pz/scenes/scenes-<game>.json beside the
 *  page (J1), made from EXP-WPA-DECOMP-02 attempt-001 and checked value by value against it. The file is validated here; anything wrong →
 *  nothing is shown (host condition #38). Display rule (fixed, VERIFY-FABLE): ③ shows value ± MC standard error; ①·② show the grey words
 *  "잡음 범위" with no number, arrow, colour or sign when NOISE. Only plate appearances that have already finished in the published game are
 *  opened; later ones stay locked, so a replay never tells the future. */
import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { Sheet } from './Sheets';

type Part = { display: 'SHOW' | 'NOISE'; pp: number; se_pp: number };
export type Scene = { rank: number; at_bat: number; inning: string; batter: string; pitcher: string; result: string; delta_home_pp: number; c1: Part; c2: Part; c3: Part; other_pp: number };
export type Scenes = { home: string; fixed_note: string; scenes: Scene[] };

const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const str = (v: unknown): v is string => typeof v === 'string' && v.length > 0 && v.length < 80;
const part = (v: unknown): v is Part => !!v && typeof v === 'object' && ((v as Part).display === 'SHOW' || (v as Part).display === 'NOISE') && num((v as Part).pp) && num((v as Part).se_pp) && (v as Part).se_pp >= 0;
function valid(d: unknown, gamePk: number): Scenes | null {
  if (!d || typeof d !== 'object') return null; const o = d as Record<string, unknown>;
  if (o.schema !== 'pz.scenes.v1' || o.game_pk !== gamePk || !str(o.home) || !str(o.fixed_note) || !Array.isArray(o.scenes) || o.scenes.length === 0 || o.scenes.length > 20) return null;
  const seen = new Set<number>();
  for (const s of o.scenes as Record<string, unknown>[]) {
    if (!s || !num(s.rank) || !Number.isInteger(s.at_bat) || !str(s.inning) || !str(s.batter) || !str(s.pitcher) || !str(s.result) || !num(s.delta_home_pp) || !num(s.other_pp)
      || !part(s.c1) || !part(s.c2) || !part(s.c3) || seen.has(s.at_bat as number)) return null;
    seen.add(s.at_bat as number);
  }
  return { home: o.home as string, fixed_note: o.fixed_note as string, scenes: [...(o.scenes as Scene[])].sort((a, b) => a.rank - b.rank) };
}
export function useScenes(gamePk: number, on: boolean): Scenes | null {
  const [data, setData] = useState<Scenes | null>(null);
  useEffect(() => {
    if (!on) { setData(null); return; }
    let live = true;
    fetch(new URL(`pz/scenes/scenes-${gamePk}.json`, document.baseURI).href, { cache: 'no-cache', credentials: 'same-origin' })
      .then(r => r.ok ? r.json() : null).then(d => { if (live) setData(valid(d, gamePk)); }).catch(() => { if (live) setData(null); });
    return () => { live = false; };
  }, [gamePk, on]);
  return data;
}

/** −7.1 / +4.0 with a real minus sign, one decimal (rounding only at display) */
export const pp = (v: number) => `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(1)}`;
const pm = (p: Part) => `${pp(p.pp)} ±${p.se_pp.toFixed(1)}%p`;

/** SA-1: the line under the result card. Opens the scene sheet. */
export function SceneLine({ scene, onOpen }: { scene: Scene; onOpen: () => void }) {
  return <button className="f1-scene-line" onClick={onOpen} aria-haspopup="dialog" aria-label={`주요 장면 ${scene.rank}위. 던진 뒤 ${pm(scene.c3)}. 자세히 보기`}>
    <span className="f1-scene-tag" aria-hidden="true">주요 장면</span><span className="f1-scene-val" aria-hidden="true">③ 던진 뒤 {pm(scene.c3)}</span><span className="f1-scene-go" aria-hidden="true">›</span></button>;
}

function Row({ n, label, sub, p }: { n: string; label: string; sub: string; p: Part }) {
  const noise = p.display === 'NOISE';
  return <div className={`f1-sc-row${noise ? ' noise' : ''}`}><div><b>{n} {label}</b><small>{sub}</small></div><span>{noise ? '구분 안 됨' : pm(p)}</span></div>;
}
/** SA-2: one scene */
export function SceneSheet({ data, scene, other, onClose, onList, card = null }: { data: Scenes; scene: Scene; other: boolean; onClose: () => void; onList: () => void; card?: ReactNode }) {
  return <Sheet title={`주요 장면 · ${scene.inning} ${scene.batter.split(' ').slice(-1)[0] || scene.batter}`} onClose={onClose} startFull>
    <div className="f1-sc">
      <p className="f1-sc-who">{scene.inning} · {scene.batter} vs {scene.pitcher}</p>
      <p className="f1-sc-head">{scene.result} · 홈({data.home}) 승리확률 {pp(scene.delta_home_pp)}%p</p>
      {card}
      <Row n="①" label="구종 선택" sub="어떤 공을 골랐나" p={scene.c1} />
      <Row n="②" label="공의 질 (위치 포함)" sub="얼마나 잘 던졌나" p={scene.c2} />
      <Row n="③" label="던진 뒤" sub="타격·수비·운" p={scene.c3} />
      {other && <div className="f1-sc-row noise"><div><b>그 밖</b><small>타석이 시작될 때 이미 달라진 부분</small></div><span>{pp(scene.other_pp)}%p</span></div>}
      <button className="f1-sc-list" onClick={onList}>지난 주요 장면 모두 보기 ›</button>
      <p className="f1-sc-fixed">{data.fixed_note}</p>
      <p className="f1-sc-small">"구분 안 됨" = 차이가 너무 작아 모델로는 가릴 수 없어요. ±는 계산을 여러 번 돌렸을 때 흔들리는 폭이에요. ①·②는 숫자를 빼서 ③과 그 밖을 더해도 전체와 꼭 같지는 않아요.</p>
    </div></Sheet>;
}
/** SA-3: the top-10 list; plate appearances not finished yet are locked */
export function SceneList({ data, done, onClose, onPick }: { data: Scenes; done: Set<number>; onClose: () => void; onPick: (s: Scene) => void }) {
  return <Sheet title={`주요 장면 분석 · 상위 ${data.scenes.length}`} onClose={onClose} startFull>
    <SceneListBody data={data} done={done} onPick={onPick} />
  </Sheet>;
}
/** 분석 tab (11/22): the model and limits links on top, then the same list */
export function AnalysisSheet({ data, done, onClose, onPick, onInfo }: { data: Scenes | null; done: Set<number>; onClose: () => void; onPick: (s: Scene) => void; onInfo: () => void }) {
  return <Sheet title="분석" onClose={onClose} startFull>
    <div className="f1-an-links"><button onClick={onInfo}>기준과 한계 · 모델 소개 ›</button></div>
    {data ? <><h3 className="f1-an-title">주요 장면 분석 · 상위 {data.scenes.length}</h3><SceneListBody data={data} done={done} onPick={onPick} /></> : <p className="f1-sc-small">주요 장면 자료를 불러오지 못했어요.</p>}
  </Sheet>;
}
function SceneListBody({ data, done, onPick }: { data: Scenes; done: Set<number>; onPick: (s: Scene) => void }) {
  return <>
    <p className="f1-sc-small">{data.home}(홈) 승리확률 변화가 큰 순(%p = 퍼센트포인트)</p>
    {/* F1 review (11/22): only scenes already played; the rest is one line (rank numbers stay, so the gaps say where they will go) */}
    <ol className="f1-sc-list">{data.scenes.filter(s => done.has(s.at_bat)).map(s =>
      <li key={s.rank}><button onClick={() => onPick(s)} aria-haspopup="dialog"><i>{s.rank}</i><span><b>{s.inning} {s.batter} · {s.result}</b></span><em>{pp(s.delta_home_pp)}%p</em></button></li>)}</ol>
    {data.scenes.some(s => !done.has(s.at_bat)) && <p className="f1-sc-rest">나머지 {data.scenes.filter(s => !done.has(s.at_bat)).length}장면은 그 타석이 지나면 열려요.</p>}
    <p className="f1-sc-small">{data.fixed_note}</p>
  </>;
}
