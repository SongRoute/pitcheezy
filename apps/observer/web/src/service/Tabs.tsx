/** New bottom tabs (owner 11/22, plan 1): 예측 · 경기 · 선수 · 분석. This file holds the pieces the 경기 and 선수 tabs add:
 *  - win probability card (owner's own pick: team-colour area around 50%, dashed lines down to the x axis at the biggest passed changes,
 *    numbered in game order, one plain line per number underneath; nothing to tap to read it)
 *  - pitch-by-pitch log (recent first) with a neutral "which pick of the recommendation was this pitch type" note, only for pitches that
 *    carry a recorded recommendation
 *  - pitcher mix (today vs season, both against the current batter's side), batter season line, head-to-head line
 *  Data beside the page (J1, validated, anything wrong → that block is hidden): pz/we/we-<game>.json (per-pitch home win probability,
 *  frozen C0 model) and the host's pz/matchup/matchup-<game>.sheet.json. Only pitches already published are drawn (no future). */
import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { Candidate, Detail, PitchRow } from './model';
import { inningText, pitchKind, short, surname } from './model';
import Zone, { G1Dot } from './Zone';
import type { Scene, Scenes } from './Scenes';

// ---------- data files ----------
const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
async function getJson(rel: string): Promise<unknown> {
  const r = await fetch(new URL(rel, document.baseURI).href, { cache: 'no-cache', credentials: 'same-origin' });
  return r.ok ? r.json() : null;
}
export type WeData = { points: Map<string, [number, number]> };
function validWe(d: unknown, gamePk: number): WeData | null {
  if (!d || typeof d !== 'object') return null; const o = d as Record<string, unknown>;
  if (o.schema !== 'pz.we.v1' || o.game_pk !== gamePk || !o.points || typeof o.points !== 'object') return null;
  const m = new Map<string, [number, number]>();
  for (const [k, v] of Object.entries(o.points as Record<string, unknown>)) {
    if (!Array.isArray(v) || v.length !== 2 || !num(v[0]) || !num(v[1]) || v[0] < 0 || v[0] > 1 || v[1] < 0 || v[1] > 1) return null;
    m.set(k, [v[0], v[1]]);
  }
  return m.size ? { points: m } : null;
}
type Line = { pa: number; ab: number; h: number; hr: number; k: number; bb: number; avg: number; obp: number; slg: number; ops: number; from?: string; to?: string; n?: number };
type Mix = { side: string; n: number; from: string; to: string; list: { t: string; s: number }[] };
/** pre-pitch recommendations from the 5차전 data pack (pz/recs/recs-<game>.json): used only when its policy matches the live one */
export type PackRecs = { policyShort: string; recs: Map<string, Candidate[]>; scenePitch: Map<number, string> };
function validRecs(d: unknown, gamePk: number): PackRecs | null {
  if (!d || typeof d !== 'object') return null; const o = d as Record<string, unknown>;
  if (o.schema !== 'pz.recs.v1' || o.game_pk !== gamePk || typeof o.policy_short !== 'string' || !/^[0-9a-f]{8}$/.test(o.policy_short) || !o.recs || typeof o.recs !== 'object') return null;
  const m = new Map<string, Candidate[]>();
  for (const [k, rows] of Object.entries(o.recs as Record<string, unknown>)) {
    if (!Array.isArray(rows)) return null; const list: Candidate[] = [];
    for (const r of rows) { if (!Array.isArray(r) || r.length !== 8 || !num(r[0]) || typeof r[1] !== 'string' || typeof r[2] !== 'string' || !num(r[5]) || !num(r[6]) || !num(r[7])) return null;
      list.push({ rank: r[0], pitch_type: r[1], pitch_label: r[2], zone_id: typeof r[3] === 'string' ? r[3] : null, zone_label: typeof r[4] === 'string' ? r[4] : '', target: { x: r[5], z: r[6] }, probability: r[7] }); }
    m.set(k, list.sort((a, b) => a.rank - b.rank));
  }
  const sp = new Map<number, string>(); for (const [ab, k] of Object.entries((o.scene_pitch ?? {}) as Record<string, unknown>)) if (typeof k === 'string') sp.set(Number(ab), k);
  return { policyShort: o.policy_short, recs: m, scenePitch: sp };
}
/** the live policy's short identity, from any ready recommendation the publisher sent */
export function livePolicy(detail: Detail): string | null {
  const r = [detail.next, ...detail.pitches.map(p => p.rec)].find(r => r?.status === 'ready' && typeof r.policy_id === 'string');
  return r?.policy_id ? r.policy_id.slice(0, 8) : null;
}
/** a pitch's pre-pitch candidates: the published record when it is there, else the pack (same policy only) */
export function candidatesFor(p: PitchRow, pack: PackRecs | null, live: string | null): Candidate[] {
  if (p.rec?.status === 'ready' && p.rec.candidates.length) return p.rec.candidates;
  if (pack && live && pack.policyShort === live) return pack.recs.get(p.key) ?? [];
  return [];
}
export type Matchup = { h2h: Record<string, Line>; vh: Record<string, Line>; mix: Record<string, Mix> };
function validMatchup(d: unknown, gamePk: number): Matchup | null {
  if (!d || typeof d !== 'object') return null; const o = d as Record<string, unknown>;
  if (o.game_pk !== gamePk || typeof o.h2h !== 'object' || typeof o.vh !== 'object' || typeof o.mix !== 'object' || !o.h2h || !o.vh || !o.mix) return null;
  return { h2h: o.h2h as Record<string, Line>, vh: o.vh as Record<string, Line>, mix: o.mix as Record<string, Mix> };
}
const lineOk = (l: Line | undefined): l is Line => !!l && ['pa', 'ab', 'h', 'hr', 'k', 'bb', 'avg', 'obp', 'slg', 'ops'].every(k => num((l as Record<string, unknown>)[k]));
const mixOk = (m: Mix | undefined): m is Mix => !!m && num(m.n) && Array.isArray(m.list) && m.list.every(r => typeof r.t === 'string' && num(r.s) && r.s >= 0 && r.s <= 1);
export function useTabData(gamePk: number) {
  const [we, setWe] = useState<WeData | null>(null);
  const [mu, setMu] = useState<Matchup | null>(null);
  const [recs, setRecs] = useState<PackRecs | null>(null);
  useEffect(() => {
    let live = true;
    getJson(`pz/we/we-${gamePk}.json`).then(d => { if (live) setWe(validWe(d, gamePk)); }).catch(() => { if (live) setWe(null); });
    getJson(`pz/matchup/matchup-${gamePk}.sheet.json`).then(d => { if (live) setMu(validMatchup(d, gamePk)); }).catch(() => { if (live) setMu(null); });
    getJson(`pz/recs/recs-${gamePk}.json`).then(d => { if (live) setRecs(validRecs(d, gamePk)); }).catch(() => { if (live) setRecs(null); });
    return () => { live = false; };
  }, [gamePk]);
  return { we, mu, recs };
}

// ---------- win probability ----------
const VW = 340, VH = 200, PX = 6, PT = 22, PB = 30;  // viewBox and paddings (top room for labels, bottom for inning numbers + marker numbers)
type Pt = { x: number; y: number; p: number; pitch: PitchRow };
function weSeries(pitches: PitchRow[], we: WeData, vw = VW) {
  // x: each half-inning is one equal slot (18 for 9 innings, more in extra innings); inside it the pitches spread evenly
  const halves: { key: string; inning: number; half: string; list: PitchRow[] }[] = [];
  for (const p of pitches) { if (!we.points.has(p.key)) continue; const k = `${p.situation.inning}${p.situation.half}`;
    const last = halves[halves.length - 1]; if (last && last.key === k) last.list.push(p); else halves.push({ key: k, inning: p.situation.inning, half: p.situation.half, list: [p] }); }
  const slots = Math.max(18, halves.length ? (halves[halves.length - 1].inning * 2 - (halves[halves.length - 1].half === 'Top' ? 1 : 0)) : 18);
  const w = vw - 2 * PX, h = VH - PT - PB;
  const X = (inning: number, half: string, f: number) => PX + ((inning - 1) * 2 + (half === 'Top' ? 0 : 1) + f) / slots * w;
  const Y = (p: number) => PT + (0.06 + 0.88 * (1 - p)) * h;  // 0–100% inside 6–94% of the plot (review 11/22: never on the frame)
  const pts: Pt[] = [];
  for (const hv of halves) hv.list.forEach((p, i) => { const [b] = we.points.get(p.key)!; pts.push({ x: X(hv.inning, hv.half, i / hv.list.length), y: Y(b), p: b, pitch: p }); });
  // close each finished pitch with its "after" value at the end of its slot step; the last point = state after the newest pitch
  const after: Map<string, { x: number; y: number; p: number }> = new Map();
  pts.forEach((q, i) => { const nx = pts[i + 1]?.x ?? Math.min(PX + w - 6, q.x + w / slots / 4); const a = we.points.get(q.pitch.key)![1]; after.set(q.pitch.key, { x: nx, y: Y(a), p: a }); });
  return { pts, after, slots, X, Y, w, h, innings: Math.ceil(slots / 2) };
}
type Mark = { n: number; x: number; y: number; scene: Scene; key: string };
/** passed scenes, biggest change first, at most 5; when two lines sit closer than `gap` the smaller change goes (never fewer than 3) */
function pickMarks(scenes: Scenes, done: Set<number>, pitches: PitchRow[], after: Map<string, { x: number; y: number; p: number }>, gap = 24): Mark[] {
  const lastOf = new Map<number, PitchRow>(); for (const p of pitches) lastOf.set(p.at_bat_number, p);
  let c = scenes.scenes.filter(s => done.has(s.at_bat) && lastOf.has(s.at_bat) && after.has(lastOf.get(s.at_bat)!.key))
    .sort((a, b) => Math.abs(b.delta_home_pp) - Math.abs(a.delta_home_pp)).slice(0, 5)
    .map(s => { const a = after.get(lastOf.get(s.at_bat)!.key)!; return { n: 0, x: a.x, y: a.y, scene: s, key: lastOf.get(s.at_bat)!.key }; });
  for (;;) {
    const s = [...c].sort((a, b) => a.x - b.x); let worst: Mark | null = null;
    for (let i = 1; i < s.length; i++) if (s[i].x - s[i - 1].x < gap) { const pair = [s[i - 1], s[i]].sort((a, b) => Math.abs(a.scene.delta_home_pp) - Math.abs(b.scene.delta_home_pp)); worst = pair[0]; break; }
    if (!worst || c.length <= 3) break; c = c.filter(m => m !== worst);
  }
  return c.sort((a, b) => a.x - b.x).map((m, i) => ({ ...m, n: i + 1 }));
}
const pp1 = (v: number) => Math.abs(v).toFixed(1);
// owner 11/22 (경기 탭): each big scene shows how the game state changed — before the deciding pitch (score · outs · count · runners) → result → after
const Dia = ({ m }: { m: number }) => <svg className="f1-dia" viewBox="0 0 30 24" aria-hidden="true">{([[2, 15, 6], [1, 24, 14], [4, 6, 14]] as const).map(([bit, x, y]) =>
  <rect key={bit} x={x - 4.5} y={y - 4.5} width="9" height="9" transform={`rotate(45 ${x} ${y})`} className={(m & bit) ? 'on' : ''} />)}</svg>;
type St = { away: number; home: number; outs: number; bases: number; count?: string | null };
const StateChip = ({ st, away, home }: { st: St; away: string; home: string }) => <span className="f1-wp-st">
  <b className="f1-wp-sc">{away} {st.away}:{st.home} {home}</b><Dia m={st.bases} />
  <span className="f1-wp-outs" aria-label={`${st.outs}아웃`}>{st.outs >= 3 ? <em>3아웃</em> : [0, 1].map(i => <i key={i} className={i < st.outs ? 'on' : ''} />)}</span>
  {st.count ? <span className="f1-wp-cnt">{st.count}</span> : null}</span>;
function sceneStates(pitches: PitchRow[], atBat: number): { before: St; after: St | null; count: string } | null {
  // consultant 11/22: the win-probability change is measured from the start of the plate appearance (count not used), so the states are
  // plate-appearance start → plate-appearance end; the count of the PA's last pitch is written apart ("볼카운트 1-2에서", XVAL-UX 11/22: not 결정구)
  let first = -1, last = -1; pitches.forEach((p, i) => { if (p.at_bat_number === atBat) { if (first < 0) first = i; last = i; } }); if (last < 0) return null;
  const f = pitches[first].situation, a = pitches[last].situation, n = pitches[last + 1]?.situation;
  const before: St = { away: f.away_score, home: f.home_score, outs: f.outs, bases: f.bases };
  const count = `${a.balls}-${a.strikes}`;
  if (!n) return { before, after: null, count };
  const same = n.inning === a.inning && n.half === a.half;
  return { before, count, after: same ? { away: n.away_score, home: n.home_score, outs: n.outs, bases: n.bases } : { away: n.away_score, home: n.home_score, outs: 3, bases: 0 } };
}
export function WinProbCard({ detail, we, scenes, homeShort, awayShort, onAnalysis, onScene, abbr = null, vw = VW }: { detail: Detail; we: WeData; scenes: Scenes | null; homeShort: string; awayShort: string; onAnalysis: () => void; onScene?: (rank: number) => void; abbr?: { away: string; home: string } | null; vw?: number }) {
  const s = weSeries(detail.pitches, we, vw);  // vw: a wider drawing for a wide card (booth); same height
  if (s.pts.length < 2) return null;
  const done = new Set(detail.pas.filter(p => p.complete).map(p => p.at_bat_number));
  const marks = scenes ? pickMarks(scenes, done, detail.pitches, s.after) : [];
  const lastKey = s.pts[s.pts.length - 1].pitch.key; const end = s.after.get(lastKey)!;
  const now = typeof detail.win_expectancy?.home_probability === 'number' ? detail.win_expectancy.home_probability : end.p;
  const line = [...s.pts.flatMap((q, i) => { const a = s.after.get(q.pitch.key)!; return i === s.pts.length - 1 ? [[q.x, q.y], [a.x, a.y]] : [[q.x, q.y]]; })] as [number, number][];
  const d = 'M' + line.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L');
  const mid = s.Y(0.5), bottom = PT + s.h;
  const area = `${d} L${line[line.length - 1][0].toFixed(1)} ${mid} L${line[0][0].toFixed(1)} ${mid} Z`;
  const nowInning = s.pts[s.pts.length - 1].pitch.situation.inning;
  return <section className="f1-tabcard f1-wp" aria-label="승리확률 (모델 추정)">
    <div className="f1-tc-head"><h3>승리확률</h3><span className="f1-est">모델 추정</span></div>
    <p className="f1-wp-now"><span><i className="sw away" />{awayShort} <b>{Math.round((1 - now) * 100)}%</b></span><span><i className="sw home" />{homeShort} <b>{Math.round(now * 100)}%</b></span></p>
    <svg className="f1-wp-chart" viewBox={`0 0 ${vw} ${VH}`} role="img" aria-label={`홈 ${homeShort} 승리확률 흐름. 지금 ${Math.round(now * 100)}퍼센트.`}>
      <defs><clipPath id="f1-wp-top"><rect x="0" y="0" width={vw} height={mid} /></clipPath><clipPath id="f1-wp-bot"><rect x="0" y={mid} width={vw} height={VH} /></clipPath></defs>
      <rect className="f1-wp-frame" x={PX} y={PT} width={s.w} height={s.h} rx="6" />
      {Array.from({ length: s.innings - 1 }, (_, i) => { const x = PX + (i + 1) * 2 / s.slots * s.w; return <line key={i} className="f1-wp-grid" x1={x} x2={x} y1={PT} y2={bottom} />; })}
      <path className="f1-wp-area home" d={area} clipPath="url(#f1-wp-top)" /><path className="f1-wp-area away" d={area} clipPath="url(#f1-wp-bot)" />
      <line className="f1-wp-mid" x1={PX} x2={PX + s.w} y1={mid} y2={mid} />
      <text className="f1-wp-50" x={PX + s.w - 4} y={mid - 4} textAnchor="end">50%</text>
      {marks.map(m => <line key={m.key} className="f1-wp-drop" x1={m.x} x2={m.x} y1={m.y} y2={bottom} />)}
      <text className="f1-wp-side home" x={PX + 6} y={PT + 13}>{homeShort}(홈) 이김 ↑</text>
      <text className="f1-wp-side away" x={PX + 6} y={bottom - 6}>{awayShort}(원정) 이김 ↓</text>
      <path className="f1-wp-line" d={d} />
      {marks.map(m => <circle key={m.key + 'd'} className="f1-wp-hit" cx={m.x} cy={m.y} r="3.5" />)}
      <circle className="f1-wp-now-dot" cx={end.x} cy={end.y} r="4.5" />
      {Array.from({ length: s.innings }, (_, i) => <text key={i} className={`f1-wp-inn${i + 1 > nowInning ? ' later' : ''}`} x={PX + (i * 2 + 1) / s.slots * s.w} y={bottom + 12} textAnchor="middle">{i + 1}</text>)}
      {marks.map(m => <g key={m.key + 'n'} className="f1-wp-num"><circle cx={m.x} cy={bottom + 23} r="7.5" /><text x={m.x} y={bottom + 26.5} textAnchor="middle">{m.n}</text></g>)}
    </svg>
    {marks.length > 0 && <ol className="f1-wp-notes">{marks.map(m => { const dl = m.scene.delta_home_pp; const ch = sceneStates(detail.pitches, m.scene.at_bat);
      const body = <><i aria-hidden="true">{m.n}</i><span><b>{m.scene.inning} {surname(m.scene.batter) || m.scene.batter} {m.scene.result}</b>
        {ch && (ch.after || detail.game.status === 'final') && <span className="f1-wp-chg"><StateChip st={ch.before} away={abbr?.away ?? awayShort} home={abbr?.home ?? homeShort} /><span className="f1-wp-to"><span className="f1-wp-arrow" aria-hidden="true">→</span>{ch.after ? <StateChip st={ch.after} away={abbr?.away ?? awayShort} home={abbr?.home ?? homeShort} /> : <em>경기 끝</em>}</span><span className="f1-wp-cnt">볼카운트 {ch.count}에서</span></span>}
        <small className={dl < 0 ? 'down' : 'up'}>{dl < 0 ? awayShort : homeShort} 승리확률 {pp1(dl)}%p 올라감</small></span>{onScene ? <span className="f1-wp-go" aria-hidden="true">›</span> : null}</>;
      return <li key={m.key}>{onScene ? <button onClick={() => onScene(m.scene.rank)} aria-haspopup="dialog">{body}</button> : body}</li>; })}</ol>}
    {marks.length > 0 ? <button className="f1-tc-link" onClick={onAnalysis}>장면 더 보기 (상위 10) ›</button> : <p className="f1-wp-empty">큰 장면이 나오면 여기에 표시돼요.</p>}
    <p className="f1-tc-note">모델 추정이에요. 승리확률은 타석 시작·끝 상황으로 계산해 볼카운트는 넣지 않았어요. 지나온 장면 중 변화가 큰 곳만, 서로 가까운 장면은 하나만 표시해요.</p>
  </section>;
}

// ---------- pitch-by-pitch log ----------
function pickOf(p: PitchRow, cands: Candidate[]): string | null {
  if (!p.actual || !cands.length) return null;
  const c = cands.find(c => (p.actual!.pitch_type && c.pitch_type === p.actual!.pitch_type) || c.pitch_label === p.actual!.pitch_label);
  return c ? `예측 ${c.rank}순위 구종` : '예측에 없던 구종';
}
export function PitchLog({ detail, speed, onPick, pack, limit = 6 }: { detail: Detail; speed: (mph: number | null) => string; onPick: (index: number) => void; pack: PackRecs | null; limit?: number }) {
  const live = livePolicy(detail);
  const pickFor = (p: PitchRow) => pickOf(p, candidatesFor(p, pack, live));
  const [inn, setInn] = useState<string | null>(null);
  void limit;
  const groups: { key: string; list: PitchRow[] }[] = [];
  for (const p of detail.pitches) { const g = groups[groups.length - 1]; if (g && g.key === p.pa_key) g.list.push(p); else groups.push({ key: p.pa_key, list: [p] }); }
  // owner 11/22 (5): pick an inning → that inning's plate appearances (newest first); a pitch opens the 예측 page as predicted then
  const innings = [...new Set(groups.map(g => inningText(g.list[0].situation)))];
  const curInn = inn !== null && innings.includes(inn) ? inn : innings[innings.length - 1];
  const shown = [...groups].reverse().filter(g => inningText(g.list[0].situation) === curInn);
  const anyPick = detail.pitches.some(p => pickFor(p) !== null);
  if (!groups.length) return null;
  // owner 11/22 (4-1, LOG-b): one card per plate appearance — a small zone with the pitch numbers on the left, the pitch list on the right,
  // and under each pitch the state it was thrown in (runners · outs · count before that pitch)
  const runnerWord = (m: number) => { const r = [1, 2, 3].filter(b => (m & (1 << (b - 1))) !== 0); return r.length ? `주자 ${r.join('·')}루` : '주자 없음'; };
  return <section className="f1-tabcard f1-log f1-logb" aria-label="투구별 기록">
    <div className="f1-tc-head"><h3>투구별 기록</h3><span className="f1-est">공을 누르면 그때 예측</span></div>
    <div className="f1-inn-chips" role="tablist" aria-label="이닝">{innings.map(t => <button key={t} role="tab" aria-selected={t === curInn} onClick={() => setInn(t)}>{t}</button>)}</div>
    {anyPick && <p className="f1-tc-note f1-log-key">공마다 셋째 줄: 실제 던진 구종이 던지기 전 예측에서 몇 순위였는지예요. 맞고 틀림이 아니에요. 둘째 줄은 그 공을 던지기 전 상황(주자 · 아웃 · 볼-스트라이크)이에요.</p>}
    {shown.map(g => { const f = g.list[0]; const pa = detail.pas.find(p => p.pa_key === g.key);
      const marks = g.list.filter(p => p.actual?.x != null && p.actual?.z != null).map(p => ({ x: p.actual!.x!, z: p.actual!.z!, number: p.pitch_number, kind: pitchKind(p.actual) }));
      return <div className="f1-log-pa" key={g.key}>
        <p className="f1-log-head"><b>{surname(f.batter.name ?? '') || f.batter.name} vs {surname(f.pitcher.name ?? '') || f.pitcher.name}</b><span>{pa?.complete ? '' : '진행 중'}</span></p>
        <div className="f1-logb-body">
          <div className="f1-logb-zone" aria-hidden="true"><Zone mini bounds={detail.zone_bounds} candidates={[]} actual={null} actualLabel="" setupX={null} actuals={marks} /></div>
          <ol>{[...g.list].reverse().map(p => { const a = p.actual; const pk = pickFor(p); const st = p.situation; const end = pa?.complete && a?.event ? pa.result : null;
            return <li key={p.key}><button onClick={() => onPick(p.index)} aria-haspopup="dialog"
              aria-label={`${p.pitch_number}구 ${a?.pitch_label ?? '기록 없음'} ${a?.result_label ?? ''}. 던질 때 ${runnerWord(st.bases)}, ${st.outs}아웃, 볼 ${st.balls} 스트라이크 ${st.strikes}`}>
              <span className="f1-logb-line"><G1Dot kind={pitchKind(a)} n={p.pitch_number} /><b>{a?.pitch_label ?? '기록 없음'}</b><small>{a?.speed_mph != null ? speed(a.speed_mph) : ''}</small>
                <span className="f1-log-call">{end ?? a?.result_label ?? ''}</span></span>
              <span className="f1-logb-state" aria-hidden="true"><span className="f1-logb-st">던지기 전 · {runnerWord(st.bases)} · {st.outs}아웃 · {st.balls}-{st.strikes}</span>{anyPick && pk ? <em>{pk}</em> : null}</span></button></li>; })}</ol>
        </div></div>; })}
  </section>;
}

// ---------- players ----------
const pct = (v: number) => `${Math.round(v * 100)}%`;
const rate = (v: number) => v.toFixed(3).replace(/^0/, '');
const sideWord = (s: string | null | undefined) => s === 'L' ? '좌타자' : s === 'R' ? '우타자' : null;
const handWord = (h: string | null | undefined) => h === 'L' ? '좌투수' : h === 'R' ? '우투수' : null;
/** PIT-a (owner 11/23): prior record (gray) vs this game (black), all batters. Prior = L+R sheet rows combined (no exact all-batters field exists). */
export function PitcherMix({ detail, pitcherId, mu, upto }: { detail: Detail; pitcherId: number | null; side?: string | null; mu: Matchup | null; upto: number | null }) {
  if (pitcherId === null) return null;
  const today = detail.pitches.filter(p => p.pitcher?.id === pitcherId && p.actual?.pitch_label && (upto === null || p.index <= upto) && p.receipt?.status !== 'unconfirmed');
  const rows = mu ? (['R', 'L'] as const).map(sd => ({ sd, m: mu.mix[`${pitcherId}:${sd}`] })).filter((r): r is { sd: 'R' | 'L'; m: Mix } => mixOk(r.m) && r.m.n > 0) : [];
  const hasSeason = rows.length > 0;
  if (!today.length && !hasSeason) return null;
  const sn = rows.reduce((a, r) => a + r.m.n, 0);
  const sMap = new Map<string, number>(); rows.forEach(r => r.m.list.forEach(x => sMap.set(x.t, (sMap.get(x.t) ?? 0) + r.m.n * x.s)));
  sMap.forEach((v, k) => sMap.set(k, v / sn));
  const from = rows.map(r => r.m.from).filter(Boolean).sort()[0];
  const oneSide = rows.length === 1 ? sideWord(rows[0].sd) : null;
  const cnt = new Map<string, number>(); today.forEach(p => cnt.set(p.actual!.pitch_label, (cnt.get(p.actual!.pitch_label) ?? 0) + 1));
  const types = [...new Set([...[...cnt.entries()].sort((a, b) => b[1] - a[1]).map(e => e[0]), ...[...sMap.entries()].sort((a, b) => b[1] - a[1]).map(e => e[0])])];
  return <section className="f1-tabcard f1-mix" aria-label="구종 비율">
    <div className="f1-tc-head"><h3>구종 비율 · 이전 기록 vs 이 경기</h3></div>
    <p className="f1-mix-key">{hasSeason && <span><i className="k season" />이전 기록{oneSide ? `(${oneSide} 상대만)` : ''} {sn.toLocaleString()}구{from ? ` (${from}~이 경기 전${oneSide ? '' : ', 좌·우타 합침, 상위 구종만'})` : oneSide ? '' : ' (좌·우타 합침, 상위 구종만)'}</span>}<span><i className="k today" />이 경기 {today.length}구</span></p>
    <ul>{types.map(t => { const td = today.length ? (cnt.get(t) ?? 0) / today.length : null; const ss = sMap.get(t);
      return <li key={t}><b>{t}</b><span className="bars">{ss !== undefined && <span className="bar season"><i style={{ width: pct(ss) }} /></span>}{td !== null && <span className="bar today"><i style={{ width: pct(td) }} /></span>}</span>
        <span className="vals">{ss !== undefined && <small>{pct(ss)}</small>}{td !== null && <b>{pct(td)}</b>}</span></li>; })}</ul>
    {hasSeason && <p className="f1-tc-note">{oneSide ? `이전 기록 = 이 경기 전까지 이 투수가 ${oneSide}에게 던진 공이에요(반대쪽 기록 없음).` : '이전 기록 = 이 경기 전까지 좌·우타자 기록을 합친 값이에요. 상위 구종만 담겨 있어 드물게 던진 구종은 빠지거나 낮게 보일 수 있어요.'}</p>}
  </section>;
}
export function BatterSeason({ batterId, pitcherHand, mu }: { batterId: number | null; pitcherHand: string | null; mu: Matchup | null }) {
  const l = mu && batterId !== null && pitcherHand ? mu.vh[`${batterId}:${pitcherHand}`] : undefined;
  if (!lineOk(l) || !handWord(pitcherHand)) return null;
  return <section className="f1-tabcard f1-line" aria-label="시즌 기록">
    <div className="f1-tc-head"><h3>시즌 · {handWord(pitcherHand)} 상대</h3></div>
    <dl className="f1-stats">{([['타율', l.avg], ['출루율', l.obp], ['장타율', l.slg], ['OPS', l.ops]] as const).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{rate(v)}</dd></div>)}</dl>
    <p className="f1-line-sub">{l.pa}타석 · 홈런 {l.hr} · 삼진 {l.k} · 볼넷 {l.bb}</p>
    <p className="f1-tc-note">2026 정규시즌·포스트시즌, 이 경기 전까지예요.</p>
  </section>;
}
export function HeadToHead({ pitcherId, batterId, pitcherName, batterName, mu }: { pitcherId: number | null; batterId: number | null; pitcherName: string | null; batterName: string | null; mu: Matchup | null }) {
  const l = mu && pitcherId !== null && batterId !== null ? mu.h2h[`${pitcherId}:${batterId}`] : undefined;
  return <section className="f1-tabcard f1-h2h" aria-label="맞대결">
    <div className="f1-tc-head"><h3>맞대결</h3></div>
    <p className="f1-h2h-who">{pitcherName ?? '투수'} → {batterName ?? '타자'}</p>
    {lineOk(l) ? <>
      <p className="f1-h2h-big">{l.pa}타석 {l.ab}타수 {l.h}안타</p>
      <p className="f1-line-sub">볼넷 {l.bb} · 삼진 {l.k} · 홈런 {l.hr} · OPS {rate(l.ops)}</p>
      <p className="f1-tc-note">{l.from && l.to ? `${l.from}~${l.to} · ` : ''}이 경기 전까지{l.pa < 10 ? ` · 표본이 작아요(${l.pa}타석)` : ''}</p>
    </> : <p className="f1-line-sub">이 둘의 지난 맞대결 기록이 없어요.</p>}
  </section>;
}
// ---------- owner 11/22 (4-2): PIT-b · PIT-c · BAT-a list · BAT-b splits · today's plate appearances vs this pitcher ----------
const TYPE_COLORS = ['#2f6b12', '#c8961c', '#2b6cb0', '#7b4bb0', '#3d8a8a', '#d1709a', '#8a6410', '#5b6b60'];
const todayOf = (detail: Detail, pitcherId: number, upto: number | null) => detail.pitches.filter(p => p.pitcher?.id === pitcherId && p.actual?.pitch_label
  && (upto === null || p.index <= upto) && p.receipt?.status !== 'unconfirmed');
const typesBy = (list: PitchRow[]) => { const c = new Map<string, number>(); list.forEach(p => c.set(p.actual!.pitch_label, (c.get(p.actual!.pitch_label) ?? 0) + 1));
  return [...c.entries()].sort((a, b) => b[1] - a[1]).map(([t, n], i) => ({ t, n, color: TYPE_COLORS[i % TYPE_COLORS.length] })); };
/** PIT-b: where today's pitches went, one colour per pitch type (pitcher's view, same drawing frame as the main zone) */
export function PitchMap({ detail, pitcherId, upto }: { detail: Detail; pitcherId: number | null; upto: number | null }) {
  if (pitcherId === null) return null;
  const list = todayOf(detail, pitcherId, upto).filter(p => p.actual!.x != null && p.actual!.z != null); if (!list.length) return null;
  const types = typesBy(list); const col = new Map(types.map(t => [t.t, t.color]));
  const b = detail.zone_bounds; const W = 240, H = 260, X = (x: number) => W / 2 - x * 52, Z = (z: number) => H - 20 - (z - 0.5) * 52;  // pitcher's view: catcher's +x drawn left
  const half = 0.708;
  return <section className="f1-tabcard f1-pmap" aria-label="이 경기 구종별 위치">
    <div className="f1-tc-head"><h3>이 경기 구종별 위치 · 전체 타자 {list.length}구</h3><span className="f1-est">투수 시점</span></div>
    <svg viewBox={`0 0 ${W} ${H}`} className="f1-pmap-svg" role="img" aria-label={`이 경기 ${list.length}구의 위치`}>
      <rect x={X(half)} y={Z(b.top)} width={X(-half) - X(half)} height={Z(b.bottom) - Z(b.top)} className="f1-pmap-zone" />
      {[1, 2].map(i => <line key={'v' + i} x1={X(half) + (X(-half) - X(half)) * i / 3} x2={X(half) + (X(-half) - X(half)) * i / 3} y1={Z(b.top)} y2={Z(b.bottom)} className="f1-pmap-grid" />)}
      {[1, 2].map(i => <line key={'h' + i} y1={Z(b.top) + (Z(b.bottom) - Z(b.top)) * i / 3} y2={Z(b.top) + (Z(b.bottom) - Z(b.top)) * i / 3} x1={X(half)} x2={X(-half)} className="f1-pmap-grid" />)}
      {[...list].sort((a, c) => (types.findIndex(t => t.t === a.actual!.pitch_label)) - (types.findIndex(t => t.t === c.actual!.pitch_label))).map(p => { const x = Math.max(8, Math.min(W - 8, X(p.actual!.x!))), y = Math.max(8, Math.min(H - 8, Z(p.actual!.z!)));
        return <circle key={p.key} cx={x} cy={y} r="5" fill={col.get(p.actual!.pitch_label)} fillOpacity=".8" stroke="#fff" strokeWidth="1" />; })}
    </svg>
    <p className="f1-pmap-key">{types.map(t => <span key={t.t}><i style={{ background: t.color }} />{t.t} {t.n}</span>)}</p>
    <p className="f1-tc-note">이 경기에서 이 투수가 던진 공 {list.length}개 · 위치는 기록값이에요.</p>
  </section>;
}
/** PIT-c: today's speed range per pitch type (min–max bar, average tick) */
export function SpeedBands({ detail, pitcherId, upto, unit }: { detail: Detail; pitcherId: number | null; upto: number | null; unit: 'kmh' | 'mph' }) {
  if (pitcherId === null) return null;
  const list = todayOf(detail, pitcherId, upto).filter(p => p.actual!.speed_mph != null); if (!list.length) return null;
  const k = unit === 'kmh' ? 1.609344 : 1; const types = typesBy(list);
  const rows = types.map(t => { const v = list.filter(p => p.actual!.pitch_label === t.t).map(p => p.actual!.speed_mph! * k);
    return { ...t, lo: Math.min(...v), hi: Math.max(...v), avg: v.reduce((a, b) => a + b, 0) / v.length }; });
  const lo = Math.floor(Math.min(...rows.map(r => r.lo)) / 5) * 5, hi = Math.ceil(Math.max(...rows.map(r => r.hi)) / 5) * 5, span = Math.max(5, hi - lo);
  const fmt = (v: number) => unit === 'kmh' ? `${Math.round(v)}` : v.toFixed(1);
  return <section className="f1-tabcard f1-speed" aria-label="이 경기 구종별 구속">
    <div className="f1-tc-head"><h3>이 경기 구종별 구속</h3><span className="f1-est">{unit === 'kmh' ? 'km/h' : 'mph'}</span></div>
    <ul className="f1-speed-list">{rows.map(r => <li key={r.t}><span className="f1-speed-name"><i style={{ background: r.color }} />{r.t}</span>
      <span className="f1-speed-bar" aria-label={`${r.t} ${fmt(r.lo)}~${fmt(r.hi)}, 평균 ${fmt(r.avg)}`}>
        <i style={{ left: `${(r.lo - lo) / span * 100}%`, width: `${Math.max(1.5, (r.hi - r.lo) / span * 100)}%`, background: r.color }} />
        <b style={{ left: `${(r.avg - lo) / span * 100}%` }} /></span>
      <span className="f1-speed-val">{fmt(r.lo)}–{fmt(r.hi)}</span></li>)}</ul>
    <p className="f1-speed-axis"><span>{lo}</span><span>{hi}</span></p>
    <p className="f1-tc-note">막대 = 이 경기 가장 느린 공~가장 빠른 공 · 세로 선 = 평균</p>
  </section>;
}
/** BAT-a as a list: this batter's plate appearances today, in order */
export function BatterToday({ detail, batterId, batterName }: { detail: Detail; batterId: number | null; batterName: string | null }) {
  const pas = detail.pas.filter(p => p.complete && (batterId !== null && p.batter_id != null ? p.batter_id === batterId : p.batter === batterName));
  return <section className="f1-tabcard f1-today" aria-label="이 경기 타석">
    <div className="f1-tc-head"><h3>이 경기 타석</h3><span className="f1-est">{pas.length}타석</span></div>
    {pas.length ? <ol className="f1-today-list">{pas.map(p => <li key={p.pa_key}><span>{inningText(p)}</span><b>{p.result}</b><small>{surname(p.pitcher) || p.pitcher}{p.pitch_count ? ` · ${p.pitch_count}구` : ''}</small></li>)}</ol>
      : <p className="f1-line-sub">아직 끝난 타석이 없어요.</p>}
  </section>;
}
/** BAT-b: the same four rates, then plate appearances · HR · K · BB, for each base the data has (season / vs hand / vs this pitcher) */
export function BatterSplits({ batterId, pitcherId, pitcherHand, pitcherName, season, mu, opponent }: { batterId: number | null; pitcherId: number | null; pitcherHand: string | null;
  pitcherName: string | null; season: { avg: number | null; ops: number | null } | null; mu: Matchup | null; opponent: string }) {
  const vh = mu && batterId !== null && pitcherHand ? mu.vh[`${batterId}:${pitcherHand}`] : undefined;
  const hh = mu && batterId !== null && pitcherId !== null ? mu.h2h[`${pitcherId}:${batterId}`] : undefined;
  void season;  // season AVG/OPS stay in the player sheet's 시즌 line (Fable SONG4 MUST 4), not repeated in this table
  const rows: { k: string; r: (string)[]; c: (string)[] }[] = [];
  if (lineOk(vh) && handWord(pitcherHand)) rows.push({ k: `${handWord(pitcherHand)} 상대`, r: [rate(vh.avg), rate(vh.obp), rate(vh.slg), rate(vh.ops)], c: [String(vh.pa), String(vh.hr), String(vh.k), String(vh.bb)] });
  if (lineOk(hh)) rows.push({ k: `${surname(pitcherName ?? '') || '이 투수'} 상대 (${hh.pa}타석)`, r: [rate(hh.avg), rate(hh.obp), rate(hh.slg), rate(hh.ops)], c: [String(hh.pa), String(hh.hr), String(hh.k), String(hh.bb)] });
  if (!rows.length) return null;
  const table = (head: string[], pick: (x: typeof rows[number]) => string[]) => <table className="f1-split"><thead><tr><th scope="col">기준</th>{head.map(h => <th key={h} scope="col">{h}</th>)}</tr></thead>
    <tbody>{rows.filter(x => pick(x).length).map(x => <tr key={x.k}><th scope="row">{x.k}</th>{pick(x).map((v, i) => <td key={i}>{v}</td>)}</tr>)}</tbody></table>;
  return <section className="f1-tabcard f1-line" aria-label="기준별 성적">
    <div className="f1-tc-head"><h3>기준별 성적</h3></div>
    {table(['타율', '출루율', '장타율', 'OPS'], x => x.r)}
    {table(['타석', '홈런', '삼진', '볼넷'], x => x.c)}
    <p className="f1-tc-note">모두 이 경기 전까지. 시즌 전체(타율·OPS)는 위 "시즌" 줄에 있어요. {handWord(pitcherHand) ?? '투수 손'} 상대 = 2026 정규+포스트시즌. 이 투수 상대 = 지난 맞대결 전체. {opponent} 상대 기록은 자료가 없어 넣지 않았어요.</p>
  </section>;
}
/** H2H-b: today's plate appearances between this pitcher and this batter, each with its pitches on a small zone */
export function HeadToHeadToday({ detail, pitcherId, batterId, speed }: { detail: Detail; pitcherId: number | null; batterId: number | null; speed: (mph: number | null) => string }) {
  if (pitcherId === null || batterId === null) return null;
  const list = detail.pitches.filter(p => p.pitcher?.id === pitcherId && p.batter?.id === batterId && p.actual);
  const groups: { key: string; list: PitchRow[] }[] = [];
  for (const p of list) { const g = groups[groups.length - 1]; if (g && g.key === p.pa_key) g.list.push(p); else groups.push({ key: p.pa_key, list: [p] }); }
  return <section className="f1-tabcard f1-logb" aria-label="이 경기 맞대결">
    <div className="f1-tc-head"><h3>이 경기 이 투수 vs 이 타자</h3><span className="f1-est">{groups.length}타석</span></div>
    {groups.length ? groups.map(g => { const f = g.list[0]; const pa = detail.pas.find(p => p.pa_key === g.key);
      const marks = g.list.filter(p => p.actual?.x != null && p.actual?.z != null).map(p => ({ x: p.actual!.x!, z: p.actual!.z!, number: p.pitch_number, kind: pitchKind(p.actual) }));
      return <div className="f1-log-pa" key={g.key}><p className="f1-log-head"><b>{inningText(f.situation)}</b><span>{pa?.complete ? '' : '진행 중'}</span></p>
        <div className="f1-logb-body"><div className="f1-logb-zone" aria-hidden="true"><Zone mini bounds={detail.zone_bounds} candidates={[]} actual={null} actualLabel="" setupX={null} actuals={marks} /></div>
          <ol>{[...g.list].reverse().map(p => <li key={p.key}><span className="f1-logb-line"><G1Dot kind={pitchKind(p.actual)} n={p.pitch_number} /><b>{p.actual?.pitch_label ?? '기록 없음'}</b>
            <small>{speed(p.actual?.speed_mph ?? null)}</small><span className="f1-log-call">{(pa?.complete && p.actual?.event ? pa.result : null) ?? p.actual?.result_label ?? ''}</span></span></li>)}</ol></div></div>; })
      : <p className="f1-line-sub">이 경기에서는 아직 이 둘이 만난 타석이 없어요.</p>}
  </section>;
}
/** small helper so GameScreen can label sides without repeating team logic */
export const teamShort = (name: string) => short(name);
export const TabNote = ({ children }: { children: ReactNode }) => <p className="f1-tc-note">{children}</p>;
