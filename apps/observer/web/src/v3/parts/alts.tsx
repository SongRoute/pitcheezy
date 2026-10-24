/** Element alternatives under review (preview builds only, element sheet `#parts`). A chosen alternative moves into parts.tsx /
 *  v3.css and is deleted from here; nothing in this file ships in a public bundle. Same data and the same words as the current
 *  element: only layout and emphasis differ. */
import { inningText, short } from '../../service/model';
import type { GameView } from '../view';
import { Bases, ConnChip } from '../parts';

const abbrOf = (abbr: string | null | undefined, team: string) => abbr || short(team);
const inningOf = (v: GameView) => {
  const g = v.detail.game, s = v.situation;
  return v.final ? '경기 종료' : v.breakNow ? '공수 교대' : s ? inningText(s) : g.status === 'scheduled' ? '경기 전' : '';   // the same words as the current scoreboard
};
const label = (v: GameView, inning: string) => {
  const g = v.detail.game;
  return `${short(g.away_team)} ${v.awayScore ?? '–'} 대 ${v.homeScore ?? '–'} ${short(g.home_team)}${inning ? `, ${inning}` : ''}`;
};
function Outs({ n }: { n: number }) {
  return <span className="alt-outs" aria-hidden="true">{[0, 1].map(i => <i key={i} className={i < n ? 'on' : ''} />)}</span>;
}

/** B · 중계 바: teams stacked like a broadcast score bug — the team at bat is marked, the count is two big numerals, outs as dots
 *  under the bases. One dense dark bar. */
export function ScoreboardB({ v }: { v: GameView }) {
  const g = v.detail.game, s = v.final ? null : v.situation;
  const inning = inningOf(v);
  const bat = s ? (s.half === 'Top' ? 'away' : 'home') : null;
  const team = (k: 'away' | 'home') => <span className={`alt-b-team${bat === k ? ' bat' : ''}`}>
    <i aria-hidden="true" /><span>{k === 'away' ? abbrOf(g.away_abbr, g.away_team) : abbrOf(g.home_abbr, g.home_team)}</span>
    <b>{(k === 'away' ? v.awayScore : v.homeScore) ?? '–'}</b></span>;
  return <header className="v3-board alt-b">
    <span className="v3-back" aria-hidden="true">‹</span>
    <h1 className="alt-b-teams" aria-label={label(v, inning)}><span aria-hidden="true">{team('away')}{team('home')}</span></h1>
    <div className="alt-b-mid" aria-hidden="true"><span className="alt-b-inning">{inning}</span><ConnChip v={v} /></div>
    {s && <div className="alt-b-play" role="img" aria-label={`볼 ${s.balls}, 스트라이크 ${s.strikes}, ${s.outs}아웃`}>
      <span className="alt-b-bases"><Bases mask={s.bases} size={34} /><Outs n={s.outs} /></span>
      <span className="alt-b-count">{s.balls}<i>-</i>{s.strikes}</span>
    </div>}
  </header>;
}

/** C · 밝은 점수판: the flat white surface carries on into the header; one divider under it. The score is centred and large, the
 *  rest is a single line in words, with the bases icon. Only the live dot keeps colour. */
export function ScoreboardC({ v }: { v: GameView }) {
  const g = v.detail.game, s = v.final ? null : v.situation;
  const inning = inningOf(v);
  return <header className="alt-c">
    <span className="alt-c-back" aria-hidden="true">‹</span>
    <h1 className="alt-c-score" aria-label={label(v, inning)}><span aria-hidden="true">
      <span>{abbrOf(g.away_abbr, g.away_team)}</span><b>{v.awayScore ?? '–'}</b><i>:</i><b>{v.homeScore ?? '–'}</b><span>{abbrOf(g.home_abbr, g.home_team)}</span></span></h1>
    <span className="alt-c-chip"><ConnChip v={v} /></span>
    <p className="alt-c-line">
      <span>{inning}</span>
      {s && <><span>볼 {s.balls} 스트라이크 {s.strikes}</span><span>{s.outs}아웃</span><Bases mask={s.bases} size={24} /></>}
    </p>
  </header>;
}
