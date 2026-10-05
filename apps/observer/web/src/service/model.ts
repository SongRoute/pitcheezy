/** Service (ordinary viewer) data model — UI redesign 2026-10-05 (frontend-001).
 *  One normalized shape for live and finished games so both use the same screen. The backend contract
 *  (pm-jobs/20261005-ui-live-redesign/CONTRACT.md) is mapped onto this in api.ts only. */

export type Half = 'Top' | 'Bot';
export type Bounds = { bottom: number; top: number };
export type Person = { id: number; name: string | null };
export type Situation = { inning: number; half: Half; outs: number; balls: number; strikes: number; bases: number; home_score: number; away_score: number };
export type Candidate = { rank: number; pitch_type: string; pitch_label: string; zone_id: string | null; zone_label: string;
  target: { x: number; z: number } | null; probability: number | null };
/** A recommendation as it was recorded before the pitch. `missing` = no archived record; never filled after the fact. */
export type Provenance = 'captured_live' | 'asof_replay' | 'absent';
export type Rec = { status: 'ready' | 'unsupported' | 'missing'; reason: string | null; candidates: Candidate[]; provenance: Provenance; policy: string | null };
export type Actual = { pitch_type: string | null; pitch_label: string; description: string | null; result_label: string;
  event: string | null; event_label: string | null; play_text: string | null; speed_mph: number | null; x: number | null; z: number | null; zone_label: string;
  /** CV catcher setup, horizontal only, present only after the actual pitch (never height / zone cell). */
  setup_x: number | null };
export type PitchRow = { index: number; key: string; pa_key: string; at_bat_number: number; pitch_number: number;
  situation: Situation; batter: Person & { side: string }; pitcher: Person & { hand: string }; rec: Rec; actual: Actual | null;
  /** linescore as of just before this pitch (server `linescore_before`); null when the backend does not send it. */
  line?: Linescore | null };
export type GameStatus = 'scheduled' | 'live' | 'final' | 'postponed' | 'other';
export type GameCard = { game_pk: number; start: string | null; kst_date: string; status: GameStatus; status_text: string;
  away_team: string; home_team: string; away_score: number | null; home_score: number | null; inning: number | null; half: Half | null;
  game_number: number; double_header: boolean; embargoed: boolean;
  /** rev3 optional (api-rev3-031): venue name from the feed; absent → not shown. */
  venue?: string | null; away_abbr?: string | null; home_abbr?: string | null };
export type Phase = 'pregame' | 'pre_pitch' | 'resolving' | 'between_innings' | 'final';
export type Current = { phase: Phase; situation: Situation; batter: (Person & { side: string }) | null; pitcher: (Person & { hand: string }) | null; pa_key: string | null };
export type Pa = { pa_key: string; at_bat_number: number; inning: number; half: Half; batter: string; pitcher: string;
  first: number; last: number; complete: boolean; result: string; batter_id?: number | null };
export type Detail = { game: GameCard; pitches: PitchRow[]; pas: Pa[]; current: Current | null; next: Rec | null; cutoff: number | null; session: string | null;
  feed: { as_of: string | null; ok: boolean; stale_s: number | null }; zone_bounds: Bounds; notes: string[];
  /** rev3 optional: half-innings computed from plays published up to the cutoff only; null cell = not reached yet. */
  linescore_asof?: Linescore | null;
  /** rev3 optional: per-player game jersey and pre-game (as-of) season stats; null stat = unknown → '–', never a current-season value. */
  players?: Record<string, PlayerInfo> | null };
export type LineTeam = { innings: (number | null)[]; r: number | null; h: number | null; e: number | null; bb_hbp: number | null };
export type Linescore = { innings: number; away: LineTeam; home: LineTeam };
export type PlayerInfo = { jersey?: string | null; era?: number | null; avg?: number | null; ops?: number | null; stats_as_of?: string | null };

const TEAMS: Record<string, string> = {
  'Arizona Diamondbacks': '애리조나', 'Athletics': '애슬레틱스', 'Atlanta Braves': '애틀랜타', 'Baltimore Orioles': '볼티모어',
  'Boston Red Sox': '보스턴', 'Chicago Cubs': '컵스', 'Chicago White Sox': '화이트삭스', 'Cincinnati Reds': '신시내티',
  'Cleveland Guardians': '클리블랜드', 'Colorado Rockies': '콜로라도', 'Detroit Tigers': '디트로이트', 'Houston Astros': '휴스턴',
  'Kansas City Royals': '캔자스시티', 'Los Angeles Angels': '에인절스', 'Los Angeles Dodgers': '다저스', 'Miami Marlins': '마이애미',
  'Milwaukee Brewers': '밀워키', 'Minnesota Twins': '미네소타', 'New York Mets': '메츠', 'New York Yankees': '양키스',
  'Philadelphia Phillies': '필라델피아', 'Pittsburgh Pirates': '피츠버그', 'San Diego Padres': '샌디에이고',
  'San Francisco Giants': '샌프란시스코', 'Seattle Mariners': '시애틀', 'St. Louis Cardinals': '세인트루이스',
  'Tampa Bay Rays': '탬파베이', 'Texas Rangers': '텍사스', 'Toronto Blue Jays': '토론토', 'Washington Nationals': '워싱턴' };
export const short = (team: string) => TEAMS[team] ?? team.split(' ').slice(-1)[0];
export const halfText = (half: Half) => half === 'Top' ? '초' : '말';
export const inningText = (s: { inning: number; half: Half }) => `${s.inning}회 ${halfText(s.half)}`;

const KST = 'Asia/Seoul';
/** KST calendar date (YYYY-MM-DD) of an instant. The game's date comes from its ORIGINAL start, never from today. */
export const kstDate = (iso: string) => new Intl.DateTimeFormat('en-CA', { timeZone: KST }).format(new Date(iso));
export const kstTime = (iso: string | null) => iso ? new Intl.DateTimeFormat('ko-KR', { timeZone: KST, hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(iso)) : '시간 미정';
export const todayKst = (now = Date.now()) => kstDate(new Date(now).toISOString());
export function dateHeading(ymd: string) {
  const [y, m, d] = ymd.split('-').map(Number);
  const weekday = '일월화수목금토'[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
  return `${y}년 ${m}월 ${d}일 (${weekday})`;
}
export const shiftDate = (ymd: string, days: number) => {
  const [y, m, d] = ymd.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d + days)).toISOString().slice(0, 10);
};
/** Doubleheader identity: same KST date and same two teams → '1차전'/'2차전' (source game_number, else start order). */
export function markDoubleheaders(games: GameCard[]): GameCard[] {
  const groups = new Map<string, GameCard[]>();
  for (const g of games) { const k = `${g.kst_date}|${[g.away_team, g.home_team].sort().join('|')}`; groups.set(k, [...(groups.get(k) ?? []), g]); }
  return games.map(g => {
    const group = groups.get(`${g.kst_date}|${[g.away_team, g.home_team].sort().join('|')}`)!;
    if (group.length < 2 && !g.double_header) return g;
    const ordered = [...group].sort((a, b) => (a.start ?? '').localeCompare(b.start ?? '') || a.game_pk - b.game_pk);
    const sourceNumbers = new Set(group.map(x => x.game_number));
    const number = sourceNumbers.size === group.length && g.game_number > 0 ? g.game_number : ordered.indexOf(g) + 1;
    return { ...g, double_header: true, game_number: number };
  });
}

const POSITION: Record<string, string> = { pitcher: '투수', catcher: '포수', 'first baseman': '1루수', 'second baseman': '2루수',
  'third baseman': '3루수', shortstop: '유격수', 'left fielder': '좌익수', 'center fielder': '중견수', 'right fielder': '우익수' };
const FIELD: Record<string, string> = { 'left field': '좌익수', 'center field': '중견수', 'right field': '우익수',
  'left-center field': '좌중간', 'right-center field': '우중간', 'left center field': '좌중간', 'right center field': '우중간' };
const EVENT: Record<string, string> = { single: '1루타', double: '2루타', triple: '3루타', home_run: '홈런', walk: '볼넷', intent_walk: '고의4구',
  hit_by_pitch: '몸에 맞는 공', strikeout: '삼진', strikeout_double_play: '삼진 병살', field_out: '아웃', force_out: '포스 아웃',
  grounded_into_double_play: '병살타', double_play: '병살', sac_fly: '희생 플라이', sac_bunt: '희생 번트', field_error: '실책 출루',
  fielders_choice: '야수 선택', fielders_choice_out: '야수 선택 아웃', catcher_interf: '포수 방해', triple_play: '삼중살' };
const PLAY_KIND: [RegExp, string][] = [[/\bgrounds? out\b|\bgrounds into\b|\bground ball\b/, '땅볼'], [/\bflies out\b|\bfly ball\b|\bflies into\b/, '플라이'],
  [/\blines out\b|\bline drive\b|\blines into\b/, '직선타'], [/\bpops out\b|\bpop up\b/, '뜬공'], [/\bbunt\b/, '번트']];

/** Detailed plate-appearance result built ONLY from source fields (event code + MLB play text).
 *  A fielder or direction is named only when the play text names it; otherwise the coarser event label stays. */
export function detailResult(a: Pick<Actual, 'event' | 'event_label' | 'play_text' | 'description'>): string {
  const text = (a.play_text ?? '').toLowerCase(), event = a.event ?? '';
  const base = EVENT[event] ?? a.event_label ?? '';
  if (event === 'strikeout' || event === 'strikeout_double_play') {
    const how = /strikes out swinging/.test(text) || a.description === 'swinging_strike' || a.description === 'swinging_strike_blocked' ? '헛스윙 '
      : /called out on strikes/.test(text) || a.description === 'called_strike' ? '루킹 ' : '';
    return `${how}${base}`;
  }
  if (!text) return base || '결과 기록 없음';
  const kind = PLAY_KIND.find(([re]) => re.test(text))?.[1] ?? '';
  // first fielder named after "to"/"by" or the leading "<position> <name>" of the assist chain
  const fielderMatch = /(?:to|by|,)\s+(pitcher|catcher|first baseman|second baseman|third baseman|shortstop|left fielder|center fielder|right fielder)\b/.exec(text);
  const fieldMatch = /to (left[- ]center field|right[- ]center field|left field|center field|right field)\b/.exec(text);
  const where = fielderMatch ? POSITION[fielderMatch[1]] : fieldMatch ? FIELD[fieldMatch[1]] : '';
  if (['single', 'double', 'triple', 'home_run'].includes(event)) return where ? `${where} 방면 ${base}` : base;
  if (['field_out', 'force_out', 'sac_fly', 'double_play', 'grounded_into_double_play', 'fielders_choice_out', 'sac_bunt'].includes(event)) {
    const tail = event === 'field_out' ? '아웃' : base;
    return [where, kind, tail].filter(Boolean).join(' ') || base;
  }
  return base || a.event_label || '결과 기록 없음';
}

/** Plate appearances in game order, built from the backend pitch history (not a session cache). */
export function groupPas(pitches: PitchRow[], current: Current | null): Pa[] {
  const out: Pa[] = [];
  for (const p of pitches) {
    let pa = out[out.length - 1];
    if (!pa || pa.pa_key !== p.pa_key) {
      pa = { pa_key: p.pa_key, at_bat_number: p.at_bat_number, inning: p.situation.inning, half: p.situation.half,
        batter: p.batter.name ?? '타자 미상', pitcher: p.pitcher.name ?? '투수 미상', first: p.index, last: p.index, complete: false, result: '타석 진행 중' };
      out.push(pa);
    }
    pa.last = p.index;
    if (p.actual?.event) { pa.complete = true; pa.result = detailResult(p.actual); }
  }
  // A PA whose last pitch has no event is in progress only if it is the current one; otherwise the source simply has no event.
  for (const pa of out) if (!pa.complete && current?.pa_key !== pa.pa_key && pa !== out[out.length - 1]) pa.result = '결과 기록 없음';
  return out;
}

/** Plainer zone words (as WatchAlong.plainZone): '가운데 높이 중앙' → '한가운데', left/right in the pitcher's view (below). */
/** Zone words come from the server in the CATCHER's view ('왼쪽' = catcher's left). The screen is drawn from the pitcher's view
 *  (Zone.tsx VIEW_X), so left/right words are swapped for display; the zone id and coordinates are untouched. */
/** Korean names for Statcast codes the server's PITCH_LABELS lacks (it then sends the feed's English description or the bare code).
 *  Display only; an unknown code shows as the code itself, never an English full name. Approved by Root 2026-10-06 (CONTRACT-AUDIT §4). */
const EXTRA_PITCH_NAMES: Record<string, string> = { CS: '슬로 커브', FO: '포크볼', FA: '패스트볼', KN: '너클볼', EP: '이퓨스', SC: '스크루볼' };
export const pitchName = (label: string | null | undefined, code: string | null | undefined) =>
  label && /[가-힣]/.test(label) ? label : (code && EXTRA_PITCH_NAMES[code]) || code || label || '구종 미상';
export const pitcherSide = (label: string) => label.replace(/왼쪽|오른쪽/g, w => w === '왼쪽' ? '오른쪽' : '왼쪽');
export const plainZone = (label: string) => pitcherSide(label.replace('가운데 높이 중앙', '한가운데').replace('가운데 높이 ', '가운데 '));
export const policyShare = (p: number | null) => p === null ? '—' : `${Math.round(p * 100)}%`;
