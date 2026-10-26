/** Preview replay: the published game file of a live game, rebuilt pitch by pitch from a real finished game's fixture.
 *  Every value comes from the fixture (real 849843 pitches, their as-of recommendations, results, positions); the replay only cuts the
 *  history at a pitch and adds the live-route keys the publisher writes (current, next, display_state, receipts). Injected states
 *  (확인 중, 자료 늦음, 연결 끊김, 정정, 공수 교대) are marked `synthetic` and use the server's own copy (display_state.py COPY).
 *  game_pk 990003 is the existing synthetic live id of tests/fixtures/build-publish-fixture.mjs. Nothing here is a measurement. */
import raw from './game-849843.json';

type Raw = Record<string, any>;
export const LIVE_PK = 990003;
export const ARCHIVE_PK = 849843;
const fx = raw as Raw;
export const FINAL_FILE = fx;
const P: Raw[] = fx.pitches;
export const PITCH_COUNT = P.length;

export type Inject = 'none' | 'unconfirmed' | 'feed_delayed' | 'disconnected' | 'correction' | 'unsupported';
export type Step = { k: number; kind: 'pregame' | 'pitch' | 'break' | 'final'; inject: Inject };

/** The replay script: pregame → for each pitch the pre-pitch frame (a half-inning change gets a 공수 교대 frame first) → final.
 *  A few frames carry one injected state so every state can be seen without waiting for a real outage. */
export function script(): Step[] {
  const out: Step[] = [{ k: 0, kind: 'pregame', inject: 'none' }];
  for (let k = 0; k < P.length; k++) {
    if (k > 0 && (P[k].inning !== P[k - 1].inning || P[k].half !== P[k - 1].half)) out.push({ k, kind: 'break', inject: 'none' });
    if (k > 0 && k % 11 === 4) out.push({ k, kind: 'pitch', inject: 'unconfirmed' });
    out.push({ k, kind: 'pitch', inject: k === 9 ? 'feed_delayed' : k === 17 ? 'correction' : k === 26 ? 'disconnected' : k === 40 ? 'unsupported' : 'none' });
  }
  out.push({ k: P.length, kind: 'final', inject: 'none' });
  return out;
}

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));
const situationOf = (p: Raw) => ({ ...p.situation_before, inning: p.inning, half: p.half,
  runners: { first: (p.situation_before.bases & 1) !== 0, second: (p.situation_before.bases & 2) !== 0, third: (p.situation_before.bases & 4) !== 0 } });

/** linescore as of pitch k: runs per half-inning from the score before each half and the score now (published plays only) */
function linescore(k: number) {
  const upTo = P.slice(0, Math.max(1, k));
  const last = k < P.length ? situationOf(P[k]) : { ...situationOf(P[P.length - 1]) };
  const starts = new Map<string, Raw>();
  for (const p of upTo) { const key = `${p.inning}:${p.half}`; if (!starts.has(key)) starts.set(key, p.situation_before); }
  const keys = [...starts.keys()];
  const reached = Math.max(9, ...keys.map(k => Number(k.split(':')[0])));   // extra innings grow the table
  const innings = Array.from({ length: reached }, (_, i) => ({ inning: i + 1, away: null as number | null, home: null as number | null }));
  keys.forEach((key, i) => {
    const [inn, half] = key.split(':'); const s = starts.get(key)!; const next = i + 1 < keys.length ? starts.get(keys[i + 1])! : last;
    const side = half === 'Top' ? 'away' : 'home';
    innings[Number(inn) - 1][side] = (next[`${side}_score`] ?? 0) - (s[`${side}_score`] ?? 0);
  });
  const hits = (side: 'away' | 'home') => fx.pas.filter((pa: Raw) => pa.last_index < k && (pa.half === 'Top') === (side === 'away')
    && ['single', 'double', 'triple', 'home_run'].includes(pa.event)).length;
  const bb = (side: 'away' | 'home') => fx.pas.filter((pa: Raw) => pa.last_index < k && (pa.half === 'Top') === (side === 'away')
    && ['walk', 'intent_walk', 'hit_by_pitch'].includes(pa.event)).length;
  return { basis: 'revealed_events', current: { inning: last.inning, half: last.half }, innings,
    teams: { away: { runs: last.away_score, hits: hits('away'), errors: null, bb_hbp: bb('away') }, home: { runs: last.home_score, hits: hits('home'), errors: null, bb_hbp: bb('home') } } };
}
/** today's numbers as of pitch k, counted from the published plays only (earned runs are not derivable → null) */
function context(k: number, batterId: number | null, pitcherId: number | null) {
  const done = fx.pas.filter((pa: Raw) => pa.last_index < k);
  const mine = done.filter((pa: Raw) => pa.batter.id === batterId);
  const notAb = ['walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf'];
  const hit = ['single', 'double', 'triple', 'home_run'];
  return { basis: 'event_prefix_current_feed',
    batter: batterId === null ? null : { at_bats: mine.filter((pa: Raw) => !notAb.includes(pa.event)).length, hits: mine.filter((pa: Raw) => hit.includes(pa.event)).length,
      on_base: mine.filter((pa: Raw) => hit.includes(pa.event) || ['walk', 'intent_walk', 'hit_by_pitch'].includes(pa.event)).length },
    pitcher: pitcherId === null ? null : { pitches: P.slice(0, k).filter(p => p.pitcher.id === pitcherId).length,
      strikeouts: done.filter((pa: Raw) => pa.pitcher.id === pitcherId && /strikeout/.test(pa.event ?? '')).length, earned_runs: null },
    next_batters: [] };
}
const COPY: Record<string, [string, string | null]> = {
  recommended: ['다음 공 추천', null], unconfirmed: ['확인 중', '경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.'],
  feed_delayed: ['자료 늦음', '경기 자료가 평소보다 늦게 도착했어요. 추천이 실제 경기보다 늦을 수 있어요.'],
  disconnected: ['연결 끊김', '경기 자료 연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.'],
  unsupported: ['추천 없음', '이 상황은 추천 대상이 아니에요.'], waiting: ['공수 교대', '공수 교대 중이에요.'],
  pregame: ['경기 전', '경기 시작 전이에요.'], final: ['경기 종료', '경기가 끝났어요.'] };
const display = (state: string, pitch_key: string | null, shown: boolean, extra: Raw = {}) => ({ schema: 'pitcheezy-display-state-v1', state,
  reason: extra.reason ?? 'replay', label: COPY[state][0], message: COPY[state][1], pitch_key, recommendation_shown: shown, correction: extra.correction ?? null,
  feed: { ok: extra.feedOk ?? true, delayed: extra.delayed ?? false, lag_s: extra.lag ?? null } });

/** The body of game/990003.json at one step (without the publisher's `publish` key). */
export function liveBody(step: Step): Raw {
  const { k, kind, inject } = step;
  const pitches: Raw[] = P.slice(0, k).map(p => ({ ...clone(p), receipt: { status: 'received', corrected: false, reason: null } }));
  if (inject === 'unconfirmed' && pitches.length) {
    const last = pitches[pitches.length - 1];
    last.receipt = { status: 'unconfirmed', corrected: false, reason: 'replay' };
    last.actual = { pitch_type: null, pitch_label: null, speed_mph: null, x: null, z: null, description: null, event: null, event_label: null,
      play_text: null, call_label: '확인 중', result_label: '확인 중', zone_label: null, catcher_setup: null };
  }
  const pas = fx.pas.filter((pa: Raw) => pa.first_index < k).map((pa: Raw) => pa.last_index < k ? clone(pa)
    : { ...clone(pa), complete: false, event: null, event_label: null, result_text: null, play_text: null, last_index: k - 1, pitch_count: k - pa.first_index });
  const p = kind === 'pitch' ? P[k] : null;
  const lastSit = k > 0 ? situationOf(P[Math.min(k, P.length - 1)]) : situationOf(P[0]);
  const game = { ...clone(fx.game), game_pk: LIVE_PK, status: kind === 'final' ? 'final' : kind === 'pregame' ? 'scheduled' : 'live',
    status_text: kind === 'final' ? 'Final' : kind === 'pregame' ? 'Pre-Game' : 'In Progress', live: kind === 'pitch' || kind === 'break',
    // the final step carries the real final card (the fixture holds only the first 65 pitches, the score is the game's real final)
    away_score: kind === 'pregame' ? null : kind === 'final' ? fx.game.away_score : lastSit.away_score, home_score: kind === 'pregame' ? null : kind === 'final' ? fx.game.home_score : lastSit.home_score,
    inning: kind === 'pitch' ? p!.inning : null, half: kind === 'pitch' ? p!.half : null };
  const body: Raw = { schema: 'pitcheezy-service-game-v2', synthetic: 'replay of real 849843 pitches; game_pk, live keys and injected states are synthetic',
    zone_bounds: fx.zone_bounds, source: { kind: 'live', session_id: null }, game, feed: { as_of: null, ok: true, stale_s: null },
    cutoff: { index: k - 1, pitch_key: k > 0 ? P[k - 1].key : null }, pitches, pas, pregame: null,
    // the final step carries the game's real final score but the fixture stops after 65 pitches: a partial linescore next to it would
    // not add up, so the final step publishes none (the screen then says the inning-by-inning score is not available)
    linescore: kind === 'pregame' || kind === 'final' ? null : linescore(k), receipts: { version: `r${k}${inject === 'unconfirmed' ? 'u' : ''}`, paired: 0, corrected: 0, unconfirmed: inject === 'unconfirmed' ? 1 : 0 } };
  if (kind === 'pregame') {
    body.current = { phase: 'pregame', pa_key: null, batter: null, pitcher: null, situation: null, context: null };
    body.next = null; body.display_state = display('pregame', null, false, { reason: 'pregame' });
  } else if (kind === 'break') {
    body.current = { phase: 'between_innings', pa_key: null, batter: null, pitcher: null, situation: null, context: null };
    body.next = null; body.display_state = display('waiting', null, false, { reason: 'between_innings' });
  } else if (kind === 'final') {
    body.current = { phase: 'final', pa_key: null, batter: null, pitcher: null, situation: null, context: null };
    body.next = null; body.display_state = display('final', null, false, { reason: 'final' });
  } else {
    const rec = { ...clone(p!.rec), provenance: 'captured_live', policy: { name: 'ARM-B', identity: 'replay' } };
    body.current = { phase: 'pre_pitch', pa_key: p!.pa_key, batter: p!.batter, pitcher: p!.pitcher, situation: situationOf(p!), context: context(k, p!.batter.id, p!.pitcher.id) };
    if (inject === 'unsupported') {
      body.next = { pitch_key: p!.key, rec: { status: 'unsupported', reason: '이 투수는 추천 대상이 아니에요.', candidates: [], provenance: 'captured_live', policy: null, recorded_at: null } };
      body.display_state = display('unsupported', p!.key, false);
    } else {
      body.next = { pitch_key: p!.key, rec };
      body.display_state = inject === 'unconfirmed' ? display('unconfirmed', p!.key, false)
        : inject === 'feed_delayed' ? display('feed_delayed', p!.key, true, { delayed: true, lag: 47.2 })
        : inject === 'disconnected' ? display('disconnected', p!.key, true, { feedOk: false })
        : inject === 'correction' ? display('recommended', p!.key, true, { correction: { kind: 'replaced', message: '상황이나 기록이 바뀌어 추천이 달라졌어요.',
            previous: { pitch_type: rec.candidates[1]?.pitch_type ?? null, pitch_label: rec.candidates[1]?.pitch_label ?? null, zone_id: rec.candidates[1]?.zone_id ?? null, zone_label: rec.candidates[1]?.zone_label ?? null } } })
        : display('recommended', p!.key, true);
    }
  }
  return body;
}
