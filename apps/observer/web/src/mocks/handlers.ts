// UI 점검용 mock API. VITE_API_MOCK=1 일 때만 번들에 들어간다 (main.tsx 참고).
// 백엔드 observer_app/service.py 의 세션 의미(create → advance 로 cursor 전진 → 완료 후 manual-intent)를
// 메모리 안에서 그대로 따른다. 값은 전부 가공이며 실제 경기·모델 출력이 아니다.
import { HttpResponse, delay, http } from 'msw';
import type { Analysis, Candidate, Catalog, GameState, Pitch, Recommendation, View, Zone } from '../types';
import catalogFixture from './fixtures/catalog.json';
import zonesFixture from './fixtures/zones.json';
import paFixture from './fixtures/plate-appearances.json';

interface FixtureCandidate { pitch_type: string; pitch_label: string; zone_id: string; value: number; delta_pp: number; support: number }
interface FixturePitch { id: string; pitch_number: number; state: GameState; actual: { pitch_type: string; pitch_label: string; x: number | null; z: number | null; speed_mph: number | null; description: string; event?: string; result_label: string }; candidates: FixtureCandidate[]; baseline_value: number; context: View['context']; context_notes: string[] }
interface FixturePA { zone_bounds: { bottom: number; top: number }; pitches: FixturePitch[]; terminal_state: GameState }
interface Session { id: string; game_id: number; pa_id: number; cursor: number; revision: number; manual_zone_id: string | null; manual_revision: number }

const catalog = catalogFixture as unknown as Catalog;
const zones = zonesFixture.zones as Zone[];
const zoneById = new Map(zones.map((zone) => [zone.id, zone]));
const plateAppearances = paFixture as unknown as Record<string, FixturePA>;
const sessions = new Map<string, Session>();
const MODEL_VERSION = catalog.model_version;
const LATENCY = Number(new URLSearchParams(window.location.search).get('mockLatency') ?? 120);

const problem = (status: number, detail: string) => HttpResponse.json({ detail }, { status });
const targetPoint = (zoneId: string, bounds: FixturePA['zone_bounds']) => {
  const zone = zoneById.get(zoneId)!;
  return { x: (zone.column - 1) * 1.66 / 3, z: bounds.bottom + (zone.row + .5) * (bounds.top - bounds.bottom) / 3 };
};
const actualZoneLabel = (x: number | null, z: number | null, bounds: FixturePA['zone_bounds']) => {
  if (x === null || z === null) return '위치 정보 없음';
  if (Math.abs(x) > .83 || z < bounds.bottom || z > bounds.top) return '존 밖';
  const column = Math.min(2, Math.max(0, Math.floor((x + .83) / (1.66 / 3))));
  const row = Math.min(2, Math.max(0, Math.floor((z - bounds.bottom) / ((bounds.top - bounds.bottom) / 3))));
  return zones.find((zone) => zone.column === column && zone.row === row)?.label ?? '위치 정보 없음';
};

function recommendation(pitch: FixturePitch, pa: FixturePA): Recommendation {
  const candidates: Candidate[] = pitch.candidates.map((candidate) => ({
    ...candidate, zone_label: zoneById.get(candidate.zone_id)?.label ?? candidate.zone_id, target: targetPoint(candidate.zone_id, pa.zone_bounds),
  }));
  const outcomes = (strike: number) => [
    { outcome: 'strike', label: '스트라이크', probability: strike },
    { outcome: 'ball', label: '볼', probability: Math.max(0, .62 - strike) },
    { outcome: 'in_play', label: '인플레이', probability: .38 },
  ];
  return {
    id: `mock-rec-${pitch.id}`, status: 'ready', mode: 'experimental_location_proxy', model_version: MODEL_VERSION, candidates,
    baseline_value: pitch.baseline_value, zone_bounds: pa.zone_bounds,
    basis: ['(mock) 주자·아웃·카운트·점수 반영', '(mock) 경기 전날까지의 타자 성향 반영'], reason: null,
    explanation: {
      outcome_probabilities: [outcomes(.34), outcomes(.30), outcomes(.28)],
      comparison: { margin_pp: Number((candidates[0].value - candidates[1].value).toFixed(4)) * 100, contributions: [
        { outcome: 'strike', label: '스트라이크', probability_difference_pp: 4.0, value_contribution_pp: 0.6 },
        { outcome: 'ball', label: '볼', probability_difference_pp: -4.0, value_contribution_pp: 0.2 },
      ], definition: '(mock) 같은 이후 투구 정책을 따른다는 모델 안에서 첫 공의 결과 확률 차이를 분해했습니다.' },
      notice: '(mock) 모델이 추정한 한 공의 결과 분포입니다. 실제 빈도나 검증된 효과가 아닙니다.',
    },
  };
}

function reveal(pitch: FixturePitch, pa: FixturePA): Pitch {
  return {
    id: pitch.id, pitch_number: pitch.pitch_number, pitch_type: pitch.actual.pitch_type, pitch_label: pitch.actual.pitch_label,
    x: pitch.actual.x, z: pitch.actual.z, speed_mph: pitch.actual.speed_mph, description: pitch.actual.description,
    result_label: pitch.actual.result_label, pre_state: pitch.state, recommendation: recommendation(pitch, pa), zone_bounds: pa.zone_bounds,
  };
}

function view(session: Session): View {
  const game = catalog.games.find((item) => item.id === session.game_id)!;
  const plateAppearance = game.plate_appearances!.find((item) => item.id === session.pa_id)!;
  const pa = plateAppearances[`${session.game_id}:${session.pa_id}`];
  const complete = session.cursor >= pa.pitches.length;
  const history = pa.pitches.slice(0, session.cursor).map((pitch) => reveal(pitch, pa));
  const current = complete ? null : pa.pitches[session.cursor];
  let analysis: Analysis | null = null, summary: View['summary'] = null;
  if (complete) {
    const last = history[history.length - 1];
    const manual = session.manual_zone_id;
    const narrative = [`마지막 공의 결과는 ${last.result_label}입니다.`, '영상이 없어 포수의 실제 목표는 확인할 수 없습니다.'];
    if (manual) narrative.push('목표 구역은 사용자가 입력했으며 실제 의도를 확인한 값은 아닙니다.');
    const target = manual ? targetPoint(manual, pa.zone_bounds) : null;
    analysis = {
      id: `mock-job-${last.id}`, status: manual ? 'complete' : 'unavailable', source: manual ? 'manual' : 'none',
      selected_pitch_id: last.id, selected_pitch_number: last.pitch_number, selection_reason: '타석의 마지막 공.', cv_status: 'unavailable:no_media',
      message: manual ? '입력한 목표 구역과 실제 위치를 비교했습니다.' : '영상이 없어 자동 의도 분석은 제공되지 않습니다.',
      manual_zone_id: manual,
      comparisons: manual && target ? {
        recommended_zone_label: last.recommendation?.candidates[0]?.zone_label ?? '추천 정보 없음', intended_zone_label: zoneById.get(manual)!.label,
        actual_zone_label: actualZoneLabel(last.x, last.z, pa.zone_bounds),
        target_error_zone_units: last.x === null || last.z === null ? null : Number(Math.hypot((last.x - target.x) / 1.66, (last.z - target.z) / (pa.zone_bounds.top - pa.zone_bounds.bottom)).toFixed(3)),
        interpretation: '사용자가 입력한 목표 구역과 실제 도달 위치의 공간적 비교입니다. 선수의 인과적 책임 점수는 아닙니다.',
      } : null,
      narrative, version: 1 + session.manual_revision,
    };
    summary = { headline: '타석 관찰 완료', result_label: last.result_label, pitch_count: history.length, selected_pitch_number: last.pitch_number, notes: ['추천과 실제 결과의 비교이며 추천의 인과적 효과를 평가하지 않습니다.'] };
  }
  return {
    id: session.id, revision: session.revision, cursor: session.cursor, complete, game: { id: game.id, date: game.date, home_team: game.home_team, away_team: game.away_team, title: game.title },
    plate_appearance: plateAppearance, state: complete ? pa.terminal_state : current!.state,
    recommendation: complete ? null : recommendation(current!, pa), last_pitch: history[history.length - 1] ?? null, history, analysis, summary,
    context: complete ? null : current!.context, context_notes: complete ? [] : current!.context_notes,
    notices: ['UI 점검용 mock 데이터 · 실제 경기·모델 출력이 아닙니다.', '기록 재생 · 실제 승률 향상이 검증된 추천은 아닙니다.'],
  };
}

const findSession = (id: string) => sessions.get(decodeURIComponent(id)) ?? null;

export const handlers = [
  http.get('/api/health', () => HttpResponse.json({ status: 'ok', model_ready: true, dataset_ready: true, mode: 'mock', cv_mode: 'unavailable:no_media' })),
  http.get('/api/catalog', async () => { await delay(LATENCY); return HttpResponse.json(catalog); }),
  http.get('/api/zones', async () => { await delay(LATENCY); return HttpResponse.json(zonesFixture); }),
  http.get('/api/runtime', () => HttpResponse.json({ mode: 'mock', model_version: MODEL_VERSION, scientific_runtime: null, dataset_identity: 'mock-fixture', worker_status: 'stopped', cv_status: 'unavailable:no_media', startup_error: null })),

  http.post('/api/sessions', async ({ request }) => {
    await delay(LATENCY);
    const body = await request.json() as { game_id?: number; pa_id?: number };
    const game_id = Number(body.game_id), pa_id = Number(body.pa_id);
    if (!plateAppearances[`${game_id}:${pa_id}`]) return problem(404, '경기 또는 타석을 찾을 수 없습니다.');
    const session: Session = { id: crypto.randomUUID(), game_id, pa_id, cursor: 0, revision: 0, manual_zone_id: null, manual_revision: 0 };
    sessions.set(session.id, session);
    return HttpResponse.json(view(session));
  }),
  http.get('/api/sessions/:id', async ({ params }) => {
    await delay(LATENCY);
    const session = findSession(params.id as string);
    return session ? HttpResponse.json(view(session)) : problem(404, '관찰 세션을 찾을 수 없습니다.');
  }),
  http.post('/api/sessions/:id/advance', async ({ params, request }) => {
    await delay(LATENCY);
    const session = findSession(params.id as string);
    if (!session) return problem(404, '관찰 세션을 찾을 수 없습니다.');
    const { revision } = await request.json() as { revision: number };
    if (revision !== session.revision) return problem(409, '상태가 변경되었습니다. 현재 화면을 새로 불러와 주세요.');
    const pa = plateAppearances[`${session.game_id}:${session.pa_id}`];
    if (session.cursor >= pa.pitches.length) return problem(409, '이미 완료된 타석입니다.');
    session.cursor += 1; session.revision += 1;
    return HttpResponse.json(view(session));
  }),
  http.post('/api/sessions/:id/manual-intent', async ({ params, request }) => {
    await delay(LATENCY);
    const session = findSession(params.id as string);
    if (!session) return problem(404, '관찰 세션을 찾을 수 없습니다.');
    const { revision, zone_id } = await request.json() as { revision: number; zone_id: string };
    if (!zoneById.has(zone_id)) return problem(400, '올바른 목표 구역을 선택해 주세요.');
    if (revision !== session.revision) return problem(409, '상태가 변경되었습니다. 현재 화면을 새로 불러와 주세요.');
    const pa = plateAppearances[`${session.game_id}:${session.pa_id}`];
    if (session.cursor < pa.pitches.length) return problem(409, '타석이 끝난 뒤 마지막 공의 목표 구역을 입력할 수 있습니다.');
    if (session.manual_zone_id) session.manual_revision += 1;
    session.manual_zone_id = zone_id; session.revision += 1;
    return HttpResponse.json(view(session));
  }),

  // video-lab 은 mock 범위 밖. 화면이 "준비되지 않음" 상태를 그리도록 503을 돌려준다.
  http.get('/api/video-lab/*', () => problem(503, '(mock) 영상 실험실은 mock 모드에서 제공되지 않습니다.')),
  http.post('/api/video-lab/*', () => problem(503, '(mock) 영상 실험실은 mock 모드에서 제공되지 않습니다.')),
];
