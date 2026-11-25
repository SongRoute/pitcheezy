// Operator-only control of the server-owned demo clock (CONTRACT pitcheezy-service-v2 §5).
// The token is typed by the operator and kept in sessionStorage; it is never in the bundle or URL.
// This page does not control any video player: start/pause the external player by hand together.
import { useCallback, useEffect, useState } from 'react';

type Mapping = { mapping_id: string; game_pk: number; kind: string; verified: boolean; rows: number; first_index: number; last_index: number };
type Session = {
  session_id: string; game_pk: number; status: string; rate: number; version: number; start_index: number;
  progress_s: number; video_time_s: number; cutoff: { index: number; pitch_key: string | null };
  next_reveal: { index: number; pitch_key: string; at_video_s: number; in_s: number | null } | null;
  mapping: { mapping_id: string; kind: string; verified: boolean }; segment: { first_index: number; last_index: number };
  viewer_url_path: string;
};

const TOKEN_KEY = 'pitcheezy-demo-operator-token';
// crypto.randomUUID exists only in secure contexts (HTTPS/localhost); the preview is plain HTTP over Tailscale.
// getRandomValues works in any context and is enough for an idempotency key.
const opId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');

async function call<T>(token: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/service/operator${path}`, {
    method: body === undefined ? 'GET' : 'POST', cache: 'no-store',
    headers: { 'X-Pitcheezy-Operator': token, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({ detail: '응답을 읽지 못했습니다.' }));
  if (!response.ok) throw new Error(payload.detail ?? `요청 실패 (${response.status})`);
  return payload as T;
}

function clock(seconds: number) {
  const sign = seconds < 0 ? '-' : '';
  const s = Math.abs(Math.round(seconds));
  return `${sign}${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

export function DemoOperator() {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY) ?? '');
  const [draft, setDraft] = useState('');
  const [mappings, setMappings] = useState<Mapping[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [mappingId, setMappingId] = useState('');
  const [startIndex, setStartIndex] = useState('');
  const [seekIndex, setSeekIndex] = useState('');
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      const [catalog, list] = await Promise.all([
        call<{ mappings: Mapping[] }>(token, '/catalog'), call<{ sessions: Session[] }>(token, '/sessions')]);
      setMappings(catalog.mappings);
      setSessions(list.sessions);
      setError(null);
    } catch (caught) {
      setError((caught as Error).message);
    }
  }, [token]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const current = sessions.find((s) => s.session_id === selected) ?? null;

  async function act(body: Record<string, unknown>) {
    if (!current) return;
    try {
      await call(token, `/sessions/${current.session_id}/control`,
        { ...body, expected_version: current.version, op_id: opId() });
      await refresh();
    } catch (caught) {
      setError((caught as Error).message);
      await refresh();
    }
  }

  async function create() {
    const mapping = mappings.find((m) => m.mapping_id === mappingId);
    if (!mapping) return;
    try {
      const made = await call<Session>(token, '/sessions', {
        game_pk: mapping.game_pk, mapping_id: mapping.mapping_id, op_id: opId(),
        ...(startIndex === '' ? {} : { start_index: Number(startIndex) }),
      });
      setSelected(made.session_id);
      await refresh();
    } catch (caught) {
      setError((caught as Error).message);
    }
  }

  if (!token) {
    return (
      <main style={{ padding: 16, fontFamily: 'system-ui', maxWidth: 560 }}>
        <h1 style={{ fontSize: 20 }}>시연 운영</h1>
        <label htmlFor="op-token">운영자 토큰</label>
        <input id="op-token" type="password" autoComplete="off" value={draft} onChange={(e) => setDraft(e.target.value)} style={{ display: 'block', width: '100%', fontSize: 16, margin: '8px 0' }} />
        <button type="button" onClick={() => { sessionStorage.setItem(TOKEN_KEY, draft.trim()); setToken(draft.trim()); }}>토큰 저장</button>
      </main>
    );
  }

  return (
    <main style={{ padding: 16, fontFamily: 'system-ui', maxWidth: 640, fontSize: 16 }}>
      <h1 style={{ fontSize: 20 }}>시연 운영</h1>
      <p style={{ margin: '4px 0 12px' }}>영상 플레이어는 자동으로 제어되지 않습니다. 영상과 시연 시계를 함께 시작·정지하세요.</p>
      {error && <p role="alert" style={{ color: '#a33' }}>{error}</p>}
      <section aria-labelledby="op-new">
        <h2 id="op-new" style={{ fontSize: 17 }}>새 시연 세션</h2>
        <label htmlFor="op-mapping">대응표</label>{' '}
        <select id="op-mapping" value={mappingId} onChange={(e) => setMappingId(e.target.value)}>
          <option value="">선택</option>
          {mappings.map((m) => (
            <option key={m.mapping_id} value={m.mapping_id}>
              {m.mapping_id} · 경기 {m.game_pk} · 투구 {m.first_index}–{m.last_index} · {m.kind === 'pa_end_observed' ? '타석 끝 시각만 영상 관찰(투구별 미확인)' : m.verified ? '영상 대조 확인' : '영상 대조 미확인'}
            </option>
          ))}
        </select>{' '}
        <label htmlFor="op-start">시작 투구 번호(선택)</label>{' '}
        <input id="op-start" inputMode="numeric" value={startIndex} onChange={(e) => setStartIndex(e.target.value)} style={{ width: 70 }} />{' '}
        <button type="button" onClick={() => void create()} disabled={!mappingId}>세션 만들기</button>
      </section>
      <section aria-labelledby="op-list" style={{ marginTop: 16 }}>
        <h2 id="op-list" style={{ fontSize: 17 }}>세션</h2>
        <label htmlFor="op-session">세션 선택</label>{' '}
        <select id="op-session" value={selected ?? ''} onChange={(e) => setSelected(e.target.value || null)}>
          <option value="">선택</option>
          {sessions.map((s) => (
            <option key={s.session_id} value={s.session_id}>{s.session_id.slice(0, 8)} · 경기 {s.game_pk} · {s.status}</option>
          ))}
        </select>
      </section>
      {current && (
        <section aria-labelledby="op-ctrl" style={{ marginTop: 16 }}>
          <h2 id="op-ctrl" style={{ fontSize: 17 }}>진행</h2>
          <dl style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
            <dt>상태</dt><dd>{current.status} · {current.rate}배속 · 버전 {current.version}</dd>
            <dt>영상 기준 시각</dt><dd aria-live="polite">{clock(current.video_time_s)}</dd>
            <dt>마지막 공개 투구</dt><dd>{current.cutoff.index} ({current.cutoff.pitch_key ?? '없음'})</dd>
            <dt>다음 공개</dt><dd>{current.next_reveal ? `${current.next_reveal.pitch_key} · 영상 ${clock(current.next_reveal.at_video_s)}${current.next_reveal.in_s === null ? '' : ` · ${current.next_reveal.in_s}초 뒤`}` : '구간 끝'}</dd>
            <dt>대응표</dt><dd>{current.mapping.mapping_id} · {current.mapping.kind === 'pa_end_observed' ? '타석 끝 시각만 영상 관찰 — 타석이 끝날 때 그 타석 투구가 한꺼번에 공개됨, 투구별 시각은 모름' : current.mapping.verified ? '영상 대조 확인' : '영상 대조 미확인(진행 속도만)'}</dd>
            <dt>관람 주소</dt><dd><code>{current.viewer_url_path}</code></dd>
          </dl>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <button type="button" onClick={() => void act({ action: 'start' })} disabled={current.status !== 'ready'}>시작</button>
            <button type="button" onClick={() => void act({ action: 'pause' })} disabled={current.status !== 'running'}>일시정지</button>
            <button type="button" onClick={() => void act({ action: 'resume' })} disabled={current.status !== 'paused'}>재개</button>
            <button type="button" onClick={() => void act({ action: 'reset' })} disabled={current.status === 'ended'}>처음부터 다시</button>
            <label htmlFor="op-rate">진행 속도</label>
            <select id="op-rate" value={current.rate} onChange={(e) => void act({ action: 'rate', rate: Number(e.target.value) })} disabled={current.status === 'ended'}>
              {[0.5, 1, 1.5, 2].map((r) => <option key={r} value={r}>{r}배속</option>)}
            </select>
          </div>
          <div style={{ marginTop: 8 }}>
            <label htmlFor="op-seek">특정 투구로 이동(투구 번호)</label>{' '}
            <input id="op-seek" inputMode="numeric" value={seekIndex} onChange={(e) => setSeekIndex(e.target.value)} style={{ width: 70 }} />{' '}
            <button type="button" onClick={() => void act({ action: 'seek', index: Number(seekIndex) })} disabled={seekIndex === '' || current.status === 'ended'}>이동</button>
          </div>
          <div style={{ marginTop: 16 }}>
            <button type="button" onClick={() => { if (window.confirm('시연을 끝내면 이 경기의 전체 기록이 일반 화면에 다시 열립니다. 끝낼까요?')) void act({ action: 'end' }); }} disabled={current.status === 'ended'}>시연 종료(전체 기록 다시 열기)</button>
          </div>
        </section>
      )}
      <p style={{ marginTop: 16 }}><button type="button" onClick={() => { sessionStorage.removeItem(TOKEN_KEY); setToken(''); }}>토큰 지우기</button></p>
    </main>
  );
}
