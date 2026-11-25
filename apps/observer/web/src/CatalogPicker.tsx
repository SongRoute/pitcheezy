import { useEffect, useMemo, useRef, useState } from 'react';
import { bookmarkKey, filterCatalog } from './catalog';
import type { Bookmarks } from './catalog';
import type { Catalog, View } from './types';

interface Props {
  catalog: Catalog | null; view: View | null; busy: boolean; loading: boolean; bookmarks: Bookmarks;
  onStart: (gameId: number, paId: number) => void;
  onResume: (id: string) => Promise<boolean>;
}

export default function CatalogPicker({ catalog, view, busy, loading, bookmarks, onStart, onResume }: Props) {
  const [query, setQuery] = useState('');
  const [date, setDate] = useState('');
  const [selection, setSelection] = useState<{ gameId: number; paId: number } | null>(null);
  const disclosure = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (view) { setSelection({ gameId: view.game.id, paId: view.plate_appearance.id }); setQuery(''); setDate(''); }
  }, [view?.id]);
  const games = useMemo(() => filterCatalog(catalog?.games || [], query, date), [catalog, query, date]);
  const game = games.find((item) => item.id === selection?.gameId) || games[0];
  const pa = game?.plate_appearances?.find((item) => item.id === selection?.paId) || game?.plate_appearances?.[0];
  const matchingPAs = games.reduce((count, item) => count + (item.plate_appearances?.length || 0), 0);
  const current = !!view && game?.id === view.game.id && pa?.id === view.plate_appearance.id;
  const existing = current ? view.id : game && pa ? bookmarks[bookmarkKey(game.id, pa.id)] : undefined;
  const clear = () => { setQuery(''); setDate(''); };

  return <details ref={disclosure} className={`session-disclosure ${view ? 'has-session' : ''}`} key={view?.id || 'welcome'} open={!view}>
    <summary><span>경기·타석 바꾸기</span><span className="disclosure-current">{view ? `${view.game.date} · ${view.game.title}` : '관전할 타석을 선택하세요'}</span><span className="disclosure-arrow" aria-hidden="true">⌄</span></summary>
    <section className="session-picker searchable-picker" aria-label="경기와 타석 선택">
      <div className="picker-intro"><span className="eyebrow">관전할 순간 찾기</span><strong>어떤 타석이 궁금한가요?</strong></div>
      <div className="catalog-filters"><label htmlFor="catalog-search">팀·투수·타자 검색<input id="catalog-search" type="search" value={query} placeholder="팀 약어 또는 선수 이름" autoComplete="off" disabled={loading || busy} onChange={(event) => setQuery(event.target.value)} /></label><label htmlFor="catalog-date">경기 날짜<input id="catalog-date" type="date" value={date} disabled={loading || busy} onChange={(event) => setDate(event.target.value)} /></label><button className="button text-button" disabled={busy || (!query && !date)} onClick={clear}>필터 초기화</button></div>
      <p className="catalog-count" role="status">{loading ? '경기를 불러오는 중…' : `${games.length}경기 · ${matchingPAs}타석${query || date ? '이 검색되었어요.' : '을 볼 수 있어요.'}`}</p>
      {!loading && !games.length ? <div className="catalog-empty"><strong>{catalog?.games.length ? '조건에 맞는 타석이 없어요.' : '관전할 타석이 아직 준비되지 않았어요.'}</strong><p>{catalog?.games.length ? '선수 이름이나 팀 약어를 짧게 입력하거나, 날짜 필터를 지워 보세요.' : '경기 목록이 준비되면 이곳에서 관전할 타석을 선택할 수 있습니다.'}</p>{!!(query || date) && <button className="button secondary" onClick={clear}>전체 경기 보기</button>}{view && <p>아래의 현재 관전 기록은 그대로 유지됩니다.</p>}</div> : <>
        <label className="game-select-label">경기<select aria-label="경기" value={game?.id ?? ''} disabled={loading || busy || !games.length} onChange={(event) => { const next = games.find((item) => item.id === Number(event.target.value)); if (next) setSelection({ gameId: next.id, paId: next.plate_appearances?.[0]?.id ?? -1 }); }}><option value="" disabled>경기를 선택하세요</option>{games.map((item) => <option key={item.id} value={item.id}>{item.date} · {item.title}</option>)}</select></label>
        <label className="pa-select-label">타석<select aria-label="타석" value={pa?.id ?? ''} disabled={loading || busy || !game?.plate_appearances?.length} onChange={(event) => { if (game) setSelection({ gameId: game.id, paId: Number(event.target.value) }); }}><option value="" disabled>타석을 선택하세요</option>{game?.plate_appearances?.map((item) => <option key={item.id} value={item.id}>{item.inning ? `${item.inning}회 ${item.half === 'Top' ? '초' : '말'} · ` : ''}{item.batter_label} vs {item.pitcher_label}</option>)}</select></label>
        <div className="picker-actions">{existing && <button className="button secondary" disabled={busy || loading} onClick={async () => { if (await onResume(existing)) { if (disclosure.current) disclosure.current.open = false; } }}>{current ? '현재 관전 계속보기' : '저장한 타석 이어보기'}</button>}<button className="button primary" disabled={loading || busy || !game || !pa} onClick={() => { if (game && pa) onStart(game.id, pa.id); }}>{busy ? '불러오는 중…' : existing ? '이 타석 처음부터 보기' : view ? '새 타석 관전하기' : '타석 관전하기'}<span aria-hidden="true">↗</span></button></div>
        <p className="picker-session-note">{existing ? '이어보기는 확인한 공과 목표 메모를 유지합니다. 처음부터 보기는 별도의 관전을 시작합니다.' : '선택한 타석의 첫 공부터 새 관전을 시작합니다.'}</p>
      </>}
    </section>
  </details>;
}
