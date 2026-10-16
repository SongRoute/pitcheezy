import { Suspense, lazy, useEffect, useState } from 'react';
import Game from './Game';
import List from './List';

const Review = __PZ_REVIEW__ ? lazy(() => import('./review/Review')) : null;

/** Same address rules as the classic screen (GameScreen.gameIdFromPath): /watch|game|live/<pk>, or `?game=<pk>` (plain digits only in a
 *  public bundle). Copied, not imported, so the classic screen module (and its styles) is not part of a v3 bundle. */
export const gameIdFromPath = () => {
  if (__PZ_REPLAY__) { const h = /^#game-(\d+)$/.exec(window.location.hash); if (h) return Number(h[1]); if (window.location.hash === '#list') return null; }
  const m = /^\/(?:watch|game|live)\/(\d+)\/?$/.exec(window.location.pathname);
  const q = new URLSearchParams(window.location.search).get('game');
  const id = Number(m?.[1] ?? (__PZ_STATIC__ && !/^[0-9]+$/.test(q ?? '') ? NaN : q));
  return Number.isInteger(id) && id > 0 ? id : null;
};

export default function App({ gamePk: initial }: { gamePk: number | null }) {
  // preview builds route by hash (see links.ts); a public bundle navigates by real page loads and never listens
  const [gamePk, setGamePk] = useState(initial);
  useEffect(() => {
    if (!__PZ_REPLAY__) return;
    const on = () => { setGamePk(gameIdFromPath()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', on); return () => window.removeEventListener('hashchange', on);
  }, []);
  return <>
    {gamePk ? <Game key={gamePk} gamePk={gamePk} /> : <List key="list" />}
    {Review && <Suspense fallback={null}><Review /></Suspense>}
  </>;
}
