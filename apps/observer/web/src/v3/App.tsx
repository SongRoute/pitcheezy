import { Suspense, lazy } from 'react';
import Game from './Game';
import List from './List';

const Review = __PZ_REVIEW__ ? lazy(() => import('./review/Review')) : null;

/** Same address rules as the classic screen (GameScreen.gameIdFromPath): /watch|game|live/<pk>, or `?game=<pk>` (plain digits only in a
 *  public bundle). Copied, not imported, so the classic screen module (and its styles) is not part of a v3 bundle. */
export const gameIdFromPath = () => {
  const m = /^\/(?:watch|game|live)\/(\d+)\/?$/.exec(window.location.pathname);
  const q = new URLSearchParams(window.location.search).get('game');
  const id = Number(m?.[1] ?? (__PZ_STATIC__ && !/^[0-9]+$/.test(q ?? '') ? NaN : q));
  return Number.isInteger(id) && id > 0 ? id : null;
};

export default function App({ gamePk }: { gamePk: number | null }) {
  return <>
    {gamePk ? <Game gamePk={gamePk} /> : <List />}
    {Review && <Suspense fallback={null}><Review /></Suspense>}
  </>;
}
