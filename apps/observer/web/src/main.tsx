import React, { Suspense, lazy } from 'react';
import type { ComponentType } from 'react';
import { createRoot } from 'react-dom/client';
import WatchList from './service/WatchList';
import GameScreen, { gameIdFromPath } from './service/GameScreen';
import HostSlot from './service/HostSlot';
import './styles.css';

// iPhone safe areas: env(safe-area-inset-*) only applies with viewport-fit=cover (index.html is outside this module's scope).
const viewport = document.querySelector('meta[name="viewport"]');
if (viewport && !/viewport-fit/.test(viewport.getAttribute('content') ?? '')) viewport.setAttribute('content', `${viewport.getAttribute('content')}, viewport-fit=cover`);

// Operator console is owned by the backend author (src/demo-operator/**, CONTRACT rev2 §5: named export `DemoOperator`).
// Wired only if that module exists in the build, so this file compiles with or without it.
// Public (static-mode) bundle (`__PZ_STATIC__`, a build-time literal): the operator console and demo sessions are not part of it;
// the module is not looked up, so it is neither routed nor bundled.
function operator() {
  if (__PZ_STATIC__) return null;
  const operatorModules = import.meta.glob<{ DemoOperator: ComponentType }>('./demo-operator/index.{ts,tsx}');
  const operatorLoader = Object.values(operatorModules)[0];
  return operatorLoader ? lazy(() => operatorLoader().then(m => ({ default: m.DemoOperator }))) : null;
}
const Operator = operator();

const route = window.location.pathname.replace(/\/$/, '');
// Ordinary service: /watch (KST date list) → /watch/{gamePk} (one game screen for live, demo and finished games).
// /game and /live with a game id open the same screen. Integration (review-integration-002): this entry ships ONLY the
// service screens and the operator console. The earlier video/sync/reveal/showcase screens are not bundled, so no MLB
// media URL or scene pitch list is in any served asset; they remain in git history and in the 282dc6f rollback build.
function page() {
  const game = gameIdFromPath();
  if (!__PZ_STATIC__ && route === '/demo-operator') return Operator ? <Suspense fallback={null}><Operator /></Suspense> : <p style={{ padding: 24 }}>운영자 화면이 이 빌드에 없습니다.</p>;
  const screen = game ? <GameScreen gamePk={game} /> : <WatchList />;
  // the one empty extension point for the hosting page (feedback / analytics are the host's, never ours): see service/HostSlot.tsx
  return __PZ_STATIC__ ? <>{screen}<HostSlot screen={game ? 'game' : 'list'} gamePk={game} /></> : screen;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode>{page()}</React.StrictMode>);
