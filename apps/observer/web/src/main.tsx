import React, { Suspense, lazy } from 'react';
import type { ComponentType } from 'react';
import { createRoot } from 'react-dom/client';
import WatchList from './service/WatchList';
import GameScreen, { gameIdFromPath } from './service/GameScreen';
import './styles.css';

// iPhone safe areas: env(safe-area-inset-*) only applies with viewport-fit=cover (index.html is outside this module's scope).
const viewport = document.querySelector('meta[name="viewport"]');
if (viewport && !/viewport-fit/.test(viewport.getAttribute('content') ?? '')) viewport.setAttribute('content', `${viewport.getAttribute('content')}, viewport-fit=cover`);

// Operator console is owned by the backend author (src/demo-operator/**, CONTRACT rev2 §5: named export `DemoOperator`).
// Wired only if that module exists in the build, so this file compiles with or without it.
const operatorModules = import.meta.glob<{ DemoOperator: ComponentType }>('./demo-operator/index.{ts,tsx}');
const operatorLoader = Object.values(operatorModules)[0];
const Operator = operatorLoader ? lazy(() => operatorLoader().then(m => ({ default: m.DemoOperator }))) : null;

const route = window.location.pathname.replace(/\/$/, '');
// Ordinary service: /watch (KST date list) → /watch/{gamePk} (one game screen for live, demo and finished games).
// /game and /live with a game id open the same screen. Integration (review-integration-002): this entry ships ONLY the
// service screens and the operator console. The earlier video/sync/reveal/showcase screens are not bundled, so no MLB
// media URL or scene pitch list is in any served asset; they remain in git history and in the 282dc6f rollback build.
function page() {
  const game = gameIdFromPath();
  if (route === '/demo-operator') return Operator ? <Suspense fallback={null}><Operator /></Suspense> : <p style={{ padding: 24 }}>운영자 화면이 이 빌드에 없습니다.</p>;
  return game ? <GameScreen gamePk={game} /> : <WatchList />;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode>{page()}</React.StrictMode>);
