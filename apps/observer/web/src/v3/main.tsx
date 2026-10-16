/** Entry of a v3 build (vite.config.ts swaps index.html's script to this file when VITE_UI=v3). Same routes as the classic entry:
 *  `?game=<pk>` (or /watch/<pk>) = one game, anything else = the list. The public bundle adds the host slot, as the classic one does. */
import React, { Component } from 'react';
import type { ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import HostSlot from '../service/HostSlot';
import App, { gameIdFromPath } from './App';
import './v3.css';

const viewport = document.querySelector('meta[name="viewport"]');
if (viewport && !/viewport-fit/.test(viewport.getAttribute('content') ?? '')) viewport.setAttribute('content', `${viewport.getAttribute('content')}, viewport-fit=cover`);
document.querySelector('meta[name="theme-color"]')?.setAttribute('content', '#15191e');

/** An exception while drawing shows plain words and a reload control instead of a blank page; the exception text is never shown. */
class Boundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (!this.state.failed) return this.props.children;
    return <div className="v3"><main className="v3-center" role="alert"><p>화면을 표시하지 못했어요. 새로 고치면 다시 불러와요.</p>
      <button className="v3-btn" onClick={() => window.location.reload()}>새로 고침</button></main></div>;
  }
}
const game = gameIdFromPath();
const mount = () => createRoot(document.getElementById('root')!).render(<React.StrictMode><Boundary>
  <App gamePk={game} />{__PZ_STATIC__ && <HostSlot screen={game ? 'game' : 'list'} gamePk={game} />}
</Boundary></React.StrictMode>);
if (__PZ_REPLAY__) void import('./replay/install').then(m => { m.installReplay(); mount(); });
else mount();
