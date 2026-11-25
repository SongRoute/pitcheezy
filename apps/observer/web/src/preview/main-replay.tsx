/** Preview entry for the PUBLIC screen (not a public bundle): vite.preview.config.ts swaps index.html's script to this file.
 *  It answers the screen's v1 file reads with a replay of the real finished game 849843 (replay/, copied from ui/v3), adds a small
 *  replay panel, then starts the unchanged public entry (src/main.tsx). Nothing in src/service or src/main.tsx is changed.
 *  Artifact hosts pass only a plain #anchor to the page, so `#game-<pk>` / `#list` are mapped onto the screen's own `?game=` form. */
import { installReplay } from './replay/install';

function routeFromHash() {
  const h = /^#game-(\d+)$/.exec(window.location.hash);
  if (!h && window.location.hash !== '#list') return;
  const q = new URLSearchParams(window.location.search);
  if (h) q.set('game', h[1]); else q.delete('game');
  const s = q.toString();
  window.history.replaceState(null, '', `${window.location.pathname}${s ? `?${s}` : ''}`);
}
routeFromHash();
installReplay();
void import('./ReplayBar').then(m => m.mountReplayBar());
void import('../main');
