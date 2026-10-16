import { useEffect, useLayoutEffect, useRef } from 'react';
import type { ReactNode } from 'react';

/** One detail container in three forms:
 *   sheet    phone: modal bottom sheet over the live view (focus trapped, Esc / backdrop / 닫기 close, focus returns to the opener)
 *   overlay  tablet: modal side sheet
 *   dock     PC: a docked side panel, NOT modal — the live view beside it keeps updating and stays usable (spec §4) */
export type PanelMode = 'sheet' | 'overlay' | 'dock';
// Safari does not focus a button on click, so `document.activeElement` is <body> when a panel opens: remember the last pressed control.
let pressed: HTMLElement | null = null;
if (typeof window !== 'undefined') window.addEventListener('pointerdown', e => { pressed = (e.target as Element | null)?.closest?.('button,a[href]') as HTMLElement | null; }, true);
const focusable = 'button:not([disabled]),a[href],select,input,[tabindex="0"]';
/** where focus goes when the opener is gone (it was replaced while the panel was open): the main column of the live view */
const fallback = () => document.querySelector<HTMLElement>('[data-panel-return]');
export default function Panel({ mode, title, onClose, children, id = 'v3-panel-title' }: { mode: PanelMode; title: string; onClose: () => void; children: ReactNode; id?: string }) {
  const box = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  useLayoutEffect(() => { closeRef.current = onClose; });
  const modal = mode !== 'dock';
  useEffect(() => {
    const active = document.activeElement as HTMLElement | null;
    const opener = active && active !== document.body ? active : pressed;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); closeRef.current(); return; }
      if (!modal || e.key !== 'Tab') return;
      const nodes = [...(box.current?.querySelectorAll<HTMLElement>(focusable) ?? [])];
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (!first) { e.preventDefault(); return; }
      // focus fell out of the panel (the focused control was replaced, e.g. 이전/다음 공): bring it back inside
      if (!box.current?.contains(document.activeElement)) { e.preventDefault(); (e.shiftKey ? last : first).focus(); return; }
      if (e.shiftKey && (document.activeElement === first || document.activeElement === box.current)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && (document.activeElement === last || document.activeElement === box.current)) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', onKey);
    if (modal) document.body.classList.add('v3-locked');
    return () => { window.removeEventListener('keydown', onKey); document.body.classList.remove('v3-locked');
      if (opener?.isConnected && !opener.closest('[hidden]')) { opener.focus?.({ preventScroll: true }); return; }
      // otherwise on the next task, once the closing commit is done (a docked panel's neighbour column is un-hidden in it); skip when
      // a panel is still open (StrictMode's second mount, or another panel opened at once)
      window.setTimeout(() => {
        if (document.querySelector('.v3-panel')) return;
        const back = opener?.isConnected && !opener.closest('[hidden]') ? opener : fallback();
        back?.focus?.({ preventScroll: true });
      }, 0); };
  }, [modal]);
  const body = <div className={`v3-panel ${mode}`} role={modal ? 'dialog' : 'complementary'} aria-modal={modal || undefined} aria-labelledby={id} tabIndex={-1} ref={box}>
    {mode === 'sheet' && <span className="v3-grip" aria-hidden="true" />}
    <div className="v3-panel-head"><h2 id={id}>{title}</h2><button className="v3-btn ghost" onClick={onClose}>닫기</button></div>
    <div className="v3-panel-body">{children}</div>
  </div>;
  if (!modal) return body;
  return <div className={`v3-backdrop ${mode}`} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>{body}</div>;
}
