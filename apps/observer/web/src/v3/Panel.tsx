import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';

/** One detail container in three forms:
 *   sheet    phone: modal bottom sheet over the live view (focus trapped, Esc / backdrop / 닫기 close, focus returns to the opener)
 *   overlay  tablet: modal side sheet
 *   dock     PC: a docked side panel, NOT modal — the live view beside it keeps updating and stays usable (spec §4) */
export type PanelMode = 'sheet' | 'overlay' | 'dock';
export default function Panel({ mode, title, onClose, children, id = 'v3-panel-title' }: { mode: PanelMode; title: string; onClose: () => void; children: ReactNode; id?: string }) {
  const box = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose); closeRef.current = onClose;
  const modal = mode !== 'dock';
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); closeRef.current(); return; }
      if (!modal || e.key !== 'Tab') return;
      const nodes = [...(box.current?.querySelectorAll<HTMLElement>('button:not([disabled]),a[href],select,input,[tabindex="0"]') ?? [])];
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (!first) { e.preventDefault(); return; }
      if (e.shiftKey && (document.activeElement === first || document.activeElement === box.current)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && (document.activeElement === last || document.activeElement === box.current)) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', onKey);
    if (modal) document.body.classList.add('v3-locked');
    return () => { window.removeEventListener('keydown', onKey); document.body.classList.remove('v3-locked'); if (opener?.isConnected) opener.focus?.(); };
  }, [modal]);
  const body = <div className={`v3-panel ${mode}`} role={modal ? 'dialog' : 'complementary'} aria-modal={modal || undefined} aria-labelledby={id} tabIndex={-1} ref={box}>
    {mode === 'sheet' && <span className="v3-grip" aria-hidden="true" />}
    <div className="v3-panel-head"><h2 id={id}>{title}</h2><button className="v3-btn ghost" onClick={onClose}>닫기</button></div>
    <div className="v3-panel-body">{children}</div>
  </div>;
  if (!modal) return body;
  return <div className={`v3-backdrop ${mode}`} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>{body}</div>;
}
