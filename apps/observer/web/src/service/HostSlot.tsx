import { useEffect, useRef } from 'react';

/** Host extension point (public bundle only). Pitcheezy ships NO analytics, feedback form or video; the hosting page owns those.
 *  This is the single, deliberately empty place where the host may mount its own snippet:
 *
 *    <div id="pitcheezy-host-slot" data-screen="list|game" data-game-pk="<pk>"></div>     (rendered after the screen, zero height)
 *
 *  and, once it is in the document (and again when the screen changes), a window event:
 *
 *    window.addEventListener('pitcheezy:host-slot', (e) => { e.detail.element; e.detail.screen; e.detail.gamePk; });
 *
 *  React never writes children into the element, so whatever the host appends stays. Nothing here sends or stores anything. */
export const HOST_SLOT_ID = 'pitcheezy-host-slot';
export const HOST_SLOT_EVENT = 'pitcheezy:host-slot';
export default function HostSlot({ screen, gamePk }: { screen: 'list' | 'game'; gamePk: number | null }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    window.dispatchEvent(new CustomEvent(HOST_SLOT_EVENT, { detail: { element: ref.current, screen, gamePk } }));
  }, [screen, gamePk]);
  return <div id={HOST_SLOT_ID} ref={ref} data-screen={screen} data-game-pk={gamePk ?? undefined} />;
}
