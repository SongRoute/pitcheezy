/// <reference types="vite/client" />
/** Build-time literal (vite.config.ts): true = public static-mode bundle (VITE_PUBLISH_BASE set), false = API mode. */
declare const __PZ_STATIC__: boolean;
/** Build-time literal: which screen set is bundled ('classic' = src/service screens, 'v3' = src/v3 screens). */
declare const __PZ_UI__: 'classic' | 'v3';
/** Build-time literal: preview build that answers published-file reads from an in-page fixture replay (never in a public bundle). */
declare const __PZ_REPLAY__: boolean;
/** Build-time literal: preview build that shows the v3 variant switcher. */
declare const __PZ_REVIEW__: boolean;
