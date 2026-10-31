// Replay preview of the PUBLIC screen (not a public bundle; never deployed). Build with:
//   npx vite build --config vite.preview.config.ts --mode public
// It reuses vite.config.ts unchanged (public mode: relative assets, .env.public, __PZ_STATIC__ = true) and only swaps index.html's
// entry to src/preview/main-replay.tsx, which serves the screen's v1 file reads from a replay of the real finished game 849843.
import { defineConfig, mergeConfig } from 'vite';
import type { ConfigEnv, UserConfig } from 'vite';
import base from './vite.config';

export default defineConfig((env: ConfigEnv) => {
  const cfg = (typeof base === 'function' ? base({ ...env, mode: 'public' }) : base) as UserConfig;
  return mergeConfig(cfg, {
    plugins: [{ name: 'pz-preview-entry', transformIndexHtml: { order: 'pre' as const, handler: (html: string) => html.replace('/src/main.tsx', '/src/preview/main-replay.tsx') } }],
    build: { outDir: 'dist-preview-classic', emptyOutDir: true },
  });
});
