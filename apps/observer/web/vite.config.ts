import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// `--mode public` (npm run build:public) = the public static-mode bundle: assets are referenced relative to index.html so the bundle
// works under any hosting prefix, and `.env.public` supplies VITE_PUBLISH_* (see src/service/publish.ts). Every other mode is unchanged.
// `__PZ_STATIC__` is the mode as a literal (true when VITE_PUBLISH_BASE is non-empty), so the bundler drops the other mode's code.
const here = decodeURIComponent(new URL('.', import.meta.url).pathname);  // this folder: .env files are read from here (the shell still wins)
// UI v3 (src/v3): `VITE_UI=v3` swaps the screens; the data layer (src/service/api.ts, publish.ts, model.ts) is shared and unchanged.
// `VITE_REPLAY=1` (preview builds only) serves published files from an in-page replay of a fixture game; `VITE_V3_REVIEW=1` shows the
// variant switcher. Both are build-time literals, so a public bundle without them carries none of that code.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, here, 'VITE_');
  const flag = (v: string | undefined) => JSON.stringify(v === '1' || v === 'true');
  const v3 = env.VITE_UI === 'v3';
  // v3 builds start from src/v3/main.tsx, so none of the classic screens or their styles are bundled; classic builds are unchanged.
  const entry = { name: 'pz-entry', transformIndexHtml: { order: 'pre' as const, handler: (html: string) => v3 ? html.replace('/src/main.tsx', '/src/v3/main.tsx') : html } };
  return {
  plugins: [react(), entry],
  base: ['public', 'v3', 'preview'].includes(mode) ? './' : '/',
  define: { __PZ_STATIC__: JSON.stringify(Boolean(env.VITE_PUBLISH_BASE)),
    __PZ_UI__: JSON.stringify(env.VITE_UI === 'v3' ? 'v3' : 'classic'),
    __PZ_REPLAY__: flag(env.VITE_REPLAY), __PZ_REVIEW__: flag(env.VITE_V3_REVIEW) },
  server: { host: '127.0.0.1', proxy: { '/api': 'http://127.0.0.1:8766' } },
  };
});
