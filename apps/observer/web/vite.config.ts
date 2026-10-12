import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// `--mode public` (npm run build:public) = the public static-mode bundle: assets are referenced relative to index.html so the bundle
// works under any hosting prefix, and `.env.public` supplies VITE_PUBLISH_* (see src/service/publish.ts). Every other mode is unchanged.
// `__PZ_STATIC__` is the mode as a literal (true when VITE_PUBLISH_BASE is non-empty), so the bundler drops the other mode's code.
const here = decodeURIComponent(new URL('.', import.meta.url).pathname);  // this folder: .env files are read from here (the shell still wins)
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === 'public' ? './' : '/',
  define: { __PZ_STATIC__: JSON.stringify(Boolean(loadEnv(mode, here, 'VITE_').VITE_PUBLISH_BASE)) },
  server: { host: '127.0.0.1', proxy: { '/api': 'http://127.0.0.1:8766' } },
}));
