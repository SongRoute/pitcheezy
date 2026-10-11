import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `--mode public` (npm run build:public) = the public static-mode bundle: assets are referenced relative to index.html so the bundle
// works under any hosting prefix, and `.env.public` supplies VITE_PUBLISH_* (see src/service/publish.ts). Every other mode is unchanged.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === 'public' ? './' : '/',
  server: { host: '127.0.0.1', proxy: { '/api': 'http://127.0.0.1:8766' } },
}));
