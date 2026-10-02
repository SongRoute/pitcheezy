// Serves this checkout's built screens on their own port and forwards /api to a running observer service,
// so a new screen can be tried on a phone without touching (or re-loading the model in) the running service.
//   npm run build && node viewer.mjs [host] [port] [api]
//   node viewer.mjs 100.x.y.z 8770 http://100.x.y.z:8766   → http://100.x.y.z:8770/game?game=<gamePk>
import { preview } from 'vite';

const [host = '127.0.0.1', port = '8770', api = 'http://127.0.0.1:8766'] = process.argv.slice(2);
// Same rule as serve.sh: private addresses only (loopback or Tailscale 100.x), never a public one.
if (!/^(127\.0\.0\.1|localhost|::1|100\.\d+\.\d+\.\d+)$/.test(host)) {
  console.error(`비공개 주소(127.0.0.1 또는 Tailscale 100.x)만 허용합니다: ${host}`);
  process.exit(1);
}
const server = await preview({ preview: { host, port: Number(port), strictPort: true, proxy: { '/api': api } } });
server.printUrls();
console.log(`  ➜  API:     ${api}`);
