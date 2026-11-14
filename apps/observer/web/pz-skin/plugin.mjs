// pz-skin/plugin.mjs — Vite build plugin that reproduces the teammate's "songskin" layout (PR 90, transition-models
// claude/exhibit-records-1112 @ 10988ce6) from source: a `pz/` folder next to the built index.html plus three tags in index.html.
//
//   PZ_SKIN=off (default) no skin; index.html and assets are exactly the plain build (= c47 bytes). Review P2-4 (2026-11-15):
//                         a build without variables never changes, so the operating / publishing builds stay as they were.
//   PZ_SKIN=3d            colour theme + 3D batter: pz/pz-theme.css, pz/pz-boot.js, pz/pz-batter.js, pz/batter3d/** (three.js 0.187.0)
//   PZ_SKIN=theme         colour theme only (teammate theme-only/): pz/pz-theme.css
//   PZ_BATTER_MODEL       model base name written into pz-batter.js and pz-boot.js (`batter-ctl` | `batter-indexed`). Unset = the
//                         placeholder `__MODEL__` is kept and pz-boot.js never loads the 3D code (review P2-3: no 3D request at all,
//                         the silhouette stays).
//   PZ_BATTER_ASSETS_DIR  optional folder OUTSIDE the repository holding `<model>.glb` (+ `<model>.meta.json`); copied to
//                         pz/batter3d/assets/. Model and meta files are never committed.
//
// Tags (teammate index.html, 926 B): an import map before the first module script; after the app CSS a link to pz/pz-theme.css and
// a module script. Change ③ (2026-11-15): that script is pz/pz-boot.js, which imports pz/pz-batter.js (and with it three.js) only once
// a game-screen zone row exists, so the list screen never downloads the 3D code.
// three.js is copied from node_modules/three (package-lock pins 0.187.0) and every copied file is checked against the SHA-256 the
// teammate README lists (identical to the npm tarball, checked 2026-11-15); a mismatch fails the build.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.dirname(here);

const THREE_FILES = [
  ['build/three.module.js', 'vendor/three.module.js', '104f56145899722094556490d787aae0617cc2fc3382f31df02d724ea1aee341'],
  ['build/three.core.js', 'vendor/three.core.js', '613e370f76638fdef83926642bf256dc9615684922de1ce1f63a8583e5d78578'],
  ['examples/jsm/loaders/GLTFLoader.js', 'vendor/addons/loaders/GLTFLoader.js', '532ac726e120039acbcd6cf54328c57acbf7b438176afaa6951257126d11bff1'],
  ['examples/jsm/utils/BufferGeometryUtils.js', 'vendor/addons/utils/BufferGeometryUtils.js', '7bda2a8ba836b2bcf67cb938d4b9e1bb5d1b1bc616817103ed858c0b58296ce7'],
  ['examples/jsm/utils/SkeletonUtils.js', 'vendor/addons/utils/SkeletonUtils.js', 'b1632a703206c3d830de9fcbe515696770d04b71a15ee6b50afa6d2c3298c86f'],
  ['LICENSE', 'vendor/LICENSE', '8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc'],
];
const IMPORT_MAP = '<script type="importmap">{ "imports": { "three": "./pz/batter3d/vendor/three.module.js", "three/addons/": "./pz/batter3d/vendor/addons/" } }</script>';
const THEME_LINK = '<link rel="stylesheet" href="./pz/pz-theme.css">';
const BOOT_SCRIPT = '<script type="module" src="./pz/pz-boot.js"></script>';
const MODEL_RE = /^[a-z0-9][a-z0-9-]{0,63}$/;

const sha = (buf) => createHash('sha256').update(buf).digest('hex');

export function skinConfig(env = process.env) {
  const skin = (env.PZ_SKIN || 'off').trim();
  if (!['3d', 'theme', 'off'].includes(skin)) throw new Error(`PZ_SKIN must be 3d | theme | off, got ${JSON.stringify(skin)}`);
  const model = (env.PZ_BATTER_MODEL || '').trim();
  if (model && !MODEL_RE.test(model)) throw new Error(`PZ_BATTER_MODEL must match ${MODEL_RE}, got ${JSON.stringify(model)}`);
  const assetsDir = (env.PZ_BATTER_ASSETS_DIR || '').trim();
  return { skin, model, assetsDir };
}

/** The pz/ files for a configuration, as [fileName, Buffer]. Exported for the checks in tests/pz-skin-check.mjs. */
export function skinFiles(cfg = skinConfig()) {
  const out = [];
  if (cfg.skin === 'off') return out;
  if (cfg.skin === 'theme') { out.push(['pz/pz-theme.css', readFileSync(path.join(here, 'theme-only', 'pz-theme.css'))]); return out; }
  out.push(['pz/pz-theme.css', readFileSync(path.join(here, 'pz-theme.css'))]);
  for (const name of ['pz-boot.js', 'pz-batter.js']) {   // both carry the model name; pz-boot.js loads nothing without one
    const src = readFileSync(path.join(here, name), 'utf8');
    if (!src.includes('const MODEL = "__MODEL__";')) throw new Error(`${name}: model placeholder line not found`);
    out.push([`pz/${name}`, Buffer.from(cfg.model ? src.replace('const MODEL = "__MODEL__";', `const MODEL = "${cfg.model}";`) : src, 'utf8')]);
  }
  out.push(['pz/batter3d/batter3d.js', readFileSync(path.join(here, 'batter3d', 'batter3d.js'))]);
  const threeDir = path.join(web, 'node_modules', 'three');
  for (const [from, to, want] of THREE_FILES) {
    const buf = readFileSync(path.join(threeDir, from)), got = sha(buf);
    if (got !== want) throw new Error(`three.js ${from}: sha256 ${got} != expected ${want} (npm ci with the committed package-lock?)`);
    out.push([`pz/batter3d/${to}`, buf]);
  }
  if (cfg.assetsDir) {
    if (!cfg.model) throw new Error('PZ_BATTER_ASSETS_DIR needs PZ_BATTER_MODEL');
    const glb = path.join(cfg.assetsDir, `${cfg.model}.glb`), meta = path.join(cfg.assetsDir, `${cfg.model}.meta.json`);
    if (!existsSync(glb)) throw new Error(`model file missing: ${glb}`);
    out.push([`pz/batter3d/assets/${cfg.model}.glb`, readFileSync(glb)]);
    if (existsSync(meta)) out.push([`pz/batter3d/assets/${cfg.model}.meta.json`, readFileSync(meta)]);
  }
  return out;
}

/** index.html with the skin tags (pure string function; exported for tests). */
export function skinHtml(html, cfg = skinConfig()) {
  if (cfg.skin === 'off') return html;
  const THEME_META = '<meta name="theme-color" content="#f5f7f8" />';
  if (!html.includes(THEME_META)) throw new Error('index.html: theme-color meta not found');   // review P3-5: fail like the other tags
  let h = html.replace(THEME_META, '<meta name="theme-color" content="#091811" />');
  if (cfg.skin === '3d') {
    const i = h.indexOf('<script type="module"');
    if (i < 0) throw new Error('index.html: no module script to put the import map before');
    h = h.slice(0, i) + IMPORT_MAP + '\n    ' + h.slice(i);
  }
  const tags = cfg.skin === '3d' ? `${THEME_LINK}\n    ${BOOT_SCRIPT}` : THEME_LINK;
  const j = h.indexOf('</head>');
  if (j < 0) throw new Error('index.html: no </head>');
  return h.slice(0, j) + '  ' + tags + '\n  ' + h.slice(j);
}

export default function pzSkin() {
  const cfg = skinConfig();
  return {
    name: 'pz-skin',
    apply: 'build',
    transformIndexHtml: { order: 'post', handler: (html) => skinHtml(html, cfg) },
    generateBundle() {
      for (const [fileName, source] of skinFiles(cfg)) this.emitFile({ type: 'asset', fileName, source });
    },
  };
}
