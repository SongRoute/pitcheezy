// Merge several published folders (the `v1/` layout of the public publish contract) into ONE new folder, so a static host or an
// exhibition device can show more than one game from a single `v1/` without the publisher and without the layered preview.
//
//   node tests/publish-merge.mjs --out <new dir> <folder>...      # each <folder> holds v1/ (or is v1/ itself)
//   node tests/publish-merge.mjs --out <new dir> --ws             # the World Series replay folders (fixtures/publish-ws)
//
// Game files (`game/<pk>.json`) and date lists (`games/<date>.json`) are copied byte for byte; the same path with different bytes
// in two folders is refused (nothing is chosen for you). The three index files are joined by tests/publish-join.mjs — the same
// rule tests/ui-preview.mjs applies per request — and written once; the v1 schema is unchanged. `setup/` files are copied when
// present. Nothing else is read or written: --out must not exist or must be an empty directory; the source folders are untouched.
// Prints what it wrote with sha256 (SHA256SUMS form, paths relative to <new dir>). Exit 0 wrote everything, 2 refused.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { INDEX, joinIndex } from './publish-join.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const WS = ['2026-10-24', '2026-10-27', '2026-10-28'].map((d) => path.join(here, 'fixtures', 'publish-ws', d));   // G1 849805, G3 849804, G4 849803
const USAGE = 'usage: node tests/publish-merge.mjs --out <new dir> (<folder>... | --ws)';
const COPIED = ['game', 'games', 'setup'];

export function die(msg) { console.error(msg); process.exit(2); }
const v1Of = (dir) => existsSync(path.join(dir, 'v1', 'manifest.json')) ? path.join(dir, 'v1')
  : existsSync(path.join(dir, 'manifest.json')) ? dir : die(`no v1/manifest.json under ${dir}`);
const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');
const readJson = (p) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return null; } };

/** Merge `folders` (each a dir holding v1/, or a v1/ dir) into `out`/v1. Returns [{ path, sha256, bytes }] in written order. */
export function merge(folders, out) {
  if (!folders.length) die(USAGE);
  const v1s = folders.map((f) => v1Of(path.resolve(f)));
  out = path.resolve(out);
  if (existsSync(out) && (!statSync(out).isDirectory() || readdirSync(out).length)) die(`--out must not exist or must be an empty directory: ${out}`);
  // the sources stay untouched: --out is never inside a source folder (nor a source inside --out)
  const within = (a, b) => a === b || a.startsWith(b + path.sep);
  for (const f of folders.map((x) => path.resolve(x))) if (within(out, f) || within(f, out)) die(`refused: --out ${out} overlaps the source folder ${f}`);
  // 1. collect the copied files first; a conflict refuses the whole merge before anything is written
  const files = new Map();   // relative path under v1 → { from, bytes }
  for (const v1 of v1s) for (const sub of COPIED) {
    const dir = path.join(v1, sub);
    if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
    for (const name of readdirSync(dir).filter((f) => f.endsWith('.json')).sort()) {
      const rel = path.posix.join(sub, name), from = path.join(dir, name), bytes = readFileSync(from);
      const seen = files.get(rel);
      if (seen && !seen.bytes.equals(bytes)) die(`refused: ${rel} differs between ${seen.from} and ${from}`);
      if (!seen) files.set(rel, { from, bytes });
    }
  }
  // 2. the joined index files (a folder without one of them simply contributes nothing to it)
  const joined = INDEX.map((name) => [name, joinIndex(name, v1s.map((v1) => readJson(path.join(v1, name))))]).filter(([, body]) => body);
  if (!joined.some(([name]) => name === 'manifest.json')) die('refused: no manifest.json in any folder');
  // 3. write
  const written = [];
  const put = (rel, bytes) => { const p = path.join(out, 'v1', rel); mkdirSync(path.dirname(p), { recursive: true }); writeFileSync(p, bytes, { flag: 'wx' }); written.push({ path: path.posix.join('v1', rel), sha256: sha256(bytes), bytes: bytes.length }); };
  for (const [name, body] of joined) put(name, Buffer.from(JSON.stringify(body)));
  for (const [rel, { bytes }] of [...files].sort(([a], [b]) => a.localeCompare(b))) put(rel, bytes);
  return written;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const folders = []; let out = null, ws = false;
  for (let i = 2; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (a === '--out') { out = process.argv[++i]; if (out === undefined) die('--out needs a value'); }
    else if (a === '--ws') ws = true;
    else if (a === '-h' || a === '--help') { console.log(USAGE); process.exit(0); }
    else if (a.startsWith('-')) die(`unknown option ${a}\n${USAGE}`);
    else folders.push(a);
  }
  if (!out) die(USAGE);
  if (ws) folders.push(...WS);
  const written = merge(folders, out);
  for (const w of written) console.log(`${w.sha256}  ${w.path}`);
  console.error(`\nwrote ${written.length} files under ${path.resolve(out)}/v1 (${written.reduce((a, w) => a + w.bytes, 0)} bytes)`);
  console.error(`serve: node tests/ui-preview.mjs --fixture ${path.resolve(out)} --clock fixture`);
}
