// Self-test of tests/publish-merge.mjs and tests/publish-join.mjs on the committed World Series fixtures (no browser, no network).
//   node tests/publish-merge-check.mjs
// Merges fixtures/publish-ws/{2026-10-24,2026-10-27,2026-10-28} into a temporary folder and checks: every copied game / date file is
// byte-identical to its source; the joined index files keep the v1 schema (same keys as the newest source file) with the lists unioned
// and game_checks combined; a conflicting copy is refused; --out that is not empty is refused. The temporary folder is removed.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { INDEX, joinIndex } from './publish-join.mjs';
import { WS, merge } from './publish-merge.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const sha = (b) => createHash('sha256').update(b).digest('hex');
const read = (p) => JSON.parse(readFileSync(p, 'utf8'));
const tmp = mkdtempSync(path.join(os.tmpdir(), 'pz-publish-merge-'));
let failed = false;
try {
  const out = path.join(tmp, 'merged');
  const written = merge(WS, out);
  const paths = written.map((w) => w.path);
  // 1. copied files: byte for byte
  for (const d of WS) for (const sub of ['game', 'games']) {
    const dir = path.join(d, 'v1', sub);
    for (const name of ['849805.json', '849804.json', '849803.json', '2026-10-24.json', '2026-10-27.json', '2026-10-28.json']) {
      const src = path.join(dir, name);
      let bytes; try { bytes = readFileSync(src); } catch { continue; }
      const copy = readFileSync(path.join(out, 'v1', sub, name));
      assert.ok(copy.equals(bytes), `${sub}/${name} copied byte for byte`);
      assert.ok(paths.includes(`v1/${sub}/${name}`), `${sub}/${name} listed`);
    }
  }
  assert.equal(written.filter((w) => w.path.startsWith('v1/game/')).length, 3, 'three game files');
  assert.equal(written.filter((w) => w.path.startsWith('v1/games/')).length, 3, 'three date files');
  // 2. joined index files: v1 schema kept (same keys as the newest source), lists unioned, checks combined
  const sources = (name) => WS.map((d) => read(path.join(d, 'v1', name)));
  for (const name of INDEX) {
    const merged = read(path.join(out, 'v1', name)), newest = sources(name).reduce((a, b) => (Date.parse(b.published_at ?? b.publish?.published_at) > Date.parse(a.published_at ?? a.publish?.published_at) ? b : a));
    assert.deepEqual(Object.keys(merged).sort(), Object.keys(newest).sort(), `${name}: same keys as the newest source`);
    assert.deepEqual(merged.publish, newest.publish, `${name}: publish stamp of the newest source`);
    assert.deepEqual(merged, joinIndex(name, sources(name)), `${name}: exactly what ui-preview joins per answer`);
  }
  const manifest = read(path.join(out, 'v1', 'manifest.json')), health = read(path.join(out, 'v1', 'health.json')), dates = read(path.join(out, 'v1', 'dates.json'));
  assert.deepEqual(manifest.live_dates, ['2026-10-24', '2026-10-27', '2026-10-28'], 'live_dates unioned in folder order');
  assert.equal(manifest.commit, 'e53b89a7ec82da4dc101c945bc586b298d61a853', 'other manifest fields from the newest file');
  assert.deepEqual(Object.keys(health.game_checks).sort(), ['849803', '849804', '849805'], 'game_checks combined');
  for (const k of ['unavailable_games', 'unavailable_dates', 'feed_failing_games']) assert.deepEqual(health[k], [], `${k} union of empty lists`);
  assert.deepEqual(dates.dates, [], 'dates stays what the publisher wrote (empty in every run)');
  // 3. the listing's sha256 is the file's
  for (const w of written) assert.equal(sha(readFileSync(path.join(out, w.path))), w.sha256, `${w.path} sha`);
  // 4. refusals (in a child process: merge() exits the process)
  const run = (args) => spawnSync(process.execPath, [path.join(here, 'publish-merge.mjs'), ...args], { encoding: 'utf8' });
  assert.equal(run(['--out', out, '--ws']).status, 2, 'a non-empty --out is refused');
  const conflict = path.join(tmp, 'conflict', 'v1'); mkdirSync(path.join(conflict, 'game'), { recursive: true });
  writeFileSync(path.join(conflict, 'manifest.json'), JSON.stringify({ schema: 'pitcheezy-publish-manifest-v1', live_dates: ['2026-10-27'], published_at: '2026-10-27T03:25:49Z' }));
  writeFileSync(path.join(conflict, 'game', '849804.json'), '{"not":"the same bytes"}');
  const r = run(['--out', path.join(tmp, 'out2'), WS[1], path.join(tmp, 'conflict')]);
  assert.equal(r.status, 2, 'a game file with other bytes is refused'); assert.match(r.stderr, /refused: game\/849804\.json differs/);
  const inside = run(['--out', path.join(tmp, 'conflict', 'v1', 'merged-here'), path.join(tmp, 'conflict')]);
  assert.equal(inside.status, 2, 'an --out inside a source folder is refused'); assert.match(inside.stderr, /overlaps the source folder/);
  assert.equal(readdirSync(path.join(tmp, 'conflict', 'v1')).sort().join(','), 'game,manifest.json', 'nothing was written into the source');
  console.log(`publish-merge-check: ok (${written.length} files, ${written.reduce((a, w) => a + w.bytes, 0)} bytes)`);
} catch (e) { failed = true; console.error(e); }
finally { rmSync(tmp, { recursive: true, force: true }); }
process.exit(failed ? 1 : 0);
