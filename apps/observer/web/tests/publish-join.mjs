// How the three index files of SEVERAL published folders (the `v1/` layout of the public publish contract) become one answer.
// Shared by tests/ui-preview.mjs (joined per request, never on disk) and tests/publish-merge.mjs (written once into a new folder),
// so a merged folder served as one fixture shows exactly what the layered preview showed. The v1 schema is unchanged: every key
// keeps its name and shape; only list fields are unioned and `game_checks` is combined.
//
//   dates.json     `dates` = union (each value once, first-seen order)
//   manifest.json  `live_dates` = union
//   health.json    `game_checks` = combined per game; `unavailable_games`, `unavailable_dates`, `feed_failing_games` = union
//   every other field comes from the folder whose file was published last (`published_at`, else `publish.published_at`).
export const INDEX = ['dates.json', 'manifest.json', 'health.json'];

const isObject = (j) => j !== null && typeof j === 'object' && !Array.isArray(j);
const publishedAt = (j) => Date.parse(j.published_at ?? j.publish?.published_at ?? '') || 0;

/** `bodies`: the parsed file from each folder, in folder order (null/undefined for a folder without it). Returns a NEW object
 *  (the inputs are not changed), or null when no folder has the file. One body comes back as a copy of itself. */
export function joinIndex(name, bodies) {
  if (!INDEX.includes(name)) throw new Error(`not an index file: ${name}`);
  const all = (bodies ?? []).filter(isObject);
  if (!all.length) return null;
  const newest = all.reduce((a, b) => (publishedAt(b) > publishedAt(a) ? b : a));
  const out = structuredClone(newest);
  const union = (k) => { const seen = new Map(); for (const j of all) if (Array.isArray(j[k])) for (const v of j[k]) seen.set(JSON.stringify(v), v); return structuredClone([...seen.values()]); };
  if (name === 'dates.json') out.dates = union('dates');
  if (name === 'manifest.json') out.live_dates = union('live_dates');
  if (name === 'health.json') {
    out.game_checks = structuredClone(Object.assign({}, ...all.map((j) => (isObject(j.game_checks) ? j.game_checks : {}))));
    for (const k of ['unavailable_games', 'unavailable_dates', 'feed_failing_games']) out[k] = union(k);
  }
  return out;
}
