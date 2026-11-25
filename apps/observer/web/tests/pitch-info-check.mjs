// Pitch-type averages in src/service/pitchInfo.ts against the measured 2026 regular-season means (frozen statcast_2026 parquet,
// 716,792 pitches, release_speed dropna; PM recompute 2026-11-19 with pyarrow). km/h = mean mph × 1.609344 rounded half up to a whole
// number; mph = mean rounded to 1 decimal. No build, no browser: reads the source file.
//   node tests/pitch-info-check.mjs
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '..', 'src', 'service', 'pitchInfo.ts'), 'utf8');
const MEAN_MPH = { FF: 94.773, SI: 94.139, FC: 89.816, SL: 86.341, CH: 86.035, FS: 86.633, ST: 82.813, KC: 82.607, SV: 82.222, CU: 80.278 };
const MEAN_KMH = { FF: 152.52, SI: 151.50, FC: 144.54, SL: 138.95, CH: 138.46, FS: 139.42, ST: 133.27, KC: 132.94, SV: 132.32, CU: 129.20 };
let failed = 0;
const ok = (c, m) => { if (!c) failed++; console.log(`${c ? 'ok  ' : 'FAIL'} ${m}`); };
const rows = Object.fromEntries([...src.matchAll(/^\s+([A-Z]{2}): \{ desc: '[^']+', avgKmh: (\d+), avgMph: ([\d.]+) \}/gm)].map((m) => [m[1], { kmh: Number(m[2]), mph: m[3] }]));
ok(Object.keys(rows).sort().join() === Object.keys(MEAN_MPH).sort().join(), `exactly the 10 measured types (${Object.keys(rows).join(',')})`);
for (const [t, mph] of Object.entries(MEAN_MPH)) {
  const r = rows[t];
  ok(r && r.kmh === Math.floor(MEAN_KMH[t] + 0.5), `${t}: ${r?.kmh} km/h = ${MEAN_KMH[t]} rounded half up`);
  ok(r && Math.abs(MEAN_KMH[t] - mph * 1.609344) < 0.01, `${t}: the km/h mean is the mph mean × 1.609344`);
  ok(r && r.mph === mph.toFixed(1), `${t}: ${r?.mph} mph = ${mph} to 1 decimal`);
}
console.log(failed ? `\n${failed} FAILED` : '\nall passed');
process.exitCode = failed ? 1 : 0;
