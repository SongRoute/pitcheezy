# Disposition — same-model review of /watch spoiler-free navigation (f169c14)

Label: same_model_reviewed; Codex_review_pending. Review: `docs/reviews/OPUS-review-watch-spoiler-f169c14-2026-10-04.md`
(distinct Opus 5.5 reviewer, commit 5fcdc04 on `codex/watch-spoiler-review-1004`, returned 11:40:46 KST, 342 s).
Disposition by the delegated Opus PM worker, cycle-024, 2026-10-04. Not integrated or deployed; root/Song decide option 1/2/3.

- F1 (low, fixed by reviewer): shared `#i=N` link's furthest pitch not persisted → cherry-picked as `902ca77` (persist
  `furthest` whenever it changes; no write in embed mode).
- F2 (test gap, fixed by reviewer): per-game PA count + shared-link reload case + clean exit → cherry-picked in `902ca77`.
- F3 (data gap jumps to the last pitch): not applied. Real datasets have no regular-half gaps (audit T23: 17–20 halves
  each); fallback choice left to root (reviewer's next-half snippet is in the review).
- F4 (two "구째" labels one line apart): applied. Nav progress now `경기 {n}번째 공`; the at-bat label stays `n구째`.
  Smoke pins the wording (`경기 1번째 공`).
- F5 (unplayed 9말 jumps to the last pitch): accepted as designed; optional one-line notice left to root.
- F6: info, no change.

Checks (Mac, light; temporary node_modules symlink to the main checkout, removed):
- `tsc --noEmit -p tsconfig.json` → 0
- `vite build --outDir /tmp/pz-watch-spoiler-t24-dist` → ok
- `PZ_WEB_DIST=/tmp/pz-watch-spoiler-t24-dist node tests/watch-spoiler-smoke.mjs` → ok
- negative: same smoke on the reviewer's build `/tmp/pz-watch-spoiler-review-fix-dist` → `FAIL progress wording differs … 1구째`, exit 1
- `showcase-smoke` (temp copy reading PZ_WEB_DIST, deleted) on the new build → ok (embed has no navigation)
- not run: `watch-along-smoke.mjs` (needs a live observer), real iPhone/Safari (미측정).
