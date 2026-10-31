# Disposition — Opus review of the offline local video copy (7fae5bf → a2ee475), 2026-10-04

Status: `same_model_reviewed; Codex_review_pending`. Author, reviewer and this PM disposition are all Opus 5.5 instances (no other model).
Review: `apps/observer/web/offline/OPUS-review-offline-media-7fae5bf-2026-10-04.md` (helper 79539, exit 0, permission_denials []). Verdict there: no BLOCKER/MAJOR, 5 MINOR, 6 NOTE.

## Decisions
- M1 stale sum line refused before download (`d3f300f`) — ACCEPT (read the diff: `sum_lines` precheck in `main`, same check reused by `add_sum`).
- M2 chunk/disconnect tests (`be1a156`) — ACCEPT (tests only, no server code change).
- M3 `?media=local` lost via in-app links — ACCEPT the README note (`b305cd4`) for 10-06. Code propagation (sessionStorage etc.) deferred: it touches default-mode links and needs its own review.
- M4 smoke launches Chromium inside try (`be7871e`) — ACCEPT.
- M5 note text before playback — FIXED here: `src/Showcase.tsx:123` and the smoke's expected text now say "맥북에 저장한 같은 공식 영상 사본을 씁니다" (no "재생 중" claim before play). Copy only; no logic change.
- N1 Safari with HTTP/1.0 range server — open: real Safari rehearsal with the real file, outside Mac protected windows (manual checklist item).
- N2 rename window, N3 restart after file change, N5 import-time constant, N6 header-write disconnect — accepted as is.
- N4 no independent sha for the official file — when the real file is fetched, record the printed sha256 in the coordinator ledger.

## Checks after the disposition (PM, 2026-10-04 ≈18:27 KST, Mac mini, OMP/UV threads 1, nice 10)
- branch fast-forwarded 7fae5bf → a2ee475, then the M5 commit.
- `offline/`: `pytest -q -p no:cacheprovider test_offline_bundle.py test_offline_media.py test_fetch_media.py` → 81 passed, exit 0 (reviewer's count reproduced).
- `npx tsc -b` → exit 0; `npx vite build --outDir /tmp/pz-t64-offline-media-build` → exit 0 (node_modules = temporary symlink to codex-showcase-1006, unlinked after).
  - `index.html` c30deadb3bbc4331fbcdd837249db8c2b4e41606cf8880e421ff9eb212b83cb1
  - `assets/index-BFm0H7T_.js` 294eb2c6b6599a108941072297f1bfb60a9adce793cd9b1d9fe12357488aa56d
  - `assets/index-BGk569ec.css` c624c910496a1dd0b9c56abb372fdd4e3e6be4a1cf41422e7a50c24ba026e1db (unchanged)
- `PZ_WEB_DIST=<build> node tests/showcase-local-media-smoke.mjs` → exit 0 (decoded 390/1440 ready, mp4 702,685 bytes, synthetic).
- `showcase-smoke.mjs` → exit 0; `showcase-phone-reveal-smoke.mjs` → exit 0 (390x844, 390x664, 1440x900).

## Not done / limits
- Real 436 MB MLB file not downloaded; no request to mlb.com. Safari/iPhone unchecked. Not integrated into codex-showcase-1006 or 8771 yet (no deploy in this unit).
- The M5 copy change itself has not had a separate review (two-word UI copy; covered by the smoke's exact-text assertion).
