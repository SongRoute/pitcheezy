# Disposition — Opus review of catcher mitt in WatchAlong (`774a350`, review `c3c40c7`)

`same_model_reviewed; Codex_review_pending`. Coordinator worker (Claude Opus 5.5), 2026-10-04 cycle-003 (v4), task T52.
Branch `codex/catcher-integ-1004` from showcase HEAD `cc31d6e`, fast-forwarded to `c3c40c7` (author `774a350`, reviewer test `14789fa`, review doc).

## Decisions
- No blocking finding in the review. Author diff read in full; it matches the review's description (gate on this pitch's reveal,
  exact `game:at_bat:pitch` key, other-game answers dropped, 404/503/abort draw nothing, abstained pitches say so).
- C1 (`14789fa`, separator-less/swapped key decoys): ACCEPTED.
- N1 (mitt drawn to ±1.7 ft while the ball dot is pinned at ±1.1 ft): APPLIED. `SETUP_X_DRAWN = 1.1`, so both marks share one bound;
  beyond it the existing "존 그림 밖" sentence describes the mitt. Real 849843 x range −1.029…0.696 ft (review), so all 27 estimated
  rows are still drawn. Smoke case G changed from 2.5 ft to 1.3 ft so the new bound is tested.
- N2 (visible legend does not say AI/unreviewed): APPLIED. Legend: "점선 = 포수 미트 위치 추정 (AI·검토 전, 가로만)".
- N3 (`reveal.index === decision.index` untestable): KEPT as a harmless guard, no test added.
- N4 (Zone aria-label mentions recommendations when there are none): APPLIED (label "포수 시점 스트라이크존: 실제 투구" when no candidates).

## Checks (Mac mini, `nice -n 10`, `perl alarm 600`, node_modules = temporary symlink to main's, nothing installed)
Working dir `apps/observer/web`. Build outside the served dist: `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/t52-dist`
(js `index-CRJ8qk-l.js` sha256 a6f01db5fc10da265208e7409f7ee1675ba289f3510aa2855b0e4cb7d6793e64, css `index-CjSYbsNo.css` sha256 384e8bf9597cc2557e706ac3d05084b12c46de6f7127a978b9b3dc0bdf56eca3).

| command | exit |
|---|---|
| `npx tsc -b` | 0 |
| `npx vite build --outDir /Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/t52-dist --emptyOutDir` | 0 |
| `PZ_WEB_DIST=… node tests/watch-setup-smoke.mjs <shots>` → `ok`, 22 cases (before case G change) | 0 |
| `PZ_WEB_DIST=… node tests/pre-reason-neutral-smoke.mjs` → `ok` | 0 |
| `PZ_WEB_DIST=… node tests/watch-spoiler-smoke.mjs` → `ok` | 0 |
| `DIST=… CV_M3_RECEIPT=<main>/results/checks/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → receipt identity ok | 0 |
| `DIST=… MODEL_EVIDENCE_REPO=<main> node tests/model-evidence-smoke.mjs` → source identity ok | 0 |
| `PZ_WEB_DIST=… node tests/game-frame-smoke.mjs` → `ok` | 0 |
| after case G = 1.3 ft: `watch-setup-smoke.mjs` on this build → `ok` | 0 |
| negative control: same smoke on a build of the 774a350 `WatchAlong.tsx` → `FAIL 390 G beyond picture` | 1 |
| `showcase-smoke.mjs` (temporary `dist` symlink to the build, unlinked after) → `ok` | 0 |

## Not verified here
Actual 8771 `/watch?game=849843` and `/showcase` against the real setup response (done in the integration step, recorded in the
coordinator ledger T52). Physical iPhone/Safari. Left/right direction and units against the video or a person.
