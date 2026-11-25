# Retired web smokes (2026-11-19, PM decision D208)

These six scripts were moved here unchanged and are no longer run. Each one checks a screen or contract that an
approved redesign removed or reversed, so updating its selectors could not keep its original intent. They are kept
for history only. Their paths still assume `tests/` (`../dist`, `fixtures/`), so they do not run from this folder.

Status at retirement (HEAD before the move, bundle H2 source): every one stopped at its first check, or failed and then
stopped, both before and after bundle H.

| Script | Why retired | Current checks covering the intent |
|---|---|---|
| `game-frame-smoke.mjs` | Tests the old `/game` screen (`Game.tsx`, `.gm-*`). `/game/{id}` now opens the service `GameScreen` and `Game.tsx` is not bundled. | `display-state-smoke` (states, stale/recovery), `static-publish-smoke` (live/late/lost/recovery), `ui-1026-capture` (every state and sheet per width) |
| `game-demo-regression.mjs` | Same old `/game` screen: stale feed, catcher setup, replay scene jump and pause, `/watch → /game` link. | `static-publish-smoke` (late / lost feed and recovery), `display-state-smoke`, `pz-skin-check` (live replay) |
| `model-evidence-smoke.mjs` | Tests `ModelEvidenceCard` inside `Showcase`. Showcase is not bundled since review-integration-002. Its text expectations were updated in `60a99d1d` before the move. | None on screen. The service "추천 모델 소개" sheet (bundle H) is covered by `ui-1026-capture` and `static-publish-smoke` |
| `responsive-fit-smoke.mjs` | Contract was "one screen, NO page scroll" with the pre-10/13 blocks (`.sv-board`, strip). The 2026-10-13 redesign allows vertical scroll under the bottom tab bar, and UI 10/26 replaced the blocks. | `compact-rev3-fit` (migrated to the 10/13 contract; key info above the fold, no sideways overflow, 125% text), `ui-1026-capture` (390/360/320/402 and wide), `display-state-smoke` (`fitsViewport`) |
| `live-pinned-stale-fit.mjs` | Fits a LIVE game pinned to a past pitch. In UI 10/26 the live stage never follows a selected past pitch (no pin / "최신으로"), and its `.sv-board` layout is gone. | `static-publish-smoke` ("pinned": live stage keeps the latest behind a past-pitch sheet; stale/late states), `compact-rev3-fit` |
| `service-ui-smoke.mjs` | The 10/05 screen: one flat list, one-screen fit, `.sv-cand` with shares on the main row, pin / auto-off / 최신으로. Reversed by the sectioned list (지금 · 예정 · 기록), 10/13 scrolling, D49 / UI 10/26 ③ (rank + type only), and the removed pin. Its demo-cache part already moved to `demo-cache-smoke.mjs`. | `demo-cache-smoke`, `static-publish-smoke` (list, dates, live/final), `display-state-smoke`, `ui-1026-capture`, `compact-rev3-fit` |

To revive one, rewrite it against the current screen in `tests/` and do not lower its thresholds.
