# Disposition — Opus review of cc919b4 (phone-width reveal reachability)

Label: `same_model_reviewed (review bcfcb09); Codex_review_pending`. This disposition itself is not separately reviewed.

- F1 accepted: reviewer test `04f4884` cherry-picked as `7aa67d6`; review doc as `258aa45`.
- F2 applied, option (a): at ≤800 px `.sc-compare iframe` height reserve 64 → 84 px, so a two-line comparison header keeps the
  reveal bar inside the viewport (reviewer measured 670 > 664 at 390x664 with a long name). Real 39-scene labels are checked at
  390x664 on 8771 after deploy.
- F3 applied as suggested: the clip-end scroll runs only when the comparison is not already in view
  (`top > 0 || bottom > innerHeight`), so a viewer who scrolled past (e.g. to the evidence cards) is not pulled back. New smoke
  step (phone only): scroll to page bottom during a clip, precondition comparison top < 0, clip end → scrollY unchanged.
  Mutant (guard removed) → smoke exit 1 at 390x844 and 390x664.
- F4 applied: up button `aria-label="영상으로 올라가기"`; arrows in `<span aria-hidden="true">`; smoke locators use exact names.
- F5 no change (focus intentionally not moved into the iframe; Space there would reveal).
- F6 not applied (cosmetic).

## Checks (Mac, OMP/UV 1, nice 10, perl alarm; node_modules temporary symlink to main checkout, unlinked)
- `tsc -b --noEmit` 0; `vite build --outDir pm-jobs/20261004-end-to-end/t58c-dist` 0 (js index-D0KUefQy, css index-BGk569ec).
- `showcase-phone-reveal-smoke.mjs` on t58c-dist → exit 0 (log `t58c-phone-smoke.log`, screenshots `t58c-shots/`).
- F3 mutant build `t58c-mut-f3-dist` → exit 1 (log `t58c-mut-f3.log`); source restored, diff = intended changes only.
- Regressions on t58c-dist (log `t58c-regressions.log`): showcase-smoke (temp dist symlink, unlinked) 0, watch-spoiler 0,
  pre-reason-neutral 0, watch-setup 0, cv-evidence (CV_M3_RECEIPT main receipt) 0, model-evidence (MODEL_EVIDENCE_REPO main) 0.
- Not checked: real iPhone / Safari (fullscreen, toolbar collapse, safe area — see review "iOS Safari: untested").
