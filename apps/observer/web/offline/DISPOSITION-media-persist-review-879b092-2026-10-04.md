# PM disposition — Opus review of same-tab `?media=local` memory (`879b092`)

Status: `same_model_reviewed; Codex_review_pending`. PM worker (Opus 5.5, v4 cycle-020), 2026-10-04 19:0x KST.
Review: `OPUS-review-media-persist-879b092-2026-10-04.md` (reviewer commits `136d8b0`, `03c6f93`, `8fb1996`, `abde4b2`).
Branch `codex/media-persist-1004` fast-forwarded 879b092 → abde4b2 (no conflict, linear).

| ID | Disposition |
|---|---|
| m1 test D weaker than claimed | Accepted; reviewer fix `136d8b0` kept. My rerun: strengthened smoke exit 0 on the new build; same smoke on served ed40b66 bytes exit 1 at `plain /showcase in the same tab stays local` (core claim, not the cosmetic href). Reviewer mutations M1/M2 not rerun by me. |
| m2 README:29 inaccurate after T66 | Accepted; `03c6f93` kept (check the on-screen note, not the address). |
| N1 back-link `?media=local` cosmetic | Keep as is (makes the mode visible in the URL). |
| N2 Duplicate Tab carries the flag | Accepted; README caveat. Untested in any browser. |
| N3 opt-in on an online deployment → 404 alert in that tab | Accepted; visible note + `?media=online` / new tab recover. |
| N4 README:14 wording for bundle v5 | Deferred to the v5 bundle cut. |
| N5 Safari / iPhone / MacBook / real file not checked | Stays on the 10-06 human checklist. |

Source under review unchanged after 879b092 (`Showcase.tsx` only; WatchAlong/Game untouched). Build js `index-C2cOnYq0.js` sha256 0a072b8f… = author's and reviewer's builds.
Evidence of my rerun: coordinator worktree `reports/t69-media-persist-deploy/` (tsc, build, 9 smoke logs, build SHA256SUMS).
