# Disposition — Opus review of 7021008 (model card conditioning sentences)

`same_model_reviewed; Codex_review_pending`. Date 2026-10-04 15:2x KST, v3 coordinator cycle-016 (T44b).

- Review: `OPUS-review-model-card-conditioning-7021008-2026-10-04.md`, reviewer commit `f07a6ed` (helper 42548, exit 0, opus-5-5 only, no permission denials, 405.7 s).
- Branch `codex/model-card-conditioning-1004` fast-forwarded 7021008 → f07a6ed (no other change).

## Findings
- Blocking: none.
- F1 (card said only "위치" is averaged; the delivery vector is 8 channels) — ACCEPTED. Checked myself: `git show b38bea6:experiments/pitchmdp/pitchmdp/sequence_data.py` lines 21–24 `PHYSICAL_COLUMNS`/`PHYSICAL_CHANNELS` = speed, spin rate, spin axis sin/cos, pfx_x/z, plate_x/z.
- F2 (smoke passed when pins were removed) — ACCEPTED (commit equality, exact file list, anchor keys, non-empty anchors). Reviewer mutation table M3–M6 now exit 1; not re-run by me.
- F3 (400 draws / actual-type pinned only via policy path) — ACCEPTED (4 new pins; JSON-only Q&A fact).
- N1 (P11 "서비스 모델" = 0.7468 × mean(5 nets) + 0.2532 × frequency) — DEFERRED to its own unit (T45): new card sentence needs a pinned JSON number with smoke equality to `results/EXP-P10-001.json` and its own review. Fact kept in JSON `facts` (Q&A only).
- N2 ("조금" / count-conditioned BC), N3 ("최대 12구") — NOT APPLIED now; wording judged acceptable by reviewer; candidates for T45 together with N1.
- N4 (served identity a6dffaea… vs evaluated c13cc989…, D125 recertification), N5 (league-fallback share 미측정) — Q&A notes only; card makes no claim on either.

## Checks after disposition (worktree `apps/observer/web`, node_modules symlink to showcase)
- `npx tsc -b` → 0
- `npx vite build --outDir dist-t44b --emptyOutDir` → 0; js `index-DgRtmNyT.js` f2b6b497…, css `index-CbbIe769.css` 02cbd371… (same names as reviewer build)
- `DIST=dist-t44b MODEL_EVIDENCE_REPO=/Users/song/Projects/pitcheezy node tests/model-evidence-smoke.mjs` → 0 (`source identity: ok`, 1440/390/360)
- `DIST=dist-t44b CV_M3_RECEIPT=…/results/checks/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → 0 (`receipt identity: ok`)
- `showcase-smoke.mjs` with temporary `dist` copy (deleted after) → 0
