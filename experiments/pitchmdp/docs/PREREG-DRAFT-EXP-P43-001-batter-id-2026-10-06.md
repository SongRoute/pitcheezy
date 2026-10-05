# PREREG DRAFT — EXP-P43-001 TRAIN-only batter-ID embedding (written 2026-10-06 01:26 KST, before any P43 output)

Status: DRAFT, author_only (Opus 5.5 next-research worker, job `pitcheezy-pm-jobs/20261006-four-hour/next-research`).
Branch `codex/p43-batter-id-1006` from the registered P40 source `3ec8df1` (P40 / P41 / P42 sources untouched).
Design accepted in principle by Root 01:08 KST (ROOT-NOTES.md "P43 decision"); separate distinct review required.
Exploratory and post-selection: designed after reading P25 (batter summary matters) and P38 (pitcher-ID signal).
No promotion, verdict, policy or service claim. No 2026 data. No P40 / P41 / P42 output read.

## 1. Why
- Target (from code, not the UI): per-pitch 10-class outcome (ball, strike, foul, out, 1B, 2B, 3B, HR, HBP, DP),
  calibrated NLL / multiclass Brier, pre-pitch direct head `periodic_residual_counts`.
- The reference already reads a 17-d batter summary (6 style rates, 6 reliabilities, 5 soft memberships). P25:
  removing all of it costs Cpanel +0.00586 / MLB +0.00752 ΔNLL (3/3 seeds). So batter information matters to this
  model, but batter **identity** has never been given to any direct head. The export has no `batter` column.
- P38 showed a small pitcher-ID gain (mlb_dev ensemble −0.00093 [−0.00133, −0.00053]); P40 tests whether that is
  information or capacity. P43 asks the same question for the batter, with its own matched control, and whether
  batter identity adds to pitcher identity.

## 2. Data: batter sidecar (new input, measured, not imagined)
- `pitchmdp/batter_sidecar.py` joins each export row by its own `keys` (game_pk, at_bat_number, pitch_number) to raw
  statcast_2023/2024/2025.parquet, reading ONLY id columns (keys, batter, pitcher). Files must be named
  statcast_YYYY.parquet with YYYY ≤ 2025; any `holdout` path is refused before opening.
- Integrity: every export row matched exactly once; raw `pitcher` == export `pitcher` on every row (0 mismatches);
  per split the sidecar stores the export rows_sha256 and the batter array sha; `sidecar.json` sha is pinned in the
  config (`batter_sidecar.sidecar_sha256` = ad2eacc6…). Runner (verify_export, also called by the queue preflight)
  and scorer verify it.
- Measured on the Mac mini 01:06:42 KST (0.74 s, local verified export copy): TRAIN 1,252,824 rows / 806 batters;
  TRAIN vocabulary (≥100 TRAIN rows) 706 batters, TRAIN UNK rows 4,425 (0.35%); UNK share earlystop 2.1%,
  temperature 4.5%, blend 5.3%, dev 8.0%, mlb_dev 8.9% (ids only; no label read).
- Server: newsrv has the export but no raw statcast (checked 01:27 KST, read-only `find` depth 6). The GPU lead copies
  the Mac sidecar directory (~10.8 MB, 7 files) to a NEW sibling `$B/batter-sidecar-49097109`, `chmod -R a-w`, and the
  runner's verify_export checks sidecar.json sha, every split's rows_sha256 against the SERVER export manifest and
  every batter array sha before any fit (also in the queue preflight).

## 3. Arms (config `configs/DRAFT-CUDA-BATTER-ID-20261006.json`, sha bc94ae86…; all at the SAME seeds)
1. `periodic_residual_counts` — reference, P36/P38 entry verbatim.
2. `pitcher_id_periodic_residual_counts` — P38 candidate entry verbatim (pitcher id only).
3. `batter_id_periodic_residual_counts` — the P38 id path on the batter column: TRAIN-only vocabulary (≥100 TRAIN
   rows, UNK 0), zero-init table [n+1, 8], projection 128×8 from a private generator (seed + 7919 + 1009).
4. `pitcher_batter_id_periodic_residual_counts` — both id paths, additive (pitcher path = P38 verbatim).
5. `batter_id_row_shuffle_periodic_residual_counts` — arm 3's network; the split's batter column permuted across rows
   by `default_rng([43001, split position])` before lookup (label-free, model-seed-free). Frequencies / UNK share kept,
   row→batter association removed; parameters and compute identical to arm 3.
At a seed all id tables start at zero, so every arm starts from exactly the reference logits (tested).
Seeds **70, 71, 72** → 15 fits. Fresh: 0–69 used or reserved (P40 60–64, P42 DRAFT 65–69, P41 MPS 49–53).
Optional clock-only extension **EXP-P43-002**: the same five arms on seeds 73, 74 (10 fits), config
`DRAFT-CUDA-BATTER-ID-EXT-20261006.json` (sha 389c81b3…), only after EXP-P43-001 is complete and if the window
fits; scored as its own family; never pooled into the 70–72 bootstrap tonight.

## 4. Contrasts (fixed now; paired game bootstrap 10,000 draws, seed 20260924, per seed and seed-mean ensemble)
| name | candidate − reference | role |
|---|---|---|
| batter_id_vs_row_shuffle | 3 − 5 | **co-primary**: batter identity information (parameter / compute matched) |
| pitcher_batter_id_vs_pitcher_id | 4 − 2 | **co-primary**: adding batter identity to pitcher identity |
| pitcher_batter_id_vs_batter_id | 4 − 3 | secondary: adding PITCHER identity to batter identity |
| batter_id_vs_periodic_residual_counts | 3 − 1 | secondary |
| pitcher_id_vs_periodic_residual_counts | 2 − 1 | secondary (P38 sign at new seeds, same-seed pairing inside P43 only) |
| batter_row_shuffle_vs_periodic_residual_counts | 5 − 1 | capacity control |
Splits: dev (12,334 rows, Cpanel) and mlb_dev (311,721). Subsets: all / in-vocab / UNK by the TRUE batter's TRAIN
vocabulary membership, identical for every arm. Metrics: calibrated NLL, Brier, accuracy, top-label ECE (10 bins).

## 5. Reading rules (descriptive; fixed before any output)
- Two co-primaries, exploratory, **no multiplicity adjustment**; both are reported whatever their sign.
- Co-primary 1: ensemble mlb_dev-all ΔNLL CI entirely < 0 with 3/3 seeds negative → "consistent with batter identity
  information beyond the id path"; CI crossing 0 → "not separated from an information-free id path at this size".
- Co-primary 2: same wording for "adds to pitcher identity". Not parameter-matched (adds the batter path); read with
  co-primary 1 and the capacity control.
- 4 − 3 measures adding pitcher to batter, not the reverse. 2 − 1 is not a replication of P38's evidence weight (other
  seeds, 3 not 5) and is never paired with P40 / P42 seeds.
- UNK subset is expected to be harder (UNK row trained on 4,425 rare-batter TRAIN rows only); a UNK gain or loss alone
  is not called a cold-start result. dev (12k rows) is reported but expected to be wide.
- Limits: repeated DEV (exposed 2025 rows), fresh RNG seeds are not new data, bootstrap conditions on fixed
  predictions, the shuffle control matches the id path not a wider backbone. Any follow-up needs a new registration.

## 6. Cost and clock
Bound per task 1.25 × P38 max task wall 244.7 s = 305.9 s; projection 15 × 305.9 = 4,588.1 s (extension 3,058.75 s).
P40 actuals observed by the GPU lead 01:17 KST: about 3.5–4 min per fit, so 15 fits ≈ 53–60 min plus scoring
(estimate, not a measurement). The DRAFT has NO window: when GPU3 is free the GPU lead writes a dated copy with
`make_p43_config.py --clock YYYY-MM-DD HH:MM --out <new>` (only the five clock fields change; `p43_launch.py
dated-check`), smokes it, registers it. Scoring only after all tasks; else `incomplete`.

## 7. Gates
1. Distinct review (GPU or MPS lead, read-only) and Root acceptance of this packet.
2. P40 terminal (and P42 not running); GPU3 free; common lock.
3. Server sidecar copied as a new read-only sibling of the export, sidecar.json sha == ad2eacc6…, verify passes.
4. GPU3 TRAIN-only smoke of all 15 tasks: `p43_launch.py smoke-check` VALID (bound {GPU3}, one TRAIN vocabulary per
   column, pitcher arm P38 record, shuffle mode recorded, no DEV read).
5. Registration (`p43_launch.py register`, real clock) → `configs/EXP-P43-001.json`; only status / decision /
   execution / registration differ from the dated DRAFT. Then the pipeline from `run-p43-pipeline.TEMPLATE.py`.
No two-slot use for P43 unless a fresh distinct review + TRAIN-only bench on this exact source passes (Root rule).

## 8. Literature (primary sources; what transfers)
- Alcorn, "(batter|pitcher)2vec: Statistic-Free Talent Modeling With Neural Player Embeddings", MIT Sloan SSAC 2018,
  https://www.sloansportsconference.com/research-papers/batter-pitcher-2vec-statistic-free-talent-modeling-with-neural-player-embeddings
  , code https://github.com/airalcorn2/batter-pitcher-2vec . Learns batter and pitcher embeddings by predicting
  at-bat outcomes from identities alone (2013–2016 MLB); reports better prediction of unseen matchups than simpler
  baselines (checked at abstract level only). Transfers: identity embeddings carry outcome-relevant information.
  Does not transfer: their model has no other batter features; ours already has a strong 17-d summary, so the
  expected effect is a residual (P38-sized, ~1e-3), not their headline gain.
- Guo & Berkhahn, "Entity Embeddings of Categorical Variables", 2016, https://arxiv.org/abs/1604.06737 — learned
  embeddings for high-cardinality categoricals inside an MLP (the mechanism used here and in P38).
- Fisher, Rudin & Dominici, "All Models are Wrong, but Many are Useful", JMLR 20(177), 2019,
  https://arxiv.org/abs/1801.01489 — model reliance via permuting a variable; we apply the permutation at training
  time as an information-free control (assumption: permutation keeps marginal frequency / UNK share and breaks only
  the row association).
- Gorishniy, Rubachev & Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS 2022,
  https://arxiv.org/abs/2203.05556 — the periodic embedding backbone of the reference arm (unchanged here).

## 9. Not in this study
Pitcher×batter interaction terms (factorization-machine style), batter ID dropout / partial pooling, batter previous-PA
pitch history (needs a new export), catcher / umpire / park (not in the export). An unseen ≤2025 confirmation set is
proposed separately (next-research/OPTIONS.md, metadata only; May-2025 game-disjoint 156 games are exposure-UNKNOWN,
not yet unseen).
