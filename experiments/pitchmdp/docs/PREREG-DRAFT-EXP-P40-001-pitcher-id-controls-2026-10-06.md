# PREREG DRAFT — EXP-P40-001 pitcher-ID controls (written 2026-10-06 00:45 KST, before any P40 output)

Status: DRAFT, author_only (Opus5.5 GPU lead, job `pitcheezy-pm-jobs/20261006-four-hour/gpu-research`). Branch
`codex/p40-pitcher-id-controls-1006` from P38 head `ba0a8f1`. Exploratory and post-selection: designed after reading
the P38 score. No promotion, verdict, policy or service claim. No 2026 data.

## 1. Why
P38 (seeds 44-48) showed candidate − reference ensemble ΔNLL on mlb_dev all −0.00093 [−0.00133, −0.00053] and on dev
−0.00152 [−0.00334, +0.00025], with UNK rows near zero or slightly worse (mlb_dev UNK ΔBrier +0.00061 [+0.00006,
+0.00116]). The candidate adds pitcher identity AND 8,392 id-path parameters together, so P38 cannot say which one
produced the difference. DEV / mlb_dev are repeatedly exposed.

## 2. Questions (fixed before any P40 output)
- Q1 (primary): with the id-path network, parameter count and compute held fixed, does the TRUE pitcher index lower
  calibrated NLL / Brier vs a row-shuffled index? Contrast `pitcher_id_vs_row_shuffle` (candidate − shuffle).
- Q2 (replication): does candidate − reference repeat the P38 sign on fresh seeds 60-64?
- Q3 (capacity control): does the information-free id path alone move NLL vs the reference? (`row_shuffle_vs_periodic_residual_counts`)
- Q4 (secondary): does ID dropout 0.2 change UNK / cold-start rows and in-vocab rows vs the candidate? (`id_dropout_vs_pitcher_id`)

## 3. Arms (config `configs/DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json`, sha 3dba5445…)
1. `periodic_residual_counts`: P36/P38 reference, verbatim.
2. `pitcher_id_periodic_residual_counts`: P38 candidate, verbatim (TRAIN vocabulary ≥100 rows, UNK 0, D_ID 8, z += P·T[idx]).
3. `pitcher_id_row_shuffle_periodic_residual_counts`: arm 2's network. Before the vocabulary lookup, each split's
   `pitcher` column is permuted across rows by `np.random.default_rng([40001, split position])`. That seed is fixed and
   independent of labels, features and the model seed. Index frequencies and UNK share stay the same. The row→pitcher
   association is gone. Parameters and compute are exactly those of arm 2.
4. `pitcher_id_dropout_periodic_residual_counts`: arm 2 + ID dropout. In training mode each row's index becomes UNK with
   probability 0.2. The mask comes from a private device generator seeded id_seed + 4040, so the global RNG is untouched.
   Eval, temperature calibration and prediction use the true index. The rate 0.2 is fixed, not tuned. It follows the
   input-dropout idea for cold start in DropoutNet (Volkovs, Yu, Poutanen, NeurIPS 2017). The row-shuffle control is
   the permutation-importance idea (Breiman 2001; Fisher, Rudin, Dominici, JMLR 2019), applied at training time.

Seeds: exactly 60-64 (fresh; 0-51 used before, 52-59 left for the MPS sibling EXP-P41). Fits: 4 × 5 = 20, epoch cap
60, patience 5, cuda_direct_v1 budget, export 49097109, GPU3 UUID c334d905 only, one fit at a time, common lock.

## 4. Splits and metrics
TRAIN ≤ 2025-04-30 for fitting and vocabulary. Earlystop 05-01..15 for model selection. Temperature 05-16..31 for
calibration. Scored on dev (12,334 rows) and mlb_dev (311,721 rows). Metrics: calibrated NLL, multiclass Brier,
accuracy and top-label ECE (10 bins) per arm, per seed and seed-mean ensemble, on all rows and on the TRAIN-vocabulary
in-vocab / UNK subsets. The subsets use the TRUE pitcher for every arm, including the shuffle arm. Contrasts are paired
game-bootstrap differences (matrix_metrics.paired_game_comparison, the existing draws and seed), per seed and
ensemble. No decision, ranking or winner field.

## 5. Reading rules (fixed now; descriptive only)
- Q1: an ensemble mlb_dev-all ΔNLL CI entirely below 0 with all 5 seeds negative is read as "consistent with identity
  information, beyond the id path". A CI crossing 0 is read as "not separated from an information-free id path at this
  size". dev (12k rows) is reported but expected to be wide.
- Q2 and Q3 are read together. If shuffle − reference is about 0 and candidate − reference is negative, the P38
  difference is not explained by capacity alone. If shuffle − reference is negative as well, then extra parameters or
  regularisation account for part of it.
- Q4: UNK and in-vocab subsets are reported side by side. A lower UNK ΔNLL alone is not called a cold-start fix.
- Limits: the shuffle control matches the id path, not a generic wider backbone. Bootstrap conditions on fixed
  predictions. Repeated DEV. Fresh model RNG seeds are not new data. Any follow-up needs a new registration.

## 6. Cost and clock (projection fixed from P38 actuals)
P38 task walls were 164-245 s (max 244.7 s). Bound per task 1.25 × 244.7 = 305.9 s, so the family projection is 6,117.5 s.
Expected about 20 × 204 s ≈ 68 min plus scoring. Clock on 2026-10-06: launch_by 02:05, no_new_fit_after 03:50,
kill_own_after 04:05, aggregate_by 04:25 (checkpoint 04:28:47). The family scores only at 20/20; otherwise it is
reported `incomplete`.

## 7. Gates
1. Distinct review (MPS lead, read-only) and Root acceptance of this packet.
2. GPU3 TRAIN-only smoke: 20 tasks, 8,192 TRAIN rows, smoke epochs. `p40_launch.py smoke-check` must be VALID (20/20,
   {GPU3}, one vocabulary record, mode records, no DEV read). Smoke prediction archives are written by the runner and
   never read.
3. Registration (`p40_launch.py register`, real clock) → `configs/EXP-P40-001.json`. Only status, decision, execution
   and registration differ from the DRAFT. Then the pipeline is launched from the template.

## 8. Addendum at registration (2026-10-06 00:52 KST, before any DEV read): reviewer wording notes
These come from the distinct review (MPS lead, GPU-REVIEW.md 00:48:37, ACCEPT efa14c8) and are adopted as reading rules:
- N1: the shuffle arm learns id offsets on fixed random TRAIN row groups and applies them to differently permuted rows at
  eval, so it can inject small learned noise. If shuffle − reference > 0, candidate − shuffle is read as an UPPER bound on
  identity value, and candidate − reference is the conservative number. Both are always reported side by side.
- N2: same export and exposed DEV as P38. Fresh seeds = RNG replication only, not new data.
- N3: runner, scorer and model are edited in place. Any P38 re-score must use the P38 tree (ba0a8f1 / 5b3335e).
