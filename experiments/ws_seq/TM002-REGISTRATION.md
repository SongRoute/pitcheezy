# EXP-WS-TM-002 — preparation registration

Status: PREPARATION ONLY, execution disabled pending independent review and a separate Root work order. Author: research_astra native worker; requested model gpt-6-astra, actual response model unverified (no response metadata). Base commit `35b3001debd95d10c070cc2dd02a68294eada17e`. No raw dataset, checkpoint/model payload, EVAL predictions, GPU status or GPU job was read/run in this preparation unit.

## Question and one change

Compare Transformer T to 2-layer unidirectional LSTM R on the existing WS N1 task. T is the unchanged `train_seq.build_model(...,'N1')`. R builds exactly that N1 first and replaces only `encoder`; same-seed shared trunk/embeddings/no-context/heads are byte-identical. Encoder parameters and parameter counts need not match. CPU RNG fork restores caller state; future training resets training RNG equally. Different dropout draw consumption after construction is inherent to different architectures and is not a promise of identical dropout masks.

Inputs, heads, loss, hyperparameters and calibration inherit the hash-bound `configs/EXP-WS-SEQ-001.json`: 199 current features, 64-wide tokens, at most 32 strictly earlier pitches in the same pitcher/game, newest-to-oldest order, same-batter and relative-position embeddings, trunk [256,128], dropout 0.1, existing masked mean readout and learned no-context. Packed LSTM excludes suffix padding from state updates. The original N1 dummy token for empty context is discarded by the unchanged no-context readout. No SEQ-002 physics/zone, longer history, batter stream, focal loss, SEL calibrator, or teammate feature/checkpoint is introduced.

`train_seq.py`, `prepare_inputs.py`, `evaluate_seq.py` and the original configs remain unchanged. Past LSTM implementations are not substituted: their feature/readout contracts differ. This is a narrow encoder experiment, not teammate-model replication.

## Data and time contract

Version d20260930-h2026f. FIT 2026-03-25..08-02 (493254 rows); SELECT 08-03..08-23 (82908); CAL 08-24..09-06 (56275); EVAL 09-07..09-27 (80584 / expected275 games). Counts are inherited registrations, not remeasured here. 2026 EVAL is repeatedly **EXPOSED DEVELOPMENT**, never untouched TEST. B1/B2, paper replication, OPE and sealed artifacts remain out of scope.

Reuse the accepted SEQ-001 input package only after a reviewed manifest/ledger grant. The runner verifies the existing package's original source/config bindings plus the new grant's exact manifest hash and role counts. The original input builder is not rerun in this work. Keys, labels, eligibility, roles and order must remain identical across arms; context index runtime checks require strictly earlier keys, same game/pitcher, decreasing time and suffix padding. Existing long-history features remain prequential strictly-prior-date features; earlier SELECT/CAL/EVAL results may become subsequent query history but never refit the encoder/model/temperature. Actual feed arrival/as-of timing remains UNKNOWN and is not proved by date/key ordering.

The exposure ledger must bind allowed columns, hashes/version, collection/as-of, key/duplicate policy, prior roles/manual exposure, and unknown provenance explicitly. Its exact file hash is required by each grant. Existing manifest and data payload are not opened during preparation.

## Training and finite schedule

Two arms × seeds [0,1,2], exactly six fits and eighteen scalar head CAL optimizations. No performance-driven stop/expand or replacement seed. Stage1 = T-s0 and R-s0; stage2 = four remaining fits only after independent integrity gate for BOTH seed0 supervised receipts. Gate records `eval_scores_opened:false`, exact source+manifest, receipt hashes and completion. Seed0 EVAL scores must remain unopened; all six must finish before ensemble comparison. Root owns one grant per logical fit and the six-fit attempt ledger; this single-fit runner is not an autonomous queue or concurrency scheduler.

AdamW lr0.001, wd0.0001, batch2048, max30epochs, patience5, fp32. Existing parent CE + masked conditional whiff/GB BCE equally weighted. Earliest strictly minimal SELECT uncalibrated seven-leaf NLL selects epoch. Each arm/seed then receives its own CAL scalar temperatures [0.05,20], bounded optimum required; temperature boundary failure is STOP. Ensemble = arithmetic mean of calibrated seven-leaf probabilities, not logits. Existing models/checkpoints are **not reused** in this initial implementation: T is freshly fit for all seeds. Source/config binding reuse is not checkpoint reuse.

Old SEQ-001 `train_arm` must not be called as TM-002: it has a SELECT performance seed-stop and soft epoch wall cap. The new `train_tm002.py` uses existing helpers but a separate small loop with no performance gate, finite-gradient checks, restored SELECT check and no EVAL metric calculation. It writes EVAL probabilities only as unopened artifacts. SELECT/CAL stage readouts are allowed by their roles; writing EVAL predictions is not authorization to inspect their scores.

## Metrics fixed before result access

Primary: R ensemble minus T ensemble pitch-weighted EVAL seven-leaf NLL, natural log and clip1e-15. Auxiliary: sum-over-7-classes Brier; top-label ECE10 equal-width bins min(floor(10confidence),9), all-leaf reliability, classwise ECE, narrow-whiff/GB reliability. Report matched seed pairs separately, not as independent games.

Use existing `evaluate_seq` pure metric helpers, whole-game paired bootstrap 10000 draws seed20261211, percentile95%, same draws for all pairs; aggregate sampled loss sums/sample pitch counts. CI is conditional on predictions, not total training uncertainty. Existing CLI has N0/N1/tree requirements and **must not be used unchanged for T/R**. A six-receipt evaluation adapter and its independent review are a remaining pre-EVAL work item; no EVAL score access is authorized until that adapter binds all six supervised artifacts and common key/label/manifest identity. This does not block separately authorized finite fit, whose outputs remain unopened.

Report pitcher support S0=0/S1=1..199/S2=200..999/S3>=1000 FIT rows and batter B0=0/B1>0; row/game/player/missing/fallback/unknown-pitch-type denominators; empty cells unmeasured. No subgroup winner selection. Follow-up label only: NLL CIupper<0 and Brier CIupper<=0.0005 => FOLLOW_UP_WORTHWHILE; NLL CIlower>0 => EVIDENCE_AGAINST; otherwise INCONCLUSIVE. ECE degradation fully positive CI is a separate concern. Noninferiority NLL margin0.0010 is auxiliary, not improvement. No family-level, policy-value, real-time latency, scientific-promotion or deployment claim.

## Resource/stop contract

Budgets are **design limits, not measured runtime or feasibility**: preparation1800s, each fit including input load/CAL3600s, 6×3600=21600s total fit work, evaluation900s, total24300s. One GPU/fit; two fits in stage1, at most four concurrent in stage2. A5000 five-card authorization does not justify extra combinations. Before each future launch, `/Users/song/.local/bin/gpu st`; use only `/Users/song/.local/bin/gpu run`, no bare shell alias. Check UUID/owner/process; acquire the independently selected common per-device exclusive lock. No other process termination.

Linux CUDA only, no CPU/MPS fallback. GPU PyTorch caching-allocator reserved limit16GiB through `set_per_process_memory_fraction`; this is **not a cap on driver/context/non-PyTorch VRAM**. Separate current occupancy/headroom and chosen UUID are required in the launch review. Host per-worker RSS8GiB sampled every0.25s by external supervisor; wall3600s includes worker startup and fitting, with owned-child TERM and5s grace then KILL on breach. Self-watchdog is an additional0.5s backstop. Sampling/grace permit transient overshoot; these are enforced stop thresholds, not mathematical continuous ceilings. Supervisor failure/worker nonzero/partial outputs are INCOMPLETE. Only receipt+SUPERVISOR COMPLETED bound by receipt hash qualifies. GPU allocator OOM, missing package/hash, mismatched rows/keys/context, nonfinite loss/gradient/margin, invalid temperature, restored SELECT mismatch, watchdog read failure/time/RSS, or preexisting output => STOP; no automatic retry.

Run metadata: data version/role, immutable commit from clean gpu job.env, source/config/input/grant hashes, seed/arm/attempt, GPU UUID, Python/torch/numpy/scipy/CUDA/cuDNN, parameters, rows, best epoch, context checks, elapsed and peak allocator memory, output hashes; supervisor reports worker peak RSS and wall. CUDA bitwise reproducibility is not guaranteed.

Mac recorders/collection and PID94628/94642/99202, 8772 demo and profile refresh remain untouched. Training is remote only. Preparation CPU tests are single-thread small synthetic; no GPU probe or run occurred.

## Launch contract, not execution permission

Reviewed grant JSON requires id, execute=true, verdict=PASS, distinct_from_author=true, different named author/reviewer, sources_sha256 (from train_tm002.source_hashes()), exact arm/seed, absolute input/output paths, exposure ledger path/hash, base input manifest sha, clean commit, fresh attempt_id, GPU UUID, common_lock path, exact config budget/data_version. Seed1/2 also require seed0_gate path/hash binding both actual seed0 receipts and their supervisors. Defaults stay execution_enabled=false; only a separate reviewed explicit grant authorizes actual work. No grant is delivered by this preparation.

Future command shape, only AFTER Root work order and source/grant/resource acceptance:
`/Users/song/.local/bin/gpu run -n tm002-T-s0 -g <approved-id> -- python -m experiments.ws_seq.train_tm002 --grant <reviewed.json> --inputs <accepted-input-package> --out <new-attempt-dir> --arm T --seed 0`

Launch review must still bind data/ledger bytes, server runtime/dependencies, free host memory/disks, common lock path shared with other projects, exact GPU assignment/current status and total six-fit accounting. No performance/latency/memory outcome is measured here. Future code/config repairs require new review and fresh recorded attempt, never overwritten artifacts.
