# P47 preparation protocol — NOT REGISTERED, NOT RUN

Prepared 2026-10-06 by Astra, isolated from executor base `88860a40c3629b483db28d3eb689f4a73492ce13`.
Motivation: the 06:28 P47 proposal and P40/P43 handoff. This is adaptive exploration after repeated DEV exposure,
not an independent confirmation. This packet authorizes no GPU probe, smoke, timing, real TRAIN fit, real DEV
evaluation, 2026/OPE read, registration, service change or deployment. CPU synthetic fixtures are the only test data.
G15 mix remains the research baseline; G0 service and D131/D155 decisions remain unchanged.

## Question and four arms

Can joint pitcher/batter identity dropout reduce the unseen-batter harm observed in P43 while retaining known-player
gains and net value against the no-ID reference? Does keeping both UNK vectors at zero further help?

All arms refit at seeds **80, 81, 82, 83**, 16 fits total; no cross-family reuse, extension, pooling or best-seed choice.

1. `periodic_residual_counts`: unchanged reference entry.
2. `pitcher_batter_id_periodic_residual_counts`: unchanged P43 two-path entry.
3. `pitcher_batter_id_dropout_periodic_residual_counts`: arm 2 plus 0.2 training-only dropout, independently per
   column/row from private generators on the index device. Seeds are the corresponding ID projection seed + 4040
   (P40 mechanism); pitcher projection seed = model seed + 7919, batter adds 1009. Global RNG is untouched.
4. `pitcher_batter_id_dropout_zero_unk_periodic_residual_counts`: arm 3 plus permanent zero UNK row 0 in both tables.
   Row 0 starts zero, its effective embedding and gradient are zero, and AdamW preserves zero. Archive loading and
   fit recording reject a nonzero frozen row. This gate remains active at evaluation, temperature calibration and
   prediction. **Only dropout is training-only. Evaluation uses true indices and a zero vector for UNK in arm 4.**

Arms 2–4 have equal allocated parameters; arm 4 has 16 fewer free embedding scalars (two rows x eight dimensions).
Runtime is not matched. Arm 3 versus arm 2 changes both pitcher and batter dropout together, so the batter-unknown
effect cannot be attributed to batter dropout alone. There is no information-free capacity control in this family.
No nonzero dropout/freezing fields are added to the original P43 model/archive path.

## Fixed contrasts and strata

The primary estimands are two **contrast/split/subset triples**, not entire contrast tables:

- A: arm 3 minus arm 2, **mlb_dev, batter_unknown** (cold-start repair).
- B: arm 3 minus arm 1, **mlb_dev, all** (net predictive value).

Secondary: A on batter_known (cost to known-batter gains); arm 4 minus 3; arm 2 minus 1 (fresh-seed replication);
arm 4 minus 1. Report all five unique arm pairs on dev and mlb_dev, every seed and the seed-mean ensemble, for all
rows, both batter margins and all four pitcher-known/unknown x batter-known/unknown cells. The earlier proposal's
“six contrasts” counts A on the known-batter margin separately; it is **five unique arm pairs and six highlighted
estimands**, with further strata descriptive. No missing/empty cell is silently dropped (zero rows, null comparison).

Known = the TRUE ID belongs to its own TRAIN-only vocabulary (>=100 TRAIN rows). “Unknown” includes both rare TRAIN
players below this threshold and players unseen in TRAIN; it is not exclusively debut players. Never use dropped
or shuffled training IDs to define a scored stratum. Both vocabularies are re-derived and checked by the scorer.
No subgroup threshold or selection is chosen from DEV performance. Prior expected mlb_dev batter margins were
27,658 unknown / 284,063 known; actual counts are reported later, not fabricated in preparation.

Primary metric calibrated NLL; report multiclass Brier, accuracy, top-label ECE10 for every arm/seed/ensemble/stratum.
Keep `matrix_metrics.paired_game_comparison`, 10,000 whole-game bootstrap draws, seed 20260924, unchanged. A descriptive
“consistent on exposed DEV” reading requires its primary 95% interval to exclude zero AND >=3/4 seeds to agree.
No automated verdict field, model selection or policy promotion. Both primaries are reported even if one fails;
this is not a familywise significance test. No multiplicity correction. Fixed-prediction bootstrap omits refit,
calibration, seed-distribution and adaptive-selection uncertainty. New RNG seeds are not new evaluation data.

## Frozen data/training contracts

P43 export `49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578`, P43 batter sidecar
`ad2eacc64d46985a57527167486d7762a435f02d51fc5ed86a9e2e2644c160a4`, original splits and calibration copied verbatim.
TRAIN ends <=2025-04-30; earlystop May 1–15; temperature May 16–31. Vocabulary cutoff 100, ID width 8, cap 60,
patience 5 and all optimizer/batching constants unchanged. No extra raw-data fetch/export. No 2026/OPE reads.
Same repeatedly exposed <=2025 dev/mlb_dev rows as P24–P46. DRAFT carries no queue date or launch clock.

Identifier/seed reservation is based on existing metadata: P43 70–72, extension 73–74 reserved, P44 75–79 reserved,
earlier 0–69 used/reserved. Fresh check of the coordination ledger is required at registration; preparation does
not acquire a global ID claim. P41/P45/P46 reuse 49–53 within their MPS protocols, not the proposed 80–83 block.

## Future resource and launch plan — not executed here

Only GPU UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`, one fit at a time, shared exclusive lock
`/home/sds733/pitcheezy-autonomy/gpu-locks`. No all-GPU query, other GPU, CPU fit fallback or MPS execution.
Read-only preflight proposed: date, UUID-selected `nvidia-smi --id=<allowed UUID> --query-gpu=uuid,utilization.gpu,memory.used
--format=csv,noheader`, UUID-selected compute-app query, own queue/process and lock-holder metadata. No changes in preflight.

Existing P43/P38 timing-based fit projection: **16 x 1.25 x 244.7 = 4,894 seconds (81.57 minutes)**. This corrects the
proposal's “~75 min total”: 75 minutes cannot include an 81.57-minute fit projection. Expectation 50–60 minutes from
typical P43 fits is descriptive only. Preflight/smoke/timing are additional, unmeasured costs. Future queue window must
allow fit projection plus 15-minute stop grace and 20-minute scorer/report reserve, and measured preflight time.
Keep inherited task/family budgets and max attempts; deadline and resource failure stop, no partial-family scoring.

Sequence, requiring a distinct review of fixed code and Root acceptance before any actual resource work:

1. Check global identifier/seeds, active jobs and shared lock; select a new dated window, review its exact diff.
2. Pin reviewed source/config/sidecar/export hashes and stage a new read-only checkout.
3. TRAIN/earlystop-only wiring smoke for all 16 tasks; validate modes, private streams, vocab hashes, UUID and no DEV
   use. **Existing `run_cuda_direct --smoke` writes tiny DEV prediction archives, so it must NOT be used as-is for this
   stronger TRAIN-only gate.** A reviewed preparation-only probe or strict train-only smoke mode remains required.
4. Full-TRAIN two-epoch arm-4 timing: expected vocab 920 pitchers/706 batters from P43; both UNK rows must remain exactly
   zero after optimization; record task/train seconds, memory and source/array identity, no DEV output or score.
5. Verify budget projection fits the new real clock. Independent evidence review / Root acceptance, then dated
   registration **before any real DEV score or main fit**, durable serial pipeline with scoring after 16/16 done.

`p47_plan.py` is pure planning metadata and accepts only EXP-P47-001. It has no registration or launch entrypoint.
The existing P43 launch helper/template still reject the P47 design; making them executable is a later reviewed
step, intentionally outside preparation. CPU test fixture configs marked REGISTERED are synthetic in temporary
directories only; they are not experiment registration and do not alter this committed DRAFT.

## Required synthetic checks and review status

CPU fixtures: per-column independent private RNG, train-only dropout and eval equality, zero UNK rows after AdamW,
archive roundtrip/mode mismatch/corrupt UNK refusal, unmodified P43 routing and archive keys, 16-task runner/scorer
path on generated rows with all 2x2 strata and exact primary labels, invalid registration/seed/config/mode refusal.
Mutation probes: dropout accidentally at eval, missing zero gate and wrong-column stratum must fail focused checks.
Author preparation only until a distinct Astra reviewer returns a verdict on a fixed commit. Tests establish
apparatus behavior on synthetic data, not GPU compatibility, learning quality or any scientific conclusion.
