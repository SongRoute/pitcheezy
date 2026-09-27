# G0/F1 five-seed runner contract v1

2026-09-27. D72 authorizes the first four follow-up tasks. This additive implementation executes the frozen choices in [the reviewed design](ML-G0-F1-CONFIRMATION-DRAFT-v1.md). Its scope is Cpanel G0 baseline stability and the five-seed F1 full-minus-masked development contrast. Whole-MLB evaluation is a separate, unactivated task. No 2026 data may be accessed.

## Registration and immutable inputs

C1 `EXP-P10-001` uses `selection_status=baseline_stability_only`, `cells=[G0-global]`, and no primary comparisons. C2 is not applicable after D71 closed F4 at its budget gate without an eligible candidate. L6 uses fixed seed sets 1/3/5 and is descriptive.

F1 extension `EXP-P9-002` references the three existing full G0 members from `EXP-P4-001`, the three existing masked members from `EXP-P9-001-v2`, and the declared C1 output. Exactly four distinct production fits are new: full seeds 3/4 and masked seeds 3/4. Six distinct fits are reused; ten ordered member predictions enter the paired analysis. Extension-local references to the two new C1 fits must not be mislabeled as old fits in the combined report.

Both arm configs, the bundle, parent hashes, output locations, source closure including both frozen scorers, contracts, and code commit are frozen before C1 prepare. Execution uses an immutable code checkout C; registration configs may be committed subsequently and passed as absolute paths. The bundle hashes both new and both old configs, but does not contain an impossible hash of itself. Its immutable external registration and authoritative ledger bind its canonical digest. Draft null pins cannot execute.

Old source, configs, datasets, fits, predictions and analyses are immutable. Only verified AppleDouble metadata is ignored. Parent verification checks manifests and dependencies, ordered TRAIN/early/May/June/DEV keys and labels, layout and native runtime. Before any new production fit, replay both original three-seed summaries, raw/calibrated/primary arrays, per-seed summaries and June weights exactly. An identity, dependency, row or replay mismatch stops execution.

## Fixed model and scientific decision

Retain the 52-channel flatten MLP, width128, H5 and seven routing channels; D100 TRAIN1,252,824, early16,000, May2,603, June4,821, CpanelDEV12,334 pitches/328 games; 30 epochs, patience5, batch1024, learning rate0.0005, automatic device and400 delivery draws. The masked arm zeros context[11:28] at every stage and preserves network shape. Same-seed full/masked fits must have equal ordered TRAIN/early rows, device and network signature; derive the C1 signature from its pinned checkpoint. May temperatures and June seed/ensemble blends follow the unchanged scorer. Five-seed weights are refitted on June.

F1 primary is full-minus-masked pitch-weighted NLL on the exposed Cpanel. Pass requires delta <= -0.003, whole-game95%CI upper <0, centered one-sided bootstrap p<=0.05, paired Brier95%CI upper<=0.001 and at least4/5 negative paired seed deltas. Use10,000 whole-game draws, seed20260924. Status is exactly `development_stability_pass`, `worse_or_guardrail_failure` (NLL CI lower>0 or Brier CI lower>0.001), or `inconclusive`. Report reversal separately; never automatically promote masked. This is a second look on the same hypothesis/games, without a fresh alpha guarantee.

R retains all12 groups x2 metrics=24 slots,100,000 whole-game draws, seed20260924, alpha0.05/24, minimum30 games/500 pitches, NLL margin0.010 and Brier margin0.002. Missing `volume_zero` remains structurally unconfirmed. Batter-volume breakdowns use the original TRAIN q25 and remain descriptive. Report the old three-seed result alongside five seeds, `held_out_confirmation=false`, `independent_confirmation=null`; no independent confirmation, whole-MLB robustness or recommendation-policy superiority is established.

## Dependency order and scoring gates

All stages run through `supervise_ml_five_seed_extension.py` with the frozen bundle and local config:

1. `c1-prepare` into the empty C1 output.
2. `c1-profile` through the additive restricted loader, preserving the old8192-row mandatory C1 profile logic.
3. `profile-full`, then `profile-masked`, matched65,536TRAIN x2epochs, early2,048, May64 x400. Load only TRAIN/early/May for these profiles; temporary models are not production checkpoints. Record verification/loading, fit/calibration/inference/scoring projection and memory. Both profiles and the old C1 gate must pass before production fits.
4. `c1-fit --seed 3`, `--seed 4`; then `c1-predict --seed 3`, `--seed 4`.
5. `freeze-third-parent` creates the immutable manifest outside C1, pinning C1 config/source, preparation, fit states/checkpoints and prediction states/arrays for both new full seeds.
6. `f1-prepare` verifies that manifest; `f1-fit --seed 3`, `--seed 4`; `f1-predict --seed 3`, `--seed 4`.
7. After all ten predictions and dependencies verify, `c1-score`, then `f1-score`.

The supervisor refuses C1 scoring while any member is incomplete. F1 scoring requires its baseline NPZ SHA to equal C1 `parent_baseline_predictions.npz`, and reconstructed full-five primary/calibrated/raw/seed_primary arrays to be byte-equal to C1 G0 analysis. No scoring command triggers fitting or selection changes.

## Authoritative wall and interruption handling

Before prepare, freeze C1 share7,200 seconds and F1-extension share7,200 seconds, combined14,400 without transfer. Each member's cumulative fit+predict attempts are limited to7,200. Profile cap600 includes termination cleanup; prepare/fit/predict/freeze/score caps7,200 are further limited by the remaining arm/member allowance. Keep the30-second termination grace inside the effective cap.

One external durable ledger charges full caller wall including verification/loading, prepare, profiles, fitting, prediction, freezing, scoring, failures and interrupted attempts. The masked pre-prepare profile belongs to F1. Internal runner timings are descriptive cross-checks and are never added again. Every start has a unique job ID, command, arm/stage/seed, pinned bundle, UTC time, effective cap and argv; every observed end records elapsed wall, outcome and exit status. Unresolved starts reserve their entire cap and block further work. Validate ledger identity and finite nonnegative accounting before launch. A completed stage requires a successful end and verified artifact binding; file existence alone is insufficient.

Before each launch, charge spent/reserved time and check the projected remaining obligation against that arm's allowance. Fit and predict caps include all earlier attempts for that member. Source/config verification occurs at each command. The worker holds the shared artifact-root `.heavy.lock`; no second heavy worker or nested lock is allowed. Timeout terminates and reaps the process group. Failures preserve outputs and stop the queue for review; no automatic retry, threshold change, reduced rows/seeds/draws or budget expansion is authorized. Any changed budget requires a new registration/output identity.

## Required evidence

Synthetic tests exercise registration serialization, missing/tampered dependencies, paired shape/row/device checks, exact replay/array equality, status thresholds, ledger failures/unresolved starts, timeouts and sequence/ten-member gates. Synthetic success is not an experiment result. After source review, freeze registration, execute one queue and report measured profiles, gates, actual wall, all four fits and ten predictions (or the preserved stop point), C1/L6 and F1/R results, parent/source checks and limits. Keep the Claude calls and actual observed model IDs in the collaboration record; a quota failure is not a completed review.
