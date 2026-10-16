# 2026 fullseason T candidate — October 16 handoff

July–September expanding-window OOF predictions and one pooled calibrator are complete and independently accepted. Final three-seed training and verified byte custody are complete and independently accepted. No operational model replacement or public push occurred.

## Frozen recipe and descriptive OOF results

T uses seeds 0, 1, 2, fixed 30 epochs, the final checkpoint, AdamW learning rate 0.001 / weight decay 0.0001, fp32, and batch size 2048. No SELECT, early stopping, seed selection, or per-seed scoring was used. Each fold learned its own FIT-only encoder and date-prior donor assets. Metrics below describe the arithmetic mean of three raw seven-leaf probabilities, before pooled calibration.

| OOF window | FIT rows | OOF rows | NLL | Brier (sum) | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| July | 375,745 | 108,804 | 1.5840613783986264 | 0.755009916988328 | 0.004752967721557551 |
| August | 484,549 | 122,643 | 1.5744651815071151 | 0.7517810868315651 | 0.003262191483357688 |
| September | 607,192 | 105,829 | 1.5806778870731368 | 0.7536571710945396 | 0.008861134357146845 |

September was registered as September 1–30; the fixed S2 source contains rows through September 27. That observation alone does not establish missing scheduled games. These are 2026 EXPOSED DEVELOPMENT, post-hoc recipe-conditional historical replays, not independent tests. No matched samefold baseline or confidence intervals were registered. Differences across months are not improvement evidence.

The pooled calibrator fits one scalar temperature after the three-seed mean, with equal pitch weight across 337,276 unique OOF keys. Accepted T = 1.0493659565735807. No post-calibration metric was computed; this is not an independent evaluation of calibration. Attempt 001 stopped at header validation before optimization. Generated producer keys exposed a C-only layout mismatch; the reviewed C/F compatibility fix preceded successful attempt 002, which performed the sole temperature fit. The actual offending header was not reopened, so its precise cause is not independently established.

## Final candidate data and provenance

The final package combines the fixed S2 and I3 corrected snapshots through game-date October 11: 724,326 source rows produce 720,505 FIT rows in 2,455 games, with 3,821 exclusions (3,820 unmapped labels and one missing BIP type). The I3 contribution is 7,484 eligible postseason rows. There are no final OOF rows. The 199-feature inputs were generated using a fresh FIT-only encoder and strictly-prior profiles. The 665,763,370-byte package contains encoded features, support, and context; deployable serving-profile export remains pending.

Prior accepted schedule metadata reconciles 2,429/2,429 expected pitched regular games and 26/26 postseason games at that cutoff; game 823490 was canceled. This is a fixed-snapshot reconciliation, not a latest-data or future completeness claim. Official-source historical segregation-game rows were authorized for the final candidate; sealed artifacts were not read.

Historical reception and correction availability remains UNKNOWN. Date-prior donors and same-game earlier-pitch context preserve the registered reconstruction semantics, without claiming those corrected values were received live. Actual pitch type is a conditional covariate, not an intended-action/OPE claim. Final FIT reuses earlier OOF labels, so the table is not an evaluation of final weights.

Runtime commits are distinct: July `5af6abb`, August/September `b16b1b4`, final model `622007cb178a54765282c797ee5933e893870895`, successful calibration `bf5a37cf26f4c2e78ac7a11e43b20a90ccda3553`. The later documentation commit does not change those snapshots. Final training saves only finite state dictionaries and performs no final predictions, scoring, or recalibration. A generated last-optimizer-step NaN test verifies that no checkpoint is written in that case. CUDA attention uses deterministic warn-only behavior; bitwise reproducibility is not promised.

## Resource and evidence boundaries

Mac preparations ran sequentially under the existing heavy lock and 4 GiB sampled child-RSS cap. Final preparation completed in 301.46 seconds with peak sampled RSS 3,807,215,616 bytes. August and September used separate GPU3/GPU0 jobs, at most two concurrent jobs and one fit per device. Final training completed on GPU3 in 1,133 seconds (19:44:38–20:03:31 KST), using three sequential seeds, with 8 GiB sampled RSS / 16 GiB VRAM / 3,600-second per-seed ceilings. Other processes were untouched. Outputs are recovered through reviewed byte-custody gates and independent metadata acceptance; no reviewer claim of rerunning raw inference is made.

Checks are reported by scope rather than adding repeated author/reviewer executions: producer 8, bundled worker 2, final adapter 7, final worker 3, trainer 5, calibration 19 (plus 13 subtests), and final recovery 3 generated tests. Independent reviews reran their relevant subsets; those are not extra unique tests. AST-only graph update at `bf5a37c` records 1,129 nodes / 2,981 edges, with zero semantic calls and payload exclusions.

Final custody contains 10 files / 1,991,928 bytes. Each seed consumed every one of 720,505 FIT keys once per epoch for 30 epochs: 21,615,150 presentations and 10,560 optimizer steps. All three supervised children returned zero and were reaped. Independent final acceptance is recorded in `reviews/FINAL-RESULT-ACCEPTANCE-001.json` (SHA `fe39896c6ed81bd66a9451af98285cdacb47d09d0b6015e418ec116d0cb4b1fe`). Integrity verification streams checkpoint bytes; it does not deserialize them or score the final models. This job ran six OOF neural fits plus three final fits; including the earlier July window, the program totals twelve neural fits and one temperature fit.

## Next boundary: seven-leaf shadow serving

The candidate is not deployment-ready. A bound manifest and loader must connect the three checkpoints, encoder, profile/support/context assets, and calibrator. The runtime order is raw seven-leaf seed mean, then one temperature transform. Online context, unknown entities, cache updates, and schema/key/pin checks need generated parity and separately registered replay, latency, memory, and as-of validation.

The existing observer engine uses a five-seed, ten-outcome interface whose WE semantics do not match the new seven-leaf output. Build an isolated seven-leaf prediction/shadow path first; do not directly substitute this vector into existing WE logic. Existing N1/sequence-shadow loading can supply patterns, but its per-seed calibration/selection assumptions differ.

Local audit records are under `20261016-oof-completion-003/{research,planning,reviews}` in the PM job archive; July is preserved under `20261016-followon-execution-002`. The current job's `research/HANDOFF.md` links the exact receipts, accepted custody, runtime snapshots, and the independently reviewed `planning/NEXT-SERVING-WORK-ORDER.md`. All three launch worktrees had their own eight staged binding files removed after accepted custody; original packages, grants, and outputs remain preserved. Main-repository research state and prior sealed artifacts remain unchanged.
