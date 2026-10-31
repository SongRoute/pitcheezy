# G0/F1 five-seed 결과 교차 검토

실제 모델 `claude-opus-5-5`, 세션 `4298d67d-0ea6-412e-abc6-e9819919fa45`, 기준 root HEAD `89deb566fea006697e26f872a03362b52c82604a`. 한도로 실패한 attempt-001은 완료 검토로 세지 않는다. attempt-002의 갱신 전 보고서 지적을 수정하고 attempt-003에서 **RESOLVED — no residual blockers**를 받았다. 총괄이 수정 내용과 재검토를 확인해 수락했다. 독립 예측 확인이 아닌 문서·수치 정합성 검토다.

SSD 기록: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-confirmation/COOP-004`.

- attempt-002 result SHA256: `0bc498aec1bde2d9fdb0aa38fe2ccf1631cb525418ea7d5258def4aa6bce9d40`
- attempt-003 result SHA256: `74133bdebab3e29cd724312a3d2babb1d224633fc16af3e0d1e26672ad4b7dd8`

## 최초 검토 원문

**Verdict: one blocker.** The sealed results, ledger and pins all match the frozen registration. But the root report still shows the in-progress state and has to be rewritten from the sealed results before publication.

## Blocker

**The report `docs/reports/G0-F1-five-seed-2026-09-27.md` is still the pre-result draft.**
- It was last written at 18:43 KST. Scoring finished at 09:48:48Z and 09:48:55Z (18:48 KST).
- Its header still says "실행 큐 진행 중, DEV 품질·최종 비용 결과 미측정".
- Every C1/F1 table cell, the R status, the final walls and the stop point are "미측정". It says full seed 4 is still fitting.
- Meanwhile the docs edited at 18:52 publish the final numbers and link this file as the detailed report: `SESSION_HANDOFF.md`, `decisions.md` D73, `experiments.md`, `ML-matrix-execution-2026-09-24.md` and `ML-followup-activation-review-2026-09-27.md`. So the linked report contradicts D73.

These are the correct values to fill in; I checked each one against the sealed JSON:
- **C1 (EXP-P10-001):**
  - `selection_status=baseline_stability_only`, `comparisons=[]`.
  - `registered_config.registration.c2="not_applicable"`, with the reason given in `c2_basis`.
  - Primary NLL 1.483333, Brier 0.723842.
  - L6 1/3/5 seeds: NLL 1.485071 / 1.483479 / 1.483333; Brier 0.724407 / 0.723924 / 0.723842.
  - `L6.status=descriptive_only_no_independent_confirmation`. Its 3−1 and 5−3 p-values must be presented as descriptive only.
- **F1 (EXP-P9-002), `primary`:**
  - ΔNLL −0.005216 [−0.007177, −0.003311], p 9.999e-05.
  - ΔBrier −0.001465 [−0.002232, −0.000719].
  - Per-seed deltas −.004957 / −.005062 / −.005255 / −.004172 / −.004485, so 5/5 negative. I recomputed these from the per-arm seed NLLs and they match.
  - `decision.status=development_stability_pass`, `reverse_point_estimate=false`.
  - Full / masked NLL 1.483333 / 1.488549.
  - `three_seed_result` (sha 870fdab…, equal to the parent file) keeps the old `predictive_improvement` status and "3seed screen" label separately.
- **R:** `family_size=24`, `per_bound_alpha=0.05/24`. 22 slots pass; the two `volume_zero` slots have `passed=null`, `structural_missing=true`. Overall `status="unconfirmed"`.
- **Bounds:** `held_out_confirmation=false`; `independent_confirmation`, `whole_mlb_robustness`, `policy_effect` and `f4_transfer_claim` are all null.
- **Costs:** C1 282.816 s, F1 279.858 s, total 562.674 s, against the 7,200 + 7,200 s budgets.
- **Collaboration section:** it currently says no Opus review was completed. If the coordinator accepts this review, update that line. Do not describe this review as predictive confirmation.

## Non-blocking (should be disclosed in the report)

1. **Clock correction uses a different timing boundary.**
   - The corrected job is charged 0.929720 s, which is the queue's full-process time for that job. That figure includes the ledger-write and exit tail.
   - Every other job, including failure 1, is charged on the ledger's own boundary. For failure 1 that is 2.090805 s in the ledger versus 2.236179 s in the queue log.
   - This is conservative and not double-counted: one value per job, and runner-internal times are never added. The report should still name the boundary used.
2. **"volume_zero" means two different things.**
   - F1 `limits` says "zero-TRAIN volume is structurally absent on Cpanel". But `batter_volume.groups.zero` has n=757 pitches from 47 batters.
   - The R slot's `missing_reason` makes clear it refers to zero-TRAIN **pitchers**.
   - The report should say "pitcher" explicitly. The batter zero group stays descriptive: NLL CI crosses 0, Brier +0.00052.
3. **C1's own results.json has no config/bundle/commit fields; F1's does.** C1 is still bound through other files:
   - `preparation.identity.config_sha256` 74e1… equals `EXP-P9-002.yaml` `parent_c1.config_sha256`.
   - `registered_config.json` b863… equals the root config.
   - All 6 of its `scoring_sources` match d409031.
4. `results/*.json` and the report are untracked, and five docs are modified but not committed.

## What I checked

- **Results:** the root `results/*.json` files are byte-identical to the SSD `analysis/results.json` files. Manifest `results_sha256` values match.
- **Configs:** root configs match `registration.json` hashes and c8b28f0; the output-side `registered_config`/`bundle`/`declared_c1_config` copies match too. Bundle canonical hash 8640… is recorded in the F1 results, ledger, failure reviews and V3 status.
- **Source:** all 41 `scoring_sources` hashes (6 C1 + 35 F1) match blobs at d409031.
- **Parent pins:** EXP-P4-001 preparation, analysis manifest/results and registered config, and EXP-P9-001-v2 preparation, manifest/results, registered config and config file all match the `EXP-P9-002.yaml` pins.
- **Fits and predictions:**
  - 4 new and 6 reused fits (`unique_new_fits=2` in C1; F1 `costs`); `c1_binding.baseline_sha256_equal` and `c1_binding.full_arrays_byte_equal` are both true.
  - 5 full and 5 masked members, masked seeds 3/4 matched to full seeds 3/4.
  - C1 primary equals F1 full primary.
- **Ledger:**
  - 18 job starts with 18 matching ends, so nothing is unresolved: 16 steps plus 2 failures. The V3 `status.json` shows `completed` with all 16 steps at exit 0, and its final charged totals equal the ledger sum.
  - Failure charges total 3.020525 s. The original record (0a50…) is preserved, the active record equals the corrected one (9aaf…), and the timeout exit code 124 and cap are kept.
  - The hashes of the correction manifest, clock evidence, failure review 002 and queue log all match what the report and D73 cite.
- **2026 data:** I found no 2026 data in C1 preparation apart from timestamps.

**Not checked:** prediction/NPZ arrays (Astra's scope), raw data, and running tests or the scorers (outside this task). This was a report and consistency review, not independent predictive confirmation.

## 수정 후 재검토 원문

**RESOLVED — no residual blockers.** Baseline was root HEAD 89deb56. I re-read both files as they stand now (19:00:04 KST); nothing was edited or run.

1. **Stale report (the blocker): resolved.** The report now says all 16 steps and both scorers finished, and it gives the final C1, L6, F1, R and cost numbers.
   - I checked the new figures against the sealed results and they match: masked NLL, Brier and weight; C1's seed-4 NLL; the L6 3-seed NLL and Brier; the batter-volume counts and ΔNLL; R 22 passed / 0 failed / 2 null; and the queue's outer totals (281.205 / 281.262 / 562.4675 s).
   - The numbers I verified last session also match.
   - The three-seed and five-seed results sit in separate columns, and the three-seed status `predictive_improvement` keeps its "3시드 screen" label.
2. **Timing boundary: resolved.** The report defines the official cost as running from just before the queue launches Python until the supervisor records the end. It says the second failure's 0.929720 s is a conservative substitute taken from the queue's full-process time, charged once per job with no double-counting. It also explains why the outer total (562.4675 s) differs from the ledger total (562.674 s): the outer total excludes the two failures.
3. **Pitcher vs batter volume: resolved.** R's `volume_zero` is described as zero-TRAIN pitchers, left `passed: null` and not counted as a pass. The batter zero/low/high split (757 / 876 / 10,701 pitches) is labelled descriptive, not a hypothesis test.
4. **Opus review status: resolved.** The failed early attempt is still marked as not a completed review. The report says the Opus results/documents review is complete, lists what it changed, and says it is a documents-and-calculation review, not independent predictive confirmation. That is accurate provided this recheck is accepted.

The audit file (`docs/reviews/G0-F1-five-seed-audit-2026-09-27.md`) agrees with the report and the sealed results: the same headline numbers, R 22/0/2 left unconfirmed, the same costs, and the same claim limits.

Once the coordinator commits it, the report can be published.
