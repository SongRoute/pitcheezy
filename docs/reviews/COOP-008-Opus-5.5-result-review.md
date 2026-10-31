# G0 전체 MLB 결과·보고서 교차 검토

실제 응답 모델 `claude-opus-5-5`, 세션 `0a388dc5-7dfe-4166-849c-aa10f2849add`, COOP-008 attempt-003. 아래 원검토에서 차단 사항이 없음을 확인했다. 총괄은 네 가지 선택적 표현 수정을 반영했다: 검토 완료 표기, bootstrap 해상도 하한, 전체 큐 비용의 범위, R margin의 중립적 설명. 봉인된 실험 산출물·점수·판정은 변경하지 않았다.

**No remaining blockers.** I checked every number and denominator in the report against the sealed SSD artifacts and they match exactly. I changed no files and reran nothing.

**Identity and provenance**
- HEAD is D `3f10f61`, with C `5e82beb` as its parent.
- The config, plan and registration hashes in the report match the files and `registration.json` (`b6565f16…`, `75d51ecf…`, `f83e0e95…`, `bb1e1394…`).
- The sealed files match their reported hashes: stage-2 and stage-3 `results.json` and manifests (`9fefaaa6…`, `aaf5db0d…`, `c5b9f293…`, `8fe8f970…`), `stage2_binding.json` (`c2fe82bb…`), `activation.json` (`2b84e899…`), `fits/manifest.json` (`96c0739e…`) and `fits/I1.json` (`dbeaa2de…`).
- The queue status (`c784953d…`) and Astra's post-run audit (`60e88ecc…`) also match.

**Stage 2: frozen five-seed G0 against frequency**
- **Denominators:** 341,941 requested; 311,721 pitches and 1,161 games evaluated. Cpanel overlap is 12,334 pitches in 328 games; the complement is 299,387 pitches in 1,161 games, sharing 328 games with Cpanel.
- **Replay:** the maximum absolute error for `raw`, `calibrated`, `primary` and `seed_primary` is 0.
- **Primary contrasts:** whole-MLB ΔNLL −0.013673 and complement −0.013666. The CIs, ΔBrier values and 5/5 seed deltas match to every printed digit.
- **Holm-2:** both adjusted p = 0.00019998. Both slots are `development_advantage_over_frequency`.
- **R24:** 24/24 bounds pass, with 0 unmeasured.
- **Other reported figures:** ECE10 (G0 0.00871 vs frequency 0.00166; complement 0.00890 vs 0.00149), pitcher-macro NLL, all 10 classes with at least 30 events (minimum 272), and 0 missing/unknown metadata all match.
- **Fits:** `new_fits` 0, `reused_fits` 5.

**Stage 3: activation and June-only fits**
- **Activation:** I1 qualifies on classes {0, 1, 5}, with gaps matching the report. I2's `unmeasured` status is correct: only the high group meets 30 games and 500 pitches in both June and DEV (zero 0/0, low 12/635, middle 19/1,014, high 86/3,172), and the rule needs two groups.
- **Fits:** all six I1 fits used `june_cpanel` with n=4,821 and the pinned keys/labels hashes. All six passed SLSQP success, feasibility, finiteness and the not-worse-than-initial check.
- **Sealing order:** the fit manifest is `sealed_before_dev_apply`. I2 is recorded as `not_activated` with no fit and no candidate output.
- **Calibration fit counts:** 6 new calibration-parameter fits, 0 neural fits.
- **I1 contrast:** ΔNLL +0.000876 [+0.000633, +0.001114], ΔBrier +0.000306, 0/5 seeds negative, p = 1. Status is `worse_or_guardrail_failure`.
- **Holm:** adjusted p = [1.0, 1.0].
- **R52:** I1's 26 bounds pass. I2 keeps 26 inactive slots in the denominator. The 95% two-sided lower bound is above 0 in 11/13 groups for NLL and 10/13 for Brier, as reported.
- **Interpretation:** the report correctly states that R passing does not reverse the N deterioration, and it doesn't present the fallback weights as an I2 candidate.

**Cost and attempts**
- There is one attempt directory, and the ledger has 12 starts and 12 ends, all completed with exit 0. There are no failed or orphaned records.
- Ledger sums: evaluation 1,821.246 s of 7,200 s; improvement 34.423 s of 3,600 s.
- The outer queue wall of 1,857.487 s is kept separate and not added to either phase.

**Interpretation boundaries**
- The report and both result files keep `independent_confirmation: null`, `held_out_confirmation: false`, `policy_effect: null` and `service_adoption: null`.
- The report states the prior T4 exposure and the 328 shared games.
- It denies any unseen-player learned correction and makes no target-location, policy or 2026 claims.

**Optional presentation fixes (not blockers)**
1. Report lines 3 and 44 still say Opus's final review is pending. Update them to record this review.
2. Line 19–20: Holm p 0.00019998 is the resolution floor of a 10,000-draw bootstrap (raw p = 1/10,001 with the plus-one correction, ×2). It could say "at the bootstrap resolution floor".
3. The EXP-P11-002 `cost` field `outer_queue_full_seconds_separate: 1857.49` is the whole-queue value. The stage-3 supervisor's own wall was 34.97 s, so labelling it as the shared whole-queue cross-check would avoid a misreading.
4. Line 38 calls I1's R margins "넉넉한" (comfortable). The largest Brier upper bound is 0.001014 against a 0.002 margin, about half the margin, so "margin 내" (within margin) is more accurate.

This is a code, provenance and number-consistency review. It is not an independent predictive confirmation. Both stages use already-exposed 2025 DEV, and stage 3 was activated adaptively, so the result shows G0's development advantage over frequency and that I1 made things worse. It is not independent generalization evidence.

원본 결과 SHA256 `4b8c0abfe1200c7f753a9662a19b9e0f701f93a1b88cdc93aa51904e3fcd43eb`. 원본 경로 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-whole-mlb/COOP-008/attempt-003/result.json`.
