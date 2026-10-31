# G0 동결·자료 노출 감사 교차 검토

실제 모델 `claude-opus-5-5`, 세션 `fef39276-1f6d-4f22-8b3f-7cba61728394`. 차단 사항 없음. 총괄은 선택적 지적 중 raw/runtime 검증 범위 플래그와 노출 감사 문구의 범위를 명확히 했다. 이 변경 후 Astra가 최종 번들96개와 해시를 다시 검증했다. 그 밖의 검증 한계는 통합 보고서에 보존하며 자기완결 재현 보장으로 해석하지 않는다.

No blockers in either artifact. For G0, the pure verifier passes against today's bytes. For the audit, all 12 evidence hashes match the `cf20fb4` snapshot and every claim I spot-checked matches its cited source. This review covers implementation and provenance only; it is not independent predictive confirmation. I wrote no files and ran no tests, fits, inference or 2026 access. I didn't open arrays, parquet or `.pt` content, only hashed their bytes.

## 1. G0 research freeze (Sol) — no blockers

**What I verified**
- `python3 scripts/freeze_g0_research.py verify` gives PASS: 96 files, 5 members, and bundle SHA `aa18b9d3…`, which matches the stored result. `builder_sha256` equals the current script's hash (`d9d2eec3…`), so the freeze was rebuilt after Sol's guard fix.
- All 30 `source/` and 5 `scoring_source/` files match the git blobs at `5e82beb`. The verifier itself doesn't check this (see nit 1).
- **Members:** seeds 0–2 come from P4 (`g`) and seeds 3–4 from P10 (`c1`). Each member is checked against five prep fields (model, fit report, fit state, May calibration, prediction state) and is fail-closed on order.
  - May temperature must match in the calibration file, the prep and the frozen calibration.
  - The June ensemble weight must equal the C1 selection, and the per-seed June weights must equal the C1 seed reports.
  - `refit_on_whole_mlb` must be false.
- **Features, clusters, panel:** these must be identical across P11, P4 and P10.
  - The aux, panel, clusters, key and frequency files are pinned by hash. I confirmed separately that their hashes equal the entries in P4/P11 `artifact_hashes`.
  - Frequency and metadata are linked to the P11 manifest and to the P11 config's parents.
- **Boundaries:** `research_only`, `service_promotion=false` and `independent_confirmation=false` are all enforced.
  - The `independent_confirmation` guard at `scripts/freeze_g0_research.py:117` runs before any parent read, and its mutation test (`tests/test_freeze_g0_research.py:103`) is placed correctly.
  - Any edit to the bundle's free-text fields changes `bundle_sha256`, so verify fails against the stored result.

**Non-blocking nits**
1. `scripts/freeze_g0_research.py:96`: scoring sources are hashed from a mutable worktree (`pitcheezy-worktrees/g0-whole-mlb-execution`). Only the manifest ties them to `5e82beb`, not git. Checking the git blob, or reading via `git show 5e82beb:…`, would remove the worktree dependency.
2. The raw-data distinction is only structural. `archived_references` (raw parquet SHAs, `processed_sha256`, P3 prep hash) is copied from P4 prep and never re-hashed; I did not re-hash it either. The verification result only reports `file_count: 96 / PASS` and has no `archived_raw_identity_verified: false` field. A reader could take PASS to cover the raw data, so that flag is worth adding.
3. `verify_semantics` doesn't assert the aux/panel/cluster/key-to-P4 `artifact_hashes` links. They hold today, so this would only be an extra safeguard.
4. `:172`: June seed weights are mapped by list position (`enumerate`), not by an explicit `seed` field.
5. Tests don't cover flipping `research_only` or `service_promotion`, a mismatched May temperature, a mismatched June weight, or source-closure drift.

**Verification limits**
- **External SSD:** everything except the 4 repo configs lives on the T7 SSD, and verification fails if it isn't mounted. The `availability` string says so.
- **Environment:** native libraries, the SSD `deps/` folder, the Python binary and `.venv` are recorded only as version and path strings, not hashes.
- **Not re-bound:** P4 `blend_metadata`/`dev_metadata`, the P3 prep and the `eef5417` repo results aren't bound by the verifier. `evidence_head` is only a constant.
- **Scope:** this is a reference inventory, not a reproducibility guarantee. `pytest` was not run.

## 2. Unopened-data audit (Astra) — no blockers

**What I verified**
- The JSON's SHA is `be306090…`, which matches the report. All 12 evidence files match both `git show cf20fb4:` and the working tree.
- **Claims checked against their sources:**
  - The temporal table gives 608,722 / 16,000 / 1,396 / 2,415 / 7,737 / 88 for 2024 and 1,252,824 / 1,146 for 2025, and says 2025 TRAIN includes the 2024 evaluation period so the two years aren't independent repeats.
  - The old split in `data.py:72` is history before 2023-05-15, train through 2025-04-30, calibration through 06-30, and DEV for everything after, with no upper date bound.
  - The legacy CAL in `run_sequence_ablations.py:76` samples all eligible May–June rows with no panel filter. `FOLLOWUP_RESULTS.md:132-134` records the 4,000/12,000 split and that the 12,000 were also used for epoch selection.
  - Prior-day batter histories come from terminal events (`data.py:165-182`).
  - The whole-MLB report has 341,941 requested, 311,721 eligible, 12,334/328 Cpanel, 299,387/1,161 complement, 328 shared games, June group support of 0/635/1,014/3,172 pitches, and says T4 already scored this DEV.
  - The P1 config shows 2023–25 training and a 2025 holdout. The contract gives 2,145,111 rows, regular season only.
- **Conclusion is scoped correctly:**
  - `verified_unexposed_population_count: 0` and `independent_confirmation_available: null`, not a claim that nothing exists anywhere.
  - Missing logs are explicitly not treated as proof of unseen data.
  - June non-Cpanel stays *unknown* because the legacy CAL overlap wasn't computed.
  - There is no new 2026 authorization, and 2026 is explicitly "not a fallback holdout".

**Non-blocking nits**
- The report says "12월까지 DEV" (DEV through December), but `data.py` actually has no upper bound. That is a slightly softer wording, not a wrong one.
- The section "June 지원·비용 감사의 과학적 경계" is written in the present tense ("…만 읽는다") and says "이번 메타데이터 열람도…". It could be misread as work this audit performed. A note that it sets bounds for a future, separately registered June audit would fix that.

**Verification limits:** I checked a sample of claims against the cited `cf20fb4` files only. I did not do any row joins, open raw data, inventory the file store, or check whether the SHAs in `archived_references` match current bytes. The report may change after `cf20fb4`; these hashes only certify that snapshot.

원본 SHA256 `5ae488d3cf77301847890a865e7b90b77a274bc7839c8032e56c408362b56ab0`; 원본 경로 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-g0-freeze-audits/COOP-010/attempt-001/result.json`.
