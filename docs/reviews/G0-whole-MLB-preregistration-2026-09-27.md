# G0 전체 MLB 실행 전 독립 검토

실행 코드 `5e82bebc239ced378911919de48c106034ee24b0`, 등록 커밋 `3f10f61bea591871f69aad6a869d89d0bbe598ed`. 아래 검토는 등록 직전 수행했으며 등록 후 실제 결과 검토와 구분한다.

## Astra 감사

# Independent preregistration audit — PASS

Audited at 2026-09-27T11:06:31.382287+00:00. Execution source `5e82bebc239ced378911919de48c106034ee24b0`.

The actual two configs and12-stage execution plan pass the independent delta audit:145 experiment Python sources plus supervisor (146pins/stage),26 declared old-parent filehash checks, contract pins, existing June4821-row identity pins, frozen five-seed/400draw protocol, exact stage order and all caps/gates match. Root and execution worktree tracked files are clean and source bytes agree. Both new experiment output directories are absent.

The final delta correctly documents that any activated ensemble/seed fit failure stops the whole correction family without application/scoring; only inactive slots receive p=1. Stage3 caps are900/600/900/900 under3600seconds; evaluation remains7200seconds. Strict stage2 top-level preparation experiment identity is enforced.

## Artifact SHA256

- `EXP-P11-001.yaml`: `b6565f167993bdb31481ce18aee6d49276f3db00b61f518b5eee45e8f00a319b`
- `EXP-P11-002.yaml`: `75d51ecfacfcf41d8d0519d47f3bc1c60f5e8d4949cf3fac25f568668c3acaad`
- `ML-G0-WHOLE-MLB-EXECUTION-v1.json`: `f83e0e95b41d1caec3e8172bf517de38871901be8775cb70b137610b5a4642a0`
- Contract: `6f8f4dc5e850488f40dce0f1433450e1815123b4a10672b3c368b8bf7ad92995`

## Disposition and limits

Approved for coordinator registration commit and execution after other required review completes. No unresolved scientific blocker. No new DEV arrays, raw data, inference, training, fit or quality result was opened. June pins were compared with the existing registration pin file rather than recomputed from arrays. Test results reported by the coordinator were not rerun in this audit. Full checked identities and12stage gates are in `preregistration-astra-audit.json`. This audit is not a model result or independent confirmation.

## Claude Opus 5.5 재검토

실제 응답 모델 `claude-opus-5-5`, 세션 `0a388dc5-7dfe-4166-849c-aa10f2849add`. COOP-008 attempt-001에서 발견한 실패 처리 규약/구현 불일치를 수정한 뒤 attempt-002에서 해소를 확인했다. 원래 지적과 모든 호출 기록을 SSD에 보존했다.

**RESOLVED.** B1 is closed and I found no remaining blockers. The only non-zero exit came from `ls` on the ledger directory, which doesn't exist yet, as expected.

- **B1 (fit-failure semantics):**
  - At `5e82beb`, contract line 48 now says: if any activated candidate has a failed ensemble or seed fit, keep the parameters and the failed manifest and stop all of stage 3. Neither candidate is applied or scored, no p-values are produced, and no rerun or replacement is allowed.
  - p=1 now applies only to inactive slots.
  - The runner already behaves this way:
    - `fit` checks all six fits (count, `fit_success`, and the scipy success flag for I1), writes a `failed_preserved` manifest and exits non-zero (`run_ml_g0_calibration.py:416-429`).
    - `verify_fits` then rejects that manifest, so `apply` and `score` cannot run, and the supervisor won't start later stages after a failure.
    - Code and contract now agree.
- **Stage-3 caps:** the contract now states 900/600/900/900 seconds, bounded by the 3,600-second phase cap. The plan matches: 3,300 seconds total, improvement phase only.
- **Stricter prep identity:** `bind_stage2` now requires a top-level `experiment_id == 'EXP-P11-001'`. The adapter writes exactly that field (`run_ml_g0_whole.py:376`, `'experiment_id': config['experiment_id']`), so the check can pass.
- **Generator:** it now asserts:
  - both HEADs equal C;
  - both trees are clean for tracked files;
  - hashes come from the execution tree and match root;
  - the root and execution-tree contract hashes are identical.
- **Regenerated pins at C = `5e82bebc…`:**
  - Root and execution worktree HEADs both equal C.
  - All 145 sources match, none are unpinned, and both configs carry identical sources.
  - All four commit pins equal C.
  - The contract hash is `6f8f4dc5…` in both configs and in the execution tree.
  - The stage-2 config pin, the plan's config and source pins, and the supervisor's own pin all hold.
  - Gates are seed 0 = profile `five_member_seconds` + 600, then seed 0's full-process wall × 4/3/2/1 + 600.
  - Budgets are 7,200/3,600 seconds and the member, profile and score caps are 1,200/600/600 seconds.
  - The June fit-population pins are unchanged.
- **Output state:** there are no `EXP-P11-*` directories and no ledger. The configs and plan are still uncommitted.

Registration steps still to do, as before:
1. Before committing D, run both `validate_config`s offline and `supervise_ml_whole.py … status`.
2. Commit D and record C, D and the plan's canonical hash on the board.
3. Confirm `.heavy.lock` is free before running `g0-prepare`.
4. The score caps have still not been timed at 311,721 rows (미측정), and a score timeout means no result.

## 등록 완료 증거

재검토 당시 미완료였던 등록 전 검증·등록 커밋·작업대장 기록은 이후 완료했다. 실행 직전 잠금은 실행 담당이 다시 확인한다. 새 품질 점수와 실험 성공 여부는 이 검토의 대상이 아니다.

- `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-whole-mlb/registration.json` — SHA256 `bea0c2ddadfd30bf501c346522827dd81103e5c925a4e1a2c1dfa8fe529d21e2`
- `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-whole-mlb/preregistration-astra-audit.json` — SHA256 `f86ec8fc2aefbb67828af2a7f79b2d4e2b07c6d7ea6162ac759c758773466327`
- `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-whole-mlb/COOP-008/attempt-001/result.json` — SHA256 `1fba9614a17458046e746ca8eb4e05b5f4caef58bfce74749b6754942e5f0826`
- `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-whole-mlb/COOP-008/attempt-002/result.json` — SHA256 `fbb0e64584f0d5f43a407cdf6bb9bc28c841bf7ab6daee805c47998c6a4b5feb`
