# 시드·샘플 수 안정성 진단 실행기 — 실제 smoke 사전점검 (2026-10-03)

작성: Claude Fable 5.1 (root PM 위임, 읽기 전용 감사). 기준 커밋 `e72f04a` (작업트리 `codex-showcase-research-1006`, clean).
범위: `experiments/pitchmdp/scripts/diagnose_policy_seed_stability.py` + `configs/drafts/policy-seed-stability-DRAFT-v1.json`의 `--smoke` 실행 가능성. 텐서·데이터·모델 payload는 열지 않았고 추론·실험은 돌리지 않았다. 실행은 root가 한다.

## 요약
- 차단 요인 없음. import·`--help`·합성 테스트(13개) 통과, 봉인 소스 해시 11개가 EXP-P18-001 기록과 일치, T7 마운트·deps·잠금 파일·봉인 run 파일 확인.
- smoke는 **3경기 전체 결정(이전 smoke 기준 888개)에 대해 5개 변형 모두 비교**하고, **20결정은 봉인 seed-0 재현(replay)에만** 적용된다. 20결정 smoke가 아니다.
- 예상 비용(추정, 미측정): 약 5~7분, 피크 RSS 약 6~7 GB. 근거는 아래.

## 실행 명령 (작업트리 루트에서)
메인 체크아웃(`/Users/song/Projects/pitcheezy`, HEAD `ee4bddf`)에는 이 실행기가 없다(`git ls-files` 빈 결과). 반드시 이 작업트리에서 실행한다. 실행기의 `REPO`는 `__file__` 기준이므로 작업트리의 `configs/ML-POLICY-MATERIALIZATION-v1.json`(sha256 `869b2611…` 일치 확인)을 읽는다.

```
cd /Users/song/Projects/pitcheezy-worktrees/codex-showcase-research-1006
DYLD_LIBRARY_PATH=/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages/torch/lib \
PYTHONPATH='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/deps:experiments/pitchmdp' \
/Users/song/Projects/pitcheezy/.venv/bin/python experiments/pitchmdp/scripts/diagnose_policy_seed_stability.py \
  --config experiments/pitchmdp/configs/drafts/policy-seed-stability-DRAFT-v1.json \
  --local-config experiments/pitchmdp/configs/local.json \
  --output '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/policy-seed-stability-DRAFT-v1-smoke' \
  --smoke
```

- 출력 디렉터리 이름은 `-smoke`로 끝나야 하고(아니면 ValueError), `ML-MATRIX-20260924` 바로 아래의 새 디렉터리여야 하며(`check_location`), T7가 마운트돼 있어야 한다(`assert_output_path`). 위 이름은 현재 존재하지 않는다.
- config 인자는 draft JSON 그대로 쓴다. 본 실행에만 `configs/<id>.yaml` 위치·`REGISTERED`·`execution.enabled`가 요구된다(코드 `main()` 확인). draft의 `execution.enabled: false`는 smoke에서 무시된다(`require(smoke or …)`).
- `KMP_DUPLICATE_LIB_OK`를 설정하면 거부된다. `DYLD_LIBRARY_PATH`가 torch/lib를 포함하지 않으면 `validate_native_runtime`이 거부한다.
- 성공 표시: 표준출력 마지막 줄 `SEED_STABILITY_COMPLETE {...}`와 출력 디렉터리의 `results.json`(`smoke: true`, `verdict: null`).

## 확인한 것 (증거)
| 항목 | 결과 |
|---|---|
| import·`--help` | 위 환경변수로 1.1초에 정상 종료. torch·lightgbm(deps 4.6.0) import 성공 |
| 합성 테스트 | `tests/test_diagnose_policy_seed_stability.py` + `…_aggregate.py` → 13 passed (0.84s). 커밋 메시지의 "18개"와 다름(테스트 함수 5+8=13) |
| 소스 고정 | `swap.SOURCES` 10개 + `diagnose_policy_seed_noise.py` 해시가 `results/EXP-P18-001.json`의 `source_hashes`와 모두 SAME. `SOURCES`에 자기 파일도 포함돼 `started.json`에 기록됨 |
| 등록 체인 | `configs/ML-POLICY-MATERIALIZATION-v1.json` sha256 = config의 `config_sha256` |
| 봉인 run | `…/ML-POLICY-VAL-v1/S3b-tau-a4/`에 `manifest.json`·`ledger-blend-0.jsonl`(91 MB)·`tau_freeze.json` 존재. 내용은 열지 않음; 해시는 실행 시 `sealed_ledger`가 검증 |
| 공유 heavy lock | `…/ML-MATRIX-20260924/.heavy.lock` 존재, `lsof` 결과 점유 없음. `flock LOCK_NB`라 점유 중이면 즉시 RuntimeError(출력 디렉터리 생성 전) |
| local.json | `artifact_root` T7, `raw_allowlist` 2023·2024·2025 parquet만. 2026 파일 없음 |
| 2026 배제 | 입력은 2025-06 blend split 봉인 결정만. `run_policy_validation.py`의 2026 경로는 `ope-2026` 스테이지 전용이며 이 실행기는 호출하지 않음. `outcomes_read: False` 기록 |
| 판정 불변 | `verdict: None`, `comparison: None`, `verdict_rule`/`stop_rule` 있으면 거부. `results/` 저장소 결과 미작성. P17·P18 파일 미수정(해시 SAME) |
| 호출 계약 | `RowBudget(limit, seed_count=5)`·`RolloutImprovement(..., samples, pitch_cap, seed)`·`q_values → (q, diag)`·`simulator.budget.conditional_rows`가 P18 실행기와 동일 패턴, 라이브러리 시그니처와 일치 |
| row_budget | 9,200,000 ≥ 바닥 9,132,084 (1.2×3,805,035×12/6). smoke는 변형당 ≈ 83,670×2 이하라 여유 큼 |
| 머신 | 24 GB. 현재 무거운 python 프로세스 없음(observer worker 2개·시연 기록기 대기 루프만) |

## smoke 범위: 3경기 전체 vs 20결정
- `games = games[:3]` → 3경기의 **평가된 결정 전부**(P17·P18 smoke 기록상 888개, 요청 916개)에 대해 5개 변형 planner를 모두 실행하고 비교·저장한다.
- `replay_limit = 20`은 봉인 설정(seed 0, 6샘플)으로 봉인 Q를 재현하는 검사에만 쓴다(`atol 1e-6`; P17 full·smoke 모두 0.0). 3경기에 결정이 20개 미만이면 실패하나 888개라 해당 없음.
- 산출물: `started.json`, `sealed.npz`, `decisions-{seed2-s6,seed3-s6,seed4-s6,seed0-s12,seed1-s12}.npz`, `results.json`. 집계기는 `smoke: true`를 거부하므로 smoke 출력은 집계에 못 쓴다(설계대로).

## 예상 비용 (추정, 이 실행기로는 미측정)
- 근거: EXP-P18-001-smoke(1변형·6샘플·888결정) 50.0초·83,670행·피크 6.1 GB. EXP-P17-001-smoke(mix 1변형 + 20 replay) 66.1초·피크 6.4 GB.
- 이번 smoke 계산량 ≈ 6샘플 3변형 + 12샘플 2변형 = 6샘플 환산 7변형 + 20 replay → **약 5~7분**, 행 ≈ 60만. 피크 RSS는 같은 입력을 한 프로세스가 공유하므로 **약 6~7 GB** (planner 5개는 캐시 4096항목씩만 추가).
- 본 실행 추정(약 3.8h)은 draft config 기재값 그대로이며 이번 감사에서 재추정하지 않았다.

## 실패·취소 동작
- 모든 검사는 `ValueError`(require)·`RuntimeError`(잠금)로 즉시 중단. 출력 디렉터리 생성(`exist_ok=False`)은 등록 체인·봉인 run 검증 뒤이므로 그 전 실패는 아무것도 남기지 않는다.
- 생성 이후 실패·Ctrl-C는 `started.json`만 남은 디렉터리를 남긴다. 재시도하려면 그 디렉터리를 다른 이름으로 바꾸거나(P17 선례 `EXP-P17-001-smoke.failed1-…`) 다른 `-smoke` 이름을 써야 한다. T7 삭제·이름변경은 root 결정.
- 잠금은 프로세스 종료 시 해제된다(flock). 저장소 파일은 어떤 경우에도 쓰지 않는다.
- 결과는 끝에 한 번에 저장된다. 중간 저장 없음.

## 운영 창 (D159·OPS freeze)
- 현재 11:41 KST. 첫 시연 기록기 `--not-before 2026-10-03T16:50Z`(= 10/4 01:50 KST) 대기 중. 실험 종료 시한 01:40 KST까지 여유가 충분하다.
- 기록기 완료 뒤 `demo_precompute.py live-check` 루프가 같은 `.venv` python을 쓰므로 그 시간대는 피한다. 지금은 유휴.

## 남은 주의점 (차단 아님)
1. 커밋 메시지 "18 synthetic tests"는 실제 13개. 문서 정정 대상.
2. 출력 이름에 ID가 없으므로(UNASSIGNED) smoke 디렉터리명은 PM이 정한다. 위 제안 이름은 아직 쓰이지 않았다.
3. 사용자 권한: PM-charter §2·§5에 따라 기존 자료·규칙 안의 실험 실행은 PM 판단이며 재승인 불필요. smoke는 저장소 결과를 쓰지 않으므로 중대사안 아님.

## PM 확인 메모

이전 보고의 18 passed는 새 테스트 2개 파일(13 tests)과 기존 P17/P18 회귀 2개 파일(5 tests)을 함께 실행한 값이다. 이번 사전점검의 13 passed와 검사 범위가 다르며 기존 18개 결과를 오류로 정정하지 않는다.
