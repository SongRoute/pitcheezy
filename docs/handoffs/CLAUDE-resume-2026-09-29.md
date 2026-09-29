# Claude 재개 인수인계 — D93 이후

2026-09-29 작성(D92판을 D93으로 갱신; D92판은 git 이력에 보존). 새 세션의 시작점이다. [이전 재개 문서(D91)](CLAUDE-resume-2026-09-28.md)와 이 문서의 D92판 작업 묶음은 끝났으므로 다시 하지 않는다. 과학 규약을 바꾸거나 실행을 승인하지 않는다.

## 1. 현재 위치

- 저장소 `/Users/song/Projects/pitcheezy`, 브랜치 `codex/ml-matrix-execution`, PR [#34](https://github.com/SongRoute/pitcheezy/pull/34)(merge 금지·새 PR 중복 금지). reset하지 말고 현재 HEAD에서 이어간다. 로컬에서 대상이 바뀐 `runs` symlink(` M runs`)는 건드리지 않는다.
- **협업 체제(D93): Opus 5.5 단독.** Sol·Astra·Codex를 호출하지 않는다. 같은 모델의 워크플로 검토는 자기 검토이며 독립 검토로 부르지 않는다.
- D93 커밋은 [기계 기록](../../results/ML-policy-validation-code-v1.json)의 `commits`에 있다. 재개 시 `git status --short`, `git log -5 --oneline`, ListAgents, `docs/decisions.md` 꼬리를 확인한다.

읽기 순서: [AGENTS](../../AGENTS.md)·[CLAUDE](../../CLAUDE.md)·[협업 규칙](../AI_COLLABORATION.md) → 이 문서 → [D93 보고서](../reports/ML-policy-validation-code-2026-09-29.md) → [계약](../contracts/ML-POLICY-MATERIALIZATION-v1.md)·[config](../../configs/ML-POLICY-MATERIALIZATION-v1.json)(권위) → [결정 선택지](../reviews/COOP-018-decision-options-2026-09-29.md) → [코드 검토](../reviews/COOP-018-code-review-2026-09-29.md).

## 2. 이미 끝난 것 — 다시 하지 말 것

| 경로 | 역할 |
|---|---|
| [policy_identity.py](../../experiments/pitchmdp/pitchmdp/policy_identity.py) | `bind_components`·완전한 식별자(D92), 손 등록부·단일 손 지원 template, 후보 모드 맥락 확인(D93) |
| [policy_runtime.py](../../experiments/pitchmdp/pitchmdp/policy_runtime.py) | `ML-POLICY-RUNTIME-v2`: 원인 분리 무행동 거절, `PITCHER_HAND`, E0 `start_population`, 평가 seed q̂, 독립 cand=ref 짝 런타임, alarm 안전 원장 |
| [policy_requests.py](../../experiments/pitchmdp/pitchmdp/policy_requests.py) | 전체 이력 요청, 구조 결함 prefix, 종료용 무투구 행 제외, structural-end-v1·game_final_v1 보상, 창 시작 스냅샷, S0 census |
| [policy_estimator.py](../../experiments/pitchmdp/pitchmdp/policy_estimator.py) | D89 DR, 재귀 안 검열 경계, L0/L1/L2, 경기 부트스트랩, ESS 게이트, 자연 경과 2차 |
| [policy_tau.py](../../experiments/pitchmdp/pitchmdp/policy_tau.py) | 결과 비열람 τ 규칙, S3 측정 기반 탐색 설정 |
| [policy_semisynthetic.py](../../experiments/pitchmdp/pitchmdp/policy_semisynthetic.py) | V2/V3 `absorb_off_mask` 세계, 선언된 검열 hazard |
| [run_policy_validation.py](../../experiments/pitchmdp/scripts/run_policy_validation.py) | 9단계 실행기(등록·검토 게이트, addendum 사슬, 봉인 manifest 결속, 로드 전 선행 조건) |

유지할 의미: 로깅 법칙 = 전체 어휘 BC-P(재정규화 금지). G0 = member별 보정 → 5확률 평균 → 공통 앙상블 혼합. 결정 뒤 정보로 적합·평가 모집단을 고르지 않는다. 검열 PA는 재귀 안 경계로만 다룬다. ≤2025 결과는 노출된 개발 검증이며 인과 효과가 아니다.

## 3. 다음 작업 — 순서

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | **코드 검토 게이트(S0–S2) 판정.** 3차 검증에서 남은 결함을 고친 뒤 사용자가 `review_gates.code_review`를 PASS로 올릴지 결정 | 코드 검토 기록 §3차, 사용자 결정 |
| 2 | **S0 전 등록값 결정**: D-4 두 문턱(`thresholds_before_S0`), source commit, 등록 상태 전환 | decisions 한 줄, config 등록본 |
| 3 | **S0→S2 실행**(사용자 승인 뒤): census → 등록 addendum → materialize-bc·style-snapshot → bind-probe | stage manifest, S1 게이트·S2 합격 |
| 4 | **S3 전 독립 검토**(M-3). Opus 단독 체제에서는 불가 → 사용자가 독립 검토 방법을 정할 때까지 S3 이후는 닫힘 | 독립 검토 기록 |
| 5 | S3 후보·예산, S3b 문턱·경기 수, S5 설계값·허용치, S6 경기 수·예산·무효 비율 등록(모두 S0–S4 실측 뒤, D-11 D) | addendum과 등록본 |

## 4. 실행 경계와 협업

- `policy_frozen=false`, `execution.enabled=false`, config `PROPOSAL_UNREGISTERED`. 2026 추가 열람·수집·fit·추론·OPE, 정책/서비스 승격, 봉인 파일 덮어쓰기, PR merge 금지. 실제 실행은 등록·게이트·사용자 승인 뒤에만.
- Opus 5.5 단독(D93). Fable 제외, Claude→Codex 재호출 금지.
- 코드 변경 후 `graphify update .`(AST만). Ponytail 기본 모드와 연구 예외를 따른다.
