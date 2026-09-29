# Claude 재개 인수인계 — D95 이후

2026-09-29 작성(D92판을 D93~D95로 갱신; 이전 판은 git 이력에 보존). 새 세션의 시작점이다. [이전 재개 문서(D91)](CLAUDE-resume-2026-09-28.md)와 이 문서의 D92판 작업 묶음은 끝났으므로 다시 하지 않는다. 과학 규약을 바꾸거나 실행을 승인하지 않는다.

## 1. 현재 위치

- 저장소 `/Users/song/Projects/pitcheezy`, 브랜치 `codex/ml-matrix-execution`, PR [#34](https://github.com/SongRoute/pitcheezy/pull/34)(merge 금지·새 PR 중복 금지). reset하지 말고 현재 HEAD에서 이어간다. 로컬에서 대상이 바뀐 `runs` symlink(` M runs`)는 건드리지 않는다.
- **협업 체제(D93): Opus 5.5 단독.** Sol·Astra·Codex를 호출하지 않는다. 같은 모델의 워크플로 검토는 자기 검토이며 독립 검토로 부르지 않는다.
- D93 커밋은 [기계 기록](../../results/ML-policy-validation-code-v1.json)의 `commits`에 있다(검토한 코드 `5186810`). **D94: 코드 검토 게이트(S0–S2) PASS**(사용자 판정, 같은 모델 검토 근거, 독립 검토 아님). **D95: S0 등록값 추천안대로 반영**(D-4 문턱 1.0·0.0, source commit `5186810`); 등록·실행 스위치만 S0 실행 승인 때 켠다. 재개 시 `git status --short`, `git log -5 --oneline`, ListAgents, `docs/decisions.md` 꼬리를 확인한다.

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

### 3.1 S0 등록값 — 결정됨 (D95, 추천안대로)

사용자가 아래 추천안대로 정했다. D-4 두 문턱과 source commit은 config에 반영·커밋됐고, 등록·실행 스위치 네 개만 S0 실행 승인 때 켠다. S0를 돌리려면 아래 값이 모두 있어야 한다. 이미 들어 있고 이번에 해시를 다시 확인한 값: local config pin(`eb364ca7…`), G0 번들 pin, member loader pin, 출력 루트 `…/ML-MATRIX-20260924/ML-POLICY-VAL-v1`, S0 hang guard 7,200초, 결과 인접 분할 `['train']`, 무투구 목록 `automatic_ball/automatic_strike`, seed base 20260929. 추천값을 메모리에서 적용해 보면 S0–S2 등록 검사는 통과하고 S3 이후는 독립 검토 게이트로 거부된다(확인함).

| 항목 | 추천 | 대안 | 이유 |
|---|---|---|---|
| D-4 문턱 T1 `switch_to_secondary_primary_if_unknown_change_share_above` (DEV PA 중 TRAIN에 없는 새 투수로 바뀐 PA 비율 = census `dev.pa_change_to_pitcher_unseen_in_train / dev.pas`) | **1.0 (올리지 않음)** | 0.01 | ≤2025는 2026 추정 대상의 리허설이고 2026 proposal은 "계속 평가가 주"다. 이 비율로 주 추정량을 바꾸면 리허설이 2026과 달라진다. 그런 PA는 D-5 경계로 이미 유효하게 다루고 2차 추정이 자연 경과를 보여 준다. [결정 선택지](../reviews/COOP-018-decision-options-2026-09-29.md) D-4는 '비율이 무시하기 어려우면 자연 경과를 주로'를 조건으로 적었다. 추천은 그 전환을 ≤2025에서 자동으로 하지 않고, 비율이 크면 2026 계약 변경 안건(사용자 결정)으로 올려 두 시기를 함께 바꾸는 쪽이다. 대안 0.01: 이 원인만으로 L1 폭이 약 0.01×(Πρ_c+Πρ_r)만큼 커져 주 경계가 쓸모없어지는 지점(ρ곱≈1이면 약 2%p, 산술 예시이며 측정 아님) |
| D-4 문턱 T2 `light_version_if_below` (투수 교체 PA 비율 = `dev.pa_with_pitcher_change / dev.pas`) | **0.0 (생략하지 않음)** | 0.001 | 2차 추정은 같은 원장에서 계산해 추가 추론 비용이 0이다. 생략할 이유가 없다 |
| `le2025_validation_plan.source_commit` | **`5186810c5b6bf461f2549a9b7f445dedb70ca9f0`** | — | 코드 검토 게이트가 PASS한 코드. 이후 커밋은 문서·등록만 허용되고 코드 경로가 바뀌면 실행기가 거부한다 |
| `registered`, `status`, `execution.enabled`, `execution.real_data_enabled` | **S0 실행 승인과 함께 `true`, `"REGISTERED"`, `true`, `true`** | 승인 전까지 그대로 | 이 네 값이 실제 실행 허가다. 켜도 코드 게이트상 S0–S2만 열리고 S3 이후는 독립 검토 게이트가 막는다. stage마다 따로 승인받아 실행한다 |

### 3.2 결정 뒤 순서

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | **S0 실행 승인**을 받으면 등록·실행 스위치 네 개를 켜고 **커밋·푸시**(D-4 문턱·source commit은 D95에서 반영됨; 실행기는 `--config`가 HEAD에 바이트 그대로 커밋돼 있어야 실행) | `git status`에 config 변경 없음 |
| 2 | **S0 실행**(사용자 승인 뒤, tmux `runs` 창에서; 단일 heavy lock이 비어 있는지 확인). 명령은 아래 | `S0-census-a1/manifest.json`, 실측 비용 |
| 3 | S0 결과 검토: 무행동 원인 교차표, 카운트 경로 단절, 교체, 손 애매, 경계 경기, TRAIN 결과 인접 항목. 무투구 목록 조정은 **S0 구조 수로만**(D-3) | 조정 여부를 decisions에 한 줄 |
| 4 | addendum 1 등록(`census`) → 커밋 → S1 `materialize-bc`, S1b `style-snapshot` → S1 게이트 확인 | 봉인 manifest |
| 5 | S2 `bind-probe`(실제 G0 5 member, 64행, atol 1e-6) → addendum 2(`bc`, `support`, `hands`, `materialize`, `style_*`, `bind_probe`, `bind_identity`) | probe `pass: true` |
| 6 | **S3 이후를 열 독립 검토 방법**을 사용자가 정함(M-3). Opus 단독 체제에서는 불가 | 결정 기록 |
| 7 | S0–S4 실측 뒤 `null` 값 등록(D-11 D): S3 후보·예산, S3b 경기 수·문턱 5개, S5 설계값·허용치(선택: declared hazard), S6 경기 수·행 예산·계획 결정 수·D-7 진단 표본 수, 부트스트랩 무효 비율 | 재등록 기록 |
| 8 | 2026 proposal 두 항목(`profile_as_of`, `mid_pa_pitcher_change_rule`)을 2026 계약 검토 안건에 올림(검토 방법은 6번 결정을 따름) | 검토 기록 |

**D96 상태:** 1번(스위치 커밋 `0b42117`)·2번(S0 실행, exit 0, 495.4초) 완료. 3번 D-3 유지(D97), 4번 addendum 1 커밋(`2ddf9ca`) 완료. S1·S1b는 정렬 결함으로 실패(D98) — 정렬 수정 완료(D99, `9c25d8e`); 코드 검토 게이트를 `9c25d8e`에 대해 재판정받은 뒤 source_commit 갱신 → S0-a2 → 새 addendum → S1·S1b-a2. 결과는 SESSION_HANDOFF D96 항목.

S0 명령(승인·스위치 커밋 뒤, 이미 실행함 — 같은 출력 경로로 재실행 금지). 인터프리터는 `.venv/bin/python`이다(config `environment.invoke_with`; Homebrew 파이썬을 직접 부르면 numpy import에서 멈춘다):

```bash
cd /Users/song/Projects/pitcheezy && PYTHONPATH="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/deps:experiments/pitchmdp" .venv/bin/python experiments/pitchmdp/scripts/run_policy_validation.py --config configs/ML-POLICY-MATERIALIZATION-v1.json --local-config experiments/pitchmdp/configs/local.json --output "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/ML-POLICY-VAL-v1/S0-census-a1" census
```

미구현으로 남긴 것(코드 검토 기록 §3차): D-10 테스트 중 BC 전용 코드 경우, D-3 description 허용 목록 게이트, 합성 인코더의 membership 진단.

## 4. 실행 경계와 협업

- `policy_frozen=false`, `execution.enabled=false`, config `PROPOSAL_UNREGISTERED`, 코드 검토 게이트 PASS(D94), 독립 검토 게이트 `null`. 2026 추가 열람·수집·fit·추론·OPE, 정책/서비스 승격, 봉인 파일 덮어쓰기, PR merge 금지. 실제 실행은 등록·게이트·사용자 승인 뒤에만.
- Opus 5.5 단독(D93). Fable 제외, Claude→Codex 재호출 금지.
- 코드 변경 후 `graphify update .`(AST만). Ponytail 기본 모드와 연구 예외를 따른다.
