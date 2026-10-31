# COOP-019 — Fable 5.1 핵심 과학 검토 (S3 이후 게이트)

2026-09-30. 검토자 **Fable 5.1**(`claude-fable-5-1`, D100 사용자 지정), 작성자 Opus 5.5 — 작성과 검토 모델이 다르다. 읽기 전용, 검토 커밋 `9c25d8e`(HEAD `8bc30c6`은 문서만 다름). 합성 검사 `test_policy_validation.py` 38 passed·9 subtests. 아래는 반환 원문의 정리이며 판정은 검토자의 것이다. 게이트 PASS 기록은 사용자 결정이다.

## 판정: PASS-with-conditions (blocker 0, major 2, minor 5)

| ID | 심각도 | 요지 | 위치 | 근거 |
|---|---|---|---|---|
| F1 | major (S6 열람 전) | V2 합격 규칙(`CI∋0 and |gap|≤tolerance`)에 검정력 조건이 없다. q̂가 같은 세계의 MC 값이라 기준 정책의 DR은 어떤 ρ로도 불편이고, 후보 쪽 보정항은 작은 τ에서 작다. 그래서 V2는 추정량 편향 탐지기가 아니라 연결 검사다(비율 산술은 열거 toy 오라클이 1e-12로 고정) | `policy_semisynthetic.py:169-176` | 추론 |
| F2 | major (2026 등록 전) | `profile_as_of`가 날짜로 고정되지 않고 원천 날짜 가드가 없다. 평가 창이 개막일 뒤에 시작하면 2026 행이 타자 프로필에 들어간다. 2026에서는 rolling 동일성 검사를 돌릴 수 없다 | `MLB-2026-POLICY-PREPARATION-v1.json common.profile_as_of`, `policy_requests.py:277-298,319-328` | 추론+추적 |
| F3 | minor | ESS 게이트가 완료 PA만 보고 검열 PA의 Πρ를 무시한다 | `policy_estimator.py:297-314` | 추론 |
| F4 | minor | ρ=0 흡수 뒤 거절된 PA와 완료된 PA의 L2 포함이 비대칭(서술 층만) | `policy_estimator.py:146-152` | 추론 |
| F5 | minor | 후보는 q̂=Q^ref라 단일 강건, 기준은 이중 강건 | `policy_runtime.py:563-567` | 추론 |
| F6 | minor | τ 규칙이 마지막 행 `events` 유무를 읽는다(값 비열람·등록됨, 약한 결과 인접 선택) | `run_policy_validation.py:1057-1058` | 기록만 |
| F7 | minor | 경기 부트스트랩은 π̂_b·q̂·스냅샷을 고정한 조건부 추론, 투수 군집 없음 | — | 한계 |

## 조건
1. S3b 문턱·격자는 June 원장을 열기 전에 커밋한다(실행기가 non-null을 이미 강제).
2. S5 전: `tolerance`와 함께 `delta_gap_se` 상한(예: ≤ tolerance/2)과 그것을 만족하는 크기를 등록한다. S6 열람 전: 봉인된 V2 원장에서 IS-only(q̂:=0, 같은 행) 추정을 참값과 비교하는 검토자 쪽 확인(파이프라인 변경 없음).
3. S6 보고서: 게이트 옆에 E0 전체(완료∪검열) 가중치의 최소 ESS를 싣고(F3), 후보의 단일 강건성을 명시한다(F5).
4. 2026 등록 전: `profile_as_of`를 명시 날짜(2026 개막일)로 고정하고, 2026 스냅샷 stage에 `max(source game_date) ≤ 2025-12-31` 가드와 대체 일관성 검사를 등록한다(F2).

## 검토자가 확인하고 건전하다고 본 것
비율(로깅 = 전체 어휘 BC-P, M으로 재정규화 없음; 기준 = BC mask+재정규화; 후보 = M 위 `kl_policy`), D-10 off-mask ρ=0 처리, D-5 경계와 L0/L1/L2·E0, 경기 단위 짝 부트스트랩, seed·M-7(planning seed는 기록 행동과 무관), 누수(요청은 엄격히 이전 행만, BC-P mask, τ는 보상·WE 비열람, DEV는 S4–S6만, 2026은 `guard_dates`로 차단), G0 순서(member별 온도→softmax→5개 평균→공통 blend, bind에서 pin), V2/V3 참값(DR nuisance와 독립인 CRN Monte Carlo), D-4 2차 추정량.
