# COOP-020 — Fable 5.1 2026 사전등록 좁은 검토

2026-09-30. 검토자 Fable 5.1(작성자 Opus 5.5와 다른 모델), 읽기 전용, 데이터 비열람. 대상 `432e675`(초안 개정)·HEAD `122864b`. 판정 **PASS-with-conditions**(blocker 1, major 3, minor 4). B1과 M1–M3를 §3/§5a/§9와 JSON에 반영한 뒤에만 등록한다.

| ID | 심각도 | 요지 | 조치 |
|---|---|---|---|
| B1 | blocker | 최소 효과 0.001을 뒷받침하는 편향 보증이 없다. S5 V2 보증 범위는 약 0.01–0.02 WE(MEI의 10–20배). V2 로그를 π̂_b 자체로 만들므로 π̂_b 오지정(후보의 유일한 편향 경로, 단일 강건)을 V2가 검사할 수 없다 | 라벨 분리: `IMPROVEMENT_SUPPORTED_STATISTICAL`(D108 규칙) / `IMPROVEMENT_SUPPORTED`(CI 하한 > MEI + b_V, b_V = |gap_V2| + 1.96·se_V2, 봉인 S5에서) + V3(α 0.5) 오지정 보고, |gap_V3| > MEI면 경고 표시. S5 열람 전 등록 |
| M1 | major | 보상(FrozenWE) 식별자가 사전등록에 pin되지 않았다 | model·manifest·lineage·game_values SHA pin, §4 legacy 문장 대체, WE의 2025 DEV Brier 게이트 통과(노출) 공개 |
| M2 | major | 2026 규모(약 2,430경기, S6의 16배)에서 1회 규칙이 hang guard 중단에 취약 | hang guard = S6 실측 × (n/150) × 2, 중단 시 부분 원장 해시 봉인·비열람, 최대 2회 뒤 `FAILED_INFRA`, 결정론이 아니면 재개 금지 |
| M3 | major | 2026 검정력 공개 | S6 봉인 뒤 Δ·SE와 투영 2026 SE를 §5a에 기록(문턱 불변) |
| m1 | minor | §5 Holm 보조 family 문구가 §5a와 모순 | 삭제, 주 대비 1개 |
| m2 | minor | H_t "또는 충분 상태" 문구 | 요청 필드 목록 명시, 충분성은 식별 가정 |
| m3 | minor | 스냅샷 출처(`6bd76fd-dirty`, 경로) | dirty 내용·경로 명시, ≤2025 stage의 `guard_dates` |
| m4 | minor | τ 0.1에서 ESS 게이트는 구속력 없음 | 명시 |

검토자 답: 판정 규칙은 건전(Δ_lo CI 하한·Δ_hi CI 상한 짝, 대비 1개, 사후 변경 금지). τ 확장은 결과 비열람·공개로 타당성 문제가 아니라 검정력 문제다. 식별 가정은 정직하게 적혀 있다. 2026 정보가 동결 정책으로 새는 경로는 찾지 못했다.
