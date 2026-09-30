# COOP-023 — Fable 5.1 런타임 v3 코드 검토

2026-09-30. 검토자 Fable 5.1(작성자 Opus 5.5와 다른 모델), 읽기 전용, 대상 `git diff a454520 b3bd334 -- experiments/`. 합성 검사 45 passed. b_V는 봉인 S5에서 다시 계산해 **0.002734729542731599**(seed 0: 0.00098334 + 1.96·0.00089356)이고, V3 α 0.5 최대 |gap| 0.000867 < MEI라 경고 없음.

**COOP-022 필수 변경 확인:** 런타임 v3(후보 계산 뒤 양성 판정, ρ=0 명시, IntegrityError 가드, sticky MID_PA·validity_violation 유지, 기본 경로 동일), 추정기(`_positivity_step`, 점값, L2 = COMPLETE ∪ ρ0, 전체 E0 ESS, H_k 거절 목록, G-R, S-v1/S-NP/S-B/S-C), ope-2026(고정 스냅샷만, 정규시즌, 스타일 스냅샷 가드, 후보 식별자 pin, 1회·2회 시도 가드, b_V 1e-12 일치).

| 심각도 | 요지 | 위치 |
|---|---|---|
| major (2026 등록) | 어휘 밖 구종 코드 거부가 시도 안에서 일어나 결정론적으로 두 번 모두 실패 → `FAILED_INFRA`. 등록 전에 `MISSING`으로 매핑(G-R 잔여에 포함)하고 개수를 보고할 것 | `run_policy_validation.py:1315` |
| minor | S-B 순서를 (game_date, game_pk, at_bat_number) 사전식으로 | `policy_estimator.py:471` |
| minor | 실패 기록에 부분 원장 해시 추가(COOP-020 M2) | `run_policy_validation.py:361` |
| minor | `history_length=5` 하드코딩 | `:1319` |
| minor | `mlb2026_ope`에 hang guard 키를 두지 말 것 | `:1093` |

작성자 질문: 경기 수를 센 뒤 hang guard를 거는 방식은 **괜찮음**(결과 비열람). 어휘 밖 코드로 시도를 소모하는 방식은 **그대로는 안 됨**(위 major). b_V는 D111의 0.00272가 틀렸고 0.002734729542731599를 그대로 등록(문턱 0.0037347).

**판정:** (a) ≤2025 S6-dr-a3 지금 실행: PASS-with-conditions(estimator 블록·source commit·addendum 등록, S6-a2 열람 뒤 변경임을 공개, S-v1 층 = S6-a2 기대). (b) S6-a3 봉인 뒤 2026 ope-2026: PASS-with-conditions(어휘 처리 수정, b_V 정확 등록, hang guard·검정력 필드, S-B 순서·실패 해시 minor).
