# 추천 안정성 진단 — 실제 데이터 smoke 결과

2026-10-03 11:51 KST, 실행·검증 Codex PM. **실행 경로 확인 성공이며 성능 우위/정책 효과/채택 판정이 아니다.**

## 실행과 범위

- 소스: `/Users/song/Projects/pitcheezy-worktrees/codex-showcase-research-1006`, `dd070d28ab22d06255422b060d5f507a94efd518`. 구현 e72f04a, 읽기 전용 사전점검 dd070d2. 실행 중 코드·설정 변경 없음(PM이 사전점검 보고서에 테스트 개수 설명만 덧붙임).
- 초안: `experiments/pitchmdp/configs/drafts/policy-seed-stability-DRAFT-v1.json`, UNASSIGNED/DRAFT 그대로 `--smoke`. 정식 ID·본 실행 등록 아님.
- 2025 June 봉인 선택의 첫 3경기, 요청 916개·비교 888개, 5개 seed/sample 설정. 2026 새 연구 평가·학습 없음.
- 시간: 11:45:48 시작 → 11:48:49 감독기 종료. 실행기 178.14초, 감독기 포함 181.50초, 피크 RSS 5,184,913,408 bytes(약 5.18GB).
- 봉인 설정 재현 20결정의 최대 절대 Q 차이 **0.0**(허용 1e-6). `verdict: null`, `policy_identity_registered: false`. 서비스 G0·D131·D155 판정 그대로.

## 기술 점검 결과 (소표본, 선택 근거로 쓰지 않음)

| 설정 | 봉인 seed0-s6 대비 평균 TV | 1순위 일치율 |
|---|---:|---:|
| seed2-s6 | 0.051820 | 89.86% |
| seed3-s6 | 0.053926 | 90.20% |
| seed4-s6 | 0.055749 | 90.99% |
| seed0-s12 | 0.026961 | 94.71% |
| seed1-s12 | 0.046881 | 91.22% |

seed0-s12는 같은 seed의 난수 접두를 공유하므로 독립 비교보다 거리가 작을 수 있다. 모든 비교가 같은 봉인 seed0을 공유한다. **이 수치로 ‘샘플 수를 늘리면 개선된다’고 확정하거나 설정을 채택하지 않는다.** 본 실행은 아직 하지 않았다. 집계기가 smoke를 정식 결과로 읽지 않도록 거부하는 계약도 유지한다.

## 증거와 운영

- 산출물: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/PM-STABILITY-20261003-01-smoke`. `started.json`, `sealed.npz`, `decisions-*.npz` 5개, `results.json`.
- results.json SHA256: `cf97fabd30197c45c7a099e28850be55041901d9ab60b82d40b0da00ea953838`.
- PM 배열 검사: 888개 고유 키, 3경기, 5변형과 봉인 키·경기 정렬 일치, 유한 확률·합1·TV 범위, 요약 평균과 배열 일치. PASS.
- 감독기/로그: `/Users/song/Projects/pitcheezy-pm-jobs/20261003-next/smoke-002/`. 최대 600초, 공유 `.heavy.lock`, 운영 health·실제 계산 충돌 감시, 중단은 해당 실험 프로세스 그룹만.
- 11:44:42 사전 계산 주기 done=[]·failed={} 뒤 실행, 다음 주기 전 종료. 기존 서버·운영 체크아웃·기록기를 수정하지 않았다. 종료 후 8766 demo=ok/live_policy=ready, 기록기4개·대조·요약 창 존재 확인.
- smoke-001은 대기 중 기록기를 활성 기록으로 분류하여 **추론 시작 전에** 보류. 실제 `--not-before` 대기 코드를 확인해 시작 시각을 구분한 smoke-002에서 실행. 산출물 덮어쓰기·실패 실험 재실행 없음.

## 다음

전체 150경기 진단을 등록하기 전 운영 사전 계산과의 장시간 자원 분리를 확정한다. 이번 3분 smoke를 곱한 비용은 추정일 뿐이며 긴 실행 완료를 보장하지 않는다. 다른 seed·12샘플 후보를 이번 소표본 점수로 골라 버리지 않는다. GPU 서버는 17시 가용 예상이며 최대2장·타 작업 보호를 유지한다.
