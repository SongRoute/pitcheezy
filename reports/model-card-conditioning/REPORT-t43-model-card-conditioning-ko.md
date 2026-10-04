# T43 — 모델 근거 카드: 구종 조건·추천 방식 문장 (검토 제안 #7)

- 날짜: 2026-10-04 15:0x KST, 브랜치 `codex/model-card-conditioning-1004` (기준 showcase `88976ab`)
- 상태: 직접 검사 통과, 같은 모델(Opus) 검토 대기, Codex 검토 없음. 8771 배포 전.

## 바꾼 것
1. "무엇을 예측하나": "던질 구종이 정해졌을 때" 결과 10종 확률. 위치는 학습 기간 투구 400개로 평균. 점수는 평가 투구마다 실제로 던진 구종으로 계산.
2. "2. 추천 정책 효과"에 "추천 방식" 한 줄: 구종만 고름, 구종 후보마다 타석 끝까지 수비 승리확률 모의 계산, 학습 기간 구종 비율에서 승리확률 높은 쪽으로 비중 이동. 관전 화면 위치는 2025년까지 실제 투구 분포 근사, 평가 안 함.

## 근거 (커밋 b38bea6, `git show`로 읽은 바이트의 sha256을 `model-evidence.json.conditioning`에 고정)
- `experiments/pitchmdp/pitchmdp/matrix_features.py:61` — 현재 공 구종 칸 = 행의 실제 `pitch_type` (후보를 넘기지 않으면). `experiments/pitchmdp/scripts`에는 `candidate_pitch_types` 사용 0건(git grep).
- `experiments/pitchmdp/pitchmdp/sequence_delivery.py:7-12` — JointDelivery: TRAIN 실제 투구 벡터, 투수·구종·손·카운트 단계별 pool, 리그 fallback.
- `experiments/pitchmdp/pitchmdp/matrix_policy.py:111-121` — 정책은 400 draw 필수, 후보 행동마다 `pitch_type=action`.
- `docs/contracts/MLB-2026-POLICY-PREPARATION-v1.md` §3 — ARM-B 구종-only, P3 = π_ref·exp(Q/τ) (TRAIN BC 기준에서 비중 이동).
- 서빙 중 `/api/watch/849843`(T42 스냅샷 사본): `policy.name` "ARM-B P3 (pitch type only)", `location.basis` `realized_delivery_proxy`, `evaluated: false`.
- 미확인으로 남김: 빈도 기준선이 구종을 조건으로 하는지(카드에 쓰지 않음), τ 값(카드에 쓰지 않음; 관전 화면 API가 이미 표시).

## 검사 (맥미니, 가벼운 합성 검사만)
- `npx tsc -b` 0, `npx vite build --outDir dist-t43` 0 (index-C16DW5kO.js, css 기존과 같음 index-CbbIe769.css).
- `DIST=dist-t43 MODEL_EVIDENCE_REPO=<main> node tests/model-evidence-smoke.mjs` → `source identity: ok`, `model evidence smoke: ok` (1440/390/360).
- `cv-evidence-smoke` (CV_M3_RECEIPT 고정) → `receipt identity: ok`, ok. `showcase-smoke` (임시 dist 복사 후 삭제) → ok.
- 음성 대조: 이전 배포 빌드(t34-build, index 2db4feab…)에 새 smoke → exit 1 (`detail: 던질 구종이 정해졌을 때…`). anchor 변조(`pitch_type=candidate_mutant`) → `anchor` 실패. JSON 원복 확인.
- 화면: `model-card-open-390.png`, `model-card-open-1440.png`.
