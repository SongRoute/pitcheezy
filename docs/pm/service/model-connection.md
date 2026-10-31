# Step 5 합성 모델 연결 (서비스)

소유: 서비스 PM · 갱신: Root가 새 수용 결정·검토를 넘길 때 · 마지막 확인: Root 동결 상태 2026-10-11T00:11:58Z(09:11:58 KST).

이 쪽은 연구 [측정 정의](../research/ws-adaptation/measurement-schema.md)에서 주제별로 옮겨 온 Step 5 합성 모델 연결만 다룬다(2026-10-11 주제 분리, 문구·수치·증거 링크는 그대로 옮김). 값은 Root 동결 상태 receipt [ROOT-FINAL-STATUS.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-final-status-checkpoint-001/ROOT-FINAL-STATUS.json)(SHA-256 `12a439268b442a4562a15847c697a3a97a30e8e694e45b619717ec89f66df6fa`)에서 옮겨 적었고, 작성자는 배열·raw·모델·결과 payload를 열지 않았다. 아래 어느 항목도 고정된 199-feature source, label, live-support, 서비스 측정 계약을 바꾸지 않는다. Step 2 자료 역할은 **2026 EXPOSED DEVELOPMENT**이며 독립 TEST, 2024 대비 우위, 물리적 투구 전체 커버리지, 자동 서비스 채택은 없다.

## 측정 단위

Root `ACCEPT_BOUNDED_ACTUAL_SYNTHETIC_MODEL_CONNECTION`([STEP5_2024_ACTUAL_SMOKE_ROOT_DECISION.json](/Users/song/Projects/pitcheezy-pm-jobs/20261010-ws-service-006/STEP5_2024_ACTUAL_SMOKE_ROOT_DECISION.json) `2f9054b10696ca0d4ee7366077b5ffb48e87e3aa5938164e6665b1eaebbf0f6b`)의 단위는 서로 다르므로 섞어 세지 않는다.

- **투구 시도(pitch attempt) 1** × **pitch-type action 17** × **leaf 값 7**. 입력은 합성 17×199 행렬 하나다.
- 모델 load 3회·predict 9회, direct 대 service 최대 차이 0, 직접 자식 1개 rc0. 모델/native SHA와 경로는 grant와 일치했고 크기는 자식 관측값뿐이다.
- 별개 출력 검토는 **REWORK 33/35**로 보존한다([REVIEW.json](/Users/song/Projects/pitcheezy-pm-jobs/20261010-ws-service-006/step5-2024-actual-smoke-result-only-review-001/outputs/REVIEW.json) `cb2bdbf0de40d17644ce0802fc552e93257e6cf2755e49fd242de84ed7252f3b`). Root가 잘못된 관리 지시 비교 2건을 조정했을 뿐 검토자 PASS로 바꿔 부르지 않는다([정오표](/Users/song/Projects/pitcheezy-pm-jobs/20261010-ws-service-006/STEP5_2024_ACTUAL_SMOKE_ORDER_ERRATUM.md) `b5a26acae8f6c7fc0545d0f3a505c4000c6711052668b3c41eef11747058f675`).
- 합성/실제 경계: 검증된 2026 실제 pre-pitch profile, roster-as-of, 입력 가용성, 물리적 시각, reconciliation, calibration, 전체 투구 커버리지, 정책·배포는 없다. 관측된 header parity로는 모델의 학습 선택 parent iteration이 입증되지 않는다.
