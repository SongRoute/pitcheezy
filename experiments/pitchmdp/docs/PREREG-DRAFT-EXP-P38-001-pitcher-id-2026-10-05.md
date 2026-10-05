# PREREG DRAFT — EXP-P38-001 직접 head 투수 ID 임베딩 (2026-10-05, 미등록·미실행)

## 질문 (하나)
고정된 periodic_residual_counts(P36·P37과 같은 팔)에 TRAIN 전용 투수 ID 임베딩을 더하면, 이미 있는 투수 profile(ctx 28:51)·support(ctx 51) 위에 정보가 더해져 NLL이 낮아지는가.

## 중복 아님
P1-001/003 투수 임베딩은 예전 전이 신경망. P27은 profile/support 가리기. P13 레퍼토리 특징. 직접 head가 export `pitcher` 열을 읽은 적 없음(`run_cuda_direct.arm_inputs`).

## 설계 (결과 보기 전 고정)
- 팔 2개: `periodic_residual_counts`(P36 항목 그대로) / `pitcher_id_periodic_residual_counts`(같은 항목 + `pitcher_id`).
- seed 44–48(설정 파일 0–38, P37 초안 39–43과 겹치지 않음) → 10 fit, GPU3 UUID c334d905만, 동시 1 fit, 공통 lock.
- 학습: P36 그대로(epoch 60, patience 5, batch 1024, lr 5e-4, HPO 0), 같은 export manifest 49097109…, 같은 조기 종료(5/1–15)·온도(5/16–31) 분할.
- 어휘: TRAIN `pitcher` 배열만, TRAIN 100구 이상 투수, id 오름차순. 나머지와 DEV에만 나오는 투수는 UNK 0(학습에서도 희소 투수가 UNK를 학습 → 신규 투수 대체값). 날짜 ≤ 20250430 검사. 라벨·후속 경기·시즌 통계 안 씀. 100·d=8은 고정값, DEV로 고르지 않음.
- 초기화: ID 표 0, 투영은 별도 생성기 → 같은 seed에서 학습 전 출력이 기준선과 정확히 같다(CPU 시험). 차이는 ID 경로에서만 생긴다.

## 판정 기준 (반복 DEV 탐색, 채택 아님)
- 양(+): dev·mlb_dev 모두 짝지은 ΔNLL 95% CI < 0, 각 분할 seed 4/5 이상 음수 → "반복 DEV에서 투수 ID가 정보를 더함".
- 음(−): 어느 분할이든 CI가 0을 포함하거나 0보다 큼 → "profile/support 위 추가 정보 근거 없음", 직접 head ID 축 닫기.
- −0.003(D118)은 옆에 적기만 한다. 넘어도 채택 아님. 모든 seed·두 분할·UNK/어휘 내 부분집합을 승자 표시 없이 보고.
- NLL·Brier·ECE10만. RL·정책값·OPE와 같은 표에 두지 않음. 2026 미사용.

## 용량 교란 (정직하게)
후보는 (n_vocab+1)×8 + 128×8 파라미터가 더 많다(n_vocab 미측정). 정보를 끊는 같은 크기 대조군은 만들 수 없다: 모두 UNK면 편향 하나라 기준선과 거의 같고, 해시 묶음은 여전히 ID 정보, 일대일 치환은 같은 정보다. 그래서 대조군 없이 2팔로 두고, 증가분은 처리의 일부로 공개한다.

## 남은 관문
1. P37 최종 계보 위에 runner/scorer/queue 배선(`arm_inputs` → `id_inputs`, `build_model`·`load_model`, fit.json에 어휘 기록) + CPU 시험
2. 별도 검토 → GPU3 smoke 2/2 → 새 설정 sha로 등록 → P37 끝난 뒤 직렬 실행
