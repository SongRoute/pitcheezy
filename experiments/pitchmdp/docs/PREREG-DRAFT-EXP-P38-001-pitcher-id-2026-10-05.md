# PREREG DRAFT — EXP-P38-001 직접 head 투수 ID 임베딩 (2026-10-05, 미등록·미실행)

설정: `configs/DRAFT-CUDA-PITCHER-ID-20261005.json` (생성기 `docs/launch/make_p38_config.py`, status DRAFT,
`execution.enabled` false·`executable` false). 모델 작성 `684f4b4`(cherry-pick -x), 배선 integration-002.

## 1. 질문 (하나)
고정된 periodic_residual_counts(P36·P37과 같은 팔)에 TRAIN 전용 투수 ID 임베딩을 더하면, 이미 있는 투수 profile(ctx 28:51)·support(ctx 51) 위에서 보정된 직접 NLL·Brier가 달라지는가.

## 2. 중복 아님
P1-001/003 투수 임베딩은 예전 전이 신경망. P27은 profile/support 가리기. P13 레퍼토리 특징. 직접 head가 export `pitcher` 열을 모델 입력으로 읽은 적 없음(`run_cuda_direct.arm_inputs`).

## 3. 설계 (결과 보기 전 고정)
- 팔 2개: `periodic_residual_counts`(EXP-P36-001 항목 그대로, 기준) / `pitcher_id_periodic_residual_counts`(같은 항목 + `pitcher_id` 블록, 후보). flatten control 없음(기준 팔이 비교 대상, runner의 control 요구는 이 marker 계열에서만 면제).
- seed 44, 45, 46, 47, 48 → 10 fit. 설정 0–38, P37 초안 39–43과 겹치지 않음. 새 모델 난수일 뿐 새 데이터 아님.
- 학습: P36 그대로(epoch 60, patience 5, batch 1024, lr 5e-4, HPO 0), export manifest 49097109…, 조기 종료(5/1–15)·온도(5/16–31) 분할 그대로.
- 어휘: TRAIN `pitcher` 배열만(라벨 안 읽음), TRAIN 100구 이상 투수, id 오름차순. 나머지(희소 TRAIN 투수, TRAIN 뒤 처음 나온 투수)는 UNK 0. TRAIN 행 날짜 ≤ 20250430, manifest TRAIN max_date ≤ 20250430, 이후 분할 min_date > 20250430을 runner·scorer가 검사. 100·d=8은 고정값, DEV로 고르지 않음. DEV/mlb_dev는 어휘·최소 횟수·폭·보정·선택에 쓰지 않음.
- 초기화: ID 표 0, 투영은 별도 생성기(seed+7919) → 같은 seed에서 학습 전 출력이 기준선과 정확히 같다(CPU 시험).
- 장치: GPU3 UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`만, `max_parallel` 1, 공통 lock `/home/sds733/pitcheezy-autonomy/gpu-locks`. GPU4 `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`는 반납: 허용·probe·대체 금지(registry `check_device`가 거부).

## 4. TRAIN 전용 자원 관문 (측정 전 고정, P37 linear/periodic 1.5 관문을 물려받지 않음)
GPU3 TRAIN smoke 10 task(TRAIN 8,192행 × 2 epoch, DEV 미사용) 뒤 `p38_smoke_gate.py`: 팔별 5 seed 중앙값 `report.seconds / report.epochs_run`, **후보/기준 ≤ 1.25**. ID 경로는 행마다 표 조회 1번과 8→128 투영 1번만 더하므로 이보다 크면 배선·장치 문제로 보고 실행하지 않는다. 같이 거부하는 유효성 검사(자원 관문 아님): 10/10 identity, 묶인 UUID 집합 == {GPU3}, GPU4 없음, `dev_used_for_fit_or_selection` False, 후보 fit마다 같은 TRAIN 어휘 기록(cutoff 이하)·기준 fit엔 기록 없음. smoke는 cap-60·전체 어휘 측정이 아니다.
가족 예상 시간(고정): seed 5 × (기준 2 × 235.20 s + 후보 1.25 × 2 × 235.20 s) = 5,292 s(P36 실제 periodic 최대 task 시간 기준, smoke로 바꾸지 않음).

## 5. 보고 (판정 없음)
- 후보 − 기준, dev·mlb_dev 각각, 전체 행 / TRAIN 어휘 안 행 / UNK 행, 모든 seed와 seed 평균 앙상블, 경기 묶음 bootstrap 95% CI. 점수 파일 `decision` null, 승자·순위 없음.
- 어휘 크기 `n_vocab`, 최소 횟수 100, TRAIN 투수 수, TRAIN UNK 행, 분할별 UNK 비율, TRAIN 날짜 범위·행 해시, 추가 파라미터 (n_vocab+1)×8 + 128×8을 함께 적는다. scorer가 export TRAIN 배열에서 어휘를 다시 만들어 fit 기록 해시와 대조한다.
- −0.003(D118)은 옆에 적기만 한다. 넘어도 채택 아님. NLL·Brier·ECE10만. RL·정책값·OPE와 같은 표에 두지 않음. 2026 미사용.

## 6. 시계
기본 2026-10-05 KST: launch_by 20:45, no_new_fit_after 22:15, kill_own_after 22:30, aggregate_by 22:50(< 23:00). 규칙: no_new ≥ launch_by + 5,292 s(5분 올림), kill = +15분, aggregate = +20분, ≤ 23:00. P37이 끝나지 않았거나 GPU3가 바쁘면 `make_p38_config.py --clock YYYY-MM-DD HH:MM`으로 시계 필드만 다시 만든다(다른 키는 바이트 동일, 시험으로 확인). 선행 조건: P37 파이프라인 상태 complete/incomplete/blocked, GPU3 compute 프로세스 없음.

## 7. 용량 교란 (정직하게)
후보는 (n_vocab+1)×8 + 128×8 파라미터가 더 많다(n_vocab은 실행 때 TRAIN에서 측정). 크기만 맞추고 정보를 끊은 대조군(예: 정보 없는 같은 크기 표)은 **이번 제한된 2팔 연구에 포함하지 않았다**. 그래서 차이가 나와도 ID 정보 때문인지 늘어난 용량 때문인지 나눠 말할 수 없다(인과 귀속 한계). 증가분은 처리의 일부로 공개한다.

## 8. 남은 관문
1. 별도 검토(이 묶음: registry·runner·scorer·생성기·관문·파이프라인·CPU 시험)
2. P37 최종 검토 head와 의미 호환 확인(이 브랜치 기준은 P37 delivery `2265f44`; 검토자가 고치면 rebase 또는 root 통합)
3. P37 종료 + GPU3 비어 있음 → GPU3 smoke 10/10 → 자원 관문 → 등록(새 설정 sha) → 직렬 10 fit
