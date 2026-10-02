# 전체 June 보정 — 추가 자료 노출 기록

D81 준비·D83 설계 감사 이후 실제 실행에서 발생한 접근을 추가한다. 기존 노출 기록은 당시 이력으로 보존한다. 파일·등록 C/D·출력 해시·전체 열 목록은 [기계 판독 원장](../../results/ML-JUNE-CALIBRATION-EXPOSURE-v1.json)에 있다.

| 단계 | UTC 시각 | 접근과 사용 |
|---|---|---|
| prepare | 2026-09-28 04:14:50.738797 | 승인된 2023-03-30–2025-09-28 정규시즌 가공 캐시 2,145,111행의 저장된 모든 열과 물리 sidecar를 디코딩. 가공 71열·물리 5열(키 3열 중복), join 후 73열. 결과·물리·경기 종료 정보도 메모리에 들어옴 |
| 특징 생성 | prepare 내부 | 2025-06-30 이후 행 제거 후 동결 normalizer/vocabulary/context/cluster 및 시점 일관 특징 생성. 전체 과거 store 1,803,170행에서 모든 June 질의와 같은 타석 선행 투구 보존; 봉인 store 104,970행, 전체 질의 입력 동등성 확인 |
| profile | 봉인 profile manifest 시각 | June store를 unpickle하므로 결과 열도 메모리에 존재. 과거 투구 결과를 입력으로 사용. 별도 June label archive는 열지 않으며 품질 선택·채점 없음; profile 확률은 폐기 |
| 5개 전체 추론 | 각 member manifest 시각 | 결과 열이 있는 store와 별도 June label archive를 모두 열었다. 봉인된 june_y를 member 산출물 스키마에 복사하기 위한 접근이며 품질 선택·채점에는 사용하지 않음 |
| fit | 2026-09-28 04:21:46.216657 | 후보 scalar 30회는 June 104,970구 전체 또는 해당 volume 부분집합에서 적합하고, 기존 가중치 재현 6회는 별도 보존된 Cpanel 4,821구에서 수행. 목적함수는 in-sample 감사 수치이며 독립 성능 추정이 아님 |
| score | 2026-09-28 04:21:54.796989 | 보정·5시드 예측·적용을 봉인한 뒤 기존 P11/P10 DEV label archive를 고정 N3/R78 및 기술 통계에 사용 |

DEV label archive를 score에서 열었다는 사실은 DEV 값이 그때 처음 노출됐다는 뜻이 아니다. 이전 연구에서 이미 사용했고 이번 prepare에서도 전체 캐시 결과 열을 디코딩했다. 현재 결과를 독립 확인이나 새 유의수준 예산으로 해석하지 않는다. 2026 자료 접근·원자료 추가 수집·신경망/temperature 신규 fit은 없다. [실행 결과](ML-June-calibration-execution-2026-09-28.md).
