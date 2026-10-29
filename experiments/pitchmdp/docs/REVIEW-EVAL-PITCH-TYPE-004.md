# EVAL-PITCH-TYPE-004 사전 등록·코드 독립 검토

- 검토일: 2026-10-29 KST
- 최초 고정 검토 커밋: `c1423e9f93469d6c9f512774f53a7dd41c913e1e`
- 최종 실행 후보: `5e8b7e4e545ff17ebd3ae8fea229b5c85b595e8c`
- 작성자: Codex `evaluation_author`; 검토자: 별도 Codex `evaluation_review` 에이전트. 같은 모델 계열의 별도 에이전트 검토이며 독립 모델 검토라고 주장하지 않는다.
- 범위: 등록·설정·신규 adapter/tree·runner·의존성 pin·합성 검사, 기존 가변K 점수/경기 구간 재사용. 검토자는 원자료/가공 자료/기존 결과/가중치/봉인 OPE·RS를 읽거나 서버에 접근하지 않았다.
- 판정: **PASS — Root 수용 후 지정한 역사 개발 파일럿 단일 실행에 한정**. 차단할 미해결 사항 없음. 실험 성공·성능 개선·승격 판정은 아님.

## 등록과 과학적 경계

원천은 등록한 `statcast_2024.parquet` 한 개와 SHA256 `3cf05edddd22a2eac3d55be11cc56e00fe56f7fcf75823a8ecc8fa89e7e9a274`에 고정됐다. 작성자의 사전 메타 실행 범위(전체 byte hash, footer 행수/스키마, manifest 허용 필드)는 별도로 제한 검토했고 Root가 수용했다. 검토자가 원자료 hash/행수/기간을 직접 재검증한 것은 아니다. 실제 runner는 크기/hash를 확인한 뒤 17열만 decode하고 행수를 확인한다. 다른 연도/대체 자료/봉인 산출물로 fallback하지 않는다.

기간은 FIT `[2024-04-01,2024-06-01)`, SELECT `[06-01,06-08)` 예약 미사용, CAL `[06-08,06-15)` identity/추정 미사용, EVAL `[06-15,07-01)`이다. 전체 파일 decode와 물리키/경기 날짜 무결성 확인에는 예약·기간 밖 행도 들어가므로 미접근이라고 주장하지 않는다. FIT/EVAL 분리 이후에만 정답/피처로 빈도·인코더·모델·점수를 만든다. 물리키 중복과 동일 경기의 날짜 충돌은 중단한다.

9분류 매핑을 새 설정에서 명시 채택했고 현재 실현 구종은 정답으로만 읽는다. 결과/위치 적격 필터를 상속하지 않는다. 원천 기간행 → 정규시즌/지원 구종 → 유효 입력의 분모를 구분하며, EVAL 입력 오류는 라벨 적격 분모에 공통 error로 남는다. 미지/결측 라벨은 raw label·key·사유 원장에 남긴다. description/event가 없으므로 모집단은 지원 구종의 원천 기록행이지 모든 실제 물리투구 완전성이 아니다.

빈도 계층·전역 prior·OneHotEncoder·tree·leaf count는 FIT-only다. 앞선 EVAL 정답으로도 갱신하지 않는다. 단일 depth8/minleaf100/log_loss/seed0 트리와 α=4, identity 보정이 고정돼 있으며 SELECT 탐색/추가 seed/대안 모델이 없다. 모든 모델의 예측 대상과 정보 범위가 같고 투수 빈도는 제한된 정보만 쓰는 참고 기준선이다.

주 비교는 공통 유효 입력에서 tree minus context의 투구 가중 NLL 차이다. Brier/top1/top2는 보조다. 정렬된 키와 전체 경기 재표집, 고정 seed/draws, 두 경기 미만 구간 미측정, 0확률/NLL 바닥값 및 coverage를 보존한다. 실용 개선 문턱 없이 **기술적 개발 비교만** 수행하고 과학적 승자를 내지 않는다는 등록과 코드가 일치한다. 이미 노출되었거나 노출 불명인 역사 자료이므로 독립 TEST나 실시간 정확도 증거로 쓸 수 없다.

## 실행 경계와 최종 개정

최초 커밋 검토 이후 실제 GPU0/1의 다른 compute 프로세스가 보고됐다. Root 지시로 최종 커밋은 GPU 유휴 검사/공통 GPU lock을 제거하고 task-owned CPU exclusive flock으로 좁혔다. 이 개정은 모델/자료/분할/지표를 바꾸지 않는다.

최종 runner는 라이브러리 import 전에 `CUDA_VISIBLE_DEVICES=''`와 CPU thread 환경변수 1을 설정한다. GPU 명령·물리 GPU lock·타 프로세스 신호는 없다. CPU lock은 nonblocking이고 raw decode보다 먼저 획득한다. 기존 attempt 디렉터리는 거절하며 주소공간 8GiB·실행 alarm 1800초·등록된 행수 상한과 한 번의 fit을 유지한다. timeout/부분 산출물은 성공이 아니며 외부 종료코드 0과 완료 metrics의 존재/hash 확인이 필요하다.

`gpu run`은 등록한 운송/기록 경로로만 사용한다. 실제 GPU 계산은 이 PASS 범위 밖이며 타 작업의 소유 조정 전 대기한다. staging의 정확한 source/config hash와 최종 검토 커밋의 일치는 작성자 launch receipt에서 확인해야 한다. 원천 한 개의 지정 복사, task-owned 환경/CPU lock/새 attempt 경로만 사용하고 남의 환경·프로세스·봉인 산출물은 보존한다.

## 독립 합성 검사

등록한 격리 Python 환경에서 원자료를 열지 않고 수행했다.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/pitchmdp python -m pytest -p no:cacheprovider experiments/pitchmdp/tests/test_pitch_type_pilot.py experiments/pitchmdp/tests/test_pitch_type_evaluation.py experiments/pitchmdp/tests/test_matrix_metrics.py -q
```

최초 고정 커밋: **97 passed, 1.48초**. 명부/매핑/손계산 평활/정답 및 행순서 불변/미관측 투수/예약 구간 미사용/오류 분모/키·경기·시간 경계/예산/identity mismatch decode 차단/단일 경기 미측정과 재사용 지표 검사를 확인했다.

최종 개정은 과학 코드가 그대로이므로 해당 97개를 반복하지 않고 diff를 검토했다. 추가된 `test_runner_exclusive_output_and_cpu_lock`만 독립 실행해 **1 passed, 0.70초**를 확인했다. 합성 mock으로 점유된 CPU lock과 기존 출력이 raw 접근 전에 거절되며 CUDA 비활성화가 유지됨을 검사한다. 실제 서버 자원 제한/운송/학습 성공을 이 검사로 대체하지 않는다.

## 남는 한계와 인계

최종 정정 snapshot의 당시 수신 가용성은 UNKNOWN이고 과거 예측시각을 만들지 않는다. 실제 지원 수·제외 수·실행 비용·모델 점수는 아직 이 검토에서 측정하지 않았다. 고정 예측의 경기 구간은 학습/모델 선택 불확실성을 포함하지 않는다. 선택 가치·의도 위치·팬 효용·정책 효과·운영 교체는 범위 밖이다.

검토자는 이 파일만 작성했으며 구현/git index/큐를 수정하지 않았다. 기록 작성 직전 공동 큐를 재조회하여 CLAIM 확장 `6492371971`과 CPU 경계 개정 `6492584610`을 확인했고 충돌 댓글은 없었다. 기록을 작성자에게 반환한다. Root 수용 → 최종 staging pin 확인 → 등록된 단일 실행 → 종료·집계·출처의 별도 결과 검토 순서를 유지한다.
