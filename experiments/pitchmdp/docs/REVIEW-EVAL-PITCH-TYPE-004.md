# EVAL-PITCH-TYPE-004 사전 등록·코드 독립 검토

- 검토일: 2026-10-29 KST
- 최초 고정 검토 커밋: `c1423e9f93469d6c9f512774f53a7dd41c913e1e`
- 최종 실행 후보: `5e8b7e4e545ff17ebd3ae8fea229b5c85b595e8c`
- 버그 수정 재시도 후보: `0f3ec92d6137e6ce390373920392ecb7f03303a7` — 아래 추가 검토가 attempt002에 적용된다.
- 추가 학습 없는 복구 후보: `d8a006002eaf29bcba518bd90d83590191b64ece` — 아래 복구 검토는 원학습 실행의 성공 판정이 아니다.
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

## attempt001 실패 이후 국소 수정 검토

작성자 실행 보고와 큐 `6492701621`에 따르면 attempt001은 원자료 17열 decode·고정 모델 1fit·점수 계산 뒤, provenance의 상대 source 경로에 절대 cwd를 대상으로 `relative_to`를 호출하여 종료코드 1로 끝났다. 결과 파일 생성 전 실패했고 출력 디렉터리는 비어 있으며 점수 열람은 없었다. 실패도 실제 fit 1회로 센다. 검토자는 당시 성공 출력 경로를 실행하지 않고 lock/기존 출력 등 차단 경로만 검사해 이 결함을 놓쳤다. 위 사전 PASS를 실제 실행 성공으로 해석하면 안 된다.

수정 후보 `0f3ec92d6137e6ce390373920392ecb7f03303a7`의 diff를 검토했다. runner의 source hash key를 이미 상대경로인 `str(path)`로 기록하도록 한 줄 수정했고, 모델/자료/분할/설정/지표는 바뀌지 않았다. 등록은 실패 attempt001 보존, 새 attempt002 한 번, attempt당 1fit·누적 최대 2fit을 명시한다. 점수를 보고 후보를 바꾸는 재탐색이 아니다.

새 `test_complete_runner_writes_provenance_without_absolute_relative_mismatch`를 독립 실행하여 **1 passed, 0.73초**를 확인했다. 이 검사는 자료 loader와 모델 계산을 mock하고 임시 source 파일로 runner의 실제 provenance/hash 생성과 세 출력 파일 쓰기를 실행한다. 원자료 decode나 실제 모델 학습까지 포함한 전체 실행 성공 검사가 아니다. 기존 97개와 CPU lock 검사는 과학/잠금 코드 불변이므로 반복하지 않았다.

추가 판정: **PASS — Root 재시도 GO 이후 동일 조건 attempt002 한 번에 한정**. 실제 성공 여부는 새 실행의 종료코드·등록 버전·출력·분모·집계와 별도 결과 검토로 확인해야 한다. 재시도 staging은 수정 커밋과 hash로 다시 고정하며 실패 디렉터리/로그/receipt를 덮어쓰지 않는다. 리뷰 기록 갱신 전 큐를 재조회했고 다른 작업과의 범위 충돌은 없었다.

## attempt002 직렬화 실패 이후 학습 없는 복구 검토

작성자 보고에 따르면 attempt002도 고정 fit·점수 계산 뒤 `tree.leaves`의 numpy.int64를 JSON으로 쓰다가 종료코드 1로 끝났다. predictions/제외원장은 작성됐고 metrics는 일부만 작성됐다. 누적 실제 fit은 2회이며 세 번째 fit는 허용하지 않는다. 검토자는 합성 `evaluate_frame`의 실제 반환값을 JSON 직렬화하여 `leaves`만 int64인 동일 오류를 독립 재현했다. 앞선 추가 검사에서 자료와 모델을 mock했기 때문에 실제 반환 타입 경계를 검증하지 못한 한계를 인정한다.

복구 후보 `d8a006002eaf29bcba518bd90d83590191b64ece`는 `int(tree.get_n_leaves())` 한 줄과 별도 no-fit helper, 제한 등록 및 회귀를 추가한다. 기존 저장 확률을 바꾸거나 새 예측을 만들지 않는다. `recover_pitch_type_pilot.py`는 등록 config hash와 attempt002의 predictions/원장/partial metrics 세 파일 hash를 고정한다. 동일 원천 17열에 `load_registered → prepare`만 적용해 저장 audit의 키·정답·입력·단계별 FIT 지원수와 제외원장을 완전히 대사한다. 모델 fit/인코더 fit/추론 호출은 없다.

확률 유효성 검사 후 고정된 손실·전체 경기 구간을 재계산하고 partial metrics에서 완결된 최상위 점수/구간/분모/coverage/fallback/경기수를 엄격 추출해 정확 일치를 요구한다. 입력 행 정렬 및 C-contiguous 확률 배열은 원계산의 합산 순서를 유지한다. 불일치하면 중단한다. 원파일을 고치거나 잘린 JSON을 닫지 않으며 tree 구조 통계는 null로 남긴다. 원모형 실행 커밋 `0f3ec92d…`, 새 복구 커밋/helper hash, 추가 fit 0/이전 fit 2/원실행 종료코드 1을 구분해 새 파일에 기록한다. CPU 전용·task lock·8GiB/1800초와 exclusive output을 유지한다.

신규 두 회귀를 독립 실행해 **2 passed, 0.85초**를 확인했다. 첫 검사는 실제 합성 tree 반환값의 JSON 직렬화와 fit 호출 금지 상태의 재채점 동치를 확인한다. 두 번째는 실제 합성 모델 산출물을 parquet/원장/직렬화 중단 JSON으로 저장한 뒤 복구 main을 실행하여 새 결과·원파일 hash 보존·덮어쓰기 거절을 확인한다. 이 검사도 실제 원천 데이터 또는 서버 복구 실행 결과는 아니다. 기존 과학 검사는 반복하지 않았다.

복구 사전 판정: **PASS — Root 복구 GO 후 등록된 원천 한 개 17열과 지정한 자체 산출물 세 개를 읽어 한 번 재검증·재채점하는 범위**. 추가 실제 fit/새 후보/분할·지표 선택은 허용하지 않는다. 복구 성공 여부와 실제 결과는 종료·출처·집계의 후속 검토가 필요하다. 원 attempt001/002 실패를 성공으로 재기록하지 않는다. 큐 개정 `6492896778`과 충돌 없는 파일 범위를 확인한 뒤 이 기록을 작성자에게 반환한다.
