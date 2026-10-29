# 첫 구종 예측 개발평가 등록 — EVAL-PITCH-TYPE-004

상태: **실자료 실행 전 등록/검토 후보**. 작성자와 다른 에이전트 검토 및 Root 수용 뒤 단일 실행한다.
목적은 동일한 투구 전 정보에서 고정 작은 트리가 동결 빈도 기준선보다 어느 정도 다른 예측을 내는지 측정하는 것이다.
독립 TEST·실시간 검증·투구 추천 최적성·팬 효용 실험이 아니다. GPU 사용을 위해 모델을 바꾸지 않는다.

## 자료와 접근

- 원천 `data/raw/d20260911-s2325/statcast_2024.parquet`, 103060977 bytes, footer 711899 rows.
- SHA256 `3cf05edddd22a2eac3d55be11cc56e00fe56f7fcf75823a8ecc8fa89e7e9a274`.
- manifest는 2024-03-20~09-30, frozen=true를 선언한다. 이는 행별 기간/완전성 검증 결과가 아니다.
- 사전 메타 점검은 hash 계산 때문에 **전체 raw bytes를 읽었으나 행/라벨은 decode하지 않았다**.
- 이후 등록 실행은 설정에 적힌 **17열 전체 파일을 decode**하고 날짜로 즉시 분리한다. SELECT/CAL·기간 밖 행도 내부 decode될 수 있으며 미접근이라고 주장하지 않는다.
- 허용열: pitch_type, game_date, game_type, game_pk, at_bat_number, pitch_number, pitcher,
  stand, p_throws, balls, strikes, on_1b, on_2b, on_3b, outs_when_up, bat_score, fld_score.
- SELECT/CAL·기간 밖 라벨/특징은 모델·빈도·통계·선택·보정에 사용하지 않는다. 전체 파일의 물리키 중복/경기 날짜 충돌 검사는 무결성 용도다.
- 새 원천 다운로드·2025/2026 대체·봉인 OPE/RS·기존 결과/가중치 열람은 없다. 해시/크기/행수 불일치 시 중단한다.
- 기존 노출 감사상 2023~25는 인증 미개봉 자료가 없고 노출 UNKNOWN/기존 개발 가능성이 있다.
  따라서 **HISTORICAL_DEVELOPMENT_REPLAY**이며 독립 미사용 검증자료라고 하지 않는다.
- game_date는 원천의 timezone 없는 경기 달력 날짜로만 해석한다. 최종 정정 snapshot의 최초 수신·당시 가용성은 UNKNOWN이다.
  과거 입력수신/예측시각을 만들지 않는다. 각 feature의 pre-pitch 의미와 실제 당시에 받았다는 증거는 다르다.

## 명부·목표·입력

- 원천행 분모 → 정규시즌 game_type=R → 지원 구종 라벨 → 입력 유효성 순서로 구분한다.
- 이는 **지원 구종이 있는 원천 기록행**이며 description/event가 없어 모든 실제 물리투구 완전성을 보장하지 않는다.
- 물리키 `(game_pk, at_bat_number, pitch_number)`는 양의 정수, 전체 중복이면 중단한다.
  경기별 서로 다른 날짜/역할도 중단한다. 경기/행을 결과에 따라 자르거나 임의 downsample하지 않는다.
- 이번 연구결정으로 9분류 순서 **FF,SI,FC,SL,ST,CU,CH,FS,OT**를 명시 채택한다.
  FF/FA→FF; SI→SI; FC→FC; SL/SV→SL; ST→ST; CU/KC/CS→CU;
  CH→CH; FS/FO→FS; KN/SC/EP→OT. 기존 서비스 함수를 자동 상속하는 것이 아니다.
- PO/IN/UN/null/기타 미등록은 개별 key·raw label·제외사유 원장에 남긴다. OTHER로 흡수하지 않는다.
- 입력은 pitcher, balls, strikes, stand, p_throws, base occupancy, outs, **fld_score−bat_score**뿐이다.
  주자 ID는 점유 bitmask 0..7만 쓴다. on_base 결측은 빈 베이스이며 입력 오류가 아니다.
- 현재 실현 구종은 정답만, 현재 위치/속도/타구/득점결과는 읽지 않는다. 직전 투구 sequence·투수 미래 레퍼토리도 없다.
- 입력 오류는 FIT에서 제외하고 원장에 기록한다. EVAL에서는 라벨 적격 분모에 남기며 세 모델 공통 error로 기록한다.
  오류가 없는 공통행에서만 점수를 계산하고 전체 적격/제공률을 함께 보고한다. 사후 시간지연은 미측정(null)이다.
- count 범위 0..3/0..2, outs 0..2, 양의 pitcher ID, L/R 손잡이, 정수 점수 0..100을 요구한다.
  부적절한 경기 종류, 라벨 제외, 입력 오류는 서로 다른 사유다. 물리키/날짜 오류는 숨겨 제외하지 않고 중단한다.

## 날짜 역할과 단일 모델

| 역할 | 시작 포함 / 끝 제외 | 사용 |
|---|---|---|
| FIT | 2024-04-01 / 2024-06-01 | 빈도·encoder·tree·leaf prior |
| SELECT | 2024-06-01 / 2024-06-08 | 예약, 선택 미사용 |
| CAL | 2024-06-08 / 2024-06-15 | 예약, 추정 미사용; identity calibration |
| EVAL | 2024-06-15 / 2024-07-01 | 동결 모델의 개발 점수만 |

- 모든 학습량은 FIT-only이며 앞선 EVAL 정답으로도 갱신하지 않는다. 세 모델이 같은 snapshot 정보를 받는다.
- global prior: `(FIT class count + 4/9)/(FIT N+4)`.
- 투수 기준선: `(pitcher count +4×global prior)/(pitcher N+4)`.
- 상황 기준선: 투수 → 투수+count → 앞항목+타자/투수 손 → 앞항목+점유/outs/score의 고정 계층.
  각 단계 `(group count+4×parent probability)/(group N+4)`, 미관측 단계는 부모로 돌아간다.
  각 행의 단계 지원수·최종 fallback과 미관측 투수를 기록한다. α=4는 이번 사전 고정값으로 성능 선택하지 않는다.
- 후보는 **DecisionTreeClassifier(criterion=log_loss,max_depth=8,min_samples_leaf=100,random_state=0)** 한 개, 한 번 fit.
  FIT-only OneHotEncoder(handle_unknown=ignore)로 pitcher/stand/p_throws를 인코딩하고 나머지는 숫자로 전달한다.
  leaf의 9클래스 실제 FIT count에 `(leaf count+4×global prior)/(leaf N+4)` 평활을 적용한다.
- SELECT 튜닝·CAL 추정·시드 검색·대안 트리·후속 모델을 이번 결과에 섞지 않는다. 의도/위치 예측은 없다.

## 측정·예산·실행

- 한 개 주 비교: tree minus context의 pitch-weighted NLL 차이. Brier(class-sum), top1, top2는 보조.
  pitcher-reference 점수는 기술적 참고다. top2는 실제 투수의 두 구종만 고려하는 추천 성공률이 아니다.
- NLL floor=1e-12, zero true-class probability 수 공개. 고정 schema순 tie break.
- 정렬된 물리키, whole-game paired percentile95% CI, 1000 draws, seed20261029.
  구간은 동결 예측에 조건부이며 학습/선택 불확실성은 포함하지 않는다. 2경기 미만이면 CI=null.
- **기술적 첫 비교만**: 실용개선 문턱/승자/정책채택을 판정하지 않는다. 실제 차이와 구간을 보고하고 다음 실험을 별도 등록한다.
- 실원천 최대800000행, 역할 원천 FIT300000/EVAL100000행, 1 fit, CPU thread1, 주소공간8GiB, wall1800초.
  초과 시 중단하고 범위를 점수에 맞춰 자동 축소/재시도하지 않는다. 버그 수정 재시도는 원인과 새 고정커밋을 남기고 재검토한다.
- `requirements-pitch-type-004.txt`에 전이 의존성까지 버전 고정, task-owned Python3.12 venv만 설치한다. 기존 환경은 불변.
- `gpu run -g 0,1`은 정확 source/config만 들어있는 별도 staging git에서 사용한다. 큰 repo/결과/봉인자료 전체 패킹은 하지 않는다.
  staging 파일 hash가 검토 커밋과 일치함을 launch receipt에 기록한다. raw 파일 한 개만 승인 이후 task-owned input에 복사한다.
- 실행 직전 gpu st의 로그열은 읽지 않도록 메모리에서 last_line을 비활성화한 동일 도구 status를 사용한다.
  UUID/프로세스 메타 확인에서 GPU0/1에 다른 compute PID가 발견됐으며 owner는 현재 namespace에서 UNKNOWN이다. 남의 프로세스에는 신호를 보내지 않는다.
- 실제 연산은 **CPU, CUDA_VISIBLE_DEVICES=''**. task-owned CPU flock만 실행 중 nonblocking 보유한다. 물리GPU lock/유휴를 요구하지 않는다. gpu run의0/1은 transport metadata이며 물리 사용이 아니다. 사용자 GPU0/1 배정 의도는 유지하되 실제 GPU실험은 소유조정 전 대기한다.
- 새 attempt001 디렉토리 exclusive create, 기존 산출물 overwrite 금지. own metrics.json/exclusions.jsonl/predictions.parquet만 회수한다.
  외부 실행 종료코드0과 metrics 존재/해시를 확인해야 완료다. timeout/부분 산출물은 성공으로 해석하지 않는다.
- raw row·개별 예측/제외원장은 task-owned 비공개 산출물에만 보존. repo 결과 문서는 집계/한계/재현 정보만 담는다.

## 근거와 완료조건

기준 코드 `cc98d3671413cac054f05d7a6eb826ab38bd619c`의 synthetic evaluator에서 가변K 점수/경기CI만 재사용한다.
기존 outcome 적격 필터/옛 고정 split/시간 선언 검증기는 사용하지 않는다. model.py의 현재 위치/구종 입력은 이 비교와 호환되지 않는다.
소스 감사: data.py RAW_COLUMNS/add_splits; src/pitcheezy/interfaces/pitch_types.py의9분류; ope/behavior.py는 위치적격/비시간순crossfit이라 미사용.
노출 근거: docs/reports/ML-unopened-data-audit-2026-09-27.md, 현행 docs/pm/research/ws-adaptation/authorization.md.
공식 API 근거: [DecisionTreeClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [OneHotEncoder](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

완료조건: 사전 합성검사 → 별도 등록/코드 검토 → Root 수용 → 단일 등록 실행 → 종료/버전/집계 확인 → 별도 결과 검토.
현재 문서 작성은 실자료 결과가 아니며 미완 단계는 실행 기록과 최종 검토에서 구분한다.

실행준비 개정: GPU 점유 발견 후 Root 지시로 CPU 전용 lock으로 좁혔다. 과학적 모델/기간/지표는 불변이며 새 runner 고정커밋을 재검토한다.

## 실행 실패와 버그 수정 재시도 등록

- attempt001: RootGO 후 등록 source17열 decode와 고정1fit/점수계산을 수행했으나, 결과쓰기 전 provenance의 상대경로를 절대 cwd에 relative_to한 오류로 종료코드1. 출력 디렉토리는 비어 있으며 점수 열람0이다. 이 실행도 실제 fit1회로 센다.
- 국소수정은 상대 source 경로를 그대로 기록하는 runner 한 줄이다. 전체 runner의 합성 mock 성공 경로(출력3개/hash/provenance) 회귀를 추가한다. 과학 코드·모델·데이터·분할·설정은 바꾸지 않는다.
- Root 승인에 따라 새 고정커밋의 독립 diff PASS 뒤 **attempt002 한 번** 재시도한다. 예산은 각 attempt1fit, 누적 최대2fit(실패1+재시도1)이다. 결과 선택/추가 후보 탐색이 아니며 기존 attempt001은 보존한다.

## 두 번째 저장 실패와 fit 없는 복구 등록

- attempt002도 실제 고정 fit/점수계산 뒤 `tree.leaves`의 numpy.int64 JSON 직렬화 오류로 rc1. 누적 실학습2회, 저장실패2회다. mocked runner 성공검사가 실제 반환 scalar를 검사하지 못했다.
- 국소수정은 `int(tree.get_n_leaves())` 한 줄이며 실제 evaluate_frame 결과 전체의 JSON 직렬화 회귀를 추가한다. **추가 fit는 하지 않는다.**
- attempt002의 predictions.parquet(62642행43열, SHA e5d95310f0a52b68ad7124bbe3ae56d2628c3ad0083ed31e7650df8758dfee8f), exclusions.jsonl(1277줄, SHA ace2560a8a137d4767678343034a53464238b4abc6ded10a069d402453b9d579), partial metrics.json(SHA 9d80cd0fd60285fce6f0474c5718a6473229e50c0932fcb4e27253481e8af786)은 보존한다. footer/원장JSON 완결성과 회수hash까지 확인했고 점수는 아직 열람하지 않았다.
- 고정 `recover_pitch_type_pilot.py` 한 개를 별도 검토한다. 동일 raw17열의 load_registered→prepare만 다시 실행하고 저장된 key/label/입력/지원수·제외원장을 정확히 대사한다. fit/encoder/예측 생성은 하지 않는다.
- 저장된 27확률의 유효성을 검사하고 NLL/Brier/topk/같은 경기CI를 그대로 재계산한다. 원partial의 완결된 최상위 점수/분모/구간과 provenance만 엄격 추출해 대조한다. partialJSON을 닫거나 원파일을 덮어 성공으로 만들지 않는다.
- 잘린 tree 구조통계는 미측정(null)로 두고 원실행실패1을 명시한 새 `metrics-recovered.json`을 exclusive 작성한다. JSON 전체 직렬화를 먼저 완료한다. 원모형 커밋0f3ec92d와 복구 커밋/입출력hash를 별도 기록한다.
- Root 복구GO 후 동일 CPU tasklock/1thread/8GiB/1800초로 한 번 재산출한다. 추가fit0·새후보0·분할/설정/지표 변경0. 복구 성공은 저장된 개발 비교를 회수한 것이며 원학습job의 rc1을 성공으로 바꾸지 않는다.
