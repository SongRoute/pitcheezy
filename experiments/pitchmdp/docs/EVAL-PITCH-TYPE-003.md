# EVAL-PITCH-TYPE-003 — 다음 구종 평가와 빈도 기준선의 합성 검증

상태: **SYNTHETIC_ONLY / 실자료 게이트 CLOSED**. Song이 동의한 “투수의 평소 빈도 → 상황별 빈도 → 같은 투구 전 정보를 쓰는 작은 트리 하나”의 첫 준비 단계다. 이번에는 앞의 두 기준선과 평가를 합성 자료로만 구현한다. 트리는 선정·학습하지 않으며, 비교 입력을 확인하는 후보는 **HYPOTHETICAL_NOT_FITTED_TREE**다. 가짜 후보 점수를 학습 성능으로 보고하지 않는다.

## 소스만 읽어 확인한 가능성과 한계

기준은 `3863ba1b80b57ac702a969a976a6b70827586bc7`의 아래 파일이다. 원시/처리 자료·결과·가중치·봉인 OPE/RS는 읽지 않았고 loader나 기존 행동 모형을 호출하지 않았다. 열이 존재한다는 사실은 실제 투구 전 수신 가능성의 증거가 아니다.

| 읽은 소스 | 확인한 사실 | 이 작업에 적용할 경계 |
|---|---|---|
| `experiments/pitchmdp/pitchmdp/data.py`의 `KEY`, `RAW_COLUMNS`(20–32행) | `game_pk/at_bat_number/pitch_number`, 투수·카운트·좌우·주자·아웃·점수 관련 열이 있다. 실제 구종/좌표/구속과 `post_*`도 있다. | 물리 키와 투구 전 상태 개념만 참고. 현재 공 실현값·사후 점수는 입력 금지. 최초 수신 시각·의도 목표 관측은 이 열 목록으로 보장되지 않는다. |
| 같은 파일 `add_splits`(69–75행) | 기존 2023–2025 자료를 history/train/calibration/dev로 나눈다. | 새 구종 예측의 FIT/SELECT/CAL/TEST·노출 계약으로 자동 상속하지 않는다. |
| `experiments/pitchmdp/pitchmdp/model.py`의 `CATEGORICAL/NUMERIC_INPUTS`, `eligible`(20–24, 43–48행) | 현재 구종·좌표를 조건부 결과 입력으로 쓰며 결과 라벨/좌표/구종 유무로 적격성을 제한한다. | 다음 구종 정답을 미리 입력하거나 결과/좌표 지원에 따라 새 명부를 걸러내지 않는다. 기존 학습 가중치도 사용하지 않는다. |
| `src/pitcheezy/interfaces/pitch_types.py`(1–43행) | 9분류 매핑: FA→FF, SV→SL, KC/CS→CU, FO→FS, KN/SC/EP→OT. PO/IN/UN/null/미지 코드는 제외한다. | 기존 행동 축의 계약이며 새 예측 정답으로 자동 채택하지 않는다. 병합/제외의 과학적 의미와 제외 분모를 별도 등록한다. |
| `src/pitcheezy/ope/behavior.py`의 `behavior_factors`, `behavior_logged_crossfit`(46–90행) | 계층 평활을 쓰지만 `action_id>=0`의 구종·위치 공동 적격 자료를 쓰고 경기 번호 나머지로 교차 적합한다. | 평활 아이디어만 참고. 위치 지원 코호트·비시간순 폴드·자료 실행을 재사용하지 않는다. 새 전역 사전분포는 모든 등록 클래스에 양의 질량을 준다. |
| `experiments/pitchmdp/pitchmdp/matrix_metrics.py` | 확률 검증은 클래스 수 인자가 있지만 손실/쌍 비교는 10분류에 고정돼 있다. | 검증 함수만 재사용. 새 구종 확률을 10칸으로 패딩하지 않고 작은 가변 클래스 수 손실·경기 구간 계산을 별도 구현한다. |

## 합성 계약과 입력

설정은 `configs/EVAL-PITCH-TYPE-003.json`이다. 정답 순서는 합성 전용 **FF/SL/CH/OTHER**이며 identity map만 허용한다. 실제 MLB 최종 분류가 아니다. OTHER는 명시적인 합성 클래스이고 null/미등록 라벨을 받아주는 통이 아니다. 예를 들어 FA·PO·OT도 자동 병합 없이 실행을 거절한다. 실제 미매핑 자료는 별도 사유 원장과 적격/제외 분모를 등록해야 한다.

공통 투구 전 snapshot은 `pitcher, balls, strikes, batter_side, pitcher_hand, bases, outs, score_diff`만 허용한다. `bases`는 1루=1·2루=2·3루=4의 비트 합, `score_diff`는 수비 팀 점수−공격 팀 점수의 정수다. 합성 검증 범위는 카운트 0–3/0–2, 아웃 0–2, bases 0–7, 점수차 −100…100, 좌우 L/R, 양의 투수 ID다. 실제 희귀 상태·스위치 타자·결측 처리 규칙은 미등록이다. 실제 현재 공 구종/좌표/구속, 사후 결과와 `post_*`, 직전 구종/시퀀스 필드는 모두 거절한다. 시퀀스는 후속의 별도 비교다.

모든 비교 모델은 같은 snapshot을 받고 후보가 그 정규화된 JSON SHA256과 같은 가용시각을 선언해야 한다. 투수 빈도 기준선은 의도적으로 그중 투수 ID만 쓴다. 새 트리는 이 공통 정보 안에서만 비교할 예정이며 선수 추가 정보나 시퀀스를 몰래 더하지 않는다. snapshot 해시와 시각은 **선언의 일관성** 검사이지 실제 수신·자료 provenance의 증명은 아니다.

FIT 행의 구종 관측시각은 동결 cutoff보다 엄격히 앞서야 한다. query 명부의 각 생성시각은 `cutoff < generated_at < deadline`, 입력은 `input_available_at ≤ generated_at`을 만족하고 deadline은 holdout 시작 이상이어야 한다. cutoff는 holdout보다 앞서야 한다. 시각은 시간대가 필수다. 후보도 같은 학습/입력 시각 조건을 검사하며 마감 이상이면 `late`로 남긴다. 두 명부의 물리 키 중복·FIT/query 자기 키 겹침·**같은 경기의 역할 겹침**은 거절한다.

## 두 빈도 기준선: 정의와 지원 수

합성 smoothing 강도는 각 층 **α=4**로 설정 파일에 고정했다. 이 값은 합성 평가 점수로 선택하지 않았다. 실제 강도/계층 선택은 SELECT 역할과 예산을 별도 등록해야 한다. 합성 어휘 크기 K=4이고 빈도는 승인한 FIT 명부 전체에서만 센다.

- 전역 사전분포: `p_global(g) = (n_global(g) + α_global/K) / (N_global + α_global)`.
- 각 다음 층: `p_level(g) = (n_level(g) + α_level · p_parent(g)) / (N_level + α_level)`. N=0이면 부모 분포를 그대로 쓴다.
- 계층: 전역 → 투수 → 투수+볼/스트라이크 → 앞 조건+타자 좌우/투수 손 → 앞 조건+주자/아웃/점수차. 상황의 정확한 일치 빈도를 쓰며 구간을 결과에 맞춰 바꾸지 않는다.
- `pitcher_reference`: 전역→투수까지만 적용한 참고 기준선. `context_control`: 전체 계층을 적용한 주 비교 기준선. 표본 1개도 부모 쪽으로 평활하며, 최소 표본 문턱을 숨겨 적용하지 않는다.
- 각 query에 층별 N, 가장 깊은 비어 있지 않은 층, unseen pitcher와 context fallback 여부를 남긴다. 미관측 투수는 전역 분포로 돌아간다. 등록한 클래스가 FIT에 없더라도 양의 확률을 유지하며 미래 구종을 보고 투수 레퍼토리를 만들거나 마스킹하지 않는다.

빈도는 평가 전에 동결된다. 앞선 TEST 정답을 뒤 query의 빈도에 추가하지 않는다. 정답만 바꾸거나 FIT/query 행 순서를 섞어도 기준선 확률은 그대로다. 실시간 누적 재학습/prequential 평가는 이 계약 밖이다.

## 점수·전체 분모·쌍 비교

주 비교는 `hypothetical_candidate − context_control`이며 투수 빈도는 별도 참고다. 각 모델의 적격 전체 수·제시간 성공 수·coverage와 missing/error/abstain/late를 보고한다. 내장 기준선은 검증된 query 전부에 확률을 생성하는 합성 참조 경로이며 실제 모델의 100% 제공률을 보증하지 않는다. 후보의 누락/실패를 조용히 제거하지 않는다. 잘못된 정상 후보 확률·스키마·물리 키는 전체 실행을 거절한다.

NLL 주지표는 `−log(max(p_true, 1e−12))`, Brier는 클래스 제곱오차 합이다. 확률은 유한 숫자·[0,1]·합1(절대 오차 1e−6)을 검사하고 재정규화하지 않는다. 정답 확률0 건수를 따로 출력한다. top-1 정확도와 top-2 포함률은 보조이며 동률은 고정 라벨 순서다. 두 구종만 지원하는 문제에서 top-2는 자명해질 수 있으므로 주판정 지표로 쓰지 않는다. 현재 계산의 분모는 성공 예측 수이며 전체 제공률/상태별 분모를 바로 옆에 둔다.

쌍 점수는 동일한 **공통 제시간 성공 투구**만 사용한다. 자체 성공 표본 점수와 구분하며 coverage가 다르거나 불완전한 점수를 전체 모집단의 우위로 부르지 않는다. 물리 키 정렬 후 경기 전체를 재표집해 표집 총 투구 수로 나눈 평균 손실차를 계산한다(경기별 평균의 단순 평균 아님). 95% percentile 구간, seed20261029·1000회는 설정에 고정한다. 경기 2개 미만은 구간 null, 공통 성공0개면 점수/구간 미측정이다. 학습/선택 불확실성은 포함하지 않는다. 통계적 검정·승자·정책 가치 판정은 하지 않는다.

## 실행, 재현과 이번 완료 범위

```sh
PYTHONPATH=experiments/pitchmdp python -m pytest experiments/pitchmdp/tests/test_pitch_type_evaluation.py experiments/pitchmdp/tests/test_matrix_metrics.py -q
python experiments/pitchmdp/scripts/evaluate_pitch_type.py --config configs/EVAL-PITCH-TYPE-003.json --output /tmp/eval-pitch-type-synthetic.json
```

CLI는 합성 FIT 6구와 query 4구, 정답을 참조하지 않는 고정 후보 확률 `[.4,.3,.2,.1]`만 내부 생성한다. 외부 투구 payload 옵션이 없고 기존 출력은 덮어쓰지 않는다. 출력에는 모드/후보 종류/클래스 순서/cutoff·holdout, source·설정·정렬한 합성 입력 fingerprint, 층별 지원 수, 분모와 확률/점수/구간을 남긴다. 코드와 설정의 고정 검토 커밋도 함께 보존한다. Python 함수는 재사용 시 소진될 iterator를 거절하고 JSON 배열로만 검사 행을 받는다. 자기 선언 행을 받으며 실제 자료 진위를 판별하는 보안 장치가 아니다.

검사는 손계산 평활·손실/top-2/0확률, 희소/미관측 투수 fallback, query 정답에 불변인 두 기준선, FIT/query/후보 순서 불변, 동결 cutoff·경기 겹침·키·시각 경계, 금지 입력, 미지 라벨/확률 거절, 실패 분모·공통 정렬, 투구 가중 경기 차이, CLI 결정성·덮어쓰기/외부입력 거절을 다룬다. [독립 검토](REVIEW-EVAL-PITCH-TYPE-003.md)는 별도 작성자가 고정 커밋을 검토한다.

완료 대상으로 구현한 것은 **구종 합성 평가와 동결 빈도 기준선**이다. 실제 데이터/라벨 매핑·노출·FIT/SELECT/CAL/TEST 기간·입력 수신 근거·작은 트리 사양/pin·실용 개선/coverage 규칙·예산과 독립 등록 검토는 모두 후속 미완 조건이다. 설정의 실자료 게이트를 OPEN으로 바꾸거나 미완 값을 채우는 것만으로 실제 실행이 열리지 않는다. 의도한 목표 위치·유리/불리 선택 순위·OPE·팬 연구·UI·운영 변경도 범위 밖이다.
