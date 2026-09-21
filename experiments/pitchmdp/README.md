# PitchMDP 실제 데이터 파일럿

기존 pitcheezy의 라벨·캐시·RE24 모델과 분리한 연구용 구현이다. 짧은 투구 이력의 결과 예측,
타석 종료 전이와 구종×목표 위치 추천을 연구한다. 최신 결과는
**[HISTORY_BATTER_VALIDATION_RESULTS.md](HISTORY_BATTER_VALIDATION_RESULTS.md)**다.

이번 실행에서는 기존5구 모델60개를 재사용하고 신규0구 모델60개를 약32분에 학습했다.
6개 타자 표현 모두 두 연도에서0구와5구의 차이를 확인하지 못했다. 군집을 추가할
이득도 확인되지 않았다. 전체157개 테스트와 저장 예측의 점수·구간 재계산이 통과했다.

이전 표현·길이 결과는 **[REPRESENTATION_HISTORY_RESULTS.md](REPRESENTATION_HISTORY_RESULTS.md)**다.

주자·아웃 등 경기 상황을 유지하고, 타자 표현과 같은 타석의 이전1~5구를 각각 비교했다.
2024/2025·초기값5개·신규110개 MLP 실험에서 기존 성향+군집 구성은 좌·우타만 또는
개인 ID 방식보다 두 연도 모두 로그손실이 낮았다(비교군 보정95%구간).
최적 군집 수와5구 이력의 추가 이득은 확인되지 않았다. 전체148개 테스트와 독립 감사가 통과했다.
ID와 성향 방식은 과거 정보량·갱신 시점도 달라 순수 인코딩 효과로 단정할 수 없다.
이미 사용한 역사 데이터의 탐색적 비교이며, 실제 추천의 승률 개선을 입증한 것은 아니다.

앞선 [TEMPORAL_BLEND_RESULTS.md](TEMPORAL_BLEND_RESULTS.md)에서는 같은 상황 빈도 기준선과
혼합한 MLP와 Transformer가 모두 기준선보다 좋았지만, 두 신경망의 우열은 미확정이었다.

이전 후속 배치는 [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md), 첫 시퀀스는
[SEQUENCE_RESULT.md](SEQUENCE_RESULT.md), 최초 MLB/LAD 결과는
[REVISED_PILOT.md](REVISED_PILOT.md)와 [FIRST_RESULT.md](FIRST_RESULT.md)에 보존한다.
현재 실행·인수인계 상태는 [HANDOFF.md](HANDOFF.md)를 따른다.

## 실행

과거 투구0/5구와 타자 표현6종을 함께 검증하는 실행은 완료했다.
기존 모델60개를 재사용하고 신규60개를 순차 학습했다. [설계·M4 설정·재개 안내](docs/HISTORY_BATTER_VALIDATION.md).

```sh
.venv/bin/python experiments/pitchmdp/scripts/run_history_batter_validation.py --dry-run
caffeinate -i .venv/bin/python experiments/pitchmdp/scripts/run_history_batter_validation.py
```

첫 명령은 계획만 출력하고, 두 번째 명령은 완료 실행의 해시를 검사한 뒤 종료한다.
새 학습을 실행하려면 별도 SSD `--output`을 지정한다.

저장소 루트에서 기존 `.venv`를 사용한다. 설치나 업그레이드는 필요하지 않다.
T7 Shield가 마운트되어 있어야 하며, 원본·중간 데이터·모델·그림은 설정된 SSD에 둔다.

```sh
.venv/bin/python experiments/pitchmdp/scripts/preflight.py --prepare
.venv/bin/python experiments/pitchmdp/scripts/prepare_data.py
.venv/bin/python -m unittest discover -s experiments/pitchmdp/tests -v
```

현재 실행: MLB 전체의 충분한 선발 이닝, 타자 유형, 5구 이력+후보 공의 시퀀스.

```sh
.venv/bin/python experiments/pitchmdp/scripts/run_sequence_pilot.py
.venv/bin/python experiments/pitchmdp/scripts/audit_sequence_run.py --run "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/sequence-20260921T020823Z" --replay-binary
.venv/bin/python experiments/pitchmdp/scripts/recommend_sequence.py --run "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/sequence-20260921T020823Z"
.venv/bin/python experiments/pitchmdp/scripts/run_sequence_ablations.py --run "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/sequence-20260921T020823Z"
```

기본 추천은2구 시퀀스 분기 후 카운트 기준 모형으로 이어가는 근사다.
추가 실행기는 실제 타석 중간 이력을 사용하며, 같은 시퀀스 모형으로 타석 종료까지
고정 정책을 따라 모의 평가한다. 첫 행동의 선택/평가 난수를 분리했다.
전체 이력의 최적 타석 정책은 아직 구현하지 않았다.
설계는 [SEQUENCE_PROTOCOL.md](docs/SEQUENCE_PROTOCOL.md),
[SEQUENCE_ROLLOUT_PROTOCOL.md](docs/SEQUENCE_ROLLOUT_PROTOCOL.md)를 따른다.

후속 실행기(완료 산출물 위치와 수치는 최신 보고서 참조):

- `run_sequence_robustness.py`: 동일 표본·5초기값의 정보 제거 및 같은 규모 MLP 비교.
- `run_sequence_frequency_baselines.py`, `run_sequence_context_frequency.py`: 강한 빈도 기준선.
- `run_sequence_calibration.py`, `run_sequence_frequency_blend.py`: CAL-only 확률 혼합.
- `run_sequence_integration_reevaluation.py`: 같은30체크포인트의100표본 물리 적분 및400표본 진단.
- `run_sequence_context_blend100.py`: 기존 주자·아웃 기준선과100표본 앙상블의 CAL-only 혼합.
- `run_delivery_adaptation.py`: 현재 공 이전의 당일 구위만 사용하는 진단.
- `run_exact_delivery_diagnostic.py`: 모델·보정값을 고정한 전체 경험 분포 평균과 표본 적분 비교.
- `run_sequence_midpa.py`, `run_sequence_rollout.py`: 실제 이력·동일 시퀀스 타석 종료 모의 평가.

원 학습 소스·프로토콜은 완료 실행의 해시로 고정돼 있다. 후속 변경은 새 실행기에 두며
원 산출물을 덮어쓰지 않는다. 25/100/400표본 분포는 중첩되지 않으므로 수렴 증명이 아닌
민감도 비교다. 예측 개선을 추천 제구 가정의 검증으로 해석하지 않는다.

이전 MLB 유형 모델 재현:

```sh
.venv/bin/python experiments/pitchmdp/scripts/inspect_mlb_cohort.py
.venv/bin/python experiments/pitchmdp/scripts/run_first_result.py --config experiments/pitchmdp/configs/mlb_archetype_pilot.json
```

첫 실행 결과 경로:

```text
/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/first-20260921T013426Z
```

저장 모델로 실제 평가 타석의 첫 추천을 재생한다.

```sh
.venv/bin/python experiments/pitchmdp/scripts/recommend_one.py --run "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/first-20260921T013426Z"
```

카운트·직전 구종·상황을 바꾼 가상 질의도 가능하다. 타자와 경기일 이전 선수 통계는
저장 사례의 값을 유지하며, 변경 값은 출력의 `hypothetical_context_overrides`에 기록된다.

```sh
.venv/bin/python experiments/pitchmdp/scripts/recommend_one.py --run "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/first-20260921T013426Z" --balls 1 --strikes 2 --prev-pitch-type FF --bases 3 --outs 1
```

`bases`는 1루=1, 2루=2, 3루=4의 비트 합이다. 목표 좌표는 포수 관점 ft이다.
훈련된 모델을 임의의 현재 선수에게 적용할 수 있다는 의미는 아니다.

## 정보·평가 계약

- 2023–2025의 명시된 세 parquet 파일만 사용한다. 2026 파일은 읽지 않는다.
- 선정 및 모델 학습은 2023-05-15–2025-04-30, 보정/epoch 선택은 2025년 5–6월,
  DEV는 2025년 7월 이후로 분리한다. 프로젝트 전체의 완전 미사용 DEV가 아니다.
- 타자 누적 통계는 경기일보다 이전 날짜만 사용한다. 직전 구종은 타석 경계에서 초기화한다.
- `model.py`는 실제 도달 위치에 조건부인 결과 모델이다. 투구 전 평가에서는 학습 자료의
  위치 분포를 적분하고, 추천에서는 명시한 Gaussian 제구 가정으로 적분한다.
  실제 현재 위치·구속, 타구 정보, 사후 점수, 공급자 WE, 다음 경기 정보는 투구 전 입력이 아니다.
- `game.py`는 실제 경기 승패로 W를 학습하고, 타석 종료 후 주자·아웃·점수를 갱신한다.
  반이닝 전환 후에도 평가하는 수비 팀의 기준을 유지한다.
- `planner.py`는 카운트×직전 구종을 풀며, 2스트라이크 파울 때 직전 구종도 갱신한다.
  실제 미래 투구를 분기에 대입하지 않는다.
- `sequence_model.py`는 물리 시퀀스와 경기·타자 유형 입력을 받는다.
  `sequence_delivery.py`는 학습 자료의 공동 물리 분포를 적분하며, 투구 전 예측에는
  실제 현재 공의 물리량을 넣지 않는다. `sequence_planner.py`는 후보마다 이력과
  카운트를 갱신하는 유한 깊이 분기와 명시적인 이후 가치 함수를 사용한다.
- 관측 예측 성능과 동일 모델 안에서 최적화한 정책 가치를 분리한다.
  실제 목표 라벨이나 독립 정책 평가가 없으므로 실제 승률 개선을 주장하지 않는다.

## 구현과 재현 범위

`data.py` / `model.py` / `game.py` / `planner.py` / `recommend.py`가 각각 데이터,
예측, 경기 전이, 타석 계산, 목표 질의를 담당한다. 각 run에 설정·데이터 해시·소스 스냅샷·
체크포인트·학습 기록·예측 지표·추천 JSON을 보관한다.

`--resume`는 동일 소스·설정·데이터에서 완료된 체크포인트를 재사용한다.
학습 중간 optimizer 상태 복구, 작업 큐, W&B 대시보드 연결은 아직 구현하지 않았다.
학습은 순차 실행한다. 초기 실행의 정확한 소스는 해당 run의 `source/`에 보존되어 있다.

현재 모델은 오류·특수 사건, 중간 주루 변화, 경기 종료 타석 등 일부 관측 타석을
품질/지원 조건으로 제외한다. 이 결과 기반 제외는 선택 편향을 만들 수 있으며,
이 조건부 파일럿을 전체 경기의 정책 효과로 일반화하지 않는다.
