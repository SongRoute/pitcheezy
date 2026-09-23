# B — 동결 추천 경계와 작은 한 축 비교

2026-09-23. B1/B3 경계와 B2 단일 탐색 실험 완료. B4 실제 CV 기반 제구 학습은 입력이 없어 미수행이다.

## 결과

- `Recommender.evaluate_pre_pitch`는 동일 동결 모델의 지원 구종×구역 전체 확률·후보 Q·기준 정책·지원도·action 대응을 명시적으로 돌려준다. 기존 `recommend` 상위 3개 순위/값·캐시 형식은 유지한다. 실제 C0 예시에서 39개 지원 후보, 확률합 1, 저장 공개 추천 동일, 상위 후보 값/순위가 예시와 일치했다. 실제/미래 필드 변조와 미지원 처리 검사 통과.
- 현재 실행 분포를 `ExecutionDistribution` 인터페이스로 분리했다. 구현은 기존 관측 TRAIN delivery Gaussian 커널뿐이다. 의도/제구 모델이라고 주장하지 않는다.
- `EXP-B-PAHISTORY-001`: 동일 4월 작은 TRAIN 145 지원 PA, 7월 DEV 151 지원 PA·563구. CountBaseline에 **같은 PA 직전 구종 family** 셀 하나를 추가하고 count-only 셀 분포로 50 pseudo-count 평활했다. 첫 공 `NONE`; 훈련과 평가 모두 이전 공만 사용한다. A의 선정·지원 마스크와 저장 `april_count` 예측은 정확히 일치했다.

| DEV 범위 | count-only NLL | +직전 구종 family NLL | paired 차이 (+가 악화) |
|---|---:|---:|---:|
| S0, 2경기·211구 | 1.476064 | 1.479712 | +.003648 [−.004661, +.011572] |
| S1, 6경기·563구 | 1.589641 | 1.602247 | +.012606 [+.003048, +.025801] |

경기 단위 paired bootstrap 1,000회, seed 20260923. 작은 노출된 개발 분할에서 이 피처의 예측 이점이 없었다. 이 결과로 서비스 채택·정책 WE 개선·인과 기여도를 주장하지 않는다. 2026/최종 CV를 열지 않았다. 추가 피처·sample·hyperparameter를 결과 뒤 조정하지 않았다.

## 산출물과 검증

- 경계: `apps/observer/backend/observer_app/recommendation_adapter.py`, `recommender.py`; 사양 `docs/contracts/B-recommendation-v1.md`.
- 평가 고정 config `configs/EXP-B-PAHISTORY-001.json`, 코드 `scripts/b_pa_history_eval.py`는 점수 계산 전에 `1113e60`으로 커밋했다. 수치·지원 분모·코드/config/A 부모 hash는 `results/EXP-B-PAHISTORY-001/results.json`.
- SSD `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/B-PAHISTORY-001/`: `predictions.npz`에는 `pitch_keys`, `game_pk`, `y`, `count_only`, `count_plus_prior_family`, `previous_family`; `results.json` 복사본. A의 `predictions.npz`와 동일 키/라벨/대조군 확률을 검증했다. A 결과를 덮어쓰지 않았다.
- `scripts/b_archive_fit.py`는 점수 후 동일 고정 설정으로 작은 두 count table을 결정적 재적합해 `fitted_count_tables.pkl`에 저장했다. 복원한 두 모델의 DEV 확률은 이미 저장된 `predictions.npz`와 배열 단위 정확히 일치한다. 해시와 범위는 `results/EXP-B-PAHISTORY-001/checkpoint_manifest.json`; 이 별도 보관 단계에서 새 점수나 튜닝을 만들지 않았다.
- `PYTHONPATH=.:apps/observer/backend .venv-observer-standalone/bin/python -m pytest tests/test_b_recommendation.py -q`: 5 passed. 합성 실행 분포 입출력 검사도 포함한다. 별도 standalone 실모델 smoke도 ready/39 후보/확률합/공개 추천 일치를 확인했다.

실험 실행시간·RSS는 당시 계측하지 않아 미측정이다. 결과 파일의 script/config SHA256은 점수 시 사용 파일과 일치한다.

남은 B4는 실제 의도 라벨, 당시 입력과 분할, 명시적 목표→도달 위치 분포를 받은 뒤의 별도 연구다. 이번 고정 관측 커널과 섞지 않는다.
