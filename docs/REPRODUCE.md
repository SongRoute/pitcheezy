# 재현 방법

이 레포에서 다시 만들 수 있는 것은 **저장된 공별 예측 확률 → 9줄 표·구간 → 그림** 단계입니다. 학습은 다시 하지 않습니다.

## 1. 무엇이 필요한가

| 파일 | 내용 | 레포에 있나 |
|---|---|---|
| `results/final-mix-weights.json` | 최종 모델의 성분 3개, 섞는 비중, 온도 | 있음 |
| `data/eps/rows.csv` | 채점한 공 8,943개의 키(`game_pk:at_bat_number:pitch_number`)와 실제 구종 | 없음 (MLB 파생) |
| `data/eps/probs.npz` | 모델별 공별 확률(8,943 × 9) | 없음 (MLB 파생) |

`data/`는 `.gitignore`에 있습니다. 두 파일은 보관 중인 실행 산출물에서 `scripts/export_saved_probs.py`로 만듭니다.

- `--lit-final`: 기준 모델 5종 실행(EXP-LIT-BASE-001)의 `final/<모델>/eps-pred.npz`
- `--final-predict`: 최종 모델 실행(EXP-EXPL-1121-FINAL v4)의 `predict/<성분>/eps-pred.npz`
- `--cr`: 잔차 앙상블 1·2 채점 묶음(EXP-SVC-P9-003 CR)의 `candidate.jsonl`, `baselines.jsonl`, `receipt.json`
  - 잔차 앙상블 1·2의 확률, 9월 빈도표 확률, 실제 구종이 여기에 있습니다. `receipt.json`의 sha256으로 두 jsonl을 확인합니다.

```bash
python scripts/export_saved_probs.py --lit-final <...>/final --final-predict <...>/predict --cr <...>/CR --out data/eps
python scripts/rescore_table.py           # 표만: --no-bootstrap
python scripts/make_structure_figure.py   # 그림 0
python scripts/make_figures.py            # 그림 1~4, 표 1~3 (png + csv)
python scripts/make_onepage.py            # docs/onepage-16x9.{png,pdf}
```

글꼴은 Apple SD Gothic Neo → Noto Sans CJK → matplotlib 기본 순서로 고릅니다(`scripts/_fonts.py`). 리눅스에서는 `fonts-noto-cjk`가 있으면 됩니다.

## 2. 채점 규칙

원래 채점 코드(`experiments/lit_base/final2.py:score`, `run.py:score`)와 같은 규칙이고, 공통 함수는 `scripts/svc_p9/eval_prediction.py`를 그대로 씁니다.

- 2026 포스트시즌 31경기, 8,943구. 5차전(849799)은 데모용이라 뺐습니다.
- 모델이 예측을 내지 않은 275구는 9개 구종에 1/9씩 줍니다.
- 하한: q′ = 0.999·q + 0.001/9. NLL = 평균 −log q′[정답]. 1·3순위는 q′의 큰 순서.
- 정답: 실제 구종 코드를 9가지로 묶음(`src/pitcheezy/interfaces/pitch_types.py`).
- 최종 모델: 성분 3개 확률의 가중 산술평균 → log → 온도로 나눔 → softmax.
- 구간: 경기 단위 짝 부트스트랩 2,000회, seed 20261110, 2.5·97.5 백분위.

## 3. 이 레포를 만들 때 확인한 결과 (2026-11-25)

보관본 확률로 `rescore_table.py`를 돌린 결과, 9줄의 NLL·1순위·3순위가 README 표와 마지막 자리까지 같았고, 원래 채점 파일(`result.json`)의 값과는 차이가 0이었습니다. GRU 대비 차이와 구간(−0.01088 [−0.01436, −0.00748])도 같았습니다. `make_figures.py`가 만든 표 1~3의 csv는 원래 csv와 바이트 단위로 같았습니다.

## 4. 정답 라벨을 직접 만들기

실제 구종은 MLB 자료라 이 레포에 없습니다. 직접 받아 쓰세요.

1. pybaseball 등으로 2026 포스트시즌 Statcast 투구 자료를 받습니다(`pybaseball.statcast(start_dt, end_dt)`; 포스트시즌 날짜 범위는 받는 사람이 확인).
2. `python scripts/labels_from_statcast.py --statcast <파일> --out data/eps/labels-statcast.csv --check data/eps/rows.csv`
   - 키는 `game_pk:at_bat_number:pitch_number`, 정답은 `pitch_type`입니다.

원래 라벨은 Statcast가 아니라 MLB Stats API의 경기 종료 피드에서 읽었습니다(`allPlays[].playEvents[]` 중 `isPitch`, `details.type.code`; `at_bat_number = atBatIndex + 1`, `pitch_number`는 그 타석의 투구·자동 판정 이벤트 순번). 두 출처의 키와 구종이 이 31경기에서 모두 같은지는 **측정하지 않았습니다**.

모델 확률은 학습한 모델이 있어야 만들 수 있습니다. 라벨만으로는 표를 다시 만들 수 없습니다.

## 5. 처음부터 다시 학습하려면 (코드만 있음)

| 단계 | 코드 | 이 레포만으로 되나 |
|---|---|---|
| 원자료 받기 | pybaseball (Statcast 2023~2026) | 직접 받아야 함 |
| 9월까지 가공 자료(package) 만들기 | 없음 | 안 됨: 원래 레포 밖 파이프라인에서 만들었음 |
| 과거 3시즌 자료 투영 | `scripts/lit_base/build_hist.py` | 가공 자료가 있어야 함 |
| 평가 입력 내보내기 | `scripts/lit_base/export_eps.py` | 경기 피드가 있어야 함 |
| 기준 모델 5종 | `python -m experiments.lit_base.run driver` (`configs/EXP-LIT-BASE-001.yaml`) | 위 자료가 있어야 함 |
| 최종 모델 성분 학습 | `python -m experiments.lit_base.expl lane ...` (`configs/EXP-EXPL-1121-*.yaml`) | GRU 성분은 CUDA GPU에서 학습했고, 그때의 정확한 실행 명령은 남아 있지 않음 |
| 섞기 선정 | `python scripts/v4/run_final_v4.py select` | 후보 선정용 DEV 자료(`candidates_dir`)가 레포 밖에 있음 |
| 예측·채점 | `python scripts/v4/run_v4_rest.py` | 위 산출물이 있어야 함 |
| 잔차 앙상블 1·2 | 없음 | 학습 코드는 보관 중인 원래 레포에 있음 |

설정 파일의 경로는 환경변수로 바꿔 적었습니다(`experiments/lit_base/run.py`가 읽을 때 펼침).

- `PITCHEEZY_RUNS` — 실행 산출물 폴더
- `PITCHEEZY_JOBS` — 등록 문서·보조 자료 폴더
- `PITCHEEZY_SVC_RUNS` — 서비스 채점 산출물 폴더
- `PITCHEEZY_DATA_DIR` — 원자료 폴더
- `PITCHEEZY_REPO`, `PITCHEEZY_STORAGE`, `PITCHEEZY_ARTIFACT_ROOT` — 서비스 코드의 기본 경로
- `PITCHEEZY_HEAVY_LOCK` — (선택) 무거운 작업끼리 겹치지 않게 하는 잠금 파일

그래서 설정 파일 바이트가 원래 등록본과 다릅니다. 설정의 sha256을 확인하는 단계(`require_registered`, `same_config`)는 원래 산출물과 맞지 않습니다. 같은 seed라도 MPS·CUDA·CPU 사이에서 같은 숫자가 나온다는 보장은 없습니다.

## 6. 테스트 상태 (2026-11-25, 이 레포에서 실행)

- `pytest experiments/lit_base/test_lit_base.py tests/test_wc8.py`: 33 통과, 2 건너뜀, 7 실패.
  - 실패 7건은 모두 이 레포에 넣지 않은 것을 읽어서입니다: 경기 등록 정보 파일(MLB 파생, 4건), 옛 실험 모듈 `experiments.ws_seq`·`experiments.gpu14h_prepitch`(2건), 곁가지 실험 코드 `experiments/preq`(1건).
- `apps/observer`의 백엔드·화면 테스트는 돌리지 않았습니다. 실제 경기 자료로 만든 테스트 고정 파일을 뺐고, 원래 레포의 다른 모듈(`pitcheezy.seq_shadow` 등)이 없어서 상당수가 실패하거나 건너뛸 것으로 봅니다(미측정).
- 화면(`apps/observer/web`)은 `npm install` 뒤 빌드할 수 있게 팀원 스킨 플러그인을 설정에서 뺐지만, 이 레포에서 빌드해 보지는 않았습니다(미확인).
