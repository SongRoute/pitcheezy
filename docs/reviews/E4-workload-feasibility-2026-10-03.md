# E4 workload 입력 실행 가능성 검토 (2026-10-03, Opus 5.5 작업자)

범위: `docs/reviews/PM-research-frontier-2026-10-03.md` E4(MX-F6)의 선행조건 점검만 한다. 실험, fit, 추론, 실제 행 스캔, 2026 열람, newsrv 사용은 하지 않았다. **성능 이득은 추정하지 않는다(미측정).** graphify 그래프(`graphify-out/`)는 이 worktree에 없다.

## 결론

- 세 후보 모두 2023·2024·2025 원본에 **열이 있거나 기존 키로 다시 계산할 수 있다.** 다만 TRAIN 행 결측률과 재계산 일치율은 **미측정**이다(아래 감사 스크립트가 아직 실행 전).
- 추천 안전 부분집합: **(a) 경기 내 이전 투구 수**와 **(b) 재계산한 타순 회차**는 우리 정렬 키만으로 만들고, **(c) 휴식일**은 Savant 값을 그대로 쓰지 않고 우리 원본의 이전 날짜로 재계산한다. Savant의 `n_thruorder_pitcher`·`pitcher_days_since_prev_game`은 일치율 점검용으로만 쓴다.
- **사용 금지:** `pitcher_days_until_next_game`, `batter_days_until_next_game`. 미래 정보이며 세 시즌 파일 모두에 들어 있다.

## 근거 1: 공식 문서 (baseballsavant.mlb.com/csv-docs, 2026-10-03 확인)

| 열 | 공식 설명 | 투구 전 값인가 |
|---|---|---|
| `n_thruorder_pitcher` | "pitcher times through the order" | 의미상 타석 시작 때 알 수 있다. 다만 계산 규칙(타순 칸 기준인지 상대 타자 수 기준인지, 타석 도중 교체 처리)은 문서에 없다 |
| `pitcher_days_since_prev_game` | "number of days since pitcher's last game" | 의미상 경기 전 값이다. 다만 "last game"에 어떤 경기 유형(시범·포스트시즌·더블헤더)이 들어가는지, 사후에 다시 계산되는지는 문서에 없다 |
| `pitcher_days_until_next_game` | "number of days until the pitcher's next game" | **아니다. 미래 값이므로 누수다** |
| `at_bat_number` / `pitch_number` | "Plate appearance number of the game" / "Total pitch number of the plate appearance" | 키. 현재 공의 키는 투구 전에 정해진다 |

문서에 갱신 시점이나 "투구 전 상태" 여부를 밝힌 문장은 없다. 그래서 Savant 값을 그대로 투구 전 값으로 가정하지 않는다.

## 근거 2: 스키마 (parquet footer 메타데이터만 읽음, 행은 읽지 않음)

원본: `$PITCHEEZY_DATA_DIR/raw/d20260911-s2325/statcast_{2023,2024,2025}.parquet`. 세 파일 모두 119열, row group 1개.

| 파일 | 행 수 | `n_thruorder_pitcher` | `pitcher_days_since_prev_game` null | `pitcher_days_until_next_game` |
|---|---|---|---|---|
| 2023 | 720,684 | int64, null 0 | 34,865 (4.84%) | 있음 (null 28,083) |
| 2024 | 711,899 | int64, null 0 | 35,010 (4.92%) | 있음 (null 27,910) |
| 2025 | 712,528 | int64, null 0 | 34,442 (4.83%) | 있음 (null 27,481) |

주의: 위 null 수는 **파일 전체** footer 통계다. TRAIN만 센 값이 아니며, 2025 파일에는 calibration·DEV 기간 행도 들어 있다. 결과나 라벨을 담은 값은 아니지만, E4 선행조건인 "TRAIN 행에서만" 기준은 아직 충족하지 못했다. 키 `game_pk`, `at_bat_number`, `pitch_number`, `pitcher`, `game_date`는 세 파일 모두 int64/timestamp이고 null 0이다. 정수 열 정책은 `src/pitcheezy/data/statcast_fetch.py:74-91`(`INT64_COLUMNS`, 금지 열인 `*_until_next_game`도 여기서 저장됨)에 있다.

## 근거 3: 현재 맥락 구성 코드와 어디에 들어갈지

- 연구 준비 단계 `experiments/pitchmdp/pitchmdp/data.py:22-30`의 `RAW_COLUMNS`에는 workload 열이 **없다**. 그래서 E4를 하려면 새 prepare 해시가 필요하다(프런티어 문서에 적힌 대로).
- 정렬 기준은 `data.py:326`(`["game_date", *KEY]`), 키 중복 금지는 `data.py:327`, 분할 경계(history < 2023-05-15 ≤ train ≤ 2025-04-30)는 `data.py:69-75`에 있다.
- 숫자 맥락 블록은 `experiments/pitchmdp/pitchmdp/model.py:93-110`이다. G15 직접 head 러너(`run_ml_direct_head*`)는 **이 브랜치에 없어서** 정확히 어느 줄에 넣을지는 확인하지 못했다.
- "같은 경기에서 이전 공만" 쓰는 기존 사례는 `apps/observer/backend/observer_app/context.py:28-34`(`ObservedContext.before`)에 있다. 엄격한 `<` 비교를 쓴다. `experiments/pitchmdp/pitchmdp/sequence_data.py:162`의 `build_history_indices`는 필터링 전 전체 프레임의 행 위치를 쓴다.

## 설계 (이번에 추가한 코드)

`experiments/pitchmdp/pitchmdp/workload_features.py`의 `rebuild()`는 `(game_date, game_pk, at_bat_number, pitch_number)` 순서로 정렬한 뒤 다음 세 값을 만든다.

1. `prior_pitches_game`: 같은 `(game_pk, pitcher)` 안에서 **현재 공을 빼고** 앞선 행 수(`cumcount`).
2. `rebuilt_thruorder`: 같은 경기에서 이 투수가 이번 타석 전에 상대한 타석 수를 기준으로 `// 9 + 1`.
3. `rebuilt_days_since`: 이 투수가 **더 이른 날짜**에 등판한 마지막 날과의 일수. 불러온 원본에 이전 날짜가 없으면 결측이다.

안전장치: `*_until_next_game` 열이 들어오면 거부하고, 키 결측·중복도 거부한다. `audit()`은 2025-04-30 이후 행이 하나라도 있으면 거부한다. 2023-05-15 이전 행은 휴식일 이력 계산에만 쓰고 통계에는 넣지 않는다.

`tests/test_workload_features.py`의 합성 테스트 6개가 통과했다. 확인한 내용: 현재 공 제외, 이후 행을 바꿔도 값 불변, 이전 날짜만 사용, 9타자마다 회차 증가, 금지 열·TRAIN 이후 행 거부, TRAIN 행만 집계. `scripts/audit_workload_columns.py`는 키·날짜·투수·Savant 두 열만 읽는다. `game_date <= 2025-04-30` 조건을 읽는 시점에 걸고, 결과·물리·점수 열은 열지 않는다. 합성 parquet로 실행 경로만 확인했고 **실제 데이터로는 실행하지 않았다.**

## 누수·운영 재구성 위험

1. **미래 열:** `*_days_until_next_game`이 원본 세 파일 모두에 있다. 열 이름을 고르는 실수 하나로 누수가 생기므로, 코드에서 거부하도록 했다.
2. **Savant 휴식일 정의 불명:** 시범·포스트시즌·더블헤더가 포함되는지 모르고, 2023 첫 등판은 2022 경기를 볼 수 있다. 우리 재계산과 정의가 다르면 학습과 서비스 입력이 어긋난다. 서비스에서는 일정·박스스코어로 재구성해야 한다(`configs/EXP-C-ROSTER-001.json`의 7일 전 일정·완료 박스스코어 규칙과 같은 계열). 이 경로가 학습 정의와 같은지는 미검증이다.
3. **같은 날 더블헤더:** 재계산은 날짜 단위라서 두 번째 경기도 "이전 날짜"까지의 일수가 된다. Savant는 0일로 줄 수도 있다. 재계산에서는 `game_pk`의 시간 순서를 가정하지 않았다.
4. **이전 투구 수 정의:** Statcast 행에 자동 볼·스트라이크(피치클락) 같은 비투구 행이 섞여 있는지 확인하지 않았다. 실시간 피드(`isPitch`)와 같은 정의인지도 맞춰야 한다. 그리고 **코호트·적격성 필터 전 전체 원본**에서 세야 한다. 필터 뒤에 세면 값이 작아진다.
5. **타석 도중 교체:** 바뀐 투수에게는 그 타석이 새 타석으로 잡힌다. Savant 회차와 다를 수 있어 일치율로 확인해야 한다.
6. **시즌 경계:** 비시즌 휴식일(100일 이상)은 큰 값이 된다. 상한·구간화 규칙은 결과를 보기 전에 config로 고정해야 한다.
7. 이 값들은 관측 workload일 뿐이다. "피로"라고 부르지 않는다(ML_EXPERIMENT_DESIGN F4).

## 다음 명령 (Root PM 승인 뒤, P20 실행이 끝난 뒤에만)

```
cd experiments/pitchmdp
/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_workload_features.py -q
/Users/song/Projects/pitcheezy/.venv/bin/python scripts/audit_workload_columns.py \
  --raw-root "$PITCHEEZY_DATA_DIR/raw/d20260911-s2325" \
  --out "$PITCHEEZY_RUNS_DIR/_audit/E4-workload-train-columns.json"
```

통과 기준은 결과를 보기 전에 PM이 정해야 한다(제안: 재계산 결측률과 Savant 일치율을 보고만 하고, 입력은 재계산 값으로 한정). 실제 실행 장치는 기존 감사 러너처럼 heavy lock과 wall ledger로 감쌀지 PM이 결정한다. 그 뒤 E4는 새 prepare 해시를 등록하고, 서비스 재구성 경로(일정 API) 검증을 별도로 진행한다.
