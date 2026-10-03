# CUDA 데이터량·투수 정보·조합 가족 (DRAFT) — 2026-10-04

작성: Claude Opus 5.5 (구현자, 자기 점검만 했고 독립 검토는 아님). 기준 커밋 `7949f5f`. 실제 fit·데이터·SSH·GPU 실행 0회.
config `configs/DRAFT-CUDA-DATA-COMBO-RUN-20261004.json` (status DRAFT, `execution.enabled false`, 최종 ID 없음, 제안 P27; 날짜·마감은 root가 정함).

## 무엇
새로 학습하는 대조군 1개와 팔 14개를 시드 3개(0·1·2)로 돌려 fit 45회(모두 priority 1). 구조(flatten·residual·periodic) × workload(없음·counts·block)가 3×3 완전 요인 설계다: 대조군 = flatten/없음, counts·block·residual·periodic = DRAFT-CUDA-DIRECT 기본 팔 정의 그대로, 나머지 4칸 = 조합 팔. 데이터량·투수 정보 팔은 대조군에서 한 가지만 바꾼다. **조합 팔은 대조군 대비 구조와 정보를 동시에 바꾼다**(이전 문서의 "한 가지만" 주장은 틀렸다). 그래서 조합 팔은 config에 고정한 `factorial_contrasts` 8개로만 해석한다: 같은 구조의 workload 없는 팔 대비 = workload 효과, 같은 workload의 flatten 팔 대비 = 구조 효과.
- **data25 / data50 / data75**: TRAIN 행만 고정된 중첩 해시 부분집합으로 줄인다. `u = blake2b-64(key=salt, int64 keys 행)/2^64`이고 `u < fraction`인 행만 남긴다. 모든 비율에 같은 u를 쓰므로 부분집합이 중첩되고, boolean 마스크라 원래 행 순서를 유지한다. salt와 비율은 config의 `train_subset`이 정한다. 선택은 keys와 salt에만 의존하고 모델 시드·y·DEV·결과값과 무관하다. earlystop·temperature·blend·dev·mlb_dev 행과 동결된 TRAIN 변환·프로필은 그대로 둔다.
- **no_pitcher_profile (ctx 28:51) / no_pitcher_support (ctx 51) / no_pitcher_all (ctx 28:52)**: 기존 `MaskedFlattenModel`로 이름 붙은 열을 0으로 만든다. 이름은 export manifest lineage로 검증한 52개 이름에서 나온다. 타자·경기 상황·투구 이력은 유지한다. 새 클러스터링을 시험하는 것이 아니라, 이미 있는 투수 정보에 모델이 얼마나 기대는지를 본다.
- **residual_counts / residual_block**: 기존 `DirectResidualModel`에 workload 첫 열 또는 네 열 전부를 더한다(ctx 53 / 56).
- **periodic_counts / periodic_block**: 기존 `DirectPeriodicModel`에 workload를 더한다. 연속 열 목록은 기본 목록(11..51, left_share 32 제외)에 52 또는 52..55를 이어 붙인 것이다. `rest_missing` 플래그도 수치로 임베딩한다. 정확한 인덱스는 레지스트리와 config에 고정했다.

## 코드
- `pitchmdp/direct_combo_registry.py`(torch 없음): 팔 레지스트리, 부분집합 규칙과 출처 기록, 정확한 마스크와 periodic 인덱스.
- `pitchmdp/direct_combo_model.py`: 새 네트워크나 학습 루프 없이 기존 MatrixModel·MaskedFlattenModel·DirectResidualModel·DirectPeriodicModel을 재사용한다. `load_combo`는 종류·마스크·연속 열·컨텍스트 폭이 다른 아카이브를 거부한다.
- `run_cuda_direct.py`: `"combo": true` 팔은 레지스트리 항목과 정확히 같아야 한다. `train_subset`은 combo 데이터 팔에만, TRAIN에만 적용되고 base·frontier 팔에는 쓸 수 없다. `fit.json`에 `train_subset`(fraction·salt·n_source·n_selected·원본/선택 행 sha·마스크 sha)을 남긴다. MODEL_SOURCES에 combo 두 모듈을 추가했으므로 source hash가 바뀌어 새 대조군이 필요하다. 기존 base·frontier 경로, CUDA 가드, export 고정, 2026 제외, 큐는 바꾸지 않았다.
- `score_cuda_direct.py`: `train_subset`이 있는 config에서는 데이터 팔이 기록한 부분집합이 config 규칙과 export TRAIN `rows_sha256`에 맞는지 확인하고, 다른 팔은 전체 TRAIN으로 학습했는지 확인한다. 이전 가족(규칙 없음)은 예전과 똑같이 채점하고, 예전 source map은 여전히 `--source-manifest`로 쓸 수 있다. 결과는 기술 비교만 하며 결정·순위·자동 선택은 없다. `comparison.factorial_contrasts`가 있으면 예측을 읽기 전에 이름 유일·팔 존재·서로 다름·정확히 한 요인만 다름(데이터·투수 팔 제외)을 확인하고, 대조군 비교와 같은 helper(`paired_game_comparison`, 경기 단위 부트스트랩)로 같은 시드·앙상블 짝 비교(후보 − 기준)를 `factorial_contrasts`에 기록한다. 키가 없는 이전 가족 결과 형식은 그대로다.

## 검사 (실제 실행)
- `tests/test_cuda_data_combo.py` 24개 통과. config/마스크/인덱스 정확성, 엄격성 17가지, torch 없이 config_check, 부분집합의 중첩·순서·키/salt 의존·중복키 거부, 데이터 팔 CPU smoke(TRAIN만 줄고 시드 0·1이 같은 행, earlystop·dev·mlb_dev 불변), 투수 마스크의 정확성·불변성·아카이브 왕복, workload 조합의 입력 배치·체크포인트 왕복·쌍둥이 아카이브 거부, scorer 출처 확인 6가지 변조 거부, 예전 source map 수용을 다룬다.
- 집중 검사 `test_cuda_direct.py + test_cuda_frontier_run.py + test_cuda_data_combo.py` 90개 통과. 전체 pitchmdp 결과는 커밋 메시지에 적는다.

## 한계
- 데이터량 팔은 **최적화에 쓰는 예시 수만** 바꾼다. 컨텍스트의 과거 타자·투수 프로필 요약은 전체 TRAIN으로 만든 것을 그대로 쓰므로 전체 정보량을 줄인 것이 아니다. 30 epoch 고정이라 행이 적으면 최적화 step도 적어진다. 따라서 compute-matched 비교가 아니다.
- 마스크된 입력은 0을 넣는다(TRAIN 표준화 기준으로 평균 근처 값). 투수 정보를 완전히 지웠다는 주장은 아니다. 모델 재학습 없이 0을 넣는 방식도 아니고, 각 팔을 따로 다시 학습한다.
- periodic에서 `rest_missing`(0/1 플래그)을 연속값으로 임베딩한 것은 지시에 따른 선택이다. 기본 periodic이 `left_share`를 연속 열에서 뺀 것과 일관되지 않을 수 있다.
- dev·mlb_dev는 이미 반복 노출된 DEV라 결과는 기술 비교로만 쓴다. 6월 블렌드·G0 혼합·정책은 범위 밖이다.
- 합성 CPU 검사만 했다. 실제 CUDA smoke, 실제 export 행 수, 실행 시간은 **미측정**이다. 해시 부분집합 계산은 맥미니에서 합성 100만 행에 0.46초가 걸렸다(1회 측정). newsrv에서 실제 TRAIN으로 잰 시간은 미측정이다.

## 독립 검토 (Claude Opus 5.5, 같은 날)
- 검토 범위: 부분집합 통계 의미(키·salt만 의존, 중첩, 순서 유지, TRAIN만), 마스크 인덱스(28:51·51·28:52), periodic 연속 인덱스(기본 map + 52 / 52..55), 모델 팩토리·로드 거부, scorer 출처 확인. 실제 배열·GPU·SSH 0회.
- 중대 버그는 찾지 못했다. 바꾼 것: 위 3×3 완성(기본 팔 4개 추가, fit 33→45, 전부 priority 1), 요인 대비 8개와 검증, "한 가지 변경" 문구 정정. scorer는 메모리 상한을 위해 대비 계산 때 예측을 다시 읽는다.
- 추가 검사 12개(대비 설정 엄격성 8, 예측 읽기 전 거부, 대비 부호 = 대조군 차이의 차, 대비 없는 가족 형식 불변, 불완전 가족 거부). `test_cuda_data_combo.py` 36개 + `test_cuda_direct.py`·`test_cuda_frontier_run.py` 66개 통과(맥미니). 전체 스위트는 돌리지 않았다.
- 해석 주의: 9칸이 같은 시드·같은 export지만 GPU 비결정성은 미측정이다. 상호작용(2×2 차의 차)은 따로 보고하지 않는다; 두 대비의 차로 계산은 가능하지만 판정에 쓰지 않는다.
