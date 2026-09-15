# 인터페이스 스펙 v1

기준일 2026-09-08 (D4). 코드화 `src/pitcheezy/interfaces/`, 검증 `tests/interfaces/`.
변경하면 이 파일의 변경 이력에 한 줄, `docs/decisions.md`에 한 줄, 계약 테스트를 같이 고친다.

## 공통 자산 0 — RE24 테이블
- 생산 data / 소비 transition·policy·ope·decomp. 버전 = 시즌 창 + 계산 커밋. 한 실험 안에서 모두 같은 버전
- `RE24[base_out_id]` — 24상태 기대득점. `base_out_id` = 아웃 × 8 + 주자 비트마스크(1루=1, 2루=2, 3루=4) → 0..23. **전이 텐서·Q·RE24 모두 이 id**
- **Phase 0: `dRE24[outcome_id, base_out_id]`** — 종결 결과 8종 × 24상태의 학습 창 평균 ΔRE24 (스칼라). 인플레이 아웃 안의 병살·희생플라이는 리그 평균으로 뭉갬 (ASM-7)
- Phase 1: 타구 특성 조건부 분포 P(ΔRE24 | 타구 특성, base_out)로 교체

## 자산 1 — 전이 확률 텐서 (transition → policy)

### 축 `P[투수, state, action, outcome]` — 4차원 dense
- 투수 `pitcher_idx` 0..P−1 ← `pitchers.parquet` (MLBAM id, 이름, 학습 창 투구 수)
- 상태 `state_id` = (count_id × 24 + base_out_id) × K + cluster_id, S = 288 × K
  - `count_id` = 볼 × 3 + 스트라이크 (0..11)
  - `base_out_id` 위와 동일
  - `cluster_id` 0..K−1, K ≤ 8, **좌우 층화 필수** (cluster = 타자 좌우 × 성향 군집). 배정 파일 `batter_clusters_{버전}.parquet`
  - 맥락 피처 없음. 추가 시 `[투수, state, context, action, outcome]`로 state 뒤에 삽입 (변경 이력 + decisions 한 줄). Phase 1 "시퀀스 맥락" 축이 이 자리를 쓴다
- 행동 `action_id` = pitch_id × 25 + loc_id, A = 225
  - `pitch_id` 0..8: FF{FF,FA} / SI / FC / SL{SL,SV} / ST / CU{CU,KC,CS} / CH / FS{FS,FO} / OT{KN,SC,EP}. PO·IN·UN·null은 행동에서 제외 (카운트만 진행, 비율은 data/versions.md)
  - `loc_id` = z행 × 5 + x열 (0..24). x: `plate_x` 경계 −0.83 / −0.28 / +0.28 / +0.83 ft, 바깥 클립. z: `(plate_z − sz_bot) / (sz_top − sz_bot)` 경계 0 / ⅓ / ⅔ / 1, 바깥 클립. 포수 시점 절대 좌표, 좌우 반전 없음
- 결과 `outcome_id` 0..10: 비종결 {볼, 스트라이크, 파울} + 종결 {K, BB, HBP, 1B, 2B, 3B, HR, 인플레이 아웃}. 다음 카운트는 `outcomes.parquet`의 `count_rule`로 결정 (볼→b+1, 스트라이크→s+1, 파울→s<2면 s+1 / s=2면 유지)
  - Statcast 매핑: 볼 ← ball, blocked_ball / 스트라이크 ← called_strike, swinging_strike, swinging_strike_blocked, foul_tip(s=2), missed_bunt / 파울 ← foul, foul_bunt(s<2) / 종결 ← events: K(strikeout, strikeout_double_play), BB(walk), HBP, 안타 4종, 인플레이 아웃(나머지 전부: field_out, force_out, double_play, sac_fly, sac_bunt, fielders_choice, triple_play 등)
  - Phase 1 확장: 인플레이 5종 → 타구 특성 셀, 투구 이벤트(폭투·포일) 추가. 이 축만 교체

### 값
- float32. 계산(VI·OPE)은 float64
- `valid` 행은 |Σ − 1| ≤ 1e-5. 실패 시 재정규화 없이 검증 실패

### 마스크 3층 (동봉 배열)

| 층 | 대상 | 근거 | 값 |
|---|---|---|---|
| 규칙 | outcome | 카운트 규칙 (3볼에 "볼", 2스트에 "스트라이크" 불가) | 해당 셀 0, 행 합 1 |
| 레퍼토리 | (투수, pitch) | 학습 창 투구 수 < `repertoire_min_pitches` (기본 100) | 행 전체 0, `valid=False`, 행 합 0 |
| 지원 카운트 | (투수, state, action) | 관측 수 | `n_obs` 동봉, 마스킹 없음 |

### 저장 `runs/{실험ID}/s{seed}/transition/` (홀드아웃용은 `transition_holdout/`)
```
P.npy          float32 [P, S, A, O]
valid.npy      bool    [P, S, A]
n_obs.npy      int32   [P, S, A]
states.parquet          state_id ↔ count_id, base_out_id, cluster_id
outcomes.parquet        outcome_id, terminal, count_rule, statcast 매핑
pitchers.parquet        pitcher_idx ↔ MLBAM id, 이름, 학습 창 투구 수
meta.json
sha256.txt              위 파일 전부
```
`meta.json` 필수 키: spec_version, data_version, season_window, holdout_split ("season:2025" / "none"), model_arch, seed, train_commit, K, cluster_file_version, pitch_type_map_version, repertoire_min_pitches, row_sum_tol, holdout_nll, holdout_ece, holdout_ece_hr, excluded_pitchers

### 홀드아웃 (D3 구체화)
- 2023–24 학습 → 2025 홀드아웃. 실험당 텐서 2개 (홀드아웃용 / 2023–25 전체 학습), 같은 config·시드
- NLL = `valid` 행의 홀드아웃 투구 평균 −log P(관측 결과 | s, a, 투수)
- ECE = 결과별 15-bin reliability 평균 + HR 단독 병기
- 2025에만 있는 투수는 집계 제외·목록 기록 (`excluded_pitchers`)
- 2026은 어느 텐서도 건드리지 않음

### 검증 계약 (`tests/interfaces`)
형상·dtype / NaN 없음 / `valid` 행 합 1±1e-5 / `valid=False` 행 합 0 / 규칙 마스크 셀 0 / 룩업 테이블 크기 = 축 길이 / meta 필수 키 / sha256 일치

## 자산 2 — 가치함수 (policy → ope·decomp)
- `Q.npy [P, S, A]` float32, 단위 RE24. `V.npy [P, S]` = `valid` 행동 위 max. `policy.npy [P, S, A]` = 완화 후 분포 (softmax 온도 / top-k, 파라미터는 meta)
- 조회는 `interfaces/value.py::lookup(Q, pitcher_idx, state_id, pitch_id, plate_x, z_norm, mode)`로만. Phase 0 `mode="snap"`, `"bilinear"`는 자리만. **V(의도)·V(실제)는 같은 mode** (같은 셀이면 실책 0)
- 저장 `runs/{실험ID}/s{seed}/value/` + `meta.json` (전이 텐서 경로·sha256, RE24·dRE24 버전, 종결 보상 축, 인플레이 보상, γ=1, 완화 파라미터, lookup_mode)
- 검증 계약: 정합성 Σ 성분 = V(추천) − ΔRE24(실제), 같은 RE24 버전. 진단: 기준점이 셀 경계 ±10% 안에 놓인 비율 (Phase 2 분해에서 사용)

## 베이스라인 재구현이 쓰는 축 (D6)
- B1 (SmartPitch류): 같은 텐서에 K=1, base_out 축을 붕괴(모든 base_out_id를 0으로 매핑). config 옵션 `state.collapse_base_out: true`, `state.K: 1`
- B2 (Takamido류): 전이 모델 아키텍처만 교체(Transformer 시퀀스 인코더 → outcome 11-way). 텐서 축·저장 포맷은 동일. 시퀀스 맥락은 context 축으로 삽입

## 변경 이력
- v1 (2026-09-08, D4): 슬롯 전부 채움. 공통 자산 0에 Phase 0 `dRE24` 스칼라표. 군집 좌우 층화 제약. 마스크 3층·동봉 배열. 홀드아웃 분할 구체화. `lookup` 인터페이스
- v1 (2026-09-15, D5·D6): 노션에서 레포로 이관. 생산·소비 주체를 트랙에서 모듈명으로. 베이스라인 재구현 절 추가. 스키마 변경 없음
- v0 (2026-09-07): 템플릿
