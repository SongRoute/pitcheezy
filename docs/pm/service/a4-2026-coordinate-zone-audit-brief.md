# A4 2026 좌표·존 정의 정합 감사 — 브리프

작성 2026-10-27 · Claude 연구 차선 · 소유 PM/운영 차선(D186 6-①) · 상태 **브리프**(등록·실행 아님). 원자료·모델·저장소 사본은 열지 않고 코드·문서·설정만 읽었다.
경로 약칭 — **M** `/Users/song/Projects/pitcheezy`(codex/ml-matrix-execution) · **J** `/Users/song/Projects/pitcheezy-pm-jobs`(레포 밖 작업 폴더)
**L** `/Users/song/Projects/pitcheezy-worktrees/ws-launch-a-1017`(ws/launch-2026 `e53b89a7`, 지금 라이브). **발행기 코드는 M에 없고 L에만 있다.** 줄 번호는 이 기준.
용어: **P0 계열**=옛 25칸 전이·OPE 파이프라인 · **G0/G15**=동결 연구 예측 모델과 그 개선 혼합 · **ARM-B**=서비스용 동결 구종 추천 경로 · **봉인**=바이트를 고정해 못 고치게 한 모델·결과 · **EXPOSED DEV**=이미 본 자료라 "확인" 증거가 못 되는 개발용 자료 · **GUMBO**=MLB 실시간 피드 JSON

## 세 줄 요약
- 무엇: Savant가 2026부터 `plate_x/z`(공이 홈플레이트를 지나는 위치)를 "앞면" 대신 "한가운데" 기준으로 재고, `sz_top/bot`(존 위·아래)을 ABS(로봇 심판) 기준으로 바꿨다. 우리는 모든 시즌을 같은 격자에 넣는다.
- 확인한 것: 좌표가 모델에 들어가는 길은 ① 25칸 격자(P0 계열) ② G0/G15·ARM-B의 이전 공 입력 중 `plate_x/z` 2채널 ③ `zone` 토큰(SEQ-002). RS-01의 Step5·N1·T_FINAL은 좌표를 안 읽는다. 라이브는 피드의 존 높이를 일부러 버리고 ≤2025 고정값을 쓴다.
- 다음: #37 CLAIM → 등록·독립 검토 → 방법 1~3단계. 어긋남의 크기는 전부 **미측정**.

## 1. 목적
- 같은 공도 어느 깊이에서 재느냐에 따라 숫자가 다르다. 홈플레이트 앞 모서리와 한가운데 사이에서도 공은 계속 떨어진다. 낙차 큰 공일수록 클 것으로 보이나 **크기는 미측정**.
- 25칸 경계는 고정 피트(x ±0.83/±0.28)와 존 비율(z ⅓)이다. 정의가 바뀌면 같은 공이 다른 칸에 들어간다. 이미 본 2026 숫자와 라이브 표시가 이 값에 기댄다.
- 막힌 일: A3 초안은 이전 공 위치·물리 입력을 금지하며 이유로 A4를 인용(`docs/pm/research/track-a/A3-behavior-policy-registration-draft.json` :202). A2 초안도 "A4 먼저"(`A2-ope-redesign-registration-draft.json` :541).
- 첫 질문: 로컬 2023~2025 파일은 버전 ID 기준 2026-09-11 수집(`data/versions.md` :13-15)이라 Savant 변경(제안서 인용 changelog 2026-02-19)보다 늦다. ≤2025 값이 옛 정의 그대로인지 새 정의로 다시 쓰였는지 **미확인**.

## 2. 범위 — 어디에 들어오나 (파일:줄)
**격자 정의 (M)**
- `src/pitcheezy/interfaces/grid.py` :1-7(설명) · :15-20(경계·25칸) · :25-33 `z_norm` · :36-56 `x_col/z_row/loc_id`
- `docs/interface-spec.md` :30(행동=구종×25+칸) · :32(경계) · :26(맥락 v1이 가운데 3×3을 "존 안"으로 씀)
- `tests/interfaces/test_grid.py` :10-12가 경계를 계약으로 고정. `docs/design.md`에는 격자 문장이 없다(검색 0건).

**≤2025 빌더·내보내기 (M)**
- 25칸을 쓰는 P0 계열: `src/pitcheezy/data/prepare.py` :21-25(읽는 열) · :73-81(칸 배정) · :47(2026 holdout)
  - 같은 계열: `baselines/b2.py` :65·:199-203 · `interfaces/value.py` :66-76 · `experiment.py` :306-311(2026 OPE) · 2026 결과 `docs/results-p0.md` :3-4
- 좌표를 입력으로 쓰는 G0/G15/ARM-B 계열(`experiments/pitchmdp/pitchmdp/`):
  - `sequence_data.py` :21-24(물리 8채널 중 `plate_x/z`) · :126-150(정규화기, TRAIN 창 2023-05-15~2025-04-30)
  - `matrix_features.py` :1-5·:50-64 · `data.py` :28·:162-163
- **지원 PA(모델이 다루는 타석) 판정** `data.py` :271-273·:287: 좌표 결측, `sz_top>sz_bot`, 관측 카운트 경로 불일치가 표본을 가른다. 2026의 이 비율이 2025와 다른지 **미측정**.
- D131 2026 OPE `experiments/pitchmdp/scripts/run_policy_validation.py` :66-70: 2026에서 `plate_x/z`를 읽고 `sz`·`zone`은 안 읽는다.
- `zone` 열(Savant 존 번호): `experiments/ws_seq/prepare_inputs2.py` :56-57 · `configs/EXP-WS-SEQ-002.yaml` :9·:33
  - 같은 파일 :17-19는 `plate_x/z`·`sz`가 투영 밖이라 안 읽는다고 적는다.
- 그 밖: `scripts/prepare_june_eligibility.py` :40-41·:306-307(좌표 결측 제외)
  - 타자별 이전 날짜 `sz` 중앙값을 만드는 데모 빌더: `apps/observer/scripts/build_catalog.py` :89-90·:122·:138, `build_demo.py` :63-77
- CUDA 내보내기 `experiments/pitchmdp/scripts/export_cuda_direct.py`는 M에 없다. `/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026`에 있고 :65-66·:74가 2026 행을 거부한다.
- 수집은 열을 거르지 않고 정규시즌만 받는다(`src/pitcheezy/data/statcast_fetch.py` :242-254·:64). 2026 전체본은 `d20260930-h2026f` 716,792행.
  - 코드에 `vx0…az`·`release_pos_*`를 읽는 곳이 없어 이 열이 파일에 있는지는 **미확인**.

**라이브 피드·발행기 (L)**
- 흐름: `apps/observer/ops/ws_operate.py` :1-14·:70 → `backend/observer_app/publisher.py` :5 → `service_api.ServiceBodies` → 아래
- `apps/observer/live_config.json` :27-28: 존 높이를 ≤2025 리그 중앙값 1.6/3.3899 ft로 고정. "피드의 2026 `strikeZoneTop/Bottom`은 일부러 무시"라고 적혀 있다.
  - 같은 값: `observer_app/game_history.py` :34·:422 · `web/src/service/api.ts` :316 · `web/src/WatchAlong.tsx` :29 · 발행 스키마 `publish_contract/v1/schema/game.schema.json` :2365
- 피드 좌표를 읽는 곳(`observer_app/`): `live_feed.py` :573-574·:614-615 · `live_armb.py` :42-53 · `live_store.py` :290-313 · `game_history.py` :538·:563
  - 팬 화면의 "실제 구역"은 피드 `pX/pZ` + 고정 존 높이로 정한다(`domain.py` :12-31).
- 모델 입력 `scripts/demo_precompute.py`:
  - :48-54 호출코드→description 표 · :58-77·:199-209 피드→Statcast 열(`plate_x←pX`, `sz_top←strikeZoneTop`)
  - :353-373·:616-637 TRAIN 배달 풀의 ≤2025 좌표로 "목표 구역"을 정한다 → 실제 위치와 정의가 다를 수 있다.
- `live_store.py` :293-296: 849831에서 257구 모두 "원시 궤적 항"만 바뀐 정정을 받았다. 피드에 궤적 항이 있다는 증거(항목 이름은 미확인)이자 첫 발행 뒤 바뀔 수 있다는 경고다.

**RS-01 / Step5 / RS-02**
- Step5-2024 3-head 199개 특징(원이름 91개)에 좌표·`sz`·`zone`이 **없다**: `J/20261010-ws-service-006/step5-f1-author-001/step5/features.py` :25·:31-32, `outputs/FIELD_CONTRACT_199.json`
  - RS-01 어댑터: `/Users/song/Projects/pitcheezy-worktrees/ws-rs01-1018/experiments/ws_prospective/ws_prospective/adapter_step5.py` :1-14 → `step5_bridge.py`(17×199 행렬)
- N1/T_FINAL 토큰도 `pitchData`를 안 읽는다(`M/src/pitcheezy/seq_shadow/tokens.py` :20, seq_shadow 폴더 검색에서 좌표 읽기 0건). 그래서 RS-01은 좌표 정의가 아니라 **판정·카운트 기록**(ABS 번복)에만 닿는다.
  - 849831 Step 5 그림자에서 번복으로 투구 전 카운트가 바뀐 예측이 2건 있었다(`docs/pm/service/state.md` :27).
- RS-02(봉인)는 이전 공 `plate_x/z`를 `pX/pZ` 그대로 쓴다(`J/20261020-rs02/RS02-CONVERTER-SPEC.md` :172 · `REGISTRATION-RS02-SEALED.json` :694-703).
- 선행 검사 V-1은 **2025**에서 피드 `pX/pZ`=Statcast `plate_x/z`를 확인했다(최대 |차이| 0.00045/0.00105 ft, `J/20261020-rs02-v1/RESULT.md`). 2026은 **미측정**.

**범위 밖:** 모델 재학습·교체, D131 재실행, 영상(CV) 좌표 사슬(`apps/observer/backend/observer_app/intent.py` :35·:110-125, `docs/contracts/C0-v1.md` :11), 라이브 코드 변경(권고만).

## 3. 방법 (순서대로, 앞 단계가 실패하면 멈춘다)
0. **등록 먼저.** 자료 역할(2025·2026 모두 읽기 전용 EXPOSED DEV), 표본 규칙(V-1처럼 경기 id 해시 순서로 시즌당 N경기), 재계산 식, 깊이 후보, 허용오차, 재채점 대상·지표를 **열람 전에** 고정하고 작성자와 다른 에이전트가 독립 검토한다.
1. **열 확인.** 2023~2026 파일에 `vx0 vy0 vz0 ax ay az release_pos_x/y/z`가 있는지, 시즌별 결측률(스키마와 결측률만 본다).
2. **식 검증.** 등가속도 궤적을 풀어 지정한 깊이의 `x,z`를 계산한다(기준점·단위는 Savant 컬럼 정의로 확정). 2025 표본에서 raw와 대조해 식이 raw를 못 맞추면 멈춘다.
3. **깊이 식별.** 같은 공을 두 깊이(앞면·중앙)로 재계산해 시즌·월별로 raw가 어느 쪽과 맞는지 가른다. ≤2025 파일이 옛 정의인지, 2026은 개막부터 바뀌었는지(중간 전환 여부)를 본다.
4. **한 깊이로 통일.** 모든 시즌 `plate_x/z`를 한 깊이로 재계산해 새 열로 둔다(원본 불변). raw 격자 대비 재계산 격자의 **칸 일치율**을 시즌별로 낸다.
5. **존 높이.** `sz_top/bot` 시즌별 분포(타자별 퍼짐·분위수)를 비교하고 라이브 고정값(`live_config.json` :27)과도 견준다. z축 정의(열린 질문 2)를 정한 뒤 25칸이 시즌 불변인지 확인한다.
6. **라이브 피드.** 피드에 궤적 항이 있다면 그것으로 `pX/pZ`의 깊이를 같은 방법으로 판정한다(Statcast 없이 가능). 2026 정규시즌과 맞대는 V-1식 교차 대조는 새 피드 수집이 필요해 별도 결정(열린 질문 3).
7. **ABS 기록.** 번복 판정이 `description`·`events`·카운트·호출코드에 어떻게 남는지 본다.
   - 시즌별 `description` 값 목록(새 값·`unmapped` 건수): `src/pitcheezy/interfaces/outcomes.py` :8-9·:40-42, `demo_precompute.py` :48-54
   - 지원 PA 제외 사유 중 "카운트 경로 불일치" 비율(`data.py` :287), 849844의 개정 사례(`live_armb.py` :95-98)
   - 같은 위치의 판정 확률이 달라지는 것은 정의 문제가 아니므로 **보정하지 않고** 기술 보고만 한다.
8. **정합 전/후 재채점(기술 통계만).** 동결 모델 추론만(새 학습·튜닝·선택·정책 재평가 없음). 2026 EXPOSED DEV에서 좌표 채널을 원값/통일값으로 넣고 같은 투구 짝으로 NLL·Brier·결과 10종별 보정을 비교한다. 대상은 범위에서 좌표를 입력으로 쓰는 모델만.

## 4. 산출물
- **감사 노트** 1건(제안 경로 `docs/pm/service/a4-2026-coordinate-zone-audit-note.md`)
  - 시즌별 표본 수, 재계산 대 raw 일치(허용오차 이내 비율·|차이| 분위수·최대), 깊이 판정, 칸 일치율, `sz` 분포, ABS 기록 방식, 정합 전/후 수치
  - 모든 수치에 seed·데이터 버전·커밋을 붙인다.
- **의존 표**: 2026 지표 × 쓰는 필드(좌표 / `sz` / `zone` / 판정·카운트).
  - EXP-WS-CAL-001·RECENCY-001/002는 Step5와 같은 199 특징 계열로 보이나 확인 전이라 이 표가 먼저다.
- **격자 정의 한 줄**: "x열 = 깊이 y*의 포수 시점 `plate_x`(경계 ±0.83/±0.28 ft), z행 = <열린 질문 2의 규칙>, 전 시즌 같은 y*·같은 규칙"
  - `interface-spec.md` 변경 이력 + `decisions.md` 한 줄 후보로 올린다. `grid.py`·라이브 코드 수정은 별도 승인(라이브는 권고 메모만).
- **"정합 전" 표기 목록**: D131 OPE(`results/MLB-2026-OPE-v1.json`), P0 계열 2026 OPE(25칸·`sz`), SEQ-002(`zone`), 지원 PA 비율, 팬 화면 구역·목표 오차, RS-02 해석.
  - Step5·N1·T_FINAL·SEQ-001은 7단계 결과에 따라 표시한다.

## 5. 통과 기준 (숫자는 2026을 열기 전에 등록)
- 2단계: 2025 표본에서 재계산≈raw가 허용오차 이내 비율 ≥ 사전 고정 값.
  - 허용오차는 두 후보 깊이 재계산값의 차이(같은 공)보다 충분히 작아야 깊이를 가른다.
  - 출발점은 V-1이 같은 두 필드에 등록한 0.02 ft(`J/20261020-rs02-v1/REGISTRATION-V1.json` :60-61).
- 3단계: 시즌별 깊이 판정이 한쪽으로 분명해야 한다. 아니면 "판정 불가"로 쓰고 격자 불변을 주장하지 않는다. 2026 결과는 보고용이며 통과/탈락에 쓰지 않는다.
- 4~5단계: 2025에서 재계산 격자와 raw 격자의 칸 일치율 ≥ 사전 고정 값(옛 산출물과 연속).
  - 노트에 "**이 정의에서 격자 칸은 시즌에 따라 변하지 않는다**"를 명시하고 근거(칸 배정 함수에 시즌 인자 없음 + 테스트)를 단다. 못 쓰면 미통과.
- 봉인 불변: G0·G15 목록 `verify` PASS(`scripts/freeze_g0_research.py`, `scripts/freeze_g15_mix_research.py`), 정책 식별자 `c13cc989…`, `results/MLB-2026-OPE-v1.json`, RS-02 봉인 sha가 바이트 그대로.
  - 기존 파일은 덮어쓰지 않고 새 이름으로만 쓴다.

## 6. 제약
- 소유는 PM/운영 차선. 착수 전 #37에 CLAIM(`docs/pm/work-queue.md`; 범위: 노트·새 스크립트 경로·맥 CPU 창), 코드는 별도 worktree.
- WS 경기일 06:30~13:30 KST에는 맥 무거운 작업 금지(D186 2-②). 기록기·수집 프로세스에 신호·메모리 압박을 주지 않는다.
- 2026 자료는 학습·튜닝·선택에 쓰지 않는다(기술 비교만). 등록·독립 검토 전에는 실자료를 열지 않는다.
- RS-01 저장소 사본은 열람일(2026-11-06T00Z) 전에 열지 않는다. G15·정책 식별자 `c13cc989…`·서비스는 불변(D131·D155).
- 라이브 worktree(`e53b89a7`)와 launchd 설정은 건드리지 않는다. 배포는 별도 결정이며 경기 사이에만(D183).
- 숫자 추정 금지, 모르면 "미측정". 결과마다 seed·데이터 버전(`d20260911-s2325`, `d20260930-h2026f`)·커밋을 적는다.

## 7. 예상 작업량 (대략치 — 측정 아님)
- 제안서의 "CPU 수 시간"과 선례에서 거칠게 잡았다. 총합은 **미측정**이고 등록 때 예산을 확정한다.
  - 선례: V-1 대조는 30경기 0.33초(`J/20261020-rs02-v1/RUN.json`), 재채점은 G0 seed당 전체 MLB 약 355초(D132, 2025 표본). 2026 전 시즌 재채점은 이보다 크다.
- 사람 시간은 이틀 안팎으로 추정: 등록·독립 검토 반나절~1일, 코드·실행 반일, 노트 반일.

## 8. 열린 질문 (결정은 PM/운영 차선)
1. 고정 깊이: 앞면(≤2025 정의) vs 중앙(2026 정의). **추천 앞면** — 동결 모델·정규화기·봉인 산출물이 전부 ≤2025 정의라 바뀌는 것이 가장 적다. 단 3단계에서 ≤2025 파일이 이미 중앙 기준이면 재검토.
2. z축: 25칸의 타자별 `sz`는 2026에 ABS 정의라 의미가 시즌마다 다르다. ① 절대 피트 경계(라이브가 쓰는 1.6/3.3899) ② 키 기반으로 전 시즌 재구성(키 자료 유무 미확인) ③ 시즌별 격자를 따로. **추천 ①** — 라이브와 정합.
3. 2026 피드 대조 표본: 정규시즌 N경기를 새로 받기(MLB 공개 피드 요청 발생) vs 기존 캡처만(포스트시즌이라 Statcast 정답 없음).
4. 허용오차·표본 크기 N·재채점 대상 모델 목록은 누가 정하나(등록서 작성자 + 독립 검토자).
5. 격자·라이브 코드 수정은 감사 통과 뒤 누가 언제 하나(라이브는 경기 사이 창). RS-02 봉인과 정의 차이가 확인되면 결과 해석문에만 반영하나.
