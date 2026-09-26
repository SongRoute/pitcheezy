# EXP-P8-001 정책 실행 등록 — 초안·미실행

작성 2026-09-27. **초안이며 실행하지 않았다.** June(blend)·DEV 정책 값은 아무것도 열지 않았다. 이 문서와 [`configs/EXP-P8-001-execution.json`](../../configs/EXP-P8-001-execution.json)은 총괄 검토 후 커밋되어야 하고, 커밋 전에는 `policy-run`을 시작하지 않는다([러너 규약](ML-POLICY-RUNNER-v1.md) "Freeze/commit it before June policy values are opened"). 작성 중 실제 데이터 로드·모델 실행·heavy lock 사용은 없었다. 읽은 것은 준비·profile 산출물 JSON, 요청표 parquet의 선택/지원/경기 열, P0 parquet의 행 수 메타데이터뿐이다.

## 1. 고정하는 입력

| 항목 | 값 | 출처 |
|---|---|---|
| 준비 config | `configs/EXP-P8-001.yaml` (G3-cluster 계획 / G2-feature 공통 평가기, `diagnostic_only_not_promoted`) | 레포 |
| `preparation_sha256` | `7f911ad639eccaac3bc703b9f4f4682e501bfe2f43eadec27487e6c6567f10eb` | `EXP-P8-001/preparation.json` SHA256 직접 계산 |
| `profile_result_sha256` | `63cf00fd6b72ca16ef00f5ac73924f1287f88a717d3e7a62e7bbb09b48b152af` | `stages/profile/result.json` SHA256 직접 계산 |
| profile manifest | `bf790dae4c27eaca15a6699a90c2d6ca9809c53f6cdf4ec42ea95108c5381701`, `preparation_sha256` 일치 | `stages/profile/manifest.json` |

실행 폴더: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-P8-001`.

### 표본(줄이지 않음)

`requested_starts = {blend: 128, dev: 128}` — 준비 등록 최대치(`maximum_requested_starts`)와 같다. 러너는 선택된 요청의 앞 128개를 쓰므로 선택된 PA 전부다. 지원 안 되는 PA는 교체하지 않고 분모에 남는다.

| 구간 | 요청 PA | 선택 | 지원 | 선택 경기 수 | 지원 경기 수 | 비지원 사유 |
|---|---:|---:|---:|---:|---:|---|
| May profile(temperature) | 729 | 4 | 4 | 4 | 4 | — |
| June(blend) | 1,338 | 128 | 120 | 79 | 77 | retrospectively_unsupported_pa 8 |
| DEV(Jul–Sep) | 3,357 | 128 | 117 | 97 | 91 | retrospectively_unsupported_pa 11 |

요청·선택·지원 수는 `preparation.json.coverage`, 경기 수는 `{split}_requests.parquet`의 `selected`·`supported`·`game_pk`로 셌다. DEV 추론 최소 조건(서로 다른 경기 30, 지원 PA 50)은 **91경기·117 PA로 충족**한다. 단, 지원 판정은 사후(retrospective) 기준이다.

P0 평가 대상 투구 수(parquet 메타데이터 행 수): blend 5,212구, DEV 13,246구. context 템플릿은 blend 40, DEV 48개.

## 2. 제안 실행 파라미터

기본안은 profile 설정을 그대로 쓴다: `evaluation_rollouts=4`, `search_rollouts=2`, `search_cap=8`, `evaluation_cap=8`, `planning_seed=701`, `evaluation_seed=1701`, `tau_grid=[0.001,0.003,0.01,0.03]`(WE 절대 단위, 0.1/0.3/1/3%p), `bootstrap_draws=10000`, `bootstrap_seed=20260924`, `row_budget=460000`, `seconds_budget=1500`.

대안 R8은 `evaluation_rollouts=8`만 바꾸고 `row_budget=920000`, `seconds_budget=3000`으로 둔다. 어느 쪽을 쓸지는 총괄이 정한다. 둘 다 아래 외삽상 4시간 안에 든다.

**budget 적용 범위.** 코드(`run_ml_policy.policy_run`)는 stage 호출마다 `TimedBudget(row_budget, seconds_budget)`을 새로 만든다. 그래서 두 값은 **stage 하나당 상한**이고, 네 stage가 같은 값을 공유한다. 가장 큰 stage(blend-control) 외삽치의 2배로 잡았다. 시간 검사는 조건부 예측 호출(`consume`) 직전에만 하므로, 마지막 호출 뒤의 요약·10,000회 bootstrap·저장 시간은 `seconds_budget`에 걸리지 않는다. 이 부분은 외부 wall 상한이 막아야 한다.

## 3. 비용 외삽 (계산식 포함)

### profile 측정값

`stages/profile/result.json`: May 4 PA, P2 한 정책(candidate가 계획, control이 평가), R=4, S=2, Cs=Ce=8.

- 4.8309초(모델 로드 포함), peak RSS 981,221,376 B(0.98 GB), conditional rows 1,501, seed predictor rows 4,503 (=1,501×3), neural subnetwork rows candidate 8,628 / control 378, prediction calls 229.
- PA별 rows(`audit/p8-preparation-profile/profile.log` 누적값의 차분): 359, 582, 276, 284 → 평균 375.25, 표본 SD 142.8, 최대 582.
- 단가: 초/row = 4.8309/1,501 = 0.0032185, 초/call = 4.8309/229 = 0.021096. 로드 시간이 분모에 섞여 있어 큰 stage에서는 과대 추정 쪽이다(로드 시간 자체는 미측정).

### 외삽 규칙 (보수적으로 잡은 가정)

- **P2 1개(P2-eq)** = PA당 1,501/4 = 375.25 rows, 229/4 = 57.25 calls (R=4 기준).
- **P3 각 τ**와 **P1**은 P2-eq 하나로 계산한다. 실제로는 한 PA 안에서 planner Q 캐시를 공유하고(루트 상태 Q를 재사용), P1은 루트에서만 탐색하므로 이보다 적게 든다. 캐시 절감은 반영하지 않았다.
- **P0** = 상한 R×Ce rows(=32), 상한 Ce calls(=8). 탐색 없이 평가 궤적만 돈다.
- **R8 대안**: P2-eq rows·calls를 R에 비례해 ×2 (루트 탐색은 캐시되므로 실제로는 2배보다 조금 적다). P0 상한은 8×8=64 rows.
- stage 초 = max(rows×0.0032185, calls×0.021096). 보수 계수 ×2를 곱한 값을 상한 제안에 쓴다.
- stage 구성: blend-control = 120 PA × (P0 + P2 + P3×4 = P0 + 5 P2-eq). dev-control, dev-candidate = 117 PA × (P0 + P1 + P2 + P3 = P0 + 3 P2-eq). p0-evaluate는 시뮬레이터 row를 쓰지 않는다(BC 확률 계산만).

### 기본안 (R=4)

| stage | 지원 PA | rows/PA | rows | calls | 예상 초 | ×2 rows | ×2 초 |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend-control | 120 | 32 + 5×375.25 = 1,908.25 | 228,990 | 120×(8+5×57.25) = 35,310 | 744.9 (calls 기준) | 457,980 | 1,489.8 |
| dev-control | 117 | 32 + 3×375.25 = 1,157.75 | 135,457 | 117×179.75 = 21,031 | 443.7 | 270,914 | 887.3 |
| dev-candidate | 117 | 1,157.75 | 135,457 | 21,031 | 443.7 | 270,914 | 887.3 |
| 합계 | | | 499,904 | 77,371 | 1,632.2 | 999,807 | 3,264.4 |

→ `row_budget = 460,000` (≥ 457,980을 1만 단위로 올림), `seconds_budget = 1,500` (≥ 1,489.8을 100 단위로 올림).

### 대안 R8 (evaluation_rollouts=8)

| stage | 지원 PA | rows/PA | rows | calls | 예상 초 | ×2 rows | ×2 초 |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend-control | 120 | 64 + 5×750.5 = 3,816.5 | 457,980 | 120×(8+5×114.5) = 69,660 | 1,474.0 (rows 기준) | 915,960 | 2,948.0 |
| dev-control | 117 | 64 + 3×750.5 = 2,315.5 | 270,914 | 117×351.5 = 41,126 | 871.9 | 541,827 | 1,743.8 |
| dev-candidate | 117 | 2,315.5 | 270,914 | 41,126 | 871.9 | 541,827 | 1,743.8 |
| 합계 | | | 999,807 | 151,911 | 3,217.8 | 1,999,614 | 6,435.7 |

→ `row_budget = 920,000`, `seconds_budget = 3,000`.

### 한도 확인

| 기준 | 기본안 ×2 | R8 ×2 | 판정 |
|---|---:|---:|---|
| 단일 stage ≤ 7,200초 | 최대 1,489.8 | 최대 2,948.0 | 둘 다 안에 듦 |
| family ≤ 14,400초 (policy-run 3개) | 3,264.4 | 6,435.7 | 둘 다 안에 듦 (p0-evaluate·로드·bootstrap 시간은 미측정이라 제외) |
| PA가 모두 profile 최대 PA(582 rows)만큼 든다면 | 582/375.25 = 1.55배 | 1.55배 | ×2 안에 듦 |

- **RSS**: profile 0.98 GB만 측정했다. stage는 PA마다 planner(캐시 상한 4,096 상태)를 새로 만들고 결과 배열은 [PA, R] 크기라 크게 늘 이유는 코드상 보이지 않지만, **큰 stage의 RSS는 미측정**이다. 외삽하지 않는다.
- **candidate 세계**: profile에서 candidate(G3-cluster)는 conditional row당 neural subnetwork row가 control보다 훨씬 많았다(8,628 vs 378). profile의 candidate row는 모두 탐색 row였으므로 탐색 비용에는 이미 들어 있다. dev-candidate에서는 평가 row까지 candidate가 맡아 조금 더 무거울 수 있다. 평가 row 비중이 작아 표의 ×2 안에 들 것으로 보지만 **따로 측정한 값은 없다**.
- 편차 근거는 May 4 PA뿐이다. 이것이 June/DEV PA 길이 분포를 대표한다는 보장은 없다. 초과하면 러너가 실패를 보존하고 멈춘다. 표본이나 rollout을 몰래 줄이지 않는다. 새 시도는 새 등록으로만 한다.
- R8은 후속 offline RL 평가(`run_ml_offline_rl.py`가 `parent_policy_execution`의 `evaluation_rollouts`·`evaluation_cap`을 재사용)의 비용도 약 2배로 늘린다. RL 비용은 이 표에 없다.

## 4. 실행 순서·명령 초안

preparation과 profile은 완료·봉인됐다(`audit/p8-preparation-profile/queue-state.json`: prepare 187.9초, profile 6.0초, 모두 exit 0). 남은 단계는 한 번에 하나씩, heavy lock 아래에서 앞 단계가 봉인(`manifest.json`)된 뒤에만 시작한다. T3 등 다른 heavy 작업이 끝난 뒤에 돌린다.

```sh
RUN="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-P8-001"
AUDIT="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/audit/p8-policy"
CONFIG=configs/EXP-P8-001.yaml
EXEC=configs/EXP-P8-001-execution.json      # 커밋된 해시와 같아야 함
LOCAL_CONFIG=...                             # 질문 1: prepare/profile 때 쓴 파일과 동일해야 identity 통과
P="python experiments/pitchmdp/scripts/run_ml_policy.py --config $CONFIG --local-config $LOCAL_CONFIG --output $RUN"

$P p0-evaluate                                                      > "$AUDIT/p0.log"            # 외부 wall 1,800초
$P policy-run --execution-config $EXEC --split blend --world control > "$AUDIT/blend-control.log" # 외부 wall 7,200초
$P policy-run --execution-config $EXEC --split dev   --world control > "$AUDIT/dev-control.log"   # 외부 wall 7,200초
$P policy-run --execution-config $EXEC --split dev   --world candidate > "$AUDIT/dev-candidate.log" # 외부 wall 7,200초
```

- 맥미니에는 `timeout`/`gtimeout`이 없다. 외부 wall 상한은 queue wrapper가 걸어야 한다. prepare/profile 때처럼 `queue-state.json`에 단계별 wall_seconds·exit_code·failure를 남긴다.
- family 전체 wall 상한은 14,400초다. 한 단계라도 실패하면 뒤 단계는 시작하지 않는다. 실패 stage는 보존하고, 재시도는 새 등록으로만 한다.
- blend-control이 끝나면 June 결과는 **τ 선정과 RL 비교기 선정에만** 쓴다. DEV 첫 stage가 `stages/tuning_freeze.json`을 만들고, 두 DEV 세계가 같은 의존성을 공유하는지 러너가 확인한다.
- DEV 뒤 하위집단 보고(기술 통계만): `report_ml_policy_groups.py --policy-run $RUN --stage p0|dev-control|dev-candidate --output <실행 폴더 밖 새 폴더>`.

## 5. 결과 전에 고정하는 판정 기준

[준비 config](../../configs/EXP-P8-001.yaml)의 `inference` 블록(= `POLICY_INFERENCE`)과 [러너 규약](ML-POLICY-RUNNER-v1.md)을 그대로 쓴다. 새 규칙은 추가하지 않는다.

1. **τ 선정(June, control 세계만)**: 벌점 없는 원래 평균 수비 WE가 가장 큰 τ를 고른다. 정확히 같으면 큰 τ(BC에 가까운 쪽)를 고른다. DEV에서 다시 고르지 않는다.
2. **DEV 주 family (control 세계)**: P1−P0, P2−P1, P3−P2의 세 비교, Holm α=.05. PA 가중 경기 단위 bootstrap 10,000회(seed 20260924), 영가설 중심 단측 p.
3. **최소 조건**: 서로 다른 경기 30개와 지원 PA 50개 미만이면 세 Holm 칸을 모두 null로 두고 기술 통계만 쓴다(현재 선택: 91경기·117 PA).
4. **예비 screen (모델 내부)**: 평균 ΔWE ≥ 0.0001(=타석 시작당 +0.01%p), bootstrap 하한 > 0, Holm ≤ .05. MX-P 행의 `P` 기준 초안(ΔWE ≥ +0.01%p/PA, 짝지은 CI 하한 > 0, 다중 비교 통과)과 같다. 잘린 궤적의 꼬리 가정에 의존하므로 "예비"로만 적는다.
5. **강한 모델 내부 주장**: 같은 bootstrap 표본으로 최악 경우(worst-case) 차이도 검정한다. 두 CI 하한 > 0, 대체값 평균 ≥ .0001, 최악 경우 평균 > 0, max(대체 p, 최악 p)의 Holm ≤ .05를 모두 만족해야 한다.
6. **candidate 세계**: 민감도 분석일 뿐이다. 두 번째 성공 기회가 아니다.
7. **해석 한계**: 이 결과는 G2 시뮬레이터 안에서의 **모델 내부** 비교다. 관측 OPE·인과 효과·서비스 채택은 모두 null이다. "정책 효과"라고 주장하지 않는다. MX-P 판정표에는 `모델 내부` 칸에만 기입하고 `관측 OPE`·`전향적 실제 적용` 칸은 비워 둔다. bootstrap은 모델·보정·선택을 고정한 채 시작 상태만 바꾼 변동이다. MC SE는 정책 효과의 신뢰구간이 아니다.
8. **하위집단**: [ML-POLICY-SUBGROUP-REPORT-v1](ML-POLICY-SUBGROUP-REPORT-v1.md). 전체·TRAIN 역할·투수별 기술 통계만 내고, 새 검정·승격 규칙·성공 family는 만들지 않는다.
9. **P0**: action log-loss와 support는 행동 모형 검증용이다. 일치율을 가치 개선으로 판정하지 않는다(MX-P0).

## 6. 미정 사항

`configs/EXP-P8-001-execution.json`의 `registration` 객체는 러너가 읽지 않는 부가 필드다. 실행 hash(`canonical_hash(execution)`)에는 포함되므로 판정 기준을 결과 전에 고정하는 효과가 있다. 러너 규약의 필드 목록에는 없으므로 넣을지는 총괄이 정한다. 넣든 빼든 러너 검사(`execution_check`)는 통과한다(러너 검사 로직을 hash 계산만으로 재현해 확인했다).

## 총괄 결정(2026-09-27, 실행 전 고정)

- `evaluation_rollouts=4` 기본안을 채택한다. profile과 같은 값이며 후속 offline RL 평가 비용을 늘리지 않는다. R8 대안은 기록으로만 남긴다. MC 표준오차는 별도 기술 통계로 보고한다.
- `row_budget`/`seconds_budget`은 코드 동작대로 **policy-run stage당** 상한이다(가장 큰 stage 외삽의 2배). 규약의 "one aggregate budget" 문구와 다르다는 점을 기록하며, family 합계는 queue wrapper의 외부 wall(stage 7,200초·family 14,400초)로 강제한다.
- `registration` 부가 필드는 유지한다. runner가 읽지 않지만 실행 hash에 포함되어 판정 기준을 결과 전에 고정한다.
- `--local-config`는 다른 실험과 같은 `experiments/pitchmdp/configs/local.json`이다(SHA256 `eb364ca7…`가 preparation identity와 일치 확인).
- 추가 profile은 돌리지 않는다. 한도 초과나 실패는 산출물을 보존하고 새 attempt로 등록한다.
- 실행은 T3 stress(EXP-P7-002)가 heavy lock을 반환한 뒤 시작한다.
