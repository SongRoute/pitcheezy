# 추천 안정성 진단: 구현 초안과 PM 검토 요청 (2026-10-03)

작성: Claude Fable 5.1 작업자(Codex PM 배정). 작업트리 `codex-showcase-research-1006`, 기준 커밋 `a971a1d`.
상태: **코드·합성 테스트 완료, 실데이터 미실행, 실험 ID 미배정, 등록 아님.** P17·P18의 공식 판정(D152 `DIFFERENT_RECOMMENDATIONS`, D155 `WITHIN_SEARCH_NOISE`)은 바꾸지 않는다.

## 요약

- 한 일: G0 고정 상태에서 계획 seed·탐색 횟수를 여러 개 돌려 P3 추천 분포의 안정성을 재는 **추가 실행기**와, 끝난 진단 결과(P17·P18·새 실행)를 **오프라인으로 묶어 읽는 집계기**를 만들고 합성 테스트 18개를 통과시켰다.
- 결과: 새 파일 5개(실행기 1, 집계기 1, 테스트 2, 초안 config 1). 기존 등록 config·결과·실행기는 수정하지 않았다.
- 다음: PM이 ① 실험 ID 충돌 확인·배정 ② 변형 부분집합·비용 결정 ③ 실제 입력 smoke ④ 집계 보고서의 저장소 결과 채택 여부를 정한다.

## 1. 무엇을 재는가 (그리고 재지 않는가)

**재는 것.** 봉인된 6월 τ 설계 표본의 결정 40,702개(150경기)에서, 예측기는 G0 그대로 두고 계획기의 탐색 seed와 표본 수만 바꿨을 때 P3 분포(τ 0.1)가 봉인 seed-0 분포에서 얼마나 떨어지는지. P18은 seed 1 하나만 봤다. 이번 초안은 seed {2,3,4}(6회)와 seed {0,1}(12회)를 더해 D155의 부수 관찰("seed만 바꿔도 1순위 11.3% 변경, 탐색 증가 효과 미측정")을 측정 범위로 바꾸려는 것이다.

**재지 않는 것(보고서 계약에 고정).**
- 정책 효용·개선 여부. 경기 결과·보상·WE 결과는 읽지 않는다(`outcomes_read: false`를 집계기가 검사하고, 없거나 true면 거부).
- 정책 재검증(단계 C) 상태. D155로 닫힌 줄기를 다시 열지 않는다. 혼합 예측기는 넣지 않는다.
- 서비스·데모 채택 근거. 정책 식별자 `c13cc989…`는 그대로.
- P17·P18 판정 변경. 집계기는 입력 결과에 있는 등록 판정 문자열을 그대로 옮기고(`registered_verdicts_unchanged`) 다시 계산하지 않는다. 새 실행기는 판정·문턱을 config에 붙이면 거부한다.

## 2. 설계 결정: 왜 P18 실행기를 고치지 않고 따로 뒀나

P18 실행기는 `EXPERIMENT='EXP-P18-001'`, noise seed 하나, P17 결과 파일 해시를 상수로 묶고 있어 봉인된 결과의 재현 경로다. 거기에 다중 seed 분기를 넣으면 등록된 실행기의 source hash가 바뀌어 `results/EXP-P18-001.json`의 `source_hashes`와 어긋난다. 그래서 **P18·P17의 순수 함수(`sealed_ledger`, `bind`, `law`, `decision_metrics`, `require`, `SMOKE`)를 import로 재사용하는 추가 실행기**를 만들었다. 기존 파일은 한 줄도 바꾸지 않았다.

재사용 경계: 봉인 원장 검증(fingerprint·mask·기준 분포·맥락 해시), 등록 사슬 로딩, 컴포넌트 바인딩과 S2 식별자 검사, P3 분포 계산식은 P17·P18과 같은 코드다. 새로 쓴 것은 변형 목록 검증, 변형별 계획기 생성, 결정별 비교 행(분포 보존), npz 저장이다.

## 3. 파일

| 파일 | 역할 | 실데이터 |
|---|---|---|
| `experiments/pitchmdp/scripts/diagnose_policy_seed_stability.py` | 실행기. 변형(seed, samples)별 계획기 → 결정별 P3 분포·TV·1순위 저장 | **미실행** |
| `experiments/pitchmdp/scripts/diagnose_policy_stability_aggregate.py` | 집계기. 끝난 실행 폴더들(P18-001, P17-001, 새 형식)을 request_id로 정렬해 기술통계 JSON 생성. numpy·json만 사용, 모델·자료 배열 안 읽음 | 미실행(합성만) |
| `experiments/pitchmdp/tests/test_diagnose_policy_seed_stability.py` | config 검증, 변형 규칙, 결정 비교, 난수 접두 성질 | 합성 |
| `experiments/pitchmdp/tests/test_diagnose_policy_stability_aggregate.py` | 집계·정렬·식별자·로더·CLI | 합성 |
| `experiments/pitchmdp/configs/drafts/policy-seed-stability-DRAFT-v1.json` | 다음 후보 config 초안. `experiment_id: UNASSIGNED`, `status: DRAFT`, `execution.enabled: false` | — |

## 4. 실행기 계약 (`diagnose_policy_seed_stability.py`)

- config 검사: protocol, `origin`에 P18 이후 작성임을 명시, P18과 같은 `policy_registration`·`sealed_g0_run`·`settings`·`data_rules`, `replay`, `row_budget`, `variants`, `bootstrap`. `verdict_rule`/`stop_rule`가 있으면 거부(기술통계 전용). **본 실행**은 `EXP-P\d+-\d{3}` 형식의 배정된 ID, `status: REGISTERED`, `execution.enabled: true`, `configs/<id>.yaml` 위치를 요구한다. **smoke**(3경기, 저장소 결과 없음, 출력명 `-smoke`)는 DRAFT·UNASSIGNED 파일에서도 돌릴 수 있어 ID 배정 전에 배선을 확인할 수 있다.
- 행 예산 바닥: P18 실측 3,805,035행(6회)을 가장 큰 표본 수로 비례시키고 1.2배 여유를 둔 값 이상이어야 한다(12회 포함 시 9,132,084). 본 실행 몇 시간 뒤 예산 초과로 실행을 잃는 일을 막기 위한 규칙이며, 초안은 9,200,000으로 둔다.
- 변형 규칙(`variant_plan`): seed는 0 이상 정수(bool·float 거부), samples는 양의 정수, 봉인 설정(0, 6)과 같은 쌍 거부, 중복 쌍·중복 라벨 거부, `seed_basis` ∈ {planning_main, planning_v2, extension}를 정책 등록의 `seeds` 블록(main 0, v2 [0,1,2])과 대조. seed 3·4는 등록된 계획 seed가 아니므로 `extension`으로만 선언 가능하고 정책 식별자 필드로는 쓰지 않는다.
- 난수 접두 성질(테스트로 확인): numpy는 순차로 뽑으므로 같은 seed에서 12회 텐서의 앞 6행은 6회 텐서와 같다. 따라서 `seed0-s12`는 봉인 난수를 포함한 **중첩 비교**이며 독립 seed 거리보다 작게 나온다. `shares_prefix_draws`로 표시한다.
- 무결성: P17과 같이 처음 200결정을 봉인 설정(seed 0, 6회)으로 재계산해 봉인 Q와 `atol` 안에서 일치해야 진행. 모든 요청은 봉인 fingerprint·mask·기준 분포·맥락 해시와 일치해야 한다.
- 출력: `sealed.npz`(request_id, game_pk, strikes, law, top1, mask), `decisions-<label>.npz`(law, top1, tv, top1_same, kl, max_abs_q_difference), `results.json`(`comparison: null`, `verdict: null`, 변형별 점 평균만). **저장소 결과는 쓰지 않는다**; 집계 보고서를 저장소 결과로 삼을지는 PM 결정.

## 5. 집계기 계약 (`diagnose_policy_stability_aggregate.py`)

입력 폴더마다 `results.json`에서 봉인 측 식별자(경로, manifest sha, 최종 식별자 sha, 원장명, 평가 결정 수, τ, pitch_cap, 봉인 seed·samples)를 뽑아 **모든 입력이 같은 봉인 측**을 가리키는지 검사한다. 다르면 어떤 키가 다른지 적고 거부. `outcomes_read`·`policy_identity_registered`·`smoke`가 없거나 false가 아니면 거부.

정렬: 첫 입력의 request_id 순서를 기준으로 나머지를 재배열. 집합이 다르면(누락·초과·중복) 거부. game_pk 불일치도 거부.

보고서 구조(`aggregate()` 반환, 테스트에서 JSON 직렬화 확인):
- `fixed_predictor_seed_variability` — seed만 바꾼 행(P18-001의 seed 1 포함). 변형별 평균 TV(경기 부트스트랩 95% 구간), 1순위 일치율, 분위수, TV>0.05 비율.
- `sample_count_variants` — 표본 수를 바꾼 행. `shares_prefix_draws` 표시.
- `predictor_replacement` — P17-001 행을 그대로 옮긴 것(교체 TV·일치율, 등록 판정 문자열).
- `cross_variant` — 분포를 저장한 변형끼리의 쌍별 평균 TV, 봉인 포함 1순위 만장일치 비율, 평균 최빈 일치율. 옛 형식(P18·P17)은 분포가 없어 여기 들어가지 않는다.
- `samples_trend` — 표본 수별 평균 TV.
- `support_coverage` — 비교 결정 수 / 봉인 평가 결정 수, 결정당 지원 구종 수 평균, 지원 구종 1개(어떤 seed에서도 못 움직임) 비율. 봉인 mask가 없으면 "unmeasured".
- `bootstrap` — 새 실행 형식의 results.json에 기록된 등록값(초안 10,000회·seed 20260924)을 자동으로 쓰고, 입력끼리 다르면 거부. CLI `--draws/--seed`로만 덮어쓰며 `source`(registered/cli/default)를 기록한다.
- `limits`, `claims_excluded`, `registered_verdicts_unchanged`, `outcomes_read: false`.

짝 불확실성의 한계(보고서 `limits`에 고정): 모든 행이 같은 봉인 seed-0 한쪽을 공유해 독립 반복이 아니다. 경기 부트스트랩은 봉인 분포를 고정한 채 경기 표집만 다룬다. 교체 비교(P17)는 공통 난수를 쓰고 seed 비교는 쓰지 않으므로 seed 거리가 교체 안 잡음을 과대평가한다(D155 그대로). 접두 중첩 행은 독립 거리보다 작다.

## 6. 테스트

```
/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest \
  experiments/pitchmdp/tests/test_diagnose_policy_stability_aggregate.py \
  experiments/pitchmdp/tests/test_diagnose_policy_seed_stability.py \
  experiments/pitchmdp/tests/test_diagnose_policy_seed_noise.py \
  experiments/pitchmdp/tests/test_diagnose_policy_predictor_swap.py -q
→ 18 passed
```

합성 테스트가 덮는 것: 집계 평균·구간 포함, 쌍별 TV 대칭·0 대각, 1순위 투표, kind 분리, 표본 수 추세, 지원 범위; request_id 순열 재정렬, 누락·초과·중복·game_pk 불일치 거부; outcome 플래그 누락·true 거부, 봉인 식별자 불일치 키 나열, 라벨 중복; P18·P17 옛 형식과 새 형식 로더 왕복, 깨진 형식 거부; CLI 출력·덮어쓰기 거부; config 초안의 미배정·DRAFT·실행 꺼짐 거부와 등록 복사본 통과; seed/samples 잘못된 값 14가지; 정책 등록 seeds 블록 읽기; 난수 접두 성질; 결정 비교 행.

덮지 않는 것: 실제 원장·바인딩·계획기 경로 전체. 봉인 자료·가중치는 이번 작업에서 읽지 않았다.

## 7. 비용 추정 (추정, 미측정)

P18 실측: seed 1개·6회·40,702결정 = 1,965초, 최대 RSS 7.2GB. 초안의 5개 변형(6회 3개 + 12회 2개)을 한 프로세스에서 결정별로 돌리면 약 3 × 1,965 + 2 × ~3,930 ≈ 13,800초(**약 3.8시간, 추정**) + 200결정 재현. 12회 변형의 행 수는 P18 3.8M의 두 배 근방이 예상되며 초안 `row_budget` 9,200,000은 실행기가 요구하는 바닥(9,132,084) 위에 있다. 최소 부분집합(`seed2-s6`, `seed0-s12`)은 약 1.6시간(추정). D159 시간 분리와 운영 동결 안에서만 실행.

## 8. PM 검토 요청 사항

1. 실험 ID: 인테이크 문서가 EXP-P18-002를 제안했으나 다른 브랜치·작업트리와의 충돌은 확인하지 못했다. 배정 후 초안을 `configs/<id>.yaml`로 옮기고 `status: REGISTERED`, `execution.enabled: true`로 바꾼다. 등록 줄에 "P17·P18 결과 열람 뒤 작성(P18 기반 규칙)"을 그대로 적는다.
2. 변형 부분집합과 실행 시각(동결·D159).
3. 실제 입력 smoke(`--smoke`, 3경기·20결정 재현). 이 작업에서는 돌리지 않았다. 초안 파일 그대로 smoke가 가능하다.
4. 집계 결과를 `results/<id>-aggregate.json`으로 커밋할지.
5. 독립 검토: 작성 모델은 Fable 5.1. 실행기 데이터 경로는 Opus 5.5 하위 에이전트의 읽기 전용 검토를 받았다(§9). 검토는 판정 변경 권한이 아니며, 기존 독립 검토 게이트 통과로 기록하지 않는다.

## 9. Opus 5.5 읽기 전용 검토 결과 (2026-10-03, 반영 완료)

검토 범위: 실행기의 실데이터 경로를 P17·P18 실행기와 호출 단위로 대조, 집계기와의 파일 계약 확인. 검토자는 허용한 읽기 도구 외에 `python3 -c`로 저장소 안 결과 JSON 두 개의 키를 한 번 출력했다(/Volumes 접근 없음).

- 호출 서명·반환 형태·키 불일치: **없음**. 집계기가 새 형식과 P17·P18 옛 형식을 모두 읽는 것 확인.
- 지적 1 — 행 예산: 변형별 `RowBudget(6,000,000)`은 12회 변형(≈7.6M행 예상)에서 몇 시간 뒤 초과해 실행을 잃는다 → 실행기에 예산 바닥 검사 추가, 초안 9,200,000.
- 지적 2 — 절차: 초안(UNASSIGNED·DRAFT)은 smoke조차 거부돼 "등록 전 smoke"가 불가능했다 → smoke는 초안에서 허용, 본 실행만 등록 요구.
- 지적 3 — 부트스트랩: config의 10,000회가 results.json에 기록되지 않아 집계기가 기본값 2,000회를 썼다 → 실행기가 `bootstrap`·`row_budget`을 기록하고 집계기가 등록값을 읽는다.
- 낮은 위험으로 남긴 것: `data_rules` 문구는 P18 config와 같은지 해시로 대조하지 않는다(설정·봉인 run은 τ freeze·manifest 해시로 재검증). 집계기는 smoke 출력(`smoke: true`)을 거부하므로 실행기→집계기 실파일 계약은 본 실행 뒤에야 확인된다.

## 10. 남은 불확실성

- 실행기 데이터 경로는 실행되지 않았다. 검토에서 오류를 찾지 못했지만 첫 smoke가 대신하지는 않는다.
- 12회 변형의 실제 시간·메모리·행 수는 미측정.
- `top1_unanimous_share` 등 교차 통계는 새 실행 형식에서만 나온다. P18-001은 분포를 저장하지 않아 seed 1은 봉인 대비 거리로만 들어간다.
- 이 진단이 보여줄 수 있는 것은 "표시되는 추천이 탐색 잡음에 얼마나 흔들리는가"의 기술통계뿐이다. 데모에서 성능 우위·안정성 입증으로 표현하지 않는다.
