# G0 전체 MLB 평가와 제한된 확률 보정

**결과:** 동결 5시드 G0는 기존 frequency 기준선보다 노출된 2025 전체 MLB DEV와 정확한 non-Cpanel 차집합에서 모두 NLL·Brier가 낮았고, 두 사전 등록 N 슬롯이 `development_advantage_over_frequency`였다. 그러나 top-label ECE10은 G0가 더 컸다. 진단으로 활성화된 I1 class-bias 보정은 G0보다 NLL·Brier가 악화해 `worse_or_guardrail_failure`였고, I2 TRAIN-volume 보정은 June 그룹 지원이 부족해 **비활성**이었다. 기존 G0를 유지하며 두 보정 후보를 승격하지 않는다. 이는 이미 노출된 DEV의 개발 결과로, 독립 확인이나 서비스·정책 채택 근거가 아니다. Astra의 독립 산출물·주 비교 재계산 감사는 **PASS**다. [Opus의 최종 보고서 교차 검토](../reviews/COOP-008-Opus-5.5-result-review.md)도 차단 사항 없이 완료됐다.

## 등록·실행과 입력 경계

실행 코드 C는 `5e82bebc239ced378911919de48c106034ee24b0`, 등록 D는 `3f10f61bea591871f69aad6a869d89d0bbe598ed`다. [실행 계약](../contracts/ML-G0-WHOLE-MLB-v1.md)의 `EXP-P11-001`과 `EXP-P11-002`를 분리했다. 두 등록 설정 SHA256은 각각 `b6565f167993bdb31481ce18aee6d49276f3db00b61f518b5eee45e8f00a319b`, `75d51ecfacfcf41d8d0519d47f3bc1c60f5e8d4949cf3fac25f568668c3acaad`다. 실행 계획 파일 SHA256은 `f83e0e95b41d1caec3e8172bf517de38871901be8775cb70b137610b5a4642a0`, canonical SHA256은 `bb1e1394d4cdd6698f5235614ec12007826f469abeb41680e66b090b72efcc23`다. SSD `coordination/20260927-whole-mlb/registration.json`이 이를 연결한다.

등록 때 Astra의 독립 과학·소스 사전 감사 `PASS`와 미해결 blocker 0건을 기록했다(감사 JSON SHA256 `f86ec8fc2aefbb67828af2a7f79b2d4e2b07c6d7ea6162ac759c758773466327`). Opus 재검토 기록 SHA256은 `fbb0e64584f0d5f43a407cdf6bb9bc28c841bf7ab6daee805c47998c6a4b5feb`다. 145개 실험 소스와 단계당 146개 source pin, 기존 G·C1·T4 부모, June ordered keys/labels를 등록 전에 확인했다. 전체 코드 검사 863개·17 subtests 및 최종 집중 검사 8개가 통과했다. 코드·감사 검사는 예측 우위를 입증하지 않는다. Fable은 stage-2 worker 초안, Opus는 순수 보정 알고리즘, Astra는 scorer·과학 검토, Sol은 stage-3 runner·감독 경계, 총괄 Codex는 계약·동결·등록·큐 통합을 맡았다.

등록된 큐 **12/12단계가 종료 코드 0으로 완료**됐다. Stage 2는 G 부모 `EXP-P4-001`의 seed 0–2와 C1 `EXP-P10-001`의 seed 3–4, 총 다섯 동결 모델을 사용했다. 새 신경망 fit은 **0개**이며 기존 May 온도·C1 5시드 June ensemble/시드별 가중치를 재사용했다. 적격 규칙 그대로 요청 **341,941구** 중 **311,721구·1,161경기**를 평가했다. Cpanel 교집합은 **12,334구·328경기**, 정확한 키 차집합은 **299,387구·1,161경기**이고 두 부분에 공유되는 경기는 **328개**다. Key·label·game·pitcher 정렬 및 `in_cpanel` 일치 게이트를 통과했다. 원본 C1의 `raw`, `calibrated`, `primary`, `seed_primary` 배열과 교집합 재현 최대 절대 오차는 **모두 0**(`atol=1e-6`)이다.

## Stage 2: 동결 G0 대 frequency

주 결과는 투구 가중 평균, G0−frequency 방향이다. 음수는 G0 손실이 더 낮다는 뜻이다. 조건부 95% 구간과 단측 p는 seed `20260924`의 완전 경기 10,000회 재표집이다.

| 모집단 | G0 / frequency NLL | G0 / frequency Brier | ΔNLL [95% CI] | ΔBrier [95% CI] | Holm p·음수 seed·N |
|---|---|---|---|---|---|
| 전체 MLB, 311,721구·1,161경기 | 1.490408196725 / 1.504081016265 | 0.726592367393 / 0.729480617036 | **−0.013672819540** `[−0.014415351081, −0.012924306116]` | **−0.002888249643** `[−0.003208395779, −0.002563278172]` | 0.000199980002 · **5/5** · `development_advantage_over_frequency` |
| non-Cpanel, 299,387구·1,161경기 | 1.490699666523 / 1.504365619910 | 0.726705695464 / 0.729586669744 | **−0.013665953387** `[−0.014421095506, −0.012903051809]` | **−0.002880974280** `[−0.003205768308, −0.002552629166]` | 0.000199980002 · **5/5** · `development_advantage_over_frequency` |

조정 p≈0.000200은 10,000회 bootstrap의 해상도 하한이다(plus-one 원 p=1/10,001에 두 슬롯 보정).

등록 기준은 각 슬롯 ΔNLL ≤ −0.003, NLL CI 상한 < 0, Holm 조정 단측 p ≤ 0.05, Brier CI 상한 ≤ +0.001, 음수 seed ≥4/5다. 전체의 seed별 G0−frequency ΔNLL은 `[−0.012657589511, −0.012540358917, −0.012823851974, −0.012240007886, −0.012896530209]`; 차집합은 `[−0.012680493519, −0.012521260838, −0.012804384444, −0.012217132919, −0.012903406776]`이다. R의 12그룹 × NLL/Brier **24/24 Bonferroni 동시 상한이 등록 margin 내에 들어 `passed`**이며 미측정 슬롯 0개다. 이는 동결 G0와 frequency 사이의 등록된 개발 집단 결과이지 독립 일반화가 아니다.

손실 우위가 모든 보정 지표의 우위는 아니다. 전체의 top-label ECE10은 **G0 0.008710236256**, frequency **0.001660086965**였고, 차집합은 **0.008904281796 대 0.001487556226**였다. 동일 투수 가중 macro NLL은 G0 **1.502837086712**, frequency **1.516634146010**이다. 열 메타데이터의 결측과 기록된 game-role/throwing-hand `unknown`은 모두 **0**이었다. 10개 클래스 모두 최소 30개 사건 조건을 충족했다. 상세 월별·seen/unseen·투수별·클래스·delivery tier는 봉인된 full results에 남겼으며 여기서는 추가 가설 판정으로 확장하지 않는다.

## Stage 3: I1은 악화, I2는 지원 부족으로 비활성

Stage 2 결과 해시를 `stage2_binding.json`(SHA256 `c2fe82bbbd4fc38b06bf60b516e9e9795b74d8620c163ed66154d276a310652e`)에 묶은 뒤 activation을 봉인했다(SHA256 `2b84e8991af415c114d23e8afef1491b353d607d46ca9afabee0bc6629a6587a`). I1은 event ≥30인 클래스 0, 1, 5에서 예측−관측 prevalence 절대 차이가 각각 **0.003751041, 0.005767337, 0.001310055**로 등록 문턱 0.001을 넘어서 활성화됐다. I1의 June ensemble 및 seed 0–4 **6/6 class-bias fit은 optimizer 성공·유한성·제약 검사를 통과**했고, 전체 확률에 적용되기 전에 파라미터 manifest로 봉인됐다(SHA256 `96c0739e9bef9c578a497abee218b469c97c0b163ea1f2c0ec98dec1dc87653d`). Ensemble의 클래스 0–9 bias는 `[+0.080267409, +0.167565097, +0.045888001, +0.024825607, +0.052429244, −0.164842742, −0.010726348, −0.115480194, −0.124568999, +0.044642923]`이다. 시드별 전체 벡터는 봉인된 `fits/I1.json`(SHA256 `dbeaa2de0f996ea46246f95e6c7c3b08c4353aba977bf79b79598415e7c5ba9f`)에 있다.

I2의 June 그룹 지원은 zero **0구·0경기**, low **635구·12경기**, middle **1,014구·19경기**, high **3,172구·86경기**였다. 최소 30경기와 500구를 양쪽 자료에서 만족한 양의 TRAIN-volume 그룹은 high 하나뿐이므로 두 그룹 이상이 필요한 활성화 지표는 `unmeasured`, I2는 **비활성**이다. I2 fit·적용은 0건이고, inactive p=1 및 R 26슬롯을 보존했다. Zero/low/middle에서 특수 보정을 학습하지 않았으며 unseen-player 보정 효과는 **미측정**이다. 등록 알고리즘의 기존 전체 가중치 fallback도 실제 I2 후보로 적용된 결과로 제시하지 않는다.

| I1 후보−동결 G0 | NLL Δ [95% CI] | Brier Δ [95% CI] | 음수 seed·Holm p·N 상태 |
|---|---|---|---|
| 전체 MLB, 311,721구 | **+0.000876227760** `[+0.000633119024, +0.001114141026]` | **+0.000306153225** `[+0.000206268934, +0.000403401122]` | **0/5** · **1.0** · `worse_or_guardrail_failure` |
| Cpanel 교집합, 12,334구(기술) | +0.001300333479 `[+0.000244846851, +0.002365043618]` | +0.000567803346 `[+0.000132518574, +0.001004864317]` | 주 판정 슬롯 아님 |
| non-Cpanel 차집합, 299,387구(기술) | +0.000858755659 `[+0.000612860312, +0.001101128119]` | +0.000295373890 `[+0.000194294420, +0.000394625110]` | 주 판정 슬롯 아님 |

I1의 전체 절대 NLL은 **1.491284424485**, Brier는 **0.726898520618**로 동결 G0의 1.490408196725/0.726592367393보다 높다. I1의 13그룹 × NLL/Brier **26개 동시 R 상한은 모두 등록된 비열등 margin(NLL +0.010, Brier +0.002) 내**에 들어 `passed`로 기록됐으나, 이는 **N 악화를 뒤집지 않는다**. 통상 95% 양측 구간의 하한이 0보다 높은 그룹은 NLL 11/13, Brier 10/13으로 악화 방향 신호도 존재한다. I2의 26개 R 슬롯은 비활성으로 미측정이며 전체 52슬롯 분모는 그대로다. June Cpanel에서 한 번 맞춘 단순 class-bias가 전체 MLB 보정 약점을 해결했다는 증거는 없다.

## 비용·산출물 감사

외부 감독기의 권위 Popen→wait worker wall은 평가 단계 **1,821.2459352919832초 / 7,200초**, 보정 단계 **34.42338987579569초 / 3,600초**다. 12개 명령이 모두 완료됐고 별도 실패·미종료 job은 기록되지 않았다. 전체 외부 큐 wall **1,857.4866603328846초**는 감독 프로세스와 단계 사이 overhead를 포함한 교차 확인값으로, 두 단계의 권위 비용에 **더하지 않는다**. 큐 status SHA256은 `c784953daebb8445de26ce87fe00b6550c2f9dc90acf6ed25061df1d05aaed70`이다.

Stage 2 full results SHA256 `9fefaaa644adb108e044c0e85a8f3d2dcae54dbfc80cb4051c22f0d097b60f3c`, 분석 manifest SHA256 `aaf5db0d0a3d134b4d92393efa640dc67dba8efb4a4af330d21f2ab66d9ebda9`다. Stage 3 full results SHA256 `c5b9f29374a9ee841f1fcff64f6442a0a943615d9e2b9e6ce4180f9fc1ced5f7`, 분석 manifest SHA256 `8fe8f970d51dd009a518ce03bae671c1dbb98a35e1926daef4bbc8c0d4fc3e21`다. 각 manifest는 입력·결과 SHA256을 보존한다. Astra의 [최종 독립 감사](../reviews/G0-whole-MLB-result-audit-2026-09-27.md)는 **PASS**, 미해결 blocker 0건이다([감사 요약](../../results/G0-whole-MLB-audit-2026-09-27.json); SSD 감사 원본 SHA256 `60e88ecc2f2e71dd5ca1bcb8686aef53ea5a5e8c616700fa647f7ac8f15c91b0`). 별도 해시·점손실 검사는 `PASS`; 전체·non-Cpanel G0−frequency와 I1−G0의 NLL/Brier 주 비교를 독립 10,000회 경기 재표집으로 다시 계산해 CI 최대 오차 **0**, p값 정확 일치를 확인했다. 이 추가 감사 full-command wall **5.257020499790087초**는 실험 단계 비용에 넣지 않았다. R 동시상한 및 절대 손실 CI의 **수치 재계산은 하지 않았고**, 봉인 해시와 판정 규칙만 검토했다. I1 optimizer도 재적합하지 않았으며 봉인된 성공·제약 증거를 확인했다. [Opus의 최종 보고서 교차 검토](../reviews/COOP-008-Opus-5.5-result-review.md)도 차단 사항 없이 완료됐다.

## 해석 경계

두 단계 모두 이전에 노출된 2025 DEV를 사용한다. Non-Cpanel은 행만 제외한 정확한 키 차집합이고 Cpanel과 **328경기를 공유**하며, 이전 T4 전체 MLB DEV 관찰도 지우지 못한다. Stage 3는 알고리즘·문턱을 미리 고정했지만 stage-2 진단을 본 뒤 활성화한 적응적 후속 작업이다. Holm p와 R 구간은 고정 출력의 국소 비교이며 새로운 전체 연구 유의수준 보증이 아니다. Bootstrap은 학습·보정·선택 불확실성을 포함하지 않는다. `independent_confirmation: null`, `held_out_confirmation: false`, `policy_effect: null`, `service_adoption: null`이다. TRAIN0은 D100 TRAIN 투구 0건을 뜻하며 앞선 DEV의 합법 문맥까지 없는 strict zero-shot은 아니다. 개별 선수의 손실과 월별 차이는 원인적 표현 실패나 drift 판정이 아니다. 2026 자료·정책 효과·실경기 성과는 평가하지 않았다.
