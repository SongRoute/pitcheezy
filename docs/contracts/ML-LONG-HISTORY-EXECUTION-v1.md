# F4 긴 타자 이력 비교 실행 규약

작성 2026-09-24. 후보 결과를 보기 전 등록이며, G 준비 산출물의 SHA256이 확정되면 `EXP-P4-002` config에 연결한다. G 후보의 성능으로 이력 길이·표본을 선택하지 않는다.

## 고정하는 것과 바꾸는 것

- G와 같은 D100 TRAIN, early-stop, TRAIN 선정 Cpanel의 temperature/blend/DEV 키를 쓴다. 동일 TRAIN auxiliary·연속 투수 물리/구종/좌우 프로필과 표본 수·타자/상황 context를 사용한다. 군집 one-hot·전문가·개인 모델은 추가하지 않는다.
- F4-H0/32/128 모두 타석 내 H5와 현재 후보 구종을 유지한다. H0는 **두 번째 과거 타석 stream이 없다**는 뜻이다. 32/128은 같은 타자의 이전 완료 타석에서 가져오는 최대 구 수이며 현재 타석 전체는 제외한다.
- 같은 경기의 이전 완료 타석은 허용한다. 같은 날짜의 다른 경기는 실제 시작 순서가 확실하지 않아 제외한다. 과거 날짜의 더블헤더는 game key 순서 근사라는 한계를 남긴다. 시간상 앞선 DEV 관측 이력도 허용하는 온라인 과거 관측 평가이며 엄격한 새 타자 zero-shot이 아니다.
- 공통 dual-stream MLP는 H5 flatten 경로, 고정 위치 코드가 있는 토큰 encoder와 masked mean 경로, 공통 결과 head를 갖는다. 3개 arm의 최대128구 구조·파라미터 수는 같다. 관측 길이와 마스크만 바꾸며 H0의 사용되지 않는 경로도 유지한다. 논문 아이디어를 적용한 새 모델이며 원문 충실 재현으로 표시하지 않는다.
- seeds0/1/2, width128, 최대30epoch, patience5, batch256, AdamW lr.0005, weight_decay.01, gradient clip5. 총9 full fits. 각 arm의 TRAIN/early-stop 목록과 보정 절차는 동일하다.
- 현재 물리는 TRAIN의 공통 400 delivery draws로 적분한다. 현재 결과 입력은 0이다. May temperature 후 3seed 확률 평균, June 빈도 혼합을 주 예측으로 사용한다. seed 방향은 각각의 June 혼합으로 계산한다.
- 예측 키·정답·보정 전후 확률·모델·auxiliary pool 실제 바이트·소스 해시를 보존하고 완전9개 family 전에 성능을 열지 않는다. G0와 F4-H0는 구조가 다르므로 두 점수 차이를 긴 이력 효과라고 쓰지 않는다.

## 비교와 판정

- 주 비교는 F4-32−F4-H0, F4-128−F4-H0이다. 전체 N 및 저표본 TRAIN 투수 G의 NLL 가설 총4개를 Holm .05로 보정하며 미측정 슬롯을 유지한다. 이 저표본 집단은 타자 표본량이 아니다.
- N: 점 추정 ΔNLL≤−.003, 경기 bootstrap95% 상한<0, Holm p≤.05, ΔBrier95% 상한≤.001, 3seed 중2개 이상 개선. bootstrap10,000회 seed20260924, 경기 단위 pitch-weighted 평균이다.
- G: 저표본 N 통과와 전체 ΔNLL/ΔBrier95% 상한 각각≤.001. 희귀 타자 일반화의 근거로 확대하지 않는다.
- R: 2비교×12그룹×2지표=48 단측 Bonferroni 상한, bootstrap100,000회. 상한 ΔNLL≤.010, ΔBrier≤.002; 최소30경기·500구. 미측정은 unconfirmed이고 측정 실패가 우선이다.
- 클래스별 사건30개 미만은 해석 제한. 학습/추론 시간·epochs/updates·메모리·실제 이력 길이/지원·fallback을 함께 보고한다. N/G 통과 후보만 후속 조합에 고려하며 최대2후보 규칙을 유지한다. 독립 확인·정책 개선·전체 MLB 강건성은 별도다.

## 자원

1. O(N) int32 연결 목록과 batch 단위128구 확장으로 메모리를 제한한다. 전체 TRAIN×128 tensor를 미리 만들지 않는다.
2. 각 arm을 TRAIN8192/early2048/temp16구, 2epoch·400draw로 profile한다. DEV 특징/성능은 profile에 사용하지 않는다. 각 profile600초 상한.
3. 한 full member의 fit+temperature+blend/DEV 추론 추정이7,200초 이하여야 시작한다. 9개 family 초기예산8시간, 무거운 실행1개. 작은 profile의 선형 외삽은 시간 보장이 아니다.
4. 자원 초과 예상 시 결과를 열기 전에 실행 계획을 수정한다. 데이터·seed·draws·길이를 조용히 줄이지 않는다. 실패 attempt는 그대로 보존한다.

bootstrap은 고정된 예측에 조건부이며 모델 선택·학습·보정 불확실성 전체를 포함하지 않는다. 2024/25 노출 DEV를 사용하며2026/최종 확인셋을 열지 않는다.

## Identity-only source amendment, 2026-09-24

The first F4 preparation stopped at the global multi-batter-PA guard, before any F4 profile, fit or quality score. The sole execution owner's metadata audit found 47 PAs / 248 source rows with two distinct non-null batter IDs among 2,145,111 rows. All were marked `supported_pa=False`, and none intersects frozen TRAIN, early-stop, temperature, blend, DEV or MLB query partitions. This support observation is diagnostic only and does not define the rule below.

| Source split | Ambiguous PAs | Rows |
|---|---:|---:|
| TRAIN | 33 | 173 |
| Early-stop | 1 | 4 |
| Temperature | 0 | 0 |
| Blend | 2 | 11 |
| DEV | 9 | 48 |
| Unused | 2 | 12 |
| Total | 47 | 248 |

Feature version **`batter_dual_stream_v2`** excludes the entire PA from the **long stream only** when its observed batter identities are inconsistent. The decision uses only `(game_pk, at_bat_number, batter)` identity fields, never outcomes, labels, `supported_pa`, scores or query membership. The queried history contains only prior completed PAs in the same game or strictly earlier dates, so this identity inconsistency is available when a source PA becomes eligible. For a query inside an ambiguous PA, long-history access explicitly fails, including H0; it does not choose one batter or infer a replacement identity. No frozen query is removed by this guard according to the owner's audit.

All base/H5 rows and values, original row/query indices, parent keys, physical normalizer, type vocabulary and context fits remain unchanged. Four O(N) int32 predecessor/root arrays are built on identity-consistent rows and scattered back to original positions; excluded rows have no outgoing predecessor or root and can never be a long-stream token. Roots skip excluded date/game prefixes while retaining eligible prior dates. Same-game previous completed PAs remain available, current PAs remain entirely excluded, and other same-date games remain unavailable. A boolean identity mask adds one byte per source row. The report includes excluded PA/row counts overall and by source split, the exclusion rule and mask storage.

Synthetic CPU checks cover excluded prefixes and interior PAs for both batter IDs, later-date ordering without duplicates, unchanged earlier queries after later identity changes, fully excluded/empty histories, explicit ambiguous-query failure, base/H5/normalizer preservation, and link invariance to outcomes and retrospective support. This is a source-validity repair before scores, not outcome-based filtering or a new model-quality choice. Preserve the failed preparation and use a fresh preparation/source identity. The optional long-encoding inference optimization is a separate unadopted change and is not part of this amendment.

## Resource-profile v2 amendment, held until optional proof completes

The original resource step 2 above is preserved as the history of the completed cold 8,192-row/two-epoch profiles. A **fresh** `ml_long_history_v2` attempt may use [ML-LONG-RESOURCE-PROFILE-v2.md](ML-LONG-RESOURCE-PROFILE-v2.md): a discarded TRAIN-only warmup, independent 65,536-row/four-epoch measurement, conservative full-30-update projection, two loads per member, and all-three profile plus evidence-backed prior-cost/nine-member budget gates. The complete scientific model, full training data/batch/epochs/seeds, 400 draws and statistical tests remain unchanged. Do not integrate these source changes until the original-source MPS equivalence proof finishes; do not mutate earlier preparations or profile artifacts.
