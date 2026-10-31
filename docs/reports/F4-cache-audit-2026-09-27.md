# F4 observed-context cache 실측 감사 — 2026-09-27

**결과: 17/17명령 완료, 세 arm의 제한된 수치 동등성 통과, 캐시 미채택.** 원본과 캐시의 표본 자원 profile을 동결식으로 확장한 9member family projection은 캐시 전체 생성 비용을 **제외해도 61,431.69초**로 28,800초 상한을 넘는다. `full_fit_started=false`, `adoption=null`이다. 이 감사에는 DEV 품질 채점이나 정책 평가가 없다.

## 등록과 실행

- [D70 사전 등록](AI-collaboration-2026-09-27.md#f4-실측-등록--결과-열람-전): 실행 소스 `d022711368057a75badaba46c50f728e718c3e52`, config SHA256 `3b3bd37d0e7459db323c52f8cedd8db454e06ba6e404cfaabb413c1cdb0d4230`. 원래 F4 학습 설정·허용 오차·400 draws는 바꾸지 않았다.
- SSD 출력 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-P4-002-v3-cache-audit-v2`; 외부 큐 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-claude/execution/f4-cache-audit-queue-v2`. prepare, H0/32/128 각각 stage1·stage2·원본/캐시 stage3·compare, summary가 모두 exit 0·상한 내로 종료됐다. 자동 재시도나 실패 단계는 없다. 완료 시각 `2026-09-27T08:19:18.279546+00:00`.
- TRAIN·early-stop·May 고정 표본만 사용했다. stage1은 context 원본/캐시의 65,536 TRAIN 및 2,048 평가 행과 변형 입력 가드를 검사했고, stage2는 같은 초기화·미니배치의 전방·손실·기울기·갱신을 비교했다. stage3는 각 경로에서 8,192행 warmup 후 65,536행×4epoch fit과 May16×400 보정·추론을 측정했다. DEV·June 캐시와 점수는 열지 않았다.

## 동등성과 표본 비용

세 arm에서 stage1 bitwise/guard, stage2 동등성, stage3 원본·캐시 비교가 모두 통과했다. stage3의 선택 상태, 학습 이력, 보정 목적함수, raw/보정 확률 최대 차이는 모두 **0**이었다. 총괄의 독립 산출물 검사는 17 manifest와 70 artifact hash에서 오류 0을 기록했고, Astra 최종 검토는 48개 배열쌍의 bitwise 일치를 확인해 `PASSED`로 회신했다. 검토 결과는 채택 또는 DEV 품질 통과를 뜻하지 않는다.

| arm | 4epoch fit 원본/캐시 (초) | fit 감소 | warmup 원본/캐시 (초) | May 보정 원본/캐시 (초) | 추론 원본/캐시 (초) | stage3 peak RSS 원본/캐시 (GB) |
|---|---:|---:|---:|---:|---:|---:|
| F4-H0 | 41.44 / 39.96 | 3.57% | 2.08 / 2.10 | 0.69 / 0.65 | 0.68 / 0.66 | 6.74 / 6.73 |
| F4-32 | 41.95 / 40.63 | 3.14% | 2.10 / 2.08 | 0.69 / 0.66 | 0.68 / 0.65 | 6.73 / 6.78 |
| F4-128 | 43.55 / 42.21 | 3.07% | 2.18 / 2.13 | 0.70 / 0.67 | 0.69 / 0.66 | 6.80 / 6.73 |

표의 RSS는 각 stage3 프로세스의 측정 fit·추론 중 최대 resident set size이며 십진 GB다. stage1/2도 포함한 전체 감사 프로세스 최고 RSS는 6.80 GB.

| arm | stage1 캐시 생성: 행/초 | stage3 fit 프로세스 캐시 생성: 행/초 | v3 원본 projection (초/member) | 감사 원본 projection | 캐시 projection, 전체 캐시 생성 제외 |
|---|---:|---:|---:|---:|---:|
| F4-H0 | 67,474 / 0.291 | 75,358 / 0.321 | 6858.1 | 6793.9 | 6556.9 |
| F4-32 | 67,474 / 0.292 | 75,358 / 0.318 | 6934.2 | 6865.7 | 6640.8 |
| F4-128 | 67,474 / 0.291 | 75,358 / 0.321 | 7137.0 | 7110.8 | 6876.6 |

stage1 생성은 TRAIN·평가 행의 union, stage3 생성은 warmup·평가·TRAIN·early-stop·May의 fit 프로세스 표본 범위다. 전체 1,252,824 TRAIN행 및 별도 예측 프로세스에서 필요한 캐시 생성은 **미측정**이다. 따라서 캐시 member의 실제 전체 시간과 7,200초 member 상한 통과 여부는 `null`이다. 위 캐시 projection은 그 비용을 빼고 계산한 동결식 수치이며 실제 총 실행 시간의 보장치나 하한이 아니다.

## 비용 대조와 판정

- 검증된 과거 16명령: **244.084859208초**. 이번 외부 큐 17명령: **385.467954665초**; 두 지출의 합계 **629.552813874초**. 외부 큐는 Python 시작부터 종료까지 포함하며 17개 end 기록의 `math.fsum`으로 재계산했다.
- 봉인된 summary가 실행 중 기록한 내부 감독기 비용은 **964.862325544초**로, 그 안에는 아직 미종료였던 summary 자체의 **600초 상한 예약**이 들어 있다(이전 내부 완료 지출 364.862325544초). 종료 후 내부 17명령 실측 합은 365.927870585초다. 내부와 외부 지출은 중첩되어 **합산하지 않는다**.
- 봉인된 판정용 family projection(전체 캐시 생성 제외)은 **61431.685925821초**, `family_projection_gate_excluding_cache_construction=false`, `decision=not_adopted`다. 내부 예약 비용 대신 종료된 외부 큐 실측을 한 번 치환한 별도 설명용 수치는 **60852.291554943초**이며 역시 28,800초 초과다. 이 치환은 봉인된 판정을 수정하지 않는다.
- 세 arm의 4epoch 표본 fit 감소는 약 3.07–3.57%다. 전체 캐시 생성이 미측정이고 30epoch family projection이 실패했으므로 이번 감사로 캐시 채택이나 F4 full fit을 열지 않는다. 실제 모집단 비용 검증과 새 등록·총괄 결정이 별도로 필요하다.

## 증거와 독립 검토

- 봉인 summary: `summary/attempt1/results.json` SHA256 `c59cc7be1356953717cc3cf2b221b8294dbdd384a4c333fed4da782f1c41d22e`; manifest `4b49374b0776ac2578eb1c75b0be47992e476e27500d0e2da2facfda4cc9e3da`.
- 외부 큐 `complete.json` SHA256 `2284c73bb92e48300c24e1200223cdc4775fa7d58411c0a8042b422102be4596`; `freeze.json` `4c60aa7aa2e9c6efebe1f55df7614fc6a6a1dd96518976068bff3d4ee6e9956b`. 총괄 산출물 검사 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-claude/f4-cache-root-artifact-review.json` SHA256 `7613685199c85fe1b5ef421bf25552957feb803547319369559cf94e46821538`; 17 manifest/70 artifact hash 검증 오류 0.
- 주요 단계 결과 SHA256 (전체 15개 결과와 17개 큐 로그는 [기계 판독 결과](../../results/EXP-P4-002-v3-cache-audit-v2.json)의 `evidence_sha256`에 수록):

| arm | stage1 | stage2 | original stage3 | cached stage3 | compare |
|---|---|---|---|---|---|
| F4-H0 | `805ddd49928a…` | `2bfcbc1d4c43…` | `f1287b792c02…` | `7c9bdf8c3865…` | `8a16101ea8fd…` |
| F4-32 | `d872f028c837…` | `1f21124185fa…` | `2f555753c6dd…` | `59df4974db4c…` | `025ff74df63e…` |
| F4-128 | `8e17062acfb7…` | `37d28875fd04…` | `d395e9abbdac…` | `ae6e4cb06af0…` | `f7f121c42ec0…` |

Astra의 최종 결과 검토는 `PASSED`(48 배열쌍 bitwise 일치), 총괄 독립 검토는 위 hash·manifest·17개 종료 기록을 통과했다. 총괄은 실행 완료 후 전체 프로젝트 검사 **252 passed, 19.75초**도 보고했다. 실측과 검토 범위는 수치 동등성·제한된 자원 감사이며 예측 품질·인과 효과·정책 채택은 평가하지 않았다.
