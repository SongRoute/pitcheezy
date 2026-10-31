# 11월 B2→B3 등록 초안과 저장 산출물 근거 감사

상태: **DRAFT_UNREGISTERED / 실행 불가**. 작성 기준 `2a3d4d23`(main `d04afab1` 위 11월 설계). Song의 후속 추천안 진행 지시에 따른 문서·manifest 감사다. 새 원자료·예측확률 배열 열람, 재채점, fit, GPU 접속·점유, 설정 변경, 기존 산출물 복사·수정은 하지 않았다. 이 문서는 실행 승인이나 기존 등록의 개정이 아니다. G0·정책 식별자 `c13cc989`·전시 서비스·RS-03은 불변이며 **2026 자료는 학습·튜닝·모델 선택에 사용하지 않는다**.

계획 식별자 `NOV-B2-NN-001-DRAFT`, `NOV-B3-TEMP-001-DRAFT`는 현재 checkout의 configs/docs/results 검색에서 기존 사용이 없었다. 실제 EXP 실행 ID는 아니다. 다른 활성 작업의 범위와 실행 ID를 다시 확인한 뒤 별도 등록하며, 이번에는 configs·decisions.md·experiments.md를 수정하지 않는다. 기계 실행 설정을 만들기에는 §8의 pin·실제 실행 코드가 부족하여 문서만 작성한다.

## 1. 질문과 이름의 경계

- 근거는 [11월 설계](model-improvement-2026-11.md), [10/26 제안 §3·§5](experiment-proposals-2026-10-26.md), [실험 매트릭스](../../ML_EXPERIMENT_MATRIX.md), [D188·D192·D197](../../decisions.md), [P48 집계 결과](../../../results/EXP-P48-001.json), [B1 동결 목록](prospective-2027/arm-C-B1-freeze-list.json)이다.
- **B2-NN:** 11월 설계의 최소 재사용안, P48 TabM residual과 periodic residual의 저장 확률을 섞어 상보성을 검증한다. 10/26 원안의 LightGBM+NN+G0 전체 교차 가족 실험은 아니다. 두 NN은 같은 residual 줄기를 공유하므로 독립적인 가족이라고 부르지 않는다. LightGBM 추가·새 backbone fit은 별도 등록 범위다.
- **B3-TEMP:** B1 TabM을 미리 고정하고, 평균 전 온도 보정을 평균 뒤 단일 온도로 바꾼다. B2 결과를 보고 B3의 부모·보정 방식·CAL을 고르지 않는다. B2→B3는 실행 순서이며 승자 선택 사슬이 아니다. B2에 보정을 결합하려면 다음 등록이 필요하다.
- D188의 과거 결정은 B1을 RS-03 arm C 후보로 동결한 것이며 B2/B3 실행 승인이 아니었다. 이번 후속은 감사·초안까지 구체화한다. D192의 행동정책 B2/A2b 분모는 이름만 같고 별도 소유 작업이다. D197의 계획 예산 확대는 열지 않는다.

## 2. 이번에 확인한 것과 확인하지 않은 것

집계 JSON·설정·등록 문서·소스·manifest의 내용을 읽고, 모델/예측 파일에는 `stat`만 했다. 아래 파일 개수·크기는 이번 로컬 metadata 관찰이며, 예측값 일치·라벨 정렬·성능 재현을 새로 확인한 수치가 아니다.

| 근거 | 관찰 | 의미·한계 |
|---|---|---|
| P48 B1 mac 회수본 | 동결 목록 35/35 파일 존재, 35/35 크기 일치, 합계 260,316,064 bytes | checkpoint/예측을 열거나 바이트 hash를 재계산하지 않음 |
| P48 B1 T7 보존본 | 같은 목록 35/35 존재·크기 일치, 합계 260,316,064 bytes | 원본 freeze-list의 동일 SHA 기록은 과거 증거; 이번 동일 바이트 검증 아님 |
| P48 periodic mac 회수본 | 5seed×7파일=35/35 존재, 합계 259,797,861 bytes | B1의 35파일 동결 번들에는 포함되지 않음. 새 재사용 manifest와 소유자 범위 확인 필요 |
| P48 control mac 회수본 | 35/35 존재, 합계 257,781,710 bytes | 파이프라인 재현 대조용. 새 후보로 선택하지 않음 |
| G15 설정의 files | 61개 참조 경로 모두 로컬 존재 | 파일 존재만 확인. 새 실행 전 설정→member state→archive의 고정 사슬 검증 필요 |
| R metadata | P4 `dev_metadata.parquet` 50,182 bytes, `mlb_dev_metadata.parquet` 953,995 bytes, `panel.json` 443,208 bytes; P11 `dev_metadata.parquet` 953,995 bytes 존재 | parquet·panel 내용은 열지 않음. 크기 같음은 같은 키/내용의 증명이 아님 |

P48 mac 기준 경로는 `/Users/song/gpu-pull/pitcheezy/261027-152429-p48-main/runs/EXP-P48-001`; T7는 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-P48-001/stageA-family`다. G15/R 자료의 공통 runs root는 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924`다. 활성 소유자의 결과 디렉터리에는 새 파일도 만들지 않는다.

등록/집계/manifest 텍스트의 SHA는 이번에 재계산하여 기존 pin과 일치했다:

| 파일 | SHA-256 |
|---|---|
| `results/EXP-P48-001.json` | `54e38dc6734eed202e201e15349b5dd88d55e9506f613512dea0f05b3fa34bc0` |
| P48 소유 worktree의 `configs/EXP-P48-001.json` | `2a417d8f6721d0ff158af662512bbce24a5e446cb4be9ddf66127b3e5f7feb5b` |
| `configs/G15-MIX-RESEARCH-FROZEN-v1.json` | `b4c19988722c25d994b6f14c9b87b3ff3a177fb95702fe4c61a518fbbb201e6d` |
| `configs/G0-RESEARCH-FROZEN-v1.json` | `c839dbf710c02ca5a1baafcc168e6ca3eaa6d80e269a99f71f08d7d537ec8062` |
| `results/EXP-P16-001.json` | `9d372da26b39fca07d6963a6861044721b31cf42209c92d5edb66f7b8a778597` |
| `CUDA-EXPORT-20261004/manifest.json` | `49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578` |

P48 소유 worktree는 `/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026`다. P48 등록 커밋 `c5f61d47618546867fec9ebfdb9ea68062effc10`, 검토 소스 `011c257fc29f021626d90b1c128c653183baab51`, B1 동결 bundle SHA `6271b70e834215aff96141b8319e84ba041fb980600cc91791248a1b95f5e4ec`는 기존 기록의 pin이다. 현 main에 P48 등록/러너 전체가 포함되어 있다고 가정하지 않는다. 새 실행은 이 커밋과 파일 closure를 명시한 별도 checkout으로 구현·검토한다.

P48의 `confirmed`는 반복 노출 DEV에서의 등록된 N 통과다. R은 `not_computed`, adoption/policy effect는 null이다. 두 DEV의 중복, 과거 후보 선택, June MIX_FIT·빈도 blend 선택의 불확실성은 기존 bootstrap에 포함되지 않는다. 이번 감사는 R이나 독립 확인을 소급해 채우지 않는다.

## 3. 고정 자료 역할·키·분모

아래는 위 export manifest의 집계다. 기간은 계획상 DEV 7/1~9/30이나 실제 마지막 eligible 행은 9/28이며, 9/29~30 행을 새로 채우지 않는다. TRAIN/CAL도 새 수집 없이 기존 고정 모집단을 사용한다.

| split | 역할·실제 날짜 범위 | 경기 / 행 | ordered int64 key SHA-256 |
|---|---|---:|---|
| train | TRAIN, 2023-05-15~2025-04-30 | 4,671 / 1,252,824 | `0a2e81aecbf9dbb0453544440561b090017f3903ea7456ea7be47fcdf6b3ba6a` |
| earlystop | 기존 epoch 선택, 2025-05-01~05-15 | 195 / 16,000 | `ff0fce97fe5bd807c9517214b4204a932f2cdaab2ebc1268e456985c171666b0` |
| temperature | Cpanel CAL-T, 2025-05-16~05-31 | 60 / 2,603 | `1b5a76cd8e378b3aec3e606220e5e6adb2e971e19f4c629f9b6cda00769ad59a` |
| blend | Cpanel CAL-MIX, 2025-06-01~06-30 | 110 / 4,821 | `0f66807ba93c185e40416e6543183f389dac595fec310fb5a56cfe21bd9c4674` |
| dev | EXPOSED DEVELOPMENT Cpanel, 2025-07-01~09-28 | 328 / 12,334 | `461222d0059d71f134e1255c1a8d719b8b1b01b98e26864c8d1e23f16c33a8c1` |
| mlb_dev | EXPOSED DEVELOPMENT MLB, 2025-07-01~09-28 | 1,161 / 311,721 | `afa9d8e3098aa91510a8dce9e63691202d87d65871665a179dbaf8b36736a516` |

- 비교 키는 `(game_pk, at_bat_number, pitch_number)` int64 3열이다. split별 중복·결측을 거부하고 키 집합·순서·`y`·game/pitcher를 대조한다. 순열만 다르면 키로 정렬하고 permutation과 hash를 기록한다. 한쪽에만 있는 행을 inner join으로 버리지 않는다.
- 각 파일의 10-class 순서는 고정 `pitchmdp.model.OUTCOMES`와 label mapping source hash로 고정한다. 클래스 이름을 추측해 재정렬하지 않는다. 유한 [0,1]·행합 오차 ≤1e-6, label 범위 0~9, 날짜 ≤2025를 로더 경계에서 검사한다. epsilon·정규화는 기존 채점기와 동일하게 고정하고 잘못된 벡터를 조용히 보정하지 않는다.
- 저장된 eligible 분모 12,334/311,721과 **전체 요청 분모**를 분리한다. 전체 요청 수·지원 제외·기권·fallback·unknown grouping 수는 이번에 미측정이다. 기존 preparation/요청 metadata를 고정하고 각 전이의 개수를 보고해야 지원 범위 주장이 가능하다. 후보별 가용 행만 평가하면 유효성 실패다.
- TRAIN 학습 변환·선수 빈도·Cpanel 선정은 고정한다. 현재 공 물리 8채널을 0으로 둔 direct-head 계약, H5의 관측 완료된 과거만 사용한다. 새 현재 공 결과/구속/좌표 입력, 2026·봉인 OPE·2021–22 원자료 열람은 범위 밖이다. G0의 기존 delivery 적분은 변경하지 않는다.
- seed는 기존 모델 RNG `[0,1,2,3,4]` 전부이며 seed screen/재추첨 없음. 5개 seed는 새로운 경기 표본이 아니다. P48/P15/P16의 반복 노출을 새 검증셋으로 부르지 않는다.

## 4. B2-NN 후보·대조군·CAL 절차

정의: `F_T`는 동결 P48 TabM `MIX_FIT`의 **June 빈도 blend까지 완료한** 확률, `F_P`는 같은 P48 periodic `MIX_FIT`의 같은 단계 확률이다. seed별 값과 seed-mean 기반 ensemble 값을 구분한다. 각 branch의 기존 CAL temperature, G0 혼합비, 빈도 blend 계수는 원래 기록/재현 결과 그대로 동결한다. ensemble 혼합비를 seed별 평균으로 대체하지 않는다.

1. **대조군:** B1=`F_T`, G15 동결, P48 control/MIX_FIXED 파이프라인 진단. G0는 provenance·재현 anchor이고 주 비교 대상으로 늘리지 않는다.
2. **후보 하나:** `F_B2(a)=a*F_T+(1-a)*F_P`. `a∈{0,0.25,0.5,0.75,1}`만 허용한다. ensemble용 a 하나와 seed별 a 5개를 CAL-MIX의 투구 가중 NLL 최소로 선택한다. 정확한 동률은 더 큰 a(B1 우선); `a=1`이면 `NO_CHANGE_B1`이지 새 개선이 아니다.
3. CAL-MIX 외 행에서 a 선택 금지. 선택한 ensemble a를 그대로 MLB DEV에 적용한다. seed별 값은 같은 seed의 비교에만 쓴다. 새 빈도 blend나 G0 weight를 다시 적합하지 않는다. 따라서 변경 축은 이미 보정된 두 branch 사이의 최종 혼합 하나다.
4. 이 CAL-MIX는 branch의 기존 weight/빈도 blend 선택에도 쓰였다. 새 a까지 같은 6월을 재사용하는 **제한된 추가 개발 선택**이라고 기록한다. 독립 CAL 확인으로 표현하지 않는다. 새 분할로 branch들을 다시 최적화하는 실험은 이번 B2가 아니며 별도 등록한다.
5. 완성 branch의 June 확률이 파일로 없으면 기존 고정 계수와 저장 head/G0/frequency 확률로 재구성한다. 계수 누락·재현 오차·pin 불일치면 중지한다. DEV 수치에 맞춰 계수를 역추정하지 않는다.

## 5. B3-TEMP 후보·대조군·CAL 절차

B3 부모는 **B1 TabM**으로 지금 고정한다. B2 성패에 따른 부모 교체 없음. vector/matrix scaling, refine-then-calibrate, 새 학습·조기 종료는 제외한다.

- 대조군 `PRE`: 기존 P48 B1. `DirectTabMModel.logits`는 이미 `log(mean_k softmax(member_logits))`를 반환한다. 각 seed의 내부 k=8 확률 평균 `q_s`에 기존 `T_s`를 적용하고, 보정된 5seed 확률을 평균한다. 이후 G0와 June 빈도 혼합은 기존 규칙이다.
- 후보 `POST`: 동일한 seed별 raw 확률 `q_s`에서 `q_bar=mean_s(q_s)`를 먼저 만든 뒤 `softmax(log(clip(q_bar,1e-30,1))/T_post)`를 적용한다. **변경은 seed 평균과 보정의 순서**이며 내부 k 평균은 그대로다. logits 평균과 확률 평균을 혼동하지 않는다.
- `T_post` 하나는 **CAL-T 2,603행만**으로 투구 가중 NLL 최소화: 기존 bounded scalar 방식, bounds `[0.5,2.5]`. 최적화 허용오차·평가 상한은 새 구현에서 pin할 블로커다. 범위 끝 값은 공개하고 실패/비유한 결과는 `CALIBRATION_FAILED`; 자동 다른 보정기로 넘어가지 않는다.
- G0의 기존 보정은 그대로다. POST head를 B1의 고정 ensemble G0 weight **0.7**로 섞고, B1의 고정 June frequency coefficient를 적용한다. seed 진단은 공통 `T_post`를 각 `q_s`에 적용하고 B1의 원래 seed weight `[0.65,0.70,0.70,0.75,0.70]` 및 원래 seed frequency coefficient를 쓴다. 이 seed 진단 평균이 ensemble POST와 같다고 주장하지 않는다.
- **June에서 온도를 적합하거나 새 weight를 고르지 않는다.** PRE/POST 모두 동일한 B1 downstream 계수를 사용한다. 미래 DEV나 B2 승자를 보고 CAL을 분할하지 않는다. 기존 B1의 June 선택 이력은 보존하며, 이 비교는 새로운 독립 CAL이나 독립 일반화가 아니다.
- 기존 `run_cuda_direct.py`는 CAL-T로 온도를 적합하지만 저장 archive에는 blend/dev/mlb_dev의 calibrated/raw만 쓴다. CAL-T raw 확률을 저장한다고 가정할 수 없다. **B3는 5개 checkpoint의 CAL-T inference가 필요한 상태**다. 새로 저장할 확률·키·label·source hash·device/runtime을 독립 검토 후 등록해야 한다. 이 문서에서는 checkpoint/배열을 로드하지 않았다.

## 6. 판정·다중비교·강건성 계약

### 6.1 유효성과 주 비교

기존 G15/G0 저장 예측 재현은 P48과 같은 지표 항목에서 절대 오차 ≤1e-9, B1/periodic branch 재현도 기존 집계 대비 ≤1e-9를 요구한다. 완성 branch를 재구성하는 단계와 새 후보 생성 단계를 구분한다. 실패하면 새 점수·순위를 산출하지 않고 원인을 기록한다. 실제 새 장치 추론의 허용오차는 별도 재현 gate로 등록해야 하며 저장 예측 허용오차를 임의 확대하지 않는다.

주 비교 family는 **네 개**: B2−G15, B2−B1, B3−G15, B3−B1. 각 비교는 Cpanel과 MLB 모두 통과해야 한다. 각 분할의 단측 NLL p를 계산하고 비교별 `max(p_Cpanel,p_MLB)`로 결합한 네 p에 Holm α=.05를 적용한다. 겹치는 DEV를 독립 검정으로 곱하지 않는다. 미실행/실패 비교는 p=1로 family에 남기고 사후 family 축소 금지다. B3 완료 전 B2의 최종 family 판정은 보류한다.

**N 문턱은 네 비교에 동일하게 유지:** ensemble ΔNLL≤−0.003, 짝 95% CI 상한<0, Holm 통과, ΔBrier 95% 상한≤+0.001, 같은 seed 비교 4/5에서 ΔNLL<0. G15 대비만 통과하면 그 사실만 보고하고 B1 대비 추가 개선이라고 쓰지 않는다. 새 연구 후보 추천에는 G15와 B1 양쪽 N을 요구한다. 작은 보정 개선이 N을 넘지 못하면 `INCONCLUSIVE_N`으로 남기며 문턱을 낮추지 않는다.

B3의 원안 보정 전용 기준(ECE 개선+NLL 비열등)은 이번 기본 N을 대체하지 않는다. ECE10·클래스별 calibration은 기술 보고이며 그것만으로 N 실패를 뒤집지 않는다. 보정 전용 목적을 주 기준으로 바꾸려면 새 결과 열람 전에 별도 결정·등록이 필요하다.

### 6.2 불확실성·보고

- 고정 예측에 조건부, 경기 단위 paired bootstrap 10,000회, RNG seed `20260924`. 경기 내 모든 행을 함께 재표집하고 총 loss 합/총 투구 수로 계산한다. NLL/Brier는 float64; seed를 경기 표본에 합산하지 않는다. 같은 split/비교에서 같은 경기 resample을 쓴다.
- NLL/Brier·ECE10(기존 10개 equal-width bin 규칙)·클래스별 reliability/지원 수·seed 차이·CAL 선택값/전체 grid·유효성·시간을 모두 보고한다. 클래스 사건<30이면 희귀 클래스 해석은 미확정이며 빈 bin/0사건을 성능 0으로 채우지 않는다.
- 모든 CI에 `repeated DEV; conditional on fixed predictions; fit/CAL/model-selection uncertainty excluded`를 붙인다. 두 후보의 DEV 재평가 자체가 새 노출임을 기록한다. 신선한 독립 확인·정책 가치·서비스 승격 주장은 불가하다.

### 6.3 R과 지원

기존 `matrix_group_metrics.GROUPS`의 12집단을 그대로 사용한다: starter/relief, L/R, TRAIN volume low/middle/high/zero, two_strikes/less_two_strikes, runners_on/bases_empty. 역할은 해당 경기 starter_pitcher와 pitcher 일치, 좌우는 p_throws, 2스트라이크는 투구 전 strikes=2, 주자는 투구 전 bases>0. TRAIN 양수 투수별 투구 수의 global linear q25/q75에서 low≤q25, middle≤q75, high>q75, TRAIN 미등장 선수는 zero다. 경계는 동결 panel.json에서 가져오며 DEV로 다시 계산하지 않는다.

- 보고 적격성: 집단당 30경기·500구 이상. unknown metadata 행은 삭제하지 않고 전체 분모/unknown 개수에 남긴다. 미달은 `R_UNCONFIRMED`/null이고 집단을 family에서 빼지 않는다.
- 네 비교×두 split×12집단×2손실=**192개** 사전 단측 비열등 상한. Bonferroni α=.05/192, 경기 bootstrap **100,000회**, seed `20260924`; 각 집단 ΔNLL 상한≤+.010 및 ΔBrier 상한≤+.002. 개별 95% CI도 별도 병기하되 동시 상한을 대신하지 않는다.
- 기존 helper는 candidate_family_size 2/4를 받지만 split 간까지 자동 보정하지 않는다. 기존 `bootstrap_difference(..., upper_alpha=.05/192, draws=100000)`를 재사용하는 새 wrapper와 회귀 검증이 필요하다. 기존 등록용 helper의 기본 family를 바꾸지 않는다.
- 12집단 모두 측정/통과해야 해당 비교의 R PASS. 하나라도 부족하면 전체 R은 미확정, 하나라도 실패면 FAIL. 지원/기권/fallback 분모와 비용이 미확정이면 광범위 후보 채택을 권고하지 않는다. P48의 과거 R 결과를 소급 변경하지 않는다.

## 7. 실행 예산 초안과 중단

아래 숫자는 **검토할 강제 상한 제안**이며 예상 소요시간이 아니다. 새 벽시계·GPU시간·RSS·저장량·추론 p50/p95는 모두 **미측정**이다.

| 단계 | 제안 상한 | 실행 전 조건 |
|---|---|---|
| 현재 문서/metadata 감사 | 새 backbone fit 0, inference 0, 재채점 0, GPU 0 | 지금 수행한 범위 |
| B2 저장 확률 재구성·CAL | a=5점×(ensemble+5seed)=30회 objective 평가, fit 0, 단일 CPU 작업 벽시계 1시간 | 소유권·입력 pin·검토·실자료 읽기 등록 |
| B3 CAL-T inference | 기존 TabM checkpoint 5개 각 1회, 새 backbone fit 0, 우선 A5000 1장·동시 1작업; 작업당 30분·총 벽시계 2시간 상한 | exporter/고정 runtime·재현 gate·권한·장치 lock |
| B3 온도 적합 | 온도 파라미터 1개, 최적화 평가 횟수/허용오차는 §8에서 고정 전 실행 금지 | CAL-T만 접근하는 loader 검사 |
| 통합 N/R 채점 | 주 비교 4개·split 2개, N 10,000/R 100,000 bootstrap, CPU 직렬 벽시계 4시간 상한 | R metadata 키/분모 고정, 합성 검증 |
| 공통 자원 | 새 출력 합계 5 GiB, 프로세스 RSS 8 GiB 상한 제안; 디스크 여유 24 GiB 미만이면 새 작업 시작 안 함 | 기존 공통 .heavy.lock·장치 lock·운영 보호 조건 중 더 엄격한 조건 우선 |

상한 소진·pin/키/라벨/지원 불일치·2026 행·잘못된 확률·재현 실패는 중지하고 실패 산출물을 별도 회차에 보존한다. 예산 연장은 결과를 보고 정하지 않는다. 코드 버그 재시도는 원인·커밋·노출을 기록하고 독립 검토 후 새 실행 계약으로 제한하며, 동일 실패를 무제한 재실행하지 않는다. 목적 없는 10fit 예산 소진이나 없어진 checkpoint 재학습은 하지 않는다.

A5000 5장은 허용 상한이지 예약이 아니다. 이번 B2는 GPU가 필요 없고 B3도 새 학습 없이 1장부터 검증한다. 5장 병렬화는 이 초안에 없으며 필요 시 공유 CPU/RAM/IO 실측 후 별도 자원 계획을 고정한다. 실행 직전 `gpu st`, 실행은 `gpu run`, 장치당 1작업·공통 lock·다른 소유자 프로세스 보호. D192 A2b의 밤 창/자원과 중복 점유하지 않는다. 11/20 전시 동결을 연구 일정 때문에 미루지 않는다.

## 8. 실행 전 블로커와 완료 증거

| 게이트 | 현재 부족한 정확한 항목 | 닫는 증거 |
|---|---|---|
| G-OWN | P48/G15 원본·산출물 활성 소유권 및 새 읽기 범위 합의 | root의 범위 확인, 별도 새 출력 경로. 기존 결과 덮어쓰기·대리 RELEASE 없음 |
| G-ID | 실제 EXP ID·실행 등록 커밋·작성자와 다른 검토자 | 최신 ID 충돌 조회, 고정 커밋 검토, 실행 등록. 이 DRAFT 이름을 실행 ID로 사용하지 않음 |
| G-PIN | periodic 전체 35파일/state source closure, G15→G0→P11 사슬, B1 모든 downstream 계수의 정확한 값·hash | 소유자 제공 manifest 또는 승인된 읽기에서 pin 검증; metadata 존재 확인만으로 PASS 불가 |
| G-DATA | CAL/DEV probability·label·key 검증은 미실행; 전체 요청 분모·panel/R metadata hash 미확정 | 허용 파일·역할별 접근 manifest, split key hash·label·정렬·제외 사유, 2026 거부 검사 |
| G-B3 | CAL-T raw 예측이 기존 archive에 없음; runtime/device 재현 허용오차 미고정 | 5checkpoint inference 전용 경로·출력 계약/sha·재현 기준 독립 검토. 원자료·새 fit 없이 가능한지 확인 |
| G-CODE | 새 B2 혼합, B3 POST/온도 optimizer options, Holm 4개 IUT·R 192개 보정 wrapper 미구현 | 기존 helper 재사용, synthetic tiny check: a=1=B1, label permutation 거부, T=1 identity, member/seed 평균 순서, unknown group, 미실행 p=1, split 보정 검증 |
| G-BASE | G15/P48 저장 지표 재현과 B1/periodic 계수 복원 미실행 | 새 read-only 재현 보고; 기준 불일치 때 후보 채점 안 하는 로그 |
| G-BUDGET | 제안 상한의 운영 적합성·optimizer maxeval/허용오차·실행 시작/종료 창 미확정 | 독립 검토 후 정확한 숫자·장치·lock·출력 quota 고정. 원화/시간 성능은 측정 전 미측정 |
| G-GO | 현재 승인은 문서·근거 감사; 실자료 확률 읽기·inference·채점 실행 미개시 | 위 게이트와 실행 범위를 충족하는 후속 지시/등록. G0/서비스 교체와 별개 |

권장 다음 단계는 **이 초안 독립 검토 → 소유자 manifest·분모 확인 → 합성 구현 → 등록**이다. B3 CAL-T 파일이 없다는 이유로 즉시 GPU를 쓰거나, 미측정 R을 집계보고서에서 추정하지 않는다. 실제 실행 전 충분히 구체화되지 않은 항목은 위 표에 남겨 두며 기계용 JSON의 `enabled=true`를 대신 만들지 않는다.
