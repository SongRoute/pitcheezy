# F1: 기존 타자 표현의 현재 모집단 bridge

2026-09-24. G/D2/I1 결과 공개 후 활성화한 개발 비교다. 새로운 절제 대조군의 학습·품질 결과 전에 질문과 비교를 고정한다. 과거 C6 근거를 현재 D100/Cpanel의 독립 재현으로 간주하지 않는다.

현재 G0의 52채널 context는 기본 경기/좌우11개, 타자 성향6개·신뢰도6개·soft membership5개, 정적 TRAIN 투수 표현23개·표본 수1개다. 따라서 현재 타자 입력을 ‘연속 성향만’이라고 부르면 부정확하다. 이 bridge는 **기존 타자 표현17개 전체의 추가 정보**를 검증한다. 과거 frozen config/source/report를 고치지 않고 이 명세로 현재 해석을 명확히 한다.

## 고정 비교

| 항목 | full | masked |
|---|---|---|
| 모델 | 보존된 G0-global seeds0/1/2 | 동일 구조 새3seed |
| context `[0:11]` | 기본 경기·좌우 | 유지 |
| context `[11:28]` | 타자 성향·신뢰도·soft membership | 학습/보정/추론에서 모두0 |
| context `[28:52]` | 연속 투수 표현·표본 수 | 유지 |
| 주 비교 | full−masked, 음수가 타자 표현 묶음의 이득 | 대조군 |

SharingContext의 추가7개 routing 채널은 그대로 유지하고 기존 wrapper가 제거한다. 두 모델은 같은52채널과 파라미터 수/초기화 모양을 갖는다. 마스킹 후 유효 입력 용량은 달라지며 이를 정보 제거 대조라고 표현한다. H5의 과거 물리·구종·결과는 보존하므로 모든 타자 관련 정보를 제거하는 실험이 아니다.

같은 D100 TRAIN1,252,824구, early16,000구, May temperature2,603구, June blend4,821구, DEV Cpanel12,334구/328경기를 사용한다. `flatten_mlp`, width128, 최대30epoch/patience5/batch1024/lr.0005, seeds0/1/2, 400draw를 유지한다. 모델의 optimizer·weight decay 등은 frozen G source 그대로다.

G preparation·모든 사용 member·auxiliary·예측 해시를 확인한다. 정규화·타자 통계·투수 표현·frequency·delivery를 재적합하지 않는다. masked의 May temperature와 June seed/ensemble 혼합만 동일 절차로 새로 적합한다. full은 원래 G0의 정확한 temperature·June 가중치·raw/calibrated/primary 예측을 그대로 재사용하고 재구성 검증한다. conditional softmax→log wrapper도 일치시켜 수치 경로 변경을 피한다.

## 판정

- 주 family는 NLL `full−masked` 한 개다. ΔNLL≤−.003, 경기 paired95%CI 상한<0, one-sided centered p≤.05, ΔBrier95%CI 상한≤+.001, 3seed 중2개 이상 음수를 모두 요구한다. bootstrap10,000회, seed20260924, whole-game 재표집/투구 가중 평균, 고정 예측에 조건부다.
- 기존 G의 저표본 투수 가설을 저표본 타자 효과로 바꾸지 않는다. 주 G 검정은 없다.
- R은 기존 실제 선발/불펜·좌우·TRAIN투수low/middle/high/zero·2스트라이크 여부·주자 여부12집단×2지표×1비교=24개 상한이다. Bonferroni α=.05/24,100,000회, 최소30경기·500구, NLL+.010/Brier+.002 허용폭. 미관측 집단은 미확인이다.
- 타자별 및 D100 TRAIN 양의 타자 표본 수 q25 이하/초과/0의 결과는 기술 통계다. q25는 linear 방식, 경계 동률 포함이며 TRAIN에서만 고정한다. post-cutoff 누적 관측 수와 TRAIN 표본 수를 혼동하지 않는다.
- 반대 방향의 좋은 점 추정치만 보고 masked를 자동 승격하지 않는다. 그런 신호가 있으면 후속 확인 비교를 결과 공개 후 새 개발 가설로 등록한다. full의 N 통과 시만 추가5seed 확인을 검토하며, full G0의3/4는 같은 C1 구성원이 있으면 재사용한다. 확인 전에는3seed bridge screen으로만 보고한다.

## 비용·구현·범위

새3fit, 보존3fit이다. 별도 adapter/runner/scorer로 구현하며 frozen G 소스를 바꾸지 않는다. TRAIN65,536구×2epoch/early2,048구/May64구×400draw의 fresh profile 후 전체30epoch·모든 CAL/DEV·fit/predict 각 로드를 외삽한다. 프로파일 최대600초, fit+predict member 최대7,200초, 준비/profile/실패 포함 신규 family14,400초의 실제 ledger를 적용한다. 전체3개가 완료되기 전 품질 검정을 열지 않는다. 보존 member의 논리적 학습 비용과 신규 지출을 구분한다.

구체 실행 ID/config 및 source hash는 코드 검토 후 첫 실제 준비 전에 동결한다. F4 결과를 기다릴 필요는 없지만 F4에 이 bridge 효과를 전이해 주장하지 않는다. 직접 목표 위치·정책 가치·MLB 전체 강건성·미개봉 확인은 이 비교의 결과가 아니다.
