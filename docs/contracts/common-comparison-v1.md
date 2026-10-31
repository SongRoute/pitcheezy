# 작은 공통 비교 v1 — EXP-A-COMMON-001

2026-09-23. 기존 S1 결과/첫 B 후보 악화는 알려진 상태다. 이 비교의 config·코드는 새 기준선 적합/점수 계산 전에 커밋한다. 기존 고정 4월 TRAIN, 7월 DEV의 같은 145/151 지원 타석을 사용하며 결과로 다시 선정하지 않는다. CV·2026·최종 평가셋·대규모 학습은 제외한다.

## 실행 대상과 같은 조건

1. count/좌우(add-one global→count50): 기존 A baseline과 배열 단위 일치 확인.
2. known-type(count50→type50), pitcher-type(+pitcher100): 기존 `HierarchicalFrequencyBaseline`을 재사용한다. 같은 작은 TRAIN만 적합, 새 온도 보정 없음. 현재 관측 구종은 조건부 예측 점수의 행동 인덱스이며 실제 현재 위치·구속·결과는 입력하지 않는다.
3. 기존 frozen frequency/blend: 원래 큰 TRAIN과 CAL의 학습량 차이를 명시하고 A 저장 예측을 재사용한다. 정책 계산에 필요한 전체 행동 텐서만 기존 Engine으로 생성하고, 관측 구종 인덱스가 저장 예측과 일치하는지 확인한다.

공통 10결과, 기계적 DP 불가능 조건, 같은 563구/라벨/경기, 동일 bootstrap 1000회(seed20260923). 전체/2스트라이크(159구) 예측 NLL·Brier·ECE를 따로 기록한다. 계층 fallback/지원 수, 저장/복원, 시간/RSS·코드/config/자료 해시를 보존한다. 계층 구조·강도는 이전 `STRONG_BASELINE_PROTOCOL.md`에 있던 값이며 이번 DEV로 선택하지 않는다.

## 정책과 가치

구종-only 행동 집합·TRAIN 레퍼토리 기준 정책·동결 WE/진루 분포를 모든 비교에 공유한다. 타석 시작에서 count MDP를 끝까지 풀고, 각 정책을 동결 frequency 및 blend라는 **같은 두 평가 텐서** 아래 교차 평가한다. count-only는 행동 차이를 식별하지 못하므로 임의 첫 구종을 최고 정책으로 표시하지 않고 기준 레퍼토리 정책으로 둔다. 자기 정책을 자기 모형에서 최적화한 대각 값의 기계적 이점을 명시한다.

내부 수비 WE 확률과 기준 정책 대비 %p만 계산한다. 이 텐서는 독립적인 실세계 검증기가 아니므로 인과 정책 가치·OPE·OPE ESS는 null이다. 관측 지원 행/행동 커버리지와 OPE 지원을 혼동하지 않는다. Observer 구종×목표 위치 정책, 이닝 종료 교체 가치, RE24 연구와 별도다. 예측/내부 가치가 좋아도 서비스 채택 조건을 충족한 것으로 간주하지 않는다.

## 선행연구 충실도

SmartPitch의 count-MDP+지도 전이 접근은 [MIT 공식 서지](https://dspace.mit.edu/handle/1721.1/145144)에서 재확인했다. 이 실행은 전이 추정·행동·보상을 현재 서비스 과제에 맞춘 통제 비교이며 논문 충실 재현이 아니다. Takamido & Nakamoto는 [공식 초록](https://arxiv.org/abs/2606.17345)의 in-play/swing-out 예측 및 최종/셋업 투구 반사실 최적화와 현재 10결과 PA WE 과제가 다르다. 2스트라이크 절편을 냈다는 이유로 그 논문을 재현하거나 이겼다고 하지 않는다.

원문 구조의 상세 확인은 기존 `experiments/pitchmdp/docs/RESEARCH_GAP_REVIEW.md`(2026-09-21) 기록을 따른다. 오래된 `docs/baselines.md`의 SmartPitch 본문 미열람 표시는 초기 기록이며 최신 검토와 구분한다. 이번에는 논문 수치를 새로 전재하거나 전 논문 재현으로 범위를 넓히지 않는다.
