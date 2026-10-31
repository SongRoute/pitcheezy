# 선행연구의 역할과 확인 상태

**최신 문헌 검토(2026-09-24):** [ML0·ML1 실행 전 문헌 검토](../../docs/reports/ML-literature-review-2026-09-24.md). 아래는 초기 확인 기록이다. SmartPitch·Melville의 원문 미열람 상태는 이후 해소됐으며, 현재 해석은 새 검토와 [연구 차이 검토](docs/RESEARCH_GAP_REVIEW.md)를 따른다.

2026-09-21 확인. 사용자 제공 요약과 원문에서 확인한 내용을 구분한다. 다음 목록은 초기 비교 설계이며 완전한 문헌조사는 아니다.

## SmartPitch — Otremba, MIT MEng, 2022

- 원문: [MIT DSpace](https://dspace.mit.edu/handle/1721.1/145144).
- 이번 확인 상태: DSpace 원문 열람 실패. [지도교수의 공식 CV](https://www.chrisrackauckas.com/assets/Professional/ChrisRackauckasCV.pdf)에서 제목·저자·2022년 학위논문 정보를 확인했다.
- 상세 방법은 사용자 첨부 요약에 근거한다: 카운트 MDP, 4분류 결과 모델, RE/xwOBA 보상, 구종×위치 질의와 동적계획법.
- 사용 목적: 원문 확인 후 단순 MDP 기준선. 동일 데이터·정보 조건으로 바꾼 버전은 ‘SmartPitch-inspired’라고 표시하고 원문 충실 재현과 구분한다.
- 첨부 체크리스트의 CE≤0.861/Brier≤0.482를 새 데이터 성공 조건으로 채택하지 않는다. 입력 77차원 주장과 나열된 피처 구성의 대응, 보상 식·계수·분할은 원문에서 확인할 항목이다.
- ‘예측 손실이 작아지면 정책이 반드시 좋아진다’는 결론은 채택하지 않는다. Bellman 최적성은 주어진 모델 안의 성질이며, 정책이 선택하는 영역의 오차와 행동 지지를 따로 검사한다.
- 첨부 요약: `/Users/song/.codex/attachments/83723ff3-0fbc-4fda-84f2-dace003aea1b/붙여넣은 텍스트.txt`.

## Douglas, Witt, Bendy, Vorobeychik — 2021 공개본

- [arXiv 메타데이터](https://arxiv.org/abs/2110.04321), [PDF](https://arxiv.org/pdf/2110.04321).
- 원문 방법·평가 절 확인. 목표 위치/도달 위치, 스윙 반응을 모델링하는 투수–타자 확률 게임이며 OBP를 다룬다.
- 보고된 OBP 0.329→0.242는 관측 OBP와 모델 균형 OBP의 비교다. 실제 경기 개입으로 25% 개선을 입증한 수치로 사용하지 않는다.
- 역할: command와 상대 반응을 포함한 전략 기준선 후보. WE 목적과 다르므로 원 목적의 재현 평가와 공통 WE 평가를 분리한다.

## Takamido & Nakamoto — 2026 공개본

- [arXiv 2606.17345](https://arxiv.org/abs/2606.17345), [PDF](https://arxiv.org/pdf/2606.17345).
- 첫 arXiv 공개는 2026-06-15. 사용자 요약의 ‘2025’는 확인한 공개본의 인용 연도와 다르다. 별도 2025 판본이 있다면 그 판본을 명시한다.
- 원문은 최종 결정구의 in-play/swing-out 예측 및 최종/셋업 투구의 대체를 다루고, 시즌 지표 효과는 모델 출력과 시즌 통계 사이 회귀로 추정한다.
- 역할: 시퀀스 인코더와 2스트라이크 작업의 비교군. 모든 카운트의 타석 정책으로 확장한 버전은 원문 그대로의 재현이 아니다.
- 셋업을 바꾸고 나머지 실제 시퀀스·맥락을 고정하는 반사실 질의와, 결과에 따라 분기하는 타석 정책을 구분한다. 보고된 K/9 개선을 현실 정책 개선의 정답으로 사용하지 않는다.

## Mott, Bradshaw, Grimsman, Archibald — 2025

- [arXiv](https://arxiv.org/abs/2511.17733), [HTML 원문](https://arxiv.org/html/2511.17733v1).
- 원문 방법·평가 절 확인. 타석 결과 분포와 다음 주자/아웃 상태 분포를 결합하며, 감독 수준 결정을 승리 확률로 평가한다.
- 시뮬레이션은 가장 상세한 BR 모델을 ground truth로 두고 비교한다. 약 1승/162경기라는 결과는 그 환경과 가정 아래의 추정이다.
- 최근 기록 추가가 예측 손실과 승리 가치에 서로 다른 영향을 준 결과는 두 지표를 분리해야 한다는 참고 근거다.
- 역할: WE continuation·상태 전이·검증 설계 참고. 투구별 전략 모델의 직접 성능 기준선은 아니다.

## Melville et al. — Sloan 2023, 추가 검토 대상

- [공식 발표 페이지](https://www.sloansportsconference.com/research-papers/a-game-theoretical-approach-to-optimal-pitch-sequencing).
- 공식 초록 확인: 투수–타자 제로섬 게임으로 투구 순서와 균형 전략을 다룬다. PDF 링크 추출은 이번 도구 호출에서 실패해 상세 구현은 확인하지 않았다.
- 역할: 시퀀스·게임이론 부분의 선행 기여 점검. ‘시퀀스 최적화 최초’ 등의 주장을 피하고 상세 비교 필요.

## 평가·알고리즘 방법론

- [Jiang & Li (2016), Sequential DR](https://proceedings.mlr.press/v48/jiang16.html): 순차 정책 평가 후보. 이론의 전제와 현재 자료의 관측 행동·교란 조건을 구분한다.
- [Voloshin et al., OPE empirical study](https://arxiv.org/pdf/1911.06854): 평가기 간 비교 및 support·교란·전략적 환경 한계의 근거. 평가 방법 하나를 무조건 정답으로 취급하지 않는다.
- [Kumar et al. (2020), CQL](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html): offline RL 비교 후보. 잠재 목표 위치를 관측 행동처럼 취급하는 문제를 해결해 주는 알고리즘은 아니다.
- [NIST, 표본 수·검정력](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm): pilot 기반 최소 검출 효과 계산의 참고. 정규 근사를 중량 OPE에 검증 없이 적용하지 않는다.

## 초기 비교의 우선순위

1. 행동 모방, 같은 전이 모델의 myopic/MDP, neutral/RE24/WE 목적 비교로 하네스를 만든다.
2. SmartPitch-inspired 기준선과 B/C/D 절제를 공통 조건에서 비교한다.
3. Takamido의 sequence 조건과 Douglas의 목표/도달 위치·상대 반응을 추가 비교한다.
4. Melville의 상세 설계와 더 넓은 선행연구를 확인해 논문 기여 주장을 확정한다.
5. Offline RL은 같은 정보/행동 의미에서 비교할 수 있을 때 추가한다.

기존 pitcheezy의 B1은 원문 완전 재현이 아니었고, B2도 11분류 등으로 변경한 구현이다. 기존 코드·결과를 가져올 때 정확한 변형 내역을 함께 기록한다.
