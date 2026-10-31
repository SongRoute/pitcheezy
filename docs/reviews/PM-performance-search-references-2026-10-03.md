# 성능 탐색 참고 문헌과 적용 가설

2026-10-03, Codex PM. 원문 초록/공식 논문 페이지를 직접 확인했다. 아래는 우리 데이터에서 입증된 효과가 아닌 개발 가설이다. D165에 따라 새 Claude 작업은 Opus 5.5만 사용한다.

| 원문 | 확인한 아이디어 | 이번 적용 / 한계 |
|---|---|---|
| [Guo et al., ICML 2017, On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | 분류 확률의 신뢰도 보정을 위한 스칼라 temperature scaling | 기존 예측의 T 7개 비교. 논문은 주로 이미지/문서 분류이므로 야구 효용 개선 근거로 인용하지 않는다. 보정용 자료와 평가용 자료를 구분한다 |
| [Lakshminarayanan et al., NeurIPS 2017, Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | 여러 네트워크의 예측을 함께 써 확률·불확실성 추정 | 기존 G0·직접 head 혼합비 탐색의 참고. 서로 다른 두 예측기를 가중 혼합하는 이번 실험은 원 논문의 재현이라고 부르지 않는다 |
| [Gorishniy et al., NeurIPS 2021, Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) · [공식 구현](https://github.com/yandex-research/rtdl-revisiting-models) | 공정한 공통 비교에서 ResNet형 기준선과 tabular Transformer 검토. 모든 데이터셋에서 우월한 방법은 없음 | 다음 학습 모델 축의 후보. 이미 진행한 P0~P16 모델과 중복인지 확인하고 비교 예산을 고정한 뒤 구현한다. 기존 모델의 이름만 바꾼 중복 실험은 제외 |

## 첫 묶음의 고정 계획

- 봉인된 <=2025 G0/직접 head의 동일 키 예측을 재사용한다. 5개 seed, Cpanel/전체 MLB 두 범위를 유지한다.
- 직접 head 비중 {0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1}, temperature {.8,.9,.95,1,1.05,1.1,1.2}: 77조합. 결과를 보기 전 설정 파일에 고정한다.
- P16의 G0/직접/0.5혼합 기준을 먼저 재현하고, 기존 June 빈도 혼합·support 의미를 유지한다. 하나라도 재현 불가하면 새 점수 해석 전에 원인을 해결한다.
- 이미 노출된 DEV에 대한 탐색 통계이며 독립 확인·새로운 공식 성능 통과 판정이 아니다. 77개 중 좋은 행만 보고하지 않는다. 정식 모델 채택·서비스 적용 없음.
- 2026 자료는 이 탐색에서 열지 않는다. 새로운 학습 실험은 기존 TRAIN/earlystop/calibration 계약을 유지한다.
- 새로운 등록·소스·설정·입력 해시와 실패/실행 시간을 남긴다. GPU를 쓸 필요 없는 후보는 CPU에서 먼저 검사한다.
