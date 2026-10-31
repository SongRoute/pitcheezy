# 연구 PM 현재 상태

관측 2026-10-08T13:34:51.135812+09:00. [현재 후속 2·4·5·6 보고](/Users/song/Projects/pitcheezy-pm-jobs/20261008-coherent-service-001/research/REPORT.md). 새 4 CPU 조건부 fit / 4 CAL 온도 완료, 추가 학습 없음.

- 고정 10결과 부모 아래 헛스윙|strike·땅볼|인플레이 모델을 두 기간에 학습했다. 보정 후 실제 7사건 합·포함 관계 위반 0. 모든 점손실 개선이나 early 헛스윙 NLL과 late 땅볼 Brier 구간은 0 포함. 개발 자료이며 땅볼 비교는 50%/100% 패키지 차이가 있다.
- 실제 610경기/180,774투구 키를 여섯 아카이브와 대조해 비중복 확인. 이전 학습/특징 계보가 남아 미노출 인증은 미완료다. 범위 밖 과거 OPE 요약의 우발 열람을 [사고 기록](/Users/song/Projects/pitcheezy-pm-jobs/20261008-coherent-service-001/research/exposure/INCIDENT.md)에 공개했으며 해당 수치를 사용한 학습·선택은 없다.
- 기존 유사도 실험의 58개 고정 집단을 fit0으로 분해했다. RE24 전이 가치 우선안·별도 WP 축과 가상 산술 예제를 준비했다. 실제 가치 정책을 채택하지 않았다.
- [모델 레지스트리](/Users/song/Projects/pitcheezy-pm-jobs/20261008-coherent-service-001/research/coherence/MODELS.json). 조건부 head를 서비스 격리 추론에 인계했다. 운영 교체·baseline 변경 없음. 이전 44fit/8fit와 이번 4fit를 구분한다.
- [작업 전 상태](history-2026-10-08-before-coherent-service.md). 통합·서비스 상세는 해당 상태판을 따른다.
