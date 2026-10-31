# 통합 PM 현재 상태

관측 2026-10-08T12:55:43.675320+09:00. 승인된 후속 연구·격리 서비스 실행을 마쳤고 독립 검토를 통과했다. 신규 실행은 없다. [통합 1–8 결과](/Users/song/Projects/pitcheezy-pm-jobs/20261008-research-service-next-001/REPORT.md).

- 연구: 새 8 CPU classifier / 8 CAL T 완료, 실패 0. 기존 generic 비교 기준을 유지하며 이번 고정 물리 유사도 가중의 개선은 확인하지 못했다. 이전 44 fit과 구분한다.
- 확률 진단: 헛스윙 확률이 포괄적인 strike 확률보다 큰 개발 EVAL 행이 약 1.8–1.9%다. 서비스 실험은 10결과 head만 사용하며 별도 head를 결합 확률처럼 표시하지 않는다.
- 서비스: 격리 코드 `750f1a4`, 51개 테스트 통과. 원자료 재구성 후 실제 모델의 조건부 분포 544개를 생성했고 관측 32개를 정확히 재현했다. 투수별 실행 가능성·Top3·라이브 가용성 검증과 배포는 남았다.
- 포수 셋업/실제 위치 구분 계약·어댑터와 팬 설명 예시·체험 가이드는 준비했다. 실제 영상 보정·팬 이해도 검증을 완료했다고 주장하지 않는다.
- 미완료: 미노출 pre2026 평가 인증. 이는 사용자 승인 부족이 아니라 과거 사용 계보 증거 부족이다. [체크포인트](/Users/song/Projects/pitcheezy-pm-jobs/20261008-research-service-next-001/CHECKPOINTS.md).
- 정리: 연구·서비스 소유 프로세스 종료, 공통 잠금 반환, 기록기 idle, 서비스 health 정상. [최종 영수증](/Users/song/Projects/pitcheezy-pm-jobs/20261008-research-service-next-001/service/CLEANUP.json). 실패한 첫 서비스 재구성 기록은 보존했다.
- 보존: 정본 HEAD `f7e8f47`에 병합·배포하지 않았다. 운영 8766/DB·논문 baseline·2026/OPE 변경 없음. 기존 dirty 파일과 봉인 결과를 유지한다.

[연구 state](../research/state.md) · [서비스 state](../service/state.md) · [이번 작업 전 상태](history-2026-10-08-before-research-service-next.md).
