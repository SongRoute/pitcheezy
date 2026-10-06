# 서비스 PM 상태·실행 큐

초기값: 2026-10-06 09:46–09:48 감사 및 [전환 기준점](../transition-2026-10-06.md). 책임 서비스 PM(Astra), 실제 기록은 지정 작업자. `/root/service_pm`이 이 전환 턴에 관리 인수를 회신하고 반환했다. 첫 작업은 기존 `/root/status_operations`(Astra)이 10:38:37 착수·10:39:05 관측·10:40 완료했고 Root가 보고서와 증거를 읽어 제한된 기준점 점검으로 수용했다. Sol 우선 계획에서 실제 가용 핸들로 변경했으며 새 stage·서버 재시작은 배정하지 않았다.

8771제품566ade5/서빙문서2005702,8766운영 분리. 실제 아이폰·맥북 동시 리허설 미확인. 4경기 수집 대조 완료지만 추천 범위·지연 미달 사례 존재. 기존 SQLite 사용 중, 새 지연버퍼e5b943b는 미통합. CV공개 후x제한 유지.

| ID | 상태 | 다음 실행 | 자원/선행조건 | 완료 기준 |
|---|---|---|---|---|
| SVC-001 기준점·복구 자료 감사 | 완료(제한 범위) | 서비스 PM은 남은 실제 리허설을 별도 배정 | 서버 재시작·새 stage·복구 실행 없음 | Root 수용, 아래 실제 관측 근거 |
| SVC-002 라이브 범위·지연 | 준비 | 4경기 실패/미지원 원인별 분해 | 원본보고서·API계약 | 재현 가능한 문제와 개선 검증 |
| SVC-003 DB 지연버퍼 | 준비 | 미해결항목 해결 후 격리 통합 | e5b943b 검토·DB사본·복구 | 현행 서비스 미변경 상태에서 장애/복원 증거 |
| SVC-004 최신 UI 유지 | 준비 | 최신 합의와 실제 코드 보존, 회귀 확인 | 시연 작업과 단일 작성자 조정 | 과거 시안으로 회귀하지 않음 |

작업 소스는 오래된 ui-frame이 아니라 현재 제품/서빙 작업트리에서 확인한다. [시연 수용](/Users/song/Projects/pitcheezy-pm-jobs/20261006-ui-astra-delivery/ROOT-ACCEPTANCE.md), [DB 검토](/Users/song/Projects/pitcheezy-pm-jobs/20261005-storage-review/REPORT.md).

## SVC-001 수용 증거

2026-10-06 10:39:05 관측: 8771 PID99202, HEAD2005702 clean, 서빙 자산5개 일치, 두 서버 health 정상, ZIP 해시 일치, 복구 자료 존재 확인. **복구 실행·실제 기기/영상/연구실망 리허설은 검증하지 않았다.** Root가 이 제한된 점검만 수용했다. [REPORT](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-readiness/svc-001-baseline-103905/REPORT.md), [evidence.json](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-readiness/svc-001-baseline-103905/evidence.json), [links.json](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-readiness/svc-001-baseline-103905/links.json).
