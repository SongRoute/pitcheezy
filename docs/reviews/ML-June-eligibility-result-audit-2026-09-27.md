# 전체 June 적격 준비 — Astra 독립 결과 감사

2026-09-27. **PASS, 미해결 차단 사항 0건.** 봉인된 keys·metadata·제외 flags와 기존 P4 Cpanel keys/metadata를 읽어 **352개 확인 항목**을 검사했다. 실행 worker의 집계 함수를 호출하지 않고 별도 코드로 표본·지원·제외·교집합을 재계산했다. 원천 `description/events/balls/strikes/supported_pa/pitch_type/plate_x/plate_z`는 다시 디코드하지 않았다.

## 확인된 결과

- 요청 **115,816구·397경기·552투수**, 적격 **104,970구·394경기·549투수**, 제외 **10,846구**다.
- Cpanel **4,821개 ordered key와 7개 metadata 열**이 P4와 정확히 같다. 패널 제외 391키도 패널 요청−동결 Cpanel 키와 같다.
- 전체 적격−Cpanel의 정확한 차집합은 **100,149구·394경기·529투수**이며 패널 밖 적격과 같은 집합이다. Cpanel과 **110경기**를 공유한다. 요청 키 차집합 110,995구, 패널 밖 요청 110,604구와 구분했다.
- 일곱 제외 flag의 OR가 `~eligible`와 같고, 겹치는 원인별 수/경기 수와 disjoint bitmask를 모두 재계산했다. 제외 10,846구 전체가 unsupported_pa에 포함되며 원인 합을 고유 제외 수로 해석하지 않았다.
- 11개 모집단의 모든 고정 그룹별 투구/경기/투수 수와 500구 **AND** 30경기 판정, 보고된 shared-game 수가 재계산과 일치한다.

| TRAIN-volume | 전체 적격 구/경기 | 적격 차집합 구/경기 | 지원 |
|---|---|---|---|
| zero | 4,788 / 127 | 4,788 / 127 | 양쪽 충족 |
| low | 5,338 / 184 | 4,703 / 173 | 양쪽 충족 |
| middle | 38,546 / 393 | 37,532 / 393 | 양쪽 충족 |
| high | 56,298 / 386 | 53,126 / 382 | 양쪽 충족 |

전체 적격 교차 집단에서는 starter×low **1,657구·24경기**, L×zero **1,087구·29경기**만 지원 미충족이다. 이 판정은 새 보정 성능이나 I2 활성화를 뜻하지 않는다.

## 무결성·비용·노출

코드 C `53ae9fd0b56650cf076a83919dc1d92cfe2f2da7`, 등록 D `5025bd79577180bf90850642234bb1ddd8e5fc42`, config·실행 계획·등록·감독기·완료 ledger의 연결을 확인했다. **7개 산출물의 실제 SHA256**, parquet 순서/키 해시·schema·중복·날짜/분할, **12개 입력/guard의 현재 opaque-byte hash**가 모두 맞는다. 저장소 결과 복사본은 봉인된 worker result와 바이트가 같다. 원자료나 runtime 바이너리를 새로 검증했다고 주장하지 않는다.

공식 worker 비용은 **1.219837540993467초 / 600초**, exit0이다. 내부 시작 이후 결과 쓰기 전까지의 0.9353139579761773초와 supervisor postflight 0.0012692499440163374초는 다른 경계이며 합산하지 않는다. 공식 worker 시작/종료 기록은 각각 1개다. 앞선 shell guard 거절은 도구 출력 관찰로 재구성한 `launch-guard-note.json`에 기록됐다. 원래 guard 로그 파일과 정확한 시각은 없으며, 원본 실행 로그를 보존했다고 주장하지 않는다. 이 기록은 Popen 전 종료를 설명하므로 실패한 worker 실행으로 세지 않는다.

이번 독립 감사의 full-command wall은 **0.9590264998842031초**로 별도 측정했고 준비 비용에 넣지 않았다. 해시·봉인 flag/metadata 집계 검사에 대한 시간이며 신경망·optimizer·추론·bootstrap 실행은 없다. 문서 교차 검토와 작성 시간은 이 명령 wall에 포함되지 않는다.

새 필드 접근 시각 `2026-09-27T14:45:07.391167+00:00`, June/R 범위·8개 새 열·10개 metadata 열·용도와 출력 해시가 추가 노출 원장에 연결된다. 출력 parquet에 원래 결과·좌표 값이나 label 열이 없음을 검사했다. 적격 key와 bool 역시 결과 의존적 노출임을 명시한다. [준비 보고서](../reports/ML-June-eligibility-2026-09-27.md)의 수치·해석과 [추가 노출 기록](../reports/ML-June-eligibility-exposure-2026-09-27.md)을 교차 검토했다. 추론 proxy 약599초는 기존 비용의 선형 환산일 뿐 실제 June 추론 비용이 아니다.

## 재현 근거와 한계

감사 코드는 SSD `coordination/20260927-june-eligibility/audit_sealed_preparation.py`에 보존했다. 구조화 결과는 [ML-JUNE-ELIGIBILITY-REVIEW-v1.json](../../results/ML-JUNE-ELIGIBILITY-REVIEW-v1.json), SHA256 **`362424b5599178af1f7e1498e4e8710e52a3e4e695dddd4777abf6e3136c7500`**다. Worker manifest SHA256은 `f564145deb6b6b58a079766d582573b95154c3515af39fb39b0dba56088bdc88`, result SHA256은 `d00606b8f59d0244cbd8346a09f4b6bf9a2ac45f54be59e6e892891b1095671c`다.

원천 적격 8개 열을 다시 계산한 감사는 아니다. 기존 규칙과 구현의 동등성은 실행 전 소스·합성 검토, 실자료에서는 exact Cpanel 재현 및 봉인 flag 일관성으로 확인했다. 비-Cpanel role/hand를 원천에서 독립 재구성하지 않았다. 결과 클래스 분포·보정 품질·추론 비용은 미측정이며 2026 접근, 독립 확인, 서비스 승격도 없다. 이전 I1/I2 결과는 변경하지 않는다.
