# 측정 정의·비교 범위 (월드시리즈 적응)

소유: 연구 PM(서비스 측정 단위는 서비스 PM과 공유) · 갱신: Root가 새 수용 결정·정의 검토를 넘길 때 · 마지막 확인: Root 동결 상태 2026-10-11T00:11:58Z(09:11:58 KST).

이 쪽은 state 파일이 짧게 가리키는 정의를 한 번만 적는다. 값은 Root 동결 상태 receipt [ROOT-FINAL-STATUS.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-final-status-checkpoint-001/ROOT-FINAL-STATUS.json)(SHA-256 `12a439268b442a4562a15847c697a3a97a30e8e694e45b619717ec89f66df6fa`)에서 옮겨 적었고, 작성자는 배열·raw·모델·결과 payload를 열지 않았다. 아래 어느 항목도 고정된 199-feature source, label, live-support, 서비스 측정 계약을 바꾸지 않는다. Step 2 자료 역할은 **2026 EXPOSED DEVELOPMENT**이며 독립 TEST, 2024 대비 우위, 물리적 투구 전체 커버리지, 자동 서비스 채택은 없다.

2026-10-11 주제 분리: 이 쪽에는 Savant 측정 정의만 남긴다. 함께 적혀 있던 두 주제는 문구·수치·증거 링크 그대로 옮겼다 — OLD–fit100 비교 범위는 [fit100-old-comparison.md](fit100-old-comparison.md), Step 5 합성 모델 연결의 측정 단위는 [서비스 model-connection.md](../../service/model-connection.md).

## Savant 네 필드 정의 (공식 캐시 문서 검토)

출처: 공식 Baseball Savant CSV 문서 [https://baseballsavant.mlb.com/csv-docs](https://baseballsavant.mlb.com/csv-docs)의 캐시 응답 `Sat, 10 Oct 2026 20:35:35 GMT`. 수용된 검토 [CACHED-DOCS-REVIEW.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-official-cached-docs-review-001/worker-001/CACHED-DOCS-REVIEW.json) SHA-256 `899fe623dc08db16140d2e32fa3ae6be3176c284ef89b0dc18976a46882b9811`.

| 필드 | 2025까지 | 2026부터 | 이 문서에서 UNKNOWN |
|---|---|---|---|
| `plate_x`, `plate_z` | 홈플레이트 앞면(front) 기준 위치 | 홈플레이트 중앙(middle) 기준 위치 | 단위, 좌표 원점·부호 |
| `sz_top`, `sz_bot` | 2026 이전 정의 | ABS 정의 | 단위, 경계 포함 여부, ABS 계산 기하 |

- 수용 범위는 해당 문서의 2026 문구까지다. 위 UNKNOWN은 이 페이지로 닫히지 않았다.
- 같은 검토에서 포스트시즌 대진은 지정된 캐시 HTML 경로 오류로 **UNKNOWN**으로 남았다. 관리 정오표 [CACHE-INPUT-ERRATUM.md](/Users/song/Projects/pitcheezy-pm-jobs/20261011-official-cached-docs-review-001/CACHE-INPUT-ERRATUM.md)(SHA-256 `a3921e6ea61a3cb47f63a9c159c1f598aad1db6accea3b064f4fab18fec5b6af`)는 정정된 HTML을 직접 읽지 않았다.
