# SVC-006-OUTSIDE-ZONE-20261006 — 존 밖 위치 후보 연구 검토 요청

- 보낸 PM: 서비스 PM. 받는 PM: 연구 PM. 작성: `/root/service_readiness`(서비스 측 단일 기록자), 2026-10-06 11:10 KST.
- 상태: **인계 요청 / 연구 PM 미인수**. 이 파일 생성은 전달 완료·연구 착수·정책 승인을 뜻하지 않는다. 받는 쪽 담당 작업자·모델·인수 시각은 아직 미정이다. 통합 PM이 채팅 간 전달과 소유권 전환을 조정한다.

## 목적과 완료 기준

Song 요청은 구종 추천 위치가 스트라이크존 밖일 가능성에도 대응하는 것이다. 서비스의 최신 UI는 유한한 외부 `target.x/z`를 이미 표시하지만, 현행 추천 생성기는 내부 3×3 후보만 만든다. 연구 PM에 존 밖 action 후보의 과학적 정당성, 좌표 계약, 지원 기준, 검증 경계를 검토해 달라고 요청한다. 연구가 별도 등록된 평가와 검토를 마친 후보·API 의미·한계를 명시해 인계하면 서비스 PM이 실제 데이터 경로 통합 가능성을 검증한다. 현재는 실제 외부 추천 생성·정책 채택·운영 승격 모두 미완료다.

## 고정 기준과 재현 증거

- 실제 8771 서비스 소스: `/Users/song/Projects/pitcheezy-worktrees/codex-ui-live-preview-1005`, `codex/demo-pa-anchors-1006`, `2005702eead0cb5b0e0d2ce15b2655664c32bb04`. 최신 UI 제품 `566ade5`, 기존 G0 서비스 유지. SVC-003 DB 격리 후보 `f1bfc46`은 이 요청의 소스나 시연 서빙 제품이 아니다.
- 서비스 합성 표시 검사: `/Users/song/Projects/pitcheezy-worktrees/codex-svc-outside-zone-1006`, test-only HEAD `9445d27c2b20cbeab80f554a81f59b36e2f78255`; [보고](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-outside-zone/implement-1106/REPORT.md), [초안](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-outside-zone/implement-1106/HANDOFF-RESEARCH.md), [로그](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-outside-zone/implement-1106/zone-geometry.log) SHA256 `1b22ce8735a9ca1d26d8d10d756cb2176f98b1528aaf299837e21ea96f6f0a4e`; [test-only diff](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-outside-zone/implement-1106/test-only.diff) SHA256 `f337f1bc99a2ff59410d1ea135b0ff666b520a2312ec9556222fdaa4b67a91e5`. 작성자와 [별도 작업자 최종 검토](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-outside-zone/review-110500/FINAL-REVIEW.md) 각각 합성 Chromium 371 ok/0 FAIL; 서비스 PM은 이 UI 범위로 수용했다. 실제 외부 추천 데이터 검증은 아님. 기존 오래된 label smoke 84 ok/36 FAIL 및 strict locator 중단이 있어 전체 UI suite green 주장은 하지 않는다.
- 현재 구현 원인: `apps/observer/backend/observer_app/domain.py:12-21`의 `ZONES`/`target_point`가 내부 3×3 중심을 만들고, `scripts/demo_precompute.py:353-372`의 `location_proxy`가 그중 하나를 선택한다. [서비스 계약 검토](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-outside-zone/review-110500/CONTRACT-REVIEW.md)가 최신 UI 진입점·호출자도 확인했다.
- config·모델·자료: 이 서비스 작업에서 변경 또는 새 평가 없음. 2026 봉인 자료/payload는 읽거나 복제하지 않았다. 연구 측 기존 config, seed, source, 데이터 식별자와 검증 게이트는 연구 PM이 소유한다. 여기서 새 연구 모델 식별자나 결과 해시를 만들어 쓰지 않는다.

## 입력·출력과 지원 범위

현 서비스 API Candidate의 `target`은 포수 시점 feet 단위 `{x,z}` 또는 null이다. `service/api.ts`는 유한한 숫자를 수용하며 좌표 범위 자체를 제한하지 않는다. 최신 `service/Zone.tsx`는 외부 좌표를 원위치 또는 가장자리 방향 화살표로 표시하고 원좌표를 유지한다. `target=null` 또는 무효 좌표면 구종 후보를 임의 삭제하거나 `zone_id`로 위치를 만들어 내지 않고 위치 미지원으로 둔다. 1·2·3은 추천 순위이며 zone 번호나 성공 확률이 아니다. 연구의 225-action/5×5 격자에 외부 칸이 있다는 사실만으로 현 ARM-B 서비스 후보가 외부 위치를 지원한다고 결론 내리지 않는다.

## 요청할 연구 판단과 한계

존 밖 후보 공간·원점/축/단위·존 경계, TRAIN 기반 위치근사의 표본 지지(ESS/mass)와 결측·거절 규칙, 기존 후보 대비 예측 및 추천 정책 평가를 구분해 검토해 달라. 특히 후보 확대가 지원률, 추론 지연, 메모리·비용에 주는 영향을 측정하고 복귀 후보/모델을 명시해야 한다. 이 서비스 요청에서 해당 수치들은 **미측정**이다. 2026 자료를 학습·튜닝·모델 선택에 쓰지 않고, 봉인 OPE와 기존 과학 게이트를 유지한다. 승인된 연구 인계 전 서비스가 존 밖 위치를 꾸며 내거나 현재 실제 추천 범위가 넓어졌다고 발표하지 않는다.

## 서비스 영향·복구와 다음 조치

합성 UI 검사는 test-only이며 실제 8771·운영 8766/DB를 변경하지 않았다. 시연 현행 HEAD `2005702`의 source·dist·private exact launch 사본과 해시는 [복구 기준](/Users/song/Projects/pitcheezy-pm-jobs/20261006-service-outside-zone/readiness-110252/REPORT.md)에 있다. 복구 실행은 미검증이다. 연구 PM이 소유자·모델·인수 시각을 명시해 수용하거나, 이유와 필요한 근거를 적어 반려해야 다음 상태로 이동한다. 서비스 PM은 그 결과와 검토 증거를 받아 격리 통합·시연 적용 여부를 별도로 판단한다.
