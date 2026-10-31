# Fullseason T shadow — CPU 구현 동치 확인

2026-10-16. 실행 소스 **43d937f22a6193c5352ef5eab179b6ed9183c744**, branch `codex/ws-fullseason-shadow-1016`. 이 문서는 이후 문서 전용 커밋이며 실행 소스와 구분한다. 기존5seed/10outcome Observer 추천기는 유지했다.

별도 `src/pitcheezy/seq_shadow/fullseason.py`는 검증한 바이트에서 최종3seed T state를 CPU/weights_only/strict로 읽고,199차원 입력과 최대32개 same-game/pitcher strictly-prior 토큰을 검사한다. raw7leaf의3seed 평균에 수용된 단일 calibrator를 한 번 적용한다. `scripts/fullseason_shadow_parity.py`와 기존 감독기의 명시 worker 분기로 입력·자원·프로세스 수명을 제한했다.

생성자료 **17개**(어댑터8 + parity9)가 통과했고 독립 검토자의 같은17개 재실행도 통과했다. 총34개로 합산하지 않는다. raw91→199 저장 인코더 동치는 생성자료에서만 확인했다.

별도 등록·독립 검토·Root grant 후 실제 수용된 최종 weights/encoder/calibrator와 기존 최종 패키지의 고정 **100개 encoded199 query**를 검사했다. 과거 context edge2,093개, reference subset2,193행. 동일 CPU의 기존 campaign 계산과 세 시드·raw평균·보정 결과의 최대 절대 차이는 모두0.0(사전 atol1e-7/rtol1e-6),2.0308초, sampled RSS310,902,784bytes, child exit0/reaped였다. 새 fit/CAL/성능 지표 계산은0이다.

이는 **해당100개 encoded 입력의 동일 CPU 구현 동치**다. 전체 자료 동치, CPU↔GPU 동치, 예측 성능 향상, 실제 raw91/profile/live 문맥 완전성이나 운영 승격을 뜻하지 않는다. 전체 sequence9개 member를 디코드해 현재·미래 행의 outcome 값도 메모리에 들어갔지만 모델 토큰에는 strictly-prior 값만 사용했다. corpus/y7 평가라벨은 읽지 않았고 개별 확률은 출력하지 않았다.

증빙: [최종 인계](/Users/song/Projects/pitcheezy-pm-jobs/20261016-fullseason-shadow-004/implementation/HANDOFF.md), [source freeze](/Users/song/Projects/pitcheezy-pm-jobs/20261016-fullseason-shadow-004/service/SOURCE-FREEZE.json), [실행 receipt](/Users/song/Projects/pitcheezy-pm-jobs/20261016-fullseason-shadow-004/parity/attempt-001/receipt.json), [독립 수용](/Users/song/Projects/pitcheezy-pm-jobs/20261016-fullseason-shadow-004/reviews/SHADOW-PARITY-RESULT-ACCEPTANCE-001.json). 수용 SHA256 `6d14347234090bbb9a0ad9aa88aacd3ef19ab902c4e31683217a55b0d60275e9`.

다음은 별도 날짜별 profile exporter와 실제 수신시점의 raw91/previous-type/과거토큰 연결이다. 생성검사 후 필요한 실자료·캡처 범위를 등록·검토한다. 현재7leaf에서 기존10outcome/WE를 역복원하지 않으며 팬 표시/추천/운영 교체·호스트 연결은 별도 gate다. 기존 비활성 HTTPS 설정과 팀원 회신 요구사항은 [후속 작업지시](/Users/song/Projects/pitcheezy-pm-jobs/20261016-oof-completion-003/planning/NEXT-SERVING-WORK-ORDER.md)를 따른다. 이번 단계는 push/deploy/운영접속을 하지 않았다.
