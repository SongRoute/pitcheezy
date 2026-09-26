# B 추천 경계 v1

`Recommender.recommend(pitch, pa)`의 기존 공개 응답과 캐시는 유지한다. 같은 동결 계산의 전체 지원 후보가 필요하면 `Recommender.evaluate_pre_pitch(pitch, pa) -> PrePitchEvaluation`을 호출한다. 두 메서드는 `pitch.request`, `pa.zone_bounds`, `pa.repertoire_counts`만 추출한다. 현재 공의 actual, 이후 공, PA 종료 상태는 입력으로 쓰지 않는다.

`PrePitchEvaluation`에는 `model_identity`(어댑터·설정 포함), `model_sha256`(동결 `bundle_manifest.json`의 SHA256), `model_version=observer-zone-v1`, `value_spec_version=defense-we-pa-v1`, `baseline_policy_id=observer-repertoire-kernel-v1`가 있다. 후보 `actions[i]`는 `index=i`, 구종, zone_id, 목표 plate-feet 좌표를 가진다. `probabilities`는 `(4,3,1,A,10)`; 결과 순서는 `ball,strike,foul,out,single,double,triple,home_run,hbp,double_play`다. `candidate_values`는 `(4,3,1,A)`의 수비 WE Q, `values`와 `baseline_values`는 `(4,3,1)`이다. `baseline_policy`는 A개 지원 후보 확률, `support_ess`와 `kernel_mass`는 `(4,3,A)`이다. 모두 같은 지원 후보 순서를 쓴다. `recommendation`은 현재 카운트의 기존 공개 응답이다.

미지원 결과는 `status=unavailable`, 빈 `actions`, 텐서·가치 `None`, 사유와 기존 빈 후보 추천으로 표현한다. 지원 판정은 모든 카운트에서 ESS≥20 및 kernel mass≥.01인 기존 규칙이다. Q는 초기 수비 팀의 PA 종료 후 동결 continuation 승리 확률이며 `delta_pp=100*(Q−baseline_values)`다. 이 값은 정책의 실제 인과 효과가 아니다.

`ExecutionDistribution.weights(xz,targets,sigma)`는 교체 지점이다. 현 구현 `ObservedDeliveryKernel`은 TRAIN joint delivery 400 draws를 σ=.45ft Gaussian으로 재가중한다. 이는 관측 도달 위치의 근사이고 학습된 의도나 제구 오차 모델이 아니다. 현재 추천 사용에서는 검증된 이 구현 identity만 허용한다. 실제 목표 개입 분포로 교체할 때는 별도 데이터·지원/품질 검증과 모델 버전 변경이 필요하다.

추천 캐시 identity는 이 경계 모듈의 SHA256과 실행 분포 identity를 포함하고, 입력 키도 실행 분포 identity를 포함한다. 현재 서비스는 정확히 `ObservedDeliveryKernel` 타입만 허용해 같은 identity 문자열을 사용하는 다른 수학 구현을 거부한다. 전체 후보 계산은 공개 추천과 같은 인스턴스 lock 아래에서 실행한다.
