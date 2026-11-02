# GPU14h 세 번째 배치 — 현재 구종 예측

현재 `EXP-GPU14H-003`는 DRAFT/실행0이다. 기존 현재구종 조건부 결과모델과 달리 현재 구종을 예측하는9그룹 연구다. 원패키지에서투구전state11만남기고현재구종·그에조건부인physics/transfer/모든hinge를제거한다.이전32구이력만추가해MLP/Transformer/LSTM을고정빈도와비교한다.

기존패키지는7leaf outcome적격행으로선별된Q_pkg이며전체투구Q가아니다.모든Q_pkg에예측하되16raw code→9그룹에매핑되는E9_pkg만채점한다.UNKNOWN/PO/UN과지원밖·신규투수를분모/진단으로보존하고현재정답으로추론을gate하지않는다.실시간수신가용성은UNKNOWN이다.

첫9fit(3후보×3월×seed0)뒤완결후보중NLL개선/Brier비악화최대2만추가seed를확인해GPU상한21fit다.α4 global→pitcher→count빈도는fold별FIT에서3countfit후동결한다.새CAL/FINAL/6경기평가·운영교체는없다.동일2026노출개발자료의탐색이며독립확인이아니다.

[정확 사전등록](/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch3/REGISTRATION.md), [계획](/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch3/PLAN.md), [생성검사](/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch3/synthetic.log)를참조한다.001/002원본을보존하며전체14시간자원·마감예산을공유한다.
