# COOP-017 — 정책 구성요소 연결·식별자·≤2025 등록 준비 검토 패킷

2026-09-29. **상태: 독립 검토 미수행(PENDING).** 이 세션은 Codex 감독기 없이 Claude가 직접 재개했다. 사용한 모델은 `claude-opus-5-5`뿐이다. Sol(`gpt-6-sol`)·Astra(`gpt-6-astra`)는 이 Claude 환경에서 호출할 수 없었고, Claude가 Codex를 호출하는 순환 위임은 금지이므로 시도하지 않았다. 아래 검사는 모두 작성자 자신의 검사다. 같은 모델(Claude Opus) 하위 에이전트가 한 번 코드 점검을 했고 major 3·minor 6을 지적해 모두 수정했다(보고서 §4). 이 점검은 독립 검토가 아니다. Sol·Astra는 수정 뒤 최종 bytes를 처음부터 봐야 한다. **이 패킷의 어떤 항목도 PASS로 기록하지 않는다.**

- 보고서: [ML-policy-materialization-prep-2026-09-29](../reports/ML-policy-materialization-prep-2026-09-29.md)
- 계약·설정: [ML-POLICY-MATERIALIZATION-v1](../contracts/ML-POLICY-MATERIALIZATION-v1.md), [config](../../configs/ML-POLICY-MATERIALIZATION-v1.json)
- 기계 기록: [results/ML-policy-materialization-prep-v1.json](../../results/ML-policy-materialization-prep-v1.json)
- 기준 `713c7ea`. 검토 대상 커밋은 기계 기록의 `commits`를 따른다.

## Astra — 핵심 과학 검토 1회 (짧은 요약과 정확한 diff)

코드 diff: `experiments/pitchmdp/pitchmdp/policy_identity.py`(신규), `policy_runtime.py`(support_check·verify_components·build_runtime 진입 변경). 판정할 질문:

0. **상태 변경(D92).** 후보 모드에서 지원 표 누락·빈 마스크인데 연결된 pool이 지원하는 요청, 맥락 행과 요청의 투수·타자면이 다른 요청을 `UNSUPPORTED_EMPTY_SUPPORT`에서 `FAILED_INTEGRITY`로 바꿨다. 맥락 행 SHA는 요청 fingerprint에 들어간다. 분모 의미에 문제가 없는가?
1. **식별자 완전성.** 후보 P3의 행동 법칙을 바꾸는 요소가 식별자에 모두 있는가: BC·지원 표, G0 member·loader·보정, frequency 객체·T, PolicyInputs 전처리(encoder·clusters·어휘·normalizer report), 400 draw pool, C0 WE terminal/cutoff, τ·MC·cap·seed, 연결 객체의 저장소 소스 closure, 버전. budget은 실패만 바꾸므로 제외했다. 요청 생성기(원자료→H_t)는 실행 등록 manifest에 두었다. 이 경계가 타당한가?
2. **D-1 로깅 법칙 적합 모집단.** eligible(`supported_pa`·결과 라벨·좌표) 조건은 결정 뒤 정보이므로 π̂_b 주 적합을 TRAIN 전체 투구(BC-P)로, P8 호환 eligible(BC-E)은 비교로 두는 제안.
3. **D-2 분모.** 모든 정규시즌 PA. P8·G0의 eligible 분모에서 바뀐다.
4. **D-3/R3(c).** 구종 결측·`automatic_ball/strike`를 새 거절 상태로 두고 PA를 sticky 처리하는 제안. 비율은 미측정이다.
5. **D-4/R7.** 알려진 투수로 PA 중 교체되면 평가를 계속하는 제안. 시뮬레이터는 교체를 모형화하지 않으므로 q̂ 오차가 되고 DR은 π̂_b 경로에 기댄다.
6. **D-5/R6와 D-6/R5.** 평가 가능 PA 조건부 점추정과 정책별 [0,1] 최악 경계, 그리고 관측된 다음 행 상태의 C0 WE 보상. PA 사이 사건 불일치 플래그 포함.
7. **D-7/R4b.** ≤2025 리허설의 타자 프로필을 평가 창 시작 스냅샷으로 고정하는 제안. 연결 probe만 rolling을 쓴다.
8. **D-10.** V2 세계에서 M 밖 로깅 행동의 delivery로 JointDelivery tier+fallback을 선언해 쓰는 제안. D89의 V2 전제와 모순되지 않는가?
9. **S2 연결 probe.** 봉인된 `mlb_dev` 순서에서 5개 member 모두 fallback이 아닌 첫 64행, primary atol 1e-6(EXP-P11-001 기준), frequency raw atol 1e-12, 라벨 배열 비열람. 이미 노출된 DEV의 재현 검사로 새 품질 정보를 만들지 않는다는 해석이 타당한가?
10. 노출 표와 "개발 검증일 뿐 독립 확인 아님" 표현.

## Sol — 재현·일반 검토

1. 신규·수정 검사 재현:
   `PYTHONPATH=src:experiments/pitchmdp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider experiments/pitchmdp/tests/test_policy_identity.py experiments/pitchmdp/tests/test_policy_runtime.py`
2. 관련 묶음: 위 두 파일 + `tests/test_2026_policy_contract.py experiments/pitchmdp/tests/test_matrix_policy.py experiments/pitchmdp/tests/test_rollout_policy.py`, `scripts/check_2026_policy_contract.py`의 `all_pass`.
3. 기계 기록의 변이 목록을 임시로 적용해 각각 실패하는지 확인한 뒤 원복한다.
4. 메타데이터 인용 확인: 계약 §1과 config `metadata`의 값·SHA가 G0 번들 pin과 해시가 같은 JSON에서 온 것인지 확인한다(`EXP-P4-001/preparation.json`, `EXP-P11-001/preparation.json`·`frozen_calibration.json`, `EXP-P8-001/preparation.json`·`stages/*/runtime.json`, `minimal-pitch-service-v1/source_hashes.json`·`bundle_manifest.json`). pickle/npz/parquet/pt payload는 열지 않는다.
5. 이 세션에서 payload 접근이 0인지, 기존 봉인 파일·config·attempt가 바뀌지 않았는지(`git diff --stat`, 새 파일만) 확인한다.

## 남는 사항

실제 bind(S2)가 첫 실물 확인이다. 등록 class 이름, report 일치, clusters representation SHA, MPS 추론 뒤 digest 불변은 실제 객체에서 아직 확인하지 않았다. 실행기 `run_policy_validation.py`, 요청 생성기(R2·R3(c)·R4b·R4c·R8), 추정기(R5·R6)는 미구현이다. 실행은 이 검토와 Song 결정, 별도 실행 승인 뒤에만 한다.
