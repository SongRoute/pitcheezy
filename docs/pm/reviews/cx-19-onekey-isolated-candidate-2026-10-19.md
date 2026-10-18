# CX-19 — one-key 실행기 격리 수정 후보의 합성 검증

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 정확 diff·실행 경계·최종 결과 독립 검토 PASS 및 Root의 candidate-only 수용. 후보14method 통과, 검토자의 추가 재실행 없음. Base `da14cf91d89ba4cc51bf6f0ae3d6c999f185bb8c`.
사용자 명시 허용 “격리 수정 후보 작성·검증 허용”에 따른 [PLAN UPDATE](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6255324252) / [CLAIM](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6255324496).

## 허용 범위와 정확 수정

기존 [정적 제안](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249776984)은 작성·실행하지 않은 diff였다. 이번 사용자 허용은 격리 후보에 한해 그 보류 조건을 대체하며 CL02의 원본 소유권·원본 적용·운영 실행 권한을 이전하지 않는다.

원본 `L-onekey/onekey_wrapper.py`의 SHA `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`를 재확인한 뒤 새 전용 `20261019-codex-continuous/CX19-candidate/onekey_wrapper-candidate.py`에 사본을 만들었다. 다음 두 수정(+3/-1줄)만 적용하고 mode0400으로 동결했다. 원본과 과거 읽기 전용 CX15 snapshot은 수정하지 않았다.

```diff
-    if type(clock.get("ntp_offset_s")) not in (int, float) or abs(clock["ntp_offset_s"]) > MAX_NTP_OFFSET_S \
+    if type(clock.get("ntp_offset_s")) not in (int, float) or not (-MAX_NTP_OFFSET_S <= clock["ntp_offset_s"] <= MAX_NTP_OFFSET_S) \
             or not filled(clock.get("measured_at_utc")):
@@
     if log.get("run_id") != pre["run_id"] or "token_fingerprint" not in log:
         raise Refuse("run_log_mismatch")
+    if log.get("target_sha256") != sha256(canonical([pre["target"]["upload_base"], pre["target"]["public_origin"]])):
+        raise Refuse("run_target_mismatch")
```

첫 변경은 exact int/float 검사와 경계 ±MAX를 유지하며 NaN/±Inf를 거부한다. 둘째는 deferred probe가 토큰을 다시 받거나 HEAD를 하기 전에 main run이 기록한 target hash와 현재 pre.target의 canonical hash를 대조한다. 누락 hash도 불일치로 거부한다. 독립 검토자가 원본→후보의 전체 bytes를 대조해 이 두 수정 외 차이가 없음을 확인했다.

## 실행 경계와 실제 결과

기존 `tests/coordination/test_onekey_acceptance.py` SHA `c395c7d42872961361e0923bb0a63f7532927e881e607383d772cf5ce5597d2c`의14method를 모두 그대로 사용했다. 최신 독립 checker SHA는 `e8dcd6e18acbfde8cc52ff00e27ad3bc7ee2e77d2aa5a16c28e2f50621726e82`다. 새 fixture·기대 완화·skip/xfail·repo 코드 변경은 없다.

외부 `run_acceptance.py`는 `python3 -I -B`에서 test/checker/candidate의 exact SHA를 먼저 확인하고, 경로 확장 없이 정확한 local checker만 import하여 기존 test의 CLI를 실행한다. 후보 module은 불변 Acceptance.setUp이 `ForbiddenIO` network/process 차단과 빈 환경을 설치한 뒤 실행한다. 실제 cmd_run/cmd_probe에는 기존 fake HEAD·fake child·가짜 시간/TTY/생성 registration·소유 temporary directory만 사용한다. wrapper의 운영 main·실제 registration·실제 runner·실호스트 transport는 호출하지 않는다. 독립 검토자가 driver·guard·diff를 확인한 뒤1회 실행했다.

```sh
python3 -I -B CX19-candidate/run_acceptance.py
```

실제 `run-001`: **exit0, 14method PASS, 0.062s**, failure/error/skip0. 기존 `test_clock_offset_requires_finite_bounded_number`와 `test_deferred_changed_target_refused_before_head`가 후보에서 통과했다. 정상 control, deferred same-target/재시도 예산, expiry/revocation 미완 판정, 실제 생성로그→checker3method도 통과했다. 기존 tearDown의 ForbiddenIO 위반 목록은 비어 있었으며 원본/과거 snapshot/candidate/test/checker/driver의 실행 전후 SHA가 모두 동일하다.

이것은 **격리 후보의 합성 검증**이다. 원본에서 보존한 CX17/CX18 실패 근거를 삭제하거나 원본14PASS로 바꾸지 않는다. 원본은 여전히53af이며 이번 후보 적용·운영 GO를 의미하지 않는다. subcase 수나 과거 실행을14method에 더하지 않는다.

| 전용 CX19-candidate 산출물 | SHA-256 |
|---|---|
| onekey_wrapper-candidate.py | `05a9cc2066e80bc08a419baf5882d215cfc408dde7f190e77ae7f1981a90db05` |
| candidate.patch | `0d6a85af9b84cbc11c8f7600dba46db199c40e1a956b5bfdadf1c927058fed60` |
| run_acceptance.py | `faef75084ea63fd5e341862006024a13058e067a6486b9624b897c1df103ca74` |
| run-001.stderr.txt | `51a1c744ba0ce1de0857a921126ec946f2f701f8e8303dfc9e411a818cd994df` |
| run-001.pins.json | `a8cf5236594a7cbbf80328c0df981443b451e91b4cac095176c7ddcb6dd908e8` |

동일 폴더의 argv/stdout/stderr/exit/pins와 정확 diff를 보존했다. 후보 및 결과는 외부 산출물이며 repository에는 이 보고서1파일만 추가한다. `git diff --check`와 보고서의 로컬 근거·SHA 연결을 확인했고, 새 제품 테스트·브라우저·자료 수집·build·배포는 수행하지 않았다.

## 소유자 인계와 남은 게이트

CL02 소유자에게 원본53af→후보05a9의 정확 patch와14method 합성 결과를 검토 후보로 인계한다. 원본 반영 여부·정식 source/registration/독립 검토 쌍 갱신은 별도 소유자 절차이며, 기존 CX09 정식 pair를 이 후보로 자동 대체하지 않는다. 실제 target/인증/시험 창/회수·단일 writer·파트너 연결과 운영 승격도 미검증이다. 이 검사는 로그 진위나 운영 대상의 신원을 증명하지 않는다.

독립 최종 결과 검토 후 COORD-05 index 소유자가 새 보고서1파일만 커밋한다. 작성자는 stage/commit하지 않았으며 원본·기존 회귀·실패 증거를 보존한다.
