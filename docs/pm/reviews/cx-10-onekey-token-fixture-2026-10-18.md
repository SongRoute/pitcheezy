# CX-10 — 정상 입력용 합성 bearer 형식 정합

상태: 최소 수정·정적 sanity 완료, 독립 정적 검토 PASS/root 최종 수용. 외부 wrapper 읽기·load·실행0회.
작성자 `coordination_docs`, 검토자 `coordination_review`; CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249037513 .

[Claude 요청](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248947418)에 따라 `tests/coordination/test_onekey_acceptance.py:100`의 정상 scenario 가짜 bearer만 교체했다. 기존21자는 gateway 정상형식 `[A-Za-z0-9_-]{43,128}`보다 짧아, N23의 첫 요청 전 형식 거부를 강화하면 정상 control 자체가 잘못된 입력이 된다.

변경값은 `cx04-fake-bearer-only-0123456789abcdefghijklm`(45자)이며 실제 발급 토큰·secret이 아닌 공개 가능한 합성 문자열이다. cookie와 저수준 probe의 가짜 값은 건드리지 않았다. 기존7개 test method·기대값·4개 contract assertion·네트워크/프로세스 guard·helper 흐름은 그대로다. 새 malformed negative 검사는 추가하지 않았다. 원본 wrapper 수리는 Claude 소유다.

| 고정 근거 | 값 |
|---|---|
| 변경 전 git 기준 | `e57fa64d34041f6638b3dda8c0cd398ea352e1bd:tests/coordination/test_onekey_acceptance.py` |
| 변경 전 test SHA-256 | `f687bc9f4f4efd86a43cdbfc08f63a42cef00e2b782cf75f0fc1e94095f1d855` |
| 새 test SHA-256 | `339f936e110367f9e8aeb448fa3dab20fd36383dbf77ff2e67493c66ae2e9618` |
| 후보 commit | COORD-04 선별 커밋 후 큐 DONE에 기록; 위 SHA로 커밋 전 본문을 식별 |

검사: Python 표준 `ast.parse`로 파일을 import/실행 없이 읽고, 해당 문자열 literal1개·정규식 fullmatch·45자·test_ 함수 정의7개를 확인했다. 새 literal을 이전 값으로 되돌린 bytes의 SHA가 변경 전 SHA와 완전히 같아 한 줄 외 assertion/guard/흐름 보존을 확인했다. `git show`의 고정 base 파일 SHA도 일치한다. 검사 스니펫 첫 실행의 불필요한 길이44 단정은 실패했고, 실제 길이45를 확인하여 요구범위43–128 검사로 재확인했다. 테스트 파일은 재수정하지 않았다.

이 결과는 **문법·합성 입력 형식·diff 보존 검사**다. 기존7사례를 이번에 재실행하거나7 PASS를 확인한 것이 아니다. 변경 중인 외부 v4 wrapper/registration/review를 읽지 않았으며, 검토·고정된 후보 쌍 인계 뒤 CX-09에서 실제 회귀 결과를 대조한다. CLI opt-in, source SHA 검증, 가상 입력/시계·격리 실행 경계는 기존 CX-04대로 유지한다.

graphify: `tests/coordination`에서 `graphify update .` 1회 exit0. AST topology 변화 없음으로 기존 ignored 출력은 그대로이며 전체 저장소/의미 추출은 실행하지 않았다. 제품 코드·실자료·운영·네트워크·자격 접근은 없다. 정확 test/report2파일만 COORD-04가 커밋·인계한다.

같은 모델의 별도 에이전트 `coordination_review`도 AST·형식·45자·7개 함수 정의와 literal 역치환 SHA를 직접 대조하여 PASS했다. 검토자 역시 외부 wrapper 읽기/실행과7사례 재실행은0회다.
