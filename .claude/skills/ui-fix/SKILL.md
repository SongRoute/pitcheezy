---
name: ui-fix
description: UI PR의 Astra 점검 코멘트에서 FAIL/BLOCKED 항목을 읽어 수정하고 재검을 요청한다. 사용 /ui-fix {PR 번호}
---

# /ui-fix {PR번호}

## 절차
1. `gh pr view {PR번호} --comments` → 가장 최근 "## Astra 점검" 코멘트를 찾는다. 없으면 멈추고 알린다.
2. `CHK-NN FAIL — 근거` / `CHK-NN BLOCKED — 이유` 줄을 전부 추출한다. 요약 줄(PASS n / FAIL n)과 대조해 개수가 맞는지 확인.
3. PR 브랜치 체크아웃. 각 항목에 대해 `docs/ui/{화면}.md`의 원 항목과 Astra의 근거를 나란히 놓고 수정한다.
   - BLOCKED가 "확인 절차 자체가 불가능"이면 코드가 아니라 체크리스트의 확인 절차를 고치고, 그 사실을 커밋 메시지에 적는다.
   - 근거가 틀렸다고 판단되면 수정하지 않고 재검 요청 코멘트에 "CHK-NN 이의: {이유}"로 적는다.
4. 로컬에서 해당 항목만 다시 확인 (390px / 1280px).
5. 커밋 `fix(web): CHK-02, CHK-07 — …`, push.
6. 코멘트: `gh pr comment {PR번호} --body "재검 요청: CHK-02, CHK-07 (커밋 {sha7})"` (이의가 있으면 같은 코멘트에).
7. 보고: 고친 항목 / 이의 항목 / "Astra에 다시 보낼 것: {PR URL}".

## 하지 않는 것
- PASS 항목 건드리기
- 머지. 최종 PASS 전부 확인 후 머지는 사용자가
