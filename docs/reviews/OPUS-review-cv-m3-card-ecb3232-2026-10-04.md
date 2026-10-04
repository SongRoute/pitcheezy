# Opus 검토 — showcase의 CV M3 근거 카드 (`ecb3232`)

`same_model_review; Codex_review_pending`

2026-10-04 KST. 검토자는 Opus 5.5로, 작성자와 다른 인스턴스다. 범위는 diff `840eeee..ecb3232`(`CvEvidenceCard.tsx`, `cv-m3-evidence.json`, `Showcase.tsx`, `showcase.css`, `tests/cv-evidence-smoke.mjs`, `tests/showcase-smoke.mjs`, 작성자 기록 `docs/reviews/PM-cv-m3-card-2026-10-04.md`)다.
근거 자료(읽기만 함): 작업 지시 `docs/handoffs/CV-M3-demo-work-order-2026-10-04.md`, PM 수령 검토 `docs/reviews/PM-M3-receipt-2026-10-04.md`, 영수증 `results/checks/CV-M3-receipt-recalculation-20261004.json`(main 체크아웃에서 미추적, sha256 `b9b2ab19…f630` 재확인, 수정 안 함).

## 처분

**8771 통합을 막지 않음(non-blocking).** 숫자는 모두 영수증과 정확히 일치한다. 기권은 드러나 있고 링크·영상·추천 연결은 없다. 오버레이·의도·높이를 과장한 표현도 없다. 작업 지시와 글자 그대로 어긋난 두 곳(F1·F2)과 iOS 표시 위험(F3)은 이 브랜치에서 바로 고쳤다(아래 "검토자 변경"). 남은 F4~F9는 문구·테스트 강화 제안이다.

## 1. 숫자 충실도 — 영수증과 대조 (모두 일치)

`python3`로 영수증을 직접 읽고 `per_frame`에서 다시 계산했다.

- 경기별 수치: NBC 849845은 프레임 57, AI 출력 43, 기권 14, 사람 표시 56, 사람 기권 1, `person_only` 13. FOX 823407은 29 / 15 / 14 / 28 / 1 / 13. 합산은 86 / 58 / 28 / 84 / 2 / 26이다. JSON과 화면이 모두 같다.
  - 화면의 `28→26`: "AI가 기권한 28장 중 26장은 사람이 미트를 표시"는 영수증의 `pooled_counts.person_only`=26과 같다.
- 픽셀 차이: 중앙값/90분위가 NBC 2.0616/5.0912, FOX 5.7983/9.4430, 합산 2.5298/6.5115이다. 화면은 소수 한 자리로 2.1/5.1, 5.8/9.4, 2.5/6.5px를 보이며 반올림이 맞다.
- 출력 좌표 차이(ft → 인치): NBC 0.08844/0.17431 → 1.1/2.1, FOX 0.13913/0.26612 → 1.7/3.2, 합산 0.09237/0.20090 → 1.1/2.4. PM 검토의 "NBC 약 1.1인치 / FOX 약 1.7인치"와 같다.
- 카메라: NBC는 `error_status` "measured", RMS 0.1488494…ft → 화면 0.15ft. FOX는 "unmeasured", `rms_feet` null → 화면 "미측정 (카메라 미확인)".
- 예시: 영수증의 `per_frame`에서 다시 계산했다.
  - `849845:10:4`: px 2.3195, 출력 0.092366ft = 합산 출력 중앙값. nearest-rank 순위 29/58이 정확히 이 투구다.
  - `849845:23:6`: px 3.8210, 0.008465ft로 전체 최소(→ 0.1인치).
  - `849845:38:6`: px 1.0440, 0.32226ft로 전체 최대(→ 3.9인치).
  - `823407:50:1`: px 7.3824, 0.139128ft = FOX 출력 중앙값(15개 중 8번째).
  - 앞의 세 키는 영수증 `examples.{representative,best,worst}`와 같다.
- 리뷰 본문에서 인용한 두 값: "경기 전체 605구 중 86구"와 "포구 공 대조 13구"는 PM 검토 본문과 글자가 같다. 영수증 JSON에는 없는 값이며 JSON `source.reviewFacts`에 그렇게 밝혀져 있다.
- 표시하지 않는 `outputHeightSigned`(−0.04789827…, −0.049)도 영수증 `output_z_signed`와 같다(검토자가 테스트에 대조를 추가했다).
- 오해를 부르는 반올림은 없다. 다만 F7(픽셀의 기준 해상도)과 F8(대표 예시 라벨)을 참고할 것.

## 2. 문구 — 작업 지시와 대조

- 요약 세 구절 `평가 2경기 · 86장`, `AI 출력 58 / 기권 28`, `사람과 비교한 58구의 미트 점 차이 중앙값 2.5px`가 있다.
- 명칭은 PM 판단대로 "포수 셋업 추정"을 썼다. "CV 검증 결과"는 작업 지시의 카드 이름이다.
- FOX 카메라 미확인(경기별 줄과 FOX 예시 라벨), 높이 미검증, 의도 미검증, 기권을 숨기지 않음, 판독자 1명, 605구 중 86구 편향, 추천·9구역·예측 모델 미사용이 모두 들어 있다.
- 영상 오버레이·영상 대응·의도를 검증했다는 주장은 없다. 다른 경기의 추천을 붙이지 않았고 `a/button/iframe/video`는 0개다.
- 어긋난 곳은 F1(설명 문장의 단위)과 F2(예시 줄의 면책)다. 둘 다 고쳤다.

## 발견 사항

### F1 (낮음, 고침) 설명 문장이 작업 지시와 글자가 다름
- 위치: `apps/observer/web/src/CvEvidenceCard.tsx:21`
- 내용: 작업 지시의 설명 문장은 "…점 차이 중앙값은 2.5픽셀입니다."인데 카드는 "2.5px입니다."를 썼다. 기존 테스트는 문장 앞부분(`두 경기 86장 중 58장에서 좌표를 냈고`)만 확인해서 이 차이를 잡지 못했다.
- 실패 상황: PM·root가 작업 지시 문장 그대로 화면을 대조하면 불일치로 보인다.
- 수정: `{px(p.pixels.median)}입니다` → `{p.pixels.median.toFixed(1)}픽셀입니다`. 요약 줄은 작업 지시대로 `2.5px`를 유지한다. 테스트는 설명 문장 전체를 확인하도록 바꿨다.

### F2 (중간, 고침) 예시 줄의 인치 값에 면책 문구가 없음
- 위치: `CvEvidenceCard.tsx:37`
- 내용: 작업 지시는 "ft/in 값 옆에 `같은 변환 기준의 판독 차이 · 물리 정확도 아님`을 붙인다"이다. 그런데 예시 4줄(`약 1.1인치`, `0.1인치`, `3.9인치`, `1.7인치`)에는 문구가 없었다. 판단: 필요하다.
  - 예시 줄은 독립된 목록 항목이라 한 줄만 캡처·인용되기 쉽다.
  - 특히 `849845:23:6 … 약 0.1인치`는 "AI가 0.1인치까지 맞힌다"로 읽힐 위험이 가장 크다.
  - 경기별 줄에 적은 문구는 예시 항목에 이어지지 않는다.
- 실패 상황: 발표 화면을 캡처하면 "출력 좌표 차이가 가장 작음 · 약 0.1인치"가 면책 없이 돈다.
- 수정: 예시 `<span>` 끝에 `<em>{SAME}</em>`을 붙였다. 테스트는 "em 개수 == 3" 대신 다음을 확인한다.
  - '인치'를 포함한 모든 목록 항목·문단이 면책 문구를 포함하는지.
  - 그런 블록이 `3 + examples.length`개인지.

### F3 (중간, 고침) iOS Safari에서 기본 펼침 삼각형이 grid 칸으로 나타날 수 있음
- 위치: `apps/observer/web/src/showcase.css:19`
- 내용: Chromium은 `summary`가 `display:list-item`이 아니면 마커를 숨긴다. WebKit은 `::-webkit-details-marker`로 마커를 그리므로 `display:grid`만으로는 숨겨지지 않을 수 있다. 그러면 ▶가 첫 grid 행을 따로 차지하거나 ▾ 힌트와 겹쳐 보인다.
  - 레포의 다른 summary(`watch-along.css`의 `.wa-sync`·`.wa-foot`, `inning-decisions.css`)는 모두 `list-style:none` + `::-webkit-details-marker{display:none}`을 쓴다.
  - 검토 머신에는 Playwright WebKit이 없어(chromium만 설치) 실제 재현은 하지 못했다. 그래서 위험으로 분류했다.
- 실패 상황: 시연용 아이폰에서 카드 맨 위에 삼각형이 따로 한 줄을 차지한다.
- 수정: 레포 관용대로 `list-style:none;` + `.cv-card summary::-webkit-details-marker{display:none}`을 넣었다. Chromium 화면은 달라지지 않는다(스크린샷 확인).

### F4 (낮음, 제안) 펼친 뒤에는 접기 안내가 없음
- 위치: `showcase.css:22` (`.cv-card[open] .cv-more{display:none}`)
- 내용: 접힌 상태의 "경기별 결과와 한계 보기 ▾"는 충분하다. summary가 네이티브 펼침 상태를 알려 주므로 스크린리더 접근성도 괜찮다. 다만 펼치면 힌트가 사라져, 휴대폰 사용자는 카드 머리를 다시 누르면 접힌다는 것을 알 수 없다. 레포의 `.wa-sync`는 펼치면 ▴로 바꾼다.
- 제안 diff:
  ```css
  -.cv-card[open] .cv-more{display:none}
  +.cv-card[open] .cv-more{font-size:0}.cv-card[open] .cv-more::after{content:'접기 ▴';font-size:13px}
  ```
  또는 JSX에 `<i className="cv-less">접기 ▴</i>`를 두고 열림 상태에 따라 하나만 보이게 한다. 문구는 작성자·PM이 고를 일이라 반영하지 않았다.

### F5 (낮음, 제안) NBC 카메라 RMS의 단서가 약함
- 위치: `CvEvidenceCard.tsx:29`
- 내용: `카메라 보정: 포구 공 13구 대조 RMS 0.15ft (고정 보정 파일 값)`. 이 값만 ft로 남아 있어, 인치 값의 면책과 달리 "물리 정확도 0.15ft"로 읽힐 수 있다. PM 검토는 "고정 보정 파일의 값으로 확인, PM이 새 영상 판독이나 Statcast 물리 측정을 반복한 것은 아니다"이고, 영수증 `human_verified_count`도 0이다. 같은 변환 기준의 판독 차이가 아니므로 SAME 문구를 붙이는 것은 틀리다. 대신 출처를 분명히 하는 쪽을 권한다.
- 제안 diff: `(고정 보정 파일 값)` → `(팀 보정 파일 보고값 · 새로 측정하지 않음)`.
- 검토자 테스트는 지금의 `(고정 보정 파일 값)`이 지워지면 실패하도록 고정했다. 문구를 바꾸면 테스트 문자열도 같이 바꾼다.

### F6 (낮음, 제안) 영수증 대조가 선택 사항이라 조용히 건너뜀
- 위치: `tests/cv-evidence-smoke.mjs:22,45`
- 내용: `CV_M3_RECEIPT`가 없으면 `receipt identity: skipped`로 통과한다. 변이 M5b(NBC px 중앙값을 2.0615로 바꿔 반올림 뒤 화면이 같음)는 영수증 없이 실행하면 **통과**했다. 영수증은 main 체크아웃에서도 미추적 파일이다.
- 실패 상황: root가 8771 통합 때 환경변수 없이 실행하면 숫자 표류를 잡지 못한다.
- 제안: 통합 체크리스트에 `CV_M3_RECEIPT=…`를 반드시 적는다. 또는 `CV_M3_RECEIPT_REQUIRED=1`이면 건너뛸 때 실패하게 한다:
  `} else if (process.env.CV_M3_RECEIPT_REQUIRED) throw new Error('CV_M3_RECEIPT required'); else console.log(…skipped…)`

### F7 (낮음, 제안) 픽셀 값의 기준 해상도가 화면에 없음
- 위치: `CvEvidenceCard.tsx:18,26,37`
- 내용: "2.5px"는 압축 중계 프레임 기준의 차이다. 영수증과 PM 검토에는 프레임 해상도가 기록돼 있지 않다(미측정). NBC와 FOX의 해상도가 다르면 두 경기의 px를 그대로 비교할 수 없다.
- 제안: 한계 목록에 "픽셀 차이는 각 중계 프레임 기준이며 해상도는 기록되지 않아 경기 간 직접 비교에 주의" 한 줄을 넣는다. 해상도를 팀원 자료에서 확인할 수 있으면 숫자를 적는다.

### F8 (낮음, 제안) 대표 예시 라벨이 모호함
- 위치: `CvEvidenceCard.tsx:8`
- 내용: `대표(중앙값에 가장 가까움)`은 출력 좌표 차이의 중앙값 기준이다. 바로 옆의 픽셀 값 2.3px는 픽셀 중앙값 2.5px와 다르다. 작은 글씨의 "순서가 다를 수 있습니다"가 있지만 라벨 자체가 기준을 말하지 않는다.
- 제안: `대표(출력 좌표 차이 중앙값)`.
- 참고: 가장 큰 예시 `849845:38:6`의 3.9인치는 거의 세로 성분(z −0.322ft)이다. PM 검토가 이 값들의 인용을 허용했고 "높이 미검증"이 같은 카드에 있으므로 막을 사유는 아니다.

### F9 (정보) showcase-smoke 수정은 원래 의도를 약화하지 않음
- 위치: `tests/showcase-smoke.mjs:119`
- 확인: 840eeee 판 테스트를 ecb3232 빌드에 돌리면 `comparison fits the viewport: {"y":-97.4,…}`로 실패한다. 앞선 Playwright 클릭이 카드 때문에 길어진 데스크톱 페이지를 약 97px 스크롤했기 때문이다.
- 원래 주석의 의도는 "desktop: from the page top"이다. `scrollTo(0,0)`은 그 전제를 다시 세울 뿐이고 높이 검사(`y>=0 && y+h<=900`)는 그대로다. 840eeee 빌드에서도 통과한다.
- 실제 사용자도 페이지 맨 위에서 비교 영역과 공개 막대를 함께 본다. 스크롤된 상태(y=−97)에서도 하단 574px < 900이라 공개 막대는 보인다.

## 3. 화면

- 기본 상태: `<details>`가 접힌 채로 시작하고, 펼쳤다 다시 접힌다(테스트로 확인).
- 폭: 360·390·1440에서 가로 넘침이 없다(`scrollWidth <= innerWidth`). 긴 투구 키와 해시는 `overflow-wrap:anywhere`로 줄바꿈된다. 스크린샷으로 확인했다.
- 위치: 카드는 `.sc-layout` grid 밖, `main`의 마지막 자식이다. 휴대폰에서는 영상 → 조작 → 비교 → 카드 순서라 비교·공개 흐름을 밀어내지 않는다. 데스크톱에서는 레이아웃 아래 전체 폭(최대 880px)에 놓인다.
- 한계: 검사용 timeline에 decisions가 없어 iframe 비교 영역은 "자료 없음" 상태로 렌더된다. 실제 iframe이 있는 상태는 showcase-smoke가 다룬다.
- 사소한 점: "합산" 문단과 그다음 기권 문단 사이에 간격이 없어 붙어 보인다(`.cv-card p` 여백 없음). 막을 사유는 아니다.

## 4. TypeScript

- `'caughtBalls' in g.cameraCheck`는 문제없다.
  - JSON import의 배열 리터럴 정규화로 FOX 쪽 타입에 `caughtBalls?: undefined`가 붙는다.
  - `status`는 `string`으로 넓혀져 `=== 'measured'`가 타입을 좁히지 못한다. 실제로 좁히는 것은 `rmsFeet !== null`이다.
  - 앞으로 FOX가 measured가 되는데 caughtBalls가 없으면 "포구 공 " 없이 표시될 뿐 깨지지 않는다.
- `tsc --noEmit` 결과 0, vite 빌드 경고 0(JS 366.6kB, 청크 경고 기준 미만).

## 실행한 명령과 결과

모든 명령은 `apps/observer/web`에서 `perl -e 'alarm N; exec @ARGV'`로 제한해 실행했다. node_modules는 `codex-showcase-1006`을 가리키는 임시 심볼릭 링크를 썼고 커밋 전에 지웠다. `dist`도 임시 링크로 썼고 매번 지웠다.

```
shasum -a 256 <main>/results/checks/CV-M3-receipt-recalculation-20261004.json
→ b9b2ab1911f55facf98ef74f202f9c00b0837e6a0c827132dc1566091d2bf630

./node_modules/.bin/tsc --noEmit -p tsconfig.json                         → tsc=0
./node_modules/.bin/vite build --outDir /tmp/pz-cv-m3-card-review-dist    → ✓ built (ecb3232 그대로)

# ecb3232 원본 테스트
DIST=/tmp/pz-cv-m3-card-review-dist CV_M3_RECEIPT=<receipt> node tests/cv-evidence-smoke.mjs <shots>
→ receipt identity: ok / cv evidence smoke: ok   (1.2 s)
DIST=/tmp/pz-reconnect-1004-old-dist (840eeee) …                          → TimeoutError waiting for details.cv-card (30 s), exit 1
  (old dist의 JS에 "cv-card" 문자열 없음 확인)

dist→ecb3232 빌드: node tests/showcase-smoke.mjs                          → ok
dist→840eeee 빌드: node tests/showcase-smoke.mjs                          → ok
dist→ecb3232 빌드: 840eeee 판 showcase-smoke                              → AssertionError comparison fits the viewport y=-97.4 (F9)

# 변이 검사 (ecb3232 원본 테스트, 변이마다 /tmp 빌드 후 복원)
M1 합산 줄 면책 삭제                    → 실패 (em 개수 2 ≠ 3)
M2 JSON pooled aiOutput 58→57           → 실패 (pooled aiOutput)
M3 예시에 <a href=/watch?game=…> 추가    → 실패 (no navigation from the evidence card)
M4 기권 28→26 문장 삭제                  → 실패 (detail: AI가 기권한 28장 중 26장…)
M5 NBC px 중앙값 2.0615 (영수증 있음)     → 실패 (game 849845)
M5b 같은 변이, 영수증 없음                → 통과  ← F6
M6 "(고정 보정 파일 값)" 삭제             → 통과  ← 검토자 테스트로 보강

# 검토자 변경 뒤
tsc=0; vite build --outDir /tmp/pz-cv-m3-card-review-dist2               → ✓ built
DIST=…dist2 CV_M3_RECEIPT=<receipt> node tests/cv-evidence-smoke.mjs <shots2>
→ receipt identity: ok / cv evidence smoke: ok (desktop 1440 + phone 390/360, synthetic timeline)
같은 테스트를 ecb3232 빌드에                                              → 실패 (설명 문장 2.5픽셀, F1)
같은 테스트를 840eeee 빌드에                                              → TimeoutError (카드 없음)
M6 / 예시 면책 삭제 / 경기별 person_only 13→12 변이                         → 모두 실패
dist→dist2: node tests/showcase-smoke.mjs                                 → ok
```

스크린샷(커밋 안 함): `/tmp/pz-cv-m3-card-review-shots2/cv-card-{closed,open,page}-{1440,390,360}.png`

## 검토자 변경 (이 커밋)

1. `CvEvidenceCard.tsx:21`: 설명 문장 단위 `px` → `픽셀`(작업 지시 원문, F1).
2. `CvEvidenceCard.tsx:37`: 예시 줄마다 `<em>{SAME}</em>`(F2).
3. `showcase.css:19`: `list-style:none` + `.cv-card summary::-webkit-details-marker{display:none}`(F3, 레포 관용).
4. `tests/cv-evidence-smoke.mjs`:
   - 영수증 대조에 경기별 `person_marked/person_abstained/person_only`, 합산 `person_abstained`, `output_z_signed`를 추가했다.
   - 360px 폭과 접힌 상태 스크린샷을 추가했다.
   - 설명 문장 전체와 카메라 단서 `(고정 보정 파일 값)`을 확인한다.
   - 면책 검사를 "'인치' 포함 블록마다 면책이 있음 + 블록 수 3+예시 수"로 바꿨다.

사용하지 않은 것: 다른 모델, 8766/8770/8771, tmux, ssh, GPU, main 체크아웃 수정, 다른 작업트리 수정, 실행 중인 서비스로의 HTTP 요청, 전체 테스트 묶음. push와 merge도 하지 않았다.
