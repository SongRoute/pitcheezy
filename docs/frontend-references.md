# 프론트엔드 레퍼런스

2026-10-02 조사. 화면 틀을 정할 때 참고한 서비스·글·코드 모음이다. 무엇을 가져오고 무엇을 피할지 한 줄씩 적었다.
틀 제안과 판단 기록은 [D-UI-004](handoffs/D-UI-004-frame.md)에 있다.

확인 범위: 중계 그래픽(ESPN·Apple)은 기사로만 확인했고 실제 화면 캡처는 보지 못했다. 오픈소스 세 개는 레포가 있는 것만 확인했고 코드를 읽지는 않았다.

## 1. 경기 페이지 — 다음 공 추천을 보여 주는 방식

| 레퍼런스 | 가져올 것 | 주의 |
|---|---|---|
| [ESPN Sunday Night Baseball: Statcast Edition](https://www.sportsvideo.org/2024/07/22/espns-sunday-night-baseball-statcast-edition-powered-by-google-cloud-alternate-broadcast-returns-with-a-shohei-ohtani-bang/) | 투구 전에 "Pitch Predictor"와 상황별 "Pitch Location"을 띄우는 가장 가까운 선례. 제작 원칙이 "대수학 수업이 되지 않게" | 화면 배치는 기사에 없다 |
| [NFL Next Gen Stats Decision Guide](https://www.nfl.com/news/introducing-the-next-gen-stats-decision-guide-a-new-analytics-tool-for-fourth-do) | 플레이 전에 최적 선택을 팬에게 보여 준다. 선택지별 승리확률을 나란히 놓는 형식(예: 69% 대 56%) | 후보 비교는 우리 쪽에서 후순위(D49) → 누른 뒤 화면에만 |
| [Apple TV+ Friday Night Baseball](https://www.apple.com/newsroom/2022/03/apple-and-major-league-baseball-to-offer-friday-night-baseball) | 상황별 결과 확률을 화면에 계속 띄우는 그래픽 | — |
| [ESPN 승리확률 그래픽 비판 (Defector)](https://defector.com/espns-win-probability-graphic-wants-to-give-you-gambling-brain) | 반면교사. 확률을 늘 띄우면 도박처럼 읽힌다 | 첫 화면에 숫자를 줄이는 근거 |
| [Chess.com Game Review](https://support.chess.com/article/488-how-do-i-review-the-whole-game) · [Chessiro](https://chessiro.com/blog/inside-chessiro-game-review) | 판은 고정, 옆 내용만 바뀐다. "최선 수 대 실제 수"를 승리확률 손실로 등급화한다 | A안(존 무대)의 바탕 |

## 2. 경기 페이지 — 한눈에 읽히게 줄이는 방식

| 레퍼런스 | 가져올 것 |
|---|---|
| [Apple Sports 앱](https://macrumors.com/2024/02/21/apple-sports-app-simplicity) | "들어와서 필요한 것만 보고 바로 나간다"는 원칙. 점수·박스 스코어·플레이 기록을 한 경기 화면에 둔다 |
| [Apple Sports 비판 (Six Colors)](https://sixcolors.com/link/2024/04/the-poor-design-of-apple-sports/) | 박스 스코어 열 정렬이 틀어진 사례. 단순함을 내세운 화면에서 표가 약점이 된다 |
| [iOS 잠금화면 실시간 점수 설계](https://vp0.com/blogs/live-activities-lock-screen-sports-scores-ui) | 팀·점수·경기 상태만 남기고 나머지는 뺀다는 기준. 점수판 한 줄의 정보 한도 |
| [MLB Gameday "At Bat Detail"](https://www.mlb.com/news/mlb-app-redesign-new-technology-2023) | 타석 단위로 공을 넘겨 보는 구성, 승부처에 승리확률 연결 |
| [두 번째 화면 연구 (Chalmers 석사 논문)](https://odr.chalmers.se/items/1dce5172-18b1-47d3-81f0-370be8695465) | 중계를 보며 쓰는 앱은 타이밍·주의 분산·부가 가치가 핵심이라는 지침 |
| [mlb-rs/mlbt](https://github.com/mlb-rs/mlbt) | 터미널용 Gameday(MIT, 별 166개). 좁은 화면에 존·카운트·타석 목록을 눌러 담은 밀도 |

## 3. 존 그림과 추천·실제 비교

| 레퍼런스 | 가져올 것 |
|---|---|
| [Baseball Savant Gamefeed · 3D Pitch Tracks](https://baseballsavant.mlb.com/changelog) | 존 위 투구 표시의 사실상 표준. 2026년부터 타자별 ABS 존 크기로 그린다 |
| [Umpire Scorecards](https://umpscorecards.com/FAQ) | 경기 한 장에 존 그림, 영향 큰 공, 득점 영향값. 부호 있는 영향값을 팬에게 전하는 형식 |

## 4. 지난 경기 분석

| 레퍼런스 | 가져올 것 |
|---|---|
| [FanGraphs 승리확률 그래프](https://blogs.fangraphs.com/win-probability-graphs-update/) | 그래프의 점을 짚으면 그 플레이와 변화량이 나온다. P2(그래프형)의 바탕 |
| Chess.com Game Review 요약 화면 | 먼저 요약과 승부처를 보여 주고, 원하면 한 수씩 본다. P1(요약형)의 바탕 |
| [nuotsu/mlb](https://github.com/nuotsu/mlb) ([데모](https://mlb.theohtani.com)) | Svelte. 승리확률 차트와 플레이 목록 연동. 라이선스 없음(구조만 참고) |

## 5. 신뢰·불확실성 표시

| 레퍼런스 | 가져올 것 |
|---|---|
| [Google PAIR Guidebook: Explainability + Trust](https://pair.withgoogle.com/chapter/People%20+%20AI%20Guidebook%20-%20Explainability%20+%20Trust.pdf) | 믿을 때와 스스로 판단할 때를 사용자가 구분하게 한다. 신뢰도를 언제 보여 줄지, 판단 보류를 어떻게 표시할지 |

## 6. 같은 스택의 코드

| 레포 | 내용 |
|---|---|
| [statsleuthgame/baseball-app](https://github.com/statsleuthgame/baseball-app) ([데모](https://statsleuthgame.github.io/baseball-app/)) | React 19 + Vite + FastAPI로 우리와 같다. 모바일 우선, 존·오심 SVG 컴포넌트. 별 0개, 라이선스 없음(구조만 참고) |

## 찾지 못한 것

- 네이버 스포츠 문자중계 등 국내 서비스의 화면 설계 자료(검색으로는 쓸 만한 것이 나오지 않았다).
- ESPN·Apple 중계 그래픽의 실제 배치(기사에 설명이 없다).
