# MLB 자료 이용 조건 조사 (2026-10-11)

상태: 조사 보고. 작성은 Codex CLI `gpt-6-sol`(웹 검색, 읽기 전용), 의뢰와 정리는 Claude PM. 법률 자문이 아니며 결정은 사람이 한다. MLBAM 고지(`gdx.mlb.com/components/copyright.txt`) 원문은 PM이 같은 날 직접 받아 인용문과 일치함을 확인했다. 그 밖의 인용은 PM이 다시 열어 보지 않았다.

**결론:** Pitcheezy가 무료 학생 프로젝트여도, 누구나 접속하는 웹사이트에 MLB의 투구별 실시간 데이터를 표시할 권한이 확인되지는 않았습니다. 가장 분명한 경로는 공개 표시·자동 수집·보관·모델 활용 범위를 적어 MLBAM에 **사전 서면 허가**를 받거나, 해당 범위를 포함한 데이터 라이선스를 확보하는 것입니다. 지연 표시나 출처 표기만으로 허가가 생기지는 않습니다.

## 1. 확인한 원문

아래 자료는 모두 **2026-10-11 접속**했습니다.

- [Stats API 일정 응답](https://statsapi.mlb.com/api/v1/schedule?sportId=1): “Use of any content on this page acknowledges agreement to the terms posted here”. 응답 자체가 [MLBAM 별도 고지](https://gdx.mlb.com/components/copyright.txt)를 연결합니다.
- [MLBAM 고지](https://gdx.mlb.com/components/copyright.txt): “Only individual, non-commercial, non-bulk use of the Materials is permitted”; 그 밖의 사용에는 “prior written authorization from MLBAM”을 요구합니다. 무료라는 조건 하나로는 부족합니다.
- [MLB.com Terms of Use §1·§4](https://www.mlb.com/official-information/terms-of-use): “without first obtaining the written permission of MLB”; “use automated scripts to collect information”; “for your private, non-commercial use”; “unreasonable or disproportionately large load”. 약관은 개인적 이용, 재배포, 자동 수집, 서버 부하를 각각 제한합니다.
- [MLB 법적 고지](https://www.mlb.com/official-information/legal-notices): 구단의 “names, nicknames, logos, uniform designs, color combinations”은 “may be used only with permission”. 구단 표장에 관한 고지입니다.
- [Baseball Savant](https://baseballsavant.mlb.com/): “© MLB Advanced Media, LP. All rights reserved.” 하단 이용약관 링크는 MLB.com 약관으로 연결됩니다. 별도의 모델 학습 허가는 확인되지 않았습니다.
- [MLBPA FAQ](https://www.mlbplayers.com/frequently-asked-questions): 현역 선수의 “names, numbers, nicknames, likenesses, signatures, playing records and biographical data”를 집단 라이선스 대상 권리로 설명합니다. 단순한 사실 보도의 허용 범위까지 답하지는 않습니다.
- [MLB·Sportradar 공동 발표](https://investors.sportradar.com/news-releases/news-release-details/major-league-baseball-and-sportradar-announce-expanded-exclusive): Sportradar가 MLB 공식 데이터를 “exclusively distribute MLB’s ultra-low latency official data”한다고 발표했습니다. 제3자 배포 계약의 존재를 보여주지만, 공개 Stats API의 학생 이용 허가는 아닙니다.
- [Sportradar 미디어 데이터 부속약관](https://developer.sportradar.com/sportradar-updates/page/official-media-data-addendum): 계약 고객에게만 “for media and editorial purposes and non-commercial use only”라는 범위의 권리를 부여합니다. 실시간 경기 재현·파생 데이터·팀과 선수 권리에도 별도 제한을 둡니다. **Pitcheezy에 자동 적용되는 허가는 아닙니다.**

## 2. 서비스 요소별 판단 근거

아래 신뢰도는 *원문에 근거해 위험을 판단할 수 있는 정도*입니다.

- **공개 투구별 실제 결과, 점수·이닝·카운트·주자 — 높음:** **원문 명시:** 개인·비상업·비대량 사용 및 서면 허가 조건, 자동 수집 제한. **해석:** 반복 폴링으로 경기 진행을 공개 표시하는 구상은 허용 범위라고 단정하기 어렵습니다. 모델 추천을 실제 투구와 나란히 보여줄 권한도 명시되지 않았습니다.
- **실시간 대신 지연 표시 — 높음:** **원문 명시:** 확인한 API 고지에는 허용 지연 시간이 없습니다. **해석:** 지연은 위험을 줄일 수 있으나 이용 허가를 대신하지 않습니다.
- **선수 이름·등번호 — 중간:** **원문 명시:** MLBPA는 두 항목을 집단 라이선스 설명에 포함합니다. **해석:** 정보성 경기 표기가 허가 대상인지는 별도 판단이 필요합니다. 사진·초상은 사용하지 않는 편이 안전합니다.
- **팀명·약칭 — 중간 / 로고 — 높음:** **원문 명시:** MLB는 구단 명칭·로고 등 표장의 허가를 요구합니다. **해석:** 경기 식별을 위한 문자 표기 범위는 불명확하지만, 로고 사용은 허가 전 피해야 합니다.
- **기본 시즌 통계 — 중간:** **원문 명시:** MLB 약관은 콘텐츠 재배포를 제한하고 MLBPA는 선수 기록을 열거합니다. **해석:** 개별 수치가 경기 사실이더라도, API에서 가져온 통계의 공개 제공 허가는 확인되지 않았습니다.
- **2023–2025 Statcast 자료의 pybaseball 학습 — 중간:** **원문 명시:** Savant는 MLB 약관으로 연결되며 별도 학습 허가를 제시하지 않습니다. **해석:** 공개 다운로드 기능이나 pybaseball 사용이 대량 수집·학습·공개 서비스 이용권을 부여하지는 않습니다.
- **피드 스냅샷 저장 — 높음:** **원문 명시:** MLBAM은 비대량 사용만 허용하고 MLB 약관은 복제·배포를 제한합니다. **해석:** 장기 보관과 전시 재생은 허가 요청 범위에 포함해야 합니다.
- **누구나 여는 URL·11월 대학 전시 — 높음:** **원문 명시:** 개인적 이용 예외와 공개 전시·배포 제한이 구분됩니다. **해석:** 접속자가 적고 광고가 없어도 공개 서비스입니다.

## 3. 위험을 줄이는 선택지

- [MLBAM 고지](https://gdx.mlb.com/components/copyright.txt)와 [MLB 약관](https://www.mlb.com/official-information/terms-of-use)을 근거로, 공개 범위·갱신 주기·저장 기간·전시 방식·모델 출력을 적어 **서면 허가 범위**를 확인합니다.
- [API 응답](https://statsapi.mlb.com/api/v1/schedule?sportId=1)의 저작권 문구와 연결된 조건 URL을 출처 안내에 표시하고, “비상업 학생 프로젝트·MLB 비공식”이라고 명확히 적습니다. 확인한 원문에는 **공개 사이트용 의무 표시 문안이 없으며**, 이 표기는 허가를 대체하지 않습니다.
- [법적 고지](https://www.mlb.com/official-information/legal-notices)에 따라 로고·공식 디자인을 빼고, [MLBPA FAQ](https://www.mlbplayers.com/frequently-asked-questions)에 비추어 선수 사진·초상을 빼며 이름·번호 사용 범위를 문의합니다.
- 허가를 받는다면 지연 표시, 경기당 단일 수집기·캐시·낮은 호출 빈도로 운영하고 연락처와 신속한 삭제 절차를 둡니다. 이는 [약관의 서버 부하·권리 제한](https://www.mlb.com/official-information/terms-of-use)에 대응하는 조치이지 독립적인 이용권은 아닙니다.

## 4. 확인하지 못한 것과 문의 대상

[실제 GUMBO 경기 피드](https://statsapi.mlb.com/api/v1.1/game/849831/feed/live)는 열람 도구에서 열리지 않았고, [Stats API 문서](https://docs.statsapi.mlb.com/)는 본문이 표시되지 않았습니다. 공식적인 **숫자형 호출 한도, 허용 지연 시간, 공개 재배포·스냅샷·Statcast 학습 허가, 필수 저작권 표시 문안**도 확인하지 못했습니다. 이 범위는 **MLB/MLBAM**, 공식 데이터 계약 경로는 **Sportradar**, 현역 선수의 집단 권리는 **MLB Players, Inc./MLBPA**에 확인해야 합니다.

이 보고서는 법률 자문이 아닙니다. 공개 여부와 허가 요청에 관한 결정은 사람이 내려야 합니다.