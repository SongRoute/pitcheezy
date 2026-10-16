# MLB 자료 이용 허가 요청 — 보낼 글 초안

상태: 초안, 2026-10-11. 아직 아무에게도 보내지 않았다. Song이 내용을 확인하고 직접 보낸다. 근거는 [이용 조건 조사](mlb-data-terms-research-2026-10-11.md). 법률 자문이 아니다.

`[대괄호]`는 Song이 채울 자리다. 사실과 다른 문장이 있으면 보내기 전에 고친다.

## 1. MLB에 보내는 메일 (영문)

받는 곳: `legaldepartment@mlb.com` — MLB.com 이용약관 페이지에 적힌 법무 연락처다(2026-10-11 페이지에서 확인). 허가 요청 전용 창구가 따로 있는지는 확인하지 못했다. 같은 페이지의 "Help/Contact Us"로도 함께 보내는 것을 권한다.

제목: Request for written authorization — non-commercial student project using MLB Stats API data (2026 World Series)

```
Dear MLB Advanced Media Legal Department,

My name is [이름], a [학년/과정] student at [학교, 학과] in South Korea. I am writing
to ask for written authorization, as required by the notice at
gdx.mlb.com/components/copyright.txt, for a limited, non-commercial use of data from
the MLB Stats API in a university project.

What the project is
"Pitcheezy" is a student research project. During a game it shows, as text and a simple
strike-zone diagram, a statistical model's suggested pitch type and location before a
pitch, and then the actual pitch (type, speed, location, call) next to it, so that
beginner fans can follow pitch selection. It is free, carries no advertising, takes no
payment, and is not affiliated with or endorsed by MLB; the page says so.

What data we would use
- Stats API schedule and live game feed for 2026 postseason games: score, inning, count,
  runners, pitch type, speed, location, call, play description, player names and uniform
  numbers, and basic season statistics.
- No video, no audio, no photographs, no team logos or marks beyond team names and
  abbreviations as plain text.

How we would access it
- One server polls the feed for each live game about every 10 seconds. Visitors' browsers
  never contact MLB servers; they read a copy published by our server, so traffic to MLB
  does not grow with the number of visitors.
- We store the feed responses we receive so that a finished game can be replayed for
  evaluation and for a demonstration.

Where and when it would be shown
- A public web page open to anyone with the address during the 2026 World Series
  (October 23–31, 2026).
- A demonstration at a university exhibition in late November 2026, to faculty and
  reviewers, using games recorded during that period.

What we ask
1. Authorization for the use described above, or the conditions under which it would be
   acceptable (for example a display delay, a polling limit, a retention limit, required
   attribution wording, or restricting access to invited users).
2. If this use requires a data license instead, the appropriate contact.

We will follow any condition you set, display any notice you require, and remove the page
promptly on request. If we have not received authorization before October 23, we will
[허가가 없을 때의 계획: 예) limit access to invited users only].

Contact: [이메일]
Faculty advisor: [지도교수 이름, 이메일 — 넣을지 교수님과 상의]

Thank you for your time.

Sincerely,
[이름]
```

보내기 전에 확인할 것

- "about every 10 seconds"는 현재 설정값이다. 리허설 뒤 바뀌면 고친다.
- 과거 Statcast 자료(2023~2025)로 모델을 학습한 사실을 이 메일에 넣을지 정한다. 조사 보고는 이것도 허가가 확인되지 않은 범위로 보았다. 넣으면 정직하고 범위가 분명해지지만 질문이 커진다. 지도교수와 상의해 정하는 것을 권한다.
- 허가가 없을 때의 계획 문장은 Song의 결정에 맞춰 쓴다.

## 2. 지도교수·학교에 보내는 글 (국문)

```
교수님, 안녕하세요. [이름]입니다.

Pitcheezy 프로젝트를 10월 23~31일 월드시리즈 기간에 누구나 접속할 수 있는 웹 주소로
공개하고, 그때 모은 접속 기록과 의견을 11월 말 전시에서 함께 보여 드리려고 준비하고
있습니다.

준비 중에 확인한 점이 있어 여쭙니다. 저희가 쓰는 MLB 공개 경기 자료에는
"개인적·비상업·비대량 이용만 허용하며 그 밖의 이용은 MLB의 사전 서면 허가가 필요하다"는
고지가 붙어 있습니다. 무료 학생 프로젝트라도 불특정 다수에게 공개하는 것은 이 범위에
들어간다고 확인되지 않았습니다. 영상·로고·선수 사진은 쓰지 않고 글자와 간단한 그림만
표시합니다.

그래서 아래 두 가지를 하려고 합니다.
1. MLB 법무 쪽에 이용 범위를 적어 서면 허가를 요청하는 메일을 보냅니다(초안 첨부).
2. 월드시리즈 전까지 답이 없으면 [초대받은 사람만 접속 / 다른 계획]으로 범위를 줄입니다.

여쭙고 싶은 것은 다음과 같습니다.
- 학교 이름으로 전시하는 프로젝트에서 이런 외부 자료를 쓸 때 따라야 할 학교 기준이나
  문의할 부서가 있는지
- 허가 요청 메일에 교수님 성함과 연락처를 지도교수로 적어도 되는지
- 과거 경기 자료로 모델을 학습한 부분도 함께 문의하는 것이 좋을지

조사한 원문과 메일 초안을 함께 보내 드립니다. 감사합니다.
```
