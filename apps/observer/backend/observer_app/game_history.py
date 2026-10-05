"""Full pitch/PA history of one game from a GUMBO feed, for the service game API (pitcheezy-service-v2).

Display only. Facts come from the feed; the only recommendation attached is one stored before the pitch
(precomputed DEMO-WS-2026 watch dataset: ``asof_replay``). Nothing here computes a recommendation, and a
PA result text names a fielder or direction only when the feed itself carries it (fielding credits,
``hitData.location``, ``hitData.trajectory``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .domain import PITCH_LABELS, RESULT_LABELS, actual_zone

KST = timezone(timedelta(hours=9))
POSITIONS = {'1': '투수', '2': '포수', '3': '1루수', '4': '2루수', '5': '3루수', '6': '유격수',
             '7': '좌익수', '8': '중견수', '9': '우익수'}
TRAJECTORY = {'ground_ball': '땅볼', 'fly_ball': '플라이', 'line_drive': '라인드라이브', 'popup': '팝플라이',
              'bunt_grounder': '번트 땅볼', 'bunt_popup': '번트 팝플라이', 'bunt_line_drive': '번트 라인드라이브'}
CALLS = {'B': '볼', '*B': '볼', 'V': '볼', 'I': '고의4구 볼', 'P': '피치아웃', 'C': '루킹 스트라이크', 'S': '헛스윙',
         'W': '헛스윙', 'Q': '헛스윙', 'M': '번트 헛스윙', 'F': '파울', 'R': '파울', 'L': '번트 파울', 'T': '파울팁',
         'O': '파울팁', 'X': '인플레이', 'D': '인플레이', 'E': '인플레이', 'H': '몸에 맞는 공'}
HITS = {'single': '1루타', 'double': '2루타', 'triple': '3루타', 'home_run': '홈런'}
EXTRA_EVENTS = {'field_error': '실책 출루', 'catcher_interf': '타격 방해', 'intent_walk': '고의4구',
                'caught_stealing_2b': '도루 실패', 'caught_stealing_3b': '도루 실패', 'caught_stealing_home': '도루 실패',
                'pickoff_1b': '견제 아웃', 'pickoff_2b': '견제 아웃', 'pickoff_3b': '견제 아웃',
                'sac_bunt': '희생 번트', 'strikeout_double_play': '삼진 병살', 'triple_play': '삼중살'}
DEFAULT_ZONE_BOUNDS = {'bottom': 1.6, 'top': 3.3899}


def kst_fields(start_utc):
    """``(start_kst ISO, date_kst)`` of an original scheduled start in UTC (``2026-09-30T02:00:00Z``)."""
    if not start_utc:
        return None, None
    moment = datetime.fromisoformat(start_utc.replace('Z', '+00:00')).astimezone(KST)
    return moment.isoformat(), moment.date().isoformat()


def status_kind(abstract, detailed, coded=None):
    detailed = detailed or ''
    if 'Postponed' in detailed or 'Suspended' in detailed or 'Cancel' in detailed:
        return 'postponed'
    return {'Live': 'live', 'Final': 'final', 'Preview': 'scheduled'}.get(abstract, 'other')


def game_card(feed):
    """List entry facts of one game from its feed (scores/progress filled by the caller's cutoff)."""
    data = feed['gameData']
    start = data['datetime'].get('dateTime')
    start_kst, date_kst = kst_fields(start)
    status = data.get('status') or {}
    code = data['game'].get('doubleHeader') or 'N'
    teams = data['teams']
    return {'game_pk': data['game']['pk'], 'start': start, 'start_kst': start_kst, 'date_kst': date_kst,
            'official_date': data['datetime'].get('officialDate'),
            'status': status_kind(status.get('abstractGameState'), status.get('detailedState')),
            'status_text': status.get('detailedState'), 'live': status.get('abstractGameState') == 'Live',
            'away_team': teams['away'].get('name'), 'home_team': teams['home'].get('name'),
            'away_abbr': teams['away'].get('abbreviation'), 'home_abbr': teams['home'].get('abbreviation'),
            'away_score': None, 'home_score': None, 'inning': None, 'half': None,
            'game_number': data['game'].get('gameNumber') or 1, 'double_header': code in ('Y', 'S'),
            'double_header_code': code, 'embargoed': False, 'has_history': True}


def _player(feed, player_id):
    return feed['gameData'].get('players', {}).get(f'ID{player_id}', {})


def _person(feed, matchup, role):
    person = matchup.get(role) or {}
    pid = person.get('id')
    side = (matchup.get('batSide') if role == 'batter' else matchup.get('pitchHand')) or {}
    key = 'side' if role == 'batter' else 'hand'
    return {'id': pid, 'name': person.get('fullName') or _player(feed, pid).get('fullName'), key: side.get('code')}


def _batter_credits(play):
    for runner in play.get('runners') or []:
        if (runner.get('movement') or {}).get('originBase') is None and \
                (runner.get('details') or {}).get('runner', {}).get('id') == (play.get('matchup') or {}).get('batter', {}).get('id'):
            return runner.get('credits') or []
    return []


def result_text(play):
    """Grounded Korean PA result. Fielder/direction only from feed fields; otherwise the event label only."""
    result = play.get('result') or {}
    event = result.get('eventType')
    if not event:
        return None
    pitches = [e for e in play.get('playEvents') or [] if e.get('isPitch')]
    last = pitches[-1] if pitches else {}
    hit = last.get('hitData') or {}
    trajectory = TRAJECTORY.get(hit.get('trajectory'))
    location = POSITIONS.get(str(hit.get('location') or ''))
    credits = _batter_credits(play)
    fielder = next((POSITIONS.get(c['position']['code']) for c in credits
                    if c.get('credit') in ('f_fielded_ball', 'f_assist', 'f_putout', 'f_fielding_error', 'f_throwing_error')
                    and c.get('position')), None) or location
    if event == 'strikeout':
        code = ((last.get('details') or {}).get('call') or {}).get('code')
        return {'C': '루킹 삼진', 'S': '헛스윙 삼진', 'W': '헛스윙 삼진', 'T': '파울팁 삼진', 'O': '파울팁 삼진',
                'M': '번트 헛스윙 삼진', 'L': '스리번트 삼진'}.get(code, '삼진')
    if event in HITS:
        return f'{location} 방면 {HITS[event]}' if location else HITS[event]
    if event in ('field_out', 'force_out', 'grounded_into_double_play', 'double_play', 'fielders_choice_out',
                 'fielders_choice', 'sac_fly', 'sac_fly_double_play', 'sac_bunt'):
        base = {'field_out': '아웃', 'force_out': '포스 아웃', 'grounded_into_double_play': '병살타',
                'double_play': '병살', 'fielders_choice_out': '야수 선택 아웃', 'fielders_choice': '야수 선택',
                'sac_fly': '희생 플라이', 'sac_fly_double_play': '희생플라이 병살', 'sac_bunt': '희생 번트'}[event]
        if event == 'field_out' and trajectory:
            base = f'{trajectory} 아웃'
        elif event == 'grounded_into_double_play':
            base = '땅볼 병살타'
        return f'{fielder} {base}' if fielder else base
    if event == 'field_error':
        error = next((POSITIONS.get(c['position']['code']) for c in credits
                      if c.get('credit') in ('f_fielding_error', 'f_throwing_error') and c.get('position')), None)
        return f'{error} 실책 출루' if error else '실책 출루'
    return RESULT_LABELS.get(event) or EXTRA_EVENTS.get(event) or result.get('event') or event


def _bases(play):
    matchup = play.get('matchup') or {}
    first, second, third = ('postOnFirst' in matchup, 'postOnSecond' in matchup, 'postOnThird' in matchup)
    return {'first': first, 'second': second, 'third': third}


def _rec_from_decision(decision, generated_utc, policy):
    if decision is None:
        return absent_rec()
    pre = decision.get('pre') or {}
    stamp = None
    if generated_utc:
        stamp = datetime.strptime(generated_utc, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
    base = {'provenance': 'asof_replay', 'recorded_at': stamp,
            'policy': {'name': policy.get('name'), 'identity': (policy.get('identity_sha256') or '')[:8] or None}}
    if pre.get('status') == 'ready' and (pre.get('recommendation') or {}).get('candidates'):
        return base | {'status': 'ready', 'reason': None, 'candidates': [
            {'rank': c['rank'], 'pitch_type': c['pitch_type'], 'pitch_label': c['pitch_label'], 'zone_id': c.get('zone_id'),
             'zone_label': c.get('zone_label'), 'target': c.get('target'),
             'detail': {'probability': (c.get('detail') or {}).get('probability')}}
            for c in pre['recommendation']['candidates']]}
    return base | {'status': 'unsupported', 'reason': pre.get('reason') or '이 상황은 추천을 지원하지 않습니다.', 'candidates': []}


def absent_rec(reason='이 투구의 저장된 추천이 없습니다.'):
    return {'provenance': 'absent', 'status': 'missing', 'reason': reason, 'recorded_at': None, 'policy': None,
            'candidates': []}


def build(feed, watch=None, setup=None):
    """Complete game record (server-only; callers cut it at their visibility cutoff).

    ``watch``: optional DEMO-WS-2026 dataset of the same game (pre-pitch situations and stored
    recommendations, keyed by ``gamePk:atBat:pitchNumber``). ``setup``: optional catcher setup view.
    """
    data, live = feed['gameData'], feed['liveData']
    game_pk = data['game']['pk']
    decisions = {f"{d['pa_id']}:{d['pitch_number']}": d for d in (watch or {}).get('decisions', [])}
    generated, policy = (watch or {}).get('generated_utc'), (watch or {}).get('policy') or {}
    bounds = dict(((watch or {}).get('location') or {}).get('zone_bounds') or DEFAULT_ZONE_BOUNDS)
    setups = {(p['at_bat_number'], p['pitch_number']): p for p in (setup or {}).get('pitches', [])}
    plays = list(live['plays'].get('allPlays') or [])
    pitches, pas = [], []
    away = home = 0
    prev_half, outs_start, bases_start = None, 0, {'first': False, 'second': False, 'third': False}
    for play in plays:
        about, matchup = play.get('about') or {}, play.get('matchup') or {}
        half = 'Top' if about.get('isTopInning') else 'Bot'
        inning, ab = about.get('inning'), about.get('atBatIndex', 0)+1
        if (inning, half) != prev_half:
            outs_start, bases_start, prev_half = 0, {'first': False, 'second': False, 'third': False}, (inning, half)
        pa_key = f'{game_pk}:{ab}'
        batter, pitcher = _person(feed, matchup, 'batter'), _person(feed, matchup, 'pitcher')
        complete = bool(about.get('isComplete'))
        result = play.get('result') or {}
        pa = {'pa_key': pa_key, 'at_bat_number': ab, 'inning': inning, 'half': half, 'batter': batter, 'pitcher': pitcher,
              'complete': complete, 'event': result.get('eventType') if complete else None,
              'event_label': (RESULT_LABELS.get(result.get('eventType')) or result.get('event')) if complete else None,
              'result_text': result_text(play) if complete else None,
              'play_text': result.get('description') if complete else None,
              'pitch_count': 0, 'first_index': None, 'last_index': None}
        balls = strikes = 0
        outs = outs_start
        numbered = 0
        for event in play.get('playEvents') or []:
            details = event.get('details') or {}
            call = (details.get('call') or {}).get('code')
            is_pitch = bool(event.get('isPitch'))
            if is_pitch or (event.get('type') == 'no_pitch' and call):
                numbered += 1
            if is_pitch:
                key = f'{pa_key}:{event.get("pitchNumber") or numbered}'
                decision = decisions.get(key)
                code = (details.get('type') or {}).get('code')
                coords = (event.get('pitchData') or {}).get('coordinates') or {}
                x, z = coords.get('pX'), coords.get('pZ')
                last_of_pa = complete and event is [e for e in play['playEvents'] if e.get('isPitch')][-1]
                situation = (dict(decision['situation']) if decision else
                             {'outs': outs, 'balls': balls, 'strikes': strikes,
                              'bases': bases_start['first']+2*bases_start['second']+4*bases_start['third'],
                              'home_score': home, 'away_score': away})
                situation['runners'] = {'first': bool(situation['bases'] & 1), 'second': bool(situation['bases'] & 2),
                                        'third': bool(situation['bases'] & 4)}
                cv = setups.get((ab, event.get('pitchNumber') or numbered))
                pitch = {'index': len(pitches), 'key': key, 'pa_key': pa_key, 'at_bat_number': ab,
                         'pitch_number': event.get('pitchNumber') or numbered, 'inning': inning, 'half': half,
                         'situation_before': situation, 'batter': batter, 'pitcher': pitcher,
                         'rec': _rec_from_decision(decision, generated, policy) if watch else absent_rec(),
                         'actual': {'pitch_type': code,
                                    'pitch_label': PITCH_LABELS.get(code, (details.get('type') or {}).get('description')),
                                    'description': (decision or {}).get('actual', {}).get('description') or details.get('description'),
                                    'call_label': CALLS.get(call, details.get('description')),
                                    'result_label': pa['event_label'] if last_of_pa and pa['event_label'] else CALLS.get(call, details.get('description')),
                                    'event': pa['event'] if last_of_pa else None,
                                    'event_label': pa['event_label'] if last_of_pa else None,
                                    'play_text': pa['play_text'] if last_of_pa else None,
                                    'speed_mph': (event.get('pitchData') or {}).get('startSpeed'), 'x': x, 'z': z,
                                    'zone_label': actual_zone(x, z, bounds),
                                    'catcher_setup': None if cv is None else
                                    {'status': cv['status'], 'x_band': cv.get('x_band'), 'plate_x_feet': cv.get('plate_x_feet')}}}
                pitches.append(pitch)
                pa['pitch_count'] += 1
                pa['first_index'] = pitch['index'] if pa['first_index'] is None else pa['first_index']
                pa['last_index'] = pitch['index']
            count = event.get('count') or {}
            balls, strikes, outs = count.get('balls', balls), count.get('strikes', strikes), count.get('outs', outs)
        pas.append(pa)
        if complete:
            away, home = result.get('awayScore', away), result.get('homeScore', home)
            outs_start = (play.get('count') or {}).get('outs', outs)
            bases_start = _bases(play)
    final = None
    if (data.get('status') or {}).get('abstractGameState') == 'Final':
        line = live.get('linescore') or {}
        final = {'away': ((line.get('teams') or {}).get('away') or {}).get('runs', away),
                 'home': ((line.get('teams') or {}).get('home') or {}).get('runs', home)}
    return {'card': game_card(feed), 'pitches': pitches, 'pas': pas, 'final': final, 'zone_bounds': bounds,
            'has_watch': watch is not None}
