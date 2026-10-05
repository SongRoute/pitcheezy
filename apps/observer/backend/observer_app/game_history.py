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
LINE_PROVENANCE = {'runs': 'score_deltas', 'hits': 'pa_result_events',
                   'errors': 'fielding_error_credits_dedup', 'bb_hbp': 'pa_result_events(walk,intent_walk,hit_by_pitch)'}
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
            'venue': (data.get('venue') or {}).get('name'),
            'game_number': data['game'].get('gameNumber') or 1, 'double_header': code in ('Y', 'S'),
            'double_header_code': code, 'embargoed': False, 'has_history': True}


def _player(feed, player_id):
    return feed['gameData'].get('players', {}).get(f'ID{player_id}', {})


def jersey(feed, player_id):
    """Game-specific number from this game's boxscore (never ``gameData.players.primaryNumber``); None if absent."""
    if player_id is None:
        return None
    teams = ((feed.get('liveData') or {}).get('boxscore') or {}).get('teams') or {}
    for side in ('away', 'home'):
        number = (((teams.get(side) or {}).get('players') or {}).get(f'ID{player_id}') or {}).get('jerseyNumber')
        if number not in (None, ''):
            return str(number)
    return None


def _person(feed, matchup, role):
    person = matchup.get(role) or {}
    pid = person.get('id')
    side = (matchup.get('batSide') if role == 'batter' else matchup.get('pitchHand')) or {}
    key = 'side' if role == 'batter' else 'hand'
    return {'id': pid, 'name': person.get('fullName') or _player(feed, pid).get('fullName'), key: side.get('code'),
            'jersey': jersey(feed, pid)}


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


BASES = {'1B': 'first', '2B': 'second', '3B': 'third'}


def _runner_moves(play):
    """``runners`` grouped by the ``playEvents`` index they belong to (``details.playIndex``), feed order kept.

    A move without a usable index is attached to the last event of the play (the PA-ending one)."""
    events = play.get('playEvents') or []
    moves = {}
    for runner in play.get('runners') or []:
        index = (runner.get('details') or {}).get('playIndex')
        if not isinstance(index, int) or not 0 <= index < len(events):
            index = len(events)-1
        moves.setdefault(index, []).append(runner)
    return moves


def _apply_moves(bases, runners):
    """Apply one event's runner movements to ``bases`` ({first|second|third: runner id or None}). Returns runs.

    The feed lists a runner's advance in segments (batter: none->1B, then 1B->2B on the throw), not in base
    order, so each runner is reduced to its first ``start`` and last segment first; every start is vacated
    before any end is occupied. A runner whose last segment ends at 'score' (not out) scores one run."""
    paths = {}
    for runner in runners:
        move = runner.get('movement') or {}
        rid = ((runner.get('details') or {}).get('runner') or {}).get('id')
        key = rid if rid is not None else object()
        first, _ = paths.get(key, (move, None))
        paths[key] = (first, move)
    for first, _ in paths.values():
        start = BASES.get(first.get('start'))
        if start:
            bases[start] = None
    runs = 0
    for rid, (_, last) in paths.items():
        if last.get('isOut'):
            continue
        end = BASES.get(last.get('end'))
        if end:
            bases[end] = rid if isinstance(rid, int) else True
        elif last.get('end') == 'score':
            runs += 1
    return runs


def _pinch_runner(bases, event):
    """Feed ``offensive_substitution`` at position PR: the new runner takes the replaced runner's base."""
    details = event.get('details') or {}
    if details.get('eventType') != 'offensive_substitution' or (event.get('position') or {}).get('abbreviation') != 'PR':
        return
    old, new = (event.get('replacedPlayer') or {}).get('id'), (event.get('player') or {}).get('id')
    for base, occupant in bases.items():
        if old is not None and occupant == old:
            bases[base] = new


def _occupied(bases):
    return {base: bases[base] is not None for base in ('first', 'second', 'third')}


def _substitution_before(play, role):
    """Mid-PA substitutions in ``role`` made after a pitch: {event index: the feed's ``replacedPlayer``}.

    Only a feed ``pitching_substitution`` (role pitcher) / ``offensive_substitution`` at position PH (role
    batter) that follows an earlier pitch counts; pitches before it belong to the replaced person."""
    kinds = {'pitcher': ('pitching_substitution',), 'batter': ('offensive_substitution',)}[role]
    pitched, changes = False, {}
    for index, event in enumerate(play.get('playEvents') or []):
        if event.get('isPitch'):
            pitched = True
        elif pitched and (event.get('details') or {}).get('eventType') in kinds and \
                (role == 'pitcher' or (event.get('position') or {}).get('abbreviation') == 'PH'):
            changes[index] = event.get('replacedPlayer') or {}
    return changes


def _replaced(feed, replaced, role):
    """The feed's ``replacedPlayer`` of a mid-PA substitution, as a batter/pitcher card (unknown fields None)."""
    pid = replaced.get('id')
    info = _player(feed, pid) if pid is not None else {}
    side = (info.get('batSide') if role == 'batter' else info.get('pitchHand')) or {}
    return {'id': pid, 'name': replaced.get('fullName') or info.get('fullName'),
            'side' if role == 'batter' else 'hand': side.get('code'), 'jersey': jersey(feed, pid)}


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
    prev_half, outs = None, 0
    bases = dict.fromkeys(('first', 'second', 'third'))
    previous = None  # last pitch so far: its after-state and whether a state-changing non-pitch event follows it
    # linescore as of each state point, from revealed events only (never the feed's final liveData.linescore)
    totals = {'away': dict.fromkeys(('hits', 'errors', 'bb_hbp'), 0), 'home': dict.fromkeys(('hits', 'errors', 'bb_hbp'), 0)}
    halves = []  # [inning, half, batting team's score when the half started]
    errors_seen = set()

    def line_asof():
        if not halves:
            return None
        rows = {}
        for i, (inning, half, start) in enumerate(halves):
            team = 'away' if half == 'Top' else 'home'
            later = next((h for h in halves[i+1:] if h[1] == half), None)
            rows.setdefault(inning, {'inning': inning, 'away': None, 'home': None})[team] = \
                (later[2] if later else (away if team == 'away' else home)) - start
        last = max(rows)
        return {'basis': 'revealed_events',
                'innings': [rows.get(i) or {'inning': i, 'away': None, 'home': None} for i in range(1, max(9, last)+1)],
                'teams': {'away': {'runs': away} | totals['away'], 'home': {'runs': home} | totals['home']},
                'current': {'inning': halves[-1][0], 'half': halves[-1][1]}, 'provenance': LINE_PROVENANCE}

    def credit_result(play, top):
        event = (play.get('result') or {}).get('eventType')
        team = 'away' if top else 'home'
        if event in HITS:
            totals[team]['hits'] += 1
        elif event in ('walk', 'intent_walk', 'hit_by_pitch'):
            totals[team]['bb_hbp'] += 1

    def credit_errors(ab, runners, top):
        for runner in runners:
            for credit in runner.get('credits') or []:
                if credit.get('credit') in ('f_fielding_error', 'f_throwing_error'):
                    key = (ab, (runner.get('details') or {}).get('playIndex'), (credit.get('player') or {}).get('id'),
                           credit.get('credit'))
                    if key not in errors_seen:
                        errors_seen.add(key)
                        totals['home' if top else 'away']['errors'] += 1

    def snapshot(inning, half, balls, strikes):
        state = {'inning': inning, 'half': half, 'outs': outs, 'balls': balls, 'strikes': strikes,
                 'home_score': home, 'away_score': away, 'runners': _occupied(bases)}
        state['bases'] = sum(bit for bit, base in ((1, 'first'), (2, 'second'), (4, 'third')) if state['runners'][base])
        return state

    for play in plays:
        about, matchup = play.get('about') or {}, play.get('matchup') or {}
        half = 'Top' if about.get('isTopInning') else 'Bot'
        inning, ab = about.get('inning'), about.get('atBatIndex', 0)+1
        if (inning, half) != prev_half:  # new half-inning: nobody on, nobody out
            outs, bases, prev_half = 0, dict.fromkeys(('first', 'second', 'third')), (inning, half)
            halves.append([inning, half, away if half == 'Top' else home])
        credited = False
        pa_key = f'{game_pk}:{ab}'
        batter, pitcher = _person(feed, matchup, 'batter'), _person(feed, matchup, 'pitcher')
        earlier = {role: _substitution_before(play, role) for role in ('batter', 'pitcher')}
        complete = bool(about.get('isComplete'))
        result = play.get('result') or {}
        pa = {'pa_key': pa_key, 'at_bat_number': ab, 'inning': inning, 'half': half, 'batter': batter, 'pitcher': pitcher,
              'complete': complete, 'event': result.get('eventType') if complete else None,
              'event_label': (RESULT_LABELS.get(result.get('eventType')) or result.get('event')) if complete else None,
              'result_text': result_text(play) if complete else None,
              'play_text': result.get('description') if complete else None,
              'pitch_count': 0, 'first_index': None, 'last_index': None}
        balls = strikes = 0
        numbered = 0
        moves = _runner_moves(play)
        events = play.get('playEvents') or []
        last_pitch = next((e for e in reversed(events) if e.get('isPitch')), None)
        # Does the PA end ON its last pitch, or on a later non-pitch event (caught stealing / pickoff for the 3rd out,
        # runner moves after the last pitch)? Only in the first case do the PA's result fields belong to that pitch.
        last_at = next((i for i in range(len(events)-1, -1, -1) if events[i] is last_pitch), -1)
        pitch_outs = ((last_pitch or {}).get('count') or {}).get('outs', 0)
        trailing = any(moves.get(i) or ((events[i].get('count') or {}).get('outs', 0) > pitch_outs)
                       for i in range(last_at+1, len(events)) if not events[i].get('isPitch'))
        pa['ends_on_pitch'] = complete and last_pitch is not None and not trailing
        for event_index, event in enumerate(events):
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
                last_of_pa = pa['ends_on_pitch'] and event is last_pitch
                # state before this pitch: every earlier event of the game applied, nothing from this one or later
                situation = {'inning': inning, 'half': half, 'outs': outs, 'balls': balls, 'strikes': strikes,
                             'home_score': home, 'away_score': away, 'runners': _occupied(bases)}
                situation['bases'] = sum(bit for bit, base in ((1, 'first'), (2, 'second'), (4, 'third')) if situation['runners'][base])
                line_before = line_asof()
                people = {}
                for role, current in (('batter', batter), ('pitcher', pitcher)):
                    later = [i for i in earlier[role] if i > event_index]
                    people[role] = current if not later else _replaced(feed, earlier[role][min(later)], role)
                cv = setups.get((ab, event.get('pitchNumber') or numbered))
                pitch = {'index': len(pitches), 'key': key, 'pa_key': pa_key, 'at_bat_number': ab,
                         'pitch_number': event.get('pitchNumber') or numbered, 'inning': inning, 'half': half,
                         'situation_before': situation, 'line_before': line_before, 'batter': people['batter'], 'pitcher': people['pitcher'],
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
            # after this event: its runner moves, its runs, the feed's count (balls/strikes/outs after the event)
            before_event = (outs, away, home, tuple(_occupied(bases).values()))
            _pinch_runner(bases, event)
            runs = _apply_moves(bases, moves.get(event_index, []))
            credit_errors(ab, moves.get(event_index, []), about.get('isTopInning'))
            if about.get('isTopInning'):
                away += runs
            else:
                home += runs
            if details.get('awayScore') is not None and details.get('homeScore') is not None:
                away, home = details['awayScore'], details['homeScore']
            count = event.get('count') or {}
            balls, strikes, outs = count.get('balls', balls), count.get('strikes', strikes), count.get('outs', outs)
            if is_pitch:
                if last_of_pa:  # the PA's own result (runs, outs) belongs to its last pitch
                    away, home = result.get('awayScore', away), result.get('homeScore', home)
                    outs = (play.get('count') or {}).get('outs', outs)
                    credit_result(play, about.get('isTopInning'))
                    credited = True
                after = snapshot(inning, half, 0 if last_of_pa else balls, 0 if last_of_pa else strikes)
                if after['outs'] >= 3:  # side retired: the board shows the half as over, nobody on
                    after |= {'runners': {'first': False, 'second': False, 'third': False}, 'bases': 0}
                pitches[-1]['situation_after'], pitches[-1]['gap_after'] = after, False
                pitches[-1]['line_after'] = line_asof()
                previous = pitches[-1]
            elif previous is not None and (
                    before_event != (outs, away, home, tuple(_occupied(bases).values()))
                    or details.get('eventType') in ('pitching_substitution', 'offensive_substitution')):
                previous['gap_after'] = True  # runner/score/out change or a new pitcher/batter between two pitches
        pas.append(pa)
        if complete:
            away, home = result.get('awayScore', away), result.get('homeScore', home)
            outs = (play.get('count') or {}).get('outs', outs)
            if not credited:
                credit_result(play, about.get('isTopInning'))
    final = None
    if (data.get('status') or {}).get('abstractGameState') == 'Final':
        line = live.get('linescore') or {}
        final = {'away': ((line.get('teams') or {}).get('away') or {}).get('runs', away),
                 'home': ((line.get('teams') or {}).get('home') or {}).get('runs', home)}
    last_half = prev_half or (None, None)
    now = snapshot(last_half[0], last_half[1], balls, strikes)  # after every event in the feed
    if now['outs'] >= 3:
        now |= {'runners': {'first': False, 'second': False, 'third': False}, 'bases': 0}
    return {'card': game_card(feed), 'pitches': pitches, 'pas': pas, 'final': final, 'zone_bounds': bounds,
            'has_watch': watch is not None, 'state_now': now, 'line_now': line_asof(),
            'official_date': data['datetime'].get('officialDate') or (data['datetime'].get('dateTime') or '')[:10] or None}
