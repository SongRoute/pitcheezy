"""Revealed-prefix summary of one watch-along game: line score, home WE curve and notable plays.

Display only (DEMO-WS-2026). Built solely from the precomputed watch-along dataset; no model is
loaded and nothing is estimated. ``through=k`` summarizes pitches ``0..k`` (``-1`` = none yet). It
reads the actual pitch and WE of those pitches, the pre-pitch situation of pitch ``k+1`` (already
public in the timeline; it carries the score after pitch ``k``) and the final score only once the
last pitch is revealed. A value the dataset does not carry stays ``None``. WE changes are observed
differences of a frozen model's estimate, never a player's responsibility or a causal effect.
"""
from __future__ import annotations

import math

SCHEMA = 'pitcheezy-watch-summary-v1'
HALVES = ('Top', 'Bot')
BATTING = {'away': 'Top', 'home': 'Bot'}
TOP_LIMIT = 20
WE_KIND = 'observed_model_change'
NOTE = '승리확률 변화는 고정된 모델 추정치의 관측 차이이며, 선수 책임이나 인과 기여도가 아닙니다.'


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return float(value)


def _half(decision):
    situation = decision.get('situation') or {}
    inning, half = situation.get('inning'), situation.get('half')
    if type(inning) is not int or inning < 1 or half not in HALVES:
        raise ValueError('watch-along pitch has no valid inning/half')
    return inning, half


def _order(key):
    return key[0], HALVES.index(key[1])


def _score(source):
    """``{'away','home'}`` from a pre-pitch situation or the final record; unknown teams are None."""
    source = source or {}
    if 'away_score' in source or 'home_score' in source:
        return {'away': _count(source.get('away_score')), 'home': _count(source.get('home_score'))}
    return {'away': _count(source.get('away')), 'home': _count(source.get('home'))}


def _we(decision):
    we = decision.get('we') or {}
    before, after = _probability(we.get('home_before')), _probability(we.get('home_after'))
    delta = None if before is None or after is None else after - before
    batting = None if delta is None else (-delta if decision['situation']['half'] == 'Top' else delta)
    return {'kind': WE_KIND, 'home_before': before, 'home_after': after, 'home_delta': delta, 'batting_delta': batting}


def _play(decision, runs, score_after):
    actual = decision.get('actual') or {}
    person = lambda p: {'id': (p or {}).get('id'), 'name': (p or {}).get('name')}
    inning, half = _half(decision)
    return {'index': decision['index'], 'pa_id': decision.get('pa_id'), 'at_bat_number': decision.get('at_bat_number'),
            'pitch_number': decision.get('pitch_number'), 'inning': inning, 'half': half,
            'pitcher': person(decision.get('pitcher')), 'batter': person(decision.get('batter')),
            'event_label': actual.get('event_label'), 'result_label': actual.get('result_label'),
            'play_text': actual.get('play_text'), 'runs_scored': runs, 'score_after': score_after, 'we': _we(decision)}


def summarize(data, through, top=5):
    decisions = data['decisions']
    if type(through) is not int or not -1 <= through < len(decisions):
        raise IndexError(through)
    if type(top) is not int or not 0 <= top <= TOP_LIMIT:
        raise IndexError(top)
    game = data['game']
    complete = bool(decisions) and through == len(decisions) - 1
    final = _score(game.get('final')) if complete and isinstance(game.get('final'), dict) else None
    issues, halves, plays, per_pitch = [], {}, [], {}

    def after(i):
        """Score after pitch i and where it comes from (next pre-pitch situation or the final record)."""
        if i + 1 < len(decisions):
            return _score(decisions[i + 1].get('situation')), 'next_pre_pitch_situation'
        if final is not None:
            return final, 'final_record'
        return {'away': None, 'home': None}, 'unavailable'

    for i in range(through + 1):
        d = decisions[i]
        key = _half(d)
        if halves and _order(key) < _order(next(reversed(halves))):
            raise ValueError('watch-along half-innings are out of game order')
        halves.setdefault(key, {'runs': 0})
        nxt = _half(decisions[i + 1]) if i + 1 < len(decisions) else None
        before, (score_after, _source) = _score(d.get('situation')), after(i)
        runs = {}
        for team, side in BATTING.items():
            b, a = before[team], score_after[team]
            delta = None if a is None or b is None else a - b
            if delta is not None and delta < 0:
                issues.append(f'{team}_score_decreased_after_index_{i}')
                delta = None
            runs[team] = delta
            # Runs belong to the half in which that team bats: this pitch's half, else (rarely, a run
            # before its first pitch) the next pitch's half. An unknown change there is not guessed.
            target = next((k for k in (key, nxt) if k is not None and k[1] == side), None)
            if target is None:
                if delta:
                    issues.append(f'{team}_runs_outside_batting_half_after_index_{i}')
            elif target == key:
                cell = halves[key]
                cell['runs'] = None if delta is None or cell['runs'] is None else cell['runs'] + delta
            elif delta:
                cell = halves.setdefault(target, {'runs': 0})
                cell['runs'] = None if cell['runs'] is None else cell['runs'] + delta
        per_pitch[i] = runs, score_after
        if (d.get('actual') or {}).get('event_label') is not None or any(runs.values()):
            plays.append(_play(d, runs, score_after))

    if through >= 0:
        score, source = after(through)
    elif decisions:
        score, source = _score(decisions[0].get('situation')), 'pre_pitch_situation'
    else:
        score, source = {'away': None, 'home': None}, 'unavailable'

    last = _half(decisions[through]) if through >= 0 else None
    upcoming = _half(decisions[through + 1]) if 0 <= through < len(decisions) - 1 else None
    innings = []
    for inning in range(1, max((k[0] for k in halves), default=0) + 1):
        row = {'inning': inning}
        for team, side in BATTING.items():
            key = (inning, side)
            if key in halves:
                ended = complete or _order(key) < _order(last) or (upcoming is not None and _order(upcoming) > _order(key))
                row[team] = {'runs': halves[key]['runs'], 'status': 'complete' if ended else 'in_progress'}
            elif last is not None and _order(key) < _order(last):
                row[team] = {'runs': None, 'status': 'unknown'}  # a half that must have happened has no pitch
            else:
                row[team] = {'runs': None, 'status': 'not_played' if complete else 'not_reached'}
        innings.append(row)

    for team in BATTING:
        cells = [row[team] for row in innings if row[team]['status'] not in ('not_played', 'not_reached')]
        if (score[team] is not None and all(c['runs'] is not None for c in cells)
                and sum(c['runs'] for c in cells) != score[team]):
            issues.append(f'{team}_line_score_differs_from_score')

    revealed = decisions[:through + 1]
    curve = [{'index': d['index'], 'inning': _half(d)[0], 'half': _half(d)[1], 'at_bat_number': d.get('at_bat_number'),
              'pitch_number': d.get('pitch_number'), **{k: v for k, v in _we(d).items() if k in ('home_before', 'home_after')}}
             for d in revealed]
    changed = sorted((d for d in revealed if _we(d)['home_delta'] is not None),
                     key=lambda d: (-abs(_we(d)['home_delta']), d['index']))[:top]
    largest = [_play(d, *per_pitch[d['index']]) for d in changed]
    return {'schema': SCHEMA, 'game_pk': game['game_pk'], 'through': through, 'revealed_pitches': through + 1,
            'complete': complete, 'teams': {'away': game.get('away_team'), 'home': game.get('home_team')},
            'score': {**score, 'source': source, 'after_index': through if through >= 0 else None},
            'innings': innings, 'we_curve': curve, 'events': plays, 'largest_we_changes': largest,
            'notes': {'we': NOTE, 'we_model': (data.get('we') or {}).get('note'), 'attribution': None},
            'issues': issues}

