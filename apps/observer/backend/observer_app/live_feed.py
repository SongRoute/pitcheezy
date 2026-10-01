"""Delayed-live MLB Stats API (GUMBO) poller for the Observer demo; stdlib only.

Only the *game state* comes from the feed. Player features stay frozen: the
pitcher must be pinned in the served bundle (else ``unsupported_pitcher``), the
batter profile is the bundle snapshot or its league row, repertoire counts are
the bundle's TRAIN pins and the zone bounds are a <=2025 league pin from
``live_config.json``. No 2026 row is written into any feature or pin.

Postseason game types are accepted here only; research fetchers are unchanged.

    python -m observer_app.live_feed --date 2026-09-29
    python -m observer_app.live_feed --game 849845 --polls 3 [--record-dir D] [--replay-dir D]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .recommendation_adapter import PrePitchInput

APP = Path(__file__).resolve().parents[2]
LIVE_CONFIG = json.loads((APP/'live_config.json').read_text())
FROZEN_SOURCE_MAX_DATE = '2025-12-31'


class LiveFeedError(RuntimeError):
    pass


def http_json(url, timeout):
    request = urllib.request.Request(url, headers={'User-Agent': 'pitcheezy-observer-demo'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _load(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


class ReplayTransport:
    """Serves recorded ``{gamePk}_{stamp}.json[.gz]`` snapshots in order, then holds the last one."""

    def __init__(self, directory):
        self.directory, self.cursor = Path(directory), {}

    def __call__(self, url, timeout):
        if (match := re.search(r'[?&]date=(\d{4}-\d{2}-\d{2})', url)):
            path = next(self.directory.glob(f'schedule_{match[1]}.json*'), None)
            return _load(path) if path else {'dates': []}
        game = re.search(r'/game/(\d+)/feed/live', url)[1]
        files = sorted(self.directory.glob(f'{game}_*.json*'))
        if not files:
            raise LiveFeedError(f'no recorded snapshot for game {game}')
        index = min(self.cursor.get(game, 0), len(files)-1)
        self.cursor[game] = index+1
        return _load(files[index])


def _wall(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def event_wall(feed):
    """Wall-clock time (epoch s) of the newest game event a snapshot contains, or None.

    A replay transport states it as ``metaData.eventWall``. A real feed gives each play event a
    ``startTime``/``endTime``; a pitch counts from its ``endTime`` so the result is never shown before
    the ball has arrived. ``metaData.timeStamp`` (UTC, the newest event's start) is the floor.
    """
    meta = feed.get('metaData') or {}
    if meta.get('eventWall') is not None:
        return float(meta['eventWall'])
    times = []
    try:
        times.append(datetime.strptime(meta.get('timeStamp') or '', '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        pass
    plays = (feed.get('liveData') or {}).get('plays') or {}
    for play in [plays.get('currentPlay') or {}, *(plays.get('allPlays') or [])[-2:]]:
        for event in play.get('playEvents') or []:
            stamp = event.get('endTime') or event.get('startTime')
            if stamp:
                times.append(_wall(stamp))
    return max(times) if times else None


class LiveFeed:
    """Polls with timeout + exponential backoff and serves each viewer the game as it stood ``delay_s`` ago.

    ``delay_s`` is the viewer's broadcast delay: the served snapshot is the newest one whose newest game
    event (``event_wall``) happened at least that long ago, so a pitch appears here when it appears on a
    broadcast running that far behind the ballpark. A snapshot that reached us later than that is served as
    soon as it is fetched. Snapshots are kept for ``max_delay_s`` so viewers can use different delays.
    """

    def __init__(self, transport=http_json, *, config=LIVE_CONFIG, delay_s=None, record_dir=None,
                 clock=time.monotonic, sleep=time.sleep, wall=time.time):
        self.transport, self.config, self.clock, self.sleep, self.wall = transport, config, clock, sleep, wall
        self.delay_s = config['delay_s'] if delay_s is None else delay_s
        self.max_delay_s = max(config.get('max_delay_s', 120), self.delay_s)
        self.record_dir = Path(record_dir) if record_dir else None
        self.buffers, self.fetched_at, self._lock = {}, {}, threading.Lock()

    def _get(self, url):
        attempts = self.config['retries']+1
        for attempt in range(attempts):
            try:
                return self.transport(url, self.config['timeout_s'])
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise LiveFeedError(f'HTTP {exc.code} for {url}') from exc
                error = exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                error = exc
            if attempt+1 < attempts:
                self.sleep(min(self.config['max_backoff_s'], self.config['backoff_s']*2**attempt))
        raise LiveFeedError(f'feed unavailable after {attempts} attempts: {error}') from error

    def schedule(self, date):
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
            raise ValueError('date must be YYYY-MM-DD')
        data = self._get(f"{self.config['base_url']}/api/v1/schedule?sportId=1&date={date}")
        allowed = self.config['allowed_game_types']
        return [{'game_pk': g['gamePk'], 'game_type': g['gameType'], 'start': g.get('gameDate'),
                 'status': g['status'].get('detailedState'), 'live': g['status'].get('abstractGameState') == 'Live',
                 'away_team': g['teams']['away']['team']['name'], 'home_team': g['teams']['home']['team']['name']}
                for day in data.get('dates', []) for g in day['games'] if g['gameType'] in allowed]

    def _record(self, game_pk, feed):
        stamp = feed.get('metaData', {}).get('timeStamp') or time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        path = self.record_dir/f'{game_pk}_{stamp}.json.gz'
        if not path.exists():  # the feed timestamp only changes with game events
            self.record_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'wt') as handle:
                json.dump(feed, handle)

    def snapshot(self, game_pk, delay_s=None):
        """Returns (entry or None, seconds until this delay has something to show). An entry is
        ``{'feed', 'event', 'fetched'}``: the snapshot, its newest event's wall time and when it was fetched."""
        game_pk = int(game_pk)
        delay = self.delay_s if delay_s is None else min(max(0.0, float(delay_s)), self.max_delay_s)
        with self._lock:
            now, wall = self.clock(), self.wall()
            buffer = self.buffers.setdefault(game_pk, [])
            if now-self.fetched_at.get(game_pk, -1e18) >= self.config['poll_interval_s']:
                feed = self._get(f"{self.config['base_url']}/api/v1.1/game/{game_pk}/feed/live")
                self.fetched_at[game_pk] = now
                event = event_wall(feed)
                entry = {'feed': feed, 'event': wall if event is None else min(event, wall), 'fetched': wall}
                if buffer and entry['event'] <= buffer[-1]['event']:
                    buffer[-1]['feed'] = feed  # no newer game event: keep its first-seen times, refresh the content
                else:
                    buffer.append(entry)
                if self.record_dir:
                    self._record(game_pk, feed)
            old = [i for i, entry in enumerate(buffer) if entry['event'] <= wall-self.max_delay_s]
            if old:
                del buffer[:old[-1]]  # the newest snapshot older than every allowed delay, and everything newer
            ready = [entry for entry in buffer if entry['event'] <= wall-delay]
            if not ready:
                return None, round(max(0.0, buffer[0]['event']+delay-wall), 1)
            return ready[-1], 0.0

    def latest(self, game_pk):
        """The newest fetched (not yet delayed) snapshot, or None."""
        with self._lock:
            buffer = self.buffers.get(int(game_pk))
            return buffer[-1]['feed'] if buffer else None


def _player(feed, player_id):
    return feed['gameData'].get('players', {}).get(f'ID{player_id}', {})


def parse_state(feed, pins, *, config=LIVE_CONFIG):
    """Current pre-pitch game state from a GUMBO feed, plus a ``PrePitchInput`` when supported by the
    legacy bundle ``pins`` (``pins=None``: state only)."""
    data, live = feed['gameData'], feed['liveData']
    line, play = live['linescore'], live['plays'].get('currentPlay') or {}
    game = {'game_pk': data['game']['pk'], 'game_type': data['game']['type'], 'date': data['datetime']['officialDate'],
            'status': data['status'].get('detailedState'), 'feed_timestamp': feed.get('metaData', {}).get('timeStamp'),
            'away_team': data['teams']['away']['name'], 'home_team': data['teams']['home']['name']}
    result = {'status': 'ready', 'reason': None, 'game': game, 'situation': None, 'pitcher': None, 'batter': None,
              'pitch_sequence': [], 'pre_pitch': None}
    if game['game_type'] not in config['allowed_game_types']:
        return result | {'status': 'unsupported_game_type', 'reason': f"game type {game['game_type']} is not in the demo scope"}
    if data['status'].get('abstractGameState') != 'Live':
        return result | {'status': 'not_live', 'reason': game['status']}
    about = play.get('about', {})
    in_pa = bool(play) and not about.get('isComplete', True)
    if not in_pa and (line.get('outs') == 3 or line.get('inningState') in ('Middle', 'End')):
        return result | {'status': 'between_innings', 'reason': '이닝 교대 중입니다.'}
    if in_pa:
        matchup, count = play['matchup'], play['count']
        batter_id, pitcher_id = matchup['batter']['id'], matchup['pitcher']['id']
        stand, hand = matchup['batSide']['code'], matchup['pitchHand']['code']
        inning, top, balls, strikes, outs = about['inning'], about['isTopInning'], count['balls'], count['strikes'], count['outs']
        result['pitch_sequence'] = [
            {'pitch_number': e.get('pitchNumber'), 'pitch_type': (e['details'].get('type') or {}).get('code'),
             'pitch_label': (e['details'].get('type') or {}).get('description'),
             'description': e['details'].get('description'), 'call': (e['details'].get('call') or {}).get('code'),
             'count_after': {'balls': e['count']['balls'], 'strikes': e['count']['strikes']},
             'speed_mph': (e.get('pitchData') or {}).get('startSpeed'),
             'x': ((e.get('pitchData') or {}).get('coordinates') or {}).get('pX'),
             'z': ((e.get('pitchData') or {}).get('coordinates') or {}).get('pZ')}
            for e in play.get('playEvents', []) if e.get('isPitch')]
    else:  # between batters: the next PA starts 0-0 with the linescore's due-up batter
        batter_id, pitcher_id = line['offense']['batter']['id'], line['defense']['pitcher']['id']
        hand = _player(feed, pitcher_id).get('pitchHand', {}).get('code')
        stand = _player(feed, batter_id).get('batSide', {}).get('code')
        if stand == 'S':  # switch hitter bats opposite the pitcher's hand
            stand = {'L': 'R', 'R': 'L'}.get(hand)
        inning, top, balls, strikes, outs = line['currentInning'], line['isTopInning'], 0, 0, line['outs']
    offense = line.get('offense', {})
    runners = {base: base in offense for base in ('first', 'second', 'third')}
    situation = {'inning': inning, 'half': 'Top' if top else 'Bot', 'outs': outs,
                 'bases': runners['first'] + 2*runners['second'] + 4*runners['third'], 'runners': runners,
                 'home_score': line['teams']['home']['runs'], 'away_score': line['teams']['away']['runs'],
                 'balls': balls, 'strikes': strikes}
    result |= {'situation': situation,
               'pitcher': {'id': pitcher_id, 'name': _player(feed, pitcher_id).get('fullName'), 'hand': hand},
               'batter': {'id': batter_id, 'name': _player(feed, batter_id).get('fullName'), 'side': stand}}
    if pins is None:  # ARM-B service view: support is decided by the frozen policy runtime, not bundle pins
        return result
    pin = pins['pitchers'].get(str(pitcher_id))
    profile = pins['profiles'].get(str(batter_id))
    result |= {'situation': situation,
               'pitcher': {'id': pitcher_id, 'name': _player(feed, pitcher_id).get('fullName'), 'hand': hand,
                           'supported': pin is not None},
               'batter': {'id': batter_id, 'name': _player(feed, batter_id).get('fullName'), 'side': stand,
                          'profile_source': 'frozen_snapshot' if profile else 'league_default'}}
    if pin is None:
        return result | {'status': 'unsupported_pitcher', 'reason': '이 투수는 동결된 모델 번들에 없어 추천하지 않습니다.'}
    if stand not in ('L', 'R'):
        return result | {'status': 'unknown_batter_side', 'reason': '타자의 타석 방향을 확인할 수 없습니다.'}
    profile = profile or pins['default_profile']
    # Trust boundary: frozen pins only. A >=2026 as-of would mean a 2026 row reached the features.
    if not profile['as_of'] <= FROZEN_SOURCE_MAX_DATE or not profile['as_of'] < game['date']:
        raise ValueError('batter profile is not a frozen <=2025 snapshot')
    request = {key: situation[key] for key in ('inning', 'outs', 'bases', 'home_score', 'away_score', 'balls', 'strikes')}
    request.update(topbot=situation['half'], pitcher_id=int(pitcher_id), batter_stand=stand, date=game['date'], top_k=3,
                   batter_profile={key: list(profile[key]) for key in ('rates', 'reliabilities')} | {'as_of': profile['as_of']})
    result['pre_pitch'] = PrePitchInput(request, dict(config['zone_bounds']),
                                        dict(pins['repertoire_counts'][str(pitcher_id)]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--date')
    parser.add_argument('--game', type=int)
    parser.add_argument('--polls', type=int, default=1)
    parser.add_argument('--delay', type=float, default=None)
    parser.add_argument('--record-dir')
    parser.add_argument('--replay-dir')
    parser.add_argument('--metadata', help='bundle metadata.json with frozen pins (default: served bundle)')
    args = parser.parse_args(argv)
    transport = ReplayTransport(args.replay_dir) if args.replay_dir else http_json
    feed = LiveFeed(transport, delay_s=0 if args.replay_dir and args.delay is None else args.delay,
                    record_dir=args.record_dir)
    if args.date:
        print(json.dumps(feed.schedule(args.date), ensure_ascii=False, indent=1))
    if args.game:
        if args.metadata:
            pins = json.loads(Path(args.metadata).read_text())
        else:
            from .settings import BUNDLE
            pins = json.loads((BUNDLE/'metadata.json').read_text())
        for poll in range(args.polls):
            if poll:
                time.sleep(feed.config['poll_interval_s'])
            entry, wait = feed.snapshot(args.game)
            state = parse_state(entry['feed'], pins) if entry else {'status': 'buffering', 'wait_s': wait}
            if state.get('pre_pitch'):
                state['pre_pitch'] = state['pre_pitch'].request
            print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
