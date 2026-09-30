"""Watch-along replay of one completed game precomputed by ``scripts/demo_precompute.py``.

Display only (DEMO-WS-2026): the dataset holds the frozen <=2025 ARM-B recommendation made before
each pitch and the actual pitch after it. The pre-pitch view never carries the actual pitch, its
result, the WE change or the final score; those come only from ``reveal``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = 'pitcheezy-watch-along-v1'
DEFAULT_DIR = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/watch')
PRE_FIELDS = ('index', 'pa_id', 'at_bat_number', 'pitch_number', 'situation', 'pitcher', 'batter', 'status', 'pre')
POST_FIELDS = ('index', 'actual', 'we')


class WatchAlong:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('PITCHEEZY_WATCH_DIR', DEFAULT_DIR))
        self._cache = {}

    def _path(self, game_pk):
        if type(game_pk) is not int or game_pk < 1:
            raise ValueError('gamePk must be a positive integer')
        return self.directory/f'{game_pk}.json'

    def load(self, game_pk):
        path = self._path(game_pk)
        if not path.is_file():
            raise KeyError(game_pk)
        stamp = path.stat().st_mtime_ns
        cached = self._cache.get(game_pk)
        if cached is None or cached[0] != stamp:
            data = json.loads(path.read_text())
            if data.get('schema') != SCHEMA or data.get('game', {}).get('game_pk') != game_pk:
                raise ValueError('watch-along dataset schema or gamePk differs')
            if [d['index'] for d in data['decisions']] != list(range(len(data['decisions']))):
                raise ValueError('watch-along decisions must be indexed 0..n-1 in game order')
            cached = self._cache[game_pk] = (stamp, data)
        return cached[1]

    def games(self):
        out = []
        for path in sorted(self.directory.glob('*.json')) if self.directory.is_dir() else []:
            if not path.stem.isdigit():
                continue
            try:
                game = self.load(int(path.stem))['game']
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
            out.append({k: game[k] for k in ('game_pk', 'date', 'game_type', 'away_team', 'home_team')})
        return out

    def timeline(self, game_pk):
        """Everything a viewer may see before any pitch is revealed (no final score)."""
        data = self.load(game_pk)
        game = {k: v for k, v in data['game'].items() if k != 'final'}
        return {'schema': SCHEMA, 'badge': data['badge'], 'game': game,
                'policy': {k: data['policy'][k] for k in ('name', 'identity_sha256', 'tau', 'validated', 'note')},
                'location': {k: data['location'][k] for k in ('basis', 'evaluated', 'note', 'zone_bounds')},
                'we_note': data['we']['note'], 'coverage': {k: data['coverage'][k] for k in
                                                            ('pitch_decisions', 'ready', 'ready_share_of_pitches')},
                'decisions': [{k: d[k] for k in PRE_FIELDS} for d in data['decisions']]}

    def reveal(self, game_pk, index):
        decisions = self.load(game_pk)['decisions']
        if type(index) is not int or not 0 <= index < len(decisions):
            raise IndexError(index)
        return {k: decisions[index][k] for k in POST_FIELDS}
