"""Catcher-setup estimates of one game from the video module, served per pitch (display only).

``<directory>/<game_pk>.jsonl`` holds one IntentEstimate v1 per line and ``<game_pk>.source.json``
pins where it came from and its sha256. Every line must pass ``validate_intent_estimate``; a file
that differs from its pin or carries one bad line is refused as a whole.

A setup estimate is a proxy for intent and is unreviewed. Only the horizontal position is offered:
the vertical setup position is not served (catchers may hold the glove low until release, so it is
not a target height). Lines exist only for the pitches the source video shows.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .domain import ZONES
from .intent import PLATE_HALF_WIDTH_FEET, validate_intent_estimate

SCHEMA = 'pitcheezy-setup-estimates-v1'
BADGE = '포수 셋업 위치 추정(검토 전) · 투수의 실제 의도가 아님 · 가로 위치만'
DEFAULT_DIR = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/intent')
SOURCE_FIELDS = ('repository', 'branch', 'commit', 'path', 'sha256', 'video', 'selection')
SIDES = [zone['id'].split('_')[1] for zone in ZONES[:3]]  # column 0 = catcher's left (CONTRACT /api/zones)


def x_band(x):
    """Horizontal third of the plate in the catcher's view, or which side it misses on."""
    if abs(x) > PLATE_HALF_WIDTH_FEET:
        return 'off_left' if x < 0 else 'off_right'
    return SIDES[min(2, int((x + PLATE_HALF_WIDTH_FEET) / (2 * PLATE_HALF_WIDTH_FEET) * 3))]


class Unreadable(ValueError):
    """The estimate files exist but cannot be read as estimates at all (unreadable, corrupt, wrong shape), as
    opposed to a readable file that fails validation. A ``ValueError`` for every existing caller."""


class SetupEstimates:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('PITCHEEZY_INTENT_DIR', DEFAULT_DIR))
        self._cache = {}

    def load(self, game_pk):
        if type(game_pk) is not int or game_pk < 1:
            raise ValueError('gamePk must be a positive integer')
        lines, pin = self.directory/f'{game_pk}.jsonl', self.directory/f'{game_pk}.source.json'
        if not lines.is_file() or not pin.is_file():
            raise KeyError(game_pk)
        stamp = (lines.stat().st_mtime_ns, pin.stat().st_mtime_ns)
        cached = self._cache.get(game_pk)
        if cached is None or cached[0] != stamp:
            try:
                cached = self._cache[game_pk] = (stamp, *self._read(game_pk, lines, pin))
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - a pin that is not an object, an unreadable file, a line of the wrong shape
                raise Unreadable('setup estimates are unreadable or malformed') from exc
        return cached[1], cached[2]

    @staticmethod
    def _read(game_pk, lines, pin):
        source, raw = json.loads(pin.read_text()), lines.read_bytes()
        if not set(SOURCE_FIELDS) <= set(source) or hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('setup estimates differ from their pinned source')
        pitches = {}
        for line in raw.decode().splitlines():
            estimate = validate_intent_estimate(json.loads(line))
            game, at_bat, number = (int(part) for part in estimate['pitch_id'].split(':'))
            if game != game_pk or (at_bat, number) in pitches:
                raise ValueError('setup estimate for another game or a repeated pitch')
            pitches[at_bat, number] = estimate
        return source, pitches

    def available(self):
        return self.directory.is_dir()

    def view(self, game_pk):
        """Per-pitch horizontal setup position; pitches without a line are simply absent."""
        source, pitches = self.load(game_pk)
        try:
            return self._view(game_pk, source, pitches)
        except Exception as exc:  # noqa: BLE001 - validated lines should never get here; refuse rather than half-serve
            raise Unreadable('setup estimates are malformed') from exc

    @staticmethod
    def _view(game_pk, source, pitches):
        out = []
        for (at_bat, number), estimate in sorted(pitches.items()):
            feet = estimate['points'].get('plate_feet')
            out.append({'pitch_id': estimate['pitch_id'], 'at_bat_number': at_bat, 'pitch_number': number,
                        'status': estimate['status'], 'unavailable_reason': estimate['unavailable_reason'],
                        'plate_x_feet': None if feet is None else feet['x'],
                        'x_band': None if feet is None else x_band(feet['x']),
                        'label_source': estimate['provenance']['label_source'],
                        'review_status': estimate['provenance']['review_status'],
                        'is_intent_proxy': estimate['is_intent_proxy']})
        located = sum(pitch['plate_x_feet'] is not None for pitch in out)
        return {'schema': SCHEMA, 'badge': BADGE, 'game_pk': game_pk,
                'source': {key: source[key] for key in SOURCE_FIELDS},
                'x_convention': 'statcast_plate_x_catcher_view',
                'display': {'horizontal': True, 'vertical': False,
                            'vertical_reason': '셋업 높이는 목표 높이가 아닐 수 있어 제공하지 않습니다.'},
                'coverage': {'lines': len(out), 'estimated': sum(p['status'] == 'estimated' for p in out),
                             'unavailable': sum(p['status'] == 'unavailable' for p in out), 'with_plate_x': located},
                'accuracy': {'measured': False, 'note': '사람 검토와 다른 경기 정확도 측정 전입니다.'},
                'pitches': out}
