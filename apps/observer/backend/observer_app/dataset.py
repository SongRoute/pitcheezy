import hashlib
import json
from pathlib import Path
from .domain import PITCH_LABELS, RESULT_LABELS


class DemoDataset:
    def __init__(self, path):
        path = Path(path)
        manifest = json.loads(path.with_name('dataset_manifest.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
            raise RuntimeError('데모 데이터 해시가 일치하지 않습니다.')
        self.data = json.loads(path.read_text())
        if self.data['schema_version'] != 1:
            raise RuntimeError('지원하지 않는 데모 데이터 버전입니다.')
        self.identity = manifest['sha256']
        self.games = {game['id']: game for game in self.data['games']}

    def catalog(self):
        return [{key: game[key] for key in ('id', 'date', 'home_team', 'away_team', 'title')} |
                {'plate_appearances': [{key: pa[key] for key in ('id', 'batter_label', 'pitcher_label', 'inning', 'half')}
                                      | {key: pa[key] for key in ('pitcher_id', 'batter_id') if key in pa}
                                      for pa in game['plate_appearances']]}
                for game in self.games.values()]

    def get(self, game_id, pa_id):
        game = self.games.get(game_id)
        if game is None:
            raise KeyError('경기를 찾을 수 없습니다.')
        pa = next((pa for pa in game['plate_appearances'] if pa['id'] == pa_id), None)
        if pa is None:
            raise KeyError('타석을 찾을 수 없습니다.')
        return game, pa

    @staticmethod
    def reveal(pitch, recommendation, bounds):
        actual = pitch['actual']
        result = actual.get('event') or actual['description']
        return {'id': pitch['id'], 'pitch_number': pitch['pitch_number'], 'pitch_type': actual['pitch_type'],
                'pitch_label': PITCH_LABELS.get(actual['pitch_type'], actual['pitch_type'] or '미상'),
                'x': actual['x'], 'z': actual['z'], 'speed_mph': actual['speed_mph'],
                'description': actual['description'], 'result_label': RESULT_LABELS.get(result, result),
                'pre_state': pitch['state'], 'recommendation': recommendation, 'zone_bounds': bounds}
