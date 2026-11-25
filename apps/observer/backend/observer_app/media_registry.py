"""Read-only serving of explicitly registered local research clips, no uploads."""
import hashlib
import json
from pathlib import Path


class MediaRegistry:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self._verified = {}

    def _entries(self):
        manifest = self.run/'media_sources.json'
        if not manifest.is_file():
            return []
        return json.loads(manifest.read_text()).get('clips', [])

    def path(self, clip_id):
        entry = next((item for item in self._entries() if item['key'] == clip_id), None)
        if entry is None:
            raise KeyError('등록된 영상을 찾을 수 없습니다.')
        path = Path(entry['local_file']).resolve()
        if not path.is_relative_to(self.run/'media') or not path.is_file() or path.suffix.lower() != '.mp4':
            raise ValueError('등록된 로컬 영상 경로가 올바르지 않습니다.')
        stat = path.stat()
        fingerprint = (str(path), stat.st_size, stat.st_mtime_ns, entry['sha256'])
        if fingerprint not in self._verified:
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('영상 파일 해시가 등록된 출처와 다릅니다.')
            self._verified[fingerprint] = True
        return path

    def catalog(self):
        clips = []
        for entry in self._entries():
            self.path(entry['key'])
            stream = next((stream for stream in entry['ffprobe']['streams'] if stream.get('width')), None)
            if stream is None:
                continue
            numerator, denominator = stream['r_frame_rate'].split('/')
            usable = entry['key'] == 'seven_strikeouts'
            clips.append({'id': entry['key'], 'clip_sha256': entry['sha256'],
                'title': 'Logan Webb · 2025-08-17 삼진 장면' if usable else 'Logan Webb · 투구 통계 화면',
                'source_url': entry['source_page'], 'duration_seconds': float(entry['ffprobe']['format']['duration']),
                'width': stream['width'], 'height': stream['height'], 'fps': float(numerator)/float(denominator),
                'video_url': '/api/video-lab/clips/'+entry['key'], 'usable_for_tracking': usable,
                'reason': '투구 전 미트가 보이는 짧은 구간만 검토하세요. 결과가 선별된 모음입니다.' if usable
                          else '실제 투구 장면이 없는 통계 화면이므로 의도 추정에서 제외합니다.'})
        return {'clips': clips, 'annotations': [], 'limitations': [
            '연구용 영상 검토실입니다. 실제 결과가 포함되어 있습니다.',
            '목표 표시와 추적은 투구 전 구간에 한정합니다. 포구 위치를 의도 정답으로 사용하지 않습니다.',
            '화면상의 존 좌표는 실제 홈플레이트 좌표나 검증된 제구 오차가 아닙니다.',
            'AI 검토 추정과 사람이 입력한 라벨을 구분하고 독립 검토 전에는 정답으로 취급하지 않습니다.']}
