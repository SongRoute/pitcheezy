"""Explicit public views: only advancing exposes the next recorded actual pitch."""
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from .settings import BUNDLE

from .dataset import DemoDataset
from .domain import ZONE_BY_ID, spatial_comparison
from .settings import CONFIG
from .store import Store

LOGGER = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


class ObserverService:
    def __init__(self, dataset, recommender, store):
        self.dataset, self.recommender, self.store = dataset, recommender, store

    def catalog(self):
        return {'games': self.dataset.catalog(), 'model_version': CONFIG['model_version'],
                'coverage': {key: value for key, value in self.dataset.data.get('coverage', {}).items()
                             if key in ('totals', 'support_rate', 'gap_notice')},
                'limitations': ['기록 재생이며 실시간 중계가 아닙니다.',
                    '목표 구역은 근사 모델의 제안이며 실제 승률 향상을 입증하지 않습니다.',
                    '영상이 없어 포수의 실제 의도와 자동 영상 분석을 제공하지 않습니다.']}

    def _recommend(self, pitch, pa):
        try:
            if not self.recommender.ready:
                raise RuntimeError('model unavailable')
            return self.recommender.recommend(pitch, pa)
        except Exception:
            LOGGER.exception('Recommendation unavailable for pitch %s', pitch['id'])
            # A recommendation failure must not prevent historical replay.
            return {'id': hashlib.sha256((pitch['id']+CONFIG['model_version']).encode()).hexdigest(),
                    'status': 'unavailable', 'mode': 'experimental_location_proxy',
                    'model_version': CONFIG['model_version'], 'candidates': [], 'baseline_value': None,
                    'zone_bounds': pa['zone_bounds'], 'basis': [],
                    'reason': '이 상황의 추천을 준비하지 못했습니다. 기록 재생은 계속할 수 있습니다.'}

    def _session(self, db, session_id):
        session = db.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        if session is None:
            raise ServiceError(404, '관찰 세션을 찾을 수 없습니다.')
        if session['dataset_identity'] != self.dataset.identity:
            raise ServiceError(409, '데이터가 변경되었습니다. 새 관찰 세션을 시작해 주세요.')
        game, pa = self._pa(session['game_id'], session['pa_id'])
        return session, game, pa

    def _pa(self, game_id, pa_id):
        try:
            game, pa = self.dataset.get(game_id, pa_id)
        except KeyError:
            raise ServiceError(404, '경기 또는 타석을 찾을 수 없습니다.') from None
        if not pa['pitches']:
            raise ServiceError(503, '이 타석에는 재생할 투구가 없습니다.')
        return game, pa

    def create(self, game_id, pa_id):
        _, pa = self._pa(game_id, pa_id)
        session_id = str(uuid.uuid4())
        rec = self._recommend(pa['pitches'][0], pa)
        with self.store.transaction() as db:
            db.execute('INSERT INTO sessions(id,game_id,pa_id,dataset_identity,created) VALUES (?,?,?,?,?)',
                       (session_id, game_id, pa_id, self.dataset.identity, time.time()))
            Store.save_recommendation(db, session_id, pa['pitches'][0]['id'], rec)
            return self._view(db, session_id)

    def get(self, session_id):
        self._ensure_event_result(session_id)
        # A transaction makes cursor/recommendations/job reads one consistent snapshot.
        with self.store.transaction(write=False) as db:
            return self._view(db, session_id)

    def advance(self, session_id, revision):
        # Read the expected cursor, then compute without holding a SQLite writer
        # lock. A competing advance may win; the second CAS check below rejects
        # this result instead of publishing a recommendation for a stale cursor.
        with self.store.transaction(write=False) as db:
            session, _, pa = self._session(db, session_id)
            if session['revision'] != revision:
                raise ServiceError(409, '상태가 변경되었습니다. 현재 화면을 새로 불러와 주세요.')
            cursor = session['cursor']
            if cursor >= len(pa['pitches']):
                raise ServiceError(409, '이미 완료된 타석입니다.')
            next_cursor = cursor+1
        recommendation = self._recommend(pa['pitches'][next_cursor], pa) if next_cursor < len(pa['pitches']) else None
        with self.store.transaction() as db:
            session, _, pa = self._session(db, session_id)
            if session['revision'] != revision:
                raise ServiceError(409, '상태가 변경되었습니다. 현재 화면을 새로 불러와 주세요.')
            cursor = session['cursor']
            if cursor >= len(pa['pitches']):
                raise ServiceError(409, '이미 완료된 타석입니다.')
            next_cursor = cursor+1
            result = db.execute('UPDATE sessions SET cursor=?,revision=revision+1 WHERE id=? AND revision=?',
                                (next_cursor, session_id, revision))
            if result.rowcount != 1:
                raise ServiceError(409, '상태가 변경되었습니다.')
            if next_cursor < len(pa['pitches']):
                pitch = pa['pitches'][next_cursor]
                Store.save_recommendation(db, session_id, pitch['id'], recommendation)
            else:
                Store.enqueue(db, pa['pitches'][-1]['id'], CONFIG['cv_version'])
            view = self._view(db, session_id)
        if view['complete']:
            return self.get(session_id)
        return view

    def _ensure_event_result(self, session_id):
        """Value the revealed PA outside the SQLite writer lock, once per input revision."""
        with self.store.transaction(write=False) as db:
            session, _, pa = self._session(db, session_id)
            if session['cursor'] != len(pa['pitches']):
                return
            pitch = pa['pitches'][-1]
            if Store.event_result(db, session_id, pitch['id']):
                return
            recommendation = Store.recommendation(db, session_id, pitch['id'])
            metadata = db.execute('SELECT created_at FROM recommendation_meta WHERE session_id=? AND pitch_id=?',
                                  (session_id, pitch['id'])).fetchone()
        if recommendation is None or metadata is None:
            return  # Legacy sessions have no trustworthy pre-pitch recommendation timestamp.
        try:
            result = self._calculate_event(session_id, pitch, pa, recommendation, metadata['created_at'])
        except Exception:
            LOGGER.exception('Event analysis failed for revealed pitch %s', pitch['id'])
            result = self._failed_event(session_id, pitch, recommendation, metadata['created_at'])
        if result is None:
            return
        with self.store.transaction() as db:
            session, _, _ = self._session(db, session_id)
            if session['cursor'] == len(pa['pitches']):
                Store.save_event_result(db, session_id, pitch['id'], result)

    @staticmethod
    def _failed_event(session_id, pitch, recommendation, created_at):
        """A calculation exception must never become invented numeric attribution."""
        pitch_id = pitch['id']
        missing = lambda: {'value_pp': None, 'status': 'unavailable', 'reason': 'calculation_error', 'abs_share': None}
        model_sha = hashlib.sha256((BUNDLE/'bundle_manifest.json').read_bytes()).hexdigest() if (BUNDLE/'bundle_manifest.json').is_file() else 'unavailable-frozen-evaluator'
        return {'schema_version': 'event-analysis-v1', 'analysis_id': f'{session_id}:{pitch_id}',
                'revision': 1, 'status': 'failed', 'reason': 'calculation_error',
                'linkage': {'session_id': session_id, 'pitch_id': pitch_id,
                            'recommendation_id': recommendation['id'], 'recommendation_created_at': created_at,
                            'recommendation_sha256': hashlib.sha256(json.dumps(recommendation, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                            'event_input_revision': 1},
                'identity': {'model_version': recommendation.get('model_version') or CONFIG['model_version'],
                             'model_sha256': model_sha, 'value_spec_version': 'defense-we-pa-v1',
                             'baseline_policy_id': 'observer-repertoire-kernel-v1'},
                'scope': {'horizon': 'current_pa', 'initial_defender': 'home' if pitch['state']['half'] == 'Top' else 'away',
                          'unit': 'defense_win_probability', 'difference_unit': 'percentage_points'},
                'values': {'reference': None, 'plan': None, 'execution': None, 'observed': None, 'total_pp': None},
                'components': {'strategy_contrast_pp': missing(), 'execution_contrast_pp': missing(),
                               'outcome_residual_pp': missing(), 'unallocated_residual_pp': None},
                'shares': {'stable': False, 'denominator_pp': None},
                'interactions': {'status': 'unallocated', 'value_pp': None, 'reason': 'calculation_error'},
                'evidence': {'actual_source': 'historical_replay_record', 'intent_source': None,
                             'intent_is_proxy': None, 'intent_review_status': None, 'development_only': False,
                             'references': [pitch_id], 'use_for_performance_evaluation': False},
                'provenance': {'received_at': datetime.now(timezone.utc).isoformat(),
                               'generated_at': datetime.now(timezone.utc).isoformat(), 'correction_of_revision': None},
                'replacement': {'status': 'unavailable', 'horizon': 'inning_end', 'value_pp': None,
                                'reason': 'missing_contemporaneous_candidates_and_inning_evaluator'}}

    def _calculate_event(self, session_id, pitch, pa, recommendation, created_at):
        try:
            from .event_analysis import analyze_event, frozen_observed_value, recommendation_sha256, value_point
        except ImportError:
            return None  # C module is integrated in the shared checkout after its handoff.
        now = datetime.now(timezone.utc).isoformat()
        engine = getattr(self.recommender, 'engine', None)
        model_sha = hashlib.sha256((BUNDLE/'bundle_manifest.json').read_bytes()).hexdigest() if engine else 'unavailable-frozen-evaluator'
        identity = {'model_version': recommendation.get('model_version') or CONFIG['model_version'],
                    'model_sha256': model_sha, 'value_spec_version': 'defense-we-pa-v1',
                    'baseline_policy_id': 'observer-repertoire-kernel-v1'}
        defender = 'home' if pitch['state']['half'] == 'Top' else 'away'
        reference = recommendation.get('baseline_value') if recommendation.get('status') == 'ready' else None
        values = {'reference': value_point(reference, identity=identity, initial_defender=defender,
                                          source='saved_pre_pitch_policy_baseline') if isinstance(reference, (int, float)) else None,
                  'plan': None, 'execution': None, 'observed': None}
        if engine is not None:
            from pitchmdp.game import GameState
            def game_state(row):
                return GameState(row['inning'], row['half'], row['outs'], row['bases'],
                                 row['home_score'], row['away_score'])
            values['observed'] = frozen_observed_value(engine=engine, initial_state=game_state(pitch['state']),
                                                        post_state=game_state(pa['terminal_state']), identity=identity)
        return analyze_event(
            linkage={'session_id': session_id, 'pitch_id': pitch['id'],
                     'recommendation_id': recommendation['id'], 'recommendation_created_at': created_at,
                     'recommendation_sha256': recommendation_sha256(recommendation), 'event_input_revision': 1},
            identity=identity, initial_defender=defender, values=values,
            evidence={'actual_source': 'historical_replay_record', 'action_mapping': None,
                      'references': [pitch['id']], 'development_only': False,
                      'use_for_performance_evaluation': False},
            provenance={'received_at': now, 'generated_at': now, 'correction_of_revision': None},
            stored_recommendation=recommendation)

    def manual_intent(self, session_id, revision, zone_id):
        if zone_id not in ZONE_BY_ID:
            raise ServiceError(400, '올바른 목표 구역을 선택해 주세요.')
        with self.store.transaction() as db:
            session, _, pa = self._session(db, session_id)
            if session['revision'] != revision:
                raise ServiceError(409, '상태가 변경되었습니다. 현재 화면을 새로 불러와 주세요.')
            if session['cursor'] < len(pa['pitches']):
                raise ServiceError(409, '타석이 끝난 뒤 마지막 공의 목표 구역을 입력할 수 있습니다.')
            db.execute('''INSERT INTO manual_annotations(session_id,pitch_id,zone_id,updated) VALUES (?,?,?,?)
                       ON CONFLICT(session_id) DO UPDATE SET zone_id=excluded.zone_id,
                       revision=manual_annotations.revision+1,updated=excluded.updated''',
                       (session_id, pa['pitches'][-1]['id'], zone_id, time.time()))
            db.execute('UPDATE sessions SET revision=revision+1 WHERE id=? AND revision=?', (session_id, revision))
            return self._view(db, session_id)

    def _view(self, db, session_id):
        session, game, pa = self._session(db, session_id)
        cursor, pitches = session['cursor'], pa['pitches']
        complete = cursor == len(pitches)
        history = [DemoDataset.reveal(pitch, Store.recommendation(db, session_id, pitch['id']), pa['zone_bounds'])
                   for pitch in pitches[:cursor]]
        current = None if complete else pitches[cursor]
        analysis, summary = None, None
        if complete:
            last = history[-1]
            job = db.execute('SELECT * FROM jobs WHERE pitch_id=? AND version=?',
                             (last['id'], CONFIG['cv_version'])).fetchone()
            manual = db.execute('SELECT * FROM manual_annotations WHERE session_id=?', (session_id,)).fetchone()
            analysis = {'id': job['id'], 'status': 'complete' if manual else job['status'],
                        'source': 'manual' if manual else 'none', 'selected_pitch_id': last['id'],
                        'selected_pitch_number': last['pitch_number'], 'selection_reason': '타석의 마지막 공.',
                        'cv_status': job['cv_status'],
                        'message': '입력한 목표 구역과 실제 위치를 비교했습니다.' if manual else
                                   (job['message'] or '영상이 없어 자동 의도 분석은 제공되지 않습니다.'),
                        'manual_zone_id': manual['zone_id'] if manual else None,
                        'comparisons': spatial_comparison(last, manual['zone_id']) if manual else None,
                        'narrative': ['마지막 공의 결과는 '+last['result_label']+'입니다.',
                                      '영상이 없어 포수의 실제 목표는 확인할 수 없습니다.'],
                        'version': job['revision']+(manual['revision'] if manual else 0)}
            if manual:
                analysis['narrative'].append('목표 구역은 사용자가 입력했으며 실제 의도를 확인한 값은 아닙니다.')
            summary = {'headline': '타석 관찰 완료', 'result_label': last['result_label'],
                       'pitch_count': len(history), 'selected_pitch_number': last['pitch_number'],
                       'notes': ['추천과 실제 결과의 비교이며 추천의 인과적 효과를 평가하지 않습니다.']}
        return {'id': session_id, 'revision': session['revision'], 'cursor': cursor, 'complete': complete,
                'game': {key: game[key] for key in ('id','date','home_team','away_team','title')},
                'plate_appearance': {key: pa[key] for key in ('id','batter_label','pitcher_label','batter_stand')} |
                                    {key: pa[key] for key in ('pitcher_id', 'batter_id') if key in pa},
                'state': pa['terminal_state'] if complete else current['state'],
                'recommendation': None if complete else Store.recommendation(db, session_id, current['id']),
                'last_pitch': history[-1] if history else None, 'history': history,
                'analysis': analysis, 'event_analysis': Store.event_result(db, session_id, pitches[-1]['id']) if complete else None,
                'summary': summary,
                'context_notes': [] if complete else current.get('context_notes', []),
                'context': None if complete else current.get('context'),
                'notices': ['기록 재생 · 실제 승률 향상이 검증된 추천은 아닙니다.']}
