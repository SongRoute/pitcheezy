"""미검토 추정과 사람 검토 라벨을 서로 다른 표에 따로 보관한다.

승격 함수는 없다. 검토 라벨은 reviewer_id 와 decision 이 실린 레코드를 사람이 넣을 때만
생기고, 추정 쪽 행은 그대로 남아 절대 바뀌지 않는다. 두 표를 합쳐 읽는 곳은
``reviewed_labels`` 하나뿐이며 그 결과에는 항상 검토자와 결정이 붙어 나온다.
"""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import threading

from .intent import validate_intent_estimate, validate_reviewed_label
from .video_annotations import encoded, identity


class IntentLabels:
    def __init__(self, run):
        self.run = Path(run)
        self.path = self.run/'intent_labels.sqlite3'
        self.lock = threading.Lock()

    def connect(self):
        self.run.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA foreign_keys=ON')
        db.executescript('''
          CREATE TABLE IF NOT EXISTS estimates(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY, estimate_id TEXT NOT NULL
            REFERENCES estimates(id), reviewer_id TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS reviews_by_estimate ON reviews(estimate_id);
          CREATE TRIGGER IF NOT EXISTS immutable_estimates BEFORE UPDATE ON estimates
            BEGIN SELECT RAISE(ABORT,'immutable intent estimate'); END;
          CREATE TRIGGER IF NOT EXISTS immutable_reviews BEFORE UPDATE ON reviews
            BEGIN SELECT RAISE(ABORT,'immutable review decision'); END;
        ''')
        return db

    def save_estimate(self, raw):
        estimate = validate_intent_estimate(raw)
        if len(encoded(estimate)) > 65536:
            raise ValueError('의도 추정 한 건은 64KB 이하여야 합니다.')
        key = identity(estimate)
        with closing(self.connect()) as db, db:
            db.execute('INSERT OR IGNORE INTO estimates VALUES (?,?)', (key, encoded(estimate)))
        return self.get(key)

    def review(self, estimate_id, raw):
        """사람의 검토 결정을 붙인다. decision 과 reviewer_id 에 기본값은 없다."""
        review = validate_reviewed_label({**raw, 'source_estimate_id': estimate_id})
        with self.lock, closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM estimates WHERE id=?', (estimate_id,)).fetchone() is None:
                raise KeyError('검토할 의도 추정을 찾을 수 없습니다.')
            db.execute('INSERT OR IGNORE INTO reviews VALUES (?,?,?,?)',
                       (identity(review), estimate_id, review['reviewer_id'], encoded(review)))
        return self.get(estimate_id)

    def get(self, estimate_id):
        with closing(self.connect()) as db:
            row = db.execute('SELECT payload FROM estimates WHERE id=?', (estimate_id,)).fetchone()
            if row is None:
                raise KeyError('저장된 의도 추정을 찾을 수 없습니다.')
            reviews = [json.loads(item['payload']) for item in
                       db.execute('SELECT payload FROM reviews WHERE estimate_id=? ORDER BY rowid', (estimate_id,))]
        return {'id': estimate_id, 'estimate': json.loads(row['payload']), 'reviews': reviews,
                'review_status': 'reviewed' if any(r['decision'] != 'rejected' for r in reviews) else 'unreviewed',
                'reviewer_ids': sorted({r['reviewer_id'] for r in reviews})}

    def list(self, limit=200):
        with closing(self.connect()) as db:
            keys = [row['id'] for row in db.execute('SELECT id FROM estimates ORDER BY rowid DESC LIMIT ?', (limit,))]
        return {'estimates': [self.get(key) for key in keys]}

    def reviewed_labels(self):
        """검토 라벨을 얻는 유일한 경로. 검토자와 결정 없이는 아무것도 나오지 않는다."""
        return {'labels': [record for record in self.list()['estimates']
                           if record['review_status'] == 'reviewed']}

    def sample_composition(self):
        """확보한 라벨 수와 표본 구성 실측. 추정하지 않고 저장된 것만 센다."""
        records = self.list(limit=100000)['estimates']
        reviewed = [record for record in records if record['review_status'] == 'reviewed']
        decisions = {}
        reviewers = set()
        for record in records:
            for review in record['reviews']:
                decisions[review['decision']] = decisions.get(review['decision'], 0)+1
                reviewers.add(review['reviewer_id'])
        by_clip = {}
        for record in records:
            by_clip[record['estimate']['clip_id']] = by_clip.get(record['estimate']['clip_id'], 0)+1
        return {'estimates': len(records), 'reviewed_labels': len(reviewed),
                'unreviewed_estimates': len(records)-len(reviewed), 'review_decisions': decisions,
                'reviewers': sorted(reviewers), 'estimates_by_clip': by_clip,
                'estimates_with_zone9': sum(1 for record in records
                                            if record['estimate'].get('deepest_frame') == 'zone9'),
                'selection_bias': '표본 구성은 저장된 클립의 선정 규칙을 그대로 따른다. '
                                  '여기서는 세지 않으므로 클립별 선정 근거를 문서에서 확인할 것.'}
