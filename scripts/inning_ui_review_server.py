"""Local UI verification: real decision DB, empty PA catalog, no model or worker.

PITCHEEZY_INNING_REVIEW_DB must name the separately imported review database.
Run with uvicorn --app-dir scripts inning_ui_review_server:create_review_app --factory.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/observer/backend'))
from observer_app.main import create_app, UnavailableRecommender
from observer_app.service import ObserverService
from observer_app.store import Store


class EmptyReplay:
    identity = 'ui-review-empty-pa-catalog'
    data = {'games': []}

    def catalog(self):
        return []

    def get(self, *_args):
        raise KeyError('No PA replay in this review server')


def create_review_app():
    database = Path(os.environ['PITCHEEZY_INNING_REVIEW_DB'])
    if not database.is_file():
        raise ValueError('Import the real C decision into a new review database first')
    return create_app(ObserverService(EmptyReplay(), UnavailableRecommender(), Store(database)), start_worker=False)
