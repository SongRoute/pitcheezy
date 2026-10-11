"""With the viewer-delay switch OFF (the default) the fan routes answer what the delay-0-fixed code of commit
a72f1fa answered: the current code is compared with the golden bodies that tree produced for the same fixed sequence
of feed states (``golden_viewer_off.py`` says how the fixture was made). ``?delay=`` changes nothing while off."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from golden_viewer_off import scenario
from test_display_state import GAME

GOLDEN = Path(__file__).parent/'fixtures/viewer_off/golden-a72f1fa.json.gz'
# The ONLY differences allowed, all additive top-level keys that a72f1fa did not send:
#   features  the public owner switches (game and list bodies), so the screen needs no rebuild to obey them
#   viewer    live game bodies: the delay this answer was served at (0 while off) and the switch
ADDED = ('features', 'viewer')


def golden():
    with gzip.open(GOLDEN, 'rt') as handle:
        return json.load(handle)


def strip(step):
    return {'status': step['status'], 'body': {key: value for key, value in step['body'].items() if key not in ADDED}}


@pytest.mark.parametrize('switch', [None, '0'])
def test_switched_off_the_fan_routes_answer_exactly_what_a72f1fa_answered(tmp_path, monkeypatch, switch):
    for name in ('VIEWER_DELAY', 'PROFILE_REFERENCE', 'STEP5_EXPECTATION'):  # all three owner switches off
        monkeypatch.delenv(f'PITCHEEZY_OBSERVER_{name}', raising=False)
        if switch is not None:
            monkeypatch.setenv(f'PITCHEEZY_OBSERVER_{name}', switch)
    now, then = scenario(tmp_path), golden()
    assert '"reference"' not in json.dumps(now) and '"expectation"' not in json.dumps(now)  # service_extras adds nothing
    assert sorted(now) == sorted(then) and len(then) == 19
    for name in sorted(then):
        assert set(now[name]['body']) - set(then[name]['body']) <= set(ADDED), name  # nothing else was added
        assert strip(now[name]) == then[name], name
    live = now['01-pending']['body']
    assert live['viewer'] == {'enabled': False, 'default_s': 30, 'provisional': True, 'delay_s': 0.0, 'max_delay_s': 120,
                              'buffering': False, 'wait_s': 0.0}
    assert live['features']['viewer_delay'] == {'enabled': False, 'default_s': 30, 'sync_prompt': False, 'fold_actual_until_synced': False}
    assert 'viewer' not in now['10-list']['body'] and now['10-list']['body']['features'] == live['features']


def test_switched_off_a_delay_parameter_changes_no_body(tmp_path, monkeypatch):
    monkeypatch.delenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', raising=False)
    plain, asked = scenario(tmp_path/'plain'), scenario(tmp_path/'delay', {'delay': 30})
    assert asked == plain  # game bodies (also ``viewer``: served at 0) and the list, score included
    assert {name: strip(step) for name, step in asked.items()} == golden()


def test_switched_off_the_tap_route_refuses_and_reads_nothing(tmp_path, monkeypatch):
    from test_display_state import Run
    monkeypatch.delenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', raising=False)
    run = Run(tmp_path)
    monkeypatch.setattr(run.feed, 'newest', lambda game_pk: pytest.fail('the tap route read the feed while switched off'))
    answer = run.client.get(f'/api/service/games/{GAME}/tap', params={'ago': '1,20', 'current': 30})
    assert answer.status_code == 404 and set(answer.json()) == {'detail'}
    assert run.client.get('/api/service/features').json()['features']['viewer_delay']['enabled'] is False
