"""With the viewer-delay switch OFF (the default) the fan routes answer exactly the baseline bodies: the current
code is compared with the golden bodies that commit 505932e produced for the same fixed sequence of feed states
(``golden_viewer_off.py`` says how the fixture was made). ``?delay=`` changes nothing while off.

Baseline history. The first baseline (``golden-a72f1fa.json.gz``, kept as history) was the delay-0-fixed code of
a72f1fa. It was rebaselined to 505932e on 2026-10-19 (BLK-06, Song's decision) because three intended commits had
changed the switched-off bodies, and every difference from a72f1fa was traced to exactly one of them (the trees just
before each commit answer the same as the previous one; 505932e answers the same as 75e25189):
  0fe66af9  responsive live UI: added body keys (win_expectancy, context boxscore/lineups, game/list series and
            probable-pitcher fields, pitch tracking) and the event-prefix context of the same-stamp step
  a52012eb  hold accepted pitches, scope the same-stamp gate, correct only on shown changes (Song 2026-10-12,
            independently reviewed): the 05 same-stamp step (held pitch, correction, receipts, recommendation)
  75e25189  policy context inputs in the decision key (reviewed): every ledger decision_id, and the harness feed
            now writes "Called Strike" for a called strike (step 04; with the harness of 75e25189 the a72f1fa-era
            backend differs from the old fixture by exactly these 2 descriptions)"""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from golden_viewer_off import scenario
from test_display_state import GAME

GOLDEN = Path(__file__).parent/'fixtures/viewer_off/golden-505932e.json.gz'
# The ONLY differences allowed, all additive top-level keys that the golden bodies leave out (a72f1fa did not send them):
#   features  the public owner switches (game and list bodies), so the screen needs no rebuild to obey them
#   viewer    live game bodies: the delay this answer was served at (0 while off) and the switch
ADDED = ('features', 'viewer')


def golden():
    """The baseline bodies without ``ADDED``: the 505932e fixture is the generator's raw output, so it also holds the
    switched-off ``features``/``viewer`` (asserted on their own below); a72f1fa's had neither."""
    with gzip.open(GOLDEN, 'rt') as handle:
        return {name: strip(step) for name, step in json.load(handle).items()}


def strip(step):
    return {'status': step['status'], 'body': {key: value for key, value in step['body'].items() if key not in ADDED}}


# Also not compared (replaced by fixed words in ``golden_viewer_off.stored``, in both trees): the random demo session id
# and ``feed.as_of`` of an archive/demo body, which is the wall clock of the request.


@pytest.mark.parametrize('switch', [None, '0'])
def test_switched_off_the_fan_routes_answer_exactly_the_baseline_bodies(tmp_path, monkeypatch, switch):
    for name in ('VIEWER_DELAY', 'PROFILE_REFERENCE', 'STEP5_EXPECTATION'):  # all three owner switches off
        monkeypatch.delenv(f'PITCHEEZY_OBSERVER_{name}', raising=False)
        if switch is not None:
            monkeypatch.setenv(f'PITCHEEZY_OBSERVER_{name}', switch)
    now, then = scenario(tmp_path), golden()
    assert '"reference"' not in json.dumps(now) and '"expectation"' not in json.dumps(now)  # service_extras adds nothing
    assert sorted(now) == sorted(then) and len(then) == 29  # 19 live, 6 archive (2 refusals), 4 demo/embargo
    for name in sorted(then):
        assert set(now[name]['body']) - set(then[name]['body']) <= set(ADDED), name  # nothing else was added
        assert strip(now[name]) == then[name], name
    live = now['01-pending']['body']
    assert live['viewer'] == {'enabled': False, 'default_s': 30, 'provisional': True, 'delay_s': 0.0, 'max_delay_s': 120,
                              'buffering': False, 'wait_s': 0.0}
    assert live['features']['viewer_delay'] == {'enabled': False, 'default_s': 30, 'sync_prompt': False, 'fold_actual_until_synced': False}
    assert 'viewer' not in now['10-list']['body'] and now['10-list']['body']['features'] == live['features']


@pytest.mark.parametrize('value', ['30', '0', '120.5', 'inf', '-1', 'abc', 'nan', ''])
def test_switched_off_a_delay_parameter_changes_no_body(tmp_path, monkeypatch, value):
    """Valid or not, the parameter is not looked at: same status and same body as without it, live, archive, demo,
    refusals (404, 400 for a bad date, 423) and the list included (review A1: an invalid value used to answer 400)."""
    monkeypatch.delenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', raising=False)
    plain, asked = scenario(tmp_path/'plain'), scenario(tmp_path/'delay', {'delay': value})
    assert asked == plain  # game bodies (also ``viewer``: served at 0) and the list, score included
    assert {name: strip(step) for name, step in asked.items()} == golden()
    assert {step['status'] for step in asked.values()} == {200, 400, 404, 423}


@pytest.mark.parametrize('value', ['-1', 'abc', 'nan', ''])
def test_switched_on_an_invalid_delay_is_treated_as_absent(tmp_path, monkeypatch, value):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', '1')
    asked = scenario(tmp_path, {'delay': value})
    assert {name: {'status': step['status'], 'body': {k: v for k, v in strip(step)['body'].items()}} for name, step in asked.items()} == golden()
    assert asked['01-pending']['body']['viewer']['delay_s'] == 0.0


def test_switched_off_the_tap_route_refuses_and_reads_nothing(tmp_path, monkeypatch):
    from test_display_state import Run
    monkeypatch.delenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', raising=False)
    run = Run(tmp_path)
    monkeypatch.setattr(run.feed, 'newest', lambda game_pk: pytest.fail('the tap route read the feed while switched off'))
    answer = run.client.get(f'/api/service/games/{GAME}/tap', params={'ago': '1,20', 'current': 30})
    assert answer.status_code == 404 and set(answer.json()) == {'detail'}
    assert run.client.get('/api/service/features').json()['features']['viewer_delay']['enabled'] is False
