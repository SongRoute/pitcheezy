"""A2 reference card (``next.reference``) on the PUBLIC publish path (owner decision 2026-10-18: on for the World
Series publish, sourced from the daily refresh pointer the operator names in the environment).

Covered: the publisher's own switch (``publish_config.json`` ``profile_reference``) and its environment kill switch,
the strict contract block, the audit invariant ``reference_card_integrity``, a missing or unusable pointer (no card,
no failure), the as-of rule (cutoff strictly before officialDate) and the staleness mark (cutoff more than 2 days
before officialDate). Synthetic games, a fake policy and synthetic committed profiles only; the real module runs
only where its pinned Step 5 source exists (those tests skip elsewhere). No network, no model, no fit."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_app import features as feature_flags, profile_reference as PR, publish_audit, publish_schema, publisher
from publish_fixture import LIVE_GAME, World, reference_world

CONTRACT_DIR = Path(__file__).resolve().parents[2]/'publish_contract/v1'
GAME_SCHEMA = json.loads((CONTRACT_DIR/'schema/game.schema.json').read_text())
PINNED = all(Path(PR.SOURCE[k]).is_file() for k in PR.SOURCE['pins'])
pinned = pytest.mark.skipif(not PINNED, reason='pinned Step 5 source is not on this machine')
FORBIDDEN = {'rank', 'target', 'zone_id', 'zone_label', 'x', 'z', 'probability', 'candidates', 'provenance', 'count', 'share'}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ('PITCHEEZY_OBSERVER_PROFILE_REFERENCE', 'PITCHEEZY_OBSERVER_PROFILE_SOURCES'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def worlds(tmp_path):
    made = []

    def make(kind='reference', **options):
        folder = tmp_path/f'w{len(made)}'
        made.append(reference_world(folder, **options) if kind == 'reference' else World(folder, **options))
        return made[-1]
    yield make
    for world in made:
        world.close()


def keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*map(keys, value.values()))
    return set().union(*map(keys, value)) if isinstance(value, list) else set()


def audit_all(world, **options):
    audit = publish_audit.PublishAudit(**options)
    for path in world.files():
        data = world.path(path).read_bytes()
        audit.check(path, data, hashlib.sha256(data).hexdigest())
    return audit


# ---------- configuration ----------
def test_the_committed_publish_config_turns_the_card_on_with_the_path_left_to_the_operator():
    config = publisher.load_config(env={})
    assert config['profile_reference'] == {'enabled': True, 'profile_sources': None}
    assert publisher.feature_base(config) == {'profile_reference': {'enabled': True}}
    note = config['profile_reference_note']
    assert 'PITCHEEZY_OBSERVER_PROFILE_SOURCES' in note and '/Users/' not in json.dumps(config['profile_reference'])
    # the reader's own provisional rules stay as they were (Song 2026-10-18): 20 per cell, no broad fallback
    assert PR.SAMPLE_RULE['cell_min_pitches'] == 20 and PR.FRESHNESS['stale_after_days'] == 2
    assert PR.ProfileReference([]).health()['broad_fallback'] is False


@pytest.mark.parametrize('block', [{'enabled': 'yes', 'profile_sources': None}, {'enabled': True, 'profile_sources': '/a'},
                                   {'enabled': True, 'profile_sources': ['']}, ['enabled']])
def test_a_malformed_profile_reference_block_refuses_to_load(block):
    with pytest.raises(ValueError, match='profile_reference'):
        publisher.load_config(env={}, profile_reference=block)


def test_without_the_block_the_server_switches_apply_as_before(tmp_path):
    raw = json.loads(publisher.CONFIG_PATH.read_text())
    raw.pop('profile_reference')
    (tmp_path/'c.json').write_text(json.dumps(raw))
    config = publisher.load_config(tmp_path/'c.json', env={})
    assert publisher.feature_base(config) is None


def test_the_layers_are_live_config_then_publisher_then_environment(monkeypatch):
    live = {'features': {'profile_reference': {'enabled': False, 'profile_sources': ['live']}}}
    base = {'profile_reference': {'enabled': True}}
    assert feature_flags.features(live, base)['profile_reference'] == {'enabled': True, 'profile_sources': ['live']}
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_SOURCES', 'one'+__import__('os').pathsep+'two')
    assert feature_flags.features(live, base)['profile_reference']['profile_sources'] == ['one', 'two']
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_REFERENCE', '0')
    assert feature_flags.features(live, base)['profile_reference']['enabled'] is False  # the kill switch wins
    assert feature_flags.features(live)['profile_reference']['enabled'] is False  # no base: as before


# ---------- enabled / disabled on the published file ----------
def test_enabled_the_published_file_carries_the_public_card_only(worlds):
    world = worlds()
    world.cycle()
    body = world.game()
    card = body['next']['reference']
    assert body['display_state']['reason'] == 'unknown_pitcher' and body['next']['rec']['candidates'] == []
    assert body['features']['profile_reference'] == {'enabled': True}
    assert set(card) == {'kind', 'reason_code', 'chip', 'title', 'items', 'other', 'sample', 'as_of', 'stale', 'note'}
    assert not keys(card) & FORBIDDEN  # no rank, location, raw counts or provenance
    assert card['as_of'] < body['game']['official_date'] and card['stale'] is False
    assert publish_schema.validate(body, GAME_SCHEMA) == []
    assert world.reference.asked[-1][4] == body['game']['official_date']
    audit = audit_all(world)
    assert audit.violations == [] and audit.checked > 0


@pytest.mark.parametrize('how', ['config', 'environment'])
def test_disabled_there_is_no_card_and_the_module_is_not_asked(worlds, monkeypatch, how):
    if how == 'config':
        world = worlds(profile_reference={'enabled': False, 'profile_sources': ['synthetic-profile-pointer']})
    else:
        monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_REFERENCE', '0')
        world = worlds()
    world.cycle()
    body = world.game()
    assert 'reference' not in body['next'] and body['features']['profile_reference'] == {'enabled': False}
    assert world.reference.asked == []
    assert publish_schema.validate(body, GAME_SCHEMA) == []


def test_on_but_without_a_source_nothing_is_attached_and_publishing_goes_on(worlds):
    world = worlds('plain', profile_reference={'enabled': True, 'profile_sources': None})
    world.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    world.cycle()
    body = world.game()
    assert body['display_state']['reason'] == 'unknown_pitcher' and 'reference' not in body['next']
    assert publisher.reference_status(world.app) == {'enabled': True, 'sources': 0}


def test_a_supported_pitcher_never_gets_a_card(worlds):
    world = worlds()
    world.policy.status = None
    world.cycle()
    body = world.game()
    assert body['display_state']['state'] == 'recommended' and 'reference' not in body['next']


# ---------- contract and audit ----------
def test_the_contract_example_with_a_card_validates_and_the_schema_is_strict_about_the_card():
    example = json.loads((CONTRACT_DIR/'examples/game-live-unsupported-reference.json').read_text())
    assert publish_schema.validate(example, GAME_SCHEMA) == []
    assert 'reference' not in GAME_SCHEMA['properties']['next'].get('required', [])
    changes = [lambda c: c.update(kind='recommendation'), lambda c: c.update(as_of='10/04'),
               lambda c: c['items'][0].update(share_pct=101), lambda c: c['items'][0].update(share_pct=55.5),
               lambda c: c['items'][0].pop('pitch_label'), lambda c: c.pop('stale'), lambda c: c.update(stale='no'),
               lambda c: c['sample'].update(batter_side='S'), lambda c: c['sample'].update(balls=4),
               lambda c: c.update(chip='추천'), lambda c: c.update(other={'pct': 1})]
    for change in changes:
        broken = copy.deepcopy(example)
        change(broken['next']['reference'])
        assert publish_schema.validate(broken, GAME_SCHEMA), change


def test_the_audit_refuses_a_card_that_is_not_a_frequency_card():
    example = json.loads((CONTRACT_DIR/'examples/game-live-unsupported-reference.json').read_text())

    def violations(change):
        body = copy.deepcopy(example)
        change(body)
        data = json.dumps(body).encode()
        audit = publish_audit.PublishAudit()
        audit.check(f'v1/game/{LIVE_GAME}.json', data, hashlib.sha256(data).hexdigest())
        return [v for v in audit.violations if v['invariant'] == 'reference_card_integrity']

    assert violations(lambda b: None) == []
    bad = {
        'rank': lambda b: b['next']['reference']['items'][0].update(rank=1),
        'target': lambda b: b['next']['reference'].update(target={'x': 0, 'z': 2}),
        'as_of_same_day': lambda b: b['next']['reference'].update(as_of=b['game']['official_date']),
        'as_of_after': lambda b: b['next']['reference'].update(as_of='2026-10-06'),
        'shares': lambda b: b['next']['reference']['other'].update(share_pct=50),
        'kind': lambda b: b['next']['reference'].update(kind='armb'),
        'supported_state': lambda b: b['display_state'].update(state='recommended', recommendation_shown=True),
        'game_reason': lambda b: b['display_state'].update(reason='at_bat_rejected'),
        'past_phase': lambda b: b['current'].update(phase='between_innings'),
        'stale_missing': lambda b: b['next']['reference'].pop('stale'),
    }
    for name, change in bad.items():
        found = violations(change)
        assert found and all('/next/reference' == v['pointer'] for v in found), name
        assert all(example['game']['official_date'] not in v['detail'] for v in found)  # names rules, not values
    assert 'reference_card_integrity' in publish_audit.INVARIANTS


# ---------- the real module behind the publisher: missing pointer, as-of rule, staleness ----------
def real_world(worlds, monkeypatch, sources):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_PROFILE_SOURCES', __import__('os').pathsep.join(map(str, sources)))
    world = worlds('plain', profile_reference={'enabled': True, 'profile_sources': None})
    world.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
    return world


def test_a_missing_pointer_gives_no_card_and_no_failure(worlds, monkeypatch, tmp_path):
    world = real_world(worlds, monkeypatch, [tmp_path/'nowhere'/'LATEST.json'])
    health = world.cycle()
    body = world.game()
    assert 'reference' not in body['next'] and str(LIVE_GAME) not in health['unavailable_games']
    assert body['display_state']['reason'] == 'unknown_pitcher'
    status = publisher.reference_status(world.app)
    assert status['enabled'] is True and status['sources'] == 1 and [s['usable'] for s in status['usable']] == [False]
    assert '/nowhere' not in json.dumps(body)  # the operator reason never reaches the file


def _pitcher(world):
    world.cycle(settle=False)
    return world.game()['current']['pitcher']['id'], world.game()['current']['batter']['side']


@pinned
@pytest.mark.parametrize('cutoff, card, stale', [('2026-10-05', False, None), ('2026-10-06', False, None),
                                                 ('2026-10-04', True, False), ('2026-10-03', True, False),
                                                 ('2026-10-02', True, True), ('2026-09-20', True, True)])
def test_as_of_rule_and_staleness_through_the_daily_pointer(worlds, monkeypatch, tmp_path, cutoff, card, stale):
    """officialDate of the fixture game is 2026-10-05: a cutoff on or after it is never used; one or two days before
    is fresh; more than two days before is served with ``stale: true``."""
    from test_profile_reference import cell, pointer, write_profile
    probe = World(tmp_path/'probe', profile_reference={'enabled': False, 'profile_sources': None})
    try:
        probe.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
        pitcher, side = _pitcher(probe)
        situation = probe.game()['current']
    finally:
        probe.close()
    balls, strikes = situation['situation']['balls'], situation['situation']['strikes']
    stats = [cell('FF', 24, pitcher=pitcher, stand=side, balls=balls, strikes=strikes, day='2026-09-01', hand=h)
             for h in ('L', 'R')]+[cell('SL', 16, pitcher=pitcher, stand=side, balls=balls, strikes=strikes, day='2026-09-01', hand=h)
                                  for h in ('L', 'R')]
    directory = write_profile(tmp_path/'profiles'/f'target-{cutoff}', cutoff=cutoff, pitchers={pitcher: {'FF': 600, 'SL': 400}},
                              stats=stats)
    latest = pointer(tmp_path/'profiles'/'LATEST.json', directory, as_of_cutoff=cutoff)
    world = real_world(worlds, monkeypatch, [latest])
    world.cycle()
    body = world.game()
    assert ('reference' in body['next']) is card
    if card:
        got = body['next']['reference']
        assert got['as_of'] == cutoff < body['game']['official_date'] and got['stale'] is stale
        assert publish_schema.validate(body, GAME_SCHEMA) == []
        assert audit_all(world).violations == []
        assert [i['pitch_type'] for i in got['items']] == ['FF', 'SL'] and not keys(got) & FORBIDDEN


@pinned
def test_the_pointer_is_reread_and_moves_the_card_to_the_newer_profile(worlds, monkeypatch, tmp_path):
    from test_profile_reference import cell, pointer, write_profile
    probe = World(tmp_path/'probe', profile_reference={'enabled': False, 'profile_sources': None})
    try:
        probe.policy.status = 'UNSUPPORTED_UNKNOWN_PITCHER'
        pitcher, side = _pitcher(probe)
        count = probe.game()['current']['situation']
    finally:
        probe.close()
    stats = [cell('FF', 30, pitcher=pitcher, stand=side, balls=count['balls'], strikes=count['strikes'], hand=h) for h in ('L', 'R')]
    old = write_profile(tmp_path/'p'/'old', cutoff='2026-09-30', pitchers={pitcher: {'FF': 600}}, stats=stats)
    new = write_profile(tmp_path/'p'/'new', cutoff='2026-10-04', pitchers={pitcher: {'FF': 600}}, stats=stats)
    latest = pointer(tmp_path/'p'/'LATEST.json', old, as_of_cutoff='2026-09-30')
    world = real_world(worlds, monkeypatch, [latest])
    world.app.state.service_extras.reference(world.app.state.features()['profile_reference'])._recheck = 0  # the 30 s recheck, now
    world.cycle()
    first = world.game()['next']['reference']
    pointer(latest, new, as_of_cutoff='2026-10-04')
    world.cycle(after=40.0)  # the same pending pitch, one cycle later
    second = world.game()['next'].get('reference')
    assert (first['as_of'], first['stale']) == ('2026-09-30', True)
    assert second is not None and (second['as_of'], second['stale']) == ('2026-10-04', False)


def test_replay_facts_count_the_card_and_the_reference_free_digest_ignores_only_the_excluded_fields():
    example = json.loads((CONTRACT_DIR/'examples/game-live-unsupported-reference.json').read_text())
    facts = publish_audit.game_facts(example)
    assert (facts['reference'], facts['reference_as_of'], facts['reference_stale']) == ('published', '2026-10-04', False)
    off = copy.deepcopy(example)
    off['next'].pop('reference')
    off['features']['profile_reference']['enabled'] = False
    off['publish'] = {'contract': 'v1', 'sequence': 999, 'published_at': '2030-01-01T00:00:00Z'}
    assert publish_audit.game_facts(off)['reference'] == 'none'
    assert publish_audit.reference_free_digest(off) == facts['reference_free_digest']
    off['next']['pitch_key'] = 'other'
    assert publish_audit.reference_free_digest(off) != facts['reference_free_digest']
    assert publish_audit.REFERENCE_FREE_EXCLUDED == ('/publish', '/next/reference', '/features/profile_reference/enabled')
