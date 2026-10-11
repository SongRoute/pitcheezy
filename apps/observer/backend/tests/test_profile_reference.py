"""A2 reference card ("이 투수가 올해 이 카운트·타자 방향에서 자주 던진 공"): descriptive record statistics only.

Synthetic committed-profile fixtures for the unit logic; the real committed profile bytes are read only when the
pm-jobs paths exist (otherwise those tests skip). The pinned Step 5 source is executed in place by the module, so
every test here skips when that source is absent. No network, no model, no fit."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import profile_reference as PR

pytestmark = pytest.mark.skipif(not all(Path(PR.SOURCE[k]).is_file() for k in PR.SOURCE['pins']),
                                reason='pinned Step 5 source is not on this machine')

JOBS = Path('/Users/song/Projects/pitcheezy-pm-jobs')
OLD = JOBS/'20261010-ws-service-006/step5-2026-profile-reader-source-repair-001/actual-s2-profile-read-001'
LATEST = JOBS/'20261011-profile-refresh-001/profiles/LATEST.json'
REPLAY = JOBS/'20261011-rc-replay-849831/replay-849831'
real = pytest.mark.skipif(not ((OLD/'PROFILE.json').is_file() and LATEST.is_file()),
                          reason='committed profile bytes are not on this machine')
P, TAYLOR = 700001, 691799
FORBIDDEN = {'rank', 'ranking', 'target', 'zone', 'zone_id', 'zone_label', 'recommendation', 'recommended',
             'probability', 'prob', 'score', 'value', 'expected', 'prediction', 'x', 'z'}
ZERO = [0.]*10


def cell(kind, n, *, pitcher=P, hand='R', stand='L', balls=1, strikes=1, day='2026-09-20'):
    """One ``stats`` row: n pitches of ``kind`` (all put in outcome class 0; the card only uses the sum)."""
    return [kind, hand, stand, balls, strikes, pitcher, [float(n)]+ZERO[1:], date.fromisoformat(day).toordinal()]


def write_profile(directory, *, cutoff='2026-09-27', not_before='2026-03-25', pitchers=None, stats=None,
                  terminal='COMMITTED', verdict=True, verdict_changes=None, corrupt=False):
    """A committed attempt directory in the reader's own format (``dump_store`` layout)."""
    directory.mkdir(parents=True, exist_ok=True)
    pitchers = pitchers if pitchers is not None else {P: {'FF': 600, 'SL': 300, 'CH': 100}}
    stats = stats if stats is not None else [cell('FF', 30), cell('SL', 15), cell('CH', 5)]
    rows = sum(sum(types.values()) for types in pitchers.values())
    profile = {'schema': 'step5-profile-sufficient-v1',
               'provenance': {'source_id': 'synthetic test fixture', 'source_sha256': '0'*64, 'role': PR.ROLE,
                              'as_of_cutoff': cutoff, 'rows': rows, 'not_before': not_before},
               'pitcher': [[k, sum(t.values()), '2026-09-20', '2026-04-01', [[a, n] for a, n in t.items()]]
                           for k, t in pitchers.items()],
               'ptype': [], 'batter': [], 'physical': [], 'scale': [], 'stats': stats}
    blob = json.dumps(profile).encode()
    receipt = {'attempt_id': 'synthetic-'+directory.name, 'terminal': terminal, 'profile_committed': terminal == 'COMMITTED',
               'scope': 'ACTUAL_S2_ONE_PASS', 'counts': {'counts': {'rows_projected': rows}},
               'profile': {'file': 'PROFILE.json', 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}}
    receipt_raw = json.dumps(receipt).encode()
    (directory/'PROFILE.json').write_bytes(blob[:-1]+b' }' if corrupt else blob)
    (directory/'RECEIPT.json').write_bytes(receipt_raw)
    if verdict:
        (directory/'SUPERVISOR-RESULT.json').write_text(json.dumps({
            'success': True, 'profile_usable': True, 'status': 'COMPLETED', 'attempt_id': receipt['attempt_id'],
            'receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(), 'profile_sha256': receipt['profile']['sha256'],
            'profile_bytes': len(blob)} | (verdict_changes or {})))
    return directory


def pointer(path, directory, **changes):
    path.write_text(json.dumps({
        'schema': PR.POINTER_SCHEMA, 'profile_dir': str(directory),
        'profile_sha256': hashlib.sha256((directory/'PROFILE.json').read_bytes()).hexdigest()} | changes))
    return path


def ref(*sources, **kw):
    return PR.ProfileReference(list(sources), **kw)


def ask(reference, **kw):
    args = {'pitcher_id': P, 'batter_side': 'L', 'balls': 1, 'strikes': 1, 'official_date': '2026-10-10'} | kw
    return reference.explain(**args)


def keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*map(keys, value.values()))
    return set().union(*map(keys, value)) if isinstance(value, list) else set()


def test_card_is_a_labelled_reference_of_past_frequencies_without_rank_or_target(tmp_path):
    card, reason = ask(ref(write_profile(tmp_path/'a')))
    assert reason is None
    assert card['kind'] == 'profile_reference' and card['reason_code'] == 'unsupported/profile_reference'
    assert card['title'] == '이 투수가 올해 이 카운트·타자 방향에서 자주 던진 공' and card['chip'] == '참고 · 기록 통계'
    assert card['items'] == [
        {'pitch_type': 'FF', 'pitch_label': '포심', 'share': .6, 'share_pct': 60, 'count': 30},
        {'pitch_type': 'SL', 'pitch_label': '슬라이더', 'share': .3, 'share_pct': 30, 'count': 15},
        {'pitch_type': 'CH', 'pitch_label': '체인지업', 'share': .1, 'share_pct': 10, 'count': 5}]
    assert card['other'] is None
    assert card['sample'] == {'pitches': 50, 'batter_side': 'L', 'scope': 'count_and_batter_side', 'balls': 1, 'strikes': 1}
    assert card['as_of'] == '2026-09-27'
    assert '9월 27일까지' in card['note'] and '50개' in card['note'] and '추천이나 다음 공 예측이 아닙니다' in card['note']
    assert card['provenance']['role'] == 'SERVICE_PROFILE_STRICTLY_PRIOR' and card['provenance']['cutoff'] == '2026-09-27'
    assert len(card['provenance']['profile_sha256_prefix']) == 12
    assert card['provenance']['thresholds'] == {
        'fraction': .01, 'min_count': 5, 'min_total': 50, 'min_known_coverage': .9, 'max_age_days': 365,
        'status': 'provisional_configuration', 'cell_min_pitches': 20,
        'cell_min_pitches_status': 'provisional_awaiting_pm_decision'}
    assert not keys(card) & FORBIDDEN
    assert '추천' not in card['title']+card['chip'] and '예측' not in card['title']+card['chip']
    json.dumps(card)  # plain JSON types only


def test_cell_minimum_is_the_provisional_twenty_and_there_is_no_broader_fallback_by_default(tmp_path):
    elsewhere = [cell('FF', 200, balls=0, strikes=0), cell('FF', 200, stand='R')]
    below = ref(write_profile(tmp_path/'a', stats=[cell('FF', 12), cell('SL', 7)]+elsewhere))
    assert ask(below) == (None, 'cell_below_provisional_minimum')
    at = ref(write_profile(tmp_path/'b', stats=[cell('FF', 12), cell('SL', 8)]+elsewhere))
    card, _ = ask(at)
    assert card['sample']['pitches'] == 20 and [i['count'] for i in card['items']] == [12, 8]
    assert PR.SAMPLE_RULE == {'cell_min_pitches': 20, 'status': 'provisional_awaiting_pm_decision'}
    # optional broader card: its own honest title, never the count title
    broad, _ = ask(ref(tmp_path/'a', broad_fallback=True))
    assert broad['title'] == PR.BROAD_TITLE != PR.TITLE and '모든 카운트' in broad['title']
    assert broad['sample'] == {'pitches': 219, 'batter_side': 'L', 'scope': 'batter_side'}


@pytest.mark.parametrize('types, expected', [
    ({'FF': 49}, 'unsupported_player:insufficient_history'),
    ({'FF': 50}, None),
    ({'FF': 89, 'XX': 11}, 'unsupported_player:low_known_type_coverage'),
    ({'FF': 90, 'XX': 10}, None)])
def test_pitcher_minimum_is_the_adapters_general_support(tmp_path, types, expected):
    reference = ref(write_profile(tmp_path/'a', pitchers={P: types}, stats=[cell('FF', 40)]))
    card, reason = ask(reference)
    assert reason == expected and (card is None) == (expected is not None)
    assert ask(reference, pitcher_id=P+1) == (None, 'unsupported_player:no_prior_pitcher_history')


def test_a_pitch_type_below_the_adapters_per_action_thresholds_is_never_listed(tmp_path):
    types = {'FF': 500, 'SL': 5, 'CH': 4, 'XX': 30}  # CH: below min_count; SL 5/539 < 1 %: below fraction
    stats = [cell('FF', 20), cell('SL', 5), cell('CH', 4), cell('XX', 11)]
    card, _ = ask(ref(write_profile(tmp_path/'a', pitchers={P: types}, stats=stats)))
    assert [i['pitch_type'] for i in card['items']] == ['FF']
    # shares stay over every counted pitch of the cell; what is not listed is reported as the rest
    assert card['items'][0] == {'pitch_type': 'FF', 'pitch_label': '포심', 'share': .5, 'share_pct': 50, 'count': 20}
    assert card['other'] == {'share': .5, 'count': 20, 'share_pct': 50} and card['sample']['pitches'] == 40
    only_unlisted = ref(write_profile(tmp_path/'b', pitchers={P: types}, stats=[cell('XX', 30)]))
    assert ask(only_unlisted) == (None, 'no_listed_pitch_type_in_cell')
    assert ask(ref(write_profile(tmp_path/'c', pitchers={P: {'FF': 4, 'XX': 2, 'KN': 60}}, stats=[cell('KN', 30)])))[0][
        'items'][0]['pitch_label'] == '너클볼'


def test_items_are_most_frequent_first_at_most_four_and_tiny_shares_are_dropped(tmp_path):
    types = {'FF': 300, 'SI': 300, 'SL': 300, 'CU': 300, 'CH': 300, 'FC': 300}
    stats = [cell('CH', 10), cell('FF', 40), cell('FC', 4), cell('SL', 20), cell('SI', 30), cell('CU', 16)]
    card, _ = ask(ref(write_profile(tmp_path/'a', pitchers={P: types}, stats=stats)))
    assert [(i['pitch_type'], i['count']) for i in card['items']] == [('FF', 40), ('SI', 30), ('SL', 20), ('CU', 16)]
    counts = [i['count'] for i in card['items']]
    assert counts == sorted(counts, reverse=True)
    assert card['other'] == {'share': round(14/120, 4), 'count': 14, 'share_pct': 100-sum(i['share_pct'] for i in card['items'])}
    card, _ = ask(ref(tmp_path/'a', display={'max_items': 6}))
    assert [i['pitch_type'] for i in card['items']] == ['FF', 'SI', 'SL', 'CU', 'CH']  # FC 4/120 < 5 %


def test_percentages_are_floor_or_ceiling_equal_for_ties_and_sum_to_100_or_just_under():
    assert PR._percentages([1, 1, 1]) == [33, 33, 33] and PR._percentages([2, 1]) == [67, 33]
    assert PR._percentages([49, 49, 42, 33]) == [28, 28, 24, 19] and PR._percentages([30, 15, 5]) == [60, 30, 10]
    rng = random.Random(0)
    for _ in range(3000):
        counts = sorted((rng.randint(1, 200) for _ in range(rng.randint(1, 7))), reverse=True)
        total, got = sum(counts), PR._percentages(counts)
        assert all(n*100//total <= p <= -(-n*100//total) for n, p in zip(counts, got))
        assert all(p == q for (n, p) in zip(counts, got) for (m, q) in zip(counts, got) if n == m)
        assert 100-len(counts) < sum(got) <= 100
        if len(set(counts)) == len(counts):
            assert sum(got) == 100


def test_strictly_prior_boundary_and_newest_usable_profile(tmp_path):
    early = write_profile(tmp_path/'early', cutoff='2026-09-27', stats=[cell('FF', 20)])
    late = write_profile(tmp_path/'late', cutoff='2026-10-10', stats=[cell('FF', 21)])
    both = ref(early, late)
    assert ask(both, official_date='2026-09-27') == (None, 'no_usable_profile:profile_as_of_not_strictly_before_game_date')
    assert ask(both, official_date='2026-09-28')[0]['as_of'] == '2026-09-27'
    same_day, _ = ask(both, official_date='2026-10-10')  # cutoff 10-10 is not before officialDate 10-10
    assert same_day['as_of'] == '2026-09-27' and same_day['sample']['pitches'] == 20
    next_day, _ = ask(both, official_date='2026-10-11')
    assert next_day['as_of'] == '2026-10-10' and next_day['sample']['pitches'] == 21
    assert ask(ref(late, early), official_date='2026-10-11')[0]['as_of'] == '2026-10-10'  # order of sources is free
    assert ask(ref(late), official_date='2026-10-10') == (None, 'no_usable_profile:profile_as_of_not_strictly_before_game_date')
    # "올해": a profile of another season is never shown under this title
    assert ask(both, official_date='2027-04-01') == (None, 'no_usable_profile:profile_window_is_not_this_season')
    assert ask(ref()) == (None, 'no_usable_profile:no_source_configured')


def test_latest_pointer_is_followed_and_must_name_the_loaded_bytes(tmp_path):
    first = write_profile(tmp_path/'first', cutoff='2026-10-08', stats=[cell('FF', 20)])
    second = write_profile(tmp_path/'second', cutoff='2026-10-09', stats=[cell('FF', 25)])
    latest = pointer(tmp_path/'LATEST.json', first, as_of_cutoff='2026-10-08')
    now = [0.]
    reference = ref(latest, recheck_seconds=30, clock=lambda: now[0])
    assert ask(reference)[0]['as_of'] == '2026-10-08'
    pointer(latest, second)
    assert ask(reference)[0]['as_of'] == '2026-10-08'  # files are not looked at again inside the recheck window
    now[0] = 31.
    assert ask(reference)[0]['as_of'] == '2026-10-09'
    health = reference.health()
    assert health['profile_loads'] == 2 and len(reference._cache) == 1  # the profile no longer named was dropped
    assert health['sources'][0]['profile_dir'] == str(second.resolve()) and health['sources'][0]['usable']
    for bad in ({'profile_sha256': '0'*64}, {'as_of_cutoff': '2026-10-01'}):
        other = ref(pointer(tmp_path/'OTHER.json', second, **bad))
        assert ask(other) == (None, 'no_usable_profile:pointer_names_other_profile_bytes_or_cutoff')


def test_a_missing_corrupt_or_uncommitted_source_gives_no_card_and_an_operator_reason(tmp_path):
    cases = {
        'stopped': write_profile(tmp_path/'stopped', terminal='STOP_ROW_INVALID'),
        'corrupt': write_profile(tmp_path/'corrupt', corrupt=True),
        'no_verdict': write_profile(tmp_path/'no_verdict', verdict=False),
        'failed_verdict': write_profile(tmp_path/'failed', verdict_changes={'success': False}),
        'unusable_verdict': write_profile(tmp_path/'unusable', verdict_changes={'profile_usable': False}),
        'other_bytes_verdict': write_profile(tmp_path/'other', verdict_changes={'profile_sha256': '1'*64}),
        'missing': tmp_path/'nowhere',
        'empty_dir': (tmp_path/'empty').mkdir() or tmp_path/'empty'}
    (tmp_path/'garbage.json').write_text('{not json')
    (tmp_path/'list.json').write_text('[1, 2]')
    cases |= {'garbage_pointer': tmp_path/'garbage.json', 'not_a_pointer': tmp_path/'list.json',
              'pointer_to_nowhere': pointer(tmp_path/'p.json', cases['stopped'], profile_dir=str(tmp_path/'gone'))}
    (cases['stopped']/'broken').mkdir()
    for name in ('PROFILE.json', 'RECEIPT.json', 'SUPERVISOR-RESULT.json'):
        (cases['stopped']/'broken'/name).write_text('{')
    cases['broken_json'] = cases['stopped']/'broken'
    for name, source in cases.items():
        reference = ref(source)
        card, reason = ask(reference)
        assert card is None and reason.startswith('no_usable_profile:'), name
        assert reference.card(P, 'L', 1, 1, '2026-10-10') is None
        health = reference.health()
        assert health['sources'][0]['usable'] is False and health['sources'][0]['reason'], name
        assert health['last_outcome'] == reason and health['outcomes'][reason] == 2
    # a bad source next to a good one does not hide the good one
    good = write_profile(tmp_path/'good')
    assert ask(ref(cases['corrupt'], cases['missing'], good))[0]['sample']['pitches'] == 50
    assert 'profile bytes are not the ones a COMMITTED receipt names' in ask(ref(cases['corrupt']))[1]
    assert 'supervisor' in ask(ref(cases['no_verdict']))[1] and 'supervisor' in ask(ref(cases['failed_verdict']))[1]


@pytest.mark.parametrize('change', [
    {'pitcher_id': True}, {'pitcher_id': str(P)}, {'pitcher_id': float(P)}, {'pitcher_id': None}, {'pitcher_id': 0},
    {'pitcher_id': 10**7+1}, {'balls': 4}, {'balls': -1}, {'balls': True}, {'balls': 1.0}, {'balls': '1'},
    {'strikes': 3}, {'strikes': None}, {'strikes': True}, {'batter_side': 'S'}, {'batter_side': 'l'},
    {'batter_side': None}, {'batter_side': 0}, {'official_date': None}, {'official_date': '2026-10-10T00:00:00Z'},
    {'official_date': '20261010'}, {'official_date': '2026-13-01'}, {'official_date': date(2026, 10, 10)},
    {'official_date': 20261010}, {'pitcher_hand': 'S'}, {'pitcher_hand': 1}, {'pitcher_hand': ''}])
def test_bad_inputs_give_no_card_and_never_raise(tmp_path, change):
    reference = ref(write_profile(tmp_path/'a'))
    card, reason = ask(reference, **change)
    assert card is None and reason == 'bad_input:'+next(iter(change))
    assert reference.health()['profile_loads'] == 0  # refused before any profile is touched


def test_pitcher_hand_restricts_the_cell_and_absent_hand_counts_both(tmp_path):
    stats = [cell('FF', 30, hand='R'), cell('SL', 10, hand='R'), cell('FF', 10, hand='L')]
    reference = ref(write_profile(tmp_path/'a', stats=stats))
    assert ask(reference)[0]['sample']['pitches'] == 50
    assert ask(reference, pitcher_hand='R')[0]['sample']['pitches'] == 40
    assert ask(reference, pitcher_hand='L') == (None, 'cell_below_provisional_minimum')
    assert ask(reference, batter_side='R') == (None, 'cell_below_provisional_minimum')
    assert ask(reference, balls=0) == (None, 'cell_below_provisional_minimum')


def test_a_differing_pin_fails_closed_and_importing_leaves_no_trace(tmp_path):
    good = write_profile(tmp_path/'a')
    wrong = dict(PR.SOURCE) | {'pins': dict(PR.SOURCE['pins']) | {'live_adapter': '0'*64}}
    reference = ref(good, source=wrong)
    card, reason = ask(reference)
    assert card is None and 'pinned_source_unusable' in reason and 'live_adapter' in reason
    assert reference.health()['pinned_source_error'] and reference.health()['profile_loads'] == 0
    gone = dict(PR.SOURCE) | {'features': str(tmp_path/'absent.py')}
    assert 'pinned_source_unusable' in ask(ref(good, source=gone))[1]
    assert ask(ref(good))[0] is not None
    assert 'profile_reader' not in sys.modules and 'live_adapter' not in sys.modules and 'step5' not in sys.modules
    for name in PR.SOURCE['pins']:
        assert not (Path(PR.SOURCE[name]).parent/'__pycache__').exists()
        assert hashlib.sha256(Path(PR.SOURCE[name]).read_bytes()).hexdigest() == PR.SOURCE['pins'][name]


def test_profile_is_loaded_once_reloaded_when_its_files_change_and_sources_are_capped(tmp_path):
    directory = write_profile(tmp_path/'a', stats=[cell('FF', 20)])
    now = [0.]
    reference = ref(directory, recheck_seconds=30, clock=lambda: now[0])
    assert reference.health() == reference.health() | {'checked': False, 'profile_loads': 0}  # lazy
    for _ in range(50):
        assert ask(reference)[0]['sample']['pitches'] == 20
    assert reference.health()['profile_loads'] == 1
    now[0] = 100.
    assert ask(reference)[0] and reference.health()['profile_loads'] == 1  # unchanged files: no reload
    write_profile(directory, stats=[cell('FF', 33)])
    now[0] = 200.
    assert ask(reference)[0]['sample']['pitches'] == 33 and reference.health()['profile_loads'] == 2
    many = ref(*[write_profile(tmp_path/f'm{i}') for i in range(PR.MAX_SOURCES+2)])
    health = many.warm()
    assert len(health['sources']) == PR.MAX_SOURCES == len(many._cache) and len(health['sources_ignored_over_limit']) == 2
    assert ref(str(directory)).warm()['sources'][0]['usable']  # a single path is accepted too


def test_concurrent_first_requests_load_once_and_agree(tmp_path):
    reference = ref(write_profile(tmp_path/'a'), write_profile(tmp_path/'b', cutoff='2026-10-01'))
    with ThreadPoolExecutor(16) as pool:
        got = list(pool.map(lambda i: reference.explain(P, 'L', 1, 1, '2026-10-10' if i % 2 else '2026-09-30'),
                            range(400)))
    assert {reason for _, reason in got} == {None}
    assert {json.dumps(card, sort_keys=True) for card, _ in got[::2]} == {json.dumps(got[0][0], sort_keys=True)}
    assert {card['as_of'] for card, _ in got[1::2]} == {'2026-10-01'} and got[0][0]['as_of'] == '2026-09-27'
    health = reference.health()
    assert health['profile_loads'] == 2 and health['outcomes'] == {'card': 400}


# ---- the real committed profile bytes (read-only; skipped when absent) -----------------------------------------------

@pytest.fixture(scope='module')
def real_reference():
    reference = PR.ProfileReference([OLD, LATEST])
    health = reference.warm()
    assert [s['usable'] for s in health['sources']] == [True, True], health
    return reference


@real
def test_real_profiles_connect_and_the_same_day_cutoff_is_refused(real_reference):
    cutoffs = [s['cutoff'] for s in real_reference.health()['sources']]
    assert cutoffs == ['2026-09-27', '2026-10-10']
    assert real_reference.health()['sources'][1]['profile_sha256'] == json.loads(LATEST.read_text())['profile_sha256']
    assert real_reference.explain(TAYLOR, 'R', 0, 0, '2026-10-10')[0]['as_of'] == '2026-09-27'
    assert real_reference.explain(TAYLOR, 'R', 0, 0, '2026-10-11')[0]['as_of'] == '2026-10-10'
    assert PR.ProfileReference([LATEST]).explain(TAYLOR, 'R', 0, 0, '2026-10-10') == (
        None, 'no_usable_profile:profile_as_of_not_strictly_before_game_date')
    assert real_reference.explain(TAYLOR, 'R', 0, 0, '2026-09-27')[1].startswith('no_usable_profile:')


@real
@pytest.mark.parametrize('source, official', [(OLD, '2026-10-10'), (LATEST, '2026-10-11')])
def test_every_real_pitcher_agrees_with_the_adapters_support_and_the_raw_counts(source, official):
    """All pitchers of the profile x all 24 cells: the card exists only where the adapter's ``support`` passes, lists
    only its provisionally eligible actions, and its counts equal a recount from the raw PROFILE.json rows."""
    reference = PR.ProfileReference([source])
    reference.warm()
    entry = reference._active[0][1]
    profile = entry['profile']
    raw = json.loads((Path(entry['directory'])/'PROFILE.json').read_bytes())
    by_cell = {}
    for kind, _hand, stand, balls, strikes, pitcher, counts, _last in raw['stats']:
        key = (pitcher, stand, balls, strikes)
        by_cell.setdefault(key, {})[kind] = by_cell.get(key, {}).get(kind, 0)+int(sum(counts))
    seen = {'general': set(), 'cards': 0, 'below_cell': 0, 'not_listed': 0, 'boundary_total': set(), 'boundary_count': set()}
    for pitcher, total, _newest, _oldest, types in raw['pitcher']:
        support = profile.support(pitcher, official)
        eligible = {c['pitch_type'] for c in support['candidates'] if c['status'] == 'provisionally_eligible'}
        assert support['total'] == total
        if total in (49, 50, 51):
            seen['boundary_total'].add(total)
        seen['boundary_count'] |= {n for _, n in types if n in (4, 5)}
        for side in 'LR':
            for balls in range(4):
                for strikes in range(3):
                    card, reason = reference.explain(pitcher, side, balls, strikes, official)
                    counts = by_cell.get((pitcher, side, balls, strikes), {})
                    n = sum(counts.values())
                    if support['general_reason']:
                        assert (card, reason) == (None, 'unsupported_player:'+support['general_reason'])
                        seen['general'].add(support['general_reason'])
                        continue
                    if not eligible:
                        assert (card, reason) == (None, 'no_action_meets_support_thresholds')
                        continue
                    if n < 20:
                        assert (card, reason) == (None, 'cell_below_provisional_minimum')
                        seen['below_cell'] += 1
                        continue
                    listed = [k for k in sorted(counts, key=lambda k: (-counts[k], k))
                              if k in eligible and counts[k]/n >= .05][:4]
                    if not listed:
                        assert (card, reason) == (None, 'no_listed_pitch_type_in_cell')
                        continue
                    seen['cards'] += 1
                    assert reason is None and card['sample']['pitches'] == n
                    assert [(i['pitch_type'], i['count']) for i in card['items']] == [(k, counts[k]) for k in listed]
                    assert all(support['candidates'][[c['pitch_type'] for c in support['candidates']].index(
                        i['pitch_type'])]['prior_pitch_count'] >= 5 for i in card['items'])
                    shown = sum(i['share_pct'] for i in card['items'])+(card['other'] or {'share_pct': 0})['share_pct']
                    assert 100-len(counts) < shown <= 100
                    seen['not_listed'] += len(set(counts)-set(listed)) > 0
                    assert not keys(card) & FORBIDDEN
    assert seen['cards'] > 1000 and seen['below_cell'] > 1000 and seen['not_listed'] > 0
    assert 'insufficient_history' in seen['general']
    assert {4, 5} <= seen['boundary_count']  # the per-action count boundary occurs in the real bytes
    below = [p for p, total, *_ in raw['pitcher'] if total < 50]
    assert below and all(reference.card(p, s, 0, 0, official) is None for p in below for s in 'LR')


@real
def test_grant_taylor_849831_pre_pitch_states(real_reference):
    """Game 849831, officialDate 2026-10-10. With the morning profile (cutoff 2026-09-27) the card exists for 12 of
    his 13 pre-pitch states; the refreshed profile (cutoff 2026-10-10) is not strictly prior for that date."""
    def line(side, balls, strikes, reference=real_reference, day='2026-10-10'):
        card = reference.card(TAYLOR, side, balls, strikes, day, pitcher_hand='R')
        return card and ([(i['pitch_type'], i['share_pct'], i['count']) for i in card['items']], card['sample']['pitches'])
    assert line('R', 0, 0) == ([('SI', 29, 47), ('FF', 28, 44), ('SL', 24, 38), ('CU', 19, 31)], 160)
    assert line('L', 0, 0) == ([('CU', 47, 83), ('FF', 45, 79)], 176)
    assert line('L', 2, 0) == ([('FF', 64, 16), ('CU', 28, 7)], 25)
    assert line('L', 3, 0) is None and real_reference.explain(TAYLOR, 'L', 3, 0, '2026-10-10')[1] == \
        'cell_below_provisional_minimum'
    refreshed = PR.ProfileReference([LATEST])
    assert all(line(s, b, k, refreshed) is None for s in 'LR' for b in range(4) for k in range(3))
    assert line('R', 0, 0, refreshed, '2026-10-11')[1] == 173


@real
@pytest.mark.skipif(not REPLAY.is_dir(), reason='replay bodies are not on this machine')
def test_grant_taylor_states_from_the_replay_bodies(real_reference):
    import gzip
    final = json.loads(gzip.open(sorted(REPLAY.glob('*.json.gz'))[-1]).read())
    assert final['gameData']['datetime']['officialDate'] == '2026-10-10'
    assert final['gameData']['players'][f'ID{TAYLOR}']['fullName'] == 'Grant Taylor'
    states = []
    for play in final['liveData']['plays']['allPlays']:
        if play['matchup']['pitcher']['id'] != TAYLOR:
            continue
        balls = strikes = 0
        for event in play['playEvents']:
            if event.get('isPitch'):
                states.append((play['matchup']['batSide']['code'], balls, strikes))
                balls, strikes = event['count']['balls'], event['count']['strikes']
    assert len(states) == 13
    cards = [real_reference.card(TAYLOR, side, b, s, '2026-10-10', pitcher_hand='R') for side, b, s in states]
    assert sum(c is not None for c in cards) == 12 and states[cards.index(None)] == ('L', 3, 0)
    assert {c['as_of'] for c in cards if c} == {'2026-09-27'}
