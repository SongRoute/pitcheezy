"""Generated field comparisons only; no actual I3 or capture payload."""
import copy
import gzip
import hashlib
import json
from types import SimpleNamespace

import pandas as pd

from scripts import fullseason_capture_diagnostic_v2 as v2
from test_captured_bridge import feed


def record(body):
    key = (v2.v1.GAME, 1, 1)
    return key, {'base': v2.captured_diagnostic.inspect_body(body, game_pk=key[0], game_date=v2.v1.DATE)[0],
                 'before': v2.captured_beforestate.prefix_candidates(body, game_pk=key[0], game_date=v2.v1.DATE)[0],
                 'identity_bip': v2.captured_identity_bip.inspect_hypotheses(body, game_pk=key[0], game_date=v2.v1.DATE)[0]}


def source_row(**kwargs):
    row = {'game_pk': v2.v1.GAME, 'at_bat_number': 1, 'pitch_number': 1,
           'game_date': v2.v1.DATE, 'game_type': 'L', 'pitcher': 10, 'batter': 20,
           'pitch_type': 'FF', 'description': 'called_strike', 'events': None,
           'bb_type': None, 'balls': 0, 'strikes': 0, 'outs_when_up': 0,
           'home_score': 0, 'away_score': 0, 'stand': 'L', 'p_throws': 'R'}
    row.update(kwargs)
    return row


def test_first_to_final_categories_and_unpaired_identity():
    initial = feed(); initial['gameData']['game']['pk'] = v2.v1.GAME
    initial['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'SL'}
    final = copy.deepcopy(initial)
    event = final['liveData']['plays']['currentPlay']['playEvents'][0]
    event['details']['type']['code'] = 'FF'
    event['details']['call']['code'] = 'C'
    key, old = record(initial)
    _, new = record(final)
    refs = v2.v1.reference(pd.DataFrame([source_row()]))
    change = v2.transitions({key: old}, {key: new}, refs)
    assert change['paired_both_views'] == 1
    assert change['event_type_code_raw_changed'] == 1
    assert change['event_type_code_reference_mismatch_to_exact'] == 1
    assert change['event_call_code_raw_changed'] == 1
    assert change['description_hypothesis_reference_mismatch_to_exact'] == 1
    assert change['outcome10_hypothesis_raw_changed'] == 1
    paired = v2.paired({key: new}, refs)
    base_paired = v2.v1.paired({key: new['base']}, refs)
    assert paired['paired_key'] == base_paired['paired_key'] == 1
    assert paired['pitcher_exact'] == 1
    assert paired['pitcher_eligibility_hypothesis_unpaired'] == 1
    assert paired['before_outs_unavailable'] == 1
    assert paired['accepted_false'] == 1


def test_null_reference_transition_is_unavailable():
    body = feed(); body['gameData']['game']['pk'] = v2.v1.GAME
    body['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'FF'}
    key, value = record(body)
    reference = SimpleNamespace(pitch_type=pd.NA, description=float('nan'), source_outcome10=None)
    change = v2.transitions({key: value}, {key: value}, {key: reference})
    assert change['event_type_code_reference_reference_unavailable_to_reference_unavailable'] == 1
    assert change['description_hypothesis_reference_reference_unavailable_to_reference_unavailable'] == 1
    assert change['outcome10_hypothesis_reference_reference_unavailable_to_reference_unavailable'] == 1


def test_bip_and_reference_only_are_descriptive():
    body = feed(); body['gameData']['game']['pk'] = v2.v1.GAME
    play = body['liveData']['plays']['currentPlay']
    play['about']['isComplete'] = True
    play['result']['eventType'] = 'single'
    event = play['playEvents'][0]
    event['details'].update(call={'code': 'X'}, isInPlay=True)
    event['hitData'] = {'trajectory': 'ground_ball'}
    key, value = record(body)
    refs = v2.v1.reference(pd.DataFrame([source_row(description='hit_into_play',
        events='single', bb_type='ground_ball')]))
    counts = v2.paired({key: value}, refs)
    for field in ('bip_description', 'bip_events', 'bip_outcome10', 'bb_type'):
        assert counts[field + '_exact'] == 1
        assert counts[field + '_eligibility_hypothesis_unpaired'] == 1
    assert counts['accepted_false'] == 1
    only = v2.reference_only({}, refs)
    assert only['capture_absent_statcast_row'] == 1
    assert only['events_present'] == 1


def test_collect_first_receipt_final_body_and_provider_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(v2.v1.replay, 'MAX_FILES', 2)
    source = feed(); source['gameData']['game']['pk'] = v2.v1.GAME
    later = copy.deepcopy(source)
    later['liveData']['plays']['currentPlay']['playEvents'][0]['details']['call']['code'] = 'C'
    capture = tmp_path / 'capture'; capture.mkdir()
    files = []
    for seq, body in enumerate((source, later), 1):
        payload = json.dumps(body).encode()
        name = f'849809_{seq:06d}.json.gz'
        (capture / name).write_bytes(gzip.compress(payload))
        files.append({'file': name, 'seq': seq, 'version': 1,
                      'sha256': hashlib.sha256(payload).hexdigest(), 'fetched': float(seq)})
    manifest = capture / 'MANIFEST.json'
    manifest.write_text(json.dumps({'game_pk': v2.v1.GAME, 'status': 'OK', 'files': files}))
    allow = tmp_path / 'allow.json'
    allow.write_text(json.dumps({'manifest': {'path': str(manifest), 'sha256': v2.sha(manifest)},
                                 'root': str(capture), 'files': files}))
    reg = {'capture_root': str(capture), 'manifest': {'path': str(manifest), 'sha256': v2.sha(manifest)},
           'allowlist': {'path': str(allow), 'sha256': v2.sha(allow)},
           'capture_compressed_bytes_reported': sum((capture / f['file']).stat().st_size for f in files),
           'capture_inflated_bytes_reported': sum(len(json.dumps(b).encode()) for b in (source, later))}
    first, final, counts, _, _ = v2.collect(reg)
    key = (v2.v1.GAME, 1, 1)
    assert counts['bodies'] == 2
    assert first[key]['base']['description_hypothesis'] == 'ball'
    assert final[key]['base']['description_hypothesis'] == 'called_strike'
    later['liveData']['plays']['currentPlay']['playEvents'][0]['playId'] = 'different'
    payload = json.dumps(later).encode(); path = capture / files[1]['file']
    path.write_bytes(gzip.compress(payload))
    files[1]['sha256'] = hashlib.sha256(payload).hexdigest()
    manifest.write_text(json.dumps({'game_pk': v2.v1.GAME, 'status': 'OK', 'files': files}))
    allow.write_text(json.dumps({'manifest': {'path': str(manifest), 'sha256': v2.sha(manifest)},
                                 'root': str(capture), 'files': files}))
    reg['manifest']['sha256'] = v2.sha(manifest)
    reg['allowlist']['sha256'] = v2.sha(allow)
    reg['capture_compressed_bytes_reported'] = sum((capture / f['file']).stat().st_size for f in files)
    reg['capture_inflated_bytes_reported'] = sum(len(json.dumps(b).encode()) for b in (source, later))
    first, final, counts, _, _ = v2.collect(reg)
    assert counts['provider_conflict_keys_excluded'] == 1
    assert first == final == {}
