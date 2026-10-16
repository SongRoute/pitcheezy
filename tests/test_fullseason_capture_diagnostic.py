"""Generated paired field checks; no captured body or I3 payload read."""
import copy
import gzip
import hashlib
import json

import pandas as pd
import pytest

from scripts import fullseason_capture_diagnostic as diagnostic
from test_captured_bridge import feed


def test_exact_field_pairing_and_missing_reference_stops():
    body = feed()
    body['gameData']['game']['pk'] = diagnostic.GAME
    pitch = body['liveData']['plays']['currentPlay']['playEvents'][0]
    pitch['details']['type'] = {'code': 'FF'}
    row = diagnostic.captured_diagnostic.inspect_body(body, game_pk=diagnostic.GAME,
                                                        game_date=diagnostic.DATE)[0]
    ref = pd.DataFrame([{'game_pk': diagnostic.GAME, 'at_bat_number': 1, 'pitch_number': 1,
        'game_date': diagnostic.DATE, 'game_type': 'L', 'pitcher': 10, 'batter': 20,
        'pitch_type': 'FF', 'description': 'ball', 'events': None, 'bb_type': None,
        'balls': 0, 'strikes': 0, 'outs_when_up': 0, 'home_score': 0,
        'away_score': 0, 'stand': 'L', 'p_throws': 'R'}])
    refs = diagnostic.reference(ref)
    counts = diagnostic.paired({tuple(row['key']): row}, refs)
    assert counts['paired_key'] == 1
    for field in diagnostic.DIRECT_FIELDS:
        assert counts[field + '_exact'] == 1
    assert counts['pitch_level_identity_unavailable'] == 1
    assert counts['before_outs_score_unavailable'] == 1
    assert counts['bb_type_equivalence_unverified'] == 1
    assert counts['accepted_false'] == 1
    altered = ref.copy(); altered.loc[0, 'game_pk'] = diagnostic.GAME + 1
    with pytest.raises(ValueError, match='game/date absent'):
        diagnostic.reference(altered)


def test_first_seen_and_final_body_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.replay, 'MAX_FILES', 2)
    capture = tmp_path / 'capture'; capture.mkdir()
    source = feed(); source['gameData']['game']['pk'] = diagnostic.GAME
    source['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'FF'}
    later = copy.deepcopy(source)
    later['liveData']['plays']['currentPlay']['playEvents'][0]['details']['call']['code'] = 'C'
    bodies = [source, later]
    files = []
    for seq, value in enumerate(bodies, 1):
        body = json.dumps(value).encode(); name = f'849809_{seq:06d}.json.gz'
        (capture / name).write_bytes(gzip.compress(body))
        files.append({'file': name, 'seq': seq, 'version': 1,
                      'sha256': hashlib.sha256(body).hexdigest(), 'fetched': float(seq)})
    manifest = capture / 'MANIFEST.json'
    manifest.write_text(json.dumps({'game_pk': diagnostic.GAME, 'status': 'OK', 'files': files}))
    allow = tmp_path / 'allow.json'
    allow.write_text(json.dumps({'manifest': {'path': str(manifest), 'sha256': diagnostic.sha(manifest)},
                                 'root': str(capture), 'files': files}))
    reg = {'capture_root': str(capture), 'manifest': {'path': str(manifest), 'sha256': diagnostic.sha(manifest)},
           'allowlist': {'path': str(allow), 'sha256': diagnostic.sha(allow)},
           'capture_compressed_bytes_reported': sum((capture / f['file']).stat().st_size for f in files),
           'capture_inflated_bytes_reported': sum(len(json.dumps(b).encode()) for b in bodies)}
    first, final, counts, _, _ = diagnostic.receipts(reg)
    assert counts['bodies'] == 2 and counts['first_to_final_changed_keys'] == 1
    key = (diagnostic.GAME, 1, 1)
    assert first[key]['description_hypothesis'] == 'ball'
    assert final[key]['description_hypothesis'] == 'called_strike'
    assert all(not r['accepted'] for r in (*first.values(), *final.values()))
