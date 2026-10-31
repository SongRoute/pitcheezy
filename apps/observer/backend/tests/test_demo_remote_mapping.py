"""Remote-observed video provenance for demo mappings (build_demo_mapping --video-url) on SYNTHETIC games only.

Every video id, reviewer and evidence string below is a labelled test fixture, not an observation of any real
video. No network, no T7, no real mapping directory.
"""
import csv
import gzip
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from observer_app.demo_clock import DemoClock, validate_mapping
from test_service_demo_api import GAME, _mapping_script, env, feed, start_demo  # noqa: F401  (env is a fixture)

FAKE_ID = 'SYNTHfixt01'  # 11 characters, shaped like a YouTube id; no such video is implied
FAKE_URL = f'https://www.youtube.com/watch?v={FAKE_ID}'
REMOTE = ['--video-url', f'https://youtu.be/{FAKE_ID}', '--observed-duration-s', '120', '--observed-on', '2026-10-05',
          '--observed-channel', 'SYNTHETIC CHANNEL (test)', '--verified-by', 'synthetic-tester']


@pytest.fixture
def feeds(tmp_path):
    folder = tmp_path/'script-feeds'
    folder.mkdir()
    with gzip.open(folder/f'{GAME}_final.json.gz', 'wt') as handle:
        json.dump(feed(GAME, '2026-09-30T02:00:00Z'), handle)
    return folder


def run(tmp_path, feeds, *args):
    _mapping_script().main(['--game', str(GAME), '--feed-dir', str(feeds), '--demo-dir', str(tmp_path/'demo'), *args])


def template(tmp_path, feeds, name='t.csv'):
    out = tmp_path/name
    run(tmp_path, feeds, '--id', 'unused', '--template', str(out), '--first-ab', '2', '--last-ab', '3')
    with open(out, newline='') as handle:
        return out, list(csv.DictReader(handle))


def write(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def filled(rows):
    return [r | {'video_offset_result_visible_s': str(30.0+7.5*i), 'status': 'OBSERVED_RESULT_VISIBLE',
                 'verified_by': 'synthetic-tester', 'evidence': f'SYNTHETIC FIXTURE row {i}: not a real observation'}
            for i, r in enumerate(rows)]


def test_template_is_canonical_blank_pending_and_never_overwritten(tmp_path, feeds):
    out, rows = template(tmp_path, feeds)
    assert [r['pitch_key'] for r in rows] == [f'{GAME}:2:1', f'{GAME}:2:2', f'{GAME}:3:1', f'{GAME}:3:2']
    assert [r['index'] for r in rows] == ['2', '3', '4', '5']
    assert all(r['video_offset_result_visible_s'] == '' and r['status'] == 'PENDING_NOT_OBSERVED' and not r['verified_by']
               and not r['evidence'] for r in rows)
    assert rows[0]['batter'] == 'B12' and rows[0]['pitcher'] == 'P21' and rows[0]['count_before'] == '0-0'
    assert rows[1]['feed_call'] == 'In play, no out' and rows[1]['pa_result']  # PA-ending pitch named for matching
    with pytest.raises(SystemExit):  # pending template cannot be imported as verified
        run(tmp_path, feeds, '--id', 'pending', '--from-csv', str(out), *REMOTE)
    with pytest.raises(FileExistsError):
        template(tmp_path, feeds)
    assert not (tmp_path/'demo'/'mappings').exists()


def test_valid_synthetic_remote_mapping_loads_without_content_hash(tmp_path, feeds):
    _, rows = template(tmp_path, feeds)
    sheet = write(tmp_path/'filled.csv', filled(rows))
    evidence = tmp_path/'notes.txt'
    evidence.write_text('SYNTHETIC evidence notes for a test fixture\n')
    run(tmp_path, feeds, '--id', 'remote-fx', '--from-csv', str(sheet), *REMOTE, '--evidence-file', str(evidence))
    made = json.loads((tmp_path/'demo'/'mappings'/'remote-fx.json').read_text())
    assert made['kind'] == 'video_verified' and made['verified'] is True
    video, check = made['video'], made['verification']
    assert video['source'] == 'remote_observed' and video['sha256'] is None and video['url'] == FAKE_URL
    assert video['video_id'] == FAKE_ID and video['observed_duration_s'] == 120.0
    assert check['csv_sha256'] == hashlib.sha256(sheet.read_bytes()).hexdigest()
    assert check['evidence_file_sha256'] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert check['by'] == 'synthetic-tester' and check['observed_on'] == '2026-10-05' and check['rows_reviewed'] == 4
    url_hashes = {hashlib.sha256(u.encode()).hexdigest() for u in (FAKE_URL, f'https://youtu.be/{FAKE_ID}', FAKE_ID)}
    assert not url_hashes & {v for v in json.dumps(made).replace('"', ' ').split() if len(v) == 64}  # never a URL "hash"
    assert [r['reveal_at_s'] for r in made['rows']] == [30.0, 37.5, 45.0, 52.5]
    assert all(r['review']['evidence'].startswith('SYNTHETIC') for r in made['rows'])

    clock = DemoClock(tmp_path/'demo', wall=lambda: 1_800_000_000.0)
    mapping = clock.mapping('remote-fx')
    keys = [f'{GAME}:{ab}:{n}' for ab in range(1, 6) for n in (1, 2)]
    session = clock.create(GAME, mapping, keys, start_index=2)
    assert session['mapping']['video_sha256'] is None and session['mapping']['verified'] is True
    assert session['mapping']['video_source'] == 'remote_observed' and session['mapping']['video_url'] == FAKE_URL
    assert all(set(r) == {'index', 'pitch_key', 'reveal_at_s'} for r in session['rows'])  # reviews stay in the mapping file
    public = clock.public(session, keys)
    assert public['cutoff']['index'] == 1 and 'mapping' not in public and FAKE_ID not in json.dumps(public)
    assert clock.operator_view(session, keys)['mapping']['video_source'] == 'remote_observed'


def test_viewer_api_unchanged_and_embargoed_with_remote_mapping(env, tmp_path, feeds):
    make, wall, demo = env
    _, rows = template(tmp_path, feeds)
    sheet = write(tmp_path/'filled.csv', filled(rows))
    run(tmp_path, feeds, '--id', 'remote-fx', '--from-csv', str(sheet), *REMOTE)
    (demo/'mappings'/'fx.json').write_text((tmp_path/'demo'/'mappings'/'remote-fx.json').read_text().replace('"remote-fx"', '"fx"'))
    with TestClient(make()) as client:
        session, head = start_demo(client, demo, start_index=2)
        sid = session['session_id']
        assert session['mapping']['video_source'] == 'remote_observed'  # operator response only
        body = client.get(f'/api/service/games/{GAME}', params={'demo': sid})
        text = body.text
        assert body.status_code == 200 and FAKE_ID not in text and 'remote_observed' not in text and 'synthetic-tester' not in text
        assert max(p['index'] for p in body.json()['pitches']) == 1 and f'{GAME}:2:' not in text  # nothing past the cutoff
        assert client.get(f'/api/service/games/{GAME}').status_code == 423
        catalog = client.get('/api/service/operator/catalog', headers=head).json()
        listed = next(m for m in (catalog.get('data') or catalog)['mappings'] if m['mapping_id'] == 'fx')
        assert listed['kind'] == 'video_verified' and listed['verified'] is True and listed['video_sha256'] is None


def _edit(rows, i, **changes):
    rows = [dict(r) for r in rows]
    rows[i] |= changes
    return rows


REJECT = {
    'blank_row': lambda rows: _edit(rows, 2, video_offset_result_visible_s='', status='PENDING_NOT_OBSERVED', verified_by='', evidence=''),
    'pending_status': lambda rows: _edit(rows, 1, status='PENDING_NOT_OBSERVED'),
    'no_evidence': lambda rows: _edit(rows, 0, evidence=' '),
    'no_reviewer': lambda rows: _edit(rows, 3, verified_by=''),
    'hyphen_key': lambda rows: _edit(rows, 0, pitch_key=f'{GAME}-AB2-P1'),
    'unknown_key': lambda rows: _edit(rows, 0, pitch_key=f'{GAME}:9:1'),
    'wrong_index': lambda rows: _edit(rows, 0, index='7'),
    'wrong_game': lambda rows: [r | {'game_pk': '1'} for r in rows],
    'other_video': lambda rows: [r | {'source_url': 'https://www.youtube.com/watch?v=otherVideo1'} for r in rows],
    'nan': lambda rows: _edit(rows, 1, video_offset_result_visible_s='nan'),
    'inf': lambda rows: _edit(rows, 1, video_offset_result_visible_s='inf'),
    'text': lambda rows: _edit(rows, 1, video_offset_result_visible_s='0:37'),
    'negative': lambda rows: _edit(rows, 0, video_offset_result_visible_s='-1'),
    'beyond_duration': lambda rows: _edit(rows, 3, video_offset_result_visible_s='120.5'),
    'out_of_order': lambda rows: _edit(rows, 2, video_offset_result_visible_s='31'),
    'same_time': lambda rows: _edit(rows, 1, video_offset_result_visible_s='30'),
    'gap': lambda rows: rows[:1]+rows[2:],
    'swapped_rows': lambda rows: [rows[1], rows[0], *rows[2:]],
}


@pytest.mark.parametrize('case', sorted(REJECT))
def test_remote_import_refuses_incomplete_or_wrong_rows(tmp_path, feeds, case):
    _, rows = template(tmp_path, feeds)
    sheet = write(tmp_path/'bad.csv', REJECT[case](filled(rows)))
    with pytest.raises(SystemExit):
        run(tmp_path, feeds, '--id', 'bad', '--from-csv', str(sheet), *REMOTE)
    assert not (tmp_path/'demo'/'mappings'/'bad.json').exists()


@pytest.mark.parametrize('args', [
    ['--video-url', 'https://vimeo.com/123'], ['--video-url', 'http://www.youtube.com/watch?v=' + FAKE_ID],
    ['--video-url', 'https://www.youtube.com/watch?v=short'], ['--observed-on', '2999-01-01'], ['--observed-on', ''],
    ['--observed-duration-s', 'nan'], ['--observed-duration-s', '0'], ['--verified-by', ' '], ['--observed-channel', ''],
    ['--video', 'local.mp4']])
def test_remote_import_refuses_bad_provenance(tmp_path, feeds, args):
    _, rows = template(tmp_path, feeds)
    sheet = write(tmp_path/'ok.csv', filled(rows))
    (tmp_path/'local.mp4').write_bytes(b'synthetic')
    option = dict(zip(REMOTE[::2], REMOTE[1::2])) | {args[0]: str(tmp_path/args[1]) if args[0] == '--video' else args[1]}
    with pytest.raises(SystemExit):
        run(tmp_path, feeds, '--id', 'bad', '--from-csv', str(sheet), *[x for kv in option.items() for x in kv])


def test_validator_refuses_hand_made_remote_documents(tmp_path, feeds):
    _, rows = template(tmp_path, feeds)
    run(tmp_path, feeds, '--id', 'ok', '--from-csv', str(write(tmp_path/'ok.csv', filled(rows))), *REMOTE)
    good = json.loads((tmp_path/'demo'/'mappings'/'ok.json').read_text())
    validate_mapping(good)
    bad = [
        good | {'video': good['video'] | {'sha256': hashlib.sha256(FAKE_URL.encode()).hexdigest()}},  # URL hash posing as content
        good | {'verified': False},
        good | {'kind': 'test_fixture', 'verified': False},
        good | {'video': good['video'] | {'url': f'https://youtu.be/{FAKE_ID}'}},
        good | {'video': good['video'] | {'observed_duration_s': float('inf')}},
        good | {'verification': good['verification'] | {'csv_sha256': None}},
        good | {'verification': good['verification'] | {'method': 'player_opened'}},
        good | {'rows': [good['rows'][0] | {'review': {'by': 'x'}}, *good['rows'][1:]]},
        good | {'rows': [good['rows'][0] | {'reveal_at_s': 500.0}]},
        good | {'rows': [good['rows'][0], *good['rows'][2:]]},
    ]
    for doc in bad:
        with pytest.raises(ValueError):
            validate_mapping(doc)


def test_local_file_mode_unchanged_and_new_alias_still_needs_every_reviewer(tmp_path, feeds):
    video = tmp_path/'clip.mp4'
    video.write_bytes(b'synthetic local bytes')
    own = tmp_path/'own.csv'
    own.write_text(f'index,pitch_key,video_s\n2,{GAME}:2:1,10\n3,{GAME}:2:2,20\n')
    run(tmp_path, feeds, '--id', 'local', '--from-csv', str(own), '--video', str(video), '--verified-by', 'synthetic-tester')
    made = json.loads((tmp_path/'demo'/'mappings'/'local.json').read_text())
    assert made['kind'] == 'video_verified' and made['video'] == {'file_name': 'clip.mp4', 'sha256': hashlib.sha256(b'synthetic local bytes').hexdigest()}
    assert made['verification'] == {'by': 'synthetic-tester', 'csv': 'own.csv'} and 'review' not in made['rows'][0]
    assert DemoClock(tmp_path/'demo').mapping('local')['video']['sha256']
    with pytest.raises(ValueError):  # local video_verified without a content hash is still refused
        validate_mapping(made | {'video': {'file_name': 'clip.mp4'}})
    alias = tmp_path/'alias.csv'
    alias.write_text(f'pitch_key,video_seconds_result_confirmed,verified_by\n{GAME}:2:1,10,a\n{GAME}:2:2,20,\n')
    run(tmp_path, feeds, '--id', 'alias', '--from-csv', str(alias), '--video', str(video), '--verified-by', 'synthetic-tester')
    downgraded = json.loads((tmp_path/'demo'/'mappings'/'alias.json').read_text())
    assert downgraded['kind'] == 'game_timestamps_unsynced' and downgraded['verified'] is False
    assert [r['reveal_at_s'] for r in downgraded['rows']] == [10.0, 20.0]
