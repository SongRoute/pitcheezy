"""Generated-only source/output custody checks; no actual six-game files."""
import json
from pathlib import Path
import hashlib
import stat

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.ws_fullseason.audit import sha as file_sha
from pitcheezy.seq_shadow.postcutoff_offline import SIX, prepare
from scripts.fullseason_postcutoff_offline import _plan, _private_write, _source, score_population
from scripts.build_fullseason_postcutoff_registration import build
from scripts.verify_fullseason_postcutoff_offline import verify
from test_postcutoff_offline import SHA, source


def fake(snapshot):
    return {'status': 'scored', 'leaf_order': ['ball', 'foul', 'hit_by_pitch', 'narrow_whiff',
             'other_strike', 'ground_ball_in_play', 'other_in_play'],
            'probabilities': [1 / 7] * 7, 'bundle_sha256': SHA,
            'snapshot_sha256': snapshot['snapshot_sha256'],
            'stream_evidence_kind': 'retrospective', 'pitch_type_basis': 'observed_actual',
            'prospective_claim': False}


def test_private_rows_bind_same_prepared_population_and_single_score_call():
    frame, M = source()
    prepared = prepare(frame, SHA, M)
    calls = []
    def once(snapshot):
        calls.append(snapshot['query']['pitch_number'])
        return fake(snapshot)
    metrics, artifact = score_population(prepared, once, source_sha256=SHA, bundle_sha256=SHA)
    assert calls == [1, 2, 3]
    assert metrics['scored'] == len(artifact['rows']) == 3
    assert all(row['status'] == 'scored' and row['probabilities'] == [1/7]*7 for row in artifact['rows'])
    assert artifact['source_sha256'] == SHA and artifact['bundle_sha256'] == SHA
    with pytest.raises(ValueError, match='binding'):
        score_population(prepared, lambda s: dict(fake(s), stream_evidence_kind='captured'),
                         source_sha256=SHA, bundle_sha256=SHA)


def test_private_rows_include_abstention_without_probability():
    frame, M = source(); frame = frame.drop(index=1).reset_index(drop=True)
    prepared = prepare(frame, SHA, M)
    metrics, artifact = score_population(prepared, fake, source_sha256=SHA, bundle_sha256=SHA)
    assert metrics['scored'] == 1 and len(artifact['rows']) == 2
    assert artifact['rows'][1]['status'] == 'abstain'
    assert artifact['rows'][1]['probabilities'] is None


def test_source_verifies_exact_manifest_and_projected_bytes_before_decode(tmp_path):
    frame, M = source()
    projected = tmp_path / 'postseason_six.parquet'
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), projected)
    item = {'file': projected.name, 'bytes': projected.stat().st_size, 'sha256': file_sha(projected),
            'rows': len(frame), 'columns': M.cp.PROJECT}
    manifest = {'schema': 'postseason-six-exact-game-v1', 'status': 'COMPLETE',
                'games': [{'game_pk': g, 'officialDate': day, 'gameType': 'L',
                           'rows': len(frame) if g == 849812 else 1} for g, day in SIX.items()],
                'projected': item}
    # A generated partial cohort is rejected before Parquet decode.
    mpath = tmp_path / 'manifest.json'; mpath.write_text(json.dumps(manifest))
    reg = {'bounds': {'source_rows': len(frame)},
           'source': {'manifest': {'path': str(mpath), 'bytes': mpath.stat().st_size,
                                   'sha256': file_sha(mpath)},
                      'projected': {'path': str(projected), 'bytes': item['bytes'],
                                    'sha256': item['sha256'], 'rows': len(frame)}}}
    with pytest.raises(ValueError, match='projected source'):
        _source(reg, M)
    manifest['projected']['rows'] = sum(g['rows'] for g in manifest['games'])
    reg['source']['projected']['rows'] = manifest['projected']['rows']
    reg['bounds']['source_rows'] = manifest['projected']['rows']
    mpath.write_text(json.dumps(manifest)); reg['source']['manifest'].update(bytes=mpath.stat().st_size,
                                                                               sha256=file_sha(mpath))
    with pytest.raises(ValueError, match='Parquet footer'):
        _source(reg, M)
    projected.write_bytes(projected.read_bytes() + b'x')
    with pytest.raises(ValueError, match='registered regular input bytes'):
        _source(reg, M)


def test_independent_private_output_arithmetic_and_population_digest(tmp_path):
    frame, M = source(); prepared = prepare(frame, SHA, M)
    metrics, body = score_population(prepared, fake, source_sha256=SHA, bundle_sha256=SHA)
    raw = (json.dumps(body, separators=(',', ':'), sort_keys=True) + '\n').encode()
    (tmp_path / 'score_rows.json').write_bytes(raw)
    reg_path = tmp_path / 'registration.json'
    reg_path.write_text(json.dumps({'bounds': {'source_rows': len(frame)},
                                    'output': str(tmp_path)}))
    receipt = {'status': 'COMPLETED', 'score_rows_file': 'score_rows.json',
               'score_rows_bytes': len(raw), 'score_rows_sha256': hashlib.sha256(raw).hexdigest(),
               'registration_sha256': file_sha(reg_path), 'collector_rows': len(frame),
               'prospective_claim': False, 'superiority_claim': False,
               'projected_sha256': SHA, 'runtime_bundle_sha256': SHA,
               'census': prepared['census'], 'metrics': metrics}
    path = tmp_path / 'receipt.json'; path.write_text(json.dumps(receipt))
    assert verify(path, reg_path)['scored'] == 3
    changed = json.loads(path.read_text())
    changed['metrics']['overall']['model_nll'] += .1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='metric arithmetic'):
        verify(path, reg_path)
    changed['metrics'] = metrics
    changed['census']['accepted_keys_sha256'] = '0' * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='population'):
        verify(path, reg_path)


def test_inactive_registration_builder_uses_collector_metadata_only(tmp_path):
    from scripts.fullseason_postcutoff_offline import CONFIG_SHA, PROFILE_SHA
    projection = tmp_path / 'postseason_six.parquet'; projection.write_bytes(b'generated fixture bytes')
    manifest = {'schema': 'postseason-six-exact-game-v1', 'status': 'COMPLETE',
                'games': [{'game_pk': g, 'officialDate': day, 'gameType': 'L'} for g, day in SIX.items()],
                'projected': {'file': projection.name, 'bytes': projection.stat().st_size,
                              'sha256': 'b'*64, 'rows': 1}}
    mpath = tmp_path / 'manifest.json'; mpath.write_text(json.dumps(manifest))
    config = tmp_path / 'config.json'; config.write_text('{}')
    old = tmp_path / 'prior.json'; old.write_text(json.dumps({
        'adapter_config': {'path': str(config), 'sha256': CONFIG_SHA},
        'profile_bundle': {'directory': str(tmp_path), 'manifest_sha256': PROFILE_SHA},
        'leaf_order': ['ball', 'foul', 'hit_by_pitch', 'narrow_whiff', 'other_strike',
                       'ground_ball_in_play', 'other_in_play'],
        'recipe': str(config), 'model_artifacts': {}}))
    reg = build(mpath, tmp_path / 'attempt', prior=old)
    assert reg['execute'] is False and reg['source']['projected']['sha256'] == 'b'*64
    assert reg['profile_bundle']['serving_origin'] == '2026-10-12'


def test_private_artifact_exclusive_owner_only_mode(tmp_path):
    target = tmp_path / 'score_rows.json'
    _private_write(target, b'generated')
    assert target.read_bytes() == b'generated'
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _private_write(target, b'changed')


def test_changed_frozen_analysis_plan_refused_before_source_decode(tmp_path):
    changed = tmp_path / 'plan.md'; changed.write_text('generated changed plan')
    with pytest.raises(ValueError, match='analysis plan changed'):
        _plan({'analysis_plan_path': str(changed)})
