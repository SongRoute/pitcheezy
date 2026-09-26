"""Small filesystem-only checks; no model construction or real run reads."""
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from pitchmdp.data import hash_file
from pitchmdp.matrix_policy_artifacts import APPLEDOUBLE_MAGIC, artifact_names, is_appledouble
import run_ml_policy as policy
import run_ml_offline_rl as rl
import report_ml_policy_groups as groups


def sidecar(path):
    path.write_bytes(APPLEDOUBLE_MAGIC+b'\x00'*28)


def test_metadata_requires_both_prefix_and_magic(tmp_path):
    for name, payload in [('._actual', APPLEDOUBLE_MAGIC), ('._ordinary', b'data'),
                          ('ordinary', APPLEDOUBLE_MAGIC), ('._short', b'\x00\x05')]:
        (tmp_path/name).write_bytes(payload)
    assert is_appledouble(tmp_path/'._actual')
    assert artifact_names(tmp_path) == ['._ordinary', '._short', 'ordinary']
    assert not is_appledouble(tmp_path/'._absent')


def stage(tmp_path, kind):
    policy.dump(tmp_path/'preparation.json', {'frozen': True})
    directory = tmp_path/'stages/p0'; directory.mkdir(parents=True)
    policy.dump(directory/'started.json', {'preparation_sha256': hash_file(tmp_path/'preparation.json')})
    (directory/'payload.bin').write_bytes(b'science')
    (directory/'._ordinary').write_bytes(b'registered science')
    sidecar(directory/'._payload.bin')
    if kind == 'rl':
        rl.seal(directory)
        verify = lambda: rl.verify_sealed(directory)
    else:
        policy.seal_stage(directory)
        verify = (lambda: policy.verify_stage(tmp_path, 'p0')) if kind == 'policy' else (lambda: groups.Reader().sealed(directory))
    return directory, verify


@pytest.mark.parametrize('kind', ['policy', 'rl', 'groups'])
def test_metadata_added_or_changed_after_seal_does_not_change_science(tmp_path, kind):
    directory, verify = stage(tmp_path, kind)
    manifest_bytes = (directory/'manifest.json').read_bytes()
    names = policy.read_json(directory/'manifest.json')['artifact_hashes']
    assert '._payload.bin' not in names and '._ordinary' in names
    sidecar(directory/'._manifest.json')
    (directory/'._payload.bin').write_bytes(APPLEDOUBLE_MAGIC+b'new metadata')
    verify()
    assert (directory/'manifest.json').read_bytes() == manifest_bytes
    (directory/'payload.bin').write_bytes(b'tampered science')
    with pytest.raises(ValueError): verify()


@pytest.mark.parametrize('kind', ['policy', 'rl', 'groups'])
@pytest.mark.parametrize('name', ['._unregistered', 'extra.bin', 'nested/manifest.json'])
def test_unexpected_ordinary_files_remain_rejected(tmp_path, kind, name):
    directory, verify = stage(tmp_path, kind)
    path = directory/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'not AppleDouble')
    with pytest.raises(ValueError, match='family'): verify()


@pytest.mark.parametrize('kind', ['policy', 'rl', 'groups'])
def test_metadata_magic_tampering_or_registered_payload_hidden_as_metadata_fails(tmp_path, kind):
    directory, verify = stage(tmp_path, kind)
    (directory/'._payload.bin').write_bytes(b'changed to ordinary data')
    with pytest.raises(ValueError, match='family'): verify()
    sidecar(directory/'._payload.bin')
    sidecar(directory/'._ordinary')
    with pytest.raises(ValueError, match='family'): verify()


def test_rollout_npz_enumeration_ignores_only_real_metadata(tmp_path):
    np.savez(tmp_path/'1-1-1.npz', P0_values=np.array([.2,.3]), P0_truncated=np.array([False,False]))
    sidecar(tmp_path/'._1-1-1.npz')
    values, flags = groups.load_rollouts(groups.Reader(), tmp_path, {'pa_keys': ['1:1:1']}, ['P0'])
    np.testing.assert_array_equal(values['P0'], [[.2,.3]])
    assert not flags['P0'].any()
    (tmp_path/'._unexpected.npz').write_bytes(b'ordinary npz-like payload')
    with pytest.raises(ValueError, match='PA file family'): groups.load_rollouts(groups.Reader(), tmp_path, {'pa_keys': ['1:1:1']}, ['P0'])
