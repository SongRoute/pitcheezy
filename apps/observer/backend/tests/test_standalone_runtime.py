"""Runtime packaging provenance and import-isolation contracts; no model load."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2]/'scripts')]
import prepare_runtime
from observer_app import standalone_engine


def test_extraction_keeps_exact_decorated_definition_without_cli_side_effects():
    source = 'import unavailable_training_module\n\n@decorate\ndef predict(x):\n    # keep this arithmetic and comment\n    return x / 3.0\n\nlaunch_training()\n'
    chosen = prepare_runtime.extract_definitions(source, ['predict'])
    assert chosen['predict']['source'] == '@decorate\ndef predict(x):\n    # keep this arithmetic and comment\n    return x / 3.0\n'
    assert chosen['predict']['sha256'] == hashlib.sha256(chosen['predict']['source'].encode()).hexdigest()
    assert 'launch_training' not in chosen['predict']['source']
    with pytest.raises(ValueError, match='Missing captured symbols'):
        prepare_runtime.extract_definitions(source, ['unknown'])


def test_preparation_verifies_capture_and_preserves_symbol_provenance(tmp_path, monkeypatch):
    bundle, output = tmp_path/'bundle', tmp_path/'runtime'
    (bundle/'source').mkdir(parents=True)
    original = 'import forbidden_cli\n\nVALUE = (1, 2, 3)\n\ndef predict(x):\n    return x + VALUE[0]\n\nforbidden_cli.train()\n'
    (bundle/'source/original.py').write_text(original)
    pinned = hashlib.sha256(original.encode()).hexdigest()
    (bundle/'source_hashes.json').write_text(json.dumps({'original.py': pinned}))
    (bundle/'bundle_manifest.json').write_text('{}')
    monkeypatch.setattr(prepare_runtime, 'COPIES', [])
    monkeypatch.setattr(prepare_runtime, 'EXTRACTIONS', {'prediction.py': ('original.py', ['VALUE', 'predict'], '')})
    manifest = prepare_runtime.prepare_runtime(bundle, output, include_dependency_snapshot=False)
    code = (output/'prediction.py').read_text()
    assert 'forbidden_cli' not in code
    assert manifest['extraction']['prediction.py']['source_sha256'] == pinned
    assert manifest['runtime_files']['prediction.py'] == hashlib.sha256(code.encode()).hexdigest()
    assert not manifest['portable_verified'] and not manifest['training_invoked']
    with pytest.raises(FileExistsError):
        prepare_runtime.prepare_runtime(bundle, output, include_dependency_snapshot=False)
    (bundle/'source/original.py').write_text(original+'\n# changed\n')
    with pytest.raises(ValueError, match='captured source mismatch'):
        prepare_runtime.prepare_runtime(bundle, tmp_path/'second', include_dependency_snapshot=False)


def runtime_manifest(tmp_path):
    source = tmp_path/'minimal.py'
    source.write_text('x = 1\n')
    manifest = {'schema_version': 1, 'runtime_files': {'minimal.py': hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path/'runtime_manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_runtime_rejects_external_module_even_if_generated_files_are_valid(tmp_path, monkeypatch):
    runtime_manifest(tmp_path)
    monkeypatch.setattr(standalone_engine, 'NAMESPACES', ('fake_scientific_namespace',))
    monkeypatch.setitem(sys.modules, 'fake_scientific_namespace', SimpleNamespace(__file__='/original/experiments/module.py'))
    with pytest.raises(RuntimeError, match='isolated process'):
        standalone_engine.verify_runtime(tmp_path)
    monkeypatch.setitem(sys.modules, 'fake_scientific_namespace', SimpleNamespace(__file__=str(tmp_path/'minimal.py')))
    assert standalone_engine.verify_runtime(tmp_path)['schema_version'] == 1


def test_runtime_tampering_fails_before_any_model_load(tmp_path, monkeypatch):
    runtime_manifest(tmp_path)
    (tmp_path/'minimal.py').write_text('x = 2\n')
    with pytest.raises(ValueError, match='source mismatch'):
        standalone_engine.verify_runtime(tmp_path)
