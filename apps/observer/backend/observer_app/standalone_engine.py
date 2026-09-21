"""Load the generated local scientific runtime in an uncontaminated process."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

DEFAULT_RUNTIME = Path(__file__).resolve().parents[2]/'runtime_src'
NAMESPACES = ('pitchmdp', 'minimal_pitch_service', 'representation_adapters',
              'run_sequence_frequency_baselines', 'run_sequence_context_frequency', 'diagnose_sequence_legality')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime(runtime_root=DEFAULT_RUNTIME, bundle=None):
    runtime = Path(runtime_root).resolve()
    manifest = json.loads((runtime/'runtime_manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported scientific runtime manifest')
    for relative, expected in manifest['runtime_files'].items():
        path = (runtime/relative).resolve()
        if not path.is_relative_to(runtime) or _sha(path) != expected:
            raise ValueError('Scientific runtime source mismatch: '+relative)
    if bundle is not None:
        bundle = Path(bundle).resolve()
        for name, expected in [('bundle_manifest.json', manifest['bundle_manifest_sha256']),
                               ('source_hashes.json', manifest['source_manifest_sha256'])]:
            if _sha(bundle/name) != expected:
                raise ValueError('Scientific runtime and model bundle identity differ: '+name)
    for name, module in list(sys.modules.items()):
        if any(name == namespace or name.startswith(namespace+'.') for namespace in NAMESPACES):
            source = getattr(module, '__file__', None)
            if source is None or not Path(source).resolve().is_relative_to(runtime):
                raise RuntimeError('External scientific module already imported; start an isolated process: '+name)
    return manifest


def load_engine(bundle, *, runtime_root=DEFAULT_RUNTIME, device='cpu'):
    manifest = verify_runtime(runtime_root, bundle)
    runtime = Path(runtime_root).resolve()
    sys.path.insert(0, str(runtime))
    importlib.invalidate_caches()
    module = importlib.import_module('minimal_pitch_service')
    engine = module.Engine(bundle, device=device)
    verify_runtime(runtime, bundle)
    engine.runtime_manifest = manifest
    engine.runtime_root = runtime
    return engine


def imported_scientific_paths():
    return {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
            if getattr(module, '__file__', None) is not None and
            any(name == namespace or name.startswith(namespace+'.') for namespace in NAMESPACES)}
