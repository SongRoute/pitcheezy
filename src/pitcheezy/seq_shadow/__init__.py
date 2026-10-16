"""SEQ-001 N1 sequence-model SHADOW (job 20261012-seq-shadow-001). Research shadow only: predictions are recorded
in a separate append-only ledger, nothing is shown to a fan and nothing is adopted.

This module holds only what every part shares: the configuration loader, file hashing, the list of this job's own
source files, and the sha256-pinned import of the reviewed Step 5 shadow modules (``shadow_sources``,
``shadow_guard``, ``shadow_runner``), which are imported in place by file path and never copied or edited.

Importing this package sets the numeric thread environment variables to 1 when they are not set yet (the scripts
set them unconditionally before importing anything numeric).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'MKL_NUM_THREADS')
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, '1')

REPO = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO/'configs'/'seq_shadow.json'
# This job's own source: what a grant's ``source_sha256.own`` must name, byte for byte.
OWN_SOURCE = ('src/pitcheezy/seq_shadow/__init__.py', 'src/pitcheezy/seq_shadow/tokens.py',
              'src/pitcheezy/seq_shadow/model.py', 'src/pitcheezy/seq_shadow/calibration.py',
              'src/pitcheezy/seq_shadow/adapter.py', 'src/pitcheezy/seq_shadow/ledger.py',
              'src/pitcheezy/seq_shadow/summary.py', 'scripts/seq_shadow_runner.py', 'scripts/seq_shadow_parity.py',
              'configs/seq_shadow.json')
STEP5_SHADOW_MODULES = ('shadow_sources', 'shadow_guard', 'shadow_runner')  # import order matters
REQUIRED_KEYS = ('caps', 'protected', 'heavy_lock_path', 'forbidden_path_parts', 'step5_shadow', 'step5_ledger',
                 'model', 'calibration', 'encoder_seq', 'tokens', 'metrics', 'bootstrap', 'parity', 'runtime_python',
                 'grant_name', 'author', 'threads', 'claims')


class PinMismatch(ValueError):
    """A pinned file does not hash to its registered sha256 (or is absent). Nothing was loaded from it."""


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_pinned(path, expected_sha256, cap=None):
    """Bytes of ``path`` read ONCE and checked against ``expected_sha256``; the caller parses exactly these bytes,
    so the file is never reopened between the check and the use."""
    try:
        with open(path, 'rb') as handle:
            raw = handle.read() if cap is None else handle.read(cap+1)
    except OSError as exc:
        raise PinMismatch(f'pinned file unreadable: {path} ({type(exc).__name__})') from None
    if cap is not None and len(raw) > cap:
        raise PinMismatch(f'pinned file exceeds its read cap: {path}')
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PinMismatch(f'sha256 differs from the pin: {path}')
    return raw


def load_config(path=None):
    config = json.loads(Path(path or CONFIG_PATH).read_text())
    if not str(config.get('status', '')).startswith('provisional'):
        raise ValueError('seq shadow configuration must be marked provisional')
    absent = [k for k in REQUIRED_KEYS if k not in config]
    if absent:
        raise ValueError('seq shadow configuration lacks: '+','.join(absent))
    return config


def own_source_sha256(repo=REPO):
    return {name: file_sha256(Path(repo)/name) for name in OWN_SOURCE}


def load_step5_shadow(config):
    """Namespace (S, G, RUN, sha256) of the reviewed Step 5 shadow modules, imported by file path from their own
    directory after all four registered sha256 (three modules and ``shadow_config.json``) match. Raises
    ``PinMismatch`` otherwise, before any of them is executed. No bytecode is written next to them.

    ``shadow_runner`` itself says ``import shadow_sources`` / ``from shadow_guard import ...``, so the three are
    registered in ``sys.modules`` under those names; a module of another origin already holding a name refuses."""
    sys.dont_write_bytecode = True
    spec = config['step5_shadow']
    directory = Path(spec['dir'])
    seen = {}
    for name, expected in spec['sha256'].items():
        try:
            seen[name] = file_sha256(directory/name)
        except OSError:
            seen[name] = None
    wrong = sorted(name for name, sha in seen.items() if sha != spec['sha256'][name])
    needed = {m+'.py' for m in STEP5_SHADOW_MODULES} | {'shadow_config.json'}
    if wrong or not needed <= set(spec['sha256']):
        raise PinMismatch('Step 5 shadow source differs from the pinned bytes: '+','.join(wrong or sorted(needed)))
    modules = {}
    for name in STEP5_SHADOW_MODULES:
        path = (directory/(name+'.py')).resolve()
        held = sys.modules.get(name)
        if held is not None:
            if Path(getattr(held, '__file__', '') or '/').resolve() != path:
                raise PinMismatch('a module of another origin holds the name '+name)
            modules[name] = held
            continue
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        modules[name] = module
    return SimpleNamespace(S=modules['shadow_sources'], G=modules['shadow_guard'], RUN=modules['shadow_runner'],
                           sha256=seen)
