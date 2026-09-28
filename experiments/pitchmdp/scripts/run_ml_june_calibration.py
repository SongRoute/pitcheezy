"""Registered additive worker for ML-JUNE-CALIBRATION-v1 (B0 archived / B1 global / B2 volume blend).

One invocation runs exactly one stage under the shared heavy lock:

    --config EXEC.json --output-dir ABS --stage prepare|profile|predict|fit|apply|score [--seed 0..4]

The execution JSON (registration checkout D) pins the clean source commit C,
the immutable scientific config/contract, the absolute source closure, the
numerical environment and the processed/cache data.  Every stage re-verifies all
pins before decoding any value, writes into a fresh stage directory and writes
``manifest.json`` last.  A failure leaves ``failure.json`` and no manifest; there
is no retry, fallback or tolerance change.  Wall cost is owned by the external
supervisor (Popen-to-reap); internal timings are diagnostics only.  No stage
fetches raw data, touches 2026, fits a network, or writes into a parent run.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
SCRIPT_REL = 'experiments/pitchmdp/scripts/run_ml_june_calibration.py'
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]

import numpy as np
import pandas as pd
import pyarrow
import scipy
import torch

from pitchmdp import matrix_june_calibration as mj
from pitchmdp import matrix_june_calibration_metrics as mm
from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_benchmark import predict_streamed, select_keys
from pitchmdp.matrix_data import canonical_hash, load_verified_processed_cache, ordered_key_hash
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_g0_whole import frozen_weights
from pitchmdp.matrix_g0_whole_metrics import aligned_population
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.matrix_sharing import SharingContext, SharingPredictor
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_data import HistoryStore
from run_temporal_blend import assign_fold
from run_sequence_frequency_baselines import temperature_predictions
import run_sequence_context_frequency  # noqa: F401  pickled frequency classes inside aux.pkl

CELL = 'G0-global'
STAGE_DIRS = {'prepare': 'prepared', 'profile': 'profile', 'fit': 'fit', 'apply': 'apply', 'score': 'analysis'}
EXEC_KEYS = ('protocol', 'family_id', 'enabled', 'code_commit_c', 'scientific_config', 'contract', 'source_hashes',
             'environment', 'data', 'output_dir', 'heavy_lock')
ENV_KEYS = ('python', 'python_version', 'numpy', 'scipy', 'pandas', 'torch', 'pyarrow', 'machine',
            'dynamic_library_path', 'pythonpath')
DATA_KEYS = ('local_config', 'processed_pitches', 'physics_cache', 'quality_manifest', 'cache_manifest')
# Static project closure; every other project module loaded at runtime must also be pinned.
REQUIRED_SOURCES = (
    'pitchmdp/__init__.py', 'pitchmdp/data.py', 'pitchmdp/model.py', 'pitchmdp/archetypes.py',
    'pitchmdp/sequence_data.py', 'pitchmdp/sequence_delivery.py', 'pitchmdp/sequence_model.py',
    'pitchmdp/matrix_data.py', 'pitchmdp/matrix_features.py', 'pitchmdp/matrix_models.py',
    'pitchmdp/matrix_benchmark.py', 'pitchmdp/matrix_sharing.py', 'pitchmdp/matrix_panel.py',
    'pitchmdp/matrix_metrics.py', 'pitchmdp/matrix_group_metrics.py', 'pitchmdp/matrix_g0_whole.py',
    'pitchmdp/matrix_g0_whole_metrics.py', 'pitchmdp/matrix_policy_artifacts.py',
    'pitchmdp/matrix_june_calibration.py', 'pitchmdp/matrix_june_calibration_metrics.py',
    'scripts/run_ml_june_calibration.py', 'scripts/run_temporal_blend.py',
    'scripts/run_sequence_frequency_baselines.py', 'scripts/run_sequence_context_frequency.py',
    'scripts/run_sequence_calibration.py')
PINNED_SOURCE_IDENTITIES = {  # scientific-config pins of reused pure sources, relative to the execution project
    'pitchmdp/model.py': 'p4_model_source',
    'scripts/run_sequence_frequency_baselines.py': 'p4_frequency_temperature_source',
    'scripts/score_ml_matrix.py': 'p10_scalar_summarizer_source'}
NETWORK = {'kind': 'flatten_mlp', 'n_context': 52, 'width': 128, 'individual_tau': 1000, 'cluster_tau': 10000,
           'may_rows': 2603}
MEMBER_FIELDS = ('seed', 'june_keys', 'june_y', 'june_game_pk', 'june_pitcher', 'june_calibrated', 'june_raw',
                 'june_delivery_level')
APPLY_FIELDS = ('keys', 'game_pk', 'pitcher', 'groups', 'predictors', 'B0_primary', 'B1_primary', 'B2_primary',
                'B0_seed_primary', 'B1_seed_primary', 'B2_seed_primary')
BLEND_META_COMPARE = ('pitcher', 'batter', 'in_cpanel', 'train_volume', 'game_role', 'throwing_hand', 'month')


class Terminated(BaseException):
    """SIGTERM from the supervisor; converted so the failure record is still written."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- execution config

def _pin(value, name):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError(f'{name} must be exactly {{path, sha256}}')
    if not isinstance(value['path'], str) or not Path(value['path']).is_absolute():
        raise ValueError(f'{name}.path must be absolute')
    if not mj.is_sha256(value['sha256']):
        raise ValueError(f'{name}.sha256 must be lowercase hex SHA256')
    return value


def validate_execution_config(config):
    """Explicit root-supplied execution schema; the scientific config itself stays execution-disabled."""
    if not isinstance(config, dict) or not set(EXEC_KEYS) <= set(config) or set(config) - set(EXEC_KEYS) - {'registration'}:
        raise ValueError('Execution config must have exactly: ' + ', '.join(EXEC_KEYS) + ' (+ optional registration)')
    if config['protocol'] != mj.EXECUTION_PROTOCOL or config['family_id'] != mj.FAMILY_ID:
        raise ValueError('Execution protocol/family differs')
    if config['enabled'] is not True:
        raise ValueError('Execution registration is not enabled')
    if not mj.is_commit(config['code_commit_c']):
        raise ValueError('code_commit_c must be a full 40-hex commit')
    _pin(config['scientific_config'], 'scientific_config')
    _pin(config['contract'], 'contract')
    sources = config['source_hashes']
    if not isinstance(sources, dict) or not sources or any(
            not Path(p).is_absolute() or not mj.is_sha256(d) for p, d in sources.items()):
        raise ValueError('source_hashes must map absolute paths to SHA256')
    env = config['environment']
    if not isinstance(env, dict) or set(env) != set(ENV_KEYS) or any(not isinstance(v, str) for v in env.values()):
        raise ValueError('environment must pin exactly: ' + ', '.join(ENV_KEYS))
    data = config['data']
    if not isinstance(data, dict) or set(data) != set(DATA_KEYS):
        raise ValueError('data must pin exactly: ' + ', '.join(DATA_KEYS))
    for name in DATA_KEYS:
        _pin(data[name], 'data.' + name)
    for name in ('output_dir', 'heavy_lock'):
        if not isinstance(config[name], str) or not Path(config[name]).is_absolute():
            raise ValueError(name + ' must be an absolute path')
    if 'registration' in config and not isinstance(config['registration'], dict):
        raise ValueError('registration must be an object')
    return config


def runtime_environment():
    return {'python': str(Path(sys.executable).resolve()), 'python_version': platform.python_version(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__,
            'torch': torch.__version__, 'pyarrow': pyarrow.__version__, 'machine': platform.machine(),
            'dynamic_library_path': os.environ.get('DYLD_LIBRARY_PATH', ''),
            'pythonpath': os.environ.get('PYTHONPATH', '')}


def validate_native_runtime():
    if sys.platform == 'darwin':
        torch_lib = (Path(torch.__file__).resolve().parent / 'lib').resolve()
        configured = [Path(v).resolve() for v in os.environ.get('DYLD_LIBRARY_PATH', '').split(':') if v]
        if torch_lib not in configured:
            raise ValueError(f'Launch with DYLD_LIBRARY_PATH={torch_lib} (single OpenMP runtime)')
    if os.environ.get('KMP_DUPLICATE_LIB_OK', '').lower() in ('true', '1', 'yes'):
        raise ValueError('Unsafe duplicate OpenMP override is forbidden')


def loaded_project_sources():
    root = PROJECT.resolve()
    found = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and Path(name).is_absolute():  # e.g. torch._classes has a relative pseudo-file
            path = Path(name).resolve()
            if path.is_relative_to(root) and path.suffix == '.py':
                found[str(path)] = path
    return found


def foreign_project_modules():
    """Project modules imported from any checkout other than this execution source C."""
    root, foreign = PROJECT.resolve(), []
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not isinstance(path, str) or not Path(path).is_absolute():
            continue
        path = Path(path).resolve()
        if not path.is_relative_to(root) and (name == 'pitchmdp' or name.startswith('pitchmdp.')
                                              or '/experiments/pitchmdp/' in path.as_posix()):
            foreign.append(str(path))
    return sorted(foreign)


def source_closure():
    """Absolute project sources actually loaded by this worker (no data, no lock)."""
    paths = {str((PROJECT / rel).resolve()) for rel in REQUIRED_SOURCES} | set(loaded_project_sources())
    return {path: hash_file(Path(path)) for path in sorted(paths)}


# ---------------------------------------------------------------- provenance (git)

def _git(*args, cwd=REPO):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, env=env)


def _blob_sha1(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def tracked_clean(path, label):
    path = Path(path).resolve()
    top = _git('rev-parse', '--show-toplevel', cwd=path.parent)
    if top.returncode != 0 or not top.stdout.strip():
        raise ValueError(f'{label} is not inside a git checkout')
    root = Path(top.stdout.strip()).resolve()
    rel = path.relative_to(root).as_posix()
    if _git('ls-files', '--error-unmatch', '--', rel, cwd=root).returncode != 0:
        raise ValueError(f'{label} is not tracked')
    if _git('status', '--porcelain', '--untracked-files=all', '--', rel, cwd=root).stdout.strip():
        raise ValueError(f'{label} has staged or unstaged changes')
    if _git('rev-parse', '--verify', f'HEAD:{rel}', cwd=root).stdout.strip() != _blob_sha1(path.read_bytes()):
        raise ValueError(f'{label} bytes differ from its HEAD blob')
    return {'path': str(path), 'repository_root': str(root), 'relpath': rel,
            'repository_head': _git('rev-parse', '--verify', 'HEAD^{commit}', cwd=root).stdout.strip()}


def execution_provenance(config, config_path):
    top = Path(_git('rev-parse', '--show-toplevel').stdout.strip() or '/nonexistent').resolve()
    if top != REPO.resolve() or Path(__file__).resolve() != (REPO / SCRIPT_REL).resolve():
        raise ValueError('Running script is not the execution checkout script')
    head = _git('rev-parse', '--verify', 'HEAD^{commit}').stdout.strip()
    if head != config['code_commit_c']:
        raise ValueError('Execution HEAD differs from registered code_commit_c')
    if _git('status', '--porcelain', '--untracked-files=no').stdout.strip():
        raise ValueError('Execution checkout has tracked changes')
    for rel in (SCRIPT_REL, 'experiments/pitchmdp/pitchmdp/matrix_june_calibration.py',
                'experiments/pitchmdp/pitchmdp/matrix_june_calibration_metrics.py'):
        tracked_clean(REPO / rel, rel)
    return {'registered_code_commit': config['code_commit_c'], 'execution_head': head, 'execution_root': str(REPO),
            'execution_config': tracked_clean(config_path, 'execution config'),
            'scientific_config': tracked_clean(config['scientific_config']['path'], 'scientific config'),
            'contract': tracked_clean(config['contract']['path'], 'contract')}


# ---------------------------------------------------------------- verification

@dataclass
class Context:
    config: dict
    config_path: Path
    config_sha256: str
    sci: dict
    output: Path
    parents: dict
    derived: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    members: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    exposure: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    fit_ledger: object = None

    @property
    def code_commit(self):
        return self.config['code_commit_c']


def check_hash(path, digest, label):
    path = Path(path)
    if not path.is_file() or hash_file(path) != digest:
        raise ValueError(f'Pinned identity changed: {label} ({path})')
    return str(path.resolve()), digest


def verify_sources(config):
    pinned = {str(Path(p).resolve()): d for p, d in config['source_hashes'].items()}
    missing = [rel for rel in REQUIRED_SOURCES if str((PROJECT / rel).resolve()) not in pinned]
    if missing:
        raise ValueError(f'Source closure lacks required files: {missing}')
    unpinned = sorted(set(loaded_project_sources()) - set(pinned))
    if unpinned:
        raise ValueError(f'Loaded project modules are not pinned: {unpinned[:5]}')
    if foreign_project_modules():
        raise ValueError(f'Project modules loaded outside execution source C: {foreign_project_modules()[:5]}')
    for path, digest in pinned.items():
        check_hash(path, digest, 'source')
    return pinned


def verify_output_path(output, config, sci):
    output = Path(output)
    if not output.is_absolute():
        raise ValueError('--output-dir must be absolute')
    if output.resolve() != Path(config['output_dir']).resolve():
        raise ValueError('--output-dir differs from the registered output_dir')
    root = Path(sci['future_output_root']).resolve()
    if not (output.resolve() == root or output.resolve().is_relative_to(root)):
        raise ValueError('Output must be the registered family root or inside it')
    return output.resolve()


def derive_chain(sci, parents):
    """Files reached only through pinned parent preparations; hashes come from those pinned records."""
    p4_run, p11_run = parents['p4_preparation'].parent, parents['p11_preparation'].parent
    p4_prep, p11_prep = read_json(parents['p4_preparation']), read_json(parents['p11_preparation'])
    derived, inputs = {}, {}
    for name, run, prep, rel in (('p4_parent_preparation', p4_run, p4_prep, 'parent_preparation.json'),
                                 ('p4_blend_metadata', p4_run, p4_prep, 'blend_metadata.parquet'),
                                 ('p11_dev_metadata', p11_run, p11_prep, 'dev_metadata.parquet')):
        path = run / rel
        derived[name] = path
        inputs.update([check_hash(path, prep['artifact_hashes'][rel], name)])
    if p4_prep['artifact_hashes']['aux.pkl'] != sci['parents']['p4_auxiliary']['sha256'] or \
            p4_prep['artifact_hashes']['panel.json'] != sci['parents']['p4_panel']['sha256']:
        raise ValueError('P4 preparation disagrees with the pinned auxiliary/panel')
    if read_json(parents['p11_analysis_manifest'])['predictions_sha256'] != sci['parents']['p11_predictions']['sha256']:
        raise ValueError('P11 analysis manifest disagrees with the pinned predictions')
    if read_json(parents['p10_manifest'])['results_sha256'] != sci['parents']['p10_results']['sha256']:
        raise ValueError('P10 analysis manifest disagrees with the pinned results')
    return derived, inputs, p4_prep, p11_prep


def member_records(sci, parents, p11_prep):
    """Frozen member identity from pinned P11 preparation; every referenced file is hash-checked."""
    runs = {'EXP-P4-001': parents['p4_preparation'].parent.resolve(),
            'EXP-P10-001': parents['p10_preparation'].parent.resolve()}
    frozen = p11_prep['frozen_calibration']['may_delivery_temperatures']
    records, inputs = {}, {}
    for seed in mj.SEEDS:
        m = p11_prep['members'][str(seed)]
        pinned = sci['parents'][f'cpanel_member_seed{seed}']
        folder = Path(m['directory'])
        if m['seed'] != seed or Path(m['source_run']).resolve() != runs[mj.SEED_PARENT[seed]] or \
                (folder / 'predictions.npz').resolve() != Path(pinned['path']).resolve() or \
                m['predictions_sha256'] != pinned['sha256']:
            raise ValueError(f'Seed {seed} member differs from the pinned G0 source arm')
        for path, digest, label in ((m['model_path'], m['model_sha256'], 'model'),
                                    (m['state_path'], m['prediction_state_sha256'], 'prediction state'),
                                    (folder / 'calibration.json', m['calibration_sha256'], 'calibration'),
                                    (Path(m['fit_dir']) / 'fit.json', m['fit_sha256'], 'fit report'),
                                    (Path(m['fit_dir']) / 'state.json', m['fit_state_sha256'], 'fit state')):
            inputs.update([check_hash(path, digest, f'seed{seed} {label}')])
        calibration = read_json(folder / 'calibration.json')
        network = m['network']
        if calibration.get('cell') != CELL or calibration['delivery_temperature'] != m['delivery_temperature'] or \
                frozen[str(seed)] != m['delivery_temperature'] or \
                calibration['delivery_calibration_rows'] != NETWORK['may_rows'] or \
                (network['kind'], network['n_context'], network['width'], network['n_classes']) != (
                    NETWORK['kind'], NETWORK['n_context'], NETWORK['width'], 10):
            raise ValueError(f'Seed {seed} May calibration/network differs from the frozen G0 member')
        records[seed] = m
    if len({canonical_hash(r['network']) for r in records.values()}) != 1 or \
            len({r['device'] for r in records.values()}) != 1:
        raise ValueError('Members must share one network signature and device')
    return records, inputs


def verify_everything(config, config_path, output, *, git=True):
    """All pins before any value: execution C, config D, scientific config/contract, sources, env, parents, data."""
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config_sha = hash_file(config_path)
    sci_path, contract_path = Path(config['scientific_config']['path']), Path(config['contract']['path'])
    check_hash(sci_path, config['scientific_config']['sha256'], 'scientific config')
    check_hash(contract_path, config['contract']['sha256'], 'contract')
    sci = read_json(sci_path)
    if sci['contract']['sha256'] != config['contract']['sha256']:
        raise ValueError('Scientific config pins a different contract')
    mj.check_scientific_config(sci)
    mm.check_scoring_config(sci)
    if Path(config['heavy_lock']).resolve() != Path(sci['budget']['heavy_lock']).resolve():
        raise ValueError('Heavy lock differs from the scientific registration')
    output = verify_output_path(output, config, sci)
    provenance = execution_provenance(config, config_path) if git else {}
    validate_native_runtime()
    env = runtime_environment()
    if env != config['environment']:
        differing = sorted(k for k in ENV_KEYS if env[k] != config['environment'][k])
        raise ValueError('Runtime environment differs from the registered pins: ' + ', '.join(differing))
    recorded = sci['B0_provenance']['recorded_c1_versions']
    if (env['numpy'], env['scipy'], env['python']) != (recorded['numpy'], recorded['scipy'], recorded['python_executable']):
        raise ValueError('Numerical environment differs from the recorded C1 NumPy/SciPy/Python; stop, never widen')
    inputs = dict(verify_sources(config))
    for rel, parent in PINNED_SOURCE_IDENTITIES.items():
        if hash_file(PROJECT / rel) != sci['parents'][parent]['sha256']:
            raise ValueError(f'Execution source {rel} differs from pinned {parent}')
    legacy = sci['optimizer']['legacy_source']
    if hash_file(REPO / legacy['path']) != legacy['sha256']:
        raise ValueError('Legacy fit_blend source differs from its pin')
    parents = {}
    for name, pin in sci['parents'].items():
        inputs.update([check_hash(pin['path'], pin['sha256'], name)])
        parents[name] = Path(pin['path'])
    data = {}
    for name in DATA_KEYS:
        inputs.update([check_hash(config['data'][name]['path'], config['data'][name]['sha256'], 'data.' + name)])
        data[name] = Path(config['data'][name]['path'])
    derived, chain, p4_prep, p11_prep = derive_chain(sci, parents)
    inputs.update(chain)
    provenance_record = read_json(derived['p4_parent_preparation'])['source_provenance']
    if (provenance_record['quality_manifest_sha256'], provenance_record['cache_manifest_sha256'],
            provenance_record['processed_sha256'], provenance_record['cache_sha256']) != (
            config['data']['quality_manifest']['sha256'], config['data']['cache_manifest']['sha256'],
            config['data']['processed_pitches']['sha256'], config['data']['physics_cache']['sha256']):
        raise ValueError('Pinned processed/cache data differ from the frozen P4 source provenance')
    local = read_json(data['local_config'])
    root = Path(local['artifact_root'])
    for name, rel in (('processed_pitches', 'processed/pitches.parquet'), ('physics_cache', 'cache/sequence_physics.parquet'),
                      ('quality_manifest', 'reports/data_quality.json'), ('cache_manifest', 'cache/sequence_physics.json')):
        if (root / rel).resolve() != data[name].resolve():
            raise ValueError(f'data.{name} is not the local-config verified cache path')
    members, member_inputs = member_records(sci, parents, p11_prep)
    inputs.update(member_inputs)
    inputs[str(config_path)] = config_sha
    ctx = Context(config=config, config_path=config_path, config_sha256=config_sha, sci=sci, output=output,
                  parents=parents, derived=derived, data=data, members=members, inputs=inputs, provenance=provenance)
    ctx.timings['verify_seconds_internal'] = time.perf_counter() - started
    return ctx


def stage_inputs(ctx, seed):
    """Profile/predict decode only these parents; their end-of-stage recheck stays small (cost is projected x12.8)."""
    return [*ctx.config['source_hashes'], ctx.parents['p4_auxiliary'], ctx.parents['p4_preparation'],
            ctx.parents['p11_preparation'], ctx.parents[f'cpanel_member_seed{seed}'], ctx.members[seed]['model_path']]


def reverify_inputs(ctx, names=None):
    """Re-hash pinned inputs before sealing a stage; ``names`` limits it to the parents that stage decoded."""
    inputs = ctx.inputs if names is None else {
        str(Path(p).resolve()): ctx.inputs[str(Path(p).resolve())] for p in names}
    for path, digest in inputs.items():
        if hash_file(Path(path)) != digest:
            raise ValueError(f'Pinned input changed during the stage: {path}')
    if set(loaded_project_sources()) - {str(Path(p).resolve()) for p in ctx.config['source_hashes']} or \
            foreign_project_modules():
        raise ValueError('An unpinned or foreign project module was loaded during the stage')


# ---------------------------------------------------------------- lock, stage directories, writes

class heavy_lock:
    """Non-blocking exclusive hold on the existing shared lock; never creates it."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        env = os.environ.get('PITCHEEZY_HEAVY_LOCK')
        if env is not None and Path(env).resolve() != self.path.resolve():
            raise ValueError('PITCHEEZY_HEAVY_LOCK differs from the registered heavy lock')
        if not self.path.is_file():
            raise ValueError('Shared heavy lock file does not exist')
        self.stream = self.path.open('rb')
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            raise RuntimeError(f'Another heavy job holds {self.path}') from error
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        return False


def content_files(directory):
    """Relative files below ``directory`` excluding verified AppleDouble companions."""
    directory = Path(directory)
    if not directory.exists():
        return set()
    return {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and not is_appledouble(p)}


def stage_dir(output, stage, seed=None):
    if stage == 'predict':
        if seed not in mj.SEEDS:
            raise ValueError('predict requires --seed 0..4')
        return Path(output) / 'members' / f'seed{seed}'
    if seed is not None:
        raise ValueError('--seed is only valid for predict')
    return Path(output) / STAGE_DIRS[stage]


def claim_stage_dir(output, stage, seed=None):
    """A stage writes only into a directory it creates now; existing content is preserved, never reused."""
    output = Path(output)
    if stage == 'prepare':
        stale = [name for name in ('prepared', 'profile', 'members', 'fit', 'apply', 'analysis') if (output / name).exists()]
        if stale:
            raise ValueError(f'Family output already has stage content {stale}; register a fresh attempt')
    target = stage_dir(output, stage, seed)
    if target.exists() or target.is_symlink():
        raise ValueError(f'Stage output {target} exists; preserve it and register a fresh attempt')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def write_exclusive(path, writer):
    """Write a new file atomically (temp + rename) and refuse any overwrite; exFAT has no hard links."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    temporary = path.with_name(f'.{path.name}.partial-{os.getpid()}')
    with temporary.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    os.rename(temporary, path)
    return hash_file(path)


def write_json_exclusive(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + '\n').encode()
    return write_exclusive(path, lambda stream: stream.write(data))


def write_npz_exclusive(path, arrays):
    return write_exclusive(path, lambda stream: np.savez(stream, **arrays))


def write_pickle_exclusive(path, value):
    return write_exclusive(path, lambda stream: pickle.dump(value, stream, protocol=5))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def load_npz(path, names):
    """Decode only the named arrays (labels stay unopened unless requested); never pickle."""
    with np.load(Path(path), allow_pickle=False) as saved:
        missing = set(names) - set(saved.files)
        if missing:
            raise ValueError(f'{path}: missing arrays {sorted(missing)}')
        return {name: saved[name].copy() for name in names}


def load_pickle_verified(path, digest, label):
    data = Path(path).read_bytes()
    if sha_bytes(data) != digest:
        raise ValueError(f'{label} bytes differ from their pin')
    return pickle.loads(data)


def write_manifest(ctx, directory, stage, seed, extra):
    directory = Path(directory)
    names = sorted(content_files(directory) - {'manifest.json'})
    if 'failure.json' in names or any(Path(n).name.startswith('.') for n in names):
        raise ValueError('Stage directory holds failure or partial files; no success manifest')
    manifest = {'family_id': mj.FAMILY_ID, 'stage': stage, 'seed': seed, 'status': 'complete',
                'config_sha256': ctx.config_sha256, 'config_path': str(ctx.config_path),
                'scientific_config_sha256': ctx.config['scientific_config']['sha256'],
                'contract_sha256': ctx.config['contract']['sha256'],
                'registered_code_commit': ctx.code_commit,
                'execution_head': ctx.provenance.get('execution_head'),
                'outputs': {name: hash_file(directory / name) for name in names},
                'completed_utc': utc(), **extra}
    return write_json_exclusive(directory / 'manifest.json', manifest)


def verify_manifest(ctx, stage, seed=None):
    """A completed upstream stage: same config/commit, exact file set, every output hash."""
    directory = stage_dir(ctx.output, stage, seed)
    path = directory / 'manifest.json'
    if not path.is_file():
        raise ValueError(f'Completed {stage}{"" if seed is None else f" seed{seed}"} manifest required first')
    manifest = read_json(path)
    expected = {'family_id': mj.FAMILY_ID, 'stage': stage, 'seed': seed, 'status': 'complete',
                'config_sha256': ctx.config_sha256, 'registered_code_commit': ctx.code_commit}
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f'{stage} manifest field {name} differs from this registration')
    if set(manifest['outputs']) != content_files(directory) - {'manifest.json'}:
        raise ValueError(f'{stage} output inventory differs from its manifest')
    for name, digest in manifest['outputs'].items():
        if hash_file(directory / name) != digest:
            raise ValueError(f'{stage} output changed: {name}')
    return manifest, hash_file(path)


# ---------------------------------------------------------------- June store and frozen components

def build_store_frame(raw, date_max):
    """Regular-season rows through ``date_max`` only, then the frozen G0 point-in-time feature path.

    Rows are chronological, so this is the exact prefix the full-season frame had;
    prior-date style aggregates and same-PA histories of June rows are unchanged
    and no later observation can enter any June feature.
    """
    regular = raw.loc[raw.game_type.eq('R')]
    dates = pd.to_datetime(regular.game_date)
    frame = regular.loc[dates.le(pd.Timestamp(date_max))].copy().reset_index(drop=True)
    return assign_fold(add_batter_style_history(frame), 2025)


def store_digest(store):
    digest = hashlib.sha256()
    for name, array in (('keys', store.frame[KEY].to_numpy(np.int64)), ('physical', store.physical),
                        ('indices', store.indices), ('types', store.type_channels), ('outcomes', store.outcome_channels)):
        array = np.ascontiguousarray(array)
        digest.update(f'{name}:{array.dtype.str}:{array.shape}'.encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(store.report(), sort_keys=True).encode())
    return digest.hexdigest()


def delivery_digest(delivery):
    digest = hashlib.sha256(f'draws:{delivery.draws}'.encode())
    for key in sorted(delivery.pools, key=repr):
        pool = np.ascontiguousarray(delivery.pools[key])
        digest.update(repr(key).encode())
        digest.update(f'{pool.dtype.str}:{pool.shape}'.encode())
        digest.update(pool.tobytes())
    digest.update(np.ascontiguousarray(delivery.fallback).tobytes())
    return digest.hexdigest()


def compact_store(store, query_positions):
    """Closed June PA-prefix store: every query row plus all earlier pitches of its PA.

    History references stay inside the subset (an earlier pitch's history is an
    even earlier pitch of the same PA), so each query's tokens, type/outcome
    channels, frame row and prior-date features are exactly those of the full
    store; only rows no June query can ever reference are omitted.
    """
    frame, n = store.frame, len(store.frame)
    game, pa = frame.game_pk.to_numpy(), frame.at_bat_number.to_numpy()
    first = np.r_[True, (game[1:] != game[:-1]) | (pa[1:] != pa[:-1])]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    queries = np.asarray(query_positions, dtype=np.int64)
    marks = np.zeros(n + 1, dtype=np.int64)
    np.add.at(marks, start[queries], 1)
    np.add.at(marks, queries + 1, -1)
    needed = np.flatnonzero(np.cumsum(marks[:-1]) > 0)
    remap = np.full(n, -1, dtype=np.int64)
    remap[needed] = np.arange(len(needed))
    old = store.indices[needed]
    new = np.where(old >= 0, remap[np.maximum(old, 0)], -1)
    if ((old >= 0) & (new < 0)).any() or (remap[queries] < 0).any():
        raise ValueError('Compact store is not closed under same-PA history')
    base = HistoryStore(frame.iloc[needed].reset_index(drop=True), store.physical[needed],
                        np.ascontiguousarray(new.astype(store.indices.dtype)), store.normalizer)
    return MatrixHistoryStore(base, store.type_vocabulary), remap[queries], needed


def require_same_inputs(full, compact, full_rows, compact_rows, context, delivery, chunk=8192):
    """Exact equality of every June query's production inputs between the full and compact stores."""
    for begin in range(0, len(full_rows), chunk):
        rows, mine = full_rows[begin:begin + chunk], compact_rows[begin:begin + chunk]
        left, right = full.gather(rows), compact.gather(mine)
        if not (np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1])):
            raise ValueError('Compact store history tokens differ from the full store')
        a, b = full.frame.iloc[rows], compact.frame.iloc[mine]
        if not np.array_equal(context.transform(a), context.transform(b)):
            raise ValueError('Compact store context differs from the full store')
        if not np.array_equal(delivery.sample(a)[1], delivery.sample(b)[1]):
            raise ValueError('Compact store delivery tiers differ from the full store')
    return {'rows_checked': int(len(full_rows)), 'tokens_mask_context_tiers_equal': True}


def load_aux(ctx):
    aux = load_pickle_verified(ctx.parents['p4_auxiliary'], ctx.sci['parents']['p4_auxiliary']['sha256'], 'P4 aux.pkl')
    if aux['delivery'].draws != mj.DRAWS:
        raise ValueError('Frozen delivery draw count differs from 400')
    return aux


def build_store(frame, aux, tokens):
    store = MatrixHistoryStore.from_frame(frame, normalizer=aux['normalizer'], history_length=tokens['history_length'],
                                          type_vocabulary=tokens['type_vocabulary'])
    if store.report() != tokens or tokens['history_length'] != 5:
        raise ValueError('Frozen H5 feature contract changed')
    return store


def frozen_features(ctx):
    p4 = read_json(ctx.parents['p4_preparation'])
    p11 = read_json(ctx.parents['p11_preparation'])
    if canonical_hash(p4['clusters']) != canonical_hash(p11['clusters']) or \
            p4['features']['tokens'] != p11['features']['tokens']:
        raise ValueError('P4 and P11 frozen feature/cluster identities differ')
    return p4, p4['features']['tokens'], p4['clusters']


def load_member(ctx, seed, clusters):
    record = ctx.members[seed]
    check_hash(record['model_path'], record['model_sha256'], f'seed{seed} model')
    model = MatrixModel.load(Path(record['model_path']), device=None)
    if (model.kind, model.seed, model.width, model.n_classes) != (NETWORK['kind'], seed, NETWORK['width'], 10):
        raise ValueError('Loaded checkpoint identity differs from the member record')
    if model.report['network'] != record['network'] or model.report['parameter_count'] != record['parameter_count']:
        raise ValueError('Loaded checkpoint network differs from its fit report')
    if model.device != record['device']:
        raise ValueError(f'Device {model.device} differs from frozen member device {record["device"]}')
    predictor = SharingPredictor(CELL, model, clusters, individual_tau=NETWORK['individual_tau'],
                                 cluster_tau=NETWORK['cluster_tau'])
    predictor.delivery_temperature = float(record['delivery_temperature'])
    return predictor, model.device


def load_prepared(ctx):
    manifest, manifest_sha = verify_manifest(ctx, 'prepare')
    prepared = ctx.output / 'prepared'
    _, tokens, clusters = frozen_features(ctx)
    store = load_pickle_verified(prepared / 'june_store.pkl', manifest['outputs']['june_store.pkl'], 'sealed June store')
    aux = load_aux(ctx)
    if not isinstance(store, MatrixHistoryStore) or store.report() != tokens or \
            store.normalizer.report() != aux['normalizer'].report():
        raise ValueError('Sealed June store is not the frozen H5 feature contract')
    if store_digest(store) != manifest['store_digest'] or delivery_digest(aux['delivery']) != manifest['delivery_digest']:
        raise ValueError('Sealed feature/history store or delivery draws differ from the preparation')
    context = SharingContext(aux['context'], clusters)
    population = load_npz(prepared / 'june_population.npz',
                          ('june_keys', 'june_game_pk', 'june_pitcher', 'june_store_positions', 'june_groups',
                           'cpanel_keys', 'cpanel_june_index'))
    if mj.ordered_key_sha256(population['june_keys']) != mj.JUNE['ordered_key_sha256']:
        raise ValueError('Sealed June keys differ')
    positions = population['june_store_positions']
    if not np.array_equal(store.frame[KEY].to_numpy(np.int64)[positions], population['june_keys']):
        raise ValueError('Sealed June store positions differ from the keys')
    return {'manifest': manifest, 'manifest_sha256': manifest_sha, 'store': store, 'context': context, 'aux': aux,
            'population': population, 'clusters': clusters}


# ---------------------------------------------------------------- prepare

def stage_prepare(ctx, dest, seed=None):
    clock = time.perf_counter()
    P = ctx.parents
    p4, tokens, clusters = frozen_features(ctx)
    temperature = p4['baseline_temperature']['temperature']
    if not isinstance(temperature, float) or not .5 <= temperature <= 2.5:
        raise ValueError('Frozen frequency temperature invalid')
    lookup = mj.volume_lookup(read_json(P['p4_panel']))
    june_result = read_json(P['june_result'])
    eligible_counts = june_result['counts']['eligible']
    if (eligible_counts['pitches'], eligible_counts['games'], eligible_counts['pitchers']) != (
            mj.JUNE['rows'], mj.JUNE['games'], mj.JUNE['pitchers']):
        raise ValueError('D81 eligible census differs from registration')
    keys = pd.read_parquet(P['june_keys'])
    metadata = pd.read_parquet(P['june_metadata'])
    record = {'n': mj.JUNE['rows'], 'rows_sha256': mj.JUNE['ordered_key_sha256']}
    if ordered_key_hash(keys) != record['rows_sha256'] or not np.array_equal(
            metadata[KEY].to_numpy(np.int64), keys[KEY].to_numpy(np.int64)):
        raise ValueError('D81 eligible keys/metadata differ from the sealed ordered identity')
    parent = read_json(ctx.derived['p4_parent_preparation'])

    decoded_utc = utc()
    raw = load_verified_processed_cache(read_json(ctx.data['local_config']))
    if raw.attrs['sequence_data_identity'] != parent['dataset_identity'] or \
            raw.attrs['matrix_source_provenance'] != parent['source_provenance']:
        raise ValueError('Verified processed cache differs from the frozen P4 dataset identity')
    ctx.exposure.append({'event': 'processed_cache_decoded', 'utc': decoded_utc,
                         'files': {k: str(v) for k, v in ctx.data.items() if k != 'local_config'},
                         'decoded_scope': 'whole verified 2023-25 processed/physics cache via load_verified_processed_cache',
                         'retained_scope': f'regular season through {mj.JUNE["date_max"]}; later rows dropped before any feature',
                         'purpose': 'June labels, frozen frequency and complete point-in-time feature/history store'})
    date_max = pd.to_datetime(raw.game_date).max()
    frame = build_store_frame(raw, mj.JUNE['date_max'])
    del raw
    if pd.to_datetime(frame.game_date).max() > pd.Timestamp(mj.JUNE['date_max']):
        raise ValueError('Store frame contains rows after June 30')
    ctx.timings['load_and_features_internal'] = time.perf_counter() - clock
    part = select_keys(frame, keys, record)
    positions = part.index.to_numpy(np.int64)
    dates = pd.to_datetime(part.game_date)
    if not part.split.eq('blend').all() or not part.game_type.eq('R').all() or \
            not dates.between(mj.JUNE['date_min'], mj.JUNE['date_max']).all() or not np.asarray(eligible(part)).all():
        raise ValueError('Selected June rows are not the eligible regular-season June blend fold')
    june_keys = part[KEY].to_numpy(np.int64)
    games, pitchers = part.game_pk.to_numpy(np.int64), part.pitcher.to_numpy(np.int64)
    if (len(np.unique(games)), len(np.unique(pitchers))) != (mj.JUNE['games'], mj.JUNE['pitchers']):
        raise ValueError('June game/pitcher census differs')
    y = mj.validate_labels(outcome_labels(part), mj.JUNE['rows'], 'June labels')
    if not np.array_equal(metadata.pitcher.to_numpy(np.int64), pitchers) or \
            not np.array_equal(metadata.game_pk.to_numpy(np.int64), games) or \
            not np.array_equal(pd.to_datetime(metadata.game_date).dt.strftime('%Y-%m-%d').to_numpy(),
                               dates.dt.strftime('%Y-%m-%d').to_numpy()):
        raise ValueError('D81 metadata differs from the selected June rows')
    if not metadata.month.astype(str).eq('2025-06').all():
        raise ValueError('June metadata month differs')
    groups = mj.volume_groups(pitchers, lookup, metadata.train_volume.to_numpy())
    support = mj.group_support(groups, games)
    mj.require_expected_support(support)

    aux = load_aux(ctx)
    frequency_raw = np.asarray(aux['baseline'].predict(part), dtype=np.float64)
    frequency = np.asarray(temperature_predictions(frequency_raw, temperature), dtype=np.float64)
    mj.validate_simplex(frequency_raw, len(y), 'June raw frequency')
    mj.validate_simplex(frequency, len(y), 'June frequency')

    p10 = load_npz(P['p10_frequency'], ('blend', 'blend_raw', 'blend_keys', 'blend_y', 'blend_game_pk', 'blend_pitcher'))
    cpanel_keys = mj.key_array(p10['blend_keys'], 'Cpanel keys')
    if mj.ordered_key_sha256(cpanel_keys) != mj.CPANEL['ordered_key_sha256'] or \
            len(np.unique(cpanel_keys[:, 0])) != mj.CPANEL['games']:
        raise ValueError('Archived Cpanel keys differ from the frozen P4 blend sample')
    index = mj.subset_positions(june_keys, cpanel_keys, 'Cpanel keys')
    reports = [mj.compare_exact(y[index], p10['blend_y'], name='cpanel_labels'),
               mj.compare_exact(games[index], p10['blend_game_pk'], name='cpanel_game_pk'),
               mj.compare_exact(pitchers[index], p10['blend_pitcher'], name='cpanel_pitcher'),
               mj.compare_probabilities(frequency[index], p10['blend'], atol=mj.PROBABILITY_REPLAY_ATOL,
                                        name='cpanel_frequency'),
               mj.compare_probabilities(frequency_raw[index], p10['blend_raw'], atol=mj.PROBABILITY_REPLAY_ATOL,
                                        name='cpanel_frequency_raw')]
    frozen_mask = np.zeros(len(y), dtype=bool)
    frozen_mask[index] = True
    reports.append(mj.compare_exact(metadata.frozen_cpanel_eligible.to_numpy(bool), frozen_mask,
                                    name='cpanel_membership_metadata'))
    blend_meta = pd.read_parquet(ctx.derived['p4_blend_metadata'])
    if not np.array_equal(blend_meta[KEY].to_numpy(np.int64), cpanel_keys):
        raise ValueError('P4 blend metadata order differs from archived Cpanel keys')
    for column in BLEND_META_COMPARE:
        reports.append(mj.compare_exact(metadata[column].to_numpy()[index].astype(str),
                                        blend_meta[column].to_numpy().astype(str), name='cpanel_metadata_' + column))
    for s in mj.SEEDS:
        member = load_npz(P[f'cpanel_member_seed{s}'], ('blend_keys', 'blend_y'))
        reports += [mj.compare_exact(np.ascontiguousarray(member['blend_keys']), cpanel_keys, name=f'cpanel_seed{s}_keys'),
                    mj.compare_exact(member['blend_y'], p10['blend_y'], name=f'cpanel_seed{s}_labels')]
    mj.require_passed(reports, 'June preparation Cpanel replay')

    full_store = build_store(frame, aux, tokens)
    context = SharingContext(aux['context'], clusters)
    store, store_positions, _ = compact_store(full_store, positions)
    equivalence = require_same_inputs(full_store, store, positions, store_positions, context, aux['delivery'])
    probe = store_positions[:mj.PROBE_ROWS]
    features = context.transform(store.frame.iloc[probe])
    samples, levels = aux['delivery'].sample(store.frame.iloc[probe])
    tokens_probe, valid = store.gather(np.repeat(probe, mj.DRAWS), current=samples.reshape(-1, samples.shape[-1]))
    if not np.isfinite(features).all() or samples.shape != (len(probe), mj.DRAWS, 8) or not np.isfinite(tokens_probe).all():
        raise ValueError('Sealed store cannot produce the frozen production inputs')
    digest, draws_digest = store_digest(store), delivery_digest(aux['delivery'])
    full_rows = int(len(frame))
    del full_store, frame
    ctx.timings['prepare_before_write_internal'] = time.perf_counter() - clock

    store_dates = pd.to_datetime(store.frame.game_date)
    outputs = {}
    outputs['june_store.pkl'] = write_pickle_exclusive(dest / 'june_store.pkl', store)
    outputs['june_population.npz'] = write_npz_exclusive(dest / 'june_population.npz', {
        'june_keys': june_keys, 'june_game_pk': games, 'june_pitcher': pitchers, 'june_store_positions': store_positions,
        'june_groups': groups, 'cpanel_keys': cpanel_keys, 'cpanel_june_index': index})
    outputs['june_labels.npz'] = write_npz_exclusive(dest / 'june_labels.npz', {'june_keys': june_keys, 'june_y': y})
    outputs['june_frequency.npz'] = write_npz_exclusive(dest / 'june_frequency.npz', {
        'june_keys': june_keys, 'june_y': y, 'june_frequency': frequency, 'june_frequency_raw': frequency_raw})
    hashes = {'june_ordered_key_sha256': mj.ordered_key_sha256(june_keys), 'june_label_sha256': mj.array_sha256(y, np.int64),
              'june_frequency_sha256': mj.array_sha256(frequency, np.float64),
              'june_frequency_raw_sha256': mj.array_sha256(frequency_raw, np.float64),
              'june_groups_sha256': mj.array_sha256(groups, '<U6')}
    report = {'family_id': mj.FAMILY_ID, 'stage': 'prepare', 'rows': int(len(y)), 'games': mj.JUNE['games'],
              'pitchers': mj.JUNE['pitchers'], 'full_store_rows': full_rows, 'sealed_store_rows': int(len(store.frame)),
              'store_date_range': [str(store_dates.min().date()), str(store_dates.max().date())],
              'decoded_cache_date_max': str(date_max.date()),
              'store_policy': ('features built on all regular-season pitches through 2025-06-30 (unfiltered, frozen '
                               'normalizer/vocabulary/context/clusters, H5); sealed store keeps every June query row and '
                               'all earlier pitches of its PA, exactly equal inputs for all June queries'),
              'compact_store_equivalence': equivalence,
              'store_digest': digest, 'delivery_digest': draws_digest, 'draws': mj.DRAWS, 'tokens': tokens,
              'frequency_temperature': temperature, 'frequency_transform': 'softmax(log(clip(raw,1e-12,1))/T); no refit',
              'label_function': 'P4 pitchmdp.model.outcome_labels (hash-pinned)', 'support': support,
              'volume_thresholds': list(mj.VOLUME_THRESHOLDS), 'cpanel_reports': reports, 'hashes': hashes,
              'exposure': ctx.exposure, 'new_fits': 0, 'labels_sealed_before_fit': True,
              'timings_internal_nonadditive': ctx.timings,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    outputs['prepared.json'] = write_json_exclusive(dest / 'prepared.json', report)
    reverify_inputs(ctx)
    write_manifest(ctx, dest, 'prepare', None, {'store_digest': digest, 'delivery_digest': draws_digest, **hashes,
                                                'inputs': ctx.inputs, 'provenance': ctx.provenance})
    return {'rows': int(len(y)), 'sealed_store_rows': int(len(store.frame))}


# ---------------------------------------------------------------- profile and predict

def _tier_counts(levels):
    unique, counts = np.unique(levels, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def _validate_levels(levels, n):
    levels = np.asarray(levels)
    if levels.shape != (n,) or not np.issubdtype(levels.dtype, np.integer) or ((levels < -1) | (levels > 3)).any():
        raise ValueError('Delivery tier levels outside the frozen contract')
    return levels


def stage_profile(ctx, dest, seed=None):
    clock = time.perf_counter()
    loaded = load_prepared(ctx)
    load_seconds = time.perf_counter() - clock
    population, store, context, aux = loaded['population'], loaded['store'], loaded['context'], loaded['aux']
    predictor, device = load_member(ctx, 0, loaded['clusters'])
    probe_index = population['cpanel_june_index'][:mj.PROBE_ROWS]
    archived = load_npz(ctx.parents['cpanel_member_seed0'], ('blend_keys', 'blend', 'blend_raw', 'blend_delivery_level'))
    if not np.array_equal(np.ascontiguousarray(archived['blend_keys'][:mj.PROBE_ROWS]), population['june_keys'][probe_index]):
        raise ValueError('Probe keys differ from the first archived Cpanel keys')
    before = time.perf_counter()
    calibrated, raw, levels = predict_streamed(predictor, aux['delivery'], store, context,
                                               population['june_store_positions'][probe_index])
    probe_seconds = time.perf_counter() - before
    reports = [mj.compare_probabilities(calibrated, archived['blend'][:mj.PROBE_ROWS], atol=mj.PROBABILITY_REPLAY_ATOL,
                                        name='probe_calibrated'),
               mj.compare_probabilities(raw, archived['blend_raw'][:mj.PROBE_ROWS], atol=mj.PROBABILITY_REPLAY_ATOL,
                                        name='probe_raw'),
               mj.compare_exact(levels, archived['blend_delivery_level'][:mj.PROBE_ROWS], name='probe_delivery_level')]
    rows = population['june_store_positions'][:mj.PROFILE_ROWS]
    before = time.perf_counter()
    p, p_raw, tiers = predict_streamed(predictor, aux['delivery'], store, context, rows)
    inference_seconds = time.perf_counter() - before
    mj.validate_simplex(p, len(rows), 'profile calibrated')
    mj.validate_simplex(p_raw, len(rows), 'profile raw')
    _validate_levels(tiers, len(rows))
    result = {'family_id': mj.FAMILY_ID, 'stage': 'profile', 'seed': 0, 'device': device, 'draws': mj.DRAWS,
              'rows': int(len(rows)), 'row_selection': 'first frozen eligible June keys in sealed order',
              'probe_rows': int(len(probe_index)), 'probe_selection': 'first archived Cpanel keys',
              'probe_reports': reports, 'delivery_tier_counts': _tier_counts(tiers),
              'maximum_mass_error': float(np.abs(p.sum(1) - 1).max()),
              'same_store_as_full_member': loaded['manifest']['store_digest'],
              'profile_probabilities_discarded': True, 'labels_read': False, 'quality_computed': False,
              'profiled_utc': utc(),
              'internal_seconds_diagnostic_only': {'load': load_seconds, 'probe': probe_seconds,
                                                   'inference': inference_seconds, **ctx.timings},
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    del p, p_raw
    write_json_exclusive(dest / 'profile.json', result)
    mj.require_passed(reports, 'Profile Cpanel identity probe')
    reverify_inputs(ctx, stage_inputs(ctx, 0))
    write_manifest(ctx, dest, 'profile', None, {'prepared_manifest_sha256': loaded['manifest_sha256'],
                                                'store_digest': loaded['manifest']['store_digest'],
                                                'rows': int(len(rows)), 'probe_rows': int(len(probe_index))})
    return {'rows': int(len(rows))}


def stage_predict(ctx, dest, seed):
    clock = time.perf_counter()
    profile, profile_sha = verify_manifest(ctx, 'profile')
    loaded = load_prepared(ctx)
    population, store, context, aux = loaded['population'], loaded['store'], loaded['context'], loaded['aux']
    labels = load_npz(ctx.output / 'prepared' / 'june_labels.npz', ('june_keys', 'june_y'))
    if not np.array_equal(labels['june_keys'], population['june_keys']):
        raise ValueError('Sealed June labels are not aligned with the population')
    predictor, device = load_member(ctx, seed, loaded['clusters'])
    before, generated_utc = time.perf_counter(), utc()
    calibrated, raw, levels = predict_streamed(predictor, aux['delivery'], store, context,
                                               population['june_store_positions'])
    inference_seconds = time.perf_counter() - before
    n = len(population['june_keys'])
    mj.validate_simplex(calibrated, n, f'seed{seed} calibrated')
    mj.validate_simplex(raw, n, f'seed{seed} raw')
    levels = _validate_levels(levels, n)
    values = {'seed': np.asarray(seed, dtype=np.int64), 'june_keys': population['june_keys'], 'june_y': labels['june_y'],
              'june_game_pk': population['june_game_pk'], 'june_pitcher': population['june_pitcher'],
              'june_calibrated': calibrated, 'june_raw': raw, 'june_delivery_level': levels}
    write_npz_exclusive(dest / 'predictions.npz', values)
    index = population['cpanel_june_index']
    archived = load_npz(ctx.parents[f'cpanel_member_seed{seed}'],
                        ('blend_keys', 'blend_y', 'blend', 'blend_raw', 'blend_delivery_level'))
    reports = [mj.compare_exact(population['june_keys'][index], np.ascontiguousarray(archived['blend_keys']), name='cpanel_keys'),
               mj.compare_exact(labels['june_y'][index], archived['blend_y'], name='cpanel_labels'),
               mj.compare_probabilities(calibrated[index], archived['blend'], atol=mj.PROBABILITY_REPLAY_ATOL,
                                        name='cpanel_calibrated'),
               mj.compare_probabilities(raw[index], archived['blend_raw'], atol=mj.PROBABILITY_REPLAY_ATOL, name='cpanel_raw'),
               mj.compare_exact(levels[index], archived['blend_delivery_level'], name='cpanel_delivery_level')]
    runtime = {'family_id': mj.FAMILY_ID, 'stage': 'predict', 'seed': seed, 'source_parent': mj.SEED_PARENT[seed],
               'device': device, 'rows': int(n), 'draws': mj.DRAWS,
               'delivery_temperature': ctx.members[seed]['delivery_temperature'],
               'model_sha256': ctx.members[seed]['model_sha256'], 'cpanel_replay': reports,
               'delivery_tier_counts': _tier_counts(levels), 'same_store_as_profile': loaded['manifest']['store_digest'],
               'profile_manifest_sha256': profile_sha, 'quality_computed': False,
               'labels_embedded_from_sealed_prepare': True, 'probabilities_generated_utc': generated_utc,
               'purpose': 'full frozen June member for B1/B2 calibration fits; no quality computed',
               'internal_seconds_diagnostic_only': {'inference': inference_seconds,
                                                    'total': time.perf_counter() - clock, **ctx.timings},
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json_exclusive(dest / 'prediction_runtime.json', runtime)
    mj.require_passed(reports, f'Seed {seed} full Cpanel replay')
    reverify_inputs(ctx, stage_inputs(ctx, seed))
    write_manifest(ctx, dest, 'predict', seed, {'prepared_manifest_sha256': loaded['manifest_sha256'],
                                                'profile_manifest_sha256': profile_sha,
                                                'store_digest': loaded['manifest']['store_digest'],
                                                'model_sha256': ctx.members[seed]['model_sha256'], 'rows': int(n)})
    return {'rows': int(n)}


# ---------------------------------------------------------------- fit

def load_members(ctx, population, labels):
    arrays, manifests = [], {}
    for seed in mj.SEEDS:
        _, digest = verify_manifest(ctx, 'predict', seed)
        manifests[f'members/seed{seed}/manifest.json'] = digest
        item = load_npz(stage_dir(ctx.output, 'predict', seed) / 'predictions.npz', MEMBER_FIELDS)
        if int(item['seed']) != seed:
            raise ValueError(f'Member archive seed order differs at position {seed}')
        for name, reference in (('june_keys', population['june_keys']), ('june_y', labels['june_y']),
                                ('june_game_pk', population['june_game_pk']), ('june_pitcher', population['june_pitcher'])):
            if not np.array_equal(item[name], reference):
                raise ValueError(f'Seed {seed} member rows differ from the sealed June identity: {name}')
        arrays.append(mj.validate_simplex(item['june_calibrated'], len(labels['june_y']), f'seed{seed} calibrated'))
    return np.stack(arrays), manifests


def dev_separation(ctx, population, june_dates):
    """No DEV key, game or date may enter a June fit input."""
    dev = load_npz(ctx.parents['p10_frequency'], ('mlb_dev_keys', 'mlb_dev_game_pk'))
    dev_keys = mj.key_array(dev['mlb_dev_keys'], 'DEV keys')
    june = mj.key_array(population['june_keys'], 'June keys')
    both = np.concatenate([june, dev_keys])
    if len(np.unique(both, axis=0)) != len(both):
        raise ValueError('A DEV pitch key is present in the June fit population')
    if np.intersect1d(population['june_game_pk'], dev['mlb_dev_game_pk']).size:
        raise ValueError('A DEV game is present in the June fit population')
    months = pd.read_parquet(ctx.derived['p11_dev_metadata'], columns=['month']).month.astype(str)
    dates = pd.to_datetime(june_dates)
    if not set(months.unique()) <= set(mj.DEV['months']) or dates.min() < pd.Timestamp(mj.JUNE['date_min']) or \
            dates.max() > pd.Timestamp(mj.JUNE['date_max']):
        raise ValueError('June fit dates and DEV dates are not separated')
    return {'dev_keys_in_fit': 0, 'dev_games_in_fit': 0, 'june_date_range': [str(dates.min().date()), str(dates.max().date())],
            'dev_months': sorted(months.unique().tolist())}


def replay_inputs(ctx):
    P = ctx.parents
    ji = load_npz(P['p11_cpanel_june_inputs'], ('keys', 'y', 'game_pk', 'pitcher', 'frequency', 'calibrated',
                                                'seed_calibrated', 'primary', 'seed_primary', 'weights'))
    p10 = load_npz(P['p10_frequency'], ('blend', 'blend_keys', 'blend_y'))
    if not np.array_equal(ji['keys'], np.ascontiguousarray(p10['blend_keys'])) or not np.array_equal(ji['y'], p10['blend_y']) \
            or not np.array_equal(ji['frequency'], p10['blend']):
        raise ValueError('Archived Cpanel fit inputs differ from the canonical P10 frequency archive')
    for s in mj.SEEDS:
        member = load_npz(P[f'cpanel_member_seed{s}'], ('blend',))
        if not np.array_equal(ji['seed_calibrated'][s], member['blend']):
            raise ValueError(f'Archived Cpanel seed{s} array differs from its pinned member archive')
    results = read_json(P['p10_results'])['reports'][CELL]
    temperatures = {str(s): ctx.members[s]['delivery_temperature'] for s in mj.SEEDS}
    frozen = frozen_weights(results, temperatures)
    weights = {'ensemble': frozen['june_ensemble_model_weight'],
               **{f'seed{s}': frozen['june_seed_model_weights'][str(s)] for s in mj.SEEDS}}
    p11 = read_json(P['p11_preparation'])['frozen_calibration']
    if list(weights.values()) != [float(w) for w in ji['weights']] or \
            p11['june_ensemble_model_weight'] != weights['ensemble'] or \
            any(p11['june_seed_model_weights'][str(s)] != weights[f'seed{s}'] for s in mj.SEEDS):
        raise ValueError('Archived five-seed C1 weights disagree across P10/P11 records')
    return ji, weights


def stage_fit(ctx, dest, seed=None, minimizer=None):
    clock = time.perf_counter()
    prepared, prepared_sha = verify_manifest(ctx, 'prepare')
    root = ctx.output / 'prepared'
    population = load_npz(root / 'june_population.npz', ('june_keys', 'june_game_pk', 'june_pitcher', 'june_groups'))
    labels = load_npz(root / 'june_labels.npz', ('june_keys', 'june_y'))
    frequency = load_npz(root / 'june_frequency.npz', ('june_keys', 'june_y', 'june_frequency'))
    if not np.array_equal(labels['june_keys'], population['june_keys']) or \
            not np.array_equal(frequency['june_keys'], population['june_keys']) or \
            not np.array_equal(frequency['june_y'], labels['june_y']) or \
            mj.array_sha256(labels['june_y'], np.int64) != prepared['june_label_sha256'] or \
            mj.array_sha256(frequency['june_frequency'], np.float64) != prepared['june_frequency_sha256']:
        raise ValueError('Sealed June labels/frequency differ from the preparation manifest')
    seed_calibrated, member_manifests = load_members(ctx, population, labels)
    june_dates = pd.read_parquet(ctx.parents['june_metadata'], columns=['game_date']).game_date
    separation = dev_separation(ctx, population, june_dates)
    ji, b0 = replay_inputs(ctx)
    ctx.exposure.append({'event': 'june_labels_loaded_for_fit', 'utc': utc(), 'rows': int(len(labels['june_y'])),
                         'purpose': 'scalar blend fits (in-sample optimizer audit only)'})
    ledger = mj.CallLedger()
    ctx.fit_ledger = ledger  # preserved in failure.json even on termination mid-fit
    ctx.timings['fit_inputs_internal'] = time.perf_counter() - clock
    try:
        record = mj.fit_family(june={'y': labels['june_y'], 'frequency': frequency['june_frequency'],
                                     'seed_calibrated': seed_calibrated, 'groups': population['june_groups'],
                                     'game_pk': population['june_game_pk']},
                               replay=ji, b0_weights=b0, ledger=ledger,
                               **({} if minimizer is None else {'minimizer': minimizer}))
        mj.require_expected_support(record['B2_support'])
        calls = mj.require_expected_calls(record)
        if calls != mj.EXPECTED_CALLS:
            raise mj.FamilyStop(f'Optimizer calls {calls} differ from the registered 6 + 30', ledger.snapshot())
    finally:
        ctx.timings['fit_internal'] = time.perf_counter() - clock
    parameters = {**record, 'family_id': mj.FAMILY_ID, 'stage': 'fit', 'call_counts': calls,
                  'dev_separation': separation, 'dev_scores': None,
                  'inputs': {'june_ordered_key_sha256': prepared['june_ordered_key_sha256'],
                             'june_label_sha256': prepared['june_label_sha256'],
                             'june_frequency_sha256': prepared['june_frequency_sha256'],
                             'prepared_manifest_sha256': prepared_sha, 'member_manifests': member_manifests,
                             'ensemble_sha256': mj.array_sha256(mj.ensemble_mean(seed_calibrated), np.float64),
                             'replay_inputs_sha256': ctx.sci['parents']['p11_cpanel_june_inputs']['sha256'],
                             'b0_weights_source_sha256': ctx.sci['parents']['p10_results']['sha256']},
                  'optimizer_source': ctx.sci['optimizer']['legacy_source'],
                  'numerical_environment': {'numpy': np.__version__, 'scipy': scipy.__version__},
                  'exposure': ctx.exposure, 'timings_internal_nonadditive': ctx.timings}
    write_json_exclusive(dest / 'parameters.json', parameters)
    reverify_inputs(ctx)
    write_manifest(ctx, dest, 'fit', None, {'prepared_manifest_sha256': prepared_sha, 'member_manifests': member_manifests,
                                            'call_counts': calls, 'B0_weights': record['B0_weights'],
                                            'B1_weights': record['B1_weights'], 'B2_weights': record['B2_weights'],
                                            'dev_scores': None, 'sealed_before_dev_labels': True})
    return {'calls': calls}


# ---------------------------------------------------------------- apply (label-free)

def dev_components(ctx):
    """Frozen P11/P10 DEV components; labels are never decoded here."""
    P = ctx.parents
    members = []
    for s in mj.SEEDS:
        item = load_npz(P[f'dev_member_seed{s}'], ('dev_keys', 'dev_game_pk', 'dev_pitcher', 'dev'))
        members.append(item)
    frequency = load_npz(P['p10_frequency'], ('mlb_dev', 'mlb_dev_keys', 'mlb_dev_game_pk', 'mlb_dev_pitcher'))
    archived = load_npz(P['p11_predictions'], ('keys', 'game_pk', 'pitcher', 'calibrated', 'primary', 'seed_primary'))
    metadata = pd.read_parquet(ctx.derived['p11_dev_metadata'])
    keys = mj.key_array(archived['keys'], 'P11 DEV keys')
    n = len(keys)
    if n != mj.DEV['rows'] or len(np.unique(archived['game_pk'])) != mj.DEV['games']:
        raise ValueError('DEV population differs from registration')
    for name, values in (('P10 keys', frequency['mlb_dev_keys']), ('metadata keys', metadata[KEY].to_numpy(np.int64)),
                         *((f'seed{s} keys', m['dev_keys']) for s, m in zip(mj.SEEDS, members))):
        if not np.array_equal(np.ascontiguousarray(values), keys):
            raise ValueError(f'DEV row order differs: {name}')
    for name, values in (('P10 games', frequency['mlb_dev_game_pk']), ('P10 pitchers', frequency['mlb_dev_pitcher']),
                         *((f'seed{s} games', m['dev_game_pk']) for s, m in zip(mj.SEEDS, members)),
                         *((f'seed{s} pitchers', m['dev_pitcher']) for s, m in zip(mj.SEEDS, members))):
        reference = archived['game_pk'] if 'games' in name else archived['pitcher']
        if not np.array_equal(values, reference):
            raise ValueError(f'DEV identity differs: {name}')
    if not np.array_equal(keys[:, 0], archived['game_pk']) or \
            not np.array_equal(metadata.pitcher.to_numpy(np.int64), archived['pitcher']):
        raise ValueError('DEV key/game/pitcher metadata differ')
    lookup = mj.volume_lookup(read_json(P['p4_panel']))
    groups = mj.volume_groups(archived['pitcher'], lookup, metadata.train_volume.to_numpy())
    seeds = np.stack([mj.validate_simplex(m['dev'], n, f'DEV seed{s}') for s, m in zip(mj.SEEDS, members)])
    return {'keys': keys, 'game_pk': archived['game_pk'], 'pitcher': archived['pitcher'], 'groups': groups,
            'seed_calibrated': seeds, 'frequency': mj.validate_simplex(frequency['mlb_dev'], n, 'DEV frequency'),
            'archived': archived}


def stage_apply(ctx, dest, seed=None):
    clock = time.perf_counter()
    prepared, prepared_sha = verify_manifest(ctx, 'prepare')
    members = {f'members/seed{s}/manifest.json': verify_manifest(ctx, 'predict', s)[1] for s in mj.SEEDS}
    fit_manifest, fit_sha = verify_manifest(ctx, 'fit')
    parameters = read_json(ctx.output / 'fit' / 'parameters.json')
    if parameters['predictors'] != list(mj.PREDICTORS) or parameters['call_counts'] != mj.EXPECTED_CALLS or \
            parameters['B1_weights'] != fit_manifest['B1_weights'] or parameters['B2_weights'] != fit_manifest['B2_weights']:
        raise ValueError('Sealed fit parameters are incomplete or differ from the fit manifest')
    dev = dev_components(ctx)
    out = mj.apply_family(parameters, seed_calibrated=dev['seed_calibrated'], frequency=dev['frequency'],
                          groups=dev['groups'])
    archived = dev['archived']
    reports = [mj.compare_probabilities(out['ensemble_calibrated'], archived['calibrated'], atol=mj.PROBABILITY_REPLAY_ATOL,
                                        name='dev_ensemble_calibrated'),
               mj.compare_probabilities(out['B0_primary'], archived['primary'], atol=mj.PROBABILITY_REPLAY_ATOL,
                                        name='B0_dev_primary')]
    for s in mj.SEEDS:
        reports.append(mj.compare_probabilities(out['B0_seed_primary'][s], archived['seed_primary'][s],
                                                atol=mj.PROBABILITY_REPLAY_ATOL, name=f'B0_dev_seed{s}'))
    arrays = {'keys': dev['keys'], 'game_pk': dev['game_pk'], 'pitcher': dev['pitcher'], 'groups': dev['groups'],
              'predictors': np.asarray(mj.PREDICTORS)}
    arrays.update({name: out[name] for name in APPLY_FIELDS if name in out})
    application = {'family_id': mj.FAMILY_ID, 'stage': 'apply', 'rows': int(len(dev['keys'])),
                   'b0_replay': reports, 'fit_manifest_sha256': fit_sha, 'prepared_manifest_sha256': prepared_sha,
                   'member_manifests': members,
                   'parameters_sha256': fit_manifest['outputs']['parameters.json'],
                   'label_archive_reference': {'p11_predictions_sha256': ctx.sci['parents']['p11_predictions']['sha256'],
                                               'p10_frequency_sha256': ctx.sci['parents']['p10_frequency']['sha256'],
                                               'labels_decoded': False},
                   'dev_group_counts': {g: int((dev['groups'] == g).sum()) for g in mj.GROUPS},
                   'timings_internal_nonadditive': {'apply': time.perf_counter() - clock, **ctx.timings}}
    write_json_exclusive(dest / 'application.json', application)
    mj.require_passed(reports, 'B0 DEV replay against P11 primary/seed_primary')
    write_npz_exclusive(dest / 'predictions.npz', arrays)
    reverify_inputs(ctx)
    write_manifest(ctx, dest, 'apply', None, {'fit_manifest_sha256': fit_sha, 'prepared_manifest_sha256': prepared_sha,
                                              'member_manifests': members, 'candidates': ['B1', 'B2'],
                                              'seeds': list(mj.SEEDS), 'labels_decoded': False})
    return {'rows': int(len(dev['keys']))}


# ---------------------------------------------------------------- score

def stage_score(ctx, dest, seed=None):
    clock = time.perf_counter()
    upstream = {'prepared/manifest.json': verify_manifest(ctx, 'prepare')[1],
                'profile/manifest.json': verify_manifest(ctx, 'profile')[1],
                **{f'members/seed{s}/manifest.json': verify_manifest(ctx, 'predict', s)[1] for s in mj.SEEDS},
                'fit/manifest.json': verify_manifest(ctx, 'fit')[1]}
    apply_manifest, apply_sha = verify_manifest(ctx, 'apply')
    upstream['apply/manifest.json'] = apply_sha
    if apply_manifest['fit_manifest_sha256'] != upstream['fit/manifest.json'] or \
            apply_manifest['candidates'] != ['B1', 'B2'] or apply_manifest['seeds'] != list(mj.SEEDS):
        raise ValueError('Application is incomplete or belongs to another fit')
    applied = load_npz(ctx.output / 'apply' / 'predictions.npz', APPLY_FIELDS)
    if list(applied['predictors']) != list(mj.PREDICTORS):
        raise ValueError('Applied predictor order differs')
    decoded = utc()
    labels = load_npz(ctx.parents['p11_predictions'], ('keys', 'y', 'game_pk', 'pitcher'))
    frequency = load_npz(ctx.parents['p10_frequency'], ('mlb_dev_keys', 'mlb_dev_y', 'mlb_dev_game_pk', 'mlb_dev_pitcher',
                                                        'dev_keys', 'dev_y', 'dev_game_pk', 'dev_pitcher'))
    ctx.exposure.append({'event': 'dev_labels_decoded_for_scoring', 'utc': decoded,
                         'source': [str(ctx.parents['p11_predictions']), str(ctx.parents['p10_frequency'])],
                         'purpose': 'fixed N3/R78 and descriptive scoring after both candidates and five seeds sealed'})
    for name in ('keys', 'game_pk', 'pitcher'):
        if not np.array_equal(np.ascontiguousarray(labels[name]), applied[name]):
            raise ValueError(f'Applied DEV rows differ from the P11 archive: {name}')
    if not np.array_equal(labels['y'], frequency['mlb_dev_y']):
        raise ValueError('DEV labels differ between P11 and P10 archives')
    metadata = pd.read_parquet(ctx.derived['p11_dev_metadata'])
    base = {'dev_keys': np.ascontiguousarray(frequency['mlb_dev_keys']), 'dev_y': frequency['mlb_dev_y'],
            'dev_game_pk': frequency['mlb_dev_game_pk'], 'dev_pitcher': frequency['mlb_dev_pitcher']}
    panel = {'dev_keys': np.ascontiguousarray(frequency['dev_keys']), 'dev_y': frequency['dev_y'],
             'dev_game_pk': frequency['dev_game_pk'], 'dev_pitcher': frequency['dev_pitcher']}
    population, _, overlap = aligned_population(base, metadata, panel)
    if (overlap['overlap_pitches'], overlap['overlap_games'], overlap['complement_pitches']) != (
            mj.DEV['cpanel_rows'], mj.DEV['cpanel_games'], mj.DEV['complement_rows']):
        raise ValueError('DEV Cpanel/complement identity differs from registration')
    predictions = {v: {'primary': applied[v + '_primary'], 'seed_primary': applied[v + '_seed_primary']}
                   for v in mj.VARIANTS}
    result = mm.score_family(y=labels['y'], games=labels['game_pk'], pitchers=labels['pitcher'], metadata=metadata,
                             cpanel_mask=population['cpanel'], predictions=predictions)
    fit_manifest = read_json(ctx.output / 'fit' / 'manifest.json')
    result.update(overlap=overlap, upstream_manifests=upstream, weights={v: fit_manifest[v + '_weights'] for v in mj.VARIANTS},
                  exposure=ctx.exposure, scored_utc=utc(),
                  timings_internal_nonadditive={'score': time.perf_counter() - clock, **ctx.timings})
    write_json_exclusive(dest / 'results.json', result)
    reverify_inputs(ctx)
    write_manifest(ctx, dest, 'score', None, {
        'upstream_manifests': upstream, 'contrast_additivity_passed': result['contrast_additivity_passed'],
        'N_decisions': {cid: result['N'][cid]['decision']['status'] for cid in mm.CONTRAST_IDS},
        'R_status': result['R']['contrast_status'],
        'candidates': {c: result['candidates'][c]['research_improvement_candidate'] for c in ('B1', 'B2')},
        'R_family_size': result['R']['family_size'], 'N_family_size': 3, **mj.RESEARCH_STATUS})
    return {'contrast_additivity_passed': result['contrast_additivity_passed']}


STAGE_FUNCTIONS = {'prepare': stage_prepare, 'profile': stage_profile, 'predict': stage_predict, 'fit': stage_fit,
                   'apply': stage_apply, 'score': stage_score}


# ---------------------------------------------------------------- failure records and entry point

def failure_record(stage, seed, error, ctx, dest):
    record = {'family_id': mj.FAMILY_ID, 'stage': stage, 'seed': seed, 'status': 'failed', 'failed_utc': utc(),
              'error_type': type(error).__name__, 'message': str(error)[:4000],
              'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__))[-12000:],
              'stage_directory': str(dest) if dest else None, 'retry': False, 'partial_outputs_preserved': True}
    if isinstance(error, mj.FamilyStop):
        record['diagnostics'] = error.diagnostics
    if ctx is not None:
        if ctx.fit_ledger is not None:
            record['optimizer_call_ledger'] = ctx.fit_ledger.snapshot()
        record.update(config_sha256=ctx.config_sha256, registered_code_commit=ctx.code_commit,
                      execution_head=ctx.provenance.get('execution_head'), exposure=ctx.exposure,
                      timings_internal_nonadditive=ctx.timings)
    return record


def write_failure(output, stage, seed, error, ctx, dest):
    record = failure_record(stage, seed, error, ctx, dest)
    try:
        if dest is not None and Path(dest).is_dir():
            target = Path(dest) / 'failure.json'
        else:
            folder = Path(output) / 'failures'
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f'{stage}{"" if seed is None else f"-seed{seed}"}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{os.getpid()}.json'
        write_json_exclusive(target, record)
        return target
    except BaseException as nested:  # the original failure stays authoritative
        print(f'FAILURE_RECORD_NOT_WRITTEN {type(nested).__name__}: {nested}', file=sys.stderr, flush=True)
        return None


def run(config_path, output, stage, seed=None, *, git=True):
    config_path = Path(config_path)
    config = validate_execution_config(json.loads(config_path.read_bytes()))
    if stage not in mj.STAGES:
        raise ValueError('Unknown stage')
    if (stage == 'predict') != (seed is not None):
        raise ValueError('--seed is required for predict and forbidden otherwise')
    output = Path(output)
    if not output.is_absolute() or output.resolve() != Path(config['output_dir']).resolve():
        raise ValueError('--output-dir must be the registered absolute output_dir')
    with heavy_lock(config['heavy_lock']):
        ctx, dest = None, None
        try:
            ctx = verify_everything(config, config_path, output, git=git)
            output = ctx.output
            if stage == 'prepare':
                family_root = Path(ctx.sci['future_output_root']).resolve()
                if not family_root.parent.is_dir():
                    raise ValueError('Registered run root for the family output does not exist')
                output.mkdir(parents=True, exist_ok=True)  # output is the family root or inside it
            elif not (output / 'prepared' / 'manifest.json').is_file():
                raise ValueError('prepare must complete before any later stage')
            dest = claim_stage_dir(output, stage, seed)
            summary = STAGE_FUNCTIONS[stage](ctx, dest, seed)
        except BaseException as error:
            write_failure(output, stage, seed, error, ctx, dest)  # output is the validated registered path
            raise
    print(json.dumps({'family_id': mj.FAMILY_ID, 'stage': stage, 'seed': seed, 'status': 'complete', **summary}), flush=True)
    return summary


def _terminate(signum, frame):
    raise Terminated(f'signal {signum}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-closure', action='store_true',
                        help='print the absolute project source closure and exit (no data, no lock)')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--stage', choices=mj.STAGES)
    parser.add_argument('--seed', type=int, choices=mj.SEEDS)
    args = parser.parse_args(argv)
    if args.source_closure:
        print(json.dumps(source_closure(), indent=2, sort_keys=True))
        return 0
    if args.config is None or args.output_dir is None or args.stage is None:
        parser.error('--config, --output-dir and --stage are required')
    signal.signal(signal.SIGTERM, _terminate)
    run(args.config, args.output_dir, args.stage, args.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
