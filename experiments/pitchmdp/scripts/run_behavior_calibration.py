"""EXP-BC-CAL-01 (track A3): a better-calibrated behavior policy pi_b_hat for the OPE weights.

Fail closed. Every real-data stage, the smoke included, is refused unless the config passes the
registered contract (``behavior_calibration_registry.config_check``) AND ``registration_gate``:
status REGISTERED, registered / execute / real_data_enabled true, review references and every
registration pin filled, the config content sha256 and the runner sha256 pinned, and no code
change since the pinned commit. Order is fixed: smoke (first registered step: TRAIN rows only,
cost and integrity, no metric) -> census (label-free) -> fit -> calibrate -> freeze-primary ->
s6-integrity -> xf-fit -> freeze-xf -> score-dev -> s6-weights -> decide. A stage opens only
after the previous one passed in the same attempt directory; DEV pitch-type labels stay sealed
until freeze-primary is pinned and s6-integrity passed; no DEV metric is computed before
freeze-xf is pinned. Each stage writes JSON (+ npy/text artifacts) and SHA256SUMS under the
attempt directory (never the repository results/), with seeds, config sha256, git commit,
seconds and peak RSS. Caps: total wall 14,400 s, stage hang guard 7,200 s, B2 fit soft cap
3,600 s (warning only), RSS 10 GiB; the protection window 06:30-13:30 KST is never entered
(refuse to start inside it, abort and keep outputs when it begins). Aborted outputs are not
citable.

After registration, outside 06:30-13:30 KST, with the common heavy lock free (torch and LightGBM
must share one OpenMP runtime: DYLD_LIBRARY_PATH=<venv torch/lib>, never KMP_DUPLICATE_LIB_OK):
  cd <repo>; DYLD_LIBRARY_PATH=/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages/torch/lib \
      PYTHONPATH="experiments/pitchmdp" /Users/song/Projects/pitcheezy/.venv/bin/python \
      experiments/pitchmdp/scripts/run_behavior_calibration.py --config configs/EXP-BC-CAL-01.json \
      --local-config experiments/pitchmdp/configs/local.json \
      --destination "<runner.output_root>/<attempt>" smoke
  (the same command with ``run`` afterwards: census .. decide in one process)
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
import pandas as pd

from pitchmdp import behavior_calibration_registry as reg
from pitchmdp.behavior_calibration import CALIBRATION_FAILED, calibrate, calibrator_from_identity, public
from pitchmdp.behavior_features import (SEALED, FeatureContext, check_splits, census_counts, game_fold,
                                        guard_dates, label_free_view, population_masks, split_table, style_tables,
                                        workload_features)
from pitchmdp.behavior_metrics import (group_calibration, holm_adjust, paired_game_bootstrap, prediction_summary,
                                       ratio_bins, strata_metrics)
from pitchmdp.behavior_model import (BackoffBC, Calibrated, Keys, LGBMBehavior, RegisteredBC, RepertoireSupport,
                                     VectorBC, alpha_from_token, arrays_sha256, pitch_key, trailing_bits)
from pitchmdp.behavior_weights import (EVALUATED as EVALUATED_STATUSES, by_pa, e0_from_ledger,
                                       public as public_weights, r1b_logging, r2_reproduction, read_ledger,
                                       weight_diagnostics)
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.policy_artifacts import IntegrityError
from pitchmdp.rollout_policy import PAState, PastPitch

SELF = 'experiments/pitchmdp/scripts/run_behavior_calibration.py'
KST = timezone(timedelta(hours=9))
CODE_PATHS = ('experiments', 'src', 'scripts')
PREFLIGHT_TESTS = ('tests/test_behavior_calibration_registry.py', 'tests/test_behavior_calibration_model.py',
                   'tests/test_behavior_calibration_features.py', 'tests/test_behavior_calibration_calibration.py',
                   'tests/test_behavior_calibration_metrics.py', 'tests/test_behavior_calibration_weights.py',
                   'tests/test_behavior_calibration_runner.py')
ARTIFACT_EXCLUDE = ('stage.json', 'SHA256SUMS')


class Refused(Exception):
    """Registration, destination or stage-order refusal: nothing runs, the refusal is recorded."""


class Abort(BaseException):
    """Cap, hang guard or protection window: the stage stops, outputs are kept and not citable.
    A BaseException, so no ``except Exception`` inside a stage turns it into a failure record."""
    def __init__(self, reason, where=''):
        super().__init__(f'{reason} at {where}' if where else reason)
        self.reason, self.where = reason, where


# ---------------------------------------------------------------- JSON and hashes

def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return 'inf' if value > 0 else ('-inf' if value < 0 else None)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def canonical_bytes(value):
    return (json.dumps(plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
            + '\n').encode()


def dump_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(plain(value), indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
    return path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def config_content_sha256(config):
    """Registered config identity: canonical sha256 with code.config_sha256 nulled and the
    append-only registration.addenda_records removed (both are written after the pin)."""
    body = copy.deepcopy(config)
    body.setdefault('code', {})['config_sha256'] = None
    (body.get('registration') or {}).pop('addenda_records', None)
    return canonical_hash(body)


# ---------------------------------------------------------------- registration gate and destination

def _field(config, dotted):
    node = config
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def git_state(commit, config_path=None):
    """HEAD, uncommitted code-path changes, code paths changed since ``commit`` and whether the
    config file is committed at HEAD byte for byte."""
    run = lambda *args: subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
    state = {'head': run('rev-parse', 'HEAD'),
             'dirty': bool(run('status', '--porcelain', '--', *CODE_PATHS)), 'commit_is_ancestor': None,
             'changed_since_commit': None, 'config_committed': None}
    if commit:
        state['commit_is_ancestor'] = subprocess.call(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO,
                                                      stderr=subprocess.DEVNULL) == 0
        if state['commit_is_ancestor']:
            state['changed_since_commit'] = [x for x in run('diff', '--name-only', commit, 'HEAD', '--',
                                                            *CODE_PATHS).splitlines() if x]
    if config_path is not None:
        path = Path(config_path).resolve()
        try:
            blob = subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(REPO).as_posix()}'], cwd=REPO,
                                           stderr=subprocess.DEVNULL)
            state['config_committed'] = blob == path.read_bytes()
        except (ValueError, subprocess.CalledProcessError):
            state['config_committed'] = False
    return state


def registration_gate(config, *, git, runner_sha256):
    """Refuse unless EXP-BC-CAL-01 is registered and execution-enabled with every pin filled.
    Runs before any data is touched; the smoke is a registered step and passes through here."""
    try:
        reg.config_check(config)
    except reg.RegistrationError as error:
        raise Refused(f'config deviates from the registered contract: {error}') from error
    reasons = []
    if not (config.get('status') == 'REGISTERED' and config.get('registered') is True
            and config.get('execute') is True and config.get('real_data_enabled') is True):
        reasons.append(f'status {config.get("status")!r} is not REGISTERED with execute/real_data_enabled true')
    registration = config.get('registration') or {}
    for name in ('registered_kst', 'design_review_reference', 'code_review_reference'):
        if not registration.get(name):
            reasons.append(f'registration.{name} missing')
    for dotted in registration.get('fields_pinned_at_registration') or []:
        if _field(config, dotted) in (None, ''):
            reasons.append(f'registration pin {dotted} missing')
    if _field(config, 'code.config_sha256') != config_content_sha256(config):
        reasons.append('code.config_sha256 differs from the config content')
    if _field(config, 'code.runner_sha256') != runner_sha256:
        reasons.append('code.runner_sha256 differs from this runner')
    commit = _field(config, 'code.commit')
    if not git.get('commit_is_ancestor') or git.get('changed_since_commit'):
        reasons.append(f'code at HEAD differs from the registered commit {commit}')
    if git.get('dirty'):
        reasons.append('uncommitted changes under experiments/, src/ or scripts/')
    if git.get('config_committed') is not True:
        reasons.append('the config file is not committed at HEAD byte for byte')
    if reasons:
        raise Refused('registration gate: ' + '; '.join(reasons))
    return True


def check_destination(destination, *, output_root=None, repos=(REPO,)):
    """Never inside a repository results/ directory or a sealed S5/S6 stage directory; for real runs
    a new directory directly under the registered output root."""
    path = Path(destination).expanduser().resolve()
    for repo in repos:
        if path == (Path(repo) / 'results').resolve() or (Path(repo) / 'results').resolve() in path.parents:
            raise Refused(f'destination inside the repository results/: {path}')
    for parent in (path, *path.parents):
        if parent.name == 'results' and (parent.parent / '.git').exists():
            raise Refused(f'destination inside a repository results/: {path}')
    if 'ML-POLICY-VAL-v1' in path.parts:
        raise Refused('destination inside the sealed ML-POLICY-VAL-v1 (S5/S6) stage directories')
    if output_root is not None and path.parent != Path(output_root).expanduser().resolve():
        raise Refused(f'destination must be a new directory directly under {output_root}')
    return path


# ---------------------------------------------------------------- caps, clock, memory

def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == 'darwin' else raw * 1024)


def free_share():
    """macOS ``memory_pressure`` system-wide free share (STAB-01 precedent)."""
    text = subprocess.run(['memory_pressure'], capture_output=True, text=True, timeout=120).stdout
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    if not match:
        raise Refused('memory_pressure free percentage unavailable')
    return int(match.group(1)) / 100


def _minutes(text):
    hours, minutes = (int(x) for x in text.split(':'))
    return hours * 60 + minutes


class Guard:
    """Cooperative caps (checked in loops, LightGBM callbacks and at stage ends) plus a SIGALRM
    backstop for the time limits. RSS is the process peak (ru_maxrss)."""
    def __init__(self, caps, window, *, clock=time.monotonic, now=None, rss=peak_rss_bytes, alarm=True):
        self.total_cap, self.stage_cap = caps['total_wall_cap_seconds'], caps['stage_hang_guard_seconds']
        self.rss_cap, self.b2_soft_cap = caps['abort_rss_bytes'], caps['b2_fit_soft_cap_seconds']
        self.window = (_minutes(window['start']), _minutes(window['end']))
        self.clock, self.now = clock, now or (lambda: datetime.now(KST))
        self.rss, self.alarm = rss, alarm
        self.run_start = self.stage_start = None
        self.stage_name, self.armed = '', False
        self.warnings = []

    def in_window(self):
        t = self.now().astimezone(KST)
        minutes = t.hour * 60 + t.minute + t.second / 60
        return self.window[0] <= minutes < self.window[1]

    def seconds_to_window(self):
        t = self.now().astimezone(KST)
        start = t.replace(hour=self.window[0] // 60, minute=self.window[0] % 60, second=0, microsecond=0)
        if start <= t:
            start += timedelta(days=1)
        return (start - t).total_seconds()

    def start_run(self):
        if self.in_window():
            raise Refused('inside the recorder/publisher protection window 06:30-13:30 KST; refusing to start')
        self.run_start = self.clock()

    def start_stage(self, name):
        if self.run_start is None:
            self.start_run()
        self.stage_name, self.stage_start = name, self.clock()
        self.check(f'{name}:start')
        if self.alarm:
            # The handler stays installed for the process (a late signal after stop_stage is ignored,
            # never the default action that would kill the process).
            signal.signal(signal.SIGALRM, self._on_alarm)
            self.armed = True
            self._arm()

    def _arm(self):
        remaining = min(self.stage_cap - (self.clock() - self.stage_start),
                        self.total_cap - (self.clock() - self.run_start), self.seconds_to_window())
        signal.setitimer(signal.ITIMER_REAL, max(remaining, 0.05))

    def _on_alarm(self, signum, frame):
        if not self.armed:
            return
        self.check(f'{self.stage_name}:alarm')
        self._arm()

    def stop_stage(self):
        self.armed = False
        if self.alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def check(self, where=''):
        if self.in_window():
            raise Abort('PROTECTION_WINDOW', where)
        now = self.clock()
        if self.run_start is not None and now - self.run_start > self.total_cap:
            raise Abort('TOTAL_WALL_CAP', where)
        if self.stage_start is not None and now - self.stage_start > self.stage_cap:
            raise Abort('STAGE_HANG_GUARD', where)
        if self.rss() > self.rss_cap:
            raise Abort('RSS_CAP', where)

    def soft_b2(self, seconds, where):
        if seconds > self.b2_soft_cap:
            warning = {'warning': 'B2_FIT_SOFT_CAP_EXCEEDED', 'seconds': seconds, 'soft_cap_seconds': self.b2_soft_cap,
                       'where': where, 'action': 'recorded; run continues (abort only by hang guard / wall cap)'}
            self.warnings.append(warning)
            return warning
        return None


# ---------------------------------------------------------------- attempt directory, records, pins

def _chain_line(body, prev, seq):
    row = {**body, 'seq': seq, 'prev': prev}
    row['sha256'] = sha256_bytes(canonical_bytes(row))
    return row


def read_chain(path):
    rows, prev = [], None
    if not Path(path).exists():
        return rows
    for seq, line in enumerate(Path(path).read_bytes().splitlines()):
        row = json.loads(line)
        body = {k: v for k, v in row.items() if k != 'sha256'}
        if row.get('seq') != seq or row.get('prev') != prev or sha256_bytes(canonical_bytes(body)) != row.get('sha256'):
            raise IntegrityError(f'{Path(path).name}: chain broken at row {seq}')
        rows.append(row)
        prev = row['sha256']
    return rows


def append_chain(path, body):
    rows = read_chain(path)
    row = _chain_line(body, rows[-1]['sha256'] if rows else None, len(rows))
    with Path(path).open('ab') as stream:
        stream.write(canonical_bytes(row))
    return row


def sha256sums(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    return ''.join(f'{sha256_file(p)}  {p.relative_to(directory).as_posix()}\n' for p in files)


def verify_sha256sums(directory):
    directory = Path(directory)
    listed = (directory / 'SHA256SUMS').read_text() if (directory / 'SHA256SUMS').exists() else None
    return listed is not None and listed == sha256sums(directory)


class StageDir:
    def __init__(self, path):
        self.path = Path(path)

    def json(self, name, value):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return dump_json(target, value)

    def npy(self, name, array):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as stream:
            np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
        return target

    def text(self, name, text):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def artifacts(self):
        return sorted(p for p in self.path.rglob('*') if p.is_file() and p.name not in ARTIFACT_EXCLUDE)


class Attempt:
    """One attempt directory: numbered stage directories, the stages.jsonl and addenda.jsonl chains."""
    def __init__(self, root):
        self.root = Path(root)

    def stage_path(self, stage):
        return self.root / f'{reg.STAGES.index(stage):02d}-{stage}'

    def records(self):
        return read_chain(self.root / 'stages.jsonl')

    def status(self, stage):
        rows = [r for r in self.records() if r['stage'] == stage]
        return rows[-1]['status'] if rows else None

    def passed(self, stage):
        if self.status(stage) != 'PASS':
            return False
        path = self.stage_path(stage)
        if not verify_sha256sums(path):
            raise IntegrityError(f'stage {stage}: SHA256SUMS no longer match its files')
        return True

    def addenda(self):
        return read_chain(self.root / 'addenda.jsonl')

    def pin(self, name):
        rows = [r for r in self.addenda() if r.get('addendum') == name]
        return rows[-1] if rows else None


class Gate:
    """Pins and passed stages read back from the attempt (append-only chains)."""
    def __init__(self, attempt):
        self.attempt = attempt

    def require_pinned(self, name, purpose):
        row = self.attempt.pin(name)
        if row is None:
            raise Refused(f'{purpose} needs the {name} manifest pinned first')
        manifest = self.attempt.stage_path(name) / 'manifest.json'
        if not manifest.exists() or sha256_file(manifest) != row['manifest_sha256']:
            raise IntegrityError(f'{name} manifest differs from its pinned sha256')
        return row

    def require_passed(self, stage, purpose):
        if not self.attempt.passed(stage):
            raise Refused(f'{purpose} needs a passed {stage} stage in this attempt')


class SealedLabels:
    """DEV pitch-type labels kept out of the frame (SEALED placeholder) until their release."""
    def __init__(self, frame, split='dev'):
        mask = frame.split.eq(split)
        self.index = frame.index[mask]
        self.values = frame.loc[mask, 'pitch_type'].astype(object).copy()
        self.sha256 = sha256_bytes('\n'.join('' if pd.isna(v) else str(v) for v in self.values).encode())
        frame['pitch_type'] = frame['pitch_type'].astype(object)
        frame.loc[mask, 'pitch_type'] = SEALED
        self.released = None

    def release(self, frame, gate, purpose):
        gate.require_pinned('freeze-primary', purpose)
        gate.require_passed('s6-integrity', purpose)
        frame.loc[self.index, 'pitch_type'] = self.values
        self.released = purpose
        return self.sha256


# ---------------------------------------------------------------- manifests

def build_manifest(root, files, *, name, groups, extra):
    """Deterministic freeze manifest: sha256 and bytes of every artifact (paths relative to the
    attempt root) plus groups with their input-row date ranges. Same files -> same bytes."""
    root = Path(root)
    artifacts = {Path(p).resolve().relative_to(root.resolve()).as_posix(): {'sha256': sha256_file(p),
                                                                          'bytes': Path(p).stat().st_size}
                 for p in files}
    return {'experiment': reg.EXPERIMENT_ID, 'manifest': name, 'artifacts': dict(sorted(artifacts.items())),
            'groups': groups, **extra}


def verify_manifest(root, manifest):
    for rel, entry in manifest['artifacts'].items():
        path = Path(root) / rel
        if not path.is_file() or sha256_file(path) != entry['sha256']:
            raise IntegrityError(f'frozen artifact changed or missing: {rel}')
    return True


def load_frozen_npy(root, manifest, rel):
    entry = manifest['artifacts'].get(rel)
    if entry is None:
        raise IntegrityError(f'{rel} is not in the frozen manifest')
    data = (Path(root) / rel).read_bytes()
    if sha256_bytes(data) != entry['sha256']:
        raise IntegrityError(f'frozen artifact changed: {rel}')
    return np.load(io.BytesIO(data), allow_pickle=False)


def load_frozen_text(root, manifest, rel):
    entry = manifest['artifacts'].get(rel)
    data = (Path(root) / rel).read_bytes() if entry else None
    if data is None or sha256_bytes(data) != entry['sha256']:
        raise IntegrityError(f'frozen artifact changed or missing: {rel}')
    return data.decode()


# ---------------------------------------------------------------- run object

class Run:
    def __init__(self, config, *, destination, source, guard, commit, preflight=None, output_root=None, lock=None):
        self.config, self.destination = config, Path(destination)
        self.config_sha256 = config_content_sha256(config)
        self.source, self.guard, self.commit = source, guard, commit
        self.preflight, self.output_root, self.lock = preflight, output_root, lock
        self.attempt = Attempt(self.destination)
        self.gate = Gate(self.attempt)
        self.state = {}
        self._inputs = None

    def inputs(self):
        if self._inputs is None:
            self._inputs = self.source.load(self)
            check_splits(self._inputs.frame)
        return self._inputs

    def seeds(self):
        return self.config['seeds']

    def require_order(self, stage):
        index = reg.STAGES.index(stage)
        if self.attempt.status(stage) is not None or self.attempt.stage_path(stage).exists():
            raise Refused(f'{stage} already exists in this attempt; register a new attempt')
        if index and not self.attempt.passed(reg.STAGES[index - 1]):
            raise Refused(f'{stage} needs a passed {reg.STAGES[index - 1]} first (fixed stage order)')
        failed = [r['stage'] for r in self.attempt.records() if r['status'] != 'PASS']
        if failed:
            raise Refused(f'an earlier stage did not pass ({failed}); no later stage runs in this attempt')


def execute_stage(run, stage):
    """Run one stage with its record, SHA256SUMS and chain entry; return its status."""
    run.require_order(stage)
    path = run.attempt.stage_path(stage)
    path.mkdir(parents=True, exist_ok=False)
    out = StageDir(path)
    started, t0 = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    status, error, summary = 'PASS', None, None
    try:
        try:
            run.guard.start_stage(stage)
            summary = STAGE_FUNCTIONS[stage](run, out)
            run.guard.check(f'{stage}:end')
        finally:
            run.guard.stop_stage()  # disarmed before any record is written
    except Abort as abort:
        status, error = 'ABORTED', {'reason': abort.reason, 'where': abort.where}
    except KeyboardInterrupt:
        status, error = 'ABORTED', {'reason': 'INTERRUPTED'}
    except Refused as refusal:
        status, error = 'REFUSED', {'detail': str(refusal)}
    except IntegrityError as failure:
        status, error = 'FAILED_INTEGRITY', {'detail': str(failure)}
    except Exception as failure:  # a bug: recorded; a bug-fix retry is a new attempt before any DEV metric
        status, error = 'FAILED_RUNTIME', {'detail': f'{type(failure).__name__}: {failure}',
                                           'traceback': traceback.format_exc()}
    record = {'experiment': reg.EXPERIMENT_ID, 'stage': stage, 'status': status, 'started_utc': started,
              'ended_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - t0,
              'peak_rss_bytes': run.guard.rss(), 'seeds': run.seeds(), 'config_sha256': run.config_sha256,
              'git_commit': run.commit, 'summary': summary, 'error': error, 'warnings': list(run.guard.warnings),
              'citable': status == 'PASS', 'environment': environment()}
    run.guard.warnings.clear()
    out.json('stage.json', record)
    out.text('SHA256SUMS', sha256sums(path))
    append_chain(run.attempt.root / 'stages.jsonl',
                 {'stage': stage, 'status': status, 'stage_json_sha256': sha256_file(path / 'stage.json'),
                  'sha256sums_sha256': sha256_file(path / 'SHA256SUMS')})
    if status in ('FAILED_INTEGRITY', 'ABORTED'):
        dump_json(run.attempt.root / 'final.json', {'label': 'FAILED_INTEGRITY' if status == 'FAILED_INTEGRITY'
                                                    else 'ABORTED', 'stage': stage, 'error': error,
                                                    'citable': False, 'config_sha256': run.config_sha256})
    return status


def environment():
    """Versions only (importlib.metadata: LightGBM is never imported just to be recorded)."""
    import importlib.metadata
    out = {'python': platform.python_version(), 'machine': platform.machine(),
           'dynamic_library_path': os.environ.get('DYLD_LIBRARY_PATH', '')}
    for name in ('numpy', 'pandas', 'scipy', 'lightgbm', 'torch'):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def validate_native_runtime():
    """run_ml_benchmark.validate_native_runtime rule: torch (pulled in by pitchmdp.model) and
    LightGBM must share ONE OpenMP runtime; the unsafe duplicate override is refused."""
    import importlib.util
    if os.environ.get('KMP_DUPLICATE_LIB_OK', '').lower() in ('true', '1', 'yes'):
        raise Refused('the unsafe KMP_DUPLICATE_LIB_OK override is forbidden')
    if sys.platform == 'darwin':
        spec = importlib.util.find_spec('torch')
        torch_lib = (Path(spec.origin).resolve().parent / 'lib') if spec else None
        configured = [Path(v).resolve() for v in os.environ.get('DYLD_LIBRARY_PATH', '').split(':') if v]
        if torch_lib is None or torch_lib.resolve() not in configured:
            raise Refused(f'launch with DYLD_LIBRARY_PATH={torch_lib} so torch and LightGBM share one OpenMP runtime')
    return True


# ---------------------------------------------------------------- shared pieces of the stages

def oof_spec(config):
    return {'salt': config['seeds']['fold_salts']['train_oof'], 'k': config['cross_fitting']['folds']['k'],
            'prior_strength': float(config['candidates']['B0']['parameters']['prior_strength']),
            'minimum_action_count': int(config['candidates']['B0']['parameters']['minimum_action_count'])}


def feature_context(run, frame, train):
    from pitchmdp.policy_requests import volume_edges
    inputs = run.inputs()
    volumes = train.groupby('pitcher').size()
    edges = volume_edges(train, (0.25, 0.75))
    raw = inputs.raw_keys.loc[inputs.raw_keys.game_pk.isin(frame.game_pk.unique())]
    return FeatureContext(vocabulary=reg.ACTION_CODES, archetypes=inputs.archetypes, b0=inputs.b0.vector,
                          train_volume={int(p): int(n) for p, n in volumes.items()}, volume_edges=tuple(edges),
                          style=style_tables(frame), workload=workload_features(frame, raw), hands=inputs.hands,
                          repertoire=inputs.b0.repertoire)


def date_range(table, rows=None):
    dates = table.frame.game_date if rows is None else table.frame.game_date.iloc[rows]
    if not len(dates):
        return None
    return [str(pd.Timestamp(dates.min()).date()), str(pd.Timestamp(dates.max()).date())]


def labels(table, rows=None):
    values = table.frame.label.to_numpy(np.int64)
    return values if rows is None else values[rows]


def s_in_rows(table):
    return np.flatnonzero(table.frame.S_in.to_numpy(bool))


def b0_cell_check(b1, b0_vector):
    """B1 at the B0 setting must be fit on exactly the BC_P rows: its league, pitcher and
    (pitcher, count, side, previous) counts equal the registered artifact's."""
    pitchers, cells = b1.tables['pitcher'], b1.tables['previous_type']
    expected = b0_vector.repertoire.table
    return {'league': bool(np.array_equal(b1.league, b0_vector.repertoire.league)),
            'pitchers': bool(np.array_equal(pitchers.keys >> trailing_bits(1), expected.keys)
                             and np.array_equal(pitchers.counts, expected.counts)),
            'cells': bool(np.array_equal(cells.keys, b0_vector.cells.keys)
                          and np.array_equal(cells.counts, b0_vector.cells.counts))}


def collect_dates(node):
    """Every [first, last] date pair under a ``dates`` key of a nested record."""
    if isinstance(node, dict):
        found = [node['dates']] if isinstance(node.get('dates'), list) else []
        return found + [d for k, v in node.items() if k != 'dates' for d in collect_dates(v)]
    return []


def token(alpha):
    return 'inf' if alpha == math.inf else (int(alpha) if float(alpha).is_integer() else float(alpha))


def save_arrays(out, prefix, arrays):
    for name, array in sorted(arrays.items()):
        out.npy(f'{prefix}/{name}.npy', array)


def load_arrays(root, manifest, prefix, names):
    return {name: load_frozen_npy(root, manifest, f'{prefix}/{name}.npy') for name in names}


# ---------------------------------------------------------------- stages

def stage_smoke(run, out):
    """First registered step: TRAIN rows only, cost and integrity (wall time, RSS, determinism,
    hashes). No metric is computed or read."""
    inputs = run.inputs()
    spec = run.config['runner']['smoke']
    frame = inputs.frame
    train = frame.loc[frame.split.eq('train')]
    masks = population_masks(train, hands=inputs.hands, repertoire=inputs.b0.repertoire)
    plus = np.flatnonzero(masks['P_plus'])
    n = min(int(spec['train_rows']), len(plus))
    if not n:
        raise IntegrityError('no TRAIN P_plus rows for the smoke')
    part = train.iloc[:plus[n - 1] + 1]
    timings = {}
    t = time.perf_counter()
    ctx = feature_context(run, part, train)
    table, info, _, oof_models = split_table(part, 'train', ctx, oof=oof_spec(run.config))
    timings['features_seconds'] = time.perf_counter() - t
    run.guard.check('smoke:features')
    rows = np.arange(min(int(spec['state_checks']), len(table)))
    states = [PAState(int(r.balls), int(r.strikes), str(int(r.pitcher)), str(r.batter_side),
                      () if r.previous == '<START>' else (PastPitch(str(r.previous), (), 'unknown', 0, 0),))
              for r in table.frame.iloc[rows].itertuples()]
    per_state = np.array([inputs.b0.probabilities(s) for s in states])
    vector = inputs.b0.batch(table, rows)
    r1b_train = float(np.abs(per_state - vector).max()) if len(rows) else None
    keys, y = table.keys, labels(table)
    t = time.perf_counter()
    b1_b0 = BackoffBC.fit(keys, y, repertoire=RepertoireSupport.from_rows(keys.values['pitcher'], y, 18))
    refit = VectorBC.fit(keys, y, prior_strength=oof_spec(run.config)['prior_strength'],
                         minimum_action_count=oof_spec(run.config)['minimum_action_count'])
    b1_equals_b0 = float(np.abs(b1_b0.batch(table) - refit.batch(table)).max())
    timings['b1_seconds'] = time.perf_counter() - t
    features = table.matrix(reg.B2_FEATURES, None)
    fits = []
    for i in range(int(spec['b2_fits'])):
        t = time.perf_counter()
        model = LGBMBehavior.train_only(features, y, params=run.config['candidates']['B2']['params'],
                                        rounds=int(spec['b2_rounds']), guard=run.guard)
        seconds = time.perf_counter() - t
        b2 = LGBMBehavior(model, int(spec['b2_rounds']), inputs.b0.repertoire)
        predictions = b2.batch(table)
        fits.append({'seconds': seconds, 'peak_rss_bytes_after': run.guard.rss(),
                     'model_sha256': sha256_bytes(model.encode()),
                     'predictions_sha256': sha256_bytes(np.ascontiguousarray(predictions).tobytes())})
        warning = run.guard.soft_b2(seconds, f'smoke b2 fit {i}')
        if warning:
            fits[-1]['warning'] = warning
    deterministic = len({(f['model_sha256'], f['predictions_sha256']) for f in fits}) == 1
    atol = run.config['ope_weight_diagnostics']['integrity_gates']['R1b_atol']
    pinned_b0 = run.config['data']['b0_artifact']['file_sha256']
    checks = {'b0_artifact_sha256_pinned': inputs.identity['b0']['file_sha256'] == pinned_b0,
              'vector_b0_equals_categorical_bc_on_train': r1b_train is not None and r1b_train <= atol,
              'b1_at_b0_setting_equals_b0_recipe': b1_equals_b0 <= atol,
              'b2_deterministic': deterministic, 'no_2026_row': guard_dates(frame) is frame,
              'dev_labels_sealed': inputs.sealed.released is None}
    record = {'rows': {'train_slice_rows': int(len(part)), 'train_P_plus_rows': int(len(table)),
                       'state_checks': int(len(rows))}, 'timings': timings, 'b2_fits': fits,
              'max_abs_diff': {'vector_vs_categorical': r1b_train, 'b1_b0_setting_vs_vector_fit': b1_equals_b0},
              'checks': checks, 'feature_matrix_sha256': sha256_bytes(features.tobytes()),
              'oof_models_sha256': [arrays_sha256(m.arrays()) for m in oof_models],
              'metrics_computed': None, 'note': 'cost and integrity only; no comparison metric computed or read'}
    out.json('smoke.json', record)
    if not all(checks.values()):
        raise IntegrityError(f'smoke integrity failed: {[k for k, v in checks.items() if not v]}')
    append_chain(run.attempt.root / 'addenda.jsonl',
                 {'addendum': 'smoke', 'record_sha256': sha256_file(out.path / 'smoke.json'),
                  'config_sha256': run.config_sha256, 'git_commit': run.commit})
    return {'checks': checks, 'b2_fit_seconds': [f['seconds'] for f in fits]}


def stage_census(run, out):
    inputs = run.inputs()
    counts = census_counts(label_free_view(inputs.frame), hands=inputs.hands, repertoire=inputs.b0.repertoire)
    out.json('census.json', counts)
    return {'splits': {k: v['rows'] for k, v in counts['splits'].items()}}


def stage_fit(run, out):
    inputs, config = run.inputs(), run.config
    frame = inputs.frame
    train = frame.loc[frame.split.eq('train')]
    ctx = feature_context(run, frame, train)
    run.guard.check('fit:context')
    train_table, train_info, _, oof_models = split_table(train, 'train', ctx, oof=oof_spec(config))
    run.guard.check('fit:train-table')
    expected_rows = inputs.b0.artifact.provenance['train_rows']
    if len(train_table) != expected_rows or train_info['counts']['bc_rule']['rows'] != expected_rows:
        raise IntegrityError(f'TRAIN P_plus rows {len(train_table)} differ from the BC_P TRAIN rows {expected_rows}')
    y = labels(train_table)
    b1 = BackoffBC.fit(train_table.keys, y, repertoire=inputs.b0.repertoire, provenance={'fit': 'TRAIN P_plus = BC_P'})
    checks = b0_cell_check(b1, inputs.b0.vector)
    if not all(checks.values()):
        raise IntegrityError(f'B1 counts differ from the registered BC_P artifact: {checks}')
    es_table, es_info, _, _ = split_table(frame.loc[frame.split.eq('earlystop')], 'earlystop', ctx)
    es_rows = s_in_rows(es_table)
    if not len(es_rows):
        raise IntegrityError('no earlystop S_in rows')
    tune = b1.tune(es_table.keys.take(es_rows), labels(es_table, es_rows), sweeps=config['candidates']['B1']['search']['sweeps'],
                   guard=run.guard)
    t = time.perf_counter()
    model_string, best, b2_record = LGBMBehavior.train(
        train_table.matrix(reg.B2_FEATURES, None), y, es_table.matrix(reg.B2_FEATURES, es_rows),
        labels(es_table, es_rows), inputs.b0.repertoire.mask(es_table.keys.values['pitcher'][es_rows]),
        params=config['candidates']['B2']['params'], rounds=config['candidates']['B2']['rounds']['train_rounds'],
        guard=run.guard)
    b2_record['seconds'] = time.perf_counter() - t
    warning = run.guard.soft_b2(b2_record['seconds'], 'fit b2')
    b2 = LGBMBehavior(model_string, best, inputs.b0.repertoire, b2_record)
    save_arrays(out, 'b1', b1.arrays())
    out.json('b1/b1.json', {'levels': list(reg.B1_LEVELS), 'alphas': tune['alphas'], 'tune': tune,
                            'b0_cell_check': checks})
    out.text('b2/model.txt', model_string)
    out.json('b2/b2.json', {'best_round': best, 'record': b2_record, 'cap_hit': b2_record['cap_hit'],
                            'warning': warning, 'features': list(reg.B2_FEATURES)})
    for f, model in enumerate(oof_models):
        save_arrays(out, f'oof/fold{f}', model.arrays())
    out.json('oof/oof.json', {'salt': oof_spec(config)['salt'], 'k': oof_spec(config)['k'],
                              'fit_rows': [m.provenance['fit_rows'] for m in oof_models]})
    from pitchmdp.policy_artifacts import save_style_snapshot
    (out.path / 'style').mkdir(parents=True, exist_ok=True)
    for split, snapshot in ctx.style.snapshots.items():
        save_style_snapshot(snapshot, reg.STYLE_AS_OF[split],
                            {'rule': 'A3 batter-style inputs (registry STYLE_INPUT_FILTER)',
                             'input_rows': ctx.style.input_rows, 'input_dates': ctx.style.input_dates},
                            out.path / 'style' / f'snapshot_{split}.json')
    out.json('style/style.json', {'input_rows': ctx.style.input_rows, 'input_dates': ctx.style.input_dates,
                                  'train_rolling_sha256': sha256_bytes(
                                      np.ascontiguousarray(ctx.style.rolling.to_numpy(np.float32)).tobytes()),
                                  'as_of': reg.STYLE_AS_OF})
    out.json('train_volume.json', {'volumes': {str(k): v for k, v in sorted(ctx.train_volume.items())},
                                   'edges_q25_q75': list(ctx.volume_edges)})
    out.json('fit.json', {'train': train_info, 'earlystop': es_info, 'b0_cell_check': checks,
                          'b1_alphas': tune['alphas'], 'b2_best_round': best, 'b2_cap_hit': b2_record['cap_hit']})
    run.state.update(ctx=ctx, train_table=train_table, es_table=es_table, es_rows=es_rows, b1=b1, b2=b2)
    return {'b1_alphas': tune['alphas'], 'b2_best_round': best, 'b2_cap_hit': b2_record['cap_hit'],
            'train_rows': len(train_table), 'earlystop_s_in_rows': int(len(es_rows))}


def stage_calibrate(run, out):
    inputs, config = run.inputs(), run.config
    ctx, b1, b2 = run.state['ctx'], run.state['b1'], run.state['b2']
    table, info, _, _ = split_table(inputs.frame.loc[inputs.frame.split.eq('temperature')], 'temperature', ctx)
    rows = s_in_rows(table)
    if not len(rows):
        raise IntegrityError('no temperature S_in rows')
    y, games = labels(table, rows), table.frame.game_pk.to_numpy()[rows]
    mask = inputs.b0.repertoire.mask(table.keys.values['pitcher'][rows])
    spec = {'bounds': config['calibration']['methods']['temperature']['bounds'],
            'l2': config['calibration']['methods']['vector_scaling']['l2_penalty']['value'],
            'margin': config['calibration']['selection']['margin'], 'k': config['calibration']['selection']['cv']['k'],
            'seed': config['calibration']['selection']['cv']['seed'],
            'lower': config['calibration']['vector_a_lower_bound']}
    records = {'B1': calibrate(b1.batch(table, rows), mask, y, games, **spec),
               'B2': calibrate(b2.batch(table, rows), mask, y, games, **spec),
               'B0-T': calibrate(inputs.b0.batch(table, rows), mask, y, games, **spec, vector=False)}
    for name, record in records.items():
        out.json(f'calibrators/{name}.json', public(record))
    out.json('calibrate.json', {'temperature': info, 'status': {k: v['status'] for k, v in records.items()}})
    run.state.update(temp_table=table, temp_rows=rows, calibration=records)
    return {name: {'status': r['status'], 'method': r['method']} for name, r in records.items()}


def stage_freeze_primary(run, out):
    root = run.attempt.root
    files = [p for stage in ('fit', 'calibrate') for p in StageDir(run.attempt.stage_path(stage)).artifacts()]
    train, es, temp = run.state['train_table'], run.state['es_table'], run.state['temp_table']
    ctx = run.state['ctx']
    groups = {
        'B1': {'files': 'b1/', 'input_rows': {'fit': {'split': 'train', 'dates': date_range(train), 'rows': len(train)},
                                               'alpha_selection': {'split': 'earlystop', 'dates': date_range(es, run.state['es_rows']),
                                                                   'rows': int(len(run.state['es_rows']))}}},
        'B2': {'files': 'b2/', 'input_rows': {'fit': {'split': 'train', 'dates': date_range(train), 'rows': len(train)},
                                               'round_selection': {'split': 'earlystop',
                                                                   'dates': date_range(es, run.state['es_rows'])}}},
        'calibrators': {'files': 'calibrators/', 'input_rows': {'split': 'temperature',
                                                                 'dates': date_range(temp, run.state['temp_rows']),
                                                                 'rows': int(len(run.state['temp_rows']))}},
        'oof_encoders': {'files': 'oof/', 'input_rows': {'split': 'train', 'dates': date_range(train)}},
        'style_snapshots': {'files': 'style/', 'input_rows': {'splits': reg.STYLE_INPUT_FILTER['splits'],
                                                              'dates': ctx.style.input_dates}, 'as_of': reg.STYLE_AS_OF},
        'train_volume': {'files': 'train_volume.json', 'input_rows': {'split': 'train', 'dates': date_range(train)}}}
    latest = max(d[1] for d in collect_dates(groups) if d)
    extra = {'config_sha256': run.config_sha256, 'git_commit': run.commit, 'seeds': run.seeds(),
             'inputs': run.inputs().identity, 'latest_input_row_date': latest,
             'inputs_on_or_before_2025_05_31': latest <= reg.STYLE_INPUT_FILTER['end'],
             'dev_pitch_type_labels_read': run.inputs().sealed.released is not None,
             'calibration_status': {k: v['status'] for k, v in run.state['calibration'].items()}}
    if extra['dev_pitch_type_labels_read'] or not extra['inputs_on_or_before_2025_05_31']:
        raise IntegrityError('freeze-primary inputs must be <= 2025-05-31 with DEV labels still sealed')
    manifest = build_manifest(root, files, name='freeze-primary', groups=groups, extra=extra)
    data = canonical_bytes(manifest)
    (out.path / 'manifest.json').write_bytes(data)
    append_chain(root / 'addenda.jsonl', {'addendum': 'freeze-primary', 'manifest_sha256': sha256_bytes(data),
                                          'latest_input_row_date': latest})
    reloaded = reload_primary(run, manifest)
    temp, rows = run.state['temp_table'], run.state['temp_rows']
    roundtrip = {name: float(np.abs(reloaded[name].batch(temp, rows) - run.state[key].batch(temp, rows)).max())
                 for name, key in (('B1', 'b1'), ('B2', 'b2'))}
    if any(v != 0 for v in roundtrip.values()):
        raise IntegrityError(f'frozen models do not reproduce the fitted ones: {roundtrip}')
    run.state.update(manifest_primary=manifest, b1=reloaded['B1'], b2=reloaded['B2'],
                     calibrated=reloaded['calibrated'])
    return {'manifest_sha256': sha256_bytes(data), 'artifacts': len(manifest['artifacts']), 'roundtrip_max_abs': roundtrip}


def reload_primary(run, manifest):
    """Rebuild B1, B2 and the calibrators from the frozen files (sha256 verified)."""
    root, inputs = run.attempt.root, run.inputs()
    verify_manifest(root, manifest)
    fit = run.attempt.stage_path('fit').name
    cal = run.attempt.stage_path('calibrate').name
    b1_meta = json.loads(load_frozen_text(root, manifest, f'{fit}/b1/b1.json'))
    names = ['league', 'support_league', 'support_pitchers', 'support_counts',
             *(f'{level}_{part}' for level in reg.B1_LEVELS[1:] for part in ('keys', 'counts'))]
    b1 = BackoffBC.from_arrays(load_arrays(root, manifest, f'{fit}/b1', names),
                               {k: alpha_from_token(v) for k, v in b1_meta['alphas'].items()},
                               provenance={'frozen': f'{fit}/b1'})
    b2_meta = json.loads(load_frozen_text(root, manifest, f'{fit}/b2/b2.json'))
    b2 = LGBMBehavior(load_frozen_text(root, manifest, f'{fit}/b2/model.txt'), b2_meta['best_round'],
                      inputs.b0.repertoire, b2_meta['record'])
    calibrated = {}
    for name, model in (('B1', b1), ('B2', b2), ('B0-T', inputs.b0)):
        record = json.loads(load_frozen_text(root, manifest, f'{cal}/calibrators/{name}.json'))
        calibrated[name] = (None if record['status'] == CALIBRATION_FAILED
                            else Calibrated(model, calibrator_from_identity(record['identity']), name))
    return {'B1': b1, 'B2': b2, 'calibrated': calibrated}


def stage_s6_integrity(run, out):
    run.gate.require_pinned('freeze-primary', 's6-integrity')
    inputs, config = run.inputs(), run.config
    ledger = inputs.ledger
    header, decisions, record = read_ledger(ledger['path'], manifest_path=ledger['manifest_path'],
                                            ledger_sha256=ledger['ledger_sha256'],
                                            manifest_sha256=ledger['manifest_sha256'],
                                            runtime_sha256=ledger['runtime_sha256'])
    pas = by_pa(decisions)
    e0 = e0_from_ledger(pas, inputs.b0.bc, inputs.support, inputs.hands)
    gates = config['ope_weight_diagnostics']['integrity_gates']
    r1b = r1b_logging(decisions, inputs.b0.vector)
    r2 = r2_reproduction(pas, e0, gates['R2']['expected'], gates['R2_rtol'])
    s6 = config['data']['s6_ledger']
    checks = {'S6_identity': record['sha256_before'] == record['sha256_after']
              and ledger['runtime_sha256'] == s6['runtime_sha256']['expected_value'],
              'R1': inputs.identity['b0']['file_sha256'] == config['data']['b0_artifact']['file_sha256'],
              'R1b': r1b['rows'] > 0 and r1b['max_abs_diff'] <= gates['R1b_atol'], 'R2': r2['pass']}
    out.json('s6_integrity.json', {'ledger': record, 'pas': len(pas), 'e0_pas': sum(v[0] for v in e0.values()),
                                   'r1b': r1b, 'r2': r2, 'checks': checks,
                                   'dev_metric': None, 'note': 'registered fixed-B0 integrity checks only'})
    if not all(checks.values()):
        raise IntegrityError(f'FAILED_INTEGRITY: {[k for k, v in checks.items() if not v]}')
    run.state.update(ledger_pas=pas, e0=e0)
    return {'checks': checks, 'e0_pas': sum(v[0] for v in e0.values())}


def stage_xf_fit(run, out):
    inputs, config = run.inputs(), run.config
    released = inputs.sealed.release(inputs.frame, run.gate, 'xf-fit')
    ctx, train = run.state['ctx'], run.state['train_table']
    dev, info, _, _ = split_table(inputs.frame.loc[inputs.frame.split.eq('dev')], 'dev', ctx)
    folds = game_fold(dev.frame.game_pk.to_numpy(), config['seeds']['fold_salts']['dev_xf'],
                      config['cross_fitting']['folds']['k'])
    width = len(reg.ACTION_CODES)
    b0xf, b1xf_raw = np.zeros((len(dev), width)), np.zeros((len(dev), width))
    y_train, y_dev = labels(train), labels(dev)
    alphas = run.state['b1'].alphas
    prior = oof_spec(config)
    per_fold = []
    for k in range(config['cross_fitting']['folds']['k']):
        other = np.flatnonzero(folds != k)
        keys = Keys.concat([train.keys, dev.keys.take(other)])
        actions = np.concatenate([y_train, y_dev[other]])
        # B0 recipe on TRAIN + the other DEV folds; its support R_j is redefined from the same rows
        vector = VectorBC.fit(keys, actions, prior_strength=prior['prior_strength'],
                              minimum_action_count=prior['minimum_action_count'],
                              provenance={'xf_fold': k, 'fit': 'TRAIN + other DEV folds'})
        backoff = BackoffBC.fit(keys, actions, repertoire=vector.repertoire, alphas=alphas)
        rows = np.flatnonzero(folds == k)
        if len(rows):
            b0xf[rows] = vector.batch(dev, rows)
            b1xf_raw[rows] = backoff.batch(dev, rows)
        per_fold.append({'fold': k, 'fit_rows': int(len(actions)), 'dev_fit_rows': int(len(other)),
                         'predicted_rows': int(len(rows))})
        run.guard.check(f'xf-fit:fold{k}')
    calibrator = run.state['calibrated'].get('B1')
    b1xf = None if calibrator is None else calibrator.calibrator.apply(b1xf_raw, b1xf_raw > 0)
    out.npy('xf/pitch_keys.npy', dev.pitch_keys)
    out.npy('xf/folds.npy', folds)
    out.npy('xf/b0_xf.npy', b0xf)
    out.npy('xf/b1_xf_raw.npy', b1xf_raw)
    if b1xf is not None:
        out.npy('xf/b1_xf.npy', b1xf)
    out.json('xf/xf.json', {'arms': ['B0-XF', 'B1-XF'], 'not_run': ['B2-XF'], 'folds': per_fold,
                            'salt': config['seeds']['fold_salts']['dev_xf'],
                            'b1_alphas': {k: token(v) for k, v in alphas.items()},
                            'b1_xf_calibration': 'frozen primary B1 calibrator' if b1xf is not None else
                            'B1 CALIBRATION_FAILED: raw only',
                            'exposure': 'DEV pitch-type labels of the other four folds fit the XF arms (descriptive)',
                            'dev_labels_sha256': released,
                            'dev_label_aggregates': 'reported by score-dev only (P, S_in, S_off, P_plus, TRAIN0)'})
    run.state.update(dev_table=dev, dev_info=info)
    return {'folds': len(per_fold)}


def stage_freeze_xf(run, out):
    run.gate.require_pinned('freeze-primary', 'freeze-xf')
    files = StageDir(run.attempt.stage_path('xf-fit')).artifacts()
    manifest = build_manifest(run.attempt.root, files, name='freeze-xf', groups={
        'XF': {'files': 'xf/', 'input_rows': {'fit': 'TRAIN plus the other four DEV folds', 'predict': 'DEV fold k'}}},
        extra={'config_sha256': run.config_sha256, 'git_commit': run.commit})
    data = canonical_bytes(manifest)
    (out.path / 'manifest.json').write_bytes(data)
    append_chain(run.attempt.root / 'addenda.jsonl', {'addendum': 'freeze-xf', 'manifest_sha256': sha256_bytes(data)})
    run.state['manifest_xf'] = manifest
    return {'manifest_sha256': sha256_bytes(data), 'artifacts': len(manifest['artifacts'])}


def one_shot_guard(destination):
    """One DEV scoring attempt: refuse when any sibling attempt already has a score-dev stage."""
    stage = Attempt(destination).stage_path('score-dev').name
    for sibling in Path(destination).parent.iterdir():
        if sibling.resolve() != Path(destination).resolve() and (sibling / stage).exists():
            raise Refused(f'DEV was already scored in {sibling.name}; no re-scoring under this registration')


def dev_arms(run):
    """Every arm's probabilities on the DEV table rows (frozen models / frozen XF predictions)."""
    dev, inputs = run.state['dev_table'], run.inputs()
    manifest, root = run.state['manifest_xf'], run.attempt.root
    xf = run.attempt.stage_path('xf-fit').name
    keys = load_frozen_npy(root, manifest, f'{xf}/xf/pitch_keys.npy')
    if not np.array_equal(keys, dev.pitch_keys):
        raise IntegrityError('frozen XF rows differ from the DEV table')
    calibrated = run.state['calibrated']
    arms = {'B0': inputs.b0.batch(dev), 'B1_raw': run.state['b1'].batch(dev), 'B2_raw': run.state['b2'].batch(dev),
            'B0-XF': load_frozen_npy(root, manifest, f'{xf}/xf/b0_xf.npy'),
            'B1-XF_raw': load_frozen_npy(root, manifest, f'{xf}/xf/b1_xf_raw.npy')}
    for name in ('B1', 'B2', 'B0-T'):
        if calibrated.get(name) is not None:
            arms[name] = calibrated[name].batch(dev)
    if f'{xf}/xf/b1_xf.npy' in manifest['artifacts']:
        arms['B1-XF'] = load_frozen_npy(root, manifest, f'{xf}/xf/b1_xf.npy')
    return arms


def stage_score_dev(run, out):
    run.gate.require_pinned('freeze-xf', 'score-dev')
    one_shot_guard(run.destination)
    config, dev = run.config, run.state['dev_table']
    thresholds = {'p1': config['positivity']['P1']['threshold'], 'p2': config['positivity']['P2']['threshold']}
    draws, seed = config['seeds']['bootstrap_draws'], config['seeds']['bootstrap']
    arms = dev_arms(run)
    f = dev.frame
    y, games = labels(dev), f.game_pk.to_numpy()
    s_in, s_off = f.S_in.to_numpy(bool), f.S_off.to_numpy(bool)
    p_rows, train0 = f.P.to_numpy(bool), f.TRAIN0.to_numpy(bool)
    summaries = {name: prediction_summary(y[s_in], p[s_in], **thresholds) for name, p in arms.items()}
    paired = {}
    for name in arms:
        if name == 'B0':
            continue
        paired[name] = paired_game_bootstrap(y[s_in], arms[name][s_in], arms['B0'][s_in], games[s_in], draws=draws,
                                             seed=seed, p1_threshold=thresholds['p1'])
        run.guard.check(f'score-dev:{name}')
    family = config['verdict_rule']['eligibility']['E1_nll_superiority']['holm_family']
    raw_p = [(paired.get(c) or {}).get('nll', {}) or {} for c in family]
    holm = holm_adjust([p.get('p_less') for p in raw_p])
    rescued = {}
    for name in ('B0-XF', 'B1-XF'):
        if name in arms:
            off = np.flatnonzero(s_off)
            mass = arms[name][off, y[off]] if len(off) else np.zeros(0)
            hit = mass > 0
            rescued[name] = {'s_off_rows': int(len(off)), 'rescued_rows': int(hit.sum()),
                             'rescued_share': float(hit.mean()) if len(off) else None,
                             'rescued_nll': float(-np.log(mass[hit]).mean()) if hit.any() else None}
    groups = config['self_consistency']['SC1_groups']
    sc1 = {name: {'overall': group_calibration(y[p_rows], p[p_rows], reg.ACTION_CODES, groups),
                  'by_volume': {v: group_calibration(y[p_rows & (f.volume_group == v).to_numpy()],
                                                     p[p_rows & (f.volume_group == v).to_numpy()], reg.ACTION_CODES, groups)
                                for v in ('low', 'middle', 'high')},
                  'by_count': {c: group_calibration(y[p_rows & (f.count_label == c).to_numpy()],
                                                    p[p_rows & (f.count_label == c).to_numpy()], reg.ACTION_CODES, groups)
                               for c in sorted(f.count_label.unique())}}
           for name, p in arms.items()}
    sc3 = {name: ratio_bins(y[p_rows], p[p_rows], config['self_consistency']['SC3_small_probability_ratio']['bins'])
           for name, p in arms.items()}
    strata = strata_metrics(y[s_in], {n: p[s_in] for n, p in arms.items()},
                            {k: f[k].to_numpy()[s_in] for k in ('volume_group', 'count_label', 'role', 'month',
                                                                 'two_strikes', 'runners_on', 'platoon')}, **thresholds)
    train0_summary = {name: prediction_summary(y[train0], p[train0], **thresholds) for name, p in arms.items()}
    counts = {'P': int(p_rows.sum()), 'S_in': int(s_in.sum()), 'S_off': int(s_off.sum()), 'P_plus': int(len(dev)),
              'TRAIN0': int(train0.sum()), 'P_plus_minus_P': int((~p_rows).sum()),
              'S_off_share': float(s_off.sum() / p_rows.sum()) if p_rows.any() else None,
              'S_off_share_by_volume': {v: (float((s_off & (f.volume_group == v).to_numpy()).sum()
                                                   / max((p_rows & (f.volume_group == v).to_numpy()).sum(), 1)))
                                        for v in ('low', 'middle', 'high')}, 'games_S_in': int(len(np.unique(games[s_in])))}
    eligibility = partial_eligibility(config, paired, holm, family, run.state['calibration'])
    result = {'populations': counts, 'dev_table': run.state['dev_info'], 'arms': sorted(arms), 'summaries': summaries,
              'paired_vs_b0': paired,
              'holm': dict(zip(family, holm)), 'xf_rescued': rescued, 'sc1': sc1, 'sc3': sc3, 'strata': strata,
              'train0_P_plus': train0_summary, 'eligibility_E1_E3': eligibility,
              'dev': 'exposed development; one scoring read; no OPE value computed'}
    out.json('score.json', result)
    run.state.update(score=result, arms=arms)
    return {'populations': counts, 'eligibility_E1_E3': eligibility}


def partial_eligibility(config, paired, holm, family, calibration):
    rule = config['verdict_rule']['eligibility']
    out = {}
    for candidate, p_holm in zip(family, holm):
        if calibration[candidate]['status'] == CALIBRATION_FAILED:
            out[candidate] = {'status': CALIBRATION_FAILED}
            continue
        comparison = paired.get(candidate)
        if comparison is None or comparison['status'] != 'measured' or p_holm is None:
            out[candidate] = {'status': 'NOT_EVALUABLE'}
            continue
        nll, ece, p1 = comparison['nll'], comparison['ece'], comparison['p1']
        out[candidate] = {'status': 'EVALUATED', 'E1': bool(nll['delta'] <= rule['E1_nll_superiority']['margin']
                                                           and nll['ci95'][1] < 0
                                                           and p_holm <= rule['E1_nll_superiority']['holm_alpha']),
                          'E2': bool(ece['ci95'][1] <= rule['E2_ece_noninferiority']['margin']),
                          'E3': bool(p1['ci95'][1] <= rule['E3_positivity_noninferiority']['margin']),
                          'nll_delta': nll['delta'], 'nll_ci95': nll['ci95'], 'p_holm': p_holm,
                          'ece_ci95': ece['ci95'], 'p1_ci95': p1['ci95']}
    return out


def decision_rows(run):
    """(request ids, frozen DEV positions) of every evaluated decision of an E0 PA, each checked
    against its frozen DEV row (pitcher, side, count, previous type); computed once."""
    if 'decision_rows' in run.state:
        return run.state['decision_rows']
    dev = run.state['dev_table']
    rows = [row for pa_id, pa_rows in run.state['ledger_pas'] if run.state['e0'][pa_id][0]
            for row in itertools.takewhile(lambda r: r['status'] in EVALUATED_STATUSES, pa_rows)]
    if not rows:
        run.state['decision_rows'] = ([], np.zeros(0, dtype=np.int64))
        return run.state['decision_rows']
    parts = np.array([[int(x) for x in r['context_key'].split(':')] for r in rows], dtype=np.int64)
    if parts.shape != (len(rows), 3):
        raise IntegrityError('S6 context keys must be game:at_bat:pitch')
    positions = dev.positions(pitch_key(parts[:, 0], parts[:, 1], parts[:, 2]))
    if (positions < 0).any():
        raise IntegrityError(f'{int((positions < 0).sum())} S6 decisions have no frozen DEV row')
    frozen = dev.frame.iloc[positions]
    previous = np.array([r['history_actions'][-1] if r['history_actions'] else '<START>' for r in rows], dtype=object)
    same = ((frozen.pitcher.to_numpy(np.int64) == np.array([int(r['pitcher']) for r in rows]))
            & (frozen.batter_side.to_numpy(object) == np.array([r['batter_side'] for r in rows], dtype=object))
            & (frozen.balls.to_numpy(np.int64) == np.array([int(r['balls']) for r in rows]))
            & (frozen.strikes.to_numpy(np.int64) == np.array([int(r['strikes']) for r in rows]))
            & (frozen.previous.to_numpy(object) == previous))
    if not same.all():
        raise IntegrityError(f'{int((~same).sum())} S6 decisions disagree with their frozen DEV rows')
    run.state['decision_rows'] = ([r['request_id'] for r in rows], positions)
    return run.state['decision_rows']


def decision_vectors(run, arm):
    """pi_b_hat_new vectors for every evaluated decision of an E0 PA, via its frozen DEV row."""
    ids, positions = decision_rows(run)
    return dict(zip(ids, run.state['arms'][arm][positions]))


def stage_s6_weights(run, out):
    run.gate.require_pinned('freeze-xf', 's6-weights')
    run.gate.require_passed('score-dev', 's6-weights')
    config = run.config
    pas, e0 = run.state['ledger_pas'], run.state['e0']
    common = {'large': config['ope_weight_diagnostics']['large_weight_threshold']['value'],
              'mei': config['ope_weight_diagnostics']['mei'],
              'ess_min': config['verdict_rule']['eligibility']['E5_thresholds'],
              'draws': config['seeds']['bootstrap_draws'], 'seed': config['seeds']['bootstrap']}
    baseline = weight_diagnostics(pas, e0, **common)
    result = {'B0': public_weights(baseline)}
    for arm in ('B1', 'B2', 'B0-T', 'B0-XF', 'B1-XF'):
        if arm in run.state['arms']:
            result[arm] = public_weights(weight_diagnostics(pas, e0, new_vectors=decision_vectors(run, arm),
                                                            baseline=baseline, **common))
            run.guard.check(f's6-weights:{arm}')
    out.json('weights.json', {'arms': result, 'E5': {k: v['E5'] for k, v in result.items()},
                              'note': 'weights only; rewards, values and OPE results are never read'})
    run.state['weights'] = result
    return {'E5': {k: v['E5']['pass'] for k, v in result.items()}}


def candidate_status(partial, e5, integrity_ok):
    """ELIGIBLE / INELIGIBLE / CALIBRATION_FAILED / NOT_EVALUABLE for one candidate."""
    if partial['status'] == CALIBRATION_FAILED:
        return CALIBRATION_FAILED, {}
    if partial['status'] == 'NOT_EVALUABLE' or e5 is None:
        return 'NOT_EVALUABLE', {}
    criteria = {'E1': partial['E1'], 'E2': partial['E2'], 'E3': partial['E3'], 'E4': bool(integrity_ok),
                'E5': bool(e5)}
    return ('ELIGIBLE' if all(criteria.values()) else 'INELIGIBLE'), criteria


def final_label(integrity_ok, statuses, nll, *, tie_margin, simpler='B1'):
    """Exhaustive label partition (registry.RUN_LABELS minus ABORTED, which only the runner sets)."""
    if not integrity_ok:
        return 'FAILED_INTEGRITY'
    if all(statuses[c] == 'NOT_EVALUABLE' for c in reg.ADOPTION_CANDIDATES):
        return 'NOT_EVALUABLE'
    eligible = [c for c in reg.ADOPTION_CANDIDATES if statuses[c] == 'ELIGIBLE']
    if not eligible:
        return 'KEEP_B0'
    if len(eligible) == 1:
        return f'ADOPT_FOR_2027_REGISTRATION:{eligible[0]}'
    if abs(nll['B1'] - nll['B2']) < tie_margin:  # strict <: practically equal -> the simpler B1
        return f'ADOPT_FOR_2027_REGISTRATION:{simpler}'
    return f'ADOPT_FOR_2027_REGISTRATION:{min(eligible, key=lambda c: nll[c])}'


def stage_decide(run, out):
    config = run.config
    partial, weights = run.state['score']['eligibility_E1_E3'], run.state['weights']
    integrity_ok = run.attempt.passed('s6-integrity') and run.attempt.passed('smoke')
    statuses, criteria = {}, {}
    for candidate in reg.ADOPTION_CANDIDATES:
        e5 = (weights.get(candidate) or {}).get('E5', {}).get('pass') if candidate in weights else None
        statuses[candidate], criteria[candidate] = candidate_status(partial[candidate], e5, integrity_ok)
    nll = {c: (run.state['score']['summaries'].get(c) or {}).get('nll') for c in reg.ADOPTION_CANDIDATES}
    label = final_label(integrity_ok, statuses, nll, tie_margin=config['verdict_rule']['selection']['tie_margin'],
                        simpler=config['verdict_rule']['selection']['simpler'])
    decision = {'label': label, 'candidate_status': statuses, 'criteria': criteria, 'dev_nll': nll,
                'tie_rule': f"both eligible and |NLL_B1 - NLL_B2| < {config['verdict_rule']['selection']['tie_margin']}"
                            ' (strict) -> B1', 'E4_integrity': integrity_ok,
                'descriptive_only': config['verdict_rule']['descriptive_only'],
                'not_claimed': 'no OPE value or Delta; DEV is exposed development; no service change'}
    out.json('decision.json', decision)
    dump_json(run.attempt.root / 'final.json', {'label': label, 'stage': 'decide', 'citable': True,
                                                'config_sha256': run.config_sha256, 'git_commit': run.commit})
    return {'label': label, 'candidate_status': statuses}


STAGE_FUNCTIONS = {'smoke': stage_smoke, 'census': stage_census, 'fit': stage_fit, 'calibrate': stage_calibrate,
                   'freeze-primary': stage_freeze_primary, 's6-integrity': stage_s6_integrity, 'xf-fit': stage_xf_fit,
                   'freeze-xf': stage_freeze_xf, 'score-dev': stage_score_dev, 's6-weights': stage_s6_weights,
                   'decide': stage_decide}


# ---------------------------------------------------------------- real inputs (only after the gate)

class Inputs:
    def __init__(self, *, frame, raw_keys, b0, support, hands, archetypes, ledger, identity):
        self.frame, self.raw_keys, self.b0 = frame, raw_keys, b0
        self.support, self.hands, self.archetypes = support, hands, archetypes
        self.ledger, self.identity = ledger, identity
        self.sealed = SealedLabels(frame)
        self.identity['dev_labels_sealed_sha256'] = self.sealed.sha256


FRAME_COLUMNS = ['game_pk', 'at_bat_number', 'pitch_number', 'game_date', 'split', 'pitcher', 'batter', 'stand',
                 'p_throws', 'balls', 'strikes', 'outs_when_up', 'bases', 'inning', 'inning_topbot', 'home_score',
                 'away_score', 'pitch_type', 'description', 'events', 'is_pa_terminal', 'launch_angle']


class RealSource:
    """Pinned real inputs. Constructed only by ``main`` after ``registration_gate`` passed."""
    def __init__(self, config, local, config_path):
        self.config, self.local, self.config_path = config, local, config_path

    def load(self, run):
        # Defense in depth: the real loader re-runs the registration gate before opening any file.
        if json.loads(Path(self.config_path).read_bytes()) != self.config:
            raise Refused('the config in memory differs from the registered config file')
        registration_gate(self.config, git=git_state(_field(self.config, 'code.commit'), self.config_path),
                          runner_sha256=sha256_file(Path(__file__)))
        validate_native_runtime()
        from pitchmdp.matrix_data import load_verified_processed_cache
        from pitchmdp.policy_artifacts import load_hand_registry, load_support_table
        from pitchmdp.policy_identity import pinned_json, pinned_pickle
        from run_temporal_blend import assign_fold
        config, pins = self.config, reg.DATA_PINS
        bundle = pinned_json(REPO / 'configs' / 'G0-RESEARCH-FROZEN-v1.json', pins['g0_bundle_sha256'])
        files = bundle['files']
        parent = pinned_json(files['p4_parent_preparation']['path'], files['p4_parent_preparation']['sha256'])
        raw = load_verified_processed_cache(self.local)
        if raw.attrs['sequence_data_identity'] != parent['dataset_identity'] or \
                raw.attrs['matrix_source_provenance'] != parent['source_provenance']:
            raise IntegrityError('processed data differs from the G0 parent preparation')
        if raw.attrs['matrix_source_provenance']['processed_sha256'] != pins['processed_sha256']:
            raise IntegrityError('processed sha256 differs from the registered data version')
        frame = raw.loc[raw.game_type.eq('R')]
        del raw
        frame = assign_fold(frame, 2025)
        frame = frame.loc[frame.split.isin(reg.USED_SPLITS), FRAME_COLUMNS]
        frame = frame.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'], kind='stable')
        frame = check_splits(frame.reset_index(drop=True))
        aux = pinned_pickle(files['p4_auxiliary']['path'], files['p4_auxiliary']['sha256'])
        archetypes = aux['context'].archetypes
        artifact = config['data']['b0_artifact']
        b0 = RegisteredBC.load(artifact['path'], artifact['file_sha256'])
        support, support_content = load_support_table(artifact['support_table_path'], artifact['support_table_sha256'],
                                                      b0.artifact)
        hands, hands_content = load_hand_registry(artifact['hands_path'], artifact['hands_sha256'], b0.artifact)
        raw_keys = read_raw_keys(self.local, pins['raw_sha256'], frame)
        s6 = config['data']['s6_ledger']
        ledger = {'path': s6['path'], 'manifest_path': str(Path(s6['path']).parent / 'manifest.json'),
                  'ledger_sha256': s6['ledger_sha256'], 'manifest_sha256': s6['manifest_sha256'],
                  'runtime_sha256': s6['runtime_sha256']['value']}
        identity = {'processed_sha256': pins['processed_sha256'], 'g0_bundle_sha256': pins['g0_bundle_sha256'],
                    'g0_auxiliary_sha256': files['p4_auxiliary']['sha256'],
                    'b0': {'file_sha256': b0.artifact.file_sha256, 'bc_sha256': b0.artifact.sha256,
                           'support_file_sha256': artifact['support_table_sha256'], 'support_content_sha256': support_content,
                           'hands_file_sha256': artifact['hands_sha256'], 'hands_content_sha256': hands_content},
                    'raw_sha256': pins['raw_sha256'], 'rows_by_split': frame.split.value_counts().sort_index().to_dict(),
                    's6_ledger': {k: v for k, v in ledger.items() if k != 'path'}}
        return Inputs(frame=frame, raw_keys=raw_keys, b0=b0, support=support, hands=hands, archetypes=archetypes,
                      ledger=ledger, identity=identity)


def read_raw_keys(local, pins, frame):
    """Keys/date/pitcher of the pinned 2023-2025 raw Statcast files (run_ml_direct_workload.read_raw),
    restricted to the frame's games; the bytes parsed are the bytes hashed."""
    names = list(pins)
    if list(local['raw_allowlist']) != names:
        raise IntegrityError('raw allowlist must be exactly the pinned 2023-2025 Statcast files')
    parts, games = [], set(frame.game_pk.unique().tolist())
    for name in names:
        payload = (Path(local['raw_root']) / name).read_bytes()
        if sha256_bytes(payload) != pins[name]:
            raise IntegrityError(f'raw Statcast file differs from its pin: {name}')
        part = pd.read_parquet(io.BytesIO(payload), columns=['game_date', 'game_pk', 'at_bat_number', 'pitch_number',
                                                            'pitcher'])
        del payload
        parts.append(part.loc[part.game_pk.isin(games)])
    raw = pd.concat(parts, ignore_index=True)
    raw['game_date'] = pd.to_datetime(raw.game_date)
    if (raw.game_date > pd.Timestamp(reg.LAST_DATE)).any():
        raise IntegrityError('raw workload rows after 2025-09-30')
    return raw


# ---------------------------------------------------------------- CLI

def preflight_tests():
    """Leakage and synthetic tests must pass before any real-data stage (E4)."""
    done = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *PREFLIGHT_TESTS],
                          cwd=PROJECT, capture_output=True, text=True, timeout=1800)
    summary = (done.stdout.strip().splitlines() or [''])[-1]
    if done.returncode != 0:
        raise Refused(f'synthetic/leakage tests failed: {summary}')
    return {'returncode': done.returncode, 'summary': summary, 'files': list(PREFLIGHT_TESTS)}


class heavy_lock:
    """Common heavy lock, non-blocking (another heavy job -> refuse)."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.stream = self.path.open('a+b')
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            raise Refused(f'another heavy job holds {self.path}') from error
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--local-config', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('command', choices=('smoke', 'run'))
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_bytes())
    destination = None
    try:
        runner_sha = sha256_file(Path(__file__))
        commit = _field(config, 'code.commit')
        registration_gate(config, git=git_state(commit, config_path), runner_sha256=runner_sha)
        validate_native_runtime()
        destination = check_destination(args.destination, output_root=config['runner']['output_root'])
        guard = Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST)
        guard.start_run()
        share = free_share()
        if share < config['cost_budget']['memory']['start_requires_free_share_min']:
            raise Refused(f'memory_pressure free share {share:.2f} below the registered start minimum')
        with heavy_lock(config['runner']['heavy_lock']):
            if args.command == 'smoke' and destination.exists():
                raise Refused('the smoke opens a new attempt directory; it already exists')
            if args.command == 'run' and not destination.exists():
                raise Refused('run continues an attempt whose smoke passed; the attempt directory is missing')
            destination.parent.mkdir(parents=False, exist_ok=True)  # <ML-MATRIX-20260924>/EXP-BC-CAL-01
            destination.mkdir(parents=False, exist_ok=args.command == 'run')
            tests = preflight_tests()
            git = git_state(commit, config_path)
            dump_json(destination / f'preflight-{args.command}.json', {'tests': tests, 'free_share': share, 'git': git,
                                                                       'native_runtime': environment()})
            local = json.loads(Path(args.local_config).read_bytes())
            run = Run(config, destination=destination, source=RealSource(config, local, config_path), guard=guard,
                      commit=git['head'], output_root=config['runner']['output_root'])
            stages = ('smoke',) if args.command == 'smoke' else reg.STAGES[1:]
            for stage in stages:
                status = execute_stage(run, stage)
                print(json.dumps({'stage': stage, 'status': status}), flush=True)
                if status != 'PASS':
                    return 3 if status == 'ABORTED' else 2 if status in ('FAILED_INTEGRITY', 'REFUSED') else 1
    except Refused as refusal:
        print(json.dumps({'refused': str(refusal)}), file=sys.stderr, flush=True)
        if destination is not None and destination.is_dir():  # the refusal is recorded in the attempt
            append_chain(destination / 'refusals.jsonl', {'command': args.command, 'refused': str(refusal),
                                                          'utc': datetime.now(timezone.utc).isoformat()})
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
