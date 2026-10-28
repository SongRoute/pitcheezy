"""EXP-BC-CAL-01 (track A3): a better-calibrated behavior policy pi_b_hat for the OPE weights.

Fail closed. Every real-data stage, the smoke included, is refused unless the config passes the
registered contract (``behavior_calibration_registry.config_check``) AND ``registration_gate``:
status REGISTERED with every flag true, every pin of the registry's immutable registration schema
present and well formed (the S6 header's pins.candidate.sha256 equal to the sealed c13cc989..., the
header runtime identity pinned from the header), the config content sha256
and the runner sha256 pinned, no change since the pinned commit inside the reviewed scope
(experiments/, src/, scripts/, pyproject.toml), the config committed at HEAD, one OpenMP runtime and
the pinned research-a3 dependency. The gate is re-run before EVERY stage and again by the loader
before it opens any file (only inside an executing stage of an open attempt). Order is fixed: smoke
(first registered step: TRAIN rows only, cost and integrity incl. a split-aligned processed layout,
no metric) -> census (label-free) -> fit -> calibrate -> freeze-primary -> s6-integrity -> xf-fit ->
freeze-xf -> score-dev -> s6-weights -> decide. A stage opens only after every earlier stage passed in
the same attempt (verified against the digests anchored in stages.jsonl, under the guard). The
processed input is the split-aligned copy (data.a3_input) written once before registration by
scripts/prepare_behavior_calibration_input.py: its identity manifest and sha256 are registration pins
and the reader refuses a file whose row groups span a split date. Data are
read stage by stage (pitchmdp.behavior_data): no guarded column chunk of a DEV-bearing row group is
passed to pyarrow before freeze-primary is pinned and s6-integrity passed (checked from the
pyarrow-level read log at freeze-primary); every acquisition is logged before it happens (reads.jsonl
and the stage record, decoded-then-dropped rows of straddling row groups counted); no DEV metric is
computed before freeze-xf is pinned. Each stage writes JSON (+ npy/text artifacts) and SHA256SUMS
under the attempt directory (never the repository results/), with seeds, config sha256, git commit,
seconds and peak RSS. Caps: total wall 14,400 s over the whole attempt (smoke through decide,
budget.jsonl, bound to every stage record; a missing or truncated history refuses), stage hang guard
7,200 s, B2 fit soft cap 3,600 s (warning only), RSS 10 GiB for the runner peak RSS plus the
supervised child tree (ps); the protection window 06:30-13:30 KST is never entered (refuse to start
inside it; the memory probe, the preflight tests, inter-stage verification and every stage abort
when it begins, keeping their outputs). Every stage is sealed (stage.json, SHA256SUMS, chain record)
inside the failure handler, a PASS under the guard. Every unsuccessful attempt ends with a non-citable
final.json (REFUSED / FAILED_RUNTIME / ABORTED / FAILED_INTEGRITY); the citable final.json is the
last write of a complete attempt, by publish_final, after decide is sealed, the command budget is
closed and a final guard check passed.

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
from dataclasses import dataclass, field
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
from pitchmdp.behavior_data import DEV_START, TRAIN_END, A3Inputs, ProcessedReader, RawKeysReader, ReadLog
from pitchmdp.behavior_features import (FeatureContext, census_counts, game_fold, guard_dates, population_masks,
                                        split_table, style_tables, workload_features)
from pitchmdp.behavior_metrics import (group_calibration, holm_adjust, paired_game_bootstrap, prediction_summary,
                                       ratio_bins, strata_metrics)
from pitchmdp.behavior_model import (BackoffBC, Calibrated, Keys, LGBMBehavior, RegisteredBC, RepertoireSupport,
                                     VectorBC, alpha_from_token, arrays_sha256, pitch_key, trailing_bits)
from pitchmdp.behavior_weights import (EVALUATED as EVALUATED_STATUSES, by_pa, e0_from_ledger,
                                       public as public_weights, r1b_logging, r2_reproduction, read_ledger,
                                       weight_diagnostics)
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.policy_artifacts import IntegrityError
from pitchmdp.rollout_policy import PAState, PastPitch

SELF = 'experiments/pitchmdp/scripts/run_behavior_calibration.py'
KST = timezone(timedelta(hours=9))
PREFLIGHT_TESTS = ('tests/test_behavior_calibration_registry.py', 'tests/test_behavior_calibration_model.py',
                   'tests/test_behavior_calibration_features.py', 'tests/test_behavior_calibration_calibration.py',
                   'tests/test_behavior_calibration_metrics.py', 'tests/test_behavior_calibration_weights.py',
                   'tests/test_behavior_calibration_data.py', 'tests/test_behavior_calibration_input.py',
                   'tests/test_behavior_calibration_runner.py')
ARTIFACT_EXCLUDE = ('stage.json', 'SHA256SUMS')
SINCE_DEV, SINCE_AFTER_TRAIN = str(DEV_START.date()), str((TRAIN_END + pd.Timedelta(days=1)).date())


class Refused(Exception):
    """Registration, destination, stage-order or launch refusal. ``terminal`` refusals close the attempt
    (final.json REFUSED); transient ones (window, busy lock, memory) are only recorded."""
    def __init__(self, message, terminal=False):
        super().__init__(message)
        self.terminal = terminal


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


def sha256_file(path, check=None):
    """sha256 of a file; ``check`` (the guard) runs before every 8 MiB chunk."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if check is not None:
                check()
            chunk = stream.read(8 << 20)
            if not chunk:
                break
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


def git_state(commit, config_path=None, scope=reg.REVIEWED_CODE_SCOPE):
    """HEAD, uncommitted changes and changes since ``commit`` inside the reviewed scope, and whether
    the config file is committed at HEAD byte for byte."""
    run = lambda *args: subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
    state = {'head': run('rev-parse', 'HEAD'), 'scope': list(scope),
             'dirty': bool(run('status', '--porcelain', '--', *scope)), 'commit_is_ancestor': None,
             'changed_since_commit': None, 'config_committed': None}
    if commit:
        state['commit_is_ancestor'] = subprocess.call(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO,
                                                      stderr=subprocess.DEVNULL) == 0
        if state['commit_is_ancestor']:
            state['changed_since_commit'] = [x for x in run('diff', '--name-only', commit, 'HEAD', '--',
                                                            *scope).splitlines() if x]
    if config_path is not None:
        path = Path(config_path).resolve()
        try:
            blob = subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(REPO).as_posix()}'], cwd=REPO,
                                           stderr=subprocess.DEVNULL)
            state['config_committed'] = blob == path.read_bytes()
        except (ValueError, subprocess.CalledProcessError):
            state['config_committed'] = False
    return state


def read_digest_manifest(config):
    """The rows of the external, append-only digest manifest named by registration.digest_manifest_path
    (hash-chained, outside the repository); refuses a missing, in-repository or broken manifest."""
    path = _field(config, 'registration.digest_manifest_path')
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise Refused('registration.digest_manifest_path must be an absolute path', terminal=True)
    path = Path(path).resolve()
    if REPO.resolve() in path.parents or any((parent / '.git').exists() for parent in path.parents) \
            or not path.is_file():
        raise Refused(f'the digest manifest {path} must be an existing file outside every repository', terminal=True)
    return read_chain(path)


def registration_gate(config, *, git, runner_sha256, digest_entries):
    """Refuse unless EXP-BC-CAL-01 is registered and execution-enabled with every pin of the registry's
    immutable schema present and well formed, its code commit and config digest are not those of a void
    registration (bug_fix_retries), and the external digest manifest (``digest_entries``) holds one original
    registration entry then exactly the re-registration entries of the retry records, the last one being
    this config at this commit (``digest_entries``: the manifest rows, or a callable reading them for the
    config; a reading failure is one more refusal reason). Runs before any data is touched and before
    every stage."""
    try:
        reg.config_check(config)
    except reg.RegistrationError as error:
        raise Refused(f'config deviates from the registered contract: {error}', terminal=True) from error
    reasons = []
    if not (config.get('status') == 'REGISTERED' and config.get('registered') is True
            and config.get('execute') is True and config.get('real_data_enabled') is True):
        reasons.append(f'status {config.get("status")!r} is not REGISTERED with execute/real_data_enabled true')
    reasons += reg.registration_pin_errors(config)
    if _field(config, 'code.config_sha256') != config_content_sha256(config):
        reasons.append('code.config_sha256 differs from the config content')
    if _field(config, 'code.runner_sha256') != runner_sha256:
        reasons.append('code.runner_sha256 differs from this runner')
    commit = _field(config, 'code.commit')
    if git.get('scope') != list(reg.REVIEWED_CODE_SCOPE):
        reasons.append('git state was not computed over the reviewed code scope')
    if not git.get('commit_is_ancestor') or git.get('changed_since_commit'):
        reasons.append(f'code at HEAD differs from the registered commit {commit} inside the reviewed scope')
    if git.get('dirty'):
        reasons.append('uncommitted changes inside the reviewed scope (experiments/, src/, scripts/, pyproject.toml)')
    if git.get('config_committed') is not True:
        reasons.append('the config file is not committed at HEAD byte for byte')
    void_commits, void_digests = reg.void_registrations(config)
    if commit in void_commits:
        reasons.append('code.commit is a commit of a void registration (bug_fix_retries): register the fixed commit')
    if config_content_sha256(config) in void_digests:
        reasons.append('the config digest is that of a void registration (bug_fix_retries)')
    try:
        rows = digest_entries(config) if callable(digest_entries) else digest_entries
        active = reg.reconcile_registration_chain(rows or [], reg.retry_records(config))
    except (reg.RegistrationError, Refused, IntegrityError, OSError, ValueError) as error:
        reasons.append(f'digest manifest: {error}')
    else:
        if active.get('config_content_sha256') != config_content_sha256(config):
            reasons.append('the active digest-manifest entry is not this config (config_content_sha256)')
        if active.get('commit') != commit:
            reasons.append('the active digest-manifest entry was registered at another commit')
    if reasons:
        raise Refused('registration gate: ' + '; '.join(reasons), terminal=True)
    return True


def dependency_pins(pyproject=REPO / 'pyproject.toml'):
    """The research-a3 group of pyproject.toml pins the registered dependency versions, and the installed
    versions equal them (importlib.metadata; LightGBM is never imported just to be checked)."""
    import importlib.metadata
    import tomllib
    data = Path(pyproject).read_bytes()
    group = tomllib.loads(data.decode())['project']['optional-dependencies'].get(reg.DEPENDENCY_GROUP)
    pinned = dict(entry.split('==', 1) for entry in group or [] if '==' in entry)
    if pinned != reg.DEPENDENCY_PINS:
        raise Refused(f'pyproject {reg.DEPENDENCY_GROUP} pins {pinned} differ from {reg.DEPENDENCY_PINS}')
    installed = {}
    for name, version in pinned.items():
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    if installed != pinned:
        raise Refused(f'installed {installed} differ from the pinned {pinned}')
    return {'pyproject_sha256': sha256_bytes(data), 'group': reg.DEPENDENCY_GROUP, 'pinned': pinned,
            'installed': installed}


def validate_native_runtime():
    """run_ml_benchmark.validate_native_runtime rule: torch (pulled in by pitchmdp.model) and
    LightGBM must share ONE OpenMP runtime; the unsafe duplicate override is refused. (A launch
    condition the operator can correct: the refusal is recorded, the attempt stays open.)"""
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


def native_and_dependencies():
    validate_native_runtime()
    return dependency_pins()


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
    if path.name == 'inputs' or 'inputs' in path.parent.parts[-1:]:
        raise Refused('the inputs/ folder holds the prepared A3 input; an attempt never uses it')
    if output_root is not None and path.parent != Path(output_root).expanduser().resolve():
        raise Refused(f'destination must be a new directory directly under {output_root}')
    return path


# ---------------------------------------------------------------- caps, clock, attempt budget, children

def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == 'darwin' else raw * 1024)


def _minutes(text):
    hours, minutes = (int(x) for x in text.split(':'))
    return hours * 60 + minutes


class Guard:
    """Cooperative caps (loops, LightGBM callbacks, stage ends, supervised children) plus a SIGALRM
    backstop for the time limits inside stages. ``consumed`` = attempt seconds already spent by earlier
    commands (budget.jsonl); the total cap covers the whole attempt. RSS is the process peak."""
    def __init__(self, caps, window, *, clock=time.monotonic, now=None, rss=peak_rss_bytes, alarm=True, consumed=0.):
        self.total_cap, self.stage_cap = caps['total_wall_cap_seconds'], caps['stage_hang_guard_seconds']
        self.rss_cap, self.b2_soft_cap = caps['abort_rss_bytes'], caps['b2_fit_soft_cap_seconds']
        self.window = (_minutes(window['start']), _minutes(window['end']))
        self.clock, self.now = clock, now or (lambda: datetime.now(KST))
        self.rss, self.alarm, self.consumed = rss, alarm, float(consumed)
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

    def elapsed(self):
        return 0. if self.run_start is None else self.clock() - self.run_start

    def remaining(self):
        return self.total_cap - self.consumed - self.elapsed()

    def start_run(self):
        if self.in_window():
            raise Refused('inside the recorder/publisher protection window 06:30-13:30 KST; refusing to start')
        if self.consumed >= self.total_cap:
            raise Abort('TOTAL_WALL_CAP', 'attempt budget already consumed')
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
        remaining = min(self.stage_cap - (self.clock() - self.stage_start), self.remaining(), self.seconds_to_window())
        signal.setitimer(signal.ITIMER_REAL, max(remaining, 0.05))

    def _on_alarm(self, signum, frame):
        if not self.armed:
            return
        self.check(f'{self.stage_name}:alarm')
        self._arm()

    def disarm(self):
        """Stop the alarm but keep the stage clock: cooperative checks still apply the stage cap."""
        self.armed = False
        if self.alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def stop_stage(self):
        self.disarm()
        self.stage_start = None

    def check(self, where=''):
        if self.in_window():
            raise Abort('PROTECTION_WINDOW', where)
        if self.run_start is not None and self.remaining() < 0:
            raise Abort('TOTAL_WALL_CAP', where)
        if self.stage_start is not None and self.clock() - self.stage_start > self.stage_cap:
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


def process_tree_rss(root_pid):
    """{pid: RSS bytes} of ``root_pid`` and every descendant, found through parent pids (``ps``; RSS in
    KiB), so children that started their own session are included. None when ``ps`` fails."""
    try:
        done = subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,rss='], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    children, rss = {}, {}
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            pid, ppid, kib = (int(p) for p in parts)
            children.setdefault(ppid, []).append(pid)
            rss[pid] = kib * 1024
    tree, stack = {}, [root_pid]
    while stack:
        pid = stack.pop()
        if pid in tree or pid not in rss:
            continue
        tree[pid] = rss[pid]
        stack.extend(children.get(pid, ()))
    return tree


def _alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _stop_child(process):
    """Stop the runner-owned child, its process group and every descendant (collected first); no other
    process is signalled."""
    descendants = [pid for pid in (process_tree_rss(process.pid) or {}) if pid != process.pid]
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        for pid in descendants:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        if not any(_alive(pid) for pid in descendants):
            break
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def supervised(command, guard, *, timeout, where, cwd=None, env=None, poll=0.5, tree=process_tree_rss):
    """Run a runner-owned child under the guard: every ``poll`` seconds the protection window, the
    attempt budget and the memory rule are checked; on any of them (or ``timeout``) the child tree is
    stopped. Memory rule: runner peak RSS + current RSS of the whole child tree <= the registered cap
    (the peak child-tree RSS is returned)."""
    started = time.monotonic()
    process = subprocess.Popen(list(command), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    peak = 0
    try:
        while True:
            try:
                out, err = process.communicate(timeout=poll)
                return {'returncode': process.returncode, 'stdout': out, 'stderr': err,
                        'seconds': time.monotonic() - started, 'peak_tree_rss_bytes': peak}
            except subprocess.TimeoutExpired:
                pass
            sizes = tree(process.pid)
            if sizes is None:
                raise Refused(f'{where}: the memory of the child process tree cannot be measured')
            child = int(sum(sizes.values()))
            peak = max(peak, child)
            guard.check(where)
            if guard.rss() + child > guard.rss_cap:
                raise Abort('RSS_CAP', f'{where} (runner {guard.rss()} B + child tree {child} B)')
            if time.monotonic() - started > timeout:
                raise Refused(f'{where} exceeded its {timeout} s timeout')
    except BaseException:
        _stop_child(process)
        raise


def parse_free_share(text):
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text or '')
    if not match:
        raise Refused('memory_pressure free percentage unavailable')
    return int(match.group(1)) / 100


BUDGET = 'budget.jsonl'


def consumed_seconds(root):
    """Attempt seconds spent by earlier commands. Fail closed: a command without its end record (a
    killed process) leaves the budget unknown; an attempt that already ran a stage needs a nonempty
    budget history that every stage record is bound to (the command start it ran under) and that is
    not shorter than the attempt seconds those records hold (missing, empty, truncated or replaced
    histories refuse; a fresh budget is never granted)."""
    root = Path(root)
    rows, attempt = read_chain(root / BUDGET), Attempt(root)
    stage_rows = attempt.records()
    if attempt.started() and not rows:
        raise Refused('the attempt already ran a stage but its budget history is missing or empty', terminal=True)
    total, open_start, starts = 0., None, set()
    for row in rows:
        if row.get('event') == 'start':
            if open_start is not None:
                raise Refused('attempt budget unknown: a command has no end record', terminal=True)
            before = _duration(row.get('consumed_before'), 'consumed_before')
            if abs(before - total) > 1e-6:
                raise Refused(f'budget history is not monotone: a command started at {before} s after {total} s '
                              'were spent', terminal=True)
            open_start = row
            starts.add(row['sha256'])
        elif row.get('event') == 'end':
            if open_start is None or row.get('start_sha256') != open_start['sha256']:
                raise IntegrityError('budget.jsonl end record does not close the open command')
            total += _duration(row.get('seconds'), 'seconds')
            open_start = None
        else:
            raise Refused(f'malformed budget record (event {row.get("event")!r})', terminal=True)
    if open_start is not None:
        raise Refused('attempt budget unknown: a command has no end record', terminal=True)
    unbound = [r['stage'] for r in stage_rows if r.get('budget_start_sha256') not in starts]
    if unbound:
        raise Refused(f'stage records {unbound} are not bound to a command of budget.jsonl (history truncated or '
                      'replaced)', terminal=True)
    recorded = [_duration(r.get('attempt_seconds'), 'attempt_seconds') for r in stage_rows]
    if any(b < a - 1e-6 for a, b in zip(recorded, recorded[1:])):
        raise Refused('stage records hold attempt seconds that decrease (not monotone)', terminal=True)
    if recorded and total + 1e-6 < recorded[-1]:
        raise Refused(f'budget history ({total:.1f} s) is shorter than the stage records ({recorded[-1]:.1f} s)',
                      terminal=True)
    return total


def _duration(value, name):
    """A budget duration: a finite, nonnegative JSON number (not a boolean, not a string)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise Refused(f'malformed budget duration {name}={value!r} (finite nonnegative number required)',
                      terminal=True)
    return float(value)


def begin_command(root, command, consumed):
    return append_chain(Path(root) / BUDGET, {'event': 'start', 'command': command, 'consumed_before': consumed,
                                              'utc': datetime.now(timezone.utc).isoformat()})


def end_command(root, start, seconds):
    return append_chain(Path(root) / BUDGET, {'event': 'end', 'command': start['command'], 'start_sha256': start['sha256'],
                                              'seconds': float(seconds), 'utc': datetime.now(timezone.utc).isoformat()})


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
        try:
            row = json.loads(line)
        except ValueError as error:
            raise IntegrityError(f'{Path(path).name}: unreadable row {seq} (truncated or edited)') from error
        if not isinstance(row, dict):
            raise IntegrityError(f'{Path(path).name}: row {seq} is not an object')
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


def sha256sums(directory, check=None):
    directory = Path(directory)
    # macOS AppleDouble sidecars (._name with the 00051607 magic; the exFAT T7 volume writes them beside
    # files, also after sealing) are metadata, never artifacts; a ._ file without the magic is listed
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and not is_appledouble(p))
    return ''.join(f'{sha256_file(p, check)}  {p.relative_to(directory).as_posix()}\n' for p in files)


def verify_sha256sums(directory, check=None):
    directory = Path(directory)
    listed = (directory / 'SHA256SUMS').read_text() if (directory / 'SHA256SUMS').exists() else None
    return listed is not None and listed == sha256sums(directory, check)


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
        return sorted(p for p in self.path.rglob('*')
                      if p.is_file() and p.name not in ARTIFACT_EXCLUDE and not is_appledouble(p))


class Attempt:
    """One attempt directory: numbered stage directories, the stages / addenda / budget / reads chains.
    ``check`` (the guard) runs inside every verification hash loop."""
    def __init__(self, root, check=None):
        self.root, self.check = Path(root), check

    def stage_path(self, stage):
        return self.root / f'{reg.STAGES.index(stage):02d}-{stage}'

    def records(self):
        return read_chain(self.root / 'stages.jsonl')

    def record(self, stage):
        rows = [r for r in self.records() if r['stage'] == stage]
        return rows[-1] if rows else None

    def status(self, stage):
        row = self.record(stage)
        return None if row is None else row['status']

    def closed(self):
        return (self.root / 'final.json').exists()

    def started(self):
        """Any stage ran (a record or a stage directory exists)."""
        return (self.root / 'stages.jsonl').exists() or any(self.stage_path(s).exists() for s in reg.STAGES)

    def passed(self, stage):
        """PASS in the chain, and the stage directory still matches the digests anchored in the chain:
        stage.json and SHA256SUMS first, then every file listed in SHA256SUMS."""
        row = self.record(stage)
        if row is None or row['status'] != 'PASS':
            return False
        path = self.stage_path(stage)
        for name, key in (('stage.json', 'stage_json_sha256'), ('SHA256SUMS', 'sha256sums_sha256')):
            if not (path / name).is_file() or sha256_file(path / name, self.check) != row.get(key):
                raise IntegrityError(f'stage {stage}: {name} differs from the digest anchored in stages.jsonl')
        if not verify_sha256sums(path, self.check):
            raise IntegrityError(f'stage {stage}: files no longer match SHA256SUMS')
        if json.loads((path / 'stage.json').read_text()).get('status') != 'PASS':
            raise IntegrityError(f'stage {stage}: stage.json status differs from the chain')
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


def write_final(root, *, outcome, stage, reason, config_sha256, citable=False, extra=None):
    """Terminal attempt record (TERMINAL_FAILURES for unsuccessful attempts, a decision label otherwise)."""
    if outcome not in reg.RUN_LABELS:
        raise IntegrityError(f'unregistered terminal outcome {outcome}')
    dump_json(Path(root) / 'final.json', {'label': outcome, 'outcome': outcome, 'stage': stage, 'reason': reason,
                                          'citable': citable, 'config_sha256': config_sha256, **(extra or {})})


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
    def __init__(self, config, *, destination, source, guard, commit):
        self.config, self.destination = config, Path(destination)
        self.config_sha256 = config_content_sha256(config)
        self.source, self.guard, self.commit = source, guard, commit
        self.attempt = Attempt(self.destination, check=lambda: self.guard.check('verification'))
        self.gate = Gate(self.attempt)
        self.state = {}
        self._inputs = None
        self.current_stage = None  # set by execute_stage while a stage runs (the loader refuses otherwise)
        self.budget_start = None  # sha256 of this command's budget.jsonl start record (launch)
        self.log = ReadLog(sink=self._persist_read)  # every acquisition, persisted event by event

    def _persist_read(self, event):
        append_chain(self.destination / 'reads.jsonl', event)

    def inputs(self):
        if self._inputs is None:
            self._inputs = self.source.open(self)
        return self._inputs

    def reads(self, stage):
        return self.log.for_phase(stage)

    def seeds(self):
        return self.config['seeds']

    def require_order(self, stage):
        index = reg.STAGES.index(stage)
        if self.attempt.closed():
            raise Refused('this attempt is closed (final.json exists); register a new attempt')
        if self.attempt.status(stage) is not None or self.attempt.stage_path(stage).exists():
            raise Refused(f'{stage} already exists in this attempt; register a new attempt')
        if index and not self.attempt.passed(reg.STAGES[index - 1]):
            raise Refused(f'{stage} needs a passed {reg.STAGES[index - 1]} first (fixed stage order)')
        failed = [r['stage'] for r in self.attempt.records() if r['status'] != 'PASS']
        if failed:
            raise Refused(f'an earlier stage did not pass ({failed}); no later stage runs in this attempt')
        for earlier in reg.STAGES[:index]:
            if not self.attempt.passed(earlier):  # every earlier stage against its chain-anchored digests
                raise Refused(f'{stage} needs every earlier stage passed ({earlier} is not)')
            record = json.loads((self.attempt.stage_path(earlier) / 'stage.json').read_text())
            if record.get('config_sha256') != self.config_sha256:
                raise Refused(f'the config changed since {earlier} ran in this attempt; register a new attempt')


def _outcome(exc):
    """(status, error) of an exception raised by a stage."""
    if isinstance(exc, Abort):
        return 'ABORTED', {'reason': exc.reason, 'where': exc.where}
    if isinstance(exc, KeyboardInterrupt):
        return 'ABORTED', {'reason': 'INTERRUPTED'}
    if isinstance(exc, Refused):
        return 'REFUSED', {'detail': str(exc)}
    if isinstance(exc, IntegrityError):
        return 'FAILED_INTEGRITY', {'detail': str(exc)}
    return 'FAILED_RUNTIME', {'detail': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}


def seal_stage(run, stage, path, out, record, *, guarded):
    """stage.json, SHA256SUMS and the stage chain record; the chain record is the last write, and a stage
    without it never counts as passed. ``guarded`` (a PASS record): the guard runs before and inside every
    hash loop, so a cap or window crossing while sealing raises and the stage cannot pass. Unguarded (a
    record that is already non-citable): the outputs are preserved; checksums are skipped once a cap or
    the protection window is reached."""
    check = lambda: run.guard.check(f'{stage}:sealing')
    if guarded:
        check()
    out.json('stage.json', record)
    try:
        out.text('SHA256SUMS', sha256sums(path, check))
        sums_sha256 = sha256_file(path / 'SHA256SUMS', check)
    except Abort:
        if guarded:
            raise
        (path / 'SHA256SUMS').unlink(missing_ok=True)
        out.json('stage.json', {**record, 'checksums': 'skipped: a cap or the protection window was reached'})
        sums_sha256 = None
    row = {'stage': stage, 'status': record['status'], 'stage_json_sha256': sha256_file(path / 'stage.json'),
           'sha256sums_sha256': sums_sha256, 'budget_start_sha256': run.budget_start,
           'attempt_seconds': record['budget']['attempt_seconds_at_end']}
    if guarded:
        check()
    append_chain(run.attempt.root / 'stages.jsonl', row)


def execute_stage(run, stage):
    """Verify the earlier stages (under the guard), revalidate the registration, run one stage, then seal
    it (stage.json, SHA256SUMS, chain record bound to this command's budget record) inside the failure
    handler; a PASS is sealed under the guard. A non-PASS stage, or a failure / cap or window crossing
    while sealing a PASS (then ABORTED or FAILED_RUNTIME), closes the attempt with a non-citable
    final.json. A passed decide stage writes no final.json: the citable outcome is published last by
    ``publish_final``."""
    run.guard.check(f'{stage}:verification')
    run.require_order(stage)  # hashes every earlier stage; the guard runs inside each hash loop
    path = run.attempt.stage_path(stage)
    path.mkdir(parents=True, exist_ok=False)
    out = StageDir(path)
    started, t0 = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    try:
        status, error, summary, revalidation = 'PASS', None, None, None
        try:
            try:
                run.current_stage = stage
                revalidation = run.source.revalidate(run, stage)
                run.guard.check(f'{stage}:revalidation')
                run.guard.start_stage(stage)
                summary = STAGE_FUNCTIONS[stage](run, out)
                run.guard.check(f'{stage}:end')
            finally:
                run.current_stage = None
        except (Abort, KeyboardInterrupt, Exception) as exc:  # a bug is recorded as FAILED_RUNTIME
            status, error = _outcome(exc)
        return _finish_stage(run, stage, path, out, status, error, summary, revalidation, started, t0)
    finally:
        run.guard.stop_stage()  # only now: the stage clock and alarm stay on through the sealing of a PASS


def _finish_stage(run, stage, path, out, status, error, summary, revalidation, started, t0):
    """Seal one stage. A PASS is sealed with the stage clock, the alarm, the total budget, the RSS cap
    and the window all still in force: a breach while sealing is a non-citable ABORTED (a failure a
    FAILED_RUNTIME), never a cap reset or bypass. A record that is already non-citable is preserved with
    the alarm disarmed but the stage clock kept, so any cap (the stage cap included) or the window skips
    its checksums; the abort reason is kept in the record and in final.json."""
    record = {'experiment': reg.EXPERIMENT_ID, 'stage': stage, 'status': status, 'error': error, 'citable': False,
              'budget': {'command_start_sha256': run.budget_start,
                         'attempt_seconds_at_end': run.guard.consumed + run.guard.elapsed()}}
    try:
        if status != 'PASS':
            run.guard.disarm()
        attempt_seconds = run.guard.consumed + run.guard.elapsed()
        record.update({'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
                       'seconds': time.perf_counter() - t0, 'peak_rss_bytes': run.guard.rss(), 'seeds': run.seeds(),
                       'config_sha256': run.config_sha256, 'git_commit': run.commit, 'summary': summary,
                       'warnings': list(run.guard.warnings), 'citable': status == 'PASS' and stage != 'decide',
                       'revalidation': revalidation,
                       'budget': {'command_start_sha256': run.budget_start, 'attempt_seconds_at_end': attempt_seconds}})
        if stage == 'decide':
            record['citable_outcome'] = 'final.json, published last by publish_final (never by this record)'
        run.guard.warnings.clear()
        record.update(reads=run.reads(stage), read_summary=run.log.summary([stage]), environment=environment())
        seal_stage(run, stage, path, out, record, guarded=status == 'PASS')
    except (Abort, KeyboardInterrupt, Exception) as exc:  # sealing failed or crossed a cap / the window
        run.guard.disarm()  # no alarm while the non-citable record is preserved; the stage clock is kept
        sealed = 'ABORTED' if isinstance(exc, (Abort, KeyboardInterrupt)) else 'FAILED_RUNTIME'
        status, error = (sealed if status == 'PASS' else status), {'stage': error, 'sealing': _outcome(exc)[1]}
        record.update(status=status, error=error, citable=False)
        try:  # preserve a non-citable record of the stage (best effort; final.json below closes the attempt)
            seal_stage(run, stage, path, out, record, guarded=False)
        except (Abort, KeyboardInterrupt, Exception):
            pass
    if status != 'PASS':
        write_final(run.attempt.root, outcome=status, stage=stage, reason=error, config_sha256=run.config_sha256)
    return status


def publish_final(run):
    """The citable attempt outcome, written last: only when the decide stage is sealed as PASS (verified
    against its chain digests), after the command budget is closed (launch) and a final guard check
    passed. Any failure before this write leaves a non-citable terminal record (launch / caller)."""
    if run.attempt.closed():
        raise Refused('the attempt is already closed; no outcome is published twice')
    if not run.attempt.passed('decide'):
        raise IntegrityError('decide is not sealed as PASS; no citable outcome')
    record = json.loads((run.attempt.stage_path('decide') / 'stage.json').read_text())
    label = (record.get('summary') or {}).get('label')
    if label not in reg.DECISION_LABELS:
        raise IntegrityError(f'decide recorded no registered decision label ({label!r})')
    decision = run.attempt.stage_path('decide') / 'decision.json'
    extra = {'git_commit': run.commit, 'budget_start_sha256': run.budget_start,
             'decision_sha256': sha256_file(decision) if decision.exists() else None,
             'decide_stage_json_sha256': run.attempt.record('decide')['stage_json_sha256']}
    run.guard.check('publish final')  # the last guard check; the write below is the last write
    write_final(run.attempt.root, outcome=label, stage='decide', reason=None, config_sha256=run.config_sha256,
                citable=True, extra=extra)
    return label


def environment():
    """Versions only (importlib.metadata: LightGBM is never imported just to be recorded)."""
    import importlib.metadata
    out = {'python': platform.python_version(), 'machine': platform.machine(),
           'dynamic_library_path': os.environ.get('DYLD_LIBRARY_PATH', '')}
    for name in ('numpy', 'pandas', 'scipy', 'pyarrow', 'lightgbm', 'torch'):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    pyproject = REPO / 'pyproject.toml'
    out['pyproject_sha256'] = sha256_file(pyproject) if pyproject.exists() else None
    return out


# ---------------------------------------------------------------- shared pieces of the stages

def oof_spec(config):
    return {'salt': config['seeds']['fold_salts']['train_oof'], 'k': config['cross_fitting']['folds']['k'],
            'prior_strength': float(config['candidates']['B0']['parameters']['prior_strength']),
            'minimum_action_count': int(config['candidates']['B0']['parameters']['minimum_action_count'])}


def feature_context(run, frame, train, raw):
    from pitchmdp.policy_requests import volume_edges
    inputs = run.inputs()
    volumes = train.groupby('pitcher').size()
    edges = volume_edges(train, (0.25, 0.75))
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
    (pitcher, count, side, previous) counts and label vocabulary equal the registered artifact's."""
    pitchers, cells = b1.tables['pitcher'], b1.tables['previous_type']
    expected = b0_vector.repertoire
    return {'league': bool(np.array_equal(b1.league, expected.league)),
            'pitchers': bool(np.array_equal(pitchers.keys >> trailing_bits(1), expected.pitchers)
                             and np.array_equal(pitchers.counts, expected.pitcher_counts)),
            'cells': bool(cells.vocabulary == b0_vector.cells.vocabulary
                          and np.array_equal(cells.keys, b0_vector.cells.keys)
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
    """First registered step: TRAIN rows only (row groups lying wholly on or before 2025-04-30), cost
    and integrity (wall time, RSS, determinism, hashes, the split-aligned processed layout). No metric
    is computed or read."""
    inputs = run.inputs()
    spec = run.config['runner']['smoke']
    layout = inputs.layout()  # footer statistics only; refuse before any decode when not split-aligned
    out.json('layout.json', layout)
    if not layout['split_aligned']:
        raise IntegrityError(f'processed row-group layout is not split-aligned (physically safe acquisition '
                             f'impossible): {layout["problems"]}')
    train = inputs.smoke_frame('smoke')
    masks = population_masks(train, hands=inputs.hands, repertoire=inputs.b0.repertoire)
    plus = np.flatnonzero(masks['P_plus'])
    n = min(int(spec['train_rows']), len(plus))
    if not n:
        raise IntegrityError('no TRAIN P_plus rows for the smoke')
    part = train.iloc[:plus[n - 1] + 1]
    timings = {}
    t = time.perf_counter()
    raw = inputs.raw_keys('smoke', part.game_pk.unique(), upper=TRAIN_END)
    ctx = feature_context(run, part, train, raw)
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
    after_train = inputs.log.rows_at_or_after(SINCE_AFTER_TRAIN, guarded_only=False, phases=['smoke'])
    checks = {'processed_layout_split_aligned': layout['split_aligned'],
              'processed_bytes_unchanged': inputs.verify_unchanged(),
              'b0_artifact_sha256_pinned': inputs.identity['b0']['file_sha256'] == pinned_b0,
              'vector_b0_equals_categorical_bc_on_train': r1b_train is not None and r1b_train <= atol,
              'b1_at_b0_setting_equals_b0_recipe': b1_equals_b0 <= atol,
              'b2_deterministic': deterministic, 'no_2026_row': guard_dates(train) is train,
              'no_row_after_train_end_decoded': after_train == 0,
              'no_dev_bearing_guarded_chunk_read': inputs.log.dev_bearing_guarded_reads() == 0,
              'dev_guarded_rows_decoded': inputs.log.rows_at_or_after(SINCE_DEV, guarded_only=True) == 0}
    record = {'rows': {'train_rows_read': int(len(train)), 'train_slice_rows': int(len(part)),
                       'train_P_plus_rows': int(len(table)), 'state_checks': int(len(rows))},
              'timings': timings, 'b2_fits': fits,
              'max_abs_diff': {'vector_vs_categorical': r1b_train, 'b1_b0_setting_vs_vector_fit': b1_equals_b0},
              'checks': checks, 'feature_matrix_sha256': sha256_bytes(features.tobytes()),
              'oof_models_sha256': [arrays_sha256(m.arrays()) for m in oof_models],
              'rows_decoded_after_train_end': after_train, 'read_summary': inputs.log.summary(['smoke']),
              'metrics_computed': None,
              'note': 'cost and integrity only; no comparison metric computed or read'}
    out.json('smoke.json', record)
    if not all(checks.values()):
        raise IntegrityError(f'smoke integrity failed: {[k for k, v in checks.items() if not v]}')
    append_chain(run.attempt.root / 'addenda.jsonl',
                 {'addendum': 'smoke', 'record_sha256': sha256_file(out.path / 'smoke.json'),
                  'config_sha256': run.config_sha256, 'git_commit': run.commit})
    return {'checks': checks, 'b2_fit_seconds': [f['seconds'] for f in fits]}


def stage_census(run, out):
    inputs = run.inputs()
    counts = census_counts(inputs.census_view('census'), hands=inputs.hands, repertoire=inputs.b0.repertoire)
    guarded = [e for e in inputs.log.reads(['census']) if e['guarded']]
    if guarded:
        raise IntegrityError('the census read a guarded column chunk')
    out.json('census.json', counts)
    return {'splits': {k: v['rows'] for k, v in counts['splits'].items()}, 'guarded_reads': 0}


def stage_fit(run, out):
    inputs, config = run.inputs(), run.config
    frame = inputs.pre_gate_frame('fit')
    train = frame.loc[frame.split.eq('train')]
    raw = inputs.raw_keys('fit', frame.game_pk.unique())
    ctx = feature_context(run, frame, train, raw)
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
    tune = b1.tune(es_table.keys.take(es_rows), labels(es_table, es_rows),
                   sweeps=config['candidates']['B1']['search']['sweeps'], guard=run.guard)
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
    frame = inputs.pre_gate_frame('calibrate')
    table, info, _, _ = split_table(frame.loc[frame.split.eq('temperature')], 'temperature', ctx)
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
    root, inputs = run.attempt.root, run.inputs()
    files = [p for stage in ('fit', 'calibrate') for p in StageDir(run.attempt.stage_path(stage)).artifacts()]
    train, es, temp = run.state['train_table'], run.state['es_table'], run.state['temp_table']
    ctx = run.state['ctx']
    groups = {
        'B1': {'files': 'b1/', 'input_rows': {'fit': {'split': 'train', 'dates': date_range(train), 'rows': len(train)},
                                               'alpha_selection': {'split': 'earlystop',
                                                                   'dates': date_range(es, run.state['es_rows']),
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
    inputs.verify_unchanged()
    # zero DEV decode, from the reader's own pyarrow-level log: no guarded column chunk of a DEV-bearing
    # row group was passed to pyarrow (whatever it would have returned), and no DEV guarded row came back
    dev_chunks = inputs.log.dev_bearing_guarded_reads()
    dev_guarded = inputs.log.rows_at_or_after(SINCE_DEV, guarded_only=True)
    extra = {'config_sha256': run.config_sha256, 'git_commit': run.commit, 'seeds': run.seeds(),
             'inputs': inputs.identity, 'latest_input_row_date': latest,
             'inputs_on_or_before_2025_05_31': latest <= reg.STYLE_INPUT_FILTER['end'],
             'reads': inputs.reads(), 'read_summary': inputs.log.summary(),
             'dev_bearing_guarded_chunks_read': dev_chunks, 'dev_guarded_rows_decoded': dev_guarded,
             'calibration_status': {k: v['status'] for k, v in run.state['calibration'].items()}}
    if dev_chunks or dev_guarded or inputs.released is not None or not extra['inputs_on_or_before_2025_05_31']:
        raise IntegrityError('freeze-primary needs inputs <= 2025-05-31 and no guarded chunk of a DEV-bearing row '
                             'group read (pyarrow-level read log)')
    manifest = build_manifest(root, files, name='freeze-primary', groups=groups, extra=extra)
    data = canonical_bytes(manifest)
    (out.path / 'manifest.json').write_bytes(data)
    append_chain(root / 'addenda.jsonl', {'addendum': 'freeze-primary', 'manifest_sha256': sha256_bytes(data),
                                          'latest_input_row_date': latest, 'dev_guarded_rows_decoded': dev_guarded,
                                          'dev_bearing_guarded_chunks_read': dev_chunks})
    reloaded = reload_primary(run, manifest)
    temp, rows = run.state['temp_table'], run.state['temp_rows']
    roundtrip = {name: float(np.abs(reloaded[name].batch(temp, rows) - run.state[key].batch(temp, rows)).max())
                 for name, key in (('B1', 'b1'), ('B2', 'b2'))}
    if any(v != 0 for v in roundtrip.values()):
        raise IntegrityError(f'frozen models do not reproduce the fitted ones: {roundtrip}')
    run.state.update(manifest_primary=manifest, b1=reloaded['B1'], b2=reloaded['B2'],
                     calibrated=reloaded['calibrated'])
    return {'manifest_sha256': sha256_bytes(data), 'artifacts': len(manifest['artifacts']), 'roundtrip_max_abs': roundtrip,
            'dev_guarded_rows_decoded': dev_guarded, 'dev_bearing_guarded_chunks_read': dev_chunks}


def reload_primary(run, manifest):
    """Rebuild B1, B2 and the calibrators from the frozen files (sha256 verified)."""
    root, inputs = run.attempt.root, run.inputs()
    verify_manifest(root, manifest)
    fit = run.attempt.stage_path('fit').name
    cal = run.attempt.stage_path('calibrate').name
    b1_meta = json.loads(load_frozen_text(root, manifest, f'{fit}/b1/b1.json'))
    b1 = BackoffBC.from_arrays(load_arrays(root, manifest, f'{fit}/b1', BackoffBC.array_names()),
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
    for name, what in ((ledger['path'], 'S6 ledger (sha256, then the allowlisted scan)'),
                       (ledger['manifest_path'], 'S6 manifest (sha256)')):
        inputs.log.note(kind='open', phase='s6-integrity', file=str(name), what=what)
    header, decisions, record = read_ledger(ledger['path'], manifest_path=ledger['manifest_path'],
                                            ledger_sha256=ledger['ledger_sha256'],
                                            manifest_sha256=ledger['manifest_sha256'],
                                            runtime_sha256=ledger['runtime_sha256'],
                                            candidate_sha256=ledger['candidate_sha256'])
    pas = by_pa(decisions)
    e0 = e0_from_ledger(pas, inputs.b0.bc, inputs.support, inputs.hands)
    gates = config['ope_weight_diagnostics']['integrity_gates']
    r1b = r1b_logging(decisions, inputs.b0.vector)
    r2 = r2_reproduction(pas, e0, gates['R2']['expected'], gates['R2_rtol'])
    s6 = config['data']['s6_ledger']
    # identities: the header's pins.candidate.sha256 is the sealed policy identity (registered expectation)
    # and every decision row carries the PolicyRuntime identity pinned from the header at registration
    checks = {'S6_identity': record['sha256_before'] == record['sha256_after']
              and record['header_runtime_sha256'] == ledger['runtime_sha256'] == s6['runtime_sha256']['value']
              and record['header_candidate_sha256'] == ledger['candidate_sha256'] == s6['candidate_sha256']
              == s6['expected_candidate_sha256'],
              'R1': inputs.identity['b0']['file_sha256'] == config['data']['b0_artifact']['file_sha256'],
              'R1b': r1b['rows'] > 0 and r1b['max_abs_diff'] <= gates['R1b_atol'], 'R2': r2['pass']}
    out.json('s6_integrity.json', {'ledger': record, 'pas': len(pas), 'e0_pas': sum(v[0] for v in e0.values()),
                                   'r1b': r1b, 'r2': r2, 'checks': checks,
                                   'dev_metric': None, 'note': 'registered fixed-B0 integrity checks only'})
    if not all(checks.values()):
        raise IntegrityError(f'FAILED_INTEGRITY: {[k for k, v in checks.items() if not v]}')
    run.state.update(ledger_pas=pas, e0=e0)
    return {'checks': checks, 'e0_pas': sum(v[0] for v in e0.values()), 'ledger_parsing': record['parsing']}


def stage_xf_fit(run, out):
    inputs, config = run.inputs(), run.config
    released = inputs.release_dev(run.gate, 'xf-fit')
    ctx, train = run.state['ctx'], run.state['train_table']
    frame = inputs.pre_gate_frame('xf-fit')
    dev, info, _, _ = split_table(frame.loc[frame.split.eq('dev')], 'dev', ctx)
    inputs.verify_unchanged()  # after the last decode of the attempt
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
                            'dev_release': released,
                            'dev_label_aggregates': 'reported by score-dev only (P, S_in, S_off, P_plus, TRAIN0)'})
    run.state.update(dev_table=dev, dev_info=info, dev_release=released)
    return {'folds': len(per_fold)}


def stage_freeze_xf(run, out):
    run.gate.require_pinned('freeze-primary', 'freeze-xf')
    files = StageDir(run.attempt.stage_path('xf-fit')).artifacts()
    manifest = build_manifest(run.attempt.root, files, name='freeze-xf', groups={
        'XF': {'files': 'xf/', 'input_rows': {'fit': 'TRAIN plus the other four DEV folds', 'predict': 'DEV fold k'}}},
        extra={'config_sha256': run.config_sha256, 'git_commit': run.commit, 'reads': run.inputs().reads()})
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
    """Exhaustive decision-label partition (registry.DECISION_LABELS)."""
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
    preflight = {path.name: json.loads(path.read_text()).get('tests')
                 for path in sorted(run.attempt.root.glob('preflight-*.json'))}
    decision = {'label': label, 'candidate_status': statuses, 'criteria': criteria, 'dev_nll': nll,
                'tie_rule': f"both eligible and |NLL_B1 - NLL_B2| < {config['verdict_rule']['selection']['tie_margin']}"
                            ' (strict) -> B1', 'E4_integrity': integrity_ok,
                'E4_evidence': {'smoke': run.attempt.status('smoke'), 's6_integrity': run.attempt.status('s6-integrity'),
                                'preflight_tests': preflight or 'not run by this caller (launch runs them before any '
                                                                'real-data stage)'},
                'descriptive_only': config['verdict_rule']['descriptive_only'],
                'not_claimed': 'no OPE value or Delta; DEV is exposed development; no service change'}
    decision['citable'] = False  # the citable outcome is final.json, published last (publish_final)
    out.json('decision.json', decision)
    return {'label': label, 'candidate_status': statuses}


STAGE_FUNCTIONS = {'smoke': stage_smoke, 'census': stage_census, 'fit': stage_fit, 'calibrate': stage_calibrate,
                   'freeze-primary': stage_freeze_primary, 's6-integrity': stage_s6_integrity, 'xf-fit': stage_xf_fit,
                   'freeze-xf': stage_freeze_xf, 'score-dev': stage_score_dev, 's6-weights': stage_s6_weights,
                   'decide': stage_decide}


# ---------------------------------------------------------------- real inputs (only after the gate)

def verify_processed_identity(local, parent, pinned_sha256, opened=lambda name, what: None):
    """matrix_data.load_verified_processed_cache identity checks WITHOUT decoding any row: the frozen
    manifests, raw-byte sha256 of the processed parquet and the physics sidecar (never decoded), and
    equality with the G0 parent preparation's dataset identity and source provenance."""
    from pitchmdp.data import KEY, RAW_ALLOWLIST, hash_file
    from pitchmdp.sequence_data import CACHE_VERSION, SIDECAR_COLUMNS
    if local.get('raw_allowlist') != RAW_ALLOWLIST:
        raise IntegrityError('configured raw allowlist differs from the approved 2023-25 sources')
    root = Path(local['artifact_root'])
    opened(root / 'reports/data_quality.json', 'frozen quality manifest (json)')
    quality = json.loads((root / 'reports/data_quality.json').read_text())
    opened(root / 'cache/sequence_physics.json', 'frozen physics cache manifest (json)')
    metadata = json.loads((root / 'cache/sequence_physics.json').read_text())
    sources = quality.get('sources')
    if (not isinstance(sources, list) or [s.get('file') for s in sources] != RAW_ALLOWLIST
            or any(not isinstance(s.get('bytes'), int) or s['bytes'] <= 0 or not isinstance(s.get('sha256'), str)
                   or len(s['sha256']) != 64 for s in sources)):
        raise IntegrityError('frozen quality manifest lacks approved historical raw identities')
    expected = {'cache_version': CACHE_VERSION, 'sources': sources, 'processed_sha256': quality['processed_sha256'],
                'rows': quality['rows'], 'key': KEY, 'columns': list(SIDECAR_COLUMNS)}
    if metadata.get('identity') != expected:
        raise IntegrityError('physics cache identity differs from the frozen quality manifest')
    processed, cache = root / 'processed/pitches.parquet', root / 'cache/sequence_physics.parquet'
    opened(processed, 'sha256 of the processed parquet bytes (no decode)')
    opened(cache, 'sha256 of the physics sidecar bytes (never decoded)')
    if quality['processed_sha256'] != pinned_sha256 or hash_file(processed) != pinned_sha256 \
            or hash_file(cache) != metadata.get('sha256'):
        raise IntegrityError('processed or physics cache bytes differ from their pins')
    provenance = {'input_mode': 'verified_processed_cache',
                  'raw_available': all((Path(local['raw_root']) / s['file']).is_file() for s in sources),
                  'raw_hash_verification': 'historical_only', 'processed_cache_verification': 'current_verified',
                  'quality_manifest_sha256': hash_file(root / 'reports/data_quality.json'),
                  'cache_manifest_sha256': hash_file(root / 'cache/sequence_physics.json'),
                  'processed_sha256': quality['processed_sha256'], 'cache_sha256': metadata['sha256']}
    if expected != parent['dataset_identity'] or provenance != parent['source_provenance']:
        raise IntegrityError('processed data differs from the G0 parent preparation')
    return processed, {'rows': quality['rows'], 'provenance': provenance}


def verify_a3_input(config, opened=lambda name, what: None):
    """The split-aligned A3 input (data.a3_input): its identity manifest, pinned by sha256, must record
    a PASS rewrite of the pinned processed parquet into the pinned file at the registered split dates
    (rows equal, schema equal, every column digest equal, aligned). Returns (path, record)."""
    a3 = config['data']['a3_input']
    path = Path(a3['identity_manifest_path'])
    opened(path, 'A3 input identity manifest (pinned sha256)')
    data = path.read_bytes()
    if sha256_bytes(data) != a3['identity_manifest_sha256']:
        raise IntegrityError('A3 input identity manifest differs from its registration pin')
    manifest = json.loads(data)
    source, rewritten = manifest.get('source') or {}, manifest.get('rewritten') or {}
    identity = manifest.get('identity') or {}
    checks = {'result_pass': manifest.get('result') == 'PASS' and identity.get('result') == 'PASS',
              'aligned': manifest.get('aligned') is True,
              'rows_and_schema_equal': identity.get('rows_equal') is True and identity.get('schema_equal') is True
              and source.get('rows') == rewritten.get('rows'),
              'every_column_equal': bool(identity.get('columns')) and not identity.get('differing_columns')
              and all(c.get('equal') is True for c in identity['columns'].values()),
              'source_is_the_pinned_processed': source.get('sha256') == reg.DATA_PINS['processed_sha256']
              == a3['source_sha256'],
              'rewritten_is_the_pinned_input': rewritten.get('sha256') == a3['rechunked_sha256'],
              'boundaries': manifest.get('boundaries') == a3['boundaries'] and manifest.get('end') == a3['end'],
              'tool': manifest.get('tool') == a3['tool']}
    if not all(checks.values()):
        raise IntegrityError(f'the A3 input identity manifest does not certify the pinned input: '
                             f'{[k for k, v in checks.items() if not v]}')
    return Path(a3['path']), {'rechunked_sha256': a3['rechunked_sha256'],
                              'identity_manifest_sha256': a3['identity_manifest_sha256'], 'checks': checks,
                              'row_groups': rewritten.get('row_groups')}


class RealSource:
    """Pinned real inputs. Defense in depth: besides the per-stage revalidation, ``open`` itself refuses
    unless it is called inside an executing stage of an open attempt and the registration gate passes
    again, before any data or artifact file is opened; every file it opens is logged first."""
    def __init__(self, config, local, config_path, env):
        self.config, self.local, self.config_path, self.env = config, local, Path(config_path), env

    def revalidate(self, run, stage):
        if json.loads(self.config_path.read_bytes()) != self.config:
            raise Refused('the config file changed since this command started', terminal=True)
        registration_gate(self.config, git=self.env.git_state(_field(self.config, 'code.commit'), self.config_path),
                          digest_entries=self.env.digest_entries,
                          runner_sha256=self.env.runner_sha256())
        return {'stage': stage, 'registration_gate': 'PASS', 'dependencies': self.env.native_check()}

    def check_stage_state(self, run):
        stage = getattr(run, 'current_stage', None)
        attempt = getattr(run, 'attempt', None)
        if stage not in reg.STAGES or attempt is None or attempt.closed() or not attempt.stage_path(stage).is_dir() \
                or (attempt.stage_path(stage) / 'stage.json').exists():
            raise Refused('the loader opens files only inside an executing stage of an open attempt', terminal=True)
        return stage

    def open(self, run):
        stage = self.check_stage_state(run)  # loader-level gate: no file is opened before these pass
        self.revalidate(run, f'loader:{stage}')
        from pitchmdp.policy_artifacts import load_hand_registry, load_support_table
        from pitchmdp.policy_identity import pinned_json, pinned_pickle
        config, pins, local, log = self.config, reg.DATA_PINS, self.local, run.log
        opened = lambda name, what: log.note(kind='open', phase=stage, file=str(name), what=what)
        opened(REPO / 'configs' / 'G0-RESEARCH-FROZEN-v1.json', 'pinned G0 bundle (json)')
        bundle = pinned_json(REPO / 'configs' / 'G0-RESEARCH-FROZEN-v1.json', pins['g0_bundle_sha256'])
        files = bundle['files']
        opened(files['p4_parent_preparation']['path'], 'pinned parent preparation (json)')
        parent = pinned_json(files['p4_parent_preparation']['path'], files['p4_parent_preparation']['sha256'])
        processed, identity_record = verify_processed_identity(local, parent, pins['processed_sha256'],
                                                               opened=opened)
        a3_path, a3_record = verify_a3_input(config, opened)
        reader = ProcessedReader(a3_path, sha256=config['data']['a3_input']['rechunked_sha256'], log=log,
                                 phase=stage)  # refuses a file that is not split-aligned (footer statistics)
        if reader.rows != identity_record['rows']:
            raise IntegrityError('A3 input row count differs from the frozen quality manifest')
        opened(files['p4_auxiliary']['path'], 'pinned G0 auxiliary (frozen archetypes)')
        aux = pinned_pickle(files['p4_auxiliary']['path'], files['p4_auxiliary']['sha256'])
        artifact = config['data']['b0_artifact']
        opened(artifact['path'], 'registered BC_P artifact (B0)')
        b0 = RegisteredBC.load(artifact['path'], artifact['file_sha256'])
        opened(artifact['support_table_path'], 'BC_P support table')
        support, support_content = load_support_table(artifact['support_table_path'], artifact['support_table_sha256'],
                                                      b0.artifact)
        opened(artifact['hands_path'], 'BC_P hand registry')
        hands, hands_content = load_hand_registry(artifact['hands_path'], artifact['hands_sha256'], b0.artifact)
        if list(local['raw_allowlist']) != list(pins['raw_sha256']):
            raise IntegrityError('raw allowlist must be exactly the pinned 2023-2025 Statcast files')
        raw_reader = RawKeysReader(local['raw_root'], pins['raw_sha256'], log)
        s6 = config['data']['s6_ledger']
        ledger = {'path': s6['path'], 'manifest_path': str(Path(s6['path']).parent / 'manifest.json'),
                  'ledger_sha256': s6['ledger_sha256'], 'manifest_sha256': s6['manifest_sha256'],
                  'runtime_sha256': s6['runtime_sha256']['value'], 'candidate_sha256': s6['candidate_sha256']}
        identity = {'processed_sha256': pins['processed_sha256'], 'processed_provenance': identity_record['provenance'],
                    'g0_bundle_sha256': pins['g0_bundle_sha256'], 'g0_auxiliary_sha256': files['p4_auxiliary']['sha256'],
                    'b0': {'file_sha256': b0.artifact.file_sha256, 'bc_sha256': b0.artifact.sha256,
                           'support_file_sha256': artifact['support_table_sha256'], 'support_content_sha256': support_content,
                           'hands_file_sha256': artifact['hands_sha256'], 'hands_content_sha256': hands_content},
                    'raw_sha256': pins['raw_sha256'], 'processed_row_groups': len(reader.groups),
                    'a3_input': a3_record, 's6_ledger': {k: v for k, v in ledger.items() if k != 'path'}}
        return A3Inputs(reader=reader, raw_reader=raw_reader, b0=b0, support=support, hands=hands,
                        archetypes=aux['context'].archetypes, ledger=ledger, identity=identity, log=log)


# ---------------------------------------------------------------- launch

def default_preflight_command():
    return [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *PREFLIGHT_TESTS]


PREFLIGHT_OVERRIDES = ('A3_CONFIG_PATH', 'A3_NESTED_PREFLIGHT')  # test-only switches of the test files


def default_preflight_env():
    """The launch environment without the test-only switches: a real preflight always tests the
    committed config and runs the isolated end-to-end tests."""
    return {k: v for k, v in os.environ.items() if k not in PREFLIGHT_OVERRIDES}


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


@dataclass
class LaunchEnv:
    """Everything ``launch`` touches outside the attempt, injectable for synthetic tests."""
    git_state: object = git_state
    runner_sha256: object = field(default_factory=lambda: (lambda: sha256_file(Path(__file__))))
    native_check: object = native_and_dependencies
    memory_command: tuple = ('memory_pressure',)
    preflight_command: object = default_preflight_command
    preflight_env: object = default_preflight_env
    digest_entries: object = read_digest_manifest
    source_factory: object = RealSource
    now: object = None
    clock: object = time.monotonic
    rss: object = peak_rss_bytes
    alarm: bool = True
    memory_timeout: float = 120.
    preflight_timeout: float = 1800.


def _close(dest, owned, outcome, command, reason, config):
    """Terminal, non-citable final.json for an attempt this command owns (once)."""
    if owned and dest is not None and dest.is_dir() and not Attempt(dest).closed():
        write_final(dest, outcome=outcome, stage=f'launch:{command}', reason=reason,
                    config_sha256=config_content_sha256(config))


def launch(config_path, local_config_path, destination, command, env=None):
    """smoke: open a new attempt and run the smoke; run: continue a passed smoke through decide.

    Order: destination rule -> registration gate -> native runtime and dependency pins -> protection
    window -> heavy lock -> attempt directory -> attempt budget (budget.jsonl; the remaining registered
    total) -> supervised memory probe -> supervised preflight tests -> stages (each re-runs the
    registration gate first). Transient refusals (window, busy lock, memory, environment) are recorded
    in refusals.jsonl and leave the attempt open; a registration-gate or preflight failure, an abort or
    interruption, an integrity failure or a runtime failure closes the attempt this command owns with a
    non-citable final.json.
    Returns the exit code (0 PASS, 2 refused / integrity, 3 aborted, 1 runtime failure)."""
    env = env or LaunchEnv()
    config_path = Path(config_path)
    config = json.loads(config_path.read_bytes())
    dest, start, owned = None, None, False
    guard = Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, clock=env.clock, now=env.now, rss=env.rss, alarm=env.alarm)
    try:
        dest = check_destination(destination, output_root=_field(config, 'runner.output_root'))
        owned = command == 'run' and dest.is_dir()  # run continues exactly this attempt
        commit = _field(config, 'code.commit')
        registration_gate(config, git=env.git_state(commit, config_path), runner_sha256=env.runner_sha256(),
                          digest_entries=env.digest_entries)
        dependencies = env.native_check()
        if guard.in_window():  # before any directory is created
            raise Refused('inside the recorder/publisher protection window 06:30-13:30 KST; refusing to start')
        with heavy_lock(config['runner']['heavy_lock']):
            attempt = Attempt(dest)
            if command == 'smoke':
                # the smoke opens the attempt: a new directory, or one whose earlier launches were refused
                # before any stage ran (its refusals and budget records are kept and counted)
                if dest.exists() and (attempt.started() or attempt.closed()):
                    raise Refused('the smoke opens a new attempt; this attempt already ran a stage or is closed')
                dest.parent.mkdir(parents=False, exist_ok=True)  # <ML-MATRIX-20260924>/EXP-BC-CAL-01
                dest.mkdir(parents=False, exist_ok=True)
                owned = True
            elif not dest.exists():
                raise Refused('run continues an attempt whose smoke passed; the attempt directory is missing')
            if attempt.closed():
                raise Refused('this attempt is closed (final.json exists); register a new attempt')
            guard.consumed = consumed_seconds(dest)
            guard.start_run()
            start = begin_command(dest, command, guard.consumed)
            try:
                probe = supervised(env.memory_command, guard, timeout=env.memory_timeout, where='memory probe')
                share = parse_free_share(probe['stdout'])
                if share < config['cost_budget']['memory']['start_requires_free_share_min']:
                    raise Refused(f'memory_pressure free share {share:.2f} below the registered start minimum')
                child_env = env.preflight_env()
                tests = supervised(env.preflight_command(), guard, timeout=env.preflight_timeout, where='preflight',
                                   cwd=PROJECT, env=child_env)
                git = env.git_state(commit, config_path)
                dump_json(dest / f'preflight-{command}.json', {
                    'tests': {'returncode': tests['returncode'], 'seconds': tests['seconds'],
                              'files': list(PREFLIGHT_TESTS),
                              'switches': {k: child_env.get(k) for k in PREFLIGHT_OVERRIDES},
                              'summary': (tests['stdout'].strip().splitlines() or [''])[-1]},
                    'memory_rule': 'runner peak RSS + current RSS of the supervised child tree <= the registered cap',
                    'peak_child_tree_rss_bytes': {'memory_probe': probe['peak_tree_rss_bytes'],
                                                  'preflight': tests['peak_tree_rss_bytes']},
                    'free_share': share, 'git': git, 'dependencies': dependencies, 'native_runtime': environment()})
                if tests['returncode'] != 0:
                    raise Refused('synthetic/leakage tests failed; no real-data stage runs', terminal=True)
                local = json.loads(Path(local_config_path).read_bytes()) if Path(local_config_path).exists() else {}
                run = Run(config, destination=dest, source=env.source_factory(config, local, config_path, env),
                          guard=guard, commit=git['head'])
                run.budget_start = start['sha256']  # every stage record of this command is bound to it
                for stage in ('smoke',) if command == 'smoke' else reg.STAGES[1:]:
                    status = execute_stage(run, stage)  # a non-PASS stage writes final.json itself
                    print(json.dumps({'stage': stage, 'status': status}), flush=True)
                    if status != 'PASS':
                        return 3 if status == 'ABORTED' else 2 if status in ('FAILED_INTEGRITY', 'REFUSED') else 1
            finally:
                end_command(dest, start, guard.elapsed())  # the command budget closes before any citable write
            if command == 'run':
                print(json.dumps({'final': publish_final(run)}), flush=True)  # the last write of the attempt
    except Abort as abort:
        _close(dest, owned, 'ABORTED', command, {'reason': abort.reason, 'where': abort.where}, config)
        print(json.dumps({'aborted': str(abort)}), file=sys.stderr, flush=True)
        return 3
    except KeyboardInterrupt:
        _close(dest, owned, 'ABORTED', command, {'reason': 'INTERRUPTED', 'where': f'launch:{command}'}, config)
        print(json.dumps({'aborted': 'INTERRUPTED'}), file=sys.stderr, flush=True)
        return 3
    except Refused as refusal:
        print(json.dumps({'refused': str(refusal)}), file=sys.stderr, flush=True)
        if dest is not None and dest.is_dir():  # the refusal is recorded in the attempt
            append_chain(dest / 'refusals.jsonl', {'command': command, 'refused': str(refusal),
                                                   'terminal': refusal.terminal,
                                                   'utc': datetime.now(timezone.utc).isoformat()})
            if refusal.terminal:
                _close(dest, owned, 'REFUSED', command, str(refusal), config)
        return 2
    except IntegrityError as failure:
        _close(dest, owned, 'FAILED_INTEGRITY', command, str(failure), config)
        print(json.dumps({'failed_integrity': str(failure)}), file=sys.stderr, flush=True)
        return 2
    except Exception as failure:
        _close(dest, owned, 'FAILED_RUNTIME', command, f'{type(failure).__name__}: {failure}', config)
        print(json.dumps({'failed_runtime': f'{type(failure).__name__}: {failure}'}), file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--local-config', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('command', choices=('smoke', 'run'))
    args = parser.parse_args(argv)
    return launch(args.config, args.local_config, args.destination, args.command)


if __name__ == '__main__':
    sys.exit(main())
