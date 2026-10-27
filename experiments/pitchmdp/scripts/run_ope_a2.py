"""EXP-OPE-RES-01 (track A2) stage runner: A2-SM smoke, A2-S0 ... A2-S4 and unblind.

A real-data command (smoke, s0, s1, s2, s3, decide, unblind) is guarded from argument parsing to its
completion receipt: the run-window rule (the whole hang guard must end strictly before the next 06:30
KST and never enter 06:30-13:30 KST), the SIGALRM hang guard and the RSS / free-memory checkpoint cover
preflight, data acquisition, computation, hashing and sealing; the hang timer stays armed through the
completion receipt (stage seal), which is staged, checked by the final guard check and committed by an
atomic rename (a failure before the rename publishes nothing and records a non-citable ABORTED); the
bounded cleanup then disarms the guards. Every post-commit step and the command-line entry's final report
are bounded by the time left before the hard cap; reaching it ends the command at once with a nonzero exit,
the committed seal standing. Preflight refuses (fail
closed) unless configs/EXP-OPE-RES-01.json passes the registry check with status REGISTERED, execution
enabled, every pin filled, the registration references recorded and the gates passed; the external
append-only digest manifest (outside results/) holds the registered config digest and the G-CODE
(commit = the running code's commit) and G-REG review receipts; the protected sources and the pinned
new files are byte-identical and HEAD's code is the registered (or recorded bug-fix) commit. The
destination is a fresh directory directly under the pinned output root, which is disjoint from every
sealed input and protected root (resolved real paths). An attempt-start receipt is appended before the
stage directory exists (at most two starts per stage; a retry after a deterministic integrity failure
needs a recorded bug-fix addendum, after a start without any receipt a recorded reconciliation). The
smoke reads TRAIN rows and listed columns only: its row groups are planned from the parquet footers
before the attempt starts and only row groups lying inside TRAIN are decoded. Every TRAIN-only read of the
processed frame (the smoke and A2-S0) decodes the pinned split-aligned copy (prepared once for A3 and reused),
verified first against its identity manifest and the processed-frame identity pin; the S6-dr-a5 stage
directory is opened by A2-S3 only, where G-INT runs on the untouched registered path before any
surrogate reward or variant is computed.

Commands that read no data: ``check``, ``pins``; manifest commands: ``register-digest``,
``record-review``, ``register-addendum``, ``reconcile-attempt`` (under the same window, memory and hang
guards, after the digest-manifest location check). Work checkpoints stop at the hang guard minus a
finalization reserve, a memory watchdog thread covers every code path, and records are written as
strict JSON (nonfinite values null, listed). Every data access of a registered command goes through an
``Environment`` (the real one by default) so the orchestration is testable on synthetic inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitchmdp import ope_decision as odc  # noqa: E402
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp import ope_sealed as osd  # noqa: E402
from pitchmdp import ope_surrogate as osg  # noqa: E402
from pitchmdp import ope_synthetic_a2 as osx  # noqa: E402
from pitchmdp.game import GameState  # noqa: E402
from pitchmdp.matrix_policy_artifacts import is_appledouble  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError, _require  # noqa: E402

COMMANDS = ('check', 'pins', 'register-digest', 'record-review', 'register-addendum', 'reconcile-attempt', 'smoke',
            's0', 's1', 's2', 's3', 'decide', 'unblind')
MANIFEST_COMMANDS = ('register-digest', 'record-review', 'register-addendum', 'reconcile-attempt')
STAGE_OF = {'smoke': 'A2-SM', 's0': 'A2-S0', 's1': 'A2-S1', 's2': 'A2-S2', 's3': 'A2-S3', 'decide': 'A2-S4',
            'unblind': 'A2-UB'}
GUARD_OF = {**{s: s for s in reg.STAGES}, 'A2-UB': 'A2-S4',  # unblind belongs to A2-S4 (3,600 s guard)
            'A2-MC': reg.MANIFEST_COMMAND_GUARD}  # the manifest commands (review round 2 #4)
UPSTREAM = {**reg.UPSTREAM, 'A2-UB': ('A2-S3', 'A2-S4')}
ATTEMPT_STAGES = (*reg.STAGES, 'A2-UB')
CODE_PATHS = ('experiments', 'src', 'scripts')
GIB = 1024 ** 3
KEY = ['game_pk', 'at_bat_number', 'pitch_number']
TRAIN_COLUMNS = ('game_pk', 'at_bat_number', 'pitch_number', 'game_date', 'game_type', 'split', 'pitch_type', 'pitcher',
                 'stand', 'balls', 'strikes', 'outs_when_up', 'inning', 'inning_topbot', 'home_score', 'away_score',
                 'post_home_score', 'post_away_score', 'on_1b', 'on_2b', 'on_3b', 'events', 'description',
                 'launch_speed', 'launch_angle')
RAW_COLUMNS = ('game_pk', 'at_bat_number', 'pitch_number', 'game_date', 'hc_x', 'hc_y', 'bb_type', 'launch_speed',
               'launch_angle', 'estimated_woba_using_speedangle')


class HangGuardExceeded(BaseException):
    """Infrastructure stop: the registered hang guard elapsed (partial outputs preserved, not citable)."""


class MemoryGuardExceeded(BaseException):
    """Infrastructure stop: RSS cap or free-memory guard (the recorder keeps priority)."""


class HardDeadlineReached(HangGuardExceeded):
    """A bounded step stopped by the time left before the armed hang guard (not by its own publication bound): the
    command ends at once with a nonzero exit, a committed receipt still standing (review round 8 #1)."""


# ---------------------------------------------------------------- digest manifest (external, append-only)
def _canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


GUARD_SIGNALS = (signal.SIGALRM, signal.SIGUSR1)
ACTIVE_GUARDS = []  # the armed CommandGuard of this process (at most one)
LAST_GUARD = []  # the latest command guard of this process: its hard cap also bounds the final report (cli)


@contextmanager
def deferred_interrupts():
    """Inside, a guard interrupt (hang guard, memory watchdog) is held as pending instead of raised, whichever
    thread the OS delivered it to, so a receipt is never torn (review round 4 #2). Yields the active guard."""
    guard = ACTIVE_GUARDS[-1] if ACTIVE_GUARDS else None
    if guard is None:
        yield None
        return
    guard.deferring += 1
    try:
        yield guard
    finally:
        guard.deferring -= 1


def bounded_io(fn, what, bound=None):
    """Blocking publication I/O in a helper thread (guard signals blocked there), waited for at most the
    registered publication bound and never past the armed hang guard (review round 8 #1): a slow create, write,
    fsync, diagnostic or report aborts or skips the step instead of stalling the command. A step stopped by the
    time left before the hard cap raises HardDeadlineReached (the command then ends at once); one stopped by its
    own publication bound raises HangGuardExceeded."""
    bound = reg.PUBLICATION_IO_TIMEOUT_SECONDS if bound is None else bound
    guard = ACTIVE_GUARDS[-1] if ACTIVE_GUARDS else None
    capped = False  # whether the hard deadline, not the publication bound, limits the wait
    if guard is not None:
        budget = guard.io_budget()
        if budget <= 0:
            raise HardDeadlineReached(f'{what}: no time left before the registered hang guard')
        capped, bound = budget < bound, min(bound, budget)
    box = {}

    def run():
        signal.pthread_sigmask(signal.SIG_BLOCK, GUARD_SIGNALS)
        try:
            box['value'] = fn()
        except BaseException as error:  # handed to the caller
            box['error'] = error
    worker = threading.Thread(target=run, name='a2-publication-io', daemon=True)
    worker.start()
    worker.join(bound)
    if worker.is_alive():
        if capped:
            raise HardDeadlineReached(f'{what} reached the time left before the registered hang guard ({bound:.3g} s)')
        raise HangGuardExceeded(f'{what} exceeded its bound ({bound:.3g} s)')
    if 'error' in box:
        raise box['error']
    return box.get('value')


_STAGING_COUNTER = itertools.count()


def _create_write(path, data):
    """In the bounded worker (review round 6 #1): exclusive creation of the write's unique staging name, write,
    fsync and close, all under the publication bound. The worker alone owns the descriptor, so an abandoned
    worker only ever touches its own never-published inode (review round 5 #2); it never renames."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


POSTCOMMIT_ERRORS = (Exception, HangGuardExceeded, MemoryGuardExceeded)


def _write_postcommit_record(manifest_path, entry, error):
    """In the bounded worker: choose the record's target (the attempt directory when it exists, else beside the
    manifest) and append the record durably."""
    output_dir = entry['payload'].get('output_dir')
    target = (Path(output_dir) / 'postcommit_errors.jsonl' if output_dir and Path(output_dir).is_dir()
              else manifest_path.with_name(manifest_path.name + '.postcommit_errors.jsonl'))
    record = {'time_utc': datetime.now(timezone.utc).isoformat(), 'receipt_kind': entry['kind'],
              'receipt_seq': entry['seq'], 'receipt_sha256': entry['sha256'], 'stage': entry['payload'].get('stage'),
              'output_dir': output_dir, 'error_type': type(error).__name__, 'error': str(error),
              'effect': 'the receipt was committed by its atomic rename; the directory sync after it failed'}
    _append_synced(target, (_canonical(record) + '\n').encode())


def _emit(text):
    sys.stderr.write(text + '\n')
    sys.stderr.flush()


def _append_synced(path, line):
    """Append one line durably (an append-only diagnostic record)."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(line)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _discard(path):
    """Remove a never-published staging file within the publication bound (review round 9 #8); a leftover is
    removed by the next writer, and the original error is what the caller raises."""
    try:
        bounded_io(lambda: _remove(path), 'the staging removal')
    except POSTCOMMIT_ERRORS:
        pass


class DigestManifest:
    """Append-only JSONL outside results/: each entry carries the previous entry's sha256 (hash chain); a
    published prefix is never changed. One writer at a time (review round 3 #3): an exclusive non-blocking
    fcntl lock on the sibling lock file, held by ``writer()`` for a whole command and by every append; a
    second writer refuses. Each append is published atomically (review round 4 #2): the verified bytes plus
    the new entry go to a staging file (bounded write + fsync), an optional final check runs, then an atomic
    rename publishes it; any failure before the rename removes the staging file and publishes nothing."""
    KINDS = ('config_digest', 'review_reference', 'attempt_start', 'attempt_reconciliation', 'stage_seal', 'cell_seal',
             'stage_failure', 'bugfix_addendum')

    def __init__(self, path):
        self.path = Path(path)
        _require(self.path.is_absolute(), 'digest manifest path must be absolute')
        _require(not self.path.resolve().is_relative_to((REPO / 'results').resolve()),
                 'the digest manifest must live outside the repository results/')
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        self._fd = None
        self.durability_errors = []  # post-commit directory-sync errors (the committed receipts stand)
        self.committed = []  # entries whose atomic rename returned in this process (known without any I/O)

    def _open_locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise IntegrityError('another writer holds the digest manifest; refusing (one writer at a time)') from None
        return fd

    @contextmanager
    def writer(self):
        """Hold the manifest's exclusive writer lock (a whole command)."""
        _require(self._fd is None, 'the digest manifest writer is already held by this command')
        fd = self._open_locked()
        self._fd = fd
        try:
            self.entries()  # the chain and its tail digest, verified under the lock
            for stale in self.path.parent.glob(f'{self.path.name}.staging-*'):  # never-published staging leftovers
                _remove(stale)
            yield self
        finally:
            self._fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextmanager
    def _locked(self):
        if self._fd is not None:
            yield
            return
        fd = self._open_locked()
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _parse(raw):
        out, prev = [], None
        for seq, line in enumerate(raw.splitlines()):
            entry = json.loads(line)
            body = {k: v for k, v in entry.items() if k != 'sha256'}
            _require(entry.get('seq') == seq and entry.get('prev') == prev
                     and entry.get('sha256') == hashlib.sha256(_canonical(body).encode()).hexdigest(),
                     f'digest manifest entry {seq} tampered or out of order')
            out.append(entry)
            prev = entry['sha256']
        return out

    def _read_raw(self):
        return self.path.read_bytes() if self.path.exists() else b''

    def entries(self):
        return self._parse(self._read_raw())

    def append(self, kind, payload, *, commit_check=None):
        """Publish one entry: the manifest read and the staging copy (verified chain + entry; create, write and
        fsync) in the bounded writer, ``commit_check`` (the final guard check of a completion receipt), the atomic
        rename os.replace (THE commit point, run synchronously in this thread), bounded directory sync. A guard
        interrupt during the publication is held; before the commit it aborts the publication (through
        ``commit_check``) and the staging file is removed (bounded), after the commit it is raised (or, for a
        completion receipt, reported after the cleanup: the published receipt stands)."""
        _require(kind in self.KINDS, f'unknown manifest entry kind {kind!r}')
        with self._locked(), deferred_interrupts() as guard:
            raw = bounded_io(self._read_raw, 'the manifest read')  # bounded like every step here (review round 9 #8)
            entries = self._parse(raw)  # verified under the lock: the new entry chains to the verified tail
            body = {'seq': len(entries), 'time_utc': datetime.now(timezone.utc).isoformat(), 'kind': kind,
                    'payload': payload, 'prev': entries[-1]['sha256'] if entries else None}
            entry = {**body, 'sha256': hashlib.sha256(_canonical(body).encode()).hexdigest()}
            data = raw + (_canonical(entry) + '\n').encode()
            attempt = Path(str(payload.get('output_dir') or '-')).name
            staging = self.path.with_name(f'{self.path.name}.staging-{os.getpid()}-{next(_STAGING_COUNTER):06d}-{kind}-'
                                          f'{payload.get("stage") or "-"}-{attempt}')
            try:
                bounded_io(lambda: _create_write(staging, data), 'the receipt staging write')  # create+write+fsync
                if commit_check is not None:
                    commit_check()
                    if guard is not None:
                        guard.publication_boundary()  # an interrupt observed so far aborts the receipt
                # THE COMMIT POINT: the atomic rename, run synchronously in this thread (never in a worker that could
                # be abandoned and commit later), so the receipt is committed exactly when it returns
                os.replace(staging, self.path)
            except BaseException:
                _discard(staging)
                raise
            self.committed.append(entry)
            # after the commit point nothing may abort the receipt's stage (review round 7 #1): every failure of the
            # directory sync or of its diagnostic is caught, persisted if possible and reported
            try:
                bounded_io(lambda: _fsync_dir(self.path.parent), 'the receipt directory sync')
            except POSTCOMMIT_ERRORS as error:
                self._postcommit_error(entry, error, guard)
        if guard is not None:
            if commit_check is not None:
                guard.defer_to_late()
            else:
                guard.raise_pending()
        return entry

    def _postcommit_error(self, entry, error, guard):
        """A failure after a committed rename: kept, persisted as an append-only record by the bounded writer
        (which also selects the target) and reported by a bounded stderr write, the exit code following after
        the cleanup (review round 7 #1); the committed receipt stands whatever the diagnostic does. Each step is
        bounded by the publication bound and by the time left before the hard cap; once that hard deadline is
        reached no further step runs and HardDeadlineReached ends the command at once, nonzero, the receipt still
        standing (review round 8 #1). Nothing else is raised."""
        self.durability_errors.append(error)
        if guard is not None:
            guard.late = guard.late or error
        deadline = error if isinstance(error, HardDeadlineReached) else None
        message = (f'warning: post-commit {type(error).__name__} after {entry["kind"]} seq {entry["seq"]}: {error} '
                   f'(the committed receipt stands)')
        for step, what in ((lambda: _write_postcommit_record(self.path, entry, error), 'the post-commit error record'),
                           (lambda: _emit(message), 'the post-commit warning')):
            if deadline is not None:
                break
            try:
                bounded_io(step, what)
            except POSTCOMMIT_ERRORS as failure:
                self.durability_errors.append(failure)
                deadline = failure if isinstance(failure, HardDeadlineReached) else None
        if deadline is not None:
            raise HardDeadlineReached(f'{deadline}; after {entry["kind"]} seq {entry["seq"]} the command ends at once '
                                      f'(the committed receipt stands)') from deadline

    def latest(self, kind, **match):
        found = [e for e in self.entries() if e['kind'] == kind
                 and all(e['payload'].get(k) == v for k, v in match.items())]
        return found[-1] if found else None


def require_config_digest(manifest, config):
    entry = manifest.latest('config_digest')
    _require(entry is not None and entry['payload'].get('digest') == reg.config_digest(config),
             'the registered config digest is missing from the digest manifest or differs (G-REG)')
    return entry


def require_reviews(manifest, config, code_commit):
    """G-CODE and G-REG receipts (PM decision, review #3): the latest receipt of each gate has verdict
    PASS, or PASS_WITH_NOTES with its notes closed, and names the registered config digest; the G-CODE
    receipt names the commit of the code that is running. Missing or mismatching receipts refuse."""
    digest = reg.config_digest(config)
    receipts = {}
    for gate in reg.REQUIRED_RECEIPTS:
        entry = manifest.latest('review_reference', gate=gate)
        _require(entry is not None, f'no {gate} review receipt in the digest manifest; refusing to run')
        r = entry['payload']
        _require(r.get('verdict') in reg.RECEIPT_VERDICTS, f'{gate} receipt verdict {r.get("verdict")!r} is not a pass')
        _require(r.get('verdict') == 'PASS' or r.get('notes_closed') is True,
                 f'{gate} receipt is PASS_WITH_NOTES with open notes; refusing to run')
        _require(r.get('config_digest') == digest, f'{gate} receipt names another config digest')
        if gate == 'G-CODE':
            _require(r.get('commit') == code_commit,
                     f'the G-CODE receipt commit {r.get("commit")!r} is not the running code commit {code_commit}')
        receipts[gate] = r
    return receipts


ABORT_MARKERS = ('ABORTED.json', 'ABORTED_UNRECORDED.json')


def attempts(entries, stage_id):
    """The one authoritative attempt rule (review round 5 #3), used by seal consumption, status, retry accounting
    and the terminal decision. An attempt (an attempt_start, or a legacy receipt without one) is 'aborted' if a
    stage_failure names it or its directory carries an abort marker, else 'committed' if its stage_seal was
    published, else 'reconciled' if a reconciliation names it, else 'unfinished'. Receipts are matched to their
    attempt by output_dir (a reconciliation also by its attempt_start_seq), else to the latest earlier start."""
    out, by_dir = [], {}
    keys = {'stage_seal': 'seal', 'stage_failure': 'failure', 'attempt_reconciliation': 'reconciled'}
    for e in entries:
        payload = e['payload']
        if payload.get('stage') != stage_id:
            continue
        if e['kind'] == 'attempt_start':
            attempt = {'start': e, 'output_dir': payload.get('output_dir'), 'seal': None, 'failure': None, 'reconciled': None}
            out.append(attempt)
            by_dir[payload.get('output_dir')] = attempt
        elif e['kind'] in keys:
            target = by_dir.get(payload.get('output_dir')) if payload.get('output_dir') is not None else None
            if target is None and payload.get('attempt_start_seq') is not None:
                target = next((a for a in out if a['start'] is not None and a['start']['seq'] == payload['attempt_start_seq']),
                              None)
            if target is None and out and payload.get('output_dir') is None:
                target = out[-1]
            if target is None:  # a receipt without a start (legacy): an attempt of its own
                target = {'start': None, 'output_dir': payload.get('output_dir'), 'seal': None, 'failure': None,
                          'reconciled': None}
                out.append(target)
                by_dir[payload.get('output_dir')] = target
            if target[keys[e['kind']]] is None:
                target[keys[e['kind']]] = e
    for attempt in out:
        marked = attempt['output_dir'] is not None and any((Path(attempt['output_dir']) / m).exists() for m in ABORT_MARKERS)
        attempt['state'] = ('aborted' if attempt['failure'] is not None or (attempt['seal'] is not None and marked)
                            else 'committed' if attempt['seal'] is not None
                            else 'reconciled' if attempt['reconciled'] is not None else 'unfinished')
    return out


def require_seal(manifest, stage_id):
    """The latest committed attempt of an upstream stage (the authoritative rule), re-verified against its output
    manifest bytes; an aborted attempt is never usable."""
    entries = manifest.entries()
    committed = [a for a in attempts(entries, stage_id) if a['state'] == 'committed']
    if not committed:
        aborted = [a for a in attempts(entries, stage_id) if a['state'] == 'aborted' and a['seal'] is not None]
        _require(not aborted, f'{stage_id} attempt {aborted[-1]["output_dir"] if aborted else ""} was aborted; its '
                              f'outputs are not citable')
    _require(committed, f'upstream stage {stage_id} is not sealed in the digest manifest')
    entry = max((a['seal'] for a in committed), key=lambda e: e['seq'])
    out = Path(entry['payload']['output_dir'])
    _require(reg.file_sha256(out / 'manifest.json') == entry['payload']['manifest_sha256'],
             f'{stage_id} output manifest differs from its seal (deterministic integrity)')
    listed = json.loads((out / 'manifest.json').read_text())['artifact_sha256']
    for name, sha in listed.items():
        _require(reg.file_sha256(out / name) == sha, f'{stage_id} artifact {name} differs from its seal')
    return out, entry


def unfinished_start(manifest, stage_id):
    """The attempt start of a stage's latest attempt if it ended without any receipt, or None."""
    real = [a for a in attempts(manifest.entries(), stage_id) if a['start'] is not None]
    return real[-1]['start'] if real and real[-1]['state'] == 'unfinished' else None


def attempt_gate(manifest, stage_id):
    """Registered attempts (design 6, PM decision review #5), by the authoritative rule: a committed stage never
    reruns; the attempt-start receipt precedes the stage directory, so a start counts even when the process dies;
    a third start refuses; a retry after a deterministic-integrity failure needs a bug-fix addendum after it, after
    a start without any receipt a recorded reconciliation (ABORTED_UNRECORDED)."""
    entries = manifest.entries()
    every = attempts(entries, stage_id)
    _require(not any(a['state'] == 'committed' for a in every), f'{stage_id} is already sealed; a registered stage is not rerun')
    real = [a for a in every if a['start'] is not None]
    _require(len(real) < 2, f'{stage_id}: two attempts already started; a third start refuses')
    if real:
        last = real[-1]
        if last['state'] == 'unfinished':
            _require(False, f'{stage_id}: the previous attempt ended without a receipt (it counts); record a '
                            f'reconciliation before the next start')
        if last['state'] == 'aborted' and last['failure'] is not None \
                and last['failure']['payload']['class'] == 'deterministic_integrity':
            _require(any(e['kind'] == 'bugfix_addendum' and e['seq'] > last['failure']['seq']
                         and e['payload'].get('stage') == stage_id for e in entries),
                     f'{stage_id}: a retry after an integrity failure needs a recorded bug-fix addendum')
    return {'starts': len(real)}


def stage_status(manifest, stages=('A2-S0', 'A2-S1', 'A2-S2', 'A2-S3')):
    """Per stage, by the authoritative rule: 'sealed' (a committed attempt); 'FAILED_INTEGRITY' (both attempts
    used, the last aborted by a deterministic integrity failure); 'DECISION_NEEDED' (both attempts used
    otherwise); 'incomplete' (an attempt remains)."""
    entries = manifest.entries()
    out = {}
    for stage_id in stages:
        every = attempts(entries, stage_id)
        real = [a for a in every if a['start'] is not None]
        if any(a['state'] == 'committed' for a in every):
            out[stage_id] = 'sealed'
        elif len(real) < 2:
            out[stage_id] = 'incomplete'
        else:
            last = real[-1]
            integrity = last['failure'] is not None and last['failure']['payload']['class'] == 'deterministic_integrity'
            out[stage_id] = 'FAILED_INTEGRITY' if integrity else 'DECISION_NEEDED'
    return out


def smoke_gate(manifest, stage_id):
    """G-SMOKE: A2-S0 and A2-S1 start only if the sealed smoke's projected cost (projection term
    included) fits their hang guard; otherwise the stage is not started (decision needed; guards
    unchanged). The record is read from the sealed A2-SM stage (its bytes are covered by the seal)."""
    out, _ = require_seal(manifest, 'A2-SM')
    record = json.loads((out / 'smoke_record.json').read_text())
    _require(record['projection']['within_guard'].get(stage_id) is True,
             f'G-SMOKE: the projected cost of {stage_id} exceeds its hang guard; not started (decision needed)')
    return record


# ---------------------------------------------------------------- sources, git and destination
def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def committed(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO):
        return False
    try:
        blob = subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(REPO).as_posix()}'], cwd=REPO,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return blob == path.read_bytes()


def new_file_hashes(root=REPO):
    return {rel: reg.file_sha256(Path(root) / rel) for rel in reg.NEW_FILES}


def code_pins(config, manifest=None):
    """The registered code identity, replaced by the latest recorded bug-fix addendum (design 6)."""
    pins = config['registration_input_pins']
    out = {'execution_source_commit': pins['execution_source_commit'], 'new_files_sha256': pins['new_files_sha256']}
    if manifest is not None:
        fix = manifest.latest('bugfix_addendum')
        if fix is not None:
            out = {'execution_source_commit': fix['payload']['fix_commit'],
                   'new_files_sha256': fix['payload']['new_files_sha256']}
    return out


def enforce_sources(config, config_path, state, manifest=None):
    """Protected sources byte-identical to the base; new files equal their pins; HEAD's code equals the
    effective execution commit (registration commits may follow); config committed."""
    reg.verify_protected_sources(REPO)
    pins = code_pins(config, manifest)
    _require(new_file_hashes() == pins['new_files_sha256'], 'new source/test files differ from their registration pins')
    _require(state['ancestor'] and not state['changed'], 'code at HEAD differs from the registered execution source commit')
    _require(not state['dirty'], 'uncommitted changes under the code paths; refusing to run')
    _require(state['config_committed'], 'the registered config is not committed at HEAD byte for byte')
    return state


def check_output(output, config, protected):
    """PM decision (review #8): a fresh directory directly under the pinned output root, and the root is
    neither inside nor around any sealed input or protected root (resolved real paths)."""
    root = Path(config['registration_input_pins']['output_root']).resolve()
    output = Path(output).resolve()
    _require(output.parent == root and not output.exists(), f'output must be a new directory directly under {root}')
    for name, other in protected.items():
        _require(reg._disjoint(root, other), f'the output root must be disjoint from {name} ({Path(other).resolve()})')
    return output


def check_manifest_location(manifest, protected):
    """PM decision (review round 2 #8): the external digest manifest lies outside every sealed input and both
    checkouts (results/ included), by resolved real paths; checked before any manifest mutation."""
    for name, other in protected.items():
        _require(reg.manifest_outside(manifest.path, other),
                 f'the digest manifest must be outside {name} ({Path(other).resolve()})')
    return manifest


# ---------------------------------------------------------------- guards
def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if platform.system() == 'Darwin' else raw * 1024)


def free_memory_bytes():
    """Free + inactive + speculative pages (macOS vm_stat, bounded by the registered sampling timeout; a
    timeout propagates as subprocess.TimeoutExpired) or MemAvailable (Linux); None if unknown."""
    try:
        if platform.system() == 'Darwin':
            text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True,
                                  timeout=reg.MEMORY_SAMPLE_TIMEOUT_SECONDS).stdout
            size = int(text.split('page size of')[1].split()[0])
            pages = {k.strip(): int(v.strip().rstrip('.')) for k, v in (
                line.split(':', 1) for line in text.splitlines()[1:] if ':' in line)}
            return size * sum(pages.get(k, 0) for k in ('Pages free', 'Pages inactive', 'Pages speculative'))
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None
    return None


class Guards:
    """RSS cap (8 GiB) and free memory (>= 4 GiB) checked at every call."""
    def __init__(self, *, rss_cap=reg.RSS_CAP_GIB * GIB, free_min=reg.FREE_MEMORY_GIB * GIB, rss=peak_rss_bytes,
                 free=free_memory_bytes):
        self.rss_cap, self.free_min, self.rss, self.free = rss_cap, free_min, rss, free

    def __call__(self):
        if self.rss() > self.rss_cap:
            raise MemoryGuardExceeded('peak RSS above the registered cap')
        try:
            free = self.free()
        except subprocess.TimeoutExpired:
            raise MemoryGuardExceeded('free-memory sampling exceeded its registered timeout (fail closed)') from None
        if free is None:  # review round 4 #3: an unavailable or unparsable measurement is never "enough memory"
            raise MemoryGuardExceeded('free memory could not be measured (fail closed)')
        if free < self.free_min:
            raise MemoryGuardExceeded('free memory below the registered guard; stopping to protect the recorder')


class CommandGuard:
    """One command's guards (PM decisions, review #4 rounds 1-4): the window rule at entry; the registered hang
    guard H as the SIGALRM hard cap, armed from entry to the cleanup; work checkpoints (memory + elapsed time)
    that stop at H minus the finalization reserve; the final check of a completion receipt (memory, no pending
    interrupt, and the cleanup reserve of H left), run between its staging write and its atomic rename; a memory
    watchdog thread through every code path; interrupts held as pending inside a publication (whatever thread
    received the signal) and never raised inside a receipt; a bounded cleanup."""
    def __init__(self, stage_id, env, entered=None):
        self.stage_id, self.env = stage_id, env
        self.seconds = float(reg.HANG_GUARDS[GUARD_OF[stage_id]])
        self.reserve = min(reg.FINALIZATION_RESERVE_SECONDS, self.seconds / 10)
        self.cleanup = min(reg.CLEANUP_RESERVE_SECONDS, self.seconds / 40)
        self.work_seconds = self.seconds - self.reserve
        self.seal_seconds = self.seconds - self.cleanup
        self.start, self.t0 = entered if entered is not None else (env.now(), env.monotonic())
        _require(reg.stage_window_ok(self.start, self.seconds),
                 f'{stage_id}: the command would not end strictly before 06:30 KST within its hang guard, or it '
                 f'overlaps the 06:30-13:30 KST protection window; not started')
        self.memory = env.guards()
        self.armed, self.previous, self.watchdog, self.stop, self.violation = False, {}, None, None, None
        self.lock, self.closed, self.late = threading.Lock(), False, None
        self.deferring, self.pending = 0, None

    def elapsed(self):
        return self.env.monotonic() - self.t0

    def remaining(self):
        """The time left before the hard cap."""
        return self.seconds - self.elapsed()

    def io_budget(self):
        """How long a bounded I/O step may still wait: the time left before the hard cap minus the bounded
        cleanup's own share (the watchdog join), so the command still ends before the cap."""
        return self.remaining() - min(reg.WATCHDOG_JOIN_SECONDS + reg.IO_CLEANUP_SLACK_SECONDS, self.cleanup / 2)

    def check(self):
        """Work checkpoint: the memory guard, then the work deadline (hang guard minus the reserve)."""
        self.memory()
        if self.elapsed() > self.work_seconds:
            raise HangGuardExceeded('registered hang guard minus the finalization reserve reached (work checkpoint)')

    def final_check(self):
        """The final guard check of a completion receipt (between its staging write and its atomic rename): the
        memory guard, enough of the hard cap left for the rename and the bounded cleanup (the cleanup reserve),
        and, after the sample, no interrupt pending (review round 5 #1). Any failure aborts the publication."""
        self.memory()
        if self.elapsed() > self.seal_seconds:
            raise HangGuardExceeded('not enough of the registered hang guard is left to seal and clean up (final check)')
        self.publication_boundary()

    def publication_boundary(self):
        """Immediately before the atomic rename of a completion receipt: a pending interrupt, a watchdog violation
        or a guard signal not yet delivered aborts it (no seal is published with a pending interrupt)."""
        if self.pending is not None:
            error, self.pending = self.pending, None
            raise error
        if self.violation is not None:
            raise self.violation
        if threading.current_thread() is threading.main_thread() and signal.sigpending() & set(GUARD_SIGNALS):
            raise HangGuardExceeded('a guard interrupt is pending at the publication boundary')

    def _on_alarm(self, signum, frame):
        error = HangGuardExceeded('registered hang guard exceeded; partial outputs preserved, not citable')
        if self.deferring:
            self.pending = self.pending or error
            return
        raise error

    def _on_memory(self, signum, frame):
        error = self.violation or MemoryGuardExceeded('memory watchdog')
        if self.deferring:
            self.pending = self.pending or error
            return
        raise error

    def raise_pending(self):
        """After a published receipt: a held interrupt takes effect now."""
        if self.pending is not None and not self.deferring:
            error, self.pending = self.pending, None
            raise error

    def defer_to_late(self):
        """After a published completion receipt: a held interrupt is reported after the cleanup (the receipt,
        published by its atomic rename after the final check, stands)."""
        if self.pending is not None and not self.deferring:
            self.late, self.pending = self.late or self.pending, None

    def arm(self):
        if not self.env.alarm:
            return
        remaining = self.seconds - self.elapsed()
        if remaining <= 0:
            raise HangGuardExceeded('registered hang guard exceeded before arming')
        self.previous = {signal.SIGALRM: signal.signal(signal.SIGALRM, self._on_alarm),
                         signal.SIGUSR1: signal.signal(signal.SIGUSR1, self._on_memory)}
        signal.setitimer(signal.ITIMER_REAL, float(remaining))
        self.armed, self.stop = True, threading.Event()
        ACTIVE_GUARDS.append(self)
        main = threading.main_thread().ident

        def watch():  # one interrupt at most, never after the cleanup started
            signal.pthread_sigmask(signal.SIG_BLOCK, GUARD_SIGNALS)  # process signals go to the main thread only
            while not self.stop.wait(reg.WATCHDOG_INTERVAL_SECONDS):
                try:
                    self.memory()
                    continue
                except MemoryGuardExceeded as error:
                    violation = error
                except Exception as error:  # a sampling failure stops the command (fail closed)
                    violation = MemoryGuardExceeded(f'memory sampling failed: {type(error).__name__}: {error}')
                with self.lock:
                    if not self.closed:
                        self.violation = violation
                        signal.pthread_kill(main, signal.SIGUSR1)
                return
        self.watchdog = threading.Thread(target=watch, name='a2-memory-watchdog', daemon=True)
        self.watchdog.start()

    def _disarm_once(self):
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, GUARD_SIGNALS)  # nothing interrupts the cleanup
        try:
            with self.lock:
                self.closed = True  # the watchdog can no longer signal
            if self.stop is not None:
                self.stop.set()
            if self.watchdog is not None:
                self.watchdog.join(timeout=reg.WATCHDOG_JOIN_SECONDS)  # bounded; a stuck sampler is abandoned
                self.watchdog = None
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig in signal.sigpending() & set(GUARD_SIGNALS):
                signal.sigwait({sig})  # consumed while blocked: never delivered after the handlers are restored
                self.late = self.late or (HangGuardExceeded('registered hang guard reached during the cleanup')
                                          if sig == signal.SIGALRM else
                                          self.violation or MemoryGuardExceeded('memory watchdog (during the cleanup)'))
            for sig, handler in self.previous.items():
                signal.signal(sig, handler)
            if self in ACTIVE_GUARDS:
                ACTIVE_GUARDS.remove(self)
            self.armed = False
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def disarm(self):
        """Bounded and idempotent cleanup (interrupts held meanwhile); returns the guard interrupt that arrived
        while it ran or after a published completion receipt (raised by the caller after the cleanup)."""
        self.deferring += 1
        try:
            while self.armed:
                try:
                    self._disarm_once()
                except (HangGuardExceeded, MemoryGuardExceeded) as error:
                    self.late = self.late or error
        finally:
            self.deferring -= 1
        if self.pending is not None:
            self.late, self.pending = self.late or self.pending, None
        return self.late


@contextmanager
def command_guard(stage_id, env, entered=None):
    guard = CommandGuard(stage_id, env, entered)
    LAST_GUARD[:] = [guard]
    guard.arm()
    late = None
    try:
        guard.check()
        yield guard
    finally:
        late = guard.disarm()
    if late is not None:  # the body succeeded but a guard interrupt arrived during the cleanup
        raise late


def sha256sums(output):
    """sha256 of every file of a stage directory except macOS AppleDouble sidecars (a ._* name with the 00051607
    magic, written beside files on the exFAT output volume; review: a ._* file without the magic stays an
    artifact). The sidecars are never sealed, so ones appearing before or after the seal change nothing."""
    names = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and not is_appledouble(p))
    return {name: reg.file_sha256(output / name) for name in names}


def json_safe(value, path='$', nulled=None):
    """JSON-safe copy (review round 2 #1): numpy scalars and arrays to Python, nonfinite floats to null with
    their paths collected in ``nulled`` (invalid flags, counts and reasons are kept by the records)."""
    nulled = [] if nulled is None else nulled
    if isinstance(value, dict):
        return {str(k): json_safe(v, f'{path}.{k}', nulled) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, f'{path}[{i}]', nulled) for i, v in enumerate(value)]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist(), path, nulled)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return float(value)
        nulled.append(path)
        return None
    return value


def dump(path, value):
    """Strict JSON (no NaN): nonfinite values are written as null and listed in nonfinite_nulled_paths."""
    nulled = []
    safe = json_safe(value, '$', nulled)
    if nulled:
        _require(isinstance(safe, dict), 'a nonfinite value in a non-object record')
        safe['nonfinite_nulled_paths'] = nulled
    Path(path).write_text(json.dumps(safe, indent=1, ensure_ascii=False, allow_nan=False) + '\n')


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


@contextmanager
def stage(output, stage_id, identity, *, manifest, final_check, start=None):
    """Attempt-start receipt (before the directory exists), fresh stage directory, outputs, SHA256SUMS and
    manifest.json; an early final check; then the completion receipt (stage seal), staged, checked by the
    final guard check and committed by an atomic rename. Any failure before the commit keeps the directory,
    writes a non-citable ABORTED record and appends a classified failure receipt; no seal is published."""
    output = Path(output)
    _require(not output.exists(), f'stage output exists; preserve it and use a new attempt directory: {output}')
    started = (start or datetime.now(timezone.utc)).isoformat()
    manifest.append('attempt_start', {'stage': stage_id, 'output_dir': str(output), 'started_utc': started})
    t0, cpu0 = time.perf_counter(), time.process_time()
    publishing = False
    try:
        output.mkdir(parents=True)
        dump(output / 'started.json', {'stage': stage_id, 'started_utc': started, **identity})
        yield output
        sums = sha256sums(output)
        (output / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name, sha in sums.items()))
        dump(output / 'manifest.json', {'stage': stage_id, 'cost': {
            'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - cpu0,
            'peak_rss_bytes': peak_rss_bytes()},
            'artifact_sha256': {**sums, 'SHA256SUMS': reg.file_sha256(output / 'SHA256SUMS')},
            **{k: identity[k] for k in ('config_digest', 'git_commit', 'seeds') if k in identity}})
        final_check()  # after the last write (early stop)
        publishing = True
        # the completion receipt: staged, final guard check, atomic rename (review round 4 #2)
        manifest.append('stage_seal', {'stage': stage_id, 'output_dir': str(output),
                                       'manifest_sha256': reg.file_sha256(output / 'manifest.json')},
                        commit_check=final_check)
    except BaseException as error:
        if publishing and _seal_committed(manifest, stage_id, output):
            # the seal was committed by its rename before this error: it stands (review round 7 #1); report only
            manifest.durability_errors.append(error)
            raise
        kind = ('infrastructure' if isinstance(error, (HangGuardExceeded, MemoryGuardExceeded))
                else 'deterministic_integrity' if isinstance(error, IntegrityError) else 'error')
        payload = {'stage': stage_id, 'class': kind, 'output_dir': str(output), 'aborted': True,
                   'during_publication': publishing}
        if output.is_dir():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            record = {'stage': stage_id, 'status': 'ABORTED', 'class': kind, 'error_type': type(error).__name__,
                      'error': str(error), 'traceback': traceback.format_exc()[-reg.TRACEBACK_TAIL_CHARS:],
                      'during_publication': publishing, 'seconds': time.perf_counter() - t0,
                      'peak_rss_bytes': peak_rss_bytes(), 'citable': False, 'partial_outputs_preserved': True}
            dump(output / f'failure-{stamp}.json', record)
            dump(output / 'ABORTED.json', record)  # non-citable; no completion receipt was published
            payload['failure_sha256'] = reg.file_sha256(output / f'failure-{stamp}.json')
        manifest.append('stage_failure', payload)
        raise


def _seal_committed(manifest, stage_id, output):
    """Whether this attempt's completion receipt is published: known without I/O once its atomic rename returned
    in this process (review round 8 #1), else read from the manifest by the bounded writer (False when the read
    fails or exceeds its bound: review round 9 #8)."""
    seal = lambda e: (e['kind'] == 'stage_seal' and e['payload'].get('stage') == stage_id
                      and e['payload'].get('output_dir') == str(output))
    if any(seal(e) for e in manifest.committed):
        return True
    try:  # bounded (review round 9 #8)
        return any(seal(e) for e in bounded_io(manifest.entries, 'the manifest read (seal check)'))
    except POSTCOMMIT_ERRORS:  # an unreadable or slow manifest is not proof of a commit
        return False


def identity_record(config, git_commit, extra=None):
    return {'experiment': reg.EXPERIMENT_ID, 'config_digest': reg.config_digest(config), 'git_commit': git_commit,
            'seeds': {'game_bootstrap': reg.GAME_BOOTSTRAP_SEED, 'a2_base': reg.A2_BASE, 'roles': reg.ROLE_VALUES,
                      'world_seeds': {r['k']: reg.world_seed(r['k']) for r in reg.S5X_RUNS}},
            'python': sys.executable, 'python_version': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, **(extra or {})}


# ================================================================ pinned readers (TRAIN-only for the smoke)
def _stat_day(value):
    if isinstance(value, bytes):
        value = value.decode()
    stamp = pd.Timestamp(value)
    return (stamp.tz_localize(None) if stamp.tzinfo is not None else stamp).normalize()


def _naive_days(values):
    days = pd.to_datetime(pd.Series(values).reset_index(drop=True))
    if getattr(days.dt, 'tz', None) is not None:
        days = days.dt.tz_localize(None)
    return days.dt.normalize()


def parquet_plan(path, columns, window, *, strict):
    """Row groups of one parquet file whose game_date statistics overlap ``window``, from the footer only
    (schema and statistics; no row is decoded). ``strict`` (the smoke): each selected row group must carry
    statistics and lie inside TRAIN, otherwise refuse (before the attempt starts)."""
    import pyarrow.parquet as pq
    path = Path(path)
    _require('2026' not in path.name, 'a 2026 path reached a reader; stopping')
    meta = pq.ParquetFile(path).metadata
    names = [meta.schema.column(i).name for i in range(meta.num_columns)]
    _require(all(c in names for c in columns), f'{path.name} lacks a listed column')
    index = names.index('game_date')
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    train_lo, train_hi = pd.Timestamp(reg.TRAIN_WINDOW[0]), pd.Timestamp(reg.TRAIN_WINDOW[1])
    groups, rows = [], 0
    for rg in range(meta.num_row_groups):
        stats = meta.row_group(rg).column(index).statistics
        if stats is None or not stats.has_min_max:
            _require(not strict, f'{path.name} row group {rg} has no game_date statistics; the TRAIN-only read refuses')
        else:
            first, last = _stat_day(stats.min), _stat_day(stats.max)
            if last < lo or first > hi:
                continue  # never decoded
            if strict:
                _require(first >= train_lo and last <= train_hi,
                         f'{path.name} row group {rg} holds non-TRAIN dates ({first.date()}..{last.date()}); the '
                         f'TRAIN-only read refuses')
        groups.append(rg)
        rows += meta.row_group(rg).num_rows
    return {'file': path.name, 'row_groups': groups, 'rows': int(rows), 'num_row_groups': meta.num_row_groups}


def read_planned(path, plan, columns, window, *, checkpoint=lambda: None, log=None):
    """Decode only the planned row groups and listed columns; keep the rows dated inside ``window``."""
    import pyarrow.parquet as pq
    handle = pq.ParquetFile(path)
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    frames = []
    for rg in plan['row_groups']:
        checkpoint()
        frame = handle.read_row_group(rg, columns=list(columns)).to_pandas()
        if log is not None:
            log.append({'file': Path(path).name, 'row_group': int(rg), 'columns': list(columns)})
        days = _naive_days(frame['game_date'])
        frames.append(frame.loc[((days >= lo) & (days <= hi)).to_numpy()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))


def verify_processed_pin(pins, local):
    """PM decision (review #6): the A2 processed-frame pin is the registered loader's file and equals the
    inherited quality manifest and the file bytes, checked before any row is decoded."""
    root = Path(local['artifact_root'])
    pin = pins['processed_frame_identity']
    _require(Path(pin['path']).resolve() == (root / 'processed' / 'pitches.parquet').resolve(),
             'processed_frame_identity path differs from the registered processed frame')
    quality = json.loads((root / 'reports' / 'data_quality.json').read_text())
    _require(quality.get('processed_sha256') == pin['sha256'],
             'the A2 processed-frame pin differs from the inherited quality manifest (processed_sha256)')
    _require(reg.file_sha256(pin['path']) == pin['sha256'], 'processed frame bytes differ from the A2 pin')
    return pin


def require_split_aligned(path):
    """Footer only (no row decoded): every row group of a processed file carries game_date statistics and lies
    wholly inside or wholly outside TRAIN; a file with a row group straddling a TRAIN boundary is refused."""
    import pyarrow.parquet as pq
    path = Path(path)
    meta = pq.ParquetFile(path).metadata
    _require(meta.num_row_groups > 0, f'{path.name} has no row group; the processed file is refused')
    names = [meta.schema.column(i).name for i in range(meta.num_columns)]
    _require('game_date' in names, f'{path.name} lacks game_date')
    index = names.index('game_date')
    lo, hi = pd.Timestamp(reg.TRAIN_WINDOW[0]), pd.Timestamp(reg.TRAIN_WINDOW[1])
    for rg in range(meta.num_row_groups):
        stats = meta.row_group(rg).column(index).statistics
        _require(stats is not None and stats.has_min_max,
                 f'{path.name} row group {rg} has no game_date statistics; the processed file is not split-aligned')
        first, last = _stat_day(stats.min), _stat_day(stats.max)
        _require(last < lo or first > hi or (first >= lo and last <= hi),
                 f'{path.name} row group {rg} ({first.date()}..{last.date()}) straddles a TRAIN boundary; the processed '
                 f'file is not split-aligned and is refused')
    return meta.num_row_groups


def verify_aligned_pin(pins):
    """PM decision (the prepared input of A3 EXP-BC-CAL-01, reused): the split-aligned, row-for-row identical copy
    that every TRAIN-only read decodes, verified before any row is decoded: the identity-manifest bytes equal their
    pin; its certificate is checked as A3 checks it (the registered preparation tool; result PASS; aligned; identity
    PASS with equal rows and schema, no differing column and every column digest equal, the TRAIN columns included;
    equal source and rewritten rows); it certifies exactly this copy (rewritten sha256 = the copy's pin) made from
    the identity source (source sha256 = processed_frame_identity.sha256 = the copy's source pin); the copy's bytes
    equal their pin; and its footer (row count = the certified rows) is split-aligned (``require_split_aligned``)."""
    import pyarrow.parquet as pq
    aligned, identity = pins['processed_frame_aligned'], pins['processed_frame_identity']
    _require(aligned['source_sha256'] == identity['sha256'],
             'the aligned copy names another source than the processed-frame identity pin')
    raw = Path(aligned['identity_manifest_path']).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == aligned['identity_manifest_sha256'],
             'the aligned identity manifest differs from its pin')
    manifest = json.loads(raw)
    _require(isinstance(manifest, dict) and manifest.get('tool') == reg.ALIGNED_TOOL,
             'the aligned identity manifest was not written by the registered A3 preparation tool')
    certificate = manifest.get('identity') if isinstance(manifest.get('identity'), dict) else {}
    _require(manifest.get('result') == reg.ALIGNED_MANIFEST_RESULT and manifest.get('aligned') is True
             and certificate.get('result') == reg.ALIGNED_MANIFEST_RESULT,
             'the aligned identity manifest does not certify a PASS (result, alignment and identity)')
    _require(certificate.get('rows_equal') is True and certificate.get('schema_equal') is True,
             'the aligned identity manifest does not certify equal rows and schema')
    _require(certificate.get('differing_columns') == [], 'the aligned identity manifest lists differing columns')
    columns = certificate.get('columns')
    _require(isinstance(columns, dict) and columns
             and all(isinstance(c, dict) and c.get('equal') is True for c in columns.values()),
             'a column digest of the aligned identity manifest is not equal')
    _require(set(TRAIN_COLUMNS) <= set(columns), 'the aligned identity manifest does not cover every TRAIN column')
    source = manifest.get('source') if isinstance(manifest.get('source'), dict) else {}
    rewritten = manifest.get('rewritten') if isinstance(manifest.get('rewritten'), dict) else {}
    _require(type(source.get('rows')) is int and source.get('rows') == rewritten.get('rows'),
             'the aligned identity manifest certifies different source and rewritten row counts')
    _require(source.get('sha256') == identity['sha256'],
             'the aligned identity manifest source is not the processed-frame identity pin')
    _require(rewritten.get('sha256') == aligned['sha256'],
             'the aligned identity manifest certifies another file than the pinned copy')
    _require(reg.file_sha256(aligned['path']) == aligned['sha256'], 'aligned processed frame bytes differ from the A2 pin')
    _require(pq.ParquetFile(aligned['path']).metadata.num_rows == rewritten['rows'],
             'the aligned copy footer row count differs from its certificate')
    require_split_aligned(aligned['path'])
    return aligned


def raw_files_for(window, local):
    """The approved raw files whose season overlaps the window (the others are never opened)."""
    first, last = int(str(window[0])[:4]), int(str(window[1])[:4])
    for name in reg.RAW_ALLOWLIST:
        if first <= int(name.split('_')[1].split('.')[0]) <= last:
            yield name, Path(local['raw_root']) / name


def plan_train_reads(pins, local, window, *, strict):
    """Footer-only plan of a TRAIN read after the pins are verified (the processed-frame identity pin first, then
    its split-aligned copy): the row groups of the aligned copy (always TRAIN-only: strict) and of the approved raw
    files (``strict`` for the smoke) that would be decoded, and the aligned rows of the TRAIN row groups (an upper
    bound for the smoke's labelled projection)."""
    _require(reg.TRAIN_WINDOW[0] <= window[0] <= window[1] <= reg.TRAIN_WINDOW[1], 'a TRAIN read stays inside TRAIN')
    verify_processed_pin(pins, local)  # the original stays the identity source
    aligned = verify_aligned_pin(pins)
    plan = {'window': list(window), 'strict': bool(strict), 'processed_path': aligned['path'],
            'processed': parquet_plan(aligned['path'], TRAIN_COLUMNS, window, strict=True), 'raw': {}}
    _require(plan['processed']['row_groups'], 'no TRAIN row group of the aligned copy overlaps the window')
    for name, path in raw_files_for(window, local):
        _require(reg.file_sha256(path) == pins['train_raw_files_sha256'][name], f'raw file {name} differs from its pin')
        plan['raw'][name] = parquet_plan(path, RAW_COLUMNS, window, strict=strict)
    _require(plan['raw'], 'no approved raw file overlaps the window')
    plan['train_rows_upper_bound'] = parquet_plan(aligned['path'], ('game_date',), reg.TRAIN_WINDOW, strict=True)['rows']
    return plan


def join_raw(frame, raw):
    """Raw quality columns joined on the pitch key (unique on both sides); launch columns from the
    processed frame when present, else from the approved raw file."""
    _require(not raw.duplicated(KEY).any() and not frame.duplicated(KEY).any(),
             'join keys are not unique (deterministic integrity)')
    joined = frame[KEY].merge(raw.drop(columns=['game_date']), on=KEY, how='left', validate='one_to_one',
                              suffixes=('', '_raw'))
    frame = frame.copy()
    for column in ('hc_x', 'hc_y', 'bb_type', 'estimated_woba_using_speedangle'):
        frame[column] = joined[column].to_numpy()
    for column in ('launch_speed', 'launch_angle'):
        frame[column] = frame[column].where(frame[column].notna(), joined[column].to_numpy())
    return frame


def train_rows(pins, local, window, *, strict, checkpoint=lambda: None, log=None):
    """Regular-season TRAIN rows inside ``window`` (pinned processed frame, listed columns) with the raw
    quality columns of the overlapping approved files joined. Returns (frame, read record)."""
    plan = plan_train_reads(pins, local, window, strict=strict)
    frame = read_planned(plan['processed_path'], plan['processed'], TRAIN_COLUMNS, window, checkpoint=checkpoint, log=log)
    rows_all_types = int(len(frame))
    frame = frame.loc[frame['game_type'].eq('R').to_numpy()].reset_index(drop=True)
    _require(bool(frame['split'].eq('train').all()), 'a TRAIN-window row is outside the train split (deterministic integrity)')
    raw = pd.concat([read_planned(Path(local['raw_root']) / name, entry, RAW_COLUMNS, window, checkpoint=checkpoint,
                                  log=log) for name, entry in plan['raw'].items()], ignore_index=True)
    frame = join_raw(frame, raw)
    osg.guard_train(frame['game_date'])
    return frame, {'plan': plan, 'rows_in_window_all_types': rows_all_types, 'regular_season_rows': int(len(frame))}


def raw_rows(pins, local, window, *, checkpoint=lambda: None):
    """A2-S3: the approved raw quality columns of the rehearsal window (pinned files, listed columns)."""
    frames = []
    for name, path in raw_files_for(window, local):
        _require(reg.file_sha256(path) == pins['train_raw_files_sha256'][name], f'raw file {name} differs from its pin')
        frames.append(read_planned(path, parquet_plan(path, RAW_COLUMNS, window, strict=False), RAW_COLUMNS, window,
                                   checkpoint=checkpoint))
    _require(frames, 'no approved raw file overlaps the rehearsal window')
    return pd.concat(frames, ignore_index=True)


# ================================================================ A2-S0 (pure on TRAIN rows)
TAB_OUTCOME_OF_10 = np.array([0, 1, 2, 4, 4, 4, 4, 4, 3, 4])  # ball, strike, foul, in play x5, hbp, double play
TB_WEIGHTS = np.array([0., 1., 2., 3., 4., 0.])  # out, single, double, triple, home run, double play


def _bases(frame):
    if 'bases' in frame:
        return frame['bases'].to_numpy(np.int64)
    return sum((frame[c].notna().to_numpy().astype(np.int64) << i) for i, c in enumerate(('on_1b', 'on_2b', 'on_3b')))


def _groups(frame, names):
    return osg.group_of_types(frame['pitch_type'].to_numpy(object), names)


def tab_marginals_from_train(train, game_values, *, cells, classes, index, table, signs, signs_prime, kappa_p, zeta_p,
                             first_half, second_half, seed_role='tab_world',
                             min_pitches=reg.IMPLEMENTATION['tab_world']['archetype_min_train_pitches']):
    """TRAIN marginals of the TAB world (A2-S0): pitch outcome frequencies, logging-law inputs, the
    eight archetypes and 64 situations (fixed seed role draws), x laws and the in-play mix."""
    from pitchmdp.model import outcome_labels
    groups = reg.TAB_GROUPS
    g = _groups(train, groups)
    labels = outcome_labels(train)
    count = train['balls'].to_numpy(np.int64) * 3 + train['strikes'].to_numpy(np.int64)
    ok = (g != None) & (labels >= 0) & (train['balls'].between(0, 3) & train['strikes'].between(0, 2)).to_numpy()  # noqa: E711
    gi = np.array([groups.index(x) if x is not None else -1 for x in g])
    outcome = TAB_OUTCOME_OF_10[np.maximum(labels, 0)]
    freq = np.zeros((osx.N_COUNTS, len(groups), 5))
    np.add.at(freq, (count[ok], gi[ok], outcome[ok]), 1.)
    pooled = freq.sum(0)  # per group over counts: the fallback of an empty count x group cell
    _require((pooled.sum(-1) > 0).all(), 'every TAB pitch group needs TRAIN pitches')
    empty = freq.sum(-1) == 0
    outcome_freq = np.where(empty[..., None], pooled[None] / pooled.sum(-1)[None, :, None],
                            freq / np.maximum(freq.sum(-1, keepdims=True), 1))
    cg = freq.sum(-1)
    count_group = cg / cg.sum(-1, keepdims=True)
    league = cg.sum(0) / cg.sum()
    pitchers = train['pitcher'].to_numpy()
    per = pd.DataFrame({'pitcher': pitchers[ok], 'g': gi[ok]}).groupby(['pitcher', 'g']).size().unstack(fill_value=0)
    per = per.reindex(columns=range(len(groups)), fill_value=0)
    eligible = per.loc[per.sum(axis=1) >= min_pitches]
    _require(len(eligible) >= reg.TAB_ARCHETYPES, 'not enough TRAIN pitchers for the TAB archetypes')
    order = sorted(eligible.index, key=str)
    pick = np.random.default_rng(reg.role_sequence(seed_role, 0, 1)).choice(len(order), reg.TAB_ARCHETYPES, replace=False)
    shares = np.array([eligible.loc[order[i]].to_numpy(float) / eligible.loc[order[i]].sum() for i in pick])
    first = train.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).drop_duplicates(['game_pk', 'at_bat_number'])
    rows = np.random.default_rng(reg.role_sequence(seed_role, 0, 0)).choice(len(first), reg.TAB_SITUATIONS, replace=False)
    situations = [osx.Situation.from_models(GameState.from_row(first.iloc[int(i)]), game_values['we'],
                                            game_values['advancement']) for i in rows]
    x_dist, mix = x_laws(train, cells, index, reg.S5X_GROUPS)
    return {'groups': list(groups), 'outcome_freq': outcome_freq, 'count_group_freq': count_group,
            'league_shares': league, 'archetype_shares': shares, 'x_dist': x_dist[:len(groups)],
            'bip_mix': mix[:len(groups)] / mix[:len(groups)].sum(), 'class_table': table,
            'first_half_table': first_half, 'second_half_table': second_half,
            'situations': [situation_doc(s) for s in situations],
            'signs': [signs[x] for x in groups], 'signs_prime': [signs_prime[x] for x in groups],
            'kappa_p': kappa_p, 'zeta_p': zeta_p, 'n_train_bip': int(len(cells)),
            'outcome_cells_pooled_fallback': int(empty.sum())}


def x_laws(train, cells, index, names):
    """P(x | group, two strikes) [G, 2, 480] and the (group, two strikes, dp) counts of the in-play rows.
    A (group, two strikes) cell without TRAIN in-play rows falls back to the two-strike-pooled law of all
    groups (reported in the census)."""
    g = _groups(train.iloc[index], names)
    gi = np.array([names.index(x) if x is not None else -1 for x in g])
    ts = (train['strikes'].to_numpy()[index] == 2).astype(int)
    keep = gi >= 0
    counts = np.zeros((len(names), 2, reg.N_X))
    np.add.at(counts, (gi[keep], ts[keep], cells[keep] // 2), 1.)
    mix = np.zeros((len(names), 2, 2))
    np.add.at(mix, (gi[keep], ts[keep], cells[keep] % 2), 1.)
    total = counts.sum(-1, keepdims=True)
    pooled = counts.sum(0)  # [2, 480] over groups
    _require((pooled.sum(-1) > 0).all(), 'TRAIN in-play rows are needed with and without two strikes')
    law = np.where(total > 0, counts / np.maximum(total, 1), (pooled / pooled.sum(-1, keepdims=True))[None])
    return law, mix


def situation_doc(s):
    st = s.state
    return {'state': {'inning': st.inning, 'half': st.half, 'outs': st.outs, 'bases': st.bases,
                      'home_score': st.home_score, 'away_score': st.away_score},
            'outcomes': {e: [list(o) for o in s.outcomes[e]] for e in reg.TERMINALS9}}


def situation_from_doc(doc):
    st = doc['state']
    state = GameState(st['inning'], st['half'], st['outs'], st['bases'], st['home_score'], st['away_score'])
    return osx.Situation(state, {e: [tuple(o) for o in rows] for e, rows in doc['outcomes'].items()})


def marginals_from_doc(doc):
    values = {k: doc[k] for k in osx.TabMarginals.FIELDS}
    values['situations'] = [situation_from_doc(s) for s in doc['situations']]
    for k in ('outcome_freq', 'count_group_freq', 'league_shares', 'archetype_shares', 'x_dist', 'bip_mix', 'class_table',
              'first_half_table', 'second_half_table'):
        values[k] = np.asarray(values[k], dtype=np.float64)
    values['groups'] = tuple(values['groups'])
    return osx.TabMarginals(**values)


def zeta_inputs(train, cells, classes, index, game_values, names, checkpoint=lambda: None):
    """Per eligible TRAIN out (runner on third, < 2 outs): actual runs - frozen C0 expectation, the C0
    scoring mass and run gap (zeta calibration), with the pitch group. C0 is a frozen, pre-registered
    model (nothing is fitted here), the registered exception to cross-fitting."""
    rows = train.iloc[index]
    bases, outs = _bases(rows), rows['outs_when_up'].to_numpy()
    sel = (classes == osg.OUT_INDEX) & ((bases & 4) > 0) & (outs < 2)
    part = rows.loc[sel]
    groups = _groups(part, names)
    cache, residual, p_score, run_gap = {}, [], [], []
    for n, ((_, row), b) in enumerate(zip(part.iterrows(), bases[sel])):
        if n % 2000 == 0:
            checkpoint()
        state = GameState(int(row.inning), str(row.inning_topbot), int(row.outs_when_up), int(b), int(row.home_score),
                          int(row.away_score))
        if state not in cache:
            outcomes, p, gap = osg.out_scoring_split(state, game_values['advancement'], game_values['we'])
            cache[state] = (sum(q * r for q, r, _ in outcomes), p, gap)
        expected, p, gap = cache[state]
        bat = (row.post_home_score - row.home_score) if row.inning_topbot == 'Bot' else (row.post_away_score - row.away_score)
        residual.append(float(bat) - expected)
        p_score.append(p)
        run_gap.append(gap)
    return np.array(residual), np.array(p_score), np.array(run_gap), groups


def xwoba_diagnostic(train, cells, index, table):
    """TRAIN Pearson correlation of MLB estimated_woba_using_speedangle with the table's P_hat(hit | x, dp)
    and expected total bases (descriptive only; never a surrogate)."""
    if 'estimated_woba_using_speedangle' not in train:
        return {'status': 'NOT_AVAILABLE', 'reason': 'the column was not joined'}
    x = pd.to_numeric(train['estimated_woba_using_speedangle'], errors='coerce').to_numpy(dtype=np.float64)[index]
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return {'status': 'NOT_AVAILABLE', 'reason': 'fewer than three used in-play rows carry the column'}
    table = np.asarray(table)
    hit = osg.hit_probability(table, cells)[ok]
    total_bases = (table[np.asarray(cells, dtype=np.int64)] @ TB_WEIGHTS)[ok]

    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    return {'status': 'COMPUTED', 'n': int(ok.sum()), 'corr_hit_probability': corr(x[ok], hit),
            'corr_expected_total_bases': corr(x[ok], total_bases),
            'label': 'TRAIN correlation diagnostic only; not a surrogate (design R1_b)'}


SMOKE_MIN_PITCHES = 50  # the smoke's short TRAIN subset: archetype eligibility only (outputs discarded)
PROVENANCE = {'kappa_residuals': 'cross-fitted: game_pk parity two folds, P_hat refitted on the other fold',
              'zeta_residuals': reg.IMPLEMENTATION['calibration']['zeta_residuals'],
              'rows': 'TRAIN rows only (2023-05-15..2025-04-30), regular season'}


def build_s0(train, game_values, *, smoke=False, seed_role='tab_world', fixed_m=None,
             min_pitches=reg.IMPLEMENTATION['tab_world']['archetype_min_train_pitches'], checkpoint=lambda: None):
    """A2-S0 on guarded TRAIN rows (raw columns already joined). ``smoke``: the same code path with a
    fixed m, no cross-validated log-loss, no delta_g, kappa_p or zeta_p (zero signs), outputs discarded."""
    timings = {}
    t = time.perf_counter()
    osg.guard_train(train['game_date'])
    cells, classes, index, census = osg.train_bip_rows(train)
    folds = (train['game_pk'].to_numpy()[index] % 2).astype(np.int64)
    timings['rows_and_mapping'] = time.perf_counter() - t
    checkpoint()
    t = time.perf_counter()
    fit = osg.fit_r1b(cells, classes, folds, m=fixed_m if smoke else None)
    timings['table_fit'] = time.perf_counter() - t
    checkpoint()
    m, table = fit['m'], fit['table']
    dates = pd.to_datetime(train['game_date']).to_numpy()[index]
    median = np.sort(dates)[len(dates) // 2] if len(dates) else None
    halves = [dates <= median, dates > median]
    _require(all(h.any() for h in halves), 'both TRAIN halves need in-play rows (era drift cell)')
    half_tables = [osg.fit_table(osg.class_counts(cells[h], classes[h]), m) for h in halves]
    out = {'census': census, 'm': m, 'cv_log_loss': fit['cv_log_loss'], 'r1_table': table,
           'r1_table_sha256': osg.table_sha256(table, m), 'timings': timings, 'provenance': PROVENANCE}
    zero = {g: 0 for g in reg.S5X_GROUPS}
    if smoke:
        signs, signs_prime, kappa, zeta = zero, zero, {'S5X': 0.0, 'TAB': 0.0}, {'S5X': 0.0, 'TAB': 0.0}
    else:
        groups = _groups(train.iloc[index], reg.S5X_GROUPS)
        residual = osg.crossfit_hit_residuals(cells, classes, folds, groups, m)
        delta = osg.group_residuals(residual, groups, reg.S5X_GROUPS)
        hit = osg.hit_probability(table, cells)
        checkpoint()
        z_res, p_score, run_gap, z_groups = zeta_inputs(train, cells, classes, index, game_values, reg.S5X_GROUPS,
                                                        checkpoint)
        delta_prime = osg.group_residuals(z_res, z_groups, reg.S5X_GROUPS)
        kappa, zeta = {}, {}
        for world, names in (('S5X', reg.S5X_GROUPS), ('TAB', reg.TAB_GROUPS)):
            kappa[world] = osg.calibrate_kappa(hit, groups, delta, names)
            zeta[world] = osg.calibrate_zeta(p_score, run_gap, z_groups, delta_prime, names)
            checkpoint()
        signs = kappa['S5X']['signs']
        signs_prime = zeta['S5X']['signs']
        out.update(train_residuals={'hit': delta, 'run': delta_prime}, kappa_calibration=kappa, zeta_calibration=zeta,
                   xwoba_train_diagnostic=xwoba_diagnostic(train, cells, index, table))
        kappa = {w: v['kappa_p'] for w, v in kappa.items()}
        zeta = {w: v['zeta_p'] for w, v in zeta.items()}
    out.update(signs=signs, signs_prime=signs_prime, kappa_p=kappa, zeta_p=zeta)
    t = time.perf_counter()
    out['tab_marginals'] = tab_marginals_from_train(
        train, game_values, cells=cells, classes=classes, index=index, table=table, signs=signs,
        signs_prime=signs_prime, kappa_p=kappa['TAB'], zeta_p=zeta['TAB'], first_half=half_tables[0],
        second_half=half_tables[1], seed_role=seed_role, min_pitches=min_pitches)
    x_dist, _ = x_laws(train, cells, index, reg.S5X_GROUPS)
    out['s5x_layers'] = {'groups': list(reg.S5X_GROUPS), 'x_dist': x_dist, 'class_table': table,
                         'signs': [signs[g] for g in reg.S5X_GROUPS], 'signs_prime': [signs_prime[g] for g in reg.S5X_GROUPS],
                         'kappa_p': kappa['S5X'], 'zeta_p': zeta['S5X']}
    timings['marginals'] = time.perf_counter() - t
    checkpoint()
    return out


S0_REQUIRED = ('census', 'm', 'cv_log_loss', 'r1_table', 'r1_table_sha256', 'provenance', 'train_residuals',
               'kappa_calibration', 'zeta_calibration', 'xwoba_train_diagnostic', 'kappa_p', 'zeta_p', 'tab_marginals',
               's5x_layers')


def require_s0_descriptive(s0):
    missing = [k for k in S0_REQUIRED if k not in s0]
    _require(not missing, f'A2-S0 descriptive output missing: {missing} (stage integrity)')
    return True


# ================================================================ A2-S1 TAB (pure on sealed A2-S0 outputs)
S1_CELL_KEYS = ('cell', 'delta_true', 'beta_0', 'tau', 'expected_delta_hat_0', 'delta_true_surrogate', 'pairs', 'selections')
S1_PAIR_KEYS = ('b_hat', 'se', 'sd', 'rmse', 'invalid_replications', 'not_evaluable_replications', 'R')
S1_SURROGATE_KEYS = ('R1_0_realized', 'R1_a', 'R1_b', 'R1_a_surrogate_share', 'R1_b_surrogate_share')


def require_s1_descriptive(results, cells):
    """Every descriptive TAB output the design requires (PM decision, review #10): each registered cell
    is recorded or listed as not run after the guard, with its per-pair statistics, selections and the
    exact surrogate-truth decomposition; a missing output is a stage integrity failure."""
    _require(set(results['cells']) | set(results['not_run']) == {c['id'] for c in cells}
             and not set(results['cells']) & set(results['not_run']), 'TAB cells missing (stage integrity)')
    for cid, record in results['cells'].items():
        _require(all(k in record for k in S1_CELL_KEYS), f'TAB cell {cid}: descriptive output missing (stage integrity)')
        _require(set(record['pairs']) == set(reg.PAIRS) and all(all(k in v for k in S1_PAIR_KEYS)
                                                                for v in record['pairs'].values()),
                 f'TAB cell {cid}: per-pair outputs missing (stage integrity)')
        _require(all(k in (record['delta_true_surrogate'] or {}) for k in S1_SURROGATE_KEYS),
                 f'TAB cell {cid}: the surrogate-truth decomposition is missing (stage integrity)')
        if record['cell']['role'] == 'acceptance':
            _require(all('outcome' in v for v in record['pairs'].values()), f'TAB cell {cid}: outcomes missing')
    wanted = {c['id'] for c in cells}
    _require(set(results['acceptance']) == {c['id'] for c in reg.TAB_ACCEPTANCE if c['id'] in wanted},
             'TAB acceptance cells missing (stage integrity)')
    return True


TAB_NOT_RUN_REASON = ('the work deadline (hang guard minus the finalization reserve) was reached after every acceptance '
                      'cell was sealed; descriptive cell not run and never rerun (design 4.6)')


def stage_s1(s0, *, write_cell, write_not_run=lambda cell, reason: None, deadline=lambda: None, cells=None,
             replications=None):
    """Acceptance cells first (each written and sealed on completion), then the descriptive cells in
    the registered order; the first step checks every V2 cell's baseline DP expectation = DP truth. A work
    deadline during the descriptive cells writes each remaining cell as a NOT_RUN record (review round 2 #15),
    inside the finalization reserve, and the stage seals."""
    marginals = marginals_from_doc(s0['tab_marginals'])
    eta_p = osx.TabWorld.eta_p(marginals)
    cells = reg.tab_cells() if cells is None else cells
    worlds = lambda cell: osx.TabWorld(marginals, law=cell['law'], eta=reg.scale(cell['eta'], eta_p),
                                       kappa=reg.scale(cell['kappa'], marginals.kappa_p),
                                       zeta=reg.scale(cell['zeta'], marginals.zeta_p), drift=cell['drift'])
    integrity = {}
    for cell in cells:
        if cell['law'] == 'V2':
            deadline()
            world = worlds(cell)
            osx.tab_integrity(world)
            integrity[cell['id']] = {'expected_delta_hat_0_minus_truth': world.expected_delta_hat_0 - world.delta_true}
    results = {'eta_p': eta_p, 'first_step_v2_dp_check': integrity, 'cells': {}, 'not_run': []}
    for cell in cells:
        try:
            record = osx.tab_cell(worlds(cell), cell, n_train=marginals.n_train_bip, replications=replications,
                                  deadline=deadline)
        except HangGuardExceeded:
            if cell['role'] == 'acceptance':
                raise  # before every acceptance cell is sealed: an infrastructure failure
            results['not_run'] = [c['id'] for c in cells if c['index'] >= cell['index']]
            for remaining in (c for c in cells if c['index'] >= cell['index']):
                write_not_run(remaining, TAB_NOT_RUN_REASON)
            break
        results['cells'][cell['id']] = record
        write_cell(cell, record)
    results['acceptance'] = {c['id']: results['cells'][c['id']]['pairs'] for c in reg.TAB_ACCEPTANCE
                             if c['id'] in results['cells']}
    require_s1_descriptive(results, cells)
    return results


# ================================================================ A2-S2 summary and A2-S4 decision (pure)
INVARIANT_RUNS = tuple(r['k'] for r in reg.S5X_RUNS if r['setting'] == 'X1' and r['zeta'] == '0')
S2_RUN_OF = {'2*kappa_p': next(r['k'] for r in reg.S5X_RUNS if r['role'] == 'S2' and r['kappa'] == '2*kappa_p'),
             '2*zeta_p': next(r['k'] for r in reg.S5X_RUNS if r['role'] == 'S2' and r['zeta'] == '2*zeta_p')}


def require_s2_descriptive(runs):
    """Every descriptive S5-X output the design requires: per run the six repetitions, every pair and the
    inherited S5 SE; per X1 run the S-2 cells, the per-pair cells and the explicit NOT_RUN marker of the
    first-order reference-trajectory approximation; per zeta = 0 X1 run the invariant record."""
    _require([r['run']['k'] for r in runs] == [r['k'] for r in reg.S5X_RUNS], 'S5-X runs missing (stage integrity)')
    for record in runs:
        k = record['run']['k']
        _require(len(record.get('repetitions') or []) == reg.S5X_REPETITIONS and 'inherited_s5_se' in record
                 and set(record.get('pairs') or {}) == set(reg.PAIRS), f'S5-X run {k}: descriptive output missing')
        if record['run']['setting'] == 'X1':
            marker = record.get('first_order_reference_approximation') or {}
            _require({'s2', 's2_pairs'} <= set(record) and marker.get('status') == 'NOT_RUN' and marker.get('reason'),
                     f'S5-X run {k}: S-2 outputs or the NOT_RUN marker missing (stage integrity)')
            if record['run']['zeta'] == '0':
                _require(record.get('invariant') is not None, f'S5-X run {k}: invariant record missing (stage integrity)')
    return True


def s5x_summary(runs):
    """S-1/S-4 nets, per-pair S-2 cells (runs 4 and 5: an invalid repetition of the pair in that run makes
    its cell INCONCLUSIVE, never PASS) and the gamma_a invariant (runs 4, 9, 10) from the run records."""
    nets = osx.nets(runs)
    by_k = {r['run']['k']: r for r in runs}
    recorded = [by_k[k].get('invariant') for k in INVARIANT_RUNS if k in by_k]
    invariant = ('PASS' if S2_RUN_OF['2*kappa_p'] in by_k and recorded
                 and all(r is not None and r.get('outcome') == 'PASS' for r in recorded) else 'INVALID')
    s2 = {}
    for pair in reg.PAIRS:
        if reg.split_pair(pair)[0] not in ('R1_a', 'R1_b'):
            continue
        s2[pair] = {'invariant': invariant}
        for label, k in S2_RUN_OF.items():
            s2[pair][label] = (((by_k.get(k) or {}).get('s2_pairs') or {}).get(pair)
                               or reg.inconclusive_invalid(f'S5-X run {k} has no S-2 record for {pair}'))
    return {'nets': nets, 's2': s2}


def require_s5x_consistent(s5x):
    """The S-1/S-2/S-4 summary recomputed from the sealed run records equals the stored one (JSON round trip),
    so an invalid repetition's INCONCLUSIVE propagation survives a restart (review round 2 #1)."""
    again = json.loads(json.dumps(json_safe(s5x_summary(s5x['runs'])), allow_nan=False))
    _require(again == {'nets': s5x['nets'], 's2': s5x['s2']},
             'the stored S5-X summary differs from its run records (deterministic integrity)')
    return True


def stage_s4(public, tab, s5x, *, integrity_failures, r3_status):
    odc.check_inputs(public, tab, s5x)
    return odc.decide(public, tab, s5x, integrity_failures=integrity_failures, r3_status=r3_status)


# ================================================================ A2-SM smoke (pure parts)
def smoke_tab(s0_subset, *, replications, guards=lambda: None):
    """A2-S1 code path at kappa = zeta = 0, eta = 0, seed role smoke: integrity checks (V2 DP expectation
    = truth, same-seed bit reproducibility) and wall per replication; every estimate is discarded."""
    choice = reg.IMPLEMENTATION['smoke']
    marginals = marginals_from_doc(s0_subset['tab_marginals'])
    t = time.perf_counter()
    world = osx.TabWorld(marginals, law='V2', eta=choice['eta'], kappa=choice['kappa'], zeta=choice['zeta'],
                         seed_role='smoke')
    build = time.perf_counter() - t
    osx.tab_integrity(world)
    fixed = {'m': choice['fixed_m'], 'lambda': choice['fixed_lambda'], 'M': choice['fixed_M']}
    walls, first = [], None
    for j in range(replications):
        guards()
        t = time.perf_counter()
        estimates = osx.tab_replication(world, 0, j, n_train=marginals.n_train_bip, smoke=fixed)[0]
        walls.append(time.perf_counter() - t)
        if j == 0:
            first = estimates
    again = osx.tab_replication(world, 0, 0, n_train=marginals.n_train_bip, smoke=fixed)[0]
    _require(json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True),
             'smoke TAB replication is not bit-reproducible with the same seed (deterministic integrity)')
    data = world.simulate(np.random.default_rng(reg.role_sequence('smoke', 2, 0)))
    steps, lengths = data['steps'], data['lengths']
    from pitchmdp.ope_diagnostics import affine_padded
    f = osx._transform_array('shrink', choice['fixed_lambda'])
    t = time.perf_counter()
    affine_padded(steps['v_c'], steps['q_a'], f(steps['rho_c']), lengths)
    affine_padded(steps['v_r'], steps['q_a'], f(np.where(np.arange(steps['v_r'].shape[1])[None] < lengths[:, None], 1., 0.)),
                  lengths)
    variant_unit = time.perf_counter() - t
    return {'world_build_seconds': build, 'replication_seconds': walls, 'variant_estimate_unit_seconds': variant_unit,
            'integrity': {'tab_v2_dp_expectation_equals_truth': 'PASS', 'same_seed_bit_reproducible': 'PASS'},
            'replications': replications}


def smoke_projection(smoke_s0_seconds, table_fit_unit, variant_unit, replication_seconds, train_row_ratio):
    """Labelled projection (not a measurement): the excluded selection workload per replication
    (6 table fits + 30 variant estimates at the smoke unit costs) added before extrapolating."""
    per_rep = float(np.mean(replication_seconds)) if replication_seconds else math.nan
    term = 6 * table_fit_unit + 30 * variant_unit
    projected_s1 = (per_rep + term) * sum(c['R'] for c in reg.TAB_ACCEPTANCE)
    projected_s0 = (smoke_s0_seconds + 6 * table_fit_unit) * train_row_ratio
    return {'label': 'projection (not a measurement)', 'projection_term_seconds_per_replication': term,
            'A2-S1_acceptance_seconds': projected_s1, 'A2-S0_seconds': projected_s0,
            'within_guard': {'A2-S1': bool(projected_s1 <= reg.HANG_GUARDS['A2-S1']),
                             'A2-S0': bool(projected_s0 <= reg.HANG_GUARDS['A2-S0'])},
            'rule': 'a projection above its hang guard -> that stage is not started (decision needed); guards unchanged'}


# ================================================================ A2-S3 core (G-INT first)
def surrogate_rows(last_rows, raw):
    """The last pitch row of each PA with the approved raw quality columns (hc_x, hc_y; launch columns
    when the processed row lacks them)."""
    _require(not raw.duplicated(KEY).any(), 'raw join keys are not unique (deterministic integrity)')
    keys = pd.DataFrame([{k: int(row[k]) for k in KEY} for row in last_rows.values()], columns=KEY)
    joined = keys.merge(raw.drop(columns=['game_date']), on=KEY, how='left', validate='one_to_one')
    rows = {}
    for (pa_id, row), (_, extra) in zip(last_rows.items(), joined.iterrows()):
        record = {**dict(row), 'hc_x': extra['hc_x'], 'hc_y': extra['hc_y']}
        for column in ('launch_speed', 'launch_angle'):
            if column not in record or pd.isna(record[column]):
                record[column] = extra[column]
        rows[pa_id] = record
    return rows


def guarded_stream(decisions, checkpoint, every=20000):
    """The same decision rows with a checkpoint every ``every`` rows (the protected code is not changed; the
    memory watchdog covers its bootstrap)."""
    def stream():
        for n, row in enumerate(decisions()):
            if n % every == 0:
                checkpoint()
            yield row
    return stream


def rehearsal(decisions, facts, last_rows, *, raw_join, game_values, s0, references, ess_gate, r3_status,
              checkpoint=lambda: None, draws=reg.DRAWS, minimum_games=reg.MIN_GAMES, minimum_pa_starts=reg.MIN_PA_STARTS,
              minimum_pitchers=reg.MIN_PITCHERS, placebos=None):
    """PM decision (review #9): G-INT runs FIRST, on the untouched registered path (policy_estimator on the
    sealed ledger and the recomputed facts), against every listed value of both references; only after it
    passes are the raw quality columns joined, the surrogate rewards built and the variants evaluated."""
    decisions = guarded_stream(decisions, checkpoint)  # memory checks while the protected estimator reads the ledger
    result0, rows0, gint = osd.gint_gate(decisions, facts, references=references, draws=draws,
                                         minimum_games=minimum_games, minimum_pa_starts=minimum_pa_starts,
                                         ess_gate=ess_gate)
    checkpoint()
    rows = surrogate_rows(last_rows, raw_join())
    surrogates = osg.surrogate_rewards(facts, rows, osg.TerminalValues(game_values['we'], game_values['advancement']),
                                       np.asarray(s0['r1_table']))
    checkpoint()
    public, sealed, notes = osd.rehearsal_evaluate(
        decisions, facts, baseline=(result0, rows0), surrogates=surrogates, gint=gint,
        train_residuals=s0.get('train_residuals'), r1_paths={'rehearsal': osg.path_shares(surrogates),
                                                             'train_census': s0['census']},
        r3_status=r3_status, draws=draws, minimum_games=minimum_games, minimum_pitchers=minimum_pitchers,
        minimum_pa_starts=minimum_pa_starts, placebos=placebos, checkpoint=checkpoint)
    return public, sealed, notes, gint


def recompute_facts(rpv, policy, inputs, blocks, games_table, game_values, no_pitch, out, checkpoint=lambda: None):
    """Planner-free PA facts of the S6 selection: requests and structural problems, E0 membership from
    a reference-only runtime (as the S4 denominators stage), PA-end reward from the frozen C0 WE."""
    from pitchmdp import policy_requests as preq
    from pitchmdp import policy_runtime as prt
    runtime = prt.build_runtime(*rpv.registered_path(policy, 'bc'), *rpv.registered_path(policy, 'support'),
                                out / 'facts-reference-ledger.jsonl', hand_registry=rpv.registered_path(policy, 'hands'))
    we = game_values['we']
    defense = lambda state, home: float(we.predict_defense(state, home))  # = BoundComponents.defense_we
    facts, last_rows = {}, {}
    frame = inputs['store'].frame  # as the registered runner: blocks index the store's frame
    for n, (pa_id, positions) in enumerate(blocks):
        if n % 500 == 0:
            checkpoint()
        requests, problem, index = preq.pa_requests(inputs['store'], positions, runtime.sha256, no_pitch)
        info = {'game': int(frame.game_pk.iloc[positions[0]]), 'problem': problem, 'problem_index': index,
                'positions': positions, 'submitted': len(requests), **preq.pa_manifest(frame, positions)}
        if requests:
            inside, reason = runtime.start_population(requests[0].state, requests[0].pitcher_hand)
        else:
            inside, reason = False, f'unsubmittable:{problem}'
        info.update(in_population=inside, start_reason=reason)
        info.update(preq.pa_outcome(frame, positions, defense, games_table, rule='structural-end-v1',
                                    score_source='next_row'))
        facts[pa_id] = info
        last_rows[pa_id] = frame.iloc[positions[-1]]
    return facts, last_rows


def x2_world_predictor(identity):
    """X2 world predictor (design: 'a frozen <=2025 G0-family member different from the planner's
    predictor, identified at registration'). Supported kind: {'kind': 'g0_single_member', 'seed': s}."""
    _require(isinstance(identity, dict) and identity.get('kind') == 'g0_single_member'
             and type(identity.get('seed')) is int, f'unsupported X2 world-predictor identity {identity!r}')
    return lambda components: osx.SingleMemberPredictor(components.g0, identity['seed'])


def build_candidate(rpv, policy, inputs, blocks, *, seed, tag, out, budget):
    """The S5 candidate runtime (same construction as run_policy_validation.dispatch ``candidate`` for
    v2-world: dev style snapshot, S2-certified components, registered tau/setting/evaluation seed)."""
    from pitchmdp import policy_identity as pid
    from pitchmdp import policy_runtime as prt
    from pitchmdp.policy_artifacts import load_style_snapshot, load_train_bc
    from pitchmdp.rollout_policy import RowBudget
    config = policy['config']
    ident = config['identity_registration']
    stages = config['le2025_validation_plan']['stages']
    pre = rpv.prerequisites('v2-world', policy)
    freeze, setting = pre['freeze'], pre['setting']
    evaluation = rpv.role_seed(config, 'evaluation') if freeze['dr_q']['source'] == 'evaluation_seed' else None
    path, sha = rpv.registered_path(policy, 'style_dev')
    snapshot, as_of, content = load_style_snapshot(path, sha)
    _require(as_of == stages['S1b_style_snapshot']['as_of_exclusive']['dev'], 'style as-of differs from registration')
    frame = inputs['frame']
    first = pd.to_datetime(frame.game_date.iloc[np.concatenate([p for _, p in blocks])]).min()
    _require(first >= pd.Timestamp(as_of), 'evaluation blocks start before the style as-of date')
    from pitchmdp import policy_requests as preq
    check = preq.snapshot_rolling_mismatches(frame.loc[frame.split.eq('dev')], snapshot, as_of)  # as snapshot_for
    _require(check['mismatches'] == 0, f'snapshot differs from rolling priors on the as-of date: {check}')
    contexts, style = rpv.pa_contexts(inputs['store'], blocks, snapshot, as_of)
    bc_path, bc_sha = rpv.registered_path(policy, 'bc')
    support_path, support_sha = rpv.registered_path(policy, 'support')
    components = pid.bind_components(inputs['bundle_path'], inputs['bundle_sha'], inputs['paths'],
                                     bc_artifact=load_train_bc(bc_path, bc_sha), context_rows=contexts,
                                     member_loader=inputs['member_loader'], classes=ident['classes'],
                                     we_contract_sha256=ident['we_contract_sha256'])
    _require(components.sha256 == pre['components_sha256'], 'bound components differ from the S2-certified identity')
    provenance = {'style_snapshot_sha256': content, 'as_of_exclusive': as_of, 'rolling_check': check,
                  'style_report': style}
    runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / f'ledger-s5x-{tag}.jsonl',
                                components=components, budget=RowBudget(int(budget), seed_count=5),
                                tau=freeze['selected_tau'], samples=setting['samples'], pitch_cap=setting['pitch_cap'],
                                seed=seed, evaluation_seed=evaluation, hand_registry=rpv.registered_path(policy, 'hands'),
                                provenance=provenance)
    return runtime, components


# ================================================================ environment (every data access)
@dataclass
class Rehearsal:
    decisions: object
    facts: dict
    last_rows: dict
    window: tuple
    ess_gate: object
    game_values: dict


class S5XWorld:
    """The registered A2-S2 inputs: per run, the S5 candidate (run_policy_validation construction) and the
    registered E0 starts of the dev selection."""
    def __init__(self, rpv, policy, inputs, blocks, no_pitch, game_values, spec, out):
        self.rpv, self.policy, self.inputs, self.blocks, self.no_pitch = rpv, policy, inputs, blocks, no_pitch
        self.game_values, self.spec, self.out = game_values, spec, out

    def candidate(self, spec_run):
        from pitchmdp import policy_requests as preq
        runtime, components = build_candidate(self.rpv, self.policy, self.inputs, self.blocks, seed=spec_run['planning_seed'],
                                              tag=f"k{spec_run['k']}", out=self.out, budget=self.spec['row_budget_per_run'])
        starts = []
        for _, positions in self.blocks:
            requests, _, _ = preq.pa_requests(self.inputs['store'], positions, runtime.sha256, self.no_pitch)
            if requests and runtime.start_population(requests[0].state, requests[0].pitcher_hand)[0]:
                starts.append((requests[0].state, requests[0].pitcher_hand))
        return runtime, components, starts


def _resolve(value):
    return Path(value) if Path(value).is_absolute() else REPO / value


class Environment:
    """The real dependencies of a registered command; tests subclass it with synthetic inputs. Nothing
    here is called before the window check, and no data is read before the registry gate in ``preflight``."""
    alarm = True

    def __init__(self, args=None):
        self.args = args

    def now(self):
        return datetime.now(timezone.utc)

    def monotonic(self):
        return time.monotonic()

    def guards(self):
        return Guards()

    def git_state(self, config_path, code):
        source = code['execution_source_commit']
        return {'head': git('rev-parse', 'HEAD'),
                'ancestor': subprocess.call(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], cwd=REPO,
                                            stderr=subprocess.DEVNULL) == 0,
                'changed': [x for x in git('diff', '--name-only', source, 'HEAD', '--', *CODE_PATHS).splitlines() if x],
                'dirty': bool(git('status', '--porcelain', '--', *CODE_PATHS)), 'config_committed': committed(config_path)}

    def committed(self, path):
        return committed(path)

    def checkouts(self):
        """This checkout and the main checkout of its git common directory (protected roots)."""
        common = Path(git('rev-parse', '--path-format=absolute', '--git-common-dir'))
        return sorted({REPO.resolve(), common.parent.resolve()})

    def chain_documents(self, config):
        """The pinned ML-POLICY registration chain (config first, then its addenda): configuration files
        only, each sha256 verified; no stage directory is opened."""
        docs = []
        for link in config['registration_input_pins']['policy_registration_chain']:
            raw = _resolve(link['path']).read_bytes()
            _require(hashlib.sha256(raw).hexdigest() == link['sha256'], f'ML-POLICY chain link changed: {link["path"]}')
            docs.append(json.loads(raw))
        return docs

    def local_config(self, base):
        raw = Path(self.args.local_config).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == base['le2025_validation_plan']['local_config']['sha256'],
                 'local config differs from its ML-POLICY pin')
        return json.loads(raw)

    def protected_roots(self, config, local, chain):
        """Directories the output root must be disjoint from: the checkouts (results/ included), the S6-dr-a5
        stage directory, the processed frame, its split-aligned copy and identity manifest, its reports and cache,
        the raw root, the G0 bundle and WE directories and the directory of every ML-POLICY registered input."""
        pins = config['registration_input_pins']
        ident = chain[0]['identity_registration']
        artifact = Path(local['artifact_root'])
        roots = {'the S6-dr-a5 stage directory': Path(pins['s6_dr_a5_ledger_path']).parent,
                 'the processed frame directory': Path(pins['processed_frame_identity']['path']).parent,
                 'the aligned processed frame directory': Path(pins['processed_frame_aligned']['path']).parent,
                 'the aligned identity manifest directory': Path(pins['processed_frame_aligned']['identity_manifest_path']).parent,
                 'the artifact reports directory': artifact / 'reports', 'the artifact cache directory': artifact / 'cache',
                 'the raw root': Path(local['raw_root']), 'the G0 bundle directory': _resolve(ident['g0_bundle']['path']).parent}
        for role, value in ident['we_paths'].items():
            roots[f'the WE {role} directory'] = _resolve(value).parent
        for doc in chain:
            for name, entry in (doc.get('registered_inputs') or {}).items():
                roots[f'the registered input {name} directory'] = _resolve(entry['path']).parent
        for root in self.checkouts():
            roots[f'the checkout {root}'] = root
        return roots

    def heavy_lock(self, local):
        from run_ml_matrix import heavy_lock  # the common heavy lock (imported only for registered runs)
        return heavy_lock(Path(local['artifact_root']).resolve())

    def nice(self):
        os.nice(10)

    def game_values(self, base):
        """The frozen C0 WE and advancement (the ML-POLICY WE identity; model files, no rows)."""
        from pitchmdp import policy_identity as pid
        ident = base['identity_registration']
        paths = {role: _resolve(value) for role, value in ident['we_paths'].items()}
        values, _ = pid.load_pinned_we(paths, ident['we_contract_sha256'], ident['classes']['we'])
        return values

    def train_plan(self, config, local, window, *, strict):
        return plan_train_reads(config['registration_input_pins'], local, window, strict=strict)

    def train(self, config, local, window, *, strict, checkpoint, log=None):
        return train_rows(config['registration_input_pins'], local, window, strict=strict, checkpoint=checkpoint, log=log)

    def policy_registration(self, config):
        """The full ML-POLICY registration (A2-S2 / A2-S3): the pinned chain through the registered loader."""
        import run_policy_validation as rpv
        chain = config['registration_input_pins']['policy_registration_chain']
        for link in chain:
            _require(reg.file_sha256(_resolve(link['path'])) == link['sha256'], f'ML-POLICY chain link changed: {link["path"]}')
        policy = rpv.load_registration(_resolve(chain[0]['path']), [_resolve(link['path']) for link in chain[1:]])
        _require([link['sha256'] for link in policy['chain']] == [link['sha256'] for link in chain],
                 'the loaded ML-POLICY chain differs from its pins')
        return rpv, policy

    def policy_inputs(self, config, local, rpv, policy):
        verify_processed_pin(config['registration_input_pins'], local)  # before any row is decoded
        return rpv.load_inputs(policy['config'], local, store=True)

    def s6_record(self, config):
        """A2-S3 only: the pinned S6-dr-a5 stage (manifest, started.json chain, dr.json, ledger bytes)."""
        pins = config['registration_input_pins']
        ledger = Path(pins['s6_dr_a5_ledger_path'])
        stage_dir = ledger.parent
        _require(reg.file_sha256(stage_dir / 'manifest.json') == pins['s6_dr_a5_manifest_sha256'],
                 'S6-dr-a5 manifest differs from its registration pin')
        _require(reg.file_sha256(ledger) == pins['s6_dr_a5_ledger_sha256'], 'S6-dr-a5 ledger differs from its pin')
        listed = json.loads((stage_dir / 'manifest.json').read_text())['artifact_sha256']
        for name in ('started.json', 'dr.json', ledger.name):
            _require(reg.file_sha256(stage_dir / name) == listed[name], f'S6-dr-a5 {name} differs from its stage manifest')
        started = json.loads((stage_dir / 'started.json').read_text())
        _require([link['sha256'] for link in started['registration_chain']]
                 == [link['sha256'] for link in pins['policy_registration_chain']],
                 'the pinned ML-POLICY chain differs from the chain the S6-dr-a5 stage ran under')
        return ledger, json.loads((stage_dir / 'dr.json').read_text())

    def s5x_inputs(self, config, local, base, out, checkpoint):
        """A2-S2: the ML-POLICY registration, the dev S5 selection, the frozen C0 game values and a builder of
        each run's S2-certified candidate with its E0 starts."""
        rpv, policy = self.policy_registration(config)
        pcfg = policy['config']
        spec = pcfg['le2025_validation_plan']['stages']['S5_V2_V3']
        inputs = self.policy_inputs(config, local, rpv, policy)
        checkpoint()
        frame = inputs['frame']
        games = rpv.select_games(frame, 'dev', spec['n_games'], rpv.role_seed(pcfg, 'selection_salt'))
        return S5XWorld(rpv, policy, inputs, rpv.game_blocks(frame, games, 'dev'),
                        frozenset(pcfg['pa_time_rules']['R3_codes']['no_pitch_descriptions']), self.game_values(base),
                        spec, out)

    def rehearsal_inputs(self, config, local, base, ledger_path, s6_dr, out, checkpoint):
        """A2-S3: planner-free facts of the S6 selection, the last row of each PA, the rehearsal date window, the
        sealed ledger stream and the registered ESS gate."""
        from pitchmdp import policy_requests as preq
        rpv, policy = self.policy_registration(config)
        pcfg = policy['config']
        inputs = self.policy_inputs(config, local, rpv, policy)
        checkpoint()
        blocks = rpv.game_blocks(inputs['frame'], s6_dr['games'], 'dev')
        no_pitch = frozenset(pcfg['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
        game_values = self.game_values(base)
        store_frame = inputs['store'].frame
        games_table = preq.game_table(store_frame, {int(store_frame.game_pk.iloc[p[0]]) for _, p in blocks})
        facts, last = recompute_facts(rpv, policy, inputs, blocks, games_table, game_values, no_pitch, out, checkpoint)
        dates = pd.to_datetime(store_frame['game_date'].iloc[np.concatenate([p for _, p in blocks])])
        _require(dates.max() <= pd.Timestamp(reg.ALL_STAGES_MAX_DATE), 'a row after 2025 reached the rehearsal')
        return Rehearsal(decisions=self.ledger_decisions(ledger_path), facts=facts, last_rows=last,
                         window=(str(dates.min().date()), str(dates.max().date())),
                         ess_gate=pcfg['ess_gate']['thresholds'], game_values=game_values)

    def ledger_decisions(self, ledger_path):
        """The sealed ledger's decision stream (the whole hash chain is replayed and verified; read-only)."""
        from pitchmdp import policy_runtime as prt
        with Path(ledger_path).open('rb') as stream:
            first = json.loads(stream.readline())
        header = {k: v for k, v in first.items() if k not in ('kind', 'seq', 'prev', 'sha256')}
        return prt.Ledger(Path(ledger_path), header).iter_decisions


# ================================================================ commands
@dataclass
class Run:
    args: argparse.Namespace
    env: Environment
    config: dict
    base: dict
    local: dict
    manifest: DigestManifest
    check: object
    plan: object = None
    out: Path = None


def preflight(args, env):
    """Registry gate, digest, sources and review receipts (fail closed, before any data)."""
    config, _ = reg.load_config(args.config)
    reg.require_executable(config)
    manifest = DigestManifest(config['registration_input_pins']['digest_manifest_path'])
    require_config_digest(manifest, config)
    code = code_pins(config, manifest)
    state = enforce_sources(config, args.config, env.git_state(args.config, code), manifest=manifest)
    require_reviews(manifest, config, code['execution_source_commit'])
    return config, manifest, state


def plan_smoke(run):
    """Before the attempt starts: the footer-only TRAIN read plan of the smoke subset (refuses on a
    straddling row group without consuming an attempt)."""
    window = tuple(run.config['registration_input_pins']['smoke_train_subset']['game_dates'])
    return run.env.train_plan(run.config, run.local, window, strict=True)


def plan_s0(run):
    """Before the A2-S0 attempt starts (review note 10): the same footer-only plan over TRAIN (pins, the aligned
    copy's certificate and footer verified), so a failed check refuses without consuming an attempt."""
    return run.env.train_plan(run.config, run.local, reg.TRAIN_WINDOW, strict=False)


DECISION_STAGES = ('A2-SM', 'A2-S0', 'A2-S1', 'A2-S2', 'A2-S3')


def descendants(stages):
    """Every decision stage downstream of ``stages`` (registered upstream relation)."""
    out, frontier = set(), set(stages)
    while frontier:
        frontier = {s for s in DECISION_STAGES if s not in out and set(reg.UPSTREAM[s]) & (frontier | out)} - out
        out |= frontier
    return out - set(stages)


def plan_decide(run):
    """Before the A2-S4 attempt: a stage that exhausted its attempts in a deterministic integrity failure makes
    the terminal FAILED_INTEGRITY decision available at once (review round 2 #16): its unsealed descendants
    are BLOCKED, other unsealed stages NOT_RUN; otherwise every stage must be sealed."""
    status = stage_status(run.manifest, DECISION_STAGES)
    failed = [s for s in DECISION_STAGES if status[s] == 'FAILED_INTEGRITY']
    if failed:
        blocked = sorted(s for s in descendants(failed) if status[s] != 'sealed')
        return {'terminal': 'FAILED_INTEGRITY', 'stage_status': status, 'failed_integrity': failed, 'blocked': blocked,
                'not_run': sorted(s for s in DECISION_STAGES
                                  if status[s] not in ('sealed', 'FAILED_INTEGRITY') and s not in blocked)}
    _require(all(v == 'sealed' for v in status.values()), f'upstream stages are not complete (decision needed): {status}')
    return {'terminal': None, 'stage_status': status}


PLANS = {'A2-SM': plan_smoke, 'A2-S0': plan_s0, 'A2-S4': plan_decide}


def run_command(args, env, entered=None):
    """One registered command, guarded from argument parsing to its completion receipt."""
    stage_id = STAGE_OF[args.command]
    with command_guard(stage_id, env, entered) as guard:
        config, manifest, state = preflight(args, env)
        guard.check()
        _require(args.output is not None, '--output (a new directory under the pinned output root) is required')
        chain = env.chain_documents(config)
        local = env.local_config(chain[0])
        protected = env.protected_roots(config, local, chain)
        check_manifest_location(manifest, protected)  # before any manifest mutation
        output = check_output(args.output, config, protected)
        env.nice()
        # one manifest writer at a time (review round 3 #3), then every substantive verification and read under the
        # common heavy lock
        with manifest.writer(), env.heavy_lock(local):
            attempt_gate(manifest, stage_id)
            if stage_id != 'A2-S4':
                for upstream in UPSTREAM[stage_id]:
                    require_seal(manifest, upstream)
            if stage_id in ('A2-S0', 'A2-S1'):
                smoke_gate(manifest, stage_id)
            run = Run(args, env, config, chain[0], local, manifest, guard.check)
            run.plan = PLANS[stage_id](run) if stage_id in PLANS else None
            guard.check()
            identity = identity_record(config, state.get('head'), {'command': args.command})
            with stage(output, stage_id, identity, manifest=manifest, final_check=guard.final_check,
                       start=guard.start) as out:
                run.out = out
                BODIES[stage_id](run)
    return 0


def body_smoke(run):
    pins = run.config['registration_input_pins']
    window = tuple(pins['smoke_train_subset']['game_dates'])
    log = []
    t = time.perf_counter()
    train, read = run.env.train(run.config, run.local, window, strict=True, checkpoint=run.check, log=log)
    _require(read['plan']['processed']['row_groups'] == run.plan['processed']['row_groups'],
             'the smoke read differs from its pre-attempt plan (deterministic integrity)')
    load = time.perf_counter() - t
    run.check()
    t = time.perf_counter()
    s0 = build_s0(train, run.env.game_values(run.base), smoke=True, seed_role='smoke',
                  fixed_m=reg.IMPLEMENTATION['smoke']['fixed_m'], min_pitches=SMOKE_MIN_PITCHES, checkpoint=run.check)
    s0_seconds = time.perf_counter() - t
    tab = smoke_tab(s0, replications=pins['smoke_tab_replications_per_cell'], guards=run.check)
    ratio = run.plan['train_rows_upper_bound'] / max(read['rows_in_window_all_types'], 1)
    record = {'stage': 'A2-SM', 'subset': list(window), 'rows': read['regular_season_rows'],
              'rows_in_window_all_types': read['rows_in_window_all_types'], 'decoded': log, 'plan': run.plan,
              'load_seconds': load, 's0_seconds': s0_seconds, 'census_counts': s0['census'],
              'table_fit_unit_seconds': s0['timings']['table_fit'], 'tab': tab,
              'train_row_ratio': ratio, 'train_row_ratio_basis': reg.IMPLEMENTATION['smoke_projection_ratio'],
              'projection': smoke_projection(s0_seconds, s0['timings']['table_fit'], tab['variant_estimate_unit_seconds'],
                                             tab['replication_seconds'], ratio),
              'peak_rss_bytes': peak_rss_bytes(),
              'never_computed': ['cross-validated log-loss', 'm selection', 'lambda/M selection', 'delta_g', 'kappa_p',
                                 'zeta_p', 'bias, SD or RMSE', 'acceptance cells', 'H_point', 'R5 diagnostics']}
    dump(run.out / 'smoke_record.json', record)


def body_s0(run):
    train, read = run.env.train(run.config, run.local, reg.TRAIN_WINDOW, strict=False, checkpoint=run.check)
    _require({k: read['plan'][k] for k in ('processed', 'raw')} == {k: run.plan[k] for k in ('processed', 'raw')},
             'the A2-S0 read differs from its pre-attempt plan (deterministic integrity)')
    run.check()
    s0 = build_s0(train, run.env.game_values(run.base), checkpoint=run.check)
    require_s0_descriptive(s0)
    dump(run.out / 's0.json', {**s0, 'read': read})


def body_s1(run):
    s0_dir, _ = require_seal(run.manifest, 'A2-S0')
    s0 = json.loads((s0_dir / 's0.json').read_text())
    cell_dir = run.out / 'cells'
    cell_dir.mkdir()

    def write_cell(cell, record):  # each cell is sealed on completion (acceptance cells first)
        dump(cell_dir / f"{cell['id']}.json", record)
        run.manifest.append('cell_seal', {'stage': 'A2-S1', 'cell': cell['id'], 'status': 'COMPLETED',
                                          'sha256': reg.file_sha256(cell_dir / f"{cell['id']}.json")})

    def write_not_run(cell, reason):  # the permitted descriptive stop: a NOT_RUN record, sealed with the stage
        dump(cell_dir / f"{cell['id']}.json", {'cell': {k: cell[k] for k in ('id', 'law', 'eta', 'kappa', 'zeta', 'R', 'role',
                                                                            'drift', 'index')},
                                              'status': 'NOT_RUN', 'reason': reason})
        run.manifest.append('cell_seal', {'stage': 'A2-S1', 'cell': cell['id'], 'status': 'NOT_RUN',
                                          'sha256': reg.file_sha256(cell_dir / f"{cell['id']}.json")})
    dump(run.out / 'tab.json', stage_s1(s0, write_cell=write_cell, write_not_run=write_not_run, deadline=run.check))


def body_s2(run):
    """S5-X: the registered 15 runs in order (the candidate of each run and its E0 starts come from the
    environment: the ML-POLICY registration, the dev selection and the S2-certified components)."""
    from pitchmdp import policy_semisynthetic as pss
    s0_dir, _ = require_seal(run.manifest, 'A2-S0')
    s0 = json.loads((s0_dir / 's0.json').read_text())
    x2 = x2_world_predictor(run.config['registration_input_pins']['x2_world_predictor_identity'])  # refuses up front
    world = run.env.s5x_inputs(run.config, run.local, run.base, run.out, run.check)
    layer_doc = s0['s5x_layers']
    runs = []
    for spec_run in reg.S5X_RUNS:
        run.check()
        runtime, components, starts = world.candidate(spec_run)
        layers = osx.S5XLayers.from_components(
            components, world.game_values, groups=layer_doc['groups'], x_dist=layer_doc['x_dist'],
            class_table=layer_doc['class_table'], signs=layer_doc['signs'], signs_prime=layer_doc['signs_prime'],
            kappa=reg.scale(spec_run['kappa'], layer_doc['kappa_p']), zeta=reg.scale(spec_run['zeta'], layer_doc['zeta_p']))
        law = (pss.tempered_law(runtime.bc, reg.S5X_V3_ALPHA) if spec_run['law'] == 'V3'
               else (lambda bc: (lambda s: (bc.actions, bc.probabilities(s))))(runtime.bc))
        predictor = x2(components) if spec_run['setting'] == 'X2' else None
        record = osx.run_s5x(runtime, components, starts, run=spec_run, law=law, layers=layers, world_predictor=predictor,
                             logs_per_start=reg.S5X_LOGS_PER_START, truth_rollouts=reg.S5X_TRUTH_ROLLOUTS,
                             cap=reg.S5X_CAP, gamma=spec_run['setting'] == 'X1',
                             invariant=spec_run['setting'] == 'X1' and spec_run['zeta'] == '0', deadline=run.check)
        dump(run.out / f"run-k{spec_run['k']:02d}.json", record)
        runs.append(record)
    require_s2_descriptive(runs)
    dump(run.out / 's5x.json', {'runs': runs, **s5x_summary(runs)})


def body_s3(run):
    """The blinded rehearsal on the sealed S6-dr-a5 ledger (read-only), no planner call: the PA facts, the
    last rows and the ledger stream come from the environment; G-INT runs first (``rehearsal``)."""
    pins = run.config['registration_input_pins']
    _require(pins['r3_status'] == 'NOT_RUN:R3', 'R3 is enabled: the A3 artifact path is not implemented; refusing')
    s0_dir, _ = require_seal(run.manifest, 'A2-S0')
    s0 = json.loads((s0_dir / 's0.json').read_text())
    _require(osg.table_sha256(np.asarray(s0['r1_table']), s0['m']) == s0['r1_table_sha256'], 'R1 table differs from its seal')
    ledger_path, s6_dr = run.env.s6_record(run.config)  # the S6 stage directory: A2-S3 only
    data = run.env.rehearsal_inputs(run.config, run.local, run.base, ledger_path, s6_dr, run.out, run.check)
    _require(data.window[1] <= reg.ALL_STAGES_MAX_DATE, 'a row after 2025 reached the rehearsal')
    references = {'registration_quoted': osd.registered_gint_reference(run.config),
                  's6_dr_a5_record': osd.recorded_gint_reference(s6_dr)}
    public, sealed, notes, gint = rehearsal(
        data.decisions, data.facts, data.last_rows,
        raw_join=lambda: raw_rows(pins, run.local, data.window, checkpoint=run.check), game_values=data.game_values,
        s0=s0, references=references, ess_gate=data.ess_gate, r3_status=pins['r3_status'], checkpoint=run.check)
    dump(run.out / 'gint.json', gint)  # compared / not comparable keys only (no value)
    public_bytes, sealed_bytes = osd.seal(public, sealed)
    (run.out / 'public.json').write_bytes(public_bytes)
    (run.out / 'centers.sealed.json').write_bytes(sealed_bytes)
    dump(run.out / 'notes.json', notes)


def body_decide(run):
    plan = run.plan
    r3 = run.config['registration_input_pins']['r3_status']
    if plan['terminal'] == 'FAILED_INTEGRITY':
        dump(run.out / 'decision_record.json', {'label': 'FAILED_INTEGRITY', 'r3_status': r3,
                                                **{k: plan[k] for k in ('stage_status', 'failed_integrity', 'blocked',
                                                                        'not_run')}})
        return
    s1_dir, _ = require_seal(run.manifest, 'A2-S1')
    s2_dir, _ = require_seal(run.manifest, 'A2-S2')
    s3_dir, _ = require_seal(run.manifest, 'A2-S3')
    public = json.loads((s3_dir / 'public.json').read_text())
    osd.validate_public(public)
    tab = json.loads((s1_dir / 'tab.json').read_text())
    s5x = json.loads((s2_dir / 's5x.json').read_text())
    require_s5x_consistent(s5x)  # the stored summary is what the sealed run records give (INCONCLUSIVE survives)
    record = stage_s4(public, tab, s5x, integrity_failures=(), r3_status=r3)
    record['inputs_sha256'] = {'public': reg.file_sha256(s3_dir / 'public.json'), 'tab': reg.file_sha256(s1_dir / 'tab.json'),
                               's5x': reg.file_sha256(s2_dir / 's5x.json')}
    dump(run.out / 'decision_record.json', record)


def body_unblind(run):
    """Open the sealed file only after the decision record is committed at HEAD byte for byte (guarded
    like every registered command; the opened values go to the sealed unblind stage directory)."""
    s3_dir, _ = require_seal(run.manifest, 'A2-S3')
    s4_dir, _ = require_seal(run.manifest, 'A2-S4')
    record_path = run.args.decision_record
    _require(record_path is not None and run.env.committed(record_path),
             'the decision record must be committed before unblinding')
    sealed = osd.unblind((s3_dir / 'centers.sealed.json').read_bytes(),
                         json.loads((s3_dir / 'public.json').read_text())['sealed_sha256'],
                         Path(record_path).read_bytes(), reg.file_sha256(s4_dir / 'decision_record.json'),
                         committed=True)
    dump(run.out / 'unblinded.json', {'selection': sealed.get('selection'),
                                      'pairs': {p: (sealed['pairs'].get(p) or {}) for p in reg.PAIRS},
                                      'decision_record_sha256': reg.file_sha256(s4_dir / 'decision_record.json')})


BODIES = {'A2-SM': body_smoke, 'A2-S0': body_s0, 'A2-S1': body_s1, 'A2-S2': body_s2, 'A2-S3': body_s3,
          'A2-S4': body_decide, 'A2-UB': body_unblind}


# ================================================================ manifest commands (no data)
def manifest_command(args, env, entered=None):
    """register-digest, record-review, reconcile-attempt, register-addendum: under the window, memory and hang
    guards; the manifest location is checked before the append (review round 2 #4, #8)."""
    with command_guard('A2-MC', env, entered) as guard:
        config, _ = reg.load_config(args.config)
        reg.require_executable(config)
        manifest = DigestManifest(config['registration_input_pins']['digest_manifest_path'])
        chain = env.chain_documents(config)
        check_manifest_location(manifest, env.protected_roots(config, env.local_config(chain[0]), chain))
        guard.check()
        with manifest.writer():  # a second writer refuses (review round 3 #3)
            return _manifest_mutation(args, config, manifest)


def _manifest_mutation(args, config, manifest):
    digest = reg.config_digest(config)
    if args.command == 'register-digest':
        _require(args.expect_digest == digest, 'registered config digest differs from the G-CODE reviewed digest')
        _require(manifest.latest('config_digest') is None, 'the config digest is registered once')
        print(json.dumps(manifest.append('config_digest', {'digest': digest, 'config_path': str(args.config)}), indent=1))
        return 0
    require_config_digest(manifest, config)
    if args.command == 'record-review':
        _require(args.gate in ('G-REV', 'G-CODE', 'G-REG') and bool(args.review) and args.verdict in reg.RECEIPT_VERDICTS,
                 '--gate (G-REV, G-CODE or G-REG), --review and --verdict (PASS or PASS_WITH_NOTES) are required')
        _require(args.gate != 'G-CODE' or (isinstance(args.commit, str) and len(args.commit) == 40),
                 'a G-CODE receipt names the reviewed code commit (--commit, 40 hex)')
        payload = {'gate': args.gate, 'reference': args.review, 'verdict': args.verdict,
                   'notes_closed': bool(args.notes_closed), 'commit': args.commit, 'config_digest': digest}
        print(json.dumps(manifest.append('review_reference', payload), indent=1))
        return 0
    if args.command == 'reconcile-attempt':
        _require(args.stage in ATTEMPT_STAGES and bool(args.note), '--stage and --note are required')
        start = unfinished_start(manifest, args.stage)
        _require(start is not None, f'{args.stage} has no attempt start without a receipt; nothing to reconcile')
        out = Path(start['payload']['output_dir'])
        if out.is_dir():  # a stage that ended without its receipt is non-citable (review round 5 #3)
            dump(out / 'ABORTED_UNRECORDED.json', {'stage': args.stage, 'status': 'ABORTED_UNRECORDED', 'citable': False,
                                                   'attempt_start_seq': start['seq'], 'note': args.note})
        print(json.dumps(manifest.append('attempt_reconciliation', {
            'stage': args.stage, 'attempt_start_seq': start['seq'], 'output_dir': start['payload']['output_dir'],
            'status': 'ABORTED_UNRECORDED', 'note': args.note, 'counts_as_attempt': True}), indent=1))
        return 0
    _require(args.stage in ATTEMPT_STAGES and args.failing_test and args.review,
             '--stage, --failing-test and --review required')
    _require(any(e['kind'] == 'stage_failure' and e['payload']['stage'] == args.stage
                 and e['payload']['class'] == 'deterministic_integrity' for e in manifest.entries()),
             'a bug-fix addendum follows a recorded deterministic-integrity failure of that stage')
    _require(not git('status', '--porcelain', '--', *CODE_PATHS), 'commit the fix first (clean code paths)')
    reg.verify_protected_sources(REPO)
    entry = manifest.append('bugfix_addendum', {'stage': args.stage, 'fix_commit': git('rev-parse', 'HEAD'),
                                                'new_files_sha256': new_file_hashes(), 'failing_test': args.failing_test,
                                                'review_reference': args.review})
    print(json.dumps(entry, indent=1))
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('--config', type=Path, default=REPO / reg.CONFIG_PATH)
    parser.add_argument('--local-config', type=Path, default=PROJECT / 'configs' / 'local.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expect-digest')
    parser.add_argument('--decision-record', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--failing-test')
    parser.add_argument('--review')
    parser.add_argument('--gate')
    parser.add_argument('--verdict')
    parser.add_argument('--notes-closed', action='store_true')
    parser.add_argument('--commit')
    parser.add_argument('--note')
    return parser.parse_args(argv)


def main(argv=None, env=None):
    env = env if env is not None else Environment()
    entered = (env.now(), env.monotonic())  # the guards count from argument parsing
    args = parse(argv)
    env.args = args
    if args.command == 'check':
        config, sha = reg.load_config(args.config)
        print(json.dumps({'status': config['status'], 'config_file_sha256': sha, 'config_digest': reg.config_digest(config),
                          'protected_sources': 'unchanged' if reg.verify_protected_sources(REPO) else None}, indent=1))
        return 0
    if args.command == 'pins':
        print(json.dumps({'new_files_sha256': new_file_hashes(), 'protected_sources': reg.PROTECTED_SOURCES}, indent=1))
        return 0
    if args.command in MANIFEST_COMMANDS:
        return manifest_command(args, env, entered)
    return run_command(args, env, entered)


def cli(argv=None, env=None):
    """The command-line entry (review round 8 #1): the final report runs in a bounded writer and the process
    exits at once (os._exit: no unbounded flush at interpreter exit). An argparse exit passes its code through
    (--help 0, a usage error 2; review round 9 #6); a failure exits nonzero (3 for a guard stop, else 1) with its
    message and the tail of its traceback (review round 9 #7); a committed seal stands."""
    try:
        code, message = main(argv, env) or 0, None
    except SystemExit as error:  # argparse (or an explicit exit): an int or None code passes through
        code, message = exit_code(error.code)
    except BaseException as error:  # noqa: BLE001 - every failure exits nonzero
        code = 3 if isinstance(error, (HangGuardExceeded, MemoryGuardExceeded)) else 1
        message = (f'run_ope_a2: {type(error).__name__}: {error}\n'
                   f'{traceback.format_exc()[-reg.TRACEBACK_TAIL_CHARS:]}').rstrip('\n')
    os._exit(finish(code, message))


def exit_code(code):
    """SystemExit semantics: None is 0, an int passes through, any other value is reported and exits 1."""
    if code is None:
        return 0, None
    if isinstance(code, int):
        return int(code), None
    return 1, f'run_ope_a2: {code}'


def finish(code, message=None):
    """The bounded final report: the failure message (if any) and the flush of the standard streams, waited
    for at most min(REPORT_TIMEOUT_SECONDS, the time left before the latest command's hard cap). Returns the
    exit code: nonzero when the report could not complete (3 when the time ran out, 1 on an error)."""
    box = {}

    def report():
        signal.pthread_sigmask(signal.SIG_BLOCK, GUARD_SIGNALS)
        try:
            if message is not None:
                sys.stderr.write(message + '\n')
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
            box['done'] = True
        except BaseException as error:  # noqa: BLE001 - reported through the exit code
            box['error'] = error
    bound = reg.REPORT_TIMEOUT_SECONDS
    if LAST_GUARD:
        bound = min(bound, LAST_GUARD[-1].remaining())
    if bound > 0:
        worker = threading.Thread(target=report, name='a2-report', daemon=True)
        worker.start()
        worker.join(bound)
    if box.get('done'):
        return code
    return code or (1 if 'error' in box else 3)


if __name__ == '__main__':
    sys.exit(cli())
