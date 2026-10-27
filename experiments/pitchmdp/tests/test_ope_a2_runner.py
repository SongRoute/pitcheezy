"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-01 stage runner: fail-closed refusals, the external append-only
digest manifest, review receipts, attempt-start receipts and reconciliation, the command-wide window,
hang and memory guards, destination disjointness, the TRAIN-only pinned smoke reader (files, row groups
and columns watched), the processed-frame pin, G-INT before any surrogate, the descriptive-output
checks and dependency-injected orchestration (smoke -> A2-S0 -> A2-S1 -> A2-S4 refusal). Every input is
a synthetic file the test writes to a temporary directory; no real data, ledger or model is read.
"""
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import copy
from datetime import datetime
import functools
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))

from ope_a2_fixtures import (KINDS, ToyAdvancement, ToyWE, synthetic_ledger, synthetic_pa, synthetic_table,  # noqa: E402
                             synthetic_train_frame)
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp import ope_sealed as osd  # noqa: E402
from pitchmdp import ope_surrogate as osg  # noqa: E402
from pitchmdp import ope_synthetic_a2 as osx  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402
import run_ope_a2 as roa  # noqa: E402
from test_ope_a2_estimators import CENSUS, _decision_inputs  # noqa: E402
from test_ope_a2_registry import load, registered  # noqa: E402

GAME_VALUES = {'we': ToyWE(), 'advancement': ToyAdvancement()}
AFTERNOON = datetime(2026, 10, 27, 14, 0, tzinfo=reg.KST)
REAL_PARQUET_FILE = pq.ParquetFile
REAL_STAGE_S1 = roa.stage_s1


def kst(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=reg.KST)


def tmpdir(test):
    path = Path(tempfile.mkdtemp(prefix='a2-runner-')).resolve()
    test.addCleanup(shutil.rmtree, path, True)
    return path


class ClockEnv(roa.Environment):
    """No data: a fixed clock, a settable monotonic clock, no SIGALRM, an injectable memory guard."""
    alarm = False

    def __init__(self, now=AFTERNOON, guards=None):
        super().__init__()
        self._now, self._guards, self.clock = now, guards or (lambda: None), 0.0

    def now(self):
        return self._now

    def monotonic(self):
        return self.clock

    def guards(self):
        return self._guards


# ---------------------------------------------------------------- a synthetic registered workspace
def _frame(start, days, seed, n_pas, n_pitchers=12):
    frame = synthetic_train_frame(seed=seed, n_pas=n_pas, n_pitchers=n_pitchers, start=start, days=days)
    frame['game_pk'] = frame['game_pk'] + 100000 * seed  # unique pitch keys across the synthetic groups
    frame['game_type'] = 'R'
    dates = pd.to_datetime(frame['game_date'])
    frame['split'] = np.select([dates < '2023-05-15', dates <= '2025-04-30', dates <= '2025-06-30'],
                               ['history', 'train', 'calibration'], 'dev')
    rng = np.random.default_rng(seed)
    frame['estimated_woba_using_speedangle'] = np.where(frame['description'].eq('hit_into_play').to_numpy(),
                                                        rng.uniform(0, 1.5, len(frame)), np.nan)
    return frame


def write_groups(path, frames, columns):
    """One parquet row group per frame (game_date statistics are written per row group)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    schema = pa.Table.from_pandas(pd.concat(frames, ignore_index=True)[columns], preserve_index=False).schema
    with pq.ParquetWriter(path, schema) as writer:
        for frame in frames:
            writer.write_table(pa.Table.from_pandas(frame[columns].reset_index(drop=True), preserve_index=False).cast(schema))


def write_aligned_copy(processed, aligned, manifest, frames, *, manifest_hook=None):
    """A synthetic split-aligned copy of ``processed`` (one row group per frame) and its identity manifest in the
    A3 prepare_behavior_calibration_input schema (tool, result, aligned, the identity certificate with its column
    digests, source and rewritten sha256 and rows)."""
    write_groups(aligned, frames, roa.TRAIN_COLUMNS)
    rows = REAL_PARQUET_FILE(aligned).metadata.num_rows
    columns = {c: {'original': f'synthetic-{c}', 'rewritten': f'synthetic-{c}', 'equal': True} for c in roa.TRAIN_COLUMNS}
    record = {'experiment': 'EXP-BC-CAL-01', 'tool': reg.ALIGNED_TOOL, 'synthetic': True, 'aligned': True,
              'identity': {'rows_equal': True, 'schema_equal': True, 'columns': columns, 'differing_columns': [],
                           'result': 'PASS'},
              'result': 'PASS', 'source': {'path': str(processed), 'sha256': reg.file_sha256(processed), 'rows': rows},
              'rewritten': {'path': str(aligned), 'sha256': reg.file_sha256(aligned), 'rows': rows}}
    if manifest_hook is not None:
        manifest_hook(record)
    manifest.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    return {'path': str(aligned), 'sha256': reg.file_sha256(aligned), 'identity_manifest_path': str(manifest),
            'identity_manifest_sha256': reg.file_sha256(manifest), 'source_sha256': reg.file_sha256(processed)}


ALIGNED_NAME = 'pitches.split-aligned.parquet'


class Workspace:
    """A registered synthetic configuration with its own processed frame (row groups: 0 history 2023-04,
    1 TRAIN 2024-04, 2 the TRAIN subset, 3 DEV 2025-08), its split-aligned copy with an identity manifest (by
    default the same row groups; ``aligned_groups`` replaces them), raw files, ML-POLICY chain, digest manifest
    and review receipts. Nothing real is referenced."""
    def __init__(self, test, *, smoke=('2024-06-01', '2024-06-20'), subset_start='2024-06-01', subset_days=20,
                 reps=1, receipts=True, pins=None, digest_rel=None, hook=None, aligned_groups=None, manifest_hook=None,
                 original_groups=None):
        self.root = tmpdir(test)
        self.artifacts, self.raw = self.root / 'artifacts', self.root / 'raw'
        groups = [_frame('2023-04-01', 20, 1, 300), _frame('2024-04-01', 20, 2, 300),
                  _frame(subset_start, subset_days, 3, 6000), _frame('2025-08-01', 20, 4, 2400, n_pitchers=40)]
        self.groups = groups
        self.processed = self.artifacts / 'processed' / 'pitches.parquet'
        write_groups(self.processed, groups if original_groups is None else original_groups(groups), roa.TRAIN_COLUMNS)
        (self.artifacts / 'reports').mkdir()
        self.quality = self.artifacts / 'reports' / 'data_quality.json'
        self.quality.write_text(json.dumps({'processed_sha256': reg.file_sha256(self.processed), 'synthetic': True}))
        self.aligned = self.root / 'inputs' / ALIGNED_NAME
        self.identity_manifest = self.root / 'inputs' / 'pitches.split-aligned.identity.json'
        aligned_pin = write_aligned_copy(self.processed, self.aligned, self.identity_manifest,
                                         groups if aligned_groups is None else aligned_groups(groups),
                                         manifest_hook=manifest_hook)
        by_year = {}
        for frame in groups:
            by_year.setdefault(str(pd.to_datetime(frame['game_date']).iloc[0].year), []).append(frame)
        for year in ('2023', '2024', '2025'):
            write_groups(self.raw / f'statcast_{year}.parquet', by_year[year], roa.RAW_COLUMNS)
        self.base = {'identity_registration': {'g0_bundle': {'path': str(self.root / 'g0' / 'bundle.json')},
                                               'we_paths': {'we_contract': str(self.root / 'we' / 'contract.json')},
                                               'we_contract_sha256': '0' * 64, 'classes': {'we': {}}},
                     'le2025_validation_plan': {'local_config': {'sha256': '0' * 64}},
                     'registered_inputs': {'bc': {'path': str(self.root / 'ml-policy-runs' / 'bc' / 'bc.json'),
                                                  'file_sha256': '0' * 64}}}
        chain = self.root / 'ml-policy.json'
        chain.write_text(json.dumps(self.base))
        self.local = {'artifact_root': str(self.artifacts), 'raw_root': str(self.raw)}
        config = registered(load(), root=str(self.root))
        values = {'processed_frame_identity': {'path': str(self.processed), 'sha256': reg.file_sha256(self.processed)},
                  'processed_frame_aligned': aligned_pin,
                  'train_raw_files_sha256': {n: reg.file_sha256(self.raw / n) for n in reg.RAW_ALLOWLIST},
                  'policy_registration_chain': [{'path': str(chain), 'sha256': reg.file_sha256(chain)}],
                  'new_files_sha256': roa.new_file_hashes(), 'smoke_train_subset': {'game_dates': list(smoke)},
                  'smoke_tab_replications_per_cell': reps}
        if digest_rel is not None:
            values['digest_manifest_path'] = str(self.root / digest_rel)
        if hook is not None:
            hook(config, self)
        config['registration_input_pins'].update({**values, **(pins or {})})
        self.config = config
        self.config_path = self.root / 'EXP-OPE-RES-01.json'
        self.config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        self.digest = reg.config_digest(config)
        self.manifest = roa.DigestManifest(config['registration_input_pins']['digest_manifest_path'])
        self.manifest.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.append('config_digest', {'digest': self.digest})
        if receipts:
            self.receipt('G-CODE', commit=config['registration_input_pins']['execution_source_commit'])
            self.receipt('G-REG')

    def receipt(self, gate, *, verdict='PASS', notes_closed=False, commit=None, digest=None):
        return self.manifest.append('review_reference', {'gate': gate, 'reference': f'SYNTHETIC-{gate}', 'verdict': verdict,
                                                         'notes_closed': notes_closed, 'commit': commit,
                                                         'config_digest': digest or self.digest})

    def output(self, name):
        return Path(self.config['registration_input_pins']['output_root']) / name

    def argv(self, command, name, *extra):
        return [command, '--config', str(self.config_path), '--output', str(self.output(name)), *extra]

    def kinds(self, stage=None):
        return [e['kind'] for e in self.manifest.entries() if stage is None or e['payload'].get('stage') == stage]


class WorkspaceEnv(ClockEnv):
    """The registered environment on a synthetic workspace (only data access is injected)."""
    def __init__(self, ws, **kw):
        super().__init__(**kw)
        self.ws = ws

    def git_state(self, config_path, code):
        return {'head': code['execution_source_commit'], 'ancestor': True, 'changed': [], 'dirty': False,
                'config_committed': True}

    checkouts_extra = ()

    def checkouts(self):
        return [roa.REPO, *self.checkouts_extra]

    def local_config(self, base):
        return dict(self.ws.local)

    def heavy_lock(self, local):
        return nullcontext()

    def nice(self):
        pass

    def game_values(self, base):
        return GAME_VALUES

    def s6_record(self, config):
        raise AssertionError('the S6-dr-a5 stage directory is opened by A2-S3 only')

    def policy_registration(self, config):
        raise AssertionError('the full ML-POLICY registration is loaded by A2-S2 / A2-S3 only')


@contextmanager
def watch_parquet():
    """Every parquet file opened and every row group decoded (file, index, columns); whole-file reads fail."""
    log = {'opened': [], 'reads': []}

    class Spy:
        def __init__(self, path, *args, **kwargs):
            self._name = Path(path).name
            log['opened'].append(self._name)
            self._file = REAL_PARQUET_FILE(path, *args, **kwargs)

        @property
        def metadata(self):
            return self._file.metadata

        def read_row_group(self, i, columns=None, **kwargs):
            log['reads'].append((self._name, i, tuple(columns or ())))
            return self._file.read_row_group(i, columns=columns, **kwargs)

    def whole_file(*args, **kwargs):
        raise AssertionError('a whole-file parquet read')
    with mock.patch.object(pq, 'ParquetFile', Spy), mock.patch.object(pq, 'read_table', whole_file), \
            mock.patch.object(pd, 'read_parquet', whole_file):
        yield log


def quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ================================================================ refusals
class FailClosedTests(unittest.TestCase):
    def test_every_real_data_command_refuses_the_draft(self):
        tmp = tmpdir(self)
        for command in ('smoke', 's0', 's1', 's2', 's3', 'decide', 'unblind'):
            with self.subTest(command), self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
                roa.main([command, '--output', str(tmp / command)], env=ClockEnv())
        for argv in (['register-digest', '--expect-digest', 'x'],
                     ['register-addendum', '--stage', 'A2-S0', '--failing-test', 't', '--review', 'r'],
                     ['record-review', '--gate', 'G-CODE', '--review', 'r', '--verdict', 'PASS'],
                     ['reconcile-attempt', '--stage', 'A2-S0', '--note', 'n']):
            with self.subTest(argv[0]), self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
                roa.main(argv, env=ClockEnv())
        self.assertEqual(list(tmp.iterdir()), [])  # nothing was created

    def test_data_free_commands(self):
        report = json.loads(quiet(lambda: (roa.main(['check']), sys.stdout.getvalue())[1]))
        self.assertEqual(report['status'], reg.DRAFT)
        self.assertEqual(report['config_digest'], reg.config_digest(load()))
        pins = json.loads(quiet(lambda: (roa.main(['pins']), sys.stdout.getvalue())[1]))
        self.assertEqual(set(pins['new_files_sha256']), set(reg.NEW_FILES))

    def test_x2_identity_must_name_a_supported_loader(self):
        with self.assertRaisesRegex(IntegrityError, 'X2'):
            roa.x2_world_predictor({'kind': 'some_member'})
        self.assertTrue(callable(roa.x2_world_predictor({'kind': 'g0_single_member', 'seed': 2})))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tmpdir(self)
        self.manifest = roa.DigestManifest(self.tmp / 'digest.jsonl')

    def test_append_only_chain_and_tamper_detection(self):
        a = self.manifest.append('config_digest', {'digest': 'a' * 64})
        b = self.manifest.append('cell_seal', {'stage': 'A2-S1', 'cell': 'T-A1', 'sha256': 'b' * 64})
        self.assertEqual([e['seq'] for e in self.manifest.entries()], [0, 1])
        self.assertEqual(b['prev'], a['sha256'])
        self.assertEqual(self.manifest.latest('config_digest')['payload']['digest'], 'a' * 64)
        lines = (self.tmp / 'digest.jsonl').read_text().splitlines()
        (self.tmp / 'digest.jsonl').write_text(lines[0].replace('aaaa', 'aaab', 1) + '\n' + lines[1] + '\n')
        with self.assertRaisesRegex(IntegrityError, 'tampered'):
            self.manifest.entries()
        with self.assertRaisesRegex(IntegrityError, 'unknown manifest entry'):
            roa.DigestManifest(self.tmp / 'x.jsonl').append('smoke_record', {})
        with self.assertRaisesRegex(IntegrityError, 'outside the repository results'):
            roa.DigestManifest(reg.REPO / 'results' / 'digest.jsonl')

    def test_config_digest_must_match(self):
        config = load()
        with self.assertRaisesRegex(IntegrityError, 'config digest'):
            roa.require_config_digest(self.manifest, config)
        self.manifest.append('config_digest', {'digest': reg.config_digest(config)})
        roa.require_config_digest(self.manifest, config)
        changed = copy.deepcopy(config)
        changed['task'] += '!'
        with self.assertRaisesRegex(IntegrityError, 'config digest'):
            roa.require_config_digest(self.manifest, changed)


class ManifestWriterTests(unittest.TestCase):
    """PM decision (review round 3 #3): an exclusive fcntl writer lock, the tail digest verified before every
    append, a second writer refuses; receipts are written with the guard signals blocked."""
    def setUp(self):
        self.tmp = tmpdir(self)
        self.path = self.tmp / 'digest.jsonl'

    def seal(self, manifest, n):
        return manifest.append('cell_seal', {'stage': 'A2-S1', 'cell': f'T-{n}', 'status': 'COMPLETED', 'sha256': 'b' * 64})

    def test_a_second_writer_refuses(self):
        holder, other = roa.DigestManifest(self.path), roa.DigestManifest(self.path)
        holder.append('config_digest', {'digest': 'a' * 64})
        with holder.writer():
            with self.assertRaisesRegex(IntegrityError, 'another writer'):
                self.seal(other, 1)
            with self.assertRaisesRegex(IntegrityError, 'another writer'):
                with other.writer():
                    pass
            self.seal(holder, 2)  # the holder appends under its lock
            code = ('import sys; sys.path[:0] = [sys.argv[2], sys.argv[3]]; import run_ope_a2 as r; '
                    'r.DigestManifest(sys.argv[1]).append("cell_seal", {"stage": "A2-S1"})')
            result = subprocess.run([sys.executable, '-c', code, str(self.path), str(HERE.parent / 'scripts'),
                                     str(HERE.parent)], capture_output=True, text=True, timeout=300)
            self.assertNotEqual(result.returncode, 0)  # another process refuses as well
            self.assertIn('another writer', result.stderr)
        self.seal(other, 3)  # released
        self.assertEqual([e['seq'] for e in other.entries()], [0, 1, 2])

    def test_concurrent_writers_never_corrupt_the_chain(self):
        roa.DigestManifest(self.path).append('config_digest', {'digest': 'a' * 64})
        refused = []

        def writer(tag):
            manifest = roa.DigestManifest(self.path)
            done = 0
            while done < 20:
                try:
                    self.seal(manifest, f'{tag}{done}')
                    done += 1
                except IntegrityError as error:
                    refused.append('another writer' in str(error))
                    time.sleep(0.001)
        threads = [threading.Thread(target=writer, args=(tag,)) for tag in 'ab']
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        entries = roa.DigestManifest(self.path).entries()  # the whole chain verifies
        self.assertEqual([e['seq'] for e in entries], list(range(41)))
        self.assertEqual(len({e['sha256'] for e in entries}), 41)
        self.assertTrue(all(refused))  # every failed attempt was a refusal, never a corrupt write

    def test_the_tail_digest_is_verified_before_appending(self):
        manifest = roa.DigestManifest(self.path)
        manifest.append('config_digest', {'digest': 'a' * 64})
        self.seal(manifest, 1)
        lines = self.path.read_text().splitlines()
        self.path.write_text(lines[0] + '\n' + lines[1].replace('T-1', 'T-9') + '\n')
        with self.assertRaisesRegex(IntegrityError, 'tampered'):
            self.seal(manifest, 2)
        with self.assertRaisesRegex(IntegrityError, 'tampered'):
            with manifest.writer():
                pass

    def test_receipts_are_written_with_the_guard_signals_blocked(self):
        seen, real = [], os.fsync

        def fsync(fd):
            seen.append(signal.pthread_sigmask(signal.SIG_BLOCK, []))
            return real(fd)
        with mock.patch.object(roa.os, 'fsync', fsync):
            roa.DigestManifest(self.path).append('config_digest', {'digest': 'a' * 64})
        self.assertTrue({signal.SIGALRM, signal.SIGUSR1} <= seen[0])
        self.assertFalse({signal.SIGALRM, signal.SIGUSR1} & signal.pthread_sigmask(signal.SIG_BLOCK, []))


class PublicationTests(unittest.TestCase):
    """PM decision (review round 4 #2): a completion receipt is staged, final-checked and committed by an atomic
    rename; a guard failure before the rename leaves no usable seal and records a non-citable ABORTED."""
    def setUp(self):
        self.tmp = tmpdir(self)
        self.manifest = roa.DigestManifest(self.tmp / 'digest.jsonl')
        self.identity = {'config_digest': 'c' * 64, 'git_commit': 'd' * 40, 'seeds': {'a2_base': reg.A2_BASE}}

    def staging(self):
        return sorted(p.name for p in self.tmp.iterdir() if '.staging-' in p.name)

    def assert_aborted(self, out, stage_id='A2-S1'):
        self.assertEqual([e['kind'] for e in self.manifest.entries()], ['attempt_start', 'stage_failure'])
        failure = self.manifest.entries()[-1]['payload']
        self.assertEqual((failure['aborted'], failure['during_publication'], failure['class']),
                         (True, True, 'infrastructure'))
        aborted = json.loads((out / 'ABORTED.json').read_text())
        self.assertEqual((aborted['status'], aborted['citable']), ('ABORTED', False))
        self.assertIn(aborted['error_type'], aborted['traceback'])  # the traceback tail (review round 9 #7)
        self.assertLessEqual(len(aborted['traceback']), reg.TRACEBACK_TAIL_CHARS)
        self.assertEqual(self.staging(), [])  # the staging artifacts are gone
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            roa.require_seal(self.manifest, stage_id)
        self.assertEqual(roa.attempt_gate(self.manifest, stage_id), {'starts': 1})  # an infrastructure retry remains

    def test_a_failed_commit_check_publishes_nothing(self):
        self.manifest.append('config_digest', {'digest': 'a' * 64})
        before = self.manifest.path.read_bytes()

        def refuse():
            raise roa.MemoryGuardExceeded('synthetic: free memory below the guard at the final check')
        with self.assertRaises(roa.MemoryGuardExceeded):
            self.manifest.append('stage_seal', {'stage': 'A2-S1'}, commit_check=refuse)
        self.assertEqual((self.manifest.path.read_bytes(), self.staging()), (before, []))

    def test_a_guard_failure_between_staging_and_rename_leaves_no_seal(self):
        calls = {'n': 0}

        def final_check():  # the early check passes, the check between staging and rename fails
            calls['n'] += 1
            if calls['n'] == 2:
                self.assertTrue(self.staging())  # the staged receipt exists, unpublished
                raise roa.HangGuardExceeded('synthetic: the hard cap reached between staging and rename')
        out = self.tmp / 'A2-S1-a1'
        with self.assertRaises(roa.HangGuardExceeded):
            with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=final_check) as d:
                roa.dump(d / 'tab.json', {'cells': {}})
        self.assert_aborted(out)

    def armed_env(self):
        env = ClockEnv()
        env.alarm = True
        env.monotonic = time.monotonic
        return env

    def test_an_interrupt_during_publication_is_held_and_aborts_before_the_rename(self):
        real = roa._create_write

        def staged_then_interrupted(path, data):
            real(path, data)
            if 'stage_seal' in data.decode().splitlines()[-1]:  # the hard cap fires while the seal is staged
                signal.pthread_kill(threading.main_thread().ident, signal.SIGALRM)
                time.sleep(0.05)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': 30}), mock.patch.object(roa, '_create_write', staged_then_interrupted):
            with self.assertRaises(roa.HangGuardExceeded):
                with roa.command_guard('A2-S1', self.armed_env()) as guard:
                    with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=guard.final_check) as d:
                        roa.dump(d / 'tab.json', {'cells': {}})
        self.assert_aborted(out)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_a_slow_staging_write_is_bounded_and_aborts(self):
        real = roa._create_write

        def slow(path, data):
            if 'stage_seal' in data.decode().splitlines()[-1]:
                time.sleep(0.5)
            real(path, data)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.object(reg, 'PUBLICATION_IO_TIMEOUT_SECONDS', 0.1), mock.patch.object(roa, '_create_write', slow):
            with self.assertRaisesRegex(roa.HangGuardExceeded, 'staging write exceeded'):
                with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
                    roa.dump(d / 'tab.json', {'cells': {}})
            time.sleep(0.6)  # the abandoned writer finishes; it never renames
        self.assertEqual([e['kind'] for e in self.manifest.entries()], ['attempt_start', 'stage_failure'])
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            roa.require_seal(self.manifest, 'A2-S1')
        with self.manifest.writer():
            pass  # the next writer removes any never-published staging leftover
        self.assertEqual(self.staging(), [])

    def test_an_interrupt_after_the_rename_keeps_the_seal_and_is_reported(self):
        real = roa.os.replace

        def replace_then_interrupted(src, dst):
            real(src, dst)
            if 'digest.jsonl' in str(dst) and 'stage_seal' in Path(dst).read_text().splitlines()[-1]:
                signal.pthread_kill(threading.main_thread().ident, signal.SIGALRM)
                time.sleep(0.05)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': 30}), mock.patch.object(roa.os, 'replace', replace_then_interrupted):
            with self.assertRaisesRegex(roa.HangGuardExceeded, 'hang guard'):
                with roa.command_guard('A2-S1', self.armed_env()) as guard:
                    with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=guard.final_check) as d:
                        roa.dump(d / 'tab.json', {'cells': {}})
        self.assertEqual(roa.require_seal(self.manifest, 'A2-S1')[0], out)  # committed before the interrupt: it stands
        self.assertFalse((out / 'ABORTED.json').exists())

    def guard_env(self, sampler):
        env = self.armed_env()
        env._guards = sampler
        return env

    def test_a_memory_interrupt_inside_the_final_check_aborts_before_the_rename(self):
        # review round 5 #1: the watchdog interrupt arrives while the final check samples memory
        state = {'publishing': False}

        def sampler():
            if state['publishing']:
                guard = roa.ACTIVE_GUARDS[-1]
                guard.violation = roa.MemoryGuardExceeded('synthetic: free memory below the guard during the final sample')
                signal.pthread_kill(threading.main_thread().ident, signal.SIGUSR1)  # held: a publication is running
                time.sleep(0.02)
        real = roa._create_write

        def staged(path, data):
            real(path, data)
            if 'stage_seal' in data.decode().splitlines()[-1]:
                state['publishing'] = True  # the next memory sample is the final check's
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': 30}), mock.patch.object(roa, '_create_write', staged):
            with self.assertRaisesRegex(roa.MemoryGuardExceeded, 'final sample'):
                with roa.command_guard('A2-S1', self.guard_env(sampler)) as guard:
                    with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=guard.final_check) as d:
                        roa.dump(d / 'tab.json', {'cells': {}})
        self.assert_aborted(out)

    def test_an_interrupt_after_the_final_check_is_caught_at_the_publication_boundary(self):
        state = {'armed': False}
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': 30}):
            with self.assertRaises(roa.HangGuardExceeded):
                with roa.command_guard('A2-S1', self.armed_env()) as guard:
                    def final_check():
                        guard.final_check()
                        if state['armed']:  # the hard cap fires right after the check, before the rename
                            signal.pthread_kill(threading.main_thread().ident, signal.SIGALRM)
                            time.sleep(0.02)
                        state['armed'] = True  # the first call is the early check; the second precedes the rename
                    with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=final_check) as d:
                        roa.dump(d / 'tab.json', {'cells': {}})
        self.assert_aborted(out)

    def test_a_timed_out_writer_never_overwrites_another_receipt(self):
        # review rounds 5 #2 / 6: the abandoned seal writer (blocked before its create) runs while the failure
        # receipt is staged and before that receipt's rename (the overlapping schedule)
        release, finished, real = threading.Event(), threading.Event(), roa._create_write

        def writer(path, data):
            last = data.decode().splitlines()[-1]
            if 'stage_seal' in last:
                release.wait(5)  # abandoned by the publication bound
                real(path, data)  # its own exclusive staging inode, never renamed
                finished.set()
                return
            real(path, data)
            if 'stage_failure' in last:  # the failure receipt is staged; the old writer runs before its rename
                release.set()
                finished.wait(5)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.object(reg, 'PUBLICATION_IO_TIMEOUT_SECONDS', 0.5), mock.patch.object(roa, '_create_write', writer):
            with self.assertRaisesRegex(roa.HangGuardExceeded, 'staging write exceeded'):
                with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
                    roa.dump(d / 'tab.json', {'cells': {}})
        self.assertTrue(finished.is_set())  # the abandoned writer did run inside the overlap
        self.assertEqual([e['kind'] for e in self.manifest.entries()], ['attempt_start', 'stage_failure'])
        leftovers = self.staging()
        self.assertEqual(len(leftovers), 1)
        self.assertIn('stage_seal', leftovers[0])  # only its own never-published inode
        self.assertEqual(roa.stage_status(self.manifest, ('A2-S1',)), {'A2-S1': 'incomplete'})
        self.assertEqual(roa.attempt_gate(self.manifest, 'A2-S1'), {'starts': 1})  # the retry uses a new attempt
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            roa.require_seal(self.manifest, 'A2-S1')

    def test_a_blocking_create_is_bounded_and_aborts(self):
        # review round 6 #1: the staging creation itself runs under the publication bound
        real, blocked = roa._create_write, threading.Event()

        def blocking_create(path, data):
            if 'stage_seal' in data.decode().splitlines()[-1]:
                blocked.wait(5)  # stands for a create that hangs on the filesystem
            real(path, data)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.object(reg, 'PUBLICATION_IO_TIMEOUT_SECONDS', 0.1), \
                mock.patch.object(roa, '_create_write', blocking_create):
            t0 = time.monotonic()
            with self.assertRaisesRegex(roa.HangGuardExceeded, 'staging write exceeded'):
                with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
                    roa.dump(d / 'tab.json', {'cells': {}})
            self.assertLess(time.monotonic() - t0, 2.0)  # bounded, not stalled by the hung create
            blocked.set()
            time.sleep(0.2)
        self.assertEqual([e['kind'] for e in self.manifest.entries()], ['attempt_start', 'stage_failure'])
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            roa.require_seal(self.manifest, 'A2-S1')

    def test_an_alarm_during_the_create_never_publishes(self):
        real = roa._create_write

        def create_with_alarm(path, data):
            if 'stage_seal' in data.decode().splitlines()[-1]:  # the hard cap fires while the staging file is created
                signal.pthread_kill(threading.main_thread().ident, signal.SIGALRM)
                time.sleep(0.05)
            real(path, data)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': 30}), mock.patch.object(roa, '_create_write', create_with_alarm):
            with self.assertRaises(roa.HangGuardExceeded):
                with roa.command_guard('A2-S1', self.armed_env()) as guard:
                    with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=guard.final_check) as d:
                        roa.dump(d / 'tab.json', {'cells': {}})
        self.assert_aborted(out)

    def test_a_post_commit_sync_error_leaves_a_consistent_committed_seal(self):
        # review round 5 #3: a transient OSError after the rename never strands the attempt
        real, calls = roa._fsync_dir, {'n': 0}

        def flaky(path):
            calls['n'] += 1
            if calls['n'] == 2:  # the seal's directory sync (the first is the attempt start's)
                raise OSError(5, 'synthetic transient I/O error')
            return real(path)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.object(roa, '_fsync_dir', flaky):
            with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
                roa.dump(d / 'tab.json', {'cells': {}})
        self.assertEqual([type(e).__name__ for e in self.manifest.durability_errors], ['OSError'])
        self.assertFalse((out / 'ABORTED.json').exists())
        # review round 6 note 8: the error survives the process as an append-only record in the attempt
        records = [json.loads(line) for line in (out / 'postcommit_errors.jsonl').read_text().splitlines()]
        self.assertEqual([(r['receipt_kind'], r['error_type'], r['output_dir']) for r in records],
                         [('stage_seal', 'OSError', str(out))])
        self.assertEqual(records[0]['receipt_sha256'], self.manifest.latest('stage_seal')['sha256'])
        self.assertEqual(roa.require_seal(self.manifest, 'A2-S1')[0], out)
        self.assertEqual(roa.stage_status(self.manifest, ('A2-S1',)), {'A2-S1': 'sealed'})
        with self.assertRaisesRegex(IntegrityError, 'already sealed'):
            roa.attempt_gate(self.manifest, 'A2-S1')
        calls['n'] = 0
        out2 = self.tmp / 'A2-S2-a1'
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S2': 30}), mock.patch.object(roa, '_fsync_dir', flaky):
            with self.assertRaises(OSError):  # with a guard: reported after the cleanup, the seal stands
                with roa.command_guard('A2-S2', self.armed_env()) as guard:
                    with roa.stage(out2, 'A2-S2', self.identity, manifest=self.manifest, final_check=guard.final_check) as d:
                        roa.dump(d / 's5x.json', {'runs': []})
        self.assertEqual(roa.require_seal(self.manifest, 'A2-S2')[0], out2)

    def postcommit_fault(self, stat):
        """The seal's directory sync fails after its rename, then the diagnostic's target lookup misbehaves."""
        real_sync, real_is_dir, state = roa._fsync_dir, Path.is_dir, {'sealed': False}

        def sync(path):
            if 'stage_seal' in (self.manifest.path.read_text().splitlines() or [''])[-1]:
                state['sealed'] = True
                raise OSError(5, 'synthetic EIO on the directory sync')
            return real_sync(path)

        def is_dir(path):
            if state['sealed'] and path.name.startswith('A2-S1'):
                return stat()
            return real_is_dir(path)
        return sync, is_dir

    def assert_seal_stands(self, out):
        self.assertEqual([e['kind'] for e in self.manifest.entries()], ['attempt_start', 'stage_seal'])
        self.assertFalse((out / 'ABORTED.json').exists())
        self.assertEqual(roa.require_seal(self.manifest, 'A2-S1')[0], out)
        self.assertEqual(roa.stage_status(self.manifest, ('A2-S1',)), {'A2-S1': 'sealed'})
        with self.assertRaisesRegex(IntegrityError, 'already sealed'):
            roa.attempt_gate(self.manifest, 'A2-S1')  # a rerun stays refused

    def test_a_failing_postcommit_diagnostic_never_invalidates_the_seal(self):
        # review round 7 #1: a stat EIO while the post-commit record picks its target
        def eio():
            raise OSError(5, 'synthetic EIO on the attempt directory stat')
        sync, is_dir = self.postcommit_fault(eio)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.object(roa, '_fsync_dir', sync), mock.patch.object(Path, 'is_dir', is_dir), \
                redirect_stderr(io.StringIO()) as err:
            with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
                roa.dump(d / 'tab.json', {'cells': {}})
        self.assert_seal_stands(out)
        self.assertEqual([type(e).__name__ for e in self.manifest.durability_errors], ['OSError', 'OSError'])
        self.assertIn('the committed receipt stands', err.getvalue())

    def test_a_slow_postcommit_diagnostic_is_bounded_and_never_invalidates_the_seal(self):
        def slow():
            time.sleep(0.5)
            return True
        sync, is_dir = self.postcommit_fault(slow)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.object(reg, 'PUBLICATION_IO_TIMEOUT_SECONDS', 0.1), mock.patch.object(roa, '_fsync_dir', sync), \
                mock.patch.object(Path, 'is_dir', is_dir), redirect_stderr(io.StringIO()):
            t0 = time.monotonic()
            with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
                roa.dump(d / 'tab.json', {'cells': {}})
            self.assertLess(time.monotonic() - t0, 2.0)
            time.sleep(0.6)  # the abandoned diagnostic writer finishes its own append-only record
        self.assert_seal_stands(out)
        self.assertEqual([type(e).__name__ for e in self.manifest.durability_errors], ['OSError', 'HangGuardExceeded'])

    def test_a_postcommit_failure_under_a_guard_is_reported_after_the_cleanup(self):
        def eio():
            raise OSError(5, 'synthetic EIO on the attempt directory stat')
        sync, is_dir = self.postcommit_fault(eio)
        out = self.tmp / 'A2-S1-a1'
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': 30}), mock.patch.object(roa, '_fsync_dir', sync), \
                mock.patch.object(Path, 'is_dir', is_dir), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(OSError, 'directory sync'):  # the exit code reports it; the seal stands
                with roa.command_guard('A2-S1', self.armed_env()) as guard:
                    with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=guard.final_check) as d:
                        roa.dump(d / 'tab.json', {'cells': {}})
        self.assert_seal_stands(out)

    def run_stalled(self, name, stall_patch, sync_stall=None):
        """Review round 8 #1 with a mocked clock: once the seal's rename returns only 0.6 s of the hang guard is
        left (an I/O budget of about 0.2 s after the cleanup share); the directory sync fails (or, with
        ``sync_stall``, stalls itself) and a post-commit step stalls for 5 s. The command ends at once with
        HardDeadlineReached and the seal stands."""
        self.manifest = roa.DigestManifest(self.tmp / f'digest-{name}.jsonl')
        env, hard = ClockEnv(), 30.0
        env.alarm = True
        real_replace, real_sync = roa.os.replace, roa._fsync_dir

        def replace(src, dst):
            real_replace(src, dst)
            if Path(dst) == self.manifest.path and 'stage_seal' in Path(dst).read_text().splitlines()[-1]:
                env.clock = hard - 0.6  # the mocked clock: the hard cap is near once the seal is committed

        def sync(path):
            if env.clock > 0 and sync_stall is not None:
                sync_stall.wait(5)
            elif env.clock > 0:
                raise OSError(5, 'synthetic EIO on the directory sync')
            return real_sync(path)
        out = self.tmp / f'A2-S1-{name}'
        t0 = time.monotonic()
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': hard}), mock.patch.object(roa.os, 'replace', replace), \
                mock.patch.object(roa, '_fsync_dir', sync), mock.patch.object(roa, 'LAST_GUARD', []), stall_patch:
            with self.assertRaisesRegex(roa.HardDeadlineReached, 'the committed receipt stands'):
                with roa.command_guard('A2-S1', env) as guard:
                    with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest,
                                   final_check=guard.final_check) as d:
                        roa.dump(d / 'tab.json', {'cells': {}})
            self.assertAlmostEqual(roa.LAST_GUARD[-1].remaining(), 0.6)
        self.assertLess(time.monotonic() - t0, 2.0)  # not held by the stalled step (5 s)
        self.assert_seal_stands(out)
        return out, [type(e).__name__ for e in self.manifest.durability_errors]

    def test_a_postcommit_record_stalled_at_the_hard_deadline_ends_the_command_at_once(self):
        stall = threading.Event()
        try:
            _, names = self.run_stalled('record', mock.patch.object(roa, '_write_postcommit_record',
                                                                    lambda *args: stall.wait(5)))
        finally:
            stall.set()
        self.assertEqual(names, ['OSError', 'HardDeadlineReached', 'HardDeadlineReached'])  # no step after it

    def test_a_postcommit_warning_stalled_at_the_hard_deadline_ends_the_command_at_once(self):
        stall = threading.Event()

        class SlowStderr(io.StringIO):
            def write(self, text):
                stall.wait(5)
                return super().write(text)
        try:
            out, names = self.run_stalled('warning', mock.patch.object(sys, 'stderr', SlowStderr()))
        finally:
            stall.set()
        self.assertEqual(names, ['OSError', 'HardDeadlineReached', 'HardDeadlineReached'])
        record = json.loads((out / 'postcommit_errors.jsonl').read_text())  # the record came first and was kept
        self.assertEqual((record['receipt_kind'], record['error_type']), ('stage_seal', 'OSError'))

    def test_a_directory_sync_reaching_the_hard_deadline_ends_the_command_at_once(self):
        stall = threading.Event()
        try:
            out, names = self.run_stalled('sync', nullcontext(), sync_stall=stall)
        finally:
            stall.set()
        self.assertEqual(names, ['HardDeadlineReached', 'HardDeadlineReached'])  # no record or warning step after it
        self.assertFalse((out / 'postcommit_errors.jsonl').exists())

    def test_the_cleanup_share_and_report_bound_are_pinned(self):
        # review round 9 #10: the numbers in the registry guard prose are the ones the runner uses
        numbers = reg.IMPLEMENTATION['guards']['guard_numbers']
        self.assertEqual((numbers['watchdog_join_seconds'], numbers['io_cleanup_slack_seconds'],
                          numbers['report_timeout_seconds'], numbers['publication_io_timeout_seconds'],
                          numbers['cleanup_reserve_seconds'], numbers['cleanup_reserve_share_of_H'],
                          numbers['traceback_tail_chars']), (1.0, 0.5, 1.0, 10.0, 30.0, 0.025, 4000))
        self.assertIn('min(1.5 s, cleanup reserve / 2)', reg.IMPLEMENTATION['guards']['publication'])
        self.assertIn('REPORT_TIMEOUT_SECONDS 1 s', reg.IMPLEMENTATION['guards']['publication'])
        for hard, share in ((30.0, 0.375), (4 * 3600.0, 1.5)):  # min(1.5 s, min(30 s, H / 40) / 2)
            with self.subTest(hard=hard), mock.patch.dict(reg.HANG_GUARDS, {'A2-S1': hard}):
                guard = roa.CommandGuard('A2-S1', ClockEnv())
                self.assertAlmostEqual(guard.remaining() - guard.io_budget(), share)

    def test_the_final_report_is_bounded_by_the_hard_cap(self):
        # review round 8 #1: the command-line entry's final report never outlives its bound or the hard cap
        stall = threading.Event()

        class SlowStderr(io.StringIO):
            def write(self, text):
                stall.wait(5)
                return super().write(text)

        class BrokenStderr(io.StringIO):
            def write(self, text):
                raise BrokenPipeError(32, 'synthetic broken pipe')
        try:
            with mock.patch.object(sys, 'stderr', SlowStderr()), mock.patch.object(reg, 'REPORT_TIMEOUT_SECONDS', 0.2), \
                    mock.patch.object(roa, 'LAST_GUARD', []):
                t0 = time.monotonic()
                self.assertEqual(roa.finish(1, 'run_ope_a2: synthetic failure'), 1)  # a failure stays nonzero
                self.assertEqual(roa.finish(0, 'a stalled report'), 3)  # the report could not complete in time
                roa.LAST_GUARD.append(types.SimpleNamespace(remaining=lambda: 0.0))
                self.assertEqual(roa.finish(0), 3)  # no time left before the hard cap: nonzero at once
                self.assertLess(time.monotonic() - t0, 1.0)
        finally:
            stall.set()
        with mock.patch.object(roa, 'LAST_GUARD', []):
            with mock.patch.object(sys, 'stderr', BrokenStderr()):
                self.assertEqual(roa.finish(0, 'unwritable'), 1)
            with redirect_stderr(io.StringIO()) as err:
                self.assertEqual(roa.finish(0, 'done'), 0)
        self.assertEqual(err.getvalue(), 'done\n')

    def test_the_command_line_entry_flushes_its_report_and_exits(self):
        def cli(*argv):
            code = 'import sys; sys.path[:0] = sys.argv[1:3]; import run_ope_a2 as r; r.cli(sys.argv[3:])'
            return subprocess.run([sys.executable, '-c', code, str(HERE.parent / 'scripts'), str(HERE.parent), *argv],
                                  capture_output=True, text=True, timeout=300)
        ok = cli('check')  # os._exit after the bounded flush: the output is delivered
        self.assertEqual(ok.returncode, 0, ok.stderr[-500:])
        self.assertEqual(json.loads(ok.stdout)['status'], 'DRAFT_UNREGISTERED')
        failed = cli('check', '--config', str(self.tmp / 'missing.json'))
        self.assertEqual(failed.returncode, 1)
        self.assertIn('run_ope_a2: ', failed.stderr)
        self.assertIn('Traceback (most recent call last)', failed.stderr)  # the bounded traceback tail (#7)
        usage = cli('--help')  # review round 9 #6: argparse codes pass through
        self.assertEqual(usage.returncode, 0, usage.stderr[-500:])
        self.assertIn('usage:', usage.stdout)
        bad = cli('no-such-command')
        self.assertEqual(bad.returncode, 2)
        self.assertIn('invalid choice', bad.stderr)

    def test_the_command_line_exit_mapping(self):
        # review round 9 #6/#7/#10: every outcome of main maps to one exit code; the report carries the traceback
        class Exited(BaseException):
            pass

        def fake_exit(code):
            raise Exited(code)

        def boom():
            raise ValueError('synthetic failure')
        cases = [(lambda: 0, 0, None), (lambda: None, 0, None), (boom, 1, 'ValueError: synthetic failure'),
                 (functools.partial(roa._require, False, 'synthetic integrity'), 1, 'IntegrityError'),
                 (lambda: (_ for _ in ()).throw(roa.HangGuardExceeded('synthetic')), 3, 'HangGuardExceeded'),
                 (lambda: (_ for _ in ()).throw(roa.MemoryGuardExceeded('synthetic')), 3, 'MemoryGuardExceeded'),
                 (lambda: (_ for _ in ()).throw(roa.HardDeadlineReached('synthetic')), 3, 'HardDeadlineReached'),
                 (lambda: sys.exit(0), 0, None), (lambda: sys.exit(None), 0, None), (lambda: sys.exit(2), 2, None),
                 (lambda: sys.exit(5), 5, None), (lambda: sys.exit('synthetic message'), 1, 'synthetic message')]
        for body, code, text in cases:
            with self.subTest(code=code, text=text), mock.patch.object(roa, 'main', lambda argv, env: body()), \
                    mock.patch.object(roa.os, '_exit', fake_exit), mock.patch.object(roa, 'LAST_GUARD', []), \
                    redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(Exited) as exited:
                    roa.cli([])
                self.assertEqual(exited.exception.args[0], code)
                if text is None:
                    self.assertEqual(err.getvalue(), '')
                else:
                    self.assertIn(text, err.getvalue())
                if code in (1, 3) and text != 'synthetic message':
                    self.assertIn('Traceback (most recent call last)', err.getvalue())
                    self.assertLessEqual(len(err.getvalue()), reg.TRACEBACK_TAIL_CHARS + 500)

    def test_pre_commit_reads_and_removals_are_bounded(self):
        # review round 9 #8: the manifest read of an append, the staging removal after a pre-commit failure and the
        # seal check's fallback read all run in the bounded writer
        stall = threading.Event()
        self.manifest.append('config_digest', {'digest': 'a' * 64})
        before = self.manifest.path.read_bytes()
        try:
            with mock.patch.object(reg, 'PUBLICATION_IO_TIMEOUT_SECONDS', 0.1):
                t0 = time.monotonic()
                with mock.patch.object(roa.DigestManifest, '_read_raw', lambda manifest: stall.wait(5)):
                    with self.assertRaisesRegex(roa.HangGuardExceeded, 'the manifest read'):
                        self.manifest.append('stage_seal', {'stage': 'A2-S1'})
                    self.assertFalse(roa._seal_committed(self.manifest, 'A2-S1', self.tmp / 'A2-S1-x'))

                def refuse():
                    raise roa.MemoryGuardExceeded('synthetic: refused at the final check')
                with mock.patch.object(roa, '_remove', lambda path: stall.wait(5)):
                    with self.assertRaises(roa.MemoryGuardExceeded):  # the original error, not the slow removal
                        self.manifest.append('stage_seal', {'stage': 'A2-S1'}, commit_check=refuse)
                self.assertLess(time.monotonic() - t0, 1.5)
        finally:
            stall.set()
        self.assertEqual(self.manifest.path.read_bytes(), before)  # nothing was published
        self.assertEqual(len(self.staging()), 1)  # the never-published staging file of the abandoned removal
        with self.manifest.writer():
            pass
        self.assertEqual(self.staging(), [])  # removed by the next writer


    def test_appledouble_sidecars_are_never_sealed_or_verified(self):
        # review note 11 (the exFAT T7 output volume, A3 attempt-1): macOS AppleDouble sidecars (._* with the 00051607
        # magic) appearing before or after the seal change nothing; a ._* file without the magic is an artifact
        sidecar = bytes.fromhex('00051607') + bytes(28)
        out = self.tmp / 'A2-S1-a1'
        with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
            roa.dump(d / 'tab.json', {'cells': {}})
            (d / 'cells').mkdir()
            roa.dump(d / 'cells' / 'T-A1.json', {'cell': 'T-A1'})
            for name in ('._tab.json', '._started.json', '._cells', 'cells/._T-A1.json'):  # sidecars before the seal
                (d / name).write_bytes(sidecar)
            (d / '._payload.json').write_bytes(b'{"an ordinary artifact": true}')  # the ._ prefix without the magic
        listed = json.loads((out / 'manifest.json').read_text())['artifact_sha256']
        self.assertEqual(set(listed), {'started.json', 'tab.json', 'cells/T-A1.json', '._payload.json', 'SHA256SUMS'})
        sums = (out / 'SHA256SUMS').read_text()
        self.assertNotIn('._tab.json', sums)
        self.assertIn('._payload.json', sums)
        for name in ('._SHA256SUMS', '._manifest.json', '._tab.json'):  # sidecars after the seal: new or rewritten
            (out / name).write_bytes(sidecar + b'after the seal')
        (out / '._started.json').unlink()  # or gone
        self.assertEqual(roa.require_seal(self.manifest, 'A2-S1')[0], out)
        self.assertEqual(roa.stage_status(self.manifest, ('A2-S1',)), {'A2-S1': 'sealed'})
        (out / '._payload.json').write_bytes(b'{"tampered": true}')  # the sealed ._ artifact is verified
        with self.assertRaisesRegex(IntegrityError, r'artifact \._payload\.json differs from its seal'):
            roa.require_seal(self.manifest, 'A2-S1')

    def test_a_short_or_magicless_dotunderscore_file_is_an_artifact(self):
        out = self.tmp / 'A2-S1-a1'
        with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
            (d / '._short').write_bytes(bytes.fromhex('0005'))  # shorter than the magic
            (d / '._other').write_bytes(bytes.fromhex('00051600') + bytes(28))  # another magic
            (d / 'x._y').write_bytes(bytes.fromhex('00051607'))  # the magic without the ._ prefix
        listed = set(json.loads((out / 'manifest.json').read_text())['artifact_sha256'])
        self.assertTrue({'._short', '._other', 'x._y'} <= listed)

    def test_one_rule_for_consumption_status_retry_and_decision(self):
        out = self.tmp / 'A2-S3-a1'
        with roa.stage(out, 'A2-S3', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
            roa.dump(d / 'public.json', {})
        self.manifest.append('stage_failure', {'stage': 'A2-S3', 'class': 'error', 'output_dir': str(out), 'aborted': True})
        self.assertEqual(roa.stage_status(self.manifest, ('A2-S3',)), {'A2-S3': 'incomplete'})  # aborted after its seal
        with self.assertRaisesRegex(IntegrityError, 'aborted'):
            roa.require_seal(self.manifest, 'A2-S3')
        self.assertEqual(roa.attempt_gate(self.manifest, 'A2-S3'), {'starts': 1})  # a new attempt can proceed

    def test_a_crash_before_the_rename_is_reconciled_into_aborted_unrecorded(self):
        ws = Workspace(self)
        env = WorkspaceEnv(ws)
        out = ws.output('A2-SM-a1')
        out.mkdir(parents=True)
        roa.dump(out / 'manifest.json', {'stage': 'A2-SM'})  # outputs complete, the process died before its rename
        ws.manifest.append('attempt_start', {'stage': 'A2-SM', 'output_dir': str(out)})
        with self.assertRaisesRegex(IntegrityError, 'reconciliation'):
            roa.main(ws.argv('smoke', 'A2-SM-a2'), env=env)
        quiet(roa.main, ['reconcile-attempt', '--config', str(ws.config_path), '--stage', 'A2-SM', '--note', 'crashed'],
              env=env)
        marker = json.loads((out / 'ABORTED_UNRECORDED.json').read_text())
        self.assertEqual((marker['status'], marker['citable']), ('ABORTED_UNRECORDED', False))
        roa.main(ws.argv('smoke', 'A2-SM-a2'), env=env)  # never a stuck attempt: the new attempt seals
        self.assertEqual(roa.require_seal(ws.manifest, 'A2-SM')[0], ws.output('A2-SM-a2').resolve())

    def test_an_aborted_attempt_is_never_usable(self):
        out = self.tmp / 'A2-S1-a1'
        with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
            roa.dump(d / 'tab.json', {'cells': {}})
        roa.require_seal(self.manifest, 'A2-S1')
        roa.dump(out / 'ABORTED.json', {'status': 'ABORTED'})
        with self.assertRaisesRegex(IntegrityError, 'aborted'):
            roa.require_seal(self.manifest, 'A2-S1')


class ReceiptTests(unittest.TestCase):
    """PM decision (review #3): G-CODE (commit = the running code) and G-REG receipts before any data."""
    def setUp(self):
        self.config = registered()
        self.digest = reg.config_digest(self.config)
        self.tmp = tmpdir(self)

    def manifest(self, *receipts):
        manifest = roa.DigestManifest(self.tmp / f'digest-{len(list(self.tmp.iterdir()))}.jsonl')
        for gate, payload in receipts:
            manifest.append('review_reference', {'gate': gate, 'reference': 'R', 'verdict': 'PASS', 'notes_closed': False,
                                                 'commit': None, 'config_digest': self.digest, **payload})
        return manifest

    def test_receipts(self):
        commit = 'd' * 40
        good = [('G-CODE', {'commit': commit}), ('G-REG', {})]
        self.assertEqual(set(roa.require_reviews(self.manifest(*good), self.config, commit)), {'G-CODE', 'G-REG'})
        roa.require_reviews(self.manifest(('G-CODE', {'commit': commit, 'verdict': 'PASS_WITH_NOTES', 'notes_closed': True}),
                                          ('G-REG', {})), self.config, commit)
        cases = {'no receipt': ([], 'no G-CODE review receipt'),
                 'no G-REG': ([('G-CODE', {'commit': commit})], 'no G-REG review receipt'),
                 'other commit': ([('G-CODE', {'commit': 'e' * 40}), ('G-REG', {})], 'running code commit'),
                 'open notes': ([('G-CODE', {'commit': commit, 'verdict': 'PASS_WITH_NOTES'}), ('G-REG', {})], 'open notes'),
                 'failed review': ([('G-CODE', {'commit': commit, 'verdict': 'FAIL'}), ('G-REG', {})], 'not a pass'),
                 'other digest': ([('G-CODE', {'commit': commit}), ('G-REG', {'config_digest': 'f' * 64})],
                                  'another config digest'),
                 'latest receipt counts': ([('G-CODE', {'commit': commit}), ('G-REG', {}),
                                            ('G-CODE', {'commit': commit, 'verdict': 'FAIL'})], 'not a pass')}
        for name, (receipts, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(IntegrityError, message):
                roa.require_reviews(self.manifest(*receipts), self.config, commit)
        fix = self.manifest(*good)
        fix.append('bugfix_addendum', {'stage': 'A2-S0', 'fix_commit': 'e' * 40, 'new_files_sha256': {}})
        with self.assertRaisesRegex(IntegrityError, 'running code commit'):  # a recorded fix needs its own G-CODE
            roa.require_reviews(fix, self.config, roa.code_pins(self.config, fix)['execution_source_commit'])

    def test_a_command_without_receipts_refuses_before_any_data(self):
        ws = Workspace(self, receipts=False)
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'no G-CODE review receipt'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual((log['opened'], ws.kinds('A2-SM'), ws.output('A2-SM-a1').exists()), ([], [], False))


# ================================================================ guards and the run window
class GuardTests(unittest.TestCase):
    def test_the_window_refuses_before_preflight(self):
        missing = ['--config', '/nonexistent/EXP-OPE-RES-01.json', '--output', '/nonexistent/out']
        for command in ('smoke', 's0', 's1', 's2', 's3', 'decide', 'unblind'):
            with self.subTest(command), self.assertRaisesRegex(IntegrityError, 'strictly before 06:30'):
                roa.main([command, *missing], env=ClockEnv(now=kst(28, 9, 0)))  # inside the protection window
        with self.assertRaisesRegex(IntegrityError, 'strictly before 06:30'):  # 7,200 s would end exactly at 06:30
            roa.main(['smoke', *missing], env=ClockEnv(now=kst(28, 4, 30)))
        with self.assertRaisesRegex(IntegrityError, 'strictly before 06:30'):  # unblind: the A2-S4 guard, 3,600 s
            roa.main(['unblind', *missing], env=ClockEnv(now=kst(28, 5, 30)))
        with self.assertRaises(FileNotFoundError):  # 05:29 + 3,600 s ends 06:29: the window passes, preflight reads
            roa.main(['unblind', *missing], env=ClockEnv(now=kst(28, 5, 29)))

    def test_the_memory_guard_covers_preflight(self):
        def low():
            raise roa.MemoryGuardExceeded('synthetic: free memory below the guard')
        with self.assertRaises(roa.MemoryGuardExceeded):
            roa.main(['s0', '--config', '/nonexistent.json', '--output', '/nonexistent/out'], env=ClockEnv(guards=low))
        with self.assertRaises(roa.MemoryGuardExceeded):
            roa.Guards(rss=lambda: 9 * roa.GIB, free=lambda: 10 * roa.GIB)()
        with self.assertRaises(roa.MemoryGuardExceeded):
            roa.Guards(rss=lambda: roa.GIB, free=lambda: 3 * roa.GIB)()
        roa.Guards(rss=lambda: roa.GIB, free=lambda: 5 * roa.GIB)()
        # review round 4 #3: an unavailable or unparsable measurement is a memory-guard stop, never "enough"
        with self.assertRaisesRegex(roa.MemoryGuardExceeded, 'could not be measured'):
            roa.Guards(rss=lambda: roa.GIB, free=lambda: None)()
        garbage = subprocess.CompletedProcess(['vm_stat'], 0, stdout='no page size here\n', stderr='')
        with mock.patch.object(roa.platform, 'system', lambda: 'Darwin'), \
                mock.patch.object(roa.subprocess, 'run', lambda *a, **k: garbage):
            self.assertIsNone(roa.free_memory_bytes())
            with self.assertRaisesRegex(roa.MemoryGuardExceeded, 'could not be measured'):
                roa.Guards(rss=lambda: roa.GIB)()

        def slow(*args, **kwargs):
            raise subprocess.TimeoutExpired(['vm_stat'], reg.MEMORY_SAMPLE_TIMEOUT_SECONDS)
        with mock.patch.object(roa.platform, 'system', lambda: 'Darwin'), mock.patch.object(roa.subprocess, 'run', slow):
            with self.assertRaisesRegex(roa.MemoryGuardExceeded, 'timeout'):
                roa.Guards(rss=lambda: roa.GIB)()
        self.assertGreater(roa.peak_rss_bytes(), 0)

    def test_the_checkpoint_counts_elapsed_time(self):
        env = ClockEnv()
        guard = roa.CommandGuard('A2-UB', env)
        self.assertEqual(guard.seconds, reg.HANG_GUARDS['A2-S4'])
        guard.check()
        env.clock = reg.HANG_GUARDS['A2-S4'] + 1
        with self.assertRaises(roa.HangGuardExceeded):
            guard.check()

    def test_the_sigalrm_hang_guard_covers_the_command_and_is_disarmed(self):
        env = ClockEnv()
        env.alarm = True
        env.monotonic = time.monotonic
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-SM': 0.2}):
            t0 = time.monotonic()
            with self.assertRaises(roa.HangGuardExceeded):
                with roa.command_guard('A2-SM', env):
                    time.sleep(3)
            self.assertLess(time.monotonic() - t0, 2.5)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        with mock.patch.dict(reg.HANG_GUARDS, {'A2-SM': 30}):
            with roa.command_guard('A2-SM', env) as guard:
                guard.final_check()  # the timer stays armed through the completion receipt
                self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))  # disarmed by the bounded cleanup

    def test_the_work_deadline_and_the_cleanup_reserve(self):
        # review rounds 2 #15 / 3 #2 with a mocked clock: work stops at H - reserve; the seal is refused within the
        # last cleanup reserve, so the receipt and the bounded cleanup end before H
        env = ClockEnv()
        guard = roa.CommandGuard('A2-S1', env)
        hard = reg.HANG_GUARDS['A2-S1']
        reserve, cleanup = min(reg.FINALIZATION_RESERVE_SECONDS, hard / 10), min(reg.CLEANUP_RESERVE_SECONDS, hard / 40)
        self.assertEqual((guard.seconds, guard.work_seconds, guard.seal_seconds), (hard, hard - reserve, hard - cleanup))
        self.assertGreater(reserve, cleanup)
        env.clock = guard.work_seconds + 1  # past the work deadline, inside the hard cap
        with self.assertRaises(roa.HangGuardExceeded):
            guard.check()
        guard.final_check()  # the finalization may still hash and seal
        env.clock = guard.seal_seconds + 0.5  # inside the cleanup reserve: no seal
        with self.assertRaisesRegex(roa.HangGuardExceeded, 'not enough'):
            guard.final_check()

    def test_the_memory_watchdog_interrupts_any_code_path(self):
        # review round 2 #4: memory checks through code that has no checkpoint of its own (the protected estimator)
        state = {'low': False}

        def guards():
            if state['low']:
                raise roa.MemoryGuardExceeded('synthetic: free memory below the guard')
        env = ClockEnv(guards=guards)
        env.alarm = True
        env.monotonic = time.monotonic
        previous = signal.getsignal(signal.SIGUSR1)
        with mock.patch.object(reg, 'WATCHDOG_INTERVAL_SECONDS', 0.05), mock.patch.dict(reg.HANG_GUARDS, {'A2-SM': 30}):
            t0 = time.monotonic()
            with self.assertRaises(roa.MemoryGuardExceeded):
                with roa.command_guard('A2-SM', env):
                    state['low'] = True
                    time.sleep(3)  # stands for a long protected call
            self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(signal.getsignal(signal.SIGUSR1), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertFalse(any(t.name == 'a2-memory-watchdog' for t in threading.enumerate()))

    def test_a_stuck_memory_sampler_cannot_delay_the_cleanup_or_signal_late(self):
        # review round 3 #2: the watchdog join is bounded and a late violation never reaches restored handlers
        calls, release = {'n': 0}, threading.Event()

        def guards():
            calls['n'] += 1
            if calls['n'] > 1:  # the watchdog's first sample hangs, then reports a violation
                release.wait(5)
                raise roa.MemoryGuardExceeded('synthetic late violation')
        env = ClockEnv(guards=guards)
        env.alarm = True
        env.monotonic = time.monotonic
        hits = []
        original = signal.signal(signal.SIGUSR1, lambda signum, frame: hits.append(signum))  # a late signal lands here
        try:
            with mock.patch.object(reg, 'WATCHDOG_INTERVAL_SECONDS', 0.01), \
                    mock.patch.object(reg, 'WATCHDOG_JOIN_SECONDS', 0.1), mock.patch.dict(reg.HANG_GUARDS, {'A2-SM': 30}):
                with roa.command_guard('A2-SM', env):
                    time.sleep(0.2)  # the watchdog is stuck inside its sample now
                    leaving = time.monotonic()
                cleanup = time.monotonic() - leaving
            self.assertLess(cleanup, 1.0)  # bounded by the 0.1 s join, not by the 5 s sampler
            release.set()
            time.sleep(0.3)  # the sampler wakes, sees the cleanup started and does not signal
            self.assertEqual(hits, [])
            self.assertEqual(signal.getsignal(signal.SIGALRM), signal.SIG_DFL)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        finally:
            release.set()
            signal.signal(signal.SIGUSR1, original)

    def test_a_guard_interrupt_during_the_cleanup_is_reported_after_it(self):
        env = ClockEnv()
        env.alarm = True
        env.monotonic = time.monotonic
        try:
            with mock.patch.dict(reg.HANG_GUARDS, {'A2-SM': 30}):
                with self.assertRaisesRegex(roa.HangGuardExceeded, 'during the cleanup'):
                    with roa.command_guard('A2-SM', env):
                        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
                        signal.pthread_kill(threading.main_thread().ident, signal.SIGALRM)  # pending at the cleanup
            self.assertFalse(signal.sigpending() & {signal.SIGALRM})  # consumed while blocked, never delivered later
            self.assertEqual(signal.getsignal(signal.SIGALRM), signal.SIG_DFL)
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})

    def test_manifest_commands_are_guarded(self):
        ws = Workspace(self)
        before = ws.manifest.path.read_bytes()
        argv = ['record-review', '--config', str(ws.config_path), '--gate', 'G-REV', '--review', 'R', '--verdict', 'PASS']

        def low():
            raise roa.MemoryGuardExceeded('synthetic')
        with self.assertRaisesRegex(IntegrityError, 'strictly before 06:30'):
            roa.main(argv, env=WorkspaceEnv(ws, now=kst(28, 9, 0)))
        with self.assertRaises(roa.MemoryGuardExceeded):
            roa.main(argv, env=WorkspaceEnv(ws, guards=low))
        self.assertEqual(ws.manifest.path.read_bytes(), before)  # nothing appended
        quiet(roa.main, argv, env=WorkspaceEnv(ws))
        self.assertEqual(ws.manifest.latest('review_reference', gate='G-REV')['payload']['verdict'], 'PASS')

    def test_the_final_check_follows_the_last_write_and_precedes_the_seal(self):
        tmp = tmpdir(self)
        manifest = roa.DigestManifest(tmp / 'digest.jsonl')
        seen = {}

        def final_check():
            seen['manifest_written'] = (tmp / 'A2-S2-a1' / 'manifest.json').exists()
            seen['kinds'] = [e['kind'] for e in manifest.entries()]
            raise roa.HangGuardExceeded('synthetic: the guard elapsed during sealing')
        with self.assertRaises(roa.HangGuardExceeded):
            with roa.stage(tmp / 'A2-S2-a1', 'A2-S2', {'config_digest': 'c' * 64}, manifest=manifest,
                           final_check=final_check) as out:
                roa.dump(out / 's5x.json', {'runs': []})
        self.assertEqual(seen, {'manifest_written': True, 'kinds': ['attempt_start']})
        self.assertEqual([(e['kind'], e['payload'].get('class')) for e in manifest.entries()],
                         [('attempt_start', None), ('stage_failure', 'infrastructure')])
        self.assertTrue((tmp / 'A2-S2-a1' / 's5x.json').exists())  # partial outputs preserved, not sealed
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            roa.require_seal(manifest, 'A2-S2')


# ================================================================ stages, attempts and destinations
class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tmpdir(self)
        self.manifest = roa.DigestManifest(self.tmp / 'digest.jsonl')
        self.identity = {'config_digest': 'c' * 64, 'git_commit': 'd' * 40, 'seeds': {'a2_base': reg.A2_BASE}}

    def test_success_writes_sums_manifest_and_seal(self):
        out = self.tmp / 'A2-S1-a1'
        with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, final_check=lambda: None) as d:
            roa.dump(d / 'tab.json', {'cells': {}})
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(set(manifest), {'stage', 'cost', 'artifact_sha256', 'config_digest', 'git_commit', 'seeds'})
        self.assertEqual(set(manifest['cost']), {'wall_seconds', 'cpu_seconds', 'peak_rss_bytes'})
        self.assertIn('SHA256SUMS', manifest['artifact_sha256'])
        self.assertTrue(any(line.endswith('  tab.json') for line in (out / 'SHA256SUMS').read_text().splitlines()))
        self.assertEqual([e['kind'] for e in self.manifest.entries()], ['attempt_start', 'stage_seal'])
        sealed, _ = roa.require_seal(self.manifest, 'A2-S1')
        self.assertEqual(sealed, out)
        (out / 'tab.json').write_text('{"cells": {"x": 1}}\n')
        with self.assertRaisesRegex(IntegrityError, 'differs from its seal'):
            roa.require_seal(self.manifest, 'A2-S1')

    def test_failures_are_classified_and_recorded(self):
        with self.assertRaises(IntegrityError):
            with roa.stage(self.tmp / 'f1', 'A2-S2', self.identity, manifest=self.manifest, final_check=lambda: None):
                raise IntegrityError('synthetic deterministic defect')
        failure = json.loads(next((self.tmp / 'f1').glob('failure-*.json')).read_text())
        self.assertEqual((failure['class'], failure['citable']), ('deterministic_integrity', False))
        self.assertEqual([(e['kind'], e['payload'].get('class')) for e in self.manifest.entries()],
                         [('attempt_start', None), ('stage_failure', 'deterministic_integrity')])
        with self.assertRaisesRegex(IntegrityError, 'stage output exists'):
            with roa.stage(self.tmp / 'f1', 'A2-S2', self.identity, manifest=self.manifest, final_check=lambda: None):
                pass
        self.assertEqual(len(self.manifest.entries()), 2)  # an existing directory refuses before the start receipt

    def test_the_attempt_start_receipt_precedes_the_directory(self):
        (self.tmp / 'blocked').write_text('a file where the stage directory would go')
        with self.assertRaises(OSError):
            with roa.stage(self.tmp / 'blocked' / 'A2-S0-a1', 'A2-S0', self.identity, manifest=self.manifest,
                           final_check=lambda: None):
                pass
        self.assertEqual([(e['kind'], e['payload'].get('class')) for e in self.manifest.entries()],
                         [('attempt_start', None), ('stage_failure', 'error')])  # the start counts

    def test_attempt_accounting(self):
        # PM decision (review #5): a start without a receipt counts (crash) and needs a reconciliation; a third
        # start refuses; a retry after an integrity failure needs a bug-fix addendum
        roa.attempt_gate(self.manifest, 'A2-S0')
        self.manifest.append('attempt_start', {'stage': 'A2-S0', 'output_dir': str(self.tmp / 'a1')})  # then killed
        self.assertIsNotNone(roa.unfinished_start(self.manifest, 'A2-S0'))
        with self.assertRaisesRegex(IntegrityError, 'reconciliation'):
            roa.attempt_gate(self.manifest, 'A2-S0')
        self.manifest.append('attempt_reconciliation', {'stage': 'A2-S0', 'note': 'process killed'})
        self.assertIsNone(roa.unfinished_start(self.manifest, 'A2-S0'))
        self.assertEqual(roa.attempt_gate(self.manifest, 'A2-S0'), {'starts': 1})
        self.manifest.append('attempt_start', {'stage': 'A2-S0', 'output_dir': str(self.tmp / 'a2')})
        self.manifest.append('stage_failure', {'stage': 'A2-S0', 'class': 'infrastructure'})
        with self.assertRaisesRegex(IntegrityError, 'third start refuses'):
            roa.attempt_gate(self.manifest, 'A2-S0')
        self.assertEqual(roa.stage_status(self.manifest)['A2-S0'], 'DECISION_NEEDED')
        self.manifest.append('attempt_start', {'stage': 'A2-S1', 'output_dir': str(self.tmp / 'b1')})
        self.manifest.append('stage_failure', {'stage': 'A2-S1', 'class': 'deterministic_integrity'})
        with self.assertRaisesRegex(IntegrityError, 'bug-fix addendum'):
            roa.attempt_gate(self.manifest, 'A2-S1')
        self.manifest.append('bugfix_addendum', {'stage': 'A2-S1', 'fix_commit': 'e' * 40, 'new_files_sha256': {}})
        roa.attempt_gate(self.manifest, 'A2-S1')
        self.manifest.append('attempt_start', {'stage': 'A2-S1', 'output_dir': str(self.tmp / 'b2')})
        self.manifest.append('stage_failure', {'stage': 'A2-S1', 'class': 'deterministic_integrity'})
        self.manifest.append('stage_seal', {'stage': 'A2-S2', 'output_dir': str(self.tmp), 'manifest_sha256': 'f' * 64})
        self.assertEqual(roa.stage_status(self.manifest), {'A2-S0': 'DECISION_NEEDED', 'A2-S1': 'FAILED_INTEGRITY',
                                                           'A2-S2': 'sealed', 'A2-S3': 'incomplete'})
        with self.assertRaisesRegex(IntegrityError, 'already sealed'):
            roa.attempt_gate(self.manifest, 'A2-S2')
        self.assertEqual(roa.code_pins(registered(), self.manifest)['execution_source_commit'], 'e' * 40)

    def test_smoke_gate_reads_the_sealed_smoke_record(self):
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            roa.smoke_gate(self.manifest, 'A2-S0')
        with roa.stage(self.tmp / 'A2-SM-a1', 'A2-SM', self.identity, manifest=self.manifest, final_check=lambda: None) as out:
            roa.dump(out / 'smoke_record.json', {'projection': {'within_guard': {'A2-S0': True, 'A2-S1': False}}})
        roa.smoke_gate(self.manifest, 'A2-S0')
        with self.assertRaisesRegex(IntegrityError, 'decision needed'):
            roa.smoke_gate(self.manifest, 'A2-S1')

    def test_output_location_is_disjoint_from_every_sealed_input(self):
        # PM decision (review #8): a fresh child of the pinned root; the root is neither inside nor around a protected
        # root (resolved real paths; the main checkout's results/ included)
        config = registered()
        root = self.tmp / 'out'
        config['registration_input_pins']['output_root'] = str(root)
        main = self.tmp / 'main-checkout'
        protected = {'the S6-dr-a5 stage directory': self.tmp / 's6', 'the checkout main': main}
        self.assertEqual(roa.check_output(root / 'A2-S0-a1', config, protected), (root / 'A2-S0-a1').resolve())
        with self.assertRaisesRegex(IntegrityError, 'directly under'):
            roa.check_output(root / 'a' / 'b', config, protected)
        root.mkdir()
        (root / 'exists').mkdir()
        with self.assertRaisesRegex(IntegrityError, 'directly under'):
            roa.check_output(root / 'exists', config, protected)
        for name, bad_root in (('inside the S6 stage directory', self.tmp / 's6' / 'out'),
                               ('the S6 stage directory', self.tmp / 's6'),
                               ('around the S6 stage directory', self.tmp),
                               ('inside the main checkout results', main / 'results' / 'a2')):
            config['registration_input_pins']['output_root'] = str(bad_root)
            with self.subTest(name), self.assertRaisesRegex(IntegrityError, 'disjoint'):
                roa.check_output(bad_root / 'A2-S0-a1', config, protected)
        (self.tmp / 'link').symlink_to(self.tmp / 's6', target_is_directory=True)  # a symlinked alias resolves
        config['registration_input_pins']['output_root'] = str(self.tmp / 'link')
        with self.assertRaisesRegex(IntegrityError, 'disjoint'):
            roa.check_output(self.tmp / 'link' / 'A2-S0-a1', config, protected)

    def test_enforce_sources_with_a_mocked_git_state(self):
        config = registered()
        config['registration_input_pins']['new_files_sha256'] = roa.new_file_hashes()
        ok = {'head': 'a' * 40, 'ancestor': True, 'changed': [], 'dirty': False, 'config_committed': True}
        roa.enforce_sources(config, reg.REPO / reg.CONFIG_PATH, state=ok)
        for key, value in (('changed', ['experiments/x.py']), ('ancestor', False), ('dirty', True),
                           ('config_committed', False)):
            with self.subTest(key), self.assertRaises(IntegrityError):
                roa.enforce_sources(config, reg.REPO / reg.CONFIG_PATH, state={**ok, key: value})
        config['registration_input_pins']['new_files_sha256'][reg.NEW_FILES[0]] = '0' * 64
        with self.assertRaisesRegex(IntegrityError, 'registration pins'):
            roa.enforce_sources(config, reg.REPO / reg.CONFIG_PATH, state=ok)


# ================================================================ the TRAIN-only reader and the processed pin
class ReaderTests(unittest.TestCase):
    def test_the_smoke_reads_train_rows_and_columns_only(self):
        # PM decision (review #2): smoke = TRAIN rows/columns only, no S6 access, other raw years never opened
        ws = Workspace(self)
        with watch_parquet() as log:
            self.assertEqual(roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws)), 0)
        self.assertEqual(sorted(set(log['opened'])), [ALIGNED_NAME, 'statcast_2024.parquet'])  # never the original
        self.assertEqual(sorted({(f, rg) for f, rg, _ in log['reads']}),
                         [(ALIGNED_NAME, 2), ('statcast_2024.parquet', 1)])  # the subset's TRAIN row groups only
        for name, _, columns in log['reads']:
            self.assertEqual(columns, roa.TRAIN_COLUMNS if name == ALIGNED_NAME else roa.RAW_COLUMNS)
        self.assertEqual(ws.kinds(), ['config_digest', 'review_reference', 'review_reference', 'attempt_start',
                                      'stage_seal'])
        out, _ = roa.require_seal(ws.manifest, 'A2-SM')
        record = json.loads((out / 'smoke_record.json').read_text())
        self.assertEqual(record['decoded'], [{'file': f, 'row_group': rg, 'columns': list(c)} for f, rg, c in log['reads']])
        self.assertEqual(record['plan']['processed']['row_groups'], [2])
        self.assertEqual(record['plan']['processed_path'], str(ws.aligned))
        self.assertEqual(record['plan']['train_rows_upper_bound'],
                         REAL_PARQUET_FILE(ws.aligned).metadata.row_group(1).num_rows
                         + REAL_PARQUET_FILE(ws.aligned).metadata.row_group(2).num_rows)
        self.assertEqual(record['projection']['label'], 'projection (not a measurement)')
        text = json.dumps({k: v for k, v in record.items() if k != 'never_computed'})
        for word in ('b_hat', 'rmse', 'H_point', 'delta_true', 'kappa_calibration'):
            self.assertNotIn(word, text)

    def test_a_straddling_row_group_refuses_before_the_attempt(self):
        # a processed file whose row groups straddle TRAIN is refused (footer only), whatever its manifest says
        ws = Workspace(self, smoke=('2023-05-15', '2023-06-05'), subset_start='2023-05-10', subset_days=27)
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'straddles a TRAIN boundary'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual(log['reads'], [])  # nothing decoded
        self.assertEqual((ws.kinds('A2-SM'), ws.output('A2-SM-a1').exists()), ([], False))  # no attempt consumed

    def test_an_original_straddling_train_is_read_through_its_aligned_copy(self):
        # review note 7 (the real layout): the pinned original has a row group straddling the TRAIN start; its aligned
        # copy (the same rows, split at TRAIN) serves the smoke and A2-S0, and the original is never opened as parquet
        straddle = lambda groups: [pd.concat(groups[:2], ignore_index=True), *groups[2:]]  # 2023-04..2024-04 in one group
        ws = Workspace(self, original_groups=straddle)
        meta = REAL_PARQUET_FILE(ws.processed).metadata
        index = [meta.schema.column(i).name for i in range(meta.num_columns)].index('game_date')
        stats = meta.row_group(0).column(index).statistics
        self.assertTrue(roa._stat_day(stats.min) < pd.Timestamp(reg.TRAIN_WINDOW[0]) <= roa._stat_day(stats.max))
        with self.assertRaisesRegex(IntegrityError, 'straddles a TRAIN boundary'):
            roa.require_split_aligned(ws.processed)  # the original itself would be refused
        env = WorkspaceEnv(ws)
        with watch_parquet() as log:
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=env)
            roa.main(ws.argv('s0', 'A2-S0-a1'), env=env)
        self.assertNotIn('pitches.parquet', log['opened'])
        self.assertEqual(sorted({rg for f, rg, _ in log['reads'] if f == ALIGNED_NAME}), [1, 2])  # TRAIN groups only
        self.assertEqual([s for s in ('A2-SM', 'A2-S0') if roa.require_seal(ws.manifest, s)], ['A2-SM', 'A2-S0'])

    def test_one_merged_train_row_group_wider_than_the_smoke_window(self):
        # review note 7 (the real copy): TRAIN is one row group, wider than the registered week; the smoke decodes the
        # whole group once and keeps the week's rows only (its load and peak RSS cover the full TRAIN decode)
        merged = lambda groups: [groups[0], pd.concat(groups[1:3], ignore_index=True), groups[3]]
        ws = Workspace(self, aligned_groups=merged)
        with watch_parquet() as log:
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual([(f, rg) for f, rg, _ in log['reads'] if f == ALIGNED_NAME], [(ALIGNED_NAME, 1)])
        out, _ = roa.require_seal(ws.manifest, 'A2-SM')
        record = json.loads((out / 'smoke_record.json').read_text())
        week = ws.groups[2]  # the registered smoke subset (2024-06-01..2024-06-20); group 1 (2024-04) is decoded too
        self.assertEqual((record['rows_in_window_all_types'], record['rows']), (len(week), len(week)))
        self.assertEqual(record['plan']['train_rows_upper_bound'], len(ws.groups[1]) + len(week))
        self.assertGreater(record['load_seconds'], 0)

    def test_a2_s0_plans_its_reads_before_its_attempt(self):
        # review note 10: a failed copy check refuses A2-S0 before its attempt-start receipt (no attempt consumed)
        ws = Workspace(self)
        env = WorkspaceEnv(ws)
        roa.main(ws.argv('smoke', 'A2-SM-a1'), env=env)
        with ws.aligned.open('ab') as stream:
            stream.write(b'\0')
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'aligned processed frame bytes differ'):
            roa.main(ws.argv('s0', 'A2-S0-a1'), env=env)
        self.assertEqual((log['reads'], ws.kinds('A2-S0'), ws.output('A2-S0-a1').exists()), ([], [], False))

    def test_a_copy_without_row_groups_is_refused(self):
        ws = Workspace(self)
        empty = ws.root / 'inputs' / 'empty.parquet'
        schema = REAL_PARQUET_FILE(ws.aligned).schema_arrow
        with pq.ParquetWriter(empty, schema):
            pass
        self.assertEqual(REAL_PARQUET_FILE(empty).metadata.num_row_groups, 0)
        with self.assertRaisesRegex(IntegrityError, 'has no row group'):
            roa.require_split_aligned(empty)

    def test_a_misaligned_copy_refuses_every_train_read(self):
        # PM decision: the smoke and the A2-S0 TRAIN reads refuse a copy whose row groups straddle TRAIN, even with
        # a PASS manifest; the raw files and the copy's rows are never decoded and no attempt is consumed
        merge = lambda groups: [pd.concat(groups[:2], ignore_index=True), *groups[2:]]  # history + TRAIN in one group
        ws = Workspace(self, aligned_groups=merge)
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'straddles a TRAIN boundary'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual((log['reads'], ws.kinds('A2-SM')), ([], []))
        pins = ws.config['registration_input_pins']
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'straddles a TRAIN boundary'):
            roa.plan_train_reads(pins, ws.local, reg.TRAIN_WINDOW, strict=False)  # the A2-S0 read
        self.assertEqual(log['reads'], [])

    def test_a_tampered_identity_manifest_or_copy_refuses(self):
        # the certificate is checked as A3 checks it (review: tool, rows/schema equal, rows, differing columns, every
        # column digest equal and covering the TRAIN columns), plus result/alignment/identity, source and copy
        ws = Workspace(self)
        good = ws.config['registration_input_pins']
        record = json.loads(ws.identity_manifest.read_text())
        ident = lambda **kw: (lambda r: {**r, 'identity': {**r['identity'], **kw}})
        columns = lambda f: (lambda r: {**r, 'identity': {**r['identity'], 'columns': f(dict(r['identity']['columns']))}})
        rows = lambda source, rewritten: (lambda r: {**r, 'source': {**r['source'], 'rows': source},
                                                     'rewritten': {**r['rewritten'], 'rows': rewritten}})
        n = record['rewritten']['rows']

        def pins_for(name, change):  # a re-pinned manifest (its bytes match its pin) that certifies something else
            path = ws.root / 'inputs' / f'alt-{name.replace(" ", "-")}.identity.json'
            path.write_text(json.dumps(change(record), sort_keys=True))
            return {**good, 'processed_frame_aligned': {**good['processed_frame_aligned'], 'identity_manifest_path': str(path),
                                                        'identity_manifest_sha256': reg.file_sha256(path)}}
        cases = [('another tool', lambda r: {**r, 'tool': 'experiments/pitchmdp/scripts/other.py'}, 'preparation tool'),
                 ('no tool', lambda r: {k: v for k, v in r.items() if k != 'tool'}, 'preparation tool'),
                 ('result FAIL', lambda r: {**r, 'result': 'FAIL'}, 'does not certify a PASS'),
                 ('not aligned', lambda r: {**r, 'aligned': False}, 'does not certify a PASS'),
                 ('identity FAIL', ident(result='FAIL'), 'does not certify a PASS'),
                 ('rows not equal', ident(rows_equal=False), 'equal rows and schema'),
                 ('schema not equal', ident(schema_equal=False), 'equal rows and schema'),
                 ('rows_equal missing', lambda r: {**r, 'identity': {k: v for k, v in r['identity'].items()
                                                                     if k != 'rows_equal'}}, 'equal rows and schema'),
                 ('differing columns', ident(differing_columns=['pitch_type']), 'lists differing columns'),
                 ('differing columns missing', lambda r: {**r, 'identity': {k: v for k, v in r['identity'].items()
                                                                            if k != 'differing_columns'}},
                  'lists differing columns'),
                 ('a column digest differs', columns(lambda c: {**c, 'pitch_type': {**c['pitch_type'], 'equal': False}}),
                  'column digest'),
                 ('no column digests', columns(lambda c: {}), 'column digest'),
                 ('a TRAIN column missing', columns(lambda c: {k: v for k, v in c.items() if k != 'pitch_type'}),
                  'every TRAIN column'),
                 ('source and rewritten rows differ', rows(n + 1, n), 'different source and rewritten row counts'),
                 ('rows missing', rows(None, None), 'different source and rewritten row counts'),
                 ('rows not the footer', rows(n + 1, n + 1), 'footer row count'),
                 ('another source', lambda r: {**r, 'source': {**r['source'], 'sha256': 'e' * 64}}, 'source is not'),
                 ('another copy', lambda r: {**r, 'rewritten': {**r['rewritten'], 'sha256': 'e' * 64}}, 'another file')]
        for name, change, message in cases:
            with self.subTest(name), watch_parquet() as log, self.assertRaisesRegex(IntegrityError, message):
                roa.plan_train_reads(pins_for(name, change), ws.local, ('2024-06-01', '2024-06-20'), strict=True)
            self.assertEqual(log['reads'], [])
        other_source = {**good, 'processed_frame_aligned': {**good['processed_frame_aligned'], 'source_sha256': 'e' * 64}}
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'another source'):
            roa.plan_train_reads(other_source, ws.local, ('2024-06-01', '2024-06-20'), strict=True)
        self.assertEqual(log['reads'], [])
        with ws.identity_manifest.open('a') as stream:  # the pinned manifest's bytes changed
            stream.write(' ')
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'identity manifest differs from its pin'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual((log['opened'], ws.kinds('A2-SM')), ([], []))
        ws.identity_manifest.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
        with ws.aligned.open('ab') as stream:  # the copy's bytes changed
            stream.write(b'\0')
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'aligned processed frame bytes differ'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual((log['opened'], ws.kinds('A2-SM')), ([], []))

    def test_the_processed_pin_is_verified_before_any_row(self):
        # PM decision (review #6): path, inherited quality manifest and bytes before anything is decoded
        ws = Workspace(self)
        quality = json.loads(ws.quality.read_text())
        ws.quality.write_text(json.dumps({**quality, 'processed_sha256': 'f' * 64}))
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'inherited quality manifest'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual((log['opened'], ws.kinds('A2-SM')), ([], []))
        ws.quality.write_text(json.dumps(quality))
        with ws.processed.open('ab') as stream:
            stream.write(b'\0')
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'bytes differ from the A2 pin'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual(log['opened'], [])
        other = Workspace(self, hook=lambda config, w: None)
        other.config['registration_input_pins']['processed_frame_identity'] = {
            'path': '/nonexistent-a2/processed/pitches.parquet', 'sha256': 'f' * 64}
        other.config['registration_input_pins']['processed_frame_aligned']['source_sha256'] = 'f' * 64
        other.config_path.write_text(json.dumps(other.config, indent=2, ensure_ascii=False) + '\n')
        other.digest = reg.config_digest(other.config)
        other.manifest.append('config_digest', {'digest': other.digest})
        other.receipt('G-CODE', commit=other.config['registration_input_pins']['execution_source_commit'])
        other.receipt('G-REG')
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'path differs'):
            roa.main(other.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(other))
        self.assertEqual(log['opened'], [])

    def test_the_destination_must_avoid_every_pinned_input(self):
        tmp = tmpdir(self)
        ws = Workspace(self, pins={'output_root': str(tmp)})
        ws.config['registration_input_pins']['output_root'] = str(ws.raw / 'out')  # inside the raw root
        ws.config_path.write_text(json.dumps(ws.config, indent=2, ensure_ascii=False) + '\n')
        ws.digest = reg.config_digest(ws.config)
        ws.manifest = roa.DigestManifest(tmp / 'other-digest.jsonl')
        ws.config['registration_input_pins']['digest_manifest_path'] = str(tmp / 'other-digest.jsonl')
        ws.config_path.write_text(json.dumps(ws.config, indent=2, ensure_ascii=False) + '\n')
        ws.digest = reg.config_digest(ws.config)
        ws.manifest.append('config_digest', {'digest': ws.digest})
        ws.receipt('G-CODE', commit=ws.config['registration_input_pins']['execution_source_commit'])
        ws.receipt('G-REG')
        with watch_parquet() as log, self.assertRaisesRegex(IntegrityError, 'disjoint from the raw root'):
            roa.main(ws.argv('smoke', 'A2-SM-a1'), env=WorkspaceEnv(ws))
        self.assertEqual((log['opened'], ws.kinds('A2-SM')), ([], []))

    def test_inputs_are_verified_under_the_common_heavy_lock(self):
        ws = Workspace(self)
        calls = []

        class Env(WorkspaceEnv):
            def heavy_lock(self, local):
                @contextmanager
                def lock():
                    calls.append('lock')
                    yield
                    calls.append('unlock')
                return lock()

            def train_plan(self, *args, **kwargs):
                calls.append('plan')
                return super().train_plan(*args, **kwargs)

            def train(self, *args, **kwargs):
                calls.append('train')
                return super().train(*args, **kwargs)
        roa.main(ws.argv('smoke', 'A2-SM-a1'), env=Env(ws))
        self.assertEqual(calls, ['lock', 'plan', 'train', 'unlock'])

    def test_the_digest_manifest_must_avoid_every_pinned_input(self):
        # PM decision (review round 2 #8): the manifest file is outside every sealed input and both checkouts
        for rel, name in (('raw/digest.jsonl', 'the raw root'), ('artifacts/reports/digest.jsonl', 'the artifact reports directory'),
                          ('main-checkout/results/digest.jsonl', 'the checkout')):
            ws = Workspace(self, digest_rel=rel)
            env = WorkspaceEnv(ws)
            env.checkouts_extra = (ws.root / 'main-checkout',)
            before = ws.manifest.path.read_bytes()
            with self.subTest(rel), watch_parquet() as log, self.assertRaisesRegex(IntegrityError, f'outside {name}'):
                roa.main(ws.argv('smoke', 'A2-SM-a1'), env=env)
            with self.subTest(rel), self.assertRaisesRegex(IntegrityError, f'outside {name}'):
                roa.main(['record-review', '--config', str(ws.config_path), '--gate', 'G-REV', '--review', 'R',
                          '--verdict', 'PASS'], env=env)
            self.assertEqual((log['opened'], ws.manifest.path.read_bytes()), ([], before))  # nothing read or appended

    def test_reconcile_attempt_command(self):
        ws = Workspace(self)
        env = WorkspaceEnv(ws)
        argv = ['reconcile-attempt', '--config', str(ws.config_path), '--stage', 'A2-S0', '--note', 'process killed']
        with self.assertRaisesRegex(IntegrityError, 'nothing to reconcile'):
            quiet(roa.main, argv, env=env)
        ws.manifest.append('attempt_start', {'stage': 'A2-S0', 'output_dir': str(ws.output('A2-S0-a1'))})
        quiet(roa.main, argv, env=env)
        entry = ws.manifest.latest('attempt_reconciliation')
        self.assertEqual((entry['payload']['stage'], entry['payload']['counts_as_attempt']), ('A2-S0', True))
        with self.assertRaisesRegex(IntegrityError, 'nothing to reconcile'):
            quiet(roa.main, argv, env=env)
        review = ['record-review', '--config', str(ws.config_path), '--gate', 'G-CODE', '--review', 'R', '--verdict', 'PASS']
        with self.assertRaisesRegex(IntegrityError, '--commit'):
            quiet(roa.main, review, env=env)
        quiet(roa.main, [*review, '--commit', 'a' * 40], env=env)
        receipt = ws.manifest.latest('review_reference', gate='G-CODE')['payload']
        self.assertEqual((receipt['commit'], receipt['config_digest'], receipt['notes_closed']), ('a' * 40, ws.digest, False))


# ================================================================ pure stage logic
class StageLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic_train_frame()

    def test_s0_full_and_smoke(self):
        train = self.train.copy()
        rng = np.random.default_rng(0)
        train['estimated_woba_using_speedangle'] = np.where(train['description'].eq('hit_into_play').to_numpy(),
                                                            rng.uniform(0, 1.5, len(train)), np.nan)
        s0 = roa.build_s0(train, GAME_VALUES, min_pitches=100)
        self.assertTrue(roa.require_s0_descriptive(s0))
        self.assertEqual(s0['r1_table_sha256'], roa.osg.table_sha256(s0['r1_table'], s0['m']))
        self.assertIn(s0['m'], reg.M_SMOOTH_GRID)
        self.assertEqual(set(s0['kappa_p']), {'S5X', 'TAB'})
        self.assertTrue(all(v > 0 for v in s0['kappa_p'].values()))
        self.assertEqual(set(s0['train_residuals']['hit']), set(reg.S5X_GROUPS))
        # PM decision (review #14): the frozen-C0 zeta residuals are a disclosed exception to cross-fitting
        self.assertIn('intentional exception to cross-fitting', s0['provenance']['zeta_residuals'])
        self.assertIn('frozen, pre-registered', s0['provenance']['zeta_residuals'])
        self.assertIn('cross-fitted', s0['provenance']['kappa_residuals'])
        self.assertEqual(s0['xwoba_train_diagnostic']['status'], 'COMPUTED')
        self.assertEqual(s0['xwoba_train_diagnostic']['n'], s0['census']['bip_used'])
        self.assertEqual(roa.build_s0(self.train, GAME_VALUES, min_pitches=100)['xwoba_train_diagnostic']['status'],
                         'NOT_AVAILABLE')
        marginals = roa.marginals_from_doc(json.loads(json.dumps(s0['tab_marginals'], default=roa._default)))
        self.assertEqual(len(marginals.situations), reg.TAB_SITUATIONS)
        smoke = roa.build_s0(self.train, GAME_VALUES, smoke=True, seed_role='smoke', fixed_m=20, min_pitches=50)
        self.assertIsNone(smoke['cv_log_loss'])
        for key in ('train_residuals', 'kappa_calibration', 'zeta_calibration', 'xwoba_train_diagnostic'):
            self.assertNotIn(key, smoke)
        self.assertEqual((smoke['kappa_p'], smoke['zeta_p']), ({'S5X': 0.0, 'TAB': 0.0}, {'S5X': 0.0, 'TAB': 0.0}))
        with self.assertRaisesRegex(IntegrityError, 'descriptive output missing'):
            roa.require_s0_descriptive(smoke)
        bad = self.train.copy()
        bad.loc[0, 'game_date'] = pd.Timestamp('2026-04-01')
        with self.assertRaisesRegex(IntegrityError, '2026'):
            roa.build_s0(bad, GAME_VALUES, min_pitches=100)

    def test_smoke_records_costs_and_never_a_metric(self):
        smoke = roa.build_s0(self.train, GAME_VALUES, smoke=True, seed_role='smoke', fixed_m=20, min_pitches=50)
        tab = roa.smoke_tab(smoke, replications=2)
        self.assertEqual(tab['integrity'], {'tab_v2_dp_expectation_equals_truth': 'PASS', 'same_seed_bit_reproducible': 'PASS'})
        projection = roa.smoke_projection(1.0, smoke['timings']['table_fit'], tab['variant_estimate_unit_seconds'],
                                          tab['replication_seconds'], 10.0)
        self.assertEqual(projection['label'], 'projection (not a measurement)')
        self.assertAlmostEqual(projection['projection_term_seconds_per_replication'],
                               6 * smoke['timings']['table_fit'] + 30 * tab['variant_estimate_unit_seconds'], places=12)
        text = json.dumps({**tab, **projection})
        for word in ('b_hat', 'rmse', 'H_point', '"sd"', 'gap', 'delta_true', 'log_loss'):
            self.assertNotIn(word, text)

    def test_s1_cells_order_not_run_and_descriptive_outputs(self):
        doc = json.loads(json.dumps(roa.build_s0(self.train, GAME_VALUES, min_pitches=100), default=roa._default))
        cells = [reg.tab_cells()[0], reg.tab_cells()[5], reg.tab_cells()[30]]
        written = []
        results = roa.stage_s1(doc, write_cell=lambda cell, record: written.append(cell['id']), cells=cells, replications=2)
        self.assertEqual(written, [c['id'] for c in cells])
        self.assertEqual(set(results['first_step_v2_dp_check']), {c['id'] for c in cells if c['law'] == 'V2'})
        self.assertTrue(all(abs(v['expected_delta_hat_0_minus_truth']) <= 1e-12
                            for v in results['first_step_v2_dp_check'].values()))
        self.assertEqual(set(results['acceptance']), {'T-A1'})
        for record in results['cells'].values():  # PM decision (review #10): the surrogate-truth decomposition
            self.assertEqual(set(record['delta_true_surrogate']), set(roa.S1_SURROGATE_KEYS))
        broken = copy.deepcopy(results)
        del broken['cells']['T-A1']['delta_true_surrogate']['R1_b_surrogate_share']
        with self.assertRaisesRegex(IntegrityError, 'surrogate-truth decomposition'):
            roa.require_s1_descriptive(broken, cells)
        broken = copy.deepcopy(results)
        del broken['cells'][cells[2]['id']]
        with self.assertRaisesRegex(IntegrityError, 'TAB cells missing'):
            roa.require_s1_descriptive(broken, cells)
        calls = {'n': 0}

        def deadline():  # 2 calls for the V2 DP checks, then one per replication
            calls['n'] += 1
            if calls['n'] > 4:  # the guard trips during the descriptive cells
                raise roa.HangGuardExceeded('synthetic')
        results = roa.stage_s1(doc, write_cell=lambda cell, record: None, cells=cells, replications=2, deadline=deadline)
        self.assertEqual(results['not_run'], [cells[1]['id'], cells[2]['id']])
        calls['n'] = 0
        with self.assertRaises(roa.HangGuardExceeded):  # before the acceptance cells are sealed: infrastructure
            roa.stage_s1(doc, write_cell=lambda cell, record: None, cells=cells, replications=4, deadline=deadline)

    def test_s4_decision(self):
        public, tab, s5x = _decision_inputs(h={'R1_axR2_3': 0.6})
        record = roa.stage_s4(public, tab, s5x, integrity_failures=(), r3_status='NOT_RUN:R3')
        self.assertEqual(record['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_axR2_3')
        self.assertEqual(record['r3_status'], 'NOT_RUN:R3')
        del tab['acceptance']['T-A2'][reg.BASELINE]
        with self.assertRaisesRegex(IntegrityError, 'T-A2'):
            roa.stage_s4(public, tab, s5x, integrity_failures=(), r3_status='NOT_RUN:R3')


def fake_runs(invalid=None, drop_marker=None):
    """S5-X run records with the registered shape (synthetic numbers)."""
    runs = []
    for spec in reg.S5X_RUNS:
        pairs = {p: {'gap': 0.0, 'se': 0.001, 'valid': True, 'invalid_repetitions': 0} for p in reg.PAIRS}
        for k, pair in (invalid or {}).items():
            if k == spec['k']:
                pairs[pair] = {'gap': None, 'se': None, 'valid': False, 'invalid_repetitions': 1}
        record = {'run': dict(spec), 'repetitions': [{}] * reg.S5X_REPETITIONS, 'inherited_s5_se': 0.001, 'pairs': pairs,
                  'invariant': None}
        if spec['setting'] == 'X1':
            record['s2'] = {'R1_a': reg.cell_outcome(0.0, 1e-6), 'R1_b': reg.cell_outcome(0.0, 1e-6)}
            record['s2_pairs'] = osx.s2_pair_cells(record['s2'], pairs, spec['k'])
            if spec['k'] != drop_marker:
                record['first_order_reference_approximation'] = {'status': 'NOT_RUN', 'reason': 'needs planner calls'}
            if spec['zeta'] == '0':
                record['invariant'] = {'max_abs_gamma_a': 0.0, 'outcome': 'PASS'}
        runs.append(record)
    return runs


class S5XSummaryTests(unittest.TestCase):
    def test_invalid_repetitions_propagate_to_the_decision(self):
        # PM decision (review #1): an invalid repetition in an acceptance-relevant run never supports adoption
        public, tab, _ = _decision_inputs(h={'R1_axR2_3': 0.5})
        decide = lambda s5x: roa.stage_s4(public, tab, s5x, integrity_failures=(), r3_status='NOT_RUN:R3')['label']
        runs = fake_runs()
        self.assertTrue(roa.require_s2_descriptive(runs))
        summary = roa.s5x_summary(runs)
        self.assertEqual(summary['s2']['R1_axR2_3']['invariant'], 'PASS')
        self.assertEqual(decide(summary), 'ADOPT_FOR_2027_PROTOCOL:R1_axR2_3')
        for k in (4, 5, 2, 7):  # S-2 (runs 4, 5), S-1 (run 2) and S-4 (run 7)
            summary = roa.s5x_summary(fake_runs(invalid={k: 'R1_axR2_3'}))
            with self.subTest(run=k):
                self.assertEqual(decide(summary), 'BIAS_NOT_CERTIFIED:R1_axR2_3')
        summary = roa.s5x_summary(fake_runs(invalid={4: 'R1_axR2_3'}))
        cell = summary['s2']['R1_axR2_3']['2*kappa_p']
        self.assertEqual((cell['outcome'], cell['invalid_repetition']), ('INCONCLUSIVE', True))
        self.assertEqual(summary['s2']['R1_axR2_0']['2*kappa_p']['outcome'], 'PASS')  # other pairs unaffected
        self.assertEqual(roa.s5x_summary(fake_runs()[4:])['s2']['R1_axR2_3']['invariant'], 'INVALID')  # run 4 missing

    def test_the_propagation_survives_a_write_and_a_reload(self):
        # PM decision (review round 2 #1): the INCONCLUSIVE cell is JSON-safe and recomputed identically after a restart
        tmp = tmpdir(self)
        runs = fake_runs(invalid={4: 'R1_axR2_3'})
        roa.dump(tmp / 's5x.json', {'runs': runs, **roa.s5x_summary(runs)})
        loaded = json.loads((tmp / 's5x.json').read_text())
        self.assertTrue(roa.require_s5x_consistent(loaded))
        public, tab, _ = _decision_inputs(h={'R1_axR2_3': 0.5})
        record = roa.stage_s4(public, tab, loaded, integrity_failures=(), r3_status='NOT_RUN:R3')
        self.assertEqual(record['label'], 'BIAS_NOT_CERTIFIED:R1_axR2_3')
        self.assertIn('invalid repetitions in run 4', loaded['s2']['R1_axR2_3']['2*kappa_p']['reason'])
        loaded['s2']['R1_axR2_3']['2*kappa_p'] = {'outcome': 'PASS'}
        with self.assertRaisesRegex(IntegrityError, 'differs from its run records'):
            roa.require_s5x_consistent(loaded)

    def test_dump_writes_strict_json_and_lists_nulled_values(self):
        tmp = tmpdir(self)
        roa.dump(tmp / 'x.json', {'a': float('nan'), 'b': [1.0, np.inf], 'c': {'d': np.float64(2.5)}, 'e': np.int64(3)})
        doc = json.loads((tmp / 'x.json').read_text())
        self.assertEqual(doc, {'a': None, 'b': [1.0, None], 'c': {'d': 2.5}, 'e': 3,
                               'nonfinite_nulled_paths': ['$.a', '$.b[1]']})

    def test_missing_s5x_descriptive_outputs_fail_the_stage(self):
        # PM decision (review #10): the first-order approximation is an explicit NOT_RUN marker with its reason
        with self.assertRaisesRegex(IntegrityError, 'NOT_RUN marker'):
            roa.require_s2_descriptive(fake_runs(drop_marker=9))
        with self.assertRaisesRegex(IntegrityError, 'runs missing'):
            roa.require_s2_descriptive(fake_runs()[:-1])
        runs = fake_runs()
        runs[3]['invariant'] = None
        with self.assertRaisesRegex(IntegrityError, 'invariant record missing'):
            roa.require_s2_descriptive(runs)


class RehearsalOrderTests(unittest.TestCase):
    """PM decision (review #9): G-INT on the untouched baseline path before any surrogate or variant."""
    def setUp(self):
        self.decisions, self.facts = synthetic_ledger(seed=5, n_pas=300, n_games=20, n_pitchers=12)
        self.last = {pa: {'game_pk': 900000 + i, 'at_bat_number': 1, 'pitch_number': 1, 'inning': 3, 'inning_topbot': 'Top',
                          'outs_when_up': 1, 'bases': 1, 'home_score': 1, 'away_score': 2, 'stand': 'R',
                          'events': 'single', 'description': 'hit_into_play', 'launch_speed': 95.0, 'launch_angle': 12.0}
                     for i, pa in enumerate(self.facts)}
        self.raw = pd.DataFrame([{'game_pk': r['game_pk'], 'at_bat_number': 1, 'pitch_number': 1,
                                  'game_date': pd.Timestamp('2025-08-01'), 'hc_x': 100.0, 'hc_y': 120.0, 'bb_type': 'line_drive',
                                  'launch_speed': np.nan, 'launch_angle': np.nan, 'estimated_woba_using_speedangle': 0.5}
                                 for r in self.last.values()])
        self.s0 = {'r1_table': synthetic_table(np.random.default_rng(0)), 'census': CENSUS,
                   'train_residuals': {'hit': {'FB': {'n': 10, 'mean_residual': -0.01, 'se': 0.005}}}}
        result, _ = roa.osd.est.estimate(self.decisions, self.facts, draws=200, seed=reg.GAME_BOOTSTRAP_SEED,
                                         invalid_share_max=reg.INVALID_SHARE_MAX, minimum={'games': 5, 'pa_starts': 10},
                                         censoring='l1r')
        self.recorded = osd.recorded_gint_reference(result)

    def run_rehearsal(self, references, order):
        real_gate, real_surrogates = osd.gint_gate, osg.surrogate_rewards

        def gate(*args, **kwargs):
            order.append('gint')
            return real_gate(*args, **kwargs)

        def surrogates(*args, **kwargs):
            order.append('surrogates')
            return real_surrogates(*args, **kwargs)

        def raw_join():
            order.append('raw_join')
            return self.raw
        with mock.patch.object(osd, 'gint_gate', gate), mock.patch.object(osg, 'surrogate_rewards', surrogates):
            return roa.rehearsal(lambda: iter(self.decisions), self.facts, self.last, raw_join=raw_join,
                                 game_values=GAME_VALUES, s0=self.s0, references=references, ess_gate=None,
                                 r3_status='NOT_RUN:R3', draws=200, minimum_games=5, minimum_pa_starts=10,
                                 minimum_pitchers=4, placebos=3, checkpoint=lambda: order.append('checkpoint'))

    def test_gint_runs_first_and_a_mismatch_stops_before_any_surrogate(self):
        order = []
        public, sealed, _, gint = self.run_rehearsal({'s6_dr_a5_record': self.recorded}, order)
        events = [e for e in order if e != 'checkpoint']
        self.assertEqual(events, ['gint', 'raw_join', 'surrogates'])
        # review round 2 #4: memory checkpoints while the protected estimator reads the ledger, before any surrogate
        self.assertIn('checkpoint', order[order.index('gint') + 1:order.index('raw_join')])
        self.assertEqual(gint['s6_dr_a5_record']['not_comparable'], [])
        self.assertEqual(sealed['gint'], gint)
        self.assertTrue(osd.validate_public(json.loads(osd.seal(public, sealed)[0])))
        upper = list(self.recorded['l1r_upper_endpoint_ci95'])
        for key, value in (('l1r_upper_endpoint_ci95', [upper[0], upper[1] + 1e-9]),
                           ('ess_all_e0_reference', {**self.recorded['ess_all_e0_reference'],
                                                     'pa': self.recorded['ess_all_e0_reference']['pa'] + 1e-9})):
            order = []
            with self.subTest(key), self.assertRaisesRegex(IntegrityError, 'G-INT failed'):
                self.run_rehearsal({'s6_dr_a5_record': {**self.recorded, key: value}}, order)
            self.assertEqual([e for e in order if e != 'checkpoint'], ['gint'])  # nothing after a failed gate


# ================================================================ dependency-injected orchestration
class OrchestrationTests(unittest.TestCase):
    def test_smoke_then_s0_then_s1_then_decide_refuses_before_its_attempt(self):
        ws = Workspace(self)
        env = WorkspaceEnv(ws)
        with self.assertRaisesRegex(IntegrityError, 'A2-SM is not sealed'):  # S0 needs the sealed smoke first
            roa.main(ws.argv('s0', 'A2-S0-a1'), env=env)
        self.assertEqual(ws.kinds('A2-S0'), [])
        roa.main(ws.argv('smoke', 'A2-SM-a1'), env=env)
        with self.assertRaisesRegex(IntegrityError, 'already sealed'):
            roa.main(ws.argv('smoke', 'A2-SM-a2'), env=env)
        with watch_parquet() as log:
            roa.main(ws.argv('s0', 'A2-S0-a1'), env=env)
        self.assertNotIn('statcast_2025.parquet', {f for f, _, _ in log['reads']})  # DEV row groups never decoded
        self.assertEqual(sorted({rg for f, rg, _ in log['reads'] if f == ALIGNED_NAME}), [1, 2])
        self.assertNotIn('pitches.parquet', {f for f, _, _ in log['reads']})  # the original is never decoded
        s0_dir, _ = roa.require_seal(ws.manifest, 'A2-S0')
        s0 = json.loads((s0_dir / 's0.json').read_text())
        self.assertIn('intentional exception to cross-fitting', s0['provenance']['zeta_residuals'])
        small = functools.partial(REAL_STAGE_S1, cells=[reg.tab_cells()[0], reg.tab_cells()[10]], replications=2)
        with mock.patch.object(roa, 'stage_s1', small):
            roa.main(ws.argv('s1', 'A2-S1-a1'), env=env)
        cells = [e['payload']['cell'] for e in ws.manifest.entries() if e['kind'] == 'cell_seal']
        self.assertEqual(cells, ['T-A1', reg.tab_cells()[10]['id']])
        tab_dir, _ = roa.require_seal(ws.manifest, 'A2-S1')
        tab = json.loads((tab_dir / 'tab.json').read_text())
        self.assertEqual(tab['cells']['T-A1']['pairs'][reg.BASELINE]['outcome'], 'EXACT_DP')
        with self.assertRaisesRegex(IntegrityError, 'decision needed'):
            roa.main(ws.argv('decide', 'A2-S4-a1'), env=env)
        self.assertEqual(ws.kinds('A2-S4'), [])  # the refusal consumed no attempt
        self.assertEqual([k for k in ws.kinds() if k in ('attempt_start', 'stage_seal')],
                         ['attempt_start', 'stage_seal'] * 3)


class TerminalTests(unittest.TestCase):
    def test_a_descriptive_tab_stop_seals_its_not_run_cells(self):
        # PM decision (review round 2 #15): the work deadline after the acceptance cells writes NOT_RUN records and
        # the stage seals inside the finalization reserve
        ws = Workspace(self)
        env = WorkspaceEnv(ws)
        roa.main(ws.argv('smoke', 'A2-SM-a1'), env=env)
        roa.main(ws.argv('s0', 'A2-S0-a1'), env=env)
        cells = [reg.tab_cells()[0], reg.tab_cells()[10], reg.tab_cells()[20]]
        hard = reg.HANG_GUARDS['A2-S1']
        reserve = min(reg.FINALIZATION_RESERVE_SECONDS, hard / 10)
        real_cell = osx.tab_cell

        def tab_cell(world, cell, **kwargs):
            record = real_cell(world, cell, **kwargs)
            if cell['role'] == 'acceptance':
                env.clock = hard - reserve + 1  # past the work deadline, inside the hard cap
            return record
        small = functools.partial(REAL_STAGE_S1, cells=cells, replications=2)
        with mock.patch.object(roa, 'stage_s1', small), mock.patch.object(osx, 'tab_cell', tab_cell):
            roa.main(ws.argv('s1', 'A2-S1-a1'), env=env)
        out, _ = roa.require_seal(ws.manifest, 'A2-S1')
        tab = json.loads((out / 'tab.json').read_text())
        self.assertEqual((tab['not_run'], list(tab['acceptance'])), ([cells[1]['id'], cells[2]['id']], ['T-A1']))
        for cid in tab['not_run']:
            record = json.loads((out / 'cells' / f'{cid}.json').read_text())
            self.assertEqual((record['status'], record['cell']['id']), ('NOT_RUN', cid))
            self.assertIn('never rerun', record['reason'])
        seals = [(e['payload']['cell'], e['payload']['status']) for e in ws.manifest.entries() if e['kind'] == 'cell_seal']
        self.assertEqual(seals, [('T-A1', 'COMPLETED'), (cells[1]['id'], 'NOT_RUN'), (cells[2]['id'], 'NOT_RUN')])

    def test_an_exhausted_s0_integrity_failure_is_a_terminal_decision(self):
        # PM decision (review round 2 #16): FAILED_INTEGRITY is recorded without running the blocked descendants
        ws = Workspace(self)
        env = WorkspaceEnv(ws)
        roa.main(ws.argv('smoke', 'A2-SM-a1'), env=env)
        commit = ws.config['registration_input_pins']['execution_source_commit']
        for attempt in (1, 2):
            ws.manifest.append('attempt_start', {'stage': 'A2-S0', 'output_dir': str(ws.output(f'A2-S0-a{attempt}'))})
            ws.manifest.append('stage_failure', {'stage': 'A2-S0', 'class': 'deterministic_integrity',
                                                 'output_dir': str(ws.output(f'A2-S0-a{attempt}'))})
            if attempt == 1:
                ws.manifest.append('bugfix_addendum', {'stage': 'A2-S0', 'fix_commit': commit,
                                                       'new_files_sha256': roa.new_file_hashes()})
        with self.assertRaisesRegex(IntegrityError, 'A2-S0 is not sealed'):
            roa.main(ws.argv('s1', 'A2-S1-a1'), env=env)
        roa.main(ws.argv('decide', 'A2-S4-a1'), env=env)
        out, _ = roa.require_seal(ws.manifest, 'A2-S4')
        record = json.loads((out / 'decision_record.json').read_text())
        self.assertEqual((record['label'], record['failed_integrity'], record['blocked'], record['not_run']),
                         ('FAILED_INTEGRITY', ['A2-S0'], ['A2-S1', 'A2-S2', 'A2-S3'], []))
        self.assertEqual(record['stage_status']['A2-SM'], 'sealed')
        self.assertEqual(ws.kinds('A2-S1'), [])  # the blocked descendants never started


def rehearsal_data(ws):
    """A synthetic sealed S6 rehearsal on the workspace's DEV row group (one ledger PA per DEV PA, every L1-R
    resolution) and its dr.json-shaped record from the untouched registered estimator."""
    dev = ws.groups[3]
    last = dev.sort_values(roa.KEY).groupby(['game_pk', 'at_bat_number']).tail(1)
    rng = np.random.default_rng(11)
    decisions, facts, rows = [], {}, {}
    for i, (_, row) in enumerate(last.iterrows()):
        pa_id = f'{row.game_pk}:{row.at_bat_number}'
        pa_rows, info = synthetic_pa(rng, pa_id, int(row.game_pk), int(row.pitcher), KINDS[i % len(KINDS)])
        decisions.extend(pa_rows)
        facts[pa_id] = info
        rows[pa_id] = row.to_dict()
    result0, _, _ = osd.gint_gate(lambda: iter(decisions), facts, references={})  # registered draws and minimums
    s6_dr = roa.json_safe({**result0, 'games': sorted(int(g) for g in last['game_pk'].unique())})
    return {'decisions': decisions, 'facts': facts, 'last': rows, 's6_dr': s6_dr}


def quote_s6(config, ws):
    """The synthetic registration quotes the synthetic S6 record (G-INT compares both references)."""
    ws.rehearsal = rehearsal_data(ws)
    values = osd.gint_values(ws.rehearsal['s6_dr'])
    s6 = config['mde_from_records']['S6_rehearsal_150_dev_games']
    for key in ('l1r_delta_bounds', 'l1r_lower_endpoint_ci95', 'l2_ci95', 'worst_case_residual_share_s', 'resolutions',
                'ess_all_e0_candidate'):
        s6[key]['value'] = values[key]
    config['mde_from_records']['reference_magnitudes']['exposed_point_estimates']['S6_l2_delta_mean'] = values['l2_delta_mean']


class ChainEnv(WorkspaceEnv):
    """Synthetic A2-S2 world (the synthetic candidate runtime with fake G0 / WE of test_policy_validation) and
    synthetic A2-S3 inputs; everything else is the registered code path."""
    def __init__(self, ws, fx):
        super().__init__(ws)
        self.fx = fx

    def s5x_inputs(self, config, local, base, out, checkpoint):
        fx = self.fx

        class World:
            game_values = fx.syn.game_values

            def candidate(self, spec_run):
                return fx.world(f"a2-k{spec_run['k']:02d}")
        return World()

    def s6_record(self, config):
        return self.ws.root / 's6' / 'ledger.jsonl', self.ws.rehearsal['s6_dr']

    def rehearsal_inputs(self, config, local, base, ledger_path, s6_dr, out, checkpoint):
        data = self.ws.rehearsal
        return roa.Rehearsal(decisions=lambda: iter(data['decisions']), facts=data['facts'], last_rows=data['last'],
                             window=('2025-08-01', '2025-08-20'), ess_gate=None, game_values=GAME_VALUES)

    def committed(self, path):
        return True


class FullChainTests(unittest.TestCase):
    """Review round 2 #20: the complete registered path on synthetic inputs, A2-SM to unblind, with the full
    15-run S5-X matrix (X2 single-member world included) and the A2-S3 evaluator at the registered draws."""
    def test_smoke_to_unblind(self):
        import test_policy_validation as tpv
        fx = tpv.SemiSyntheticTests('test_v2_matches_world_truth_and_is_not_vacuous')
        fx.setUp()
        self.addCleanup(shutil.rmtree, fx.root, True)
        seed = fx.world('a2-probe')[1].g0.models[0].seed
        ws = Workspace(self, hook=quote_s6, pins={'x2_world_predictor_identity': {'kind': 'g0_single_member', 'seed': seed}})
        env = ChainEnv(ws, fx)
        roa.main(ws.argv('smoke', 'A2-SM-a1'), env=env)
        roa.main(ws.argv('s0', 'A2-S0-a1'), env=env)
        acceptance = functools.partial(REAL_STAGE_S1, cells=reg.tab_cells()[:4], replications=2)
        with mock.patch.object(roa, 'stage_s1', acceptance):
            roa.main(ws.argv('s1', 'A2-S1-a1'), env=env)
        with mock.patch.object(reg, 'S5X_CAP', 4):
            roa.main(ws.argv('s2', 'A2-S2-a1'), env=env)
        s2_dir, _ = roa.require_seal(ws.manifest, 'A2-S2')
        s5x = json.loads((s2_dir / 's5x.json').read_text())
        self.assertEqual([r['run']['k'] for r in s5x['runs']], list(range(1, 16)))
        for record in s5x['runs']:
            expected = 'g0_single_member' if record['run']['setting'] == 'X2' else 'bound_g0'
            self.assertEqual(record['world_predictor']['kind'], expected)
        self.assertEqual({r['world_predictor'].get('seed') for r in s5x['runs'] if r['run']['setting'] == 'X2'}, {seed})
        self.assertTrue(roa.require_s5x_consistent(s5x))
        roa.main(ws.argv('s3', 'A2-S3-a1'), env=env)
        s3_dir, _ = roa.require_seal(ws.manifest, 'A2-S3')
        public = json.loads((s3_dir / 'public.json').read_text())
        self.assertTrue(osd.validate_public(public))
        gint = json.loads((s3_dir / 'gint.json').read_text())
        self.assertEqual(gint['s6_dr_a5_record']['not_comparable'], [])
        self.assertEqual(gint['listed_keys_never_compared'], [])
        self.assertEqual(public['validity'], {'point_set_meets_minimum': True, 'game_bootstrap_eligible': True,
                                              'pitcher_bootstrap_eligible': True})
        roa.main(ws.argv('decide', 'A2-S4-a1'), env=env)
        s4_dir, _ = roa.require_seal(ws.manifest, 'A2-S4')
        record = json.loads((s4_dir / 'decision_record.json').read_text())
        self.assertTrue(record['label'] in ('NO_PRECISION_GAIN', 'NOT_EVALUABLE')
                        or record['label'].startswith(('ADOPT_FOR_2027_PROTOCOL:', 'BIAS_NOT_CERTIFIED:')), record['label'])
        roa.main(ws.argv('unblind', 'A2-UB-a1', '--decision-record', str(s4_dir / 'decision_record.json')), env=env)
        ub_dir, _ = roa.require_seal(ws.manifest, 'A2-UB')
        opened = json.loads((ub_dir / 'unblinded.json').read_text())
        self.assertEqual(set(opened['pairs']), set(reg.PAIRS))
        self.assertEqual(opened['decision_record_sha256'], reg.file_sha256(s4_dir / 'decision_record.json'))
        kinds = [(e['payload'].get('stage'), e['kind']) for e in ws.manifest.entries() if e['kind'] in ('attempt_start',
                                                                                                     'stage_seal')]
        self.assertEqual(kinds, [(stage, kind) for stage in ('A2-SM', 'A2-S0', 'A2-S1', 'A2-S2', 'A2-S3', 'A2-S4', 'A2-UB')
                                 for kind in ('attempt_start', 'stage_seal')])


class S6RecordTests(unittest.TestCase):
    """The real ``Environment.s6_record`` on a synthetic S6-dr-a5 stage directory (bug-fix addendum 1: A2-S3
    attempt 1 of 2026-10-27 aborted deterministically). The stage ran under an earlier registration chain of the
    same ML-POLICY config and the same candidate identity (the D125 chain config + addendum-1..7-d124; the pinned
    chain is the D127 chain config + addendum-1..8-d127); the first version demanded byte equality of the two
    chains, which can never hold. The stage is tied to the pinned candidate by identity (dr.json, ledger header)
    and lineage (the committed blob at its recorded commit, its addenda unchanged). Synthetic files only: no real
    data, ledger or model is read; the committed-blob lookup is injected."""
    IDENTITY = 'c1' * 32
    OTHER = 'd2' * 32
    COMMIT = 'a' * 40

    def setUp(self):
        self.root = tmpdir(self)
        self.main = self.root / 'main' / 'configs'
        self.main.mkdir(parents=True)
        self.old_config = b'{"identity_registration": {}, "version": "D125"}\n'
        self.old_sha = roa.hashlib.sha256(self.old_config).hexdigest()
        (self.main / 'ml-policy.json').write_bytes(b'{"identity_registration": {}, "version": "D127", "mlb2026_ope": {}}\n')
        (self.main / 'ml-policy.addendum-1-d124.json').write_text('{"parent_sha256": "old"}\n')
        (self.main / 'ml-policy.addendum-1-d127.json').write_text('{"parent_sha256": "new"}\n')
        (self.main / 'ml-policy.addendum-2-d127.json').write_text('{"parent_sha256": "new-2"}\n')
        self.stage = self.root / 'ML-POLICY-VAL-v1' / 'S6-dr-a5'
        self.stage.mkdir(parents=True)
        self.header = {'kind': 'header', 'pins': {'candidate': {'contract': 'ML-POLICY-IDENTITY-v1', 'sha256': self.IDENTITY},
                                                  'reference': {}}, 'prev': None, 'runtime_sha256': 'af' * 32, 'seq': 0,
                       'sha256': 'ee' * 32}
        self.dr = {'identity_sha256': self.IDENTITY, 'games': [1, 2, 3], 'status': 'COMPLETE'}
        self.started = {'command': 'dr-evaluate', 'config_sha256': self.old_sha,
                        'registration_chain': [{'path': 'configs/ml-policy.json', 'sha256': self.old_sha},
                                               {'path': 'configs/ml-policy.addendum-1-d124.json',
                                                'sha256': reg.file_sha256(self.main / 'ml-policy.addendum-1-d124.json')}],
                        'git': {'commit': self.COMMIT, 'code_dirty': False}}
        self.blobs = {(self.COMMIT, 'configs/ml-policy.json'): self.old_sha}

    def write_stage(self):
        ledger = self.stage / 'ledger-dev-0.jsonl'
        ledger.write_text(json.dumps(self.header) + '\n' + json.dumps({'kind': 'decision', 'seq': 1}) + '\n')
        (self.stage / 'dr.json').write_text(json.dumps(self.dr))
        (self.stage / 'started.json').write_text(json.dumps(self.started))
        listed = {name: reg.file_sha256(self.stage / name) for name in ('started.json', 'dr.json', ledger.name)}
        (self.stage / 'manifest.json').write_text(json.dumps({'command': 'dr-evaluate', 'artifact_sha256': listed}))
        chain = [{'path': str(self.main / name), 'sha256': reg.file_sha256(self.main / name)}
                 for name in ('ml-policy.json', 'ml-policy.addendum-1-d127.json', 'ml-policy.addendum-2-d127.json')]
        config = {'identities': {'candidate_policy_sha256': self.IDENTITY},
                  'registration_input_pins': {'s6_dr_a5_ledger_path': str(ledger),
                                              's6_dr_a5_ledger_sha256': reg.file_sha256(ledger),
                                              's6_dr_a5_manifest_sha256': reg.file_sha256(self.stage / 'manifest.json'),
                                              'policy_registration_chain': chain}}
        return ledger, config

    def env(self):
        test = self

        class Env(roa.Environment):
            def blob_sha256(self, commit, relative_posix_path):
                return test.blobs.get((commit, relative_posix_path))
        return Env()

    def test_an_earlier_registration_of_the_same_candidate_is_accepted(self):
        """The A2-S3 attempt-1 situation: the chains share no link; identity and lineage agree."""
        ledger, config = self.write_stage()
        recorded = [link['sha256'] for link in self.started['registration_chain']]
        pinned = [link['sha256'] for link in config['registration_input_pins']['policy_registration_chain']]
        self.assertFalse(set(recorded) & set(pinned))
        self.assertNotEqual(len(recorded), len(pinned))
        path, dr = self.env().s6_record(config)
        self.assertEqual((path, dr), (ledger, self.dr))

    def test_an_absolute_recorded_path_under_the_main_checkout_is_accepted(self):
        self.started['registration_chain'][0]['path'] = str(self.main / 'ml-policy.json')
        self.started['registration_chain'][1]['path'] = str(self.main / 'ml-policy.addendum-1-d124.json')
        ledger, config = self.write_stage()
        self.assertEqual(self.env().s6_record(config)[0], ledger)

    def test_another_candidate_identity_is_refused(self):
        for where in ('dr', 'header', 'both'):
            with self.subTest(where=where):
                self.setUp()
                if where in ('dr', 'both'):
                    self.dr['identity_sha256'] = self.OTHER
                if where in ('header', 'both'):
                    self.header['pins']['candidate']['sha256'] = self.OTHER
                _, config = self.write_stage()
                with self.assertRaisesRegex(IntegrityError, 'another candidate identity than the pinned one'):
                    self.env().s6_record(config)

    def test_the_header_identity_contract_is_required(self):
        for pins in ({'candidate': {'sha256': self.IDENTITY}}, {'candidate': {'contract': 'OTHER', 'sha256': self.IDENTITY}}, {}):
            with self.subTest(pins=pins):
                self.setUp()
                self.header['pins'] = pins
                _, config = self.write_stage()
                with self.assertRaisesRegex(IntegrityError, 'ledger header pins another candidate identity'):
                    self.env().s6_record(config)

    def test_another_registration_file_is_refused(self):
        self.started['registration_chain'][0]['path'] = 'configs/other-policy.json'
        self.blobs[(self.COMMIT, 'configs/other-policy.json')] = self.old_sha
        _, config = self.write_stage()
        with self.assertRaisesRegex(IntegrityError, 'another ML-POLICY registration file'):
            self.env().s6_record(config)

    def test_an_absolute_recorded_path_outside_the_checkout_is_refused(self):
        elsewhere = tmpdir(self) / 'configs'
        elsewhere.mkdir()
        (elsewhere / 'ml-policy.json').write_bytes(self.old_config)
        self.started['registration_chain'][0]['path'] = str(elsewhere / 'ml-policy.json')
        _, config = self.write_stage()
        with self.assertRaisesRegex(IntegrityError, "outside the pinned chain's checkout"):
            self.env().s6_record(config)

    def test_the_recorded_config_must_be_the_committed_blob_at_the_recorded_commit(self):
        for blobs in ({}, {(self.COMMIT, 'configs/ml-policy.json'): self.OTHER}, {('b' * 40, 'configs/ml-policy.json'): self.old_sha}):
            with self.subTest(blobs=blobs):
                self.setUp()
                self.blobs = blobs
                _, config = self.write_stage()
                with self.assertRaisesRegex(IntegrityError, 'committed blob'):
                    self.env().s6_record(config)

    def test_a_changed_or_missing_addendum_is_refused(self):
        for action in ('change', 'remove'):
            with self.subTest(action=action):
                self.setUp()
                _, config = self.write_stage()
                target = self.main / 'ml-policy.addendum-1-d124.json'
                target.write_text('{"parent_sha256": "edited"}\n') if action == 'change' else target.unlink()
                with self.assertRaisesRegex(IntegrityError, 'addendum missing or changed'):
                    self.env().s6_record(config)

    def test_a_missing_chain_or_commit_is_refused(self):
        for field, value, message in (('registration_chain', [], 'records no ML-POLICY registration chain'),
                                      ('registration_chain', [{'path': 'configs/ml-policy.json'}], 'records no ML-POLICY registration chain'),
                                      ('git', {}, 'records no source commit'), ('git', {'commit': 'HEAD'}, 'records no source commit')):
            with self.subTest(field=field, value=value):
                self.setUp()
                self.started[field] = value
                _, config = self.write_stage()
                with self.assertRaisesRegex(IntegrityError, message):
                    self.env().s6_record(config)

    def test_the_stage_bytes_stay_pinned(self):
        ledger, config = self.write_stage()
        (self.stage / 'dr.json').write_text(json.dumps({**self.dr, 'games': [1, 2]}))
        with self.assertRaisesRegex(IntegrityError, 'dr.json differs from its stage manifest'):
            self.env().s6_record(config)
        self.setUp()
        ledger, config = self.write_stage()
        ledger.write_text(ledger.read_text() + json.dumps({'kind': 'decision', 'seq': 2}) + '\n')
        with self.assertRaisesRegex(IntegrityError, 'ledger differs from its pin'):
            self.env().s6_record(config)
        self.setUp()
        _, config = self.write_stage()
        config['registration_input_pins']['s6_dr_a5_manifest_sha256'] = '0' * 64
        with self.assertRaisesRegex(IntegrityError, 'manifest differs from its registration pin'):
            self.env().s6_record(config)

    def test_the_real_blob_lookup_reads_this_repository(self):
        """``Environment.blob_sha256`` hashes a committed blob of this repository and is None otherwise."""
        env = roa.Environment()
        head = roa.git('rev-parse', 'HEAD')
        rel = 'experiments/pitchmdp/pitchmdp/ope_redesign_registry.py'
        expected = roa.hashlib.sha256(subprocess.check_output(['git', 'show', f'{head}:{rel}'], cwd=roa.REPO)).hexdigest()
        self.assertEqual(env.blob_sha256(head, rel), expected)
        self.assertIsNone(env.blob_sha256(head, 'configs/does-not-exist.json'))
        self.assertIsNone(env.blob_sha256('0' * 40, rel))

    def test_policy_registration_requires_the_pinned_candidate_identity(self):
        """A pinned chain whose addendum registers another expected identity is refused; the pinned one loads."""
        for expected, ok in ((self.IDENTITY, True), (self.OTHER, False), (None, True)):
            with self.subTest(expected=expected):
                root = tmpdir(self)
                config_path = root / 'ml-policy.json'
                config_path.write_text(json.dumps({'identity_registration': {}, 'registered_inputs': {}}))
                addendum = {'parent_sha256': reg.file_sha256(config_path), 'stage': 'v2-world'}
                if expected is not None:
                    addendum['expected_identity_sha256'] = expected
                addendum_path = root / 'ml-policy.addendum-1.json'
                addendum_path.write_text(json.dumps(addendum))
                config = {'identities': {'candidate_policy_sha256': self.IDENTITY},
                          'registration_input_pins': {'policy_registration_chain': [
                              {'path': str(config_path), 'sha256': reg.file_sha256(config_path)},
                              {'path': str(addendum_path), 'sha256': reg.file_sha256(addendum_path)}]}}
                if ok:
                    rpv, policy = roa.Environment().policy_registration(config)
                    self.assertEqual(policy['expected_identity_sha256'], expected)
                else:
                    with self.assertRaisesRegex(IntegrityError, 'another candidate identity than the pinned one'):
                        roa.Environment().policy_registration(config)


class CommittedTests(unittest.TestCase):
    def test_files_outside_the_repository_are_never_committed(self):
        with tempfile.NamedTemporaryFile() as handle:
            self.assertFalse(roa.committed(handle.name))


if __name__ == '__main__':
    unittest.main()
