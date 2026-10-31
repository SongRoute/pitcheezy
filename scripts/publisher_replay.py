"""Offline rehearsal: replay ONE finished capture through the public publisher and measure what it writes.

    python -B scripts/publisher_replay.py --job-dir <job> --run-name run-001 --game 849831 \
        --capture-db <job>/input/capture-849831.sqlite3 --capture-sha256 <sha> --commit <git sha> \
        --config <job>/replay_config.json --policy armb

The capture copy is read through ``capture_to_replay``'s reader (read-only, immutable; the original capture is
never opened here: the database must sit inside the job folder). A simulated clock starts at the capture's first
receipt and moves one publish interval per cycle; at simulated time ``t`` the publisher is answered with the
newest stored version the capture had received by ``t`` (``observer_app.capture_replay``): no look-ahead.

Everything the run writes is under ``<job>/<run-name>/``: the publisher's OWN new store and ledger
(``publisher-store.sqlite3``), its output (``out/``), state, temporary files, the policy's scratch folder and style
cache copy, and the results (``CYCLES.jsonl``, ``SUMMARY.json``, ``SUMMARY.md``). No snapshot recording, no port,
no network, no other process. An interpreter audit hook (a tripwire, not isolation: native-library I/O is
invisible to it) refuses and records any write outside that folder, any new process and any socket
(``observer_app.publish_audit.FsAudit``) and lists the files read under T7. Independently, the folders named in
the config (plus this worktree and the capture original) are listed by path, size and mtime before and after, and
any difference fails the run.

The run stops at the first broken invariant, outside write, integrity or policy failure, memory or time cap; what
was written until then stays. ``SUMMARY.json`` exists from the first moment with ``incomplete: true`` and is the
ONE record of how the run stands; its final form is the last thing written.
``--policy ledger`` (DEMO-WS-MOMENTS-1029): the recommendation of every pending state is READ from a copy of the
live run's own store (``--ledger-db``: its ``served_recommendations`` rows, matched by ``state_sha256``) instead of
being computed (``observer_app.ledger_policy``); the pitchers the live policy refused are named by a registered
support map (``--ledger-support``). No model, T7 or MPS is touched; the copy and the map must sit inside the job
folder and have their registered sha256. ``--keep-moments`` keeps, under ``<run>/moments/<cycle>/``, a gzip copy of
every published file of each cycle whose game body changed and holds a pending pitch (``next``), with
``moments/INDEX.jsonl`` naming the cycle, pitch key and display state: the mid-game bodies a static demo can serve.

Exit codes: 0 completed, 2 refused before start (nothing written), 3 stopped or failed, 87 killed by its own
watchdog.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True  # before any other import: no __pycache__ outside the job folder

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import closing  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import sqlite3  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO/'scripts', REPO/'apps/observer/backend'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import capture_to_replay as c2r  # noqa: E402  (stdlib only: the capture reader and the schedule item)

T7 = Path('/Volumes/T7 Shield')
STYLE_DIR = T7/'pitcheezy/pitchmdp/runs/DEMO-WS-2026/style'
CLEARED_ENV = ('PITCHEEZY_OBSERVER_LIVE_STORE', 'PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', 'PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S',
               'PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE', 'PITCHEEZY_OBSERVER_LIVE_DELAY_S', 'PITCHEEZY_OBSERVER_REC_LEDGER',
               'PITCHEEZY_OBSERVER_FEED_LAG_DELAYED_S', 'PITCHEEZY_PUBLISH_OUT_DIR', 'PITCHEEZY_PUBLISH_STATE',
               'PITCHEEZY_PUBLISH_INTERVAL_S', 'PITCHEEZY_PUBLISH_DELAY_S', 'PITCHEEZY_PUBLISH_COMMIT', 'PITCHEEZY_PLAYER_SNAPSHOT_DIR')
# Every default folder of the observer app that would otherwise point at the external volume (or the home folder)
# is pinned to a folder inside the run, through the app's own environment variables, BEFORE the app is imported:
# ``observer_app.main`` builds a default app at import time and resolves these paths. The server's defaults are
# not changed.
PINNED_ENV = {'PITCHEEZY_OBSERVER_RUN': 'legacy-run', 'PITCHEEZY_FEED_DIR': 'empty', 'PITCHEEZY_WATCH_DIR': 'empty',
              'PITCHEEZY_INTENT_DIR': 'empty', 'PITCHEEZY_DEMO_DIR': 'demo',
              # caches a numeric library may want in the home folder: sent into the run folder before torch is imported
              # (the policy code imports neither matplotlib nor torch.hub; these are set anyway, they cost nothing)
              'MPLCONFIGDIR': 'cache/matplotlib', 'XDG_CACHE_HOME': 'cache/xdg', 'TORCH_HOME': 'cache/torch',
              'PYTORCH_KERNEL_CACHE_PATH': 'cache/torch-kernels'}
CACHE_DIRS = ('~/.cache', '~/Library/Caches')  # looked at before and after a policy run: top-level names and mtimes only
CACHE_ENTRY_LIMIT = 5000
CONFIG_KEYS = ('publish_interval_s', 'fixed_delay_s', 'tail_cycles', 'max_rss_bytes', 'min_free_bytes', 'max_seconds',
               'settle_cap_s', 'max_body_bytes', 'policy_load_cap_s', 'min_free_bytes_running', 'free_check_every_cycles',
               'policy_device', 'style_sha256', 'capture_original_dir', 'policy_dirs', 'listing_time_limit_s',
               'free_floor_consecutive', 'max_measure_failures')
PACKAGES = ('numpy', 'pandas', 'torch', 'fastapi', 'starlette', 'pydantic')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_HIGH_WATERMARK_RATIO')


class Refused(Exception):
    """Not started: nothing was written."""


class Stop(Exception):
    """A registered stop condition: the run ends here and keeps what it has."""

    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class Interrupted(BaseException):
    """A termination signal reached this process."""


UNKNOWN = object()  # a reading that could not be taken: neither a number nor 'not in use'


class MeasureError(Exception):
    """A resource reading could not be taken (distinct from a reading of zero or 'not in use')."""


def sha256_file(path):
    return c2r.file_sha256(path)


def capture_rows(db_path, game_pk, max_body_bytes):
    """(rows, info) of a capture store: ``rows`` = ``[(fetched, load)]`` in receipt (``seq``) order for
    ``capture_replay.CaptureReplayTransport``. Every stored version is read once here, before anything runs:
    a body that is not one complete gzip member within the limit, does not match its stored sha256 or is not JSON
    refuses the run (``Refused``). The same pass finds the FIRST version whose status is Final (by receipt time,
    served later or not): ``info['final_fetched']``. Bodies are then read again one at a time as the replay asks
    for them; a mismatch then sets ``info['integrity_error']`` (the runner stops on it) and raises."""
    db = c2r.open_source(db_path)
    try:
        meta = db.execute('SELECT seq, fetched, sha FROM snapshots WHERE game_pk=? ORDER BY seq', (int(game_pk),)).fetchall()
    except sqlite3.Error as error:
        db.close()
        raise Refused(f'the capture cannot be read: {type(error).__name__}') from None
    if not meta:
        db.close()
        raise Refused(f'the capture holds no stored version of game {game_pk}')
    info = {'versions': len(meta), 'first_fetched': meta[0][1], 'last_fetched': meta[-1][1], 'final_fetched': None,
            'final_seq': None, 'final_index': None, 'loads': 0, 'db': db, 'integrity_error': None, 'seqs': [seq for seq, _, _ in meta]}
    lock = threading.Lock()

    def read(seq, sha):
        with lock:
            blob = db.execute('SELECT body FROM snapshots WHERE seq=?', (seq,)).fetchone()[0]
        raw = c2r.inflate(blob, max_body_bytes)
        if raw is None or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'stored version {seq} does not match its sha256 or exceeds the body limit')
        return json.loads(raw)

    for index, (seq, fetched, sha) in enumerate(meta):
        try:
            body = read(seq, sha)
        except ValueError as error:
            db.close()
            raise Refused(str(error)) from None
        if info['final_fetched'] is None and ((body.get('gameData') or {}).get('status') or {}).get('abstractGameState') == 'Final':
            info['final_fetched'], info['final_seq'], info['final_index'] = fetched, seq, index

    def loader(seq, sha):
        def load():
            try:
                body = read(seq, sha)
            except Exception as error:  # noqa: BLE001 - the feed layer would turn this into an ordinary outage
                info['integrity_error'] = {'seq': seq, 'error': type(error).__name__}
                raise
            info['loads'] += 1
            return body
        return load
    return [(fetched, loader(seq, sha)) for seq, fetched, sha in meta], info


def free_memory_vm_stat():
    """free + inactive + speculative pages by ``vm_stat`` (a process: only before the audit is armed), or None."""
    try:
        out = subprocess.run(['/usr/bin/vm_stat'], capture_output=True, text=True, timeout=10, env={'LC_ALL': 'C'}).stdout
        page = int(re.search(r'page size of (\d+) bytes', out)[1])
        pages = sum(int(re.search(rf'^{name}:\s+(\d+)\.', out, re.M)[1]) for name in ('Pages free', 'Pages inactive', 'Pages speculative'))
        return page*pages
    except Exception:  # noqa: BLE001
        return None


def free_memory_bytes():
    """The same quantity read in-process (Mach ``host_statistics64``; no child process, so usable during the
    run), or None where that call does not exist."""
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        libc.mach_host_self.restype = ctypes.c_uint
        libc.host_statistics64.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        buffer, count = ctypes.create_string_buffer(4*64), ctypes.c_uint(64)
        if libc.host_statistics64(libc.mach_host_self(), 4, buffer, ctypes.byref(count)) != 0 or count.value < 24:  # HOST_VM_INFO64
            return None
        # vm_statistics64: free, active, inactive, wire (natural_t) then nine 64-bit counters, purgeable, speculative
        fields = struct.unpack_from('<4I9Q2I', buffer.raw)
        return os.sysconf('SC_PAGE_SIZE')*(fields[0]+fields[2]+fields[14])
    except Exception:  # noqa: BLE001
        return None


def versions():
    """Versions of what the result depends on, read from package metadata (nothing is imported for it)."""
    from importlib import metadata
    found = {'python': sys.version.split()[0], 'sqlite': sqlite3.sqlite_version}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def threads():
    out = {'env': {name: os.environ.get(name) for name in THREAD_ENV}, 'torch_threads': None, 'cpu_count': os.cpu_count()}
    torch = sys.modules.get('torch')
    if torch is not None:
        out['torch_threads'] = torch.get_num_threads()
    return out


def mps_bytes():
    """Memory the Metal (MPS) backend holds for this process; None only when torch is not loaded or has no MPS
    (nothing to measure). A reading that fails raises ``MeasureError``: it is never taken for zero. Unified
    memory: it is real RAM and is not part of ``ru_maxrss``."""
    torch = sys.modules.get('torch')
    if torch is None:
        return None
    try:
        if not torch.backends.mps.is_available():
            return None
        return int(torch.mps.driver_allocated_memory())
    except Exception as error:  # noqa: BLE001
        raise MeasureError(f'MPS memory: {type(error).__name__}') from None


def policy_devices(policy):
    """Devices the bound ARM-B members actually sit on (``['mps']``), or None while no game is bound or the
    objects cannot be read."""
    found = set()
    for binding in (getattr(policy, 'bindings', None) or {}).values():
        components = binding.get('components') if isinstance(binding, dict) else None
        for member in getattr(getattr(components, 'g0', None), 'models', None) or []:
            device = getattr(getattr(getattr(member, 'model', None), 'global_model', None), 'device', None)
            found.add('unknown' if device is None else str(device))
    return sorted(found) or None


def head_commit(repo=REPO):
    """HEAD's commit sha from the git files alone (no git process). None if it cannot be read."""
    try:
        dot = Path(repo)/'.git'
        gitdir = dot if dot.is_dir() else (Path(repo)/dot.read_text().split('gitdir:', 1)[1].strip()).resolve()
        head = (gitdir/'HEAD').read_text().strip()
        if not head.startswith('ref:'):
            return head
        ref = head.split(None, 1)[1]
        common = gitdir/(gitdir/'commondir').read_text().strip() if (gitdir/'commondir').is_file() else gitdir
        for base in (gitdir, common.resolve()):
            if (base/ref).is_file():
                return (base/ref).read_text().strip()
        for line in (common.resolve()/'packed-refs').read_text().splitlines():
            if line.endswith(' '+ref):
                return line.split()[0]
    except (OSError, IndexError):
        pass
    return None


def tracked_changes(repo=REPO):
    """Paths with uncommitted changes to tracked files, or None if git cannot say. Read-only: optional locks are
    off, so the index is not refreshed. Called before the audit."""
    try:
        done = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                              text=True, timeout=30, env=os.environ | {'GIT_OPTIONAL_LOCKS': '0'}, check=True)
        return [line[3:] for line in done.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def spread(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    pick = lambda q: values[min(len(values)-1, int(q*len(values)))]
    return {'n': len(values), 'min': values[0], 'p50': pick(.5), 'p90': pick(.9), 'p99': pick(.99), 'max': values[-1],
            'mean': sum(values)/len(values)}


def general(pointer):
    """A JSON pointer with array indices and id-like keys folded, so one line stands for every element."""
    return re.sub(r'/\d+(?::\d+)*(?=/|$)', '/*', pointer)


def armb_idle(armb):
    return not (armb.pending or armb.running is not None or armb.loading)


RECORD_LOCK = threading.Lock()  # one writer of SUMMARY.json at a time: the main flow or the watchdog
RECORD_LOCK_TIMEOUT_S = 5.0  # how long a writer waits for it; a writer that does not get it does NOT write


class RecordBusy(RuntimeError):
    """The record lock could not be taken in time: nothing was written."""


def _replace_record(run_dir, record):
    run_dir = Path(run_dir)
    temporary = run_dir/f'.SUMMARY.json.{threading.get_ident()}.tmp'  # a name of its own per writing thread
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1)+'\n')
    os.replace(temporary, run_dir/'SUMMARY.json')


class ComputeCap(BaseException):
    """A policy computation did not finish within the cap. A BaseException on purpose: the publisher isolates
    ordinary exceptions per game, and this one must reach the runner and end the run."""


class Timed:
    """The policy with every computation timed (wall seconds, in call order). Everything else is the policy's own."""

    def __init__(self, policy, perf, state_of=None):
        self._policy, self._perf, self._state_of, self.seconds, self.states = policy, perf, state_of, [], []

    def __call__(self, feed):
        state = None
        if self._state_of is not None:
            try:
                state = self._state_of(feed)  # sha256 of the decision's state key: what the ledger calls state_sha256
            except Exception:  # noqa: BLE001
                state = None
        begun = self._perf()
        try:
            return self._policy(feed)
        finally:
            self.seconds.append(self._perf()-begun)
            self.states.append(state)

    def __getattr__(self, name):
        return getattr(self._policy, name)


def write_record(run_dir, record):
    """Replace ``SUMMARY.json`` whole (temporary file + rename): a reader sees the previous record or this one.
    Serialised with the watchdog: once the watchdog has taken the lock to write ABORTED it never gives it back
    (the process exits), so no later write of the main flow can cover its record."""
    if not RECORD_LOCK.acquire(timeout=RECORD_LOCK_TIMEOUT_S):
        raise RecordBusy('SUMMARY.json is being written by the watchdog (or the lock is stuck); this record was not written')
    try:
        _replace_record(run_dir, record)
    finally:
        RECORD_LOCK.release()


def aborted_record(record):
    return {'schema': 'pitcheezy-publisher-replay-summary-v3', 'status': 'ABORTED', 'incomplete': True,
            'at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stop': {'reason': record['why'], 'detail': record}}


def mark(run_dir, status, **more):
    """A progress record in ``SUMMARY.json``: never COMPLETED, always ``incomplete: true``. There is no second
    status file that could disagree with it."""
    assert status != 'COMPLETED'
    write_record(run_dir, {'schema': 'pitcheezy-publisher-replay-summary-v3', 'status': status, 'incomplete': True,
                           'at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | more)


def run(*, rows, info, run_dir, game_pk, commit, limits, policy_factory=None, fs_audit=None, schemas=None,
        publish_config=None, perf=time.perf_counter, rss=lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        free_memory=free_memory_bytes, mps=mps_bytes, finalize=True, live_wait_s=None,
        compute_clock=time.perf_counter, keep_moments=False, policy_source=None,
        log=lambda text: print(text, file=sys.stderr, flush=True)):
    """Replay ``rows`` through a publisher that lives entirely in ``run_dir``. Returns the summary (also written).
    ``keep_moments``: gzip copies of each changed live cycle's published files under ``moments/`` (module text).
    ``policy_source``: the name the summary gives the policy ('armb', 'off', 'ledger'); None = by ``policy_factory``.
    Whatever happens inside (a stop condition, any exception, a termination signal) ends in a written summary and
    status. If the process dies without getting there, ``SUMMARY.json`` still says RUNNING and ``incomplete``."""
    from observer_app import publish_audit, publish_schema, publisher
    from observer_app.capture_replay import CaptureReplayTransport, SimClock
    from observer_app.live_armb import LiveArmB, state_key
    from observer_app.live_feed import parse_state
    from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed
    import observer_app.main as app_main

    # REPLAY IS NOT LIVE HERE, on purpose. Live, a request waits LIVE_WAIT_S (1.0 s) for the policy and otherwise
    # shows "computing" until the next poll. Whether a computation beats that wait depends on the machine's speed
    # at that moment, so a replay built that way is not reproducible. In the replay there is a completion barrier
    # BEFORE a cycle produces anything: the state the capture holds at that moment is handed to the policy (the
    # same head start the live route gives the newest snapshot) and the cycle starts only when that computation
    # is done. A computation that exceeds settle_cap_s STOPS the run (policy_compute_cap) before any body, ledger
    # row or file of that cycle exists; no "computing" body is ever published in a replay. What live would have
    # looked like is measured instead: every computation is timed and the ones over the live wait are counted.
    restore_wait = app_main.LIVE_WAIT_S
    live_wait = restore_wait if live_wait_s is None else live_wait_s
    timed = []

    def timed_factory():
        policy = Timed(policy_factory(), compute_clock, lambda feed: hashlib.sha256(state_key(feed).encode()).hexdigest())
        timed.append(policy)
        return policy

    run_dir = Path(run_dir).resolve()
    out, started = run_dir/'out', perf()
    mark(run_dir, 'RUNNING', phase='replay')
    cycles, shas, schema_problems, kinds_seen = [], {}, Counter(), Counter()
    bytes_by_kind, states, codes, recs = {}, Counter(), Counter(), Counter()
    final_published, stop, policy = None, None, {'requested': bool(policy_factory), 'registered_device': limits.get('policy_device')}
    over_interval, mps_peak, free_seen, failure = 0, None, [], None
    # "in a row" counts cycles (MPS: every cycle; free memory: every checked cycle), never individual calls
    measure = {'mps_failed_readings': 0, 'mps_unknown_cycles': 0, 'mps_unknown_cycles_in_a_row': 0, 'free_failures': 0,
               'free_failures_in_a_row': 0, 'free_below_floor_in_a_row': 0, 'free_below_floor_checks': 0}
    cycle_flags = {'mps_unknown': False}
    status, pub, store, audit, transport, armb, config = 'FAILED', None, None, None, None, None, None

    def mps_now():
        """MPS bytes now: a number, None when MPS is not in use, or ``UNKNOWN`` when the reading failed. A failed
        reading is never taken for zero; it marks the current cycle as unknown."""
        nonlocal mps_peak
        try:
            held = mps()
        except MeasureError:
            measure['mps_failed_readings'] += 1
            cycle_flags['mps_unknown'] = True
            return UNKNOWN
        if held is not None:
            mps_peak = max(mps_peak or 0, held)
        return held

    def close_cycle_measurements():
        """Once per cycle: a cycle in which MPS memory could not be read counts toward ``max_measure_failures``."""
        if cycle_flags['mps_unknown']:
            measure['mps_unknown_cycles'] += 1
            measure['mps_unknown_cycles_in_a_row'] += 1
        else:
            measure['mps_unknown_cycles_in_a_row'] = 0
        cycle_flags['mps_unknown'] = False
        if measure['mps_unknown_cycles_in_a_row'] >= limits['max_measure_failures']:
            raise Stop('measurement_failed', {'what': 'mps_memory', 'cycles_in_a_row': measure['mps_unknown_cycles_in_a_row']})

    def free_now(cycle):
        """One free-memory check. Could-not-measure and below-the-floor are different conditions: the first stops
        after ``max_measure_failures`` in a row, the second after ``free_floor_consecutive`` checks in a row (one
        momentary system-wide dip does not end a healthy run)."""
        free = free_memory()
        free_seen.append(free)
        if free is None:  # unknown: neither a breach nor an all-clear, so the below-floor streak is left as it is
            measure['free_failures'] += 1
            measure['free_failures_in_a_row'] += 1
            if measure['free_failures_in_a_row'] >= limits['max_measure_failures']:
                raise Stop('measurement_failed', {'what': 'free_memory', 'checks_in_a_row': measure['free_failures_in_a_row']})
            return None
        measure['free_failures_in_a_row'] = 0
        if free < limits['min_free_bytes_running']:
            measure['free_below_floor_checks'] += 1
            measure['free_below_floor_in_a_row'] += 1
            if measure['free_below_floor_in_a_row'] >= limits['free_floor_consecutive']:
                raise Stop('free_memory_low', {'free_bytes': free, 'cycle': cycle, 'checks_in_a_row': measure['free_below_floor_in_a_row']})
        else:
            measure['free_below_floor_in_a_row'] = 0
        return free

    def guard(stage):
        if info.get('integrity_error'):
            raise Stop('capture_integrity', info['integrity_error'])
        if fs_audit is not None and fs_audit.violations:
            raise Stop('write_process_or_network_outside_the_job_folder', fs_audit.violations[:20])
        if audit is not None and audit.violations:
            raise Stop('invariant', audit.violations[:50])
        if armb is not None and armb.failed:
            raise Stop('policy_compute_failed', {'states': len(armb.failed), 'errors': sorted(set(armb.failed.values()))})
        held = mps_now()
        # an unknown MPS reading is not zero: only what IS known is compared here (RSS alone can already break the
        # cap), and the cycle is counted as unknown (close_cycle_measurements)
        if rss()+(held if isinstance(held, int) else 0) > limits['max_rss_bytes']:
            raise Stop('memory_cap', {'stage': stage, 'rss': rss(), 'mps': held if isinstance(held, int) else None,
                                      'mps_known': held is not UNKNOWN})
        if perf()-started > limits['max_seconds']:
            raise Stop('runtime_cap', {'stage': stage})

    def check_device():
        if not policy_factory or not limits.get('policy_device'):
            return
        seen = policy_devices(armb.policy)
        if seen is not None:
            policy['devices_seen'] = seen
            if seen != [limits['policy_device']]:
                raise Stop('policy_device', {'registered': limits['policy_device'], 'seen': seen})
        elif armb.computed and getattr(armb.policy, 'binds', 0):
            raise Stop('policy_device_unknown', {'registered': limits['policy_device']})

    barrier_s = [0.0]

    def barrier():
        """Before a cycle produces anything: the policy has finished the state the capture holds now, or the run
        stops. Returns the seconds waited."""
        barrier_s[0] = 0.0
        if not policy_factory:
            return
        index = transport.index()
        try:
            body = transport.loaders[index]() if index >= 0 else None
        except Exception:  # noqa: BLE001 - a capture body that fails its check: the guard names it and stops the run
            guard('policy barrier')
            raise
        if body is None or parse_state(body, None).get('status') != 'ready':
            return
        key = state_key(body)
        begun = perf()
        armb.lookup(body, first=True)  # queue it (the head start of the live route); never waits here
        while key not in armb.cache and key not in armb.failed:
            if perf()-begun > limits['settle_cap_s']:
                raise Stop('policy_compute_cap', {'cycle': len(cycles)+1, 'where': 'before the cycle', 'partial_cycle': False,
                                                  'cap_s': limits['settle_cap_s']})
            if perf()-started > limits['max_seconds']:
                raise Stop('runtime_cap', {'stage': 'policy barrier'})
            time.sleep(.002)
        barrier_s[0] = perf()-begun

    checked_times = [0]

    def check_compute_durations(where, partial):
        """Every computation finished since the last check must itself have taken no longer than the cap, however
        short the wait for it was (it may have been running before the wait began)."""
        done = [value for policy_ in timed for value in policy_.seconds]
        fresh, checked_times[0] = done[checked_times[0]:], len(done)
        over = [value for value in fresh if value > limits['settle_cap_s']]
        if over:
            raise Stop('policy_compute_cap', {'cycle': len(cycles)+1, 'where': where, 'partial_cycle': partial,
                                              'cap_s': limits['settle_cap_s'], 'computations_over_the_cap': len(over)})

    def flush_compute_times():
        """Append the computations finished since the last call: index and unrounded wall seconds, no content."""
        done = [value for policy_ in timed for value in policy_.seconds]
        states_done = [value for policy_ in timed for value in policy_.states]
        if len(done) > written_times[0]:
            with open(run_dir/'POLICY-COMPUTE.jsonl', 'a') as handle:
                for number in range(written_times[0], len(done)):
                    handle.write(json.dumps({'index': number, 'seconds': done[number], 'by_cycle': len(cycles)+1,
                                             'state': (states_done[number] or '')[:16] or None})+'\n')
            written_times[0] = len(done)

    written_times = [0]
    published_decisions = set()
    moments = {'kept': 0, 'raw_bytes': 0, 'gzip_bytes': 0}

    def keep_moment(cycle, sequence, present, game_body, facts):
        """This cycle's published files, gzip, under ``moments/<cycle>/`` (the same relative paths plus ``.gz``), and
        one ``moments/INDEX.jsonl`` line: cycle, pitch key, display state, counts and each file's raw sha256. Names
        and counts only, no values of the game."""
        folder = run_dir/'moments'/f'{cycle:05d}'
        files = []
        for name, path in sorted(present.items()):
            data = path.read_bytes()
            packed = gzip.compress(data, compresslevel=6)
            target = folder/(name+'.gz')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(packed)
            files.append({'path': name, 'bytes': len(data), 'gzip_bytes': len(packed), 'sha256': hashlib.sha256(data).hexdigest()})
            moments['raw_bytes'] += len(data)
            moments['gzip_bytes'] += len(packed)
        moments['kept'] += 1
        following = game_body.get('next') or {}
        line = {'cycle': cycle, 'folder': f'moments/{cycle:05d}', 'sequence': sequence, 'simulated': round(clock(), 3),
                'pitch_key': following.get('pitch_key'), 'state': facts['state'], 'reason': facts['reason'],
                'recommendation': facts['recommendation'], 'reference': facts['reference'], 'phase': facts['phase'],
                'pitches': facts['pitches'], 'unconfirmed': facts['unconfirmed'], 'held': facts.get('held', 0),
                'game_sha256': shas.get(game_path), 'files': files}
        with open(run_dir/'moments'/'INDEX.jsonl', 'a') as handle:
            handle.write(json.dumps(line, ensure_ascii=False)+'\n')
        return line['folder']

    def one_cycle(label):
        nonlocal final_published, over_interval
        barrier()
        flush_compute_times()
        check_compute_durations('before the cycle', False)  # nothing of this cycle exists yet
        begun = perf()
        try:
            health = pub.cycle()
        except ComputeCap as capped:  # the in-cycle wait for another state ran out: this cycle is partial
            raise Stop('policy_compute_cap', {'cycle': len(cycles)+1, 'where': 'inside the cycle', 'partial_cycle': True,
                                              'cap_s': limits['settle_cap_s']}) from capped
        finally:
            flush_compute_times()
        check_compute_durations('inside the cycle', True)
        wall_s = perf()-begun
        settle_s = barrier_s[0]
        if not armb_idle(armb):  # nothing may still be computing once a cycle's files are out
            raise Stop('policy_compute_cap', {'cycle': len(cycles)+1, 'where': 'after the cycle', 'partial_cycle': True})
        over_interval += settle_s+wall_s > interval
        written, sizes, present = [], {}, {}
        for path in sorted(out.rglob('*')):
            if path.is_file():
                present[path.relative_to(out).as_posix()] = path
        for name in set(shas)-set(present):
            audit.gone(name)
            del shas[name]
        for name, path in present.items():
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if shas.get(name) == sha:
                continue
            shas[name] = sha
            written.append(name)
            kind = publish_audit.kind_of(name, config['contract_version'])
            body = audit.check(name, data, sha)
            packed = len(gzip.compress(data, compresslevel=6))
            entry = sizes.setdefault(kind or 'unknown', {'files': 0, 'raw': 0, 'gzip': 0})
            entry['files'] += 1
            entry['raw'] += len(data)
            entry['gzip'] += packed
            total = bytes_by_kind.setdefault(kind or 'unknown', {'writes': 0, 'raw_total': 0, 'gzip_total': 0, 'raw_max': 0, 'gzip_max': 0,
                                                                'raw_last': 0, 'gzip_last': 0})
            total['writes'] += 1
            total['raw_total'] += len(data)
            total['gzip_total'] += packed
            total['raw_max'], total['gzip_max'] = max(total['raw_max'], len(data)), max(total['gzip_max'], packed)
            total['raw_last'], total['gzip_last'] = len(data), packed
            if body is not None and kind is not None:
                kinds_seen[kind] += 1
                if schemas is not None:
                    for problem in publish_schema.problems(body, schemas[kind]):
                        schema_problems[kind, general(problem['pointer']), problem['keyword'], problem['seen']] += 1
        facts = None
        if game_path in present:
            game_body = json.loads(present[game_path].read_bytes())
            facts = publish_audit.game_facts(game_body)
            shown = (game_body.get('next') or {}).get('rec') or {}
            if shown.get('candidates') and (shown.get('ledger') or {}).get('decision_id'):
                published_decisions.add(shown['ledger']['decision_id'])
            states[f"{facts['state']}|{facts['reason']}"] += 1
            recs[facts['recommendation'] if facts['recommendation'] == 'published' else f"none|{facts['state']}|{facts['reason']}"] += 1
            if facts['final'] is True and final_published is None:
                final_published = {'simulated': clock(), 'cycle': len(cycles)+1, 'sequence': health['sequence']}
        kept = None
        if keep_moments and facts is not None and game_path in written and isinstance(game_body.get('next'), dict):
            kept = keep_moment(len(cycles)+1, health['sequence'], present, game_body, facts)
        for item in health['unavailable_games']:
            codes[item['code']] += 1
        ledger_now = len(store.served_recommendations(game_pk))
        checked = len(cycles) % limits['free_check_every_cycles'] == 0
        row = {'cycle': len(cycles)+1, 'label': label, 'moment': kept, 'simulated': round(clock(), 3), 'capture_index': transport.index(),
               'sequence': health['sequence'], 'wall_s': round(wall_s, 6), 'settle_s': round(settle_s, 6),
               'files_written': len(written), 'written': written, 'bytes': sizes, 'game': facts,
               'health': {'unavailable_games': [item['code'] for item in health['unavailable_games']],
                          'unavailable_dates': len(health['unavailable_dates']), 'feed_failing_games': len(health['feed_failing_games'])},
               'ledger_rows': ledger_now, 'policy_computations': sum(len(policy_.seconds) for policy_ in timed),
               'rss': rss(), 'mps_bytes': None, 'mps_known': True, 'free_bytes': None, 'free_checked': checked}
        cycles.append(row)
        problem = None
        try:
            held = mps_now()
            row['mps_bytes'], row['mps_known'] = (held if isinstance(held, int) else None), held is not UNKNOWN
            if checked:
                row['free_bytes'] = free_now(row['cycle'])
        except Stop as stopped:
            problem = stopped
        with open(run_dir/'CYCLES.jsonl', 'a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        if problem is not None:
            raise problem
        guard(f'cycle {row["cycle"]}')
        close_cycle_measurements()
        check_device()
        return row

    try:
        for folder in ('empty', 'demo', 'tmp'):
            (run_dir/folder).mkdir(parents=True, exist_ok=True)
        clock = SimClock(rows[0][0])
        transport = CaptureReplayTransport(rows, clock, game_pk=game_pk, schedule_game=c2r._schedule_game)
        store = LiveSnapshotStore(run_dir/'publisher-store.sqlite3')  # the publisher's own: never the capture
        feed = PersistentLiveFeed(transport, store=store, source='replay_poll', delay_s=0, clock=clock, wall=clock, sleep=lambda _: None)
        os.environ['PITCHEEZY_OBSERVER_LIVE_ARMB'] = '1' if policy_factory else '0'
        class ReplayArmB(LiveArmB):
            """In a replay a builder that asks and waits either gets the result or the run ends: it never goes on
            to publish "computing"."""
            def lookup(self, feed, *, queue=True, first=False, wait_s=0.0):
                found = super().lookup(feed, queue=queue, first=first, wait_s=wait_s)
                if found is None and wait_s > 0 and queue and self.policy is not None and self.error is None:
                    raise ComputeCap(state_key(feed))
                return found
        armb = ReplayArmB(timed_factory) if policy_factory else LiveArmB(lambda: None)
        app_main.LIVE_WAIT_S = limits['settle_cap_s']
        config = publisher.load_config(publish_config, env={}, out_dir=str(out), state_path=str(run_dir/'publisher.state.json'),
                                       publish_interval_s=limits['publish_interval_s'], fixed_delay_s=limits['fixed_delay_s'])
        app, pub = publisher.build(config, live_feed=feed, live_policy=armb, feed_dir=run_dir/'empty', watch_dir=run_dir/'empty',
                                   intent_dir=run_dir/'empty', demo_dir=run_dir/'demo', demo_wall=clock, service_wall=clock,
                                   wall=clock, commit=(commit, None))
        ledger_rows = lambda: {row['decision_id']: row for row in store.served_recommendations(game_pk) if row['outcome'] == 'first'}
        def accepted_views(pk):  # every body the run's own pitch ledger accepted, by official key
            views = {}
            for row in store.pitch_events(pk)['events']:
                if row['outcome'] != 'quarantine':
                    views.setdefault((row['at_bat_number'], row['pitch_number']), []).append(row.get('view'))
            return views
        audit = publish_audit.PublishAudit(contract=config['contract_version'], ledger_rows=ledger_rows,
                                           forbidden=[str(run_dir), str(run_dir.parent), str(Path.home()), str(REPO)],
                                           accepted_views=accepted_views)
        game_path = pub.layout['game'].format(game_pk=game_pk)
        interval = config['publish_interval_s']
        guard('start')
        if policy_factory:
            armb.start()
            waited = perf()
            while armb.loading and perf()-waited < limits['policy_load_cap_s']:
                time.sleep(.05)
                guard('policy load')
            policy |= {'load_s': round(perf()-waited, 3), 'status': armb.status(), 'rss_after_load': rss(),
                       'mps_after_load': (lambda held: held if isinstance(held, int) else None)(mps_now())}
            if armb.policy is None:
                raise Stop('policy_unavailable', {'state': armb.status()['state']})
        while not transport.exhausted():
            one_cycle('replay')
            clock.advance(interval)
        for _ in range(limits['tail_cycles']):  # the capture is over: the last body keeps being the answer
            one_cycle('tail')
            clock.advance(interval)
        if config['settled_recheck_s'] is not None:  # one settled recheck: a final file must keep its bytes
            clock.advance(config['settled_recheck_s'])
            one_cycle('settled_recheck')
        status = 'COMPLETED'
    except Stop as stopped:
        status, stop = 'STOPPED', {'reason': stopped.reason, 'detail': stopped.detail}
    except Interrupted as signalled:
        status, stop = 'INTERRUPTED', {'reason': 'signal', 'detail': str(signalled)}
    except BaseException as error:  # noqa: BLE001 - every exit leaves a record; the error is re-raised below
        status, failure = 'FAILED', error
        stop = {'reason': 'exception', 'detail': {'type': type(error).__name__, 'where': 'see the traceback on stderr'}}
    finally:
        app_main.LIVE_WAIT_S = restore_wait
        if pub is not None:
            pub.close()
    violations = [] if audit is None else audit.violations
    exact = [value for policy_ in timed for value in policy_.seconds]  # unrounded: what the counts below compare
    computed = [round(value, 6) for value in exact]  # rounded for the report only
    exact_states = [value for policy_ in timed for value in policy_.states]
    try:  # the computations whose recommendation actually went out in a body, found through the ledger's state hash
        published_states = {row['state_sha256'] for row in store.served_recommendations(game_pk) if row['decision_id'] in published_decisions}
    except Exception:  # noqa: BLE001
        published_states = set()
    shown_exact = [value for value, state in zip(exact, exact_states) if state in published_states]
    try:
        flush_compute_times()
    except Exception:  # noqa: BLE001
        pass
    held = {name: (None if audit is None else not any(v['invariant'] == name for v in violations)) for name in publish_audit.INVARIANTS}
    held['nothing_outside_the_job_folder'] = None if fs_audit is None else not fs_audit.violations  # the caller's outside check can still turn this false
    final_wait = None
    if final_published is not None and info.get('final_fetched') is not None:
        final_wait = round(final_published['simulated']-info['final_fetched'], 3)
    served = sorted(transport.served) if transport is not None else []
    never = [index for index in range(info['versions']) if index not in set(served)]

    def safely(read, default=None):
        try:
            return read()
        except Exception:  # noqa: BLE001 - the summary is written whatever state the run died in
            return default
    summary = {
        'schema': 'pitcheezy-publisher-replay-summary-v3', 'status': status, 'incomplete': status != 'COMPLETED', 'stop': stop,
        'game_pk': game_pk, 'commit': commit,
        'data_role': 'SERVICE_EXPOSED_DEV', 'claim': 'one captured game replayed offline through the publisher; descriptive only',
        'config': {key: limits.get(key) for key in CONFIG_KEYS} | ({} if config is None else {
            'settled_recheck_s': config['settled_recheck_s'], 'contract_version': config['contract_version']}),
        'environment': {'versions': versions(), 'threads': threads()},
        'capture': {'versions': info['versions'], 'first_fetched': info['first_fetched'], 'last_fetched': info['last_fetched'],
                    'span_s': round(info['last_fetched']-info['first_fetched'], 3), 'bodies_loaded': info['loads'],
                    'served_version_indices': served, 'versions_served': len(served),
                    'versions_never_served_indices': never if status == 'COMPLETED' else None,
                    'versions_never_served': len(never) if status == 'COMPLETED' else None,
                    'upstream_requests_answered': None if transport is None else transport.requests,
                    'integrity_error': info.get('integrity_error')},
        'cycles': {'n': len(cycles), 'by_label': dict(Counter(row['label'] for row in cycles)),
                   'wall_s': spread([row['wall_s'] for row in cycles]), 'settle_s': spread([row['settle_s'] for row in cycles]),
                   'files_written': spread([row['files_written'] for row in cycles]),
                   'cycles_where_compute_exceeded_the_publish_interval': over_interval,
                   'last_sequence': cycles[-1]['sequence'] if cycles else None},
        'bytes_by_kind': bytes_by_kind,
        'game_file': {'display_states': dict(states), 'recommendation_by_cycle': dict(recs),
                      'unconfirmed_pitches_per_cycle': spread([row['game']['unconfirmed'] for row in cycles if row['game']]),
                      'cycles_with_an_unconfirmed_pitch': sum(1 for row in cycles if row['game'] and row['game']['unconfirmed']),
                      'held_pitches_per_cycle': spread([row['game'].get('held', 0) for row in cycles if row['game']]),
                      'cycles_with_a_held_pitch': sum(1 for row in cycles if row['game'] and row['game'].get('held')),
                      'corrected_in_last_file': next((row['game']['corrected'] for row in reversed(cycles) if row['game']), None),
                      'pitches_in_last_file': next((row['game']['pitches'] for row in reversed(cycles) if row['game']), None),
                      'paired_past_recommendations_in_last_file': next((row['game']['paired_past_recommendations'] for row in reversed(cycles) if row['game']), None)},
        'health_codes': dict(codes),
        'moments': moments | {'requested': bool(keep_moments), 'index': 'moments/INDEX.jsonl' if moments['kept'] else None,
                              'rule': 'every cycle whose game body changed and holds a pending pitch (next): gzip copies of all '
                                      'published files of that cycle, under moments/<cycle>/'},
        'ledger': {'rows': safely(lambda: len(store.served_recommendations(game_pk))),
                   'first_rows': safely(lambda: sum(row['outcome'] == 'first' for row in store.served_recommendations(game_pk))),
                   'distinct_decisions': safely(lambda: len({row['decision_id'] for row in store.served_recommendations(game_pk)})),
                   'distinct_pitch_keys': safely(lambda: len({row['pitch_key'] for row in store.served_recommendations(game_pk)})),
                   # a row written in a cycle whose file shows no recommendation: expected 0 (the gate runs before the
                   # policy is asked); anything else is reported, cycle by cycle
                   'cycles_adding_rows_without_a_published_recommendation': [
                       row['cycle'] for previous, row in zip([None, *cycles], cycles)
                       if row['game'] and row['game']['recommendation'] != 'published'
                       and row['ledger_rows'] > (previous['ledger_rows'] if previous else 0)]},
        'final': {'feed_final_first_version_fetched': info.get('final_fetched'), 'feed_final_first_version_index': info.get('final_index'),
                  'first_final_version_was_served': None if info.get('final_index') is None else info['final_index'] in set(served),
                  'publish_final': final_published, 'simulated_seconds_from_feed_final_to_publish_final': final_wait,
                  'note': 'first capture version whose status is Final, by its receipt time (served or not); simulated capture '
                          'clock, resolution one publish interval. 미측정 if either side is null.'},
        'schema_validation': {'validated': schemas is not None, 'kinds_seen': dict(kinds_seen),
                              'kinds_never_seen': sorted(set(schemas or {})-set(kinds_seen)),
                              'problems': [{'kind': kind, 'pointer': pointer, 'keyword': keyword, 'seen': seen, 'count': count}
                                           for (kind, pointer, keyword, seen), count in sorted(schema_problems.items())]},
        'invariants': {'held': held, 'all_held': all(value is not False for value in held.values()) and status == 'COMPLETED',
                       'not_applicable': [] if policy_factory else ['recommendation_in_ledger (no policy: nothing is recommended, so nothing is compared)'],
                       'violations': violations[:200], 'violation_count': len(violations),
                       'files_checked': 0 if audit is None else audit.checked},
        'fs_audit': None if fs_audit is None else {'kind': 'tripwire (Python audit events only; native-library I/O is not seen)',
                                                   'violations': fs_audit.violations[:200], 'violation_count': len(fs_audit.violations),
                                                   'files_read_under_watch': dict(sorted(fs_audit.read.items()))},
        'policy': policy | {'source': policy_source or ('armb' if policy_factory else 'off'),
                            # a ledger-backed policy (observer_app.ledger_policy) says how it answered each state; others have no record
                            'ledger': safely(lambda: armb.policy.record() if armb.policy is not None and hasattr(armb.policy, 'record') else None),
                            'final_status': safely(lambda: armb.status()), 'binds': safely(lambda: getattr(armb.policy, 'binds', None)),
                            'failed_states': safely(lambda: len(armb.failed), 0),
                            'computations': len(computed), 'compute_s': spread(computed),
                            'live_wait_s': live_wait,
                            'computations_over_the_live_wait': sum(value > live_wait for value in exact),
                            'all_computations_note': 'every computation, including states the display gate then blocked',
                            'published': {'decisions': len(published_decisions), 'computations': len(shown_exact),
                                          'compute_s': spread([round(value, 6) for value in shown_exact]),
                                          'computations_over_the_live_wait': sum(value > live_wait for value in shown_exact),
                                          'note': 'only computations whose recommendation was published in a body: the set comparable '
                                                  'to "would have shown computing live". An OFFLINE ESTIMATE on this machine during '
                                                  'this replay, not a measurement of the live service.'},
                            'compute_times_file': 'POLICY-COMPUTE.jsonl (index, unrounded seconds, the cycle by which it was done)',
                            'replay_differs_from_live': 'the replay finishes each computation before the cycle produces anything and stops '
                                                        'if one exceeds settle_cap_s; live waits live_wait_s and shows "computing" until '
                                                        'the next poll. computations_over_the_live_wait is how many would have done that.'},
        'memory': {'peak_rss_bytes': rss(), 'source': 'resource.getrusage(RUSAGE_SELF).ru_maxrss (process peak, bytes on macOS)',
                   'mps_driver_allocated_peak_bytes': mps_peak,
                   'mps_note': 'Metal (MPS) memory is unified RAM outside ru_maxrss; sampled once per cycle, so a peak between samples is not seen. null = MPS not in use.',
                   'free_bytes_min_sampled': min((value for value in free_seen if value is not None), default=None),
                   'free_samples': len(free_seen), 'measurement': measure},
        'wall_seconds_total': round(perf()-started, 3),
        'not_measured': ['live polling against the real upstream', 'hosting and upload', 'any other game',
                         'provider versions between the capture polls', 'stale or duplicate responses (the capture stores none)',
                         'end-to-end delay to a viewer', 'I/O done inside native libraries (outside the audit; covered only by the before/after folder listing)'],
    }
    if failure is not None and isinstance(failure, Exception):
        import traceback
        traceback.print_exception(failure, file=sys.stderr)
    if finalize:
        finish(run_dir, summary)
    else:  # the caller still has checks to make: nothing on disk says COMPLETED yet
        mark(run_dir, 'POST_RUN_CHECKS_PENDING', replay_status=status, cycles=len(cycles))
    log(f"{status}: {len(cycles)} cycles")
    if failure is not None and not isinstance(failure, Exception):
        raise failure  # KeyboardInterrupt / SystemExit go on after the record is written
    return summary


def finish(run_dir, summary):
    """The end of a run: ``SUMMARY.md`` first, then ``SUMMARY.json`` with its final status as the LAST write. If
    anything before it fails there is no later write, so the record on disk stays at its last progress state
    (``incomplete: true``); nothing else on disk can contradict it."""
    run_dir = Path(run_dir)
    try:
        text = summary_text(summary)
    except Exception:  # noqa: BLE001 - a run that died before the replay has no numbers to lay out
        text = f"# 발행기 오프라인 재생 요약 — 경기 {summary.get('game_pk')}\n\n- 상태: **{summary['status']}** — **미완료, 부분 결과**\n"
    (run_dir/'SUMMARY.md').write_text(text)
    write_record(run_dir, summary)


def summary_text(s):
    """Short Korean summary. Every number is copied from the summary; a missing one is written 미측정."""
    show = lambda value: '미측정' if value is None else value
    lines = [f"# 발행기 오프라인 재생 요약 — 경기 {s['game_pk']}", '',
             f"- 상태: **{s['status']}**" + (f" (사유: {s['stop']['reason']})" if s['stop'] else '') + (' — **미완료, 부분 결과**' if s['incomplete'] else ''),
             f"- 코드 커밋: `{s['commit']}` / 자료 역할: {s['data_role']}",
             f"- 주기 수: {s['cycles']['n']} ({s['cycles']['by_label']}), 마지막 순번: {show(s['cycles']['last_sequence'])}",
             f"- 주기 실제 소요(초): 중앙값 {show(s['cycles']['wall_s'].get('p50'))}, p99 {show(s['cycles']['wall_s'].get('p99'))}, 최대 {show(s['cycles']['wall_s'].get('max'))}",
             f"- 최대 RSS(바이트): {s['memory']['peak_rss_bytes']} / MPS 최대(바이트): {show(s['memory']['mps_driver_allocated_peak_bytes'])}",
             f"- 정책 장치: 등록 {show(s['policy'].get('registered_device'))}, 실제 {show(s['policy'].get('devices_seen'))}",
             f"- 정책 출처: {show(s['policy'].get('source'))}" + (
                 f" — 장부 행 {show(s['policy']['ledger'].get('rows'))}, 응답 {show(s['policy']['ledger'].get('answers'))}"
                 if isinstance(s['policy'].get('ledger'), dict) else ''),
             f"- 보존한 시점(moments): {show((s.get('moments') or {}).get('kept'))}개" if (s.get('moments') or {}).get('requested') else '- 보존한 시점(moments): 요청 안 함',
             f"- 종료 표시까지(재생 시계 초): {show(s['final']['simulated_seconds_from_feed_final_to_publish_final'])}",
             (f"- 정책 계산: 전체 {show(s['policy'].get('computations'))}회(표시 판정이 막은 상태 포함), 그중 추천이 실제로 발행된 것 {show((s['policy'].get('published') or {}).get('computations'))}회, 발행된 것 중 {show(s['policy'].get('live_wait_s'))}초 초과 {show((s['policy'].get('published') or {}).get('computations_over_the_live_wait'))}회 — 이 기계에서 이 재생 중에 잰 **오프라인 추정**이며 실시간 측정이 아니다"
              if s['policy'].get('source') != 'ledger' else
              f"- 정책 조회(장부, 계산 아님): {show(s['policy'].get('computations'))}회 — 모델은 돌지 않았고 compute_s는 조회 시간이다; 추천은 live 장부의 행 그대로"),
             f"- 추천 원장 행: {show(s['ledger']['rows'])}", '', '## 불변 조건', '']
    lines += [f"- {name}: {'지켜짐' if ok else '미측정' if ok is None else '**깨짐**'}" for name, ok in s['invariants']['held'].items()]
    if 'outside_check' in s:
        lines += [f"- 실행 폴더 밖 전후 비교: {s['outside_check']['overall']} (캡처 원본 {s['outside_check']['capture']}, 워크트리 {s['outside_check']['worktree']}, 정책 폴더 {s['outside_check']['policy']})"]
    lines += ['', '## 파일 크기(바이트, 종류별 최대 raw / gzip)', '']
    lines += [f"- {kind}: {v['raw_max']} / {v['gzip_max']} (쓴 횟수 {v['writes']})" for kind, v in sorted(s['bytes_by_kind'].items())]
    lines += ['', '## 스키마', '',
              f"- 검증함: {s['schema_validation']['validated']}, 본 종류: {s['schema_validation']['kinds_seen']}, 못 본 종류: {s['schema_validation']['kinds_never_seen']}",
              f"- 어긋난 자리 {len(s['schema_validation']['problems'])}곳 (자리·본 타입은 SUMMARY.json)", '',
              '## 말하지 않는 것', ''] + [f'- {item}' for item in s['not_measured']]
    return '\n'.join(lines)+'\n'


def live_policy(output_root, max_games):
    import demo_precompute
    return demo_precompute.LivePolicy(output_root=output_root, max_games=max_games)


def armb_factory(run_dir, style_sha256, style_dir=STYLE_DIR, make_policy=live_policy):
    """The frozen ARM-B live policy with every write redirected into the run folder: the pinned 2026 style
    snapshot is COPIED from T7 into ``<run>/policy-root/style`` (so the policy reads its cache there and never
    writes under T7), and the per-game scratch ledger goes to ``<run>/tmp``. T7 is only read. The two style
    files must have the registered sha256 (``style_sha256``: name -> sha256), checked on the copied bytes."""
    def factory():
        import tempfile
        from observer_app.live_feed import LIVE_CONFIG
        tempfile.tempdir = str(Path(run_dir)/'tmp')
        target = Path(run_dir)/'policy-root/style'
        target.mkdir(parents=True, exist_ok=True)
        for name, sha in style_sha256.items():
            data = (Path(style_dir)/name).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise RuntimeError(f'style file {name} does not have its registered sha256')
            (target/name).write_bytes(data)
        return make_policy(Path(run_dir)/'policy-root', LIVE_CONFIG.get('max_bound_games', 2))
    return factory


def ledger_factory(rows, support, game_pk):
    """The ledger-backed stand-in policy (``observer_app.ledger_policy.LedgerPolicy``) over rows and a support map
    already read and checked by ``execute``: the factory touches no file, folder or volume."""
    def factory():
        from observer_app.ledger_policy import LedgerPolicy
        return LedgerPolicy(rows, support, game_pk=game_pk)
    return factory


def git_porcelain(repo=REPO):
    """``git status --porcelain --ignored`` (tracked, untracked and ignored paths) or None. Optional locks off:
    the index is not refreshed. A child process: only before the tripwire is armed or after it is disarmed."""
    try:
        done = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--ignored'], capture_output=True, text=True,
                              timeout=60, env=os.environ | {'GIT_OPTIONAL_LOCKS': '0'}, check=True)
        return sorted(done.stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        return None


def outside_state(limits, policy, repo=REPO):
    """What is looked at before and after the run, and nothing more:
      capture   sha256 of every file in the ORIGINAL capture folder (exact)
      worktree  ``git status --porcelain --ignored`` of the code checkout (sees new, deleted and modified paths;
                not a change inside an already ignored folder)
      policy    ONLY for a policy run: size and mtime of every entry of each registered policy source folder,
                within one time limit for all of them together; a folder whose listing is not complete is
                ``(None, reason)`` = not checked. A run without the policy does not touch these folders at all.
    The system temporary folder is NOT looked at (other programs use it). None of this can see a write that was
    undone again, or one that kept size and mtime (policy folders)."""
    from observer_app import publish_audit
    try:
        capture = {path.name: sha256_file(path) for path in sorted(Path(limits['capture_original_dir']).iterdir()) if path.is_file()}
    except OSError:
        capture = None
    state = {'capture': capture, 'worktree': git_porcelain(repo), 'policy': {}, 'caches': None}
    if policy:
        state['caches'] = {folder: cache_entries(os.path.expanduser(folder)) for folder in CACHE_DIRS}
        deadline = time.monotonic()+limits['listing_time_limit_s']
        for folder in limits['policy_dirs']:
            state['policy'][folder] = publish_audit.tree_snapshot(folder, deadline=deadline, clock=time.monotonic)
    return state


def cache_entries(folder):
    """``{name: mtime_ns}`` of the TOP-LEVEL entries of a cache folder (nothing below is read), or None if it cannot
    be listed or holds more than ``CACHE_ENTRY_LIMIT`` entries."""
    found = {}
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if len(found) >= CACHE_ENTRY_LIMIT:
                    return None
                found[entry.name] = entry.stat(follow_symlinks=False).st_mtime_ns
    except OSError:
        return None
    return found


def outside_check(before, after, policy):
    """Per scope item ``passed`` / ``failed`` / ``not_checked`` (with the reason), and ``overall``:
    ``failed`` if any item failed; else ``not_checked`` if one could not be looked at; else ``passed``.
    Registered rules: the capture originals and the worktree MUST be checked (a hash or ``git status`` that cannot
    be taken is a failure); a policy folder whose listing timed out or hit a read error is ``not_checked`` and
    does not by itself fail the run; without the policy the policy scope is ``not_applicable``."""
    from observer_app import publish_audit
    result, changes = {'policy': {}}, {}
    if before['capture'] is None or after['capture'] is None:
        result['capture'] = 'failed: the original capture files could not be hashed'
    else:
        differing = sorted(name for name in set(before['capture']) | set(after['capture']) if before['capture'].get(name) != after['capture'].get(name))
        result['capture'] = 'failed' if differing else 'passed'
        if differing:
            changes['capture'] = differing
    if before['worktree'] is None or after['worktree'] is None:
        result['worktree'] = 'failed: git status could not be read'
    else:
        differing = sorted(set(before['worktree']) ^ set(after['worktree']))
        result['worktree'] = 'failed' if differing else 'passed'
        if differing:
            changes['worktree'] = differing[:200]
    if not policy:
        result['policy'] = 'not_applicable'
    for folder, (tree, reason) in (before['policy'] if policy else {}).items():
        later, later_reason = after['policy'].get(folder, (None, 'missing'))
        if tree is None or later is None:
            result['policy'][folder] = f'not_checked: {reason or later_reason}'
            continue
        found = publish_audit.tree_changes(tree, later)
        result['policy'][folder] = 'failed' if found else 'passed'
        if found:
            changes[folder] = found[:200]
    verdicts = [result['capture'], result['worktree'], *(result['policy'].values() if isinstance(result['policy'], dict) else [])]
    result['overall'] = ('failed' if any(v.startswith('failed') for v in verdicts)
                         else 'not_checked' if any(v.startswith('not_checked') for v in verdicts) else 'passed')
    # Home caches (policy runs only): recorded, never a failure. Other programs and the OS (Metal shader cache) write
    # there all the time, so a change cannot be attributed to this run; what is listed is top-level names only.
    result['caches'] = 'not_applicable'
    if policy and before.get('caches') is not None and after.get('caches') is not None:
        result['caches'], touched = {}, {}
        for folder, earlier in before['caches'].items():
            later = after['caches'].get(folder)
            if earlier is None or later is None:
                result['caches'][folder] = 'not_checked'
                continue
            names = sorted(name for name in set(earlier) | set(later) if earlier.get(name) != later.get(name))
            result['caches'][folder] = 'changed_outside: caches' if names else 'unchanged'
            if names:
                touched[folder] = names[:200]
        result['cache_entries_changed'] = touched
    result['changes'] = changes
    result['method'] = ('capture: sha256 of every file; worktree: git status --porcelain --ignored; policy folders (policy runs only): '
                        'size and mtime of every entry within a time limit. Cannot see transient writes, or a same-size same-mtime '
                        'rewrite in a policy folder. The system temporary folder is not checked. Home cache folders (policy runs only): '
                        'top-level names and mtimes, recorded and never failing.')
    return result


def start_watchdog(run_dir, limits, *, peak=None, interval=.5):
    """A hard stop of THIS process only, for when the main thread cannot get to its own checks. Whatever happens
    while it writes its record, the process exits (87)."""
    peak = peak or (lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    begun = time.monotonic()

    def loop():
        while True:
            time.sleep(interval)
            try:
                held = peak()
                elapsed = time.monotonic()-begun
                if held <= limits['max_rss_bytes'] and elapsed <= limits['max_seconds']+60:
                    continue
            except BaseException:  # noqa: BLE001 - a watchdog that cannot measure does not keep the process alive
                held, elapsed = None, time.monotonic()-begun
            try:
                if not RECORD_LOCK.acquire(timeout=RECORD_LOCK_TIMEOUT_S):
                    continue  # the lock is not ours: write nothing (the ``finally`` below still ends the process)
                # held until the process is gone: the main flow cannot write after this
                record = {'peak_rss_bytes': held, 'elapsed_s': round(elapsed, 1),
                          'why': 'runtime_cap_hard' if elapsed > limits['max_seconds']+60 else 'memory_cap' if held is not None else 'watchdog_could_not_measure'}
                (Path(run_dir)/'ABORTED.json').write_text(json.dumps(record))
                _replace_record(run_dir, aborted_record(record))
            finally:
                os._exit(87)
    thread = threading.Thread(target=loop, name='replay-watchdog', daemon=True)
    thread.start()
    return thread


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--game', type=int, required=True)
    parser.add_argument('--capture-db', required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--config', required=True, help='replay_config.json (limits, pace, pins)')
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--policy', choices=('armb', 'off', 'ledger'), required=True)
    parser.add_argument('--ledger-db', help='ledger: COPY of the live run store (served_recommendations) inside the job folder')
    parser.add_argument('--ledger-sha256', help='ledger: registered sha256 of that copy')
    parser.add_argument('--ledger-support', help='ledger: support map file (pitcheezy-ledger-support-v1) inside the job folder')
    parser.add_argument('--ledger-support-sha256', help='ledger: registered sha256 of the support map')
    parser.add_argument('--keep-moments', action='store_true', help='keep gzip copies of the published files of every changed live cycle under <run>/moments/')
    parser.add_argument('--allow-dirty', action='store_true', help='synthetic tests only')
    return parser.parse_args(argv)


def execute(argv=None, *, make_policy=live_policy, style_dir=STYLE_DIR, volume=T7):
    """The whole command. ``make_policy``, ``style_dir`` and ``volume`` exist so a synthetic test can run this
    exact path with a stand-in policy; ``main`` passes none of them."""
    args = parse(argv)
    job, db = Path(args.job_dir).resolve(), Path(args.capture_db).resolve()
    run_dir, style_dir = job/args.run_name, Path(style_dir)
    try:
        if sha256_file(args.config) != args.config_sha256:
            raise Refused('the replay config does not have the registered sha256')
        limits = json.loads(Path(args.config).read_text())
        missing = [key for key in CONFIG_KEYS if key not in limits]
        if missing:
            raise Refused(f'replay config lacks {missing}')
        if limits['fixed_delay_s'] != 0:
            raise Refused('this rehearsal is registered for fixed delay 0 only')
        if not job.is_dir() or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_name) is None or '..' in args.run_name:
            raise Refused('the job folder must exist and the run name must be a plain name')
        if run_dir.exists():
            raise Refused(f'{run_dir} already exists: one run, one new folder')
        if not db.is_relative_to(job) or not db.is_file():
            raise Refused('the capture database must be a COPY inside the job folder')
        if any(Path(str(db)+suffix).exists() and Path(str(db)+suffix).stat().st_size for suffix in ('-wal',)):
            raise Refused('the capture copy has a non-empty WAL')
        before = sha256_file(db)
        if before != args.capture_sha256:
            raise Refused('the capture copy does not have the registered sha256')
        if head_commit() != args.commit:
            raise Refused(f'the code is not at the registered commit ({head_commit()})')
        changed = tracked_changes()
        if changed is None or (changed and not args.allow_dirty):
            raise Refused(f'tracked files differ from the commit: {changed}')
        free, in_process = free_memory_vm_stat(), free_memory_bytes()
        if free is None or in_process is None:
            raise Refused(f'free memory could not be measured (vm_stat {free}, in-process {in_process})')
        if min(free, in_process) < limits['min_free_bytes']:
            raise Refused(f'free memory (vm_stat {free}, in-process {in_process}) is below the registered minimum {limits["min_free_bytes"]}')
        policy_on = args.policy == 'armb'  # OFF: no policy folder, style file or external volume is touched below, at all
        ledger_on, ledger_rows, ledger_support, ledger_input = args.policy == 'ledger', None, None, None
        if ledger_on:  # LEDGER: two frozen inputs inside the job folder, read and checked here, before anything is written
            given = {'--ledger-db': args.ledger_db, '--ledger-sha256': args.ledger_sha256,
                     '--ledger-support': args.ledger_support, '--ledger-support-sha256': args.ledger_support_sha256}
            if not all(given.values()):
                raise Refused(f'--policy ledger needs {[name for name, value in given.items() if not value]}')
            ledger_db, support_path = Path(args.ledger_db).resolve(), Path(args.ledger_support).resolve()
            for name, path, sha in (('ledger copy', ledger_db, args.ledger_sha256), ('support map', support_path, args.ledger_support_sha256)):
                if not path.is_relative_to(job) or not path.is_file():
                    raise Refused(f'the {name} must be a file inside the job folder')
                if sha256_file(path) != sha:
                    raise Refused(f'the {name} does not have the registered sha256')
            from observer_app import ledger_policy  # imports no other module of the app (see its header)
            try:
                ledger_rows, ledger_support = ledger_policy.load_rows(ledger_db, args.game), ledger_policy.load_support(support_path)
            except ledger_policy.LedgerUnusable as error:
                raise Refused(f'ledger input unusable: {error}') from None
            ledger_input = {'db': str(ledger_db), 'db_sha256': args.ledger_sha256, 'rows': len(ledger_rows),
                            'live_policy_identity': next(iter(ledger_rows.values()))['policy_identity'],
                            'support': str(support_path), 'support_sha256': args.ledger_support_sha256,
                            'unsupported_pitchers': {str(k): v for k, v in sorted(ledger_support.items())}}
        elif any((args.ledger_db, args.ledger_sha256, args.ledger_support, args.ledger_support_sha256)):
            raise Refused('--ledger-* options belong to --policy ledger')
        absent = [str(folder) for folder in (limits['capture_original_dir'], *(limits['policy_dirs'] if policy_on else ()))
                  if not Path(folder).is_dir()]
        if absent:
            raise Refused(f'folders registered as unchanged do not exist: {absent}')
        if policy_on:
            if not Path(volume).is_mount():
                raise Refused('the policy needs the T7 volume (read-only use)')
            if not limits['style_sha256'] or not limits['policy_device']:
                raise Refused('a policy run needs the registered style sha256 values and the registered device')
            for name, sha in limits['style_sha256'].items():
                if not (style_dir/name).is_file() or sha256_file(style_dir/name) != sha:
                    raise Refused(f'style file {name} is missing or does not have its registered sha256')
        rows, info = capture_rows(db, args.game, limits['max_body_bytes'])  # reads and verifies every stored version
    except Refused as refused:
        print(f'REFUSED: {refused}', file=sys.stderr)
        return 2
    if 'observer_app.settings' in sys.modules or 'observer_app.main' in sys.modules:
        print('REFUSED: the observer app was imported before its folders could be pinned to the run folder', file=sys.stderr)
        return 2
    for name in CLEARED_ENV:
        os.environ.pop(name, None)
    run_dir.mkdir()
    for name, folder in PINNED_ENV.items():
        (run_dir/folder).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(run_dir/folder)
    mark(run_dir, 'RUNNING', phase='starting')
    summary = {'schema': 'pitcheezy-publisher-replay-summary-v3', 'status': 'FAILED', 'incomplete': True, 'game_pk': args.game,
               'commit': args.commit, 'stop': {'reason': 'failed_before_the_replay', 'detail': None}, 'cycles': {'n': 0, 'by_label': {}, 'wall_s': {}, 'last_sequence': None}}
    fs_audit = None
    try:
        def on_signal(number, frame):
            raise Interrupted(signal.Signals(number).name)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, on_signal)
        start_watchdog(run_dir, limits)
        state_before = outside_state(limits, policy_on)  # under the watchdog; git runs here, before the tripwire
        os.chdir(run_dir)  # a relative path opened for writing resolves inside the run folder
        os.environ['TMPDIR'] = str(run_dir/'tmp')
        from observer_app import publish_audit
        fs_audit = publish_audit.FsAudit([run_dir], watch=[volume] if policy_on else [])
        sys.addaudithook(fs_audit.event)  # tripwire from here on: no write outside the run folder, no process, no socket
        contract = REPO/'apps/observer/publish_contract/v1/schema'
        schemas = {path.name.split('.')[0]: json.loads(path.read_text()) for path in sorted(contract.glob('*.schema.json'))}
        (run_dir/'RUN.json').write_text(json.dumps({
            'argv': sys.argv if argv is None else list(argv), 'commit': args.commit, 'capture_sha256_before': before,
            'config_sha256': args.config_sha256, 'free_memory_bytes_at_start': {'vm_stat': free, 'in_process': in_process},
            'limits': limits, 'policy': args.policy, 'ledger': ledger_input, 'keep_moments': bool(args.keep_moments),
            'python': sys.version, 'versions': versions(), 'threads': threads(),
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=1)+'\n')
        try:
            summary = run(rows=rows, info=info, run_dir=run_dir, game_pk=args.game, commit=args.commit, limits=limits,
                          policy_factory=(armb_factory(run_dir, limits['style_sha256'], style_dir, make_policy) if policy_on else
                                          ledger_factory(ledger_rows, ledger_support, args.game) if ledger_on else None),
                          fs_audit=fs_audit, schemas=schemas, finalize=False, keep_moments=bool(args.keep_moments), policy_source=args.policy)
        finally:
            info['db'].close()
        tripped = len(fs_audit.violations)
        fs_audit.armed = False  # the replay is over: the checks below need git, a child process
        after = sha256_file(db)
        outside = outside_check(state_before, outside_state(limits, policy_on), policy_on)
        summary['capture_copy'] = {'sha256_before': before, 'sha256_after': after, 'unchanged': before == after}
        summary['outside_check'] = outside
        if outside['overall'] == 'failed' or before != after or tripped:
            summary['invariants']['held']['nothing_outside_the_job_folder'] = False
            if summary['status'] == 'COMPLETED':
                summary['status'] = 'FAILED'
                summary['stop'] = {'reason': 'something_outside_the_job_folder_changed_or_could_not_be_checked', 'detail': None}
        summary['incomplete'] = summary['status'] != 'COMPLETED'
        summary['invariants']['all_held'] = (summary['status'] == 'COMPLETED'
                                             and all(value is not False for value in summary['invariants']['held'].values()))
    except Interrupted as signalled:
        summary['status'], summary['incomplete'] = 'INTERRUPTED', True
        summary['stop'] = {'reason': 'signal', 'detail': str(signalled)}
    except BaseException as error:  # noqa: BLE001 - whatever broke, the record below is still written
        import traceback
        traceback.print_exception(error, file=sys.stderr)
        summary['status'], summary['incomplete'] = 'FAILED', True
        summary['stop'] = {'reason': 'exception_outside_the_replay', 'detail': {'type': type(error).__name__}}
    finally:
        if summary['status'] != 'COMPLETED' and isinstance(summary.get('invariants'), dict):
            summary['invariants']['all_held'] = False  # a run that did not complete never reports "all held"
        if fs_audit is not None:
            fs_audit.armed = False
        try:
            finish(run_dir, summary)  # the final SUMMARY.json is the last write; if it fails nothing is written after it
        except BaseException:  # noqa: BLE001 - the record on disk stays at its last progress state (incomplete)
            import traceback
            traceback.print_exc(file=sys.stderr)
            summary['status'] = 'FAILED'
    return 0 if summary['status'] == 'COMPLETED' else 3


def main(argv=None):
    return execute(argv)


if __name__ == '__main__':
    sys.exit(main())
