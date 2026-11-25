"""Pre-pitch pitch-type prediction (EXP-SVC-P9-001 W4a): an optional ``next.prediction`` card, computed OFF the
request path (plan docs/pm/research/svc-model-swap-2026-11-15.md 4-3..4-5).

Structure (4-3):
  1. ``PredictionWorker`` runs the predictor in a SEPARATE PROCESS (``multiprocessing`` spawn context). The parent
     hands it ``(pitch_key, inputs)`` with ``submit`` (a non-blocking queue put) and never waits for it.
  2. The worker answers ``{pitch_key, card, computed_at, elapsed_s}`` (or an error code) on a result queue; a small
     receiver thread in the parent puts the card in a dict guarded by a lock.
  3. ``card_for(pitch_key)`` is the only thing the fan route calls (``service_extras``): one dict lookup. It returns
     None when there is no card for that key, when the card arrived more than ``max_s`` after its submit (2.0 s,
     owner 2026-11-15), when the key does not match, or when the worker is dead/stopped.
  4. A worker that dies or hangs changes nothing else: the body and the recommendation are built and published as
     before, only without the card. Its state stays in ``health()`` (operator view, never on a fan payload).

The predictor is injected: ``factory`` is a top-level callable or a ``'module:attr'`` path to one, called once in
the worker with ``factory_kwargs``; what it returns is called as ``predictor(pitch_key, inputs)`` and must answer
``{'pitch_key': <the pitch it predicted>, 'probabilities': [...9 in CLASSES order], 'fallback': bool}``. The worker
compares the answered ``pitch_key`` with the submitted one and keeps no card when they differ (code review P1-3).
``pitcheezy.svc_p9.worker_predictor:make`` is the real factory (W1 + W2).

Code review fixes (2026-11-16):
  * P1-2 stale card: a card is shown only while it belongs to the LATEST submission of its pitch_key; an error or an
    omission for that key removes the card (a pickoff/steal after the card was computed cannot leave it on screen).
    Between a resubmit and its answer the card is absent (a short gap rather than a possibly stale card).
  * P1-4 back-pressure: at most ONE request is in flight. ``submit`` only overwrites a one-entry slot (latest wins);
    a sender thread hands the slot to the child when the previous answer is back. If an answer is missing for more
    than ``STALL_S`` the worker is 'stalled': nothing more is sent and no card is shown until an answer arrives.
  * Parent death: the child polls ``os.getppid()`` and exits when its parent is gone (no orphan).
  * A dead worker is NOT restarted: the card is simply omitted for the rest of the run (status 'dead' in health).
  * App wiring (start/submit/stop from the app or publisher) is not built here; when it is, the spawning module must
    keep its side effects under ``if __name__ == '__main__'`` (spawn re-imports the parent's main module).

The card (4-4) is a pitch-type prediction only: the top three of the nine groups as integer percentages, the model
name and fit date, the fallback flag and one fixed sentence. No rank, target, zone, candidates or location."""
from __future__ import annotations

import importlib
import logging
import math
import multiprocessing
import os
import queue
import sys
import threading
import time

LOGGER = logging.getLogger(__name__)

CLASSES = ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'OT')
GROUP_LABELS = {'FF': '포심', 'SI': '싱커', 'FC': '커터', 'SL': '슬라이더', 'ST': '스위퍼', 'CU': '커브', 'CH': '체인지업',
                'FS': '스플리터', 'OT': '기타'}
KIND = 'pitch_type_prediction'
CHIP = '예측 · 추천과 별개'
MODEL = {'name': 'B_PSR9N6', 'fit_as_of': '2026-08-31'}
NOTE = ('예측 구종은 "이 투수가 다음에 무엇을 던질까"를 맞히는 별도 모델의 값으로, 추천 정책과 따로 계산되며 '
        '추천을 바꾸지 않습니다.')
TOP = 3
MAX_S = 2.0          # owner 2026-11-15: a card that arrives later than this after its submit is not shown
MAX_THREADS = 2      # PM assumption (config ``architecture.threads``): torch CPU threads in the worker
CACHE_SIZE = 64      # cards kept (only the pending pitch's is ever asked for)
# Omission reason codes (internal only: ``health()``), counted by ``card_for``.
OMIT_ABSENT, OMIT_LATE, OMIT_MISMATCH, OMIT_DEAD = 'absent', 'late', 'key_mismatch', 'worker_unavailable'
OMIT_STALE, OMIT_STALLED = 'superseded', 'worker_stalled'
STALL_S = 10.0       # PM assumption (5 x max_s): an answer missing this long marks the worker stalled
PARENT_POLL_S = 1.0  # child: how often it checks that its parent is alive


# ---- the card ---------------------------------------------------------------------------------------------------
def shares(probabilities):
    """Integer percentages of the nine groups that add up to exactly 100 (largest remainder), so any top three add up
    to at most 100. Raises ValueError for anything that is not nine finite non-negative numbers with a positive sum."""
    values = [float(p) for p in probabilities]
    if len(values) != len(CLASSES) or not all(math.isfinite(v) and v >= 0 for v in values) or sum(values) <= 0:
        raise ValueError('not nine finite non-negative probabilities')
    total = sum(values)
    exact = [100*v/total for v in values]
    floors = [int(math.floor(x)) for x in exact]
    for index in sorted(range(len(exact)), key=lambda i: (floors[i]-exact[i], i))[:100-sum(floors)]:
        floors[index] += 1
    return floors


def build_card(pitch_key, probabilities, fallback):
    """The public card for one pending pitch."""
    if not isinstance(pitch_key, str) or not pitch_key:
        raise ValueError('no pitch key')
    if type(fallback) is not bool:
        raise ValueError('fallback is not a boolean')
    percent = shares(probabilities)
    order = sorted(range(len(CLASSES)), key=lambda i: (-float(probabilities[i]), i))[:TOP]
    return {'kind': KIND, 'chip': CHIP, 'pitch_key': pitch_key,
            'items': [{'pitch_group': CLASSES[i], 'pitch_label': GROUP_LABELS[CLASSES[i]], 'share_pct': percent[i]} for i in order],
            'model': dict(MODEL) | {'fallback': fallback}, 'note': NOTE}



def _resolve(factory):
    if callable(factory):
        return factory
    module, _, attr = str(factory).partition(':')
    target = importlib.import_module(module)
    for part in attr.split('.'):
        target = getattr(target, part)
    return target


# ---- the worker process -----------------------------------------------------------------------------------------
def _worker_main(requests, results, factory, factory_kwargs, threads, parent_pid=None):
    """Runs in the child. Receives at most one request at a time (the parent's sender keeps one in flight)."""
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)
    parent_pid = parent_pid or os.getppid()
    try:
        predictor = _resolve(factory)(**(factory_kwargs or {}))
        if 'torch' in sys.modules:  # only when the predictor brought torch in; never imported here otherwise
            sys.modules['torch'].set_num_threads(threads)
    except Exception as exc:  # noqa: BLE001
        results.put({'error': 'factory_failed', 'detail': type(exc).__name__})
        return
    results.put({'ready': True})
    while True:
        try:
            item = requests.get(timeout=PARENT_POLL_S)
        except queue.Empty:
            if os.getppid() != parent_pid:
                return  # the parent is gone: do not linger as an orphan
            continue
        except (EOFError, OSError):
            return
        if item is None:
            return
        pitch_key, inputs, token = item
        started = time.monotonic()
        try:
            out = predictor(pitch_key, inputs)
            if not isinstance(out, dict) or out.get('pitch_key') != pitch_key:
                results.put({'pitch_key': pitch_key, 'token': token, 'error': 'key_mismatch', 'detail': 'answer_for_another_pitch'})
                continue
            card = build_card(pitch_key, out.get('probabilities'), out.get('fallback'))
        except Exception as exc:  # noqa: BLE001 - reported, the worker keeps serving
            results.put({'pitch_key': pitch_key, 'token': token, 'error': 'predictor_failed', 'detail': type(exc).__name__})
            continue
        results.put({'pitch_key': pitch_key, 'token': token, 'card': card, 'computed_at': time.time(),
                     'compute_s': time.monotonic()-started})


class PredictionWorker:
    """Parent side: submit (a one-entry slot), send (one in flight), receive, cache. ``start()`` spawns the child."""

    def __init__(self, factory, factory_kwargs=None, *, max_s=MAX_S, threads=MAX_THREADS, clock=time.monotonic, stall_s=STALL_S):
        self.factory, self.factory_kwargs = factory, dict(factory_kwargs or {})
        self.max_s, self.threads, self._clock = float(max_s), max(1, min(int(threads), MAX_THREADS)), clock
        self.stall_s = float(stall_s)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._cards = {}        # pitch_key -> {'pitch_key', 'card', 'token', 'computed_at', 'elapsed_s'}
        self._latest = {}       # pitch_key -> the latest submitted token for it
        self._submitted = {}    # token -> (pitch_key, submit time)
        self._slot = None       # (pitch_key, inputs, token): the newest submission not yet sent
        self._inflight = None   # (token, pitch_key, sent time)
        self._token = 0
        self._status, self._last_error = 'not_started', None
        self._counts = {'submitted': 0, 'sent': 0, 'superseded': 0, 'results': 0, 'errors': 0, 'late': 0, 'stalls': 0}
        self._omitted = {}
        self._process = self._requests = self._results = self._receiver = self._sender = None
        self._stop = threading.Event()

    # ---- lifecycle -------------------------------------------------------------------------------------------------
    def start(self):
        context = multiprocessing.get_context('spawn')
        self._requests, self._results = context.Queue(), context.Queue()
        # a 'module:attr' path or a top-level callable (pickled by name for the spawned child)
        self._process = context.Process(target=_worker_main, name='prepitch-prediction', daemon=True,
                                        args=(self._requests, self._results, self.factory, self.factory_kwargs, self.threads, os.getpid()))
        self._process.start()
        with self._lock:
            self._status = 'starting'
        self._receiver = threading.Thread(target=self._receive, name='prepitch-prediction-receiver', daemon=True)
        self._sender = threading.Thread(target=self._send, name='prepitch-prediction-sender', daemon=True)
        self._receiver.start(); self._sender.start()
        return self

    def stop(self, timeout=2.0):
        self._stop.set(); self._wake.set()
        with self._lock:
            self._status, self._cards, self._slot = 'stopped', {}, None
        try:
            if self._requests is not None:
                self._requests.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        if self._process is not None:
            self._process.join(timeout)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout)
        for thread in (self._receiver, self._sender):
            if thread is not None:
                thread.join(timeout)
        for q in (self._requests, self._results):
            if q is not None:
                q.cancel_join_thread()
                q.close()

    @property
    def process(self):
        return self._process

    # ---- the sender thread: one request in flight --------------------------------------------------------------------
    def _send(self):
        while not self._stop.is_set():
            self._wake.wait(0.2)
            self._wake.clear()
            item = None
            with self._lock:
                now = self._clock()
                if self._inflight is not None and now-self._inflight[2] > self.stall_s and self._status == 'running':
                    self._status, self._last_error = 'stalled', 'no_answer_within_stall_s'
                    self._counts['stalls'] += 1
                    self._cards = {}
                if self._inflight is None and self._slot is not None and self._status in ('starting', 'running'):
                    item, self._slot = self._slot, None
                    self._inflight = (item[2], item[0], now)
                    self._counts['sent'] += 1
            if item is not None:
                try:
                    self._requests.put(item)
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._inflight, self._last_error = None, f'send_failed:{type(exc).__name__}'

    # ---- the receiver thread ---------------------------------------------------------------------------------------
    def _receive(self):
        while not self._stop.is_set():
            try:
                message = self._results.get(timeout=0.2)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    self._dead('worker_exited')
                    return
                continue
            except (EOFError, OSError, ValueError):
                self._dead('result_channel_closed')
                return
            self._take(message)
            self._wake.set()

    def _dead(self, code):
        with self._lock:
            if self._status != 'stopped':
                self._status, self._last_error, self._cards, self._slot = 'dead', code, {}, None

    def _take(self, message):
        now = self._clock()
        with self._lock:
            if message.get('ready'):
                if self._status == 'starting':
                    self._status = 'running'
                return
            if message.get('error') and 'pitch_key' not in message:  # the factory failed: nothing will come
                self._status, self._last_error = 'dead', f"{message['error']}:{message.get('detail')}"
                return
            token = message.get('token')
            if self._inflight is not None and self._inflight[0] == token:
                self._inflight = None
                if self._status == 'stalled':
                    self._status = 'running'
            sent = self._submitted.pop(token, None)
            if message.get('error'):
                self._counts['errors'] += 1
                self._last_error = f"{message['error']}:{message.get('detail')}"
                key = sent[0] if sent else message.get('pitch_key')
                self._cards.pop(key, None)  # P1-2: an error removes the card (second guard; the latest-token binding in card_for is the tested one)
                return
            if sent is None or sent[0] != message.get('pitch_key') or (message.get('card') or {}).get('pitch_key') != sent[0]:
                self._counts['errors'] += 1
                self._last_error = 'unmatched_result'
                return
            elapsed = now-sent[1]
            self._counts['results'] += 1
            if elapsed > self.max_s:
                self._counts['late'] += 1
            self._cards[sent[0]] = {'pitch_key': sent[0], 'card': message['card'], 'token': token,
                                    'computed_at': message.get('computed_at'), 'elapsed_s': elapsed}
            while len(self._cards) > CACHE_SIZE:
                self._cards.pop(next(iter(self._cards)))

    # ---- what the parent calls -------------------------------------------------------------------------------------
    def submit(self, pitch_key, inputs):
        """Puts one pending pitch in the one-entry slot (the newest wins); never blocks and never raises; no queue
        grows. False when it was not taken (worker not running or stalled)."""
        try:
            with self._lock:
                if self._status not in ('starting', 'running') or not isinstance(pitch_key, str):
                    return False
                self._token += 1
                token = self._token
                self._submitted[token] = (pitch_key, self._clock())
                while len(self._submitted) > CACHE_SIZE:
                    self._submitted.pop(next(iter(self._submitted)))
                if self._slot is not None:
                    self._counts['superseded'] += 1
                self._slot = (pitch_key, inputs, token)
                self._latest[pitch_key] = token
                while len(self._latest) > CACHE_SIZE:
                    self._latest.pop(next(iter(self._latest)))
                self._counts['submitted'] += 1
            self._wake.set()
            return True
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = f'submit_failed:{type(exc).__name__}'
            return False

    def card_for(self, pitch_key, max_s=None):
        """The card for ``pitch_key`` if it answers the LATEST submission of that pitch in time, else None. A dict lookup
        under a lock: no compute, no I/O. ``max_s`` (the configured switch value) can only make the limit stricter."""
        limit = self.max_s if max_s is None else min(self.max_s, float(max_s))
        with self._lock:
            if self._status == 'stalled':
                reason, found = OMIT_STALLED, None
            elif self._status not in ('starting', 'running'):
                reason, found = OMIT_DEAD, None
            else:
                entry = self._cards.get(pitch_key)
                if entry is None:
                    reason, found = OMIT_ABSENT, None
                elif entry['pitch_key'] != pitch_key or entry['card'].get('pitch_key') != pitch_key:
                    reason, found = OMIT_MISMATCH, None
                elif entry['token'] != self._latest.get(pitch_key):
                    reason, found = OMIT_STALE, None
                elif not entry['elapsed_s'] <= limit:
                    reason, found = OMIT_LATE, None
                else:
                    return entry['card']
            self._omitted[reason] = self._omitted.get(reason, 0)+1
            return found

    def omit(self, reason):
        """Counts an omission decided outside this module (``service_extras``: state not eligible, a reference card)."""
        with self._lock:
            self._omitted[reason] = self._omitted.get(reason, 0)+1

    def health(self):
        """Operator view only (``/api/health``): state, last error code and counts; no card, no input."""
        with self._lock:
            return {'status': self._status, 'last_error': self._last_error, 'max_s': self.max_s, 'threads': self.threads,
                    'stall_s': self.stall_s, 'inflight': self._inflight is not None, 'counts': dict(self._counts),
                    'cached': len(self._cards), 'omitted': dict(self._omitted)}
