"""Checks a rehearsal makes on what the publisher wrote, independent of the publisher's own guards.

``PublishAudit`` looks at every published file as bytes, cycle after cycle, and records each broken invariant as
``{'invariant', 'path', 'pointer', 'detail'}``. ``detail`` names keys, rules and types only: never a value from a
body. ``FsAudit`` classifies the interpreter's audit events (``sys.addaudithook``) so a run can show that it wrote
nothing outside its own folder, started no process and opened no socket.

Invariants (names are what the summary reports):
  unconfirmed_pitch_blank   a pitch whose receipt is not ``received`` carries no type/speed/location/result
  recommendation_in_ledger  every published recommendation names a decision the ledger holds
  sequence_monotonic        a file's ``publish.sequence`` never goes down, and goes up when its bytes change
  final_stable              a file marked final changes its bytes only when its content changed
  no_private_strings        no ``path`` key, absolute path, demo/operator marker or forbidden string anywhere
  only_public_names         only the six published file names exist
"""
from __future__ import annotations

import errno
import json
import os
import re

from . import publisher

UNCONFIRMED_KEPT = publisher.UNCONFIRMED_KEPT
ABSOLUTE = re.compile(r'^(/|~|[A-Za-z]:\\|file:)|/(Users|Volumes|private|tmp|var|home|opt|etc)/')
INVARIANTS = ('unconfirmed_pitch_blank', 'recommendation_in_ledger', 'sequence_monotonic', 'final_stable',
              'no_private_strings', 'only_public_names')


def kind_of(path, contract='v1'):
    """File kind of a published path (``game``, ``games``, ``dates``, ``setup``, ``manifest``, ``health``) or None."""
    match = re.fullmatch(re.escape(contract)+r'/(?:(manifest|health|dates)\.json|(games)/\d{4}-\d{2}-\d{2}\.json|(game|setup)/\d+\.json)', path)
    return next((group for group in match.groups() if group), None) if match else None


def recommendations(body):
    """(pointer, rec) of every recommendation a game body shows: the next pitch's and each past pitch's."""
    found = []
    rec = (body.get('next') or {}).get('rec')
    if isinstance(rec, dict) and rec.get('candidates'):
        found.append(('/next/rec', rec))
    for index, pitch in enumerate(body.get('pitches') or []):
        rec = pitch.get('rec')
        if isinstance(rec, dict) and rec.get('candidates') and rec.get('provenance') == 'captured_live':
            found.append((f'/pitches/{index}/rec', rec))
    return found


def game_facts(body):
    """What a cycle's game file says, as counts and codes (no values of the game)."""
    state = body.get('display_state') or {}
    pitches = body.get('pitches') or []
    shown = bool(((body.get('next') or {}).get('rec') or {}).get('candidates'))
    return {'pitches': len(pitches),
            'unconfirmed': sum((p.get('receipt') or {}).get('status') != 'received' for p in pitches if 'receipt' in p),
            'corrected': sum(bool((p.get('receipt') or {}).get('corrected')) for p in pitches),
            'paired_past_recommendations': sum(1 for pointer, _ in recommendations(body) if pointer != '/next/rec'),
            'state': state.get('state'), 'reason': state.get('reason'),
            'recommendation': 'published' if shown else 'none',
            'phase': (body.get('current') or {}).get('phase'), 'game_status': (body.get('game') or {}).get('status'),
            'feed_ok': (body.get('feed') or {}).get('ok'), 'final': (body.get('publish') or {}).get('final')}


class PublishAudit:
    def __init__(self, *, contract='v1', forbidden=(), ledger_ids=lambda: set()):
        """``forbidden``: strings that must appear in no file (the job folder, the home folder, a token...).
        ``ledger_ids()``: the decision ids the publisher's own ledger holds right now."""
        self.contract, self.forbidden, self.ledger_ids = contract, [text for text in forbidden if text], ledger_ids
        self.violations, self.checked, self.seen = [], 0, {}  # path -> {'sequence', 'sha', 'final', 'digest'}

    def _fail(self, invariant, path, pointer, detail):
        self.violations.append({'invariant': invariant, 'path': path, 'pointer': pointer, 'detail': detail})

    def check(self, path, data, sha):
        """One published file as it is on disk now (``sha``: sha256 of ``data``). Returns the parsed body or None."""
        self.checked += 1
        kind = kind_of(path, self.contract)
        if kind is None:
            self._fail('only_public_names', path, '', 'not one of the published file names')
            return None
        try:
            body = json.loads(data)
            publish = body['publish']
            sequence = publish['sequence']
            if type(sequence) is not int:
                raise TypeError
        except Exception:  # noqa: BLE001
            self._fail('sequence_monotonic', path, '/publish/sequence', 'no readable publish block')
            return None
        text = data.decode('utf-8', 'replace')
        for index, word in enumerate(self.forbidden):
            if word in text:
                self._fail('no_private_strings', path, '', f'forbidden string #{index} present')
        self._walk(path, body, '')
        source = body.get('source')
        if isinstance(source, dict) and kind != 'setup' and (source.get('kind') not in ('archive', 'live') or source.get('session_id') is not None):
            self._fail('no_private_strings', path, '/source', 'not the plain public source')
        previous = self.seen.get(path)
        digest = publisher.game_digest({k: v for k, v in body.items() if k != 'publish'}) if kind == 'game' else None
        if previous is not None:
            if sequence < previous['sequence']:
                self._fail('sequence_monotonic', path, '/publish/sequence', 'sequence went down')
            elif sha != previous['sha'] and sequence == previous['sequence']:
                self._fail('sequence_monotonic', path, '/publish/sequence', 'bytes changed under the same sequence')
            if kind == 'game' and previous['final'] is True and sha != previous['sha'] and digest == previous['digest']:
                self._fail('final_stable', path, '', 'a final file was rewritten although its content did not change')
        self.seen[path] = {'sequence': sequence, 'sha': sha, 'final': publish.get('final'), 'digest': digest}
        if kind == 'game':
            self._game(path, body)
        return body

    def gone(self, path):
        self.seen.pop(path, None)

    def _walk(self, path, value, pointer):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('path', 'token', 'operator', 'session_token'):
                    self._fail('no_private_strings', path, f'{pointer}/{key}', 'a key that names a location or a credential')
                self._walk(path, item, f'{pointer}/{key}')
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._walk(path, item, f'{pointer}/{index}')
        elif isinstance(value, str) and ABSOLUTE.search(value):
            self._fail('no_private_strings', path, pointer, 'a string that looks like an absolute path')

    def _game(self, path, body):
        live = (body.get('source') or {}).get('kind') == 'live'
        for index, pitch in enumerate(body.get('pitches') or []):
            receipt = pitch.get('receipt')
            if live and not isinstance(receipt, dict):
                self._fail('unconfirmed_pitch_blank', path, f'/pitches/{index}/receipt', 'a live pitch without a receipt')
                continue
            if isinstance(receipt, dict) and receipt.get('status') != 'received':
                filled = sorted(key for key, value in (pitch.get('actual') or {}).items()
                                if value is not None and key not in UNCONFIRMED_KEPT)
                if filled:
                    self._fail('unconfirmed_pitch_blank', path, f'/pitches/{index}/actual', 'filled keys: '+','.join(filled))
        if not live:
            return
        found = recommendations(body)
        held = self.ledger_ids() if found else set()
        for pointer, rec in found:
            ledger = rec.get('ledger') or {}
            if ledger.get('status') != 'recorded' or ledger.get('decision_id') not in held:
                self._fail('recommendation_in_ledger', path, pointer, 'no ledger row for this recommendation')
        shown = (body.get('display_state') or {}).get('recommendation_shown')
        if any(pointer == '/next/rec' for pointer, _ in found) and shown is not True:
            self._fail('recommendation_in_ledger', path, '/display_state/recommendation_shown', 'candidates although the state shows none')


WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
MUTATING = {'os.remove': (0,), 'os.rename': (0, 1), 'os.mkdir': (0,), 'os.rmdir': (0,), 'os.truncate': (0,), 'os.symlink': (1,),
            'os.link': (1,), 'os.chmod': (0,), 'os.chown': (0,), 'os.utime': (0,), 'shutil.rmtree': (0,), 'shutil.move': (0, 1),
            'shutil.copyfile': (1,), 'shutil.copymode': (1,), 'shutil.copystat': (1,), 'shutil.copytree': (1,),
            'os.setxattr': (0,), 'os.removexattr': (0,)}
PROCESS = ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.fork', 'os.forkpty', 'os.spawn', 'os.startfile')
NETWORK = ('socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.sendto', 'urllib.Request')


class Forbidden(PermissionError):
    """Raised INTO the operation that tried it: the write, process or connection does not happen."""


class FsAudit:
    """``event`` is a ``sys.addaudithook`` hook. Inside ``allowed`` roots anything may be written. Elsewhere a
    write, delete, rename, a non read-only SQLite open, any new process and any socket use is recorded as a
    violation and refused (``block=True``). Files opened for reading under ``watch`` roots are listed (paths
    only), so the run can state exactly which files of a read-only volume it read."""

    def __init__(self, allowed, *, watch=(), block=True, also_writable=('/dev/null',)):
        self.allowed = [os.path.realpath(root) for root in allowed]
        self.watch = [os.path.realpath(root) for root in watch]
        self.block, self.also = block, set(also_writable)
        self.violations, self.read, self.armed = [], {}, True

    def _inside(self, path, roots):
        try:
            real = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        return any(real == root or real.startswith(root+os.sep) for root in roots)

    def _refuse(self, kind, event, target):
        self.violations.append({'kind': kind, 'event': event, 'target': None if target is None else str(target)})
        if self.block:
            raise Forbidden(errno.EPERM, f'{kind} refused by the rehearsal audit ({event})')

    def _write(self, event, target):
        if isinstance(target, int) or target is None:  # an already open descriptor, or nothing to name
            return
        text = os.fsdecode(target)
        if text in self.also or self._inside(text, self.allowed):
            return
        self._refuse('write_outside', event, text)

    def event(self, name, args):
        if not self.armed:
            return
        if name == 'open':
            target, mode, flags = (list(args)+[None, None, None])[:3]
            writing = (isinstance(flags, int) and flags & WRITE_FLAGS) or (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            if writing:
                self._write(name, target)
            elif isinstance(target, (str, bytes, os.PathLike)) and self._inside(target, self.watch):
                real = os.path.realpath(os.fsdecode(target))
                self.read[real] = self.read.get(real, 0)+1
        elif name in MUTATING:
            for position in MUTATING[name]:
                if position < len(args):
                    self._write(name, args[position])
        elif name == 'sqlite3.connect':
            target = str(args[0]) if args else ''
            path = target[5:].split('?', 1)[0] if target.startswith('file:') else target
            read_only = target.startswith('file:') and 'mode=ro' in target
            if path != ':memory:' and not read_only:
                self._write(name, path)
            elif read_only and self._inside(path, self.watch):
                real = os.path.realpath(path)
                self.read[real] = self.read.get(real, 0)+1
        elif name.startswith(PROCESS):
            self._refuse('process', name, args[0] if args else None)
        elif name.startswith(NETWORK):
            self._refuse('network', name, None)
