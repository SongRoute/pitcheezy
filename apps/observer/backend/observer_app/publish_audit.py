"""Checks a rehearsal makes on what the publisher wrote, independent of the publisher's own guards.

``PublishAudit`` looks at every published file as bytes, cycle after cycle, and records each broken invariant as
``{'invariant', 'path', 'pointer', 'detail'}``. ``detail`` names keys, rules and types only: never a value from a
body. ``FsAudit`` classifies the interpreter's audit events (``sys.addaudithook``). It is a TRIPWIRE, not
isolation: it sees what Python reports (opens, removes, renames, SQLite connects, process and socket calls) and
refuses those outside the allowed folders, but I/O done inside native libraries is invisible to it and a
descriptor-relative path can only be resolved when the event carries the descriptor. ``tree_snapshot`` /
``tree_changes`` give the independent after-the-fact check: what actually changed on disk.

Invariants (names are what the summary reports):
  unconfirmed_pitch_blank   a pitch whose receipt is not ``received`` carries no type/speed/location/result,
                            and its two labels read exactly 확인 중
  recommendation_in_ledger  every recommendation in a live body is the ledger's ``first`` row for its decision:
                            same pitch, same candidates, same policy, same decision time, and the state shows it
  sequence_monotonic        a file's ``publish.sequence`` never goes down (also across a delete and a return),
                            and goes up when its bytes change
  final_stable              a file marked final changes its bytes only when its content changed
  no_private_strings        no ``path`` key, absolute path, demo/operator marker or forbidden string anywhere
  only_public_names         only the six published file names exist
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import re
from urllib.parse import parse_qs, unquote, urlsplit

from . import publisher

UNCONFIRMED_KEPT = publisher.UNCONFIRMED_KEPT
UNCONFIRMED_LABEL = '확인 중'  # service_api.UNCONFIRMED_LABEL, restated: this module checks the builders, it does not ask them
CANDIDATE_FIELDS = ('rank', 'pitch_type', 'pitch_label', 'zone_id', 'zone_label', 'target')
ABSOLUTE = re.compile(r'^(/|~|[A-Za-z]:\\|file:)|/(Users|Volumes|private|tmp|var|home|opt|etc)/')
INVARIANTS = ('unconfirmed_pitch_blank', 'recommendation_in_ledger', 'sequence_monotonic', 'final_stable',
              'no_private_strings', 'only_public_names')


def kind_of(path, contract='v1'):
    """File kind of a published path (``game``, ``games``, ``dates``, ``setup``, ``manifest``, ``health``) or None."""
    match = re.fullmatch(re.escape(contract)+r'/(?:(manifest|health|dates)\.json|(games)/\d{4}-\d{2}-\d{2}\.json|(game|setup)/\d+\.json)', path)
    return next((group for group in match.groups() if group), None) if match else None


def recommendations(body):
    """(pointer, rec, pitch key) of every recommendation a game body shows, whatever it says its provenance is:
    the next pitch's and each past pitch's. A rec without candidates recommends nothing and is not listed."""
    found = []
    following = body.get('next') or {}
    rec = following.get('rec')
    if isinstance(rec, dict) and rec.get('candidates'):
        found.append(('/next/rec', rec, following.get('pitch_key')))
    for index, pitch in enumerate(body.get('pitches') or []):
        rec = pitch.get('rec')
        if isinstance(rec, dict) and rec.get('candidates'):
            found.append((f'/pitches/{index}/rec', rec, pitch.get('key')))
    return found


def shown_candidates(candidates):
    """The projection ``main.service_rec`` / ``rec_pairing._rec`` publish of a stored candidate list."""
    return [{key: c.get(key, i+1) if key == 'rank' else c.get(key) for key in CANDIDATE_FIELDS}
            | {'detail': {'probability': (c.get('detail') or {}).get('probability')}} for i, c in enumerate(candidates or [])]


def game_facts(body):
    """What a cycle's game file says, as counts and codes (no values of the game)."""
    state = body.get('display_state') or {}
    pitches = body.get('pitches') or []
    shown = bool(((body.get('next') or {}).get('rec') or {}).get('candidates'))
    return {'pitches': len(pitches),
            'unconfirmed': sum((p.get('receipt') or {}).get('status') != 'received' for p in pitches if 'receipt' in p),
            'corrected': sum(bool((p.get('receipt') or {}).get('corrected')) for p in pitches),
            'paired_past_recommendations': sum(1 for pointer, *_ in recommendations(body) if pointer != '/next/rec'),
            'state': state.get('state'), 'reason': state.get('reason'),
            'recommendation': 'published' if shown else 'none',
            'phase': (body.get('current') or {}).get('phase'), 'game_status': (body.get('game') or {}).get('status'),
            'feed_ok': (body.get('feed') or {}).get('ok'), 'final': (body.get('publish') or {}).get('final')}


class PublishAudit:
    def __init__(self, *, contract='v1', forbidden=(), ledger_rows=lambda: {}):
        """``forbidden``: strings that must appear in no file (the job folder, the home folder, a token...).
        ``ledger_rows()``: ``{decision_id: row}`` of the ``first`` rows the publisher's own ledger holds right now
        (``LiveSnapshotStore.served_recommendations`` rows)."""
        self.contract, self.forbidden, self.ledger_rows = contract, [text for text in forbidden if text], ledger_rows
        self.violations, self.checked, self.seen = [], 0, {}  # path -> {'sequence', 'sha', 'final', 'digest'}
        self.high_water = {}  # path -> highest sequence ever seen there, kept when the file is deleted

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
        if previous is None and path in self.high_water and sequence <= self.high_water[path]:
            self._fail('sequence_monotonic', path, '/publish/sequence', 'a file that returned after a delete does not carry a higher sequence')
        self.high_water[path] = max(sequence, self.high_water.get(path, sequence))
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
                actual = pitch.get('actual') or {}
                filled = sorted(key for key, value in actual.items() if value is not None and key not in UNCONFIRMED_KEPT)
                if filled:
                    self._fail('unconfirmed_pitch_blank', path, f'/pitches/{index}/actual', 'filled keys: '+','.join(filled))
                wrong = sorted(key for key in UNCONFIRMED_KEPT if actual.get(key) != UNCONFIRMED_LABEL)
                if wrong:
                    self._fail('unconfirmed_pitch_blank', path, f'/pitches/{index}/actual', 'labels that do not read 확인 중: '+','.join(wrong))
        if not live:
            return
        found = recommendations(body)
        held = self.ledger_rows() if found else {}
        for pointer, rec, pitch_key in found:
            ledger = rec.get('ledger') or {}
            row = held.get(ledger.get('decision_id'))
            if rec.get('provenance') != 'captured_live' or ledger.get('status') != 'recorded' or row is None:
                self._fail('recommendation_in_ledger', path, pointer, 'no ledger row for this recommendation')
                continue
            displayed = row.get('displayed') or {}
            differs = [name for name, same in (
                ('pitch_key', row.get('pitch_key') == pitch_key),
                ('candidates', rec.get('candidates') == shown_candidates((displayed.get('recommendation') or {}).get('candidates'))),
                ('decided', ledger.get('decided') == row.get('decided')),
                ('policy', str(row.get('policy_identity') or '').startswith(str((rec.get('policy') or {}).get('identity') or '\0'))),
                ('stored_status', displayed.get('status') == 'ready' and row.get('outcome', 'first') == 'first')) if not same]
            if differs:
                self._fail('recommendation_in_ledger', path, pointer, 'differs from its ledger row in: '+','.join(differs))
        state = body.get('display_state') or {}
        if any(pointer == '/next/rec' for pointer, *_ in found) and (
                state.get('recommendation_shown') is not True or state.get('state') not in ('recommended', 'feed_delayed', 'disconnected')):
            self._fail('recommendation_in_ledger', path, '/display_state', 'candidates although the state shows none')


WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
# event -> ((index of the path argument, index of its dir_fd argument or None), ...); every listed path is changed
MUTATING = {'os.remove': ((0, 1),), 'os.rmdir': ((0, 1),), 'os.mkdir': ((0, 2),), 'os.rename': ((0, 2), (1, 3)),
            'os.truncate': ((0, None),), 'os.symlink': ((1, 2),), 'os.link': ((1, 3),), 'os.chmod': ((0, 2),),
            'os.chown': ((0, 3),), 'os.utime': ((0, 3),), 'os.setxattr': ((0, None),), 'os.removexattr': ((0, None),),
            'shutil.rmtree': ((0, 1),), 'shutil.move': ((0, None), (1, None)), 'shutil.copyfile': ((1, None),),
            'shutil.copymode': ((1, None),), 'shutil.copystat': ((1, None),), 'shutil.copytree': ((1, None),)}
PROCESS = ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.fork', 'os.forkpty', 'os.spawn', 'os.startfile')
NETWORK = ('socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.sendto', 'urllib.Request')
F_GETPATH = getattr(fcntl, 'F_GETPATH', 50)  # macOS: the path a descriptor was opened on


def fd_path(fd):
    """The path behind an open descriptor, or None if the system cannot say (a pipe, a socket, another OS)."""
    try:
        return fcntl.fcntl(fd, F_GETPATH, bytes(1024)).split(b'\0', 1)[0].decode() or None
    except (OSError, ValueError, TypeError):
        return None


def sqlite_target(database):
    """(file path or None for a private in-memory database, opened read-only?) of a ``sqlite3.connect`` target.
    Read-only only when the URI's own ``mode`` parameter is exactly ``ro`` (``immutable=1`` alone is not taken as
    proof); a URI is recognised by its ``file:`` scheme, as SQLite does when ``uri=True``."""
    text = os.fsdecode(database) if isinstance(database, (bytes, os.PathLike)) else str(database)
    if text == ':memory:' or text == '':
        return None, False
    if not text.startswith('file:'):
        return text, False
    parts = urlsplit(text)
    query = parse_qs(parts.query)
    path = unquote(parts.path)
    if path == ':memory:' or query.get('mode') == ['memory']:
        return None, False
    return path, query.get('mode') == ['ro']


class Forbidden(PermissionError):
    """Raised INTO the operation that tried it: the write, process or connection does not happen."""


class FsAudit:
    """``event`` is a ``sys.addaudithook`` hook: a tripwire (see the module text). Inside ``allowed`` roots
    anything may be written. Elsewhere a write, delete, rename, a SQLite open that is not read-only, any new
    process and any socket use is recorded as a violation and refused (``block=True``). Files opened for reading
    under ``watch`` roots are listed (paths only).

    Paths: a relative path is resolved against the directory descriptor the event carries (remove, rmdir, mkdir,
    rename... do; ``open`` does not, so a relative ``open`` is resolved against the working directory: run with the
    working directory inside the allowed folder). A write through an already open descriptor is judged by the
    path that descriptor was opened on; one whose path cannot be determined is refused unless it is 0, 1 or 2."""

    def __init__(self, allowed, *, watch=(), block=True, also_writable=('/dev/null',), fd_path=fd_path):
        self.allowed = [os.path.realpath(root) for root in allowed]
        self.watch = [os.path.realpath(root) for root in watch]
        self.block, self.also, self.fd_path = block, set(also_writable), fd_path
        self.violations, self.read, self.armed = [], {}, True

    def _real(self, path, dir_fd=None):
        text = os.fsdecode(path)
        if not os.path.isabs(text) and isinstance(dir_fd, int) and dir_fd >= 0:
            base = self.fd_path(dir_fd)
            if base is None:
                return None  # relative to a descriptor nobody can name: cannot be shown to be inside
            text = os.path.join(base, text)
        return os.path.realpath(text)

    def _inside(self, real, roots):
        return real is not None and any(real == root or real.startswith(root+os.sep) for root in roots)

    def _refuse(self, kind, event, target):
        self.violations.append({'kind': kind, 'event': event, 'target': None if target is None else str(target)})
        if self.block:
            raise Forbidden(errno.EPERM, f'{kind} refused by the rehearsal audit ({event})')

    def _write(self, event, target, dir_fd=None):
        if target is None:
            return
        if isinstance(target, int):  # an already open descriptor
            if target in (0, 1, 2):
                return
            real = self.fd_path(target)
            real = None if real is None else os.path.realpath(real)
            shown = real or f'<descriptor {target}>'
        else:
            try:
                real = self._real(target, dir_fd)
            except (TypeError, ValueError):
                real = None
            shown = real or os.fsdecode(target) if isinstance(target, (str, bytes, os.PathLike)) else repr(type(target))
        if real in self.also or self._inside(real, self.allowed):
            return
        self._refuse('write_outside', event, shown)

    def _note_read(self, real):
        if self._inside(real, self.watch):
            self.read[real] = self.read.get(real, 0)+1

    def event(self, name, args):
        if not self.armed:
            return
        if name == 'open':
            target, mode, flags = (list(args)+[None, None, None])[:3]
            writing = (isinstance(flags, int) and flags & WRITE_FLAGS) or (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            if writing:
                self._write(name, target)
            elif isinstance(target, (str, bytes, os.PathLike)):
                self._note_read(os.path.realpath(os.fsdecode(target)))
        elif name in MUTATING:
            for position, fd_position in MUTATING[name]:
                if position < len(args):
                    dir_fd = args[fd_position] if fd_position is not None and fd_position < len(args) else None
                    self._write(name, args[position], dir_fd)
        elif name == 'sqlite3.connect':
            path, read_only = sqlite_target(args[0] if args else '')
            if path is None:
                return
            if read_only:
                self._note_read(os.path.realpath(path))
            else:
                self._write(name, path)
        elif name.startswith(PROCESS):
            self._refuse('process', name, args[0] if args else None)
        elif name.startswith(NETWORK):
            self._refuse('network', name, None)


def tree_snapshot(root, *, skip=()):
    """``{relative path: (size, mtime_ns)}`` of every file and folder under ``root`` (links not followed);
    ``skip``: relative folder names not descended into. The after-the-fact evidence of what a run touched."""
    root, found = os.path.realpath(root), {}
    for folder, folders, files in os.walk(root, followlinks=False):
        relative = os.path.relpath(folder, root)
        for name in (*folders, *files):
            path = os.path.join(folder, name)
            try:
                info = os.lstat(path)
            except OSError:
                continue
            found[os.path.normpath(os.path.join(relative, name))] = (info.st_size if not os.path.isdir(path) else None, info.st_mtime_ns)
        folders[:] = [name for name in folders if os.path.normpath(os.path.join(relative, name)) not in skip]
    return found


def tree_changes(before, after):
    """Sorted ``[(what, relative path)]`` with what in added / removed / changed."""
    return sorted([('added', path) for path in after.keys()-before.keys()]
                  + [('removed', path) for path in before.keys()-after.keys()]
                  + [('changed', path) for path in before.keys() & after.keys() if before[path] != after[path]])
