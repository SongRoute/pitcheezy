"""REFERENCE card for a pitcher the live recommender does not support (product decision A2).

"이 투수가 올해 이 카운트·타자 방향에서 자주 던진 공": descriptive record statistics from a committed, strictly prior
2026 profile (role ``SERVICE_PROFILE_STRICTLY_PRIOR``). It is NOT a recommendation and NOT a prediction of the next
pitch: no rank, no target location, no model is loaded or run. The shares are frequencies of pitches already thrown.

Where every rule comes from (nothing here re-implements a support rule):

* The Step 5 source is executed in place, unchanged, from the exact bytes that hash to ``SOURCE['pins']`` (the
  same three pins as the shadow runner's ``shadow_config.json``): ``step5/features.py`` (``ProfileStore``),
  ``profile_reader.py`` (``load_store``) and ``live_adapter.py`` (``load_committed``, ``LiveProfile``). The bytes
  are read once, hashed, compiled and executed from memory, so no bytecode is written, ``sys.path`` is not changed
  and the hashed bytes are the executed bytes. A differing pin fails closed: no card.
* Only a COMMITTED profile whose supervisor verdict names its exact bytes is connected (``load_committed``).
* Strictly prior is ``LiveProfile.strictly_prior``: profile cutoff < game officialDate. A profile with cutoff
  2026-10-10 is NOT usable for officialDate 2026-10-10, only from 2026-10-11.
* Pitcher and per-pitch-type minimum samples are ``LiveProfile.support`` with its own defaults (keyword defaults
  of ``ProfileStore.support`` in the pinned ``step5/features.py``: min_total 50, min_known_coverage 0.9,
  max_age_days 365, min_count 5, fraction 0.01). A pitcher with a ``general_reason`` gets no card; a pitch type
  that is not ``provisionally_eligible`` is never listed.
* The adapter has NO threshold for one count x batter-side cell. ``SAMPLE_RULE['cell_min_pitches']`` is a
  PROVISIONAL value introduced here (not tuned on any data) and awaits a PM decision. Below it: no card, and no
  fallback to a broader cell under the same title.

What is counted: ``ProfileStore.stats[(pitch_type, pitcher_hand, batter_stand, balls, strikes)][pitcher]`` is
``[ten outcome10 counts, ordinal of the newest game day]``; the sum of the ten counts is the number of this pitcher's
pitches of that type in that count against that batter side. The store adds a row to ``stats`` only when the pitch
has a pitch type and all four of release_speed, release_spin_rate, pfx_x, pfx_z are finite, so a cell can hold
slightly fewer pitches than were thrown. Shares are over all of the pitcher's counted pitches in the cell.

``card()`` never raises: a missing, corrupt, uncommitted or not strictly prior source gives ``None``; the reason is
kept for an operator (``explain``, ``health``) and must never reach the fan screen.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import types

from .domain import PITCH_LABELS

_JOBS = '/Users/song/Projects/pitcheezy-pm-jobs/20261010-ws-service-006'
# Same bytes the shadow runner pins (20261011-step5-shadow-001/shadow_config.json "pins").
SOURCE = {
    'features': f'{_JOBS}/step5-2024-runner-lifecycle-f1-author-001/step5/features.py',
    'profile_reader': f'{_JOBS}/step5-2026-profile-reader-source-repair-001/profile_reader.py',
    'live_adapter': f'{_JOBS}/step5-2026-profile-reader-source-repair-001/live_adapter.py',
    'pins': {'features': '2e9e71228e2946ac88231219927ce7f40cb9c261cb9c72733db1facf6d458d03',
             'profile_reader': '554b3ac78ae038612a931a6ac78aaadc891c64fff72b60aa40b84cd9bfdb1ae7',
             'live_adapter': '264205f86fb9c4fe129530c9793af235ba662b30b102f18d7824513a17583f22'}}

SAMPLE_RULE = {
    # PROVISIONAL, introduced by this module, not tuned on any data, awaiting a PM decision. The adapter has no
    # count x batter-side threshold. At 20 pitches one pitch moves a share by 5 percentage points.
    'cell_min_pitches': 20,
    'status': 'provisional_awaiting_pm_decision'}
DISPLAY = {'max_items': 4, 'min_item_share': .05}  # presentation only; not a sample rule

KIND, REASON_CODE = 'profile_reference', 'unsupported/profile_reference'
TITLE, CHIP = '이 투수가 올해 이 카운트·타자 방향에서 자주 던진 공', '참고 · 기록 통계'
BROAD_TITLE = '이 투수가 올해 이 타자 방향에서 자주 던진 공 (모든 카운트)'  # optional fallback card, off by default
ROLE = 'SERVICE_PROFILE_STRICTLY_PRIOR'
POINTER_SCHEMA = 'profile-refresh-latest-v1'
ATTEMPT_FILES = ('PROFILE.json', 'RECEIPT.json', 'SUPERVISOR-RESULT.json')
MAX_SOURCES, MAX_POINTER_BYTES = 4, 1 << 16
# Statcast codes the app's PITCH_LABELS does not carry; a code in neither table is shown as the code itself.
EXTRA_LABELS = {'CS': '슬로 커브', 'EP': '이퓨스', 'FA': '패스트볼', 'FO': '포크볼', 'KN': '너클볼', 'SC': '스크루볼',
                'PO': '피치아웃'}
_DATE = re.compile(r'\d{4}-\d{2}-\d{2}')
_source_lock = threading.Lock()


def _execute(name, path, pin):
    """A module made from exactly the bytes that hash to ``pin`` (no reopen between hash and execution)."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError('pinned Step 5 source differs from the accepted bytes: '+name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def load_sources(source=None):
    """(features, live_adapter) executed from the pinned bytes, or ValueError/OSError. ``live_adapter`` imports
    ``profile_reader`` by name: the pinned module is put into ``sys.modules`` for that one statement only."""
    source = source or SOURCE
    pins = source['pins']
    features = _execute('pz_profile_reference_step5_features', source['features'], pins['features'])
    reader = _execute('profile_reader', source['profile_reader'], pins['profile_reader'])
    with _source_lock:
        before = sys.modules.get('profile_reader')
        sys.modules['profile_reader'] = reader
        try:
            adapter = _execute('pz_profile_reference_live_adapter', source['live_adapter'], pins['live_adapter'])
        finally:
            if before is None:
                sys.modules.pop('profile_reader', None)
            else:
                sys.modules['profile_reader'] = before
    return features, adapter


def _signature(directory):
    out = []
    for name in ATTEMPT_FILES:
        try:
            s = os.stat(directory/name)
            out.append((s.st_mtime_ns, s.st_size, s.st_ino))
        except OSError:
            out.append(None)
    return tuple(out)


def _official_date(value):
    if type(value) is not str or not _DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _percentages(counts):
    """Integer percentages of ``counts`` (given most frequent first): each is the exact percentage rounded down or
    up, equal counts always get the same percentage, and the sum is 100, or just under 100 when giving the last
    points would split a tie."""
    total = sum(counts)
    floors = [n*100//total for n in counts]
    left = 100-sum(floors)
    for n in sorted(set(counts), key=lambda n: (-(n*100 % total), -n)):
        group = [i for i, m in enumerate(counts) if m == n]
        if n*100 % total == 0 or len(group) > left:
            break
        for i in group:
            floors[i] += 1
        left -= len(group)
    return floors


def pitch_label(code):
    return PITCH_LABELS.get(code) or EXTRA_LABELS.get(code) or code


class ProfileReference:
    """Lazy, thread-safe reader of committed strictly-prior profiles.

    ``sources``: attempt directories (PROFILE.json + RECEIPT.json + SUPERVISOR-RESULT.json) and/or
    ``LATEST.json`` pointer files (``profile-refresh-latest-v1``), at most ``MAX_SOURCES``. Each profile is loaded
    once and kept until its files change, its pointer moves or it leaves ``sources``, so memory is bounded by the
    number of sources. Files are looked at again at most every ``recheck_seconds``.
    """

    def __init__(self, sources, *, source=None, sample_rule=None, display=None, broad_fallback=False,
                 recheck_seconds=30., clock=time.monotonic):
        listed = [Path(p) for p in ([sources] if isinstance(sources, (str, os.PathLike)) else list(sources or ()))]
        self._sources, self._dropped = listed[:MAX_SOURCES], [str(p) for p in listed[MAX_SOURCES:]]
        self._source = source or SOURCE
        self._rule, self._display = dict(SAMPLE_RULE) | dict(sample_rule or {}), dict(DISPLAY) | dict(display or {})
        self._broad, self._recheck, self._clock = bool(broad_fallback), float(recheck_seconds), clock
        self._lock, self._stat_lock = threading.RLock(), threading.Lock()
        self._modules, self._modules_error, self._modules_tried = None, None, None
        self._cache = {}  # resolved attempt directory -> {'signature', 'profile' | None, 'reason', 'sha256', ...}
        self._active, self._checked = [], None  # [(source text, cache entry)] in source order
        self._reasons, self._last, self._loads = Counter(), None, 0

    # ---- loading -------------------------------------------------------------------------------------------------
    def _load_modules(self, now):
        if self._modules is None and (self._modules_tried is None or now-self._modules_tried >= self._recheck):
            self._modules_tried = now
            try:
                self._modules, self._modules_error = load_sources(self._source), None
            except Exception as exc:  # fail closed: a differing or unreadable pinned source connects nothing
                self._modules_error = f'pinned_source_unusable:{type(exc).__name__}:{exc}'
        return self._modules

    def _resolve(self, path):
        """(attempt directory, pointer dict or None) of one source, or ValueError."""
        if path.is_dir():
            return path.resolve(), None
        if not path.is_file():
            raise ValueError('source_missing')
        if path.stat().st_size > MAX_POINTER_BYTES:
            raise ValueError('pointer_too_large')
        pointer = json.loads(path.read_bytes())
        if not isinstance(pointer, dict) or pointer.get('schema') != POINTER_SCHEMA \
                or type(pointer.get('profile_dir')) is not str or type(pointer.get('profile_sha256')) is not str:
            raise ValueError('pointer_not_a_profile_refresh_latest')
        directory = Path(pointer['profile_dir'])
        if not directory.is_absolute() or not directory.is_dir():
            raise ValueError('pointer_profile_dir_missing')
        return directory.resolve(), pointer

    def _load(self, directory, signature):
        features, adapter = self._modules
        entry = {'directory': str(directory), 'signature': signature, 'profile': None, 'reason': None,
                 'sha256': None, 'cutoff': None, 'not_before': None}
        try:
            self._loads += 1
            profile = adapter.load_committed(features, directory)
            store, provenance = profile._store, profile.provenance()
            if provenance.get('role') != ROLE:
                raise ValueError('profile role is not '+ROLE)
            if not isinstance(store.cutoff, date) or not isinstance(store.not_before, date):
                raise ValueError('profile window is not registered')
            entry |= {'profile': profile, 'sha256': provenance['profile_sha256'], 'cutoff': store.cutoff,
                      'not_before': store.not_before, 'rows': provenance.get('rows'),
                      'supervisor_verdict': provenance.get('supervisor_verdict')}
        except Exception as exc:
            entry['reason'] = f'profile_not_connected:{type(exc).__name__}:{exc}'
        return entry

    def _refresh(self):
        """Current [(source, entry)] list; files are touched at most once per ``recheck_seconds``."""
        now = self._clock()
        with self._lock:
            if self._checked is not None and now-self._checked < self._recheck:
                return self._active
            if self._load_modules(now) is None:
                self._active, self._checked = [(str(p), {'profile': None, 'reason': self._modules_error})
                                               for p in self._sources], now
                return self._active
            active, cache = [], {}
            for path in self._sources:
                try:
                    directory, pointer = self._resolve(path)
                except Exception as exc:
                    active.append((str(path), {'profile': None, 'reason': f'source_unusable:{type(exc).__name__}:{exc}'}))
                    continue
                entry = cache.get(directory) or self._cache.get(directory)
                signature = _signature(directory)
                if entry is None or entry['signature'] != signature:
                    entry = self._load(directory, signature)
                cache[directory] = entry
                if pointer is not None and entry['profile'] is not None and (
                        pointer['profile_sha256'] != entry['sha256']
                        or pointer.get('as_of_cutoff') not in (None, entry['cutoff'].isoformat())):
                    entry = {'profile': None, 'reason': 'pointer_names_other_profile_bytes_or_cutoff'}
                active.append((str(path), entry))
            self._cache, self._active, self._checked = cache, active, now  # anything no longer named is dropped
            return self._active

    def warm(self):
        """Load every source now (e.g. at start-up) instead of on the first fan request. Returns ``health()``."""
        try:
            self._refresh()
        except Exception:
            pass
        return self.health()

    def _pick(self, day, text):
        """(entry, None) for the newest profile strictly prior to ``day``, or (None, reason)."""
        best, reasons = None, []
        for source, entry in self._refresh():
            profile = entry.get('profile')
            if profile is None:
                reasons.append(entry.get('reason') or 'profile_not_connected')
            elif not profile.strictly_prior(text):
                reasons.append('profile_as_of_not_strictly_before_game_date')
            elif entry['cutoff'].year != day.year or entry['not_before'].year != day.year:
                reasons.append('profile_window_is_not_this_season')  # the title says 올해
            elif best is None or entry['cutoff'] > best['cutoff']:
                best = entry
        if best is not None:
            return best, None
        return None, 'no_usable_profile:'+('|'.join(sorted(set(reasons))) if reasons else 'no_source_configured')

    # ---- card ----------------------------------------------------------------------------------------------------
    def explain(self, pitcher_id, batter_side, balls, strikes, official_date, *, pitcher_hand=None):
        """(card or None, operator reason or None). Never raises. The reason is for logs/health only."""
        try:
            card, reason = self._card(pitcher_id, batter_side, balls, strikes, official_date, pitcher_hand)
        except Exception as exc:
            card, reason = None, f'internal_error:{type(exc).__name__}'
        with self._stat_lock:
            self._reasons[reason or 'card'] += 1
            self._last = reason or 'card'
        return card, reason

    def card(self, pitcher_id, batter_side, balls, strikes, official_date, *, pitcher_hand=None):
        """The reference card (see module docstring for the shape) or None. Never raises."""
        return self.explain(pitcher_id, batter_side, balls, strikes, official_date, pitcher_hand=pitcher_hand)[0]

    def _card(self, pitcher_id, batter_side, balls, strikes, official_date, pitcher_hand):
        bad = [name for name, ok in (
            ('pitcher_id', type(pitcher_id) is int and 1 <= pitcher_id <= 10**7),
            ('batter_side', type(batter_side) is str and batter_side in ('L', 'R')),
            ('balls', type(balls) is int and 0 <= balls <= 3), ('strikes', type(strikes) is int and 0 <= strikes <= 2),
            ('official_date', _official_date(official_date) is not None),
            ('pitcher_hand', pitcher_hand is None or (type(pitcher_hand) is str and pitcher_hand in ('L', 'R')))) if not ok]
        if bad:
            return None, 'bad_input:'+','.join(bad)
        entry, reason = self._pick(_official_date(official_date), official_date)
        if entry is None:
            return None, reason
        profile = entry['profile']
        support = profile.support(pitcher_id, official_date)  # the adapter's own rule and defaults
        if support['general_reason']:
            return None, 'unsupported_player:'+support['general_reason']
        eligible = [c['pitch_type'] for c in support['candidates'] if c['status'] == 'provisionally_eligible']
        if not eligible:
            return None, 'no_action_meets_support_thresholds'
        store = profile._store
        kinds = [k for k in store.pitcher[pitcher_id]['types'] if k is not None]
        hands = (pitcher_hand,) if pitcher_hand else ('L', 'R')

        def counted(cells):
            out = {}
            for kind in kinds:
                n = 0
                for hand in hands:
                    for b, s in cells:
                        own = store.stats.get((kind, hand, batter_side, b, s), {}).get(pitcher_id)
                        if own is not None:
                            n += int(round(float(own[0].sum())))
                if n:
                    out[kind] = n
            return out

        minimum = self._rule['cell_min_pitches']
        counts, title, scope = counted(((balls, strikes),)), TITLE, 'count_and_batter_side'
        if sum(counts.values()) < minimum:
            if not self._broad:
                return None, 'cell_below_provisional_minimum'
            counts, title, scope = counted([(b, s) for b in range(4) for s in range(3)]), BROAD_TITLE, 'batter_side'
            if sum(counts.values()) < minimum:
                return None, 'cell_below_provisional_minimum'
        total = sum(counts.values())
        order = sorted(counts, key=lambda k: (-counts[k], k))
        percent = dict(zip(order, _percentages([counts[k] for k in order])))
        items = [{'pitch_type': k, 'pitch_label': pitch_label(k), 'share': round(counts[k]/total, 4),
                  'share_pct': percent[k], 'count': counts[k]}
                 for k in order if k in eligible and counts[k]/total >= self._display['min_item_share']]
        items = items[:self._display['max_items']]
        if not items:
            return None, 'no_listed_pitch_type_in_cell'
        rest = total-sum(i['count'] for i in items)
        cutoff = entry['cutoff']
        sample = {'pitches': total, 'batter_side': batter_side, 'scope': scope}
        if scope == 'count_and_batter_side':
            sample |= {'balls': balls, 'strikes': strikes}
        return {
            'kind': KIND, 'reason_code': REASON_CODE, 'title': title, 'chip': CHIP, 'items': items,
            'other': {'share': round(rest/total, 4), 'count': rest,
                      'share_pct': sum(percent.values())-sum(i['share_pct'] for i in items)} if rest else None,
            'sample': sample, 'as_of': cutoff.isoformat(),
            'note': f'올해 {cutoff.month}월 {cutoff.day}일까지 이 투수가 같은 상황에서 실제로 던진 공 {total}개의 비율입니다. '
                    '추천이나 다음 공 예측이 아닙니다.',
            'provenance': {'profile_sha256_prefix': entry['sha256'][:12], 'role': ROLE, 'cutoff': cutoff.isoformat(),
                           'not_before': entry['not_before'].isoformat(),
                           'counted': 'pitches with a pitch type and complete speed/spin/movement readings',
                           'thresholds': dict(support['thresholds']) | {
                               'cell_min_pitches': minimum, 'cell_min_pitches_status': self._rule['status']}}}, None

    # ---- operator view -------------------------------------------------------------------------------------------
    def health(self):
        """Operator-only summary: per-source state and counts of ``explain`` outcomes. Touches no file."""
        with self._lock:
            sources = [{'source': source, 'usable': entry.get('profile') is not None, 'reason': entry.get('reason'),
                        'profile_dir': entry.get('directory'), 'profile_sha256': entry.get('sha256'),
                        'cutoff': entry['cutoff'].isoformat() if entry.get('cutoff') else None}
                       for source, entry in self._active]
            loads, checked, error = self._loads, self._checked is not None, self._modules_error
        with self._stat_lock:
            outcomes, last = dict(self._reasons), self._last
        return {'kind': KIND, 'checked': checked, 'pinned_source_error': error, 'sources': sources,
                'sources_ignored_over_limit': list(self._dropped), 'profile_loads': loads, 'outcomes': outcomes,
                'last_outcome': last, 'sample_rule': dict(self._rule), 'broad_fallback': self._broad}
