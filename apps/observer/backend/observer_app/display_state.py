"""One explicit display state for the pending pitch slot of a live game, derived from store and ledger facts.

The fan screen (``/api/service/games/{pk}``) and the legacy live view (``/api/live/{pk}/state``) both carry the
``display_state`` built here, so what a viewer sees is never inferred by the client from missing fields.

States (``state``) and what they mean for the recommendation area:

* ``recommended``       a recommendation is shown (ledger-committed before display when a store is on)
* ``unconfirmed``       확인 중: the data the recommendation would rest on is not admitted (SVC-WS-03); withheld
* ``unsupported``       the frozen policy does not cover this pitcher / pitch types / at-bat; one short reason
* ``feed_delayed``      a recommendation is shown, but the snapshot reached us later than the threshold
* ``disconnected``      the last upstream poll failed; only the last snapshot is shown, nothing new is recommended
* ``ledger_unavailable`` a recommendation was computed but could not be committed, so it is not shown
* ``waiting``           nothing to recommend right now (inning change, result pending, computing, buffering)
* ``unavailable``       this run has no recommender (off, model not loaded, computation failed)
* ``pregame`` / ``final``

The admission rule (``admit_body``) is the SVC-WS-03 selected-body rule; ``synthetic_game`` calls the same
function, so the synthetic walk-through and the real screen cannot drift apart.

Thresholds are PROVISIONAL configuration (``live_config.json`` ``display``), not validated acceptance limits.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

from .live_store import _canonical

SCHEMA = 'pitcheezy-display-state-v1'
STATES = ('recommended', 'unconfirmed', 'unsupported', 'feed_delayed', 'disconnected', 'ledger_unavailable',
          'waiting', 'unavailable', 'pregame', 'final')
# Provider-revision order of the served store version -> why it is not fresh input (same table as the research gate).
FRESHNESS = {'ordered': None, 'stale': 'stale_provider_stamp', 'unorderable': 'unorderable_provider_stamp',
             'same_stamp_conflict': 'same_stamp_conflict_provider_stamp'}
# Provisional default: NOT a measured acceptance limit. Background only: one live game (849844, 2026-10-02) had a
# median feed lag of about 19 s; 45 s is a first guess for "noticeably later than usual" and must be revisited.
DEFAULTS = {'feed_lag_delayed_s': 45.0}
RECEIPT_ALL = '__all__'  # receipts marker: the feed has a play without a usable atBatIndex, so no key can be trusted

POLICY_REASONS = {
    'UNSUPPORTED_UNKNOWN_PITCHER': 'unknown_pitcher', 'UNSUPPORTED_PITCHER_HAND': 'unknown_pitcher',
    'UNSUPPORTED_EMPTY_SUPPORT': 'no_supported_pitch_types',
    'UNSUPPORTED_LOGGING_POSITIVITY': 'at_bat_rejected', 'UNSUPPORTED_MID_PA': 'at_bat_rejected',
    'UNSUPPORTED_NO_LOGGED_ACTION': 'at_bat_rejected', 'UNSUPPORTED_MISSING_ACTION_LABEL': 'at_bat_rejected',
    'UNSUPPORTED_INCOMPLETE_START': 'at_bat_rejected', 'UNSUPPORTED_INCONSISTENT_HISTORY': 'at_bat_rejected',
    'NOT_SUBMITTED': 'at_bat_rejected', 'NO_DECISION': 'at_bat_rejected'}

# (state, reason) -> (short label, one calm sentence). The frontend renders these as they are.
COPY = {
    ('recommended', 'ledger_recorded'): ('다음 공 추천', None),
    ('recommended', 'ledger_conflict_first_kept'): ('다음 공 추천', None),
    ('recommended', 'shown_unrecorded_no_ledger'): ('다음 공 추천', None),
    ('unconfirmed', 'stale_provider_stamp'): ('확인 중', '방금 받은 자료가 앞선 자료보다 이른 시각이라, 확인될 때까지 추천을 잠시 멈췄어요.'),
    ('unconfirmed', 'unorderable_provider_stamp'): ('확인 중', '받은 자료의 순서를 확인할 수 없어, 확인될 때까지 추천을 잠시 멈췄어요.'),
    ('unconfirmed', 'same_stamp_conflict_provider_stamp'): ('확인 중', '같은 시각에 내용이 다른 자료가 와서, 확인될 때까지 추천을 잠시 멈췄어요.'),
    ('unconfirmed', 'provider_order_unknown'): ('확인 중', '받은 자료의 순서를 확인하는 중이라 추천을 잠시 멈췄어요.'),
    ('unconfirmed', 'prior_pitch_unconfirmed'): ('확인 중', '앞선 공의 기록을 확인하는 중이라 추천을 잠시 멈췄어요.'),
    ('unconfirmed', 'receipts_unreadable'): ('확인 중', '받은 기록을 확인할 수 없어 추천을 잠시 멈췄어요.'),
    ('unsupported', 'unknown_pitcher'): ('추천 없음', '이 투수는 2025년까지 쌓인 기록이 없어 추천하지 않아요.'),
    ('unsupported', 'no_supported_pitch_types'): ('추천 없음', '이 투수가 이 타석 방향에 던진 구종 기록이 부족해 추천하지 않아요.'),
    ('unsupported', 'at_bat_rejected'): ('추천 없음', '이 타석은 앞선 공 기록을 이어 볼 수 없어 추천을 쉬어요.'),
    ('unsupported', 'policy_refused'): ('추천 없음', '이 상황은 추천 대상이 아니에요.'),
    ('unsupported', 'game_type_out_of_scope'): ('추천 없음', '이 경기는 추천 대상이 아니에요.'),
    ('feed_delayed', 'feed_lag_above_threshold'): ('자료 늦음', '경기 자료가 평소보다 늦게 도착했어요. 추천이 실제 경기보다 늦을 수 있어요.'),
    ('ledger_unavailable', 'display_history_unreadable'): ('추천 보류', '앞서 보여 드린 추천 기록을 확인하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.'),
    ('disconnected', 'poll_failed_last_snapshot_only'): ('연결 끊김', '경기 자료 연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.'),
    ('ledger_unavailable', None): ('추천 보류', '추천 기록을 저장하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.'),
    ('waiting', 'between_innings'): ('공수 교대', '공수 교대 중이에요.'),
    ('waiting', 'resolving'): ('결과 확인 중', '방금 공의 결과를 기다리고 있어요.'),
    ('waiting', 'computing'): ('계산 중', '추천을 계산하고 있어요.'),
    ('waiting', 'no_pending_pitch'): ('대기 중', '다음 공을 기다리고 있어요.'),
    ('waiting', 'buffering'): ('대기 중', '내 중계 시점까지 기다리고 있어요.'),
    ('unavailable', 'recommender_off'): ('추천 꺼짐', '이 실행에서는 실시간 추천을 제공하지 않아요.'),
    ('unavailable', 'recommender_unavailable'): ('추천 준비 안 됨', '추천 모델을 준비하지 못했어요.'),
    ('unavailable', 'compute_failed'): ('추천 없음', '이 상황의 추천을 계산하지 못했어요.'),
    ('pregame', 'scheduled'): ('경기 전', '경기 시작 전이에요.'),
    ('pregame', 'not_live'): ('진행 중 아님', '지금은 경기가 진행 중이 아니에요.'),
    ('final', 'game_over'): ('경기 종료', '경기가 끝났어요.'),
}
CORRECTION_COPY = {'replaced': '상황이나 기록이 바뀌어 추천이 달라졌어요.',
                   'withdrawn': '조금 전에 보여 드린 추천은 지금 표시하지 않아요.'}


def config(live_config=None):
    """Display thresholds: ``live_config.json`` ``display`` over ``DEFAULTS``; the env var wins. All provisional."""
    if live_config is None:
        from .live_feed import LIVE_CONFIG as live_config
    out = DEFAULTS | {k: v for k, v in (live_config.get('display') or {}).items() if k in DEFAULTS}
    override = os.environ.get('PITCHEEZY_OBSERVER_FEED_LAG_DELAYED_S')
    if override:
        out['feed_lag_delayed_s'] = float(override)
    return out | {'provisional': True}


# ---------- SVC-WS-03 selected-body admission (shared with synthetic_game) ----------
def _sha(value):
    return value if isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value) else None


def body_sha(event):
    """The pitch ledger's hash of a playEvent body (``sha256(_canonical(playEvent))``), or None if unusable."""
    try:
        return hashlib.sha256(_canonical(event)).hexdigest()
    except (TypeError, ValueError):
        return None


def admit_body(event, key, rows):
    """May the displayed pitch body ``event`` be shown as an observed pitch? Returns (status, detail).

    ``rows`` are committed pitch-ledger rows (``LiveSnapshotStore.pitch_events`` events) and ``key`` the official
    ``(at_bat_number, pitch_number)``. The body is ``received`` only if its hash is the key's NEWEST accepted
    (new/correction) row: the ledger dedups repeat quarantine rows whatever the seq, so neither the absence of a
    quarantine row at the selected seq nor an older accepted copy of the body clears it. A body that is only
    quarantined, superseded by a later accepted revision, never accepted, or whose hash is unusable is
    ``unconfirmed`` (fail closed) with the quarantine reason, ``not_accepted`` or ``body_hash_unusable``.
    Known imprecision (Q1): an unselected stale arrival whose per-pitch correction was accepted makes the
    selected (older) body ``unconfirmed/not_accepted``; that is fail-safe, but the reason is inexact."""
    accepted = [r for r in rows if r['outcome'] != 'quarantine' and (r['at_bat_number'], r['pitch_number']) == tuple(key)]
    shown = body_sha(event)
    if shown is None or (accepted and _sha(accepted[-1]['sha']) is None):
        return 'unconfirmed', 'body_hash_unusable'
    if accepted and accepted[-1]['sha'] == shown:
        return 'received', None
    refused = [r for r in rows if r['outcome'] == 'quarantine' and r['sha'] == shown]
    return 'unconfirmed', refused[-1]['reason'] if refused else 'not_accepted'


def pitch_receipts(feed, rows, *, committed=True):
    """Admission of every pitch body in ``feed`` against the pitch ledger ``rows``:
    ``{(at_bat_number, pitch_number): {'status', 'reason', 'corrected', 'sha'}}``. A pitch without a usable official
    key (``incomplete_identity``) is keyed by its position ``(at_bat_number or None, pitchNumber or None)`` and is
    never ``received``. ``committed=False`` (memory-only entry during a store outage) relabels a body the ledger
    never accepted as ``uncommitted_memory_entry``. ``corrected`` = an accepted correction exists for the key, so a
    viewer who saw the first version is told the record changed."""
    by_key, by_sha = {}, {}
    for row in rows:
        if row['outcome'] == 'quarantine':
            by_sha.setdefault(row['sha'], []).append(row)
        else:
            by_key.setdefault((row['at_bat_number'], row['pitch_number']), []).append(row)
    out, tainted = {}, set()
    live = feed.get('liveData') if isinstance(feed, dict) else None
    plays = (live.get('plays') or {}).get('allPlays') if isinstance(live, dict) else None
    for play in plays if isinstance(plays, list) else []:
        about = play.get('about') if isinstance(play, dict) else None
        index = about.get('atBatIndex') if isinstance(about, dict) else None
        at_bat = index+1 if type(index) is int and index >= 0 else None
        events = play.get('playEvents') if isinstance(play, dict) else None
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict) or not event.get('isPitch'):
                continue
            number = event.get('pitchNumber') if type(event.get('pitchNumber')) is int else None
            key, sha = (at_bat, number), body_sha(event)
            mine = by_key.get(key, [])
            status, reason = admit_body(event, key, [*mine, *by_sha.get(sha, [])])
            if status != 'received' and reason == 'not_accepted' and not committed:
                reason = 'uncommitted_memory_entry'
            out[key] = {'status': status, 'reason': reason, 'sha': sha,
                        'corrected': status == 'received' and any(r['outcome'] == 'correction' for r in mine)}
            if number is None or at_bat is None:
                tainted.add(at_bat)
    # A pitch without an official key is listed by position downstream (``pitchNumber or its order``), which can
    # collide with a real key: every pitch of such an at-bat (of the whole feed, if the at-bat itself is unknown)
    # fails closed rather than borrow another pitch's admission.
    for key, item in out.items():
        if (None in tainted or key[0] in tainted) and item['status'] == 'received':
            item.update(status='unconfirmed', reason='incomplete_identity', corrected=False)
    if None in tainted:
        out[RECEIPT_ALL] = {'status': 'unconfirmed', 'reason': 'incomplete_identity', 'sha': None, 'corrected': False}
    return out


def fallback_receipts(feed, remembered):
    """Receipts when the pitch ledger cannot be read: a body is ``received`` only if this process already saw
    exactly that body admitted (``remembered``: key -> sha); everything else is unconfirmed."""
    out = pitch_receipts(feed, [], committed=True)
    for key, item in out.items():
        if item['sha'] is not None and remembered.get(key) == item['sha'] and item['reason'] != 'incomplete_identity':
            item.update(status='received', reason=None)
        else:
            item.update(status='unconfirmed', reason='receipts_unreadable')
    return out


def public_receipt(item):
    """A receipt as the screen gets it (no hashes)."""
    return {'status': item['status'], 'reason': item['reason'], 'corrected': bool(item['corrected'])}


def receipts_version(receipts):
    """Changes whenever any pitch's admission or correction mark changes, so a client that only asks for newer
    pitches knows to take the full history again."""
    if receipts is None:
        return None
    marks = sorted((str(k), v['status'], v['reason'] or '', v['corrected']) for k, v in receipts.items()
                   if v['status'] != 'received' or v['corrected'])
    return hashlib.sha256(_canonical(marks)).hexdigest()[:12]


# ---------- facts ----------
def pending_key(feed):
    """The policy's name (``gamePk:at_bat_number:pitch_number``) for the pitch about to be thrown, or None."""
    try:
        plays = feed['liveData']['plays']
        done = plays.get('allPlays') or []
        play = plays.get('currentPlay') or (done[-1] if done else None)
        pk = feed['gameData']['game']['pk']
        if play is None:
            return f'{pk}:1:1'
        if play['about'].get('isComplete', True):
            return f"{pk}:{play['about']['atBatIndex']+2}:1"
        numbered = [e for e in play.get('playEvents') or []
                    if e.get('isPitch') or (e.get('type') == 'no_pitch' and ((e.get('details') or {}).get('call') or {}).get('code'))]
        return f"{pk}:{play['about']['atBatIndex']+1}:{len(numbered)+1}"
    except (KeyError, TypeError, AttributeError):
        return None


def summary(recommendation):
    """What a viewer can see of a recommendation: the ordered (pitch type, zone) of its candidates."""
    return [[c.get('pitch_type'), c.get('zone_id')] for c in ((recommendation or {}).get('candidates') or [])[:3]]


def _iso(wall):
    return None if wall is None else time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(wall))


def feed_facts(feed, game_pk, entry, cfg, *, now):
    """Connection facts of the served snapshot. Polling runs only while someone watches, so ``receipt - event``
    alone may just be time nobody polled. ``lag_s`` is therefore the PROVEN minimum: the last good poll before the
    one that brought this event did not have it yet, so the event reached us at least ``prior poll - event`` late
    (``lag_max_s`` = ``receipt - event`` is the upper bound). Without such a prior poll (first poll, restart) or for
    a replayed recording (its event times are in the past) nothing is measured and nothing is called delayed."""
    game_pk = int(game_pk)
    failing = game_pk in getattr(feed, 'failing', {})
    prior = (entry or {}).get('prior_poll')
    measured = entry is not None and prior is not None and _is_real_poll(feed)
    lag = round(max(0.0, prior-entry['event']), 1) if measured else None
    since = getattr(feed, 'failing_since', {}).get(game_pk)
    has_store, store_error = hasattr(feed, 'store'), getattr(feed, 'store_error', None)
    return {'ok': not failing, 'lag_s': lag, 'lag_max_s': round(entry['fetched']-entry['event'], 1) if measured else None,
            'lag_measured': measured, 'lag_threshold_s': cfg['feed_lag_delayed_s'],
            'delayed': lag is not None and lag > cfg['feed_lag_delayed_s'], 'thresholds_provisional': True,
            'last_ok_at': _iso(getattr(feed, 'polled_ok', {}).get(game_pk)), 'failing_since': _iso(since) if failing else None,
            'failing_for_s': round(now-since, 1) if failing and since is not None else None,
            'received_at': _iso(entry['fetched']) if entry else None,
            'provider_order': (entry or {}).get('provider_order'),
            # 'failed': a store was configured but could not be opened, so nothing can be recorded (fail closed)
            'store': 'on' if has_store else 'failed' if store_error else 'off', 'store_ok': store_error is None,
            'committed': None if not has_store else bool(entry and entry.get('seq') is not None and entry.get('sha') is not None)}


def _is_real_poll(feed):
    """Only a ``LiveFeed`` polling a real upstream measures lag: a store-backed feed says so in ``source``; a plain
    one must have a transport that is not a replay (``ReplayTransport`` / the timed replay, which has a position)."""
    from .live_feed import ReplayTransport
    source = getattr(feed, 'source', None)
    if source is not None:
        return source == 'live_poll'
    transport = getattr(feed, 'transport', None)
    return transport is not None and not isinstance(transport, ReplayTransport) and not hasattr(transport, 'position')


def gate(parsed_status, abstract, facts, receipts, feed):
    """The state that is decided before any recommendation is looked up, or None when the recommender may be asked.
    Returns (state, reason, detail). For ``disconnected``, ``detail`` is None only if the last snapshot's inputs
    are themselves admitted: otherwise it names the 확인 중 reason and nothing may stay on screen."""
    if abstract == 'Final':
        return 'final', 'game_over', None
    if parsed_status == 'not_live':
        return 'pregame', 'scheduled' if abstract == 'Preview' else 'not_live', None
    if parsed_status == 'unsupported_game_type':
        return 'unsupported', 'game_type_out_of_scope', None
    blocked = _unconfirmed(facts, receipts, feed) if parsed_status == 'ready' else None
    if not facts['ok']:
        return 'disconnected', 'poll_failed_last_snapshot_only', None if blocked is None else f'unconfirmed:{blocked[0]}'
    if parsed_status in ('between_innings', 'resolving'):
        return 'waiting', parsed_status, None
    if parsed_status != 'ready':
        return 'waiting', 'no_pending_pitch', parsed_status
    return None if blocked is None else ('unconfirmed', *blocked)


def _unconfirmed(facts, receipts, feed):
    """(reason, detail) if the pending decision's inputs are not admitted (SVC-WS-03), else None."""
    if facts['store'] == 'on' and facts['committed']:
        order = facts['provider_order']
        stale = FRESHNESS.get(order, 'provider_order_unknown') if isinstance(order, str) else 'provider_order_unknown'
        if stale:
            return stale, order
    if receipts is not None:
        blocking = _blocking_receipt(feed, receipts)
        if blocking is not None:
            return ('receipts_unreadable' if blocking[1] == 'receipts_unreadable' else 'prior_pitch_unconfirmed',
                    f'{blocking[0][0]}:{blocking[0][1]}/{blocking[1]}')
    return None


def _blocking_receipt(feed, receipts):
    """The first not-admitted pitch among the pending decision's inputs: every pitch of the open at-bat and the
    game's newest pitch. Returns (key, reason) or None."""
    if not receipts:
        return None
    plays = feed['liveData']['plays']
    done = plays.get('allPlays') or []
    play = plays.get('currentPlay') or (done[-1] if done else None)
    open_ab = None
    if play is not None and not (play.get('about') or {}).get('isComplete', True):
        index = (play.get('about') or {}).get('atBatIndex')
        open_ab = index+1 if type(index) is int else None
    if RECEIPT_ALL in receipts:
        return (None, None), receipts[RECEIPT_ALL]['reason']
    newest = list(receipts)[-1]
    for key, item in receipts.items():
        if item['status'] != 'received' and (key == newest or (open_ab is not None and key[0] == open_ab)):
            return key, item['reason']
    return None


def correction(history, shown_now):
    """D1: was something else displayed for this pending pitch before? ``history`` is the earlier displays of the
    same pitch key, oldest first, each ``{'summary', 'decided', 'candidates', 'decision_id'}``, excluding the one
    shown now when it can be told apart. ``shown_now`` is the summary on screen now (None = nothing shown).
    ``previous`` is the most recent display that differs from the current one (after A, B, C, A, B it names C: the
    ledger keeps one row per decision, not the order in which repeats were served).
    Returns None or ``{'kind': 'replaced' | 'withdrawn', 'message', 'previous', 'previous_decided'}``."""
    differing = [h for h in history if h['summary'] and h['summary'] != shown_now]
    if not differing:
        return None
    last = differing[-1]
    kind = 'replaced' if shown_now else 'withdrawn'
    top = (last.get('candidates') or [{}])[0]
    return {'kind': kind, 'message': CORRECTION_COPY[kind],
            'previous': {'pitch_type': top.get('pitch_type'), 'pitch_label': top.get('pitch_label'),
                         'zone_id': top.get('zone_id'), 'zone_label': top.get('zone_label')},
            'previous_decided': _iso(last.get('decided'))}


def decide(*, gated, view, facts, history=(), pitch_key=None, recommender=True, held=False):  # noqa: C901
    """The single display state. ``gated`` is ``gate(...)``; ``view`` the raw recommendation view (``armb_view``
    after ``live_ledger.admit``) or None; ``history`` earlier displays for ``pitch_key`` (see ``correction``).
    ``history=None`` means the earlier displays could not be read: nothing is shown then.
    ``recommender=False``: this run has no recommender wired at all. ``held``: the recommendation in ``view`` was
    already displayed for this very decision (ledger row, or this process's memory when no ledger is on); only such
    a one may stay on screen while the feed is disconnected."""
    view = view or None
    ledger = (view or {}).get('ledger') or None
    rec = (view or {}).get('recommendation') if (view or {}).get('status') == 'ready' else None
    shown, detail = False, None
    if gated is not None and gated[0] != 'disconnected':
        state, reason, detail = gated
    elif gated is not None:  # disconnected: only a recommendation the ledger already held for this decision stays
        state, reason, detail = gated
        shown = bool(rec and rec.get('candidates')) and bool(held) and detail is None and history is not None
    elif not recommender:
        state, reason = 'unavailable', 'recommender_off'
    elif view is None:
        state, reason = 'waiting', 'no_pending_pitch'
    elif (ledger or {}).get('status') == 'withheld':
        state, reason, detail = 'ledger_unavailable', ledger.get('why') or 'write_failed', None
    elif view.get('status') == 'computing':
        state, reason = 'waiting', 'computing'
    elif view.get('status') == 'unavailable':
        state, reason = 'unavailable', view.get('code') or 'recommender_unavailable'
    elif view.get('status') == 'ready' and rec and rec.get('candidates'):
        shown = True
        reason = ('shown_unrecorded_no_ledger' if ledger is None else
                  'ledger_conflict_first_kept' if ledger.get('outcome') == 'conflict' else 'ledger_recorded')
        state = 'feed_delayed' if facts['delayed'] else 'recommended'
        if state == 'feed_delayed':
            detail, reason = reason, 'feed_lag_above_threshold'
    else:
        state, reason = 'unsupported', POLICY_REASONS.get(view.get('policy_status'), 'policy_refused')
        detail = view.get('policy_status')
    if shown and (history is None or facts['store'] == 'failed'):
        # D1 cannot be kept without knowing what was displayed before, and a configured store that did not open
        # cannot record anything: withhold rather than show something that may be an unannounced change.
        state, detail, shown = 'ledger_unavailable', reason, False
        reason = 'display_history_unreadable' if history is None else 'store_open_failed'
    history = history or ()
    label, message = COPY.get((state, reason)) or COPY.get((state, None)) or (state, None)
    if state == 'feed_delayed' and facts['lag_s'] is not None:
        message = f"경기 자료가 {int(facts['lag_s'])}초 넘게 늦게 도착했어요. 추천이 실제 경기보다 늦을 수 있어요."
    refusal = (view or {}).get('reason')
    if (state == 'unsupported' and reason == 'policy_refused' and isinstance(refusal, str) and len(refusal) <= 60
            and re.search('[가-힣]', refusal) and not re.search('[A-Z_]{4,}', refusal)):
        message = refusal  # the policy's own short Korean sentence; a bare status code is never shown to a fan
    fix = correction(list(history), summary(rec) if shown else None) if pitch_key else None
    return {'schema': SCHEMA, 'state': state, 'reason': reason, 'detail': detail, 'label': label, 'message': message,
            'pitch_key': pitch_key, 'recommendation_shown': shown, 'correction': fix,
            'feed': {k: facts[k] for k in ('ok', 'delayed', 'lag_s', 'lag_max_s', 'lag_measured', 'lag_threshold_s', 'thresholds_provisional',
                                           'last_ok_at', 'failing_since', 'failing_for_s', 'received_at', 'provider_order')},
            'ledger': {'status': 'off' if facts['store'] == 'off' else 'store_unavailable' if facts['store'] == 'failed' else
                       (ledger or {}).get('status') or 'not_applicable',
                       'outcome': (ledger or {}).get('outcome'), 'why': (ledger or {}).get('why'),
                       'store_ok': facts['store_ok']}}
