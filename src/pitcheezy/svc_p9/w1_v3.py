"""W1 v3 token reader (EXP-SVC-P9-003, POST-HOC): RS-01 ``captured_native.inspect_tokens`` with two W1-side changes.

The RS-01 modules (``seq_shadow.captured_native``, ``captured_native_identity``) are NOT modified and NOT patched. This
module recomposes them: ``inspect_tokens`` and ``inspect_receipt_identity``/``_mixed_timeline`` are copied here with the
reverse-action function injected, and every other helper is imported unchanged. With ``reverse=_reverse_action`` and
``exclude_unmapped_bip=False`` the copy returns exactly what RS-01 returns (tests/test_svc_p9_w1_v3.py checks this on
synthetic and real-shaped bodies), so the only differences are the two below.

  EXCL     an earlier pitch whose in-play result is present but has no outcome10 label (e.g. events field_error,
           fielders_choice, catcher_interf, triple_play) gets status ``excluded_unmapped_label`` instead of
           ``native_unknown``. W1 v3 leaves it out of the context (a gap) instead of omitting every later pitch of the
           game. Training equivalence: the B_PSR9N6 training rows went through
           experiments/ws_campaign/source/abc_corpus.py project() (excluded_unmapped_label, :75-77) before
           ws_seq build_context_index, so those pitches are absent from every training context too (pitch_number is
           not renumbered there; the index walks by position).
           Only rows whose RS-01 reason is ``source_outcome_unverified`` and whose BIP helper shows
           description hit_into_play + a string event + no outcome10 label (the pre-count was known) are excluded.
  NOPITCH  in the reverse matchup chain, an automatic ball/strike (type ``no_pitch``, isPitch False, isSubstitution
           absent or False) keeps pitcher and batter. ASSUMPTION about the provider field (not proved): a pitch-clock
           automatic call never changes the matchup. Its count is not touched here (inspect_body reads every pitch's
           pre-count from the body as before). Training: Statcast automatic_ball / automatic_strike rows have no
           outcome10 label and are excluded by the same abc_corpus rule (oof3 receipt: 2,309 such rows), so W1 v3
           producing no token for them matches training.
           Checked 2026-11-19 (top-level keys only, no label): all 41 no_pitch events in the 31 E_PS + 8 wildcard
           finals have no top-level player/replacedPlayer key, no isSubstitution key and isPitch False.

Not covered by EXCL (conservative, left strict, independent review 2026-11-19): EXCL applies only to rows the RS-01
BIP helper marks (completed PA, final pitch in play with a BIP call code, event string present, no outcome10 label) and
whose RS-01 reason is ``source_outcome_unverified``. catcher_interf and triple_play (also unlabelled in training) are not
claimed as covered: such a row that the helper does not mark, or whose matchup/pre-count is unverified, still omits
every later pitch of the game, as in v2.
"""
from __future__ import annotations

from collections import Counter

from pitcheezy.seq_shadow.captured_diagnostic import _get, inspect_body
from pitcheezy.seq_shadow.captured_identity_bip import inspect_hypotheses
from pitcheezy.seq_shadow.captured_native import CONTRACT, TOKEN_FIELDS, _hex
from pitcheezy.seq_shadow.captured_native_identity import _ordered, _positive, _reverse_action, _views

VERSION = 'v3'
EXCLUDED = 'excluded_unmapped_label'
NOPITCH_ASSUMPTION = 'w1v3_automatic_call_keeps_matchup_assumption'


def reverse_action_v3(event, pitcher, batter):
    """NOPITCH, else RS-01 ``_reverse_action`` unchanged."""
    if (isinstance(event, dict) and event.get('type') == 'no_pitch' and event.get('isPitch') is False
            and event.get('isSubstitution', False) is False):
        return pitcher, batter, NOPITCH_ASSUMPTION
    return _reverse_action(event, pitcher, batter)


# ---- copy of captured_native_identity._mixed_timeline with ``reverse`` injected (body otherwise identical)
def _mixed_timeline(play, game_pk, pa, reverse):
    events = _get(play, 'playEvents')
    if not _ordered(events):
        return {}
    pitcher = _positive(_get(play, 'matchup', 'pitcher', 'id'))
    batter = _positive(_get(play, 'matchup', 'batter', 'id'))
    if pitcher is None or batter is None:
        return {}
    valid, transitions, result = True, [], {}
    for event in reversed(events):
        if event['isPitch'] is True:
            if valid:
                key = (game_pk, pa + 1, event['pitchNumber'])
                result[key] = {'identity_hypothesis': {'pitcher': pitcher, 'batter': batter,
                                                       'stand': None, 'throws': None},
                               'identity_provenance': {
                                   'rule': 'receipt_local_reverse_matchup',
                                   'anchor': 'same_receipt_play_matchup_end_view',
                                   'pitch_event_index': event['index'],
                                   'later_action_chain': list(reversed(transitions)),
                                   'hands': 'unverified_across_nonpitch_action',
                                   'accepted': False}}
            continue
        changed = reverse(event, pitcher, batter) if valid else None
        if changed is None:
            valid = False
        else:
            pitcher, batter, assumption = changed
            transitions.append({'index': event['index'], 'assumption': assumption})
    return result


# ---- copy of captured_native_identity.inspect_receipt_identity with ``reverse`` injected (body otherwise identical)
def inspect_receipt_identity(feed, *, game_pk, game_date, all_pitch_rows, reverse=reverse_action_v3):
    if (_get(feed, 'gameData', 'game', 'pk') != game_pk
            or _get(feed, 'gameData', 'datetime', 'officialDate') != game_date):
        raise ValueError('game/date mismatch')
    by_pa, conflict = _views(feed)
    baseline = {tuple(row['key']): row for row in all_pitch_rows}
    result = {}
    for pa, play in by_pa.items():
        if pa in conflict:
            continue
        events = _get(play, 'playEvents')
        if not isinstance(events, list) or not events:
            continue
        if all(isinstance(event, dict) and event.get('isPitch') is True for event in events):
            for event in events:
                number = event.get('pitchNumber')
                key = (game_pk, pa + 1, number) if type(number) is int else None
                row = baseline.get(key)
                if row and row['identity_hypothesis'] is not None:
                    result[key] = {'identity_hypothesis': row['identity_hypothesis'],
                                   'identity_provenance': {
                                       'rule': 'pitch_only_play_matchup',
                                       'anchor': 'same_receipt_play_matchup_view',
                                       'accepted': False}}
        else:
            result.update(_mixed_timeline(play, game_pk, pa, reverse))
    providers = Counter(
        event.get('playId') for play in by_pa.values()
        for event in (_get(play, 'playEvents') or [])
        if isinstance(event, dict) and event.get('isPitch') is True
        and isinstance(event.get('playId'), str) and event.get('playId'))
    for key in list(result):
        play = by_pa[key[1] - 1]
        events = _get(play, 'playEvents')
        event = next((e for e in events if isinstance(e, dict) and e.get('isPitch') is True
                      and e.get('pitchNumber') == key[2]), None)
        provider = event.get('playId') if event is not None else None
        if not isinstance(provider, str) or not provider or providers[provider] != 1:
            del result[key]
    return result


def unmapped_bip(helper):
    """EXCL test on the RS-01 BIP helper row: in play, event present, no outcome10 label."""
    return (isinstance(helper, dict) and helper.get('description_bip_hypothesis') == 'hit_into_play'
            and isinstance(helper.get('events_bip_hypothesis'), str) and helper.get('outcome10_bip_hypothesis') is None)


# ---- copy of captured_native.inspect_tokens with ``reverse`` injected and the EXCL status (body otherwise identical)
def inspect_tokens(feed, *, game_pk, game_date, receipt_seq, source_sha256, reverse=reverse_action_v3,
                   exclude_unmapped_bip=True):
    if type(receipt_seq) is not int or receipt_seq <= 0 or not _hex(source_sha256):
        raise ValueError('receipt identity required')
    base_rows = inspect_body(feed, game_pk=game_pk, game_date=game_date)
    identity_rows = inspect_hypotheses(feed, game_pk=game_pk, game_date=game_date)
    identity_by_key = inspect_receipt_identity(feed, game_pk=game_pk, game_date=game_date,
                                               all_pitch_rows=identity_rows, reverse=reverse)
    by_key = {}
    for row in base_rows:
        if row['key'] is not None:
            by_key.setdefault(tuple(row['key']), []).append(row)
    identity = {tuple(row['key']): row for row in identity_rows}
    provider_counts = Counter(row['provider_id'] for row in base_rows if row['provider_id'] is not None)
    result = []
    for key, observed in sorted(by_key.items()):
        row = observed[0]
        helper = identity.get(key)
        native, reason = None, None
        if len(observed) != 1 or row['provider_id'] is None or provider_counts[row['provider_id']] != 1:
            reason = 'physical_or_provider_identity_unverified'
        elif key not in identity_by_key:
            reason = 'pitch_time_matchup_unverified'
        elif row['type_state'] != 'known':
            reason = 'direct_type_unavailable'
        elif row['pre_count'] is None or not (0 <= row['pre_count'][0] <= 3 and 0 <= row['pre_count'][1] <= 2):
            reason = 'source_pre_count_unverified'
        else:
            nonbip = row['native_nonbip_candidate'] and row['outcome10_hypothesis'] is not None
            bip = (helper['description_bip_hypothesis'] == 'hit_into_play'
                   and helper['events_bip_hypothesis'] is not None
                   and helper['outcome10_bip_hypothesis'] is not None)
            if not (nonbip or bip):
                reason = 'source_outcome_unverified'
            else:
                matchup = identity_by_key[key]['identity_hypothesis']
                native = dict(zip(TOKEN_FIELDS, (
                    matchup['pitcher'], matchup['batter'], row['observed']['event_type_code'],
                    row['outcome10_hypothesis'] if nonbip else helper['outcome10_bip_hypothesis'],
                    *row['pre_count'])))
                if not (type(native['outcome10']) is int and 0 <= native['outcome10'] <= 9):
                    native, reason = None, 'source_outcome_unverified'
                else:
                    reason = 'source_native_six_fields_unpaired'
        status = 'native_candidate' if native is not None else 'native_unknown'
        if exclude_unmapped_bip and native is None and reason == 'source_outcome_unverified' and unmapped_bip(helper):
            status = EXCLUDED
        result.append({'key': key, 'provider_id': row['provider_id'] if len(observed) == 1 else None,
                       'receipt_seq': receipt_seq, 'source_sha256': source_sha256,
                       'input_contract': CONTRACT,
                       'status': status,
                       'reason': reason, 'native_token': native,
                       'identity_provenance': (identity_by_key[key]['identity_provenance']
                                               if key in identity_by_key else None),
                       'statcast_training_equivalent': False,
                       'provider_identity_scope': 'single_local_receipt',
                       'accepted': False})
    return result
