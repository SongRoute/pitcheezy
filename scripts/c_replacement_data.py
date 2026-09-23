"""Bounded, retrospective MLB roster/workload evidence for fixed A DEV decisions.

`prepare` freezes decisions without any network access. `collect` reads that freeze,
stores immutable HTTP response bytes, and never treats a dated roster query as proof
of actual bullpen availability at the historical decision time.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/EXP-C-ROSTER-001.json'
API = 'https://statsapi.mlb.com/api/v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def load_inputs(cfg: dict):
    sm_path = ROOT / cfg['source_selection_manifest']
    sm = json.loads(sm_path.read_text())
    parquet = Path(sm['dataset_paths']['dev'])
    if sha(parquet) != sm['sha256']['dev_full_games.parquet']:
        raise ValueError('A selected DEV parquet hash mismatch')
    a_path = Path(cfg['source_event_packets'])
    a_manifest = json.loads(Path(cfg['source_event_manifest']).read_text())
    if sha(a_path) != a_manifest['output_sha256']:
        raise ValueError('A event packet hash mismatch')
    frame = pd.read_parquet(parquet)
    games = sorted({int(x['game_pk']) for games in sm['selection']['dev'].values() for x in games})
    if len(games) != cfg['max_selected_games'] or set(frame.game_pk.astype(int)) != set(games):
        raise ValueError('DEV selection is not the pinned six games')
    if not pd.to_datetime(frame.game_date).dt.year.eq(2025).all():
        raise ValueError('2025 DEV only')
    return frame, json.loads(a_path.read_text()), games, sm_path, parquet, a_path


def select_decisions(frame: pd.DataFrame, events: dict, games: list[int]) -> list[dict]:
    by_id = {f"{int(r.game_pk)}:{int(r.at_bat_number)}:{int(r.pitch_number)}": r
             for _, r in frame.iterrows()}
    if len(by_id) != len(frame):
        raise ValueError('duplicate pitch keys')
    out = []
    for game in games:
        rows = frame.loc[frame.game_pk.eq(game)].sort_values(['at_bat_number', 'pitch_number'])
        eligible = []
        for sub in events['observed_substitutions']:
            if sub['game_pk'] != game:
                continue
            pid = sub['first_observed_pitch_id']
            r = by_id.get(pid)
            if r is None:
                raise ValueError(f'A substitution key absent: {pid}')
            first_half_pa = int(rows.loc[rows.inning.eq(r.inning) & rows.inning_topbot.eq(r.inning_topbot),
                                          'at_bat_number'].min())
            if (int(r.pitch_number) == 1 and int(r.outs_when_up) == 0 and int(r.inning) <= 8
                    and int(r.at_bat_number) == first_half_pa):
                eligible.append((int(r.at_bat_number), pid, sub, r))
        if not eligible:
            out.append({'game_pk': game, 'status': 'missing',
                        'missing_reason': 'no_observed_fresh_half_inning_change_through_inning_8'})
            continue
        _, pid, sub, r = min(eligible, key=lambda x: (x[0], x[1]))
        prior = rows.loc[(rows.at_bat_number < r.at_bat_number) |
                         (rows.at_bat_number.eq(r.at_bat_number) & rows.pitch_number.lt(r.pitch_number))]
        fielding = str(r.fielding_team)
        own_prior = prior.loc[prior.fielding_team.eq(fielding)]
        if not len(own_prior) or int(own_prior.iloc[-1].pitcher) != sub['outgoing_pitcher']:
            raise ValueError('keep pitcher does not match preceding team pitch')
        same_game_counts = {str(int(k)): int(v) for k, v in own_prior.groupby('pitcher').size().items()}
        sides = {'Top': 'home', 'Bot': 'away'}
        side = sides[str(r.inning_topbot)]
        if side != ('home' if fielding == str(r.home_team) else 'away'):
            raise ValueError('fielding side mismatch')
        # A first observed substituted pitch is a retrospective proxy for decision time.
        # Its actual timestamp and exact manager knowledge are unavailable.
        out.append({
            'game_pk': game, 'status': 'selected', 'game_date': str(pd.Timestamp(r.game_date).date()),
            'decision_first_observed_pitch_id': pid, 'decision_timestamp': None,
            'decision_time_status': 'first_observed_pitch_proxy_actual_decision_time_unknown',
            'fielding_team': fielding, 'fielding_side': side,
            'keep_pitcher_id': int(sub['outgoing_pitcher']),
            'observed_incoming_pitcher_id_audit_only': int(sub['incoming_pitcher']),
            'state_at_first_observed_pitch': {
                'inning': int(r.inning), 'half': str(r.inning_topbot),
                'outs': int(r.outs_when_up), 'bases': int(r.bases),
                'home_score': int(r.home_score), 'away_score': int(r.away_score),
                'balls': int(r.balls), 'strikes': int(r.strikes),
                'current_batter_id': int(r.batter), 'current_batter_stand': str(r.stand),
            },
            'same_game_prior_pitch_counts': same_game_counts,
            'same_game_prior_pitcher_ids': sorted(int(k) for k in same_game_counts),
            'observed_prior_batter_ids': [int(x) for x in prior.loc[prior.fielding_team.eq(fielding), 'batter'].drop_duplicates()],
            'lineup_as_of': None,
            'lineup_missing_reason': 'Complete slot-preserving batting order and predecision substitutions not verified from predecision evidence',
            'selection_exposure': 'Conditioned on observed replacement; descriptive input example, not policy-effect sample',
        })
    return out


class Fetcher:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}

    def get(self, url: str) -> dict | None:
        key = hashlib.sha256(url.encode()).hexdigest()[:20]
        path = self.folder / f'{key}.json'
        meta = self.folder / f'{key}.meta.json'
        if path.exists():
            record = json.loads(meta.read_text())
            if record['url'] != url or record['sha256'] != sha(path):
                raise ValueError('immutable cache identity mismatch')
            self.records[url] = record
            return json.loads(path.read_text())
        if meta.exists():
            record = json.loads(meta.read_text())
            self.records[url] = record
            return None
        acquired = datetime.now(timezone.utc).isoformat()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'pitcheezy-research/1.0'}), timeout=20) as response:
                raw = response.read()
                status = response.status
            obj = json.loads(raw)
            path.write_bytes(raw)
            record = {'url': url, 'acquired_at_utc': acquired, 'http_status': status,
                      'sha256': sha(path), 'path': str(path), 'source': 'official_MLB_StatsAPI',
                      'exposure_time': 'retrieved_now_historical_response_not_point_in_time_archive'}
        except Exception as exc:
            record = {'url': url, 'acquired_at_utc': acquired, 'http_status': None,
                      'sha256': None, 'path': None, 'error': type(exc).__name__ + ': ' + str(exc),
                      'source': 'official_MLB_StatsAPI'}
            obj = None
        write_new(meta, record)
        self.records[url] = record
        return obj


def selected_metadata(fetch: Fetcher, decisions: list[dict]) -> dict[int, dict]:
    ids = ','.join(str(x['game_pk']) for x in decisions)
    data = fetch.get(f'{API}/schedule?gamePks={ids}&sportId=1')
    if data is None:
        return {}
    games = {int(g['gamePk']): g for day in data['dates'] for g in day['games']}
    for d in decisions:
        if d['status'] != 'selected':
            continue
        g = games.get(d['game_pk'])
        if g is None or g['officialDate'] != d['game_date'] or g['season'] != '2025':
            raise ValueError('selected game metadata identity mismatch')
    return games


def prior_games(fetch: Fetcher, team_id: int, day: date, lookback: int) -> list[tuple[str, int]] | None:
    start = day - timedelta(days=lookback)
    end = day - timedelta(days=1)
    url = f'{API}/schedule?sportId=1&teamId={team_id}&startDate={start}&endDate={end}&gameType=R'
    payload = fetch.get(url)
    if payload is None:
        return None
    games = []
    for block in payload.get('dates', []):
        for g in block['games']:
            d = date.fromisoformat(g['officialDate'])
            if not start <= d < day or g['season'] != '2025':
                raise ValueError('future/prior-season game crossed cutoff')
            if team_id not in [g['teams'][side]['team']['id'] for side in ('home', 'away')]:
                raise ValueError('prior schedule team mismatch')
            if g.get('status', {}).get('abstractGameState') != 'Final':
                raise ValueError('prior schedule game is not reported Final')
            games.append((g['officialDate'], int(g['gamePk'])))
    return sorted(set(games))


def workload(fetch: Fetcher, team_id: int, games: list[tuple[str, int]]) -> tuple[dict[int, list[dict]], list[str]]:
    appearances: dict[int, list[dict]] = {}
    failures = []
    for game_date, pk in games:
        url = f'{API}/game/{pk}/boxscore'
        box = fetch.get(url)
        if box is None:
            failures.append(url)
            continue
        teams = [box['teams'][side] for side in ('home', 'away') if box['teams'][side]['team']['id'] == team_id]
        if len(teams) != 1:
            raise ValueError('prior boxscore team mismatch')
        team = teams[0]
        for pitcher_id in team['pitchers']:
            info = team['players'].get(f'ID{pitcher_id}', {})
            stats = info.get('stats', {}).get('pitching', {})
            appearances.setdefault(int(pitcher_id), []).append({
                'game_pk': pk, 'official_date': game_date,
                'pitches': stats.get('numberOfPitches'), 'outs': stats.get('outs'),
                'source_url': url,
            })
    return appearances, failures


def collect(cfg: dict, decisions: list[dict], folder: Path) -> dict:
    fetch = Fetcher(folder / 'raw')
    games = selected_metadata(fetch, decisions)
    packet = []
    for d in decisions:
        if d['status'] != 'selected':
            packet.append(dict(d))
            continue
        item = dict(d)
        g = games.get(d['game_pk'])
        if g is None:
            item.update({'replacement_screen': None, 'missing_reasons': ['selected_schedule_unavailable']})
            packet.append(item)
            continue
        team_id = int(g['teams'][d['fielding_side']]['team']['id'])
        item['fielding_team_id'] = team_id
        roster_url = f"{API}/teams/{team_id}/roster?rosterType=active&date={d['game_date']}"
        roster = fetch.get(roster_url)
        prior = prior_games(fetch, team_id, date.fromisoformat(d['game_date']), cfg['lookback_calendar_days'])
        appearances, failures = workload(fetch, team_id, prior) if prior is not None else ({}, [])
        reasons = []
        if roster is None:
            reasons.append('dated_roster_unavailable')
        if prior is None:
            reasons.append('prior_schedule_unavailable')
        if failures:
            reasons.append('prior_boxscore_incomplete')
        roster_pitchers = [] if roster is None else [x for x in roster.get('roster', []) if x.get('position', {}).get('type') == 'Pitcher']
        if roster is not None and not roster_pitchers:
            reasons.append('dated_roster_has_no_pitchers')
        seen = set(d['same_game_prior_pitcher_ids'])
        candidates = []
        for r in sorted(roster_pitchers, key=lambda x: int(x['person']['id'])):
            pid = int(r['person']['id'])
            if pid == d['keep_pitcher_id'] or pid in seen:
                continue
            apps = appearances.get(pid, [])
            most_recent = max((x['official_date'] for x in apps), default=None)
            rest = (date.fromisoformat(d['game_date']) - date.fromisoformat(most_recent)).days - 1 if most_recent else None
            candidates.append({
                'pitcher_id': pid, 'name': r['person'].get('fullName'),
                'roster_listing': 'active_on_queried_historical_date_retrospective',
                'manager_available_at_decision': None,
                'prior_7d_appearances': apps if prior is not None and not failures else None,
                'prior_7d_pitches': sum(x['pitches'] for x in apps) if prior is not None and not failures and all(x['pitches'] is not None for x in apps) else None,
                'full_rest_calendar_days_since_last_appearance': rest if prior is not None and not failures else None,
                'rest_missing_reason': ('no_appearance_in_seven_day_window' if not apps else None) if prior is not None and not failures else 'prior_workload_incomplete',
            })
        item['replacement_screen'] = candidates if roster is not None else None
        item['keep_prior_7d_appearances'] = appearances.get(d['keep_pitcher_id'], []) if prior is not None and not failures else None
        item['prior_window_game_ids'] = [pk for _, pk in prior] if prior is not None else None
        item['prior_window_boxscore_failures'] = failures
        item['roster_source_url'] = roster_url
        item['actual_eligible_replacement_candidates'] = None
        item['missing_reasons'] = reasons + ['contemporaneous_roster_publication_and_manager_availability_unverified',
                                               'complete_predecision_batting_order_unverified']
        packet.append(item)
    return {'schema_version': 1, 'experiment_id': cfg['experiment_id'], 'decisions': packet,
            'raw_response_records': sorted(fetch.records.values(), key=lambda x: x['url']),
            'future_cutoff_assertion': 'Only selected-game schedule identifiers and predecision Statcast rows; workload boxscores are reported Final with official_date strictly before selected game date. Selected-game final boxscore/later appearances excluded. Actual prior game completion time was not independently archived/verified.',
            'policy_value': None, 'actual_manager_availability': None}


def reconstruct_predecision_lineup(plays: list[dict], decision: dict) -> dict:
    """Require nine stable, previously observed batting slots; abstain on changes."""
    at_bat = int(decision['decision_first_observed_pitch_id'].split(':')[1])
    index = at_bat - 1
    current = next((p for p in plays if p['about']['atBatIndex'] == index), None)
    if current is None or int(current['matchup']['batter']['id']) != decision['state_at_first_observed_pitch']['current_batter_id']:
        return {'lineup_as_of': None, 'missing_reason': 'play_by_play_current_pa_identity_mismatch'}
    if int(current['matchup']['pitcher']['id']) != decision['observed_incoming_pitcher_id_audit_only']:
        return {'lineup_as_of': None, 'missing_reason': 'play_by_play_current_pitcher_identity_mismatch'}
    half = 'top' if decision['state_at_first_observed_pitch']['half'] == 'Top' else 'bottom'
    if current['about']['halfInning'] != half:
        return {'lineup_as_of': None, 'missing_reason': 'play_by_play_half_mismatch'}
    prior = [p for p in plays if p['about']['atBatIndex'] < index]
    if any(p['about']['atBatIndex'] != i for i, p in enumerate(prior)):
        return {'lineup_as_of': None, 'missing_reason': 'play_by_play_noncontiguous_pa_index'}
    batting = [p for p in prior if p['about']['halfInning'] == half]
    offensive_actions = [e for p in prior for e in p.get('playEvents', [])
                         if e.get('details', {}).get('eventType') in
                         ('offensive_substitution', 'defensive_substitution', 'runner_placed_on_base')]
    if offensive_actions:
        return {'lineup_as_of': None, 'missing_reason': 'predecision_substitution_or_runner_action_requires_slot_verification',
                'predecision_batting_pa_count': len(batting)}
    if len(batting) < 9:
        return {'lineup_as_of': None, 'missing_reason': 'fewer_than_nine_predecision_batting_pas',
                'predecision_batting_pa_count': len(batting)}
    slots: list[dict | None] = [None] * 9
    for i, play in enumerate(batting):
        batter_id = int(play['matchup']['batter']['id'])
        stand = play['matchup'].get('batSide', {}).get('code')
        slot = i % 9
        if stand not in ('L', 'R'):
            return {'lineup_as_of': None, 'missing_reason': 'batter_stance_unknown'}
        if slots[slot] is None:
            slots[slot] = {'slot': slot + 1, 'batter_id': batter_id, 'stand': stand}
        elif slots[slot]['batter_id'] != batter_id or slots[slot]['stand'] != stand:
            return {'lineup_as_of': None, 'missing_reason': 'batting_slot_or_stance_changed_before_decision'}
    next_slot = len(batting) % 9
    if slots[next_slot]['batter_id'] != int(current['matchup']['batter']['id']):
        return {'lineup_as_of': None, 'missing_reason': 'current_batter_does_not_follow_reconstructed_order'}
    return {'lineup_as_of': {'ordered_slots': slots,
                            'next_slot_one_based': next_slot + 1,
                            'current_batter_id': slots[next_slot]['batter_id'],
                            'current_batter_stand': slots[next_slot]['stand'],
                            'source': 'strictly_predecision_play_by_play_observed_batting_cycle',
                            'historical_feed_retrieved_now': True},
            'missing_reason': None, 'predecision_batting_pa_count': len(batting)}


def collect_lineup_supplement(cfg: dict, frozen: dict, folder: Path) -> None:
    existing = folder / 'replacement_packets.json'
    original = json.loads(existing.read_text())
    decisions = original['decisions']
    fetch = Fetcher(folder / 'raw_lineup')
    items = []
    for d in decisions:
        if d['status'] != 'selected':
            items.append({'game_pk': d['game_pk'], 'lineup_as_of': None,
                          'missing_reason': d['missing_reason']})
            continue
        url = f"{API}/game/{d['game_pk']}/playByPlay"
        feed = fetch.get(url)
        if feed is None:
            reconstructed = {'lineup_as_of': None, 'missing_reason': 'play_by_play_fetch_failed'}
        else:
            reconstructed = reconstruct_predecision_lineup(feed['allPlays'], d)
        items.append({'game_pk': d['game_pk'], 'decision_first_observed_pitch_id': d['decision_first_observed_pitch_id'],
                      'source_url': url, **reconstructed})
    supplement = {'schema_version': 1, 'experiment_id': cfg['experiment_id'],
                  'selection_sha256': sha(ROOT / 'results/EXP-C-ROSTER-001/selection.json'),
                  'original_packet_sha256': sha(existing),
                  'exposure_rule': 'Finalized historical play-by-play feed retrieved now; parser uses only complete PAs with atBatIndex strictly before selected first-pitch PA and current-PA identity for verification. Raw feed contains later events, quarantined from reconstruction.',
                  'lineups': items, 'raw_response_records': list(fetch.records.values()),
                  'actual_manager_availability': None, 'policy_value': None}
    output = folder / 'lineup_supplement.json'
    write_new(output, supplement)
    summary = {'experiment_id': cfg['experiment_id'], 'selection_sha256': supplement['selection_sha256'],
               'original_packet_sha256': supplement['original_packet_sha256'],
               'script_sha256': sha(Path(__file__)), 'output_path': str(output), 'output_sha256': sha(output),
               'verified_lineup_count': sum(x['lineup_as_of'] is not None for x in items),
               'raw_response_count': len(fetch.records), 'actual_manager_availability': None, 'policy_value': None}
    write_new(folder / 'lineup_supplement_manifest.json', summary)
    write_new(ROOT / 'results/EXP-C-ROSTER-001/lineup_supplement_manifest.json', summary)
    print(json.dumps(summary, indent=2))


def reconstruct_predecision_lineup_v2(plays: list[dict], decision: dict) -> dict:
    """Follow completed batter PAs and structured prior substitutions only."""
    index = int(decision['decision_first_observed_pitch_id'].split(':')[1]) - 1
    by_index = {p['about']['atBatIndex']: p for p in plays}
    if len(by_index) != len(plays) or list(sorted(by_index))[:index + 1] != list(range(index + 1)):
        return {'lineup_as_of': None, 'missing_reason': 'play_by_play_noncontiguous_pa_index'}
    current = by_index[index]
    state = decision['state_at_first_observed_pitch']
    half = 'top' if state['half'] == 'Top' else 'bottom'
    if (current['about']['halfInning'] != half or
            int(current['matchup']['batter']['id']) != state['current_batter_id'] or
            int(current['matchup']['pitcher']['id']) != decision['observed_incoming_pitcher_id_audit_only']):
        return {'lineup_as_of': None, 'missing_reason': 'current_pa_identity_mismatch'}
    first_pitch = next((i for i, e in enumerate(current.get('playEvents', [])) if e.get('isPitch')), None)
    if first_pitch is None:
        return {'lineup_as_of': None, 'missing_reason': 'current_pa_first_pitch_absent'}
    pre_pitch_actions = current['playEvents'][:first_pitch]
    ambiguous_current = [e.get('details', {}).get('eventType') for e in pre_pitch_actions
                         if e.get('isSubstitution') and e.get('details', {}).get('eventType') != 'pitching_substitution']
    if ambiguous_current:
        return {'lineup_as_of': None, 'missing_reason': 'current_pa_pre_first_pitch_nonpitching_substitution',
                'current_pa_pre_first_pitch_action_types': ambiguous_current}

    slots: list[dict | None] = [None] * 9
    completed_batter_pas = 0
    excluded_non_pa = []
    applied_substitutions = []
    for pa_index in range(index):
        play = by_index[pa_index]
        play_half = play['about']['halfInning']
        if not play['about'].get('isComplete'):
            return {'lineup_as_of': None, 'missing_reason': 'prior_play_not_complete'}
        for event in play.get('playEvents', []):
            kind = event.get('details', {}).get('eventType')
            relevant = (kind == 'offensive_substitution' and play_half == half or
                        kind == 'defensive_substitution' and play_half != half)
            if not relevant:
                continue
            order = str(event.get('battingOrder', ''))
            replaced = event.get('replacedPlayer', {}).get('id')
            new_id = event.get('player', {}).get('id')
            if not re.fullmatch(r'[1-9]\d{2}', order) or replaced is None or new_id is None:
                return {'lineup_as_of': None, 'missing_reason': 'prior_substitution_lacks_structured_slot_or_identity'}
            slot = int(order[0]) - 1
            if slots[slot] is None or slots[slot]['batter_id'] != int(replaced):
                return {'lineup_as_of': None, 'missing_reason': 'prior_substitution_replaced_identity_unverified'}
            slots[slot] = {'slot': slot + 1, 'batter_id': int(new_id), 'observed_stands': []}
            applied_substitutions.append({'at_bat_index': pa_index, 'slot': slot + 1,
                                          'replaced_player_id': int(replaced), 'new_player_id': int(new_id)})
        if play_half != half:
            continue
        event_type = str(play.get('result', {}).get('eventType') or '')
        if event_type.startswith(('caught_stealing', 'pickoff')):
            excluded_non_pa.append({'at_bat_index': pa_index, 'event_type': event_type})
            continue
        if not event_type:
            return {'lineup_as_of': None, 'missing_reason': 'prior_batting_pa_terminal_event_unknown'}
        slot = completed_batter_pas % 9
        batter_id = int(play['matchup']['batter']['id'])
        stand = play['matchup'].get('batSide', {}).get('code')
        if stand not in ('L', 'R'):
            return {'lineup_as_of': None, 'missing_reason': 'prior_batter_observed_stance_unknown'}
        if slots[slot] is None:
            slots[slot] = {'slot': slot + 1, 'batter_id': batter_id, 'observed_stands': [stand]}
        elif slots[slot]['batter_id'] != batter_id:
            return {'lineup_as_of': None, 'missing_reason': 'batting_slot_changed_without_verified_substitution'}
        else:
            slots[slot]['observed_stands'] = sorted(set(slots[slot]['observed_stands'] + [stand]))
        completed_batter_pas += 1
    if completed_batter_pas < 9 or any(s is None for s in slots):
        return {'lineup_as_of': None, 'missing_reason': 'fewer_than_nine_verified_batting_pas'}
    next_slot = completed_batter_pas % 9
    if slots[next_slot]['batter_id'] != state['current_batter_id']:
        return {'lineup_as_of': None, 'missing_reason': 'current_batter_not_verified_next_slot'}
    return {'lineup_as_of': {'ordered_slots': [{**s, 'stand_for_substitute': None} for s in slots],
                            'next_slot_one_based': next_slot + 1,
                            'current_batter_id': state['current_batter_id'],
                            'current_batter_observed_stand': state['current_batter_stand'],
                            'source': 'complete_predecision_batter_PAs_plus_structured_predecision_substitutions',
                            'matchup_stances_for_substitutes_verified': False},
            'missing_reason': None, 'completed_predecision_batter_pa_count': completed_batter_pas,
            'excluded_non_pa_plays': excluded_non_pa,
            'applied_predecision_substitutions': applied_substitutions}


def collect_lineup_supplement_v2(cfg: dict, folder: Path) -> None:
    original = folder / 'replacement_packets.json'
    first_supplement = folder / 'lineup_supplement.json'
    decisions = json.loads(original.read_text())['decisions']
    fetch = Fetcher(folder / 'raw_lineup')
    items = []
    for d in decisions:
        url = f"{API}/game/{d['game_pk']}/playByPlay"
        feed = fetch.get(url)
        result = ({'lineup_as_of': None, 'missing_reason': 'cached_play_by_play_unavailable'} if feed is None else
                  reconstruct_predecision_lineup_v2(feed['allPlays'], d))
        items.append({'game_pk': d['game_pk'], 'decision_first_observed_pitch_id': d['decision_first_observed_pitch_id'],
                      'source_url': url, **result})
    payload = {'schema_version': 2, 'experiment_id': cfg['experiment_id'],
               'original_packet_sha256': sha(original), 'first_supplement_sha256': sha(first_supplement),
               'selection_sha256': sha(ROOT / 'results/EXP-C-ROSTER-001/selection.json'),
               'method': 'Exclude complete non-batter PA plays (caught stealing/pickoff); apply only structured substitutions in completed earlier plays; reject nonpitching substitution before current first pitch; do not read current/later outcomes.',
               'lineups': items, 'raw_response_records': list(fetch.records.values()),
               'actual_manager_availability': None, 'policy_value': None}
    output = folder / 'lineup_supplement_v2.json'
    write_new(output, payload)
    summary = {'experiment_id': cfg['experiment_id'], 'output_path': str(output), 'output_sha256': sha(output),
               'script_sha256': sha(Path(__file__)), 'original_packet_sha256': payload['original_packet_sha256'],
               'first_supplement_sha256': payload['first_supplement_sha256'],
               'verified_lineup_count': sum(x['lineup_as_of'] is not None for x in items),
               'raw_response_count': len(fetch.records), 'new_raw_fetch_count': 0,
               'actual_manager_availability': None, 'policy_value': None}
    write_new(folder / 'lineup_supplement_v2_manifest.json', summary)
    write_new(ROOT / 'results/EXP-C-ROSTER-001/lineup_supplement_v2_manifest.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'collect', 'lineup-supplement', 'lineup-supplement-v2'])
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    frame, events, games, sm_path, parquet, a_path = load_inputs(cfg)
    folder = Path(cfg['output_dir'])
    freeze = ROOT / 'results/EXP-C-ROSTER-001/selection.json'
    if args.phase == 'prepare':
        selected = select_decisions(frame, events, games)
        write_new(freeze, {'experiment_id': cfg['experiment_id'], 'phase': 'prepared_before_roster_extraction',
                           'selection': selected, 'sha256': {'config': sha(CONFIG), 'source_selection_manifest': sha(sm_path),
                                                            'source_dev_parquet': sha(parquet), 'source_event_packets': sha(a_path)}})
        print(json.dumps({'selection_path': str(freeze), 'selected': sum(x['status'] == 'selected' for x in selected)}))
        return
    frozen = json.loads(freeze.read_text())
    if frozen['sha256'] != {'config': sha(CONFIG), 'source_selection_manifest': sha(sm_path),
                            'source_dev_parquet': sha(parquet), 'source_event_packets': sha(a_path)}:
        raise ValueError('selection input changed after freeze')
    if frozen['selection'] != select_decisions(frame, events, games):
        raise ValueError('frozen decision selection changed')
    if args.phase == 'lineup-supplement':
        collect_lineup_supplement(cfg, frozen, folder)
        return
    if args.phase == 'lineup-supplement-v2':
        collect_lineup_supplement_v2(cfg, folder)
        return
    result = collect(cfg, frozen['selection'], folder)
    output = folder / 'replacement_packets.json'
    write_new(output, result)
    summary = {'experiment_id': cfg['experiment_id'], 'selection_sha256': sha(freeze),
               'config_sha256': sha(CONFIG), 'script_sha256': sha(Path(__file__)),
               'packet_path': str(output), 'packet_sha256': sha(output),
               'selected_count': sum(x['status'] == 'selected' for x in result['decisions']),
               'raw_response_count': len(result['raw_response_records']),
               'actual_manager_availability': None, 'policy_value': None,
               'source_exposure': 'retrospective_2026_retrieval_of_historical_MLB_StatsAPI'}
    write_new(folder / 'manifest.json', summary)
    write_new(ROOT / 'results/EXP-C-ROSTER-001/manifest.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
