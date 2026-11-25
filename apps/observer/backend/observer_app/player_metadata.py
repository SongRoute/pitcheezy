"""Optional local display names. IDs are join/display keys, never model inputs."""
from datetime import date
import json
from pathlib import Path


def load_player_names(path=None):
    """Read a local ``{MLB_ID: name}`` JSON; never perform a network lookup."""
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError('Player metadata must be an ID-to-name object')
    names = {}
    for player_id, name in raw.items():
        if not str(player_id).isdigit() or int(player_id) <= 0 or not isinstance(name, str) or not name.strip():
            raise ValueError('Player metadata requires positive MLB IDs and nonempty names')
        names[str(int(player_id))] = name.strip()
    return names


def player_label(player_id, role, names):
    return names.get(str(int(player_id))) or f'{role} #{int(player_id)}'


PREGAME_SCHEMA = 'pitcheezy-player-pregame-v1'
PREGAME_BASIS = 'regular_season_through_previous_day'


def _stat(value):
    if value is None or isinstance(value, bool):
        return None
    return str(value) if isinstance(value, (str, int, float)) and str(value).strip() else None


def load_pregame(snapshot_dir, game_pk, official_date):
    """Validated pre-game stats cache ``<dir>/<game_pk>.json`` (data job 032), or None.

    The whole file is rejected (None) unless it is for this exact game, uses the regular-season-through-previous-day
    basis and every ``as_of`` is strictly before the game's official date. Read-only; never a network lookup and
    never the feed's ``seasonStats`` (post-game cumulative)."""
    if not snapshot_dir or not official_date:
        return None
    path = Path(snapshot_dir)/f'{int(game_pk)}.json'
    try:
        raw = json.loads(path.read_text())
        game_day = date.fromisoformat(str(official_date)[:10])

        def before(value):
            return date.fromisoformat(str(value)) < game_day
        if not (isinstance(raw, dict) and raw.get('schema') == PREGAME_SCHEMA and raw.get('game_pk') == int(game_pk)
                and raw.get('date_basis') == PREGAME_BASIS and before(raw.get('as_of'))
                and isinstance(raw.get('players'), dict)):
            return None
        players = {}
        for player_id, entry in raw['players'].items():
            if not str(player_id).isdigit() or int(player_id) <= 0 or not isinstance(entry, dict) or not before(entry.get('as_of')):
                return None
            batting, pitching = entry.get('batting') or {}, entry.get('pitching') or {}
            if not isinstance(batting, dict) or not isinstance(pitching, dict):
                return None
            players[str(int(player_id))] = {
                'batting': {k: _stat(batting.get(k)) for k in ('avg', 'ops')} if batting else None,
                'pitching': {'era': _stat(pitching.get('era'))} if pitching else None,
                'source': entry.get('source') if isinstance(entry.get('source'), str) else None,
                'as_of': str(entry['as_of'])}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return {'as_of': str(raw['as_of']), 'date_basis': PREGAME_BASIS, 'players': players}
