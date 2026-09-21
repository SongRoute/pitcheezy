"""Optional local display names. IDs are join/display keys, never model inputs."""
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
