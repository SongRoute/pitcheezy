"""OPS-06: ws_operate forwards PITCHEEZY_PLAYER_SNAPSHOT_DIR (pre-game season stats cache) to the publisher only when
it is set and valid. Unset must give exactly the pre-OPS-06 child environment. Synthetic paths only; no processes."""
import json
import sys
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS))
import ws_operate as ws  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    body = json.loads((OPS/'ws_ops_config.json').read_text())
    body |= {'code_dir': str(tmp_path/'code'), 'profile_sources': None}
    return body


def base_env(**extra):
    return {'HOME': '/home/x', 'LANG': 'C', **extra}


def cache_dir(tmp_path):
    path = tmp_path/'outside'/'cache'
    path.mkdir(parents=True)
    return path


def test_config_default_is_null():
    assert json.loads((OPS/'ws_ops_config.json').read_text())['player_snapshot_dir'] is None


def test_absent_when_neither_config_nor_env(cfg, tmp_path):
    child = ws.child_env(cfg, tmp_path/'run', False, False, env=base_env())
    assert ws.SNAPSHOT_ENV not in child
    assert ws.player_snapshot_dir(cfg, base_env()) == (None, None)
    # identical to the environment without the new key in the config at all (pre-OPS-06 config)
    old_cfg = {k: v for k, v in cfg.items() if not k.startswith('player_snapshot_dir')}
    assert child == ws.child_env(old_cfg, tmp_path/'run', False, False, env=base_env())


def test_forwarded_from_launching_shell(cfg, tmp_path):
    path = cache_dir(tmp_path)
    child = ws.child_env(cfg, tmp_path/'run', False, False, env=base_env(**{ws.SNAPSHOT_ENV: str(path)}))
    assert child[ws.SNAPSHOT_ENV] == str(path.resolve())
    assert ws.shown_env(child)[ws.SNAPSHOT_ENV] == str(path.resolve())


def test_forwarded_from_config_with_tilde_and_config_wins(cfg, tmp_path, monkeypatch):
    path = cache_dir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    cfg['player_snapshot_dir'] = '~/outside/cache'
    other = tmp_path/'other'
    other.mkdir()
    child = ws.child_env(cfg, tmp_path/'run', False, False, env=base_env(**{ws.SNAPSHOT_ENV: str(other)}))
    assert child[ws.SNAPSHOT_ENV] == str(path.resolve())


@pytest.mark.parametrize('kind', ['missing', 'file', 'in_git'])
def test_bad_path_is_not_forwarded_and_warns(cfg, tmp_path, kind):
    if kind == 'missing':
        path = tmp_path/'nope'
    elif kind == 'file':
        path = tmp_path/'a_file'
        path.write_text('x')
    else:
        (tmp_path/'repo'/'.git').mkdir(parents=True)
        path = tmp_path/'repo'/'cache'
        path.mkdir()
    env = base_env(**{ws.SNAPSHOT_ENV: str(path)})
    value, warning = ws.player_snapshot_dir(cfg, env)
    assert value is None and ws.SNAPSHOT_ENV in warning
    assert ws.SNAPSHOT_ENV not in ws.child_env(cfg, tmp_path/'run', False, False, env=env)
