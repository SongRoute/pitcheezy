"""Owner switches for the fan service: each feature can be turned off without a rebuild.

``features()`` is ``live_config.json`` ``features`` over ``DEFAULTS`` with environment overrides on top. The
environment is read again on every call (a handful of ``os.environ`` lookups), so a switch needs a restart of
the server process at most, never a web build: the fan payload carries ``public()`` and the screen obeys it.

Everything here is PROVISIONAL configuration (first settings for the 2026 World Series dry run), not a validated
limit. An invalid environment value never breaks serving: the configured value stays and one warning is logged.

Defaults (owner decision 2026-10-11): every block is OFF. The public World Series service publishes the
``/api/service/**`` bodies as static files that every fan reads at one delay (0), so a per-viewer delay exists only
on the direct path to this server (phone demo, internal) and is switched on by ``PITCHEEZY_OBSERVER_VIEWER_DELAY=1``.
With ``viewer_delay`` off its other keys are inert: ``?delay=`` is ignored on the game, list and tap routes (the tap
route answers 404), nothing is recorded at receipt and the public subset reports no prompt and no fold.

Only ``viewer_delay`` has behaviour in this module's callers so far; ``profile_reference`` and
``step5_expectation`` are carried for the code that consumes them.

Environment (0/1 unless noted):
  PITCHEEZY_OBSERVER_VIEWER_DELAY            1 = viewer delay on the fan screen (``/api/service/games/{pk}?delay=``)
  PITCHEEZY_OBSERVER_VIEWER_DELAY_DEFAULT_S  seconds: the delay a device starts with before it is synced
  PITCHEEZY_OBSERVER_VIEWER_SYNC_PROMPT      ask once to sync on the first open of a live game
  PITCHEEZY_OBSERVER_PROFILE_REFERENCE       profile reference block
  PITCHEEZY_OBSERVER_PROFILE_SOURCES         ``os.pathsep``-separated paths
  PITCHEEZY_OBSERVER_STEP5_EXPECTATION       step-5 expectation block
  PITCHEEZY_OBSERVER_STEP5_LEDGER            path
  PITCHEEZY_OBSERVER_STEP5_ALLOW_REPLAYED    allow a replayed ledger
"""
from __future__ import annotations

import copy
import logging
import math
import os

LOGGER = logging.getLogger(__name__)
PROVISIONAL = True
DEFAULTS = {
    'viewer_delay': {'enabled': False, 'default_s': 30, 'sync_prompt': True, 'fold_actual_until_synced': True,
                     'record_at_receipt': True},
    'profile_reference': {'enabled': False, 'profile_sources': []},
    'step5_expectation': {'enabled': False, 'ledger': None, 'allow_replayed': False},
}
_warned = set()


def _warn(name, value):
    if (name, value) not in _warned:  # once per bad value, not once per request
        _warned.add((name, value))
        LOGGER.warning('ignoring invalid %s=%r; the configured value stays', name, value)


def _flag(name, current):
    value = os.environ.get(name)
    if value is None or value == '':
        return current
    if value in ('0', '1'):
        return value == '1'
    _warn(name, value)
    return current


def _seconds(name, current):
    value = os.environ.get(name)
    if value is None or value == '':
        return current
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not (math.isfinite(number) and 0 <= number <= 600):
        _warn(name, value)
        return current
    return int(number) if number == int(number) else number


def _configured(live_config):
    """``DEFAULTS`` with the well-typed values of ``live_config['features']``; anything else keeps the default."""
    out = copy.deepcopy(DEFAULTS)
    given = live_config.get('features') if isinstance(live_config, dict) else None
    for block, defaults in DEFAULTS.items():
        values = given.get(block) if isinstance(given, dict) else None
        for key, default in defaults.items():
            if not isinstance(values, dict) or key not in values:
                continue
            value = values[key]
            if isinstance(default, bool):
                ok = isinstance(value, bool)
            elif key == 'default_s':
                ok = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 600
            elif key == 'profile_sources':
                ok = isinstance(value, list) and all(isinstance(item, str) for item in value)
            else:  # ledger: a path or None
                ok = value is None or isinstance(value, str)
            if ok:
                out[block][key] = copy.deepcopy(value)
            else:
                _warn(f'live_config.features.{block}.{key}', repr(value))
    return out


def features(live_config=None):
    """The switches in force now: config, then environment. Never raises."""
    if live_config is None:
        from .live_feed import LIVE_CONFIG as live_config
    out = _configured(live_config)
    viewer, profile, step5 = out['viewer_delay'], out['profile_reference'], out['step5_expectation']
    viewer['enabled'] = _flag('PITCHEEZY_OBSERVER_VIEWER_DELAY', viewer['enabled'])
    viewer['default_s'] = _seconds('PITCHEEZY_OBSERVER_VIEWER_DELAY_DEFAULT_S', viewer['default_s'])
    viewer['sync_prompt'] = _flag('PITCHEEZY_OBSERVER_VIEWER_SYNC_PROMPT', viewer['sync_prompt'])
    profile['enabled'] = _flag('PITCHEEZY_OBSERVER_PROFILE_REFERENCE', profile['enabled'])
    sources = os.environ.get('PITCHEEZY_OBSERVER_PROFILE_SOURCES')
    if sources:
        profile['profile_sources'] = [path for path in sources.split(os.pathsep) if path]
    step5['enabled'] = _flag('PITCHEEZY_OBSERVER_STEP5_EXPECTATION', step5['enabled'])
    step5['ledger'] = os.environ.get('PITCHEEZY_OBSERVER_STEP5_LEDGER') or step5['ledger']
    step5['allow_replayed'] = _flag('PITCHEEZY_OBSERVER_STEP5_ALLOW_REPLAYED', step5['allow_replayed'])
    return out | {'provisional': PROVISIONAL}


def public(current=None):
    """What the fan payload carries: booleans and the default seconds only, never a path."""
    current = current or features()
    viewer = current['viewer_delay']
    return {'provisional': PROVISIONAL,
            'viewer_delay': {'enabled': viewer['enabled'], 'default_s': viewer['default_s'],
                             'sync_prompt': viewer['enabled'] and viewer['sync_prompt'],
                             'fold_actual_until_synced': viewer['enabled'] and viewer['fold_actual_until_synced']},
            'profile_reference': {'enabled': current['profile_reference']['enabled']},
            'step5_expectation': {'enabled': current['step5_expectation']['enabled']}}
