"""Shared Step 5 feature path for the N1 (process A) and Step 5 2024 (process B) adapters.

Both accepted runs (``scripts/seq_shadow_runner.py`` for SEQ-001 N1, ``shadow_runner.py`` for the frozen 2024
heads) ride on the same pinned Step 5 source: ``features.pre_pitch_state`` (pre-pitch state of the pitch after the
last received one in ONE body), the committed strict-prior profile (``live_adapter.load_committed``) and
``features.build_candidates`` (17 actions x 91 raw columns -> 17 x 199 with the 2024 F1-design encoder).

What this module reproduces of ``pipeline.decide`` and what it leaves out
  * reproduced, in the same order and with the same reason strings: identity of the body (game, canonical sha),
    keyed position, open plate appearance, missing inputs, profile strictly prior to the body's officialDate,
    ``profile.support`` (general reason, eligible actions), ``build_candidates``;
  * left out on purpose: the receipt-clock / poll-gap / ``late`` gates, the append-only decision ledger, held
    duplicates and matchup quarantine. In RS-01 the prospective runner owns timing (intent before the call,
    PRE_RECEIPT only when the store still has no event for the slot) and first-write-wins per slot. The forecast
    is therefore a pure function of (body, profile, frozen model): the adapters keep no state and need no ledger.

Leakage. The only body read is the stored snapshot ``ctx.source_seq`` (sha-verified); a body that already shows the
target (at_bat_index, pitch_number) is refused (``BODY_CONTAINS_TARGET``). ``pre_pitch_state`` never reads
pitchData, hitData, end times or a play result (accepted Step 5 property).
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .adapters import PinMismatch, sha256_file, verify_pins
from .runner import Abstain
from . import store_source

ACTUAL_SCOPE = 'ACTUAL_S2_ONE_PASS'
PROFILE_ROLE = 'SERVICE_PROFILE_STRICTLY_PRIOR'
SUPERVISED = 'SUCCESS_USABLE_BOUND_TO_RECEIPT_AND_PROFILE_SHA256:'


# ---------------------------------------------------------------- pinned imports
def import_seq_shadow(spec: dict):
    """Verify every pinned source byte, then import ``pitcheezy.seq_shadow`` (stdlib only: config loader, hashing
    and the sha-pinned import of the Step 5 shadow modules) from the registered source root."""
    verify_pins(Path('/'), spec['source_pins'])
    root = str(Path(spec['seq_shadow_src_root']).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import pitcheezy.seq_shadow as SS
    want = Path(root)/'pitcheezy'/'seq_shadow'/'__init__.py'
    if Path(SS.__file__).resolve() != want.resolve():
        raise PinMismatch(f'pitcheezy.seq_shadow already imported from another origin: {SS.__file__}')
    return SS


@dataclass
class Step5Env:
    mods: object       # S (shadow_sources), G (shadow_guard), RUN (shadow_runner), sha256
    config: dict       # pinned shadow_config.json (encoder, MODELS registry, limits, step5/adapter dirs, pins)
    src: object        # F (features), P (pipeline), L, R, A (live_adapter), pins
    encoder: dict      # 2024 F1-design encoder
    encoder_sha256: str

    @property
    def RUN(self):
        return self.mods.RUN


def load_step5(SS, step5_shadow: dict) -> Step5Env:
    """The accepted Step 5 source through the accepted loaders: ``seq_shadow.load_step5_shadow`` (three reviewed
    shadow modules + shadow_config.json by sha256) and ``shadow_sources.load`` (step5/* + live_adapter by pin)."""
    mods = SS.load_step5_shadow({'step5_shadow': step5_shadow})
    config = mods.S.load_config()
    src = mods.S.load(config)
    encoder, encoder_sha = src.F.load_encoder(config['encoder']['path'], config['encoder']['sha256'])
    return Step5Env(mods, config, src, encoder, encoder_sha)


def module_files_under(roots: Iterable[str]) -> Dict[str, str]:
    """{absolute file: sha256} of every loaded module whose file lies under one of ``roots``."""
    roots = [Path(r).resolve() for r in roots]
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f:
            continue
        p = Path(f).resolve()
        if p.suffix == '.py' and p.is_file() and any(r == p or r in p.parents for r in roots):
            out[str(p)] = sha256_file(p)
    return out


def registered_pins(reg: dict) -> Dict[str, str]:
    """Union of every model source's ``source_pins`` in the registration, keys resolved to absolute paths
    (relative keys against the block's ``source_root`` or ``seq_shadow_src_root``). Process A hosts several
    adapters over one shared source root, so a module pinned by any registered model is a pinned module."""
    out: Dict[str, str] = {}
    for block in (reg.get('model_sources') or {}).values():
        if not isinstance(block, dict):
            continue
        base = block.get('source_root') or block.get('seq_shadow_src_root') or '/'
        for k, v in (block.get('source_pins') or {}).items():
            p = Path(k) if Path(k).is_absolute() else Path(base)/k
            out[str(p.resolve())] = v
    return out


def check_no_unpinned(spec: dict, reg: dict) -> None:
    """After import: every loaded module under this adapter's roots is a file pinned by the registration, with
    those bytes (own pins first; then any other registered model's pins, since process A shares one source root)."""
    pins = registered_pins(reg)
    pins.update({str(Path(k).resolve()): v for k, v in spec['source_pins'].items()})
    loaded = module_files_under(spec['pin_roots'])
    stray = sorted(p for p in loaded if p not in pins)
    changed = sorted(p for p, s in loaded.items() if p in pins and pins[p] != s)
    if stray or changed:
        raise PinMismatch(f'unpinned or changed module loaded: {stray + changed}')


# ---------------------------------------------------------------- profile (O-4)
def load_game_profile(src, game: dict, scope: str = ACTUAL_SCOPE):
    """The game's daily profile (``execution.games[i].profile_path`` = <attempt dir>/PROFILE.json) through the
    accepted ``live_adapter.load_committed``: COMMITTED RECEIPT naming the PROFILE.json sha256/bytes and a
    SUPERVISOR-RESULT.json judging exactly those bytes usable. Then the live-gate rule of
    ``shadow_runner.check_live_profile`` (scope ACTUAL_S2_ONE_PASS, supervisor success, role) and the registered
    sha256 and strict-prior date. Raises before the first pitch rather than abstaining all game."""
    path = Path(game['profile_path'])
    if path.name != 'PROFILE.json':
        raise PinMismatch('profile_path must name an attempt directory PROFILE.json')
    if sha256_file(path) != game['profile_sha256']:
        raise PinMismatch('PROFILE.json sha256 differs from execution.games[i].profile_sha256')
    profile = src.A.load_committed(src.F, path.parent)
    prov = profile.provenance()
    bad = [name for name, ok in (
        ('profile_sha256', prov.get('profile_sha256') == game['profile_sha256']),
        ('read_scope', prov.get('read_scope') == scope),
        ('role', prov.get('role') == PROFILE_ROLE),
        ('supervisor_verdict', scope != ACTUAL_SCOPE
         or str(prov.get('supervisor_verdict', '')).startswith(SUPERVISED)),
        ('strictly_prior_to_official_date', profile.strictly_prior(game['official_date']))) if not ok]
    if bad:
        raise PinMismatch('profile not connectable: ' + ','.join(bad))
    return profile


# ---------------------------------------------------------------- one body
def read_feed(game: dict, ctx, F) -> dict:
    """The stored snapshot named by the context, verified, or Abstain."""
    if ctx.source_seq is None:
        raise Abstain('NO_SOURCE_BODY')
    try:
        snap = store_source.read_snapshot_raw(Path(game['store_path']), int(ctx.source_seq))
    except KeyError:
        raise Abstain('SOURCE_BODY_ABSENT') from None
    except ValueError:
        raise Abstain('SOURCE_BODY_SHA_INVALID') from None
    if snap['game_pk'] != int(game['game_pk']) or ctx.game_pk != int(game['game_pk']):
        raise Abstain('SOURCE_BODY_OTHER_GAME')
    if not ctx.source_sha or snap['sha256'] != ctx.source_sha:
        raise Abstain('SOURCE_SHA_MISMATCH')
    try:
        feed = json.loads(snap['body_bytes'])
    except ValueError:
        raise Abstain('bad_schema_or_hash:feed_not_json') from None
    # pipeline._identity: the body must be the named committed version of the named game
    if not isinstance(feed, dict) or feed.get('gamePk') != int(game['game_pk']) or type(feed.get('gamePk')) is not int:
        raise Abstain('bad_schema_or_hash:snapshot_belongs_to_another_game')
    if hashlib.sha256(F.canonical(feed)).hexdigest() != snap['sha256']:
        raise Abstain('bad_schema_or_hash:snapshot_sha_mismatch')
    _refuse_target(feed, ctx.state.at_bat_index, ctx.state.pitch_number)
    return feed


def _refuse_target(feed, at_bat_number, pitch_number):
    plays = (feed.get('liveData') or {}).get('plays') or {}
    listed = plays.get('allPlays') if isinstance(plays.get('allPlays'), list) else []
    for play in listed + [plays.get('currentPlay')]:
        if not isinstance(play, dict) or (play.get('about') or {}).get('atBatIndex') != at_bat_number - 1:
            continue
        for ev in play.get('playEvents') or []:
            if isinstance(ev, dict) and ev.get('isPitch') and ev.get('pitchNumber') == pitch_number:
                raise Abstain('BODY_CONTAINS_TARGET')


@dataclass
class Prepared:
    state: dict
    eligible: List[int]
    x_f1: object       # 17 x 199 float32 (2024 F1-design)
    raw: object        # 17 x 91 float64
    support: dict
    vectors_sha256: str


def prepare(env: Step5Env, feed: dict, ctx, profile) -> Prepared:
    """``pipeline.decide`` up to the head call, minus timing/ledger (see module docstring)."""
    import numpy as np
    F = env.src.F
    try:
        state, missing = F.pre_pitch_state(feed)
    except Exception as exc:  # noqa: BLE001 - same vocabulary as decide
        raise Abstain('bad_schema_or_hash:feed_parse_error:' + type(exc).__name__) from None
    ab, n = state['at_bat_number'], state['last_pitch_number']
    if ab is None:
        raise Abstain('missing_input:' + sorted(set(missing.values()))[0])
    if state['pa_status'] != 'open':
        raise Abstain('missing_input:' + sorted(set(missing.values()))[0])
    st = ctx.state
    if (ab, n + 1) != (st.at_bat_index, st.pitch_number) or any(
            state.get(k) is not None and v and state[k] != v
            for k, v in (('pitcher', st.pitcher_id), ('batter', st.batter_id))):
        raise Abstain('CONTEXT_STATE_MISMATCH')
    if missing:
        raise Abstain('missing_input:' + sorted(set(missing.values()))[0])
    if profile is None:
        raise Abstain('missing_profile:historical_profile_bytes_not_registered')
    if not profile.strictly_prior(state['official_date']):
        raise Abstain('missing_profile:profile_as_of_not_strictly_before_game_date')
    support = profile.support(state['pitcher'], state['official_date'])
    eligible = [i for i, c in enumerate(support['candidates']) if c['status'] == 'provisionally_eligible']
    if support['general_reason']:
        raise Abstain('unsupported_player:' + str(support['general_reason']))
    if not eligible:
        raise Abstain('no_supported_action:no_action_meets_support_thresholds')
    try:
        x, raw = F.build_candidates(state, profile, env.encoder)
    except Exception as exc:  # noqa: BLE001
        raise Abstain(f'predictor_error:build:{type(exc).__name__}:{exc}'[:300]) from None
    vsha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    return Prepared(state, eligible, x, raw, support, vsha)
