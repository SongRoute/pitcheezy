"""Pure final-only S2/I3 projection adapter; no data I/O or execution grant.

The caller must verify the independently authorized source bytes/manifests before
constructing frames. Matching declared pins here is not proof of byte custody.
"""
import copy
import numpy as np
import pandas as pd
from experiments.ws_fullseason.data import key_digest

TYPES = ('R', 'F', 'D', 'L', 'W')
HISTORICAL_SEGREGATION = (849845, 849843, 849832, 849844, 849831)
CUTOFF = '2026-10-11'
SOURCES = {
    'S2': {'version': 'd20260930-h2026f',
           'sha256': 'c0c1eda4b515fc14dab79defe577b449e2695d40dd036f5c35ae52e37093ed9b',
           'manifest_sha256': '54dff07396c24cd278bfde3dfd11b07de68094336412b0d12ceaf7131aa5c63b'},
    'I3': {'version': 'd20261012-i2026-e20261011',
           'sha256': 'd045ab62a1694a87e383a8d30a90c4006975697a2d2a6eb294e398e422e350c0',
           'manifest_sha256': '8f91af339de0aabaca881f9083518016b0005c340b90799cebfb8dd76f88de8a'},
}


def project(frames, recipe, M, authority):
    """One global legacy projection, with original type and source restored.

    Allowed type normalization is an internal adapter only. Before it, duplicate
    rows are compared in their original schema, so R/F conflicts cannot collapse.
    Existing OOF callers and the pinned legacy projector are unchanged.
    """
    policy = recipe['row_policy']
    if (set(frames) != set(SOURCES) or authority.get('sources') != SOURCES
            or authority.get('mode') != 'PRODUCTION_FIT' or authority.get('root_approved') is not True
            or authority.get('sealed_assets_read') is not False
            or recipe.get('execution_stage') != 'FINAL_REFIT' or recipe.get('production_cutoff') != CUTOFF
            or recipe.get('data_role') != '2026 EXPOSED DEVELOPMENT'
            or policy.get('start') != '2026-03-25' or policy.get('end') != CUTOFF
            or policy.get('game_types') != list(TYPES) or policy.get('game_type') != 'R'
            or sorted(policy.get('segregation_game_pks', [])) != sorted(HISTORICAL_SEGREGATION)):
        raise ValueError('explicit fixed final-source authority required')
    if any(list(frames[name].columns) != M.cp.PROJECT for name in SOURCES):
        raise ValueError('exact final projection columns')
    raw = pd.concat([frames[name] for name in SOURCES], ignore_index=True)
    if raw.empty: raise ValueError('empty final source union')
    source = np.concatenate([np.full(len(frames[name]), name, dtype='U2') for name in SOURCES])
    # Same pre-cast identity protection as ws_fullseason.data.project.
    for name in M.cp.KEY + ['pitcher', 'batter']:
        values = raw[name].dropna().to_numpy()
        if (values.dtype.kind not in 'iuf' or not np.isfinite(values).all()
                or (values <= 0).any() or (values >= 2**53).any()
                or (values != np.floor(values)).any()):
            raise ValueError('invalid integral identity ' + name)
    raw['game_date'] = pd.to_datetime(raw.game_date, errors='coerce')
    known = raw[M.cp.KEY].notna().all(axis=1)
    duplicate = known & raw.duplicated(M.cp.KEY, keep=False)
    original = raw.loc[duplicate, M.cp.PROJECT].astype(str)
    if original.drop_duplicates().duplicated(M.cp.KEY, keep=False).any():
        raise ValueError('conflicting original source rows before type mapping')
    typed = raw.loc[raw.game_pk.notna(), ['game_pk', 'game_type']].drop_duplicates()
    if typed.game_pk.duplicated().any():
        raise ValueError('mixed original game types for one game')
    dated = raw.loc[raw.game_pk.notna() & raw.game_date.notna(), ['game_pk', 'game_date']].drop_duplicates()
    multi = set(dated.game_pk[dated.game_pk.duplicated()])
    # No mutation of caller frames: copy only this column before cp.project's
    # existing global copy. Its fate ordering and stop denominator stay intact.
    mapped = raw.copy(deep=False)
    mapped['game_type'] = raw.game_type.where(~raw.game_type.isin(TYPES), 'R')
    internal = copy.deepcopy(recipe)
    internal['row_policy']['segregation_game_pks'] = []
    g, census = M.cp.project(mapped, internal, M.safe.outcome_labels, M.safe.head_labels)
    if g.empty: raise ValueError('empty final FIT')
    if census['multi_date_games'] != len(multi):
        raise ValueError('global date quarantine changed')
    first = raw.loc[known].drop_duplicates(M.cp.KEY, keep='first')
    row_lookup = first.assign(_source_row=first.index).set_index(M.cp.KEY)
    positions = row_lookup.loc[pd.MultiIndex.from_frame(g[M.cp.KEY]), '_source_row'].to_numpy(int)
    g['game_type'] = raw.game_type.iloc[positions].to_numpy()
    accepted_source = source[positions]
    if not g.game_type.isin(TYPES).all(): raise ValueError('nonregistered accepted game type')
    lineage = {name: {'input_rows': int((source == name).sum()),
                     'accepted_unique_keys': int((accepted_source == name).sum()),
                     'nonaccepted_rows': int((source == name).sum() - (accepted_source == name).sum()),
                     'accepted_keys_sha256': key_digest(g.loc[accepted_source == name, M.cp.KEY].to_numpy(dtype='<i8'))}
               for name in SOURCES}
    if (sum(v['input_rows'] for v in lineage.values()) != census['source_rows']
            or sum(v['accepted_unique_keys'] for v in lineage.values()) != census['accepted_rows']
            or sum(v['nonaccepted_rows'] for v in lineage.values()) != sum(census['excluded'].values())):
        raise ValueError('source lineage accounting')
    census.update(source_lineage=lineage, source_precedence=['S2', 'I3'],
                  source_reason_scope='global ordered exclusion reasons; source-specific nonaccepted totals only',
                  accepted_game_types={t: int(g.game_type.eq(t).sum()) for t in TYPES},
                  historical_segregation_game_pks=list(HISTORICAL_SEGREGATION),
                  historical_segregation_rows_from_official_source=int(raw.game_pk.isin(HISTORICAL_SEGREGATION).sum()),
                  sealed_assets_read=False, production_cutoff=CUTOFF)
    fate = M.safe.core_fate(g)
    if not np.isin(fate, M.safe.CORE).all(): raise ValueError('accepted non-core label')
    y7 = np.array([int(v[-1]) for v in fate], dtype=np.int64)
    return g, census, np.ones(len(g), bool), np.zeros(len(g), bool), y7
