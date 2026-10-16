"""FIT-only producer for a fixed development origin; no automatic execution entrypoint.

Reuses campaign date-history features and original integer sequence producer. The
caller's registered gate owns source custody, memory/time caps and publication.
"""
import gc
import json
from pathlib import Path
import numpy as np
from experiments.ws_campaign.data import fit_prior_support
from experiments.ws_seq.prepare_inputs import sequence_arrays
from experiments.ws_seq import train_seq as base
from experiments.ws_fullseason import data


def raw_features(g, fit, recipe, M, chunk_rows=65536):
    cutoff = recipe['window']['fit'][1]
    donor = g.loc[fit, M.safe.DONOR].reset_index(drop=True)
    if donor.empty or str(donor.game_date.max().date()) > cutoff:
        raise ValueError('empty or future donor')
    # Whole dates keep every same-game previous-pitch lookup intact. Reuse the
    # exact feature helper on each batch; no query outcome enters the donor pool.
    batches=[]; current=[]; size=0
    for indices in g.groupby('game_date',sort=True).indices.values():
        if len(indices)>chunk_rows: raise ValueError('one date exceeds query chunk cap')
        if size+len(indices)>chunk_rows and current:
            batches.append(np.concatenate(current)); current=[]; size=0
        current.append(indices); size+=len(indices)
    if current: batches.append(np.concatenate(current))
    raw=np.empty((len(g),91),np.float64); totals={}
    for indices in batches:
        part,audit=M.scale.features(donor,g.iloc[indices],cutoff,M)
        if part.shape!=(len(indices),91): raise ValueError('raw feature schema')
        raw[indices]=part
        for name,value in audit.items(): totals[name]=totals.get(name,0)+int(value)
    return raw,totals


def fit_encoder(raw, fit):
    """Same FIT matrix reductions as fsf.encode, without allocating expanded FIT X.

    This preserves median/mean/std/continuous/quantile operations and their axis;
    only the unused 199-column FIT expansion is omitted.
    """
    training=raw[fit]
    median=np.nanmedian(training,axis=0); median=np.where(np.isfinite(median),median,0)
    filled=np.where(np.isfinite(training),training,median); del training
    mean=filled.mean(0); scale=filled.std(0); scale=np.where(scale>1e-8,scale,1)
    continuous=np.flatnonzero([len(np.unique(filled[:,j]))>2 for j in range(filled.shape[1])])
    z=np.clip((filled-mean)/scale,-8,8); del filled
    knots=np.quantile(z[:,continuous],[.25,.5,.75],axis=0)
    return {'median':median.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
        'continuous':continuous.tolist(),'knots':knots.tolist(),'dimension':raw.shape[1]+3*len(continuous)}


def arrays(g, fit, oof, y7, recipe, M):
    print('phase=raw-features',flush=True)
    raw, audit = raw_features(g, fit, recipe, M)
    # Every fitted statistic and continuous-column choice sees exactly FIT rows.
    print('phase=fit-encoder',flush=True)
    encoder = fit_encoder(raw,fit)
    dimension = recipe['base']['inputs']['feature_dimension']
    if encoder['dimension'] != dimension:
        raise ValueError(f'FIT-only encoder dimension {encoder["dimension"]} differs from frozen {dimension}; never use OOF to choose columns')
    print('phase=apply-encoder',flush=True)
    x=np.empty((len(g),dimension),np.float32)
    for start in range(0,len(g),16384):
        x[start:start+16384]=M.safe.apply_encoder(raw[start:start+16384],encoder)
    if not np.isfinite(x).all(): raise ValueError('nonfinite encoded features')
    del raw; gc.collect()
    print('phase=sequence-and-proof',flush=True)
    seq = sequence_arrays(g, M)
    corpus = dict(keys=g[M.cp.KEY].to_numpy(dtype='<i8'), y10=g.outcome10.to_numpy(dtype=np.int64),
        y7=y7, games=g.game_pk.to_numpy(dtype=np.int64), fit=fit, tuning=np.zeros(len(g),bool),
        calibration=np.zeros(len(g),bool), evaluation=oof,
        narrow_whiff=g.narrow_whiff.to_numpy(), ground_ball_per_pitch=g.ground_ball_per_pitch.to_numpy())
    ctx = base.build_context_index(seq['pitcher'], seq['game_pk'], seq['at_bat_number'], seq['pitch_number'], recipe['base']['context']['max_length'])
    base.check_context(ctx, seq['pitcher'], seq['game_pk'], seq['at_bat_number'], seq['pitch_number'])
    valid=ctx>=0
    if not (fit[np.maximum(ctx,0)] == fit[:,None])[valid].all():
        raise ValueError('context crosses role')
    days=g.game_date.dt.strftime('%Y-%m-%d').to_numpy()
    support=fit_prior_support(days, seq['pitcher'], seq['game_pk'], seq['at_bat_number'], seq['pitch_number'], fit)
    fit_digest=data.key_digest(corpus['keys'][fit])
    proof={'encoder_fit':{'rows':int(fit.sum()),'keys_sha256':fit_digest,'dimension':dimension},
        'date_profile_pool':{'rows':int(fit.sum()),'keys_sha256':fit_digest,'cutoff':recipe['window']['fit'][1],'strictly_earlier_query_date':True},
        'effective_donor_subsets':{'semantics':'eligible producer updates, not per-query matched donor edges',
            'batter_history':{'rows':int(fit.sum()),'keys_sha256':fit_digest},
            'outcome_transfer_complete_physics':{'rows':int((fit & np.isfinite(g[list(M.fsf.PHYSICS)].to_numpy(float)).all(1)).sum()),
                'keys_sha256':data.key_digest(corpus['keys'][fit & np.isfinite(g[list(M.fsf.PHYSICS)].to_numpy(float)).all(1)])},
            'physics_finite_by_column':{name:{'rows':int((fit & np.isfinite(g[name].to_numpy(float))).sum()),
                'keys_sha256':data.key_digest(corpus['keys'][fit & np.isfinite(g[name].to_numpy(float))])} for name in M.fsf.PHYSICS}},
        'support_updates':{'rows':int(fit.sum()),'keys_sha256':fit_digest,'rule':'earlier FIT dates plus earlier FIT same-game pitches; no other same-day games'},
        'sequence':{'edges':int(valid.sum()),'oof_edges':int(valid[oof].sum()),'same_pitcher_game_strict_order':True,'earlier_oof_outcomes_used':True},
        'vocabulary':{'type_values':list(M.fsf.TYPES),'learned_from_rows':False},
        'arrival':'UNKNOWN; corrected retrospective reconstruction','transfer_audit':audit}
    return x,corpus,seq,encoder,support,days.astype('U10'),proof


def save_package(out, built, recipe_sha, source_binding, sha):
    out=Path(out)
    if out.exists(): raise ValueError('fresh package output required')
    out.mkdir()
    x,c,s,e,support,days,proof=built
    np.save(out/'x-prequential.npy',x,allow_pickle=False)
    np.savez(out/'corpus-subset.npz',**c); np.savez(out/'seq-inputs.npz',**s)
    np.save(out/'fit-prior-support.npy',support,allow_pickle=False)
    np.save(out/'game-date.npy',days,allow_pickle=False)
    (out/'encoder.json').write_text(json.dumps(e,sort_keys=True,allow_nan=False)+'\n')
    names=['x-prequential.npy','corpus-subset.npz','seq-inputs.npz','fit-prior-support.npy','game-date.npy','encoder.json']
    manifest={'status':'COMPLETED','recipe_sha256':recipe_sha,'source_binding':source_binding,'proof':proof,
        'files':{name:{'sha256':sha(out/name),'bytes':(out/name).stat().st_size} for name in names},
        'fits':0,'learned_encoder':1,'roles':{'FIT':int(c['fit'].sum()),'OOF':int(c['evaluation'].sum())}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest
