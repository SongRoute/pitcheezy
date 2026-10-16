"""Generated donors only: no research parquet, learned encoder, or model reads."""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from pitcheezy.seq_shadow import fullseason_profiles as profiles


def donors():
    M=profiles.closure();n=180;i=np.arange(n);day=i//30
    g=pd.DataFrame({'game_pk':100+day,'at_bat_number':1+(i%30)//5,'pitch_number':1+i%5,
        'game_date':pd.Timestamp('2026-03-25')+pd.to_timedelta(day,'D'),
        'pitcher':10+(i%5),'batter':20+(i%4),'pitch_type':np.array(['FF','SL','UNKNOWN'])[i%3],
        'balls':i%2,'strikes':i%2,'outs_when_up':i%3,'inning':1+i%9,'inning_topbot':np.where(i%2,'Top','Bot'),
        'home_score':i%5,'away_score':i%7,'on_1b':np.where(i%2,42,np.nan),'on_2b':np.nan,'on_3b':np.nan,
        'stand':np.where(i<90,'L','R'),'p_throws':np.where(i%7,'R','L'),
        'description':np.array(['ball','foul','swinging_strike','hit_into_play','called_strike'])[i%5],
        'bb_type':np.where(i%5==3,'ground_ball','None'),'outcome10':i%10})
    rng=np.random.default_rng(17)
    for c in M.fsf.PHYSICS:g[c]=rng.normal(size=n)+np.arange(n)*.01
    g.loc[i%11==0,'release_spin_rate']=np.nan
    return g,M


def bundle(g,M):
    return profiles.export_profiles(g[M.safe.DONOR],['2026-03-25','2026-03-27','2026-03-30','2026-03-31'],'2026-03-30',M)


def compare_reference(b,g,M,date):
    rows=g.loc[g.game_date==pd.Timestamp(date)].copy()
    if rows.empty:rows=g.tail(30).copy();rows['game_date']=pd.Timestamp(date)
    expected,_=M.scale.features(g[M.safe.DONOR],rows,'2026-03-30',M)
    actual=b.raw91(M.safe.query_context(rows))
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12,equal_nan=True)
    # Frozen encoder geometry is generated, never read from accepted artifacts.
    e={'median':np.zeros(91),'mean':np.full(91,.1),'scale':np.ones(91),
       'continuous':list(range(36)),'knots':np.zeros((3,36))}
    np.testing.assert_array_equal(M.safe.apply_encoder(actual,e),M.safe.apply_encoder(expected,e))
    return actual


def test_all91_and199_empty_low_final_future_origins():
    g,M=donors();b=bundle(g,M)
    for date in ['2026-03-25','2026-03-27','2026-03-30','2026-03-31','2026-04-20']:
        compare_reference(b,g,M,date)
    last=b.raw91(M.safe.query_context(g.tail(30)))
    assert (last[:,89]==0).any()  # Other-pitcher donor weighted mixture exercised.
    first=b.raw91(M.safe.query_context(g.head(30)))
    assert np.isnan(first[:,51]).all();assert (first[:,50]==0).all();assert (first[:,89]==1).all()


def test_same_date_future_perturb_invariant_and_later_sensitive():
    g,M=donors();b=bundle(g,M);h=g.copy();later=h.game_date>=pd.Timestamp('2026-03-27')
    for c in M.fsf.PHYSICS:h.loc[later,c]=h.loc[later,c]*2+7
    h.loc[later,'outcome10']=0;h.loc[later,'description']='ball';h.loc[later,'bb_type']='None'
    alt=bundle(h,M);q=M.safe.query_context(g[g.game_date==pd.Timestamp('2026-03-27')])
    np.testing.assert_array_equal(b.raw91(q),alt.raw91(q))
    q=M.safe.query_context(g[g.game_date==pd.Timestamp('2026-03-30')])
    assert not np.array_equal(b.raw91(q),alt.raw91(q),equal_nan=True)


def test_unknown_ids_type_absence_and_previous_separate():
    g,M=donors();b=bundle(g,M);rows=g.tail(3).copy();rows['game_date']=pd.Timestamp('2026-03-31')
    rows['pitcher']=9999;rows['batter']=8888;rows['pitch_type']='CU'
    q=M.safe.query_context(rows);expected,_=M.scale.features(g[M.safe.DONOR],rows,'2026-03-30',M)
    np.testing.assert_allclose(b.raw91(q),expected,atol=1e-12,rtol=1e-12,equal_nan=True)
    q['previous_type']='FF';q['previous_valid']=True;actual=b.raw91(q)
    assert (actual[:,49]==1).all();assert (actual[:,30+M.fsf.TYPES.index('FF')]==1).all()
    q['pitcher']=123456;np.testing.assert_array_equal(actual[:,:50],b.raw91(q)[:,:50])


def test_finite_subset_distinct_from_physics_and_batter_denominators():
    g,M=donors();g.loc[g.pitch_type=='FF','release_speed']=np.nan;b=bundle(g,M)
    compare_reference(b,g,M,'2026-03-31')
    p=b.snapshots['2026-03-31'].index
    assert p['pitcher_type'][(10,M.fsf.TYPES.index('FF'))][0]>0
    assert (M.fsf.TYPES.index('FF'),0,10) not in p['physical']


def test_archive_roundtrip_tamper_and_unregistered_date(tmp_path):
    g,M=donors();b=bundle(g,M);out=tmp_path/'bundle';manifest=profiles.save_bundle(b,out,{'generated':True})
    digest=profiles._sha(out/'manifest.json');loaded=profiles.load_bundle(out,digest,M)
    q=M.safe.query_context(g.tail(30));np.testing.assert_array_equal(b.raw91(q),loaded.raw91(q))
    q['game_date']=pd.Timestamp('2026-03-26')
    with pytest.raises(ValueError):loaded.raw91(q)
    f=out/'2026-03-30.npz';raw=f.read_bytes();f.write_bytes(raw[:-1]+b'x')
    with pytest.raises(ValueError):profiles.load_bundle(out,digest,M)


def test_invalid_query_and_export_scope_rejected():
    g,M=donors();b=bundle(g,M);q=M.safe.query_context(g.tail(1));q['balls']=4
    with pytest.raises(ValueError):b.raw91(q)
    q['balls']=0;q['pitcher']=1.5
    with pytest.raises(ValueError):b.raw91(q)
    with pytest.raises(ValueError):profiles.export_profiles(g[M.safe.DONOR],['2026-03-30'],'2026-03-30',M)


def test_explicit_prestate_and_canonical_archive_origins(tmp_path):
    g,M=donors();b=bundle(g,M);query=M.safe.query_context(g.tail(1))
    for column,value in [('outs_when_up',np.nan),('outs_when_up',3),('inning',0),('home_score',-1),
                         ('away_score',.5),('inning_topbot','unknown'),('previous_valid',1),
                         ('pitcher',2**53),('on_1b',-1),('game_date','2026-03-30T01:00:00Z')]:
        q=query.copy();q[column]=value
        with pytest.raises(ValueError):b.raw91(q)
    out=tmp_path/'bundle';manifest=profiles.save_bundle(b,out,{})
    manifest['origins']['../escape']=manifest['origins'].pop('2026-03-25')
    (out/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):profiles.load_bundle(out,profiles._sha(out/'manifest.json'),M)
