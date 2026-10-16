"""Generated models/statistics/events only; never read accepted research artifacts."""
import copy
import hashlib
import numpy as np
import pandas as pd
import pytest

from pitcheezy.seq_shadow import fullseason_profiles as profiles, fullseason_stream as stream
from pitcheezy.seq_shadow.fullseason_runtime import Runtime
from test_seq_shadow_fullseason import generated_config
from test_fullseason_profiles import donors, bundle


def setup(tmp_path):
    config,_=generated_config(tmp_path)
    g,M=donors();b=bundle(g,M);directory=tmp_path/'profiles'
    profiles.save_bundle(b,directory,{'generated':True})
    digest=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()
    runtime=Runtime(config,directory,digest)
    st=stream.StreamState(tmp_path/'state.sqlite',game_pk=105,game_date='2026-03-30',
                          normalizer_sha256='a'*64,evidence_kind='synthetic')
    current=g.tail(1).iloc[0]
    for i,r in g[g.game_pk==105].iloc[:-1].iterrows():
        st.ingest(str(i),[int(r[k]) for k in stream.KEY],1,
            dict(status='accepted',reason='generated',accepted={
                k:str(r[k]) if k=='pitch_type' else int(r[k]) for k in stream.ACCEPTED}),received_ns=int(i)+1)
    q={k:None if pd.isna(current[k]) else current[k] for k in stream.CONTEXT}
    q={k:(v.item() if isinstance(v,np.generic) else v) for k,v in q.items()}
    q['game_date']='2026-03-30'
    for k in ('on_1b','on_2b','on_3b'):
        if q[k] is not None:q[k]=int(q[k])
    coverage=dict(complete=True,before_key=[q[k] for k in stream.KEY],source_sha256='b'*64)
    snap=st.decide('q',q,pitch_type_basis='hypothetical_candidate',coverage=coverage,received_ns=200)
    return runtime,st,snap,q,coverage,g,M


def test_generated_profile_stream_encoder_three_seed_composition(tmp_path):
    runtime,st,snap,q,coverage,g,M=setup(tmp_path)
    actual=runtime.predict(snap)
    target=g.tail(30)
    raw,_=M.scale.features(g[M.safe.DONOR],target,'2026-03-30',M)
    encoded=runtime.ensemble.encode(raw[-1:])
    expected=runtime.ensemble.predict(encoded,stream.arrays(snap),np.array([[q[k] for k in stream.KEY]],np.int64))
    np.testing.assert_allclose(actual['probabilities'],expected['calibrated'][0],atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(actual['raw_mean'],expected['raw_mean'][0],atol=1e-7,rtol=1e-6)
    assert len(actual['probabilities'])==7 and abs(sum(actual['probabilities'])-1)<1e-12
    assert actual['status']=='scored' and not actual['prospective_claim']
    for _ in range(12):assert runtime.predict(snap)==actual
    st.close()


def test_generated_restart_correction_isolated_future_decision(tmp_path):
    runtime,st,snap,q,coverage,g,M=setup(tmp_path)
    original=runtime.predict(snap)
    previous=g.tail(2).iloc[0]
    corrected={k:str(previous[k]) if k=='pitch_type' else int(previous[k]) for k in stream.ACCEPTED}
    corrected['pitch_type']='CH';corrected['outcome10']=8
    st.ingest(str(g.index[-2]),[int(previous[k]) for k in stream.KEY],2,
              dict(status='accepted',reason='generated_correction',accepted=corrected),received_ns=201)
    future=st.decide('q-new',q,pitch_type_basis='hypothetical_candidate',coverage=coverage,received_ns=202)
    assert future['query']['previous_type']=='CH'
    assert runtime.predict(future)['call_sha256']!=original['call_sha256']
    st.close()
    st=stream.StreamState(tmp_path/'state.sqlite',game_pk=105,game_date='2026-03-30',normalizer_sha256='a'*64,evidence_kind='synthetic')
    replay=st.decide('q',q,pitch_type_basis='hypothetical_candidate',coverage=coverage,received_ns=203)
    assert runtime.predict(replay)==original
    st.close()


def test_abstention_and_tampered_snapshot_do_not_forward(tmp_path,monkeypatch):
    runtime,st,snap,q,coverage,g,M=setup(tmp_path)
    def forbidden(*a,**k):raise AssertionError('no profile/model work for abstention')
    monkeypatch.setattr(runtime.profiles,'raw91',forbidden)
    incomplete=st.decide('gap',q,pitch_type_basis='observed_actual',coverage=dict(coverage,complete=False),received_ns=201)
    result=runtime.predict(incomplete)
    assert result['status']=='abstain' and result['probabilities'] is None and not result['prospective_claim']
    changed=copy.deepcopy(incomplete);changed['identity']['evidence_kind']='captured'
    with pytest.raises(ValueError):runtime.predict(changed)
    st.close()
