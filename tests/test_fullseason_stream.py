"""Generated normalized rows only; no capture, research artifact, model or host."""
import copy
import sqlite3

import numpy as np
import pytest

from pitcheezy.seq_shadow import fullseason_stream as s


def query(ab=1, pitch=2, pitcher=11, batter=22, kind='FF'):
    return dict(game_pk=10, at_bat_number=ab, pitch_number=pitch, game_date='2026-10-11',
        pitcher=pitcher, batter=batter, pitch_type=kind, balls=0, strikes=0,
        outs_when_up=0, inning=1, inning_topbot='Top', home_score=0, away_score=0,
        on_1b=None, on_2b=None, on_3b=None, stand='R', p_throws='R')


def row(pitcher=11, batter=22, kind='FF', out=1, status='accepted'):
    return dict(status=status, reason='generated_quality_proof', accepted=dict(
        pitcher=pitcher, batter=batter, pitch_type=kind, outcome10=out, balls=1, strikes=2
    ) if status == 'accepted' else None)


def state(tmp_path, kind='synthetic'):
    return s.StreamState(tmp_path/'state.sqlite', game_pk=10, game_date='2026-10-11',
                         normalizer_sha256='a'*64, evidence_kind=kind)


def decide(st, q=None, *, id='d1', now=100, complete=True, basis='hypothetical_candidate'):
    q=query() if q is None else q
    return st.decide(id, q, pitch_type_basis=basis, received_ns=now,
        coverage=dict(complete=complete, before_key=[q[k] for k in s.KEY], source_sha256='b'*64))


def test_previous_pitcher_independent_but_sequence_same_pitcher_and_pa_boundaries(tmp_path):
    st=state(tmp_path)
    st.ingest('p1',(10,1,1),1,row(kind='CH'),received_ns=1)
    st.ingest('p2',(10,1,2),1,row(pitcher=33,kind='SL'),received_ns=2)
    snap=decide(st,query(pitch=3))
    assert snap['query']['previous_type']=='SL' and snap['query']['previous_valid']
    tokens=s.arrays(snap)
    assert tokens['valid'].sum()==1 and tokens['context_keys'][0,0].tolist()==[10,1,1]
    nextpa=decide(st,query(ab=2,pitch=1),id='d2',now=101)
    assert not nextpa['query']['previous_valid'] and s.arrays(nextpa)['valid'].sum()==1
    # Batter substitution changes same-batter flags; pitcher substitution excludes old pitcher.
    assert s.arrays(decide(st,query(pitch=3,batter=44),id='d3',now=102))['same'][0,0]==0
    assert s.arrays(decide(st,query(pitch=3,pitcher=33),id='d4',now=103))['context_keys'][0,0].tolist()==[10,1,2]
    st.close()


def test_correction_is_append_only_and_first_decision_survives_restart(tmp_path):
    st=state(tmp_path)
    first=st.ingest('p1',(10,1,1),1,row(),received_ns=10)
    original=decide(st,now=20)
    correction=st.ingest('p1',(10,1,1),2,row(kind='SL',out=2),received_ns=30)
    assert first['first_received_ns']==correction['first_received_ns']==10
    assert correction['status']=='correction'
    newer=decide(st,id='d2',now=40)
    assert newer['query']['previous_type']=='SL' and original['query']['previous_type']=='FF'
    assert s.arrays(newer)['out'][0,0]==2
    st.close();st=state(tmp_path)
    assert decide(st,now=50)==original
    assert st.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]==2
    for table in ('identity','events','decisions'):
        with pytest.raises(sqlite3.IntegrityError): st.db.execute(f'DELETE FROM {table}')
        st.db.rollback()
    st.close()


def test_duplicate_reconnect_identity_collision_revision_and_clock_fail_closed(tmp_path):
    st=state(tmp_path)
    st.ingest('p1',(10,1,1),1,row(),received_ns=10)
    assert st.ingest('p1',(10,1,1),1,row(),received_ns=20)['status']=='duplicate'
    assert st.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]==1
    st.close();st=state(tmp_path)
    with pytest.raises(ValueError): st.ingest('p1',(10,1,1),2,row(),received_ns=15)
    for eid,key,rev,r,ns in [('p1',(10,1,1),1,row(kind='CH'),30),('other',(10,1,1),2,row(),30),
                            ('p1',(10,1,2),2,row(),30),('p1',(10,1,1),2,row(),9)]:
        with pytest.raises(ValueError): st.ingest(eid,key,rev,r,received_ns=ns)
    snap=decide(st,now=40)
    with pytest.raises(ValueError): decide(st,query(kind='CH'),now=50)
    assert decide(st,now=60)==snap
    st.close()
    with pytest.raises(ValueError): state(tmp_path,kind='captured')


def test_unknown_and_incomplete_prefix_abstain_rejected_absence_and_unknown_type_valid(tmp_path):
    st=state(tmp_path)
    assert decide(st,id='incomplete',complete=False)['reason']=='stored_prefix_incomplete'
    st.ingest('p1',(10,1,1),1,row(status='unknown'),received_ns=101)
    held=decide(st,id='unknown',now=102)
    assert held['reason']=='prior_quality_unknown'
    with pytest.raises(ValueError): s.arrays(held)
    st.ingest('p1',(10,1,1),2,row(status='rejected'),received_ns=103)
    empty=decide(st,id='excluded',now=104)
    assert empty['status']=='ready' and not empty['query']['previous_valid']
    st.ingest('p1',(10,1,1),3,row(kind='UNKNOWN'),received_ns=105)
    known=decide(st,id='known',now=106)
    assert known['query']['previous_valid'] and known['query']['previous_type']=='UNKNOWN'
    assert s.arrays(known)['valid'][0,0] and s.arrays(known)['type'][0,0]==0
    assert decide(st,id='unknown',now=107)==held
    st.close()


def test_context_latest32_order_padding_and_strict_prior(tmp_path):
    st=state(tmp_path)
    for i in range(1,41): st.ingest(str(i),(10,1,i),1,row(),received_ns=i)
    for pn in (1,5,40,41):
        snap=decide(st,query(pitch=pn),id=str(pn),now=100+pn)
        tok=s.arrays(snap);n=min(pn-1,32)
        assert tok['valid'].sum()==n
        assert tok['context_keys'][0,:n,2].tolist()==list(range(pn-1,pn-1-n,-1))
        np.testing.assert_array_equal(tok['pos'][0],np.arange(1,33))
        assert not tok['context_keys'][0,n:].any()
    st.close()


def test_conditional_type_semantics_and_no_current_realization_fields(tmp_path):
    st=state(tmp_path,kind='retrospective')
    st.ingest('p1',(10,1,1),1,row(),received_ns=1)
    hypothetical=decide(st)
    observed=decide(st,query(kind='SL'),id='actual',now=101,basis='observed_actual')
    assert hypothetical['tokens']==observed['tokens']
    assert observed['identity']['evidence_kind']=='retrospective' and not observed['prospective_claim']
    for extra in ('outcome10','description','release_speed','plate_x'):
        q=query();q[extra]=0
        with pytest.raises(ValueError): decide(st,q,id=extra,now=102)
    with pytest.raises(ValueError): decide(st,id='ambiguous',basis='prediction',now=103)
    st.close()


def test_prestate_required_no_defaults_and_input_vocabulary(tmp_path):
    st=state(tmp_path)
    for field in ('balls','strikes','outs_when_up','home_score','away_score','on_1b','pitcher'):
        q=query();del q[field]
        with pytest.raises(ValueError): decide(st,q,id=field)
    for field,value in [('balls',4),('strikes',3),('pitch_type','UNREGISTERED'),('on_1b',0),('stand','?')]:
        q=query();q[field]=value
        with pytest.raises(ValueError): decide(st,q,id=field)
    for field,value in [('pitcher',False),('outcome10',10),('pitch_type','bad')]:
        r=row();r['accepted'][field]=value
        with pytest.raises(ValueError): st.ingest('p',(10,1,1),1,r,received_ns=1)
    snap=decide(st)
    altered=copy.deepcopy(snap);altered['tokens']['out'][0][0]=1
    with pytest.raises(ValueError): s.arrays(altered)
    st.close()


def test_exact_frozen_schema_and_token_vocabulary():
    from experiments.ws_fullseason import data
    modules=data.closure();safe=modules.safe
    assert list(s.CONTEXT)==safe.CONTEXT and list(s.SCHEMA)==safe.SCHEMA
    assert s.TYPES==tuple(safe.TYPES)


def test_bounded_intake_and_decision_capacity_keeps_original_evidence(tmp_path, monkeypatch):
    st=state(tmp_path)
    monkeypatch.setattr(s,'MAX_KEYS',1);monkeypatch.setattr(s,'MAX_EVENTS',2)
    monkeypatch.setattr(s,'MAX_DECISIONS',1)
    st.ingest('p1',(10,1,1),1,row(),received_ns=1)
    with pytest.raises(ValueError): st.ingest('p2',(10,1,2),1,row(),received_ns=2)
    st.ingest('p1',(10,1,1),2,row(kind='CH'),received_ns=3)
    with pytest.raises(ValueError): st.ingest('p1',(10,1,1),3,row(),received_ns=4)
    saved=decide(st)
    with pytest.raises(ValueError): decide(st,id='overflow',now=101)
    assert decide(st,now=102)==saved
    r=row(pitcher=1<<53)
    with pytest.raises(ValueError): st.ingest('p',(10,1,1),3,r,received_ns=103)
    st.close()


def test_unknown_is_ignored_only_outside_complete_last32_dependencies(tmp_path):
    st=state(tmp_path)
    st.ingest('old',(10,1,1),1,row(status='unknown'),received_ns=1)
    for i in range(2,34):st.ingest(str(i),(10,1,i),1,row(),received_ns=i)
    assert decide(st,query(pitch=33),id='31newer',now=100)['reason']=='prior_quality_unknown'
    ready=decide(st,query(pitch=34),id='32newer',now=101)
    assert ready['status']=='ready' and ready['context_dependency_floor']==[10,1,2]
    assert s.arrays(ready)['valid'].sum()==32
    assert decide(st,query(pitch=34),id='gap',now=102,complete=False)['reason']=='stored_prefix_incomplete'
    st.ingest('20',(10,1,20),2,row(status='unknown'),received_ns=103)
    assert decide(st,query(pitch=34),id='within',now=104)['reason']=='prior_quality_unknown'
    # The original ready decision remains immutable after a later unknown correction.
    assert decide(st,query(pitch=34),id='32newer',now=105)==ready
    st.ingest('20',(10,1,20),3,row(),received_ns=106)
    st.ingest('34',(10,1,34),1,row(status='unknown'),received_ns=107)
    assert decide(st,query(pitch=35),id='exactprevious',now=108)['reason']=='prior_quality_unknown'
    st.close()
