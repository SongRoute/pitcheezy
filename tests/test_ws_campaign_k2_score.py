"""Generated K2 metadata and prediction fixtures only; no real campaign inputs."""
import json
import numpy as np
import pytest
from experiments.ws_campaign.score_k2 import selected_jobs,read_comparator,WINDOWS
from experiments.ws_campaign.runtime import Stop


def fixture():
    jobs=[{'id':f'{w}-K-s{s}','arm':'K','kind':'supervised','window':w,'seed':s}
          for w in WINDOWS for s in (0,1,2)]
    return {'jobs':jobs},{'jobs':{j['id']:{'status':'COMPLETED'} for j in jobs}}


def test_exact_k2_seed_window_and_completion_gate():
    m,s=fixture();assert len(selected_jobs(m,s))==12
    m['jobs'][0]['seed']=1
    with pytest.raises(Stop,match='four complete seed triples'):selected_jobs(m,s)
    m,s=fixture();s['jobs'][m['jobs'][0]['id']]['status']='RUNNING'
    with pytest.raises(Stop,match='K2 completion'):selected_jobs(m,s)
    m,s=fixture();s['jobs']['extra']={'status':'COMPLETED'}
    with pytest.raises(Stop,match='exact K2 state'):selected_jobs(m,s)


def test_comparator_requires_exact_ordered_keys(tmp_path):
    keys=np.array([[1,2,3],[1,2,4]]);q=np.full((2,7),1/7)
    p=tmp_path/'generated.npz';np.savez(p,keys=keys,probabilities=q)
    np.testing.assert_array_equal(read_comparator(p,keys),q)
    with pytest.raises(Stop,match='identical comparator ordered keys'):read_comparator(p,keys[::-1])
    np.savez(p,keys=keys,probabilities=q,unexpected=np.array([1]))
    with pytest.raises(Stop,match='identical comparator ordered keys'):read_comparator(p,keys)
