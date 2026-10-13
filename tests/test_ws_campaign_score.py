"""Generated probabilities only: fixed game-paired panel and seed gate."""
import numpy as np
import pytest
from experiments.ws_campaign.score import panel
from experiments.ws_campaign.runtime import Stop


def test_panel_identity_and_three_seed_gate():
    q = np.full((14,7),1/7); y = np.arange(14)%7; games=np.repeat([1,2],7)
    predictions={a:{s:q.copy() for s in (0,1,2)} for a in ('T','C')}
    result=panel(predictions,y,games,[('T','C')])
    assert result['bootstrap']['draws'] == 10000
    assert result['ensemble']['T']['nll'] == pytest.approx(np.log(7))
    for metric in ('nll','brier','ece_top_label_10bin'):
        assert result['paired_differences']['T-C'][metric] == {'difference':0.,'ci95':[0.,0.]}
    del predictions['C'][2]
    with pytest.raises(Stop,match='matched three seeds'): panel(predictions,y,games,[('T','C')])


def test_bad_composition_stops_before_artifact_decode():
    from experiments.ws_campaign.score import inspect_artifacts
    jobs=[{'id':str(i),'arm':'T' if i < 75 else 'K','kind':'supervised'} for i in range(87)]
    with pytest.raises(Stop,match='predecode 66/6/3 composition'):
        inspect_artifacts({'jobs':jobs},{'jobs':{}})
