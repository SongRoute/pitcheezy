"""Saved-policy diagnosis contracts; no actual evaluation or model inference."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'research'))
from diagnose_adaptive_location import cluster_stats, conditional_frame, policy_frame, group_table


def test_game_cluster_mean_preserves_row_weighting_and_one_game_has_no_interval():
    result = cluster_stats([1., 3., 0.], [1, 1, 2])
    assert result['difference'] == pytest.approx(4/3)
    assert result['ci95'] == pytest.approx([0., 2.])
    assert result['games'] == 2 and result['rows'] == 3
    single = cluster_stats([1., 3.], [1, 1])
    assert single['ci95'] is None
    assert single['cluster_caution'] == 'no_interval_one_game'


def test_stratum_contributions_sum_to_overall_difference():
    frame = pd.DataFrame({'game_id':[1,1,2], 'pitcher':[10,10,20], 'delta':[-.2,-.4,.3]})
    rows = group_table(frame, ['pitcher'], ['delta'], len(frame))
    assert sum(row['contrasts']['delta']['contribution_to_overall_mean'] for row in rows) == pytest.approx(frame.delta.mean())


def test_missing_policy_hand_remains_unknown_and_later_policy_changes_are_separate():
    record = {'game_id':1,'pa_id':1,'pitcher':10,'date':'2025-08-17','inning':1,'outs':0,'bases':0,
              'status':'ready','action_count':2,'policy_different':True,'first_action_different':False,
              'chosen': {'baseline':{'type':'FF','zone':'low_left','judge_observed_cell_count':30},
                         'adaptive':{'type':'FF','zone':'low_left','judge_observed_cell_count':30}},
              'evaluated': {sigma:{'baseline':.5,'adaptive':.49} for sigma in ('0.3','0.45','0.65')}}
    observed = pd.DataFrame({'game_pk':[2], 'at_bat_number':[1], 'pitcher':[10], 'stand':['L']})
    frame, unavailable = policy_frame([record], observed)
    assert not unavailable
    assert frame.iloc[0]['hand'] == 'unknown_not_in_saved_proxy_rows'
    assert frame.iloc[0]['change_pattern'] == 'same_root_changed_later'
    assert frame.iloc[0]['we_delta_pp_0.45'] == pytest.approx(-1.)
    record['policy_different'] = False
    with pytest.raises(ValueError, match='Identical policies'):
        policy_frame([record], observed)


def test_conditional_support_strata_and_key_alignment_are_not_inferred_from_policy():
    rows = pd.DataFrame({'game_pk':[1,1,2], 'at_bat_number':[1,1,1], 'pitch_number':[1,2,1],
                         'pitcher':[10]*3,'pitch_type':['FF']*3,'stand':['R']*3,
                         'balls':[0,0,1], 'strikes':[0,1,2], 'outs_when_up':[0]*3,'bases':[0]*3})
    broad = np.full((3,10), .1)
    narrow = broad.copy()
    narrow[0] = np.array([.2]+[.8/9]*9)
    proxy = {'game_ids':np.array([1,1,2]), 'at_bat_number':np.array([1,1,1]),'pitch_number':np.array([1,2,1]),
             'y':np.array([0,1,2]), 'sigma030_supported':np.array([True,False,False]),
             'sigma045_supported':np.array([True,True,False]), 'sigma030':narrow, 'sigma045':broad}
    adaptive = {'game_ids':proxy['game_ids'], 'y':proxy['y'], 'p':narrow,
                'supported':np.array([True,True,False])}
    result = conditional_frame(proxy, adaptive, rows)
    assert result.fallback.tolist() == ['narrow_kernel','broad_kernel_fallback','type_only_fallback']
    assert result.log_loss_delta.iloc[0] == pytest.approx(-np.log(2))
    assert result.log_loss_delta.iloc[1:].eq(0).all()
    assert result['count'].tolist() == ['0-0','0-1','1-2']
    wrong = rows.copy()
    wrong.loc[0, 'pitch_number'] = 8
    with pytest.raises(AssertionError):
        conditional_frame(proxy, adaptive, wrong)
