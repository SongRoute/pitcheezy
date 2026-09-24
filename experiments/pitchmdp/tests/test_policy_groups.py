"""Synthetic descriptive reporting only; no simulation, model or training."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'scripts')]
from pitchmdp.matrix_policy_groups import p0_groups, rollout_groups, POLICY_PAIRS, RL_PAIRS, KEY
import report_ml_policy_groups as cli

PANEL = {'train_players': [{'pitcher': 11, 'train_role': 'starter'}, {'pitcher': 22, 'train_role': 'relief'}]}


def requests():
    return pd.DataFrame({'game_pk': [1,2,3,4,5], 'at_bat_number': [1]*5, 'pitch_number': [1]*5,
        'pitcher': [11,22,11,22,22], 'train_role': ['starter','relief','starter','relief','relief'],
        'selected': [True,True,True,True,False], 'supported': [True,False,True,True,False],
        'reason': [None,'no_support',None,None,'not_selected_by_fixed_budget']})


def rollouts():
    values = {name: np.array([[.2,.4],[.6,.8]])+i*.02 for i,name in enumerate(['P0','P1','P2','P3'])}
    flags = {name: np.array([[False,False],[False,True]]) for name in values}
    return values, flags


def test_requested_and_prefix_unsupported_retention_and_group_paired_means():
    values, flags = rollouts()
    report = rollout_groups(requests(), ['1:1:1','3:1:1'], [1,3], values, flags, POLICY_PAIRS, 3, PANEL)
    overall = report['groups']['overall']
    assert [overall[k] for k in ['requested_pa_starts','preparation_selected','execution_selected','not_execution_selected','supported_selected','selected_unsupported']] == [5,4,3,2,2,1]
    assert overall['policies']['P0']['mean_original_defensive_we'] == pytest.approx(.5)
    assert overall['paired_delta_we']['P2_minus_P1'] == pytest.approx(.02)
    assert overall['policies']['P0']['truncated_rollout_rate'] == .25
    relief = report['groups']['train_role:relief']
    assert relief['requested_pa_starts'] == 3 and relief['selected_unsupported'] == 1
    assert relief['policies']['P0']['mean_original_defensive_we'] is None
    assert relief['unsupported_reasons'] == {'no_support': 1}
    assert report['groups']['pitcher:11']['supported_selected'] == 2


def test_exact_pa_order_and_frozen_role_checked():
    values, flags = rollouts()
    with pytest.raises(ValueError, match='Exact common'):
        rollout_groups(requests(), ['3:1:1','1:1:1'], [3,1], values, flags, POLICY_PAIRS, 3, PANEL)
    wrong = requests(); wrong.loc[0, 'train_role'] = 'relief'
    with pytest.raises(ValueError, match='frozen TRAIN role'):
        rollout_groups(wrong, ['1:1:1','3:1:1'], [1,3], values, flags, POLICY_PAIRS, 3, PANEL)
    flags['P1'] = flags['P1'].astype(int)
    with pytest.raises(ValueError, match='truncation'):
        rollout_groups(requests(), ['1:1:1','3:1:1'], [1,3], values, flags, POLICY_PAIRS, 3, PANEL)


def p0_fixture():
    frame = requests().iloc[:3][KEY+['pitcher']].copy()
    frame['observed_action'] = ['FF','SL','UNKNOWN']
    arrays = {'keys': frame[KEY].to_numpy(), 'labels': np.array([0,1,-1]),
        'bc': np.array([[.8,.2],[.25,.75],[0.,0.]]), 'frequency': np.array([[.5,.5],[.5,.5],[0.,0.]]),
        'supported': np.array([True,True,False]), 'observed_supported': np.array([True,True,False]),
        'fallback': np.array([False,True,False])}
    return frame, arrays


def test_p0_same_conditional_subset_full_requested_null_and_fallback():
    frame, arrays = p0_fixture()
    report = p0_groups(frame, arrays, ['FF','SL'], PANEL)['groups']
    assert report['overall']['models']['bc']['conditional_action_nll'] == pytest.approx(-np.log([.8,.75]).mean())
    assert report['overall']['models']['bc']['full_requested_action_nll'] is None
    assert report['overall']['models']['frequency']['full_requested_action_nll'] is None
    assert report['pitcher:22']['models']['frequency']['full_requested_action_nll'] == pytest.approx(-np.log(.5))
    assert report['overall']['unsupported_observed_labels'] == 1
    assert report['pitcher:22']['unknown_pitcher_fallbacks'] == 1
    bad = deepcopy(arrays); bad['keys'] = bad['keys'][::-1]
    with pytest.raises(ValueError, match='key alignment'): p0_groups(frame, bad, ['FF','SL'], PANEL)
    bad = deepcopy(arrays); bad['observed_supported'][2] = True
    with pytest.raises(ValueError, match='inconsistent'): p0_groups(frame, bad, ['FF','SL'], PANEL)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def seal(directory, prep=None):
    manifest = {'artifact_hashes': {p.name: cli.digest(p) for p in directory.iterdir() if p.name != 'manifest.json'}}
    if prep is not None: manifest.update(stage=directory.name, preparation_sha256=cli.digest(prep))
    dump(directory/'manifest.json', manifest)


def prepared(tmp_path, mode='p0'):
    root = tmp_path/'policy'; root.mkdir()
    dump(root/'parent_preparation.json', {'panel': PANEL})
    frame, arrays = p0_fixture()
    for split in ['blend','dev']: frame.to_parquet(root/f'p0_{split}.parquet', index=False)
    requests().to_parquet(root/'dev_requests.parquet', index=False)
    dump(root/'preparation.json', {'bc_actions': ['FF','SL'], 'artifact_hashes': {
        p.name: cli.digest(p) for p in root.iterdir()}})
    directory = root/'stages'/mode; directory.mkdir(parents=True)
    start = {'preparation_sha256': cli.digest(root/'preparation.json')}
    if mode == 'p0':
        result = {}
        for split in ['blend','dev']: np.savez(directory/f'{split}.npz', **arrays)
    else:
        execution = {'preparation_sha256': cli.digest(root/'preparation.json'), 'requested_starts': {'dev': 3}}
        start.update(execution=execution, split='dev', world='control')
        result = {'pa_keys': ['1:1:1','3:1:1'], 'game_ids': [1,3], 'requested_pa_starts': 5,
            'selected_pa_starts': 3, 'supported_selected': 2, 'selected_unsupported_reasons': {'no_support': 1},
            'execution_sha256': cli.canonical(execution), 'split': 'dev', 'world': 'control'}
        values, flags = rollouts()
        for i,key in enumerate(result['pa_keys']):
            np.savez(directory/(key.replace(':','-')+'.npz'), **{n+'_values': v[i] for n,v in values.items()},
                **{n+'_truncated': v[i] for n,v in flags.items()})
    dump(directory/'started.json', start); dump(directory/'results.json', result)
    seal(directory, root/'preparation.json')
    return root, directory, result, start


def test_p0_cli_seal_and_all_consumed_hashes(tmp_path):
    root, stage, _, _ = prepared(tmp_path)
    result = cli.report(root, 'p0', tmp_path/'report')
    assert result['reports']['dev']['groups']['overall']['requested_pitches'] == 3
    assert str(root/'p0_dev.parquet') in result['input_hashes']
    assert str(stage/'manifest.json') in result['input_hashes']
    with pytest.raises(ValueError, match='outside immutable'): cli.report(root, 'p0', root/'new_report')
    (stage/'results.json').write_text('{}changed')
    with pytest.raises(ValueError, match='hash differs'): cli.report(root, 'p0', tmp_path/'report2')


def test_policy_and_rl_cli_common_planner_and_metadata(tmp_path):
    root, stage, result, started = prepared(tmp_path, 'dev-control')
    rl = tmp_path/'rl'; rl.mkdir()
    tuning = {'tau': .01}
    dump(rl/'preparation.json', {'parent_policy_run': str(root), 'comparator': 'P2', 'tuning_dependency': tuning,
        'external_hashes': {str(root/'preparation.json'): cli.digest(root/'preparation.json')}})
    execution = {'preparation_sha256': cli.digest(rl/'preparation.json')}
    dump(rl/'final_execution.json', execution)
    dest = rl/'evaluation/control'; dest.mkdir(parents=True)
    dump(dest/'started.json', {'execution_sha256': cli.canonical(execution), 'policy_execution_sha256': result['execution_sha256']})
    rr = {**result, 'execution_sha256': cli.canonical(execution), 'parent_policy_execution_sha256': result['execution_sha256'],
        'comparator': 'P2', 'tuning_dependency': tuning, 'dependencies': {str(stage/'manifest.json'): cli.digest(stage/'manifest.json')}}
    dump(dest/'results.json', rr)
    values, flags = rollouts()
    names = ['NNBC','IQL','CQL']+[f'{m}-seed{s}' for m in ['NNBC','IQL','CQL'] for s in range(3)]+['planner']
    for i,key in enumerate(result['pa_keys']):
        np.savez(dest/(key.replace(':','-')+'.npz'), **{n+'_values': values['P2'][i] for n in names},
            **{n+'_truncated': flags['P2'][i] for n in names})
    seal(dest)
    report = cli.report(root, 'dev-control', tmp_path/'report', rl)
    assert report['reports']['rl']['pairs'] == [list(p) for p in RL_PAIRS]
    assert report['reports']['rl']['groups']['overall']['paired_delta_we']['IQL_minus_planner'] == 0
    arrays = cli.Reader().arrays(dest/'1-1-1.npz'); arrays['planner_values'] += .1
    np.savez(dest/'1-1-1.npz', **arrays); seal(dest)
    with pytest.raises(ValueError, match='planner differs'): cli.report(root, 'dev-control', tmp_path/'bad-report', rl)
