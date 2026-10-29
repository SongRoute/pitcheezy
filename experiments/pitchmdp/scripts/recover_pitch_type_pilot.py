#!/usr/bin/env python3
"""Bounded recovery of attempt002 stored predictions; NEVER fits a model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
from pitchmdp.pitch_type_pilot import KEY, FEATURES, MODELS, TIERS, TIER_NAMES, digest, load_registered, prepare
from pitchmdp.pitch_type_evaluation import probability_scores, _paired_interval

PINS = {'predictions.parquet':'e5d95310f0a52b68ad7124bbe3ae56d2628c3ad0083ed31e7650df8758dfee8f',
    'exclusions.jsonl':'ace2560a8a137d4767678343034a53464238b4abc6ded10a069d402453b9d579',
    'metrics.json':'9d80cd0fd60285fce6f0474c5718a6473229e50c0932fcb4e27253481e8af786'}
CONFIG_SHA = '7660abedfbc96eaabc970b52fa120501b56aa9cc767432919c7d1254b8a8bbce'


def recover(frame, audit, ledger, c):
    fit, query, populations, expected_ledger = prepare(frame, c)
    if ledger != expected_ledger: raise ValueError('exclusion ledger mismatch')
    expected_columns = KEY + FEATURES + ['label'] + [f'{m}_{v}' for m in MODELS for v in c['labels']] + [f'support_{t}' for t in TIER_NAMES]
    if list(audit.columns) != expected_columns: raise ValueError('audit columns mismatch')
    audit = audit.sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(audit[KEY+FEATURES+['label']], query[KEY+FEATURES+['label']])
    support = []
    for fields, tier in zip(TIERS,TIER_NAMES):
        counts = fit.groupby(fields,sort=False).size().to_dict()
        values = [int(counts.get(key[0] if len(fields)==1 else key,0)) for key in query[fields].itertuples(index=False,name=None)]
        np.testing.assert_array_equal(audit[f'support_{tier}'].to_numpy(),values)
        support.append(values)
    metrics, losses = {}, {}
    for model in MODELS:
        p = audit[[f'{model}_{v}' for v in c['labels']]].to_numpy()
        metrics[model], losses[model] = probability_scores(query.label.to_numpy(dtype=int),p,len(c['labels']))
    comparison = _paired_interval(losses['tree_candidate']-losses['context_control'],query.game_pk.to_numpy(),c['bootstrap'])
    support = np.asarray(support).T
    effective = [next((TIER_NAMES[j] for j in range(3,-1,-1) if row[j]>0),'global') for row in support]
    eligible = populations['EVAL']['label_eligible']
    return {'id':c['id'],'mode':c['mode'],'scientific_winner':None,'labels':c['labels'],
        'populations':populations,'metrics_common_valid_inputs':metrics,'comparison':comparison,
        'coverage':{m:dict(eligible=eligible,ok=len(query),error=eligible-len(query),missing=0,abstain=0,late=None,fraction=len(query)/eligible) for m in MODELS},
        'fallback_counts':{level:effective.count(level) for level in sorted(set(effective))},
        'unseen_pitcher_queries':int((support[:,0]==0).sum()),'fit_games':int(fit.game_pk.nunique()),
        'eval_games':int(query.game_pk.nunique()),'primary_pair':c['primary_pair'],
        'availability':c['availability'],'exposure':c['exposure'],'nll_floor':1e-12,
        'timeliness':'unmeasured_retrospective','cuda_used':False,
        'tree':None,'tree_metadata_status':'unavailable_after_original_serialization_failure'}


def completed_field(text, name):
    """Read one complete top-level value, never close/repair the partial JSON."""
    marker = f'  "{name}": '
    if text.count(marker) != 1: raise ValueError('ambiguous/missing completed field')
    start = text.index(marker)+len(marker)
    return json.JSONDecoder().raw_decode(text,start)[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','input','attempt','output','cpu-lock'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery-commit',required=True)
    a=p.parse_args(); started=time.time()
    if a.output.exists(): raise FileExistsError(a.output)
    if digest(a.config)!=CONFIG_SHA: raise ValueError('config pin mismatch')
    for name,sha in PINS.items():
        if digest(a.attempt/name)!=sha: raise ValueError('preserved artifact pin mismatch')
    c=json.loads(a.config.read_text())
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);signal.alarm(1800)
    with a.cpu_lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=recover(load_registered(a.input,c),pd.read_parquet(a.attempt/'predictions.parquet',use_threads=False),
            [json.loads(line) for line in (a.attempt/'exclusions.jsonl').read_text().splitlines()],c)
        partial=(a.attempt/'metrics.json').read_text()
        for name in ('metrics_common_valid_inputs','comparison','populations','coverage','fallback_counts','fit_games','eval_games'):
            if result[name]!=completed_field(partial,name): raise ValueError('recomputed aggregate mismatch: '+name)
        provenance=completed_field(partial,'provenance')
        if provenance['source_commit']!='0f3ec92d6137e6ce390373920392ecb7f03303a7' or provenance['config_sha256']!=CONFIG_SHA or provenance['data_sha256']!=c['source']['sha256']:
            raise ValueError('original provenance mismatch')
        result['original_provenance']=provenance
        result['recovery']={'status':'recomputed_from_preserved_predictions','fit_count':0,'total_previous_fits':2,
            'original_attempt_exit':1,'preserved_artifacts_sha256':PINS,'source_commit':a.recovery_commit,
            'helper_sha256':digest(__file__),'elapsed_seconds':time.time()-started,
            'max_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
        with a.output.open('x') as f: f.write(payload)
        print(json.dumps({'recovery_completed':True,'additional_fits':0,'output_sha256':digest(a.output)}))


if __name__=='__main__': main()
