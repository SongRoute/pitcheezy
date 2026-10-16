"""One reviewed fixed campaign development panel; never fit or select on EVAL."""
import argparse
from collections import Counter
import fcntl
import gc
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

from experiments.ws_campaign.runtime import ROOT, require, sha, read_bound, check_exposure_ledger, SOURCES as TRAIN_SOURCES
from experiments.ws_campaign.diagnostic_supervisor import supervise

from experiments.ws_campaign.score import panel

SOURCES = (*TRAIN_SOURCES, 'experiments/ws_campaign/diagnostic_supervisor.py',
           'experiments/ws_campaign/score.py', 'experiments/ws_campaign/score_k2.py',
           'experiments/ws_seq/evaluate_seq.py')
WINDOWS = ('base','chrono1','chrono2','pitcher_holdout')


def selected_jobs(manifest, state):
    jobs = [j for j in manifest['jobs'] if j['arm'] == 'K']
    require(len(jobs) == len({j['id'] for j in jobs}) == 12, 'exact twelve K2 slots')
    require(set(state['jobs']) == {j['id'] for j in jobs}, 'exact K2 state')
    require(all(j['kind'] == 'supervised' for j in jobs), 'supervised K only')
    require(Counter((j['window'],j['seed']) for j in jobs) ==
            Counter((w,s) for w in WINDOWS for s in (0,1,2)), 'four complete seed triples')
    require(all(state['jobs'][j['id']]['status'] == 'COMPLETED' for j in jobs), 'K2 completion')
    return jobs


def inspect_artifacts(manifest, state):
    receipts = {}
    for job in selected_jobs(manifest,state):
        directory = Path(job['output']); cert = state['jobs'][job['id']]['certification']
        require(sha(directory/'receipt.json') == cert['receipt_sha256'] and
                sha(directory/'SUPERVISOR.json') == cert['supervisor_sha256'], 'K2 certification')
        r = json.loads((directory/'receipt.json').read_text())
        supervisor = json.loads((directory/'SUPERVISOR.json').read_text())
        require(supervisor['status'] == 'COMPLETED' and supervisor['receipt_sha256'] == cert['receipt_sha256'], 'supervisor')
        for k in ('job_id','arm','seed','attempt_id','input_manifest_sha256','exposure_ledger_sha256'):
            require(r.get(k) == job.get('id' if k == 'job_id' else k), 'receipt identity '+k)
        require(r['status'] == 'COMPLETED' and r['eval_scores_opened'] is False and
                r['evaluation_predictions_computed'] is False and r['grant_sha256'] == job['grant_sha256'] and
                r['execution_policy'] == 'K2_GENERATED_PROBE_FAST_V1', 'K2 provenance')
        names = {'model-state.pt','temperatures.json','curve.json','calibration-margins.npz'}
        require(set(r['outputs_sha256']) == names and
                {p.name for p in directory.iterdir()} == names | {'receipt.json','SUPERVISOR.json'}, 'exact payloads')
        for name in names:
            require((directory/name).is_file() and not (directory/name).is_symlink() and
                    sha(directory/name) == r['outputs_sha256'][name], 'K2 payload digest')
        receipts[job['id']] = r
    return receipts


def comparator_records(grant):
    old = read_bound(grant['comparator_receipt'],grant['comparator_receipt_sha256'])
    require(old['status'] == 'COMPLETED' and old['fit_count'] == old['calibration_fit_count'] == 0,
            'original fixed score receipt')
    records = {r['id']:r for r in old['predictions']}
    require(len(records) == len(old['predictions']) == 66, 'original66 predictions')
    selected = {}
    for window in WINDOWS:
        prefix = 'A' if window == 'base' else 'C-'+window
        for arm in ('C','T'):
            for seed in (0,1,2):
                ident = f'{prefix}-{arm}-s{seed}'
                record = records[ident]; path = Path(grant['comparator_receipt']).parent/(ident+'-evaluation.npz')
                require(path.is_file() and not path.is_symlink() and sha(path) == record['prediction_sha256'],
                        'comparator prediction digest')
                selected[(window,arm,seed)] = path
    return selected


def read_comparator(path, expected_keys):
    import numpy as np
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files) == {'keys','probabilities'} and np.array_equal(saved['keys'],expected_keys),
                'identical comparator ordered keys')
        return saved['probabilities'].copy()


def run(grant, manifest, state, out):
    import numpy as np
    import torch
    from experiments.ws_campaign import train
    from experiments.ws_campaign.models import build_model
    from experiments.ws_seq import train_seq as base
    torch.set_num_threads(2); torch.manual_seed(0)
    train.configure_backend({'arm':'K'},{'execution_policy':'K2_GENERATED_PROBE_FAST_V1'})
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one CUDA device')
    torch.cuda.set_per_process_memory_fraction((12 << 30)/torch.cuda.get_device_properties(0).total_memory)
    cfg = read_bound(ROOT/manifest['config'],manifest['config_sha256'])
    receipts = inspect_artifacts(manifest,state)
    comparators = comparator_records(grant)
    reports = {}; completed = []
    for window in WINDOWS:
        jobs = [j for j in manifest['jobs'] if j['window'] == window and j['kind'] == 'supervised' and j['arm'] == 'K']
        require(jobs, 'nonempty window')
        first = jobs[0]; inputs = Path(first['inputs'])
        require(all(j['input_manifest_sha256'] == first['input_manifest_sha256'] and j['inputs'] == first['inputs'] for j in jobs), 'same window package')
        require(sha(inputs/'manifest.json') == first['input_manifest_sha256'], 'input manifest')
        for job in jobs:
            check_exposure_ledger(job['exposure_ledger'],job['exposure_ledger_sha256'],job,cfg,inputs/'manifest.json')
        x,c,s,ctx = train.load_inputs(inputs,cfg)
        require(c['keys'].ndim == 2 and c['keys'].shape[1] == 3 and
                len(np.unique(c['keys'],axis=0)) == len(c['keys']), 'unique stable package keys')
        ev = c['evaluation']; rows_np = np.flatnonzero(ev)
        require(len(rows_np) > 0, 'nonempty EVAL')
        data = base.Data(torch,torch.device('cuda'),x,c,s,ctx)
        rows = torch.as_tensor(rows_np,device='cuda',dtype=torch.long)
        keys = torch.as_tensor(c['keys'],device='cuda',dtype=torch.long)
        pred = {}
        for job in jobs:
            ident = job['id']; directory = Path(job['output'])
            model = build_model(cfg,job['arm'],job['seed']).to('cuda')
            model.load_state_dict(torch.load(directory/'model-state.pt',map_location='cuda',weights_only=True),strict=True)
            temperatures = json.loads((directory/'temperatures.json').read_text())
            expected = {'leaf'} if job['arm'] == 'T-leaf7' else set(base.HEADS)
            require(set(temperatures) == expected and all(.05 < float(t) < 20 and np.isfinite(t) for t in temperatures.values()), 'frozen temperatures')
            z = train.predict(model,data,keys,rows,cfg['base']['training']['eval_batch_size'])
            q = train.probabilities(z,temperatures)
            pred.setdefault(job['arm'],{})[job['seed']] = q
            target = out/(ident+'-evaluation.npz')
            np.savez(target,keys=c['keys'][ev],probabilities=q)
            completed.append({'id':ident,'prediction_sha256':sha(target),'model_sha256':receipts[ident]['outputs_sha256']['model-state.pt']})
            del model,z; torch.cuda.empty_cache()
        labels = c['y7'][ev]; games = c['keys'][ev,0]
        for arm in ('C','T'):
            pred[arm] = {seed:read_comparator(comparators[(window,arm,seed)],c['keys'][ev]) for seed in (0,1,2)}
        reports[window] = panel(pred,labels,games,[('K','C'),('K','T')])
        del data,keys,rows,x,c,s,ctx,pred; gc.collect(); torch.cuda.empty_cache()
    require(len(completed) == 12, 'exact12 K2 predictions')
    result = {'status':'COMPLETED','data_role':'2026 EXPOSED DEVELOPMENT','independent_test':False,
        'scientific_payloads_decoded':12,'comparators_reused':24,
        'execution_policy':'K2_GENERATED_PROBE_FAST_V1',
        'interpretation':'exploratory architecture+backend; repeated EXPOSED DEVELOPMENT; no winner or promotion',
        'panels':reports,'predictions':completed,'fit_count':0,'calibration_fit_count':0,'model_selection':False,
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__},
        'grant_sha256':grant['_sha256']}
    (out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--grant',type=Path,required=True)
    p.add_argument('--worker',action='store_true'); args = p.parse_args()
    g = json.loads(args.grant.read_text())
    require(g.get('authorized_by') == 'root', 'Root grant')
    require(g.get('execute') is True and g.get('verdict') == 'PASS' and g.get('author') and
            g.get('reviewer') and g['author'] != g['reviewer'], 'independent score grant')
    require(g.get('sources_sha256') == {s:sha(ROOT/s) for s in SOURCES}, 'reviewed source')
    manifest = read_bound(g['manifest'],g['manifest_sha256']); state = read_bound(g['state'],g['state_sha256'])
    require(state['manifest_sha256'] == g['manifest_sha256'], 'state manifest pin')
    require(g.get('max_wall_seconds') == 1800 and g.get('max_rss_bytes') == 8 << 30 and
            g.get('max_output_bytes') == 200000000 and g.get('gpu_index') == 3, 'fixed GPU3 limits')
    require(sys.platform == 'linux' and os.environ.get('GPU_JOB_ID') and
            os.environ.get('CUDA_VISIBLE_DEVICES') == str(g['gpu_index']), 'gpu run only')
    env = dict(line.split('=',1) for line in (Path(os.environ['GPU_JOB_DIR'])/'job.env').read_text().splitlines() if '=' in line)
    require(len(env.get('COMMIT','')) >= 7 and g['commit'].startswith(env['COMMIT']), 'commit')
    out = Path(g['output']); require(out.is_absolute(), 'absolute output')
    if args.worker:
        g['_sha256'] = sha(args.grant); run(g,manifest,state,out); return 0
    require(not out.exists(), 'new immutable attempt')
    uuid = subprocess.check_output(['nvidia-smi','-i',str(g['gpu_index']),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    require(uuid == g['gpu_uuid'], 'GPU UUID'); lock = Path(g['common_lock']); require(lock.is_file(),'existing GPU lock')
    with lock.open() as f:
        fcntl.flock(f,fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
        require(uuid not in processes.splitlines(),'GPU idle')
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        require(available >= 16 << 30,'16GiB host headroom')
        require(shutil.disk_usage(out.parent).free >= (1 << 30) + g['max_output_bytes'], 'output disk floor')
        out.mkdir(parents=True,exist_ok=False)
        return supervise([sys.executable,'-m','experiments.ws_campaign.score_k2','--grant',str(args.grant),'--worker'],out,1800,8 << 30,sha(args.grant),f.fileno(),g['max_output_bytes'])


if __name__ == '__main__': sys.exit(main())
