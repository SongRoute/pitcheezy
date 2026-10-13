"""One reviewed fixed campaign development panel; never fit or select on EVAL."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.ws_campaign.runtime import ROOT, require, sha, read_bound, SOURCES as TRAIN_SOURCES
from experiments.ws_campaign.diagnostic_supervisor import supervise

SOURCES = (*TRAIN_SOURCES, 'experiments/ws_campaign/diagnostic_supervisor.py', 'experiments/ws_campaign/score.py', 'experiments/ws_seq/evaluate_seq.py')
PAIRS = {
    'A': [('T','N0'), ('T','T-recent4'), ('R','C'), ('C','C-shuf'), ('T','C'), ('G','C'),
          ('T-leaf7','T'), ('T-tokdrop','T'), ('T-histmask','T')],
    'B': [('T-mask-pre','T-mask-scratch'), ('T-contrast-pre','T-contrast-scratch'),
          ('T-dro','T'), ('N0-distill','N0')],
    'C': [('T','C')],
}


def inspect_artifacts(manifest, state):
    """Metadata/hash all75 before any scientific payload decoding."""
    jobs = manifest['jobs']; receipts = {}
    require(len(jobs) == 87 and len({j['id'] for j in jobs}) == 87, '87 unique slots')
    for job in jobs:
        record = state['jobs'][job['id']]
        if job['arm'] == 'K':
            require(record['status'] in ('FAILED','QUARANTINED'), 'K remains missing')
            continue
        require(record['status'] == 'COMPLETED', 'non-K completion')
        directory = Path(job['output']); cert = record['certification']
        require(sha(directory/'receipt.json') == cert['receipt_sha256'] and
                sha(directory/'SUPERVISOR.json') == cert['supervisor_sha256'], 'queue artifact pin')
        supervisor = json.loads((directory/'SUPERVISOR.json').read_text())
        require(supervisor.get('status') == 'COMPLETED' and supervisor.get('receipt_sha256') == cert['receipt_sha256'], 'completed supervisor pin')
        r = json.loads((directory/'receipt.json').read_text())
        for k in ('job_id','arm','seed','attempt_id','input_manifest_sha256','exposure_ledger_sha256'):
            require(r.get(k) == job.get('id' if k == 'job_id' else k), 'receipt identity '+k)
        require(r['status'] == 'COMPLETED' and r['eval_scores_opened'] is False and
                r['grant_sha256'] == job['grant_sha256'], 'receipt provenance')
        names = {'supervised':{'model-state.pt','temperatures.json','curve.json','calibration-margins.npz'},
                 'pretrain':{'pretrained-state.pt'}, 'teacher':{'teacher-fit.npz'}}[job['kind']]
        require(set(r['outputs_sha256']) == names, 'registered payload names')
        require({p.name for p in directory.iterdir()} == names | {'receipt.json','SUPERVISOR.json'}, 'exact directory names')
        for name in names:
            require((directory/name).is_file() and not (directory/name).is_symlink() and
                    sha(directory/name) == r['outputs_sha256'][name], 'payload byte hash')
        receipts[job['id']] = r
    require(len(receipts) == 75, '75 completed artifacts')
    for job in jobs:
        if job['id'] in receipts:
            require(all(dep in receipts for dep in job['depends_on']), 'completed dependencies')
    for objective in ('mask','contrast'):
        for seed in (0,1,2):
            p = receipts[f'B-T-{objective}-pre-s{seed}']
            s = receipts[f'B-T-{objective}-scratch-s{seed}']
            steps = receipts[f'B-{objective}-pre-s{seed}']['optimization_steps']
            require(p['optimization_steps'] + steps == s['optimization_steps'], 'paired actual optimizer steps')
    return receipts


def panel(predictions, labels, games, pairs):
    import numpy as np
    from experiments.ws_seq.evaluate_seq import arm_metrics, paired_block
    require(all(set(seed_q) == {0,1,2} for seed_q in predictions.values()), 'matched three seeds')
    for arm in predictions:
        for q in predictions[arm].values():
            require(q.shape == (len(labels),7) and np.isfinite(q).all() and (q >= 0).all() and
                    np.allclose(q.sum(1),1,rtol=0,atol=1e-9), 'probability coherence')
    ensemble = {arm:np.mean([seed_q[s] for s in (0,1,2)],axis=0) for arm,seed_q in predictions.items()}
    points = {arm:{str(s):arm_metrics(q,labels) for s,q in seed_q.items()} for arm,seed_q in predictions.items()}
    metrics, differences, unit = paired_block(ensemble,labels,games,pairs,{'seed':20261211,'chunk':32},10000)
    return {'per_seed':points,'ensemble':metrics,'paired_differences':differences,'bootstrap':unit,
            'interpretation':'2026 EXPOSED DEVELOPMENT; conditional fixed-prediction descriptive CI; no winner or adoption'}


def run(grant, manifest, state, out):
    import numpy as np
    import torch
    from experiments.ws_campaign import train
    from experiments.ws_campaign.models import build_model
    from experiments.ws_seq import train_seq as base
    torch.set_num_threads(2); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one CUDA device')
    torch.cuda.set_per_process_memory_fraction((12 << 30)/torch.cuda.get_device_properties(0).total_memory)
    cfg = read_bound(ROOT/manifest['config'],manifest['config_sha256'])
    receipts = inspect_artifacts(manifest,state)
    reports = {}; completed = []
    for window in ('base','chrono1','chrono2','pitcher_holdout'):
        jobs = [j for j in manifest['jobs'] if j['window'] == window and j['kind'] == 'supervised' and j['arm'] != 'K']
        require(jobs, 'nonempty window')
        first = jobs[0]; inputs = Path(first['inputs'])
        require(all(j['input_manifest_sha256'] == first['input_manifest_sha256'] and j['inputs'] == first['inputs'] for j in jobs), 'same window package')
        require(sha(inputs/'manifest.json') == first['input_manifest_sha256'], 'input manifest')
        require(sha(first['exposure_ledger']) == first['exposure_ledger_sha256'], 'exposure ledger')
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
        if window == 'base':
            for block in ('A','B'):
                arms = {j['arm'] for j in jobs if j['block'] == block}
                arms |= {a for pair in PAIRS[block] for a in pair}
                reports[block] = panel({a:pred[a] for a in sorted(arms)},labels,games,PAIRS[block])
        else: reports[window] = panel(pred,labels,games,PAIRS['C'])
        del data,keys,rows,x,c,s,ctx,pred; gc.collect(); torch.cuda.empty_cache()
    require(len(completed) == 66, 'exact 66 supervised predictions')
    result = {'status':'COMPLETED','data_role':'2026 EXPOSED DEVELOPMENT','independent_test':False,
        'scientific_payloads_decoded':66,'dependency_artifacts_byte_verified_only':9,'missing_K_slots':12,
        'missingness':'4 seed0 wall failures and8 dependent quarantines; architecture-dependent missingness, no K comparison',
        'panels':reports,'predictions':completed,'fit_count':0,'calibration_fit_count':0,'model_selection':False,
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__},
        'grant_sha256':grant['_sha256']}
    (out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--grant',type=Path,required=True)
    p.add_argument('--worker',action='store_true'); args = p.parse_args()
    g = json.loads(args.grant.read_text())
    require(g.get('execute') is True and g.get('verdict') == 'PASS' and g.get('author') and
            g.get('reviewer') and g['author'] != g['reviewer'], 'independent score grant')
    require(g.get('sources_sha256') == {s:sha(ROOT/s) for s in SOURCES}, 'reviewed source')
    manifest = read_bound(g['manifest'],g['manifest_sha256']); state = read_bound(g['state'],g['state_sha256'])
    require(state['manifest_sha256'] == g['manifest_sha256'], 'state manifest pin')
    require(g.get('max_wall_seconds') == 1800 and g.get('max_rss_bytes') == 8 << 30, 'fixed limits')
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
        out.mkdir(parents=True,exist_ok=False)
        return supervise([sys.executable,'-m','experiments.ws_campaign.score','--grant',str(args.grant),'--worker'],out,1800,8 << 30,sha(args.grant),f.fileno())


if __name__ == '__main__': sys.exit(main())
