"""Three sequential fixed T seeds on one granted GPU, inherited common lock per child."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.ws_fullseason.audit import sha, checked_json
from experiments.ws_campaign.diagnostic_supervisor import supervise

ROOT=Path(__file__).resolve().parents[2]


def checked_grant(path):
    g=checked_json(path)
    if not (g.get('execute') is True and g.get('verdict')=='PASS' and g.get('authorized_by')=='root'
            and g.get('author')=='research_readiness' and g.get('reviewer') and g['reviewer']!=g['author']):
        raise ValueError('independent Root fit grant')
    if g.get('seeds')!=[0,1,2] or type(g.get('gpu_index')) is not int or g['gpu_index'] not in (0,3):
        raise ValueError('three fixed seeds on granted GPU0 or GPU3')
    for rel,digest in g['source_pins'].items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=digest: raise ValueError('source pin '+rel)
    recipe=checked_json(ROOT/g['recipe'])
    if recipe['id']!=g['recipe_id'] or recipe['training_rule']['epochs']!=30:
        raise ValueError('frozen T30 recipe')
    if g['wall_seconds_per_seed']!=3600 or g['rss_bytes']!=8<<30 or g['vram_bytes']!=16<<30 or g['output_bytes_per_seed']!=1<<29:
        raise ValueError('fixed fit caps')
    if g.get('aggregate_wall_seconds')!=120 or g.get('aggregate_output_bytes')!=1<<20:raise ValueError('fixed aggregate caps')
    if sys.platform!='linux' or not os.environ.get('GPU_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(g['gpu_index']):
        raise ValueError('gpu run exact granted device only')
    env=dict(line.split('=',1) for line in (Path(os.environ['GPU_JOB_DIR'])/'job.env').read_text().splitlines() if '=' in line)
    if len(env.get('COMMIT',''))<7 or not g['commit'].startswith(env['COMMIT']): raise ValueError('source commit')
    return g,recipe


def worker(g,recipe,seed,out,grant_sha):
    import signal
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124)); signal.alarm(g['wall_seconds_per_seed']+15)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_GPU_LOCK_FD']))
    import numpy as np
    import torch
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason import train
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise ValueError('one CUDA device')
    torch.cuda.set_per_process_memory_fraction(g['vram_bytes']/torch.cuda.get_device_properties(0).total_memory)
    inputs=(ROOT/g['inputs']).resolve()
    if not inputs.is_relative_to(ROOT): raise ValueError('input path escapes snapshot')
    if sha(inputs/'manifest.json')!=g['input_manifest_sha256']: raise ValueError('input manifest pin')
    man=checked_json(inputs/'manifest.json')
    if man['recipe_sha256']!=sha(ROOT/g['recipe']) or man['roles']!=g['expected_rows']:
        raise ValueError('package recipe/rows')
    names={'x-prequential.npy','corpus-subset.npz','seq-inputs.npz','fit-prior-support.npy','game-date.npy','encoder.json'}
    if set(man['files'])!=names or {p.name for p in inputs.iterdir()}!=names|{'manifest.json'}:
        raise ValueError('exact input files')
    for name,binding in man['files'].items():
        p=inputs/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=binding['bytes'] or sha(p)!=binding['sha256']:
            raise ValueError('input byte custody '+name)
    package=base.load_package(inputs,recipe['base']);_,c,_=package
    from experiments.ws_fullseason.data import key_digest
    for role,member in [('FIT','fit'),('OOF','evaluation')]:
        if key_digest(c['keys'][c[member]])!=g['expected_key_sha256'][role]: raise ValueError('role keys')
    result=train.run(recipe,seed,package,out)
    result.update(grant_sha256=grant_sha,recipe_sha256=sha(ROOT/g['recipe']),input_manifest_sha256=g['input_manifest_sha256'],
        data_role='2026 EXPOSED DEVELOPMENT',independent_test=False,source_commit=g['commit'],
        outputs_sha256={name:sha(out/name) for name in (['model-state.pt'] if recipe.get('execution_stage')=='FINAL_REFIT' else ['model-state.pt','oof-predictions.npz'])},
        gpu_max_reserved_bytes=torch.cuda.max_memory_reserved())
    if result['gpu_max_reserved_bytes']>g['vram_bytes']: raise ValueError('VRAM cap')
    (out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')



def ensemble_metrics(predictions, labels, expected_keys):
    """Registered three-seed arithmetic mean, after all fixed fits complete."""
    import numpy as np
    from experiments.ws_seq.evaluate_seq import row_loss,ece
    if len(predictions)!=3 or len(labels)!=len(expected_keys) or not len(labels):
        raise ValueError('three complete seeds and label denominator')
    if not np.issubdtype(labels.dtype,np.integer) or (labels<0).any() or (labels>6).any():
        raise ValueError('seven-leaf labels')
    qs=[]
    for keys,q in predictions:
        if not np.array_equal(keys,expected_keys) or q.shape!=(len(labels),7):
            raise ValueError('exact ordered OOF identity')
        if not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-7):
            raise ValueError('normalized finite probabilities')
        qs.append(q.astype(np.float64,copy=False))
    q=np.mean(np.stack(qs),axis=0)
    ll,bs=row_loss(q,labels)
    return {'rows':len(labels),'seeds':[0,1,2],'aggregation':'arithmetic mean raw probabilities before metrics',
        'nll':float(ll.mean()),'brier':float(bs.mean()),'ece_top_label_10bin':ece(q,labels),
        'nll_clip':1e-15,'calibration_fits':0,'independent_test':False}


def final_aggregate(g,out,records,grant_sha):
    import numpy as np
    from experiments.ws_fullseason.data import key_digest
    if [r['seed'] for r in records]!=[0,1,2]:raise ValueError('all three seeds required before scoring')
    for r in records:
        d=out/f"seed{r['seed']}"
        if sha(d/'receipt.json')!=r['receipt_sha256'] or sha(d/'SUPERVISOR.json')!=r['supervisor_sha256']:
            raise ValueError('completed receipt pins')
        receipt=checked_json(d/'receipt.json');supervisor=checked_json(d/'SUPERVISOR.json')
        if (receipt['status']!='COMPLETED' or supervisor['status']!='COMPLETED' or receipt['seed']!=r['seed']
                or supervisor.get('child_reaped') is not True or supervisor.get('returncode')!=0
                or supervisor.get('receipt_sha256')!=r['receipt_sha256']
                or supervisor.get('grant_sha256')!=grant_sha
                or receipt.get('grant_sha256')!=grant_sha
                or receipt.get('recipe_sha256')!=sha(ROOT/g['recipe'])
                or receipt.get('input_manifest_sha256')!=g['input_manifest_sha256']
                or receipt.get('source_commit')!=g['commit']):
            raise ValueError('completed seed identity')
        if sha(d/'oof-predictions.npz')!=receipt['outputs_sha256']['oof-predictions.npz']:
            raise ValueError('prediction byte custody')
    inputs=ROOT/g['inputs']
    if sha(inputs/'manifest.json')!=g['input_manifest_sha256']:raise ValueError('aggregate manifest custody')
    manifest=checked_json(inputs/'manifest.json')
    corpus=inputs/'corpus-subset.npz'
    if sha(corpus)!=manifest['files']['corpus-subset.npz']['sha256']:raise ValueError('label custody')
    with np.load(corpus,allow_pickle=False) as c:
        keep=c['evaluation'];keys=c['keys'][keep];labels=c['y7'][keep]
    if len(keys)!=g['expected_rows']['OOF'] or key_digest(keys)!=g['expected_key_sha256']['OOF']:
        raise ValueError('aggregate registered denominator and keys')
    predictions=[]
    for seed in [0,1,2]:
        with np.load(out/f'seed{seed}'/'oof-predictions.npz',allow_pickle=False) as z:
            predictions.append((z['keys'],z['probabilities']))
    result=ensemble_metrics(predictions,labels,keys)
    result.update(OOF_keys_sha256=key_digest(keys),recipe_sha256=sha(ROOT/g['recipe']),data_role='2026 EXPOSED DEVELOPMENT')
    body=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if len(body.encode())>g['aggregate_output_bytes']:raise ValueError('aggregate output cap')
    (out/'aggregate.json').write_text(body)
    return sha(out/'aggregate.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--grant',type=Path,required=True);p.add_argument('--seed',type=int)
    args=p.parse_args();g,recipe=checked_grant(args.grant);out=Path(g['output'])
    if not out.is_absolute(): raise ValueError('absolute output')
    if args.seed is not None:
        if args.seed not in g['seeds']: raise ValueError('seed')
        worker(g,recipe,args.seed,out/f'seed{args.seed}',sha(args.grant));return 0
    if out.exists(): raise ValueError('fresh output')
    uuid=subprocess.check_output(['nvidia-smi','-i',str(g['gpu_index']),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    if uuid!=g['gpu_uuid']: raise ValueError('GPU UUID')
    with Path(g['common_lock']).open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
        if uuid in processes.splitlines(): raise ValueError('GPU occupied')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if g.get('host_memory_headroom_bytes')!=32<<30:raise ValueError('two-job host headroom floor')
        if available<g['host_memory_headroom_bytes']: raise ValueError('host memory headroom')
        existing=out.parent
        while not existing.exists():existing=existing.parent
        if shutil.disk_usage(existing).free<4<<30:raise ValueError('server disk headroom')
        out.mkdir(parents=True)
        os.environ['PITCHEEZY_FULLSEASON_GPU_LOCK_FD']=str(lock.fileno())
        records=[]
        for seed in g['seeds']:
            target=out/f'seed{seed}';target.mkdir()
            rc=supervise([sys.executable,'-m','experiments.ws_fullseason.run_gpu','--grant',str(args.grant),'--seed',str(seed)],
                target,g['wall_seconds_per_seed'],g['rss_bytes'],sha(args.grant),lock.fileno(),g['output_bytes_per_seed'])
            if rc:return rc
            records.append({'seed':seed,'receipt_sha256':sha(target/'receipt.json'),'supervisor_sha256':sha(target/'SUPERVISOR.json')})
        final_refit=recipe.get('execution_stage')=='FINAL_REFIT'
        aggregate_sha=None
        if not final_refit:
            import signal
            signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.alarm(g['aggregate_wall_seconds'])
            try:aggregate_sha=final_aggregate(g,out,records,sha(args.grant))
            finally:signal.alarm(0)
        (out/'receipt.json').write_text(json.dumps({'status':'COMPLETED','seeds':records,'fits':3,'calibration_fits':0,
            'execution_stage':'FINAL_REFIT' if final_refit else 'OOF',
            'metrics_computed':not final_refit,'aggregate_sha256':aggregate_sha,'grant_sha256':sha(args.grant)},indent=2)+'\n')
    return 0


if __name__=='__main__':sys.exit(main())
