"""현재 구종9종 연구. 원 outcome-eligible package와 순수pre-pitch 입력을 분리한다.

Pure custody/resource primitives are reused from the accepted C runner. Its C
study gates are not called or modified. Real I/O starts after this study's gate.
"""
import argparse
import json
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason_c import run as custody
from experiments.gpu14h import run as inherited
from experiments.gpu14h_prepitch import train

sha = custody.sha
object_sha = custody.object_sha
document = custody.document
binding = custody.binding
verify_file = custody.verify_file
write_json = custody.write_json
FOLDS = ('july', 'august', 'september')
THREAD_ENV = custody.THREAD_ENV
DEADLINE = '2026-11-03T03:21:56Z'


sources = inherited.sources


effective_wall = inherited.effective_wall


producer = inherited.producer


completed_phase = inherited.completed_phase
select_variants = inherited.select_variants


def selected_screen(record, cfg, config_sha):
    if record['config_sha256'] != config_sha: raise ValueError('same study screen config')
    r = completed_phase(record, cfg, 'score_screen')
    selected = r['selected_variants']
    if len(selected) > 2 or len(set(selected)) != len(selected) or any(v not in train.VARIANTS for v in selected): raise ValueError('finite selected variants')
    if selected != select_variants(r['candidates']): raise ValueError('frozen selection replay')
    return selected


def gates(config_path, grant_path, grant_sha, task):
    cfg = document(config_path)
    if cfg.get('status') != 'REGISTERED' or cfg.get('execute') is not True: raise ValueError('DRAFT: real I/O disabled')
    if (cfg['id'] != 'EXP-GPU14H-003' or cfg['variants'] != train.VARIANTS or cfg['fit_ceiling'] != 21
            or cfg['data_role'] != '2026 EXPOSED DEVELOPMENT' or cfg['seeds'] != [0, 1, 2]
            or cfg['deadline_utc'] != DEADLINE or cfg['recovery_grace_seconds'] != 900
            or cfg['minimum_fit_seconds'] != 900): raise ValueError('fixed study contract')
    expected = {(v, f, s) for v in train.VARIANTS for f in FOLDS for s in (0, 1, 2)}
    if {(j['variant'], j['fold'], j['seed']) for j in cfg['jobs']} != expected or len(cfg['jobs']) != 27: raise ValueError('finite potential task identities')
    for j in cfg['jobs']:
        if j['id'] != f"H14-B3-{j['variant']}-{j['fold']}-s{j['seed']}": raise ValueError('unique task identity')
    if sha(grant_path) != grant_sha: raise ValueError('exact reviewed grant')
    raw = document(grant_path)
    if (raw.get('execute') is not True or raw.get('verdict') != 'PASS' or raw.get('authorized_by') != 'root'
            or raw.get('author') != cfg['author'] or not raw.get('reviewer') or raw['reviewer'] == raw['author']
            or raw.get('config_sha256') != sha(config_path) or task not in raw['tasks']): raise ValueError('independent phase grant')
    g = dict(raw, **raw['tasks'][task], task=task)
    for key in ('output', 'output_root'):
        value = g[key]
        if '${GPU_JOB_DIR}' in value:
            if not os.environ.get('GPU_JOB_DIR'): raise ValueError('gpu run missing')
            value = value.replace('${GPU_JOB_DIR}', os.environ['GPU_JOB_DIR'])
        if '$' in value or '~' in value: raise ValueError('unregistered expansion')
        g[key] = value
    sources(cfg, g['source_manifest_path'])
    if sha(g['input_metadata_path']) != cfg['input_metadata']['sha256']: raise ValueError('input metadata pin')
    if any(os.environ.get(k) != '1' for k in THREAD_ENV) or os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'): raise ValueError('fixed import/thread environment')
    jobs = {j['id']: j for j in cfg['jobs']}; fit = task in jobs
    if fit:
        job = jobs[task]
        if not job['enabled']: raise ValueError('disabled fit')
        train.recipe(producer(cfg, job['fold']), job['variant'])
        if job['seed'] != 0 and job['variant'] not in selected_screen(g['screen'], cfg, sha(config_path)): raise ValueError('nonselected stability fit')
        if g['min_free_mib'] != 20480 or g['free_memory_bytes'] != 72 << 30: raise ValueError('GPU host admission')
        devices = g['gpu_candidates']
        if len(devices) != 5 or len({v['uuid'] for v in devices}) != 5 or {v['index'] for v in devices} != set(range(5)): raise ValueError('five candidate GPUs')
        chosen = [v for v in devices if str(v['index']) == os.environ.get('CUDA_VISIBLE_DEVICES')]
        if len(chosen) != 1: raise ValueError('gpu run one numeric device')
        g.update(gpu_index=chosen[0]['index'], gpu_uuid=chosen[0]['uuid'], lock=str(Path(g['gpu_lock_root']) / (chosen[0]['uuid'] + '.lock')))
    elif task not in cfg['phases'] or cfg['phases'][task] is not True: raise ValueError('disabled phase')
    limits = cfg['limits']
    if (g['rss_bytes'] != limits['rss_bytes'] or g['wall_seconds'] != limits['fit_wall_seconds' if fit else 'cpu_wall_seconds']
            or g['output_bytes'] != limits['output_bytes_per_fit' if fit else 'cpu_output_bytes']
            or g['free_disk_bytes'] != cfg['resource_admission']['remote_disk_free_bytes' if fit else 'local_T7_free_bytes']): raise ValueError('fixed resource bounds')
    if fit: effective_wall(cfg)
    return cfg, g, document(g['input_metadata_path'])


def fit_worker(cfg, g, meta, out):
    job = next(j for j in cfg['jobs'] if j['id'] == g['task'])
    original = producer(cfg, job['fold']); recipe = train.recipe(original, job['variant'])
    inputs, spec = custody.package_binding(meta, job['fold'], g)
    import torch
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason.data import key_digest
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise ValueError('one CUDA device')
    torch.cuda.set_per_process_memory_fraction(cfg['limits']['vram_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    package = base.load_package(inputs, original['base']); _, corpus, _ = package
    for role, field in (('FIT', 'fit'), ('OOF', 'evaluation')):
        if int(corpus[field].sum()) != spec['roles'][role] or key_digest(corpus['keys'][corpus[field]]) != spec['expected']['key_sha256'][role]: raise ValueError('fixed role keys')
    r = train.run(original, job['variant'], job['seed'], package, out)
    r.update(variant=job['variant'], fold=job['fold'], model_recipe_sha256=object_sha(recipe), producer_recipe_sha256=spec['producer_recipe']['recorded_sha256'], input_manifest_sha256=spec['manifest']['recorded_sha256'], gpu_index=g['gpu_index'], gpu_uuid=g['gpu_uuid'], gpu_max_reserved_bytes=torch.cuda.max_memory_reserved())
    if r['gpu_max_reserved_bytes'] > cfg['limits']['vram_bytes']: raise ValueError('VRAM cap')
    custody.package_binding(meta, job['fold'], g)
    return r


# 아래 계산은 별도 실자료 phase grant 뒤에만 호출된다.
def probabilities(q):
    import numpy as np
    q=np.asarray(q,np.float64)
    if q.ndim!=2 or q.shape[1]!=9 or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-10,rtol=0):raise ValueError('9종 확률 계약')
    return q


def population(meta,fold):
    import numpy as np
    from experiments.ws_fullseason.calibrate_worker import pinned,header_check,CORPUS,file_digest
    from experiments.ws_fullseason.data import key_digest
    from experiments.ws_seq.train_seq import SEQ_MEMBERS
    spec=meta['training']['packages'][fold];e=spec['expected'];n=e['rows']
    cb=next(binding(v) for v in spec['files'] if Path(v['path']).name=='corpus-subset.npz')
    sb=next(binding(v) for v in spec['files'] if Path(v['path']).name=='seq-inputs.npz')
    with ExitStack() as stack:
        stream,_=pinned(cb,stack);header_check(stream,CORPUS,{'keys':((n,3),'iu'),'fit':((n,),'b'),'evaluation':((n,),'b')})
        with np.load(stream,allow_pickle=False) as z:k,fit,oof=(z[v] for v in ('keys','fit','evaluation'))
        seq,_=pinned(sb,stack);fields=('game_pk','at_bat_number','pitch_number','pitcher','pitch_type','balls','strikes')
        header_check(seq,set(SEQ_MEMBERS),{v:((n,),'iu') for v in fields})
        with np.load(seq,allow_pickle=False) as z:s={v:z[v] for v in fields}
        if not np.array_equal(k,np.column_stack([s[v] for v in ('game_pk','at_bat_number','pitch_number')])):raise ValueError('sequence keys')
        if int(fit.sum())!=e['FIT'] or int(oof.sum())!=e['OOF'] or (fit&oof).any() or not (fit|oof).all():raise ValueError('원역할 분모')
        if any(key_digest(k[m])!=e['key_sha256'][role] for m,role in ((fit,'FIT'),(oof,'OOF'))):raise ValueError('원키')
        if s['balls'].min()<0 or s['balls'].max()>3 or s['strikes'].min()<0 or s['strikes'].max()>2:raise ValueError('count 범위')
        if file_digest(stream)!=cb['sha256'] or file_digest(seq)!=sb['sha256']:raise ValueError('원본 변경')
    y=train.targets(s['pitch_type']);f9=fit&(y>=0)
    census={role:{'Q_pkg':int(mask.sum()),'E9_pkg':int((mask&(y>=0)).sum()),'raw_type_counts':dict(zip(train.TYPES,np.bincount(s['pitch_type'][mask],minlength=19).astype(int).tolist()))} for role,mask in [('FIT',fit),('OOF',oof)]}
    return {'keys':k,'fit':fit,'oof':oof,'fit9':f9,'y':y,'seq':s,'census':census,'fit9_sha256':key_digest(k[f9])}


def comparison9(candidate,baseline,y):
    import numpy as np
    def metric(q):
        m=inherited.metrics(q,y)
        truth=q[np.arange(len(y)),y]
        m['nll']=float(-np.log(np.clip(truth,1e-12,1)).mean())
        m['zero_truth_probability']=int((truth==0).sum())
        m['top1']=float((q.argmax(1)==y).mean())
        m['top2']=float((np.argsort(-q,axis=1,kind='stable')[:,:2]==y[:,None]).any(1).mean())
        return m
    c,b=metric(candidate),metric(baseline)
    return {'candidate':c,'baseline':b,'delta':{k:c[k]-b[k] for k in ('nll','brier','ece10')}}


def fit_frequency(p):
    import numpy as np
    y=p['y'][p['fit9']];s=p['seq'];mask=p['fit9']
    if not len(y):raise ValueError('빈9종FIT')
    result={'alpha':4.,'classes':list(train.CLASSES),'rows':len(y),'fit9_keys_sha256':p['fit9_sha256'],'global':np.bincount(y,minlength=9).tolist(),'pitcher':{},'context':{}}
    for fields,key in [(('pitcher',),'pitcher'),(('pitcher','balls','strikes'),'context')]:
        group=np.column_stack([s[v][mask] for v in fields]);unique,inverse=np.unique(group,axis=0,return_inverse=True)
        counts=np.zeros((len(unique),9),np.int64);np.add.at(counts,(inverse,y),1)
        result[key]={':'.join(map(str,g)):row.tolist() for g,row in zip(unique,counts)}
    return result


def frequency(table,p):
    import numpy as np
    if table['alpha']!=4. or table['classes']!=list(train.CLASSES) or table['rows']!=int(p['fit9'].sum()) or table['fit9_keys_sha256']!=p['fit9_sha256']:raise ValueError('고정FIT빈도 표적')
    count=np.asarray(table['global'],np.float64)
    if count.shape!=(9,) or (count<0).any() or count.sum()!=table['rows']:raise ValueError('빈도 전역 분모')
    globalp=(count+4/9)/(count.sum()+4);s=p['seq'];oof=p['oof'];queries=np.column_stack([s[v][oof] for v in ('pitcher','balls','strikes')]);unique,inverse=np.unique(queries,axis=0,return_inverse=True)
    q=np.empty((len(unique),9));flags=[]
    for i,(pid,b,st) in enumerate(unique):
        pc=table['pitcher'].get(str(pid));cc=table['context'].get(f'{pid}:{b}:{st}')
        pp=globalp if pc is None else (np.asarray(pc)+4*globalp)/(sum(pc)+4)
        q[i]=pp if cc is None else (np.asarray(cc)+4*pp)/(sum(cc)+4)
        flags.append('unseen_pitcher_global' if pc is None else 'empty_context_pitcher' if cc is None else 'observed_context')
    allflags=np.asarray(flags)[inverse]
    e9=p['y'][oof]>=0
    counts={role:{name:int(((allflags==name)&mask).sum()) for name in ('unseen_pitcher_global','empty_context_pitcher','observed_context')} for role,mask in [('Q_pkg',np.ones(len(e9),bool)),('E9_pkg',e9)]}
    out_support=sum(1 for pid,label in zip(s['pitcher'][oof][e9],p['y'][oof][e9]) if table['pitcher'].get(str(pid),[0]*9)[label]==0)
    counts['E9_outside_FIT_pitcher_support']=out_support
    return probabilities(q[inverse]),counts


def prediction(record,cfg,meta,fold,seed,p,variant):
    import numpy as np
    from experiments.ws_fullseason.calibrate_worker import pinned,header_check,file_digest
    if record['config_sha256']!=cfg['_config_sha256']:raise ValueError('동일등록')
    r=completed_phase(record,cfg,f'H14-B3-{variant}-{fold}-s{seed}');spec=meta['training']['packages'][fold];pb=binding(record['predictions']);n=spec['expected']['OOF']
    if (r['arm']!='PREPITCH9' or r['variant']!=variant or r['seed']!=seed or r['fold']!=fold or r['epochs']!=30
            or r['model_recipe_sha256']!=object_sha(train.recipe(producer(cfg,fold),variant))
            or r['input_manifest_sha256']!=spec['manifest']['recorded_sha256'] or r['calibration_fits']!=0 or r['SELECT_queries']!=0
            or r['temperature']!=1 or r['checkpoint']!='final epoch' or r['OOF_rows']!=n or r['OOF_keys_sha256']!=spec['expected']['key_sha256']['OOF']
            or r['FIT_E9_keys_sha256']!=p['fit9_sha256'] or r['census']!=p['census'] or r['outputs_sha256']['oof-predictions.npz']!=pb['sha256']):raise ValueError('9종후보 receipt계약')
    if len(r['consumption'])!=30 or any(t['epoch']!=i or t['rows']!=int(p['fit9'].sum()) or t['ordered_keys_sha256']!=p['fit9_sha256'] or t['every_key_once'] is not True or t['positive_type_weight_every_row'] is not True for i,t in enumerate(r['consumption'],1)):raise ValueError('mapped FIT 소비')
    with ExitStack() as stack:
        stream,_=pinned(pb,stack);header_check(stream,{'keys','probabilities'},{'keys':((n,3),'iu'),'probabilities':((n,9),'f')})
        with np.load(stream,allow_pickle=False) as z:k,q=z['keys'],z['probabilities']
        if not np.array_equal(k,p['keys'][p['oof']]) or file_digest(stream)!=pb['sha256']:raise ValueError('전체OOF Q_pkg키')
    return probabilities(q)


def score_worker(cfg,g,meta,out):
    import numpy as np
    screen=g['task']=='score_screen';variants=list(train.VARIANTS) if screen else selected_screen(g['screen'],cfg,cfg['_config_sha256']);seeds=[0] if screen else [0,1,2]
    expected={(v,f,s) for v in variants for f in FOLDS for s in seeds};records=g['results'];missing=g['unavailable_tasks']
    by={(r['variant'],r['fold'],r['seed']):r for r in records};absent={(r['variant'],r['fold'],r['seed']) for r in missing}
    if len(by)!=len(records) or len(absent)!=len(missing) or set(by)&absent or set(by)|absent!=expected:raise ValueError('완료/미완 identity대사')
    if any(r['status'] not in ('FAILED','NOT_RUN','INCOMPLETE') for r in missing):raise ValueError('미완 상태')
    complete=[v for v in variants if all((v,f,s) in by for f in FOLDS for s in seeds)]
    pops={f:population(meta,f) for f in FOLDS}
    if screen:
        tables={f:fit_frequency(pops[f]) for f in FOLDS};write_json(out/'baseline-counts.json',tables)
    else:
        receipt=completed_phase(g['screen'],cfg,'score_screen');b=g['baseline']
        if receipt['outputs_sha256']['baseline-counts.json']!=b['sha256']:raise ValueError('동결빈도 선행pin')
        tables=json.loads(verify_file(b,16<<20).read_text())
    baseline={};fallback={};keep={};ys={}
    for f,p in pops.items():
        q,fallback[f]=frequency(tables[f],p);keep[f]=p['y'][p['oof']]>=0;ys[f]=p['y'][p['oof']][keep[f]];baseline[f]=q[keep[f]]
        if not len(ys[f]):raise ValueError('빈 E9_pkg')
    all_y=np.concatenate([ys[f] for f in FOLDS]);all_base=np.concatenate([baseline[f] for f in FOLDS]);rows=[]
    for v in variants:
        if v not in complete:rows.append({'variant':v,'status':'INCOMPLETE_NOT_SELECTED','pooled':None,'unavailable':[r for r in missing if r['variant']==v]});continue
        own={f:{s:prediction(by[v,f,s],cfg,meta,f,s,pops[f],v)[keep[f]] for s in seeds} for f in FOLDS}
        monthly={f:comparison9(np.mean(list(own[f].values()),axis=0,dtype=np.float64),baseline[f],ys[f]) for f in FOLDS}
        pool=np.concatenate([np.mean(list(own[f].values()),axis=0,dtype=np.float64) for f in FOLDS])
        per_seed={str(s):comparison9(np.concatenate([own[f][s] for f in FOLDS]),all_base,all_y) for s in seeds}
        classwise={name:{'rows':int((all_y==i).sum()),'metrics':comparison9(pool[all_y==i],all_base[all_y==i],all_y[all_y==i]) if (all_y==i).any() else None} for i,name in enumerate(train.CLASSES)}
        rows.append({'variant':v,'status':'COMPLETE','classwise':classwise,'pooled':comparison9(pool,all_base,all_y),'monthly':monthly,'per_seed':per_seed,'prediction_status':{f:{'OK_Q_pkg':pops[f]['census']['OOF']['Q_pkg'],'OK_E9_pkg':len(ys[f]),'ABSTAIN':0,'ERROR':0} for f in FOLDS}})
    return {'candidates':rows,'selected_variants':select_variants(rows) if screen else variants,'selection_applied':screen,'seeds':seeds,'baseline':'FIT alpha4 global -> pitcher -> pitcher/count; frozen on OOF','count_fits':3 if screen else 0,'census':{f:pops[f]['census'] for f in FOLDS},'fallback':fallback,'FIT_group_counts':{f:dict(zip(train.CLASSES,tables[f]['global'])) for f in FOLDS},'FIT_rare_groups_lt100':{f:[name for name,count in zip(train.CLASSES,tables[f]['global']) if count<100] for f in FOLDS},'new_backbone_fits_in_phase':0,'calibration_fits':0,'FINAL_evaluations':0,'unavailable_tasks':missing,'independent_test':False,'promotion':False,'population_limit':'Q_pkg already conditioned on historical 7leaf outcome eligibility; not all pitches; mapped9 conditional target','availability':'UNKNOWN_final_snapshot','selection_bias':'seed0 and multiple candidates on exposed OOF'}


def worker(cfg, g, meta, out, config_sha, grant_sha):
    fd = int(os.environ['PITCHEEZY_H14B3_LOCK_FD']); held = os.fstat(fd); lock = Path(g['lock']).stat()
    with Path(g['lock']).open('rb') as probe:
        try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise ValueError('inherited lock absent')
    if (held.st_dev, held.st_ino) != (lock.st_dev, lock.st_ino): raise ValueError('exact inherited lock')
    custody.fresh_worker_output(out)
    cfg['_config_sha256'] = config_sha
    result = fit_worker(cfg, g, meta, out) if g['task'].startswith('H14-B3-') else score_worker(cfg, g, meta, out)
    sources(cfg, g['source_manifest_path'])
    result.update(status='COMPLETED', task=g['task'], source_commit=g['source_commit'], source_manifest_sha256=cfg['source_manifest']['sha256'], input_metadata_sha256=cfg['input_metadata']['sha256'], config_sha256=config_sha, grant_sha256=grant_sha, data_role=cfg['data_role'], independent_test=False, runtime={n: getattr(sys.modules.get(n), '__version__', None) for n in ('numpy','scipy','pandas','pyarrow','torch')} | {'python': sys.version}, outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('worker.log','receipt.json')})
    write_json(out / 'receipt.json', result)


def supervise(cfg, g, args):
    fit = g['task'].startswith('H14-B3-'); out = Path(g['output']); root = Path(g['output_root']).resolve()
    if out.exists() or out.is_symlink() or not out.resolve().is_relative_to(root) or not root.is_dir(): raise ValueError('fresh registered output under existing root')
    if g['device_mount'] is not None:
        if not os.path.ismount(g['device_mount']): raise ValueError('registered output mount absent')
        info = plistlib.loads(subprocess.check_output(['diskutil','info','-plist',g['device_mount']]))
        if info.get('VolumeUUID') != g['volume_uuid']: raise ValueError('registered volume UUID')
    output_device = root.stat().st_dev
    custody.headroom(g)
    if shutil.disk_usage(root).free < g['free_disk_bytes']: raise ValueError('output filesystem headroom')
    if fit and not out.resolve().is_relative_to(Path(os.environ['GPU_JOB_DIR']).resolve() / 'code'): raise ValueError('pullable GPU job output')
    if not fit and os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only phase')
    with Path(g['lock']).open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if fit: custody.gpu_check(g)
        if fit:
            claim = Path(g['task_claim'])
            if claim.name != g['task'] + '.json' or claim.parent.resolve() != Path(g['claim_root']).resolve(): raise ValueError('exact shared task claim')
            write_json(claim, {'task': g['task'], 'grant_sha256': args.grant_sha256, 'output': str(out), 'state': 'CLAIMED_NO_AUTO_RETRY', 'gpu_index': g['gpu_index'], 'gpu_uuid': g['gpu_uuid']})
        out.mkdir(parents=True, mode=0o700)
        env = dict(os.environ, PITCHEEZY_H14B3_LOCK_FD=str(lock.fileno()), TMPDIR=str(out / 'tmp'))
        (out / 'tmp').mkdir()
        cmd = [sys.executable, '-B', str(Path(__file__)), '--config', str(args.config), '--grant', str(args.grant), '--grant-sha256', args.grant_sha256, '--task', args.task, '--worker']
        absolute_end = datetime.fromisoformat(cfg['deadline_utc'].replace('Z', '+00:00')).timestamp() - (900 if fit else 60)
        wall = effective_wall(cfg) if fit else min(g['wall_seconds'], absolute_end - time.time())
        if wall <= 0: raise ValueError('absolute phase deadline')
        start = time.monotonic(); peak = 0; reason = None
        with (out / 'worker.log').open('x') as log:
            child = subprocess.Popen(cmd, env=env, pass_fds=(lock.fileno(),), stdout=log, stderr=log)
            try:
                while child.poll() is None:
                    rss = subprocess.run(['ps','-o','rss=','-p',str(child.pid)],text=True,capture_output=True)
                    if rss.returncode == 0 and rss.stdout.strip(): peak = max(peak, int(rss.stdout.strip()) * 1024)
                    elif child.poll() is None: reason = 'unobservable-owned-child-rss'; break
                    try: custody.headroom(g)
                    except (ValueError, OSError, subprocess.CalledProcessError): reason = 'host-memory-headroom'; break
                    if peak > g['rss_bytes']: reason = 'rss'; break
                    if time.time() >= absolute_end: reason = 'absolute-deadline'; break
                    if time.monotonic() - start > wall: reason = 'wall'; break
                    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > g['output_bytes']: reason = 'output'; break
                    if root.stat().st_dev != output_device or (g['device_mount'] is not None and not os.path.ismount(g['device_mount'])): reason = 'mount-lost'; break
                    if shutil.disk_usage(root).free < g['free_disk_bytes']: reason = 'disk'; break
                    time.sleep(.25)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > g['output_bytes']: reason = 'output'
        completed = child.returncode == 0 and reason is None and (out / 'receipt.json').is_file() and document(out / 'receipt.json').get('status') == 'COMPLETED'
        write_json(out / 'SUPERVISOR.json', {'status': 'COMPLETED' if completed else 'STOP', 'returncode': child.returncode, 'reason': reason, 'child_reaped': True, 'child_pid': child.pid, 'peak_sampled_rss_bytes': peak, 'elapsed_seconds': time.monotonic() - start, 'effective_wall_seconds': wall, 'grant_sha256': args.grant_sha256, 'receipt_sha256': sha(out / 'receipt.json') if completed else None})
        return 0 if completed else 87


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--grant', type=Path, required=True); p.add_argument('--grant-sha256', required=True); p.add_argument('--task', required=True); p.add_argument('--worker', action='store_true'); a = p.parse_args()
    cfg, g, meta = gates(a.config, a.grant, a.grant_sha256, a.task)
    if a.worker: worker(cfg, g, meta, Path(g['output']), sha(a.config), a.grant_sha256); return 0
    return supervise(cfg, g, a)


if __name__ == '__main__': sys.exit(main())
