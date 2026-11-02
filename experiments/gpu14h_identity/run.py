"""원003 대비 FIT-only pitcher logit offset의 증분가치.

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
from experiments.gpu14h_prepitch import run as parent
from experiments.gpu14h_identity import train

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
def select_variants(rows):
    return inherited.select_variants(rows)[:1]


def selected_screen(record, cfg, config_sha):
    if record['config_sha256'] != config_sha: raise ValueError('same study screen config')
    r = completed_phase(record, cfg, 'score_screen')
    selected = r['selected_variants']
    if len(selected) > 1 or len(set(selected)) != len(selected) or any(v not in train.VARIANTS for v in selected): raise ValueError('finite selected variants')
    if selected != select_variants(r['candidates']): raise ValueError('frozen selection replay')
    return selected


def gates(config_path, grant_path, grant_sha, task):
    cfg = document(config_path)
    if cfg.get('status') != 'REGISTERED' or cfg.get('execute') is not True: raise ValueError('DRAFT: real I/O disabled')
    if (cfg['id'] != 'EXP-GPU14H-005' or cfg['variants'] != train.VARIANTS or cfg['fit_ceiling'] != 21
            or cfg['data_role'] != '2026 EXPOSED DEVELOPMENT' or cfg['seeds'] != [0, 1, 2]
            or cfg['deadline_utc'] != DEADLINE or cfg['recovery_grace_seconds'] != 900
            or cfg['minimum_fit_seconds'] != 900): raise ValueError('fixed study contract')
    expected = {(v,f,s,k) for v in train.VARIANTS for f in FOLDS for s in (0,1,2) for k in ('id','control') if k=='id' or s!=0}
    if {(j['variant'],j['fold'],j['seed'],j['kind']) for j in cfg['jobs']} != expected or len(cfg['jobs']) != 45: raise ValueError('finite45 potential tasks, max21 fits')
    if cfg['reference'] != REFERENCE: raise ValueError('frozen003 comparison')
    for j in cfg['jobs']:
        if j['id'] != task_id(j['kind'],j['variant'],j['fold'],j['seed']): raise ValueError('unique task identity')
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
        if job['kind']=='control': completion_gate(job,reference_screen(g['prepitch_screen'],cfg)['selected_variants'],g['claim_root'])
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


def fit_worker(cfg,g,meta,out):
    job=next(j for j in cfg['jobs'] if j['id']==g['task']);original=producer(cfg,job['fold'])
    rcp=train.recipe(original,job['variant']) if job['kind']=='id' else train.parent.recipe(original,train.ARCH[job['variant']])
    inputs,spec=custody.package_binding(meta,job['fold'],g)
    import torch
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason.data import key_digest
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('one CUDA device')
    torch.cuda.set_per_process_memory_fraction(cfg['limits']['vram_bytes']/torch.cuda.get_device_properties(0).total_memory)
    package=base.load_package(inputs,original['base']);_,corpus,_=package
    for role,field in (('FIT','fit'),('OOF','evaluation')):
        if int(corpus[field].sum())!=spec['roles'][role] or key_digest(corpus['keys'][corpus[field]])!=spec['expected']['key_sha256'][role]:raise ValueError('fixed role keys')
    fn=train.run if job['kind']=='id' else train.run_control
    r=fn(original,job['variant'],job['seed'],package,out)
    r.update(variant=job['variant'],architecture=train.ARCH[job['variant']],kind=job['kind'],fold=job['fold'],model_recipe_sha256=object_sha(rcp),producer_recipe_sha256=spec['producer_recipe']['recorded_sha256'],input_manifest_sha256=spec['manifest']['recorded_sha256'],gpu_index=g['gpu_index'],gpu_uuid=g['gpu_uuid'],gpu_max_reserved_bytes=torch.cuda.max_memory_reserved())
    if r['gpu_max_reserved_bytes']>cfg['limits']['vram_bytes']:raise ValueError('VRAM cap')
    custody.package_binding(meta,job['fold'],g)
    return r


REFERENCE={'commit':'37a84c53f678fcf8d402b42fd9958f06ef33de8f','config_sha256':'90e0cc453e1c10320949bd0bb489be7d4784f4ecdd1caa134c1142547cf4f7a1','source_manifest_sha256':'b2081354d15954624419fbbd6c97325bc73f573e136025744cac72565743bbb4','input_metadata_sha256':'5b64a6ccc41e59048f22e9d5ae0dbe68c6e81da6d06d77006ccefa9817994a3d'}


def task_id(kind,variant,fold,seed):return f'H14-B5-{kind}-{variant}-{fold}-s{seed}'


def reference_config(cfg):
    return dict(cfg,_config_sha256=REFERENCE['config_sha256'],source_manifest={'sha256':REFERENCE['source_manifest_sha256']})


def reference_screen(record,cfg):
    old=reference_config(cfg)
    parent.selected_screen(record,old,REFERENCE['config_sha256'])
    r=completed_phase(record,old,'score_screen')
    if r['seeds']!=[0] or r['selection_applied'] is not True or r['count_fits']!=3:raise ValueError('accepted003 screen contract')
    return r


def control_origin(variant,seed,selected003):
    return '003' if seed==0 or train.ARCH[variant] in selected003 else '005'


def completion_gate(job,selected003,claim_root):
    if job['kind']!='control' or job['seed'] not in (1,2) or control_origin(job['variant'],job['seed'],selected003)!='005':raise ValueError('003 selected/planned controls never duplicated')
    old=Path(claim_root)/f"H14-B3-{train.ARCH[job['variant']]}-{job['fold']}-s{job['seed']}.json"
    if old.exists() or old.is_symlink():raise ValueError('existing003 claim: no copy or retry')
    return str(old)


def prediction(record,cfg,meta,fold,seed,p,variant,kind):
    import numpy as np
    from experiments.ws_fullseason.calibrate_worker import pinned,header_check,file_digest
    if record['config_sha256']!=cfg['_config_sha256']:raise ValueError('same005 registration')
    r=completed_phase(record,cfg,task_id(kind,variant,fold,seed));spec=meta['training']['packages'][fold];pb=binding(record['predictions']);n=spec['expected']['OOF']
    recipe=train.recipe(producer(cfg,fold),variant) if kind=='id' else train.parent.recipe(producer(cfg,fold),train.ARCH[variant])
    if (r['arm']!=('PREPITCH9_ID' if kind=='id' else 'PREPITCH9') or r['variant']!=variant or r['kind']!=kind or r['architecture']!=train.ARCH[variant] or r['seed']!=seed or r['fold']!=fold or r['epochs']!=30
            or r['model_recipe_sha256']!=object_sha(recipe) or r['input_manifest_sha256']!=spec['manifest']['recorded_sha256'] or r['calibration_fits']!=0 or r['SELECT_queries']!=0
            or r['temperature']!=1 or r['checkpoint']!='final epoch' or r['OOF_rows']!=n or r['OOF_keys_sha256']!=spec['expected']['key_sha256']['OOF']
            or r['FIT_E9_keys_sha256']!=p['fit9_sha256'] or r['census']!=p['census'] or r['outputs_sha256']['oof-predictions.npz']!=pb['sha256']):raise ValueError('005 prediction receipt contract')
    if len(r['consumption'])!=30 or any(t['epoch']!=i or t['rows']!=int(p['fit9'].sum()) or t['ordered_keys_sha256']!=p['fit9_sha256'] or t['every_key_once'] is not True or t['positive_type_weight_every_row'] is not True for i,t in enumerate(r['consumption'],1)):raise ValueError('mapped FIT consumption')
    if kind=='id':
        ids=train.vocabulary(p['seq']['pitcher'],p['fit9']);vocab={'ids':ids.tolist(),'FIT_E9_keys_sha256':p['fit9_sha256'],'unknown_index':0,'embedding_dim':16}
        unknown=int((train.indices(p['seq']['pitcher'][p['oof']],ids)==0).sum())
        if r['vocabulary_sha256']!=object_sha(vocab) or r['vocabulary_size']!=len(ids) or r['unknown_vector_zero'] is not True or r['OOF_unknown_pitchers']!=unknown:raise ValueError('FIT-only ID vocabulary receipt')
    with ExitStack() as stack:
        stream,_=pinned(pb,stack);header_check(stream,{'keys','probabilities'},{'keys':((n,3),'iu'),'probabilities':((n,9),'f')})
        with np.load(stream,allow_pickle=False) as z:k,q=z['keys'],z['probabilities']
        if not np.array_equal(k,p['keys'][p['oof']]) or file_digest(stream)!=pb['sha256']:raise ValueError('allOOF Q_pkg keys')
    return parent.probabilities(q)


def partition(records,missing,expected):
    by={(r['variant'],r['fold'],r['seed']):r for r in records};absent={(r['variant'],r['fold'],r['seed']) for r in missing}
    if len(by)!=len(records) or len(absent)!=len(missing) or set(by)&absent or set(by)|absent!=expected:raise ValueError('complete or explicit unavailable partition')
    if any(r['status'] not in ('FAILED','NOT_RUN','INCOMPLETE') for r in missing):raise ValueError('unavailable status')
    return by


def score_worker(cfg,g,meta,out):
    import numpy as np
    screen=g['task']=='score_screen';variants=list(train.VARIANTS) if screen else selected_screen(g['screen'],cfg,cfg['_config_sha256']);seeds=[0] if screen else [0,1,2]
    expected={(v,f,s) for v in variants for f in FOLDS for s in seeds}
    by=partition(g['results'],g['unavailable_tasks'],expected);controls=partition(g['controls'],g['control_unavailable'],expected)
    prior=reference_screen(g['prepitch_screen'],cfg);old=reference_config(cfg)
    for key,record in controls.items():
        v,f,s=key
        if record['origin']!=control_origin(v,s,prior['selected_variants']):raise ValueError('no-ID control provenance/duplication gate')
    b=binding(g['baseline'])
    if prior['outputs_sha256']['baseline-counts.json']!=b['sha256']:raise ValueError('frozen003 count provenance')
    tables=json.loads(verify_file(b,16<<20).read_text());pops={f:parent.population(meta,f) for f in FOLDS};keep={};ys={};frequency={};fallback={};known={}
    for f,p in pops.items():
        keep[f]=p['y'][p['oof']]>=0;ys[f]=p['y'][p['oof']][keep[f]]
        q,fallback[f]=parent.frequency(tables[f],p);frequency[f]=q[keep[f]]
        ids=train.vocabulary(p['seq']['pitcher'],p['fit9']);known[f]=train.indices(p['seq']['pitcher'][p['oof']],ids)[keep[f]]>0
        if not len(ys[f]):raise ValueError('emptyE9')
    y=np.concatenate([ys[f] for f in FOLDS]);freq=np.concatenate([frequency[f] for f in FOLDS]);known_all=np.concatenate([known[f] for f in FOLDS]);rows=[]
    for v in variants:
        if not all((v,f,s) in by and (v,f,s) in controls for f in FOLDS for s in seeds):
            rows.append({'variant':v,'status':'INCOMPLETE_NOT_SELECTED','pooled':None});continue
        own={};base={}
        for f in FOLDS:
            own[f]={};base[f]={}
            for seed in seeds:
                own[f][seed]=prediction(by[v,f,seed],cfg,meta,f,seed,pops[f],v,'id')[keep[f]]
                record=controls[v,f,seed]
                q=parent.prediction(record,old,meta,f,seed,pops[f],train.ARCH[v]) if record['origin']=='003' else prediction(record,cfg,meta,f,seed,pops[f],v,'control')
                base[f][seed]=q[keep[f]]
        means={f:np.mean(list(own[f].values()),axis=0,dtype=np.float64) for f in FOLDS};bases={f:np.mean(list(base[f].values()),axis=0,dtype=np.float64) for f in FOLDS}
        q=np.concatenate([means[f] for f in FOLDS]);bc=np.concatenate([bases[f] for f in FOLDS])
        sub={}
        for name,mask in [('known_pitcher',known_all),('cold_start_pitcher',~known_all)]:
            sub[name]={'rows':int(mask.sum()),'metrics':parent.comparison9(q[mask],bc[mask],y[mask]) if mask.any() else None}
        rows.append({'variant':v,'status':'COMPLETE','pooled':parent.comparison9(q,bc,y),'monthly':{f:parent.comparison9(means[f],bases[f],ys[f]) for f in FOLDS},'per_seed':{str(s):parent.comparison9(np.concatenate([own[f][s] for f in FOLDS]),np.concatenate([base[f][s] for f in FOLDS]),y) for s in seeds},'known_cold_start':sub,'classwise':{name:{'rows':int((y==i).sum()),'metrics':parent.comparison9(q[y==i],bc[y==i],y[y==i]) if (y==i).any() else None} for i,name in enumerate(train.parent.CLASSES)},'vs_frozen_frequency':parent.comparison9(q,freq,y),'prediction_status':{f:{'OK_Q_pkg':pops[f]['census']['OOF']['Q_pkg'],'OK_E9_pkg':len(ys[f]),'ABSTAIN':0,'ERROR':0} for f in FOLDS}})
    return {'candidates':rows,'selected_variants':select_variants(rows) if screen else variants,'selection_applied':screen,'seeds':seeds,'baseline':'same003 architecture/fold/seed, paired probability mean; missing controls exact003 recipe under005 only','count_fits':0,'census':{f:pops[f]['census'] for f in FOLDS},'fallback':fallback,'FIT_group_counts':{f:dict(zip(train.parent.CLASSES,tables[f]['global'])) for f in FOLDS},'FIT_rare_groups_lt100':{f:[name for name,count in zip(train.parent.CLASSES,tables[f]['global']) if count<100] for f in FOLDS},'reference003_selected':prior['selected_variants'],'control_origins':[{'variant':v,'fold':f,'seed':s,'origin':r['origin']} for (v,f,s),r in controls.items()],'unavailable_tasks':g['unavailable_tasks'],'control_unavailable':g['control_unavailable'],'new_backbone_fits_in_phase':0,'calibration_fits':0,'FINAL_evaluations':0,'independent_test':False,'promotion':False,'population_limit':'historical outcome-eligible Q_pkg; mappedE9 conditional target','availability':'UNKNOWN_final_snapshot','selection_bias':'multiple seed0 variants on exposedOOF; stability is not independent','identity_interpretation':'additive low-rank pitcher logit offset; no direct state interaction'}


def worker(cfg, g, meta, out, config_sha, grant_sha):
    fd = int(os.environ['PITCHEEZY_H14B5_LOCK_FD']); held = os.fstat(fd); lock = Path(g['lock']).stat()
    with Path(g['lock']).open('rb') as probe:
        try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise ValueError('inherited lock absent')
    if (held.st_dev, held.st_ino) != (lock.st_dev, lock.st_ino): raise ValueError('exact inherited lock')
    custody.fresh_worker_output(out)
    cfg['_config_sha256'] = config_sha
    result = fit_worker(cfg, g, meta, out) if g['task'].startswith('H14-B5-') else score_worker(cfg, g, meta, out)
    sources(cfg, g['source_manifest_path'])
    result.update(status='COMPLETED', task=g['task'], source_commit=g['source_commit'], source_manifest_sha256=cfg['source_manifest']['sha256'], input_metadata_sha256=cfg['input_metadata']['sha256'], config_sha256=config_sha, grant_sha256=grant_sha, data_role=cfg['data_role'], independent_test=False, runtime={n: getattr(sys.modules.get(n), '__version__', None) for n in ('numpy','scipy','pandas','pyarrow','torch')} | {'python': sys.version}, outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('worker.log','receipt.json')})
    write_json(out / 'receipt.json', result)


def supervise(cfg, g, args):
    fit = g['task'].startswith('H14-B5-'); out = Path(g['output']); root = Path(g['output_root']).resolve()
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
        env = dict(os.environ, PITCHEEZY_H14B5_LOCK_FD=str(lock.fileno()), TMPDIR=str(out / 'tmp'))
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
