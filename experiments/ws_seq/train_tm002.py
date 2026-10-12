"""Finite single-arm TM-002 CUDA runner. No execution without reviewed grant.

Launch ONLY through /Users/song/.local/bin/gpu run after gpu st.
Imports base data/model helpers but does not change frozen SEQ-001 source.
An external supervisor stops only its newly spawned worker on budget breach.
No other process is signalled. A self-watchdog is an additional backstop.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from experiments.ws_seq import train_seq as base
from experiments.ws_seq import tm002

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('experiments/ws_seq/train_tm002.py', 'experiments/ws_seq/tm002.py',
           'experiments/ws_seq/train_seq.py', 'configs/EXP-WS-SEQ-001.json',
           'configs/EXP-WS-TM-002.json', 'experiments/ws_seq/TM002-REGISTRATION.md',
           'tests/test_tm002.py')


def source_hashes():
    return {p: base.sha(ROOT / p) for p in SOURCES}


def check_grant(grant, cfg, arm, seed, inputs, out):
    """Metadata only; reject before payload access or output creation."""
    if (grant.get('id') != cfg['id'] or grant.get('execute') is not True
            or grant.get('verdict') != 'PASS' or grant.get('distinct_from_author') is not True
            or not grant.get('reviewer') or grant.get('reviewer') == grant.get('author')
            or grant.get('sources_sha256') != source_hashes()
            or grant.get('arm') != arm or grant.get('seed') != seed
            or arm not in cfg['arms'] or seed not in cfg['seeds']
            or Path(grant.get('inputs', '')).resolve() != inputs.resolve()
            or Path(grant.get('output', '')).resolve() != out.resolve()
            or not grant.get('data_exposure_ledger_sha256')
            or not grant.get('commit') or not grant.get('attempt_id')
            or not grant.get('gpu_uuid') or not grant.get('common_lock')
            or not grant.get('input_manifest_sha256')
            or grant.get('budget') != cfg['budgets']
            or grant.get('data_version') != cfg['data_version']):
        raise base.Stop('STOP missing/mismatched reviewed grant')
    ledger = Path(grant.get('data_exposure_ledger', ''))
    if not ledger.is_file() or base.sha(ledger) != grant['data_exposure_ledger_sha256']:
        raise base.Stop('STOP exposure ledger binding')
    if seed != 0:
        gate_path = Path(grant.get('seed0_gate', ''))
        if not gate_path.is_file() or base.sha(gate_path) != grant.get('seed0_gate_sha256'):
            raise base.Stop('STOP seed0 gate binding')
        gate = json.loads(gate_path.read_text())
        if (gate.get('verdict') != 'PASS' or gate.get('distinct_from_author') is not True
                or gate.get('sources_sha256') != source_hashes()
                or gate.get('eval_scores_opened') is not False
                or set(gate.get('receipts_sha256', {})) != {'T-s0', 'R-s0'}):
            raise base.Stop('STOP both seed0 integrity receipts required; no score selection')
        initial_hashes = set()
        for name, pin in gate['receipts_sha256'].items():
            receipt_path=Path(gate['receipts'][name])
            if base.sha(receipt_path) != pin:raise base.Stop('STOP seed0 receipt hash')
            receipt=json.loads(receipt_path.read_text())
            supervision=json.loads((receipt_path.parent/'SUPERVISOR.json').read_text())
            if (receipt.get('status')!='COMPLETED' or supervision.get('status')!='COMPLETED'
                    or supervision.get('receipt_sha256')!=pin
                    or receipt.get('arm')!=name[0] or receipt.get('seed')!=0
                    or receipt.get('input_manifest_sha256')!=grant['input_manifest_sha256']
                    or receipt.get('source_sha256')!=source_hashes()):
                raise base.Stop('STOP seed0 integrity receipt mismatch')
            initial_hashes.add(receipt.get('shared_initial_sha256'))
        if len(initial_hashes) != 1 or None in initial_hashes:
            raise base.Stop('STOP seed0 shared initial weights differ')
    if out.exists():
        raise base.Stop('STOP output path already exists; new attempt required')


def watchdog(out, seconds, rss_cap, done):
    start = time.monotonic()
    while not done.wait(0.5):
        try:
            rss = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                           if line.startswith('VmRSS:'))) * 1024
            reason = ('wall' if time.monotonic()-start >= seconds else
                      'rss' if rss > rss_cap else None)
        except Exception as exc:
            reason = 'watchdog-read-failure:' + repr(exc)
        if reason:
            base.save_json(out / 'STOP.json', {'reason': reason, 'status': 'INCOMPLETE',
                                              'elapsed_seconds': time.monotonic()-start})
            os._exit(87)  # self only; no retry, no complete receipt


def train(torch, cfg, arm, seed, x, C, S, ctx, out, metadata):
    """Same N1 loss/selection/CAL; no performance-triggered seed stop."""
    import numpy as np
    t = cfg['training']; device = torch.device('cuda:0')
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = tm002.build_model(torch, cfg, arm, seed)
    initial = base.hashlib.sha256()
    for key,value in sorted(tm002.shared_state(model).items()):
        initial.update(key.encode());initial.update(value.detach().numpy().tobytes())
    metadata = dict(metadata,shared_initial_sha256=initial.hexdigest())
    model = model.to(device).float()
    data = base.Data(torch, device, x, C, S, ctx)
    rows = {r: torch.as_tensor(np.flatnonzero(C[r]), dtype=torch.int64, device=device)
            for r in ('fit',) + base.ROLES}
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    gen = torch.Generator(device='cpu'); gen.manual_seed(seed)
    F = torch.nn.functional
    best = {'epoch': 0, 'nll': float('inf'), 'state': None}; curve = []
    for epoch in range(1, t['max_epochs'] + 1):
        model.train(); order = rows['fit'][torch.randperm(len(rows['fit']), generator=gen).to(device)]
        for offset in range(0, len(order), t['batch_size']):
            idx = order[offset:offset+t['batch_size']]
            zp, zw, zg = model(data.x[idx], data.tokens(idx))
            loss = (F.cross_entropy(zp, data.y10[idx])
                    + base.masked_mean(torch,F.binary_cross_entropy_with_logits(zw,data.whiff[idx],reduction='none'),data.is_whiff_row[idx].float())
                    + base.masked_mean(torch,F.binary_cross_entropy_with_logits(zg,data.gb[idx],reduction='none'),data.is_bip_row[idx].float()))
            if not torch.isfinite(loss): raise base.Stop('STOP non-finite loss')
            opt.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise base.Stop('STOP non-finite gradient')
            opt.step()
        z = base.predict(torch,model,data,rows['tuning'],t['eval_batch_size'])
        v = base.nll7(base.seven_from_margins(*(z[h] for h in base.HEADS)),C['y7'][C['tuning']],cfg['nll_clip'])
        if not np.isfinite(v): raise base.Stop('STOP non-finite SELECT')
        if v < best['nll']:
            best = {'epoch':epoch,'nll':v,'state':{k:w.detach().cpu().clone() for k,w in model.state_dict().items()}}
        curve.append({'epoch':epoch,'select_nll7':v})
        print(json.dumps(curve[-1]),flush=True)
        if epoch-best['epoch'] >= t['patience']: break
    model.load_state_dict(best['state'])
    margins={r:base.predict(torch,model,data,rows[r],t['eval_batch_size']) for r in base.ROLES}
    check=base.nll7(base.seven_from_margins(*(margins['tuning'][h] for h in base.HEADS)),C['y7'][C['tuning']],cfg['nll_clip'])
    if abs(check-best['nll'])>1e-9: raise base.Stop('STOP restored SELECT mismatch')
    cal=C['calibration']; yc=C['y10'][cal]; z=margins['calibration']
    wm=yc==1; bm=np.isin(yc,base.BIP)
    temp={base.HEADS[0]:base.temperature(z[base.HEADS[0]],yc),
          base.HEADS[1]:base.temperature(z[base.HEADS[1]][wm],C['narrow_whiff'][cal][wm]),
          base.HEADS[2]:base.temperature(z[base.HEADS[2]][bm],C['ground_ball_per_pitch'][cal][bm])}
    q=base.seven_from_margins(*(margins['evaluation'][h] for h in base.HEADS),T=temp)
    np.save(out/'seven-leaf-evaluation.npy',q,allow_pickle=False)  # no EVAL metric computed
    np.savez(out/'margins.npz',**{f'{r}__{h}':margins[r][h] for r in base.ROLES for h in base.HEADS})
    torch.save(best['state'],out/'model-state.pt')
    base.save_json(out/'temperatures.json',temp);base.save_json(out/'curve.json',curve)
    files=('seven-leaf-evaluation.npy','margins.npz','model-state.pt','temperatures.json','curve.json')
    return dict(metadata,status='COMPLETED',arm=arm,seed=seed,best_epoch=best['epoch'],
                best_select_nll7=best['nll'],eval_scores_opened=False,parameters=sum(p.numel() for p in model.parameters()),
                rows={r:int(C[r].sum()) for r in rows},
                versions={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                          'scipy':__import__('scipy').__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                outputs_sha256={f:base.sha(out/f) for f in files})


def run(args):
    cfg, bc = tm002.load_config(args.config)
    grant=json.loads(args.grant.read_text())
    check_grant(grant,cfg,args.arm,args.seed,args.inputs,args.out)
    if sys.platform != 'linux' or not os.environ.get('GPU_JOB_ID') or not os.environ.get('GPU_JOB_DIR'):
        raise base.Stop('STOP only Linux gpu run jobs allowed; no CPU/MPS fallback')
    job=base.job_env()
    if job.get('commit') != grant['commit'] or job.get('dirty') != '0':
        raise base.Stop('STOP clean launcher commit binding')
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not selected or ',' in selected: raise base.Stop('STOP single GPU required')
    uuid=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    if uuid != grant['gpu_uuid']: raise base.Stop('STOP GPU UUID mismatch')
    lock=Path(grant['common_lock']).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.out.mkdir(parents=True,exist_ok=False)
    done=threading.Event(); start=time.monotonic()
    watcher=threading.Thread(target=watchdog,args=(args.out,cfg['budgets']['per_fit_and_cal_seconds'],
                                                  cfg['budgets']['host_rss_bytes_per_fit'],done),daemon=True)
    watcher.start()
    try:
        manifest=args.inputs/bc['inputs']['manifest_file']
        if base.sha(manifest) != grant['input_manifest_sha256']: raise base.Stop('STOP input manifest grant binding')
        _,manifest_sha,input_sha,_=base.verify_package(args.inputs,bc,ROOT/cfg['base_config'])
        torch,_=base.imports()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise base.Stop('STOP one CUDA device required')
        cap=cfg['budgets']['gpu_reserved_bytes']; total=torch.cuda.get_device_properties(0).total_memory
        if cap>=total:raise base.Stop('STOP registered GPU cap leaves no device headroom')
        torch.cuda.set_per_process_memory_fraction(cap/total,0)
        torch.cuda.reset_peak_memory_stats()
        x,C,S=base.load_package(args.inputs,bc)
        for role,key in (('FIT','fit'),('SELECT','tuning'),('CAL','calibration'),('EVAL','evaluation')):
            if int(C[key].sum()) != cfg['expected_rows'][role]:raise base.Stop('STOP registered role count mismatch')
        ctx=base.build_context_index(S['pitcher'],S['game_pk'],S['at_bat_number'],S['pitch_number'],bc['context']['max_length'])
        context=base.check_context(ctx,S['pitcher'],S['game_pk'],S['at_bat_number'],S['pitch_number'])
        receipt=train(torch,bc,args.arm,args.seed,x,C,S,ctx,args.out,
                      {'id':cfg['id'],'data_role':cfg['data_role'],'data_version':cfg['data_version'],'commit':grant['commit'],
                       'attempt_id':grant['attempt_id'],'source_sha256':source_hashes(),'grant_sha256':base.sha(args.grant),
                       'input_manifest_sha256':manifest_sha,'inputs_sha256':input_sha,'gpu_uuid':uuid,'job':job,'context':context})
        receipt['elapsed_seconds']=time.monotonic()-start
        base.save_json(args.out/'receipt.json',receipt)
    except Exception as exc:
        base.save_json(args.out/'STOP.json',{'status':'INCOMPLETE','reason':repr(exc),'elapsed_seconds':time.monotonic()-start})
        raise
    finally:
        done.set();watcher.join();lock.close()


def supervise(args):
    cfg, _ = tm002.load_config(args.config)
    grant=json.loads(args.grant.read_text())
    check_grant(grant,cfg,args.arm,args.seed,args.inputs,args.out)
    if sys.platform != 'linux' or not os.environ.get('GPU_JOB_ID'):
        raise base.Stop('STOP supervisor requires Linux gpu run job')
    child=subprocess.Popen([sys.executable,'-m','experiments.ws_seq.train_tm002',
                            *sys.argv[1:],'--worker'])
    started=time.monotonic(); reason=None; peak=0
    try:
        while child.poll() is None:
            if time.monotonic()-started >= cfg['budgets']['per_fit_and_cal_seconds']:
                reason='wall';break
            try:
                status=Path(f'/proc/{child.pid}/status').read_text()
                rss=int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))*1024
                peak=max(peak,rss)
                if rss>cfg['budgets']['host_rss_bytes_per_fit']:reason='rss';break
            except (OSError,StopIteration):
                if child.poll() is not None:break
                reason='supervisor-memory-read-failure';break
            time.sleep(0.25)
    finally:
        if child.poll() is None:
            # This Popen handle belongs only to the child created above.
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    result={'status':'COMPLETED' if child.returncode==0 and reason is None else 'INCOMPLETE',
            'returncode':child.returncode,'reason':reason,'peak_worker_rss_bytes':peak,
            'elapsed_seconds':time.monotonic()-started,'grant_sha256':base.sha(args.grant),
            'receipt_sha256':base.sha(args.out/'receipt.json') if (args.out/'receipt.json').exists() else None}
    if args.out.is_dir():base.save_json(args.out/'SUPERVISOR.json',result)
    return 0 if result['status']=='COMPLETED' else 87


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/EXP-WS-TM-002.json')
    p.add_argument('--grant',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=('T','R'),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    try:
        if args.worker:run(args);return 0
        return supervise(args)
    except (base.Stop,base.HashMismatch,ValueError,KeyError,OSError) as exc:
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
