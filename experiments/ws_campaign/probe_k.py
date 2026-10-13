"""Fixed generated-only CUDA diagnostic; no package/checkpoint/result reads."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from experiments.ws_campaign.runtime import ROOT, require, sha
from experiments.ws_campaign.diagnostic_supervisor import supervise

CASES = {'K-det': ('K', True, True), 'C-det': ('C', True, True),
         'K-fast': ('K', False, True), 'K-native': ('K', True, False)}
CONFIG = 'configs/EXP-WS-CAMPAIGN-001.json'
SOURCES = ('experiments/ws_campaign/diagnostic_supervisor.py', CONFIG, 'experiments/ws_campaign/probe_k.py', 'experiments/ws_campaign/models.py',
           'experiments/ws_campaign/train.py', 'experiments/ws_campaign/runtime.py',
           'experiments/ws_seq/train_seq.py', 'experiments/ws_seq/tm002.py',
           'experiments/ws_seq/train_tm002.py')


def generated(torch, batch, device):
    g = torch.Generator(device=device).manual_seed(20261013)
    x = torch.randn(batch, 199, generator=g, device=device)
    row = torch.arange(batch, device=device)
    pos = torch.arange(32, device=device)[None, :].expand(batch, -1)
    tok = {'valid': pos < (row % 33)[:, None], 'pos': pos + 1}
    for key, upper in [('type',19), ('out',10), ('balls',4), ('strikes',3), ('same',2)]:
        tok[key] = torch.randint(upper, (batch,32), generator=g, device=device)
    data = SimpleNamespace(y10=row % 10, whiff=(row // 10 % 2).float(), gb=(row // 20 % 2).float(),
                           is_whiff_row=row % 10 == 1,
                           is_bip_row=torch.isin(row % 10, torch.tensor([3,4,5,6,7,9], device=device)))
    return x, tok, data, row


def measure(case, out, device='cuda', batch=2048, iterations=3, warmup=1):
    import torch
    from experiments.ws_campaign.models import build_model
    from experiments.ws_campaign.train import losses
    cfg = json.loads((ROOT / CONFIG).read_text())
    arm, deterministic, cudnn = CASES[case]
    torch.set_num_threads(1); torch.manual_seed(0)
    torch.backends.cudnn.enabled = cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if device == 'cuda':
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one CUDA device')
        torch.cuda.set_per_process_memory_fraction((8 << 30) / torch.cuda.get_device_properties(0).total_memory)
    (out/'case.json').write_text(json.dumps({'case':case, 'arm':arm, 'seed':0, 'data_seed':20261013,
        'batch':batch, 'length':32, 'dtype':'float32', 'deterministic':deterministic, 'cudnn':cudnn,
        'tf32':False, 'benchmark':False, 'torch':torch.__version__, 'cuda':torch.version.cuda,
        'cudnn_version':torch.backends.cudnn.version()}, indent=2)+'\n')
    model = build_model(cfg, arm, 0).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x, tok, data, idx = generated(torch, batch, device)
    y7 = idx % 7
    sync = torch.cuda.synchronize if device == 'cuda' else lambda: None
    events = []
    def emit(event):
        events.append(event)
        with (out / 'phases.jsonl').open('a') as f: f.write(json.dumps(event) + '\n')
    def timed(phase, step, fn):
        emit({'phase': phase, 'iteration':step, 'event':'start'})
        sync(); start = time.monotonic(); value = fn(); sync()
        emit({'phase':phase, 'iteration':step, 'event':'end', 'seconds':time.monotonic()-start})
        return value
    for step in range(-warmup, iterations):
        opt.zero_grad(set_to_none=True)
        z = timed('forward', step, lambda: model(x, tok))
        loss, _ = timed('loss', step, lambda: losses(z, data, idx, y7))
        timed('backward', step, loss.backward)
        require(bool(torch.isfinite(loss)), 'finite synthetic loss')
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'finite gradient')
        timed('step', step, opt.step)
    model.eval()
    ex, et, _, _ = generated(torch, 8192 if device == 'cuda' else batch, device)
    with torch.no_grad(): timed('eval_forward', 0, lambda: model(ex, et))
    result = {'status':'COMPLETED', 'synthetic_only':True, 'case':case, 'seed':0, 'data_seed':20261013,
              'batch':batch, 'length':32, 'warmup':warmup, 'iterations':iterations, 'dtype':'float32',
              'cudnn':cudnn, 'deterministic':deterministic, 'benchmark':False, 'tf32':False,
              'versions':{'torch':torch.__version__, 'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version()},
              'events':events, 'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device == 'cuda' else 0}
    (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grant', type=Path, required=True)
    p.add_argument('--worker', choices=CASES)
    args = p.parse_args(); grant = json.loads(args.grant.read_text())
    require(grant.get('execute') is True and grant.get('verdict') == 'PASS' and
            grant.get('author') and grant.get('reviewer') and grant['author'] != grant['reviewer'], 'independent grant')
    require(grant.get('sources_sha256') == {s:sha(ROOT/s) for s in SOURCES}, 'source hashes')
    require(sys.platform == 'linux' and os.environ.get('GPU_JOB_ID'), 'gpu run only')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(grant['gpu_index']), 'GPU index')
    require(grant.get('max_case_seconds') == 60 and grant.get('max_rss_bytes') == 4 << 30, 'fixed resource caps')
    env = dict(line.split('=',1) for line in (Path(os.environ['GPU_JOB_DIR'])/'job.env').read_text().splitlines() if '=' in line)
    require(len(env.get('COMMIT','')) >= 7 and grant['commit'].startswith(env['COMMIT']), 'source commit')
    out = Path(grant['output']); require(out.is_absolute(), 'absolute new output')
    if args.worker:
        try: measure(args.worker, out / args.worker)
        except Exception as exc:
            (out/args.worker/'ERROR.json').write_text(json.dumps({'type':type(exc).__name__, 'message':str(exc)}))
            raise
        return 0
    require(not out.exists(), 'immutable attempt output')
    uuid = subprocess.check_output(['nvidia-smi','-i',str(grant['gpu_index']),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    require(uuid == grant['gpu_uuid'], 'UUID')
    lock = Path(grant['common_lock']); require(lock.is_file(), 'existing shared lock')
    with lock.open() as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
        require(uuid not in processes.splitlines(), 'unoccupied GPU')
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        require(available >= 8 << 30, '8GiB host headroom')
        out.mkdir(parents=True, exist_ok=False)
        status = {}; start = time.monotonic()
        for case in CASES:
            require(time.monotonic()-start < 270, 'global wall')
            target = out / case; target.mkdir()
            status[case] = supervise([sys.executable,'-m','experiments.ws_campaign.probe_k',
                '--grant',str(args.grant),'--worker',case], target, min(60,270-(time.monotonic()-start)), 4 << 30, sha(args.grant), f.fileno())
            # CUDA/OOM errors stop; a bounded slow-path timeout allows the next fixed case.
            supervisor = json.loads((target/'SUPERVISOR.json').read_text())
            if status[case] and supervisor['reason'] != 'wall': break
        (out/'receipt.json').write_text(json.dumps({'status':'COMPLETED', 'scope':'diagnostic orchestration',
            'synthetic_only':True, 'case_returncodes':status, 'gpu_uuid':uuid,
            'grant_sha256':sha(args.grant)},indent=2)+'\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
