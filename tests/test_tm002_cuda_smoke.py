"""Explicit GPU-only synthetic smoke; no accepted package or runs-link access."""
import json
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from experiments.ws_seq import train_tm002 as trainer
from test_tm002_evaluation import make_bundle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-only smoke')


def test_cuda_uuid_allocator_t_r_and_linux_supervisor(tmp_path):
    assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
    assert torch.cuda.device_count()==1
    expected=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    prop=torch.cuda.get_device_properties(0);actual=str(getattr(prop,'uuid',''))
    print(json.dumps({'nvidia_uuid':expected,'torch_uuid':actual}),flush=True)
    assert actual and actual.removeprefix('GPU-')==expected.removeprefix('GPU-')
    torch.cuda.set_per_process_memory_fraction(16*1024**3/prop.total_memory,0)
    torch.cuda.reset_peak_memory_stats()
    _,_,grant,_,_=make_bundle(tmp_path/'synthetic',device=torch.device('cuda:0'),seeds=[0])
    for spec in grant['runs'].values():
        out=Path(spec['directory']);receipt=json.loads((out/'receipt.json').read_text())
        assert receipt['synthetic'] and receipt['status']=='COMPLETED'
        assert receipt['peak_gpu_allocated_bytes']>0
        assert np.isfinite(np.load(out/'seven-leaf-evaluation.npy',allow_pickle=False)).all()
    watched=tmp_path/'watched';watched.mkdir()
    (watched/'receipt.json').write_text(json.dumps({'status':'COMPLETED','synthetic':True}))
    rc=trainer.supervise_command([sys.executable,'-c','import time; time.sleep(1)'],watched,5,8*1024**3,'synthetic')
    result=json.loads((watched/'SUPERVISOR.json').read_text())
    assert rc==0 and result['peak_worker_rss_bytes']>0
