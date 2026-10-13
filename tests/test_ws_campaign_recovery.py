import json
import pytest
import torch
from experiments.ws_campaign import train, queue


def test_fixed_recovery_backend_and_non_k_refusal():
    try:
        train.configure_backend({'arm':'K'}, {'execution_policy':'K2_GENERATED_PROBE_FAST_V1'})
        assert torch.backends.cudnn.enabled
        assert not torch.backends.cudnn.deterministic
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        with pytest.raises(ValueError):
            train.configure_backend({'arm':'T'}, {'execution_policy':'K2_GENERATED_PROBE_FAST_V1'})
        train.configure_backend({'arm':'K'}, {})
        assert torch.backends.cudnn.deterministic and torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


def test_recovery_queue_selects_exact_twelve_only(monkeypatch,tmp_path):
    cfg=json.loads((queue.runtime.ROOT/'configs/EXP-WS-CAMPAIGN-001.json').read_text())
    jobs=[dict(j, attempt_id=j['id'], output='/generated/'+j['id'],gpu_uuid='generated', common_lock='generated') for j in cfg['jobs']]
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps({'jobs':jobs}))
    grant=tmp_path/'grant.json'; grant.write_text(json.dumps({'recovery_amendment_sha256':'a'*64}))
    calls=[]
    monkeypatch.setattr(queue.runtime,'load_job',lambda a,b,c:calls.append(c))
    selected=queue.validate(manifest,grant)
    assert len(selected)==12 and len(calls)==12 and all(j['arm']=='K' for j in selected)
    assert sum(j['calibrations'] for j in selected)==36
