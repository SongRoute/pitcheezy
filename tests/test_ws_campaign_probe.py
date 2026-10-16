"""Generated-only probe contract and exact model path; CPU, tiny batch."""
import torch
from experiments.ws_campaign.probe_k import CASES, measure


def test_fixed_probe_measurements(tmp_path):
    for case in CASES:
        out = tmp_path / case; out.mkdir()
        result = measure(case, out, device='cpu', batch=4, iterations=1, warmup=0)
        ends = [e for e in result['events'] if e['event'] == 'end']
        assert [e['phase'] for e in ends] == ['forward','loss','backward','step','eval_forward']
        assert all(e['seconds'] >= 0 for e in ends)
        assert result['synthetic_only'] and result['peak_cuda_bytes'] == 0
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.enabled = True
