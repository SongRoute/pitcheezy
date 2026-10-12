"""TM-002 preparation only: reuse frozen N1, replace only its encoder.

No payload loader or fit CLI. The original SEQ-001 trainer is NOT a TM-002
runner: its performance stop rule and soft wall cap differ from this contract.
"""
import json
import math
from pathlib import Path

from experiments.ws_seq import train_seq as base


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('non-finite configuration')
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
    finite(cfg)
    if (cfg['id'] != 'EXP-WS-TM-002' or cfg['execution_enabled'] is not False
            or cfg['arms'] != ['T', 'R'] or cfg['seeds'] != [0, 1, 2]):
        raise ValueError('preparation-only identity/arms/seeds gate')
    if (any(type(v) is not int or v <= 0 for v in cfg['budgets'].values())
            or cfg['budgets']['fits'] != 6 or cfg['budgets']['scalar_calibrations'] != 18
            or cfg['budgets']['max_fits_per_gpu'] != 1
            or cfg['budgets']['max_concurrent_fits'] > 4):
        raise ValueError('invalid finite budget')
    source = Path(__file__).resolve().parents[2] / cfg['base_config']
    if base.sha(source) != cfg['base_config_sha256']:
        raise ValueError('base config hash mismatch')
    base_cfg = base.load_config(source)
    if base.sha(Path(base.__file__)) != cfg['base_source_sha256']:
        raise ValueError('base source hash mismatch')
    if cfg['encoder'] != {'kind': 'lstm', 'layers': 2, 'width': 64,
                           'dropout': 0.1, 'bidirectional': False}:
        raise ValueError('unregistered encoder configuration')
    return cfg, base_cfg


def build_model(torch, base_cfg, arm, seed):
    """Same seed -> exact same N1 shared initial state; leave caller RNG unchanged.

Construct on CPU. Reset the training RNG separately to the same seed in each
future fit; dropout random consumption across architectures need not match.
"""
    if arm not in ('T', 'R') or seed not in (0, 1, 2):
        raise ValueError('unregistered arm/seed')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        net = base.build_model(torch, base_cfg, 'N1')
        if arm == 'R':
            # The N1 draw fixes trunk, embeddings, no_context AND heads first.
            torch.random.default_generator.manual_seed(seed)
            net.encoder = packed_encoder(torch, base_cfg)
    return net


def packed_encoder(torch, cfg):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            m = cfg['model']
            self.lstm = torch.nn.LSTM(m['token_dim'], m['token_dim'],
                                     num_layers=m['encoder_layers'],
                                     dropout=m['encoder_dropout'], batch_first=True)

        def forward(self, e, src_key_padding_mask):
            valid = ~src_key_padding_mask
            if valid.shape != e.shape[:2] or (valid[:, 1:] & ~valid[:, :-1]).any():
                raise base.Stop('STOP LSTM requires suffix padding')
            lengths = valid.sum(1)
            # N1 forward supplies one dummy valid step for all-empty contexts,
            # then replaces pooled output with the identical no_context vector.
            if (lengths < 1).any():
                raise base.Stop('STOP empty encoder input must use N1 dummy step')
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=e.shape[1])
            return out
    return Encoder()


def shared_state(net):
    return {k: v for k, v in net.state_dict().items() if not k.startswith('encoder.')}
