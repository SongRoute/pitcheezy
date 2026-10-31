"""Finite campaign variants; frozen SEQ-001 primitives remain untouched."""
import torch
from experiments.ws_seq import train_seq as base
from experiments.ws_seq import tm002


def build_model(cfg, arm, seed):
    bc = cfg['base']; nn = torch.nn
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = base.build_model(torch, bc, 'N0' if arm.startswith('N0') else 'N1')
        if arm in ('R', 'C', 'C-shuf', 'G'):
            torch.manual_seed(seed)
            net.encoder = tm002.packed_encoder(torch, bc)
            if arm == 'G':
                m = bc['model']
                net.encoder.lstm = nn.GRU(m['token_dim'], m['token_dim'], num_layers=m['encoder_layers'],
                                          dropout=m['encoder_dropout'], batch_first=True)
        elif arm == 'K':
            torch.manual_seed(seed)
            net.encoder = CausalTCN(bc['model']['token_dim'], cfg['variants']['tcn'])
        if arm == 'T-leaf7':
            torch.manual_seed(seed)
            net.head_leaf = nn.Linear(bc['model']['trunk_hidden'][-1] + bc['model']['token_dim'], 7)
            del net.head_parent, net.head_whiff, net.head_gb
    return Model(net, cfg, arm)


class CausalTCN(torch.nn.Module):
    def __init__(self, width, cfg):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Conv1d(width, width, cfg['kernel'], dilation=d)
                                          for d in cfg['dilations']])
        self.dropout = torch.nn.Dropout(cfg['dropout'])

    def forward(self, e, src_key_padding_mask):
        valid = (~src_key_padding_mask).unsqueeze(1)
        x = e.transpose(1, 2) * valid
        for layer in self.layers:
            pad = (layer.kernel_size[0] - 1) * layer.dilation[0]
            y = layer(torch.nn.functional.pad(x, (pad, 0)))
            x = (x + self.dropout(torch.nn.functional.gelu(y))) * valid
        return x.transpose(1, 2)


class Model(torch.nn.Module):
    def __init__(self, net, cfg, arm):
        super().__init__()
        self.net, self.cfg, self.arm, self.seq = net, cfg, arm, net.seq

    def prepare_tokens(self, tok, row_keys=None, *, augment=False):
        tok = {k: v.clone() for k, v in tok.items()}
        valid = tok['valid']; b, length = valid.shape
        if self.arm == 'T-recent4':
            tok['valid'][:, 4:] = False
        if self.arm in ('C', 'C-shuf', 'G', 'K'):
            lengths = valid.sum(1)
            order = (lengths[:, None] - 1 - torch.arange(length, device=valid.device)).clamp(min=0)
            order = torch.where(valid, order, torch.arange(length, device=valid.device)[None, :])
            tok = {k: v.gather(1, order) for k, v in tok.items()}
        if self.arm == 'C-shuf':
            if row_keys is None:
                raise ValueError('C-shuf requires stable pitch keys')
            # Fixed arithmetic permutation, keyed by game/PA/pitch, independent of labels and seed.
            key = row_keys[:, 0] * 1000003 + row_keys[:, 1] * 1009 + row_keys[:, 2]
            positions = torch.arange(length, device=valid.device)[None, :]
            rank = ((key[:, None] + positions * 104729) * (positions + 31)) % 2147483647
            rank = torch.where(tok['valid'], rank, rank + 2147483647)
            order = rank.argsort(dim=1, stable=True)
            tok = {k: v.gather(1, order) for k, v in tok.items()}
        if augment or (self.training and self.arm == 'T-tokdrop'):
            rate = self.cfg['variants']['token_dropout']
            tok['valid'] &= torch.rand((b, length), device=valid.device) >= rate
        return tok

    def encode(self, tok, *, masked_out=None):
        n = self.net; valid = tok['valid']; z = torch.zeros_like(tok['type'])
        e = (n.e_type(torch.where(valid, tok['type'], z + n.pad_type))
             + n.e_out(torch.where(valid, tok['out'], z + n.pad_out))
             + n.e_balls(torch.where(valid, tok['balls'], z))
             + n.e_strikes(torch.where(valid, tok['strikes'], z))
             + n.e_same(torch.where(valid, tok['same'], z))
             + n.e_pos(torch.where(valid, tok['pos'], z)))
        if masked_out is not None:
            # The selected event outcome cannot survive through its embedding.
            e = e - n.e_out(torch.where(valid, tok['out'], z + n.pad_out)) * masked_out[..., None]
        empty = ~valid.any(1); pad = ~valid
        pad = pad.clone(); pad[empty, 0] = False
        s = n.encoder(e, src_key_padding_mask=pad)
        w = valid[..., None].to(s.dtype)
        pooled = (s * w).sum(1) / w.sum(1).clamp(min=1)
        pooled = torch.where(empty[:, None], n.no_context.expand_as(pooled), pooled)
        return s, pooled

    def forward(self, x, tok=None, row_keys=None):
        if self.training and self.arm == 'T-histmask':
            cols = self.cfg['variants']['history_columns']
            if not cols:
                raise ValueError('history column manifest not bound')
            x = x.clone()
            mask = torch.rand(len(x), device=x.device) < self.cfg['variants']['history_mask_probability']
            rows = torch.nonzero(mask, as_tuple=True)[0]
            x[rows[:, None], torch.tensor(cols, device=x.device)[None, :]] = 0
        h = self.net.trunk(x)
        if self.seq:
            _, pooled = self.encode(self.prepare_tokens(tok, row_keys))
            h = torch.cat((h, pooled), dim=1)
        if self.arm == 'T-leaf7':
            return {'leaf': self.net.head_leaf(h)}
        return {'parent': self.net.head_parent(h), 'narrow_whiff': self.net.head_whiff(h).squeeze(-1),
                'ground_ball_per_pitch': self.net.head_gb(h).squeeze(-1)}
