"""Single-job scientific implementation. Only reviewed runtime may launch real work."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from experiments.ws_seq import train_seq as base
from experiments.ws_campaign.models import build_model


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_inputs(inputs, cfg):
    inputs = Path(inputs); bc = cfg['base']
    man = json.loads((inputs / 'manifest.json').read_text())
    for name in [bc['inputs'][k] for k in ('features_file', 'corpus_file', 'sequence_file')]:
        if base.sha(inputs / name) != man['files'][name]['sha256']:
            raise ValueError('input package hash mismatch: ' + name)
    x, c, s = base.load_package(inputs, bc)
    from experiments.ws_seq.tm002 import check_role_partition
    check_role_partition(c)
    ctx = base.build_context_index(s['pitcher'], s['game_pk'], s['at_bat_number'], s['pitch_number'], 32)
    base.check_context(ctx, s['pitcher'], s['game_pk'], s['at_bat_number'], s['pitch_number'])
    valid = ctx >= 0
    role = np.select([c[k] for k in ('fit',) + base.ROLES], [0, 1, 2, 3])
    if not (role[np.maximum(ctx, 0)] == role[:, None])[valid].all():
        raise ValueError('context crosses role boundary')
    return x, c, s, ctx


def predict(model, data, keys, rows, batch):
    model.eval(); collected = {}
    with torch.no_grad():
        for offset in range(0, len(rows), batch):
            idx = rows[offset:offset + batch]
            z = model(data.x[idx], data.tokens(idx) if model.seq else None, keys[idx])
            for k, v in z.items(): collected.setdefault(k, []).append(v.double().cpu().numpy())
    result = {k: np.concatenate(v) for k, v in collected.items()}
    if not result or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('empty/nonfinite prediction')
    return result


def probabilities(z, temperatures=None):
    if 'leaf' in z:
        from scipy.special import softmax
        return softmax(z['leaf'] / (temperatures or {}).get('leaf', 1.0), axis=1)
    return base.seven_from_margins(*(z[k] for k in base.HEADS), T=temperatures)


def losses(z, data, idx, y7):
    f = torch.nn.functional
    if 'leaf' in z:
        per = f.cross_entropy(z['leaf'], y7[idx], reduction='none')
        return per.mean(), per
    parent = f.cross_entropy(z['parent'], data.y10[idx], reduction='none')
    whiff = f.binary_cross_entropy_with_logits(z['narrow_whiff'], data.whiff[idx], reduction='none')
    ground = f.binary_cross_entropy_with_logits(z['ground_ball_per_pitch'], data.gb[idx], reduction='none')
    wm, gm = data.is_whiff_row[idx].float(), data.is_bip_row[idx].float()
    per = parent + whiff * wm * len(idx) / wm.sum().clamp(min=1) + ground * gm * len(idx) / gm.sum().clamp(min=1)
    return per.mean(), per


def support_groups(c, s, cfg):
    fit = np.flatnonzero(c['fit'])
    support = s.get('fit_prior_support')
    if support is None or support.shape != c['fit'].shape or support.dtype.kind not in 'iu' or (support < 0).any():
        raise ValueError('registered chronological FIT support counts required')
    group = np.maximum(0, np.searchsorted(cfg['boundaries'], support, side='right') - 1)
    active = list(range(len(cfg['boundaries'])))
    for g in active[:-1]:
        if int(((group == g) & c['fit']).sum()) < cfg['min_rows']:
            group[group == g] = g + 1
    present = sorted(set(group[fit].tolist()))
    if len(present) > 1 and int(((group == present[-1]) & c['fit']).sum()) < cfg['min_rows']:
        group[group == present[-1]] = present[-2]
    present = sorted(set(group[fit].tolist()))
    if not present: raise ValueError('empty FIT support groups')
    remap = {g: i for i, g in enumerate(present)}
    return np.array([remap.get(int(g), 0) for g in group]), {'strictly_prior_fit_support': True,
             'groups': {str(remap[g]): int(((group == g) & c['fit']).sum()) for g in present}}


def dependency(job, metadata, kind):
    ids = [x for x in job['depends_on'] if (('pre-s' in x) if kind == 'pretrain' else
                                             ('teacher-s' in x) if kind == 'teacher' else x.startswith('A-T-s'))]
    if len(ids) != 1: raise ValueError('exact one dependency required: ' + kind)
    root = Path(metadata['dependencies'][ids[0]])
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt.get('status') != 'COMPLETED': raise ValueError('incomplete scientific dependency')
    for name, sha in receipt['outputs_sha256'].items():
        if base.sha(root / name) != sha: raise ValueError('dependency artifact hash mismatch')
    return root, receipt


def run_pretrain(job, cfg, model, data, keys, fit, out):
    p = cfg['pretraining']; model.train()
    head = torch.nn.Linear(cfg['base']['model']['token_dim'], 10).to(data.x.device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                           lr=cfg['base']['training']['lr'], weight_decay=cfg['base']['training']['weight_decay'])
    steps = 0; start = time.monotonic(); final_loss = None
    for epoch in range(p['epochs']):
        order = fit[torch.randperm(len(fit), device=fit.device)]
        for offset in range(0, len(order), p['batch_size']):
            idx = order[offset:offset + p['batch_size']]; tok = data.tokens(idx)
            keep = tok['valid'].any(1); idx = idx[keep]
            if len(idx) < 2: continue
            tok = {k: v[keep] for k, v in tok.items()}
            if job['arm'] == 'mask':
                masked = tok['valid'] & (torch.rand_like(tok['valid'], dtype=torch.float32) < p['masked_probability'])
                masked[:, 0] |= tok['valid'][:, 0]
                encoded, _ = model.encode(tok, masked_out=masked)
                loss = torch.nn.functional.cross_entropy(head(encoded[masked]), tok['out'][masked])
            elif job['arm'] == 'contrast':
                _, a = model.encode(model.prepare_tokens(tok, keys[idx], augment=True))
                _, b = model.encode(model.prepare_tokens(tok, keys[idx], augment=True))
                a = torch.nn.functional.normalize(a, dim=1); b = torch.nn.functional.normalize(b, dim=1)
                similarity = a @ b.T / p['contrast_temperature']; labels = torch.arange(len(idx), device=fit.device)
                loss = (torch.nn.functional.cross_entropy(similarity, labels) + torch.nn.functional.cross_entropy(similarity.T, labels)) / 2
            else: raise ValueError('unknown pretraining objective')
            update(loss, model, opt, extra=head.parameters())
            steps += 1; final_loss = float(loss.detach())
            if steps >= p['max_steps']: break
        if steps >= p['max_steps']: break
    if not steps: raise ValueError('no nonempty FIT context pretraining steps')
    # Only token embeddings+encoder transfer. Trunk and supervised heads remain same-seed scratch.
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k.startswith('net.encoder.') or k.startswith('net.e_') or k == 'net.no_context'}
    torch.save(state, out / 'pretrained-state.pt')
    return {'optimization_steps': steps, 'elapsed_compute_seconds': time.monotonic() - start,
            'objective_final_loss': final_loss, 'files': ['pretrained-state.pt'], 'calibrations': 0}


def update(loss, model, opt, extra=()):
    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
    opt.zero_grad(set_to_none=True); loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(extra)):
        raise ValueError('nonfinite gradient')
    opt.step()


def run_supervised(job, cfg, model, data, keys, rows, c, s, out, metadata):
    t = cfg['base']['training']; arm = job['arm']; device = data.x.device
    y7 = torch.as_tensor(c['y7'], device=device, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    steps_per_epoch = math.ceil(len(rows['fit']) / t['batch_size'])
    max_steps = t['max_epochs'] * steps_per_epoch; paired = '-pre' in arm or '-scratch' in arm
    parent_steps = 0; paired_info = None
    if paired:
        root, receipt = dependency(job, metadata, 'pretrain'); parent_steps = receipt['optimization_steps']
        max_steps = cfg['pretraining']['finetune_epochs'] * steps_per_epoch
        if arm.endswith('-pre'):
            state = torch.load(root / 'pretrained-state.pt', map_location=device, weights_only=True)
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys: raise ValueError('invalid pretrained state')
        else: max_steps += parent_steps
        paired_info = {'rule': 'optimizer_steps', 'pretraining_steps': parent_steps,
                       'supervised_target_steps': max_steps, 'total_target_steps': max_steps + (parent_steps if arm.endswith('-pre') else 0),
                       'wall_compute_matched': False}
    group_info = None
    if arm == 'T-dro':
        groups, group_info = support_groups(c, s, cfg['dro']); groups = torch.as_tensor(groups, device=device)
        group_weights = torch.ones(len(group_info['groups']), device=device) / len(group_info['groups'])
    teacher = None
    if arm == 'N0-distill':
        from scipy.special import softmax, expit
        teachers = []
        for seed in cfg['seeds']:
            teacher_id = f'B-teacher-s{seed}'
            if teacher_id not in job['depends_on']: raise ValueError('all fixed teacher seeds required')
            root = Path(metadata['dependencies'][teacher_id])
            receipt = json.loads((root / 'receipt.json').read_text())
            if receipt.get('status') != 'COMPLETED' or receipt.get('seed') != seed:
                raise ValueError('teacher receipt identity')
            if base.sha(root / 'teacher-fit.npz') != receipt['outputs_sha256']['teacher-fit.npz']:
                raise ValueError('teacher artifact hash')
            with np.load(root / 'teacher-fit.npz', allow_pickle=False) as z:
                if not np.array_equal(z['keys'], c['keys'][c['fit']]): raise ValueError('teacher FIT keys mismatch')
                tau = cfg['distill']['temperature']
                teachers.append({k: softmax(z[k].astype(np.float64)/tau,axis=1) if k == 'parent'
                                 else expit(z[k].astype(np.float64)/tau) for k in base.HEADS})
        teacher = {k: torch.as_tensor(np.mean([z[k] for z in teachers],axis=0), device=device,dtype=torch.float32)
                   for k in base.HEADS}
        teacher_row = torch.full((len(c['fit']),), -1, dtype=torch.long, device=device)
        teacher_row[rows['fit']] = torch.arange(len(rows['fit']), device=device)
    best = None; best_nll = math.inf; best_epoch = 0; curve = []; steps = 0; epoch = 0
    while steps < max_steps:
        epoch += 1; model.train(); order = rows['fit'][torch.randperm(len(rows['fit']), device=device)]
        for offset in range(0, len(order), t['batch_size']):
            idx = order[offset:offset + t['batch_size']]
            z = model(data.x[idx], data.tokens(idx) if model.seq else None, keys[idx])
            loss, per = losses(z, data, idx, y7)
            if arm == 'T-dro':
                means = torch.stack([per[groups[idx] == g].mean() if (groups[idx] == g).any() else per.sum() * 0
                                     for g in range(len(group_weights))])
                present = torch.stack([(groups[idx] == g).any() for g in range(len(group_weights))])
                with torch.no_grad():
                    group_weights *= torch.exp(cfg['dro']['eta'] * means.detach().clamp(max=50)); group_weights /= group_weights.sum()
                weights = group_weights * present; weights = weights / weights.sum()
                loss = (weights * means).sum()
            if teacher is not None:
                tau = cfg['distill']['temperature']; ti = teacher_row[idx]; f = torch.nn.functional
                soft = teacher['parent'][ti]
                distill = f.kl_div(f.log_softmax(z['parent'] / tau, dim=1), soft, reduction='batchmean') * tau ** 2
                for name, mask in [('narrow_whiff', data.is_whiff_row[idx]), ('ground_ball_per_pitch', data.is_bip_row[idx])]:
                    target = teacher[name][ti]
                    dl = f.binary_cross_entropy_with_logits(z[name] / tau, target, reduction='none') * tau ** 2
                    distill += base.masked_mean(torch, dl, mask.float())
                w = cfg['distill']['weight']; loss = (1 - w) * loss + w * distill
            update(loss, model, opt); steps += 1
            if steps >= max_steps: break
        z = predict(model, data, keys, rows['tuning'], t['eval_batch_size'])
        nll = base.nll7(probabilities(z), c['y7'][c['tuning']], cfg['base']['nll_clip'])
        if not math.isfinite(nll): raise ValueError('nonfinite SELECT')
        if nll < best_nll:
            best_nll, best_epoch = nll, epoch
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        curve.append({'epoch': epoch, 'steps': steps, 'select_nll7': nll})
        if not paired and epoch - best_epoch >= t['patience']: break
    model.load_state_dict(best)
    selected = predict(model, data, keys, rows['tuning'], t['eval_batch_size'])
    restored = base.nll7(probabilities(selected), c['y7'][c['tuning']], cfg['base']['nll_clip'])
    if abs(restored - best_nll) > 1e-9: raise ValueError('restored SELECT mismatch')
    z = predict(model, data, keys, rows['calibration'], t['eval_batch_size']); cal = c['calibration']
    if 'leaf' in z:
        temperatures = {'leaf': base.temperature(z['leaf'], c['y7'][cal])}
    else:
        yc = c['y10'][cal]; wm = yc == 1; bm = np.isin(yc, base.BIP)
        if not wm.any() or not bm.any(): raise ValueError('empty CAL subhead')
        temperatures = {'parent': base.temperature(z['parent'], yc),
                        'narrow_whiff': base.temperature(z['narrow_whiff'][wm], c['narrow_whiff'][cal][wm]),
                        'ground_ball_per_pitch': base.temperature(z['ground_ball_per_pitch'][bm], c['ground_ball_per_pitch'][cal][bm])}
    torch.save(best, out / 'model-state.pt'); save_json(out / 'temperatures.json', temperatures)
    save_json(out / 'curve.json', curve)
    np.savez(out / 'calibration-margins.npz', keys=c['keys'][cal], **z)
    return {'optimization_steps': steps, 'best_epoch': best_epoch, 'best_select_nll7': best_nll,
            'paired_compute': paired_info, 'support_groups': group_info, 'calibrations': len(temperatures),
            'files': ['model-state.pt', 'temperatures.json', 'curve.json', 'calibration-margins.npz']}


def run_job(job, cfg, inputs, out, metadata):
    """Runtime has already checked grant/source/UUID/lock/limits; no CLI bypass."""
    synthetic = metadata.get('synthetic') is True
    if not synthetic and (not torch.cuda.is_available() or metadata.get('runtime_reviewed') is not True):
        raise ValueError('reviewed CUDA runtime required')
    registered = [x for x in cfg['jobs'] if x['id'] == job['id']]
    if len(registered) != 1 or any(job.get(k) != v for k, v in registered[0].items()):
        raise ValueError('job differs from registered scientific slot')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('scientific output must be empty')
    x, c, s, ctx = metadata['synthetic_arrays'] if synthetic else load_inputs(inputs, cfg)
    if job['arm'] == 'T-dro' and not synthetic:
        path = Path(inputs) / 'fit-prior-support.npy'
        manifest = json.loads((Path(inputs) / 'manifest.json').read_text())
        if base.sha(path) != manifest['files']['fit-prior-support.npy']['sha256']:
            raise ValueError('FIT chronological support manifest binding')
        s['fit_prior_support'] = np.load(path, allow_pickle=False)
    if job['arm'] == 'T-histmask':
        columns = cfg['variants']['history_columns']
        if not columns or len(set(columns)) != len(columns) or any(type(i) is not int or not 0 <= i < x.shape[1] for i in columns):
            raise ValueError('unbound histmask manifest')
    device = torch.device('cpu' if synthetic else 'cuda:0'); seed = job['seed']
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_model(cfg, job['arm'] if job['kind'] == 'supervised' else 'T', seed).to(device)
    data = base.Data(torch, device, x, c, s, ctx)
    keys = torch.as_tensor(c['keys'], device=device, dtype=torch.long)
    rows = {r: torch.as_tensor(np.flatnonzero(c[r]), device=device, dtype=torch.long) for r in ('fit', 'tuning', 'calibration')}
    if any(not len(v) for v in rows.values()): raise ValueError('empty FIT/SELECT/CAL role')
    started = time.monotonic()
    if job['kind'] == 'pretrain':
        fitctx = ctx[c['fit']]; valid = fitctx >= 0
        if not c['fit'][np.maximum(fitctx, 0)][valid].all(): raise ValueError('pretraining context outside FIT')
        result = run_pretrain(job, cfg, model, data, keys, rows['fit'], out)
    elif job['kind'] == 'teacher':
        root, _ = dependency(job, metadata, 'model')
        model.load_state_dict(torch.load(root / 'model-state.pt', map_location=device, weights_only=True))
        z = predict(model, data, keys, rows['fit'], cfg['base']['training']['eval_batch_size'])
        np.savez(out / 'teacher-fit.npz', keys=c['keys'][c['fit']], **{k:v.astype(np.float32) for k,v in z.items()})
        result = {'files': ['teacher-fit.npz'], 'calibrations': 0, 'optimization_steps': 0}
    elif job['kind'] == 'supervised':
        result = run_supervised(job, cfg, model, data, keys, rows, c, s, out, metadata)
    else: raise ValueError('unregistered job kind')
    files = result.pop('files')
    if result['calibrations'] != job['calibrations']: raise ValueError('CAL budget mismatch')
    return dict(result, status='COMPLETED', id=job['id'], arm=job['arm'], seed=seed, kind=job['kind'],
                data_role=cfg['data_role'], data_version=cfg['data_version'], eval_scores_opened=False,
                evaluation_predictions_computed=False, elapsed_seconds=time.monotonic()-started,
                rows={r:len(v) for r,v in rows.items()}, outputs_sha256={f:base.sha(out/f) for f in files},
                versions={'torch':torch.__version__, 'numpy':np.__version__, 'cuda':torch.version.cuda})
