"""WC8-CONFIRM-001 step 1: one game x one predictor -> svc-r1-predictions-v1 json (no label is extracted or written).

    python -B scripts/wc8/predict.py --game <pk> --model {v4,bseq,r} --out <dir>     # fresh process per run

Per-game config = the booth instance config bytes (v4: XGBT+XGRUA3p+XGRUA5b24_AM-w_+T.json; bseq: the same bytes with the
predictor block = generic predictor_templates lit B-SEQ; r: R.json), with only these replaced: game (WC feed + sha from
registration-facts), predictions (path in <dir>), predictor.game_date / as_of_note, and for v4 predictor.config =
the repository path of the same mix pin bytes (291dc914). Load checks are build_pred.load's predictor side: pin sha,
instance rule (on a probe with the booth game block), generic sha, ext-expl vendored code + EXPL mix gate receipt,
mix trained_through from its components, lit refit gate + gate-time bundle pin. Then the unchanged booth predict
stage ``scripts/svc_r1/predict_stage.run`` (replay snapshots -> W1 v4 -> Bundle.predict).

Writes <dir>/<pk>-<model>.config.json, <pk>-<model>-w1v4.json, <pk>-<model>.receipt.json (never overwrites).
Refuses E_PS games, games outside WC8 (849799 only with --gate), a dirty tracked tree, trained_through >= game date.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402

# per-game config keys that may differ from the template; predictor keys that may differ (v4 / r)
CONFIG_REPLACED = {'game', 'predictions', 'predictor', 'wc8'}
PREDICTOR_REPLACED = {'game_date', 'as_of_note', 'config'}


def template(model):
    rel, sha = C.MODEL_SPEC[model]['template']
    if C.sha256_file(C.REPO / rel) != sha:
        raise SystemExit(f'{rel} differs from its pin (booth instance bytes)')
    return json.loads((C.REPO / rel).read_text())


def predictor_for(model, t):
    import build_pred as BP
    pin_rel, pin_sha = C.MODEL_SPEC[model]['pin']
    if model == 'bseq':
        blk = BP.predictor_block(BP.generic(), 'lit', family='B-SEQ')
        if blk['config'] != pin_rel:
            raise SystemExit('lit B-SEQ template points at another pin')
        blk['config_sha256'] = C.sha256_file(C.REPO / pin_rel)
    else:
        blk = copy.deepcopy(t['predictor'])
        if model == 'v4':
            blk['config'] = pin_rel     # booth bytes name the svc-gru-1121 absolute path; same file bytes (sha below)
    if blk['config_sha256'] != pin_sha or C.sha256_file(C.REPO / blk['config']) != pin_sha:
        raise SystemExit(f'{model}: predictor pin differs from {pin_sha[:8]}')
    if blk['module'] != C.MODEL_SPEC[model]['module'] or tuple(blk['classes']) != C.CLASSES or blk.get('w1_version') != 'v4':
        raise SystemExit(f'{model}: predictor module / classes / W1 version differ from the registration')
    return blk


def game_config(model, game, out_dir, commit):
    """Pure (apart from pinned reads): the per-game config of ``model`` for ``game``."""
    t = template(model)
    c = copy.deepcopy(t)
    pred = predictor_for(model, t)
    pred['game_date'] = game['official_date']
    pred['as_of_note'] = (f"{game['game_pk']} ({game['official_date']}) is after the predictor training end "
                          f"({pred['trained_through']}); WC8-CONFIRM-001 prediction run, no label read.")
    if not pred['trained_through'] < game['official_date']:
        raise SystemExit(f"as-of: {model} trained_through {pred['trained_through']} is not before {game['official_date']}")
    c['predictor'] = pred
    c['game'] = {'game_pk': game['game_pk'], 'final_feed': game['final_feed'], 'final_feed_sha256': game['final_feed_sha256']}
    c['predictions'] = {'path': str(pathlib.Path(out_dir) / C.pred_name(game['game_pk'], model)), 'sha256': 'PENDING'}
    c['wc8'] = {'registration': 'WC8-CONFIRM-001 (<private-jobs>/20261122-wc8/PREREG.md)', 'model': model,
                'template': C.MODEL_SPEC[model]['template'][0], 'template_sha256': C.MODEL_SPEC[model]['template'][1],
                'code_commit': commit, 'facts_sha256': C.FACTS_SHA, 'official_date': game['official_date']}
    check_config(model, c, t)
    return c


def check_config(model, c, t):
    import build_pred as BP
    other = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - CONFIG_REPLACED
    if other:
        raise SystemExit(f'per-game config differs from the booth instance outside the replaced blocks: {sorted(other)}')
    if model != 'bseq':
        dp = {k for k in set(c['predictor']) | set(t['predictor']) if c['predictor'].get(k) != t['predictor'].get(k)}
        if not dp <= PREDICTOR_REPLACED:
            raise SystemExit(f'predictor block differs from the booth instance: {sorted(dp - PREDICTOR_REPLACED)}')
    if c['generic_sha256'] != C.sha256_file(BP.GENERIC) or C.sha256_file(BP.GENERIC) != C.GENERIC_SHA:
        raise SystemExit('generic registration changed since the instance was made')
    probe = copy.deepcopy(c)
    probe.pop('wc8')
    probe['game'] = t['game']
    BP.check_instance_rule(probe)
    if c['status'] != 'REGISTERED' or c['execute'] is not True or c['evaluation'] is not False or c['promotion'] is not False:
        raise SystemExit('instance status / execute / evaluation / promotion differ')


def load_checks(c):
    """build_pred.load, predictor side, on the per-game config."""
    import build_pred as BP
    pin = C.REPO / c['predictor']['config']
    if C.sha256_file(pin) != c['predictor']['config_sha256']:
        raise SystemExit('predictor pin file differs from predictor.config_sha256')
    refit_sha = BP.sha256_file(BP.REFIT)
    checked = []
    for d, lit_pin in BP.refit_dirs(c, pin):   # mix: trained_through vs components, EXPL gate receipt vs pin + ext sha
        r = json.loads((pathlib.Path(d) / 'refit.json').read_text())
        gate_pin = json.loads((pathlib.Path(d) / 'bundle-pin.json').read_text())
        if r.get('pass') is not True or r.get('refit_config_sha256') != refit_sha or r.get('family') != lit_pin['family']:
            raise SystemExit(f'refit gates did not pass under the registered refit config for {d}')
        if gate_pin != lit_pin:
            raise SystemExit(f'pin differs from the gate-time bundle pin for {d}')
        checked.append(str(d))
    return {'refit_dirs_checked': checked, 'ext_expl_registered': BP.extension() is not None}


def dry_run(pk, model, out_dir, gate=False):
    """Config + load checks + feed byte sha only (the feed is not decompressed, nothing is written or predicted)."""
    game = C.game_entry(pk, allow_demo=gate)
    if C.sha256_file(game['final_feed']) != game['final_feed_sha256']:
        raise SystemExit(f'{pk}: final feed sha differs from registration-facts')
    c = game_config(model, game, pathlib.Path(out_dir).resolve(), C.head_commit())
    return {'game_pk': game['game_pk'], 'model': model, 'official_date': game['official_date'],
            'trained_through': c['predictor']['trained_through'], 'pin_sha256': c['predictor']['config_sha256'][:8],
            'feed_sha_ok': True, **load_checks(c)}


def run(pk, model, out_dir, gate=False):
    if model not in C.MODELS:
        raise SystemExit(f'unknown model {model}')
    commit = C.require_clean()
    game = C.game_entry(pk, allow_demo=gate)
    out_dir = pathlib.Path(out_dir).resolve()
    paths = {k: out_dir / f(game['game_pk'], model) for k, f in
             (('config', C.config_name), ('pred', C.pred_name), ('receipt', C.receipt_name))}
    for p in paths.values():
        if p.exists():
            raise SystemExit(f'{p} exists')
    final = C.read_feed_verified(game)   # sha + game + officialDate + Final (gameData only)
    del final
    c = game_config(model, game, out_dir, commit)
    checks = load_checks(c)
    cfg_sha = C.write_new(paths['config'], c)
    import predict_stage
    if C.sha256_file(predict_stage.__file__) != C.sha256_file(C.REPO / 'scripts/svc_r1/predict_stage.py'):
        raise SystemExit('predict_stage imported from another tree')
    t0 = time.perf_counter()
    res = predict_stage.run(paths['config'], paths['pred'])
    wall = time.perf_counter() - t0
    doc = json.loads(paths['pred'].read_text())
    ident = doc['predictor_identity']
    problems = []
    if doc['game_pk'] != game['game_pk'] or doc['final_feed_sha256'] != game['final_feed_sha256']:
        problems.append('predictions are for another game / feed')
    if tuple(doc['classes']) != C.CLASSES or ident['config_sha256'] != C.MODEL_SPEC[model]['pin'][1]:
        problems.append('classes / predictor pin in the predictions file differ')
    if len(doc['rows']) != game['is_pitch']:
        problems.append(f"rows {len(doc['rows'])} != registration-facts is_pitch {game['is_pitch']}")
    if ident.get('w1_version') != 'v4':
        problems.append('predictions not made with W1 v4')
    omit = {}
    for r in doc['rows']:
        if r['omit_reason']:
            omit[r['omit_reason']] = omit.get(r['omit_reason'], 0) + 1
    receipt = {'format': 'wc8-predict-receipt-v1', 'model': model, 'game_pk': game['game_pk'], 'official_date': game['official_date'],
               'trained_through': c['predictor']['trained_through'], 'as_of_ok': c['predictor']['trained_through'] < game['official_date'],
               'final_feed_sha256': game['final_feed_sha256'], 'facts_sha256': C.FACTS_SHA,
               'template': C.MODEL_SPEC[model]['template'][0], 'template_sha256': C.MODEL_SPEC[model]['template'][1],
               'predictor_pin': c['predictor']['config'], 'predictor_pin_sha256': c['predictor']['config_sha256'],
               'predictor_module': c['predictor']['module'], 'generic_sha256': c['generic_sha256'],
               'config': str(paths['config']), 'config_sha256': cfg_sha,
               'predict_stage_sha256': C.sha256_file(C.REPO / 'scripts/svc_r1/predict_stage.py'),
               'code_commit': commit, 'load_checks': checks,
               'predictions': str(paths['pred']), 'predictions_sha256': res['sha256'], 'rows': res['rows'],
               'predicted': res['predicted'], 'omit_reasons': omit, 'timing': res['timing'], 'wall_s': round(wall, 3),
               'python': sys.executable, 'labels_read': False, 'problems': problems, 'ok': not problems}
    C.write_new(paths['receipt'], receipt)
    if problems:
        raise SystemExit(f'post-checks failed (receipt written): {problems}')
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--game', type=int, required=True)
    ap.add_argument('--model', choices=C.MODELS, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--gate', action='store_true', help='allow 849799 (reproduction gate only)')
    ap.add_argument('--dry-run', action='store_true', help='config + load checks + feed sha only; no prediction, no file')
    a = ap.parse_args(argv)
    if a.dry_run:
        print(json.dumps(dry_run(a.game, a.model, a.out, gate=a.gate), indent=1))
        return
    r = run(a.game, a.model, a.out, gate=a.gate)
    print(json.dumps({k: r[k] for k in ('model', 'game_pk', 'rows', 'predicted', 'omit_reasons', 'predictions_sha256', 'wall_s', 'ok')},
                     indent=1))


if __name__ == '__main__':
    main()
