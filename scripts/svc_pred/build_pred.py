"""EXP-SVC-PRED-001: the zo-ring8 watch build (EXP-SVC-R4-003 c_all + location ring8) with pi_ref from "predictor X".

    python -B scripts/svc_pred/build_pred.py new-instance --name B-SEQ --predictor-kind lit --family B-SEQ        # DRAFT instance config
    python -B scripts/svc_pred/build_pred.py new-instance --name R+B-SEQ-AM --predictor-kind mix --mix <mix pin>  # combination
    python -B scripts/svc_pred/build_pred.py predict --config configs/EXP-SVC-PRED-001/<name>.json   # fresh process, CPU
    python -B scripts/svc_pred/build_pred.py build   --config ...   # watch under <root>/<name>/v4/c_all/pred-ring8 (never overwrites)
    python -B scripts/svc_pred/build_pred.py compare --config ... --out <json>   # paired with the zo-ring8 watch

An instance config is the R4-003 config bytes the zo-ring8 watch was built from with ONLY these blocks replaced: id
(EXP-SVC-PRED-001), instance, predictor, predictions, output_root, variants (c_all only), display; plus location_layout ring8
and the compare block. predict / build are ``precompute_r4.predict`` and the ``build_zo.build`` body. Display build only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
for p in (REPO / 'src', REPO / 'scripts', REPO / 'scripts/svc_r1', REPO / 'scripts/svc_r3', REPO / 'scripts/svc_r4'):
    if str(p) in sys.path:
        sys.path.remove(str(p))
    sys.path.insert(0, str(p))

GENERIC = REPO / 'configs/EXP-SVC-PRED-001.json'
BASE_R4 = REPO / 'configs/EXP-SVC-ZO-001/base-R4-003-at-3c8b3e51.json'
PINS_DIR = REPO / 'configs/EXP-SVC-PRED-001/pins'
REFIT = REPO / 'configs/EXP-SVC-PRED-001/refit.json'
EXT = REPO / 'configs/EXP-SVC-PRED-001/ext-expl.json'
# instance rule (generic base.instance_rule): only these keys may differ from the base R4-003 bytes
REPLACED = {'id', 'instance', 'status', 'execute', 'created_kst', 'author', 'purpose', 'single_change', 'predictor', 'predictions',
            'output_root', 'variants', 'display'}
DROPPED = {'review', 'base_r4', 'census_eps', 'compare', 'compare_ab', 'adoption_basis'}
ADDED = {'location_layout', 'compare', 'generic_sha256', 'review', 'decision'}


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def generic():
    g = json.loads(GENERIC.read_text())
    if g.get('id') != 'EXP-SVC-PRED-001' or g.get('part') != 'generic':
        raise SystemExit('generic registration missing')
    if sha256_file(BASE_R4) != g['base']['r4_config_sha256']:
        raise SystemExit('base R4-003 copy differs from its pin')
    return g


def predictor_block(g, kind, family=None, mix=None, svc_candidate=None):
    t = g['predictor_templates']
    if kind == 'lit':
        if family not in t['lit']['families']:
            raise SystemExit(f'unknown family {family}')
        blk = copy.deepcopy(t['lit']['block']); blk.update(t['lit']['families'][family])
        blk.update(label=f'LIT_{family.replace("-", "")}', candidate=family, config=f'configs/EXP-SVC-PRED-001/pins/{family}.json',
                   config_sha256='PENDING')
    elif kind == 'mix':
        m = json.loads(pathlib.Path(mix).read_text())
        blk = copy.deepcopy(t['mix']['block'])
        through = mix_trained_through(g, m)
        blk.update(label=f"MIX_{m['name']}", candidate=m['name'], config=str(pathlib.Path(mix)), config_sha256=sha256_file(mix),
                   trained_through=through, trained_window=f'2026 regular season, components end <= {through}',
                   description=f"{m['method']} of {[c['name'] for c in m['components']]} weights {m['weights']} T {m.get('temperature')}")
    elif kind == 'svc_p9':
        blk = copy.deepcopy(t['svc_p9'][svc_candidate])
    elif kind == 'expl':
        e = extension()
        if e is None:
            raise SystemExit('single X arm needs a REGISTERED ext-expl.json')
        pin_path = PINS_DIR / f'{family}.json'
        pin = json.loads(pin_path.read_text()); tpl = e['templates'][pin['data']]
        blk = copy.deepcopy(t['mix']['block'])
        blk.update(module='pitcheezy.svc_expl.bundle', label=f"EXPL_{family.replace('-', '')}", candidate=family,
                   config=str(pin_path.relative_to(REPO)), config_sha256=sha256_file(pin_path), trained_through=tpl['trained_through'],
                   trained_window=tpl['trained_window'], description=f"EXPL-1121 arm {family} ({pin['kind']}, data {pin['data']}), seed mean + DEV tau {pin['tau']}",
                   calibration='single temperature on the renormalised seed mean (EXPL DEV tau)',
                   as_of_note=f"849799 (2026-10-29) is after the training end ({tpl['trained_through']}); no 2026 postseason game in its training data. Inputs = pre-pitch information as R.")
    else:
        raise SystemExit(kind)
    return blk


def check_expl_pin(path):
    """Load-time: an svc_expl pin with data plus must carry the ext-expl hist pins (chain re-verified at gate time)."""
    pin = json.loads(pathlib.Path(path).read_text())
    if pin.get('data') == 'plus':
        e = extension()
        if (pin.get('plus_chain') or {}).get('hist_sha256') != e['templates']['plus']['hist_sha256']:
            raise SystemExit(f"{pin['family']}: plus pin without the registered hist chain")
    elif pin.get('data') != 'base':
        raise SystemExit(f"{pin['family']}: data {pin.get('data')} not registered")


def extension():
    """ext-expl.json when REGISTERED (EXPL arms), else None. Its vendored code must be byte-pinned."""
    if not EXT.exists():
        return None
    e = json.loads(EXT.read_text())
    if e.get('status') != 'REGISTERED':
        return None
    for rel, want in e['vendored'].items():
        if rel != 'source' and sha256_file(REPO / rel) != want:
            raise SystemExit(f'{rel} differs from its ext-expl pin')
    return e


def allowed_modules(gen):
    e = extension()
    return list(gen['allowed_modules']) + (e.get('single_modules_add', []) if e else [])


def mix_trained_through(g, m):
    """Latest training end over the mix components, from the registered templates (never a hand-written value)."""
    t = g['predictor_templates']; ends = []
    for c in m['components']:
        if c['module'] == 'pitcheezy.svc_lit.bundle':
            ends.append(t['lit']['block']['trained_through'])
        elif c['module'] == 'pitcheezy.svc_p9.predictor' and c.get('candidate') in t['svc_p9']:
            ends.append(t['svc_p9'][c['candidate']]['trained_through'])
        elif c['module'] == 'pitcheezy.svc_expl.bundle' and extension() is not None:
            p = pathlib.Path(c['config']); pin = json.loads((p if p.is_absolute() else REPO / p).read_text())
            if pin.get('data') not in extension()['templates']:
                raise SystemExit(f"mix component {c['name']}: data {pin.get('data')} not registered")
            ends.append(extension()['templates'][pin['data']]['trained_through'])
        else:
            raise SystemExit(f"mix component {c['name']}: module {c['module']} / candidate {c.get('candidate')} not allowed (no nested mix)")
    return max(ends)


def check_instance_rule(g):
    base = json.loads(BASE_R4.read_text()); gen = generic()
    extra = set(g) - (set(base) | REPLACED | ADDED)
    if extra:
        raise SystemExit(f'instance has keys outside the rule: {sorted(extra)}')
    for k in set(base) - REPLACED - DROPPED:
        if g.get(k) != base[k]:
            raise SystemExit(f'instance key {k} differs from the base R4-003 config')
    if g.get('compare') != gen['compare'] or g.get('location_layout') != 'ring8':
        raise SystemExit('instance compare / location_layout differ from the generic registration')
    if g['predictor']['module'] not in allowed_modules(gen):
        raise SystemExit('predictor module not allowed')


def new_instance(name, kind, family=None, mix=None, svc_candidate=None, out_dir=None):
    g = generic()
    base = json.loads(BASE_R4.read_text())
    root = f"{g['output_base']}/{name}/v4"
    note = g['display']['note_r4_insert'].get(kind)
    if kind in ('expl', 'mix') and extension() is not None:
        plus = False; plus_names = []
        if kind == 'expl':
            plus = json.loads((PINS_DIR / f'{family}.json').read_text()).get('data') == 'plus'
        else:
            for c in json.loads(pathlib.Path(mix).read_text())['components']:
                if c['module'] == 'pitcheezy.svc_expl.bundle':
                    cp = pathlib.Path(c['config'])
                    if json.loads((cp if cp.is_absolute() else REPO / cp).read_text()).get('data') == 'plus':
                        plus = True; plus_names.append(c['name'])
        notes = extension()['display']['note_r4_insert']
        note = notes['expl_plus' if plus else 'expl_base'] if kind == 'expl' else (notes['mix_plus'] if plus else note)
    blk = predictor_block(g, kind, family, mix, svc_candidate)
    if kind == 'mix' and extension() is not None and plus:
        tp = extension()['templates']['plus']
        pn = ', '.join(plus_names)
        blk.update(trained_window=f"{tp['trained_window']} ({pn}); other components 2026 regular season only",
                   as_of_note=f'every component is trained before 2026-10-29 (2023~2025 regular-season hist plus 2026 for {pn}, 2026 regular season only for the others); no 2026 postseason game is in any training data')
    inst = copy.deepcopy(base)
    for k in ('review', 'base_r4', 'census_eps', 'compare', 'compare_ab', 'adoption_basis'):
        inst.pop(k, None)
    inst.update(id='EXP-SVC-PRED-001', instance=name, status='DRAFT', execute=False, created_kst=g['created_kst'], author=g['author'],
                purpose=g['purpose'], single_change=g['single_change'].replace('{X}', name),
                predictor=blk, output_root=root,
                predictions={'path': f'{root}/predictions/849799-{name}-w1v4.json', 'sha256': 'PENDING'},
                variants={'c_all': {'out_name': 'c_all/pred-ring8', 'r3_variant': 'c_all'}},
                display={'policy_name_r4': g['display']['policy_name_r4'].replace('{X}', name),
                         'note_r4_insert': note, 'note_rule': base['display']['note_rule']},
                location_layout='ring8', compare=g['compare'], generic_sha256=sha256_file(GENERIC))
    out = pathlib.Path(out_dir or REPO / 'configs/EXP-SVC-PRED-001') / f'{name}.json'
    if out.exists():
        raise SystemExit(f'{out} exists')
    out.write_text(json.dumps(inst, indent=1, ensure_ascii=False) + '\n')
    return {'config': str(out), 'sha256': sha256_file(out)}


def load(path):
    import precompute_r4 as R4
    g, g_sha = R4.load_r4(path)
    if g['id'] != 'EXP-SVC-PRED-001' or g['status'] != 'REGISTERED' or g['execute'] is not True:
        raise SystemExit('EXP-SVC-PRED-001 instance must be REGISTERED with execute true')
    if g['generic_sha256'] != sha256_file(GENERIC):
        raise SystemExit('generic registration changed since the instance was made')
    check_instance_rule(g)
    pin = REPO / g['predictor']['config'] if not pathlib.Path(g['predictor']['config']).is_absolute() else pathlib.Path(g['predictor']['config'])
    if R4.sha256_file(pin) != g['predictor']['config_sha256']:
        raise SystemExit('predictor pin file differs from predictor.config_sha256')
    if g['predictor']['module'] == 'pitcheezy.svc_expl.bundle':
        check_expl_pin(pin)
        rc = REPO / 'configs/EXP-SVC-PRED-001/gates' / f"{g['predictor']['candidate']}.json"
        r = json.loads(rc.read_text()) if rc.exists() else {}
        if r.get('passed') is not True or r.get('pin_sha256') != g['predictor']['config_sha256'] or r.get('ext_sha256') != sha256_file(EXT):
            raise SystemExit(f'single X gate receipt missing / failed / for another pin: {rc}')
    refit_sha = sha256_file(REFIT)
    for d, lit_pin in refit_dirs(g, pin):
        r = json.loads((pathlib.Path(d) / 'refit.json').read_text())
        gate_pin = json.loads((pathlib.Path(d) / 'bundle-pin.json').read_text())
        if r.get('pass') is not True or r.get('refit_config_sha256') != refit_sha or r.get('family') != lit_pin['family']:
            raise SystemExit(f'refit gates did not pass under the registered refit config for {d}')
        if gate_pin != lit_pin:
            raise SystemExit(f'pin differs from the gate-time bundle pin for {d}')
    return g, g_sha, R4


def refit_dirs(g, pin):
    """Refit directories behind this predictor (svc_lit pins directly, or inside a mix)."""
    doc = json.loads(pathlib.Path(pin).read_text())
    if doc.get('format') == 'svc-lit-bundle-v1':
        return [(doc['dir'], doc)]
    if doc.get('format') == 'svc-mix-bundle-v1':
        if g['predictor']['trained_through'] != mix_trained_through(generic(), doc):
            raise SystemExit('mix trained_through differs from its components')
        out = []
        for c in doc['components']:
            if c['module'] == 'pitcheezy.svc_lit.bundle':
                p = pathlib.Path(c['config']); d = json.loads((p if p.is_absolute() else REPO / p).read_text()); out.append((d['dir'], d))
            elif c['module'] == 'pitcheezy.svc_expl.bundle':
                if extension() is None:
                    raise SystemExit('svc_expl component needs a REGISTERED ext-expl.json')
                cp = pathlib.Path(c['config']); check_expl_pin(cp if cp.is_absolute() else REPO / cp)
                rc = REPO / 'configs/EXP-SVC-PRED-001/gates' / f"{doc['name']}.json"
                r = json.loads(rc.read_text()) if rc.exists() else {}
                if r.get('passed') is not True or r.get('mix_pin_sha256') != g['predictor']['config_sha256'] or r.get('ext_sha256') != sha256_file(EXT):
                    raise SystemExit(f'EXPL mix gate receipt missing / failed / for another pin: {rc}')
            elif c['module'] != 'pitcheezy.svc_p9.predictor':
                raise SystemExit('nested mix is not allowed')
        return out
    return []


def predict(path):
    g, _, R4 = load(path)
    out = pathlib.Path(g['predictions']['path'])
    if out.exists():
        raise SystemExit(f'{out} exists')
    return R4.predict(out, path)


def build(path, copy_dir=None):
    g, g_sha, R4 = load(path)
    import precompute_r3 as P3
    from pitcheezy.svc_r1 import adapter as A
    variant = 'c_all'
    R4.check_base(g)
    r3_cfg, r3_sha, _ = P3.load_config(REPO / g['base_r3']['config'], variant)
    if r3_sha != g['base_r3']['config_sha256']:
        raise SystemExit('R3 config sha differs')
    if g['predictions']['sha256'] in (None, '', 'PENDING'):
        raise SystemExit('pin predictions.sha256 first')
    doc, _ = A.load_predictions(g['predictions']['path'], g['predictions']['sha256'])
    R4.check_predictions_identity(doc, g)
    cfg = R4.overlay_cfg(r3_cfg, g, R4.config_rel(path), variant)
    extra = R4.extra_policy(g, g_sha, variant)
    extra['pred'] = {'id': g['id'], 'instance': g['instance'], 'config_sha256': g_sha, 'single_change': g['single_change'],
                     'location_layout': g['location_layout'], 'predictor_pin_sha256': g['predictor']['config_sha256'],
                     'base_watch_sha256': g['compare']['base_watch_sha256']}
    result = P3.run_cfg('r3r1', cfg, r3_sha, g['variants'][variant]['out_name'], extra_policy=extra,
                        location_layout=g['location_layout'])
    copy_ = pathlib.Path(copy_dir or generic()['copy_dir']).expanduser() / g['instance'] / '849799.json'
    copy_.parent.mkdir(parents=True, exist_ok=True)
    if copy_.exists():
        raise SystemExit(f'{copy_} exists; not overwritten')
    shutil.copyfile(result['dataset'], copy_)
    result['copy'] = str(copy_)
    return result


TOP_ALLOWED = {'policy', 'git', 'timing', 'paths', 'generated_utc', 'decisions'}   # every other top-level key must be identical
DECISION_ALLOWED = {'pre', 'r1'}                                                  # every other decision key must be identical
REC_ALLOWED = {'candidates', 'candidate_law', 'reference_law', 'predictor_reference_law'}
POLICY_ALLOWED = {'identity_sha256', 'name', 'note', 'pred', 'zo', 'r1', 'r4'}            # every other policy key must be identical


def _cands(rec):
    return [(c['pitch_type'], c['zone_id']) for c in rec['candidates']] if rec else None


def law_ok(law):
    if not isinstance(law, dict) or not law:
        return False
    v = list(law.values())
    return all(isinstance(x, (int, float)) and x >= 0 for x in v) and abs(sum(v) - 1.0) < 1e-6


def compare(base, new):
    """Metrics fixed in configs/EXP-SVC-PRED-001.json compare.metrics_fixed_before_build."""
    bad, rank1_type, rank1_cell, any_change, presence = [], [], [], [], {'both': 0, 'base_only': 0, 'new_only': 0, 'neither': 0}
    io = {'inside_to_outside': 0, 'outside_to_inside': 0, 'outside_to_outside': 0, 'inside_to_inside': 0}
    law_bad = []
    for key in sorted((set(base) | set(new)) - TOP_ALLOWED):
        if base.get(key) != new.get(key):
            bad.append(f'top:{key}')
    for key in sorted((set(base['policy']) | set(new['policy'])) - POLICY_ALLOWED):
        if base['policy'].get(key) != new['policy'].get(key):
            bad.append(f'policy:{key}')
    if len(base['decisions']) != len(new['decisions']):
        bad.append('decision count')
    for b, n in zip(base['decisions'], new['decisions']):
        i = b['index']
        for k in sorted((set(b) | set(n)) - DECISION_ALLOWED):
            if b.get(k) != n.get(k):
                bad.append(f'{i}:{k}')
        for k in sorted((set(b['pre']) | set(n['pre'])) - {'recommendation'}):
            if b['pre'].get(k) != n['pre'].get(k):
                bad.append(f'{i}:pre.{k}')
        brr, nrr = b['pre'].get('recommendation') or {}, n['pre'].get('recommendation') or {}
        for k in sorted((set(brr) | set(nrr)) - REC_ALLOWED):
            if brr.get(k) != nrr.get(k):
                bad.append(f'{i}:recommendation.{k}')
        br, nr = b['pre'].get('recommendation'), n['pre'].get('recommendation')
        presence['both' if br and nr else 'base_only' if br else 'new_only' if nr else 'neither'] += 1
        if nr is not None and not law_ok(nr.get('predictor_reference_law')) and n.get('r1', {}).get('pi_ref_source') == 'predictor':
            law_bad.append(i)
        if not (br and nr):
            continue
        b1, n1 = br['candidates'][0], nr['candidates'][0]
        if b1['pitch_type'] != n1['pitch_type']:
            rank1_type.append({'index': i, 'from': b1['pitch_type'], 'to': n1['pitch_type']})
        if b1['zone_id'] != n1['zone_id']:
            rank1_cell.append({'index': i, 'from': b1['zone_id'], 'to': n1['zone_id'], 'type_changed': b1['pitch_type'] != n1['pitch_type']})
            key = ('outside' if b1['zone_id'].startswith('out_') else 'inside') + '_to_' + ('outside' if n1['zone_id'].startswith('out_') else 'inside')
            io[key] += 1
        if _cands(br) != _cands(nr):
            any_change.append(i)
    src = {}
    for d in new['decisions']:
        s = (d.get('r1') or {}).get('pi_ref_source'); src[s] = src.get(s, 0) + 1
    omit = {}
    for d in new['decisions']:
        r = (d.get('r1') or {}).get('omit_reason')
        if r:
            omit[r] = omit.get(r, 0) + 1
    return {'non_recommendation_differences': bad, 'recommendation_presence': presence,
            'rank1_pitch_type_changed': len(rank1_type), 'rank1_pitch_type_changes': rank1_type,
            'rank1_cell_changed': len(rank1_cell), 'rank1_cell_inside_outside': io, 'rank1_cell_changes': rank1_cell,
            'candidate_set_changed': len(any_change), 'candidate_set_changed_index': any_change,
            'pi_ref_sources': src, 'predictor_omit_reasons': omit, 'predictor_reference_law_invalid_rows': law_bad,
            'base_identity': base['policy']['identity_sha256'], 'new_identity': new['policy']['identity_sha256']}


def check(path):
    """Step 4 of SWAP-CHECKLIST: predictions file + watch format, omissions and as-of (no label, no model)."""
    import math
    g = json.loads(pathlib.Path(path).read_text())
    gen = generic()
    bad = []
    pred = json.loads(pathlib.Path(g['predictions']['path']).read_text())
    if sha256_file(g['predictions']['path']) != g['predictions']['sha256']:
        bad.append('predictions sha differs from the instance pin')
    base_pred = json.loads(pathlib.Path(json.loads(BASE_R4.read_text())['predictions']['path']).read_text())
    if [(r['n'], r['service_key'], r['w1_key']) for r in pred['rows']] != [(r['n'], r['service_key'], r['w1_key']) for r in base_pred['rows']]:
        bad.append('prediction keys differ from the R4-003 R file')
    omit = {(r['n'], r['omit_reason']) for r in pred['rows'] if r['omit_reason']}
    if omit != {(r['n'], r['omit_reason']) for r in base_pred['rows'] if r['omit_reason']}:
        bad.append(f'omitted rows differ from R: {sorted(omit)}')
    for r in pred['rows']:
        q = r['q']
        if q is None:
            continue
        if len(q) != 9 or any((not math.isfinite(v)) or v < 0 for v in q) or abs(sum(q) - 1) > 1e-9:
            bad.append(f"n {r['n']}: q format")
    if g['predictor']['trained_through'] >= g['predictor']['game_date']:
        bad.append('as-of: trained_through >= game_date')
    w_path = pathlib.Path(g['output_root']) / g['variants']['c_all']['out_name'] / 'watch/849799.json'
    w = json.loads(w_path.read_text())
    pol = w['policy']
    if w['location'].get('zone_layout') != g['location_layout']:
        bad.append('watch location layout')
    if pol.get('pred', {}).get('instance') != g['instance'] or pol.get('r4', {}).get('trained_through') != g['predictor']['trained_through']:
        bad.append('watch policy pred / r4 block')
    if len(w['decisions']) != len(pred['rows']):
        bad.append('decision count != prediction rows')
    return {'instance': g['instance'], 'ok': not bad, 'problems': bad, 'rows': len(pred['rows']),
            'predicted': sum(r['q'] is not None for r in pred['rows']), 'omitted': sorted(omit), 'watch': str(w_path),
            'watch_sha256': sha256_file(w_path), 'generic_sha256_matches': g.get('generic_sha256') == sha256_file(GENERIC),
            'location_layout': w['location'].get('zone_layout')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('kind', choices=('new-instance', 'predict', 'build', 'check', 'compare'))
    ap.add_argument('--config', type=pathlib.Path)
    ap.add_argument('--name'); ap.add_argument('--predictor-kind', dest='pkind', choices=('lit', 'mix', 'svc_p9', 'expl'))
    ap.add_argument('--family'); ap.add_argument('--mix'); ap.add_argument('--svc-candidate', choices=('R', 'C'))
    ap.add_argument('--out', type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.kind == 'new-instance':
        r = new_instance(a.name, a.pkind, a.family, a.mix, a.svc_candidate)
    elif a.kind == 'predict':
        r = predict(a.config)
    elif a.kind == 'build':
        r = build(a.config)
    elif a.kind == 'check':
        r = check(a.config)
    else:
        g = json.loads(a.config.read_text())
        bp = pathlib.Path(g['compare']['base_watch'])
        if sha256_file(bp) != g['compare']['base_watch_sha256']:
            raise SystemExit('base watch sha differs')
        np_ = pathlib.Path(g['output_root']) / g['variants']['c_all']['out_name'] / 'watch/849799.json'
        r = compare(json.loads(bp.read_text()), json.loads(np_.read_text()))
        r['instance'] = g['instance']
        if a.out:
            if a.out.exists():
                raise SystemExit(f'{a.out} exists')
            a.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(r, indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
