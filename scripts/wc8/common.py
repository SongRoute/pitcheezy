"""WC8-CONFIRM-001 shared constants and guards (registration: <private-jobs>/20261122-wc8/PREREG.md).

Games: the 8 wildcard games of registration-facts ``excluded_before_window`` (2026-09-29/30). 849799 is allowed only
for the reproduction gate (predictions, never labels). E_PS games are refused everywhere in scripts/wc8.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
for p in (REPO / 'src', REPO / 'scripts', REPO / 'scripts/svc_r1', REPO / 'scripts/svc_r4', REPO / 'scripts/svc_pred',
          REPO / 'scripts/svc_p9', REPO / 'scripts/wc8'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WC8 = (849841, 849842, 849843, 849845, 849846, 849848, 849849, 849851)
DEMO_GAME = 849799
SENSITIVITY_DROP = 849843          # PREREG §3/§6: 7-game sensitivity without 849843 (demo exposure)
FACTS = REPO / 'docs/pm/research/svc-p9/registration-facts.json'
FACTS_SHA = '92cc3d8cedd6473a9490b5080ec6e57da1be442e3d069e0bbd2f7a9ed604dcee'
GENERIC = REPO / 'configs/EXP-SVC-PRED-001.json'
GENERIC_SHA = '20250cb7b0071bb055962f399d37303393ae4a020ca7635e588e8c3896be40ae'
PINS = REPO / 'configs/EXP-SVC-PRED-001/pins'
V4_MIX_PIN = 'configs/EXP-SVC-PRED-001/pins/XGBT+XGRUA3p+XGRUA5b24_AM-w_+T-mix.json'
CLASSES = ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'OT')
MODELS = ('v4', 'bseq', 'r')
# template = the booth instance config whose bytes the per-game config starts from (sha pinned);
# pin = the predictor pin file (sha pinned); module = predictor module; booth = the 849799 booth predictions (gate)
MODEL_SPEC = {
    'v4': {'template': ('configs/EXP-SVC-PRED-001/XGBT+XGRUA3p+XGRUA5b24_AM-w_+T.json',
                        '735f80655cc55ff4db0eda643888cb085d08e349208dc40905308bdcd9222ea5'),
           'pin': (V4_MIX_PIN, '291dc9142d200466b9aa2537a3b9c288f8795e0791ee184c05e0a069d657699f'),
           'module': 'pitcheezy.svc_lit.mix',
           'booth_sha': '9d0dfcc4ae87c0cd7dfc07e9b064cfc41d628c9a264a1da79d21fbd3c62773ae'},
    'bseq': {'template': ('configs/EXP-SVC-PRED-001/XGBT+XGRUA3p+XGRUA5b24_AM-w_+T.json',
                          '735f80655cc55ff4db0eda643888cb085d08e349208dc40905308bdcd9222ea5'),
             'pin': ('configs/EXP-SVC-PRED-001/pins/B-SEQ.json', '735c12046aaa2ba01914cbc8872c2c96048756f1794039b48272e868402bc1e4'),
             'module': 'pitcheezy.svc_lit.bundle', 'booth_sha': None},
    'r': {'template': ('configs/EXP-SVC-PRED-001/R.json', '42d07b84057f91f8461d4298649870546f3cec81e141613bde50a94453d75496'),
          'pin': ('configs/EXP-SVC-P9-003.json', '6bef6ba88195e8811e3e6c0949c53c5df6d15b56e6f5181f41b3ce2fe484ec81'),
          'module': 'pitcheezy.svc_p9.predictor',
          'booth_sha': '64da605c8c65a3288afcf8ccce0f4626b372fb26d5a1534d3857f7ae03d079f2'},
}
# FS-001 fullseason baseline counts (2026 FIT 03-25..09-27), the file EXP-PREQ-001 pinned as fs_table
FREQ_TABLE = {'path': '${PITCHEEZY_RUNS}/MPS-20261119/EXP-MPS-FS-001/attempt-001/returned/MPSFS-tables-fullseason/fullseason-baseline-counts.json',
              'sha256': 'e7c2f2aaf287e9c439125136a71471655768f331d540f87eed7640d28302d73e', 'entry': 'fullseason',
              'alpha': 4.0, 'rows': 712603}
EVAL_REFERENCE = {'path': '${PITCHEEZY_REPO}/scripts/svc_p9/eval_prediction.py',
                  'sha256': '1c49e173c837ed0e4b571937a8c087eb35405a4c34a48e588002ced2165059da'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as f:
        f.write(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
    return sha256_file(path)


def head_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def tracked_dirty():
    out = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def require_clean():
    d = tracked_dirty()
    if d:
        raise SystemExit(f'tracked files modified in {REPO}; commit first: {d[:5]}')
    return head_commit()


def facts(path=FACTS):
    if sha256_file(path) != FACTS_SHA:
        raise SystemExit('registration-facts.json differs from its pin')
    return json.loads(pathlib.Path(path).read_text())


def game_entry(pk, *, allow_demo=False, f=None):
    """The registration-facts entry of a WC8 game (or 849799 when ``allow_demo``). E_PS and anything else refused."""
    f = f or facts()
    pk = int(pk)
    eps = {int(x) for x in f['E_PS']['game_pks']}
    if pk in eps:
        raise SystemExit(f'{pk} is an E_PS game: never opened by WC8-CONFIRM-001')
    if pk == DEMO_GAME and not allow_demo:
        raise SystemExit('849799 is allowed only in the reproduction gate')
    if pk != DEMO_GAME and pk not in WC8:
        raise SystemExit(f'{pk} is not one of the 8 wildcard games')
    hit = [g for g in f['games'] if int(g['game_pk']) == pk]
    if len(hit) != 1:
        raise SystemExit(f'{pk} is not in registration-facts games')
    g = hit[0]
    if pk in WC8:
        ex = {int(a): b for a, b in f['excluded_before_window']}
        if ex.get(pk) != g['official_date']:
            raise SystemExit(f'{pk}: excluded_before_window date differs from games entry')
    return {'game_pk': pk, 'official_date': g['official_date'], 'is_pitch': int(g['is_pitch']),
            'final_feed': str(pathlib.Path(f['feeds_dir']) / g['file']), 'final_feed_sha256': g['file_sha256']}


def read_feed_verified(game):
    import gzip
    if sha256_file(game['final_feed']) != game['final_feed_sha256']:
        raise SystemExit(f"{game['game_pk']}: final feed sha differs from registration-facts")
    final = json.loads(gzip.decompress(pathlib.Path(game['final_feed']).read_bytes()))
    gd = final['gameData']
    if int(gd['game']['pk']) != game['game_pk'] or gd['datetime']['officialDate'] != game['official_date']:
        raise SystemExit(f"{game['game_pk']}: feed game / officialDate differ from registration-facts")
    if gd['status'].get('abstractGameState') != 'Final':
        raise SystemExit(f"{game['game_pk']}: feed is not final")
    return final


def ab_order(final):
    """at_bat_number (atBatIndex + 1) of the n-th pitch in replay order. Reads no pitch type / outcome."""
    return [int(play['about']['atBatIndex']) + 1 for play in final['liveData']['plays']['allPlays']
            for event in play['playEvents'] if event.get('isPitch')]


def keys_match_order(keys, abs_, pk):
    """service_key 'pk:ab:pitch' of row i must name game pk and the at-bat of the i-th feed pitch."""
    if len(keys) != len(abs_):
        return False
    for k, ab in zip(keys, abs_):
        g, a, _ = k.split(':')
        if int(g) != int(pk) or int(a) != ab:
            return False
    return True


def pred_name(pk, model):
    return f'{int(pk)}-{model}-w1v4.json'


def receipt_name(pk, model):
    return f'{int(pk)}-{model}.receipt.json'


def config_name(pk, model):
    return f'{int(pk)}-{model}.config.json'


def baseline_name(pk):
    return f'{int(pk)}-baselines.json'


def expected_files(games=WC8):
    out = []
    for pk in games:
        for m in MODELS:
            out += [pred_name(pk, m), receipt_name(pk, m), config_name(pk, m)]
        out.append(baseline_name(pk))
    return sorted(out)


CODE_FILES = ('scripts/wc8/common.py', 'scripts/wc8/evalfns.py', 'scripts/wc8/score.py', 'scripts/wc8/pin.py',
              'scripts/wc8/predict.py', 'scripts/wc8/baselines.py')
