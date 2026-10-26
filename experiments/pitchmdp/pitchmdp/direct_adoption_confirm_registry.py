"""Torch-free registry of the DRAFT adoption-seed confirmation family (EXP-P48-001 DRAFT).

Pre-registration: docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md, written 2026-10-26 before any
smoke, fit or score of this family.

Question.  The 10/4-10/6 CUDA direct-head exploration (EXP-P34-001 seeds 24-28, EXP-P35-001 seeds 29-33, repeatedly
exposed DEV) read tabm_residual_counts (TabM-mini k=8 head on the periodic-residual backbone, counts workload) below
the same-family flatten control.  Those are UNBLENDED head-vs-control contrasts; they were never compared with the
research baseline G15 mix (configs/G15-MIX-RESEARCH-FROZEN-v1.json, D145: the flatten direct head of seed s mixed 1/2
with the frozen G0-global member of seed s, then the June frequency blend; results/EXP-P16-001.json).  This family
refits the three arms on the ADOPTION seeds {0, 1, 2, 3, 4}, so that each head seed s can be mixed with the frozen
G0-global member of the same seed s:

  Stage A (CUDA, newsrv; this registry + run_cuda_direct.py + cuda_direct_queue.py): 3 arms x 5 seeds = 15 fits, the
      cuda_direct_v1 budget with the epoch cap 60 (as P34-P37), the pinned export (manifest 49097109...) and the
      unchanged calibration chain; score_cuda_direct.py reports the fixed unblended contrasts below vs the
      same-family CUDA control (descriptive only).
  Stage B (Mac CPU, scripts/score_cuda_mix_vs_g15.py): every head mixed with the frozen G0-global member of the same
      seed under two pre-registered rules (MIX_FIXED = the G15 weight 1/2; MIX_FIT = one weight per seed chosen on
      the June blend split only over FIT_GRID, the ensemble weight chosen the same way on the seed mean), then the
      June frequency blend, compared with G15 on Cpanel DEV after the blend and on whole-MLB DEV blended under the
      matrix N rule (final stage, 4/5 seeds).  Stage B is the PRIMARY comparison and carries exactly ONE
      confirmatory test: the pre-specified primary candidate tabm_residual_counts MIX_FIT, confirmed by
      intersection-union (each split passes its own one-sided test at alpha .05; no Holm for a single primary).  The
      other three candidate rows are secondary / exploratory (per-split Holm across those three as a descriptive
      adjustment; a combined both-split outcome is labelled exploratory, never confirmed).  STAGE_B_RULES is the
      machine-checked part of config comparison.stage_b (check_stage_b; the runner's config_check calls it too).
      Every frozen Stage-B input is anchored to a pinned chain (STAGE_B_RULES.inputs.anchors): the G0 bundle
      listing, the G15 listing, or the EXP-P11-001 analysis manifest pinned by the G0 bundle (whole-MLB G0 members).
      Pipeline comparison, disclosed: the CUDA heads train with the epoch cap 60, the G15 flatten heads were fitted
      with the cap 30 on MPS; control at MIX_FIXED (same architecture, same mix) measures that pipeline difference.

Arms: the EXACT earlier registered config entries (same flags, same frozen code paths, no new model code)
  control                   MatrixModel flatten_mlp 128, no workload (direct_tabm_residual_registry.BASELINES)
  periodic_residual_counts  EXP-P32/P34/P35 entry (embed_residual flag; direct_tabm_residual_registry.BASELINES)
  tabm_residual_counts      EXP-P34-001 arm as registered in EXP-P35-001 (tabm_residual flag;
                            direct_ple_registry.BASELINES)
The arms keep their embed_residual / tabm_residual flags, so the family is recognised by the top-level marker MARKER
("adoption_confirm": true) and its predicate is tested FIRST everywhere (config_check, training_budget,
uses_epoch_cap, the scorer), before the pitcher-ID / embed-type / robust-clip / PLE / TabM-residual predicates;
without the marker the P34 family check would claim (and refuse) this config.

Seeds and exposure (disclosed, not hidden): SEEDS = [0, 1, 2, 3, 4] are the adoption seeds of the G15 and G0-global
members, not fresh seeds.  Seeds 0-2 were already MODEL RNG seeds of other arms in P24/P25/P27/P29 and seeds 3-4 of
EXP-P30-001 arms (RNG reuse across families on the same export: not new data, not a new evaluation population).  The
family was designed AFTER reading P34/P35 scores on the repeatedly exposed Cpanel DEV / whole-MLB DEV
(post-selection), so even a Stage-B pass is a repeated-DEV result; an honest confirmation is prospective (2027 or an
RS-02-style arm).  No 2026 rows.  Nothing here changes G15, G0-global, the policy identifier or the service.
"""
from __future__ import annotations

import copy
import json

from pitchmdp import direct_ple_registry as ple
from pitchmdp import direct_tabm_residual_registry as trr

MARKER = 'adoption_confirm'
CONTRAST_KEY = 'adoption_confirm_contrasts'
STAGE_B_KEY = 'stage_b'
# Top-level markers of the other marker families (== direct_embed_type_registry.MARKER and
# direct_pitcher_id_registry.MARKER; tested): a config carries at most one family marker.
OTHER_MARKERS = ('embed_type_family', 'pitcher_id_family')
SEEDS = [0, 1, 2, 3, 4]
# Earlier uses of these seed values (disclosure only; reuse is the point of this family, so no disjointness check).
EARLIER_USES = {'P24/P25/P27/P29 CUDA arms (MODEL RNG)': [0, 1, 2], 'EXP-P30-001 CUDA arms (MODEL RNG)': [3, 4],
                'EXP-P15-001/EXP-P16-001 MPS flatten direct heads (the G15 members)': [0, 1, 2, 3, 4],
                'EXP-P4-001/EXP-P10-001 G0-global members': [0, 1, 2, 3, 4]}
LEVEL = trr.LEVEL
WORKLOAD = list(trr.WORKLOAD)
EPOCH_CAP = trr.EPOCH_CAP
TRAINING = trr.TRAINING  # cuda_direct_v1 budget with the epoch cap 60 (== P34/P35; the same object)

# Exact earlier registered config entries (old flags, old behavior, old code paths).
BASELINES = {'control': trr.BASELINES['control'],
             'periodic_residual_counts': trr.BASELINES['periodic_residual_counts'],
             'tabm_residual_counts': ple.BASELINES['tabm_residual_counts']}
ARM_ORDER = ('control', 'periodic_residual_counts', 'tabm_residual_counts')
FITS = len(ARM_ORDER) * len(SEEDS)

# The five A5000 UUIDs Song allowed on 2026-10-11 (all five GPUs); a config may allow any non-empty subset.
GPU_UUIDS = ('GPU-edb6bc12-c9aa-d79b-d3e7-9bc28118e4d4', 'GPU-0cc63040-5522-7fea-20d8-93b9196ce019',
             'GPU-18e648d0-b68e-9321-9c77-69da241aad85', 'GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9',
             'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99')
MAX_PARALLEL = 5
DEVICE_KEYS = ('kind', 'allowed_gpu_uuids', 'max_parallel', 'cpu_fallback')

# Stage-A contrasts: the direct_tabm_residual_registry pairs of these three arms (same isolates / held vocabulary).
CONTRAST_NAMES = ('tabm_residual_counts_vs_control', 'periodic_residual_counts_vs_control',
                  'tabm_residual_counts_vs_periodic_residual_counts')
CONTRASTS = [copy.deepcopy(next(c for c in trr.CONTRASTS if c['name'] == name)) for name in CONTRAST_NAMES]
FACTOR_NAMES, FACTORS, ISOLATES = trr.FACTOR_NAMES, trr.FACTORS, trr.ISOLATES

# ------------------------------------------------------------------ Stage B (Mac) rules, machine-checked

MIX_FIXED, MIX_FIT = 'MIX_FIXED', 'MIX_FIT'
FIT_GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
REFERENCE_ARM = 'G15-direct-g0-mix'  # == run_ml_direct_head.MIX (tested)
G0_CELL = 'G0-global'  # == run_ml_direct_head.CELL (tested)
STAGE_B_RULES = {
    'reference': {
        'arm': REFERENCE_ARM, 'rule': 'mixed(flatten direct head seed s, frozen G0-global seed s, 0.5) -> June blend',
        'config': 'configs/G15-MIX-RESEARCH-FROZEN-v1.json',
        'config_sha256': 'b4c19988722c25d994b6f14c9b87b3ff3a177fb95702fe4c61a518fbbb201e6d',
        'g0_bundle': 'configs/G0-RESEARCH-FROZEN-v1.json',
        'g0_bundle_sha256': 'c839dbf710c02ca5a1baafcc168e6ca3eaa6d80e269a99f71f08d7d537ec8062',
        'recorded_results': 'results/EXP-P16-001.json',
        'recorded_results_sha256': '9d372da26b39fca07d6963a6861044721b31cf42209c92d5edb66f7b8a778597'},
    'inputs': {  # relative to --runs-root (its directory name must be runs_root_name); every file sha-checked
        'runs_root_name': 'ML-MATRIX-20260924',
        'cpanel_baseline': 'EXP-P4-001/baseline_predictions.npz',
        'mlb_baseline': 'EXP-P11-001/baseline_predictions.npz',
        'g0_cpanel': {'0': 'EXP-P4-001/members/G0-global/seed0', '1': 'EXP-P4-001/members/G0-global/seed1',
                      '2': 'EXP-P4-001/members/G0-global/seed2', '3': 'EXP-P10-001/members/G0-global/seed3',
                      '4': 'EXP-P10-001/members/G0-global/seed4'},
        'g0_mlb': {str(s): f'EXP-P11-001/members/G0-global/seed{s}' for s in SEEDS},
        'flatten': {'0': 'EXP-P15-001/0', '1': 'EXP-P15-001/1', '2': 'EXP-P15-001/2', '3': 'EXP-P16-001/3',
                    '4': 'EXP-P16-001/4'},
        # The whole-MLB G0 members are not listed in the G0 bundle itself; the bundle pins the EXP-P11-001 analysis
        # manifest, whose `inputs` record every P11 member state and archive (scripts/score_ml_g0_whole.py).
        'p11_manifest': {'g0_bundle_key': 'p11_manifest', 'path': 'EXP-P11-001/analysis/manifest.json'},
        # Pinned chain of every loaded frozen file: source map and the files that must be listed there.  Any other
        # pinned map that lists the same file must agree with it.
        'anchors': {'cpanel_baseline': {'source': 'g0_bundle', 'files': ['archive']},
                    'mlb_baseline': {'source': 'g0_bundle', 'files': ['archive']},
                    'g0_cpanel': {'source': 'g0_bundle', 'files': ['state']},
                    'g0_mlb': {'source': 'p11_manifest', 'files': ['state', 'archive']},
                    'flatten': {'source': 'g15', 'files': ['state']}}},
    'candidate_arms': ['periodic_residual_counts', 'tabm_residual_counts'],
    'descriptive_arms': ['control'],
    'mix_rules': {MIX_FIXED: {'weight': 0.5},
                  MIX_FIT: {'grid': FIT_GRID, 'split': 'blend', 'probabilities': 'calibrated',
                            'objective': 'pitch_weighted_nll', 'tie_break': 'smallest_weight',
                            'ensemble': 'weight chosen the same way on the seed-mean head and seed-mean G0'}},
    'primary': {'arm': 'tabm_residual_counts', 'mix_rule': MIX_FIT},
    'splits': ['cpanel_dev_after_blend', 'whole_mlb_dev_blended'],
    'N': {'delta_nll_max': -0.003, 'nll_ci95_upper_max': 0, 'brier_ci95_upper_max': 0.001,
          'required_negative_seeds': 4, 'stage': 'final'},
    'alpha': 0.05,  # one-sided, == the fixed threshold inside matrix_metrics.prediction_decision
    'bootstrap': {'draws': 10000, 'seed': 20260924, 'unit': 'whole_game'},
    'multiplicity': {
        'primary': 'single_prespecified_test_intersection_union_both_splits_one_sided_alpha_no_adjustment',
        'secondary': 'holm_per_split_across_the_three_secondary_candidate_rows_descriptive_adjustment',
        'combined_secondary_outcome': 'exploratory_never_confirmed',
        'descriptive': 'control_rows_and_unblended_head_rows_no_N_decision'},
    'confirmation': 'primary_only_predictive_improvement_on_both_splits',
    'reproduction_gate': {
        'tolerance': 1e-9,
        'fields': {'g15_cpanel_dev_after_blend_nll': ['cpanel_dev', 'reports', REFERENCE_ARM, 'primary', 'log_loss'],
                   'g15_whole_mlb_dev_blended_nll': ['mlb_dev_blended', 'comparison', REFERENCE_ARM, 'candidate',
                                                     'log_loss'],
                   'g0_cpanel_dev_after_blend_nll': ['cpanel_dev', 'reports', G0_CELL, 'primary', 'log_loss'],
                   'g0_whole_mlb_dev_blended_nll': ['mlb_dev_blended', 'comparison', REFERENCE_ARM, 'control',
                                                    'log_loss']},
        # Per-seed G15 - G0 deltas (seeds 0-4): the aggregate NLL is invariant to swapping members between seeds,
        # the per-seed pairing used by the 4/5-seed rule is not.
        'seed_delta_fields': {
            'g15_minus_g0_cpanel_dev_after_blend_seed_deltas': ['cpanel_dev', 'comparison', REFERENCE_ARM,
                                                                'seed_deltas'],
            'g15_minus_g0_whole_mlb_dev_blended_seed_deltas': ['mlb_dev_blended', 'comparison', REFERENCE_ARM,
                                                               'seed_deltas']}},
}
PRIMARY_ROW = f"{STAGE_B_RULES['primary']['arm']}/{STAGE_B_RULES['primary']['mix_rule']}"
SECONDARY_ROWS = [f'{arm}/{rule}' for arm in STAGE_B_RULES['candidate_arms'] for rule in STAGE_B_RULES['mix_rules']
                  if f'{arm}/{rule}' != PRIMARY_ROW]


def same(value, expected):
    """JSON-exact equality (True != 1, 8.0 != 8), so 'exact earlier entry' means the same config text."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def is_family(config):
    """The P48 DRAFT family predicate (top-level marker; its arms keep the embed_residual / tabm_residual flags)."""
    return MARKER in config


def check_seeds(config):
    """Exactly the adoption seeds [0, 1, 2, 3, 4], no repeat_tag, 15 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('Adoption-confirm seeds must be exactly [0, 1, 2, 3, 4] with no repeat_tag')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'Adoption-confirm family is exactly {FITS} fits')
    return config['seeds']


def check_training(config):
    """The cuda_direct_v1 budget with the epoch cap 60 for every arm (== P34/P35; nothing else changed)."""
    if not same(config.get('training'), TRAINING):
        raise ValueError(f'Adoption-confirm training must be the cuda_direct_v1 budget with epochs {EPOCH_CAP}')
    return config['training']


def check_device(config):
    """CUDA on a non-empty subset of the five allowed UUIDs (no repeats), 1 <= max_parallel <= 5, no CPU fallback."""
    device = config.get('device')
    if not isinstance(device, dict) or set(device) != set(DEVICE_KEYS):
        raise ValueError('Adoption-confirm device must have exactly ' + ', '.join(DEVICE_KEYS))
    gpus = device['allowed_gpu_uuids']
    if not isinstance(gpus, list) or not gpus or len(set(gpus)) != len(gpus) or any(g not in GPU_UUIDS for g in gpus):
        raise ValueError('Adoption-confirm GPUs must be a non-empty subset of the five allowed A5000 UUIDs')
    if device['kind'] != 'cuda' or device['cpu_fallback'] is not False or type(device['max_parallel']) is not int \
            or not 1 <= device['max_parallel'] <= MAX_PARALLEL:
        raise ValueError(f'Adoption-confirm device is CUDA, max_parallel 1..{MAX_PARALLEL}, no CPU fallback')
    return device


def check_stage_b(config):
    """comparison.stage_b.rules is exactly STAGE_B_RULES (Stage-B scorer parameters are config-owned and fixed)."""
    block = config.get('comparison', {}).get(STAGE_B_KEY)
    if not isinstance(block, dict) or not same(block.get('rules'), STAGE_B_RULES):
        raise ValueError('comparison.stage_b.rules differ from the pre-registered Stage-B rules')
    return block['rules']


def check_family(config):
    """Marker true, no other family marker, the three exact arms in order, seeds 0-4, cap 60, device, contrasts,
    Stage-B rules."""
    if config.get(MARKER) is not True:
        raise ValueError(MARKER + ' must be exactly true')
    if any(marker in config for marker in OTHER_MARKERS):
        raise ValueError('The adoption-confirm family carries no other family marker')
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('Adoption-confirm family must be exactly ' + ', '.join(ARM_ORDER))
    for arm in config['arms']:
        if not same(arm, BASELINES[arm['name']]):
            raise ValueError(arm['name'] + ': arm differs from its exact earlier registered entry')
    check_seeds(config)
    check_training(config)
    check_device(config)
    if not same(config.get('comparison', {}).get(CONTRAST_KEY), CONTRASTS):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed pre-registered pairs')
    check_stage_b(config)
    return config


def check_contrasts(config):
    """comparison.adoption_confirm_contrasts (or None): exactly the fixed list, each changing its declared factors."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, CONTRASTS):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed pre-registered pairs')
    check_family(config)
    arms = {a['name']: a for a in config['arms']}
    for c in contrasts:
        cand, ref = (dict(zip(FACTOR_NAMES, FACTORS[c[k]])) for k in ('candidate', 'reference'))
        workload = {n: WORKLOAD if FACTORS[n][1] == LEVEL else [] for n in (c['candidate'], c['reference'])}
        if any(not same(arms[n]['workload'], w) for n, w in workload.items()):
            raise ValueError(c['name'] + ': workload differs from the registered arms')
        changed = {k for k in FACTOR_NAMES if cand[k] != ref[k]}
        if changed != ISOLATES[c['isolates']] or any(cand.get(k) != v or ref.get(k) != v for k, v in c['held'].items()):
            raise ValueError(c['name'] + ': contrast must change exactly the declared factors vs its reference')
    return contrasts
