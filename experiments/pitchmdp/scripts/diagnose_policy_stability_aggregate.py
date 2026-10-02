"""Offline aggregation of recommendation-stability diagnostics (descriptive; no outcome access).

Consumes finished diagnostic run directories that compared a planning law with the sealed G0
seed-0 law decision by decision:

  * EXP-P18-001 format  : results.json with ``noise_seed`` + decisions.npz (tv, top1_same)  -> kind ``seed``
  * EXP-P17-001 format  : results.json with ``mix``        + decisions.npz (tv, top1_same)  -> kind ``predictor_replacement``
  * seed-stability format: results.json with ``variants`` + sealed.npz + decisions-<label>.npz (laws kept)

It aligns the runs on ``request_id``, refuses runs whose sealed-side identity disagrees or whose
results read outcomes, and reports, separately, (1) fixed-predictor seed variability, (2) sample
count variants, (3) predictor replacement.  Registered verdicts found in the inputs are carried
through unchanged and never recomputed.  Nothing here estimates policy value, re-validates a
policy or supports a service adoption claim.

Only numpy and json are used: no model, data array or weight is read.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

KINDS = ('seed', 'samples', 'predictor_replacement')
IDENTITY_KEYS = ('sealed_path', 'sealed_manifest_sha256', 'sealed_final_identity_sha256', 'sealed_ledger',
                 'sealed_evaluated_decisions', 'tau', 'pitch_cap', 'sealed_seed', 'sealed_samples')
LEGACY_FILE = 'decisions.npz'
TV_FLAG = .05
LIMITS = (
    'Descriptive aggregation of model-internal planning-law distances; not a policy value estimate',
    'Every variant is compared with the one sealed seed-0 law, so variant-vs-sealed distances share a side and are not independent replicates',
    'Whole-game bootstrap intervals describe sampling over games for a fixed sealed law; they do not cover the sealed seed itself',
    'Variants flagged shares_prefix_draws reuse the sealed draws as a prefix (nested comparison, smaller than an independent-seed distance)',
    'Predictor-replacement rows use common random numbers with the sealed side; seed rows do not (the seed distance overstates that noise; see D155)',
    'No outcome, reward or win-expectation result is read; no claim about recommendation quality follows',
    'Registered verdicts of the input runs are carried through unchanged; this aggregation attaches no verdict',
)
CLAIMS_EXCLUDED = ('policy utility or improvement', 'policy re-validation (stage C) status', 'service or demo adoption',
                   'change to any registered verdict (EXP-P17-001, EXP-P18-001) or policy identity')


class AggregationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AggregationError(message)


@dataclass
class Variant:
    label: str
    kind: str
    source: str
    seed: int
    samples: int
    request_id: np.ndarray
    game_pk: np.ndarray
    tv: np.ndarray
    top1_same: np.ndarray
    identity: dict
    law: np.ndarray | None = None
    top1: np.ndarray | None = None
    shares_prefix_draws: bool = False
    registered_verdict: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        require(self.kind in KINDS, f'{self.label}: unknown kind {self.kind!r}')
        n = len(self.request_id)
        require(n > 0, f'{self.label}: no decisions')
        require(len(self.game_pk) == len(self.tv) == len(self.top1_same) == n, f'{self.label}: array lengths differ')
        require(np.isfinite(self.tv).all() and (self.tv >= 0).all() and (self.tv <= 1).all(), f'{self.label}: TV outside [0, 1]')
        if self.law is not None:
            require(self.law.ndim == 2 and len(self.law) == n, f'{self.label}: law array shape invalid')
            require(self.top1 is not None and len(self.top1) == n, f'{self.label}: top1 required with laws')


@dataclass
class Sealed:
    request_id: np.ndarray
    law: np.ndarray
    top1: np.ndarray
    mask: np.ndarray | None = None


def identity_of(results):
    """Sealed-side identity of a run; refuses runs that read outcomes, registered an identity or are smokes."""
    for key in ('outcomes_read', 'policy_identity_registered', 'smoke'):
        require(key in results, f'results.json missing {key!r}: cannot establish no-outcome access')
        require(results[key] is False, f'results.json has {key} = {results[key]!r}; refused')
    sealed, settings = results.get('sealed_g0_run') or {}, results.get('settings') or {}
    identity = {'sealed_path': sealed.get('path'), 'sealed_manifest_sha256': sealed.get('manifest_sha256'),
                'sealed_final_identity_sha256': sealed.get('final_identity_sha256'), 'sealed_ledger': sealed.get('ledger'),
                'sealed_evaluated_decisions': sealed.get('evaluated_decisions'), 'tau': settings.get('tau'),
                'pitch_cap': settings.get('pitch_cap'), 'sealed_seed': settings.get('seed'), 'sealed_samples': settings.get('samples')}
    missing = [k for k, v in identity.items() if v is None]
    require(not missing, f'results.json identity incomplete: {missing}')
    return identity


def registered_bootstrap(results):
    """Bootstrap draws/seed a run registered, if its results.json carries them (new format)."""
    spec = results.get('bootstrap')
    if not isinstance(spec, dict):
        return None
    require(type(spec.get('draws')) is int and spec['draws'] > 0 and type(spec.get('seed')) is int, 'bootstrap block malformed')
    return {'draws': spec['draws'], 'seed': spec['seed']}


def resolve_bootstrap(registered, draws=None, seed=None, *, default_draws=2000, default_seed=20260924):
    """CLI values win; otherwise the one registered setting shared by the inputs; otherwise the defaults."""
    found = [b for b in registered if b is not None]
    require(all(b == found[0] for b in found), f'Inputs registered different bootstrap settings: {found}')
    base = found[0] if found else {'draws': default_draws, 'seed': default_seed}
    return {'draws': base['draws'] if draws is None else draws, 'seed': base['seed'] if seed is None else seed,
            'source': 'cli' if (draws is not None or seed is not None) else ('registered' if found else 'default')}


def registered_verdict(results):
    comparison = results.get('comparison') or {}
    for key in ('verdict', 'stop_rule'):
        if isinstance(comparison.get(key), dict) and comparison[key].get('status'):
            return f"{results.get('experiment_id')}: {comparison[key]['status']} (registered, not recomputed)"
    return None


def _npz(path, keys):
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in keys if k not in data.files]
        require(not missing, f'{path.name} missing arrays {missing}')
        return {k: data[k].copy() for k in keys}


def load_run(run_dir):
    """Variants (and the sealed law if saved) of one finished run directory."""
    run_dir = Path(run_dir)
    results = json.loads((run_dir / 'results.json').read_text())
    identity, source = identity_of(results), str(results.get('experiment_id'))
    verdict, settings = registered_verdict(results), results['settings']
    bootstrap = registered_bootstrap(results)
    if 'variants' in results:
        sealed_arrays = _npz(run_dir / 'sealed.npz', ('request_id', 'law', 'top1', 'mask'))
        sealed = Sealed(**sealed_arrays)
        variants = []
        for label, spec in results['variants'].items():
            arrays = _npz(run_dir / spec['decisions_file'], ('request_id', 'game_pk', 'tv', 'top1_same', 'law', 'top1'))
            variants.append(Variant(label=label, kind=spec['kind'], source=source, seed=int(spec['seed']), samples=int(spec['samples']),
                                    identity=identity, shares_prefix_draws=bool(spec.get('shares_prefix_draws', False)),
                                    registered_verdict=verdict, extra={'seed_basis': spec.get('seed_basis')}, **arrays))
        return variants, sealed, bootstrap
    arrays = _npz(run_dir / LEGACY_FILE, ('request_id', 'game_pk', 'tv', 'top1_same'))
    if 'mix' in results:
        kind, seed, label = 'predictor_replacement', int(settings['seed']), f'{source}-predictor-swap'
    elif 'noise_seed' in results:
        kind, seed, label = 'seed', int(results['noise_seed']), f"{source}-seed{results['noise_seed']}"
    else:
        raise AggregationError(f'{run_dir}: unrecognised results.json format (no variants, mix or noise_seed)')
    return [Variant(label=label, kind=kind, source=source, seed=seed, samples=int(settings['samples']), identity=identity,
                    registered_verdict=verdict, **arrays)], None, bootstrap


def check_identities(variants):
    """All runs must describe the same sealed side; labels must be unique."""
    require(variants, 'No variants to aggregate')
    labels = [v.label for v in variants]
    require(len(set(labels)) == len(labels), f'Duplicate variant labels: {sorted(l for l in labels if labels.count(l) > 1)}')
    first = variants[0]
    for v in variants[1:]:
        differing = [k for k in IDENTITY_KEYS if first.identity.get(k) != v.identity.get(k)]
        require(not differing, f'Sealed-side identity disagrees between {first.label} and {v.label} on {differing}')
    return dict(first.identity)


def _order(request_id, label):
    ids = np.asarray(request_id).astype(str)
    unique, counts = np.unique(ids, return_counts=True)
    require((counts == 1).all(), f'{label}: duplicate request ids ({int((counts > 1).sum())})')
    return ids


def align(variants, sealed=None):
    """Reorder every variant (and the sealed arrays) to the first variant's request order.

    Refuses when the id sets differ: the diagnostic is paired per decision and a partial overlap
    would silently change the decision set.
    """
    reference = _order(variants[0].request_id, variants[0].label)
    aligned, game_pk = [], None
    for v in variants:
        ids = _order(v.request_id, v.label)
        if not np.array_equal(ids, reference):
            missing, extra = np.setdiff1d(reference, ids), np.setdiff1d(ids, reference)
            require(len(missing) == 0 and len(extra) == 0,
                    f'{v.label}: decision set differs from {variants[0].label} (missing {len(missing)}, extra {len(extra)})')
            position = {rid: i for i, rid in enumerate(ids)}
            index = np.array([position[rid] for rid in reference])
        else:
            index = np.arange(len(ids))
        take = lambda a: None if a is None else np.asarray(a)[index]
        row = Variant(label=v.label, kind=v.kind, source=v.source, seed=v.seed, samples=v.samples, request_id=reference.copy(),
                      game_pk=take(v.game_pk), tv=take(v.tv).astype(float), top1_same=take(v.top1_same).astype(bool),
                      identity=v.identity, law=take(v.law), top1=take(v.top1), shares_prefix_draws=v.shares_prefix_draws,
                      registered_verdict=v.registered_verdict, extra=v.extra)
        if game_pk is None:
            game_pk = row.game_pk
        require(np.array_equal(game_pk, row.game_pk), f'{v.label}: game_pk differs from {variants[0].label} for the same request ids')
        aligned.append(row)
    if sealed is not None:
        ids = _order(sealed.request_id, 'sealed')
        require(len(np.setdiff1d(reference, ids)) == 0 and len(np.setdiff1d(ids, reference)) == 0, 'sealed: decision set differs')
        position = {rid: i for i, rid in enumerate(ids)}
        index = np.array([position[rid] for rid in reference])
        sealed = Sealed(request_id=reference.copy(), law=sealed.law[index], top1=sealed.top1[index],
                        mask=None if sealed.mask is None else sealed.mask[index])
    return reference, game_pk, aligned, sealed


def game_bootstrap(columns, games, *, draws, seed):
    """Whole-game bootstrap 95% intervals of the mean of each column (same scheme as EXP-P17/P18)."""
    columns = [np.asarray(c, dtype=float) for c in columns]
    unique, index = np.unique(np.asarray(games), return_inverse=True)
    sums = np.stack([np.bincount(index, weights=c, minlength=len(unique)) for c in (*columns, np.ones(len(columns[0])))])
    picks = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    boot = sums[:, picks].sum(axis=2)
    return [[float(x) for x in np.quantile(boot[k] / boot[-1], (.025, .975))] for k in range(len(columns))], len(unique)


def summarize_variant(v, games, *, draws, seed):
    (tv_ci, same_ci), n_games = game_bootstrap((v.tv, v.top1_same), games, draws=draws, seed=seed)
    return {'label': v.label, 'kind': v.kind, 'source': v.source, 'seed': v.seed, 'samples': v.samples,
            'shares_prefix_draws': v.shares_prefix_draws, 'decisions': int(len(v.tv)), 'games': int(n_games),
            'mean_tv_vs_sealed': float(v.tv.mean()), 'mean_tv_ci95': tv_ci,
            'top1_agreement_vs_sealed': float(v.top1_same.mean()), 'top1_agreement_ci95': same_ci,
            'tv_quantiles': {str(q): float(np.quantile(v.tv, q)) for q in (.5, .9, .99, 1.)},
            f'share_tv_above_{TV_FLAG}': float((v.tv > TV_FLAG).mean()),
            'registered_verdict': v.registered_verdict, **v.extra}


def cross_variant(variants, sealed=None):
    """Statistics that need the laws: pairwise TV among fixed-predictor variants and top-1 agreement across them."""
    with_laws = [v for v in variants if v.law is not None and v.kind != 'predictor_replacement']
    if not with_laws:
        return {'status': 'unmeasured', 'reason': 'no variant saved its per-decision laws (legacy inputs only)'}
    pairwise = {}
    for i, a in enumerate(with_laws):
        for b in with_laws[i + 1:]:
            pairwise[f'{a.label}|{b.label}'] = float(.5 * np.abs(a.law - b.law).sum(axis=1).mean())
    members = [v.top1 for v in with_laws]
    names = [v.label for v in with_laws]
    if sealed is not None:
        members, names = [sealed.top1, *members], ['sealed', *names]
    votes = np.stack(members)                       # [members, decisions]
    unanimous = (votes == votes[0]).all(axis=0)
    modal = np.array([np.bincount(col.astype(int)).max() for col in votes.T]) / len(members)
    return {'members': names, 'pairwise_mean_tv': pairwise, 'top1_unanimous_share': float(unanimous.mean()),
            'top1_modal_agreement_mean': float(modal.mean()),
            'note': 'members with laws only; sealed included when its law was saved'}


def samples_trend(variants):
    """Mean TV to the sealed law by sample count, fixed predictor (descriptive; nesting flagged per row)."""
    rows = {}
    for v in variants:
        if v.kind == 'predictor_replacement':
            continue
        rows.setdefault(v.samples, []).append({'label': v.label, 'seed': v.seed, 'mean_tv_vs_sealed': float(v.tv.mean()),
                                               'shares_prefix_draws': v.shares_prefix_draws})
    return {str(k): rows[k] for k in sorted(rows)}


def support_coverage(sealed, identity, decisions):
    out = {'decisions_compared': int(decisions), 'sealed_evaluated_decisions': int(identity['sealed_evaluated_decisions']),
           'coverage_share': float(decisions / identity['sealed_evaluated_decisions'])}
    if sealed is None or sealed.mask is None:
        out['supported_actions'] = 'unmeasured (sealed mask not saved by the input runs)'
    else:
        count = sealed.mask.sum(axis=1)
        out['supported_actions'] = {'mean_per_decision': float(count.mean()),
                                    'share_decisions_with_single_action': float((count <= 1).mean()),
                                    'note': 'a single supported action cannot move under any seed; TV there is zero by construction'}
    return out


def aggregate(variants, sealed=None, *, draws=2000, seed=20260924):
    identity = check_identities(variants)
    request_id, games, aligned, sealed = align(variants, sealed)
    by_kind = {k: [summarize_variant(v, games, draws=draws, seed=seed) for v in aligned if v.kind == k] for k in KINDS}
    verdicts = sorted({v.registered_verdict for v in aligned if v.registered_verdict})
    return {'identity': identity,
            'alignment': {'decisions': int(len(request_id)), 'games': int(len(np.unique(games))),
                          'variants': [v.label for v in aligned], 'aligned_on': 'request_id (exact set match required)'},
            'support_coverage': support_coverage(sealed, identity, len(request_id)),
            'fixed_predictor_seed_variability': by_kind['seed'],
            'sample_count_variants': by_kind['samples'],
            'predictor_replacement': by_kind['predictor_replacement'],
            'cross_variant': cross_variant(aligned, sealed),
            'samples_trend': samples_trend(aligned),
            'bootstrap': {'draws': int(draws), 'seed': int(seed), 'unit': 'whole game'},
            'registered_verdicts_unchanged': verdicts,
            'outcomes_read': False, 'policy_identity_registered': False,
            'claims_excluded': list(CLAIMS_EXCLUDED), 'limits': list(LIMITS)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True, help='finished diagnostic run directory (repeatable)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=None, help='override the bootstrap draws registered by the inputs (default: registered, else 2000)')
    p.add_argument('--seed', type=int, default=None, help='override the bootstrap seed registered by the inputs (default: registered, else 20260924)')
    args = p.parse_args()
    require(not args.output.exists(), f'Refusing to overwrite {args.output}')
    variants, sealed, registered = [], None, []
    for run_dir in args.run:
        loaded, loaded_sealed, bootstrap = load_run(run_dir)
        variants.extend(loaded)
        registered.append(bootstrap)
        if loaded_sealed is not None:
            require(sealed is None, 'More than one run saved a sealed law; aggregate one seed-stability run at a time')
            sealed = loaded_sealed
    chosen = resolve_bootstrap(registered, args.draws, args.seed)
    report = aggregate(variants, sealed, draws=chosen['draws'], seed=chosen['seed'])
    report['bootstrap']['source'] = chosen['source']
    report['inputs'] = [str(Path(r).resolve()) for r in args.run]
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print('STABILITY_AGGREGATE_COMPLETE', {'variants': len(variants), 'decisions': report['alignment']['decisions']}, flush=True)


if __name__ == '__main__':
    main()
