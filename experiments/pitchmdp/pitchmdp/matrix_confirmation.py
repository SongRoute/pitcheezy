"""Additive C1 five-seed extension of immutable G sharing members."""
from __future__ import annotations

from .matrix_data import canonical_hash


CELLS = ('G0-global', 'G1-personal', 'G2-feature', 'G3-cluster', 'G4-partial')
SEEDS = (0, 1, 2, 3, 4)
NEW_SEEDS = (3, 4)
MODES = {
    'G0-global': ('global',),
    'G1-personal': ('global', 'personal'),
    'G2-feature': ('global', 'feature'),
    'G3-cluster': ('global', 'cluster'),
    'G4-partial': ('global', 'cluster', 'personal'),
}
MAPPED_CONTROLS = {'G1-personal': 'G0-global', 'G2-feature': 'G0-global',
                   'G3-cluster': 'G2-feature', 'G4-partial': 'G2-feature'}
BUDGET = {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_config(config: dict) -> dict:
    required = {'protocol', 'experiment_id', 'parent_run', 'parent_preparation_sha256',
                'parent_analysis_sha256', 'cells', 'primary_comparisons', 'selection_status',
                'seeds', 'kind', 'width', 'draws', 'device', 'budget',
                'individual_tau', 'cluster_tau', 'registration'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Invalid C1 confirmation config schema')
    if config['protocol'] != 'ml_confirmation_g_v1' or not isinstance(config['experiment_id'], str) or not config['experiment_id'].strip():
        raise ValueError('Unregistered C1 protocol/experiment')
    if not isinstance(config['parent_run'], str) or not config['parent_run'] or not all(
            _sha(config[name]) for name in ('parent_preparation_sha256', 'parent_analysis_sha256')):
        raise ValueError('Exact G parent run, preparation and panel analysis hashes required')
    cells = config['cells']
    if (not isinstance(cells, list) or not cells or len(cells) != len(set(cells)) or
            any(cell not in CELLS for cell in cells) or cells != [cell for cell in CELLS if cell in cells]):
        raise ValueError('C1 cells must be an ordered nonempty registered G subset')
    pairs = config['primary_comparisons']
    if not isinstance(pairs, list) or len(pairs) > 2 or any(
            not isinstance(pair, list) or len(pair) != 2 or pair[0] not in cells or
            pair[1] not in cells or MAPPED_CONTROLS.get(pair[0]) != pair[1] for pair in pairs):
        raise ValueError('C1 primary comparisons require registered candidate/mapped-control pairs')
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError('Duplicate C1 primary comparison')
    if config['selection_status'] == 'baseline_stability_only':
        if cells != ['G0-global'] or pairs:
            raise ValueError('Baseline-only route requires G0 and no comparison')
    elif config['selection_status'] == 'candidate_comparison':
        if not pairs or set(cells) != {cell for pair in pairs for cell in pair}:
            raise ValueError('C1 candidates and controls must be exactly registered pairs')
    else:
        raise ValueError('Unknown C1 selection status')
    if (config['seeds'] != list(SEEDS) or config['kind'] != 'flatten_mlp' or
            config['width'] != 128 or config['draws'] != 400 or config['budget'] != BUDGET or
            config['individual_tau'] != 1000 or config['cluster_tau'] != 10000 or
            config['device'] not in ('auto', 'cpu', 'mps') or
            not isinstance(config['registration'], dict)):
        raise ValueError('C1 fit, sharing and calibration settings differ from G')
    return config


def required_units(parent_units: dict, selected_cells: list[str]) -> dict:
    modes = {mode for cell in selected_cells for mode in MODES[cell]}
    chosen = {name: spec for name, spec in parent_units.items() if spec['mode'] in modes}
    if not chosen or 'global' not in chosen:
        raise ValueError('G parent lacks required global unit')
    for mode in modes:
        if not any(spec['mode'] == mode for spec in chosen.values()):
            raise ValueError('G parent lacks required unit mode: ' + mode)
    return chosen


def unit_identity(prep: dict, unit: str, seed: int) -> dict:
    if seed not in NEW_SEEDS or unit not in prep['units']:
        raise ValueError('C1 unit/seed not registered')
    return {'preparation_sha256': canonical_hash(prep), 'unit': unit,
            'seed': seed, 'spec': prep['units'][unit]}


def member_identity(prep: dict, cell: str, seed: int) -> dict:
    if seed not in NEW_SEEDS or cell not in prep['cells']:
        raise ValueError('C1 member/seed not registered')
    return {'preparation_sha256': canonical_hash(prep), 'cell': cell, 'seed': seed,
            'parent_preparation_sha256': prep['parent_preparation_sha256']}
