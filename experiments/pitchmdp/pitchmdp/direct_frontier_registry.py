"""Torch-free registry of the DRAFT direct-head frontier arms (shared by the model module and the CUDA runner).

The CUDA runner and queue validate configs before the GPU is bound, i.e. before
torch may be imported, so the fixed arm registry, the 52 context names and the
config guard live here; direct_frontier_model re-exports them unchanged.
"""
from __future__ import annotations

N_PHYSICAL = 8
CONTROL_WIDTH = 128
CONTROL_WEIGHT_DECAY = .01
# SequenceContext.report() features 0-10, then Archetypes.numeric_features (styles, reliabilities,
# memberships of Archetypes(n_clusters=5)); names checked against the frozen source in the tests.
GAME_STATE = ('balls', 'strikes', 'outs', 'inning', 'defensive_score_lead', 'defender_is_home',
              'first_base', 'second_base', 'third_base', 'batter_left', 'pitcher_left')
STYLE_NAMES = ('contact', 'swing', 'walk', 'strikeout', 'isolated_power', 'groundball')
BATTER_STYLE = tuple(f'batter_style_{n}_prior' for n in STYLE_NAMES)
BATTER_RELIABILITY = tuple(f'batter_style_{n}_reliability' for n in STYLE_NAMES)
BATTER_CLUSTER = tuple(f'batter_archetype_membership_{k}' for k in range(5))
BASE_CONTEXT_NAMES = GAME_STATE + BATTER_STYLE + BATTER_RELIABILITY + BATTER_CLUSTER  # 28
PITCHER_CLUSTER_FEATURES = ('effective_speed', 'release_spin_rate', 'pfx_x', 'pfx_z')  # matrix_sharing.CLUSTER_FEATURES
SUPPORT_NAME = 'pitcher_log1p_train_count_over10'

ARMS = {
    'width145': {'family': 'capacity', 'model': 'flatten_mlp', 'width': 145, 'matches': 'residual_mlp',
                 'target_parameter_count': 170634},
    'width203': {'family': 'capacity', 'model': 'flatten_mlp', 'width': 203, 'matches': 'periodic_flatten_mlp',
                 'target_parameter_count': 286738},
    'no_batter': {'family': 'information', 'model': 'masked_flatten_mlp', 'width': CONTROL_WIDTH,
                  'drop_context': list(BATTER_STYLE + BATTER_RELIABILITY + BATTER_CLUSTER), 'drop_history': False},
    'no_batter_cluster': {'family': 'information', 'model': 'masked_flatten_mlp', 'width': CONTROL_WIDTH,
                          'drop_context': list(BATTER_CLUSTER), 'drop_history': False},
    'no_history': {'family': 'information', 'model': 'masked_flatten_mlp', 'width': CONTROL_WIDTH,
                   'drop_context': [], 'drop_history': True},
    'dropout': {'family': 'regularization', 'model': 'regularized_flatten_mlp', 'width': CONTROL_WIDTH,
                'dropout': .1, 'weight_decay': CONTROL_WEIGHT_DECAY},
    'weight_decay': {'family': 'regularization', 'model': 'regularized_flatten_mlp', 'width': CONTROL_WIDTH,
                     'dropout': 0., 'weight_decay': 1e-4},
    'dropout_weight_decay': {'family': 'regularization', 'model': 'regularized_flatten_mlp', 'width': CONTROL_WIDTH,
                             'dropout': .1, 'weight_decay': 1e-4},
}
MAX_WIDTH = 203
PARAMETER_TOLERANCE = .01


def flatten_parameter_count(width, n_flat=286, n_classes=10):
    h = 2 * width
    return h * h + (n_flat + n_classes + 2) * h + n_classes


def direct_context_names(cluster_columns):
    """The 52 ordered direct-context names (routing stripped) for the exporter; layout guard as context_map."""
    columns = list(cluster_columns)
    k = len(PITCHER_CLUSTER_FEATURES)
    if columns[:k] != list(PITCHER_CLUSTER_FEATURES) or columns[k:k + 1] != ['left_share'] \
            or not all(c.startswith('type_') for c in columns[k + 1:]):
        raise ValueError('Pitcher-profile columns differ from the direct context layout')
    names = [*BASE_CONTEXT_NAMES, *(f'pitcher_{c}' for c in columns), SUPPORT_NAME]
    if len(set(names)) != len(names):
        raise ValueError('Duplicated context names')
    return names


def arm_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unregistered frontier arm: ' + str(arm))
    return ARMS[arm]


def check_config(config):
    """Fail closed unless the frontier config registers exactly ARMS, seeds 0-2 and the control budget."""
    if config.get('protocol') != 'cuda_frontier_v1' or config.get('seeds') != [0, 1, 2]:
        raise ValueError('Config is not cuda_frontier_v1 with seeds 0-2')
    if config.get('training') != {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
                                  'loss_weighting': 'uniform', 'hpo_trials': 0}:
        raise ValueError('Training budget differs from the CUDA control')
    arms = {a['name']: {k: v for k, v in a.items() if k != 'name'} for a in config.get('arms', [])}
    if len(arms) != len(config['arms']) or arms != ARMS:
        raise ValueError('Config arms differ from the frozen frontier registry')
    for name, spec in ARMS.items():
        if spec['width'] > MAX_WIDTH:
            raise ValueError(name + ': width above the bound')
        if 'target_parameter_count' in spec and abs(flatten_parameter_count(spec['width'])
                                                   / spec['target_parameter_count'] - 1) > PARAMETER_TOLERANCE:
            raise ValueError(name + ': parameter count not matched within 1%')
    return config
