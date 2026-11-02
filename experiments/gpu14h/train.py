"""Finite recipe deltas; the accepted T trainer remains the only training loop."""
from copy import deepcopy

VARIANTS = {
    'lr030': {'training.lr': .0003},
    'lr060': {'training.lr': .0006},
    'wd001': {'training.weight_decay': .001},
    'wd010': {'training.weight_decay': .01},
    'drop000': {'model.dropout': 0., 'model.encoder_dropout': 0.},
    'drop020': {'model.dropout': .2, 'model.encoder_dropout': .2},
    'depth1': {'model.encoder_layers': 1},
    'depth3': {'model.encoder_layers': 3},
    'width32': {'model.token_dim': 32, 'model.encoder_ff': 64},
    'width128': {'model.token_dim': 128, 'model.encoder_ff': 256},
}


def recipe(producer, variant):
    if variant not in VARIANTS: raise ValueError('unregistered variant')
    if (producer['arm'] != 'T' or producer['seeds'] != [0, 1, 2]
            or producer['training_rule']['epochs'] != 30
            or producer['base']['context']['max_length'] != 32
            or producer['base']['inputs']['feature_dimension'] != 199): raise ValueError('original T recipe identity')
    result = deepcopy(producer)
    for key, value in VARIANTS[variant].items():
        group, field = key.split('.')
        result['base'][group][field] = value
    return result


def run(producer, variant, seed, package, out, synthetic=False):
    from experiments.ws_fullseason.train import run as accepted_t
    return accepted_t(recipe(producer, variant), seed, package, out, synthetic=synthetic)
