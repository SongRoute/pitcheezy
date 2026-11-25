import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from observer_app.explanations import explain_choices


def planner():
    path = Path(__file__).resolve().parents[2]/'runtime_src/pitchmdp/planner.py'
    spec = importlib.util.spec_from_file_location('explanation_test_planner', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explanations_reconstruct_real_solver_including_two_strike_foul_loop():
    module = planner()
    p = np.random.default_rng(42).dirichlet(np.ones(10), size=(4, 3, 1, 4))
    terminal = dict(zip(module.TERMINALS, [.40, .64, .62, .35, .30, .25, .1, .40, .7]))
    plan = module.solve_pa(p, terminal, [0]*4)
    for balls in range(4):
        for strikes in range(3):
            ranked = plan.topk(balls, strikes, 0, 3)
            result = explain_choices([p[balls, strikes, 0, row['action_index']] for row in ranked],
                [row['value'] for row in ranked], plan.values, terminal, balls, strikes)
            terms = result['comparison']['contributions']
            assert abs(result['comparison']['reconstruction_residual_pp']) < 1e-10
            assert sum(item['value_contribution_pp'] for item in terms) == pytest.approx(
                100*(ranked[0]['value']-ranked[1]['value']))
            assert result['outcome_probabilities'][0][0]['label'] == ('볼넷' if balls == 3 else '볼')
            assert result['outcome_probabilities'][0][1]['label'] == ('삼진' if strikes == 2 else '스트라이크')
            assert result['causal_effect'] is False


def test_mismatched_q_value_cannot_generate_plausible_explanation():
    values = np.full((4, 3, 1), .5)
    terminal = {name: .5 for name in planner().TERMINALS}
    with pytest.raises(ValueError, match='reconstruct'):
        explain_choices([[.1]*10]*2, [.6, .5], values, terminal, 0, 0)
