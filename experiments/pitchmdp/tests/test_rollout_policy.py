"""Synthetic-only contracts; no data, fits, services or frozen artifacts touched."""
import unittest

import numpy as np

from pitchmdp.planner import OUTCOMES
from pitchmdp.rollout_policy import (
    BCRecord, BudgetExceeded, CategoricalBC, DeliveryPool, JointSimulator,
    PAState, PastPitch, RolloutImprovement, RowBudget, calibrated_conditional,
    kl_policy, paired_summary, planning_seed, rollouts,
)


def state(**kwargs):
    args = dict(balls=0, strikes=0, pitcher="p", batter_side="L", context_key="pa-1")
    args.update(kwargs)
    return PAState(**args)


def bc():
    return CategoricalBC(prior_strength=1).fit([
        BCRecord(state(), "A", "train"), BCRecord(state(), "B", "train"),
        BCRecord(state(pitcher="other"), "C", "train")])


def simulator(predictor, limit=100000, pool=None):
    fixed = DeliveryPool(np.arange(400)[:, None], "train", "synthetic-hash", "synthetic-key")
    return JointSimulator(pool or (lambda s, a: fixed), predictor,
                          lambda s, event: float(event in ("out", "strikeout", "double_play")),
                          RowBudget(limit))


class PolicyTests(unittest.TestCase):
    def test_train_only_bc_support_fallback_and_frequency_comparator(self):
        model = bc()
        np.testing.assert_array_equal(model.support(state()), [True, True, False])
        np.testing.assert_allclose(model.probabilities(state()), [.5, .5, 0])
        self.assertTrue(model.fallback(state(pitcher="unseen")))
        np.testing.assert_allclose(model.probabilities(state(pitcher="unseen")), [1/3]*3)
        records = [BCRecord(state(), "A", "train")]*9 + [BCRecord(state(balls=1), "B", "train")]
        fitted = CategoricalBC(prior_strength=1).fit(records)
        self.assertGreater(fitted.probabilities(state(balls=1))[1],
                           fitted.probabilities(state(balls=1), frequency=True)[1])
        with self.assertRaises(ValueError):
            model.fit([BCRecord(state(), "A", "dev")])
        with self.assertRaises(TypeError):
            PAState(0, 0, "p", "L", current_pitch_type="A")
        strict = CategoricalBC(minimum_action_count=2).fit([BCRecord(state(), "A", "train")])
        with self.assertRaisesRegex(ValueError, "abstain"):
            strict.probabilities(state())

    def test_conditional_calibration_and_integration_commute(self):
        rng = np.random.default_rng(18)
        logits = rng.normal(size=(3, 400, 10))
        freq = rng.dirichlet(np.ones(10))
        temps = [0.8, 1.4, 2.]
        p = calibrated_conditional(logits, temps, np.tile(freq, (400, 1)), .7)
        z = logits / np.array(temps)[:, None, None]
        z = np.exp(z-z.max(axis=-1, keepdims=True))
        z /= z.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(p.mean(axis=0), .7*z.mean(axis=(0, 1))+.3*freq, atol=1e-14)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-14)

    def test_exact400_train_pools_are_immutable_and_joint_history_is_coherent(self):
        for n, split in ((399, "train"), (400, "dev")):
            with self.assertRaises(ValueError):
                DeliveryPool(np.ones((n, 2)), split, "h", "k")
        pool = DeliveryPool(np.arange(400)[:, None], "train", "h", "k")
        with self.assertRaises(ValueError):
            pool.values[0] = 9
        def predictor(states, actions, physical):
            p = np.zeros((len(states), 10))
            p[np.arange(len(states)), np.where(physical[:, 0] < 200, 0, 1)] = 1
            return p
        sim = simulator(predictor, pool=lambda s, a: pool)
        steps = sim.step([state(), state()], ["A", "B"], [[.1, .4], [.9, .4]])
        self.assertEqual((steps[0].state.balls, steps[1].state.strikes), (1, 1))
        for step, physics, outcome in zip(steps, (40., 360.), ("ball", "strike")):
            self.assertEqual(step.state.history[-1].physics, (physics,))
            self.assertEqual(step.state.history[-1].outcome, outcome)
            self.assertEqual(step.state.history[-1].balls, 0)
        self.assertEqual(sim.budget.network_rows, 6)

    def test_count_transitions_walk_strikeout_foul_and_terminal_events(self):
        def predictor(states, actions, physical):
            p = np.zeros((len(states), 10))
            for i, action in enumerate(actions):
                p[i, OUTCOMES.index(action)] = 1
            return p
        sim = simulator(predictor)
        actions = list(OUTCOMES)
        steps = sim.step([state(balls=3, strikes=2)]*10, actions, np.full((10, 2), .1))
        self.assertEqual(steps[0].value, 0)
        self.assertEqual(steps[1].value, 1)
        self.assertEqual((steps[2].state.balls, steps[2].state.strikes), (3, 2))
        self.assertEqual(steps[2].state.history[-1].outcome, "foul")
        self.assertTrue(all(x.state is None for x in steps[3:]))

    def test_p1_stops_improving_p2_replans_and_p3_uses_same_q(self):
        # First pitch creates a history-dependent second decision. At the second
        # pitch A is bad, B is good; a BC continuation stays a 50/50 mixture.
        seen = []
        def predictor(states, actions, physical):
            p = np.zeros((len(states), 10))
            for i, (s, a) in enumerate(zip(states, actions)):
                seen.append(s.history)
                p[i, 0 if not s.history else 3 if a == "B" else 7] = 1
            return p
        sim = simulator(predictor)
        planner = RolloutImprovement(bc(), sim, lambda s: .5, samples=32, pitch_cap=3)
        uniforms = np.random.default_rng(10).random((64, 3, 3))
        control = rollouts(sim, [state()]*64, planner.policy("P1"), uniforms, lambda s: .5)
        improved = rollouts(sim, [state()]*64, planner.policy("P2"), uniforms, lambda s: .5)
        self.assertTrue((improved.values == 1).all())
        self.assertGreater(improved.values.mean(), control.values.mean())
        generated = state(balls=1, history=(PastPitch("A", (17.,), "ball", 0, 0),))
        q, _ = planner.q_values(generated)
        np.testing.assert_allclose(q[:2], [0, 1])
        names, p = planner.policy("P3", tau=.1)(generated, 1)
        self.assertEqual(p[2], 0)
        self.assertGreater(p[1], .999)
        self.assertAlmostEqual(p.sum(), 1)
        _, continuation = planner.policy("P1")(generated, 1)
        np.testing.assert_array_equal(continuation, planner.bc.probabilities(generated))
        self.assertTrue(any(h and h[-1].outcome == "ball" for h in seen))
        report = paired_summary(improved, control)
        self.assertIsNone(report["causal_effect"])
        self.assertIsNone(report["observational_ope"])

    def test_planning_rng_reproducibility_and_full_history_sensitivity(self):
        s = state(history=(PastPitch("A", (2.,), "foul", 0, 0),))
        changed = state(history=(PastPitch("A", (3.,), "foul", 0, 0),))
        self.assertEqual(planning_seed(s, ["A", "B"], 1), planning_seed(s, ["A", "B"], 1))
        self.assertNotEqual(planning_seed(s, ["A", "B"], 1), planning_seed(changed, ["A", "B"], 1))
        def predictor(states, actions, physical):
            p = np.zeros((len(states), 10)); p[:, 3] = .5; p[:, 7] = .5
            return p
        planner = RolloutImprovement(bc(), simulator(predictor), lambda s: .5)
        q1, _ = planner.q_values(s)
        np.random.default_rng(99).random(1000)
        q2, _ = planner.q_values(s)
        np.testing.assert_array_equal(q1, q2)
        # Same physical/outcome CRNs imply exactly equal action Q in this world.
        self.assertEqual(q1[0], q1[1])

    def test_foul_cap_is_explicit_with_worst_case_bounds(self):
        def endless(states, actions, physical):
            p = np.zeros((len(states), 10)); p[:, 2] = 1
            return p
        sim = simulator(endless)
        model = bc()
        policy = lambda s, t: (model.actions, model.probabilities(s))
        result = rollouts(sim, [state(strikes=2)]*3, policy,
                          np.full((3, 4, 3), .5), lambda s: .6)
        np.testing.assert_array_equal(result.truncated, True)
        np.testing.assert_array_equal(result.pitches, 4)
        np.testing.assert_allclose(result.values, .6)
        np.testing.assert_array_equal(result.lower, 0)
        np.testing.assert_array_equal(result.upper, 1)
        report = paired_summary(result, result)
        self.assertEqual(report["mean_delta_we"], 0)
        self.assertEqual(report["truncation_delta_lower"], -1)
        self.assertEqual(report["truncation_delta_upper"], 1)

    def test_hard_budget_stops_before_predictor_and_preserves_counts(self):
        calls = []
        def predictor(states, actions, physical):
            calls.append(len(states))
            return np.full((len(states), 10), .1)
        sim = simulator(predictor, limit=1)
        sim.step([state()], ["A"], [[.5, .5]])
        with self.assertRaises(BudgetExceeded):
            sim.step([state()], ["A"], [[.5, .5]])
        self.assertEqual(calls, [1])
        self.assertEqual(sim.budget.conditional_rows, 1)

    def test_kl_stable_and_invalid_support_rejected(self):
        p = kl_policy([1, 0, -np.inf], [.2, .8, 0], [1, 1, 0], 1e-9)
        np.testing.assert_array_equal(p, [1, 0, 0])
        with self.assertRaises(ValueError):
            kl_policy([1, 0], [.5, .5], [1, 0], .1)
        with self.assertRaises(ValueError):
            kl_policy([1, 0], [.5, .5], [1, 1], 0)


if __name__ == "__main__":
    unittest.main()
