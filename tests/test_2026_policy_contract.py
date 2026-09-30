"""SYNTHETIC-ONLY: MLB-2026-POLICY-PREPARATION-v1 semantics. No data, artifacts or 2026 access."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/pitchmdp"))
spec = importlib.util.spec_from_file_location("policy_contract", ROOT / "scripts/check_2026_policy_contract.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

from pitchmdp.rollout_policy import BCRecord, CategoricalBC, PAState, PastPitch, kl_policy  # noqa: E402
from pitcheezy.ope import dr as DR  # noqa: E402
from pitcheezy.ope import ips as IPS  # noqa: E402

TOL = 1e-12


def test_sequential_dr_exact_oracle_double_robustness():
    r = C.run_checks()
    assert abs(r["path_mass"] - 1) < TOL
    for arm in ("candidate", "reference"):
        assert abs(r[arm]["err_correct_propensity_wrong_q"]) < TOL
        assert abs(r[arm]["err_wrong_propensity_correct_q"]) < TOL
        assert abs(r[arm]["err_clip_1p5_correct_q"]) < TOL
        assert abs(r[arm]["err_both_wrong"]) > 1e-6  # the oracle can fail
    assert abs(r["candidate"]["err_clip_1p5_wrong_q"]) > 1e-6  # clipping is not free when Q is wrong


def test_paired_contrast_expectation_and_same_policy_zero():
    paths = C.trajectories(C.PI_B)
    paired = sum(w * (C.sequential_dr(s, r, C.PI_CAND, C.PI_B, C.wrong_q) - C.sequential_dr(s, r, C.PI_REF, C.PI_B, C.wrong_q))
                 for s, r, w in paths)
    contrast = C.we_contrast(C.value(C.PI_CAND), C.value(C.PI_REF))
    assert abs(paired - contrast["delta"]) < TOL
    ref = np.array([C.sequential_dr(s, r, C.PI_REF, C.PI_B, C.wrong_q) for s, r, _ in paths])
    copy = np.array([C.sequential_dr(s, r, C.PI_REF_COPY, C.PI_B, C.wrong_q) for s, r, _ in paths])
    assert (copy - ref == 0).all()                 # same law, independent callable: zero on every PA
    assert np.abs(copy - np.roll(ref, 1)).max() > 1e-3  # a mispaired contrast would be visible


@pytest.mark.parametrize("kwargs,err", [
    (dict(pi=lambda h: [-1., .5, 1.5]), C.IntegrityError),   # invalid full row, logged component looks fine
    (dict(terminal_we=1.2), C.IntegrityError),
    (dict(steps=[]), C.Unsupported),
    (dict(clip=-1.), C.IntegrityError),
    (dict(clip=np.nan), C.IntegrityError),
    (dict(q_hat=lambda h, a: np.nan), C.IntegrityError),
])
def test_sequential_dr_rejects_invalid_inputs(kwargs, err):
    args = dict(steps=[((), "FF")], terminal_we=.6, pi=C.PI_REF, pi_b=C.PI_B, q_hat=lambda h, a: .5, clip=None)
    args.update(kwargs)
    with pytest.raises(err):
        C.sequential_dr(**args)


def test_recursion_matches_legacy_plain_trajectory_dr_on_valid_inputs():
    rows, r_pa = [], []
    for j, (steps, terminal, _) in enumerate(C.trajectories(C.PI_B)):
        r_pa.append(terminal)
        for t, (h, a) in enumerate(steps):
            p, pb, i = C.PI_CAND(h), C.PI_B(h), C.VOCAB.index(a)
            q = np.array([C.wrong_q(h, b) for b in C.VOCAB])
            rows.append((1, j, t, p[i] / pb[i], q[i], float(p @ q)))
    df = pd.DataFrame(rows, columns=["game_pk", "at_bat_number", "pitch_number", "rho", "q", "v"])
    legacy = DR.traj_dr_terms_rows(df, np.zeros(len(df), np.int64), df.rho.to_numpy(), np.ones(len(df), bool),
                                   df.q.to_numpy(), df.v.to_numpy(), r_pa=np.array(r_pa))
    mine = [C.sequential_dr(s, r, C.PI_CAND, C.PI_B, C.wrong_q) for s, r, _ in C.trajectories(C.PI_B)]
    np.testing.assert_allclose(legacy.dr_plain.to_numpy(), mine, rtol=0, atol=1e-12)


def test_legacy_ratio_helpers_are_not_fail_closed():
    """Pins the compatibility gap: legacy helpers must not be reused unguarded for M1+."""
    df = pd.DataFrame({"action_id": [3, 3, 3]})
    rho, has = IPS.pitch_ratios(df, np.array([.5, .5, .5]), np.array([np.nan, 0., .25]), clip=None)
    assert not has[0] and rho[0] == 1          # missing propensity silently becomes rho=1
    assert has[1] and rho[1] == .5 / 1e-12     # zero propensity floored, not refused
    renorm = IPS.restrict_support(np.array([[.5, .5]]), np.array([[True, False]]))
    assert np.allclose(renorm, [[1., 0.]])      # off-support mass silently renormalised
    with pytest.raises(C.Unsupported):
        C.logging_ratio(.5, 0.)
    with pytest.raises(C.IntegrityError):
        C.logging_ratio(.5, np.nan)
    assert C.logging_ratio(0., .25) == 0       # candidate zero mass is legitimate


@pytest.mark.parametrize("probs,mask,vocab,err", [
    ([.5, .5, 0.], [True, False, True], C.VOCAB, C.IntegrityError),   # mass off support
    ([.5, .4, 0.], [True, True, True], C.VOCAB, C.IntegrityError),    # does not sum to one
    ([np.nan, .5, .5], [True, True, True], C.VOCAB, C.IntegrityError),
    ([-.1, .6, .5], [True, True, True], C.VOCAB, C.IntegrityError),
    ([.2, .3, .5], [True, True, True], ("FF", "CH", "SL"), C.IntegrityError),  # order/hash mismatch
    ([.2, .3, .5], [1, 1, 1], C.VOCAB, C.IntegrityError),              # mask must be boolean
    ([0., 0., 0.], [False, False, False], C.VOCAB, C.Unsupported),     # empty support is a refusal
])
def test_policy_row_fails_closed(probs, mask, vocab, err):
    with pytest.raises(err):
        C.policy_row(probs, np.array(mask), vocab, C.SHA)


def test_policy_row_does_not_renormalise_and_mapping_is_exact():
    p = np.array([.2, .3, .5])
    assert C.policy_row(p, p > 0, C.VOCAB, C.SHA) is not None and p.tolist() == [.2, .3, .5]
    assert C.action_index(C.VOCAB, "SL") == 2
    with pytest.raises(C.IntegrityError):
        C.action_index(C.VOCAB, "KN")  # unmapped code is integrity failure, not a new action


def test_we_units_sign_and_range():
    out = C.we_contrast(.51, .50)
    assert abs(out["delta"] - .01) < TOL and abs(out["delta_pp"] - 1.0) < 1e-9 and out["offense_delta"] == -out["delta"]
    for bad in (1.2, -.1, np.nan):
        with pytest.raises(C.IntegrityError):
            C.we_contrast(bad, .5)


def test_categorical_bc_as_reference_and_kl_candidate_rows():
    s = lambda **k: PAState(**{"balls": 0, "strikes": 0, "pitcher": "p1", "batter_side": "R", **k})  # noqa: E731
    recs = [BCRecord(s(), "FF", "train"), BCRecord(s(), "SL", "train"), BCRecord(s(strikes=1), "CH", "train"),
            BCRecord(s(pitcher="p2"), "FF", "train")]
    bc = CategoricalBC().fit(recs)
    sha = C.vocabulary_sha256(bc.actions)
    state = s(history=(PastPitch("FF", (0.,), "strike", 0, 0),), strikes=1)
    ref = C.policy_row(bc.probabilities(state), bc.support(state), bc.actions, sha)
    cand = kl_policy(np.array([.55, .60, .50]), ref, bc.support(state), .003)
    C.policy_row(cand, bc.support(state), bc.actions, sha)
    assert ((cand > 0) == bc.support(state)).all()
    p2 = s(pitcher="p2")
    assert not bc.support(p2)[bc.actions.index("SL")] and C.policy_row(bc.probabilities(p2), bc.support(p2), bc.actions, sha)[bc.actions.index("SL")] == 0
    assert bc.fallback(s(pitcher="new2026"))  # adapter must refuse; league fallback is not a pre-2026 profile
    with pytest.raises(ValueError):
        CategoricalBC().fit([BCRecord(s(), "FF", "dev")])


def test_restricted_reference_needs_full_logging_law():
    """Policy mask and logging law are independent: renormalising pi_b onto the policy mask is biased."""
    r = C.renormalisation_check()
    assert r["rho_ref_min"] == 0 and r["rho_ref_max"] - 1 > .1  # rho_ref is not identically 1
    assert abs(r["err_full_logging_law_wrong_q"]) < TOL and abs(r["err_full_logging_law_zero_q"]) < TOL
    for k in ("conditional_wrong_q", "conditional_zero_q", "as_zero_wrong_q", "as_zero_zero_q"):
        assert abs(r[f"err_renorm_drop_{k}"]) > 1e-3
    # Only when the policy mask carries all logging mass does restriction leave the law unchanged (rho_ref == 1).
    full = C._restrict(C.PI_B, np.array([True, True, True]))
    assert all(C.logging_ratio(full(h)[C.VOCAB.index(a)], C.PI_B(h)[C.VOCAB.index(a)]) == 1
               for s, _, _ in C.trajectories(C.PI_B) for h, a in s)


def test_logged_action_statuses_are_distinct():
    ref, pb = C.PI_REF_RESTRICTED(()), C.PI_B(())
    assert C.logging_ratio(ref[C.action_index(C.VOCAB, "CH")], pb[C.VOCAB.index("CH")]) == 0  # outside policy mask: rho=0
    with pytest.raises(C.IntegrityError):
        C.action_index(C.VOCAB, "KN")                                                        # unknown label
    with pytest.raises(C.Unsupported):
        C.logging_ratio(ref[C.VOCAB.index("FF")], 0.)                                        # zero estimated logging mass


def test_profile_as_of_2026_is_pinned_to_opening_day():
    """COOP-019 F2: config, code constant and the recorded 2026 opening day agree."""
    import json

    from pitchmdp import policy_requests as preq
    from pitcheezy.data.statcast_fetch import SEASON_DATES

    cfg = json.loads((ROOT / "configs/MLB-2026-POLICY-PREPARATION-v1.json").read_text())["common"]["profile_as_of"]
    assert cfg["status"] == "proposal"
    assert cfg["as_of_exclusive"] == preq.PROFILE_AS_OF_2026 == SEASON_DATES[2026][0].isoformat()
