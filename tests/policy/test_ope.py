import numpy as np
import pytest

from fraudshield.policy.ope import (
    deployment_gate, dm, dr, ess, evaluate_candidate, importance_weights, ips, paired_bootstrap_diff, snips,
)


def _logs(n=3000, k=3, seed=0):
    rng = np.random.default_rng(seed)
    pi0 = rng.dirichlet(np.ones(k), size=n) * 0.85 + 0.15 / k
    a = np.array([rng.choice(k, p=row) for row in pi0])
    pscore = pi0[np.arange(n), a]
    true_q = np.tile(np.array([-10.0, -5.0, -1.0]), (n, 1)) + rng.normal(0, 0.5, (n, 1))
    r = true_q[np.arange(n), a] + rng.normal(0, 1.0, n)
    return pi0, a, pscore, r, true_q


def test_ips_and_snips_equal_on_policy_mean_when_target_is_logger():
    pi0, a, ps, r, _ = _logs()
    assert ips(r, a, ps, pi0) == pytest.approx(r.mean())
    assert snips(r, a, ps, pi0) == pytest.approx(r.mean())


def test_dr_equals_dm_with_perfect_reward_model():
    pi0, a, ps, _, q = _logs()
    r = q[np.arange(len(a)), a]  # noiseless rewards, q_hat exact
    pi_e = np.zeros_like(pi0)
    pi_e[:, 2] = 1.0
    assert dr(r, a, ps, pi_e, q) == pytest.approx(dm(pi_e, q))


def test_ips_close_to_truth_for_deterministic_target():
    pi0, a, ps, r, q = _logs(n=20000)
    pi_e = np.zeros_like(pi0)
    pi_e[:, 2] = 1.0
    truth = q[:, 2].mean()
    assert ips(r, a, ps, pi_e) == pytest.approx(truth, abs=0.3)
    assert snips(r, a, ps, pi_e) == pytest.approx(truth, abs=0.3)


def test_ess_of_equal_weights_is_n():
    assert ess(np.ones(500)) == pytest.approx(500)
    assert ess(np.array([1.0, 0, 0, 0])) == pytest.approx(1)


def test_importance_weights():
    pi_e = np.array([[0.5, 0.5], [1.0, 0.0]])
    w = importance_weights(np.array([0, 1]), np.array([0.25, 0.5]), pi_e)
    assert w.tolist() == [2.0, 0.0]


def test_paired_bootstrap_zero_for_identical_policies_and_positive_for_better():
    pi0, a, ps, r, q = _logs()
    lo, hi = paired_bootstrap_diff(ips, r, a, ps, pi0, pi0, B=200)
    assert lo == pytest.approx(0) and hi == pytest.approx(0)
    good = np.zeros_like(pi0); good[:, 2] = 1
    bad = np.zeros_like(pi0); bad[:, 0] = 1
    lo, hi = paired_bootstrap_diff(dr, r, a, ps, good, bad, q_hat=q, B=200)
    assert lo > 0 and hi > lo


def test_gate_rules():
    assert deployment_gate(lo_dr_diff=0.1, ess_val=600) is True
    assert deployment_gate(lo_dr_diff=-0.1, ess_val=600) is False
    assert deployment_gate(lo_dr_diff=0.1, ess_val=499) is False
    assert deployment_gate(lo_dr_diff=0.1, ess_val=600, max_w=30, floor=0.05) is False


def test_evaluate_candidate_report():
    pi0, a, ps, r, q = _logs()
    good = np.zeros_like(pi0); good[:, 2] = 1
    rep = evaluate_candidate(r, a, ps, good, pi0, q_hat=q, B=200)
    assert {"ips", "snips", "dr", "dr_diff_ci", "ess", "max_weight", "unsupported_share", "deploy"} <= set(rep)
    assert rep["deploy"] is True
    same = evaluate_candidate(r, a, ps, pi0, pi0, q_hat=q, B=200)
    assert same["deploy"] is False  # no improvement -> baseline stays
