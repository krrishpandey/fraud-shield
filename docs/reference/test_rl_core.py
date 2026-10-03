import numpy as np
from rl_core import (ACTIONS, K, cost_matrix, expected_costs, bayes_action, logging_policy,
                     ips, snips, dr, ess, dr_pseudo_rewards, attacker_cem)


def test_bayes_extremes():
    assert bayes_action(0.0, 45.0) == "allow"
    assert bayes_action(1.0, 45.0) == "block"


def test_expected_costs_linear():
    C = cost_matrix(45.0)
    e0, e1, eh = expected_costs(0, C), expected_costs(1, C), expected_costs(0.5, C)
    for a in ACTIONS:
        assert abs(eh[a] - 0.5 * (e0[a] + e1[a])) < 1e-9


def test_logging_propensities():
    rng = np.random.default_rng(0)
    for p in [0.01, 0.1, 0.5, 0.95]:
        a, ps, d = logging_policy(p, 45.0, rng)
        assert abs(d.sum() - 1) < 1e-9 and ps == d[ACTIONS.index(a)]
        nz = d[d > 0]
        assert nz.min() >= 0.05 / len(nz) - 1e-12
        if p > 0.3:
            assert d[0] == 0.0


def test_ope_on_policy_equals_mean():
    rng = np.random.default_rng(1)
    n = 5000
    pi0 = rng.dirichlet(np.ones(K), n)
    a = np.array([rng.choice(K, p=q) for q in pi0])
    ps = pi0[np.arange(n), a]
    r = -rng.random(n)
    assert abs(ips(r, a, ps, pi0) - r.mean()) < 1e-9
    assert abs(snips(r, a, ps, pi0) - r.mean()) < 1e-9
    q = np.zeros((n, K))
    q[np.arange(n), a] = r          # perfect model on logged action -> DR == DM
    assert abs(dr(r, a, ps, pi0, q) - np.mean(np.sum(pi0 * q, 1))) < 1e-9
    assert abs(ess(np.ones(10)) - 10) < 1e-9
    G = dr_pseudo_rewards(r, a, ps, q)
    assert np.allclose(G, q)


def test_cem_runs():
    def camp(t, rng):  # toy env: faster and heavier = more value but more detection
        n = 30
        p_det = min(0.9, 0.02 * t[0] + 0.3 * t[2] * (1 - t[1]))
        passed = rng.random(n) > p_det
        burn = int(np.argmax(~passed)) if (~passed).any() else n
        return 40 * t[2] * (1 - t[1]) * burn, burn + 1, burn
    hist, mu = attacker_cem(camp, n_iter=3, pop=10, episodes=2)
    assert len(hist) == 3 and len(mu) == 5
