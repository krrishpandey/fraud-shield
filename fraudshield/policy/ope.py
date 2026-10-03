"""Off-policy evaluation from logged decisions (rewards r = -realized cost, BRL).

a: (n,) logged action indices; pscore: (n,) logged propensity of a; pi_e: (n,K) target policy
probabilities; q_hat: (n,K) cross-fitted reward model. Deployment gate (DESIGN 10 / phase3_rl A5):
DR 95% lower bound of V(candidate) - V(baseline) > 0, ESS >= 500, no unsupported actions,
SNIPS and DR agree in sign, optional max weight <= 1/floor.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def importance_weights(a, pscore, pi_e) -> np.ndarray:
    a = np.asarray(a)
    return np.asarray(pi_e)[np.arange(len(a)), a] / np.asarray(pscore, dtype=float)


def ips(r, a, pscore, pi_e) -> float:
    return float(np.mean(importance_weights(a, pscore, pi_e) * np.asarray(r)))


def snips(r, a, pscore, pi_e) -> float:
    w = importance_weights(a, pscore, pi_e)
    s = w.sum()
    return float(np.sum(w * np.asarray(r)) / s) if s > 0 else 0.0


def dm(pi_e, q_hat) -> float:
    return float(np.mean(np.sum(np.asarray(pi_e) * np.asarray(q_hat), axis=1)))


def dr(r, a, pscore, pi_e, q_hat) -> float:
    a = np.asarray(a)
    q_hat = np.asarray(q_hat)
    w = importance_weights(a, pscore, pi_e)
    base = np.sum(np.asarray(pi_e) * q_hat, axis=1)
    return float(np.mean(base + w * (np.asarray(r) - q_hat[np.arange(len(a)), a])))


def ess(w) -> float:
    w = np.asarray(w, dtype=float)
    d = np.sum(w ** 2)
    return float(w.sum() ** 2 / d) if d > 0 else 0.0


def paired_bootstrap_diff(est: Callable, r, a, pscore, pi_c, pi_b, q_hat=None, B=2000, alpha=0.05, seed=0):
    """Percentile CI of V(candidate) - V(baseline), same resample indices for both (paired)."""
    r, a, pscore, pi_c, pi_b = map(np.asarray, (r, a, pscore, pi_c, pi_b))
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        args = (r[i], a[i], pscore[i])
        if q_hat is None:
            d[b] = est(*args, pi_c[i]) - est(*args, pi_b[i])
        else:
            qh = np.asarray(q_hat)[i]
            d[b] = est(*args, pi_c[i], qh) - est(*args, pi_b[i], qh)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def deployment_gate(lo_dr_diff: float, ess_val: float, max_w: float | None = None, floor: float | None = None,
                    min_ess: float = 500, unsupported_share: float = 0.0, signs_agree: bool = True) -> bool:
    ok = lo_dr_diff > 0 and ess_val >= min_ess and unsupported_share == 0 and signs_agree
    if max_w is not None and floor is not None:
        ok = ok and max_w <= 1.0 / floor
    return bool(ok)


def evaluate_candidate(r, a, pscore, pi_c, pi_b, q_hat, pi_0=None, B=2000, alpha=0.05, floor=None,
                       min_ess=500, seed=0) -> dict:
    w = importance_weights(a, pscore, pi_c)
    unsupported = 0.0
    if pi_0 is not None:
        unsupported = float(np.mean(np.any((np.asarray(pi_c) > 0) & (np.asarray(pi_0) <= 0), axis=1)))
    lo, hi = paired_bootstrap_diff(dr, r, a, pscore, pi_c, pi_b, q_hat=q_hat, B=B, alpha=alpha, seed=seed)
    d_dr = dr(r, a, pscore, pi_c, q_hat) - dr(r, a, pscore, pi_b, q_hat)
    d_sn = snips(r, a, pscore, pi_c) - snips(r, a, pscore, pi_b)
    rep = {
        "ips": {"candidate": ips(r, a, pscore, pi_c), "baseline": ips(r, a, pscore, pi_b)},
        "snips": {"candidate": snips(r, a, pscore, pi_c), "baseline": snips(r, a, pscore, pi_b)},
        "dr": {"candidate": dr(r, a, pscore, pi_c, q_hat), "baseline": dr(r, a, pscore, pi_b, q_hat)},
        "dr_diff": d_dr,
        "dr_diff_ci": [lo, hi],
        "ess": ess(w),
        "max_weight": float(w.max()) if len(w) else 0.0,
        "unsupported_share": unsupported,
        "signs_agree": bool(np.sign(d_dr) == np.sign(d_sn)),
    }
    rep["deploy"] = deployment_gate(lo, rep["ess"], rep["max_weight"], floor, min_ess, unsupported, rep["signs_agree"])
    return rep
