"""Evaluation metrics and the deployment gate for retraining.

Operating point: the production cost rule (fraudshield/policy/decide) applied to each model's
probabilities, deterministic (no exploration). A booking is "stopped" when the action is one of
owner_confirm, review, hold, block (friction before the parcel enters the network);
FPR on hard negatives and recall on new patterns are measured at that point. Realized cost uses the
cost matrix cell for (action, true label) with the generic fraud mode.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from fraudshield.data.metrics import pr_auc
from fraudshield.policy.costs import CostConfig, cost_matrix
from fraudshield.policy.decide import PolicyConfig, PolicyContext, decide

STOP_ACTIONS = ("owner_confirm", "review", "hold", "block")
N_BINS = 15


def ece_equal_mass(y, p, bins: int = N_BINS) -> float:
    """Expected calibration error with `bins` equal-mass bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    err = 0.0
    for chunk in np.array_split(order, min(bins, len(p))):
        if len(chunk):
            err += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return float(err / len(p))


def policy_costs(p, y, F, costs: CostConfig, policy: PolicyConfig, tenure=None, owner_contact=None):
    """Actions from the cost rule on probabilities p, and realized cost per booking (BRL)."""
    p, y, F = np.asarray(p, float), np.asarray(y, bool), np.asarray(F, float)
    n = len(p)
    tenure = np.full(n, np.nan) if tenure is None else np.asarray(tenure, float)
    owner_contact = np.full(n, np.nan) if owner_contact is None else np.asarray(owner_contact, float)
    cfg = PolicyConfig(**{**policy.__dict__, "explore_eps": 0.0})
    rng = random.Random(0)
    acts = np.empty(n, dtype=object)
    cost = np.empty(n)
    for i in range(n):
        ctx = PolicyContext(carrier_cost=float(F[i]),
                            account_tenure_days=None if np.isnan(tenure[i]) else float(tenure[i]),
                            owner_contact_age_days=None if np.isnan(owner_contact[i]) else float(owner_contact[i]))
        a = decide({"misuse": float(p[i])}, ctx, costs, cfg, rng).action
        acts[i] = a
        cost[i] = cost_matrix(float(F[i]), costs.params)[a][1 if y[i] else 0]
    return acts, cost


def summary_metrics(y, p, cost, stopped, hn_mask, new_mask) -> dict:
    y = np.asarray(y, bool)
    hn = np.asarray(hn_mask, bool) & ~y
    new = np.asarray(new_mask, bool) & y
    return {
        "pr_auc": _r(pr_auc(y, p)), "ece": _r(ece_equal_mass(y, p)),
        "cost_per_1k_brl": _r(float(np.mean(cost)) * 1000, 2),
        "fpr_hard_negative": _r(float(stopped[hn].mean())) if hn.any() else None,
        "recall_new_pattern": _r(float(stopped[new].mean())) if new.any() else None,
        "n_hard_negative": int(hn.sum()), "n_new_pattern": int(new.sum()),
    }


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def paired_cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters, B: int = 200, alpha: float = 0.05,
                             seed: int = 0) -> tuple[float, float]:
    """Percentile CI of stat(index array) with whole clusters resampled; both models share each resample."""
    clusters = np.asarray(clusters)
    _, codes = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)] if len(codes) else []
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        v = stat(np.concatenate([groups[k] for k in pick]))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


@dataclass(frozen=True)
class GateConfig:
    """Pre-registered gate, docs/LEARNING_GATE.md. mode "noninferiority" (default) or "strict" (original)."""
    min_new_labels: int = 20
    mode: str = "noninferiority"
    cost_tolerance_rel: float = 0.02   # strict: CI lower bound of cost improvement >= -2% of current
    cost_margin_rel: float = 0.05      # non-inferiority: >= -5% of current
    pr_auc_margin: float = 0.02        # non-inferiority: candidate >= current - 0.02
    max_ece_increase: float = 0.02
    max_fpr_hn_increase: float = 0.005  # 0.5 percentage points


def _ece_check(cur, cand, cfg, name):
    d = cand["ece"] - cur["ece"]
    return {"name": name, "passed": bool(d <= cfg.max_ece_increase),
            "detail": f"ECE (15 equal-mass bins) {cur['ece']:.4f} -> {cand['ece']:.4f} (change {d:+.4f}, "
                      f"max +{cfg.max_ece_increase})"}


def _fpr_check(cur, cand, cfg, name):
    a, b = cur.get("fpr_hard_negative"), cand.get("fpr_hard_negative")
    if a is None or b is None:
        return {"name": name, "passed": True, "detail": "no hard negatives in the eval set; check not applicable"}
    return {"name": name, "passed": bool(b - a <= cfg.max_fpr_hn_increase + 1e-12),
            "detail": f"FPR on hard negatives {a * 100:.2f}% -> {b * 100:.2f}% "
                      f"(max +{cfg.max_fpr_hn_increase * 100:.1f} points)"}


def _labels_check(n, cfg):
    return {"name": "min_new_labels", "passed": bool(n >= cfg.min_new_labels),
            "detail": f"{n} new labels since the last retrain (need {cfg.min_new_labels})"}


def _cost_detail(cur, cand, lo, hi, tol):
    return (f"cost per 1,000 bookings {cur['cost_per_1k_brl']:.2f} -> {cand['cost_per_1k_brl']:.2f} BRL; "
            f"improvement 95% CI [{lo:.2f}, {hi:.2f}], lower bound must be >= -{tol:.2f}")


def gate_checks(cur: dict, cand: dict, ci: dict, n_new_labels: int, cfg: GateConfig) -> dict:
    if cfg.mode == "strict":
        checks = []
        tol = cfg.cost_tolerance_rel * float(cur["cost_per_1k_brl"])
        lo, hi = ci["cost_improvement"]
        checks.append({"name": "cost_not_worse", "passed": bool(lo >= -tol), "detail": _cost_detail(cur, cand, lo, hi, tol)})
        lo, hi = ci["pr_auc_diff"]
        checks.append({"name": "pr_auc_not_worse", "passed": bool(np.isnan(hi) or hi >= 0),
                       "detail": f"PR-AUC {cur['pr_auc']} -> {cand['pr_auc']}; difference 95% CI [{lo:.4f}, {hi:.4f}], "
                                 f"fails only if the whole CI is below 0"})
        checks += [_ece_check(cur, cand, cfg, "ece_not_worse"), _fpr_check(cur, cand, cfg, "fpr_hard_negative_not_worse"),
                   _labels_check(n_new_labels, cfg)]
        return {"mode": "strict", "passed": all(c["passed"] for c in checks), "checks": checks}
    if cfg.mode != "noninferiority":
        raise ValueError(f"unknown gate mode {cfg.mode}")
    checks = []
    tol = cfg.cost_margin_rel * float(cur["cost_per_1k_brl"])
    clo, chi = ci["cost_improvement"]
    checks.append({"name": "cost_noninferior", "passed": bool(clo >= -tol), "detail": _cost_detail(cur, cand, clo, chi, tol)})
    floor = cur["pr_auc"] - cfg.pr_auc_margin
    checks.append({"name": "pr_auc_noninferior", "passed": bool(cand["pr_auc"] >= floor - 1e-12),
                   "detail": f"PR-AUC {cur['pr_auc']:.4f} -> {cand['pr_auc']:.4f}; must be >= {floor:.4f} "
                             f"(current - {cfg.pr_auc_margin})"})
    checks += [_ece_check(cur, cand, cfg, "ece_noninferior"), _fpr_check(cur, cand, cfg, "fpr_hard_negative_noninferior")]
    np_ci = ci.get("new_pattern_recall_diff")
    cost_sup = bool(clo > 0)
    np_sup = bool(np_ci is not None and not np.isnan(np_ci[0]) and np_ci[0] > 0)
    np_txt = ("no new-pattern eval set available" if np_ci is None
              else f"new-pattern recall improvement 95% CI [{np_ci[0]:.3f}, {np_ci[1]:.3f}]")
    checks.append({"name": "superiority", "passed": cost_sup or np_sup,
                   "detail": f"needs a lower bound above 0 for either: cost improvement CI [{clo:.2f}, {chi:.2f}] BRL "
                             f"per 1k ({'yes' if cost_sup else 'no'}); {np_txt} ({'yes' if np_sup else 'no'})"})
    checks.append(_labels_check(n_new_labels, cfg))
    return {"mode": "noninferiority", "passed": all(c["passed"] for c in checks), "checks": checks}
