import numpy as np
import pytest

from fraudshield.learning.metrics import (GateConfig, ece_equal_mass, gate_checks, paired_cluster_bootstrap,
                                          policy_costs)
from fraudshield.learning.retrain import feedback_weights, split_feedback_by_time
from fraudshield.policy.costs import CostConfig
from fraudshield.policy.decide import PolicyConfig


def test_ece_equal_mass_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 30000)
    y = rng.uniform(0, 1, 30000) < p
    assert ece_equal_mass(y, p) < 0.02
    assert ece_equal_mass(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_weights_explored_ipw_clipped_and_self_normalised():
    labs = [{"explored": True, "propensity": 0.05}, {"explored": True, "propensity": 0.5},
            {"explored": True, "propensity": 0.01}, {"explored": False, "propensity": 0.95}]
    w = feedback_weights(labs, clip=20.0)
    raw = np.array([20.0, 2.0, 20.0])  # 1/0.01 = 100 clipped to 20
    assert w[3] == 1.0
    np.testing.assert_allclose(w[:3], raw * 3 / raw.sum())
    assert w[:3].mean() == pytest.approx(1.0)


def test_split_by_booking_time_earliest_70_percent_train():
    labs = [{"booked_at": f"2018-06-{d:02d}T10:00:00", "labelled_at": "2018-09-01", "decision_id": str(d)}
            for d in (5, 1, 9, 3, 7, 2, 8, 4, 10, 6)]
    tr, ev = split_feedback_by_time(labs, 0.7)
    assert [x["decision_id"] for x in tr] == ["1", "2", "3", "4", "5", "6", "7"]
    assert [x["decision_id"] for x in ev] == ["8", "9", "10"]


def test_policy_costs_charge_fraud_allowed_and_friction():
    costs, pol = CostConfig.default(), PolicyConfig()
    F = np.array([100.0, 100.0])
    acts, c = policy_costs(np.array([0.001, 0.999]), np.array([False, True]), F, costs, pol)
    assert acts[0] == "allow" and c[0] == 0.0
    assert acts[1] in {"hold", "review", "allow_scan_gated", "owner_confirm"} and c[1] < 100.0


def test_cluster_bootstrap_is_paired():
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    lo, hi = paired_cluster_bootstrap(lambda i: float(np.mean(a[i] - a[i])), np.arange(500) % 50, B=50)
    assert lo == 0.0 and hi == 0.0


def _m(**kw):
    m = {"pr_auc": 0.6, "ece": 0.01, "cost_per_1k_brl": 400.0, "fpr_hard_negative": 0.03, "recall_new_pattern": 0.2}
    m.update(kw)
    return m


def test_gate_passes_and_fails_with_details():
    cfg = GateConfig(min_new_labels=20, mode="strict")
    ok = gate_checks(_m(), _m(pr_auc=0.62), {"cost_improvement": (-1.0, 20.0), "pr_auc_diff": (-0.01, 0.05)},
                     n_new_labels=50, cfg=cfg)
    assert ok["passed"] is True and {c["name"] for c in ok["checks"]} == {
        "cost_not_worse", "pr_auc_not_worse", "ece_not_worse", "fpr_hard_negative_not_worse", "min_new_labels"}
    assert all(c["detail"] for c in ok["checks"])
    bad = gate_checks(_m(), _m(ece=0.05, fpr_hard_negative=0.04),
                      {"cost_improvement": (-50.0, -10.0), "pr_auc_diff": (-0.2, -0.05)}, n_new_labels=5, cfg=cfg)
    assert bad["passed"] is False
    assert {c["name"] for c in bad["checks"] if not c["passed"]} == {
        "cost_not_worse", "pr_auc_not_worse", "ece_not_worse", "fpr_hard_negative_not_worse", "min_new_labels"}


NI = GateConfig(min_new_labels=20, mode="noninferiority")
NI_NAMES = {"cost_noninferior", "pr_auc_noninferior", "ece_noninferior", "fpr_hard_negative_noninferior",
            "superiority", "min_new_labels"}


def test_noninferiority_gate_passes_on_new_pattern_superiority_only():
    ci = {"cost_improvement": (-15.0, 30.0), "pr_auc_diff": (-0.05, 0.03), "new_pattern_recall_diff": (0.10, 0.40)}
    g = gate_checks(_m(), _m(pr_auc=0.585, cost_per_1k_brl=399.0), ci, n_new_labels=50, cfg=NI)
    assert g["passed"] is True, g
    assert {c["name"] for c in g["checks"]} == NI_NAMES and g["mode"] == "noninferiority"


def test_noninferiority_gate_needs_superiority():
    ci = {"cost_improvement": (-15.0, 30.0), "pr_auc_diff": (-0.05, 0.03), "new_pattern_recall_diff": (-0.1, 0.3)}
    g = gate_checks(_m(), _m(), ci, n_new_labels=50, cfg=NI)
    assert g["passed"] is False
    assert [c["name"] for c in g["checks"] if not c["passed"]] == ["superiority"]
    ci["cost_improvement"] = (1.0, 30.0)  # cost superiority alone is enough
    assert gate_checks(_m(), _m(), ci, n_new_labels=50, cfg=NI)["passed"] is True


def test_noninferiority_margins_each_fail():
    ci = {"cost_improvement": (-25.0, 30.0), "pr_auc_diff": (-0.1, 0.0), "new_pattern_recall_diff": None}
    g = gate_checks(_m(), _m(pr_auc=0.57, ece=0.031, fpr_hard_negative=0.0351), ci, n_new_labels=5, cfg=NI)
    failed = {c["name"] for c in g["checks"] if not c["passed"]}
    assert failed == NI_NAMES  # -25 < -5% of 400 = -20; 0.57 < 0.58; ECE +0.021; FPR +0.51 pt; no superiority
    sup = next(c for c in g["checks"] if c["name"] == "superiority")
    assert "no new-pattern eval set" in sup["detail"]


def test_strict_mode_is_still_available():
    ci = {"cost_improvement": (-15.0, 30.0), "pr_auc_diff": (-0.05, 0.03), "new_pattern_recall_diff": (0.1, 0.4)}
    g = gate_checks(_m(), _m(), ci, n_new_labels=50, cfg=GateConfig(min_new_labels=20, mode="strict"))
    assert g["mode"] == "strict" and g["passed"] is False  # -15 < -2% of 400
