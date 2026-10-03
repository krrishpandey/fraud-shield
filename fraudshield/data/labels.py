"""Training labels derived from injection truth: risk_level and the per-action cost vector.

The cost matrix mirrors docs/reference/rl_core.py cost_matrix (all parameters are ASSUMPTIONS, the
decision-service agent owns the live cost config). Mode adjustments [ASSUMPTION]: the scan gate
catches 95% of T6 weight manipulation (vs 40% of other fraud); for T2 bust-out the owner is the
fraudster, so owner_confirm always "succeeds" for him; when no verified contact exists (owner_ok
False) owner_confirm falls back to review.
"""
from __future__ import annotations

from fraudshield.contracts import ACTIONS

COST_PARAMS = dict(
    c_dispute=30.0, margin=0.30, goodwill=15.0,
    c_scan=0.50, q_scan=0.40, c_intercept=8.0, ab_scan=0.02,
    c_msg=0.20, fatigue=2.0, p_unreach=0.30, ab_wait=0.30, p_spoof=0.10,
    analyst_per_min=1.00, review_min=6.0, ab_review=0.05, analyst_miss=0.10,
    c_hold=3.0, ab_hold=0.25, c_block_support=5.0,
)
COST_VERSION = "costs-v1"
Q_SCAN_T6 = 0.95


def action_costs(F: float, fraud: bool, typology: str, owner_ok: bool, prm: dict = COST_PARAMS) -> dict[str, float]:
    p = prm
    lost = p["margin"] * F + p["goodwill"]
    leak = F + p["c_dispute"]
    rev = p["analyst_per_min"] * p["review_min"]
    q_scan = Q_SCAN_T6 if typology == "T6" else p["q_scan"]
    p_spoof = 1.0 if typology == "T2" else p["p_spoof"]
    if not fraud:
        c = {
            "allow": 0.0,
            "allow_scan_gated": p["c_scan"] + p["ab_scan"] * lost,
            "owner_confirm": p["c_msg"] + p["fatigue"] + p["p_unreach"] * p["ab_wait"] * lost,
            "review": rev + p["ab_review"] * lost,
            "hold": p["c_hold"] + p["ab_hold"] * lost,
            "block": lost + p["c_block_support"],
        }
    else:
        c = {
            "allow": leak,
            "allow_scan_gated": p["c_scan"] + q_scan * p["c_intercept"] + (1 - q_scan) * leak,
            "owner_confirm": p["c_msg"] + p["p_unreach"] * p["c_hold"] + (1 - p["p_unreach"]) * p_spoof * leak,
            "review": rev + p["analyst_miss"] * leak,
            "hold": p["c_hold"],
            "block": 0.0,
        }
    if not owner_ok:
        c["owner_confirm"] = max(c["owner_confirm"], c["review"])
    return {a: round(float(c[a]), 4) for a in ACTIONS}


def risk_level(fraud: bool, cost_ratio: float, burst: bool = False) -> int:
    """0 legit; misuse 1..4 by loss relative to the account's median cost (contracts.QUESTIONS)."""
    if not fraud:
        return 0
    if burst or cost_ratio > 8:
        return 4
    if cost_ratio > 3:
        return 3
    if cost_ratio > 1.5:
        return 2
    return 1
