"""SIMULATED outcomes: owner replies, scan-gate results, analyst labels, delayed dispute labels.

Everything here is simulated from the (assumed) cost-model parameters, for dashboard and OPE demos.
Every returned record carries simulated=True. Ground truth `is_fraud` comes from the injected
scenario (Booking.meta in demo data), never from a model. Mean realized cost per (action, truth)
equals the cost-matrix cell, so OPE on simulated logs is consistent with the decision rule.
"""
from __future__ import annotations

import random
from typing import Any

from fraudshield.policy.costs import DEFAULT_PARAMS

DISPUTE_MEAN_DAYS = 14.0  # ASSUMPTION
LABEL_WINDOW_DAYS = 30.0  # ASSUMPTION


def simulate_outcome(action: str, is_fraud: bool, carrier_cost: float, rng: random.Random,
                     params: dict[str, float] = DEFAULT_PARAMS) -> dict[str, Any]:
    p = params
    F = carrier_cost
    lost = p["margin"] * F + p["goodwill"]
    leak = F + p["c_dispute"]
    rev = p["analyst_per_min"] * p["review_min"]
    o: dict[str, Any] = {"simulated": True, "action": action, "owner_reply": None, "scan_result": None,
                         "analyst_label": None, "abandoned": False, "label": None, "label_delay_days": None}

    def dispute():
        o["label"], o["label_delay_days"] = "fraud", round(rng.expovariate(1 / DISPUTE_MEAN_DAYS) + 0.1, 2)
        return leak

    def no_dispute():
        o["label"], o["label_delay_days"] = "legit", LABEL_WINDOW_DAYS

    def abandon(prob):
        if rng.random() < prob:
            o["abandoned"] = True
            return lost
        return 0.0

    cost = 0.0
    if action == "allow":
        cost = dispute() if is_fraud else (no_dispute() or 0.0)
    elif action == "allow_scan_gated":
        cost = p["c_scan"]
        if is_fraud:
            if rng.random() < p["q_scan"]:
                o["scan_result"], o["label"], o["label_delay_days"] = "mismatch", "fraud", 0.5
                cost += p["c_intercept"]
            else:
                o["scan_result"] = "match"
                cost += dispute()
        else:
            o["scan_result"] = "match"
            no_dispute()
            cost += abandon(p["ab_scan"])
    elif action == "owner_confirm":
        cost = p["c_msg"]
        unreachable = rng.random() < p["p_unreach"]
        if is_fraud:
            if unreachable:
                o["owner_reply"] = "no_answer"
                cost += p["c_hold"]
            elif rng.random() < p["p_spoof"]:
                o["owner_reply"] = "yes_me"  # fraudster controls the channel
                cost += dispute()
            else:
                o["owner_reply"], o["label"], o["label_delay_days"] = "not_me", "fraud", 0.02
        else:
            cost += p["fatigue"]
            if unreachable:
                o["owner_reply"] = "no_answer"
                cost += abandon(p["ab_wait"])
            else:
                o["owner_reply"], o["label"], o["label_delay_days"] = "yes_me", "legit", 0.02
    elif action == "review":
        cost = rev
        if is_fraud:
            if rng.random() < p["analyst_miss"]:
                o["analyst_label"] = "legit"
                cost += dispute()
            else:
                o["analyst_label"], o["label"], o["label_delay_days"] = "fraud", "fraud", 0.25
        else:
            o["analyst_label"], o["label"], o["label_delay_days"] = "legit", "legit", 0.25
            cost += abandon(p["ab_review"])
    elif action == "hold":
        cost = p["c_hold"]
        if is_fraud:
            o["label"], o["label_delay_days"] = "fraud", 2.0
        else:
            a = abandon(p["ab_hold"])
            cost += a
            if not a:
                o["label"], o["label_delay_days"] = "legit", 2.0
    elif action == "block":
        cost = 0.0 if is_fraud else lost + p["c_block_support"]
    else:
        raise ValueError(action)
    o["realized_cost_brl"] = round(cost, 4)
    o["reward"] = -o["realized_cost_brl"]
    return o
