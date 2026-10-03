"""Review queue under analyst capacity: greedy knapsack on expected loss prevented per analyst minute.

Not RL (phase3_rl section B). v_i = p*leak*(1-analyst_miss) - (1-p)*ab_review*lost, t_i = review minutes.
Items not selected keep their Bayes action without review (deferred).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from fraudshield.policy.costs import DEFAULT_PARAMS


@dataclass(frozen=True)
class QueueItem:
    decision_id: str
    p_fraud: float
    carrier_cost: float
    review_min: float = 6.0


def value_prevented(it: QueueItem, params: dict[str, float] = DEFAULT_PARAMS) -> float:
    leak = it.carrier_cost + params["c_dispute"]
    lost = params["margin"] * it.carrier_cost + params["goodwill"]
    return it.p_fraud * leak * (1 - params["analyst_miss"]) - (1 - it.p_fraud) * params["ab_review"] * lost


def build_queue(items: list[QueueItem], capacity_min: float, params: dict[str, float] = DEFAULT_PARAMS) -> dict:
    scored = []
    for it in items:
        v = value_prevented(it, params)
        scored.append({**asdict(it), "value_brl": round(v, 4), "value_per_min": round(v / max(it.review_min, 1e-9), 4)})
    scored.sort(key=lambda x: -x["value_per_min"])
    selected, deferred, used, shadow = [], [], 0.0, 0.0
    for x in scored:
        if x["value_brl"] > 0 and used + x["review_min"] <= capacity_min:
            selected.append(x)
            used += x["review_min"]
            shadow = x["value_per_min"]
        else:
            deferred.append(x)
    binding = any(x["value_brl"] > 0 for x in deferred)
    return {"selected": selected, "deferred": deferred, "minutes_used": used,
            "shadow_price_per_min": shadow if binding else 0.0}
