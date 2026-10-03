"""Deterministic business-language explanation built only from the structured decision record."""
from __future__ import annotations

from typing import Any

from fraudshield.policy.reasons import RULES_BY_CODE

ACTION_LABEL = {
    "allow": "allow the booking",
    "allow_scan_gated": "allow with a scan gate at first scan",
    "owner_confirm": "ask the account owner to confirm",
    "review": "send to analyst review",
    "hold": "hold the booking until verified",
    "block": "block the booking",
}

CHECK_BY_CODE = {
    "COST_FAR_ABOVE_ACCOUNT_NORM": "compare the parcel weight, size and service with the account's usual bookings",
    "NEW_SENDERS_UNDER_PAYER": "confirm with the account owner that the new senders are their customers",
    "NEW_ORIGINS": "confirm the new origin with the account owner",
    "NEW_LOGIN_DEVICE": "check recent logins and contact changes on the account",
    "SENDER_DIFFERS_FROM_ACCOUNT": "check that the account authorised this sender",
    "PAYER_MANY_SENDERS": "list the senders billed to this account this month",
    "CONSIGNEE_MANY_SENDERS": "look up other parcels going to this consignee",
    "WEIGHT_UNUSUAL": "verify the declared weight at first scan",
    "DIMS_UNUSUAL": "verify the declared dimensions at first scan",
    "BURST_LAST_24H": "check the account's bookings from the last 24 hours",
    "NEW_ACCOUNT": "check the account's registration details",
    "UNUSUAL_HOUR": "check whether the account usually books at this hour",
    "LINK_TO_CONFIRMED_FRAUD": "review the linked confirmed fraud case",
}


def _fmt(v: Any) -> str:
    if isinstance(v, (int, float)):
        return str(int(round(v))) if float(v).is_integer() else f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}"
    return str(v)


def _reason_text(code: str, value: Any, baseline: Any) -> str:
    r = RULES_BY_CODE.get(code)
    if r is None:
        return code.replace("_", " ").lower()
    return r.text.format(v=_fmt(value) if value is not None else "?", b=_fmt(baseline))


def build_facts(record: dict[str, Any]) -> dict[str, Any]:
    """The only data the explainer (template or LLM) may use. Never Booking.meta, never ids of people."""
    probs = record.get("probabilities") or {}
    b = record.get("booking") or {}
    feats = {t.get("code"): t for t in record.get("top_features") or []}
    reasons = []
    for code in record.get("reasons") or []:
        t = feats.get(code, {})
        reasons.append({"code": code, "value": t.get("value"), "baseline": t.get("baseline"),
                        "text": _reason_text(code, t.get("value"), t.get("baseline")),
                        "check": CHECK_BY_CODE.get(code, "review the booking details")})
    ec = record.get("expected_costs") or {}
    action = record["action"]
    return {
        "decision_id": record.get("decision_id"),
        "action": action,
        "action_label": ACTION_LABEL.get(action, action),
        "p_fraud": round(float(probs.get("misuse", record.get("gbm_score") or 0.0)), 2),
        "question_probabilities": {k: round(float(v), 2) for k, v in probs.items()},
        "carrier_cost_brl": b.get("carrier_cost"),
        "origin_zip3": b.get("origin_zip3"), "dest_zip3": b.get("dest_zip3"),
        "origin_uf": b.get("origin_uf"), "dest_uf": b.get("dest_uf"),
        "expected_cost_action_brl": round(float(ec[action]), 2) if action in ec else None,
        "expected_cost_allow_brl": round(float(ec["allow"]), 2) if "allow" in ec else None,
        "degraded": bool(record.get("degraded")),
        "reasons": reasons,
    }


def render_template(record: dict[str, Any]) -> str:
    f = build_facts(record)
    parts = [f"Recommended action: {f['action_label']} ({f['action']})."]
    cost = f["carrier_cost_brl"]
    cost_txt = f" on a R${cost:.2f} booking" if isinstance(cost, (int, float)) else ""
    parts.append(f"Estimated fraud probability {f['p_fraud']:.2f}{cost_txt}.")
    if f["reasons"]:
        parts.append("Main reasons: " + "; ".join(r["text"] for r in f["reasons"][:3]) + ".")
        checks = list(dict.fromkeys(r["check"] for r in f["reasons"][:3]))
        parts.append("What to check: " + "; ".join(checks) + ".")
    else:
        parts.append("No single feature stands out; the score comes from the combined model.")
    if f["degraded"]:
        parts.append("Note: the Laya decision model was unavailable, so this decision used the backup LightGBM score with stricter limits.")
    return " ".join(parts)
