"""Fixed reason codes computed from named features (never from LLM text, never from Booking.meta).

Feature names follow the serializer lines in DESIGN.md section 7. If the feature agent uses other
names, edit REASON_RULES (one place); unknown or missing features are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fraudshield.contracts import Booking


@dataclass(frozen=True)
class ReasonRule:
    code: str
    feature: str
    op: str            # ">=" (higher is riskier) or "<=" (lower is riskier)
    threshold: float
    baseline: float    # what a normal booking looks like, shown to analysts
    text: str          # business-language phrase for templates; {v} = value, {b} = baseline
    baseline_feature: str | None = None  # per-booking baseline feature, if the feature store has one


REASON_RULES: tuple[ReasonRule, ...] = (
    ReasonRule("COST_FAR_ABOVE_ACCOUNT_NORM", "cost_vs_median", ">=", 3.0, 1.0,
               "carrier cost is {v}x the account's median booking"),
    ReasonRule("NEW_SENDERS_UNDER_PAYER", "new_senders_l10", ">=", 3.0, 0.0,
               "{v} of the last 10 bookings used senders this account never used before (usual {b})",
               "base_senders_l10"),
    ReasonRule("NEW_ORIGINS", "new_origins_l10", ">=", 3.0, 0.0,
               "{v} of the last 10 bookings came from origins new to this account (usual {b})",
               "base_origins_l10"),
    ReasonRule("NEW_LOGIN_DEVICE", "login_device_age_days", "<=", 1.0, 30.0,
               "the login device was first seen {v} days ago"),
    ReasonRule("SENDER_DIFFERS_FROM_ACCOUNT", "sender_differs", ">=", 1.0, 0.0,
               "the sender is not the account holder"),
    ReasonRule("PAYER_MANY_SENDERS", "payer_senders_30d", ">=", 5.0, 1.0,
               "the account paid for {v} different senders in 30 days"),
    ReasonRule("CONSIGNEE_MANY_SENDERS", "consignee_accts_30d", ">=", 5.0, 1.0,
               "the consignee received parcels from {v} different accounts in 30 days"),
    ReasonRule("WEIGHT_UNUSUAL", "weight_z", ">=", 3.0, 0.0, "the weight is {v} standard units above the account norm"),
    ReasonRule("DIMS_UNUSUAL", "dims_z", ">=", 3.0, 0.0, "the dimensions are {v} standard units above the account norm"),
    ReasonRule("BURST_LAST_24H", "burst_ratio", ">=", 5.0, 1.0,
               "the last 24 hours had {v}x the account's daily booking rate"),
    ReasonRule("NEW_ACCOUNT", "tenure_days", "<=", 30.0, 180.0, "the account is {v} days old"),
    ReasonRule("UNUSUAL_HOUR", "hour_pct", "<=", 2.0, 50.0, "the booking hour is at percentile {v} of the account's hours"),
    ReasonRule("LINK_TO_CONFIRMED_FRAUD", "links_confirmed_fraud", ">=", 1.0, 0.0,
               "the booking has {v} links to confirmed fraud cases"),
    ReasonRule("DROP_ADDRESS_PATTERN", "drop_pattern", ">=", 1.0, 0.0,
               "the receiver looks like a reshipping drop: first seen recently, already receiving parcels "
               "paid by several other accounts"),
    ReasonRule("UNDER_DECLARED_PARCEL", "under_declared", ">=", 1.0, 0.0,
               "the declared weight and size are far below this account's usual parcels"),
)
RULES_BY_CODE = {r.code: r for r in REASON_RULES}


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _severity(r: ReasonRule, v: float) -> float | None:
    if r.op == ">=" and v >= r.threshold:
        return v / max(r.threshold, 1e-9)
    if r.op == "<=" and v <= r.threshold:
        return (r.threshold + 1.0) / (max(v, 0.0) + 1.0)
    return None


def reason_codes(values: dict[str, Any], max_n: int = 5) -> tuple[list[str], list[dict[str, Any]]]:
    hits = []
    for r in REASON_RULES:
        v = _num(values.get(r.feature))
        if v is None:
            continue
        s = _severity(r, v)
        if s is not None:
            hits.append((s, r, v))
    hits.sort(key=lambda h: -h[0])
    hits = hits[:max_n]
    codes = [r.code for _, r, _ in hits]
    top = []
    for _, r, v in hits:
        base = _num(values.get(r.baseline_feature)) if r.baseline_feature else None
        top.append({"name": r.feature, "value": round(v, 2), "baseline": round(base, 2) if base is not None else r.baseline,
                    "code": r.code})
    return codes, top


def booking_signals(b: Booking) -> dict[str, float]:
    """Signals readable from the booking request itself (meta is never read)."""
    return {
        "login_device_age_days": float(b.login_device_age_days),
        "sender_differs": 1.0 if b.sender_id != b.account_id else 0.0,
    }


REASON_PLAIN: dict[str, str] = {
    "COST_FAR_ABOVE_ACCOUNT_NORM": "Carrier cost far above the account's usual booking",
    "NEW_SENDERS_UNDER_PAYER": "Account is paying for senders it never used before",
    "NEW_ORIGINS": "Parcels from origins new to this account",
    "NEW_LOGIN_DEVICE": "Booked from a login device seen for the first time",
    "SENDER_DIFFERS_FROM_ACCOUNT": "Sender is not the account holder",
    "PAYER_MANY_SENDERS": "Account paid for many different senders in 30 days",
    "CONSIGNEE_MANY_SENDERS": "Consignee receives parcels from many unrelated senders",
    "WEIGHT_UNUSUAL": "Weight far above the account's usual parcels",
    "DIMS_UNUSUAL": "Dimensions far above the account's usual parcels",
    "BURST_LAST_24H": "Burst of bookings in the last 24 hours",
    "NEW_ACCOUNT": "Account is less than 30 days old",
    "UNUSUAL_HOUR": "Booked at an hour this account rarely uses",
    "LINK_TO_CONFIRMED_FRAUD": "Linked to a confirmed fraud case",
    "DROP_ADDRESS_PATTERN": "Receiver looks like a reshipping drop",
    "UNDER_DECLARED_PARCEL": "Declared weight and size far below the account's usual parcels",
    "ACCOUNT_FAILED_DEPOT_SCAN": "A parcel from this account failed a depot weight check",
}


def reason_catalog() -> dict[str, str]:
    """Full code -> plain-language text list (GET /reason-codes)."""
    return {r.code: REASON_PLAIN.get(r.code, r.code.replace("_", " ").capitalize()) for r in REASON_RULES}
