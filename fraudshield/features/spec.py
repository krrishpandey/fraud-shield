"""Feature catalogue. Tier R = computed from real Olist columns only (time, places, consignee, weight,
dims, cost, value, category, service proxy). Tier F = also uses the SYNTHETIC billing layer (channel,
login device, payment, owner contact, sender id) or simulated label feedback (verified changes,
confirmed fraud). Group "booking" = per-booking features used by B1; B2 uses everything numeric."""
from __future__ import annotations

from dataclasses import dataclass

from fraudshield.data.olist import CATEGORY_VOCAB


@dataclass(frozen=True)
class FeatureSpec:
    tier: str      # "R" or "F"
    group: str     # booking | account | profile | change | graph | event
    dtype: str = "num"  # num | str (str features are for the serializer only)


def _s(tier, group, dtype="num"):
    return FeatureSpec(tier, group, dtype)


FEATURES: dict[str, FeatureSpec] = {
    # booking (raw)
    "origin_uf": _s("R", "booking", "str"), "origin_zip3": _s("R", "booking", "str"),
    "dest_uf": _s("R", "booking", "str"), "dest_zip3": _s("R", "booking", "str"),
    "category": _s("R", "booking", "str"), "service": _s("R", "booking", "str"),
    "channel": _s("F", "booking", "str"), "payment_method": _s("F", "booking", "str"),
    "hour": _s("R", "booking"), "dow": _s("R", "booking"), "night": _s("R", "booking"),
    "dist_km": _s("R", "booking"), "weight_kg": _s("R", "booking"), "volume_l": _s("R", "booking"),
    "max_dim_cm": _s("R", "booking"), "length_cm": _s("R", "booking"),
    "width_cm": _s("R", "booking"), "height_cm": _s("R", "booking"), "declared_value": _s("R", "booking"), "carrier_cost": _s("R", "booking"),
    "cost_per_kg": _s("R", "booking"), "express": _s("R", "booking"), "category_code": _s("R", "booking"),
    "high_value": _s("R", "booking"), "tenure_days": _s("R", "booking"),
    "channel_code": _s("F", "booking"), "payment_code": _s("F", "booking"),
    "login_device_age_days": _s("F", "booking"), "owner_contact_age_days": _s("F", "booking"),
    "owner_contact_ok": _s("F", "booking"),
    # account activity
    "n_prior": _s("R", "account"), "bookings_90d": _s("R", "account"), "bookings_72h": _s("R", "account"),
    "bookings_24h": _s("R", "account"), "rate_90d": _s("R", "account"), "burst_ratio": _s("R", "account"),
    "express_72h": _s("R", "account"), "mean_kg_72h": _s("R", "account"),
    "home_origin_uf": _s("R", "account", "str"), "home_origin_zip3": _s("R", "account", "str"),
    "is_home_origin": _s("R", "account"),
    # profile vs own history
    "origin_seen": _s("R", "profile"), "payer_origin_pair_age_days": _s("R", "profile"),
    "lane_seen": _s("R", "profile"), "dest_region_seen": _s("R", "profile"), "cat_seen": _s("R", "profile"),
    "weight_z": _s("R", "profile"), "dims_z": _s("R", "profile"), "value_z": _s("R", "profile"),
    "payoff_z": _s("R", "profile"), "cost_vs_median": _s("R", "profile"), "hour_pct": _s("R", "profile"),
    "svc_exp_pct": _s("R", "profile"),
    "sender_differs": _s("F", "profile"), "sender_seen": _s("F", "profile"),
    "payer_sender_pair_age_days": _s("F", "profile"),
    # change block (last 10 bookings vs 90-day base rate per 10)
    "new_consignees_l10": _s("R", "change"), "base_consignees_l10": _s("R", "change"),
    "new_origins_l10": _s("R", "change"), "base_origins_l10": _s("R", "change"),
    "distinct_origins_7d": _s("R", "change"), "base_origins_7d": _s("R", "change"),
    "origin_entropy_jump": _s("R", "change"),
    "new_senders_l10": _s("F", "change"), "base_senders_l10": _s("F", "change"),
    "distinct_senders_7d": _s("F", "change"), "base_senders_7d": _s("F", "change"),
    "sender_entropy_jump": _s("F", "change"),
    # graph
    "payer_origins_30d": _s("R", "graph"), "consignee_first_seen_days": _s("R", "graph"),
    "consignee_prior": _s("R", "graph"), "consignee_other_accts_30d": _s("R", "graph"),
    "consignee_accts_30d": _s("R", "graph"), "consignee_bookings_30d": _s("R", "graph"),
    "payer_senders_30d": _s("F", "graph"),
    # events (simulated account-system and label feedback)
    "verified_change_30d": _s("F", "event"), "prior_confirmed_fraud": _s("F", "event"),
    "links_confirmed_fraud": _s("F", "event"),
}

CHANNELS = ("web", "api", "counter")
PAYMENTS = ("account_billing", "card", "ach")
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORY_VOCAB)}


def numeric_features(tier: str, groups: set[str] | None = None) -> list[str]:
    """Numeric feature names usable by a model of this tier (R excludes F features)."""
    allowed = {"R"} if tier == "R" else {"R", "F"}
    return [n for n, s in FEATURES.items()
            if s.dtype == "num" and s.tier in allowed and (groups is None or s.group in groups)]
