"""FeatureVector -> fixed-order text state for Laya (DESIGN section 7).

Fixed line order, rounded numbers, closed vocabularies only (no shipper-controlled free text, no ids),
so the state cannot carry a prompt injection. Every line is always present; missing profile stats
render as "n/a". Deviations from the DESIGN template (documented in DATA_OUTPUTS.md): ACCOUNT ends
with "verified change 30d", CHANGE adds distinct origins 7d, and SEQUENCE is always present.
Measured worst case is checked against the 380-token cap in tests.
"""
from __future__ import annotations

import math
import re

import pandas as pd

from fraudshield.contracts import FeatureVector
from fraudshield.data.olist import CATEGORY_VOCAB

SERIALIZER_VERSION = "ser-v1"
LINE_KEYS = ("BOOKING", "ACCOUNT", "SHIPMENT", "COST", "PROFILE", "SENDER", "CONSIGNEE", "CHANGE",
             "SEQUENCE", "GRAPH")
_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_PAY = {"account_billing": "account billing", "card": "card", "ach": "ach"}


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _i(x, cap: int = 999) -> str:
    x = _num(x)
    if math.isnan(x):
        return "n/a"
    return f"{cap}+" if x > cap else f"{int(round(x))}"


def _f(x, nd=1, cap: float = 9999.0) -> str:
    x = _num(x)
    if math.isnan(x):
        return "n/a"
    return f"{cap:.0f}+" if x > cap else f"{x:.{nd}f}"


def _z(x) -> str:
    x = _num(x)
    return "n/a" if math.isnan(x) else f"{x:+.1f}"


def _days(x) -> str:
    x = _num(x)
    if math.isnan(x):
        return "none"
    return "9999+d" if x > 9999 else f"{int(max(x, 0.0))}d"


def _uf(x) -> str:
    return x if isinstance(x, str) and re.fullmatch(r"[A-Z]{2}", x) else "??"


def _zip3(x) -> str:
    return x if isinstance(x, str) and re.fullmatch(r"\d{3}", x) else "???"


def _closed(x, allowed, default="other") -> str:
    return x if x in allowed else default


def serialize(fv: FeatureVector, gbm_risk: float | None = None) -> str:
    v = fv.values
    t = pd.Timestamp(fv.as_of)
    exp = _num(v["svc_exp_pct"])
    svc = "n/a" if math.isnan(exp) else f"std {int(round(100 - exp))}% exp {int(round(exp))}%"
    lines = [
        f"BOOKING {t:%Y-%m-%d} {_DOW[t.dayofweek]} {t:%H:%M} | channel {_closed(v['channel'], ('web', 'api', 'counter'))}"
        f" | login device seen {_days(v['login_device_age_days'])} | payment {_PAY.get(v['payment_method'], 'other')}",
        f"ACCOUNT tenure {_days(v['tenure_days'])} | bookings 90d {_i(v['bookings_90d'])} ({_f(v['rate_90d'])}/day)"
        f" | last 24h {_i(v['bookings_24h'])} | home origin {_uf(v['home_origin_uf'])} {_zip3(v['home_origin_zip3'])}"
        f" | prior confirmed fraud {_i(v['prior_confirmed_fraud'])} | verified change 30d {'yes' if v['verified_change_30d'] else 'no'}",
        f"SHIPMENT origin {_uf(v['origin_uf'])} {_zip3(v['origin_zip3'])} -> dest {_uf(v['dest_uf'])} {_zip3(v['dest_zip3'])}"
        f" | {_i(v['dist_km'], 9999)} km | {_f(v['weight_kg'])} kg | {_i(v['length_cm'])}x{_i(v['width_cm'])}x{_i(v['height_cm'])} cm"
        f" | service {_closed(v['service'], ('standard', 'express'), 'standard')} | category {_closed(v['category'], CATEGORY_VOCAB)}",
        f"COST carrier cost R${_f(v['carrier_cost'], 2, 99999.0)} | cost vs acct median x{_f(v['cost_vs_median'], 1, 999.0)} | payoff z {_z(v['payoff_z'])}",
        f"PROFILE lane seen {_i(v['lane_seen'])}x | weight z {_z(v['weight_z'])} | dims z {_z(v['dims_z'])}"
        f" | hour pct {_i(v['hour_pct'])} | service {svc}",
        f"SENDER {'differs from account' if v['sender_differs'] else '= account'}"
        f" | payer-sender pair age {_days(v['payer_sender_pair_age_days'])} | payer-origin pair age {_days(v['payer_origin_pair_age_days'])}",
        f"CONSIGNEE first seen {_days(v['consignee_first_seen_days'])} | other senders to it 30d {_i(v['consignee_other_accts_30d'])}"
        f" | dest region seen by acct {_i(v['dest_region_seen'])}x",
        f"CHANGE last 10 vs 90d base: new consignees {_i(v['new_consignees_l10'])} (base {_f(v['base_consignees_l10'])})"
        f" | new senders {_i(v['new_senders_l10'])} (base {_f(v['base_senders_l10'])})"
        f" | new origins {_i(v['new_origins_l10'])} (base {_f(v['base_origins_l10'])})"
        f" | distinct senders 7d {_i(v['distinct_senders_7d'])} (base {_i(v['base_senders_7d'])})"
        f" | distinct origins 7d {_i(v['distinct_origins_7d'])} (base {_i(v['base_origins_7d'])})",
        f"SEQUENCE last 72h {_i(v['bookings_72h'])} bookings | express {_i(v['express_72h'])} | mean {_f(v['mean_kg_72h'])} kg",
        f"GRAPH payer->senders 30d {_i(v['payer_senders_30d'])} | consignee<-accounts 30d {_i(v['consignee_accts_30d'])}"
        f" | 2-hop links to confirmed fraud {_i(v['links_confirmed_fraud'])}",
    ]
    if gbm_risk is not None:
        lines.append(f"GBM risk {float(gbm_risk):.3f}")
    return "\n".join(lines)
