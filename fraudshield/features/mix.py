"""Account mix features (bust-out, T2) and rule flags (reshipping drops, T3; under-declared parcels, T6).

Account mix: what the account shipped BEFORE this booking (bookings at the same instant excluded, as everywhere
in the feature store): share of high-value categories, share of long-distance parcels, mean distance.
A fresh account that ships mostly high-value goods far away looks unlike an honest new seller.

Rule flags (inputs to the decision rules and reason codes, never to the LightGBM model):
- drop_pattern: a receiver first seen within 35 days that already got 3+ parcels paid by 2+ other accounts in
  30 days. Thresholds from the published description of reshipping drops (Hao et al., "Drops for Stuff",
  ACM CCS 2015: drops live about 30 days and receive parcels from several stolen accounts), not fitted on T3 labels.
- under_declared: declared size at least 3 and weight at least 1 robust units below the account's own parcels,
  for accounts with 5+ earlier bookings. The threshold was chosen on the train window (honest parcels flagged
  <= 2%); only a scale at the first depot scan can confirm it, so the policy sends these to a first-scan check.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FAR_KM = 800.0
MIX_FEATURES = ("acct_hv_share", "acct_far_share", "acct_mean_dist")
RULE_FLAGS = ("drop_pattern", "under_declared")  # rule and reason inputs, deliberately not model features
NAN = float("nan")


def account_mix(n: int, hv_sum: float, far_sum: float, dist_sum: float) -> dict[str, float]:
    if n == 0:
        return {k: NAN for k in MIX_FEATURES}
    return {"acct_hv_share": hv_sum / n, "acct_far_share": far_sum / n, "acct_mean_dist": dist_sum / n}


def account_mix_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same values as the live featurizer for every row of a frame with booking_id, account_id, booked_at,
    high_value and dist_km: sums over the account's bookings strictly before this booking's instant."""
    d = df[["booking_id", "account_id", "booked_at", "high_value", "dist_km"]].copy()
    d["_far"] = (d.dist_km >= FAR_KM).astype(float)
    d["_one"] = 1.0
    per_t = d.groupby(["account_id", "booked_at"], sort=True)[["_one", "high_value", "_far", "dist_km"]].sum()
    before = per_t.groupby(level=0).cumsum() - per_t  # everything before this instant, per account
    m = d.join(before, on=["account_id", "booked_at"], rsuffix="_before")
    n = m["_one_before"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "booking_id": m.booking_id.to_numpy(),
            "acct_hv_share": np.where(n > 0, m["high_value_before"] / n, np.nan),
            "acct_far_share": np.where(n > 0, m["_far_before"] / n, np.nan),
            "acct_mean_dist": np.where(n > 0, m["dist_km_before"] / n, np.nan),
        })
    return out


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return NAN
    return x


def rule_flags(v: dict) -> dict[str, int]:
    first = _num(v.get("consignee_first_seen_days"))
    drop = (_num(v.get("consignee_bookings_30d")) >= 3 and _num(v.get("consignee_other_accts_30d")) >= 2
            and not math.isnan(first) and first <= 35)
    dz, wz = _num(v.get("dims_z")), _num(v.get("weight_z"))
    under = (not math.isnan(dz) and not math.isnan(wz) and dz <= -3.0 and wz <= -1.0
             and _num(v.get("n_prior")) >= 5)
    return {"drop_pattern": int(bool(drop)), "under_declared": int(bool(under))}


def rule_flags_frame(df: pd.DataFrame) -> pd.DataFrame:
    """rule_flags for every row of a feature frame (vectorised, same thresholds)."""
    first = df.consignee_first_seen_days
    drop = (df.consignee_bookings_30d >= 3) & (df.consignee_other_accts_30d >= 2) & (first <= 35)
    under = (df.dims_z <= -3.0) & (df.weight_z <= -1.0) & (df.n_prior >= 5)
    return pd.DataFrame({"drop_pattern": drop.fillna(False).astype(int).to_numpy(),
                         "under_declared": under.fillna(False).astype(int).to_numpy()}, index=df.index)
