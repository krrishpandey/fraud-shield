"""Assemble real + injected rows for one injection seed, and hard-negative slices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.data.inject import LABEL_COLUMNS, InjectionResult
from fraudshield.data.split import SPLIT_BY_NAME

CHANGE_COLUMNS = ["account_id", "kind", "at", "announced_at", "announced"]


def label_real(real: pd.DataFrame) -> pd.DataFrame:
    r = real.copy()
    defaults = {"is_injected": False, "scenario_id": None, "campaign_id": None, "typology": "none",
                "camouflage_level": -1, "mimic": False, "params_json": None, "seed": -1, "is_fraud": False,
                "label_misuse": False, "label_foreign_senders": False, "label_payoff_max": False,
                "label_drop_consignee": False, "risk_level": 0, "phase": "", "cost_ratio": np.nan}
    for k, v in defaults.items():
        r[k] = v
    r["true_weight_kg"] = r.weight_kg
    return r


def assemble(real: pd.DataFrame, inj: InjectionResult, legit_changes: pd.DataFrame):
    """Returns (rows sorted by time, announced changes, confirmed-fraud events)."""
    rows = pd.concat([label_real(real), inj.rows], ignore_index=True)
    rows = rows.sort_values(["booked_at", "booking_id"], kind="mergesort").reset_index(drop=True)
    ch = pd.concat([legit_changes[CHANGE_COLUMNS], inj.changes[CHANGE_COLUMNS]], ignore_index=True)
    ch = ch[ch.announced.astype(bool)].reset_index(drop=True)
    return rows, ch, inj.confirmed


def hard_negative_slices(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Boolean slices of legit bookings that look unusual (FPR is reported on each)."""
    f = feats.set_index("booking_id").reindex(df.booking_id)
    real = (~df.is_injected).to_numpy()
    first = df[real].groupby("account_id").booked_at.min()
    test_start = SPLIT_BY_NAME["test"].start
    new_seller = df.account_id.map(first).to_numpy() >= test_start
    out = pd.DataFrame({
        # established real seller ships to a state for the first time
        "hn_new_state": real & (f.n_prior.to_numpy() >= 20) & (f.dest_region_seen.to_numpy() == 0),
        # real sellers whose first booking falls in the test window
        "hn_new_seller": real & new_seller,
        # real accounts with a synthetic legit change active (warehouse, login, instrument, bundled)
        "hn_billing_change": real & (df.hn_change.fillna("").to_numpy() != ""),
        # injected legit: new marketplace channel, 3PL
        "hn_injected": df.typology.str.startswith("HN").to_numpy(),
        # consignee also received from another account in the last 30 days
        "hn_multi_account_consignee": real & (f.consignee_other_accts_30d.to_numpy() >= 1),
    }, index=df.index)
    return out
