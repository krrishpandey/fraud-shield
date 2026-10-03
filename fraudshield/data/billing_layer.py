"""SYNTHETIC billing/login layer for real Olist accounts (DESIGN section 9, phase3_eval_demo 1.2).

Olist has no carrier-account data (no logins, devices, payer instruments, contacts). Every column
listed in SYNTHETIC_COLUMNS is generated here from one seed, identically for every account, so it
carries no fraud-label information. Legit changes are injected as hard negatives:
  new_warehouse 2%, new_login 3%, new_instrument 5%, bundled_expansion 1% (all three within a week),
  of accounts with >= 20 bookings; some changes are pre-announced through a verified change flow.
All rates and distributions are [ASSUMPTION]s, listed in DATA_CARD.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SYNTHETIC_COLUMNS: tuple[str, ...] = (
    "channel", "device_id", "login_device_age_days", "payment_method", "payer_instrument_id",
    "owner_contact_age_days", "origin_synthetic", "hn_change",
)
PAYMENT_MIX = {"account_billing": 0.55, "ach": 0.25, "card": 0.20}
CHANGE_RATES = {"new_warehouse": 0.02, "new_login": 0.03, "new_instrument": 0.05, "bundled_expansion": 0.01}
ANNOUNCE_P = {"new_warehouse": 0.3, "new_login": 0.2, "new_instrument": 0.2, "bundled_expansion": 0.5}
ESTABLISHED_MIN = 20
NO_CONTACT_P = 0.10
NEW_ORIGIN_USE_P = 0.5   # share of post-change bookings shipped from the new warehouse
NEW_DEVICE_USE_P = 0.6   # share of post-change bookings made from the new login


@dataclass
class BillingLayer:
    bookings: pd.DataFrame  # input rows + SYNTHETIC_COLUMNS (origin may be replaced, flagged origin_synthetic)
    accounts: pd.DataFrame  # one row per account
    changes: pd.DataFrame   # legit change events: account_id, kind, at, announced


def _hex(rng: np.random.Generator, n: int = 12) -> str:
    return "".join(rng.choice(list("0123456789abcdef"), n))


def new_account_profile(rng: np.random.Generator, account_id: str, first_booking: pd.Timestamp,
                        n_bookings: int, created_gap_days: float | None = None) -> dict:
    """Synthetic account record. Shared with the injector so injected accounts use the same generator."""
    if created_gap_days is None:
        created_gap_days = rng.uniform(30, 365) if first_booking < pd.Timestamp("2017-03-01") else rng.uniform(1, 30)
    created = first_booking - pd.Timedelta(days=float(created_gap_days))
    n_logins = int(min(3, rng.poisson(1.0) + 1))
    api_p = 0.5 if n_bookings >= 50 else 0.05
    logins = []
    for i in range(n_logins):
        linked = created + pd.Timedelta(days=float(rng.uniform(0, max(created_gap_days, 0.01))) if i else 0.0)
        r = rng.random()
        ch = "api" if r < api_p else ("counter" if r > 0.95 else "web")
        logins.append({"device_id": "dev_" + _hex(rng), "linked_at": linked, "channel": ch})
    pm = rng.choice(list(PAYMENT_MIX), p=list(PAYMENT_MIX.values()))
    contact = None if rng.random() < NO_CONTACT_P else created
    return {
        "account_id": account_id, "account_number": _hex(rng, 9), "account_created_at": created,
        "payment_method": str(pm), "payer_instrument_id": "ins_" + _hex(rng), "owner_contact_set_at": contact,
        "n_logins": n_logins, "logins": logins,
    }


def build_billing_layer(bookings: pd.DataFrame, zip_pool: pd.DataFrame, seed: int) -> BillingLayer:
    """zip_pool: real zip5 prefixes with columns zip5, uf, lat, lng (new warehouses are drawn from it)."""
    rng = np.random.default_rng(seed)
    b = bookings.sort_values(["booked_at", "booking_id"]).reset_index(drop=True).copy()
    zips_by_uf = {uf: g.zip5.to_numpy() for uf, g in zip_pool.groupby("uf")}
    n = len(b)
    channel = np.empty(n, dtype=object)
    device = np.empty(n, dtype=object)
    dev_age = np.zeros(n)
    pay = np.empty(n, dtype=object)
    ins = np.empty(n, dtype=object)
    contact_age = np.full(n, np.nan)
    origin_syn = np.zeros(n, dtype=bool)
    hn = np.full(n, "", dtype=object)
    oz5 = b.origin_zip5.to_numpy(dtype=object).copy()

    accounts, changes = [], []
    for acc, idx in b.groupby("account_id", sort=True).indices.items():
        rows = b.iloc[idx]
        ts = rows.booked_at.to_numpy()
        prof = new_account_profile(rng, acc, pd.Timestamp(ts[0]), len(idx))
        logins = prof["logins"]
        cur_pay, cur_ins = prof["payment_method"], prof["payer_instrument_id"]
        # legit changes (hard negatives) on established accounts
        kinds = []
        if len(idx) >= ESTABLISHED_MIN:
            for kind, p in CHANGE_RATES.items():
                if rng.random() < p:
                    kinds.append(kind)
        ch_at = {}
        if kinds:
            span = ts[-1] - ts[0]
            at = pd.Timestamp(ts[0] + span * rng.uniform(0.3, 0.8)).floor("min")
            for kind in kinds:
                announced = bool(rng.random() < ANNOUNCE_P[kind])
                changes.append({"account_id": acc, "kind": kind, "at": at, "announced": announced})
                ch_at[kind] = at
        new_dev = new_ins = new_zip = None
        wh_at = min([ch_at[k] for k in ("new_warehouse", "bundled_expansion") if k in ch_at], default=None)
        lg_at = min([ch_at[k] for k in ("new_login", "bundled_expansion") if k in ch_at], default=None)
        in_at = min([ch_at[k] for k in ("new_instrument", "bundled_expansion") if k in ch_at], default=None)
        if "bundled_expansion" in ch_at:  # warehouse + login + instrument within one week
            base = ch_at["bundled_expansion"]
            lg_at = base + pd.Timedelta(days=float(rng.uniform(0, 7)))
            in_at = base + pd.Timedelta(days=float(rng.uniform(0, 7)))
        if wh_at is not None:
            uf = rows.origin_uf.iloc[0]
            pool = zips_by_uf.get(uf, np.array([rows.origin_zip5.iloc[0]]))
            pool = pool[pool != rows.origin_zip5.iloc[0]] if len(pool) > 1 else pool
            new_zip = str(rng.choice(pool))
        if lg_at is not None:
            new_dev = {"device_id": "dev_" + _hex(rng), "linked_at": lg_at,
                       "channel": "api" if rng.random() < 0.5 else "web"}
        if in_at is not None:
            new_ins = ("ins_" + _hex(rng), str(rng.choice(list(PAYMENT_MIX), p=list(PAYMENT_MIX.values()))))

        weights = np.array([3.0] + [1.0] * (len(logins) - 1))
        weights /= weights.sum()
        contact = prof["owner_contact_set_at"]
        for j, i in enumerate(idx):
            t = pd.Timestamp(ts[j])
            usable = [k for k, lg in enumerate(logins) if lg["linked_at"] <= t] or [0]
            w = weights[usable] / weights[usable].sum()
            lg = logins[int(rng.choice(usable, p=w))]
            tags = []
            if new_dev is not None and t >= new_dev["linked_at"] and rng.random() < NEW_DEVICE_USE_P:
                lg = new_dev
                tags.append("new_login")
            channel[i] = lg["channel"]
            device[i] = lg["device_id"]
            dev_age[i] = max(0.0, (t - min(lg["linked_at"], t)).total_seconds() / 86400.0)
            if new_ins is not None and t >= in_at:
                ins[i], pay[i] = new_ins
                tags.append("new_instrument")
            else:
                ins[i], pay[i] = cur_ins, cur_pay
            if contact is not None:
                contact_age[i] = max(0.0, (t - contact).total_seconds() / 86400.0)
            if new_zip is not None and t >= wh_at and rng.random() < NEW_ORIGIN_USE_P:
                oz5[i] = new_zip
                origin_syn[i] = True
                tags.append("new_warehouse")
            hn[i] = "+".join(tags)
        prof = {k: v for k, v in prof.items() if k != "logins"}
        accounts.append(prof)

    b["channel"] = channel
    b["device_id"] = device
    b["login_device_age_days"] = dev_age
    b["payment_method"] = pay
    b["payer_instrument_id"] = ins
    b["owner_contact_age_days"] = contact_age
    b["origin_synthetic"] = origin_syn
    b["hn_change"] = hn
    if origin_syn.any():
        b["origin_zip5"] = oz5
        b["origin_zip3"] = b.origin_zip5.str[:3]
    return BillingLayer(
        bookings=b,
        accounts=pd.DataFrame(accounts),
        changes=pd.DataFrame(changes, columns=["account_id", "kind", "at", "announced"]),
    )
