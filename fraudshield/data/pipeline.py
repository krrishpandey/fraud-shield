"""Shared steps for scripts/build_dataset.py: real bookings + billing layer (cached), zip pools."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.data.billing_layer import build_billing_layer
from fraudshield.data.olist import build_bookings, haversine_km, load_raw, lookup_latlng, zip5, zip_centroids
from fraudshield.data.split import assign_split
from fraudshield.features.geo import write_centroids

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
BILLING_SEED = 20261002


def zip_pool(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Real zip5 prefixes seen as seller or customer zips, with centroid, UF and customer frequency."""
    cent = zip_centroids(raw["geolocation"])
    cust = raw["customers"].assign(zip5=zip5(raw["customers"].customer_zip_code_prefix).to_numpy())
    sell = raw["sellers"].assign(zip5=zip5(raw["sellers"].seller_zip_code_prefix).to_numpy())
    c = cust.groupby("zip5").agg(uf=("customer_state", "first"), n_customers=("customer_id", "size"))
    s = sell.groupby("zip5").agg(uf_s=("seller_state", "first"), n_sellers=("seller_id", "size"))
    pool = c.join(s, how="outer")
    pool["uf"] = pool.uf.fillna(pool.uf_s)
    pool = pool.drop(columns="uf_s").fillna({"n_customers": 0, "n_sellers": 0}).reset_index()
    lat, lng = lookup_latlng(cent, pool.zip5, pool.uf)
    pool["lat"], pool["lng"] = lat, lng
    return pool.dropna(subset=["lat", "lng"]).reset_index(drop=True)


def write_geo(raw) -> None:
    cent = zip_centroids(raw["geolocation"])
    write_centroids(cent["zip3"], cent["uf"])


def recompute_distance(df: pd.DataFrame, pool: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    if not mask.any():
        return df
    p = pool.set_index("zip5")
    sub = df.loc[mask]
    d = haversine_km(sub.origin_zip5.map(p.lat), sub.origin_zip5.map(p.lng),
                     sub.dest_zip5.map(p.lat), sub.dest_zip5.map(p.lng))
    df.loc[mask, "distance_km"] = np.where(np.isnan(d), sub.distance_km, d)
    return df


def real_with_billing(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (bookings with billing layer + split, accounts, legit changes, zip pool). Cached."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    paths = [PROCESSED / f"_cache_{n}.parquet" for n in ("real", "accounts", "changes", "zips")]
    if not force and all(p.exists() for p in paths):
        return tuple(pd.read_parquet(p) for p in paths)  # type: ignore[return-value]
    raw = load_raw(RAW)
    write_geo(raw)
    pool = zip_pool(raw)
    b = build_bookings(raw)
    bl = build_billing_layer(b, pool, seed=BILLING_SEED)
    real = recompute_distance(bl.bookings, pool, bl.bookings.origin_synthetic.to_numpy())
    real["split"] = assign_split(real.booked_at).to_numpy()
    ch = bl.changes.copy()
    rng = np.random.default_rng(BILLING_SEED + 1)
    ch["announced_at"] = ch["at"] - pd.to_timedelta(rng.uniform(1, 7, len(ch)), unit="D")
    acc = bl.accounts.copy()
    for df, p in zip((real, acc, ch, pool), paths):
        df.to_parquet(p, index=False)
    return real, acc, ch, pool
