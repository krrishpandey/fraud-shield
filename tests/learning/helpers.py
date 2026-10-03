"""Small synthetic fixtures for learning tests (no real data artifacts needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.models.gbm import GBMModel, system_features

SYSTEM, TIER = "B1", "R"
FEATS = system_features(SYSTEM, TIER)


def make_frame(n: int, split: str, start: str, seed: int, typologies=("T1",), fraud_rate=0.08, hn_rate=0.03):
    """Booking-level feature rows. T1 fraud: heavy+far. T5 fraud (new pattern): high declared value at night."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATS})
    df["carrier_cost"] = rng.uniform(10, 120, n)
    df["tenure_days"] = rng.uniform(40, 900, n)
    df["booking_id"] = [f"{split}_{seed}_{i}" for i in range(n)]
    df["account_id"] = [f"acc{rng.integers(0, max(5, n // 20))}" for _ in range(n)]
    df["booked_at"] = (pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.uniform(0, 60, n)), unit="D")).astype(str)
    df["split"] = split
    is_f = rng.uniform(size=n) < fraud_rate
    typ = np.array(["none"] * n, dtype=object)
    typ[is_f] = rng.choice(list(typologies), is_f.sum())
    df["is_fraud"], df["typology"] = is_f, typ
    t1, t5 = typ == "T1", typ == "T5"
    df.loc[t1, "weight_kg"] += 3.0
    df.loc[t1, "dist_km"] += 3.0
    df.loc[t5, "declared_value"] += 3.5
    df.loc[t5, "night"] += 3.5
    hn = (~is_f) & (rng.uniform(size=n) < hn_rate)
    df.loc[hn, "weight_kg"] += 1.5  # hard negatives look a bit like T1
    df["hn_injected"] = hn
    df["campaign_id"] = np.where(is_f, [f"c{rng.integers(0, 30)}" for _ in range(n)], None)
    df["owner_contact_age_days"] = 400.0
    return df


def make_reference(seed=0):
    tr = make_frame(1500, "train", "2017-06-01", seed, ("T1",))
    cal = make_frame(800, "cal", "2018-02-15", seed + 1, ("T1",))
    te = make_frame(1500, "test", "2018-05-15", seed + 2, ("T1", "T5"), fraud_rate=0.15)
    return pd.concat([tr, cal, te], ignore_index=True)


def fit_base(df, models_dir):
    tr, cal = df[df.split == "train"], df[df.split == "cal"]
    m = GBMModel(SYSTEM, TIER).fit(tr, tr.is_fraud, cal, cal.is_fraud)
    m.save(models_dir)
    return m


def labels_from_frame(df, source="simulated_analyst", explored_every=10, model_version="gbm-B1-R-v1"):
    """Feedback label records (label_store schema) from synthetic rows, snapshot = the row's features."""
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        rd = r._asdict()
        explored = i % explored_every == 0
        out.append({"decision_id": f"dec_{rd['booking_id']}", "booking_id": rd["booking_id"],
                    "account_id": rd["account_id"], "label": "fraud" if rd["is_fraud"] else "legit",
                    "source": source, "simulated": source == "simulated_analyst",
                    "labelled_at": "2018-09-01T00:00:00", "booked_at": rd["booked_at"],
                    "features": {f: float(rd[f]) for f in FEATS}, "action": "allow_scan_gated" if explored else "allow",
                    "propensity": 0.05 if explored else 0.95, "explored": explored, "model_version": model_version,
                    "raw_laya_misuse": None, "carrier_cost": float(rd["carrier_cost"]), "state_text": "S",
                    "typology": rd["typology"] if rd["typology"] != "none" else None,
                    "campaign_id": rd["campaign_id"], "seq": i})
    return out


def make_pool(df):
    """Add Booking request fields to synthetic feature rows (the simulate pool schema)."""
    p = df.copy()
    n = len(p)
    p["is_injected"] = p["typology"] != "none"
    p["channel"], p["payment_method"], p["service"], p["category"] = "web", "account_billing", "standard", "home"
    p["login_device_age_days"] = 100.0
    p["sender_id"] = p["account_id"]
    p["origin_uf"], p["origin_zip3"], p["dest_uf"], p["dest_zip3"] = "SP", "013", "RJ", "200"
    p["consignee_id"] = [f"cns{i}" for i in range(n)]
    p["length_cm"], p["width_cm"], p["height_cm"] = 30.0, 20.0, 10.0
    p["cost_vs_median"] = 1.0  # read by the fake serializer
    return p


def make_pipeline(tmp_path, **kw):
    from fraudshield.api.pipeline import Pipeline
    from fraudshield.audit.log import AuditLog
    from tests.api import fakes
    return Pipeline(featurize=fakes.fake_featurize, serialize=fakes.fake_serialize, gbm=fakes.fake_gbm,
                    laya=fakes.laya(), audit=AuditLog(tmp_path / "audit.jsonl"), **kw)


def make_new_pattern_pool(seed=50, n=400):
    """Test-window fraud rows pooled 'across seeds': T5 (unseen by v1) plus some T1 (seen)."""
    df = make_frame(n, "test", "2018-05-15", seed, ("T1", "T5"), fraud_rate=1.0)
    df["campaign_id"] = [f"np_{t}_{i % 25}" for i, t in enumerate(df.typology)]
    df["seed"] = np.arange(n) % 5 + 1
    return df
