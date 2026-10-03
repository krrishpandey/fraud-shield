"""Demo outputs for the API / desktop app: the history store loaded at startup and the scripted bookings.

load_demo_store() -> FeatureStore with all seed-0 bookings (real + injected) except the demo bookings
themselves, plus simulated events (verified changes, confirmed-fraud feedback). Features only read rows
strictly before a booking's time, so later rows in the store never leak into a demo booking's features.
No torch or transformers import here (the app must start without the ml extra).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from fraudshield.features.store import ROW_COLUMNS, FeatureStore, to_ts

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
DEMO_STORE_PATH = PROCESSED / "feature_store_demo.parquet"
DEMO_BOOKINGS_PATH = PROCESSED / "demo_bookings.json"
DEMO_SCENARIOS = ("legit-tenured", "hard-negative-new-state", "takeover", "reshipping-drop", "weight-manipulation")
_META = ("typology", "is_injected", "scenario_id", "campaign_id", "is_fraud", "camouflage_level", "phase",
         "true_weight_kg")


def _clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if hasattr(x, "item"):
        return x.item()
    return x


def booking_dict(row: pd.Series) -> dict:
    """contracts.Booking dict from a dataset row. Labels go only into meta (never a feature)."""
    d = {c: _clean(row[c]) for c in ROW_COLUMNS}
    d["booked_at"] = to_ts(row["booked_at"]).isoformat()
    for c in ("login_device_age_days", "weight_kg", "length_cm", "width_cm", "height_cm", "declared_value", "carrier_cost"):
        d[c] = round(float(d[c]), 3)
    if d["owner_contact_age_days"] is not None:
        d["owner_contact_age_days"] = round(float(d["owner_contact_age_days"]), 3)
    d["meta"] = {k: _clean(row[k]) for k in _META if k in row.index}
    return d


def load_demo_store(path: str | Path = DEMO_STORE_PATH) -> FeatureStore:
    return FeatureStore.load(path)


def load_demo_bookings(path: str | Path = DEMO_BOOKINGS_PATH) -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []
