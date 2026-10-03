import json
from pathlib import Path

import pandas as pd
import pytest

from fraudshield.contracts import Booking
from fraudshield.data.demo import DEMO_SCENARIOS, booking_dict, load_demo_bookings, load_demo_store
from fraudshield.features import FeatureStore, featurize

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


def test_booking_dict_is_a_valid_booking_and_labels_only_in_meta():
    row = pd.Series({
        "booking_id": "bk_1", "account_id": "a", "booked_at": pd.Timestamp("2018-06-01 10:00:00"), "channel": "web",
        "login_device_age_days": 3.0, "payment_method": "card", "sender_id": "a", "origin_uf": "SP", "origin_zip3": "013",
        "dest_uf": "RJ", "dest_zip3": "220", "consignee_id": "c", "weight_kg": 1.0, "length_cm": 1.0, "width_cm": 1.0,
        "height_cm": 1.0, "service": "standard", "category": "home", "declared_value": 1.0, "carrier_cost": 2.0,
        "owner_contact_age_days": float("nan"), "typology": "T1", "is_injected": True, "scenario_id": "s",
        "campaign_id": "c1", "is_fraud": True, "camouflage_level": 0,
    })
    d = booking_dict(row)
    b = Booking(**d)
    assert b.owner_contact_age_days is None
    assert b.meta["typology"] == "T1" and b.meta["is_injected"] is True
    assert "typology" not in d and "is_fraud" not in d
    json.dumps(d)


@pytest.mark.skipif(not (PROC / "demo_bookings.json").exists(), reason="demo outputs not built")
def test_demo_outputs_load_and_score():
    items = load_demo_bookings()
    assert [i["scenario"] for i in items] == list(DEMO_SCENARIOS)
    store = load_demo_store()
    assert isinstance(store, FeatureStore) and len(store) > 10_000
    for it in items:
        b = Booking(**it["booking"])
        assert len(store.history(b.account_id, b.booked_at)) > 0 or it["scenario"] == "reshipping-drop"
        assert store.frame.booking_id.ne(b.booking_id).all()  # the demo booking itself is not in history
        fv = featurize(b, store)
        assert fv.values["n_prior"] >= 0
        assert it["expected"] is None or isinstance(it["expected"], str)
