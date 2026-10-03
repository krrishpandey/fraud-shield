"""Realistic feedback: analyst labels only where we stopped/gated, delayed accurate outcomes for allows."""
from datetime import datetime, timedelta

import pandas as pd
from pathlib import Path
ROOT_FEATURES = Path(__file__).resolve().parents[2] / "data/processed/features/seed_0.parquet"

from fraudshield.learning.label_store import LabelStore, label_from_decision
from fraudshield.learning.simulate import RealisticFeedbackSim
from tests.learning.helpers import make_frame, make_pool


class StubPipe:
    """Action by rule: weight_kg > 1 -> hold, row index % 10 == 0 -> explored scan-gate, else allow."""

    def __init__(self, store):
        self.label_store, self.recs, self.by_b, self.n = store, {}, {}, 0

    def score(self, booking, fv=None):
        self.n += 1
        did = f"dec_{self.n}"
        w = fv.values["weight_kg"]
        explored = self.n % 10 == 0
        act = "hold" if w > 1 else ("allow_scan_gated" if explored else "allow")
        rec = {"decision_id": did, "booking_id": booking.booking_id, "account_id": booking.account_id,
               "booked_at": booking.booked_at, "action": act, "propensity": 0.05 if explored else 0.95,
               "explored": explored, "feature_values": dict(fv.values), "model_versions": {"gbm": "v1"},
               "booking": {"meta": booking.meta, "carrier_cost": booking.carrier_cost}}
        self.recs[did], self.by_b[booking.booking_id] = rec, did
        return rec

    def get_by_booking(self, bid):
        return self.recs.get(self.by_b.get(bid))

    def analyst_feedback(self, did, label, note="", source="analyst", at=None):
        self.label_store.append(label_from_decision(self.recs[did], label, source, labelled_at=at, note=note))


def _setup(tmp_path, n=300, error_rate=0.0, seed=0):
    pool = make_pool(make_frame(n, "test", "2018-05-15", 11, ("T1", "T5"), fraud_rate=0.2))
    store = LabelStore(tmp_path / "l.jsonl")
    pipe = StubPipe(store)
    return pool, store, pipe, RealisticFeedbackSim(pool, seed=seed, error_rate=error_rate)


def _ts(s):
    return datetime.fromisoformat(str(s).replace(" ", "T"))


def test_analyst_labels_only_on_stopped_or_gated_and_flagged(tmp_path):
    pool, store, pipe, sim = _setup(tmp_path)
    out = sim.step(pipe, n=300)
    assert out["simulated"] is True and out["mode"] == "realistic" and out["scored"] == 300
    analyst = [x for x in store.all() if x["source"] == "simulated_analyst"]
    assert analyst and all(x["action"] in {"allow_scan_gated", "owner_confirm", "review", "hold", "block"}
                           and not x["explored"] for x in analyst)
    assert all(x["simulated"] for x in store.all())
    assert all(abs((_ts(x["labelled_at"]) - _ts(x["booked_at"]) - timedelta(days=1)).total_seconds()) < 1
               for x in analyst
               if _ts(x["booked_at"]) + timedelta(days=1) <= _ts(out["now"]))


def test_immature_allowed_legit_stays_unlabelled_until_60_days(tmp_path):
    pool, store, pipe, sim = _setup(tmp_path)
    out = sim.step(pipe, n=300)
    now = _ts(out["now"])
    truth = dict(zip(pool.booking_id, pool.is_fraud))
    legit_out = [x for x in store.all() if x["source"] == "simulated_outcome" and x["label"] == "legit"]
    assert all(_ts(x["booked_at"]) + timedelta(days=60) <= now for x in legit_out)
    allowed_legit = [r for r in pipe.recs.values() if r["action"] == "allow" and not truth[r["booking_id"]]]
    labelled = {x["booking_id"] for x in store.all()}
    immature = [r for r in allowed_legit if _ts(r["booked_at"]) + timedelta(days=60) > now]
    assert immature and not any(r["booking_id"] in labelled for r in immature)
    assert out["pending"] >= len(immature)
    out2 = sim.step(pipe, n=0, advance_days=61)
    labelled = {x["booking_id"] for x in store.all()}
    assert all(r["booking_id"] in labelled for r in allowed_legit) and out2["pending"] == 0


def test_allowed_fraud_gets_accurate_delayed_outcome(tmp_path):
    pool, store, pipe, sim = _setup(tmp_path, error_rate=1.0)  # analyst always wrong, outcomes still right
    sim.step(pipe, n=300, advance_days=61)
    truth = dict(zip(pool.booking_id, pool.is_fraud))
    outc = [x for x in store.all() if x["source"] == "simulated_outcome"]
    fr = [x for x in outc if truth[x["booking_id"]]]
    assert fr and all(x["label"] == "fraud" for x in fr)
    delays = [(_ts(x["labelled_at"]) - _ts(x["booked_at"])).total_seconds() / 86400 for x in fr]
    assert all(7 <= d <= 60 for d in delays) and len(set(round(d, 3) for d in delays)) > 1
    assert all(x["label"] == "legit" for x in outc if not truth[x["booking_id"]])
    analyst = [x for x in store.all() if x["source"] == "simulated_analyst"]
    assert all((x["label"] == "fraud") != bool(truth[x["booking_id"]]) for x in analyst)  # 100% analyst error


def test_explored_rows_get_outcomes_with_propensity(tmp_path):
    pool, store, pipe, sim = _setup(tmp_path)
    sim.step(pipe, n=300, advance_days=61)
    ex = [x for x in store.all() if x["explored"]]
    assert ex and all(x["source"] == "simulated_outcome" and x["propensity"] == 0.05 for x in ex)


def test_stream_continues_in_time_order(tmp_path):
    pool, store, pipe, sim = _setup(tmp_path)
    a = sim.step(pipe, n=100)
    b = sim.step(pipe, n=100)
    assert b["cursor"] == 200 and _ts(b["now"]) >= _ts(a["now"])
    order = [r["booked_at"] for r in pipe.recs.values()]
    assert order == sorted(order)


def test_feature_vector_from_row_keeps_missing_features_so_serialize_works():
    import numpy as np
    import pandas as pd

    from fraudshield.features.serialize import serialize
    from fraudshield.learning.simulate import FEATURES, feature_vector_from_row

    df = pd.read_parquet(ROOT_FEATURES).head(50) if ROOT_FEATURES.exists() else None
    if df is None:
        import pytest
        pytest.skip("features not built")
    row = df.iloc[0].to_dict()
    row["base_consignees_l10"] = np.nan  # 6,538 real test-window rows look like this
    fv = feature_vector_from_row(row)
    assert "base_consignees_l10" in fv.values
    assert serialize(fv).startswith("BOOKING ")
