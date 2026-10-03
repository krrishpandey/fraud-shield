"""What-if search (fraudshield/redteam/search.py) and GET /decisions/{id}/counterfactual."""
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from fraudshield.api.pipeline import Pipeline
from fraudshield.audit.log import AuditLog
from fraudshield.contracts import FeatureVector
from fraudshield.features.store import FeatureStore
from fraudshield.redteam.search import (BUDGET, KG_SLOPE_BRL, Change, HistoryView, WhatIf, apply_changes,
                                        counterfactual_payload, search)
from tests.api import fakes
from tests.policy.fixtures import booking_json, make_booking

ROOT = Path(__file__).resolve().parents[2]


def dv_featurize(b):
    return FeatureVector(b.booking_id, b.booked_at, {"declared_value": b.declared_value, "tenure_days": 412.0, "cost_vs_median": 1.0})


def dv_gbm(fv):
    """Risky only above R$1000 declared value."""
    return 0.9 if fv.values["declared_value"] > 1000 else 0.001


def pipeline(tmp_path, gbm=dv_gbm):
    return Pipeline(dv_featurize, fakes.fake_serialize, gbm, fakes.DownLaya(), AuditLog(tmp_path / "a.jsonl"))


def test_finds_minimal_declared_value_change_and_refines_it(tmp_path):
    pipe = pipeline(tmp_path)
    b = make_booking()  # declared 1450
    rec = pipe.score(b)
    assert rec["action"] not in ("allow", "allow_scan_gated")
    res = search(WhatIf.for_pipeline(pipe), b)
    assert res.original.action == rec["action"]
    best = res.best["allow"]
    assert best is not None and len(best.changes) == 1 and best.changes[0].field == "declared_value"
    assert 725.0 <= best.changes[0].after <= 1000.0  # x0.5 found, then bisection toward no change
    assert res.evaluations <= BUDGET
    body = counterfactual_payload(res)
    assert body["found"] and body["analyst_only"] and body["counterfactuals"][0]["action"] == "allow"
    assert "declared value were R$" in body["counterfactuals"][0]["summary"]


def test_search_is_pure_no_audit_no_store(tmp_path):
    pipe = pipeline(tmp_path)
    b = make_booking()
    pipe.score(b)
    n_dec, n_audit = len(pipe.list()), len((tmp_path / "a.jsonl").read_text().splitlines())
    search(WhatIf.for_pipeline(pipe), b)
    assert len(pipe.list()) == n_dec
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == n_audit


def test_budget_respected_and_closest_reported_when_nothing_flips(tmp_path):
    pipe = pipeline(tmp_path, gbm=lambda fv: 0.9)
    b = make_booking()
    pipe.score(b)
    res = search(WhatIf.for_pipeline(pipe), b)
    assert res.evaluations <= BUDGET and not res.flipped("softer")
    body = counterfactual_payload(res)
    assert body["found"] is False and "No change" in body["message"] and body["closest"] is not None


def test_action_only_attacker_also_flips(tmp_path):
    pipe = pipeline(tmp_path)
    b = make_booking()
    pipe.score(b)
    res = search(WhatIf.for_pipeline(pipe), b, feedback="action", seed=3)
    assert res.flipped("allow") and res.evaluations <= BUDGET


def test_already_allowed_needs_no_search(tmp_path):
    pipe = pipeline(tmp_path)
    b = make_booking(declared_value=100.0)
    assert pipe.score(b)["action"] == "allow"
    res = search(WhatIf.for_pipeline(pipe), b)
    assert res.evaluations == 0 and counterfactual_payload(res)["message"].startswith("This booking is already")


def test_weight_change_moves_carrier_cost_by_lane_slope():
    b = make_booking(weight_kg=10.0, carrier_cost=50.0)
    nb = apply_changes(b, (Change("weight_kg", 10.0, 8.0, math.log(1.25), 0.8),))
    assert nb.weight_kg == 8.0 and nb.carrier_cost == pytest.approx(50.0 - 2 * KG_SLOPE_BRL, abs=0.01)
    floor = apply_changes(b, (Change("weight_kg", 10.0, 0.1, 5.0, 0.01),))
    assert floor.carrier_cost >= 1.0
    assert apply_changes(b, (Change("service", "express", "standard", 1.0),)).carrier_cost == 50.0


def test_history_view_hides_the_booking_and_never_appends(toy_bookings):
    toy_bookings = toy_bookings.assign(channel="web", login_device_age_days=100.0, payment_method="account_billing",
                                       owner_contact_age_days=100.0)
    store = FeatureStore.from_frame(toy_bookings)
    acc = toy_bookings.account_id.iloc[0]
    last = toy_bookings[toy_bookings.account_id == acc].iloc[-1]
    n = len(store)
    v = HistoryView(store, {last.booking_id})
    h = v.history(acc, "2030-01-01")
    assert last.booking_id not in set(h.booking_id) and len(h) == (toy_bookings.account_id == acc).sum() - 1
    assert v.history(acc, "2030-01-01") is h  # memoised
    assert len(store) == n


def test_counterfactual_endpoint_is_audited(tmp_path):
    cfg = {"audit_path": str(tmp_path / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
           "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
           "calibration_path": str(tmp_path / "nocal.json"), "learning": {"enabled": False}}
    c = TestClient(build_app(cfg, components={"featurize": dv_featurize, "serialize": fakes.fake_serialize,
                                              "gbm": dv_gbm, "laya": fakes.DownLaya(), "llm_client": None}))
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    r = c.get(f"/decisions/{did}/counterfactual")
    assert r.status_code == 200
    body = r.json()
    assert body["analyst_only"] is True and body["found"] is True and body["evaluations"] <= BUDGET
    assert body["counterfactuals"][0]["changes"][0]["field"] == "declared_value"
    assert body["latency_ms"] >= 0
    last = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["event_type"] == "counterfactual_view" and last["record_hash"] == body["audit_hash"]
    assert last["payload"]["decision_id"] == did and last["payload"]["shown"]
    assert c.get("/decisions/nope/counterfactual").status_code == 404
    assert c.get("/audit/verify").json()["ok"] is True


DEMO = ROOT / "data/processed/demo_bookings.json"


@pytest.mark.skipif(not DEMO.exists(), reason="demo data not built")
def test_what_if_matches_score_on_real_components(tmp_path):
    """The pure path gives the same action and probability as /score (real store, model, calibration, cost rule)."""
    import yaml

    from fraudshield.api import real_components as rc
    from fraudshield.contracts import Booking
    rc.reset()
    cfg = yaml.safe_load((ROOT / "e2e/app.e2e.yaml").read_text(encoding="utf-8"))
    cfg.update({"audit_path": str(tmp_path / "audit.jsonl"), "web_dist": None, "replay_path": None})
    cfg["learning"] = {"enabled": False}
    cfg["policy"]["explore_eps"] = 0.05  # production exploration: the what-if must reproduce the same draw
    pipe = build_app(cfg).state.pipeline
    wi = WhatIf.for_pipeline(pipe)
    items = rc.stream_source(0)
    sample = [Booking(**x["booking"]) for x in json.loads(DEMO.read_text())]
    sample += [it.booking for it in items if it.is_fraud][:15] + [it.booking for it in items[:10]]
    n_store = rc.store_size()
    for b in sample:
        out = pipe.score(b)
        got = wi.evaluate([b], wi.view(b))[0]
        assert got.action == out["action"], b.booking_id
        assert round(got.probability, 4) == out["probabilities"]["misuse"], b.booking_id
    n_store_scored = rc.store_size()
    assert n_store_scored - n_store == 5  # only the demo bookings were new to the store (appended by /score)
    takeover = sample[2]
    n_audit = len((tmp_path / "audit.jsonl").read_text().splitlines())
    res = search(wi, takeover)
    assert res.evaluations <= BUDGET
    assert rc.store_size() == n_store_scored  # the search never appends
    assert len((tmp_path / "audit.jsonl").read_text().splitlines()) == n_audit
    rc.reset()
