"""Red-team hardening from live attacks (fraudshield/redteam/harden.py): evasions become fraud labels only once the
original booking's truth is known; retrain goes through the unchanged learning service and gate."""
import json

import pytest
from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from fraudshield.learning.label_store import LabelStore
from fraudshield.redteam.harden import EvasionStore, MAX_PER_ATTACK, original_truth
from fraudshield.redteam.search import WhatIf, search
from tests.api import fakes
from tests.api.test_learning_routes import files  # noqa: F401  (module fixture: tiny models and pool)
from tests.policy.fixtures import booking_json, make_booking
from tests.redteam.test_search import dv_featurize, dv_gbm, pipeline


def test_original_truth_prefers_the_analyst_then_injected_ground_truth(tmp_path):
    ls = LabelStore(tmp_path / "labels.jsonl")
    rec = {"decision_id": "d1", "booking": {"meta": {"scenario": "T1"}}}
    assert original_truth(rec, ls) == {"label": "fraud", "source": "injected ground truth (T1)", "simulated": True}
    ls.append({"decision_id": "d1", "label": "legit", "source": "analyst", "features": {}})
    assert original_truth(rec, ls) == {"label": "legit", "source": "analyst", "simulated": False}
    assert original_truth({"decision_id": "d2", "booking": {"meta": {}}}, ls) is None


def test_harvest_waits_for_truth_then_promotes_minimal_evasions(tmp_path):
    pipe = pipeline(tmp_path)
    b = make_booking()  # stopped above R$1000 declared
    rec = pipe.score(b)
    res = search(WhatIf.for_pipeline(pipe), b, feedback="action", seed=0)
    ls = LabelStore(tmp_path / "labels.jsonl")
    st = EvasionStore(tmp_path / "evasions.jsonl")
    h = st.harvest(rec, res, WhatIf.for_pipeline(pipe), ls)
    assert 1 <= h["evasions"] <= MAX_PER_ATTACK and h["labelled"] == 0 and h["needs_confirmation"] is True
    assert len(ls) == 0 and st.summary(ls)["pending_decisions"] == [rec["decision_id"]]

    ls.append({"decision_id": rec["decision_id"], "label": "fraud", "source": "analyst", "features": {}})
    assert st.promote(ls) == h["evasions"]
    rows = [r for r in ls.all() if r["source"] == "redteam"]
    assert len(rows) == h["evasions"] and all(r["label"] == "fraud" and r["simulated"] is False for r in rows)
    assert all(r["features"]["declared_value"] <= 1000 for r in rows)  # the evading variants' own features
    assert all(r["decision_id"].startswith(rec["decision_id"] + "~rt") for r in rows)
    assert st.promote(ls) == 0  # never twice
    s = st.summary(ls)
    assert s["attacks"] == 1 and s["evasions"] == h["evasions"] and s["labelled"] == h["evasions"]
    assert s["pending_decisions"] == []


def test_a_legit_original_adds_no_labels(tmp_path):
    pipe = pipeline(tmp_path)
    b = make_booking()
    rec = pipe.score(b)
    ls = LabelStore(tmp_path / "labels.jsonl")
    ls.append({"decision_id": rec["decision_id"], "label": "legit", "source": "analyst", "features": {}})
    st = EvasionStore(tmp_path / "evasions.jsonl")
    h = st.harvest(rec, search(WhatIf.for_pipeline(pipe), b, feedback="action", seed=0), WhatIf.for_pipeline(pipe), ls)
    assert h["labelled"] == 0 and h["needs_confirmation"] is False and "legit" in h["note"]
    assert [r for r in ls.all() if r["source"] == "redteam"] == []


@pytest.fixture
def client(files, tmp_path):  # noqa: F811
    cfg = {"audit_path": str(tmp_path / "audit" / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
           "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
           "calibration_path": str(tmp_path / "nocal.json"),
           "learning": {"system": "B1", "tier": "R", "models_dir": str(files / "models"),
                        "registry_path": str(tmp_path / "registry.json"), "versions_dir": str(tmp_path / "versions"),
                        "features_path": str(files / "features.parquet"),
                        "bookings_path": str(files / "bookings.parquet"), "bootstrap_B": 30, "new_pattern_glob": None,
                        "max_fpr_hn_increase": 0.10}}
    c = TestClient(build_app(cfg, components={"featurize": dv_featurize, "serialize": fakes.fake_serialize,
                                              "gbm": dv_gbm, "laya": fakes.DownLaya(), "llm_client": None}))
    c.tmp = tmp_path
    return c


def test_attack_confirm_retrain_flow(client):
    assert client.post("/redteam/retrain").status_code == 409  # nothing harvested yet
    unknown = booking_json(booking_id="rt-a", meta={})  # no injected scenario: truth unknown until an analyst says
    did = client.post("/score", json=unknown).json()["decision_id"]
    a = client.post(f"/decisions/{did}/redteam").json()
    assert a["harvest"]["evasions"] >= 1 and a["harvest"]["needs_confirmation"] is True
    hs = client.get("/redteam/hardening").json()
    assert hs["attacks"] == 1 and hs["labelled"] == 0 and hs["pending_decisions"] == [did]

    client.post(f"/decisions/{did}/analyst", json={"label": "fraud", "note": "red team: original confirmed"})
    hs = client.get("/redteam/hardening").json()
    assert hs["labelled"] == a["harvest"]["evasions"] and hs["labelled_since_last_retrain"] == hs["labelled"]

    inj = booking_json(booking_id="rt-b", meta={"scenario": "T1"})
    b = client.post(f"/decisions/{client.post('/score', json=inj).json()['decision_id']}/redteam").json()
    assert b["harvest"]["labelled"] == b["harvest"]["evasions"] and b["harvest"]["simulated"] is True

    r = client.post("/redteam/retrain")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["handover"]["verdict"] in ("new_model_in_use", "previous_model_kept")
    assert out["redteam_labels"] == hs["labelled"] + b["harvest"]["labelled"]
    hs = client.get("/redteam/hardening").json()
    assert hs["labelled_since_last_retrain"] == 0 and hs["last_retrain"]["run_id"] == out["run_id"]
    audit = [json.loads(x)["event_type"] for x in (client.tmp / "audit" / "audit.jsonl").read_text().splitlines()]
    assert "redteam_attack" in audit and "retrain" in audit
    assert client.get("/audit/verify").json()["ok"] is True
