"""/learning/* routes (docs/API.md v1.2) on small synthetic data (no real artifacts needed)."""
import json

import pytest
from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from tests.api.test_app import components
from tests.learning.helpers import fit_base, make_pool, make_reference
from tests.policy.fixtures import booking_json


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("lr")
    df = make_reference()
    fit_base(df, d / "models")
    make_pool(df).drop(columns=["sender_id", "consignee_id"]).to_parquet(d / "features.parquet", index=False)
    make_pool(df)[["booking_id", "sender_id", "consignee_id"]].to_parquet(d / "bookings.parquet", index=False)
    return d


@pytest.fixture
def client(files, tmp_path):
    cfg = {"audit_path": str(tmp_path / "audit" / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
           "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
           "calibration_path": str(tmp_path / "nocal.json"),
           "learning": {"system": "B1", "tier": "R", "models_dir": str(files / "models"),
                        "registry_path": str(tmp_path / "registry.json"),
                        "versions_dir": str(tmp_path / "versions"),
                        "features_path": str(files / "features.parquet"),
                        "bookings_path": str(files / "bookings.parquet"), "bootstrap_B": 30, "new_pattern_glob": None,
                        # route wiring test: with only ~18 synthetic hard negatives one booking moves FPR by
                        # 5.6 points; gate thresholds themselves are tested in tests/learning
                        "max_fpr_hn_increase": 0.10}}
    c = TestClient(build_app(cfg, components=components()))
    c.tmp = tmp_path
    return c


def test_status_shape(client):
    for prefix in ("", "/api"):
        st = client.get(f"{prefix}/learning/status").json()
        assert {"active_version", "labels_since_last_retrain", "label_sources", "versions", "calibration_version",
                "laya_export"} <= set(st)
        assert st["active_version"] == "gbm-B1-R-v1"
        assert set(st["versions"][0]) >= {"version", "created_at", "parent", "n_train", "n_feedback_labels",
                                          "metrics", "deployed"}
        assert set(st["laya_export"]) == {"path", "rows"}


def test_analyst_label_reaches_label_store(client):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    client.post(f"/decisions/{did}/analyst", json={"label": "fraud", "note": "confirmed"})
    st = client.get("/learning/status").json()
    assert st["label_sources"] == {"analyst": 1} and st["labels_since_last_retrain"] == 1
    lines = (client.tmp / "feedback" / "labels.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["features"]["cost_vs_median"] == 11.6


def test_simulate_retrain_rollback_flow(client):
    r = client.post("/api/learning/simulate_feedback", json={"n": 900, "seed": 1, "mode": "uniform_noisy"})
    assert r.status_code == 200
    sim = r.json()
    assert sim["added"] == 900 and sim["simulated"] is True and sim["fraud"] + sim["legit"] == 900
    r = client.post("/learning/retrain", json={"min_new_labels": 20})
    assert r.status_code == 200, r.text
    out = r.json()
    assert {"run_id", "n_new_labels", "n_fraud", "n_legit", "eval_set", "current", "candidate", "gate",
            "deployed_version", "audit_hash"} <= set(out)
    for k in ("pr_auc", "ece", "cost_per_1k_brl", "fpr_hard_negative", "recall_new_pattern", "version"):
        assert k in out["current"] and k in out["candidate"]
    assert {"n", "description"} <= set(out["eval_set"])
    assert all({"name", "passed", "detail"} <= set(c) for c in out["gate"]["checks"])
    assert out["deployed_version"] == "gbm-B1-R-v2", json.dumps(out["gate"], indent=1) + json.dumps(out["current"]) + json.dumps(out["candidate"])
    # new decisions are scored by v2
    assert client.post("/score", json=booking_json(booking_id="x2")).json()["model_versions"]["gbm"] == "gbm-B1-R-v2"
    rb = client.post("/learning/rollback", json={"version": "gbm-B1-R-v1"})
    assert rb.status_code == 200 and rb.json()["active_version"] == "gbm-B1-R-v1" and rb.json()["audit_hash"]
    assert client.get("/learning/status").json()["active_version"] == "gbm-B1-R-v1"
    assert client.post("/learning/rollback", json={"version": "nope"}).status_code == 404
    assert client.get("/audit/verify").json()["ok"] is True


def test_retrain_with_too_few_labels_is_409(client):
    r = client.post("/learning/retrain", json={"min_new_labels": 20})
    assert r.status_code == 409 and r.json() == {"detail": "Need at least 20 new labels, have 0"}


def test_ask_route_returns_audit_hash(client):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    r = client.post(f"/decisions/{did}/ask", json={"instructions": "Is it a drop?"}).json()
    lines = (client.tmp / "audit" / "audit.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["event_type"] == "ask" and json.loads(lines[-1])["record_hash"] == r["audit_hash"]


def test_restart_scores_with_deployed_version(files, tmp_path):
    cfg = {"audit_path": str(tmp_path / "audit" / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
           "web_dist": str(tmp_path / "nodist"), "calibration_path": str(tmp_path / "nocal.json"),
           "learning": {"system": "B1", "tier": "R", "models_dir": str(files / "models"),
                        "registry_path": str(tmp_path / "registry.json"), "versions_dir": str(tmp_path / "versions"),
                        "features_path": str(files / "features.parquet"),
                        "bookings_path": str(files / "bookings.parquet"), "bootstrap_B": 30, "new_pattern_glob": None,
                        "max_fpr_hn_increase": 0.10, "wire_active": True}}
    c = TestClient(build_app(cfg, components=components()))
    c.post("/learning/simulate_feedback", json={"n": 900, "seed": 1, "mode": "uniform_noisy"})
    assert c.post("/learning/retrain", json={}).json()["deployed_version"] == "gbm-B1-R-v2"
    comps = {k: v for k, v in components().items() if k != "gbm"}  # real (non-injected) gbm slot
    c2 = TestClient(build_app(cfg, components=comps))
    h = c2.get("/health").json()
    assert h["components"]["gbm"] == "learning registry (gbm-B1-R-v2)" and h["versions"]["gbm"] == "gbm-B1-R-v2"
    assert c2.get("/learning/status").json()["active_version"] == "gbm-B1-R-v2"


def test_realistic_simulate_route_and_gate_param(client):
    r = client.post("/learning/simulate_feedback", json={"n": 300, "seed": 1, "advance_days": 61}).json()
    assert r["mode"] == "realistic" and r["simulated"] is True and "simulated_analyst" in r["by_source"]
    assert client.post("/learning/simulate_feedback", json={"mode": "bogus"}).status_code == 422
    assert client.post("/learning/retrain", json={"gate": "bogus"}).status_code == 422


def test_wiring_reads_training_options(files, tmp_path):
    from fraudshield.api.app import ROOT
    from fraudshield.learning.wiring import build_learning
    from tests.learning.helpers import make_pipeline
    pipe = make_pipeline(tmp_path)
    svc = build_learning({"learning": {"system": "B1", "tier": "R", "models_dir": str(files / "models"),
                                       "max_feedback_share": 0.3, "ipw_clip": 5,
                                       "lgb_params": {"lambda_l2": 10}}},
                         pipe, pipe.audit, tmp_path / "audit.jsonl", lambda p: None if p is None else ROOT / p)
    assert svc.train_options == {"max_feedback_share": 0.3, "ipw_clip": 5.0, "lgb_params": {"lambda_l2": 10}}
