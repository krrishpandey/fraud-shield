from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from tests.api.test_app import components
from tests.policy.fixtures import booking_json


def test_claims_route_template_claims_and_attribution_status(tmp_path):
    cfg = {"audit_path": str(tmp_path / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
           "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
           "calibration_path": str(tmp_path / "nocal.json")}
    c = TestClient(build_app(cfg, components=components()))
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    d = c.get(f"/decisions/{did}").json()
    assert d["explanation"]["mode"] == "template" and d["explanation"]["claims"]["claims_source"] == "template"
    v = c.get(f"/decisions/{did}/claims").json()
    assert v["decision_id"] == did and v["mode"] == "template" and v["claims_source"] == "template"
    assert [x["reason_code"] for x in v["claims"]] == d["reasons"][:3]
    assert all(x["ok"] for x in v["claims"]) and v["ok"] is True
    # the fake scorer is not a LightGBM model: the page says so instead of inventing an agreement
    assert v["attribution"]["available"] is False and v["attribution"]["why"]
    assert c.get("/decisions/nope/claims").status_code == 404
