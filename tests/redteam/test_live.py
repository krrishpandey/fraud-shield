"""Live red-team demo (fraudshield/redteam/live.py): POST /decisions/{id}/redteam, GET /redteam/results and targets."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from fraudshield.redteam.live import attack_payload, load_results, load_targets
from fraudshield.redteam.search import BUDGET, WhatIf, search
from tests.api import fakes
from tests.policy.fixtures import booking_json, make_booking
from tests.redteam.test_search import dv_featurize, dv_gbm, pipeline

ROOT = Path(__file__).resolve().parents[2]


def test_attack_payload_lists_every_try_in_order_and_shows_the_attacker_only_actions(tmp_path):
    pipe = pipeline(tmp_path)
    b = make_booking()  # declared R$1450: stopped; below R$1000 it is allowed
    pipe.score(b)
    out = attack_payload(search(WhatIf.for_pipeline(pipe), b, feedback="action", seed=0))
    assert out["attacker_sees"] == "action only" and out["budget"] == BUDGET and out["max_fields"] == 2
    tries = out["attempts"]
    assert [t["n"] for t in tries] == list(range(1, len(tries) + 1)) and len(tries) == out["queries"] <= BUDGET
    assert all("probability" not in t for t in tries)  # the attacker never sees the risk score
    first = next(t["n"] for t in tries if t["allow"])
    assert out["evaded"] == "allow" and out["first_allow_at"] == first
    assert out["evasion"]["action"] == "allow" and out["evasion"]["changes"][0]["field"] == "declared_value"
    assert out["original_action"] not in ("allow", "allow_scan_gated")
    assert out["best_action"] == "allow" and out["still_stopped"] is False


def test_attack_payload_when_the_model_holds(tmp_path):
    pipe = pipeline(tmp_path, gbm=lambda fv: 0.95)  # risky whatever the booker changes
    b = make_booking()
    pipe.score(b)
    out = attack_payload(search(WhatIf.for_pipeline(pipe), b, feedback="action", seed=0))
    assert out["evaded"] is None and out["evasion"] is None and out["first_softer_at"] is None
    assert out["queries"] == BUDGET and "resisted" in out["message"]
    assert out["best_action"] is None and out["still_stopped"] is True


def test_load_results_summarises_the_measured_run(tmp_path):
    s = load_results(ROOT / "artifacts" / "results_redteam.json")
    assert s["trained"]["all"]["n"] == 320 and abs(s["trained"]["all"]["flip_allow_mean"] - 0.181) < 0.001
    assert {r["type"] for r in s["by_type"]} >= {"T1", "T2", "T4", "T7"}
    assert s["hardening"]["passed"] is False
    assert [c["name"] for c in s["hardening"]["failed_checks"]] == ["fpr_hard_negative_noninferior"]
    assert s["split"] == "test" and s["budget"] == 50
    assert load_results(tmp_path / "missing.json") is None


def test_load_targets_keeps_only_bookings(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([booking_json(booking_id="t1"), {"not": "a booking"}]), encoding="utf-8")
    t = load_targets(p)
    assert [x["booking"]["booking_id"] for x in t] == ["t1"] and t[0]["title"]
    assert load_targets(tmp_path / "missing.json") == []


def _client(tmp_path, gbm=dv_gbm):
    res = tmp_path / "rt.json"
    res.write_text((ROOT / "artifacts" / "results_redteam.json").read_text(encoding="utf-8"), encoding="utf-8")
    tg = tmp_path / "targets.json"
    tg.write_text(json.dumps([booking_json(booking_id="tgt-1")]), encoding="utf-8")
    cfg = {"audit_path": str(tmp_path / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
           "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
           "calibration_path": str(tmp_path / "nocal.json"), "learning": {"enabled": False},
           "redteam": {"results_path": str(res), "targets_path": str(tg)}}
    return TestClient(build_app(cfg, components={"featurize": dv_featurize, "serialize": fakes.fake_serialize,
                                                 "gbm": gbm, "laya": fakes.DownLaya(), "llm_client": None}))


def test_attack_endpoint_is_audited(tmp_path):
    c = _client(tmp_path)
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    r = c.post(f"/decisions/{did}/redteam")
    assert r.status_code == 200
    body = r.json()
    assert body["decision_id"] == did and body["evaded"] == "allow" and body["attempts"]
    last = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["event_type"] == "redteam_attack" and last["record_hash"] == body["audit_hash"]
    assert last["payload"]["decision_id"] == did and last["payload"]["queries"] == body["queries"]
    assert c.post("/decisions/nope/redteam").status_code == 404
    assert c.get("/audit/verify").json()["ok"] is True


def test_results_and_targets_endpoints(tmp_path):
    c = _client(tmp_path)
    s = c.get("/redteam/results").json()
    assert s["trained"]["all"]["n"] == 320 and s["hardening"]["passed"] is False
    t = c.get("/redteam/targets").json()
    assert [x["booking"]["booking_id"] for x in t] == ["tgt-1"]
