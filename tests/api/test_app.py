from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from tests.api import fakes
from tests.policy.fixtures import booking_json


def components(**kw):
    c = {"featurize": fakes.fake_featurize, "serialize": fakes.fake_serialize, "gbm": fakes.fake_gbm,
         "laya": fakes.laya(), "llm_client": None}
    c.update(kw)
    return c


@pytest.fixture
def cfg(tmp_path):
    return {"audit_path": str(tmp_path / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
            "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
            "calibration_path": str(tmp_path / "nocal.json")}


@pytest.fixture
def client(cfg):
    return TestClient(build_app(cfg, components=components()))


def test_score_and_idempotency(client):
    r1 = client.post("/score", json=booking_json())
    assert r1.status_code == 200
    body = r1.json()
    assert body["explanation_status"] == "pending" and body["decision_id"].startswith("dec_")
    r2 = client.post("/score", json=booking_json())
    assert r2.json()["decision_id"] == body["decision_id"]
    assert client.get("/audit/verify").json()["records"] == 2  # decision + explanation, not duplicated


def test_score_validation_error(client):
    bad = booking_json()
    del bad["carrier_cost"]
    assert client.post("/score", json=bad).status_code == 422


def test_decision_detail_has_booking_and_explanation(client):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    d = client.get(f"/decisions/{did}").json()
    assert d["booking"]["booking_id"] == "demo-takeover-01"
    assert d["explanation"]["source"] == "template" and d["explanation_status"] == "ready"
    assert d["analyst"] is None
    assert client.get("/decisions/nope").status_code == 404


def test_list_decisions_filters_and_summary_shape(client):
    client.post("/score", json=booking_json(booking_id="a1"))
    client.post("/score", json=booking_json(booking_id="a2"))
    lst = client.get("/decisions?limit=1").json()
    assert len(lst) == 1 and lst[0]["booking_id"] == "a2"
    assert set(lst[0]) == {"decision_id", "booking_id", "account_id", "booked_at", "action", "misuse",
                           "carrier_cost", "analyst_label", "scenario"}
    act = lst[0]["action"]
    assert all(x["action"] == act for x in client.get(f"/decisions?action={act}").json())
    assert client.get("/decisions?action=allow").json() == [] or act == "allow"


def test_analyst_feedback_is_audited(client, cfg):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    r = client.post(f"/decisions/{did}/analyst", json={"label": "fraud", "note": "owner denied"})
    assert r.status_code == 200 and r.json()["ok"] is True
    lines = [json.loads(x) for x in open(cfg["audit_path"], encoding="utf-8").read().splitlines()]
    assert lines[-1]["event_type"] == "analyst_feedback" and lines[-1]["record_hash"] == r.json()["audit_hash"]
    assert client.get(f"/decisions/{did}").json()["analyst"]["label"] == "fraud"
    assert client.post(f"/decisions/{did}/analyst", json={"label": "maybe"}).status_code == 422
    assert client.post("/decisions/zzz/analyst", json={"label": "legit"}).status_code == 404


def test_ask_custom_question(client):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    r = client.post(f"/decisions/{did}/ask", json={"instructions": "Is the consignee a drop?", "yes": "drop",
                                                   "no": "ordinary"})
    assert r.status_code == 200
    j = r.json()
    assert j["qid"] == "custom_1" and j["calibrated"] is False
    assert set(j["raw_probabilities"]) == {"a", "b"} and 0 <= j["probability_yes"] <= 1


def test_ask_when_laya_down_is_503(cfg):
    c = TestClient(build_app(cfg, components=components(laya=fakes.DownLaya())))
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    assert c.post(f"/decisions/{did}/ask", json={"instructions": "x", "yes": "y", "no": "n"}).status_code == 503


def test_audit_verify_detects_tamper(client, cfg):
    client.post("/score", json=booking_json())
    assert client.get("/audit/verify").json()["ok"] is True
    p = cfg["audit_path"]
    txt = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(txt.replace('"action":"', '"action":"x', 1))
    v = client.get("/audit/verify").json()
    assert v["ok"] is False and v["first_bad_index"] == 0


def test_dashboard_metrics_from_store(client):
    assert client.get("/dashboard/metrics").json()["totals"]["bookings"] == 0
    client.post("/score", json=booking_json())
    m = client.get("/dashboard/metrics").json()
    assert m["totals"]["bookings"] == 1 and m["latency"]["p50_ms"] is not None


def test_health_reports_components_and_mode(client):
    h = client.get("/health").json()
    assert h["ok"] is True and h["laya_mode"] == "local"
    assert "gpu" in h and "versions" in h
    assert h["calibrated"] is False
    assert any("calibration" in w for w in h["warnings"])


def test_replay_file_loaded_at_startup(cfg, tmp_path):
    rp = tmp_path / "replay.jsonl"
    recs = [{"decision_id": "dec_r1", "booking_id": "r1", "account_id": "acc", "booked_at": "2018-06-01T10:00:00",
             "action": "hold", "probabilities": {"misuse": 0.9}, "label": "fraud",
             "booking": {"carrier_cost": 50.0, "meta": {"scenario": "T1"}}, "latency_ms": {"total": 120.0}}]
    rp.write_text("\n".join(json.dumps(r) for r in recs))
    c = TestClient(build_app({**cfg, "replay_path": str(rp)}, components=components()))
    assert c.get("/decisions").json()[0]["scenario"] == "T1"
    m = c.get("/dashboard/metrics").json()
    assert m["totals"]["bookings"] == 1 and m["revenue_loss_prevented_brl"] > 0
    assert c.get("/health").json()["replay"]["loaded"] == 1


def test_demo_bookings(cfg, tmp_path):
    c = TestClient(build_app(cfg, components=components()))
    assert c.get("/demo/bookings").json() == []
    wrapped = [{"scenario": "takeover", "title": "Takeover", "description": "d", "expected": "hold",
                "booking": booking_json()}]
    (tmp_path / "demo.json").write_text(json.dumps(wrapped))
    assert c.get("/demo/bookings").json() == wrapped
    (tmp_path / "demo.json").write_text(json.dumps([booking_json()]))  # bare bookings get wrapped
    item = c.get("/demo/bookings").json()[0]
    assert set(item) == {"scenario", "title", "description", "expected", "booking"}
    assert item["booking"]["booking_id"] == "demo-takeover-01" and item["scenario"] == "T1"


def test_cors_for_vite(client):
    r = client.options("/score", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_static_spa_served_after_api_routes(cfg, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>console</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    c = TestClient(build_app({**cfg, "web_dist": str(dist)}, components=components()))
    assert "console" in c.get("/").text
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert "console" in c.get("/some/spa/route").text  # SPA fallback
    assert c.get("/health").json()["ok"] is True  # API wins over static
    assert c.get("/api/health").json()["ok"] is True  # same routes also under /api


def test_fallback_components_when_nothing_injected(cfg):
    c = TestClient(build_app({**cfg, "laya": {"mode": "cached", "cache_path": cfg["replay_path"]}}))
    h = c.get("/health").json()
    assert h["components"]["gbm"].startswith("fallback")
    r = c.post("/score", json=booking_json())
    assert r.status_code == 200 and r.json()["degraded"] is True  # empty cache -> degraded path


def test_reason_codes_endpoint(client):
    from fraudshield.policy.reasons import REASON_RULES
    rc = client.get("/reason-codes").json()
    assert set(rc) == {r.code for r in REASON_RULES}
    assert all(isinstance(v, str) and "{" not in v for v in rc.values())


def test_ask_yes_no_optional(client):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    assert client.post(f"/decisions/{did}/ask", json={"instructions": "Is it odd?"}).status_code == 200


def test_health_degraded_flag(cfg):
    c = TestClient(build_app(cfg, components=components(laya=fakes.DownLaya())))
    c.post("/score", json=booking_json())
    assert c.get("/health").json()["degraded"] is True
    c2 = TestClient(build_app(cfg, components=components()))
    assert c2.get("/health").json()["degraded"] is False


def test_explanation_failed_status(tmp_path):
    from fraudshield.api.pipeline import Pipeline
    from fraudshield.audit.log import AuditLog
    from tests.policy.fixtures import make_booking

    def bad_explainer(rec):
        raise RuntimeError("boom")
    p = Pipeline(fakes.fake_featurize, fakes.fake_serialize, fakes.fake_gbm, fakes.laya(),
                 AuditLog(tmp_path / "a.jsonl"), explainer=bad_explainer)
    did = p.score(make_booking())["decision_id"]
    p.explain_decision(did)
    assert p.get(did)["explanation_status"] == "failed"


def test_account_story_route_uses_story_component(cfg):
    seen = []

    def story(booking):
        seen.append(booking.booking_id)
        return {"account_id": booking.account_id, "bookings": [], "last10": {}, "usual": None, "n_prior": 0}

    c = TestClient(build_app(cfg, components=components(story=story)))
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    r = c.get(f"/decisions/{did}/account-story")
    assert r.status_code == 200 and r.json()["account_id"] == "acc_7c3e" and seen == ["demo-takeover-01"]
    assert c.get("/decisions/nope/account-story").status_code == 404


def test_account_story_without_component_is_503(client):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    assert client.get(f"/decisions/{did}/account-story").status_code == 503


def test_explanation_check_runs_the_validator(client):
    did = client.post("/score", json=booking_json()).json()["decision_id"]
    text = client.get(f"/decisions/{did}").json()["explanation"]["text"]
    ok = client.post(f"/decisions/{did}/explanation/check", json={"text": text}).json()
    assert ok["ok"] is True and ok["numbers"] and all(n["ok"] for n in ok["numbers"])
    bad = client.post(f"/decisions/{did}/explanation/check", json={"text": text + " Loss R$98765."}).json()
    assert bad["ok"] is False and [n["text"] for n in bad["numbers"] if not n["ok"]] == ["98765"]
    assert client.post("/decisions/nope/explanation/check", json={"text": "x"}).status_code == 404



ASK = {"instructions": "Is the consignee a drop?", "yes": "drop", "no": "ordinary"}


def _cached_empty(tmp_path):
    from fraudshield.models.laya_client import LayaClient
    return LayaClient.cached(tmp_path / "no_cache.json", mode_reason="test: no fine-tuned model")


def test_ask_uses_the_ask_model_while_the_decision_model_is_not_live(cfg, tmp_path):
    from fraudshield.models.laya_client import LazyLaya
    stock = LazyLaya(lambda: fakes.laya(), label="stock Laya (not fine-tuned)")
    c = TestClient(build_app(cfg, components=components(laya=_cached_empty(tmp_path), ask_laya=stock)))
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    r = c.post(f"/decisions/{did}/ask", json=ASK)
    assert r.status_code == 200 and r.json()["answered_by"] == "stock Laya (not fine-tuned)"
    assert c.get("/health").json()["ask_model"]["label"] == "stock Laya (not fine-tuned)"


def test_ask_uses_the_decision_model_once_it_is_live(cfg):
    from fraudshield.models.laya_client import LazyLaya
    never = LazyLaya(lambda: (_ for _ in ()).throw(AssertionError("stock model must not load")), label="stock")
    c = TestClient(build_app(cfg, components=components(ask_laya=never)))  # fake live (local) Laya
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    r = c.post(f"/decisions/{did}/ask", json=ASK)
    assert r.status_code == 200 and "not fine-tuned" not in r.json()["answered_by"]
    assert never.state() == "not loaded"


def test_auto_mode_without_a_fine_tuned_checkpoint_decides_from_the_cache(cfg, monkeypatch, tmp_path):
    from fraudshield.api import app as A

    def no_local(*a, **k):
        raise AssertionError("stock Laya must not be loaded for decisions")

    monkeypatch.setattr(A.LayaClient, "local", staticmethod(no_local))
    monkeypatch.setattr(A.LayaClient, "auto", classmethod(lambda cls, *a, **k: no_local()))
    cfg = {**cfg, "laya": {"mode": "auto", "model_path": "convaiinnovations/laya", "device": "cuda",
                           "cache_path": str(tmp_path / "c.json")}}
    comps = {"featurize": fakes.fake_featurize, "serialize": fakes.fake_serialize, "gbm": fakes.fake_gbm,
             "llm_client": None}  # no "laya": the app builds it from the config
    h = TestClient(build_app(cfg, components=comps)).get("/health").json()
    assert h["laya_mode"] == "cached" and "fine-tuned" in h["laya_reason"]


def test_auto_mode_with_a_fine_tuned_checkpoint_loads_it(monkeypatch, tmp_path):
    from fraudshield.api import app as A
    ck = tmp_path / "artifacts" / "laya" / "fraudshield-laya"
    ck.mkdir(parents=True)
    seen = {}

    def fake_auto(cls, model_path, cache_path, **kw):
        seen["path"] = model_path
        return "fine-tuned client"

    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A.LayaClient, "auto", classmethod(fake_auto))
    cfg = {"laya": {"mode": "auto", "model_path": "artifacts/laya/fraudshield-laya", "device": "cuda", "cache_path": None}}
    assert A._build_laya(cfg, []) == "fine-tuned client" and Path(seen["path"]) == ck


def test_decide_false_keeps_decisions_on_the_backup_model_even_with_a_laya_cache(cfg, tmp_path):
    import json as _json
    cache = tmp_path / "laya_cache.json"
    cache.write_text(_json.dumps({"some": {"misuse": {"a": 0.9, "b": 0.1}}}))
    comps = {"featurize": fakes.fake_featurize, "serialize": fakes.fake_serialize, "gbm": fakes.fake_gbm, "llm_client": None}
    cfg = {**cfg, "laya": {"mode": "cached", "decide": False, "cache_path": str(cache)}}
    c = TestClient(build_app(cfg, components=comps))
    h = c.get("/health").json()
    assert h["laya_mode"] == "cached" and "LightGBM" in h["laya_reason"]
    assert c.post("/score", json=booking_json()).json()["degraded"] is True


def test_question_model_is_labelled_fine_tuned_when_it_loads_a_local_checkpoint(cfg, tmp_path, monkeypatch):
    from fraudshield.api import app as A
    ck = tmp_path / "fraudshield-laya"
    ck.mkdir()
    monkeypatch.setattr(A, "cuda_available", lambda: True)
    fake = fakes.laya()  # built before patching LayaClient.local, which fakes.laya() itself uses
    monkeypatch.setattr(A.LayaClient, "local", staticmethod(lambda *a, **k: fake))
    comps = {"featurize": fakes.fake_featurize, "serialize": fakes.fake_serialize, "gbm": fakes.fake_gbm, "llm_client": None}
    cfg = {**cfg, "laya": {"mode": "cached", "decide": False, "cache_path": str(tmp_path / "none.json")},
           "laya_ask": {"enabled": True, "model_path": str(ck), "device": "cuda", "timeout_s": 3.0}}
    c = TestClient(build_app(cfg, components=comps))
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    r = c.post(f"/decisions/{did}/ask", json=ASK)
    assert r.status_code == 200 and r.json()["answered_by"] == "fine-tuned Laya"


def test_question_model_falls_back_to_stock_when_the_fine_tuned_folder_is_missing(cfg, tmp_path, monkeypatch):
    from fraudshield.api import app as A
    seen = {}
    fake = fakes.laya()

    def fake_local(path, *a, **k):
        seen["path"] = path
        return fake
    monkeypatch.setattr(A, "cuda_available", lambda: True)
    monkeypatch.setattr(A.LayaClient, "local", staticmethod(fake_local))
    comps = {"featurize": fakes.fake_featurize, "serialize": fakes.fake_serialize, "gbm": fakes.fake_gbm, "llm_client": None}
    cfg = {**cfg, "laya": {"mode": "cached", "decide": False, "cache_path": str(tmp_path / "none.json")},
           "laya_ask": {"enabled": True, "model_path": "artifacts/laya/not-imported-here", "device": "cuda"}}
    c = TestClient(build_app(cfg, components=comps))
    did = c.post("/score", json=booking_json()).json()["decision_id"]
    r = c.post(f"/decisions/{did}/ask", json=ASK)
    assert r.status_code == 200 and r.json()["answered_by"] == "stock Laya (not fine-tuned)"
    assert seen["path"] == "convaiinnovations/laya"
