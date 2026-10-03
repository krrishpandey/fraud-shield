"""Stream routes: a separate decision-service instance scores replayed bookings; the main service is untouched."""
import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from fraudshield.api.app import build_app
from fraudshield.sim.stream import StreamItem
from tests.api import fakes
from tests.api.test_app import components
from tests.policy.fixtures import make_booking


def source(n=12):
    def src(seed):
        if seed != 0:
            raise ValueError("only injection seed 0 can be streamed")
        t0 = datetime(2018, 6, 1, 8)
        return [StreamItem(make_booking(booking_id=f"st{i:03d}", account_id=f"acc{i}",
                                        booked_at=(t0 + timedelta(hours=3 * i)).isoformat()),
                           is_fraud=i % 4 == 0, typology="T1" if i % 4 == 0 else "none", offline_score=0.71)
                for i in range(n)]
    return src


@pytest.fixture
def scfg(tmp_path):
    return {"audit_path": str(tmp_path / "audit.jsonl"), "replay_path": str(tmp_path / "none.jsonl"),
            "demo_bookings_path": str(tmp_path / "demo.json"), "web_dist": str(tmp_path / "nodist"),
            "calibration_path": str(tmp_path / "nocal.json"),
            "stream": {"audit_path": str(tmp_path / "stream_audit.jsonl"), "explain_llm_per_min": 1}}


def wait_done(c, timeout=15):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = c.get("/stream/status").json()
        if s["state"] in ("done", "stopped"):
            return s
        time.sleep(0.05)
    raise AssertionError("stream did not finish")


def test_stream_scores_every_booking_in_its_own_service(scfg, tmp_path):
    c = TestClient(build_app(scfg, components=components(stream_source=source())))
    assert c.get("/stream/status").json()["state"] == "idle"
    r = c.post("/stream/start", json={"rate": 200, "concurrency": 2, "seed": 0})
    assert r.status_code == 200
    assert wait_done(c)["sent"] == 12
    m = c.get("/stream/metrics").json()
    assert m["load"]["scored"] == 12 and m["load"]["errors"] == 0
    assert sum(m["actions"].values()) == 12
    assert m["offline_full"]["n"] == 12 and m["accuracy"]["n_fraud"] == 3
    feed = c.get("/stream/feed?limit=5").json()
    assert len(feed) == 5
    # a streamed decision opens on the normal decision page, but stays out of the main queue and dashboard
    d = c.get(f"/decisions/{feed[0]['decision_id']}")
    assert d.status_code == 200 and d.json()["booking"]["booking_id"].startswith("st")
    assert c.get("/decisions").json() == []
    assert (tmp_path / "stream_audit.jsonl").exists()
    assert not (tmp_path / "audit.jsonl").exists() or "st0" not in (tmp_path / "audit.jsonl").read_text()


def test_streamed_decision_accepts_analyst_label(scfg):
    c = TestClient(build_app(scfg, components=components(stream_source=source(4))))
    c.post("/stream/start", json={"rate": 200, "concurrency": 1})
    wait_done(c)
    did = c.get("/stream/feed?limit=1").json()[0]["decision_id"]
    assert c.post(f"/decisions/{did}/analyst", json={"label": "fraud", "note": "stream"}).status_code == 200


def test_unknown_seed_and_double_start_are_refused(scfg):
    c = TestClient(build_app(scfg, components=components(stream_source=source(400))))
    r = c.post("/stream/start", json={"rate": 5, "seed": 1})
    assert r.status_code == 400 and "seed 0" in r.json()["detail"]
    assert c.post("/stream/start", json={"rate": 5}).status_code == 200
    assert c.post("/stream/start", json={"rate": 5}).status_code == 409
    assert c.post("/stream/stop").json()["state"] in ("stopped", "running")
    wait_done(c)


def test_without_a_source_the_stream_is_unavailable(scfg):
    c = TestClient(build_app(scfg, components=components()))
    assert c.post("/stream/start", json={"rate": 5}).status_code == 503


def test_llm_explanations_are_capped_and_the_rest_use_the_template(scfg):
    c = TestClient(build_app(scfg, components=components(stream_source=source(6),
                                                         llm_client=fakes.llm_client("Hold this booking."))))
    c.post("/stream/start", json={"rate": 200, "concurrency": 1})
    wait_done(c)
    time.sleep(0.5)
    ex = c.get("/stream/metrics").json()["explanations"]
    assert ex["llm_cap_per_min"] == 1 and ex["llm"] <= 1 and ex["llm"] + ex["template"] == 6



def test_streamed_decision_ids_never_collide_with_main_decisions(scfg):
    from tests.policy.fixtures import booking_json
    c = TestClient(build_app(scfg, components=components(stream_source=source(4))))
    main_id = c.post("/score", json=booking_json()).json()["decision_id"]
    c.post("/stream/start", json={"rate": 200, "concurrency": 1})
    wait_done(c)
    feed = c.get("/stream/feed?limit=10").json()
    assert all(r["decision_id"].startswith("str_") for r in feed)
    assert main_id not in {r["decision_id"] for r in feed}
    for r in feed:  # every streamed id opens its own booking, never a main decision
        assert c.get(f"/decisions/{r['decision_id']}").json()["booking"]["booking_id"] == r["booking_id"]
    assert c.get(f"/decisions/{main_id}").json()["booking"]["booking_id"] == "demo-takeover-01"


def test_flagged_lists_every_held_or_blocked_booking_newest_first_with_review_status(scfg):
    c = TestClient(build_app(scfg, components=components(stream_source=source(12))))
    c.post("/stream/start", json={"rate": 200, "concurrency": 1})
    wait_done(c)
    allrows = c.get("/stream/feed?limit=200").json()
    stopped = [r for r in allrows if r["action"] in ("hold", "block")]
    f = c.get("/stream/flagged").json()
    assert f["counts"]["hold"] + f["counts"]["block"] == len(stopped) == len(f["rows"]) > 0
    assert all(r["action"] in ("hold", "block") for r in f["rows"])
    assert [r["decision_id"] for r in f["rows"]] == sorted((r["decision_id"] for r in f["rows"]), reverse=True)
    first = f["rows"][0]
    assert first["reasons"] and "route" in first and first["reviewed"] is None
    c.post(f"/decisions/{first['decision_id']}/analyst", json={"label": "fraud", "note": "checked"})
    again = c.get("/stream/flagged").json()
    assert again["rows"][0]["reviewed"] == "fraud" and again["counts"]["unreviewed"] == len(stopped) - 1
    only_block = c.get("/stream/flagged?actions=block").json()
    assert all(r["action"] == "block" for r in only_block["rows"])


def test_flagged_before_any_stream_is_empty(scfg):
    c = TestClient(build_app(scfg, components=components(stream_source=source())))
    assert c.get("/stream/flagged").json() == {"rows": [], "counts": {"hold": 0, "block": 0, "unreviewed": 0}}
