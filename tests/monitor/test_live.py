"""Live monitor payload and GET /monitor/estimate on the stream."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudshield.api.app import ROOT, build_app
from fraudshield.monitor import cbpe
from fraudshield.monitor.live import PSI_MIN_N, REALIZED_LABEL, live_estimate, load_results
from tests.api.test_app import components
from tests.api.test_stream_routes import scfg, source, wait_done  # noqa: F401  (scfg is a fixture)

RESULTS = {"blind_spot": {"ui_note": "Blind spot: test note", "missed_gap_per_period": -1.0, "period": "week",
                          "seeds": 10, "split": "test"},
           "live_reference": {"window": "cal", "edges": [0.1, 0.5], "shares": [0.8, 0.15, 0.05]}}


def row(score, action, fraud=False, typ="none"):
    return {"score": score, "action": action, "is_fraud": fraud, "typology": typ}


def test_live_estimate_known_answer_and_labels():
    rows = [row(0.9, "hold", True, "T1"), row(0.5, "review"), row(0.05, "allow", True, "T3"),
            row(0.02, "allow_scan_gated", True, "T6"), row(0.01, "allow")]
    out = live_estimate(rows, RESULTS)
    e, r = out["estimated"], out["realized"]
    assert out["n"] == 5
    assert e["precision_stopped"] == pytest.approx(0.7) and e["fraud_missed"] == pytest.approx(0.08)
    assert r["precision_stopped"] == pytest.approx(0.5) and r["fraud_missed"] == 2
    assert r["label"] == REALIZED_LABEL and "not available in production" in r["label"]
    assert r["missed_by_type"] == {"T3": 1, "T6": 1} and r["missed_held_out"] == 1
    assert out["drift"]["psi"] is None  # fewer than PSI_MIN_N bookings
    assert out["blind_spot"]["ui_note"] == "Blind spot: test note"


def test_live_estimate_psi_after_enough_bookings_and_without_results():
    rows = [row(0.05, "allow")] * PSI_MIN_N
    out = live_estimate(rows, RESULTS)
    assert out["drift"]["psi"] == pytest.approx(cbpe.psi_from_shares([0.8, 0.15, 0.05], [1.0, 0.0, 0.0]))
    bare = live_estimate([], None)
    assert bare["n"] == 0 and bare["blind_spot"] is None and bare["drift"]["psi"] is None


def test_shipped_results_file_has_what_the_live_card_needs():
    res = load_results(ROOT / "artifacts" / "results_monitor.json")
    if res is None:
        pytest.skip("artifacts/results_monitor.json not built (scripts/eval_monitor.py)")
    assert res["blind_spot"]["ui_note"].startswith("Blind spot")
    ref = res["live_reference"]
    assert len(ref["shares"]) == len(ref["edges"]) + 1 and sum(ref["shares"]) == pytest.approx(1.0)


def test_monitor_route_on_the_stream(scfg, tmp_path):  # noqa: F811
    (tmp_path / "mon.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    c = TestClient(build_app({**scfg, "monitor_results_path": str(tmp_path / "mon.json")},
                             components=components(stream_source=source())))
    idle = c.get("/monitor/estimate").json()
    assert idle["n"] == 0 and idle["estimated"]["precision_stopped"] is None  # NaN is sent as null
    c.post("/stream/start", json={"rate": 200, "concurrency": 2})
    wait_done(c)
    out = c.get("/monitor/estimate").json()
    assert out["n"] == 12 and out["source"] == "stream"
    rows = c.get("/stream/metrics").json()
    st = sum(rows["actions"][a] for a in cbpe.STOPPED)
    assert out["estimated"]["n_stopped"] == out["realized"]["n_stopped"] == st
    assert out["realized"]["fraud_caught"] + out["realized"]["fraud_missed"] == 3
    assert out["realized"]["label"] == REALIZED_LABEL
    assert out["blind_spot"]["ui_note"] == "Blind spot: test note"
    assert c.get("/api/monitor/estimate").status_code == 200
    assert np.isfinite(out["estimated"]["fraud_missed"])
