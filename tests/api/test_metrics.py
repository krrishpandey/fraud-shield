import pytest

from fraudshield.api.metrics import dashboard_metrics
from fraudshield.policy.costs import CostConfig, cost_matrix

P = CostConfig.default()


def rec(i, action, label=None, date="2018-06-01", F=50.0, scenario=None, hard_negative=False, total=100.0,
        source="live", analyst=None):
    meta = {}
    if scenario:
        meta["scenario"] = scenario
    if hard_negative:
        meta["hard_negative"] = True
    return {"decision_id": f"d{i}", "booking_id": f"b{i}", "booked_at": f"{date}T10:00:00", "action": action,
            "booking": {"carrier_cost": F, "meta": meta}, "label": label, "analyst": analyst,
            "latency_ms": {"total": total}, "source": source}


def test_empty_store_reports_nothing_invented():
    m = dashboard_metrics([], P)
    assert m["totals"]["bookings"] == 0
    assert m["revenue_loss_prevented_brl"] == 0.0 and m["fpr_legit"] is None
    assert m["window"] == {"from": None, "to": None}
    assert m["latency"] == {"p50_ms": None, "p99_ms": None}


def test_counts_money_and_fpr_from_labels():
    rs = [
        rec(1, "hold", "fraud", scenario="T1"),
        rec(2, "allow", "fraud", date="2018-06-02"),
        rec(3, "review", "legit"),
        rec(4, "allow", "legit"),
        rec(5, "allow_scan_gated", "legit", hard_negative=True),
        rec(6, "block", None, analyst={"label": "fraud"}, scenario="T4"),
        rec(7, "allow"),
    ]
    m = dashboard_metrics(rs, P)
    C = cost_matrix(50.0, P.params)
    leak = C["allow"][1]
    assert m["totals"]["bookings"] == 7
    assert m["totals"]["by_action"]["allow"] == 3
    assert m["held_shipments"] == 2
    assert m["revenue_loss_prevented_brl"] == pytest.approx((leak - C["hold"][1]) + (leak - C["block"][1]))
    assert m["friction_cost_brl"] == pytest.approx(C["review"][0] + C["allow_scan_gated"][0])
    assert m["net_prevented_brl"] == pytest.approx(m["revenue_loss_prevented_brl"] - m["friction_cost_brl"])
    assert m["fpr_legit"] == pytest.approx(1 / 3)  # review is friction; scan-gated is not
    assert m["fpr_hard_negative"] == pytest.approx(0.0)
    assert m["labels"]["labelled"] == 6
    day1 = [t for t in m["trend"] if t["date"] == "2018-06-01"][0]
    assert day1["fraud_stopped"] == 2
    assert day1["by_typology"] == {**{f"T{i}": 0 for i in range(1, 8)}, "T1": 1, "T4": 1, "unlabelled": 0}
    assert m["window"] == {"from": "2018-06-01", "to": "2018-06-02"}
    assert m["latency"]["p50_ms"] == pytest.approx(100.0)
    assert "assumption" in m["assumptions"]["note"]


def test_unlabelled_non_allow_counted_and_typology_normalised():
    rs = [rec(1, "hold"), rec(2, "review"), rec(3, "allow"), rec(4, "block", "fraud", scenario="T3_0042")]
    t = dashboard_metrics(rs, P)["trend"][0]["by_typology"]
    assert t["unlabelled"] == 2 and t["T3"] == 1
