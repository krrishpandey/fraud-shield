from fraudshield.policy.queue import QueueItem, build_queue, value_prevented


def item(i, p, F, t=6.0):
    return QueueItem(decision_id=f"d{i}", p_fraud=p, carrier_cost=F, review_min=t)


def test_value_prevented_grows_with_payoff_and_probability():
    assert value_prevented(item(0, 0.5, 200)) > value_prevented(item(0, 0.5, 20))
    assert value_prevented(item(0, 0.9, 50)) > value_prevented(item(0, 0.2, 50))


def test_greedy_by_value_per_minute_under_capacity():
    items = [item(0, 0.5, 200, t=30), item(1, 0.5, 100, t=5), item(2, 0.5, 100, t=5), item(3, 0.01, 10, t=1)]
    q = build_queue(items, capacity_min=12)
    picked = [x["decision_id"] for x in q["selected"]]
    assert picked == ["d1", "d2"]  # high value per minute first; d0 does not fit
    assert sum(x["review_min"] for x in q["selected"]) <= 12
    assert {"d0", "d3"} == {x["decision_id"] for x in q["deferred"]}


def test_negative_value_items_never_selected():
    q = build_queue([item(0, 0.001, 5)], capacity_min=100)
    assert q["selected"] == []


def test_output_has_ratio_and_shadow_price():
    q = build_queue([item(0, 0.6, 100), item(1, 0.6, 50)], capacity_min=6)
    assert q["selected"][0]["value_per_min"] > 0
    assert q["shadow_price_per_min"] >= 0
