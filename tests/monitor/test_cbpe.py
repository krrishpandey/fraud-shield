"""Label-free estimate (CBPE): known answers on small arrays, unbiased on calibrated synthetic scores, blind to
fraud scored low, and the PSI drift signal."""
import math

import numpy as np
import pytest

from fraudshield.monitor import cbpe


def test_estimate_known_answer():
    p = [0.9, 0.7, 0.1, 0.2, 0.0]
    s = [True, True, False, False, False]
    e = cbpe.estimate(p, s)
    assert e["n"] == 5 and e["n_stopped"] == 2 and e["n_let_through"] == 3
    assert e["precision_stopped"] == pytest.approx(0.8)
    assert e["fraud_caught"] == pytest.approx(1.6)
    assert e["fraud_missed"] == pytest.approx(0.3)
    assert e["recall"] == pytest.approx(1.6 / 1.9)


def test_realized_known_answer():
    r = cbpe.realized([1, 0, 1, 0, 0], [True, True, False, False, False])
    assert r["precision_stopped"] == pytest.approx(0.5)
    assert (r["fraud_caught"], r["fraud_missed"]) == (1, 1)
    assert r["recall"] == pytest.approx(0.5)


def test_no_stops_and_no_fraud_are_nan_not_errors():
    e = cbpe.estimate([0.0, 0.0], [False, False])
    assert math.isnan(e["precision_stopped"]) and math.isnan(e["recall"])
    r = cbpe.realized([0, 0], [False, False])
    assert math.isnan(r["precision_stopped"]) and math.isnan(r["recall"])
    assert cbpe.estimate([], [])["n"] == 0


def test_stopped_mask_matches_the_stream_definition():
    acts = ["allow", "allow_scan_gated", "owner_confirm", "review", "hold", "block"]
    assert cbpe.stopped_mask(acts).tolist() == [False, False, True, True, True, True]


def test_calibrated_scores_give_an_unbiased_estimate():
    rng = np.random.default_rng(0)
    p = rng.beta(0.3, 8.0, 200_000)
    y = rng.random(len(p)) < p           # calibrated by construction
    s = p > 0.3
    e, r = cbpe.estimate(p, s), cbpe.realized(y, s)
    assert e["precision_stopped"] == pytest.approx(r["precision_stopped"], abs=0.01)
    assert e["fraud_missed"] == pytest.approx(r["fraud_missed"], rel=0.03)
    assert e["recall"] == pytest.approx(r["recall"], abs=0.01)


def test_fraud_the_model_scores_low_is_invisible_to_the_estimate():
    rng = np.random.default_rng(1)
    p = rng.beta(0.3, 8.0, 100_000)
    y = rng.random(len(p)) < p
    s = p > 0.3
    base = cbpe.estimate(p, s)["fraud_missed"] - cbpe.realized(y, s)["fraud_missed"]
    # a new fraud type: 300 frauds the model gives p = 0.001, all let through
    p2, y2, s2 = np.r_[p, np.full(300, 0.001)], np.r_[y, np.ones(300, bool)], np.r_[s, np.zeros(300, bool)]
    gap = cbpe.estimate(p2, s2)["fraud_missed"] - cbpe.realized(y2, s2)["fraud_missed"]
    assert gap - base == pytest.approx(-300 + 0.3)


def test_period_ids_and_per_period():
    ts = np.array(["2018-05-15T00:10", "2018-05-21T23:59", "2018-05-22T00:00", "2018-06-05T12:00"], dtype="datetime64[ns]")
    ids = cbpe.period_ids(ts, "2018-05-15", 7)
    assert ids.tolist() == [0, 0, 1, 3]
    rows = cbpe.per_period([0.5, 0.1, 0.9, 0.2], [True, False, True, False], ids, y=[1, 0, 1, 1])
    assert [r["period"] for r in rows] == [0, 1, 3]
    assert rows[0]["estimated"]["precision_stopped"] == pytest.approx(0.5)
    assert rows[2]["realized"]["fraud_missed"] == 1 and rows[2]["estimated"]["fraud_missed"] == pytest.approx(0.2)


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(2)
    ref = rng.beta(0.5, 20.0, 50_000)
    assert cbpe.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert cbpe.psi(ref, rng.beta(0.5, 20.0, 50_000)) < 0.01
    assert cbpe.psi(ref, ref * 3) > 0.25
    e = cbpe.psi_edges(ref)
    assert len(e) == 9 and np.all(np.diff(e) > 0)
    assert cbpe.shares(ref, e).sum() == pytest.approx(1.0)


def test_psi_edges_drop_ties():
    assert len(cbpe.psi_edges(np.r_[np.zeros(900), np.linspace(0.1, 1, 100)])) < 9
