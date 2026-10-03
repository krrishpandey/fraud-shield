import numpy as np

from fraudshield.learning.calibration_refresh import refresh_misuse_calibration


def _labels(n, seed=0, overconfident=3.0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = rng.uniform(size=n) < p
    z = np.log(p / (1 - p)) * overconfident  # raw Laya is overconfident
    raw = 1 / (1 + np.exp(-z))
    return [{"decision_id": f"d{i}", "booked_at": f"2018-06-{1 + i % 28:02d}T{i % 24:02d}:00:00",
             "labelled_at": "x", "label": "fraud" if y[i] else "legit", "raw_laya_misuse": float(raw[i])}
            for i in range(n)]


def test_skipped_when_too_few_rows():
    r = refresh_misuse_calibration(_labels(30) + [{"decision_id": "z", "label": "fraud", "raw_laya_misuse": None}],
                                   current=None)
    assert r["status"] == "skipped" and r["n_rows"] == 30 and "50" in r["reason"]
    assert r["calibration"] is None


def test_refit_improves_ece_and_keeps_other_fields():
    cur = {"version": "cal-old", "temperatures": {"misuse": 1.0, "payoff_max": 1.3},
           "platt": {"misuse": {"a": 1.0, "b": 0.0}}, "conformal": {"lambda_allow": 0.2}}
    r = refresh_misuse_calibration(_labels(600), current=cur)
    assert r["status"] == "updated", r
    assert r["ece_after"] < r["ece_before"]
    new = r["calibration"]
    assert new["version"].startswith("cal-") and new["version"] != "cal-old" and new["parent"] == "cal-old"
    assert new["temperatures"]["payoff_max"] == 1.3 and new["conformal"] == {"lambda_allow": 0.2}
    assert 0.2 < new["platt"]["misuse"]["a"] < 0.6  # undoes the 3x overconfidence


def test_rejected_when_not_better():
    cur = {"version": "cal-good", "platt": {"misuse": {"a": 1 / 3.0, "b": 0.0}}}
    r = refresh_misuse_calibration(_labels(600, overconfident=3.0), current=cur, min_improvement=0.01)
    assert r["status"] == "rejected" and r["calibration"] is None
