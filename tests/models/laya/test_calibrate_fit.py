import json

import numpy as np
import pytest

from fraudshield.models import calibrate_fit as C
from fraudshield.models.calibration import apply_calibration


def _overconfident(n=4000, seed=0, T=2.5):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, n)                      # true logit
    y = rng.random(n) < 1 / (1 + np.exp(-z))
    p_raw = 1 / (1 + np.exp(-z * T))             # over-confident by factor T
    return p_raw, y.astype(int)


def test_fit_temperature_recovers_overconfidence():
    p, y = _overconfident(T=2.5)
    probs = np.stack([p, 1 - p], 1)              # options a (yes), b (no)
    target = np.stack([y, 1 - y], 1)
    T = C.fit_temperature(probs, target)
    assert T == pytest.approx(2.5, rel=0.15)


def test_fit_temperature_respects_weights():
    p, y = _overconfident(T=2.0)
    probs = np.stack([p, 1 - p], 1)
    target = np.stack([y, 1 - y], 1)
    t1 = C.fit_temperature(probs, target, weights=np.ones(len(y)))
    t2 = C.fit_temperature(probs, target, weights=np.full(len(y), 7.0))
    assert t1 == pytest.approx(t2, rel=1e-3)


def test_temper_matches_apply_calibration():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    out = C.temper(probs, 2.0)
    raw = {"misuse": {"a": 0.9, "b": 0.1}}
    cal, _ = apply_calibration(raw, {"temperatures": {"misuse": 2.0}})
    assert out[0, 0] == pytest.approx(cal["misuse"]["a"], abs=1e-9)


def test_fit_platt_corrects_base_rate_shift():
    rng = np.random.default_rng(1)
    n = 20000
    z = rng.normal(-1, 2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 3)))).astype(int)   # true rate much lower than p says
    p = 1 / (1 + np.exp(-z))
    a, b = C.fit_platt(p, y)
    assert a == pytest.approx(1.0, abs=0.1) and b == pytest.approx(-3.0, abs=0.2)
    pc = C.platt(p, a, b)
    assert abs(pc.mean() - y.mean()) < 0.01


def test_platt_matches_apply_calibration():
    cal, _ = apply_calibration({"misuse": {"a": 0.3, "b": 0.7}}, {"platt": {"misuse": {"a": 1.5, "b": -2.0}}})
    assert C.platt(np.array([0.3]), 1.5, -2.0)[0] == pytest.approx(cal["misuse"]["a"], abs=1e-6)


def test_conformal_lambda_bounds_allowed_fraud_rate():
    rng = np.random.default_rng(0)
    s = rng.random(199)
    lam = C.conformal_lambda(s, alpha=0.05)
    n = len(s)
    allowed = (s <= lam).sum()
    assert (n / (n + 1)) * allowed / n + 1 / (n + 1) <= 0.05 + 1e-12
    # and it is the largest such: one more allowed would break it
    nxt = np.sort(s)[allowed]
    assert (n / (n + 1)) * (allowed + 1) / n + 1 / (n + 1) > 0.05


def test_conformal_lambda_too_few_fraud_returns_zero():
    assert C.conformal_lambda(np.array([0.3, 0.6, 0.9]), alpha=0.05) == 0.0


def test_ece_equal_mass_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.random(30000)
    y = (rng.random(30000) < p).astype(int)
    assert C.ece(p, y) < 0.02
    assert C.ece(np.clip(p * 0 + 0.9, 0, 1), y) > 0.3


def test_ece_weighted_equals_duplicated():
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.5, 0.4])
    y = np.array([0, 1, 1, 1, 0, 0])
    w = np.array([1, 2, 1, 3, 1, 1])
    assert C.ece(p, y, w, bins=3) == pytest.approx(C.ece(np.repeat(p, w), np.repeat(y, w), bins=3), abs=1e-9)


def test_brier_weighted():
    assert C.brier(np.array([1.0, 0.0]), np.array([0, 0]), np.array([1.0, 3.0])) == pytest.approx(0.25)


def test_calibration_json_schema(tmp_path):
    w = tmp_path / "model.safetensors"
    w.write_bytes(b"weights")
    doc = C.calibration_json(w, {"misuse": 1.3, "foreign_senders": 1.1}, {"misuse": {"a": 1.0, "b": -2.0}},
                             lambda_allow=0.12, alpha=0.05, n_fraud=67, prevalence=0.0092, extra={"fit": "x"})
    assert doc["version"].startswith("cal-") and len(doc["version"].split("-")[-1]) == 8
    assert doc["model_revision"] == C.sha12(w)
    assert doc["conformal"] == {"lambda_allow": 0.12, "alpha": 0.05, "n_fraud": 67}
    assert set(doc) >= {"version", "model_revision", "temperatures", "platt", "conformal", "prevalence"}
    json.dumps(doc)
