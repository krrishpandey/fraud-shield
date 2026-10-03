import json
import math

import pytest

from fraudshield.models.calibration import apply_calibration, load_calibration, p_yes

RAW = {"misuse": {"a": 0.97, "b": 0.03}, "payoff_max": {"a": 0.6, "b": 0.4}, "drop_consignee": {"a": 0.12, "b": 0.88}}
CAL = {
    "version": "cal-20260101-abcd1234", "model_revision": "x",
    "temperatures": {"misuse": 2.0, "payoff_max": 1.0},
    "platt": {"misuse": {"a": 1.0, "b": -1.0}},
    "conformal": {"lambda_allow": 0.04, "alpha": 0.05, "n_fraud": 300}, "prevalence": 0.01,
}


def test_identity_without_calibration_file_is_flagged():
    cal, info = apply_calibration(RAW, None)
    assert cal == RAW
    assert info["calibrated"] is False and info["version"] == "uncalibrated"


def test_temperature_is_softmax_of_log_p_over_T():
    cal, info = apply_calibration(RAW, {**CAL, "platt": {}})
    pa = 0.97 ** 0.5 / (0.97 ** 0.5 + 0.03 ** 0.5)
    assert cal["misuse"]["a"] == pytest.approx(pa)
    assert sum(cal["misuse"].values()) == pytest.approx(1.0)
    assert cal["payoff_max"] == pytest.approx(RAW["payoff_max"])  # T=1 is identity
    assert info["calibrated"] is True and info["version"] == CAL["version"]


def test_question_without_temperature_left_raw():
    cal, _ = apply_calibration(RAW, CAL)
    assert cal["drop_consignee"] == RAW["drop_consignee"]


def test_platt_on_misuse_after_temperature():
    cal, _ = apply_calibration(RAW, CAL)
    pa_T = 0.97 ** 0.5 / (0.97 ** 0.5 + 0.03 ** 0.5)
    z = math.log(pa_T / (1 - pa_T)) - 1.0
    assert cal["misuse"]["a"] == pytest.approx(1 / (1 + math.exp(-z)))
    assert cal["misuse"]["b"] == pytest.approx(1 - cal["misuse"]["a"])


def test_extreme_probabilities_do_not_crash():
    cal, _ = apply_calibration({"misuse": {"a": 1.0, "b": 0.0}}, CAL)
    assert 0 < cal["misuse"]["a"] <= 1


def test_load_calibration(tmp_path):
    assert load_calibration(tmp_path / "missing.json") is None
    p = tmp_path / "c.json"
    p.write_text(json.dumps(CAL))
    assert load_calibration(p)["conformal"]["lambda_allow"] == 0.04


def test_p_yes():
    assert p_yes({"a": 0.3, "b": 0.7}) == 0.3
