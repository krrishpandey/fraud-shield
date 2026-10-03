from fraudshield.policy.reasons import REASON_RULES, booking_signals, reason_codes
from tests.policy.fixtures import make_booking


def test_codes_come_from_fixed_list_ranked_by_severity():
    vals = {"new_senders_l10": 8, "cost_vs_median": 11.6, "login_device_age_days": 0.0}
    codes, top = reason_codes(vals)
    assert set(codes) <= {r.code for r in REASON_RULES}
    assert codes[0] == "COST_FAR_ABOVE_ACCOUNT_NORM"  # 11.6/3 is the largest exceedance
    assert set(codes) == {"NEW_SENDERS_UNDER_PAYER", "COST_FAR_ABOVE_ACCOUNT_NORM", "NEW_LOGIN_DEVICE"}
    names = [t["name"] for t in top]
    assert "cost_vs_median" in names
    assert all({"name", "value", "baseline", "code"} <= set(t) for t in top)


def test_normal_booking_has_no_reasons():
    codes, top = reason_codes({"new_senders_l10": 0, "cost_vs_median": 1.1, "login_device_age_days": 300})
    assert codes == [] and top == []


def test_missing_and_non_numeric_features_are_ignored():
    codes, _ = reason_codes({"cost_vs_median": "n/a", "other": 3})
    assert codes == []


def test_max_n_limits_output():
    vals = {"new_senders_l10": 8, "cost_vs_median": 11.6, "login_device_age_days": 0.0,
            "new_origins_l10": 7, "payer_senders_30d": 9, "weight_z": 5.1}
    codes, _ = reason_codes(vals, max_n=3)
    assert len(codes) == 3


def test_booking_signals_never_use_meta():
    b = make_booking(meta={"scenario": "T1", "typology": "T1", "login_device_age_days": 999})
    s = booking_signals(b)
    assert s["login_device_age_days"] == b.login_device_age_days
    assert s["sender_differs"] == 1
    assert "scenario" not in s and "typology" not in s


def test_rules_use_feature_spec_names_and_per_booking_baseline():
    from fraudshield.features.spec import FEATURES
    assert {r.feature for r in REASON_RULES} - {"sender_differs"} <= set(FEATURES)
    _, top = reason_codes({"new_senders_l10": 6, "base_senders_l10": 0.4})
    assert top[0]["baseline"] == 0.4
