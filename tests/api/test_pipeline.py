import json

import pytest

from fraudshield.api.pipeline import Pipeline
from fraudshield.audit.log import AuditLog
from fraudshield.policy.decide import PolicyConfig
from tests.api import fakes
from tests.policy.fixtures import make_booking


def make_pipeline(tmp_path, laya=None, **kw):
    return Pipeline(featurize=fakes.fake_featurize, serialize=fakes.fake_serialize, gbm=fakes.fake_gbm,
                    laya=laya or fakes.laya(), audit=AuditLog(tmp_path / "audit.jsonl"), **kw)


def test_score_returns_api_shape_and_audits(tmp_path):
    p = make_pipeline(tmp_path)
    r = p.score(make_booking())
    for k in ["decision_id", "booking_id", "action", "greedy_action", "propensity", "explored", "degraded",
              "probabilities", "raw_probabilities", "gbm_score", "expected_costs", "reasons", "top_features",
              "state_text", "model_versions", "latency_ms", "explanation_status", "audit_hash"]:
        assert k in r, k
    assert r["raw_probabilities"]["misuse"] == pytest.approx(0.97)
    assert r["probabilities"]["misuse"] == pytest.approx(0.97)  # no calibration file -> identity
    assert r["calibrated"] is False
    assert r["action"] in {"hold", "block"}
    assert "COST_FAR_ABOVE_ACCOUNT_NORM" in r["reasons"] and "NEW_LOGIN_DEVICE" in r["reasons"]
    assert set(r["latency_ms"]) >= {"total", "features", "gbm", "serialize", "laya", "decide", "audit"}
    assert r["explanation_status"] == "pending"
    rec = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert rec["event_type"] == "decision" and rec["record_hash"] == r["audit_hash"]
    assert rec["payload"]["state_text"] == r["state_text"]
    assert rec["payload"]["policy_trace"]["allowed"]


def test_idempotent_on_booking_id(tmp_path):
    p = make_pipeline(tmp_path)
    a = p.score(make_booking())
    b = p.score(make_booking())
    assert a == b
    assert p.audit.verify()["records"] == 1


def test_calibration_applied_and_versioned(tmp_path):
    cal = {"version": "cal-x", "temperatures": {"misuse": 3.0}, "platt": {},
           "conformal": {"lambda_allow": 0.03, "alpha": 0.05, "n_fraud": 100}}
    p = make_pipeline(tmp_path, calibration=cal)
    r = p.score(make_booking())
    assert r["probabilities"]["misuse"] < 0.97
    assert r["model_versions"]["calibration"] == "cal-x" and r["calibrated"] is True


def test_degraded_when_laya_down_uses_gbm_and_never_plain_allow(tmp_path):
    p = make_pipeline(tmp_path, laya=fakes.DownLaya())
    r = p.score(make_booking())
    assert r["degraded"] is True
    assert r["probabilities"] == {"misuse": pytest.approx(0.71)}
    assert r["action"] not in {"allow", "allow_scan_gated", "owner_confirm"}


def test_block_cap_per_account(tmp_path):
    hi = {"misuse": 0.999, "foreign_senders": 0.99, "payoff_max": 0.99, "drop_consignee": 0.9}
    p = make_pipeline(tmp_path, laya=fakes.laya(hi))
    r1 = p.score(make_booking(booking_id="b1", carrier_cost=400.0))
    r2 = p.score(make_booking(booking_id="b2", carrier_cost=400.0, booked_at="2018-06-14T03:00:00"))
    assert r1["action"] == "block"
    assert r2["action"] == "hold"


def test_meta_never_reaches_state_or_features(tmp_path):
    p = make_pipeline(tmp_path)
    r = p.score(make_booking(meta={"scenario": "SECRET_T9"}))
    assert "SECRET_T9" not in r["state_text"]
    assert "SECRET_T9" not in json.dumps(r["top_features"])


def test_explanation_completes_and_is_audited(tmp_path):
    p = make_pipeline(tmp_path, llm_client=None)
    r = p.score(make_booking())
    p.explain_decision(r["decision_id"])
    d = p.get(r["decision_id"])
    assert d["explanation_status"] == "ready"
    assert d["explanation"]["source"] == "template"
    assert p.audit.verify()["records"] == 2


def test_audit_failure_means_no_decision(tmp_path):
    class BadAudit:
        def append(self, *a):
            raise OSError("disk full")
    p = Pipeline(featurize=fakes.fake_featurize, serialize=fakes.fake_serialize, gbm=fakes.fake_gbm,
                 laya=fakes.laya(), audit=BadAudit())
    with pytest.raises(OSError):
        p.score(make_booking())
    assert p.list() == []


def test_deterministic_with_exploration_seeded(tmp_path):
    cfg = PolicyConfig(explore_eps=0.5)
    a = make_pipeline(tmp_path / "a", policy=cfg).score(make_booking(booking_id="x1"))
    b = make_pipeline(tmp_path / "b", policy=cfg).score(make_booking(booking_id="x1"))
    assert (a["action"], a["propensity"]) == (b["action"], b["propensity"])


def _featurize_with(**extra):
    from fraudshield.contracts import FeatureVector

    def f(booking):
        base = fakes.fake_featurize(booking).values
        return FeatureVector(booking.booking_id, booking.booked_at, {**base, "tenure_days": 400.0, **extra})
    return f


def _low_gbm(fv):
    return 0.002  # an ordinary-looking booking for the model


def test_drop_address_pattern_raises_risk_and_is_a_named_reason(tmp_path):
    p = make_pipeline(tmp_path, laya=fakes.DownLaya())  # backup model decides
    p.featurize, p.gbm = _featurize_with(drop_pattern=1, under_declared=0), _low_gbm
    d = p.score(make_booking(booking_id="drop-1"))
    assert d["probabilities"]["misuse"] == pytest.approx(1 - (1 - 0.002) * 0.5, abs=1e-4)
    assert d["gbm_score"] == pytest.approx(0.002, abs=1e-4)  # the model score itself is unchanged
    assert "DROP_ADDRESS_PATTERN" in d["reasons"] and d["action"] != "allow"


def test_under_declared_parcel_gets_at_least_a_first_scan_check(tmp_path):
    p = make_pipeline(tmp_path, laya=fakes.DownLaya())
    p.featurize, p.gbm = _featurize_with(drop_pattern=0, under_declared=1), _low_gbm
    d = p.score(make_booking(booking_id="small-1"))
    assert d["action"] != "allow" and "UNDER_DECLARED_PARCEL" in d["reasons"]
    plain = make_pipeline(tmp_path / "b", laya=fakes.DownLaya())
    plain.featurize, plain.gbm = _featurize_with(drop_pattern=0, under_declared=0), _low_gbm
    assert plain.score(make_booking(booking_id="plain-1"))["action"] == "allow"


def test_failed_depot_scan_sends_the_accounts_later_parcels_to_a_scan_check(tmp_path):
    p = make_pipeline(tmp_path, laya=fakes.DownLaya())
    p.featurize, p.gbm = _featurize_with(drop_pattern=0, under_declared=1), _low_gbm
    first = p.score(make_booking(booking_id="s1", account_id="accZ", weight_kg=3.0))
    assert first["action"] == "allow_scan_gated"
    ok = p.record_first_scan(first["decision_id"], measured_weight_kg=3.05)  # within 0.5 lb / 3%
    assert ok["mismatch"] is False and "accZ" not in p.scan_failed_accounts
    bad = p.record_first_scan(first["decision_id"], measured_weight_kg=7.0)
    assert bad["mismatch"] is True and bad["audit_hash"] and "accZ" in p.scan_failed_accounts
    p.featurize = _featurize_with(drop_pattern=0, under_declared=0)  # the next parcel looks ordinary
    nxt = p.score(make_booking(booking_id="s2", account_id="accZ", booked_at="2018-06-15T10:00:00"))
    assert nxt["action"] != "allow" and "ACCOUNT_FAILED_DEPOT_SCAN" in nxt["reasons"]
    other = p.score(make_booking(booking_id="s3", account_id="accY", booked_at="2018-06-15T10:00:00"))
    assert other["action"] == "allow"


def test_carrier_weight_tolerance():
    from fraudshield.api.pipeline import scan_mismatch
    assert scan_mismatch(10.0, 10.29) is False   # within 3%
    assert scan_mismatch(10.0, 10.31) is True
    assert scan_mismatch(1.0, 1.2) is False      # within 0.5 lb (0.227 kg)
    assert scan_mismatch(1.0, 1.3) is True
