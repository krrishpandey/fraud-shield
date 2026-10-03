"""Pipeline hooks used by continuous learning (fraudshield/learning)."""
import json

from fraudshield.audit.log import verify_file
from fraudshield.contracts import FeatureVector
from fraudshield.learning.label_store import LabelStore
from tests.api.test_pipeline import make_pipeline
from tests.policy.fixtures import make_booking


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_decision_keeps_as_of_feature_snapshot(tmp_path):
    p = make_pipeline(tmp_path)
    did = p.score(make_booking())["decision_id"]
    snap = p.get(did)["feature_values"]
    assert snap == {"cost_vs_median": 11.6, "new_senders_l10": 8, "tenure_days": 412.0}
    assert "feature_values" not in p.score(make_booking())  # not part of the public score response


def test_score_accepts_precomputed_feature_vector(tmp_path):
    from fraudshield.api.pipeline import Pipeline
    from fraudshield.audit.log import AuditLog
    from tests.api import fakes

    def boom(b):
        raise AssertionError("must not featurize")
    p = Pipeline(boom, fakes.fake_serialize, fakes.fake_gbm, fakes.laya(), AuditLog(tmp_path / "audit.jsonl"))
    fv = FeatureVector("demo-takeover-01", "2018-06-14T02:41:00", {"cost_vs_median": 1.0, "tenure_days": 900.0})
    did = p.score(make_booking(), fv=fv)["decision_id"]
    assert p.get(did)["feature_values"]["tenure_days"] == 900.0


def test_analyst_feedback_writes_label_store_and_audit(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    p = make_pipeline(tmp_path, label_store=store)
    did = p.score(make_booking())["decision_id"]
    p.analyst_feedback(did, "fraud", "owner denied")
    p.analyst_feedback(did, "legit", "sim", source="simulated_analyst")
    labs = store.all()
    assert [x["source"] for x in labs] == ["analyst", "simulated_analyst"]
    assert labs[0]["features"]["cost_vs_median"] == 11.6 and labs[0]["typology"] == "T1"
    ev = _events(tmp_path)
    assert [e["event_type"] for e in ev].count("analyst_feedback") == 2
    assert ev[-1]["payload"]["source"] == "simulated_analyst" and ev[-1]["payload"]["simulated"] is True


def test_ask_is_audited(tmp_path):
    p = make_pipeline(tmp_path)
    did = p.score(make_booking())["decision_id"]
    out = p.ask(did, "Is the consignee a drop?", "drop", "ordinary")
    ev = _events(tmp_path)[-1]
    assert ev["event_type"] == "ask"
    pl = ev["payload"]
    assert pl["decision_id"] == did and pl["instructions"] == "Is the consignee a drop?"
    assert pl["criteria"] == {"a": "drop", "b": "ordinary"}
    assert pl["probability_yes"] == out["probability_yes"] and pl["raw_probabilities"] == out["raw_probabilities"]
    assert pl["model_version"] == "fake-laya" and pl["qid"] == out["qid"]
    assert out["audit_hash"] == ev["record_hash"]
    assert verify_file(tmp_path / "audit.jsonl")["ok"] is True


def test_swap_gbm_is_atomic_and_versioned(tmp_path):
    p = make_pipeline(tmp_path)
    old = p.swap_gbm(lambda fv: 0.05, "gbm-B2-F-v2")
    assert old[1] == "unknown" and callable(old[0])
    r = p.score(make_booking(booking_id="after-swap"))
    assert r["gbm_score"] == 0.05 and r["model_versions"]["gbm"] == "gbm-B2-F-v2"


def test_feedback_accepts_simulated_label_time_and_outcome_source(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    p = make_pipeline(tmp_path, label_store=store)
    did = p.score(make_booking())["decision_id"]
    p.analyst_feedback(did, "fraud", "dispute", source="simulated_outcome", at="2018-07-20T10:00:00")
    lab = store.all()[-1]
    assert lab["labelled_at"] == "2018-07-20T10:00:00" and lab["simulated"] is True
    ev = _events(tmp_path)[-1]["payload"]
    assert ev["at"] == "2018-07-20T10:00:00" and ev["simulated"] is True and ev["source"] == "simulated_outcome"
