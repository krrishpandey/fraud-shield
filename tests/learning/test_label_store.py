import json

from fraudshield.learning.label_store import LabelStore, label_from_decision


def _rec(**kw):
    rec = {"decision_id": "dec_000001", "booking_id": "bk1", "account_id": "acc1",
           "booked_at": "2018-06-01T10:00:00", "action": "allow_scan_gated", "propensity": 0.05,
           "explored": True, "feature_values": {"cost_vs_median": 2.0, "tenure_days": 400.0},
           "model_versions": {"gbm": "gbm-B2-F-v1"}, "raw_probabilities": {"misuse": 0.3},
           "state_text": "BOOKING x", "booking": {"meta": {"typology": "T1"}}}
    rec.update(kw)
    return rec


def test_label_snapshot_is_a_copy_taken_at_decision_time(tmp_path):
    rec = _rec()
    lab = label_from_decision(rec, "fraud", "analyst", labelled_at="2018-06-03T09:00:00")
    rec["feature_values"]["cost_vs_median"] = 99.0  # later mutation must not leak into the label
    assert lab["features"]["cost_vs_median"] == 2.0
    assert lab["label"] == "fraud" and lab["source"] == "analyst" and lab["simulated"] is False
    assert lab["booked_at"] == "2018-06-01T10:00:00" and lab["labelled_at"] == "2018-06-03T09:00:00"
    assert lab["propensity"] == 0.05 and lab["explored"] is True and lab["action"] == "allow_scan_gated"
    assert lab["model_version"] == "gbm-B2-F-v1" and lab["typology"] == "T1"
    assert lab["raw_laya_misuse"] == 0.3


def test_store_is_append_only_jsonl_and_reloads(tmp_path):
    p = tmp_path / "fb" / "labels.jsonl"
    s = LabelStore(p)
    s.append(label_from_decision(_rec(), "fraud", "analyst"))
    s.append(label_from_decision(_rec(), "legit", "analyst"))  # relabel: appended, not overwritten
    s.append(label_from_decision(_rec(decision_id="dec_2", booking_id="bk2"), "legit", "simulated_analyst"))
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["label"] == "fraud"
    s2 = LabelStore(p)
    assert len(s2.all()) == 3
    latest = s2.latest_by_decision()
    assert len(latest) == 2 and latest[0]["label"] == "legit"
    assert s2.counts_by_source() == {"analyst": 2, "simulated_analyst": 1}
    assert [x["simulated"] for x in s2.all()] == [False, False, True]


def test_rejects_bad_label_or_source(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        label_from_decision(_rec(), "maybe", "analyst")
    with pytest.raises(ValueError):
        label_from_decision(_rec(), "fraud", "guess")


def test_missing_snapshot_is_recorded_as_none_never_recomputed():
    rec = _rec()
    del rec["feature_values"]
    assert label_from_decision(rec, "legit", "analyst")["features"] is None
