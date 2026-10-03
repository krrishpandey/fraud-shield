import json

from fraudshield.contracts import QUESTIONS
from fraudshield.learning.laya_export import LayaExport, laya_row


def _lab(label="fraud", state="BOOKING 2018-06-01 Fri 10:00 | channel api", **kw):
    d = {"decision_id": "dec_1", "booking_id": "bk1", "account_id": "acc1", "label": label,
         "source": "simulated_analyst", "simulated": True, "booked_at": "2018-06-01T10:00:00",
         "labelled_at": "2018-06-02T10:00:00", "state_text": state, "typology": "T1", "seq": 0}
    d.update(kw)
    return d


def test_row_follows_official_notebook_schema_with_misuse_gold_only():
    r = laya_row(_lab())
    assert r["workflow"] == "fraudshield_booking" and r["split"] == "feedback"
    assert json.loads(r["state"]) == "BOOKING 2018-06-01 Fri 10:00 | channel api"
    assert json.loads(r["questions"]) == {"misuse": QUESTIONS["misuse"]}
    assert json.loads(r["gold"]) == {"misuse": {"label": "a", "probabilities": {"a": 1.0, "b": 0.0}}}
    assert r["label_source"] == "simulated_analyst" and r["simulated"] is True
    legit = json.loads(laya_row(_lab("legit"))["gold"])
    assert legit == {"misuse": {"label": "b", "probabilities": {"a": 0.0, "b": 1.0}}}


def test_row_without_state_is_skipped():
    assert laya_row(_lab(state=None)) is None


def test_export_appends_and_counts(tmp_path):
    ex = LayaExport(tmp_path / "laya_feedback.jsonl")
    assert ex.add(_lab()) is True
    assert ex.add(_lab(state=None)) is False
    assert ex.rows() == 1 and ex.skipped == 1
    assert LayaExport(tmp_path / "laya_feedback.jsonl").rows() == 1
